# -*- coding: utf-8 -*-
"""
sniper_ticket_bakeoff.py — 「ここだ」のレース (◎オッズ ≤1.9 等) で、どの券種を買うのが最も負けないか (2026-11-12)
================================================================================
ユーザー「複勝じゃなくてもいい」。参戦条件を固定し、AI 上位馬 (◎〇▲△ = 生スコア順、2026 は bundle の p_win 順) で組む
1 点〜3 点の券種セットを 発見 2023-24 (A) / 確認 2025 (B) / 前向き 2026 (C) で比べる。1R 1 万円を点数で均等割り。
払戻は確定払戻 (単勝 = オッズ×100、複勝・馬連・馬単・三連複・三連単は kekka、ワイドは data/wide_payouts parquet と wide_kekka.csv)。
出力: reports/sniper_ticket_bakeoff.md (+ docs/research 控え)
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
import shadow_a95 as sa  # noqa: E402
from analysis.market_bias_atlas import load_ext  # noqa: E402

OUT = BASE / "reports" / "sniper_ticket_bakeoff.md"
SCORES = BASE / "data" / "_ev_grid_scores.parquet"
KEKKA_MASTER = BASE / "data" / "kekka_20130105-20251228.csv"
WIDE_PARQ = BASE / "data" / "wide_payouts_2016-2025.parquet"
BUDGET = 10000

# (名前, [(kind, idx tuple into AI order 0=◎ 1=〇 2=▲ 3=△)...])
TICKETS = {
    "単勝◎": [("tan", (0,))],
    "複勝◎": [("fuku", (0,))],
    "馬連◎-〇": [("umaren", (0, 1))],
    "馬単◎→〇": [("umatan", (0, 1))],
    "ワイド◎-〇": [("wide", (0, 1))],
    "三連複◎〇▲": [("sanpuku", (0, 1, 2))],
    "三連単◎→〇→▲": [("sanrentan", (0, 1, 2))],
    "馬連◎流し3点(〇▲△)": [("umaren", (0, 1)), ("umaren", (0, 2)), ("umaren", (0, 3))],
    "ワイド◎流し3点": [("wide", (0, 1)), ("wide", (0, 2)), ("wide", (0, 3))],
    "馬単◎→〇/▲ 2点": [("umatan", (0, 1)), ("umatan", (0, 2))],
    "三連複◎軸2頭流し3点": [("sanpuku", (0, 1, 2)), ("sanpuku", (0, 1, 3)), ("sanpuku", (0, 2, 3))],
    "単勝◎+複勝◎ 2点": [("tan", (0,)), ("fuku", (0,))],
    "複勝◎+ワイド◎-〇 2点": [("fuku", (0,)), ("wide", (0, 1))],
}


def _num(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def kekka_master() -> dict:
    """rid16 → shadow_a95.load_kekka と同形の dict (2013-2025)。"""
    k = pd.read_csv(KEKKA_MASTER, encoding="cp932", dtype=str)
    k["rid"] = k["レースID(新)"].str[:16]; k["ban"] = pd.to_numeric(k["馬番"], errors="coerce"); k["fin"] = pd.to_numeric(k["確定着順"], errors="coerce")
    w = pd.read_parquet(WIDE_PARQ)
    wide = {}
    for r in w.itertuples():
        wide[str(r.race_id)] = {f"{min(r.w1_i, r.w1_j)}-{max(r.w1_i, r.w1_j)}": float(r.w1_pay),
                                f"{min(r.w2_i, r.w2_j)}-{max(r.w2_i, r.w2_j)}": float(r.w2_pay),
                                f"{min(r.w3_i, r.w3_j)}-{max(r.w3_i, r.w3_j)}": float(r.w3_pay)}
    res = {}
    for rid, g in k.groupby("rid"):
        fin = {int(f): grp for f, grp in g.groupby("fin") if np.isfinite(f)}
        if any(len(fin.get(i, [])) != 1 for i in (1, 2, 3)):
            continue
        r1, r2, r3 = fin[1].iloc[0], fin[2].iloc[0], fin[3].iloc[0]
        t = str(r1["単勝配当"])
        res[rid] = dict(valid=True, first=int(r1.ban), second=int(r2.ban), third=int(r3.ban), w1=0, w2=0,
                        tan=_num(t) if t.isdigit() else np.nan,
                        fuku={int(r.ban): _num(r["複勝配当"]) for _, r in (("a", r1), ("b", r2), ("c", r3))},
                        wakuren=np.nan, umaren=_num(r2["馬連"]), umatan=_num(r2["馬単"]),
                        sanpuku=_num(r3["３連複"]), sanrentan=_num(r3["３連単"]), wide=wide.get(rid))
    return res


def offline_races() -> list[dict]:
    df = load_ext()
    df = df[df.year >= 2023]
    sc = pd.read_parquet(SCORES)
    sc["rid16"] = sc["レースID(新/馬番無)"].astype(str).str[:16]; sc["ban"] = pd.to_numeric(sc["馬番"], errors="coerce")
    df = df.merge(sc[["rid16", "ban", "_score"]], on=["rid16", "ban"], how="inner").sort_values(["rid16", "_score"], ascending=[True, False])
    out = []
    for rid, g in df.groupby("rid16", sort=False):
        if len(g) < 5:
            continue
        order = g.ban.astype(int).tolist()
        o1 = float(g.odds.iloc[0])
        out.append(dict(rid=rid, date=str(g.date.iloc[0]), period="A" if int(g.year.iloc[0]) <= 2024 else "B", order=order[:4], o1=o1))
    return out


def served_races() -> list[dict]:
    out = []
    for bp in sorted(glob.glob(str(BASE / "reports" / "cowork_input" / "2026????_bundle.json"))):
        d = Path(bp).name[:8]
        b = json.loads(Path(bp).read_text(encoding="utf-8"))
        for r in (b["races"] if isinstance(b["races"], list) else list(b["races"].values())):
            rid = re.sub(r"\D", "", str(r["race_id"]))[:16]
            hs = [h for h in r["horses"] if h.get("p_win") is not None]
            if len(hs) < 5:
                continue
            hs = sorted(hs, key=lambda h: (-float(h["p_win"]), int(h["umaban"])))
            out.append(dict(rid=rid, date=d, period="C", order=[int(h["umaban"]) for h in hs[:4]], o1=None))
    return out


def ci(vals_by_date: dict[str, tuple[float, float]], n_boot=1500, rng=np.random.default_rng(0)):
    g = np.array(list(vals_by_date.values()), float)
    if len(g) < 2:
        return np.nan, np.nan
    idx = rng.integers(0, len(g), (n_boot, len(g)))
    v = 100 * g[idx, 1].sum(1) / g[idx, 0].sum(1)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    print("[1/3] offline ...", flush=True)
    km = kekka_master(); off = offline_races()
    print(f"   {len(off):,} R", flush=True)
    print("[2/3] 2026 ...", flush=True)
    wide26 = sa.load_wide(); srv = served_races()
    kek26 = {}
    for d in sorted({r["date"] for r in srv}):
        kek26.update(sa.load_kekka(d, wide26))
    # 2026 の ◎確定オッズ (T-10 代理) は kekka の単勝配当/paren から
    o26 = {}
    for d in sorted({r["date"] for r in srv}):
        p = BASE / "data" / "kekka" / f"{d}.csv"
        if not p.exists():
            continue
        k = pd.read_csv(p, encoding="cp932", low_memory=False, dtype=str)
        k["rid"] = k["レースID(新)"].str[:16]
        for r in k.itertuples():
            t = str(r.単勝配当)
            o = float(t) / 100 if t.isdigit() else _num(t.strip("()"))
            o26[(r.rid, int(float(r.馬番)))] = o
    for r in srv:
        r["o1"] = o26.get((r["rid"], r["order"][0]))
    kek26 = {k: v for k, v in kek26.items() if v.get("valid")}
    races = [r for r in off if r["rid"] in km] + [r for r in srv if r["rid"] in kek26 and r["o1"] is not None and np.isfinite(r["o1"])]
    K = {**km, **kek26}
    print(f"   {sum(1 for r in races if r['period']=='C'):,} R (2026)", flush=True)

    print("[3/3] 集計 ...", flush=True)
    subsets = {"◎オッズ ≤1.4": lambda o: o <= 1.4, "◎オッズ ≤1.9": lambda o: o <= 1.9, "◎オッズ ≤2.4": lambda o: o <= 2.4, "全レース": lambda o: True}
    md = [f"# ここだのレースで何を買うか (sniper_ticket_bakeoff, {pd.Timestamp.now():%Y-%m-%d})", "",
          "AI 上位 (◎〇▲△) で組む券種セット、1R 1 万円均等割り、確定払戻。A=2023-24 / B=2025 (offline, 生スコア順) / C=2026 実配信。",
          "ROI = 払戻/投資 (%)。的中率 = 1 点以上的中したレース率。C の CI は開催日 bootstrap 95%。nan = 払戻データ欠損 (ワイドの一部) を除外。", ""]
    summary = []
    for sname, cond in subsets.items():
        sub = [r for r in races if cond(r["o1"])]
        md += [f"## {sname}  (A {sum(r['period']=='A' for r in sub):,}R / B {sum(r['period']=='B' for r in sub):,}R / C {sum(r['period']=='C' for r in sub):,}R)", "",
               "| 券種セット | 的中率 A/B/C | ROI A | ROI B | ROI C [CI] | 最悪期 |", "|---|---|---|---|---|---|"]
        rows = []
        for tname, legs in TICKETS.items():
            res = {}
            for per in ("A", "B", "C"):
                st = rt = 0.0; hit = 0; n = 0; by_date = {}
                for r in sub:
                    if r["period"] != per:
                        continue
                    k = K[r["rid"]]; o = r["order"]
                    if len(o) < 4:
                        continue
                    stake = BUDGET / len(legs); race_ret = 0.0; bad = False
                    for kind, idx in legs:
                        sel = dict(a1=o[idx[0]], a2=o[idx[1]] if len(idx) > 1 else o[1], a3=o[idx[2]] if len(idx) > 2 else o[2], waku1=0, waku2=0)
                        if kind == "sanpuku" and len(idx) == 3:
                            sel = dict(a1=o[idx[0]], a2=o[idx[1]], a3=o[idx[2]], waku1=0, waku2=0)
                        pay = sa.payout_per_100(kind, sel, k)
                        if not np.isfinite(pay):
                            bad = True; break
                        race_ret += stake / 100 * pay
                    if bad:
                        continue
                    n += 1; st += BUDGET; rt += race_ret; hit += race_ret > 0
                    a = by_date.setdefault(r["date"], [0.0, 0.0]); a[0] += BUDGET; a[1] += race_ret
                res[per] = dict(n=n, hit=100 * hit / n if n else np.nan, roi=100 * rt / st if st else np.nan, by_date=by_date)
            lo, hi = ci({d: tuple(v) for d, v in res["C"]["by_date"].items()}) if res["C"]["n"] else (np.nan, np.nan)
            worst = min(res["A"]["roi"], res["B"]["roi"], res["C"]["roi"])
            rows.append((worst, f"| {tname} | {res['A']['hit']:.0f}/{res['B']['hit']:.0f}/{res['C']['hit']:.0f}% | {res['A']['roi']:.1f} | {res['B']['roi']:.1f} | {res['C']['roi']:.1f} [{lo:.0f},{hi:.0f}] | {worst:.1f} |"))
            summary.append((sname, tname, worst, res["A"]["roi"], res["B"]["roi"], res["C"]["roi"]))
        md += [r for _, r in sorted(rows, key=lambda x: -x[0])] + [""]
    md += ["## 読み方", "", "- 「最悪期」= 3 期の ROI の最小値。これが高い券種が『どの期でも負けが小さい』。", "- 同じレース集合なので、券種の差は純粋に『どの券種が市場に対して相対的に安いか』を示す。"]
    OUT.write_bytes("\n".join(md).encode("utf-8"))
    (BASE / "docs" / "research" / "SNIPER_TICKET_BAKEOFF_20261112.md").write_bytes("\n".join(md).encode("utf-8"))
    print("\n".join(md[:60]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
