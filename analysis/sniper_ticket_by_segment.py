# -*- coding: utf-8 -*-
"""
sniper_ticket_by_segment.py — ◎ ≤1.9 の参戦集合をクラス・芝ダ・距離・頭数・◎帯で割り、券種を変えるべきセグメントがあるか (2026-11-12)
=================================================================================================
ユーザー「条件戦とかで変えないの？」。sniper_ticket_bakeoff と同じ集合・同じ券種セットを、セグメントごとに A=2023-24 / B=2025 / C=2026 で比較。
『複勝◎ 1 点』を基準に、3 期すべてで複勝を上回る券種があるセグメントだけを「候補」として出す。
出力: reports/sniper_ticket_by_segment.md (+ docs/research 控え)
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
from analysis.market_bias_atlas import load_ext, load_master, norm_class  # noqa: E402
from analysis.sniper_ticket_bakeoff import BUDGET, SCORES, TICKETS, _num, kekka_master  # noqa: E402

OUT = BASE / "reports" / "sniper_ticket_by_segment.md"
ODDS_MAX = 1.9
NON_DIGIT = re.compile(r"[^0-9]")


def class_group(k: str) -> str:
    k = norm_class(k)
    if k in ("G1", "G2", "G3", "OP"):
        return "OP/重賞"
    return k if k in ("新馬", "未勝利", "1勝", "2勝", "3勝") else "他"


def dist_bucket(d) -> str:
    d = float(d)
    return "短(≤1400)" if d <= 1400 else "マイル(1401-1800)" if d <= 1800 else "中(1801-2200)" if d <= 2200 else "長(2201+)"


def field_bucket(n) -> str:
    n = int(n)
    return "≤10頭" if n <= 10 else "11-14頭" if n <= 14 else "15頭+"


def offline_races() -> list[dict]:
    df = load_ext(); df = df[df.year >= 2023]
    sc = pd.read_parquet(SCORES)
    sc["rid16"] = sc["レースID(新/馬番無)"].astype(str).str[:16]; sc["ban"] = pd.to_numeric(sc["馬番"], errors="coerce")
    m = load_master()[["rid16", "klass"]].drop_duplicates("rid16")
    df = df.merge(sc[["rid16", "ban", "_score"]], on=["rid16", "ban"], how="inner").merge(m, on="rid16", how="left")
    df = df.sort_values(["rid16", "_score"], ascending=[True, False])
    out = []
    for rid, g in df.groupby("rid16", sort=False):
        if len(g) < 5:
            continue
        h = g.iloc[0]
        out.append(dict(rid=rid, date=str(h.date), period="A" if int(h.year) <= 2024 else "B", order=g.ban.astype(int).tolist()[:4],
                        o1=float(h.odds), klass=class_group(h.klass), surface=str(h.surface), dist=dist_bucket(h.dist), field=field_bucket(len(g))))
    return out


def served_races() -> list[dict]:
    out = []
    for bp in sorted(glob.glob(str(BASE / "reports" / "cowork_input" / "2026????_bundle.json"))):
        d = Path(bp).name[:8]
        b = json.loads(Path(bp).read_text(encoding="utf-8"))
        for r in (b["races"] if isinstance(b["races"], list) else list(b["races"].values())):
            rid = NON_DIGIT.sub("", str(r["race_id"]))[:16]
            hs = [h for h in r["horses"] if h.get("p_win") is not None]
            if len(hs) < 5:
                continue
            hs = sorted(hs, key=lambda h: (-float(h["p_win"]), int(h["umaban"])))
            rm = r.get("race_meta") or {}
            course = str(rm.get("course", ""))
            dnum = NON_DIGIT.sub("", course)
            out.append(dict(rid=rid, date=d, period="C", order=[int(h["umaban"]) for h in hs[:4]], o1=None,
                            klass=class_group(rm.get("class", "")), surface=course[:1] if course[:1] in ("芝", "ダ") else "?",
                            dist=dist_bucket(dnum) if dnum else "?", field=field_bucket(rm.get("field_size") or len(hs))))
    return out


def roi_table(sub: list[dict], K: dict) -> dict[str, dict]:
    res = {}
    for tname, legs in TICKETS.items():
        per = {}
        for p in ("A", "B", "C"):
            st = rt = 0.0; n = 0
            for r in sub:
                if r["period"] != p:
                    continue
                k = K[r["rid"]]; o = r["order"]; stake = BUDGET / len(legs); ret = 0.0; bad = False
                for kind, idx in legs:
                    sel = dict(a1=o[idx[0]], a2=o[idx[1]] if len(idx) > 1 else o[1], a3=o[idx[2]] if len(idx) > 2 else o[2], waku1=0, waku2=0)
                    pay = sa.payout_per_100(kind, sel, k)
                    if not np.isfinite(pay):
                        bad = True; break
                    ret += stake / 100 * pay
                if bad:
                    continue
                n += 1; st += BUDGET; rt += ret
            per[p] = (n, 100 * rt / st if st else np.nan)
        res[tname] = per
    return res


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    km = kekka_master(); off = offline_races()
    wide26 = sa.load_wide(); srv = served_races(); kek26 = {}
    for d in sorted({r["date"] for r in srv}):
        kek26.update(sa.load_kekka(d, wide26))
    kek26 = {k: v for k, v in kek26.items() if v.get("valid")}
    o26 = {}
    for d in sorted({r["date"] for r in srv}):
        p = BASE / "data" / "kekka" / f"{d}.csv"
        if not p.exists():
            continue
        k = pd.read_csv(p, encoding="cp932", low_memory=False, dtype=str); k["rid"] = k["レースID(新)"].str[:16]
        for r in k.itertuples():
            t = str(r.単勝配当); o26[(r.rid, int(float(r.馬番)))] = float(t) / 100 if t.isdigit() else _num(t.strip("()"))
    for r in srv:
        r["o1"] = o26.get((r["rid"], r["order"][0]))
    K = {**km, **kek26}
    races = [r for r in off if r["rid"] in km] + [r for r in srv if r["rid"] in kek26 and r["o1"] is not None and np.isfinite(r["o1"])]
    base = [r for r in races if r["o1"] <= ODDS_MAX and len(r["order"]) >= 4]
    base_tab = roi_table(base, K)

    md = [f"# ◎ ≤{ODDS_MAX} の参戦集合をセグメントで割る (sniper_ticket_by_segment, {pd.Timestamp.now():%Y-%m-%d})", "",
          f"全体 (A {sum(r['period']=='A' for r in base):,}R / B {sum(r['period']=='B' for r in base):,}R / C {sum(r['period']=='C' for r in base):,}R): "
          f"複勝◎ = {base_tab['複勝◎']['A'][1]:.1f} / {base_tab['複勝◎']['B'][1]:.1f} / {base_tab['複勝◎']['C'][1]:.1f}%。",
          "各セグメントで『複勝◎ 1 点』と、最悪期 ROI が最も高い別券種を並べる。**候補** = 3 期すべてで複勝◎を上回った券種 (n が各期 30R 以上のときだけ)。", ""]
    axes = {"クラス": "klass", "芝ダ": "surface", "距離": "dist", "頭数": "field",
            "◎帯": lambda r: "1.0-1.4" if r["o1"] <= 1.4 else "1.5-1.9"}
    cand = []
    for ax, key in axes.items():
        md += [f"## {ax}", "", "| セグメント | n A/B/C | 複勝◎ A/B/C | 最良の別券種 | A/B/C | 判定 |", "|---|---|---|---|---|---|"]
        f = key if callable(key) else (lambda r, k=key: r[k])
        for seg in sorted({f(r) for r in base}):
            sub = [r for r in base if f(r) == seg]
            n = {p: sum(r["period"] == p for r in sub) for p in "ABC"}
            if sum(n.values()) < 30:
                continue
            tab = roi_table(sub, K); fk = tab["複勝◎"]
            others = {t: v for t, v in tab.items() if t != "複勝◎"}
            best = max(others, key=lambda t: np.nanmin([others[t][p][1] for p in "ABC"]) if all(others[t][p][0] for p in "ABC") else -1)
            bv = others[best]
            beats = all(bv[p][0] >= 30 and np.isfinite(bv[p][1]) and bv[p][1] > fk[p][1] for p in "ABC")
            verdict = "**候補**" if beats else ("複勝のまま" if all(fk[p][0] >= 30 for p in "ABC") else "n不足")
            if beats:
                cand.append((ax, seg, best, bv, fk))
            md.append(f"| {seg} | {n['A']}/{n['B']}/{n['C']} | {fk['A'][1]:.1f}/{fk['B'][1]:.1f}/{fk['C'][1]:.1f} | {best} | {bv['A'][1]:.1f}/{bv['B'][1]:.1f}/{bv['C'][1]:.1f} | {verdict} |")
        md.append("")
    md += ["## 候補 (3 期すべてで複勝◎を上回った券種)", ""]
    md += [f"- {ax}={seg}: {best} {bv['A'][1]:.1f}/{bv['B'][1]:.1f}/{bv['C'][1]:.1f} vs 複勝 {fk['A'][1]:.1f}/{fk['B'][1]:.1f}/{fk['C'][1]:.1f}" for ax, seg, best, bv, fk in cand] or ["- なし"]
    OUT.write_bytes("\n".join(md).encode("utf-8"))
    (BASE / "docs" / "research" / "SNIPER_TICKET_BY_SEGMENT_20261112.md").write_bytes("\n".join(md).encode("utf-8"))
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
