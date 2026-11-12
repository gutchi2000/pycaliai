# -*- coding: utf-8 -*-
"""
easy_ticket_map.py — 「当たりやすいのに配当もそこそこ」の券種・買い方が複勝以外にあるか (2026-11-12)
=====================================================================================
全券種 (単勝・枠連・馬連・ワイド・馬単・三連複・三連単) を、発走約 28 分前の人気順で機械的に組んだ 1 点〜10 点の買い方と、
同日に見つけた 2 つの信号 (朝から売れなくなった馬 / 複勝票数比) を使った買い方で、的中率・的中時の戻り倍率・回収率を 3 期で並べる。
採点は確定払戻。ワイドは払戻データが 2016 年から。同着レースは除外。AI は入れない (2013-2025 を通しで見るため。上位候補だけ別途 AI を重ねる)。
出力: reports/easy_ticket_map.md (+ docs/research 控え)、data/_joint/easy_ticket_races.parquet
"""
from __future__ import annotations

import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
OUT = BASE / "reports" / "easy_ticket_map.md"
PER = {"P1 13-18": (2013, 2018), "P2 19-22": (2019, 2022), "P3 23-25": (2023, 2025)}
KEKKA = BASE / "data" / "kekka_20130105-20251228.csv"
WIDE = BASE / "data" / "wide_payouts_2016-2025.parquet"
MASTER = BASE / "data" / "master_v2_20130105-20251228.csv"


def _f(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def results() -> dict:
    k = pd.read_csv(KEKKA, encoding="cp932", dtype=str)
    k["rid"] = k["レースID(新)"].str[:16]; k["ban"] = pd.to_numeric(k["馬番"], errors="coerce"); k["fin"] = pd.to_numeric(k["確定着順"], errors="coerce")
    w = pd.read_parquet(WIDE)
    wide = {str(r.race_id): {frozenset((int(r.w1_i), int(r.w1_j))): float(r.w1_pay), frozenset((int(r.w2_i), int(r.w2_j))): float(r.w2_pay),
                             frozenset((int(r.w3_i), int(r.w3_j))): float(r.w3_pay)} for r in w.itertuples()}
    top = k[k.fin.isin([1, 2, 3])]
    res = {}
    for rid, g in top.groupby("rid"):
        if len(g) != 3 or set(g.fin) != {1, 2, 3}:
            continue
        g = g.sort_values("fin")
        r1, r2, r3 = g.iloc[0], g.iloc[1], g.iloc[2]
        res[rid] = dict(f=int(r1.ban), s=int(r2.ban), t=int(r3.ban), tan=_f(r1["単勝配当"]), wakuren=_f(r1["枠連"]), umaren=_f(r1["馬連"]),
                        umatan=_f(r1["馬単"]), sanpuku=_f(r1["３連複"]), sanrentan=_f(r1["３連単"]), wide=wide.get(rid))
    return res


def pay(kind: str, sel: tuple, k: dict, waku: dict) -> float:
    f, s, t = k["f"], k["s"], k["t"]
    if kind == "tan":
        return k["tan"] if sel[0] == f else 0.0
    if kind == "umaren":
        return k["umaren"] if set(sel) == {f, s} else 0.0
    if kind == "umatan":
        return k["umatan"] if tuple(sel) == (f, s) else 0.0
    if kind == "sanpuku":
        return k["sanpuku"] if set(sel) == {f, s, t} else 0.0
    if kind == "sanrentan":
        return k["sanrentan"] if tuple(sel) == (f, s, t) else 0.0
    if kind == "wide":
        if k["wide"] is None:
            return np.nan
        return k["wide"].get(frozenset(sel), 0.0)
    if kind == "wakuren":
        a, b = waku.get(sel[0]), waku.get(sel[1]); wf, ws = waku.get(f), waku.get(s)
        if None in (a, b, wf, ws):
            return np.nan
        return k["wakuren"] if sorted((a, b)) == sorted((wf, ws)) else 0.0
    raise ValueError(kind)


def recipes(c: dict) -> dict:
    """c: m (人気順の馬番), o (同オッズ), drift (同), n, pool_ratio。返り値: 名前 → [(kind, sel), ...] (該当しないものは入れない)。"""
    m, o, dr = c["m"], c["o"], c["drift"]
    r = {}
    r["単勝 1番人気"] = [("tan", (m[0],))]
    r["単勝 2番人気"] = [("tan", (m[1],))]
    r["枠連 1-2番人気"] = [("wakuren", (m[0], m[1]))]
    r["馬連 1-2番人気"] = [("umaren", (m[0], m[1]))]
    r["馬連 1-3番人気"] = [("umaren", (m[0], m[2]))]
    r["馬単 1→2番人気"] = [("umatan", (m[0], m[1]))]
    r["馬単 1⇄2番人気 2点"] = [("umatan", (m[0], m[1])), ("umatan", (m[1], m[0]))]
    r["ワイド 1-2番人気"] = [("wide", (m[0], m[1]))]
    r["ワイド 1-3番人気"] = [("wide", (m[0], m[2]))]
    r["ワイド 2-3番人気"] = [("wide", (m[1], m[2]))]
    r["ワイド 1番人気流し 3点 (2-4番人気)"] = [("wide", (m[0], m[i])) for i in (1, 2, 3)]
    r["ワイド 3頭BOX 3点 (1-3番人気)"] = [("wide", p) for p in combinations(m[:3], 2)]
    r["ワイド 4頭BOX 6点"] = [("wide", p) for p in combinations(m[:4], 2)]
    r["馬連 1番人気流し 3点"] = [("umaren", (m[0], m[i])) for i in (1, 2, 3)]
    r["馬連 3頭BOX 3点"] = [("umaren", p) for p in combinations(m[:3], 2)]
    r["三連複 1-2-3番人気 1点"] = [("sanpuku", tuple(m[:3]))]
    r["三連複 4頭BOX 4点"] = [("sanpuku", p) for p in combinations(m[:4], 3)]
    r["三連複 5頭BOX 10点"] = [("sanpuku", p) for p in combinations(m[:5], 3)]
    r["三連複 1番人気軸 2-5番人気 6点"] = [("sanpuku", (m[0],) + p) for p in combinations(m[1:5], 2)]
    r["三連単 1→2→3番人気 1点"] = [("sanrentan", tuple(m[:3]))]
    # --- 条件つき ---
    if o[0] <= 1.9:
        r["[1番人気 ≤1.9倍] 単勝"] = [("tan", (m[0],))]
        r["[1番人気 ≤1.9倍] ワイド 1-2番人気"] = [("wide", (m[0], m[1]))]
        r["[1番人気 ≤1.9倍] ワイド 1番人気流し 2点"] = [("wide", (m[0], m[1])), ("wide", (m[0], m[2]))]
        r["[1番人気 ≤1.9倍] 馬連 1-2番人気"] = [("umaren", (m[0], m[1]))]
        r["[1番人気 ≤1.9倍] 馬単 1→2番人気"] = [("umatan", (m[0], m[1]))]
        r["[1番人気 ≤1.9倍] 馬単 1→2,3番人気 2点"] = [("umatan", (m[0], m[1])), ("umatan", (m[0], m[2]))]
        r["[1番人気 ≤1.9倍] 三連複 1軸 2-4番人気 3点"] = [("sanpuku", (m[0],) + p) for p in combinations(m[1:4], 2)]
        if c["pool_ratio"] >= 2.19:
            r["[≤1.9倍 + 複勝票数比 ≥2.19] 単勝"] = [("tan", (m[0],))]
            r["[≤1.9倍 + 複勝票数比 ≥2.19] ワイド 1-2番人気"] = [("wide", (m[0], m[1]))]
            r["[≤1.9倍 + 複勝票数比 ≥2.19] ワイド 1番人気流し 2点"] = [("wide", (m[0], m[1])), ("wide", (m[0], m[2]))]
            r["[≤1.9倍 + 複勝票数比 ≥2.19] ワイド 1番人気流し 3点"] = [("wide", (m[0], m[i])) for i in (1, 2, 3)]
            r["[≤1.9倍 + 複勝票数比 ≥2.19] 馬連 1-2番人気"] = [("umaren", (m[0], m[1]))]
            r["[≤1.9倍 + 複勝票数比 ≥2.19] 馬単 1→2番人気"] = [("umatan", (m[0], m[1]))]
            r["[≤1.9倍 + 複勝票数比 ≥2.19] 馬単 1→2,3番人気 2点"] = [("umatan", (m[0], m[1])), ("umatan", (m[0], m[2]))]
            r["[≤1.9倍 + 複勝票数比 ≥2.19] 三連複 1軸 2-4番人気 3点"] = [("sanpuku", (m[0],) + p) for p in combinations(m[1:4], 2)]
    if o[0] >= 3.0:
        r["[1番人気 3倍以上 = 混戦] ワイド 3頭BOX 3点"] = [("wide", p) for p in combinations(m[:3], 2)]
        r["[1番人気 3倍以上 = 混戦] 三連複 5頭BOX 10点"] = [("sanpuku", p) for p in combinations(m[:5], 3)]
    if c["n"] <= 10:
        r["[10頭以下] ワイド 1-2番人気"] = [("wide", (m[0], m[1]))]
        r["[10頭以下] 三連複 4頭BOX 4点"] = [("sanpuku", p) for p in combinations(m[:4], 3)]
        r["[10頭以下] 馬連 3頭BOX 3点"] = [("umaren", p) for p in combinations(m[:3], 2)]
    # 朝から売れなくなった馬 (発走前 3〜6.9 倍、朝比 +13% 以上)。複数いれば最も伸びた馬。
    cand = [i for i in range(len(m)) if 3.0 <= o[i] < 7.0 and dr[i] > 0.12]
    if cand:
        j = max(cand, key=lambda i: dr[i]); x = m[j]
        r["[朝から売れなくなった馬 X] 単勝 X"] = [("tan", (x,))]
        if j != 0:
            r["[朝から売れなくなった馬 X] ワイド 1番人気-X"] = [("wide", (m[0], x))]
            r["[朝から売れなくなった馬 X] 馬連 1番人気-X"] = [("umaren", (m[0], x))]
            r["[朝から売れなくなった馬 X] 馬単 X→1番人気"] = [("umatan", (x, m[0]))]
            r["[朝から売れなくなった馬 X] 馬単 1番人気⇄X 2点"] = [("umatan", (x, m[0])), ("umatan", (m[0], x))]
            others = [h for h in m[:4] if h != x][:3]
            r["[朝から売れなくなった馬 X] ワイド X流し 3点 (上位人気へ)"] = [("wide", (x, h)) for h in others]
            r["[朝から売れなくなった馬 X] 三連複 1番人気-X 2頭軸 3点"] = [("sanpuku", (m[0], x, h)) for h in [h for h in m[1:5] if h != x][:3]]
        if dr[j] > 0.3:
            r["[朝から大きく売れなくなった馬 X (+35%)] 単勝 X"] = [("tan", (x,))]
            if j != 0:
                r["[朝から大きく売れなくなった馬 X (+35%)] ワイド 1番人気-X"] = [("wide", (m[0], x))]
    return r


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    print("[1/3] load ...", flush=True)
    d = pd.read_parquet(BASE / "data" / "_joint" / "tanpuk_all_drift.parquet")
    d["drift"] = np.log(d.o_pre / d.o_win9)
    fv = pd.read_parquet(BASE / "data" / "_joint" / "tanpuk_fav.parquet", columns=["rid", "pool_tan_pre", "pool_fuku_pre"])
    pr = dict(zip(fv.rid, fv.pool_fuku_pre / fv.pool_tan_pre))
    mw = pd.read_csv(MASTER, encoding="utf-8-sig", usecols=["レースID(新/馬番無)", "馬番", "枠番"], low_memory=False)
    mw["rid"] = mw["レースID(新/馬番無)"].astype(str).str[:16]
    waku_all = {rid: dict(zip(g["馬番"].astype(int), g["枠番"].astype(int))) for rid, g in mw.dropna().groupby("rid")}
    K = results()
    print(f"   結果 {len(K):,}R", flush=True)
    print("[2/3] settle ...", flush=True)
    rows = []
    d = d.sort_values(["rid", "o_pre", "ban"])
    for rid, g in d.groupby("rid", sort=False):
        k = K.get(rid)
        if k is None or len(g) < 8:
            continue
        c = dict(m=g.ban.astype(int).tolist(), o=g.o_pre.tolist(), drift=g.drift.tolist(), n=len(g), pool_ratio=pr.get(rid, np.nan))
        if not np.isfinite(c["pool_ratio"]):
            c["pool_ratio"] = 0.0
        waku = waku_all.get(rid, {})
        year = int(rid[:4])
        for name, legs in recipes(c).items():
            ret = 0.0; bad = False
            for kind, sel in legs:
                p = pay(kind, sel, k, waku)
                if not np.isfinite(p):
                    bad = True; break
                ret += p
            if bad:
                continue
            rows.append((rid, year, name, len(legs), ret))
    R = pd.DataFrame(rows, columns=["rid", "year", "name", "pts", "ret"])
    R["stake"] = R.pts * 100.0; R["date"] = R.rid.str[:8]
    R.to_parquet(BASE / "data" / "_joint" / "easy_ticket_races.parquet")

    print("[3/3] 集計 ...", flush=True)
    rng = np.random.default_rng(0)
    out = []
    for name, g in R.groupby("name", sort=False):
        per = {}
        for p, (a, b) in PER.items():
            x = g[(g.year >= a) & (g.year <= b)]
            per[p] = (len(x), 100 * x.ret.sum() / x.stake.sum() if len(x) else np.nan)
        hit = g.ret > 0
        byd = g.groupby("date")[["ret", "stake"]].sum().values
        i = rng.integers(0, len(byd), (2000, len(byd)))
        v = 100 * byd[i, 0].sum(1) / byd[i, 1].sum(1)
        out.append(dict(name=name, n=len(g), per_year=len(g) / g.year.nunique(), hit=100 * hit.mean(), mult=(g.ret[hit] / g.stake[hit]).mean(),
                        med=(g.ret[hit] / g.stake[hit]).median(), roi=100 * g.ret.sum() / g.stake.sum(), lo=np.percentile(v, 2.5), hi=np.percentile(v, 97.5),
                        p=per, worst=np.nanmin([q[1] for q in per.values()])))
    T = pd.DataFrame(out)

    def table(df, title):
        md = [f"### {title}", "", "| 買い方 | 年あたり | 的中率 | 的中時の戻り (平均 / 中央値) | 回収率 [95% CI] | 3 期 | 最悪期 |", "|---|---|---|---|---|---|---|"]
        for r in df.itertuples():
            md.append(f"| {r.name} | {r.per_year:,.0f} | {r.hit:.0f}% | {r.mult:.2f} 倍 / {r.med:.2f} 倍 | **{r.roi:.1f}** [{r.lo:.0f},{r.hi:.0f}] | "
                      f"{'/'.join(f'{q[1]:.0f}' for q in r.p.values())} | {r.worst:.1f} |")
        return md + [""]

    plain = T[~T.name.str.startswith("[")]
    cond = T[T.name.str.startswith("[")]
    md = [f"# 当たりやすくて配当もそこそこ、は複勝以外にあるか (easy_ticket_map, {pd.Timestamp.now():%Y-%m-%d})", "",
          "発走約 28 分前の人気順 (単勝オッズ順) で機械的に組んだ買い方。8 頭以上、同着レース除外、確定払戻、1 点 100 円。",
          "『的中時の戻り』= 当たったレースで、そのレースに賭けた総額の何倍が返るか (2 点買いで片方 300 円的中なら 1.5 倍)。",
          "3 期 = 2013-18 / 2019-22 / 2023-25 (ワイドは 2016 年から)。AI は使っていない。", "",
          "## 1. 無条件 (全レース)", ""] + table(plain.sort_values("hit", ascending=False), "的中率の高い順")
    md += ["## 2. 条件つき (1 番人気の強さ・票数比・朝から売れなくなった馬)", ""] + table(cond.sort_values("roi", ascending=False), "回収率の高い順")
    easy = T[(T.hit >= 35) & (T.mult >= 1.8)].sort_values("worst", ascending=False)
    md += ["## 3. 『的中率 35% 以上 かつ 的中時 1.8 倍以上』を満たす買い方", ""] + (table(easy, "最悪期の回収率が高い順") if len(easy) else ["- なし", ""])
    mid = T[(T.hit >= 20) & (T.mult >= 3.0)].sort_values("worst", ascending=False)
    md += ["## 4. 『的中率 20% 以上 かつ 的中時 3 倍以上』を満たす買い方", ""] + (table(mid, "最悪期の回収率が高い順") if len(mid) else ["- なし", ""])
    OUT.write_bytes("\n".join(md).encode("utf-8"))
    (BASE / "docs" / "research" / "EASY_TICKET_MAP_20261112.md").write_bytes("\n".join(md).encode("utf-8"))
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
