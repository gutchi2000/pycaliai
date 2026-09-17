# -*- coding: utf-8 -*-
"""
deep_bet_search.py — 全券種 × 買い方 × 点数 × 条件 × 金額 の総当たり探索
=======================================================================
「単勝/複勝/枠連/馬連/馬単/ワイド/三連複/三連単 のどれを、どう組んで、何点、
 どの条件のとき、いくらで買えば ROI > 100% になるのか」を OOS 実払戻で総当たりする。

substrate: data/_policy/bet_substrate.pkl  (本番 v6 の OOS 確率 × 実払戻, 2023-25 10,299R)

3分割プロトコル (ここが本体):
  discovery = 2023   … 全セルを見る。ここで良かったものだけ次へ。
  confirm   = 2024   … discovery 通過セルのみ評価 + BH-FDR 補正
  holdout   = 2025   … discovery/confirm の両方を通ったセルだけ、最後に一度だけ触る

なぜ3分割か:
  控除率 (単複20% / 枠連馬連馬単ワイド22.5% / 三連複25% / 三連単27.5%) を超えるには
  群衆を 25〜38% 上回る必要がある。1万セル探せば「ある年に ROI>100%」のセルは
  必ず数百個出る。独立期間での再現と多重比較補正を通らないものは全部ノイズ。

実行:
  python -m analysis.deep_bet_search
  python -m analysis.deep_bet_search --max-m 10 --reps 4000
"""
from __future__ import annotations
import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np

BASE = Path(__file__).resolve().parents[1]
SUB = BASE / "data/_policy/bet_substrate.pkl"
OUTDIR = BASE / "reports/deep_bet_search"

TAKEOUT = {"単勝": 80.0, "複勝": 80.0, "枠連": 77.5, "馬連": 77.5, "馬単": 77.5,
           "ワイド": 77.5, "三連複": 75.0, "三連単": 72.5}
BLEND_LAMBDA = 1.5  # data/t10_blend.json


# ----------------------------------------------------------------- orderings
def orderings(r):
    p = r["p"]
    o = np.where(np.isfinite(r["odds9"]) & (r["odds9"] > 0), r["odds9"], 9999.0)
    mkt = 1.0 / o
    mkt = mkt / mkt.sum()
    with np.errstate(divide="ignore"):
        blend = np.log(np.clip(p, 1e-9, None)) + BLEND_LAMBDA * np.log(np.clip(mkt, 1e-9, None))
    ev = p * o
    pf = r.get("p_fuku")
    out = {
        "ai": np.argsort(-p, kind="stable"),
        "blend": np.argsort(-blend, kind="stable"),
        "mkt": np.argsort(-mkt, kind="stable"),
        "value": np.argsort(-ev, kind="stable"),
        "antivalue": np.argsort(ev, kind="stable"),
    }
    if pf is not None and np.isfinite(pf).all():
        out["aifuku"] = np.argsort(-pf, kind="stable")
    else:
        out["aifuku"] = out["ai"]
    return out


# ------------------------------------------------------------- ticket makers
def tickets(kind, struct, order_idx, r, m):
    ban = r["ban"]
    pool = [int(ban[i]) for i in order_idx[:m]]
    if not pool:
        return []
    if kind in ("単勝", "複勝"):
        return [(b,) for b in pool]
    if kind == "枠連":
        waku = {int(ban[i]): int(r["waku"][i]) for i in range(len(ban)) if np.isfinite(r["waku"][i])}
        ws, seen = [], set()
        for b in pool:
            w = waku.get(b)
            if w is not None and w not in seen:
                seen.add(w); ws.append(w)
        if struct == "axis":
            return [tuple(sorted((ws[0], w))) for w in ws[1:]] if len(ws) >= 2 else []
        return [tuple(sorted(c)) for c in itertools.combinations(ws, 2)]
    if kind in ("馬連", "ワイド"):
        if struct == "axis":
            return [tuple(sorted((pool[0], b))) for b in pool[1:]]
        return [tuple(sorted(c)) for c in itertools.combinations(pool, 2)]
    if kind == "馬単":
        if struct == "axis":
            return [(pool[0], b) for b in pool[1:]]
        if struct == "axis2":
            return [(b, pool[0]) for b in pool[1:]]
        return list(itertools.permutations(pool, 2))
    if kind == "三連複":
        if struct == "axis":
            return [tuple(sorted((pool[0],) + c)) for c in itertools.combinations(pool[1:], 2)]
        if struct == "axis2":
            if len(pool) < 3:
                return []
            return [tuple(sorted((pool[0], pool[1], b))) for b in pool[2:]]
        return [tuple(sorted(c)) for c in itertools.combinations(pool, 3)]
    if kind == "三連単":
        if struct == "axis":
            return [(pool[0],) + c for c in itertools.permutations(pool[1:], 2)]
        if struct == "axis2":
            if len(pool) < 3:
                return []
            out = []
            for a, b in ((pool[0], pool[1]), (pool[1], pool[0])):
                out += [(a, b, c) for c in pool[2:]]
            return out
        return list(itertools.permutations(pool, 3))
    raise ValueError(kind)


# ------------------------------------------------------------------- settle
def settle(kind, tks, r):
    cost = 100.0 * len(tks)
    if cost == 0:
        return 0.0, 0.0
    f, s, t = r["first"], r["second"], r["third"]
    ret = 0.0
    if kind == "単勝":
        if (f,) in tks and np.isfinite(r["tan_pay"]):
            ret = r["tan_pay"]
    elif kind == "複勝":
        for b, pay in r["fuku_pay"].items():
            if (b,) in tks:
                ret += pay
    elif kind == "枠連":
        wf, ws = r["waku_first"], r["waku_second"]
        if np.isfinite(r["wakuren"]) and np.isfinite(wf) and np.isfinite(ws):
            if tuple(sorted((int(wf), int(ws)))) in tks:
                ret = r["wakuren"]
    elif kind == "馬連":
        if tuple(sorted((f, s))) in tks and np.isfinite(r["umaren"]):
            ret = r["umaren"]
    elif kind == "馬単":
        if (f, s) in tks and np.isfinite(r["umatan"]):
            ret = r["umatan"]
    elif kind == "ワイド":
        if r["wide"]:
            for i, j, pay in r["wide"]:
                if tuple(sorted((i, j))) in tks:
                    ret += pay
    elif kind == "三連複":
        if tuple(sorted((f, s, t))) in tks and np.isfinite(r["sanpuku"]):
            ret = r["sanpuku"]
    elif kind == "三連単":
        if (f, s, t) in tks and np.isfinite(r["sanrentan"]):
            ret = r["sanrentan"]
    return cost, ret


# ---------------------------------------------------------------- segments
def race_features(r):
    p = np.sort(r["p"])[::-1]
    o = np.where(np.isfinite(r["odds9"]) & (r["odds9"] > 0), r["odds9"], 9999.0)
    ai_top = int(np.argmax(r["p"]))
    return dict(
        n=r["n"], p1=float(p[0]), gap=float(p[0] - p[1]),
        fav_odds=float(np.min(o)), ai_odds=float(o[ai_top]),
        agree=int(np.argmin(o) == ai_top),
        month=int(r["date"][4:6]), place=r["place"],
    )


SEGMENTS = {
    "ALL": lambda f: True,
    "少頭数<=12": lambda f: f["n"] <= 12,
    "中頭数13-15": lambda f: 13 <= f["n"] <= 15,
    "多頭数>=16": lambda f: f["n"] >= 16,
    "◎堅p1>=.25": lambda f: f["p1"] >= 0.25,
    "◎並.12-.25": lambda f: 0.12 <= f["p1"] < 0.25,
    "◎薄p1<.12": lambda f: f["p1"] < 0.12,
    "独走gap>=.10": lambda f: f["gap"] >= 0.10,
    "混戦gap<.04": lambda f: f["gap"] < 0.04,
    "AI=市場一致": lambda f: f["agree"] == 1,
    "AI≠市場": lambda f: f["agree"] == 0,
    "◎9時>=6倍": lambda f: f["ai_odds"] >= 6.0,
    "◎9時3-6倍": lambda f: 3.0 <= f["ai_odds"] < 6.0,
    "◎9時<3倍": lambda f: f["ai_odds"] < 3.0,
    "1人気>=4倍": lambda f: f["fav_odds"] >= 4.0,
    "1人気<2倍": lambda f: f["fav_odds"] < 2.0,
    "夏6-9月": lambda f: 6 <= f["month"] <= 9,
    "冬春10-5月": lambda f: not (6 <= f["month"] <= 9),
}

# 金額(レース間の張り分け)。ticket 内は常に均等。
STAKES = {
    "flat": lambda f: 1.0,
    "p1比例": lambda f: max(f["p1"], 1e-3),
    "p1逆比例": lambda f: 1.0 / max(f["p1"], 1e-3),
    "◎オッズ比例": lambda f: min(f["ai_odds"], 50.0),
}


# ------------------------------------------------------------------- stats
def block_boot(cost, ret, days, reps=3000, seed=7):
    uniq, inv = np.unique(days, return_inverse=True)
    idx_by_day = [np.where(inv == i)[0] for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    n = len(uniq)
    out = np.empty(reps)
    for k in range(reps):
        sel = np.concatenate([idx_by_day[i] for i in rng.integers(0, n, n)])
        c = cost[sel].sum()
        out[k] = 100.0 * ret[sel].sum() / c if c > 0 else np.nan
    out = out[np.isfinite(out)]
    if len(out) == 0:
        return np.nan, np.nan, np.nan
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)),
            float(np.mean(out > 100.0)))


def bh_fdr(pvals, q=0.10):
    p = np.asarray(pvals, float)
    m = len(p)
    if m == 0:
        return np.zeros(0, bool)
    order = np.argsort(p)
    passed = p[order] <= q * (np.arange(1, m + 1) / m)
    keep = np.zeros(m, bool)
    if passed.any():
        keep[order[:np.max(np.where(passed)[0]) + 1]] = True
    return keep


# -------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-m", type=int, default=8)
    ap.add_argument("--reps", type=int, default=3000)
    ap.add_argument("--fdr", type=float, default=0.10)
    ap.add_argument("--min-races", type=int, default=200)
    args = ap.parse_args()

    races = joblib.load(SUB)
    feats = [race_features(r) for r in races]
    ords = [orderings(r) for r in races]
    year = np.array([r["date"][:4] for r in races])
    days = np.array([r["date"] for r in races])
    print(f"substrate: {len(races):,} races  "
          f"2023={int((year=='2023').sum())} 2024={int((year=='2024').sum())} "
          f"2025={int((year=='2025').sum())}")

    ORDERS = ["ai", "aifuku", "blend", "mkt", "value", "antivalue"]
    grid = []
    for on in ORDERS:
        for m in range(1, min(args.max_m, 6) + 1):
            grid += [("単勝", "topk", on, m), ("複勝", "topk", on, m)]
        for m in range(2, args.max_m + 1):
            for kind in ("馬連", "ワイド", "枠連"):
                grid += [(kind, "box", on, m), (kind, "axis", on, m)]
            grid += [("馬単", "axis", on, m), ("馬単", "axis2", on, m)]
            if m <= 6:
                grid.append(("馬単", "box", on, m))
        for m in range(3, args.max_m + 1):
            grid += [("三連複", "axis", on, m), ("三連複", "axis2", on, m),
                     ("三連単", "axis", on, m), ("三連単", "axis2", on, m)]
            if m <= 6:
                grid.append(("三連複", "box", on, m))
            if m <= 5:
                grid.append(("三連単", "box", on, m))
    ncells = len(grid) * len(SEGMENTS) * len(STAKES)
    print(f"policies={len(grid):,}  segments={len(SEGMENTS)}  stakes={len(STAKES)}  "
          f"cells={ncells:,}")

    COST = np.zeros((len(grid), len(races)))
    RET = np.zeros((len(grid), len(races)))
    for gi, (kind, struct, on, m) in enumerate(grid):
        if gi % 100 == 0:
            print(f"  settling {gi}/{len(grid)} ...", flush=True)
        for ri, r in enumerate(races):
            c, v = settle(kind, set(tickets(kind, struct, ords[ri][on], r, m)), r)
            COST[gi, ri] = c
            RET[gi, ri] = v

    segmasks = {s: np.array([fn(f) for f in feats]) for s, fn in SEGMENTS.items()}
    stakevec = {s: np.array([fn(f) for f in feats]) for s, fn in STAKES.items()}
    for s in stakevec:                       # 平均1になるよう正規化 (資金量を揃える)
        stakevec[s] = stakevec[s] / stakevec[s].mean()

    m23, m24, m25 = year == "2023", year == "2024", year == "2025"

    def roi(gi, mask, sv):
        c = (COST[gi] * sv)[mask].sum()
        return (100.0 * (RET[gi] * sv)[mask].sum() / c) if c > 0 else np.nan

    # ---------- stage 1: discovery = 2023 ----------
    cand = []
    for gi, (kind, struct, on, m) in enumerate(grid):
        act = COST[gi] > 0
        for sname, smask in segmasks.items():
            base = smask & act
            if (base & m23).sum() < args.min_races or (base & m24).sum() < args.min_races:
                continue
            for stname, sv in stakevec.items():
                r23 = roi(gi, base & m23, sv)
                if np.isfinite(r23) and r23 > 100.0:
                    cand.append(dict(gi=gi, kind=kind, struct=struct, order=on, m=m,
                                     seg=sname, stake=stname, roi23=round(r23, 2),
                                     pts=round(COST[gi][base & m23].mean() / 100.0, 2)))
    print(f"\n[stage1] discovery(2023) で ROI>100%: {len(cand):,} / {ncells:,} セル")

    # ---------- stage 2: confirm = 2024 ----------
    for c in cand:
        base = segmasks[c["seg"]] & (COST[c["gi"]] > 0)
        sv = stakevec[c["stake"]]
        c["roi24"] = round(roi(c["gi"], base & m24, sv), 2)
        c["n24"] = int((base & m24).sum())
    passed = [c for c in cand if c["roi24"] > 100.0]
    print(f"[stage2] うち confirm(2024) でも >100%: {len(passed):,} "
          f"(偶然なら期待 ~{len(cand)*0.35:,.0f})")

    for c in passed:
        base = segmasks[c["seg"]] & (COST[c["gi"]] > 0)
        sv = stakevec[c["stake"]]
        mk = base & m24
        lo, hi, p100 = block_boot((COST[c["gi"]] * sv)[mk], (RET[c["gi"]] * sv)[mk],
                                  days[mk], reps=args.reps)
        c["ci24"] = [round(lo, 1), round(hi, 1)]
        c["p24_gt100"] = round(p100, 4)
    keep = bh_fdr([1.0 - c["p24_gt100"] for c in passed], q=args.fdr)
    for c, k in zip(passed, keep):
        c["fdr_pass"] = bool(k)
    surv = [c for c in passed if c["fdr_pass"]]
    print(f"[stage2] BH-FDR(q={args.fdr}) 通過: {len(surv):,}")

    # ---------- stage 3: holdout = 2025 (ここで初めて触る) ----------
    for c in surv:
        base = segmasks[c["seg"]] & (COST[c["gi"]] > 0)
        sv = stakevec[c["stake"]]
        mk = base & m25
        c["n25"] = int(mk.sum())
        c["roi25"] = round(roi(c["gi"], mk, sv), 2)
        lo, hi, p100 = block_boot((COST[c["gi"]] * sv)[mk], (RET[c["gi"]] * sv)[mk],
                                  days[mk], reps=args.reps)
        c["ci25"] = [round(lo, 1), round(hi, 1)]
        c["p25_gt100"] = round(p100, 4)
        c["hit25"] = round(100 * float(np.mean(RET[c["gi"]][mk] > 0)), 2)
    surv.sort(key=lambda c: -c.get("roi25", 0))

    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "survivors.json").write_text(
        json.dumps([{k: v for k, v in c.items() if k != "gi"} for c in surv],
                   ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n===== stage3: holdout(2025) の結果 — 生存 {len(surv)} セル =====")
    hdr = f"{'券種':<5}{'型':<6}{'順':<10}{'m':>2} {'点数':>6} {'条件':<12}{'金額':<9}"
    print(hdr + f"{'ROI23':>7}{'ROI24':>7}{'ROI25':>7}  CI25            n25")
    final = []
    for c in surv:
        ok = c["roi25"] > 100.0
        print(f"{c['kind']:<5}{c['struct']:<6}{c['order']:<10}{c['m']:>2} {c['pts']:>6.1f}"
              f" {c['seg']:<12}{c['stake']:<9}{c['roi23']:>7.1f}{c['roi24']:>7.1f}"
              f"{c['roi25']:>7.1f}  {str(c['ci25']):<16}{c['n25']}"
              + ("  ★3期連続>100%" if ok else ""))
        if ok:
            final.append(c)

    print(f"\n===== 最終生存 (2023/2024/2025 の3期すべてで ROI>100%) : {len(final)} =====")
    if not final:
        print("  なし。")
        print("  → 「この券種のこの形・この点数・この金額なら儲かる」セルは、")
        print("     10,299R・{:,}セルの総当たりでは1つも再現しなかった。".format(ncells))
    for c in final:
        print(f"  ★ {c['kind']} {c['struct']} {c['order']} m={c['m']} {c['pts']:.1f}点 "
              f"{c['seg']} {c['stake']} : {c['roi23']:.1f} / {c['roi24']:.1f} / "
              f"{c['roi25']:.1f}%  CI25={c['ci25']} P(>100)={c['p25_gt100']}")

    # 参考: 各券種のベースライン (AI順・素直な買い方・ALL)
    print("\n===== 参考: 各券種 素の AI 順 (seg=ALL, flat) の3期 ROI =====")
    ref = [("単勝", "topk", 1), ("複勝", "topk", 1), ("枠連", "box", 2), ("馬連", "box", 2),
           ("馬単", "axis", 2), ("ワイド", "box", 3), ("三連複", "box", 4), ("三連単", "axis", 4)]
    sv = stakevec["flat"]
    allm = segmasks["ALL"]
    for kind, struct, m in ref:
        try:
            gi = grid.index((kind, struct, "ai", m))
        except ValueError:
            continue
        act = allm & (COST[gi] > 0)
        print(f"  {kind:<4} {struct}/m={m} {COST[gi][act].mean()/100:>4.0f}点 "
              f"控除床{TAKEOUT[kind]:>5.1f}%  "
              f"2023={roi(gi, act & m23, sv):>6.1f}  2024={roi(gi, act & m24, sv):>6.1f}  "
              f"2025={roi(gi, act & m25, sv):>6.1f}")
    print(f"\n書き出し: {OUTDIR}")


if __name__ == "__main__":
    main()
