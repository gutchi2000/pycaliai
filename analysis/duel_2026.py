# -*- coding: utf-8 -*-
"""
duel_2026.py — 2026 実 serve (朝の bundle 印) × 実払戻で、買い目設計を対決させる。

  Arm A: 「2023-25 表で 90 以上のセル全採用」(◎オッズ帯ごとに券種セットが変わる、平均 5.8 点/R)
  Arm B: 「馬単1点 + ワイド1点 + 三連単1点」(全レース 3 点)
  参照 : 馬単1点のみ / ◎複勝1点 / 各券種 1 点

期間: bundle が存在する 2026-04-18 以降〜指定日まで。全券種 100 円均等。
実行: python -m analysis.duel_2026 [--until 20260831]
"""
from __future__ import annotations
import argparse, glob, json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from analysis.deep_bet_search import block_boot  # noqa
import build_site  # parse_wide_kekka

COLS = ["tan", "fuku", "umaren", "umatan", "wide", "wakuren", "sanpuku", "sanrentan"]
JP = {"tan": "単勝", "fuku": "複勝", "umaren": "馬連", "umatan": "馬単", "wide": "ワイド",
      "wakuren": "枠連", "sanpuku": "三連複", "sanrentan": "三連単"}
BANDS = [(0, 2, "<2倍"), (2, 3, "2-3倍"), (3, 5, "3-5倍"), (5, 8, "5-8倍"), (8, 15, "8-15倍"), (15, 1e9, "15倍〜")]
# 2023-25 全期間表 (analysis/top1_by_odds_band.py) の ROI>=90 セル
ARM_A = {
    "<2倍": ["fuku", "umaren", "umatan", "wide"],
    "2-3倍": ["umaren", "umatan", "wide", "sanpuku", "sanrentan"],
    "3-5倍": ["tan", "umaren", "umatan", "wide", "wakuren", "sanpuku", "sanrentan"],
    "5-8倍": ["tan", "umaren", "umatan", "wide", "wakuren", "sanpuku", "sanrentan"],
    "8-15倍": ["tan", "umaren", "umatan"],
    "15倍〜": ["fuku", "umaren", "wide", "wakuren"],
}
ARM_B = ["umatan", "wide", "sanrentan"]
# 2023-25 全期間表の各セル ROI (期待値の参照用)
BT = {
    "<2倍":  dict(tan=81.4, fuku=91.4, umaren=96.3, umatan=93.2, wide=91.7, wakuren=77.7, sanpuku=74.6, sanrentan=64.7),
    "2-3倍": dict(tan=83.8, fuku=88.4, umaren=97.7, umatan=98.9, wide=92.6, wakuren=86.3, sanpuku=108.0, sanrentan=139.7),
    "3-5倍": dict(tan=90.6, fuku=89.3, umaren=99.8, umatan=104.4, wide=94.4, wakuren=94.3, sanpuku=95.4, sanrentan=126.7),
    "5-8倍": dict(tan=91.9, fuku=88.4, umaren=92.8, umatan=99.5, wide=95.0, wakuren=96.5, sanpuku=98.1, sanrentan=108.5),
    "8-15倍": dict(tan=96.2, fuku=84.7, umaren=96.8, umatan=125.3, wide=86.6, wakuren=89.9, sanpuku=86.3, sanrentan=39.1),
    "15倍〜": dict(tan=78.1, fuku=110.9, umaren=117.2, umatan=59.9, wide=94.4, wakuren=90.2, sanpuku=38.6, sanrentan=0.0),
}


def waku_of(umaban: int, n: int) -> int:
    """JRA の枠番割当 (出馬表確定時の頭数 n)。n<=8 は馬番=枠番。"""
    if n <= 8:
        return umaban
    if n <= 16:
        singles = 16 - n                      # 1 頭枠の数 (1 枠から)
        if umaban <= singles:
            return umaban
        return singles + (umaban - singles + 1) // 2
    if n == 17:                               # 1-7 枠 2 頭、8 枠 3 頭
        return (umaban + 1) // 2 if umaban <= 14 else 8
    # n == 18: 1-6 枠 2 頭、7-8 枠 3 頭
    if umaban <= 12:
        return (umaban + 1) // 2
    return 7 if umaban <= 15 else 8


def band_of(odds: float) -> str:
    for lo, hi, name in BANDS:
        if lo <= odds < hi:
            return name
    return "15倍〜"


def _num(x):
    try:
        s = str(x).strip()
        if s.startswith("("):
            return np.nan
        return float(s.replace(",", ""))
    except Exception:
        return np.nan


def load(until: str, since: str = "20260101"):
    wide = build_site.parse_wide_kekka()
    rows, waku_check = [], [0, 0]
    for bp in sorted(glob.glob(str(BASE / "reports/cowork_input/2026*_bundle.json"))):
        date = Path(bp).name[:8]
        if date > until or date < since:
            continue
        kp = BASE / "data" / "kekka" / f"{date}.csv"
        if not kp.exists():
            continue
        k = pd.read_csv(kp, encoding="cp932", low_memory=False)
        k["rid16"] = k["レースID(新)"].astype(str).str[:16]
        k["ban"] = pd.to_numeric(k["馬番"], errors="coerce")
        k["fin"] = pd.to_numeric(k["確定着順"], errors="coerce")
        k["waku"] = pd.to_numeric(k["枠番"], errors="coerce")
        res = {}
        for rid, g in k.groupby("rid16"):
            g = g[g.fin.notna()]
            f1, f2, f3 = g[g.fin == 1], g[g.fin == 2], g[g.fin == 3]
            if len(f1) != 1 or len(f2) != 1 or len(f3) != 1:
                continue  # 同着・欠損は除外
            r1, r2, r3 = f1.iloc[0], f2.iloc[0], f3.iloc[0]
            y, m, d = int("20" + str(r1["日付"])[:2]), int(str(r1["日付"])[2:4]), int(str(r1["日付"])[4:6])
            wkey = (y, m, d, str(r1["場所"]).strip(), int(r1["Ｒ"]))
            res[rid] = dict(
                first=int(r1.ban), second=int(r2.ban), third=int(r3.ban),
                w1=int(r1.waku), w2=int(r2.waku), w3=int(r3.waku),
                tan=_num(r1["単勝配当"]),
                fuku={int(r.ban): _num(r["複勝配当"]) for _, r in g[g.fin <= 3].iterrows()},
                wakuren=_num(r1["枠連"]), umaren=_num(r1["馬連"]), umatan=_num(r1["馬単"]),
                sanpuku=_num(r1["３連複"]), sanrentan=_num(r1["３連単"]),
                wide=wide.get(wkey, {}),
            )
        b = json.loads(Path(bp).read_text(encoding="utf-8"))
        races = b["races"] if isinstance(b["races"], list) else list(b["races"].values())
        for r in races:
            rid = re.sub(r"\D", "", str(r.get("race_id", "")))[:16]
            if rid not in res:
                continue
            hs = [h for h in r.get("horses", []) if h.get("p_win") is not None and h.get("umaban") is not None]
            if len(hs) < 5:
                continue
            n = len(hs)
            order = sorted(hs, key=lambda h: -float(h["p_win"]))
            a1, a2, a3 = (int(order[i]["umaban"]) for i in range(3))
            o1 = order[0].get("tansho_odds")
            try:
                o1 = float(o1)
            except Exception:
                o1 = np.nan
            if not np.isfinite(o1) or o1 <= 0:
                continue
            x = res[rid]
            # 枠番割当の検算 (上位3着の実枠番と照合)
            for ban, w in ((x["first"], x["w1"]), (x["second"], x["w2"]), (x["third"], x["w3"])):
                waku_check[0] += 1
                waku_check[1] += int(waku_of(ban, n) == w)
            f, s, t = x["first"], x["second"], x["third"]
            wa1, wa2 = waku_of(a1, n), waku_of(a2, n)
            wp = x["wide"].get(f"{min(a1, a2)}-{max(a1, a2)}", 0) if x["wide"] else np.nan
            rows.append(dict(
                date=date, month=date[:6], rid=rid, n=n, ai_odds=o1, band=band_of(o1),
                tan=(x["tan"] if a1 == f and np.isfinite(x["tan"]) else 0.0),
                fuku=(x["fuku"].get(a1, 0.0) if a1 in x["fuku"] else 0.0),
                umaren=(x["umaren"] if {a1, a2} == {f, s} and np.isfinite(x["umaren"]) else 0.0),
                umatan=(x["umatan"] if (a1, a2) == (f, s) and np.isfinite(x["umatan"]) else 0.0),
                wide=(float(wp) if isinstance(wp, (int, float)) and np.isfinite(wp) else (0.0 if x["wide"] else np.nan)),
                wakuren=(x["wakuren"] if {wa1, wa2} == {x["w1"], x["w2"]} and np.isfinite(x["wakuren"]) else 0.0),
                sanpuku=(x["sanpuku"] if {a1, a2, a3} == {f, s, t} and np.isfinite(x["sanpuku"]) else 0.0),
                sanrentan=(x["sanrentan"] if (a1, a2, a3) == (f, s, t) and np.isfinite(x["sanrentan"]) else 0.0),
            ))
    df = pd.DataFrame(rows)
    return df, waku_check


def settle(df: pd.DataFrame, plan) -> pd.DataFrame:
    """plan: dict band->cols または list cols。ベット単位の DataFrame を返す。"""
    out = []
    for row in df.itertuples():
        cols = plan[row.band] if isinstance(plan, dict) else plan
        for c in cols:
            ret = getattr(row, c)
            if not np.isfinite(ret):
                continue  # ワイド払戻が無い日は当該ベットを除外
            out.append(dict(date=row.date, month=row.month, rid=row.rid, band=row.band, kind=c, ret=ret,
                            expect=BT[row.band][c]))
    return pd.DataFrame(out)


def report(name: str, b: pd.DataFrame, races: int):
    cost = np.full(len(b), 100.0)
    lo, hi, p = block_boot(cost, b.ret.values, b.date.values, reps=3000)
    roi = 100 * b.ret.sum() / cost.sum()
    exp = b.expect.mean()
    top = np.sort(b.ret.values)[::-1]
    print(f"\n=== {name} ===")
    print(f"  レース {races} / ベット {len(b)} ({len(b)/races:.1f}点/R) / 投資 ¥{int(cost.sum()):,} / 払戻 ¥{int(b.ret.sum()):,} / 収支 ¥{int(b.ret.sum()-cost.sum()):+,}")
    print(f"  ROI {roi:.1f}%  CI95 [{lo:.1f}, {hi:.1f}]  P(>100)={p:.3f}  | 期待(2023-25表の加重) {exp:.1f}%  差 {roi-exp:+.1f}pt")
    print(f"  何か当たったレース {100*b.groupby('rid').ret.max().gt(0).mean():.1f}%  | 上位10払戻を除くと {100*(b.ret.sum()-top[:10].sum())/cost.sum():.1f}%  最大払戻 ¥{int(top[0]):,}")
    print("  月別: " + " | ".join(f"{m}: {100*g.ret.mean()/100:.1f}% ({len(g)})" for m, g in b.groupby("month")))
    print("  券種別:")
    for k, g in b.groupby("kind"):
        print(f"    {JP[k]:<4} n={len(g):>5} 的中 {100*(g.ret>0).mean():5.1f}%  ROI {g.ret.mean():6.1f}%  期待 {g.expect.mean():6.1f}%  収支 ¥{int(g.ret.sum()-100*len(g)):+,}")
    print("  帯別:")
    for k, g in b.groupby("band"):
        print(f"    {k:<6} n={len(g):>5} ROI {g.ret.mean():6.1f}%  期待 {g.expect.mean():6.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--until", default="20260831")
    ap.add_argument("--since", default="20260101")
    a = ap.parse_args()
    df, wc = load(a.until, a.since)
    print(f"母集団: {len(df)}R / {df.date.nunique()}開催日 {df.date.min()}〜{df.date.max()} (bundle は 2026-04-18 から)")
    print(f"枠番割当の検算: 上位3着 {wc[1]}/{wc[0]} 一致 ({100*wc[1]/max(wc[0],1):.1f}%)")
    print(f"ワイド払戻あり: {int(df.wide.notna().sum())}R / 無し {int(df.wide.isna().sum())}R")
    print("◎オッズ帯の分布: " + ", ".join(f"{k}:{v}" for k, v in df.band.value_counts().reindex([b[2] for b in BANDS]).items()))
    A = settle(df, ARM_A); B = settle(df, ARM_B)
    report("Arm A: 90以上セル全採用 (帯別券種セット)", A, len(df))
    report("Arm B: 馬単1点 + ワイド1点 + 三連単1点", B, len(df))
    report("参照: 馬単1点のみ", settle(df, ["umatan"]), len(df))
    report("参照: ◎複勝1点のみ (null policy)", settle(df, ["fuku"]), len(df))
    print("\n=== 参照: 各券種 1 点 (全レース) ===")
    for c in COLS:
        s = settle(df, [c])
        cost = np.full(len(s), 100.0); lo, hi, _ = block_boot(cost, s.ret.values, s.date.values, reps=1500)
        print(f"  {JP[c]:<4} n={len(s):>4} 的中 {100*(s.ret>0).mean():5.1f}%  ROI {s.ret.mean():6.1f}% [{lo:5.1f},{hi:5.1f}]  期待 {s.expect.mean():6.1f}%")


if __name__ == "__main__":
    main()
