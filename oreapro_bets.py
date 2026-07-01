"""
oreapro_bets.py
===============
netkeiba「俺プロ」用の買い目ジェネレータ。
検証済み戦略 (馬単3点 + 三連複4点, prob-first, gap ゲート) を bundle.json から実買い目化。

俺プロ 4段の制約に合わせた金額調整:
  - 週5R以上 かつ 週5万以上、1Rにつき1万まで
  - 7点を均等割 (検証は 1/7 等分) → 1点1,400円 × 7 = 9,800円/R
  - 6R/週 × 9,800 = 58,800円/週 ... 5R以上 & 5万以上を満たす (推奨)
  - ROI battle なので的中率は不問。狙いは ROI 上側裾。

ゲート: gap = (PL勝率 top1 - top2) > 0.24 のレース (最も本命が抜けた上位~10%) は買わない。
        (2024 q90 で算出し 2025 でも汎化を確認した唯一の頑健ゲート)

実行: python oreapro_bets.py                         # 最新 bundle
      python oreapro_bets.py --date 20260614
      python oreapro_bets.py --stake 1400 --pick 6
"""
from __future__ import annotations
import argparse, io, json, sys
from itertools import combinations
from pathlib import Path
import numpy as np
import pl_probs as PL

try:
    if not isinstance(sys.stdout, io.TextIOWrapper) or (sys.stdout.encoding or "").lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
except Exception:
    pass

BASE = Path(__file__).parent
GAP_GATE = 0.24       # gap>これ は本命堅すぎ -> 買わない
K_UM, K_SP = 3, 4


def latest_bundle():
    files = sorted((BASE / "reports/cowork_input").glob("*_bundle.json"))
    return files[-1] if files else None


def race_bets(horses, stake):
    """1レースの ai_score から 馬単top3 / 三連複top4 (prob-first) を返す."""
    hs = [h for h in horses if isinstance(h.get("ai_score"), (int, float)) and h["ai_score"] == h["ai_score"]]
    if len(hs) < 3:
        return None
    uma = np.array([int(h["umaban"]) for h in hs])
    sc = np.array([float(h["ai_score"]) for h in hs])
    w = PL.pl_weights(sc)
    p = w / w.sum()
    ps = np.sort(p)[::-1]
    gap = float(ps[0] - ps[1])
    n = len(w)

    # 馬単 top3 (順序付き i->j)
    pairs = [(i, j, PL.p_umatan(w, i, j)) for i in range(n) for j in range(n) if i != j]
    pairs.sort(key=lambda x: -x[2])
    umatan = [(int(uma[i]), int(uma[j])) for i, j, _ in pairs[:K_UM]]

    # 三連複 top4
    tri = [(c, PL.p_sanrenpuku(w, *c)) for c in combinations(range(n), 3)]
    tri.sort(key=lambda x: -x[1])
    sanpuku = [tuple(sorted(int(uma[x]) for x in c)) for c, _ in tri[:K_SP]]

    return {"gap": gap, "n": n, "umatan": umatan, "sanpuku": sanpuku, "stake": stake}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="bundle 日付 YYYYMMDD")
    ap.add_argument("--stake", type=int, default=1400, help="1点あたり金額 (default 1400, 7点で9800/R)")
    ap.add_argument("--pick", type=int, default=6, help="週に買うレース数 (>=5)")
    args = ap.parse_args()

    bpath = (BASE / f"reports/cowork_input/{args.date}_bundle.json") if args.date else latest_bundle()
    if not bpath or not bpath.exists():
        print(f"bundle 無し: {bpath}"); return
    d = json.load(open(bpath, encoding="utf-8"))
    races = d["races"]
    print(f"[bundle] {bpath.name}  model={d.get('model')}  {len(races)}R")
    print(f"[gate]   gap>{GAP_GATE} の本命堅レースは除外  /  1点={args.stake}円 × 7点 = {args.stake*7}円/R")

    ok, chalk = [], 0
    for r in races:
        rb = race_bets(r.get("horses", []), args.stake)
        if rb is None:
            continue
        meta = r.get("race_meta", {})
        name = meta.get("race_name") or meta.get("place") or ""
        rb["label"] = f"{r['race_id']} {name}".strip()
        if rb["gap"] > GAP_GATE:
            chalk += 1
            continue
        ok.append(rb)

    # 本命堅(gap>GATE)を除けば検証上の優劣は無い → レース順に並べ、★は先頭pick Rの暫定
    ok.sort(key=lambda x: x["label"])
    print(f"\nゲート通過 {len(ok)}R / 本命堅で除外 {chalk}R")
    print("（★は暫定。ゲート通過レース間に検証上の優劣なし→好きな {0}R を選んでOK）".format(args.pick))
    print("=" * 72)
    week_total = 0
    for idx, rb in enumerate(ok):
        flag = "★買い" if idx < args.pick else "  控え"
        um = ", ".join(f"{i}→{j}" for i, j in rb["umatan"])
        sp = ", ".join("-".join(map(str, c)) for c in rb["sanpuku"])
        if idx < args.pick:
            week_total += args.stake * 7
        print(f"[{flag}] {rb['label']}  (gap={rb['gap']:.3f}, {rb['n']}頭)")
        print(f"        馬単3点 @{args.stake}: {um}")
        print(f"        三連複4点 @{args.stake}: {sp}")

    picked = min(args.pick, len(ok))
    print("=" * 72)
    print(f"[週次サマリ] 買い {picked}R × {args.stake*7}円 = {week_total:,}円")
    cond_r = "OK" if picked >= 5 else f"NG (5R未満)"
    cond_y = "OK" if week_total >= 50000 else f"NG (5万未満)"
    print(f"   制約: 週5R以上 -> {picked}R [{cond_r}]   /   週5万以上 -> {week_total:,}円 [{cond_y}]")
    if picked < 5 or week_total < 50000:
        print("   ※ 不足分は --pick を増やすか 1点金額を上げて調整")


if __name__ == "__main__":
    main()
