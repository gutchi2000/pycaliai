# -*- coding: utf-8 -*-
"""
hit_rate_frontier.py — ◎複勝を「信頼度」で選別したときの 的中率×回収率×カバレッジ
=================================================================================
問い: 「的中70% / 回収90% は現実的か?」
方法: ev_grid のスコアキャッシュ(v6, leak-safe)を使い、各レースの ◎(=PL最確馬)について
  - 複勝の的中(◎が3着以内か) と 実複勝配当(kekka, リーク無し)
  - 信頼度シグナル = ◎の p_win(PL単勝確率)  ※発走前に出る・単調
を集計。信頼度でレースを降順に並べ、上位 cover% だけ買ったときの
  的中率 / 回収率 / 信頼度閾値 を valid(2023) と test(2024-25) 別に出す。
  → valid と test で同じ操作点が一致すれば「運でなく本物の選別」。
出力: 標準出力テーブル + reports/_hit_rate_frontier.parquet
実行: PYTHONUTF8=1 ./venv311/Scripts/python.exe analysis/hit_rate_frontier.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(r"E:\PyCaLiAI"); sys.path.insert(0, str(BASE))
import pl_probs as PL
import backtest_pl_ev as BT

COL_RID = "レースID(新/馬番無)"; COL_BAN = "馬番"; COL_JYUN = "着順"
SCORE_CACHE = BASE / "data/_ev_grid_scores.parquet"


def main():
    sc = pd.read_parquet(SCORE_CACHE)
    sc[COL_JYUN] = pd.to_numeric(sc[COL_JYUN], errors="coerce")
    pays = BT.load_payouts()
    print(f"[load] scores rows={len(sc):,} races={sc[COL_RID].nunique():,}  payouts={len(pays):,}")

    rows = []
    miss = 0
    for rid, g in sc.groupby(COL_RID, sort=False):
        if len(g) < 3:
            continue
        rid_s = str(int(rid)) if isinstance(rid, (int, np.integer)) else str(rid)
        po = pays.get(rid_s)
        if po is None:
            miss += 1; continue
        g = g.sort_values(COL_BAN)
        ban = g[COL_BAN].astype(int).values
        w = PL.pl_weights(g["_score"].values)
        pw = np.asarray(PL.all_tansho(w), float); pw = pw / pw.sum()
        top = int(np.argmax(pw)); top_ban = int(ban[top]); conf = float(pw[top])
        hit, pay = 0, 0.0
        for pos, key in [("win", "fuku_win"), ("plc", "fuku_plc"), ("sho", "fuku_sho")]:
            if po.get(pos) == top_ban:
                hit = 1
                try: pay = float(po.get(key) or 0.0)
                except Exception: pay = 0.0
                break
        yr = int(g["year"].iloc[0]); spl = str(g["split"].iloc[0])
        period = "valid" if spl == "valid" else ("test" if yr in (2024, 2025) else "other")
        rows.append((period, conf, hit, pay))

    df = pd.DataFrame(rows, columns=["period", "conf", "hit", "pay"])
    print(f"[races used] {len(df):,}  (payout欠 {miss})")
    df.to_parquet(BASE / "reports/_hit_rate_frontier.parquet")

    COVERS = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5]
    for period in ["valid", "test"]:
        d = df[df.period == period].sort_values("conf", ascending=False).reset_index(drop=True)
        N = len(d)
        print(f"\n=== {period}  ◎複勝 信頼度選別フロンティア  (全{N:,}R) ===")
        print(f"{'買う割合':>8} {'R数':>6} {'的中率':>7} {'回収率':>7} {'信頼度≥':>8}")
        for cov in COVERS:
            k = max(1, int(round(N * cov / 100)))
            sub = d.iloc[:k]
            hit = sub.hit.mean() * 100
            roi = sub.pay.sum() / (len(sub) * 100) * 100
            thr = float(sub.conf.iloc[-1])
            print(f"{cov:>7}% {k:>6,} {hit:>6.1f}% {roi:>6.1f}% {thr:>8.3f}")


if __name__ == "__main__":
    main()
