# -*- coding: utf-8 -*-
"""augment_joint_substrate.py — 既存 mkt_horses.parquet に確定複勝配当 fpay を高速マージ
(オッズCSV再読込なし。load_payouts のみ ~10s)。fpay = この馬が複勝圏なら確定複勝配当, else NaN。"""
import sys, io
import numpy as np, pandas as pd
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
sys.path.insert(0, "E:/PyCaLiAI")
from backtest_pl_ev import load_payouts

OUT = Path("E:/PyCaLiAI/data/_joint")
H = pd.read_parquet(OUT / "mkt_horses.parquet")
pays = load_payouts()

fmap = {}  # (rid, ban) -> fuku payout
for rid, po in pays.items():
    for bn, kk in [(po["win"], "fuku_win"), (po["plc"], "fuku_plc"), (po["sho"], "fuku_sho")]:
        v = po.get(kk)
        if v == v: fmap[(rid, int(bn))] = float(v)

H["fpay"] = [fmap.get((r, b), np.nan) for r, b in zip(H.rid, H.ban)]
H.to_parquet(OUT / "mkt_horses.parquet")
cov = H[H.top3 == 1].fpay.notna().mean()
print(f"[ok] fpay マージ rows={len(H):,}  複勝圏fpay被覆={cov:.3f}  "
      f"中央複勝配当(top3)={H[H.top3==1].fpay.median():.0f}")
