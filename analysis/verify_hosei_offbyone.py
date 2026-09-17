# -*- coding: utf-8 -*-
"""
verify_hosei_offbyone.py — serve の prev_hosei が「前々走」になっていないかを実データで確認
==========================================================================================
学習側 (build_master_v2.py:68-70):
    prev_hosei := H_master[ 今走18桁 ].前走補正        = 前走の補正タイム
serve 側 (make_weekly_hosei.py:238,276 + load_hosei_lookup):
    prev_hosei := H_lookup[ 前走18桁 ].前走補正        = 前走のさらに前走 = 前々走

この2つが一致するのは H[X].前走補正 == H[prev(X)].前走補正 のときだけで、
正しい恒等式は H[X].前走補正 == H[prev(X)].補正 のはず。どちらが成り立つかを
実データ (馬名で前走を辿れる kekka マスター × H マスター) で数える。

実行: python -m analysis.verify_hosei_offbyone
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
H_MASTER = BASE / "data/hosei/H_20130105-20251228.csv"
KEKKA = Path(r"E:\競馬過去走データ\kekka_20130105-20251228_v2.csv")


def main() -> None:
    h = pd.read_csv(H_MASTER, encoding="cp932", dtype={"レースID(新)": str})
    h["rid18"] = h["レースID(新)"].astype(str).str.strip().str.zfill(18)
    for c in ("補正", "前走補正", "補9", "前走補9"):
        h[c] = pd.to_numeric(h[c], errors="coerce")
    hm = h.set_index("rid18")[["補正", "前走補正", "補9", "前走補9"]]
    print(f"H マスター: {len(h):,} 行  (補正 充足 {h.補正.notna().mean():.3f} / "
          f"前走補正 充足 {h.前走補正.notna().mean():.3f})")

    k = pd.read_csv(KEKKA, encoding="cp932", low_memory=False,
                    usecols=["日付", "馬名", "レースID(新)"])
    k["rid18"] = k["レースID(新)"].astype(str).str.strip().str.zfill(18)
    k["d"] = pd.to_numeric(k["日付"], errors="coerce")
    k = k.dropna(subset=["d", "馬名"]).sort_values(["馬名", "d"])
    k["prev18"] = k.groupby("馬名")["rid18"].shift(1)
    k = k.dropna(subset=["prev18"])
    print(f"前走を辿れた出走: {len(k):,}")

    k = k.join(hm, on="rid18").rename(columns={"補正": "cur_hosei", "前走補正": "cur_prev_hosei",
                                               "補9": "cur_h9", "前走補9": "cur_prev_h9"})
    k = k.join(hm.rename(columns={"補正": "p_hosei", "前走補正": "p_prev_hosei",
                                  "補9": "p_h9", "前走補9": "p_prev_h9"}), on="prev18")
    t = k.dropna(subset=["cur_prev_hosei", "p_hosei", "p_prev_hosei"])
    print(f"3値そろった行: {len(t):,}\n")

    a = float((t.cur_prev_hosei == t.p_hosei).mean())
    b = float((t.cur_prev_hosei == t.p_prev_hosei).mean())
    print("=== どちらの恒等式が成り立つか ===")
    print(f"  H[今走].前走補正 == H[前走].補正      : {100*a:6.2f}%   ← serve が使うべき列")
    print(f"  H[今走].前走補正 == H[前走].前走補正   : {100*b:6.2f}%   ← serve が実際に使っている列")

    d_ok = (t.cur_prev_hosei - t.p_hosei).abs()
    d_ng = (t.cur_prev_hosei - t.p_prev_hosei).abs()
    print(f"\n  正しい列との平均絶対差: {d_ok.mean():.3f}")
    print(f"  現状の列との平均絶対差: {d_ng.mean():.3f}  "
          f"(= serve が学習時と違う値を入れている大きさ)")
    print(f"  現状の列がズレている行の割合: {100*float((d_ng > 0).mean()):.2f}%")

    if a > 0.95 and b < 0.5:
        print("\n判定: **off-by-one 確定**。serve は『前々走の補正タイム』を "
              "prev_hosei として渡している。")
        print("  修正: make_weekly_hosei.load_hosei_lookup の usecols を "
              "前走補9/前走補正 → 補9/補正 に変える。")
        print("  ただし週次生成の H_*.csv には『補正』列が無いため、2026 以降の前走を")
        print("  引くには TARGET から hosei マスターを 2026 まで再エクスポートする必要がある。")
    else:
        print("\n判定: off-by-one とは断定できない。上の一致率を見て判断すること。")


if __name__ == "__main__":
    main()
