# -*- coding: utf-8 -*-
"""
hosei_cutoff_probe.py — hosei エクスポートの 前走補正 がどこで切れているかを特定する
================================================================================
TARGET から出した H_20260104-20260913.csv の 前走補正 充足率は
  1月 83% / 5月 92% / 6月 74% / 7月 38% / 8月 19% / 9月 9%
と直近ほど落ちる。仮説は「TARGET 側で 補正タイムが計算済みなのは ある日付まで」で、
前走がその日付より後の馬は 前走補正 が空になる、というもの。

各馬の前走日を kekka から復元し、「前走日 × 前走補正の有無」を数えて切れ目を出す。
ユーザーに TARGET 側で何をしてもらうかを正確に言うための調査。

実行: python -m analysis.hosei_cutoff_probe
"""
from __future__ import annotations
import glob
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
HOSEI_2026 = BASE / "data/hosei/H_20260104-20260913.csv"
KEKKA_MASTER = Path(r"E:\競馬過去走データ\kekka_20130105-20251228_v2.csv")


def main() -> None:
    h = pd.read_csv(HOSEI_2026, encoding="cp932", low_memory=False,
                    dtype={"レースID(新)": str})
    h["rid18"] = h["レースID(新)"].astype(str).str.strip().str.zfill(18)
    h["has"] = pd.to_numeric(h["前走補正"], errors="coerce").notna()
    print(f"hosei 2026: {len(h):,} 行 / 前走補正あり {100*h.has.mean():.1f}%")

    # 馬ごとの出走履歴 (2013-2025 マスター + 2026 週次 kekka)
    hist = []
    m = pd.read_csv(KEKKA_MASTER, encoding="cp932", low_memory=False,
                    usecols=["日付", "馬名", "レースID(新)"])
    m["rid18"] = m["レースID(新)"].astype(str).str.strip().str.zfill(18)
    m["date"] = m["rid18"].str[:8]
    hist.append(m[["馬名", "date", "rid18"]])
    for f in sorted(glob.glob(str(BASE / "data/kekka/2026*.csv"))):
        k = pd.read_csv(f, encoding="cp932", low_memory=False)
        if "馬名" not in k.columns:
            continue
        k["rid18"] = k["レースID(新)"].astype(str).str.strip().str.zfill(18)
        k["date"] = k["rid18"].str[:8]
        hist.append(k[["馬名", "date", "rid18"]])
    hist = pd.concat(hist, ignore_index=True).dropna(subset=["馬名"])
    hist["馬名"] = hist["馬名"].astype(str).str.strip()
    hist = hist.drop_duplicates(subset=["rid18"]).sort_values(["馬名", "date"])
    hist["prev_date"] = hist.groupby("馬名")["date"].shift(1)
    print(f"出走履歴: {len(hist):,} 行 / 前走日が取れた {100*hist.prev_date.notna().mean():.1f}%")

    d = h.merge(hist[["rid18", "prev_date"]], on="rid18", how="left")
    d = d.dropna(subset=["prev_date"])
    print(f"突合できた 2026 出走: {len(d):,}\n")

    print("=== 前走がいつなら 前走補正 が入っているか (前走の月別) ===")
    g = d.assign(pm=d.prev_date.str[:6]).groupby("pm")["has"].agg(["size", "mean"])
    g = g[g["size"] >= 30]
    for pm, r in g.iterrows():
        bar = "#" * int(round(r["mean"] * 40))
        print(f"  前走 {pm}  n={int(r['size']):>5}  前走補正あり {100*r['mean']:>5.1f}%  {bar}")

    print("\n=== 切れ目 (前走日の半月単位) ===")
    d["half"] = d.prev_date.str[:6] + np.where(d.prev_date.str[6:8].astype(int) <= 15, "a", "b")
    g2 = d[d.prev_date >= "20260401"].groupby("half")["has"].agg(["size", "mean"])
    for hf, r in g2.iterrows():
        if r["size"] < 20:
            continue
        print(f"  前走 {hf}  n={int(r['size']):>5}  前走補正あり {100*r['mean']:>5.1f}%")

    ok = d[d.has]
    if len(ok):
        print(f"\n  前走補正が入っている行の『前走日』の最大値 = {ok.prev_date.max()}")
        q = ok.prev_date.quantile([0.99, 0.999]) if ok.prev_date.dtype != object else None
    ng = d[(~d.has) & (d.prev_date >= "20260101")]
    print(f"  前走補正が空の行の『前走日』の中央値   = "
          f"{ng.prev_date.median() if len(ng) else 'n/a'}")
    print("\n→ ある日付以降に前走がある馬だけ空なら、TARGET 側の補正タイムが"
          "その日付までしか作られていない。")


if __name__ == "__main__":
    main()
