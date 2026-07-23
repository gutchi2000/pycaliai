# -*- coding: utf-8 -*-
"""
win5_merge_analyze.py — data/win5/ の年別+一括CSVを全結合し top-N ボックスを再検証
==================================================================================
2011-2025 の WIN5 を統合(~772回)。positional読み(310列)、日付でdedup。
top-N 人気ボックス(hit=全5勝ち馬人気≤N, 払戻=配当, コスト=N^5×100)の
ROI + bootstrap CI + 時系列OOS を、標本倍増で再計算。
"""
from __future__ import annotations
import glob
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(r"E:\PyCaLiAI")
DIR = BASE / "data/win5"


def log(m): print(m, flush=True)


def load_all() -> pd.DataFrame:
    frames = []
    for f in sorted(glob.glob(str(DIR / "*.csv")) + glob.glob(str(DIR / "*.CSV"))):
        if Path(f).name.startswith("_"):
            continue
        try:
            df = pd.read_csv(f, encoding="cp932", header=None, skiprows=1, low_memory=False)
        except Exception as e:
            log(f"  SKIP {Path(f).name}: {e}"); continue
        log(f"  {Path(f).name}: rows={len(df)} cols={df.shape[1]}")
        if df.shape[1] < 270:
            log(f"    !! cols<270 → 列構造違い、除外"); continue
        frames.append(df)
    allc = pd.concat(frames, ignore_index=True)
    return allc


def numi(d, i):
    return pd.to_numeric(d[i].astype(str).str.replace(",", "", regex=False)
                         .str.replace("円", "", regex=False), errors="coerce")


def main():
    log("[load] data/win5/*.csv")
    d = load_all()
    # 日付(pos2 日付S)で dedup
    d["_date"] = d[2].astype(str).str.strip()
    d = d[d["_date"].str.match(r"\d{4}") == True].copy()
    d = d.drop_duplicates("_date").reset_index(drop=True)
    year = pd.to_numeric(d["_date"].str[:4], errors="coerce")
    log(f"  結合後ユニーク回数={len(d)}  期間={int(year.min())}-{int(year.max())}")
    log(f"  年別件数:\n{year.value_counts().sort_index().to_string()}")

    pay = numi(d, 16); hit = numi(d, 17); sold = numi(d, 14); sales = sold * 100.0
    R = (pay * hit) / sales
    N = pd.concat([numi(d, 58 + j) for j in range(5)], axis=1); N.columns = range(5)
    maxn = N.max(axis=1); valid = N.notna().all(axis=1)
    rng = np.random.default_rng(42)
    log(f"\n  群衆平均R (的中>0) mean={R[hit>0].mean():.3f}")

    def cells(Nn, mask):
        m = valid & mask; cost = Nn ** 5 * 100.0
        ret = np.where((maxn <= Nn) & m, pay, 0.0); idx = m[m].index.to_numpy()
        return ret[idx], cost, int(len(idx))

    def roi(Nn, mask):
        ret, cost, n = cells(Nn, mask); return (ret.sum() / (cost * n) if n else 0.0), n, int((ret > 0).sum())

    def ci(Nn, mask):
        ret, cost, n = cells(Nn, mask)
        if n < 3: return None
        b = [ret[rng.integers(0, n, n)].sum() / (cost * n) for _ in range(4000)]
        return np.percentile(b, [2.5, 97.5])

    log("\n=== top-N 人気ボックス 全期間 (~772回) ===")
    allm = pd.Series(True, index=d.index)
    for Nn in [1, 2, 3, 4, 5]:
        r, n, h = roi(Nn, allm); c = ci(Nn, allm)
        cis = f"CI95[{c[0]:.3f},{c[1]:.3f}]" if c is not None else ""
        star = " ★CI>1" if (c is not None and c[0] > 1) else ""
        log(f"  N={Nn} ({Nn**5:4d}点): ROI={r:.3f} 的中{h:3d}/{n} {cis}{star}")

    log("\n=== N=2 の時系列OOS (3分割) ===")
    for lab, mask in [("2011-2015", year <= 2015), ("2016-2020", (year >= 2016) & (year <= 2020)),
                      ("2021-2025", year >= 2021)]:
        r, n, h = roi(2, mask); c = ci(2, mask)
        cis = f"CI95[{c[0]:.3f},{c[1]:.3f}]" if c is not None else ""
        log(f"  N=2 {lab}: ROI={r:.3f} 的中{h}/{n} {cis}")

    log("\n=== N=2 前半(≤2017)/後半(≥2018) ===")
    for lab, mask in [("≤2017", year <= 2017), ("≥2018", year >= 2018)]:
        r, n, h = roi(2, mask); c = ci(2, mask)
        cis = f"CI95[{c[0]:.3f},{c[1]:.3f}]" if c is not None else ""
        log(f"  N=2 {lab}: ROI={r:.3f} 的中{h}/{n} {cis}")


if __name__ == "__main__":
    main()
