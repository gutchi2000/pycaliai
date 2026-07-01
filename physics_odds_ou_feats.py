# -*- coding: utf-8 -*-
"""
physics_odds_ou_feats.py — オッズ・プールの確率過程（econophysics）特徴
========================================================================
発想: 賭けプールのオッズは資金流入で動く確率過程。確定オッズが効率的でも、
  「均衡へ緩和する過程の形」には情報（informed money）が残りうる。
  log オッズ系列を Ornstein-Uhlenbeck 過程
      dx = θ(μ - x) dt + σ dW
  とみなし、決定論的ドリフト（情報）と拡散ノイズ（流動性）を分離する。

★リーク防止（生命線）: 特徴は「当日09:00以前に確定した前売りスナップ」のみから算出。
  build_odds_traj_feats.py と同一の区分1 & (mmdd<race or hhmm<=900) フィルタを継承し、
  検査で違反0を確認する（>0 で FAIL）。9時以降・確定オッズは1列も使わない。

特徴（log単勝オッズ x_1..x_k, 時刻順。x小=買われている）:
  ou_n        : 9時前スナップ数 k
  ou_net      : x_k - x_1（符号付き総移動。負=短縮=買われた）
  ou_pathlen  : Σ|Δx|（全変動＝経路長＝荒さ）
  ou_s2n      : |x_k-x_1| / (Σ|Δx|)  ∈[0,1]  1=単調クリーン(情報), →0=往復(ノイズ)
  ou_informed : -ou_net * ou_s2n   （符号付きクリーン資金。正=きれいに買われた＝informed）
  ou_termvel  : x_k - x_{k-1}       （締切直前の速度＝モメンタム）
  ou_termaccel: 末2ステップの加速（k>=3, 流入が加速しているか）
  ou_theta    : AR(1)平均回帰速度 θ（k>=4）
  ou_sigma    : 拡散ノイズ σ（k>=4）
  ou_equilgap : x_k - μ̂（9時時点の均衡乖離→9時以降の継続ドリフトを予測, k>=4）
  ou_has      : k>=2 フラグ

出力: data/odds_ou_feats.parquet / reports/odds_ou_leakcheck.json
実行: PYTHONUTF8=1 python physics_odds_ou_feats.py
"""
from __future__ import annotations
import glob, io, json, re, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

BASE = Path(__file__).parent
ODIR = BASE / "data/Time _series_odds"
OUT_PARQUET = BASE / "data/odds_ou_feats.parquet"
OUT_LEAK = BASE / "reports/odds_ou_leakcheck.json"
LEAK_SAMPLE_EVERY = 40


def _rid16(x):
    return re.sub(r"\D", "", str(x))[:16]


def _ou_features(series: np.ndarray) -> dict:
    """前売り単勝オッズ系列(>1) から OU/econophysics 特徴を算出。"""
    series = series[(~np.isnan(series)) & (series > 1.0)]
    k = len(series)
    rec = dict(ou_n=k, ou_has=1 if k >= 2 else 0,
               ou_net=np.nan, ou_pathlen=np.nan, ou_s2n=np.nan, ou_informed=np.nan,
               ou_termvel=np.nan, ou_termaccel=np.nan,
               ou_theta=np.nan, ou_sigma=np.nan, ou_equilgap=np.nan,
               ou_lastlog=np.nan, ou_firstlog=np.nan)
    if k < 1:
        return rec
    x = np.log(series)
    rec["ou_lastlog"] = float(x[-1])     # 9時直前の log単勝オッズ水準（leak-safe な人気代理）
    rec["ou_firstlog"] = float(x[0])
    if k < 2:
        return rec
    d = np.diff(x)
    net = float(x[-1] - x[0])
    pathlen = float(np.abs(d).sum())
    s2n = abs(net) / pathlen if pathlen > 1e-9 else 0.0
    rec["ou_net"] = net
    rec["ou_pathlen"] = pathlen
    rec["ou_s2n"] = float(s2n)
    rec["ou_informed"] = float(-net * s2n)         # 正=きれいに買われた
    rec["ou_termvel"] = float(d[-1])
    if k >= 3:
        rec["ou_termaccel"] = float(d[-1] - d[-2])
    if k >= 4:
        # AR(1): Δx_t = a + b·x_{t-1} + ε  →  θ=-b, μ=-a/b, σ=std(ε)
        x_prev = x[:-1]
        try:
            b, a = np.polyfit(x_prev, d, 1)          # slope b, intercept a
            theta = -b
            resid = d - (b * x_prev + a)
            sigma = float(np.std(resid))
            mu = (-a / b) if abs(b) > 1e-6 else np.nan
            rec["ou_theta"] = float(theta)
            rec["ou_sigma"] = sigma
            if np.isfinite(mu):
                rec["ou_equilgap"] = float(x[-1] - mu)
        except Exception:
            pass
    return rec


def main():
    files = sorted(glob.glob(str(ODIR / "TANPUK_*.csv")))
    print(f"[load] {len(files)} TANPUK files")
    parts = []
    tan_cols = None
    for f in files:
        tp = pd.read_csv(f, encoding="cp932", low_memory=False)
        cols = list(tp.columns); RID, KB, TM = cols[0], cols[1], cols[2]
        if tan_cols is None:
            tan_cols = {}
            for c in cols:
                m = re.match(r"^\s*(\d+)\s*単\s*$", str(c))
                if m:
                    tan_cols[int(m.group(1))] = c
        tp[KB] = pd.to_numeric(tp[KB], errors="coerce")
        tp = tp[tp[KB] == 1].copy()                  # ★前売りのみ
        tp["rid16"] = tp[RID].map(_rid16)
        tp = tp[tp["rid16"].str.len() == 16]
        tp[TM] = pd.to_numeric(tp[TM], errors="coerce")
        parts.append(tp[["rid16", TM] + list(tan_cols.values())])
    df = pd.concat(parts, ignore_index=True)
    TMc = [c for c in df.columns if c not in ["rid16"] + list(tan_cols.values())][0]
    df["mmdd"] = (df[TMc] // 10000).astype("Int64")
    df["hhmm"] = (df[TMc] % 10000).astype("Int64")
    print(f"  区分1 行数={len(df):,}  単勝列={len(tan_cols)}")

    recs = []; leak_checked = 0; leak_viol = 0; rc = 0
    cov_total = cov_has = 0
    for rid, g in df.groupby("rid16", sort=False):
        rmmdd = int(rid[4:8])
        pre = g[(g["mmdd"] < rmmdd) | ((g["mmdd"] == rmmdd) & (g["hhmm"] <= 900))].copy()
        if len(pre) == 0:
            continue
        pre = pre.sort_values(TMc)
        rc += 1
        if rc % LEAK_SAMPLE_EVERY == 0:
            leak_checked += len(pre)
            bad = (((pre["mmdd"] == rmmdd) & (pre["hhmm"] > 900)) | (pre["mmdd"] > rmmdd)).sum()
            leak_viol += int(bad)
        for b, c in tan_cols.items():
            series = pd.to_numeric(pre[c], errors="coerce").values
            rec = _ou_features(series)
            if rec["ou_n"] < 1:
                continue
            cov_total += 1
            if rec["ou_has"]:
                cov_has += 1
            rec["rid16"] = rid; rec["ban"] = b
            recs.append(rec)

    out = pd.DataFrame(recs)
    cols = ["rid16", "ban", "ou_n", "ou_has", "ou_net", "ou_pathlen", "ou_s2n",
            "ou_informed", "ou_termvel", "ou_termaccel", "ou_theta", "ou_sigma", "ou_equilgap",
            "ou_lastlog", "ou_firstlog"]
    out = out[cols]
    out.to_parquet(OUT_PARQUET, index=False)
    leak = {"leak_policy": "区分1 かつ (mmdd<race_mmdd or (mmdd==race & hhmm<=900)) のみ。9時以降/確定 不使用。",
            "leak_sample_every": LEAK_SAMPLE_EVERY, "leak_checked_snaps": leak_checked,
            "leak_violations": leak_viol, "PASS": bool(leak_viol == 0),
            "n_races_with_pre": rc, "n_horse_rows": len(out),
            "coverage_has(k>=2)/total": round(cov_has / cov_total * 100, 1) if cov_total else 0.0,
            "coverage_k>=4": round((out["ou_n"] >= 4).mean() * 100, 1) if len(out) else 0.0}
    OUT_LEAK.parent.mkdir(exist_ok=True)
    with open(OUT_LEAK, "w", encoding="utf-8") as f:
        json.dump(leak, f, indent=2, ensure_ascii=False)
    print(f"\n[リーク検査] 検査snap={leak_checked:,}  違反={leak_viol}  {'PASS' if leak_viol==0 else 'FAIL'}")
    print(f"  k>=2率={leak['coverage_has(k>=2)/total']}%  k>=4率={leak['coverage_k>=4']}%  馬行={len(out):,}")
    print("\n=== OU feature describe ===")
    print(out[["ou_n", "ou_net", "ou_s2n", "ou_informed", "ou_termvel", "ou_theta", "ou_equilgap"]]
          .describe().T[["mean", "std", "min", "max"]].to_string())
    print(f"[saved] {OUT_PARQUET}")
    return 0 if leak_viol == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
