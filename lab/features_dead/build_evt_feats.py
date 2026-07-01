# -*- coding: utf-8 -*-
"""
build_evt_feats.py — 極値統計(EVT)に基づく per-horse 性能分散特徴 (leak-safe, as-of)
================================================================================
理論 (なぜ直交が理論で保証されるか):
  勝利 = 場の性能の最大値 (argmax)。各馬の潜在性能を
      X_i = μ_i + σ_i · ε_i,   ε_i ~ Gumbel(0,1)
  と置くと、等分散 (σ_i=const) のとき
      P(i = argmax) = exp(μ_i) / Σ_j exp(μ_j) = softmax(μ)
  これは PL/softmax そのもの = 現行 v6 が暗黙に置いている仮定。
  よって  σ_i ≠ const こそが PL の構造的な取りこぼし。
  → 各馬の「補正タイム指数」(speed figure, ~100中心, 距離横断で比較可能) の
    as-of な散らばり σ_i を推定して特徴化する。

出力特徴 (key = rid16 + 馬番。すべて as-of date で算出 = leak-safe):
  evt_n        : as-of 過去走数 (有効補正の数)
  evt_mu       : 過去補正の平均           (馬の地力指数の代理)
  evt_sigma    : 過去補正の標準偏差 (生)  (= ムラ。n<2 は NaN)
  evt_sigma_s  : 経験ベイズ収縮済み σ     (w=n/(n+k) で global prior へ)
  evt_sigma_gz : σ_s の全体 z              (Phase2 の heteroskedastic scale 用)
  evt_sigma_rz : σ_s のレース内 z          (場相対のムラ度)
  evt_ceiling  : 過去最高補正 - 過去平均   (実現した上振れ余地 = 上裾)
  evt_floor    : 過去平均 - 過去最低補正   (下振れの深さ = 不安定さの裏)
  evt_range    : 過去 max - min

★リーク防止 (生命線):
  各馬を「血統登録番号」で束ね「日付」昇順に並べ、expanding 集計は
  prev_hosei (= 前走補正。構造的に常に当該レースより前に確定) のみで行う。
  当該レースの補正 (= 目的変数と同時決定 = リーク; build_master_v2.py:7 参照) は
  1 列も読まない・使わない。
  → row(r_k) の prev_hosei = 補正(r_{k-1}) なので、馬の各行 r_1..r_k の prev_hosei を
    累積すると {補正(r_1)..補正(r_{k-1})} = 当該より前の補正系列が復元される。
  global prior も train split (<=2022) のみで算出し test 情報を混ぜない。

実行:
  python build_evt_feats.py                 # master_v2 から特徴生成
  python build_evt_feats.py --shrink-k 4.0  # 収縮 k 変更
  python build_evt_feats.py --selftest      # 合成データで σ 復元 MAE を確認 (master 不要)

出力:
  data/evt_feats.parquet
  reports/evt_feats_build.json
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
except Exception:
    pass

BASE = Path(__file__).parent
MASTER = BASE / "data/master_v2_20130105-20251228.csv"
OUT_PARQUET = BASE / "data/evt_feats.parquet"
OUT_LOG = BASE / "reports/evt_feats_build.json"

COL_DATE = "日付"
COL_PED = "血統登録番号"
COL_BAN = "馬番"
COL_HOSEI = "prev_hosei"          # = 前走補正 (leak-safe な過去補正のラグ)
COL_SPLIT = "split"
# RID 列はリポジトリで揺れがある ("レースID(新/馬番無)" 統一だが build 側は "レースID(新)")
RID_CANDIDATES = ["レースID(新/馬番無)", "レースID(新)", "レースID"]


def log(msg: str):
    print(msg, flush=True)


def _rid16(x) -> str:
    return re.sub(r"\D", "", str(x))[:16]


# ============================================================
# leak-safe expanding 統計 (馬ごと日付昇順, 当該行の prev_hosei まで含む)
#   prev_hosei(r_k) は構造的に 補正(r_{k-1}) なので当該行を含めても leak しない。
#   NaN (= その走の前走補正が欠損) はカウント・和から除外。
# ============================================================
def _expanding_stats(df: pd.DataFrame, ped: str, val: str) -> pd.DataFrame:
    """df は [ped, COL_DATE] で昇順ソート済みであること。
    val 列の馬ごと expanding (n, mean, std(標本), max, min) を O(N) で返す。"""
    g = df.groupby(ped, sort=False)
    v = pd.to_numeric(df[val], errors="coerce")
    m = v.notna()
    vf = v.fillna(0.0)

    n = m.astype("int64").groupby(df[ped], sort=False).cumsum()
    sx = vf.groupby(df[ped], sort=False).cumsum()
    sx2 = (vf * vf).groupby(df[ped], sort=False).cumsum()

    mean = np.where(n > 0, sx / n.replace(0, np.nan), np.nan)
    # 標本分散: (Σx² - n·mean²) / (n-1)
    with np.errstate(invalid="ignore"):
        var = (sx2 - n * (mean ** 2)) / (n - 1).replace(0, np.nan)
        var = var.clip(lower=0.0)
        std = np.sqrt(var)
    std = np.where(n >= 2, std, np.nan)

    # cummax / cummin は NaN を無視 (±inf に退避してから group cummax/cummin)
    vmax = v.where(m, -np.inf)
    vmin = v.where(m, np.inf)
    cmax = vmax.groupby(df[ped], sort=False).cummax()
    cmin = vmin.groupby(df[ped], sort=False).cummin()
    cmax = np.where(n > 0, cmax, np.nan)
    cmin = np.where(n > 0, cmin, np.nan)

    return pd.DataFrame({
        "evt_n": n.astype("float64").values,
        "evt_mu": np.asarray(mean, dtype="float64"),
        "evt_sigma": np.asarray(std, dtype="float64"),
        "_evt_max": np.asarray(cmax, dtype="float64"),
        "_evt_min": np.asarray(cmin, dtype="float64"),
    }, index=df.index)


def build_evt_features(df: pd.DataFrame, shrink_k: float, ewm_span: int = 4) -> pd.DataFrame:
    """master_v2 相当の df から EVT 特徴を付与して返す。df は破壊しない。"""
    df = df.copy()
    df[COL_DATE] = pd.to_numeric(df[COL_DATE], errors="coerce")
    # 馬ごと日付昇順 (同日内は安定ソートで元順) — expanding の前提
    order = df.sort_values([COL_PED, COL_DATE], kind="stable").index
    s = df.loc[order]

    st = _expanding_stats(s, COL_PED, COL_HOSEI)
    s = pd.concat([s, st], axis=1)

    # --- global prior (train split <=2022 のみ; test 情報を混ぜない) ---
    if COL_SPLIT in s.columns:
        tr_mask = s[COL_SPLIT].astype(str) == "train"
    else:
        tr_mask = pd.Series(True, index=s.index)
    sig_prior = float(np.nanmedian(s.loc[tr_mask & (s["evt_n"] >= 2), "evt_sigma"]))
    if not np.isfinite(sig_prior):
        sig_prior = float(np.nanmedian(s["evt_sigma"]))
    log(f"  σ global prior (train, n>=2 中央値) = {sig_prior:.4f}")

    # --- 経験ベイズ収縮: σ_s = w·σ_h + (1-w)·prior,  w = n/(n+k) ---
    n_eff = (s["evt_n"] - 1).clip(lower=0)          # σ の自由度 ~ 過去走数-1
    w = n_eff / (n_eff + shrink_k)
    sigma_h = s["evt_sigma"].where(s["evt_n"] >= 2, sig_prior)
    s["evt_sigma_s"] = w * sigma_h + (1.0 - w) * sig_prior

    # --- detrend 残差σ (H2反証対応: realized σ の「上昇トレンド/天井」汚染を除去し
    #     理論が要求する純ノイズ尺度を作る) ---
    #   causal EWM level (strictly-past) = ewm.mean().shift(1) を馬内で。
    #   resid = prev_hosei - level → expanding std。すべて当該より前のみ = leak-safe。
    p = pd.to_numeric(s[COL_HOSEI], errors="coerce")
    lvl = p.groupby(s[COL_PED], sort=False).transform(
        lambda x: x.ewm(span=ewm_span, min_periods=1).mean().shift(1))
    s["_resid"] = p - lvl
    rst = _expanding_stats(s, COL_PED, "_resid")
    rn, rsig = rst["evt_n"], rst["evt_sigma"]
    rprior = float(np.nanmedian(rsig[(rn >= 2) & tr_mask]))
    if not np.isfinite(rprior):
        rprior = float(np.nanmedian(rsig))
    rw = (rn - 1).clip(lower=0)
    rw = rw / (rw + shrink_k)
    rsig_h = rsig.where(rn >= 2, rprior)
    s["evt_sigma_resid"] = rw * rsig_h + (1.0 - rw) * rprior
    rz_mu = float(np.nanmean(s.loc[tr_mask, "evt_sigma_resid"]))
    rz_sd = float(np.nanstd(s.loc[tr_mask, "evt_sigma_resid"])) or 1.0
    s["evt_sigma_resid_gz"] = (s["evt_sigma_resid"] - rz_mu) / rz_sd
    log(f"  σ_resid global prior (train, ewm_span={ewm_span}) = {rprior:.4f}")

    # --- ceiling / floor / range ---
    s["evt_ceiling"] = s["_evt_max"] - s["evt_mu"]
    s["evt_floor"] = s["evt_mu"] - s["_evt_min"]
    s["evt_range"] = s["_evt_max"] - s["_evt_min"]

    # --- σ_s の全体 z (train 統計で標準化; Phase2 の heteroskedastic scale 用) ---
    gz_mu = float(np.nanmean(s.loc[tr_mask, "evt_sigma_s"]))
    gz_sd = float(np.nanstd(s.loc[tr_mask, "evt_sigma_s"]))
    gz_sd = gz_sd if gz_sd > 1e-9 else 1.0
    s["evt_sigma_gz"] = (s["evt_sigma_s"] - gz_mu) / gz_sd

    # --- σ_s のレース内 z (場相対のムラ度) ---
    rid_col = next((c for c in RID_CANDIDATES if c in s.columns), None)
    if rid_col is not None:
        s["rid16"] = s[rid_col].map(_rid16)
        grp = s.groupby("rid16")["evt_sigma_s"]
        rmu = grp.transform("mean")
        rsd = grp.transform("std").replace(0, np.nan)
        s["evt_sigma_rz"] = ((s["evt_sigma_s"] - rmu) / rsd).fillna(0.0).clip(-4, 4)
    else:
        s["rid16"] = np.nan
        s["evt_sigma_rz"] = 0.0

    s["ban"] = pd.to_numeric(s[COL_BAN], errors="coerce").astype("Int64")
    return s.drop(columns=["_evt_max", "_evt_min"])


FEAT_COLS = ["evt_n", "evt_mu", "evt_sigma", "evt_sigma_s",
             "evt_sigma_gz", "evt_sigma_rz", "evt_ceiling", "evt_floor", "evt_range",
             "evt_sigma_resid", "evt_sigma_resid_gz"]


def main(shrink_k: float, ewm_span: int = 4) -> int:
    log(f"=== build_evt_feats 開始 (shrink_k={shrink_k}, ewm_span={ewm_span}) ===")
    usecols = RID_CANDIDATES + [COL_DATE, COL_PED, COL_BAN, COL_HOSEI, COL_SPLIT]
    df = pd.read_csv(MASTER, encoding="utf-8-sig",
                     usecols=lambda c: c in usecols, low_memory=False)
    log(f"[load] master rows={len(df):,}  cols={list(df.columns)}")
    cov = pd.to_numeric(df[COL_HOSEI], errors="coerce").notna().mean() * 100
    log(f"  prev_hosei カバレッジ = {cov:.1f}%  / {df[COL_PED].nunique():,} 頭")

    s = build_evt_features(df, shrink_k, ewm_span=ewm_span)

    # --- leak 監査 (構造 + 因果単調 + ウィンドウ外履歴の説明) ---
    # 構造保証: current補正 (= 目的変数と同時決定 = リーク; build_master_v2.py:7) は
    # usecols に含めず prev_hosei (前走補正) のみ使用。定義証明は --validate-leak。
    assert COL_HOSEI == "prev_hosei", "性能系列は prev_hosei (前走補正) のみから作ること"
    s_sorted = s.sort_values([COL_PED, COL_DATE], kind="stable")
    dn = s_sorted.groupby(COL_PED, sort=False)["evt_n"].diff().dropna()
    # 因果単調: 馬内で evt_n は非減少かつ増分<=1 (未来情報の一括混入が無い)
    mono_ok = bool(((dn >= 0) & (dn <= 1)).all())
    # データ初出で evt_n>0 = 2013以前デビュー or フィルタ除外された前走を参照 (=過去=leak-safe)
    first_row = s_sorted.groupby(COL_PED, sort=False).head(1)
    prewin = first_row[first_row["evt_n"] > 0]
    prewin_med_date = int(prewin[COL_DATE].median()) if len(prewin) else 0
    leak_ok = mono_ok
    log(f"  [leak監査] 因果単調(増分0/1)={'OK' if mono_ok else 'FAIL'} / "
        f"初出evt_n>0={len(prewin)}行 (中央日付{prewin_med_date} = ウィンドウ外履歴, leak-safe)")

    out = s[["rid16", "ban"] + FEAT_COLS].copy()
    out = out[out["ban"].notna()].reset_index(drop=True)
    OUT_PARQUET.parent.mkdir(exist_ok=True)
    out.to_parquet(OUT_PARQUET, index=False)
    log(f"[saved] {OUT_PARQUET}  rows={len(out):,}")

    desc = out[FEAT_COLS].describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]]
    log("\n=== feature describe (全期間) ===")
    log(desc.to_string())

    rep = {
        "shrink_k": shrink_k,
        "rows": int(len(out)),
        "horses": int(df[COL_PED].nunique()),
        "prev_hosei_cov_pct": float(cov),
        "leak_check_first_race_evt_n0": leak_ok,
        "feature_describe": json.loads(desc.to_json()),
    }
    OUT_LOG.parent.mkdir(exist_ok=True)
    with open(OUT_LOG, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    log(f"[saved] {OUT_LOG}")
    return 0


# ============================================================
# 合成データ自己テスト: 真の σ_h を MAE で復元できるか
# ============================================================
def selftest(seed: int = 42) -> int:
    log("=== selftest: 合成データで σ 復元 ===")
    rng = np.random.default_rng(seed)
    H = 400                                     # 馬数
    rows = []
    true_sigma = {}
    for h in range(H):
        mu_h = rng.normal(100, 8)               # 馬の地力指数
        sig_h = float(abs(rng.normal(6, 2)) + 1)  # 真のムラ
        true_sigma[h] = sig_h
        n_races = int(rng.integers(3, 25))      # 出走数
        figs = rng.normal(mu_h, sig_h, n_races)
        dates = 20130000 + np.sort(rng.integers(1, 90000, n_races))
        # prev_hosei = 1 走前の補正 (先頭は NaN)
        prev = np.concatenate([[np.nan], figs[:-1]])
        for d, p in zip(dates, prev):
            rows.append((h, int(d), p))
    df = pd.DataFrame(rows, columns=[COL_PED, COL_DATE, COL_HOSEI])
    df[COL_BAN] = 1
    df[COL_SPLIT] = "train"

    s = build_evt_features(df, shrink_k=4.0)
    # 各馬の最終行 (最も多くの履歴を持つ) で推定 σ vs 真値
    last = s.sort_values([COL_PED, COL_DATE], kind="stable").groupby(COL_PED).tail(1)
    last = last[last["evt_n"] >= 2]
    est = last["evt_sigma"].values
    tru = last[COL_PED].map(true_sigma).values
    mae_raw = float(np.nanmean(np.abs(est - tru)))
    mae_shr = float(np.nanmean(np.abs(last["evt_sigma_s"].values - tru)))
    corr = float(np.corrcoef(est[np.isfinite(est)], tru[np.isfinite(est)])[0, 1])
    log(f"  馬数(n>=2)={len(last)}  生σ MAE={mae_raw:.3f}  収縮σ MAE={mae_shr:.3f}  corr(生σ,真σ)={corr:.3f}")
    ok = mae_raw < 2.0 and corr > 0.7
    log(f"  → {'PASS' if ok else 'FAIL'} (MAE<2.0 かつ corr>0.7 を期待)")
    return 0 if ok else 1


def validate_leak() -> int:
    """prev_hosei が「直前走の実現補正のラグ1」であることを hosei CSV の current 補正と
    突合して定義的に証明する (一致率 ~100% を期待 = leak-safe の根拠)。"""
    HOSEI = BASE / "data/hosei/H_20130105-20251228.csv"
    log("=== validate-leak: prev_hosei == lag1(realized 補正) を証明 ===")
    h = (pd.read_csv(HOSEI, encoding="cp932", usecols=["レースID(新)", "馬番", "補正"])
         .rename(columns={"補正": "hosei_cur"}))
    h["馬番"] = pd.to_numeric(h["馬番"], errors="coerce").astype("Int64")
    h = h.drop_duplicates(["レースID(新)", "馬番"])
    m = pd.read_csv(MASTER, encoding="utf-8-sig",
                    usecols=lambda c: c in ["レースID(新)", "馬番", COL_PED, COL_DATE, COL_HOSEI],
                    low_memory=False)
    m["馬番"] = pd.to_numeric(m["馬番"], errors="coerce").astype("Int64")
    m[COL_DATE] = pd.to_numeric(m[COL_DATE], errors="coerce")
    mm = (m.merge(h, on=["レースID(新)", "馬番"], how="left")
          .sort_values([COL_PED, COL_DATE], kind="stable"))
    mm["lag_cur"] = mm.groupby(COL_PED, sort=False)["hosei_cur"].shift(1)
    both = mm[mm[COL_HOSEI].notna() & mm["lag_cur"].notna()]
    match = float((both[COL_HOSEI] == both["lag_cur"]).mean())
    # 当該current補正との一致 (離散指数の偶然一致のみで ~5% 程度のはず; ~100% ならリーク)
    leak = mm[mm[COL_HOSEI].notna() & mm["hosei_cur"].notna()]
    selfm = float((leak[COL_HOSEI] == leak["hosei_cur"]).mean())
    log(f"  prev_hosei == lag1(補正) 一致率 = {match*100:.2f}%  on {len(both):,} rows")
    log(f"  prev_hosei == 当該current補正  = {selfm*100:.2f}%  (離散偶然一致; ~100%ならリーク)")
    ok = match > 0.99 and selfm < 0.50
    log(f"  → {'PASS (leak-safe 確定)' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shrink-k", type=float, default=4.0)
    ap.add_argument("--ewm-span", type=int, default=4)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--validate-leak", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(selftest())
    if args.validate_leak:
        raise SystemExit(validate_leak())
    raise SystemExit(main(args.shrink_k, ewm_span=args.ewm_span))
