# -*- coding: utf-8 -*-
"""
evaluate.py — M0-M3/RAW/EWMA/ZERO比較、meeting-day paired bootstrap、permutation
placebo test。

モデル形式: logit(p_i) = offset_i + X_i @ beta。offset_i = baseline_logit_top3
(v6+市場、固定・fitしない)。betaのみL2正則化ロジスティック回帰でfitする
(statsmodels不在のためscipy.optimize.minimizeで手実装、依存追加を避ける)。

fit: 2023年development。評価: 2024年・2025年(genuinely OOS、再fitしない)。
主判定: top3 logloss(仕様書Gate2C)。

実行: PYTHONUTF8=1 ./venv311/Scripts/python.exe -m analysis.mcond.exp08_online_track_state_dev.evaluate
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
HERE = Path(__file__).resolve().parent

from analysis.mcond.exp08_online_track_state_dev.online_state import STATE_DIMS  # noqa: E402

EPS = 1e-9

MODEL_FEATURES = {
    "M0": [],
    "RAW": [f"{d}_raw" for d in STATE_DIMS],
    "EWMA": [f"{d}_ewma" for d in STATE_DIMS],
    "ZERO": [],
    "M1": [f"{d}_raw" for d in STATE_DIMS],
    "M2": (
        [f"{d}_pre_mean" for d in STATE_DIMS]
        + [f"{d}_pre_var" for d in STATE_DIMS]
        + [f"{d}_pre_n_obs" for d in STATE_DIMS]
    ),
    "M3": (
        [f"{d}_raw" for d in STATE_DIMS]
        + [f"{d}_pre_mean" for d in STATE_DIMS]
        + [f"interaction_{d}" for d in STATE_DIMS]
    ),
}


def logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _neg_loglik_l2(beta, X, y, offset, l2):
    z = offset + X @ beta
    p = expit(z)
    p = np.clip(p, EPS, 1 - EPS)
    nll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    return nll + l2 * np.sum(beta ** 2)


def fit_offset_logistic(X: np.ndarray, y: np.ndarray, offset: np.ndarray, l2: float = 1.0) -> np.ndarray:
    if X.shape[1] == 0:
        return np.zeros(0)
    beta0 = np.zeros(X.shape[1])
    res = minimize(_neg_loglik_l2, beta0, args=(X, y, offset, l2), method="L-BFGS-B")
    return res.x


def predict_offset_logistic(X: np.ndarray, offset: np.ndarray, beta: np.ndarray) -> np.ndarray:
    z = offset + (X @ beta if X.shape[1] else 0.0)
    return expit(z)


def fit_and_eval_model(
    model_name: str, train_df: pd.DataFrame, eval_dfs: dict[str, pd.DataFrame], l2: float = 1.0,
) -> dict:
    """model_nameをtrain_df(2023)でfitし、eval_dfs({"2024":df,"2025":df})でOOS評価する。
    戻り値: {"beta":..., "2024": {"logloss":..., "p":...}, "2025": {...}}"""
    feats = MODEL_FEATURES[model_name]
    Xtr = train_df[feats].to_numpy(dtype=float) if feats else np.zeros((len(train_df), 0))
    ytr = train_df["top3"].to_numpy(dtype=float)
    offtr = train_df["baseline_logit_top3"].to_numpy(dtype=float)
    beta = fit_offset_logistic(Xtr, ytr, offtr, l2=l2)

    out = {"beta": beta.tolist(), "features": feats}
    for label, df in eval_dfs.items():
        Xev = df[feats].to_numpy(dtype=float) if feats else np.zeros((len(df), 0))
        yev = df["top3"].to_numpy(dtype=float)
        offev = df["baseline_logit_top3"].to_numpy(dtype=float)
        p = predict_offset_logistic(Xev, offev, beta)
        out[label] = {"logloss": logloss(yev, p), "p": p, "y": yev,
                      "rid16": df["rid16"].to_numpy(), "date": df["date"].to_numpy()}
    return out


# =============================================================================
# meeting-day paired bootstrap(EXP07と同じ規約: rid16[0:10]=日付+場)
# =============================================================================

def meeting_day_key(rid16: np.ndarray) -> np.ndarray:
    return np.array([str(r)[0:10] for r in rid16])


def paired_bootstrap_delta_logloss(
    y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, rid16: np.ndarray,
    n_boot: int = 2000, seed: int = 20260920,
) -> dict:
    """Δlogloss = logloss(model_a) - logloss(model_b) の meeting-day単位paired
    bootstrap。負=aがbより改善。97.5%CIの上限<0なら「改善が頑健」と判定できる。"""
    md = meeting_day_key(rid16)
    p_a_c = np.clip(p_a, EPS, 1 - EPS)
    p_b_c = np.clip(p_b, EPS, 1 - EPS)
    ll_a_row = -(y * np.log(p_a_c) + (1 - y) * np.log(1 - p_a_c))
    ll_b_row = -(y * np.log(p_b_c) + (1 - y) * np.log(1 - p_b_c))

    df = pd.DataFrame({"md": md, "ll_a": ll_a_row, "ll_b": ll_b_row})
    groups = df.groupby("md").agg(sum_a=("ll_a", "sum"), sum_b=("ll_b", "sum"), n=("ll_a", "size"))
    mds = groups.index.to_numpy()
    sum_a = groups["sum_a"].to_numpy()
    sum_b = groups["sum_b"].to_numpy()
    n = groups["n"].to_numpy()

    point = (sum_a.sum() - sum_b.sum()) / n.sum()

    rng = np.random.default_rng(seed)
    n_md = len(mds)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n_md, size=n_md)
        boot[i] = (sum_a[idx].sum() - sum_b[idx].sum()) / n[idx].sum()
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    return {"point": float(point), "ci95": [float(ci_lo), float(ci_hi)], "n_meeting_days": int(n_md)}


# =============================================================================
# permutation placebo(競馬場×日×芝ダート内で観測の時系列順序を置換)
# =============================================================================

def shuffle_observations_within_unit(obs: pd.DataFrame, seed: int) -> pd.DataFrame:
    """観測列(STATE_DIMS)の値だけを、同一unit(date,venue,surface)内でシャッフルする。
    欠損位置・レース数・更新回数・対象レース行・baseline予測はすべて不変
    (avail_ts/decision_timestamp/baseline_logit_top3/outcomeは一切変更しない、
    観測**値**の対応関係だけを崩す)。"""
    rng = np.random.default_rng(seed)
    out = obs.copy()
    for _, idx in obs.groupby(["date", "venue", "surface"], sort=False).groups.items():
        idx = np.array(idx)
        for d in STATE_DIMS:
            vals = out.loc[idx, d].to_numpy()
            non_na_mask = ~pd.isna(vals)
            non_na_idx = idx[non_na_mask]
            if len(non_na_idx) > 1:
                shuffled = rng.permutation(out.loc[non_na_idx, d].to_numpy())
                out.loc[non_na_idx, d] = shuffled
    return out
