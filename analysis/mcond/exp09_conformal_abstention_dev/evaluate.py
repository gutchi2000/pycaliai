# -*- coding: utf-8 -*-
"""
evaluate.py — Stage2実装順ステップ8: Gate1-4(情報指標比較・full-control・安定性)。

情報指標: レース単位のmulticlass logloss(-log(p_true_winner))・top-pick Brier
((1{top_pick==winner} - p_top_pick)^2)。**全7方式で同じ生PL確率を使い**、
方式が異なるのは「どのレースを参加/見送りに選ぶか」だけ(予測自体は変えない)
という設計(spec.json gate5_economicの「Conformal方式だけ賭け金/候補馬券/券種を
変更しない」と同じ精神を情報指標比較にも適用)。

実行: 単体実行しない。stage2_run.py から呼ぶ。
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
HERE = Path(__file__).resolve().parent

EPS = 1e-12


def _tansho_probs(scores: np.ndarray) -> np.ndarray:
    w = np.exp(scores - np.max(scores))
    return w / w.sum()


def race_level_metrics(race_frames: dict) -> pd.DataFrame:
    """全方式共通の生PL確率に基づくlogloss/Brier(方式に依存しない、参加選別前の
    「もし参加したら」の値として全レース分計算しておく)。"""
    rows = []
    for rid16, g in race_frames.items():
        scores = g["v6_score"].to_numpy(dtype=float)
        probs = _tansho_probs(scores)
        true_idx = int(np.where(g["win"].to_numpy() == 1)[0][0])
        logloss = -np.log(max(probs[true_idx], EPS))
        top_idx = int(np.argmax(probs))
        top_p = float(probs[top_idx])
        brier = (float(top_idx == true_idx) - top_p) ** 2
        rows.append({"rid16": rid16, "logloss": logloss, "brier": brier})
    return pd.DataFrame(rows)


def select_top_n(score_df: pd.DataFrame, n: int) -> set:
    """スコア昇順(小さいほど参加寄り)で上位n件のrid16集合を返す。"""
    ordered = score_df.sort_values("score", kind="mergesort")  # 安定ソート
    return set(ordered["rid16"].iloc[:n].tolist())


def meeting_day_key(rid16) -> str:
    return str(rid16)[0:10]


def paired_bootstrap_delta(metric_a: np.ndarray, metric_b: np.ndarray, rid16: np.ndarray,
                          n_boot: int = 2000, seed: int = 20260921) -> dict:
    """Δ = mean(metric_a) - mean(metric_b) のmeeting-day単位paired bootstrap。
    負=aがbより改善(logloss/Brierは小さいほど良い)。"""
    md = np.array([meeting_day_key(r) for r in rid16])
    df = pd.DataFrame({"md": md, "a": metric_a, "b": metric_b})
    g = df.groupby("md").agg(sum_a=("a", "sum"), sum_b=("b", "sum"), n=("a", "size"))
    sum_a, sum_b, n = g["sum_a"].to_numpy(), g["sum_b"].to_numpy(), g["n"].to_numpy()
    point = (sum_a.sum() - sum_b.sum()) / n.sum() if n.sum() else float("nan")
    rng = np.random.default_rng(seed)
    n_md = len(g)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n_md, size=n_md)
        boot[i] = (sum_a[idx].sum() - sum_b[idx].sum()) / n[idx].sum()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"point": float(point), "ci95": [float(lo), float(hi)], "n_meeting_days": int(n_md)}


def gate2_same_participation_rate(
    metrics: pd.DataFrame, method_scores: dict, participation_rate: float, n_eligible: int,
) -> dict:
    """participation_rate点1つについて、7方式全ての参加サブセットで
    logloss/Brierを比較する。method_scoresは{method_name: DataFrame(rid16,score)}。"""
    n_select = max(1, round(participation_rate * n_eligible))
    selected = {name: select_top_n(df, n_select) for name, df in method_scores.items()}
    for name, s in selected.items():
        assert len(s) == n_select, f"{name}: 選択件数{len(s)}!=期待{n_select}"

    conformal_sel = selected["conformal"]
    result = {"participation_rate": participation_rate, "n_select": n_select, "vs": {}}
    conf_metrics = metrics[metrics["rid16"].isin(conformal_sel)]
    result["conformal_mean_logloss"] = float(conf_metrics["logloss"].mean())
    result["conformal_mean_brier"] = float(conf_metrics["brier"].mean())

    for name, sel in selected.items():
        if name == "conformal":
            continue
        other_metrics = metrics[metrics["rid16"].isin(sel)]
        # ペア比較はrid16が異なりうる(選ばれたレース集合そのものが違う)ため、
        # meeting-day単位で集計しbootstrapする(個別レースの1:1対応は前提にしない)
        boot_ll = paired_bootstrap_delta(
            metrics.set_index("rid16").loc[list(conformal_sel), "logloss"].to_numpy(),
            metrics.set_index("rid16").loc[list(sel), "logloss"].to_numpy(),
            np.array(list(conformal_sel)) if len(conformal_sel) <= len(sel) else np.array(list(sel)),
        ) if len(conformal_sel) == len(sel) else None
        result["vs"][name] = {
            "other_mean_logloss": float(other_metrics["logloss"].mean()),
            "other_mean_brier": float(other_metrics["brier"].mean()),
            "conformal_minus_other_logloss": result["conformal_mean_logloss"] - float(other_metrics["logloss"].mean()),
        }
    return result


def gate3_full_control(
    metrics: pd.DataFrame, conformal_score: pd.DataFrame, controls: pd.DataFrame,
) -> dict:
    """APS-derived scoreを、controls(モデル自身の確信度等)全てを統制した
    ロジスティック回帰に追加項として入れ、係数の95%CIがゼロを跨がないかを見る。
    目的変数はcorrect=1{argmax pick == true winner}(2値、full-control回帰の
    標準的な目的変数選択)。"""
    from scipy.special import expit
    df = metrics.merge(conformal_score.rename(columns={"score": "conformal_score"}), on="rid16")
    df = df.merge(controls, on="rid16")
    y = (df["brier"] < 0.5).astype(int).to_numpy()  # top_pick==winnerの代理(Brier<0.5⇔的中)
    control_cols = [c for c in controls.columns if c != "rid16"]
    X_controls = df[control_cols].to_numpy(dtype=float)
    X_full = np.column_stack([X_controls, df["conformal_score"].to_numpy(dtype=float)])
    X_full = (X_full - X_full.mean(axis=0)) / (X_full.std(axis=0) + EPS)

    def nll(beta, X, y):
        z = X @ beta
        p = np.clip(expit(z), EPS, 1 - EPS)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    beta0 = np.zeros(X_full.shape[1])
    res = minimize(nll, beta0, args=(X_full, y), method="L-BFGS-B")
    beta_hat = res.x
    conformal_coef_idx = X_full.shape[1] - 1

    rng = np.random.default_rng(20260921)
    n = len(y)
    boot_coefs = np.empty(500)
    for b in range(500):
        idx = rng.integers(0, n, size=n)
        r = minimize(nll, beta0, args=(X_full[idx], y[idx]), method="L-BFGS-B")
        boot_coefs[b] = r.x[conformal_coef_idx]
    lo, hi = np.percentile(boot_coefs, [2.5, 97.5])
    return {
        "conformal_coefficient": float(beta_hat[conformal_coef_idx]),
        "ci95": [float(lo), float(hi)],
        "survives_full_control": bool(lo > 0 or hi < 0),
        "control_columns": control_cols,
    }
