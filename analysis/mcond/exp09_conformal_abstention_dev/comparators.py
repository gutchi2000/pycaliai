# -*- coding: utf-8 -*-
"""
comparators.py — Stage2実装順ステップ7: 6つの比較方式(Conformal以外)。
spec.json `seven_methods` の2-7番。全方式は「小さいほど参加寄り」のスコアを
返す統一規約とし、各participation_rate点で同じN件を選べるようにする。

方式2 max_probability: -max(生PL単勝確率)
方式3 entropy: 全馬生PL確率分布のnormalized entropy
方式4 ood_support: 2023-only fit(median/IQR標準化+PCA+k-NN距離)、
  ood_support.py(EXP06)と同型設計の独自軽量実装(コード自体は再利用しない)
方式5 feature_missing: v6の実特徴量列(120列)に対するレース平均NaN率
方式6 lr_control_2023_only: 2023年のみでfitしたロジスティック回帰
  (目的変数="argmax v6生確率の馬が勝たなかった"、2023結果を見て確定済み)
方式7 current_gate: production_policy.pyのchaos percentile(読み取り専用参照、
  唯一この方式だけ本番実運用の較正済みpipelineを使う)

実行: 単体実行しない。evaluate.py / stage2_run.py から import して使う。
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
HERE = Path(__file__).resolve().parent

import production_policy as PP  # noqa: E402

K_NEIGHBORS = 20
PCA_MAX_DIMS = 5  # 特徴数が少ないため軽量版(EXP06の20よりスコープを縮小、独自実装)


def _tansho_probs(scores: np.ndarray) -> np.ndarray:
    w = np.exp(scores - np.max(scores))
    return w / w.sum()


def _race_level_stats(race_frames: dict) -> pd.DataFrame:
    """race_frames(rid16 -> DataFrame(ban,v6_score,fin,win))から
    max_prob/entropy/n_field/winner_is_favoriteを計算する共通ヘルパ。"""
    rows = []
    for rid16, g in race_frames.items():
        scores = g["v6_score"].to_numpy(dtype=float)
        probs = _tansho_probs(scores)
        n = len(probs)
        ent = float(-(probs * np.log(np.clip(probs, 1e-12, 1))).sum() / np.log(n))
        max_p = float(probs.max())
        true_idx = int(np.where(g["win"].to_numpy() == 1)[0][0])
        favorite_idx = int(np.argmax(probs))
        rows.append({"rid16": rid16, "max_prob": max_p, "entropy": ent, "n_field": n,
                     "top_pick_wrong": int(favorite_idx != true_idx)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 方式2・3: max probability / entropy(fitなし、生確率から直接計算)
# ---------------------------------------------------------------------------

def score_max_probability(race_frames: dict) -> pd.DataFrame:
    stats = _race_level_stats(race_frames)
    stats["score"] = -stats["max_prob"]
    return stats[["rid16", "score"]]


def score_entropy(race_frames: dict) -> pd.DataFrame:
    stats = _race_level_stats(race_frames)
    stats["score"] = stats["entropy"]
    return stats[["rid16", "score"]]


# ---------------------------------------------------------------------------
# 方式4: OOD/support(2023-only fit、独自軽量実装)
# ---------------------------------------------------------------------------

class OODSupportModel:
    """ood_support.py(EXP06)と同型設計(median/IQR標準化→PCA→k-NN距離→ECDF)の
    独自軽量実装。特徴数を絞った版(n_field/max_prob/entropyの3次元)。"""

    def __init__(self, k: int = K_NEIGHBORS, pca_dims: int = PCA_MAX_DIMS):
        self.k = k
        self.pca_dims = pca_dims
        self.scaler_ = None
        self.pca_ = None
        self.ref_pca_ = None
        self.d20_ref_sorted_ = None

    def fit(self, stats_2023: pd.DataFrame) -> "OODSupportModel":
        X = stats_2023[["n_field", "max_prob", "entropy"]].to_numpy(dtype=float)
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        n_dims = min(self.pca_dims, Xs.shape[1], Xs.shape[0])
        self.pca_ = PCA(n_components=n_dims).fit(Xs)
        self.ref_pca_ = self.pca_.transform(Xs)
        nn = NearestNeighbors(n_neighbors=min(self.k + 1, len(self.ref_pca_)))
        nn.fit(self.ref_pca_)
        dist, _ = nn.kneighbors(self.ref_pca_)
        col = min(self.k, dist.shape[1] - 1)
        self.d20_ref_sorted_ = np.sort(dist[:, col])
        return self

    def score(self, stats_query: pd.DataFrame) -> np.ndarray:
        X = stats_query[["n_field", "max_prob", "entropy"]].to_numpy(dtype=float)
        Xs = self.scaler_.transform(X)
        query_pca = self.pca_.transform(Xs)
        nn = NearestNeighbors(n_neighbors=min(self.k, len(self.ref_pca_)))
        nn.fit(self.ref_pca_)
        dist, _ = nn.kneighbors(query_pca)
        d20 = dist[:, min(self.k, dist.shape[1]) - 1]
        idx = np.searchsorted(self.d20_ref_sorted_, d20, side="right")
        support = 1.0 - idx / len(self.d20_ref_sorted_)
        return support  # 高いほどin-distribution


def fit_ood_model(race_frames_2023: dict) -> OODSupportModel:
    stats = _race_level_stats(race_frames_2023)
    return OODSupportModel().fit(stats)


def score_ood_support(model: OODSupportModel, race_frames: dict) -> pd.DataFrame:
    stats = _race_level_stats(race_frames)
    support = model.score(stats)
    stats["score"] = -support  # supportが高い=参加寄り=scoreは小さく
    return stats[["rid16", "score"]]


# ---------------------------------------------------------------------------
# 方式5: feature missingness(v6の実特徴量列に対するレース平均NaN率)
# ---------------------------------------------------------------------------

def score_feature_missing(years: list[int]) -> pd.DataFrame:
    import joblib
    m = joblib.load(BASE / "models/unified_rank_v6.pkl")
    feature_cols = m["feature_cols"]
    usecols = ["レースID(新/馬番無)", "日付"] + [c for c in feature_cols if c not in
                                              ("着順", "fukusho_flag", "roi_target")]
    df = pd.read_csv(BASE / "data/master_v2_20130105-20251228.csv", encoding="utf-8-sig",
                     low_memory=False, usecols=lambda c: c in usecols)
    # 2026-09-21実装中に発覚: yearsフィルタを忘れると全年(2013-2025、n≈44,907レース)
    # を返してしまう実害バグだった(2023単年で呼んでも他の年が混入する)。
    df["_year"] = df["日付"].astype(str).str[:4].astype(int)
    df = df[df["_year"].isin(years)]
    df["rid16"] = pd.to_numeric(df["レースID(新/馬番無)"], errors="coerce").astype("Int64").astype(str)
    feat_present = [c for c in feature_cols if c in df.columns]
    df["_missing_rate"] = df[feat_present].isna().mean(axis=1)
    race_missing = df.groupby("rid16")["_missing_rate"].mean().reset_index()
    race_missing = race_missing.rename(columns={"_missing_rate": "score"})
    return race_missing[["rid16", "score"]]


# ---------------------------------------------------------------------------
# 方式6: 2023年のみでfitしたLR_CONTROL
# ---------------------------------------------------------------------------

LR_CONTROL_TARGET_DEFINITION = (
    "target = 1 if argmax(生PL単勝確率)の馬が勝たなかった場合(top_pick_wrong)、"
    "0ならその馬が勝った。2023年developmentのみで固定、2024-2025結果を見て変更しない。"
)


def fit_lr_control(race_frames_2023: dict):
    stats = _race_level_stats(race_frames_2023)
    X = stats[["n_field", "max_prob", "entropy"]].to_numpy(dtype=float)
    y = stats["top_pick_wrong"].to_numpy(dtype=int)
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=5000).fit(scaler.transform(X), y)
    return clf, scaler


def score_lr_control(clf, scaler, race_frames: dict) -> pd.DataFrame:
    stats = _race_level_stats(race_frames)
    X = stats[["n_field", "max_prob", "entropy"]].to_numpy(dtype=float)
    p_error = clf.predict_proba(scaler.transform(X))[:, 1]
    stats["score"] = p_error  # 誤り確率が高い=見送り寄り=scoreは大きく
    return stats[["rid16", "score"]]


# ---------------------------------------------------------------------------
# 方式7: 現行chaos/participation gate(読み取り専用参照)
# ---------------------------------------------------------------------------

def score_current_gate(race_frames: dict) -> pd.DataFrame:
    """本番のfield_chaos_score定義(較正後p_win分布の正規化エントロピー)を
    pl_calibrators_v6_serve.pklで再現し、production_policy.percentile()で
    本番と同じ変換をかける(読み取り専用参照、コード変更なし)。"""
    import joblib
    cal = joblib.load(BASE / "models/pl_calibrators_v6_serve.pkl")["calibrators"]["tansho"]
    rows = []
    for rid16, g in race_frames.items():
        scores = g["v6_score"].to_numpy(dtype=float)
        raw_probs = _tansho_probs(scores)
        cal_probs = cal.predict(raw_probs)
        cal_probs = np.clip(cal_probs, 1e-12, None)
        cal_probs = cal_probs / cal_probs.sum()
        n = len(cal_probs)
        chaos_raw = float(-(cal_probs * np.log(cal_probs)).sum() / np.log(n))
        chaos_pct = PP.percentile(chaos_raw, "field_chaos_score")
        rows.append({"rid16": rid16, "score": chaos_pct if chaos_pct is not None else 1.0})
    return pd.DataFrame(rows)
