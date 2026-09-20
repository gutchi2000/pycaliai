# -*- coding: utf-8 -*-
"""
ood_support.py — in_distribution_support / similar_past_case_count の実装
================================================================================
2026-09-20、ユーザー指定の固定アルゴリズム。結果・着順・払戻を一切使わない、
2023年をsupport reference兼development期間として構築する距離ベース指標。

【アルゴリズム(ユーザー指定、結果を見て変更しない)】
基準期間: 2023年=support reference兼development、2024-2025年=主評価(support変換・
閾値は2024-2025で再fitしない)。

距離入力(race-state vector、historical_state.pyが計算する下記のみ。in_distribution_support/
similar_past_case_count自身は距離入力に含めない):
  M1/M3/M4上位確率, 各モデルのentropy, モデル間順位不一致, モデル間確率分散,
  T-35市場確率, 市場entropy, AI対市場乖離, 頭数, 競馬場, 芝/ダート, 距離帯, クラス帯,
  人気帯, 特徴欠損率, 未知カテゴリ率

変換:
  連続値: 2023年のmedian/IQRでrobust standardization
  カテゴリ: 2023年で固定したone-hot、未知カテゴリは__UNK__
  欠損: 2023年median補完+missing indicator
  PCA: 2023年だけでfit、累積寄与率95%、最大20次元
  距離: Euclidean、k=20
  2023年内はself-matchを除外したleave-one-out距離

指標定義:
  d20_ref   = 2023年各レースの(LOOで)20番目近傍までの距離
  d20_query = 対象レースから2023年referenceへの20番目近傍距離
  in_distribution_support = clip(1 - ECDF_2023(d20_query), 0, 1)
  similar_radius = 2023年のd20_refの95パーセンタイル
  similar_past_case_count = 2023年referenceのうちqueryからsimilar_radius以内のレース数

k・PCA次元・半径・距離関数は結果を見て変更しない(spec.jsonのsupport_metric_definitionに
固定済み)。
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

K_NEIGHBORS = 20
PCA_MAX_DIMS = 20
PCA_VARIANCE_TARGET = 0.95
RADIUS_PERCENTILE = 95.0
UNK_TOKEN = "__UNK__"

# 2026-09-20夜、ユーザー指摘により unknown_category_rate をここから除外した。
# 歴史データセットからは真の未知カテゴリ率を再構成できないため0で埋めていたが、
# 0は「未知カテゴリが存在しない」という有効な観測値であり、再構成不能の代用には
# 不適切だった(availability=falseとして完全に省略するのが正しい、CONTINUOUS_COLSの
# 対象=distance入力からも除外する)。詳細はstate_schema_v3のspec.json amendment参照。
CONTINUOUS_COLS = [
    "m1_top_prob", "m3_top_prob", "m4_top_prob",
    "entropy_m1", "entropy_m3", "entropy_m4",
    "model_rank_disagreement", "model_prob_variance",
    "market_prob", "market_entropy", "ai_market_divergence",
    "field_size", "feature_missing_rate",
]
CATEGORICAL_COLS = ["venue", "surface", "distance_band", "class_band", "popularity_band"]


@dataclass
class SupportModel:
    """2023年でfitし、2024-2025年へは変換のみ適用する(再fitしない)凍結モデル。"""
    median_: dict = field(default_factory=dict)
    iqr_: dict = field(default_factory=dict)
    category_vocab_: dict = field(default_factory=dict)
    pca_mean_: np.ndarray | None = None
    pca_components_: np.ndarray | None = None
    n_pca_dims_: int = 0
    ref_pca_: np.ndarray | None = None  # 2023参照集合のPCA空間座標
    ref_index_: object = None  # ref_pca_の各行に対応するインデックス(rid16等)
    d20_ref_: np.ndarray | None = None
    ecdf_sorted_: np.ndarray | None = None  # d20_ref_をソートしたもの(ECDF評価用)
    similar_radius_: float = 0.0
    n_reference_: int = 0

    def _build_continuous_matrix(self, df: pd.DataFrame, fitting: bool) -> np.ndarray:
        cols = []
        for c in CONTINUOUS_COLS:
            x = df[c].to_numpy(dtype=float)
            if fitting:
                med = np.nanmedian(x)
                q75, q25 = np.nanpercentile(x[~np.isnan(x)], [75, 25]) if np.isfinite(x).any() else (1.0, 0.0)
                iqr = q75 - q25
                if iqr <= 0:
                    iqr = 1.0
                self.median_[c] = float(med)
                self.iqr_[c] = float(iqr)
            med = self.median_[c]
            iqr = self.iqr_[c]
            missing_mask = np.isnan(x)
            x_filled = np.where(missing_mask, med, x)
            z = (x_filled - med) / iqr
            cols.append(z.reshape(-1, 1))
            cols.append(missing_mask.astype(float).reshape(-1, 1))  # missing indicator
        return np.hstack(cols)

    def _build_categorical_matrix(self, df: pd.DataFrame, fitting: bool) -> np.ndarray:
        blocks = []
        for c in CATEGORICAL_COLS:
            vals = df[c].astype(str).to_numpy()
            if fitting:
                vocab = sorted(set(vals))
                self.category_vocab_[c] = vocab
            vocab = self.category_vocab_[c]
            vocab_with_unk = vocab  # UNKは列として持たず、既知カテゴリ全て0の行=UNK相当にする
            block = np.zeros((len(vals), len(vocab_with_unk)), dtype=float)
            vocab_index = {v: i for i, v in enumerate(vocab_with_unk)}
            for i, v in enumerate(vals):
                j = vocab_index.get(v)
                if j is not None:
                    block[i, j] = 1.0
                # v が vocab に無い(未知カテゴリ)場合は全列0のまま = __UNK__ 相当
            blocks.append(block)
        return np.hstack(blocks) if blocks else np.zeros((len(df), 0))

    def _transform(self, df: pd.DataFrame, fitting: bool) -> np.ndarray:
        cont = self._build_continuous_matrix(df, fitting)
        cat = self._build_categorical_matrix(df, fitting)
        return np.hstack([cont, cat])

    def fit(self, df_2023: pd.DataFrame) -> "SupportModel":
        """2023年のrace-state vectorだけでfitする。結果ラベル列は一切参照しない
        (df_2023には距離入力の列だけを渡すこと、呼び出し側の責務)。"""
        Z = self._transform(df_2023, fitting=True)

        pca_full = PCA(n_components=min(Z.shape[0], Z.shape[1]))
        pca_full.fit(Z)
        cum = np.cumsum(pca_full.explained_variance_ratio_)
        n_dims = int(np.searchsorted(cum, PCA_VARIANCE_TARGET) + 1)
        n_dims = min(n_dims, PCA_MAX_DIMS, Z.shape[1])
        self.n_pca_dims_ = n_dims
        self.pca_mean_ = pca_full.mean_
        self.pca_components_ = pca_full.components_[:n_dims]

        ref_pca = (Z - self.pca_mean_) @ self.pca_components_.T
        self.ref_pca_ = ref_pca
        self.ref_index_ = df_2023.index
        self.n_reference_ = ref_pca.shape[0]

        # d20_ref: 2023年内のleave-one-out 20番目近傍距離
        nn = NearestNeighbors(n_neighbors=min(K_NEIGHBORS + 1, ref_pca.shape[0]), metric="euclidean")
        nn.fit(ref_pca)
        dist, _ = nn.kneighbors(ref_pca)
        # dist[:,0] は自分自身との距離(0)のはず。K_NEIGHBORS番目(0-indexで K_NEIGHBORS)の列を採用
        col = min(K_NEIGHBORS, dist.shape[1] - 1)
        d20_ref = dist[:, col]
        self.d20_ref_ = d20_ref
        self.ecdf_sorted_ = np.sort(d20_ref)
        self.similar_radius_ = float(np.percentile(d20_ref, RADIUS_PERCENTILE))
        return self

    def _ecdf(self, values: np.ndarray) -> np.ndarray:
        """2023年のd20_ref分布に対するECDF(各valueが何分位に位置するか、0-1)。"""
        sorted_ref = self.ecdf_sorted_
        idx = np.searchsorted(sorted_ref, values, side="right")
        return idx / len(sorted_ref)

    def score(self, df_query: pd.DataFrame) -> pd.DataFrame:
        """2023年で凍結したパラメータのみで変換する(再fitしない)。"""
        Z = self._transform(df_query, fitting=False)
        query_pca = (Z - self.pca_mean_) @ self.pca_components_.T

        nn = NearestNeighbors(n_neighbors=min(K_NEIGHBORS, self.ref_pca_.shape[0]), metric="euclidean")
        nn.fit(self.ref_pca_)
        dist, _ = nn.kneighbors(query_pca)
        d20_query = dist[:, min(K_NEIGHBORS, dist.shape[1]) - 1]

        in_distribution_support = np.clip(1.0 - self._ecdf(d20_query), 0.0, 1.0)

        # similar_past_case_count: 2023参照全件との距離をradius以内で数える
        # (kNNのk=20近傍だけでなく全参照点との比較が必要なのでradius_neighbors を使う)
        counts = nn.radius_neighbors(query_pca, radius=self.similar_radius_, return_distance=False)
        similar_past_case_count = np.array([len(c) for c in counts], dtype=int)

        return pd.DataFrame({
            "in_distribution_support": in_distribution_support,
            "similar_past_case_count": similar_past_case_count,
            "d20_query": d20_query,
        }, index=df_query.index)

    def score_reference_self(self) -> pd.DataFrame:
        """2023年参照集合自身をスコアする専用メソッド(2026-09-20夜追加、ユーザー指定で
        2023年もJev問い合わせ対象に含めることになったため)。score()を2023年自身に
        適用すると各点が「自分自身」を参照集合の中に見つけてしまい距離0=support高すぎに
        水増しされる(self-match)。fit()時に既にleave-one-outで計算済みのd20_ref_を
        再利用し、similar_past_case_countも自分自身を除いてカウントする。"""
        in_distribution_support = np.clip(1.0 - self._ecdf(self.d20_ref_), 0.0, 1.0)
        nn = NearestNeighbors(n_neighbors=min(K_NEIGHBORS + 1, self.ref_pca_.shape[0]), metric="euclidean")
        nn.fit(self.ref_pca_)
        counts_incl_self = nn.radius_neighbors(self.ref_pca_, radius=self.similar_radius_,
                                                return_distance=False)
        similar_past_case_count = np.array([max(len(c) - 1, 0) for c in counts_incl_self], dtype=int)
        return pd.DataFrame({
            "in_distribution_support": in_distribution_support,
            "similar_past_case_count": similar_past_case_count,
            "d20_query": self.d20_ref_,
        }, index=self.ref_index_)

    def to_meta_dict(self) -> dict:
        """spec amendment / 監査用のメタデータ(パラメータの値そのものではなく形状・統計のみ)。"""
        return {
            "n_reference_races_2023": self.n_reference_,
            "n_pca_dims": self.n_pca_dims_,
            "similar_radius": self.similar_radius_,
            "d20_ref_median": float(np.median(self.d20_ref_)),
            "d20_ref_p95": float(np.percentile(self.d20_ref_, 95)),
            "category_vocab_sizes": {k: len(v) for k, v in self.category_vocab_.items()},
        }
