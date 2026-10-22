# -*- coding: utf-8 -*-
"""
model_v2.py — 馬券判断 AI v2 の推定器（仕様 v1.0 §5）
=====================================================
- 主の推定器（2 段）: (i) P(R>0) を LightGBM 2 値分類 → 学習期間内の isotonic で較正、
  (ii) R>0 のときの log R を LightGBM 回帰 → 対数正規の補正 exp(μ + σ²/2) で期待値へ。E[R] = (i) × (ii)。
  実装上の取り決め（仕様に書かれていない細部、DEV_REPORT_V2.md に明記）:
    * 「学習期間内の isotonic」は、学習期間の開催日を 3 つに分けた交差予測（OOF）の生確率で fit する
      （学習に使った行そのものの予測で fit すると過学習分だけ自信過剰になるため）。分類器の本体は学習期間全体で学習。
    * σ² は同じ 3 分割の交差予測の残差分散（券種別）。
- 比較の推定器: R − 1 を学習データの上下 1% 点で打ち切り、LightGBM の Huber 損失（δ = 1.0 = 賭け金 1 単位）で回帰。
  E[R] = 1 + 予測。
- 不確かさ: 開催日単位の bootstrap（学習期間の開催日を復元抽出し、行を複製して作り直す）でモデル 10 本。
  控えめな採算 = 10 本の E[R] の下から 20% 点（numpy.quantile 線形補間 = 2 番目と 3 番目の小さい値の 0.2:0.8 内分）。
- 購入規則: 控えめな採算 > 1.00 の候補を 100 円で買う。
- ハイパーパラメータ候補 12 個（葉の数 × 最小葉サイズ × (学習率, 木の本数)）。
"""
from __future__ import annotations

import hashlib
import io
import itertools
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from analysis.bet_decision_ai.v2.data_v2 import CAT_COLS

N_BOOT = 10
BOOT_Q = 0.20
BUY_THRESHOLD = 1.00
KFOLD_CAL = 3
HUBER_DELTA = 1.0
CLIP_Q = (0.01, 0.99)
NUM_THREADS = 16

GRID = [{"num_leaves": nl, "min_data_in_leaf": md, "learning_rate": lr, "n_trees": nt}
        for nl, md, (lr, nt) in itertools.product([15, 63], [200, 2000], [(0.02, 800), (0.05, 300), (0.10, 100)])]
assert len(GRID) <= 12

_COMMON = {"verbose": -1, "num_threads": NUM_THREADS, "feature_fraction": 0.8, "bagging_fraction": 0.8,
           "bagging_freq": 1, "lambda_l2": 1.0, "max_bin": 255, "deterministic": True, "force_row_wise": True}


def _params(cfg: dict, objective: str, seed: int, **extra) -> dict:
    p = dict(_COMMON)
    p.update(objective=objective, num_leaves=cfg["num_leaves"], min_data_in_leaf=cfg["min_data_in_leaf"],
             learning_rate=cfg["learning_rate"], seed=seed, bagging_seed=seed, feature_fraction_seed=seed,
             data_random_seed=seed)
    p.update(extra)
    return p


def _cat(X: pd.DataFrame) -> list[str]:
    return [c for c in CAT_COLS if c in X.columns]


def _train(X: pd.DataFrame, y: np.ndarray, params: dict, n_trees: int) -> lgb.Booster:
    ds = lgb.Dataset(X, label=y, categorical_feature=_cat(X), free_raw_data=True)
    return lgb.train(params, ds, num_boost_round=n_trees)


def _bt_codes(X: pd.DataFrame) -> np.ndarray:
    return X["bet_type"].astype(str).to_numpy()


# ================================================================ 主の推定器（2 段）
class TwoStage:
    kind = "two_stage"

    def __init__(self):
        self.clf = self.reg = None
        self.iso = None
        self.sigma2: dict = {}
        self.cfg: dict = {}
        self.info: dict = {}

    def fit(self, X: pd.DataFrame, R: np.ndarray, day: np.ndarray, cfg: dict, seed: int) -> "TwoStage":
        self.cfg = dict(cfg)
        y = (R > 0).astype(np.float64)
        pos = R > 0
        logR = np.log(np.where(pos, R, 1.0))
        rng = np.random.default_rng(10_000 + seed)
        udays = np.unique(day)
        perm = rng.permutation(len(udays))
        fold_of_day = dict(zip(udays[perm], np.arange(len(udays)) % KFOLD_CAL))
        fold = np.array([fold_of_day[x] for x in day])
        pc = _params(cfg, "binary", seed)
        pr = _params(cfg, "regression", seed)
        oof_p = np.full(len(X), np.nan)
        oof_mu = np.full(len(X), np.nan)
        for f in range(KFOLD_CAL):
            trm, tem = fold != f, fold == f
            m = _train(X[trm], y[trm], pc, cfg["n_trees"])
            oof_p[tem] = m.predict(X[tem])
            trp = trm & pos
            r = _train(X[trp], logR[trp], pr, cfg["n_trees"])
            tep = tem & pos
            oof_mu[tep] = r.predict(X[tep])
        self.iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
        self.iso.fit(oof_p, y)
        bt = _bt_codes(X)
        res = (logR - oof_mu)[pos]
        for b in np.unique(bt):
            mb = (bt[pos] == b)
            self.sigma2[b] = float(np.mean(res[mb] ** 2)) if mb.any() else 0.0
        self.clf = _train(X, y, pc, cfg["n_trees"])
        self.reg = _train(X[pos], logR[pos], pr, cfg["n_trees"])
        self.info = {"n_rows": int(len(X)), "n_pos": int(pos.sum()), "n_days": int(len(udays)),
                     "sigma2": self.sigma2,
                     "oof_brier": float(np.mean((self.iso.predict(oof_p) - y) ** 2))}
        return self

    def predict_parts(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        p = self.iso.predict(self.clf.predict(X))
        mu = self.reg.predict(X)
        s2 = np.array([self.sigma2.get(b, 0.0) for b in _bt_codes(X)])
        m = np.exp(mu + s2 / 2.0)
        return p, m

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        p, m = self.predict_parts(X)
        return p * m

    def to_state(self) -> dict:
        return {"kind": self.kind, "cfg": self.cfg, "clf": self.clf.model_to_string(),
                "reg": self.reg.model_to_string(), "iso": pickle.dumps(self.iso), "sigma2": self.sigma2,
                "info": self.info}

    @classmethod
    def from_state(cls, s: dict) -> "TwoStage":
        o = cls()
        o.cfg, o.sigma2, o.info = s["cfg"], s["sigma2"], s["info"]
        o.clf = lgb.Booster(model_str=s["clf"])
        o.reg = lgb.Booster(model_str=s["reg"])
        o.iso = pickle.loads(s["iso"])
        return o


# ================================================================ 比較の推定器（Huber）
class HuberDirect:
    kind = "huber"

    def __init__(self):
        self.m = None
        self.clip = (None, None)
        self.cfg: dict = {}
        self.info: dict = {}

    def fit(self, X: pd.DataFrame, R: np.ndarray, day: np.ndarray, cfg: dict, seed: int) -> "HuberDirect":
        self.cfg = dict(cfg)
        t = R - 1.0
        lo, hi = np.quantile(t, CLIP_Q[0]), np.quantile(t, CLIP_Q[1])
        self.clip = (float(lo), float(hi))
        p = _params(cfg, "huber", seed, alpha=HUBER_DELTA)
        self.m = _train(X, np.clip(t, lo, hi), p, cfg["n_trees"])
        self.info = {"n_rows": int(len(X)), "clip": self.clip, "n_days": int(len(np.unique(day)))}
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return 1.0 + self.m.predict(X)

    def to_state(self) -> dict:
        return {"kind": self.kind, "cfg": self.cfg, "m": self.m.model_to_string(), "clip": self.clip,
                "info": self.info}

    @classmethod
    def from_state(cls, s: dict) -> "HuberDirect":
        o = cls()
        o.cfg, o.clip, o.info = s["cfg"], tuple(s["clip"]), s["info"]
        o.m = lgb.Booster(model_str=s["m"])
        return o


ESTIMATORS = {"two_stage": TwoStage, "huber": HuberDirect}


def from_state(s: dict):
    return ESTIMATORS[s["kind"]].from_state(s)


# ================================================================ bootstrap 10 本
def boot_index(day: np.ndarray, b: int, base_seed: int) -> np.ndarray:
    """開催日の復元抽出 → 行の複製インデックス。"""
    udays, inv = np.unique(day, return_inverse=True)
    rng = np.random.default_rng(base_seed * 1000 + b)
    cnt = np.bincount(rng.integers(0, len(udays), len(udays)), minlength=len(udays))
    rows_by_day = np.split(np.argsort(inv, kind="stable"), np.cumsum(np.bincount(inv, minlength=len(udays)))[:-1])
    idx = [np.tile(rows_by_day[k], cnt[k]) for k in np.flatnonzero(cnt)]
    return np.concatenate(idx)


def fit_bootstrap(kind: str, X: pd.DataFrame, R: np.ndarray, day: np.ndarray, cfg: dict, base_seed: int = 0,
                  n_boot: int = N_BOOT, log=None) -> list:
    models = []
    for b in range(n_boot):
        idx = boot_index(day, b, base_seed)
        Xb = X.iloc[idx].reset_index(drop=True)
        m = ESTIMATORS[kind]().fit(Xb, R[idx], day[idx], cfg, seed=base_seed * 100 + b)
        m.info["boot"] = {"b": b, "n_rows": int(len(idx)), "n_unique_days": int(len(np.unique(day[idx])))}
        models.append(m)
        if log:
            log(f"      boot {kind} {b + 1}/{n_boot} rows={len(idx):,}")
    return models


def predict_ensemble(models: list, X: pd.DataFrame) -> np.ndarray:
    return np.stack([m.predict(X) for m in models], 0)          # [n_boot, n]


def conservative(P: np.ndarray) -> np.ndarray:
    return np.quantile(P, BOOT_Q, axis=0)


def buy_mask(cons: np.ndarray) -> np.ndarray:
    return cons > BUY_THRESHOLD


def dump_models(models: list) -> bytes:
    buf = io.BytesIO()
    pickle.dump([m.to_state() for m in models], buf, protocol=4)
    return buf.getvalue()


def load_models(blob: bytes) -> list:
    return [from_state(s) for s in pickle.loads(blob)]


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()
