# -*- coding: utf-8 -*-
"""
model_v3.py — 馬券判断 AI v3 の推定器（仕様 v1.0 §2）
=====================================================
- 基準の平均払戻 R̄_cell: 学習データで、セル = 券種 × 判断時点オッズ十分位（境界も学習データで決める）ごとの平均払戻倍率。
- 目標 y = R − R̄_cell。上側を学習データの 99.9% 点で打ち切る（下側は打ち切らない）。
- 二乗損失の LightGBM 回帰で y を推定。採算 E[R] = R̄_cell + ŷ。
- 不確かさ: 開催日単位の bootstrap（開催日を復元抽出して行を複製）でモデル 10 本。仕様 v1.1 §6 により、R̄_cell・境界・
  上側打ち切り点は bootstrap の各回で、その回の学習データから作り直す（基準の推定誤差も不確かさに含める）。
  各モデルは自分の基準を持ち、E[R] = その回の R̄_cell + その回の ŷ。
  控えめな採算 = 10 本の E[R] の下から 20% 点（numpy.quantile の線形補間）。
- 購入規則: 控えめな採算 > 1.00 の候補を 100 円で買う。
- ハイパーパラメータ: 学習データの開催日を 3 分割した交差検証で、保持側の y の二乗誤差（全保持行の平均）が最小の設定。
  各分割では、境界・R̄_cell・打ち切り点を学習側だけで作り直し、保持側の y もその値で作る。候補は GRID の 12 個。
"""
from __future__ import annotations

import hashlib
import io
import itertools
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd

from analysis.bet_decision_ai.v3.data_v3 import CAT_COLS, odds_cell, odds_decile_edges

N_BOOT = 10
BOOT_Q = 0.20
BUY_THRESHOLD = 1.00
CLIP_HI_Q = 0.999
KFOLD_SELECT = 3
NCELL = 20
NUM_THREADS = 16

# 候補 12 個（事前登録で固定）。二乗損失は裾の重い y で分散が大きいので、v2 の Huber 用より浅い木・大きい葉・弱い学習まで含める。
GRID = [{"num_leaves": nl, "min_data_in_leaf": md, "learning_rate": lr, "n_trees": nt}
        for nl, md, (lr, nt) in itertools.product([7, 31], [1000, 3000], [(0.02, 150), (0.02, 500), (0.05, 500)])]
assert len(GRID) <= 12

_COMMON = {"verbose": -1, "num_threads": NUM_THREADS, "feature_fraction": 0.8, "bagging_fraction": 0.8,
           "bagging_freq": 1, "lambda_l2": 1.0, "max_bin": 255, "deterministic": True, "force_row_wise": True}


def _params(cfg: dict, seed: int) -> dict:
    p = dict(_COMMON)
    p.update(objective="regression", num_leaves=cfg["num_leaves"], min_data_in_leaf=cfg["min_data_in_leaf"],
             learning_rate=cfg["learning_rate"], seed=seed, bagging_seed=seed, feature_fraction_seed=seed,
             data_random_seed=seed)
    return p


def _cat(X: pd.DataFrame) -> list[str]:
    return [c for c in CAT_COLS if c in X.columns]


def _train(X: pd.DataFrame, y: np.ndarray, cfg: dict, seed: int) -> lgb.Booster:
    ds = lgb.Dataset(X, label=y, categorical_feature=_cat(X), free_raw_data=True)
    return lgb.train(_params(cfg, seed), ds, num_boost_round=cfg["n_trees"])


# ================================================================ 基準（R̄_cell・境界・上側打ち切り点）
class CellBase:
    def __init__(self, edges: dict, rbar: np.ndarray, clip_hi: float, n_cell: np.ndarray):
        self.edges = edges
        self.rbar = np.asarray(rbar, float)
        self.clip_hi = float(clip_hi)
        self.n_cell = np.asarray(n_cell, int)

    @classmethod
    def fit(cls, X: pd.DataFrame, R: np.ndarray) -> "CellBase":
        edges = odds_decile_edges(X)
        cell = odds_cell(X, edges)
        n = np.bincount(cell, minlength=NCELL)
        assert (n > 0).all(), n
        rbar = np.bincount(cell, weights=R, minlength=NCELL) / n
        y = R - rbar[cell]
        return cls(edges, rbar, float(np.quantile(y, CLIP_HI_Q)), n)

    def cell(self, X: pd.DataFrame) -> np.ndarray:
        return odds_cell(X, self.edges)

    def target(self, X: pd.DataFrame, R: np.ndarray) -> np.ndarray:
        return np.minimum(R - self.rbar[self.cell(X)], self.clip_hi)

    def to_state(self) -> dict:
        return {"edges": self.edges, "rbar": self.rbar.tolist(), "clip_hi": self.clip_hi, "n_cell": self.n_cell.tolist()}

    @classmethod
    def from_state(cls, s: dict) -> "CellBase":
        return cls(s["edges"], s["rbar"], s["clip_hi"], s["n_cell"])


class SqExcess:
    """y = R − R̄_cell（上側打ち切り）の二乗損失 LightGBM。E[R] = R̄_cell + ŷ。"""
    kind = "sq_excess"

    def __init__(self):
        self.m = None
        self.base: CellBase | None = None
        self.cfg: dict = {}
        self.info: dict = {}

    def fit(self, X: pd.DataFrame, R: np.ndarray, base: CellBase, cfg: dict, seed: int) -> "SqExcess":
        self.cfg, self.base = dict(cfg), base
        y = base.target(X, R)
        self.m = _train(X, y, cfg, seed)
        self.info = {"n_rows": int(len(X)), "y_mean": float(y.mean()), "n_clipped": int((R - base.rbar[base.cell(X)]
                                                                                         > base.clip_hi).sum())}
        return self

    def predict_excess(self, X: pd.DataFrame) -> np.ndarray:
        return self.m.predict(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.base.rbar[self.base.cell(X)] + self.m.predict(X)

    def to_state(self) -> dict:
        return {"kind": self.kind, "cfg": self.cfg, "m": self.m.model_to_string(), "base": self.base.to_state(),
                "info": self.info}

    @classmethod
    def from_state(cls, s: dict) -> "SqExcess":
        o = cls()
        o.cfg, o.info = s["cfg"], s["info"]
        o.base = CellBase.from_state(s["base"])
        o.m = lgb.Booster(model_str=s["m"])
        return o


# ================================================================ bootstrap 10 本
def boot_index(day: np.ndarray, b: int, base_seed: int) -> np.ndarray:
    udays, inv = np.unique(day, return_inverse=True)
    rng = np.random.default_rng(base_seed * 1000 + b)
    cnt = np.bincount(rng.integers(0, len(udays), len(udays)), minlength=len(udays))
    rows_by_day = np.split(np.argsort(inv, kind="stable"), np.cumsum(np.bincount(inv, minlength=len(udays)))[:-1])
    return np.concatenate([np.tile(rows_by_day[k], cnt[k]) for k in np.flatnonzero(cnt)])


def fit_bootstrap(X: pd.DataFrame, R: np.ndarray, day: np.ndarray, cfg: dict, base_seed: int = 0,
                  n_boot: int = N_BOOT, log=None) -> list:
    """bootstrap の各回で、その回の学習データから基準（境界・R̄_cell・打ち切り点）を作り直す（仕様 v1.1 §6）。"""
    models = []
    for b in range(n_boot):
        idx = boot_index(day, b, base_seed)
        Xb = X.iloc[idx].reset_index(drop=True)
        base = CellBase.fit(Xb, R[idx])
        m = SqExcess().fit(Xb, R[idx], base, cfg, seed=base_seed * 100 + b)
        m.info["boot"] = {"b": b, "n_rows": int(len(idx)), "n_unique_days": int(len(np.unique(day[idx])))}
        models.append(m)
        if log:
            log(f"      boot {b + 1}/{n_boot} rows={len(idx):,}")
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
    return [SqExcess.from_state(s) for s in pickle.loads(blob)]


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ================================================================ ハイパーパラメータ: 開催日 3 分割 CV の y の二乗誤差
def day_folds(day: np.ndarray, k: int, seed: int) -> np.ndarray:
    udays = np.unique(day)
    rng = np.random.default_rng(20_000 + seed)
    perm = rng.permutation(len(udays))
    f = dict(zip(udays[perm], np.arange(len(udays)) % k))
    return np.array([f[x] for x in day])


def select_cfg_cv(X: pd.DataFrame, R: np.ndarray, day: np.ndarray, seed: int = 0, log=None) -> tuple[dict, list]:
    fold = day_folds(day, KFOLD_SELECT, seed)
    bases = []
    for f in range(KFOLD_SELECT):
        trm = fold != f
        bases.append(CellBase.fit(X[trm], R[trm]))
    rows = []
    for ci, cfg in enumerate(GRID):
        tot, n, per = 0.0, 0, []
        for f in range(KFOLD_SELECT):
            trm, tem = fold != f, fold == f
            b = bases[f]
            m = SqExcess().fit(X[trm].reset_index(drop=True), R[trm], b, cfg, seed=seed)
            yte = b.target(X[tem], R[tem])
            L = (yte - m.predict_excess(X[tem])) ** 2
            per.append(float(L.mean()))
            tot += float(L.sum())
            n += int(tem.sum())
        rows.append({"cfg_id": ci, **cfg, "cv_mse": tot / n, "fold_mse": per})
        if log:
            log(f"    cv cfg{ci:02d} {cfg} mse={tot / n:.6f}")
    # 参考: 定数（学習側の y の平均）だけの予測の CV 二乗誤差
    tot0 = 0.0
    for f in range(KFOLD_SELECT):
        trm, tem = fold != f, fold == f
        b = bases[f]
        c = float(b.target(X[trm], R[trm]).mean())
        tot0 += float(((b.target(X[tem], R[tem]) - c) ** 2).sum())
    best = min(rows, key=lambda r: (r["cv_mse"], r["cfg_id"]))
    rows.append({"cfg_id": -1, "note": "reference: constant prediction", "cv_mse": tot0 / len(R)})
    return GRID[best["cfg_id"]], rows


def oof_conservative(X: pd.DataFrame, R: np.ndarray, day: np.ndarray, cfg: dict, seed: int = 0, log=None) -> dict:
    """記述用の交差予測: 開催日 3 分割（select_cfg_cv と同じ分割）の各保持側を、残り 2 つで学習した bootstrap 10 本で予測。"""
    fold = day_folds(day, KFOLD_SELECT, seed)
    cons = np.full(len(R), np.nan)
    mean = np.full(len(R), np.nan)
    rbar = np.full(len(R), np.nan)
    for f in range(KFOLD_SELECT):
        trm, tem = fold != f, fold == f
        Xtr = X[trm].reset_index(drop=True)
        b = CellBase.fit(Xtr, R[trm])
        ms = fit_bootstrap(Xtr, R[trm], day[trm], cfg, base_seed=0)
        P = predict_ensemble(ms, X[tem])
        cons[tem] = conservative(P)
        mean[tem] = P.mean(0)
        rbar[tem] = b.rbar[b.cell(X[tem])]
        if log:
            log(f"    oof fold {f}: n={int(tem.sum()):,} buys={int((cons[tem] > BUY_THRESHOLD).sum())}")
    return {"fold": fold, "cons": cons, "mean": mean, "rbar_fold": rbar}
