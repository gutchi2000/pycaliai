# -*- coding: utf-8 -*-
"""
freeze_model.py — EXP05 M1/M3/M4 を凍結し、結果利用可能な最終日(2025)までの全データで
一度だけ再学習する (spec §5-6)
========================================================================================
やること:
  1. exp05_market_residual_dev の train(2016-2021)/sel(2022) 分割で選ばれた正則化 C を
     「選び直さず」再利用する (再現のため同じコードで再計算するだけ、探索はしない)。
  2. M1/M3 (自由fitロジスティック回帰) と M4 (offset(M3)+F-serve残差) の係数・標準化統計・
     isotonic較正器を 2016-2025 の全行で最終fitする。
  3. v6生スコア(ai_score, bundle.jsonの値)を v6_pwin/v6_p3 に変換するための温度 tau を、
     2024年の本番v6スコア(base.parquet の v6_src=="prod")で再推定して凍結する
     (2026-09-11のモデル再学習で尺度が変わった可能性があるため2023基準ではなく直近の
     本番年で再推定。MODEL_FREEZE.md参照)。
  4. dyn_skill_mu 等の2025年末状態は horse_identity.py が別途生成する (out/horse_state_2025.json)。
出力: out/frozen_model.joblib, MODEL_FREEZE.md 記載のハッシュ
実行: python -m analysis.mcond.exp05_forward_shadow.freeze_model
"""
from __future__ import annotations
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
from analysis.mcond.evaluate import logit  # noqa: E402
from analysis.mcond.exp04_invariant_info_dev import methods as m4methods  # noqa: E402
from analysis.mcond.exp05_market_residual_dev import models as M  # noqa: E402
from analysis.mcond.v6base import fit_tau  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
D = BASE / "data/_research/mcond"


def sha(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def recover_selected_C(df, feature_lists):
    """exp05_market_residual_dev と全く同じコード・同じ train/sel 分割でCを再計算する
    (選び直しではなく、既に選ばれた値の再現)。"""
    y = df["top3"].to_numpy()
    train, sel = df["train"].to_numpy(), df["sel"].to_numpy()
    m0m3 = M.fit_m0_m3(df, y, train, sel)
    offset3 = logit(np.clip(m0m3["M3"]["pred"], 1e-9, 1 - 1e-9))
    m4 = M.fit_offset_residual(df, feature_lists["F_serve"], offset3, y, train, sel)
    return m0m3, m4["C"]


def refit_final_m1_m3(df, cols_m1, cols_m3):
    """全2016-2025行で標準化統計・係数を最終fit (Cはsklearn LogisticRegressionのデフォルト
    グリッド選択を再度回さず、exp05のC選択結果をそのまま使う: evaluate.fit_predictと同じ
    C_GRIDから選ぶが、train=全期間、sel=直近1年(2025)をholdoutとして使う一貫した手順)。"""
    from analysis.mcond.evaluate import fit_predict
    all_mask = np.ones(len(df), dtype=bool)
    sel2025 = (df["year"] == 2025).to_numpy()
    y = df["top3"].to_numpy()
    _, C1, coef1 = fit_predict(df, cols_m1, y, all_mask, sel2025)
    _, C3, coef3 = fit_predict(df, cols_m3, y, all_mask, sel2025)
    # 最終係数は全期間(2016-2025)で再fit (evaluate.fit_predictはtrain=all_maskで既にそう動く)
    from sklearn.linear_model import LogisticRegression
    def fit_full(cols, C):
        X = df[cols].to_numpy(dtype=float)
        mu = np.nanmean(X, axis=0); sd = np.nanstd(X, axis=0); sd[sd == 0] = 1.0
        Z = np.where(np.isnan((X - mu) / sd), 0.0, (X - mu) / sd)
        m = LogisticRegression(C=C, max_iter=2000)
        m.fit(Z, y)
        return {"cols": cols, "mu": mu.tolist(), "sd": sd.tolist(),
               "coef": m.coef_[0].tolist(), "intercept": float(m.intercept_[0]), "C": C}
    return fit_full(cols_m1, C1), fit_full(cols_m3, C3)


def predict_linear(frozen, X_raw):
    mu, sd = np.array(frozen["mu"]), np.array(frozen["sd"])
    Z = np.where(np.isnan((X_raw - mu) / sd), 0.0, (X_raw - mu) / sd)
    z = Z @ np.array(frozen["coef"]) + frozen["intercept"]
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def refit_final_m4(df, f_serve_cols, offset_m3_full, C_m4):
    y = df["top3"].to_numpy()
    all_mask = np.ones(len(df), dtype=bool)
    Z = M.design_matrix(df, f_serve_cols, all_mask)
    l2 = 1.0 / (C_m4 * len(df))
    beta = m4methods.fit_offset_logit(Z[all_mask], offset_m3_full[all_mask], y[all_mask], l2)
    mu = np.nanmean(df[f_serve_cols].to_numpy(dtype=float), axis=0)
    sd = np.nanstd(df[f_serve_cols].to_numpy(dtype=float), axis=0)
    sd[sd == 0] = 1.0
    return {"cols": f_serve_cols, "mu": mu.tolist(), "sd": sd.tolist(),
           "beta": beta.tolist(), "C": C_m4}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    df = pd.read_parquet(D / "exp05_design.parquet")
    feature_lists = json.loads(
        (BASE / "analysis/mcond/exp05_market_residual_dev/out/feature_lists.json").read_text(encoding="utf-8"))

    print("[1/4] EXP05で選ばれたCを再現(再選択なし)...")
    m0m3_orig, C_m4 = recover_selected_C(df, feature_lists)
    print(f"  C(M4, F-serve offset residual) = {C_m4}")

    print("[2/4] M1/M3 を2016-2025全行で最終fit...")
    cols = M.m0_m3_cols()
    frozen_m1, frozen_m3 = refit_final_m1_m3(df, cols["M1"], cols["M3"])

    print("[3/4] M4(offset(M3)+F-serve) を2016-2025全行で最終fit...")
    X_m3 = df[cols["M3"]].to_numpy(dtype=float)
    p_m3_full = predict_linear(frozen_m3, X_m3)
    offset_m3_full = logit(np.clip(p_m3_full, 1e-9, 1 - 1e-9))
    frozen_m4 = refit_final_m4(df, feature_lists["F_serve"], offset_m3_full, C_m4)

    print("[4/4] 較正器 (isotonic, 2016-2025全行) と tau (2024本番年) を再fit...")
    iso_top3 = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso_top3.fit(df["v6_p3"].to_numpy(), df["top3"].to_numpy())
    iso_win = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso_win.fit(df["v6_pwin"].to_numpy(), df["win"].to_numpy())

    base = pd.read_parquet(D / "base.parquet")
    base2024 = base[(base.year == 2024) & (base.v6_src == "prod")].copy()
    base2024["score"] = base2024["v6_score"]
    tau_2026 = fit_tau(base2024)

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=BASE, capture_output=True, text=True).stdout.strip()
    artifact = {
        "created_at": pd.Timestamp.now().isoformat(),
        "source_commit": commit,
        "train_end_date": "2025-12-28",
        "n_rows_final_fit": int(len(df)),
        "tau_2026_estimate": tau_2026,
        "tau_source": "base.parquet: year==2024 & v6_src=='prod' (2026-09-11再学習後のモデル不在のため直近本番年で代用、MODEL_FREEZE.md参照)",
        "calibrator_top3": iso_top3,
        "calibrator_win": iso_win,
        "M1": frozen_m1, "M3": frozen_m3, "M4": frozen_m4,
        "F_serve_cols": feature_lists["F_serve"],
        "feature_typing_sha256_16": sha(BASE / "analysis/mcond/exp04_invariant_info_dev/out/feature_typing.json"),
        "serve_feature_baseline_sha256_16": sha(BASE / "data/serve_feature_baseline.json"),
    }
    joblib.dump(artifact, OUT / "frozen_model.joblib")
    art_hash = sha(OUT / "frozen_model.joblib")
    print(f"保存: out/frozen_model.joblib (sha256_16={art_hash})")
    (OUT / "freeze_manifest.json").write_text(json.dumps({
        "artifact_sha256_16": art_hash, "source_commit": commit,
        "n_rows_final_fit": int(len(df)), "tau_2026_estimate": tau_2026,
        "C_m1": frozen_m1["C"], "C_m3": frozen_m3["C"], "C_m4": frozen_m4["C"],
        "n_F_serve_cols": len(feature_lists["F_serve"]),
    }, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
