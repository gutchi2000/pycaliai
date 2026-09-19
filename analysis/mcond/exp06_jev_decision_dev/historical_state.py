# -*- coding: utf-8 -*-
"""
historical_state.py — 2023-2025のレース単位race-state vectorを構築する
================================================================================
2026-09-20、Stage B用。EXP05既存の時点安全設計行列(exp05_design.parquet)と、
train(2016-2021)のみでfit・sel(2022)でハイパラ選択する既存モデル(models.py)を
再利用する。EXP05-Fの凍結モデル(frozen_model.joblib、2016-2025全期間の最終fit)は
2024-2025評価にとって真のOOSにならないため使わない(STAGE_B_DATA_AUDIT.md参照)。

対応関係(EXP05既存モデル → Jevのstate概念):
  M1 (lp_cal_top3+f_mkt)                              → m1_top_prob 相当
  M3 (M1+v6_score+rank+gap×2+percentile+dispersion)   → m3_top_prob 相当
  M4 (offset(M3)+F_serve残差回帰)                      → m4_top_prob 相当

レース単位への集約はEXP05-Fのpredict_and_store.pyと同じ選び方
(top_i = argmax(p_m4)) で「候補馬」を選び、その馬の値をレースの代表値とする。

結果ラベル(top3/win/fin/fpay)はsupport構築には一切渡さない(呼び出し側の責務、
build_race_state_vectors()の戻り値には含めるが、ood_support.SupportModel.fit()へ
渡すのは CONTINUOUS_COLS/CATEGORICAL_COLS だけに限定すること)。

実行: このファイルは単体実行を想定しない。stage_b_dry_run.py 等から import して使う。
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
from analysis.mcond.evaluate import logit  # noqa: E402
from analysis.mcond.exp05_market_residual_dev import models as M  # noqa: E402

HERE = Path(__file__).resolve().parent
EXP05_DIR = BASE / "analysis" / "mcond" / "exp05_market_residual_dev"
D = BASE / "data" / "_research" / "mcond"
EPS = 1e-9

# 距離帯・クラス帯は固定境界(JRA慣行に基づく、データから閾値を選んでいない)。
# 結果を見て変更しない。
_DISTANCE_BINS = [0, 1400, 1800, 2200, 2800, 99999]
_DISTANCE_LABELS = ["sprint", "mile", "intermediate", "long", "extended"]
_CLASS_BINS = [-0.1, 3, 6, 9.1]  # c1__クラス名_ordは0-9の固定レンジ、レンジの三等分
_CLASS_LABELS = ["low", "mid", "high"]
_POPULARITY_BINS = [0, 3, 6, 999]
_POPULARITY_LABELS = ["1-3", "4-6", "7+"]

_VENUE_COLS = ["c1__場所__中山", "c1__場所__京都", "c1__場所__函館", "c1__場所__小倉",
              "c1__場所__新潟", "c1__場所__札幌", "c1__場所__東京", "c1__場所__福島",
              "c1__場所__阪神"]


def _entropy(p: np.ndarray) -> float:
    p = p[np.isfinite(p) & (p > 0)]
    if len(p) == 0:
        return float("nan")
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)))


def load_design_and_features() -> tuple[pd.DataFrame, dict]:
    df = pd.read_parquet(D / "exp05_design.parquet")
    feature_lists = json.loads((EXP05_DIR / "out" / "feature_lists.json").read_text(encoding="utf-8"))
    return df, feature_lists


def fit_oos_safe_predictions(df: pd.DataFrame, feature_lists: dict) -> dict[str, np.ndarray]:
    """EXP05のtrain/sel-fitモデルでM1/M3/M4相当の予測を全行分作る
    (2024-2025を含む全期間について、train(2016-2021)だけを見て学習しているため
    genuinely OOS)。target_col="top3"(fukusho_flag相当)固定。"""
    y = df["top3"].to_numpy()
    train = df["train"].to_numpy()
    sel = df["sel"].to_numpy()
    m0_m3 = M.fit_m0_m3(df, y, train, sel)
    offset3 = logit(np.clip(m0_m3["M3"]["pred"], EPS, 1 - EPS))
    m4 = M.fit_offset_residual(df, feature_lists["F_serve"], offset3, y, train, sel)
    return {"m1_pred": m0_m3["M1"]["pred"], "m3_pred": m0_m3["M3"]["pred"], "m4_pred": m4["pred"],
           "F_serve_cols": feature_lists["F_serve"]}


def _venue_name(row: pd.Series) -> str:
    for col in _VENUE_COLS:
        if row.get(col) == 1:
            return col.split("__")[-1]
    return "unknown"


def build_race_state_vectors(df: pd.DataFrame, preds: dict) -> pd.DataFrame:
    """レース単位(rid16)のstate vectorを作る。年度フィルタはしない(呼び出し側の責務)。
    戻り値には結果ラベル(top3/win/fin/fpay)も含むが、これらはsupport構築へは渡さないこと。"""
    d = df.copy()
    d["m1_pred"] = preds["m1_pred"]
    d["m3_pred"] = preds["m3_pred"]
    d["m4_pred"] = preds["m4_pred"]
    F_serve_cols = preds["F_serve_cols"]

    # F_serve欠損率(馬単位)
    fserve_arr = d[F_serve_cols].to_numpy(dtype=float)
    d["_feature_missing_rate"] = np.isnan(fserve_arr).mean(axis=1)

    # カテゴリ不明率: 歴史データセット(F_serveのone-hot列)からは再構成できないと判断し、
    # 一律0とする。当初「場所」one-hotグループ(9列)を「JRA中央9場を完全網羅、参照
    # カテゴリなし」と仮定して使ったが、実データ確認で「全列0」が全体の10.8%を占め、
    # これは中京競馬場(この9列に含まれない10番目のJRA場)の実際の開催シェアと一致した
    # (n-1ダミー方式で中京が参照カテゴリとして削除されているだけであり、真の未知カテゴリ
    # ではなかった)。他のone-hotグループ(天気・馬場状態等)も同様にn-1方式の疑いが強く、
    # 「全列0」から「真に未知」と「単に参照カテゴリ(最頻値)」を区別する情報が
    # F_serveの一部保存済み列には残っていない。EXP05-Fの前向きシステムは
    # frozen_encode.pyのfoundフラグでエンコード時点にこの区別を保持しているが、
    # 歴史データはこの情報が失われた後の状態しか残っていないため、正直に「不明」を
    # 一律0(算出不可)として扱う(STAGE_B_DATA_AUDIT.md参照、根拠のない代理値は使わない)。
    d["_unknown_category_rate"] = 0.0

    rows = []
    for rid, g in d.groupby("rid16", sort=False):
        top_i = int(np.argmax(g["m4_pred"].to_numpy()))
        cand = g.iloc[top_i]

        m1_arr, m3_arr, m4_arr = g["m1_pred"].to_numpy(), g["m3_pred"].to_numpy(), g["m4_pred"].to_numpy()
        mkt_arr = g["mkt_p3_pre"].to_numpy()

        def _rank_of(arr, i):
            return int(pd.Series(-arr).rank(method="first").iloc[i])

        ranks = [_rank_of(m1_arr, top_i), _rank_of(m3_arr, top_i), _rank_of(m4_arr, top_i)]
        top_probs = [float(m1_arr[top_i]), float(m3_arr[top_i]), float(m4_arr[top_i])]

        market_prob = float(mkt_arr[top_i]) if np.isfinite(mkt_arr[top_i]) else np.nan
        m4_top = top_probs[2]
        ai_market_divergence = (float(np.log(np.clip(m4_top, EPS, 1) / np.clip(market_prob, EPS, 1)))
                                if np.isfinite(market_prob) and market_prob > 0 else np.nan)

        dist_band = pd.cut([cand.get("c1__距離", np.nan)], _DISTANCE_BINS, labels=_DISTANCE_LABELS)[0]
        class_band = pd.cut([cand.get("c1__クラス名_ord", np.nan)], _CLASS_BINS, labels=_CLASS_LABELS)[0]
        pop_rank = cand.get("rank_mkt_pre", np.nan)
        pop_band = pd.cut([pop_rank], _POPULARITY_BINS, labels=_POPULARITY_LABELS)[0]

        rows.append({
            "rid16": rid, "year": int(cand["year"]),
            "m1_top_prob": top_probs[0], "m3_top_prob": top_probs[1], "m4_top_prob": top_probs[2],
            "entropy_m1": _entropy(m1_arr), "entropy_m3": _entropy(m3_arr), "entropy_m4": _entropy(m4_arr),
            "model_rank_disagreement": float(np.var(ranks)),
            "model_prob_variance": float(np.var(top_probs)),
            "market_prob": market_prob, "market_entropy": _entropy(mkt_arr),
            "ai_market_divergence": ai_market_divergence,
            "field_size": int(cand["n_field"]),
            "venue": _venue_name(cand), "surface": "turf" if cand.get("c1__芝・ダ__芝") == 1 else "dirt",
            "distance_band": str(dist_band) if pd.notna(dist_band) else "unknown",
            "class_band": str(class_band) if pd.notna(class_band) else "unknown",
            "popularity_band": str(pop_band) if pd.notna(pop_band) else "unknown",
            "feature_missing_rate": float(cand["_feature_missing_rate"]),
            "unknown_category_rate": float(cand["_unknown_category_rate"]),
            # 結果ラベル(support構築には使わない、prediction error計算専用)
            "_result_top3": int(cand["top3"]), "_result_win": int(cand["win"]),
            "_candidate_ban": int(cand["ban"]),
        })
    return pd.DataFrame(rows).set_index("rid16")
