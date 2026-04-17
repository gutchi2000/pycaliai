"""
train_trifecta_model.py
=======================
三連単フォーメーション専用モデル (trifecta_model_v1) 学習スクリプト。

アーキテクチャ:
  1. LightGBM LambdaRank で各馬の「レース内ランキングスコア」を学習
  2. Plackett-Luce で p(f,s,t) = p1(f) × p2(s|f) × p3(t|f,s) を計算
  3. 三連単フォーメーション ROI を最終評価指標として報告

特徴量（オッズ一切不使用 = リーク防止）:
  - ensemble_prob (既存アンサンブル確率)
  - race-relative: prob_rank, prob_z, gap_top1, gap_top2, field_size, entropy, stddev
  - mark flags: is_honmei, is_taikou, is_sabo, is_delta

データ分割:
  - Train : 2024 1月〜9月
  - Val   : 2024 10月〜12月
  - OOS   : 2025 (全年)

使い方:
  python train_trifecta_model.py          # 学習 + 評価
  python train_trifecta_model.py --eval   # 保存済みモデルで評価のみ
"""
from __future__ import annotations

import argparse
import io
import itertools
import json
import sys
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import softmax

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

BASE          = Path(__file__).parent
PRED_CSV      = BASE / "reports/ensemble_predictions.csv"
KEKKA_CSV     = BASE / "data/kekka_20160105_20251228_v2.csv"
OUT_MODEL     = BASE / "models/trifecta_model_v1.pkl"
OUT_JSON      = BASE / "data/trifecta_model_meta.json"

BUDGET        = 9600
EXCLUDE_PLACES = {"東京", "小倉"}

# 2024 の月区切り (YYYYMM)
TRAIN_MONTHS_MAX = 202409   # ～9月 train
VAL_MONTHS_MIN   = 202410   # 10-12月 val
OOS_YEAR         = 2025


# ============================================================
# データ読み込み
# ============================================================
def load_data() -> pd.DataFrame:
    """ensemble_predictions + kekka を結合してレース×馬 単位の DataFrame を返す。"""
    print("ensemble_predictions 読み込み...")
    pred = pd.read_csv(PRED_CSV, encoding="utf-8-sig")
    pred.columns = ["race_id", "umaban", "horse_name", "ensemble_prob", "mark", "fukusho_flag"]
    pred["race_id"] = pred["race_id"].astype(str)
    pred["umaban"]  = pd.to_numeric(pred["umaban"], errors="coerce").astype("Int64")
    pred["mark"]    = pred["mark"].fillna("").astype(str)
    pred["year"]    = pred["race_id"].str[:4].astype(int)
    pred["ym"]      = pred["race_id"].str[:6].astype(int)
    print(f"  {pred['race_id'].nunique():,} races, {len(pred):,} rows")

    print("kekka 読み込み...")
    kk = pd.read_csv(KEKKA_CSV, encoding="cp932", dtype=str)
    kk["race_id"]    = kk["レースID(新)"].astype(str).str[:16]
    kk["umaban"]     = pd.to_numeric(kk["馬番"], errors="coerce").astype("Int64")
    kk["jyun"]       = pd.to_numeric(kk["確定着順"], errors="coerce")
    kk["santan_pay"] = pd.to_numeric(kk["３連単"],    errors="coerce")
    kk["place"]      = kk["場所"].astype(str)
    # レース単位で三連単配当を1着馬のみ保持（先頭）
    kk_race = (
        kk.sort_values("jyun")
          .groupby("race_id", as_index=False)
          .first()[["race_id", "place", "santan_pay"]]
          .rename(columns={"santan_pay": "race_santan_pay"})
    )
    kk_horse = kk[["race_id", "umaban", "jyun"]].copy()

    print("  結合...")
    df = pred.merge(kk_horse, on=["race_id", "umaban"], how="inner")
    df = df.merge(kk_race,   on="race_id",              how="inner")
    df = df[~df["place"].isin(EXCLUDE_PLACES)]
    df = df.dropna(subset=["jyun", "ensemble_prob"])
    print(f"  結合後: {df['race_id'].nunique():,} races, {len(df):,} rows")
    return df


# ============================================================
# 特徴量エンジニアリング
# ============================================================
def add_race_features(df: pd.DataFrame) -> pd.DataFrame:
    """レース内相対特徴量を追加する。"""
    df = df.copy()

    # 基本 mark flags
    df["is_honmei"] = (df["mark"] == "◎").astype(int)
    df["is_taikou"] = (df["mark"] == "◯").astype(int)
    df["is_sabo"]   = (df["mark"] == "▲").astype(int)
    df["is_delta"]  = (df["mark"] == "△").astype(int)

    # レース内統計
    grp = df.groupby("race_id")["ensemble_prob"]
    df["race_prob_max"]  = grp.transform("max")
    df["race_prob_2nd"]  = grp.transform(lambda x: x.nlargest(2).iloc[-1] if len(x) >= 2 else x.max())
    df["race_prob_mean"] = grp.transform("mean")
    df["race_prob_std"]  = grp.transform("std").fillna(0)
    df["field_size"]     = grp.transform("count")

    # レース内 z スコア
    df["prob_z"] = np.where(
        df["race_prob_std"] > 0,
        (df["ensemble_prob"] - df["race_prob_mean"]) / df["race_prob_std"],
        0.0
    )

    # トップとの差
    df["gap_top1"] = df["race_prob_max"] - df["ensemble_prob"]
    df["gap_top2"] = df["race_prob_2nd"] - df["ensemble_prob"]

    # レース内順位 (1=highest prob)
    df["prob_rank"] = grp.rank(ascending=False, method="min")

    # Shannon エントロピー (per race)
    def _entropy(probs: pd.Series) -> float:
        p = probs / probs.sum()
        return float(-np.sum(p * np.log(p + 1e-12)))

    ent_map = df.groupby("race_id")["ensemble_prob"].apply(_entropy)
    df["race_entropy"] = df["race_id"].map(ent_map)

    # 上位 2 頭スコア差 (レース単位)
    df["race_gap_12"] = df["race_prob_max"] - df["race_prob_2nd"]

    # ランキングラベル: 1着=2, 2-3着=1, 4着以降=0
    df["label"] = 0
    df.loc[df["jyun"] <= 3, "label"] = 1
    df.loc[df["jyun"] == 1, "label"] = 2

    return df


FEATURE_COLS = [
    "ensemble_prob",
    "prob_rank",
    "prob_z",
    "gap_top1",
    "gap_top2",
    "field_size",
    "race_prob_std",
    "race_entropy",
    "race_gap_12",
    "is_honmei",
    "is_taikou",
    "is_sabo",
    "is_delta",
]


# ============================================================
# Plackett-Luce ユーティリティ
# ============================================================
def pl_combo_probs(scores: dict[int, float],
                   top_n: int = 6) -> list[tuple[tuple[int, int, int], float]]:
    """Plackett-Luce で三連単コンボ確率を計算する。

    Args:
        scores: {umaban: raw_ranking_score}
        top_n:  上位 N 頭のみで組合せを生成 (計算量削減)

    Returns:
        [(combo, probability), ...] 確率降順ソート済み
    """
    if len(scores) < 3:
        return []

    horses = sorted(scores, key=lambda h: scores[h], reverse=True)[:top_n]
    raw = np.array([scores[h] for h in horses], dtype=float)
    # softmax で正規化
    p = softmax(raw)

    results = []
    for f, s, t in itertools.permutations(range(len(horses)), 3):
        # Plackett-Luce: p(f wins) × p(s wins | f removed) × p(t wins | f,s removed)
        denom_2 = 1.0 - p[f]
        denom_3 = denom_2 - p[s]
        if denom_2 <= 0 or denom_3 <= 0:
            continue
        prob = p[f] * (p[s] / denom_2) * (p[t] / denom_3)
        results.append(((horses[f], horses[s], horses[t]), float(prob)))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


# ============================================================
# シミュレーション（ROI 評価）
# ============================================================
def simulate_roi(races: list[dict], model: lgb.Booster,
                 top_combos: int = 9) -> dict:
    """trifecta_model_v1 を使った三連単フォーメーション ROI を計算する。"""
    total_bet    = 0
    total_return = 0
    hits         = 0
    n_races      = 0

    for r in races:
        # モデルスコア取得
        X = pd.DataFrame([
            {f: row.get(f, 0) for f in FEATURE_COLS}
            for row in r["horses"]
        ])
        if X.empty:
            continue
        model_scores = model.predict(X)
        umabans = [row["umaban"] for row in r["horses"]]
        score_map = {ub: float(s) for ub, s in zip(umabans, model_scores)}

        combos_with_prob = pl_combo_probs(score_map, top_n=min(8, len(score_map)))
        if not combos_with_prob:
            continue

        # 上位 top_combos コンボを選択
        selected = [c for c, _ in combos_with_prob[:top_combos]]
        if not selected:
            continue

        n = len(selected)
        per_bet = max(100, (BUDGET // n // 100) * 100)
        total_bet += per_bet * n
        n_races   += 1

        # 的中判定
        result_top3 = r.get("result_top3")
        pay         = r.get("santan_pay", 0)
        if result_top3 and pay > 0:
            act_f, act_s, act_t = result_top3
            for (f, s, t) in selected:
                if f == act_f and s == act_s and t == act_t:
                    total_return += per_bet * pay / 100
                    hits += 1
                    break

    roi = (total_return / total_bet * 100) if total_bet > 0 else 0.0
    return {"roi": roi, "hits": hits, "n_races": n_races,
            "total_bet": total_bet, "total_return": total_return}


# ============================================================
# レース辞書の構築
# ============================================================
def build_race_list(df: pd.DataFrame) -> list[dict]:
    """DataFrame → [{race_id, horses:[{umaban, features...}], result_top3, santan_pay}] 変換。"""
    races = []
    for rid, gdf in df.groupby("race_id"):
        if len(gdf) < 3:
            continue
        place = gdf["place"].iloc[0]
        if place in EXCLUDE_PLACES:
            continue

        # 実際の 1-3 着
        top3 = gdf.dropna(subset=["jyun"]).sort_values("jyun").head(3)
        if len(top3) < 3:
            continue
        result = (int(top3.iloc[0]["umaban"]),
                  int(top3.iloc[1]["umaban"]),
                  int(top3.iloc[2]["umaban"]))
        pay = float(gdf["race_santan_pay"].iloc[0]) if pd.notna(gdf["race_santan_pay"].iloc[0]) else 0

        horses = []
        for _, row in gdf.iterrows():
            h = {f: float(row.get(f, 0)) for f in FEATURE_COLS}
            h["umaban"] = int(row["umaban"])
            horses.append(h)

        races.append({
            "race_id":      rid,
            "horses":       horses,
            "result_top3":  result,
            "santan_pay":   pay,
        })
    return races


# ============================================================
# LightGBM LambdaRank 学習
# ============================================================
def train_model(df_train: pd.DataFrame,
                df_val:   pd.DataFrame) -> lgb.Booster:
    print("LightGBM LambdaRank 学習...")

    def _make_dataset(df: pd.DataFrame, label: str = ""):
        # race_id でソートしてからグループを計算（順序整合性を保証）
        df = df.sort_values("race_id").reset_index(drop=True)
        X      = df[FEATURE_COLS].fillna(0).values
        y      = df["label"].values.astype(int)
        # row-order で group サイズを計算
        from itertools import groupby as _grpby
        groups = np.array([len(list(g)) for _, g in _grpby(df["race_id"])])
        ds     = lgb.Dataset(X, label=y, group=groups, free_raw_data=False)
        return ds, X, y, groups, df

    ds_train, X_tr, y_tr, g_tr, df_train = _make_dataset(df_train, "train")
    ds_val,   X_vl, y_vl, g_vl, df_val   = _make_dataset(df_val,   "val")

    params = {
        "objective":                  "lambdarank",
        "lambdarank_truncation_level": 3,
        "metric":                     "ndcg",
        "eval_at":                    [1, 3],
        "learning_rate":              0.01,   # 低めにして収束を緩やかに
        "num_leaves":                 31,
        "max_depth":                  5,
        "min_data_in_leaf":           10,
        "feature_fraction":           0.9,
        "bagging_fraction":           0.9,
        "bagging_freq":               5,
        "lambda_l1":                  0.05,
        "lambda_l2":                  0.05,
        "verbose":                   -1,
        "n_jobs":                    -1,
        "seed":                      42,
    }

    callbacks = [
        lgb.early_stopping(100, verbose=False),  # 100 round patience
        lgb.log_evaluation(period=200),
    ]

    model = lgb.train(
        params,
        ds_train,
        num_boost_round=1000,
        valid_sets=[ds_val],
        callbacks=callbacks,
    )

    print(f"  best iteration: {model.best_iteration}")
    return model


# ============================================================
# メイン
# ============================================================
def main(eval_only: bool = False) -> None:
    print("=" * 60)
    print("trifecta_model_v1  学習/評価")
    print("=" * 60)

    # ── データ読み込み ──────────────────────────────────────
    df = load_data()
    df = add_race_features(df)

    # ── 分割 ────────────────────────────────────────────────
    df_2024 = df[df["year"] == 2024].copy()
    df_2025 = df[df["year"] == OOS_YEAR].copy()

    df_train = df_2024[df_2024["ym"] <= TRAIN_MONTHS_MAX].copy()
    df_val   = df_2024[df_2024["ym"] >  TRAIN_MONTHS_MAX].copy()
    df_oos   = df_2025.copy()

    print()
    print(f"Train (2024 1-9月): {df_train['race_id'].nunique():,} races, {len(df_train):,} rows")
    print(f"Val   (2024 10-12月): {df_val['race_id'].nunique():,} races, {len(df_val):,} rows")
    print(f"OOS   (2025 全年):  {df_oos['race_id'].nunique():,} races, {len(df_oos):,} rows")

    # ── 学習 or ロード ────────────────────────────────────────
    if eval_only:
        if not OUT_MODEL.exists():
            print(f"[ERROR] モデルが見つかりません: {OUT_MODEL}")
            return
        print(f"\n保存済みモデルを読み込み: {OUT_MODEL}")
        model = joblib.load(OUT_MODEL)["model"]
    else:
        print()
        model = train_model(df_train, df_val)
        # 保存
        OUT_MODEL.parent.mkdir(exist_ok=True)
        payload = {
            "model":        model,
            "feature_cols": FEATURE_COLS,
            "description":  "trifecta_model_v1: LightGBM LambdaRank + Plackett-Luce",
        }
        joblib.dump(payload, OUT_MODEL)
        print(f"\nモデル保存: {OUT_MODEL}")

    # ── ROI 評価 ─────────────────────────────────────────────
    print()
    print("=== ROI 評価 (top_combos=9) ===")
    for label, sub_df in [("Train 2024/1-9",   df_train),
                           ("Val   2024/10-12", df_val),
                           ("OOS   2025",       df_oos)]:
        races = build_race_list(sub_df)
        if not races:
            print(f"  {label}: レースなし")
            continue
        res = simulate_roi(races, model, top_combos=9)
        print(f"  {label}: ROI={res['roi']:6.2f}%  hits={res['hits']}/{res['n_races']}  "
              f"bet=¥{res['total_bet']:,.0f}  ret=¥{res['total_return']:,.0f}")

    # top_combos を変えて感度分析
    print()
    print("=== OOS 2025 top_combos 感度分析 ===")
    races_oos = build_race_list(df_oos)
    for tc in [3, 6, 9, 12, 15]:
        res = simulate_roi(races_oos, model, top_combos=tc)
        print(f"  top_combos={tc:2d}: ROI={res['roi']:6.2f}%  hits={res['hits']}/{res['n_races']}")

    # ── メタデータ保存 ────────────────────────────────────────
    if not eval_only:
        # 最適 top_combos を ROI 最大で記録
        best_tc  = max(range(3, 16, 3),
                       key=lambda tc: simulate_roi(races_oos, model, top_combos=tc)["roi"])
        res_best = simulate_roi(races_oos, model, top_combos=best_tc)
        meta = {
            "feature_cols":    FEATURE_COLS,
            "best_top_combos": best_tc,
            "oos_roi":         round(res_best["roi"], 4),
            "oos_hits":        res_best["hits"],
            "oos_n_races":     res_best["n_races"],
            "train_months":    f"2024/01-{TRAIN_MONTHS_MAX % 100:02d}",
            "val_months":      "2024/10-12",
            "oos_year":        OOS_YEAR,
        }
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"\nメタデータ保存: {OUT_JSON}")
        print(f"best top_combos={best_tc}  OOS ROI={res_best['roi']:.2f}%")


# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true",
                        help="保存済みモデルで評価のみ（再学習しない）")
    args = parser.parse_args()
    main(eval_only=args.eval)
