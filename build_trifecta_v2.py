"""
build_trifecta_v2.py
=====================
trifecta_model_v2 を再現性のある手順で構築する。

目的: trifecta_model_v1 は 2024/1-9月 学習・2024/10-12月 val で、backtest (2024-2025)
と学習期間が完全に重なりリークしていた。v2 は以下の方針でリークを除去する。

  Split:
    train = 2023/1-9月  (base models の valid 期間, 真の OOS)
    val   = 2023/10-12月
    test  = 2024 全年 (OOS)
    test2 = 2025 全年 (OOS)

  確率生成:
    - master_kako5.csv を使用 (backtest_v2 と同一)
    - backtest.ensemble_predict_batch (LGBM/CatBoost/Transformer 重み付き平均)
    - ensemble_calibrator_v1.pkl (valid=2023 fit, リーク無し)

決定的 seed (random_state=42, lgb seed=42) で実行すれば同じモデルが再現される。

実行:
    python build_trifecta_v2.py

出力:
    models/trifecta_model_v2.pkl
    data/trifecta_model_v2_meta.json
    reports/ensemble_predictions_2023_2025.csv  (再利用用)
"""
from __future__ import annotations

import io
import itertools
import json
import sys
import warnings
from pathlib import Path
from unittest.mock import MagicMock

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import softmax

warnings.filterwarnings("ignore")

# streamlit stub (app.py import 対策は train_trifecta_model と同じ発想で不要; backtest を直接使う)
class _St:
    def __init__(self):
        self.cache_data=self._pt; self.cache_resource=self._pt; self.cache=self._pt; self.session_state={}
    def _pt(self,*a,**k):
        if len(a)==1 and callable(a[0]) and not k: return a[0]
        def w(fn): return fn
        return w
    def __getattr__(self,n):
        m=MagicMock(); setattr(self,n,m); return m
sys.modules['streamlit']=_St()
sys.modules.setdefault('shap', MagicMock())

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

import backtest as bt  # master_kako5.csv 使用版

MASTER_CSV  = BASE / "data/master_kako5.csv"
KEKKA_CSV   = BASE / "data/kekka_20160105_20251228_v2.csv"
CAL_PATH    = BASE / "models/ensemble_calibrator_v1.pkl"   # valid=2023 fit, クリーン
OUT_MODEL   = BASE / "models/trifecta_model_v2.pkl"
OUT_META    = BASE / "data/trifecta_model_v2_meta.json"
OUT_PRED    = BASE / "reports/ensemble_predictions_2023_2025.csv"

COL_RID     = "レースID(新/馬番無)"
COL_BAN     = "馬番"
COL_DATE    = "日付"
COL_NAME    = "馬名"
COL_JYUN    = "着順"
COL_PLACE   = "場所"
BUDGET      = 9600
EXCLUDE_PLACES = {"東京", "小倉"}
RANDOM_SEED = 42

# ============================================================
# 1. 予測生成 (2023-2025)
# ============================================================
def generate_predictions() -> pd.DataFrame:
    """master_kako5 から 2023-2025 の ensemble_prob (calibrated) を生成する。"""
    if OUT_PRED.exists():
        print(f"[cache] 既存の予測 CSV を再利用: {OUT_PRED}")
        df = pd.read_csv(OUT_PRED, encoding="utf-8-sig", low_memory=False)
        df["race_id"] = df["race_id"].astype(str)
        return df

    print(f"[1/3] master 読み込み: {MASTER_CSV}")
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    df[COL_DATE] = pd.to_numeric(df[COL_DATE], errors="coerce")

    # 2023-2025 全件
    mask = (df[COL_DATE] >= 20230101) & (df[COL_DATE] < 20260101)
    sub = df[mask].copy().reset_index(drop=True)
    print(f"  対象: {len(sub):,} 行 / {sub[COL_RID].nunique():,} races")

    print("[2/3] アンサンブル予測 (LGBM + CatBoost + Transformer)...")
    raw_prob = bt.ensemble_predict_batch(sub)

    print("[3/3] 較正 (ensemble_calibrator_v1 / valid=2023 fit)...")
    cal_obj = joblib.load(CAL_PATH)
    cal = cal_obj["calibrator"] if isinstance(cal_obj, dict) else cal_obj
    cal_prob = cal.transform(raw_prob)

    # 印 (レース内 rank 1-5 → ◎◯▲△× それ以外 '')
    sub["ensemble_prob_raw"] = raw_prob
    sub["ensemble_prob"]      = cal_prob
    sub["prob_rank_race"]     = sub.groupby(COL_RID)["ensemble_prob"].rank(ascending=False, method="first")

    mark_by_rank = {1: "◎", 2: "◯", 3: "▲", 4: "△", 5: "×"}
    sub["mark"] = sub["prob_rank_race"].map(mark_by_rank).fillna("")

    # fukusho_flag は着順 ≤ 3
    sub["fukusho_flag"] = (pd.to_numeric(sub[COL_JYUN], errors="coerce") <= 3).astype(int)

    out = sub[[COL_RID, COL_BAN, COL_NAME, "ensemble_prob", "mark", "fukusho_flag"]].copy()
    out.columns = ["race_id", "umaban", "horse_name", "ensemble_prob", "mark", "fukusho_flag"]
    out["race_id"] = out["race_id"].astype(str)

    OUT_PRED.parent.mkdir(exist_ok=True)
    out.to_csv(OUT_PRED, index=False, encoding="utf-8-sig")
    print(f"  saved: {OUT_PRED} ({len(out):,} rows)")
    return out


# ============================================================
# 2. kekka merge + features
# ============================================================
def load_and_merge(pred: pd.DataFrame) -> pd.DataFrame:
    print("\n[kekka 読み込み + 結合]")
    pred = pred.copy()
    pred["umaban"] = pd.to_numeric(pred["umaban"], errors="coerce").astype("Int64")
    pred["year"]   = pred["race_id"].str[:4].astype(int)
    pred["ym"]     = pred["race_id"].str[:6].astype(int)

    kk = pd.read_csv(KEKKA_CSV, encoding="cp932", dtype=str)
    kk["race_id"]    = kk["レースID(新)"].astype(str).str[:16]
    kk["umaban"]     = pd.to_numeric(kk["馬番"], errors="coerce").astype("Int64")
    kk["jyun"]       = pd.to_numeric(kk["確定着順"], errors="coerce")
    kk["santan_pay"] = pd.to_numeric(kk["３連単"], errors="coerce")
    kk["place"]      = kk["場所"].astype(str)

    kk_race = (
        kk.sort_values("jyun").groupby("race_id", as_index=False).first()
          [["race_id", "place", "santan_pay"]]
          .rename(columns={"santan_pay": "race_santan_pay"})
    )
    kk_horse = kk[["race_id", "umaban", "jyun"]]

    df = pred.merge(kk_horse, on=["race_id", "umaban"], how="inner")
    df = df.merge(kk_race,    on="race_id",            how="inner")
    df = df[~df["place"].isin(EXCLUDE_PLACES)]
    df = df.dropna(subset=["jyun", "ensemble_prob"])
    print(f"  結合後: {df['race_id'].nunique():,} races, {len(df):,} rows")
    return df


def add_race_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_honmei"] = (df["mark"] == "◎").astype(int)
    df["is_taikou"] = (df["mark"] == "◯").astype(int)
    df["is_sabo"]   = (df["mark"] == "▲").astype(int)
    df["is_delta"]  = (df["mark"] == "△").astype(int)

    grp = df.groupby("race_id")["ensemble_prob"]
    df["race_prob_max"]  = grp.transform("max")
    df["race_prob_2nd"]  = grp.transform(lambda x: x.nlargest(2).iloc[-1] if len(x) >= 2 else x.max())
    df["race_prob_mean"] = grp.transform("mean")
    df["race_prob_std"]  = grp.transform("std").fillna(0)
    df["field_size"]     = grp.transform("count")
    df["prob_z"] = np.where(df["race_prob_std"] > 0,
                            (df["ensemble_prob"] - df["race_prob_mean"]) / df["race_prob_std"], 0.0)
    df["gap_top1"] = df["race_prob_max"] - df["ensemble_prob"]
    df["gap_top2"] = df["race_prob_2nd"] - df["ensemble_prob"]
    df["prob_rank"] = grp.rank(ascending=False, method="min")

    def _entropy(probs: pd.Series) -> float:
        p = probs / probs.sum()
        return float(-np.sum(p * np.log(p + 1e-12)))
    ent_map = df.groupby("race_id")["ensemble_prob"].apply(_entropy)
    df["race_entropy"] = df["race_id"].map(ent_map)
    df["race_gap_12"]  = df["race_prob_max"] - df["race_prob_2nd"]

    df["label"] = 0
    df.loc[df["jyun"] <= 3, "label"] = 1
    df.loc[df["jyun"] == 1, "label"] = 2
    return df


FEATURE_COLS = [
    "ensemble_prob", "prob_rank", "prob_z",
    "gap_top1", "gap_top2", "field_size",
    "race_prob_std", "race_entropy", "race_gap_12",
    "is_honmei", "is_taikou", "is_sabo", "is_delta",
]


# ============================================================
# 3. LambdaRank 学習
# ============================================================
def train_model(df_train: pd.DataFrame, df_val: pd.DataFrame) -> lgb.Booster:
    print("\n[LightGBM LambdaRank 学習]")

    def _make(df):
        df = df.sort_values("race_id").reset_index(drop=True)
        X = df[FEATURE_COLS].fillna(0).values
        y = df["label"].values.astype(int)
        from itertools import groupby as _gb
        groups = np.array([len(list(g)) for _, g in _gb(df["race_id"])])
        return lgb.Dataset(X, label=y, group=groups, free_raw_data=False)

    ds_tr = _make(df_train)
    ds_vl = _make(df_val)

    params = {
        "objective": "lambdarank", "lambdarank_truncation_level": 3,
        "metric": "ndcg", "eval_at": [1, 3],
        "learning_rate": 0.01, "num_leaves": 31, "max_depth": 5,
        "min_data_in_leaf": 10,
        "feature_fraction": 0.9, "bagging_fraction": 0.9, "bagging_freq": 5,
        "lambda_l1": 0.05, "lambda_l2": 0.05,
        "verbose": -1, "n_jobs": -1, "seed": RANDOM_SEED,
        "deterministic": True, "force_col_wise": True,
    }
    model = lgb.train(
        params, ds_tr, num_boost_round=1000, valid_sets=[ds_vl],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(period=200)],
    )
    print(f"  best iter: {model.best_iteration}")
    return model


# ============================================================
# 4. Plackett-Luce + ROI
# ============================================================
def pl_combo_probs(scores, top_n=8):
    if len(scores) < 3: return []
    horses = sorted(scores, key=lambda h: scores[h], reverse=True)[:top_n]
    p = softmax(np.array([scores[h] for h in horses], dtype=float))
    results = []
    for f, s, t in itertools.permutations(range(len(horses)), 3):
        d2 = 1.0 - p[f]; d3 = d2 - p[s]
        if d2 <= 0 or d3 <= 0: continue
        results.append(((horses[f], horses[s], horses[t]), float(p[f]*(p[s]/d2)*(p[t]/d3))))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def build_race_list(df):
    races = []
    for rid, gdf in df.groupby("race_id"):
        if len(gdf) < 3: continue
        place = gdf["place"].iloc[0]
        if place in EXCLUDE_PLACES: continue
        top3 = gdf.dropna(subset=["jyun"]).sort_values("jyun").head(3)
        if len(top3) < 3: continue
        result = (int(top3.iloc[0]["umaban"]), int(top3.iloc[1]["umaban"]), int(top3.iloc[2]["umaban"]))
        pay = float(gdf["race_santan_pay"].iloc[0]) if pd.notna(gdf["race_santan_pay"].iloc[0]) else 0
        horses = []
        for _, row in gdf.iterrows():
            h = {f: float(row.get(f, 0)) for f in FEATURE_COLS}
            h["umaban"] = int(row["umaban"])
            horses.append(h)
        races.append({"race_id": rid, "horses": horses, "result_top3": result, "santan_pay": pay})
    return races


def simulate_roi(races, model, top_combos=9):
    total_bet = 0; total_return = 0; hits = 0; n_races = 0
    for r in races:
        X = pd.DataFrame([{f: row.get(f, 0) for f in FEATURE_COLS} for row in r["horses"]])
        if X.empty: continue
        scores = model.predict(X)
        umabans = [row["umaban"] for row in r["horses"]]
        score_map = {ub: float(s) for ub, s in zip(umabans, scores)}
        combos = pl_combo_probs(score_map, top_n=min(8, len(score_map)))
        if not combos: continue
        selected = [c for c, _ in combos[:top_combos]]
        if not selected: continue
        n = len(selected)
        per_bet = max(100, (BUDGET // n // 100) * 100)
        total_bet += per_bet * n; n_races += 1
        pay = r.get("santan_pay", 0)
        res = r.get("result_top3")
        if res and pay > 0:
            af, asc, at = res
            for f, s, t in selected:
                if f == af and s == asc and t == at:
                    total_return += per_bet * pay / 100; hits += 1; break
    roi = (total_return/total_bet*100) if total_bet>0 else 0.0
    return {"roi": roi, "hits": hits, "n_races": n_races,
            "total_bet": total_bet, "total_return": total_return}


# ============================================================
# main
# ============================================================
def main():
    print("=" * 60)
    print("build_trifecta_v2.py  (reproducible: seed=42, deterministic=True)")
    print("=" * 60)

    pred = generate_predictions()
    df = load_and_merge(pred)
    df = add_race_features(df)

    df["year"] = df["race_id"].str[:4].astype(int)
    df["ym"]   = df["race_id"].str[:6].astype(int)

    df_tr  = df[(df["year"] == 2023) & (df["ym"] <= 202309)].copy()
    df_vl  = df[(df["year"] == 2023) & (df["ym"] >  202309)].copy()
    df_24  = df[df["year"] == 2024].copy()
    df_25  = df[df["year"] == 2025].copy()

    print(f"\nTrain 2023/1-9  : {df_tr['race_id'].nunique():,} races, {len(df_tr):,} rows")
    print(f"Val   2023/10-12: {df_vl['race_id'].nunique():,} races, {len(df_vl):,} rows")
    print(f"OOS   2024      : {df_24['race_id'].nunique():,} races, {len(df_24):,} rows")
    print(f"OOS   2025      : {df_25['race_id'].nunique():,} races, {len(df_25):,} rows")

    model = train_model(df_tr, df_vl)

    payload = {
        "model": model, "feature_cols": FEATURE_COLS,
        "description": "trifecta_model_v2: LambdaRank + PL, train=2023/1-9 val=2023/10-12 OOS=2024-2025",
        "train_months": "2023/01-09", "val_months": "2023/10-12",
        "test_years":   [2024, 2025], "seed": RANDOM_SEED,
        "calibrator":   "ensemble_calibrator_v1.pkl",
        "master_csv":   MASTER_CSV.name,
    }
    OUT_MODEL.parent.mkdir(exist_ok=True)
    joblib.dump(payload, OUT_MODEL)
    print(f"\n[saved] {OUT_MODEL}")

    print("\n=== ROI 評価 (top_combos=9) ===")
    splits = [("Train 2023/1-9", df_tr), ("Val 2023/10-12", df_vl),
              ("OOS 2024", df_24), ("OOS 2025", df_25)]
    results = {}
    for label, sub in splits:
        races = build_race_list(sub)
        if not races:
            print(f"  {label}: n/a"); continue
        r = simulate_roi(races, model, top_combos=9)
        print(f"  {label}: ROI={r['roi']:7.2f}% hits={r['hits']}/{r['n_races']} bet=¥{r['total_bet']:,.0f}")
        results[label] = r

    print("\n=== OOS 2024+2025 top_combos 感度 ===")
    races_oos = build_race_list(pd.concat([df_24, df_25], ignore_index=True))
    sens = {}
    for tc in [3, 6, 9, 12, 15]:
        r = simulate_roi(races_oos, model, top_combos=tc)
        print(f"  tc={tc:2d}: ROI={r['roi']:7.2f}% hits={r['hits']}/{r['n_races']}")
        sens[tc] = r["roi"]

    meta = {
        "seed": RANDOM_SEED,
        "train_months": "2023/01-09", "val_months": "2023/10-12",
        "test_years": [2024, 2025],
        "best_iter": model.best_iteration,
        "roi": {k: round(v["roi"], 4) for k, v in results.items()},
        "sens_oos_2024_2025": {str(k): round(v, 4) for k, v in sens.items()},
        "calibrator": "ensemble_calibrator_v1.pkl",
        "master_csv": MASTER_CSV.name,
    }
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {OUT_META}")


if __name__ == "__main__":
    main()
