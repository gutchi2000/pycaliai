"""
grid_search_halo.py
HALO 三連単フォーメーション スコア差閾値 Optuna 最適化

最適化対象パラメータ:
  gap_12_hi   : スコア差 ≥ この値 → ◎突出 (現状 10)
  gap_12_lo   : スコア差 ≤ この値 → ◎◯拮抗候補 (現状 5)
  gap_top4_lo : ◎-4番手スコア差 ≤ この値 → ◎◯拮抗確定 (現状 15)
  pw_min      : AI高信頼: p_win(◎) ≥ この値 (現状 0.50)
  pw_ratio    : AI高信頼: p_win(◎) / p_win(◯) ≥ この値 (現状 2.0)

データ分割:
  Validation : 2024 (3,454R) → Optuna 目的関数
  Test (OOS) : 2025 (3,455R) → 最終評価

使い方:
    python grid_search_halo.py              # デフォルト 200 trials
    python grid_search_halo.py --trials 50  # 試行数指定
    python grid_search_halo.py --apply      # 最適化済みパラメータを本番反映
"""
from __future__ import annotations
import argparse
import io
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

BASE       = Path(__file__).parent
PRED_CSV   = BASE / "reports/ensemble_predictions.csv"
KEKKA_CSV  = BASE / "data/kekka_20160105_20251228_v2.csv"
ORDER_MODEL = BASE / "models/order_model_v1.pkl"
MASTER_CSV  = BASE / "data/master_kako5.csv"
OUT_JSON   = BASE / "data/halo_thresholds.json"

BUDGET         = 9600
EXCLUDE_PLACES = {"東京", "小倉"}

# 現行 Stage 0 Hybrid の固定値（ベースライン）
DEFAULT = {
    "gap_12_hi":   10.0,
    "gap_12_lo":    5.0,
    "gap_top4_lo": 15.0,
    "pw_min":       0.50,
    "pw_ratio":     2.0,
}


# ============================================================
# データ読み込み
# ============================================================
def load_races(years: set[int]) -> list[dict]:
    """指定年のレースリストを構築して返す。"""
    print(f"ensemble_predictions 読み込み (years={years})...")
    pred = pd.read_csv(PRED_CSV, encoding="utf-8-sig")
    pred = pred.rename(columns={"レースID(新/馬番無)": "race_id", "馬番": "umaban"})
    pred["race_id"] = pred["race_id"].astype(str)
    pred["umaban"]  = pd.to_numeric(pred["umaban"], errors="coerce").astype("Int64")
    pred["mark"]    = pred["mark"].astype(str).replace({"nan": ""})
    pred["year"]    = pred["race_id"].str[:4].astype(int)
    pred = pred[pred["year"].isin(years)].copy()

    print(f"  対象: {pred['race_id'].nunique():,} R")

    print("kekka 読み込み...")
    kk = pd.read_csv(KEKKA_CSV, encoding="cp932", dtype=str)
    kk["race_id"]    = kk["レースID(新)"].astype(str).str[:16]
    kk["umaban"]     = pd.to_numeric(kk["馬番"], errors="coerce").astype("Int64")
    kk["jyun"]       = pd.to_numeric(kk["確定着順"], errors="coerce")
    kk["santan_pay"] = pd.to_numeric(kk["３連単"], errors="coerce")
    kk["place"]      = kk["場所"].astype(str)
    kk["year"]       = kk["race_id"].str[:4].astype(int)
    kk = kk[kk["year"].isin(years)].copy()
    kk_grp = {rid: g for rid, g in kk.groupby("race_id")}

    print("order_model 読み込み...")
    order_obj = joblib.load(ORDER_MODEL)
    o_model = order_obj["model"]
    o_feats = order_obj["features"]
    o_encs  = order_obj["encoders"]

    from utils import parse_time_str
    TIME_COLS = ["前走走破タイム", "前走着差タイム"]

    def _prep(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col, le in o_encs.items():
            if col in df.columns:
                df[col] = df[col].astype(str).fillna("__NaN__")
                known = set(le.classes_)
                df[col] = df[col].apply(lambda x: x if x in known else "__unknown__")
                if "__unknown__" not in le.classes_:
                    le.classes_ = np.append(le.classes_, "__unknown__")
                df[col] = le.transform(df[col])
        for col in TIME_COLS:
            if col in df.columns:
                df[col] = parse_time_str(df[col])
        for f in o_feats:
            if f not in df.columns:
                df[f] = np.nan
        return df

    def predict_order(mgrp: pd.DataFrame) -> dict[int, float]:
        df = _prep(mgrp)
        proba = o_model.predict_proba(df[o_feats])
        return {int(ub): float(proba[i, 0])
                for i, ub in enumerate(df["馬番"].values)
                if pd.notna(ub)}

    # master_kako5 (order_model 推論用)
    print("master_kako5 読み込み...")
    master = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    master["race_id"] = master["レースID(新/馬番無)"].astype(str)
    master["馬番"]    = pd.to_numeric(master["馬番"], errors="coerce").astype("Int64")
    master["year"]    = master["race_id"].str[:4].astype(int)
    master = master[master["year"].isin(years)].copy()
    master_grp = {rid: g for rid, g in master.groupby("race_id")}

    # レース構築
    races = []
    for rid, p_grp in pred.groupby("race_id"):
        kgrp = kk_grp.get(rid)
        if kgrp is None or kgrp.empty:
            continue
        place = kgrp.iloc[0]["place"]
        if place in EXCLUDE_PLACES:
            continue
        top3 = kgrp.dropna(subset=["jyun"]).sort_values("jyun").head(3)
        if len(top3) < 3:
            continue
        pay_series = top3["santan_pay"].dropna()
        if pay_series.empty or pay_series.iloc[0] <= 0:
            continue
        first  = int(top3.iloc[0]["umaban"])
        second = int(top3.iloc[1]["umaban"])
        third  = int(top3.iloc[2]["umaban"])
        santan_pay = float(pay_series.iloc[0])

        marks: dict[str, int] = {}
        scores: dict[int, float] = {}
        for _, r in p_grp.iterrows():
            ub = int(r["umaban"]) if pd.notna(r["umaban"]) else None
            if ub is None:
                continue
            scores[ub] = float(r["ensemble_prob"]) if pd.notna(r["ensemble_prob"]) else 0.0
            if r["mark"] in ("◎", "◯", "▲", "△", "×"):
                marks[r["mark"]] = ub
        if "◎" not in marks or "◯" not in marks:
            continue

        pw_order: dict[int, float] = {}
        mgrp = master_grp.get(rid)
        if mgrp is not None and len(mgrp) >= 5:
            try:
                pw_order = predict_order(mgrp)
            except Exception:
                pass

        races.append({
            "race_id": rid, "place": place,
            "marks": marks, "scores": scores,
            "pw_order": pw_order,
            "first": first, "second": second, "third": third,
            "santan_pay": santan_pay,
        })

    print(f"  構築完了: {len(races):,} R")
    return races


# ============================================================
# フォーメーション生成（パラメータ付き）
# ============================================================
def _build(first, second, third) -> list[tuple]:
    combos: set[tuple] = set()
    for f in first:
        for s in second:
            for t in third:
                if len({f, s, t}) == 3:
                    combos.add((f, s, t))
    return list(combos)


def make_formation(r: dict, params: dict) -> list[tuple]:
    """パラメータ付きフォーメーション生成。"""
    marks  = r["marks"]
    scores = r["scores"]
    pw     = r["pw_order"]
    h = marks.get("◎"); o = marks.get("◯")
    h3 = marks.get("▲"); h4 = marks.get("△"); h5 = marks.get("×")
    if h is None or o is None:
        return []

    s_hon = scores.get(h, 0); s_tai = scores.get(o, 0)
    s3    = scores.get(h3, 0) if h3 else 0
    s4    = scores.get(h4, 0) if h4 else 0
    gap_12   = (s_hon - s_tai) * 100
    gap_top4 = ((s_hon - s4) if s4 > 0 else (s_hon - s3)) * 100

    thr_hi    = params["gap_12_hi"]
    thr_lo    = params["gap_12_lo"]
    thr_top4  = params["gap_top4_lo"]

    if gap_12 >= thr_hi:
        f = [h]
        s = [x for x in [o, h3] if x]
        t = [x for x in [o, h3, h4, h5] if x]
    elif gap_12 <= thr_lo and gap_top4 <= thr_top4:
        f = [h, o]
        s = [x for x in [h, o, h3] if x]
        t = [x for x in [h, o, h3, h4] if x]
    else:
        f = [h, o]
        s = [x for x in [h, o, h3, h4] if x]
        t = [x for x in [h, o, h3, h4, h5] if x]

    # AI 高信頼補強
    pw_min   = params["pw_min"]
    pw_ratio = params["pw_ratio"]
    if pw and h in pw and o in pw:
        pw_h = pw[h]; pw_o = pw[o]
        if pw_h >= pw_min and pw_h >= pw_o * pw_ratio:
            f = [h]
            s = [x for x in [o, h3] if x]
            t = [x for x in [o, h3, h4, h5] if x]

    combos = _build(f, s, t)
    # 点数上限 (36 超えたら third を短縮)
    if len(combos) > 36:
        t = t[:4]
        combos = _build(f, s, t)
    return combos


def simulate(races: list[dict], params: dict) -> dict:
    """指定パラメータでシミュレーションし ROI などを返す。"""
    invest = 0
    returns = 0
    hits = 0
    n = 0
    for r in races:
        combos = make_formation(r, params)
        if not combos:
            continue
        n += 1
        nc = len(combos)
        per_bet = max(100, (BUDGET // nc // 100) * 100)
        total_inv = per_bet * nc
        invest += total_inv
        # 的中確認
        tgt = (r["first"], r["second"], r["third"])
        if tgt in combos:
            hits += 1
            returns += per_bet * r["santan_pay"] / 100

    roi = (returns / invest * 100) if invest > 0 else 0.0
    hit_rate = (hits / n * 100) if n > 0 else 0.0
    return {
        "roi": roi, "invest": invest, "returns": returns,
        "hits": hits, "n_races": n, "hit_rate": hit_rate,
    }


# ============================================================
# Optuna 最適化
# ============================================================
def objective(trial: optuna.Trial, races: list[dict]) -> float:
    params = {
        "gap_12_hi":   trial.suggest_float("gap_12_hi",   3.0, 25.0),
        "gap_12_lo":   trial.suggest_float("gap_12_lo",   1.0, 12.0),
        "gap_top4_lo": trial.suggest_float("gap_top4_lo", 5.0, 35.0),
        "pw_min":      trial.suggest_float("pw_min",      0.25, 0.80),
        "pw_ratio":    trial.suggest_float("pw_ratio",    1.2,  4.0),
    }
    # gap_12_lo < gap_12_hi を強制
    if params["gap_12_lo"] >= params["gap_12_hi"]:
        raise optuna.TrialPruned()

    res = simulate(races, params)
    return res["roi"]


# ============================================================
# メイン
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="HALO スコア差閾値 Optuna 最適化")
    parser.add_argument("--trials", type=int, default=200, help="Optuna 試行数 (default: 200)")
    parser.add_argument("--apply", action="store_true", help="最適化済み閾値を app.py / predict_weekly.py に反映")
    args = parser.parse_args()

    # ── データ読み込み ──────────────────────────────────────────
    print("=" * 60)
    print("Validation 用データ (2024) 読み込み...")
    val_races  = load_races({2024})
    print()
    print("Test 用データ (2025) 読み込み...")
    test_races = load_races({2025})
    print()

    # ── ベースライン (Stage 0 Hybrid) ──────────────────────────
    print("=" * 60)
    print("ベースライン (Stage 0 Hybrid 固定値) で計測...")
    bl_val  = simulate(val_races,  DEFAULT)
    bl_test = simulate(test_races, DEFAULT)
    print(f"  Validation 2024: ROI {bl_val['roi']:.2f}%  "
          f"({bl_val['hits']}/{bl_val['n_races']}R  "
          f"hit={bl_val['hit_rate']:.2f}%)")
    print(f"  Test       2025: ROI {bl_test['roi']:.2f}%  "
          f"({bl_test['hits']}/{bl_test['n_races']}R  "
          f"hit={bl_test['hit_rate']:.2f}%)")

    # ── Optuna 最適化 ──────────────────────────────────────────
    print()
    print(f"Optuna 最適化開始 ({args.trials} trials, validation=2024)...")
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    # ベースラインを初期点として追加
    study.enqueue_trial(DEFAULT)
    study.optimize(
        lambda t: objective(t, val_races),
        n_trials=args.trials,
        show_progress_bar=True,
    )

    best_params = study.best_params
    best_val_roi = study.best_value
    print()
    print("=" * 60)
    print(f"最適化完了  ベスト Validation ROI: {best_val_roi:.2f}%")
    print(f"最適パラメータ: {best_params}")

    # ── テストセット (2025) で OOS 評価 ──────────────────────────
    oos = simulate(test_races, best_params)
    print()
    print(f"OOS Test 2025: ROI {oos['roi']:.2f}%  "
          f"({oos['hits']}/{oos['n_races']}R  hit={oos['hit_rate']:.2f}%)")
    delta_val  = best_val_roi  - bl_val['roi']
    delta_test = oos['roi'] - bl_test['roi']
    print(f"  Delta vs Baseline: val {delta_val:+.2f}pt  test {delta_test:+.2f}pt")

    # ── ROI 上位 10 試行 ──────────────────────────────────────
    print()
    print("Validation ROI 上位 10 試行:")
    df_trials = study.trials_dataframe()
    top10 = df_trials.nlargest(10, "value")[
        ["number", "value"] + [c for c in df_trials.columns if c.startswith("params_")]
    ]
    print(top10.to_string(index=False))

    # ── JSON 保存 ──────────────────────────────────────────────
    out = {
        "best_params": best_params,
        "baseline":    DEFAULT,
        "validation":  {"year": 2024, "roi": best_val_roi,
                        "hits": bl_val["hits"], "n_races": bl_val["n_races"]},
        "test_oos":    {"year": 2025, "roi": oos["roi"],
                        "hits": oos["hits"],    "n_races": oos["n_races"]},
        "baseline_val_roi":  bl_val["roi"],
        "baseline_test_roi": bl_test["roi"],
        "delta_val":  delta_val,
        "delta_test": delta_test,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(f"最適化結果を保存: {OUT_JSON}")

    # ── --apply: 本番コードに反映 ─────────────────────────────
    if args.apply:
        _apply_to_production(best_params)


def _apply_to_production(params: dict) -> None:
    """最適化済み閾値を app.py / predict_weekly.py の _score_pattern / logic_hybrid へ反映。

    sim 上 delta_test > 0 (OOS 改善) を確認してから手動で呼ぶこと。
    """
    import re
    files = {
        BASE / "app.py":             "_build_sanrentan_formation",
        BASE / "predict_weekly.py":  "HALO",
    }
    # 具体的な置き換えターゲットは sim_halo_formation_v2.py の _score_pattern と同等
    # ここでは halo_thresholds.json を参照する形の実装を前提にする
    print()
    print("[--apply] halo_thresholds.json を書き込みました。")
    print("app.py / predict_weekly.py 側で json を読み込む実装は Stage 2-04 で対応予定。")
    print(f"パラメータ: {params}")


if __name__ == "__main__":
    main()
