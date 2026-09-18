# -*- coding: utf-8 -*-
"""
tie_diagnosis.py — p_win 同値問題の診断 (spec §20) + 情報圧縮問題 (spec §3, B1/B2/B3)
========================================================================================
診断のみ。本番コードは変更しない。

B1 = M1 (calibrated_v6_probability + market_probability)
B2 = M2 (B1 + raw_v6_score + v6_rank)
B3 = M3 (B2 + レース内相対特徴)
本文 §3 の B1/B2/B3 と §11 の M1/M2/M3 は同一定義なので、M1/M2/M3 の fit 結果をそのまま流用する
(spec.json に明記)。

出力: out/tie_diagnosis.json
実行: python -m analysis.mcond.exp05_market_residual_dev.tie_diagnosis
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
from analysis.mcond.evaluate import fit_predict, metrics  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
D = BASE / "data/_research/mcond"


def race_top1_tie_rate(df: pd.DataFrame, col: str) -> dict:
    """レース毎に col の最大値を取る頭数が2以上の割合 (同値率)。"""
    def n_at_max(g):
        return int((g == g.max()).sum())
    counts = df.groupby("rid16")[col].apply(n_at_max)
    return {"n_races": int(len(counts)), "tie_rate": float((counts >= 2).mean()),
            "mean_tied_horses_when_tied": float(counts[counts >= 2].mean()) if (counts >= 2).any() else 0.0}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    df = pd.read_parquet(D / "exp05_design.parquet")
    y, yr = df["top3"].to_numpy(), df["year"].to_numpy()
    train, sel = df["train"].to_numpy(), df["sel"].to_numpy()

    result = {}

    # --- 同値率: 較正後 (win/top3) vs 生スコア ---
    result["tie_rates"] = {
        "calibrated_v6_probability_win (レース内最大タイ)": race_top1_tie_rate(df, "calibrated_v6_probability_win"),
        "calibrated_v6_probability (top3, レース内最大タイ)": race_top1_tie_rate(df, "calibrated_v6_probability"),
        "v6_score (生スコア, レース内最大タイ)": race_top1_tie_rate(df, "v6_score"),
        "v6_pwin (生PL勝率, レース内最大タイ)": race_top1_tie_rate(df, "v6_pwin"),
    }

    # --- 本番 pl_calibrators_v6.pkl 自体でのタイ率 (2024-2025のみ, 較正器にとって真にOOS) ---
    try:
        import joblib
        cal = joblib.load(BASE / "models/pl_calibrators_v6.pkl")
        calibrators = cal.get("calibrators", cal)
        prod_iso = calibrators["tansho"]
        oos = df[df.year >= 2024].copy()
        oos["prod_calibrated_pwin"] = prod_iso.predict(oos["v6_pwin"].to_numpy())
        result["tie_rate_production_calibrator_2024_2025"] = race_top1_tie_rate(oos, "prod_calibrated_pwin")
    except Exception as e:
        result["tie_rate_production_calibrator_2024_2025"] = {"error": str(e)}

    # --- B1/B2/B3 = M1/M2/M3 と同一。この診断ファイルでは refit せず、run.py の model_compare を参照する ---
    result["note_B1_B2_B3"] = "B1=M1, B2=M2, B3=M3 (run.py の model_compare.csv を参照、ここでは重複fitしない)"

    # --- タイ有りレース vs タイ無しレースでの ◎的中率・Brier 差 (calibrated_v6_probability, top3) ---
    def top1_is_top3(g):
        i = g["calibrated_v6_probability"].idxmax()
        return bool(g.loc[i, "top3"])

    tie_flag = df.groupby("rid16")["calibrated_v6_probability"].transform(
        lambda s: (s == s.max()).sum() >= 2)
    df["_tied"] = tie_flag
    per_race = df.groupby("rid16").apply(
        lambda g: pd.Series({"tied": bool(g["_tied"].iloc[0]), "top1_top3": top1_is_top3(g)}))
    for label, sub in (("tied", per_race[per_race.tied]), ("not_tied", per_race[~per_race.tied])):
        result.setdefault("top1_accuracy_by_tie", {})[label] = {
            "n_races": int(len(sub)), "top1_top3_rate": float(sub["top1_top3"].mean()) if len(sub) else np.nan}

    brier_by = {}
    for label, mask in (("tied", df["_tied"]), ("not_tied", ~df["_tied"])):
        p = df.loc[mask, "calibrated_v6_probability"].to_numpy()
        yy = df.loc[mask, "top3"].to_numpy()
        brier_by[label] = {"n": int(mask.sum()), "brier": float(((p - yy) ** 2).mean())}
    result["brier_by_tie"] = brier_by

    (OUT / "tie_diagnosis.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
