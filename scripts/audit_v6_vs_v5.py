"""
audit_v6_vs_v5.py
=================
v5 と v6 の高 EV 帯 calibration を直接比較。

両モデルで test 2024-2025 の predictions を生成 → audit_ev_calibration と
同じロジックで EV ビン別 ROI を測定 → 並列比較。

最終判断:
  v6 採用基準: 単勝 高 EV (≥1.2) ROI が v5 比 +0.05 以上改善
              または 複勝 高 EV ROI が v5 比 +0.03 以上改善
"""
from __future__ import annotations

import io
import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent))
import pl_probs as PL

BASE       = Path(__file__).parent.parent
MASTER_CSV = BASE / "data" / "master_v2_20130105-20251228.csv"
MODEL_V5   = BASE / "models" / "unified_rank_v5.pkl"
MODEL_V6   = BASE / "models" / "unified_rank_v6.pkl"
CAL_V5     = BASE / "models" / "pl_calibrators_v5.pkl"
ODDS_2024  = BASE / "data" / "odds" / "odds_20240106-20241228.csv"
ODDS_2025  = BASE / "data" / "odds" / "odds_20250105-20261228.csv"
OUT_MD     = BASE / "reports" / f"audit_v6_vs_v5_{datetime.now():%Y%m%d}.md"

COL_RID  = "レースID(新/馬番無)"
COL_JYUN = "着順"
COL_BAN  = "馬番"


def predict_with_model(df_test: pd.DataFrame, model_pkl: Path) -> pd.DataFrame:
    """test data に v5 or v6 を適用、PL raw + calibrated p_win, p_sho を計算。"""
    print(f"[load] {model_pkl.name}")
    bundle = joblib.load(model_pkl)
    model = bundle["model"]
    feats = bundle["feature_cols"]
    encs = bundle["encoders"]

    df = df_test.copy()
    for c, le in encs.items():
        if c not in df.columns:
            continue
        v = df[c].astype(str).fillna("__NaN__")
        known = set(le.classes_)
        v = v.where(v.isin(known), "__NaN__")
        df[c] = le.transform(v)

    X = df[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    df["_score"] = model.predict(X)

    # PL raw
    p_win_raw = np.zeros(len(df))
    p_sho_raw = np.zeros(len(df))
    rids = df[COL_RID].values
    indices_by_rid = {}
    for i, rid in enumerate(rids):
        indices_by_rid.setdefault(rid, []).append(i)
    for rid, idx_list in indices_by_rid.items():
        idx = np.array(idx_list)
        s = df["_score"].values[idx]
        w = PL.pl_weights(s)
        p_win_raw[idx] = PL.all_tansho(w)
        p_sho_raw[idx] = PL.all_fukusho(w)
    return p_win_raw, p_sho_raw


def fit_v6_calibrator(df: pd.DataFrame, p_win_raw: np.ndarray, p_sho_raw: np.ndarray):
    """v6 用の isotonic calibrator を valid 2023 で fit (シンプル版)。"""
    print("[fit] v6 calibrator on valid 2023")
    df_valid = df[df["split"] == "valid"].copy()
    valid_idx = df_valid.index.values
    p_win_v = p_win_raw[valid_idx] if len(valid_idx) == len(p_win_raw) else None

    # index alignment が難しいので、df 自体に raw を入れて filter
    df["_p_win_raw"] = p_win_raw
    df["_p_sho_raw"] = p_sho_raw

    valid_p_win = df[df["split"] == "valid"]["_p_win_raw"].values
    valid_p_sho = df[df["split"] == "valid"]["_p_sho_raw"].values
    valid_jyun = df[df["split"] == "valid"][COL_JYUN].astype(int).values

    iso_tan = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso_tan.fit(valid_p_win, (valid_jyun == 1).astype(int))
    iso_fuku = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso_fuku.fit(valid_p_sho, (valid_jyun <= 3).astype(int))
    return iso_tan, iso_fuku


def merge_odds(df_test: pd.DataFrame):
    odds_24 = pd.read_csv(ODDS_2024, encoding="cp932", dtype=str)
    odds_25 = pd.read_csv(ODDS_2025, encoding="cp932", dtype=str)
    odds = pd.concat([odds_24, odds_25], ignore_index=True)
    odds = odds.rename(columns={
        "レースID(新)": "rid_full", "馬番": "ban",
        "単勝オッズ": "tansho_odds",
        "複勝オッズ下限": "fuku_low",
        "複勝オッズ上限": "fuku_high",
    })
    odds["ban"] = pd.to_numeric(odds["ban"], errors="coerce").astype("Int64")
    odds["tansho_odds"] = pd.to_numeric(odds["tansho_odds"], errors="coerce")
    odds["fuku_low"] = pd.to_numeric(odds["fuku_low"], errors="coerce")
    odds["fuku_high"] = pd.to_numeric(odds["fuku_high"], errors="coerce")
    odds["rid_no_ban"] = odds["rid_full"].astype(str).str[:16]
    df_test["_ban_int"] = pd.to_numeric(df_test[COL_BAN], errors="coerce").astype("Int64")
    df_test["_rid_str"] = df_test[COL_RID].astype(str)
    return df_test.merge(
        odds[["rid_no_ban", "ban", "tansho_odds", "fuku_low", "fuku_high"]],
        left_on=["_rid_str", "_ban_int"], right_on=["rid_no_ban", "ban"], how="left",
    )


def audit_bins(df: pd.DataFrame, p_col: str, bet_type: str):
    if bet_type == "tansho":
        df = df.dropna(subset=["tansho_odds", p_col]).copy()
        df["EV"] = df[p_col] * df["tansho_odds"]
        df["hit"] = (df[COL_JYUN] == 1).astype(int)
        odds_col = "tansho_odds"
    else:
        df = df.dropna(subset=["fuku_low", "fuku_high", p_col]).copy()
        df["fuku_mid"] = (df["fuku_low"] + df["fuku_high"]) / 2
        df["EV"] = df[p_col] * df["fuku_mid"]
        df["hit"] = (df[COL_JYUN] <= 3).astype(int)
        odds_col = "fuku_mid"

    bins = [(0, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.05),
            (1.05, 1.20), (1.20, 1.50), (1.50, 2.00), (2.00, 999)]
    rows = []
    for lo, hi in bins:
        mask = (df["EV"] >= lo) & (df["EV"] < hi)
        sub = df[mask]
        n = len(sub)
        if n == 0:
            rows.append({"EV帯": f"[{lo}, {hi})", "n": 0, "ROI": np.nan, "p_obs": np.nan, "p_est": np.nan})
            continue
        rows.append({
            "EV帯": f"[{lo}, {hi})", "n": n,
            "p_obs": float(sub["hit"].mean()),
            "p_est": float(sub[p_col].mean()),
            "ROI": float((sub["hit"] * sub[odds_col]).sum() / n),
        })
    return rows


def main():
    print("=" * 70)
    print("[1/4] master 読み込み")
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df["日付"] = pd.to_numeric(df["日付"], errors="coerce").astype("Int64")
    df = df[df["split"].isin(["valid", "test"])].copy()
    df = df.dropna(subset=[COL_JYUN, COL_RID, "日付"]).reset_index(drop=True)
    print(f"  rows: {len(df):,} / races: {df[COL_RID].nunique():,}")

    # v5 推論
    print("\n[2/4] v5 推論")
    p_win_v5, p_sho_v5 = predict_with_model(df, MODEL_V5)
    df["p_win_v5_raw"] = p_win_v5
    df["p_sho_v5_raw"] = p_sho_v5

    # v6 推論
    print("\n[3/4] v6 推論")
    p_win_v6, p_sho_v6 = predict_with_model(df, MODEL_V6)
    df["p_win_v6_raw"] = p_win_v6
    df["p_sho_v6_raw"] = p_sho_v6

    # v5 calibrator (既存)
    cal_v5 = joblib.load(CAL_V5)["calibrators"]
    df["p_win_v5"] = cal_v5["tansho"].predict(df["p_win_v5_raw"].values)
    df["p_sho_v5"] = cal_v5["fukusho"].predict(df["p_sho_v5_raw"].values)

    # v6 calibrator: valid 2023 で fit (シンプル isotonic)
    print("\n[fit] v6 calibrator on valid 2023 (raw v6 PL)")
    df_v = df[df["split"] == "valid"].copy()
    iso_tan = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso_tan.fit(df_v["p_win_v6_raw"].values, (df_v[COL_JYUN] == 1).astype(int).values)
    iso_fuku = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso_fuku.fit(df_v["p_sho_v6_raw"].values, (df_v[COL_JYUN] <= 3).astype(int).values)
    df["p_win_v6"] = iso_tan.predict(df["p_win_v6_raw"].values)
    df["p_sho_v6"] = iso_fuku.predict(df["p_sho_v6_raw"].values)

    # test split に絞って audit
    df_test = df[df["split"] == "test"].copy()
    df_test = merge_odds(df_test)
    print(f"\n[4/4] test {len(df_test):,} 馬, EV ビン audit ...")

    audit = {
        "v5_tansho":   audit_bins(df_test, "p_win_v5", "tansho"),
        "v6_tansho":   audit_bins(df_test, "p_win_v6", "tansho"),
        "v5_fukusho":  audit_bins(df_test, "p_sho_v5", "fukusho"),
        "v6_fukusho":  audit_bins(df_test, "p_sho_v6", "fukusho"),
    }

    # レポート生成
    md = []
    md.append(f"# v5 vs v6 EV ビン別 ROI 比較\n")
    md.append(f"**実施日**: {datetime.now():%Y-%m-%d %H:%M}")
    md.append(f"**audit set**: test 2024-2025 ({len(df_test):,} 馬, "
              f"{df_test[COL_RID].nunique():,} race)\n")

    md.append("## モデル概要\n")
    md.append("| | v5 (旧) | v6 (新) |")
    md.append("|---|---------|---------|")
    md.append("| sample_weight alpha | 1.325 | **0.031** (ほぼゼロ) |")
    md.append("| Optuna objective | composite mark accuracy | composite - 0.5×ECE_high_p |")
    md.append("| valid NDCG@5 | 0.6027 | 0.5938 (-0.009) |")
    md.append("| valid ECE_high_p | (未測定) | 0.0112 |")
    md.append("")

    for bet_type in ["tansho", "fukusho"]:
        bt_jp = "単勝" if bet_type == "tansho" else "複勝"
        md.append(f"## {bt_jp} EV 帯別 ROI\n")
        md.append("| EV帯 | v5 n | v5 ROI | v5 p_obs | v6 n | v6 ROI | v6 p_obs | ΔROI |")
        md.append("|------|------|--------|----------|------|--------|----------|------|")
        v5_rows = audit[f"v5_{bet_type}"]
        v6_rows = audit[f"v6_{bet_type}"]
        for v5r, v6r in zip(v5_rows, v6_rows):
            if v5r["n"] == 0 and v6r["n"] == 0:
                md.append(f"| {v5r['EV帯']} | 0 | — | — | 0 | — | — | — |")
                continue
            d = v6r["ROI"] - v5r["ROI"] if not np.isnan(v6r["ROI"]) and not np.isnan(v5r["ROI"]) else np.nan
            arrow = ""
            if not np.isnan(d):
                if d >= 0.05:   arrow = " ↑"
                elif d <= -0.05: arrow = " ↓"
                else:            arrow = " →"
            md.append(
                f"| {v5r['EV帯']} | {v5r['n']:,} | {v5r['ROI']:.3f} | {v5r['p_obs']:.3f} | "
                f"{v6r['n']:,} | **{v6r['ROI']:.3f}** | {v6r['p_obs']:.3f} | "
                f"{d:+.3f}{arrow} |"
            )
        md.append("")

    # 高 EV 平均 ROI
    high_lbl = ("[1.2,", "[1.5,", "[2.0,")

    def avg_high(rows):
        vals = [r["ROI"] for r in rows
                if r["n"] >= 100 and r["EV帯"].startswith(high_lbl)]
        return np.mean(vals) if vals else np.nan

    v5_tan_high = avg_high(audit["v5_tansho"])
    v6_tan_high = avg_high(audit["v6_tansho"])
    v5_fuku_high = avg_high(audit["v5_fukusho"])
    v6_fuku_high = avg_high(audit["v6_fukusho"])

    md.append("## 高 EV (≥1.2) 平均 ROI\n")
    md.append("| 馬券種 | v5 | v6 | 差分 | 判定 |")
    md.append("|--------|----|----|------|------|")

    def judge(v5, v6, threshold):
        d = v6 - v5
        if d >= threshold: return f"✅ 改善 ({d:+.3f})"
        elif d >= 0:        return f"→ 同等 ({d:+.3f})"
        else:               return f"❌ 悪化 ({d:+.3f})"

    md.append(f"| 単勝 | {v5_tan_high:.3f} | **{v6_tan_high:.3f}** | "
              f"{v6_tan_high-v5_tan_high:+.3f} | {judge(v5_tan_high, v6_tan_high, 0.05)} |")
    md.append(f"| 複勝 | {v5_fuku_high:.3f} | **{v6_fuku_high:.3f}** | "
              f"{v6_fuku_high-v5_fuku_high:+.3f} | {judge(v5_fuku_high, v6_fuku_high, 0.03)} |")
    md.append("")

    # 採用判定
    adopt = (v6_tan_high - v5_tan_high) >= 0.05 or (v6_fuku_high - v5_fuku_high) >= 0.03
    if adopt:
        md.append("## 結論: ✅ v6 採用推奨\n")
        md.append("- 高 EV 帯の calibration が改善")
        md.append("- 推奨アクション:")
        md.append("  1. `models/pl_calibrators_v6.pkl` を build (`python build_pl_calibrators.py` v6 用)")
        md.append("  2. `data/pl_payout_curve_v6.pkl` を build")
        md.append("  3. `export_weekly_marks.py` の MODEL_PKL/CAL_PKL を v6 に切替")
        md.append("  4. ROI 動的閾値を v6 audit 結果に合わせて再調整")
    else:
        md.append("## 結論: ❌ v6 採用見送り\n")
        md.append(f"- 単勝高 EV ROI 改善幅: {v6_tan_high-v5_tan_high:+.3f} (基準 +0.05)")
        md.append(f"- 複勝高 EV ROI 改善幅: {v6_fuku_high-v5_fuku_high:+.3f} (基準 +0.03)")
        md.append("- 大幅な改善なし。v5 維持。")
        md.append("- Option B (rank-based) または別の角度を検討")

    md_text = "\n".join(md)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(md_text)
    print(f"\n[saved] {OUT_MD}")
    print()
    print(md_text)


if __name__ == "__main__":
    main()
