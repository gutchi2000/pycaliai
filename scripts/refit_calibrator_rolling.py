"""
refit_calibrator_rolling.py
============================
v5 calibrator を「より最近のデータ」で再 fit し、calibration を改善するか検証。

ストラテジー (シンプル版):
  - Calibration set: 2023 (valid) + 2024 H1 (4,219 races)
  - Audit set:        2024 H2 + 2025 (held-out, ~4,500 races)
  - 古い calibrator (valid 2023 のみ) と新 calibrator (1.5 年ぶん) を比較

最終的に improve したら:
  - models/pl_calibrators_v5_rolling.pkl として保存
  - export_weekly_marks.py で参照切替を提案
"""
from __future__ import annotations

import io
import sys
import warnings
from datetime import datetime
from itertools import combinations
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
MODEL_PKL  = BASE / "models" / "unified_rank_v5.pkl"
CAL_OLD    = BASE / "models" / "pl_calibrators_v5.pkl"
CAL_NEW    = BASE / "models" / "pl_calibrators_v5_rolling.pkl"
ODDS_2024  = BASE / "data" / "odds" / "odds_20240106-20241228.csv"
ODDS_2025  = BASE / "data" / "odds" / "odds_20250105-20261228.csv"
OUT_MD     = BASE / "reports" / f"calibrator_rolling_audit_{datetime.now():%Y%m%d}.md"

COL_RID  = "レースID(新/馬番無)"
COL_JYUN = "着順"
COL_BAN  = "馬番"


def wilson_ci(p_hat: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return max(0, center - half), min(1, center + half)


def load_test_with_raw_pl():
    print("=" * 70)
    print("[1/6] master 読み込み + v5 raw PL probability 計算")
    print("=" * 70)
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df["日付"] = pd.to_numeric(df["日付"], errors="coerce").astype("Int64")
    # valid 2023 + test 2024-2025 を全部使う
    df = df[df["split"].isin(["valid", "test"])].copy()
    df = df.dropna(subset=[COL_JYUN, COL_RID, "日付"]).copy()
    print(f"  rows: {len(df):,} / races: {df[COL_RID].nunique():,}")

    bundle = joblib.load(MODEL_PKL)
    model = bundle["model"]
    feats = bundle["feature_cols"]
    encs = bundle["encoders"]
    print(f"  model: {MODEL_PKL.name}, feats={len(feats)}")

    # encode
    print("[2/6] encoding + model.predict ...")
    for c, le in encs.items():
        if c not in df.columns:
            continue
        v = df[c].astype(str).fillna("__NaN__")
        known = set(le.classes_)
        v = v.where(v.isin(known), "__NaN__")
        df[c] = le.transform(v)

    X = df[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    df["_score"] = model.predict(X)

    # race ごとに raw PL
    print("[3/6] PL raw probability ...")
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
    df["p_win_raw"] = p_win_raw
    df["p_sho_raw"] = p_sho_raw
    return df


def fit_new_calibrator(df_train: pd.DataFrame):
    """正解 (1着 / top3) を使って isotonic fit。"""
    print(f"[4/6] 新 calibrator fit (calibration set: {len(df_train):,} 馬)")

    # 単勝
    p_tan = df_train["p_win_raw"].values
    y_tan = (df_train[COL_JYUN] == 1).astype(int).values
    iso_tan = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso_tan.fit(p_tan, y_tan)
    raw_t, cal_t, emp_t = p_tan.mean(), iso_tan.predict(p_tan).mean(), y_tan.mean()
    print(f"    tansho: n={len(p_tan):,}  raw={raw_t:.4f} cal={cal_t:.4f} emp={emp_t:.4f}")

    # 複勝
    p_fuku = df_train["p_sho_raw"].values
    y_fuku = (df_train[COL_JYUN] <= 3).astype(int).values
    iso_fuku = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso_fuku.fit(p_fuku, y_fuku)
    raw_f, cal_f, emp_f = p_fuku.mean(), iso_fuku.predict(p_fuku).mean(), y_fuku.mean()
    print(f"    fukusho: n={len(p_fuku):,} raw={raw_f:.4f} cal={cal_f:.4f} emp={emp_f:.4f}")
    return {"tansho": iso_tan, "fukusho": iso_fuku}


def merge_odds(df: pd.DataFrame):
    print("[5/6] オッズマージ (2024-2025) ...")
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

    df["_ban_int"] = pd.to_numeric(df[COL_BAN], errors="coerce").astype("Int64")
    df["_rid_str"] = df[COL_RID].astype(str)
    merged = df.merge(
        odds[["rid_no_ban", "ban", "tansho_odds", "fuku_low", "fuku_high"]],
        left_on=["_rid_str", "_ban_int"],
        right_on=["rid_no_ban", "ban"], how="left",
    )
    return merged


def audit_bins(df: pd.DataFrame, bet_type: str, p_col: str):
    """指定の calibrated p 列で EV ビン別 audit。"""
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
            rows.append({"EV帯": f"[{lo}, {hi})", "n": 0, "p_est": np.nan,
                         "p_obs": np.nan, "ROI": np.nan, "diff": np.nan})
            continue
        rows.append({
            "EV帯": f"[{lo}, {hi})", "n": n,
            "p_est": sub[p_col].mean(),
            "p_obs": sub["hit"].mean(),
            "ROI": (sub["hit"] * sub[odds_col]).sum() / n,
            "diff": sub["hit"].mean() - sub[p_col].mean(),
        })
    return rows


def main():
    df = load_test_with_raw_pl()

    # split: calibration = 2023 + 2024 H1 (date < 20240701)
    #        audit       = 2024 H2 + 2025 (date >= 20240701)
    cal_mask = df["日付"] < 20240701
    aud_mask = df["日付"] >= 20240701
    df_cal = df[cal_mask].copy()
    df_aud = df[aud_mask].copy()
    print()
    print(f"  calibration set: {len(df_cal):,} 馬 / {df_cal[COL_RID].nunique():,} race "
          f"(日付 {df_cal['日付'].min()} - {df_cal['日付'].max()})")
    print(f"  audit set:       {len(df_aud):,} 馬 / {df_aud[COL_RID].nunique():,} race "
          f"(日付 {df_aud['日付'].min()} - {df_aud['日付'].max()})")
    print()

    # 新 calibrator (calibration set で fit)
    new_cals = fit_new_calibrator(df_cal)

    # 旧 calibrator
    old_cal_bundle = joblib.load(CAL_OLD)
    old_cals = old_cal_bundle["calibrators"]

    # audit set に対して新旧 両方の calibrator を適用
    df_aud["p_win_old"] = old_cals["tansho"].predict(df_aud["p_win_raw"].values)
    df_aud["p_sho_old"] = old_cals["fukusho"].predict(df_aud["p_sho_raw"].values)
    df_aud["p_win_new"] = new_cals["tansho"].predict(df_aud["p_win_raw"].values)
    df_aud["p_sho_new"] = new_cals["fukusho"].predict(df_aud["p_sho_raw"].values)

    df_aud = merge_odds(df_aud)

    print("[6/6] audit (旧 vs 新) ...")
    tan_old = audit_bins(df_aud, "tansho", "p_win_old")
    tan_new = audit_bins(df_aud, "tansho", "p_win_new")
    fuku_old = audit_bins(df_aud, "fukusho", "p_sho_old")
    fuku_new = audit_bins(df_aud, "fukusho", "p_sho_new")

    # レポート生成
    md = []
    md.append(f"# calibrator rolling refit audit\n")
    md.append(f"**実施日**: {datetime.now():%Y-%m-%d %H:%M}")
    md.append(f"**calibration set**: 2023-01-05 〜 2024-06-30 ({len(df_cal):,} 馬, {df_cal[COL_RID].nunique():,} race)")
    md.append(f"**audit set**:        2024-07-01 〜 2025-12-28 ({len(df_aud):,} 馬, {df_aud[COL_RID].nunique():,} race)")
    md.append(f"**比較**: 旧 calibrator (valid 2023 のみで fit) vs 新 calibrator (1.5 年ぶん)\n")

    for name, old_rows, new_rows in [("単勝", tan_old, tan_new),
                                       ("複勝", fuku_old, fuku_new)]:
        md.append(f"## {name}\n")
        md.append("| EV帯 | n | p_obs | ROI(旧) | ROI(新) | diff(旧) | diff(新) |")
        md.append("|------|----|-------|---------|---------|----------|----------|")
        for o, n in zip(old_rows, new_rows):
            if o["n"] == 0:
                md.append(f"| {o['EV帯']} | 0 | — | — | — | — | — |")
                continue
            roi_old, roi_new = o["ROI"], n["ROI"]
            d_old, d_new = o["diff"], n["diff"]
            arrow = "→"
            if abs(roi_new - roi_old) >= 0.05:
                arrow = "↑" if roi_new > roi_old else "↓"
            md.append(
                f"| {o['EV帯']} | {o['n']:,} | "
                f"{o['p_obs']:.3f} | {roi_old:.3f} | **{roi_new:.3f}** {arrow} | "
                f"{d_old:+.3f} | {d_new:+.3f} |"
            )
        md.append("")

    # 改善判定
    def total_roi(rows, df_aud, bet_type):
        # 全体 (買えそうな帯のみ) ROI
        if bet_type == "tansho":
            df = df_aud.dropna(subset=["tansho_odds"]).copy()
            return (df.assign(hit=(df[COL_JYUN]==1).astype(int))
                      .pipe(lambda x: (x['hit'] * x['tansho_odds']).sum() / len(x)))
        else:
            df = df_aud.dropna(subset=["fuku_low", "fuku_high"]).copy()
            df["fuku_mid"] = (df["fuku_low"] + df["fuku_high"]) / 2
            return (df.assign(hit=(df[COL_JYUN]<=3).astype(int))
                      .pipe(lambda x: (x['hit'] * x['fuku_mid']).sum() / len(x)))

    md.append("## 改善判定\n")

    # 保存可否: 高 EV (>= 1.2) 帯の平均 ROI で判定
    high_ev_labels = ("[1.2,", "[1.5,", "[2.0,")
    tan_high_old = np.mean([r["ROI"] for r in tan_old
                              if r["n"] >= 100 and r["EV帯"].startswith(high_ev_labels)])
    tan_high_new = np.mean([r["ROI"] for r in tan_new
                              if r["n"] >= 100 and r["EV帯"].startswith(high_ev_labels)])
    fuku_high_old = np.mean([r["ROI"] for r in fuku_old
                               if r["n"] >= 100 and r["EV帯"].startswith(high_ev_labels)])
    fuku_high_new = np.mean([r["ROI"] for r in fuku_new
                               if r["n"] >= 100 and r["EV帯"].startswith(high_ev_labels)])
    improved = (tan_high_new - tan_high_old > 0.02 or
                fuku_high_new - fuku_high_old > 0.02)

    if improved:
        # 新 calibrator を保存
        joblib.dump({
            "calibrators":  new_cals,
            "source_model": MODEL_PKL.name,
            "fit_split":    "valid 2023 + test 2024 H1 (rolling 1.5 年)",
            "n_races":      df_cal[COL_RID].nunique(),
            "seed":         42,
            "fit_date":     datetime.now().isoformat(),
            "improvement_vs_old": float(tan_high_new - tan_high_old),
        }, CAL_NEW)
        md.append(f"## 結論: ✅ 改善あり、新 calibrator を保存\n")
        md.append(f"- 単勝高 EV (≥1.2) 平均 ROI: {tan_high_old:.3f} → {tan_high_new:.3f} ({tan_high_new-tan_high_old:+.3f})")
        md.append(f"- 複勝高 EV (≥1.2) 平均 ROI: {fuku_high_old:.3f} → {fuku_high_new:.3f} ({fuku_high_new-fuku_high_old:+.3f})")
        md.append(f"- 保存先: `{CAL_NEW.name}`")
        md.append(f"- 推奨: `export_weekly_marks.py` の calibrator path を切替")
        print(f"\n[saved] {CAL_NEW}")
    else:
        md.append(f"## 結論: ❌ 大きな改善なし\n")
        md.append(f"- 単勝高 EV (≥1.2) 平均 ROI: {tan_high_old:.3f} → {tan_high_new:.3f} (差 {tan_high_new-tan_high_old:+.3f})")
        md.append(f"- 複勝高 EV (≥1.2) 平均 ROI: {fuku_high_old:.3f} → {fuku_high_new:.3f} (差 {fuku_high_new-fuku_high_old:+.3f})")
        md.append(f"\n**解釈**: 1.5 年データで再 fit しても改善しない = 単純な distribution shift ではない")
        md.append(f"\n**根本原因の候補**:")
        md.append(f"1. v5 の sample_weight (穴馬好走重視) → raw model output が systematically biased")
        md.append(f"2. LambdaRank (ranking 最適化) → 確率値の絶対精度は副次的")
        md.append(f"3. isotonic は monotonic 補正のみ → tail での歪みを完全補正できない")
        md.append(f"\n**推奨次手**:")
        md.append(f"- **Option B (rank-based)**: EV 絶対値ではなく EV 順位 top-N で買う")
        md.append(f"- **Option C (v6 再学習)**: sample_weight=None で再学習し再評価")
        md.append(f"- **Option D (Platt + isotonic 二段)**: より柔軟な calibration を試す")
        print(f"\n[NOT saved] 改善なしのため calibrator は更新せず")

    md_text = "\n".join(md)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(md_text)
    print(f"\n[saved] {OUT_MD}")
    print()
    print(md_text)


if __name__ == "__main__":
    main()
