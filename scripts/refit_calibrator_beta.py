"""
refit_calibrator_beta.py
========================
Option D: より柔軟な calibration を試す。

3 種類比較:
  1. 旧 isotonic (valid 2023, 既存 baseline)
  2. Beta calibration (Kull et al. 2017): logistic on [log p, log(1-p)]
  3. Platt + isotonic 二段: Platt scaling → isotonic で fine-tune

isotonic は単調補正のみ。Beta は parametric 3-param で tail を柔軟に処理。
Platt+iso は両方の良いとこ取り (parametric overall + nonparametric local correction)。
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
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent))
import pl_probs as PL

BASE       = Path(__file__).parent.parent
MASTER_CSV = BASE / "data" / "master_v2_20130105-20251228.csv"
MODEL_PKL  = BASE / "models" / "unified_rank_v5.pkl"
CAL_OLD    = BASE / "models" / "pl_calibrators_v5.pkl"
CAL_BETA   = BASE / "models" / "pl_calibrators_v5_beta.pkl"
CAL_2STAGE = BASE / "models" / "pl_calibrators_v5_2stage.pkl"
ODDS_2024  = BASE / "data" / "odds" / "odds_20240106-20241228.csv"
ODDS_2025  = BASE / "data" / "odds" / "odds_20250105-20261228.csv"
OUT_MD     = BASE / "reports" / f"calibrator_beta_audit_{datetime.now():%Y%m%d}.md"

COL_RID  = "レースID(新/馬番無)"
COL_JYUN = "着順"
COL_BAN  = "馬番"


class BetaCalibrator:
    """Beta calibration (Kull et al. 2017): logistic on [log p, log(1-p)].

    f(p) = sigmoid(a*log(p) + b*log(1-p) + c)
    3 params (a, b, c) で Beta 分布の累積に相当する補正を学習。
    isotonic より tail に柔軟。
    """
    def fit(self, p: np.ndarray, y: np.ndarray):
        eps = 1e-9
        p_clip = np.clip(p, eps, 1 - eps)
        X = np.column_stack([np.log(p_clip), np.log(1 - p_clip)])
        self.lr = LogisticRegression(max_iter=1000, C=1e6)  # 弱正則化
        self.lr.fit(X, y)
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        eps = 1e-9
        p_clip = np.clip(p, eps, 1 - eps)
        X = np.column_stack([np.log(p_clip), np.log(1 - p_clip)])
        return self.lr.predict_proba(X)[:, 1]


class TwoStageCalibrator:
    """Platt scaling (sigmoid on logit p) → isotonic で fine-tune。"""
    def fit(self, p: np.ndarray, y: np.ndarray):
        eps = 1e-9
        p_clip = np.clip(p, eps, 1 - eps)
        # Stage 1: Platt
        logit_p = np.log(p_clip / (1 - p_clip)).reshape(-1, 1)
        self.platt = LogisticRegression(max_iter=1000, C=1e6)
        self.platt.fit(logit_p, y)
        p_platt = self.platt.predict_proba(logit_p)[:, 1]
        # Stage 2: isotonic
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.iso.fit(p_platt, y)
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        eps = 1e-9
        p_clip = np.clip(p, eps, 1 - eps)
        logit_p = np.log(p_clip / (1 - p_clip)).reshape(-1, 1)
        p_platt = self.platt.predict_proba(logit_p)[:, 1]
        return self.iso.predict(p_platt)


def wilson_ci(p_hat, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return max(0, center - half), min(1, center + half)


def load_data_with_raw_pl():
    print("=" * 70)
    print("[1/5] data + raw PL probability 計算")
    print("=" * 70)
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df["日付"] = pd.to_numeric(df["日付"], errors="coerce").astype("Int64")
    df = df[df["split"].isin(["valid", "test"])].copy()
    df = df.dropna(subset=[COL_JYUN, COL_RID, "日付"]).copy()

    bundle = joblib.load(MODEL_PKL)
    model, feats, encs = bundle["model"], bundle["feature_cols"], bundle["encoders"]
    for c, le in encs.items():
        if c not in df.columns:
            continue
        v = df[c].astype(str).fillna("__NaN__")
        known = set(le.classes_)
        v = v.where(v.isin(known), "__NaN__")
        df[c] = le.transform(v)
    X = df[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    df["_score"] = model.predict(X)

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


def merge_odds(df: pd.DataFrame):
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
    return df.merge(
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
            rows.append({"EV帯": f"[{lo}, {hi})", "n": 0, "ROI": np.nan})
            continue
        rows.append({
            "EV帯": f"[{lo}, {hi})", "n": n,
            "p_obs": sub["hit"].mean(),
            "ROI": (sub["hit"] * sub[odds_col]).sum() / n,
        })
    return rows


def main():
    df = load_data_with_raw_pl()
    cal_mask = df["日付"] < 20240701
    aud_mask = df["日付"] >= 20240701
    df_cal = df[cal_mask].copy()
    df_aud = df[aud_mask].copy()
    print(f"  cal: {len(df_cal):,} 馬 / aud: {len(df_aud):,} 馬")

    print("[2/5] calibrator fit (3 種)")
    methods = {
        "isotonic_old": joblib.load(CAL_OLD)["calibrators"],
        "beta_new":     None,
        "2stage_new":   None,
    }

    # Beta
    bc_tan = BetaCalibrator().fit(df_cal["p_win_raw"].values,
                                    (df_cal[COL_JYUN] == 1).astype(int).values)
    bc_fuku = BetaCalibrator().fit(df_cal["p_sho_raw"].values,
                                     (df_cal[COL_JYUN] <= 3).astype(int).values)
    methods["beta_new"] = {"tansho": bc_tan, "fukusho": bc_fuku}
    print(f"  beta tansho coefs: {bc_tan.lr.coef_[0]}, intercept: {bc_tan.lr.intercept_[0]:.3f}")
    print(f"  beta fukusho coefs: {bc_fuku.lr.coef_[0]}, intercept: {bc_fuku.lr.intercept_[0]:.3f}")

    # 2-stage
    ts_tan = TwoStageCalibrator().fit(df_cal["p_win_raw"].values,
                                        (df_cal[COL_JYUN] == 1).astype(int).values)
    ts_fuku = TwoStageCalibrator().fit(df_cal["p_sho_raw"].values,
                                         (df_cal[COL_JYUN] <= 3).astype(int).values)
    methods["2stage_new"] = {"tansho": ts_tan, "fukusho": ts_fuku}

    print("[3/5] audit set への適用")
    df_aud["p_win_iso"]    = methods["isotonic_old"]["tansho"].predict(df_aud["p_win_raw"].values)
    df_aud["p_sho_iso"]    = methods["isotonic_old"]["fukusho"].predict(df_aud["p_sho_raw"].values)
    df_aud["p_win_beta"]   = methods["beta_new"]["tansho"].predict(df_aud["p_win_raw"].values)
    df_aud["p_sho_beta"]   = methods["beta_new"]["fukusho"].predict(df_aud["p_sho_raw"].values)
    df_aud["p_win_2stage"] = methods["2stage_new"]["tansho"].predict(df_aud["p_win_raw"].values)
    df_aud["p_sho_2stage"] = methods["2stage_new"]["fukusho"].predict(df_aud["p_sho_raw"].values)

    df_aud = merge_odds(df_aud)

    print("[4/5] EV ビン audit")
    audit_results = {}
    for cal_name, p_win_col, p_sho_col in [
        ("iso (旧)",     "p_win_iso",    "p_sho_iso"),
        ("beta",         "p_win_beta",   "p_sho_beta"),
        ("2stage",       "p_win_2stage", "p_sho_2stage"),
    ]:
        audit_results[cal_name] = {
            "tansho":  audit_bins(df_aud, p_win_col, "tansho"),
            "fukusho": audit_bins(df_aud, p_sho_col, "fukusho"),
        }

    # レポート
    print("[5/5] レポート生成")
    md = []
    md.append("# Beta + 2-stage Calibration Audit\n")
    md.append(f"**実施日**: {datetime.now():%Y-%m-%d %H:%M}")
    md.append(f"**calibration**: 2023-01〜2024-06 ({df_cal[COL_RID].nunique():,} race)")
    md.append(f"**audit**:        2024-07〜2025-12 ({df_aud[COL_RID].nunique():,} race)\n")

    for bet_type in ["tansho", "fukusho"]:
        md.append(f"## {'単勝' if bet_type=='tansho' else '複勝'}\n")
        md.append("| EV帯 | n | ROI(iso) | ROI(beta) | ROI(2stage) |")
        md.append("|------|----|----------|-----------|-------------|")
        n_bins = len(audit_results["iso (旧)"][bet_type])
        for i in range(n_bins):
            r_iso = audit_results["iso (旧)"][bet_type][i]
            r_beta = audit_results["beta"][bet_type][i]
            r_ts = audit_results["2stage"][bet_type][i]
            if r_iso["n"] == 0:
                md.append(f"| {r_iso['EV帯']} | 0 | — | — | — |")
                continue
            md.append(f"| {r_iso['EV帯']} | {r_iso['n']:,} | "
                      f"{r_iso['ROI']:.3f} | {r_beta['ROI']:.3f} | "
                      f"{r_ts['ROI']:.3f} |")
        md.append("")

    # 高 EV (≥1.2) 平均 ROI 比較
    md.append("## 高 EV (≥1.2) 平均 ROI 比較\n")
    high_ev_lbl = ("[1.2,", "[1.5,", "[2.0,")
    for bet_type in ["tansho", "fukusho"]:
        avgs = {}
        for cal_name in audit_results:
            rows = audit_results[cal_name][bet_type]
            high = [r["ROI"] for r in rows if r["n"] >= 100 and r["EV帯"].startswith(high_ev_lbl)]
            avgs[cal_name] = np.mean(high) if high else np.nan
        md.append(f"### {'単勝' if bet_type=='tansho' else '複勝'}")
        for k, v in avgs.items():
            md.append(f"- {k}: **{v:.3f}**")
        # best
        best_method = max(avgs, key=lambda k: avgs[k] if not np.isnan(avgs[k]) else -1)
        md.append(f"  → 最良: **{best_method}** ({avgs[best_method]:.3f})")
        md.append("")

    # 結論判定
    md.append("## 結論\n")
    iso_tan = avgs_tan = None
    # tansho avgs を再計算 (上書きされてるので)
    for bet_type in ["tansho", "fukusho"]:
        avgs = {}
        for cal_name in audit_results:
            rows = audit_results[cal_name][bet_type]
            high = [r["ROI"] for r in rows if r["n"] >= 100 and r["EV帯"].startswith(high_ev_lbl)]
            avgs[cal_name] = np.mean(high) if high else np.nan
        if bet_type == "tansho":
            iso_tan = avgs["iso (旧)"]
            best_tan = max(avgs.values())
        else:
            iso_fuku = avgs["iso (旧)"]
            best_fuku = max(avgs.values())

    delta_tan = best_tan - iso_tan
    delta_fuku = best_fuku - iso_fuku
    md.append(f"- 単勝高 EV ROI: iso {iso_tan:.3f} → 最良 {best_tan:.3f} (差 {delta_tan:+.3f})")
    md.append(f"- 複勝高 EV ROI: iso {iso_fuku:.3f} → 最良 {best_fuku:.3f} (差 {delta_fuku:+.3f})")

    if delta_tan > 0.02 or delta_fuku > 0.02:
        # 改善あり、Beta or 2stage を保存
        md.append(f"\n✅ **改善あり** (>2pp)。新 calibrator 保存:")
        # Beta or 2-stage、どちらが良いか判定
        beta_avg = (avgs["beta_new"] if False else 0)  # placeholder
        joblib.dump({
            "calibrators": methods["beta_new"],
            "source_model": MODEL_PKL.name,
            "fit_split":    "valid 2023 + test 2024 H1, Beta calibration",
            "n_races":      df_cal[COL_RID].nunique(),
            "method":       "beta_calibration",
            "fit_date":     datetime.now().isoformat(),
        }, CAL_BETA)
        joblib.dump({
            "calibrators": methods["2stage_new"],
            "source_model": MODEL_PKL.name,
            "fit_split":    "valid 2023 + test 2024 H1, Platt + isotonic",
            "n_races":      df_cal[COL_RID].nunique(),
            "method":       "platt_then_isotonic",
            "fit_date":     datetime.now().isoformat(),
        }, CAL_2STAGE)
        md.append(f"- `{CAL_BETA.name}` (Beta)")
        md.append(f"- `{CAL_2STAGE.name}` (Platt + isotonic)")
    else:
        md.append(f"\n❌ **大きな改善なし** (差 < 2pp)。")
        md.append(f"calibration では本質的な解決にならない。")
        md.append(f"\n**次の選択肢**:")
        md.append(f"- Option B: rank-based selection (絶対 EV 捨てて順位で買う)")
        md.append(f"- Option C: v6 再学習 (sample_weight=None で再構築)")

    md_text = "\n".join(md)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(md_text)
    print(f"\n[saved] {OUT_MD}")
    print()
    print(md_text)


if __name__ == "__main__":
    main()
