"""
audit_ev_calibration.py
=======================
v5 model の EV ビン別 calibration audit。
他 AI フィードバック Task 4 対応。

目的:
  「EV >= 1.10 の馬券は本当に + 期待値か?」を実測で確認。
  EV 帯ごとに p_estimate vs p_observed を比較し、
  どの EV 帯で model が信頼できるかを判定する。

出力:
  reports/ev_calibration_audit_{YYYYMMDD}.md (Markdown レポート)
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

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# pl_probs は親ディレクトリにあるので path 追加
sys.path.insert(0, str(Path(__file__).parent.parent))
import pl_probs as PL

BASE       = Path(__file__).parent.parent
MASTER_CSV = BASE / "data" / "master_v2_20130105-20251228.csv"
MODEL_PKL  = BASE / "models" / "unified_rank_v5.pkl"
CAL_PKL    = BASE / "models" / "pl_calibrators_v5.pkl"
ODDS_2024  = BASE / "data" / "odds" / "odds_20240106-20241228.csv"
ODDS_2025  = BASE / "data" / "odds" / "odds_20250105-20261228.csv"
OUT_MD     = BASE / "reports" / f"ev_calibration_audit_{datetime.now():%Y%m%d}.md"

COL_RID  = "レースID(新/馬番無)"
COL_RID_FULL = "レースID(新)"
COL_JYUN = "着順"
COL_BAN  = "馬番"


def wilson_ci(p_hat: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval (95%)。"""
    if n == 0:
        return 0.0, 0.0
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return max(0, center - half), min(1, center + half)


def load_test_with_predictions():
    """test split + v5 model 推論 + calibrator 適用結果を返す。"""
    print("=" * 70)
    print("[1/5] test 期間 (2024-2025) の master 読み込み")
    print("=" * 70)
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    df = df[df["split"] == "test"].copy()
    df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df = df.dropna(subset=[COL_JYUN, COL_RID]).copy()
    print(f"  test rows: {len(df):,} / races: {df[COL_RID].nunique():,}")

    # v5 モデル & calibrator
    bundle = joblib.load(MODEL_PKL)
    model = bundle["model"]
    feats = bundle["feature_cols"]
    encs = bundle["encoders"]
    cal_bundle = joblib.load(CAL_PKL)
    cals = cal_bundle["calibrators"]
    print(f"[2/5] model: {MODEL_PKL.name}, feats={len(feats)}")
    print(f"      calibrator: {CAL_PKL.name}")

    # encode + predict
    print("[3/5] encoding + model.predict ...")
    for c, le in encs.items():
        if c not in df.columns:
            continue
        v = df[c].astype(str).fillna("__NaN__")
        known = set(le.classes_)
        v = v.where(v.isin(known), "__NaN__")
        df[c] = le.transform(v)

    X = df[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    df["_score"] = model.predict(X)

    # race ごとに PL → calibrated p_win, p_sho
    print("[4/5] PL probability + calibration (race ごと) ...")
    p_win_all = np.zeros(len(df))
    p_sho_all = np.zeros(len(df))

    rids = df[COL_RID].values
    scores = df["_score"].values
    indices_by_rid = {}
    for i, rid in enumerate(rids):
        indices_by_rid.setdefault(rid, []).append(i)

    for rid, idx_list in indices_by_rid.items():
        idx = np.array(idx_list)
        s = scores[idx]
        w = PL.pl_weights(s)
        p_win_raw = PL.all_tansho(w)
        p_sho_raw = PL.all_fukusho(w)
        # calibrate
        p_win_cal = cals["tansho"].predict(p_win_raw)
        p_sho_cal = cals["fukusho"].predict(p_sho_raw)
        p_win_all[idx] = p_win_cal
        p_sho_all[idx] = p_sho_cal

    df["p_win"] = p_win_all
    df["p_sho"] = p_sho_all
    return df


def merge_odds(df: pd.DataFrame):
    """odds_2024, odds_2025 を結合。レースID(新) と 馬番でマージ。"""
    print("[5/5] オッズ 2024-2025 マージ ...")
    odds_24 = pd.read_csv(ODDS_2024, encoding="cp932", dtype=str)
    odds_25 = pd.read_csv(ODDS_2025, encoding="cp932", dtype=str)
    odds = pd.concat([odds_24, odds_25], ignore_index=True)
    print(f"  odds rows: {len(odds):,}")

    # 列名統一
    odds = odds.rename(columns={
        COL_RID_FULL:   "rid_full",
        "馬番":         "ban",
        "単勝オッズ":   "tansho_odds",
        "複勝オッズ下限": "fuku_low",
        "複勝オッズ上限": "fuku_high",
    })
    odds["ban"] = pd.to_numeric(odds["ban"], errors="coerce").astype("Int64")
    odds["tansho_odds"] = pd.to_numeric(odds["tansho_odds"], errors="coerce")
    odds["fuku_low"] = pd.to_numeric(odds["fuku_low"], errors="coerce")
    odds["fuku_high"] = pd.to_numeric(odds["fuku_high"], errors="coerce")

    # master 側の race_id (16 桁) と odds 側の レースID(新) (18 桁) のマッピング:
    # レースID(新) = 馬番付き 18 桁。先頭 16 桁が race_id (馬番なし) と一致。
    odds["rid_no_ban"] = odds["rid_full"].astype(str).str[:16]

    # df 側の COL_RID (16 桁) と ban で merge
    df["_ban_int"] = pd.to_numeric(df[COL_BAN], errors="coerce").astype("Int64")
    df["_rid_str"] = df[COL_RID].astype(str)

    merged = df.merge(
        odds[["rid_no_ban", "ban", "tansho_odds", "fuku_low", "fuku_high"]],
        left_on=["_rid_str", "_ban_int"],
        right_on=["rid_no_ban", "ban"],
        how="left",
    )
    n_matched = merged["tansho_odds"].notna().sum()
    print(f"  オッズマッチ: {n_matched:,} / {len(df):,} ({n_matched/len(df):.1%})")
    return merged


def audit_bins(df: pd.DataFrame, bet_type: str):
    """単一馬券種 (tansho or fukusho) の EV ビン別 calibration."""
    if bet_type == "tansho":
        df = df.dropna(subset=["tansho_odds", "p_win"]).copy()
        df["EV"] = df["p_win"] * df["tansho_odds"]
        df["hit"] = (df[COL_JYUN] == 1).astype(int)
        p_col = "p_win"
        odds_col = "tansho_odds"
    elif bet_type == "fukusho":
        df = df.dropna(subset=["fuku_low", "fuku_high", "p_sho"]).copy()
        df["fuku_mid"] = (df["fuku_low"] + df["fuku_high"]) / 2
        df["EV"] = df["p_sho"] * df["fuku_mid"]
        df["hit"] = (df[COL_JYUN] <= 3).astype(int)
        p_col = "p_sho"
        odds_col = "fuku_mid"
    else:
        raise ValueError(bet_type)

    bins = [(0, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.05),
            (1.05, 1.20), (1.20, 1.50), (1.50, 2.00), (2.00, 999)]

    rows = []
    for lo, hi in bins:
        mask = (df["EV"] >= lo) & (df["EV"] < hi)
        sub = df[mask]
        n = len(sub)
        if n == 0:
            rows.append({
                "EV帯": f"[{lo}, {hi})", "n": 0, "p_est_avg": np.nan,
                "p_obs": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "ROI": np.nan, "diff": np.nan,
            })
            continue
        p_est_avg = sub[p_col].mean()
        p_obs = sub["hit"].mean()
        ci_lo, ci_hi = wilson_ci(p_obs, n)
        # ROI: 単勝なら hit時の払戻 = odds × stake、外れは -stake
        # = (Σ hit × odds - n) / n  (per ¥1 stake)
        roi = (sub["hit"] * sub[odds_col]).sum() / n
        rows.append({
            "EV帯": f"[{lo}, {hi})",
            "n": n,
            "p_est_avg": p_est_avg,
            "p_obs": p_obs,
            "ci_low": ci_lo, "ci_high": ci_hi,
            "ROI": roi,
            "diff": p_obs - p_est_avg,
        })
    return rows


def render_report(tansho_rows, fukusho_rows, n_total) -> str:
    md = []
    md.append(f"# v5 EV ビン別 Calibration Audit\n")
    md.append(f"**実施日**: {datetime.now():%Y-%m-%d %H:%M}")
    md.append(f"**対象**: test 2024-2025 の {n_total:,} 馬")
    md.append(f"**目的**: EV 帯ごとに `p_estimate` (model + calibrator) vs `p_observed` (実測) を比較\n")

    md.append("## サマリ\n")
    md.append("`diff = p_observed - p_estimate`")
    md.append("- diff > 0: model が **過小** 評価 (実測 > 予測) → calibrator 上方修正必要")
    md.append("- diff < 0: model が **過大** 評価 (実測 < 予測) → 危険、その EV 帯のベットは控える")
    md.append("- |diff| < 0.02: ほぼ calibration OK\n")

    for name, rows in [("単勝", tansho_rows), ("複勝", fukusho_rows)]:
        md.append(f"## {name} EV 帯別\n")
        md.append("| EV 帯 | n | p_est | p_obs | 95% CI | ROI | diff |")
        md.append("|------|----|-------|-------|--------|-----|------|")
        for r in rows:
            if r["n"] == 0:
                md.append(f"| {r['EV帯']} | 0 | — | — | — | — | — |")
                continue
            ci = f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]"
            md.append(
                f"| {r['EV帯']} | {r['n']:,} | "
                f"{r['p_est_avg']:.3f} | **{r['p_obs']:.3f}** | "
                f"{ci} | **{r['ROI']:.3f}** | "
                f"{r['diff']:+.3f} |"
            )
        md.append("")

        # 解釈
        md.append(f"### {name} の解釈\n")
        good = [r for r in rows if r["n"] > 100 and r["ROI"] >= 1.0]
        bad  = [r for r in rows if r["n"] > 100 and r["ROI"] < 0.85]
        if good:
            md.append(f"**+EV 帯 (ROI≥1.0)**: {', '.join(r['EV帯'] for r in good)}")
        if bad:
            md.append(f"**危険帯 (ROI<0.85)**: {', '.join(r['EV帯'] for r in bad)}")
        md.append("")

        # サンプル数注意
        small = [r for r in rows if r["n"] < 100 and r["n"] > 0]
        if small:
            md.append(f"⚠️ サンプル少 (n<100): {', '.join(r['EV帯'] for r in small)}\n")

    # 総合判定
    md.append("## 総合判定\n")

    # 単勝: 各帯で ROI を見る
    tan_buy_zones = []
    tan_avoid_zones = []
    for r in tansho_rows:
        if r["n"] >= 50:
            if r["ROI"] >= 1.05:
                tan_buy_zones.append(r["EV帯"])
            elif r["ROI"] < 0.85:
                tan_avoid_zones.append(r["EV帯"])

    fuku_buy_zones = []
    fuku_avoid_zones = []
    for r in fukusho_rows:
        if r["n"] >= 50:
            if r["ROI"] >= 1.05:
                fuku_buy_zones.append(r["EV帯"])
            elif r["ROI"] < 0.85:
                fuku_avoid_zones.append(r["EV帯"])

    md.append(f"### 単勝")
    md.append(f"- **買うべき EV 帯**: {tan_buy_zones if tan_buy_zones else '無し'}")
    md.append(f"- **避けるべき EV 帯**: {tan_avoid_zones if tan_avoid_zones else '無し'}\n")

    md.append(f"### 複勝")
    md.append(f"- **買うべき EV 帯**: {fuku_buy_zones if fuku_buy_zones else '無し'}")
    md.append(f"- **避けるべき EV 帯**: {fuku_avoid_zones if fuku_avoid_zones else '無し'}\n")

    md.append("## 推奨アクション (Cowork prompt 反映)\n")
    md.append(f"`docs/cowork_prompt.md` の動的 EV 閾値を、上記実測 ROI に基づき調整:\n")
    md.append(f"- ROI ≥ 1.05 を満たす最低 EV を新閾値とする")
    md.append(f"- ROI < 0.85 の帯は強制的に抜く (購入額 ¥0)\n")

    return "\n".join(md)


def main():
    df = load_test_with_predictions()
    df = merge_odds(df)
    n_total = len(df)
    tansho_rows = audit_bins(df, "tansho")
    fukusho_rows = audit_bins(df, "fukusho")
    md = render_report(tansho_rows, fukusho_rows, n_total)

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\n[saved] {OUT_MD}")
    print()
    print(md)


if __name__ == "__main__":
    main()
