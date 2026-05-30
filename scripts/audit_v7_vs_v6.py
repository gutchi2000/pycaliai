"""
audit_v7_vs_v6.py
=================
v6 と v7 の高 EV 帯 calibration とワイド ROI を直接比較。

両モデルで test 2024-2025 の predictions を生成 → audit_ev_calibration と
同じロジックで EV ビン別 ROI を測定 → 並列比較。ワイド box 3 点 ROI も計測。

最終判断 (v7 採用基準):
  ワイド box 3 点 ROI が v6 比 +0.05 以上改善 (Cowork ワイド黒字化の本命)
  または 複勝/馬連 ECE が v6 比 維持以上 で 単勝/複勝 高 EV ROI 改善
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
MODEL_V6   = BASE / "models" / "unified_rank_v6.pkl"
MODEL_V7   = BASE / "models" / "unified_rank_v7.pkl"
CAL_V6     = BASE / "models" / "pl_calibrators_v6.pkl"
CAL_V7     = BASE / "models" / "pl_calibrators_v7.pkl"
WIDE_PQ    = BASE / "data" / "wide_payouts_2016-2025.parquet"
ODDS_2024  = BASE / "data" / "odds" / "odds_20240106-20241228.csv"
ODDS_2025  = BASE / "data" / "odds" / "odds_20250105-20261228.csv"
OUT_MD     = BASE / "reports" / f"audit_v7_vs_v6_{datetime.now():%Y%m%d}.md"

COL_RID  = "レースID(新/馬番無)"
COL_JYUN = "着順"
COL_BAN  = "馬番"


def predict_with_model(df_test: pd.DataFrame, model_pkl: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """test data に v6/v7 を適用、PL raw p_win / p_sho / score を返す。

    bundle.race_relative_mode が設定されているモデル (v7+) では、
    race_relative_feats をその場で適用してから feats を取り出す。
    冪等なので複数回呼んでも結果は同じ。
    """
    print(f"[load] {model_pkl.name}")
    bundle = joblib.load(model_pkl)
    model = bundle["model"]
    feats = bundle["feature_cols"]
    encs = bundle["encoders"]
    rr_mode = bundle.get("race_relative_mode")
    print(f"  feats={len(feats)}  race_relative_mode={rr_mode}")

    df = df_test.copy()
    if rr_mode:
        from race_relative_feats import add_race_relative_feats
        df = add_race_relative_feats(df, mode=rr_mode)

    for c, le in encs.items():
        if c not in df.columns:
            continue
        v = df[c].astype(str).fillna("__NaN__")
        known = set(le.classes_)
        v = v.where(v.isin(known), "__NaN__")
        df[c] = le.transform(v)

    X = df[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    scores = model.predict(X)

    p_win_raw = np.zeros(len(df))
    p_sho_raw = np.zeros(len(df))
    rids = df[COL_RID].values
    indices_by_rid = {}
    for i, rid in enumerate(rids):
        indices_by_rid.setdefault(rid, []).append(i)
    for rid, idx_list in indices_by_rid.items():
        idx = np.array(idx_list)
        s = scores[idx]
        w = PL.pl_weights(s)
        p_win_raw[idx] = PL.all_tansho(w)
        p_sho_raw[idx] = PL.all_fukusho(w)
    return p_win_raw, p_sho_raw, scores


def load_wide_payouts():
    p = pd.read_parquet(WIDE_PQ)
    out = {}
    for r in p.itertuples(index=False):
        rid = str(r.race_id)[:16]
        out[rid] = [
            (int(r.w1_i), int(r.w1_j), int(r.w1_pay)),
            (int(r.w2_i), int(r.w2_j), int(r.w2_pay)),
            (int(r.w3_i), int(r.w3_j), int(r.w3_pay)),
        ]
    return out


def wide_box3_roi(df: pd.DataFrame, score_col: str, wide_pays: dict) -> dict:
    """◎〇▲ box 3 点流しの ROI を計算する (馬番ベース)。"""
    n_race = 0
    stake = 0
    ret = 0
    hits = 0
    for rid, g in df.groupby(COL_RID, sort=False):
        if len(g) < 5: continue
        g = g.sort_values(COL_BAN).reset_index(drop=True)
        rid_s = str(rid)[:16] if not isinstance(rid, (int, np.integer)) else str(int(rid))
        if rid_s not in wide_pays:
            continue
        scores = g[score_col].values
        ban = pd.to_numeric(g[COL_BAN], errors="coerce").astype(int).values
        order = np.argsort(-scores)
        hon_ban = int(ban[order[0]])
        tai_ban = int(ban[order[1]])
        san_ban = int(ban[order[2]])
        box = {
            tuple(sorted((hon_ban, tai_ban))),
            tuple(sorted((hon_ban, san_ban))),
            tuple(sorted((tai_ban, san_ban))),
        }
        pay_map = {tuple(sorted((wi, wj))): wp for (wi, wj, wp) in wide_pays[rid_s]}
        n_race += 1
        stake += 300
        for pair in box:
            if pair in pay_map:
                ret += pay_map[pair]
                hits += 1
    return {
        "n_race": n_race,
        "stake": stake,
        "return": ret,
        "roi": (ret / stake) if stake else 0.0,
        "hit_rate": (hits / (3 * n_race)) if n_race else 0.0,
        "n_hits": hits,
    }


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


def format_bin_table(rows, label):
    out = [f"\n### {label}", "", "| EV帯 | n | p_obs | p_est | ROI |", "|---|---|---|---|---|"]
    for r in rows:
        if r["n"] == 0:
            out.append(f"| {r['EV帯']} | 0 | - | - | - |")
        else:
            out.append(f"| {r['EV帯']} | {r['n']:,} | {r['p_obs']:.4f} | {r['p_est']:.4f} | "
                       f"{r['ROI']:.3f} |")
    return "\n".join(out)


def apply_isotonic(p_raw_v: np.ndarray, p_raw_t: np.ndarray,
                   y_v: np.ndarray) -> np.ndarray:
    """valid raw → fit, test に apply。"""
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p_raw_v, y_v)
    return iso.predict(p_raw_t)


def main():
    print("=" * 70)
    print("[1/5] master 読み込み")
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df["日付"] = pd.to_numeric(df["日付"], errors="coerce").astype("Int64")
    df = df[df["split"].isin(["valid", "test"])].copy()
    df = df.dropna(subset=[COL_JYUN, COL_RID, "日付"]).reset_index(drop=True)
    print(f"  rows: {len(df):,} / races: {df[COL_RID].nunique():,}")

    print("\n[2/5] v6 推論")
    p_win_v6, p_sho_v6, sc_v6 = predict_with_model(df, MODEL_V6)
    df["p_win_v6_raw"] = p_win_v6
    df["p_sho_v6_raw"] = p_sho_v6
    df["score_v6"] = sc_v6

    print("\n[3/5] v7 推論")
    p_win_v7, p_sho_v7, sc_v7 = predict_with_model(df, MODEL_V7)
    df["p_win_v7_raw"] = p_win_v7
    df["p_sho_v7_raw"] = p_sho_v7
    df["score_v7"] = sc_v7

    # valid で fit, test に apply
    valid_mask = df["split"] == "valid"
    test_mask  = df["split"] == "test"
    y_win_v = (df.loc[valid_mask, COL_JYUN] == 1).astype(int).values
    y_sho_v = (df.loc[valid_mask, COL_JYUN] <= 3).astype(int).values
    df["p_win_v6_cal"] = np.nan
    df["p_sho_v6_cal"] = np.nan
    df["p_win_v7_cal"] = np.nan
    df["p_sho_v7_cal"] = np.nan
    df.loc[test_mask, "p_win_v6_cal"] = apply_isotonic(
        df.loc[valid_mask, "p_win_v6_raw"].values,
        df.loc[test_mask,  "p_win_v6_raw"].values, y_win_v)
    df.loc[test_mask, "p_sho_v6_cal"] = apply_isotonic(
        df.loc[valid_mask, "p_sho_v6_raw"].values,
        df.loc[test_mask,  "p_sho_v6_raw"].values, y_sho_v)
    df.loc[test_mask, "p_win_v7_cal"] = apply_isotonic(
        df.loc[valid_mask, "p_win_v7_raw"].values,
        df.loc[test_mask,  "p_win_v7_raw"].values, y_win_v)
    df.loc[test_mask, "p_sho_v7_cal"] = apply_isotonic(
        df.loc[valid_mask, "p_sho_v7_raw"].values,
        df.loc[test_mask,  "p_sho_v7_raw"].values, y_sho_v)

    print("\n[4/5] オッズ JOIN")
    df_test = df[test_mask].copy()
    df_test = merge_odds(df_test)

    print("\n[5/5] EV ビン audit + ワイド box ROI")
    wide_pays = load_wide_payouts()
    out_lines = [f"# audit_v7_vs_v6  ({datetime.now():%Y-%m-%d %H:%M})", ""]
    out_lines.append("## 単勝 EV ビン別 ROI (test 2024-2025)")
    out_lines.append(format_bin_table(audit_bins(df_test, "p_win_v6_cal", "tansho"), "v6 (calibrated)"))
    out_lines.append(format_bin_table(audit_bins(df_test, "p_win_v7_cal", "tansho"), "v7 (calibrated)"))
    out_lines.append("\n## 複勝 EV ビン別 ROI (test 2024-2025)")
    out_lines.append(format_bin_table(audit_bins(df_test, "p_sho_v6_cal", "fukusho"), "v6 (calibrated)"))
    out_lines.append(format_bin_table(audit_bins(df_test, "p_sho_v7_cal", "fukusho"), "v7 (calibrated)"))

    print("\n--- ワイド box 3 点流し ROI (test 2024-2025) ---")
    r_v6 = wide_box3_roi(df_test, "score_v6", wide_pays)
    r_v7 = wide_box3_roi(df_test, "score_v7", wide_pays)
    delta_roi = r_v7["roi"] - r_v6["roi"]
    print(f"  v6: races={r_v6['n_race']:,}  stake={r_v6['stake']:,}  ret={r_v6['return']:,}  "
          f"ROI={r_v6['roi']*100:.2f}%  hit={r_v6['hit_rate']*100:.2f}%")
    print(f"  v7: races={r_v7['n_race']:,}  stake={r_v7['stake']:,}  ret={r_v7['return']:,}  "
          f"ROI={r_v7['roi']*100:.2f}%  hit={r_v7['hit_rate']*100:.2f}%")
    print(f"  Δ ROI = {delta_roi*100:+.2f} ppt")

    out_lines.append("\n## ワイド box 3 点流し ROI (test 2024-2025)")
    out_lines.append("| model | races | stake | return | ROI | hit_rate |")
    out_lines.append("|---|---|---|---|---|---|")
    out_lines.append(f"| v6 | {r_v6['n_race']:,} | {r_v6['stake']:,} | {r_v6['return']:,} | "
                     f"{r_v6['roi']*100:.2f}% | {r_v6['hit_rate']*100:.2f}% |")
    out_lines.append(f"| v7 | {r_v7['n_race']:,} | {r_v7['stake']:,} | {r_v7['return']:,} | "
                     f"{r_v7['roi']*100:.2f}% | {r_v7['hit_rate']*100:.2f}% |")
    out_lines.append(f"\n**Δ ROI (v7 - v6) = {delta_roi*100:+.2f} ppt**")

    if delta_roi >= 0.05:
        out_lines.append("\n>  v7 採用候補 (ワイド ROI +0.05 以上改善)")
    else:
        out_lines.append("\n>  ワイド ROI 改善は +0.05 未満。複勝/単勝 calibration とのトレードオフ要検討。")

    OUT_MD.parent.mkdir(exist_ok=True)
    OUT_MD.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"\n[saved] {OUT_MD}")


if __name__ == "__main__":
    main()
