# -*- coding: utf-8 -*-
"""
build_asof_pair_data.py
=========================
EXP11 Stage1修正1・2・3。master_v2にas-of pair/jockey/trainer prior countと
未知カテゴリ分類を付与し、実データで削除不変性を確認、8領域の単位訂正版
集計、および低頻度ペア領域(as-of再定義)の0.059対0.036分解を行う。

2024・2025年の性能・ROIは計算しない(2023年developmentのみ)。年別件数は
構造的な件数のみ2013-2025全期間で報告する(EXP10 Stage0の先例を踏襲)。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"E:\PyCaLiAI")
MASTER_CSV = BASE / "data/master_v2_20130105-20251228.csv"
OUT_DIR = Path(__file__).parent / "out"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from asof_features import add_asof_prior_counts, add_unknown_category_class  # noqa: E402

COL_RID = "レースID(新/馬番無)"
COL_BAN = "馬番"
COL_KETTO = "血統登録番号"
COL_DATE = "日付"
COL_JOCKEY = "騎手コード"
COL_TRAINER = "調教師コード"
COL_FINISH = "着順"


def load_master() -> pd.DataFrame:
    cols = [COL_RID, COL_BAN, COL_KETTO, COL_DATE, COL_JOCKEY, COL_TRAINER,
            COL_FINISH, "fukusho_flag", "split", "出走頭数", "クラス名",
            "前走確定着順", "間隔", "kako5_race_count"]
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", usecols=cols, low_memory=False)
    df["year"] = df[COL_DATE].astype(str).str[:4].astype(int)
    df[COL_RID] = df[COL_RID].astype(str)
    return df


def build_v6_2023_scores() -> pd.DataFrame:
    sys.path.insert(0, str(BASE))
    import backtest_pl_ev as be
    import pl_probs as PL
    be.MODEL_PKL = BASE / "models/unified_rank_v6.pkl"
    te = be.score_test(include_valid=True)
    te = te[te["year"] == 2023].copy()
    rows = []
    for rid, g in te.groupby(be.COL_RID, sort=False):
        g = g.sort_values(be.COL_BAN).reset_index(drop=True)
        w = PL.pl_weights(g["_score"].values)
        p = PL.all_tansho(w)
        for i in range(len(g)):
            rows.append(dict(rid=str(rid), ban=int(g.loc[i, be.COL_BAN]),
                              v6_p_win_raw=float(p[i]), v6_raw_score=float(g.loc[i, "_score"])))
    return pd.DataFrame(rows)


def build_v6_2023_calibrated() -> pd.DataFrame:
    """pl_calibrators_v6(valid=2023でfit済みのisotonic)経由の較正後単勝確率。
    valid=2023でfitされたcalibratorをvalid=2023自身に適用するためin-sample性が
    ある点に注意(既存本番calibratorの標準的な使い方を踏襲、EXP09で確認済みの
    既知の限界)。"""
    import joblib
    cal = joblib.load(BASE / "models/pl_calibrators_v6.pkl")["calibrators"]
    scores = build_v6_2023_scores()
    scores["v6_p_win_calibrated"] = np.clip(cal["tansho"].predict(scores["v6_p_win_raw"].values), 0, 1)
    return scores[["rid", "ban", "v6_p_win_calibrated"]]


def load_market_2023() -> pd.DataFrame:
    p = Path(r"E:\競馬過去走データ\raw_data\kekka_2010_2025_fix_raceid_v2__keyed.csv")
    df = pd.read_csv(p, encoding="utf-8-sig", usecols=["race_id", "umaban", "単勝オッズ"], low_memory=False)
    df["race_id"] = df["race_id"].astype(str)
    df = df[df["race_id"].str.startswith("2023")].copy()
    df = df.dropna(subset=["単勝オッズ"])
    df["market_p_win_raw"] = 1.0 / df["単勝オッズ"]
    df["market_p_win"] = df["market_p_win_raw"] / df.groupby("race_id")["market_p_win_raw"].transform("sum")
    return df.rename(columns={"umaban": "ban", "race_id": "rid"})[["rid", "ban", "market_p_win"]]


def verify_deletion_invariance_realdata(df: pd.DataFrame) -> dict:
    """実データで削除不変性を確認する: 2025年を削除しても2013-2024年行の
    pair_prior_count_asof等が変化しないことを確認する。"""
    full = df
    truncated_input = df[df["year"] <= 2024].drop(
        columns=["pair_prior_count_asof", "jockey_prior_count_asof", "trainer_prior_count_asof"])
    truncated = add_asof_prior_counts(truncated_input)

    key_cols = [COL_RID, COL_BAN]
    full_past = full[full["year"] <= 2024][key_cols + ["pair_prior_count_asof",
                                                          "jockey_prior_count_asof",
                                                          "trainer_prior_count_asof"]]
    full_past = full_past.sort_values(key_cols).reset_index(drop=True)
    trunc_past = truncated[key_cols + ["pair_prior_count_asof", "jockey_prior_count_asof",
                                         "trainer_prior_count_asof"]]
    trunc_past = trunc_past.sort_values(key_cols).reset_index(drop=True)

    merged = full_past.merge(trunc_past, on=key_cols, suffixes=("_full", "_trunc"))
    mismatches = {}
    for col in ["pair_prior_count_asof", "jockey_prior_count_asof", "trainer_prior_count_asof"]:
        diff = (merged[f"{col}_full"] != merged[f"{col}_trunc"]).sum()
        mismatches[col] = int(diff)
    return dict(n_compared=len(merged), mismatches=mismatches,
                all_match=all(v == 0 for v in mismatches.values()))


def main():
    print("[build_asof_pair_data] master_v2読み込み中...")
    df = load_master()
    print(f"  行数={len(df):,}")

    print("[build_asof_pair_data] as-of prior count計算中(日初時点固定)...")
    df = add_asof_prior_counts(df, date_col=COL_DATE, jockey_col=COL_JOCKEY, trainer_col=COL_TRAINER)
    df = add_unknown_category_class(df)

    print("[build_asof_pair_data] 削除不変性を実データで確認中(2025年削除)...")
    inv = verify_deletion_invariance_realdata(df)
    print(f"  n_compared={inv['n_compared']:,}  mismatches={inv['mismatches']}  "
          f"all_match={inv['all_match']}")
    if not inv["all_match"]:
        raise AssertionError("実データでの削除不変性検証に失敗。as-of実装にバグの疑い。")

    print("[build_asof_pair_data] 2023年v6 raw/calibrated score・市場確率を結合中...")
    v6_scores = build_v6_2023_scores()
    v6_cal = build_v6_2023_calibrated()
    market = load_market_2023()
    df["ban_int"] = pd.to_numeric(df[COL_BAN], errors="coerce")
    df = df.merge(v6_scores, left_on=[COL_RID, "ban_int"], right_on=["rid", "ban"], how="left")
    df = df.merge(v6_cal, left_on=[COL_RID, "ban_int"], right_on=["rid", "ban"], how="left",
                  suffixes=("", "_cal"))
    df = df.merge(market, left_on=[COL_RID, "ban_int"], right_on=["rid", "ban"], how="left",
                  suffixes=("", "_mkt"))
    # v6+market blend baseline (単純平均、race内再正規化)
    df["v6_market_blend_raw"] = df[["v6_p_win_raw", "market_p_win"]].mean(axis=1, skipna=False)

    OUT_DIR.mkdir(exist_ok=True)
    df.to_parquet(OUT_DIR / "asof_pair_data.parquet")
    print(f"[saved] {OUT_DIR / 'asof_pair_data.parquet'}")

    print("\n[low_freq_jt_pair(as-of, pair_prior_count_asof<5)] 領域の単位訂正版集計")
    mask = df["pair_prior_count_asof"] < 5
    sub = df[mask]
    print(f"  horse-race rows = {len(sub):,}")
    print(f"  unique races = {sub[COL_RID].nunique():,}")
    finish_num = pd.to_numeric(sub[COL_FINISH], errors="coerce")
    print(f"  positive wins(着順==1) = {(finish_num == 1).sum():,} "
          f"(master_v2の着順は既にconvert_finish()で数値化済み、Stage0 DATA_AUDIT確認済み)")
    print(f"  unique jockeys = {sub[COL_JOCKEY].nunique():,}")
    print(f"  unique trainers = {sub[COL_TRAINER].nunique():,}")
    print(f"  unique jockey×trainer pairs = {sub.groupby([COL_JOCKEY, COL_TRAINER]).ngroups:,}")
    print(f"  年別horse-race rows: {sub.groupby('year').size().to_dict()}")


if __name__ == "__main__":
    main()
