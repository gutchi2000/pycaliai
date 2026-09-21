# -*- coding: utf-8 -*-
"""
sparse_region_audit.py
========================
EXP11 Stage0監査。事前固定した8つの疎データ領域について、年別件数・
既存特徴の有無・(2023年developmentのみでの)v6実力・市場確率との関係を集計する。
2024・2025年の性能・ROIは計算しない(件数のみ2013-2025全期間で報告可、
これはEXP10 Stage0の先例=構造的件数は性能指標ではないという扱いを踏襲)。

対象領域:
  1. career starts 0-2
  2. career starts 3-5
  3. 初距離(このレースの距離をこの馬が過去に一度も走ったことがない)
  4. 初コース(場所×芝ダの組み合わせが初めて)
  5. 昇級初戦(クラスが前走比で上がった最初のレース、前走クラスは自己結合で取得)
  6. 転厩後初戦(調教師コードが前走と異なる、前走調教師コードは自己結合で取得)
  7. 騎手×調教師ペアの低頻度(全期間の共演回数が閾値未満)
  8. 種牡馬産駒の低サンプル条件(種牡馬×芝ダ×距離帯の組み合わせサンプル数が閾値未満)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"E:\PyCaLiAI")
MASTER_CSV = BASE / "data/master_v2_20130105-20251228.csv"

COL_RID = "レースID(新/馬番無)"
COL_BAN = "馬番"
COL_KETTO = "血統登録番号"
COL_DATE = "日付"
COL_TIME = "発走時刻"
COL_CLASS = "クラス名"
COL_TRAINER = "調教師コード"
COL_JOCKEY = "騎手コード"
COL_PREV_RID = "前走レースID(新/馬番無)"
COL_DIST = "距離"
COL_SURF = "芝・ダ"
COL_PLACE = "場所"
COL_SIRE = "種牡馬"

CLASS_TIER = {
    "新馬": 0, "未勝利": 0,
    "500万": 1, "1勝": 1,
    "1000万": 2, "2勝": 2,
    "1600万": 3, "3勝": 3,
    "ｵｰﾌﾟﾝ": 4, "OP(L)": 4,
    "Ｇ３": 5, "ＪＧ３": 5, "重賞": 5,
    "Ｇ２": 6, "ＪＧ２": 6,
    "Ｇ１": 7, "ＪＧ１": 7,
}

LOW_FREQ_JOCKEY_TRAINER_THRESHOLD = 5   # 全期間の共演回数がこれ未満
LOW_SAMPLE_SIRE_COND_THRESHOLD = 30     # 種牡馬×芝ダ×距離帯のサンプル数がこれ未満


def load_master() -> pd.DataFrame:
    cols = [COL_RID, COL_BAN, COL_KETTO, COL_DATE, COL_TIME, COL_CLASS,
            COL_TRAINER, COL_JOCKEY, COL_PREV_RID, COL_DIST, COL_SURF,
            COL_PLACE, COL_SIRE, "着順", "fukusho_flag", "split"]
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", usecols=cols, low_memory=False)
    df["year"] = df[COL_DATE].astype(str).str[:4].astype(int)
    df[COL_RID] = df[COL_RID].astype(str)
    df[COL_PREV_RID] = df[COL_PREV_RID].astype(str)
    df = df.sort_values([COL_KETTO, COL_DATE, COL_TIME]).reset_index(drop=True)
    return df


def add_career_start_count(df: pd.DataFrame) -> pd.DataFrame:
    """このレースが、この馬にとって何走目か(0-indexed、0=デビュー戦)。"""
    df["career_start_idx"] = df.groupby(COL_KETTO).cumcount()
    return df


def add_first_time_distance(df: pd.DataFrame) -> pd.DataFrame:
    """このレースの距離を、この馬が過去に一度も走ったことがないか(1回目のデビュー戦は
    定義上True=初距離とする)。単純パスで距離集合を追跡する。"""
    seen: dict[str, set] = {}
    out = np.zeros(len(df), dtype=bool)
    for i, (ketto, dist) in enumerate(zip(df[COL_KETTO].values, df[COL_DIST].values)):
        s = seen.setdefault(ketto, set())
        out[i] = dist not in s
        s.add(dist)
    df["is_first_time_distance"] = out
    return df


def add_first_time_course(df: pd.DataFrame) -> pd.DataFrame:
    """場所×芝ダの組み合わせがこの馬にとって初めてか。"""
    seen: dict[str, set] = {}
    out = np.zeros(len(df), dtype=bool)
    for i, (ketto, place, surf) in enumerate(zip(df[COL_KETTO].values, df[COL_PLACE].values, df[COL_SURF].values)):
        key = (place, surf)
        s = seen.setdefault(ketto, set())
        out[i] = key not in s
        s.add(key)
    df["is_first_time_course"] = out
    return df


def add_class_up_and_stable_transfer(df: pd.DataFrame) -> pd.DataFrame:
    """前走の(レースID(新/馬番無), 血統登録番号)をキーに自己結合し、前走クラス・
    前走調教師コードを取得する(master_v2に前走クラス列が無いための補完)。"""
    lookup = df[[COL_RID, COL_KETTO, COL_CLASS, COL_TRAINER]].copy()
    lookup = lookup.rename(columns={COL_RID: COL_PREV_RID, COL_CLASS: "prev_class_selfjoin",
                                     COL_TRAINER: "prev_trainer_selfjoin"})
    lookup = lookup.drop_duplicates(subset=[COL_PREV_RID, COL_KETTO])
    df = df.merge(lookup, on=[COL_PREV_RID, COL_KETTO], how="left")

    df["cur_tier"] = df[COL_CLASS].map(CLASS_TIER)
    df["prev_tier"] = df["prev_class_selfjoin"].map(CLASS_TIER)
    df["is_class_up"] = (df["prev_tier"].notna()) & (df["cur_tier"] > df["prev_tier"])
    df["is_stable_transfer"] = (
        df["prev_trainer_selfjoin"].notna()
        & (df[COL_TRAINER] != df["prev_trainer_selfjoin"])
    )
    return df


def add_low_freq_jockey_trainer(df: pd.DataFrame) -> pd.DataFrame:
    """騎手×調教師ペアの全期間共演回数(as-ofではなく全期間集計、閾値判定用の
    記述統計。時点安全なas-of版は別途Stage1で検討)。"""
    pair_counts = df.groupby([COL_JOCKEY, COL_TRAINER]).size().rename("jt_pair_count_fullperiod")
    df = df.merge(pair_counts, on=[COL_JOCKEY, COL_TRAINER], how="left")
    df["is_low_freq_jt_pair"] = df["jt_pair_count_fullperiod"] < LOW_FREQ_JOCKEY_TRAINER_THRESHOLD
    return df


def add_low_sample_sire_condition(df: pd.DataFrame) -> pd.DataFrame:
    dist_bucket = pd.cut(df[COL_DIST], [0, 1400, 1800, 2200, 2600, 9999],
                          labels=["~1400", "1401-1800", "1801-2200", "2201-2600", "2601+"])
    df["sire_cond_key"] = df[COL_SIRE].astype(str) + "|" + df[COL_SURF].astype(str) + "|" + dist_bucket.astype(str)
    cond_counts = df.groupby("sire_cond_key").size().rename("sire_cond_count_fullperiod")
    df = df.merge(cond_counts, on="sire_cond_key", how="left")
    df["is_low_sample_sire_cond"] = df["sire_cond_count_fullperiod"] < LOW_SAMPLE_SIRE_COND_THRESHOLD
    return df


def build_v6_2023_scores() -> pd.DataFrame:
    """2023年(valid split)についてv6 raw scoreからPL単勝確率を計算する。
    2024・2025年は一切スコアリングしない(include_valid=Trueだがyear==2023のみ残す)。"""
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
                              v6_p_win=float(p[i]), v6_raw_score=float(g.loc[i, "_score"])))
    return pd.DataFrame(rows)


def load_market_2023() -> pd.DataFrame:
    p = Path(r"E:\競馬過去走データ\raw_data\kekka_2010_2025_fix_raceid_v2__keyed.csv")
    df = pd.read_csv(p, encoding="utf-8-sig", usecols=["race_id", "umaban", "単勝オッズ"], low_memory=False)
    df["race_id"] = df["race_id"].astype(str)
    df = df[df["race_id"].str.startswith("2023")].copy()
    df = df.dropna(subset=["単勝オッズ"])
    df["market_p_win_raw"] = 1.0 / df["単勝オッズ"]
    df["market_p_win"] = df["market_p_win_raw"] / df.groupby("race_id")["market_p_win_raw"].transform("sum")
    return df.rename(columns={"umaban": "ban", "race_id": "rid"})[["rid", "ban", "market_p_win"]]


def summarize_region(df: pd.DataFrame, mask: pd.Series, name: str) -> dict:
    sub = df[mask]
    by_year = sub.groupby("year").size().to_dict()
    fuku_rate = float(sub["fukusho_flag"].mean()) if len(sub) else float("nan")
    valid_sub = sub[sub["split"] == "valid"]  # 2023年development限定
    valid_fuku_rate = float(valid_sub["fukusho_flag"].mean()) if len(valid_sub) else float("nan")

    result = dict(
        region=name, n_total=int(mask.sum()), by_year=by_year,
        fukusho_rate_all_years=round(fuku_rate, 4) if fuku_rate == fuku_rate else None,
        n_2023_valid=len(valid_sub),
        fukusho_rate_2023_valid=round(valid_fuku_rate, 4) if valid_fuku_rate == valid_fuku_rate else None,
    )
    win_sub = valid_sub[valid_sub["v6_p_win"].notna()]
    if len(win_sub) >= 20:
        y = (win_sub["着順"] == 1).astype(float)
        brier_v6 = float(((win_sub["v6_p_win"] - y) ** 2).mean())
        result["n_2023_valid_with_v6_score"] = len(win_sub)
        result["v6_win_brier_2023_valid"] = round(brier_v6, 5)
        result["v6_mean_pred_p_win_2023_valid"] = round(float(win_sub["v6_p_win"].mean()), 4)
        result["realized_win_rate_2023_valid"] = round(float(y.mean()), 4)
        mkt_sub = win_sub[win_sub["market_p_win"].notna()]
        if len(mkt_sub) >= 20:
            corr = float(np.corrcoef(mkt_sub["v6_p_win"], mkt_sub["market_p_win"])[0, 1])
            brier_mkt = float(((mkt_sub["market_p_win"] - (mkt_sub["着順"] == 1).astype(float)) ** 2).mean())
            result["n_2023_valid_with_market"] = len(mkt_sub)
            result["corr_v6_vs_market_p_win_2023_valid"] = round(corr, 4)
            result["market_win_brier_2023_valid"] = round(brier_mkt, 5)
    else:
        result["n_2023_valid_with_v6_score"] = len(win_sub)
        result["note"] = "2023年development内サンプル不足(<20)のためBrier/相関は算出せず"
    return result


def main():
    print("[sparse_region_audit] master_v2読み込み中...")
    df = load_master()
    print(f"  行数={len(df):,}")
    df = add_career_start_count(df)
    df = add_first_time_distance(df)
    df = add_first_time_course(df)
    df = add_class_up_and_stable_transfer(df)
    df = add_low_freq_jockey_trainer(df)
    df = add_low_sample_sire_condition(df)

    print("[sparse_region_audit] 2023年v6 raw score計算中(include_valid, year==2023のみ保持)...")
    v6_scores = build_v6_2023_scores()
    market = load_market_2023()
    df["ban_int"] = pd.to_numeric(df[COL_BAN], errors="coerce")
    df = df.merge(v6_scores, left_on=[COL_RID, "ban_int"], right_on=["rid", "ban"], how="left")
    df = df.merge(market, left_on=[COL_RID, "ban_int"], right_on=["rid", "ban"], how="left",
                  suffixes=("", "_mkt"))

    regions = {
        "career_0_2": (df["career_start_idx"] <= 2),
        "career_3_5": (df["career_start_idx"].between(3, 5)),
        "first_time_distance": df["is_first_time_distance"],
        "first_time_course": df["is_first_time_course"],
        "class_up_first_race": df["is_class_up"].fillna(False),
        "stable_transfer_first_race": df["is_stable_transfer"].fillna(False),
        "low_freq_jockey_trainer_pair": df["is_low_freq_jt_pair"],
        "low_sample_sire_condition": df["is_low_sample_sire_cond"],
    }

    results = {}
    for name, mask in regions.items():
        r = summarize_region(df, mask, name)
        results[name] = r
        print(f"\n[{name}] n_total={r['n_total']:,}  "
              f"fukusho_rate(全期間)={r['fukusho_rate_all_years']}  "
              f"n_2023valid={r['n_2023_valid']:,}  "
              f"fukusho_rate(2023valid)={r['fukusho_rate_2023_valid']}")
        print(f"  年別件数: {r['by_year']}")

    # 全体(参考)
    overall_fuku = float(df["fukusho_flag"].mean())
    print(f"\n[参考: 全体] n={len(df):,}  fukusho_rate(全期間)={overall_fuku:.4f}")

    import json
    out_dir = Path(__file__).parent / "out"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "sparse_region_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[saved] {out_dir / 'sparse_region_summary.json'}")

    df.to_parquet(out_dir / "sparse_region_flags.parquet")
    print(f"[saved] {out_dir / 'sparse_region_flags.parquet'}")


if __name__ == "__main__":
    main()
