# -*- coding: utf-8 -*-
"""
build_evaluation_table.py
===========================
EXP12 Stage1。O0-O3比較用のレース単位評価テーブルを2023年developmentのみで
構築する。2024・2025年は使わない。

対象馬の選定: EXP10/EXP11と同じ規律で「raw v6 scoreが最大の馬」を対象馬
(◎)とする(主目的はwin確率head、後述)。ただし本実験は「◎自身の勝率予測」
というより「対象馬のwin確率をO2→O3でどれだけ改善できるか」に主眼があるため、
評価は◎に限定せず**全出走馬**を対象とする(race-level multinomial損失を
使うため、レース内の全馬が必要)。

含む特徴:
  O0: v6 raw win probability(OOF、score_test経由) + 市場確率
  O2: 既存ELO(elo_T1M1_horse) + Glicko(g2_mu) + EXP02(dyn_skill_mu)
  O3: opponent_graph.pyのO3b特徴10種 + external_history_gap
  confound統制用: career starts(kako5_race_count) + 休養日数(間隔) +
    年齢 + 頭数 + クラス(序数) + 人気(市場確率順位) + market_p_win +
    v6_p_win + entropy(レース単位) + missing_history_rate(NaN率の代理) +
    unique_opponent_count(O3と重複するがconfoundとしても明記) +
    external_history_gap
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"E:\PyCaLiAI")
MASTER_CSV = BASE / "data/master_v2_20130105-20251228.csv"
OUT_DIR = Path(__file__).parent / "out"

COL_RID = "レースID(新/馬番無)"
COL_BAN = "馬番"
COL_HID = "血統登録番号"

CR = {"新馬": 0, "未勝利": 1, "500万": 2, "1勝": 2, "1000万": 3, "2勝": 3,
      "1600万": 4, "3勝": 4, "ｵｰﾌﾟﾝ": 5, "OP(L)": 6, "Ｇ３": 7, "重賞": 7,
      "ＪＧ３": 7, "Ｇ２": 8, "ＪＧ２": 8, "Ｇ１": 9, "ＪＧ１": 9}


def load_v6_and_market_2023() -> pd.DataFrame:
    sys.path.insert(0, str(BASE))
    import backtest_pl_ev as be
    import pl_probs as PL
    be.MODEL_PKL = BASE / "models/unified_rank_v6.pkl"
    te = be.score_test(include_valid=True)
    te = te[te["year"] == 2023].copy()
    rows = []
    for rid, g in te.groupby(be.COL_RID, sort=False):
        g = g.sort_values(be.COL_BAN).reset_index(drop=True)
        scores = g["_score"].values.astype(float)
        w = PL.pl_weights(scores)
        p = PL.all_tansho(w)
        ent = float(-(p * np.log(np.clip(p, 1e-12, 1))).sum())
        ss = np.sort(scores)[::-1]
        gap_12 = float(ss[0] - ss[1]) if len(ss) >= 2 else np.nan
        n = len(g)
        for i in range(n):
            rows.append(dict(rid=str(rid), ban=int(g.loc[i, be.COL_BAN]),
                              v6_p_win=float(p[i]), v6_raw_score=float(scores[i]),
                              n_field=n, entropy=ent, gap_12=gap_12, max_prob=float(p.max())))
    v6 = pd.DataFrame(rows)

    market_path = Path(r"E:\競馬過去走データ\raw_data\kekka_2010_2025_fix_raceid_v2__keyed.csv")
    m = pd.read_csv(market_path, encoding="utf-8-sig", usecols=["race_id", "umaban", "単勝オッズ"],
                     low_memory=False)
    m["race_id"] = m["race_id"].astype(str)
    m = m[m["race_id"].str.startswith("2023")].dropna(subset=["単勝オッズ"])
    m["market_p_win_raw"] = 1.0 / m["単勝オッズ"]
    m["market_p_win"] = m["market_p_win_raw"] / m.groupby("race_id")["market_p_win_raw"].transform("sum")
    m["popularity_rank"] = m.groupby("race_id")["単勝オッズ"].rank(method="first", ascending=True)
    m = m.rename(columns={"umaban": "ban", "race_id": "rid"})[["rid", "ban", "market_p_win", "popularity_rank"]]
    return v6.merge(m, on=["rid", "ban"], how="left")


def load_master_confounds_2023() -> pd.DataFrame:
    cols = [COL_RID, COL_BAN, COL_HID, "日付", "着順", "年齢", "出走頭数", "クラス名",
            "間隔", "kako5_race_count"]
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", usecols=cols,
                      dtype={COL_HID: str, COL_RID: str}, low_memory=False)
    df["着順"] = pd.to_numeric(df["着順"], errors="coerce")
    df = df.dropna(subset=["着順"])
    df["year"] = df["日付"].astype(str).str[:4].astype(int)
    df = df[df["year"] == 2023].copy()
    df["class_ord"] = df["クラス名"].map(CR)
    df["missing_history_rate"] = df[["間隔", "kako5_race_count"]].isna().mean(axis=1)
    return df.rename(columns={COL_RID: "rid", COL_BAN: "ban"})


def load_o2_features() -> pd.DataFrame:
    elo = pd.read_parquet(BASE / "data/elo_feats.parquet", columns=["rid16", "ban", "elo_T1M1_horse"])
    glicko = pd.read_parquet(BASE / "data/glicko_feats.parquet", columns=["rid16", "ban", "g2_mu", "g2_rd"])
    exp02 = pd.read_parquet(BASE / "data/_research/mcond/exp02_features.parquet",
                             columns=["rid16", "ban", "dyn_skill_mu", "dyn_skill_conservative"])
    o2 = elo.merge(glicko, on=["rid16", "ban"], how="outer").merge(exp02, on=["rid16", "ban"], how="outer")
    return o2.rename(columns={"rid16": "rid"})


def main():
    print("[build_evaluation_table] v6+市場(2023)...")
    v6mkt = load_v6_and_market_2023()
    print(f"  {len(v6mkt):,}行")

    print("[build_evaluation_table] confound特徴(2023)...")
    conf = load_master_confounds_2023()
    print(f"  {len(conf):,}行")

    print("[build_evaluation_table] O2既存特徴(ELO/Glicko/EXP02)...")
    o2 = load_o2_features()

    print("[build_evaluation_table] O3(opponent_graph, o3b)...")
    o3 = pd.read_parquet(OUT_DIR / "o3_features_2023_o3b.parquet")
    o3 = o3.rename(columns={"rid16": "rid"})
    o3["ban"] = o3["ban"].astype(int)

    table = conf.merge(v6mkt, on=["rid", "ban"], how="inner")
    table = table.merge(o2, on=["rid", "ban"], how="left")
    table = table.merge(o3.drop(columns=["hid"]), on=["rid", "ban"], how="left")

    table["win"] = (table["着順"] == 1).astype(int)
    table["meeting_day"] = table["rid"].str[:10]

    print(f"\n[結果] 最終評価テーブル: {len(table):,}行、{table['rid'].nunique():,}レース")
    print(f"  欠損率: v6_p_win={table['v6_p_win'].isna().mean():.3f}  "
          f"market_p_win={table['market_p_win'].isna().mean():.3f}  "
          f"elo={table['elo_T1M1_horse'].isna().mean():.3f}  "
          f"glicko={table['g2_mu'].isna().mean():.3f}  "
          f"exp02={table['dyn_skill_mu'].isna().mean():.3f}  "
          f"O3(opponent_current_strength_mean)={table['opponent_current_strength_mean'].isna().mean():.3f}")

    table.to_parquet(OUT_DIR / "evaluation_table_2023.parquet")
    print(f"[saved] {OUT_DIR / 'evaluation_table_2023.parquet'}")


if __name__ == "__main__":
    main()
