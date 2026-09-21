# -*- coding: utf-8 -*-
"""
o1_opponent_quality_check.py
==============================
EXP12 Stage0監査。「単純なO1(過去対戦相手平均)がO0(v6+市場)へ追加情報を
持つか」を、2023年developmentのみの実データで検定する(O4以降=PageRank/
embedding/GNNへ進む合理性の判断材料)。2024・2025年は使わない。

設計:
  1. v6を master_v2 全体(train+valid+test)に対してpredict()し、各行の
     raw scoreを「その時点でのAI推定能力」の代理として使う
     (train期間はin-sample、これはEXP12の対戦相手強さ代理としての用途に
     限り許容する。当該行自身の予測精度検証には使わない)。
  2. 各行について、前走レースID(新/馬番無)で自己結合し、前走の同一レース
     内の他馬(対戦相手)のv6 raw scoreの平均を「opponent_avg_quality_prev」
     として計算する(前走時点の対戦相手=時点安全、今回のレースの結果は
     一切使わない)。
  3. 2023年developmentのみで、opponent_avg_quality_prevが、今回レースの
     v6自身のraw score・市場確率を統制した後も複勝結果と相関を持つかを
     簡易な偏相関/ロジスティック回帰係数で確認する。
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
COL_KETTO = "血統登録番号"
COL_PREV_RID = "前走レースID(新/馬番無)"
COL_FINISH = "着順"


def score_all_master_v6() -> pd.DataFrame:
    """v6モデルでmaster_v2全体をpredict()する(train期間はin-sample、
    対戦相手強さ代理としての用途に限定)。"""
    import joblib
    bundle = joblib.load(BASE / "models/unified_rank_v6.pkl")
    model, feats, encs = bundle["model"], bundle["feature_cols"], bundle["encoders"]
    rr_mode = bundle.get("race_relative_mode")
    ca_mode = bundle.get("course_affinity_mode")
    gf_mode = bundle.get("grade_feats_mode")

    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    df[COL_FINISH] = pd.to_numeric(df[COL_FINISH], errors="coerce")
    df = df.dropna(subset=[COL_FINISH, COL_RID, "split"]).copy()

    if rr_mode:
        from race_relative_feats import add_race_relative_feats
        df = add_race_relative_feats(df, mode=rr_mode)
    if ca_mode:
        from course_affinity_feats import add_course_affinity_feats
        df = add_course_affinity_feats(df, mode=ca_mode)
    if gf_mode:
        from grade_feats import add_grade_feats
        df = add_grade_feats(df, mode=gf_mode)

    sys.path.insert(0, str(BASE))
    from backtest_pl_ev import apply_encoders
    df = apply_encoders(df, encs)
    X = df[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    df["v6_score_any"] = model.predict(X)
    df["year"] = df[COL_RID].astype(str).str[:4].astype(int)
    df[COL_RID] = df[COL_RID].astype(str)
    df[COL_PREV_RID] = df[COL_PREV_RID].astype(str)
    return df[[COL_RID, COL_BAN, COL_KETTO, COL_PREV_RID, COL_FINISH,
               "fukusho_flag", "split", "year", "v6_score_any"]]


def add_opponent_quality(df: pd.DataFrame) -> pd.DataFrame:
    """前走の同一レース内・他馬のv6_score_any平均を付与する。"""
    race_mean = df.groupby(COL_RID)["v6_score_any"].transform("mean")
    race_n = df.groupby(COL_RID)["v6_score_any"].transform("count")
    # 自分を除いた平均 = (レース平均*n - 自分のスコア) / (n-1)
    df = df.copy()
    df["_race_mean"] = race_mean
    df["_race_n"] = race_n
    df["opponent_avg_quality_self_race"] = np.where(
        df["_race_n"] > 1,
        (df["_race_mean"] * df["_race_n"] - df["v6_score_any"]) / (df["_race_n"] - 1),
        np.nan,
    )
    lookup = df[[COL_RID, COL_KETTO, "opponent_avg_quality_self_race"]].rename(
        columns={COL_RID: COL_PREV_RID, "opponent_avg_quality_self_race": "opponent_avg_quality_prev"})
    lookup = lookup.drop_duplicates(subset=[COL_PREV_RID, COL_KETTO])
    df = df.merge(lookup, on=[COL_PREV_RID, COL_KETTO], how="left")
    return df


def load_v6_market_2023() -> pd.DataFrame:
    """2023年のv6 OOFスコア(既存score_test経由、valid split)とPL勝率、
    市場確率を取得する。"""
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
            rows.append(dict(rid=str(rid), ban=int(g.loc[i, be.COL_BAN]), v6_p_win_oof=float(p[i])))
    v6 = pd.DataFrame(rows)

    market_path = Path(r"E:\競馬過去走データ\raw_data\kekka_2010_2025_fix_raceid_v2__keyed.csv")
    m = pd.read_csv(market_path, encoding="utf-8-sig", usecols=["race_id", "umaban", "単勝オッズ"],
                     low_memory=False)
    m["race_id"] = m["race_id"].astype(str)
    m = m[m["race_id"].str.startswith("2023")].dropna(subset=["単勝オッズ"])
    m["market_p_win_raw"] = 1.0 / m["単勝オッズ"]
    m["market_p_win"] = m["market_p_win_raw"] / m.groupby("race_id")["market_p_win_raw"].transform("sum")
    m = m.rename(columns={"umaban": "ban", "race_id": "rid"})[["rid", "ban", "market_p_win"]]
    return v6.merge(m, on=["rid", "ban"], how="left")


def main():
    print("[o1_opponent_quality_check] v6でmaster_v2全体をpredict()中...")
    scored = score_all_master_v6()
    print(f"  行数={len(scored):,}")

    print("[o1_opponent_quality_check] 前走対戦相手平均を計算中...")
    scored = add_opponent_quality(scored)
    n_with_opp = scored["opponent_avg_quality_prev"].notna().sum()
    print(f"  opponent_avg_quality_prev利用可能行数={n_with_opp:,} / {len(scored):,}")

    print("[o1_opponent_quality_check] 2023年v6 OOF・市場確率と結合中...")
    v6mkt = load_v6_market_2023()
    df2023 = scored[scored["year"] == 2023].copy()
    df2023["ban_int"] = pd.to_numeric(df2023[COL_BAN], errors="coerce")
    df2023 = df2023.merge(v6mkt, left_on=[COL_RID, "ban_int"], right_on=["rid", "ban"], how="left")

    sub = df2023.dropna(subset=["opponent_avg_quality_prev", "v6_p_win_oof", "market_p_win"]).copy()
    print(f"\n[2023年development] 解析対象行数={len(sub):,}")

    y = sub["fukusho_flag"].astype(float).values
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    def fit_and_report(cols, label):
        X = sub[cols].values
        Xs = StandardScaler().fit_transform(X)
        clf = LogisticRegression(max_iter=1000).fit(Xs, y)
        print(f"  [{label}] coef={dict(zip(cols, np.round(clf.coef_[0], 4)))}")
        return clf

    print("\n[単純相関]")
    corr_opp_y = np.corrcoef(sub["opponent_avg_quality_prev"], y)[0, 1]
    corr_opp_v6 = np.corrcoef(sub["opponent_avg_quality_prev"], sub["v6_p_win_oof"])[0, 1]
    print(f"  corr(opponent_avg_quality_prev, fukusho_flag) = {corr_opp_y:.4f}")
    print(f"  corr(opponent_avg_quality_prev, v6_p_win_oof) = {corr_opp_v6:.4f}")

    print("\n[full-control: v6 OOF + 市場確率のみ vs +opponent_avg_quality_prev]")
    fit_and_report(["v6_p_win_oof", "market_p_win"], "O0のみ(v6+市場)")
    clf_full = fit_and_report(["v6_p_win_oof", "market_p_win", "opponent_avg_quality_prev"],
                                "O0+O1(opponent_avg_quality_prev追加)")

    from sklearn.metrics import log_loss
    X0 = StandardScaler().fit_transform(sub[["v6_p_win_oof", "market_p_win"]].values)
    X1 = StandardScaler().fit_transform(sub[["v6_p_win_oof", "market_p_win", "opponent_avg_quality_prev"]].values)
    ll0 = log_loss(y, LogisticRegression(max_iter=1000).fit(X0, y).predict_proba(X0)[:, 1])
    ll1 = log_loss(y, LogisticRegression(max_iter=1000).fit(X1, y).predict_proba(X1)[:, 1])
    print(f"\n  in-sample logloss O0のみ={ll0:.5f}  O0+O1={ll1:.5f}  差={ll1-ll0:+.5f}")
    print("  (in-sample全数fit、CV/OOSではないため参考値。正式な検定はStage1で実施)")

    sub.to_parquet(OUT_DIR / "o1_check_2023.parquet") if OUT_DIR.exists() else None


if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)
    main()
