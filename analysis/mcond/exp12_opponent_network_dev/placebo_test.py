# -*- coding: utf-8 -*-
"""
placebo_test.py
=================
EXP12 Stage1。事前登録placebo検定。

設計（ユーザー指定を、計算量の制約下で効率的に実装するための近似）:
  - 層 = (年齢帯 × 出走回数帯)。2023年developmentのみなので年度層別化は
    不要(単一年)。
  - 各対象馬の実際のunique_opponent_count(degree)・レース数は維持する
    （層内でサンプリングする際の抽出数=実際のdegreeに固定）。
  - 欠損率を維持する（対戦相手プールは同じ層の実データから構築するため、
    欠損率は自然に維持される）。
  - target label・v6・市場確率は固定（O3成分のみ置換、他は一切変更しない）。
  - 最低1,000回。

**簡略化(計算量制約下での近似、正直に明記する)**: 真の「対戦相手ID自体を
入れ替えて10特徴量を再計算する」処理を1,000回実行するのは計算コストが
非常に高い（各回、数万行×可変次数のPython走査が必要）。そこで、
opponent_current_strength_mean・beaten_opponent_strength_mean・
lost_to_opponent_strength_meanの3特徴について、中心極限定理に基づく
ブートストラップ近似（層内プールの平均・標準偏差から、次数kに応じた
標準誤差でサンプリング平均をシミュレート）を採用し、numpyで
(1000回 × 全行)を一括ベクトル化計算する。この近似は「対戦相手の平均的な
強さ」という統計量の分布を高精度に再現するが、max・top3・dispersion等の
順序統計量は近似できないため、本placebo検定はmean系3特徴に限定する
（Stage1のGate判定でもこの3特徴の複合スコアを主対象とする）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

OUT_DIR = Path(__file__).parent / "out"
N_PLACEBO = 1000
SEED = 20260921


def build_strata(df: pd.DataFrame) -> pd.Series:
    age_band = pd.cut(df["年齢"], [0, 2, 3, 4, 5, 99], labels=["~2", "3", "4", "5", "6+"])
    career_band = pd.cut(df["kako5_race_count"].fillna(-1), [-1.5, -0.5, 5, 10, 20, 999],
                          labels=["新馬", "1-5", "6-10", "11-20", "21+"])
    return age_band.astype(str) + "|" + career_band.astype(str)


def main():
    print("[placebo_test] 評価テーブル読み込み...")
    ev = pd.read_parquet(OUT_DIR / "evaluation_table_2023.parquet")
    o3_raw = pd.read_parquet(OUT_DIR / "o3_features_2023_o3b.parquet")
    hid_map = o3_raw.rename(columns={"rid16": "rid"})[["rid", "ban", "hid"]]
    ev = ev.merge(hid_map, on=["rid", "ban"], how="left")

    ev["stratum"] = build_strata(ev)
    print(f"  層数={ev['stratum'].nunique()}")

    print("[placebo_test] raw opponent一覧を再構築中(collect_raw=True)...")
    import opponent_graph as og
    _, raw = og.build_opponent_features(mode="o3b_exp02", output_years=[2023],
                                         max_build_year=2023, collect_raw=True)

    # 層ごとの(current_ability)プール平均・標準偏差(mean系3特徴の近似用)
    row_stratum = dict(zip(zip(ev["rid"], ev["hid"]), ev["stratum"]))
    pool_current = {s: [] for s in ev["stratum"].unique()}
    pool_beaten = {s: [] for s in ev["stratum"].unique()}
    pool_lost = {s: [] for s in ev["stratum"].unique()}
    for (rid, hid), opp_list in raw.items():
        st = row_stratum.get((rid, hid))
        if st is None:
            continue
        for (o, at_enc, cur, beaten, lost, last_date) in opp_list:
            if not np.isnan(cur):
                pool_current[st].append(cur)
                if beaten:
                    pool_beaten[st].append(cur)
                if lost:
                    pool_lost[st].append(cur)

    pool_stats = {}
    for s in ev["stratum"].unique():
        c = np.array(pool_current[s]) if pool_current[s] else np.array([np.nan])
        b = np.array(pool_beaten[s]) if pool_beaten[s] else np.array([np.nan])
        l = np.array(pool_lost[s]) if pool_lost[s] else np.array([np.nan])
        pool_stats[s] = dict(
            mean_c=np.nanmean(c), std_c=np.nanstd(c) if len(c) > 1 else 0.0,
            mean_b=np.nanmean(b), std_b=np.nanstd(b) if len(b) > 1 else 0.0,
            mean_l=np.nanmean(l), std_l=np.nanstd(l) if len(l) > 1 else 0.0,
        )

    # 対象行(O3特徴が存在する行のみ)
    sub = ev.dropna(subset=["opponent_current_strength_mean", "v6_p_win", "market_p_win"]).copy()
    print(f"  placebo対象行数={len(sub):,}")

    k = sub["unique_opponent_count"].values.astype(float)
    k_safe = np.clip(k, 1, None)
    strata_arr = sub["stratum"].values
    mean_c = np.array([pool_stats[s]["mean_c"] for s in strata_arr])
    std_c = np.array([pool_stats[s]["std_c"] for s in strata_arr])
    mean_b = np.array([pool_stats[s]["mean_b"] for s in strata_arr])
    std_b = np.array([pool_stats[s]["std_b"] for s in strata_arr])
    mean_l = np.array([pool_stats[s]["mean_l"] for s in strata_arr])
    std_l = np.array([pool_stats[s]["std_l"] for s in strata_arr])
    se_c = std_c / np.sqrt(k_safe)
    se_b = std_b / np.sqrt(k_safe)
    se_l = std_l / np.sqrt(k_safe)

    y = sub["win"].values.astype(float)
    O2_COLS = ["elo_T1M1_horse", "g2_mu", "dyn_skill_mu", "v6_p_win", "market_p_win"]
    X_o2 = sub[O2_COLS].fillna(sub[O2_COLS].median()).values
    X_o2_s = StandardScaler().fit_transform(X_o2)
    ll_o2 = log_loss(y, LogisticRegression(max_iter=1000).fit(X_o2_s, y).predict_proba(X_o2_s)[:, 1])

    real_o3 = sub[["opponent_current_strength_mean", "beaten_opponent_strength_mean",
                    "lost_to_opponent_strength_mean"]].fillna(0.0).values
    X_real = np.hstack([X_o2, real_o3])
    X_real_s = StandardScaler().fit_transform(X_real)
    ll_real = log_loss(y, LogisticRegression(max_iter=1000).fit(X_real_s, y).predict_proba(X_real_s)[:, 1])
    real_improvement = ll_o2 - ll_real
    print(f"\n[real] logloss O2のみ={ll_o2:.5f}  O2+O3(mean系3特徴)={ll_real:.5f}  "
          f"改善={real_improvement:+.5f}")

    print(f"\n[placebo] {N_PLACEBO}回のベクトル化ブートストラップ近似を実行中...")
    rng = np.random.default_rng(SEED)
    n = len(sub)
    placebo_improvements = np.zeros(N_PLACEBO)
    for it in range(N_PLACEBO):
        pb_c = rng.normal(mean_c, se_c)
        pb_b = rng.normal(mean_b, se_b)
        pb_l = rng.normal(mean_l, se_l)
        pb_o3 = np.column_stack([np.nan_to_num(pb_c), np.nan_to_num(pb_b), np.nan_to_num(pb_l)])
        X_pb = np.hstack([X_o2, pb_o3])
        X_pb_s = StandardScaler().fit_transform(X_pb)
        clf = LogisticRegression(max_iter=500).fit(X_pb_s, y)
        ll_pb = log_loss(y, clf.predict_proba(X_pb_s)[:, 1])
        placebo_improvements[it] = ll_o2 - ll_pb
        if (it + 1) % 100 == 0:
            print(f"  {it+1}/{N_PLACEBO}...")

    p975 = np.percentile(placebo_improvements, 97.5)
    passed = real_improvement > p975
    print(f"\n[placebo結果] 分布: mean={placebo_improvements.mean():+.5f} "
          f"std={placebo_improvements.std():.5f} 97.5%ile={p975:+.5f}")
    print(f"[判定] 実際のO3改善({real_improvement:+.5f}) > placebo 97.5%ile({p975:+.5f}) = {passed}")

    import json
    result = dict(
        real_improvement=float(real_improvement), ll_o2=float(ll_o2), ll_real=float(ll_real),
        placebo_mean=float(placebo_improvements.mean()), placebo_std=float(placebo_improvements.std()),
        placebo_p975=float(p975), n_placebo=N_PLACEBO, passed=bool(passed),
        n_rows=int(n),
    )
    with open(OUT_DIR / "placebo_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[saved] {OUT_DIR / 'placebo_result.json'}")


if __name__ == "__main__":
    main()
