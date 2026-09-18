# -*- coding: utf-8 -*-
"""
build_features.py — EXP05 の設計行列を組み立てる
==================================================
入力:
  data/_research/mcond/base.parquet          (v6base: 時点安全な v6 スコア・市場確率・結果)
  data/_research/mcond/exp04_candidates.parquet (EXP04 の 145 候補、時点安全性は EXP04 で検証済み)
  data/serve_feature_baseline.json            (2026-07 serve 監査: 本番で取得できない列)

やること:
  1. train 期間 (2016-2021) だけで isotonic calibrator (win / top3) を自前で fit し、
     v6_pwin / v6_p3 を「calibrated_v6_probability」に変換する。
     本番 pl_calibrators_v6.pkl は valid=2023 で fit されているため 2023 年の
     評価に使うと較正器自身がその年を見ている (leak)。自前 train-only 較正で置き換える。
  2. レース内相対特徴 (score_gap_to_top, score_gap_to_second, score_percentile_in_race,
     field_score_dispersion) を v6_score から計算する。
  3. F-full = EXP04 の 145 候補そのまま。F-serve = F-full から
     serve_feature_baseline.json の serve_dead 列に由来する列を除いた部分集合。
出力:
  data/_research/mcond/exp05_design.parquet
  out/feature_lists.json
実行: python -m analysis.mcond.exp05_market_residual_dev.build_features
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
from analysis.mcond.evaluate import logit  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
D = BASE / "data/_research/mcond"
TRAIN_Y0, TRAIN_Y1 = 2016, 2021
SEL_YEAR = 2022


def load_base() -> pd.DataFrame:
    b = pd.read_parquet(D / "base.parquet")
    b["rid16"] = b["rid16"].astype(str)
    b = b[(b.year >= TRAIN_Y0) & (b.year <= 2025) & (b.n_field >= 5)
          & b.mkt_p3_pre.notna() & b.v6_p3.notna() & b.mkt_pi_pre.notna()].copy()
    b["day"] = b["rid16"].str[:8]
    return b.reset_index(drop=True)


def add_relative_v6_features(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("rid16")["v6_score"]
    mx = g.transform("max")
    disp = g.transform("std").fillna(0.0)

    def second_max(s: pd.Series) -> float:
        u = s.sort_values(ascending=False)
        return u.iloc[1] if len(u) > 1 else u.iloc[0]

    smax2 = g.transform(second_max)
    df["score_gap_to_top"] = df["v6_score"] - mx
    df["score_gap_to_second"] = df["v6_score"] - smax2
    df["score_percentile_in_race"] = df.groupby("rid16")["v6_score"].rank(pct=True)
    df["field_score_dispersion"] = disp
    return df


def add_calibrated_v6(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    train = (df.year >= TRAIN_Y0) & (df.year <= TRAIN_Y1)
    meta = {}
    for target, src, name in (("win", "v6_pwin", "calibrated_v6_probability_win"),
                              ("top3", "v6_p3", "calibrated_v6_probability")):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(df.loc[train, src].to_numpy(), df.loc[train, target].to_numpy())
        df[name] = iso.predict(df[src].to_numpy())
        meta[name] = {"fit_period": f"{TRAIN_Y0}-{TRAIN_Y1}", "source": src,
                      "n_fit": int(train.sum()), "n_unique_output": int(np.unique(iso.predict(
                          df.loc[train, src].to_numpy())).size)}
    return df, meta


def serve_dead_source_columns() -> set[str]:
    p = BASE / "data/serve_feature_baseline.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    return set(d.get("serve_dead", []))


def split_full_serve(cand_cols: list[str]) -> tuple[list[str], list[str], list[str]]:
    dead = serve_dead_source_columns()
    excluded = []
    serve_cols = []
    for c in cand_cols:
        if not c.startswith("c1__"):
            # C2 (exp01 choice) / C3 (exp02/03 skill) は EXP04 で raw 由来の再エンジニアリング特徴。
            # serve_feature_baseline.json は v6 の生 120 特徴の監査なのでここには載らない。
            # 供給元 (騎手・場所・芝ダ・距離・クラス・日付) は全て serve_dead に無いので保持。
            serve_cols.append(c)
            continue
        body = c[len("c1__"):]
        source = body.split("__")[0]
        if source in dead:
            excluded.append(c)
        else:
            serve_cols.append(c)
    return serve_cols, excluded, sorted(dead)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    df = load_base()
    df = add_relative_v6_features(df)
    df, cal_meta = add_calibrated_v6(df)

    cand = pd.read_parquet(D / "exp04_candidates.parquet")
    cand["rid16"] = cand["rid16"].astype(str)
    cand_cols = [c for c in cand.columns if c not in ("rid16", "ban", "year")]
    df = df.merge(cand[["rid16", "ban"] + cand_cols], on=["rid16", "ban"], how="inner")

    m = pd.read_csv(BASE / "data/master_v2_20130105-20251228.csv", encoding="utf-8-sig",
                    low_memory=False, usecols=["レースID(新)", "場所"])
    m["rid16"] = m["レースID(新)"].astype(str).str[:16]
    m = m.drop_duplicates("rid16")[["rid16", "場所"]].rename(columns={"場所": "venue"})
    df = df.merge(m, on="rid16", how="left")

    serve_cols, excluded_cols, dead_sources = split_full_serve(cand_cols)

    df["f_v6_win"] = logit(df["v6_pwin"])
    df["f_mkt_win"] = logit(df["mkt_pi_pre"])
    df["f_v6"] = logit(df["v6_p3"])
    df["f_mkt"] = logit(df["mkt_p3_pre"])
    df["lp_cal_top3"] = logit(df["calibrated_v6_probability"])
    df["lp_cal_win"] = logit(df["calibrated_v6_probability_win"])
    df["train"] = (df.year >= TRAIN_Y0) & (df.year <= TRAIN_Y1)
    df["sel"] = df.year == SEL_YEAR

    df.to_parquet(D / "exp05_design.parquet", index=False)

    info = {
        "n_rows": int(len(df)),
        "n_races": int(df.rid16.nunique()),
        "years": sorted(df.year.unique().tolist()),
        "train_period": f"{TRAIN_Y0}-{TRAIN_Y1}",
        "sel_year": SEL_YEAR,
        "n_candidates_full": len(cand_cols),
        "n_candidates_serve": len(serve_cols),
        "n_excluded_for_serve": len(excluded_cols),
        "excluded_for_serve": excluded_cols,
        "serve_dead_source_columns": dead_sources,
        "F_full": cand_cols,
        "F_serve": serve_cols,
        "calibrator_meta": cal_meta,
    }
    (OUT / "feature_lists.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"rows={len(df):,} races={df.rid16.nunique():,} "
          f"F-full={len(cand_cols)} F-serve={len(serve_cols)} excluded={len(excluded_cols)}")
    for k, v in cal_meta.items():
        print(f"  {k}: fit={v['fit_period']} n={v['n_fit']:,} unique_outputs={v['n_unique_output']}")


if __name__ == "__main__":
    main()
