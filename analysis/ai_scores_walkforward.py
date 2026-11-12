# -*- coding: utf-8 -*-
"""
ai_scores_walkforward.py — 2023 年より前の「その時点で作れた AI 順位」を作る (2026-11-13)
==========================================================================
本番 v6 は 〜2022 年で学習しているので、2022 年以前の v6 スコアは答えを見た後の値で検証に使えない。
同じ 120 特徴・同じハイパーパラメータ・同じエンコーダで、
  fold A: 〜2015 年で学習 → 2016-2018 を採点
  fold B: 〜2018 年で学習 → 2019-2022 を採点
とし、AI 順位を重ねる検証 (朝から売れなくなった馬 × AI 上位 等) を 2023-25 以外の独立期間でも行えるようにする。
ラベルは 1 着 3 / 2 着 2 / 3 着 1 (LambdaRank)。v6 本体の重み付けは再現しない (順位の粗い近似で十分)。
出力: data/_ai_scores_wf_2016_2022.parquet (rid, ban, year, _score, ai_rk, fold)
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
import backtest_pl_ev as BT  # noqa: E402

MASTER = BASE / "data" / "master_v2_20130105-20251228.csv"
OUT = BASE / "data" / "_ai_scores_wf_2016_2022.parquet"
RID, JYUN, BAN = "レースID(新/馬番無)", "着順", "馬番"
FOLDS = {"A": (2015, 2016, 2018), "B": (2018, 2019, 2022)}


def main() -> int:
    import lightgbm as lgb
    b = joblib.load(BASE / "models" / "unified_rank_v6.pkl")
    feats, encs, params, rounds = b["feature_cols"], b["encoders"], dict(b["base_lgbm_params"]), int(b["num_boost_round"])
    df = pd.read_csv(MASTER, encoding="utf-8-sig", low_memory=False)
    df[JYUN] = pd.to_numeric(df[JYUN], errors="coerce")
    df = df.dropna(subset=[JYUN, RID]).copy()
    df["year"] = df[RID].astype(str).str[:4].astype(int)
    df = df[df.year <= 2022].copy()
    enc = BT.apply_encoders(df, encs)
    X = enc[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999)
    y = df[JYUN].map({1: 3, 2: 2, 3: 1}).fillna(0).astype(int)
    rid = df[RID].astype(str)
    order = np.argsort(rid.values, kind="stable")
    X, y, rid, df = X.iloc[order], y.iloc[order], rid.iloc[order], df.iloc[order]
    outs = []
    for name, (tr_end, te_a, te_b) in FOLDS.items():
        tr = (df.year <= tr_end).values; te = ((df.year >= te_a) & (df.year <= te_b)).values
        grp = rid[tr].groupby(rid[tr], sort=False).size().values
        print(f"fold {name}: train {tr.sum():,} 行 ({len(grp):,}R) → score {te.sum():,} 行", flush=True)
        m = lgb.train(params, lgb.Dataset(X[tr].values, label=y[tr].values, group=grp), num_boost_round=rounds)
        o = pd.DataFrame({"rid": rid[te].str[:16].values, "ban": pd.to_numeric(df[BAN][te], errors="coerce").values, "year": df.year[te].values,
                          "_score": m.predict(X[te].values), "fin": df[JYUN][te].values, "fold": name})
        o["ai_rk"] = o.groupby("rid")._score.rank(ascending=False, method="first")
        top = o[o.ai_rk == 1]
        print(f"   AI 1位の 1着率 {100 * (top.fin == 1).mean():.1f}%  3着内率 {100 * (top.fin <= 3).mean():.1f}%", flush=True)
        outs.append(o)
    pd.concat(outs, ignore_index=True).to_parquet(OUT, index=False)
    print("saved", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
