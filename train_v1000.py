# -*- coding: utf-8 -*-
"""
train_v1000.py — unified_rank_v_1000（v6 とは別系統の新規モデル）学習・評価
================================================================================
方針:
  - 学習ハーネス（V6 ハイパラ / market 重み / LambdaRank / 分割 / metrics）は
    exp_feateng_v7.py から import して土俵を完全に揃える。
  - baseline = master 基礎特徴のみ（= v6 再現。REPRO と照合）。
  - v_1000   = baseline + data/feats_1000.parquet の 1039 生成特徴。
  - 同条件で test(2024-25) の hon_top3 / auc_win / pairwise_acc 等を比較。

★v6 本番（unified_rank_v6.pkl）には一切触らない。新規 pkl として保存。
  （注: これを実運用 serve するには predict 時にも 1000 特徴を再生成する配線が要る。
        本スクリプトは「1000特徴の新モデルが天井を破るか」の決着が目的。）

出力:
  models/unified_rank_v_1000.pkl
  reports/unified_rank_v_1000_eval.json
実行: PYTHONUTF8=1 ./venv311/Scripts/python.exe train_v1000.py
"""
from __future__ import annotations
import io, json, sys, time, warnings
from itertools import groupby
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# v6 再現ハーネスを流用（土俵を揃える）
from exp_feateng_v7 import (
    V6, V6_ALPHA, V6_BEST_ITER, REPRO, CAT_COLS, LEAK_COLS,
    metrics, load_winner_pay, fit_encoders, apply_encoders,
)

BASE = Path(__file__).parent
MASTER_CSV = BASE / "data/master_v2_20130105-20251228.csv"
FEATS_PARQUET = BASE / "data/feats_1000.parquet"
OUT_MODEL = BASE / "models/unified_rank_v_1000.pkl"
OUT_EVAL = BASE / "reports/unified_rank_v_1000_eval.json"
SEED = 42
COL_RID = "レースID(新/馬番無)"; COL_JYUN = "着順"; COL_BAN = "馬番"


def log(m): print(m, flush=True)


def _rid16(s):
    return s.astype(str).str.replace(r"\D", "", regex=True).str[:16]


def load():
    log(f"[load] {MASTER_CSV.name}")
    t0 = time.time()
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df = df.dropna(subset=[COL_JYUN, COL_RID, "split"]).copy().reset_index(drop=True)
    df["label"] = np.clip(6 - df[COL_JYUN].astype(int), 0, 5).astype(int)
    df["rid_s"] = df[COL_RID].astype(str).str[:16]
    df["rid16"] = _rid16(df[COL_RID])
    df["ban"] = pd.to_numeric(df[COL_BAN], errors="coerce").fillna(0).astype(int)
    log(f"  rows={len(df):,}  ({time.time()-t0:.0f}s)")

    # master 基礎特徴（v6 baseline）= 非リーク数値/カテゴリ列。生成特徴の前に確定させる。
    EXTRA = {"label", "rid_s", "rid16", "ban"}
    base_feats = [c for c in df.columns if c not in LEAK_COLS and c not in EXTRA]

    # カテゴリは train で label-encode
    encs = fit_encoders(df[df["split"] == "train"])
    df = apply_encoders(df, encs)

    # 1000 生成特徴をマージ
    log(f"[merge] {FEATS_PARQUET.name}")
    g = pd.read_parquet(FEATS_PARQUET)
    g["rid16"] = g["rid16"].astype(str)
    g["ban"] = pd.to_numeric(g["ban"], errors="coerce").fillna(0).astype(int)
    gen_cols = [c for c in g.columns if c not in ("rid16", "ban")]
    df = df.merge(g, on=["rid16", "ban"], how="left")
    log(f"  gen_cols={len(gen_cols)}  cov(test非欠損,例)="
        f"{df.loc[df.split=='test', gen_cols[0]].notna().mean()*100:.1f}%")

    # 全特徴列を float32 化（OOM 回避 & make_ds を高速化）
    all_feats = base_feats + gen_cols
    for c in all_feats:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

    # スモーク用: 各 split を層化サンプリング（env FEATS1000_FRAC）
    import os
    frac = os.environ.get("FEATS1000_FRAC")
    if frac:
        frac = float(frac)
        df = (df.groupby("split", group_keys=False)
                .apply(lambda x: x.sample(frac=frac, random_state=SEED))
                .reset_index(drop=True))
        log(f"  [SMOKE] frac={frac} → rows={len(df):,}")
    return df, base_feats, gen_cols, encs


def make_ds(d, feats, win_pay, alpha):
    d = d.sort_values(COL_RID, kind="mergesort").reset_index(drop=True)
    X = d[feats].fillna(-9999.0).to_numpy(dtype=np.float32)
    y = d["label"].to_numpy(dtype=np.int32)
    grp = np.array([len(list(gr)) for _, gr in groupby(d[COL_RID])])
    if alpha > 0:
        wt = d["rid_s"].map(win_pay).fillna(100.0).to_numpy(dtype=np.float64)
        w = (1.0 + alpha * np.log1p(wt / 100.0)).astype(np.float32)
    else:
        w = np.ones(len(d), np.float32)
    return lgb.Dataset(X, label=y, group=grp, weight=w, free_raw_data=True)


def train(tr, vl, feats, win_pay, rounds):
    params = dict(objective="lambdarank", lambdarank_truncation_level=5, metric="ndcg",
                  eval_at=[5], **V6, bagging_freq=5, verbose=-1, n_jobs=-1, seed=SEED,
                  deterministic=True, force_col_wise=True, feature_pre_filter=False)
    ds_tr = make_ds(tr, feats, win_pay, V6_ALPHA)
    ds_vl = make_ds(vl, feats, win_pay, 0.0)
    return lgb.train(params, ds_tr, num_boost_round=rounds, valid_sets=[ds_vl],
                     callbacks=[lgb.early_stopping(100, verbose=False)])


def predict_into(df, model, feats, col):
    X = df[feats].fillna(-9999.0).to_numpy(dtype=np.float32)
    df[col] = model.predict(X)


def main():
    T0 = time.time()
    OUT_MODEL.parent.mkdir(exist_ok=True)
    OUT_EVAL.parent.mkdir(exist_ok=True)
    df, base_feats, gen_cols, encs = load()
    win_pay = load_winner_pay()
    tr = df[df["split"] == "train"]; vl = df[df["split"] == "valid"]
    log(f"  base_feats={len(base_feats)}  gen_cols={len(gen_cols)}  "
        f"v1000_feats={len(base_feats)+len(gen_cols)}")

    arms = {}
    # ---- baseline (= v6 再現) ----
    log("\n[arm] baseline (master基礎のみ)")
    t0 = time.time()
    m_base = train(tr, vl, base_feats, win_pay, rounds=int(V6_BEST_ITER * 1.1))
    predict_into(df, m_base, base_feats, "_s_base")
    mb = metrics(df, "_s_base")
    arms["baseline"] = {"n_feats": len(base_feats), "best_iter": m_base.best_iteration, "metrics": mb}
    log(f"  {time.time()-t0:.0f}s it={m_base.best_iteration} "
        f"◎top3={mb['hon_top3']*100:.2f}% AUC={mb['auc_win']} pair={mb['pairwise_acc']}")

    # ---- v_1000 ----
    v1000_feats = base_feats + gen_cols
    log("\n[arm] v_1000 (base + 1000生成特徴)")
    t0 = time.time()
    # 特徴が10倍なので boost round を伸ばす（早期停止で頭打ちを検出）
    m_v1000 = train(tr, vl, v1000_feats, win_pay, rounds=2000)
    predict_into(df, m_v1000, v1000_feats, "_s_v1000")
    mv = metrics(df, "_s_v1000")
    arms["v_1000"] = {"n_feats": len(v1000_feats), "best_iter": m_v1000.best_iteration, "metrics": mv}
    log(f"  {time.time()-t0:.0f}s it={m_v1000.best_iteration} "
        f"◎top3={mv['hon_top3']*100:.2f}% AUC={mv['auc_win']} pair={mv['pairwise_acc']}")

    # ---- gain（生成特徴ファミリ別寄与）----
    gain = pd.Series(m_v1000.feature_importance("gain"), index=v1000_feats)
    fam = {}
    for pre, name in [("f1_", "F1_within_race"), ("f2_", "F2_pair_prod"),
                       ("f3_", "F3_pair_ratio"), ("f4_", "F4_pastform"),
                       ("f5_", "F5_context_change"), ("f6_", "F6_career_asof")]:
        sub = gain[[c for c in gain.index if c.startswith(pre)]]
        fam[name] = {"sum_gain": round(float(sub.sum()), 1), "n": int(len(sub))}
    base_gain = round(float(gain[base_feats].sum()), 1)
    total_gain = round(float(gain.sum()), 1)
    top20 = gain.sort_values(ascending=False).head(20)
    fam["_base_master"] = {"sum_gain": base_gain, "n": len(base_feats)}
    gen_share = round(100.0 * (total_gain - base_gain) / total_gain, 1) if total_gain else 0.0

    # ---- 判定 ----
    A = arms["baseline"]["metrics"]
    repro_ok = (abs(A["hon_top3"] - REPRO["hon_top3"]) < 0.004 and
                abs(A["auc_win"] - REPRO["auc_win"]) < 0.004 and
                abs(A["pairwise_acc"] - REPRO["pairwise_acc"]) < 0.004)
    delta = {kk: round(mv[kk] - mb[kk], 5)
             for kk in ["hon_top3", "hon_top1", "winner_in_top5", "auc_win", "pairwise_acc", "ndcg5"]}
    # 事前固定ゲート（exp_feateng_v7 と同じ厳しさ）
    gate_pass = bool(delta["auc_win"] >= 0.008 and delta["pairwise_acc"] >= 0.005)

    result = {
        "protocol": "v6再現ハーネス(exp_feateng_v7)流用 / base=master / v_1000=base+1039生成 / train≤2022→valid2023→test2024-25",
        "n_generated_features": len(gen_cols),
        "baseline_reproduced": bool(repro_ok),
        "repro_target": REPRO,
        "arms": arms,
        "delta_v1000_minus_baseline": delta,
        "gate_rule": "ΔAUC≥+0.008 かつ Δpairwise≥+0.005 で本物の増分",
        "gate_pass": gate_pass,
        "gen_gain_share_pct": gen_share,
        "family_gain": fam,
        "top20_features": {k: round(float(v), 1) for k, v in top20.items()},
        "elapsed_sec": round(time.time() - T0, 1),
    }
    with open(OUT_EVAL, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    log(f"\n[saved] {OUT_EVAL}")

    joblib.dump({
        "model": m_v1000,
        "feature_cols": v1000_feats,
        "base_feats": base_feats,
        "gen_cols": gen_cols,
        "encoders": encs,
        "cat_cols": CAT_COLS,
        "feats_parquet": FEATS_PARQUET.name,
        "build_script": "build_feats_1000.py",
        "split": {"train_max": 2022, "valid": 2023, "test": [2024, 2025]},
        "seed": SEED,
        "metrics_test": mv,
        "description": "unified_rank_v_1000: v6 baseline + 1039 leak-safe生成特徴 (F1-F6). v6本番とは別系統の研究モデル。serve には predict時 build_feats_1000 の配線が必要。",
    }, OUT_MODEL)
    log(f"[saved] {OUT_MODEL}")

    # ---- サマリ ----
    log("\n" + "=" * 64)
    log(f"baseline再現  : {'OK' if repro_ok else 'NG'}  "
        f"(◎top3={mb['hon_top3']*100:.2f}% / REPRO {REPRO['hon_top3']*100:.2f}%)")
    log(f"v_1000        : ◎top3={mv['hon_top3']*100:.2f}%  AUC={mv['auc_win']}  pair={mv['pairwise_acc']}")
    log(f"Δ(v1000-base) : ◎top3={delta['hon_top3']*100:+.2f}pt  "
        f"AUC={delta['auc_win']:+.5f}  pair={delta['pairwise_acc']:+.5f}")
    log(f"生成特徴 gain寄与: {gen_share}%   ゲート(ΔAUC≥.008&Δpair≥.005): {'通過' if gate_pass else '未達'}")
    log("=" * 64)


if __name__ == "__main__":
    main()
