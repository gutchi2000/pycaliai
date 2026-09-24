# -*- coding: utf-8 -*-
"""
build_oof.py — EXP16A Stage 1-1: rolling OOF (Q1) の構築 40 fit
================================================================
spec v0.4 (commit e7603678, 凍結) の q1_rolling_oof / OOF_STACKING_PLAN.md に従う。
各評価年 Y と各 seed について:
    train  : 年 <= Y-2
    ES     : Y-1  (ndcg@5, early_stopping_rounds=100)
    predict: Y
    特徴   : R0-clean 111 列 (EXP15 out/feature_contract.json の features.clean)
    encoder: **train (<= Y-2) だけ**で語彙を作る。Y 以降は絶対に見ない
    温度 τ : Y-1 の **正式 race set 行だけ**で条件付きロジット最尤
学習・予測しかしない。2019-2023 の結果指標は計算しない (評価は evaluate.py)。
2024/2025 は読み込み時に破棄し assert する。

出力:
  data/_research/mcond/exp16a/scores/Q1_Y{year}_s{seed}.npz   (gitignore)
  out/oof_manifest.json                                       (artifact 契約)
  out/oof_build_checks.json                                   (重複・未来行・行数)
  (削除不変性・確率和・行対応の検査は oof_tests.py が out/oof_tests.json へ書く)
実行: python -m analysis.mcond.exp16a_close_market_residual_dev.build_oof
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from .provenance import BASE, OUT, MASTER, sha256
from .stage0_dry_run import EXP15_OUT, p05_c1_check, sha_text

RESEARCH = BASE / "data" / "_research" / "mcond" / "exp16a"
SCORES = RESEARCH / "scores"
ROWS_CACHE = RESEARCH / "rows_2013_2023.parquet"
D_MIN, D_MAX = 20130101, 20231231          # 2024/2025 は読まない
YEARS = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
SEEDS = [20260923, 20260924, 20260925, 20260926, 20260927]
COL_RID, COL_DATE, COL_JYUN, COL_BAN = "レースID(新/馬番無)", "日付", "着順", "馬番"

# EXP15 で 2022 選択済みの v6_lr_half (exp15_race_as_set_dev/spec.json models.R0-clean.grid)
CFG = {"learning_rate": 0.02541579250771342, "num_leaves": 59, "max_depth": 12,
       "min_data_in_leaf": 197, "feature_fraction": 0.876098829427658,
       "bagging_fraction": 0.7031707810405968, "lambda_l1": 0.0011077902520957399,
       "lambda_l2": 7.537933313450104}
FIXED = {"objective": "lambdarank", "lambdarank_truncation_level": 5, "metric": "ndcg",
         "eval_at": [5], "bagging_freq": 5, "deterministic": True, "force_col_wise": True,
         "feature_pre_filter": False, "verbose": -1, "n_jobs": -1}
MAX_ROUNDS, ES_ROUNDS = 3000, 100


# ---------------------------------------------------------------- データ
def load_rows(contract: dict) -> pd.DataFrame:
    """master_v2 の 2013-2023 行。着順が数値の行のみ (v6/EXP15 と同じ母集団)。"""
    RESEARCH.mkdir(parents=True, exist_ok=True)
    cols = list(dict.fromkeys([COL_DATE, COL_RID, COL_BAN, COL_JYUN] + contract["features"]["clean"]))
    if ROWS_CACHE.exists():
        df = pd.read_parquet(ROWS_CACHE)
    else:
        keep = []
        for ch in pd.read_csv(MASTER, encoding="utf-8-sig", dtype=str, usecols=cols,
                              chunksize=100_000):
            d = pd.to_numeric(ch[COL_DATE], errors="coerce")
            keep.append(ch[(d >= D_MIN) & (d <= D_MAX)])
            del ch
        df = pd.concat(keep, ignore_index=True)
        df.to_parquet(ROWS_CACHE, index=False)
    df = df.copy()
    df["date"] = pd.to_numeric(df[COL_DATE], errors="coerce").astype("int64")
    assert int(df["date"].max()) <= D_MAX, "2024 以降が混入している"
    df["jyun"] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df = df.dropna(subset=["jyun", COL_RID]).copy()
    df["jyun"] = df["jyun"].astype(int)
    df["rid16"] = df[COL_RID].astype(str).str.replace(r"\.0$", "", regex=True).str[:16]
    df["ban"] = pd.to_numeric(df[COL_BAN], errors="coerce").astype(int)
    df["year"] = df["date"] // 10000
    df["rel"] = np.clip(6 - df["jyun"], 0, 5).astype(int)
    df["win"] = (df["jyun"] == 1).astype(int)
    df = df.sort_values(["rid16", "ban"]).reset_index(drop=True)
    return df


def numeric_block(df: pd.DataFrame, nums: list[str]) -> np.ndarray:
    """数値列 (train 年に依存しない): v6 と同じ pd.to_numeric(coerce)"""
    return np.column_stack([pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float32)
                            for c in nums]) if nums else np.zeros((len(df), 0), dtype=np.float32)


def cat_block(df: pd.DataFrame, cats: list[str], fit_mask: np.ndarray):
    """カテゴリ列を **fit_mask (train) の語彙だけ**で整数化。未知は __NaN__"""
    codes = np.zeros((len(df), len(cats)), dtype=np.float32)
    vocab = {}
    for j, c in enumerate(cats):
        v = df[c].astype(str)
        classes = np.array(sorted(set(v[fit_mask]) | {"__NaN__"}))
        vocab[c] = classes.tolist()
        vv = v.where(v.isin(set(classes)), "__NaN__")
        codes[:, j] = np.searchsorted(classes, vv.to_numpy())
    return codes, vocab


def assemble(cols, cats, nums, cat_codes, num) -> np.ndarray:
    X = np.zeros((cat_codes.shape[0], len(cols)), dtype=np.float32)
    ci = {c: j for j, c in enumerate(cats)}
    ni = {c: j for j, c in enumerate(nums)}
    for k, c in enumerate(cols):
        X[:, k] = cat_codes[:, ci[c]] if c in ci else num[:, ni[c]]
    X[np.isnan(X)] = -9999.0
    return X


def groups(rid: np.ndarray) -> np.ndarray:
    cut = np.flatnonzero(rid[1:] != rid[:-1]) + 1
    return np.diff(np.concatenate([[0], cut, [len(rid)]]))


# ---------------------------------------------------------------- τ
def fit_tau(score: np.ndarray, win: np.ndarray, races: list[np.ndarray]) -> tuple[float, float]:
    """レース内 1着条件付きロジットの温度 τ (単独勝ちレースのみ)"""
    S, W = [], []
    H = max(len(i) for i in races)
    for idx in races:
        if win[idx].sum() != 1:
            continue
        s = np.full(H, 0.0)
        m = np.zeros(H, dtype=bool)
        s[:len(idx)] = score[idx]
        m[:len(idx)] = True
        S.append(np.where(m, s, -np.inf))
        W.append(np.argmax(win[idx]))
    S = np.array(S)
    W = np.array(W)

    def nll(tau):
        z = S / tau
        mx = np.max(np.where(np.isfinite(z), z, -np.inf), axis=1)
        lse = mx + np.log(np.exp(np.where(np.isfinite(z), z - mx[:, None], -np.inf)).sum(1))
        return float((lse - z[np.arange(len(W)), W]).mean())

    r = minimize_scalar(nll, bounds=(0.02, 50.0), method="bounded", options={"xatol": 1e-5})
    return float(r.x), float(r.fun)


def race_index(rid: np.ndarray, rows: np.ndarray) -> list[np.ndarray]:
    """rows (昇順) を rid16 ごとに分割 (df は rid16, ban でソート済み)"""
    r = rid[rows]
    cut = np.flatnonzero(r[1:] != r[:-1]) + 1
    return np.split(rows, cut)


# ---------------------------------------------------------------- main
def main():
    t00 = time.time()
    SCORES.mkdir(parents=True, exist_ok=True)
    contract = json.loads((EXP15_OUT / "feature_contract.json").read_text(encoding="utf-8"))
    cols = contract["features"]["clean"]
    cats = [c for c in cols if c in contract["cat_cols"]]
    nums = [c for c in cols if c not in contract["cat_cols"]]
    official = json.loads((RESEARCH / "official_rids_by_year.json").read_text(encoding="utf-8"))

    df = load_rows(contract)
    print(f"rows {len(df):,} races {df['rid16'].nunique():,} "
          f"dates {df['date'].min()}..{df['date'].max()} ({time.time()-t00:.0f}s)", flush=True)
    num = numeric_block(df, nums)
    rid = df["rid16"].to_numpy()
    y_rel = df["rel"].to_numpy()
    win = df["win"].to_numpy()
    year = df["year"].to_numpy()
    key_all = (df["rid16"] + "_" + df["ban"].astype(str)).to_numpy()

    # ---- 重複・欠損の検査 (母集団レベル)
    dup = int(pd.Series(key_all).duplicated().sum())
    tests = {"duplicate_rid_ban_rows": dup, "n_rows": int(len(df)),
             "n_races": int(df["rid16"].nunique()),
             "date_range": [int(df["date"].min()), int(df["date"].max())],
             "assert_no_2024_or_later": bool(df["date"].max() <= D_MAX)}
    assert dup == 0, "同一 (rid16, 馬番) の重複行がある"

    manifest = {
        "spec_version": "0.4 (frozen at commit e7603678)",
        "role": "Q1 rolling OOF の artifact 契約。学習・予測のみ。結果指標は含まない",
        "created": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "master_v2": {
            "absolute_path": str(MASTER), "size_bytes": int(os.stat(MASTER).st_size),
            "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.stat(MASTER).st_mtime)),
            "sha256": sha256(MASTER),
            "row_hash_2016_2023": sha_text("\n".join(sorted(
                (df["rid16"] + "_" + df["ban"].astype(str))[df["year"].between(2016, 2023)]))),
            "exp15_recorded_row_hash": contract.get("row_hash_2016_2023"),
        },
        "feature_schema": {
            "source": str(EXP15_OUT / "feature_contract.json"),
            "file_sha256": sha256(EXP15_OUT / "feature_contract.json"),
            "n_clean": len(cols), "n_cat": len(cats),
            "schema_hash": sha_text("\n".join(sorted(cols))),
        },
        "race_population": {
            "json_sha256": sha256(OUT / "race_population.json"),
            "official_rid_list_sha256": sha_text("\n".join(
                sorted(r for v in official.values() for r in v))),
            "n_official_races": sum(len(v) for v in official.values()),
            "rule": "平地 (トラックコード(JV) 51..59 を除外) / 勝馬一意 / starter>=5 / DNF なし",
        },
        "p0_5_status": p05_c1_check(df),
        "feature_provenance": {
            "jockey_fuku90": {
                "train_side": "build_dataset.py add_rolling_stats。直近90『馬走』の複勝率を shift(1) で "
                              "as-of 計算し、同一 (騎手, レース) ブロックへ先頭値を配る (C1, 2026-09-11)",
                "serve_side": "data/jockey_stats.csv のスナップショット (predict_weekly.py:523 / app.py:858)。"
                              "**as-of ではなく現在値**",
                "used_here": "train 側 (master_v2) のみ",
            },
            "prev_hosei": {
                "train_side": "build_master_v2.py:70 で data/hosei の 『前走補正』を rename (TARGET 由来)",
                "serve_side": "make_weekly_hosei.py。旧実装は前々走を入れる off-by-one があり proxy で修正済み",
                "used_here": "train 側 (master_v2) のみ",
            },
        },
        "recipe": {"family": "v6_lr_half (EXP15 で 2022 選択済み)", "params": CFG, "fixed": FIXED,
                    "max_rounds": MAX_ROUNDS, "early_stopping_rounds": ES_ROUNDS,
                    "label": "rel = clip(6 - 着順, 0, 5)", "sample_weight": "uniform",
                    "train_population": "master_v2 の着順が数値の全行 (EXP15 R0-clean と同一。障害を含む)",
                    "tau_population": "Y-1 の **正式 race set 行のみ** (平地・DNF なし) で条件付きロジット最尤。"
                                       "ただし Y=2016 は ES 年 2015 の正式 race set が存在しない (市場データ突合は 2016 以降) ため "
                                       "2015 の全レース (単独勝ちのみ) で最尤推定する。いずれも Y-1 に閉じている"},
        "fits": {},
    }

    # ---- 年ごとに encoder を作り、seed ごとに学習
    for Y in YEARS:
        tr = year <= Y - 2
        es = year == Y - 1
        pr = year == Y
        assert tr.sum() and es.sum() and pr.sum(), Y
        cat_codes, vocab = cat_block(df, cats, tr)
        X = assemble(cols, cats, nums, cat_codes, num)
        enc_hash = sha_text(json.dumps(vocab, ensure_ascii=False, sort_keys=True))
        tr_rows, es_rows, pr_rows = np.flatnonzero(tr), np.flatnonzero(es), np.flatnonzero(pr)
        # 未来行不使用の実測 (train/ES の日付が Y より前で閉じていること)
        future_check = {
            "train_max_date": int(df["date"].to_numpy()[tr_rows].max()),
            "train_max_year": int(year[tr_rows].max()),
            "es_year_set": sorted(set(year[es_rows].tolist())),
            "predict_year_set": sorted(set(year[pr_rows].tolist())),
            "train_before_Y_minus_1": bool(year[tr_rows].max() <= Y - 2),
            "es_is_Y_minus_1": bool(set(year[es_rows].tolist()) == {Y - 1}),
            "predict_is_Y": bool(set(year[pr_rows].tolist()) == {Y}),
        }
        assert future_check["train_before_Y_minus_1"] and future_check["es_is_Y_minus_1"] \
            and future_check["predict_is_Y"], (Y, future_check)
        # τ 用: Y-1 の正式 race set 行。2015 は市場データ突合の対象外なので ES 年の全レースを使う
        off_prev = set(official.get(str(Y - 1), []))
        if off_prev:
            es_off_rows = es_rows[np.isin(rid[es_rows], list(off_prev))]
            tau_pop = "official_race_set_of_Y_minus_1"
        else:
            es_off_rows = es_rows
            tau_pop = ("all_races_of_Y_minus_1 (正式 race set が無い年。市場データ突合は 2016 以降のため "
                       "Y=2016 のみ該当。Y-1 に閉じているので時間安全)")
        races_es_off = race_index(rid, es_off_rows)
        assert len(es_off_rows) > 0 and len(races_es_off) > 0, (Y, "tau 母集団が空")

        dtr = lgb.Dataset(X[tr_rows], label=y_rel[tr_rows], group=groups(rid[tr_rows]),
                          free_raw_data=False)
        for seed in SEEDS:
            t0 = time.time()
            params = {**FIXED, **CFG, "seed": int(seed)}
            dsl = lgb.Dataset(X[es_rows], label=y_rel[es_rows], group=groups(rid[es_rows]),
                              reference=dtr)
            m = lgb.train(params, dtr, num_boost_round=MAX_ROUNDS, valid_sets=[dsl],
                          callbacks=[lgb.early_stopping(ES_ROUNDS, verbose=False)])
            s_pred = m.predict(X[pr_rows], num_iteration=m.best_iteration).astype(np.float64)
            s_es = m.predict(X[es_off_rows], num_iteration=m.best_iteration).astype(np.float64)
            se = np.full(len(df), np.nan)
            se[es_off_rows] = s_es
            tau, tau_nll = fit_tau(se, win, races_es_off)
            f = SCORES / f"Q1_Y{Y}_s{seed}.npz"
            np.savez(f, rows=pr_rows.astype(np.int64), score=s_pred,
                     key=key_all[pr_rows], year=np.int32(Y), seed=np.int64(seed),
                     tau=np.float64(tau), best_iter=np.int64(m.best_iteration))
            mdl = m.model_to_string(num_iteration=m.best_iteration)
            manifest["fits"][f"Y{Y}_s{seed}"] = {
                "year": Y, "seed": seed,
                "train_period": [int(df["date"].to_numpy()[tr_rows].min()),
                                 int(df["date"].to_numpy()[tr_rows].max())],
                "es_period": [int(df["date"].to_numpy()[es_rows].min()),
                              int(df["date"].to_numpy()[es_rows].max())],
                "predict_period": [int(df["date"].to_numpy()[pr_rows].min()),
                                   int(df["date"].to_numpy()[pr_rows].max())],
                "n_train_rows": int(len(tr_rows)), "n_train_races": int(len(groups(rid[tr_rows]))),
                "n_es_rows": int(len(es_rows)), "n_es_races": int(len(groups(rid[es_rows]))),
                "n_es_official_rows": int(len(es_off_rows)),
                "n_es_official_races": int(len(races_es_off)),
                "n_predict_rows": int(len(pr_rows)),
                "n_predict_races": int(len(groups(rid[pr_rows]))),
                "encoder_hash": enc_hash, "tau_population": tau_pop,
                "best_iter": int(m.best_iteration), "tau": tau, "es_official_tau_nll": tau_nll,
                "model_sha256": sha_text(mdl),
                "prediction_artifact": str(f.relative_to(BASE)),
                "prediction_artifact_sha256": sha256(f),
                "future_row_check": future_check,
                "seconds": round(time.time() - t0, 1),
            }
            print(f"  Y{Y} s{seed}: iter={m.best_iteration} tau={tau:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        del dtr, X, cat_codes
        print(f"[year {Y}] done ({time.time()-t00:.0f}s total)", flush=True)

    (OUT / "oof_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1,
                                                      default=float), encoding="utf-8")
    tests["fits"] = len(manifest["fits"])
    tests["elapsed_sec"] = round(time.time() - t00, 1)
    (OUT / "oof_build_checks.json").write_text(json.dumps(tests, ensure_ascii=False, indent=1),
                                              encoding="utf-8")
    print(f"[saved] out/oof_manifest.json ({len(manifest['fits'])} fits, "
          f"{tests['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
