# -*- coding: utf-8 -*-
"""
oof_tests.py — EXP16A Stage 1-1b: rolling OOF の検査と Q1 確率の整形
=====================================================================
検査するもの (いずれも **勝敗を使わない**。2019-2023 の結果指標は計算しない):
  T1 行対応      : npz の key が master の (rid16, 馬番) と一致し、行が一意・昇順
  T2 被覆        : 正式 race set の各レースの全 starter がちょうど 1 回スコアされている
  T3 確率和      : レース内 softmax(score/τ) の和が 1 (最大誤差を報告)
  T4 再現性      : 1 fit を同一条件で再学習し、保存済みスコアと厳密一致するか
  T5 削除不変性  : 予測行からレース単位で 20% を削除して再予測し、残った行のスコアが厳密一致するか
  T6 未来行不使用: manifest の train/ES/predict 期間が Y-2 / Y-1 / Y に閉じているか
  T7 seed ばらつき: 標準化した log(Q1/π) の seed 間 per-horse SD (検出力監査の実測値に使う)
出力:
  out/oof_tests.json
  data/_research/mcond/exp16a/q1_official.npz  (正式 race set に整列した Q1 確率, seed 別)
実行: python -m analysis.mcond.exp16a_close_market_residual_dev.oof_tests
"""
from __future__ import annotations

import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from .provenance import BASE, OUT
from .build_oof import (CFG, ES_ROUNDS, FIXED, MAX_ROUNDS, RESEARCH, SCORES, SEEDS, YEARS,
                        assemble, cat_block, groups, load_rows, numeric_block)
from .stage0_dry_run import EXP15_OUT

NPZ_OFF = RESEARCH / "official_races_all.npz"


def softmax_race(score: np.ndarray, off: np.ndarray, tau: float) -> np.ndarray:
    p = np.empty_like(score)
    for a, b in zip(off[:-1], off[1:]):
        z = score[a:b] / tau
        z = z - z.max()
        e = np.exp(z)
        p[a:b] = e / e.sum()
    return p


def main():
    t0 = time.time()
    contract = json.loads((EXP15_OUT / "feature_contract.json").read_text(encoding="utf-8"))
    cols = contract["features"]["clean"]
    cats = [c for c in cols if c in contract["cat_cols"]]
    nums = [c for c in cols if c not in contract["cat_cols"]]
    manifest = json.loads((OUT / "oof_manifest.json").read_text(encoding="utf-8"))
    df = load_rows(contract)
    key_all = (df["rid16"] + "_" + df["ban"].astype(str)).to_numpy()
    year = df["year"].to_numpy()
    res = {"role": "rolling OOF の検査。勝敗を使わない。2019-2023 の結果指標は計算しない",
           "created": time.strftime("%Y-%m-%d %H:%M:%S %z")}

    # ---------------- T1 行対応 / T3 確率和 (全 fit)
    t1 = {"checked_fits": 0, "key_mismatch": 0, "rows_not_unique": 0, "rows_not_sorted": 0,
          "row_count_mismatch": 0}
    t3 = {"max_abs_sum_minus_one": 0.0, "races_checked": 0}
    d = np.load(NPZ_OFF, allow_pickle=True)
    off_rid, off_ban, off_off = d["rid16"].astype(str), d["ban"], d["offsets"]
    off_year = d["year"]
    off_key = np.array([f"{off_rid[i]}_{off_ban[j]}"
                        for i in range(len(off_rid))
                        for j in range(off_off[i], off_off[i + 1])])
    # 正式 set の horse 行 → 年
    off_key_year = np.repeat(off_year, np.diff(off_off))
    q1 = {}
    for Y in YEARS:
        for seed in SEEDS:
            z = np.load(SCORES / f"Q1_Y{Y}_s{seed}.npz", allow_pickle=True)
            rows, score, key = z["rows"], z["score"], z["key"].astype(str)
            tau = float(z["tau"])
            t1["checked_fits"] += 1
            t1["key_mismatch"] += int((key_all[rows] != key).sum())
            t1["rows_not_unique"] += int(len(np.unique(rows)) != len(rows))
            t1["rows_not_sorted"] += int(not np.all(np.diff(rows) > 0))
            t1["row_count_mismatch"] += int(len(rows) != int((year == Y).sum()))
            q1[(Y, seed)] = (dict(zip(key, score)), tau)

    # ---------------- T2 被覆 + Q1 確率の整形
    t2 = {"missing_horse_rows": 0, "duplicate_horse_rows": 0, "n_official_horse_rows": len(off_key)}
    P = np.zeros((len(SEEDS), len(off_key)), dtype=np.float64)
    S = np.zeros((len(SEEDS), len(off_key)), dtype=np.float64)
    for si, seed in enumerate(SEEDS):
        sc = np.full(len(off_key), np.nan)
        taus = {}
        for Y in YEARS:
            m = off_key_year == Y
            dct, tau = q1[(Y, seed)]
            taus[Y] = tau
            miss = 0
            vals = np.empty(int(m.sum()))
            for i, k in enumerate(off_key[m]):
                v = dct.get(k)
                if v is None:
                    miss += 1
                    vals[i] = np.nan
                else:
                    vals[i] = v
            sc[m] = vals
            t2["missing_horse_rows"] += miss
        S[si] = sc
        # レース内 softmax (年ごとの τ)
        p = np.empty_like(sc)
        for i in range(len(off_rid)):
            a, b = off_off[i], off_off[i + 1]
            tau = taus[int(off_year[i])]
            z2 = sc[a:b] / tau
            z2 = z2 - z2.max()
            e = np.exp(z2)
            p[a:b] = e / e.sum()
        P[si] = p
    sums = np.array([P[:, a:b].sum(axis=1) for a, b in zip(off_off[:-1], off_off[1:])])
    t3["max_abs_sum_minus_one"] = float(np.abs(sums - 1.0).max())
    t3["races_checked"] = int(len(off_rid))
    t2["duplicate_horse_rows"] = int(len(off_key) - len(set(off_key.tolist())))

    np.savez_compressed(RESEARCH / "q1_official.npz", key=off_key, year=off_key_year,
                        offsets=off_off, seeds=np.array(SEEDS, dtype=np.int64),
                        score=S, prob=P)

    # ---------------- T4 再現性 / T5 削除不変性 (1 fit)
    Y, seed = 2022, SEEDS[0]
    tr, es, pr = year <= Y - 2, year == Y - 1, year == Y
    cat_codes, _ = cat_block(df, cats, tr)
    X = assemble(cols, cats, nums, cat_codes, numeric_block(df, nums))
    rid = df["rid16"].to_numpy()
    y_rel = df["rel"].to_numpy()
    tr_rows, es_rows, pr_rows = np.flatnonzero(tr), np.flatnonzero(es), np.flatnonzero(pr)
    dtr = lgb.Dataset(X[tr_rows], label=y_rel[tr_rows], group=groups(rid[tr_rows]),
                      free_raw_data=False)
    dsl = lgb.Dataset(X[es_rows], label=y_rel[es_rows], group=groups(rid[es_rows]), reference=dtr)
    m = lgb.train({**FIXED, **CFG, "seed": int(seed)}, dtr, num_boost_round=MAX_ROUNDS,
                  valid_sets=[dsl], callbacks=[lgb.early_stopping(ES_ROUNDS, verbose=False)])
    s_new = m.predict(X[pr_rows], num_iteration=m.best_iteration)
    z = np.load(SCORES / f"Q1_Y{Y}_s{seed}.npz", allow_pickle=True)
    res["T4_reproducibility"] = {
        "fit": f"Y{Y}_s{seed}",
        "best_iter_saved": int(z["best_iter"]), "best_iter_rerun": int(m.best_iteration),
        "max_abs_score_diff": float(np.abs(s_new - z["score"]).max()),
        "bitwise_identical": bool(np.array_equal(s_new, z["score"])),
    }
    # 削除不変性: 予測行からレース単位で 20% を削除して再予測
    rng = np.random.default_rng(20260924)
    races_pr = np.split(pr_rows, np.flatnonzero(rid[pr_rows][1:] != rid[pr_rows][:-1]) + 1)
    keep = rng.random(len(races_pr)) >= 0.20
    sub_rows = np.concatenate([r for r, k in zip(races_pr, keep) if k])
    s_sub = m.predict(X[sub_rows], num_iteration=m.best_iteration)
    pos = {r: i for i, r in enumerate(pr_rows)}
    s_full_at_sub = s_new[[pos[r] for r in sub_rows]]
    res["T5_deletion_invariance"] = {
        "removed_races": int((~keep).sum()), "kept_rows": int(len(sub_rows)),
        "max_abs_diff": float(np.abs(s_sub - s_full_at_sub).max()),
        "bitwise_identical": bool(np.array_equal(s_sub, s_full_at_sub)),
        "meaning": "他の馬・他のレースを消しても残った行のスコアが変わらない (行単位スコアであること)",
    }

    # ---------------- T6 未来行不使用 (manifest から再確認)
    t6 = {"violations": [], "checked": 0}
    for k, f in manifest["fits"].items():
        Yf = f["year"]
        t6["checked"] += 1
        if not (f["train_period"][1] // 10000 <= Yf - 2):
            t6["violations"].append(f"{k}: train max {f['train_period'][1]}")
        if not (f["es_period"][0] // 10000 == Yf - 1 == f["es_period"][1] // 10000):
            t6["violations"].append(f"{k}: es {f['es_period']}")
        if not (f["predict_period"][0] // 10000 == Yf == f["predict_period"][1] // 10000):
            t6["violations"].append(f"{k}: predict {f['predict_period']}")
        if not f["future_row_check"]["train_before_Y_minus_1"]:
            t6["violations"].append(f"{k}: future_row_check")

    # ---------------- T7 seed ばらつき (log(Q1/π) の標準化方向の per-horse SD)
    pi_close = d["pi_close"]
    dirs = []
    for si in range(len(SEEDS)):
        raw = np.log(np.clip(P[si], 1e-12, None)) - np.log(np.clip(pi_close, 1e-12, None))
        z2 = np.empty_like(raw)
        for a, b in zip(off_off[:-1], off_off[1:]):
            v = raw[a:b]
            v = v - v.mean()
            sd = v.std()
            z2[a:b] = v / (sd if sd > 0 else 1.0)
        dirs.append(z2)
    dirs = np.array(dirs)
    res["T7_seed_spread"] = {
        "definition": "各 seed の log(Q1/π_close) をレース内標準化した方向 s_seed の per-horse SD",
        "mean_per_horse_sd_across_seeds": float(dirs.std(axis=0, ddof=1).mean()),
        "median_per_horse_sd_across_seeds": float(np.median(dirs.std(axis=0, ddof=1))),
        "mean_pairwise_correlation": float(np.mean([np.corrcoef(dirs[i], dirs[j])[0, 1]
                                                    for i in range(len(SEEDS))
                                                    for j in range(i + 1, len(SEEDS))])),
        "use": "検出力監査 (power_audit) の seed_jitter を実測値で置き換えるための参照値",
    }

    res["T1_row_correspondence"] = t1
    res["T2_coverage"] = t2
    res["T3_probability_sum"] = t3
    res["T6_future_rows"] = t6
    res["all_passed"] = bool(
        t1["key_mismatch"] == 0 and t1["rows_not_unique"] == 0 and t1["rows_not_sorted"] == 0
        and t1["row_count_mismatch"] == 0 and t2["missing_horse_rows"] == 0
        and t2["duplicate_horse_rows"] == 0 and t3["max_abs_sum_minus_one"] < 1e-9
        and res["T4_reproducibility"]["bitwise_identical"]
        and res["T5_deletion_invariance"]["bitwise_identical"] and not t6["violations"])
    res["elapsed_sec"] = round(time.time() - t0, 1)
    (OUT / "oof_tests.json").write_text(json.dumps(res, ensure_ascii=False, indent=1,
                                                   default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "T7_seed_spread"},
                     ensure_ascii=False)[:1400])
    print(json.dumps(res["T7_seed_spread"], ensure_ascii=False))
    print(f"[saved] out/oof_tests.json  all_passed={res['all_passed']} ({res['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
