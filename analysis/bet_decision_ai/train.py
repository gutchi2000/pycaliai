# -*- coding: utf-8 -*-
"""
train.py — 馬券判断 AI の学習・選択（仕様 v1.0 §4・§6）
=======================================================
学習 2017-2020、選択 2021-2022。**2023-2025 は読まない**（data.py の唯一の入口で year<=2022 に絞る）。

1. 特徴量（仕様 §4 の許可列のみ）→ 学習期間で標準化
2. NN 判断 AI: 16 設定のグリッド（幅 × 層 × L2 × 学習率、エポックは選択期間の平均対数成長で早期打ち切り）を
   種 0 で回し、選択期間の平均対数成長が最大の設定を選ぶ → その設定で種 0-4 を学習・保存
3. B4 市場版: 同じ手順（同じグリッド）を、p_model→q_market に差し替えた入力で
4. GBM 版（比較）: 12 設定（葉数 × 最小葉サンプル × 配分閾値）を種 0 で選び、種 0-4 を学習・保存
5. 選択期間（と参考に学習期間）で全方策を評価 → out/dev_results.json
モデル・設定・sha256 は analysis/bet_decision_ai/models/ に保存（manifest.json）。

実行: python -m analysis.bet_decision_ai.train
"""
from __future__ import annotations

import argparse
import itertools
import warnings
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from analysis.bet_decision_ai import baselines as BL
from analysis.bet_decision_ai import data as D
from analysis.bet_decision_ai import evaluate as E
from analysis.bet_decision_ai import policy as P

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "models"
OUT_DIR = HERE / "out"
SEEDS = [0, 1, 2, 3, 4]

# 1 回目（run1、out/run1_*）は softmax_abstain だけの 16 設定で、全設定が学習の数エポックで「全見送り」に潰れた
# （学習期間だけの診断で in-sample でも潰れることを確認＝勾配が消える最適化の罠）。2 回目は配分の作り方を
# 3 種（softmax_abstain / sigmoid_ticket / relu_st）に広げた 16 設定。延べの試行設定数は 32（run1 は 1R も買わず
# 選択期間の回収率の情報は得ていない）。
_NN_BASE = dict(max_epochs=60, patience=8, batch=512)
NN_GRID = ([dict(_NN_BASE, stake_mode="softmax_abstain", abstain_bias=4.0, width=w, depth=2, l2=1e-4, lr=lr)
            for w, lr in itertools.product([32, 128], [1e-3, 3e-3])]
           + [dict(_NN_BASE, stake_mode="sigmoid_ticket", abstain_bias=5.0, width=w, depth=2, l2=1e-4, lr=lr)
              for w, lr in itertools.product([32, 128], [1e-3, 3e-3])]
           + [dict(_NN_BASE, stake_mode="relu_st", abstain_bias=0.0, width=w, depth=2, l2=l2, lr=lr)
              for w, l2, lr in itertools.product([32, 128], [1e-4, 3e-3], [3e-4, 1e-3])])
GBM_GRID = [dict(num_leaves=nl, min_child_samples=mc, thr=thr)
            for nl, mc, thr in itertools.product([15, 63], [500, 5000], [0.0, 0.002, 0.005])]
GBM_FIXED = dict(objective="regression", learning_rate=0.03, n_estimators=2000, subsample=0.8, subsample_freq=1,
                 colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)
assert len(NN_GRID) <= 20 and len(GBM_GRID) <= 20


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def make_tensors(cand: pd.DataFrame, mode: str, mkt_ctx: pd.DataFrame | None, std: D.Standardizer | None,
                 fit_mask: np.ndarray | None = None):
    f = D.ticket_features(cand, mode, mkt_ctx)
    if std is None:
        std = D.Standardizer().fit(f[fit_mask] if fit_mask is not None else f)
    X = std.transform(f)
    return D.RaceTensor(cand, X), std, f.columns.tolist()


def split(T: D.RaceTensor, lo: int, hi: int) -> D.RaceTensor:
    return T.subset((T.years >= lo) & (T.years <= hi))


def nn_search(Ttr, Tsel, device, tag):
    rows = []
    for ci, cfg in enumerate(NN_GRID):
        t0 = time.time()
        m, info = P.train_nn(cfg, Ttr.X, Ttr.mask, Ttr.ret, Tsel.X, Tsel.mask, Tsel.ret, seed=0, device=device)
        h = info["history"][info["best_epoch"]]
        s = P.predict_nn(m, Tsel.X, Tsel.mask, device)
        sr = E.round_stakes(s, 10000) / 10000.0
        rows.append({"cfg_id": ci, **cfg, "best_epoch": info["best_epoch"], "sel_lg": info["best_sel_lg"],
                     "sel_mean_stake_frac": h["sel_mean_stake_frac"], "train_lg": h["train_lg"],
                     "sel_lg_rounded_B10000": lg_of(sr, Tsel), "sel_n_bet_races_B10000": int((sr.sum(1) > 0).sum())})
        log(f"  [{tag}] cfg{ci:02d} {cfg['stake_mode']} w={cfg['width']} d={cfg['depth']} l2={cfg['l2']} lr={cfg['lr']} "
            f"ep={info['best_epoch']} sel_lg={info['best_sel_lg']:.6f} frac={h['sel_mean_stake_frac']:.4f} "
            f"lg_round={rows[-1]['sel_lg_rounded_B10000']:.6f} betR={rows[-1]['sel_n_bet_races_B10000']} "
            f"({time.time() - t0:.0f}s)")
    best = max(rows, key=lambda r: r["sel_lg"])
    return best, rows


def nn_seeds(cfg, Ttr, Tsel, device, tag):
    out = []
    for sd in SEEDS:
        m, info = P.train_nn(cfg, Ttr.X, Ttr.mask, Ttr.ret, Tsel.X, Tsel.mask, Tsel.ret, seed=sd, device=device)
        path = MODEL_DIR / f"{tag}_seed{sd}.pt"
        torch.save(m.state_dict(), path)
        out.append({"seed": sd, "path": path.name, "best_epoch": info["best_epoch"], "sel_lg": info["best_sel_lg"],
                    "model": m})
        log(f"  [{tag}] seed{sd} ep={info['best_epoch']} sel_lg={info['best_sel_lg']:.6f}")
    return out


def gbm_fit(cfg, Ttr, Tsel, seed):
    import lightgbm as lgb
    mtr, msel = Ttr.mask, Tsel.mask
    Xtr, ytr = Ttr.X[mtr], P.gbm_target(Ttr.ret[mtr])
    Xs, ys = Tsel.X[msel], P.gbm_target(Tsel.ret[msel])
    m = lgb.LGBMRegressor(num_leaves=cfg["num_leaves"], min_child_samples=cfg["min_child_samples"],
                          random_state=seed, **GBM_FIXED)
    m.fit(Xtr, ytr, eval_set=[(Xs, ys)], callbacks=[lgb.early_stopping(100, verbose=False)])
    return m


def gbm_pred(m, T) -> np.ndarray:
    pred = np.full(T.mask.shape, -np.inf, np.float64)
    pred[T.mask] = m.predict(T.X[T.mask], num_iteration=m.best_iteration_)
    return pred


def lg_of(s, T) -> float:
    return float(np.log1p(D.KAPPA * (s * (T.ret - 1.0)).sum(1)).mean())


def gbm_search(Ttr, Tsel):
    rows, cache = [], {}
    for ci, cfg in enumerate(GBM_GRID):
        key = (cfg["num_leaves"], cfg["min_child_samples"])
        if key not in cache:
            m = gbm_fit(cfg, Ttr, Tsel, 0)
            cache[key] = (m, gbm_pred(m, Tsel))
        m, pr = cache[key]
        s = P.gbm_stakes(pr, Tsel.mask, cfg["thr"])
        rows.append({"cfg_id": ci, **cfg, "best_iter": int(m.best_iteration_), "sel_lg": lg_of(s, Tsel),
                     "sel_n_bet_races": int((s.sum(1) > 0).sum()), "sel_pred_pos_rate": float((pr[Tsel.mask] > 0).mean())})
        log(f"  [gbm] cfg{ci:02d} {cfg} it={m.best_iteration_} sel_lg={rows[-1]['sel_lg']:.6f} "
            f"bet_races={rows[-1]['sel_n_bet_races']}")
    best = max(rows, key=lambda r: r["sel_lg"])
    return best, rows


def strip_books(res: dict) -> dict:
    return {k: v for k, v in res.items() if k != "books"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="動作確認（グリッド 2 設定・エポック少）")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    t_start = time.time()
    spec_sha = D.check_spec()
    MODEL_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"spec sha {spec_sha[:16]}  device={device}")
    global NN_GRID, GBM_GRID
    if args.quick:
        NN_GRID = [dict(c, max_epochs=3) for c in NN_GRID[:2]]
        GBM_GRID = GBM_GRID[:2]

    # ---------------- data（2017-2022 のみ）
    cand = D.load_candidates(2017, D.DEV_MAX_YEAR)
    assert cand["year"].max() <= D.DEV_MAX_YEAR
    races15 = D.load_races(2015, D.DEV_MAX_YEAR)
    base15 = D.load_base_market(2015, D.DEV_MAX_YEAR)
    mkt_ctx = D.market_race_context(base15, races15)
    log(f"cand rows={len(cand):,} races={cand['rid16'].nunique():,}  unsettleable excluded="
        f"{cand.attrs['n_unsettleable_races']} deadheat→100={cand.attrs['n_deadheat_100']}")
    trmask = cand["year"].between(*D.TRAIN_YEARS).to_numpy()
    Tm, std_m, cols_m = make_tensors(cand, "model", None, None, trmask)
    Tq, std_q, cols_q = make_tensors(cand, "market", mkt_ctx, None, trmask)
    assert (Tm.races == Tq.races).all()
    Tm_tr, Tm_sel = split(Tm, *D.TRAIN_YEARS), split(Tm, *D.SELECT_YEARS)
    Tq_tr, Tq_sel = split(Tq, *D.TRAIN_YEARS), split(Tq, *D.SELECT_YEARS)
    log(f"train races={len(Tm_tr.races):,} select races={len(Tm_sel.races):,} features={len(cols_m)}")
    (MODEL_DIR / "standardizer_model.json").write_text(json.dumps(std_m.to_dict()), encoding="utf-8")
    (MODEL_DIR / "standardizer_market.json").write_text(json.dumps(std_q.to_dict()), encoding="utf-8")

    # ---------------- NN（判断 AI）と B4
    log("NN grid (model inputs)")
    best_m, grid_m = nn_search(Tm_tr, Tm_sel, device, "nn")
    log(f"  selected: {best_m}")
    log("NN grid (market inputs = B4)")
    best_q, grid_q = nn_search(Tq_tr, Tq_sel, device, "b4")
    log(f"  selected: {best_q}")
    cfg_m = {k: best_m[k] for k in NN_GRID[0]}
    cfg_q = {k: best_q[k] for k in NN_GRID[0]}
    log("NN seeds")
    seeds_m = nn_seeds(cfg_m, Tm_tr, Tm_sel, device, "nn_model")
    seeds_q = nn_seeds(cfg_q, Tq_tr, Tq_sel, device, "nn_market")

    # ---------------- GBM
    log("GBM grid")
    best_g, grid_g = gbm_search(Tm_tr, Tm_sel)
    log(f"  selected: {best_g}")
    cfg_g = {k: best_g[k] for k in GBM_GRID[0]}
    seeds_g = []
    for sd in SEEDS:
        m = gbm_fit(cfg_g, Tm_tr, Tm_sel, sd)
        path = MODEL_DIR / f"gbm_model_seed{sd}.txt"
        m.booster_.save_model(str(path), num_iteration=m.best_iteration_)
        seeds_g.append({"seed": sd, "path": path.name, "best_iter": int(m.best_iteration_), "model": m})
        log(f"  [gbm] seed{sd} it={m.best_iteration_}")

    edges = E.quintile_edges(cand[trmask])
    config = {"spec_sha256": spec_sha, "created": datetime.now().isoformat(timespec="seconds"),
              "train_years": D.TRAIN_YEARS, "select_years": D.SELECT_YEARS, "kappa": D.KAPPA,
              "feature_cols_model": cols_m, "feature_cols_market": cols_q,
              "nn_model": {"cfg": cfg_m, "seeds": [{k: v for k, v in s.items() if k != "model"} for s in seeds_m]},
              "nn_market_B4": {"cfg": cfg_q, "seeds": [{k: v for k, v in s.items() if k != "model"} for s in seeds_q]},
              "gbm_model": {"cfg": cfg_g, "fixed": GBM_FIXED,
                            "seeds": [{k: v for k, v in s.items() if k != "model"} for s in seeds_g]},
              "grid_nn_model": grid_m, "grid_nn_market": grid_q, "grid_gbm": grid_g,
              "quintile_edges_lpq_train": edges,
              "selection_rule": "選択期間 2021-22 の平均対数成長（連続の割合、種 0）が最大の設定"}
    (MODEL_DIR / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=1), encoding="utf-8")
    files = sorted(p for p in MODEL_DIR.iterdir() if p.is_file() and p.name != "manifest.json")
    manifest = {"created": config["created"], "spec_sha256": spec_sha,
                "files": {p.name: D.sha256(p) for p in files}}
    (MODEL_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    log(f"saved {len(files)} files + manifest")

    # ---------------- 評価（選択期間・学習期間）
    def fracs_for(Tm_, Tq_):
        fr = {}
        sm = [P.predict_nn(s["model"], Tm_.X, Tm_.mask, device) for s in seeds_m]
        sq = [P.predict_nn(s["model"], Tq_.X, Tq_.mask, device) for s in seeds_q]
        sg = [P.gbm_stakes(gbm_pred(s["model"], Tm_), Tm_.mask, cfg_g["thr"]) for s in seeds_g]
        fr["AI"] = np.mean(sm, 0)
        fr["B4_market_AI"] = np.mean(sq, 0)
        fr["GBM"] = np.mean(sg, 0)
        for i, s in enumerate(sm):
            fr[f"AI_seed{i}"] = s
        for i, s in enumerate(sq):
            fr[f"B4_seed{i}"] = s
        return fr

    waku_map = D.load_waku_map(2017, D.DEV_MAX_YEAR)
    tan_snap = D.load_tan_snapshot_full_field(2017, D.DEV_MAX_YEAR)
    wak = D.load_wakuren_for_b1(2017, D.DEV_MAX_YEAR)
    base17 = base15[base15["year"] >= 2017]
    results = {"config": {k: v for k, v in config.items() if not k.startswith("grid_")}, "periods": {}}
    for pname, (lo, hi), Tm_, Tq_ in [("select_2021_2022", D.SELECT_YEARS, Tm_sel, Tq_sel),
                                      ("train_2017_2020", D.TRAIN_YEARS, Tm_tr, Tq_tr)]:
        cp = cand[cand["year"].between(lo, hi)]
        universe = pd.DataFrame({"rid16": Tm_.races, "date": Tm_.dates})
        fr = fracs_for(Tm_, Tq_)
        per = {}
        for B in (10000, 2000):
            bb = {"B0_null": BL.b0_null(cp, B), "B3_prob_first": BL.b3_prob_first(cp, B)}
            b1, info1 = BL.b1_a95(cp, base17[base17["year"].between(lo, hi)], wak[wak["year"].between(lo, hi)],
                                  waku_map, B)
            b2, info2 = BL.b2_market_fav(cp, base17[base17["year"].between(lo, hi)], tan_snap, B)
            bb["B1_a95"], bb["B2_market_fav"] = b1, b2
            comps = ["B0_null", "B1_a95", "B2_market_fav", "B3_prob_first", "B4_market_AI"]
            res = E.evaluate_period(Tm_, fr, bb, universe, B, edges,
                                    mains=["AI", "GBM"] + [f"AI_seed{i}" for i in SEEDS], compare_to=comps + ["GBM", "AI"])
            res["baseline_info"] = {"B1": info1, "B2": info2}
            if B == 10000 and pname.startswith("select"):
                for k in ("AI", "B4_market_AI", "GBM"):
                    res["policies"][k]["lg_continuous"] = lg_of(fr[k], Tm_)
                    res["policies"][k]["mean_stake_frac_continuous"] = float(fr[k].sum(1).mean())
                # 判断 AI の乖離五分位: 選択期間の全候補を 100 円均等で買った場合（記述用の参照）
                allc = cp.assign(stake=100, payout100=cp["payout"], lpq=cp["log_p_over_q"], date=cp["date"].astype(str))
                res["quintiles"]["ALL_CANDIDATES_flat100"] = E.quintile_table(
                    allc[["rid16", "date", "bet_type", "combo", "stake", "payout100", "p_model", "lpq"]], edges,
                    universe, 10000)
                books = res["books"]
                books_out = OUT_DIR / "dev_books_select.parquet"
                pd.concat([b.assign(policy=k) for k, b in books.items() if len(b) and not k.startswith(("AI_seed", "B4_seed"))]
                          ).to_parquet(books_out, index=False)
            per[f"B{B}"] = strip_books(res)
            log(f"[{pname} B={B}] " + "  ".join(
                f"{k}:{(v['roi'] or 0) * 100:.1f}%/{v['n_bet_races']}R" for k, v in res["policies"].items()
                if not k.startswith(("AI_seed", "B4_seed"))))
        results["periods"][pname] = per
    results["elapsed_sec"] = round(time.time() - t_start, 1)
    (OUT_DIR / "dev_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1, default=float),
                                              encoding="utf-8")
    log(f"done {results['elapsed_sec']}s → out/dev_results.json")


if __name__ == "__main__":
    main()
