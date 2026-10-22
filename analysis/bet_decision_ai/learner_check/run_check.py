# -*- coding: utf-8 -*-
"""
run_check.py — v1 学習器の合成データ検証（事前登録 docs/research/BET_DECISION_AI_LEARNER_CHECK_PREREG_20261023.md）
====================================================================================================================
v1 の train.py / policy.py / evaluate.py を **変更せずに import** して、合成払戻の上で同じ手順（学習 2017-20 → 選択 2021-22、
NN 16 設定グリッド＋種 5 本平均、GBM 12 設定グリッド＋種 5 本平均）を回し、§6 の判定量と §7 の判定を出す。

実行: python -m analysis.bet_decision_ai.learner_check.run_check [--seeds 11,22,33] [--tag v1]
      修正版の学習器は --learner-module で差し替える（learner_check/ 配下の別ファイル）。
出力: learner_check/out/results_{tag}.json, histories_{tag}.json, log_{tag}.txt
"""
from __future__ import annotations

import argparse
import importlib
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from analysis.bet_decision_ai import data as D
from analysis.bet_decision_ai import evaluate as E
from analysis.bet_decision_ai.learner_check import synth as SY

OUT = Path(__file__).resolve().parent / "out"
B = 10000
_LOGF = None


def log(msg):
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    if _LOGF:
        with open(_LOGF, "a", encoding="utf-8") as f:
            f.write(line + "\n")


class Ctx:
    """特徴量・テンソル・部分集合の固定部分（払戻以外）。"""

    def __init__(self):
        cand = SY.load_universe()
        self.cand = cand
        self.edges = SY.subset_edges(cand)
        self.S = SY.subset_mask(cand, self.edges)
        self.lay = SY.race_layout(cand)
        trmask = cand["year"].between(*D.TRAIN_YEARS).to_numpy()
        f = D.ticket_features(cand, "model", None)
        self.std = D.Standardizer().fit(f[trmask])
        X = self.std.transform(f)
        c0 = cand.assign(payout=0.0)
        self.T = D.RaceTensor(c0, X)
        assert (self.T.races == self.lay["races"]).all()
        self.inv = self.lay["inv"]
        self.slot = D.slot_index(cand)
        self.Sslot = np.zeros(self.T.mask.shape, bool)
        self.Sslot[self.inv, self.slot] = self.S
        self.tr = (self.T.years >= D.TRAIN_YEARS[0]) & (self.T.years <= D.TRAIN_YEARS[1])
        self.sel = (self.T.years >= D.SELECT_YEARS[0]) & (self.T.years <= D.SELECT_YEARS[1])
        self.sel_rows = np.nonzero(self.sel)[0]
        self.universe_sel = pd.DataFrame({"rid16": self.T.races[self.sel], "date": self.T.dates[self.sel]})
        self.universe_tr = pd.DataFrame({"rid16": self.T.races[self.tr], "date": self.T.dates[self.tr]})

    def ret_matrix(self, payout: np.ndarray) -> np.ndarray:
        r = np.zeros(self.T.mask.shape, np.float32)
        r[self.inv, self.slot] = payout / 100.0
        return r


def tensor_with(ctx: Ctx, ret: np.ndarray, rows: np.ndarray) -> D.RaceTensor:
    T = ctx.T.subset(rows)
    T.ret = ret[rows]
    return T


def eval_stakes(frac: np.ndarray, T: D.RaceTensor, universe: pd.DataFrame, Sslot: np.ndarray, W: np.ndarray,
                oracle_lg: float | None = None) -> dict:
    st = E.round_stakes(frac, B)
    book = E.book_from_tensor(T, st)
    m = E.metrics(book, universe, B, W)
    tot = st.sum()
    out = {k: m.get(k) for k in ["n_bet_races", "n_tickets", "stake", "return", "roi", "roi_ci", "log_growth",
                                 "log_growth_ci", "roi_ex_top10", "race_hit_rate"]}
    out["bet_race_share"] = m["n_bet_races"] / len(universe)
    out["subset_stake_share"] = float(st[Sslot].sum() / tot) if tot > 0 else None
    out["mean_stake_frac_continuous"] = float(frac.sum(1).mean())
    out["lg_continuous"] = float(np.log1p(D.KAPPA * (frac * (T.ret - 1.0)).sum(1)).mean())
    if tot > 0:
        bt = {}
        for b in D.BET_TYPES:
            cols = D.SLOT_BET_TYPE == b
            s = st[:, cols].sum()
            if s > 0:
                bt[b] = {"stake_share": float(s / tot), "roi": float((st[:, cols] * T.ret[:, cols]).sum() / s)}
        out["by_bet_type"] = bt
    if oracle_lg is not None and oracle_lg != 0:
        out["ratio_to_oracle"] = float(m["log_growth"] / oracle_lg)
    return out


def run_nn(L, Ttr, Tsel, device):
    grid_rows, hists = [], []
    for ci, cfg in enumerate(L.NN_GRID):
        t0 = time.time()
        m, info = L.train_nn(cfg, Ttr.X, Ttr.mask, Ttr.ret, Tsel.X, Tsel.mask, Tsel.ret, seed=0, device=device)
        h = info["history"]
        b = h[info["best_epoch"]]
        grid_rows.append({"cfg_id": ci, "stake_mode": cfg["stake_mode"], "width": cfg["width"], "l2": cfg["l2"],
                          "lr": cfg["lr"], "best_epoch": info["best_epoch"], "n_epochs": len(h),
                          "sel_lg": info["best_sel_lg"], "sel_frac_at_best": b["sel_mean_stake_frac"],
                          "train_lg_at_best": b["train_lg"], "max_train_lg": max(x["train_lg"] for x in h),
                          "max_sel_lg": max(x["sel_lg"] for x in h)})
        hists.append([[x["epoch"], x["train_lg"], x["sel_lg"], x["sel_mean_stake_frac"]] for x in h])
        log(f"    nn cfg{ci:02d} {cfg['stake_mode']:15s} w={cfg['width']:3d} l2={cfg['l2']} lr={cfg['lr']} "
            f"ep={info['best_epoch']:2d}/{len(h)} sel_lg={info['best_sel_lg']:.6f} frac={b['sel_mean_stake_frac']:.3f} "
            f"maxtrain={grid_rows[-1]['max_train_lg']:.6f} ({time.time() - t0:.0f}s)")
    best = max(grid_rows, key=lambda r: r["sel_lg"])
    cfg = L.NN_GRID[best["cfg_id"]]
    seeds = []
    for sd in range(5):
        m, info = L.train_nn(cfg, Ttr.X, Ttr.mask, Ttr.ret, Tsel.X, Tsel.mask, Tsel.ret, seed=sd, device=device)
        seeds.append((m, info["best_epoch"], info["best_sel_lg"]))
    return best, grid_rows, hists, seeds


def run_gbm(L, Ttr, Tsel):
    rows, cache = [], {}
    for ci, cfg in enumerate(L.GBM_GRID):
        key = (cfg["num_leaves"], cfg["min_child_samples"])
        if key not in cache:
            m = L.gbm_fit(cfg, Ttr, Tsel, 0)
            cache[key] = (m, L.gbm_pred(m, Tsel))
        m, pr = cache[key]
        s = L.gbm_stakes(pr, Tsel.mask, cfg["thr"])
        lg = float(np.log1p(D.KAPPA * (s * (Tsel.ret - 1.0)).sum(1)).mean())
        rows.append({"cfg_id": ci, **cfg, "best_iter": int(m.best_iteration_), "sel_lg": lg,
                     "sel_n_bet_races": int((s.sum(1) > 0).sum()), "sel_pred_pos_rate": float((pr[Tsel.mask] > 0).mean())})
        log(f"    gbm cfg{ci:02d} {cfg} it={m.best_iteration_} sel_lg={lg:.6f} betR={rows[-1]['sel_n_bet_races']}")
    best = max(rows, key=lambda r: r["sel_lg"])
    cfg = L.GBM_GRID[best["cfg_id"]]
    models = [L.gbm_fit(cfg, Ttr, Tsel, sd) for sd in range(5)]
    return best, rows, cfg, models


def judge(results: dict, seeds: list[int], learners: list[str]) -> dict:
    out = {}
    for ln in learners:
        crit = {}
        for cond in ("e110", "e120"):
            ok, denom, det = 0, 0, []
            for sd in seeds:
                r = results[str(sd)][cond]
                orc = r["oracle"]
                if orc["log_growth_ci"][0] <= 0:
                    det.append({"seed": sd, "excluded_oracle_underpowered": True})
                    continue
                denom += 1
                p = r[ln]["main"]
                c1 = p["log_growth_ci"][0] > 0
                c2 = (p["subset_stake_share"] or 0) >= 0.5
                ok += int(c1 and c2)
                det.append({"seed": sd, "ci_lo_gt0": c1, "subset_share_ge50": c2})
            crit[cond] = {"pass": (denom >= 2 and ok >= 2), "n_ok": ok, "denom": denom,
                          "undecidable": denom < 2, "detail": det}
        okc = sum(int(results[str(sd)]["control"][ln]["main"]["bet_race_share"] <= 0.05) for sd in seeds)
        crit["control"] = {"pass": okc >= 2, "n_ok": okc, "denom": len(seeds)}
        out[ln] = {"criteria": crit, "PASS": all(c["pass"] for c in crit.values())}
    return out


def main():
    global _LOGF
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="11,22,33")
    ap.add_argument("--conditions", default="control,e105,e110,e120")
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--learner-module", default="analysis.bet_decision_ai.learner_check.v1_learner")
    ap.add_argument("--learners", default="nn,gbm")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    OUT.mkdir(exist_ok=True)
    _LOGF = OUT / f"log_{args.tag}.txt"
    L = importlib.import_module(args.learner_module)
    learners = args.learners.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    conds = args.conditions.split(",")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    log(f"tag={args.tag} learner={args.learner_module} seeds={seeds} conds={conds} device={device}")
    ctx = Ctx()
    cand = ctx.cand
    log(f"cand={len(cand):,} races={len(ctx.T.races):,} subset share={ctx.S.mean():.4f} sel races={ctx.sel.sum()}")
    Wsel = E.boot_counts(ctx.universe_sel["date"].nunique())
    Wtr = E.boot_counts(ctx.universe_tr["date"].nunique())
    Tsel0 = ctx.T.subset(ctx.sel)
    Ssel = ctx.Sslot[ctx.sel]
    oracle_cache = {}
    results, histories = {}, {}
    for sd in seeds:
        U, V = SY.draw_uniforms(len(cand), len(ctx.T.races), sd)
        U2, V2 = SY.draw_uniforms(len(cand), len(ctx.T.races), sd + 1000)
        results[str(sd)], histories[str(sd)] = {}, {}
        for cond in conds:
            boost = SY.CONDITIONS[cond]
            pi, pinfo = SY.true_probs(cand, ctx.S, boost)
            hit = SY.draw_outcomes(cand, pi, ctx.lay, U, V)
            ret = ctx.ret_matrix(SY.synthetic_payout(cand, hit))
            hit2 = SY.draw_outcomes(cand, pi, ctx.lay, U2, V2)
            ret2 = ctx.ret_matrix(SY.synthetic_payout(cand, hit2))
            Ttr = tensor_with(ctx, ret, ctx.tr)
            Tsel = tensor_with(ctx, ret, ctx.sel)
            Tfresh = tensor_with(ctx, ret2, ctx.sel)
            # 合成の素性（記述）
            synth_desc = {"mean_R_all_train": float(Ttr.ret[Ttr.mask].mean()),
                          "mean_R_subset_train": float(Ttr.ret[ctx.Sslot[ctx.tr]].mean()),
                          "mean_R_subset_sel": float(Tsel.ret[Ssel].mean()),
                          "mean_R_nonsubset_sel": float(Tsel.ret[Tsel.mask & ~Ssel].mean()), **pinfo}
            # 神託
            if cond not in oracle_cache:
                s6, ev = SY.oracle_stakes(cand, pi, ctx.S, ctx.lay, ctx.sel_rows, device=device)
                oracle_cache[cond] = (SY.oracle_slot_stakes(Tsel, cand, s6, ctx.lay, ctx.sel_rows), float(ev.mean()))
            ofrac, oev = oracle_cache[cond]
            orc = eval_stakes(ofrac, Tsel, ctx.universe_sel, Ssel, Wsel)
            orc["expected_log_growth"] = oev
            orc_f = eval_stakes(ofrac, Tfresh, ctx.universe_sel, Ssel, Wsel)
            R = {"synth": synth_desc, "oracle": orc, "oracle_fresh": orc_f}
            H = {}
            olg = orc["log_growth"] if cond != "control" else None
            olg_f = orc_f["log_growth"] if cond != "control" else None
            log(f"[seed {sd} {cond}] meanR subset sel={synth_desc['mean_R_subset_sel']:.3f} "
                f"oracle lg={orc['log_growth']:.6f} CI={orc['log_growth_ci']}")
            if "nn" in learners:
                best, grid, hists, sds = run_nn(L, Ttr, Tsel, device)
                fr_sel = [L.predict_nn(m, Tsel.X, Tsel.mask, device) for m, _, _ in sds]
                fr_tr = [L.predict_nn(m, Ttr.X, Ttr.mask, device) for m, _, _ in sds]
                fr_fr = [L.predict_nn(m, Tfresh.X, Tfresh.mask, device) for m, _, _ in sds]
                main_ = eval_stakes(np.mean(fr_sel, 0), Tsel, ctx.universe_sel, Ssel, Wsel, olg)
                R["nn"] = {"selected_cfg": best, "seed_best_epochs": [e for _, e, _ in sds],
                           "seed_sel_lg": [g for _, _, g in sds], "main": main_,
                           "per_seed": [eval_stakes(f, Tsel, ctx.universe_sel, Ssel, Wsel, olg) for f in fr_sel],
                           "fresh": eval_stakes(np.mean(fr_fr, 0), Tfresh, ctx.universe_sel, Ssel, Wsel, olg_f),
                           "train_insample": eval_stakes(np.mean(fr_tr, 0), Ttr, ctx.universe_tr, ctx.Sslot[ctx.tr], Wtr),
                           "grid": grid}
                H["nn"] = hists
                log(f"  NN main: betR={main_['n_bet_races']} lg={main_['log_growth']:.6f} CI={main_['log_growth_ci']} "
                    f"roi={main_['roi']} subset={main_['subset_stake_share']} ratio={main_.get('ratio_to_oracle')}")
            if "gbm" in learners:
                best, grid, cfg, models = run_gbm(L, Ttr, Tsel)
                st = lambda T: np.mean([L.gbm_stakes(L.gbm_pred(m, T), T.mask, cfg["thr"]) for m in models], 0)
                main_ = eval_stakes(st(Tsel), Tsel, ctx.universe_sel, Ssel, Wsel, olg)
                R["gbm"] = {"selected_cfg": best, "main": main_,
                            "fresh": eval_stakes(st(Tfresh), Tfresh, ctx.universe_sel, Ssel, Wsel, olg_f),
                            "train_insample": eval_stakes(st(Ttr), Ttr, ctx.universe_tr, ctx.Sslot[ctx.tr], Wtr),
                            "grid": grid}
                log(f"  GBM main: betR={main_['n_bet_races']} lg={main_['log_growth']:.6f} CI={main_['log_growth_ci']} "
                    f"roi={main_['roi']} subset={main_['subset_stake_share']} ratio={main_.get('ratio_to_oracle')}")
            results[str(sd)][cond] = R
            histories[str(sd)][cond] = H
            (OUT / f"results_{args.tag}.json").write_text(json.dumps(results, ensure_ascii=False, indent=1, default=float),
                                                          encoding="utf-8")
    out = {"tag": args.tag, "learner_module": args.learner_module, "seeds": seeds, "conditions": conds,
           "subset_share": float(ctx.S.mean()), "subset_edges": ctx.edges, "elapsed_sec": round(time.time() - t0, 1),
           "results": results}
    if set(conds) >= {"control", "e110", "e120"}:
        out["judgement"] = judge(results, seeds, learners)
        log("JUDGEMENT " + json.dumps({k: v["PASS"] for k, v in out["judgement"].items()}))
    (OUT / f"results_{args.tag}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float),
                                                  encoding="utf-8")
    (OUT / f"histories_{args.tag}.json").write_text(json.dumps(histories), encoding="utf-8")
    log(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
