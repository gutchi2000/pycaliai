# -*- coding: utf-8 -*-
"""
train_nn.py — R1 (DeepSets) / R1-noctx / R1-noctx-2x の学習
===========================================================
--stage grid : selection_seed=7 で 8 格子点 × {R1, R1-noctx} を学習し、2022 win logloss 平均最小を選択
--stage seeds: 選択点で main 5 seed × {R1, R1-noctx, R1-noctx-2x}。予測 (2022/2023) と容量指標を保存
--stage post : 学習後テスト T1 (学習済み・2023 実レース)、T6 (2023 Σp=1)、T15 (再現性)
2023 は学習・ES・較正・選択に使わない。
出力: data/_research/mcond/exp15/{scores,models}/, out/nn_grid.json, out/nn_seeds.json, out/tests_post.json
"""
from __future__ import annotations

import argparse
import itertools
import json

import numpy as np
import torch

from . import common as C
from . import nnlib as N

SCORES = C.CACHE / "scores"
MODELS = C.CACHE / "models"


def setup():
    N.set_determinism()
    df = C.load_rows()
    con = C.load_contract()
    enc = C.Encoded(df, con, "clean")
    T = {k: N.RaceTensors(enc, df, k) for k in ("train", "sel", "dev")}
    return df, enc, T


def flat(S, Tk, n_rows):
    out = np.full(n_rows, np.nan)
    v = Tk.rows >= 0
    out[Tk.rows[v]] = S[v]
    return out


def run(kind, enc, T, cfg, seed, log=print):
    torch.manual_seed(seed)
    model = N.build_model(kind, enc.card, enc.num_nn.shape[1], cfg["W"], cfg["dropout"])
    info = N.train(model, T["train"], T["sel"], seed, cfg["lr"],
                   max_epochs=C.SPEC["models"]["NN_common"]["max_epochs"],
                   patience=C.SPEC["models"]["NN_common"]["early_stopping"]["patience"],
                   bs=C.SPEC["models"]["NN_common"]["batch_races"], log=None)
    Ssel = N.predict(model, T["sel"])
    tau, ll = N.tau_ll(Ssel, T["sel"])
    info.update({"tau": tau, "sel_ll": ll, "params": N.n_params(model),
                 "flops_per_race_n14": N.flops_per_race(model, enc.card, enc.num_nn.shape[1])})
    info.pop("hist_full", None)
    return model, info


def stage_grid():
    df, enc, T = setup()
    g = C.SPEC["models"]["NN_common"]["grid"]
    seed = C.SPEC["seeds"]["selection_seed"]
    res = []
    for W, p, lr in itertools.product(g["W"], g["dropout"], g["lr"]):
        cfg = {"W": W, "dropout": p, "lr": lr}
        row = {"cfg": cfg}
        for kind in ("R1", "R1-noctx"):
            _, info = run(kind, enc, T, cfg, seed)
            row[kind] = {k: info[k] for k in ("sel_ll", "tau", "es_epoch", "params", "epoch_seconds")}
            print(f"[grid] {cfg} {kind:9s} sel_ll={info['sel_ll']:.5f} es={info['es_epoch']} "
                  f"params={info['params']} ({info['epoch_seconds']:.1f}s/ep)", flush=True)
        row["mean_sel_ll"] = (row["R1"]["sel_ll"] + row["R1-noctx"]["sel_ll"]) / 2
        res.append(row)
    best = min(res, key=lambda r: r["mean_sel_ll"])
    C.dump({"grid": res, "selected": best["cfg"], "rule": C.SPEC["models"]["NN_common"]["selection"]},
           "nn_grid.json")
    print("[select]", best["cfg"], flush=True)


def stage_seeds():
    df, enc, T = setup()
    cfg = json.loads((C.OUT / "nn_grid.json").read_text(encoding="utf-8"))["selected"]
    SCORES.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    out = {"cfg": cfg, "runs": {}}
    for seed in C.SPEC["seeds"]["main"]:
        for kind in ("R1", "R1-noctx", "R1-noctx-2x"):
            model, info = run(kind, enc, T, cfg, seed)
            s = np.full(len(df), np.nan)
            for k in ("sel", "dev"):
                s = np.where(np.isnan(s), flat(N.predict(model, T[k]), T[k], len(df)), s)
            ev = ~np.isnan(s)
            np.savez(SCORES / f"{kind}_s{seed}.npz", rows=np.where(ev)[0], score=s[ev],
                     key=(df["rid16"] + "_" + df["ban"].astype(str)).to_numpy()[ev],
                     info=json.dumps({k: v for k, v in info.items() if k != "hist"}))
            torch.save(model.state_dict(), MODELS / f"{kind}_s{seed}.pt")
            out["runs"][f"{kind}_s{seed}"] = {k: v for k, v in info.items() if k != "hist"}
            out["runs"][f"{kind}_s{seed}"]["hist"] = info["hist"]
            print(f"[{kind:11s} s{seed}] sel_ll={info['sel_ll']:.5f} es={info['es_epoch']} "
                  f"params={info['params']} {info['epoch_seconds']:.1f}s/ep peak={info['peak_gpu_MB']:.0f}MB",
                  flush=True)
            C.dump(out, "nn_seeds.json")


@torch.no_grad()
def t1_trained(model, Tk, reps=3, seed=11):
    model.eval()
    g = torch.Generator().manual_seed(seed)
    worst = 0.0
    base = N.predict(model, Tk)
    for _ in range(reps):
        R = len(Tk)
        cat2, num2 = Tk.cat.clone(), Tk.num.clone()
        perms = []
        for r in range(R):
            n = int(Tk.mask[r].sum())
            pi = torch.randperm(n, generator=g)
            perms.append(pi.numpy())
            cat2[r, :n] = Tk.cat[r, pi.to(N.DEV)]
            num2[r, :n] = Tk.num[r, pi.to(N.DEV)]
        out = np.full(base.shape, np.nan)
        for a in range(0, R, 1024):
            sl = slice(a, a + 1024)
            out[sl] = model(cat2[sl], num2[sl], Tk.mask[sl]).double().cpu().numpy()
        for r, pi in enumerate(perms):
            n = len(pi)
            worst = max(worst, float(np.abs(out[r, :n] - base[r, pi]).max()))
    return worst


def stage_post():
    df, enc, T = setup()
    cfg = json.loads((C.OUT / "nn_grid.json").read_text(encoding="utf-8"))["selected"]
    res = {}
    for kind in ("R1", "R1-noctx"):
        worst = 0.0
        for seed in C.SPEC["seeds"]["main"]:
            m = N.build_model(kind, enc.card, enc.num_nn.shape[1], cfg["W"], cfg["dropout"]).to(N.DEV)
            m.load_state_dict(torch.load(MODELS / f"{kind}_s{seed}.pt", map_location=N.DEV))
            worst = max(worst, t1_trained(m, T["dev"]))
        # float32 の加算順序差のみ許容 (spec: 1e-5)
        res[f"T1_trained_{kind}_2023"] = {"pass": worst <= 1e-5, "detail": f"max|diff|={worst:.2e}"}
        print(res[f"T1_trained_{kind}_2023"], flush=True)
    # T6: 2023 Σp_win=1 (evaluate と同じ softmax)
    dev = (df["period"] == "dev").to_numpy()
    races = C.race_index(df, dev)
    worst = 0.0
    for seed in C.SPEC["seeds"]["main"]:
        z = np.load(SCORES / f"R1_s{seed}.npz")
        s = np.full(len(df), np.nan)
        s[z["rows"]] = z["score"]
        tau = json.loads(str(z["info"]))["tau"]
        p = C.softmax_race(s, races, tau)
        worst = max(worst, max(abs(p[i].sum() - 1) for i in races))
    res["T6_sum_to_one_2023"] = {"pass": worst < 1e-6, "detail": f"max|Σp-1|={worst:.2e}"}
    # T15: 同一 seed・同一設定で2回学習 → 2022 予測一致
    seed = C.SPEC["seeds"]["main"][0]
    preds = []
    for _ in range(2):
        m, _ = run("R1", enc, T, cfg, seed)
        preds.append(N.predict(m, T["sel"]))
    d = float(np.nanmax(np.abs(preds[0] - preds[1])))
    z = np.load(SCORES / f"R1_s{seed}.npz")
    s0 = np.full(len(df), np.nan)
    s0[z["rows"]] = z["score"]
    d0 = float(np.nanmax(np.abs(flat(preds[0], T["sel"], len(df)) - s0)[(df["period"] == "sel").to_numpy()]))
    res["T15_reproducible"] = {"pass": d <= 1e-6 and d0 <= 1e-6,
                               "detail": f"rerun_diff={d:.2e} vs_saved={d0:.2e}"}
    for k, v in res.items():
        print(k, v, flush=True)
    C.dump(res, "tests_post.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["grid", "seeds", "post"])
    a = ap.parse_args()
    {"grid": stage_grid, "seeds": stage_seeds, "post": stage_post}[a.stage]()
