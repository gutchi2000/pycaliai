# -*- coding: utf-8 -*-
"""
placebo.py — C1 (構成) / C2 (個体間関係) placebo。Context Gate PASS 時のみ実行 (spec.gates)。
=============================================================================================
自馬経路は真の x_i、文脈経路の相手集合だけを置換する。相手は horse vector 一まとまり。
race 内一定列 (out/feature_contract.json race_constant) は相手側も自レース値に上書き。
相手は同一区分 (train/sel/dev) 内から取り、学習・ES・評価すべて同規則で置換 (全体再学習)。
  P3 race-context exchange (C1): 頭数+場所+芝ダ+クラス群 一致の別レース r' (−1頭)
  P2 composition-preserving (C2): P3 + 能力三分位 + 先行頭数区分 一致の別レース r' (−1頭)
  P4 coherence break (C2): 相手を1頭ずつ同セル (芝ダ×クラス群 × 先行フラグ×能力十分位) の他レース実在馬へ
B=39/placebo、draw b は seeds[b mod 5]、乱数 seed 1000+b。判定: median_s G_real > q97.5(G_b)。
出力: out/placebo_{P2,P3,P4}.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch

from . import common as C
from . import nnlib as N
from .train_nn import setup

SCORES = C.CACHE / "scores"


def race_attrs(df, T, con):
    cg_map = con["class_group_map"]
    out = {}
    for per, Tk in T.items():
        first = Tk.rows[:, 0]
        n = (Tk.rows >= 0).sum(1)
        abil = pd.to_numeric(df["kako5_avg_pos"], errors="coerce").to_numpy()
        ppr = pd.to_numeric(df["prev_pos_rel"], errors="coerce").to_numpy()
        am = np.array([np.nanmean(abil[r[r >= 0]]) if np.isfinite(abil[r[r >= 0]]).any() else np.nan
                       for r in Tk.rows])
        nf = np.array([int(np.nansum(ppr[r[r >= 0]] <= 0.25)) for r in Tk.rows])
        q = np.nanquantile(am, [1 / 3, 2 / 3])
        terc = np.where(np.isnan(am), -1, np.digitize(am, q))
        out[per] = pd.DataFrame({
            "n": n, "venue": df["場所"].astype(str).to_numpy()[first],
            "surf": df["芝・ダ"].astype(str).to_numpy()[first],
            "cg": pd.Series(df["クラス名"].astype(str).to_numpy()[first]).map(cg_map).fillna("OP以上").to_numpy(),
            "terc": terc, "fbin": np.digitize(nf, [2, 4]),
        })
    return out


def pick_partner(A: pd.DataFrame, levels, rng):
    """各レースに別レース partner を割当。levels = [(keys, n_tol), ...]。戻り: partner idx, level"""
    R = len(A)
    partner = np.full(R, -1)
    level = np.full(R, -1)
    for L, (keys, tol) in enumerate(levels):
        todo = np.where(partner < 0)[0]
        if not len(todo):
            break
        grp = A.groupby(keys).indices if keys else {(): np.arange(R)}
        key_of = (A[keys].apply(tuple, axis=1) if keys else pd.Series([()] * R))
        by_n = {}
        for k, idx in grp.items():
            by_n[k if isinstance(k, tuple) else (k,)] = idx
        for r in todo:
            k = key_of.iloc[r]
            k = k if isinstance(k, tuple) else (k,)
            cand = by_n.get(k, np.array([], int))
            if tol > 0:
                cand = cand[np.abs(A["n"].to_numpy()[cand] - A["n"].iat[r]) <= tol]
            else:
                cand = cand[A["n"].to_numpy()[cand] == A["n"].iat[r]]
            cand = cand[cand != r]
            if len(cand):
                partner[r] = rng.choice(cand)
                level[r] = L
    return partner, level


def horse_cells(df, T, A):
    """P4 用: 行ごとのセル (区分, 芝ダ, クラス群, 先行フラグ, 能力十分位)"""
    cells = {}
    abil = pd.to_numeric(df["kako5_avg_pos"], errors="coerce").to_numpy()
    ppr = pd.to_numeric(df["prev_pos_rel"], errors="coerce").to_numpy()
    for per, Tk in T.items():
        rows = Tk.rows[Tk.rows >= 0]
        dq = np.nanquantile(abil[rows], np.linspace(0.1, 0.9, 9))
        dec = np.where(np.isnan(abil), -1, np.digitize(abil, dq))
        ff = np.where(np.isnan(ppr), -1, (ppr <= 0.25).astype(int))
        rr = np.repeat(np.arange(len(Tk)), (Tk.rows >= 0).sum(1))
        cells[per] = pd.DataFrame({"row": rows, "race": rr, "surf": A[per]["surf"].to_numpy()[rr],
                                   "cg": A[per]["cg"].to_numpy()[rr], "ff": ff[rows], "dec": dec[rows]})
    return cells


def build_ctx(kind, enc, T, A, cells, con, rng):
    """区分ごとの文脈テンソル dict(cat, num, mask, excl) と置換水準の統計"""
    nums = list(enc.nums)
    miss_cols = list(enc.fit_stats["miss_cols"])
    rc = set(con["race_constant"])
    rc_cat = [j for j, c in enumerate(enc.cats) if c in rc]
    rc_z = [j for j, c in enumerate(nums) if c in rc]
    rc_miss = [len(nums) + miss_cols.index(j) for j in rc_z if j in miss_cols]
    ctx, stats = {}, {}
    for per, Tk in T.items():
        R = len(Tk)
        own_first = Tk.rows[:, 0]
        if kind in ("P2", "P3"):
            if kind == "P3":
                levels = [(["n", "venue", "surf", "cg"], 0), (["surf", "cg"], 0),
                          (["surf", "cg"], 1), (["surf", "cg"], 2)]
            else:
                levels = [(["n", "venue", "surf", "cg", "terc", "fbin"], 0),
                          (["surf", "cg", "terc", "fbin"], 0), (["surf", "cg", "fbin"], 0),
                          (["surf", "cg", "fbin"], 1)]
            partner, level = pick_partner(A[per], levels, rng)
            real = partner < 0
            partner = np.where(real, np.arange(R), partner)
            crow = Tk.rows[partner]                                        # [R,18]
            cn = (crow >= 0).sum(1)
            excl = np.full((R, N.HMAX), -1, dtype=np.int64)
            for r in range(R):
                n_own = int((Tk.rows[r] >= 0).sum())
                if real[r]:
                    excl[r, :n_own] = np.arange(n_own)
                else:
                    excl[r, :n_own] = rng.integers(0, cn[r], n_own)
            stats[per] = {"level_counts": {str(k): int(v) for k, v in
                                           zip(*np.unique(level, return_counts=True))},
                          "no_partner_kept_real": int(real.sum())}
        else:  # P4
            cl = cells[per]
            key = ["surf", "cg", "ff", "dec"]
            grp = cl.groupby(key).indices
            key2 = ["ff", "dec"]
            grp2 = cl.groupby(key2).indices
            sub = np.empty(len(cl), dtype=np.int64)
            fb = 0
            kv = cl[key].to_numpy()
            kv2 = cl[key2].to_numpy()
            race = cl["race"].to_numpy()
            rows = cl["row"].to_numpy()
            for i in range(len(cl)):
                cand = grp[tuple(kv[i])]
                cand = cand[race[cand] != race[i]]
                if not len(cand):
                    cand = grp2[tuple(kv2[i])]
                    cand = cand[race[cand] != race[i]]
                    fb += 1
                sub[i] = rows[rng.choice(cand)]
            crow = np.full_like(Tk.rows, -1)
            v = Tk.rows >= 0
            crow[v] = sub                                                   # 行順 = own 行順
            excl = np.where(v, np.arange(N.HMAX)[None, :], -1).astype(np.int64)
            stats[per] = {"fallback_cell": fb, "n_horses": int(len(cl))}
        cmask = crow >= 0
        safe = np.where(cmask, crow, 0)
        cat = enc.cat_codes[safe].copy()
        num = enc.num_nn[safe].copy()
        # race 内一定列は自レース値に上書き
        cat[:, :, rc_cat] = enc.cat_codes[own_first][:, None, rc_cat]
        cat[~cmask] = 0
        zc = rc_z + rc_miss
        num[:, :, zc] = enc.num_nn[own_first][:, None, zc]
        num[~cmask] = 0.0
        ctx[per] = {"cat": torch.tensor(cat, device=N.DEV), "num": torch.tensor(num, device=N.DEV),
                    "mask": torch.tensor(cmask, device=N.DEV), "excl": torch.tensor(excl, device=N.DEV)}
    return ctx, stats


def sub_ctx(c, idx):
    return {k: v[idx] for k, v in c.items()}


def dev_metrics(df, S, Tdev, tau):
    races = [r[r >= 0] for r in Tdev.rows]
    s = np.full(len(df), np.nan)
    v = Tdev.rows >= 0
    s[Tdev.rows[v]] = S[v]
    races5 = [i for i in races if len(i) >= C.SPEC["population"]["eval_min_field"]]
    p = C.softmax_race(s, races5, tau)
    t = C.race_metrics(df, races5, s, p)
    return float(t["ll"].mean()), float(t["brier"].mean())


def main(kind, B):
    df, enc, T = setup()
    con = C.load_contract()
    cfg = json.loads((C.OUT / "nn_grid.json").read_text(encoding="utf-8"))["selected"]
    seeds = C.SPEC["seeds"]["main"]
    A = race_attrs(df, T, con)
    cells = horse_cells(df, T, A) if kind == "P4" else None
    # 実文脈の per-seed 効果 (R1-noctx − R1)
    base = {}
    for m in ("R1", "R1-noctx"):
        for s in seeds:
            z = np.load(SCORES / f"{m}_s{s}.npz")
            sc = np.full(len(df), np.nan)
            sc[z["rows"]] = z["score"]
            S = np.where(T["dev"].rows >= 0, sc[np.where(T["dev"].rows >= 0, T["dev"].rows, 0)], np.nan)
            base[(m, s)] = dev_metrics(df, S, T["dev"], json.loads(str(z["info"]))["tau"])
    g_real = [base[("R1-noctx", s)][0] - base[("R1", s)][0] for s in seeds]
    g_real_b = [base[("R1-noctx", s)][1] - base[("R1", s)][1] for s in seeds]
    out = {"kind": kind, "B": B, "cfg": cfg, "G_real_ll_per_seed": g_real, "G_real_brier_per_seed": g_real_b,
           "draws": []}
    for b in range(B):
        rng = np.random.default_rng(1000 + b)
        seed = seeds[b % len(seeds)]
        ctx, stats = build_ctx(kind, enc, T, A, cells, con, rng)
        torch.manual_seed(seed)
        model = N.build_model("R1", enc.card, enc.num_nn.shape[1], cfg["W"], cfg["dropout"])
        N.train(model, T["train"], T["sel"], seed, cfg["lr"],
                max_epochs=C.SPEC["models"]["NN_common"]["max_epochs"],
                patience=C.SPEC["models"]["NN_common"]["early_stopping"]["patience"],
                bs=C.SPEC["models"]["NN_common"]["batch_races"],
                ctx_tr=lambda idx: sub_ctx(ctx["train"], idx),
                ctx_sel=lambda sl: sub_ctx(ctx["sel"], sl))
        Ssel = N.predict(model, T["sel"], ctx_fn=lambda sl: sub_ctx(ctx["sel"], sl))
        tau, _ = N.tau_ll(Ssel, T["sel"])
        Sdev = N.predict(model, T["dev"], ctx_fn=lambda sl: sub_ctx(ctx["dev"], sl))
        ll, br = dev_metrics(df, Sdev, T["dev"], tau)
        g = base[("R1-noctx", seed)][0] - ll
        gb = base[("R1-noctx", seed)][1] - br
        out["draws"].append({"b": b, "seed": seed, "ll": ll, "brier": br, "G_ll": g, "G_brier": gb,
                             "stats": stats})
        print(f"[{kind} b{b:02d} s{seed}] ll={ll:.5f} G={g:+.5f}", flush=True)
        C.dump(out, f"placebo_{kind}.json")
    G = np.array([d["G_ll"] for d in out["draws"]])
    Gb = np.array([d["G_brier"] for d in out["draws"]])
    out["decision"] = {"median_G_real_ll": float(np.median(g_real)), "q975_placebo_ll": float(np.quantile(G, 0.975)),
                       "PASS_ll": bool(np.median(g_real) > np.quantile(G, 0.975)),
                       "median_G_real_brier": float(np.median(g_real_b)),
                       "q975_placebo_brier": float(np.quantile(Gb, 0.975)),
                       "rank_real_among_placebo_ll": int((G < np.median(g_real)).sum())}
    C.dump(out, f"placebo_{kind}.json")
    print(out["decision"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["P2", "P3", "P4"])
    ap.add_argument("--B", type=int, default=C.SPEC["placebos"]["B_per_placebo"])
    a = ap.parse_args()
    main(a.kind, a.B)
