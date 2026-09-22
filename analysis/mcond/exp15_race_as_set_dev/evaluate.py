# -*- coding: utf-8 -*-
"""
evaluate.py — 2023 fixed-model development 評価と Gate 判定 (spec.gates)
=======================================================================
主推定: per-race Δ_r = mean_s[L_r(A_s) - L_r(B_s)] (seed 対)。meeting-day bootstrap 3000。
Context Gate = R1 vs R1-noctx。Replacement Gate (Stage1 参考) = R1 vs R0-clean。
出力: out/gate_results.json, out/results_tables.md
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import common as C

SCORES = C.CACHE / "scores"
MODELS = ["R0-clean", "R0-prodref", "R1-noctx", "R1-noctx-2x", "R1"]
MET = ["ll", "brier", "ndcg3", "ndcg5", "hon_top3"]
LOWER_BETTER = {"ll": True, "brier": True, "ndcg3": False, "ndcg5": False, "hon_top3": False}
SPEC_NAME = {"ll": "race_win_logloss", "brier": "race_win_brier", "ndcg3": "ndcg@3",
             "hon_top3": "hon_top3"}


def load_scores(df, model, seed):
    z = np.load(SCORES / f"{model}_s{seed}.npz", allow_pickle=True)
    key = (df["rid16"] + "_" + df["ban"].astype(str)).to_numpy()
    assert np.array_equal(key[z["rows"]], z["key"]), f"{model} s{seed} 行対応不一致"
    s = np.full(len(df), np.nan)
    s[z["rows"]] = z["score"]
    return s, json.loads(str(z["info"]))


def build(df, races, seeds):
    tabs, P = {}, {}
    for m in MODELS:
        for sd in seeds:
            f = SCORES / f"{m}_s{sd}.npz"
            if not f.exists():
                continue
            s, info = load_scores(df, m, sd)
            p = C.softmax_race(s, races, info["tau"])
            tabs[(m, sd)] = C.race_metrics(df, races, s, p)
            P[(m, sd)] = p
    return tabs, P


def compare(tabs, a, b, seeds, day, metrics=MET):
    out = {}
    for k in metrics:
        D = np.vstack([tabs[(a, s)][k].to_numpy(float) - tabs[(b, s)][k].to_numpy(float) for s in seeds])
        d = D.mean(0)
        bd = C.boot_delta(d, day)
        per_seed = np.nanmean(D, axis=1)
        good = per_seed < 0 if LOWER_BETTER[k] else per_seed > 0
        bd.update({"per_seed": per_seed.tolist(), "n_seeds_improve": int(good.sum()),
                   "seed_median": float(np.median(per_seed)), "seed_sd": float(np.std(per_seed, ddof=1)),
                   "seed_min": float(per_seed.min()), "seed_max": float(per_seed.max())})
        out[k] = bd
    return out, None


def strata_delta(tabs, a, b, seeds, day, key: np.ndarray, groups: dict):
    D = np.vstack([tabs[(a, s)]["ll"].to_numpy(float) - tabs[(b, s)]["ll"].to_numpy(float)
                   for s in seeds]).mean(0)
    res = {}
    for name, m in groups.items():
        sel = m(key)
        res[name] = C.boot_delta(np.where(sel, D, np.nan), day) if sel.any() else None
    return res


def ece_compare(P, a, b, seeds, y, day_row, reps=1000, seed=20260923):
    days = np.unique(day_row)
    pos = {d: np.where(day_row == d)[0] for d in days}
    pa = [P[(a, s)] for s in seeds]
    pb = [P[(b, s)] for s in seeds]
    base = float(np.mean([C.ece10(x, y) - C.ece10(z, y) for x, z in zip(pa, pb)]))
    rng = np.random.default_rng(seed)
    bs = np.empty(reps)
    for k in range(reps):
        ii = np.concatenate([pos[d] for d in rng.choice(days, len(days))])
        bs[k] = np.mean([C.ece10(x[ii], y[ii]) - C.ece10(z[ii], y[ii]) for x, z in zip(pa, pb)])
    return {"delta": base, "ci_lo": float(np.quantile(bs, 0.025)), "ci_hi": float(np.quantile(bs, 0.975)),
            "ece_a": float(np.mean([C.ece10(x, y) for x in pa])),
            "ece_b": float(np.mean([C.ece10(z, y) for z in pb]))}


def longshot(P, models, seeds, y, rows, day_row, ref="R0-clean"):
    pref = np.mean([P[(ref, s)] for s in seeds], axis=0)[rows]
    bands = [(0, 0.02), (0.02, 0.05), (0.05, 0.10)]
    res = {}
    for m in models:
        if (m, seeds[0]) not in P:
            continue
        pm = np.mean([P[(m, s)] for s in seeds], axis=0)[rows]
        for lo, hi in bands:
            sel = (pref >= lo) & (pref < hi)
            own = (pm >= lo) & (pm < hi)
            res[f"{m}|{lo}-{hi}|ref_band"] = {"n": int(sel.sum()), "AE": float(y[sel].sum() / pm[sel].sum())}
            res[f"{m}|{lo}-{hi}|own_band"] = {"n": int(own.sum()), "AE": float(y[own].sum() / pm[own].sum())}
    return res


def seed_table(tabs, seeds):
    rows = []
    for m in MODELS:
        vals = {k: [] for k in MET}
        for s in seeds:
            if (m, s) not in tabs:
                continue
            t = tabs[(m, s)]
            for k in MET:
                vals[k].append(float(t[k].mean()))
        if not vals["ll"]:
            continue
        for k in MET:
            v = np.array(vals[k])
            rows.append({"model": m, "metric": k, "per_seed": v.tolist(), "median": float(np.median(v)),
                         "min": float(v.min()), "max": float(v.max()), "sd": float(v.std(ddof=1))})
    return rows


def main():
    df = C.load_rows()
    seeds = C.SPEC["seeds"]["main"]
    dev = (df["period"] == "dev").to_numpy()
    races = [i for i in C.race_index(df, dev) if len(i) >= C.SPEC["population"]["eval_min_field"]]
    tabs, P = build(df, races, seeds)
    t0 = tabs[("R1", seeds[0])]
    day = t0["meeting_day"].to_numpy()
    n = t0["n"].to_numpy()
    first = t0["row0"].to_numpy()
    surf = df["芝・ダ"].astype(str).to_numpy()[first]
    venue = df["場所"].astype(str).to_numpy()[first]
    rows = np.concatenate(races)
    y = df["win"].to_numpy()[rows]
    day_row = df["meeting_day"].to_numpy()[rows]
    Prow = {k: v[rows] for k, v in P.items()}

    mde = json.loads((C.OUT / "mde.json").read_text(encoding="utf-8"))["metrics"]
    tests_pre = json.loads((C.OUT / "tests_pre.json").read_text(encoding="utf-8"))
    tests_post = json.loads((C.OUT / "tests_post.json").read_text(encoding="utf-8"))

    fs_groups = {f"{a}-{b}": (lambda x, a=a, b=b: (n >= a) & (n <= b))
                 for a, b in C.SPEC["metrics"]["strata"]["field_size"]}
    res = {"n_races": len(races), "n_days": int(len(np.unique(day))),
           "deadheat_excluded": int((~t0["single_win"]).sum()), "seeds": seeds}

    pairs = {"context": ("R1", "R1-noctx"), "diag_R1_vs_2x": ("R1", "R1-noctx-2x"),
             "diag_2x_vs_noctx": ("R1-noctx-2x", "R1-noctx"), "replacement_R1_vs_R0": ("R1", "R0-clean"),
             "ref_noctx_vs_R0": ("R1-noctx", "R0-clean"), "ref_prodref_vs_R0": ("R0-prodref", "R0-clean")}
    for name, (a, b) in pairs.items():
        cmp, _ = compare(tabs, a, b, seeds, day)
        res[name] = {"A": a, "B": b, "metrics": cmp,
                     "field_size": strata_delta(tabs, a, b, seeds, day, n, fs_groups),
                     "surface": strata_delta(tabs, a, b, seeds, day, surf,
                                             {"芝": lambda x: x == "芝", "ダ": lambda x: x == "ダ"}),
                     "venue": strata_delta(tabs, a, b, seeds, day, venue,
                                           {v: (lambda x, v=v: x == v) for v in sorted(set(venue))})}
        if name in ("context", "replacement_R1_vs_R0"):
            res[name]["ece10"] = ece_compare({k: v for k, v in Prow.items()}, a, b, seeds, y, day_row)
    res["seed_table"] = seed_table(tabs, seeds)
    res["longshot_AE"] = longshot(Prow, MODELS, seeds, y, np.arange(len(rows)), day_row)

    # ---------------- Context Gate
    cm = res["context"]["metrics"]
    thr = {k: mde[SPEC_NAME[k]]["context_threshold_min_MDE_practical"] for k in ("ll", "brier")}
    fs_bad = [k for k, v in res["context"]["field_size"].items() if v and v["ci_lo"] > 0]
    t1_ok = all(v["pass"] for k, v in tests_post.items() if k.startswith("T1_trained"))
    g0_ok = all(v["pass"] for v in tests_pre.values()) and all(v["pass"] for v in tests_post.values())
    cond = {
        "1_both_improve": cm["ll"]["delta"] < 0 and cm["brier"]["delta"] < 0,
        "2_ci_upper_lt0": cm["ll"]["ci_hi"] < 0 and cm["brier"]["ci_hi"] < 0,
        "3_seeds_ge4_both": int(sum((np.array(cm["ll"]["per_seed"]) < 0) & (np.array(cm["brier"]["per_seed"]) < 0))) >= 4,
        "4_magnitude_ge_min_MDE_practical": (-cm["ll"]["delta"] >= thr["ll"]) and (-cm["brier"]["delta"] >= thr["brier"]),
        "5_no_fatal_field_size": not fs_bad,
        "6_permutation_invariance_trained": t1_ok,
    }
    ctx_pass = all(cond.values()) and g0_ok
    taxonomy = {}
    for k in ("ll", "brier", "ndcg3", "hon_top3"):
        m = mde[SPEC_NAME[k]]
        v = cm[k]
        taxonomy[k] = {"statistically_nonsignificant": bool(v["ci_lo"] <= 0 <= v["ci_hi"]),
                       "practically_small": bool(abs(v["delta"]) < m["practical_context"]),
                       "underpowered": bool(m["MDE_primary"] > m["practical_context"])}
    res["context_gate"] = {"conditions": cond, "G0_all_tests_pass": g0_ok, "PASS": ctx_pass,
                           "thresholds": thr, "fatal_field_bins": fs_bad, "fail_taxonomy": taxonomy,
                           "placebo_C1_C2": "run_only_if_PASS"}

    # ---------------- Replacement Gate (Stage1 参考: R1 vs R0-clean)
    rm = res["replacement_R1_vs_R0"]["metrics"]
    rthr = {k: mde[SPEC_NAME[k]]["replacement_threshold_max_MDE_practical"] for k in ("ll", "brier")}
    ece = res["replacement_R1_vs_R0"]["ece10"]
    rcond = {
        "1_both_improve_ci": rm["ll"]["delta"] < 0 and rm["brier"]["delta"] < 0
        and rm["ll"]["ci_hi"] < 0 and rm["brier"]["ci_hi"] < 0,
        "2_seeds_ge4_both": int(sum((np.array(rm["ll"]["per_seed"]) < 0) & (np.array(rm["brier"]["per_seed"]) < 0))) >= 4,
        "3_ndcg3_or_top3_not_sig_worse": (rm["ndcg3"]["ci_hi"] >= 0) or (rm["hon_top3"]["ci_hi"] >= 0),
        "4_ece_not_sig_worse": not (ece["ci_lo"] > 0),
        "5_magnitude_ge_max_MDE_practical": (-rm["ll"]["delta"] >= rthr["ll"]) and (-rm["brier"]["delta"] >= rthr["brier"]),
    }
    res["replacement_gate_stage1_reference"] = {"conditions": rcond, "PASS": all(rcond.values()),
                                                "thresholds": rthr}
    C.dump(res, "gate_results.json")

    # ---------------- 表
    L = ["# EXP15 Stage 1 結果表 (2023 fixed-model development, 自動生成)", "",
         f"races={res['n_races']} meeting_days={res['n_days']} deadheat_excluded={res['deadheat_excluded']}", ""]
    L += ["## seed 別 (各 seed の 2023 平均)", "", "| model | metric | median | min | max | SD | per seed |",
          "|---|---|---|---|---|---|---|"]
    for r in res["seed_table"]:
        L.append(f"| {r['model']} | {r['metric']} | {r['median']:.5f} | {r['min']:.5f} | {r['max']:.5f} | "
                 f"{r['sd']:.5f} | {' / '.join(f'{x:.5f}' for x in r['per_seed'])} |")
    for name in pairs:
        L += ["", f"## {name}: {res[name]['A']} − {res[name]['B']}", "",
              "| metric | Δ (seed対平均) | CI95 | 改善seed数 | seed median | seed SD |", "|---|---|---|---|---|---|"]
        for k, v in res[name]["metrics"].items():
            L.append(f"| {k} | {v['delta']:+.5f} | [{v['ci_lo']:+.5f}, {v['ci_hi']:+.5f}] | {v['n_seeds_improve']}/5 | "
                     f"{v['seed_median']:+.5f} | {v['seed_sd']:.5f} |")
        L.append("")
        L.append("出走頭数別 ΔLL: " + ", ".join(
            f"{k}: {v['delta']:+.4f} [{v['ci_lo']:+.4f},{v['ci_hi']:+.4f}] (n={v['n_races']})"
            for k, v in res[name]["field_size"].items() if v))
        L.append("")
        L.append("芝ダ別 ΔLL: " + ", ".join(
            f"{k}: {v['delta']:+.4f} [{v['ci_lo']:+.4f},{v['ci_hi']:+.4f}]" for k, v in res[name]["surface"].items() if v))
        vv = res[name]["venue"]
        L.append("")
        L.append(f"競馬場別 ΔLL<0 の場数: {sum(1 for v in vv.values() if v and v['delta'] < 0)}/{len(vv)}")
    L += ["", "## Context Gate", "", "```", json.dumps(res["context_gate"], ensure_ascii=False, indent=1), "```",
          "", "## Replacement Gate (Stage1 参考)", "", "```",
          json.dumps(res["replacement_gate_stage1_reference"], ensure_ascii=False, indent=1), "```"]
    (C.OUT / "results_tables.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[-60:]))


if __name__ == "__main__":
    main()
