# -*- coding: utf-8 -*-
"""
pre_market_size_tertiles.py — EXP16A 副腕 pre_market_size_tertiles の評価 (一発実行)
====================================================================================
spec v0.4 (凍結 e7603678) の `subsets.learned_boundaries_per_eval_year.items` に登録済みの
`pre_market_size_tertiles` を開ける。spec に無い判定規則は結果開封前に
docs/research/EXP16A_PRE_MARKET_SIZE_TERTILES_RULES_20261022.md に固定した (sha256 を記録)。

主判定: 各分位で Δ_k = LL(Q2a) − LL(Q0)。Q2a は EXP16A 本体と同一係数 (再 fit しない)。
等級は gate_grade.grade_gate("A") のみ。placebo は本体と同一構成を分位に切り出す。
記述のみ: Q2b−Q2a / Q3a−PRE / Q3b−Q3a / Q1−Q0 / PRE−Q0 / T1−T3 差 / 分位内再 fit 版 Q2a。

2024/2025 は読まない (official_races_all.npz は 2016-2023 のみ、assert)。
再実行禁止: 出力 JSON が既にあれば停止する。
出力: out/pre_market_size_tertiles.json
実行: python -m analysis.mcond.exp16a_close_market_residual_dev.pre_market_size_tertiles
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time

import numpy as np

from .provenance import OUT, BASE
from .gate_grade import FLOOR, grade_gate
from .evaluate import (Seg, zscore, clogit_fit, race_prob, load_all, RESIDUAL,
                       EVAL_YEARS, FIT_YEAR_MIN, BOOT, BOOT_SEED, PLACEBO_DRAWS, FIELD_BINS)
from .pre_market_size_labelfree import pre_votes_by_race, boundaries_by_year, assign_tertile

OUT_FILE = OUT / "pre_market_size_tertiles.json"
RULES = BASE / "docs" / "research" / "EXP16A_PRE_MARKET_SIZE_TERTILES_RULES_20261022.md"
NAMES = ["T1_small", "T2_mid", "T3_large"]
N_TESTS = 3
ALPHA = 0.05


def boot_dist(val, years, days, rng, drop_year=None, boot=BOOT):
    """年層化 meeting-day bootstrap の分布 (evaluate.DayBoot.ci と同じ算法、分布を返す)"""
    sums, cnts = [], []
    for y in EVAL_YEARS:
        if drop_year is not None and y == drop_year:
            continue
        sel = years == y
        if not sel.any():
            continue
        dd, inv = np.unique(days[sel], return_inverse=True)
        nd = len(dd)
        ds = np.bincount(inv, weights=val[sel], minlength=nd)
        dc = np.bincount(inv, minlength=nd).astype(float)
        pick = rng.integers(0, nd, size=(boot, nd))
        sums.append(ds[pick].sum(1))
        cnts.append(dc[pick].sum(1))
    return np.sum(sums, axis=0) / np.sum(cnts, axis=0)


def main():
    t0 = time.time()
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    assert not OUT_FILE.exists(), f"{OUT_FILE} が既にある。再実行は禁止 (補足規則 §6)"
    assert RULES.exists(), "補足規則ファイルが無い (結果開封前に固定すること)"
    rules_sha = hashlib.sha256(RULES.read_bytes()).hexdigest()
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=BASE, capture_output=True,
                          text=True).stdout.strip()

    D = load_all()
    assert D["year"].max() <= 2023 and D["year"].min() >= 2016, "封印期間が混入"
    V = pre_votes_by_race()
    assert list(V["rid"]) == list(D["rid"]), "票数と正式 race の並びが一致しない"
    bnd = boundaries_by_year(V)
    tq = assign_tertile(V, bnd)
    votes = V["votes"]

    sg_all = Seg(D["offsets"])
    year_r = D["year"]
    off = D["offsets"]
    nR = sg_all.R
    win_rows_all = off[:-1] + D["winner_pos"]
    Z = np.column_stack([zscore(D["feats"][c], sg_all) for c in RESIDUAL])
    logQ0 = np.log(np.clip(D["pi_close"], 1e-12, None))
    logPre = np.log(np.clip(D["pi_pre"], 1e-12, None))
    pop = json.loads((OUT / "race_population.json").read_text(encoding="utf-8"))
    fav_bounds = pop["learned_subset_boundaries_by_eval_year"]

    def rows_of(race_idx):
        r = np.asarray(race_idx)
        seg = Seg(np.concatenate([[0], np.cumsum(sg_all.cnt[r])]))
        hrows = np.concatenate([np.arange(off[i], off[i + 1]) for i in r])
        wr = seg.off[:-1] + (win_rows_all[r] - off[r])
        return r, hrows, seg, wr

    def races_of_years(years):
        return np.flatnonzero(np.isin(year_r, years))

    def tertile_under(Y, race_idx):
        t1, t2 = bnd[Y]
        x = votes[race_idx]
        return np.where(x <= t1, 0, np.where(x <= t2, 1, 2))

    arms = ["Q0", "PRE", "Q1", "Q2a", "Q2b", "Q3a", "Q3b", "Q2a_refit"]
    LL = {a: {} for a in arms}
    for si, seed in enumerate(D["seeds"]):
        logQ1_all = np.log(np.clip(D["prob"][si], 1e-12, None))
        full = {a: np.full(nR, np.nan) for a in arms}
        for Y in EVAL_YEARS:
            fy_r = races_of_years(list(range(FIT_YEAR_MIN, Y)))
            ey_r = races_of_years([Y])
            rf, hf, sgf, wf = rows_of(fy_r)
            re_, he, sge, we = rows_of(ey_r)
            b2a = clogit_fit(logQ0[hf], logQ1_all[hf][:, None], sgf, wf)
            b2b = clogit_fit(logQ0[hf], np.column_stack([logQ1_all[hf], Z[hf]]), sgf, wf,
                             x0=np.concatenate([b2a, np.zeros(len(RESIDUAL))]))
            b3a = clogit_fit(logPre[hf], logQ1_all[hf][:, None], sgf, wf)
            b3b = clogit_fit(logPre[hf], np.column_stack([logQ1_all[hf], Z[hf]]), sgf, wf,
                             x0=np.concatenate([b3a, np.zeros(len(RESIDUAL))]))
            P = {
                "Q0": D["pi_close"][he],
                "PRE": D["pi_pre"][he],
                "Q1": D["prob"][si][he] / sge.spread(sge.sum(D["prob"][si][he])),
                "Q2a": race_prob(logQ0[he], logQ1_all[he][:, None], b2a, sge),
                "Q2b": race_prob(logQ0[he], np.column_stack([logQ1_all[he], Z[he]]), b2b, sge),
                "Q3a": race_prob(logPre[he], logQ1_all[he][:, None], b3a, sge),
                "Q3b": race_prob(logPre[he], np.column_stack([logQ1_all[he], Z[he]]), b3b, sge),
            }
            for a, p in P.items():
                full[a][re_] = -np.log(np.clip(p[we], 1e-12, None))
            # 記述のみ: 分位内再 fit 版 Q2a (境界は評価年 Y のもの、fit は 2016..Y-1 の同分位)
            tf = tertile_under(Y, fy_r)
            te = tq[ey_r]
            for k in range(3):
                rfk, hfk, sgfk, wfk = rows_of(fy_r[tf == k])
                rek, hek, sgek, wek = rows_of(ey_r[te == k])
                bk = clogit_fit(logQ0[hfk], logQ1_all[hfk][:, None], sgfk, wfk)
                pk = race_prob(logQ0[hek], logQ1_all[hek][:, None], bk, sgek)
                full["Q2a_refit"][rek] = -np.log(np.clip(pk[wek], 1e-12, None))
        for a in arms:
            LL[a][seed] = full[a]
        print(f"  seed {seed} done ({time.time()-t0:.0f}s)", flush=True)

    ev = np.isin(year_r, EVAL_YEARS)
    assert (tq[ev] >= 0).all()
    # ---- 本体の再現 assert (全体の seed 別 pooled Δ(Q2a−Q0))
    ref = json.loads((OUT / "eval_2019_2023.json").read_text(encoding="utf-8"))
    ref_ps = ref["deltas"]["Q2a_vs_Q0"]["per_seed_pooled"]
    repro = {}
    for s in D["seeds"]:
        v = float((LL["Q2a"][s][ev] - LL["Q0"][s][ev]).mean())
        repro[str(s)] = {"this": v, "eval_2019_2023": ref_ps[str(s)], "abs_diff": abs(v - ref_ps[str(s)])}
        assert abs(v - ref_ps[str(s)]) < 1e-9, f"本体再現に失敗 seed {s}: {v} vs {ref_ps[str(s)]}"
    print("  本体再現 OK", flush=True)

    rng = np.random.default_rng(BOOT_SEED)
    yrs_all, days_all = year_r, D["day"]
    bonf_q = ALPHA / N_TESTS / 2

    def delta_sub(arm, base, mask):
        dvals = {s: LL[arm][s] - LL[base][s] for s in D["seeds"]}
        ps = {s: float(dvals[s][mask].mean()) for s in D["seeds"]}
        pv = np.array([ps[s] for s in D["seeds"]])
        med = D["seeds"][int(np.argsort(pv)[len(D["seeds"]) // 2])]
        dd = dvals[med][mask]
        yy, dy = yrs_all[mask], days_all[mask]
        dist = boot_dist(dd, yy, dy, rng)
        per_year = {str(Y): float(dd[yy == Y].mean()) for Y in EVAL_YEARS}
        per_seed_year = {str(s): {str(Y): float(dvals[s][mask & (year_r == Y)].mean())
                                  for Y in EVAL_YEARS} for s in D["seeds"]}
        loo = {}
        for Y in EVAL_YEARS:
            dl = boot_dist(dd, yy, dy, rng, drop_year=Y)
            loo[str(Y)] = {"mean": float(dd[yy != Y].mean()),
                           "ci95": [float(np.quantile(dl, 0.025)), float(np.quantile(dl, 0.975))]}
        return {
            "arm": arm, "baseline": base, "median_seed": int(med),
            "n_races": int(mask.sum()), "n_meeting_days": int(len(set(dy.tolist()))),
            "pooled_point_estimate": float(dd.mean()),
            "ci95": [float(np.quantile(dist, 0.025)), float(np.quantile(dist, 0.975))],
            "ci_bonferroni_3": [float(np.quantile(dist, bonf_q)), float(np.quantile(dist, 1 - bonf_q))],
            "point_estimate_beyond_floor": bool(dd.mean() < -FLOOR),
            "per_seed_pooled": {str(s): ps[s] for s in D["seeds"]},
            "seeds_improved": int(sum(v < 0 for v in ps.values())),
            "per_year_median_seed": per_year,
            "per_seed_per_year": per_seed_year,
            "years_improved_median_seed": int(sum(v < 0 for v in per_year.values())),
            "races_by_year": {str(Y): int((yy == Y).sum()) for Y in EVAL_YEARS},
            "leave_one_year_out": loo,
        }

    # ---- 主判定の Δ (分位ごと)
    masks = {NAMES[k]: ev & (tq == k) for k in range(3)}
    primary = {nm: delta_sub("Q2a", "Q0", m) for nm, m in masks.items()}

    # ---- placebo (本体と同一構成、per-race Δ を返して分位へ切り出す)
    def cells_for_year(Y, race_idx):
        t1, t2 = fav_bounds[str(Y)]["favorite_pre_odds_tertiles"]
        cid = []
        for i in race_idx:
            a, bb = off[i], off[i + 1]
            n = bb - a
            fav = 1.0 / D["pi_pre"][a:bb].max()
            fb = 0 if fav <= t1 else (1 if fav <= t2 else 2)
            nb = next((j for j, (lo2, hi2) in enumerate(FIELD_BINS) if lo2 <= n <= hi2),
                      len(FIELD_BINS))
            sfc = str(D["surface"][a])
            cid.extend([f"{sfc}|{nb}|{fb}"] * n)
        return np.array(cid, dtype=object)

    def placebo_per_race(seed, draws=PLACEBO_DRAWS):
        si = D["seeds"].index(seed)
        logQ1_all = np.log(np.clip(D["prob"][si], 1e-12, None))
        rr = np.random.default_rng(20260925)
        out = []
        for _ in range(draws):
            full = np.full(nR, np.nan)
            for Y in EVAL_YEARS:
                fy_r = races_of_years(list(range(FIT_YEAR_MIN, Y)))
                ey_r = races_of_years([Y])
                rf, hf, sgf, wf = rows_of(fy_r)
                re_, he, sge, we = rows_of(ey_r)

                def permute(vec_rows, race_idx):
                    cid = cells_for_year(Y, race_idx)
                    o = vec_rows.copy()
                    for c in np.unique(cid):
                        m = np.flatnonzero(cid == c)
                        o[m] = vec_rows[m][rr.permutation(len(m))]
                    return o
                q_f = permute(logQ1_all[hf], rf)
                q_e = permute(logQ1_all[he], re_)
                bb = clogit_fit(logQ0[hf], q_f[:, None], sgf, wf)
                p = race_prob(logQ0[he], q_e[:, None], bb, sge)
                ll = -np.log(np.clip(p[we], 1e-12, None))
                ll0 = -np.log(np.clip(D["pi_close"][he][we], 1e-12, None))
                full[re_] = ll - ll0
            out.append(full)
        return out

    overall_med = int(ref["deltas"]["Q2a_vs_Q0"]["median_seed"])
    need = sorted({primary[nm]["median_seed"] for nm in NAMES} | {overall_med})
    pl_cache = {}
    for s in need:
        pl_cache[s] = placebo_per_race(s)
        print(f"  placebo seed {s} done ({time.time()-t0:.0f}s)", flush=True)
    # 本体 placebo の再現 assert
    ref_pl = ref["placebo_gateA"]["placebo_deltas"]
    this_pl = [float(v[ev].mean()) for v in pl_cache[overall_med]]
    pl_err = max(abs(a - b) for a, b in zip(this_pl, ref_pl))
    assert pl_err < 1e-9, f"本体 placebo の再現に失敗 {pl_err}"
    print("  placebo 本体再現 OK", flush=True)

    results = {}
    for k, nm in enumerate(NAMES):
        b = primary[nm]
        m = masks[nm]
        pl = [float(v[m].mean()) for v in pl_cache[b["median_seed"]]]
        thr = float(np.quantile(pl, 0.025))
        exceeded = bool(b["pooled_point_estimate"] < thr)
        g = grade_gate("A", point=b["pooled_point_estimate"], ci_lower=b["ci95"][0],
                       ci_upper=b["ci95"][1], years_improved=b["years_improved_median_seed"],
                       n_years=len(EVAL_YEARS), seeds_improved=b["seeds_improved"],
                       n_seeds=len(D["seeds"]),
                       loo_ci_uppers=[v["ci95"][1] for v in b["leave_one_year_out"].values()],
                       placebo_exceeded=exceeded)
        results[nm] = {**b, "placebo": {"draws": PLACEBO_DRAWS, "seed": b["median_seed"],
                                        "placebo_deltas": pl, "placebo_2.5pct": thr,
                                        "real_exceeds_placebo_97.5pct": exceeded},
                       "grade": g}

    # ---- 副腕の結論 (補足規則 §3)
    verdict = "CLOSED"
    why = []
    for nm in NAMES:
        g = results[nm]["grade"]["grade"]
        bu = results[nm]["ci_bonferroni_3"][1]
        if g == "PASS-PRACTICAL" and bu < -FLOOR:
            verdict = "OPEN-PRACTICAL"
            why.append(nm)
    if verdict == "CLOSED":
        for nm in NAMES:
            g = results[nm]["grade"]["grade"]
            bu = results[nm]["ci_bonferroni_3"][1]
            if g in ("PASS-PRACTICAL", "PASS-SIGNAL") and bu < 0:
                verdict = "OPEN-SIGNAL"
                why.append(nm)

    # ---- 記述のみ
    desc = {}
    for nm, (arm, base) in [("Q2b_vs_Q2a", ("Q2b", "Q2a")), ("Q3a_vs_PRE", ("Q3a", "PRE")),
                            ("Q3b_vs_Q3a", ("Q3b", "Q3a")), ("Q1_vs_Q0", ("Q1", "Q0")),
                            ("PRE_vs_Q0_gap", ("PRE", "Q0")),
                            ("Q2a_refit_within_tertile_vs_Q0", ("Q2a_refit", "Q0"))]:
        desc[nm] = {}
        for tn in NAMES:
            d = delta_sub(arm, base, masks[tn])
            desc[nm][tn] = {k: d[k] for k in ["median_seed", "n_races", "pooled_point_estimate",
                                              "ci95", "ci_bonferroni_3", "seeds_improved",
                                              "years_improved_median_seed",
                                              "per_year_median_seed"]}
    # T1 − T3 (Q2a − Q0、全体 median seed、同じ day 再標本で両分位を同時に)
    dd = LL["Q2a"][overall_med] - LL["Q0"][overall_med]
    m1, m3 = masks["T1_small"], masks["T3_large"]
    s1, c1, s3, c3 = [], [], [], []
    for Y in EVAL_YEARS:
        sel = ev & (year_r == Y)
        udays, inv = np.unique(days_all[sel], return_inverse=True)
        nd = len(udays)
        w1 = m1[sel].astype(float)
        w3 = m3[sel].astype(float)
        a1 = np.bincount(inv, weights=dd[sel] * w1, minlength=nd)
        n1 = np.bincount(inv, weights=w1, minlength=nd)
        a3 = np.bincount(inv, weights=dd[sel] * w3, minlength=nd)
        n3 = np.bincount(inv, weights=w3, minlength=nd)
        pick = rng.integers(0, nd, size=(BOOT, nd))
        s1.append(a1[pick].sum(1)); c1.append(n1[pick].sum(1))
        s3.append(a3[pick].sum(1)); c3.append(n3[pick].sum(1))
    diff = np.sum(s1, 0) / np.sum(c1, 0) - np.sum(s3, 0) / np.sum(c3, 0)
    desc["T1_minus_T3_Q2a_vs_Q0"] = {
        "seed": overall_med,
        "point": float(dd[m1].mean() - dd[m3].mean()),
        "ci95": [float(np.quantile(diff, 0.025)), float(np.quantile(diff, 0.975))]}

    res = {
        "role": "EXP16A 副腕 pre_market_size_tertiles の評価 (一発実行)",
        "spec": "v0.4 (commit e7603678)",
        "supplementary_rules": str(RULES.relative_to(BASE)).replace("\\", "/"),
        "supplementary_rules_sha256": rules_sha,
        "git_head_at_run": head,
        "created": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "eval_years": EVAL_YEARS, "seeds": D["seeds"],
        "boundaries_sha256": hashlib.sha256(json.dumps({str(k): v for k, v in bnd.items()},
                                                       sort_keys=True).encode()).hexdigest(),
        "boundaries_raw": "公開物に出さない",
        "reproduction_checks": {"overall_Q2a_vs_Q0_per_seed": repro,
                                "overall_placebo_gateA_max_abs_err": pl_err},
        "primary_Q2a_vs_Q0_by_tertile": results,
        "verdict": {"result": verdict, "tertiles_supporting": why,
                    "rule": "補足規則 §3: OPEN-PRACTICAL = PASS-PRACTICAL かつ Bonferroni 上限 < -0.005 / "
                            "OPEN-SIGNAL = PASS-SIGNAL 以上かつ Bonferroni 上限 < 0 / それ以外 CLOSED",
                    "gate_A_overall_unchanged": "FAIL (spec subsets.rule により分位結果で変わらない)"},
        "descriptive_only": desc,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    OUT_FILE.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float),
                        encoding="utf-8")
    for nm in NAMES:
        r = results[nm]
        print(f"  {nm}: n={r['n_races']} Δ={r['pooled_point_estimate']:+.6f} "
              f"CI[{r['ci95'][0]:+.6f},{r['ci95'][1]:+.6f}] "
              f"Bonf[{r['ci_bonferroni_3'][0]:+.6f},{r['ci_bonferroni_3'][1]:+.6f}] "
              f"years={r['years_improved_median_seed']}/5 seeds={r['seeds_improved']}/5 "
              f"placebo={r['placebo']['real_exceeds_placebo_97.5pct']} "
              f"grade={r['grade']['grade']}")
    print(f"  verdict: {verdict} {why}")
    print(f"[saved] {OUT_FILE} ({res['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
