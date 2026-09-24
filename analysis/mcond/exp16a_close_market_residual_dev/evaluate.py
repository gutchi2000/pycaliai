# -*- coding: utf-8 -*-
"""
evaluate.py — EXP16A Stage 1-3: 2019-2023 crossfit 評価 (Gate A / Gate B)
==========================================================================
spec v0.4 (凍結, commit e7603678) に従う。power Gate 通過後にだけ実行する。
  arms  : Q0 (terminal_close_market) / Q1 (rolling OOF) / Q2a / Q2b / Q3a / Q3b / placebo
  race set: 正式 (平地・DNF なし・勝馬一意・starter>=5)
  年 Y の係数・較正・セル境界は **Y-1 以前のみ**
  5 seed それぞれ独立に fit、主判定は median seed
  pooled は year-stratified meeting-day bootstrap
  Gate 判定は必ず gate_grade.grade_gate() を呼ぶ (手計算で上書きしない)
出力: out/eval_2019_2023.json
実行: python -m analysis.mcond.exp16a_close_market_residual_dev.evaluate
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .provenance import OUT
from .gate_grade import FLOOR, economic_checks_allowed, grade_gate
from .build_oof import RESEARCH, SEEDS, load_rows
from .stage0_dry_run import EXP15_OUT

EVAL_YEARS = [2019, 2020, 2021, 2022, 2023]
FIT_YEAR_MIN = 2016
BOOT = 3000
BOOT_SEED = 20260924
PLACEBO_DRAWS = 20
FIELD_BINS = [(5, 8), (9, 12), (13, 15), (16, 18)]
RESIDUAL = ["前走確定着順", "前走上り3F順", "kako5_avg_pos", "kako5_best_pos",
            "horse_fuku30", "jockey_fuku90", "prev_hosei", "間隔"]


# ---------------------------------------------------------------- utils
class Seg:
    def __init__(self, off):
        self.off = off
        self.start = off[:-1]
        self.cnt = np.diff(off)
        self.R = len(self.cnt)

    def sum(self, x):
        return np.add.reduceat(x, self.start)

    def spread(self, v):
        return np.repeat(v, self.cnt)


def zscore(x, sg: Seg):
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    xs = np.where(ok, x, 0.0)
    n = sg.sum(ok.astype(float))
    mu = np.where(n > 0, sg.sum(xs) / np.maximum(n, 1), 0.0)
    c = np.where(ok, xs - sg.spread(mu), 0.0)
    sd = np.sqrt(np.maximum(sg.sum(c * c) / np.maximum(n, 1), 0.0))
    return c / np.maximum(sg.spread(sd), 1e-9)


def clogit_fit(offset, X, sg: Seg, win_rows, x0=None):
    """offset + X·beta のレース内条件付きロジット最尤 (winner 行を与える)"""
    k = X.shape[1]
    xw = X[win_rows].sum(axis=0)

    def f(b):
        eta = offset + X @ b
        m = sg.spread(np.maximum.reduceat(eta, sg.start))
        e = np.exp(eta - m)
        Z = sg.sum(e)
        p = e / sg.spread(Z)
        nll = -(eta[win_rows].sum() - (np.log(Z) + np.maximum.reduceat(eta, sg.start)).sum())
        g = -(xw - np.array([(p * X[:, j]).sum() for j in range(k)]))
        return nll / sg.R, g / sg.R

    r = minimize(f, np.zeros(k) if x0 is None else x0, jac=True, method="L-BFGS-B",
                 options={"maxiter": 300, "ftol": 1e-12, "gtol": 1e-10})
    return r.x


def race_prob(offset, X, beta, sg: Seg):
    eta = offset + (X @ beta if X.shape[1] else 0.0)
    m = sg.spread(np.maximum.reduceat(eta, sg.start))
    e = np.exp(eta - m)
    return e / sg.spread(sg.sum(e))


def metrics(p, sg: Seg, win_rows, rel):
    """race-level 指標。改善は logloss が小さいこと"""
    ll = -np.log(np.clip(p[win_rows], 1e-12, None))
    # multiclass Brier (race ごと): Σ_i (p_i - y_i)^2
    y = np.zeros(len(p))
    y[win_rows] = 1.0
    br = sg.sum((p - y) ** 2)
    top1, top3, ndcg3, ndcg5 = [], [], [], []
    for r, (a, b) in enumerate(zip(sg.off[:-1], sg.off[1:])):
        pp = p[a:b]
        order = np.argsort(-pp)
        w = win_rows[r] - a
        top1.append(int(order[0] == w))
        top3.append(int(w in order[:3]))
        g = rel[a:b][order]
        for kk, acc in ((3, ndcg3), (5, ndcg5)):
            disc = 1.0 / np.log2(np.arange(2, min(kk, len(pp)) + 2))
            dcg = ((2 ** g[:kk] - 1) * disc).sum()
            ideal = np.sort(rel[a:b])[::-1][:kk]
            idcg = ((2 ** ideal - 1) * disc[:len(ideal)]).sum()
            acc.append(dcg / idcg if idcg > 0 else np.nan)
    return {"ll": ll, "brier": br, "top1": np.array(top1, dtype=float),
            "top3": np.array(top3, dtype=float),
            "ndcg3": np.array(ndcg3, dtype=float), "ndcg5": np.array(ndcg5, dtype=float)}


class DayBoot:
    """year-stratified meeting-day bootstrap"""

    def __init__(self, years, days, eval_years):
        self.g = []
        for y in eval_years:
            sel = years == y
            dd, inv = np.unique(days[sel], return_inverse=True)
            self.g.append((y, np.flatnonzero(sel), inv, len(dd)))

    def ci(self, val, rng, drop_year=None, boot=BOOT):
        sums, cnts = [], []
        for y, rows, inv, nd in self.g:
            if drop_year is not None and y == drop_year:
                continue
            ds = np.bincount(inv, weights=val[rows], minlength=nd)
            dc = np.bincount(inv, minlength=nd).astype(float)
            pick = rng.integers(0, nd, size=(boot, nd))
            sums.append(ds[pick].sum(1))
            cnts.append(dc[pick].sum(1))
        tot = np.sum(sums, axis=0) / np.sum(cnts, axis=0)
        return float(np.quantile(tot, 0.025)), float(np.quantile(tot, 0.975))


# ---------------------------------------------------------------- data
def load_all():
    d = np.load(RESEARCH / "official_races_all.npz", allow_pickle=True)
    q = np.load(RESEARCH / "q1_official.npz", allow_pickle=True)
    rid, year, day = d["rid16"].astype(str), d["year"], d["day"].astype(str)
    ban, off = d["ban"], d["offsets"]
    keys = np.array([f"{rid[i]}_{ban[j]}" for i in range(len(rid))
                     for j in range(off[i], off[i + 1])])
    assert list(q["key"].astype(str)) == list(keys), "q1_official.npz の並びが一致しない"
    contract = json.loads((EXP15_OUT / "feature_contract.json").read_text(encoding="utf-8"))
    df = load_rows(contract)
    kk = (df["rid16"] + "_" + df["ban"].astype(str)).to_numpy()
    pos = {k: i for i, k in enumerate(kk)}
    take = np.array([pos[k] for k in keys])
    jyun = df["jyun"].to_numpy()[take]
    feats = {c: pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)[take]
             for c in RESIDUAL}
    meta = {c: df[c].to_numpy()[take] for c in ["芝・ダ"]}
    return {"rid": rid, "year": year, "day": day, "ban": ban, "offsets": off,
            "winner_pos": d["winner_pos"], "pi_close": d["pi_close"], "pi_pre": d["pi_pre"],
            "prob": q["prob"], "seeds": q["seeds"].tolist(), "keys": keys,
            "jyun": jyun, "feats": feats, "surface": meta["芝・ダ"]}


def main():
    t0 = time.time()
    try:                                  # cp932 コンソールでも落ちないようにする
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    D = load_all()
    sg_all = Seg(D["offsets"])
    year_r = D["year"]
    win_rows_all = D["offsets"][:-1] + D["winner_pos"]        # race ごとの winner の global row
    rel_all = np.clip(6 - D["jyun"], 0, 5).astype(float)
    pop = json.loads((OUT / "race_population.json").read_text(encoding="utf-8"))
    bounds = pop["learned_subset_boundaries_by_eval_year"]
    rng = np.random.default_rng(BOOT_SEED)

    # 残差特徴のレース内 z (全レース一括で作る。レース内変換なので年をまたがない)
    Z = np.column_stack([zscore(D["feats"][c], sg_all) for c in RESIDUAL])
    logQ0 = np.log(np.clip(D["pi_close"], 1e-12, None))
    logPre = np.log(np.clip(D["pi_pre"], 1e-12, None))

    # race → 行スライス
    def rows_of_years(years):
        m = np.isin(year_r, years)
        r = np.flatnonzero(m)
        seg = Seg(np.concatenate([[0], np.cumsum(sg_all.cnt[r])]))
        hrows = np.concatenate([np.arange(D["offsets"][i], D["offsets"][i + 1]) for i in r])
        wr = seg.off[:-1] + (win_rows_all[r] - D["offsets"][r])
        return r, hrows, seg, wr

    arms = ["Q0", "Q1", "Q2a", "Q2b", "Q3a", "Q3b", "PRE"]
    per = {a: {} for a in arms}          # per[arm][(seed, year)] = metrics dict
    coefs = {}
    for si, seed in enumerate(D["seeds"]):
        logQ1_all = np.log(np.clip(D["prob"][si], 1e-12, None))
        for Y in EVAL_YEARS:
            fy = [y for y in range(FIT_YEAR_MIN, Y)]
            rf, hf, sgf, wf = rows_of_years(fy)
            re_, he, sge, we = rows_of_years([Y])
            # --- 係数 (Y-1 以前のみ)
            b2a = clogit_fit(logQ0[hf], logQ1_all[hf][:, None], sgf, wf)
            b2b = clogit_fit(logQ0[hf], np.column_stack([logQ1_all[hf], Z[hf]]), sgf, wf,
                             x0=np.concatenate([b2a, np.zeros(len(RESIDUAL))]))
            b3a = clogit_fit(logPre[hf], logQ1_all[hf][:, None], sgf, wf)
            b3b = clogit_fit(logPre[hf], np.column_stack([logQ1_all[hf], Z[hf]]), sgf, wf,
                             x0=np.concatenate([b3a, np.zeros(len(RESIDUAL))]))
            coefs[f"s{seed}_Y{Y}"] = {"Q2a": b2a.tolist(), "Q2b": b2b.tolist(),
                                       "Q3a": b3a.tolist(), "Q3b": b3b.tolist(),
                                       "fit_years": fy, "n_fit_races": int(sgf.R)}
            # --- 評価年の確率
            P = {
                "Q0": D["pi_close"][he],
                "PRE": D["pi_pre"][he],
                "Q1": D["prob"][si][he] / sge.spread(sge.sum(D["prob"][si][he])),
                "Q2a": race_prob(logQ0[he], logQ1_all[he][:, None], b2a, sge),
                "Q2b": race_prob(logQ0[he], np.column_stack([logQ1_all[he], Z[he]]), b2b, sge),
                "Q3a": race_prob(logPre[he], logQ1_all[he][:, None], b3a, sge),
                "Q3b": race_prob(logPre[he], np.column_stack([logQ1_all[he], Z[he]]), b3b, sge),
            }
            for a in arms:
                per[a][(seed, Y)] = metrics(P[a], sge, we, rel_all[he])
            per.setdefault("_rows", {})[(seed, Y)] = (re_, he)
        print(f"  seed {seed} done ({time.time()-t0:.0f}s)", flush=True)

    # ---------------- 集約
    def stack(arm, seed):
        """評価年をつなげた per-race 値 (race 単位)"""
        out = {k: [] for k in ["ll", "brier", "top1", "top3", "ndcg3", "ndcg5"]}
        yrs, days = [], []
        for Y in EVAL_YEARS:
            m = per[arm][(seed, Y)]
            for k in out:
                out[k].append(m[k])
            r = per["_rows"][(seed, Y)][0]
            yrs.append(year_r[r])
            days.append(D["day"][r])
        return ({k: np.concatenate(v) for k, v in out.items()},
                np.concatenate(yrs), np.concatenate(days))

    boot_years, boot_days = None, None
    res = {"role": "EXP16A Stage 1 の 2019-2023 crossfit 評価", "spec": "v0.4 (commit e7603678)",
           "created": time.strftime("%Y-%m-%d %H:%M:%S %z"),
           "eval_years": EVAL_YEARS, "seeds": D["seeds"],
           "race_counts": {str(Y): int((year_r == Y).sum()) for Y in EVAL_YEARS},
           "meeting_days": {str(Y): int(len(set(D["day"][year_r == Y].tolist())))
                            for Y in EVAL_YEARS},
           "coefficients": coefs, "arms": {}, "deltas": {}}

    M = {}
    for a in arms:
        M[a] = {}
        for seed in D["seeds"]:
            mm, yrs, days = stack(a, seed)
            M[a][seed] = mm
            boot_years, boot_days = yrs, days
        res["arms"][a] = {
            "per_seed": {str(s): {k: float(np.nanmean(M[a][s][k])) for k in M[a][s]}
                         for s in D["seeds"]},
            "median_ll": float(np.median([np.nanmean(M[a][s]["ll"]) for s in D["seeds"]])),
            "per_year_ll": {str(Y): float(np.median([per[a][(s, Y)]["ll"].mean()
                                                     for s in D["seeds"]]))
                            for Y in EVAL_YEARS},
        }
    boot = DayBoot(boot_years, boot_days, EVAL_YEARS)

    def delta_block(arm, base):
        """Δ = LL(arm) − LL(base) を per-race で作り、pooled/年別/seed別/LOO を集計"""
        per_seed_pooled, per_seed_year = {}, {}
        dvals = {}
        for s in D["seeds"]:
            dd = M[arm][s]["ll"] - M[base][s]["ll"]
            dvals[s] = dd
            per_seed_pooled[s] = float(dd.mean())
            per_seed_year[s] = {str(Y): float((per[arm][(s, Y)]["ll"] -
                                               per[base][(s, Y)]["ll"]).mean())
                                for Y in EVAL_YEARS}
        pooled_vals = np.array([per_seed_pooled[s] for s in D["seeds"]])
        k = int(np.argsort(pooled_vals)[len(D["seeds"]) // 2])
        med_seed = D["seeds"][k]
        dd = dvals[med_seed]
        lo, hi = boot.ci(dd, rng)
        loo = {}
        for Y in EVAL_YEARS:
            l2, h2 = boot.ci(dd, rng, drop_year=Y)
            loo[str(Y)] = {"dropped_year": Y, "mean": float(
                dd[boot_years != Y].mean()), "ci95": [l2, h2]}
        out = {
            "arm": arm, "baseline": base,
            "median_seed": int(med_seed),
            "pooled_point_estimate": float(dd.mean()),
            "ci95": [lo, hi],
            "point_estimate_beyond_floor": bool(dd.mean() < -FLOOR),
            "ci_entirely_beyond_floor": bool(hi < -FLOOR),
            "per_seed_pooled": {str(s): per_seed_pooled[s] for s in D["seeds"]},
            "seeds_improved": int(sum(v < 0 for v in per_seed_pooled.values())),
            "per_year_median_seed": per_seed_year[med_seed],
            "per_seed_per_year": {str(s): per_seed_year[s] for s in D["seeds"]},
            "years_improved_median_seed": int(sum(v < 0 for v in
                                                  per_seed_year[med_seed].values())),
            "leave_one_year_out": loo,
            "subsets": {"2019_2022_only": float(dd[boot_years != 2023].mean()),
                         "2023_only": float(dd[boot_years == 2023].mean())},
            "year_heterogeneity": {
                "sd_of_year_effects": float(np.std(list(per_seed_year[med_seed].values()), ddof=1)),
                "max_minus_min": float(max(per_seed_year[med_seed].values()) -
                                        min(per_seed_year[med_seed].values()))},
        }
        return out

    # Δ は各対について **一度だけ** 計算する (CI を二重に作らない)。等級は後で付ける
    for nm, (arm, base) in [("Q2a_vs_Q0", ("Q2a", "Q0")), ("Q2b_vs_Q2a", ("Q2b", "Q2a")),
                            ("Q3a_vs_PRE", ("Q3a", "PRE")), ("Q3b_vs_Q3a", ("Q3b", "Q3a")),
                            ("Q1_vs_Q0", ("Q1", "Q0")),
                            ("Q2b_vs_Q0_diagnostic", ("Q2b", "Q0")),
                            ("Q3b_vs_PRE_diagnostic", ("Q3b", "PRE"))]:
        res["deltas"][nm] = delta_block(arm, base)
    for nm in ("Q2b_vs_Q0_diagnostic", "Q3b_vs_PRE_diagnostic"):
        res["deltas"][nm]["status"] = ("診断値。spec v0.4 の decomposition は Q2a−Q0 と Q2b−Q2a "
                                        "(および Q3a−pre と Q3b−Q3a) であり、この合成は事前登録の "
                                        "Gate 比較ではない。等級判定には使わない")

    # ---------------- placebo (事前登録: セル内で馬単位に置換、20 draw)
    def cells_for_year(Y, rows_races, hrows):
        """芝ダ × 頭数帯 × pre favorite odds 三分位 (境界は Y-1 以前から)"""
        b = bounds[str(Y)]
        t1, t2 = b["favorite_pre_odds_tertiles"]
        cid = np.empty(len(hrows), dtype=object)
        k = 0
        for i in rows_races:
            a, bb = D["offsets"][i], D["offsets"][i + 1]
            n = bb - a
            fav = 1.0 / D["pi_pre"][a:bb].max()
            fb = 0 if fav <= t1 else (1 if fav <= t2 else 2)
            nb = next((j for j, (lo2, hi2) in enumerate(FIELD_BINS) if lo2 <= n <= hi2),
                      len(FIELD_BINS))
            sfc = str(D["surface"][a])
            for _ in range(n):
                cid[k] = f"{sfc}|{nb}|{fb}"
                k += 1
        return cid

    def run_placebo(kind: str, seed, draws=PLACEBO_DRAWS):
        """kind='gateA' は log(Q1) を、'q3b' は残差特徴を、セル内で置換する"""
        si = D["seeds"].index(seed)
        logQ1_all = np.log(np.clip(D["prob"][si], 1e-12, None))
        rr = np.random.default_rng(20260925)
        vals = []
        for dr in range(draws):
            per_year = []
            for Y in EVAL_YEARS:
                fy = list(range(FIT_YEAR_MIN, Y))
                rf, hf, sgf, wf = rows_of_years(fy)
                re_, he, sge, we = rows_of_years([Y])
                # セル内置換 (fit 年と評価年の両方を同じ規則で置換する)
                def permute(vec_rows, rows_races, hrows):
                    cid = cells_for_year(Y, rows_races, hrows)
                    out = vec_rows.copy()
                    for c in np.unique(cid):
                        m = np.flatnonzero(cid == c)
                        out[m] = vec_rows[m][rr.permutation(len(m))]
                    return out
                if kind == "gateA":
                    q_f = permute(logQ1_all[hf], rf, hf)
                    q_e = permute(logQ1_all[he], re_, he)
                    bb = clogit_fit(logQ0[hf], q_f[:, None], sgf, wf)
                    p = race_prob(logQ0[he], q_e[:, None], bb, sge)
                    base = D["pi_close"][he]
                else:
                    Zf = np.column_stack([permute(Z[hf][:, j], rf, hf) for j in range(Z.shape[1])])
                    Ze = np.column_stack([permute(Z[he][:, j], re_, he) for j in range(Z.shape[1])])
                    bb = clogit_fit(logPre[hf], np.column_stack([logQ1_all[hf], Zf]), sgf, wf)
                    p = race_prob(logPre[he], np.column_stack([logQ1_all[he], Ze]), bb, sge)
                    base = race_prob(logPre[he], logQ1_all[he][:, None],
                                     np.array(coefs[f"s{seed}_Y{Y}"]["Q3a"]), sge)
                ll = -np.log(np.clip(p[we], 1e-12, None))
                ll0 = -np.log(np.clip(base[we], 1e-12, None))
                per_year.append(ll - ll0)
            vals.append(float(np.concatenate(per_year).mean()))
        return vals

    med_a = res["deltas"]["Q2a_vs_Q0"]["median_seed"]
    pl_a = run_placebo("gateA", med_a)
    real_a = res["deltas"]["Q2a_vs_Q0"]["pooled_point_estimate"]
    # 改善は負なので「placebo の 2.5 パーセンタイルより小さい」= 97.5 パーセンタイルを超える
    thr_a = float(np.quantile(pl_a, 0.025))
    placebo_a = {"kind": "gateA (log Q1 をセル内置換)", "draws": PLACEBO_DRAWS,
                 "median_seed": int(med_a), "placebo_deltas": pl_a,
                 "placebo_2.5pct": thr_a, "real_delta": real_a,
                 "real_exceeds_placebo_97.5pct": bool(real_a < thr_a)}
    res["placebo_gateA"] = placebo_a

    q3b = res["deltas"]["Q3b_vs_Q3a"]
    if q3b["pooled_point_estimate"] < 0:
        med_b = q3b["median_seed"]
        pl_b = run_placebo("q3b", med_b)
        thr_b = float(np.quantile(pl_b, 0.025))
        res["placebo_q3b"] = {"kind": "q3b (残差特徴をセル内置換)", "draws": PLACEBO_DRAWS,
                              "median_seed": int(med_b), "placebo_deltas": pl_b,
                              "placebo_2.5pct": thr_b,
                              "real_delta": q3b["pooled_point_estimate"],
                              "real_exceeds_placebo_97.5pct":
                                  bool(q3b["pooled_point_estimate"] < thr_b),
                              "trigger": "Q3b が改善を示したため事前登録の必須検証として実行"}
    else:
        res["placebo_q3b"] = {"skipped": "Q3b は改善を示していない (pooled >= 0)",
                              "real_delta": q3b["pooled_point_estimate"]}

    # ---------------- Gate 判定 (grade_gate のみ。Δ は上で計算した値をそのまま使う)
    def attach(name, gate, placebo=None):
        b = res["deltas"][name]
        g = grade_gate(gate, point=b["pooled_point_estimate"], ci_lower=b["ci95"][0],
                       ci_upper=b["ci95"][1], years_improved=b["years_improved_median_seed"],
                       n_years=len(EVAL_YEARS), seeds_improved=b["seeds_improved"],
                       n_seeds=len(D["seeds"]),
                       loo_ci_uppers=[v["ci95"][1] for v in b["leave_one_year_out"].values()],
                       placebo_exceeded=placebo)
        return {**b, "grade": g}

    res["gate_A"] = attach("Q2a_vs_Q0", "A", placebo_a["real_exceeds_placebo_97.5pct"])
    ga = res["gate_A"]["grade"]["grade"]
    # 凍結 spec: Gate B の run_if は「Gate A が PASS-PRACTICAL または PASS-SIGNAL」
    gate_b_allowed = ga in ("PASS-PRACTICAL", "PASS-SIGNAL")
    res["gate_B"] = attach("Q3a_vs_PRE", "B")
    res["gate_B"]["protocol_status"] = (
        "in_protocol (Gate A が PASS のため実行条件を満たす)" if gate_b_allowed else
        "not_applicable: Gate A が FAIL なので Gate B は事前登録の実行条件 (run_if) を満たさない。"
        "下の数値は診断として残すが、**Gate の等級としては扱わない**")
    if not gate_b_allowed:
        res["gate_B"]["computed_grade_for_reference_only"] = res["gate_B"]["grade"]["grade"]
        res["gate_B"]["grade"]["grade"] = "NOT_APPLICABLE"
        res["gate_B"]["grade"]["statement"] = (
            "Gate A が FAIL のため Gate B は事前登録の実行条件を満たさない。"
            "Q3a 対 historical_pre_snapshot の数値は診断として報告する")
        res["gate_B"]["grade"]["consequences"] = [
            "economic checks は実施しない", "候補生成・ROI 評価・配分最適化へ進まない",
            "Gate B の等級を Gate A の結論の代わりに使わない"]
    gb = res["gate_B"]["grade"]["grade"]
    gap = res["arms"]["PRE"]["median_ll"] - res["arms"]["Q0"]["median_ll"]
    rec = ((res["arms"]["PRE"]["median_ll"] - res["arms"]["Q3a"]["median_ll"]) / gap
           if gap >= 0.01 else None)
    res["recovery_ratio"] = {"definition": "(LL(pre) − LL(Q3a)) / (LL(pre) − LL(Q0))",
                             "pre_minus_close_gap_nats": gap,
                             "value": rec,
                             "floor": 0.10,
                             "guard": None if rec is not None else
                             "gap too small (< 0.01 nats) のため比を報告しない"}
    res["economic_checks_allowed"] = economic_checks_allowed(ga, gb)
    res["summary"] = {"gate_A": ga, "gate_B": gb,
                      "gate_B_protocol_status": res["gate_B"]["protocol_status"],
                      "gate_A_statement": res["gate_A"]["grade"]["statement"],
                      "gate_B_statement": res["gate_B"]["grade"]["statement"],
                      "economic_checks_allowed": res["economic_checks_allowed"]}
    res["elapsed_sec"] = round(time.time() - t0, 1)
    (OUT / "eval_2019_2023.json").write_text(json.dumps(res, ensure_ascii=False, indent=1,
                                                        default=float), encoding="utf-8")
    print(json.dumps(res["summary"], ensure_ascii=False))
    for k in ["Q2a_vs_Q0", "Q2b_vs_Q2a", "Q3a_vs_PRE", "Q3b_vs_Q3a", "Q1_vs_Q0"]:
        v = res["deltas"][k]
        print(f"  {k}: {v['pooled_point_estimate']:+.6f} CI[{v['ci95'][0]:+.6f},"
              f"{v['ci95'][1]:+.6f}] years={v['years_improved_median_seed']}/5 "
              f"seeds={v['seeds_improved']}/5")
    print(f"[saved] out/eval_2019_2023.json ({res['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
