# -*- coding: utf-8 -*-
"""
M-2: ROI 推定量の分散-バイアス frontier と券種序列の τ、a95 の裾依存 (記述と SE のみ)。

事前登録: docs/research/M2_ROI_ESTIMATOR_FRONTIER_PREREG_20261022.md
実行:     python -m analysis.roi_estimator_frontier.run [--reps 2000]
出力:     analysis/roi_estimator_frontier/out/results.json, summary.txt

母集団は data/_policy/bet_substrate.pkl (2023-25)。2026 年のデータ・a95 前向き台帳は読まない。
凍結方策 data/shadow_policies/a95_equal_v1.json は読むだけ (sha256 を記録)。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import kendalltau

BASE = Path(__file__).resolve().parents[2]
SUB = BASE / "data/_policy/bet_substrate.pkl"
POL = BASE / "data/shadow_policies/a95_equal_v1.json"
OUT = Path(__file__).resolve().parent / "out"

KINDS = ["tan", "fuku", "umaren", "umatan", "wide", "wakuren", "sanpuku", "sanrentan"]
JP = dict(tan="単勝", fuku="複勝", umaren="馬連", umatan="馬単", wide="ワイド",
          wakuren="枠連", sanpuku="三連複", sanrentan="三連単")
WIN_Q = {"win_q99": 0.99, "win_q95": 0.95, "win_q90": 0.90, "win_q75": 0.75}
LOG_F = {"log_f001": 0.001, "log_f01": 0.01}
ESTS = ["plain", *WIN_Q, *LOG_F]
ROTATIONS = [("2023", ("2024", "2025")),          # 一次
             ("2024", ("2023", "2025")),
             ("2025", ("2023", "2024"))]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ---------------------------------------------------------------- substrate
def load():
    races = joblib.load(SUB)
    rows = []
    for r in races:
        p, ban, o = r["p"], r["ban"], r["odds9"]
        if len(p) < 5 or not np.isfinite(o).any():
            continue
        order = np.argsort(-p)
        i1, i2, i3 = order[0], order[1], order[2]
        o1 = float(o[i1]) if np.isfinite(o[i1]) and o[i1] > 0 else np.nan
        if not np.isfinite(o1):
            continue
        a1, a2, a3 = int(ban[i1]), int(ban[i2]), int(ban[i3])
        w1, w2 = int(r["waku"][i1]), int(r["waku"][i2])
        f, s, t = r["first"], r["second"], r["third"]
        wide_map = {frozenset((x, y)): v for x, y, v in (r.get("wide") or ())}
        pay = dict(
            tan=(r["tan_pay"] if a1 == f else 0.0),
            fuku=float(r["fuku_pay"].get(a1, 0.0)),
            umaren=(r["umaren"] if {a1, a2} == {f, s} and np.isfinite(r["umaren"]) else 0.0),
            umatan=(r["umatan"] if (a1 == f and a2 == s) and np.isfinite(r["umatan"]) else 0.0),
            wide=float(wide_map.get(frozenset((a1, a2)), 0.0) or 0.0),
            wakuren=(r["wakuren"] if np.isfinite(r["wakuren"]) and {w1, w2} == {int(r["waku_first"]), int(r["waku_second"])} else 0.0),
            sanpuku=(r["sanpuku"] if {a1, a2, a3} == {f, s, t} and np.isfinite(r["sanpuku"]) else 0.0),
            sanrentan=(r["sanrentan"] if (a1, a2, a3) == (f, s, t) and np.isfinite(r["sanrentan"]) else 0.0),
        )
        rows.append((r["date"], int(r["n"]), o1, [pay[k] / 100.0 for k in KINDS]))
    date = np.array([x[0] for x in rows])
    n = np.array([x[1] for x in rows])
    o1 = np.array([x[2] for x in rows])
    R = np.array([x[3] for x in rows], float)               # (races, 8) 回収倍率
    valid = np.ones_like(R, bool)
    valid[:, KINDS.index("wakuren")] = n >= 9                # 枠連は 9 頭以上のみ発売
    year = np.array([d[:4] for d in date])
    return date, year, n, o1, R, valid


# ---------------------------------------------------------- weighted tools
def wquantile_pos(x, w, q):
    """x>0 の値の重み付き q 分位 (inverted CDF)。w は整数重み。"""
    m = (x > 0) & (w > 0)
    if not m.any():
        return np.inf
    v, ww = x[m], w[m]
    o = np.argsort(v, kind="stable")
    v, ww = v[o], ww[o]
    cw = np.cumsum(ww)
    return float(v[np.searchsorted(cw, q * cw[-1], side="left")])


def caps_for(R, valid, w):
    """券種ごとの打ち切り値 {est: (8,)}。"""
    return {e: np.array([wquantile_pos(R[:, j], w * valid[:, j], q) for j in range(R.shape[1])])
            for e, q in WIN_Q.items()}


def estimates(R, valid, w, caps):
    """{est: (8,)} の ROI(%) 推定値。w は race 重み。"""
    W = w[:, None] * valid
    den = W.sum(0)
    out = {"plain": 100.0 * (W * R).sum(0) / den}
    for e in WIN_Q:
        out[e] = 100.0 * (W * np.minimum(R, caps[e][None, :])).sum(0) / den
    for e, f in LOG_F.items():
        g = (W * np.log1p(f * (R - 1.0))).sum(0) / den
        out[e] = 100.0 * (1.0 + np.expm1(g) / f)
    return out


def day_weights(day_idx, n_days, rng):
    """開催日ブロック bootstrap: 日を復元抽出し、race 重み (整数) を返す。"""
    cnt = np.bincount(rng.integers(0, n_days, n_days), minlength=n_days)
    return cnt[day_idx].astype(float)


def tau(a, b):
    t = kendalltau(a, b).statistic
    return float(t) if np.isfinite(t) else np.nan


def ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


# ---------------------------------------------------------------- rotation
def run_rotation(fy, eys, date, year, R, valid, reps, seed):
    fm = year == fy
    em = np.isin(year, eys)
    Rf, Vf, Re, Ve = R[fm], valid[fm], R[em], valid[em]
    fd_u, fd_i = np.unique(date[fm], return_inverse=True)
    ed_u, ed_i = np.unique(date[em], return_inverse=True)

    caps0 = caps_for(Rf, Vf, np.ones(len(Rf)))
    fit0 = estimates(Rf, Vf, np.ones(len(Rf)), caps0)
    ev0 = estimates(Re, Ve, np.ones(len(Re)), caps0)

    rng = np.random.default_rng(seed)
    B = {e: np.empty((reps, len(KINDS))) for e in ESTS}
    tc = {e: np.empty(reps) for e in ESTS}   # τ_cross: fit 序列 vs 評価序列 (両側 resample)
    tb = {e: np.empty(reps) for e in ESTS}   # τ_boot: 評価複製序列 vs 評価全標本序列
    tt = {e: np.empty(reps) for e in ESTS}   # τ_target: 推定量の評価序列 vs plain の評価序列
    capB = {e: np.empty((reps, len(KINDS))) for e in WIN_Q}
    for k in range(reps):
        wf = day_weights(fd_i, len(fd_u), rng)
        we = day_weights(ed_i, len(ed_u), rng)
        cp = caps_for(Rf, Vf, wf)
        fe = estimates(Rf, Vf, wf, cp)
        ee = estimates(Re, Ve, we, cp)
        for e in ESTS:
            B[e][k] = ee[e]
            tc[e][k] = tau(fe[e], ee[e])
            tb[e][k] = tau(ee[e], ev0[e])
            tt[e][k] = tau(ee[e], ee["plain"])
        for e in WIN_Q:
            capB[e][k] = cp[e]

    n_eval = Ve.sum(0)
    se_plain = B["plain"].std(0, ddof=1)
    res = dict(fit_year=fy, eval_years=list(eys), n_fit_races=int(fm.sum()), n_eval_races=int(em.sum()),
               n_eval_by_kind={JP[k]: int(n_eval[j]) for j, k in enumerate(KINDS)},
               n_fit_days=len(fd_u), n_eval_days=len(ed_u),
               caps_x100yen={e: {JP[k]: float(caps0[e][j]) for j, k in enumerate(KINDS)} for e in WIN_Q},
               caps_ci={e: {JP[k]: ci(capB[e][:, j]) for j, k in enumerate(KINDS)} for e in WIN_Q},
               estimators={})
    for e in ESTS:
        se = B[e].std(0, ddof=1)
        diff = B[e] - B["plain"]
        bias = ev0[e] - ev0["plain"]
        se500 = se * np.sqrt(n_eval / 500.0)
        per = {}
        for j, kd in enumerate(KINDS):
            per[JP[kd]] = dict(
                fit=float(fit0[e][j]), eval=float(ev0[e][j]), eval_ci=ci(B[e][:, j]),
                se=float(se[j]), se_ratio=float(se[j] / se_plain[j]),
                bias_vs_plain=float(bias[j]), bias_se=float(diff[:, j].std(ddof=1)),
                se500=float(se500[j]), rmse500=float(np.sqrt(se500[j] ** 2 + bias[j] ** 2)),
            )
        rank_fit = [JP[KINDS[i]] for i in np.argsort(-fit0[e])]
        rank_eval = [JP[KINDS[i]] for i in np.argsort(-ev0[e])]
        t_point = tau(fit0[e], ev0[e])
        res["estimators"][e] = dict(
            per_kind=per,
            se_ratio_median=float(np.median(se / se_plain)),
            se_ratio_geomean=float(np.exp(np.mean(np.log(se / se_plain)))),
            rank_fit=rank_fit, rank_eval=rank_eval,
            tau_cross=dict(point=t_point, ci=ci(tc[e]), boot_mean=float(np.nanmean(tc[e])),
                           p_tau_eq_1=float(np.mean(tc[e] >= 1 - 1e-12)),
                           gate_ci_contains_1=bool(ci(tc[e])[1] >= 1 - 1e-12)),
            tau_boot=dict(ci=ci(tb[e]), mean=float(np.nanmean(tb[e])),
                          p_tau_eq_1=float(np.mean(tb[e] >= 1 - 1e-12))),
            tau_target_vs_plain=dict(point=tau(ev0[e], ev0["plain"]), ci=ci(tt[e])),
        )
    return res


# ---------------------------------------------------------------------- a95
def a95_frame(o1, n, R, pol):
    """凍結表の帯×券種・均等配分 (1R 1 万円、100 円格子、枠連は 9 頭以上) の stake 行列 (races, 8)。"""
    cap, unit = int(pol["cap_per_race_yen"]), int(pol["unit_yen"])
    S = np.zeros_like(R)
    for i in range(len(R)):
        band = None
        for lo, hi, name in pol["bands"]:
            if lo <= o1[i] < (hi if hi is not None else np.inf):
                band = name
                break
        if band is None:
            band = pol["bands"][-1][2]
        kinds = [k for k in pol["plan"][band] if not (k == "wakuren" and n[i] <= 8)]
        base = (cap // len(kinds)) // unit * unit
        st = {k: base for k in kinds}
        rem, j = cap - base * len(kinds), 0
        while rem >= unit:
            st[kinds[j % len(kinds)]] += unit
            rem -= unit
            j += 1
        for k, v in st.items():
            S[i, KINDS.index(k)] = v
    return S


def tail_stats(Y, cost, w):
    """Y: レース単位払戻(円)、cost: レース単位投資(円)、w: 整数重み。"""
    tot_ret, tot_cost = float((w * Y).sum()), float((w * cost).sum())
    m = (Y > 0) & (w > 0)
    v = np.repeat(Y[m], w[m].astype(int))
    v = np.sort(v)[::-1]
    cs = np.concatenate([[0.0], np.cumsum(v)])
    nh = len(v)
    out = dict(roi=100 * tot_ret / tot_cost, hit_races=nh)
    for lab, k in (("top1pct", int(round(0.01 * nh))), ("top5pct", int(round(0.05 * nh))), ("top10", 10)):
        out[f"roi_ex_{lab}"] = 100 * (tot_ret - cs[k]) / tot_cost
        out[f"share_{lab}"] = cs[k] / tot_ret
        out[f"k_{lab}"] = k
    for thr in (100.0, 95.0):
        need = tot_ret - thr / 100 * tot_cost          # これを超える分を上から削ると thr 未満
        if need <= 0:
            out[f"n_to_below_{int(thr)}"] = 0
        else:
            out[f"n_to_below_{int(thr)}"] = int(np.searchsorted(cs, need, side="right"))
        out[f"pct_hits_to_below_{int(thr)}"] = 100 * out[f"n_to_below_{int(thr)}"] / nh
    return out


def a95_block(date, year, n, o1, R, valid, pol, reps, seed):
    S = a95_frame(o1, n, R, pol)
    cost = S.sum(1)
    Y = (S * R).sum(1)
    out = {}
    # 裾依存 (全期間 / 評価 2024-25 / 年別)
    for lab, msk in (("2023-25", np.ones(len(R), bool)), ("2024-25", np.isin(year, ["2024", "2025"])),
                     ("2023", year == "2023"), ("2024", year == "2024"), ("2025", year == "2025")):
        du, di = np.unique(date[msk], return_inverse=True)
        pt = tail_stats(Y[msk], cost[msk], np.ones(msk.sum()))
        rng = np.random.default_rng(seed + 11)
        bs = [tail_stats(Y[msk], cost[msk], day_weights(di, len(du), rng)) for _ in range(reps)]
        rr = Y[msk] / cost[msk]
        d = dict(n_races=int(msk.sum()), n_days=len(du),
                 race_sd_x=float(rr.std(ddof=1)),
                 se500_iid=float(100 * rr.std(ddof=1) / np.sqrt(500)),
                 race_hit=float(100 * np.mean(Y[msk] > 0)))
        for key, val in pt.items():
            arr = np.array([b[key] for b in bs], float)
            d[key] = dict(point=float(val), se=float(arr.std(ddof=1)), ci=ci(arr))
        d["se500_block"] = float(d["roi"]["se"] * np.sqrt(msk.sum() / 500.0))
        out[lab] = d

    # winsor / log (fit=2023 → 評価 2024-25)
    fm, em = year == "2023", np.isin(year, ["2024", "2025"])
    fd_u, fd_i = np.unique(date[fm], return_inverse=True)
    ed_u, ed_i = np.unique(date[em], return_inverse=True)

    def est(caps, w):
        c, s, r = cost[em], S[em], R[em]
        o = {"plain": 100 * (w * Y[em]).sum() / (w * c).sum()}
        for e in WIN_Q:
            o[e] = 100 * (w * (s * np.minimum(r, caps[e][None, :])).sum(1)).sum() / (w * c).sum()
        rr = Y[em] / c
        for e, f in LOG_F.items():
            g = (w * np.log1p(f * (rr - 1))).sum() / w.sum()
            o[e] = 100 * (1 + np.expm1(g) / f)
        return o

    caps0 = caps_for(R[fm], valid[fm], np.ones(fm.sum()))
    p0 = est(caps0, np.ones(em.sum()))
    rng = np.random.default_rng(seed + 23)
    bs = []
    for _ in range(reps):
        cp = caps_for(R[fm], valid[fm], day_weights(fd_i, len(fd_u), rng))
        bs.append(est(cp, day_weights(ed_i, len(ed_u), rng)))
    sep = np.std([b["plain"] for b in bs], ddof=1)
    out["estimators_fit2023_eval2024_25"] = {
        e: dict(point=float(p0[e]), se=float(np.std([b[e] for b in bs], ddof=1)),
                se_ratio=float(np.std([b[e] for b in bs], ddof=1) / sep),
                ci=ci([b[e] for b in bs]), bias_vs_plain=float(p0[e] - p0["plain"]),
                bias_se=float(np.std([b[e] - b["plain"] for b in bs], ddof=1)),
                se500=float(np.std([b[e] for b in bs], ddof=1) * np.sqrt(em.sum() / 500.0)))
        for e in ESTS}
    out["in_sample_note"] = "凍結表は 2023-25 全期間の表から作成。ここでの a95 ROI 水準は in-sample で、SE と裾依存度の記述用。"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261022)
    a = ap.parse_args()

    date, year, n, o1, R, valid = load()
    pol = json.loads(POL.read_text(encoding="utf-8"))
    meta = dict(substrate=str(SUB.relative_to(BASE)), substrate_sha256=sha(SUB),
                policy=str(POL.relative_to(BASE)), policy_sha256_file=sha(POL),
                n_races=int(len(R)), years={y: int((year == y).sum()) for y in ("2023", "2024", "2025")},
                reps=a.reps, seed=a.seed, kinds=[JP[k] for k in KINDS],
                prereg="docs/research/M2_ROI_ESTIMATOR_FRONTIER_PREREG_20261022.md")
    print(f"races={len(R)} {meta['years']} reps={a.reps}", flush=True)

    rot = []
    for i, (fy, eys) in enumerate(ROTATIONS):
        print(f"rotation fit={fy} eval={'+'.join(eys)} ...", flush=True)
        rot.append(run_rotation(fy, eys, date, year, R, valid, a.reps, a.seed + i))
    print("a95 ...", flush=True)
    a95 = a95_block(date, year, n, o1, R, valid, pol, a.reps, a.seed)

    OUT.mkdir(parents=True, exist_ok=True)
    res = dict(meta=meta, rotations=rot, a95=a95)
    (OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- summary
    L = []
    p = rot[0]
    L.append(f"=== 一次ローテーション fit={p['fit_year']} → eval={'+'.join(p['eval_years'])} ===")
    L.append("推定量      SE比(中央) SE比(幾何)  τ_cross 点 [CI95]        P(τ=1)  ゲート(CI∋1)  τ_boot平均  τ_target(vs plain)")
    for e in ESTS:
        d = p["estimators"][e]
        tc = d["tau_cross"]
        L.append(f"{e:<10} {d['se_ratio_median']:9.3f} {d['se_ratio_geomean']:9.3f}   {tc['point']:+.3f} [{tc['ci'][0]:+.3f},{tc['ci'][1]:+.3f}]  {tc['p_tau_eq_1']:.3f}   {str(tc['gate_ci_contains_1']):<5}       {d['tau_boot']['mean']:.3f}      {d['tau_target_vs_plain']['point']:+.3f}")
    L.append("")
    for e in ESTS:
        d = p["estimators"][e]
        L.append(f"[{e}] fit序列: {' > '.join(d['rank_fit'])}")
        L.append(f"{'':>{len(e)+2}} eval序列: {' > '.join(d['rank_eval'])}")
        for kd, v in d["per_kind"].items():
            L.append(f"   {kd:<4} eval {v['eval']:6.1f} [{v['eval_ci'][0]:6.1f},{v['eval_ci'][1]:6.1f}] SE {v['se']:5.2f} 比 {v['se_ratio']:.3f} bias {v['bias_vs_plain']:+6.1f}±{v['bias_se']:.1f} SE500 {v['se500']:5.1f} RMSE500 {v['rmse500']:5.1f}")
    for r in rot[1:]:
        L.append(f"\n=== 副次 fit={r['fit_year']} → eval={'+'.join(r['eval_years'])} ===")
        for e in ESTS:
            d = r["estimators"][e]; tc = d["tau_cross"]
            L.append(f"{e:<10} SE比中央 {d['se_ratio_median']:.3f}  τ_cross {tc['point']:+.3f} [{tc['ci'][0]:+.3f},{tc['ci'][1]:+.3f}] CI∋1={tc['gate_ci_contains_1']}")
    L.append("\n=== a95 裾依存 ===")
    for lab in ("2023-25", "2024-25", "2023", "2024", "2025"):
        d = a95[lab]
        g = lambda k: f"{d[k]['point']:6.1f}±{d[k]['se']:4.1f}"
        L.append(f"{lab}: n={d['n_races']} ROI {g('roi')} [{d['roi']['ci'][0]:.1f},{d['roi']['ci'][1]:.1f}] R的中 {d['race_hit']:.1f}% SD/R {d['race_sd_x']:.2f} SE500 iid {d['se500_iid']:.1f} block {d['se500_block']:.1f}")
        L.append(f"     除外後ROI 上位1%({d['k_top1pct']['point']:.0f}本) {g('roi_ex_top1pct')} / 上位5%({d['k_top5pct']['point']:.0f}本) {g('roi_ex_top5pct')} / 上位10本 {g('roi_ex_top10')}")
        L.append(f"     裾シェア 上位1% {100*d['share_top1pct']['point']:.1f}±{100*d['share_top1pct']['se']:.1f}% / 5% {100*d['share_top5pct']['point']:.1f}±{100*d['share_top5pct']['se']:.1f}% / 10本 {100*d['share_top10']['point']:.1f}±{100*d['share_top10']['se']:.1f}%")
        L.append(f"     100%未満まで {d['n_to_below_100']['point']:.0f}本({d['pct_hits_to_below_100']['point']:.2f}%±{d['pct_hits_to_below_100']['se']:.2f}) / 95%未満まで {d['n_to_below_95']['point']:.0f}本({d['pct_hits_to_below_95']['point']:.2f}%±{d['pct_hits_to_below_95']['se']:.2f})")
    L.append("a95 推定量 (fit 2023 → eval 2024-25):")
    for e, v in a95["estimators_fit2023_eval2024_25"].items():
        L.append(f"   {e:<10} {v['point']:6.1f} [{v['ci'][0]:6.1f},{v['ci'][1]:6.1f}] SE {v['se']:.2f} 比 {v['se_ratio']:.3f} bias {v['bias_vs_plain']:+.1f}±{v['bias_se']:.1f} SE500 {v['se500']:.1f}")
    txt = "\n".join(L)
    (OUT / "summary.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
