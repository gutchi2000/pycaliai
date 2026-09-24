# -*- coding: utf-8 -*-
"""
equivalence_audit.py — EXP17 Stage 0: EXP02 scalar ability との数理・実測等価性監査 (label-free, 2022)
=====================================================================================================
2022 の対象レース結果 (着順) は一切使わない。使うのは
  (a) 対象日より前の履歴対戦 (= 特徴そのもの)、(b) EXP02 の as-of 能力 dyn_skill_mu (pre-race 値)。

■ 事前固定した判定基準 (実行前に確定、結果を見て変えない)
  E0-A 代数: Hodge 射影の出力 s は定義上「馬ごとのスカラー」。pair 固有情報は射影残差 (curl) にのみ残る。
        → 確率 arm (softmax(s/tau)) は構造的に scalar-rating モデル。合成テストで確認済み。
  E0-B 実測 affine: 2022 の (i) 射影後 s_i と EXP02 μ_i のレース内回帰 R²、(ii) 未対戦 pair の d_ij と b·Δμ_ij の関係。
        判定量は「BT null に対する超過分散比」 rho = Var_real(d_ij - b·Δμ_ij) / E_null[Var(d_ij - b·Δμ_ij)]。
        null = 各過去対戦の勝敗を σ(b·(μ_h(m) - μ_c(m))) から再抽選 (μ は対戦時点の EXP02 pre-race 値)、
        200 draw、count 構造 (n_hc, ω) は実データのまま。
        **FAIL 条件: rho の 95% 区間 (meeting-day block bootstrap) の上限が 1.10 未満** (pair 固有状態が BT 抽選ノイズと
        区別できない)。補助: 共通対戦馬 c1,c2 を介した残差の一致相関 corr(r_c1, r_c2) が null の 97.5%ile を超えるか。
  E0-C 射影: pair 証拠の分散のうち射影で捨てられる割合 (curl share) を報告。確率 arm に pair 固有情報が残らないことは
        E0-A の帰結であり、実測値は「捨てた量」の大きさを示すだけ。
実行: python -m analysis.mcond.exp17_transitive_pl_graph_dev.equivalence_audit
出力: out/equivalence_audit.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .coverage_audit import PRM, load_master
from .graph_core import build_history_store, ell, hodge_project, pair_evidence

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[2]
OUT = HERE / "out"
EXP02 = BASE / "data" / "_research" / "mcond" / "exp02_features.parquet"
YEAR = 2022
N_NULL = 200
SEED = 20260925
RHO_FAIL_UPPER = 1.10
B_FIT_MAX_DATE = 20181231      # pair scale b は 2018 以前の対戦だけで fit (development 2019-2023 の結果を使わない)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    m = load_master()
    flat = m[~m["track"].between(51, 59)].dropna(subset=["fin"]).copy()
    runs = flat[["date", "rid16", "hid", "fin", "surf", "dband"]]
    store = build_history_store(runs)
    mu = pd.read_parquet(EXP02, columns=["rid16", "hid", "dyn_skill_mu"])
    mu["rid16"] = mu["rid16"].astype(str); mu["hid"] = mu["hid"].astype(str)
    mu_map = dict(zip(zip(mu["rid16"], mu["hid"]), mu["dyn_skill_mu"].to_numpy()))

    # ---- pair scale b: P(h beats c at meeting m) = σ(b (μ_h(m) - μ_c(m))), meetings ≤ 2018 のみ、300k サンプル
    hh, cc, dd, ww, rc = store.h, store.c, store.date, store.won, store.ridc
    sel = np.flatnonzero((dd <= B_FIT_MAX_DATE) & (hh < cc))
    sel = rng.choice(sel, size=min(300_000, len(sel)), replace=False)
    dmu = np.array([mu_map.get((str(store.rid_of[rc[k]]), str(store.hid_of[hh[k]])), np.nan) -
                    mu_map.get((str(store.rid_of[rc[k]]), str(store.hid_of[cc[k]])), np.nan) for k in sel])
    y = ww[sel].astype(float)
    ok = ~np.isnan(dmu)
    dmu, y = dmu[ok], y[ok]
    # 1-parameter logistic MLE (no intercept, antisymmetric by construction)
    b = 0.05
    for _ in range(50):
        p = sigmoid(b * dmu); g = np.sum((y - p) * dmu); H = -np.sum(p * (1 - p) * dmu ** 2)
        step = g / H; b -= step
        if abs(step) < 1e-10:
            break
    b_fit = {"b": float(b), "n_meetings": int(len(y)), "max_meeting_date": B_FIT_MAX_DATE,
             "pair_logloss_bt": float(-np.mean(y * np.log(sigmoid(b * dmu)) + (1 - y) * np.log(1 - sigmoid(b * dmu)))),
             "pair_logloss_coinflip": float(np.log(2))}
    print("b fit", b_fit, flush=True)

    # ---- 2022 official races (same rule as coverage_audit) → per race: d, s, per_c, meetings
    g = flat[flat["year"] == YEAR].groupby("rid16")
    meta = g.agg(date=("date", "first"), n_fin=("hid", "size"), tou=("tou", "first"), n_win=("fin", lambda s: int((s == 1).sum())),
                 surf=("surf", "first"), dband=("dband", "first"))
    off = meta[(meta.n_win == 1) & (meta.n_fin >= 5) & (meta.n_fin == meta.tou)]
    tgt = flat[flat["rid16"].isin(off.index)].sort_values(["date", "rid16", "ban"])

    s_rows = []            # projected score vs μ (race-demeaned)
    pair_rows = []         # (rid, day, i, j, d_ij, dmu_ij, n_common)
    edge_meet = {}         # (h_code, c_code, day) -> (p_null array, won array)
    edge_index = {}
    trip = []              # per (i,j,c): pair_row_idx, edge_ic_id, edge_jc_id, omega, n_ic, n_jc
    curl_shares = []
    for rid, gr in tgt.groupby("rid16", sort=True):
        day = int(gr["date"].iloc[0]); hids = gr["hid"].tolist(); n = len(hids)
        res = pair_evidence(store, hids, day, int(gr["surf"].iloc[0]), int(gr["dband"].iloc[0]), PRM)
        s, unc, comps, resid = hodge_project(res.d, res.w, PRM.lam)
        mus = np.array([mu_map.get((rid, h), np.nan) for h in hids])
        cov = ~unc & ~np.isnan(mus)
        if cov.sum() >= 3:
            for k in np.flatnonzero(cov):
                s_rows.append(dict(rid16=rid, day=day, s=float(s[k]), mu=float(mus[k]), n_cov=int(cov.sum())))
        obs = ~np.isnan(res.d); np.fill_diagonal(obs, False)
        if obs.any():
            curl_shares.append(float((resid[obs] ** 2).sum() / max((res.d[obs] ** 2).sum(), 1e-12)))
        codes = [store.code_of[h] for h in hids]
        for (i, j), lst in res.per_c.items():
            if res.direct[i, j] or np.isnan(mus[i]) or np.isnan(mus[j]):
                continue
            pidx = len(pair_rows)
            pair_rows.append(dict(rid16=rid, day=day, i=i, j=j, d_ij=float(res.d[i, j]), dmu=float(mus[i] - mus[j]),
                                  n_common=len(lst), w_ij=float(res.w[i, j])))
            for (c, dc, om) in lst:
                for h_code, side in ((codes[i], "i"), (codes[j], "j")):
                    key = (h_code, c, day)
                    if key not in edge_index:
                        rids, dts, won = store.meetings_before(h_code, c, day)
                        pn = np.array([sigmoid(b * (mu_map.get((r, str(store.hid_of[h_code])), np.nan) -
                                                    mu_map.get((r, str(store.hid_of[c])), np.nan))) for r in rids])
                        pn = np.where(np.isnan(pn), 0.5, pn)      # μ 欠損 (稀) は 0.5
                        edge_index[key] = len(edge_meet)
                        edge_meet[len(edge_meet)] = (pn, won.astype(float))
                trip.append((pidx, edge_index[(codes[i], c, day)], edge_index[(codes[j], c, day)], om))
    P = pd.DataFrame(pair_rows); S = pd.DataFrame(s_rows)
    print(f"2022 races {len(off)}  pairs {len(P)}  triples {len(trip)}  edges {len(edge_meet)}  {time.time()-t0:.0f}s", flush=True)

    # ---- E0-B (i): projected score s vs EXP02 μ, race-demeaned
    S["s_dm"] = S["s"] - S.groupby("rid16")["s"].transform("mean")
    S["mu_dm"] = S["mu"] - S.groupby("rid16")["mu"].transform("mean")
    x, yv = S["mu_dm"].to_numpy(), S["s_dm"].to_numpy()
    beta = float((x * yv).sum() / max((x * x).sum(), 1e-12))
    r2_s = float(1 - ((yv - beta * x) ** 2).sum() / max((yv ** 2).sum(), 1e-12))
    corr_s = float(np.corrcoef(x, yv)[0, 1])

    # ---- E0-B (ii): d_ij vs b·Δμ ; real residual variance
    d_real = P["d_ij"].to_numpy(); dmu_ij = P["dmu"].to_numpy() * b
    r_real = d_real - dmu_ij
    slope_d = float((dmu_ij * d_real).sum() / max((dmu_ij ** 2).sum(), 1e-12))
    r2_d = float(1 - ((d_real - slope_d * dmu_ij) ** 2).sum() / max(((d_real - d_real.mean()) ** 2).sum(), 1e-12))
    var_real = float(np.var(r_real))

    # ---- null: resample every historical meeting outcome under BT(b, μ at meeting), rebuild ell → d_c → d_ij
    E = len(edge_meet)
    meet_edge = np.concatenate([np.full(len(edge_meet[e][0]), e) for e in range(E)])
    meet_p = np.concatenate([edge_meet[e][0] for e in range(E)])
    meet_won = np.concatenate([edge_meet[e][1] for e in range(E)])
    n_edge = np.bincount(meet_edge, minlength=E).astype(float)
    W_real = np.bincount(meet_edge, weights=meet_won, minlength=E)
    trip_arr = np.array(trip, dtype=object)
    t_pair = np.array([t[0] for t in trip], int); t_ic = np.array([t[1] for t in trip], int)
    t_jc = np.array([t[2] for t in trip], int); t_om = np.array([t[3] for t in trip], float)
    om_sum = np.bincount(t_pair, weights=t_om, minlength=len(P))

    def ell_vec(W, n):
        p = (W + PRM.alpha) / (n + 2 * PRM.alpha)
        p = np.clip(p, PRM.eps, 1 - PRM.eps)
        return np.log(p / (1 - p))

    def d_from_W(W):
        e = ell_vec(W, n_edge)
        dc = e[t_ic] - e[t_jc]
        return np.bincount(t_pair, weights=t_om * dc, minlength=len(P)) / om_sum

    d_check = d_from_W(W_real)
    recon_err = float(np.nanmax(np.abs(d_check - d_real)))
    days = P["day"].to_numpy(); uday = np.unique(days); day_pos = {d: np.flatnonzero(days == d) for d in uday}

    var_null = np.empty(N_NULL); agree_null = np.empty(N_NULL)
    # cross-common-opponent agreement: pairs with >= 2 c → corr between residual via first and second c
    multi = P.index[P["n_common"] >= 2].to_numpy()
    first_c = {}; second_c = {}
    for k, (pidx, eic, ejc, om) in enumerate(trip):
        if pidx in first_c and pidx not in second_c and first_c[pidx] != k:
            second_c[pidx] = k
        first_c.setdefault(pidx, k)
    mi = np.array([p for p in multi if p in second_c]); k1 = np.array([first_c[p] for p in mi]); k2 = np.array([second_c[p] for p in mi])

    def agreement(W):
        e = ell_vec(W, n_edge); dc = e[t_ic] - e[t_jc]
        r1 = dc[k1] - dmu_ij[mi]; r2 = dc[k2] - dmu_ij[mi]
        return float(np.corrcoef(r1, r2)[0, 1])

    agree_real = agreement(W_real)
    for it in range(N_NULL):
        won_sim = (rng.random(len(meet_p)) < meet_p).astype(float)
        Wn = np.bincount(meet_edge, weights=won_sim, minlength=E)
        dn = d_from_W(Wn)
        var_null[it] = np.var(dn - dmu_ij)
        agree_null[it] = agreement(Wn)
    rho = var_real / var_null.mean()
    # day-block bootstrap of var_real (null mean treated as fixed; null draw variance reported separately)
    B = 1000; boots = np.empty(B)
    for bb in range(B):
        pick = rng.choice(uday, len(uday), replace=True)
        idx = np.concatenate([day_pos[d] for d in pick])
        boots[bb] = np.var(r_real[idx]) / var_null.mean()
    rho_ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]

    # counts structure
    n_meet_used = n_edge
    out = {
        "population": {"year": YEAR, "official_races": int(len(off)), "never_met_pairs_with_common_opponent_and_mu": int(len(P)),
                        "triples_ijc": int(len(trip)), "unique_hc_edges": int(E), "meetings": int(len(meet_p))},
        "bt_pair_scale": b_fit,
        "E0_A_algebraic": {
            "statement": "Hodge 射影の出力 s は馬ごとのスカラー。pair 固有情報は射影残差にのみ残り、softmax(s/tau) には入らない",
            "synthetic_check": "test_invariants.json: algebraic_reduction_when_transitive / pair_specific_info_survives_only_in_residual",
            "verdict": "race-level 確率 arm は構造的に scalar-rating モデル (EXP02 と同じモデル族、推定量が異なるだけ)",
        },
        "E0_B_projected_score_vs_exp02_mu": {"n_horse_rows": int(len(S)), "race_demeaned_slope": beta, "race_demeaned_R2": r2_s,
                                              "race_demeaned_corr": corr_s},
        "E0_B_pair_d_vs_bt_gap": {"slope_d_on_bdmu": slope_d, "R2": r2_d, "corr": float(np.corrcoef(d_real, dmu_ij)[0, 1]),
                                   "var_d_real": float(np.var(d_real)), "var_bdmu": float(np.var(dmu_ij)),
                                   "var_residual_real": var_real},
        "E0_B_excess_variance_vs_bt_null": {
            "n_null_draws": N_NULL, "seed": SEED, "var_residual_null_mean": float(var_null.mean()),
            "var_residual_null_sd_across_draws": float(var_null.std()),
            "rho_real_over_null": float(rho), "rho_ci95_day_block_bootstrap": rho_ci,
            "fail_rule": f"rho CI95 upper < {RHO_FAIL_UPPER} → pair 固有状態は BT 抽選ノイズと区別できない",
            "fail": bool(rho_ci[1] < RHO_FAIL_UPPER),
            "reconstruction_max_abs_err_real_d": recon_err,
        },
        "E0_B_cross_common_opponent_agreement": {
            "n_pairs_with_ge2_common_opponents": int(len(mi)), "corr_real": agree_real,
            "corr_null_mean": float(agree_null.mean()), "corr_null_p97_5": float(np.percentile(agree_null, 97.5)),
            "exceeds_null_97_5": bool(agree_real > np.percentile(agree_null, 97.5)),
            "caveat": "null は μ を真値扱い。μ 推定誤差は real の相関を正側に押すので、超過は pair 固有状態の十分条件ではない",
        },
        "E0_C_projection": {"curl_share_median": float(np.median(curl_shares)), "curl_share_mean": float(np.mean(curl_shares)),
                            "meaning": "pair 証拠の二乗和のうち射影で捨てられる割合 = 確率 arm に届かない pair 固有成分の大きさ"},
        "meeting_count_structure": {"share_edges_n_eq1": float((n_meet_used == 1).mean()), "share_edges_n_le2": float((n_meet_used <= 2).mean()),
                                    "mean_meetings_per_edge": float(n_meet_used.mean()),
                                    "note": "n=1 の辺の ell は ±logit((1+α)/(1+2α)) の2値。d_ij^(c) は事実上 3 値の符号情報"},
        "elapsed_sec": time.time() - t0,
    }
    (OUT / "equivalence_audit.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
