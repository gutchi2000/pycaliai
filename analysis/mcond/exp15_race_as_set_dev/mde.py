# -*- coding: utf-8 -*-
"""
mde.py — 学習開始前の MDE (最小検出効果) と実務閾値を 2023 development で算出する
==================================================================================
R0 代理 A: oof_scores_c1master (2023 = train<=2021 / ES 2022 の v6 レシピ OOF)
R0 代理 B: oof_scores_v6params (旧 OOF)。A-B の per-race 差の meeting-day クラスタ SE から MDE。
τ は各代理の 2022 OOF スコアで最尤 (analysis.mcond.v6base の慣行)。
感度: Var(Δ_r) = 2σ²(1-ρ)、ρ∈{0.90,0.95,0.98}、design effect は A の per-race 指標から推定。
出力: out/mde.json  (学習前にコミットする)
実行: python -m analysis.mcond.exp15_race_as_set_dev.mde
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import common as C

Z = 1.959964 + 0.841621  # 両側 5% / 検出力 80%


def proxy_scores(df, path):
    sc = pd.read_parquet(path, filters=[("year", "in", [2022, 2023])])
    sc["rid16"] = sc["rid"].astype(str).str[:16]
    m = df[["rid16", "ban"]].merge(sc[["rid16", "ban", "score"]], on=["rid16", "ban"], how="left")
    return m["score"].to_numpy()


def main():
    df = C.load_rows()
    df = df[df["period"].isin(["sel", "dev"])].reset_index(drop=True)
    sel = (df["period"] == "sel").to_numpy()
    dev = (df["period"] == "dev").to_numpy()
    r_sel = C.race_index(df, sel)
    r_dev_all = C.race_index(df, dev)
    r_dev = [i for i in r_dev_all if len(i) >= C.SPEC["population"]["eval_min_field"]]
    win = df["win"].to_numpy()

    tabs = {}
    for name, path in [("A", C.BASE / "data/_research/mcond/oof_scores_c1master.parquet"),
                       ("B", C.BASE / "data/oof_scores_v6params.parquet")]:
        s = proxy_scores(df, path)
        assert not np.isnan(s[dev]).any() and not np.isnan(s[sel]).any(), name
        tau = C.fit_tau(s, win, r_sel)
        p = C.softmax_race(s, r_dev, tau)
        tabs[name] = C.race_metrics(df, r_dev, s, p)
        tabs[name + "_tau"] = tau
    A, B = tabs["A"], tabs["B"]
    day = A["meeting_day"].to_numpy()
    level = {"race_win_logloss": float(A["ll"].mean()), "race_win_brier": float(A["brier"].mean()),
             "ndcg@3": float(A["ndcg3"].mean()), "hon_top3": float(A["hon_top3"].mean())}
    colmap = {"race_win_logloss": "ll", "race_win_brier": "brier", "ndcg@3": "ndcg3",
              "hon_top3": "hon_top3"}
    pt = C.SPEC["mde"]["practical_thresholds"]
    practical = {
        "race_win_logloss": pt["context"]["race_win_logloss_rel"] * level["race_win_logloss"],
        "race_win_brier": pt["context"]["race_win_brier_rel"] * level["race_win_brier"],
        "ndcg@3": pt["context"]["ndcg@3_abs"],
        "hon_top3": pt["context"]["hon_top3_abs"],
    }
    practical_rep = {
        "race_win_logloss": pt["replacement"]["race_win_logloss_rel"] * level["race_win_logloss"],
        "race_win_brier": pt["replacement"]["race_win_brier_rel"] * level["race_win_brier"],
    }
    res = {}
    for m, col in colmap.items():
        a, b = A[col].to_numpy(dtype=float), B[col].to_numpy(dtype=float)
        d = a - b
        bd = C.boot_delta(d, day)
        ok = ~np.isnan(d)
        se_iid = float(np.nanstd(d, ddof=1) / np.sqrt(ok.sum()))
        # A 単体の per-race 指標の design effect
        ba = C.boot_delta(a - np.nanmean(a), day)
        sa_iid = float(np.nanstd(a, ddof=1) / np.sqrt((~np.isnan(a)).sum()))
        deff = (ba["se"] / sa_iid) ** 2
        rho_emp = float(pd.Series(a).corr(pd.Series(b)))
        sigma2 = float(np.nanvar(a, ddof=1))
        grid = {}
        for rho in (0.90, 0.95, 0.98):
            se = np.sqrt(2 * sigma2 * (1 - rho) / ok.sum() * deff)
            grid[str(rho)] = float(Z * se)
        mde = float(Z * bd["se"])
        res[m] = {
            "r0_proxy_level": level[m], "pair_delta_A_minus_B": bd["delta"],
            "se_cluster": bd["se"], "se_iid": se_iid, "design_effect_A": float(deff),
            "rho_empirical_A_B": rho_emp, "MDE_primary": mde, "MDE_rho_grid": grid,
            "practical_context": practical[m],
            "practical_replacement": practical_rep.get(m),
            "context_threshold_min_MDE_practical": min(mde, practical[m]),
            "replacement_threshold_max_MDE_practical": (max(mde, practical_rep[m])
                                                        if m in practical_rep else None),
            "underpowered_vs_practical": bool(mde > practical[m]),
        }
    out = {
        "computed_before_training": True,
        "planned_counts_2023": {
            "rows": int(dev.sum()), "races_all": len(r_dev_all), "races_eval_n>=5": len(r_dev),
            "races_single_winner": int(A["single_win"].sum()),
            "races_deadheat_excluded": int((~A["single_win"]).sum()),
            "meeting_days": int(A["meeting_day"].nunique()),
        },
        "tau": {"A": tabs["A_tau"], "B": tabs["B_tau"]},
        "proxies": {"A": "oof_scores_c1master (train<=2021/ES2022)", "B": "oof_scores_v6params (旧OOF)"},
        "z": Z,
        "metrics": res,
    }
    C.dump(out, "mde.json")
    print(out["planned_counts_2023"], out["tau"])
    for m, r in res.items():
        print(f"{m:18s} level={r['r0_proxy_level']:.5f} MDE={r['MDE_primary']:.5f} "
              f"practical={r['practical_context']:.5f} ctx_thr={r['context_threshold_min_MDE_practical']:.5f} "
              f"rho={r['rho_empirical_A_B']:.3f} deff={r['design_effect_A']:.2f} grid={r['MDE_rho_grid']} "
              f"underpowered={r['underpowered_vs_practical']}")


if __name__ == "__main__":
    main()
