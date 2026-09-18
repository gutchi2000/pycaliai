# -*- coding: utf-8 -*-
"""
run.py — EXP04 の Gate 0〜3, leave-one-environment-out 診断を spec.json どおりに実行する
実行: python -m analysis.mcond.exp04_invariant_info_dev.run
"""
from __future__ import annotations
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
from analysis.mcond.evaluate import fit_predict, delta_boot, metrics, logit, ll_vec  # noqa: E402
from analysis.mcond.exp04_invariant_info_dev import environments, methods  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
D = BASE / "data/_research/mcond"
SPEC = json.loads((HERE / "spec.json").read_text(encoding="utf-8"))
MASTER = BASE / "data/master_v2_20130105-20251228.csv"
C_GRID = tuple(SPEC["residual_model"]["C_grid"])


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def load():
    b = pd.read_parquet(D / "base.parquet")
    b["rid16"] = b["rid16"].astype(str)
    cand = pd.read_parquet(D / "exp04_candidates.parquet")
    cand["rid16"] = cand["rid16"].astype(str)
    cand_cols = [c for c in cand.columns if c not in ("rid16", "ban", "year")]
    df = b.merge(cand[["rid16", "ban"] + cand_cols], on=["rid16", "ban"], how="inner")

    m = pd.read_csv(MASTER, encoding="utf-8-sig", low_memory=False,
                    usecols=["レースID(新)", "芝・ダ", "距離"])
    m["rid16"] = m["レースID(新)"].astype(str).str[:16]
    m = m.drop_duplicates("rid16")[["rid16", "芝・ダ", "距離"]]
    df = df.merge(m, on="rid16", how="left")

    y0, y1 = 2016, 2025
    df = df[(df.year >= y0) & (df.year <= y1) & (df.n_field >= 5)
            & df.mkt_p3_pre.notna() & df.v6_p3.notna()].copy()
    df["f_v6"] = logit(df["v6_p3"])
    df["f_mkt"] = logit(df["mkt_p3_pre"])
    df["day"] = df["rid16"].str[:8]
    train0 = (df.year >= 2016) & (df.year <= 2021)
    envs, merge_info = environments.build(df["rid16"], df["芝・ダ"], df["距離"], df["year"],
                                          train0.to_numpy())
    df = pd.concat([df.reset_index(drop=True), envs.reset_index(drop=True)], axis=1)
    return df.reset_index(drop=True), cand_cols, merge_info


def design_matrix(df, cols, train_mask):
    """標準化(train統計)+欠損はtrain平均(=標準化後0)で補完した行列を返す。"""
    X = df[cols].to_numpy(dtype=float)
    mu = np.nanmean(X[train_mask], axis=0)
    sd = np.nanstd(X[train_mask], axis=0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    Z = np.where(np.isnan(Z), 0.0, Z)
    return Z


def fit_residual(df, cols, offset, y, train, sel, l2_grid_by_C=True):
    Z = design_matrix(df, cols, train)
    n_tr = int(train.sum())
    best = None
    for C in C_GRID:
        l2 = 1.0 / (C * n_tr)
        beta = methods.fit_offset_logit(Z[train], offset[train], y[train], l2)
        p = methods.predict_offset_logit(Z[sel], offset[sel], beta)
        ll = -np.mean(y[sel] * np.log(np.clip(p, 1e-9, 1)) + (1 - y[sel]) * np.log(np.clip(1 - p, 1e-9, 1)))
        if best is None or ll < best[0]:
            best = (ll, beta, C)
    _, beta, C = best
    pred_all = methods.predict_offset_logit(Z, offset, beta)
    return pred_all, beta, C, Z


def main() -> None:
    OUT.mkdir(exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=BASE, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", str(HERE / "spec.json")], cwd=BASE,
                           capture_output=True, text=True).stdout.strip()
    df, cand_cols, merge_info = load()
    y, yr = df["top3"].to_numpy(), df["year"].to_numpy()
    train = (yr >= 2016) & (yr <= 2021)
    sel = yr == 2022
    conf = yr == 2023
    exp = (yr >= 2024) & (yr <= 2025)
    pooled = conf | exp
    day = df["day"].to_numpy()

    res = {"experiment_id": SPEC["experiment_id"], "commit": commit, "spec_uncommitted_changes": bool(dirty),
           "inputs": {p: sha(D / p) for p in ["base.parquet", "exp04_candidates.parquet"]},
           "n_rows": int(len(df)), "n_races": int(df.rid16.nunique()), "n_candidates": len(cand_cols),
           "environment_merge_info": merge_info}
    print(f"母集団 {len(df):,} 行 / {df.rid16.nunique():,} R  候補特徴 {len(cand_cols)}  commit={commit[:10]}"
          f"{' (spec未コミット!)' if dirty else ''}")
    for mi in merge_info:
        print(f"  {mi['axis']}: 水準{mi['levels_total']} 統合{mi['levels_merged']} {mi['merged_levels']}")

    # ---- Gate 0 ----
    join_rate = len(df) / len(pd.read_parquet(D / "base.parquet").query("year>=2016 and year<=2025 and n_field>=5"))
    env_na = df[["e1_year", "e2_course", "e3_surfdist"]].isna().mean().max()
    env_counts = {a: int((df.loc[train, a].value_counts() >= 1000).sum())
                 for a in ["e1_year", "e2_course", "e3_surfdist"]}
    g0 = {"join_rate": float(join_rate), "env_label_na_max": float(env_na), "env_levels_ge_1000_train": env_counts}
    g0_ok = join_rate > 0.98 and env_na < 0.01
    res["gate0"] = {"pass": g0_ok, **g0}
    print(f"\n[Gate 0] {'PASS' if g0_ok else 'FAIL'} {g0}")
    if not g0_ok:
        (OUT / "gate_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        return

    # ---- M0, M1 (offset, EXP01-03と同じ evaluate.fit_predict を再利用) ----
    P0, _, _ = fit_predict(df, ["f_mkt"], y, train, sel)
    P1, C1_, coef1 = fit_predict(df, ["f_v6", "f_mkt"], y, train, sel)
    offset = logit(np.clip(P1, 1e-9, 1 - 1e-9))
    res["M1_coef"] = {k: float(v) for k, v in coef1.items()}
    res["M1_C"] = C1_

    # ---- I1: 環境安定性選別 ----
    Xc = {c: df[c].to_numpy(dtype=float) for c in cand_cols}
    print("\n[I1] 環境安定性選別を実行中 (200 bootstrap × 145特徴)...", flush=True)
    stab = methods.env_stability_select(Xc, offset, y, train, df[["e1_year", "e2_course", "e3_surfdist"]],
                                        day, boot_reps=200, seed=0)
    selected = [c for c, v in stab.items() if v["selected"]]
    (OUT / "stability_selection.json").write_text(
        json.dumps(stab, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"  選択: {len(selected)} / {len(cand_cols)}  {selected}")
    res["I1_selected_features"] = selected
    res["I1_n_selected"] = len(selected)

    if len(selected) == 0:
        res["gate1"] = {"pass": False, "reason": "I1が1特徴も選ばなかった"}
        res["verdict"] = "Gate 1 FAIL: 環境間で安定した特徴が1つもない → 中止"
        (OUT / "gate_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(f"\n判定: {res['verdict']}")
        return

    # ---- Gate 1: 追加診断 (重複・offsetの代理・欠損/年度の代理・LOYO選択安定性) ----
    Zsel = design_matrix(df, selected, train)
    corr = np.corrcoef(Zsel[train].T)
    np.fill_diagonal(corr, 0)
    dup_pairs = int((np.abs(np.triu(corr, 1)) > 0.95).sum())
    dup_ok = dup_pairs <= 0.30 * len(selected)
    off_corr = max(abs(np.corrcoef(Zsel[train, i], offset[train])[0, 1]) for i in range(Zsel.shape[1]))
    off_ok = off_corr < 0.95
    year_dummy = pd.get_dummies(df.loc[train, "e1_year"]).to_numpy(dtype=float)
    miss_corr = max(abs(np.corrcoef(Zsel[train, i], year_dummy[:, j])[0, 1])
                    for i in range(Zsel.shape[1]) for j in range(year_dummy.shape[1]))
    miss_ok = miss_corr < 0.7

    print("\n[Gate 1追加] leave-one-year-out での選択数の安定性 (計算量の都合でbootstrap=30に簡略化)...", flush=True)
    loyo_counts = []
    for yy in range(2016, 2022):
        m = train & (yr != yy)
        stab_y = methods.env_stability_select(Xc, offset, y, m, df[["e1_year", "e2_course", "e3_surfdist"]],
                                              day, boot_reps=30, seed=yy)
        loyo_counts.append(sum(1 for v in stab_y.values() if v["selected"]))
    cv = float(np.std(loyo_counts) / (np.mean(loyo_counts) + 1e-9))
    fold_ok = cv < 0.5
    print(f"  年別除外ごとの選択数: {loyo_counts}  CV={cv:.3f}")

    g1 = {"dup_pairs": dup_pairs, "dup_ok": dup_ok, "offset_proxy_corr": float(off_corr), "offset_proxy_ok": off_ok,
          "missingness_year_proxy_corr": float(miss_corr), "missingness_proxy_ok": miss_ok,
          "loyo_selection_counts": loyo_counts, "loyo_cv": cv, "loyo_stable_ok": fold_ok}
    g1_ok = dup_ok and off_ok and miss_ok and fold_ok
    res["gate1"] = {"pass": g1_ok, **g1}
    print(f"\n[Gate 1] {'PASS' if g1_ok else 'FAIL'} {json.dumps(g1, default=str)[:300]}")
    if not g1_ok:
        res["verdict"] = "Gate 1 FAIL: 選択特徴の性質が不適格 → 中止"
        (OUT / "gate_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(f"\n判定: {res['verdict']}")
        return

    # ---- M2, M3, M4, M5 ----
    print("\n[M2] 全候補プール...", flush=True)
    P2, beta2, C2_, Z_all = fit_residual(df, cand_cols, offset, y, train, sel)
    print("[M3] 安定特徴...", flush=True)
    P3, beta3, C3_, _ = fit_residual(df, selected, offset, y, train, sel)
    print("[M4] Group-DRO...", flush=True)
    n_tr = int(train.sum())
    l2_m4 = 1.0 / (C2_ * n_tr)
    group_ids = [df["e1_year"].to_numpy(), df["e2_course"].to_numpy(), df["e3_surfdist"].to_numpy()]
    beta4 = methods.group_dro_fit(Z_all, offset, y, train, group_ids, l2=l2_m4)
    P4 = methods.predict_offset_logit(Z_all, offset, beta4)
    print("[M5] 通常top-k...", flush=True)
    k = len(selected)
    topk = methods.pooled_topk(Xc, offset, y, train, k)
    P5, beta5, C5_, _ = fit_residual(df, topk, offset, y, train, sel)
    res["M5_features"] = topk
    res["C_selected"] = {"M2": C2_, "M3": C3_, "M4(=M2のCを流用)": C2_, "M5": C5_}

    P = {"M0": P0, "M1": P1, "M2": P2, "M3": P3, "M4": P4, "M5": P5}
    rows = [{"model": k_, "period": pn, **metrics(y[m], P[k_][m], df.rid16.to_numpy()[m])}
            for k_ in P for pn, m in {"selection_2022": sel, "confirm_2023": conf, "oos_2024": yr == 2024,
                                      "oos_2025": yr == 2025, "oos_2024_25": exp}.items()]
    mc = pd.DataFrame(rows)
    mc.to_csv(OUT / "model_compare.csv", index=False, encoding="utf-8-sig")
    print("\n[モデル比較 logloss]")
    print(mc.pivot(index="model", columns="period", values="logloss").round(5).to_string())

    # ---- Gate 2A-D ----
    band = pd.cut(df["rank_mkt_pre"], [0, 1, 3, 6, 99], labels=["1", "2-3", "4-6", "7+"]).astype(str).to_numpy()

    def pair(a, b, level):
        out = {"compare": f"{a} vs {b}"}
        for pn, m in [("confirm_2023", conf), ("oos_2024", yr == 2024), ("oos_2025", yr == 2025)]:
            out[pn] = delta_boot(y[m], P[a][m], P[b][m], day[m], level=level)
        d = ll_vec(y, P[a]) - ll_vec(y, P[b])
        e2_levels = df.loc[pooled, "e2_course"].unique()
        e3_levels = df.loc[pooled, "e3_surfdist"].unique()
        e2r = {lv: float(d[pooled & (df.e2_course == lv).to_numpy()].mean()) for lv in e2_levels}
        e3r = {lv: float(d[pooled & (df.e3_surfdist == lv).to_numpy()].mean()) for lv in e3_levels}
        bandr = {bb: float(d[pooled & (band == bb)].mean()) for bb in ["1", "2-3", "4-6", "7+"]}
        rr = pd.DataFrame({"rid": df.rid16, "d": d}).groupby("rid")["d"].sum()
        top100 = set(rr.sort_values().index[:100])
        drop_race = float(d[pooled & ~df.rid16.isin(top100).to_numpy()].mean())
        out.update({"e2_delta": e2r, "e3_delta": e3r, "band_delta": bandr, "drop_top100_race_delta": drop_race})
        return out

    lvl = SPEC["gate2A_additional_info"]["ci_level"]
    g2A = pair("M3", "M1", lvl)
    g2B = pair("M3", "M2", SPEC["gate2B_vs_pooled"]["ci_level"])
    g2C = pair("M3", "M5", SPEC["gate2C_vs_topk"]["ci_level"])
    g2A_pass = g2A["confirm_2023"]["delta"] < 0 and g2A["confirm_2023"]["ci_hi"] < 0
    g2B_pass = g2B["confirm_2023"]["delta"] < 0 and g2B["confirm_2023"]["ci_hi"] < 0
    g2C_pass = g2C["confirm_2023"]["delta"] < 0 and g2C["confirm_2023"]["ci_hi"] < 0
    yr_sign = g2B["oos_2024"]["delta"] < 0 and g2B["oos_2025"]["delta"] < 0
    e2_ok = sum(v < 0 for v in g2B["e2_delta"].values()) >= 0.75 * len(g2B["e2_delta"])
    e3_ok = sum(v < 0 for v in g2B["e3_delta"].values()) >= 0.75 * len(g2B["e3_delta"])
    band_ok = sum(v < 0 for v in g2B["band_delta"].values()) >= 3
    race_ok = g2B["drop_top100_race_delta"] < 0
    g2D_pass = yr_sign and e2_ok and e3_ok and band_ok and race_ok
    print("\n[Gate 2A] M3 vs M1  99%CI" if lvl == 0.99 else "\n[Gate 2A] M3 vs M1")
    print(f"  2023 Δ={g2A['confirm_2023']['delta']:+.6f} [{g2A['confirm_2023']['ci_lo']:+.6f},{g2A['confirm_2023']['ci_hi']:+.6f}] → {'PASS' if g2A_pass else 'fail'}")
    print("[Gate 2B] M3 vs M2")
    print(f"  2023 Δ={g2B['confirm_2023']['delta']:+.6f} [{g2B['confirm_2023']['ci_lo']:+.6f},{g2B['confirm_2023']['ci_hi']:+.6f}] "
          f"| 2024 {g2B['oos_2024']['delta']:+.6f} | 2025 {g2B['oos_2025']['delta']:+.6f} → {'PASS' if g2B_pass else 'fail'}")
    print("[Gate 2C] M3 vs M5")
    print(f"  2023 Δ={g2C['confirm_2023']['delta']:+.6f} [{g2C['confirm_2023']['ci_lo']:+.6f},{g2C['confirm_2023']['ci_hi']:+.6f}] → {'PASS' if g2C_pass else 'fail'}")
    print(f"[Gate 2D] 年度符号一致={yr_sign} 競馬場3/4以上={e2_ok}({sum(v<0 for v in g2B['e2_delta'].values())}/{len(g2B['e2_delta'])}) "
          f"馬場距離帯3/4以上={e3_ok} 人気帯3/4以上={band_ok} 上位100R除外後も負={race_ok} → {'PASS' if g2D_pass else 'fail'}")

    res["gate2A"] = g2A
    res["gate2B"] = g2B
    res["gate2C"] = g2C
    res["gate2D"] = {"pass": g2D_pass, "year_sign_match": yr_sign, "e2_frac_negative": e2_ok,
                     "e3_frac_negative": e3_ok, "band_frac_negative": band_ok, "drop_top100_ok": race_ok}
    res["gate2A_pass"], res["gate2B_pass"], res["gate2C_pass"] = g2A_pass, g2B_pass, g2C_pass

    if not g2A_pass:
        verdict = "Gate 2A FAIL: M3に市場条件付き追加情報なし → 仮説終了"
    elif not (g2B_pass and g2C_pass):
        verdict = "Gate 2A PASS だが 2B/2C FAIL: 追加情報はあるが不変選別固有の価値なし"
    elif not g2D_pass:
        verdict = "Gate 2A/2B/2C PASS だが 2D FAIL: 頑健でない (環境・部分集合依存) → 採用しない"
    else:
        verdict = "Gate 2A-D 全PASS"
    res["gate2_verdict"] = verdict
    print(f"\n判定: {verdict}")

    # ---- robust-only (M4) 判定 ----
    d_m4_m2 = ll_vec(y, P4) - ll_vec(y, P2)
    m4_conf_delta_all = float(d_m4_m2[conf].mean())
    non_inf = SPEC["non_inferiority_margin"]["definition"]
    non_inf_val = 0.0005
    worst_e2_m2 = min({lv: float((ll_vec(y, P2) - ll_vec(y, P0))[conf & (df.e2_course == lv).to_numpy()].mean())
                       for lv in df.loc[conf, "e2_course"].unique()}.values())
    worst_e2_m4 = min({lv: float((ll_vec(y, P4) - ll_vec(y, P0))[conf & (df.e2_course == lv).to_numpy()].mean())
                       for lv in df.loc[conf, "e2_course"].unique()}.values())
    robust_only = (m4_conf_delta_all < non_inf_val) and (worst_e2_m4 < worst_e2_m2)
    res["robust_only"] = {"pass": bool(robust_only), "m4_vs_m2_overall_delta_confirm": m4_conf_delta_all,
                          "non_inferiority_margin": non_inf_val,
                          "worst_e2_loss_vs_m0_M2": worst_e2_m2, "worst_e2_loss_vs_m0_M4": worst_e2_m4,
                          "note": non_inf}
    print(f"\n[robust-only判定(参考, M4)] {res['robust_only']}")

    # ---- leave-one-environment-out 診断 (β再フィットのみ、spec記載の簡略化) ----
    print("\n[leave-one-environment-out 診断] (選択特徴は固定、βだけ再フィット)", flush=True)
    loeo_rows = []
    for axis in ["e1_year", "e2_course", "e3_surfdist"]:
        for lv in sorted(df.loc[train, axis].unique()):
            held = train & (df[axis] == lv).to_numpy()
            fit_m = train & ~held
            if fit_m.sum() < 500 or held.sum() < 50:
                continue
            Zh = design_matrix(df, selected, fit_m)
            l2 = 1.0 / (C3_ * int(fit_m.sum()))
            b3 = methods.fit_offset_logit(Zh[fit_m], offset[fit_m], y[fit_m], l2)
            p3h = methods.predict_offset_logit(Zh[held], offset[held], b3)
            ll1 = -np.mean(y[held] * np.log(np.clip(P1[held], 1e-9, 1)) + (1 - y[held]) * np.log(np.clip(1 - P1[held], 1e-9, 1)))
            ll3 = -np.mean(y[held] * np.log(np.clip(p3h, 1e-9, 1)) + (1 - y[held]) * np.log(np.clip(1 - p3h, 1e-9, 1)))
            loeo_rows.append({"axis": axis, "level": lv, "n_held": int(held.sum()), "n_fit": int(fit_m.sum()),
                             "logloss_M1": ll1, "logloss_M3_refit": ll3, "delta": ll3 - ll1})
    loeo = pd.DataFrame(loeo_rows)
    loeo.to_csv(OUT / "leave_one_env_out.csv", index=False, encoding="utf-8-sig")
    print(loeo.groupby("axis")["delta"].agg(["mean", "count", lambda s: float((s < 0).mean())])
         .rename(columns={"<lambda_0>": "frac_negative"}).round(4).to_string())
    res["leave_one_env_out_summary"] = loeo.groupby("axis")["delta"].agg(
        ["mean", "count"]).reset_index().to_dict("records")

    # ---- Gate 3 (2A-2D 全PASS の場合のみ) ----
    if g2A_pass and g2B_pass and g2C_pass and g2D_pass:
        res["gate3"] = economic(df, P, "M3")
    else:
        res["gate3"] = f"未実施 ({verdict})"
    res["verdict"] = verdict
    (OUT / "gate_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\n書き出し: {OUT}")


def economic(df, P, use):
    yr, fpay = df["year"].to_numpy(), df["fpay"].to_numpy()
    day = df["rid16"].str[:8].to_numpy()
    pooled = (yr >= 2023) & (yr <= 2025)
    sel = {}
    for k in ["M1", "M2", use]:
        p = P[k]
        d = pd.DataFrame({"rid": df.rid16, "p": p})
        r2 = np.zeros(len(df), bool)
        r2[d.groupby("rid").p.idxmax().to_numpy()] = True
        sel[(k, "R1")], sel[(k, "R2")] = p / df["mkt_p3_pre"].to_numpy() >= 1.15, r2
    hon = df["rank_v6"].to_numpy() == 1
    rows = []
    for (k, rule), s in list(sel.items()) + [(("v6◎複勝", "flat"), hon)]:
        for per, m in [("2023", yr == 2023), ("2024", yr == 2024), ("2025", yr == 2025), ("2023-25", pooled)]:
            mm = s & m
            n = int(mm.sum())
            rng = np.random.default_rng(0)
            g = pd.DataFrame({"s": np.where(mm, 100.0, 0), "r": np.where(mm, fpay, 0), "day": day})[m] \
                .groupby("day")[["s", "r"]].sum().sort_index()
            bs = [g.r.to_numpy()[i].sum() / max(g.s.to_numpy()[i].sum(), 1) * 100
                  for i in (rng.integers(0, len(g), len(g)) for _ in range(2000))] if len(g) else [np.nan]
            rows.append({"model": k, "rule": rule, "period": per, "bets": n,
                        "hits": int((fpay[mm] > 0).sum()),
                        "roi": float(fpay[mm].sum() / (100 * n) * 100) if n else np.nan,
                        "roi_ci_lo": float(np.quantile(bs, .025)), "roi_ci_hi": float(np.quantile(bs, .975))})
    e = pd.DataFrame(rows)
    e.to_csv(OUT / "economic.csv", index=False, encoding="utf-8-sig")
    print("\n[Gate 3]")
    print(e[e.period == "2023-25"].round(2).to_string(index=False))
    return e.to_dict("records")


if __name__ == "__main__":
    main()
