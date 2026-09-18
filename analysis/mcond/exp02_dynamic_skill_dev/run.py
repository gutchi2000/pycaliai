# -*- coding: utf-8 -*-
"""
run.py — EXP02 の Gate 0〜3 と副次評価を spec.json どおりに実行する
====================================================================
入力: data/_research/mcond/base.parquet, exp02_features.parquet, data/elo_feats.parquet,
      data/glicko_feats.parquet, (昇級・降級の副次評価のみ) exp01_features.parquet
出力: out/gate_results.json, out/gate1.csv, out/model_compare.csv, out/deltas.csv,
      out/by_popband.csv, out/by_year.csv, out/secondary.csv, out/economic*.csv
実行: python -m analysis.mcond.exp02_dynamic_skill_dev.run
"""
from __future__ import annotations
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
from analysis.mcond.evaluate import fit_predict, metrics, delta_boot, logit, ll_vec  # noqa: E402
from grade_feats import class_name_to_ord  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
D = BASE / "data/_research/mcond"
SPEC = json.loads((HERE / "spec.json").read_text(encoding="utf-8"))
PLACE = {"01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京", "06": "中山",
         "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def load():
    b = pd.read_parquet(D / "base.parquet")
    f = pd.read_parquet(D / "exp02_features.parquet").drop(columns=["date"])
    f["rid16"] = f["rid16"].astype(str)
    e = pd.read_parquet(BASE / "data/elo_feats.parquet")
    g = pd.read_parquet(BASE / "data/glicko_feats.parquet")
    df = b.merge(f, on=["rid16", "ban"], how="inner").merge(e, on=["rid16", "ban"], how="left") \
          .merge(g, on=["rid16", "ban"], how="left")
    y0, y1 = SPEC["population"]["years"]
    df = df[(df.year >= y0) & (df.year <= y1) & (df.n_field >= 5)
            & df.mkt_p3_pre.notna() & df.v6_p3.notna()].copy()
    df["f_v6"] = logit(df["v6_p3"])
    df["f_mkt"] = logit(df["mkt_p3_pre"])
    df["disagree"] = ((df["rank_v6"] - df["rank_mkt_pre"]).abs() >= 3).astype(float)
    df["i_dis_sigma"] = df["disagree"] * df["dyn_skill_sigma"]
    df["i_dis_mf"] = df["disagree"] * df["horse_skill_minus_field"]
    df["i_mkt_sigma"] = df["f_mkt"] * df["dyn_skill_sigma"]
    df["day"] = df["rid16"].str[:8]
    return df.reset_index(drop=True)


def gate1(df_all_feats: pd.DataFrame) -> tuple[bool, list]:
    """全期間の特徴 (評価母集団に限らない) で能力表現の成立を確認。"""
    from analysis.mcond.exp02_dynamic_skill_dev.dyn_skill import load_runs, MU0
    f = df_all_feats.copy()
    runs = load_runs()[["rid16", "ban", "hid", "date", "fin"]]
    f = f.merge(runs[["rid16", "ban", "fin"]], on=["rid16", "ban"], how="left")
    f = f.sort_values(["hid", "date"])
    f["prev_fin"] = f.groupby("hid")["fin"].shift(1)
    chk = []
    # G1a
    x = f[f.prev_fin.between(1, 10)].groupby("prev_fin")["dyn_skill_last_change"].mean()
    rho = spearmanr(x.index, x.values).correlation
    ok = x.loc[1] > 0 and x.loc[10] < 0 and rho <= -0.9
    chk.append(("G1a 着順で平均が動く", {"by_prev_fin": x.round(4).to_dict(), "spearman": rho}, ok))
    # G1b
    bins = pd.cut(f["dyn_skill_num_updates"], [-1, 0, 1, 2, 5, 10, 20, 1e9],
                  labels=["0", "1", "2", "3-5", "6-10", "11-20", "21+"])
    recent = f["dyn_skill_days_since_update"].isna() | (f["dyn_skill_days_since_update"] <= 60)
    med = f[recent].groupby(bins[recent], observed=True)["dyn_skill_sigma"].median()
    ok = bool((np.diff(med.values) <= 1e-9).all())
    chk.append(("G1b 履歴で不確実性が減る", med.round(4).to_dict(), ok))
    # G1c
    mid = f.dyn_skill_num_updates.between(6, 10)
    a = f[mid & (f.dyn_skill_days_since_update > 180)].dyn_skill_sigma.median()
    c = f[mid & (f.dyn_skill_days_since_update <= 60)].dyn_skill_sigma.median()
    chk.append(("G1c 長期休養で不確実性が増える", {"gt180": a, "le60": c}, bool(a > c)))
    # G1d
    new = f.dyn_skill_num_updates == 0
    pct = f.loc[new, "horse_skill_percentile"].mean()
    ok = bool(np.allclose(f.loc[new, "dyn_skill_mu"], MU0) and 0.2 <= pct <= 0.8)
    chk.append(("G1d 新馬が極端値でない", {"mean_percentile": pct}, ok))
    # G1e
    m = pd.read_csv(BASE / "data/master_v2_20130105-20251228.csv", encoding="utf-8-sig", low_memory=False,
                    usecols=["レースID(新)", "馬番", "前走確定着順", "クラス名"])
    m["rid16"] = m["レースID(新)"].astype(str).str[:16]
    m["ban"] = pd.to_numeric(m["馬番"], errors="coerce")
    m["cls"] = m["クラス名"].map(class_name_to_ord)
    m["pf"] = pd.to_numeric(m["前走確定着順"], errors="coerce")
    ff = f.merge(m[["rid16", "ban", "cls", "pf"]], on=["rid16", "ban"], how="left")
    ff = ff[ff.dyn_skill_num_updates > 0]
    rs = {k: abs(spearmanr(ff.dyn_skill_mu, ff[k], nan_policy="omit").correlation)
          for k in ["pf", "cls", "dyn_skill_num_updates"]}
    X = ff[["pf", "cls", "dyn_skill_num_updates"]].fillna(ff[["pf", "cls", "dyn_skill_num_updates"]].mean())
    X = np.column_stack([np.ones(len(X)), X.to_numpy()])
    yv = ff.dyn_skill_mu.to_numpy()
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    r2 = 1 - ((yv - X @ beta) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
    ok = all(v < 0.8 for v in rs.values()) and r2 < 0.8
    chk.append(("G1e 直近着順・クラス・出走数の写しでない", {"abs_spearman": rs, "r2": r2}, bool(ok)))
    # G1f
    w = f[(f.fin == 1) & (f.date.dt.year >= 2016)]
    by_y = w.groupby(w.date.dt.year)["horse_skill_minus_field"].median()
    by_p = w.groupby(w.rid16.str[8:10])["horse_skill_minus_field"].median()
    cv_y, cv_p = by_y.std() / abs(by_y.mean()), by_p.std() / abs(by_p.mean())
    chk.append(("G1f 年・競馬場で尺度が崩れない", {"cv_year": cv_y, "cv_place": cv_p,
                                             "by_year": by_y.round(3).to_dict()}, bool(cv_y < 0.3 and cv_p < 0.3)))
    return all(c[2] for c in chk), chk


def main() -> None:
    OUT.mkdir(exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=BASE, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", str(HERE / "spec.json")], cwd=BASE,
                           capture_output=True, text=True).stdout.strip()
    allf = pd.read_parquet(D / "exp02_features.parquet")
    df = load()
    res = {"experiment_id": SPEC["experiment_id"], "commit": commit, "spec_uncommitted_changes": bool(dirty),
           "inputs": {p: sha(D / p) for p in ["base.parquet", "exp02_features.parquet"]},
           "build_meta": json.loads((OUT / "build_meta.json").read_text(encoding="utf-8"))["selected"],
           "n_rows": int(len(df)), "n_races": int(df.rid16.nunique())}
    print(f"母集団 {len(df):,} 行 / {df.rid16.nunique():,} R  commit={commit[:10]}{' (spec未コミット!)' if dirty else ''}")

    # ---- Gate 0 ----
    cov_elo = float(df["elo_T1M2_horse"].notna().mean())
    cov_g = float(df["g2_mu"].notna().mean())
    base_n = pd.read_parquet(D / "base.parquet")
    base_n = base_n[(base_n.year >= 2016) & (base_n.n_field >= 5)]
    join_rate = len(df) / len(base_n)
    g0 = {"join_rate_vs_base": join_rate, "coverage_elo": cov_elo, "coverage_glicko": cov_g,
          "warmup_years": [2013, 2015], "time_safety_tests": "test_time_safety.py 4本 / test_boundary.py 7本 (実行ログは README)"}
    g0_ok = join_rate > 0.98 and cov_elo > 0.98 and cov_g > 0.98
    res["gate0"] = {"pass": g0_ok, **g0}
    print(f"\n[Gate 0] {'PASS' if g0_ok else 'FAIL'} {g0}")
    if not g0_ok:
        (OUT / "gate_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        return

    # ---- Gate 1 ----
    g1_ok, chk = gate1(allf)
    pd.DataFrame([{"check": c[0], "value": json.dumps(c[1], ensure_ascii=False, default=str), "pass": c[2]}
                  for c in chk]).to_csv(OUT / "gate1.csv", index=False, encoding="utf-8-sig")
    res["gate1"] = {"pass": g1_ok, "checks": [{"check": c[0], "value": c[1], "pass": c[2]} for c in chk]}
    print(f"\n[Gate 1] {'PASS' if g1_ok else 'FAIL'}")
    for c in chk:
        print(f"  {'PASS' if c[2] else 'FAIL'}  {c[0]}: {json.dumps(c[1], ensure_ascii=False, default=str)[:220]}")
    if not g1_ok:
        res["verdict"] = "Gate 1 FAIL → 中止"
        (OUT / "gate_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        return

    F = SPEC["features"]
    T1, T2, EX = F["T1"], F["T2"], F["existing_rating"]
    base = ["f_v6", "f_mkt"]
    MODELS = {"M0": ["f_mkt"], "M1": base, "M2": base + EX, "M3": base + T1, "M4": base + EX + T1,
              "M5": base + T1 + T2, "M6": base + T1 + ["i_dis_sigma", "i_dis_mf", "i_mkt_sigma"]}
    y, yr = df["top3"].to_numpy(), df["year"].to_numpy()
    train, sel = (yr >= 2016) & (yr <= 2021), yr == 2022
    conf, exp = yr == 2023, (yr >= 2024) & (yr <= 2025)
    pooled = conf | exp
    day = df["day"].to_numpy()
    P, CS, COEF = {}, {}, {}
    for k, cols in MODELS.items():
        P[k], CS[k], COEF[k] = fit_predict(df, cols, y, train, sel)
    res["C_selected"] = CS
    res["coef"] = {k: {kk: float(vv) for kk, vv in v.items()} for k, v in COEF.items()}

    rows = []
    for k in MODELS:
        for pn, m in {"selection_2022": sel, "confirm_2023": conf, "oos_2024": yr == 2024,
                      "oos_2025": yr == 2025, "oos_2024_25": exp}.items():
            rows.append({"model": k, "period": pn, **metrics(y[m], P[k][m], df.rid16.to_numpy()[m])})
    mc = pd.DataFrame(rows)
    mc.to_csv(OUT / "model_compare.csv", index=False, encoding="utf-8-sig")
    print("\n[モデル比較 logloss]")
    print(mc.pivot(index="model", columns="period", values="logloss").round(5).to_string())

    # ---- Gate 2 ----
    band = pd.cut(df["rank_mkt_pre"], [0, 1, 3, 6, 99], labels=["1", "2-3", "4-6", "7+"]).astype(str).to_numpy()
    nupd = df["dyn_skill_num_updates"].to_numpy()
    hid = df["hid"].to_numpy()

    def pair(a, b, level=0.975):
        out = {"compare": f"{a} vs {b}"}
        for pn, m in [("confirm_2023", conf), ("oos_2024", yr == 2024), ("oos_2025", yr == 2025)]:
            out[pn] = delta_boot(y[m], P[a][m], P[b][m], day[m], level=level)
        d = ll_vec(y, P[a]) - ll_vec(y, P[b])
        bands = {bb: float(d[pooled & (band == bb)].mean()) for bb in ["1", "2-3", "4-6", "7+"]}
        t = pd.DataFrame({"d": d[pooled], "h": hid[pooled]})
        drop = set(t.groupby("h")["d"].sum().sort_values().index[:100])
        rest_h = float(t[~t.h.isin(drop)]["d"].mean())
        rest_hist = float(d[pooled & (nupd > 2)].mean())
        c = [out["confirm_2023"]["delta"] < 0 and out["confirm_2023"]["ci_hi"] < 0,
             out["oos_2024"]["delta"] < 0 and out["oos_2025"]["delta"] < 0,
             sum(v < 0 for v in bands.values()) >= 3, rest_h < 0, rest_hist < 0]
        out.update({"popband_delta": bands, "drop_top100_horses_delta": rest_h,
                    "exclude_nupd_le2_delta": rest_hist,
                    "conditions": dict(zip(["confirm_ci", "year_signs", "popband", "few_horses", "history"], c)),
                    "pass": all(c)})
        return out

    comps = [pair("M3", "M1"), pair("M4", "M2"), pair("M2", "M1"), pair("M5", "M1"), pair("M6", "M1"),
             pair("M5", "M3"), pair("M6", "M3"), pair("M1", "M0")]
    res["gate2"] = comps
    dr = [{"compare": c["compare"], "period": pn, **c[pn]} for c in comps
          for pn in ["confirm_2023", "oos_2024", "oos_2025"]]
    pd.DataFrame(dr).to_csv(OUT / "deltas.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"compare": c["compare"], "band": k, "delta": v} for c in comps
                  for k, v in c["popband_delta"].items()]).to_csv(OUT / "by_popband.csv", index=False, encoding="utf-8-sig")
    print("\n[Gate 2] Δlogloss (負=左が良い), 97.5% CI")
    for c in comps:
        c3 = c["confirm_2023"]
        tag = "(主)" if c["compare"] in ("M3 vs M1", "M4 vs M2") else "(副)"
        print(f"  {tag}{c['compare']:<9} 2023 Δ={c3['delta']:+.6f} [{c3['ci_lo']:+.6f},{c3['ci_hi']:+.6f}] "
              f"| 2024 {c['oos_2024']['delta']:+.6f} | 2025 {c['oos_2025']['delta']:+.6f} "
              f"| 上位100頭除外 {c['drop_top100_horses_delta']:+.6f} | 履歴≤2除外 {c['exclude_nupd_le2_delta']:+.6f} "
              f"| 人気帯 {', '.join(f'{k}:{v:+.5f}' for k, v in c['popband_delta'].items())} → {'PASS' if c['pass'] else 'fail'}")
    p31, p42 = comps[0]["pass"], comps[1]["pass"]
    p51, p61 = comps[3]["pass"], comps[4]["pass"]
    if p31 and p42:
        verdict = "EXP02 固有の追加情報あり (M3>M1 かつ M4>M2)"
    elif p31:
        verdict = "能力系情報は有効だが EXP02 固有の価値はない (M3>M1, M4≯M2)"
    elif p51:
        verdict = "T2 のみ: 条件別の部分プーリングに価値がある可能性 (探索扱い)"
    elif p61:
        verdict = "M6 のみ: 主仮説の成功とせず探索結果"
    else:
        verdict = "Gate 2 FAIL: 動的能力に市場条件付きの追加情報なし"
    res["gate2_verdict"] = verdict
    print(f"\n判定: {verdict}")

    # ---- 副次 ----
    sec = []
    for a, b in [("M3", "M1"), ("M4", "M2")]:
        dd = ll_vec(y, P[a]) - ll_vec(y, P[b])
        for lab, m in [("履歴0", nupd == 0), ("履歴1-2", (nupd >= 1) & (nupd <= 2)),
                       ("履歴3-5", (nupd >= 3) & (nupd <= 5)), ("履歴6-10", (nupd >= 6) & (nupd <= 10)),
                       ("履歴11+", nupd >= 11)]:
            mm = pooled & m
            sec.append({"analysis": "出走履歴数別", "compare": f"{a} vs {b}", "subset": lab, "n": int(mm.sum()),
                        **delta_boot(y[mm], P[a][mm], P[b][mm], day[mm], level=0.95)})
        ds = df["dyn_skill_days_since_update"].to_numpy()
        mm = pooled & (nupd > 0) & ~(ds > 180)
        sec.append({"analysis": "感度: 新馬と180日超休養明けを除外", "compare": f"{a} vs {b}", "subset": "",
                    "n": int(mm.sum()), **delta_boot(y[mm], P[a][mm], P[b][mm], day[mm], level=0.95)})
        mm = pooled & (df["disagree"].to_numpy() == 1)
        sec.append({"analysis": "AI・市場不一致の馬", "compare": f"{a} vs {b}", "subset": "", "n": int(mm.sum()),
                    **delta_boot(y[mm], P[a][mm], P[b][mm], day[mm], level=0.95)})
    e1 = pd.read_parquet(D / "exp01_features.parquet")[["rid16", "ban", "raw_cls_chg"]]
    e1["rid16"] = e1["rid16"].astype(str)
    cl = df[["rid16", "ban"]].merge(e1, on=["rid16", "ban"], how="left")["raw_cls_chg"].to_numpy()
    for a, b in [("M3", "M1"), ("M4", "M2")]:
        for lab, v in [("降級", -1), ("同級", 0), ("昇級", 1)]:
            mm = pooled & (cl == v)
            sec.append({"analysis": "昇級・降級別", "compare": f"{a} vs {b}", "subset": lab, "n": int(mm.sum()),
                        **delta_boot(y[mm], P[a][mm], P[b][mm], day[mm], level=0.95)})
    ywin = df["win"].to_numpy()
    PW = {k: fit_predict(df, MODELS[k], ywin, train, sel)[0] for k in ["M1", "M2", "M3", "M4"]}
    for a, b in [("M3", "M1"), ("M4", "M2")]:
        for pn, m in [("confirm_2023", conf), ("oos_2024_25", exp)]:
            sec.append({"analysis": "目的変数=1着", "compare": f"{a} vs {b}", "subset": pn, "n": int(m.sum()),
                        **delta_boot(ywin[m], PW[a][m], PW[b][m], day[m], level=0.95)})
    # 順位指標
    fin = df["fin"].to_numpy()
    for a, b in [("M3", "M1"), ("M4", "M2")]:
        for pn, m in [("confirm_2023", conf), ("oos_2024_25", exp)]:
            t = pd.DataFrame({"rid": df.rid16[m], "day": day[m], "pa": P[a][m], "pb": P[b][m], "fin": fin[m]})
            rr = []
            for rid, g in t.groupby("rid"):
                if len(g) < 3:
                    continue
                rr.append((g.day.iloc[0], spearmanr(g.pa, -g.fin).correlation - spearmanr(g.pb, -g.fin).correlation))
            r = pd.DataFrame(rr, columns=["day", "d"]).groupby("day")["d"].agg(["sum", "count"])
            rng = np.random.default_rng(5)
            bs = [r["sum"].to_numpy()[i].sum() / r["count"].to_numpy()[i].sum()
                  for i in (rng.integers(0, len(r), len(r)) for _ in range(2000))]
            sec.append({"analysis": "レース内順位相関 Δ (正=左が良い)", "compare": f"{a} vs {b}", "subset": pn,
                        "n": int(r["count"].sum()), "delta": float(r["sum"].sum() / r["count"].sum()),
                        "ci_lo": float(np.quantile(bs, .025)), "ci_hi": float(np.quantile(bs, .975))})
    sdf = pd.DataFrame(sec)
    sdf.to_csv(OUT / "secondary.csv", index=False, encoding="utf-8-sig")
    print("\n[副次評価] 95%CI")
    for r in sec:
        print(f"  {r['analysis'][:22]:<22} {r['compare']:<9} {str(r['subset']):<12} n={r['n']:>6} "
              f"Δ={r['delta']:+.6f} [{r['ci_lo']:+.6f},{r['ci_hi']:+.6f}]")

    by = [{"year": int(y0), "model": k, **metrics(y[yr == y0], P[k][yr == y0], df.rid16.to_numpy()[yr == y0])}
          for y0 in sorted(set(yr)) for k in ["M0", "M1", "M2", "M3", "M4", "M5"]]
    pd.DataFrame(by).to_csv(OUT / "by_year.csv", index=False, encoding="utf-8-sig")

    # ---- Gate 3 ----
    if p31:
        use = "M4" if p42 else "M3"
        res["gate3"] = economic(df, P, use)
    else:
        res["gate3"] = "未実施 (M3 vs M1 が FAIL のため)"
    res["verdict"] = verdict
    (OUT / "gate_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\n書き出し: {OUT}")


def economic(df, P, use):
    yr, fpay = df["year"].to_numpy(), df["fpay"].to_numpy()
    day = df["rid16"].str[:8].to_numpy()
    band = pd.cut(df["rank_mkt_pre"], [0, 1, 3, 6, 99], labels=["1", "2-3", "4-6", "7+"]).astype(str).to_numpy()
    pooled = (yr >= 2023) & (yr <= 2025)
    sel = {}
    for k in ["M1", use]:
        p = P[k]
        d = pd.DataFrame({"rid": df.rid16, "p": p})
        r2 = np.zeros(len(df), bool)
        r2[d.groupby("rid").p.idxmax().to_numpy()] = True
        sel[(k, "R1")] = p / df["mkt_p3_pre"].to_numpy() >= 1.15
        sel[(k, "R2")] = r2
    hon, fav = df["rank_v6"].to_numpy() == 1, df["rank_mkt_pre"].to_numpy() == 1
    rows = []
    for (k, rule), s in list(sel.items()) + [(("v6◎複勝", "flat"), hon), (("市場1番人気複勝", "flat"), fav)]:
        for per, m in [("2023", yr == 2023), ("2024", yr == 2024), ("2025", yr == 2025), ("2023-25", pooled)]:
            mm = s & m
            n = int(mm.sum())
            g = pd.DataFrame({"s": np.where(mm, 100.0, 0), "r": np.where(mm, fpay, 0), "day": day})[m] \
                .groupby("day")[["s", "r"]].sum().sort_index()
            cum = (g.r - g.s).cumsum()
            prof = pd.DataFrame({"rid": df.rid16[mm], "pl": fpay[mm] - 100}).groupby("rid")["pl"].sum()
            rng = np.random.default_rng(0)
            bs = [g.r.to_numpy()[i].sum() / max(g.s.to_numpy()[i].sum(), 1) * 100
                  for i in (rng.integers(0, len(g), len(g)) for _ in range(2000))]
            row = {"model": k, "rule": rule, "period": per, "bets": n, "hits": int((fpay[mm] > 0).sum()),
                   "stake": 100 * n, "return": float(fpay[mm].sum()),
                   "roi": float(fpay[mm].sum() / (100 * n) * 100) if n else np.nan,
                   "roi_ci_lo": float(np.quantile(bs, .025)), "roi_ci_hi": float(np.quantile(bs, .975)),
                   "max_drawdown": float((cum - cum.cummax()).min()),
                   "top10_share": float(prof.sort_values(ascending=False).head(10).clip(lower=0).sum() / max(fpay[mm].sum(), 1))}
            for bb in ["1", "2-3", "4-6", "7+"]:
                mb = mm & (band == bb)
                row[f"roi_{bb}"] = float(fpay[mb].sum() / (100 * mb.sum()) * 100) if mb.sum() else np.nan
            rows.append(row)
    e = pd.DataFrame(rows)
    e.to_csv(OUT / "economic.csv", index=False, encoding="utf-8-sig")
    diffs = []
    for rule in ["R1", "R2"]:
        a, b = sel[(use, rule)] & pooled, sel[("M1", rule)] & pooled
        g = pd.DataFrame({"day": day, "sa": np.where(a, 100.0, 0), "ra": np.where(a, fpay, 0),
                          "sb": np.where(b, 100.0, 0), "rb": np.where(b, fpay, 0)})[pooled].groupby("day").sum()
        rng = np.random.default_rng(1)
        bs = []
        for _ in range(2000):
            x = g.iloc[rng.integers(0, len(g), len(g))].sum()
            bs.append(x.ra / x.sa * 100 - x.rb / x.sb * 100)
        x = g.sum()
        diffs.append({"rule": rule, "compare": f"{use} - M1", "roi_diff_pt": float(x.ra / x.sa * 100 - x.rb / x.sb * 100),
                      "ci_lo": float(np.quantile(bs, .025)), "ci_hi": float(np.quantile(bs, .975))})
    pd.DataFrame(diffs).to_csv(OUT / "economic_diff.csv", index=False, encoding="utf-8-sig")
    print("\n[Gate 3]")
    print(e[e.period == "2023-25"][["model", "rule", "bets", "roi", "roi_ci_lo", "roi_ci_hi", "max_drawdown", "top10_share"]].round(2).to_string(index=False))
    for d in diffs:
        print(f"  {d['rule']} {d['compare']}: {d['roi_diff_pt']:+.2f}pt [{d['ci_lo']:+.2f},{d['ci_hi']:+.2f}]")
    return {"table": e.to_dict("records"), "diff": diffs}


if __name__ == "__main__":
    main()
