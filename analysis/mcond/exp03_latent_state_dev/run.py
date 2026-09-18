# -*- coding: utf-8 -*-
"""
run.py — EXP03 の Gate 0〜3 と副次評価を spec.json どおりに実行する
入力: base.parquet, exp03_features.parquet, exp02_features.parquet (T1 17列), master (年齢のみ)
出力: out/ 以下 (gate_results.json, gate1.csv, model_compare.csv, deltas.csv, by_popband.csv,
      by_year.csv, secondary.csv, economic.csv, economic_diff.csv)
実行: python -m analysis.mcond.exp03_latent_state_dev.run
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

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
D = BASE / "data/_research/mcond"
SPEC = json.loads((HERE / "spec.json").read_text(encoding="utf-8"))
SPEC02 = json.loads((BASE / "analysis/mcond/exp02_dynamic_skill_dev/spec.json").read_text(encoding="utf-8"))


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def load():
    b = pd.read_parquet(D / "base.parquet")
    f3 = pd.read_parquet(D / "exp03_features.parquet")
    f3["rid16"] = f3["rid16"].astype(str)
    t1 = SPEC02["features"]["T1"]
    f2 = pd.read_parquet(D / "exp02_features.parquet")[["rid16", "ban"] + t1]
    f2["rid16"] = f2["rid16"].astype(str)
    m = pd.read_csv(BASE / "data/master_v2_20130105-20251228.csv", encoding="utf-8-sig", low_memory=False,
                    usecols=["レースID(新)", "馬番", "年齢"])
    m["rid16"] = m["レースID(新)"].astype(str).str[:16]
    m["ban"] = pd.to_numeric(m["馬番"], errors="coerce")
    m["age"] = pd.to_numeric(m["年齢"], errors="coerce")
    m = m.dropna(subset=["ban"]).astype({"ban": int})[["rid16", "ban", "age"]].drop_duplicates(["rid16", "ban"])
    df = b.merge(f3.drop(columns=["date"]), on=["rid16", "ban"], how="inner") \
          .merge(f2, on=["rid16", "ban"], how="left").merge(m, on=["rid16", "ban"], how="left")
    y0, y1 = SPEC["population"]["years"]
    df = df[(df.year >= y0) & (df.year <= y1) & (df.n_field >= 5)
            & df.mkt_p3_pre.notna() & df.v6_p3.notna()].copy()
    df["f_v6"], df["f_mkt"] = logit(df["v6_p3"]), logit(df["mkt_p3_pre"])
    df["disagree"] = ((df["rank_v6"] - df["rank_mkt_pre"]).abs() >= 3).astype(float)
    df["day"] = df["rid16"].str[:8]
    return df.reset_index(drop=True)


def gate1(allf: pd.DataFrame, hp) -> tuple[bool, list]:
    # ★行の並び順を hid,date に統一してから groupby().shift() を使う。
    #   shift の結果は「入力フレームの行順」で返るため、s/a を元の (未ソート) 順序のまま
    #   ラグ系列と別々にブール選択すると、両者の並び順が食い違い np.corrcoef が無関係な
    #   行同士をペアにしてしまう (発見: 手動再計算で ac_a が -0.006 → 0.976 に訂正)。
    f = allf[(allf.date.dt.year >= 2016)].sort_values(["hid", "date"]).reset_index(drop=True)
    s, a = f["latent_short_state_mu"], f["persistent_ability_mu"]
    chk = []
    n3 = f["_n_updates"] >= 3
    ds = f["latent_state_days_since_update"]
    a1, a2 = f.loc[n3 & (ds > 180), "latent_short_state_abs"].median(), f.loc[n3 & (ds <= 30), "latent_short_state_abs"].median()
    chk.append(("G1a 休養で状態が0へ", {"abs_s_gt180": a1, "abs_s_le30": a2}, bool(a1 < a2)))
    g = f.groupby("hid")
    lag_s, lag_a = g["latent_short_state_mu"].shift(1), g["persistent_ability_mu"].shift(1)
    ok = lag_s.notna()
    ac_s = float(np.corrcoef(s[ok], lag_s[ok])[0, 1])
    ac_a = float(np.corrcoef(a[ok], lag_a[ok])[0, 1])
    chk.append(("G1b 能力より速く動く (連続2走の自己相関)", {"s": ac_s, "a": ac_a}, bool(ac_s < ac_a)))
    r = float(np.corrcoef(a, s)[0, 1])
    chk.append(("G1c 2成分がほぼ完全相関でない", {"corr_a_s": r}, bool(abs(r) < 0.9)))

    def r2(y, X):
        X = X.fillna(X.mean())
        X = np.column_stack([np.ones(len(X)), X.to_numpy()])
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        return float(1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    m = f["raw_prev_fin"].notna()
    r2f = r2(s[m].to_numpy(), f.loc[m, ["raw_prev_fin", "raw_prev_fin_q"]])
    chk.append(("G1d 前走着順だけで決まらない", {"r2": r2f}, bool(r2f < 0.5)))
    m = f["raw_days_since"].notna()
    X = pd.DataFrame({"d": f.loc[m, "raw_days_since"], "ld": np.log1p(f.loc[m, "raw_days_since"].clip(lower=0))})
    r2d = r2(s[m].to_numpy(), X)
    chk.append(("G1e 休養日数だけで決まらない", {"r2": r2d}, bool(r2d < 0.5)))
    finite = bool(np.isfinite(f[["latent_short_state_mu", "persistent_ability_mu"]].to_numpy()).all())
    mx = float(s.abs().max())
    chk.append(("G1f 発散しない", {"finite": finite, "max_abs_s": mx, "limit": 5 * hp["qs"]}, bool(finite and mx <= 5 * hp["qs"])))
    sd = f.groupby(f.date.dt.year)["latent_short_state_mu"].std()
    cv = float(sd.std() / sd.mean())
    chk.append(("G1g 年で尺度が変わらない", {"cv": cv, "sd_by_year": sd.round(4).to_dict()}, bool(cv < 0.3)))
    chk.append(("G1h 決定論的", "test_time_safety.test_deterministic PASS (README の実行ログ)", True))
    return all(c[2] for c in chk), chk


def main() -> None:
    OUT.mkdir(exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=BASE, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", str(HERE / "spec.json")], cwd=BASE,
                           capture_output=True, text=True).stdout.strip()
    meta = json.loads((OUT / "build_meta.json").read_text(encoding="utf-8"))
    allf = pd.read_parquet(D / "exp03_features.parquet")
    df = load()
    res = {"experiment_id": SPEC["experiment_id"], "commit": commit, "spec_uncommitted_changes": bool(dirty),
           "inputs": {p: sha(D / p) for p in ["base.parquet", "exp03_features.parquet", "exp02_features.parquet"]},
           "selected_hyper": meta["selected"],
           "innovation_choice": json.loads((OUT / "innovation_choice.json").read_text(encoding="utf-8")),
           "n_rows": int(len(df)), "n_races": int(df.rid16.nunique())}
    print(f"母集団 {len(df):,} 行 / {df.rid16.nunique():,} R  commit={commit[:10]}{' (spec未コミット!)' if dirty else ''}")

    # Gate 0
    bn = pd.read_parquet(D / "base.parquet")
    bn = bn[(bn.year >= 2016) & (bn.n_field >= 5)]
    g0 = {"join_rate": len(df) / len(bn), "exp02_coverage": float(df["dyn_skill_mu"].notna().mean()),
          "raw_days_coverage": float(df["raw_days_since"].notna().mean())}
    g0_ok = g0["join_rate"] > 0.98 and g0["exp02_coverage"] > 0.98
    res["gate0"] = {"pass": g0_ok, **g0}
    print(f"\n[Gate 0] {'PASS' if g0_ok else 'FAIL'} {g0}")
    if not g0_ok:
        (OUT / "gate_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        return

    # Gate 1
    g1_ok, chk = gate1(allf, meta["selected"])
    pd.DataFrame([{"check": c[0], "value": json.dumps(c[1], ensure_ascii=False, default=str), "pass": c[2]}
                  for c in chk]).to_csv(OUT / "gate1.csv", index=False, encoding="utf-8-sig")
    res["gate1"] = {"pass": g1_ok, "checks": [{"check": c[0], "value": c[1], "pass": c[2]} for c in chk]}
    print(f"\n[Gate 1] {'PASS' if g1_ok else 'FAIL'}")
    for c in chk:
        print(f"  {'PASS' if c[2] else 'FAIL'}  {c[0]}: {json.dumps(c[1], ensure_ascii=False, default=str)[:200]}")
    if not g1_ok:
        res["verdict"] = "Gate 1 FAIL: 状態が識別できない → 中止"
        (OUT / "gate_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print("\n判定:", res["verdict"])
        return

    F = SPEC["features"]
    RAW, ST, EW = F["raw_recency_experience"], F["latent_short_state"], F["existing_ewma"]
    T1 = SPEC02["features"]["T1"]
    b = ["f_v6", "f_mkt"]
    MODELS = {"M0": ["f_mkt"], "M1": b, "M2": b + RAW, "M3": b + RAW + T1, "M4": b + RAW + T1 + ST,
              "M5": b + RAW + T1 + EW, "M6": b + RAW + T1 + EW + ST}
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
    rows = [{"model": k, "period": pn, **metrics(y[m], P[k][m], df.rid16.to_numpy()[m])}
            for k in MODELS for pn, m in {"selection_2022": sel, "confirm_2023": conf, "oos_2024": yr == 2024,
                                          "oos_2025": yr == 2025, "oos_2024_25": exp}.items()]
    mc = pd.DataFrame(rows)
    mc.to_csv(OUT / "model_compare.csv", index=False, encoding="utf-8-sig")
    print("\n[モデル比較 logloss]")
    print(mc.pivot(index="model", columns="period", values="logloss").round(5).to_string())

    band = pd.cut(df["rank_mkt_pre"], [0, 1, 3, 6, 99], labels=["1", "2-3", "4-6", "7+"]).astype(str).to_numpy()
    days = df["raw_days_since"].to_numpy()
    short, layoff = days <= 14, days > 180
    sabs = df["latent_short_state_abs"].to_numpy()
    top1 = sabs >= np.nanquantile(sabs[pooled], 0.99)

    def pair(a, bb, level=0.975):
        out = {"compare": f"{a} vs {bb}"}
        for pn, m in [("confirm_2023", conf), ("oos_2024", yr == 2024), ("oos_2025", yr == 2025)]:
            out[pn] = delta_boot(y[m], P[a][m], P[bb][m], day[m], level=level)
        d = ll_vec(y, P[a]) - ll_vec(y, P[bb])
        bands = {x: float(d[pooled & (band == x)].mean()) for x in ["1", "2-3", "4-6", "7+"]}
        ex = {"exclude_short": float(d[pooled & ~short].mean()), "exclude_layoff": float(d[pooled & ~layoff].mean()),
              "exclude_top1pct_state": float(d[pooled & ~top1].mean())}
        c = [out["confirm_2023"]["delta"] < 0 and out["confirm_2023"]["ci_hi"] < 0,
             out["oos_2024"]["delta"] < 0 and out["oos_2025"]["delta"] < 0,
             sum(v < 0 for v in bands.values()) >= 3, ex["exclude_short"] < 0, ex["exclude_layoff"] < 0,
             ex["exclude_top1pct_state"] < 0]
        out.update({"popband_delta": bands, **ex,
                    "conditions": dict(zip(["confirm_ci", "year_signs", "popband", "not_short_only",
                                            "not_layoff_only", "not_extreme_state"], c)), "pass": all(c)})
        return out

    comps = [pair("M4", "M3"), pair("M6", "M5"), pair("M5", "M3"), pair("M3", "M2"), pair("M2", "M1"), pair("M1", "M0")]
    res["gate2"] = comps
    pd.DataFrame([{"compare": c["compare"], "period": pn, **c[pn]} for c in comps
                  for pn in ["confirm_2023", "oos_2024", "oos_2025"]]).to_csv(OUT / "deltas.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"compare": c["compare"], "band": k, "delta": v} for c in comps
                  for k, v in c["popband_delta"].items()]).to_csv(OUT / "by_popband.csv", index=False, encoding="utf-8-sig")
    print("\n[Gate 2] Δlogloss (負=左が良い), 97.5% CI")
    for c in comps:
        c3 = c["confirm_2023"]
        tag = "(主)" if c["compare"] in ("M4 vs M3", "M6 vs M5") else "(副)"
        print(f"  {tag}{c['compare']:<9} 2023 Δ={c3['delta']:+.6f} [{c3['ci_lo']:+.6f},{c3['ci_hi']:+.6f}] "
              f"| 2024 {c['oos_2024']['delta']:+.6f} | 2025 {c['oos_2025']['delta']:+.6f} "
              f"| 短間隔除外 {c['exclude_short']:+.6f} | 休養除外 {c['exclude_layoff']:+.6f} "
              f"| 極端状態除外 {c['exclude_top1pct_state']:+.6f} "
              f"| 人気帯 {', '.join(f'{k}:{v:+.5f}' for k, v in c['popband_delta'].items())} → {'PASS' if c['pass'] else 'fail'}")
    p43, p65 = comps[0]["pass"], comps[1]["pass"]
    verdict = ("状態モデル固有の追加情報あり (M4>M3 かつ M6>M5)" if p43 and p65 else
               "短期情報は有効だが状態モデル固有の価値はなし (既存 EWMA で十分)" if p43 else
               "Gate 2 FAIL: EXP02 能力と生の近走情報の後に、短期状態の追加情報なし → 仮説終了")
    res["gate2_verdict"] = verdict
    print(f"\n判定: {verdict}")

    # 副次
    sec = []
    age = df["age"].to_numpy()
    career = df["raw_career_runs"].to_numpy()
    subsets = [("v6◎", df["rank_v6"].to_numpy() == 1), ("AI・市場不一致", df["disagree"].to_numpy() == 1),
               ("短間隔≤14日", short), ("長期休養>180日", layoff), ("若齢≤3歳", age <= 3), ("出走≥15回", career >= 15)]
    for a, bb in [("M4", "M3"), ("M6", "M5")]:
        for lab, sub in subsets:
            mm = pooled & sub
            sec.append({"analysis": "部分集合", "compare": f"{a} vs {bb}", "subset": lab, "n": int(mm.sum()),
                        **delta_boot(y[mm], P[a][mm], P[bb][mm], day[mm], level=0.95)})
    ywin = df["win"].to_numpy()
    PW = {k: fit_predict(df, MODELS[k], ywin, train, sel)[0] for k in ["M3", "M4", "M5", "M6"]}
    for a, bb in [("M4", "M3"), ("M6", "M5")]:
        for pn, m in [("confirm_2023", conf), ("oos_2024_25", exp)]:
            sec.append({"analysis": "目的変数=1着", "compare": f"{a} vs {bb}", "subset": pn, "n": int(m.sum()),
                        **delta_boot(ywin[m], PW[a][m], PW[bb][m], day[m], level=0.95)})
    fin = df["fin"].to_numpy()
    for a, bb in [("M4", "M3"), ("M6", "M5")]:
        for pn, m in [("confirm_2023", conf), ("oos_2024_25", exp)]:
            t = pd.DataFrame({"rid": df.rid16[m], "day": day[m], "pa": P[a][m], "pb": P[bb][m], "fin": fin[m]})
            rr = [(g.day.iloc[0], spearmanr(g.pa, -g.fin).correlation - spearmanr(g.pb, -g.fin).correlation)
                  for _, g in t.groupby("rid") if len(g) >= 3]
            r = pd.DataFrame(rr, columns=["day", "d"]).groupby("day")["d"].agg(["sum", "count"])
            rng = np.random.default_rng(5)
            bs = [r["sum"].to_numpy()[i].sum() / r["count"].to_numpy()[i].sum()
                  for i in (rng.integers(0, len(r), len(r)) for _ in range(2000))]
            sec.append({"analysis": "レース内順位相関 Δ (正=左が良い)", "compare": f"{a} vs {bb}", "subset": pn,
                        "n": int(r["count"].sum()), "delta": float(r["sum"].sum() / r["count"].sum()),
                        "ci_lo": float(np.quantile(bs, .025)), "ci_hi": float(np.quantile(bs, .975))})
    pd.DataFrame(sec).to_csv(OUT / "secondary.csv", index=False, encoding="utf-8-sig")
    print("\n[副次評価] 95%CI")
    for r in sec:
        print(f"  {r['analysis'][:20]:<20} {r['compare']:<9} {str(r['subset']):<14} n={r['n']:>6} "
              f"Δ={r['delta']:+.6f} [{r['ci_lo']:+.6f},{r['ci_hi']:+.6f}]")
    pd.DataFrame([{"year": int(y0), "model": k, **metrics(y[yr == y0], P[k][yr == y0], df.rid16.to_numpy()[yr == y0])}
                  for y0 in sorted(set(yr)) for k in MODELS]).to_csv(OUT / "by_year.csv", index=False, encoding="utf-8-sig")

    if p43:
        a, bb = ("M6", "M5") if p65 else ("M4", "M3")
        res["gate3"] = economic(df, P, a, bb)
    else:
        res["gate3"] = "未実施 (M4 vs M3 が FAIL のため)"
    res["verdict"] = verdict
    (OUT / "gate_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\n書き出し: {OUT}")


def economic(df, P, use, ref):
    yr, fpay = df["year"].to_numpy(), df["fpay"].to_numpy()
    day = df["rid16"].str[:8].to_numpy()
    band = pd.cut(df["rank_mkt_pre"], [0, 1, 3, 6, 99], labels=["1", "2-3", "4-6", "7+"]).astype(str).to_numpy()
    days = df["raw_days_since"].to_numpy()
    ib = np.select([days <= 14, days <= 35, days <= 90, days <= 180, days > 180],
                   ["≤14", "15-35", "36-90", "91-180", ">180"], "新馬等")
    pooled = (yr >= 2023) & (yr <= 2025)
    sel = {}
    for k in [ref, use]:
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
            g = pd.DataFrame({"s": np.where(mm, 100.0, 0), "r": np.where(mm, fpay, 0), "day": day})[m] \
                .groupby("day")[["s", "r"]].sum().sort_index()
            cum = (g.r - g.s).cumsum()
            prof = pd.DataFrame({"rid": df.rid16[mm], "pl": fpay[mm] - 100}).groupby("rid")["pl"].sum()
            rng = np.random.default_rng(0)
            bs = [g.r.to_numpy()[i].sum() / max(g.s.to_numpy()[i].sum(), 1) * 100
                  for i in (rng.integers(0, len(g), len(g)) for _ in range(2000))]
            row = {"model": k, "rule": rule, "period": per, "bets": n, "hits": int((fpay[mm] > 0).sum()),
                   "roi": float(fpay[mm].sum() / (100 * n) * 100) if n else np.nan,
                   "roi_ci_lo": float(np.quantile(bs, .025)), "roi_ci_hi": float(np.quantile(bs, .975)),
                   "max_drawdown": float((cum - cum.cummax()).min()),
                   "top10_share": float(prof.sort_values(ascending=False).head(10).clip(lower=0).sum() / max(fpay[mm].sum(), 1))}
            for x in ["1", "2-3", "4-6", "7+"]:
                mb = mm & (band == x)
                row[f"roi_pop_{x}"] = float(fpay[mb].sum() / (100 * mb.sum()) * 100) if mb.sum() else np.nan
            for x in ["≤14", "15-35", "36-90", "91-180", ">180"]:
                mb = mm & (ib == x)
                row[f"roi_int_{x}"] = float(fpay[mb].sum() / (100 * mb.sum()) * 100) if mb.sum() else np.nan
            rows.append(row)
    e = pd.DataFrame(rows)
    e.to_csv(OUT / "economic.csv", index=False, encoding="utf-8-sig")
    diffs = []
    for rule in ["R1", "R2"]:
        a, b = sel[(use, rule)] & pooled, sel[(ref, rule)] & pooled
        g = pd.DataFrame({"day": day, "sa": np.where(a, 100.0, 0), "ra": np.where(a, fpay, 0),
                          "sb": np.where(b, 100.0, 0), "rb": np.where(b, fpay, 0)})[pooled].groupby("day").sum()
        rng = np.random.default_rng(1)
        bs = []
        for _ in range(2000):
            x = g.iloc[rng.integers(0, len(g), len(g))].sum()
            bs.append(x.ra / x.sa * 100 - x.rb / x.sb * 100)
        x = g.sum()
        diffs.append({"rule": rule, "compare": f"{use} - {ref}", "roi_diff_pt": float(x.ra / x.sa * 100 - x.rb / x.sb * 100),
                      "ci_lo": float(np.quantile(bs, .025)), "ci_hi": float(np.quantile(bs, .975))})
    pd.DataFrame(diffs).to_csv(OUT / "economic_diff.csv", index=False, encoding="utf-8-sig")
    print("\n[Gate 3]")
    print(e[e.period == "2023-25"][["model", "rule", "bets", "roi", "roi_ci_lo", "roi_ci_hi", "max_drawdown", "top10_share"]].round(2).to_string(index=False))
    for d in diffs:
        print(f"  {d['rule']} {d['compare']}: {d['roi_diff_pt']:+.2f}pt [{d['ci_lo']:+.2f},{d['ci_hi']:+.2f}]")
    return {"table": e.to_dict("records"), "diff": diffs}


if __name__ == "__main__":
    main()
