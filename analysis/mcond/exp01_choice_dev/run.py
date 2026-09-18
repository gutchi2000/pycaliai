# -*- coding: utf-8 -*-
"""
run.py — 第1実験 (陣営選択の逸脱) の Gate 1〜3 を spec.json どおりに実行する
==========================================================================
入力: data/_research/mcond/base.parquet, exp01_features_bp.parquet
出力: analysis/mcond/exp01_choice_dev/out/
  gate_results.json   各 Gate の合否・数値・コミットハッシュ・入力ハッシュ
  model_compare.csv   M0-M5 × 期間 の指標
  deltas.csv          主比較の Δlogloss と CI
  by_year.csv / by_popband.csv / secondary.csv / gate1.csv
  economic.csv        (Gate 2 PASS 時のみ)
実行: python -m analysis.mcond.exp01_choice_dev.run
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
from analysis.mcond.evaluate import (fit_predict, metrics, delta_boot, logit, ll_vec)  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
SPEC = json.loads((HERE / "spec.json").read_text(encoding="utf-8"))
D = BASE / "data/_research/mcond"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def load() -> pd.DataFrame:
    b = pd.read_parquet(D / "base.parquet")
    f = pd.read_parquet(D / "exp01_features_bp.parquet")
    f["rid16"] = f["rid16"].astype(str)
    df = b.merge(f, on=["rid16", "ban"], how="inner", suffixes=("", "_f"))
    y0, y1 = SPEC["population"]["years"]
    df = df[(df.year >= y0) & (df.year <= y1) & (df.n_field >= 5)
            & df.mkt_p3_pre.notna() & df.v6_p3.notna()].copy()
    df["f_v6"] = logit(df["v6_p3"])
    df["f_mkt"] = logit(df["mkt_p3_pre"])
    df["f_mkt_am9"] = logit(df["mkt_p3_am9"])
    df["disagree"] = ((df["rank_v6"] - df["rank_mkt_pre"]).abs() >= 3).astype(float)
    df["dis_x_ss"] = df["disagree"] * df["ss_total"]
    df["dis_x_bp"] = df["disagree"] * df["bp_total"]
    df["day"] = df["date"].astype(str).str[:8] if "date" in df else df["rid16"].str[:8]
    df["day"] = df["rid16"].str[:8]
    return df.reset_index(drop=True)


def gate1(df: pd.DataFrame) -> tuple[bool, pd.DataFrame]:
    c = SPEC["gate1_representation"]
    raw = SPEC["features"]["raw_choice"]
    rows, ok_all = [], True
    for col in c["applies_to"]:
        x = df[col]
        r_n = abs(spearmanr(x, df["n_prev_runs"], nan_policy="omit").correlation)
        r_t = abs(spearmanr(x, df["trainer_n_prior"], nan_policy="omit").correlation)
        thr = x.quantile(0.99)
        small = df["trainer_n_prior"] < 30
        share_top = small[x >= thr].mean()
        share_pop = small.mean()
        ratio = share_top / share_pop if share_pop > 0 else np.inf
        R = df[raw].fillna(df[raw].mean()).to_numpy()
        R = np.column_stack([np.ones(len(R)), R])
        yv = x.fillna(x.mean()).to_numpy()
        beta, *_ = np.linalg.lstsq(R, yv, rcond=None)
        r2 = 1 - ((yv - R @ beta) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
        cov = df.groupby("year")[col].apply(lambda s: s.notna().mean())
        checks = {
            "spearman_n_prev_runs": (r_n, r_n < c["abs_spearman_with_n_prev_runs_lt"]),
            "spearman_trainer_n_prior": (r_t, r_t < c["abs_spearman_with_trainer_n_prior_lt"]),
            "top1pct_small_trainer_ratio": (ratio, ratio <= c["top1pct_small_trainer_share_max_ratio"]),
            "r2_on_raw": (r2, r2 < c["r2_total_on_raw_lt"]),
            "min_coverage_year": (float(cov.min()), cov.min() >= c["coverage_min_each_year"]),
        }
        for k, (v, ok) in checks.items():
            rows.append({"feature": col, "check": k, "value": round(float(v), 4), "pass": bool(ok)})
            ok_all &= bool(ok)
    return ok_all, pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=BASE, capture_output=True,
                            text=True).stdout.strip()
    spec_dirty = subprocess.run(["git", "status", "--porcelain", str(HERE / "spec.json")],
                                cwd=BASE, capture_output=True, text=True).stdout.strip()
    df = load()
    y = df["top3"].to_numpy()
    yr = df["year"].to_numpy()
    sp = SPEC["splits"]
    train = (yr >= sp["train"][0]) & (yr <= sp["train"][1])
    sel = yr == sp["selection"][0]
    conf = yr == sp["confirm"][0]
    exp = (yr >= sp["exploratory_oos"][0]) & (yr <= sp["exploratory_oos"][1])
    res = {"experiment_id": SPEC["experiment_id"], "commit": commit,
           "spec_uncommitted_changes": bool(spec_dirty),
           "inputs": {p: sha(D / p) for p in ["base.parquet", "exp01_features_bp.parquet", "market.parquet"]},
           "n_rows": int(len(df)), "n_rows_by_year": df.groupby("year").size().astype(int).to_dict(),
           "n_races": int(df.rid16.nunique())}
    print(f"母集団 {len(df):,} 行 / {df.rid16.nunique():,} レース  commit={commit[:10]}"
          f"{' (spec未コミット!)' if spec_dirty else ''}")

    # ---------------- Gate 1 ----------------
    g1_ok, g1 = gate1(df)
    g1.to_csv(OUT / "gate1.csv", index=False, encoding="utf-8-sig")
    res["gate1"] = {"pass": g1_ok, "checks": g1.to_dict("records")}
    print(f"\n[Gate 1] {'PASS' if g1_ok else 'FAIL'}")
    print(g1.to_string(index=False))
    if not g1_ok:
        res["verdict"] = "Gate 1 FAIL: 逸脱表現が成立しない → 中止"
        (OUT / "gate_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str),
                                               encoding="utf-8")
        return

    F = SPEC["features"]
    raw, dev = F["raw_choice"], F["deviation_simple"] + F["deviation_behavior"]
    MODELS = {"M0": ["f_mkt"], "M1": ["f_v6", "f_mkt"], "M2": ["f_v6", "f_mkt"] + raw,
              "M3": ["f_v6", "f_mkt"] + dev, "M4": ["f_v6", "f_mkt"] + raw + dev,
              "M5": ["f_v6", "f_mkt", "ss_total", "bp_total", "disagree", "dis_x_ss", "dis_x_bp"]}
    P, CS, COEF = {}, {}, {}
    for name, cols in MODELS.items():
        P[name], CS[name], COEF[name] = fit_predict(df, cols, y, train, sel)
    res["C_selected"] = CS
    res["coef"] = {k: {kk: float(vv) for kk, vv in v.items()} for k, v in COEF.items()}

    # ---------------- 指標 ----------------
    periods = {"confirm_2023": conf, "oos_2024": yr == 2024, "oos_2025": yr == 2025,
               "oos_2024_25": exp, "selection_2022": sel}
    rows = []
    for name in MODELS:
        for pn, m in periods.items():
            r = metrics(y[m], P[name][m], df.rid16.to_numpy()[m])
            rows.append({"model": name, "period": pn, **r})
    mc = pd.DataFrame(rows)
    mc.to_csv(OUT / "model_compare.csv", index=False, encoding="utf-8-sig")
    print("\n[モデル比較 logloss]")
    print(mc.pivot(index="model", columns="period", values="logloss").round(5).to_string())

    # ---------------- Gate 2 ----------------
    day = df["day"].to_numpy()
    rank_band = pd.cut(df["rank_mkt_pre"], [0, 1, 3, 6, 99], labels=["1", "2-3", "4-6", "7+"])
    pooled = conf | exp

    def evaluate_pair(a, b, level=0.975):
        out = {"compare": f"{a} vs {b}"}
        out["confirm_2023"] = delta_boot(y[conf], P[a][conf], P[b][conf], day[conf], level=level)
        out["oos_2024"] = delta_boot(y[yr == 2024], P[a][yr == 2024], P[b][yr == 2024], day[yr == 2024], level=level)
        out["oos_2025"] = delta_boot(y[yr == 2025], P[a][yr == 2025], P[b][yr == 2025], day[yr == 2025], level=level)
        d_row = ll_vec(y, P[a]) - ll_vec(y, P[b])
        t = pd.DataFrame({"d": d_row[pooled], "tr": df["trainer"].to_numpy()[pooled]})
        by_tr = t.groupby("tr")["d"].sum().sort_values()
        drop = set(by_tr.index[:20])
        rest = t[~t["tr"].isin(drop)]["d"].mean()
        out["drop_top20_trainers_delta"] = float(rest)
        bands = {}
        for bnd in ["1", "2-3", "4-6", "7+"]:
            mm = pooled & (rank_band == bnd).to_numpy()
            bands[bnd] = float(d_row[mm].mean())
        out["popband_delta"] = bands
        c = [
            out["confirm_2023"]["delta"] < 0 and out["confirm_2023"]["ci_hi"] < 0,
            out["oos_2024"]["delta"] < 0 and out["oos_2025"]["delta"] < 0,
            rest < 0,
            sum(v < 0 for v in bands.values()) >= 3,
        ]
        out["conditions"] = {"confirm_ci": c[0], "year_signs": c[1], "trainer_robust": c[2], "popband": c[3]}
        out["pass"] = all(c)
        return out

    comps = [evaluate_pair("M3", "M1"), evaluate_pair("M4", "M1"), evaluate_pair("M4", "M2"),
             evaluate_pair("M2", "M1"), evaluate_pair("M5", "M1"), evaluate_pair("M1", "M0")]
    res["gate2"] = comps
    drows = []
    for cp in comps:
        for pn in ["confirm_2023", "oos_2024", "oos_2025"]:
            drows.append({"compare": cp["compare"], "period": pn, **cp[pn]})
    pd.DataFrame(drows).to_csv(OUT / "deltas.csv", index=False, encoding="utf-8-sig")
    band_rows = [{"compare": cp["compare"], "band": k, "delta": v}
                 for cp in comps for k, v in cp["popband_delta"].items()]
    pd.DataFrame(band_rows).to_csv(OUT / "by_popband.csv", index=False, encoding="utf-8-sig")

    print("\n[Gate 2] Δlogloss (負=左が良い), 97.5% CI")
    for cp in comps:
        c3 = cp["confirm_2023"]
        print(f"  {cp['compare']:<9} 2023 Δ={c3['delta']:+.6f} [{c3['ci_lo']:+.6f},{c3['ci_hi']:+.6f}] "
              f"| 2024 {cp['oos_2024']['delta']:+.6f} | 2025 {cp['oos_2025']['delta']:+.6f} "
              f"| 上位20調教師除外 {cp['drop_top20_trainers_delta']:+.6f} "
              f"| 人気帯 {', '.join(f'{k}:{v:+.5f}' for k, v in cp['popband_delta'].items())} "
              f"→ {'PASS' if cp['pass'] else 'fail'}")

    p31, p41, p42 = comps[0]["pass"], comps[1]["pass"], comps[2]["pass"]
    if not (p31 or p41):
        verdict = "Gate 2 FAIL: 現行AIと市場を与えた後、陣営選択の逸脱に追加情報は確認できない"
    elif not p42:
        verdict = "選択情報は有効だが、逸脱モデル固有の価値はなし (M4 が M2 に勝てない)"
    else:
        verdict = "Gate 2 PASS: 市場条件付きで逸脱に固有の追加情報あり"
    m21 = comps[3]["pass"]
    res["gate2_verdict"] = verdict
    res["raw_choice_vs_M1_pass"] = m21
    print(f"\n判定: {verdict}")
    print(f"  (参考: 生の選択列だけ M2 vs M1 は {'PASS' if m21 else 'fail'})")

    # ---------------- 副次評価 ----------------
    sec = []
    ywin = df["win"].to_numpy()
    for name in ["M1", "M2", "M3", "M4"]:
        pw, _, _ = fit_predict(df, MODELS[name], ywin, train, sel)
        df[f"pw_{name}"] = pw
    for a, b in [("M3", "M1"), ("M4", "M1"), ("M4", "M2")]:
        for pn, m in [("confirm_2023", conf), ("oos_2024_25", exp)]:
            r = delta_boot(ywin[m], df[f"pw_{a}"].to_numpy()[m], df[f"pw_{b}"].to_numpy()[m], day[m], level=0.95)
            sec.append({"analysis": "target=win", "compare": f"{a} vs {b}", "period": pn, **r})
    for label, sub in [("v6◎ の top3", df["rank_v6"].to_numpy() == 1),
                       ("disagree==1 の top3", df["disagree"].to_numpy() == 1)]:
        for a, b in [("M3", "M1"), ("M4", "M1"), ("M4", "M2")]:
            for pn, m in [("confirm_2023", conf), ("oos_2024_25", exp)]:
                mm = m & sub
                r = delta_boot(y[mm], P[a][mm], P[b][mm], day[mm], level=0.95)
                sec.append({"analysis": label, "compare": f"{a} vs {b}", "period": pn, **r})
    # 感度: 市場 9時
    dfa = df.copy()
    dfa["f_mkt"] = dfa["f_mkt_am9"]
    ok_am9 = dfa["f_mkt"].notna().to_numpy()
    PA = {}
    for name in ["M1", "M3", "M4"]:
        PA[name], _, _ = fit_predict(dfa, MODELS[name], y, train & ok_am9, sel & ok_am9)
    for a, b in [("M3", "M1"), ("M4", "M1")]:
        for pn, m in [("confirm_2023", conf), ("oos_2024_25", exp)]:
            mm = m & ok_am9
            r = delta_boot(y[mm], PA[a][mm], PA[b][mm], day[mm], level=0.95)
            sec.append({"analysis": "感度: 市場=9時", "compare": f"{a} vs {b}", "period": pn, **r})
    secd = pd.DataFrame(sec)
    secd.to_csv(OUT / "secondary.csv", index=False, encoding="utf-8-sig")
    print("\n[副次評価] (95% CI、主検定ではない)")
    for r in sec:
        print(f"  {r['analysis']:<18} {r['compare']:<9} {r['period']:<13} Δ={r['delta']:+.6f} "
              f"[{r['ci_lo']:+.6f},{r['ci_hi']:+.6f}]")

    # 年別 (M1 と M4)
    by = []
    for y0 in sorted(set(yr)):
        m = yr == y0
        for name in ["M0", "M1", "M2", "M3", "M4"]:
            by.append({"year": int(y0), "model": name, **metrics(y[m], P[name][m], df.rid16.to_numpy()[m])})
    pd.DataFrame(by).to_csv(OUT / "by_year.csv", index=False, encoding="utf-8-sig")

    # ---------------- Gate 3 (Gate 2 PASS 時のみ) ----------------
    if p31 or p41:
        use = "M4" if p41 else "M3"
        res["gate3"] = economic(df, P, use)
    else:
        res["gate3"] = "未実施 (Gate 2 FAIL のため。仕様どおり経済評価は行わない)"

    res["verdict"] = verdict
    (OUT / "gate_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str),
                                           encoding="utf-8")
    print(f"\n書き出し: {OUT}")


def economic(df, P, use):
    rows = []
    yr = df["year"].to_numpy()
    fpay = df["fpay"].to_numpy()
    for name in ["M1", use]:
        p = P[name]
        ratio = p / df["mkt_p3_pre"].to_numpy()
        r1 = ratio >= 1.15
        d = pd.DataFrame({"rid": df.rid16, "p": p})
        r2 = np.zeros(len(df), bool)
        r2[d.groupby("rid").p.idxmax().to_numpy()] = True
        for rule, sel in [("R1", r1), ("R2", r2)]:
            for period, m in [("2023", yr == 2023), ("2024", yr == 2024), ("2025", yr == 2025)]:
                mm = sel & m
                n = int(mm.sum())
                ret = fpay[mm].sum()
                rows.append({"model": name, "rule": rule, "period": period, "bets": n,
                             "hits": int((fpay[mm] > 0).sum()), "stake": 100 * n, "return": float(ret),
                             "roi": float(ret / (100 * n) * 100) if n else np.nan})
    hon = df["rank_v6"].to_numpy() == 1
    fav = df["rank_mkt_pre"].to_numpy() == 1
    for name, sel in [("v6◎複勝", hon), ("市場1番人気複勝", fav)]:
        for period, m in [("2023", yr == 2023), ("2024", yr == 2024), ("2025", yr == 2025)]:
            mm = sel & m
            n = int(mm.sum())
            rows.append({"model": name, "rule": "flat", "period": period, "bets": n,
                         "hits": int((fpay[mm] > 0).sum()), "stake": 100 * n, "return": float(fpay[mm].sum()),
                         "roi": float(fpay[mm].sum() / (100 * n) * 100) if n else np.nan})
    e = pd.DataFrame(rows)
    e.to_csv(OUT / "economic.csv", index=False, encoding="utf-8-sig")
    return e.to_dict("records")


if __name__ == "__main__":
    main()
