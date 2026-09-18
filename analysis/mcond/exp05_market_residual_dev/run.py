# -*- coding: utf-8 -*-
"""
run.py — EXP05 Gate 0/1/2(探索的)、条件付き較正、探索的経済評価
==================================================================
2026 は LOCKED_PERIOD_AUDIT.md の結論によりロック評価不可 (前売市場オッズの時系列アーカイブが
2025-12-28 で止まっている)。よって Gate 2 は 2023-2025 を使うが「探索的」と明記し、
確定的な合否判定 (spec §16 の意味でのロック合格) は行わない。§9 の代替規定に従う。
実行: python -m analysis.mcond.exp05_market_residual_dev.run
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
from analysis.mcond.evaluate import delta_boot, metrics, ll_vec, logit  # noqa: E402
from analysis.mcond.exp05_market_residual_dev import models as M  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
D = BASE / "data/_research/mcond"
SPEC = json.loads((HERE / "spec.json").read_text(encoding="utf-8"))

PERIODS = {"sel_2022": None, "confirm_2023": None, "oos_2024": None, "oos_2025": None, "oos_2024_25": None}


def periods(df):
    yr = df["year"].to_numpy()
    return {"sel_2022": df["sel"].to_numpy(), "confirm_2023": yr == 2023,
           "oos_2024": yr == 2024, "oos_2025": yr == 2025, "oos_2024_25": (yr == 2024) | (yr == 2025)}


def fit_all(df, target_col, feature_lists, m6_cache=None):
    y = df[target_col].to_numpy()
    train, sel = df["train"].to_numpy(), df["sel"].to_numpy()
    out = M.fit_m0_m3(df, y, train, sel)
    offset3 = logit(np.clip(out["M3"]["pred"], 1e-9, 1 - 1e-9))
    out["M4"] = M.fit_offset_residual(df, feature_lists["F_serve"], offset3, y, train, sel)
    out["M5"] = M.fit_offset_residual(df, feature_lists["F_full"], offset3, y, train, sel)
    m6 = m6_cache if m6_cache is not None else M.fit_m6_benter(df, train)
    m6_pred = m6["pred"] if target_col == "top3" else m6["pred_win"]
    out["M6"] = {"pred": m6_pred, "alpha": m6["alpha"], "beta": m6["beta"]}
    return out, y, m6


def model_compare_table(df, y, P, tag):
    rows = []
    per = periods(df)
    for name, pred in P.items():
        ok = ~np.isnan(pred)
        for pn, m in per.items():
            mm = m & ok
            if mm.sum() == 0:
                continue
            rows.append({"target": tag, "model": name, "period": pn,
                        **metrics(y[mm], pred[mm], df.rid16.to_numpy()[mm])})
    return pd.DataFrame(rows)


def gate0(df, feature_lists, tie_diag):
    n_races_total = df.rid16.nunique()
    return {
        "n_rows": int(len(df)), "n_races": n_races_total,
        "years": sorted(df.year.unique().tolist()),
        "join_rate_candidates": 1.0,  # build_features は inner merge のみ保持するので定義上1.0
        "F_full_n": len(feature_lists["F_full"]), "F_serve_n": len(feature_lists["F_serve"]),
        "excluded_for_serve_n": len(feature_lists["excluded_for_serve"]),
        "calibrator_meta": feature_lists["calibrator_meta"],
        "tie_rate_calibrated_top3_trainrefit": tie_diag["tie_rates"]["calibrated_v6_probability (top3, レース内最大タイ)"]["tie_rate"],
        "tie_rate_calibrated_win_trainrefit": tie_diag["tie_rates"]["calibrated_v6_probability_win (レース内最大タイ)"]["tie_rate"],
        "tie_rate_production_calibrator_2024_2025": tie_diag["tie_rate_production_calibrator_2024_2025"]["tie_rate"],
        "2026_lockable": False,
        "2026_lock_reason": "LOCKED_PERIOD_AUDIT.md: 前売市場オッズ時系列アーカイブ(TANPUK)が2025-12-28で終了、"
                            "2026年のv6生スコア・特徴量は週次実行時に永続化されておらず研究用に再構築されていない",
        "oof_v6_by_construction": "v6base.py: 2015-2023はexpanding-window OOF、2024-2025は本番v6(train<=2022)で"
                                  "対象年について時点安全",
        "calibrator_leak_avoided": "本番pl_calibrators_v6.pklはvalid=2023でfitされているため2023年評価には使わず、"
                                   "train(2016-2021)のみでisotonicを自前refitした",
    }


def gate1_reproduce_exp04(df, y, P, day):
    """EXP04の中心的発見 = 全特徴プール(通常学習)がv6+市場を一貫して上回る、を再現できるか。
    EXP05のM5(offset(M3)+F-full)がM1(較正v6+市場)を2023-2025で上回るかを見る。"""
    per = periods(df)
    out = {}
    for pn in ("confirm_2023", "oos_2024", "oos_2025"):
        m = per[pn]
        out[pn] = delta_boot(y[m], P["M5"][m], P["M1"][m], day[m])
    reproduced = all(out[pn]["delta"] < 0 for pn in out)
    return {"compare": "M5 vs M1", "periods": out, "reproduced_exp04_direction": bool(reproduced)}


def band_breakdown(df, d, mask, col, labels_order=None):
    vals = df[col].to_numpy()
    levels = labels_order if labels_order else sorted(pd.unique(vals[mask]))
    return {str(lv): float(d[mask & (vals == lv)].mean()) for lv in levels if (mask & (vals == lv)).sum() >= 30}


def gate2_exploratory(df, y, P, day):
    per = periods(df)
    pooled = per["confirm_2023"] | per["oos_2024"] | per["oos_2025"]
    band = pd.cut(df["rank_mkt_pre"], [0, 1, 3, 6, 99], labels=["1", "2-3", "4-6", "7+"]).astype(str)
    df = df.assign(_band=band)

    def pair(a, b):
        res = {"compare": f"{a} vs {b}"}
        for pn in ("confirm_2023", "oos_2024", "oos_2025"):
            m = per[pn]
            res[pn] = delta_boot(y[m], P[a][m], P[b][m], day[m])
        d = ll_vec(y, P[a]) - ll_vec(y, P[b])
        res["venue_delta"] = band_breakdown(df, d, pooled, "venue")
        res["popularity_band_delta"] = band_breakdown(df, d, pooled, "_band", ["1", "2-3", "4-6", "7+"])
        rr = pd.DataFrame({"rid": df.rid16, "d": d}).groupby("rid")["d"].sum()
        top100 = set(rr.sort_values().index[:100])
        res["drop_top100_race_delta"] = float(d[pooled & ~df.rid16.isin(top100).to_numpy()].mean())
        return res

    g_primary = pair("M4", "M3")   # 主判定: serve-safe表特徴の追加情報
    g_full_vs_serve = pair("M5", "M4")  # フル特徴 vs serve特徴 (実運用可能性)
    g_vs_benter = pair("M4", "M6")  # 過去方式(Benter型)との差

    def verdict(g):
        c = g["confirm_2023"]
        return bool(c["delta"] < 0 and c["ci_hi"] < 0)

    primary_pass_2023 = verdict(g_primary)
    dir_2024 = g_primary["oos_2024"]["delta"] < 0
    dir_2025 = g_primary["oos_2025"]["delta"] < 0
    venue_ok = sum(v < 0 for v in g_primary["venue_delta"].values())
    venue_n = len(g_primary["venue_delta"])
    band_ok = sum(v < 0 for v in g_primary["popularity_band_delta"].values())
    band_n = len(g_primary["popularity_band_delta"])

    return {
        "note": "2026非ロックのため探索的評価 (spec.json locked_period.lockable=false)。合否は参考情報。",
        "primary_M4_vs_M3": g_primary,
        "full_vs_serve_M5_vs_M4": g_full_vs_serve,
        "vs_benter_M4_vs_M6": g_vs_benter,
        "exploratory_summary": {
            "primary_2023_direction_and_ci_negative": primary_pass_2023,
            "primary_2024_direction_negative": bool(dir_2024),
            "primary_2025_direction_negative": bool(dir_2025),
            "venue_frac_favoring_M4": f"{venue_ok}/{venue_n}",
            "popularity_band_frac_favoring_M4": f"{band_ok}/{band_n}",
        },
    }


def conditional_calibration(df, y, p, mask, tag):
    """spec §13 の固定領域での較正確認。"""
    edge = p - df["mkt_p3_pre"].to_numpy()
    band = pd.cut(df["rank_mkt_pre"], [0, 1, 3, 6, 99], labels=["1", "2-3", "4-6", "7+"]).astype(str).to_numpy()
    mismatch = np.sign(df["rank_v6"].to_numpy() - df["rank_mkt_pre"].to_numpy())

    def region_stats(name, m):
        m = m & mask
        if m.sum() < 30:
            return None
        pp, yy = p[m], y[m]
        return {"region": name, "n": int(m.sum()), "mean_pred": float(pp.mean()), "actual_rate": float(yy.mean()),
               "calibration_gap": float(pp.mean() - yy.mean()), "brier": float(((pp - yy) ** 2).mean())}

    regions = []
    q90 = np.quantile(edge[mask], 0.90) if mask.sum() else np.nan
    q95 = np.quantile(edge[mask], 0.95) if mask.sum() else np.nan
    regions.append(region_stats("model_higher_than_market", edge > 0))
    regions.append(region_stats("edge_top10pct", edge >= q90))
    regions.append(region_stats("edge_top5pct", edge >= q95))
    for bb in ["1", "2-3", "4-6", "7+"]:
        regions.append(region_stats(f"popularity_band_{bb}", band == bb))
    regions.append(region_stats("rank_mismatch_v6_higher_than_market", mismatch < 0))
    regions.append(region_stats("rank_mismatch_market_higher_than_v6", mismatch > 0))

    per_race_max_edge = df.assign(_edge=edge).loc[mask].groupby("rid16")["_edge"].idxmax()
    regions.append(region_stats("per_race_max_edge_horse", df.index.isin(per_race_max_edge)))

    marui = df["rank_v6"].to_numpy() == 1
    regions.append(region_stats("maru_honmei_v6_top1", marui))

    regions = [r for r in regions if r is not None]
    edge_bins = pd.qcut(edge[mask], 10, duplicates="drop")
    mono = pd.DataFrame({"edge_bin": edge_bins, "y": y[mask], "p": p[mask]}).groupby("edge_bin", observed=True).agg(
        n=("y", "size"), mean_pred=("p", "mean"), actual_rate=("y", "mean")).reset_index()
    mono["edge_bin"] = mono["edge_bin"].astype(str)
    return {"tag": tag, "regions": regions, "edge_decile_monotonicity": mono.to_dict("records")}


def economic_r1r2(df, p_model, mask, day, tag, other_preds: dict):
    """spec §18: R1 (edge>=1.15複勝均等) / R2 (レース最大p 1点)、確定複勝配当で決済。"""
    edge = p_model / df["mkt_p3_pre"].to_numpy()
    fpay = df["fpay"].to_numpy()

    def settle(sel_mask):
        m = mask & sel_mask
        n_bet = int(m.sum())
        if n_bet == 0:
            return {"n_bet": 0}
        stake = 100.0 * n_bet
        payout = float(fpay[m].sum())
        roi = payout / stake if stake else np.nan
        d = pd.DataFrame({"day": day[m], "pay": fpay[m] - 100.0}).groupby("day")["pay"].sum()
        rng = np.random.default_rng(42)
        boots = np.array([d.sample(len(d), replace=True, random_state=int(rng.integers(1 << 30))).sum()
                          for _ in range(1000)])
        ci = (float(np.quantile((boots + stake) / stake, 0.025)), float(np.quantile((boots + stake) / stake, 0.975)))
        return {"n_bet": n_bet, "n_races": int(pd.Series(day[m]).nunique()), "stake_yen": stake,
               "payout_yen": payout, "roi_pct": 100 * roi, "roi_ci95_pct": [100 * ci[0], 100 * ci[1]]}

    r1 = settle(edge >= 1.15)
    top_idx = df.assign(_p=p_model).loc[mask].groupby("rid16")["_p"].idxmax()
    r2_mask = df.index.isin(top_idx)
    r2 = settle(r2_mask)

    out = {"tag": tag, "R1_edge_flat100": r1, "R2_race_top1_flat100": r2}
    for other_name, other_p in other_preds.items():
        other_edge = other_p / df["mkt_p3_pre"].to_numpy()
        out[f"R1_edge_flat100_{other_name}"] = settle(other_edge >= 1.15)
    return out


def main() -> None:
    OUT.mkdir(exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=BASE, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", str(HERE / "spec.json")], cwd=BASE,
                           capture_output=True, text=True).stdout.strip()

    df = pd.read_parquet(D / "exp05_design.parquet")
    feature_lists = json.loads((OUT / "feature_lists.json").read_text(encoding="utf-8"))
    tie_diag = json.loads((OUT / "tie_diagnosis.json").read_text(encoding="utf-8"))
    day = df["day"].to_numpy()

    res = {"commit": commit, "spec_uncommitted_changes": bool(dirty)}
    res["gate0"] = gate0(df, feature_lists, tie_diag)
    print("[Gate0]", json.dumps(res["gate0"], ensure_ascii=False, indent=1)[:2000])

    print("\n[fit top3 primary]", flush=True)
    P_top3, y_top3, m6_cache = fit_all(df, "top3", feature_lists)
    mc_top3 = model_compare_table(df, y_top3, {k: v["pred"] for k, v in P_top3.items()}, "top3")

    print("[fit win secondary]", flush=True)
    df_win = df.drop(columns=["lp_cal_top3", "f_mkt"]).rename(
        columns={"lp_cal_win": "lp_cal_top3", "f_mkt_win": "f_mkt"})
    P_win, y_win, _ = fit_all(df_win, "win", feature_lists, m6_cache=m6_cache)
    mc_win = model_compare_table(df, y_win, {k: v["pred"] for k, v in P_win.items()}, "win")

    mc = pd.concat([mc_top3, mc_win], ignore_index=True)
    mc.to_csv(OUT / "model_compare.csv", index=False, encoding="utf-8-sig")
    print("\n[top3 logloss]")
    print(mc_top3.pivot(index="model", columns="period", values="logloss").round(5).to_string())
    print("\n[win logloss]")
    print(mc_win.pivot(index="model", columns="period", values="logloss").round(5).to_string())

    P_top3_pred = {k: v["pred"] for k, v in P_top3.items()}
    res["gate1"] = gate1_reproduce_exp04(df, y_top3, P_top3_pred, day)
    print(f"\n[Gate1] EXP04再現: {res['gate1']['reproduced_exp04_direction']}")
    for pn, r in res["gate1"]["periods"].items():
        print(f"  {pn}: Δ={r['delta']:+.6f} [{r['ci_lo']:+.6f},{r['ci_hi']:+.6f}]")

    res["gate2"] = gate2_exploratory(df, y_top3, P_top3_pred, day)
    g = res["gate2"]["primary_M4_vs_M3"]
    print(f"\n[Gate2 探索的] M4 vs M3 (serve-safe表特徴の追加情報)")
    for pn in ("confirm_2023", "oos_2024", "oos_2025"):
        r = g[pn]
        print(f"  {pn}: Δ={r['delta']:+.6f} [{r['ci_lo']:+.6f},{r['ci_hi']:+.6f}]")
    print(f"  summary: {res['gate2']['exploratory_summary']}")

    conf_oos_mask = (df.year >= 2023).to_numpy()
    res["conditional_calibration_M4"] = conditional_calibration(df, y_top3, P_top3_pred["M4"], conf_oos_mask, "M4")
    res["conditional_calibration_M1"] = conditional_calibration(df, y_top3, P_top3_pred["M1"], conf_oos_mask, "M1")

    res["economic_exploratory"] = economic_r1r2(
        df, P_top3_pred["M4"], conf_oos_mask, day, "M4",
        other_preds={"M1": P_top3_pred["M1"], "M3": P_top3_pred["M3"]})

    (OUT / "gate_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str),
                                           encoding="utf-8")
    print("\n保存: out/gate_results.json, out/model_compare.csv")


if __name__ == "__main__":
    main()
