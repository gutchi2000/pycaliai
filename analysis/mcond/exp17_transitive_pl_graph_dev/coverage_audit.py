# -*- coding: utf-8 -*-
"""
coverage_audit.py — EXP17 Stage 0: 時点安全グラフの被覆・構造監査 (2019-2023)。結果精度・logloss は計算しない。
=====================================================================================================
入力: data/master_v2_20130105-20251228.csv (2024/2025 は読み込み時に破棄)
出力:
  out/GRAPH_COVERAGE.json      年・頭数帯・芝ダ・年齢帯・出走数帯 別の被覆と構造
  out/race_population.json     正式 race set の funnel
  out/pairs_2019_2023.parquet  未対戦かつ共通対戦馬あり pair の d_ij/w/n_common (label 無し; 検出力監査の構造入力)
  out/horses_2019_2023.parquet 対象レースの馬単位 (uncovered/degree/prior_starts/age; label 無し)
  out/edges_2022.parquet       2022 の (i,j,c) 単位の d_ij^(c)・ω・W/L/n (等価性監査の入力; label 無し)
  out/compute_timings.json     build/query の所要時間・メモリ (COMPUTE_DRY_RUN の実測部)
実行: python -m analysis.mcond.exp17_transitive_pl_graph_dev.coverage_audit
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .graph_core import (PairParams, build_history_store, dist_band, hodge_project, pair_evidence,
                         validate_hids)

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[2]
OUT = HERE / "out"
MASTER = BASE / "data" / "master_v2_20130105-20251228.csv"
DEV_YEARS = [2019, 2020, 2021, 2022, 2023]
HIST_MIN_YEAR = 2013
SEALED_FROM = 20240101
PRM = PairParams(alpha=1.0, eps=0.01, half_life_days=None, cond_bonus=0.0, lookback_days=None, lam=1e-6)
FIELD_BINS = [(5, 8), (9, 12), (13, 15), (16, 18)]
AGE_BANDS = [(2, 2), (3, 3), (4, 4), (5, 5), (6, 99)]
CAREER_BANDS = [(0, 0), (1, 5), (6, 10), (11, 20), (21, 999)]


def rss_mb() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1e6
    except Exception:
        return float("nan")


def band(v, bins):
    for lo, hi in bins:
        if lo <= v <= hi:
            return f"{lo}-{hi}" if lo != hi else f"{lo}"
    return "other"


def load_master() -> pd.DataFrame:
    cols = ["日付", "レースID(新/馬番無)", "血統登録番号", "馬名", "馬番", "着順", "出走頭数", "トラックコード(JV)",
            "芝・ダ", "距離", "年齢"]
    parts = []
    for ch in pd.read_csv(MASTER, encoding="utf-8-sig", dtype=str, usecols=cols, chunksize=200_000):
        d = pd.to_numeric(ch["日付"], errors="coerce")
        parts.append(ch[d < SEALED_FROM])          # 2024/2025 を読み込み時に破棄
    df = pd.concat(parts, ignore_index=True)
    df["date"] = pd.to_numeric(df["日付"], errors="coerce").astype(int)
    assert df["date"].max() < SEALED_FROM
    df["rid16"] = df["レースID(新/馬番無)"].astype(str).str.replace(r"\.0$", "", regex=True).str[:16]
    df["hid"] = df["血統登録番号"].astype(str).str.strip()
    df["fin"] = pd.to_numeric(df["着順"], errors="coerce")
    df["ban"] = pd.to_numeric(df["馬番"], errors="coerce")
    df["tou"] = pd.to_numeric(df["出走頭数"], errors="coerce")
    df["track"] = pd.to_numeric(df["トラックコード(JV)"], errors="coerce")
    df["surf"] = (df["芝・ダ"].astype(str).str.startswith("芝")).astype(int)
    df["dband"] = pd.to_numeric(df["距離"], errors="coerce").map(dist_band).astype(int)
    df["age"] = pd.to_numeric(df["年齢"], errors="coerce")
    df["year"] = df["date"] // 10000
    return df


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    timings = {"rss_mb_start": rss_mb()}
    t0 = time.time()
    m = load_master()
    timings["load_master_sec"] = time.time() - t0
    n_all = len(m)

    # ---- ID 監査 (fail-closed): 形式・欠損・馬名衝突
    id_audit = {
        "rows_total_2013_2023": int(n_all),
        "hid_missing_or_nan": int((m["hid"].isin(["", "nan", "None"]) | m["血統登録番号"].isna()).sum()),
        "hid_not_10_digits": int((~m["hid"].str.fullmatch(r"\d{10}")).sum()),
    }
    bad = m["hid"].isin(["", "nan", "None"]) | (~m["hid"].str.fullmatch(r"\d{10}"))
    m = m[~bad].copy()
    names_per_hid = m.groupby("hid")["馬名"].nunique()
    id_audit["hids_with_multiple_names"] = int((names_per_hid > 1).sum())
    id_audit["hids_total"] = int(len(names_per_hid))
    hids_per_name = m.groupby("馬名")["hid"].nunique()
    id_audit["names_shared_by_multiple_hids"] = int((hids_per_name > 1).sum())
    id_audit["note"] = "同一 ID に複数馬名 = 衝突 (fail-closed 対象)。同一馬名に複数 ID は正常 (馬名 join 禁止の根拠)"
    if id_audit["hids_with_multiple_names"] > 0:
        ex = names_per_hid[names_per_hid > 1].index[:5].tolist()
        id_audit["collision_examples"] = {h: m.loc[m.hid == h, "馬名"].unique().tolist() for h in ex}
    validate_hids(m["hid"].unique())        # 形式は通るはず (上で除外済み)

    # ---- 平地のみ (履歴も対象も)。障害 51-59 を除外
    is_jump = m["track"].between(51, 59)
    jump_races = int(m.loc[is_jump, "rid16"].nunique())
    flat = m[~is_jump].dropna(subset=["fin"]).copy()

    # ---- 正式 race set (2019-2023)
    g = flat[flat["year"].isin(DEV_YEARS)].groupby("rid16")
    meta = g.agg(date=("date", "first"), year=("year", "first"), n_fin=("hid", "size"), tou=("tou", "first"),
                 n_win=("fin", lambda s: int((s == 1).sum())), surf=("surf", "first"), dband=("dband", "first"))
    funnel = {"flat_races_2019_2023": int(len(meta))}
    ok = pd.Series(True, index=meta.index)
    c1 = meta["n_win"] == 1; funnel["excl_no_unique_winner"] = int((~c1).sum()); ok &= c1
    c2 = meta["n_fin"] >= 5; funnel["excl_small_field"] = int((ok & ~c2).sum()); ok &= c2
    c3 = meta["n_fin"] == meta["tou"]; funnel["excl_dnf_or_late_scratch_proxy(rows<出走頭数)"] = int((ok & ~c3).sum()); ok &= c3
    official = meta[ok].copy()
    funnel["official_races"] = int(len(official))
    funnel["official_by_year"] = official.groupby("year").size().to_dict()
    funnel["jump_races_excluded_2013_2023"] = jump_races
    funnel["note"] = ("DNF/締切後除外の proxy は master 行数 < 出走頭数。EXP16A (確定オッズ由来) の 2019-2023 正式 set は "
                      "3185/3169/3200/3206/3191 = 15,951R で、本監査値と突合すること")

    # ---- 履歴ストア (平地 finisher 全行 2013-2023; クエリ側で date<day を保証)
    t1 = time.time()
    runs = flat[["date", "rid16", "hid", "fin", "surf", "dband"]]
    store = build_history_store(runs)
    timings["build_store_sec"] = time.time() - t1
    timings["store_directed_meetings"] = int(len(store.h))
    timings["store_bytes_arrays"] = int(sum(a.nbytes for a in (store.h, store.c, store.date, store.won, store.surf, store.dband, store.ridc)))
    timings["rss_mb_after_store"] = rss_mb()

    # ---- 対象レースごとの pair 証拠 (day-start snapshot)
    tgt = flat[flat["rid16"].isin(official.index)].sort_values(["date", "rid16", "ban"])
    race_rows = []; horse_rows = []; pair_rows = []; edge_rows_2022 = []
    ages_all = []; match_all = []; nmeet_all = []
    t2 = time.time(); n_done = 0
    for rid, gr in tgt.groupby("rid16", sort=True):
        day = int(gr["date"].iloc[0]); yr = day // 10000
        hids = gr["hid"].tolist()
        res = pair_evidence(store, hids, day, int(gr["surf"].iloc[0]), int(gr["dband"].iloc[0]), PRM)
        n = len(hids)
        iu = np.triu_indices(n, 1)
        direct = res.direct[iu]
        has_co = res.n_common[iu] > 0
        never_met = ~direct
        s, unc, comps, resid = hodge_project(res.d, res.w, PRM.lam)
        comp_sizes = sorted([len(c) for c in comps if len(c) > 1], reverse=True)
        covered_h = ~unc
        prior = np.array([store.prior_starts(store.code_of[h], day) for h in hids])
        debut = prior == 0
        # curl share of pair evidence removed by projection (label-free diagnostic)
        obs = ~np.isnan(res.d)
        np.fill_diagonal(obs, False)
        if obs.any():
            dv = res.d[obs]; rv = resid[obs]
            curl_share = float((rv ** 2).sum() / max((dv ** 2).sum(), 1e-12))
        else:
            curl_share = np.nan
        race_rows.append(dict(
            rid16=rid, date=day, year=yr, n=n, surf=int(gr["surf"].iloc[0]), dband=int(gr["dband"].iloc[0]),
            pairs_total=int(len(iu[0])), pairs_direct=int(direct.sum()), pairs_never_met=int(never_met.sum()),
            pairs_never_met_with_co=int((never_met & has_co).sum()), pairs_direct_with_co=int((direct & has_co).sum()),
            n_common_median_never_met=float(np.median(res.n_common[iu][never_met & has_co])) if (never_met & has_co).any() else np.nan,
            horses_covered=int(covered_h.sum()), horses_debut=int(debut.sum()),
            nondebut_uncovered=int((unc & ~debut).sum()),
            n_components_ge2=len(comp_sizes), largest_component_share=(comp_sizes[0] / n) if comp_sizes else 0.0,
            curl_share=curl_share,
        ))
        for k, h in enumerate(hids):
            horse_rows.append(dict(rid16=rid, date=day, year=yr, hid=h, ban=int(gr["ban"].iloc[k]) if not np.isnan(gr["ban"].iloc[k]) else -1,
                                   age=float(gr["age"].iloc[k]), prior_starts=int(prior[k]),
                                   degree=int((~np.isnan(res.d[k])).sum() - 0), uncovered=bool(unc[k]),
                                   n_field=n, surf=int(gr["surf"].iloc[0])))
        for (i, j), lst in res.per_c.items():
            if res.direct[i, j]:
                continue
            pair_rows.append(dict(rid16=rid, date=day, year=yr, i=i, j=j, hid_i=hids[i], hid_j=hids[j],
                                  d_ij=float(res.d[i, j]), w_ij=float(res.w[i, j]), n_common=int(res.n_common[i, j]), n_field=n))
            if yr == 2022:
                for (c, dc, om) in lst:
                    edge_rows_2022.append(dict(rid16=rid, date=day, i=i, j=j, hid_i=hids[i], hid_j=hids[j],
                                               hid_c=str(store.hid_of[c]), d_c=float(dc), omega=float(om)))
        ages_all.extend(res.edge_age_days); match_all.extend(res.cond_match); nmeet_all.extend(res.n_meet_hc)
        n_done += 1
        if n_done % 2000 == 0:
            print(f"  {n_done} races  {time.time() - t2:.0f}s", flush=True)
    timings["query_races"] = n_done
    timings["query_sec_total"] = time.time() - t2
    timings["query_sec_per_race"] = timings["query_sec_total"] / max(n_done, 1)
    timings["rss_mb_end"] = rss_mb()

    R = pd.DataFrame(race_rows); Hh = pd.DataFrame(horse_rows); P = pd.DataFrame(pair_rows); E22 = pd.DataFrame(edge_rows_2022)
    R.to_parquet(OUT / "races_2019_2023.parquet", index=False)
    Hh.to_parquet(OUT / "horses_2019_2023.parquet", index=False)
    P.to_parquet(OUT / "pairs_2019_2023.parquet", index=False)
    E22.to_parquet(OUT / "edges_2022.parquet", index=False)

    # ---- 集計
    def pair_summary(sub: pd.DataFrame) -> dict:
        tot = int(sub["pairs_total"].sum()); nm = int(sub["pairs_never_met"].sum()); nmco = int(sub["pairs_never_met_with_co"].sum())
        return {
            "races": int(len(sub)), "pairs_total": tot, "pairs_direct": int(sub["pairs_direct"].sum()),
            "pairs_never_met": nm, "pairs_never_met_with_common_opponent": nmco,
            "never_met_pair_common_opponent_rate": (nmco / nm) if nm else None,
            "race_rate_with_half_of_all_pairs_covered": float(((sub["pairs_never_met_with_co"] + sub["pairs_direct_with_co"]) / sub["pairs_total"] >= 0.5).mean()) if len(sub) else None,
            "race_rate_with_half_of_never_met_pairs_covered": float((sub["pairs_never_met_with_co"] / sub["pairs_never_met"].clip(lower=1) >= 0.5).mean()) if len(sub) else None,
            "median_n_common_per_covered_never_met_pair": float(sub["n_common_median_never_met"].median()) if len(sub) else None,
            "components_ge2_mean": float(sub["n_components_ge2"].mean()) if len(sub) else None,
            "largest_component_share_mean": float(sub["largest_component_share"].mean()) if len(sub) else None,
            "race_rate_single_component_covering_all": float((sub["largest_component_share"] >= 0.999).mean()) if len(sub) else None,
            "curl_share_of_pair_evidence_removed_by_projection_median": float(sub["curl_share"].median()) if len(sub) else None,
        }

    def horse_summary(sub: pd.DataFrame) -> dict:
        nd = sub[sub["prior_starts"] > 0]
        return {"horses": int(len(sub)), "debut": int((sub["prior_starts"] == 0).sum()),
                "covered_rate_all": float((~sub["uncovered"]).mean()) if len(sub) else None,
                "non_debut_graph_membership_rate": float((~nd["uncovered"]).mean()) if len(nd) else None,
                "degree_median": float(sub["degree"].median()) if len(sub) else None}

    cov = {"params_stage0_provisional": PRM.__dict__, "history_scope": f"JRA平地 finisher 行 {HIST_MIN_YEAR}-2023 (date<day)",
           "overall": pair_summary(R), "horses_overall": horse_summary(Hh)}
    cov["by_year"] = {int(y): pair_summary(s) for y, s in R.groupby("year")}
    cov["horses_by_year"] = {int(y): horse_summary(s) for y, s in Hh.groupby("year")}
    R["field_band"] = R["n"].map(lambda v: band(v, FIELD_BINS)); cov["by_field_band"] = {k: pair_summary(s) for k, s in R.groupby("field_band")}
    cov["by_surface"] = {("芝" if k == 1 else "ダ"): pair_summary(s) for k, s in R.groupby("surf")}
    Hh["age_band"] = Hh["age"].map(lambda v: band(v, AGE_BANDS) if v == v else "na"); cov["horses_by_age_band"] = {k: horse_summary(s) for k, s in Hh.groupby("age_band")}
    Hh["career_band"] = Hh["prior_starts"].map(lambda v: band(v, CAREER_BANDS)); cov["horses_by_career_band"] = {k: horse_summary(s) for k, s in Hh.groupby("career_band")}
    ages = np.array(ages_all); mt = np.array(match_all); nm_ = np.array(nmeet_all)
    cov["edge_age_days_used"] = {"p10": float(np.percentile(ages, 10)), "p50": float(np.percentile(ages, 50)), "p90": float(np.percentile(ages, 90)), "mean": float(ages.mean()), "n": int(len(ages))}
    cov["condition_match_rate_used_edges"] = {"mean": float(mt.mean()), "share_zero": float((mt == 0).mean()), "share_one": float((mt == 1).mean())}
    cov["meetings_per_used_hc_edge"] = {"p50": float(np.percentile(nm_, 50)), "p90": float(np.percentile(nm_, 90)), "share_eq1": float((nm_ == 1).mean()), "share_le2": float((nm_ <= 2).mean()), "mean": float(nm_.mean())}
    cov["provisional_floors_from_spec"] = {"never_met_pair_common_opponent_rate": 0.5, "race_rate_with_half_pairs_covered": 0.8,
                                            "non_debut_starter_graph_membership": 0.9, "major_band_relative_coverage_floor": 0.5}
    o = cov["overall"]; hh = cov["horses_overall"]
    major = [v["never_met_pair_common_opponent_rate"] for k, v in cov["by_field_band"].items() if v["races"] > 500] + \
            [v["never_met_pair_common_opponent_rate"] for v in cov["by_surface"].values()]
    cov["floor_check"] = {
        "never_met_pair_common_opponent_rate": {"value": o["never_met_pair_common_opponent_rate"], "floor": 0.5, "pass": o["never_met_pair_common_opponent_rate"] >= 0.5},
        "race_rate_with_half_pairs_covered": {"value": o["race_rate_with_half_of_all_pairs_covered"], "floor": 0.8, "pass": o["race_rate_with_half_of_all_pairs_covered"] >= 0.8},
        "non_debut_starter_graph_membership": {"value": hh["non_debut_graph_membership_rate"], "floor": 0.9, "pass": hh["non_debut_graph_membership_rate"] >= 0.9},
        "major_band_relative_coverage_floor": {"min_major_band_rate": min(major), "overall_rate": o["never_met_pair_common_opponent_rate"],
                                                "pass": min(major) >= 0.5 * o["never_met_pair_common_opponent_rate"]},
    }
    cov["floor_check"]["all_pass"] = all(v["pass"] for v in cov["floor_check"].values() if isinstance(v, dict))
    cov["id_audit"] = id_audit
    (OUT / "GRAPH_COVERAGE.json").write_text(json.dumps(cov, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    (OUT / "race_population.json").write_text(json.dumps(funnel, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    timings["pairs_rows_saved"] = int(len(P)); timings["edges_2022_rows_saved"] = int(len(E22))
    (OUT / "compute_timings.json").write_text(json.dumps(timings, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps({"funnel": funnel, "overall": cov["overall"], "horses": cov["horses_overall"], "floor_check": cov["floor_check"],
                      "meetings_per_edge": cov["meetings_per_used_hc_edge"], "timings": timings}, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
