# -*- coding: utf-8 -*-
"""
population.py — EXP19 Stage 0 S0-B: 母集団・被覆・特徴 artifact (結果性能を計算しない)
=====================================================================================
1. 既知の構造値を再現: 2019-2023 平地 231,068 行 / 16,645R、current kg 有効率 99.807%、血統登録番号欠損 0
2. starter 集合 = terminal 単勝オッズ > 1.0 の馬番 (EXP16A/18 と同定義)。torch の非 starter 行 (取消・除外) は
   体重履歴の slot に入れない。DNF 馬は starter なので実測体重があれば slot に残る
3. 主母集団 = EXP16A 正式 race set (2019-2023、DNF 含有 race 除外、15,951R) を基盤に、全 starter が torch 行・
   pre/terminal 単勝オッズを持つ race。基盤との race ID 差分を報告
4. full-starter 感度母集団 = 平地・starter>=5・勝馬一意・DNF 含有 race も残す。N1 は DNF 馬に無い (0 埋めしない)
5. 年別の race/馬行数、体重 status、履歴数、馬場値被覆、WP 同時利用可能率 (2021-2023 各年 >= 90% が floor)
6. 層別被覆 (競馬場・芝ダ・年齢・休養帯・人気集中帯) で全体の 0.8 倍未満の層
7. raw input・race set・特徴 manifest の sha256
結果 loader (着順) は finisher 判定と勝馬一意性にだけ使い、性能は計算しない。
出力: out/population_coverage.json、out/stage0_manifest.json、data/_research/mcond/exp19/features_2013_2023.parquet
"""
from __future__ import annotations

import hashlib
import json
import sys
import time

import numpy as np
import pandas as pd

from . import features as F
from .loaders import (BABA, BASE, MASTER, OUT, RESEARCH, TORCH, is_jump, load_finishers, load_market,
                      load_torch_struct, loader_sha256, sha256_file)

EVAL = [2019, 2020, 2021, 2022, 2023]
WP_YEARS = [2021, 2022, 2023]
KNOWN = {"rows": 231068, "races": 16645, "valid_kg_rate": 0.998069832257171, "pid_missing": 0}
REF16 = BASE / "data" / "_research" / "mcond" / "exp16a" / "official_rids_by_year.json"
FEAT_PARQUET = RESEARCH / "features_2013_2023.parquet"


def sha_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def main():
    t0 = time.time()
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    RESEARCH.mkdir(parents=True, exist_ok=True)
    t = load_torch_struct()
    flat = ~is_jump(t["トラックコード(JV)"])

    # ---- 1 既知構造値の再現 (DNF 除外前、全 torch 行)
    e = t[t["year"].between(2019, 2023) & flat]
    known = {"rows": int(len(e)), "races": int(e["rid16"].nunique()),
             "valid_kg_rate": float(e["kg"].between(200, 800).mean()),
             "pid_missing": int((e["pid"].isin(["", "nan", "None"]) | e["pid"].isna()).sum())}
    known["reproduced"] = (known["rows"] == KNOWN["rows"] and known["races"] == KNOWN["races"]
                           and abs(known["valid_kg_rate"] - KNOWN["valid_kg_rate"]) < 1e-12
                           and known["pid_missing"] == KNOWN["pid_missing"])
    print(f"[known] {known}", flush=True)

    # ---- 2 starter 集合 (TANPUK terminal 単勝 > 1.0)
    idx, Wm = load_market(range(2013, 2024))
    term = {}
    pre_ok = {}
    for rid, r in idx.iterrows():
        w = Wm[int(r["term_i"])]
        bans = np.flatnonzero(np.isfinite(w) & (w > 1.0)) + 1
        term[rid] = set(int(b) for b in bans)
        if r["pre_i"] >= 0:
            wp = Wm[int(r["pre_i"])]
            pre_ok[rid] = bool(np.all(np.isfinite(wp[bans - 1]) & (wp[bans - 1] > 1.0)))
        else:
            pre_ok[rid] = False
    in_mkt = t["rid16"].isin(term.keys())
    is_starter = np.array([(b in term[r]) if r in term else True for r, b in zip(t["rid16"], t["ban"])])
    t["starter"] = is_starter
    t["in_market"] = in_mkt.to_numpy()
    nonstarter = t[t["in_market"] & ~t["starter"]]
    ts = F.starter_rows(t.drop(columns=["starter", "in_market"]), term)   # 履歴 slot は starter 行だけ
    assert len(ts) == int(t["starter"].sum())
    print(f"[starter] torch rows {len(t):,} non-starter rows dropped {len(nonstarter):,} ({time.time()-t0:.0f}s)",
          flush=True)

    # ---- W / P / WP
    w = F.build_w(ts)
    baba = pd.read_parquet(BABA)
    pz = F.build_p(baba)
    races = ts.groupby("rid16", sort=False).agg(date=("date", "first"), 場所=("場所", "first"),
                                                芝ダ=("芝・ダ", "first"), year=("year", "first"),
                                                track=("トラックコード(JV)", "first"),
                                                n_rows=("ban", "size"), n_measured=("kg", lambda s: int(s.between(200, 800).sum())))
    races = races.rename(columns={"芝ダ": "芝・ダ"}).reset_index()
    rp = F.race_p(races[["rid16", "date", "場所", "芝・ダ"]], pz)
    races = races.merge(rp[["rid16", "surface", "cushion_z", "moist_gp_z", "moist_4c_z", "moist_gradient_z",
                            "track_extreme_z", "p_available"]], on="rid16", how="left")
    rp_rows = ts[["rid16"]].merge(races[["rid16", "surface", "cushion_z", "moist_gp_z", "moist_gradient_z",
                                         "track_extreme_z", "p_available"]], on="rid16", how="left")
    rp_rows.index = ts.index
    wp = F.build_wp(w, rp_rows)
    feat = pd.concat([ts[["rid16", "ban", "pid", "date", "year", "場所", "芝・ダ", "性別", "年齢", "kg"]], w,
                      rp_rows[["surface", "cushion_z", "moist_gp_z", "moist_gradient_z", "track_extreme_z",
                               "p_available"]], wp], axis=1)
    feat["all_measured_race"] = feat.groupby("rid16")["bw_status_not_measured"].transform("max") == 0
    feat.to_parquet(FEAT_PARQUET, index=False)

    # ---- 3 主母集団
    ref = json.loads(REF16.read_text(encoding="utf-8"))
    fin = load_finishers(20231231)
    fin_sets = fin.groupby("rid16")["ban"].apply(lambda s: set(int(x) for x in s))
    n_first = fin[fin["jyun"] == 1].groupby("rid16").size()
    torch_keys = set(zip(ts["rid16"], ts["ban"]))
    main, diffs = {}, {}
    for y in EVAL:
        base = set(ref[str(y)])
        keep, reasons = [], {}
        for rid in sorted(base):
            st = term.get(rid)
            if st is None:
                reasons.setdefault("no_market", []).append(rid); continue
            if not pre_ok.get(rid, False):
                reasons.setdefault("pre_odds_incomplete", []).append(rid); continue
            if not all((rid, b) in torch_keys for b in st):
                reasons.setdefault("starter_missing_torch_row", []).append(rid); continue
            keep.append(rid)
        main[y] = keep
        diffs[str(y)] = {"base_exp16a": len(base), "main": len(keep),
                         "excluded": {k: len(v) for k, v in reasons.items()},
                         "excluded_rids": reasons}
    main_all = [r for y in EVAL for r in main[y]]

    # ---- 4 full-starter 感度母集団 (DNF 含有 race を残す)
    fs = {}
    for y in EVAL:
        rr = races[(races["year"] == y) & ~is_jump(races["track"])]
        n_ok = n_dnf = n_h_no_n1 = 0
        for rid in rr["rid16"]:
            st = term.get(rid)
            if st is None or len(st) < 5 or n_first.get(rid, 0) != 1 or not pre_ok.get(rid, False):
                continue
            n_ok += 1
            dnf = st - fin_sets.get(rid, set())
            if dnf:
                n_dnf += 1
                n_h_no_n1 += len(dnf)
        fs[str(y)] = {"races": n_ok, "races_with_dnf": n_dnf, "dnf_horses_without_N1": n_h_no_n1,
                      "N1_rule": "DNF 馬は master 行が無く N1 score を持たない。0 埋めしない。"
                                 "感度分析は market-only と W-only の比較だけ実行可能"}

    # ---- 5 年別被覆 (主母集団)
    mf = feat[feat["rid16"].isin(set(main_all))]
    per_year = {}
    for y in EVAL:
        g = mf[mf["year"] == y]
        rg = g.groupby("rid16").agg(p=("p_available", "first"), allm=("all_measured_race", "first"),
                                    surf=("surface", "first"))
        joint = rg["p"].astype(bool) & rg["allm"].astype(bool)
        per_year[str(y)] = {
            "races": int(len(rg)), "horse_rows": int(len(g)),
            "status_measured": int((g["bw_status_not_measured"] == 0).sum()),
            "status_not_measured": int((g["bw_status_not_measured"] == 1).sum()),
            "history_n_distribution": {str(k): int(v) for k, v in g["bw_history_n"].value_counts().sort_index().items()},
            "robust_z5_available_rate": float(g["bw_robust_z5"].notna().mean()),
            "change_x_layoff_available_rate": float(g["bw_change_x_layoff"].notna().mean()),
            "sex_age_z_available_rate": float(g["bw_sex_age_z"].notna().mean()),
            "race_p_available_rate": float(rg["p"].astype(bool).mean()),
            "race_cushion_available_rate_turf": float(g[g["surface"] == "turf"].groupby("rid16")["cushion_z"].first().notna().mean()),
            "race_all_starters_measured_rate": float(rg["allm"].astype(bool).mean()),
            "wp_joint_available_rate": float(joint.mean()),
            "chg_check": {k: int(v) for k, v in g["chg_check"].value_counts().items()},
        }
    wp_floor = {str(y): per_year[str(y)]["wp_joint_available_rate"] >= 0.90 for y in WP_YEARS}

    # ---- 6 層別被覆 (0.8 倍未満の層を報告)
    mf = mf.copy()
    mf["age_band"] = pd.cut(pd.to_numeric(mf["年齢"], errors="coerce"), [0, 2, 3, 4, 99], labels=["2", "3", "4", "5+"])
    mf["layoff_band"] = pd.cut(mf["layoff_days"], [-1, 14, 35, 90, 180, 10_000],
                               labels=["<=14", "15-35", "36-90", "91-180", ">180"]).astype(str).replace("nan", "first_run")
    ent = {}
    for rid in set(main_all):
        wv = Wm[int(idx.loc[rid, "pre_i"])]
        st = np.array(sorted(term[rid])) - 1
        p = (1 / wv[st]) / (1 / wv[st]).sum()
        ent[rid] = float(-(p * np.log(p)).sum())
    cuts = np.quantile(list(ent.values()), [1 / 3, 2 / 3])
    mf["conc_band"] = mf["rid16"].map(lambda r: ["concentrated", "middle", "open"][int(np.searchsorted(cuts, ent[r]))])
    mf["wp_joint"] = mf["p_available"].astype(bool) & mf["all_measured_race"].astype(bool)
    targets = {"measured": (mf["bw_status_not_measured"] == 0), "robust_z5": mf["bw_robust_z5"].notna(),
               "wp_joint_2021_2023": mf["wp_joint"]}
    layers = {}
    for name, flag in targets.items():
        base_mask = mf["year"].isin(WP_YEARS) if name.startswith("wp") else pd.Series(True, index=mf.index)
        overall = float(flag[base_mask].mean())
        low = {}
        for col in ["場所", "surface", "age_band", "layoff_band", "conc_band"]:
            for k, v in flag[base_mask].groupby(mf.loc[base_mask, col].astype(str)).mean().items():
                if v < 0.8 * overall:
                    low[f"{col}={k}"] = round(float(v), 4)
        layers[name] = {"overall": overall, "layers_below_0.8x": low}

    # ---- 7 manifest
    tan_files = sorted((BASE / "data" / "Time _series_odds").glob("TANPUK_*.csv"))
    manifest = {
        "raw_input_sha256": {"torch": sha256_file(TORCH), "baba_feats": sha256_file(BABA), "master_v2": sha256_file(MASTER),
                             **{f.name: sha256_file(f) for f in tan_files}},
        "loader_sha256": loader_sha256(),
        "features_py_sha256": sha256_file(F.__file__ if hasattr(F, "__file__") else ""),
        "feature_manifest": {"W_main": F.W_MAIN, "W_missing_indicators": F.W_MISS, "W_descriptive_only": F.W_DESCRIPTIVE,
                             "WP_columns": F.WP_COLS, "params": {"MAD_FLOOR_KG": F.MAD_FLOOR_KG, "WINSOR_Z": F.WINSOR_Z,
                                                                  "MIN_HIST": F.MIN_HIST, "P_MIN_HIST": F.P_MIN_HIST,
                                                                  "SEX_AGE_MIN_N": F.SEX_AGE_MIN_N, "FILL": F.FILL},
                             "feature_list_sha256": sha_text("\n".join(F.W_MAIN + F.W_MISS + F.WP_COLS))},
        "race_set": {"main_2019_2023_sha256": sha_text("\n".join(sorted(main_all))), "n": len(main_all),
                     **{f"main_{y}_sha256": sha_text("\n".join(sorted(main[y]))) for y in EVAL}},
        "features_parquet": str(FEAT_PARQUET.relative_to(BASE)), "features_parquet_sha256": sha256_file(FEAT_PARQUET),
    }
    res = {
        "role": "EXP19 Stage 0 S0-B 母集団・被覆 (結果性能は計算しない)",
        "known_structure_reproduction": known,
        "starter_filter": {"torch_rows_2013_2023": int(len(t)), "non_starter_rows_dropped": int(len(nonstarter)),
                           "rows_in_races_without_market": int((~t["in_market"]).sum())},
        "main_population": {"per_year": {y: {k: v for k, v in d.items() if k != "excluded_rids"} for y, d in diffs.items()},
                            "total": len(main_all), "base_exp16a_total": sum(len(ref[str(y)]) for y in EVAL),
                            "excluded_rids": {y: d["excluded_rids"] for y, d in diffs.items()}},
        "full_starter_sensitivity": fs,
        "per_year_main": per_year,
        "wp_joint_floor_0.90_each_year": wp_floor, "wp_joint_floor_pass": all(wp_floor.values()),
        "entropy_cuts_pre_market": cuts.tolist(),
        "layer_coverage": layers,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (OUT / "population_coverage.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    (OUT / "stage0_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"known": known, "main_total": len(main_all), "wp_floor": wp_floor,
                      "per_year_wp": {y: per_year[y]["wp_joint_available_rate"] for y in per_year}}, ensure_ascii=False))
    print(f"[saved] population_coverage.json ({res['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
