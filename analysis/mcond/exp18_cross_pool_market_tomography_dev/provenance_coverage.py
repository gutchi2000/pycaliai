# -*- coding: utf-8 -*-
"""
provenance_coverage.py — EXP18 S0-B/S0-D: provenance・schema・coverage (結果を使わない)
=====================================================================================
市場構造 loader だけを使う。着順・払戻・realized top2 は読まない (DNF・同着の件数は
≤2018 分だけ結果 loader で別集計し、2019-2023 は Stage 1 で適用すると記録する)。

事前に固定した coverage の定義:
  予定件数 (scheduled)      master のレース情報で平地 (トラックコード(JV) 51..59 以外) かつ 出走頭数 >= 5
  base race set             TANPUK terminal があり base_race_status == 'base'
                            (starter = terminal 単勝 > 1.0、starter >= 5、全 starter が複勝 Lo/Hi を持つ)
  eligible                  base ∩ UMAREN terminal あり ∩ UMAREN 格子が C(starter,2) で完全
  terminal formal-race coverage = eligible / scheduled                     (暫定 floor 0.95)
  全組合せ完全 race          = 格子完全 / (base ∩ UMAREN terminal あり)       (暫定 floor 0.99)
  pre/terminal 同時 coverage = (eligible ∩ 両プールに pre snapshot) / eligible (暫定 floor 0.90)
出力: POOL_SCHEMA_MANIFEST.json, RACE_AND_TICKET_COVERAGE.json, out/provenance_times.json
実行: python -m analysis.mcond.exp18_cross_pool_market_tomography_dev.provenance_coverage
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from .loaders import (HERE, ODIR, OUT, RESULT_MAX_YEAR, SEALED_FROM_YEAR, load_outcomes,
                      load_structure, realized_top2, sha256)
from .market_build import (base_race_status, classify_zero_cells, pool_matrices, race_arrays,
                           snapshot_index, tan_entropy, umaren_grid_complete)

YEARS = list(range(2013, 2024))
COV_YEARS = [2019, 2020, 2021, 2022, 2023]
FLOORS = {"terminal_race_coverage": 0.95, "complete_pair_grid": 0.99, "pre_terminal_joint": 0.90}
FIELD_BANDS = [(5, 8), (9, 12), (13, 15), (16, 18)]


def file_manifest():
    files = {}
    for f in sorted(ODIR.glob("*.csv")):
        rec = {"path": str(f), "size_bytes": f.stat().st_size, "sha256": sha256(f), "encoding": "cp932"}
        if f.name.startswith(("TANPUK_", "UMAREN_")):
            df = pd.read_csv(f, encoding="cp932", low_memory=False)
            rid = df.iloc[:, 0].astype(str).str.replace(r"\D", "", regex=True).str[:16]
            yr = rid.str[:4].astype(int)
            sealed = int((yr >= SEALED_FROM_YEAR).sum())
            keep = yr < SEALED_FROM_YEAR
            rec.update({
                "header": True, "n_columns": int(df.shape[1]),
                "columns_head": [str(c) for c in df.columns[:8]],
                "columns_tail": [str(c) for c in df.columns[-3:]],
                "rows_total": int(len(df)), "rows_2024_2025_discarded_uninspected": sealed,
                "years_kept": [int(yr[keep].min()), int(yr[keep].max())],
                "rows_by_year_kept": {str(k): int(v) for k, v in yr[keep].value_counts().sort_index().items()},
                "kubun_counts_kept": {str(k): int(v) for k, v in
                                      df.loc[keep.to_numpy(), df.columns[1]].value_counts().items()},
                "time_column": "月日時分 = MMDDHHMM (8 桁、前日 23 時台の行を含む)",
            })
            if f.name.startswith("TANPUK_"):
                rec["pool_kinds"] = "単勝 (n単) / 複勝 Lo・Hi (n複Lo, n複Hi)、n=1..18"
                rec["vote_columns"] = {"単勝票数": "単勝プール全体の票数 (馬別ではない)",
                                       "複勝票数": "複勝プール全体の票数 (馬別ではない)"}
            else:
                rec["pool_kinds"] = "馬連 全 153 組 (馬ii-jj, i<j<=18)"
                rec["vote_columns"] = {"馬連票数": "馬連プール全体の票数 (組合せ別ではない)"}
            del df
        else:
            with open(f, "rb") as fh:
                nlines = sum(1 for _ in fh)
            head = pd.read_csv(f, encoding="cp932", header=None, nrows=3, low_memory=False)
            rec.update({"header": False, "n_columns": int(head.shape[1]), "rows_total_lines": nlines,
                        "key_format_example": str(head.iloc[0, 0]).strip(),
                        "decision": "EXP18 では使わない。ヘッダ無し、race key が 16 桁 race_id と別体系 "
                                    "(例 0620581201)、区分・記録時刻の列が同定できないため、列・券種・時点の監査を "
                                    "通すまで入力にしない (spec §2.1)"})
        files[f.name] = rec
    return files


def band_of(n):
    for lo, hi in FIELD_BANDS:
        if lo <= n <= hi:
            return f"{lo}-{hi}"
    return "other"


def main():
    t0 = time.time()
    manifest = {"role": "EXP18 Stage 0 の市場データ provenance と schema。2024/2025 の行は読み込み直後に破棄し中身を見ない",
                "files": file_manifest()}
    st = load_structure(YEARS)
    tan, um, info = st["tan"], st["um"], st["info"]
    W, LO, HI, U = pool_matrices(tan, um)
    idx = snapshot_index(tan, um, info)

    # ---- 同一時点結合
    join = {"terminal_tan_rows": int(idx["term_tan"].notna().sum()),
            "terminal_um_matched": int((idx["term_um"].fillna(-1) >= 0).sum()),
            "pre_tan_rows": int(idx["pre_tan"].notna().sum()),
            "pre_um_matched": int((idx["pre_um"].fillna(-1) >= 0).sum()),
            "am9_tan_rows": int(idx["am9_tan"].notna().sum()),
            "am9_um_matched": int((idx["am9_um"].fillna(-1) >= 0).sum())}

    # ---- snapshot 時刻 (発走時刻との差)
    def q(s):
        s = pd.to_numeric(s, errors="coerce").dropna()
        return {"p01": float(s.quantile(.01)), "p50": float(s.median()), "p99": float(s.quantile(.99)),
                "n": int(len(s))} if len(s) else None
    idx["year"] = idx.index.str[:4].astype(int)
    times = {str(y): {"terminal_minus_post_min": q(g["term_minus_post"]),
                      "pre_minus_post_min": q(g["pre_minus_post"])}
             for y, g in idx.groupby("year")}

    # ---- race ごとの分類
    rows = []
    zero_tot = {}
    vote_mono = {"races": 0, "win_total_increases": 0, "umaren_total_increases": 0}
    for rid, r in idx.iterrows():
        inf = info.loc[rid] if rid in info.index else None
        y = int(rid[:4])
        rec = {"rid16": rid, "year": y,
               "venue": (inf["venue"] if inf is not None else None),
               "scheduled": bool(inf is not None and not bool(inf["is_jump"])
                                 and (inf["entrants_master"] >= 5)),
               "is_jump": bool(inf["is_jump"]) if inf is not None else None}
        if not (r.get("term_tan") == r.get("term_tan")):
            rec["status"] = "no_tanpuk_terminal"
            rows.append(rec)
            continue
        at = race_arrays(W, LO, HI, U, int(r["term_tan"]), int(r["term_um"]))
        rec["n"] = at["n"]
        rec["field_band"] = band_of(at["n"])
        rec["status"] = base_race_status(inf, at)
        rec["umaren_terminal"] = int(r["term_um"]) >= 0
        rec["grid_complete"] = bool(rec["umaren_terminal"] and umaren_grid_complete(at))
        rec["theoretical_pairs"] = at["n"] * (at["n"] - 1) // 2
        rec["valid_pairs"] = int(np.sum(np.isfinite(at["umaren"]) & (at["umaren"] >= 1.0)))
        rec["entropy_tan"] = tan_entropy(at["win"]) if at["n"] >= 2 else np.nan
        rec["pre_both"] = bool(r.get("pre_tan") == r.get("pre_tan") and r.get("pre_um", -1) == r.get("pre_um", -1)
                               and int(r.get("pre_um", -1)) >= 0)
        rec["votes_win_total"] = float(pd.to_numeric(tan.iloc[int(r["term_tan"])]["votes_win_total"], errors="coerce"))
        rec["votes_umaren_total"] = (float(pd.to_numeric(um.iloc[int(r["term_um"])]["votes_umaren_total"],
                                                         errors="coerce")) if rec["umaren_terminal"] else np.nan)
        if rec["umaren_terminal"]:
            tou = int(pd.to_numeric(um.iloc[int(r["term_um"])]["tou"], errors="coerce"))
            z = classify_zero_cells(U[int(r["term_um"])], at["bans"], tou)
            for k2, v2 in z.items():
                zero_tot.setdefault(str(y), {}).setdefault(k2, 0)
                zero_tot[str(y)][k2] += v2
        if rec["pre_both"]:
            vote_mono["races"] += 1
            pw = float(pd.to_numeric(tan.iloc[int(r["pre_tan"])]["votes_win_total"], errors="coerce"))
            pu = float(pd.to_numeric(um.iloc[int(r["pre_um"])]["votes_umaren_total"], errors="coerce"))
            vote_mono["win_total_increases"] += int(rec["votes_win_total"] >= pw)
            vote_mono["umaren_total_increases"] += int(rec["umaren_terminal"] and rec["votes_umaren_total"] >= pu)
        rows.append(rec)
    R = pd.DataFrame(rows)
    R["eligible"] = (R["status"] == "base") & R["grid_complete"].fillna(False)
    R["base_um"] = (R["status"] == "base") & R["umaren_terminal"].fillna(False)

    # scheduled だが TANPUK に居ない race (master にあって市場データに無い)
    sched_info = info[(~info["is_jump"].astype(bool)) & (info["entrants_master"] >= 5)]
    missing_market = sorted(set(sched_info.index) - set(R["rid16"]))

    def cov_block(g, sched_n=None):
        sched = int(g["scheduled"].sum()) if sched_n is None else sched_n
        elig = int(g["eligible"].sum())
        base_um = int(g["base_um"].sum())
        grid = int((g["base_um"] & g["grid_complete"].fillna(False)).sum())
        pre = int((g["eligible"] & g["pre_both"].fillna(False)).sum())
        return {"scheduled": sched, "base": int((g["status"] == "base").sum()),
                "base_with_umaren_terminal": base_um, "grid_complete": grid,
                "eligible": elig,
                "excluded_umaren_settlement_unavailable": int(((g["status"] == "base") & ~g["eligible"]).sum()),
                "terminal_race_coverage": elig / sched if sched else None,
                "complete_pair_grid": grid / base_um if base_um else None,
                "pre_terminal_joint": pre / elig if elig else None,
                "pool_total_vote_coverage": float(g.loc[g["eligible"], "votes_umaren_total"].notna().mean())
                if elig else None}

    per_year = {}
    for y in YEARS:
        g = R[R["year"] == y]
        s_missing = sum(1 for rid in missing_market if rid.startswith(str(y)))
        blk = cov_block(g, int(g["scheduled"].sum()) + s_missing)
        blk["scheduled_missing_from_market_files"] = s_missing
        blk["status_counts"] = {k: int(v) for k, v in g["status"].value_counts().items()}
        per_year[str(y)] = blk
    cov = R[R["year"].isin(COV_YEARS)]
    ent_cut = list(np.quantile(cov.loc[cov["eligible"], "entropy_tan"].dropna(), [1 / 3, 2 / 3]))

    def ent_band(e):
        if not (e == e):
            return "unknown"
        return "low" if e <= ent_cut[0] else ("mid" if e <= ent_cut[1] else "high")
    cov = cov.assign(entropy_band=cov["entropy_tan"].map(ent_band))
    breakdowns = {}
    for col in ["venue", "field_band", "entropy_band"]:
        breakdowns[col] = {str(k): cov_block(g) for k, g in cov.groupby(col, dropna=False)}
    pooled = cov_block(cov, int(cov["scheduled"].sum()) +
                       sum(1 for rid in missing_market if int(rid[:4]) in COV_YEARS))
    floors = {k: {"value": pooled[k], "floor": v, "pass": bool(pooled[k] is not None and pooled[k] >= v)}
              for k, v in FLOORS.items()}
    yearly_floor = {str(y): {k: bool(per_year[str(y)][k] is not None and per_year[str(y)][k] >= v)
                             for k, v in FLOORS.items()} for y in COV_YEARS}

    # ---- 結果条件付き除外 (≤2018 だけ結果 loader で集計)
    outc = load_outcomes(RESULT_MAX_YEAR)
    top = realized_top2(outc).set_index("rid16")
    le18 = R[(R["year"] <= RESULT_MAX_YEAR) & R["eligible"]]
    oc = {}
    for y, g in le18.groupby("year"):
        t_ = top.reindex(g["rid16"])
        oc[str(y)] = {"eligible": int(len(g)), "dnf_races": int((t_["dnf"] > 0).sum()),
                      "top2_dead_heat_races": int(t_["dead_heat_top2"].fillna(False).sum()),
                      "no_outcome_row": int(t_["dnf"].isna().sum())}

    coverage = {
        "role": "結果ラベルを使わない coverage (2019-2023 は市場構造 loader のみ)。DNF/同着は ≤2018 だけ別集計",
        "definitions": {
            "scheduled": "master レース情報で平地・出走頭数>=5 (市場ファイルに無い race も数える)",
            "base": "TANPUK terminal あり・starter(単勝>1.0)>=5・全 starter が複勝 Lo/Hi を持つ・平地",
            "eligible": "base ∩ UMAREN terminal あり ∩ 格子が C(starter,2) で完全 (odds>=1.0)",
            "terminal_race_coverage": "eligible / scheduled",
            "complete_pair_grid": "格子完全 / (base ∩ UMAREN terminal あり)",
            "pre_terminal_joint": "(eligible ∩ 両プールの pre snapshot が同一時刻で存在) / eligible",
            "entropy_band": f"terminal 単勝 entropy の三分位 (2019-2023 eligible で算出: {ent_cut}). UMAREN は使わない",
        },
        "floors_provisional": FLOORS,
        "pooled_2019_2023": pooled,
        "floor_check_pooled_2019_2023": floors,
        "floor_check_by_year": yearly_floor,
        "all_floors_pass_pooled": all(v["pass"] for v in floors.values()),
        "per_year": per_year,
        "breakdowns_2019_2023": breakdowns,
        "zero_cell_classification_terminal_by_year": zero_tot,
        "same_time_join": join,
        "pool_total_votes_monotone_pre_to_terminal": vote_mono,
        "outcome_conditioned_exclusions": {
            "le2018_measured": oc,
            "2019_2023": "DNF・1着2着同着の除外は結果条件付きなので Stage 1 で結果 loader により適用する (Stage 0 では未計測)",
        },
        "post_close_refund": "締切後の取消・返還は市場表示に現れない。EXP16A の監査では締切後除外の痕跡 0 件",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (HERE / "POOL_SCHEMA_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1),
                                                    encoding="utf-8")
    (HERE / "RACE_AND_TICKET_COVERAGE.json").write_text(
        json.dumps(coverage, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    (OUT / "provenance_times.json").write_text(json.dumps({"snapshot_times": times, "join": join},
                                                          ensure_ascii=False, indent=1, default=float),
                                               encoding="utf-8")
    print(json.dumps(pooled, ensure_ascii=False))
    print(json.dumps(floors, ensure_ascii=False))
    print(json.dumps(join, ensure_ascii=False))
    print(f"[saved] POOL_SCHEMA_MANIFEST.json / RACE_AND_TICKET_COVERAGE.json ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
