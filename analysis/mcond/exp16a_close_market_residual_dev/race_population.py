# -*- coding: utf-8 -*-
"""
race_population.py — EXP16A Stage 0 (改訂2): 正式 race set の確定と母集団監査
============================================================================
学習しない。2023 の結果指標は計算しない (基準値は 2022 以前のみ)。ROI も候補生成もしない。

改訂2 (2026-09-24, Fable 再レビュー + ユーザー指示):
  * 障害 (トラックコード(JV) 51..59) を正式 race set から除外。production の
    P0 hard gate (race_eligibility.py) と同一定義にそろえる。
  * 馬の集合を 4 種類に **別々に** 構築する:
      starter    = 確定単勝オッズ > 1.0 を持つ馬番 (締切プールに居た馬)
      finisher   = master に完走順位を持つ馬番
      scratch    = pre スナップショットにだけ居る馬番 (締切前取消)
      dnf        = starter − finisher (出走したが完走順位が無い = 止/失格 等)
    旧実装は starter を master 由来の finisher から作っていたため、
    starter 正規化と finisher 再正規化の差が構造的に 0 になっていた (Fable 指摘)。
  * 学習で決まる subset 境界 (favorite odds / market entropy) は
    評価年 Y ごとに **Y-1 以前だけ** から作る (2022 固定値の遡及適用をしない)。
  * 検出力監査は power_audit.py へ分離。STAGE0_DRY_RUN.json は stage0_dry_run.py が書く。
出力:
  out/race_population.json
  data/_research/mcond/exp16a/official_races_le2022.npz  (power_audit 用、gitignore)
実行: python -m analysis.mcond.exp16a_close_market_residual_dev.race_population
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .provenance import (BASE, OUT, MIN_GAP_PRE, JUMP_MIN, JUMP_MAX, load_master,
                         load_tanpuk, odds_matrix, devig)

EXT_KEKKA = Path(r"E:\競馬過去走データ\raw_data\kekka_1986_2025_enhanced.csv")
RESEARCH = BASE / "data" / "_research" / "mcond" / "exp16a"
# 事前固定 (結果を見ずに決めたドメイン区分。年ごとに学習しない)
JRA_VENUES = ["札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"]
FIELD_SIZE_BINS = [[5, 8], [9, 12], [13, 15], [16, 18]]
CROSSFIT_YEARS = [2019, 2020, 2021, 2022, 2023]
REF_YEAR_MAX = 2022          # 基準値・感度分析は 2022 以前のみ (2023 は開封しない)


def load_ext_codes() -> pd.DataFrame:
    """外部 kekka の 異常コード (2016-2023)。join key = 日付+場所+R+馬番"""
    use = ["年", "月", "日", "場所", "レース番号", "馬番", "確定着順", "異常コード"]
    parts = []
    for ch in pd.read_csv(EXT_KEKKA, encoding="cp932", dtype=str, usecols=use, chunksize=300_000):
        y = pd.to_numeric(ch["年"], errors="coerce")
        parts.append(ch[(y >= 16) & (y <= 23)])
    e = pd.concat(parts, ignore_index=True)
    e["date"] = (2000 + pd.to_numeric(e["年"], errors="coerce")) * 10000 + \
                pd.to_numeric(e["月"], errors="coerce") * 100 + pd.to_numeric(e["日"], errors="coerce")
    e["R"] = pd.to_numeric(e["レース番号"], errors="coerce")
    e["ban"] = pd.to_numeric(e["馬番"], errors="coerce")
    e["code"] = pd.to_numeric(e["異常コード"], errors="coerce").fillna(0).astype(int)
    return e.dropna(subset=["date", "R", "ban"])[["date", "場所", "R", "ban", "code"]]


def ll_and_top1(sub: pd.DataFrame, key: str, mode: str):
    """race-level categorical logloss と favorite top1。mode: starters / finishers"""
    lls, hit = [], []
    for _, r in sub.iterrows():
        od = r[key]
        denom = r["starters"] if mode == "starters" else r["finishers"]
        pi, _, _ = devig(od, denom)
        w = r["winner"]
        if not pi or w not in pi:
            continue
        lls.append(-np.log(max(pi[w], 1e-12)))
        hit.append(int(max(pi, key=pi.get) == w))
    return np.array(lls), (float(np.mean(hit)) if hit else None)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    RESEARCH.mkdir(parents=True, exist_ok=True)
    m = load_master()
    m["R"] = pd.to_numeric(m["レースID(新/馬番無)"].astype(str).str[14:16], errors="coerce")
    t = load_tanpuk(set(m["rid16"]))
    ext = load_ext_codes()
    print(f"master {len(m):,} / tanpuk {len(t):,} / ext {len(ext):,}", flush=True)

    meta = (m.groupby("rid16")
              .agg(date=("date", "first"), year=("year", "first"), period=("period", "first"),
                   post_min=("post_min", "first"), venue=("場所", "first"), surface=("芝・ダ", "first"),
                   cls=("クラス名", "first"), R=("R", "first"), track_code=("track_code", "first"),
                   is_jump=("is_jump", "first"), n_fin=("ban", "size")))
    fin_set = m.groupby("rid16")["ban"].apply(set)
    winner = m[m["win"] == 1].groupby("rid16")["ban"].apply(list)
    ext_idx = ext.set_index(["date", "場所", "R", "ban"])["code"]

    rows, code_stat = [], {}
    for rid, g in t.groupby("rid16", sort=False):
        if rid not in meta.index:
            continue
        md = meta.loc[rid]
        mmdd = f"{(md.date % 10000):04d}"
        fin_rows = g[g["kubun"] == 4]
        pre_c = g[(g["kubun"] == 1) & (g["mmdd"] == mmdd)]
        if not len(fin_rows) or not len(pre_c):
            continue
        f0 = fin_rows.iloc[-1]
        gap = f0.snap_min - pre_c["snap_min"]
        ok = pre_c[gap >= MIN_GAP_PRE]
        if not len(ok):
            continue
        p0 = ok.iloc[-1]
        od_fin, od_pre = odds_matrix(f0), odds_matrix(p0)
        # ---- 4 集合を別々に構築する
        starters = sorted(od_fin)                      # 確定オッズ > 1.0 を持つ馬 = 締切プールに居た
        finishers = sorted(fin_set[rid])               # 完走順位を持つ馬
        dnf = sorted(set(starters) - set(finishers))   # 出走したが完走順位なし
        scratch = sorted(set(od_pre) - set(starters))  # pre にだけ居る = 締切前取消
        fin_not_market = sorted(set(finishers) - set(starters))
        is_jump = bool(md.is_jump)
        for b in dnf + scratch:
            k = (int(md.date), md.venue, int(md.R) if pd.notna(md.R) else -1, int(b))
            c = str(int(ext_idx.get(k, -9)))
            d = code_stat.setdefault(c, {"n": 0, "has_final_odds": 0, "in_jump_race": 0})
            d["n"] += 1
            d["has_final_odds"] += int(b in od_fin)
            d["in_jump_race"] += int(is_jump)
        rows.append({
            "rid16": rid, "date": int(md.date), "year": int(md.year), "period": md.period,
            "venue": md.venue, "surface": md.surface, "cls": md.cls,
            "track_code": (int(md.track_code) if pd.notna(md.track_code) else None),
            "is_jump": is_jump,
            "starters": starters, "finishers": finishers, "dnf": dnf, "scratch": scratch,
            "fin_not_in_market": fin_not_market,
            "n_starters": len(starters), "n_finishers": len(finishers),
            "od_fin": od_fin, "od_pre": od_pre,
            "winner": winner.get(rid, [None])[0] if rid in winner.index else None,
            "dead_heat": (len(winner.get(rid, [])) > 1) if rid in winner.index else False,
        })
    for c, d in code_stat.items():
        d["share_with_final_odds"] = d["has_final_odds"] / d["n"] if d["n"] else None
    R = pd.DataFrame(rows)
    print(f"races joined {len(R):,} (jump {int(R['is_jump'].sum()):,})", flush=True)

    # ---- 正式 race set (funnel: 障害 → 一意winner → odds欠損 → 頭数 → DNF)
    def classify(r):
        if r["is_jump"]:
            return "excl_jump"
        if r["dead_heat"] or r["winner"] is None:
            return "excl_no_unique_winner"
        if r["fin_not_in_market"] or r["winner"] not in r["od_fin"] or r["winner"] not in r["od_pre"]:
            return "excl_odds_missing"
        if r["n_starters"] < 5:
            return "excl_small_field"
        if r["dnf"]:
            return "excl_flat_dnf"
        return "official_eligible"

    R["cls_set"] = R.apply(classify, axis=1)
    FUNNEL = ["excl_jump", "excl_no_unique_winner", "excl_odds_missing", "excl_small_field",
              "excl_flat_dnf", "official_eligible"]
    per_year = {}
    for y, g in R.groupby("year"):
        flat = g[~g["is_jump"]]
        per_year[int(y)] = {
            "races_joined": int(len(g)),
            "excl_jump": int(g["is_jump"].sum()),
            "flat_races": int(len(flat)),
            "flat_races_with_dnf_any": int((flat["dnf"].map(len) > 0).sum()),
            "excl_no_unique_winner": int((g["cls_set"] == "excl_no_unique_winner").sum()),
            "excl_odds_missing": int((g["cls_set"] == "excl_odds_missing").sum()),
            "excl_small_field": int((g["cls_set"] == "excl_small_field").sum()),
            "excl_flat_dnf": int((g["cls_set"] == "excl_flat_dnf").sum()),
            "official_eligible": int((g["cls_set"] == "official_eligible").sum()),
            "meeting_days_official": int(
                g[g["cls_set"] == "official_eligible"]["rid16"].str[:10].nunique()),
            "horses_dnf_flat": int(flat["dnf"].map(len).sum()),
            "horses_dnf_jump": int(g[g["is_jump"]]["dnf"].map(len).sum()),
            "horses_scratch_pre_only": int(g["scratch"].map(len).sum()),
            "races_with_scratch": int((g["scratch"].map(len) > 0).sum()),
        }
        assert per_year[int(y)]["races_joined"] == sum(
            per_year[int(y)][k] for k in FUNNEL), f"funnel mismatch {y}"
    totals = {k: int(sum(v[k] for v in per_year.values())) for k in per_year[2016]}

    # ---- 基準値 (2022 以前のみ)。official set 上で starter 正規化
    ref = {}
    for y in range(2016, REF_YEAR_MAX + 1):
        off_y = R[(R["cls_set"] == "official_eligible") & (R["year"] == y)]
        ll_c, t1_c = ll_and_top1(off_y, "od_fin", "starters")
        ll_p, t1_p = ll_and_top1(off_y, "od_pre", "starters")
        ref[str(y)] = {
            "n_races": int(len(ll_c)),
            "n_meeting_days": int(off_y["rid16"].str[:10].nunique()),
            "terminal_close_market_logloss": float(ll_c.mean()),
            "historical_pre_snapshot_logloss": float(ll_p.mean()),
            "pre_minus_close_gap_nats": float(ll_p.mean() - ll_c.mean()),
            "favorite_top1_close": t1_c, "favorite_top1_pre": t1_p,
        }

    # ---- DNF 感度分析 (starter 全頭正規化 vs finisher のみ正規化)
    sens = {}
    for label, sub in [("official_set_le2022",
                        R[(R["cls_set"] == "official_eligible") & (R["year"] <= REF_YEAR_MAX)]),
                       ("flat_dnf_races_le2022",
                        R[(R["cls_set"] == "excl_flat_dnf") & (R["year"] <= REF_YEAR_MAX)]),
                       ("jump_races_le2022",
                        R[(R["cls_set"] == "excl_jump") & (R["year"] <= REF_YEAR_MAX)])]:
        a, _ = ll_and_top1(sub, "od_fin", "starters")
        b, _ = ll_and_top1(sub, "od_fin", "finishers")
        sens[label] = {
            "n_races_total": int(len(sub)),
            "n_races_scored": int(len(a)),
            "Q0_ll_starter_normalization": float(a.mean()) if len(a) else None,
            "Q0_ll_finisher_renormalization": float(b.mean()) if len(b) else None,
            "diff_starter_minus_finisher": float(a.mean() - b.mean()) if len(a) and len(b) else None,
            "mean_dnf_horses_per_race": float(sub["dnf"].map(len).mean()) if len(sub) else None,
        }

    # ---- 学習で決まる subset 境界: 評価年 Y ごとに Y-1 以前だけから作る
    def boundaries(sub):
        fav, ent = [], []
        for _, r in sub.iterrows():
            pi, _, _ = devig(r["od_pre"], r["starters"])
            if pi:
                fav.append(1.0 / max(pi.values()))
                ent.append(float(-sum(p * np.log(p) for p in pi.values() if p > 0)))
        if not fav:
            return {}
        fav, ent = np.array(fav), np.array(ent)
        return {"n_races_used": int(len(fav)),
                "favorite_pre_odds_tertiles": [float(np.quantile(fav, 1 / 3)),
                                               float(np.quantile(fav, 2 / 3))],
                "market_entropy_pre_tertiles": [float(np.quantile(ent, 1 / 3)),
                                                float(np.quantile(ent, 2 / 3))]}

    learned = {}
    offi = R[R["cls_set"] == "official_eligible"]
    for y in CROSSFIT_YEARS:
        past = offi[offi["year"] <= y - 1]
        learned[str(y)] = {"fit_years": [int(x) for x in sorted(past["year"].unique())],
                           **boundaries(past)}

    pop = {
        "scope": "2016-2023。結果を使う指標は 2022 以前のみ (2023 は開封しない)",
        "horse_set_definitions": {
            "starter": "確定 (区分4) の単勝オッズ > 1.0 を持つ馬番。締切プールに居た馬",
            "finisher": "master (着順が数値) に完走順位を持つ馬番",
            "scratch": "pre スナップショットに居て確定に居ない馬番 = 締切前取消",
            "dnf": "starter − finisher。出走したが完走順位が無い (止/失格 等)",
            "finisher_without_market": "finisher − starter。着順があるのに確定オッズが無い異常",
            "fix_note": "旧実装は starter を finisher から作っていたため starter/finisher 正規化の差が "
                        "構造的に 0 になっていた。本改訂で別々に構築した",
        },
        "jump_exclusion": {
            "definition": f"トラックコード(JV) {JUMP_MIN}..{JUMP_MAX} = 障害",
            "production_parity": "race_eligibility.py:48-49 の P0 hard gate と同一定義",
            "observed_track_codes_jump": sorted(
                {int(v) for v in R[R["is_jump"]]["track_code"].dropna().unique()}),
        },
        "anomaly_code_identification": {
            "method": "異常コード別に『確定オッズを持つ割合』を実測。締切プールに残っていた側を DNF 系とみなす",
            "by_code": code_stat,
            "code_-9": "外部 kekka に該当行が無い (join 失敗)",
        },
        "official_race_set_rule": [
            f"平地 (トラックコード(JV) が {JUMP_MIN}..{JUMP_MAX} でない)",
            "勝馬が一意 (同着除外)",
            "勝馬が pre・確定の両方でオッズを持ち、着順のある馬が全員 starter",
            "starter >= 5",
            "DNF が 0 頭",
        ],
        "provisional_population_note": "DNF ありレースの除外は『完走したか』という結果による条件付けである。"
                                       "DNF-inclusive な Q1 を構築できるまでの暫定母集団と明記する",
        "funnel_order": FUNNEL,
        "per_year": per_year,
        "totals": totals,
        "reference_values_by_year_le2022": ref,
        "dnf_sensitivity": sens,
        "learned_subset_boundaries_by_eval_year": learned,
        "fixed_domain_partitions": {"venues": JRA_VENUES, "field_size_bins": FIELD_SIZE_BINS,
                                    "surface": ["芝", "ダ"],
                                    "class_groups": "新馬/未勝利/1勝/2勝/3勝/OP以上",
                                    "note": "結果を見ずに決めたドメイン区分。年ごとに学習しない"},
    }
    (OUT / "race_population.json").write_text(
        json.dumps(pop, ensure_ascii=False, indent=1, default=float), encoding="utf-8")

    # ---- power_audit 用に 2022 以前の official race を保存 (gitignore 配下)
    keep = R[(R["cls_set"] == "official_eligible") & (R["year"] <= REF_YEAR_MAX)].reset_index(drop=True)
    rid, yr, day, wpos, ban, pic, pip, off = [], [], [], [], [], [], [], [0]
    for _, r in keep.iterrows():
        pic_d, _, _ = devig(r["od_fin"], r["starters"])
        pip_d, _, _ = devig(r["od_pre"], r["starters"])
        bs = [b for b in r["starters"] if b in pic_d and b in pip_d]
        if r["winner"] not in bs or len(bs) < 5:
            continue
        sc = np.array([pic_d[b] for b in bs]); sc = sc / sc.sum()
        sp = np.array([pip_d[b] for b in bs]); sp = sp / sp.sum()
        rid.append(r["rid16"]); yr.append(r["year"]); day.append(r["rid16"][:10])
        wpos.append(bs.index(r["winner"]))
        ban.extend(bs); pic.extend(sc.tolist()); pip.extend(sp.tolist())
        off.append(len(ban))
    np.savez_compressed(RESEARCH / "official_races_le2022.npz",
                        rid16=np.array(rid), year=np.array(yr, dtype=np.int32),
                        day=np.array(day), winner_pos=np.array(wpos, dtype=np.int32),
                        ban=np.array(ban, dtype=np.int32), pi_close=np.array(pic),
                        pi_pre=np.array(pip), offsets=np.array(off, dtype=np.int64))
    print(f"[saved] official_races_le2022.npz races={len(rid):,} horses={len(ban):,}")

    print(json.dumps(totals, ensure_ascii=False))
    print(json.dumps(ref.get("2022"), ensure_ascii=False))
    print(json.dumps(sens, ensure_ascii=False))
    print("code_stat:", json.dumps(code_stat, ensure_ascii=False))


if __name__ == "__main__":
    main()
