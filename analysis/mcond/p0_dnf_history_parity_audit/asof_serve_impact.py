# -*- coding: utf-8 -*-
"""
asof_serve_impact.py
====================
欠落した障害競走が **2026 年の各 serve horse-date で実際に特徴を変えるか** を
as-of (判断時点より前の履歴のみ) で再計算する。

「影響馬候補」(その馬が欠落障害に出走していた) と
「実際に特徴が変わる serve 行」を**厳密に分けて**報告する。

判定可能性の区分:
  [決定可能] horse_fuku10/30, hist_same_place_best_pos,
             hist_same_cond_*, course_*
             → 欠落行から date/place/pos が取れるため実測できる
  [決定不能] jockey_n_prev/win_rate/top3_rate, jockey_fuku30/90,
             trainer_fuku30/90
             → 欠落行に騎手・調教師コードが無く (44/46 日)、
               上限のみ提示する

READ-ONLY。production 変更・再学習・ROI 評価は行わない。
出力: out/asof_serve_impact.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
OUT = Path(__file__).resolve().parent / "out"

# 欠落障害レースの surface 契約 2 branch
#   A: surface='障' (障害を独立サーフェスとして扱う)
#   B: surface=芝/ダ へ recode (master_v2 2013-2025 の実際の挙動)
SURFACE_BRANCHES = ["A_shogai_own_surface", "B_recoded_to_flat"]


def log(m):
    print(m, flush=True)


def main():
    from serve_history_feats import _HistoryIndex, compute_row_feats, _clean_name
    from build_horse_history import parse_weekly_light

    log("=" * 74)
    log("as-of serve 影響率 再計算")
    log("=" * 74)

    # ---- 1. 実 production 履歴 ----
    hist = pd.read_parquet(BASE / "data" / "_horse_history.parquet")
    idx = _HistoryIndex(hist)
    log(f"\n_horse_history.parquet: {len(hist):,} 行 / "
        f"{hist['ped_id'].nunique():,} 頭")

    # ---- 2. 欠落障害行を production と同じ resolver で馬に紐付ける ----
    miss = pd.read_csv(OUT / "missing_set_rows.csv", encoding="utf-8-sig",
                       dtype=str)
    miss["date"] = miss["race_id"].str[:8].astype(int)
    miss["pos_v"] = pd.to_numeric(miss["pos"], errors="coerce")
    miss["nm"] = miss["horse_name"].map(_clean_name)

    # 欠落レースの距離 (照合できた 17 レースのみ既知、他は NaN)
    cls = json.load(open(OUT / "missing_race_class_verification.json",
                         encoding="utf-8"))
    dist_by_rid = {}
    for r in cls["races"]:
        d = str(r.get("distance") or "").strip()
        if d.isdigit():
            dist_by_rid[r["rid16"]] = float(d)
    miss["dist_v"] = miss["race_id"].map(dist_by_rid)

    resolved, status_ct = [], Counter()
    for _, r in miss.iterrows():
        # 欠落行は sire/birth_year を持たないため名前のみで解決する
        # (production の resolve_2026_ped_ids と同じ名前ベース、限界を明記)
        ent, st = idx.resolve(r["nm"], "", None)
        status_ct[st] += 1
        if ent is not None:
            resolved.append({"ped_id": ent["ped_id"], "nm": r["nm"],
                             "date": int(r["date"]), "place": r["venue"],
                             "pos": r["pos_v"], "dist": r["dist_v"],
                             "race_id": r["race_id"]})
    res = pd.DataFrame(resolved)
    log(f"\n欠落 {len(miss)} 行の馬解決: {dict(status_ct)}")
    log(f"  → ped_id 解決できた欠落行: {len(res)} "
        f"({len(res)/len(miss)*100:.1f}%) / "
        f"unique 馬 {res['ped_id'].nunique() if len(res) else 0}")

    jump_by_ped = defaultdict(list)
    for _, r in res.iterrows():
        jump_by_ped[int(r["ped_id"])].append(r)

    # ---- 3. serve 母集団 = 2026 weekly 全行 ----
    serve_rows = []
    for p in sorted((BASE / "data" / "weekly").glob("2026*.csv")):
        try:
            w = parse_weekly_light(p)
        except Exception:
            continue
        if w.empty:
            continue
        w["date"] = int(p.stem)
        serve_rows.append(w)
    serve = pd.concat(serve_rows, ignore_index=True)
    log(f"\nserve 母集団 (2026 weekly): {len(serve):,} 行 / "
        f"{serve['rid16'].nunique():,} レース / {serve['name'].nunique():,} 頭")

    # ---- 4. as-of 判定 ----
    results = {b: {"rows_changed": 0, "races_changed": set(),
                   "horses_changed": set(),
                   "feat_changed": Counter(),
                   "by_date": Counter(), "by_venue": Counter(),
                   "slot_pos": Counter()} for b in SURFACE_BRANCHES}

    n_candidate_rows = 0          # 影響馬候補 (欠落障害を過去に持つ serve 行)
    candidate_races = set()
    candidate_horses = set()
    jump_then_flat_horses = set()  # 障害→平地に戻ったケース
    flat_only_jump_after = set()   # 平地後に障害へ移っただけ (as-of で影響なし)

    FEATS_DETERMINABLE = [
        "horse_fuku10", "horse_fuku30", "hist_same_place_best_pos",
        "hist_same_cond_best_pos", "hist_same_cond_top3_rate",
        "hist_same_cond_count", "course_n_prev", "course_win_rate",
        "course_top3_rate"]

    for row in serve.itertuples(index=False):
        rdate = int(row.date)
        ent, st = idx.resolve(row.name, getattr(row, "sire", "") or "", None)
        if ent is None:
            continue
        ped = ent["ped_id"]
        jumps = jump_by_ped.get(ped)
        if not jumps:
            continue
        # as-of: 判断時点より前の欠落障害のみ
        prior = [j for j in jumps if int(j["date"]) < rdate]
        if not prior:
            # この serve 行より後の障害 = live flat serve には影響しない
            flat_only_jump_after.add(ped)
            continue
        n_candidate_rows += 1
        candidate_races.add(row.rid16)
        candidate_horses.add(ped)
        jump_then_flat_horses.add(ped)

        # 欠落障害が「直近何走前」に当たるか
        past_dates = ent["date"][ent["date"] < rdate]
        for j in prior:
            slot = int((past_dates > int(j["date"])).sum()) + 1
            results[SURFACE_BRANCHES[0]]["slot_pos"][
                f"{slot}走前" if slot <= 5 else "6走前以前"] += 1

        base = compute_row_feats(ent, rdate, row.place, row.surface,
                                 row.dist, np.nan)

        for branch in SURFACE_BRANCHES:
            # shadow entity = 実履歴 + 欠落障害行
            add_n = len(prior)
            d = np.concatenate([ent["date"], [int(j["date"]) for j in prior]])
            pl = np.concatenate([ent["place"], [str(j["place"]) for j in prior]])
            if branch == "A_shogai_own_surface":
                sf = np.concatenate([ent["surface"], ["障"] * add_n])
            else:
                # master_v2 2013-2025 と同じく芝/ダへ recode
                # (実 surface は不明のため、当該 serve 行の surface に
                #  一致させる最悪ケースを採る = 影響の上限)
                sf = np.concatenate([ent["surface"],
                                     [str(row.surface)] * add_n])
            ds = np.concatenate([ent["dist"],
                                 [float(j["dist"]) if pd.notna(j["dist"])
                                  else np.nan for j in prior]])
            ps = np.concatenate([ent["pos"],
                                 [float(j["pos"]) if pd.notna(j["pos"])
                                  else np.nan for j in prior]])
            jk = np.concatenate([ent["jockey"], [np.nan] * add_n])
            order = np.argsort(d, kind="stable")
            shadow = {"date": d[order], "place": pl[order],
                      "surface": sf[order], "dist": ds[order],
                      "pos": ps[order], "jockey": jk[order],
                      "ped_id": ped, "sire": ent["sire"],
                      "birth_year": ent["birth_year"]}
            new = compute_row_feats(shadow, rdate, row.place, row.surface,
                                    row.dist, np.nan)

            changed = []
            for f in FEATS_DETERMINABLE:
                a, b = base.get(f), new.get(f)
                if (pd.isna(a) and pd.isna(b)):
                    continue
                if pd.isna(a) != pd.isna(b) or (
                        not pd.isna(a) and abs(float(a) - float(b)) > 1e-12):
                    changed.append(f)
            if changed:
                R = results[branch]
                R["rows_changed"] += 1
                R["races_changed"].add(row.rid16)
                R["horses_changed"].add(ped)
                R["by_date"][rdate] += 1
                R["by_venue"][str(row.place)] += 1
                for f in changed:
                    R["feat_changed"][f] += 1

    # ---- 5. 出力 ----
    n_serve_rows, n_serve_races = len(serve), serve["rid16"].nunique()
    log(f"\n--- 影響馬候補 (欠落障害を as-of で保有する serve 行) ---")
    log(f"  serve 行: {n_candidate_rows:,} / {n_serve_rows:,} "
        f"({n_candidate_rows/n_serve_rows*100:.3f}%)")
    log(f"  serve レース: {len(candidate_races):,} / {n_serve_races:,} "
        f"({len(candidate_races)/n_serve_races*100:.2f}%)")
    log(f"  馬: {len(candidate_horses):,}")
    log(f"  障害→平地に戻った馬: {len(jump_then_flat_horses):,}")
    log(f"  平地後に障害のみ (as-of で live flat serve に影響なし): "
        f"{len(flat_only_jump_after - jump_then_flat_horses):,}")

    payload = {
        "generated_at": datetime.now().isoformat(),
        "serve_population": {"rows": int(n_serve_rows),
                             "races": int(n_serve_races),
                             "horses": int(serve["name"].nunique())},
        "missing_rows_total": int(len(miss)),
        "missing_rows_ped_resolved": int(len(res)),
        "ped_resolve_status": dict(status_ct),
        "candidate": {
            "serve_rows": n_candidate_rows,
            "serve_rows_share": round(n_candidate_rows / n_serve_rows, 6),
            "serve_races": len(candidate_races),
            "serve_races_share": round(len(candidate_races) / n_serve_races, 6),
            "horses": len(candidate_horses),
            "jump_then_flat_horses": len(jump_then_flat_horses),
            "flat_then_jump_only_horses":
                len(flat_only_jump_after - jump_then_flat_horses),
        },
        "branches": {},
        "undeterminable_features": {
            "features": ["jockey_n_prev", "jockey_win_rate", "jockey_top3_rate",
                         "jockey_fuku30", "jockey_fuku90",
                         "trainer_fuku30", "trainer_fuku90"],
            "reason": "欠落行に騎手コード・調教師コードが存在せず "
                      "(44/46 日で復元不能、ALT_SOURCE_RECOVERY_AUDIT §5)、"
                      "変化の有無を判定できない",
            "upper_bound_note": "jockey_fuku30/90・trainer_fuku30/90 は "
                                "馬ではなく騎手・調教師の直近 N 走 rolling の"
                                "ため、欠落騎乗 1 件がその騎手・調教師の"
                                "後続 serve 行すべてに波及しうる。"
                                "上限は serve 全行、下限は 0 で、"
                                "現データでは特定不能。",
        },
    }
    for b in SURFACE_BRANCHES:
        R = results[b]
        payload["branches"][b] = {
            "rows_changed": R["rows_changed"],
            "rows_changed_share_of_serve":
                round(R["rows_changed"] / n_serve_rows, 6),
            "races_changed": len(R["races_changed"]),
            "races_changed_share_of_serve":
                round(len(R["races_changed"]) / n_serve_races, 6),
            "horses_changed": len(R["horses_changed"]),
            "feature_change_counts": dict(R["feat_changed"]),
            "by_date": {str(k): v for k, v in sorted(R["by_date"].items())},
            "by_venue": dict(R["by_venue"].most_common()),
            "history_slot_position": dict(R["slot_pos"]),
        }
        log(f"\n--- branch {b} ---")
        log(f"  実際に特徴が変わった serve 行: {R['rows_changed']:,} "
            f"({R['rows_changed']/n_serve_rows*100:.3f}% of serve rows)")
        log(f"  影響レース: {len(R['races_changed']):,} "
            f"({len(R['races_changed'])/n_serve_races*100:.2f}%)")
        log(f"  影響馬: {len(R['horses_changed']):,}")
        log(f"  変化した特徴: {dict(R['feat_changed'])}")
        if R["by_venue"]:
            log(f"  競馬場別: {dict(R['by_venue'].most_common(8))}")

    log(f"\n履歴枠の位置: "
        f"{dict(results[SURFACE_BRANCHES[0]]['slot_pos'])}")

    with open(OUT / "asof_serve_impact.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    log("\n保存: out/asof_serve_impact.json")


if __name__ == "__main__":
    main()
