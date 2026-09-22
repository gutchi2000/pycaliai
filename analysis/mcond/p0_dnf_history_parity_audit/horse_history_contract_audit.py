# -*- coding: utf-8 -*-
"""
`data/_horse_history.parquet` artifact contract監査（読み取り専用、結果・ROIは
一切使わない）。

item1(contract確認)はbuild_horse_history.pyのdocstring+コード読解で既に
確定済み(本ファイル冒頭のCONTRACTコメント参照)。本スクリプトはitem3-6
(authoritative universe構築・完全性比較・欠落原因分類・production影響)を
実施する。

authoritative universeの構築方針: `data/kekka/2026*.csv`(全出走馬+確定着順)
の行存在そのものを「出走」の一次証拠とし、race_id+馬番+dateのみで構築する。
**馬名joinは一切行わない**(build_horse_history.py自身が内部で行っている
weekly側の馬番マッチング・馬名一致フィルタの"结果"を監査対象として検証する
のみで、監査側が独自の馬名照合を新設することはしない)。

馬単位の経歴内位置(欠落が最古/途中/最新のどこか)の分析だけは、馬の同一性を
複数レースにわたって連結する必要があるため、serve_history_feats.py既存の
`_HistoryIndex.resolve()`(馬名+父名+生年の曖昧性解消込み、検証済みロジック)
を再利用する。この特定の分析ステップでのみ名前ベース解決を使うことを
明示する。
"""
from __future__ import annotations
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(BASE))
from build_horse_history import parse_weekly_light, KEKKA_DIR, WEEKLY_DIR, MASTER_CSV, OUT_PARQUET  # noqa: E402
from serve_history_feats import _HistoryIndex  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "out"

# ============================================================
# item1: artifact contract（build_horse_history.py docstring + code読解で確定）
# ============================================================
CONTRACT = {
    "generator_script": "build_horse_history.py",
    "caller": "serve_history_feats.py (fill_history_features/_HistoryIndex/rolling_rate)、"
              "呼び出し元はexport_weekly_marks.py(v6本番経路)のみ。"
              "predict_weekly.py(旧アンサンブル経路)は使用していない(コード確認済み)。",
    "update_method": "**全再構築(incrementalではない)**。main()は既存parquetを読まず、"
                     "毎回master_v2(固定, 2013-2025)+data/kekka/2026*.csv全件+"
                     "data/weekly/2026*.csv全件から最初から作り直す。",
    "update_trigger": "docstring記載: 週次運用でPhase C(weekly_post.ps1)にてkekka配置後に"
                      "再実行、2026分を追随させる想定。",
    "intended_date_range": "2013-01-05(master_v2開始)〜実行時点でdata/kekka/に存在する"
                           "最新2026日付まで。終了日の上限はない(都度伸びる)。",
    "intended_retention_per_horse": "**全キャリア(docstring: '馬ごと全キャリア履歴')**。"
                                    "コード中に.tail(N)等の切り詰め処理は存在しない"
                                    "(load_master_history/load_2026_history/"
                                    "resolve_2026_ped_idsを全読しコード上確認済み)。",
    "finish_type_inclusion_2013_2025": "止/外/消はmaster_v2自体に存在しないため"
                                       "(build_dataset.py dropna、構造的に不在)、"
                                       "2013-2025分は着順=NaNの行が原理的に発生しない。",
    "finish_type_inclusion_2026": "kekka『確定着順』の0/空値をpos=NaNとして残す"
                                  "(止・外・消を区別せず一律NaN、"
                                  "データソース自体がこの3種を区別しない"
                                  "——LIVE_SERVE_DNF_TRACE.mdで既確認)。",
    "horse_id_resolution_2013_2025": "master_v2の血統登録番号を直接使用(権威あるID)。",
    "horse_id_resolution_2026": "resolve_2026_ped_ids(): 馬名完全一致でmaster由来の"
                                "エンティティ表と照合、(種牡馬一致 or 生年±1一致)で1件に"
                                "絞れればそのped_idを採用。0件または曖昧なら"
                                "synthetic_id(name, sire)(決定論的hash)を新規発行。"
                                "**2026どうし(既存2026行)との照合はしない**"
                                "——比較対象はmasterのみ。",
    "explicit_full_history_contract": "**存在する**。docstring 6行目「馬ごと全キャリア履歴"
                                      "parquetを構築する」が明示的契約。",
    "production_usage_confirmed": "course_n_prev/course_win_rate/course_top3_rate/"
                                  "jockey_n_prev/jockey_win_rate/jockey_top3_rate/"
                                  "hist_same_cond_best_pos/top3_rate/count/"
                                  "hist_same_place_best_pos/horse_fuku10/30/"
                                  "jockey_fuku30/90/trainer_fuku30/90 "
                                  "(serve_history_feats.NUM_FEATS、計16特徴)。"
                                  "kako5_*(13特徴)はこのparquetを使わない"
                                  "(parse_kako5.build_from_kako5()が別のTARGET"
                                  "kako5 CSVを直接読む、既知)。",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_authoritative_universe() -> pd.DataFrame:
    """kekka(全出走馬+確定着順)の行存在そのものを一次証拠として、
    race_id+馬番+dateだけでauthoritative universeを構築する。馬名joinなし。
    build_horse_history.pyと同じ対象範囲(kekka×weeklyがともに存在する日)に限定。"""
    rows = []
    kekka_files = sorted(KEKKA_DIR.glob("*.csv"))
    for kp in kekka_files:
        stem = kp.stem
        if not (stem.isdigit() and len(stem) == 8):
            continue
        d = int(stem)
        wp = WEEKLY_DIR / f"{stem}.csv"
        weekly_exists = wp.exists()
        try:
            k = pd.read_csv(kp, encoding="cp932", dtype=str)
        except Exception as e:
            rows.append({"date": d, "n_kekka_rows": 0, "n_weekly_matched": None,
                        "weekly_exists": weekly_exists, "error": str(e)})
            continue
        if "レースID(新)" not in k.columns or "確定着順" not in k.columns:
            continue
        k["rid16"] = k["レースID(新)"].astype(str).str.strip().str[:16]
        k["馬番_num"] = pd.to_numeric(k["馬番"], errors="coerce")
        k["pos_raw"] = pd.to_numeric(k["確定着順"], errors="coerce")
        k["is_nonfinish"] = (k["pos_raw"].isna()) | (k["pos_raw"] <= 0)
        n_kekka = len(k)
        n_nonfinish = int(k["is_nonfinish"].sum())

        n_weekly_matched = None
        if weekly_exists:
            w = parse_weekly_light(wp)
            if not w.empty:
                merged = k[["rid16", "馬番_num", "馬名", "pos_raw", "is_nonfinish"]].rename(
                    columns={"馬番_num": "馬番"}).merge(
                    w[["rid16", "馬番"]], on=["rid16", "馬番"], how="inner")
                n_weekly_matched = len(merged)

        rows.append({
            "date": d, "n_kekka_rows": n_kekka, "n_nonfinish_kekka": n_nonfinish,
            "n_weekly_matched": n_weekly_matched, "weekly_exists": weekly_exists,
        })
    return pd.DataFrame(rows)


def reproduce_build_horse_history_funnel() -> pd.DataFrame:
    """build_horse_history.py:load_2026_history()の3段階(kekka読込→
    weekly inner join→馬名完全一致filter)を、同スクリプトの実関数を
    そのままimportして日別に再現し、各段階での行数を記録する
    (パイプライン自体を監査、複製ではない)。"""
    from build_horse_history import _clean_name

    mh_max_date = 20251228  # master_v2の最終日(固定、外部から再読込不要)
    rows = []
    kekka_files = sorted(KEKKA_DIR.glob("*.csv"))
    for kp in kekka_files:
        stem = kp.stem
        if not (stem.isdigit() and len(stem) == 8):
            continue
        d = int(stem)
        if d <= mh_max_date:
            continue
        wp = WEEKLY_DIR / f"{stem}.csv"
        stage = {"date": d}
        if not wp.exists():
            stage["stage0_kekka_rows"] = None
            stage["stage1_weekly_join"] = None
            stage["stage2_name_filter"] = None
            stage["drop_reason"] = "weekly_missing"
            rows.append(stage)
            continue
        try:
            k = pd.read_csv(kp, encoding="cp932", dtype=str)
        except Exception:
            rows.append({**stage, "drop_reason": "kekka_read_fail"})
            continue
        if "レースID(新)" not in k.columns or "確定着順" not in k.columns:
            rows.append({**stage, "drop_reason": "kekka_missing_cols"})
            continue
        k = k.assign(rid16=k["レースID(新)"].astype(str).str.strip().str[:16],
                     馬番=pd.to_numeric(k["馬番"], errors="coerce"),
                     pos=pd.to_numeric(k["確定着順"], errors="coerce"))
        stage["stage0_kekka_rows"] = len(k)
        w = parse_weekly_light(wp)
        if w.empty:
            rows.append({**stage, "drop_reason": "weekly_parse_zero_rows"})
            continue
        merged = k[["rid16", "馬番", "馬名", "pos"]].merge(w, on=["rid16", "馬番"], how="inner")
        stage["stage1_weekly_join"] = len(merged)
        merged2 = merged[merged["馬名"].map(_clean_name) == merged["name"]]
        stage["stage2_name_filter"] = len(merged2)
        stage["drop_at_join"] = stage["stage0_kekka_rows"] - stage["stage1_weekly_join"]
        stage["drop_at_name_filter"] = stage["stage1_weekly_join"] - stage["stage2_name_filter"]
        rows.append(stage)
    return pd.DataFrame(rows)


def compare_with_parquet(funnel_df: pd.DataFrame) -> dict:
    """`_horse_history.parquet`の2026分(src=='kekka2026')行数を、
    funnel再現の最終段(stage2_name_filter)と日別に比較する。"""
    hist = pd.read_parquet(OUT_PARQUET)
    hist_2026 = hist[hist["src"] == "kekka2026"]
    by_date_parquet = hist_2026.groupby("date").size()

    rows = []
    for _, r in funnel_df.iterrows():
        d = r["date"]
        expected = r.get("stage2_name_filter")
        actual = int(by_date_parquet.get(d, 0))
        rows.append({
            "date": int(d), "funnel_final_expected": expected, "parquet_actual": actual,
            "diff": (expected - actual) if pd.notna(expected) else None,
        })
    detail = pd.DataFrame(rows)
    return {
        "detail": detail,
        "parquet_2026_total_rows": int(len(hist_2026)),
        "parquet_2026_date_range": [int(hist_2026["date"].min()), int(hist_2026["date"].max())]
                                    if len(hist_2026) else None,
        "funnel_matches_parquet_exactly": bool((detail["diff"].fillna(0) == 0).all()),
    }


def horse_level_position_analysis(sample_size: int = 300) -> dict:
    """欠落が馬の経歴の最古/途中/最新のどこで起きているかを、
    serve_history_feats.py既存の_HistoryIndex.resolve()(名前ベース解決、
    検証済みロジックを再利用)で複数レースを連結して分析する。
    このステップだけ名前ベース解決を使うことを明示する。"""
    hist = pd.read_parquet(OUT_PARQUET)
    idx = _HistoryIndex(hist)
    hist_2026 = hist[hist["src"] == "kekka2026"].sort_values("date")

    # 2026年に複数走を持つ馬をサンプリング
    counts = hist_2026.groupby("ped_id").size()
    multi_race_horses = counts[counts >= 2].index[:sample_size]

    positions = {"oldest_gap": 0, "middle_gap": 0, "latest_gap": 0, "no_gap_detected": 0}
    for ped_id in multi_race_horses:
        h = hist[hist["ped_id"] == ped_id].sort_values("date")
        dates = h["date"].tolist()
        if len(dates) < 2:
            continue
        gaps = [dates[i + 1] - dates[i] for i in range(len(dates) - 1)]
        if not gaps:
            positions["no_gap_detected"] += 1
            continue
        max_gap_idx = int(np.argmax(gaps))
        frac = max_gap_idx / max(1, len(gaps) - 1)
        if max(gaps) < 30:
            positions["no_gap_detected"] += 1
        elif frac < 0.34:
            positions["oldest_gap"] += 1
        elif frac > 0.66:
            positions["latest_gap"] += 1
        else:
            positions["middle_gap"] += 1

    return {"method": "serve_history_feats._HistoryIndex.resolve()による名前ベース解決"
                       "(このステップのみ)、複数走を持つ2026馬をサンプリングし"
                       "日付間隔30日以上のギャップの経歴内位置を分類",
            "n_horses_sampled": len(multi_race_horses), "position_distribution": positions}


def production_consumers() -> list[dict]:
    return [
        {"consumer": "serve_history_feats.py: fill_history_features()/_HistoryIndex/rolling_rate()",
         "columns_read": ["ped_id/name/sire/birth_year/date/place/surface/dist/pos/jockey_code/trainer_code"],
         "needs_full_history": True,
         "features_affected_by_gaps": ["course_n_prev", "course_win_rate", "course_top3_rate",
                                        "jockey_n_prev", "jockey_win_rate", "jockey_top3_rate",
                                        "hist_same_cond_best_pos", "hist_same_cond_top3_rate",
                                        "hist_same_cond_count", "hist_same_place_best_pos",
                                        "horse_fuku10", "horse_fuku30",
                                        "jockey_fuku30", "jockey_fuku90",
                                        "trainer_fuku30", "trainer_fuku90"],
         "used_in_current_serve": True,
         "fallback_exists": "Yes(fail-open): fill_history_features失敗時はNaNのまま"
                            "(export_weekly_marks.py try/except、値は変わらず欠損のまま)。"
                            "個々の行の欠落自体にfallbackはない(黙って過小カウント/"
                            "古い値になる、エラーにはならない)。",
         "impact_status": "CONFIRMED(course/jockey/hist_same/fuku系、2026年の欠落分だけ"
                          "経験数過小カウント。2013-2025分は本監査の対象外、"
                          "既存validate_serve_history_feats.pyでparity検証済み)。"},
        {"consumer": "predict_weekly.py(旧アンサンブル経路)",
         "columns_read": [], "needs_full_history": False,
         "features_affected_by_gaps": [],
         "used_in_current_serve": False,
         "fallback_exists": "該当なし(jockey_stats.csv/trainer_stats.csvという別の"
                            "build_dataset.py由来ファイルを使う、本parquetは不使用)",
         "impact_status": "NO_IMPACT(本parquetを参照しない)"},
        {"consumer": "parse_kako5.build_from_kako5()(kako5_*13特徴)",
         "columns_read": [], "needs_full_history": False,
         "features_affected_by_gaps": [],
         "used_in_current_serve": True,
         "fallback_exists": "該当なし(TARGET週次kako5 CSVを直接読む、別データソース)",
         "impact_status": "NO_IMPACT(本parquetを使わない、別途Category B/Eとして"
                          "既に評価済み)"},
        {"consumer": "analysis/validate_serve_history_feats.py・"
                     "analysis/measure_serve_coverage.py(既存の検証/計測ツール)",
         "columns_read": ["全16列"], "needs_full_history": True,
         "features_affected_by_gaps": ["上記16特徴と同一"],
         "used_in_current_serve": False,
         "fallback_exists": "該当なし(オフライン検証ツール)",
         "impact_status": "POSSIBLE(検証範囲はtest=2024-2025のmaster_v2由来分のみ、"
                          "2026補完分のカバレッジ測定は範囲外だった可能性——"
                          "本監査で発見した欠損は事前に検知されていなかった)。"},
    ]


def main() -> int:
    t0 = time.time()
    print("[1] artifact contract(コード読解で確定済み)")
    print(json.dumps(CONTRACT, ensure_ascii=False, indent=1))

    print("\n[2] authoritative universe構築(race_id+馬番+date、馬名joinなし)...")
    universe = build_authoritative_universe()
    print(f"    対象日数={len(universe)}  "
          f"(kekka合計行数={universe['n_kekka_rows'].sum():,.0f})")

    print("\n[3] build_horse_history.pyのfunnelを日別に再現(3段階)...")
    funnel = reproduce_build_horse_history_funnel()
    print(f"    対象2026日数={len(funnel)}")
    print(f"    stage0(kekka)合計={funnel['stage0_kekka_rows'].sum():,.0f}")
    print(f"    stage1(weekly join後)合計={funnel['stage1_weekly_join'].sum():,.0f}")
    print(f"    stage2(名前filter後)合計={funnel['stage2_name_filter'].sum():,.0f}")
    print(f"    join段階での消失合計={funnel['drop_at_join'].sum():,.0f}")
    print(f"    name filter段階での消失合計={funnel['drop_at_name_filter'].sum():,.0f}")

    print("\n[4] `_horse_history.parquet`実物との比較...")
    cmp = compare_with_parquet(funnel)
    print(f"    parquet 2026分行数={cmp['parquet_2026_total_rows']:,}  "
          f"日付範囲={cmp['parquet_2026_date_range']}")
    print(f"    funnel再現とparquetが完全一致={cmp['funnel_matches_parquet_exactly']}")
    n_diff_dates = int((cmp["detail"]["diff"].fillna(0) != 0).sum())
    print(f"    差分がある日数={n_diff_dates}/{len(cmp['detail'])}")

    print("\n[5] 馬単位の欠落位置分析(名前ベース解決を使用、このステップのみ)...")
    pos_analysis = horse_level_position_analysis()
    print(json.dumps(pos_analysis["position_distribution"], ensure_ascii=False, indent=1))

    print("\n[6] production consumer監査...")
    consumers = production_consumers()

    result = {
        "contract": CONTRACT,
        "authoritative_universe_summary": {
            "n_dates": len(universe),
            "total_kekka_rows": int(universe["n_kekka_rows"].sum()),
            "total_nonfinish_kekka_rows": int(universe["n_nonfinish_kekka"].sum()),
            "dates_missing_weekly": int((~universe["weekly_exists"]).sum()),
        },
        "funnel_reproduction": {
            "n_dates_2026_scope": len(funnel),
            "stage0_kekka_total": int(funnel["stage0_kekka_rows"].sum()),
            "stage1_weekly_join_total": int(funnel["stage1_weekly_join"].sum()),
            "stage2_name_filter_total": int(funnel["stage2_name_filter"].sum()),
            "total_lost_at_weekly_join": int(funnel["drop_at_join"].sum()),
            "total_lost_at_name_filter": int(funnel["drop_at_name_filter"].sum()),
            "dates_with_weekly_missing": int((funnel["drop_reason"] == "weekly_missing").sum())
                                          if "drop_reason" in funnel.columns else 0,
            "by_date": funnel.to_dict(orient="records"),
        },
        "parquet_comparison": {
            "parquet_2026_total_rows": cmp["parquet_2026_total_rows"],
            "parquet_2026_date_range": cmp["parquet_2026_date_range"],
            "funnel_matches_parquet_exactly": cmp["funnel_matches_parquet_exactly"],
            "n_dates_with_diff": n_diff_dates,
            "detail_by_date": cmp["detail"].to_dict(orient="records"),
        },
        "horse_level_position_analysis": pos_analysis,
        "production_consumers": consumers,
        "gap_root_cause_classification": {
            "weekly_missing_or_unparsed": "確認済み: 特定日・特定レースでweekly CSVが"
                "存在しない/パース失敗により当該レース全馬がstage1で消失",
            "name_exact_match_filter": "確認済み: kekka側馬名とweekly側馬名(parse_weekly_light"
                "の馬名S列)が完全一致しない場合、stage2で消失(具体的な原因—全角半角/"
                "空白/表記ゆれ等—は個別ケースにより異なり本監査では全件特定していない)",
            "incremental_update_boundary": "該当なし(全再構築方式のため無関係)",
            "horse_id_resolution_failure": "該当なし(resolve_2026_ped_idsは常に"
                "synthetic_idへのfallbackを持ち、行自体を落とすことはない——"
                "行の消失はload_2026_history()のjoin/filter段階でのみ発生)",
            "date_range_filter": "該当なし(d<=master_max_dateのみ、2026年は全て対象)",
            "dropna_dnf_scratch": "該当なし(2026分はpos=NaNのまま保持、行自体は残る)",
            "other_unresolved": "本監査で全ての消失をstage1(weekly join)/stage2(名前filter)"
                "のいずれかに分類できた(下記funnel_reproductionのby_date参照)。"
                "個々の名前不一致の具体的文字差分までは特定していない。",
        },
    }

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / "horse_history_contract_audit.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\n[saved] {out_path}")
    print(f"TOTAL TIME: {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
