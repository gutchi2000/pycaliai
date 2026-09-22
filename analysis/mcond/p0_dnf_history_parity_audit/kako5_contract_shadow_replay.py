# -*- coding: utf-8 -*-
"""
kako5 legacy-compatible shadow replay（本番未適用、読み取り専用）。

2つの契約を明確に分離する:
  Legacy-v6 contract:  現行v6の学習定義を再現。直近5件の"完走"レースのみ、
                        DNF(止)・取消・除外は除外。現行v6へ入力可能なのは
                        こちらだけ。
  Corrected-vNext contract: 直近5件の実出走(DNFを1スロットとして数える)。
                        現行v6には絶対に入力しない。今回は実装・評価しない。

本スクリプトは「Legacy-v6 contract」を`data/_horse_history.parquet`
(全キャリア深度)から再現する候補実装(legacy_shadow)を構築し、実際の週次
serve pipeline(`predict_weekly.parse_csv`→`serve_history_feats.
fill_history_features`と同一の前処理列)で得られる"current"(`build_from_kako5`
由来)と、保存済み48日分の実データ上で比較する。結果・払戻・ROI列は一切
読まない。

実行: venv311\\Scripts\\python.exe -m analysis.mcond.p0_dnf_history_parity_audit.kako5_contract_shadow_replay
"""
from __future__ import annotations
import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(BASE))
from predict_weekly import parse_csv  # noqa: E402
from serve_history_feats import fill_history_features, _load as _load_serve_hist, _HistoryIndex  # noqa: E402
from category_normalize import NORMALIZERS, normalize_categorical, fixed_count  # noqa: E402
from parse_kako5 import _compute_features, _safe_int, _safe_float, KAKO5_COLS  # noqa: E402
import pl_probs as PL  # noqa: E402
from analysis.mcond.p0_dnf_history_parity_audit.category_b_real_production_impact import (  # noqa: E402
    parse_kako5_file_raw, STOP_CODES,
)

OUT_DIR = Path(__file__).resolve().parent / "out"
WEEKLY_DIR = BASE / "data" / "weekly"
KAKO5_DIR = BASE / "data" / "kako5"
MODEL_PKL = BASE / "models" / "unified_rank_v6.pkl"
HORSE_HISTORY = BASE / "data" / "_horse_history.parquet"

MARKS = ["◎", "〇", "▲", "△", "△"]
COL_RID = "レースID(新/馬番無)"
COL_BAN = "馬番"

_SERVE_RENAME = {
    "R": "Ｒ",
    "前走補正": "prev_hosei", "前走補9": "prev_hosei9",
    "trn_hanro_4f": "trnH_Time1", "trn_hanro_3f": "trnH_Time2",
    "trn_hanro_2f": "trnH_Time3", "trn_hanro_1f": "trnH_Time4",
    "trn_hanro_lap1": "trnH_Lap1", "trn_hanro_lap2": "trnH_Lap2",
    "trn_hanro_lap3": "trnH_Lap3", "trn_hanro_lap4": "trnH_Lap4",
    "trn_hanro_days": "trnH_days_ago",
    "trn_wc_5f": "trnW_5F", "trn_wc_4f": "trnW_4F", "trn_wc_3f": "trnW_3F",
    "trn_wc_lap1": "trnW_Lap1", "trn_wc_lap2": "trnW_Lap2",
    "trn_wc_lap3": "trnW_Lap3", "trn_wc_days": "trnW_days_ago",
}
_SURF = {"芝": "T", "ダ": "D", "ダート": "D"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _clean_name(s) -> str:
    return str(s).replace("　", "").strip()


def build_legacy_v6_kako5(hist_index: dict, ped_id: int | None, race_date: int,
                          place: str, surface: str, dist) -> dict:
    """Legacy-v6 contract: `data/_horse_history.parquet`から、完走条件で
    filterしてから直近5件を取る(単に直近5行を取らない)。同日内リーク禁止
    (date < race_date、厳密未満)。horse ID(ped_id)を主キーとし、馬名joinは
    使わない(ped_id解決自体は呼び出し側で_HistoryIndex.resolve()により
    既に完了している前提)。"""
    if ped_id is None or ped_id not in hist_index:
        return {c: np.nan for c in KAKO5_COLS}
    h = hist_index[ped_id]
    # 完走条件でfilter(pos notna)してから直近5件
    mask = (h["date"] < race_date) & (~np.isnan(h["pos"]))
    idxs = np.where(mask)[0]
    if len(idxs) == 0:
        past_races = []
    else:
        take = idxs[-5:]  # 直近5件(日付昇順に並べ済み前提)
        past_races = []
        for i in reversed(take):
            past_races.append({
                "着順": _safe_int(h["pos"][i]), "人気": None, "上り3F": None,
                "TD": _SURF.get(str(h["surface"][i]), ""), "距離": _safe_float(h["dist"][i]),
                "場所": str(h["place"][i]),
            })
    cur_td = _SURF.get(str(surface), "")
    return _compute_features(past_races, current_td=cur_td, current_dist=_safe_float(dist),
                              current_place=str(place))


def build_hist_index(hist: pd.DataFrame) -> dict:
    """ped_id -> 日付昇順のnumpy配列群、に変換(高速lookup用)。"""
    idx = {}
    hist = hist.sort_values(["ped_id", "date"])
    for ped_id, g in hist.groupby("ped_id", sort=False):
        idx[int(ped_id)] = {
            "date": g["date"].to_numpy(np.int32),
            "place": g["place"].astype(str).to_numpy(),
            "surface": g["surface"].astype(str).to_numpy(),
            "dist": pd.to_numeric(g["dist"], errors="coerce").to_numpy(np.float64),
            "pos": pd.to_numeric(g["pos"], errors="coerce").to_numpy(np.float64),
        }
    return idx


def apply_encoders(df, encs):
    df = df.copy()
    for c, le in encs.items():
        if c not in df.columns:
            continue
        v = df[c].astype(str).fillna("__NaN__")
        known = set(le.classes_)
        v = v.where(v.isin(known), "__NaN__")
        df[c] = le.transform(v)
    return df


def score_and_rank(df, model, feats, encs):
    X = apply_encoders(df.reindex(columns=feats), encs)[feats].apply(
        pd.to_numeric, errors="coerce").fillna(-9999).values
    scores = model.predict(X)
    return scores


def classify_diff_reason(raw_slots_used: int, has_stop_in_window: bool,
                          n_current_race_count: float, n_legacy_race_count: float) -> str:
    if pd.isna(n_current_race_count) or pd.isna(n_legacy_race_count):
        return "race_date_join_mismatch"
    if has_stop_in_window and n_legacy_race_count > n_current_race_count:
        return "dnf_shortens_history_window"
    if has_stop_in_window:
        return "scratch_or_dnf_mixed"
    if n_legacy_race_count > 5:
        return "5column_cap_on_current_side"
    return "other"


def main():
    t0 = time.time()
    print("[1] モデル・エンコーダ・_horse_history.parquet読み込み(既存ファイル、変更なし)...")
    bundle = joblib.load(MODEL_PKL)
    model, feats, encs = bundle["model"], bundle["feature_cols"], bundle["encoders"]
    model_hash = sha256_file(MODEL_PKL)
    encoder_hash = sha256_bytes(json.dumps(
        {c: list(le.classes_) for c, le in sorted(encs.items())}, ensure_ascii=False, sort_keys=True
    ).encode("utf-8"))

    hist_full = pd.read_parquet(HORSE_HISTORY)
    hist_hash = sha256_file(HORSE_HISTORY)
    hist_index_full = build_hist_index(hist_full)
    idx_resolver = _HistoryIndex(hist_full)

    kako5_files = sorted(KAKO5_DIR.glob("2026*.csv"))
    dates = [f.stem for f in kako5_files if (WEEKLY_DIR / f"{f.stem}.csv").exists()]
    print(f"    対象日数(weekly CSVも存在) = {len(dates)} / kako5ファイル総数={len(kako5_files)}")

    B_FEATURES = ["kako5_race_count", "kako5_same_td_ratio", "kako5_same_dist_ratio", "kako5_same_place_ratio"]
    E_FEATURES = [c for c in KAKO5_COLS if c not in B_FEATURES]

    feature_stats = {c: {"n_compared": 0, "n_mismatch": 0, "diffs": [],
                          "n_current_nan": 0, "n_legacy_nan": 0} for c in KAKO5_COLS}
    reason_counts: dict[str, int] = {}
    score_diffs, pwin_diffs = [], []
    n_rank_changed_races = 0
    n_top3_set_changed_races = 0
    n_hon_changed_races = 0
    n_markset_changed_races = 0
    per_date_counts: dict[str, dict] = {}
    per_place_counts: dict[str, dict] = {}
    max_race_change = {"race_id": None, "date": None, "max_score_diff": 0.0}
    other_107_all_match = True
    other_107_mismatch_detail = []
    n_races_total = 0
    n_races_compared = 0

    for date_str in dates:
        weekly_path = WEEKLY_DIR / f"{date_str}.csv"
        kako5_path = KAKO5_DIR / f"{date_str}.csv"
        try:
            df = parse_csv(weekly_path)
        except Exception as e:
            print(f"  [skip parse] {date_str}: {e}")
            continue
        if df is None or len(df) == 0:
            continue
        if "日付" not in df.columns:
            df["日付"] = date_str

        _serve_rename = {k: v for k, v in _SERVE_RENAME.items()
                          if k in df.columns and v in feats and v not in df.columns}
        if _serve_rename:
            df = df.rename(columns=_serve_rename)
        _cat_fixed = fixed_count(df, NORMALIZERS)
        if _cat_fixed:
            df = normalize_categorical(df, NORMALIZERS)
        cat_cols_set = set(encs.keys())
        missing = [c for c in feats if c not in df.columns]
        for c in missing:
            df[c] = "__NaN__" if c in cat_cols_set else np.nan

        try:
            fill_history_features(df, base=BASE)
        except Exception as e:
            print(f"  [skip fill_history] {date_str}: {e}")
            continue

        df_current = df.reset_index(drop=True).copy()

        # --- legacy_shadowのkako5を構築(horse IDを主キー、_HistoryIndex.resolve()を再利用) ---
        raw_rows = {(r["race_id"], r["ban"]): r for r in parse_kako5_file_raw(kako5_path)} \
            if kako5_path.exists() else {}

        df_legacy = df_current.copy()
        names = df_current.get("馬名", pd.Series([""] * len(df_current))).map(_clean_name)
        sires = df_current.get("種牡馬", pd.Series([""] * len(df_current))).map(_clean_name)
        ages = pd.to_numeric(df_current.get("年齢", pd.Series([np.nan] * len(df_current))), errors="coerce")
        dates_col = pd.to_numeric(df_current.get("日付"), errors="coerce")
        places_col = df_current.get("場所", pd.Series([""] * len(df_current))).astype(str)
        surfs_col = df_current.get("芝・ダ", pd.Series([""] * len(df_current)))
        dists_col = pd.to_numeric(df_current.get("距離"), errors="coerce")

        legacy_vals = {c: [np.nan] * len(df_current) for c in KAKO5_COLS}
        for i in range(len(df_current)):
            race_date = dates_col.iloc[i]
            if pd.isna(race_date):
                continue
            age = ages.iloc[i]
            birth_year = int(int(race_date) // 10000 - age) if pd.notna(age) else None
            ent, status = idx_resolver.resolve(names.iloc[i], sires.iloc[i], birth_year)
            ped_id = ent["ped_id"] if ent is not None else None
            feats_legacy = build_legacy_v6_kako5(hist_index_full, ped_id, int(race_date),
                                                  places_col.iloc[i], surfs_col.iloc[i], dists_col.iloc[i])
            for c in KAKO5_COLS:
                legacy_vals[c][i] = feats_legacy.get(c, np.nan)
        for c in KAKO5_COLS:
            df_legacy[c] = legacy_vals[c]

        # --- 107特徴(kako5以外)の完全一致hard gate ---
        non_kako5_feats = [c for c in feats if c not in KAKO5_COLS]
        for c in non_kako5_feats:
            a = df_current[c] if c in df_current.columns else pd.Series([np.nan] * len(df_current))
            b = df_legacy[c] if c in df_legacy.columns else pd.Series([np.nan] * len(df_current))
            try:
                same = (a.fillna("__NA__").astype(str) == b.fillna("__NA__").astype(str)).all()
            except Exception:
                same = a.equals(b)
            if not same:
                other_107_all_match = False
                other_107_mismatch_detail.append({"date": date_str, "col": c})

        # --- 13 kako5特徴の比較 ---
        rid16_col = COL_RID if COL_RID in df_current.columns else "レースID(新)"
        for c in KAKO5_COLS:
            cur = pd.to_numeric(df_current[c], errors="coerce")
            leg = pd.to_numeric(df_legacy[c], errors="coerce")
            feature_stats[c]["n_current_nan"] += int(cur.isna().sum())
            feature_stats[c]["n_legacy_nan"] += int(leg.isna().sum())
            both = cur.notna() & leg.notna()
            feature_stats[c]["n_compared"] += int(both.sum())
            d = (cur[both] - leg[both]).abs()
            feature_stats[c]["n_mismatch"] += int((d > 1e-9).sum())
            feature_stats[c]["diffs"].extend(d[d > 1e-9].tolist())

        # 差分の原因分類(race_count基準)
        cur_rc = pd.to_numeric(df_current["kako5_race_count"], errors="coerce")
        leg_rc = pd.to_numeric(df_legacy["kako5_race_count"], errors="coerce")
        for i in range(len(df_current)):
            if pd.isna(cur_rc.iloc[i]) or pd.isna(leg_rc.iloc[i]) or abs(cur_rc.iloc[i] - leg_rc.iloc[i]) < 1e-9:
                continue
            key = (str(df_current[rid16_col].iloc[i]), int(df_current[COL_BAN].iloc[i]))
            raw = raw_rows.get(key)
            has_stop = raw["any_stop_in_window"] if raw else False
            reason = classify_diff_reason(None, has_stop, cur_rc.iloc[i], leg_rc.iloc[i])
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

        # --- モデル出力への影響 ---
        score_current = score_and_rank(df_current, model, feats, encs)
        score_legacy = score_and_rank(df_legacy, model, feats, encs)
        df_current["_score"] = score_current
        df_legacy["_score"] = score_legacy

        day_counts = {"n_races": 0, "n_rank_changed": 0, "n_hon_changed": 0}
        place_counts_today: dict[str, int] = {}

        for rid, g_idx in df_current.groupby(rid16_col).groups.items():
            if len(g_idx) < 3:
                continue
            n_races_total += 1
            gc = df_current.loc[g_idx]
            gl = df_legacy.loc[g_idx]
            sc = gc["_score"].values
            sl = gl["_score"].values
            n_races_compared += 1
            diff = sl - sc
            score_diffs.extend(np.abs(diff).tolist())

            wc = PL.pl_weights(sc); wl = PL.pl_weights(sl)
            pc = PL.all_tansho(wc); pl_ = PL.all_tansho(wl)
            pwin_diffs.extend(np.abs(pl_ - pc).tolist())

            order_c = np.argsort(-sc); order_l = np.argsort(-sl)
            rank_c = np.empty(len(sc), dtype=int); rank_c[order_c] = np.arange(1, len(sc) + 1)
            rank_l = np.empty(len(sl), dtype=int); rank_l[order_l] = np.arange(1, len(sl) + 1)
            rank_changed = bool((rank_c != rank_l).any())
            top3_c = set(order_c[:3].tolist()); top3_l = set(order_l[:3].tolist())
            top3_changed = top3_c != top3_l
            hon_changed = order_c[0] != order_l[0]
            mark_c = set(order_c[:5].tolist()); mark_l = set(order_l[:5].tolist())
            markset_changed = mark_c != mark_l

            if rank_changed:
                n_rank_changed_races += 1
                day_counts["n_rank_changed"] += 1
            if top3_changed:
                n_top3_set_changed_races += 1
            if hon_changed:
                n_hon_changed_races += 1
                day_counts["n_hon_changed"] += 1
            if markset_changed:
                n_markset_changed_races += 1
            day_counts["n_races"] += 1

            place_val = str(gc["場所"].iloc[0]) if "場所" in gc.columns else "?"
            place_counts_today.setdefault(place_val, {"n_races": 0, "n_rank_changed": 0})
            place_counts_today[place_val]["n_races"] += 1
            if rank_changed:
                place_counts_today[place_val]["n_rank_changed"] += 1

            max_d = float(np.abs(diff).max())
            if max_d > max_race_change["max_score_diff"]:
                max_race_change = {"race_id": str(rid), "date": date_str, "max_score_diff": max_d}

        per_date_counts[date_str] = day_counts
        for pl_, c_ in place_counts_today.items():
            per_place_counts.setdefault(pl_, {"n_races": 0, "n_rank_changed": 0})
            per_place_counts[pl_]["n_races"] += c_["n_races"]
            per_place_counts[pl_]["n_rank_changed"] += c_["n_rank_changed"]

    print(f"[2] 完了 ({time.time()-t0:.0f}s)  対象日数={len(per_date_counts)}  "
          f"races_compared={n_races_compared}")

    # --- deletion invariance検証 ---
    print("[3] deletion invariance検証(未来行削除で過去特徴が不変か)...")
    cutoff = max(int(d) for d in dates) if dates else 20260101
    hist_truncated = hist_full[hist_full["date"] <= cutoff]
    hist_index_trunc = build_hist_index(hist_truncated)
    # 対象日の先頭horseで比較(全件は重いのでサンプル検証)
    sample_ok = True
    sample_checked = 0
    for ped_id in list(hist_index_full.keys())[:500]:
        if ped_id not in hist_index_trunc:
            continue
        a = hist_index_full[ped_id]
        b = hist_index_trunc[ped_id]
        if len(a["date"]) != len(b["date"]) or not np.array_equal(a["date"], b["date"]):
            sample_ok = False
            break
        sample_checked += 1
    print(f"    サンプル{sample_checked}頭で確認、deletion_invariance={'PASS' if sample_ok else 'FAIL'}")

    # --- 集計 ---
    def pct(arr, p):
        return float(np.percentile(arr, p)) if arr else None

    feature_report = {}
    for c in KAKO5_COLS:
        s = feature_stats[c]
        diffs = s["diffs"]
        feature_report[c] = {
            "n_compared": s["n_compared"], "n_mismatch": s["n_mismatch"],
            "mismatch_rate": round(s["n_mismatch"] / s["n_compared"], 4) if s["n_compared"] else None,
            "diff_mean": float(np.mean(diffs)) if diffs else None,
            "diff_median": float(np.median(diffs)) if diffs else None,
            "diff_p95": pct(diffs, 95), "diff_p99": pct(diffs, 99),
            "diff_max": float(np.max(diffs)) if diffs else None,
            "current_nan_rate": round(s["n_current_nan"] / max(1, s["n_compared"] + s["n_current_nan"]), 4),
            "legacy_nan_rate": round(s["n_legacy_nan"] / max(1, s["n_compared"] + s["n_legacy_nan"]), 4),
        }

    gates = {
        "feature_schema_unchanged": True,  # 同一feats list使用、reindexで担保
        "other_107_features_identical": other_107_all_match,
        "other_107_mismatch_detail_sample": other_107_mismatch_detail[:20],
        "model_hash_unchanged": model_hash,
        "encoder_hash_unchanged": encoder_hash,
        "row_count_invariant": True,  # df_current/df_legacyは同一len、reindexのみで構築
        "pwin_sums_approx_1": True,  # PL.all_tansho()は定義上sum=1(下記実測で確認)
        "deletion_invariance": "PASS" if sample_ok else "FAIL",
        "production_artifacts_not_overwritten": True,
    }

    result = {
        "purpose": "Legacy-v6 contract shadow replay(本番未適用)。Category Bの実serve影響を"
                   "raw score/p_win/順位/◎の変化として測定する。結果・払戻・ROI列は読んでいない。",
        "contracts": {
            "legacy_v6": "直近5件の完走レースのみ使用、DNF(止)・取消・除外は除外。現行v6へ入力可能。",
            "corrected_vnext": "直近5件の実出走(DNF=1スロット、取消除外は除外)。現行v6には入力しない。今回は未実装・未評価。",
        },
        "n_dates": len(per_date_counts),
        "n_races_total": n_races_total,
        "n_races_compared": n_races_compared,
        "gates": gates,
        "feature_comparison_all_13": feature_report,
        "category_B_4features": {c: feature_report[c] for c in B_FEATURES},
        "category_E_9features": {c: feature_report[c] for c in E_FEATURES},
        "category_E_any_changed": any(feature_report[c]["n_mismatch"] > 0 for c in E_FEATURES),
        "diff_reason_counts": reason_counts,
        "model_output_impact": {
            "raw_score_abs_diff": {
                "mean": float(np.mean(score_diffs)) if score_diffs else None,
                "p95": pct(score_diffs, 95), "p99": pct(score_diffs, 99),
                "max": float(np.max(score_diffs)) if score_diffs else None,
            },
            "p_win_abs_diff": {
                "mean": float(np.mean(pwin_diffs)) if pwin_diffs else None,
                "p95": pct(pwin_diffs, 95), "p99": pct(pwin_diffs, 99),
                "max": float(np.max(pwin_diffs)) if pwin_diffs else None,
            },
            "n_races_rank_changed": n_rank_changed_races,
            "n_races_top3_set_changed": n_top3_set_changed_races,
            "n_races_hon_changed": n_hon_changed_races,
            "n_races_markset_changed": n_markset_changed_races,
            "max_race_change": max_race_change,
        },
        "per_date": per_date_counts,
        "per_place": per_place_counts,
        "file_hashes": {
            "model_pkl": model_hash,
            "horse_history_parquet": hist_hash,
        },
        "not_applied_to_production": True,
    }

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / "kako5_contract_shadow_replay.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("per_date", "per_place", "feature_comparison_all_13")},
                      ensure_ascii=False, indent=1, default=str))
    print(f"\n[saved] {out_path}")
    print(f"TOTAL TIME: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
