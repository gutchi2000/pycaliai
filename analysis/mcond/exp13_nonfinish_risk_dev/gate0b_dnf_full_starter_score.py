# -*- coding: utf-8 -*-
"""
Gate 0B（EXP13再開） — 2023年flat DNF馬(145頭)のfull-starter特徴再構築+raw v6 score
+ 全starter確率再計算（読み取り専用、モデル学習は一切行わない）
==========================================================================
2026-09-22。ユーザー要求の3分割比較のうち②③を実施する:
  ① 同じ完走馬集合で計算した確率のparity → gate0b_raw_score_parity.py で対応済み
     (course/jockey 6特徴のみ差し替えた場合のdelta)
  ② 全starterを含めて再計算した新しい確率 → 本スクリプト
  ③ 非完走馬追加による既存馬の確率変化 → 本スクリプト

120特徴のうち、DNF馬は一度もmaster_v2に行が存在しないため全て新規構築が必要。
`GATE0B_FEATURE_AUDIT.md` の分類に従い、以下の経路で構築する:
  - raw_passthrough 68特徴: pre-dropna全体(631,965行)に既に存在する値をそのまま使う
  - jockey/trainer_fuku(4) + horse_fuku(2) + prev_pos_rel/closing_power(2):
    build_dataset.py の実関数をpre-dropna全体に対して再実行(import、複製ではない)
  - prev_hosei/prev_hosei9(2) + trnH_*/trnW_*(16): build_master_v2.py の
    merge_hosei()/merge_training()を実行(asof結合、母集団非依存なのでDNF行だけでも可)
  - kako5_*/hist_same_*(20): parse_kako5.py の _compute_features() を対象馬の
    自分の全履歴のみで実行(母集団全体のループ不要、高速)
  - course_n_prev系/jockey_n_prev系(6): build_master_v2.py の
    compute_history_features()相当をpre-dropna全体に対して実行

確率の全starter正規化は「有効なhistorical_pre_snapshotを持つstarter」ではなく
「started(止を含む、外消除外)」を母集団とする(ユーザー指示: 市場ではなくv6確率の
全starter再計算が目的のため市場snapshotの有無は無関係)。

実行: venv311\\Scripts\\python.exe -m analysis.mcond.exp13_nonfinish_risk_dev.gate0b_dnf_full_starter_score
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(BASE))
from analysis.market_provenance_audit import load_kekka_labels  # noqa: E402
from build_dataset import (  # noqa: E402
    add_rolling_stats, add_horse_rolling_stats, add_pace_features,
    load_csv, normalize_date, convert_finish,
    CSV_LGBM, CSV_CAT, CSV_TORCH, CSV_ADD, JOIN_KEY, COL_FINISH, COL_DATE as BD_COL_DATE, TARGET,
)
from build_master_v2 import merge_hosei, merge_training  # noqa: E402


def build_full_pre_dropna_universe_all_cols() -> pd.DataFrame:
    """build_dataset.py:build_master() の JOIN 部分(全120特徴の生成元となる
    lgbm+cat+torch+addの全列)を、dropna適用前の状態でそのまま再現する。
    analysis/p0_5_verification/reconstruct_true_pipeline_universe.py の
    build_full_pre_dropna_universe() は騎手/調教師コード+着順のみの狭い部分集合
    だったため(jockey/trainer_fuku検証専用)、本関数はDNF馬のraw scoreに必要な
    全120特徴の原材料(種牡馬・性別・年齢・斤量等cat/torch由来の列を含む)を
    揃えるために新規に書く。build_dataset.pyの実関数(load_csv/normalize_date/
    convert_finish)をそのままimportして使い、JOIN順序もbuild_master()と同一。
    """
    df_lgbm = load_csv(CSV_LGBM, "lgbm")
    df_cat = load_csv(CSV_CAT, "cat")
    df_torch = load_csv(CSV_TORCH, "torch")
    df_add = load_csv(CSV_ADD, "add")
    df_add[COL_FINISH] = convert_finish(df_add[COL_FINISH])

    master = df_lgbm.merge(df_cat, on=JOIN_KEY, how="left", suffixes=("", "_cat"))
    master = master.merge(df_torch, on=JOIN_KEY, how="left", suffixes=("", "_torch"))
    master = master.merge(df_add, on=JOIN_KEY, how="left", suffixes=("", "_add"))
    dup_cols = [c for c in master.columns if c.endswith("_cat") or c.endswith("_torch") or c.endswith("_add")]
    master = master.drop(columns=dup_cols)
    master[BD_COL_DATE] = normalize_date(master[BD_COL_DATE])
    master["date_dt"] = pd.to_datetime(master[BD_COL_DATE].astype(str), format="%Y%m%d")
    master[TARGET] = (master[COL_FINISH] <= 3).astype("Int8")
    assert len(master) == 631_965, f"行数異常: {len(master):,}"
    return master
from parse_kako5 import _compute_features, _safe_float, _safe_int, KAKO5_COLS, HIST_COLS  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "out"
MASTER_V2 = BASE / "data" / "master_v2_20130105-20251228.csv"
MODEL_PKL = BASE / "models" / "unified_rank_v6.pkl"

COL_RID16, COL_BAN = "レースID(新/馬番無)", "馬番"


def compute_history_features_expanding(df: pd.DataFrame) -> pd.DataFrame:
    """build_master_v2.py:compute_history_features() と同一ロジック(再掲、
    gate0b_course_jockey_history_parity.py と同一実装)。"""
    COL_DATE, COL_PEDIGREE, COL_PLACE, COL_SURFACE, COL_DIST = "日付", "血統登録番号", "場所", "芝・ダ", "距離"
    COL_JOCKEY, COL_JYUN = "騎手コード", "着順"
    df = df.copy()
    df[COL_DATE] = pd.to_numeric(df[COL_DATE], errors="coerce").astype("Int64")
    df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df["_is_win"] = (df[COL_JYUN] == 1).astype("Int8")
    df["_is_top3"] = (df[COL_JYUN] <= 3).astype("Int8")

    def dist_band(d):
        if pd.isna(d):
            return "?"
        d = int(d)
        if d <= 1400:
            return "短"
        if d <= 1700:
            return "マ"
        if d <= 2200:
            return "中"
        return "長"

    df["_dist_b"] = df[COL_DIST].apply(dist_band)
    df["_course_key"] = (df[COL_PLACE].astype(str) + "|" + df[COL_SURFACE].astype(str) + "|" + df["_dist_b"])
    df = df.sort_values([COL_PEDIGREE, COL_DATE]).reset_index(drop=True)
    g = df.groupby([COL_PEDIGREE, "_course_key"])
    df["course_n_prev"] = g.cumcount()
    df["course_wins_prev"] = g["_is_win"].cumsum().astype("Int64") - df["_is_win"].astype("Int64")
    df["course_top3_prev"] = g["_is_top3"].cumsum().astype("Int64") - df["_is_top3"].astype("Int64")
    df["course_win_rate"] = np.where(df["course_n_prev"] > 0, df["course_wins_prev"] / df["course_n_prev"], np.nan)
    df["course_top3_rate"] = np.where(df["course_n_prev"] > 0, df["course_top3_prev"] / df["course_n_prev"], np.nan)
    gj = df.groupby([COL_PEDIGREE, COL_JOCKEY])
    df["jockey_n_prev"] = gj.cumcount()
    df["jockey_wins_prev"] = gj["_is_win"].cumsum().astype("Int64") - df["_is_win"].astype("Int64")
    df["jockey_top3_prev"] = gj["_is_top3"].cumsum().astype("Int64") - df["_is_top3"].astype("Int64")
    df["jockey_win_rate"] = np.where(df["jockey_n_prev"] > 0, df["jockey_wins_prev"] / df["jockey_n_prev"], np.nan)
    df["jockey_top3_rate"] = np.where(df["jockey_n_prev"] > 0, df["jockey_top3_prev"] / df["jockey_n_prev"], np.nan)
    return df


def kako5_for_target_rows(full: pd.DataFrame, target_keys: pd.DataFrame) -> pd.DataFrame:
    """target_keys (rid16, ban) の各行について、その馬自身の全履歴(full内、
    その馬のgroup)だけを使ってkako5_*/hist_same_*を計算する。母集団全体を
    ループしないため高速。"""
    td_map = {"芝": "T", "ダ": "D", "ダート": "D", "T": "T", "D": "D"}
    full = full.copy()
    full["date_dt"] = pd.to_datetime(full["日付"].astype(str), format="%Y%m%d", errors="coerce")
    full["_td_code"] = full["芝・ダ"].map(td_map).fillna("")
    full["_dist"] = pd.to_numeric(full["距離"], errors="coerce")
    full["_place"] = full["場所"].astype(str)

    target = full.merge(target_keys, on=[COL_RID16, COL_BAN], how="inner")
    rows_out = []
    for _, trow in target.iterrows():
        hid = trow["血統登録番号"]
        group = full[full["血統登録番号"] == hid].sort_values("date_dt")
        idxs = group.index.tolist()
        this_idx = trow.name
        # this_idx は元のfullのindexと一致する前提(mergeでindexは振り直されるため、
        # rid16+banで自分自身の行を特定し直す)
        self_mask = (group[COL_RID16] == trow[COL_RID16]) & (group[COL_BAN] == trow[COL_BAN])
        if not self_mask.any():
            continue
        seq_i = list(group.index).index(group.index[self_mask][0])
        idx = group.index[self_mask][0]

        past_indices = idxs[max(0, seq_i - 5): seq_i]
        past_races = []
        for pi in reversed(past_indices):
            pr = group.loc[pi]
            past_races.append({
                "着順": _safe_int(pr.get("着順")),
                "人気": None,
                "上り3F": _safe_float(pr.get("前走上り3F")),
                "TD": td_map.get(str(pr.get("芝・ダ", "")), ""),
                "距離": _safe_float(pr.get("距離")),
                "場所": str(pr.get("場所", "")),
            })
        row = group.loc[idx]
        feats = _compute_features(
            past_races, current_td=row.get("_td_code"),
            current_dist=_safe_float(row.get("_dist")), current_place=row.get("_place"))
        cur_td, cur_dist, cur_place = row.get("_td_code", ""), _safe_float(row.get("_dist")), row.get("_place", "")
        for hcol in HIST_COLS:
            feats[hcol] = np.nan
        all_td, all_dist, all_place = group["_td_code"].values, group["_dist"].values, group["_place"].values
        着順_vals = pd.to_numeric(group["着順"], errors="coerce").values
        if seq_i > 0 and cur_td and cur_dist is not None:
            same_cond_pos = []
            for pi_seq in range(seq_i):
                pos = 着順_vals[pi_seq]
                if np.isnan(pos):
                    continue
                if all_td[pi_seq] == cur_td and not np.isnan(all_dist[pi_seq]) and abs(all_dist[pi_seq] - cur_dist) <= 200:
                    same_cond_pos.append(int(pos))
            if same_cond_pos:
                feats["hist_same_cond_best_pos"] = min(same_cond_pos)
                feats["hist_same_cond_top3_rate"] = sum(1 for p in same_cond_pos if p <= 3) / len(same_cond_pos)
                feats["hist_same_cond_count"] = len(same_cond_pos)
        if seq_i > 0 and cur_place:
            same_place_pos = []
            for pi_seq in range(seq_i):
                pos = 着順_vals[pi_seq]
                if np.isnan(pos):
                    continue
                if all_place[pi_seq] == cur_place:
                    same_place_pos.append(int(pos))
            if same_place_pos:
                feats["hist_same_place_best_pos"] = min(same_place_pos)
        feats[COL_RID16] = trow[COL_RID16]
        feats[COL_BAN] = trow[COL_BAN]
        rows_out.append(feats)
    return pd.DataFrame(rows_out)


def apply_encoders(df: pd.DataFrame, encs: dict) -> pd.DataFrame:
    df = df.copy()
    for c, le in encs.items():
        if c not in df.columns:
            continue
        v = df[c].astype(str).fillna("__NaN__")
        known = set(le.classes_)
        v = v.where(v.isin(known), "__NaN__")
        df[c] = le.transform(v)
    return df


def main() -> int:
    t0 = time.time()
    print("[1] pre-dropna full-starter母集団(631,965行、全120特徴の原材料込み)を再構築...")
    full = build_full_pre_dropna_universe_all_cols()

    print("[2] 2023年flat DNF(止)の(rid16,ban)を特定...")
    kk = load_kekka_labels(2023, 2023)
    dnf = kk[kk["is_dnf"]][["rid16", "ban_i"]].rename(columns={"rid16": COL_RID16, "ban_i": COL_BAN})
    dnf[COL_RID16] = dnf[COL_RID16].astype(np.int64)
    dnf[COL_BAN] = dnf[COL_BAN].astype(np.int64)
    print(f"    2023 flat DNF = {len(dnf)}")

    print("[3] build_dataset.py実関数(pre-dropna全体)でjockey/trainer/horse_fuku, "
          "prev_pos_rel/closing_powerを計算...")
    full2 = add_rolling_stats(full.copy())
    full2 = add_horse_rolling_stats(full2)
    full2 = add_pace_features(full2)
    print(f"    完了 ({time.time()-t0:.0f}s)")

    print("[4] build_master_v2.py実関数でcourse/jockey_n_prev系(expanding cumcount)を計算...")
    full3 = compute_history_features_expanding(full2)
    print(f"    完了 ({time.time()-t0:.0f}s)")

    dnf_rows = full3.merge(dnf, on=[COL_RID16, COL_BAN], how="inner")
    print(f"    fullに実在するDNF行 = {len(dnf_rows)} / {len(dnf)}")

    print("[5] merge_hosei/merge_training (asof結合、DNF行のみで可) ...")
    dnf_rows2 = merge_hosei(dnf_rows)
    dnf_rows3 = merge_training(dnf_rows2)
    print(f"    完了 ({time.time()-t0:.0f}s)")

    print("[6] kako5_*/hist_same_* (対象馬ごとの個別計算) ...")
    kako5_feats = kako5_for_target_rows(full, dnf[[COL_RID16, COL_BAN]])
    print(f"    完了 ({time.time()-t0:.0f}s)")

    dnf_full = dnf_rows3.merge(kako5_feats, on=[COL_RID16, COL_BAN], how="left")
    print(f"    最終DNF特徴行 = {len(dnf_full)}")

    print("[7] モデルロード & scoring ...")
    bundle = joblib.load(MODEL_PKL)
    model, feats, encs = bundle["model"], bundle["feature_cols"], bundle["encoders"]
    missing = [f for f in feats if f not in dnf_full.columns]
    print(f"    120特徴中、構築できなかった列 = {missing}")

    X_df = dnf_full.reindex(columns=feats)
    X_enc = apply_encoders(X_df, encs)
    X = X_enc[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    dnf_full["_score"] = model.predict(X)

    print("[8] 全starter PL再計算 (DNFを含む race のみ) ...")
    v2 = pd.read_csv(MASTER_V2, encoding="utf-8-sig", low_memory=False)
    v2_feats = v2.reindex(columns=feats)
    v2_enc = apply_encoders(v2_feats.copy(), encs)
    Xv2 = v2_enc[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    v2["_score"] = model.predict(Xv2)

    race_deltas = []
    affected_races = dnf_full[COL_RID16].unique()
    for rid16 in affected_races:
        finishers = v2[v2[COL_RID16] == rid16][[COL_BAN, "_score"]].copy()
        if len(finishers) == 0:
            continue
        dnf_r = dnf_full[dnf_full[COL_RID16] == rid16][[COL_BAN, "_score"]].copy()

        w_fin_only = np.exp(finishers["_score"].values - finishers["_score"].values.max())
        p_fin_only = w_fin_only / w_fin_only.sum()

        all_scores = np.concatenate([finishers["_score"].values, dnf_r["_score"].values])
        w_full = np.exp(all_scores - all_scores.max())
        p_full = w_full / w_full.sum()
        p_full_fin = p_full[:len(finishers)]

        delta = p_full_fin - p_fin_only
        race_deltas.append({
            "rid16": rid16, "n_finishers": int(len(finishers)), "n_dnf_added": int(len(dnf_r)),
            "dnf_score": dnf_r["_score"].tolist(),
            "mean_abs_delta_finisher_prob": float(np.abs(delta).mean()),
            "max_abs_delta_finisher_prob": float(np.abs(delta).max()),
            "sum_prob_full_starter": float(p_full.sum()),
        })

    result = {
        "n_dnf_2023_flat": int(len(dnf)),
        "n_dnf_feature_rows_built": int(len(dnf_full)),
        "n_missing_feature_cols": missing,
        "n_races_with_dnf_affected": len(race_deltas),
        "race_level_detail": race_deltas,
        "summary": {
            "mean_of_mean_abs_delta_finisher_prob": float(np.mean([r["mean_abs_delta_finisher_prob"] for r in race_deltas])) if race_deltas else None,
            "max_of_max_abs_delta_finisher_prob": float(np.max([r["max_abs_delta_finisher_prob"] for r in race_deltas])) if race_deltas else None,
        },
        "dnf_raw_score_stats": {
            "mean": float(dnf_full["_score"].mean()) if len(dnf_full) else None,
            "min": float(dnf_full["_score"].min()) if len(dnf_full) else None,
            "max": float(dnf_full["_score"].max()) if len(dnf_full) else None,
        },
    }
    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / "gate0b_dnf_full_starter_score.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "race_level_detail"},
                      ensure_ascii=False, indent=1))
    print(f"\n[saved] {out_path}  (total {time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
