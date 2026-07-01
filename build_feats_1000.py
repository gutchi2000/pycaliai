# -*- coding: utf-8 -*-
"""
build_feats_1000.py — 独自特徴量 ~1000個を機械生成（全て leak-safe / as-of 安全）
================================================================================
目的: unified_rank_v_1000（v6 とは別系統の新規モデル）用に、master_v2 から
      ~1000 の派生特徴を生成して data/feats_1000.parquet に保存する。

★leak-safe 原則（このプロジェクトの鉄則）:
  - 当日結果（着順 / fukusho_flag / roi_target）は一切使わない
  - 当日オッズ・当日馬体重は master_v2 に無い（学習特徴から除外済み）
  - レース内相対変換は「現レースの事前既知列」のみで計算 → 構造的に leak-safe
  - 馬キャリア集計は 血統登録番号 で日付ソートし shift(1)＝厳密に過去のみ参照
    （prev_date <= cur_date を全行アサート、違反 0 を要求）

生成ファミリ:
  F1 レース内相対変換  : z / rankpct / 平均差 / max差 / min差 / レンジ正規化  (×TRANSFORM_COLS)
  F2 ペア積            : コア列の総当たり積
  F3 ペア比            : 厳選列の比
  F4 過去走派生        : kako5 / hist の比・差・ボラティリティ
  F5 条件変化          : クラス昇降・距離/馬場/斤量/頭数変化（exp_feateng_v7 と同系）
  F6 馬キャリア as-of   : 生涯戦数・平均/最良着順・直近3走・休養・勝率（厳密過去）

出力: data/feats_1000.parquet  （キー: rid16, ban + 生成特徴）
       data/_feats_1000_manifest.json （ファミリ別件数・leak チェック）
実行: PYTHONUTF8=1 ./venv311/Scripts/python.exe build_feats_1000.py
"""
from __future__ import annotations
import json, time, warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
BASE = Path(__file__).parent
MASTER_CSV = BASE / "data/master_v2_20130105-20251228.csv"
OUT_PARQUET = BASE / "data/feats_1000.parquet"
OUT_MANIFEST = BASE / "data/_feats_1000_manifest.json"

COL_RID = "レースID(新/馬番無)"
COL_BAN = "馬番"
COL_PED = "血統登録番号"
COL_DATE = "日付"
COL_JYUN = "着順"

# ---- レース内相対変換の土台（全て事前既知の連続/順序量, leak-safe）----
TRANSFORM_COLS = [
    "jockey_fuku30", "jockey_fuku90", "trainer_fuku30", "trainer_fuku90",
    "horse_fuku10", "horse_fuku30", "prev_pos_rel", "closing_power",
    "kako5_avg_pos", "kako5_std_pos", "kako5_best_pos", "kako5_avg_ninki",
    "kako5_pos_vs_ninki", "kako5_avg_agari3f", "kako5_best_agari3f",
    "kako5_same_td_ratio", "kako5_same_dist_ratio", "kako5_same_place_ratio",
    "kako5_pos_trend", "kako5_race_count", "kako5_expected_good_count",
    "kako5_upset_good_count", "kako5_hidden_good_count", "kako5_same_cond_best_pos",
    "hist_same_cond_best_pos", "hist_same_cond_top3_rate", "hist_same_cond_count",
    "hist_same_place_best_pos", "prev_hosei", "prev_hosei9",
    "trnH_Time1", "trnH_Time2", "trnH_Time3", "trnH_Time4",
    "trnH_Lap1", "trnH_Lap2", "trnH_Lap3", "trnH_Lap4", "trnH_days_ago",
    "trnW_5F", "trnW_4F", "trnW_3F", "trnW_Lap1", "trnW_Lap2", "trnW_Lap3", "trnW_days_ago",
    "course_n_prev", "course_win_rate", "course_top3_rate",
    "jockey_n_prev", "jockey_win_rate", "jockey_top3_rate",
    "前走確定着順", "前走上り3F", "前走上り3F順", "前走Ave-3F", "前PCI", "前走PCI3",
    "前走RPCI", "前走平均1Fタイム", "前走馬体重", "前走馬体重増減", "前距離", "前走出走頭数",
    "距離", "枠番", "馬番", "年齢", "間隔", "休み明け～戦目", "出走頭数",
    "フルゲート頭数", "馬齢斤量差", "斤量体重比", "騎手年齢", "調教師年齢",
]

# ---- ペア積のコア列（強い事前シグナル, 30 列 → C(30,2)=435 積）----
CORE_COLS = [
    "jockey_fuku30", "jockey_fuku90", "trainer_fuku30", "trainer_fuku90",
    "horse_fuku10", "horse_fuku30", "prev_pos_rel", "closing_power",
    "kako5_avg_pos", "kako5_best_pos", "kako5_std_pos", "kako5_avg_ninki",
    "kako5_pos_vs_ninki", "kako5_avg_agari3f", "kako5_best_agari3f",
    "kako5_pos_trend", "kako5_race_count", "prev_hosei", "prev_hosei9",
    "course_win_rate", "course_top3_rate", "jockey_win_rate", "jockey_top3_rate",
    "前走確定着順", "前走上り3F", "前走Ave-3F", "前PCI", "前走RPCI",
    "馬齢斤量差", "斤量体重比",
]

# ---- ペア比の厳選列（15 列 → C(15,2)=105 比）----
RATIO_COLS = [
    "horse_fuku30", "jockey_fuku30", "trainer_fuku30", "closing_power",
    "kako5_avg_pos", "kako5_best_pos", "kako5_avg_ninki", "kako5_avg_agari3f",
    "prev_hosei", "course_win_rate", "jockey_win_rate",
    "前走確定着順", "前走上り3F", "前PCI", "斤量体重比",
]

# F5 用マップ（exp_feateng_v7 準拠）
CLASS_MAP = {
    "新馬": 0, "未勝利": 0, "500万": 1, "1勝": 1, "1000万": 2, "2勝": 2,
    "1600万": 3, "3勝": 3, "ｵｰﾌﾟﾝ": 4, "オープン": 4, "OP(L)": 4, "OP": 4,
    "重賞": 5, "Ｇ３": 5, "Ｇ２": 6, "Ｇ１": 7,
}
GOING_MAP = {"良": 0, "稍": 1, "稍重": 1, "重": 2, "不": 3, "不良": 3}


def log(m): print(m, flush=True)


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def _rid16(s):
    return s.astype(str).str.replace(r"\D", "", regex=True).str[:16]


def safe_div(a, b):
    """0除算を NaN に、inf を NaN に。"""
    b = b.replace(0, np.nan)
    r = a / b
    return r.replace([np.inf, -np.inf], np.nan)


def load_master():
    import os
    nrows = os.environ.get("FEATS1000_NROWS")
    nrows = int(nrows) if nrows else None
    log(f"[load] {MASTER_CSV.name}  nrows={nrows or 'ALL'}")
    t0 = time.time()
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False, nrows=nrows)
    log(f"  rows={len(df):,} cols={len(df.columns)}  ({time.time()-t0:.0f}s)")
    return df


def build():
    t0 = time.time()
    df = load_master()

    # ---- キー ----
    out = pd.DataFrame(index=df.index)
    out["rid16"] = _rid16(df[COL_RID])
    out["ban"] = _num(df[COL_BAN]).fillna(0).astype(int)

    # 事前に数値化した土台列を辞書に
    base = {}
    for c in set(TRANSFORM_COLS) | set(CORE_COLS) | set(RATIO_COLS):
        if c in df.columns:
            base[c] = _num(df[c]).astype("float32")
    grp = df.groupby(COL_RID, sort=False)

    feat_count = {"F1_within_race": 0, "F2_pair_prod": 0, "F3_pair_ratio": 0,
                  "F4_pastform": 0, "F5_context_change": 0, "F6_career_asof": 0}

    # =========================================================
    # F1 レース内相対変換（6 変換 × TRANSFORM_COLS）leak-safe
    # =========================================================
    log("[F1] レース内相対変換 ...")
    for c in TRANSFORM_COLS:
        if c not in base:
            continue
        s = base[c]
        gmean = grp[c].transform("mean")
        gstd = grp[c].transform("std").replace(0, np.nan)
        gmax = grp[c].transform("max")
        gmin = grp[c].transform("min")
        grng = (gmax - gmin).replace(0, np.nan)
        out[f"f1_{c}__rz"] = ((s - gmean) / gstd).fillna(0.0).astype("float32")
        rank = grp[c].rank(method="average", na_option="keep")
        n = grp[c].transform("count")
        out[f"f1_{c}__rk"] = (((rank - 1) / (n - 1).replace(0, np.nan)).fillna(0.5)).astype("float32")
        out[f"f1_{c}__dm"] = (s - gmean).astype("float32")
        out[f"f1_{c}__dmax"] = (s - gmax).astype("float32")
        out[f"f1_{c}__dmin"] = (s - gmin).astype("float32")
        out[f"f1_{c}__rr"] = ((s - gmin) / grng).astype("float32")
        feat_count["F1_within_race"] += 6
    log(f"  F1={feat_count['F1_within_race']}  ({time.time()-t0:.0f}s)")

    # =========================================================
    # F2 ペア積（コア列の総当たり）leak-safe
    # =========================================================
    log("[F2] ペア積 ...")
    core = [c for c in CORE_COLS if c in base]
    for a, b in combinations(core, 2):
        out[f"f2_{a}__x__{b}"] = (base[a] * base[b]).astype("float32")
        feat_count["F2_pair_prod"] += 1
    log(f"  F2={feat_count['F2_pair_prod']}")

    # =========================================================
    # F3 ペア比（厳選列）leak-safe
    # =========================================================
    log("[F3] ペア比 ...")
    rat = [c for c in RATIO_COLS if c in base]
    for a, b in combinations(rat, 2):
        out[f"f3_{a}__over__{b}"] = safe_div(base[a], base[b]).astype("float32")
        feat_count["F3_pair_ratio"] += 1
    log(f"  F3={feat_count['F3_pair_ratio']}")

    # =========================================================
    # F4 過去走派生（kako5 / hist の比・差・ボラ）leak-safe
    # =========================================================
    log("[F4] 過去走派生 ...")
    def add4(name, series):
        out[name] = series.astype("float32")
        feat_count["F4_pastform"] += 1

    k = base
    if "kako5_avg_pos" in k and "kako5_best_pos" in k:
        add4("f4_kako5_pos_gap", k["kako5_avg_pos"] - k["kako5_best_pos"])
    if "kako5_avg_agari3f" in k and "kako5_best_agari3f" in k:
        add4("f4_kako5_agari_gap", k["kako5_avg_agari3f"] - k["kako5_best_agari3f"])
    if "kako5_std_pos" in k and "kako5_avg_pos" in k:
        add4("f4_kako5_pos_cv", safe_div(k["kako5_std_pos"], k["kako5_avg_pos"]))
    if "kako5_pos_vs_ninki" in k:
        add4("f4_kako5_overperf", -k["kako5_pos_vs_ninki"])  # 人気より上=好走
    if "kako5_avg_pos" in k and "kako5_avg_ninki" in k:
        add4("f4_kako5_pos_per_ninki", safe_div(k["kako5_avg_pos"], k["kako5_avg_ninki"]))
    if "kako5_expected_good_count" in k and "kako5_race_count" in k:
        add4("f4_kako5_exp_rate", safe_div(k["kako5_expected_good_count"], k["kako5_race_count"]))
    if "kako5_upset_good_count" in k and "kako5_race_count" in k:
        add4("f4_kako5_upset_rate", safe_div(k["kako5_upset_good_count"], k["kako5_race_count"]))
    if "kako5_hidden_good_count" in k and "kako5_race_count" in k:
        add4("f4_kako5_hidden_rate", safe_div(k["kako5_hidden_good_count"], k["kako5_race_count"]))
    if "hist_same_cond_top3_rate" in k and "kako5_same_place_ratio" in k:
        add4("f4_cond_x_place", k["hist_same_cond_top3_rate"] * k["kako5_same_place_ratio"])
    if "hist_same_cond_best_pos" in k and "kako5_best_pos" in k:
        add4("f4_cond_vs_overall_best", k["hist_same_cond_best_pos"] - k["kako5_best_pos"])
    # 調教の絶対値→ラップ差・5F-3F 等
    if "trnH_Lap1" in k and "trnH_Lap4" in k:
        add4("f4_trnH_accel", k["trnH_Lap4"] - k["trnH_Lap1"])  # 終い加速
    if "trnW_5F" in k and "trnW_3F" in k:
        add4("f4_trnW_5f_3f", k["trnW_5F"] - k["trnW_3F"])
    if "trnW_Lap1" in k and "trnW_Lap3" in k:
        add4("f4_trnW_accel", k["trnW_Lap3"] - k["trnW_Lap1"])
    # 前走指標の合成
    if "前走確定着順" in k and "前走出走頭数" in k:
        add4("f4_prev_pos_rel2", safe_div(k["前走確定着順"], k["前走出走頭数"]))
    if "前PCI" in k and "前走RPCI" in k:
        add4("f4_prev_pci_gap", k["前PCI"] - k["前走RPCI"])
    if "前走馬体重増減" in k and "前走馬体重" in k:
        add4("f4_prev_wt_pct", safe_div(k["前走馬体重増減"], k["前走馬体重"]))
    if "closing_power" in k and "前走上り3F" in k:
        add4("f4_close_x_agari", k["closing_power"] * (-_num(k["前走上り3F"]).astype("float32")))
    if "horse_fuku30" in k and "jockey_fuku30" in k:
        add4("f4_horse_jockey_synergy", k["horse_fuku30"] * k["jockey_fuku30"])
    if "horse_fuku30" in k and "horse_fuku10" in k:
        add4("f4_horse_form_trend", k["horse_fuku10"] - k["horse_fuku30"])  # 直近上昇
    if "jockey_fuku30" in k and "jockey_fuku90" in k:
        add4("f4_jockey_form_trend", k["jockey_fuku30"] - k["jockey_fuku90"])
    if "trainer_fuku30" in k and "trainer_fuku90" in k:
        add4("f4_trainer_form_trend", k["trainer_fuku30"] - k["trainer_fuku90"])
    if "斤量体重比" in k and "馬齢斤量差" in k:
        add4("f4_weight_burden", k["斤量体重比"] * k["馬齢斤量差"])
    log(f"  F4={feat_count['F4_pastform']}")

    # =========================================================
    # F5 条件変化（クラス昇降・距離/馬場/斤量/頭数）leak-safe
    # =========================================================
    log("[F5] 条件変化 ...")
    cls = df["クラス名"].astype(str).str.strip().map(CLASS_MAP).astype("float32")
    out["f5_class_ord"] = cls; feat_count["F5_context_change"] += 1
    # 前走クラス: 馬ごと日付ソートで直前 class_ord（厳密に過去）
    dd = _num(df[COL_DATE])
    tmp = pd.DataFrame({"_ped": df[COL_PED].astype(str), "_dd": dd,
                        "_rid": df[COL_RID].astype(str), "_cls": cls.values})
    tmp = tmp.sort_values(["_ped", "_dd", "_rid"], kind="mergesort")
    tmp["_prev_cls"] = tmp.groupby("_ped", sort=False)["_cls"].shift(1)
    tmp["_prev_dd"] = tmp.groupby("_ped", sort=False)["_dd"].shift(1)
    prev_cls = tmp["_prev_cls"].reindex(df.index)
    class_leak = int((tmp["_prev_dd"] > tmp["_dd"]).sum())
    out["f5_prev_class_ord"] = prev_cls.astype("float32"); feat_count["F5_context_change"] += 1
    out["f5_class_delta"] = (cls - prev_cls).astype("float32"); feat_count["F5_context_change"] += 1
    out["f5_class_drop"] = ((cls - prev_cls) < 0).astype("float32"); feat_count["F5_context_change"] += 1
    out["f5_class_up"] = ((cls - prev_cls) > 0).astype("float32"); feat_count["F5_context_change"] += 1
    dist = _num(df.get("距離")); pdist = _num(df.get("前距離"))
    out["f5_dist_change"] = (dist - pdist).astype("float32"); feat_count["F5_context_change"] += 1
    sd = df["芝・ダ"].astype(str).str.strip(); psd = df["前芝・ダ"].astype(str).str.strip()
    out["f5_surface_switch"] = ((sd != psd) & psd.ne("nan") & psd.ne("")).astype("float32"); feat_count["F5_context_change"] += 1
    gg = df["馬場状態"].astype(str).str.strip().map(GOING_MAP)
    pg = df["前走馬場状態"].astype(str).str.strip().map(GOING_MAP)
    out["f5_going_delta"] = (gg - pg).astype("float32"); feat_count["F5_context_change"] += 1
    w = _num(df.get("斤量")); pw = _num(df.get("前走斤量"))
    out["f5_weight_change"] = (w - pw).astype("float32"); feat_count["F5_context_change"] += 1
    f = _num(df.get("出走頭数")); pf = _num(df.get("前走出走頭数"))
    out["f5_field_change"] = (f - pf).astype("float32"); feat_count["F5_context_change"] += 1
    log(f"  F5={feat_count['F5_context_change']}  prev_class as-of違反={class_leak}")

    # =========================================================
    # F6 馬キャリア as-of 集計（厳密に過去のみ）leak-safe
    #   累積演算でベクトル化（apply+expanding は数十万グループで激重なので不使用）
    # =========================================================
    log("[F6] 馬キャリア as-of 集計 ...")
    car = pd.DataFrame({
        "_ped": df[COL_PED].astype(str), "_dd": dd, "_rid": df[COL_RID].astype(str),
        "_jyun": _num(df[COL_JYUN]).astype("float32"),
    })
    car = car.sort_values(["_ped", "_dd", "_rid"], kind="mergesort")
    ped = car["_ped"]
    gp = car.groupby(ped, sort=False)
    jyun = car["_jyun"]
    win = (jyun == 1).astype("float32")
    top3 = (jyun <= 3).astype("float32")

    runs = gp.cumcount().astype("float32")                 # 過去走数（当該を含めない＝0始まり）
    runs_nz = runs.replace(0, np.nan)
    # 過去のみの合計 = 累積(含む) - 当該
    cum_sum = gp["_jyun"].cumsum() - jyun
    cum_sq = (jyun * jyun).groupby(ped, sort=False).cumsum() - (jyun * jyun)
    cum_win = win.groupby(ped, sort=False).cumsum() - win
    cum_top3 = top3.groupby(ped, sort=False).cumsum() - top3
    prev_jyun = gp["_jyun"].shift(1)                       # 直前着順
    prev_dd = gp["_dd"].shift(1)                           # 直前日付

    c_mean = cum_sum / runs_nz
    c_var = (cum_sq / runs_nz) - c_mean * c_mean
    car["c_runs"] = runs
    car["c_mean_pos"] = c_mean
    car["c_best_pos"] = prev_jyun.groupby(ped, sort=False).cummin()
    car["c_std_pos"] = np.sqrt(c_var.clip(lower=0))
    car["c_last3_pos"] = (prev_jyun.groupby(ped, sort=False)
                          .rolling(3, min_periods=1).mean().reset_index(level=0, drop=True))
    car["c_last1_pos"] = prev_jyun
    car["c_win_rate"] = cum_win / runs_nz
    car["c_top3_rate"] = cum_top3 / runs_nz
    car["c_win_cnt"] = cum_win
    car["c_days_since"] = _to_days(car["_dd"]) - _to_days(prev_dd)
    car["c_recent_vs_mean"] = car["c_last1_pos"] - car["c_mean_pos"]   # <0=直近好調

    career_leak = int((prev_dd > car["_dd"]).sum())        # as-of 違反（過去日付 <= 当日）
    car_cols = ["c_runs", "c_mean_pos", "c_best_pos", "c_std_pos", "c_last3_pos",
                "c_last1_pos", "c_win_rate", "c_top3_rate", "c_win_cnt",
                "c_days_since", "c_recent_vs_mean"]
    car_re = car[car_cols].reindex(df.index)
    for c in car_cols:
        out[f"f6_{c}"] = car_re[c].astype("float32")
        feat_count["F6_career_asof"] += 1
    log(f"  F6={feat_count['F6_career_asof']}  career as-of違反={career_leak}")

    # =========================================================
    # 保存
    # =========================================================
    feat_cols = [c for c in out.columns if c not in ("rid16", "ban")]
    # 完全重複列の除去（万一の同名衝突は無いが、定数列は残す＝LGBMが無視）
    n_total = len(feat_cols)
    log(f"\n[total] 生成特徴 {n_total} 列  (+ keys rid16,ban)")

    # キー重複チェック
    dup = out.duplicated(subset=["rid16", "ban"]).sum()
    log(f"  key dup rows = {dup}")

    OUT_PARQUET.parent.mkdir(exist_ok=True)
    out.to_parquet(OUT_PARQUET, index=False)
    log(f"[saved] {OUT_PARQUET}  ({OUT_PARQUET.stat().st_size/1e6:.0f} MB)")

    manifest = {
        "n_features": n_total,
        "family_counts": feat_count,
        "leak_checks": {"class_prev_asof_violations": class_leak,
                         "career_asof_violations": career_leak,
                         "key_dup_rows": int(dup)},
        "transform_cols": len([c for c in TRANSFORM_COLS if c in base]),
        "core_cols": len(core), "ratio_cols": len(rat),
        "elapsed_sec": round(time.time() - t0, 1),
        "leak_safe": bool(class_leak == 0 and career_leak == 0),
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[saved] {OUT_MANIFEST}")
    log(json.dumps(manifest, ensure_ascii=False, indent=2))

    assert class_leak == 0, f"class as-of leak: {class_leak}"
    assert career_leak == 0, f"career as-of leak: {career_leak}"
    log("\n✅ leak-safe 確認OK")


def _to_days(s):
    """yyyymmdd 数値を概算 days に（日付差の単調尺度として十分）。"""
    s = pd.to_numeric(s, errors="coerce")
    y = (s // 10000); m = (s // 100) % 100; d = s % 100
    return y * 365 + m * 31 + d


if __name__ == "__main__":
    build()
