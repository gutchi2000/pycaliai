# -*- coding: utf-8 -*-
"""
Gate 0B（EXP13再開） — course_n_prev/jockey_n_prev系6特徴の数値照合（読み取り専用）
==========================================================================
2026-09-22。`GATE0B_FEATURE_AUDIT.md` §1-G/§3-2で指摘された「dropna後(DNF除去済み)
の母集団の上でexpanding cumulative集計をしている」というneeds_verification群のうち、
優先度が最も高いとされた6特徴(course_n_prev/win_rate/top3_rate、
jockey_n_prev/win_rate/top3_rate)について、dropna**前**の真のfull-starter母集団
(631,965行、止馬含む)で `build_master_v2.py:compute_history_features()` と
全く同一ロジックを再計算し、`master_v2_20130105-20251228.csv` の格納値と突合する。

方法論は `analysis/p0_5_verification/reconstruct_true_pipeline_universe.py`
(jockey_fuku30/90・trainer_fuku30/90を "diff=0" で無罪と実証した既存監査)を踏襲する。
本番ファイル・build_master_v2.py・master_v2は一切変更しない。読み取りのみ。

実行: venv311\\Scripts\\python.exe -m analysis.mcond.exp13_nonfinish_risk_dev.gate0b_course_jockey_history_parity
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(BASE))
from analysis.p0_5_verification.reconstruct_true_pipeline_universe import (  # noqa: E402
    build_full_pre_dropna_universe,
)

OUT_DIR = Path(__file__).resolve().parent / "out"
MASTER_V2 = BASE / "data" / "master_v2_20130105-20251228.csv"

COL_RID, COL_BAN, COL_DATE = "レースID(新)", "馬番", "日付"
COL_PEDIGREE, COL_PLACE, COL_SURFACE, COL_DIST = "血統登録番号", "場所", "芝・ダ", "距離"
COL_JOCKEY, COL_JYUN = "騎手コード", "着順"


def compute_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """build_master_v2.py:compute_history_features() と全く同一のロジック
    (コピーではなく意図的な再実装比較 — dropna前 vs dropna後の母集団差だけを
    分離して見るため、同一関数をそのままimportするのではなくロジックを複製する。
    差異が出た場合に「関数自体を変えていないか」を目視確認しやすくするため)。
    """
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
    df["_course_key"] = (df[COL_PLACE].astype(str) + "|" +
                          df[COL_SURFACE].astype(str) + "|" + df["_dist_b"])

    df = df.sort_values([COL_PEDIGREE, COL_DATE]).reset_index(drop=True)
    g = df.groupby([COL_PEDIGREE, "_course_key"])
    df["course_n_prev"] = g.cumcount()
    df["course_wins_prev"] = g["_is_win"].cumsum().astype("Int64") - df["_is_win"].astype("Int64")
    df["course_top3_prev"] = g["_is_top3"].cumsum().astype("Int64") - df["_is_top3"].astype("Int64")
    df["course_win_rate"] = np.where(df["course_n_prev"] > 0,
                                      df["course_wins_prev"] / df["course_n_prev"], np.nan)
    df["course_top3_rate"] = np.where(df["course_n_prev"] > 0,
                                       df["course_top3_prev"] / df["course_n_prev"], np.nan)

    gj = df.groupby([COL_PEDIGREE, COL_JOCKEY])
    df["jockey_n_prev"] = gj.cumcount()
    df["jockey_wins_prev"] = gj["_is_win"].cumsum().astype("Int64") - df["_is_win"].astype("Int64")
    df["jockey_top3_prev"] = gj["_is_top3"].cumsum().astype("Int64") - df["_is_top3"].astype("Int64")
    df["jockey_win_rate"] = np.where(df["jockey_n_prev"] > 0,
                                      df["jockey_wins_prev"] / df["jockey_n_prev"], np.nan)
    df["jockey_top3_rate"] = np.where(df["jockey_n_prev"] > 0,
                                       df["jockey_top3_prev"] / df["jockey_n_prev"], np.nan)
    return df


def main() -> int:
    print("[1] 631,965行のpre-dropna full-starter母集団を再構築...")
    full = build_full_pre_dropna_universe()
    assert len(full) == 631_965

    print("[2] true-universe版 compute_history_features() 実行 (DNF込み母集団)...")
    true_universe = compute_history_features(full.copy())
    survivors_true = true_universe[true_universe[COL_JYUN].notna()].copy()
    print(f"    survivors (着順notna) = {len(survivors_true):,}")

    print("[3] stored版(post-dropna母集団のみで計算)も同一ロジックで再現し、"
          "真に「post-dropna母集団で計算したら」を再現する(=現行のmaster_v2格納値相当)...")
    # 現行パイプラインは「dropna後の626,774行」を入力として同じロジックを回している。
    # ここではそのpost-dropna母集団を full から作り、同じcompute_history_features()を適用する。
    post_dropna_only = full[full[COL_JYUN].notna()].copy()
    stored_repro = compute_history_features(post_dropna_only)

    print("[4] master_v2格納値を読み込み...")
    cols = ["レースID(新/馬番無)", COL_BAN, "course_n_prev", "course_win_rate", "course_top3_rate",
            "jockey_n_prev", "jockey_win_rate", "jockey_top3_rate"]
    v2 = pd.read_csv(MASTER_V2, encoding="utf-8-sig", usecols=cols, low_memory=False)

    key = ["レースID(新/馬番無)", COL_BAN]
    stored_repro_k = stored_repro
    rid16_col = "レースID(新/馬番無)" if "レースID(新/馬番無)" in stored_repro.columns else None

    result = {"n_true_universe": len(true_universe), "n_survivors_true": len(survivors_true),
              "n_post_dropna_only": len(post_dropna_only)}

    def compare(a: pd.DataFrame, b: pd.DataFrame, label: str, join_cols):
        m = a[join_cols + ["course_n_prev", "course_win_rate", "course_top3_rate",
                            "jockey_n_prev", "jockey_win_rate", "jockey_top3_rate"]].merge(
            b[join_cols + ["course_n_prev", "course_win_rate", "course_top3_rate",
                            "jockey_n_prev", "jockey_win_rate", "jockey_top3_rate"]],
            on=join_cols, suffixes=("_a", "_b"))
        out = {}
        for c in ["course_n_prev", "course_win_rate", "course_top3_rate",
                  "jockey_n_prev", "jockey_win_rate", "jockey_top3_rate"]:
            x = pd.to_numeric(m[f"{c}_a"], errors="coerce")
            y = pd.to_numeric(m[f"{c}_b"], errors="coerce")
            both = x.notna() & y.notna()
            d = (x[both] - y[both]).abs()
            n_diff = int((d > 1e-9).sum())
            nan_mismatch = int((x.isna() != y.isna()).sum())
            out[c] = {"compared": int(both.sum()), "n_diff": n_diff, "nan_mismatch": nan_mismatch,
                      "max_abs_diff": float(d.max()) if len(d) else None,
                      "mean_abs_diff_where_diff": float(d[d > 1e-9].mean()) if n_diff else 0.0}
        result[label] = out
        print(f"  [{label}]")
        for c, v in out.items():
            print(f"    {c}: compared={v['compared']:,} diff={v['n_diff']:,} "
                  f"nan_mismatch={v['nan_mismatch']:,} max_abs_diff={v['max_abs_diff']}")

    if rid16_col:
        print("\n[5a] sanity check: stored_repro(post-dropna再現) vs master_v2格納値 "
              "(生成条件が同一なら一致するはず)")
        compare(stored_repro_k, v2, "sanity_stored_repro_vs_master_v2", key)

    print("\n[5b] 本題: true_universe(DNF込み) vs stored(DNF除去後) の差")
    if rid16_col:
        compare(survivors_true, stored_repro_k, "true_universe_vs_post_dropna_repro", key)

    print("\n[5c] 「対象馬の過去(cumcountされる範囲)にDNF走が実在する行」に絞った不一致率")
    dnf_hids = set(full.loc[full[COL_JYUN].isna(), COL_PEDIGREE])
    survivors_true["_hid_has_any_dnf_in_data"] = survivors_true[COL_PEDIGREE].isin(dnf_hids)
    n_affected_candidate = int(survivors_true["_hid_has_any_dnf_in_data"].sum())
    result["n_rows_belonging_to_a_horse_with_any_dnf_in_dataset"] = n_affected_candidate
    print(f"    生涯のどこかでDNF走を持つ馬に属する行(上限見積り) = {n_affected_candidate:,} "
          f"/ {len(survivors_true):,}")

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / "gate0b_course_jockey_history_parity.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\n[saved] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
