# -*- coding: utf-8 -*-
"""
serve_parity_check.py — F-forward生成ロジック(live_history.py)の定義同値性チェック
========================================================================================
本来のspec要求(§6)は「過去期間でライブ特徴生成処理を再現し、M4-forward vs M3を
2023-2025で比較する」ことだが、data/weekly/にはEXP05-F開始時点で2023-2025分の
週次CSVアーカイブが1つも無い(2026年分のみ保持)。よって「ライブと同じ入力経路」での
真のreplayは実行不可能 (LOCKED_PERIOD_AUDIT.md/README.md同様のデータ制約として記録)。

代わりに: live_history.py の C2/C3計算ロジックを、master由来のtarget行(週次CSVの
代わりにmaster_v2.csvの該当日の行を使う。値そのものはmasterから取るので週次CSV
特有のフォーマットゆれ(このセッションで見つかった"ダート"表記等)は再現されないが、
「時点安全なchain構築・突合ロジック自体が既存のexp01/exp02/exp03の値と一致するか」
という定義同値性は検証できる)を使ってhistorical dateで計算し、
既存のexp01_features.parquet/exp02_features.parquet/exp03_features.parquetの値と突合する。

実行: python -m analysis.mcond.exp05_forward_shadow.serve_parity_check
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
from grade_feats import class_name_to_ord  # noqa: E402
from analysis.mcond.exp05_forward_shadow import live_history as LH  # noqa: E402

TEST_DATE = "2025-06-01"
MASTER = BASE / "data/master_v2_20130105-20251228.csv"


def build_master_target(date_str: str) -> pd.DataFrame:
    m = pd.read_csv(MASTER, encoding="utf-8-sig", low_memory=False,
                    usecols=["日付", "レースID(新)", "血統登録番号", "馬番", "場所", "芝・ダ",
                             "距離", "クラス名", "騎手コード", "調教師コード"])
    m["date"] = pd.to_datetime(m["日付"].astype(str), format="%Y%m%d", errors="coerce")
    day = m[m["date"] == pd.Timestamp(date_str)].copy()
    day["rid16"] = day["レースID(新)"].astype(str).str[:16]
    day["ban"] = pd.to_numeric(day["馬番"], errors="coerce").astype(int)
    return pd.DataFrame({
        "ident": day["血統登録番号"].astype(str), "date": day["date"],
        "rid16": day["rid16"], "ban": day["ban"],
        "venue": day["場所"], "surface": day["芝・ダ"],
        "dist": pd.to_numeric(day["距離"], errors="coerce"),
        "cls_ord": day["クラス名"].map(class_name_to_ord),
        "jockey": day["騎手コード"].astype(str), "trainer": day["調教師コード"].astype(str),
    })


def main() -> None:
    target = build_master_target(TEST_DATE)
    print(f"target rows: {len(target)} ({TEST_DATE})")

    hist = LH.build_historical_chain()
    chain = hist[hist["date"] < pd.Timestamp(TEST_DATE)].reset_index(drop=True)
    print(f"chain rows (< {TEST_DATE}): {len(chain)}")

    c2_live = LH.compute_c2_from_chain(target, chain)
    dyn_state = LH.compute_dyn_skill_live(chain)

    c2_ref = pd.read_parquet(BASE / "data/_research/mcond/exp01_features.parquet")
    c2_ref = c2_ref[c2_ref["date"] == pd.Timestamp(TEST_DATE)]
    c3a_ref = pd.read_parquet(BASE / "data/_research/mcond/exp02_features.parquet")
    c3b_ref = pd.read_parquet(BASE / "data/_research/mcond/exp03_features.parquet")

    key = target[["rid16", "ban"]].reset_index(drop=True)
    c2_live = c2_live.reset_index(drop=True)
    merged = key.assign(**{f"live_{c}": c2_live[c] for c in c2_live.columns})
    c2_ref["rid16"] = c2_ref["rid16"].astype(str)
    merged = merged.merge(c2_ref[["rid16", "ban", "raw_log_int", "raw_dist_chg", "raw_venue_chg",
                                  "raw_surface_chg", "raw_cls_chg", "raw_jockey_same", "raw_jq_delta",
                                  "raw_jt_pair"]].add_prefix("ref_").rename(
        columns={"ref_rid16": "rid16", "ref_ban": "ban"}), on=["rid16", "ban"], how="left")

    print("\n=== C2 (陣営選択) 定義同値性 ===")
    for col in ["raw_log_int", "raw_dist_chg", "raw_venue_chg", "raw_surface_chg",
               "raw_cls_chg", "raw_jockey_same", "raw_jt_pair"]:
        a, b = merged[f"live_{col}"], merged[f"ref_{col}"]
        both = a.notna() & b.notna()
        if both.sum() == 0:
            print(f"  {col}: 比較可能行なし")
            continue
        diff = (a[both] - b[both]).abs()
        match_rate = float((diff < 1e-6).mean())
        print(f"  {col}: n={both.sum()} 完全一致率={match_rate:.3f} 最大絶対差={diff.max():.4f}")

    # raw_jq_delta は基準側がbatch定義(同日内の他レースとの突合込み)で厳密一致しない設計
    # (README.md参照)。相関だけ見る。
    both = merged["live_raw_jq_delta"].notna() & merged["ref_raw_jq_delta"].notna()
    if both.sum() > 2:
        corr = merged.loc[both, "live_raw_jq_delta"].corr(merged.loc[both, "ref_raw_jq_delta"])
        print(f"  raw_jq_delta: n={both.sum()} 相関={corr:.3f} (定義上の理由で完全一致は期待しない、README.md参照)")

    print("\n=== C3 (dyn_skill) 定義同値性 ===")
    hid_key = target["ident"]
    live_mu = hid_key.map(dyn_state["g_mu"])
    c3a_ref["rid16"] = c3a_ref["rid16"].astype(str)
    ref_row = key.merge(c3a_ref[["rid16", "ban", "dyn_skill_mu"]], on=["rid16", "ban"], how="left")
    both = live_mu.notna().to_numpy() & ref_row["dyn_skill_mu"].notna().to_numpy()
    if both.sum() > 0:
        diff = np.abs(live_mu.to_numpy()[both] - ref_row["dyn_skill_mu"].to_numpy()[both])
        print(f"  dyn_skill_mu: n={both.sum()} 完全一致率(<1e-4)={float((diff < 1e-4).mean()):.3f} "
             f"最大絶対差={diff.max():.4f}")
    else:
        print("  dyn_skill_mu: 比較可能行なし")


if __name__ == "__main__":
    main()
