# -*- coding: utf-8 -*-
"""
Gate 0B（EXP13再開） — kako5_*(16特徴)+hist_same_cond/place_*(4特徴)の数値照合
==========================================================================
2026-09-22。`GATE0B_FEATURE_AUDIT.md` §1-E/§1-F/§3-1で指摘された、
parse_kako5.py `build_from_master()` が **dropna後(DNF除去済み)の626,774行**
(`data/master_20130105-20251228.csv`)を入力にしている問題を検証する。

`data/master_kako5.csv`(既存の本番中間生成物、post-dropna母集団でparse_kako5.pyを
実行した結果そのもの)を "stored" として使い、true-universe(631,965行、DNF込み)版を
`parse_kako5.py` の `_compute_features()` を**直接import**して(ロジック複製ではなく)
同一関数で計算し、突合する。本番ファイル(`data/master_kako5.csv`等)は一切変更しない。

実行: venv311\\Scripts\\python.exe -m analysis.mcond.exp13_nonfinish_risk_dev.gate0b_kako5_history_parity
（parse_kako5.pyと同じ馬ごとPythonループのため、631,965行で相応の時間を要する）
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(BASE))
from analysis.p0_5_verification.reconstruct_true_pipeline_universe import (  # noqa: E402
    build_full_pre_dropna_universe,
)
from parse_kako5 import _compute_features, _safe_float, _safe_int, KAKO5_COLS, HIST_COLS  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "out"
MASTER_KAKO5 = BASE / "data" / "master_kako5.csv"


def run_kako5_loop(df: pd.DataFrame) -> pd.DataFrame:
    """parse_kako5.py:build_from_master() の中核ループをそのまま踏襲
    (ファイル入出力だけをインメモリDataFrameに差し替え)。"""
    df = df.copy()
    df["_date"] = pd.to_datetime(df["date_dt"], errors="coerce")
    td_map = {"芝": "T", "ダ": "D", "ダート": "D", "T": "T", "D": "D"}
    df["_td_code"] = df["芝・ダ"].map(td_map).fillna("")
    df["_dist"] = pd.to_numeric(df["距離"], errors="coerce")
    df["_place"] = df["場所"].astype(str)

    horse_col = "血統登録番号"
    df = df.sort_values([horse_col, "_date"]).reset_index(drop=True)

    feat_rows = []
    t0 = time.time()
    n_horses = df[horse_col].nunique()
    for hi, (horse_id, group) in enumerate(df.groupby(horse_col, sort=False)):
        if hi % 20000 == 0:
            print(f"    {hi:,}/{n_horses:,} 頭処理済み ({time.time()-t0:.0f}s)", flush=True)
        group = group.sort_values("_date")
        idxs = group.index.tolist()
        着順_vals = pd.to_numeric(group["着順"], errors="coerce").values
        all_td = group["_td_code"].values
        all_dist = group["_dist"].values
        all_place = group["_place"].values

        for seq_i, idx in enumerate(idxs):
            row = group.loc[idx]
            past_indices = idxs[max(0, seq_i - 5): seq_i]
            past_races = []
            for pi in reversed(past_indices):
                pr = group.loc[pi]
                past_races.append({
                    "着順": _safe_int(pr.get("着順")),
                    "人気": _safe_int(pr.get("人気")) if "人気" in pr.index else None,
                    "上り3F": _safe_float(pr.get("前走上り3F")),
                    "TD": td_map.get(str(pr.get("芝・ダ", "")), ""),
                    "距離": _safe_float(pr.get("距離")),
                    "場所": str(pr.get("場所", "")),
                })
            feats = _compute_features(
                past_races,
                current_td=row.get("_td_code"),
                current_dist=_safe_float(row.get("_dist")),
                current_place=row.get("_place"),
            )
            cur_td = row.get("_td_code", "")
            cur_dist = _safe_float(row.get("_dist"))
            cur_place = row.get("_place", "")
            for hcol in HIST_COLS:
                feats[hcol] = np.nan
            if seq_i > 0 and cur_td and cur_dist is not None:
                same_cond_pos = []
                for pi_seq in range(seq_i):
                    pos = 着順_vals[pi_seq]
                    if np.isnan(pos):
                        continue
                    if all_td[pi_seq] == cur_td and not np.isnan(all_dist[pi_seq]) and \
                            abs(all_dist[pi_seq] - cur_dist) <= 200:
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
            feats["_idx"] = idx
            feat_rows.append(feats)

    feat_df = pd.DataFrame(feat_rows).set_index("_idx")
    all_cols = KAKO5_COLS + HIST_COLS
    for col in all_cols:
        df[col] = feat_df[col] if col in feat_df.columns else np.nan
    return df


def main() -> int:
    print("[1] pre-dropna full-starter母集団(631,965行)を再構築...")
    full = build_full_pre_dropna_universe()
    full["date_dt"] = pd.to_datetime(full["日付"].astype(str), format="%Y%m%d", errors="coerce")

    print("[2] true-universe版 kako5/hist_same_* を計算 (馬ごとPythonループ、時間がかかる)...")
    t0 = time.time()
    true_universe = run_kako5_loop(full)
    print(f"    完了: {time.time()-t0:.0f}s")
    survivors_true = true_universe[pd.to_numeric(true_universe["着順"], errors="coerce").notna()].copy()
    print(f"    survivors = {len(survivors_true):,}")

    print("[3] 既存の本番中間生成物 data/master_kako5.csv (post-dropna, stored) を読み込み...")
    stored = pd.read_csv(MASTER_KAKO5, encoding="utf-8-sig", low_memory=False,
                          usecols=["レースID(新/馬番無)", "馬番"] + KAKO5_COLS + HIST_COLS)

    key = ["レースID(新/馬番無)", "馬番"]
    all_cols = KAKO5_COLS + HIST_COLS
    m = survivors_true[key + all_cols].merge(stored, on=key, suffixes=("_true", "_stored"))
    print(f"    突合行数 = {len(m):,}")

    result = {"n_true_universe": int(len(true_universe)), "n_survivors_true": int(len(survivors_true)),
              "n_stored": int(len(stored)), "n_matched": int(len(m))}
    per_col = {}
    for c in all_cols:
        a = pd.to_numeric(m[f"{c}_true"], errors="coerce")
        b = pd.to_numeric(m[f"{c}_stored"], errors="coerce")
        both = a.notna() & b.notna()
        d = (a[both] - b[both]).abs()
        n_diff = int((d > 1e-9).sum())
        nan_mismatch = int((a.isna() != b.isna()).sum())
        per_col[c] = {"compared": int(both.sum()), "n_diff": n_diff, "nan_mismatch": nan_mismatch,
                      "max_abs_diff": float(d.max()) if len(d) else None}
        print(f"    {c}: compared={per_col[c]['compared']:,} diff={n_diff:,} "
              f"nan_mismatch={nan_mismatch:,} max_abs_diff={per_col[c]['max_abs_diff']}")
    result["per_column"] = per_col

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / "gate0b_kako5_history_parity.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[saved] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
