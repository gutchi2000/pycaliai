"""
build_umaren_pair_dataset.py
============================
master_v2 の各レースから全 (i, j) ペアを生成して、馬連二値分類用の
データセットを作る。

ターゲット: is_top2 = ({ban_i, ban_j} == {1着馬番, 2着馬番})

特徴量設計:
  - 個馬特徴 (NUM_FEATS) × 4 stat (mean, |diff|, min, max)
  - レース文脈 (CONTEXT_NUM, CONTEXT_CAT) はペアで共通の値

時系列分割 (master_v2 の split 列を継承):
  train: ~2022, valid: 2023, test: 2024-25

出力:
  data/umaren_pair_dataset.parquet  (約 130 万行 * 100+ 列)

実行:
  python build_umaren_pair_dataset.py
"""
from __future__ import annotations
import io
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

BASE       = Path(__file__).parent
MASTER_CSV = BASE / "data/master_v2_20130105-20251228.csv"
OUT_PARQ   = BASE / "data/umaren_pair_dataset.parquet"

COL_RID  = "レースID(新/馬番無)"
COL_JYUN = "着順"
COL_BAN  = "馬番"

# 個馬数値特徴 (ペア集約対象)
NUM_FEATS = [
    "jockey_fuku30", "jockey_fuku90", "jockey_top3_rate",
    "trainer_fuku30", "trainer_fuku90",
    "horse_fuku10", "horse_fuku30",
    "kako5_avg_pos", "kako5_best_pos", "kako5_avg_ninki",
    "kako5_pos_vs_ninki", "kako5_avg_agari3f", "kako5_best_agari3f",
    "kako5_same_td_ratio", "kako5_same_dist_ratio",
    "kako5_pos_trend", "kako5_race_count",
    "kako5_expected_good_count", "kako5_upset_good_count",
    "hist_same_cond_top3_rate", "hist_same_cond_count",
    "hist_same_place_best_pos",
    "prev_pos_rel", "closing_power", "prev_hosei9",
    "course_top3_rate", "course_n_prev", "course_win_rate",
    "前PCI", "前PCI3", "前走RPCI",
    "前走Ave-3F", "前走上り3F",
    "trnH_Time1", "trnH_Lap1", "trnH_days_ago",
    "trnW_3F", "trnW_Lap1", "trnW_days_ago",
]

# レース文脈 (数値、ペアで共通)
CONTEXT_NUM = [
    "距離", "出走頭数", "頭数",
]

# レース文脈 (カテゴリ、ペアで共通) — LabelEncoder で encode
CONTEXT_CAT = [
    "場所", "芝・ダ", "コース区分", "芝(内・外)", "馬場状態", "クラス名",
]


def gen_pairs_for_race(g: pd.DataFrame, num_feats_avail: list, ctx_num_avail: list,
                       ctx_cat_avail: list, split_val: str):
    g = g.sort_values(COL_BAN).reset_index(drop=True)
    n = len(g)
    if n < 3:
        return None
    i_idx, j_idx = np.triu_indices(n, k=1)
    n_pair = len(i_idx)
    if n_pair == 0:
        return None

    out = {
        COL_RID: [str(g[COL_RID].iloc[0])] * n_pair,
        "split": [split_val] * n_pair,
        "ban_i": g[COL_BAN].values[i_idx].astype(int),
        "ban_j": g[COL_BAN].values[j_idx].astype(int),
    }

    for c in num_feats_avail:
        v = pd.to_numeric(g[c], errors="coerce").values.astype(np.float32)
        xi = v[i_idx]
        xj = v[j_idx]
        out[f"{c}_mean"] = ((xi + xj) / 2).astype(np.float32)
        out[f"{c}_absdiff"] = np.abs(xi - xj).astype(np.float32)
        out[f"{c}_min"] = np.minimum(xi, xj).astype(np.float32)
        out[f"{c}_max"] = np.maximum(xi, xj).astype(np.float32)

    # レース文脈 (ペアで共通の値)
    for c in ctx_num_avail:
        v = pd.to_numeric(g[c].iloc[0], errors="coerce")
        out[c] = np.full(n_pair, v, dtype=np.float32)
    for c in ctx_cat_avail:
        out[c] = [str(g[c].iloc[0])] * n_pair

    # ターゲット: is_top2
    jyun = pd.to_numeric(g[COL_JYUN], errors="coerce").values
    ban  = g[COL_BAN].astype(int).values
    valid_jyun = ~np.isnan(jyun)
    if valid_jyun.sum() < 2:
        out["is_top2"] = np.zeros(n_pair, dtype=np.int8)
        return out
    order = np.argsort(np.where(valid_jyun, jyun, 1e9))
    win_ban = int(ban[order[0]])
    plc_ban = int(ban[order[1]])
    top2 = frozenset([win_ban, plc_ban])
    ban_i_arr = out["ban_i"]
    ban_j_arr = out["ban_j"]
    is_top2 = np.array([
        frozenset([int(a), int(b)]) == top2
        for a, b in zip(ban_i_arr, ban_j_arr)
    ], dtype=np.int8)
    out["is_top2"] = is_top2
    return out


def main():
    print("=" * 70)
    print("build_umaren_pair_dataset.py")
    print("=" * 70)
    print(f"[load] {MASTER_CSV}")
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    df = df.dropna(subset=[COL_RID, "split"]).copy()
    print(f"  rows={len(df):,}  races={df[COL_RID].nunique():,}")

    num_feats_avail = [c for c in NUM_FEATS if c in df.columns]
    ctx_num_avail   = [c for c in CONTEXT_NUM if c in df.columns]
    ctx_cat_avail   = [c for c in CONTEXT_CAT if c in df.columns]
    miss_num = [c for c in NUM_FEATS if c not in df.columns]
    miss_ctx = [c for c in CONTEXT_NUM + CONTEXT_CAT if c not in df.columns]
    print(f"  NUM_FEATS available: {len(num_feats_avail)}/{len(NUM_FEATS)}  "
          f"miss={miss_num}")
    print(f"  CONTEXT available:   "
          f"num={len(ctx_num_avail)}/{len(CONTEXT_NUM)}  "
          f"cat={len(ctx_cat_avail)}/{len(CONTEXT_CAT)}  "
          f"miss={miss_ctx}")

    print(f"\n[generate pairs] for each race...")
    parts = []
    n_total = 0
    n_skip = 0
    for rid, g in df.groupby(COL_RID, sort=False):
        split_val = g["split"].iloc[0]
        out = gen_pairs_for_race(
            g, num_feats_avail, ctx_num_avail, ctx_cat_avail, split_val
        )
        if out is None:
            n_skip += 1
            continue
        parts.append(pd.DataFrame(out))
        n_total += len(out["ban_i"])
        if len(parts) % 5000 == 0:
            print(f"  ..races={len(parts):,}  pairs={n_total:,}  skip={n_skip}")

    print(f"\n[concat] parts={len(parts):,}  pairs={n_total:,}")
    out_df = pd.concat(parts, ignore_index=True)
    print(f"  rows={len(out_df):,}  cols={out_df.shape[1]}")
    print(f"  split distribution:")
    print(out_df["split"].value_counts())
    print(f"  is_top2 rate by split:")
    for s in ["train", "valid", "test"]:
        sub = out_df[out_df["split"] == s]
        if len(sub) == 0: continue
        print(f"    {s:6s}: n={len(sub):>10,}  positive={int(sub['is_top2'].sum()):>7,}  "
              f"rate={sub['is_top2'].mean()*100:.3f}%")

    OUT_PARQ.parent.mkdir(exist_ok=True)
    out_df.to_parquet(OUT_PARQ, index=False)
    print(f"\n[saved] {OUT_PARQ}  ({OUT_PARQ.stat().st_size/(1024*1024):.1f} MB)")


if __name__ == "__main__":
    main()
