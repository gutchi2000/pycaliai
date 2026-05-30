"""
race_relative_feats.py
=======================
master_v2 に race-relative な新規特徴量を付加する。
train と predict の両方から使う共通モジュール。

v1 (legacy):
  base × 2 stat = 12 新規 (z-score, rank)
v2 (v7 で導入): 追加で
  - {col}_race_max / _race_min : レース内 max/min (相手陣営の強さ)
  - kako5_agari3f_volatility   : kako5_avg_agari3f - kako5_best_agari3f
                                  (ペース耐性: 上がりブレ幅。小さいほど安定脚質)
  - kako5_pos_volatility       : kako5_std_pos / max(kako5_avg_pos, 1)
                                  (着順安定度)
  - peer_count                 : 同レース頭数

公開関数:
  add_race_relative_feats(df, mode='v2')
      mode='v1' で legacy 列のみ、'v2' で全部追加。
  added_columns(mode='v2')
      mode に応じて追加列名のリスト。
"""
from __future__ import annotations
import numpy as np
import pandas as pd

COL_RID = "レースID(新/馬番無)"

BASE_COLS = [
    "jockey_fuku30",
    "horse_fuku30",
    "trainer_fuku30",
    "kako5_avg_pos",
    "closing_power",
    "prev_pos_rel",
]

# v7 追加: peer max を取る列 (相手陣営の強さ指標)
PEER_MAX_COLS = [
    "jockey_fuku30",
    "horse_fuku30",
    "trainer_fuku30",
]


def add_race_relative_feats(df: pd.DataFrame, mode: str = "v2") -> pd.DataFrame:
    """df に race-relative 特徴量を追加して返す.

    mode='v1': z-score + rank (12 列)
    mode='v2': v1 + peer_max + peer_min + pace volatility + peer_count
    """
    df = df.copy()
    for c in BASE_COLS:
        if c not in df.columns:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")

    g = df.groupby(COL_RID, sort=False)

    # ===== v1: z-score, rank =====
    for c in BASE_COLS:
        if c not in df.columns:
            continue
        mean = g[c].transform("mean")
        std  = g[c].transform("std").replace(0, np.nan)
        df[f"{c}_race_z"] = ((df[c] - mean) / std).fillna(0.0)
        rank = g[c].rank(method="average", na_option="keep")
        n = g[c].transform("count")
        df[f"{c}_race_rank"] = ((rank - 1) / (n - 1).replace(0, np.nan)).fillna(0.5)

    if mode == "v1":
        return df

    # ===== v2: peer_max, peer_min, pace volatility, peer_count =====
    for c in PEER_MAX_COLS:
        if c not in df.columns:
            continue
        df[f"{c}_peer_max"] = g[c].transform("max")
        df[f"{c}_peer_min"] = g[c].transform("min")

    # ペース耐性: kako5 平均上がり3F - 最高 (= 上がりのブレ幅, 小さいほど安定)
    if "kako5_avg_agari3f" in df.columns and "kako5_best_agari3f" in df.columns:
        df["kako5_avg_agari3f"]  = pd.to_numeric(df["kako5_avg_agari3f"],  errors="coerce")
        df["kako5_best_agari3f"] = pd.to_numeric(df["kako5_best_agari3f"], errors="coerce")
        df["kako5_agari3f_volatility"] = (
            df["kako5_avg_agari3f"] - df["kako5_best_agari3f"]
        ).astype(float)
        # レース内 z
        mean = g["kako5_agari3f_volatility"].transform("mean")
        std  = g["kako5_agari3f_volatility"].transform("std").replace(0, np.nan)
        df["kako5_agari3f_volatility_race_z"] = (
            (df["kako5_agari3f_volatility"] - mean) / std
        ).fillna(0.0)

    # 着順安定度
    if "kako5_std_pos" in df.columns and "kako5_avg_pos" in df.columns:
        std_pos = pd.to_numeric(df["kako5_std_pos"], errors="coerce")
        avg_pos = pd.to_numeric(df["kako5_avg_pos"], errors="coerce").clip(lower=1)
        df["kako5_pos_volatility"] = (std_pos / avg_pos).astype(float)

    # 同レース頭数 (重い印に対する相手の強さスケール)
    df["peer_count"] = g.transform("size").iloc[:, 0] if False else g[COL_RID].transform("size")

    return df


def added_columns(mode: str = "v2"):
    out = []
    for c in BASE_COLS:
        out.append(f"{c}_race_z")
        out.append(f"{c}_race_rank")
    if mode == "v1":
        return out
    for c in PEER_MAX_COLS:
        out.append(f"{c}_peer_max")
        out.append(f"{c}_peer_min")
    out += [
        "kako5_agari3f_volatility",
        "kako5_agari3f_volatility_race_z",
        "kako5_pos_volatility",
        "peer_count",
    ]
    return out


if __name__ == "__main__":
    # 軽量スモークテスト
    import pandas as pd, numpy as np
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        COL_RID: ["R1"]*8 + ["R2"]*10,
        "jockey_fuku30": rng.uniform(0.05, 0.4, 18),
        "horse_fuku30":  rng.uniform(0.05, 0.5, 18),
        "trainer_fuku30":rng.uniform(0.05, 0.4, 18),
        "kako5_avg_pos": rng.uniform(2, 10, 18),
        "kako5_std_pos": rng.uniform(0.5, 4, 18),
        "kako5_avg_agari3f": rng.uniform(33, 38, 18),
        "kako5_best_agari3f": rng.uniform(32, 36, 18),
        "closing_power": rng.uniform(-1, 1, 18),
        "prev_pos_rel": rng.uniform(0, 1, 18),
    })
    out = add_race_relative_feats(df, mode="v2")
    added = added_columns(mode="v2")
    print(f"add_race_relative_feats(mode='v2'):")
    print(f"  original cols: {len(df.columns)}")
    print(f"  added cols:    {len(added)}")
    print(f"  result cols:   {len(out.columns)}")
    for c in added:
        assert c in out.columns, f"missing: {c}"
        # 数値であること
        assert pd.api.types.is_numeric_dtype(out[c]) or out[c].dtype == object, c
        n_nan = out[c].isna().sum()
        print(f"  {c:42s}  range=[{out[c].min():>+8.3f}, {out[c].max():>+8.3f}]  nan={n_nan}")
    print("OK")
