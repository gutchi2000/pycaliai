"""
corr_features.py
================
ゲート1（相関補正 M1）用の「馬同士の相関ドライバー」特徴量。

master_v2 から前走情報を使って各馬の脚質(running style)を導出する（前売り時点で既知＝leak-free）:
  style_rank ∈ [0,1] = (前コーナー通過順 − 1) / (前走出走頭数 − 1)
     0 = 逃げ(前) ... 1 = 追込(後)。前2角優先、欠損は前1角→前3角→前4角でフォールバック。
  style_cat: 逃げ(<0.2) / 先行(<0.45) / 差し(<0.75) / 追込(>=0.75) / unknown(前走なし)

レース単位:
  pace_pressure = 逃げ・先行頭数 / 出走頭数（ハイペース度）
  n_front       = style_rank < 0.30 の頭数

ペア/トリオの相関特徴量は gate1 側で style_rank と 枠番 から構築する。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COL_RID = "レースID(新/馬番無)"

# 前走コーナー通過順（早い順に優先） + 前走出走頭数
_CORNER_COLS = ["前2角", "前1角", "前3角", "前4角"]
_NTOU_COL = "前走出走頭数"


def _to_num(s):
    return pd.to_numeric(s, errors="coerce")


def add_style(df: pd.DataFrame) -> pd.DataFrame:
    """df に style_rank / style_cat / pace 系を付加して返す。"""
    df = df.copy()
    ntou = _to_num(df[_NTOU_COL]) if _NTOU_COL in df.columns else pd.Series(np.nan, index=df.index)
    # 早いコーナーを優先して最初に有効な通過順を採用
    corner = pd.Series(np.nan, index=df.index)
    for c in _CORNER_COLS:
        if c in df.columns:
            cv = _to_num(df[c])
            corner = corner.where(corner.notna(), cv)
    # style_rank = (通過順-1)/(頭数-1)。頭数<=1 or 欠損は NaN
    denom = (ntou - 1).where(ntou > 1, np.nan)
    style = (corner - 1) / denom
    style = style.clip(0.0, 1.0)
    df["style_rank"] = style
    df["has_style"] = style.notna()

    def _cat(x):
        if pd.isna(x):
            return "unknown"
        if x < 0.20:
            return "逃げ"
        if x < 0.45:
            return "先行"
        if x < 0.75:
            return "差し"
        return "追込"
    df["style_cat"] = df["style_rank"].apply(_cat)

    # レース単位ペース
    g = df.groupby(COL_RID)
    is_front = (df["style_rank"] < 0.30).astype(float)  # 逃げ・先行寄り
    df["_is_front"] = is_front
    df["n_front"] = g["_is_front"].transform("sum")
    df["field_size_eff"] = g["style_rank"].transform("size")
    df["pace_pressure"] = df["n_front"] / df["field_size_eff"].clip(lower=1)
    df.drop(columns=["_is_front"], inplace=True)
    return df


def pair_features(style_i, style_j, waku_i, waku_j, pace_pressure, field_size):
    """ペア (i,j) の相関特徴量ベクトルを返す（gate1 の条件付きロジット用）。
    style は NaN なら中立 0.5 とみなす。
    """
    si = 0.5 if pd.isna(style_i) else float(style_i)
    sj = 0.5 if pd.isna(style_j) else float(style_j)
    both_front = 1.0 if (si < 0.30 and sj < 0.30) else 0.0       # 両逃げ/先行 → ペース潰し
    both_closer = 1.0 if (si >= 0.70 and sj >= 0.70) else 0.0    # 両差し/追込
    front_closer = 1.0 if (min(si, sj) < 0.30 and max(si, sj) >= 0.55) else 0.0
    style_gap = abs(si - sj)
    draw_gap = abs(float(waku_i) - float(waku_j)) / 8.0 if (waku_i and waku_j) else 0.0
    return np.array([
        both_front,
        both_closer,
        front_closer,
        style_gap,
        both_front * float(pace_pressure),   # ハイペースでの両前は特に潰し合い
        draw_gap,
    ], dtype=np.float64)


PAIR_FEATURE_NAMES = ["both_front", "both_closer", "front_closer",
                      "style_gap", "both_front_x_pace", "draw_gap"]


if __name__ == "__main__":
    import io, sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    usecols = [COL_RID, "馬番", "枠番", "着順", "split", _NTOU_COL] + _CORNER_COLS
    df = pd.read_csv(r"E:\PyCaLiAI\data\master_v2_20130105-20251228.csv",
                     encoding="utf-8-sig", usecols=lambda c: c in usecols, low_memory=False)
    print(f"rows={len(df):,}")
    df = add_style(df)
    cov = df["has_style"].mean()
    print(f"style coverage (前走通過順あり): {cov*100:.1f}%")
    print("style_cat 分布:")
    print(df["style_cat"].value_counts(normalize=True).round(3).to_string())
    print(f"\npace_pressure: mean={df['pace_pressure'].mean():.3f} "
          f"min={df['pace_pressure'].min():.3f} max={df['pace_pressure'].max():.3f}")
    print("\nstyle_rank sample (head):")
    print(df[[ "前2角", _NTOU_COL, "style_rank", "style_cat"]].head(8).to_string())
