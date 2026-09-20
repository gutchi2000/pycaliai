# -*- coding: utf-8 -*-
"""
expected_time_model.py — 距離・芝ダ・コース区分・公表馬場状態・クラスで条件付けた
「期待走破タイム」ルックアップを2013-2022年(train)だけで構築する。

設計判断(DATA_AUDIT.md §6参照):
  - 公表馬場状態(良/稍重/重/不良)は条件付け変数に含める。EXP08の目的は
    「公表馬場状態の後にも同日先行レースの残差情報が残るか」であり、公表
    馬場状態自体を差し引いた残差を見る必要がある(v6+市場を差し引いてから
    残差を見る§6の枠組みと同型)。
  - 距離は50m単位でbucket化(コース設定上、実際の距離は概ねこの粒度で離散)。
  - セル(距離bucket, 芝・ダ, コース区分, 馬場状態, クラス名)が2023-2025年で
    train期間に存在しない/samples不足の場合、段階的にフォールバックする
    (exact -> 馬場状態を除く -> クラスも除く -> 芝ダ+距離のみ)。
  - 中央値(median)を使う(平均より外れ値=事故・出遅れに頑健)。
  - 2024-2025年の結果を見て再fitしない(train<=2022固定)。

実行: 単体実行しない。build_observations.py から import して使う。
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[3]
KEKKA_EXT = "E:/競馬過去走データ/raw_data/kekka_2010_2025_fix_raceid_v2__keyed.csv"

MIN_CELL_N = 30  # このセルの中央値を信頼できる最低サンプル数


_FULLWIDTH_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")


def _parse_finish_pos(s: pd.Series) -> pd.Series:
    """着順列は1-9着が全角数字("１"〜"９")、10着以降が半角数字という不均一エンコード
    (2026-09-20実装中に発覚: pd.to_numericへそのまま通すと全角の1-9着が軒並みNaNへ
    落ち、勝ち馬を含む上位9頭が解析全体から消える実害バグだった)。半角へ正規化してから
    数値化する。取消・除外・中止等の非数値(全角の別記号)は正規化後もNaNのまま残り、
    正しく除外される。"""
    normalized = s.astype(str).str.translate(_FULLWIDTH_DIGITS)
    return pd.to_numeric(normalized, errors="coerce")


def _parse_time_sec(t) -> float:
    """走破タイムの4桁(M+SS+d)エンコードを秒(float)へ。例: 1336 -> 93.6"""
    t = pd.to_numeric(t, errors="coerce")
    minute = (t // 1000)
    sec = (t % 1000) // 10
    decisec = (t % 10)
    return minute * 60 + sec + decisec / 10.0


def load_kekka_for_time_model(usecols=None) -> pd.DataFrame:
    cols = usecols or [
        "yyyymmdd", "race_id16", "レース名", "馬番", "着順", "走破タイム", "距離", "芝・ダ",
        "コース区分", "馬場状態", "クラス名", "多頭出し", "上り3F", "RPCI", "PCI",
    ]
    df = pd.read_csv(KEKKA_EXT, encoding="utf-8-sig", low_memory=False, usecols=cols)
    # 障害(jump)レースはクラス名が下級クラスで平地と同じラベル("未勝利"等)を
    # 共有するためクラス名だけでは判別不能。レース名の"障害"部分文字列で明示的に除外
    # (混在すると同じ距離bucketの中央値が平地より大幅に遅い障害タイムで歪む)。
    n_before = len(df)
    df = df[~df["レース名"].astype(str).str.contains("障害", na=False)]
    n_obstacle = n_before - len(df)
    df["date"] = df["yyyymmdd"].astype(str)
    df["year"] = df["date"].str[:4].astype(int)
    df["time_sec"] = _parse_time_sec(df["走破タイム"])
    df["dist_bucket"] = (pd.to_numeric(df["距離"], errors="coerce") // 50 * 50).astype("Int64")
    fin = _parse_finish_pos(df["着順"])
    df = df[fin.notna() & (fin >= 1)]  # 中止・取消・失格(着順非数値)を除外
    df["fin"] = fin
    # コース区分はダートでは構造的に常にNaN(A/B/C/D等の回り設定は芝のみの概念)。
    # 「欠損」として行ごと落とすと全ダートレースが消える実害バグになるため、
    # ダートは単一カテゴリ"D_NA"として埋め、条件付けキーとして有効にする。
    df["コース区分"] = df["コース区分"].fillna("D_NA")
    df = df.dropna(subset=["time_sec", "dist_bucket", "芝・ダ", "コース区分",
                            "馬場状態", "クラス名"])
    return df


def fit_expected_time_lookup(train_df: pd.DataFrame) -> dict:
    """train<=2022のデータから段階的フォールバック付きの期待タイムlookupを作る。"""
    keys_full = ["dist_bucket", "芝・ダ", "コース区分", "馬場状態", "クラス名"]
    keys_no_going = ["dist_bucket", "芝・ダ", "コース区分", "クラス名"]
    keys_coarse = ["dist_bucket", "芝・ダ"]

    def build_level(keys):
        g = train_df.groupby(keys)["time_sec"]
        med = g.median()
        n = g.size()
        med = med[n >= MIN_CELL_N]
        return med

    lvl_full = build_level(keys_full)
    lvl_no_going = build_level(keys_no_going)
    lvl_coarse = build_level(keys_coarse)
    global_median = float(train_df["time_sec"].median())

    return {
        "keys_full": keys_full, "keys_no_going": keys_no_going, "keys_coarse": keys_coarse,
        "lvl_full": lvl_full, "lvl_no_going": lvl_no_going, "lvl_coarse": lvl_coarse,
        "global_median": global_median,
    }


def predict_expected_time(lookup: dict, df: pd.DataFrame) -> pd.Series:
    """段階的フォールバックで期待タイム(秒)を予測する。"""
    out = pd.Series(np.nan, index=df.index, dtype=float)
    fallback_used = pd.Series("none", index=df.index, dtype=object)

    idx_full = pd.MultiIndex.from_frame(df[lookup["keys_full"]])
    m = lookup["lvl_full"].reindex(idx_full)
    hit = m.notna().to_numpy()
    out.loc[hit] = m.to_numpy()[hit]
    fallback_used.loc[hit] = "full"

    miss = ~hit
    if miss.any():
        idx_ng = pd.MultiIndex.from_frame(df.loc[miss, lookup["keys_no_going"]])
        m2 = lookup["lvl_no_going"].reindex(idx_ng)
        hit2 = m2.notna().to_numpy()
        idxer = df.index[miss]
        out.loc[idxer[hit2]] = m2.to_numpy()[hit2]
        fallback_used.loc[idxer[hit2]] = "no_going"
        miss2_idx = idxer[~hit2]
        if len(miss2_idx):
            idx_c = pd.MultiIndex.from_frame(df.loc[miss2_idx, lookup["keys_coarse"]])
            m3 = lookup["lvl_coarse"].reindex(idx_c)
            hit3 = m3.notna().to_numpy()
            out.loc[miss2_idx[hit3]] = m3.to_numpy()[hit3]
            fallback_used.loc[miss2_idx[hit3]] = "coarse"
            still_miss = miss2_idx[~hit3]
            out.loc[still_miss] = lookup["global_median"]
            fallback_used.loc[still_miss] = "global"

    return out, fallback_used
