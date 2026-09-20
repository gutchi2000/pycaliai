# -*- coding: utf-8 -*-
"""
expected_time_model.py — 距離・競馬場・芝ダ・コース区分・公表馬場状態・クラスで
条件付けた「期待走破タイム」ルックアップ。

【2026-09-20夜、ユーザー指摘により全面改訂】
旧版は train<=2022 全体で一発fitし、それを2013-2022年自身の残差計算にも
そのまま適用していた(2013-2022年内の行にとって部分的にin-sample)。
以下の設計へ訂正する:

  - **expanding-window(年単位leave-year-out)**: 年Yの行の期待値は、Yより前の
    年だけでfitしたモデルから作る。2013-2022年の各年はそれぞれ「その年より前」
    だけでfit(年ごとにモデルが違う、拡大窓)。
  - **2023年以降(development/OOS)は2022年末まででfitした単一モデルに固定**。
    2023・2024・2025のどの年の観測も同じモデルを使い、これらの年の結果を見て
    再fitしない(2024/2025だけでなく2023 developmentの結果でも再fitしない)。
  - **削除不変性**: 未来年のデータを削除しても、過去年の期待値・残差は変わらない
    (expanding-windowなら年Yのモデルは定義上「Yより前」しか見ないため自動的に
    成立するはずだが、`test_build_observations.py`で直接検証する)。
  - **競馬場(場所)を条件付けキーに追加**。EXP08が推定する潜在状態は
    z[開催日,競馬場,芝ダ,時刻]であり、競馬場固有の恒常的な時計の速さ/遅さは
    「日ごとに変動する状態」ではなく「その競馬場の恒常特性」。これを期待値側で
    吸収しておかないと、特定の競馬場が持つ固定バイアスが「毎日その競馬場は
    speed_signalが高い」という形で漏れ込み、日次状態と混同される。

段階的フォールバック(セルのサンプル数がMIN_CELL_N未満の場合に粗いキーへ):
  full(距離,競馬場,芝ダ,コース区分,馬場状態,クラス) →
  no_going(馬場状態を外す) → no_class(クラスも外す、競馬場は残す) →
  coarse(距離,芝ダのみ、競馬場も外す) → global(全体中央値)
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[3]
KEKKA_EXT = "E:/競馬過去走データ/raw_data/kekka_2010_2025_fix_raceid_v2__keyed.csv"

MIN_CELL_N = 30  # このセルの中央値を信頼できる最低サンプル数
FROZEN_CUTOFF_YEAR = 2023  # この年以降は2022年末までのモデルに固定(以降再fitしない)

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
        "yyyymmdd", "race_id", "レース名", "場所", "馬番", "着順", "走破タイム", "距離",
        "芝・ダ", "コース区分", "馬場状態", "クラス名", "多頭出し", "上り3F", "RPCI", "PCI",
    ]
    df = pd.read_csv(KEKKA_EXT, encoding="utf-8-sig", low_memory=False, usecols=cols)
    # 【2026-09-20夜訂正】このファイルには"race_id16"という紛らわしい列も存在するが、
    # master_v2の「レースID(新/馬番無)」とは別スキーマ(uu/pp/yy/k/n/rr起源の内部再採番、
    # 場所コードが一致しない)。実データ突合で発覚: master_v2と結合可能なのは
    # 無印の"race_id"列のみ(venue/kai/day/race番号の構成がmaster_v2と一致することを
    # 実データで確認済み)。以後は必ず"race_id"を使い、"race_id16"には触れない。
    df = df.rename(columns={"race_id": "rid16"})
    # 障害(jump)レースはクラス名が下級クラスで平地と同じラベル("未勝利"等)を
    # 共有するためクラス名だけでは判別不能。レース名の"障害"部分文字列で明示的に除外
    # (混在すると同じ距離bucketの中央値が平地より大幅に遅い障害タイムで歪む)。
    df = df[~df["レース名"].astype(str).str.contains("障害", na=False)]
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
    df = df.dropna(subset=["time_sec", "dist_bucket", "場所", "芝・ダ", "コース区分",
                            "馬場状態", "クラス名"])
    return df


def fit_expected_time_lookup(train_df: pd.DataFrame, value_col: str = "time_sec") -> dict:
    """train_dfのデータから段階的フォールバック付きの期待値lookupを作る。
    value_colを変えるとRPCI等の他の値にも同じ仕組みを流用できる。"""
    keys_full = ["dist_bucket", "場所", "芝・ダ", "コース区分", "馬場状態", "クラス名"]
    keys_no_going = ["dist_bucket", "場所", "芝・ダ", "コース区分", "クラス名"]
    keys_no_class = ["dist_bucket", "場所", "芝・ダ", "コース区分"]
    keys_coarse = ["dist_bucket", "芝・ダ"]

    def build_level(keys):
        g = train_df.groupby(keys)[value_col]
        med = g.median()
        n = g.size()
        med = med[n >= MIN_CELL_N]
        return med

    lvl_full = build_level(keys_full)
    lvl_no_going = build_level(keys_no_going)
    lvl_no_class = build_level(keys_no_class)
    lvl_coarse = build_level(keys_coarse)
    global_median = float(train_df[value_col].median()) if len(train_df) else np.nan

    return {
        "keys_full": keys_full, "keys_no_going": keys_no_going,
        "keys_no_class": keys_no_class, "keys_coarse": keys_coarse,
        "lvl_full": lvl_full, "lvl_no_going": lvl_no_going,
        "lvl_no_class": lvl_no_class, "lvl_coarse": lvl_coarse,
        "global_median": global_median, "n_train_rows": len(train_df),
    }


def predict_expected_time(lookup: dict, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """段階的フォールバックで期待値を予測する(時計・RPCI共通)。"""
    out = pd.Series(np.nan, index=df.index, dtype=float)
    fallback_used = pd.Series("none", index=df.index, dtype=object)

    levels = [
        ("full", lookup["keys_full"], lookup["lvl_full"]),
        ("no_going", lookup["keys_no_going"], lookup["lvl_no_going"]),
        ("no_class", lookup["keys_no_class"], lookup["lvl_no_class"]),
        ("coarse", lookup["keys_coarse"], lookup["lvl_coarse"]),
    ]
    remaining = df.index
    for name, keys, table in levels:
        if len(remaining) == 0:
            break
        sub = df.loc[remaining, keys]
        idx = pd.MultiIndex.from_frame(sub)
        m = table.reindex(idx)
        hit = m.notna().to_numpy()
        hit_idx = remaining[hit]
        out.loc[hit_idx] = m.to_numpy()[hit]
        fallback_used.loc[hit_idx] = name
        remaining = remaining[~hit]
    if len(remaining):
        out.loc[remaining] = lookup["global_median"]
        fallback_used.loc[remaining] = "global"

    return out, fallback_used


def fit_expanding_lookups(df: pd.DataFrame, value_col: str = "time_sec") -> dict:
    """年単位のexpanding-window(leave-year-out)lookup群を作る。

    - year Y < FROZEN_CUTOFF_YEAR(2023): その年より前の全年だけでfitしたモデル
      (年ごとに別モデル、拡大窓)。
    - year Y >= FROZEN_CUTOFF_YEAR: 2022年末まで(年<2023)の全データで一度だけ
      fitした単一モデルを固定して使う(2023・2024・2025は全て同じモデル、
      これらの年の結果を見て再fitしない)。

    戻り値: {year: lookup_dict}。年がdfに存在しない場合はそのキーを作らない。
    """
    years = sorted(df["year"].unique())
    lookups: dict[int, dict] = {}

    frozen_train = df[df["year"] < FROZEN_CUTOFF_YEAR]
    frozen_lookup = fit_expected_time_lookup(frozen_train, value_col=value_col)

    for y in years:
        if y >= FROZEN_CUTOFF_YEAR:
            lookups[y] = frozen_lookup
        else:
            prior = df[df["year"] < y]
            lookups[y] = fit_expected_time_lookup(prior, value_col=value_col)
    return lookups


def predict_expected_time_expanding(
    lookups: dict, df: pd.DataFrame, value_col: str = "time_sec",
) -> tuple[pd.Series, pd.Series]:
    """年ごとに対応するexpanding-window lookupを使って期待値を予測する。"""
    out = pd.Series(np.nan, index=df.index, dtype=float)
    fb = pd.Series("none", index=df.index, dtype=object)
    for y, sub_idx in df.groupby("year").groups.items():
        lookup = lookups.get(y)
        if lookup is None or lookup["n_train_rows"] == 0:
            fb.loc[sub_idx] = "no_prior_data"
            continue
        pred, fb_y = predict_expected_time(lookup, df.loc[sub_idx])
        out.loc[sub_idx] = pred
        fb.loc[sub_idx] = fb_y
    return out, fb
