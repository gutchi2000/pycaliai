# -*- coding: utf-8 -*-
"""
weekly_parse.py — 週次出走表 CSV (data/weekly/YYYYMMDD.csv) の読み込み部 (旧 predict_weekly.py 122〜735 行を無改変で移動)
================================================================================
docs/LEGACY_RETIREMENT_PLAN_20261004.md 段階 1 (2026-11-04)。本番 (export_weekly_marks / build_bet_plan /
build_horse_history) が旧 8 モデルアンサンブル predict_weekly.py から借りていたのは parse_csv と列定義だけで、
予測・買い目部 (ev_filter / 旧モデル pkl) には依存しない。切り出しにより本番の bundle 生成が optuna / lightgbm /
ev_filter の import に依存しなくなる。中身は 1 行も変えていない (import 先の調整のみ)。受け入れ検査:
  python -m analysis.legacy_parse_baseline check --out <dump dir>   # 2026 年全週の parse_csv 出力が切り出し前と同一
predict_weekly.py は後方互換のため同じ名前を再 export する (研究スクリプトは無修正で動く)。
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent


RACE_COLS = [
    "レースID(新)","日付S","曜日","場所","開催","R","レース名","クラス名",
    "芝・ダート","距離","コース区分","コーナー回数","馬場状態(暫定)","天候(暫定)",
    "フルゲート頭数","発走時刻","性別限定","重量種別","年齢限定",
]
HORSE_COLS_33 = [
    "枠番","B","馬番","馬名S","性別","年齢","人気_今走","単勝","ZI印","ZI","ZI順",
    "斤量","減M","替","騎手","所属","調教師","父","母父","父タイプ","母父タイプ",
    "前走月","前走日","前走場所","前走TD","前走距離","前走馬場状態","前走着順",
    "前走人気","前走レース名","前走上り3F","前走決手","前走間隔",
]
HORSE_COLS_46 = [
    "枠番","B","馬番","馬名S","性別","年齢","人気_今走","単勝","ZI印","ZI","ZI順",
    "斤量","減M","替","騎手","所属","調教師","父","母父","父タイプ","母父タイプ",
    "前走月","前走日","前走開催","前走間隔","前走レース名","前走TD","前走距離","前走馬場状態",
    "前走B","前走騎手","前走斤量","前走減","前走人気","前走単勝オッズ","前走着順","前走着差",
    "マイニング順位","前走通過1","前走通過2","前走通過3","前走通過4","前走Ave3F",
    "前走上り3F","前走上り3F順位","前走1_2着馬",
]
HORSE_COLS_48 = [
    "枠番","B","馬番","馬名S","性別","年齢","人気_今走","単勝","ZI印","ZI","ZI順",
    "斤量","減M","替","騎手","所属","調教師","父","母父","父タイプ","母父タイプ",
    "前走月","前走日","前走開催","前走間隔","前走レース名","前走TD","前走距離","前走馬場状態",
    "前走B","前走騎手","前走斤量","前走減","前走人気","前走単勝オッズ","前走着順","前走着差",
    "マイニング順位","前走通過1","前走通過2","前走通過3","前走通過4","前走Ave3F",
    "前走上り3F","前走上り3F順位","前走1_2着馬",
    "騎手コード","調教師コード",   # 48列形式: 末尾2列にコード追加
]
HORSE_COLS_49 = [
    "枠番","B","馬番","馬名S","性別","年齢","馬体重","馬体重増減_raw","馬体重増減",
    "人気_今走","単勝","ZI印","ZI","ZI順","斤量","減M","替","騎手","所属","調教師",
    "父","母父","父タイプ","母父タイプ",
    "前走月","前走日","前走開催","前走間隔","前走レース名","前走TD","前走距離","前走馬場状態",
    "前走B","前走騎手","前走斤量","前走減","前走人気","前走単勝オッズ","前走着順","前走着差",
    "マイニング順位","前走通過1","前走通過2","前走通過3","前走通過4","前走Ave3F",
    "前走上り3F","前走上り3F順位","前走1_2着馬",
]
# 46列版から「人気_今走」「単勝」(今走オッズ/人気) の2列を欠いた版。
# 2026-09-21 発見: TARGET が開催によってこの2列を出力しないケースがある
# (該当開催の今走オッズ/人気が export 時点で未確定?)。旧コードは幅不一致で
# 該当レースを丸ごと無言破棄しており、開催が1つだけ残るせいで
# export_weekly_marks.py の 場所/開催 定数カナリアが誤発火していた。
HORSE_COLS_44 = [
    "枠番","B","馬番","馬名S","性別","年齢","ZI印","ZI","ZI順",
    "斤量","減M","替","騎手","所属","調教師","父","母父","父タイプ","母父タイプ",
    "前走月","前走日","前走開催","前走間隔","前走レース名","前走TD","前走距離","前走馬場状態",
    "前走B","前走騎手","前走斤量","前走減","前走人気","前走単勝オッズ","前走着順","前走着差",
    "マイニング順位","前走通過1","前走通過2","前走通過3","前走通過4","前走Ave3F",
    "前走上り3F","前走上り3F順位","前走1_2着馬",
]
HORSE_COLS_99 = [
    "枠番","B","馬番","馬名S","性別","年齢","馬体重","馬体重増減_raw","馬体重増減",
    "人気_今走","単勝","ZI印","ZI","ZI順","斤量","減M","替","騎手","所属","調教師",
    "父","母父","父タイプ","母父タイプ",
    # 1走前
    "前走月","前走日","前走開催","前走間隔","前走レース名","前走TD","前走距離","前走馬場状態",
    "前走B","前走騎手","前走斤量","前走減","前走人気","前走単勝オッズ","前走着順","前走着差",
    "マイニング順位","前走通過1","前走通過2","前走通過3","前走通過4","前走Ave3F",
    "前走上り3F","前走上り3F順位","前走1_2着馬",
    # 2走前
    "二走前月","二走前日","二走前開催","二走前間隔","二走前レース名","二走前TD","二走前距離","二走前馬場状態",
    "二走前B","二走前騎手","二走前斤量","二走前減","二走前人気","二走前単勝オッズ","二走前着順","二走前着差",
    "二走前マイニング順位","二走前通過1","二走前通過2","二走前通過3","二走前通過4","二走前Ave3F",
    "二走前上り3F","二走前上り3F順位","二走前1_2着馬",
    # 3走前
    "三走前月","三走前日","三走前開催","三走前間隔","三走前レース名","三走前TD","三走前距離","三走前馬場状態",
    "三走前B","三走前騎手","三走前斤量","三走前減","三走前人気","三走前単勝オッズ","三走前着順","三走前着差",
    "三走前マイニング順位","三走前通過1","三走前通過2","三走前通過3","三走前通過4","三走前Ave3F",
    "三走前上り3F","三走前上り3F順位","三走前1_2着馬",
]
COLUMN_MAP = {
    "馬名S":         "馬名",
    "芝・ダート":     "芝・ダ",
    "馬場状態(暫定)": "馬場状態",
    "天候(暫定)":    "天気",
    "人気_今走":     "人気",
    "ZI順":         "ZI順位",
    "父":            "種牡馬",
    "母父":          "母父馬",
    "父タイプ":       "父タイプ名",
    "母父タイプ":     "母父タイプ名",
    "前走着順":       "前走確定着順",
    "前走上り3F":     "前走上り3F",
    "前走TD":        "前芝・ダ",
    "前走間隔":       "間隔",
    "前走着差":       "前走着差タイム",
    "前走斤量":       "前走斤量",
    # master CSV の列名に合わせる（モデルが参照する名前）
    "前走Ave3F":     "前走Ave-3F",
    "前走上り3F順位":  "前走上り3F順",
    "前走通過1":      "前1角",
    "前走通過2":      "前2角",
    "前走通過3":      "前3角",
    "前走通過4":      "前4角",
    "マイニング順位":  "マイニング順位",
    "前走単勝オッズ":  "前走単勝オッズ",
    # 前走距離 → 前距離（モデルが使う列名）
    "前走距離":       "前距離",
}
CLASS_NORMALIZE = {
    "新馬":"新馬","未勝利":"未勝利","1勝":"1勝","500万":"1勝",
    "2勝":"2勝","1000万":"2勝","3勝":"3勝","1600万":"3勝",
    "OP(L)":"OP(L)","Ｇ１":"Ｇ１","Ｇ２":"Ｇ２","Ｇ３":"Ｇ３",
}


# =========================================================
# 着度数CSV パース
# =========================================================
TYAKU_DIR  = BASE_DIR / "data" / "tyaku"
HOSSEI_DIR = BASE_DIR / "data" / "hosei"
KAKO5_DIR  = BASE_DIR / "data" / "kako5"

TYAKU_HORSE_COLS = [
    "枠番","B","馬番","印","M2","M3","M4","馬名S","C","性別","年齢","替","騎手","斤量","減M","単勝",
    "馬体重","増減",
    "中央平地全:1着","中央平地全:2着","中央平地全:3着","中央平地全:外","中央平地全:連対率",
    "同騎手:1着","同騎手:2着","同騎手:3着","同騎手:外","同騎手:連対率",
    "全ダート:1着","全ダート:2着","全ダート:3着","全ダート:外","全ダート:連対率",
    "同距離:1着","同距離:2着","同距離:3着","同距離:外","同距離:連対率",
    "同場所:1着","同場所:2着","同場所:3着","同場所:外","同場所:連対率",
    "同コース:1着","同コース:2着","同コース:3着","同コース:外","同コース:連対率",
    "同クラス:1着","同クラス:2着","同クラス:3着","同クラス:外","同クラス:連対率",
    "穴傾向","BESTタイム",
]
# TARGET の着度数出力には 3 形式ある。
# 55 列版だけが「馬体重」「増減」を持つ。53 列版は両列なし、
# 52 列版はさらにレース単位で「単勝」列が省略される。
# 同じ列名リストへ無理に zip すると中央平地成績以降がずれるため、
# 実ヘッダーに合わせた別スキーマとして扱う。
TYAKU_HORSE_COLS_53 = [
    c for c in TYAKU_HORSE_COLS if c not in {"馬体重", "増減"}
]
TYAKU_HORSE_COLS_52 = [c for c in TYAKU_HORSE_COLS_53 if c != "単勝"]
TYAKU_HORSE_SCHEMAS = {
    52: TYAKU_HORSE_COLS_52,
    53: TYAKU_HORSE_COLS_53,
    55: TYAKU_HORSE_COLS,
}



def _load_tyaku(date_str: str) -> pd.DataFrame | None:
    """data/tyaku/YYYYMMDD.csv を読み込み、馬番→着度数の対応表を返す。
    ファイルが存在しない場合は None を返す。"""
    path = TYAKU_DIR / f"{date_str}.csv"
    if not path.exists():
        return None

    for enc in ["cp932", "shift_jis", "utf-8"]:
        try:
            text = path.read_bytes().decode(enc)
            break
        except Exception:
            continue
    else:
        return None

    rows: list[dict] = []
    current_race_id: str | None = None
    horse_like_rows = 0
    unsupported_widths: dict[int, int] = {}

    for line in text.splitlines():
        cols = line.split(",")
        if len(cols) == 19 and cols[0] not in ("レースID(新)", ""):
            current_race_id = cols[0].strip()[:16]  # 16桁に切る
        elif current_race_id and cols[0].strip().isdigit():
            horse_like_rows += 1
            schema = TYAKU_HORSE_SCHEMAS.get(len(cols))
            if schema is None:
                unsupported_widths[len(cols)] = unsupported_widths.get(len(cols), 0) + 1
                continue
            row = dict(zip(schema, cols))
            row["レースID(新/馬番無)"] = current_race_id
            rows.append(row)

    if not rows:
        logger.warning(
            f"着度数CSVをパースできません: {path.name} "
            f"(馬行候補={horse_like_rows}, 未対応列数={unsupported_widths or 'なし'})"
        )
        return None
    if unsupported_widths or (horse_like_rows and len(rows) < horse_like_rows * 0.5):
        logger.warning(
            f"着度数CSVの一部行を無視: {path.name} "
            f"(取得={len(rows)}/{horse_like_rows}, 未対応列数={unsupported_widths or 'なし'})"
        )

    df = pd.DataFrame(rows)
    # 数値変換
    for col in ["馬番","馬体重",
                "中央平地全:1着","中央平地全:2着","中央平地全:3着","中央平地全:外",
                "同コース:1着","同コース:2着","同コース:3着","同コース:外",
                "同クラス:1着","同クラス:2着","同クラス:3着","同クラス:外"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 馬ごとの複勝率を計算（ベイズ平滑化: 少走経験馬でも 0 にならないよう事前分布で補正）
    # prior = 訓練valid中央値 0.286, 仮想サンプル数 = 5
    # smoothed = (着内回数 + 1.43) / (総走数 + 5.0)
    # → 0走: 1.43/5.00 = 0.286, 1走0着: 1.43/6.00 = 0.238, 10走3着内: 4.43/15.0 = 0.295
    _PRIOR_ALPHA = 1.43   # 0.286 × 5
    _PRIOR_BETA  = 5.0
    for prefix, out_col in [("中央平地全", "horse_fuku_career"),
                             ("同コース",   "horse_fuku_course"),
                             ("同クラス",   "horse_fuku_class")]:
        w = df[f"{prefix}:1着"].fillna(0) + df[f"{prefix}:2着"].fillna(0) + df[f"{prefix}:3着"].fillna(0)
        total = w + df[f"{prefix}:外"].fillna(0)
        df[out_col] = (w + _PRIOR_ALPHA) / (total + _PRIOR_BETA)

    # 馬体重の増減を数値化（"- 8" → -8, "+ 2" → 2）
    if "増減" in df.columns:
        df["増減"] = df["増減"].astype(str).str.replace(" ", "").str.replace("－","-").str.replace("＋","+")
        df["増減"] = pd.to_numeric(df["増減"], errors="coerce")

    keep = ["レースID(新/馬番無)","馬番","馬体重","増減",
            "horse_fuku_career","horse_fuku_course","horse_fuku_class"]
    return df[[c for c in keep if c in df.columns]]


def _load_kako5_warnings(date_str: str) -> dict[str, str]:
    """kako5/YYYYMMDD.csv を読み、過去5走に中止(止)・除外(外)・取消(消)がある馬名→警告文 を返す。

    警告例: {"アドマイヤテラ": "2走前中止", "○○": "1走前除外・3走前中止"}
    ファイルが存在しない場合は空dictを返す。
    """
    # 日付変換: "2026.3.8" → "20260308"
    try:
        from datetime import datetime
        d = datetime.strptime(date_str.strip(), "%Y.%m.%d") if "-" not in date_str else \
            datetime.strptime(date_str.strip(), "%Y-%m-%d")
    except ValueError:
        # "2026.3.8" のようにゼロ埋めなし
        parts = date_str.strip().replace("-", ".").split(".")
        d = datetime(int(parts[0]), int(parts[1]), int(parts[2]))

    kako5_path = KAKO5_DIR / f"{d.strftime('%Y%m%d')}.csv"
    if not kako5_path.exists():
        return {}

    STOP_CODES = {"止", "外", "消"}
    CODE_LABEL = {"止": "中止", "外": "除外", "消": "取消"}
    # 着順の列インデックス（1走前=18, 2走前=30, ..., 5走前=66）
    CHAKU_IDXS = {1: 18, 2: 30, 3: 42, 4: 54, 5: 66}
    UMANAME_IDX = 7  # 馬名S

    warnings: dict[str, str] = {}
    in_horse_section = False

    try:
        with open(kako5_path, encoding="cp932", errors="replace") as f:
            for line in f:
                parts = line.rstrip("\n").split(",")
                # 馬ヘッダー行を検出
                if parts and parts[0].strip() == "枠番":
                    in_horse_section = True
                    continue
                # レース情報行（先頭がレースIDっぽい数字16桁）はスキップ
                if not in_horse_section:
                    continue
                if len(parts) < 20:
                    continue
                # 馬名を取得
                umaname = parts[UMANAME_IDX].strip() if len(parts) > UMANAME_IDX else ""
                if not umaname:
                    continue
                # 過去5走の着順チェック
                warns = []
                for n, idx in CHAKU_IDXS.items():
                    if idx >= len(parts):
                        break
                    val = parts[idx].strip()
                    if val in STOP_CODES:
                        warns.append(f"{n}走前{CODE_LABEL[val]}")
                if warns:
                    warnings[umaname] = "・".join(warns)
    except Exception as e:
        logger.warning(f"kako5警告読み込み失敗: {e}")

    return warnings


def _load_hosei(date_str: str) -> pd.DataFrame | None:
    """data/hosei/H_*.csv を全て結合してレースID×馬番→補正タイムの対応表を返す。
    毎週 H_YYYYMMDD-YYYYMMDD.csv 形式でフォルダに追加するだけで自動取り込みされる。
    """
    hosei_files = sorted(HOSSEI_DIR.glob("H_*.csv"))
    if not hosei_files:
        return None
    dfs = []
    for path in hosei_files:
        for enc in ["cp932", "utf-8-sig", "utf-8"]:
            try:
                tmp = pd.read_csv(path, encoding=enc,
                                  usecols=["レースID(新)", "前走補9", "前走補正"],
                                  dtype={"レースID(新)": str})
                dfs.append(tmp)
                break
            except Exception:
                continue
    if not dfs:
        return None
    df = pd.concat(dfs, ignore_index=True).drop_duplicates()
    df["レースID(新)"] = df["レースID(新)"].astype(str)
    for col in ["前走補9", "前走補正"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["レースID(新)", "前走補9", "前走補正"]]


# =========================================================
# CSV パース
# =========================================================
def parse_csv(path: Path) -> pd.DataFrame:
    """ターゲット形式CSV（33列 or 46列）をDataFrameに変換する。"""
    for enc in ["cp932", "shift_jis", "utf-8"]:
        try:
            text = path.read_bytes().decode(enc)
            break
        except Exception:
            continue

    races: list[dict] = []
    current_race: dict | None = None
    _unmatched_widths: dict[int, int] = {}

    for line in text.splitlines():
        cols = line.split(",")
        if cols[0] in ("レースID(新)", "枠番", "番", ""):
            continue
        if len(cols) == 19:
            current_race = dict(zip(RACE_COLS, cols))
        elif len(cols) == 33 and current_race:
            horse = dict(zip(HORSE_COLS_33, cols))
            horse.update(current_race)
            races.append(horse)
        elif len(cols) == 44 and current_race:
            horse = dict(zip(HORSE_COLS_44, cols))
            horse.update(current_race)
            races.append(horse)
        elif len(cols) == 46 and current_race:
            horse = dict(zip(HORSE_COLS_46, cols))
            horse.update(current_race)
            races.append(horse)
        elif len(cols) == 48 and current_race:
            horse = dict(zip(HORSE_COLS_48, cols))
            horse.update(current_race)
            races.append(horse)
        elif len(cols) == 49 and current_race:
            horse = dict(zip(HORSE_COLS_49, cols))
            horse.update(current_race)
            races.append(horse)
        elif len(cols) == 99 and current_race:
            horse = dict(zip(HORSE_COLS_99, cols))
            horse.update(current_race)
            races.append(horse)
        elif current_race and len(cols) > 1:
            _unmatched_widths[len(cols)] = _unmatched_widths.get(len(cols), 0) + 1

    if _unmatched_widths:
        logger.warning(
            f"parse_csv: 未対応の列幅で無言破棄した行あり: {_unmatched_widths} "
            f"({path.name}) → HORSE_COLS_* に新schema追加が必要な可能性")

    df = pd.DataFrame(races).rename(columns=COLUMN_MAP)
    df["レースID(新/馬番無)"] = df["レースID(新)"].astype(str).str[:16]

    for col in ["枠番","馬番","斤量","ZI","ZI順位","距離","人気","単勝",
                "前走確定着順","前走上り3F","前走距離","間隔","前走人気",
                "前走着差タイム","前走斤量","前走Ave-3F","前走上り3F順",
                "マイニング順位","前走単勝オッズ",
                "前1角","前2角","前3角","前4角",
                "フルゲート頭数","年齢","出走頭数","コーナー回数",
                # COLUMN_MAP で rename された後の列名
                "前距離","前走馬体重","前走馬体重増減"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 出走頭数：週次CSVは直接持たないため、レースIDごとの馬数から算出
    if "出走頭数" not in df.columns:
        df["出走頭数"] = df.groupby("レースID(新/馬番無)")["馬番"].transform("count")
    df["出走頭数"] = pd.to_numeric(df["出走頭数"], errors="coerce").fillna(
        pd.to_numeric(df.get("フルゲート頭数"), errors="coerce")
    )

    # 脚質特徴量：前走コーナー通過順位から計算（モデルと同じ定義）
    if "前1角" in df.columns and "前4角" in df.columns:
        n = df["出走頭数"].clip(lower=2)
        df["prev_pos_rel"]   = (df["前1角"] - 1) / (n - 1)
        df["closing_power"]  = (df["前1角"] - df["前4角"]) / (n - 1)

    # 騎手・調教師ローリング成績（週次CSVに騎手コードなし → 訓練データ中央値で補完）
    # 訓練 valid 中央値: jockey_fuku30=0.200, jockey_fuku90=0.200,
    #                    trainer_fuku30=0.200, trainer_fuku90=0.211
    _ROLLING_TRAIN_MEDIANS = {
        "jockey_fuku30": 0.200,
        "jockey_fuku90": 0.200,
        "trainer_fuku30": 0.200,
        "trainer_fuku90": 0.211,
    }
    for fname, code_col, stat_cols in [
        ("jockey_stats.csv",  "騎手コード",  ["jockey_fuku30", "jockey_fuku90"]),
        ("trainer_stats.csv", "調教師コード", ["trainer_fuku30", "trainer_fuku90"]),
    ]:
        stats_path = BASE_DIR / "data" / fname
        if stats_path.exists():
            stats = pd.read_csv(stats_path, encoding="utf-8-sig")
            if code_col in df.columns:
                # "01122" 形式（先頭ゼロ付き文字列）→ int変換（1122）してマージ
                df[code_col] = pd.to_numeric(df[code_col], errors="coerce")
                df = df.merge(stats[[code_col] + stat_cols], on=code_col, how="left")
                for col in stat_cols:
                    if col in df.columns:
                        df[col] = df[col].fillna(_ROLLING_TRAIN_MEDIANS.get(col, 0.200))
            else:
                for col in stat_cols:
                    df[col] = _ROLLING_TRAIN_MEDIANS.get(col, 0.200)
        else:
            for col in stat_cols:
                df[col] = _ROLLING_TRAIN_MEDIANS.get(col, 0.200)

    # ── 着度数CSV（data/tyaku/YYYYMMDD.csv）があればマージ ──
    date_str = path.stem   # ファイル名の日付部分（例: "20260308"）
    tyaku_df = _load_tyaku(date_str)
    if tyaku_df is not None:
        df = df.merge(tyaku_df, on=["レースID(新/馬番無)", "馬番"], how="left",
                      suffixes=("", "_tyaku"))
        # 馬体重: tyaku が優先（週次CSVに馬体重がある場合は上書き）
        if "馬体重_tyaku" in df.columns:
            df["馬体重"] = df["馬体重_tyaku"].combine_first(
                pd.to_numeric(df.get("馬体重"), errors="coerce"))
            df.drop(columns=["馬体重_tyaku"], inplace=True)
        if "増減_tyaku" in df.columns:
            df["馬体重増減"] = df["増減_tyaku"]
            df.drop(columns=["増減_tyaku"], inplace=True)
        # horse_fuku: tyaku CSV 由来 (horse_fuku_career_tyaku) を優先、
        #             なければ週次CSV由来 (horse_fuku_career) を使用。
        #             app.py も tyaku CSV を優先するので両者の予測を一致させる。
        if "horse_fuku_career_tyaku" in df.columns:
            fuku_src = df["horse_fuku_career_tyaku"].combine_first(df.get("horse_fuku_career", pd.Series(dtype=float)))
            df.drop(columns=["horse_fuku_career_tyaku"], inplace=True, errors="ignore")
        elif "horse_fuku_career" in df.columns:
            fuku_src = df["horse_fuku_career"]
        else:
            fuku_src = None
        if fuku_src is not None:
            df["horse_fuku10"] = fuku_src.fillna(0.286)
            df["horse_fuku30"] = fuku_src.fillna(0.312)
            logger.info(
                f"着度数CSV読み込み済: horse_fuku mean={df['horse_fuku10'].mean():.3f}, "
                f"std={df['horse_fuku10'].std():.3f}"
            )
        else:
            df["horse_fuku10"] = 0.286
            df["horse_fuku30"] = 0.312
    else:
        # 着度数CSVなし → 訓練データ中央値で補完
        df["horse_fuku10"] = 0.286
        df["horse_fuku30"] = 0.312

    # 週次CSVに含まれないペース指数・体重比を訓練データ中央値で補完
    # （0 のままだと分布外でモデルが誤った方向に流れる）
    _PACE_MEDIANS = {
        "前PCI": 49.0,
        "前走PCI3": 50.2,
        "前走RPCI": 48.5,
        "前走平均1Fタイム": 12.26,
    }
    for col, med in _PACE_MEDIANS.items():
        if col not in df.columns:
            df[col] = med

    # 斤量体重比 = 斤量 / 馬体重 × 100（学習側スケール ≈ 11.7。×100 を欠くと
    # 馬体重が入る週に 0.117 vs 11.7 の分布外に飛ぶ。馬体重 0/欠損は訓練中央値補完）
    if "斤量体重比" not in df.columns:
        wt = pd.to_numeric(df["馬体重"], errors="coerce").replace(0, np.nan) if "馬体重" in df.columns else np.nan
        jk = pd.to_numeric(df["斤量"], errors="coerce") if "斤量" in df.columns else np.nan
        if isinstance(wt, pd.Series) and isinstance(jk, pd.Series):
            df["斤量体重比"] = (jk / wt * 100).fillna(11.8)
        else:
            df["斤量体重比"] = 11.8

    # 訓練データに含まれるが週次CSVにない特徴量を中央値で補完
    # （0 のままだと out-of-distribution でモデルが誤判断する）
    _MISSING_FEATURE_MEDIANS = {
        "馬齢斤量差":            -1,    # 訓練 valid 中央値
        "トラックコード(JV)":      23,   # 訓練 valid 中央値（芝・内=23）
        "前走トラックコード(JV)":  23,   # 訓練 valid 中央値
        "前走競走種別":           13,   # 訓練 valid 中央値
        "前走出走頭数":           15,   # 訓練 valid 中央値
        "前走馬体重":            472,   # 訓練 valid 中央値
        "前走馬体重増減":           0,   # 訓練 valid 中央値（増減ゼロが最多）
        "騎手年齢":              30,   # 訓練 valid 中央値
        "調教師年齢":             53,   # 訓練 valid 中央値
        "休み明け～戦目":           2,   # 訓練 valid 中央値
    }
    for col, med in _MISSING_FEATURE_MEDIANS.items():
        if col not in df.columns:
            df[col] = med

    for col in ["馬体重","馬体重増減","前走斤量","生産者"]:
        if col not in df.columns:
            df[col] = 0
    for col in ["前走走破タイム","前走着差タイム"]:
        if col not in df.columns:
            df[col] = float("nan")  # LGBMのNaN処理に任せる（0だと分布が大きく外れる）

    # ── 補正タイムCSV（data/hosei/H_*.csv）があればマージ ──
    hosei_df = _load_hosei(date_str)
    if hosei_df is not None:
        # hosei は18桁ID（レース16桁+馬番2桁）なので一致させてマージ
        df["_hosei_key"] = (
            df["レースID(新/馬番無)"].astype(str)
            + df["馬番"].astype(int).astype(str).str.zfill(2)
        )
        hosei_df = hosei_df.rename(columns={"レースID(新)": "_hosei_key"})
        df = df.merge(hosei_df, on="_hosei_key", how="left")
        df = df.drop(columns=["_hosei_key"])
        logger.info(
            f"hosei CSV読み込み済: 前走補9 mean={df['前走補9'].mean():.1f}, "
            f"カバレッジ={df['前走補9'].notna().mean()*100:.1f}%"
        )
    else:
        df["前走補9"]  = float("nan")
        df["前走補正"] = float("nan")

    # ── 過去5走特徴量（data/kako5/YYYYMMDD.csv）があればマージ ──
    kako5_path = KAKO5_DIR / f"{date_str}.csv"
    if kako5_path.exists():
        try:
            from parse_kako5 import build_from_kako5, KAKO5_COLS
            kako5_df = build_from_kako5(kako5_path)
            if not kako5_df.empty:
                # kako5_df: [レースID(新), 馬番, ...kako5特徴量]
                # df: レースID(新/馬番無) = 16桁, 馬番 = int
                # kako5: レースID(新) = 18桁(レース16+馬番2)
                # kako5のレースID(新)は18桁なので16桁に変換
                if kako5_df["レースID(新)"].astype(str).str.len().mode().iloc[0] > 16:
                    kako5_df["レースID(新)"] = kako5_df["レースID(新)"].astype(str).str[:16]
                df = df.merge(
                    kako5_df.rename(columns={"レースID(新)": "レースID(新/馬番無)"}),
                    on=["レースID(新/馬番無)", "馬番"],
                    how="left",
                    suffixes=("", "_kako5"),
                )
                valid_pct = df["kako5_avg_pos"].notna().mean() * 100
                logger.info(f"kako5 CSV読み込み済: カバレッジ={valid_pct:.1f}%")
        except Exception as e:
            logger.warning(f"kako5マージ失敗: {e}")
            for col in ["kako5_avg_pos", "kako5_std_pos", "kako5_best_pos",
                         "kako5_avg_agari3f", "kako5_best_agari3f",
                         "kako5_same_td_ratio", "kako5_same_dist_ratio", "kako5_same_place_ratio",
                         "kako5_pos_trend", "kako5_race_count"]:
                if col not in df.columns:
                    df[col] = float("nan")
    else:
        logger.info(f"kako5 CSV なし: {kako5_path}")
        for col in ["kako5_avg_pos", "kako5_std_pos", "kako5_best_pos",
                     "kako5_avg_agari3f", "kako5_best_agari3f",
                     "kako5_same_td_ratio", "kako5_same_dist_ratio", "kako5_same_place_ratio",
                     "kako5_pos_trend", "kako5_race_count"]:
            if col not in df.columns:
                df[col] = float("nan")

    # ── 調教データ（坂路・WCマスターCSVからJOIN）──
    try:
        from training_join import load_chukyo, merge_chukyo
        hanro, wc = load_chukyo()
        # merge_chukyo は "日付"(YYYYMMDD) を参照するが週次CSVは "日付S"(YYYY.M.D) なので変換
        if "日付" not in df.columns and "日付S" in df.columns:
            df["日付"] = df["日付S"].apply(
                lambda s: "{:04d}{:02d}{:02d}".format(*[int(x) for x in str(s).split(".")])
            )
        df = merge_chukyo(df, hanro, wc)
    except Exception as e:
        logger.warning(f"調教JOIN失敗（スキップ）: {e}")
        for col in ["trn_hanro_4f","trn_hanro_3f","trn_hanro_2f","trn_hanro_1f",
                    "trn_hanro_lap1","trn_hanro_lap2","trn_hanro_lap3","trn_hanro_lap4",
                    "trn_hanro_days",
                    "trn_wc_5f","trn_wc_4f","trn_wc_3f",
                    "trn_wc_lap1","trn_wc_lap2","trn_wc_lap3",
                    "trn_wc_days"]:
            df[col] = float("nan")

    # 障害レース除外: 距離列に"障"接頭辞が付く旧形式に加え、2026-09-22発覚の
    # 新形式(距離は素の数値、"障害"は芝・ダ列側="障害(直線ダ)"等)にも対応。
    # 学習データ(master_v2)に障害カテゴリが存在せず encoder が全欠損化するため、
    # 距離側だけの判定では新形式の障害レースが無言でモデルに混入していた。
    _is_hurdle = df["距離"].astype(str).str.contains("障", na=False)
    if "芝・ダ" in df.columns:
        _is_hurdle = _is_hurdle | df["芝・ダ"].astype(str).str.contains("障害", na=False)
    df = df[~_is_hurdle].copy()

    if "前走単勝オッズ" not in df.columns:
        df["前走単勝オッズ"] = float("nan")

    # ── 出走馬分析（bunseki, 2026-09〜）があれば serve 死んでいた特徴を回収 ──
    # data/bunseki/{date_str}.csv が無い週は fail-open で何も変わらない。
    try:
        from parse_bunseki import apply_bunseki
        df = apply_bunseki(df, date_str)
    except Exception as e:
        logger.warning(f"bunseki 適用失敗（スキップ、既存フォールバックのまま）: {e}")

    logger.info(f"パース完了（障害除外済）: {len(df)}頭 / {df['レースID(新/馬番無)'].nunique()}レース")
    return df
