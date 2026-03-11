"""
build_dataset.py
PyCaLiAI - マスターデータセット構築スクリプト

lgbm / cat / torch_transformer / add の4CSVをJOINし、
ターゲット列（複勝フラグ）を生成してmaster_dataset.csvを出力する。

Usage:
    python build_dataset.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

# =========================================================
# パス設定
# =========================================================
BASE_DIR        = Path(r"E:\PyCaLiAI")
DATA_DIR        = BASE_DIR / "data"
REPORT_DIR      = BASE_DIR / "reports"

CSV_LGBM        = DATA_DIR / "lgbm_20130105-20251228.csv"
CSV_CAT         = DATA_DIR / "cat_20130105-20251228.csv"
CSV_TORCH       = DATA_DIR / "torch_transformer_20130105-20251228.csv"
CSV_ADD         = DATA_DIR / "add_20130105-20251228.csv"
MASTER_CSV      = DATA_DIR / "master_20130105-20251228.csv"
KEKKA_CSV       = DATA_DIR / "kekka_20130105-20251228.csv"

JOIN_KEY        = ["レースID(新)", "馬番"]
COL_FINISH      = "着順"          # 全角文字列 → 数値変換する
TARGET          = "fukusho_flag"  # 3着以内=1
COL_DATE        = "日付"          # YYYYMMDD整数 or YYMMDD整数

# 時系列分割（日付整数）
TRAIN_END       = 20221231
VALID_END       = 20231231

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# =========================================================
# ユーティリティ
# =========================================================
def load_csv(path: Path, name: str) -> pd.DataFrame:
    """CSVをUTF-8 → CP932フォールバックで読み込む。"""
    logger.info(f"読み込み: {path.name}")
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="cp932", low_memory=False)
    logger.info(f"  [{name}] {len(df):,}行 × {len(df.columns)}列")
    return df


def normalize_date(series: pd.Series) -> pd.Series:
    """
    日付列を8桁整数（YYYYMMDD）に統一する。
    251228（6桁）→ 20251228（8桁）
    """
    s = series.astype(str).str.strip()
    mask6 = s.str.len() == 6
    s[mask6] = "20" + s[mask6]
    return s.astype(int)


def convert_finish(series: pd.Series) -> pd.Series:
    """
    着順を全角文字列→数値に変換する。
    除外・中止・失格等は NaN にする。
    例: '１' → 1, '１２' → 12, '除' → NaN
    """
    # 全角数字 → 半角
    zen2han = str.maketrans("０１２３４５６７８９", "0123456789")
    converted = series.astype(str).str.translate(zen2han)
    return pd.to_numeric(converted, errors="coerce")


# =========================================================
# 特徴量エンジニアリング
# =========================================================
def add_rolling_stats(master: pd.DataFrame) -> pd.DataFrame:
    """騎手・調教師の直近N走複勝率を時系列リークなしで追加する。

    shift(1) を使うことで「当該レースより前のレース」のみを参照し、
    データリークを防ぐ。
    追加列: jockey_fuku30, jockey_fuku90, trainer_fuku30, trainer_fuku90
    """
    logger.info("騎手・調教師ローリング成績を計算中...")
    # 日付・発走時刻・レース内馬番 の順でソートして時系列を正確に保つ
    master = master.sort_values(
        ["日付", "発走時刻", "レースID(新/馬番無)", "馬番"]
    ).reset_index(drop=True)

    for code_col, prefix in [("騎手コード", "jockey"), ("調教師コード", "trainer")]:
        for window in [30, 90]:
            col = f"{prefix}_fuku{window}"
            master[col] = (
                master.groupby(code_col, sort=False)["fukusho_flag"]
                .transform(
                    lambda x: x.shift(1).rolling(window, min_periods=5).mean()
                )
            )
            na_cnt = master[col].isna().sum()
            logger.info(f"  {col}: NaN={na_cnt:,}件（キャリア浅い等）")

    return master


def add_horse_rolling_stats(master: pd.DataFrame) -> pd.DataFrame:
    """馬の直近N走複勝率を時系列リークなしで追加する。

    追加列: horse_fuku10, horse_fuku30
    """
    logger.info("馬ローリング成績を計算中...")
    for window, min_p in [(10, 3), (30, 5)]:
        col = f"horse_fuku{window}"
        master[col] = (
            master.groupby("血統登録番号", sort=False)["fukusho_flag"]
            .transform(
                lambda x: x.shift(1).rolling(window, min_periods=min_p).mean()
            )
        )
        na_cnt = master[col].isna().sum()
        logger.info(f"  {col}: NaN={na_cnt:,}件")
    return master


def add_pace_features(master: pd.DataFrame) -> pd.DataFrame:
    """前1角〜前4角から脚質指標を生成する。

    prev_pos_rel : 前走の相対的な先行度 (0=先頭, 1=最後方)
    closing_power: 前走の追込み度 (正=前から来た, 負=後退)
    """
    logger.info("脚質特徴量を計算中...")
    n = pd.to_numeric(master["前走出走頭数"], errors="coerce").clip(lower=2)
    front = pd.to_numeric(master["前1角"], errors="coerce")
    back  = pd.to_numeric(master["前4角"], errors="coerce")
    master["prev_pos_rel"]  = (front - 1) / (n - 1)
    master["closing_power"] = (front - back) / (n - 1)
    na_rate = master["prev_pos_rel"].isna().mean()
    logger.info(f"  prev_pos_rel: NaN={na_rate:.1%}")
    return master


def save_entity_stats(master: pd.DataFrame) -> None:
    """最新の騎手・調教師ローリング成績を保存する（週次予測参照用スナップショット）。"""
    for code_col, stat_cols, fname in [
        ("騎手コード",  ["jockey_fuku30", "jockey_fuku90"],   "jockey_stats.csv"),
        ("調教師コード", ["trainer_fuku30", "trainer_fuku90"], "trainer_stats.csv"),
    ]:
        available = [c for c in stat_cols if c in master.columns]
        if not available:
            continue
        latest = (
            master.sort_values(["日付", "発走時刻"])
            .groupby(code_col)[available]
            .last()
            .reset_index()
            .dropna(subset=available, how="all")
        )
        out = DATA_DIR / fname
        latest.to_csv(out, index=False, encoding="utf-8-sig")
        logger.info(f"  保存: {out.name} ({len(latest):,}件)")


def add_roi_target(master: pd.DataFrame) -> pd.DataFrame:
    """kekka CSV の複勝配当から roi_target を生成する。

    roi_target = 複勝配当 / 100  (3着以内の場合)
               = 0               (圏外の場合)
    kekka は3着以内のみ収録のため、LEFT JOIN で自然に圏外=0 になる。
    """
    logger.info(f"kekka CSV 読み込み: {KEKKA_CSV.name}")
    try:
        kekka = pd.read_csv(KEKKA_CSV, encoding="utf-8-sig", low_memory=False)
    except UnicodeDecodeError:
        kekka = pd.read_csv(KEKKA_CSV, encoding="cp932", low_memory=False)

    kekka = kekka[["レースID(新)", "複勝配当"]].copy()
    kekka["複勝配当"] = pd.to_numeric(kekka["複勝配当"], errors="coerce")

    master = master.merge(kekka, on="レースID(新)", how="left")
    master["roi_target"] = (master["複勝配当"].fillna(0) / 100).round(3)
    master = master.drop(columns=["複勝配当"])

    pos_mean = master.loc[master["fukusho_flag"] == 1, "roi_target"].mean()
    logger.info(
        f"roi_target 生成完了: 3着以内平均={pos_mean:.3f}倍 "
        f"(全体mean={master['roi_target'].mean():.3f})"
    )
    return master


# =========================================================
# メイン処理
# =========================================================
def build_master() -> pd.DataFrame:

    # ---- 読み込み ----
    df_lgbm  = load_csv(CSV_LGBM,  "lgbm")
    df_cat   = load_csv(CSV_CAT,   "cat")
    df_torch = load_csv(CSV_TORCH, "torch")
    df_add   = load_csv(CSV_ADD,   "add")

    # ---- add CSVの前処理 ----
    # 着順を数値変換
    df_add[COL_FINISH] = convert_finish(df_add[COL_FINISH])
    finish_na = df_add[COL_FINISH].isna().sum()
    if finish_na > 0:
        logger.info(f"  着順変換不能（除外・中止等）: {finish_na:,}件 → NaN")

    # フルゲート頭数の欠損補完（12件）
    median_fg = df_add["フルゲート頭数"].median()
    df_add["フルゲート頭数"] = df_add["フルゲート頭数"].fillna(median_fg)
    logger.info(f"  フルゲート頭数欠損補完: 中央値={median_fg}")

    # ---- lgbm をベースにJOIN ----
    logger.info("JOIN開始: lgbm + cat...")
    master = df_lgbm.merge(df_cat,  on=JOIN_KEY, how="left", suffixes=("", "_cat"))
    logger.info(f"  lgbm+cat: {len(master):,}行 × {len(master.columns)}列")

    logger.info("JOIN: + torch...")
    master = master.merge(df_torch, on=JOIN_KEY, how="left", suffixes=("", "_torch"))
    logger.info(f"  +torch: {len(master):,}行 × {len(master.columns)}列")

    logger.info("JOIN: + add...")
    master = master.merge(df_add,   on=JOIN_KEY, how="left", suffixes=("", "_add"))
    logger.info(f"  +add: {len(master):,}行 × {len(master.columns)}列")

    # ---- 重複列の整理（suffixがついた列を除去）----
    dup_cols = [c for c in master.columns
                if c.endswith("_cat") or c.endswith("_torch") or c.endswith("_add")]
    if dup_cols:
        logger.warning(f"重複列を除去: {dup_cols}")
        master = master.drop(columns=dup_cols)
    logger.info(f"重複列除去後: {len(master.columns)}列")

    # ---- 日付を8桁整数に統一 ----
    master[COL_DATE] = normalize_date(master[COL_DATE])

    # ---- date_dt列（datetime型）を生成 ----
    master["date_dt"] = pd.to_datetime(
        master[COL_DATE].astype(str), format="%Y%m%d"
    )

    # ---- ターゲット生成 ----
    master[TARGET] = (master[COL_FINISH] <= 3).astype("Int8")  # NaN対応
    pos_rate = master[TARGET].mean()
    logger.info(f"ターゲット生成完了: 複勝率={pos_rate:.3f}")

    # ---- 騎手・調教師ローリング成績特徴量を追加 ----
    master = add_rolling_stats(master)

    # ---- 馬ローリング成績特徴量を追加 ----
    master = add_horse_rolling_stats(master)

    # ---- 脚質特徴量を追加 ----
    master = add_pace_features(master)

    # ---- 期待値ターゲット (roi_target) を追加 ----
    master = add_roi_target(master)

    # ---- 時系列分割列を追加 ----
    def assign_split(date_int: int) -> str:
        if date_int <= TRAIN_END:
            return "train"
        elif date_int <= VALID_END:
            return "valid"
        else:
            return "test"

    master["split"] = master[COL_DATE].apply(assign_split)
    split_counts = master["split"].value_counts()
    logger.info(f"分割: train={split_counts.get('train',0):,} / "
                f"valid={split_counts.get('valid',0):,} / "
                f"test={split_counts.get('test',0):,}")

    # ---- 行数チェック ----
    assert len(master) == 631_965, f"行数異常: {len(master):,}"
    logger.info(f"行数確認OK: {len(master):,}行")

    # ---- 除外・中止・失格を除去 ----
    before = len(master)
    master = master.dropna(subset=["着順"])
    logger.info(f"除外・中止除去: {before - len(master):,}件 → {len(master):,}行")

    # ---- 騎手・調教師スタッツスナップショット保存（週次予測用） ----
    logger.info("エンティティスタッツを保存中...")
    save_entity_stats(master)

    # ---- 保存 ----
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    master.to_csv(MASTER_CSV, index=False, encoding="utf-8-sig")
    logger.info(f"保存完了: {MASTER_CSV}")

    return master


# =========================================================
# 検査
# =========================================================
def inspect(master: pd.DataFrame) -> None:
    """マスターデータの概要を表示・レポート保存する。"""
    lines = []
    lines.append("=" * 60)
    lines.append("マスターデータセット概要")
    lines.append("=" * 60)
    lines.append(f"行数 : {len(master):,}")
    lines.append(f"列数 : {len(master.columns)}")
    lines.append(f"期間 : {master[COL_DATE].min()} ～ {master[COL_DATE].max()}")
    lines.append(f"複勝率: {master[TARGET].mean():.3f}")

    # 分割確認
    lines.append("\n【時系列分割】")
    for s in ["train", "valid", "test"]:
        n = (master["split"] == s).sum()
        lines.append(f"  {s}: {n:,}行")

    # 着順変換後の確認
    lines.append(f"\n【着順統計】")
    lines.append(master[COL_FINISH].describe().to_string())

    # 欠損確認
    missing = master.isnull().mean()
    high = missing[missing > 0.1].sort_values(ascending=False)
    lines.append(f"\n【欠損10%超の列】({len(high)}列)")
    for col, rate in high.items():
        lines.append(f"  {col}: {rate:.1%}")

    # 全列一覧
    lines.append(f"\n【全列一覧】({len(master.columns)}列)")
    for i, col in enumerate(master.columns, 1):
        dtype = str(master[col].dtype)
        miss  = master[col].isnull().mean()
        lines.append(f"  {i:3d}. {col:<35} dtype={dtype:<15} 欠損={miss:.1%}")

    report = "\n".join(lines)
    print(report)

    report_path = REPORT_DIR / "master_inspection.txt"
    report_path.write_text(report, encoding="utf-8")
    logger.info(f"レポート保存: {report_path}")


def main() -> None:
    master = build_master()
    inspect(master)


if __name__ == "__main__":
    main()