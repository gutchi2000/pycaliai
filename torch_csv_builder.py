"""
torch_csv_builder.py
PyCaLiAI - torch CSV再構築スクリプト
lgbm/cat CSVから前日予想可能な特徴量を最大限抽出し、
Transformer（同レース全頭Attention）用のtorch CSVを生成する。

Usage:
    python torch_csv_builder.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

# =========================================================
# パス設定
# =========================================================
BASE_DIR   = Path(r"E:\PyCaLiAI")
DATA_DIR   = BASE_DIR / "data"

CSV_LGBM   = DATA_DIR / "lgbm_20130105-20251228.csv"
CSV_CAT    = DATA_DIR / "cat_20130105-20251228.csv"
CSV_TORCH_OLD = DATA_DIR / "torch_20130105-20251228.csv"   # 旧ファイル（参照用）
CSV_TORCH_NEW = DATA_DIR / "torch_transformer_20130105-20251228.csv"  # 新出力

JOIN_KEY   = ["レースID(新)", "馬番"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# =========================================================
# 前日予想禁止列（当日情報）
# =========================================================
FORBIDDEN = {
    # オッズ・人気（締切後）
    "人気", "単勝オッズ", "複勝オッズ下限", "複勝オッズ上限", "複勝シェア",
    "複勝人気", "複勝配当", "単勝配当",
    "指定時系列オッズ1・単勝", "指定時系列オッズ2・単勝",
    "指定時系列オッズ3・単勝", "指定時系列オッズ4・単勝",
    "指定時系列オッズ1・複勝", "指定時系列オッズ2・複勝",
    "指定時系列オッズ3・複勝", "指定時系列オッズ4・複勝",
    "指時系1・単勝", "指時系2・単勝", "指時系3・単勝", "指時系4・単勝",
    "指時系1・人気", "指時系2・人気", "指時系3・人気", "指時系4・人気",
    "指時系1・複下", "複上1", "複人気1",
    "指時系2・複下", "複上2", "複人気2",
    "指時系3・複下", "複上3", "複人気3",
    "指時系4・複下", "複上4", "複人気4",
    "補正",
    # 当日馬体重
    "馬体重", "馬体重増減",
    # 結果系（リーク）
    "着順", "確定着順", "走破タイム", "走破タイムS",
    "着差", "着差タイム", "上り3F", "上り3F順",
    "通過順", "通過順1-4", "決め手", "Ave-3F",
    "上り3F地点差", "PCI", "PCI3", "RPCI",
    "平均1Fタイム", "平均速度", "-3F平均速度", "上り3F平均速度",
    "結果コメント", "結果コメントS", "予想コメント", "予想コメントS",
    "レースコメント", "ワーク1", "ワーク1S", "ワーク2", "ワーク2S",
}

# =========================================================
# lgbm CSVから取る列
# =========================================================
LGBM_COLS = [
    # キー
    "レースID(新)", "レースID(新/馬番無)", "血統登録番号", "馬番",
    # レース条件
    "日付", "開催", "場所", "Ｒ", "レース名", "発走時刻",
    "芝・ダ", "距離", "コース区分", "芝(内・外)", "馬場状態", "天気",
    "クラス名", "トラックコード(JV)",
    # 馬プロフィール
    "枠番",
    # 前走成績
    "前走確定着順", "前走走破タイム", "前走着差タイム",
    "前1角", "前2角", "前3角", "前4角",
    "前走上り3F", "前走上り3F順",
    "前走Ave-3F", "前PCI", "前走PCI3", "前走RPCI",
    "前走平均1Fタイム",
    "前走日付", "前走場所", "前芝・ダ", "前距離",
    "前走馬場状態", "前走出走頭数", "前走競走種別",
    "前走トラックコード(JV)", "前走斤量",
    "前走馬体重", "前走馬体重増減",
    "前好走",
    "前走レースID(新)", "前走レースID(新/馬番無)",
]

# =========================================================
# cat CSVから取る列（lgbmにない列のみ）
# =========================================================
CAT_COLS = [
    # キー（JOIN用）
    "レースID(新)", "馬番",
    # 馬プロフィール
    "馬名",
    "種牡馬", "父タイプ名", "母父馬", "母父タイプ名", "毛色",
    # 人的要素
    "騎手コード", "調教師コード",
    "馬主(最新/仮想)", "生産者",
    # レース条件（cat固有）
    "年齢限定", "限定", "性別限定", "指定条件", "重量種別",
]

# =========================================================
# 旧torch CSVから取る列（lgbm/catにない列のみ）
# =========================================================
TORCH_OLD_COLS = [
    # キー（JOIN用）
    "レースID(新)", "馬番",
    # 馬プロフィール
    "性別", "年齢",
    "斤量", "馬齢斤量差", "斤量体重比",
    "ブリンカー", "間隔", "休み明け～戦目",
    # レース条件
    "出走頭数", "フルゲート頭数",
    # 騎手情報（旧torchにあれば）
    "騎手S", "騎手年齢", "チェック騎手タイプ",
    "調教師S", "調教師年齢", "チェック調教師タイプ",
    # 馬印（指数）
    "馬印D1", "馬印D2", "馬印D3", "馬印D4",
    "馬印D5", "馬印D6", "馬印D7", "馬印D8",
]


def load_csv(path: Path, name: str) -> pd.DataFrame:
    """CSVをUTF-8 → CP932フォールバックで読み込む。"""
    logger.info(f"読み込み: {path.name}")
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="cp932", low_memory=False)
    logger.info(f"  [{name}] {len(df):,}行 × {len(df.columns)}列")
    return df


def select_existing(df: pd.DataFrame, cols: list[str], label: str) -> pd.DataFrame:
    """指定列のうち実際に存在するものだけ選択する。欠落列は警告を出す。"""
    existing = [c for c in cols if c in df.columns]
    missing  = [c for c in cols if c not in df.columns]
    if missing:
        logger.warning(f"  [{label}] 存在しない列をスキップ: {missing}")
    return df[existing].copy()


def remove_forbidden(df: pd.DataFrame) -> pd.DataFrame:
    """当日情報・結果リーク列を除去する。"""
    to_drop = [c for c in df.columns if c in FORBIDDEN]
    if to_drop:
        logger.info(f"  禁止列除去: {to_drop}")
        df = df.drop(columns=to_drop)
    return df


def build_torch_csv() -> pd.DataFrame:
    """lgbm + cat + 旧torch をJOINしてtorch用CSVを生成する。"""

    # ---- 読み込み ----
    df_lgbm  = load_csv(CSV_LGBM,      "lgbm")
    df_cat   = load_csv(CSV_CAT,       "cat")
    df_torch_old = load_csv(CSV_TORCH_OLD, "torch_old") if CSV_TORCH_OLD.exists() else None

    # ---- 必要列だけ選択 ----
    df_lgbm  = select_existing(df_lgbm,  LGBM_COLS,      "lgbm")
    df_cat   = select_existing(df_cat,   CAT_COLS,       "cat")

    # ---- lgbm をベースに cat をJOIN ----
    logger.info("lgbm + cat をJOIN中...")
    df = df_lgbm.merge(df_cat, on=JOIN_KEY, how="left")
    logger.info(f"  JOIN後: {len(df):,}行 × {len(df.columns)}列")

    # ---- 旧torchからの補完 ----
    if df_torch_old is not None:
        df_t = select_existing(df_torch_old, TORCH_OLD_COLS, "torch_old")
        # 禁止列を除去
        df_t = remove_forbidden(df_t)
        # すでにdfにある列はJOINキー以外スキップ
        new_cols = [c for c in df_t.columns if c not in df.columns or c in JOIN_KEY]
        df_t = df_t[new_cols]
        logger.info("旧torch からJOIN中...")
        df = df.merge(df_t, on=JOIN_KEY, how="left")
        logger.info(f"  JOIN後: {len(df):,}行 × {len(df.columns)}列")

    # ---- 禁止列の最終チェック ----
    df = remove_forbidden(df)

    # ---- 日付を整数→datetime変換 ----
    if "日付" in df.columns:
        def parse_date(s: str) -> pd.Timestamp:
            s = str(s).strip()
            if len(s) == 6:   # 251220 形式 → 20251220
                s = "20" + s
            return pd.to_datetime(s, format="%Y%m%d")

    df["date_dt"] = df["日付"].astype(str).apply(parse_date)

    # ---- 行数チェック ----
    assert len(df) == 631_965, f"行数異常: {len(df)}"
    logger.info(f"行数確認OK: {len(df):,}行")

    # ---- 保存 ----
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_TORCH_NEW, index=False, encoding="utf-8-sig")
    logger.info(f"保存完了: {CSV_TORCH_NEW}")

    return df


def inspect(df: pd.DataFrame) -> None:
    """生成したCSVの概要を表示する。"""
    print("\n" + "=" * 60)
    print("torch CSV（Transformer用）概要")
    print("=" * 60)
    print(f"行数 : {len(df):,}")
    print(f"列数 : {len(df.columns)}")

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    print(f"数値列: {len(num_cols)}  カテゴリ列: {len(cat_cols)}")

    missing = df.isnull().mean()
    high = missing[missing > 0.1].sort_values(ascending=False)
    print(f"\n欠損10%超の列 ({len(high)}列):")
    for col, rate in high.items():
        print(f"  {col}: {rate:.1%}")

    print(f"\n全列一覧 ({len(df.columns)}列):")
    for i, col in enumerate(df.columns, 1):
        dtype = str(df[col].dtype)
        miss  = df[col].isnull().mean()
        print(f"  {i:3d}. {col:<30} dtype={dtype:<10} 欠損={miss:.1%}")


def main() -> None:
    df = build_torch_csv()
    inspect(df)


if __name__ == "__main__":
    main()