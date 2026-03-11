"""
app.py
PyCaLiAI - Streamlit UI v2（週末CSV対応版）

変更点:
  - トップ: 重賞バナー + 今日のねらい目カード(右側) + 会場タブ + Rボタン
  - レース詳細: 右側に同会場Rナビ + 開催場傾向パネル
  - 馬連: 3点流し（◎-◯・◎-▲・◯-▲）
  - 三連複: ◎◯2頭軸×▲△ 2点
  - 三連単: 廃止
  - 除外: 東京・小倉・新馬

Usage:
    streamlit run app.py
"""

from __future__ import annotations

import json
import itertools
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import streamlit as st

try:
    import japanize_matplotlib  # noqa: F401
except ImportError:
    pass

import matplotlib.font_manager as fm
import glob as _glob
fm._load_fontmanager(try_read_cache=False)

_jp_font = None
for _kw in ["Noto Sans CJK", "NotoSansCJK", "IPA", "Hiragino", "Yu Gothic", "Meiryo"]:
    _found = [f.fname for f in fm.fontManager.ttflist if _kw in f.name]
    if _found:
        _jp_font = _found[0]
        break
if _jp_font is None:
    _candidates = (
        _glob.glob("/usr/share/fonts/**/*.otf", recursive=True) +
        _glob.glob("/usr/share/fonts/**/*.ttf", recursive=True)
    )
    _jp_font = next((p for p in _candidates if "CJK" in p or "Noto" in p), None)
if _jp_font:
    fm.fontManager.addfont(_jp_font)
    prop = fm.FontProperties(fname=_jp_font)
    plt.rcParams["font.family"] = prop.get_name()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR      = Path(__file__).parent
DATA_DIR      = BASE_DIR / "data"
MODEL_DIR     = BASE_DIR / "models"
STRATEGY_JSON    = DATA_DIR / "strategy_weights.json"
COURSE_TREND_JSON = DATA_DIR / "course_trend.json"
RESULTS_JSON      = DATA_DIR / "results.json"
TYAKU_DIR     = DATA_DIR / "tyaku"
LGBM_PATH     = MODEL_DIR / "lgbm_optuna_v1.pkl"
CAT_PATH      = MODEL_DIR / "catboost_optuna_v1.pkl"
CAL_PATH       = MODEL_DIR / "ensemble_calibrator_v1.pkl"
TORCH_PATH     = MODEL_DIR / "transformer_optuna_v1.pkl"
META_PATH      = MODEL_DIR / "stacking_meta_v1.pkl"
STACK_CAL_PATH = MODEL_DIR / "stacking_calibrator_v1.pkl"

# モデルキャッシュ（プロセス内で1回だけロード）
_model_cache: dict = {}

def _get_cached(path: Path, key: str):
    if key not in _model_cache and path.exists():
        _model_cache[key] = joblib.load(path)
    return _model_cache.get(key)

MIN_UNIT = 100
MARKS    = ["◎", "◯", "▲", "△", "×"]

EXCLUDE_PLACES  = {"東京", "小倉"}
EXCLUDE_CLASSES = {"新馬", "障害"}

CLASS_NORMALIZE = {
    "新馬":"新馬","未勝利":"未勝利",
    "1勝":"1勝","500万":"1勝",
    "2勝":"2勝","1000万":"2勝",
    "3勝":"3勝","1600万":"3勝",
    "OP(L)":"OP(L)","オープン":"ｵｰﾌﾟﾝ",
    "Ｇ１":"Ｇ１","Ｇ２":"Ｇ２","Ｇ３":"Ｇ３",
}

GRADE_ORDER = {"Ｇ１":0,"Ｇ２":1,"Ｇ３":2,"OP(L)":3}

FEATURE_LABEL = {
    "前走確定着順":"前走着順","前走上り3F":"前走上り",
    "枠番":"枠番","馬番":"馬番","斤量":"斤量",
    "距離":"距離","前走距離":"前走距離",
    "前走着差タイム":"前走着差","前走走破タイム":"前走タイム",
}

MARK_CLASS = {
    "◎":"mk-hon","◯":"mk-tai","▲":"mk-sabo",
    "△":"mk-del","×":"mk-batu","":"",
}

COLUMN_MAP = {
    "馬名S":"馬名","芝・ダート":"芝・ダ",
    "馬場状態(暫定)":"馬場状態","天候(暫定)":"天気",
    "人気_今走":"人気","ZI順":"ZI順位",
    "父":"種牡馬","母父":"母父馬",
    "父タイプ":"父タイプ名","母父タイプ":"母父タイプ名",
    "前走着順":"前走確定着順","前走上り3F":"前走上り3F",
    "前走TD":"前芝・ダ","前走間隔":"間隔",
    "前走着差":"前走着差タイム","前走斤量":"前走斤量",
    "前走Ave3F":"前走Ave-3F","前走上り3F順位":"前走上り3F順",
    "マイニング順位":"マイニング順位","前走単勝オッズ":"前走単勝オッズ",
    "前走通過1":"前1角","前走通過2":"前2角",
    "前走通過3":"前3角","前走通過4":"前4角",
    # 前走距離 → 前距離（モデルが使う列名）
    "前走距離":"前距離",
}

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
HORSE_COLS_99 = [
    "枠番","B","馬番","馬名S","性別","年齢","馬体重","馬体重増減_raw","馬体重増減",
    "人気_今走","単勝","ZI印","ZI","ZI順","斤量","減M","替","騎手","所属","調教師",
    "父","母父","父タイプ","母父タイプ",
    "前走月","前走日","前走開催","前走間隔","前走レース名","前走TD","前走距離","前走馬場状態",
    "前走B","前走騎手","前走斤量","前走減","前走人気","前走単勝オッズ","前走着順","前走着差",
    "マイニング順位","前走通過1","前走通過2","前走通過3","前走通過4","前走Ave3F",
    "前走上り3F","前走上り3F順位","前走1_2着馬",
    "二走前月","二走前日","二走前開催","二走前間隔","二走前レース名","二走前TD","二走前距離","二走前馬場状態",
    "二走前B","二走前騎手","二走前斤量","二走前減","二走前人気","二走前単勝オッズ","二走前着順","二走前着差",
    "二走前マイニング順位","二走前通過1","二走前通過2","二走前通過3","二走前通過4","二走前Ave3F",
    "二走前上り3F","二走前上り3F順位","二走前1_2着馬",
    "三走前月","三走前日","三走前開催","三走前間隔","三走前レース名","三走前TD","三走前距離","三走前馬場状態",
    "三走前B","三走前騎手","三走前斤量","三走前減","三走前人気","三走前単勝オッズ","三走前着順","三走前着差",
    "三走前マイニング順位","三走前通過1","三走前通過2","三走前通過3","三走前通過4","三走前Ave3F",
    "三走前上り3F","三走前上り3F順位","三走前1_2着馬",
]


# =========================================================
# 着度数CSV パース（predict_weekly.py と同一ロジック）
# =========================================================
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


def _load_tyaku(date_str: str) -> "pd.DataFrame | None":
    """data/tyaku/YYYYMMDD.csv を読み込み、馬番→着度数の対応表を返す。"""
    path = TYAKU_DIR / f"{date_str}.csv"
    if not path.exists():
        return None
    for enc in ["cp932", "shift_jis", "utf-8"]:
        try:
            text = path.read_bytes().decode(enc); break
        except Exception:
            continue
    else:
        return None

    rows: list[dict] = []
    current_race_id: str | None = None
    for line in text.splitlines():
        cols = line.split(",")
        if len(cols) == 19 and cols[0] not in ("レースID(新)", ""):
            current_race_id = cols[0].strip()[:16]
        elif len(cols) == 55 and cols[0] not in ("枠番", "") and current_race_id:
            row = dict(zip(TYAKU_HORSE_COLS, cols))
            row["レースID(新/馬番無)"] = current_race_id
            rows.append(row)

    if not rows:
        return None

    df = pd.DataFrame(rows)
    for col in ["馬番","馬体重","増減",
                "中央平地全:1着","中央平地全:2着","中央平地全:3着","中央平地全:外",
                "同コース:1着","同コース:2着","同コース:3着","同コース:外",
                "同クラス:1着","同クラス:2着","同クラス:3着","同クラス:外"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Bayesian smoothing: prior = 0.286, 仮想サンプル数 5
    _PRIOR_ALPHA = 1.43
    _PRIOR_BETA  = 5.0
    for prefix, out_col in [("中央平地全", "horse_fuku_career"),
                             ("同コース",   "horse_fuku_course"),
                             ("同クラス",   "horse_fuku_class")]:
        w = df[f"{prefix}:1着"].fillna(0) + df[f"{prefix}:2着"].fillna(0) + df[f"{prefix}:3着"].fillna(0)
        total = w + df[f"{prefix}:外"].fillna(0)
        df[out_col] = (w + _PRIOR_ALPHA) / (total + _PRIOR_BETA)

    if "増減" in df.columns:
        df["増減"] = df["増減"].astype(str).str.replace(" ", "").str.replace("－","-").str.replace("＋","+")
        df["増減"] = pd.to_numeric(df["増減"], errors="coerce")

    keep = ["レースID(新/馬番無)","馬番","馬体重","増減",
            "horse_fuku_career","horse_fuku_course","horse_fuku_class"]
    return df[[c for c in keep if c in df.columns]]


# =========================================================
# CSV パース
# =========================================================
def parse_target_csv(source) -> pd.DataFrame:
    if isinstance(source, (str, Path)):
        with open(source, "rb") as f:
            raw = f.read()
    else:
        raw = source.read()
    text = ""
    for enc in ["cp932", "shift_jis", "utf-8"]:
        try:
            text = raw.decode(enc); break
        except Exception:
            continue
    if not text:
        return pd.DataFrame()

    races: list[dict] = []
    current_race: dict | None = None
    for line in text.splitlines():
        cols = line.split(",")
        if cols[0] in ("レースID(新)", "枠番", "番", ""):
            continue
        if len(cols) == 19:
            current_race = dict(zip(RACE_COLS, cols))
        elif len(cols) == 33 and current_race:
            h = dict(zip(HORSE_COLS_33, cols)); h.update(current_race); races.append(h)
        elif len(cols) == 46 and current_race:
            h = dict(zip(HORSE_COLS_46, cols)); h.update(current_race); races.append(h)
        elif len(cols) == 48 and current_race:
            h = dict(zip(HORSE_COLS_48, cols)); h.update(current_race); races.append(h)
        elif len(cols) == 49 and current_race:
            h = dict(zip(HORSE_COLS_49, cols)); h.update(current_race); races.append(h)
        elif len(cols) == 99 and current_race:
            h = dict(zip(HORSE_COLS_99, cols)); h.update(current_race); races.append(h)

    df = pd.DataFrame(races)
    if df.empty:
        return df
    df = df.rename(columns=COLUMN_MAP)

    # 障害レース除外（距離列またはクラス名列に"障害"を含む）
    mask_shogai = pd.Series([False] * len(df), index=df.index)
    if "距離" in df.columns:
        mask_shogai |= df["距離"].astype(str).str.contains("障害", na=False)
    if "クラス名" in df.columns:
        mask_shogai |= df["クラス名"].astype(str).str.contains("障害", na=False)
    if mask_shogai.any():
        before = len(df)
        df = df[~mask_shogai].copy()
        logger.info(f"障害レース除外: {before - len(df)}頭")

    df["レースID(新/馬番無)"] = df["レースID(新)"].astype(str).str[:16]
    for col in ["枠番","馬番","斤量","ZI","ZI順位","距離","人気","単勝",
                "前走確定着順","前走上り3F","前走距離","間隔","前走人気",
                "前走着差タイム","前走斤量","前走Ave-3F","前走上り3F順",
                "マイニング順位","前走単勝オッズ",
                "前1角","前2角","前3角","前4角",
                # COLUMN_MAP で rename された後の列名
                "前距離","前走馬体重","前走馬体重増減"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["日付"] = pd.to_datetime(df["日付S"], format="%Y.%m.%d", errors="coerce")
    df["日付"] = df["日付"].dt.strftime("%Y%m%d").astype("Int64")
    for col in ["前走走破タイム","前走着差タイム","馬体重","馬体重増減",
                "前走斤量","生産者","馬主(最新/仮想)"]:
        if col not in df.columns:
            df[col] = 0

    # 出走頭数：週次CSVは直接持たないため、レースIDごとの馬数から算出
    if "出走頭数" not in df.columns:
        df["出走頭数"] = df.groupby("レースID(新/馬番無)")["馬番"].transform("count")
    df["出走頭数"] = pd.to_numeric(df["出走頭数"], errors="coerce").fillna(
        pd.to_numeric(df.get("フルゲート頭数"), errors="coerce")
    )

    # 脚質特徴量（前1角・前4角から計算）
    if "前1角" in df.columns and "前4角" in df.columns:
        n = df["出走頭数"].clip(lower=2)
        front = pd.to_numeric(df["前1角"], errors="coerce")
        back  = pd.to_numeric(df["前4角"], errors="coerce")
        df["prev_pos_rel"]  = (front - 1) / (n - 1)
        df["closing_power"] = (front - back) / (n - 1)
    else:
        df["prev_pos_rel"]  = np.nan
        df["closing_power"] = np.nan

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
        stats_path = DATA_DIR / fname
        if stats_path.exists():
            stats = pd.read_csv(stats_path, encoding="utf-8-sig")
            if code_col in df.columns:
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
    if isinstance(source, (str, Path)):
        date_str = Path(source).stem
    else:
        date_str = Path(getattr(source, "name", "")).stem
    tyaku_df = _load_tyaku(date_str)
    if tyaku_df is not None:
        df = df.merge(tyaku_df, on=["レースID(新/馬番無)", "馬番"], how="left",
                      suffixes=("", "_tyaku"))
        if "馬体重_tyaku" in df.columns:
            df["馬体重"] = df["馬体重_tyaku"].combine_first(
                pd.to_numeric(df.get("馬体重"), errors="coerce"))
            df.drop(columns=["馬体重_tyaku"], inplace=True)
        if "増減_tyaku" in df.columns:
            df["馬体重増減"] = df["増減_tyaku"]
            df.drop(columns=["増減_tyaku"], inplace=True)
        if "horse_fuku_career" in df.columns:
            df["horse_fuku10"] = df["horse_fuku_career"].fillna(0.286)
            df["horse_fuku30"] = df["horse_fuku_career"].fillna(0.312)
        else:
            df["horse_fuku10"] = 0.286
            df["horse_fuku30"] = 0.312
    else:
        # 着度数CSVなし → 訓練データ中央値で補完
        df["horse_fuku10"] = 0.286   # 訓練 valid 中央値
        df["horse_fuku30"] = 0.312   # 訓練 valid 中央値

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

    # 斤量体重比 = 斤量 / 馬体重（馬体重 0 or 欠損のときは訓練中央値で補完）
    if "斤量体重比" not in df.columns:
        wt = pd.to_numeric(df["馬体重"], errors="coerce").replace(0, np.nan) if "馬体重" in df.columns else np.nan
        jk = pd.to_numeric(df["斤量"], errors="coerce") if "斤量" in df.columns else np.nan
        if isinstance(wt, pd.Series) and isinstance(jk, pd.Series):
            df["斤量体重比"] = (jk / wt).fillna(11.8)
        else:
            df["斤量体重比"] = 11.8

    # 訓練データに含まれるが週次CSVにない特徴量を中央値で補完
    _MISSING_FEATURE_MEDIANS = {
        "馬齢斤量差":            -1,
        "トラックコード(JV)":      23,
        "前走トラックコード(JV)":  23,
        "前走競走種別":           13,
        "前走出走頭数":           15,
        "前走馬体重":            472,
        "前走馬体重増減":           0,
        "騎手年齢":              30,
        "調教師年齢":             53,
        "休み明け～戦目":           2,
    }
    for col, med in _MISSING_FEATURE_MEDIANS.items():
        if col not in df.columns:
            df[col] = med

    return df


# =========================================================
# モデルロード
# =========================================================
@st.cache_resource(show_spinner="モデル読み込み中...")
def load_models() -> tuple:
    missing = [p for p in (LGBM_PATH, CAT_PATH) if not p.exists()]
    if missing:
        names = ", ".join(p.name for p in missing)
        st.error(f"モデルファイルが見つかりません: {names}\n`optuna_lgbm.py` / `optuna_catboost.py` を先に実行してください。")
        st.stop()
    return joblib.load(LGBM_PATH), joblib.load(CAT_PATH)


@st.cache_data(show_spinner="戦略データ読み込み中...")
def load_strategy() -> dict:
    if not STRATEGY_JSON.exists():
        st.error(f"戦略ファイルが見つかりません: {STRATEGY_JSON}")
        st.stop()
    with open(STRATEGY_JSON, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_results() -> dict:
    """的中実績データ読み込み（results.json）。"""
    if not RESULTS_JSON.exists():
        return {}
    with open(RESULTS_JSON, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_course_trend() -> dict:
    """コース別傾向データ読み込み（course_trend.json）。"""
    if not COURSE_TREND_JSON.exists():
        return {}
    with open(COURSE_TREND_JSON, encoding="utf-8") as f:
        return json.load(f)


# =========================================================
# 予測
# =========================================================
from utils import parse_time_str


def predict_lgbm(df: pd.DataFrame, obj: dict) -> np.ndarray:
    model, encoders, feature_cols = obj["model"], obj["encoders"], obj["feature_cols"]
    df = df.copy()
    for col in ["前走走破タイム","前走着差タイム"]:
        if col in df.columns:
            df[col] = parse_time_str(df[col])
    for col, le in encoders.items():
        if col not in df.columns:
            df[col] = 0; continue
        df[col] = df[col].astype(str).fillna("__NaN__")
        known = set(le.classes_)
        df[col] = df[col].apply(lambda x: x if x in known else "__NaN__")
        if "__NaN__" not in le.classes_:
            le.classes_ = np.append(le.classes_, "__NaN__")
        df[col] = le.transform(df[col])
    _ROLLING_COLS = {"jockey_fuku30","jockey_fuku90","trainer_fuku30","trainer_fuku90",
                     "horse_fuku10","horse_fuku30","prev_pos_rel","closing_power"}
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan if col in _ROLLING_COLS else 0
    return model.predict_proba(df[feature_cols])[:, 1]


def predict_catboost(df: pd.DataFrame, obj: dict) -> np.ndarray:
    from catboost import Pool
    model, feature_cols = obj["model"], obj["feature_cols"]
    cat_list = [
        "種牡馬","父タイプ名","母父馬","母父タイプ名","毛色",
        "馬主(最新/仮想)","生産者","芝・ダ","コース区分","芝(内・外)",
        "馬場状態","天気","クラス名","場所","性別","斤量",
        "ブリンカー","重量種別","年齢限定","限定","性別限定","指定条件",
        "前走場所","前芝・ダ","前走馬場状態","前走斤量","前好走",
    ]
    df = df.copy()
    for col in ["前走走破タイム","前走着差タイム"]:
        if col in df.columns:
            df[col] = parse_time_str(df[col])
    for col in cat_list:
        df[col] = df[col].fillna("__NaN__").astype(str) if col in df.columns else "__NaN__"
    _ROLLING_COLS = {"jockey_fuku30","jockey_fuku90","trainer_fuku30","trainer_fuku90",
                     "horse_fuku10","horse_fuku30","prev_pos_rel","closing_power"}
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan if col in _ROLLING_COLS else 0.0
    cat_idx = [i for i, c in enumerate(feature_cols) if c in cat_list]
    pool = Pool(df[feature_cols], cat_features=cat_idx)
    return model.predict_proba(pool)[:, 1]


_META_EXTRA = ["芝・ダ", "距離", "クラス名", "場所", "馬場状態", "出走頭数", "枠番", "馬番"]


def predict_transformer_local(df: pd.DataFrame) -> np.ndarray:
    """Transformer予測。モデル未存在 or 失敗時はゼロ配列を返す。"""
    torch_obj = _get_cached(TORCH_PATH, "torch")
    if torch_obj is None:
        return np.zeros(len(df))
    try:
        import torch
        from train_transformer import RaceTransformer, RaceDataset, MAX_HORSES
        from train_transformer import preprocess as torch_preprocess
        from torch.utils.data import DataLoader

        DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_config = torch_obj["model_config"]
        encoders     = torch_obj["encoders"]
        num_stats    = torch_obj["num_stats"]
        num_cols     = torch_obj["num_cols"]
        cat_cols     = torch_obj["cat_cols"]

        df2 = df.copy()
        if "fukusho_flag" not in df2.columns:
            df2["fukusho_flag"] = 0  # 予測時はターゲット列不要だが RaceDataset が参照するため
        df2, _, _ = torch_preprocess(df2, encoders=encoders, fit=False, num_stats=num_stats)

        if "torch_model" not in _model_cache:
            m = RaceTransformer(
                cat_vocab_sizes=model_config["cat_vocab_sizes"],
                cat_cols=model_config["cat_cols"],
                n_num=model_config["n_num"],
                d_model=model_config.get("d_model", 128),
                n_heads=model_config.get("n_heads", 4),
                n_layers=model_config.get("n_layers", 2),
                d_ff=model_config.get("d_ff", 256),
                dropout=model_config.get("dropout", 0.1),
            ).to(DEVICE)
            m.load_state_dict(torch_obj["model_state"])
            m.eval()
            _model_cache["torch_model"] = (m, DEVICE)
        model, DEVICE = _model_cache["torch_model"]

        ds     = RaceDataset(df2, cat_cols, num_cols, model_config["cat_vocab_sizes"])
        loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)

        all_proba: list[float] = []
        with torch.no_grad():
            for batch in loader:
                logits = model(batch["cat"].to(DEVICE), batch["num"].to(DEVICE), batch["mask"].to(DEVICE))
                proba  = torch.sigmoid(logits).cpu().numpy()
                valid  = ~batch["mask"].numpy()
                for b in range(len(proba)):
                    for h in range(MAX_HORSES):
                        if valid[b, h]:
                            all_proba.append(float(proba[b, h]))

        result  = np.zeros(len(df))
        df_sort = df.sort_values("レースID(新/馬番無)").reset_index(drop=True)
        idx = 0
        for _, group in df_sort.groupby("レースID(新/馬番無)", sort=True):
            for orig_idx in list(group.index)[:MAX_HORSES]:
                if idx < len(all_proba):
                    result[orig_idx] = all_proba[idx]
                    idx += 1
        return result
    except Exception as e:
        logger.warning(f"Transformer予測失敗（0で埋め）: {e}")
        return np.zeros(len(df))


def predict_stacking(df: pd.DataFrame, lgbm_obj: dict, cat_obj: dict) -> np.ndarray | None:
    """スタッキングモデルで予測。未存在 or 失敗時は None を返す。"""
    if not META_PATH.exists() or not TORCH_PATH.exists():
        return None
    try:
        p_lgbm  = predict_lgbm(df, lgbm_obj)
        p_cat   = predict_catboost(df, cat_obj)
        p_torch = predict_transformer_local(df)

        meta_obj      = _get_cached(META_PATH, "meta")
        meta_model    = meta_obj["meta_model"]
        meta_encoders = meta_obj["meta_encoders"]
        meta_cols     = meta_obj["meta_cols"]

        meta_df = df.copy()
        if "出走頭数" not in meta_df.columns or meta_df["出走頭数"].isna().all():
            meta_df["出走頭数"] = meta_df.groupby("レースID(新/馬番無)")["馬番"].transform("count")
        meta_df["クラス名"] = meta_df["クラス名"].map(CLASS_NORMALIZE).fillna(meta_df["クラス名"])
        meta_df = meta_df.reindex(columns=_META_EXTRA).copy()

        for col in meta_df.select_dtypes(include="object").columns:
            if col in meta_encoders:
                le    = meta_encoders[col]
                meta_df[col] = meta_df[col].fillna("__NaN__").astype(str)
                known = set(le.classes_)
                meta_df[col] = meta_df[col].apply(lambda x: x if x in known else "__NaN__")
                meta_df[col] = le.transform(meta_df[col])
            else:
                meta_df[col] = 0

        meta_df = meta_df.fillna(0)
        meta_df["lgbm"]        = p_lgbm
        meta_df["catboost"]    = p_cat
        meta_df["transformer"] = p_torch

        return meta_model.predict_proba(meta_df[meta_cols])[:, 1]
    except Exception as e:
        logger.debug(f"スタッキング予測失敗（フォールバック）: {e}")
        return None


@st.cache_resource
def _load_calibrator():
    if CAL_PATH.exists():
        return joblib.load(CAL_PATH)["calibrator"]
    logger.warning("キャリブレーター未生成。calibrate.py を先に実行してください。")
    return None


def ensemble_predict(df: pd.DataFrame, lgbm_obj: dict, cat_obj: dict) -> np.ndarray:
    # スタッキング優先（キャリブレーター妥当性チェック付き）
    stacking = predict_stacking(df, lgbm_obj, cat_obj)
    if stacking is not None:
        cal_obj = _get_cached(STACK_CAL_PATH, "stack_cal")
        if cal_obj is not None:
            calibrated = cal_obj["calibrator"].transform(stacking)
            # キャリブレーター出力の妥当性チェック：平均 5% 以上かつ最大 50% 未満
            if calibrated.mean() >= 0.05 and calibrated.max() < 0.50:
                return calibrated
            logger.warning(
                f"スタッキングキャリブレーター出力が異常（mean={calibrated.mean():.3f}, "
                f"max={calibrated.max():.3f}）→ エンサンブルにフォールバック"
            )
    # フォールバック: 2モデル平均 + キャリブレーション
    raw = 0.5 * predict_lgbm(df, lgbm_obj) + 0.5 * predict_catboost(df, cat_obj)
    cal = _load_calibrator()
    return cal.transform(raw) if cal is not None else raw


def assign_marks(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["mark"] = ""
    ranked = df["prob"].rank(ascending=False, method="first")
    for idx, rank in ranked.items():
        if rank <= 5:
            df.at[idx, "mark"] = {1:"◎",2:"◯",3:"▲",4:"△",5:"×"}[int(rank)]
    return df


@st.cache_data(show_spinner="全レース予想計算中...")
def predict_all_races(cache_key: str, df_json: str, _lgbm_obj: dict, _cat_obj: dict) -> str:
    import io
    df = pd.read_json(io.StringIO(df_json))
    result_frames = []
    for race_id, race_df in df.groupby("レースID(新/馬番無)"):
        race_df = race_df.copy()
        try:
            race_df["prob"]  = ensemble_predict(race_df, _lgbm_obj, _cat_obj)
            race_df          = assign_marks(race_df)
            race_df["score"] = (race_df["prob"] * 100).round(1)
        except Exception as e:
            logger.warning(f"予測失敗 {race_id}: {e}")
            race_df["prob"]  = 0.0
            race_df["mark"]  = ""
            race_df["score"] = 0.0
        result_frames.append(race_df)
    return pd.concat(result_frames, ignore_index=True).to_json(force_ascii=False)


# =========================================================
# SHAP
# =========================================================
@st.cache_data(show_spinner="SHAP計算中...")
def compute_shap(_lgbm_obj: dict, df_json: str) -> tuple[list, list]:
    import io
    df = pd.read_json(io.StringIO(df_json))
    model, encoders, feature_cols = _lgbm_obj["model"], _lgbm_obj["encoders"], _lgbm_obj["feature_cols"]
    df_enc = df.copy()
    for col in ["前走走破タイム","前走着差タイム"]:
        if col in df_enc.columns:
            df_enc[col] = parse_time_str(df_enc[col])
    for col, le in encoders.items():
        if col not in df_enc.columns:
            df_enc[col] = 0; continue
        df_enc[col] = df_enc[col].astype(str).fillna("__NaN__")
        known = set(le.classes_)
        df_enc[col] = df_enc[col].apply(lambda x: x if x in known else "__NaN__")
        if "__NaN__" not in le.classes_:
            le.classes_ = np.append(le.classes_, "__NaN__")
        df_enc[col] = le.transform(df_enc[col])
    for col in feature_cols:
        if col not in df_enc.columns:
            df_enc[col] = 0
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(df_enc[feature_cols])
    if isinstance(sv, list):
        sv = sv[1]
    return sv.tolist(), feature_cols


def make_shap_fig(sv_row: list, feature_cols: list, horse_name: str) -> plt.Figure:
    sv    = np.array(sv_row)
    order = np.argsort(np.abs(sv))[::-1][:12]
    labels = [FEATURE_LABEL.get(feature_cols[i], feature_cols[i]) for i in order]
    values = sv[order]
    colors = ["tomato" if v > 0 else "steelblue" for v in values]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.barh(labels[::-1], values[::-1], color=colors[::-1])
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_title(horse_name, fontsize=10)
    ax.set_xlabel("SHAP値", fontsize=8)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig


def make_comment(sv_row: list, feature_cols: list, horse_name: str, score: float, mark: str = "") -> str:
    sv    = np.array(sv_row)
    pairs = sorted(zip(sv, feature_cols), reverse=True)
    pos   = [(v, c) for v, c in pairs if v > 0][:3]
    neg   = [(v, c) for v, c in pairs if v < 0][-2:]
    level = ("モデルが最上位クラスの評価" if score >= 60 else
             "上位圏の評価" if score >= 40 else
             "中位圏の評価" if score >= 20 else "下位圏の評価")
    mark_txt = {"◎":"本命として最も信頼できる一頭。","◯":"対抗として本命を脅かす存在。",
                "▲":"単穴として一発の魅力がある。","△":"連下候補として抑えておきたい。",
                "×":"押さえ程度だが圏外とも言えない。"}.get(mark, "")
    pos_s = [f"{FEATURE_LABEL.get(c,c)}が好材料" for _,c in pos]
    neg_s = [f"{FEATURE_LABEL.get(c,c)}がやや不安" for _,c in neg]
    lines = []
    if mark_txt: lines.append(mark_txt)
    lines.append(f"{horse_name}はスコア{score:.1f}%で{level}。")
    if pos_s: lines.append("好材料: " + "、".join(pos_s) + "。")
    if neg_s: lines.append("不安: " + "、".join(neg_s) + "。")
    return " ".join(lines)


# =========================================================
# 買い目生成
# =========================================================
def floor_to_unit(x: int, unit: int = MIN_UNIT) -> int:
    return max((x // unit) * unit, unit)


def _normalize_baba(raw: str) -> str:
    """'重(暫定)' → '重'  / '稍重(暫定)' → '稍重' など、（暫定）表記を除去する。"""
    return raw.replace("(暫定)", "").replace("暫定", "").strip()


def is_in_strategy(place: str, cls_raw: str, strategy: dict) -> bool:
    if place in EXCLUDE_PLACES or cls_raw in EXCLUDE_CLASSES:
        return False
    cls = CLASS_NORMALIZE.get(cls_raw, cls_raw)
    return place in strategy and (cls in strategy[place] or cls_raw in strategy[place])


def get_bets(race_df: pd.DataFrame, place: str, cls_raw: str,
             strategy: dict, budget: int) -> dict:
    """HAHO（馬連◎軸2点+三連複ボックス1点）/ HALO（三連複ボックス1点のみ）
       / LALO（複勝◎1点のみ）を返す。
    戻り値: {"HAHO": [bets...], "HALO": [bets...], "LALO": [bets...]}  ※空の場合はキーなし"""
    if place in EXCLUDE_PLACES or cls_raw in EXCLUDE_CLASSES:
        return {}
    cls      = CLASS_NORMALIZE.get(cls_raw, cls_raw)
    bet_info = strategy.get(place, {}).get(cls) or strategy.get(place, {}).get(cls_raw, {})
    if not bet_info:
        return {}
    marks_df = {m: race_df[race_df["mark"] == m] for m in ["◎","◯","▲"]}
    hon    = marks_df["◎"]
    taikou = marks_df["◯"]
    sabo   = marks_df["▲"]
    if hon.empty:
        return {}
    h1 = int(hon.iloc[0]["馬番"])
    h2 = int(taikou.iloc[0]["馬番"]) if not taikou.empty else None
    h3 = int(sabo.iloc[0]["馬番"])   if not sabo.empty  else None

    result = {}

    # ── HAHO: 馬連◎軸2点 + 三連複ボックス1点 ──────────────────────────
    haho_types = {k: v for k, v in bet_info.items() if k in ("馬連", "三連複")}
    if haho_types and h2 and h3:
        haho_bets = []
        total_ratio = sum(v["bet_ratio"] for v in haho_types.values()) or 1.0
        if "馬連" in haho_types:
            info = haho_types["馬連"]
            amt  = floor_to_unit(int(budget * info["bet_ratio"] / total_ratio))
            cbs  = [(h1, h2), (h1, h3)]
            per  = floor_to_unit(amt // len(cbs))
            for a, b in cbs:
                haho_bets.append({"馬券種":"馬連","買い目":f"{min(a,b)}-{max(a,b)}",
                                  "購入額":per,"ROI":info.get("roi_oos", info.get("roi", 0))})
        if "三連複" in haho_types:
            info = haho_types["三連複"]
            amt  = floor_to_unit(int(budget * info["bet_ratio"] / total_ratio))
            c_sf = tuple(sorted([h1, h2, h3]))
            haho_bets.append({"馬券種":"三連複","買い目":"-".join(map(str, c_sf)),
                              "購入額":amt,"ROI":info.get("roi_oos", info.get("roi", 0))})
        if haho_bets:
            result["HAHO"] = haho_bets

    # ── HALO: 三連複ボックス1点のみ（全予算）──────────────────────────
    if "三連複" in bet_info and h2 and h3:
        info = bet_info["三連複"]
        c_sf = tuple(sorted([h1, h2, h3]))
        result["HALO"] = [{"馬券種":"三連複","買い目":"-".join(map(str, c_sf)),
                           "購入額":floor_to_unit(budget),"ROI":info.get("roi_oos", info.get("roi", 0))}]

    # ── LALO: 複勝◎1点のみ（全予算）────────────────────────────────────
    any_info = next(iter(bet_info.values()), {})
    result["LALO"] = [{"馬券種":"複勝","買い目":str(h1),
                       "購入額":floor_to_unit(budget),"ROI":any_info.get("roi_oos", any_info.get("roi", 0))}]

    # ── CQC: 単勝◎1点のみ（全予算）─────────────────────────────────────
    result["CQC"]  = [{"馬券種":"単勝","買い目":str(h1),
                       "購入額":floor_to_unit(budget),"ROI":any_info.get("roi_oos", any_info.get("roi", 0))}]

    return result


# =========================================================
# レース傾向データ（過去10年実績・2016〜2025）
# =========================================================
TREND_DATA: dict[str, dict] = {
    "札幌": {
        "好調枠番": [
            "6枠",
            "7枠",
            "1枠"
        ],
        "脚質ランク": [
            {
                "脚質": "逃げ",
                "勝率": 47.1,
                "複勝率": 100.0
            },
            {
                "脚質": "ﾏｸﾘ",
                "勝率": 37.1,
                "複勝率": 100.0
            },
            {
                "脚質": "先行",
                "勝率": 36.2,
                "複勝率": 100.0
            },
            {
                "脚質": "中団",
                "勝率": 23.1,
                "複勝率": 100.0
            },
            {
                "脚質": "後方",
                "勝率": 18.2,
                "複勝率": 100.0
            }
        ],
        "好調騎手": [
            {
                "騎手": "モレイラ",
                "勝率": 54.7,
                "複勝率": 100.0
            },
            {
                "騎手": "ルメール",
                "勝率": 46.2,
                "複勝率": 100.0
            },
            {
                "騎手": "武豊",
                "勝率": 40.8,
                "複勝率": 100.0
            }
        ],
        "好調調教師": [
            {
                "調教師": "堀宣行",
                "勝率": 56.6,
                "複勝率": 100.0
            },
            {
                "調教師": "須貝尚介",
                "勝率": 42.1,
                "複勝率": 100.0
            },
            {
                "調教師": "伊藤圭三",
                "勝率": 39.1,
                "複勝率": 100.0
            }
        ],
        "好調血統_父": [
            {
                "種牡馬": "キズナ",
                "勝率": 48.2,
                "複勝率": 100.0
            },
            {
                "種牡馬": "ドゥラメンテ",
                "勝率": 46.7,
                "複勝率": 100.0
            },
            {
                "種牡馬": "リオンディーズ",
                "勝率": 46.0,
                "複勝率": 100.0
            }
        ],
        "好調血統_母父": [
            {
                "母父馬": "ネオユニヴァース",
                "勝率": 40.7,
                "複勝率": 100.0
            },
            {
                "母父馬": "サクラバクシンオー",
                "勝率": 40.6,
                "複勝率": 100.0
            },
            {
                "母父馬": "ディープインパクト",
                "勝率": 39.0,
                "複勝率": 100.0
            }
        ],
        "配当傾向": {
            "単勝中央値": 460,
            "馬連中央値": 1790,
            "三連複中央値": 4690,
            "荒れ率": 5.7
        }
    },
    "函館": {
        "好調枠番": [
            "5枠",
            "6枠",
            "8枠"
        ],
        "脚質ランク": [
            {
                "脚質": "逃げ",
                "勝率": 51.4,
                "複勝率": 100.0
            },
            {
                "脚質": "ﾏｸﾘ",
                "勝率": 36.0,
                "複勝率": 100.0
            },
            {
                "脚質": "先行",
                "勝率": 34.3,
                "複勝率": 100.0
            },
            {
                "脚質": "中団",
                "勝率": 23.0,
                "複勝率": 100.0
            },
            {
                "脚質": "後方",
                "勝率": 16.7,
                "複勝率": 100.0
            }
        ],
        "好調騎手": [
            {
                "騎手": "藤岡佑介",
                "勝率": 45.7,
                "複勝率": 100.0
            },
            {
                "騎手": "池添謙一",
                "勝率": 40.3,
                "複勝率": 100.0
            },
            {
                "騎手": "吉田隼人",
                "勝率": 39.7,
                "複勝率": 100.0
            }
        ],
        "好調調教師": [
            {
                "調教師": "矢作芳人",
                "勝率": 44.4,
                "複勝率": 100.0
            },
            {
                "調教師": "伊藤圭三",
                "勝率": 35.7,
                "複勝率": 100.0
            },
            {
                "調教師": "須貝尚介",
                "勝率": 34.1,
                "複勝率": 100.0
            }
        ],
        "好調血統_父": [
            {
                "種牡馬": "キズナ",
                "勝率": 45.5,
                "複勝率": 100.0
            },
            {
                "種牡馬": "モーリス",
                "勝率": 45.3,
                "複勝率": 100.0
            },
            {
                "種牡馬": "ダイワメジャー",
                "勝率": 39.6,
                "複勝率": 100.0
            }
        ],
        "好調血統_母父": [
            {
                "母父馬": "ハーツクライ",
                "勝率": 44.4,
                "複勝率": 100.0
            },
            {
                "母父馬": "マンハッタンカフェ",
                "勝率": 42.9,
                "複勝率": 100.0
            },
            {
                "母父馬": "サクラバクシンオー",
                "勝率": 36.6,
                "複勝率": 100.0
            }
        ],
        "配当傾向": {
            "単勝中央値": 470,
            "馬連中央値": 1670,
            "三連複中央値": 4520,
            "荒れ率": 5.0
        }
    },
    "福島": {
        "好調枠番": [
            "3枠",
            "2枠",
            "1枠"
        ],
        "脚質ランク": [
            {
                "脚質": "逃げ",
                "勝率": 47.0,
                "複勝率": 100.0
            },
            {
                "脚質": "先行",
                "勝率": 34.7,
                "複勝率": 100.0
            },
            {
                "脚質": "ﾏｸﾘ",
                "勝率": 32.1,
                "複勝率": 100.0
            },
            {
                "脚質": "中団",
                "勝率": 25.7,
                "複勝率": 100.0
            },
            {
                "脚質": "後方",
                "勝率": 21.7,
                "複勝率": 100.0
            }
        ],
        "好調騎手": [
            {
                "騎手": "荻野極",
                "勝率": 45.1,
                "複勝率": 100.0
            },
            {
                "騎手": "戸崎圭太",
                "勝率": 43.8,
                "複勝率": 100.0
            },
            {
                "騎手": "田辺裕信",
                "勝率": 42.4,
                "複勝率": 100.0
            }
        ],
        "好調調教師": [
            {
                "調教師": "手塚貴久",
                "勝率": 47.9,
                "複勝率": 100.0
            },
            {
                "調教師": "斎藤誠",
                "勝率": 45.4,
                "複勝率": 100.0
            },
            {
                "調教師": "栗田徹",
                "勝率": 43.9,
                "複勝率": 100.0
            }
        ],
        "好調血統_父": [
            {
                "種牡馬": "マンハッタンカフェ",
                "勝率": 49.1,
                "複勝率": 100.0
            },
            {
                "種牡馬": "ブラックタイド",
                "勝率": 42.3,
                "複勝率": 100.0
            },
            {
                "種牡馬": "スクリーンヒーロー",
                "勝率": 42.2,
                "複勝率": 100.0
            }
        ],
        "好調血統_母父": [
            {
                "母父馬": "ハーツクライ",
                "勝率": 43.0,
                "複勝率": 100.0
            },
            {
                "母父馬": "ダイワメジャー",
                "勝率": 42.0,
                "複勝率": 100.0
            },
            {
                "母父馬": "ゼンノロブロイ",
                "勝率": 41.8,
                "複勝率": 100.0
            }
        ],
        "配当傾向": {
            "単勝中央値": 530,
            "馬連中央値": 2250,
            "三連複中央値": 6350,
            "荒れ率": 6.9
        }
    },
    "新潟": {
        "好調枠番": [
            "8枠",
            "5枠",
            "6枠"
        ],
        "脚質ランク": [
            {
                "脚質": "逃げ",
                "勝率": 46.4,
                "複勝率": 100.0
            },
            {
                "脚質": "ﾏｸﾘ",
                "勝率": 35.5,
                "複勝率": 100.0
            },
            {
                "脚質": "先行",
                "勝率": 34.8,
                "複勝率": 100.0
            },
            {
                "脚質": "中団",
                "勝率": 27.4,
                "複勝率": 100.0
            },
            {
                "脚質": "後方",
                "勝率": 25.6,
                "複勝率": 100.0
            }
        ],
        "好調騎手": [
            {
                "騎手": "ルメール",
                "勝率": 49.6,
                "複勝率": 100.0
            },
            {
                "騎手": "福永祐一",
                "勝率": 44.1,
                "複勝率": 100.0
            },
            {
                "騎手": "戸崎圭太",
                "勝率": 42.9,
                "複勝率": 100.0
            }
        ],
        "好調調教師": [
            {
                "調教師": "藤原英昭",
                "勝率": 49.4,
                "複勝率": 100.0
            },
            {
                "調教師": "中内田充",
                "勝率": 48.1,
                "複勝率": 100.0
            },
            {
                "調教師": "牧浦充徳",
                "勝率": 46.8,
                "複勝率": 100.0
            }
        ],
        "好調血統_父": [
            {
                "種牡馬": "モーリス",
                "勝率": 45.3,
                "複勝率": 100.0
            },
            {
                "種牡馬": "ゴールドアリュール",
                "勝率": 44.9,
                "複勝率": 100.0
            },
            {
                "種牡馬": "ネオユニヴァース",
                "勝率": 40.3,
                "複勝率": 100.0
            }
        ],
        "好調血統_母父": [
            {
                "母父馬": "アフリート",
                "勝率": 50.8,
                "複勝率": 100.0
            },
            {
                "母父馬": "Storm Cat",
                "勝率": 44.6,
                "複勝率": 100.0
            },
            {
                "母父馬": "ステイゴールド",
                "勝率": 42.0,
                "複勝率": 100.0
            }
        ],
        "配当傾向": {
            "単勝中央値": 490,
            "馬連中央値": 2140,
            "三連複中央値": 5950,
            "荒れ率": 7.0
        }
    },
    "東京": {
        "好調枠番": [
            "3枠",
            "1枠",
            "7枠"
        ],
        "脚質ランク": [
            {
                "脚質": "逃げ",
                "勝率": 41.5,
                "複勝率": 100.0
            },
            {
                "脚質": "先行",
                "勝率": 36.0,
                "複勝率": 100.0
            },
            {
                "脚質": "中団",
                "勝率": 30.4,
                "複勝率": 100.0
            },
            {
                "脚質": "後方",
                "勝率": 26.3,
                "複勝率": 100.0
            },
            {
                "脚質": "ﾏｸﾘ",
                "勝率": 22.9,
                "複勝率": 100.0
            }
        ],
        "好調騎手": [
            {
                "騎手": "ルメール",
                "勝率": 48.4,
                "複勝率": 100.0
            },
            {
                "騎手": "モレイラ",
                "勝率": 48.0,
                "複勝率": 100.0
            },
            {
                "騎手": "レーン",
                "勝率": 46.5,
                "複勝率": 100.0
            }
        ],
        "好調調教師": [
            {
                "調教師": "堀宣行",
                "勝率": 50.3,
                "複勝率": 100.0
            },
            {
                "調教師": "田中博康",
                "勝率": 47.1,
                "複勝率": 100.0
            },
            {
                "調教師": "友道康夫",
                "勝率": 46.7,
                "複勝率": 100.0
            }
        ],
        "好調血統_父": [
            {
                "種牡馬": "キタサンブラック",
                "勝率": 46.5,
                "複勝率": 100.0
            },
            {
                "種牡馬": "ダンカーク",
                "勝率": 41.5,
                "複勝率": 100.0
            },
            {
                "種牡馬": "リアルスティール",
                "勝率": 40.3,
                "複勝率": 100.0
            }
        ],
        "好調血統_母父": [
            {
                "母父馬": "Distorted Humor",
                "勝率": 43.1,
                "複勝率": 100.0
            },
            {
                "母父馬": "Unbridled's Song",
                "勝率": 42.1,
                "複勝率": 100.0
            },
            {
                "母父馬": "ハーツクライ",
                "勝率": 41.3,
                "複勝率": 100.0
            }
        ],
        "配当傾向": {
            "単勝中央値": 430,
            "馬連中央値": 1700,
            "三連複中央値": 4410,
            "荒れ率": 5.6
        }
    },
    "中山": {
        "好調枠番": [
            "8枠",
            "2枠",
            "4枠"
        ],
        "脚質ランク": [
            {
                "脚質": "逃げ",
                "勝率": 44.8,
                "複勝率": 100.0
            },
            {
                "脚質": "先行",
                "勝率": 35.8,
                "複勝率": 100.0
            },
            {
                "脚質": "ﾏｸﾘ",
                "勝率": 34.5,
                "複勝率": 100.0
            },
            {
                "脚質": "中団",
                "勝率": 25.8,
                "複勝率": 100.0
            },
            {
                "脚質": "後方",
                "勝率": 22.1,
                "複勝率": 100.0
            }
        ],
        "好調騎手": [
            {
                "騎手": "ルメール",
                "勝率": 49.5,
                "複勝率": 100.0
            },
            {
                "騎手": "川田将雅",
                "勝率": 47.1,
                "複勝率": 100.0
            },
            {
                "騎手": "Ｍ．デム",
                "勝率": 42.3,
                "複勝率": 100.0
            }
        ],
        "好調調教師": [
            {
                "調教師": "萩原清",
                "勝率": 49.6,
                "複勝率": 100.0
            },
            {
                "調教師": "堀宣行",
                "勝率": 48.7,
                "複勝率": 100.0
            },
            {
                "調教師": "宮田敬介",
                "勝率": 47.2,
                "複勝率": 100.0
            }
        ],
        "好調血統_父": [
            {
                "種牡馬": "キタサンブラック",
                "勝率": 47.6,
                "複勝率": 100.0
            },
            {
                "種牡馬": "タートルボウル",
                "勝率": 44.4,
                "複勝率": 100.0
            },
            {
                "種牡馬": "レイデオロ",
                "勝率": 43.3,
                "複勝率": 100.0
            }
        ],
        "好調血統_母父": [
            {
                "母父馬": "ホワイトマズル",
                "勝率": 46.1,
                "複勝率": 100.0
            },
            {
                "母父馬": "エンパイアメーカー",
                "勝率": 44.7,
                "複勝率": 100.0
            },
            {
                "母父馬": "ウォーエンブレム",
                "勝率": 43.3,
                "複勝率": 100.0
            }
        ],
        "配当傾向": {
            "単勝中央値": 470,
            "馬連中央値": 1790,
            "三連複中央値": 5090,
            "荒れ率": 6.4
        }
    },
    "中京": {
        "好調枠番": [
            "1枠",
            "4枠",
            "6枠"
        ],
        "脚質ランク": [
            {
                "脚質": "逃げ",
                "勝率": 42.7,
                "複勝率": 100.0
            },
            {
                "脚質": "先行",
                "勝率": 36.3,
                "複勝率": 100.0
            },
            {
                "脚質": "ﾏｸﾘ",
                "勝率": 31.7,
                "複勝率": 100.0
            },
            {
                "脚質": "中団",
                "勝率": 28.0,
                "複勝率": 100.0
            },
            {
                "脚質": "後方",
                "勝率": 24.1,
                "複勝率": 100.0
            }
        ],
        "好調騎手": [
            {
                "騎手": "川田将雅",
                "勝率": 46.5,
                "複勝率": 100.0
            },
            {
                "騎手": "福永祐一",
                "勝率": 42.7,
                "複勝率": 100.0
            },
            {
                "騎手": "ルメール",
                "勝率": 42.5,
                "複勝率": 100.0
            }
        ],
        "好調調教師": [
            {
                "調教師": "中内田充",
                "勝率": 50.0,
                "複勝率": 100.0
            },
            {
                "調教師": "斉藤崇史",
                "勝率": 45.7,
                "複勝率": 100.0
            },
            {
                "調教師": "大橋勇樹",
                "勝率": 43.6,
                "複勝率": 100.0
            }
        ],
        "好調血統_父": [
            {
                "種牡馬": "ロードカナロア",
                "勝率": 41.1,
                "複勝率": 100.0
            },
            {
                "種牡馬": "シニスターミニスター",
                "勝率": 40.6,
                "複勝率": 100.0
            },
            {
                "種牡馬": "エイシンフラッシュ",
                "勝率": 40.3,
                "複勝率": 100.0
            }
        ],
        "好調血統_母父": [
            {
                "母父馬": "Galileo",
                "勝率": 49.3,
                "複勝率": 100.0
            },
            {
                "母父馬": "ホワイトマズル",
                "勝率": 44.3,
                "複勝率": 100.0
            },
            {
                "母父馬": "Kingmambo",
                "勝率": 42.7,
                "複勝率": 100.0
            }
        ],
        "配当傾向": {
            "単勝中央値": 490,
            "馬連中央値": 1890,
            "三連複中央値": 5190,
            "荒れ率": 5.8
        }
    },
    "京都": {
        "好調枠番": [
            "6枠",
            "7枠",
            "4枠"
        ],
        "脚質ランク": [
            {
                "脚質": "逃げ",
                "勝率": 47.5,
                "複勝率": 100.0
            },
            {
                "脚質": "ﾏｸﾘ",
                "勝率": 37.5,
                "複勝率": 100.0
            },
            {
                "脚質": "先行",
                "勝率": 34.5,
                "複勝率": 100.0
            },
            {
                "脚質": "中団",
                "勝率": 28.1,
                "複勝率": 100.0
            },
            {
                "脚質": "後方",
                "勝率": 23.3,
                "複勝率": 100.0
            }
        ],
        "好調騎手": [
            {
                "騎手": "ルメール",
                "勝率": 43.8,
                "複勝率": 100.0
            },
            {
                "騎手": "Ｃ．デム",
                "勝率": 42.6,
                "複勝率": 100.0
            },
            {
                "騎手": "川田将雅",
                "勝率": 42.1,
                "複勝率": 100.0
            }
        ],
        "好調調教師": [
            {
                "調教師": "友道康夫",
                "勝率": 46.3,
                "複勝率": 100.0
            },
            {
                "調教師": "木原一良",
                "勝率": 45.0,
                "複勝率": 100.0
            },
            {
                "調教師": "野中賢二",
                "勝率": 44.2,
                "複勝率": 100.0
            }
        ],
        "好調血統_父": [
            {
                "種牡馬": "イスラボニータ",
                "勝率": 43.4,
                "複勝率": 100.0
            },
            {
                "種牡馬": "ジャスタウェイ",
                "勝率": 40.5,
                "複勝率": 100.0
            },
            {
                "種牡馬": "ロードカナロア",
                "勝率": 40.5,
                "複勝率": 100.0
            }
        ],
        "好調血統_母父": [
            {
                "母父馬": "デヒア",
                "勝率": 45.1,
                "複勝率": 100.0
            },
            {
                "母父馬": "ステイゴールド",
                "勝率": 43.6,
                "複勝率": 100.0
            },
            {
                "母父馬": "アフリート",
                "勝率": 42.7,
                "複勝率": 100.0
            }
        ],
        "配当傾向": {
            "単勝中央値": 470,
            "馬連中央値": 1740,
            "三連複中央値": 4630,
            "荒れ率": 5.5
        }
    },
    "阪神": {
        "好調枠番": [
            "8枠",
            "4枠",
            "5枠"
        ],
        "脚質ランク": [
            {
                "脚質": "逃げ",
                "勝率": 45.1,
                "複勝率": 100.0
            },
            {
                "脚質": "先行",
                "勝率": 35.9,
                "複勝率": 100.0
            },
            {
                "脚質": "ﾏｸﾘ",
                "勝率": 31.4,
                "複勝率": 100.0
            },
            {
                "脚質": "中団",
                "勝率": 27.5,
                "複勝率": 100.0
            },
            {
                "脚質": "後方",
                "勝率": 22.7,
                "複勝率": 100.0
            }
        ],
        "好調騎手": [
            {
                "騎手": "Ｃ．デム",
                "勝率": 50.8,
                "複勝率": 100.0
            },
            {
                "騎手": "川田将雅",
                "勝率": 44.5,
                "複勝率": 100.0
            },
            {
                "騎手": "Ｍ．デム",
                "勝率": 43.1,
                "複勝率": 100.0
            }
        ],
        "好調調教師": [
            {
                "調教師": "角居勝彦",
                "勝率": 47.2,
                "複勝率": 100.0
            },
            {
                "調教師": "中内田充",
                "勝率": 44.4,
                "複勝率": 100.0
            },
            {
                "調教師": "上村洋行",
                "勝率": 43.7,
                "複勝率": 100.0
            }
        ],
        "好調血統_父": [
            {
                "種牡馬": "エスポワールシチー",
                "勝率": 50.8,
                "複勝率": 100.0
            },
            {
                "種牡馬": "ダノンレジェンド",
                "勝率": 43.9,
                "複勝率": 100.0
            },
            {
                "種牡馬": "リアルスティール",
                "勝率": 43.7,
                "複勝率": 100.0
            }
        ],
        "好調血統_母父": [
            {
                "母父馬": "Sadler's Wells",
                "勝率": 50.0,
                "複勝率": 100.0
            },
            {
                "母父馬": "エンドスウィープ",
                "勝率": 47.5,
                "複勝率": 100.0
            },
            {
                "母父馬": "Seeking the Gold",
                "勝率": 47.0,
                "複勝率": 100.0
            }
        ],
        "配当傾向": {
            "単勝中央値": 440,
            "馬連中央値": 1660,
            "三連複中央値": 4450,
            "荒れ率": 5.6
        }
    },
    "小倉": {
        "好調枠番": [
            "5枠",
            "6枠",
            "7枠"
        ],
        "脚質ランク": [
            {
                "脚質": "逃げ",
                "勝率": 48.1,
                "複勝率": 100.0
            },
            {
                "脚質": "ﾏｸﾘ",
                "勝率": 39.0,
                "複勝率": 100.0
            },
            {
                "脚質": "先行",
                "勝率": 34.5,
                "複勝率": 100.0
            },
            {
                "脚質": "中団",
                "勝率": 24.0,
                "複勝率": 100.0
            },
            {
                "脚質": "後方",
                "勝率": 23.8,
                "複勝率": 100.0
            }
        ],
        "好調騎手": [
            {
                "騎手": "川田将雅",
                "勝率": 52.2,
                "複勝率": 100.0
            },
            {
                "騎手": "福永祐一",
                "勝率": 45.9,
                "複勝率": 100.0
            },
            {
                "騎手": "藤岡佑介",
                "勝率": 44.3,
                "複勝率": 100.0
            }
        ],
        "好調調教師": [
            {
                "調教師": "南井克巳",
                "勝率": 45.8,
                "複勝率": 100.0
            },
            {
                "調教師": "中内田充",
                "勝率": 45.5,
                "複勝率": 100.0
            },
            {
                "調教師": "藤原英昭",
                "勝率": 43.6,
                "複勝率": 100.0
            }
        ],
        "好調血統_父": [
            {
                "種牡馬": "キタサンブラック",
                "勝率": 52.6,
                "複勝率": 100.0
            },
            {
                "種牡馬": "ネオユニヴァース",
                "勝率": 49.1,
                "複勝率": 100.0
            },
            {
                "種牡馬": "シニスターミニスター",
                "勝率": 42.2,
                "複勝率": 100.0
            }
        ],
        "好調血統_母父": [
            {
                "母父馬": "Storm Cat",
                "勝率": 47.5,
                "複勝率": 100.0
            },
            {
                "母父馬": "ゴールドアリュール",
                "勝率": 43.1,
                "複勝率": 100.0
            },
            {
                "母父馬": "マンハッタンカフェ",
                "勝率": 39.0,
                "複勝率": 100.0
            }
        ],
        "配当傾向": {
            "単勝中央値": 500,
            "馬連中央値": 2050,
            "三連複中央値": 6070,
            "荒れ率": 6.3
        }
    }
}


# =========================================================
# コース分析パネル
# =========================================================
SMILE_LABEL = {
    "S": "SPRINT (〜1300m)",
    "M": "MILE (1301〜1899m)",
    "I": "INTERMEDIATE (1900〜2100m)",
    "L": "LONG (2101〜2700m)",
    "E": "EXTENDED (2701m〜)",
}
SMILE_ORDER = ["S", "M", "I", "L", "E"]
CLASS_OPTIONS = ["全クラス", "新馬", "未勝利", "1勝", "2勝", "3勝", "OP/重賞"]
SEASON_OPTIONS = ["通年", "春", "夏", "秋", "冬"]

CLASS_NORMALIZE_TREND = {
    "新馬":"新馬","未勝利":"未勝利",
    "1勝":"1勝","500万":"1勝",
    "2勝":"2勝","1000万":"2勝",
    "3勝":"3勝","1600万":"3勝",
    "OP":"OP/重賞","オープン":"OP/重賞",
    "Ｇ１":"OP/重賞","Ｇ２":"OP/重賞","Ｇ３":"OP/重賞","(L)":"OP/重賞",
}


def _smile_from_dist(dist) -> str:
    try:
        d = int(dist)
    except Exception:
        return "M"
    if d <= 1300: return "S"
    elif d <= 1899: return "M"
    elif d <= 2100: return "I"
    elif d <= 2700: return "L"
    else: return "E"


def _render_course_stat(data: dict) -> None:
    """1つのコース傾向ブロックをレンダリング。"""
    if data.get("insufficient"):
        st.markdown(
            f'<div style="color:#555;font-size:15px;padding:8px;border:1px solid #313244;border-radius:6px">'
            f'⚠️ データ不足（{data.get("n",0)}件）　絞り込み条件を緩めてください。</div>',
            unsafe_allow_html=True,
        )
        return

    n = data.get("n", 0)
    st.markdown(
        f'<div style="color:#666;font-size:20px;margin-bottom:8px">サンプル数: {n:,}件</div>',
        unsafe_allow_html=True,
    )

    # 好調枠番
    wakus = data.get("好調枠番", [])
    if wakus:
        waku_html = "".join([
            f'<span style="background:{WAKU_COLORS.get(w,"#555")};'
            f'color:{WAKU_TEXT_COLORS.get(w,"#fff")};'
            f'border-radius:4px;padding:2px 10px;font-size:23px;margin:2px;font-weight:bold">{w}</span>'
            for w in wakus
        ])
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'padding:6px 0;border-bottom:1px solid #2a2a3e">'
            f'<span style="color:#888;font-size:23px;min-width:90px">好調枠番</span>'
            f'<span>{waku_html}</span></div>',
            unsafe_allow_html=True,
        )

    # 脚質ランク
    sas = data.get("脚質ランク", [])
    if sas:
        sas_html = " ＞ ".join([
            f'<span style="color:#f1c40f;font-weight:bold">{s["脚質"]}</span>'
            f'<span style="color:#888;font-size:20px">({s["勝率"]}%)</span>'
            for s in sas
        ])
        st.markdown(
            f'<div style="padding:6px 0;border-bottom:1px solid #2a2a3e">'
            f'<span style="color:#888;font-size:23px">有利脚質</span>'
            f'<div style="margin-top:4px;font-size:23px">{sas_html}</div></div>',
            unsafe_allow_html=True,
        )

    # 好調騎手
    jockeys = data.get("好調騎手", [])
    if jockeys:
        j_rows = "<br>".join([
            f'<span style="color:#4ade80;font-weight:bold;margin-right:4px">{j["騎手"]}</span>'
            f'<span style="color:#888;font-size:20px">{j["勝率"]}% ({j["出走"]}戦)</span>'
            for j in jockeys
        ])
        st.markdown(
            f'<div style="padding:6px 0;border-bottom:1px solid #2a2a3e">'
            f'<span style="color:#888;font-size:23px">好調騎手</span>'
            f'<div style="margin-top:4px;font-size:23px">{j_rows}</div></div>',
            unsafe_allow_html=True,
        )

    # 好調調教師
    trainers = data.get("好調調教師", [])
    if trainers:
        t_rows = "<br>".join([
            f'<span style="color:#cdd6f4;font-weight:bold;margin-right:4px">{t["調教師"]}</span>'
            f'<span style="color:#888;font-size:20px">{t["勝率"]}% ({t["出走"]}戦)</span>'
            for t in trainers
        ])
        st.markdown(
            f'<div style="padding:6px 0;border-bottom:1px solid #2a2a3e">'
            f'<span style="color:#888;font-size:23px">好調調教師</span>'
            f'<div style="margin-top:4px;font-size:23px">{t_rows}</div></div>',
            unsafe_allow_html=True,
        )

    # 好調血統（父）
    sires = data.get("好調血統_父", [])
    if sires:
        s_rows = "<br>".join([
            f'<span style="color:#89b4fa;font-weight:bold;margin-right:4px">{s["種牡馬"]}</span>'
            f'<span style="color:#888;font-size:20px">{s["勝率"]}% ({s["出走"]}戦)</span>'
            for i, s in enumerate(sires)
        ])
        st.markdown(
            f'<div style="padding:6px 0;border-bottom:1px solid #2a2a3e">'
            f'<span style="color:#888;font-size:23px">好調血統(父)</span>'
            f'<div style="margin-top:4px;font-size:23px">{s_rows}</div></div>',
            unsafe_allow_html=True,
        )

    # 好調血統（母父）
    bms = data.get("好調血統_母父", [])
    if bms:
        b_rows = "<br>".join([
            f'<span style="color:#89dceb;font-weight:bold;margin-right:4px">{b["母父馬"]}</span>'
            f'<span style="color:#888;font-size:20px">{b["勝率"]}% ({b["出走"]}戦)</span>'
            for i, b in enumerate(bms)
        ])
        st.markdown(
            f'<div style="padding:6px 0;border-bottom:1px solid #2a2a3e">'
            f'<span style="color:#888;font-size:23px">好調血統(母父)</span>'
            f'<div style="margin-top:4px;font-size:23px">{b_rows}</div></div>',
            unsafe_allow_html=True,
        )

    # 配当傾向
    haitou = data.get("配当", {})
    if haitou:
        are = haitou.get("荒れ率", 0)
        are_color = "#e74c3c" if are >= 8 else "#f39c12" if are >= 6 else "#4ade80"
        st.markdown(
            f'<div style="padding:6px 0">'
            f'<span style="color:#888;font-size:23px">配当傾向</span>'
            f'<div style="margin-top:4px;font-size:23px">'
            f'単勝中央値 <span style="color:#cdd6f4;font-weight:bold">{haitou.get("単勝中央値",0):,}円</span>　'
            f'馬連 <span style="color:#cdd6f4;font-weight:bold">{haitou.get("馬連中央値",0):,}円</span>　'
            f'三連複 <span style="color:#cdd6f4;font-weight:bold">{haitou.get("三連複中央値",0):,}円</span>'
            f'</div><div style="margin-top:2px;font-size:23px">'
            f'荒れ率 <span style="color:{are_color};font-weight:bold">{are}%</span>'
            f'　<span style="color:#666;font-size:20px">(単勝3000円超レースの割合)</span>'
            f'</div></div>',
            unsafe_allow_html=True,
        )


def _render_course_analysis(course_trend: dict, place: str, meta: pd.Series) -> None:
    """コース分析タブ全体をレンダリング。季節のみ選択可、他は自動設定。"""
    dist    = meta.get("距離", 0)
    shida_raw = str(meta.get("芝・ダ", "芝"))
    shida   = "ダ" if shida_raw.startswith("ダ") else "芝"   # "ダート" → "ダ" に正規化
    cls_raw = str(meta.get("クラス名", ""))
    cls_g   = CLASS_NORMALIZE_TREND.get(cls_raw, "OP/重賞")
    sm_auto = _smile_from_dist(dist)

    # ヘッダー
    st.markdown(
        f'<div style="font-size:22px;font-weight:bold;margin-bottom:4px">'
        f'🏇 コース傾向分析</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="display:flex;gap:10px;margin-bottom:8px;flex-wrap:wrap">'
        f'<span style="background:#1e1e2e;border:1px solid #5865f2;border-radius:6px;'
        f'padding:4px 14px;font-size:15px;color:#cdd6f4;font-weight:bold">{place}</span>'
        f'<span style="background:#1e1e2e;border:1px solid #313244;border-radius:6px;'
        f'padding:4px 14px;font-size:15px;color:#89b4fa">{shida}</span>'
        f'<span style="background:#1e1e2e;border:1px solid #313244;border-radius:6px;'
        f'padding:4px 14px;font-size:15px;color:#a6e3a1">{sm_auto} : {SMILE_LABEL.get(sm_auto,"")}</span>'
        f'<span style="background:#1e1e2e;border:1px solid #313244;border-radius:6px;'
        f'padding:4px 14px;font-size:15px;color:#f9e2af">{cls_g}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="color:#666;font-size:13px;margin-bottom:16px">'
        f'過去10年（2016〜2025）の実績データに基づく集計</div>',
        unsafe_allow_html=True,
    )

    # 季節のみ選択
    sel_season = st.selectbox("季節", SEASON_OPTIONS, key="ca_season")

    st.markdown("---")

    # データ取得
    try:
        data = course_trend[place][shida][sm_auto][cls_g][sel_season]
    except KeyError:
        st.warning(f"{place}×{shida}×{sm_auto}×{cls_g}×{sel_season} のデータがありません。")
        return

    _render_course_stat(data)


# =========================================================
# 的中実績ページ
# =========================================================
def _render_plan_results(plan_data: dict, plan_key: str) -> None:
    """HAHO / HALO 共通の実績描画ロジック。"""
    total = plan_data.get("total", {})
    bet   = total.get("bet", 0)
    ret   = total.get("ret", 0)
    pnl   = total.get("pnl", 0)
    roi   = total.get("roi", 0)
    races = total.get("races", 0)

    # サマリーカード
    c1, c2, c3, c4, c5 = st.columns(5)
    roi_color = "#4ade80" if roi >= 100 else "#f39c12" if roi >= 70 else "#e74c3c"
    pnl_color = "#4ade80" if pnl >= 0 else "#e74c3c"
    for col, label, val in [
        (c1, "分析レース数", f"{races}R"),
        (c2, "総投資額",    f"¥{bet:,}"),
        (c3, "総払戻額",    f"¥{ret:,}"),
        (c4, "収支",        f"{'+'if pnl>=0 else ''}¥{pnl:,}"),
        (c5, "ROI",         f"{roi}%"),
    ]:
        color = roi_color if label == "ROI" else pnl_color if label == "収支" else "#cdd6f4"
        col.markdown(
            f'<div style="background:#1e1e2e;border:1px solid #313244;border-radius:8px;'
            f'padding:12px;text-align:center">'
            f'<div style="color:#888;font-size:18px">{label}</div>'
            f'<div style="color:{color};font-size:30px;font-weight:bold;margin-top:4px">{val}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # 馬券種別
    st.markdown("#### 馬券種別成績")
    by_type = plan_data.get("by_type", {})
    if plan_key == "HAHO":
        type_keys = ["馬連", "三連複"]
    elif plan_key == "LALO":
        type_keys = ["複勝"]
    elif plan_key == "CQC":
        type_keys = ["単勝"]
    else:
        type_keys = ["三連複"]
    cols_bt = st.columns(len(type_keys))
    for col, k in zip(cols_bt, type_keys):
        d = by_type.get(k, {})
        if not d:
            continue
        r  = d.get("roi", 0)
        rc = "#4ade80" if r >= 100 else "#f39c12" if r >= 70 else "#e74c3c"
        col.markdown(
            f'<div style="background:#1e1e2e;border:1px solid #313244;border-radius:8px;padding:14px">'
            f'<div style="font-size:24px;font-weight:bold;color:#cdd6f4;margin-bottom:8px">{k}</div>'
            f'<div style="font-size:20px;color:#888">ROI　<span style="color:{rc};font-size:27px;font-weight:bold">{r}%</span></div>'
            f'<div style="font-size:20px;color:#888;margin-top:4px">的中　<span style="color:#cdd6f4">{d.get("hit",0)}/{d.get("races",0)}R ({d.get("hit_rate",0)}%)</span></div>'
            f'<div style="font-size:20px;color:#888;margin-top:4px">払戻　<span style="color:#cdd6f4">¥{d.get("ret",0):,}</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # 週次ROI推移グラフ
    st.markdown("#### 週次ROI推移")
    weekly = plan_data.get("weekly", [])
    if weekly:
        wdf = pd.DataFrame(weekly)
        fig, ax = plt.subplots(figsize=(10, 3))
        fig.patch.set_facecolor("#1e1e2e")
        ax.set_facecolor("#1e1e2e")
        colors = ["#4ade80" if r >= 100 else "#f39c12" if r >= 70 else "#e74c3c"
                  for r in wdf["ROI"]]
        ax.bar(wdf["週"], wdf["ROI"], color=colors, alpha=0.85)
        ax.axhline(100, color="#888", linestyle="--", linewidth=0.8)
        ax.set_ylabel("ROI (%)", color="#888")
        ax.tick_params(colors="#888", labelsize=14)
        ax.spines[:].set_color("#313244")
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(45)
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")

    # 会場別成績
    st.markdown("#### 会場別成績")
    by_place = plan_data.get("by_place", [])
    if by_place:
        for row in by_place:
            r  = float(row.get("ROI", 0))
            rc = "#4ade80" if r >= 100 else "#f39c12" if r >= 70 else "#e74c3c"
            収支 = int(row.get("総払戻", 0)) - int(row.get("総投資", 0))
            pc = "#4ade80" if 収支 >= 0 else "#e74c3c"
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;align-items:center;'
                f'padding:6px 0;border-bottom:1px solid #2a2a3e;font-size:21px">'
                f'<span style="color:#cdd6f4;font-weight:bold;min-width:60px">{row["場所"]}</span>'
                f'<span style="color:#888">{int(row["レース数"])}R</span>'
                f'<span style="color:#888">投資 ¥{int(row["総投資"]):,}</span>'
                f'<span style="color:{pc}">収支 {"+" if 収支>=0 else ""}¥{収支:,}</span>'
                f'<span style="color:{rc};font-weight:bold">ROI {r}%</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # 個別レース結果（日付フィルタ）
    st.markdown("#### 個別レース結果")
    race_list = plan_data.get("races", [])
    if race_list:
        rdf = pd.DataFrame(race_list)
        for _c in ["総投資", "総払戻", "馬連_投資", "馬連_払戻", "三連複_投資", "三連複_払戻"]:
            if _c in rdf.columns:
                rdf[_c] = pd.to_numeric(rdf[_c], errors="coerce").fillna(0)
        if "収支" not in rdf.columns:
            rdf["収支"] = rdf["総払戻"] - rdf["総投資"]
        rdf["収支"] = pd.to_numeric(rdf["収支"], errors="coerce").fillna(0)

        fl1, fl2, fl3 = st.columns(3)
        places    = ["全会場"] + sorted(rdf["場所"].unique().tolist())
        dates     = ["通年"] + sorted(rdf["日付"].unique().tolist(),
                                     key=lambda d: pd.to_datetime(d.replace(".", "/"), errors="coerce"))
        if plan_key == "HAHO":
            filter_opts = ["全馬券", "馬連的中", "三連複的中"]
        elif plan_key == "LALO":
            filter_opts = ["全馬券", "複勝的中"]
        elif plan_key == "CQC":
            filter_opts = ["全馬券", "単勝的中"]
        else:
            filter_opts = ["全馬券", "三連複的中"]
        sel_place = fl1.selectbox("会場", places, key=f"res_place_{plan_key}")
        sel_date  = fl2.selectbox("日付", dates,  key=f"res_date_{plan_key}")
        sel_type  = fl3.selectbox("絞り込み", filter_opts, key=f"res_type_{plan_key}")

        disp = rdf.copy()
        if sel_place != "全会場":
            disp = disp[disp["場所"] == sel_place]
        if sel_date != "通年":
            disp = disp[disp["日付"] == sel_date]
        if sel_type == "馬連的中" and "馬連_的中" in disp.columns:
            disp = disp[disp["馬連_的中"] == 1]
        elif sel_type == "三連複的中" and "三連複_的中" in disp.columns:
            disp = disp[disp["三連複_的中"] == 1]
        elif sel_type == "複勝的中" and "複勝_的中" in disp.columns:
            disp = disp[disp["複勝_的中"] == 1]
        elif sel_type == "単勝的中" and "単勝_的中" in disp.columns:
            disp = disp[disp["単勝_的中"] == 1]

        # 日次サマリー（通年以外の場合）
        if sel_date != "通年" and len(disp) > 0:
            d_bet = int(disp["総投資"].sum())
            d_ret = int(disp["総払戻"].sum())
            d_pnl = d_ret - d_bet
            d_roi = round(d_ret / d_bet * 100, 1) if d_bet > 0 else 0
            rc = "#4ade80" if d_roi >= 100 else "#f39c12" if d_roi >= 70 else "#e74c3c"
            pc = "#4ade80" if d_pnl >= 0 else "#e74c3c"
            st.markdown(
                f'<div style="background:#1e1e2e;border:1px solid #313244;border-radius:8px;'
                f'padding:12px 16px;margin-bottom:12px;display:flex;gap:24px;align-items:center">'
                f'<span style="color:#888;font-size:20px">{sel_date}　{len(disp)}R</span>'
                f'<span style="color:#888;font-size:20px">投資 <b style="color:#cdd6f4">¥{d_bet:,}</b></span>'
                f'<span style="color:#888;font-size:20px">払戻 <b style="color:#cdd6f4">¥{d_ret:,}</b></span>'
                f'<span style="color:#888;font-size:20px">収支 <b style="color:{pc}">{"+"if d_pnl>=0 else ""}¥{d_pnl:,}</b></span>'
                f'<span style="color:#888;font-size:20px">ROI <b style="color:{rc};font-size:24px">{d_roi}%</b></span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        for _, row in disp.sort_values(["日付","R"], ascending=[False,True]).iterrows():
            hits = []
            if "馬連_的中"   in row and row["馬連_的中"]:   hits.append("馬連✅")
            if "三連複_的中" in row and row["三連複_的中"]: hits.append("三連複✅")
            if "複勝_的中"   in row and row["複勝_的中"]:   hits.append("複勝✅")
            if "単勝_的中"   in row and row["単勝_的中"]:   hits.append("単勝✅")
            hit_str = "　".join(hits) if hits else "❌"
            pnl_v   = int(row["収支"])
            rc      = "#4ade80" if pnl_v >= 0 else "#e74c3c"
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;align-items:center;'
                f'padding:8px 0;border-bottom:1px solid #2a2a3e;font-size:21px">'
                f'<span style="color:#888;min-width:80px">{row["日付"]}</span>'
                f'<span style="color:#cdd6f4;font-weight:bold;min-width:90px">{row["場所"]} {row["R"]}R</span>'
                f'<span style="color:#888;min-width:100px">{row["クラス"]}</span>'
                f'<span style="min-width:150px">{hit_str}</span>'
                f'<span style="color:#888">¥{int(row["総投資"]):,}</span>'
                f'<span style="color:{rc};font-weight:bold;margin-left:12px">{"+"if pnl_v>=0 else ""}¥{pnl_v:,}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )


def page_results(results: dict) -> None:
    """的中実績ページ（HAHO / HALO タブ）。"""
    if not results:
        st.warning("results.json が見つかりません。data/results.json を配置してください。")
        return

    st.markdown("## 📊 的中実績")
    st.markdown(
        f'<div style="color:#666;font-size:20px;margin-bottom:16px">'
        f'集計期間: {results.get("generated_at","")[:10]} 時点</div>',
        unsafe_allow_html=True,
    )

    # 新フォーマット（HAHO/HALO/LALO/CQC キーあり）
    if "HAHO" in results or "HALO" in results or "LALO" in results or "CQC" in results:
        tab_haho, tab_halo, tab_lalo, tab_cqc = st.tabs([
            "🛡️ HAHO  安定積み上げ",
            "🎯 HALO  高配当特化",
            "🍀 LALO  コツコツ複勝",
            "⚔️ CQC   孤高の真髄",
        ])
        with tab_haho:
            haho_data = results.get("HAHO", {})
            if not haho_data or not haho_data.get("total", {}).get("races"):
                st.info("HAHOのデータがありません。generate_results.py を実行してください。")
            else:
                _render_plan_results(haho_data, "HAHO")
        with tab_halo:
            halo_data = results.get("HALO", {})
            if not halo_data or not halo_data.get("total", {}).get("races"):
                st.info("HALOのデータがありません。generate_results.py を実行してください。")
            else:
                _render_plan_results(halo_data, "HALO")
        with tab_lalo:
            lalo_data = results.get("LALO", {})
            if not lalo_data or not lalo_data.get("total", {}).get("races"):
                st.info("LALOのデータがありません。generate_results.py を実行してください。")
            else:
                _render_plan_results(lalo_data, "LALO")
        with tab_cqc:
            cqc_data = results.get("CQC", {})
            if not cqc_data or not cqc_data.get("total", {}).get("races"):
                st.info("CQCのデータがありません。generate_results.py を実行してください。")
            else:
                _render_plan_results(cqc_data, "CQC")
    else:
        # 旧フォーマット（後方互換）
        st.info("旧フォーマットのresults.jsonです。generate_results.py を再実行するとHAHO/HALO/LALO/CQCタブが表示されます。")
        total = results.get("total", {})
        bet   = total.get("bet", 0)
        ret   = total.get("ret", 0)
        pnl   = total.get("pnl", 0)
        roi   = total.get("roi", 0)
        races = total.get("races", 0)
        c1, c2, c3, c4, c5 = st.columns(5)
        roi_color = "#4ade80" if roi >= 100 else "#f39c12" if roi >= 70 else "#e74c3c"
        pnl_color = "#4ade80" if pnl >= 0 else "#e74c3c"
        for col, label, val in [
            (c1, "分析レース数", f"{races}R"),
            (c2, "総投資額",    f"¥{bet:,}"),
            (c3, "総払戻額",    f"¥{ret:,}"),
            (c4, "収支",        f"{'+'if pnl>=0 else ''}¥{pnl:,}"),
            (c5, "ROI",         f"{roi}%"),
        ]:
            color = roi_color if label == "ROI" else pnl_color if label == "収支" else "#cdd6f4"
            col.markdown(
                f'<div style="background:#1e1e2e;border:1px solid #313244;border-radius:8px;'
                f'padding:12px;text-align:center">'
                f'<div style="color:#888;font-size:18px">{label}</div>'
                f'<div style="color:{color};font-size:30px;font-weight:bold;margin-top:4px">{val}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# =========================================================
# 展開予想図
# =========================================================
def classify_pace_style(row: pd.Series, n_horses: int) -> str:
    """前走通過1位から脚質を推定。"""
    try:
        p1 = float(row.get("前走通過1", None))
        if pd.isna(p1) or n_horses <= 0:
            return "不明"
        ratio = p1 / n_horses
        if ratio <= 0.2:
            return "逃げ"
        elif ratio <= 0.45:
            return "先行"
        elif ratio <= 0.7:
            return "中団"
        else:
            return "後方"
    except Exception:
        return "不明"


def render_pace_scenario(race_df: pd.DataFrame) -> None:
    """展開予想図をHTMLテーブルで表示。"""
    n = len(race_df)
    if n == 0:
        return

    styles: dict[str, list[str]] = {"逃げ": [], "先行": [], "中団": [], "後方": [], "不明": []}
    for _, row in race_df.iterrows():
        style = classify_pace_style(row, n)
        name  = str(row.get("馬名", ""))
        mark  = str(row.get("mark", ""))
        label = f"{mark}{name}" if mark else name
        styles[style].append(label)

    # 不明は表示しない
    cols = ["逃げ", "先行", "中団", "後方"]
    max_rows = max(len(styles[c]) for c in cols) if any(styles[c] for c in cols) else 0
    if max_rows == 0:
        st.caption("展開予想データなし（前走通過順位が取得できませんでした）")
        return

    col_colors = {
        "逃げ": "#e74c3c", "先行": "#e67e22", "中団": "#3498db", "後方": "#8e44ad"
    }

    # ヘッダー
    header_html = "".join(
        f'<th style="color:{col_colors[c]};font-size:16px;padding:8px 16px;text-align:center;'
        f'border-bottom:2px solid {col_colors[c]}">{c}</th>'
        for c in cols
    )

    # 行
    rows_html = ""
    for i in range(max_rows):
        row_html = ""
        for c in cols:
            horses = styles[c]
            if i < len(horses):
                name = horses[i]
                mark = name[0] if name and name[0] in "◎◯▲△×" else ""
                rest = name[1:] if mark else name
                mc   = {"◎":"#f1c40f","◯":"#3498db","▲":"#e74c3c","△":"#2ecc71","×":"#888"}.get(mark,"#cdd6f4")
                cell = (f'<span style="color:{mc};font-weight:bold">{mark}</span>'
                        f'<span style="color:#cdd6f4">{rest}</span>') if mark else f'<span style="color:#cdd6f4">{rest}</span>'
            else:
                cell = ""
            row_html += f'<td style="padding:4px 16px;text-align:center;font-size:14px">{cell}</td>'
        rows_html += f"<tr>{row_html}</tr>"

    html = (
        '<div style="margin:8px 0 16px">' 
        '<div style="font-size:13px;color:#888;margin-bottom:6px">※前走通過順位から自動推定</div>'
        '<table style="border-collapse:collapse;width:100%">'
        f"<thead><tr>{header_html}</tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        "</table></div>"
    )
    st.markdown("#### 展開予想図")
    st.markdown(html, unsafe_allow_html=True)



# =========================================================
# 今日の買い目専用ページ
# =========================================================
def page_buylist(all_df: pd.DataFrame, strategy: dict, budget: int) -> None:
    """今日の買い目一覧ページ（HAHO/HALO プラン切替）。"""
    race_id_col = "レースID(新/馬番無)"
    race_metas  = []
    for race_id, grp in all_df.groupby(race_id_col):
        meta    = grp.iloc[0]
        place   = str(meta.get("場所", ""))
        cls_raw = str(meta.get("クラス名", ""))
        hon_row = grp[grp["mark"] == "◎"]
        hon_name  = str(hon_row.iloc[0]["馬名"])  if not hon_row.empty else "-"
        hon_score = float(hon_row.iloc[0]["score"]) if not hon_row.empty else 0.0
        bets_all  = get_bets(grp, place, cls_raw, strategy, budget)
        if not bets_all:
            continue
        race_metas.append({
            "race_id":   race_id,
            "場所":      place,
            "R":         int(meta.get("R", 0)),
            "クラス":    cls_raw,
            "発走":      str(meta.get("発走時刻", "")),
            "距離":      f'{meta.get("芝・ダ","")}{meta.get("距離","")}m',
            "馬場":      _normalize_baba(str(meta.get("馬場状態",""))),
            "◎":         hon_name,
            "◎スコア":   hon_score,
            "HAHO_bets": bets_all.get("HAHO", []),
            "HALO_bets": bets_all.get("HALO", []),
            "LALO_bets": bets_all.get("LALO", []),
            "CQC_bets":  bets_all.get("CQC",  []),
        })

    if not race_metas:
        st.info("本日の買い目対象レースがありません。")
        return

    # プラン選択
    plan = st.radio(
        "プラン選択",
        ["🛡️ HAHO  安定積み上げ（馬連◎軸2点 ＋ 三連複1点）",
         "🎯 HALO  高配当特化（三連複ボックス1点のみ）",
         "🍀 LALO  コツコツ複勝（複勝◎1点のみ）",
         "⚔️ CQC   孤高の真髄（単勝◎1点のみ）"],
        horizontal=True, key="buylist_plan",
    )
    if plan.startswith("🛡️"):
        plan_key = "HAHO_bets"
    elif plan.startswith("🎯"):
        plan_key = "HALO_bets"
    elif plan.startswith("🍀"):
        plan_key = "LALO_bets"
    else:
        plan_key = "CQC_bets"

    # 馬場フィルタ
    all_babas    = sorted({r["馬場"] for r in race_metas if r["馬場"]})
    baba_options = ["全馬場"] + all_babas
    baba_filter  = st.selectbox("🏟 馬場フィルタ", baba_options, key="buylist_baba")

    active = [r for r in race_metas if r[plan_key]]
    if baba_filter != "全馬場":
        active = [r for r in active if r["馬場"] == baba_filter]
    if not active:
        st.info("このプランの対象レースがありません。")
        return

    total_all = sum(sum(b["購入額"] for b in r[plan_key]) for r in active)
    st.markdown(
        f'<div style="background:#1e1e2e;border:1px solid #313244;border-radius:8px;'
        f'padding:12px 16px;margin-bottom:16px;display:flex;gap:32px;align-items:center">'
        f'<span style="color:#888">対象レース <b style="color:#cdd6f4;font-size:18px">{len(active)}R</b></span>'
        f'<span style="color:#888">合計投資予定 <b style="color:#cdd6f4;font-size:18px">¥{total_all:,}</b></span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    for r in sorted(active, key=lambda x: (x["場所"], x["R"])):
        bets  = r[plan_key]
        rengo = [b for b in bets if b["馬券種"] == "馬連"]
        sanf  = [b for b in bets if b["馬券種"] == "三連複"]
        fuku  = [b for b in bets if b["馬券種"] == "複勝"]
        tan   = [b for b in bets if b["馬券種"] == "単勝"]

        # ヘッダー行
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:12px;'
            f'padding:10px 0 4px;border-top:1px solid #2a2a3e;margin-top:4px">'
            f'<span style="background:#313244;color:#cdd6f4;border-radius:4px;'
            f'padding:2px 10px;font-weight:bold;font-size:15px">{r["場所"]} {r["R"]}R</span>'
            f'<span style="color:#888;font-size:13px">{r["クラス"]}　{r["発走"]}　{r["距離"]}　馬場:{r["馬場"]}</span>'
            f'<span style="color:#a6e3a1;font-size:13px;margin-left:auto">'
            f'◎{r["◎"]}　{r["◎スコア"]:.1f}%</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # 買い目行
        lines = []
        if rengo:
            combos = "　".join(b["買い目"] for b in rengo)
            amt    = sum(b["購入額"] for b in rengo)
            lines.append(
                f'<span style="color:#888;min-width:50px;display:inline-block">馬連</span>'
                f'<span style="color:#cdd6f4">{combos}</span>'
                f'<span style="color:#888;margin-left:8px">¥{amt:,}</span>'
                f'<span style="color:#f39c12;font-size:12px;margin-left:8px">ROI目安{rengo[0]["ROI"]:.0f}%</span>'
            )
        if sanf:
            combos = "　".join(b["買い目"] for b in sanf)
            amt    = sum(b["購入額"] for b in sanf)
            lines.append(
                f'<span style="color:#888;min-width:50px;display:inline-block">三連複</span>'
                f'<span style="color:#cdd6f4">{combos}</span>'
                f'<span style="color:#888;margin-left:8px">¥{amt:,}</span>'
                f'<span style="color:#f39c12;font-size:12px;margin-left:8px">ROI目安{sanf[0]["ROI"]:.0f}%</span>'
            )
        if fuku:
            combos = "　".join(b["買い目"] for b in fuku)
            amt    = sum(b["購入額"] for b in fuku)
            lines.append(
                f'<span style="color:#888;min-width:50px;display:inline-block">複勝</span>'
                f'<span style="color:#cdd6f4">◎ {combos}番</span>'
                f'<span style="color:#888;margin-left:8px">¥{amt:,}</span>'
                f'<span style="color:#f39c12;font-size:12px;margin-left:8px">ROI目安{fuku[0]["ROI"]:.0f}%</span>'
            )
        if tan:
            combos = "　".join(b["買い目"] for b in tan)
            amt    = sum(b["購入額"] for b in tan)
            lines.append(
                f'<span style="color:#888;min-width:50px;display:inline-block">単勝</span>'
                f'<span style="color:#cdd6f4">◎ {combos}番</span>'
                f'<span style="color:#888;margin-left:8px">¥{amt:,}</span>'
                f'<span style="color:#f39c12;font-size:12px;margin-left:8px">ROI目安{tan[0]["ROI"]:.0f}%</span>'
            )

        body = "".join(
            f'<div style="padding:3px 0;font-size:14px">{l}</div>' for l in lines
        )
        st.markdown(
            f'<div style="padding:4px 8px 10px">{body}</div>',
            unsafe_allow_html=True,
        )


WAKU_COLORS = {
    "1枠":"#fff","2枠":"#222","3枠":"#e74c3c",
    "4枠":"#3498db","5枠":"#f1c40f","6枠":"#27ae60",
    "7枠":"#e67e22","8枠":"#ff69b4",
}

WAKU_TEXT_COLORS = {
    "1枠":"#000","2枠":"#fff","3枠":"#fff",
    "4枠":"#fff","5枠":"#000","6枠":"#fff",
    "7枠":"#fff","8枠":"#000",
}


def render_trend_panel(places: list[str]) -> None:
    st.markdown("#### 📊 開催場傾向（過去10年）")
    for place in places:
        data = TREND_DATA.get(place)
        if not data:
            continue
        st.markdown(f"**{place}**")

        # 好調枠番
        wakus = data.get("好調枠番", [])
        waku_html = "".join([
            f'<span style="background:{WAKU_COLORS.get(w,"#555")};'
            f'color:{WAKU_TEXT_COLORS.get(w,"#fff")};'
            f'border-radius:4px;padding:1px 8px;font-size:18px;margin:2px;font-weight:bold">{w}</span>'
            for w in wakus
        ])
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;padding:4px 0;'
            f'border-bottom:1px solid #2a2a3e;font-size:20px">'
            f'<span style="color:#888">好調枠番</span><span>{waku_html}</span></div>',
            unsafe_allow_html=True,
        )

        # 有利脚質（ランク表示）
        sas = data.get("脚質ランク", [])
        if sas:
            top = sas[0]
            sas_html = " > ".join([f'{s["脚質"]}({s["勝率"]}%)' for s in sas])
            st.markdown(
                f'<div style="padding:4px 0;border-bottom:1px solid #2a2a3e;font-size:20px">'
                f'<span style="color:#888">有利脚質</span>'
                f'<div style="color:#f1c40f;font-size:17px;margin-top:2px">{sas_html}</div></div>',
                unsafe_allow_html=True,
            )

        # 好調騎手TOP3
        jockeys = data.get("好調騎手", [])
        if jockeys:
            j_html = "<br>".join([f'{j["騎手"]}<span style="color:#888;font-size:15px">({j["勝率"]}%)</span>' for j in jockeys])
            st.markdown(
                f'<div style="padding:4px 0;border-bottom:1px solid #2a2a3e;font-size:20px">'
                f'<span style="color:#888">好調騎手</span>'
                f'<div style="color:#4ade80;font-size:18px;margin-top:2px">{j_html}</div></div>',
                unsafe_allow_html=True,
            )

        # 好調調教師TOP3
        trainers = data.get("好調調教師", [])
        if trainers:
            t_html = "<br>".join([f'{t["調教師"]}<span style="color:#888;font-size:15px">({t["勝率"]}%)</span>' for t in trainers])
            st.markdown(
                f'<div style="padding:4px 0;border-bottom:1px solid #2a2a3e;font-size:20px">'
                f'<span style="color:#888">好調調教師</span>'
                f'<div style="color:#cdd6f4;font-size:18px;margin-top:2px">{t_html}</div></div>',
                unsafe_allow_html=True,
            )

        # 好調血統（父）TOP3
        sires = data.get("好調血統_父", [])
        if sires:
            s_html = "<br>".join([f'{s["種牡馬"]}<span style="color:#888;font-size:15px">({s["勝率"]}%)</span>' for s in sires])
            st.markdown(
                f'<div style="padding:4px 0;border-bottom:1px solid #2a2a3e;font-size:20px">'
                f'<span style="color:#888">好調血統(父)</span>'
                f'<div style="color:#89b4fa;font-size:18px;margin-top:2px">{s_html}</div></div>',
                unsafe_allow_html=True,
            )

        # 好調血統（母父）TOP3
        bms = data.get("好調血統_母父", [])
        if bms:
            b_html = "<br>".join([f'{b["母父馬"]}<span style="color:#888;font-size:15px">({b["勝率"]}%)</span>' for b in bms])
            st.markdown(
                f'<div style="padding:4px 0;border-bottom:1px solid #2a2a3e;font-size:20px">'
                f'<span style="color:#888">好調血統(母父)</span>'
                f'<div style="color:#89dceb;font-size:18px;margin-top:2px">{b_html}</div></div>',
                unsafe_allow_html=True,
            )

        # 配当傾向
        haitou = data.get("配当傾向", {})
        if haitou:
            are = haitou.get("荒れ率", 0)
            are_color = "#e74c3c" if are >= 8 else "#f39c12" if are >= 6 else "#4ade80"
            st.markdown(
                f'<div style="padding:4px 0;border-bottom:1px solid #2a2a3e;font-size:18px">'
                f'<span style="color:#888">配当傾向</span>'
                f'<div style="margin-top:2px">'
                f'単勝中央値 <span style="color:#cdd6f4">{haitou.get("単勝中央値",0):,}円</span>　'
                f'馬連 <span style="color:#cdd6f4">{haitou.get("馬連中央値",0):,}円</span>　'
                f'三連複 <span style="color:#cdd6f4">{haitou.get("三連複中央値",0):,}円</span>'
                f'</div><div style="margin-top:2px">'
                f'荒れ率 <span style="color:{are_color};font-weight:bold">{are}%</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )
        st.markdown("")


# =========================================================
# CSS
# =========================================================
CSS = """
<style>
.tbl-header {
    display:grid;
    grid-template-columns:36px 44px 44px 1fr 64px 56px 120px 150px;
    background:#1e1e2e; color:#cdd6f4; font-weight:bold;
    font-size:13px; padding:6px 12px; border-radius:6px 6px 0 0; gap:8px;
}
.tbl-row {
    display:grid;
    grid-template-columns:36px 44px 44px 1fr 64px 56px 120px 150px;
    font-size:13px; padding:6px 12px;
    border-bottom:1px solid #313244; align-items:center; gap:8px;
}
.mk-hon  {color:#e74c3c;font-weight:bold;font-size:16px;}
.mk-tai  {color:#e67e22;font-weight:bold;font-size:16px;}
.mk-sabo {color:#f1c40f;font-weight:bold;}
.mk-del  {color:#2ecc71;}
.mk-batu {color:#95a5a6;}
.sbar-wrap{display:flex;align-items:center;gap:6px;font-size:12px;}
.sbar{height:10px;border-radius:4px;background:#5865f2;}
.main-race-banner {
    background:linear-gradient(135deg,#1a1a2e 0%,#16213e 50%,#0f3460 100%);
    border:1px solid #e74c3c; border-radius:12px;
    padding:18px 24px; margin-bottom:16px; position:relative; overflow:hidden;
}
.main-race-banner::before {
    content:''; position:absolute; top:0; left:0; right:0; height:3px;
    background:linear-gradient(90deg,#e74c3c,#f39c12,#e74c3c);
}
.grade-g1{display:inline-block;background:#e74c3c;color:#fff;padding:2px 10px;border-radius:4px;font-size:12px;font-weight:bold;margin-right:8px;}
.grade-g2{display:inline-block;background:#9b59b6;color:#fff;padding:2px 10px;border-radius:4px;font-size:12px;font-weight:bold;margin-right:8px;}
.grade-g3{display:inline-block;background:#2980b9;color:#fff;padding:2px 10px;border-radius:4px;font-size:12px;font-weight:bold;margin-right:8px;}
.grade-op{display:inline-block;background:#27ae60;color:#fff;padding:2px 10px;border-radius:4px;font-size:12px;font-weight:bold;margin-right:8px;}
.race-row {
    display:flex; align-items:center; gap:10px;
    padding:8px 6px; border-bottom:1px solid #2a2a3e;
}
.r-badge {
    background:#e74c3c;color:#fff;border-radius:5px;
    padding:2px 8px;font-size:15px;font-weight:bold;min-width:32px;text-align:center;
}
.r-badge-ex {
    background:#444;color:#888;border-radius:5px;
    padding:2px 8px;font-size:15px;min-width:32px;text-align:center;
}
.bet-card {
    background:#1e1e2e;border:1px solid #313244;
    border-radius:8px;padding:12px 16px;margin:6px 0;
}
</style>
"""


# =========================================================
# レース一覧ページ
# =========================================================
def page_race_list(all_df: pd.DataFrame, strategy: dict, budget: int) -> None:
    race_id_col = "レースID(新/馬番無)"

    race_metas: list[dict] = []
    for race_id, grp in all_df.groupby(race_id_col):
        meta    = grp.iloc[0]
        place   = str(meta.get("場所",""))
        cls_raw = str(meta.get("クラス名",""))
        hon_row = grp[grp["mark"] == "◎"]
        hon_name  = str(hon_row.iloc[0]["馬名"])  if not hon_row.empty else "-"
        hon_score = float(hon_row.iloc[0]["score"]) if not hon_row.empty else 0.0
        race_metas.append({
            "race_id":  race_id,
            "場所":     place,
            "R":        int(meta.get("R", 0)),
            "クラス":   cls_raw,
            "レース名": str(meta.get("レース名","")),
            "距離":     f'{meta.get("芝・ダ","")}{meta.get("距離","")}m',
            "発走":     str(meta.get("発走時刻","")),
            "天気":     str(meta.get("天気","")),
            "馬場":     _normalize_baba(str(meta.get("馬場状態",""))),
            "頭数":     len(grp),
            "◎":        hon_name,
            "◎スコア":  hon_score,
            "戦略":     is_in_strategy(place, cls_raw, strategy),
            "グレード":  GRADE_ORDER.get(CLASS_NORMALIZE.get(cls_raw, cls_raw), 99),
        })

    # 重賞バナー
    graded = sorted([r for r in race_metas if r["グレード"] <= 3], key=lambda x: x["グレード"])
    if graded:
        mr = graded[0]
        grade_label = CLASS_NORMALIZE.get(mr["クラス"], mr["クラス"])
        grade_cls   = {"Ｇ１":"grade-g1","Ｇ２":"grade-g2","Ｇ３":"grade-g3"}.get(grade_label,"grade-op")
        st.markdown(
            f'<div class="main-race-banner">'
            f'<div style="margin-bottom:6px">'
            f'<span class="{grade_cls}">{grade_label}</span>'
            f'<span style="color:#888;font-size:13px">{mr["場所"]} {mr["R"]}R　{mr["発走"]}発走</span></div>'
            f'<div style="font-size:22px;font-weight:bold;color:#cdd6f4;margin-bottom:4px">'
            f'{mr["レース名"] or mr["クラス"]}</div>'
            f'<div style="color:#888;font-size:14px">{mr["距離"]}　{mr["頭数"]}頭立て'
            f'　天気:{mr["天気"]}　馬場:{mr["馬場"]}</div>'
            f'<div style="margin-top:8px;font-size:14px;color:#a6e3a1">'
            f'◎ <b>{mr["◎"]}</b>　スコア {mr["◎スコア"]:.1f}%</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    col_left, col_right = st.columns([3, 1], gap="medium")

    with col_left:
        # 会場タブ
        by_place: dict[str, list] = {}
        for r in race_metas:
            by_place.setdefault(r["場所"], []).append(r)
        places = list(by_place.keys())

        if "selected_place" not in st.session_state or st.session_state.selected_place not in places:
            st.session_state.selected_place = places[0] if places else ""

        tab_cols = st.columns(len(places))
        for tc, place in zip(tab_cols, places):
            with tc:
                btn_type = "primary" if st.session_state.selected_place == place else "secondary"
                if st.button(place, key=f"place_tab_{place}", type=btn_type, use_container_width=True):
                    st.session_state.selected_place = place
                    st.rerun()

        cur_place  = st.session_state.selected_place
        races_here = sorted(by_place.get(cur_place, []), key=lambda x: x["R"])

        if races_here:
            m0 = races_here[0]
            # 芝とダートで馬場が異なる場合を個別表示
            shiba_babas = sorted({r["馬場"] for r in races_here if r["距離"].startswith("芝")})
            dirt_babas  = sorted({r["馬場"] for r in races_here if not r["距離"].startswith("芝")})
            baba_parts  = []
            if shiba_babas: baba_parts.append(f'芝:{"/".join(shiba_babas)}')
            if dirt_babas:  baba_parts.append(f'ダ:{"/".join(dirt_babas)}')
            baba_str = "　".join(baba_parts) if baba_parts else m0["馬場"]
            st.markdown(
                f'<div style="font-size:13px;color:#888;padding:4px 0 8px">'
                f'天気: {m0["天気"]}　馬場: {baba_str}</div>',
                unsafe_allow_html=True,
            )

        # Rボタン行
        r_btn_cols = st.columns(min(len(races_here), 12))
        for rc_col, r in zip(r_btn_cols, races_here):
            with rc_col:
                btn_type = "primary" if r["戦略"] else "secondary"
                if st.button(f'{r["R"]}R', key=f"rbtn_{r['race_id']}", type=btn_type, use_container_width=True):
                    st.session_state.selected_race_id = r["race_id"]
                    st.rerun()

        st.markdown("---")

        # レース行
        for r in races_here:
            excluded = r["場所"] in EXCLUDE_PLACES or r["クラス"] in EXCLUDE_CLASSES
            badge = ""
            if r["戦略"]:
                badge = '<span style="background:#2d4a2d;color:#4ade80;border-radius:3px;padding:1px 6px;font-size:10px;margin-left:4px">✅買い</span>'
            elif excluded:
                badge = '<span style="background:#3a2a1a;color:#f39c12;border-radius:3px;padding:1px 6px;font-size:10px;margin-left:4px">除外</span>'

            st.markdown(
                f'<div class="race-row">'
                f'<span class="{"r-badge" if not excluded else "r-badge-ex"}">{r["R"]}R</span>'
                f'<div style="flex:1">'
                f'<span style="font-size:18px;color:#cdd6f4">{r["クラス"]}</span>{badge}'
                f'<span style="color:#888;font-size:14px;margin-left:8px">'
                f'{r["発走"]}　{r["距離"]}　馬場:{r["馬場"]}　{r["頭数"]}頭</span>'
                f'<br><span style="font-size:14px;color:#a6e3a1">◎ {r["◎"]}　{r["◎スコア"]:.1f}%</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )
            if st.button("詳細→", key=f'btn_{r["race_id"]}'):
                st.session_state.selected_race_id = r["race_id"]
                st.rerun()

    with col_right:
        # ねらい目
        targets = sorted([r for r in race_metas if r["戦略"]], key=lambda x: -x["◎スコア"])[:5]
        if targets:
            st.markdown("#### 🎯 今日のねらい目")
            for t in targets:
                label = f'{t["場所"]} {t["R"]}R {t["クラス"]}\n◎{t["◎"]}　{t["◎スコア"]:.1f}%'
                if st.button(label, key=f'target_{t["race_id"]}', use_container_width=True):
                    st.session_state.selected_race_id = t["race_id"]
                    st.rerun()
            st.markdown("---")

        active_places = [p for p in places if p not in EXCLUDE_PLACES]
        render_trend_panel(active_places)


# =========================================================
# 出走表ページ
# =========================================================
def page_race_detail(
    race_df: pd.DataFrame,
    all_df: pd.DataFrame,
    strategy: dict,
    budget: int,
    lgbm_obj: dict,
    course_trend: dict | None = None,
) -> None:
    meta    = race_df.iloc[0]
    place   = str(meta.get("場所",""))
    cls_raw = str(meta.get("クラス名",""))
    dist    = meta.get("距離","")
    shida   = meta.get("芝・ダ","")
    r_num   = meta.get("R","")

    in_strategy = is_in_strategy(place, cls_raw, strategy)

    col_main, col_nav = st.columns([3, 1], gap="medium")

    with col_main:
        if st.button("← レース一覧に戻る"):
            st.session_state.selected_race_id = None
            st.rerun()

        baba_raw = str(meta.get("馬場状態",""))
        baba_disp = _normalize_baba(baba_raw)
        st.markdown(f"## {place} {r_num}R / {cls_raw} / {shida}{dist}m　🏟 馬場:{baba_disp}")

        if in_strategy:
            cls_norm = CLASS_NORMALIZE.get(cls_raw, cls_raw)
            cls_key  = cls_norm if cls_norm in strategy.get(place,{}) else cls_raw
            roi_vals = [v.get("roi_oos", v.get("roi", 0)) for v in strategy[place][cls_key].values()]
            st.success(f"✅ 戦略対象レース　平均ROI: {sum(roi_vals)/len(roi_vals):.1f}%")
        elif place in EXCLUDE_PLACES:
            st.warning(f"⚠️ {place}は除外会場（参考予想）")
        elif cls_raw in EXCLUDE_CLASSES:
            st.warning(f"⚠️ {cls_raw}は除外クラス（参考予想）")
        else:
            st.info("ℹ️ 戦略対象外（参考予想）")

        shap_ok = False; shap_vals: list = []; feature_cols: list = []
        with st.spinner("SHAP計算中..."):
            try:
                shap_vals, feature_cols = compute_shap(lgbm_obj, race_df.to_json())
                shap_ok = True
            except Exception as e:
                st.warning(f"SHAP計算失敗: {e}")

        tab1, tab2, tab3 = st.tabs(["📋 出走表 / 買い目", "🔍 全頭分析", "🏇 コース分析"])

        with tab1:
            st.markdown("### 出走表")
            st.markdown(
                '<div class="tbl-header">'
                '<span>枠</span><span>馬番</span><span>印</span>'
                '<span>馬名</span><span>性齢</span><span>斤量</span>'
                '<span>騎手</span><span>スコア</span>'
                '</div>', unsafe_allow_html=True,
            )
            for _, row in race_df.sort_values("馬番").iterrows():
                mark   = row.get("mark","")
                ban    = int(row.get("馬番",0))
                waku   = int(row.get("枠番",0))
                name   = str(row.get("馬名", f"{ban}番"))
                seire  = str(row.get("性別","")) + str(row.get("年齢",""))
                kin    = str(row.get("斤量",""))
                jockey = str(row.get("騎手",""))
                score  = float(row.get("score", 0))
                mk_cls  = MARK_CLASS.get(mark,"")
                mk_html = f'<span class="{mk_cls}">{mark}</span>' if mark else ""
                bar_w   = min(int(score * 1.2), 120)
                score_html = (
                    f'<div class="sbar-wrap">'
                    f'<div class="sbar" style="width:{bar_w}px"></div>'
                    f'<span>{score:.1f}%</span></div>'
                )
                st.markdown(
                    f'<div class="tbl-row">'
                    f'<span>{waku}</span><span>{ban}</span>'
                    f'<span>{mk_html}</span><span>{name}</span>'
                    f'<span>{seire}</span><span>{kin}</span>'
                    f'<span>{jockey}</span>{score_html}'
                    f'</div>', unsafe_allow_html=True,
                )

            if in_strategy:
                st.markdown("---")
                st.markdown("### 🎯 買い目")
                bets_all = get_bets(race_df, place, cls_raw, strategy, budget)
                if not bets_all:
                    st.warning("買い目を生成できませんでした。")
                else:
                    det_tab_haho, det_tab_halo, det_tab_lalo, det_tab_cqc = st.tabs([
                        "🛡️ HAHO  安定積み上げ",
                        "🎯 HALO  高配当特化",
                        "🍀 LALO  コツコツ複勝",
                        "⚔️ CQC   孤高の真髄",
                    ])
                    _no_bet_msg = {
                        "HAHO": "このレースはHAHO戦略の対象外です。",
                        "HALO": "このレースは三連複戦略の対象外です。",
                        "LALO": "このレースは複勝戦略の対象外です。",
                        "CQC":  "このレースは単勝戦略の対象外です。",
                    }
                    for det_tab, plan_key, plan_label in [
                        (det_tab_haho, "HAHO", "馬連◎軸2点 ＋ 三連複ボックス1点"),
                        (det_tab_halo, "HALO", "三連複ボックス1点のみ"),
                        (det_tab_lalo, "LALO", "複勝◎1点のみ"),
                        (det_tab_cqc,  "CQC",  "単勝◎1点のみ"),
                    ]:
                        with det_tab:
                            bets = bets_all.get(plan_key, [])
                            if not bets:
                                st.info(_no_bet_msg.get(plan_key, "買い目がありません。"))
                                continue
                            bets_df   = pd.DataFrame(bets)
                            total_amt = bets_df["購入額"].sum()
                            m1, m2, m3 = st.columns(3)
                            m1.metric("合計購入額", f"{total_amt:,}円")
                            m2.metric("馬券種数",   f"{bets_df['馬券種'].nunique()}種")
                            m3.metric("総点数",     f"{len(bets_df)}点")
                            st.caption(plan_label)
                            for bet_type, grp_b in bets_df.groupby("馬券種"):
                                type_total = grp_b["購入額"].sum()
                                combos_html = "　".join([
                                    f'<span style="font-size:16px;font-weight:bold;color:#cdd6f4">{row["買い目"]}</span>'
                                    f'<span style="color:#888;font-size:12px">({row["購入額"]:,}円)</span>'
                                    for _, row in grp_b.iterrows()
                                ])
                                roi_val = grp_b.iloc[0]["ROI"]
                                st.markdown(
                                    f'<div class="bet-card">'
                                    f'<div style="display:flex;justify-content:space-between;margin-bottom:6px">'
                                    f'<span style="color:#5865f2;font-weight:bold">{bet_type}</span>'
                                    f'<span style="color:#888;font-size:12px">ROI目安:{roi_val:.1f}%　計{type_total:,}円</span>'
                                    f'</div><div>{combos_html}</div></div>',
                                    unsafe_allow_html=True,
                                )

        with tab2:
            if not shap_ok:
                st.error("SHAP計算に失敗しました。")
            else:
                st.markdown("### 🔍 全頭分析")
                for i, row in race_df.sort_values("馬番").reset_index(drop=True).iterrows():
                    sv_row  = shap_vals[i]
                    name    = str(row.get("馬名", f"{int(row['馬番'])}番"))
                    mark    = str(row.get("mark",""))
                    score   = float(row.get("score",0))
                    comment = make_comment(sv_row, feature_cols, name, score, mark)
                    mk_cls  = MARK_CLASS.get(mark,"")
                    mk_html = f'<span class="{mk_cls}">{mark}</span> ' if mark else ""
                    sv_arr  = np.array(sv_row)
                    pairs   = sorted(zip(sv_arr, feature_cols), reverse=True)
                    pos4    = [(v,c) for v,c in pairs if v > 0][:4]
                    neg3    = [(v,c) for v,c in pairs if v < 0][-3:]
                    c_left, c_mid, c_right = st.columns([2, 3, 2])
                    with c_left:
                        st.markdown(
                            f'<div style="padding:8px 0">'
                            f'<span style="font-size:15px;font-weight:bold">{mk_html}{name}</span>'
                            f'<span style="font-size:12px;color:#888;margin-left:8px">{score:.1f}%</span>'
                            f'</div>'
                            f'<div style="font-size:13px;color:#a6adc8;line-height:1.7">{comment}</div>',
                            unsafe_allow_html=True,
                        )
                    with c_mid:
                        fig = make_shap_fig(sv_row, feature_cols, name)
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                    with c_right:
                        for v, c in pos4:
                            label = FEATURE_LABEL.get(c,c)
                            st.markdown(f'<div style="color:#e74c3c;font-size:12px;margin:2px 0">🟥 {label} +{v:.3f}</div>', unsafe_allow_html=True)
                        for v, c in neg3:
                            label = FEATURE_LABEL.get(c,c)
                            st.markdown(f'<div style="color:#5865f2;font-size:12px;margin:2px 0">🟦 {label} {v:.3f}</div>', unsafe_allow_html=True)
                    st.markdown("<hr style='border-color:#313244;margin:8px 0'>", unsafe_allow_html=True)


        with tab3:
            render_pace_scenario(race_df)
            st.markdown("---")
            course_trend = load_course_trend()
            if not course_trend:
                st.warning("course_trend.json が見つかりません。data/course_trend.json を配置してください。")
            else:
                _render_course_analysis(course_trend or {}, place, meta)

    with col_nav:
        # 同会場の他Rナビ
        race_id_col = "レースID(新/馬番無)"
        other_races = []
        for rid, grp in all_df[all_df["場所"] == place].groupby(race_id_col):
            other_races.append({
                "race_id": rid,
                "R":       int(grp.iloc[0].get("R", 0)),
                "クラス":  str(grp.iloc[0].get("クラス名","")),
                "戦略":    is_in_strategy(place, str(grp.iloc[0].get("クラス名","")), strategy),
            })
        other_races.sort(key=lambda x: x["R"])

        st.markdown(f"#### {place} レース")
        for r in other_races:
            is_cur   = (r["race_id"] == st.session_state.selected_race_id)
            btn_type = "primary" if is_cur else "secondary"
            label    = f'{"▶ " if is_cur else ""}{r["R"]}R　{r["クラス"]}'
            if st.button(label, key=f'nav_{r["race_id"]}', type=btn_type, use_container_width=True):
                st.session_state.selected_race_id = r["race_id"]
                st.rerun()

        st.markdown("---")
        if place not in EXCLUDE_PLACES:
            render_trend_panel([place])


# =========================================================
# main
# =========================================================
def main() -> None:
    st.set_page_config(page_title="PyCaLiAI", page_icon="🏇", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("🏇 PyCaLiAI 競馬予想システム")

    lgbm_obj, cat_obj = load_models()
    strategy          = load_strategy()
    course_trend      = load_course_trend()

    weekly_dir = BASE_DIR / "data" / "weekly"
    weekly_dir.mkdir(parents=True, exist_ok=True)

    csv_files    = sorted(weekly_dir.glob("????????.csv"), reverse=True)
    date_options = [f.stem for f in csv_files]

    with st.sidebar:
        st.header("📅 出走表CSV")
        if date_options:
            selected_date = st.selectbox(
                "開催日を選択", date_options,
                format_func=lambda x: f"{x[:4]}/{x[4:6]}/{x[6:]}",
            )
        else:
            selected_date = None
            st.info("data/weekly/ に 20260308.csv 形式で保存してください。")

        st.divider()
        uploaded = st.file_uploader("CSVをアップロード", type=["csv"])
        if uploaded is not None:
            import datetime
            stem = Path(uploaded.name).stem
            save_name = f"{stem}.csv" if stem.isdigit() and len(stem)==8 \
                        else datetime.date.today().strftime("%Y%m%d")+".csv"
            (weekly_dir / save_name).write_bytes(uploaded.getvalue())
            st.success(f"保存: {save_name}")
            st.rerun()

        st.divider()
        budget = st.number_input("1レース予算（円）", min_value=1_000,
                                 max_value=1_000_000, value=10_000, step=1_000)
        st.divider()
        st.caption(f"除外会場: {', '.join(EXCLUDE_PLACES)}")
        st.caption(f"除外クラス: {', '.join(EXCLUDE_CLASSES)}")

    if selected_date is None:
        st.info("サイドバーからCSVを選択してください。")
        return

    csv_path = weekly_dir / f"{selected_date}.csv"
    with st.spinner("CSV読み込み中..."):
        raw_df = parse_target_csv(csv_path)
        if raw_df.empty:
            st.error("CSVの読み込みに失敗しました。")
            return

    predicted_json = predict_all_races(selected_date, raw_df.to_json(), lgbm_obj, cat_obj)
    import io
    all_df = pd.read_json(io.StringIO(predicted_json))

    if "selected_race_id" not in st.session_state:
        st.session_state.selected_race_id = None
    if "selected_place" not in st.session_state:
        st.session_state.selected_place = ""
    if "show_buylist" not in st.session_state:
        st.session_state.show_buylist = False

    race_id_col = "レースID(新/馬番無)"

    # メインタブ
    main_tab1, main_tab2, main_tab3 = st.tabs(["🏇 レース予想", "🎫 今日の買い目", "📊 的中実績"])

    results = load_results()

    with main_tab3:
        page_results(results)

    with main_tab2:
        page_buylist(all_df, strategy, budget)

    with main_tab1:
        # 選択中のrace_idが現在のCSVに存在しない場合リセット
        if st.session_state.selected_race_id is not None:
            if st.session_state.selected_race_id not in all_df[race_id_col].values:
                st.session_state.selected_race_id = None

        if st.session_state.selected_race_id is None:
            n_races    = all_df[race_id_col].nunique()
            n_strategy = sum(1 for _, grp in all_df.groupby(race_id_col)
                             if is_in_strategy(str(grp.iloc[0].get("場所","")),
                                               str(grp.iloc[0].get("クラス名","")), strategy))
            c1, c2, c3 = st.columns(3)
            c1.metric("レース数",      f"{n_races}R")
            c2.metric("頭数",          f"{len(all_df)}頭")
            c3.metric("戦略対象レース", f"{n_strategy}R")
            st.markdown("---")
            page_race_list(all_df, strategy, budget)
        else:
            race_df = all_df[all_df[race_id_col] == st.session_state.selected_race_id].copy()
            page_race_detail(race_df.reset_index(drop=True), all_df, strategy, budget, lgbm_obj, course_trend)


if __name__ == "__main__":
    main()
