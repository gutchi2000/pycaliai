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
HOSSEI_DIR    = DATA_DIR / "hosei"
KAKO5_DIR     = DATA_DIR / "kako5"
LGBM_PATH     = MODEL_DIR / "lgbm_optuna_v1.pkl"
CAT_PATH      = MODEL_DIR / "catboost_optuna_v1.pkl"
CAL_PATH       = MODEL_DIR / "ensemble_calibrator_v4.pkl"   # Test 2024-fit (2026-03-25更新)
CAL_PATH_V3    = MODEL_DIR / "ensemble_calibrator_v3.pkl"   # Train-based fallback
CAL_PATH_V2    = MODEL_DIR / "ensemble_calibrator_v2.pkl"   # Valid-based fallback
CAL_PATH_V1    = MODEL_DIR / "ensemble_calibrator_v1.pkl"   # legacy fallback
FUKU_CAL_PATH  = MODEL_DIR / "fukusho_calibrator_v1.pkl"   # Sprint 1.2 複勝専用
FUKU_CAL_GATE  = 0.55
WALKFORWARD_CSV = BASE_DIR / "reports" / "all_profitable_conditions.csv"
PYCALI_HIST_CSV = BASE_DIR / "data" / "pycali_history.csv"
PYCALI_HIST_PARQUET = BASE_DIR / "data" / "pycali_history.parquet"
WIN_MODEL_PATH = MODEL_DIR / "lgbm_win_v1.pkl"
TORCH_PATH     = MODEL_DIR / "transformer_optuna_v1.pkl"
META_PATH      = MODEL_DIR / "stacking_meta_v1.pkl"
STACK_CAL_PATH = MODEL_DIR / "stacking_calibrator_v1.pkl"
VALUE_MODEL_PATH = MODEL_DIR / "value_model_v2.pkl"
# Phase 5: 8モデルアンサンブル用パス
RANK_PATH       = MODEL_DIR / "catboost_rank_v1.pkl"
FUKU_LGBM_PATH  = MODEL_DIR / "lgbm_fukusho_v1.pkl"
FUKU_CAT_PATH   = MODEL_DIR / "catboost_fukusho_v1.pkl"
RANK_LGBM_PATH  = MODEL_DIR / "lgbm_rank_v1.pkl"
REGRESS_PATH    = MODEL_DIR / "lgbm_regression_v1.pkl"
ENS_WEIGHTS_PATH = MODEL_DIR / "ensemble_weights.json"
ORDER_MODEL_PATH = MODEL_DIR / "order_model_v1.pkl"
# Phase 5+: 距離別 Mixture of Experts
EXPERT_DIR      = MODEL_DIR
EXPERT_PATHS    = {
    "turf_short": EXPERT_DIR / "expert_turf_short.pkl",
    "turf_mid":   EXPERT_DIR / "expert_turf_mid.pkl",
    "turf_long":  EXPERT_DIR / "expert_turf_long.pkl",
    "dirt":       EXPERT_DIR / "expert_dirt.pkl",
}

# モデルキャッシュ（プロセス内で1回だけロード）
_model_cache: dict = {}

def _get_cached(path: Path, key: str):
    if key not in _model_cache and path.exists():
        _model_cache[key] = joblib.load(path)
    return _model_cache.get(key)

MIN_UNIT = 100
MARKS    = ["◎", "◯", "▲", "△", "×"]

EXCLUDE_PLACES  = {"東京", "小倉"}  # Phase 5: 阪神・京都の全面除外解除 (2026-04-05)
EXCLUDE_CLASSES = {"新馬", "障害"}

# Phase 5+: SegmentBetFilter — ROI<80% の (距離セグメント, 券種) を購入対象から除外
# 根拠: 2024-2025 backtest 実測 ROI（dirt三連複 68.9%, turf_short馬連 57.2%, turf_short複勝 74.6%, turf_mid馬連 72.4%）
SEGMENT_BET_BLACKLIST = {
    ("dirt",       "三連複"),
    ("turf_short", "馬連"),
    ("turf_short", "複勝"),
    ("turf_mid",   "馬連"),
}

# Phase 5+ Step2: クラス×セグメント×券種 ブラックリスト
# 根拠: dirt馬連 89.8%だが内訳で未勝利71.8%, 重賞系0-77.5%が損失源
SEGMENT_CLASS_BET_BLACKLIST = {
    ("dirt", "未勝利", "馬連"),
    ("dirt", "オープン","馬連"),
    ("dirt", "3勝",   "馬連"),
    ("dirt", "GⅠ",   "馬連"),
    ("dirt", "GⅡ",   "馬連"),
    ("dirt", "GⅢ",   "馬連"),
    ("dirt", "OP(L)", "馬連"),
    # Phase 5+ Step3: 複勝の統計的有意な負ROIブロック (2026-04-08, analyst+statistician推奨)
    ("dirt",      "1勝", "複勝"),   # n=202 ROI 82.1% p≈0.03 有意
}


def _is_class_blacklisted(seg: str, cls_raw: str, bt: str) -> bool:
    return (seg, cls_raw, bt) in SEGMENT_CLASS_BET_BLACKLIST


def _race_segment(td: str, dist) -> str:
    """距離・芝/ダから 4 セグメントを返す。"""
    try:
        d = int(dist)
    except (TypeError, ValueError):
        return "unknown"
    if td == "ダ":
        return "dirt"
    if d <= 1400:
        return "turf_short"
    if d <= 2200:
        return "turf_mid"
    return "turf_long"


def _is_blacklisted(td: str, dist, bet_type: str) -> bool:
    return (_race_segment(td, dist), bet_type) in SEGMENT_BET_BLACKLIST

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


def _load_hosei(date_str: str) -> "pd.DataFrame | None":
    """data/hosei/H_*.csv を glob して全期間の補正タイムを返す。
    date_str は互換性のために受け取るが現在は使用しない。
    """
    files = sorted(HOSSEI_DIR.glob("H_*.csv"))
    if not files:
        return None
    dfs = []
    for path in files:
        for enc in ["cp932", "utf-8-sig", "utf-8"]:
            try:
                df = pd.read_csv(path, encoding=enc,
                                 usecols=["レースID(新)", "馬番", "前走補9", "前走補正"])
                dfs.append(df)
                break
            except Exception:
                continue
    if not dfs:
        return None
    result = pd.concat(dfs, ignore_index=True).drop_duplicates()
    result["レースID(新/馬番無)"] = result["レースID(新)"].astype(str).str[:16]
    result["馬番"] = pd.to_numeric(result["馬番"], errors="coerce")
    for col in ["前走補9", "前走補正"]:
        result[col] = pd.to_numeric(result[col], errors="coerce")
    return result[["レースID(新/馬番無)", "馬番", "前走補9", "前走補正"]]


# =========================================================
# CSV パース
# =========================================================
def _compute_pycali(row) -> float:
    try:
        s = float(row.get("score", 0) or 0)
        if s > 0:
            return max(0.0, min(100.0, s))
    except Exception:
        pass
    try:
        pop = float(row.get("人気", 99) or 99)
        return max(0.0, min(100.0, 100.0 - pop * 5.0))
    except Exception:
        return 0.0


@st.cache_data(show_spinner=False)
def _load_pycali_history() -> pd.DataFrame:
    """data/pycali_history.(parquet|csv) を読む。(馬名, 日付, スコア) の実データ。"""
    for path, reader in [
        (PYCALI_HIST_PARQUET, lambda p: pd.read_parquet(p)),
        (PYCALI_HIST_CSV,     lambda p: pd.read_csv(p, encoding="utf-8-sig")),
    ]:
        if path.exists():
            try:
                df = reader(path)
                df["日付"] = df["日付"].astype(str).str[:8]
                df["スコア"] = pd.to_numeric(df["スコア"], errors="coerce")
                return df.dropna(subset=["スコア"])
            except Exception as e:
                logger.warning(f"pycali_history読込失敗 {path.name}: {e}")
    return pd.DataFrame()


def _norm_horse_name(s) -> str:
    import unicodedata
    if not isinstance(s, str):
        return ""
    return unicodedata.normalize("NFKC", s).strip().replace("　", "")


def _real_pycali_history(horse_name: str, current_date: str | None, n: int = 5) -> list[float]:
    """馬名ベースで過去N走のスコアを時系列順で返す（最新を末尾）。NFKC正規化済みキーで照合。"""
    hist = _load_pycali_history()
    if hist.empty or not horse_name:
        return []
    key = _norm_horse_name(horse_name)
    sub = hist[hist["馬名"] == key]
    if current_date:
        cd = str(current_date)[:8]
        sub = sub[sub["日付"] < cd]
    sub = sub.sort_values("日付").tail(n)
    return sub["スコア"].astype(float).tolist()


def _pycali_form_history(row, n: int = 4) -> list:
    """過去3走の実着順と当該レース推定PyCaLiから簡易推移を作る（実データ由来）。
    着順は race_df の `三走前着順/二走前着順/前走着順` 列をそのまま使用。"""
    cur = _compute_pycali(row)
    hist = []
    for pre in ["三走前", "二走前", "前走"]:
        try:
            rank = float(row.get(f"{pre}着順", 0) or 0)
        except Exception:
            rank = 0
        if rank > 0:
            hist.append(max(0.0, min(100.0, 100.0 - rank * 6.0)))
        else:
            hist.append(None)
    hist.append(cur)
    # None を前後補完
    clean = [v for v in hist if v is not None]
    if not clean:
        return [cur] * n
    hist = [v if v is not None else clean[0] for v in hist]
    return hist[-n:]


def _make_sparkline(values: list, width: float = 1.6, height: float = 0.35):
    fig, ax = plt.subplots(figsize=(width, height), dpi=100)
    if values:
        ax.plot(range(len(values)), values, color="#5865f2", linewidth=1.5, marker="o", markersize=2)
        ax.fill_between(range(len(values)), values, alpha=0.2, color="#5865f2")
    ax.set_ylim(0, 100)
    ax.axis("off")
    fig.patch.set_alpha(0)
    fig.tight_layout(pad=0)
    return fig


def render_danger_favorite_badge(race_df) -> None:
    if race_df is None or race_df.empty:
        return
    df = race_df.copy()
    try:
        df["_pyca"] = df.apply(_compute_pycali, axis=1)
        df["_pop"] = pd.to_numeric(df.get("人気"), errors="coerce")
    except Exception:
        return
    if df["_pop"].isna().all():
        return
    median_pyca = df["_pyca"].median()
    danger = df[(df["_pop"].between(1, 3)) & (df["_pyca"] < median_pyca)]
    if danger.empty:
        st.markdown(
            '<div style="padding:8px 12px;background:#1e3a24;border-left:3px solid #22c55e;'
            'border-radius:4px;margin:6px 0;font-size:13px;color:#a7f3d0">'
            '🟢 危険な人気馬は検出されませんでした</div>', unsafe_allow_html=True)
        return
    parts = []
    for _, r in danger.sort_values("_pop").iterrows():
        pop = int(r["_pop"])
        icon = "🔴" if pop == 1 else "🟡"
        name = str(r.get("馬名", f'{int(r.get("馬番",0))}番'))
        parts.append(f'{icon} {pop}人気 {name} (PyCaLi {r["_pyca"]:.0f})')
    html = "　".join(parts)
    st.markdown(
        f'<div style="padding:10px 14px;background:#3a1e1e;border-left:3px solid #ef4444;'
        f'border-radius:4px;margin:6px 0;font-size:13px;color:#fecaca">'
        f'⚠️ 危険な人気馬: {html}</div>', unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def _load_walkforward_conditions() -> pd.DataFrame:
    if not WALKFORWARD_CSV.exists():
        return pd.DataFrame()
    for enc in ("utf-8-sig", "cp932", "utf-8"):
        try:
            return pd.read_csv(WALKFORWARD_CSV, encoding=enc)
        except Exception:
            continue
    return pd.DataFrame()


def render_roi_heatmap() -> None:
    st.markdown("### 📊 ROI ヒートマップ (場所 × クラス)")
    df = _load_walkforward_conditions()
    if df.empty:
        st.info("reports/all_profitable_conditions.csv が見つかりません。")
        return
    bet_types = sorted(df["馬券種"].dropna().unique().tolist())
    if not bet_types:
        st.info("馬券種データがありません。")
        return
    sel = st.selectbox("馬券種", bet_types, key="roi_heatmap_bet_type")
    sub = df[df["馬券種"] == sel]
    if sub.empty:
        st.info("該当データなし")
        return
    try:
        pivot = sub.pivot_table(index="場所", columns="クラス", values="回収率", aggfunc="mean")
    except Exception as e:
        st.error(f"ピボット失敗: {e}")
        return
    try:
        import plotly.express as px
        pivot_cap = pivot.clip(upper=200)
        fig = px.imshow(
            pivot_cap, text_auto=".0f", aspect="auto",
            color_continuous_scale="RdYlGn", origin="upper",
            labels=dict(color="ROI %"), zmin=80, zmax=200,
        )
        fig.update_layout(height=420, margin=dict(l=40, r=20, t=30, b=40))
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.dataframe(pivot.round(1))
    st.caption(
        f"件数: {len(sub)} 条件 / 平均ROI: {sub['回収率'].mean():.1f}% "
        f"※ `all_profitable_conditions.csv` は ROI≥80% の条件のみを収録した抽出データ。"
        f"表示は200%でクリップ。全体グリッドではない点に注意。"
    )


def render_weekly_portfolio() -> None:
    """今週の pred CSV から買い目ポートフォリオを自動集計して表示。"""
    st.markdown("### 💼 今週の推奨ポートフォリオ")

    # 最新 pred CSV を探す
    pred_dir = BASE_DIR / "reports"
    preds = sorted(pred_dir.glob("pred_*.csv"), reverse=True)
    if not preds:
        st.info("reports/pred_*.csv が見つかりません。weekly_pre.ps1 を実行してください。")
        return

    latest = preds[0]
    date_str = latest.stem.replace("pred_", "")
    try:
        df = pd.read_csv(latest, encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(latest, encoding="cp932")

    st.markdown(
        f'<div style="padding:12px 16px;background:#1a1a2e;border-radius:6px;margin:8px 0">'
        f'<div style="color:#a6adc8;font-size:12px">対象日</div>'
        f'<div style="font-size:18px;font-weight:bold;color:#cdd6f4">{date_str[:4]}/{date_str[4:6]}/{date_str[6:]}</div>'
        f'</div>', unsafe_allow_html=True)

    plans = [
        ("TRIPLE", "三連複", "TRIPLE_三連複_購入額", "複勝", "TRIPLE_複勝_購入額"),
        ("HAHO",   "馬連",   "HAHO_馬連_購入額",   "三連複", "HAHO_三連複_購入額"),
        ("HALO",   "三連複", "HALO_三連複_購入額",  None, None),
        ("LALO",   "複勝",   "LALO_複勝_購入額",    None, None),
        ("CQC",    "単勝",   "CQC_単勝_購入額",     None, None),
    ]

    summary_rows = []
    for plan, t1, c1, t2, c2 in plans:
        target_col = f"{plan}_戦略対象"
        if target_col not in df.columns:
            continue
        sub = df[df[target_col] == 1] if target_col in df.columns else pd.DataFrame()
        if sub.empty:
            sub = df[pd.to_numeric(df.get(c1, 0), errors="coerce").fillna(0) > 0]
        if sub.empty:
            continue
        inv1 = pd.to_numeric(sub.get(c1, 0), errors="coerce").fillna(0).sum()
        inv2 = pd.to_numeric(sub.get(c2, 0), errors="coerce").fillna(0).sum() if c2 and c2 in sub.columns else 0
        total_inv = inv1 + inv2
        n_races = sub["レースID"].nunique() if "レースID" in sub.columns else len(sub)
        types = t1 if not t2 else f"{t1}+{t2}"
        summary_rows.append({
            "プラン": plan, "券種": types, "対象R": n_races,
            "投資額": int(total_inv),
        })

    if not summary_rows:
        st.info("この日の買い目データがありません。")
        return

    sum_df = pd.DataFrame(summary_rows)
    total_inv = sum_df["投資額"].sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("プラン数", f"{len(sum_df)}")
    c2.metric("合計投資", f"{total_inv:,}円")
    c3.metric("対象レース", f"{df['レースID'].nunique() if 'レースID' in df.columns else '-'}R")

    st.markdown("#### プラン別 投資配分")
    for _, row in sum_df.iterrows():
        pct = row["投資額"] / total_inv * 100 if total_inv > 0 else 0
        bar_w = int(pct * 2)
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:10px;margin:6px 0">'
            f'<div style="width:70px;font-weight:bold;color:#89b4fa">{row["プラン"]}</div>'
            f'<div style="width:80px;color:#a6adc8;font-size:13px">{row["券種"]}</div>'
            f'<div style="flex:1;height:16px;background:#313244;border-radius:4px;overflow:hidden">'
            f'<div style="height:100%;width:{bar_w}%;background:#5865f2"></div></div>'
            f'<div style="width:100px;text-align:right;color:#cdd6f4">{row["投資額"]:,}円</div>'
            f'<div style="width:50px;text-align:right;color:#6c7086">{pct:.0f}%</div>'
            f'</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 全買い目一覧")
    show_cols = [c for c in df.columns if c in ("場所", "R", "クラス", "印", "馬名") or "買い目" in c or "購入額" in c]
    if show_cols:
        st.dataframe(df[show_cols].head(100), use_container_width=True, hide_index=True)


def render_feedback_dashboard() -> None:
    st.markdown("### 📋 結果フィードバック")
    st.caption("全プラン横断の総合サマリ（プラン別内訳は「📋 的中実績」タブ）")
    results = load_results()
    if not results:
        st.info("results.json が見つかりません。`python generate_results.py` で生成してください。")
        return

    # 新フォーマット: {HAHO: {total/by_type/by_place/weekly}, HALO: {...}, ...}
    # 旧フォーマット fallback: top-level に by_type/by_place
    plan_keys = [k for k in ["HAHO", "HALO", "STANDARD", "TRIPLE"]
                 if k in results and isinstance(results[k], dict)]

    if not plan_keys:
        # 旧フォーマット fallback（念のため）
        by_type = results.get("by_type", {})
        if by_type:
            st.markdown("#### 馬券種別パフォーマンス（旧形式）")
            rows = []
            for k, v in by_type.items():
                if isinstance(v, dict):
                    rows.append({"馬券種": k, **v})
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            return
        st.info("results.json に有効なプランデータがありません。")
        return

    # --- プラン横断サマリ ---
    total_bet = total_ret = total_races = 0
    plan_rows = []
    for pk in plan_keys:
        t = results[pk].get("total", {})
        bet = int(t.get("bet", 0) or 0)
        ret = int(t.get("ret", 0) or 0)
        races = int(t.get("races", 0) or 0)
        roi = float(t.get("roi", 0) or 0)
        total_bet += bet
        total_ret += ret
        total_races = max(total_races, races)
        plan_rows.append({
            "プラン": pk,
            "対象R": races,
            "投資": bet,
            "払戻": ret,
            "収支": ret - bet,
            "ROI(%)": roi,
        })
    total_pnl = total_ret - total_bet
    total_roi = (total_ret / total_bet * 100) if total_bet else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("対象レース", f"{total_races}R")
    c2.metric("総投資", f"¥{total_bet:,}")
    c3.metric("総払戻", f"¥{total_ret:,}")
    roi_icon = "🟢" if total_roi >= 100 else "🟡" if total_roi >= 80 else "🔴"
    c4.metric("総合ROI", f"{roi_icon} {total_roi:.1f}%",
              delta=f"{total_pnl:+,}円")

    st.markdown("---")
    st.markdown("#### プラン別パフォーマンス")
    pdf = pd.DataFrame(plan_rows)
    st.dataframe(pdf, use_container_width=True, hide_index=True)

    # プラン別 ROI バーチャート
    try:
        st.bar_chart(pdf.set_index("プラン")["ROI(%)"])
    except Exception:
        pass

    # --- 馬券種横断 ---
    agg_bt: dict[str, dict] = {}
    for pk in plan_keys:
        for k, v in results[pk].get("by_type", {}).items():
            if not isinstance(v, dict):
                continue
            d = agg_bt.setdefault(k, {"bet":0,"ret":0,"hit":0,"races":0})
            d["bet"]   += int(v.get("bet", 0) or 0)
            d["ret"]   += int(v.get("ret", 0) or 0)
            d["hit"]   += int(v.get("hit", 0) or 0)
            d["races"] += int(v.get("races", 0) or 0)
    if agg_bt:
        st.markdown("---")
        st.markdown("#### 馬券種横断パフォーマンス")
        rows = []
        for k, d in agg_bt.items():
            roi = (d["ret"]/d["bet"]*100) if d["bet"] else 0
            hr  = (d["hit"]/d["races"]*100) if d["races"] else 0
            rows.append({
                "馬券種": k,
                "対象R": d["races"],
                "的中": d["hit"],
                "的中率(%)": round(hr, 1),
                "投資": d["bet"],
                "払戻": d["ret"],
                "ROI(%)": round(roi, 1),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # --- 先週 ---
    last_weekly = None
    for pk in plan_keys:
        wl = results[pk].get("weekly", [])
        if wl:
            last_weekly = (pk, wl[-1])
            break
    if last_weekly:
        pk, last = last_weekly
        st.markdown("---")
        st.markdown(f"#### 直近週ハイライト（{pk}基準）")
        c1, c2, c3 = st.columns(3)
        c1.metric("週", str(last.get("週", "-")))
        c2.metric("レース数", str(last.get("レース数", "-")))
        c3.metric("ROI", f"{float(last.get('ROI', 0)):.1f}%")


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
    for col in ["馬体重","馬体重増減","前走斤量","生産者","馬主(最新/仮想)"]:
        if col not in df.columns:
            df[col] = 0
    for col in ["前走走破タイム","前走着差タイム"]:
        if col not in df.columns:
            df[col] = float("nan")  # LGBMのNaN処理に任せる（0だと分布が大きく外れる）

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

    # ── 補正タイムCSV（data/hosei/YYYYMMDD.csv）があればマージ ──
    hosei_df = _load_hosei(date_str)
    if hosei_df is not None:
        df = df.merge(hosei_df, on=["レースID(新/馬番無)", "馬番"], how="left")
    else:
        df["前走補9"]  = float("nan")
        df["前走補正"] = float("nan")

    # ── 過去5走特徴量（data/kako5/YYYYMMDD.csv）があればマージ ──
    kako5_path = KAKO5_DIR / f"{date_str}.csv"
    if kako5_path.exists():
        try:
            from parse_kako5 import build_from_kako5
            kako5_df = build_from_kako5(kako5_path)
            if not kako5_df.empty:
                if kako5_df["レースID(新)"].astype(str).str.len().mode().iloc[0] > 16:
                    kako5_df["レースID(新)"] = kako5_df["レースID(新)"].astype(str).str[:16]
                df = df.merge(
                    kako5_df.rename(columns={"レースID(新)": "レースID(新/馬番無)"}),
                    on=["レースID(新/馬番無)", "馬番"], how="left", suffixes=("", "_kako5"),
                )
                logger.info(f"kako5 カバレッジ={df['kako5_avg_pos'].notna().mean()*100:.1f}%")
        except Exception as e:
            logger.warning(f"kako5マージ失敗: {e}")

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


@st.cache_resource(show_spinner=False)
def _load_trifecta_model():
    """trifecta_model_v1.pkl をロード（存在しない場合は None を返す）。"""
    path = BASE_DIR / "models" / "trifecta_model_v1.pkl"
    if not path.exists():
        return None, None
    try:
        obj = joblib.load(path)
        return obj.get("model"), obj.get("feature_cols")
    except Exception as e:
        logger.warning(f"trifecta_model_v1 ロード失敗: {e}")
        return None, None


@st.cache_data(show_spinner="戦略データ読み込み中...")
def load_strategy() -> dict:
    if not STRATEGY_JSON.exists():
        st.error(f"戦略ファイルが見つかりません: {STRATEGY_JSON}")
        st.stop()
    with open(STRATEGY_JSON, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False, ttl=300)
def _load_results_cached(mtime: float) -> dict:
    if not RESULTS_JSON.exists():
        return {}
    with open(RESULTS_JSON, encoding="utf-8") as f:
        return json.load(f)


def load_results() -> dict:
    """的中実績データ読み込み（results.json）。mtime をキーにキャッシュ。"""
    if not RESULTS_JSON.exists():
        return {}
    return _load_results_cached(RESULTS_JSON.stat().st_mtime)


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
                     "horse_fuku10","horse_fuku30","prev_pos_rel","closing_power",
                     "前走補9","前走補正"}
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan if col in _ROLLING_COLS else 0
    # lgb.Booster と LGBMClassifier 両対応
    import lightgbm as lgb
    if isinstance(model, lgb.Booster):
        return model.predict(df[feature_cols])
    return model.predict_proba(df[feature_cols])[:, 1]


def predict_lgbm_win(df: pd.DataFrame) -> np.ndarray:
    """lgbm_win_v1 (is_1st_place) 予測。モデル未存在時は None を返す。"""
    if not WIN_MODEL_PATH.exists():
        return None
    obj = _get_cached(WIN_MODEL_PATH, "lgbm_win")
    if obj is None:
        return None
    model, encoders, feature_cols = obj["model"], obj["encoders"], obj["feature_cols"]
    df = df.copy()
    for col in ["前走走破タイム", "前走着差タイム"]:
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
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    import lightgbm as lgb
    if isinstance(model, lgb.Booster):
        return model.predict(df[feature_cols])
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
                     "horse_fuku10","horse_fuku30","prev_pos_rel","closing_power",
                     "前走補9","前走補正"}
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
    for path, name in [(CAL_PATH, "v4"), (CAL_PATH_V3, "v3"), (CAL_PATH_V2, "v2"), (CAL_PATH_V1, "v1")]:
        if path.exists():
            logger.info(f"キャリブレーター {name} をロード: {path}")
            return joblib.load(path)["calibrator"]
    logger.warning("キャリブレーター未生成。calibrate.py を先に実行してください。")
    return None


@st.cache_resource(show_spinner=False)
def _load_fukusho_calibrator():
    """fukusho_calibrator_v1.pkl (Sprint 1.2) をロード。bare IsotonicRegression。"""
    if FUKU_CAL_PATH.exists():
        try:
            return joblib.load(FUKU_CAL_PATH)
        except Exception as e:
            logger.warning(f"複勝キャリブレーターロード失敗: {e}")
    return None


SEGMENT_WEIGHTS_AVAILABLE = {"turf_mid", "dirt"}


def _load_ensemble_weights_app(segment: str | None = None) -> dict | None:
    """Phase 5+: segment 指定時は Expert 別重み優先、なければグローバル。"""
    if segment is not None:
        seg_path = MODEL_DIR / f"ensemble_weights_{segment}.json"
        if seg_path.exists():
            try:
                with open(seg_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data.get("weights", {})
            except Exception as e:
                logger.warning(f"ensemble_weights_{segment}.json ロード失敗: {e}")
    if ENS_WEIGHTS_PATH.exists():
        try:
            with open(ENS_WEIGHTS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("weights", {})
        except Exception as e:
            logger.warning(f"ensemble_weights.json ロード失敗: {e}")
    return None


def _select_segment_app(df: pd.DataFrame) -> str | None:
    if len(df) == 0:
        return None
    td = str(df.iloc[0].get("芝・ダ", ""))
    try:
        d = int(df.iloc[0].get("距離", 0))
    except (TypeError, ValueError):
        return None
    if td == "ダ":
        seg = "dirt"
    elif d <= 1400:
        seg = "turf_short"
    elif d <= 2200:
        seg = "turf_mid"
    else:
        seg = "turf_long"
    return seg if seg in SEGMENT_WEIGHTS_AVAILABLE else None


def _select_expert(td: str, dist) -> str | None:
    """距離・芝/ダから適切な Expert 名を返す。Expert モデルが無ければ None。"""
    try:
        d = int(dist)
    except (TypeError, ValueError):
        return None
    if td == "ダ":
        name = "dirt"
    elif d <= 1400:
        name = "turf_short"
    elif d <= 2200:
        name = "turf_mid"
    else:
        name = "turf_long"
    if EXPERT_PATHS[name].exists():
        return name
    return None


def _predict_expert(df: pd.DataFrame, expert_name: str) -> np.ndarray | None:
    """Expert モデルで予測。失敗時は None。"""
    obj = _get_cached(EXPERT_PATHS[expert_name], f"expert_{expert_name}")
    if obj is None:
        return None
    try:
        # Expert は train_lgbm のフォーマットなので predict_lgbm を再利用
        return predict_lgbm(df, obj)
    except Exception as e:
        logger.warning(f"Expert {expert_name} 予測失敗: {e}")
        return None


def _ensemble_with_optimized_weights_app(
    df: pd.DataFrame, lgbm_obj: dict, cat_obj: dict, weights: dict
) -> np.ndarray:
    """8モデル最適化重みでアンサンブル。predict_weekly.py の関数を再利用。"""
    # 遅延 import (循環回避)
    from predict_weekly import (
        predict_lgbm_fukusho, predict_catboost_fukusho,
        predict_catboost_rank, predict_lgbm_rank, predict_lgbm_regression,
    )
    model_map = {
        "lgbm":       (LGBM_PATH,       "lgbm_opt",   predict_lgbm,             lgbm_obj),
        "catboost":   (CAT_PATH,        "cat_opt",    predict_catboost,         cat_obj),
        "fuku_lgbm":  (FUKU_LGBM_PATH,  "fuku_lgbm",  predict_lgbm_fukusho,     None),
        "fuku_cat":   (FUKU_CAT_PATH,   "fuku_cat",   predict_catboost_fukusho, None),
        "rank_cat":   (RANK_PATH,       "rank",       predict_catboost_rank,    None),
        "rank_lgbm":  (RANK_LGBM_PATH,  "rank_lgbm",  predict_lgbm_rank,        None),
        "regression": (REGRESS_PATH,    "regression", predict_lgbm_regression,  None),
        "lgbm_win":   (WIN_MODEL_PATH,  "lgbm_win",   None,                     None),  # 特殊ケース
    }
    # Phase 5+: Expert モデルも候補登録
    for exp_name, exp_path in EXPERT_PATHS.items():
        model_map[f"expert_{exp_name}"] = (exp_path, f"expert_{exp_name}", predict_lgbm, None)
    raw = np.zeros(len(df))
    total_w = 0.0
    used = []
    for name, w in weights.items():
        if w < 0.001 or name not in model_map:
            continue
        path, key, fn, pre = model_map[name]
        if name == "lgbm_win":
            p = predict_lgbm_win(df)
            if p is None:
                continue
        else:
            obj = pre if pre is not None else _get_cached(path, key)
            if obj is None:
                continue
            try:
                p = fn(df, obj)
            except Exception as e:
                logger.warning(f"{name} 予測失敗: {e}")
                continue
        raw += w * p
        total_w += w
        used.append(name)
    if total_w < 0.01:
        raise RuntimeError("有効モデルなし")
    if abs(total_w - 1.0) > 0.01:
        raw = raw / total_w
    logger.info(f"app.py アンサンブル: {len(used)}モデル ({', '.join(used)})")
    return raw


def ensemble_predict(df: pd.DataFrame, lgbm_obj: dict, cat_obj: dict) -> np.ndarray:
    """Phase 5+: 8モデル最適化重み + 距離別Expert(あれば加重平均)。"""
    # --- Phase 5+: Expert別重み優先、なければグローバル ---
    segment = _select_segment_app(df)
    opt_weights = _load_ensemble_weights_app(segment)
    if opt_weights is not None:
        try:
            raw = _ensemble_with_optimized_weights_app(df, lgbm_obj, cat_obj, opt_weights)
        except Exception as e:
            logger.warning(f"最適化アンサンブル失敗→フォールバック: {e}")
            raw = None
    else:
        raw = None

    # --- フォールバック（旧3モデル）---
    if raw is None:
        p_lgbm = predict_lgbm(df, lgbm_obj)
        p_cat  = predict_catboost(df, cat_obj)
        p_win  = predict_lgbm_win(df)
        if p_win is not None:
            raw = 0.375 * p_lgbm + 0.375 * p_cat + 0.25 * p_win
        else:
            raw = 0.50 * p_lgbm + 0.50 * p_cat

    # --- Phase 5+: セグメント別重みを使っていない場合のみ旧式 Expert 単純加重 ---
    if segment is None and len(df) > 0:
        td = str(df.iloc[0].get("芝・ダ", ""))
        dist = df.iloc[0].get("距離", None)
        expert_name = _select_expert(td, dist)
        if expert_name is not None:
            p_exp = _predict_expert(df, expert_name)
            if p_exp is not None:
                raw = 0.7 * raw + 0.3 * p_exp
                logger.info(f"Expert {expert_name} 加重平均適用")

    cal = _load_calibrator()
    return cal.transform(raw) if cal is not None else raw


def assign_marks(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["mark"] = ""
    ranked = df["prob"].rank(ascending=False, method="first")
    MARK_MAP = {1:"◎", 2:"◯", 3:"▲", 4:"△", 5:"☆", 6:"★"}
    for idx, rank in ranked.items():
        if rank <= 6:
            df.at[idx, "mark"] = MARK_MAP[int(rank)]
    return df


def _compute_value_scores_app(race_df: pd.DataFrame) -> tuple:
    """Value Model v2 で predicted ROI と cal_prob を計算する (app.py内用)。"""
    nan_s = pd.Series(np.nan, index=race_df.index)
    value_obj = _get_cached(VALUE_MODEL_PATH, "value_model")
    if value_obj is None:
        return nan_s, nan_s
    vm    = value_obj["model"]
    feats = value_obj["features"]
    iso   = value_obj.get("calibrator")

    tansho   = pd.to_numeric(race_df.get("単勝", pd.Series(dtype=float)), errors="coerce")
    fuku_low  = tansho.pow(0.6).round(1)   # 複勝オッズ推定
    fuku_high = (tansho.pow(0.6) * 1.5).round(1)
    if tansho.isna().all():
        return nan_s, nan_s

    raw_prob     = race_df["prob"].values
    cal_prob_v   = iso.transform(raw_prob) if iso is not None else raw_prob
    fuku_mid     = (fuku_low + fuku_high) / 2
    model_rank   = race_df["prob"].rank(ascending=False, method="first")
    ninki        = tansho.rank(method="first", ascending=True)

    X = pd.DataFrame({
        "cal_prob":             cal_prob_v,
        "model_rank":           model_rank.values,
        "ninki":                ninki.values,
        "tan_odds":             tansho.values,
        "fuku_mid":             fuku_mid.values,
        "EV_fuku":              cal_prob_v * fuku_mid.values,
        "disagree":             model_rank.values - ninki.values,
        "abs_disagree":         np.abs(model_rank.values - ninki.values),
        "log_tan_odds":         np.log1p(tansho.values),
        "log_fuku_mid":         np.log1p(fuku_mid.values),
        "odds_rank_ratio":      tansho.values / (ninki.values + 0.5),
        "model_vs_market_prob": cal_prob_v - (1 / tansho.values),
        "shutsuu":              len(race_df),
        "fuku_spread":          (fuku_high - fuku_low).values,
        "fuku_spread_ratio":    ((fuku_high - fuku_low) / (fuku_mid + 0.01)).values,
    }, index=race_df.index)

    pred_roi = vm.predict(X[feats])
    return pd.Series(pred_roi, index=race_df.index), pd.Series(cal_prob_v, index=race_df.index)


def _load_precomputed_predictions(date_str: str) -> pd.DataFrame | None:
    """reports/buylist_horses_YYYYMMDD.parquet が存在すれば読み込む。
    Stage 1-07: 画面切替の待ち時間ゼロ化のため、予測結果を週次 precompute から取得。
    """
    pq = BASE_DIR / "reports" / f"buylist_horses_{date_str}.parquet"
    if not pq.exists():
        return None
    try:
        return pd.read_parquet(pq)
    except Exception as e:
        logger.warning(f"precompute parquet 読込失敗 {pq}: {e}")
        return None


@st.cache_data(show_spinner="全レース予想計算中...")
def predict_all_races(cache_key: str, df_json: str, _lgbm_obj: dict, _cat_obj: dict) -> str:
    """予測を実行。

    Stage 1-07 (2026-04-16): cache_key の先頭が "PRECOMPUTED:" なら、
    precompute parquet から merge する高速パスを使う（推論不要）。
    """
    import io
    df = pd.read_json(io.StringIO(df_json))

    # ── 高速パス: precompute parquet があればそれを merge ──
    if cache_key.startswith("PRECOMPUTED:"):
        date_str = cache_key.split(":", 1)[1].split("_")[0]
        pre = _load_precomputed_predictions(date_str)
        if pre is not None and not pre.empty:
            # 結合キー: race_id + 馬番
            need_cols = ["race_id", "馬番", "prob", "score", "印",
                         "p_win", "p_place23", "p_fuku",
                         "popularity", "value_score", "cal_prob"]
            keep = [c for c in need_cols if c in pre.columns]
            pre_sub = pre[keep].rename(columns={"印": "mark"})
            # 結合キーは両側を str/Int64 に統一して型不一致を防ぐ
            rid_col = "レースID(新/馬番無)" if "レースID(新/馬番無)" in df.columns else "race_id"
            df["_rid"]  = df[rid_col].astype(str) if rid_col in df.columns else ""
            df["_hnum"] = pd.to_numeric(df.get("馬番"), errors="coerce").astype("Int64")
            pre_sub     = pre_sub.copy()
            pre_sub["_rid"]  = pre_sub["race_id"].astype(str)
            pre_sub["_hnum"] = pd.to_numeric(pre_sub["馬番"], errors="coerce").astype("Int64")
            merged = df.merge(
                pre_sub.drop(columns=["race_id", "馬番"]),
                on=["_rid", "_hnum"], how="left",
            )
            merged = merged.drop(columns=["_rid", "_hnum"])
            # ev_score 計算（precompute には未収録）
            tansho = pd.to_numeric(merged.get("単勝", pd.Series(dtype=float)), errors="coerce")
            merged["ev_score"] = (merged["prob"].fillna(0) * tansho / 0.80).round(3)
            # 欠損の補完
            for col, default in [("prob", 0.0), ("mark", ""), ("score", 0.0)]:
                if col in merged.columns:
                    merged[col] = merged[col].fillna(default)
            logger.info(f"[predict_all_races] precompute parquet 採用 (date={date_str}) → 推論スキップ")
            return merged.to_json(force_ascii=False)

    # ── 通常パス: モデル推論 ──
    result_frames = []
    for race_id, race_df in df.groupby("レースID(新/馬番無)"):
        race_df = race_df.copy()
        try:
            race_df["prob"]     = ensemble_predict(race_df, _lgbm_obj, _cat_obj)
            race_df             = assign_marks(race_df)
            race_df["score"]    = (race_df["prob"] * 100).round(1)
            tansho = pd.to_numeric(race_df.get("単勝", pd.Series(dtype=float)), errors="coerce")
            race_df["ev_score"] = (race_df["prob"] * tansho / 0.80).round(3)
        except Exception as e:
            logger.warning(f"予測失敗 {race_id}: {e}")
            race_df["prob"]     = 0.0
            race_df["mark"]     = ""
            race_df["score"]    = 0.0
            race_df["ev_score"] = 0.0
        # Value Model v2 scoring
        try:
            race_df["value_score"], race_df["cal_prob"] = _compute_value_scores_app(race_df)
        except Exception:
            race_df["value_score"] = np.nan
            race_df["cal_prob"]    = np.nan
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


# =========================================================
# 発走時刻カウントダウン（Stage 1-08 / 2026-04-16）
# =========================================================
def _parse_hassou_hm(hassou: str) -> tuple[int, int] | None:
    """発走時刻 "15:35" 形式を (hour, minute) に。解析失敗時 None。"""
    if not hassou:
        return None
    import re as _re
    m = _re.search(r"(\d{1,2})[:：](\d{2})", str(hassou))
    if not m:
        return None
    try:
        return int(m.group(1)), int(m.group(2))
    except Exception:
        return None


def countdown_html(date_str: str, hassou: str, *,
                   size: str = "12px", color: str = "#f38ba8") -> str:
    """発走までの残り時間を表示する <span>。JS で 1 秒ごと更新。

    date_str: "2026.4.12" / "20260412" 等
    hassou:   "15:35"
    """
    hm = _parse_hassou_hm(hassou)
    if hm is None:
        return ""
    h, m = hm
    # 日付正規化
    ds = str(date_str).replace(".", "-").replace("/", "-").strip()
    parts = ds.split("-")
    try:
        if len(parts) == 3:
            y = int(parts[0]); mo = int(parts[1]); d = int(parts[2])
        elif len(ds) == 8 and ds.isdigit():
            y = int(ds[:4]); mo = int(ds[4:6]); d = int(ds[6:8])
        else:
            return ""
    except Exception:
        return ""

    # 発走 ISO (JST = +09:00)
    iso = f"{y:04d}-{mo:02d}-{d:02d}T{h:02d}:{m:02d}:00+09:00"
    # 各カウントダウンにユニーク id
    cd_id = f"cd_{y}{mo:02d}{d:02d}_{h:02d}{m:02d}_{abs(hash(iso))%10000}"
    html = (
        f'<span id="{cd_id}" data-deadline="{iso}" '
        f'style="font-size:{size};color:{color};margin-left:8px;font-weight:bold">⏰ --</span>'
        '<script>'
        '(function(){var el=document.getElementById("' + cd_id + '");if(!el)return;'
        'function upd(){var d=new Date(el.dataset.deadline);var now=new Date();'
        'var diff=Math.floor((d-now)/1000);'
        'if(diff<=-60){el.textContent="🏁 発走済";el.style.color="#6c7086";return;}'
        'if(diff<=0){el.textContent="🏇 発走中";el.style.color="#a6e3a1";return;}'
        'var h=Math.floor(diff/3600);var m=Math.floor((diff%3600)/60);var s=diff%60;'
        'var txt;if(h>0){txt="⏰ "+h+"h"+m+"m";}'
        'else if(m>=10){txt="⏰ "+m+"分";}'
        'else{txt="⏰ "+m+":"+String(s).padStart(2,"0");'
        'el.style.color=(m<3?"#f38ba8":"#fab387");}'
        'el.textContent=txt;}upd();setInterval(upd,1000);'
        '})();</script>'
    )
    return html


def is_in_strategy(place: str, cls_raw: str, strategy: dict) -> bool:
    if place in EXCLUDE_PLACES or cls_raw in EXCLUDE_CLASSES:
        return False
    cls = CLASS_NORMALIZE.get(cls_raw, cls_raw)
    return place in strategy and (cls in strategy[place] or cls_raw in strategy[place])


def _predict_order_proba(race_df: pd.DataFrame) -> pd.DataFrame | None:
    """着順予測モデルで各馬の (p_win, p_place23, p_out) を返す。
    モデルがなければ None。"""
    order_obj = _get_cached(ORDER_MODEL_PATH, "order_model")
    if order_obj is None:
        return None
    try:
        model = order_obj["model"]
        feats = order_obj["features"]
        encs  = order_obj["encoders"]
        df = race_df.copy()
        # カテゴリエンコード
        for col, le in encs.items():
            if col in df.columns:
                df[col] = df[col].astype(str).fillna("__NaN__")
                known = set(le.classes_)
                df[col] = df[col].apply(lambda x: x if x in known else "__unknown__")
                if "__unknown__" not in le.classes_:
                    le.classes_ = np.append(le.classes_, "__unknown__")
                df[col] = le.transform(df[col])
        # タイム文字列変換
        from utils import parse_time_str
        for col in ["前走走破タイム", "前走着差タイム"]:
            if col in df.columns:
                df[col] = parse_time_str(df[col])
        # 欠損特徴量を NaN で埋める
        for f in feats:
            if f not in df.columns:
                df[f] = np.nan
        proba = model.predict_proba(df[feats])
        result = pd.DataFrame(proba, columns=["p_win", "p_place23", "p_out"], index=race_df.index)
        result["馬番"] = race_df["馬番"].values
        return result
    except Exception as e:
        logger.debug(f"order model predict failed: {e}")
        return None


def _build_sanrentan_formation(race_df: pd.DataFrame, h_marks: dict,
                                budget: int = 9600) -> tuple[list[dict], dict]:
    """着順予測ベースで三連単フォーメーションを自動構築する。

    Returns: (bets_list, info_dict)
      info_dict: {"pattern": str, "first": [...], "second": [...], "third": [...],
                  "n_combos": int, "source": "order_model"|"score_rule"}
    """
    h1 = h_marks.get("◎")
    h2 = h_marks.get("◯")
    if not h1 or not h2:
        return [], {}

    info: dict = {"source": "score_rule"}

    # === Stage 2-05: HALO playbook (条件別 top_n + NO_BET) を先にチェック ===
    try:
        from utils import lookup_halo_policy as _lhp
        _meta_row = race_df.iloc[0] if len(race_df) > 0 else None
        _shiba_da = str(_meta_row.get("芝・ダ", "")) if _meta_row is not None else ""
        _fsize    = len(race_df)
        _policy   = _lhp(_shiba_da, _fsize)
        if _policy.get("no_bet"):
            info.update({"source": "playbook_no_bet",
                         "pattern": f"NO_BET[{_policy.get('cell','')}]",
                         "n_combos": 0, "first": [], "second": [], "third": []})
            return [], info
        _playbook_top_n = max(3, int(_policy.get("top_n", 3)))
        _playbook_gate  = float(_policy.get("ev_gate", 0.0))
        info["_playbook_cell"] = _policy.get("cell", "")
    except Exception:
        _playbook_top_n = 3
        _playbook_gate  = 0.0

    # === Stage 2-02: trifecta_model_v1 (LambdaRank + Plackett-Luce) を優先 ===
    try:
        tri_model, tri_feats = _load_trifecta_model()
        if tri_model is not None and tri_feats is not None:
            from train_trifecta_model import add_race_features, pl_combo_probs, FEATURE_COLS as TRI_FEATS
            # race_df に必要な列を補完
            _rdf = race_df.copy()
            if "mark" not in _rdf.columns and h_marks:
                mark_map = {v: k for k, v in h_marks.items()}
                _rdf["mark"] = pd.to_numeric(_rdf["馬番"], errors="coerce") \
                                  .map(lambda ub: mark_map.get(int(ub), "") if pd.notna(ub) else "")
            if "jyun" not in _rdf.columns:
                _rdf["jyun"] = float("nan")
            if "race_id" not in _rdf.columns:
                _rdf["race_id"] = "tmp"
            if "place" not in _rdf.columns:
                _rdf["place"] = ""
            if "race_santan_pay" not in _rdf.columns:
                _rdf["race_santan_pay"] = 0.0
            if "ensemble_prob" not in _rdf.columns:
                _rdf["ensemble_prob"] = pd.to_numeric(_rdf.get("prob", pd.Series(dtype=float)),
                                                        errors="coerce").fillna(0.0)
            _rdf = add_race_features(_rdf)
            _X = pd.DataFrame([
                {f: float(row.get(f, 0)) for f in TRI_FEATS}
                for _, row in _rdf.iterrows()
            ])
            _model_scores = tri_model.predict(_X.fillna(0))
            _umabans = [int(r["馬番"]) for _, r in race_df.iterrows()
                        if pd.notna(r.get("馬番"))]
            _score_map = {ub: float(s) for ub, s in zip(_umabans, _model_scores)}
            _combos_with_prob = pl_combo_probs(_score_map, top_n=min(8, len(_score_map)))
            if _combos_with_prob:
                _TOP_N = _playbook_top_n  # playbook 由来 (default 3)
                _selected = [c for c, _ in _combos_with_prob[:_TOP_N]]
                if _selected:
                    n = len(_selected)
                    per_bet = max(100, (budget // n // 100) * 100)
                    bets = [{"馬券種": "三連単", "買い目": f"{f}→{s}→{t}",
                             "購入額": per_bet, "ROI": 0}
                            for f, s, t in sorted(_selected)]
                    info.update({"pattern": f"trifecta_v1[top{_TOP_N}]", "source": "trifecta_model_v1",
                                 "n_combos": n, "first": [], "second": [], "third": []})
                    return bets, info
    except Exception as _tri_e:
        logger.debug(f"trifecta_model_v1 推論失敗、スコア差ルールへフォールバック: {_tri_e}")

    # === Stage 2-01b: Optuna 最適化済み閾値を halo_thresholds.json から読み込み ===
    # backtest 5,309R: スコア差ルール ROI 70.37% > 着順モデル ROI 67.58%
    # OOS 2025: 最適化後 65.15% (+4.36pt vs baseline 60.78%)
    from utils import load_halo_thresholds as _lht
    _hthr = _lht()
    _gap12_hi   = _hthr["gap_12_hi"]    # default 10 → opt 3.68
    _gap12_lo   = _hthr["gap_12_lo"]    # default  5 → opt 2.54
    _gap_top4   = _hthr["gap_top4_lo"]  # default 15 → opt 27.04
    _pw_min     = _hthr["pw_min"]       # default 0.50 → opt 0.73
    _pw_ratio   = _hthr["pw_ratio"]     # default 2.0  → opt 1.58

    scores = race_df.set_index("馬番")["score"].to_dict() if "score" in race_df.columns else {}
    s_hon = float(scores.get(h1, 0))
    s_tai = float(scores.get(h2, 0))
    s_sab = float(scores.get(h_marks.get("▲", -1), 0))
    s_del = float(scores.get(h_marks.get("△", -1), 0))
    gap_12 = s_hon - s_tai
    gap_top4 = s_hon - s_del if s_del > 0 else s_hon - s_sab

    if gap_12 >= _gap12_hi:
        first  = [h1]
        second = [h_marks[m] for m in ["◯","▲"] if m in h_marks]
        third  = [h_marks[m] for m in ["◯","▲","△","☆","★"] if m in h_marks]
        info["pattern"] = "◎突出(スコア差)"
    elif gap_12 <= _gap12_lo and gap_top4 <= _gap_top4:
        first  = [h1, h2]
        second = [h_marks[m] for m in ["◎","◯","▲"] if m in h_marks]
        third  = [h_marks[m] for m in ["◎","◯","▲","△","☆"] if m in h_marks]
        info["pattern"] = "◎◯拮抗(スコア差)"
    else:
        first  = [h1, h2]
        second = [h_marks[m] for m in ["◎","◯","▲","△"] if m in h_marks]
        third  = [h_marks[m] for m in ["◎","◯","▲","△","☆","★"] if m in h_marks]
        info["pattern"] = "標準(スコア差)"

    # === 着順モデルは補強用途のみ参照 (高信頼な ◎突出ケースのみ AI 採用) ===
    # 条件: p_win(◎) ≥ pw_min かつ p_win(◎) ≥ p_win(◯) × pw_ratio
    order_proba = _predict_order_proba(race_df)
    if order_proba is not None:
        try:
            pw = order_proba.set_index("馬番")["p_win"].to_dict()
            pw_hon = float(pw.get(h1, 0))
            pw_tai = float(pw.get(h2, 0))
            if pw_hon >= _pw_min and pw_hon >= pw_tai * _pw_ratio:
                first  = [h1]
                second = [h_marks[m] for m in ["◯","▲"] if m in h_marks]
                third  = [h_marks[m] for m in ["◯","▲","△","☆","★"] if m in h_marks]
                info["pattern"] = "◎突出(着順モデル[高信頼])"
                info["source"] = "order_model_high_conf"
        except Exception:
            # AI 補強失敗時はスコア差判定をそのまま使う
            pass

    info["first"] = first
    info["second"] = second
    info["third"] = third

    # フォーメーション生成（同一馬排除）
    combos = set()
    for f in first:
        for s in second:
            for t in third:
                if len({f, s, t}) == 3:
                    combos.add((f, s, t))

    # 点数が多すぎる場合は3着候補を絞る
    if len(combos) > 36:
        third = third[:4]
        info["third"] = third
        combos = set()
        for f in first:
            for s in second:
                for t in third:
                    if len({f, s, t}) == 3:
                        combos.add((f, s, t))

    if not combos:
        return [], info

    n_combos = len(combos)
    per_bet = max(100, (budget // n_combos // 100) * 100)
    info["n_combos"] = n_combos

    bets = []
    for f, s, t in sorted(combos):
        key = f"{f}→{s}→{t}"
        bets.append({"馬券種":"三連単", "買い目":key, "購入額":per_bet, "ROI":0})

    return bets, info


def get_bets(race_df: pd.DataFrame, place: str, cls_raw: str,
             strategy: dict, budget: int) -> dict:
    """HAHO（三連複◎軸5頭流し）/ HALO（三連単フォーメーション）
       / LALO（複勝◎1点のみ）/ CQC（単勝◎1点のみ）/ TRIPLE（三連複+複勝）を返す。
    戻り値: {"HAHO": [bets...], "HALO": [bets...], "LALO": [bets...], "CQC": [bets...], "TRIPLE": [bets...]}
    Phase 5 (2026-04-05): TRIPLE は strategy_weights.json に無い会場でも生成する。
    （HAHO/HALO/LALO/CQC は引き続き戦略テーブル必須）"""
    if place in EXCLUDE_PLACES or cls_raw in EXCLUDE_CLASSES:
        return {}
    cls      = CLASS_NORMALIZE.get(cls_raw, cls_raw)
    bet_info = strategy.get(place, {}).get(cls) or strategy.get(place, {}).get(cls_raw, {})
    # Phase 5+: SegmentBetFilter 用にレース条件取得
    _meta_row = race_df.iloc[0] if len(race_df) > 0 else None
    _td   = str(_meta_row.get("芝・ダ", "")) if _meta_row is not None else ""
    _dist = _meta_row.get("距離", None) if _meta_row is not None else None
    _seg  = _race_segment(_td, _dist)
    # ── 全マーク馬の馬番を抽出 ──────────────────────────────────
    h_marks = {}
    hon_row = None
    for m in ["◎","◯","▲","△","☆","★"]:
        rows = race_df[race_df["mark"] == m]
        if not rows.empty:
            h_marks[m] = int(rows.iloc[0]["馬番"])
            if m == "◎":
                hon_row = rows.iloc[0]
    if "◎" not in h_marks:
        return {}
    h1 = h_marks["◎"]
    h2 = h_marks.get("◯")
    h3 = h_marks.get("▲")
    others_5 = [h_marks[m] for m in ["◯","▲","△","☆","★"] if m in h_marks]  # ◎以外最大5頭
    opponents_4 = [h_marks[m] for m in ["▲","△","☆","★"] if m in h_marks]    # ◎◯以外最大4頭

    # ◎の単勝オッズ（複勝ガード用）
    hon_tansho = 0.0
    try:
        hon_tansho = float(hon_row.get("単勝", 0) or 0)
    except Exception:
        pass
    MIN_TANSHO_FOR_FUKU = 2.0
    odds_too_low = 0 < hon_tansho < MIN_TANSHO_FOR_FUKU

    # SegmentBetFilter
    san_blocked   = (_seg, "三連複") in SEGMENT_BET_BLACKLIST or _is_class_blacklisted(_seg, cls_raw, "三連複")
    fuku_blocked  = (_seg, "複勝")   in SEGMENT_BET_BLACKLIST or _is_class_blacklisted(_seg, cls_raw, "複勝")
    umaren_blocked= (_seg, "馬連")   in SEGMENT_BET_BLACKLIST or _is_class_blacklisted(_seg, cls_raw, "馬連")
    tansho_blocked= (_seg, "単勝")   in SEGMENT_BET_BLACKLIST

    result = {}

    # ── HAHO: 三連複◎1頭軸-5頭流し（10点×¥1,000）──────────────────
    if len(others_5) >= 2 and not san_blocked:
        import itertools
        haho_bets = []
        per_bet = 1000
        for a, b in itertools.combinations(others_5, 2):
            key = "-".join(map(str, sorted([h1, a, b])))
            haho_bets.append({"馬券種":"三連複", "買い目":key, "購入額":per_bet, "ROI":0})
        result["HAHO"] = haho_bets

    # ── HALO: 三連単フォーメーション（AI自動選択）──────────────────
    if h2 and (_seg, "三連単") not in SEGMENT_BET_BLACKLIST:
        halo_bets, halo_info = _build_sanrentan_formation(race_df, h_marks, budget=9600)
        if halo_bets:
            result["HALO"] = halo_bets
            result["_HALO_INFO"] = halo_info  # UI 表示用メタ情報

    # ── STANDARD: 単複馬連 EV ベース選別（Stage 1-06: 2026-04-16）─────
    # 旧仕様 (◎-◯固定 1 点 + 単勝20%+複勝60%+馬連20%) は撤廃。
    # 単勝 ◎/◯, 複勝 ◎/◯, 馬連 ◎-{◯,▲,△,☆} の各候補を EV 計算し、
    # 閾値 (utils.MIN_EV_*) を通過したものだけ採用。予算は EV 重み配分。
    if not (tansho_blocked and fuku_blocked and umaren_blocked):
        try:
            from ev_gate import (
                make_race_meta, compute_ev_tansho, compute_ev_fuku,
                compute_ev_umaren, pass_ev_gate,
            )
            try:
                _dist_int = int(float(_dist)) if _dist is not None else 1600
            except Exception:
                _dist_int = 1600
            race_meta = make_race_meta(place, _td, _dist_int, len(race_df))

            tansho_series = pd.to_numeric(race_df.get("単勝", pd.Series(dtype=float)), errors="coerce")
            pop_series = tansho_series.rank(method="min", ascending=True)
            pop_map = {int(r["馬番"]): int(p)
                       for (_, r), p in zip(race_df.iterrows(), pop_series)
                       if pd.notna(r.get("馬番")) and pd.notna(p)}

            order_proba = _predict_order_proba(race_df)
            if order_proba is not None and "馬番" in order_proba.columns:
                p_win_map  = {int(r["馬番"]): float(r["p_win"])
                              for _, r in order_proba.iterrows() if pd.notna(r.get("馬番"))}
                p_p23_map  = {int(r["馬番"]): float(r["p_place23"])
                              for _, r in order_proba.iterrows() if pd.notna(r.get("馬番"))}
                p_fuku_map = {k: min(1.0, p_win_map.get(k, 0) + p_p23_map.get(k, 0)) for k in p_win_map}
            else:
                p_win_map  = {int(r["馬番"]): float(r.get("prob", 0))
                              for _, r in race_df.iterrows() if pd.notna(r.get("馬番"))}
                p_fuku_map = {k: min(1.0, v * 2.5) for k, v in p_win_map.items()}

            cand: list[dict] = []  # 候補 buylist

            # 単勝: ◎, ◯
            if not tansho_blocked:
                for mark in ["◎", "◯"]:
                    if mark in h_marks:
                        h = h_marks[mark]
                        p = p_win_map.get(h, 0.0)
                        ev, pay = compute_ev_tansho(p, pop_map.get(h, 7), race_meta)
                        if pass_ev_gate("単勝", ev):
                            cand.append({"馬券種":"単勝","買い目":str(h),"ev":ev,"pay":pay,"p":p})

            # 複勝: ◎, ◯
            if not fuku_blocked:
                for mark in ["◎", "◯"]:
                    if mark in h_marks:
                        h = h_marks[mark]
                        p = p_fuku_map.get(h, p_win_map.get(h, 0.0) * 2.5)
                        ev, pay = compute_ev_fuku(p, pop_map.get(h, 7), race_meta)
                        if pass_ev_gate("複勝", ev):
                            cand.append({"馬券種":"複勝","買い目":str(h),"ev":ev,"pay":pay,"p":p})

            # 馬連: ◎-{◯,▲,△,☆}
            if not umaren_blocked:
                for mark in ["◯", "▲", "△", "☆"]:
                    if mark in h_marks:
                        ho = h_marks[mark]
                        a, b = sorted([h1, ho])
                        # 馬連 p_hit ≈ 「両者とも 3 着以内」÷ 3（粗い近似）
                        p_a = p_fuku_map.get(h1, 0.0)
                        p_b = p_fuku_map.get(ho, 0.0)
                        p_hit = p_a * p_b / 3.0
                        ev, pay = compute_ev_umaren(p_hit, pop_map.get(a, 7), pop_map.get(b, 7), race_meta)
                        if pass_ev_gate("馬連", ev):
                            cand.append({"馬券種":"馬連","買い目":f"{a}-{b}","ev":ev,"pay":pay,"p":p_hit})

            # 予算配分: EV-1 を重みに（簡易 Kelly 近似）。Stage 2 で本格 Kelly に置換。
            if cand:
                pos = [max(0.0, c["ev"] - 1.0) for c in cand]
                tot = sum(pos)
                if tot > 0:
                    weights = [w / tot for w in pos]
                else:
                    weights = [1.0 / len(cand)] * len(cand)
                std_bets = []
                for c, w in zip(cand, weights):
                    amt = floor_to_unit(int(budget * w))
                    if amt > 0:
                        std_bets.append({"馬券種": c["馬券種"], "買い目": c["買い目"],
                                          "購入額": amt, "ROI": 0,
                                          "EV": round(float(c["ev"]), 3)})
                if std_bets:
                    result["STANDARD"] = std_bets
        except Exception as e:
            logger.warning(f"STANDARD EV 計算失敗 {place} {cls_raw}: {e}")

    # ── TRIPLE: 三連複◎◯▲1点(¥1,000) + 複勝◎(残り) ──────────────
    if h2 and h3 and not san_blocked and not fuku_blocked and not odds_too_low:
        FIXED_SAN = 1000
        san_key  = "-".join(map(str, sorted([h1, h2, h3])))
        amt_fuku = max(100, floor_to_unit(budget - FIXED_SAN))
        triple_bets = [
            {"馬券種":"三連複", "買い目":san_key, "購入額":FIXED_SAN, "ROI":0},
            {"馬券種":"複勝",   "買い目":str(h1), "購入額":amt_fuku,  "ROI":0},
        ]
        result["TRIPLE"] = triple_bets

    return result


# =========================================================
# pred CSV 自動保存（Streamlit の表示結果を基準にする）
# =========================================================
def _get_bets_flat(race_df: pd.DataFrame, place: str, cls_raw: str,
                   strategy: dict, budget: int) -> dict:
    """get_bets() の結果を flat dict に変換してCSVエクスポート用に返す。"""
    result: dict = {
        "HAHO_戦略対象": False,
        "HAHO_三連複_買い目": "", "HAHO_三連複_購入額": 0, "HAHO_三連複_点数": 0,
        "HALO_戦略対象": False,
        "HALO_三連単_買い目": "", "HALO_三連単_購入額": 0, "HALO_三連単_点数": 0,
        "STANDARD_戦略対象": False,
        "STANDARD_戦略対象": False,
        "STANDARD_単勝_買い目": "", "STANDARD_単勝_購入額": 0,
        "STANDARD_複勝_買い目": "", "STANDARD_複勝_購入額": 0,
        "STANDARD_馬連_買い目": "", "STANDARD_馬連_購入額": 0,
        "TRIPLE_戦略対象": False,
        "TRIPLE_三連複_買い目": "", "TRIPLE_三連複_購入額": 0,
        "TRIPLE_複勝_買い目": "", "TRIPLE_複勝_購入額": 0,
    }
    bets_all = get_bets(race_df, place, cls_raw, strategy, budget)
    if not bets_all:
        return result

    for plan, bets in bets_all.items():
        if plan.startswith("_"):
            continue  # メタ情報（_HALO_INFO 等）はスキップ
        result[f"{plan}_戦略対象"] = True
        for b in bets:
            typ = b["馬券種"]
            key = f"{plan}_{typ}_買い目"
            amt_key = f"{plan}_{typ}_購入額"
            pts_key = f"{plan}_{typ}_点数"
            if key in result:
                existing = result[key]
                result[key] = f"{existing} / {b['買い目']}" if existing else b["買い目"]
                result[amt_key] = result.get(amt_key, 0) + b["購入額"]
                if pts_key in result:
                    result[pts_key] = result.get(pts_key, 0) + 1

    return result


def save_pred_csv_from_streamlit(all_df: pd.DataFrame, selected_date: str,
                                  strategy: dict, budget: int) -> None:
    """Streamlit が表示した予測をそのまま pred CSV として保存する。
    URL で確認した内容 = 的中実績の基準にするためのエクスポート。"""
    out_path = BASE_DIR / "reports" / f"pred_{selected_date}.csv"
    race_id_col = "レースID(新/馬番無)"
    rows: list[dict] = []

    for race_id, race_df in all_df.groupby(race_id_col):
        race_df = race_df.copy()
        meta    = race_df.iloc[0]
        place   = str(meta.get("場所", "")).strip()
        cls_raw = str(meta.get("クラス名", meta.get("クラス", ""))).strip()
        r_num   = str(meta.get("R", ""))
        date_s  = str(meta.get("日付S", selected_date))
        dist    = str(meta.get("距離", ""))

        if place in EXCLUDE_PLACES or cls_raw in EXCLUDE_CLASSES:
            bets: dict = {
                "HAHO_戦略対象": False,
                "HAHO_馬連_買い目": "", "HAHO_馬連_購入額": 0,
                "HAHO_三連複_買い目": "", "HAHO_三連複_購入額": 0,
                "HALO_戦略対象": False,
                "HALO_三連複_買い目": "", "HALO_三連複_購入額": 0,
                "LALO_戦略対象": False,
                "LALO_複勝_買い目": "", "LALO_複勝_購入額": 0,
                "CQC_戦略対象": False,
                "CQC_単勝_買い目": "", "CQC_単勝_購入額": 0,
            }
        else:
            bets = _get_bets_flat(race_df, place, cls_raw, strategy, budget)

        for _, row in race_df.sort_values("馬番").iterrows():
            is_hon = str(row.get("mark", "")) == "◎"
            rows.append({
                "日付":               date_s,
                "場所":               place,
                "R":                  r_num,
                "クラス":             cls_raw,
                "距離":               dist,
                "レースID":           race_id,
                "馬番":               int(row["馬番"]) if pd.notna(row.get("馬番")) else "",
                "馬名":               str(row.get("馬名", "")),
                "騎手":               str(row.get("騎手", "")),
                "スコア":             float(row.get("score", 0.0)),
                "単勝オッズ":         float(row["単勝"]) if pd.notna(row.get("単勝")) else "",
                "期待値スコア":       float(row.get("ev_score", 0.0)),
                "印":                 str(row.get("mark", "")),
                "HAHO_戦略対象":      "✅" if bets["HAHO_戦略対象"] else "",
                "HAHO_馬連_買い目":   bets["HAHO_馬連_買い目"]   if is_hon else "",
                "HAHO_馬連_購入額":   bets["HAHO_馬連_購入額"]   if is_hon else "",
                "HAHO_三連複_買い目": bets["HAHO_三連複_買い目"] if is_hon else "",
                "HAHO_三連複_購入額": bets["HAHO_三連複_購入額"] if is_hon else "",
                "HALO_戦略対象":      "✅" if bets["HALO_戦略対象"] else "",
                "HALO_三連複_買い目": bets["HALO_三連複_買い目"] if is_hon else "",
                "HALO_三連複_購入額": bets["HALO_三連複_購入額"] if is_hon else "",
                "LALO_戦略対象":      "✅" if bets["LALO_戦略対象"] else "",
                "LALO_複勝_買い目":   bets["LALO_複勝_買い目"]   if is_hon else "",
                "LALO_複勝_購入額":   bets["LALO_複勝_購入額"]   if is_hon else "",
                "CQC_戦略対象":       "✅" if bets["CQC_戦略対象"]  else "",
                "CQC_単勝_買い目":    bets["CQC_単勝_買い目"]    if is_hon else "",
                "CQC_単勝_購入額":    bets["CQC_単勝_購入額"]    if is_hon else "",
            })

    if rows:
        pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
        logger.info(f"pred CSV 保存（Streamlit基準）: {out_path}")


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
        type_keys = ["三連複"]
    elif plan_key == "HALO":
        type_keys = ["三連単"]
    elif plan_key == "TRIPLE":
        type_keys = ["三連複", "複勝"]
    elif plan_key == "STANDARD":
        type_keys = ["単勝", "複勝", "馬連"]
    else:
        type_keys = list(by_type.keys()) or ["三連複"]
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

    # 新フォーマット（HAHO/HALO/LALO/CQC/TRIPLE キーあり）
    _plan_keys = ["HAHO", "HALO", "STANDARD", "TRIPLE"]
    if any(k in results for k in _plan_keys):
        tab_triple, tab_haho, tab_halo, tab_std = st.tabs([
            "🔱 TRIPLE  三連複+複勝",
            "🛡️ HAHO  ◎軸流し",
            "🎯 HALO  三連単マルチ",
            "📋 STANDARD 単複馬連",
        ])
        _res_tab_map = [
            (tab_triple, "TRIPLE"),
            (tab_haho,   "HAHO"),
            (tab_halo,   "HALO"),
            (tab_std,    "STANDARD"),
        ]
        for _tab, _pk in _res_tab_map:
            with _tab:
                _pd = results.get(_pk, {})
                if not _pd or not _pd.get("total", {}).get("races"):
                    st.info(f"{_pk}のデータがありません。generate_results.py を実行してください。")
                else:
                    _render_plan_results(_pd, _pk)
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
                mark = name[0] if name and name[0] in "◎◯▲△☆★" else ""
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
# EV単勝候補ページ
# =========================================================
def page_ev_candidates(all_df: pd.DataFrame) -> None:
    """EV >= 閾値 の◎馬を一覧表示するページ。"""
    st.markdown("### ⭐ EV単勝候補")
    st.caption("◎馬の期待値スコア (モデル確率 × 単勝オッズ / 0.80) が閾値以上のレースを表示")

    col_thr, col_odds = st.columns(2)
    with col_thr:
        ev_thr = st.slider("EV閾値", min_value=0.8, max_value=2.5, value=1.0,
                           step=0.05, format="%.2f")
    with col_odds:
        max_odds = st.slider("単勝オッズ上限（夢馬カット）", min_value=10.0, max_value=99.9,
                             value=50.0, step=1.0, format="%.0f倍")

    hon_df = all_df[all_df["mark"] == "◎"].copy()
    hon_df["ev_score"] = pd.to_numeric(hon_df.get("ev_score", 0.0), errors="coerce").fillna(0.0)
    hon_df["_tansho"]  = pd.to_numeric(hon_df.get("単勝", pd.Series(dtype=float)), errors="coerce").values

    cands = hon_df[(hon_df["ev_score"] >= ev_thr) & (hon_df["_tansho"] <= max_odds)].copy()
    cands = cands.sort_values("ev_score", ascending=False)

    st.metric("該当レース数", f"{len(cands)} R")

    if cands.empty:
        st.info(f"EV >= {ev_thr:.2f} かつ 単勝 ≤ {max_odds:.0f}倍 の◎馬はありません")
        return

    disp_cols = []
    for c in ["場所", "R", "クラス名", "距離", "馬名", "騎手", "単勝", "ev_score", "score"]:
        alt = {"馬名": "馬名S"}.get(c, c)
        if c in cands.columns:
            disp_cols.append(c)
        elif alt in cands.columns:
            disp_cols.append(alt)

    disp = cands[disp_cols].rename(columns={"ev_score": "EV", "score": "スコア(%)"}).reset_index(drop=True)

    def _ev_color(val):
        if isinstance(val, float):
            if val >= 2.0: return "background-color:#2d4a2a; color:#a6e3a1; font-weight:bold"
            if val >= 1.5: return "background-color:#3a3a1a; color:#f9e2af"
        return ""

    if "EV" in disp.columns:
        styled = disp.style.map(_ev_color, subset=["EV"])
        st.dataframe(styled, use_container_width=True, hide_index=True)
    else:
        st.dataframe(disp, use_container_width=True, hide_index=True)

    st.markdown("""
| EV | 目安 |
|---|---|
| ≥ 2.0 | 🟢 バックテストで ROI ~199% |
| ≥ 1.5 | 🟡 バックテストで ROI ~125% |
| ≥ 1.0 | ⚪ 期待値プラス域 |
""")


# =========================================================
# VALUE複勝候補ページ
# =========================================================
def page_value_candidates(all_df: pd.DataFrame) -> None:
    """Value Model v2 の買い候補を表示するページ。"""
    st.markdown("### 💰 VALUE複勝候補")
    st.caption(
        "Value Model v2 (2nd-stage LightGBM) が予測する高ROI複勝候補。"
        "バックテスト: balanced戦略で的中38.5%, ROI 141%（2024年9-12月 out-of-sample）"
    )

    value_obj = _get_cached(VALUE_MODEL_PATH, "value_model")
    if value_obj is None:
        st.warning("value_model_v2.pkl が見つかりません。train_value_model.py を実行してください。")
        return

    strategies = value_obj.get("strategies", {})

    col_strat, col_info = st.columns([2, 3])
    with col_strat:
        strat_name = st.selectbox(
            "戦略",
            options=list(strategies.keys()),
            index=0,
            format_func=lambda k: strategies[k].get("desc", k),
        )
    strat      = strategies[strat_name]
    pr_thr     = strat["pred_roi_thr"]
    cp_thr     = strat["cal_prob_thr"]
    with col_info:
        st.info(f"pred_roi ≥ {pr_thr}  かつ  cal_prob ≥ {cp_thr}")

    # value_score / cal_prob が存在するか確認
    if "value_score" not in all_df.columns or "cal_prob" not in all_df.columns:
        st.warning("Value Scoreが計算されていません。CSVを再読込してください。")
        return

    df_v = all_df.copy()
    df_v["value_score"] = pd.to_numeric(df_v["value_score"], errors="coerce")
    df_v["cal_prob"]    = pd.to_numeric(df_v["cal_prob"],    errors="coerce")

    cands = df_v[(df_v["value_score"] >= pr_thr) & (df_v["cal_prob"] >= cp_thr)].copy()
    cands = cands.sort_values("value_score", ascending=False)

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("該当頭数", f"{len(cands)} 頭")
    col_m2.metric("対象レース", f"{cands['レースID(新/馬番無)'].nunique()} R")
    col_m3.metric("平均ValueScore", f"{cands['value_score'].mean():.2f}" if len(cands) > 0 else "—")

    if cands.empty:
        st.info(f"該当なし (pred_roi≥{pr_thr}, cal_prob≥{cp_thr})")
        return

    disp_cols = []
    for c in ["場所", "R", "クラス名", "距離", "馬名", "馬名S", "印", "騎手", "単勝",
              "value_score", "cal_prob"]:
        if c in cands.columns and c not in disp_cols:
            disp_cols.append(c)
    # 馬名は馬名S優先
    if "馬名S" in disp_cols and "馬名" in disp_cols:
        disp_cols.remove("馬名")

    disp = cands[disp_cols].rename(columns={
        "馬名S": "馬名",
        "value_score": "ValueScore",
        "cal_prob": "CalProb(%)",
    }).copy()
    if "CalProb(%)" in disp.columns:
        disp["CalProb(%)"] = (disp["CalProb(%)"] * 100).round(1)

    def _val_color(val):
        if isinstance(val, (int, float)):
            if val >= 1.1: return "background-color:#2d4a2a; color:#a6e3a1; font-weight:bold"
            if val >= 0.95: return "background-color:#3a3a1a; color:#f9e2af"
        return ""

    if "ValueScore" in disp.columns:
        styled = disp.style.map(_val_color, subset=["ValueScore"])
        st.dataframe(styled, use_container_width=True, hide_index=True)
    else:
        st.dataframe(disp, use_container_width=True, hide_index=True)

    st.markdown("""
| ValueScore | 目安 |
|---|---|
| ≥ 1.1 | 🟢 高確度VALUE候補 |
| ≥ 0.95 | 🟡 VALUE候補 |
| ≥ 0.88 | ⚪ balanced閾値域 |
""")


# =========================================================
# プラン選択ガイドページ
# =========================================================
def page_plan_selector(all_df: pd.DataFrame, strategy: dict, budget: int) -> None:
    """全レース横断で「どのプランを買うか」を1画面に提示する専用タブ。"""
    import math

    race_id_col = "レースID(新/馬番無)"

    # ── データ収集 ──────────────────────────────────────────
    rows = []
    for race_id, grp in all_df.groupby(race_id_col):
        meta     = grp.iloc[0]
        place    = str(meta.get("場所", ""))
        cls_raw  = str(meta.get("クラス名", ""))
        hon_row  = grp[grp["mark"] == "◎"]
        if hon_row.empty:
            continue
        hon_name  = str(hon_row.iloc[0]["馬名"])
        hon_score = float(hon_row.iloc[0].get("score", 0) or 0)
        ev_score  = float(hon_row.iloc[0].get("ev_score", 0) or 0)

        bets_all = get_bets(grp, place, cls_raw, strategy, budget)
        if not bets_all:
            continue

        halo_bets  = bets_all.get("HALO",     [])
        halo_info  = bets_all.get("_HALO_INFO", {})
        haho_bets  = bets_all.get("HAHO",     [])
        std_bets   = bets_all.get("STANDARD", [])
        triple_bets= bets_all.get("TRIPLE",   [])

        halo_src   = halo_info.get("source", "") if halo_info else ""
        halo_pat   = halo_info.get("pattern", "") if halo_info else ""
        halo_ai    = halo_src in ("trifecta_model_v1", "order_model", "order_model_high_conf")

        # ── 推奨プラン決定ロジック ─────────────────────────
        # スコアリング: 数値が高いほど優先
        plan_scores: dict[str, float] = {}
        if halo_bets:
            base = 80.0 if halo_ai else 50.0
            plan_scores["HALO"] = base + min(ev_score * 5, 20)
        if std_bets:
            std_ev_max = max((b.get("EV", 1.0) for b in std_bets), default=1.0)
            plan_scores["STANDARD"] = 40.0 + (std_ev_max - 1.0) * 30
        if haho_bets:
            plan_scores["HAHO"] = 20.0
        if triple_bets:
            plan_scores["TRIPLE"] = 15.0

        if not plan_scores:
            continue

        rec_plan = max(plan_scores, key=lambda k: plan_scores[k])

        # 推奨プランの買い目テキスト
        def _fmt_bets(bets: list) -> tuple[str, int]:
            lines, total = [], 0
            for b in bets:
                lines.append(f'{b.get("馬券種","")}{b.get("買い目","")} ¥{b.get("購入額",0):,}')
                total += int(b.get("購入額", 0))
            return " / ".join(lines[:4]) + ("…" if len(lines) > 4 else ""), total

        rec_bets_map = {
            "HALO": halo_bets, "STANDARD": std_bets,
            "HAHO": haho_bets, "TRIPLE": triple_bets,
        }
        rec_bets_text, rec_total = _fmt_bets(rec_bets_map.get(rec_plan, []))

        # 推奨理由テキスト
        if rec_plan == "HALO":
            reason = f"{'🧠AI ' + halo_pat if halo_ai else '📐Rule ' + halo_pat}"
            if ev_score >= 1.3:
                reason += f" EV:{ev_score:.2f}"
        elif rec_plan == "STANDARD":
            std_ev_max = max((b.get("EV", 1.0) for b in std_bets), default=1.0)
            types = list(dict.fromkeys(b.get("馬券種","") for b in std_bets))
            reason = f"EV:{std_ev_max:.2f} {'+'.join(types)}"
        elif rec_plan == "HAHO":
            reason = f"三連複◎軸{len(haho_bets)}点"
        else:
            reason = f"三連複+複勝{len(triple_bets)}点"

        # 有効プラン一覧
        active = [p for p in ["HALO","STANDARD","HAHO","TRIPLE"] if p in plan_scores and p != rec_plan]

        rows.append({
            "_race_id":    race_id,
            "場所":        place,
            "R":           int(meta.get("R", 0)),
            "クラス":      cls_raw,
            "距離":        f'{meta.get("芝・ダ","")}{meta.get("距離","")}m',
            "発走":        str(meta.get("発走時刻", "")),
            "◎":           hon_name,
            "スコア":      round(hon_score, 1),
            "推奨":        rec_plan,
            "理由":        reason,
            "買い目":      rec_bets_text,
            "総額":        rec_total,
            "他プラン":    " ".join(active),
            # 詳細表示用
            "_halo_bets":  halo_bets,
            "_halo_info":  halo_info,
            "_std_bets":   std_bets,
            "_haho_bets":  haho_bets,
            "_triple_bets":triple_bets,
            "_plan_scores":plan_scores,
        })

    if not rows:
        st.info("データがありません。CSVを読み込んでください。")
        return

    # ── フィルター UI ───────────────────────────────────────
    all_places = sorted({r["場所"] for r in rows})
    col_f1, col_f2 = st.columns([2, 2])
    with col_f1:
        sel_places = st.multiselect("会場", all_places, default=all_places, key="ps_place")
    with col_f2:
        sel_plan = st.selectbox("推奨プランで絞り込み",
                                ["すべて","HALO","STANDARD","HAHO","TRIPLE"],
                                key="ps_plan_filter")

    filtered = [r for r in rows
                if r["場所"] in sel_places
                and (sel_plan == "すべて" or r["推奨"] == sel_plan)]

    if not filtered:
        st.warning("該当レースなし")
        return

    # ── サマリ統計 ─────────────────────────────────────────
    total_budget = sum(r["総額"] for r in filtered)
    plan_counts  = {}
    for r in filtered:
        plan_counts[r["推奨"]] = plan_counts.get(r["推奨"], 0) + 1

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("対象レース数", f"{len(filtered)} R")
    mc2.metric("推奨総投資額", f"¥{total_budget:,}")
    mc3.metric("主力プラン", max(plan_counts, key=lambda k: plan_counts[k]) if plan_counts else "-")
    mc4.metric("HALO(AI)本数", sum(1 for r in filtered if r["推奨"]=="HALO" and "🧠AI" in r["理由"]))

    st.divider()

    # ── レース別テーブル + 詳細 expander ──────────────────────
    _PLAN_COLOR = {
        "HALO":     "#89b4fa",   # blue
        "STANDARD": "#a6e3a1",   # green
        "HAHO":     "#fab387",   # peach
        "TRIPLE":   "#cba6f7",   # mauve
    }

    for r in sorted(filtered, key=lambda x: (x["場所"], x["R"])):
        plan  = r["推奨"]
        color = _PLAN_COLOR.get(plan, "#cdd6f4")

        # ヘッダ行
        hcol1, hcol2, hcol3, hcol4, hcol5 = st.columns([1, 0.6, 1.8, 2.5, 1.5])
        with hcol1:
            st.markdown(
                f'<span style="font-size:13px;color:#cdd6f4">'
                f'<b>{r["場所"]}</b>{r["R"]}R&nbsp;{r["発走"]}</span>',
                unsafe_allow_html=True)
        with hcol2:
            st.markdown(
                f'<span style="background:{color};color:#1e1e2e;'
                f'padding:2px 8px;border-radius:4px;font-weight:bold;font-size:12px">'
                f'{plan}</span>',
                unsafe_allow_html=True)
        with hcol3:
            st.markdown(
                f'<span style="color:#cdd6f4;font-size:12px">◎ {r["◎"]}'
                f' <span style="color:#888">(スコア:{r["スコア"]})</span></span>',
                unsafe_allow_html=True)
        with hcol4:
            st.markdown(
                f'<span style="color:#a6e3a1;font-size:11px">{r["理由"]}</span>',
                unsafe_allow_html=True)
        with hcol5:
            other_str = r["他プラン"]
            st.markdown(
                f'<span style="color:#888;font-size:11px">'
                f'¥{r["総額"]:,}'
                + (f' | 他: {other_str}' if other_str else '')
                + '</span>',
                unsafe_allow_html=True)

        # expander で全プランの詳細
        with st.expander(f'　詳細・プラン切替 [{r["クラス"]} {r["距離"]}]', expanded=False):
            det_tabs_list = []
            det_labels    = []
            bets_by_plan  = {
                "HALO":     r["_halo_bets"],
                "STANDARD": r["_std_bets"],
                "HAHO":     r["_haho_bets"],
                "TRIPLE":   r["_triple_bets"],
            }
            for p in ["HALO","STANDARD","HAHO","TRIPLE"]:
                if bets_by_plan[p]:
                    star = "⭐ " if p == plan else ""
                    det_labels.append(f"{star}{p}")
                    det_tabs_list.append(p)

            if det_labels:
                tabs_obj = st.tabs(det_labels)
                for tab_obj, p_name in zip(tabs_obj, det_tabs_list):
                    with tab_obj:
                        bets = bets_by_plan[p_name]
                        if p_name == "HALO" and r["_halo_info"]:
                            info  = r["_halo_info"]
                            src   = info.get("source","")
                            pat   = info.get("pattern","")
                            src_tag = "🧠 AIモデル" if src in ("trifecta_model_v1","order_model","order_model_high_conf") else "📐 スコアルール"
                            f_lst = info.get("first",  [])
                            s_lst = info.get("second", [])
                            t_lst = info.get("third",  [])
                            fmt = ""
                            if f_lst and s_lst and t_lst:
                                fmt = f'{"・".join(map(str,f_lst))} → {"・".join(map(str,s_lst))} → {"・".join(map(str,t_lst))}'
                            st.caption(f"{src_tag}  パターン: {pat}  {fmt}")
                        total_p = 0
                        for b in bets:
                            amt = int(b.get("購入額", 0))
                            total_p += amt
                            ev_str = f'  EV:{b["EV"]:.2f}' if "EV" in b else ""
                            st.markdown(
                                f'<span style="color:#cdd6f4">'
                                f'{b.get("馬券種","")} '
                                f'<b>{b.get("買い目","")}</b>'
                                f'</span>'
                                f'<span style="color:#888;margin-left:8px">¥{amt:,}{ev_str}</span>',
                                unsafe_allow_html=True)
                        st.markdown(
                            f'<span style="color:#f38ba8;font-size:12px">合計 ¥{total_p:,}</span>',
                            unsafe_allow_html=True)

        st.markdown('<hr style="border-color:#313244;margin:4px 0">', unsafe_allow_html=True)


# =========================================================
# 今日の買い目専用ページ
# =========================================================
def page_buylist(all_df: pd.DataFrame, strategy: dict, budget: int) -> None:
    """今日の買い目一覧ページ（HAHO/HALO プラン切替）。"""
    race_id_col = "レースID(新/馬番無)"
    # Stage 1-08: countdown 用に日付を取得
    if "日付S" in all_df.columns and not all_df.empty:
        selected_date = str(all_df["日付S"].iloc[0])
    else:
        selected_date = ""
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
            "race_id":    race_id,
            "場所":       place,
            "R":          int(meta.get("R", 0)),
            "クラス":     cls_raw,
            "発走":       str(meta.get("発走時刻", "")),
            "距離":       f'{meta.get("芝・ダ","")}{meta.get("距離","")}m',
            "馬場":       _normalize_baba(str(meta.get("馬場状態",""))),
            "◎":          hon_name,
            "◎スコア":    hon_score,
            "HAHO_bets":    bets_all.get("HAHO",     []),
            "HALO_bets":    bets_all.get("HALO",     []),
            "HALO_info":    bets_all.get("_HALO_INFO", {}),
            "STANDARD_bets":bets_all.get("STANDARD", []),
            "TRIPLE_bets":  bets_all.get("TRIPLE",   []),
        })

    # デバッグ可視化（Phase 5+: 阪神表示問題切り分け用）
    n_total          = all_df[race_id_col].nunique()
    n_meta           = len(race_metas)
    venue_counts     = {}
    venue_triple     = {}
    for r in race_metas:
        v = r["場所"]
        venue_counts[v] = venue_counts.get(v, 0) + 1
        if r["TRIPLE_bets"]:
            venue_triple[v] = venue_triple.get(v, 0) + 1
    venue_str = " / ".join(
        f"{v}{venue_counts[v]}R(TRIPLE可{venue_triple.get(v,0)})"
        for v in sorted(venue_counts.keys())
    )
    st.caption(
        f"📊 全{n_total}R / get_bets通過{n_meta}R / 会場別: {venue_str or '(なし)'}"
    )

    if not race_metas:
        st.info("本日の買い目対象レースがありません。")
        return

    # プラン選択
    plan = st.radio(
        "プラン選択",
        ["🔱 TRIPLE  三連複1点＋複勝（¥1,000固定＋残り複勝）",
         "🛡️ HAHO   三連複◎軸5頭流し（10点×¥1,000）",
         "🎯 HALO   三連単フォーメーション（AI自動選択）",
         "📋 STANDARD 単勝＋複勝＋馬連（2:6:2）"],
        horizontal=True, key="buylist_plan",
    )
    if plan.startswith("🔱"):
        plan_key = "TRIPLE_bets"
    elif plan.startswith("🛡️"):
        plan_key = "HAHO_bets"
    elif plan.startswith("🎯"):
        plan_key = "HALO_bets"
    else:
        plan_key = "STANDARD_bets"

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

        # ヘッダー行（発走時刻カウントダウン付き）
        _bet_cd = countdown_html(selected_date, r["発走"])
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:12px;'
            f'padding:10px 0 4px;border-top:1px solid #2a2a3e;margin-top:4px">'
            f'<span style="background:#313244;color:#cdd6f4;border-radius:4px;'
            f'padding:2px 10px;font-weight:bold;font-size:15px">{r["場所"]} {r["R"]}R</span>'
            f'<span style="color:#888;font-size:13px">{r["クラス"]}　{r["発走"]}{_bet_cd}　{r["距離"]}　馬場:{r["馬場"]}</span>'
            f'<span style="color:#a6e3a1;font-size:13px;margin-left:auto">'
            f'◎{r["◎"]}　{r["◎スコア"]:.1f}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # 買い目行
        sant  = [b for b in bets if b["馬券種"] == "三連単"]

        lines = []
        if rengo:
            combos = "　".join(b["買い目"] for b in rengo)
            amt    = sum(b["購入額"] for b in rengo)
            lines.append(
                f'<span style="color:#888;min-width:60px;display:inline-block">馬連</span>'
                f'<span style="color:#cdd6f4">{combos}</span>'
                f'<span style="color:#888;margin-left:8px">¥{amt:,}</span>'
            )
        if sanf:
            n = len(sanf)
            per = sanf[0]["購入額"]
            amt = sum(b["購入額"] for b in sanf)
            if n <= 3:
                combos = "　".join(b["買い目"] for b in sanf)
                lines.append(
                    f'<span style="color:#888;min-width:60px;display:inline-block">三連複</span>'
                    f'<span style="color:#cdd6f4">{combos}</span>'
                    f'<span style="color:#888;margin-left:8px">¥{amt:,}</span>'
                )
            else:
                lines.append(
                    f'<span style="color:#888;min-width:60px;display:inline-block">三連複</span>'
                    f'<span style="color:#cdd6f4">◎軸{n}点 @¥{per:,}</span>'
                    f'<span style="color:#888;margin-left:8px">計¥{amt:,}</span>'
                )
        if sant:
            n = len(sant)
            per = sant[0]["購入額"]
            amt = sum(b["購入額"] for b in sant)
            info = r.get("HALO_info", {}) if isinstance(r, dict) else {}
            pattern = info.get("pattern", "フォーメーション")
            src     = info.get("source", "")
            src_tag = "🧠AI" if src in ("order_model", "trifecta_model_v1", "order_model_high_conf") else "📐Rule"
            first   = info.get("first", [])
            second  = info.get("second", [])
            third   = info.get("third", [])
            fmt_str = ""
            if first and second and third:
                fmt_str = (
                    f'<span style="color:#89b4fa">'
                    f'{"・".join(map(str,first))} → {"・".join(map(str,second))} → {"・".join(map(str,third))}'
                    f'</span>'
                )
            lines.append(
                f'<span style="color:#888;min-width:60px;display:inline-block">三連単</span>'
                f'<span style="color:#cdd6f4">{pattern} {n}点 @¥{per:,}</span>'
                f'<span style="color:#fab387;margin-left:8px;font-size:11px">{src_tag}</span>'
                f'<span style="color:#888;margin-left:8px">計¥{amt:,}</span>'
            )
            if fmt_str:
                lines.append(
                    f'<span style="color:#888;min-width:60px;display:inline-block"></span>'
                    f'{fmt_str}'
                )
        if fuku:
            combos = "　".join(b["買い目"] for b in fuku)
            amt    = sum(b["購入額"] for b in fuku)
            lines.append(
                f'<span style="color:#888;min-width:60px;display:inline-block">複勝</span>'
                f'<span style="color:#cdd6f4">◎ {combos}番</span>'
                f'<span style="color:#888;margin-left:8px">¥{amt:,}</span>'
            )
        if tan:
            combos = "　".join(b["買い目"] for b in tan)
            amt    = sum(b["購入額"] for b in tan)
            lines.append(
                f'<span style="color:#888;min-width:60px;display:inline-block">単勝</span>'
                f'<span style="color:#cdd6f4">◎ {combos}番</span>'
                f'<span style="color:#888;margin-left:8px">¥{amt:,}</span>'
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
    # Stage 1-08: countdown 用に日付を取得
    if "日付S" in all_df.columns and not all_df.empty:
        selected_date = str(all_df["日付S"].iloc[0])
    else:
        selected_date = ""

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
        _banner_cd = countdown_html(selected_date, mr["発走"], size="13px")
        st.markdown(
            f'<div class="main-race-banner">'
            f'<div style="margin-bottom:6px">'
            f'<span class="{grade_cls}">{grade_label}</span>'
            f'<span style="color:#888;font-size:13px">{mr["場所"]} {mr["R"]}R　{mr["発走"]}発走{_banner_cd}</span></div>'
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

            _row_cd = countdown_html(selected_date, r["発走"])
            st.markdown(
                f'<div class="race-row">'
                f'<span class="{"r-badge" if not excluded else "r-badge-ex"}">{r["R"]}R</span>'
                f'<div style="flex:1">'
                f'<span style="font-size:18px;color:#cdd6f4">{r["クラス"]}</span>{badge}'
                f'<span style="color:#888;font-size:14px;margin-left:8px">'
                f'{r["発走"]}{_row_cd}　{r["距離"]}　馬場:{r["馬場"]}　{r["頭数"]}頭</span>'
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
# =========================================================
# PyCa 出走馬評価リスト (全頭分析タブ用)
# =========================================================
PYCA_INDICATORS = [
    ("a", "総合力",   ["score"],                                            True),
    ("b", "スピード",  ["前走補正", "前走走破タイム"],                        True),
    ("c", "末脚",     ["前走補9", "前走上り3F"],                             True),
    ("d", "前走成績", ["前走確定着順"],                                      False),  # 小さい=良い
    ("e", "市場評価", ["前走人気", "前走単勝オッズ"],                         False),  # 小さい=良い
    ("f", "ペース適性", ["前走RPCI", "前走PCI3", "前走Ave-3F"],              True),
]


def _norm_0_10(series: pd.Series) -> tuple[pd.Series, bool]:
    """レース内で 0〜10 に min-max 正規化。
    戻り値: (正規化済み Series, valid(有効フラグ))
    欠損全部 or 全馬同値なら valid=False, 5.0 固定。
    """
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series([5.0] * len(series), index=series.index), False
    vmin, vmax = s.min(), s.max()
    if vmax - vmin < 1e-9:
        return pd.Series([5.0] * len(series), index=series.index), False
    out = (s - vmin) / (vmax - vmin) * 10.0
    return out.fillna(5.0), True


def _pyca_index(row_score: float, indicators: dict) -> float:
    """PyCaLi総合指数 (0-100)。ensemble score (0-100) を主軸に各指標で微調整。"""
    base = float(row_score)  # score は既に 0-100
    boost = sum(indicators.values()) / max(len(indicators), 1) - 5.0   # -5..+5
    return float(max(0.0, min(100.0, base + boost * 1.5)))


def _pyca_radar_fig(values: list[float], labels: list[str], name: str):
    """6軸レーダーチャート (matplotlib polar)。"""
    import matplotlib.pyplot as _plt
    import numpy as _np
    n = len(values)
    angles = _np.linspace(0, 2 * _np.pi, n, endpoint=False).tolist()
    vals   = values + values[:1]
    angs   = angles + angles[:1]
    fig = _plt.figure(figsize=(3.2, 3.2), facecolor="#1e1e2e")
    ax = fig.add_subplot(111, polar=True, facecolor="#181825")
    ax.plot(angs, vals, color="#89b4fa", linewidth=2)
    ax.fill(angs, vals, color="#89b4fa", alpha=0.28)
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(["2","4","6","8","10"], color="#6c7086", fontsize=7)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, color="#cdd6f4", fontsize=9)
    ax.grid(color="#313244", linewidth=0.8)
    ax.spines["polar"].set_color("#313244")
    ax.set_title(name, color="#f5e0dc", fontsize=10, pad=8)
    fig.tight_layout()
    return fig


def render_pyca_evaluation_list(race_df: pd.DataFrame) -> None:
    """全頭分析タブ: スコア＋指標内訳の評価リスト。"""
    st.markdown("### 🔍 出走馬評価リスト（PyCaLi指数）")
    st.caption(
        "**PyCaLi指数** = アンサンブルモデル(LightGBM+CatBoost)の複勝確率予測。"
        "右側の a〜f は指数算出に使われた特徴量の内訳（レース内 0〜10 正規化）。"
    )

    df = race_df.sort_values("馬番").reset_index(drop=True).copy()

    # 指標値を 0〜10 正規化して DF に追加 (valid フラグも保持)
    # 各指標は候補カラムを順に試し、レース内で分散がある最初のカラムを採用
    norm_cols: dict[str, str] = {}
    valid_map: dict[str, bool] = {}
    for key, label, col_candidates, higher in PYCA_INDICATORS:
        nkey = f"_pyca_{key}"
        used_norm, used_valid = None, False
        for col in col_candidates:
            if col not in df.columns:
                continue
            norm, valid = _norm_0_10(df[col])
            if valid:
                used_norm, used_valid = norm, True
                break
            if used_norm is None:
                used_norm = norm   # フォールバック用 (全 5.0)
        if used_norm is None:
            used_norm = pd.Series([5.0] * len(df), index=df.index)
        df[nkey] = used_norm
        if used_valid and not higher:
            df[nkey] = 10.0 - df[nkey]
        valid_map[key] = used_valid
        norm_cols[key] = nkey

    # レース内順位 (値の大きい順, 1=最良)。valid=False なら順位 0 (表示で "−")
    rank_cols: dict[str, str] = {}
    for key, _, _, _ in PYCA_INDICATORS:
        rkey = f"_rank_{key}"
        if valid_map[key]:
            df[rkey] = df[norm_cols[key]].rank(ascending=False, method="min").astype(int)
        else:
            df[rkey] = 0
        rank_cols[key] = rkey

    # スコア（= モデル予測そのまま。加減補正なし）
    df["_pyca"] = df["score"].astype(float)
    df["_pyca_rank"] = df["_pyca"].rank(ascending=False, method="min").astype(int)

    labels = [lbl for _, lbl, _, _ in PYCA_INDICATORS]

    for _, row in df.iterrows():
        name  = str(row.get("馬名", f"{int(row['馬番'])}番"))
        uma   = int(row.get("馬番", 0))
        mark  = str(row.get("mark", ""))
        sex   = str(row.get("性齢", ""))
        kin   = row.get("斤量", "")
        jockey= str(row.get("騎手", ""))
        pyca  = float(row["_pyca"])
        prank = int(row["_pyca_rank"])
        mk_cls  = MARK_CLASS.get(mark, "")
        mk_html = f'<span class="{mk_cls}">{mark}</span> ' if mark else ""

        c_left, c_mid, c_right = st.columns([2, 2, 3], gap="small")
        with c_left:
            st.markdown(
                f'<div style="padding:4px 0;line-height:1.5">'
                f'<div style="font-size:17px;color:#6c7086">{uma}番</div>'
                f'<div style="font-size:24px;font-weight:bold">{mk_html}{name}</div>'
                f'<div style="font-size:17px;color:#a6adc8">{sex} / {kin}kg / {jockey}</div>'
                f'<div style="margin-top:14px">'
                f'<span style="font-size:17px;color:#6c7086">PyCaLi指数</span><br>'
                f'<span style="font-size:48px;font-weight:bold;color:#89b4fa">{pyca:.1f}</span>'
                f'<span style="font-size:20px;color:#cdd6f4;margin-left:8px">({prank}位)</span>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            try:
                cur_date = str(row.get("日付", "")) if "日付" in row.index else None
                past = _real_pycali_history(name, cur_date, n=5)
                series = past + [float(row["_pyca"])]
                if len(series) >= 2:
                    _sp = _make_sparkline(series)
                    st.pyplot(_sp, use_container_width=False)
                    plt.close(_sp)
                    st.caption(
                        f"PyCaLi指数履歴 ({len(past)}走+今走): "
                        + " → ".join(f"{v:.0f}" for v in series)
                    )
                else:
                    st.caption("PyCaLi指数履歴: 過去出走データなし（初出走扱い）")
            except Exception as _e:
                logger.debug(f"sparkline skip: {_e}")
        with c_mid:
            vals = [float(row[norm_cols[k]]) for k, *_ in PYCA_INDICATORS]
            fig = _pyca_radar_fig(vals, labels, name)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with c_right:
            st.markdown(
                '<div style="font-size:17px;color:#6c7086;margin-bottom:6px">指数内訳 / 値 / レース内順位</div>',
                unsafe_allow_html=True,
            )
            rows_html = []
            for key, label, _, _ in PYCA_INDICATORS:
                v = float(row[norm_cols[key]])
                rk = int(row[rank_cols[key]])
                valid = valid_map[key]
                bar = int(round(v * 10))  # 0..100
                if not valid:
                    color = "#6c7086"
                    rank_txt = "−"
                    top_mark = ""
                else:
                    color = "#a6e3a1" if rk == 1 else ("#89b4fa" if rk <= 3 else "#cdd6f4")
                    rank_txt = f"{rk}位"
                    top_mark = "★" if rk <= 3 else ""
                rows_html.append(
                    f'<div style="display:flex;align-items:center;gap:8px;margin:5px 0;font-size:18px">'
                    f'<div style="width:90px;color:#a6adc8">{key}. {label}</div>'
                    f'<div style="flex:1;height:10px;background:#313244;border-radius:4px;overflow:hidden">'
                    f'<div style="height:100%;width:{bar}%;background:{color}"></div>'
                    f'</div>'
                    f'<div style="width:46px;text-align:right;color:{color};font-weight:bold">{v:.1f}</div>'
                    f'<div style="width:56px;text-align:right;color:#6c7086">{rank_txt}{top_mark}</div>'
                    f'</div>'
                )
            st.markdown("".join(rows_html), unsafe_allow_html=True)
        st.markdown("<hr style='border-color:#313244;margin:8px 0'>", unsafe_allow_html=True)


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

        render_danger_favorite_badge(race_df)

        tab1, tab2, tab3 = st.tabs(["📋 出走表 / 買い目", "🔍 全頭分析", "🏇 コース分析"])

        with tab1:
            st.markdown(
                '### 出走表　<span style="font-size:14px;color:#cdd6f4;font-weight:normal">'
                '◎ &gt; ◯ &gt; ▲ &gt; △ &gt; ☆ &gt; ★</span>',
                unsafe_allow_html=True,
            )
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
                    det_tabs = st.tabs([
                        "🔱 TRIPLE",
                        "🛡️ HAHO ◎軸流し",
                        "🎯 HALO 三連単マルチ",
                        "📋 STANDARD 単複馬連",
                    ])
                    _no_bet_msg = {
                        "TRIPLE":   "このレースはTRIPLE対象外です（三連複or複勝がブロック）。",
                        "HAHO":     "このレースはHAHO対象外です（◎以外の馬不足or三連複ブロック）。",
                        "HALO":     "このレースはHALO対象外です（◎◯不足or三連単ブロック）。",
                        "STANDARD": "このレースはSTANDARD対象外です（3券種揃わず）。",
                    }
                    for det_tab, plan_key, plan_label in [
                        (det_tabs[0], "TRIPLE",   "三連複◎◯▲1点(¥1,000) + 複勝◎(残り)"),
                        (det_tabs[1], "HAHO",     "三連複◎1頭軸-5頭流し（10点×¥1,000）"),
                        (det_tabs[2], "HALO",     "三連単フォーメーション（AI自動選択）"),
                        (det_tabs[3], "STANDARD", "単勝◎(20%) + 複勝◎(60%) + 馬連◎-◯(20%)"),
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
                            # HALO: フォーメーション情報表示
                            if plan_key == "HALO":
                                halo_info = bets_all.get("_HALO_INFO", {})
                                if halo_info:
                                    pattern = halo_info.get("pattern", "")
                                    src = halo_info.get("source", "")
                                    src_tag = "🧠 AIモデル" if src in ("order_model", "trifecta_model_v1", "order_model_high_conf") else "📐 スコアルール"
                                    first  = halo_info.get("first", [])
                                    second = halo_info.get("second", [])
                                    third  = halo_info.get("third", [])
                                    st.markdown(
                                        f'<div style="background:#1e1e2e;border:1px solid #313244;'
                                        f'border-radius:6px;padding:8px 12px;margin:6px 0;font-size:13px">'
                                        f'<div style="color:#fab387;margin-bottom:4px">'
                                        f'<b>パターン:</b> {pattern} '
                                        f'<span style="color:#6c7086;margin-left:8px">判定:{src_tag}</span></div>'
                                        f'<div style="color:#89b4fa">'
                                        f'1着: [{", ".join(map(str,first))}] → '
                                        f'2着: [{", ".join(map(str,second))}] → '
                                        f'3着: [{", ".join(map(str,third))}]'
                                        f'</div></div>',
                                        unsafe_allow_html=True,
                                    )
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
                                    f'<span style="color:#888;font-size:12px">計{type_total:,}円</span>'
                                    f'</div><div>{combos_html}</div></div>',
                                    unsafe_allow_html=True,
                                )

        with tab2:
            render_pyca_evaluation_list(race_df)


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

    # cache_key にモデルの更新時刻を含める（モデル再訓練後も正しく再計算される）
    # Stage 1-07: precompute parquet があれば優先 → 画面切替の待ち時間ゼロ化
    import os
    _precompute_pq = BASE_DIR / "reports" / f"buylist_horses_{selected_date}.parquet"
    if _precompute_pq.exists():
        _pq_mtime = os.path.getmtime(str(_precompute_pq))
        _cache_key = f"PRECOMPUTED:{selected_date}_{int(_pq_mtime)}"
        logger.info(f"[predict_all_races] precompute parquet 発見 → 高速パス採用 ({_precompute_pq.name})")
    else:
        _model_mtime = max(
            os.path.getmtime(str(LGBM_PATH)) if LGBM_PATH.exists() else 0,
            os.path.getmtime(str(CAT_PATH))  if CAT_PATH.exists()  else 0,
        )
        _cache_key = f"{selected_date}_{int(_model_mtime)}"
    predicted_json = predict_all_races(_cache_key, raw_df.to_json(), lgbm_obj, cat_obj)
    import io
    all_df = pd.read_json(io.StringIO(predicted_json))

    # ── Streamlit の表示結果を pred CSV として自動保存（的中実績の基準）──
    try:
        save_pred_csv_from_streamlit(all_df, selected_date, strategy, budget)
    except Exception as _e:
        logger.warning(f"pred CSV 自動保存失敗: {_e}")

    if "selected_race_id" not in st.session_state:
        st.session_state.selected_race_id = None
    if "selected_place" not in st.session_state:
        st.session_state.selected_place = ""
    if "show_buylist" not in st.session_state:
        st.session_state.show_buylist = False

    race_id_col = "レースID(新/馬番無)"

    # メインタブ
    main_tab1, main_tab2, main_tab_ps, main_tab3, main_tab4, main_tab5, main_tab6, main_tab7 = st.tabs(
        ["🏇 レース予想", "🎫 今日の買い目", "🎯 プラン選択", "⭐ EV候補", "💰 VALUE複勝", "📊 的中実績",
         "📊 ROIヒートマップ", "📋 結果フィードバック"]
    )

    results = load_results()

    with main_tab6:
        render_roi_heatmap()
    with main_tab7:
        render_feedback_dashboard()

    with main_tab5:
        page_results(results)

    with main_tab4:
        page_value_candidates(all_df)

    with main_tab3:
        page_ev_candidates(all_df)

    with main_tab_ps:
        page_plan_selector(all_df, strategy, budget)

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
