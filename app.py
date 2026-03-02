"""
app.py
PyCaLiAI - Streamlit UI（週末CSV対応版）

Usage:
    streamlit run app.py
"""

from __future__ import annotations

import io
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
fm._load_fontmanager(try_read_cache=False)
import streamlit as st
japanese_fonts = [f.name for f in fm.fontManager.ttflist if any(
    x in f.name for x in ["IPA", "Noto", "Gothic", "Hiragino", "Yu"]
)]
st.write("利用可能な日本語フォント:", japanese_fonts)
ipa_fonts = [f.fname for f in fm.fontManager.ttflist if "IPA" in f.name]
if ipa_fonts:
    fm.fontManager.addfont(ipa_fonts[0])
    prop = fm.FontProperties(fname=ipa_fonts[0])
    plt.rcParams["font.family"] = prop.get_name()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
DATA_DIR      = BASE_DIR / "data"
MODEL_DIR     = BASE_DIR / "models"
STRATEGY_JSON = DATA_DIR / "strategy_weights.json"
LGBM_PATH     = MODEL_DIR / "lgbm_optuna_v1.pkl"
CAT_PATH      = MODEL_DIR / "catboost_optuna_v1.pkl"

MIN_UNIT = 100
MARKS    = ["◎", "◯", "▲", "△", "×"]

CLASS_NORMALIZE = {
    "新馬": "新馬", "未勝利": "未勝利",
    "1勝": "1勝", "500万": "1勝",
    "2勝": "2勝", "1000万": "2勝",
    "3勝": "3勝", "1600万": "3勝",
    "OP(L)": "OP(L)", "オープン": "ｵｰﾌﾟﾝ",
    "Ｇ１": "Ｇ１", "Ｇ２": "Ｇ２", "Ｇ３": "Ｇ３",
}

FEATURE_LABEL = {
    "前走確定着順": "前走着順", "前走上り3F": "前走上り",
    "枠番": "枠番", "馬番": "馬番", "斤量": "斤量",
    "距離": "距離", "前走距離": "前走距離",
    "前走着差タイム": "前走着差", "前走走破タイム": "前走タイム",
}

MARK_CLASS = {
    "◎": "mk-hon", "◯": "mk-tai", "▲": "mk-sabo",
    "△": "mk-del", "×": "mk-batu", "": "",
}

# ターゲットCSV列名 → マスター相当列名
COLUMN_MAP = {
    # 馬情報
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
    # 前走情報（共通）
    "前走着順":       "前走確定着順",
    "前走上り3F":     "前走上り3F",
    "前走TD":        "前芝・ダ",
    "前走間隔":       "間隔",
    # 前走情報（46列版で追加）
    "前走着差":       "前走着差タイム",
    "前走斤量":       "前走斤量",
    "前走Ave3F":     "前走平均3F",
    "前走上り3F順位":  "前走上り3F順位",
    "マイニング順位":  "マイニング順位",
    "前走単勝オッズ":  "前走単勝オッズ",
    "前走通過1":      "前走通過1",
    "前走通過2":      "前走通過2",
    "前走通過3":      "前走通過3",
    "前走通過4":      "前走通過4",
}

RACE_COLS  = [
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


# =========================================================
# CSV パース
# =========================================================
def parse_target_csv(source) -> pd.DataFrame:
    """ターゲット形式CSV（2段構造）をマスター相当のDataFrameに変換する。

    Args:
        source: ファイルパス(str/Path) または Streamlit UploadedFile
    """
    if isinstance(source, (str, Path)):
        with open(source, "rb") as f:
            raw = f.read()
    else:
        raw = source.read()
    text = ""
    for enc in ["cp932", "shift_jis", "utf-8"]:
        try:
            text = raw.decode(enc)
            break
        except Exception:
            continue
    if not text:
        return pd.DataFrame()

    lines = text.splitlines()
    races: list[dict] = []
    current_race: dict | None = None

    # 列数から自動判定（33列=旧形式 / 46列=新形式）
    for line in lines:
        cols = line.split(",")
        if len(cols) == 19 and cols[0] not in ("レースID(新)", ""):
            current_race = dict(zip(RACE_COLS, cols))
        elif len(cols) == 33 and cols[0] not in ("枠番", "") and current_race:
            horse = dict(zip(HORSE_COLS_33, cols))
            horse.update(current_race)
            races.append(horse)
        elif len(cols) == 46 and cols[0] not in ("枠番", "") and current_race:
            horse = dict(zip(HORSE_COLS_46, cols))
            horse.update(current_race)
            races.append(horse)

    df = pd.DataFrame(races)
    if df.empty:
        return df

    # 列名リネーム
    df = df.rename(columns=COLUMN_MAP)

    # レースID正規化（馬番なし形式に）
    df["レースID(新/馬番無)"] = df["レースID(新)"].astype(str).str[:16]

    # 数値変換
    for col in ["枠番","馬番","斤量","ZI","ZI順位","距離","人気","単勝",
                "前走確定着順","前走上り3F","前走距離","間隔","前走人気",
                "前走着差タイム","前走斤量","前走平均3F","前走上り3F順位",
                "マイニング順位","前走単勝オッズ",
                "前走通過1","前走通過2","前走通過3","前走通過4"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 日付
    df["日付"] = pd.to_datetime(df["日付S"], format="%Y.%m.%d", errors="coerce")
    df["日付"] = df["日付"].dt.strftime("%Y%m%d").astype("Int64")

    # 欠損補完（モデルが必要な列を0埋め）
    for col in ["前走走破タイム","前走着差タイム","馬体重","馬体重増減",
                "前走斤量","生産者","馬主(最新/仮想)"]:
        if col not in df.columns:
            df[col] = 0

    return df


# =========================================================
# モデルロード
# =========================================================
@st.cache_resource(show_spinner="モデル読み込み中...")
def load_models() -> tuple:
    return joblib.load(LGBM_PATH), joblib.load(CAT_PATH)


@st.cache_data(show_spinner="戦略データ読み込み中...")
def load_strategy() -> dict:
    with open(STRATEGY_JSON, encoding="utf-8") as f:
        return json.load(f)


# =========================================================
# 予測
# =========================================================
def parse_time_str(series: pd.Series) -> pd.Series:
    def _conv(val):
        try:
            parts = str(val).strip().split(".")
            if len(parts) == 3:
                return int(parts[0]) * 60 + int(parts[1]) + int(parts[2]) / 10
            return float(val)
        except Exception:
            return None
    return series.apply(_conv)


def predict_lgbm(df: pd.DataFrame, obj: dict) -> np.ndarray:
    model, encoders, feature_cols = obj["model"], obj["encoders"], obj["feature_cols"]
    df = df.copy()
    for col in ["前走走破タイム", "前走着差タイム"]:
        if col in df.columns:
            df[col] = parse_time_str(df[col])
    for col, le in encoders.items():
        if col not in df.columns:
            df[col] = 0
            continue
        df[col] = df[col].astype(str).fillna("__NaN__")
        known = set(le.classes_)
        df[col] = df[col].apply(lambda x: x if x in known else "__NaN__")
        if "__NaN__" not in le.classes_:
            le.classes_ = np.append(le.classes_, "__NaN__")
        df[col] = le.transform(df[col])
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
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
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    cat_idx = [i for i, c in enumerate(feature_cols) if c in cat_list]
    pool = Pool(df[feature_cols], cat_features=cat_idx)
    return model.predict_proba(pool)[:, 1]


def ensemble_predict(df: pd.DataFrame, lgbm_obj: dict, cat_obj: dict) -> np.ndarray:
    return 0.5 * predict_lgbm(df, lgbm_obj) + 0.5 * predict_catboost(df, cat_obj)


def assign_marks(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["mark"] = ""
    ranked = df["prob"].rank(ascending=False, method="first")
    for idx, rank in ranked.items():
        if rank <= 5:
            df.at[idx, "mark"] = {1:"◎",2:"◯",3:"▲",4:"△",5:"×"}[int(rank)]
    return df


# =========================================================
# 全レース一括予想
# =========================================================
@st.cache_data(show_spinner="全レース予想計算中...")
def predict_all_races(cache_key: str, df_json: str, _lgbm_obj: dict, _cat_obj: dict) -> str:
    """全レース一括でスコア・印を付与してJSON返却。"""
    df = pd.read_json(df_json)
    result_frames = []
    race_id_col = "レースID(新/馬番無)"

    for race_id, race_df in df.groupby(race_id_col):
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
    df = pd.read_json(df_json)
    model, encoders, feature_cols = _lgbm_obj["model"], _lgbm_obj["encoders"], _lgbm_obj["feature_cols"]
    df_enc = df.copy()
    for col in ["前走走破タイム","前走着差タイム"]:
        if col in df_enc.columns:
            df_enc[col] = parse_time_str(df_enc[col])
    for col, le in encoders.items():
        if col not in df_enc.columns:
            df_enc[col] = 0
            continue
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
    sv     = np.array(sv_row)
    order  = np.argsort(np.abs(sv))[::-1][:12]
    labels = [FEATURE_LABEL.get(feature_cols[i], feature_cols[i]) for i in order]
    values = sv[order]
    colors = ["tomato" if v > 0 else "steelblue" for v in values]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.barh(labels[::-1], values[::-1], color=colors[::-1])
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_title(horse_name, fontsize=10)
    ax.set_xlabel("SHAP値（赤=好材料 / 青=不安材料）", fontsize=8)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig


def make_comment(sv_row: list, feature_cols: list, horse_name: str,
                 score: float, mark: str = "") -> str:
    sv    = np.array(sv_row)
    pairs = sorted(zip(sv, feature_cols), reverse=True)
    pos   = [(v, c) for v, c in pairs if v > 0][:3]
    neg   = [(v, c) for v, c in pairs if v < 0][-2:]

    if score >= 60:
        level = "モデルが最上位クラスの評価を与えており"
    elif score >= 40:
        level = "上位圏の評価を与えており"
    elif score >= 20:
        level = "中位圏の評価を与えており"
    else:
        level = "下位圏の評価を与えており"

    mark_txt = {
        "◎": "本命として最も信頼できる一頭。",
        "◯": "対抗として本命を脅かす存在。",
        "▲": "単穴として一発の魅力がある。",
        "△": "連下候補として抑えておきたい。",
        "×": "押さえ程度だが圏外とも言えない。",
    }.get(mark, "")

    pos_s = [f"{FEATURE_LABEL.get(c,c)}が好材料" for _, c in pos]
    neg_s = [f"{FEATURE_LABEL.get(c,c)}がやや不安" for _, c in neg]

    lines = []
    if mark_txt:
        lines.append(mark_txt)
    lines.append(f"{horse_name}はスコア{score:.1f}%で、{level}上位入線の可能性を評価しています。")
    if pos_s:
        lines.append("好材料として、" + "、".join(pos_s) + "が挙げられます。")
    if neg_s:
        lines.append("一方で" + "、".join(neg_s) + "点があります。")
    else:
        lines.append("目立った不安材料は少なく安定感のある評価です。")
    return " ".join(lines)


# =========================================================
# 買い目生成
# =========================================================
def floor_to_unit(x: int, unit: int = MIN_UNIT) -> int:
    return max((x // unit) * unit, unit)


def get_bets(race_df: pd.DataFrame, place: str, cls_raw: str,
             strategy: dict, budget: int) -> list[dict]:
    cls      = CLASS_NORMALIZE.get(cls_raw, cls_raw)
    bet_info = strategy.get(place, {}).get(cls) or strategy.get(place, {}).get(cls_raw, {})
    if not bet_info:
        return []
    marks_df = {m: race_df[race_df["mark"] == m] for m in MARKS}
    hon    = marks_df["◎"]
    taikou = marks_df["◯"]
    sabo   = marks_df["▲"]
    delta  = marks_df["△"]
    if hon.empty:
        return []
    h1 = int(hon.iloc[0]["馬番"])
    h2 = int(taikou.iloc[0]["馬番"]) if not taikou.empty else None
    h3 = int(sabo.iloc[0]["馬番"])   if not sabo.empty  else None
    h4 = int(delta.iloc[0]["馬番"])  if not delta.empty else None
    top3 = [h for h in [h1, h2, h3] if h is not None]
    top4 = [h for h in [h1, h2, h3, h4] if h is not None]
    results = []
    for bet_type, info in bet_info.items():
        amt = floor_to_unit(int(budget * info["bet_ratio"]))
        if bet_type == "複勝":
            results.append({"馬券種":"複勝","買い目":str(h1),"購入額":amt,
                            "ROI":info["roi"],"ウェイト":round(info["weight"]*100,1)})
        elif bet_type == "馬連":
            if h2 is None: continue
            results.append({"馬券種":"馬連","買い目":"-".join(map(str,sorted([h1,h2]))),
                            "購入額":amt,"ROI":info["roi"],"ウェイト":round(info["weight"]*100,1)})
        elif bet_type == "三連複":
            if len(top3) < 3: continue
            combos  = list(itertools.combinations(top4[:4],3))[:3]
            per_bet = floor_to_unit(amt // max(len(combos),1))
            for c in combos:
                results.append({"馬券種":"三連複","買い目":"-".join(map(str,sorted(c))),
                                "購入額":per_bet,"ROI":info["roi"],"ウェイト":round(info["weight"]*100,1)})
        elif bet_type == "三連単":
            if len(top3) < 3: continue
            perms   = list(itertools.permutations(top3[:3],3))[:3]
            per_bet = floor_to_unit(amt // max(len(perms),1))
            for p in perms:
                results.append({"馬券種":"三連単","買い目":"-".join(map(str,p)),
                                "購入額":per_bet,"ROI":info["roi"],"ウェイト":round(info["weight"]*100,1)})
    return results


# =========================================================
# CSS
# =========================================================
CSS = """
<style>
.tbl-header {
    display:grid;
    grid-template-columns:40px 50px 50px 1fr 70px 60px 130px 160px;
    background:#1e1e2e; color:#cdd6f4; font-weight:bold;
    font-size:13px; padding:6px 12px; border-radius:6px 6px 0 0; gap:8px;
}
.tbl-row {
    display:grid;
    grid-template-columns:40px 50px 50px 1fr 70px 60px 130px 160px;
    font-size:13px; padding:7px 12px;
    border-bottom:1px solid #313244; align-items:center; gap:8px;
}
.mk-hon  {color:#e74c3c;font-weight:bold;font-size:16px;}
.mk-tai  {color:#e67e22;font-weight:bold;font-size:16px;}
.mk-sabo {color:#f1c40f;font-weight:bold;}
.mk-del  {color:#2ecc71;}
.mk-batu {color:#95a5a6;}
.sbar-wrap{display:flex;align-items:center;gap:6px;font-size:12px;}
.sbar{height:10px;border-radius:4px;background:#5865f2;}
/* レース一覧テーブル */
.race-list-header {
    display:grid;
    grid-template-columns:60px 40px 80px 1fr 60px 80px 80px 100px 60px;
    background:#1e1e2e; color:#cdd6f4; font-weight:bold;
    font-size:13px; padding:6px 12px; border-radius:6px 6px 0 0; gap:6px;
}
.race-list-row {
    display:grid;
    grid-template-columns:60px 40px 80px 1fr 60px 80px 80px 100px 60px;
    font-size:13px; padding:6px 12px;
    border-bottom:1px solid #313244; align-items:center; gap:6px;
}
.strategy-badge {
    background:#2d4a2d; color:#4ade80;
    border-radius:4px; padding:2px 6px; font-size:11px;
}
</style>
"""


# =========================================================
# レース一覧ページ（会場別縦列）
# =========================================================
def page_race_list(
    all_df: pd.DataFrame,
    strategy: dict,
    budget: int,
) -> None:
    st.markdown("### 📅 レース一覧")

    race_id_col = "レースID(新/馬番無)"

    # 会場別にレースリストを作成
    by_place: dict[str, list] = {}
    for race_id, grp in all_df.groupby(race_id_col):
        meta     = grp.iloc[0]
        place    = str(meta.get("場所",""))
        cls_raw  = str(meta.get("クラス名",""))
        cls_norm = CLASS_NORMALIZE.get(cls_raw, cls_raw)
        in_strat = (
            place in strategy and
            (cls_norm in strategy[place] or cls_raw in strategy[place])
        )
        hon_row  = grp[grp["mark"] == "◎"]
        hon_name = str(hon_row.iloc[0]["馬名"]) if not hon_row.empty else "-"
        entry = {
            "race_id": race_id,
            "場所":    place,
            "R":       int(meta.get("R", 0)),
            "クラス":  cls_raw,
            "距離":    f'{meta.get("芝・ダ","")}{meta.get("距離","")}m',
            "発走":    str(meta.get("発走時刻","")),
            "天気":    str(meta.get("天気","")),
            "馬場":    str(meta.get("馬場状態","")),
            "◎":       hon_name,
            "戦略":    in_strat,
            "頭数":    len(grp),
        }
        by_place.setdefault(place, []).append(entry)

    # 会場ごとにRでソート
    for place in by_place:
        by_place[place].sort(key=lambda x: x["R"])

    places = list(by_place.keys())
    cols   = st.columns(len(places))

    for col, place in zip(cols, places):
        with col:
            races_in_place = by_place[place]
            meta0  = races_in_place[0]
            # 会場ヘッダー
            st.markdown(
                f'<div style="background:#1e1e2e;border-radius:8px 8px 0 0;'
                f'padding:10px 14px;margin-bottom:0">' 
                f'<span style="font-size:20px;font-weight:bold;color:#cdd6f4">{place}</span>'
                f'<span style="font-size:16px;color:#888;margin-left:8px">'
                f'天気:{meta0["天気"]} 馬場:{meta0["馬場"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            for r in races_in_place:
                badge_html = (
                    '<span style="background:#2d4a2d;color:#4ade80;'
                    'border-radius:2px;padding:1px 5px;font-size:10px">✅</span>'
                    if r["戦略"] else ""
                )
                st.markdown(
                    f'<div style="border-bottom:1px solid #313244;padding:7px 4px;'
                    f'display:flex;align-items:center;gap:6px">' 
                    f'<span style="background:#e74c3c;color:#fff;border-radius:4px;'
                    f'padding:2px 7px;font-size:24px;font-weight:bold;min-width:28px;text-align:center">'
                    f'{r["R"]}R</span>'
                    f'<div style="flex:1;min-width:0">'
                    f'<div style="font-size:20px;color:#cdd6f4;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">'
                    f'{r["クラス"]} {badge_html}</div>'
                    f'<div style="font-size:16px;color:#888">{r["発走"]} {r["距離"]} {r["頭数"]}頭</div>'
                    f'<div style="font-size:11px;color:#a6e3a1">◎ {r["◎"]}</div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )
                if st.button("詳細→", key=f'btn_{r["race_id"]}'):
                    st.session_state.selected_race_id = r["race_id"]
                    st.rerun()


# =========================================================
# 出走表ページ
# =========================================================
def page_race_detail(
    race_df: pd.DataFrame,
    strategy: dict,
    budget: int,
    lgbm_obj: dict,
) -> None:
    meta    = race_df.iloc[0]
    place   = str(meta.get("場所",""))
    cls_raw = str(meta.get("クラス名",""))
    dist    = meta.get("距離","")
    shida   = meta.get("芝・ダ","")

    cls_norm    = CLASS_NORMALIZE.get(cls_raw, cls_raw)
    in_strategy = (
        place in strategy and
        (cls_norm in strategy[place] or cls_raw in strategy[place])
    )

    # 戻るボタン
    if st.button("← レース一覧に戻る"):
        st.session_state.selected_race_id = None
        st.rerun()

    st.markdown(f"## {place} {meta.get('R','')}R / {cls_raw} / {shida}{dist}m")

    if in_strategy:
        cls_key  = cls_norm if cls_norm in strategy.get(place,{}) else cls_raw
        roi_vals = [v["roi"] for v in strategy[place][cls_key].values()]
        st.success(f"✅ 戦略対象レース　平均ROI: {sum(roi_vals)/len(roi_vals):.1f}%")
    else:
        st.info("ℹ️ 戦略対象外（参考予想）")

    # SHAP計算
    shap_ok = False
    shap_vals: list = []
    feature_cols: list = []
    with st.spinner("SHAP計算中..."):
        try:
            shap_vals, feature_cols = compute_shap(lgbm_obj, race_df.to_json())
            shap_ok = True
        except Exception as e:
            st.warning(f"SHAP計算失敗: {e}")

    tab1, tab2 = st.tabs(["📋 出走表 / 買い目", "🔍 全頭分析"])

    # ---- Tab1: 出走表 + 買い目 ----
    with tab1:
        st.markdown("### 出走表")
        st.markdown(
            '<div class="tbl-header">'
            '<span>枠</span><span>馬番</span><span>印</span>'
            '<span>馬名</span><span>性齢</span><span>斤量</span>'
            '<span>騎手</span><span>スコア</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        race_sorted = race_df.sort_values("馬番").reset_index(drop=True)
        for _, row in race_sorted.iterrows():
            mark   = row.get("mark","")
            ban    = int(row.get("馬番",0))
            waku   = int(row.get("枠番",0))
            name   = str(row.get("馬名",f"{ban}番"))
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
                f'</div>',
                unsafe_allow_html=True,
            )

        if in_strategy:
            st.markdown("---")
            st.markdown("### 🎯 買い目")
            bets = get_bets(race_df, place, cls_raw, strategy, budget)
            if not bets:
                st.warning("買い目を生成できませんでした。")
            else:
                bets_df = pd.DataFrame(bets)
                total   = bets_df["購入額"].sum()
                m1, m2, m3 = st.columns(3)
                m1.metric("合計購入額", f"{total:,}円")
                m2.metric("馬券種数",   f"{bets_df['馬券種'].nunique()}種")
                m3.metric("総点数",     f"{len(bets_df)}点")

                summary = (
                    bets_df.groupby("馬券種")
                    .agg(ROI=("ROI","first"), ウェイト=("ウェイト","first"),
                         点数=("買い目","count"), 合計=("購入額","sum"))
                    .reset_index().sort_values("ROI", ascending=False)
                )
                summary["ROI"]    = summary["ROI"].apply(lambda x: f"{x:.1f}%")
                summary["ウェイト"] = summary["ウェイト"].apply(lambda x: f"{x:.1f}%")
                summary["合計"]   = summary["合計"].apply(lambda x: f"{x:,}円")
                st.dataframe(summary, use_container_width=True, hide_index=True)

                disp = bets_df[["馬券種","買い目","購入額","ROI"]].copy()
                disp["購入額"] = disp["購入額"].apply(lambda x: f"{x:,}円")
                disp["ROI"]   = disp["ROI"].apply(lambda x: f"{x:.1f}%")
                st.dataframe(disp, use_container_width=True, hide_index=True)

    # ---- Tab2: 全頭分析 ----
    with tab2:
        if not shap_ok:
            st.error("SHAP計算に失敗しました。")
            return
        st.markdown("### 🔍 全頭分析")
        race_sorted = race_df.sort_values("馬番").reset_index(drop=True)
        for i, row in race_sorted.iterrows():
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
                    f'<div style="font-size:14px;color:#a6adc8;line-height:1.7">{comment}</div>',
                    unsafe_allow_html=True,
                )
            with c_mid:
                fig = make_shap_fig(sv_row, feature_cols, name)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            with c_right:
                for v, c in pos4:
                    label = FEATURE_LABEL.get(c,c)
                    st.markdown(f'<div style="color:#e74c3c;font-size:12px;margin:2px 0">🟥 {label} +{v:.3f}</div>',
                                unsafe_allow_html=True)
                for v, c in neg3:
                    label = FEATURE_LABEL.get(c,c)
                    st.markdown(f'<div style="color:#5865f2;font-size:12px;margin:2px 0">🟦 {label} {v:.3f}</div>',
                                unsafe_allow_html=True)
            st.markdown("<hr style='border-color:#313244;margin:8px 0'>", unsafe_allow_html=True)


# =========================================================
# main
# =========================================================
def main() -> None:
    st.set_page_config(page_title="PyCaLiAI", page_icon="🏇", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("🏇 PyCaLiAI 競馬予想システム")

    lgbm_obj, cat_obj = load_models()
    strategy          = load_strategy()

    # weekly ディレクトリ
    weekly_dir = BASE_DIR / "data" / "weekly"
    weekly_dir.mkdir(parents=True, exist_ok=True)

    # 利用可能なCSVファイル一覧（20260301.csv 形式）
    csv_files = sorted(weekly_dir.glob("????????.csv"), reverse=True)
    date_options = [f.stem for f in csv_files]  # yyyymmdd 文字列

    # サイドバー
    with st.sidebar:
        st.header("📅 出走表CSV")
        if date_options:
            selected_date = st.selectbox(
                "開催日を選択",
                date_options,
                format_func=lambda x: f"{x[:4]}/{x[4:6]}/{x[6:]} ({x})",
            )
        else:
            selected_date = None
            st.info("CSVがありません。E:\\PyCaLiAI\\data\\weekly\\ に 20260301.csv 形式で保存してください。")

        st.divider()
        st.markdown("**CSVを追加する場合**")
        uploaded = st.file_uploader("CSVをアップロードして保存", type=["csv"],
                                    help="ファイル名は自動で日付から付けます")
        if uploaded is not None:
            # ファイル名から日付取得（なければ today）
            stem = Path(uploaded.name).stem
            if stem.isdigit() and len(stem) == 8:
                save_name = f"{stem}.csv"
            else:
                import datetime
                save_name = datetime.date.today().strftime("%Y%m%d") + ".csv"
            save_path = weekly_dir / save_name
            save_path.write_bytes(uploaded.getvalue())
            st.success(f"保存: {save_name}")
            st.rerun()

        st.divider()
        budget = st.number_input("1レース予算（円）", min_value=1_000,
                                 max_value=1_000_000, value=10_000, step=1_000)

    if selected_date is None:
        st.info("サイドバーの指示に従ってCSVを追加してください。")
        return

    csv_path = weekly_dir / f"{selected_date}.csv"

    # CSV パース
    with st.spinner("CSV読み込み中..."):
        raw_df = parse_target_csv(csv_path)
        if raw_df.empty:
            st.error("CSVの読み込みに失敗しました。")
            return

    # 全レース一括予想（キャッシュ済みなら再計算しない）
    predicted_json = predict_all_races(selected_date, raw_df.to_json(), lgbm_obj, cat_obj)
    all_df         = pd.read_json(predicted_json)

    # session_state でレース選択管理
    if "selected_race_id" not in st.session_state:
        st.session_state.selected_race_id = None

    if st.session_state.selected_race_id is None:
        # レース一覧表示
        race_id_col = "レースID(新/馬番無)"
        n_races     = all_df[race_id_col].nunique()
        n_strategy  = sum(
            1 for _, grp in all_df.groupby(race_id_col)
            if (lambda p, c: p in strategy and (
                CLASS_NORMALIZE.get(c,c) in strategy[p] or c in strategy[p]
            ))(str(grp.iloc[0].get("場所","")), str(grp.iloc[0].get("クラス名","")))
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("レース数",     f"{n_races}R")
        c2.metric("頭数",         f"{len(all_df)}頭")
        c3.metric("戦略対象レース", f"{n_strategy}R")
        st.markdown("---")
        page_race_list(all_df, strategy, budget)
    else:
        # レース詳細表示
        race_id_col = "レースID(新/馬番無)"
        race_df     = all_df[all_df[race_id_col] == st.session_state.selected_race_id].copy()
        race_df     = race_df.reset_index(drop=True)
        page_race_detail(race_df, strategy, budget, lgbm_obj)


if __name__ == "__main__":
    main()