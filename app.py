"""
app.py
PyCaLiAI - Streamlit UI・磯ｱ譛ｫCSV蟇ｾ蠢懃沿・・
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
    plt.rcParams["font.family"] = "MS Gothic"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR      = Path(r"E:\PyCaLiAI")
DATA_DIR      = BASE_DIR / "data"
MODEL_DIR     = BASE_DIR / "models"
STRATEGY_JSON = DATA_DIR / "strategy_weights.json"
LGBM_PATH     = MODEL_DIR / "lgbm_optuna_v1.pkl"
CAT_PATH      = MODEL_DIR / "catboost_optuna_v1.pkl"

MIN_UNIT = 100
MARKS    = ["笳・, "笳ｯ", "笆ｲ", "笆ｳ", "ﾃ・]

CLASS_NORMALIZE = {
    "譁ｰ鬥ｬ": "譁ｰ鬥ｬ", "譛ｪ蜍晏茜": "譛ｪ蜍晏茜",
    "1蜍・: "1蜍・, "500荳・: "1蜍・,
    "2蜍・: "2蜍・, "1000荳・: "2蜍・,
    "3蜍・: "3蜍・, "1600荳・: "3蜍・,
    "OP(L)": "OP(L)", "繧ｪ繝ｼ繝励Φ": "・ｵ・ｰ・鯉ｾ滂ｾ・,
    "・ｧ・・: "・ｧ・・, "・ｧ・・: "・ｧ・・, "・ｧ・・: "・ｧ・・,
}

FEATURE_LABEL = {
    "蜑崎ｵｰ遒ｺ螳夂捩鬆・: "蜑崎ｵｰ逹鬆・, "蜑崎ｵｰ荳翫ｊ3F": "蜑崎ｵｰ荳翫ｊ",
    "譫逡ｪ": "譫逡ｪ", "鬥ｬ逡ｪ": "鬥ｬ逡ｪ", "譁､驥・: "譁､驥・,
    "霍晞屬": "霍晞屬", "蜑崎ｵｰ霍晞屬": "蜑崎ｵｰ霍晞屬",
    "蜑崎ｵｰ逹蟾ｮ繧ｿ繧､繝": "蜑崎ｵｰ逹蟾ｮ", "蜑崎ｵｰ襍ｰ遐ｴ繧ｿ繧､繝": "蜑崎ｵｰ繧ｿ繧､繝",
}

MARK_CLASS = {
    "笳・: "mk-hon", "笳ｯ": "mk-tai", "笆ｲ": "mk-sabo",
    "笆ｳ": "mk-del", "ﾃ・: "mk-batu", "": "",
}

# 繧ｿ繝ｼ繧ｲ繝・ヨCSV蛻怜錐 竊・繝槭せ繧ｿ繝ｼ逶ｸ蠖灘・蜷・COLUMN_MAP = {
    # 鬥ｬ諠・ｱ
    "鬥ｬ蜷拘":         "鬥ｬ蜷・,
    "闃昴・繝繝ｼ繝・:     "闃昴・繝",
    "鬥ｬ蝣ｴ迥ｶ諷・證ｫ螳・": "鬥ｬ蝣ｴ迥ｶ諷・,
    "螟ｩ蛟・證ｫ螳・":    "螟ｩ豌・,
    "莠ｺ豌誉莉願ｵｰ":     "莠ｺ豌・,
    "ZI鬆・:         "ZI鬆・ｽ・,
    "辷ｶ":            "遞ｮ迚｡鬥ｬ",
    "豈咲宛":          "豈咲宛鬥ｬ",
    "辷ｶ繧ｿ繧､繝・:       "辷ｶ繧ｿ繧､繝怜錐",
    "豈咲宛繧ｿ繧､繝・:     "豈咲宛繧ｿ繧､繝怜錐",
    # 蜑崎ｵｰ諠・ｱ・亥・騾夲ｼ・    "蜑崎ｵｰ逹鬆・:       "蜑崎ｵｰ遒ｺ螳夂捩鬆・,
    "蜑崎ｵｰ荳翫ｊ3F":     "蜑崎ｵｰ荳翫ｊ3F",
    "蜑崎ｵｰTD":        "蜑崎茅繝ｻ繝",
    "蜑崎ｵｰ髢馴囈":       "髢馴囈",
    # 蜑崎ｵｰ諠・ｱ・・6蛻礼沿縺ｧ霑ｽ蜉・・    "蜑崎ｵｰ逹蟾ｮ":       "蜑崎ｵｰ逹蟾ｮ繧ｿ繧､繝",
    "蜑崎ｵｰ譁､驥・:       "蜑崎ｵｰ譁､驥・,
    "蜑崎ｵｰAve3F":     "蜑崎ｵｰ蟷ｳ蝮・F",
    "蜑崎ｵｰ荳翫ｊ3F鬆・ｽ・:  "蜑崎ｵｰ荳翫ｊ3F鬆・ｽ・,
    "繝槭う繝九Φ繧ｰ鬆・ｽ・:  "繝槭う繝九Φ繧ｰ鬆・ｽ・,
    "蜑崎ｵｰ蜊伜享繧ｪ繝・ぜ":  "蜑崎ｵｰ蜊伜享繧ｪ繝・ぜ",
    "蜑崎ｵｰ騾夐℃1":      "蜑崎ｵｰ騾夐℃1",
    "蜑崎ｵｰ騾夐℃2":      "蜑崎ｵｰ騾夐℃2",
    "蜑崎ｵｰ騾夐℃3":      "蜑崎ｵｰ騾夐℃3",
    "蜑崎ｵｰ騾夐℃4":      "蜑崎ｵｰ騾夐℃4",
}

RACE_COLS  = [
    "繝ｬ繝ｼ繧ｹID(譁ｰ)","譌･莉牢","譖懈律","蝣ｴ謇","髢句ぎ","R","繝ｬ繝ｼ繧ｹ蜷・,"繧ｯ繝ｩ繧ｹ蜷・,
    "闃昴・繝繝ｼ繝・,"霍晞屬","繧ｳ繝ｼ繧ｹ蛹ｺ蛻・,"繧ｳ繝ｼ繝翫・蝗樊焚","鬥ｬ蝣ｴ迥ｶ諷・證ｫ螳・","螟ｩ蛟・證ｫ螳・",
    "繝輔Ν繧ｲ繝ｼ繝磯ｭ謨ｰ","逋ｺ襍ｰ譎ょ綾","諤ｧ蛻･髯仙ｮ・,"驥埼㍼遞ｮ蛻･","蟷ｴ鮨｢髯仙ｮ・,
]
HORSE_COLS_33 = [
    "譫逡ｪ","B","鬥ｬ逡ｪ","鬥ｬ蜷拘","諤ｧ蛻･","蟷ｴ鮨｢","莠ｺ豌誉莉願ｵｰ","蜊伜享","ZI蜊ｰ","ZI","ZI鬆・,
    "譁､驥・,"貂娥","譖ｿ","鬨取焔","謇螻・,"隱ｿ謨吝ｸｫ","辷ｶ","豈咲宛","辷ｶ繧ｿ繧､繝・,"豈咲宛繧ｿ繧､繝・,
    "蜑崎ｵｰ譛・,"蜑崎ｵｰ譌･","蜑崎ｵｰ蝣ｴ謇","蜑崎ｵｰTD","蜑崎ｵｰ霍晞屬","蜑崎ｵｰ鬥ｬ蝣ｴ迥ｶ諷・,"蜑崎ｵｰ逹鬆・,
    "蜑崎ｵｰ莠ｺ豌・,"蜑崎ｵｰ繝ｬ繝ｼ繧ｹ蜷・,"蜑崎ｵｰ荳翫ｊ3F","蜑崎ｵｰ豎ｺ謇・,"蜑崎ｵｰ髢馴囈",
]
HORSE_COLS_46 = [
    "譫逡ｪ","B","鬥ｬ逡ｪ","鬥ｬ蜷拘","諤ｧ蛻･","蟷ｴ鮨｢","莠ｺ豌誉莉願ｵｰ","蜊伜享","ZI蜊ｰ","ZI","ZI鬆・,
    "譁､驥・,"貂娥","譖ｿ","鬨取焔","謇螻・,"隱ｿ謨吝ｸｫ","辷ｶ","豈咲宛","辷ｶ繧ｿ繧､繝・,"豈咲宛繧ｿ繧､繝・,
    "蜑崎ｵｰ譛・,"蜑崎ｵｰ譌･","蜑崎ｵｰ髢句ぎ","蜑崎ｵｰ髢馴囈","蜑崎ｵｰ繝ｬ繝ｼ繧ｹ蜷・,"蜑崎ｵｰTD","蜑崎ｵｰ霍晞屬","蜑崎ｵｰ鬥ｬ蝣ｴ迥ｶ諷・,
    "蜑崎ｵｰB","蜑崎ｵｰ鬨取焔","蜑崎ｵｰ譁､驥・,"蜑崎ｵｰ貂・,"蜑崎ｵｰ莠ｺ豌・,"蜑崎ｵｰ蜊伜享繧ｪ繝・ぜ","蜑崎ｵｰ逹鬆・,"蜑崎ｵｰ逹蟾ｮ",
    "繝槭う繝九Φ繧ｰ鬆・ｽ・,"蜑崎ｵｰ騾夐℃1","蜑崎ｵｰ騾夐℃2","蜑崎ｵｰ騾夐℃3","蜑崎ｵｰ騾夐℃4","蜑崎ｵｰAve3F",
    "蜑崎ｵｰ荳翫ｊ3F","蜑崎ｵｰ荳翫ｊ3F鬆・ｽ・,"蜑崎ｵｰ1_2逹鬥ｬ",
]


# =========================================================
# CSV 繝代・繧ｹ
# =========================================================
def parse_target_csv(uploaded_file) -> pd.DataFrame:
    """繧ｿ繝ｼ繧ｲ繝・ヨ蠖｢蠑修SV・・谿ｵ讒矩・峨ｒ繝槭せ繧ｿ繝ｼ逶ｸ蠖薙・DataFrame縺ｫ螟画鋤縺吶ｋ縲・""
    content = uploaded_file.read()
    for enc in ["cp932", "shift_jis", "utf-8"]:
        try:
            text = content.decode(enc)
            break
        except Exception:
            continue

    lines = text.splitlines()
    races: list[dict] = []
    current_race: dict | None = None

    # 蛻玲焚縺九ｉ閾ｪ蜍募愛螳夲ｼ・3蛻・譌ｧ蠖｢蠑・/ 46蛻・譁ｰ蠖｢蠑擾ｼ・    for line in lines:
        cols = line.split(",")
        if len(cols) == 19 and cols[0] not in ("繝ｬ繝ｼ繧ｹID(譁ｰ)", ""):
            current_race = dict(zip(RACE_COLS, cols))
        elif len(cols) == 33 and cols[0] not in ("譫逡ｪ", "") and current_race:
            horse = dict(zip(HORSE_COLS_33, cols))
            horse.update(current_race)
            races.append(horse)
        elif len(cols) == 46 and cols[0] not in ("譫逡ｪ", "") and current_race:
            horse = dict(zip(HORSE_COLS_46, cols))
            horse.update(current_race)
            races.append(horse)

    df = pd.DataFrame(races)
    if df.empty:
        return df

    # 蛻怜錐繝ｪ繝阪・繝
    df = df.rename(columns=COLUMN_MAP)

    # 繝ｬ繝ｼ繧ｹID豁｣隕丞喧・磯ｦｬ逡ｪ縺ｪ縺怜ｽ｢蠑上↓・・    df["繝ｬ繝ｼ繧ｹID(譁ｰ/鬥ｬ逡ｪ辟｡)"] = df["繝ｬ繝ｼ繧ｹID(譁ｰ)"].astype(str).str[:16]

    # 謨ｰ蛟､螟画鋤
    for col in ["譫逡ｪ","鬥ｬ逡ｪ","譁､驥・,"ZI","ZI鬆・ｽ・,"霍晞屬","莠ｺ豌・,"蜊伜享",
                "蜑崎ｵｰ遒ｺ螳夂捩鬆・,"蜑崎ｵｰ荳翫ｊ3F","蜑崎ｵｰ霍晞屬","髢馴囈","蜑崎ｵｰ莠ｺ豌・,
                "蜑崎ｵｰ逹蟾ｮ繧ｿ繧､繝","蜑崎ｵｰ譁､驥・,"蜑崎ｵｰ蟷ｳ蝮・F","蜑崎ｵｰ荳翫ｊ3F鬆・ｽ・,
                "繝槭う繝九Φ繧ｰ鬆・ｽ・,"蜑崎ｵｰ蜊伜享繧ｪ繝・ぜ",
                "蜑崎ｵｰ騾夐℃1","蜑崎ｵｰ騾夐℃2","蜑崎ｵｰ騾夐℃3","蜑崎ｵｰ騾夐℃4"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 譌･莉・    df["譌･莉・] = pd.to_datetime(df["譌･莉牢"], format="%Y.%m.%d", errors="coerce")
    df["譌･莉・] = df["譌･莉・].dt.strftime("%Y%m%d").astype("Int64")

    # 谺謳崎｣懷ｮ鯉ｼ医Δ繝・Ν縺悟ｿ・ｦ√↑蛻励ｒ0蝓九ａ・・    for col in ["蜑崎ｵｰ襍ｰ遐ｴ繧ｿ繧､繝","蜑崎ｵｰ逹蟾ｮ繧ｿ繧､繝","鬥ｬ菴馴㍾","鬥ｬ菴馴㍾蠅玲ｸ・,
                "蜑崎ｵｰ譁､驥・,"逕溽肇閠・,"鬥ｬ荳ｻ(譛譁ｰ/莉ｮ諠ｳ)"]:
        if col not in df.columns:
            df[col] = 0

    return df


# =========================================================
# 繝｢繝・Ν繝ｭ繝ｼ繝・# =========================================================
@st.cache_resource(show_spinner="繝｢繝・Ν隱ｭ縺ｿ霎ｼ縺ｿ荳ｭ...")
def load_models() -> tuple:
    return joblib.load(LGBM_PATH), joblib.load(CAT_PATH)


@st.cache_data(show_spinner="謌ｦ逡･繝・・繧ｿ隱ｭ縺ｿ霎ｼ縺ｿ荳ｭ...")
def load_strategy() -> dict:
    with open(STRATEGY_JSON, encoding="utf-8") as f:
        return json.load(f)


# =========================================================
# 莠域ｸｬ
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
    for col in ["蜑崎ｵｰ襍ｰ遐ｴ繧ｿ繧､繝", "蜑崎ｵｰ逹蟾ｮ繧ｿ繧､繝"]:
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
        "遞ｮ迚｡鬥ｬ","辷ｶ繧ｿ繧､繝怜錐","豈咲宛鬥ｬ","豈咲宛繧ｿ繧､繝怜錐","豈幄牡",
        "鬥ｬ荳ｻ(譛譁ｰ/莉ｮ諠ｳ)","逕溽肇閠・,"闃昴・繝","繧ｳ繝ｼ繧ｹ蛹ｺ蛻・,"闃・蜀・・螟・",
        "鬥ｬ蝣ｴ迥ｶ諷・,"螟ｩ豌・,"繧ｯ繝ｩ繧ｹ蜷・,"蝣ｴ謇","諤ｧ蛻･","譁､驥・,
        "繝悶Μ繝ｳ繧ｫ繝ｼ","驥埼㍼遞ｮ蛻･","蟷ｴ鮨｢髯仙ｮ・,"髯仙ｮ・,"諤ｧ蛻･髯仙ｮ・,"謖・ｮ壽擅莉ｶ",
        "蜑崎ｵｰ蝣ｴ謇","蜑崎茅繝ｻ繝","蜑崎ｵｰ鬥ｬ蝣ｴ迥ｶ諷・,"蜑崎ｵｰ譁､驥・,"蜑榊･ｽ襍ｰ",
    ]
    df = df.copy()
    for col in ["蜑崎ｵｰ襍ｰ遐ｴ繧ｿ繧､繝","蜑崎ｵｰ逹蟾ｮ繧ｿ繧､繝"]:
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
            df.at[idx, "mark"] = {1:"笳・,2:"笳ｯ",3:"笆ｲ",4:"笆ｳ",5:"ﾃ・}[int(rank)]
    return df


# =========================================================
# 蜈ｨ繝ｬ繝ｼ繧ｹ荳諡ｬ莠域Φ
# =========================================================
@st.cache_data(show_spinner="蜈ｨ繝ｬ繝ｼ繧ｹ莠域Φ險育ｮ嶺ｸｭ...")
def predict_all_races(df_json: str, _lgbm_obj: dict, _cat_obj: dict) -> str:
    """蜈ｨ繝ｬ繝ｼ繧ｹ荳諡ｬ縺ｧ繧ｹ繧ｳ繧｢繝ｻ蜊ｰ繧剃ｻ倅ｸ弱＠縺ｦJSON霑泌唆縲・""
    df = pd.read_json(df_json)
    result_frames = []
    race_id_col = "繝ｬ繝ｼ繧ｹID(譁ｰ/鬥ｬ逡ｪ辟｡)"

    for race_id, race_df in df.groupby(race_id_col):
        race_df = race_df.copy()
        try:
            race_df["prob"]  = ensemble_predict(race_df, _lgbm_obj, _cat_obj)
            race_df          = assign_marks(race_df)
            race_df["score"] = (race_df["prob"] * 100).round(1)
        except Exception as e:
            logger.warning(f"莠域ｸｬ螟ｱ謨・{race_id}: {e}")
            race_df["prob"]  = 0.0
            race_df["mark"]  = ""
            race_df["score"] = 0.0
        result_frames.append(race_df)

    return pd.concat(result_frames, ignore_index=True).to_json(force_ascii=False)


# =========================================================
# SHAP
# =========================================================
@st.cache_data(show_spinner="SHAP險育ｮ嶺ｸｭ...")
def compute_shap(_lgbm_obj: dict, df_json: str) -> tuple[list, list]:
    df = pd.read_json(df_json)
    model, encoders, feature_cols = _lgbm_obj["model"], _lgbm_obj["encoders"], _lgbm_obj["feature_cols"]
    df_enc = df.copy()
    for col in ["蜑崎ｵｰ襍ｰ遐ｴ繧ｿ繧､繝","蜑崎ｵｰ逹蟾ｮ繧ｿ繧､繝"]:
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
    ax.set_xlabel("SHAP蛟､・郁ｵ､=螂ｽ譚先侭 / 髱・荳榊ｮ画攝譁呻ｼ・, fontsize=8)
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
        level = "繝｢繝・Ν縺梧怙荳贋ｽ阪け繝ｩ繧ｹ縺ｮ隧穂ｾ｡繧剃ｸ弱∴縺ｦ縺翫ｊ"
    elif score >= 40:
        level = "荳贋ｽ榊恟縺ｮ隧穂ｾ｡繧剃ｸ弱∴縺ｦ縺翫ｊ"
    elif score >= 20:
        level = "荳ｭ菴榊恟縺ｮ隧穂ｾ｡繧剃ｸ弱∴縺ｦ縺翫ｊ"
    else:
        level = "荳倶ｽ榊恟縺ｮ隧穂ｾ｡繧剃ｸ弱∴縺ｦ縺翫ｊ"

    mark_txt = {
        "笳・: "譛ｬ蜻ｽ縺ｨ縺励※譛繧ゆｿ｡鬆ｼ縺ｧ縺阪ｋ荳鬆ｭ縲・,
        "笳ｯ": "蟇ｾ謚励→縺励※譛ｬ蜻ｽ繧定у縺九☆蟄伜惠縲・,
        "笆ｲ": "蜊倡ｩｴ縺ｨ縺励※荳逋ｺ縺ｮ鬲・鴨縺後≠繧九・,
        "笆ｳ": "騾｣荳句呵｣懊→縺励※謚代∴縺ｦ縺翫″縺溘＞縲・,
        "ﾃ・: "謚ｼ縺輔∴遞句ｺｦ縺縺悟恟螟悶→繧りｨ縺医↑縺・・,
    }.get(mark, "")

    pos_s = [f"{FEATURE_LABEL.get(c,c)}縺悟･ｽ譚先侭" for _, c in pos]
    neg_s = [f"{FEATURE_LABEL.get(c,c)}縺後ｄ繧・ｸ榊ｮ・ for _, c in neg]

    lines = []
    if mark_txt:
        lines.append(mark_txt)
    lines.append(f"{horse_name}縺ｯ繧ｹ繧ｳ繧｢{score:.1f}%縺ｧ縲＋level}荳贋ｽ榊・邱壹・蜿ｯ閭ｽ諤ｧ繧定ｩ穂ｾ｡縺励※縺・∪縺吶・)
    if pos_s:
        lines.append("螂ｽ譚先侭縺ｨ縺励※縲・ + "縲・.join(pos_s) + "縺梧嫌縺偵ｉ繧後∪縺吶・)
    if neg_s:
        lines.append("荳譁ｹ縺ｧ" + "縲・.join(neg_s) + "轤ｹ縺後≠繧翫∪縺吶・)
    else:
        lines.append("逶ｮ遶九▲縺滉ｸ榊ｮ画攝譁吶・蟆代↑縺丞ｮ牙ｮ壽─縺ｮ縺ゅｋ隧穂ｾ｡縺ｧ縺吶・)
    return " ".join(lines)


# =========================================================
# 雋ｷ縺・岼逕滓・
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
    hon    = marks_df["笳・]
    taikou = marks_df["笳ｯ"]
    sabo   = marks_df["笆ｲ"]
    delta  = marks_df["笆ｳ"]
    if hon.empty:
        return []
    h1 = int(hon.iloc[0]["鬥ｬ逡ｪ"])
    h2 = int(taikou.iloc[0]["鬥ｬ逡ｪ"]) if not taikou.empty else None
    h3 = int(sabo.iloc[0]["鬥ｬ逡ｪ"])   if not sabo.empty  else None
    h4 = int(delta.iloc[0]["鬥ｬ逡ｪ"])  if not delta.empty else None
    top3 = [h for h in [h1, h2, h3] if h is not None]
    top4 = [h for h in [h1, h2, h3, h4] if h is not None]
    results = []
    for bet_type, info in bet_info.items():
        amt = floor_to_unit(int(budget * info["bet_ratio"]))
        if bet_type == "隍・享":
            results.append({"鬥ｬ蛻ｸ遞ｮ":"隍・享","雋ｷ縺・岼":str(h1),"雉ｼ蜈･鬘・:amt,
                            "ROI":info["roi"],"繧ｦ繧ｧ繧､繝・:round(info["weight"]*100,1)})
        elif bet_type == "鬥ｬ騾｣":
            if h2 is None: continue
            results.append({"鬥ｬ蛻ｸ遞ｮ":"鬥ｬ騾｣","雋ｷ縺・岼":"-".join(map(str,sorted([h1,h2]))),
                            "雉ｼ蜈･鬘・:amt,"ROI":info["roi"],"繧ｦ繧ｧ繧､繝・:round(info["weight"]*100,1)})
        elif bet_type == "荳蛾｣隍・:
            if len(top3) < 3: continue
            combos  = list(itertools.combinations(top4[:4],3))[:3]
            per_bet = floor_to_unit(amt // max(len(combos),1))
            for c in combos:
                results.append({"鬥ｬ蛻ｸ遞ｮ":"荳蛾｣隍・,"雋ｷ縺・岼":"-".join(map(str,sorted(c))),
                                "雉ｼ蜈･鬘・:per_bet,"ROI":info["roi"],"繧ｦ繧ｧ繧､繝・:round(info["weight"]*100,1)})
        elif bet_type == "荳蛾｣蜊・:
            if len(top3) < 3: continue
            perms   = list(itertools.permutations(top3[:3],3))[:3]
            per_bet = floor_to_unit(amt // max(len(perms),1))
            for p in perms:
                results.append({"鬥ｬ蛻ｸ遞ｮ":"荳蛾｣蜊・,"雋ｷ縺・岼":"-".join(map(str,p)),
                                "雉ｼ蜈･鬘・:per_bet,"ROI":info["roi"],"繧ｦ繧ｧ繧､繝・:round(info["weight"]*100,1)})
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
/* 繝ｬ繝ｼ繧ｹ荳隕ｧ繝・・繝悶Ν */
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
# 繝ｬ繝ｼ繧ｹ荳隕ｧ繝壹・繧ｸ
# =========================================================
def page_race_list(
    all_df: pd.DataFrame,
    strategy: dict,
    budget: int,
) -> None:
    st.markdown("### 套 繝ｬ繝ｼ繧ｹ荳隕ｧ")

    race_id_col = "繝ｬ繝ｼ繧ｹID(譁ｰ/鬥ｬ逡ｪ辟｡)"
    race_list   = []
    for race_id, grp in all_df.groupby(race_id_col):
        meta     = grp.iloc[0]
        place    = str(meta.get("蝣ｴ謇",""))
        cls_raw  = str(meta.get("繧ｯ繝ｩ繧ｹ蜷・,""))
        cls_norm = CLASS_NORMALIZE.get(cls_raw, cls_raw)
        in_strat = (
            place in strategy and
            (cls_norm in strategy[place] or cls_raw in strategy[place])
        )
        hon_row = grp[grp["mark"] == "笳・]
        hon_name = str(hon_row.iloc[0]["鬥ｬ蜷・]) if not hon_row.empty else "-"
        race_list.append({
            "race_id":  race_id,
            "蝣ｴ謇":     place,
            "R":        str(meta.get("R","")),
            "繧ｯ繝ｩ繧ｹ":   cls_raw,
            "霍晞屬":     f'{meta.get("闃昴・繝","")}{meta.get("霍晞屬","")}m',
            "逋ｺ襍ｰ":     str(meta.get("逋ｺ襍ｰ譎ょ綾","")),
            "笳・:        hon_name,
            "謌ｦ逡･":     in_strat,
            "鬆ｭ謨ｰ":     str(len(grp)),
        })

    # 繝倥ャ繝繝ｼ
    st.markdown(
        '<div class="race-list-header">'
        '<span>蝣ｴ謇</span><span>R</span><span>繧ｯ繝ｩ繧ｹ</span><span>笳取悽蜻ｽ</span>'
        '<span>鬆ｭ謨ｰ</span><span>霍晞屬</span><span>逋ｺ襍ｰ</span><span>謌ｦ逡･蟇ｾ雎｡</span><span></span>'
        '</div>',
        unsafe_allow_html=True,
    )

    for r in race_list:
        badge = '<span class="strategy-badge">笨・蟇ｾ雎｡</span>' if r["謌ｦ逡･"] else '<span style="color:#555">-</span>'
        st.markdown(
            f'<div class="race-list-row">'
            f'<span>{r["蝣ｴ謇"]}</span>'
            f'<span>{r["R"]}</span>'
            f'<span>{r["繧ｯ繝ｩ繧ｹ"]}</span>'
            f'<span>笳・{r["笳・]}</span>'
            f'<span>{r["鬆ｭ謨ｰ"]}鬆ｭ</span>'
            f'<span>{r["霍晞屬"]}</span>'
            f'<span>{r["逋ｺ襍ｰ"]}</span>'
            f'<span>{badge}</span>'
            f'<span></span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if st.button(f'隧ｳ邏ｰ 竊・, key=f'btn_{r["race_id"]}'):
            st.session_state.selected_race_id = r["race_id"]
            st.rerun()


# =========================================================
# 蜃ｺ襍ｰ陦ｨ繝壹・繧ｸ
# =========================================================
def page_race_detail(
    race_df: pd.DataFrame,
    strategy: dict,
    budget: int,
    lgbm_obj: dict,
) -> None:
    meta    = race_df.iloc[0]
    place   = str(meta.get("蝣ｴ謇",""))
    cls_raw = str(meta.get("繧ｯ繝ｩ繧ｹ蜷・,""))
    dist    = meta.get("霍晞屬","")
    shida   = meta.get("闃昴・繝","")

    cls_norm    = CLASS_NORMALIZE.get(cls_raw, cls_raw)
    in_strategy = (
        place in strategy and
        (cls_norm in strategy[place] or cls_raw in strategy[place])
    )

    # 謌ｻ繧九・繧ｿ繝ｳ
    if st.button("竊・繝ｬ繝ｼ繧ｹ荳隕ｧ縺ｫ謌ｻ繧・):
        st.session_state.selected_race_id = None
        st.rerun()

    st.markdown(f"## {place} {meta.get('R','')}R / {cls_raw} / {shida}{dist}m")

    if in_strategy:
        cls_key  = cls_norm if cls_norm in strategy.get(place,{}) else cls_raw
        roi_vals = [v["roi"] for v in strategy[place][cls_key].values()]
        st.success(f"笨・謌ｦ逡･蟇ｾ雎｡繝ｬ繝ｼ繧ｹ縲蟷ｳ蝮⑲OI: {sum(roi_vals)/len(roi_vals):.1f}%")
    else:
        st.info("邃ｹ・・謌ｦ逡･蟇ｾ雎｡螟厄ｼ亥盾閠・ｺ域Φ・・)

    # SHAP險育ｮ・    shap_ok = False
    shap_vals: list = []
    feature_cols: list = []
    with st.spinner("SHAP險育ｮ嶺ｸｭ..."):
        try:
            shap_vals, feature_cols = compute_shap(lgbm_obj, race_df.to_json())
            shap_ok = True
        except Exception as e:
            st.warning(f"SHAP險育ｮ怜､ｱ謨・ {e}")

    tab1, tab2 = st.tabs(["搭 蜃ｺ襍ｰ陦ｨ / 雋ｷ縺・岼", "剥 蜈ｨ鬆ｭ蛻・梵"])

    # ---- Tab1: 蜃ｺ襍ｰ陦ｨ + 雋ｷ縺・岼 ----
    with tab1:
        st.markdown("### 蜃ｺ襍ｰ陦ｨ")
        st.markdown(
            '<div class="tbl-header">'
            '<span>譫</span><span>鬥ｬ逡ｪ</span><span>蜊ｰ</span>'
            '<span>鬥ｬ蜷・/span><span>諤ｧ鮨｢</span><span>譁､驥・/span>'
            '<span>鬨取焔</span><span>繧ｹ繧ｳ繧｢</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        race_sorted = race_df.sort_values("鬥ｬ逡ｪ").reset_index(drop=True)
        for _, row in race_sorted.iterrows():
            mark   = row.get("mark","")
            ban    = int(row.get("鬥ｬ逡ｪ",0))
            waku   = int(row.get("譫逡ｪ",0))
            name   = str(row.get("鬥ｬ蜷・,f"{ban}逡ｪ"))
            seire  = str(row.get("諤ｧ蛻･","")) + str(row.get("蟷ｴ鮨｢",""))
            kin    = str(row.get("譁､驥・,""))
            jockey = str(row.get("鬨取焔",""))
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
            st.markdown("### 識 雋ｷ縺・岼")
            bets = get_bets(race_df, place, cls_raw, strategy, budget)
            if not bets:
                st.warning("雋ｷ縺・岼繧堤函謌舌〒縺阪∪縺帙ｓ縺ｧ縺励◆縲・)
            else:
                bets_df = pd.DataFrame(bets)
                total   = bets_df["雉ｼ蜈･鬘・].sum()
                m1, m2, m3 = st.columns(3)
                m1.metric("蜷郁ｨ郁ｳｼ蜈･鬘・, f"{total:,}蜀・)
                m2.metric("鬥ｬ蛻ｸ遞ｮ謨ｰ",   f"{bets_df['鬥ｬ蛻ｸ遞ｮ'].nunique()}遞ｮ")
                m3.metric("邱冗せ謨ｰ",     f"{len(bets_df)}轤ｹ")

                summary = (
                    bets_df.groupby("鬥ｬ蛻ｸ遞ｮ")
                    .agg(ROI=("ROI","first"), 繧ｦ繧ｧ繧､繝・("繧ｦ繧ｧ繧､繝・,"first"),
                         轤ｹ謨ｰ=("雋ｷ縺・岼","count"), 蜷郁ｨ・("雉ｼ蜈･鬘・,"sum"))
                    .reset_index().sort_values("ROI", ascending=False)
                )
                summary["ROI"]    = summary["ROI"].apply(lambda x: f"{x:.1f}%")
                summary["繧ｦ繧ｧ繧､繝・] = summary["繧ｦ繧ｧ繧､繝・].apply(lambda x: f"{x:.1f}%")
                summary["蜷郁ｨ・]   = summary["蜷郁ｨ・].apply(lambda x: f"{x:,}蜀・)
                st.dataframe(summary, use_container_width=True, hide_index=True)

                disp = bets_df[["鬥ｬ蛻ｸ遞ｮ","雋ｷ縺・岼","雉ｼ蜈･鬘・,"ROI"]].copy()
                disp["雉ｼ蜈･鬘・] = disp["雉ｼ蜈･鬘・].apply(lambda x: f"{x:,}蜀・)
                disp["ROI"]   = disp["ROI"].apply(lambda x: f"{x:.1f}%")
                st.dataframe(disp, use_container_width=True, hide_index=True)

    # ---- Tab2: 蜈ｨ鬆ｭ蛻・梵 ----
    with tab2:
        if not shap_ok:
            st.error("SHAP險育ｮ励↓螟ｱ謨励＠縺ｾ縺励◆縲・)
            return
        st.markdown("### 剥 蜈ｨ鬆ｭ蛻・梵")
        race_sorted = race_df.sort_values("鬥ｬ逡ｪ").reset_index(drop=True)
        for i, row in race_sorted.iterrows():
            sv_row  = shap_vals[i]
            name    = str(row.get("鬥ｬ蜷・, f"{int(row['鬥ｬ逡ｪ'])}逡ｪ"))
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
                    st.markdown(f'<div style="color:#e74c3c;font-size:12px;margin:2px 0">衍 {label} +{v:.3f}</div>',
                                unsafe_allow_html=True)
                for v, c in neg3:
                    label = FEATURE_LABEL.get(c,c)
                    st.markdown(f'<div style="color:#5865f2;font-size:12px;margin:2px 0">洶 {label} {v:.3f}</div>',
                                unsafe_allow_html=True)
            st.markdown("<hr style='border-color:#313244;margin:8px 0'>", unsafe_allow_html=True)


# =========================================================
# main
# =========================================================
def main() -> None:
    st.set_page_config(page_title="PyCaLiAI", page_icon="順", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("順 PyCaLiAI 遶ｶ鬥ｬ莠域Φ繧ｷ繧ｹ繝・Β")

    lgbm_obj, cat_obj = load_models()
    strategy          = load_strategy()

    # 繧ｵ繧､繝峨ヰ繝ｼ
    with st.sidebar:
        st.header("唐 蜃ｺ襍ｰ陦ｨCSV")
        uploaded = st.file_uploader("繧ｿ繝ｼ繧ｲ繝・ヨ蠖｢蠑修SV繧偵い繝・・繝ｭ繝ｼ繝・, type=["csv"])
        st.divider()
        budget = st.number_input("1繝ｬ繝ｼ繧ｹ莠育ｮ暦ｼ亥・・・, min_value=1_000,
                                 max_value=1_000_000, value=10_000, step=1_000)

    if uploaded is None:
        st.info("繧ｵ繧､繝峨ヰ繝ｼ縺九ｉ騾ｱ譛ｫ縺ｮ蜃ｺ襍ｰ陦ｨCSV繧偵い繝・・繝ｭ繝ｼ繝峨＠縺ｦ縺上□縺輔＞縲・)
        return

    # CSV 繝代・繧ｹ
    with st.spinner("CSV隱ｭ縺ｿ霎ｼ縺ｿ荳ｭ..."):
        raw_df = parse_target_csv(uploaded)
        if raw_df.empty:
            st.error("CSV縺ｮ隱ｭ縺ｿ霎ｼ縺ｿ縺ｫ螟ｱ謨励＠縺ｾ縺励◆縲・)
            return

    # 蜈ｨ繝ｬ繝ｼ繧ｹ荳諡ｬ莠域Φ・医く繝｣繝・す繝･貂医∩縺ｪ繧牙・險育ｮ励＠縺ｪ縺・ｼ・    predicted_json = predict_all_races(raw_df.to_json(), lgbm_obj, cat_obj)
    all_df         = pd.read_json(predicted_json)

    # session_state 縺ｧ繝ｬ繝ｼ繧ｹ驕ｸ謚樒ｮ｡逅・    if "selected_race_id" not in st.session_state:
        st.session_state.selected_race_id = None

    if st.session_state.selected_race_id is None:
        # 繝ｬ繝ｼ繧ｹ荳隕ｧ陦ｨ遉ｺ
        race_id_col = "繝ｬ繝ｼ繧ｹID(譁ｰ/鬥ｬ逡ｪ辟｡)"
        n_races     = all_df[race_id_col].nunique()
        n_strategy  = sum(
            1 for _, grp in all_df.groupby(race_id_col)
            if (lambda p, c: p in strategy and (
                CLASS_NORMALIZE.get(c,c) in strategy[p] or c in strategy[p]
            ))(str(grp.iloc[0].get("蝣ｴ謇","")), str(grp.iloc[0].get("繧ｯ繝ｩ繧ｹ蜷・,"")))
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("繝ｬ繝ｼ繧ｹ謨ｰ",     f"{n_races}R")
        c2.metric("鬆ｭ謨ｰ",         f"{len(all_df)}鬆ｭ")
        c3.metric("謌ｦ逡･蟇ｾ雎｡繝ｬ繝ｼ繧ｹ", f"{n_strategy}R")
        st.markdown("---")
        page_race_list(all_df, strategy, budget)
    else:
        # 繝ｬ繝ｼ繧ｹ隧ｳ邏ｰ陦ｨ遉ｺ
        race_id_col = "繝ｬ繝ｼ繧ｹID(譁ｰ/鬥ｬ逡ｪ辟｡)"
        race_df     = all_df[all_df[race_id_col] == st.session_state.selected_race_id].copy()
        race_df     = race_df.reset_index(drop=True)
        page_race_detail(race_df, strategy, budget, lgbm_obj)


if __name__ == "__main__":
    main()

