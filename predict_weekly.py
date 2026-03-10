"""
predict_weekly.py
週末出走表CSV → モデル予想結果CSV出力

Usage:
    python predict_weekly.py --csv data/weekly/20260301.csv
    python predict_weekly.py --csv data/weekly/20260301.csv --budget 10000
    python predict_weekly.py --csv data/weekly/20260301.csv --out reports/pred_20260301.csv

Output columns:
    日付, 場所, R, クラス, 距離, レースID, 馬番, 馬名, 騎手,
    スコア, 印, 複勝_購入額, 馬連_買い目, 馬連_購入額,
    三連複_買い目, 三連複_購入額, 三連単_買い目, 三連単_購入額,
    戦略対象
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR      = Path(__file__).parent
MODEL_DIR     = BASE_DIR / "models"
STRATEGY_JSON = BASE_DIR / "data" / "strategy_weights.json"
LGBM_PATH      = MODEL_DIR / "lgbm_optuna_v1.pkl"
CAT_PATH       = MODEL_DIR / "catboost_optuna_v1.pkl"
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

MIN_UNIT        = 100
EXCLUDE_PLACES  = {"東京", "小倉"}
EXCLUDE_CLASSES = {"新馬", "障害"}

# =========================================================
# 列定義
# =========================================================
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
    "前走Ave3F":     "前走平均3F",
    "前走上り3F順位":  "前走上り3F順位",
    "マイニング順位":  "マイニング順位",
    "前走単勝オッズ":  "前走単勝オッズ",
    "前走通過1":      "前走通過1",
    "前走通過2":      "前走通過2",
    "前走通過3":      "前走通過3",
    "前走通過4":      "前走通過4",
}
CLASS_NORMALIZE = {
    "新馬":"新馬","未勝利":"未勝利","1勝":"1勝","500万":"1勝",
    "2勝":"2勝","1000万":"2勝","3勝":"3勝","1600万":"3勝",
    "OP(L)":"OP(L)","Ｇ１":"Ｇ１","Ｇ２":"Ｇ２","Ｇ３":"Ｇ３",
}


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
        elif len(cols) == 46 and current_race:
            horse = dict(zip(HORSE_COLS_46, cols))
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

    df = pd.DataFrame(races).rename(columns=COLUMN_MAP)
    df["レースID(新/馬番無)"] = df["レースID(新)"].astype(str).str[:16]

    for col in ["枠番","馬番","斤量","ZI","ZI順位","距離","人気","単勝",
                "前走確定着順","前走上り3F","前走距離","間隔","前走人気",
                "前走着差タイム","前走斤量","前走平均3F","前走上り3F順位",
                "マイニング順位","前走単勝オッズ",
                "前走通過1","前走通過2","前走通過3","前走通過4",
                "フルゲート頭数","年齢","出走頭数","コーナー回数"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["前走走破タイム","前走着差タイム","馬体重","馬体重増減","前走斤量","生産者"]:
        if col not in df.columns:
            df[col] = 0

    df = df[~df["距離"].astype(str).str.contains("障", na=False)].copy()
    logger.info(f"パース完了（障害除外済）: {len(df)}頭 / {df['レースID(新/馬番無)'].nunique()}レース")
    return df


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

        df_copy = df.copy()
        if "fukusho_flag" not in df_copy.columns:
            df_copy["fukusho_flag"] = 0
        df2, _, _ = torch_preprocess(df_copy, encoders=encoders, fit=False, num_stats=num_stats)

        # モデルをキャッシュ（起動後1回だけビルド）
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
        logger.warning(f"スタッキング予測失敗（フォールバック）: {e}")
        return None


def ensemble_predict(df: pd.DataFrame, lgbm_obj: dict, cat_obj: dict) -> np.ndarray:
    # スタッキング優先
    stacking = predict_stacking(df, lgbm_obj, cat_obj)
    if stacking is not None:
        cal_obj = _get_cached(STACK_CAL_PATH, "stack_cal")
        if cal_obj is not None:
            return cal_obj["calibrator"].transform(stacking)
        return stacking
    # フォールバック: 2モデル平均 + キャリブレーション
    raw = 0.5 * predict_lgbm(df, lgbm_obj) + 0.5 * predict_catboost(df, cat_obj)
    cal_obj = _get_cached(CAL_PATH, "ens_cal")
    if cal_obj is not None:
        return cal_obj["calibrator"].transform(raw)
    logger.warning("キャリブレーター未生成。calibrate.py を先に実行してください。")
    return raw


def assign_marks(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["mark"] = ""
    ranked = df["prob"].rank(ascending=False, method="first")
    for idx, r in ranked.items():
        if r <= 5:
            df.at[idx, "mark"] = {1:"◎",2:"◯",3:"▲",4:"△",5:"×"}[int(r)]
    return df


# =========================================================
# 買い目生成
# =========================================================
def floor_to_unit(x: int, unit: int = MIN_UNIT) -> int:
    return max((x // unit) * unit, unit)


def get_bets(race_df: pd.DataFrame, place: str, cls_raw: str,
             strategy: dict, budget: int) -> dict:
    """HAHO（馬連◎軸2点+三連複ボックス1点）/ HALO（三連複ボックス1点のみ）
       / LALO（複勝◎1点のみ）/ CQC（単勝◎1点のみ）を生成する。"""
    cls      = CLASS_NORMALIZE.get(cls_raw, cls_raw)
    bet_info = strategy.get(place, {}).get(cls) or strategy.get(place, {}).get(cls_raw, {})

    result: dict = {
        "HAHO_戦略対象":      False,   # 実際にベットが生成されたときのみ True
        "HAHO_馬連_買い目":   "", "HAHO_馬連_購入額":   0,
        "HAHO_三連複_買い目": "", "HAHO_三連複_購入額": 0,
        "HALO_戦略対象":      False,
        "HALO_三連複_買い目": "", "HALO_三連複_購入額": 0,
        "LALO_戦略対象":      False,
        "LALO_複勝_買い目":   "", "LALO_複勝_購入額":   0,
        "CQC_戦略対象":       False,
        "CQC_単勝_買い目":    "", "CQC_単勝_購入額":    0,
    }
    if not bet_info:
        return result

    marks_df = {m: race_df[race_df["mark"] == m] for m in ["◎","◯","▲"]}
    hon    = marks_df["◎"]
    taikou = marks_df["◯"]
    sabo   = marks_df["▲"]
    if hon.empty:
        return result

    h1 = int(hon.iloc[0]["馬番"])
    h2 = int(taikou.iloc[0]["馬番"]) if not taikou.empty else None
    h3 = int(sabo.iloc[0]["馬番"])   if not sabo.empty  else None

    # ── HAHO: 馬連◎軸2点 + 三連複ボックス1点 ──────────────────────────
    haho_types = {k: v for k, v in bet_info.items() if k in ("馬連", "三連複")}
    if haho_types and h2 and h3:
        total_ratio = sum(v["bet_ratio"] for v in haho_types.values()) or 1.0
        # 馬連: ◎-◯, ◎-▲ の2点
        if "馬連" in haho_types:
            r   = haho_types["馬連"]["bet_ratio"]
            amt = floor_to_unit(int(budget * r / total_ratio))
            cbs = [(h1, h2), (h1, h3)]
            per = floor_to_unit(amt // len(cbs))
            result["HAHO_馬連_買い目"] = " / ".join(f"{min(a,b)}-{max(a,b)}" for a, b in cbs)
            result["HAHO_馬連_購入額"] = per * len(cbs)
        # 三連複: ◎◯▲ボックス1点
        if "三連複" in haho_types:
            r    = haho_types["三連複"]["bet_ratio"]
            amt  = floor_to_unit(int(budget * r / total_ratio))
            c_sf = tuple(sorted([h1, h2, h3]))
            result["HAHO_三連複_買い目"] = "-".join(map(str, c_sf))
            result["HAHO_三連複_購入額"] = amt
        # ベットが1件でも生成されたら対象フラグを立てる
        if result["HAHO_馬連_買い目"] or result["HAHO_三連複_買い目"]:
            result["HAHO_戦略対象"] = True

    # ── HALO: 三連複ボックス1点のみ（全予算）──────────────────────────
    if "三連複" in bet_info and h2 and h3:
        result["HALO_戦略対象"] = True
        c_sf = tuple(sorted([h1, h2, h3]))
        result["HALO_三連複_買い目"] = "-".join(map(str, c_sf))
        result["HALO_三連複_購入額"] = floor_to_unit(budget)

    # ── LALO: 複勝◎1点のみ（全予算）────────────────────────────────────
    if bet_info:   # 戦略対象レースなら常に発動
        result["LALO_戦略対象"]    = True
        result["LALO_複勝_買い目"] = str(h1)
        result["LALO_複勝_購入額"] = floor_to_unit(budget)

    # ── CQC: 単勝◎1点のみ（全予算）─────────────────────────────────────
    if bet_info:   # 戦略対象レースなら常に発動
        result["CQC_戦略対象"]   = True
        result["CQC_単勝_買い目"] = str(h1)
        result["CQC_単勝_購入額"] = floor_to_unit(budget)

    return result


# =========================================================
# メイン
# =========================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="週末CSV予想結果出力")
    parser.add_argument("--csv",    required=True, help="入力CSVパス (例: data/weekly/20260301.csv)")
    parser.add_argument("--budget", type=int, default=10000, help="1レース予算（円）")
    parser.add_argument("--out",    default="", help="出力CSVパス（省略時は自動命名）")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        logger.error(f"CSVが見つかりません: {csv_path}")
        sys.exit(1)

    # モデル・戦略読み込み
    logger.info("モデル読み込み中...")
    lgbm_obj = joblib.load(LGBM_PATH)
    cat_obj  = joblib.load(CAT_PATH)
    with open(STRATEGY_JSON, encoding="utf-8") as f:
        strategy = json.load(f)

    # CSV パース
    df = parse_csv(csv_path)

    # 全レース予測
    logger.info("予測開始...")
    rows = []
    race_id_col = "レースID(新/馬番無)"

    for race_id, race_df in df.groupby(race_id_col):
        race_df = race_df.copy().reset_index(drop=True)
        meta    = race_df.iloc[0]
        place   = str(meta.get("場所",""))
        cls_raw = str(meta.get("クラス名",""))
        shida   = str(meta.get("芝・ダ",""))
        dist    = str(meta.get("距離",""))
        date    = str(meta.get("日付S",""))
        r_num   = str(meta.get("R",""))

        try:
            race_df["prob"]  = ensemble_predict(race_df, lgbm_obj, cat_obj)
            race_df          = assign_marks(race_df)
            race_df["score"] = (race_df["prob"] * 100).round(1)
        except Exception as e:
            logger.warning(f"予測失敗 {race_id}: {e}")
            race_df["prob"]  = 0.0
            race_df["mark"]  = ""
            race_df["score"] = 0.0

        # 除外フィルタ: 東京・小倉・新馬・障害
        if place in EXCLUDE_PLACES or cls_raw in EXCLUDE_CLASSES:
            bets = {
                "HAHO_戦略対象": False,
                "HAHO_馬連_買い目": "", "HAHO_馬連_購入額": 0,
                "HAHO_三連複_買い目": "", "HAHO_三連複_購入額": 0,
                "HALO_戦略対象": False,
                "HALO_三連複_買い目": "", "HALO_三連複_購入額": 0,
                "LALO_戦略対象": False,
                "LALO_複勝_買い目": "", "LALO_複勝_購入額": 0,
                "CQC_戦略対象":  False,
                "CQC_単勝_買い目": "", "CQC_単勝_購入額":  0,
            }
        else:
            # 買い目生成
            bets = get_bets(race_df, place, cls_raw, strategy, args.budget)

        for _, row in race_df.sort_values("馬番").iterrows():
            is_hon = row["mark"] == "◎"
            rows.append({
                "日付":                date,
                "場所":                place,
                "R":                   r_num,
                "クラス":              cls_raw,
                "距離":                f"{shida}{dist}m",
                "レースID":            race_id,
                "馬番":                int(row["馬番"]) if pd.notna(row["馬番"]) else "",
                "馬名":                str(row.get("馬名","")),
                "騎手":                str(row.get("騎手","")),
                "スコア":              float(row["score"]),
                "印":                  str(row["mark"]),
                "HAHO_戦略対象":       "✅" if bets["HAHO_戦略対象"] else "",
                "HAHO_馬連_買い目":    bets["HAHO_馬連_買い目"]    if is_hon else "",
                "HAHO_馬連_購入額":    bets["HAHO_馬連_購入額"]    if is_hon else "",
                "HAHO_三連複_買い目":  bets["HAHO_三連複_買い目"]  if is_hon else "",
                "HAHO_三連複_購入額":  bets["HAHO_三連複_購入額"]  if is_hon else "",
                "HALO_戦略対象":       "✅" if bets["HALO_戦略対象"] else "",
                "HALO_三連複_買い目":  bets["HALO_三連複_買い目"]  if is_hon else "",
                "HALO_三連複_購入額":  bets["HALO_三連複_購入額"]  if is_hon else "",
                "LALO_戦略対象":       "✅" if bets["LALO_戦略対象"] else "",
                "LALO_複勝_買い目":    bets["LALO_複勝_買い目"]    if is_hon else "",
                "LALO_複勝_購入額":    bets["LALO_複勝_購入額"]    if is_hon else "",
                "CQC_戦略対象":        "✅" if bets["CQC_戦略対象"]  else "",
                "CQC_単勝_買い目":     bets["CQC_単勝_買い目"]     if is_hon else "",
                "CQC_単勝_購入額":     bets["CQC_単勝_購入額"]     if is_hon else "",
            })

    out_df = pd.DataFrame(rows)

    # 出力
    out_path = Path(args.out) if args.out else BASE_DIR / "reports" / f"pred_{csv_path.stem}.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(f"出力完了: {out_path}")

    # サマリ表示
    haho_races = out_df[out_df["HAHO_戦略対象"]=="✅"]["レースID"].nunique()
    halo_races = out_df[out_df["HALO_戦略対象"]=="✅"]["レースID"].nunique()
    lalo_races = out_df[out_df["LALO_戦略対象"]=="✅"]["レースID"].nunique()
    cqc_races  = out_df[out_df["CQC_戦略対象"] =="✅"]["レースID"].nunique()
    print(f"\n{'='*50}")
    print(f"予想完了: {out_df['レースID'].nunique()}レース / {len(out_df)}頭")
    print(f"HAHO対象: {haho_races}R  HALO対象: {halo_races}R  LALO対象: {lalo_races}R  CQC対象: {cqc_races}R")
    print(f"出力先:   {out_path}")
    print(f"{'='*50}")

    # 戦略対象レースの買い目サマリ
    hon_rows = out_df[out_df["印"]=="◎"].copy()
    if not hon_rows.empty:
        haho_disp = hon_rows[hon_rows["HAHO_戦略対象"]=="✅"][[
            "日付","場所","R","クラス","距離","馬名",
            "HAHO_馬連_買い目","HAHO_馬連_購入額",
            "HAHO_三連複_買い目","HAHO_三連複_購入額",
        ]]
        if not haho_disp.empty:
            print("\n【HAHO 買い目一覧（馬連◎軸2点 + 三連複ボックス1点）】")
            print(haho_disp.to_string(index=False))
        halo_disp = hon_rows[hon_rows["HALO_戦略対象"]=="✅"][[
            "日付","場所","R","クラス","距離","馬名",
            "HALO_三連複_買い目","HALO_三連複_購入額",
        ]]
        if not halo_disp.empty:
            print("\n【HALO 買い目一覧（三連複ボックス1点のみ）】")
            print(halo_disp.to_string(index=False))
        lalo_disp = hon_rows[hon_rows["LALO_戦略対象"]=="✅"][[
            "日付","場所","R","クラス","距離","馬名",
            "LALO_複勝_買い目","LALO_複勝_購入額",
        ]]
        if not lalo_disp.empty:
            print("\n【LALO 買い目一覧（複勝◎1点のみ）】")
            print(lalo_disp.to_string(index=False))
        cqc_disp = hon_rows[hon_rows["CQC_戦略対象"]=="✅"][[
            "日付","場所","R","クラス","距離","馬名",
            "CQC_単勝_買い目","CQC_単勝_購入額",
        ]]
        if not cqc_disp.empty:
            print("\n【CQC 買い目一覧（単勝◎1点のみ）】")
            print(cqc_disp.to_string(index=False))


if __name__ == "__main__":
    main()