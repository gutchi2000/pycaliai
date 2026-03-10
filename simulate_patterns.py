"""
simulate_patterns.py  ─  PyCaLiAI 買い目構造 一括比較シミュレーション

馬連・三連複・複勝の買い方パターンを複数定義し、
同じモデル予測・戦略フィルタで ROI / 的中率 / 収支を比較する。

テストするパターン:
  複勝  : なし / ◎1点 / ◎◯2点
  馬連  : なし / ◎-◯1点 / ◎軸2点(◎-◯,◎-▲) / ボックス3点(◎-◯,◎-▲,◯-▲)
  三連複: なし / ◎◯▲ボックス1点 / ◎◯2頭軸×▲△2点 / ◎1頭軸×◯▲△3点

Usage:
    venv311\\Scripts\\python simulate_patterns.py
    venv311\\Scripts\\python simulate_patterns.py --period valid
    venv311\\Scripts\\python simulate_patterns.py --period 2024
    venv311\\Scripts\\python simulate_patterns.py --budget 5000
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import Pool
from tqdm import tqdm

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# =========================================================
# パス・定数
# =========================================================
BASE_DIR      = Path(r"E:\PyCaLiAI")
DATA_DIR      = BASE_DIR / "data"
MODEL_DIR     = BASE_DIR / "models"
REPORT_DIR    = BASE_DIR / "reports"

MASTER_CSV    = DATA_DIR  / "master_20130105-20251228.csv"
KEKKA_CSV     = DATA_DIR  / "kekka_20130105-20251228.csv"
STRATEGY_JSON = DATA_DIR  / "strategy_weights.json"
LGBM_PATH     = MODEL_DIR / "lgbm_optuna_v1.pkl"
CAT_PATH      = MODEL_DIR / "catboost_optuna_v1.pkl"
STACK_PATH    = MODEL_DIR / "stacking_meta_v1.pkl"
STACK_CAL     = MODEL_DIR / "stacking_calibrator_v1.pkl"
ENS_CAL       = MODEL_DIR / "ensemble_calibrator_v1.pkl"

TARGET      = "fukusho_flag"
COL_RACE_ID = "レースID(新/馬番無)"
BUDGET      = 10_000
MIN_UNIT    = 100

EXCLUDE_PLACES  = {"東京", "小倉"}
EXCLUDE_CLASSES = {"新馬", "障害"}

CLASS_NORMALIZE = {
    "新馬":"新馬","未勝利":"未勝利","1勝":"1勝","500万":"1勝",
    "2勝":"2勝","1000万":"2勝","3勝":"3勝","1600万":"3勝",
    "OP(L)":"OP(L)","Ｇ１":"Ｇ１","Ｇ２":"Ｇ２","Ｇ３":"Ｇ３",
    "ｵｰﾌﾟﾝ":"ｵｰﾌﾟﾝ","オープン":"ｵｰﾌﾟﾝ",
}

_META_EXTRA = ["芝・ダ", "距離", "クラス名", "場所", "馬場状態", "出走頭数", "枠番", "馬番"]

# =========================================================
# 比較するパターン定義
# =========================================================
# 各パターンは {"複勝": str|None, "馬連": str|None, "三連複": str|None} で定義
# 値は後述の generate_bets() で解釈される

PATTERNS: dict[str, dict] = {
    "P01_predict現状":   {"複勝": "◎1",    "馬連": "ボックス3",  "三連複": "◎◯軸2"},
    "P02_backtest現状":  {"複勝": "◎◯2",   "馬連": "◎軸2",       "三連複": "期待値3"},
    "P03_複勝なし_馬連ボックス_三連複◎◯軸2":   {                    "馬連": "ボックス3",  "三連複": "◎◯軸2"},
    "P04_複勝なし_馬連◎軸_三連複◎◯軸2":        {                    "馬連": "◎軸2",       "三連複": "◎◯軸2"},
    "P05_複勝なし_馬連ボックス_三連複◎1軸3":    {                    "馬連": "ボックス3",  "三連複": "◎1軸3"},
    "P06_複勝なし_馬連◎軸_三連複◎1軸3":         {                    "馬連": "◎軸2",       "三連複": "◎1軸3"},
    "P07_複勝なし_馬連ボックス_三連複ボックス1": {                    "馬連": "ボックス3",  "三連複": "ボックス1"},
    "P08_複勝なし_馬連◎軸_三連複ボックス1":      {                    "馬連": "◎軸2",       "三連複": "ボックス1"},
    "P09_複勝なし_馬連1点_三連複◎1軸3":          {                    "馬連": "◎◯1",        "三連複": "◎1軸3"},
    "P10_三連複のみ_◎1軸3":                      {                                          "三連複": "◎1軸3"},
    "P11_三連複のみ_◎◯軸2":                      {                                          "三連複": "◎◯軸2"},
    "P12_三連複のみ_ボックス1":                   {                                          "三連複": "ボックス1"},
    # ── 追加パターン ──────────────────────────────────
    # ◎◯2頭軸 × ▲△× 3頭流し（3点）
    "P13_三連複のみ_◎◯軸3流し":                  {                                          "三連複": "◎◯軸3"},
    "P14_複勝なし_馬連◎軸_三連複◎◯軸3流し":     {                    "馬連": "◎軸2",       "三連複": "◎◯軸3"},
    "P15_複勝なし_馬連ボックス_三連複◎◯軸3流し": {                    "馬連": "ボックス3",  "三連複": "◎◯軸3"},
    # ◎1頭軸 × C(◯▲△×, 2) 流し（6点）
    "P16_三連複のみ_◎1軸4頭流し":                 {                                          "三連複": "◎1軸6"},
    "P17_複勝なし_馬連◎軸_三連複◎1軸4頭流し":    {                    "馬連": "◎軸2",       "三連複": "◎1軸6"},
    "P18_複勝なし_馬連ボックス_三連複◎1軸4頭流し":{                    "馬連": "ボックス3",  "三連複": "◎1軸6"},
    # ◎◯▲△ 4頭ボックス（C(4,3)=4点）
    "P19_三連複のみ_4頭ボックス":                  {                                          "三連複": "4頭ボックス"},
    "P20_複勝なし_馬連◎軸_三連複4頭ボックス":     {                    "馬連": "◎軸2",       "三連複": "4頭ボックス"},
    "P21_複勝なし_馬連ボックス_三連複4頭ボックス": {                    "馬連": "ボックス3",  "三連複": "4頭ボックス"},
    # ── 複勝のみ比較（LALOプラン候補）─────────────────────
    "PL1_複勝のみ_◎1点":    {"複勝": "◎1"},
    "PL2_複勝のみ_◎◯2点":   {"複勝": "◎◯2"},
    "PL3_複勝のみ_◎◯▲3点":  {"複勝": "◎◯▲3"},
    # ── 馬連追加パターン（三連複はボックス1固定で比較）─────
    # ◎軸3点: ◎-◯、◎-▲、◎-△
    "P22_複勝なし_馬連◎軸3_三連複ボックス1":      {                    "馬連": "◎軸3",       "三連複": "ボックス1"},
    # 4頭ボックス: C(◎◯▲△, 2) = 6点
    "P23_複勝なし_馬連4頭ボックス6点_三連複ボックス1": {               "馬連": "4頭ボックス6", "三連複": "ボックス1"},
    # ◎軸4点: ◎-◯、◎-▲、◎-△、◎-×
    "P24_複勝なし_馬連◎軸4_三連複ボックス1":      {                    "馬連": "◎軸4",       "三連複": "ボックス1"},
}

# =========================================================
# ユーティリティ
# =========================================================
def floor100(x: float) -> int:
    return max((int(x) // MIN_UNIT) * MIN_UNIT, MIN_UNIT)


def get_payout(combo: list[int], bet_type: str, kekka: dict) -> int:
    """実払戻配当（100円あたり）を返す。外れ=0。"""
    if bet_type == "複勝":
        return int(kekka.get("複勝", {}).get(combo[0], 0) or 0)
    elif bet_type == "馬連":
        key = "-".join(map(str, sorted(combo)))
        return int(kekka.get("馬連", {}).get(key, 0) or 0)
    elif bet_type == "三連複":
        key = "-".join(map(str, sorted(combo)))
        return int(kekka.get("三連複", {}).get(key, 0) or 0)
    return 0


# =========================================================
# 買い目生成（パターン別）
# =========================================================
def generate_bets(
    h1: int | None,   # ◎
    h2: int | None,   # ◯
    h3: int | None,   # ▲
    h4: int | None,   # △
    h5: int | None,   # ×
    bet_type: str,
    pattern_key: str,
    budget: int,
    kekka: dict,
) -> list[dict]:
    """
    1馬券種・1パターン分の買い目リストを返す。
    各要素: {combo, 購入額, 的中, 払戻, 馬券種, パターン構造}
    """
    if h1 is None:
        return []

    combos: list[list[int]] = []

    if bet_type == "複勝":
        if pattern_key == "◎1":
            combos = [[h1]]
        elif pattern_key == "◎◯2" and h2:
            combos = [[h1], [h2]]
        elif pattern_key == "◎◯▲3":
            c = [[h1]]
            if h2: c.append([h2])
            if h3: c.append([h3])
            combos = c

    elif bet_type == "馬連":
        if pattern_key == "◎◯1" and h2:
            # ◎-◯ 1点
            combos = [sorted([h1, h2])]
        elif pattern_key == "◎軸2":
            # ◎-◯、◎-▲ 2点
            c = []
            if h2: c.append(sorted([h1, h2]))
            if h3: c.append(sorted([h1, h3]))
            combos = c
        elif pattern_key == "ボックス3":
            # ◎◯▲ボックス 3点
            c = []
            if h2: c.append(sorted([h1, h2]))
            if h3: c.append(sorted([h1, h3]))
            if h2 and h3: c.append(sorted([h2, h3]))
            combos = c
        elif pattern_key == "◎軸3":
            # ◎-◯、◎-▲、◎-△ 3点（◎軸×◯▲△流し）
            c = []
            if h2: c.append(sorted([h1, h2]))
            if h3: c.append(sorted([h1, h3]))
            if h4: c.append(sorted([h1, h4]))
            combos = c
        elif pattern_key == "4頭ボックス6":
            # ◎◯▲△ 4頭ボックス C(4,2)=6点
            top4 = [h for h in [h1, h2, h3, h4] if h is not None]
            combos = [sorted(list(pair)) for pair in itertools.combinations(top4, 2)]
        elif pattern_key == "◎軸4":
            # ◎-◯、◎-▲、◎-△、◎-× 4点（◎軸×◯▲△×流し）
            c = []
            if h2: c.append(sorted([h1, h2]))
            if h3: c.append(sorted([h1, h3]))
            if h4: c.append(sorted([h1, h4]))
            if h5: c.append(sorted([h1, h5]))
            combos = c

    elif bet_type == "三連複":
        if pattern_key == "ボックス1":
            if h2 and h3:
                combos = [sorted([h1, h2, h3])]
        elif pattern_key == "◎◯軸2":
            # ◎◯2頭軸 × ▲△ 2点
            if h2 and h3:
                combos = [sorted([h1, h2, h3])]
                if h4:
                    combos.append(sorted([h1, h2, h4]))
        elif pattern_key == "◎◯軸3":
            # ◎◯2頭軸 × ▲△× 3点（追加）
            if h2:
                c = []
                for h in [h3, h4, h5]:
                    if h:
                        c.append(sorted([h1, h2, h]))
                combos = c
        elif pattern_key == "◎1軸3":
            # ◎1頭軸 × C(◯▲△, 2) = 3点
            c = []
            if h2 and h3: c.append(sorted([h1, h2, h3]))
            if h2 and h4: c.append(sorted([h1, h2, h4]))
            if h3 and h4: c.append(sorted([h1, h3, h4]))
            combos = c
        elif pattern_key == "◎1軸6":
            # ◎1頭軸 × C(◯▲△×, 2) = 最大6点（追加）
            others = [h for h in [h2, h3, h4, h5] if h is not None]
            combos = [sorted([h1, a, b]) for a, b in itertools.combinations(others, 2)]
        elif pattern_key == "4頭ボックス":
            # ◎◯▲△ 4頭ボックス = C(4,3) = 4点（追加）
            top4 = [h for h in [h1, h2, h3, h4] if h is not None]
            combos = [sorted(list(c)) for c in itertools.combinations(top4, 3)] if len(top4) >= 3 else []
        elif pattern_key == "期待値3":
            top = [x for x in [h1, h2, h3, h4] if x is not None]
            combos = [list(sorted(c)) for c in itertools.combinations(top[:4], 3)] if len(top) >= 3 else []
            # 期待値順は省略（確率情報なし）→上から取る
            combos = combos[:3]

    if not combos:
        return []

    per_bet = floor100(budget / len(combos))
    results = []
    for combo in combos:
        pay_per100 = get_payout(combo, bet_type, kekka)
        hit        = 1 if pay_per100 > 0 else 0
        payout     = int(per_bet * pay_per100 / 100) if hit else 0
        results.append({
            "bet_type":    bet_type,
            "pattern_key": pattern_key,
            "combo":       "-".join(map(str, combo)),
            "購入額":      per_bet,
            "的中":        hit,
            "払戻":        payout,
            "払戻配当":    pay_per100,
        })
    return results


# =========================================================
# 1レース分：全パターン適用
# =========================================================
def simulate_race(
    race_df: pd.DataFrame,
    kekka: dict,
    strategy: dict,
    budget: int,
) -> dict[str, list[dict]]:
    """
    returns: {pattern_name: [bet_dict, ...]}
    """
    if race_df.empty:
        return {}

    place   = str(race_df["場所"].iloc[0])
    cls_raw = str(race_df["クラス名"].iloc[0])
    cls     = CLASS_NORMALIZE.get(cls_raw, cls_raw)
    bet_info = strategy.get(place, {}).get(cls) or strategy.get(place, {}).get(cls_raw) or {}
    if not bet_info:
        return {}

    # 印ごとの馬番
    def first_ban(mark):
        rows = race_df[race_df["mark"] == mark]
        return int(rows.iloc[0]["馬番"]) if not rows.empty else None

    h1 = first_ban("◎")
    h2 = first_ban("◯")
    h3 = first_ban("▲")
    h4 = first_ban("△")
    h5 = first_ban("×")

    if h1 is None:
        return {}

    out: dict[str, list[dict]] = {}

    for pat_name, pat_def in PATTERNS.items():
        # パターンが使う馬券種 × 戦略にある馬券種の積集合
        active_types = [bt for bt in pat_def if bt in bet_info]
        if not active_types:
            continue

        # 予算按分: bet_ratio を active_types で再正規化
        total_ratio = sum(bet_info[bt]["bet_ratio"] for bt in active_types)
        if total_ratio <= 0:
            continue

        bets: list[dict] = []
        for bt in active_types:
            alloc = floor100(budget * bet_info[bt]["bet_ratio"] / total_ratio)
            b = generate_bets(h1, h2, h3, h4, h5, bt, pat_def[bt], alloc, kekka)
            for item in b:
                item["pattern"] = pat_name
            bets.extend(b)

        if bets:
            out[pat_name] = bets

    return out


# =========================================================
# モデル推論
# =========================================================
from utils import parse_time_str  # backtest.py と同じ


def _predict_lgbm(df: pd.DataFrame) -> np.ndarray:
    obj   = joblib.load(LGBM_PATH)
    model = obj["model"]
    enc   = obj["encoders"]
    fcols = obj["feature_cols"]
    df    = df.copy()
    for col in ["前走走破タイム", "前走着差タイム"]:
        if col in df.columns:
            df[col] = parse_time_str(df[col])
    for col, le in enc.items():
        if col not in df.columns:
            df[col] = 0
            continue
        df[col] = df[col].astype(str).fillna("__NaN__")
        known   = set(le.classes_)
        df[col] = df[col].apply(lambda x: x if x in known else "__NaN__")
        if "__NaN__" not in le.classes_:
            le.classes_ = np.append(le.classes_, "__NaN__")
        df[col] = le.transform(df[col])
    for col in fcols:
        if col not in df.columns:
            df[col] = 0
    return model.predict_proba(df[fcols])[:, 1]


def _predict_cat(df: pd.DataFrame) -> np.ndarray:
    obj   = joblib.load(CAT_PATH)
    model = obj["model"]
    fcols = obj["feature_cols"]
    cat_cols = [
        "種牡馬","父タイプ名","母父馬","母父タイプ名","毛色","馬主(最新/仮想)","生産者",
        "芝・ダ","コース区分","芝(内・外)","馬場状態","天気","クラス名","場所",
        "性別","斤量","ブリンカー","重量種別","年齢限定","限定","性別限定","指定条件",
        "前走場所","前芝・ダ","前走馬場状態","前走斤量","前好走",
    ]
    df = df.copy()
    for col in ["前走走破タイム", "前走着差タイム"]:
        if col in df.columns:
            df[col] = parse_time_str(df[col])
    for col in cat_cols:
        df[col] = df[col].fillna("__NaN__").astype(str) if col in df.columns else "__NaN__"
    for col in fcols:
        if col not in df.columns:
            df[col] = 0.0
    ci = [i for i, c in enumerate(fcols) if c in cat_cols]
    pool = Pool(df[fcols], cat_features=ci)
    return model.predict_proba(pool)[:, 1]


def ensemble_predict(df: pd.DataFrame) -> np.ndarray:
    """LightGBM + CatBoost の加重平均（スタッキング優先）。"""
    logger.info("LightGBM 推論中...")
    p_lgbm = _predict_lgbm(df)
    logger.info("CatBoost 推論中...")
    p_cat  = _predict_cat(df)

    if STACK_PATH.exists():
        try:
            logger.info("スタッキング推論中...")
            meta_obj  = joblib.load(STACK_PATH)
            meta_model = meta_obj["meta_model"]
            meta_enc   = meta_obj["meta_encoders"]
            meta_cols  = meta_obj["meta_cols"]
            meta_df    = df.copy()
            meta_df["クラス名"] = meta_df["クラス名"].map(CLASS_NORMALIZE).fillna(meta_df["クラス名"])
            if "出走頭数" not in meta_df.columns or meta_df["出走頭数"].isna().all():
                meta_df["出走頭数"] = meta_df.groupby(COL_RACE_ID)["馬番"].transform("count")
            meta_df = meta_df.reindex(columns=_META_EXTRA).copy()
            for col in meta_df.select_dtypes(include="object").columns:
                if col in meta_enc:
                    le = meta_enc[col]
                    meta_df[col] = meta_df[col].fillna("__NaN__").astype(str)
                    known = set(le.classes_)
                    meta_df[col] = meta_df[col].apply(lambda x: x if x in known else "__NaN__")
                    if "__NaN__" not in le.classes_:
                        le.classes_ = np.append(le.classes_, "__NaN__")
                    meta_df[col] = le.transform(meta_df[col])
            meta_df["lgbm"]        = p_lgbm
            meta_df["catboost"]    = p_cat
            meta_df["transformer"] = 0.0  # torch なし
            proba = meta_model.predict_proba(meta_df[meta_cols])[:, 1]
            if STACK_CAL.exists():
                cal = joblib.load(STACK_CAL)
                return cal["calibrator"].transform(proba)
            return proba
        except Exception as e:
            logger.warning(f"スタッキング失敗: {e}。2モデル平均にフォールバック")

    raw = 0.5 * p_lgbm + 0.5 * p_cat
    if ENS_CAL.exists():
        cal = joblib.load(ENS_CAL)
        return cal["calibrator"].transform(raw)
    return raw


def assign_marks(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["mark"] = ""
    for _, g in df.groupby(COL_RACE_ID, sort=False):
        ranked = g["prob"].rank(ascending=False, method="first")
        for idx, r in ranked.items():
            if r <= 5:
                df.at[idx, "mark"] = {1:"◎",2:"◯",3:"▲",4:"△",5:"×"}[int(r)]
    return df


# =========================================================
# kekka ロード
# =========================================================
def load_kekka(path: Path) -> dict:
    logger.info(f"kekka 読み込み: {path}")
    df = pd.read_csv(path, encoding="cp932", low_memory=False)
    df["race_id"]  = df["レースID(新)"].astype(str).str[:16]
    df["確定着順"] = pd.to_numeric(df["確定着順"], errors="coerce")
    df["馬番"]     = pd.to_numeric(df["馬番"],     errors="coerce")

    kd = {}
    for rid, g in df.groupby("race_id"):
        g    = g.sort_values("確定着順")
        ent  = {"複勝": {}, "馬連": {}, "三連複": {}}
        top3 = [int(h) for h in g[g["確定着順"] <= 3]["馬番"].tolist() if pd.notna(h)]
        for _, row in g[g["確定着順"] <= 3].iterrows():
            ban, pay = row["馬番"], row["複勝配当"]
            if pd.notna(ban) and pd.notna(pay):
                ent["複勝"][int(ban)] = int(pay)
        r1_rows = g[g["確定着順"] == 1]
        if not r1_rows.empty and len(top3) >= 2:
            r1 = r1_rows.iloc[0]
            key = "-".join(map(str, sorted(top3[:2])))
            if pd.notna(r1.get("馬連")):
                ent["馬連"][key] = int(r1["馬連"])
            if len(top3) >= 3:
                key = "-".join(map(str, sorted(top3[:3])))
                if pd.notna(r1.get("３連複")):
                    ent["三連複"][key] = int(r1["３連複"])
        kd[str(rid)] = ent
    logger.info(f"払戻辞書: {len(kd):,}レース")
    return kd


# =========================================================
# 集計・出力
# =========================================================
def summarize(records: list[dict], period: str) -> pd.DataFrame:
    if not records:
        logger.warning(f"[{period}] 結果なし")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    rows = []
    for pat, grp in df.groupby("pattern"):
        invest = grp["購入額"].sum()
        ret    = grp["払戻"].sum()
        hits   = grp["的中"].sum()
        total  = len(grp)
        roi    = ret / invest * 100 if invest > 0 else 0
        races  = grp["race_id"].nunique()
        # 馬券種別
        by_type = {}
        for bt, sg in grp.groupby("bet_type"):
            i2  = sg["購入額"].sum()
            r2  = sg["払戻"].sum()
            h2  = sg["的中"].sum()
            n2  = len(sg)
            by_type[bt] = {
                "投資": int(i2), "払戻": int(r2),
                "ROI": round(r2/i2*100,1) if i2>0 else 0,
                "的中率": round(h2/n2*100,1) if n2>0 else 0,
                "的中": int(h2), "点数": int(n2),
            }
        rows.append({
            "パターン":      pat,
            "対象レース数":  races,
            "総点数":        total,
            "総投資":        int(invest),
            "総払戻":        int(ret),
            "ROI":           round(roi, 1),
            "総的中数":      int(hits),
            "的中率":        round(hits/total*100, 1) if total > 0 else 0,
            # 複勝
            "複勝_ROI":      by_type.get("複勝", {}).get("ROI", "-"),
            "複勝_的中率":   by_type.get("複勝", {}).get("的中率", "-"),
            "複勝_点数":     by_type.get("複勝", {}).get("点数", 0),
            # 馬連
            "馬連_ROI":      by_type.get("馬連", {}).get("ROI", "-"),
            "馬連_的中率":   by_type.get("馬連", {}).get("的中率", "-"),
            "馬連_点数":     by_type.get("馬連", {}).get("点数", 0),
            # 三連複
            "三連複_ROI":    by_type.get("三連複", {}).get("ROI", "-"),
            "三連複_的中率": by_type.get("三連複", {}).get("的中率", "-"),
            "三連複_点数":   by_type.get("三連複", {}).get("点数", 0),
        })

    result_df = pd.DataFrame(rows).sort_values("ROI", ascending=False).reset_index(drop=True)

    print(f"\n{'='*80}")
    print(f"  期間: {period}  /  戦略フィルタあり  /  全{result_df['対象レース数'].max()}レース最大")
    print(f"{'='*80}")
    print(result_df[[
        "パターン","対象レース数","ROI",
        "複勝_ROI","複勝_的中率",
        "馬連_ROI","馬連_的中率",
        "三連複_ROI","三連複_的中率",
    ]].to_string(index=True))
    print()
    return result_df


# =========================================================
# メイン
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", default="all",
                        choices=["valid", "2024", "all"],
                        help="検証期間: valid(2023) / 2024(2024-) / all(両方)")
    parser.add_argument("--budget", type=int, default=BUDGET,
                        help=f"1レース予算（デフォルト {BUDGET}円）")
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # ── データロード ──────────────────────────────────────
    logger.info(f"マスターCSV読み込み: {MASTER_CSV}")
    master = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    master["日付"] = pd.to_numeric(master["日付"], errors="coerce")

    # 除外
    if "場所" in master.columns and "クラス名" in master.columns:
        before = len(master)
        master = master[
            ~master["場所"].isin(EXCLUDE_PLACES) &
            ~master["クラス名"].isin(EXCLUDE_CLASSES)
        ].copy().reset_index(drop=True)
        logger.info(f"除外後: {len(master):,}行（除外 {before-len(master):,}行）")

    kekka_dict = load_kekka(KEKKA_CSV)

    # ── 戦略読み込み ──────────────────────────────────────
    if not STRATEGY_JSON.exists():
        logger.error(f"strategy_weights.json が見つかりません: {STRATEGY_JSON}")
        return
    with open(STRATEGY_JSON, encoding="utf-8") as f:
        strategy = json.load(f)
    logger.info(f"戦略ファイル読み込み: {STRATEGY_JSON.name}")

    # ── 期間設定 ──────────────────────────────────────────
    periods = []
    if args.period in ("valid", "all"):
        periods.append(("valid(2023)", master[master["split"] == "valid"].copy()))
    if args.period in ("2024", "all"):
        periods.append(("2024+(test)", master[master["split"] == "test"].copy()))

    all_summary = []

    for period_name, df_period in periods:
        logger.info(f"\n{'='*60}")
        logger.info(f"期間: {period_name}  ({len(df_period):,}行)")

        # ── 推論 ──────────────────────────────────────────
        logger.info("アンサンブル推論中...")
        df_period = df_period.copy().reset_index(drop=True)
        df_period["prob"] = ensemble_predict(df_period)

        logger.info("印付与中...")
        df_period = assign_marks(df_period)

        # ── レース単位シミュレーション ────────────────────
        race_ids = df_period[COL_RACE_ID].unique()
        logger.info(f"シミュレーション開始: {len(race_ids):,}レース × {len(PATTERNS)}パターン")

        records: list[dict] = []
        skipped_strat = 0

        for rid in tqdm(race_ids, desc=period_name):
            rdf = df_period[df_period[COL_RACE_ID] == rid].copy()
            kekka = kekka_dict.get(str(rid), {"複勝":{}, "馬連":{}, "三連複":{}})

            pat_bets = simulate_race(rdf, kekka, strategy, args.budget)
            if not pat_bets:
                skipped_strat += 1
                continue

            for pat_name, bets in pat_bets.items():
                for b in bets:
                    b["race_id"]     = rid
                    b["period"]      = period_name
                    records.append(b)

        logger.info(f"  戦略対象外スキップ: {skipped_strat}レース")

        summary_df = summarize(records, period_name)
        summary_df["期間"] = period_name
        all_summary.append(summary_df)

    if all_summary:
        final = pd.concat(all_summary, ignore_index=True)
        out_path = REPORT_DIR / "pattern_comparison.csv"
        final.to_csv(out_path, index=False, encoding="utf-8-sig")
        logger.info(f"\n結果保存: {out_path}")
        print(f"\n>>> 結果CSV: {out_path}")


if __name__ == "__main__":
    main()
