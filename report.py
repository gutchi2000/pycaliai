"""
report.py
PyCaLiAI - SHAP値→自然言語レポート生成（CLI版）

Usage:
    python report.py --race_id 2024010606010101
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from catboost import Pool
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

try:
    import japanize_matplotlib  # noqa: F401
except ImportError:
    plt.rcParams["font.family"] = "MS Gothic"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# =========================================================
# パス設定
# =========================================================
BASE_DIR         = Path(r"E:\PyCaLiAI")
DATA_DIR         = BASE_DIR / "data"
MODEL_DIR        = BASE_DIR / "models"
REPORT_DIR       = BASE_DIR / "reports"

MASTER_CSV       = DATA_DIR  / "master_20130105-20251228.csv"
LGBM_MODEL_PATH  = MODEL_DIR / "lgbm_optuna_v1.pkl"
CAT_MODEL_PATH   = MODEL_DIR / "catboost_optuna_v1.pkl"
TORCH_MODEL_PATH = MODEL_DIR / "transformer_optuna_v1.pkl"

TARGET      = "fukusho_flag"
COL_RACE_ID = "レースID(新/馬番無)"
BUDGET      = 10_000
MIN_UNIT    = 100

MODEL_WEIGHTS = {
    "lgbm":        0.7425,
    "catboost":    0.7472,
    "transformer": 0.7496,
}

BUDGET_RATIO = {
    "複勝":   0.20,
    "馬連":   0.25,
    "三連複": 0.30,
    "三連単": 0.25,
}

TAKEOUT = {
    "複勝":   0.20,
    "馬連":   0.225,
    "三連複": 0.25,
    "三連単": 0.275,
}

# =========================================================
# 特徴量の日本語説明マップ
# =========================================================
FEATURE_LABEL = {
    "前走確定着順":       "前走着順",
    "前走着差タイム":     "前走着差",
    "前走上り3F":         "前走上り3F",
    "前走上り3F順":       "前走上り順位",
    "前走Ave-3F":         "前走平均3F",
    "前走出走頭数":       "前走頭数",
    "前走馬体重":         "前走馬体重",
    "前走馬体重増減":     "馬体重増減",
    "距離":               "距離",
    "出走頭数":           "出走頭数",
    "枠番":               "枠番",
    "馬番":               "馬番",
    "年齢":               "年齢",
    "間隔":               "中間隔",
    "騎手年齢":           "騎手年齢",
    "前1角":              "前走1角",
    "前2角":              "前走2角",
    "前3角":              "前走3角",
    "前4角":              "前走4角",
    "前走PCI3":           "前走PCI3",
    "前走RPCI":           "前走RPCI",
    "前走平均1Fタイム":   "前走1Fタイム",
    "芝・ダ":             "芝ダ",
    "馬場状態":           "馬場状態",
    "クラス名":           "クラス",
    "場所":               "競馬場",
    "騎手コード":         "騎手",
    "調教師コード":       "調教師",
    "斤量体重比":         "斤量体重比",
    "馬齢斤量差":         "馬齢斤量差",
}

# =========================================================
# SHAPコメント生成ルール
# =========================================================
def shap_to_comment(
    feature: str,
    shap_val: float,
    raw_val: float | None = None,
) -> str | None:
    """
    SHAP値と特徴量名からコメントを生成する。
    shap_val > 0 → 複勝確率を押し上げている要因
    shap_val < 0 → 複勝確率を押し下げている要因
    """
    label = FEATURE_LABEL.get(feature, feature)
    pos   = shap_val > 0
    abs_s = abs(shap_val)

    if abs_s < 0.01:   # 影響小さすぎるものはスキップ
        return None

    rules: dict[str, tuple[str, str]] = {
        "前走確定着順":   ("前走好走が好材料", "前走凡走が不安材料"),
        "前走着差タイム": ("前走接戦が好材料", "前走大差負けが不安材料"),
        "前走上り3F":     ("前走上り速く好材料", "前走上り遅く不安材料"),
        "前走上り3F順":   ("前走上り上位が好材料", "前走上り下位が不安材料"),
        "前走出走頭数":   ("前走多頭数好走が好材料", "前走少頭数のため割引"),
        "前走馬体重増減": ("馬体増減が適正範囲で好材料", "馬体の大幅変動が不安材料"),
        "距離":           ("距離適性が好材料", "距離変化が不安材料"),
        "出走頭数":       ("多頭数で複勝圏広く好材料", "少頭数で複勝圏狭く不安材料"),
        "枠番":           ("枠順が好材料", "枠順が不安材料"),
        "年齢":           ("年齢が好材料", "年齢面が不安材料"),
        "間隔":           ("間隔が好材料", "間隔が不安材料"),
        "騎手年齢":       ("騎手の経験値が好材料", "騎手面が不安材料"),
        "前1角":          ("道中位置取りが好材料", "道中位置取りが不安材料"),
        "芝・ダ":         ("コース種別適性が好材料", "コース種別が不安材料"),
        "馬場状態":       ("馬場状態適性が好材料", "馬場状態が不安材料"),
        "クラス名":       ("クラス適性が好材料", "クラス面で不安材料"),
        "斤量体重比":     ("斤量負担が軽く好材料", "斤量負担が重く不安材料"),
    }

    if feature in rules:
        pos_msg, neg_msg = rules[feature]
        return pos_msg if pos else neg_msg

    # デフォルト
    direction = "プラス材料" if pos else "マイナス材料"
    return f"{label}が{direction}"


# =========================================================
# 自然言語レポート生成
# =========================================================
def generate_comment(
    horse_name: str,
    mark: str,
    prob: float,
    shap_values: np.ndarray,
    feature_names: list[str],
    top_n: int = 3,
) -> str:
    """
    馬1頭分のコメントを生成する。

    上位top_n個の正の要因と、上位1個の負の要因を記述する。
    """
    shap_series = pd.Series(shap_values, index=feature_names)

    pos_factors = shap_series[shap_series > 0].sort_values(ascending=False)
    neg_factors = shap_series[shap_series < 0].sort_values(ascending=True)

    pos_comments = []
    for feat, val in pos_factors.head(top_n).items():
        c = shap_to_comment(feat, val)
        if c:
            pos_comments.append(c)

    neg_comments = []
    for feat, val in neg_factors.head(1).items():
        c = shap_to_comment(feat, val)
        if c:
            neg_comments.append(c)

    prob_pct = f"{prob * 100:.1f}%"
    mark_str = f"{mark} " if mark else ""

    parts = []
    if pos_comments:
        parts.append("、".join(pos_comments))
    if neg_comments:
        parts.append(f"一方{neg_comments[0]}")

    comment_body = "。".join(parts) + "。" if parts else "特記事項なし。"

    return (
        f"{mark_str}{horse_name}（複勝確率{prob_pct}）\n"
        f"  → {comment_body}"
    )


# =========================================================
# 前処理ユーティリティ（betting.pyと共通）
# =========================================================
def parse_time_str(series: pd.Series) -> pd.Series:
    def _convert(val: str) -> float | None:
        try:
            parts = str(val).strip().split(".")
            if len(parts) == 3:
                return int(parts[0]) * 60 + int(parts[1]) + int(parts[2]) / 10
            return float(val)
        except Exception:
            return None
    return series.apply(_convert)


def predict_lgbm_with_shap(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """LightGBMで予測しSHAP値も返す。"""
    obj          = joblib.load(LGBM_MODEL_PATH)
    model        = obj["model"]
    encoders     = obj["encoders"]
    feature_cols = obj["feature_cols"]

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

    X      = df[feature_cols]
    proba  = model.predict_proba(X)[:, 1]

    # SHAP値計算
    explainer   = shap.TreeExplainer(model.booster_)
    shap_values = explainer.shap_values(X)
    # 2値分類の場合、shap_valuesはリストになることがある
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    return proba, shap_values, feature_cols


def predict_catboost(df: pd.DataFrame) -> np.ndarray:
    obj          = joblib.load(CAT_MODEL_PATH)
    model        = obj["model"]
    feature_cols = obj["feature_cols"]
    cat_features_list = [
        "種牡馬", "父タイプ名", "母父馬", "母父タイプ名", "毛色",
        "馬主(最新/仮想)", "生産者",
        "芝・ダ", "コース区分", "芝(内・外)", "馬場状態", "天気",
        "クラス名", "場所", "性別", "斤量", "ブリンカー", "重量種別",
        "年齢限定", "限定", "性別限定", "指定条件",
        "前走場所", "前芝・ダ", "前走馬場状態", "前走斤量", "前好走",
    ]
    df = df.copy()
    for col in ["前走走破タイム", "前走着差タイム"]:
        if col in df.columns:
            df[col] = parse_time_str(df[col])
    for col in cat_features_list:
        df[col] = df[col].fillna("__NaN__").astype(str) if col in df.columns else "__NaN__"
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    cat_indices = [i for i, c in enumerate(feature_cols) if c in cat_features_list]
    pool = Pool(df[feature_cols], cat_features=cat_indices)
    return model.predict_proba(pool)[:, 1]


def predict_transformer(df: pd.DataFrame) -> np.ndarray:
    import torch
    from train_transformer import RaceTransformer, RaceDataset, MAX_HORSES
    from train_transformer import preprocess as torch_preprocess
    from torch.utils.data import DataLoader

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obj          = joblib.load(TORCH_MODEL_PATH)
    model_state  = obj["model_state"]
    model_config = obj["model_config"]
    encoders     = obj["encoders"]
    num_stats    = obj["num_stats"]
    num_cols     = obj["num_cols"]
    cat_cols     = obj["cat_cols"]

    df = df.copy()
    df, _, _ = torch_preprocess(df, encoders=encoders, fit=False, num_stats=num_stats)

    model = RaceTransformer(
        cat_vocab_sizes=model_config["cat_vocab_sizes"],
        cat_cols=model_config["cat_cols"],
        n_num=model_config["n_num"],
        d_model=model_config.get("d_model", 128),
        n_heads=model_config.get("n_heads", 4),
        n_layers=model_config.get("n_layers", 2),
        d_ff=model_config.get("d_ff", 256),
        dropout=model_config.get("dropout", 0.1),
    ).to(DEVICE)
    model.load_state_dict(model_state)
    model.eval()

    ds     = RaceDataset(df, cat_cols, num_cols, model_config["cat_vocab_sizes"])
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)

    all_proba = []
    with torch.no_grad():
        for batch in loader:
            cat    = batch["cat"].to(DEVICE)
            num    = batch["num"].to(DEVICE)
            mask   = batch["mask"].to(DEVICE)
            logits = model(cat, num, mask)
            probas = torch.sigmoid(logits).cpu().numpy()
            valid  = ~batch["mask"].numpy()
            for b in range(len(probas)):
                for h in range(MAX_HORSES):
                    if valid[b, h]:
                        all_proba.append(probas[b, h])

    df_sorted = df.sort_values(COL_RACE_ID).reset_index(drop=True)
    result    = np.zeros(len(df))
    idx = 0
    for _, group in df_sorted.groupby(COL_RACE_ID, sort=True):
        n = min(len(group), MAX_HORSES)
        for i, orig_idx in enumerate(group.index[:n]):
            if idx < len(all_proba):
                result[orig_idx] = all_proba[idx]
                idx += 1
    return result


def ensemble_predict_with_shap(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """アンサンブル予測 + LightGBMのSHAP値を返す。"""
    p_lgbm, shap_values, feature_cols = predict_lgbm_with_shap(df)
    p_cat   = predict_catboost(df)
    p_torch = predict_transformer(df)

    w     = MODEL_WEIGHTS
    total = sum(w.values())
    ensemble_proba = (
        (w["lgbm"]        / total) * p_lgbm  +
        (w["catboost"]    / total) * p_cat   +
        (w["transformer"] / total) * p_torch
    )
    return ensemble_proba, shap_values, feature_cols


def assign_marks(race_df: pd.DataFrame, proba: np.ndarray) -> pd.DataFrame:
    df = race_df.copy()
    df["prob"] = proba
    df["rank"] = df["prob"].rank(ascending=False, method="first").astype(int)
    df["mark"] = df["rank"].map(
        {1: "◎", 2: "◯", 3: "▲", 4: "△", 5: "×"}
    ).fillna("")
    return df.sort_values("rank").reset_index(drop=True)


# =========================================================
# 買い目生成（betting.pyと同じロジックを内包）
# =========================================================
def floor_to_unit(amount: int, unit: int = MIN_UNIT) -> int:
    return (amount // unit) * unit


def estimate_odds(win_prob: float, bet_type: str) -> float:
    rate = TAKEOUT.get(bet_type, 0.25)
    if win_prob <= 0:
        return 0.0
    return round((1 - rate) / win_prob, 1)


def calc_win_prob_pl(
    horses: list[int],
    prob_series: pd.Series,
    ordered: bool,
) -> float:
    import itertools
    total = prob_series.sum()
    if total <= 0:
        return 0.0
    norm = prob_series / total
    if not ordered:
        probs = [norm.get(h, 0.0) for h in horses]
        return float(min(np.prod(probs) ** (1 / len(probs)) * len(probs), 0.99))
    else:
        remaining = 1.0
        prob_val  = 1.0
        for h in horses:
            p = norm.get(h, 0.0)
            if remaining <= 0:
                break
            prob_val  *= p / remaining
            remaining -= p
        return max(float(prob_val), 0.0)


def build_and_allocate_bets(
    race_df: pd.DataFrame,
    budget: int = BUDGET,
) -> dict[str, pd.DataFrame]:
    import itertools

    prob_series = race_df.set_index("馬番")["prob"]
    mark_dict   = race_df.set_index("馬番")["mark"].to_dict()
    hon    = race_df[race_df["mark"] == "◎"]["馬番"].tolist()
    taikou = race_df[race_df["mark"] == "◯"]["馬番"].tolist()
    sabo   = race_df[race_df["mark"] == "▲"]["馬番"].tolist()
    top3   = hon + taikou + sabo

    def make_row(combo, bet_type, ordered):
        p    = min(calc_win_prob_pl(list(combo), prob_series, ordered), 0.99)
        odds = estimate_odds(p, bet_type)
        sep  = "→" if ordered else "-"
        return {
            "買い目":      sep.join(map(str, combo)),
            "印":          "".join(mark_dict.get(h, "") for h in combo),
            "推定的中確率": round(p, 4),
            "推定オッズ":   odds,
            "推定期待値":   round(p * odds, 3),
        }

    results = {}

    # 複勝
    fuku_rows = []
    for h in hon + taikou:
        p    = float(prob_series.get(h, 0.0))
        odds = estimate_odds(p, "複勝")
        fuku_rows.append({
            "買い目": str(h), "印": mark_dict.get(h, ""),
            "推定的中確率": round(p, 4), "推定オッズ": odds,
            "推定期待値": round(p * odds, 3),
        })
    results["複勝"] = pd.DataFrame(fuku_rows).sort_values(
        "推定期待値", ascending=False).reset_index(drop=True)

    # 馬連
    baren_rows = [
        make_row(tuple(sorted([hon[0], h])), "馬連", False)
        for h in taikou + sabo
    ] if hon else []
    results["馬連"] = pd.DataFrame(baren_rows).sort_values(
        "推定期待値", ascending=False).reset_index(drop=True)

    # 三連複
    sanfuku_rows = [
        make_row(combo, "三連複", False)
        for combo in itertools.combinations(top3[:4], 3)
    ] if len(top3) >= 3 else []
    results["三連複"] = pd.DataFrame(sanfuku_rows).sort_values(
        "推定期待値", ascending=False).head(3).reset_index(drop=True)

    # 三連単
    santen_rows = [
        make_row(perm, "三連単", True)
        for perm in itertools.permutations(top3[:3], 3)
    ] if len(top3) >= 3 else []
    results["三連単"] = pd.DataFrame(santen_rows).sort_values(
        "推定期待値", ascending=False).head(3).reset_index(drop=True)

    # 予算按分
    active      = {k: v for k, v in results.items() if not v.empty}
    total_ratio = sum(BUDGET_RATIO[k] for k in active)
    for bet_type, df in active.items():
        n       = len(df)
        alloc   = floor_to_unit(int(budget * BUDGET_RATIO[bet_type] / total_ratio))
        per_bet = max(floor_to_unit(alloc // n), MIN_UNIT)
        while per_bet * n > alloc and per_bet > MIN_UNIT:
            per_bet -= MIN_UNIT
        df = df.copy()
        df["購入額(円)"] = per_bet
        results[bet_type] = df

    # 余りを三連複に加算
    total_used = sum(
        df["購入額(円)"].sum()
        for df in results.values()
        if not df.empty and "購入額(円)" in df.columns
    )
    remainder = floor_to_unit(budget - int(total_used))
    if remainder >= MIN_UNIT and "三連複" in results and not results["三連複"].empty:
        df = results["三連複"].copy()
        extra = floor_to_unit(remainder // len(df))
        if extra >= MIN_UNIT:
            df["購入額(円)"] += extra
        results["三連複"] = df

    return results


# =========================================================
# SHAP棒グラフ保存
# =========================================================
def save_shap_bar(
    shap_values: np.ndarray,
    feature_names: list[str],
    horse_name: str,
    save_path: Path,
    top_n: int = 10,
) -> None:
    sv      = pd.Series(shap_values, index=feature_names)
    sv_abs  = sv.abs().sort_values(ascending=False).head(top_n)
    sv_top  = sv[sv_abs.index]

    colors = ["tomato" if v > 0 else "steelblue" for v in sv_top]
    labels = [FEATURE_LABEL.get(f, f) for f in sv_top.index]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(labels[::-1], sv_top.values[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"SHAP値 TOP{top_n}: {horse_name}")
    ax.set_xlabel("SHAP値（赤=プラス要因 / 青=マイナス要因）")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# CLI出力
# =========================================================
def print_report(
    race_df: pd.DataFrame,
    shap_values: np.ndarray,
    feature_cols: list[str],
    bets: dict[str, pd.DataFrame],
    budget: int,
) -> None:
    race_info = race_df.iloc[0]
    place     = race_info.get("場所", "")
    distance  = race_info.get("距離", "")
    shiba     = race_info.get("芝・ダ", "")
    cls       = race_info.get("クラス名", "")

    print("\n" + "=" * 65)
    print(f"レース予想レポート  {place} {shiba}{distance}m {cls}")
    print("=" * 65)

    # 印馬のみコメント出力（◎◯▲△×）
    marked = race_df[race_df["mark"] != ""].copy()
    for i, row in marked.iterrows():
        horse_name = row.get("馬名", f"{int(row['馬番'])}番")
        mark       = row["mark"]
        prob       = row["prob"]
        sv         = shap_values[i]
        comment    = generate_comment(horse_name, mark, prob, sv, feature_cols)
        print(comment)

    # 買い目
    print(f"\n{'=' * 65}")
    print(f"買い目（予算: {budget:,}円）")
    print(f"{'=' * 65}")
    total_all = 0
    for bet_type in ["複勝", "馬連", "三連複", "三連単"]:
        df = bets.get(bet_type, pd.DataFrame())
        if df.empty or "購入額(円)" not in df.columns:
            continue
        total = int(df["購入額(円)"].sum())
        total_all += total
        print(f"\n【{bet_type}】{len(df)}点  合計: {total:,}円")
        print(df[["買い目", "印", "推定的中確率", "推定オッズ", "推定期待値", "購入額(円)"]].to_string(index=False))
    print(f"\n全馬券種合計: {total_all:,}円  ／  予算: {budget:,}円")


# =========================================================
# main
# =========================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="PyCaLiAI レース予想レポート生成")
    parser.add_argument("--race_id", type=int, required=True)
    parser.add_argument("--budget",  type=int, default=BUDGET)
    args = parser.parse_args()

    logger.info(f"マスターCSV読み込み: {MASTER_CSV}")
    df      = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    race_df = df[df[COL_RACE_ID] == args.race_id].copy().reset_index(drop=True)
    if len(race_df) == 0:
        raise ValueError(f"レースID {args.race_id} が見つかりません。")
    logger.info(f"レース: {args.race_id}  出走頭数: {len(race_df)}頭")

    proba, shap_values, feature_cols = ensemble_predict_with_shap(race_df)
    race_df = assign_marks(race_df, proba)
    bets    = build_and_allocate_bets(race_df, budget=args.budget)

    print_report(race_df, shap_values, feature_cols, bets, args.budget)

    # SHAP棒グラフ（◎のみ）
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    honmei = race_df[race_df["mark"] == "◎"]
    if not honmei.empty:
        i          = honmei.index[0]
        horse_name = honmei.iloc[0].get("馬名", "本命馬")
        save_shap_bar(
            shap_values[i], feature_cols, horse_name,
            REPORT_DIR / f"shap_{args.race_id}_honmei.png",
        )
        logger.info(f"SHAP棒グラフ保存: shap_{args.race_id}_honmei.png")


if __name__ == "__main__":
    main()