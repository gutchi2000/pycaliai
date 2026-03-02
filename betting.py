"""
betting.py
PyCaLiAI - 買い目自動生成（回収率重視・1万円固定）

全馬券種合計が予算（デフォルト10,000円）に収まるよう按分。
複勝/馬連/三連複/三連単を同時出力。

Usage:
    python betting.py --race_id 2024010606010101
    python betting.py --race_id 2024010606010101 --budget 10000
"""

from __future__ import annotations

import argparse
import itertools
import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

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

# 馬券種ごとのJRA控除率
TAKEOUT = {
    "複勝":   0.20,
    "馬連":   0.225,
    "三連複": 0.25,
    "三連単": 0.275,
}

# 馬券種ごとの予算配分比率（合計1.0）
BUDGET_RATIO = {
    "複勝":   0.20,
    "馬連":   0.25,
    "三連複": 0.30,
    "三連単": 0.25,
}


# =========================================================
# ユーティリティ
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


def estimate_odds(win_prob: float, bet_type: str) -> float:
    """JRA控除率ベースの推定オッズ。"""
    rate = TAKEOUT.get(bet_type, 0.25)
    if win_prob <= 0:
        return 0.0
    return round((1 - rate) / win_prob, 1)


def calc_win_prob_pl(
    horses: list[int],
    prob_series: pd.Series,
    ordered: bool,
) -> float:
    """Plackett-Luce近似で的中確率を推定する。"""
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


def floor_to_unit(amount: int, unit: int = MIN_UNIT) -> int:
    """unit単位に切り捨て。"""
    return (amount // unit) * unit


# =========================================================
# 各モデル予測
# =========================================================
def predict_lgbm(df: pd.DataFrame) -> np.ndarray:
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
    return model.predict_proba(df[feature_cols])[:, 1]


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


def ensemble_predict(df: pd.DataFrame) -> np.ndarray:
    p_lgbm  = predict_lgbm(df)
    p_cat   = predict_catboost(df)
    p_torch = predict_transformer(df)
    w       = MODEL_WEIGHTS
    total   = sum(w.values())
    return (
        (w["lgbm"]        / total) * p_lgbm  +
        (w["catboost"]    / total) * p_cat   +
        (w["transformer"] / total) * p_torch
    )


# =========================================================
# 印付与
# =========================================================
def assign_marks(race_df: pd.DataFrame, proba: np.ndarray) -> pd.DataFrame:
    df = race_df.copy()
    df["prob"] = proba
    df["rank"] = df["prob"].rank(ascending=False, method="first").astype(int)
    df["mark"] = df["rank"].map(
        {1: "◎", 2: "◯", 3: "▲", 4: "△", 5: "×"}
    ).fillna("")
    return df.sort_values("rank").reset_index(drop=True)


# =========================================================
# 買い目生成
# =========================================================
def build_bets(race_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """各馬券種の買い目候補（購入額未設定）を生成する。"""
    prob_series = race_df.set_index("馬番")["prob"]
    mark_dict   = race_df.set_index("馬番")["mark"].to_dict()

    hon    = race_df[race_df["mark"] == "◎"]["馬番"].tolist()
    taikou = race_df[race_df["mark"] == "◯"]["馬番"].tolist()
    sabo   = race_df[race_df["mark"] == "▲"]["馬番"].tolist()
    top3   = hon + taikou + sabo

    def make_row(combo: tuple, bet_type: str, ordered: bool) -> dict:
        p    = calc_win_prob_pl(list(combo), prob_series, ordered)
        p    = min(p, 0.99)
        odds = estimate_odds(p, bet_type)
        ev   = round(p * odds, 3)
        sep  = "→" if ordered else "-"
        return {
            "買い目":      sep.join(map(str, combo)),
            "印":          "".join(mark_dict.get(h, "") for h in combo),
            "推定的中確率": round(p, 4),
            "推定オッズ":   odds,
            "推定期待値":   ev,
        }

    results: dict[str, pd.DataFrame] = {}

    # 複勝（◎◯ 2点）
    fuku_rows = []
    for h in hon + taikou:
        p    = float(prob_series.get(h, 0.0))
        odds = estimate_odds(p, "複勝")
        fuku_rows.append({
            "買い目":      str(h),
            "印":          mark_dict.get(h, ""),
            "推定的中確率": round(p, 4),
            "推定オッズ":   odds,
            "推定期待値":   round(p * odds, 3),
        })
    results["複勝"] = pd.DataFrame(fuku_rows).sort_values(
        "推定期待値", ascending=False
    ).reset_index(drop=True)

    # 馬連（◎軸 2点）
    baren_rows = []
    if hon:
        for h in taikou + sabo:
            combo = tuple(sorted([hon[0], h]))
            baren_rows.append(make_row(combo, "馬連", ordered=False))
    results["馬連"] = pd.DataFrame(baren_rows).sort_values(
        "推定期待値", ascending=False
    ).reset_index(drop=True)

    # 三連複（上位4頭から3頭 最大4点→期待値上位3点に絞る）
    sanfuku_rows = []
    if len(top3) >= 3:
        for combo in itertools.combinations(top3[:4], 3):
            sanfuku_rows.append(make_row(combo, "三連複", ordered=False))
    results["三連複"] = pd.DataFrame(sanfuku_rows).sort_values(
        "推定期待値", ascending=False
    ).head(3).reset_index(drop=True)

    # 三連単（◎◯▲の順列 上位3点）
    santen_rows = []
    if len(top3) >= 3:
        for perm in itertools.permutations(top3[:3], 3):
            santen_rows.append(make_row(perm, "三連単", ordered=True))
    results["三連単"] = pd.DataFrame(santen_rows).sort_values(
        "推定期待値", ascending=False
    ).head(3).reset_index(drop=True)

    return results


def allocate_budget(
    bets: dict[str, pd.DataFrame],
    budget: int = BUDGET,
    min_unit: int = MIN_UNIT,
) -> dict[str, pd.DataFrame]:
    """
    全馬券種合計がbudgetに収まるよう予算を按分する。

    配分手順:
    1. BUDGET_RATIOで馬券種ごとの予算を決定
    2. 各馬券種内で点数均等配分（min_unit単位切り捨て）
    3. 余りは最も期待値が高い馬券種に加算
    """
    allocated = {}
    remaining = budget

    # アクティブな馬券種のみ処理
    active = {k: v for k, v in bets.items() if not v.empty}
    if not active:
        return bets

    # 各馬券種の予算上限を計算
    total_ratio = sum(BUDGET_RATIO[k] for k in active)
    type_budgets = {
        k: floor_to_unit(int(budget * BUDGET_RATIO[k] / total_ratio))
        for k in active
    }

    for bet_type, df in active.items():
        n        = len(df)
        alloc    = type_budgets[bet_type]
        per_bet  = floor_to_unit(alloc // n)
        per_bet  = max(per_bet, min_unit)

        # 点数×単価が予算を超えないよう調整
        while per_bet * n > alloc and per_bet > min_unit:
            per_bet -= min_unit

        df = df.copy()
        df["購入額(円)"] = per_bet
        allocated[bet_type] = df

    # 余り予算を三連複に加算（最も期待値が安定）
    total_used = sum(
        df["購入額(円)"].sum() for df in allocated.values()
    )
    remainder = floor_to_unit(budget - total_used)
    if remainder >= min_unit and "三連複" in allocated:
        df = allocated["三連複"].copy()
        extra = floor_to_unit(remainder // len(df))
        if extra >= min_unit:
            df["購入額(円)"] += extra
        allocated["三連複"] = df

    # 空の馬券種も結果に含める
    for k in bets:
        if k not in allocated:
            allocated[k] = bets[k]

    return allocated


# =========================================================
# 表示
# =========================================================
def print_race_result(race_df: pd.DataFrame) -> None:
    print("\n" + "=" * 65)
    print("レース予測結果")
    print("=" * 65)
    cols = [c for c in ["馬番", "馬名", "mark", "prob"] if c in race_df.columns]
    print(
        race_df[cols]
        .rename(columns={"mark": "印", "prob": "複勝確率"})
        .to_string(index=False)
    )


def print_all_bets(
    results: dict[str, pd.DataFrame],
    budget: int,
) -> None:
    print(f"\n{'=' * 65}")
    print(f"買い目一覧（予算: {budget:,}円 / 回収率重視）")
    print(f"{'=' * 65}")

    total_all = 0
    for bet_type in ["複勝", "馬連", "三連複", "三連単"]:
        df = results.get(bet_type, pd.DataFrame())
        if df.empty or "購入額(円)" not in df.columns:
            continue
        total = int(df["購入額(円)"].sum())
        total_all += total
        print(f"\n【{bet_type}】{len(df)}点  合計: {total:,}円")
        print("-" * 65)
        print(
            df[["買い目", "印", "推定的中確率", "推定オッズ", "推定期待値", "購入額(円)"]]
            .to_string(index=False)
        )

    print(f"\n{'=' * 65}")
    print(f"全馬券種合計: {total_all:,}円  ／  予算: {budget:,}円")
    print(f"{'=' * 65}")
    print("\n【期待値の見方】")
    print("  期待値 > 1.0 → 長期的にプラス収支の見込み")
    print("  期待値 < 1.0 → 長期的にマイナス収支の見込み")
    print("  ※推定オッズはJRA控除率ベースの理論値。実際のオッズと異なります。")


# =========================================================
# main
# =========================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="PyCaLiAI 買い目生成（全馬券種合計1万円固定・回収率重視）"
    )
    parser.add_argument("--race_id", type=int, required=True,
                        help="レースID(新/馬番無) 例: 2024010606010101")
    parser.add_argument("--budget", type=int, default=BUDGET,
                        help=f"予算（円）デフォルト{BUDGET:,}円")
    args = parser.parse_args()

    logger.info(f"マスターCSV読み込み: {MASTER_CSV}")
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)

    race_df = df[df[COL_RACE_ID] == args.race_id].copy().reset_index(drop=True)
    if len(race_df) == 0:
        raise ValueError(f"レースID {args.race_id} が見つかりません。")
    logger.info(f"レース: {args.race_id}  出走頭数: {len(race_df)}頭")

    proba   = ensemble_predict(race_df)
    race_df = assign_marks(race_df, proba)
    print_race_result(race_df)

    bets    = build_bets(race_df)
    bets    = allocate_budget(bets, budget=args.budget)
    print_all_bets(bets, args.budget)

    # 保存
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for bet_type, bdf in bets.items():
        if not bdf.empty and "購入額(円)" in bdf.columns:
            bdf = bdf.copy()
            bdf.insert(0, "馬券種", bet_type)
            all_rows.append(bdf)
    if all_rows:
        out_path = REPORT_DIR / f"bets_{args.race_id}.csv"
        pd.concat(all_rows).to_csv(out_path, index=False, encoding="utf-8-sig")
        logger.info(f"買い目保存: {out_path}")


if __name__ == "__main__":
    main()