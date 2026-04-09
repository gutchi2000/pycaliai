"""
backtest.py
PyCaLiAI - バックテスト（実オッズ対応版）

kekka_20130105-20251228.csv の実際の払戻額を使って
正確な回収率を計算する。

払戻額の単位: 100円あたりの配当
  例）複勝120 → 100円購入で120円払戻（回収率120%）
  実際の払戻額 = 購入額 × (払戻配当 / 100)

Usage:
    python backtest.py
    python backtest.py --n_races 100
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from catboost import Pool
from tqdm import tqdm

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
KEKKA_CSV        = DATA_DIR  / "kekka_20130105-20251228.csv"
LGBM_MODEL_PATH  = MODEL_DIR / "lgbm_optuna_v1.pkl"
CAT_MODEL_PATH   = MODEL_DIR / "catboost_optuna_v1.pkl"
TORCH_MODEL_PATH = MODEL_DIR / "transformer_optuna_v1.pkl"
TARGET      = "fukusho_flag"
COL_RACE_ID = "レースID(新/馬番無)"
COL_RANK    = "着順"
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
# 結果CSV読み込み・払戻辞書構築
# =========================================================
def load_kekka(kekka_path: Path) -> dict[str, dict]:
    logger.info(f"kekka CSV読み込み: {kekka_path}")
    df = pd.read_csv(kekka_path, encoding="cp932", low_memory=False)
    logger.info(f"  {len(df):,}行 × {len(df.columns)}列")

    df["race_id"]   = df["レースID(新)"].astype(str).str[:16]
    df["確定着順"]   = pd.to_numeric(df["確定着順"], errors="coerce")
    df["馬番"]       = pd.to_numeric(df["馬番"],     errors="coerce")

    kekka_dict: dict[str, dict] = {}

    for race_id, group in df.groupby("race_id"):
        group = group.sort_values("確定着順")

        entry: dict = {
            "複勝": {}, "馬連": {}, "馬単": {}, "三連複": {}, "三連単": {}
        }

        # 複勝（1〜3着馬番→配当）
        for _, row in group[group["確定着順"] <= 3].iterrows():
            ban = row["馬番"]
            pay = row["複勝配当"]
            if pd.notna(ban) and pd.notna(pay):
                entry["複勝"][int(ban)] = int(pay)

        # 上位3頭
        top3_rows = group[group["確定着順"] <= 3].sort_values("確定着順")
        top3 = [
            int(h) for h in top3_rows["馬番"].tolist()
            if pd.notna(h)
        ]

        # 1着行から組み合わせ系を取得
        rank1 = group[group["確定着順"] == 1]
        if rank1.empty:
            kekka_dict[str(race_id)] = entry
            continue
        r1 = rank1.iloc[0]

        if len(top3) >= 2:
            # 馬連
            key = "-".join(map(str, sorted(top3[:2])))
            pay = r1["馬連"]
            if pd.notna(pay):
                entry["馬連"][key] = int(pay)
            # 馬単
            key = f"{top3[0]}-{top3[1]}"
            pay = r1["馬単"]
            if pd.notna(pay):
                entry["馬単"][key] = int(pay)

        if len(top3) >= 3:
            # 三連複
            key = "-".join(map(str, sorted(top3[:3])))
            pay = r1["３連複"]
            if pd.notna(pay):
                entry["三連複"][key] = int(pay)
            # 三連単
            key = f"{top3[0]}-{top3[1]}-{top3[2]}"
            pay = r1["３連単"]
            if pd.notna(pay):
                entry["三連単"][key] = int(pay)

        kekka_dict[str(race_id)] = entry

    logger.info(f"払戻辞書構築完了: {len(kekka_dict):,}レース")
    return kekka_dict


# =========================================================
# 的中判定＋実払戻取得
# =========================================================
def get_actual_payout(
    combo: list[int],
    ordered: bool,
    bet_type: str,
    kekka_entry: dict,
) -> int:
    """
    実際の払戻配当（100円あたり）を返す。
    的中しない場合は0を返す。
    """
    if bet_type == "複勝":
        h = combo[0]
        pay = kekka_entry["複勝"].get(h, 0)
        return int(pay) if pay else 0

    elif bet_type == "馬連":
        key = "-".join(map(str, sorted(combo)))
        pay = kekka_entry["馬連"].get(key, 0)
        return int(pay) if pay else 0

    elif bet_type == "馬単":
        key = f"{combo[0]}-{combo[1]}"
        pay = kekka_entry["馬単"].get(key, 0)
        return int(pay) if pay else 0

    elif bet_type == "三連複":
        key = "-".join(map(str, sorted(combo)))
        pay = kekka_entry["三連複"].get(key, 0)
        return int(pay) if pay else 0

    elif bet_type == "三連単":
        # 着順通りのキー: 1着-2着-3着
        key = f"{combo[0]}-{combo[1]}-{combo[2]}"
        pay = kekka_entry["三連単"].get(key, 0)
        return int(pay) if pay else 0

    return 0


# =========================================================
# 前処理ユーティリティ
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


# =========================================================
# 各モデル予測
# =========================================================
def predict_lgbm_batch(df: pd.DataFrame) -> np.ndarray:
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


def predict_catboost_batch(df: pd.DataFrame) -> np.ndarray:
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


def predict_transformer_batch(df: pd.DataFrame) -> np.ndarray:
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
    loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)

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


def ensemble_predict_batch(df: pd.DataFrame) -> np.ndarray:
    logger.info("LightGBM予測中...")
    p_lgbm  = predict_lgbm_batch(df)
    logger.info("CatBoost予測中...")
    p_cat   = predict_catboost_batch(df)
    logger.info("Transformer予測中...")
    p_torch = predict_transformer_batch(df)
    w     = MODEL_WEIGHTS
    total = sum(w.values())
    return (
        (w["lgbm"]        / total) * p_lgbm  +
        (w["catboost"]    / total) * p_cat   +
        (w["transformer"] / total) * p_torch
    )


def assign_marks_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["mark"] = ""
    for _, group in df.groupby(COL_RACE_ID, sort=False):
        ranked = group["prob"].rank(ascending=False, method="first")
        marks  = {1: "◎", 2: "◯", 3: "▲", 4: "△", 5: "×"}
        for idx, rank in ranked.items():
            if rank <= 5:
                df.at[idx, "mark"] = marks[int(rank)]
    return df


# =========================================================
# 1レース分の処理
# =========================================================
def process_one_race(
    race_df: pd.DataFrame,
    kekka_entry: dict,
    budget: int = BUDGET,
) -> list[dict]:
    import itertools

    prob_series = race_df.set_index("馬番")["prob"]
    mark_dict   = race_df.set_index("馬番")["mark"].to_dict()

    hon    = race_df[race_df["mark"] == "◎"]["馬番"].tolist()
    taikou = race_df[race_df["mark"] == "◯"]["馬番"].tolist()
    sabo   = race_df[race_df["mark"] == "▲"]["馬番"].tolist()
    top3   = hon + taikou + sabo

    def make_bet(combo, bet_type, ordered):
        p    = min(calc_win_prob_pl(list(combo), prob_series, ordered), 0.99)
        odds = estimate_odds(p, bet_type)
        sep  = "→" if ordered else "-"
        return {
            "買い目":      sep.join(map(str, combo)),
            "combo":       list(combo),
            "ordered":     ordered,
            "bet_type":    bet_type,
            "推定的中確率": round(p, 4),
            "推定オッズ":   odds,
            "推定期待値":   round(p * odds, 3),
        }

    candidates: dict[str, list[dict]] = {
        "複勝": [], "馬連": [], "三連複": [], "三連単": [],
    }

    for h in hon + taikou:
        candidates["複勝"].append(make_bet([h], "複勝", False))
    if hon:
        for h in taikou + sabo:
            candidates["馬連"].append(
                make_bet(tuple(sorted([hon[0], h])), "馬連", False)
            )
    if len(top3) >= 3:
        rows = [make_bet(c, "三連複", False) for c in itertools.combinations(top3[:4], 3)]
        candidates["三連複"] = sorted(rows, key=lambda x: x["推定期待値"], reverse=True)[:3]
        rows = [make_bet(p, "三連単", True) for p in itertools.permutations(top3[:3], 3)]
        candidates["三連単"] = sorted(rows, key=lambda x: x["推定期待値"], reverse=True)[:3]

    # 予算按分
    active      = {k: v for k, v in candidates.items() if len(v) > 0}
    total_ratio = sum(BUDGET_RATIO[k] for k in active)
    for bet_type, bets in active.items():
        n       = len(bets)
        alloc   = floor_to_unit(int(budget * BUDGET_RATIO[bet_type] / total_ratio))
        per_bet = max(floor_to_unit(alloc // n), MIN_UNIT)
        while per_bet * n > alloc and per_bet > MIN_UNIT:
            per_bet -= MIN_UNIT
        for b in bets:
            b["購入額"] = per_bet

    total_used = sum(b["購入額"] for bets in active.values() for b in bets)
    remainder  = floor_to_unit(budget - int(total_used))
    if remainder >= MIN_UNIT and "三連複" in active and active["三連複"]:
        n     = len(active["三連複"])
        extra = floor_to_unit(remainder // n)
        if extra >= MIN_UNIT:
            for b in active["三連複"]:
                b["購入額"] += extra

    # 的中判定（実払戻）
    results  = []
    race_id  = race_df[COL_RACE_ID].iloc[0]
    race_info = race_df.iloc[0]

    for bet_type, bets in active.items():
        for b in bets:
            payout_per100 = get_actual_payout(
                b["combo"], b["ordered"], bet_type, kekka_entry
            )
            hit      = payout_per100 > 0
            # 実払戻額 = 購入額 × (実配当 / 100)
            actual_pay = int(b["購入額"] * payout_per100 / 100) if hit else 0

            results.append({
                "race_id":     race_id,
                "日付":        race_info.get("日付", ""),
                "場所":        race_info.get("場所", ""),
                "距離":        race_info.get("距離", ""),
                "芝ダ":        race_info.get("芝・ダ", ""),
                "クラス":      race_info.get("クラス名", ""),
                "馬券種":      bet_type,
                "買い目":      b["買い目"],
                "推定的中確率": b["推定的中確率"],
                "推定オッズ":   b["推定オッズ"],
                "推定期待値":   b["推定期待値"],
                "購入額":      b["購入額"],
                "実配当(100円)": payout_per100,
                "実オッズ":    round(payout_per100 / 100, 1) if payout_per100 else 0,
                "的中":        int(hit),
                "実払戻額":    actual_pay,
                "収支":        actual_pay - b["購入額"],
            })

    return results


# =========================================================
# 集計・可視化
# =========================================================
def summarize(df: pd.DataFrame) -> None:
    n_races = df["race_id"].nunique()

    print("\n" + "=" * 70)
    print(f"バックテスト結果サマリ（実オッズ版）  {n_races:,}レース")
    print("=" * 70)

    total_cost = df["購入額"].sum()
    total_pay  = df["実払戻額"].sum()
    total_net  = total_pay - total_cost
    roi        = total_pay / total_cost * 100 if total_cost > 0 else 0

    print(f"\n【全体】")
    print(f"  総投資額  : {total_cost:>12,}円")
    print(f"  総払戻額  : {total_pay:>12,}円")
    print(f"  純収支    : {total_net:>+12,}円")
    print(f"  回収率    : {roi:>11.1f}%")

    print(f"\n【馬券種別】")
    print(f"{'馬券種':<6} {'投資':>10} {'払戻':>10} {'収支':>10} {'回収率':>8} {'的中率':>8} {'的中数':>6} {'点数':>6}")
    print("-" * 70)

    for bet_type in ["複勝", "馬連", "三連複", "三連単"]:
        sub = df[df["馬券種"] == bet_type]
        if sub.empty:
            continue
        cost   = sub["購入額"].sum()
        pay    = sub["実払戻額"].sum()
        net    = pay - cost
        r      = pay / cost * 100 if cost > 0 else 0
        hits   = sub["的中"].sum()
        total_b = len(sub)
        hit_r  = hits / total_b * 100 if total_b > 0 else 0
        print(
            f"{bet_type:<6} {cost:>10,} {pay:>10,} {net:>+10,} "
            f"{r:>7.1f}% {hit_r:>7.1f}% {hits:>6,} {total_b:>6,}"
        )


def plot_cumulative(df: pd.DataFrame, save_path: Path) -> None:
    race_pnl = (
        df.groupby("race_id")
        .agg(収支=("収支", "sum"), 投資=("購入額", "sum"))
        .reset_index()
    )
    race_pnl["累積収支"]     = race_pnl["収支"].cumsum()
    race_pnl["累積投資"]     = race_pnl["投資"].cumsum()
    race_pnl["累積回収率(%)"] = (
        (race_pnl["累積投資"] + race_pnl["累積収支"]) /
        race_pnl["累積投資"] * 100
    )

    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    axes[0].plot(race_pnl.index, race_pnl["累積収支"], color="steelblue")
    axes[0].axhline(0, color="gray", linewidth=0.8, linestyle="--")
    axes[0].set_title("累積収支推移（実オッズ）")
    axes[0].set_ylabel("累積収支（円）")
    axes[0].set_xlabel("レース数")

    axes[1].plot(race_pnl.index, race_pnl["累積回収率(%)"], color="tomato")
    axes[1].axhline(100, color="gray", linewidth=0.8, linestyle="--")
    axes[1].set_title("累積回収率推移（実オッズ）")
    axes[1].set_ylabel("累積回収率（%）")
    axes[1].set_xlabel("レース数")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"グラフ保存: {save_path}")


def plot_roi_by_category(df: pd.DataFrame, save_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, col, title in zip(
        axes,
        ["場所", "クラス", "芝ダ"],
        ["競馬場別回収率", "クラス別回収率", "芝ダート別回収率"],
    ):
        grp = df.groupby(col).agg(
            投資=("購入額", "sum"),
            払戻=("実払戻額", "sum"),
        ).reset_index()
        grp["回収率"] = grp["払戻"] / grp["投資"] * 100
        grp = grp.sort_values("回収率", ascending=True)
        colors = ["tomato" if v >= 100 else "steelblue" for v in grp["回収率"]]
        ax.barh(grp[col], grp["回収率"], color=colors)
        ax.axvline(100, color="gray", linewidth=0.8, linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("回収率（%）")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"グラフ保存: {save_path}")


# =========================================================
# main
# =========================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="PyCaLiAI バックテスト（実オッズ版）")
    parser.add_argument("--n_races",       type=int, default=None)
    parser.add_argument("--budget",        type=int, default=BUDGET)
    parser.add_argument("--output_suffix", type=str, default="",
                        help="出力ファイル名のサフィックス（例: _train）")
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # データロード
    logger.info(f"マスターCSV読み込み: {MASTER_CSV}")
    df      = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    df["日付"] = pd.to_numeric(df["日付"], errors="coerce")
    if args.output_suffix == "_train":
        test_df = df[df["日付"] < 20230101].copy().reset_index(drop=True)
        logger.info("対象期間: 2013〜2022年（発見期）")
    else:
        test_df = df[df["split"] == "test"].copy().reset_index(drop=True)
        logger.info("対象期間: テストデータ（2024年）")
    logger.info(f"テストデータ: {len(test_df):,}行")

    # 払戻辞書ロード
    kekka_dict = load_kekka(KEKKA_CSV)

    # アンサンブル予測
    logger.info("アンサンブル予測開始...")
    test_df["prob"] = ensemble_predict_batch(test_df)

    # 印付与
    logger.info("印付与中...")
    test_df = assign_marks_df(test_df)

    # レース単位処理
    race_ids = test_df[COL_RACE_ID].unique()
    if args.n_races:
        race_ids = race_ids[:args.n_races]
    logger.info(f"バックテスト開始: {len(race_ids):,}レース")

    all_results = []
    skipped     = 0
    for race_id in tqdm(race_ids, desc="バックテスト"):
        race_df     = test_df[test_df[COL_RACE_ID] == race_id].copy()
        kekka_entry = kekka_dict.get(str(race_id), {
            "複勝": {}, "馬連": {}, "馬単": {}, "三連複": {}, "三連単": {}
        })
        try:
            results = process_one_race(race_df, kekka_entry, budget=args.budget)
            all_results.extend(results)
        except Exception as e:
            logger.warning(f"レース {race_id} スキップ: {e}")
            skipped += 1

    logger.info(f"スキップ: {skipped}レース")

    if not all_results:
        logger.error("結果が0件です。")
        return

    result_df = pd.DataFrame(all_results)

    # 集計
    summarize(result_df)

    # グラフ


    # CSV保存
    out_path = REPORT_DIR / f"backtest_results{args.output_suffix}.csv"
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(f"結果保存: {out_path}")

    print(f"\n保存ファイル:")
    print(f"  {out_path}")
    plot_cumulative(result_df, REPORT_DIR / f"backtest_cumulative{args.output_suffix}.png")
    plot_roi_by_category(result_df, REPORT_DIR / f"backtest_roi_by_category{args.output_suffix}.png")


if __name__ == "__main__":
    main()