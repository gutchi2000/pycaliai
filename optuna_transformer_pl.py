"""
optuna_transformer_pl.py
PyCaLiAI - Transformer + Plackett-Luce ランキング損失

BCEWithLogitsLoss（二値分類）の代わりに Plackett-Luce 損失を使い、
レース内全頭の順位確率を直接最適化する。

PL損失: L = -sum_{i=1}^{N-1} [ s_{(i)} - logsumexp(s_{(i)}, ..., s_{(N)}) ]
  s_{(i)} は着順 i 番目の馬のスコア（昇順ソート後）

Usage:
    python optuna_transformer_pl.py
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import joblib
import optuna
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR   = Path(r"E:\PyCaLiAI")
DATA_DIR   = BASE_DIR / "data"
MODEL_DIR  = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MASTER_CSV = DATA_DIR  / "master_20130105-20251228.csv"
MODEL_PATH = MODEL_DIR / "transformer_pl_v2.pkl"
STUDY_PATH = REPORT_DIR / "optuna_transformer_pl_study.pkl"

TARGET       = "fukusho_flag"   # AUC評価用
COL_ORDER    = "着順"           # Plackett-Luce損失用
COL_RACE_ID  = "レースID(新/馬番無)"
RANDOM_STATE = 42
N_TRIALS     = 30
MAX_HORSES   = 18

torch.manual_seed(RANDOM_STATE)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"使用デバイス: {DEVICE}")

# optuna_transformer.py と同じ特徴量
CAT_FEATURES = [
    "芝・ダ", "コース区分", "芝(内・外)", "馬場状態", "天気",
    "クラス名", "場所", "性別", "斤量", "ブリンカー",
    "前走場所", "前芝・ダ", "前走馬場状態", "前走斤量", "前好走",
    "重量種別", "年齢限定",
]

NUM_FEATURES = [
    "距離", "トラックコード(JV)", "出走頭数", "フルゲート頭数",
    "枠番", "馬番", "年齢", "馬齢斤量差", "斤量体重比",
    "間隔", "休み明け～戦目", "騎手年齢", "調教師年齢",
    "前走確定着順", "前距離", "前走出走頭数",
    "前走馬体重", "前走馬体重増減",
    "前1角", "前2角", "前3角", "前4角",
    "前走上り3F", "前走上り3F順",
    "前走Ave-3F", "前PCI", "前走PCI3", "前走RPCI",
    "前走平均1Fタイム",
]

TIME_STR_FEATURES = ["前走走破タイム", "前走着差タイム"]

from utils import parse_time_str, backup_model
from train_transformer import RaceTransformer


# =========================================================
# 前処理（optuna_transformer.py と同じ）
# =========================================================
def preprocess(
    df: pd.DataFrame,
    encoders: dict[str, LabelEncoder] | None = None,
    fit: bool = True,
    num_stats: dict | None = None,
) -> tuple[pd.DataFrame, dict, dict]:
    df = df.copy()
    for col in TIME_STR_FEATURES:
        if col in df.columns:
            df[col] = parse_time_str(df[col])
    if encoders is None:
        encoders = {}
    for col in CAT_FEATURES:
        if col not in df.columns:
            df[col] = 0
            continue
        df[col] = df[col].fillna("__NaN__").astype(str)
        if fit:
            le   = LabelEncoder()
            vals = df[col].tolist()
            if "__NaN__" not in vals:
                vals.append("__NaN__")
            le.fit(vals)
            encoders[col] = le
        else:
            le    = encoders[col]
            known = set(le.classes_)
            df[col] = df[col].apply(lambda x: x if x in known else "__NaN__")
        df[col] = le.transform(df[col]) + 1
    if num_stats is None:
        num_stats = {}
    for col in NUM_FEATURES + TIME_STR_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if fit:
            mean = df[col].mean()
            std  = df[col].std() + 1e-8
            num_stats[col] = {"mean": mean, "std": std}
        else:
            mean = num_stats[col]["mean"]
            std  = num_stats[col]["std"]
        df[col] = (df[col] - mean) / std
        df[col] = df[col].fillna(0.0)
    return df, encoders, num_stats


# =========================================================
# データセット（着順を追加）
# =========================================================
class RaceDatasetPL(Dataset):
    """
    Plackett-Luce損失用 Dataset。
    targets: fukusho_flag（AUC評価用）
    order: 着順（PL損失用、0=不明/除外）
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cat_cols: list[str],
        num_cols: list[str],
        cat_vocab_sizes: dict[str, int],
    ) -> None:
        self.races = []
        all_num = num_cols

        for race_id, group in df.groupby(COL_RACE_ID, sort=False):
            n = len(group)
            if n < 2:
                continue

            cat_data = torch.zeros(MAX_HORSES, len(cat_cols), dtype=torch.long)
            num_data = torch.zeros(MAX_HORSES, len(all_num), dtype=torch.float32)
            targets  = torch.zeros(MAX_HORSES, dtype=torch.float32)
            order    = torch.zeros(MAX_HORSES, dtype=torch.float32)  # 着順
            mask     = torch.ones(MAX_HORSES, dtype=torch.bool)

            n_valid = min(n, MAX_HORSES)
            for i, (_, row) in enumerate(group.iterrows()):
                if i >= MAX_HORSES:
                    break
                cat_data[i] = torch.tensor(
                    [int(row.get(c, 0)) for c in cat_cols], dtype=torch.long
                )
                num_data[i] = torch.tensor(
                    [float(row.get(c, 0.0)) for c in all_num], dtype=torch.float32
                )
                targets[i] = float(row[TARGET])
                # 着順: 数値変換できなければ 0（PL損失から除外）
                try:
                    o = float(row[COL_ORDER])
                    order[i] = o if o >= 1 else 0.0
                except (ValueError, TypeError, KeyError):
                    order[i] = 0.0
                mask[i] = False

            self.races.append({
                "cat":     cat_data,
                "num":     num_data,
                "targets": targets,
                "order":   order,
                "mask":    mask,
                "n_valid": n_valid,
            })

    def __len__(self) -> int:
        return len(self.races)

    def __getitem__(self, idx: int) -> dict:
        return self.races[idx]


# =========================================================
# Plackett-Luce 損失
# =========================================================
def plackett_luce_loss(
    scores: torch.Tensor,
    order: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    scores: (batch, MAX_HORSES) モデル出力ロジット
    order:  (batch, MAX_HORSES) 着順（1=1位、0=除外）
    mask:   (batch, MAX_HORSES) True=パディング

    PL損失 = -sum_{i=1}^{N-1} [ s_{(i)} - logsumexp(s_{(i)}, ..., s_{(N)}) ]
    着順不明（order==0）の馬は損失計算から除外。
    """
    total_loss = scores.new_zeros(1)
    count = 0

    for b in range(scores.size(0)):
        valid = ~mask[b]                    # (MAX_HORSES,) 有効馬フラグ
        s = scores[b][valid]                # (n,)
        o = order[b][valid]                 # (n,) 着順

        # 着順が判明している馬のみ
        known = o > 0
        s = s[known]
        o = o[known]
        if len(s) < 2:
            continue

        # 着順昇順ソート（1位 → 最後）
        sort_idx = torch.argsort(o)
        s_sorted = s[sort_idx]              # (n,)

        # PL損失: logcumsumexp を末尾から計算
        n = s_sorted.size(0)
        rev_s    = torch.flip(s_sorted, [0])
        rev_lcse = torch.logcumsumexp(rev_s, dim=0)
        lcse     = torch.flip(rev_lcse, [0])        # (n,)

        pl = -torch.sum(s_sorted[:-1] - lcse[:-1])
        total_loss = total_loss + pl
        count += 1

    return total_loss / max(count, 1)


# =========================================================
# 訓練・評価ループ
# =========================================================
def train_one_epoch_pl(
    model: RaceTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()
    total = 0.0
    n     = 0
    for batch in loader:
        cat    = batch["cat"].to(DEVICE)
        num    = batch["num"].to(DEVICE)
        order  = batch["order"].to(DEVICE)
        mask   = batch["mask"].to(DEVICE)
        optimizer.zero_grad()
        scores = model(cat, num, mask)       # (batch, MAX_HORSES)
        loss   = plackett_luce_loss(scores, order, mask)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
        n     += 1
    return total / max(n, 1)


@torch.no_grad()
def get_val_auc(
    model: RaceTransformer,
    loader: DataLoader,
) -> float:
    """y_true をローダー内の targets から取得（順序不一致バグ回避）"""
    model.eval()
    all_scores  = []
    all_targets = []
    for batch in loader:
        cat   = batch["cat"].to(DEVICE)
        num   = batch["num"].to(DEVICE)
        mask  = batch["mask"].to(DEVICE)
        out   = model(cat, num, mask)        # (batch, MAX_HORSES)
        valid = ~batch["mask"]               # True=有効馬
        all_scores.append(out.cpu()[valid].numpy())
        all_targets.append(batch["targets"][valid].numpy())
    scores  = np.concatenate(all_scores)
    targets = np.concatenate(all_targets)
    return roc_auc_score(targets, scores)


def make_loader(
    df: pd.DataFrame,
    cat_cols: list[str],
    num_cols: list[str],
    cat_vocab_sizes: dict[str, int],
    shuffle: bool = False,
    batch_size: int = 256,
) -> DataLoader:
    ds = RaceDatasetPL(df, cat_cols, num_cols, cat_vocab_sizes)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


# =========================================================
# メイン
# =========================================================
def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("データ読み込み・前処理...")
    df       = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)

    # 着順を数値化
    df[COL_ORDER] = pd.to_numeric(df[COL_ORDER], errors="coerce")

    train_df = df[df["split"] == "train"].copy()
    valid_df = df[df["split"] == "valid"].copy()
    test_df  = df[df["split"] == "test"].copy()
    logger.info(f"分割: train={len(train_df):,} / valid={len(valid_df):,} / test={len(test_df):,}")

    train_df, encoders, num_stats = preprocess(train_df, fit=True)
    valid_df, _,        _         = preprocess(valid_df, encoders, fit=False, num_stats=num_stats)
    test_df,  _,        _         = preprocess(test_df,  encoders, fit=False, num_stats=num_stats)

    cat_vocab_sizes = {col: len(encoders[col].classes_) for col in CAT_FEATURES if col in encoders}
    num_cols = [c for c in NUM_FEATURES + TIME_STR_FEATURES if c in train_df.columns]
    cat_cols = [c for c in CAT_FEATURES if c in train_df.columns]

    logger.info("Dataset構築中...")
    train_loader = make_loader(train_df, cat_cols, num_cols, cat_vocab_sizes, shuffle=True)
    valid_loader = make_loader(valid_df, cat_cols, num_cols, cat_vocab_sizes)
    test_loader  = make_loader(test_df,  cat_cols, num_cols, cat_vocab_sizes)

    # =========================================================
    # Optuna
    # =========================================================
    def objective(trial: optuna.Trial) -> float:
        d_model  = trial.suggest_categorical("d_model",  [64, 128, 256])
        n_heads  = trial.suggest_categorical("n_heads",  [2, 4, 8])
        n_layers = trial.suggest_int("n_layers", 1, 4)
        d_ff     = trial.suggest_categorical("d_ff",     [128, 256, 512])
        dropout  = trial.suggest_float("dropout", 0.0, 0.3)
        lr       = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

        if d_model % n_heads != 0:
            raise optuna.TrialPruned()

        model = RaceTransformer(
            cat_vocab_sizes=cat_vocab_sizes,
            cat_cols=cat_cols,
            n_num=len(num_cols),
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
        ).to(DEVICE)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        best_auc = 0.0
        patience = 0
        for epoch in range(10):
            train_one_epoch_pl(model, train_loader, optimizer)
            auc = get_val_auc(model, valid_loader)
            if auc > best_auc:
                best_auc = auc
                patience = 0
            else:
                patience += 1
                if patience >= 3:
                    break
            trial.report(auc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return best_auc

    logger.info(f"Optuna開始: {N_TRIALS}試行（Plackett-Luce損失、評価=fukusho AUC）")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    logger.info(f"最適パラメータ: {study.best_params}")
    logger.info(f"Best Valid AUC: {study.best_value:.4f}")

    # 最適パラメータで再学習（フルエポック）
    logger.info("最適パラメータで再学習中...")
    bp = study.best_params
    if bp["d_model"] % bp["n_heads"] != 0:
        bp["n_heads"] = 4

    best_model = RaceTransformer(
        cat_vocab_sizes=cat_vocab_sizes,
        cat_cols=cat_cols,
        n_num=len(num_cols),
        d_model=bp["d_model"],
        n_heads=bp["n_heads"],
        n_layers=bp["n_layers"],
        d_ff=bp["d_ff"],
        dropout=bp["dropout"],
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        best_model.parameters(), lr=bp["lr"], weight_decay=1e-4
    )

    best_auc   = 0.0
    patience   = 0
    best_state = None
    for epoch in range(100):
        loss = train_one_epoch_pl(best_model, train_loader, optimizer)
        auc  = get_val_auc(best_model, valid_loader)
        logger.info(f"  Epoch {epoch+1:3d}  loss={loss:.4f}  valid_auc={auc:.4f}")
        if auc > best_auc:
            best_auc   = auc
            patience   = 0
            best_state = {k: v.cpu().clone() for k, v in best_model.state_dict().items()}
        else:
            patience += 1
            if patience >= 10:
                logger.info(f"  Early stopping at epoch {epoch+1}")
                break

    best_model.load_state_dict(best_state)
    best_model.to(DEVICE)

    auc_va = get_val_auc(best_model, valid_loader)
    auc_te = get_val_auc(best_model, test_loader)

    logger.info(f"[Valid] AUC={auc_va:.4f}  (BCE Transformer比較: optuna_transformer参照)")
    logger.info(f"[Test]  AUC={auc_te:.4f}")

    backup_model(MODEL_PATH)
    model_config = {
        "cat_vocab_sizes": cat_vocab_sizes,
        "cat_cols":        cat_cols,
        "n_num":           len(num_cols),
        "d_model":         bp["d_model"],
        "n_heads":         bp["n_heads"],
        "n_layers":        bp["n_layers"],
        "d_ff":            bp["d_ff"],
        "dropout":         bp["dropout"],
    }
    joblib.dump(
        {
            "model_state":  best_state,
            "model_config": model_config,
            "encoders":     encoders,
            "num_stats":    num_stats,
            "num_cols":     num_cols,
            "cat_cols":     cat_cols,
        },
        MODEL_PATH,
    )
    joblib.dump(study, STUDY_PATH)
    logger.info(f"モデル保存: {MODEL_PATH}")

    print("\n" + "=" * 50)
    print("Transformer Plackett-Luce 最適化完了サマリ")
    print("=" * 50)
    print(f"Valid AUC : {auc_va:.4f}")
    print(f"Test  AUC : {auc_te:.4f}")
    print(f"Best試行  : Trial {study.best_trial.number}")
    print(f"\n最適パラメータ:")
    for k, v in bp.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
