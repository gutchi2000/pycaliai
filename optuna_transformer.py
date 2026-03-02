"""
optuna_transformer.py
PyCaLiAI - Transformer Optuna ハイパーパラメータ最適化

Usage:
    python optuna_transformer.py
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
from torch.utils.data import DataLoader
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
MODEL_PATH = MODEL_DIR / "transformer_optuna_v1.pkl"
STUDY_PATH = REPORT_DIR / "optuna_transformer_study.pkl"

TARGET       = "fukusho_flag"
COL_RACE_ID  = "レースID(新/馬番無)"
RANDOM_STATE = 42
N_TRIALS     = 30   # TransformerはGPUでも時間かかるので30試行
MAX_HORSES   = 18

torch.manual_seed(RANDOM_STATE)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"使用デバイス: {DEVICE}")

# train_transformer.pyから流用
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


# train_transformer.pyと同じDataset・モデル定義を流用
from train_transformer import RaceDataset, RaceTransformer


def make_loader(
    df: pd.DataFrame,
    cat_cols: list[str],
    num_cols: list[str],
    cat_vocab_sizes: dict[str, int],
    shuffle: bool = False,
    batch_size: int = 256,
) -> DataLoader:
    ds = RaceDataset(df, cat_cols, num_cols, cat_vocab_sizes)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


@torch.no_grad()
def get_val_auc(
    model: RaceTransformer,
    loader: DataLoader,
    y_true: np.ndarray,
) -> float:
    model.eval()
    all_proba = []
    for batch in loader:
        cat    = batch["cat"].to(DEVICE)
        num    = batch["num"].to(DEVICE)
        mask   = batch["mask"].to(DEVICE)
        logits = model(cat, num, mask)
        proba  = torch.sigmoid(logits).cpu()
        valid  = ~batch["mask"]
        all_proba.append(proba[valid].numpy())
    proba = np.concatenate(all_proba)
    return roc_auc_score(y_true[:len(proba)], proba[:len(y_true)])


def train_one_epoch(
    model: RaceTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    model.train()
    total = 0.0
    n     = 0
    for batch in loader:
        cat     = batch["cat"].to(DEVICE)
        num     = batch["num"].to(DEVICE)
        targets = batch["targets"].to(DEVICE)
        mask    = batch["mask"].to(DEVICE)
        optimizer.zero_grad()
        logits = model(cat, num, mask)
        valid  = ~mask
        loss   = criterion(logits[valid], targets[valid])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
        n     += 1
    return total / max(n, 1)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("データ読み込み・前処理...")
    df       = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    train_df = df[df["split"] == "train"].copy()
    valid_df = df[df["split"] == "valid"].copy()
    test_df  = df[df["split"] == "test"].copy()

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

    y_valid = valid_df[TARGET].astype(int).values
    y_test  = test_df[TARGET].astype(int).values

    pos_weight = torch.tensor([3.68], device=DEVICE)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def objective(trial: optuna.Trial) -> float:
        d_model  = trial.suggest_categorical("d_model",  [64, 128, 256])
        n_heads  = trial.suggest_categorical("n_heads",  [2, 4, 8])
        n_layers = trial.suggest_int("n_layers", 1, 4)
        d_ff     = trial.suggest_categorical("d_ff",     [128, 256, 512])
        dropout  = trial.suggest_float("dropout", 0.0, 0.3)
        lr       = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

        # n_headsとd_modelの整合性チェック
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
        for epoch in range(20):   # 探索時は最大20 Epoch
            train_one_epoch(model, train_loader, optimizer, criterion)
            auc = get_val_auc(model, valid_loader, y_valid)
            if auc > best_auc:
                best_auc = auc
                patience = 0
            else:
                patience += 1
                if patience >= 5:
                    break
            trial.report(auc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return best_auc

    logger.info(f"Optuna開始: {N_TRIALS}試行")
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
        bp["n_heads"] = 4  # フォールバック

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

    optimizer = torch.optim.AdamW(best_model.parameters(), lr=bp["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    best_auc     = 0.0
    patience_cnt = 0
    for epoch in range(1, 51):
        train_one_epoch(best_model, train_loader, optimizer, criterion)
        scheduler.step()
        auc = get_val_auc(best_model, valid_loader, y_valid)
        logger.info(f"Epoch {epoch:3d}/50  val_auc={auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            torch.save(best_model.state_dict(), MODEL_DIR / "transformer_optuna_best.pt")
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= 7:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    best_model.load_state_dict(
        torch.load(MODEL_DIR / "transformer_optuna_best.pt", weights_only=True)
    )

    auc_va = get_val_auc(best_model, valid_loader, y_valid)
    auc_te = get_val_auc(best_model, test_loader,  y_test)
    logger.info(f"[Valid] AUC={auc_va:.4f}  (旧: 0.7475)")
    logger.info(f"[Test]  AUC={auc_te:.4f}  (旧: 0.7540)")

    joblib.dump({
        "model_state":  best_model.state_dict(),
        "model_config": {
            "cat_vocab_sizes": cat_vocab_sizes,
            "cat_cols":        cat_cols,
            "n_num":           len(num_cols),
            "d_model":         bp["d_model"],
            "n_heads":         bp["n_heads"],
            "n_layers":        bp["n_layers"],
            "d_ff":            bp["d_ff"],
            "dropout":         bp["dropout"],
        },
        "encoders":  encoders,
        "num_stats": num_stats,
        "num_cols":  num_cols,
        "cat_cols":  cat_cols,
    }, MODEL_PATH)
    joblib.dump(study, STUDY_PATH)
    logger.info(f"モデル保存: {MODEL_PATH}")

    print("\n" + "=" * 50)
    print("Transformer Optuna 最適化完了サマリ")
    print("=" * 50)
    print(f"Valid AUC : {auc_va:.4f}  (旧: 0.7475)")
    print(f"Test  AUC : {auc_te:.4f}  (旧: 0.7540)")
    print(f"Best試行  : Trial {study.best_trial.number}")
    print(f"\n最適パラメータ:")
    for k, v in bp.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()