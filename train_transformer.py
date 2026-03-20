"""
train_transformer.py
PyCaLiAI - 同レース全頭Transformer 複勝内2値分類モデルの学習・評価

アーキテクチャ:
    1馬1行 → レースIDでグループ化 → padding → TransformerEncoder
    → 各馬の出力ベクトル → Linear → 複勝確率

Usage:
    python train_transformer.py
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    RocCurveDisplay,
    average_precision_score,
    roc_auc_score,
)
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
BASE_DIR   = Path(r"E:\PyCaLiAI")
DATA_DIR   = BASE_DIR / "data"
MODEL_DIR  = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MASTER_CSV  = DATA_DIR / "master_20130105-20251228.csv"
MODEL_PATH  = MODEL_DIR / "transformer_fukusho_v1.pkl"

TARGET       = "fukusho_flag"
COL_RACE_ID  = "レースID(新/馬番無)"
RANDOM_STATE = 42

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# =========================================================
# デバイス設定
# =========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"使用デバイス: {DEVICE}")

# =========================================================
# 特徴量定義（torch_transformer CSV由来）
# =========================================================

# カテゴリ列（Embedding対象）
CAT_FEATURES = [
    "芝・ダ", "コース区分", "芝(内・外)", "馬場状態", "天気",
    "クラス名", "場所",
    "性別", "斤量", "ブリンカー",
    "前走場所", "前芝・ダ", "前走馬場状態", "前走斤量", "前好走",
    "重量種別", "年齢限定",
]

# 数値列
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

# =========================================================
# ハイパーパラメータ
# =========================================================
MAX_HORSES   = 18       # 最大出走頭数
D_MODEL      = 128      # Transformer次元数
N_HEADS      = 4        # Attentionヘッド数
N_LAYERS     = 2        # Transformerレイヤー数
D_FF         = 256      # FFN次元数
DROPOUT      = 0.1
BATCH_SIZE   = 256      # レース単位のバッチ
MAX_EPOCHS   = 50
PATIENCE     = 7        # Early stopping
LR           = 1e-3


# =========================================================
# 前処理
# =========================================================
from utils import parse_time_str


def preprocess(
    df: pd.DataFrame,
    encoders: dict[str, LabelEncoder] | None = None,
    fit: bool = True,
    num_stats: dict | None = None,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder], dict]:
    """
    前処理:
    1. タイム文字列 → 数値
    2. カテゴリ → LabelEncoding（0=PAD用に+1オフセット）
    3. 数値 → 標準化（train統計量を保存してvalid/testに適用）
    """
    df = df.copy()

    # タイム変換
    for col in TIME_STR_FEATURES:
        if col in df.columns:
            df[col] = parse_time_str(df[col])

    # カテゴリ → LabelEncoding
    if encoders is None:
        encoders = {}
    for col in CAT_FEATURES:
        if col not in df.columns:
            df[col] = 0
            continue
        df[col] = df[col].fillna("__NaN__").astype(str)
        if fit:
            le = LabelEncoder()
            # __NaN__を必ずclassesに含める
            vals = df[col].tolist()
            if "__NaN__" not in vals:
                vals.append("__NaN__")
            le.fit(vals)
            encoders[col] = le
        else:
            le = encoders[col]
            known = set(le.classes_)
            df[col] = df[col].apply(lambda x: x if x in known else "__NaN__")
        # 0をPAD用に確保するため+1オフセット
        df[col] = le.transform(df[col]) + 1

    # 数値 → 標準化
    all_num = NUM_FEATURES + TIME_STR_FEATURES
    if num_stats is None:
        num_stats = {}
    for col in all_num:
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
        df[col] = df[col].fillna(0.0)  # 欠損は標準化後0埋め

    return df, encoders, num_stats


# =========================================================
# Dataset
# =========================================================
class RaceDataset(Dataset):
    """
    レース単位のデータセット。
    1レース = 最大18頭のシーケンス（padding付き）。
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
                continue  # 1頭レースはスキップ

            # カテゴリ特徴量テンソル [n_horses, n_cat]
            cat_data = torch.zeros(MAX_HORSES, len(cat_cols), dtype=torch.long)
            # 数値特徴量テンソル [n_horses, n_num]
            num_data = torch.zeros(MAX_HORSES, len(all_num), dtype=torch.float32)
            # ターゲット [n_horses]
            targets  = torch.zeros(MAX_HORSES, dtype=torch.float32)
            # パディングマスク（True=パディング）
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
                targets[i]  = float(row[TARGET])
                mask[i]     = False  # 有効な馬

            self.races.append({
                "cat":     cat_data,
                "num":     num_data,
                "targets": targets,
                "mask":    mask,
                "n_valid": n_valid,
            })

    def __len__(self) -> int:
        return len(self.races)

    def __getitem__(self, idx: int) -> dict:
        return self.races[idx]


# =========================================================
# モデル定義
# =========================================================
class HorseEmbedding(nn.Module):
    """カテゴリ特徴量をEmbeddingして数値特徴量と結合する。"""

    def __init__(
        self,
        cat_vocab_sizes: dict[str, int],
        cat_cols: list[str],
        n_num: int,
        d_model: int,
    ) -> None:
        super().__init__()
        self.cat_cols = cat_cols

        # 各カテゴリのEmbedding（次元数は vocab_size の 1/4 程度、最小4）
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(
                cat_vocab_sizes[col] + 2,  # +2: PAD(0) + unknown
                max(4, cat_vocab_sizes[col] // 4),
                padding_idx=0,
            )
            for col in cat_cols
        })

        emb_total = sum(
            max(4, cat_vocab_sizes[col] // 4) for col in cat_cols
        )
        input_dim = emb_total + n_num

        self.proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
        )

    def forward(self, cat: torch.Tensor, num: torch.Tensor) -> torch.Tensor:
        """
        cat: [batch, max_horses, n_cat]
        num: [batch, max_horses, n_num]
        → [batch, max_horses, d_model]
        """
        embs = []
        for i, col in enumerate(self.cat_cols):
            embs.append(self.embeddings[col](cat[:, :, i]))
        x = torch.cat(embs + [num], dim=-1)
        return self.proj(x)


class RaceTransformer(nn.Module):
    """同レース全頭をSelf-Attentionで処理する。"""

    def __init__(
        self,
        cat_vocab_sizes: dict[str, int],
        cat_cols: list[str],
        n_num: int,
        d_model: int   = D_MODEL,
        n_heads: int   = N_HEADS,
        n_layers: int  = N_LAYERS,
        d_ff: int      = D_FF,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()

        self.embedding = HorseEmbedding(
            cat_vocab_sizes, cat_cols, n_num, d_model
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        cat: torch.Tensor,
        num: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        cat:  [batch, max_horses, n_cat]
        num:  [batch, max_horses, n_num]
        mask: [batch, max_horses] True=padding
        → [batch, max_horses] 各馬の複勝確率ロジット
        """
        x = self.embedding(cat, num)                    # [B, H, d_model]
        x = self.transformer(x, src_key_padding_mask=mask)  # [B, H, d_model]
        logits = self.head(x).squeeze(-1)               # [B, H]
        return logits


# =========================================================
# 学習ループ
# =========================================================
def train_epoch(
    model: RaceTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for batch in loader:
        cat     = batch["cat"].to(DEVICE)
        num     = batch["num"].to(DEVICE)
        targets = batch["targets"].to(DEVICE)
        mask    = batch["mask"].to(DEVICE)

        optimizer.zero_grad()
        logits = model(cat, num, mask)          # [B, MAX_HORSES]

        # パディング部分を除いてlossを計算
        valid_mask = ~mask                      # True=有効
        loss = criterion(
            logits[valid_mask],
            targets[valid_mask],
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def predict_proba(
    model: RaceTransformer,
    loader: DataLoader,
) -> tuple[np.ndarray, np.ndarray]:
    """全データの予測確率とターゲットを返す。"""
    model.eval()
    all_proba   = []
    all_targets = []

    for batch in loader:
        cat     = batch["cat"].to(DEVICE)
        num     = batch["num"].to(DEVICE)
        targets = batch["targets"]
        mask    = batch["mask"].to(DEVICE)

        logits = model(cat, num, mask)
        proba  = torch.sigmoid(logits).cpu()

        valid_mask = ~batch["mask"]
        all_proba.append(proba[valid_mask].numpy())
        all_targets.append(targets[valid_mask].numpy())

    return np.concatenate(all_proba), np.concatenate(all_targets)


# =========================================================
# 評価
# =========================================================
def evaluate(
    model: RaceTransformer,
    loader: DataLoader,
    split_name: str,
) -> dict[str, float]:
    proba, y = predict_proba(model, loader)
    pred     = (proba >= 0.5).astype(int)

    auc    = roc_auc_score(y, proba)
    pr_auc = average_precision_score(y, proba)

    logger.info(f"[{split_name}] AUC={auc:.4f}  PR-AUC={pr_auc:.4f}")

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y, proba, ax=ax, name=split_name)
    ax.set_title(f"ROC曲線 [{split_name}] Transformer")
    path = REPORT_DIR / f"transformer_roc_{split_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"ROC曲線保存: {path}")

    return {"auc": auc, "pr_auc": pr_auc}


# =========================================================
# main
# =========================================================
def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ---- データロード ----
    logger.info(f"マスターCSV読み込み: {MASTER_CSV}")
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    logger.info(f"  {len(df):,}行 × {len(df.columns)}列")

    train_df = df[df["split"] == "train"].copy()
    valid_df = df[df["split"] == "valid"].copy()
    test_df  = df[df["split"] == "test"].copy()

    # ---- 前処理 ----
    logger.info("前処理開始...")
    train_df, encoders, num_stats = preprocess(train_df, fit=True)
    valid_df, _,        _         = preprocess(valid_df, encoders, fit=False, num_stats=num_stats)
    test_df,  _,        _         = preprocess(test_df,  encoders, fit=False, num_stats=num_stats)

    # カテゴリ語彙サイズ
    cat_vocab_sizes = {
        col: len(encoders[col].classes_)
        for col in CAT_FEATURES
        if col in encoders
    }

    # 使用する数値列
    num_cols = [
        c for c in NUM_FEATURES + TIME_STR_FEATURES
        if c in train_df.columns
    ]
    cat_cols = [c for c in CAT_FEATURES if c in train_df.columns]

    logger.info(f"カテゴリ特徴量: {len(cat_cols)}列  数値特徴量: {len(num_cols)}列")

    # ---- Dataset / DataLoader ----
    logger.info("Dataset構築中（レース単位グループ化）...")
    train_ds = RaceDataset(train_df, cat_cols, num_cols, cat_vocab_sizes)
    valid_ds = RaceDataset(valid_df, cat_cols, num_cols, cat_vocab_sizes)
    test_ds  = RaceDataset(test_df,  cat_cols, num_cols, cat_vocab_sizes)

    logger.info(
        f"レース数: train={len(train_ds):,} / valid={len(valid_ds):,} / test={len(test_ds):,}"
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ---- モデル構築 ----
    model = RaceTransformer(
        cat_vocab_sizes=cat_vocab_sizes,
        cat_cols=cat_cols,
        n_num=len(num_cols),
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"モデルパラメータ数: {total_params:,}")

    # クラス不均衡対応
    pos_weight = torch.tensor([3.68], device=DEVICE)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS
    )

    # ---- 学習ループ ----
    best_auc      = 0.0
    patience_cnt  = 0
    history       = []

    logger.info("学習開始...")
    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        scheduler.step()

        # Valid評価
        proba, y = predict_proba(model, valid_loader)
        val_auc  = roc_auc_score(y, proba)

        history.append({"epoch": epoch, "loss": train_loss, "val_auc": val_auc})
        logger.info(
            f"Epoch {epoch:3d}/{MAX_EPOCHS}  loss={train_loss:.4f}  val_auc={val_auc:.4f}"
        )

        # ベストモデル保存
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), MODEL_DIR / "transformer_best.pt")
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    # ---- ベストモデルで評価 ----
    model.load_state_dict(torch.load(MODEL_DIR / "transformer_best.pt", weights_only=True))
    logger.info(f"ベストモデル読み込み完了 (val_auc={best_auc:.4f})")

    metrics_valid = evaluate(model, valid_loader, "Valid")
    metrics_test  = evaluate(model, test_loader,  "Test")

    # ---- 学習曲線 ----
    hist_df = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hist_df["epoch"], hist_df["val_auc"], label="Valid AUC")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC")
    ax.set_title("Transformer 学習曲線")
    ax.legend()
    fig.savefig(REPORT_DIR / "transformer_learning_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- モデル一式保存 ----
    joblib.dump(
        {
            "model_state": model.state_dict(),
            "model_config": {
                "cat_vocab_sizes": cat_vocab_sizes,
                "cat_cols":        cat_cols,
                "n_num":           len(num_cols),
            },
            "encoders":   encoders,
            "num_stats":  num_stats,
            "num_cols":   num_cols,
            "cat_cols":   cat_cols,
        },
        MODEL_PATH,
    )
    logger.info(f"モデル保存: {MODEL_PATH}")

    # ---- サマリ ----
    print("\n" + "=" * 50)
    print("Transformer 学習完了サマリ")
    print("=" * 50)
    print(f"Valid AUC    : {metrics_valid['auc']:.4f}")
    print(f"Test  AUC    : {metrics_test['auc']:.4f}")
    print(f"Valid PR-AUC : {metrics_valid['pr_auc']:.4f}")
    print(f"Test  PR-AUC : {metrics_test['pr_auc']:.4f}")
    print(f"Best Val AUC : {best_auc:.4f}")
    print(f"モデル保存先  : {MODEL_PATH}")


if __name__ == "__main__":
    main()