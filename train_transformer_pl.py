"""
train_transformer_pl.py
PyCaLiAI - Transformer Plackett-Luce loss版

Plackett-Luce モデル:
  レース結果を「1着→2着→3着...の順列が生成される確率」として捉える。
  loss = -log P(観測された着順の順列)
       = -Σ_k log( score_k / Σ_{j>=k} score_j )

これは競馬の生成過程に最も近い理論モデル。

Usage:
    python train_transformer_pl.py
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
from sklearn.metrics import average_precision_score, roc_auc_score
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

MASTER_CSV  = DATA_DIR  / "master_20130105-20251228.csv"
MODEL_PATH  = MODEL_DIR / "transformer_pl_v1.pkl"

TARGET       = "fukusho_flag"
COL_RACE_ID  = "レースID(新/馬番無)"
COL_RANK     = "着順"
RANDOM_STATE = 42

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"使用デバイス: {DEVICE}")

# =========================================================
# 特徴量定義（train_transformer.pyと同じ）
# =========================================================
CAT_FEATURES = [
    "芝・ダ", "コース区分", "芝(内・外)", "馬場状態", "天気",
    "クラス名", "場所",
    "性別", "斤量", "ブリンカー",
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

MAX_HORSES   = 18
D_MODEL      = 128
N_HEADS      = 4
N_LAYERS     = 2
D_FF         = 256
DROPOUT      = 0.1
BATCH_SIZE   = 256
MAX_EPOCHS   = 50
PATIENCE     = 7
LR           = 1e-3


# =========================================================
# Plackett-Luce Loss
# =========================================================
class PlackettLuceLoss(nn.Module):
    """
    Plackett-Luce loss（着順ベース）

    数式:
      L = -Σ_{k=1}^{n} log( exp(s_k) / Σ_{j=k}^{n} exp(s_j) )

    s_k = k着馬のスコア（モデル出力）
    着順が小さいほど先に「選ばれる」確率が高いとモデル化。

    パディング馬（mask=True）はloss計算から除外。
    """

    def forward(
        self,
        scores: torch.Tensor,    # [batch, max_horses]
        ranks:  torch.Tensor,    # [batch, max_horses] 着順（1始まり）
        mask:   torch.Tensor,    # [batch, max_horses] True=padding
    ) -> torch.Tensor:

        batch_size = scores.size(0)
        total_loss = torch.tensor(0.0, device=scores.device)
        n_valid    = 0

        for b in range(batch_size):
            valid    = ~mask[b]                    # 有効な馬のマスク
            s        = scores[b][valid]             # 有効馬のスコア
            r        = ranks[b][valid]              # 有効馬の着順
            n        = valid.sum().item()

            if n < 2:
                continue

            # 着順でソート（1着→最後着の順）
            order    = torch.argsort(r)
            s_sorted = s[order]                    # [n] 着順昇順のスコア

            # Plackett-Luce log likelihood
            # k番目の馬が選ばれる確率 = exp(s_k) / Σ_{j>=k} exp(s_j)
            loss_b = torch.tensor(0.0, device=scores.device)
            for k in range(n - 1):               # 最後の1頭は必然なのでスキップ
                log_sum_exp = torch.logsumexp(s_sorted[k:], dim=0)
                loss_b = loss_b - (s_sorted[k] - log_sum_exp)

            total_loss = total_loss + loss_b / n
            n_valid   += 1

        return total_loss / max(n_valid, 1)


# =========================================================
# 前処理（train_transformer.pyと同じ）
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
            le = LabelEncoder()
            vals = df[col].tolist()
            if "__NaN__" not in vals:
                vals.append("__NaN__")
            le.fit(vals)
            encoders[col] = le
        else:
            le = encoders[col]
            known = set(le.classes_)
            df[col] = df[col].apply(lambda x: x if x in known else "__NaN__")
        df[col] = le.transform(df[col]) + 1

    if num_stats is None:
        num_stats = {}
    all_num = NUM_FEATURES + TIME_STR_FEATURES
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
        df[col] = df[col].fillna(0.0)

    return df, encoders, num_stats


# =========================================================
# Dataset（着順を追加で保持）
# =========================================================
class RaceDatasetPL(Dataset):
    """Plackett-Luce用Dataset。着順テンソルを追加で保持する。"""

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

            cat_data = torch.zeros(MAX_HORSES, len(cat_cols),  dtype=torch.long)
            num_data = torch.zeros(MAX_HORSES, len(all_num),   dtype=torch.float32)
            targets  = torch.zeros(MAX_HORSES,                 dtype=torch.float32)
            ranks    = torch.zeros(MAX_HORSES,                 dtype=torch.float32)
            mask     = torch.ones(MAX_HORSES,                  dtype=torch.bool)

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
                ranks[i]    = float(row[COL_RANK]) if not pd.isna(row[COL_RANK]) else 99.0
                mask[i]     = False

            self.races.append({
                "cat":     cat_data,
                "num":     num_data,
                "targets": targets,
                "ranks":   ranks,
                "mask":    mask,
            })

    def __len__(self) -> int:
        return len(self.races)

    def __getitem__(self, idx: int) -> dict:
        return self.races[idx]


# =========================================================
# モデル定義（train_transformer.pyと同じ構造）
# =========================================================
class HorseEmbedding(nn.Module):
    def __init__(
        self,
        cat_vocab_sizes: dict[str, int],
        cat_cols: list[str],
        n_num: int,
        d_model: int,
    ) -> None:
        super().__init__()
        self.cat_cols = cat_cols
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(
                cat_vocab_sizes[col] + 2,
                max(4, cat_vocab_sizes[col] // 4),
                padding_idx=0,
            )
            for col in cat_cols
        })
        emb_total = sum(max(4, cat_vocab_sizes[col] // 4) for col in cat_cols)
        self.proj = nn.Sequential(
            nn.Linear(emb_total + n_num, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
        )

    def forward(self, cat: torch.Tensor, num: torch.Tensor) -> torch.Tensor:
        embs = [self.embeddings[col](cat[:, :, i]) for i, col in enumerate(self.cat_cols)]
        x = torch.cat(embs + [num], dim=-1)
        return self.proj(x)


class RaceTransformerPL(nn.Module):
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
        self.embedding = HorseEmbedding(cat_vocab_sizes, cat_cols, n_num, d_model)
        encoder_layer  = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
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
        x      = self.embedding(cat, num)
        x      = self.transformer(x, src_key_padding_mask=mask)
        logits = self.head(x).squeeze(-1)
        return logits


# =========================================================
# 学習・評価ループ
# =========================================================
def train_epoch(
    model: RaceTransformerPL,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: PlackettLuceLoss,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for batch in loader:
        cat   = batch["cat"].to(DEVICE)
        num   = batch["num"].to(DEVICE)
        ranks = batch["ranks"].to(DEVICE)
        mask  = batch["mask"].to(DEVICE)

        optimizer.zero_grad()
        scores = model(cat, num, mask)
        loss   = criterion(scores, ranks, mask)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def predict_proba(
    model: RaceTransformerPL,
    loader: DataLoader,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_proba   = []
    all_targets = []

    for batch in loader:
        cat     = batch["cat"].to(DEVICE)
        num     = batch["num"].to(DEVICE)
        mask    = batch["mask"].to(DEVICE)
        targets = batch["targets"]

        scores    = model(cat, num, mask)
        proba     = torch.sigmoid(scores).cpu()
        valid     = ~batch["mask"]

        all_proba.append(proba[valid].numpy())
        all_targets.append(targets[valid].numpy())

    return np.concatenate(all_proba), np.concatenate(all_targets)


# =========================================================
# main
# =========================================================
def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"マスターCSV読み込み: {MASTER_CSV}")
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    logger.info(f"  {len(df):,}行 × {len(df.columns)}列")

    train_df = df[df["split"] == "train"].copy()
    valid_df = df[df["split"] == "valid"].copy()
    test_df  = df[df["split"] == "test"].copy()

    logger.info("前処理開始...")
    train_df, encoders, num_stats = preprocess(train_df, fit=True)
    valid_df, _,        _         = preprocess(valid_df, encoders, fit=False, num_stats=num_stats)
    test_df,  _,        _         = preprocess(test_df,  encoders, fit=False, num_stats=num_stats)

    cat_vocab_sizes = {col: len(encoders[col].classes_) for col in CAT_FEATURES if col in encoders}
    num_cols = [c for c in NUM_FEATURES + TIME_STR_FEATURES if c in train_df.columns]
    cat_cols = [c for c in CAT_FEATURES if c in train_df.columns]

    logger.info("Dataset構築中...")
    train_ds = RaceDatasetPL(train_df, cat_cols, num_cols, cat_vocab_sizes)
    valid_ds = RaceDatasetPL(valid_df, cat_cols, num_cols, cat_vocab_sizes)
    test_ds  = RaceDatasetPL(test_df,  cat_cols, num_cols, cat_vocab_sizes)
    logger.info(f"レース数: train={len(train_ds):,} / valid={len(valid_ds):,} / test={len(test_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = RaceTransformerPL(
        cat_vocab_sizes=cat_vocab_sizes,
        cat_cols=cat_cols,
        n_num=len(num_cols),
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"モデルパラメータ数: {total_params:,}")

    criterion = PlackettLuceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    best_auc     = 0.0
    patience_cnt = 0
    history      = []

    logger.info("学習開始...")
    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        scheduler.step()

        proba, y = predict_proba(model, valid_loader)
        val_auc  = roc_auc_score(y, proba)

        history.append({"epoch": epoch, "loss": train_loss, "val_auc": val_auc})
        logger.info(f"Epoch {epoch:3d}/{MAX_EPOCHS}  loss={train_loss:.4f}  val_auc={val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), MODEL_DIR / "transformer_pl_best.pt")
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(MODEL_DIR / "transformer_pl_best.pt", weights_only=True))
    logger.info(f"ベストモデル読み込み完了 (val_auc={best_auc:.4f})")

    # 評価
    for split_name, loader in [("Valid", valid_loader), ("Test", test_loader)]:
        proba, y = predict_proba(model, loader)
        auc      = roc_auc_score(y, proba)
        pr_auc   = average_precision_score(y, proba)
        logger.info(f"[{split_name}] AUC={auc:.4f}  PR-AUC={pr_auc:.4f}")

    # Test最終評価
    proba_test, y_test = predict_proba(model, test_loader)
    auc_test   = roc_auc_score(y_test, proba_test)
    pr_auc_test = average_precision_score(y_test, proba_test)

    # 学習曲線
    hist_df = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hist_df["epoch"], hist_df["val_auc"], label="Valid AUC")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC")
    ax.set_title("Transformer PL 学習曲線")
    ax.legend()
    fig.savefig(REPORT_DIR / "transformer_pl_learning_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 保存
    joblib.dump({
        "model_state":  model.state_dict(),
        "model_config": {"cat_vocab_sizes": cat_vocab_sizes, "cat_cols": cat_cols, "n_num": len(num_cols)},
        "encoders":     encoders,
        "num_stats":    num_stats,
        "num_cols":     num_cols,
        "cat_cols":     cat_cols,
    }, MODEL_PATH)
    logger.info(f"モデル保存: {MODEL_PATH}")

    print("\n" + "=" * 50)
    print("Transformer Plackett-Luce 学習完了サマリ")
    print("=" * 50)
    print(f"Test  AUC    : {auc_test:.4f}  (旧: 0.7540)")
    print(f"Test  PR-AUC : {pr_auc_test:.4f}  (旧: 0.4722)")
    print(f"Best Val AUC : {best_auc:.4f}")
    print(f"モデル保存先  : {MODEL_PATH}")


if __name__ == "__main__":
    main()