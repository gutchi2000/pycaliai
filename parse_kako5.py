"""
parse_kako5.py
PyCaLiAI - 過去5走特徴量の生成

2つのモード:
  1) build_from_master()  : master CSV の自己結合で訓練用特徴量を生成
  2) build_from_kako5()   : 週次 kako5 CSV から予測用特徴量を生成

特徴量設計（過学習防止のため集約値のみ）:
  kako5_avg_pos         過去5走 平均着順
  kako5_std_pos         過去5走 着順の標準偏差（安定度）
  kako5_best_pos        過去5走 最高着順
  kako5_avg_ninki       過去5走 平均人気
  kako5_pos_vs_ninki    過去5走 平均(人気-着順) → +なら格上走
  kako5_avg_agari3f     過去5走 平均上り3F
  kako5_best_agari3f    過去5走 最速上り3F
  kako5_same_td_ratio   今回と同じ芝/ダ率
  kako5_same_dist_ratio 今回と同距離帯(±200m)率
  kako5_same_place_ratio 今回と同場所率
  kako5_pos_trend       着順の直近トレンド（負=上昇傾向）
  kako5_race_count      有効走数(1-5, キャリア浅い馬のフラグ)

Usage:
    # 訓練データ生成（master CSV → master + kako5特徴量）
    python parse_kako5.py --mode master

    # 週次予測用（kako5 CSV → kako5_features.csv）
    python parse_kako5.py --mode weekly --date 20260329
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(r"E:\PyCaLiAI")
DATA_DIR = BASE_DIR / "data"

KAKO5_COLS = [
    "kako5_avg_pos", "kako5_std_pos", "kako5_best_pos",
    "kako5_avg_ninki", "kako5_pos_vs_ninki",
    "kako5_avg_agari3f", "kako5_best_agari3f",
    "kako5_same_td_ratio", "kako5_same_dist_ratio", "kako5_same_place_ratio",
    "kako5_pos_trend", "kako5_race_count",
    # --- 好走3分類 (Phase 5) ---
    "kako5_expected_good_count",   # 期待通り好��: 人気<=3 AND 着順<=3（人気なし時: 着順<=3）
    "kako5_upset_good_count",      # 格上挑戦好走: 人気>=8 AND 着順<=5（人気なし時: NaN）
    "kako5_hidden_good_count",     # 隠れた好走: 着順>5 AND 上り3F < 34.5
    # --- 同条件ベスト ---
    "kako5_same_cond_best_pos",    # 過去5走のうち同条件(TD+距離帯)での最高着順
]

# 全キャリア検索で算出する特徴量（build_from_master専用、5走制限なし）
HIST_COLS = [
    "hist_same_cond_best_pos",     # 全キャリアで同TD+同距離帯の最高���順
    "hist_same_cond_top3_rate",    # 全キャリアで同条件の複勝率
    "hist_same_cond_count",        # 全キャリアで同条件出走回数
    "hist_same_place_best_pos",    # 全��ャリアで同場所の��高着順
]


# =========================================================
# 共通: 過去走リストから特徴量を計算
# =========================================================
def _compute_features(
    past_races: list[dict],
    current_td: str | None = None,
    current_dist: float | None = None,
    current_place: str | None = None,
) -> dict:
    """過去走データ(最大5走)から集約特徴量を算出する。

    past_races: [{"着順":int, "人気":int, "上り3F":float, "TD":str, "距離":float, "場所":str}, ...]
                新しい順（1走前が先頭）
    """
    n = len(past_races)
    result = {c: np.nan for c in KAKO5_COLS}

    if n == 0:
        result["kako5_race_count"] = 0
        return result

    positions = [r["着順"] for r in past_races if r["着順"] is not None]
    ninkis    = [r["人気"] for r in past_races if r["人気"] is not None]
    agari3fs  = [r["上り3F"] for r in past_races if r["上り3F"] is not None]

    result["kako5_race_count"] = n

    # --- 着順系 ---
    if positions:
        result["kako5_avg_pos"]  = np.mean(positions)
        result["kako5_std_pos"]  = np.std(positions) if len(positions) >= 2 else 0.0
        result["kako5_best_pos"] = min(positions)

    # --- 人気系 ---
    if ninkis:
        result["kako5_avg_ninki"] = np.mean(ninkis)

    # --- 人気 vs 着順 ---
    pv = [r["人気"] - r["着順"]
          for r in past_races
          if r["人気"] is not None and r["着順"] is not None]
    if pv:
        result["kako5_pos_vs_ninki"] = np.mean(pv)

    # --- 上り3F ---
    if agari3fs:
        result["kako5_avg_agari3f"]  = np.mean(agari3fs)
        result["kako5_best_agari3f"] = min(agari3fs)

    # --- 適性率 ---
    if current_td is not None:
        same_td = sum(1 for r in past_races if r.get("TD") == current_td)
        result["kako5_same_td_ratio"] = same_td / n

    if current_dist is not None:
        same_dist = sum(1 for r in past_races
                        if r.get("距離") is not None
                        and abs(r["距離"] - current_dist) <= 200)
        result["kako5_same_dist_ratio"] = same_dist / n

    if current_place is not None:
        same_place = sum(1 for r in past_races if r.get("場所") == current_place)
        result["kako5_same_place_ratio"] = same_place / n

    # --- トレンド（直近ほど index小 → 負の傾きなら上昇傾向）---
    if len(positions) >= 2:
        x = np.arange(len(positions), dtype=float)
        coeffs = np.polyfit(x, positions, 1)
        result["kako5_pos_trend"] = coeffs[0]
    elif len(positions) == 1:
        result["kako5_pos_trend"] = 0.0

    # --- 好走3分類 ---
    # 1. 期待通りの好走: 人気<=3 AND 着順<=3（人気データなし時は着順<=3のみ）
    expected_good = 0
    for r in past_races:
        pos = r.get("着順")
        ninki = r.get("人気")
        if pos is not None and pos <= 3:
            if ninki is not None:
                if ninki <= 3:
                    expected_good += 1
            else:
                # 人気データなし（master CSV）→ 着順<=3 を代用
                expected_good += 1
    result["kako5_expected_good_count"] = expected_good

    # 2. 格上挑戦での好走: 人気>=8 AND 着順<=5（人気なし→NaN）
    upset_good = 0
    has_ninki = any(r.get("人気") is not None for r in past_races)
    if has_ninki:
        for r in past_races:
            pos = r.get("着順")
            ninki = r.get("人気")
            if pos is not None and ninki is not None and ninki >= 8 and pos <= 5:
                upset_good += 1
        result["kako5_upset_good_count"] = upset_good
    # else: NaN（人気データなし）

    # 3. 隠れた好走: 着順>5 だが上り3F < 34.5（末脚が良い = 次走で爆発する予兆）
    hidden_good = 0
    for r in past_races:
        pos = r.get("着順")
        agari = r.get("上り3F")
        if pos is not None and pos > 5 and agari is not None and agari < 34.5:
            hidden_good += 1
    result["kako5_hidden_good_count"] = hidden_good

    # --- 同条件ベスト（過去5走のうち同TD+同距離帯）---
    if current_td is not None and current_dist is not None:
        same_cond_positions = [
            r["着順"] for r in past_races
            if r["着順"] is not None
            and r.get("TD") == current_td
            and r.get("距離") is not None
            and abs(r["距離"] - current_dist) <= 200
        ]
        if same_cond_positions:
            result["kako5_same_cond_best_pos"] = min(same_cond_positions)

    return result


def _safe_float(v) -> float | None:
    try:
        f = float(v)
        return f if not np.isnan(f) else None
    except (ValueError, TypeError):
        return None


def _safe_int(v) -> int | None:
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return None


# =========================================================
# Mode 1: master CSV 自己結合
# =========================================================
def build_from_master(master_path: Path, output_path: Path) -> pd.DataFrame:
    """master CSV を読み込み、各行に過去5走特徴量を追加して保存する。

    各馬の「その行の日付より前」の最大5走を使って特徴量を計算する。
    → 未来データのリーク不可能。
    """
    logger.info(f"master CSV 読み込み中: {master_path}")
    df = pd.read_csv(master_path, encoding="utf-8-sig")
    logger.info(f"  読み込み完了: {len(df):,}行")

    # 日付をdatetimeに変換
    df["_date"] = pd.to_datetime(df["date_dt"], errors="coerce")

    # 芝・ダのコード正規化
    td_map = {"芝": "T", "ダ": "D", "ダート": "D", "T": "T", "D": "D"}
    df["_td_code"] = df["芝・ダ"].map(td_map).fillna("")
    df["_dist"]    = pd.to_numeric(df["距離"], errors="coerce")
    df["_place"]   = df["場所"].astype(str)

    # 馬ごとに日付ソートしてインデックスを振る
    horse_col = "血統登録番号"
    df = df.sort_values([horse_col, "_date"]).reset_index(drop=True)

    logger.info("過去5走特徴量を計算中...")

    # 高速化: グループ単位で一括処理
    feat_rows = []
    for horse_id, group in df.groupby(horse_col, sort=False):
        group = group.sort_values("_date")
        idxs     = group.index.tolist()
        pos_vals = pd.to_numeric(group["前走確定着順"], errors="coerce").values
        着順_vals = pd.to_numeric(group["着順"], errors="coerce").values
        ninki_raw = group.get("人気")  # master に人気列があるか確認

        # 全キャリア検索用: TD・距離・場所を事前取得
        all_td    = group["_td_code"].values
        all_dist  = group["_dist"].values
        all_place = group["_place"].values

        for seq_i, idx in enumerate(idxs):
            row = group.loc[idx]

            # この馬のこの行より前の走を最大5走取得
            past_indices = idxs[max(0, seq_i - 5): seq_i]  # 自分を含まない

            past_races = []
            for pi in reversed(past_indices):  # 新しい順
                pr = group.loc[pi]
                past_races.append({
                    "着順": _safe_int(pr.get("着順")),
                    "人気": _safe_int(pr.get("人気")) if "人気" in pr.index else None,
                    "上り3F": _safe_float(pr.get("前走上り3F")),
                    "TD": td_map.get(str(pr.get("芝・ダ", "")), ""),
                    "距離": _safe_float(pr.get("距離")),
                    "場所": str(pr.get("場所", "")),
                })

            feats = _compute_features(
                past_races,
                current_td=row.get("_td_code"),
                current_dist=_safe_float(row.get("_dist")),
                current_place=row.get("_place"),
            )

            # --- 全キャリア同条件ベスト（5走制限なし） ---
            cur_td    = row.get("_td_code", "")
            cur_dist  = _safe_float(row.get("_dist"))
            cur_place = row.get("_place", "")

            for hcol in HIST_COLS:
                feats[hcol] = np.nan

            if seq_i > 0 and cur_td and cur_dist is not None:
                # 同条件 = 同TD + 同距離帯(±200m)
                same_cond_pos = []
                for pi_seq in range(seq_i):
                    pos = 着順_vals[pi_seq]
                    if np.isnan(pos):
                        continue
                    td_i = all_td[pi_seq]
                    dist_i = all_dist[pi_seq]
                    if td_i == cur_td and not np.isnan(dist_i) and abs(dist_i - cur_dist) <= 200:
                        same_cond_pos.append(int(pos))

                if same_cond_pos:
                    feats["hist_same_cond_best_pos"] = min(same_cond_pos)
                    feats["hist_same_cond_top3_rate"] = sum(1 for p in same_cond_pos if p <= 3) / len(same_cond_pos)
                    feats["hist_same_cond_count"] = len(same_cond_pos)

            if seq_i > 0 and cur_place:
                # 同場所ベスト
                same_place_pos = []
                for pi_seq in range(seq_i):
                    pos = 着順_vals[pi_seq]
                    if np.isnan(pos):
                        continue
                    if all_place[pi_seq] == cur_place:
                        same_place_pos.append(int(pos))
                if same_place_pos:
                    feats["hist_same_place_best_pos"] = min(same_place_pos)

            feats["_idx"] = idx
            feat_rows.append(feats)

    feat_df = pd.DataFrame(feat_rows).set_index("_idx")
    all_cols = KAKO5_COLS + HIST_COLS
    for col in all_cols:
        df[col] = feat_df[col] if col in feat_df.columns else np.nan

    # 一時列削除
    df = df.drop(columns=["_date", "_td_code", "_dist", "_place"])

    # 保存
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    all_feature_count = len(KAKO5_COLS) + len(HIST_COLS)
    logger.info(f"保存完了: {output_path} ({len(df):,}行, +{all_feature_count}特徴量)")

    # 統計表示
    for col in all_cols:
        valid_pct = df[col].notna().mean() * 100
        logger.info(f"  {col}: カバレッジ={valid_pct:.1f}%  mean={df[col].mean():.3f}")

    return df


# =========================================================
# Mode 2: 週次 kako5 CSV から特徴量生成
# =========================================================
def build_from_kako5(kako5_path: Path, weekly_csv_path: Path | None = None) -> pd.DataFrame:
    """kako5 CSV を読み込み、馬ごとに過去5走特徴量を生成する。

    Returns: DataFrame with columns [レースID(新), 馬番, ...kako5特徴量]
    """
    logger.info(f"kako5 CSV 読み込み中: {kako5_path}")

    race_blocks = []
    current_header = None

    with open(kako5_path, encoding="cp932", errors="replace") as f:
        reader = csv.reader(f)
        for row_num, row in enumerate(reader):
            if len(row) <= 1:
                continue
            # レースヘッダ行: 19列でレースID(新)が先頭
            if len(row) == 19 and row[0] and len(row[0]) >= 10 and row[0][:4].isdigit():
                current_header = {
                    "レースID(新)": row[0],
                    "場所": row[3],
                    "芝・ダート": row[8],
                    "距離": _safe_float(row[9]),
                }
                continue
            # カラムヘッダ行: 72列で先頭が「枠番」
            if len(row) == 72 and row[0] in ("枠番",):
                continue
            # データ行: 72列で先頭が数字(枠番)
            if len(row) == 72 and row[0].isdigit() and current_header:
                uma_ban = _safe_int(row[2])
                if uma_ban is None:
                    continue

                td_map = {"芝": "T", "ダ": "D", "ダート": "D", "T": "T", "D": "D"}
                current_td = td_map.get(current_header["芝・ダート"], "")
                current_dist = current_header["距離"]
                current_place = current_header["場所"]

                # 5走分のデータを抽出
                # Race 1: cols 14-23, Race 2: 24-35, Race 3: 36-47, Race 4: 48-59, Race 5: 60-71
                race_offsets = [
                    (14, 18, 19, 21),  # (場所, 着順, 人気, 上り3F)
                    (26, 30, 31, 33),
                    (38, 42, 43, 45),
                    (50, 54, 55, 57),
                    (62, 66, 67, 69),
                ]
                # TD offset: 場所+1, 距離offset: 場所+2
                td_offsets = [15, 27, 39, 51, 63]
                dist_offsets = [16, 28, 40, 52, 64]

                past_races = []
                for i, (place_i, pos_i, ninki_i, agari_i) in enumerate(race_offsets):
                    pos = _safe_int(row[pos_i])
                    if pos is None or pos == 0:
                        continue
                    past_races.append({
                        "着順": pos,
                        "人気": _safe_int(row[ninki_i]),
                        "上り3F": _safe_float(row[agari_i]),
                        "TD": row[td_offsets[i]] if td_offsets[i] < len(row) else "",
                        "距離": _safe_float(row[dist_offsets[i]]) if dist_offsets[i] < len(row) else None,
                        "場所": row[place_i] if place_i < len(row) else "",
                    })

                feats = _compute_features(
                    past_races,
                    current_td=current_td,
                    current_dist=current_dist,
                    current_place=current_place,
                )
                feats["レースID(新)"] = current_header["レースID(新)"]
                feats["馬番"] = uma_ban
                race_blocks.append(feats)

    if not race_blocks:
        logger.warning("kako5 CSV からデータを抽出できませんでした。")
        return pd.DataFrame(columns=["レースID(新)", "馬番"] + KAKO5_COLS)

    result = pd.DataFrame(race_blocks)
    logger.info(f"  {len(result)}頭の kako5 特徴量を生成")

    # 統計表示
    for col in KAKO5_COLS:
        if col in result.columns:
            valid_pct = result[col].notna().mean() * 100
            mean_val  = result[col].mean()
            logger.info(f"  {col}: カバレッジ={valid_pct:.1f}%  mean={mean_val:.3f}")

    return result


# =========================================================
# CLI
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="過去5走特徴量の生成")
    parser.add_argument("--mode", choices=["master", "weekly"], required=True,
                        help="master: 訓練データ生成, weekly: 予測用生成")
    parser.add_argument("--date", type=str, default=None,
                        help="週次モード時の日付 (YYYYMMDD)")
    args = parser.parse_args()

    if args.mode == "master":
        master_path = DATA_DIR / "master_20130105-20251228.csv"
        output_path = DATA_DIR / "master_kako5.csv"
        build_from_master(master_path, output_path)

    elif args.mode == "weekly":
        if not args.date:
            logger.error("--date を指定してください (例: --date 20260329)")
            sys.exit(1)
        kako5_path = DATA_DIR / "kako5" / f"{args.date}.csv"
        if not kako5_path.exists():
            logger.error(f"kako5 CSV が見つかりません: {kako5_path}")
            sys.exit(1)
        result = build_from_kako5(kako5_path)
        out_path = DATA_DIR / "kako5" / f"features_{args.date}.csv"
        result.to_csv(out_path, index=False, encoding="utf-8-sig")
        logger.info(f"保存: {out_path}")


if __name__ == "__main__":
    main()
