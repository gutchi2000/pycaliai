"""
export_weekly_marks.py
======================
週末出走表 CSV (data/weekly/YYYYMMDD.csv) から
Cowork 入力用の印 JSON を出力する。

役割:
  - PyCaLiAI 側: 印付け (◎〇▲△△) + 確率 + race_confidence を JSON 化
  - Cowork 側 (Anthropic Desktop App): JSON を読んで馬券構築

使い方:
    # 単週分を一括出力 (race ごとに 1 ファイル + 全 race を 1 ファイルにまとめた bundle)
    python export_weekly_marks.py --csv data/weekly/20260426.csv

    # モデル指定
    python export_weekly_marks.py --csv data/weekly/20260426.csv --model v5

    # 出力先指定
    python export_weekly_marks.py --csv data/weekly/20260426.csv \
        --out-dir reports/cowork_input/20260426/

出力:
    reports/cowork_input/{YYYYMMDD}/{race_id}.json   ... 1 race / 1 file
    reports/cowork_input/{YYYYMMDD}_bundle.json       ... 全 race を 1 file に集約

互換性: バックテスト用 reports/marks_v5/ と同一の JSON スキーマ
        (docs/marks_schema.md を参照)
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE = Path(__file__).parent

# 既存ロジックを再利用 (parse_csv で週次CSVを DF 化, export_race で JSON 構築)
from predict_weekly import parse_csv
import backtest_pl_ev as be
from backtest_pl_ev import COL_RID, COL_BAN
from export_marks_json import export_race


def build_tansho_idx_from_weekly(df: pd.DataFrame) -> dict:
    """週次CSV の "単勝" 列から tansho_idx[(rid_s, ban)] = float を構築。"""
    tansho_idx: dict = {}
    if "単勝" not in df.columns:
        logger.warning("週次CSV に '単勝' 列がないため tansho_odds は null になります。")
        return tansho_idx
    for _, r in df.iterrows():
        try:
            rid_s = str(r[COL_RID])[:16]
            ban = int(r[COL_BAN])
            v_raw = r["単勝"]
            if v_raw is None or pd.isna(v_raw):
                continue
            v = float(v_raw)
            if v <= 0:
                continue
            tansho_idx[(rid_s, ban)] = v
        except Exception:
            continue
    logger.info(f"単勝オッズ取得 (weekly CSV): {len(tansho_idx):,} 馬")
    return tansho_idx


def build_odds_from_od_csv(date_str: str):
    """data/odds/OD{YYMMDD}.CSV があれば読んで (tansho_idx, fuku_idx) を返す。

    OD CSV (TARGET 形式) は単勝・複勝下限/上限を含むため、
    weekly CSV のオッズより精度が高い (複勝は weekly に存在しない)。

    Returns:
        (tansho_idx, fuku_idx) tuple, or None if file missing / parse failed.
        - tansho_idx[(rid_s, ban)] = float
        - fuku_idx[(rid_s, ban)]   = (low, high)
    """
    yy = date_str[2:]  # 20260426 → 260426
    candidates = [
        BASE / "data" / "odds" / f"OD{yy}.CSV",
        BASE / "data" / "odds" / f"OD{yy}.csv",
    ]
    od_path = next((p for p in candidates if p.exists()), None)
    if od_path is None:
        logger.info(f"OD CSV 未配置: data/odds/OD{yy}.CSV (weekly CSV のオッズを使用)")
        return None

    try:
        from parse_od_csv import load_od_odds
        odds_df = load_od_odds(od_path, date=date_str)
    except Exception as e:
        logger.warning(f"OD CSV 読み込み失敗: {od_path.name}: {e}")
        return None

    tansho_idx: dict = {}
    fuku_idx: dict = {}
    for _, r in odds_df.iterrows():
        try:
            rid_s = str(r["race_id"])[:16]
            ban = int(r["horse_num"])
            tan = float(r["tan_odds"])
            if tan > 0:
                tansho_idx[(rid_s, ban)] = tan
            flow = r["fuku_low"]
            fhigh = r["fuku_high"]
            if pd.notna(flow) and pd.notna(fhigh) and float(flow) > 0:
                fuku_idx[(rid_s, ban)] = (float(flow), float(fhigh))
        except Exception:
            continue

    logger.info(
        f"OD CSV 取得: {od_path.name} → 単勝 {len(tansho_idx):,} 馬 / "
        f"複勝 {len(fuku_idx):,} 馬 ({odds_df['race_id'].nunique()} race)"
    )
    return tansho_idx, fuku_idx


def ensure_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """race_meta が "日付" を参照するため YYYYMMDD 文字列列を必ず作る。"""
    if "日付" in df.columns and df["日付"].astype(str).str.len().min() == 8:
        return df
    if "日付S" in df.columns:
        def _to_yyyymmdd(s):
            try:
                parts = [int(x) for x in str(s).split(".")]
                return "{:04d}{:02d}{:02d}".format(*parts)
            except Exception:
                return ""
        df = df.copy()
        df["日付"] = df["日付S"].apply(_to_yyyymmdd)
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True,
                    help="週次CSV (例: data/weekly/20260426.csv)")
    ap.add_argument("--model", default="v5",
                    help="モデル tag (default: v5)")
    ap.add_argument("--out-dir", default=None,
                    help="出力ディレクトリ "
                         "(default: reports/cowork_input/{YYYYMMDD}/)")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        logger.error(f"CSV not found: {csv_path}")
        return 1

    date_str = csv_path.stem  # YYYYMMDD 期待
    if not (date_str.isdigit() and len(date_str) == 8):
        logger.warning(f"ファイル名が YYYYMMDD でない: {date_str} "
                       "(date 推定不可、出力先指定推奨)")

    out_dir = Path(args.out_dir) if args.out_dir else (
        BASE / "reports" / "cowork_input" / date_str
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------ モデル & calibrator ロード ------
    tag = args.model
    be.MODEL_PKL = BASE / f"models/unified_rank_{tag}.pkl"
    be.CAL_PKL   = BASE / f"models/pl_calibrators_{tag}.pkl"
    if not be.MODEL_PKL.exists():
        logger.error(f"モデル未存在: {be.MODEL_PKL}")
        return 1

    bundle = joblib.load(be.MODEL_PKL)
    model = bundle["model"]
    feats = bundle["feature_cols"]
    encs  = bundle["encoders"]
    logger.info(f"モデル: {be.MODEL_PKL.name} ({len(feats)} feats)")

    calibrators = None
    if be.CAL_PKL.exists():
        cal_bundle = joblib.load(be.CAL_PKL)
        calibrators = cal_bundle.get("calibrators", cal_bundle)
        logger.info(f"calibrator: {be.CAL_PKL.name}")
    else:
        logger.warning(f"calibrator 未存在: {be.CAL_PKL} → raw PL 確率で出力")

    # ------ CSV パース ------
    logger.info(f"weekly CSV パース: {csv_path}")
    df = parse_csv(csv_path)
    df = ensure_date_column(df)
    logger.info(f"パース結果: {len(df):,} 馬 / "
                f"{df[COL_RID].nunique():,} レース")

    # ------ feats に含まれるが週次CSVにない列を NaN/空で補完 ------
    # 例: 騎手コード, hist_same_cond_*, trnH_*, trnW_*, course_*, jockey_*
    # export_race() 内で encoder 適用 + pd.to_numeric → NaN → -9999 fillna されるので
    # 形だけ揃える。
    cat_cols = set(encs.keys())
    missing = [c for c in feats if c not in df.columns]
    if missing:
        logger.warning(f"週次CSVに不足の {len(missing)} 列を補完: "
                       f"{missing[:10]}{'...' if len(missing) > 10 else ''}")
        for c in missing:
            if c in cat_cols:
                # カテゴリ列 → "__NaN__" 文字列 (encoder 側で未知値扱いに)
                df[c] = "__NaN__"
            else:
                df[c] = np.nan

    # オッズ: data/odds/OD{YYMMDD}.CSV (TARGET 形式) を優先、無ければ weekly CSV から
    od_result = build_odds_from_od_csv(date_str)
    if od_result is not None:
        tansho_idx, fuku_idx = od_result
        logger.info("オッズソース: data/odds/OD CSV (高精度・複勝あり)")
    else:
        tansho_idx = build_tansho_idx_from_weekly(df)
        fuku_idx = {}  # 週次CSV に複勝オッズなし
        logger.info("オッズソース: weekly CSV (単勝のみ、複勝は null)")

    # ------ race ごとに JSON 出力 ------
    n_done = 0
    n_skip = 0
    saved_paths: list[Path] = []
    for rid, g in df.groupby(COL_RID, sort=False):
        if len(g) < 5:
            n_skip += 1
            continue
        try:
            payload = export_race(rid, g, model, feats, encs,
                                   tansho_idx, fuku_idx, calibrators)
        except Exception as e:
            logger.error(f"  rid={rid}: {e}")
            n_skip += 1
            continue
        rid_s = payload["race_id"]
        out_path = out_dir / f"{rid_s}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        saved_paths.append(out_path)
        n_done += 1

    logger.info(f"[done] {n_done:,} JSON ({n_skip} skip) → {out_dir}")

    # ------ bundle (1 file に全 race) ------
    bundle_path = out_dir.parent / f"{date_str}_bundle.json"
    races = []
    for p in sorted(saved_paths):
        with open(p, encoding="utf-8") as f:
            races.append(json.load(f))
    with open(bundle_path, "w", encoding="utf-8") as f:
        json.dump({"date": date_str, "model": tag,
                   "race_count": len(races), "races": races},
                  f, indent=2, ensure_ascii=False)
    logger.info(f"[bundle] {bundle_path}  ({len(races):,} races, "
                f"{bundle_path.stat().st_size/1024:.1f} KB)")

    print()
    print(f"=== Cowork 入力 JSON 出力完了 ===")
    print(f"  個別: {out_dir}/")
    print(f"  集約: {bundle_path}")
    print(f"  → このファイルを Cowork (Anthropic Desktop App) に投げて買い目を受け取る")
    return 0


if __name__ == "__main__":
    sys.exit(main())
