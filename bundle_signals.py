"""
bundle_signals.py
=================
export_weekly_marks.py で bundle.json に追加埋め込みする 3 シグナルの
予測ヘルパー。

  1. p_havoc:           havoc_v1.pkl で各レースの「波乱予想確率」を予測
  2. p_fukusho_stacked: fukusho_binary_v1.pkl で各馬の「複勝圏内 二値分類確率」を予測
  3. p_umaren_stacked:  umaren_pair_v1.pkl と v6 PL を α=0.8 で blend した
                        各ペアの 馬連 stacked 確率

公開 API:
  predict_p_havoc(df, v6_score_col='_v6_score') -> dict[rid -> float]
  predict_p_fukusho_stacked(df, v6_score_col='_v6_score') -> dict[(rid, ban) -> float]
  predict_p_umaren_stacked(df, v6_score_col='_v6_score', alpha=0.8) -> dict[rid -> list of dict]
"""
from __future__ import annotations
import logging
from pathlib import Path
from itertools import combinations

import joblib
import numpy as np
import pandas as pd

import pl_probs as PL

logger = logging.getLogger(__name__)

BASE       = Path(__file__).parent
HAVOC_PKL  = BASE / "models/havoc_v1.pkl"
FUKUSHO_PKL = BASE / "models/fukusho_binary_v1.pkl"
UMAREN_PKL = BASE / "models/umaren_pair_v1.pkl"

COL_RID = "レースID(新/馬番無)"
COL_BAN = "馬番"


def _apply_encoders(df, encs):
    df = df.copy()
    for c, le in encs.items():
        if c not in df.columns: continue
        v = df[c].astype(str).fillna("__NaN__")
        known = set(le.classes_)
        v = v.where(v.isin(known), "__NaN__")
        df[c] = le.transform(v)
    return df


# =============================================================
# 1. p_havoc (Phase 3 波乱予想)
# =============================================================
def _build_havoc_race_features(df, v6_score_col):
    """各レースに havoc model 用 race-level 特徴量を構築。"""
    rows = []
    for rid, g in df.groupby(COL_RID, sort=False):
        g = g.reset_index(drop=True)
        scores = pd.to_numeric(g[v6_score_col], errors="coerce").values
        if np.any(np.isnan(scores)): continue
        w = PL.pl_weights(scores)
        p_win = PL.all_tansho(w)
        entropy = -(p_win * np.log(np.clip(p_win, 1e-10, 1))).sum()
        s_sorted = np.sort(scores)[::-1]
        p_sorted = np.sort(p_win)[::-1]
        row = {
            COL_RID: rid,
            "_n_horses": len(g),
            "_v6_top1": float(s_sorted[0]),
            "_v6_top3_mean": float(s_sorted[:3].mean()),
            "_v6_top5_mean": float(s_sorted[:5].mean()),
            "_v6_max": float(scores.max()),
            "_v6_min": float(scores.min()),
            "_v6_std": float(scores.std()),
            "_v6_range": float(scores.max() - scores.min()),
            "_entropy": entropy,
            "_p_win_1st": float(p_sorted[0]),
            "_p_win_3rd": float(p_sorted[2]) if len(p_sorted) >= 3 else float(p_sorted[-1]),
            "_p_win_top3_sum": float(p_sorted[:3].sum()),
        }
        for c in ["場所", "芝・ダ", "距離", "馬場状態", "天気", "クラス名",
                  "コース区分", "年齢限定", "限定", "性別限定", "重量種別"]:
            if c in g.columns:
                row[c] = g.iloc[0][c]
        rows.append(row)
    return pd.DataFrame(rows)


def predict_p_havoc(df: pd.DataFrame, v6_score_col: str = "_v6_score") -> dict:
    """各レースの p_havoc を返す: dict[rid -> float in [0,1]]."""
    if not HAVOC_PKL.exists():
        logger.warning(f"{HAVOC_PKL} 未存在 → p_havoc スキップ")
        return {}
    if v6_score_col not in df.columns:
        logger.warning(f"{v6_score_col} 列不在 → p_havoc スキップ")
        return {}

    havoc = joblib.load(HAVOC_PKL)
    race_df = _build_havoc_race_features(df, v6_score_col)
    if len(race_df) == 0: return {}

    race_enc = _apply_encoders(race_df, havoc["encoders"])
    X = race_enc[havoc["feature_cols"]].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    p_raw = havoc["model"].predict(X)
    cal = havoc.get("calibrator")
    p_havoc = cal.predict(p_raw) if cal is not None else p_raw

    return dict(zip(race_df[COL_RID], p_havoc))


def havoc_class(p: float) -> str:
    if p < 0.3:  return "固い"
    if p < 0.5:  return "中庸"
    if p < 0.7:  return "波乱"
    return "大波乱"


# =============================================================
# 2. p_fukusho_stacked (複勝特化 binary, Phase 1 案)
# =============================================================
def predict_p_fukusho_stacked(df: pd.DataFrame,
                               v6_score_col: str = "_v6_score") -> dict:
    """各馬の p_fukusho_stacked を返す: dict[(rid_s, ban) -> float]."""
    if not FUKUSHO_PKL.exists():
        logger.warning(f"{FUKUSHO_PKL} 未存在 → p_fukusho_stacked スキップ")
        return {}
    if v6_score_col not in df.columns:
        logger.warning(f"{v6_score_col} 列不在 → p_fukusho_stacked スキップ")
        return {}

    fb = joblib.load(FUKUSHO_PKL)
    df_work = df.copy()
    # fukusho_binary は v6_score を 'v6_score' 列で受け取る
    if "v6_score" not in df_work.columns:
        df_work["v6_score"] = df_work[v6_score_col]
    df_enc = _apply_encoders(df_work, fb["encoders"])
    feats = fb["feature_cols"]
    missing = [c for c in feats if c not in df_enc.columns]
    if missing:
        logger.warning(f"fukusho_binary feats 不足 {len(missing)}: {missing[:5]}...")
        for c in missing:
            df_enc[c] = -9999
    X = df_enc[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    p_raw = fb["model"].predict(X)
    cal = fb.get("calibrator")
    p_fuku = cal.predict(p_raw) if cal is not None else p_raw

    out = {}
    for i, (rid, ban) in enumerate(zip(df[COL_RID], df[COL_BAN])):
        rid_s = str(rid)[:16]
        try:
            ban_i = int(ban)
        except Exception:
            continue
        out[(rid_s, ban_i)] = float(p_fuku[i])
    return out


# =============================================================
# 3. p_umaren_stacked (馬連 pair v1 + v6 PL を α=0.8 で blend)
# =============================================================
# umaren_pair_v1 の入力特徴量
_UM_NUM_FEATS = [
    "jockey_fuku30", "jockey_fuku90", "jockey_top3_rate",
    "trainer_fuku30", "trainer_fuku90",
    "horse_fuku10", "horse_fuku30",
    "kako5_avg_pos", "kako5_best_pos", "kako5_avg_ninki",
    "kako5_pos_vs_ninki", "kako5_avg_agari3f", "kako5_best_agari3f",
    "kako5_same_td_ratio", "kako5_same_dist_ratio",
    "kako5_pos_trend", "kako5_race_count",
]
_UM_CONTEXT_NUM = ["距離", "出走頭数"]
_UM_CONTEXT_CAT = ["場所", "芝・ダ", "クラス名", "馬場状態"]


def _build_umaren_pair_features(g: pd.DataFrame):
    """1 レース内の全 (i, j) ペアの特徴量を構築。 (n_pairs × n_feats)"""
    g = g.reset_index(drop=True)
    n = len(g)
    pairs = list(combinations(range(n), 2))
    if not pairs: return None, []

    rows = []
    ban_pairs = []
    for i, j in pairs:
        row = {}
        # 個馬数値: mean / |diff| / min / max
        for feat in _UM_NUM_FEATS:
            if feat not in g.columns:
                row[f"{feat}_mean"] = np.nan
                row[f"{feat}_diff"] = np.nan
                row[f"{feat}_min"]  = np.nan
                row[f"{feat}_max"]  = np.nan
                continue
            vi = pd.to_numeric(g.iloc[i][feat], errors="coerce")
            vj = pd.to_numeric(g.iloc[j][feat], errors="coerce")
            if pd.isna(vi) or pd.isna(vj):
                row[f"{feat}_mean"] = np.nan
                row[f"{feat}_diff"] = np.nan
                row[f"{feat}_min"]  = np.nan
                row[f"{feat}_max"]  = np.nan
            else:
                row[f"{feat}_mean"] = (vi + vj) / 2
                row[f"{feat}_diff"] = abs(vi - vj)
                row[f"{feat}_min"]  = min(vi, vj)
                row[f"{feat}_max"]  = max(vi, vj)
        # レース context: 共通の値
        for c in _UM_CONTEXT_NUM:
            row[c] = g.iloc[0][c] if c in g.columns else np.nan
        for c in _UM_CONTEXT_CAT:
            row[c] = g.iloc[0][c] if c in g.columns else "__NaN__"
        rows.append(row)
        ban_a = int(g.iloc[i][COL_BAN])
        ban_b = int(g.iloc[j][COL_BAN])
        ban_pairs.append((min(ban_a, ban_b), max(ban_a, ban_b), i, j))
    return pd.DataFrame(rows), ban_pairs


def predict_p_umaren_stacked(df: pd.DataFrame,
                              v6_score_col: str = "_v6_score",
                              alpha: float = 0.8,
                              top_n: int = 15) -> dict:
    """各レースの馬連 pair stack を返す: dict[rid_s -> list of {ban_pair, p_stacked, p_v6, p_pair}].

    alpha=0.8 は backtest_umaren_pair で実証された最適値 (ROI 81.89% @ test).
    top_n: 各レース上位 top_n ペアのみ出力 (出走 16 頭で 120 ペアあるので絞る)。
    """
    if not UMAREN_PKL.exists():
        logger.warning(f"{UMAREN_PKL} 未存在 → p_umaren_stacked スキップ")
        return {}
    if v6_score_col not in df.columns:
        logger.warning(f"{v6_score_col} 列不在 → p_umaren_stacked スキップ")
        return {}

    pair_bundle = joblib.load(UMAREN_PKL)
    pair_model = pair_bundle["model"]
    pair_feats = pair_bundle["feature_cols"]
    pair_encs  = pair_bundle["encoders"]
    pair_cal   = pair_bundle.get("calibrator")

    out = {}
    for rid, g in df.groupby(COL_RID, sort=False):
        if len(g) < 5: continue
        rid_s = str(rid)[:16]
        scores = pd.to_numeric(g[v6_score_col], errors="coerce").values
        if np.any(np.isnan(scores)): continue
        bans = g[COL_BAN].astype(int).values

        # 1) v6 PL p_umaren
        w = PL.pl_weights(scores)
        # 2) pair model predict
        pair_df, ban_pairs = _build_umaren_pair_features(g)
        if pair_df is None: continue
        pair_enc = _apply_encoders(pair_df, pair_encs)
        # 不足列補完
        for c in pair_feats:
            if c not in pair_enc.columns:
                pair_enc[c] = -9999
        Xp = pair_enc[pair_feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
        p_pair_raw = pair_model.predict(Xp)
        p_pair = pair_cal.predict(p_pair_raw) if pair_cal is not None else p_pair_raw

        # 3) blend
        pairs_out = []
        for (ban_a, ban_b, i, j), pp in zip(ban_pairs, p_pair):
            p_v6 = float(PL.p_umaren(w, i, j))
            p_blended = alpha * p_v6 + (1 - alpha) * float(pp)
            pairs_out.append({
                "ban_pair": f"{ban_a}-{ban_b}",
                "p_stacked": round(p_blended, 5),
                "p_v6": round(p_v6, 5),
                "p_pair_v1": round(float(pp), 5),
            })
        # top_n に絞る (p_stacked 降順)
        pairs_out.sort(key=lambda x: -x["p_stacked"])
        out[rid_s] = pairs_out[:top_n]
    return out
