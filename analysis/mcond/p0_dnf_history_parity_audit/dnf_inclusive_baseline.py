# -*- coding: utf-8 -*-
"""
item6: DNF-inclusive baseline再計算（既存baselineの計測訂正、研究の再探索ではない）。

current v6(モデル・ハイパーパラメータとも一切変更しない)を全starterへ適用した
場合の指標を算出する。DNF馬は出走馬としてsoftmax分母へ含め、勝ち・top3では
失敗として扱う。取消・除外は分母から除く(元々scoring対象にならない)。

年別・DNFあり/なしレース別に、◎勝率・◎top3率・NDCG・Brier・loglossを
finisher-only評価との差とともに算出し、race単位bootstrap CIを付す。
パラメータ変更・モデル選択は行っていない。

実行: venv311\\Scripts\\python.exe -m analysis.mcond.p0_dnf_history_parity_audit.dnf_inclusive_baseline
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(BASE))
from build_dataset import add_rolling_stats, add_horse_rolling_stats, add_pace_features  # noqa: E402
from build_master_v2 import merge_hosei, merge_training  # noqa: E402
from analysis.mcond.exp13_nonfinish_risk_dev.gate0b_dnf_full_starter_score import (  # noqa: E402
    build_full_pre_dropna_universe_all_cols, compute_history_features_expanding,
    kako5_for_target_rows,
)

OUT_DIR = Path(__file__).resolve().parent / "out"
MASTER_V2 = BASE / "data" / "master_v2_20130105-20251228.csv"
MODEL_PKL = BASE / "models" / "unified_rank_v6.pkl"
COL_RID16, COL_BAN, COL_JYUN = "レースID(新/馬番無)", "馬番", "着順"

N_BOOT = 1000
RNG_SEED = 20260922


def apply_encoders(df, encs):
    df = df.copy()
    for c, le in encs.items():
        if c not in df.columns:
            continue
        v = df[c].astype(str).fillna("__NaN__")
        known = set(le.classes_)
        v = v.where(v.isin(known), "__NaN__")
        df[c] = le.transform(v)
    return df


def ndcg_at_k(rel: np.ndarray, score: np.ndarray, k=5) -> float:
    n = len(rel)
    k = min(k, n)
    order = np.argsort(-score)[:k]
    gains = (2.0 ** rel[order] - 1)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float((gains * discounts).sum())
    iorder = np.argsort(-rel)[:k]
    igains = (2.0 ** rel[iorder] - 1)
    idcg = float((igains * discounts).sum())
    return dcg / idcg if idcg > 0 else 0.0


def race_metrics(scores: np.ndarray, jyun: np.ndarray, is_dnf: np.ndarray) -> dict:
    """1レース分の指標。jyunはNaN=DNF。"""
    n = len(scores)
    w = np.exp(scores - scores.max())
    p_win = w / w.sum()
    actual_win = (jyun == 1).astype(float)
    actual_win = np.where(is_dnf, 0.0, actual_win)

    hon = int(np.argmax(scores))
    hon_win = bool((not is_dnf[hon]) and jyun[hon] == 1)
    hon_top3 = bool((not is_dnf[hon]) and jyun[hon] <= 3)

    rel = np.where(is_dnf, 0.0, np.clip(6 - jyun, 0, 5))
    ndcg5 = ndcg_at_k(rel, scores, k=5)

    eps = 1e-9
    p_clip = np.clip(p_win, eps, 1 - eps)
    logloss = float(-(actual_win * np.log(p_clip) + (1 - actual_win) * np.log(1 - p_clip)).mean())
    brier = float(np.mean((p_win - actual_win) ** 2))

    return {"n": n, "hon_win": hon_win, "hon_top3": hon_top3, "ndcg5": ndcg5,
            "logloss": logloss, "brier": brier, "has_dnf": bool(is_dnf.any())}


def aggregate(metrics_list: list[dict]) -> dict:
    if not metrics_list:
        return {}
    n_race = len(metrics_list)
    return {
        "n_race": n_race,
        "hon_win_rate": float(np.mean([m["hon_win"] for m in metrics_list])),
        "hon_top3_rate": float(np.mean([m["hon_top3"] for m in metrics_list])),
        "ndcg5": float(np.mean([m["ndcg5"] for m in metrics_list])),
        "logloss": float(np.mean([m["logloss"] for m in metrics_list])),
        "brier": float(np.mean([m["brier"] for m in metrics_list])),
    }


def bootstrap_ci(metrics_list: list[dict], key: str, n_boot=N_BOOT, seed=RNG_SEED):
    if not metrics_list:
        return None
    rng = np.random.default_rng(seed)
    vals = np.array([m[key] for m in metrics_list])
    n = len(vals)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if key in ("hon_win", "hon_top3"):
            boots.append(float(vals[idx].mean()))
        else:
            boots.append(float(vals[idx].mean()))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"mean": float(vals.mean()), "ci95": [float(lo), float(hi)]}


def main():
    t0 = time.time()
    print("[1] pre-dropna full universe構築(全120特徴の原材料込み)...")
    full = build_full_pre_dropna_universe_all_cols()

    print("[2] 全期間のDNF(止)行を特定...")
    from analysis.mcond.p0_dnf_history_parity_audit.build_corrected_universe import (
        load_full_raw_universe,
    )
    raw = load_full_raw_universe()
    dnf_rows_raw = raw[raw["_finish_class"] == "dnf"][[COL_RID16, COL_BAN]].copy()
    dnf_rows_raw[COL_RID16] = dnf_rows_raw[COL_RID16].astype(np.int64)
    dnf_rows_raw[COL_BAN] = dnf_rows_raw[COL_BAN].astype(np.int64)
    print(f"    全期間DNF = {len(dnf_rows_raw):,}")

    print("[3] build_dataset.py実関数でjockey/trainer/horse_fuku, pace特徴を計算...")
    full2 = add_rolling_stats(full.copy())
    full2 = add_horse_rolling_stats(full2)
    full2 = add_pace_features(full2)
    print(f"    完了 ({time.time()-t0:.0f}s)")

    print("[4] course/jockey_n_prev系(expanding cumcount)を計算...")
    full3 = compute_history_features_expanding(full2)
    print(f"    完了 ({time.time()-t0:.0f}s)")

    dnf_rows = full3.merge(dnf_rows_raw, on=[COL_RID16, COL_BAN], how="inner")
    print(f"    fullに実在するDNF行 = {len(dnf_rows):,} / {len(dnf_rows_raw):,}")

    print("[5] merge_hosei/merge_training(asof結合)...")
    dnf_rows2 = merge_hosei(dnf_rows)
    dnf_rows3 = merge_training(dnf_rows2)
    print(f"    完了 ({time.time()-t0:.0f}s)")

    print("[6] kako5_*/hist_same_*(対象馬ごと)...")
    kako5_feats = kako5_for_target_rows(full, dnf_rows_raw[[COL_RID16, COL_BAN]])
    print(f"    完了 ({time.time()-t0:.0f}s)")

    dnf_full = dnf_rows3.merge(kako5_feats, on=[COL_RID16, COL_BAN], how="left")
    print(f"    最終DNF特徴行 = {len(dnf_full):,}")

    print("[7] モデルロード & 全DNF行スコアリング...")
    bundle = joblib.load(MODEL_PKL)
    model, feats, encs = bundle["model"], bundle["feature_cols"], bundle["encoders"]
    X_dnf = apply_encoders(dnf_full.reindex(columns=feats), encs)[feats].apply(
        pd.to_numeric, errors="coerce").fillna(-9999).values
    dnf_full["_score"] = model.predict(X_dnf)
    dnf_full["year"] = dnf_full[COL_RID16].astype(str).str[:4].astype(int)
    print(f"    完了 ({time.time()-t0:.0f}s)")

    print("[8] master_v2(finisher)を全件スコアリング...")
    v2 = pd.read_csv(MASTER_V2, encoding="utf-8-sig", low_memory=False)
    X_v2 = apply_encoders(v2.reindex(columns=feats), encs)[feats].apply(
        pd.to_numeric, errors="coerce").fillna(-9999).values
    v2["_score"] = model.predict(X_v2)
    v2["year"] = v2[COL_RID16].astype(str).str[:4].astype(int)
    v2[COL_JYUN] = pd.to_numeric(v2[COL_JYUN], errors="coerce")
    print(f"    完了 ({time.time()-t0:.0f}s)")

    print("[9] レース単位でfinisher-only / DNF-inclusive 指標を計算...")
    dnf_by_race = dnf_full.groupby(COL_RID16)
    finisher_only_metrics, dnf_inclusive_metrics = [], []
    year_map = v2.drop_duplicates(COL_RID16).set_index(COL_RID16)["year"]

    for rid16, g in v2.groupby(COL_RID16):
        if len(g) < 3:
            continue
        race_has_dnf = rid16 in dnf_by_race.groups  # レース単位の属性(どちらの視点でも同じ値)

        scores_fin = g["_score"].values
        jyun_fin = g[COL_JYUN].values
        is_dnf_fin = np.zeros(len(g), dtype=bool)
        m_fin = race_metrics(scores_fin, jyun_fin, is_dnf_fin)
        m_fin["has_dnf"] = race_has_dnf
        m_fin["year"] = int(year_map.get(rid16))
        m_fin["rid16"] = rid16
        finisher_only_metrics.append(m_fin)

        if race_has_dnf:
            dg = dnf_by_race.get_group(rid16)
            scores_all = np.concatenate([scores_fin, dg["_score"].values])
            jyun_all = np.concatenate([jyun_fin, np.full(len(dg), np.nan)])
            is_dnf_all = np.concatenate([is_dnf_fin, np.ones(len(dg), dtype=bool)])
        else:
            scores_all, jyun_all, is_dnf_all = scores_fin, jyun_fin, is_dnf_fin
        m_all = race_metrics(scores_all, jyun_all, is_dnf_all)
        m_all["has_dnf"] = race_has_dnf
        m_all["year"] = int(year_map.get(rid16))
        m_all["rid16"] = rid16
        dnf_inclusive_metrics.append(m_all)

    print(f"    完了 ({time.time()-t0:.0f}s)  races={len(finisher_only_metrics):,}")

    print("[10] 集計 + bootstrap CI...")
    result = {
        "method_note": "既存baselineの計測訂正(研究の再探索ではない)。current v6の"
                       "モデル・ハイパーパラメータは一切変更していない。bootstrapは"
                       "race単位リサンプリング(meeting-day単位ではない、N=1000)。",
        "overall": {
            "finisher_only": aggregate(finisher_only_metrics),
            "dnf_inclusive": aggregate(dnf_inclusive_metrics),
        },
        "overall_bootstrap_ci": {
            "finisher_only": {k: bootstrap_ci(finisher_only_metrics, k)
                               for k in ("hon_win", "hon_top3", "ndcg5", "logloss", "brier")},
            "dnf_inclusive": {k: bootstrap_ci(dnf_inclusive_metrics, k)
                               for k in ("hon_win", "hon_top3", "ndcg5", "logloss", "brier")},
        },
        "by_year": {},
        "dnf_race_vs_no_dnf_race": {
            "no_dnf_races_finisher_only": aggregate(
                [m for m in finisher_only_metrics if not m["has_dnf"]]),
            "dnf_races_finisher_only_view": aggregate(
                [m for m in finisher_only_metrics if m["has_dnf"]]),
            "dnf_races_dnf_inclusive_view": aggregate(
                [m for m in dnf_inclusive_metrics if m["has_dnf"]]),
        },
        "n_dnf_rows_total": int(len(dnf_rows_raw)),
        "n_dnf_rows_feature_built": int(len(dnf_full)),
        "n_races_total": len(finisher_only_metrics),
        "n_races_with_dnf": int(sum(1 for m in finisher_only_metrics if m["has_dnf"])),
    }
    years = sorted(set(m["year"] for m in finisher_only_metrics))
    for y in years:
        fo = [m for m in finisher_only_metrics if m["year"] == y]
        di = [m for m in dnf_inclusive_metrics if m["year"] == y]
        result["by_year"][str(y)] = {
            "finisher_only": aggregate(fo),
            "dnf_inclusive": aggregate(di),
            "delta_dnf_inclusive_minus_finisher_only": {
                "hon_win_rate_pt": (aggregate(di).get("hon_win_rate", 0) - aggregate(fo).get("hon_win_rate", 0)) * 100,
                "hon_top3_rate_pt": (aggregate(di).get("hon_top3_rate", 0) - aggregate(fo).get("hon_top3_rate", 0)) * 100,
                "ndcg5": aggregate(di).get("ndcg5", 0) - aggregate(fo).get("ndcg5", 0),
                "logloss": aggregate(di).get("logloss", 0) - aggregate(fo).get("logloss", 0),
                "brier": aggregate(di).get("brier", 0) - aggregate(fo).get("brier", 0),
            },
        }

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / "dnf_inclusive_baseline.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "by_year"}, ensure_ascii=False, indent=1))
    print(f"\n[saved] {out_path}")
    print(f"TOTAL TIME: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
