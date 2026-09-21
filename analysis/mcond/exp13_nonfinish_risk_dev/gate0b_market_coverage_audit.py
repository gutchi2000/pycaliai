# -*- coding: utf-8 -*-
"""
Gate 0B（EXP13再開） — market coverage差の監査（読み取り専用）
==========================================================================
2026-09-22、ユーザー指示によるEXP13 Gate 0B再開作業の一部。
2023年flat started母集団（止馬=DNF含む、外・消は除外）について、
historical_pre_snapshot(TANPUK由来、発走26-30分前)のcoverage差
(正常完走馬91.0% vs 止馬87.6%)を、venue別・月別・人気帯別・snapshotファイル別・
オッズ帯別・欠損理由別に分解する。final oddsによる穴埋めは行わない
（欠損はそのまま欠損として扱う）。

出力: analysis/mcond/exp13_nonfinish_risk_dev/out/gate0b_coverage_breakdown.json
実行: venv311\\Scripts\\python.exe -m analysis.mcond.exp13_nonfinish_risk_dev.gate0b_market_coverage_audit
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(BASE))
from analysis.market_provenance_audit import (  # noqa: E402
    load_kekka_labels, load_tanpuk_interim, load_post_times, _to_ts,
    bucket_minutes, ODIR, KEKKA_EXT, MASTER_V2,
)

OUT_DIR = Path(__file__).resolve().parent / "out"


def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    adj = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return ((centre - adj) / denom, (centre + adj) / denom)


def diff_ci_normal_approx(k1, n1, k2, n2, z: float = 1.96):
    """2群比率差(p1-p2)のCI（正規近似、独立2標本）。"""
    p1, p2 = k1 / n1, k2 / n2
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    d = p1 - p2
    return d, (d - z * se, d + z * se)


def build_full_population(year: int = 2023) -> pd.DataFrame:
    """2023年flat started母集団(止含む・外消除外)にhistorical_pre_snapshot情報・
    確定オッズ(記述統計専用)・レース内人気帯を結合する。"""
    kk = load_kekka_labels(year, year)
    kk = kk[kk["started"]].copy()

    interim = load_tanpuk_interim(year, year).reset_index(drop=True)
    interim["snap_ts"] = _to_ts(interim["snap_mmddhhmm"].to_numpy(),
                                 interim["year"].to_numpy()).to_numpy()
    post = load_post_times()
    interim = interim.merge(post, on="rid16", how="left")
    interim["minutes_to_post"] = (interim["post_ts"] - interim["snap_ts"]).dt.total_seconds() / 60.0

    full = kk.merge(
        interim[["rid16", "ban", "minutes_to_post"]],
        left_on=["rid16", "ban_i"], right_on=["rid16", "ban"], how="left",
    )
    full["bucket"] = full["minutes_to_post"].apply(bucket_minutes)
    full["has_snapshot"] = full["bucket"] == "w26_30_historical_pre_snapshot"

    # 欠損理由の分類: (a) TANPUKにそもそも当該馬番のレコードが1件もない
    #                (b) レコードはあるが同日最新値が26-30分の窓に入らない(別窓)
    has_any_tanpuk = interim[["rid16", "ban"]].drop_duplicates().rename(
        columns={"ban": "ban_any"})
    has_any_tanpuk["_has_any"] = True
    full = full.merge(has_any_tanpuk, left_on=["rid16", "ban_i"],
                       right_on=["rid16", "ban_any"], how="left")
    full["_has_any"] = full["_has_any"].fillna(False)
    full = full.drop(columns=["ban_any"])

    def missing_reason(row):
        if row["has_snapshot"]:
            return "not_missing"
        if not row["_has_any"]:
            return "no_tanpuk_record_at_all"
        return "record_exists_outside_26_30_window"

    full["missing_reason"] = full.apply(missing_reason, axis=1)

    # 確定オッズ(記述統計専用、モデル入力ではない) → レース内人気帯
    kk_odds = pd.read_csv(KEKKA_EXT, encoding="utf-8-sig",
                          usecols=["日付", "race_id", "umaban", "単勝オッズ", "場所"], dtype=str)
    kk_odds["odds_f"] = pd.to_numeric(kk_odds["単勝オッズ"], errors="coerce")
    kk_odds["ban_i"] = pd.to_numeric(kk_odds["umaban"], errors="coerce")
    kk_odds = kk_odds.dropna(subset=["ban_i"])
    kk_odds["ban_i"] = kk_odds["ban_i"].astype(int)
    kk_odds["rid16"] = kk_odds["race_id"].astype(str)
    kk_odds = kk_odds[kk_odds["rid16"].str.len() == 16]
    full = full.merge(kk_odds[["rid16", "ban_i", "odds_f", "場所"]], on=["rid16", "ban_i"], how="left")

    full["ninki_rank"] = full.groupby("rid16")["odds_f"].rank(method="min", ascending=True)

    def ninki_band(r):
        if pd.isna(r):
            return "unknown"
        if r == 1:
            return "1"
        if r <= 3:
            return "2-3"
        if r <= 6:
            return "4-6"
        if r <= 9:
            return "7-9"
        return "10+"

    full["ninki_band"] = full["ninki_rank"].apply(ninki_band)

    def odds_band(o):
        if pd.isna(o):
            return "unknown"
        if o < 2:
            return "<2"
        if o < 5:
            return "2-5"
        if o < 10:
            return "5-10"
        if o < 30:
            return "10-30"
        if o < 100:
            return "30-100"
        return "100+"

    full["odds_band"] = full["odds_f"].apply(odds_band)
    full["month"] = full["rid16"].str[4:6]

    def snapshot_file(rid16: str) -> str:
        y = int(rid16[:4])
        if 2011 <= y <= 2015:
            return "TANPUK_2011_2015"
        if 2016 <= y <= 2020:
            return "TANPUK_2016_2020"
        if 2021 <= y <= 2025:
            return "TANPUK_2021_2025"
        return "other"

    full["snapshot_file"] = full["rid16"].apply(snapshot_file)
    return full


def breakdown(full: pd.DataFrame, by: str) -> dict:
    out = {}
    for (dnf, grp), sub in full.groupby(["is_dnf", by]):
        n = len(sub)
        cov = int(sub["has_snapshot"].sum())
        out.setdefault(str(grp), {})[("DNF" if dnf else "finisher")] = {
            "n": n, "covered": cov, "coverage_pct": round(cov / n * 100, 2) if n else None,
        }
    return out


def missingness_predictability(full: pd.DataFrame) -> dict:
    """missing snapshot(=has_snapshot反転)がis_dnf/venue/ninki_bandから予測できるか。
    ロジスティック回帰の係数とAUC、および単純比率差の検定(chi2)で見る。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    d = full.copy()
    d["missing"] = (~d["has_snapshot"]).astype(int)
    d = pd.get_dummies(d, columns=["場所", "ninki_band"], drop_first=True)
    feat_cols = ["is_dnf"] + [c for c in d.columns if c.startswith("場所_") or c.startswith("ninki_band_")]
    d[feat_cols] = d[feat_cols].astype(float)
    X = d[feat_cols].values
    y = d["missing"].values
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    pred = clf.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, pred)

    # 単純chi2: is_dnf x missing の2x2
    ct = pd.crosstab(full["is_dnf"], ~full["has_snapshot"])
    chi2, p, dof, exp = stats.chi2_contingency(ct)

    return {
        "logistic_auc_missing_vs_isdnf_venue_ninkiband": round(float(auc), 4),
        "interpretation": (
            "AUC~0.5ならmissingnessはほぼ予測不能(ランダム)、"
            "AUC>>0.5なら構造的欠損(選択バイアスの疑い)"
        ),
        "chi2_is_dnf_vs_missing": {
            "chi2": round(float(chi2), 4), "p_value": round(float(p), 6), "dof": int(dof),
        },
        "crosstab_is_dnf_vs_missing": ct.to_dict(),
    }


def main() -> int:
    full = build_full_population(2023)
    started = full  # already filtered to started=1

    n_fin = int((~started["is_dnf"]).sum())
    n_dnf = int(started["is_dnf"].sum())
    cov_fin = int((~started["is_dnf"] & started["has_snapshot"]).sum())
    cov_dnf = int((started["is_dnf"] & started["has_snapshot"]).sum())

    excluded_dnf_positive = n_dnf - cov_dnf
    excluded_finisher = n_fin - cov_fin

    diff, (lo, hi) = diff_ci_normal_approx(cov_fin, n_fin, cov_dnf, n_dnf)

    complete_case = started[started["has_snapshot"]]
    complete_case_pos_rate = float(complete_case["is_dnf"].mean())
    all_starter_pos_rate = float(started["is_dnf"].mean())

    result = {
        "population": "2023 flat started (止含む、外・消除外)、race_id結合",
        "n_finisher": n_fin, "n_dnf": n_dnf,
        "coverage_finisher": {"n": n_fin, "covered": cov_fin, "pct": round(cov_fin / n_fin * 100, 2)},
        "coverage_dnf": {"n": n_dnf, "covered": cov_dnf, "pct": round(cov_dnf / n_dnf * 100, 2)},
        "excluded_dnf_positive_count": excluded_dnf_positive,
        "excluded_finisher_count": excluded_finisher,
        "complete_case_positive_rate_pct": round(complete_case_pos_rate * 100, 4),
        "all_starter_positive_rate_pct": round(all_starter_pos_rate * 100, 4),
        "coverage_diff_finisher_minus_dnf": round(diff * 100, 2),
        "coverage_diff_95ci_pct": [round(lo * 100, 2), round(hi * 100, 2)],
        "coverage_diff_ci_crosses_zero": bool(lo <= 0 <= hi),
        "dnf_timing_note": (
            "止(中止)はレース開始後に発生するため、発走前のhistorical_pre_snapshot"
            "は定義上すべてDNF発生前の観測である。「DNF発生前後」という軸は"
            "本データでは自明(常にpre)であり、区別する意味のある分割にならない。"
        ),
        "by_venue": breakdown(started, "場所"),
        "by_month": breakdown(started, "month"),
        "by_ninki_band_descriptive_only": breakdown(started, "ninki_band"),
        "by_odds_band_descriptive_only": breakdown(started, "odds_band"),
        "by_snapshot_file": breakdown(started, "snapshot_file"),
        "by_missing_reason": started.groupby(["is_dnf", "missing_reason"]).size()
                                     .rename("n").reset_index()
                                     .to_dict(orient="records"),
        "missingness_predictability": missingness_predictability(started),
        "final_odds_backfill_policy": "行っていない。欠損はhas_snapshot=Falseのまま保持。",
    }

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / "gate0b_coverage_breakdown.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[saved] {out_path}")
    print(json.dumps({k: v for k, v in result.items()
                       if k not in ("by_venue", "by_month", "by_ninki_band_descriptive_only",
                                     "by_odds_band_descriptive_only", "by_snapshot_file",
                                     "by_missing_reason", "missingness_predictability")},
                      ensure_ascii=False, indent=1))
    print(json.dumps(result["missingness_predictability"], ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
