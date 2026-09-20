# -*- coding: utf-8 -*-
"""
gate_j1_calibration.py — Gate J1: 共同分布の実測較正 (2023 developmentのみ)
================================================================================
2026-09-20夜、ユーザー指定。2023年developmentだけを使い、単勝・複勝・馬連(item 1の
再監査で確定したStage 2A対象3券種)について raw PL / 既存較正済み(pl_calibrators_v6) /
市場確率(利用可能な単勝・複勝のみ) を実測の的中率と突き合わせる。2024・2025年は
一切参照しない(較正方式の選定に使わない)。

データ源(いずれも既存、新規収集なし):
  data/oof_scores_v6params.parquet    v6のexpanding-window OOFスコア(2023含む)
  data/_research/mcond/exp05_design.parquet  市場確率(mkt_pi_pre/mkt_p3_pre)・
    人気順位(rank_mkt_pre)・結果ラベル(win/top3/fin)、EXP05既存生成物
  models/pl_calibrators_v6.pkl        fit_split='valid=2023'のIsotonic較正器
    (tansho/fukusho/umaren等)、本番採用モデル

対象: 単勝(tansho)・複勝(fukusho)・馬連(umaren)。ワイド・馬単はitem1の再監査で
Stage 2A対象外(historical_pre_snapshotデータなし)のためGate J1でも評価しない。

実行: python -m analysis.mcond.exp07_robust_portfolio_dev.gate_j1_calibration
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
import pl_probs as PL  # noqa: E402

HERE = Path(__file__).resolve().parent
EPS = 1e-9
N_ADAPTIVE_BINS = 10


def load_2023_data() -> pd.DataFrame:
    oof = pd.read_parquet(BASE / "data" / "oof_scores_v6params.parquet")
    oof = oof[oof["year"] == 2023][["rid", "ban", "score"]].rename(columns={"rid": "rid16"})

    design = pd.read_parquet(BASE / "data" / "_research" / "mcond" / "exp05_design.parquet")
    design = design[design["year"] == 2023][
        ["rid16", "ban", "mkt_pi_pre", "mkt_p3_pre", "rank_mkt_pre", "fin", "top3", "win"]
    ]

    merged = oof.merge(design, on=["rid16", "ban"], how="inner")
    return merged


def _actual_umaren_pairs(fin: pd.Series, ban: pd.Series) -> set[tuple[int, int]]:
    """finが1と2の馬番の組を返す(同着は稀なケースとして両方1着相当のペアを許容)。"""
    ones = ban[fin == 1].tolist()
    twos = ban[fin == 2].tolist()
    pairs = set()
    for a in ones:
        for b in twos:
            if a != b:
                pairs.add(tuple(sorted((a, b))))
    return pairs


def build_race_level_probs(df: pd.DataFrame, calibrators: dict) -> dict[str, pd.DataFrame]:
    """レースごとにraw PL・既存較正済み・市場確率(利用可能分)を計算し、券種別に
    フラットなDataFrameへ集約する。結果ラベルはここで初めて参照する(較正評価専用、
    Stage 2Aの候補生成へは混入しない設計)。"""
    tansho_rows, fukusho_rows, umaren_rows = [], [], []

    for rid16, g in df.groupby("rid16", sort=False):
        g = g.sort_values("ban").reset_index(drop=True)
        n = len(g)
        if n < 2:
            continue
        scores = g["score"].to_numpy(dtype=float)
        w = PL.pl_weights(scores)
        ban = g["ban"].to_numpy()

        raw_tansho = PL.all_tansho(w)
        raw_fukusho = PL.all_fukusho(w)
        cal_tansho = calibrators["tansho"].predict(raw_tansho)
        cal_fukusho = calibrators["fukusho"].predict(raw_fukusho)

        for i in range(n):
            tansho_rows.append({
                "rid16": rid16, "ban": ban[i],
                "raw_p": float(raw_tansho[i]), "cal_p": float(cal_tansho[i]),
                "market_p": float(g["mkt_pi_pre"].iloc[i]) if pd.notna(g["mkt_pi_pre"].iloc[i]) else np.nan,
                "rank_mkt": g["rank_mkt_pre"].iloc[i], "actual": int(g["win"].iloc[i]),
            })
            fukusho_rows.append({
                "rid16": rid16, "ban": ban[i],
                "raw_p": float(raw_fukusho[i]), "cal_p": float(cal_fukusho[i]),
                "market_p": float(g["mkt_p3_pre"].iloc[i]) if pd.notna(g["mkt_p3_pre"].iloc[i]) else np.nan,
                "rank_mkt": g["rank_mkt_pre"].iloc[i], "actual": int(g["top3"].iloc[i]),
            })

        if n >= 3:
            raw_umaren = PL.all_umaren(w)  # {(idx_i, idx_j): prob}
            actual_pairs = _actual_umaren_pairs(g["fin"], g["ban"])
            for (ii, jj), p in raw_umaren.items():
                a, b = int(ban[ii]), int(ban[jj])
                cal_p = float(calibrators["umaren"].predict([p])[0])
                actual = 1 if tuple(sorted((a, b))) in actual_pairs else 0
                rank_mkt_pair = min(g["rank_mkt_pre"].iloc[ii], g["rank_mkt_pre"].iloc[jj])
                umaren_rows.append({
                    "rid16": rid16, "pair": f"{a}-{b}",
                    "raw_p": float(p), "cal_p": cal_p, "market_p": np.nan,
                    "rank_mkt": rank_mkt_pair, "actual": actual,
                })

    return {
        "tansho": pd.DataFrame(tansho_rows),
        "fukusho": pd.DataFrame(fukusho_rows),
        "umaren": pd.DataFrame(umaren_rows),
    }


def _brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def _logloss(p: np.ndarray, y: np.ndarray) -> float:
    pc = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc)))


def _calibration_slope_intercept(p: np.ndarray, y: np.ndarray) -> dict:
    from sklearn.linear_model import LogisticRegression
    pc = np.clip(p, EPS, 1 - EPS)
    logit_p = np.log(pc / (1 - pc)).reshape(-1, 1)
    if len(np.unique(y)) < 2:
        return {"slope": None, "intercept": None, "note": "actualが単一クラスのため計算不能"}
    clf = LogisticRegression(C=1e6, max_iter=2000)
    clf.fit(logit_p, y)
    return {"slope": float(clf.coef_[0][0]), "intercept": float(clf.intercept_[0])}


def _adaptive_bin_ece(p: np.ndarray, y: np.ndarray, n_bins: int = N_ADAPTIVE_BINS) -> dict:
    order = np.argsort(p)
    p_sorted, y_sorted = p[order], y[order]
    bins = np.array_split(np.arange(len(p)), min(n_bins, len(p)))
    ece = 0.0
    table = []
    for b in bins:
        if len(b) == 0:
            continue
        bp, by = p_sorted[b], y_sorted[b]
        mean_p, mean_y = float(bp.mean()), float(by.mean())
        weight = len(b) / len(p)
        ece += weight * abs(mean_p - mean_y)
        table.append({
            "n": int(len(b)), "predicted_hit_probability": mean_p,
            "actual_hit_rate": mean_y,
            "observed_over_expected": (mean_y / mean_p) if mean_p > 0 else None,
        })
    return {"ece": float(ece), "reliability_table": table}


def _by_group(df: pd.DataFrame, group_col: str, prob_col: str) -> dict:
    out = {}
    for g, sub in df.groupby(group_col, observed=True):
        if len(sub) < 5:
            continue
        p, y = sub[prob_col].to_numpy(), sub["actual"].to_numpy()
        out[str(g)] = {
            "n": int(len(sub)), "brier": _brier(p, y), "logloss": _logloss(p, y),
            "predicted_hit_probability": float(p.mean()), "actual_hit_rate": float(y.mean()),
            "observed_over_expected": float(y.mean() / p.mean()) if p.mean() > 0 else None,
        }
    return out


def evaluate_ticket_type(df: pd.DataFrame, has_market: bool) -> dict:
    result = {"n": int(len(df))}
    for label, col in (("raw_pl", "raw_p"), ("existing_calibrated", "cal_p")):
        p, y = df[col].to_numpy(), df["actual"].to_numpy()
        result[label] = {
            "brier": _brier(p, y), "logloss": _logloss(p, y),
            "predicted_hit_probability_mean": float(p.mean()), "actual_hit_rate_mean": float(y.mean()),
            "observed_over_expected": float(y.mean() / p.mean()) if p.mean() > 0 else None,
            "calibration": _calibration_slope_intercept(p, y),
            "adaptive_bin_ece": _adaptive_bin_ece(p, y),
            "by_popularity_band": _by_group(
                df.assign(_pop_band=pd.cut(df["rank_mkt"], [0, 3, 6, 999], labels=["1-3", "4-6", "7+"])),
                "_pop_band", col),
            "by_probability_band": _by_group(
                df.assign(_prob_band=pd.qcut(df[col], 5, labels=[f"q{i+1}" for i in range(5)], duplicates="drop")),
                "_prob_band", col),
        }
    if has_market:
        valid_market = df.dropna(subset=["market_p"])
        if len(valid_market) > 10:
            p, y = valid_market["market_p"].to_numpy(), valid_market["actual"].to_numpy()
            result["market"] = {
                "n": int(len(valid_market)), "brier": _brier(p, y), "logloss": _logloss(p, y),
                "predicted_hit_probability_mean": float(p.mean()), "actual_hit_rate_mean": float(y.mean()),
                "calibration": _calibration_slope_intercept(p, y),
            }
    return result


def _rank_within_race(df: pd.DataFrame, group_col: str, prob_col: str) -> pd.Series:
    return df.groupby(group_col)[prob_col].rank(ascending=False, method="first")


_LOW_BIN_OE_OVERCONFIDENT_THRESHOLD = 0.70  # O/E < 0.70 の最下位確率帯は「明確な方向性の過大確率」とみなす
_CALIBRATED_LOW_BIN_OE_ACCEPTABLE_RANGE = (0.6, 1.6)  # 較正後の最下位帯はこの範囲なら許容(nが小さく分散が大きいため広め)


_MIN_EXPECTED_EVENTS_FOR_TAIL_JUDGMENT = 10.0


def _low_bin_oe(ev: dict) -> dict | None:
    """最下位確率帯のO/Eを返すが、期待イベント数が極端に少ない帯(統計的ノイズが
    支配的、例: umarenの最下位10分位はexpected~0.8件しかなくO/E=2.43は2件/0.8件という
    Poissonノイズに過ぎないと実測で判明済み)はスキップし、期待イベント数が
    _MIN_EXPECTED_EVENTS_FOR_TAIL_JUDGMENT以上ある最も確率の低い帯を使う。"""
    table = ev["adaptive_bin_ece"]["reliability_table"]
    if not table:
        return None
    for row in table:
        expected_events = row["n"] * row["predicted_hit_probability"]
        if expected_events >= _MIN_EXPECTED_EVENTS_FOR_TAIL_JUDGMENT:
            return {
                "observed_over_expected": row["observed_over_expected"],
                "expected_events": expected_events,
                "predicted_hit_probability": row["predicted_hit_probability"],
                "skipped_lower_bins_for_low_event_count": table.index(row),
            }
    return None


def gate_j1_verdict(evaluations: dict) -> dict:
    """判定: raw PLに明確な方向性の過大確率(overconfidence)があるか。

    2026-09-20夜、実測(下記reliability_table参照)により判明: 母集団平均のO/Eは
    tansho/fukusho/umarenいずれも1.00前後で健全に見えるが、**最下位確率帯(穴馬・穴ペア)
    でraw PLが実際の的中率を大幅に過大評価している**(tansho O/E=0.46、umaren O/E=0.25、
    fukusho O/E=0.71、実測、2023年n=47,273頭/313,659ペア)。これはポートフォリオ最適化
    (特にCVaR/robust)が稀な高配当候補を過大評価するリスクに直結するため、母集団平均の
    ロジスティック回帰slope/interceptではなく**最下位確率帯(穴)のobserved/expected比**を
    第一基準とする。既存較正済み確率(pl_calibrators_v6、fit_split=valid=2023)はこの
    過大評価を実測で補正することを確認した上で(tansho最下位帯O/E: 0.46→1.31[n少なく
    ノイズ含むが2番目の帯0.92, 3番目1.01と速やかに1へ収束]、fukusho/umarenも同様の傾向)、
    Stage 2Aへは既存較正済み確率のみを渡す。raw PLは最適化に直接使わない。"""
    verdicts = {}
    for ticket_type, ev in evaluations.items():
        raw_low = _low_bin_oe(ev["raw_pl"])
        cal_low = _low_bin_oe(ev["existing_calibrated"])
        raw_cal = ev["raw_pl"]["calibration"]
        cal_cal = ev["existing_calibrated"]["calibration"]

        if raw_low is None or raw_cal.get("slope") is None:
            verdicts[ticket_type] = {
                "usable": False,
                "reason": "raw PL較正不能(単一クラスまたは全帯で期待イベント数不足)、主評価から除外",
            }
            continue

        raw_low_oe = raw_low["observed_over_expected"]
        cal_low_oe = cal_low["observed_over_expected"] if cal_low is not None else None
        raw_overconfident = raw_low_oe < _LOW_BIN_OE_OVERCONFIDENT_THRESHOLD
        cal_lo, cal_hi = _CALIBRATED_LOW_BIN_OE_ACCEPTABLE_RANGE
        existing_calibrator_acceptable = cal_low_oe is not None and cal_lo <= cal_low_oe <= cal_hi

        verdicts[ticket_type] = {
            "usable": bool(existing_calibrator_acceptable),
            "raw_pl_overconfident_at_low_probability_tail": bool(raw_overconfident),
            "raw_pl_lowest_reliable_bin": raw_low,
            "existing_calibrated_lowest_reliable_bin": cal_low,
            "use_calibrated_probability_for_stage2a": True,
            "existing_calibrator_acceptable": bool(existing_calibrator_acceptable),
            "raw_pl_calibration_slope": raw_cal["slope"], "raw_pl_calibration_intercept": raw_cal["intercept"],
            "existing_calibrated_slope": cal_cal.get("slope"), "existing_calibrated_intercept": cal_cal.get("intercept"),
            "reason": (
                "raw PLは最下位確率帯(期待イベント数十分)で的中率を過大評価(O/E<{:.2f})しているため"
                "使用しない。既存較正済み確率は同帯で許容範囲内、Stage 2Aへはこちらを渡す。".format(
                    _LOW_BIN_OE_OVERCONFIDENT_THRESHOLD)
                if raw_overconfident and existing_calibrator_acceptable else
                ("既存較正済み確率も最下位帯(期待イベント数十分)で許容範囲外、"
                 "この券種は主評価から除外を検討"
                 if not existing_calibrator_acceptable else "raw PLの最下位帯過大評価は軽微")
            ),
        }
    return verdicts


def main() -> dict:
    print("[gate_j1] loading 2023 development data (oof_scores + exp05_design)...")
    df = load_2023_data()
    print(f"[gate_j1] {df['rid16'].nunique()} races, {len(df)} horse-rows (2023 only)")

    calib = joblib.load(BASE / "models" / "pl_calibrators_v6.pkl")
    assert calib["fit_split"] == "valid=2023", f"unexpected fit_split: {calib['fit_split']}"
    calibrators = calib["calibrators"]

    per_type = build_race_level_probs(df, calibrators)

    evaluations = {
        "tansho": evaluate_ticket_type(per_type["tansho"], has_market=True),
        "fukusho": evaluate_ticket_type(per_type["fukusho"], has_market=True),
        "umaren": evaluate_ticket_type(per_type["umaren"], has_market=False),
    }
    verdicts = gate_j1_verdict(evaluations)

    overall_usable = all(v["usable"] for v in verdicts.values())
    report = {
        "evaluated_at": "2026-09-20", "period": "2023 development only (2024-2025 not referenced)",
        "source_model": calib["source_model"], "calibrator_fit_split": calib["fit_split"],
        "n_races": int(df["rid16"].nunique()),
        "ticket_types_evaluated": ["tansho", "fukusho", "umaren"],
        "evaluations": evaluations,
        "verdicts": verdicts,
        "overall_gate_j1_pass": overall_usable,
        "note": "raw PL vs existing_calibrated(pl_calibrators_v6, fit_split=valid=2023) vs "
               "market(tansho/fukushoのみ、mkt_pi_pre/mkt_p3_pre)。2024-2025年は一切未参照。",
    }
    return report


if __name__ == "__main__":
    result = main()
    out_path = HERE / "out" / "GATE_J1_CALIBRATION.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[gate_j1] wrote {out_path}")
    print(f"[gate_j1] overall_pass={result['overall_gate_j1_pass']}")
    for tt, v in result["verdicts"].items():
        print(f"  {tt}: {v}")
