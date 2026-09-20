# -*- coding: utf-8 -*-
"""
gate_j1_calibration.py — Gate J1: 共同分布の実測較正 (2023 developmentのみ、時系列安全版)
================================================================================
2026-09-20夜、ユーザー指定+追加指示で訂正。単勝・複勝・馬連(item 1の再監査で確定した
Stage 2A対象3券種)について、以下の**3種類を明確に分離して**報告する。2024・2025年は
一切参照しない(較正方式の選定に使わない)。

CALIBRATION_AUDIT.mdで確定した通り、models/pl_calibrators_v6.pkl は2023年全体
(valid split)でfitされているため、この較正器を2023年全体で評価すると較正器自体に
とってはin-sample評価になる(raw PLは v6 score自体がtrain<=2022のためOOSで問題ない)。
2022年での独立fitはv6モデル自身が2022を学習に使っているため実施不可能と判明したため、
仕様書提示の代替案「2023年を時系列blocked cross-fit」を採用する。

  (a) raw_pl_2023_oos              v6生スコア→PL確率(較正なし)を2023年全体の実的中と比較。
                                    スコア生成モデル(v6)がtrain<=2022のため真にOOS。
  (b) calibrator_h1fit_h2_oos       2023年前半(H1、01-05〜06-30)でIsotonic較正器を新規fit
                                    し、後半(H2、07-01〜12-28)へ時系列安全に適用・評価する。
                                    **Gate J1のPASS/FAIL判定はこの指標を使う。**
  (c) final_calibrator_insample_ref 本番のpl_calibrators_v6.pkl(2023年全体でfit)を同じ
                                    2023年全体で評価した参考値。in-sampleのため判定には
                                    使わない。Stage 2Aへはこの較正器(再fitしない)を渡す。
                                    2024・2025年の結果は一切使用していない
                                    (fit_split=valid=2023、artifact作成2026-05-20)。

実行: python -m analysis.mcond.exp07_robust_portfolio_dev.gate_j1_calibration
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
import pl_probs as PL  # noqa: E402

HERE = Path(__file__).resolve().parent
EPS = 1e-9
N_ADAPTIVE_BINS = 10
H1_H2_BOUNDARY = "20230701"  # 2023年前半/後半のblocked split境界(結果を見る前に固定)


def load_2023_data() -> pd.DataFrame:
    oof = pd.read_parquet(BASE / "data" / "oof_scores_v6params.parquet")
    oof = oof[oof["year"] == 2023][["rid", "ban", "score"]].rename(columns={"rid": "rid16"})

    design = pd.read_parquet(BASE / "data" / "_research" / "mcond" / "exp05_design.parquet")
    design = design[design["year"] == 2023][
        ["rid16", "ban", "date", "mkt_pi_pre", "mkt_p3_pre", "rank_mkt_pre", "fin", "top3", "win"]
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
    """レースごとにraw PL・既存較正済み(2023年全体fit較正器)・市場確率(利用可能分)を
    計算し、券種別にフラットなDataFrameへ集約する。dateも保持しH1/H2分割に使う。"""
    tansho_rows, fukusho_rows, umaren_rows = [], [], []

    for rid16, g in df.groupby("rid16", sort=False):
        g = g.sort_values("ban").reset_index(drop=True)
        n = len(g)
        if n < 2:
            continue
        scores = g["score"].to_numpy(dtype=float)
        w = PL.pl_weights(scores)
        ban = g["ban"].to_numpy()
        date = str(g["date"].iloc[0])

        raw_tansho = PL.all_tansho(w)
        raw_fukusho = PL.all_fukusho(w)
        cal_tansho = calibrators["tansho"].predict(raw_tansho)
        cal_fukusho = calibrators["fukusho"].predict(raw_fukusho)

        for i in range(n):
            tansho_rows.append({
                "rid16": rid16, "ban": ban[i], "date": date,
                "raw_p": float(raw_tansho[i]), "cal_p": float(cal_tansho[i]),
                "market_p": float(g["mkt_pi_pre"].iloc[i]) if pd.notna(g["mkt_pi_pre"].iloc[i]) else np.nan,
                "rank_mkt": g["rank_mkt_pre"].iloc[i], "actual": int(g["win"].iloc[i]),
            })
            fukusho_rows.append({
                "rid16": rid16, "ban": ban[i], "date": date,
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
                    "rid16": rid16, "pair": f"{a}-{b}", "date": date,
                    "raw_p": float(p), "cal_p": cal_p, "market_p": np.nan,
                    "rank_mkt": rank_mkt_pair, "actual": actual,
                })

    return {
        "tansho": pd.DataFrame(tansho_rows),
        "fukusho": pd.DataFrame(fukusho_rows),
        "umaren": pd.DataFrame(umaren_rows),
    }


def split_h1_h2(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """2023年を前半(H1、fit用)/後半(H2、OOS評価用)へ日付境界で分割する
    (ランダムfoldではなく時系列blocked split、仕様書追加指示2の要求通り)。"""
    h1 = df[df["date"] < H1_H2_BOUNDARY].copy()
    h2 = df[df["date"] >= H1_H2_BOUNDARY].copy()
    return h1, h2


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


def _evaluate_probability_column(df: pd.DataFrame, col: str) -> dict:
    p, y = df[col].to_numpy(), df["actual"].to_numpy()
    return {
        "n": int(len(df)),
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


def evaluate_market(df: pd.DataFrame) -> dict | None:
    valid_market = df.dropna(subset=["market_p"])
    if len(valid_market) <= 10:
        return None
    p, y = valid_market["market_p"].to_numpy(), valid_market["actual"].to_numpy()
    return {
        "n": int(len(valid_market)), "brier": _brier(p, y), "logloss": _logloss(p, y),
        "predicted_hit_probability_mean": float(p.mean()), "actual_hit_rate_mean": float(y.mean()),
        "calibration": _calibration_slope_intercept(p, y),
    }


def fit_h1_calibrator_and_eval_h2(df: pd.DataFrame) -> dict:
    """(b) calibrator_h1fit_h2_oos: H1でIsotonic較正器を新規fitしH2で時系列安全に評価する。
    2022年での独立fitが不可能(v6モデル自身がtrain<=2022を学習済み)なため採用した代替案。"""
    h1, h2 = split_h1_h2(df)
    if len(h1) < 50 or len(h2) < 50:
        return {"usable": False, "reason": f"H1({len(h1)})/H2({len(h2)})のいずれかがサンプル不足"}

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(h1["raw_p"].to_numpy(), h1["actual"].to_numpy())
    h2 = h2.copy()
    h2["h1fit_cal_p"] = iso.predict(h2["raw_p"].to_numpy())

    result = _evaluate_probability_column(h2, "h1fit_cal_p")
    result["h1_n"] = int(len(h1))
    result["h2_n"] = int(len(h2))
    result["h1_date_range"] = [str(h1["date"].min()), str(h1["date"].max())]
    result["h2_date_range"] = [str(h2["date"].min()), str(h2["date"].max())]
    result["usable"] = True
    return result


_LOW_BIN_OE_OVERCONFIDENT_THRESHOLD = 0.70  # O/E < 0.70 の最下位確率帯は「明確な方向性の過大確率」とみなす
_CALIBRATED_LOW_BIN_OE_ACCEPTABLE_RANGE = (0.6, 1.6)  # 較正後の最下位帯はこの範囲なら許容(nが小さく分散が大きいため広め)
_MIN_EXPECTED_EVENTS_FOR_TAIL_JUDGMENT = 10.0  # 2023 development内で決めた探索規則、2024/2025を見て変更しない


def _low_bin_oe(ev: dict) -> dict | None:
    """最下位確率帯のO/Eを返すが、期待イベント数が極端に少ない帯(統計的ノイズが
    支配的)はスキップし、期待イベント数が_MIN_EXPECTED_EVENTS_FOR_TAIL_JUDGMENT以上
    ある最も確率の低い帯を使う。"""
    table = ev.get("adaptive_bin_ece", {}).get("reliability_table")
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
    """判定: raw PLに明確な方向性の過大確率(overconfidence)があるか、それを
    **時系列安全な較正器評価(calibrator_h1fit_h2_oos)**が補正するか。

    2026-09-20夜、追加指示により訂正: 従来は本番較正器(pl_calibrators_v6、2023年全体で
    fit)を同じ2023年全体で評価しており、較正器自体にとってin-sample評価だった。
    H1(前半)でfitしH2(後半)で評価する時系列blocked splitへ切り替えた。
    PASS/FAIL判定はcalibrator_h1fit_h2_oosの最下位確率帯(期待イベント数十分)の
    observed/expected比を使う。final_calibrator_insample_refは参考記録のみで
    判定に使わない。"""
    verdicts = {}
    for ticket_type, ev in evaluations.items():
        raw_low = _low_bin_oe(ev["raw_pl_2023_oos"])
        h1h2 = ev["calibrator_h1fit_h2_oos"]
        raw_cal = ev["raw_pl_2023_oos"]["calibration"]

        if raw_low is None or raw_cal.get("slope") is None:
            verdicts[ticket_type] = {
                "usable": False,
                "reason": "raw PL較正不能(単一クラスまたは全帯で期待イベント数不足)、主評価から除外",
            }
            continue

        if not h1h2.get("usable", False):
            verdicts[ticket_type] = {
                "usable": False,
                "reason": f"H1/H2 blocked split評価が不能({h1h2.get('reason', '不明')})、主評価から除外",
            }
            continue

        h1h2_low = _low_bin_oe(h1h2)
        raw_low_oe = raw_low["observed_over_expected"]
        h1h2_low_oe = h1h2_low["observed_over_expected"] if h1h2_low is not None else None
        raw_overconfident = raw_low_oe < _LOW_BIN_OE_OVERCONFIDENT_THRESHOLD
        cal_lo, cal_hi = _CALIBRATED_LOW_BIN_OE_ACCEPTABLE_RANGE
        h1h2_acceptable = h1h2_low_oe is not None and cal_lo <= h1h2_low_oe <= cal_hi

        verdicts[ticket_type] = {
            "usable": bool(h1h2_acceptable),
            "raw_pl_overconfident_at_low_probability_tail": bool(raw_overconfident),
            "raw_pl_lowest_reliable_bin": raw_low,
            "calibrator_h1fit_h2_oos_lowest_reliable_bin": h1h2_low,
            "use_calibrated_probability_for_stage2a": True,
            "h1fit_h2_oos_calibrator_acceptable": bool(h1h2_acceptable),
            "raw_pl_calibration_slope": raw_cal["slope"], "raw_pl_calibration_intercept": raw_cal["intercept"],
            "h1fit_h2_oos_slope": h1h2["calibration"].get("slope"),
            "h1fit_h2_oos_intercept": h1h2["calibration"].get("intercept"),
            "reason": (
                "raw PLは最下位確率帯(期待イベント数十分)で的中率を過大評価(O/E<{:.2f})しているが、"
                "H1fit較正器をH2へ適用した時系列安全な評価では同帯が許容範囲内。".format(
                    _LOW_BIN_OE_OVERCONFIDENT_THRESHOLD)
                if raw_overconfident and h1h2_acceptable else
                ("H1fit較正器のH2適用でも最下位帯(期待イベント数十分)が許容範囲外、"
                 "この券種は主評価から除外を検討"
                 if not h1h2_acceptable else "raw PLの最下位帯過大評価は軽微、較正の必要性は小さい")
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

    evaluations = {}
    for ticket_type, has_market in (("tansho", True), ("fukusho", True), ("umaren", False)):
        sub = per_type[ticket_type]
        entry = {
            "raw_pl_2023_oos": _evaluate_probability_column(sub, "raw_p"),
            "calibrator_h1fit_h2_oos": fit_h1_calibrator_and_eval_h2(sub),
            "final_calibrator_insample_ref": _evaluate_probability_column(sub, "cal_p"),
        }
        if has_market:
            market_ev = evaluate_market(sub)
            if market_ev is not None:
                entry["market_2023_reference"] = market_ev
        evaluations[ticket_type] = entry

    verdicts = gate_j1_verdict(evaluations)

    stage2a_ticket_scope = [tt for tt, v in verdicts.items() if v.get("usable")]
    excluded_ticket_types = {
        tt: v.get("reason") for tt, v in verdicts.items() if not v.get("usable")
    }
    overall_usable = len(stage2a_ticket_scope) > 0
    report = {
        "evaluated_at": "2026-09-20夜(時系列安全版、追加指示反映)",
        "period": "2023 development only (2024-2025 not referenced)",
        "h1_h2_boundary": H1_H2_BOUNDARY,
        "source_model": calib["source_model"], "final_calibrator_fit_split": calib["fit_split"],
        "final_calibrator_note": "本番pl_calibrators_v6.pkl(2023年全体でfit、artifact作成2026-05-20、"
                                 "2024-2025年の結果は一切使用していない)。Stage 2Aへはこの較正器を"
                                 "再fitせずそのまま渡す。final_calibrator_insample_refはこの較正器を"
                                 "fit対象と同じ2023年全体で評価した参考値でPASS/FAIL判定には使わない。",
        "n_races": int(df["rid16"].nunique()),
        "ticket_types_evaluated": ["tansho", "fukusho", "umaren"],
        "evaluations": evaluations,
        "verdicts": verdicts,
        "stage2a_ticket_scope": stage2a_ticket_scope,
        "excluded_ticket_types": excluded_ticket_types,
        "overall_gate_j1_pass": overall_usable,
        "overall_gate_j1_pass_note": "少なくとも1券種がusable=Trueであれば全体PASSとし、"
                                     "Stage 2Aの対象券種はstage2a_ticket_scopeへ絞る"
                                     "(仕様書追加指示2『候補馬券領域で較正不能な券種は"
                                     "主評価から除外』を適用)。",
        "note": "(a)raw_pl_2023_oos: v6スコアがtrain<=2022のため真にOOS。"
               "(b)calibrator_h1fit_h2_oos: H1(01-05~06-30)でfitしH2(07-01~12-28)で評価、"
               "PASS/FAIL判定に使う唯一の指標。"
               "(c)final_calibrator_insample_ref: 本番較正器を2023年全体で評価した参考値、in-sampleのため判定に使わない。"
               "2024-2025年は一切未参照。",
    }
    return report


if __name__ == "__main__":
    result = main()
    out_path = HERE / "out" / "GATE_J1_CALIBRATION.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[gate_j1] wrote {out_path}")
    print(f"[gate_j1] overall_pass={result['overall_gate_j1_pass']}")
    print(f"[gate_j1] stage2a_ticket_scope={result['stage2a_ticket_scope']}")
    print(f"[gate_j1] excluded_ticket_types={result['excluded_ticket_types']}")
    for tt, v in result["verdicts"].items():
        print(f"  {tt}: usable={v.get('usable')} reason={v.get('reason')}")
