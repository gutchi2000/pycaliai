# -*- coding: utf-8 -*-
"""
stage2a_run.py — Stage 2A本実行(仕様書追加指示6、2026-09-20夜)

正式な主比較: P6 ROBUST_CVAR vs P1 FLAT (spec.json primary_comparison_amendment_20260920)。
対象券種: 単勝・複勝(Gate J1確定)。対象年: 2024・2025(historical_pre_snapshot使用)。

パラメータは全てstage2a_dry_run.STAGE2A_FIXED_PARAMSで結果を見る前に固定済み。

処理順序(結果情報の混入を防ぐための明確な分離):
  1. 候補生成(calibrated tansho上位3頭 x 単勝/複勝) — 結果不要
  2. historical_pre_snapshotオッズ読込(TANPUK) — 結果不要
  3. P1/P6の配分計算(candidate/odds/probability/不確実性のみ使用) — 結果不要
  4. 決済(kekka payoutで初めて結果を参照) — ここで初めて2024-2025年の結果を開く

実行: python -m analysis.mcond.exp07_robust_portfolio_dev.stage2a_run [--year 2024] [--limit N]
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
import pl_probs as PL  # noqa: E402
from analysis.mcond.exp07_robust_portfolio_dev import build_scenarios as BS  # noqa: E402
from analysis.mcond.exp07_robust_portfolio_dev import stage2a_engine as ENG  # noqa: E402
from analysis.mcond.exp07_robust_portfolio_dev import uncertainty_scenarios as US  # noqa: E402
from analysis.mcond.exp07_robust_portfolio_dev.evaluate import (  # noqa: E402
    RaceOutcome, settle_ticket,
)
from analysis.mcond.exp07_robust_portfolio_dev.stage2a_dry_run import STAGE2A_FIXED_PARAMS  # noqa: E402

HERE = Path(__file__).resolve().parent
EPS = 1e-9

SOLVER_CVAR_ALPHA = 1.0 - STAGE2A_FIXED_PARAMS["policy_params"]["P6_ROBUST_CVAR"]["spec_cvar_alpha"]
CVAR_PENALTY = STAGE2A_FIXED_PARAMS["policy_params"]["P6_ROBUST_CVAR"]["cvar_penalty_lambda"]
N_DRAWS = STAGE2A_FIXED_PARAMS["policy_params"]["P6_ROBUST_CVAR"]["uncertainty_draws"]
SIGMA = STAGE2A_FIXED_PARAMS["policy_params"]["P6_ROBUST_CVAR"]["uncertainty_score_sigma"]
ODDS_SHRINK_P10 = STAGE2A_FIXED_PARAMS["policy_params"]["P6_ROBUST_CVAR"]["uncertainty_odds_shrinkage_p10"]
SEED = STAGE2A_FIXED_PARAMS["policy_params"]["P6_ROBUST_CVAR"]["random_seed"]
BUDGET_YEN = STAGE2A_FIXED_PARAMS["budget"]["budget_yen_per_race"]
UNIT_YEN = STAGE2A_FIXED_PARAMS["budget"]["unit_yen"]
BANKROLL_YEN = STAGE2A_FIXED_PARAMS["budget"]["bankroll_yen"]
EXPOSURE_CAP_YEN = STAGE2A_FIXED_PARAMS["same_horse_exposure_cap"]["cap_yen_per_race"]
TOP_N = STAGE2A_FIXED_PARAMS["candidate_generation"]["top_n_horses"]


# =============================================================================
# 段階1-2: 候補生成 + historical_pre_snapshotオッズ (結果不要)
# =============================================================================

def load_candidates_no_results(year: int) -> pd.DataFrame:
    """候補生成に必要な列だけを読む(rid16/ban/year/date/n_field + raw score)。
    結果列(fin/top3/win/fpay)はここでは一切読み込まない。

    スコア源の注記(2026-09-20夜、実装中に判明): data/oof_scores_v6params.parquet
    は2015-2023年のみで2024-2025年をカバーしない。Gate J1(2023 development)は
    このファイルのscore列を使用したが、Stage 2A(2024-2025)ではexp05_design.parquet
    のv6_score列(v6_src='oof'、同じくOOFパイプライン由来)を使う。2023年で両者を
    突き合わせたところ相関0.9915(平均絶対差0.122、スコアの標準偏差は約1.1)と非常に
    高い一致度で、同一v6モデルの異なるOOF生成パイプライン間の軽微な差と考えられる。
    pl_calibrators_v6.pkl自体はbuild_pl_calibrators.pyの直接model.predict()という
    また別のスコア経路でfitされているため、いずれの経路を使っても較正器のfit元と
    厳密には一致しない。Isotonic回帰は単調に近い入力変動に対して頑健なため、この差が
    与える影響は限定的と考えられるが既知の近似として明記する。"""
    design = pd.read_parquet(
        BASE / "data" / "_research" / "mcond" / "exp05_design.parquet",
        columns=["rid16", "ban", "year", "date", "n_field", "v6_score"],
    )
    design = design[design["year"] == year].rename(columns={"v6_score": "score"})
    return design


def load_historical_pre_snapshot_odds(year: int) -> dict:
    """TANPUKアーカイブの最終前売り(区分1)スナップショットから単勝オッズ・複勝Lo/Hiを
    (rid16, ban) -> dict で返す。結果は一切参照しない(価格データのみ)。"""
    tdf = pd.read_csv(
        BASE / "data" / "Time _series_odds" / "TANPUK_20210105-20251228.csv",
        encoding="cp932", low_memory=False,
    )
    tdf["_year"] = tdf["レースID"].astype(str).str[:4]
    tdf = tdf[(tdf["_year"] == str(year)) & (tdf["区分"] == 1)]
    tdf = tdf.sort_values("月日時分").groupby("レースID").last().reset_index()

    odds: dict[str, dict[int, dict]] = {}
    for _, row in tdf.iterrows():
        rid16 = str(row["レースID"])
        race_odds = {}
        for ban in range(1, 19):
            tan_col, lo_col, hi_col = f"{ban}単", f"{ban}複Lo", f"{ban}複Hi"
            tan = row.get(tan_col)
            lo = row.get(lo_col)
            hi = row.get(hi_col)
            if pd.notna(tan) and float(tan) > 0:
                race_odds[ban] = {
                    "tansho_odds": float(tan),
                    "fukusho_lo": float(lo) if pd.notna(lo) else None,
                    "fukusho_hi": float(hi) if pd.notna(hi) else None,
                }
        odds[rid16] = race_odds
    return odds


def select_candidates(race_df: pd.DataFrame, calibrators: dict) -> pd.DataFrame | None:
    """レース1件分のDataFrame(ban, score)から較正済みtansho確率上位TOP_N頭を選ぶ。"""
    race_df = race_df.sort_values("ban").reset_index(drop=True)
    n = len(race_df)
    if n < TOP_N:
        return None
    scores = race_df["score"].to_numpy(dtype=float)
    w = PL.pl_weights(scores)
    raw_tansho = PL.all_tansho(w)
    cal_tansho = calibrators["tansho"].predict(raw_tansho)
    race_df = race_df.copy()
    race_df["cal_tansho_p"] = cal_tansho
    top = race_df.nlargest(TOP_N, "cal_tansho_p")
    return top


# =============================================================================
# 段階3: P1/P6の配分計算 (候補・オッズ・確率・不確実性のみ使用、結果不要)
# =============================================================================

def compute_race_allocations(
    race_df: pd.DataFrame, top: pd.DataFrame, odds_for_race: dict, calibrators: dict,
) -> dict | None:
    """1レース分のP1(FLAT)/P6(ROBUST_CVAR)配分を計算する。結果情報は一切使わない。
    戻り値がNoneの場合はそのレースを異常として扱う(オッズ欠損等)。"""
    race_df = race_df.sort_values("ban").reset_index(drop=True)
    ban_to_idx = {int(b): i for i, b in enumerate(race_df["ban"])}
    scores = race_df["score"].to_numpy(dtype=float)

    candidate_bans = [int(b) for b in top["ban"]]
    candidate_indices = tuple(ban_to_idx[b] for b in candidate_bans)

    # オッズ欠損チェック(異常検出、fail-closed)
    odds_floors = {}
    for ban in candidate_bans:
        o = odds_for_race.get(ban)
        if o is None or o["tansho_odds"] is None or o["fukusho_lo"] is None:
            return None
        odds_floors[ban] = o

    w = PL.pl_weights(scores)
    states, base_probs = ENG.fast_build_top3_states(w)
    reduced_states, reduced_probs = ENG.build_reduced_states(states, base_probs, candidate_indices)

    # TANPUKの「単」「複Lo」はJRA標準の倍率表記(例: 10.0=10.0倍)であり、
    # Ticket.odds_floorが期待する「gross return multiple」形式とそのまま一致する
    # (変換不要)。複勝はLo(下限、保守的)を採用する。
    ticket_specs = []
    for slot, ban in enumerate(candidate_bans):
        tan_odds = odds_floors[ban]["tansho_odds"]
        fuku_odds = odds_floors[ban]["fukusho_lo"]
        ticket_specs.append(("tansho", slot, tan_odds))
        ticket_specs.append(("fukusho", slot, fuku_odds))

    payoff_matrix = ENG.reduced_state_payoff_matrix(reduced_states, ticket_specs)

    # --- P1 FLAT: 6候補へ均等配分(100円単位、端数は候補0番から順に1単位ずつ加算) ---
    n_tickets = len(ticket_specs)
    budget_units = BUDGET_YEN // UNIT_YEN
    base_units = budget_units // n_tickets
    remainder = budget_units - base_units * n_tickets
    p1_units = [base_units] * n_tickets
    for i in range(remainder):
        p1_units[i] += 1
    p1_stakes_yen = [u * UNIT_YEN for u in p1_units]

    # --- P6 ROBUST_CVAR: 不確実性シナリオ + full-spend探索 ---
    # uncertainty_scenarios.py(build_scenarios.build_top3_states経由、Gate J0検証済みだが
    # 状態ごとPython呼び出しのため大量生成には遅すぎると実装中に判明)のロジック定義
    # (スコアへのi.i.d.ガウス摂動、sigma/draws/seedは同一)をfast_build_top3_states経由で
    # 再現する。摂動方式自体はuncertainty_scenarios.generate_score_perturbation_scenariosと
    # 完全に同一(乱数生成の呼び出し方も同一)。
    race_seed = SEED + abs(hash(str(candidate_bans))) % 10_000
    perturbed_scores_list = US.generate_score_perturbation_scenarios(
        scores, n_draws=N_DRAWS, sigma=SIGMA, seed=race_seed,
    )
    scenario_matrix_rows = [reduced_probs]
    for perturbed_scores in perturbed_scores_list:
        w_pert = PL.pl_weights(perturbed_scores)
        states_pert, probs_pert = ENG.fast_build_top3_states(w_pert)
        assert states_pert == states, "摂動後も状態列挙順は不変であるべき(頭数不変)"
        # 軽量Gate J0チェック(確率和が1に近いか)。フル整合性チェックは
        # test_stage2a_engine.pyでfast_build_top3_states自体が既に検証済みのため、
        # ここでは異常値(NaN等)の見逃しを防ぐ安価な安全網としてのみ確認する。
        if not (0.999 <= probs_pert.sum() <= 1.001):
            return None
        _, red = ENG.build_reduced_states(states_pert, probs_pert, candidate_indices)
        scenario_matrix_rows.append(red)
    scenario_matrix = np.vstack(scenario_matrix_rows)

    # オッズ縮小(保守化): P6のみ、historical_pre_snapshotオッズにp10縮小率を適用
    shrunk_ticket_specs = [(tt, slot, odds * ODDS_SHRINK_P10) for tt, slot, odds in ticket_specs]
    shrunk_payoff_matrix = ENG.reduced_state_payoff_matrix(reduced_states, shrunk_ticket_specs)

    # 同一馬露出上限: tansho/fukushoが同じ馬(同じslot)のticket列indexをグループ化する。
    # ticket_specsは [tansho0,fukusho0,tansho1,fukusho1,tansho2,fukusho2] の順で構築済み。
    exposure_groups = [[2 * slot, 2 * slot + 1] for slot in range(len(candidate_bans))]
    try:
        p6_result = ENG.vectorized_full_spend_search(
            shrunk_payoff_matrix, scenario_matrix,
            budget_yen=BUDGET_YEN, bankroll_yen=BANKROLL_YEN, unit_yen=UNIT_YEN,
            solver_cvar_alpha=SOLVER_CVAR_ALPHA, cvar_penalty=CVAR_PENALTY,
            exposure_groups=exposure_groups, exposure_cap_yen=EXPOSURE_CAP_YEN,
        )
    except ValueError:
        # 露出上限を満たすfull-spend配分が存在しない(理論上は予算1000円/3馬なら
        # 各馬最大333円均等で必ず満たせるはずだが、念のためfail-closedで異常計上する)
        return None
    p6_units = p6_result["best_units"]
    p6_stakes_yen = [u * UNIT_YEN for u in p6_units]

    exposure = {}
    for slot, ban in enumerate(candidate_bans):
        exposure[ban] = exposure.get(ban, 0) + p6_stakes_yen[slot * 2] + p6_stakes_yen[slot * 2 + 1]
    exposure_violation = any(v > EXPOSURE_CAP_YEN for v in exposure.values())
    assert not exposure_violation, "制約実装後は露出違反が発生しないはず(内部整合性チェック)"

    return {
        "candidate_bans": candidate_bans,
        "ticket_specs": ticket_specs,  # tansho/fukushoの順、生odds
        "p1_units": p1_units, "p1_stakes_yen": p1_stakes_yen,
        "p6_units": p6_units, "p6_stakes_yen": p6_stakes_yen,
        "p6_exposure_violation": exposure_violation,
        "p6_n_combos_evaluated": p6_result["n_combos_evaluated"],
    }


# =============================================================================
# 段階4: 決済 (ここで初めて2024-2025年の結果を参照)
# =============================================================================

def load_kekka_payouts(year: int) -> dict:
    """kekka払戻マスターから (rid16 -> {'tansho': {ban: payout}, 'fukusho': {ban: payout}}) を
    構築する。この関数の呼び出しが本パイプラインで唯一、実着順・払戻を参照する箇所。"""
    df = pd.read_csv(BASE / "data" / "kekka_20130105-20251228.csv", encoding="cp932", low_memory=False)
    df.columns = ["rid_new", "ban", "ketto", "fin", "tan_pay", "fuku_pay",
                 "umaren", "umatan", "wide", "sanpuku", "sanpuku_tan"]
    df["rid_new_str"] = df["rid_new"].astype(str)
    df["rid16"] = df["rid_new_str"].str[:-2]
    df["_year"] = df["rid16"].str[:4]
    df = df[df["_year"] == str(year)]

    payouts: dict[str, dict] = {}
    for rid16, g in df.groupby("rid16"):
        tansho_payout = {}
        fukusho_payout = {}
        for _, row in g.iterrows():
            ban = int(row["ban"])
            fin = row["fin"]
            tan_raw = str(row["tan_pay"])
            if fin == 1 and not tan_raw.startswith("("):
                try:
                    tansho_payout[ban] = int(float(tan_raw))
                except ValueError:
                    pass
            fuku_raw = row["fuku_pay"]
            if pd.notna(fuku_raw):
                try:
                    fukusho_payout[ban] = int(float(fuku_raw))
                except (ValueError, TypeError):
                    pass
        payouts[rid16] = {"tansho": tansho_payout, "fukusho": fukusho_payout}
    return payouts


def settle_allocation(
    rid16: str, candidate_bans: list[int], ticket_specs: list[tuple], stakes_yen: list[int],
    kekka_for_race: dict | None,
) -> dict:
    """P1またはP6の配分を実際の払戻で決済する。kekka_for_raceがNone(結果未着)の場合は
    is_missing=Trueで決済する(仕様書8節項目10の欠損結果テストと同じ扱い)。"""
    if kekka_for_race is None:
        outcome = RaceOutcome(race_id=rid16, is_missing=True)
    else:
        payout_table = {}
        for ban, pay in kekka_for_race["tansho"].items():
            payout_table[("tansho", (ban,))] = pay
        for ban, pay in kekka_for_race["fukusho"].items():
            payout_table[("fukusho", (ban,))] = pay
        outcome = RaceOutcome(race_id=rid16, payout_per_100_yen=payout_table)

    total_stake, total_payout = 0, 0
    any_missing = False
    for (ttype, slot, _odds), stake in zip(ticket_specs, stakes_yen):
        if stake <= 0:
            continue
        ban = candidate_bans[slot]
        r = settle_ticket(ttype, (ban,), stake, outcome)
        if r["status"] == "missing":
            any_missing = True
            continue
        total_stake += stake
        total_payout += r["payout_yen"]
    return {"total_stake_yen": total_stake, "total_payout_yen": total_payout,
           "total_profit_yen": total_payout - total_stake, "any_missing": any_missing}


# =============================================================================
# メイン
# =============================================================================

def run_year(year: int, calibrators: dict, limit: int | None = None) -> list[dict]:
    print(f"[stage2a] {year}: loading candidates (no result columns)...")
    df = load_candidates_no_results(year)
    print(f"[stage2a] {year}: loading historical_pre_snapshot odds...")
    odds = load_historical_pre_snapshot_odds(year)

    race_ids = df["rid16"].unique().tolist()
    if limit:
        race_ids = race_ids[:limit]

    print(f"[stage2a] {year}: computing P1/P6 allocations for {len(race_ids)} races "
         f"(結果列は未読込)...")
    allocations = {}
    n_anomaly = 0
    t0 = time.time()
    for i, rid16 in enumerate(race_ids):
        race_df = df[df["rid16"] == rid16]
        top = select_candidates(race_df, calibrators)
        if top is None:
            n_anomaly += 1
            continue
        odds_for_race = odds.get(rid16, {})
        alloc = compute_race_allocations(race_df, top, odds_for_race, calibrators)
        if alloc is None:
            n_anomaly += 1
            continue
        allocations[rid16] = alloc
        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"[stage2a] {year}: {i+1}/{len(race_ids)} races allocated "
                 f"({elapsed:.0f}s elapsed, ~{elapsed/(i+1)*len(race_ids):.0f}s est. total)")

    print(f"[stage2a] {year}: allocations done. n_ok={len(allocations)} n_anomaly={n_anomaly}")
    print(f"[stage2a] {year}: loading kekka payouts (結果を初めて参照)...")
    kekka = load_kekka_payouts(year)

    results = []
    for rid16, alloc in allocations.items():
        kekka_for_race = kekka.get(rid16)
        p1_settle = settle_allocation(
            rid16, alloc["candidate_bans"], alloc["ticket_specs"], alloc["p1_stakes_yen"], kekka_for_race,
        )
        p6_settle = settle_allocation(
            rid16, alloc["candidate_bans"], alloc["ticket_specs"], alloc["p6_stakes_yen"], kekka_for_race,
        )
        results.append({
            "rid16": rid16, "year": year,
            "p1_stake": p1_settle["total_stake_yen"], "p1_payout": p1_settle["total_payout_yen"],
            "p1_profit": p1_settle["total_profit_yen"],
            "p6_stake": p6_settle["total_stake_yen"], "p6_payout": p6_settle["total_payout_yen"],
            "p6_profit": p6_settle["total_profit_yen"],
            "p6_exposure_violation": alloc["p6_exposure_violation"],
            "any_missing": p1_settle["any_missing"] or p6_settle["any_missing"],
        })
    return results, n_anomaly


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=None, help="単年のみ実行(テスト用)")
    ap.add_argument("--limit", type=int, default=None, help="レース数を制限(テスト用)")
    args = ap.parse_args()

    calib = joblib.load(BASE / "models" / "pl_calibrators_v6.pkl")
    calibrators = calib["calibrators"]

    years = [args.year] if args.year else [2024, 2025]
    all_results = []
    all_anomaly = 0
    for year in years:
        results, n_anomaly = run_year(year, calibrators, limit=args.limit)
        all_results.extend(results)
        all_anomaly += n_anomaly

    out_path = HERE / "out" / "STAGE2A_RESULTS.json"
    out_path.write_text(json.dumps({"results": all_results, "n_anomaly": all_anomaly},
                                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[stage2a] wrote {out_path}")
    print(f"[stage2a] n_races={len(all_results)} n_anomaly={all_anomaly}")


if __name__ == "__main__":
    main()
