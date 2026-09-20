# -*- coding: utf-8 -*-
"""
gate1_report.py — Stage 2A本実行結果(STAGE2A_RESULTS.json)からGate 1統計を集計する。
仕様書の12項目順序で出力する。meeting-day単位はrid16[0:10](日付8桁+場コード2桁)。
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
VENUE_NAMES = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
    "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉",
}


def meeting_day(rid16: str) -> str:
    return rid16[0:10]  # 日付8桁 + 場2桁


def venue_code(rid16: str) -> str:
    return rid16[8:10]


def main():
    d = json.loads((HERE / "out" / "STAGE2A_RESULTS.json").read_text(encoding="utf-8"))
    results = d["results"]
    n_anomaly = d["n_anomaly"]
    n_races = len(results)

    # 1. 対象レース数・異常件数
    n_2024 = sum(1 for r in results if r["year"] == 2024)
    n_2025 = sum(1 for r in results if r["year"] == 2025)
    n_missing = sum(1 for r in results if r["any_missing"])
    n_exposure_violation = sum(1 for r in results if r["p6_exposure_violation"])
    print("=" * 70)
    print("1. 対象レース数・異常件数")
    print(f"  n_races={n_races} (2024={n_2024}, 2025={n_2025})  n_anomaly={n_anomaly}")
    print(f"  any_missing(決済で一部欠損)races={n_missing}  exposure_violation={n_exposure_violation}")

    # 2. P6 vs P1 同一予算確認
    p1_stake_ne_1000 = [r for r in results if r["p1_stake"] != 1000]
    p6_stake_ne_1000 = [r for r in results if r["p6_stake"] != 1000]
    print("=" * 70)
    print("2. P6 vs P1の同一予算確認")
    print(f"  p1_stake!=1000のレース数: {len(p1_stake_ne_1000)} / {n_races}")
    print(f"  p6_stake!=1000のレース数: {len(p6_stake_ne_1000)} / {n_races}")
    if p1_stake_ne_1000[:3]:
        print(f"  例(P1): {p1_stake_ne_1000[:3]}")
    if p6_stake_ne_1000[:3]:
        print(f"  例(P6): {p6_stake_ne_1000[:3]}")

    def year_stats(year):
        sub = [r for r in results if r["year"] == year]
        p1_profit = np.array([r["p1_profit"] for r in sub], dtype=float)
        p6_profit = np.array([r["p6_profit"] for r in sub], dtype=float)
        diff = p6_profit - p1_profit
        p1_stake = np.array([r["p1_stake"] for r in sub], dtype=float)
        p6_stake = np.array([r["p6_stake"] for r in sub], dtype=float)
        return {
            "n": len(sub),
            "p1_total_stake": p1_stake.sum(), "p1_total_profit": p1_profit.sum(),
            "p1_roi": (p1_stake.sum() + p1_profit.sum()) / p1_stake.sum() * 100,
            "p6_total_stake": p6_stake.sum(), "p6_total_profit": p6_profit.sum(),
            "p6_roi": (p6_stake.sum() + p6_profit.sum()) / p6_stake.sum() * 100,
            "mean_diff": diff.mean(), "sum_diff": diff.sum(),
            "diff": diff,
        }

    s2024 = year_stats(2024)
    s2025 = year_stats(2025)

    print("=" * 70)
    print("3. 2024年差")
    print(f"  P1: stake={s2024['p1_total_stake']:.0f} profit={s2024['p1_total_profit']:.0f} "
         f"ROI={s2024['p1_roi']:.2f}%")
    print(f"  P6: stake={s2024['p6_total_stake']:.0f} profit={s2024['p6_total_profit']:.0f} "
         f"ROI={s2024['p6_roi']:.2f}%")
    print(f"  P6-P1差: 平均{s2024['mean_diff']:.2f}円/レース, 合計{s2024['sum_diff']:.0f}円")

    print("=" * 70)
    print("4. 2025年差")
    print(f"  P1: stake={s2025['p1_total_stake']:.0f} profit={s2025['p1_total_profit']:.0f} "
         f"ROI={s2025['p1_roi']:.2f}%")
    print(f"  P6: stake={s2025['p6_total_stake']:.0f} profit={s2025['p6_total_profit']:.0f} "
         f"ROI={s2025['p6_roi']:.2f}%")
    print(f"  P6-P1差: 平均{s2025['mean_diff']:.2f}円/レース, 合計{s2025['sum_diff']:.0f}円")

    # 5. 全期間paired bootstrap CI (meeting-day単位)
    md_groups = defaultdict(list)
    for r in results:
        md_groups[meeting_day(r["rid16"])].append(r["p6_profit"] - r["p1_profit"])
    meeting_days = list(md_groups.keys())
    md_sum_diff = np.array([sum(v) for v in md_groups.values()], dtype=float)
    md_n_races = np.array([len(v) for v in md_groups.values()], dtype=int)

    rng = np.random.default_rng(20260920)
    n_boot = 10000
    n_md = len(meeting_days)
    boot_means = np.empty(n_boot, dtype=float)
    total_races = md_n_races.sum()
    for b in range(n_boot):
        idx = rng.integers(0, n_md, size=n_md)
        boot_sum = md_sum_diff[idx].sum()
        boot_n = md_n_races[idx].sum()
        boot_means[b] = boot_sum / boot_n
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
    point_mean = md_sum_diff.sum() / total_races
    p_improve = float((boot_means > 0).mean())

    print("=" * 70)
    print("5. 全期間paired bootstrap CI (meeting-day単位, n_meeting_days=%d, n_boot=%d)" % (n_md, n_boot))
    print(f"  点推定(P6-P1平均差/レース): {point_mean:.3f}円")
    print(f"  95%CI: [{ci_lo:.3f}, {ci_hi:.3f}]円")
    print(f"  P(改善>0) = {p_improve:.4f}")

    # 6. 最大ドローダウン (rid16昇順=時系列順の累積差)
    def max_drawdown(profits_sorted):
        cum = np.cumsum(profits_sorted)
        running_max = np.maximum.accumulate(cum)
        dd = cum - running_max
        return dd.min(), cum

    results_sorted = sorted(results, key=lambda r: r["rid16"])
    p1_profits_sorted = np.array([r["p1_profit"] for r in results_sorted], dtype=float)
    p6_profits_sorted = np.array([r["p6_profit"] for r in results_sorted], dtype=float)
    p1_dd, p1_cum = max_drawdown(p1_profits_sorted)
    p6_dd, p6_cum = max_drawdown(p6_profits_sorted)
    print("=" * 70)
    print("6. 最大ドローダウン (時系列累積、rid16昇順)")
    print(f"  P1 max drawdown: {p1_dd:.0f}円 (累積最終値 {p1_cum[-1]:.0f}円)")
    print(f"  P6 max drawdown: {p6_dd:.0f}円 (累積最終値 {p6_cum[-1]:.0f}円)")

    # 7. downside CVaR (worst 10% average loss, レース単位)
    def downside_cvar(profits, tail=0.10):
        losses = -profits
        n_tail = max(1, int(np.ceil(len(losses) * tail)))
        worst = np.sort(losses)[::-1][:n_tail]
        return worst.mean()

    p1_all = np.array([r["p1_profit"] for r in results], dtype=float)
    p6_all = np.array([r["p6_profit"] for r in results], dtype=float)
    p1_cvar10 = downside_cvar(p1_all, 0.10)
    p6_cvar10 = downside_cvar(p6_all, 0.10)
    print("=" * 70)
    print("7. downside CVaR (worst 10%平均損失、レース単位)")
    print(f"  P1: {p1_cvar10:.1f}円/レース")
    print(f"  P6: {p6_cvar10:.1f}円/レース")

    # 8. 利益集中度(上位1%/5%/10%レースが総利益に占める割合)
    def concentration(profits, pct):
        n_top = max(1, int(np.ceil(len(profits) * pct)))
        sorted_p = np.sort(profits)[::-1]
        top_sum = sorted_p[:n_top].sum()
        total_positive = profits[profits > 0].sum()
        return top_sum, total_positive, (top_sum / total_positive * 100 if total_positive > 0 else float("nan"))

    print("=" * 70)
    print("8. 利益集中度(黒字レースの総利益に対する上位N%レースの寄与)")
    for pct in (0.01, 0.05, 0.10):
        for label, arr in (("P1", p1_all), ("P6", p6_all)):
            top_sum, total_pos, ratio = concentration(arr, pct)
            print(f"  {label} 上位{pct*100:.0f}%: {top_sum:.0f}円 / 黒字合計{total_pos:.0f}円 = {ratio:.1f}%")

    # 9. 券種別・人気帯別・競馬場別
    print("=" * 70)
    print("9a. 競馬場別 (venue、rid16[8:10]由来)")
    venue_groups = defaultdict(list)
    for r in results:
        venue_groups[venue_code(r["rid16"])].append(r)
    for vc in sorted(venue_groups.keys()):
        sub = venue_groups[vc]
        p1p = sum(r["p1_profit"] for r in sub)
        p6p = sum(r["p6_profit"] for r in sub)
        name = VENUE_NAMES.get(vc, f"場コード{vc}")
        print(f"  {name}({vc}): n={len(sub)}  P1計={p1p}  P6計={p6p}  差={p6p-p1p}")
    print("9b. 券種別(P1は予算配分上常に単勝500円/複勝500円で固定。P6は別途補足パスで算出予定)")
    print("9c. 人気帯別: 別途補足パス(候補生成のみ再計算、結果は使わない軽量処理)で算出予定")

    # 10. P2_PROXY
    print("=" * 70)
    print("10. P2_PROXY(副次)")
    print("  該当なし(構築を断念、spec.json primary_comparison_amendment_20260920参照)")

    # 11. Gate PASS/FAIL
    print("=" * 70)
    print("11. Gate PASS/FAIL")
    dir_2024 = s2024["mean_diff"] > 0
    dir_2025 = s2025["mean_diff"] > 0
    direction_match = dir_2024 == dir_2025
    ci_supports_improvement = ci_lo > 0
    verdict = "PASS" if (ci_supports_improvement and direction_match and dir_2024) else "FAIL"
    print(f"  2024方向: {'改善' if dir_2024 else '悪化/横ばい'} ({s2024['mean_diff']:.2f}円/レース)")
    print(f"  2025方向: {'改善' if dir_2025 else '悪化/横ばい'} ({s2025['mean_diff']:.2f}円/レース)")
    print(f"  方向一致: {direction_match}")
    print(f"  95%CI下限>0 (改善をCIが支持): {ci_supports_improvement}  CI=[{ci_lo:.3f},{ci_hi:.3f}]")
    print(f"  → Gate1判定: {verdict}")

    # 12. Stage 2Bへ進むか
    print("=" * 70)
    print("12. Stage 2Bへ進むか")
    print(f"  Gate1={verdict} のため、{'Stage 2Bへ進む' if verdict=='PASS' else 'Stage 2Bへは進まない(現時点でP6の頑健な優位性を確認できず)'}")

    out = {
        "n_races": n_races, "n_2024": n_2024, "n_2025": n_2025, "n_anomaly": n_anomaly,
        "n_any_missing": n_missing, "n_exposure_violation": n_exposure_violation,
        "same_budget_violations_p1": len(p1_stake_ne_1000), "same_budget_violations_p6": len(p6_stake_ne_1000),
        "year_2024": {k: v for k, v in s2024.items() if k != "diff"},
        "year_2025": {k: v for k, v in s2025.items() if k != "diff"},
        "bootstrap_meeting_day": {
            "n_meeting_days": n_md, "n_boot": n_boot,
            "point_estimate_yen_per_race": point_mean,
            "ci95": [float(ci_lo), float(ci_hi)],
            "p_improve": p_improve,
        },
        "max_drawdown": {"p1": float(p1_dd), "p6": float(p6_dd)},
        "downside_cvar_10pct": {"p1": float(p1_cvar10), "p6": float(p6_cvar10)},
        "gate1_verdict": verdict,
        "direction_2024_improve": bool(dir_2024), "direction_2025_improve": bool(dir_2025),
    }
    (HERE / "out" / "GATE1_REPORT.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("=" * 70)
    print(f"wrote {HERE / 'out' / 'GATE1_REPORT.json'}")


if __name__ == "__main__":
    main()
