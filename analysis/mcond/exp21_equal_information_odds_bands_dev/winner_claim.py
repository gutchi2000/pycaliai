# -*- coding: utf-8 -*-
"""
winner_claim.py — 優勝者三連単表の +10pt を公表値だけで格付けする (手元の結果データを開かない)
=============================================================================================
公表値:
  SPEC §0   2025-2026 の 4,365 race、三連単確定オッズ 600〜1,200 倍 ROI 0.831、1,200〜2,500 倍 ROI 0.820
  付録 A3   帯別の「組合せ数」(資料記載の組合せ数。票数ではない) — 全 9 帯
控除率基準 0.725。独立単位は race (4,365)。同一 race 内の組合せは独立標本として扱わない:
  帯あたりの組合せ数は race 平均 n̄ = 組合せ数 / 4,365 に換算し、SE は race 単位で作る。
race 単位の近似: 帯内で的中は race あたり高々 1 (排反)。X_r = 的中時の払戻倍率 / n̄、
  Var(X_r) ≈ ROI · ō / n̄ (ō = 帯の幾何中央倍率、較正に近い市場を仮定)、SE(ROI) ≈ sqrt(deff · ROI · ō / (n̄ · R))
追加の分散要因 (暦日 cluster・払戻の裾) は design effect 1.0 / 1.5 / 2.0 で感度を見る。
多重比較: 付録 A3 の帯数 k = 9 を主とし、10 / 15 は感度。帯は結果を見た後の最良帯なので Bonferroni 片側 z 閾値と比べる。
出力: out/winner_claim_uncertainty.json
"""
from __future__ import annotations

import json

import numpy as np
from scipy.stats import norm

from .loaders import OUT

R = 4365
BASE = 0.725
# 付録 A3 の帯別組合せ数 (資料記載値。票数ではない)
A3_COMBINATIONS = {"13-34": 10686, "34-89": 63026, "89-150": 86885, "150-300": 212526, "300-600": 359976,
                   "600-1200": 557115, "1200-2500": 848502, "2500-5000": 1068084, "5000-15000": 2221267}
BANDS = {"600-1200": {"roi": 0.831, "odds_rep": float(np.sqrt(600 * 1200))},
         "1200-2500": {"roi": 0.820, "odds_rep": float(np.sqrt(1200 * 2500))}}
NBAR_GRID = [25, 50, 100, 200, 400, 800]
DEFF = [1.0, 1.5, 2.0]
K_PRIMARY = 9
K = [9, 10, 15]


def zthr(k):
    return float(norm.ppf(1 - 0.05 / k))


def row(v, nb, de):
    se = float(np.sqrt(de * v["roi"] * v["odds_rep"] / (nb * R)))
    z = (v["roi"] - BASE) / se
    return {"se": se, "ci95": [v["roi"] - 1.96 * se, v["roi"] + 1.96 * se], "z_vs_baseline": z,
            **{f"passes_bonferroni_k{k}": bool(z > zthr(k)) for k in K},
            "ci95_lower_above_1": bool(v["roi"] - 1.96 * se > 1.0)}


def main():
    nbar = {b: c / R for b, c in A3_COMBINATIONS.items()}
    out = {"inputs": {"races": R, "baseline": BASE, "bands_with_published_roi": BANDS,
                      "appendix_A3_combinations": A3_COMBINATIONS,
                      "appendix_A3_combinations_per_race": nbar,
                      "unit_note": "counts are the published combination counts (組合せ数), not vote counts; combinations in "
                                   "the same race are not independent samples, so they enter only through n-bar per race",
                      "design_effect_grid": DEFF, "k_primary": K_PRIMARY, "k_sensitivity": K},
           "z_thresholds_one_sided": {str(k): zthr(k) for k in K}, "published_nbar": {}, "parametric_nbar_grid": {}}
    for b, v in BANDS.items():
        nb = nbar[b]
        need = {f"deff={de}": float(de * v["roi"] * v["odds_rep"] / (R * ((v["roi"] - BASE) / zthr(k)) ** 2))
                for de in DEFF for k in [K_PRIMARY]}
        need_k15 = {f"deff={de}": float(de * v["roi"] * v["odds_rep"] / (R * ((v["roi"] - BASE) / zthr(15)) ** 2))
                    for de in DEFF}
        out["published_nbar"][b] = {
            "nbar": nb, "rows": {f"deff={de}": row(v, nb, de) for de in DEFF},
            "nbar_needed_k9": need, "nbar_needed_k15": need_k15,
            "meets_needed_k9": {k_: bool(nb >= x) for k_, x in need.items()},
            "meets_needed_k15": {k_: bool(nb >= x) for k_, x in need_k15.items()}}
        out["parametric_nbar_grid"][b] = {f"nbar={n}|deff={de}": row(v, n, de) for n in NBAR_GRID for de in DEFF}
    out["grading"] = (
        "With the published appendix-A3 combination counts, 600-1200x has 127.6 combinations/race and 1200-2500x "
        "194.4. 600-1200x clears the best-band Bonferroni threshold only with design effect 1.0 (needed 105.8 at k=15, "
        "fewer at k=9) and fails at 1.5-2.0; 1200-2500x fails even at 1.0 (needed 265.4 at k=15). Both point "
        "estimates stay below break-even. The claim remains a single-period (2025-2026), terminal-price, post-hoc "
        "band selection that EXP21 cannot replicate (no trifecta price source passes G0).")
    (OUT / "winner_claim_uncertainty.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for b, v in out["published_nbar"].items():
        print(b, "nbar", round(v["nbar"], 1), {k: round(x["z_vs_baseline"], 2) for k, x in v["rows"].items()},
              "needed k9", {k: round(x, 1) for k, x in v["nbar_needed_k9"].items()},
              "needed k15", {k: round(x, 1) for k, x in v["nbar_needed_k15"].items()})
    print("thresholds", {k: round(x, 3) for k, x in out["z_thresholds_one_sided"].items()})


if __name__ == "__main__":
    main()
