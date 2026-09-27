# -*- coding: utf-8 -*-
"""
winner_claim.py — 優勝者三連単表の +10pt を公表値だけで格付けする (手元の結果データを開かない)
=============================================================================================
公表値 (SPEC §0): 2025-2026 の 4,365 race、三連単確定オッズ 600〜1,200 倍 ROI 0.831、1,200〜2,500 倍 ROI 0.820。
控除率基準 0.725。独立単位は race (4,365)。組合せ数を独立 n に使わない。
帯あたり ticket 数 (race 平均 n̄) は公表資料の原表が手元に無いのでパラメータとして振る。
race 単位の近似: 帯内で的中は race あたり高々 1 (排反)。X_r = 的中時の払戻倍率 / n̄、
  Var(X_r) ≈ ROI · ō / n̄ (ō = 帯の代表倍率、較正に近い市場を仮定)、SE(ROI) ≈ sqrt(ROI · ō / (n̄ · R))
追加の分散要因 (暦日 cluster・払戻の裾) は design effect 1.0 / 1.5 / 2.0 で感度を見る。
多重比較: 主表の固定帯数 k を 10〜15、帯選択は結果を見た後 (最良帯) なので Bonferroni 片側 z 閾値を併記。
出力: out/winner_claim_uncertainty.json
"""
from __future__ import annotations

import json

import numpy as np
from scipy.stats import norm

from .loaders import OUT

R = 4365
BASE = 0.725
BANDS = {"600-1200": {"roi": 0.831, "odds_rep": float(np.sqrt(600 * 1200))},
         "1200-2500": {"roi": 0.820, "odds_rep": float(np.sqrt(1200 * 2500))}}
NBAR = [25, 50, 100, 200, 400, 800]
DEFF = [1.0, 1.5, 2.0]
K = [10, 15]


def main():
    out = {"inputs": {"races": R, "baseline": BASE, "bands": BANDS, "nbar_grid": NBAR, "design_effect_grid": DEFF,
                      "k_bands_grid": K, "note": "n-bar (tickets per race in band) is unknown without the original "
                                                   "table; treated as a parameter"},
           "z_thresholds_one_sided": {str(k): float(norm.ppf(1 - 0.05 / k)) for k in K}, "table": {}}
    for b, v in BANDS.items():
        rows = {}
        for nb in NBAR:
            for de in DEFF:
                se = float(np.sqrt(de * v["roi"] * v["odds_rep"] / (nb * R)))
                z = (v["roi"] - BASE) / se
                rows[f"nbar={nb}|deff={de}"] = {"se": se, "ci95": [v["roi"] - 1.96 * se, v["roi"] + 1.96 * se],
                                                "z_vs_baseline": z,
                                                "passes_bonferroni_k10": bool(z > norm.ppf(1 - 0.05 / 10)),
                                                "passes_bonferroni_k15": bool(z > norm.ppf(1 - 0.05 / 15)),
                                                "ci95_lower_above_1": bool(v["roi"] - 1.96 * se > 1.0)}
        zt = norm.ppf(1 - 0.05 / 15)
        out["table"][b] = {"rows": rows,
                           "nbar_needed_for_bonferroni_k15_deff1": float(v["roi"] * v["odds_rep"] / (R * ((v["roi"] - BASE) / zt) ** 2)),
                           "nbar_needed_for_bonferroni_k15_deff2": float(2 * v["roi"] * v["odds_rep"] / (R * ((v["roi"] - BASE) / zt) ** 2))}
    out["grading"] = ("Point estimates 0.831 / 0.820 are below break-even (1.0) under every n-bar and design effect in the "
                      "grid (CI95 upper < 1 unless n-bar is tiny). Whether the +10pt uplift over the 0.725 takeout baseline "
                      "survives best-band selection depends on n-bar: see nbar_needed_*. The claim remains a single-period "
                      "(2025-2026), terminal-price, post-hoc band selection; EXP21 cannot replicate it because no "
                      "trifecta price source passes G0.")
    (OUT / "winner_claim_uncertainty.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for b, v in out["table"].items():
        print(b, {k: round(x["z_vs_baseline"], 2) for k, x in v["rows"].items() if k.endswith("deff=1.0")},
              "nbar needed k15:", round(v["nbar_needed_for_bonferroni_k15_deff1"], 1), round(v["nbar_needed_for_bonferroni_k15_deff2"], 1))
        print("   ci95 upper (nbar=25, deff=2):", round(v["rows"]["nbar=25|deff=2.0"]["ci95"][1], 3))


if __name__ == "__main__":
    main()
