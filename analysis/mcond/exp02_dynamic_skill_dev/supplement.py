# -*- coding: utf-8 -*-
"""
supplement.py — 仕様固定 (02084801) の後に追加した【探索分析】(主判定を変えない)
====================================================================================
最終報告の問い「能力平均と不確実性のどちらが効いたか」に答えるための分解。
係数は σ と更新回数が強く相関しているため解釈できないので、特徴を2群に分けて M1 への上積みを比べる。
  平均系  : dyn_skill_mu, rank_in_race, gap_to_top, gap_to_field_mean, last_change,
            field_skill_mean/std/max/top3_mean, horse_skill_minus_field, horse_skill_percentile, race_difficulty
  不確実性系: dyn_skill_sigma, dyn_skill_num_updates, dyn_skill_days_since_update, field_uncertainty_mean
  (dyn_skill_conservative = mu − 2σ は両方を含むのでどちらにも入れない)
対照: 不確実性系と同じ情報のうち「更新回数・休養日数」は v6 や既存特徴にも近い量があるので、
      σ だけを抜いたモデルも比べる。
出力: out/exploratory_mean_vs_uncertainty.csv
実行: python -m analysis.mcond.exp02_dynamic_skill_dev.supplement
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
from analysis.mcond.evaluate import fit_predict, delta_boot  # noqa: E402
from analysis.mcond.exp02_dynamic_skill_dev.run import load  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
MEAN = ["dyn_skill_mu", "dyn_skill_rank_in_race", "dyn_skill_gap_to_top", "dyn_skill_gap_to_field_mean",
        "dyn_skill_last_change", "field_skill_mean", "field_skill_std", "field_skill_max",
        "field_skill_top3_mean", "horse_skill_minus_field", "horse_skill_percentile", "race_difficulty"]
UNC = ["dyn_skill_sigma", "dyn_skill_num_updates", "dyn_skill_days_since_update", "field_uncertainty_mean"]


def main() -> None:
    df = load()
    y, yr = df["top3"].to_numpy(), df["year"].to_numpy()
    train, sel = (yr >= 2016) & (yr <= 2021), yr == 2022
    day = df["day"].to_numpy()
    base = ["f_v6", "f_mkt"]
    M = {"M1": base, "平均系のみ": base + MEAN, "不確実性系のみ": base + UNC,
         "σだけ": base + ["dyn_skill_sigma"], "更新回数・休養日数だけ": base + UNC[1:3],
         "平均系+不確実性系": base + MEAN + UNC}
    P = {k: fit_predict(df, v, y, train, sel)[0] for k, v in M.items()}
    rows = []
    for k in M:
        if k == "M1":
            continue
        for pn, m in [("confirm_2023", yr == 2023), ("oos_2024_25", (yr >= 2024) & (yr <= 2025))]:
            rows.append({"model": k, "period": pn, **delta_boot(y[m], P[k][m], P["M1"][m], day[m], level=0.95)})
    # 固有価値の分解: 既存 ELO/Glicko に「更新回数・休養日数」を足した対照に対して、動的能力の平均系が上積みするか
    import json
    spec = json.loads((Path(__file__).resolve().parent / "spec.json").read_text(encoding="utf-8"))
    EX = spec["features"]["existing_rating"]
    M2 = {"M2+回数休養": base + EX + UNC[1:3], "M2+回数休養+平均系": base + EX + UNC[1:3] + MEAN,
          "M4(主判定と同じ)": base + EX + spec["features"]["T1"]}
    P2 = {k: fit_predict(df, v, y, train, sel)[0] for k, v in M2.items()}
    for a in ["M2+回数休養+平均系", "M4(主判定と同じ)"]:
        for pn, m in [("confirm_2023", yr == 2023), ("oos_2024_25", (yr >= 2024) & (yr <= 2025))]:
            rows.append({"model": f"{a} vs M2+回数休養", "period": pn,
                         **delta_boot(y[m], P2[a][m], P2["M2+回数休養"][m], day[m], level=0.95)})
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "exploratory_mean_vs_uncertainty.csv", index=False, encoding="utf-8-sig")
    print("M1 に対する Δlogloss (負=良い), 95%CI  ※探索分析")
    for x in rows:
        print(f"  {x['model']:<30} {x['period']:<13} Δ={x['delta']:+.6f} [{x['ci_lo']:+.6f},{x['ci_hi']:+.6f}]")


if __name__ == "__main__":
    main()
