# -*- coding: utf-8 -*-
"""
ablation.py — §4 列の影響分離 (A/B/C/D)。2023 development のみ、結果を見て仕様変更しない。
==========================================================================================
A: R0-clean (111列)                      B: A + 前走レースID 2列
C: A + 馬主(最新/仮想)                    D: A + 両方
EXP15 と同一の凍結プロトコル: train 2016-2021 / ES 2022 (NDCG@5) / 2023 fixed-model、
同一ハイパラ (EXP15 が 2022 で選んだ v6_lr_half)、同一 5 seed、同一 race set、uniform weight。
production モデルには触れない。2024/2025 は開かない。
出力: out/ablation.json, out/ablation_table.md
実行: python -m analysis.mcond.owner_id_time_safety_audit.ablation
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..exp15_race_as_set_dev import common as C
from ..exp15_race_as_set_dev import train_r0 as R0
from ..exp15_race_as_set_dev import evaluate as EV

OUT = C.BASE / "analysis" / "mcond" / "owner_id_time_safety_audit" / "out"
OWN = "馬主(最新/仮想)"
PREV = ["前走レースID(新)", "前走レースID(新/馬番無)"]
ARMS = {"A_clean": [], "B_prev_id": PREV, "C_owner": [OWN], "D_both": PREV + [OWN]}


def main():
    df = C.load_rows()
    con = C.load_contract()
    cfg = next(c for c in C.SPEC["models"]["R0-clean"]["grid"]
               if c["name"] == json.loads((C.OUT / "r0_train.json").read_text(encoding="utf-8"))["selected"])
    seeds = C.SPEC["seeds"]["main"]
    dev = (df["period"] == "dev").to_numpy()
    races = [i for i in C.race_index(df, dev) if len(i) >= C.SPEC["population"]["eval_min_field"]]
    day = df["meeting_day"].to_numpy()[[i[0] for i in races]]

    tabs, scores = {}, {}
    log = {"config": cfg["name"], "seeds": seeds, "arms": {}}
    for arm, extra in ARMS.items():
        cols = con["features"]["clean"] + extra
        loc = dict(con)
        loc["features"] = {"x": cols}
        enc = C.Encoded(df, loc, "x")
        log["arms"][arm] = {"n_features": len(cols), "added": extra, "seeds": {}}
        for sd in seeds:
            s, info = R0.fit_one(enc.X_r0, df, cfg, sd)
            p = C.softmax_race(s, races, info["tau"])
            tabs[(arm, sd)] = C.race_metrics(df, races, s, p)
            scores[(arm, sd)] = s
            log["arms"][arm]["seeds"][str(sd)] = {k: info[k] for k in ("best_iter", "tau", "sel_ll")}
            print(f"[{arm:10s} s{sd}] iter={info['best_iter']} sel_ll={info['sel_ll']:.5f}", flush=True)

    rows = np.concatenate(races)
    y = df["win"].to_numpy()[rows]
    day_row = df["meeting_day"].to_numpy()[rows]
    # 指標表
    res = {"vs_A": {}, "levels": {}}
    for arm in ARMS:
        lv = {m: float(np.mean([tabs[(arm, s)][m].mean() for s in seeds]))
              for m in ("ll", "brier", "ndcg3", "ndcg5", "hon_top3")}
        res["levels"][arm] = lv
    for arm in [a for a in ARMS if a != "A_clean"]:
        cmp, _ = EV.compare(tabs, arm, "A_clean", seeds, day)
        # ECE (seed 平均の差、開催日 bootstrap)
        Pa = {(arm, s): C.softmax_race(scores[(arm, s)], races,
                                       log["arms"][arm]["seeds"][str(s)]["tau"])[rows] for s in seeds}
        Pb = {("A_clean", s): C.softmax_race(scores[("A_clean", s)], races,
                                             log["arms"]["A_clean"]["seeds"][str(s)]["tau"])[rows] for s in seeds}
        ece = EV.ece_compare({**Pa, **Pb}, arm, "A_clean", seeds, y, day_row)
        # raw score 差と ◎変更レース数
        sd_abs, hon_ch = [], []
        for s in seeds:
            a, b = scores[(arm, s)], scores[("A_clean", s)]
            m = ~np.isnan(a) & ~np.isnan(b)
            sd_abs.append(float(np.abs(a[m] - b[m]).mean()))
            ch = 0
            ban = df["ban"].to_numpy()
            for i in races:
                pa = i[np.lexsort((ban[i], -a[i]))][0]
                pb = i[np.lexsort((ban[i], -b[i]))][0]
                ch += int(pa != pb)
            hon_ch.append(ch)
        res["vs_A"][arm] = {"metrics": cmp, "ece10": ece,
                            "mean_abs_raw_score_diff_per_seed": sd_abs,
                            "hon_changed_races_per_seed": hon_ch,
                            "hon_changed_rate": float(np.mean(hon_ch) / len(races))}
    res["n_races"] = len(races)
    res["train_log"] = log
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ablation.json").write_text(json.dumps(res, ensure_ascii=False, indent=1,
                                                  default=lambda o: float(o)), encoding="utf-8")
    L = ["# §4 列の影響分離 (2023 development, 5 seed, 同一ハイパラ・同一 race set)", "",
         f"races={res['n_races']}  config={cfg['name']}", "",
         "| arm | 列数 | logloss | Brier | NDCG@3 | NDCG@5 | ◎top3 |", "|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        lv = res["levels"][arm]
        L.append(f"| {arm} | {log['arms'][arm]['n_features']} | {lv['ll']:.5f} | {lv['brier']:.5f} | "
                 f"{lv['ndcg3']:.5f} | {lv['ndcg5']:.5f} | {lv['hon_top3']:.5f} |")
    L += ["", "## A_clean との差 (seed 対、meeting-day bootstrap CI95)", "",
          "| arm | 指標 | Δ | CI95 | 改善seed数 | seed SD |", "|---|---|---|---|---|---|"]
    for arm, v in res["vs_A"].items():
        for k, m in v["metrics"].items():
            L.append(f"| {arm} | {k} | {m['delta']:+.5f} | [{m['ci_lo']:+.5f}, {m['ci_hi']:+.5f}] | "
                     f"{m['n_seeds_improve']}/5 | {m['seed_sd']:.5f} |")
        L.append(f"| {arm} | ECE10 | {v['ece10']['delta']:+.5f} | [{v['ece10']['ci_lo']:+.5f}, {v['ece10']['ci_hi']:+.5f}] | - | - |")
        L.append(f"| {arm} | raw score 平均絶対差 | {np.mean(v['mean_abs_raw_score_diff_per_seed']):.5f} | - | - | - |")
        L.append(f"| {arm} | ◎が変わったレース | {np.mean(v['hon_changed_races_per_seed']):.0f} "
                 f"({v['hon_changed_rate']*100:.1f}%) | - | - | - |")
    (OUT / "ablation_table.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
