# -*- coding: utf-8 -*-
"""
supplement.py — 仕様固定 (a41e7e96) の後に追加した【探索分析】(仕様 §12 により主検定と分離)
==========================================================================================
仕様書との突き合わせで見つかった抜けを埋める。いずれも主判定を変えるものではない。
  1. §13.2 競馬場への集中: 2023-25 合算の Δlogloss を競馬場別に
  2. §10 副次「順位」: レース内の順位相関 (Spearman) とペア正解率
  3. §7 厳格版: 行動予測モデルから過去のレース結果由来の文脈 (前走着順・前走着差・前走騎手の
     過去成績) を外した bp で、Gate 2 と同じ4条件を再計算
出力: out/exploratory_supplement.json, out/exploratory_venue.csv, out/exploratory_rank.csv
実行: python -m analysis.mcond.exp01_choice_dev.supplement
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
from analysis.mcond.evaluate import fit_predict, delta_boot, ll_vec  # noqa: E402
from analysis.mcond.exp01_choice_dev.run import load, SPEC  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
D = BASE / "data/_research/mcond"
PLACE = {"01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京", "06": "中山",
         "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"}


def rank_metrics(df, p, mask):
    d = pd.DataFrame({"rid": df.rid16[mask].to_numpy(), "p": p[mask], "fin": df.fin[mask].to_numpy(),
                      "day": df.day[mask].to_numpy()})
    rows = []
    for rid, g in d.groupby("rid"):
        if len(g) < 3:
            continue
        rho = spearmanr(g.p, -g.fin).correlation
        pp, ff = g.p.to_numpy(), g.fin.to_numpy()
        i, j = np.triu_indices(len(g), 1)
        ok = ff[i] != ff[j]
        acc = ((pp[i] > pp[j]) == (ff[i] < ff[j]))[ok].mean()
        rows.append((rid, g.day.iloc[0], rho, acc))
    return pd.DataFrame(rows, columns=["rid", "day", "rho", "pair_acc"])


def boot_mean_diff(a, b, day, reps=2000, seed=3):
    d = pd.DataFrame({"x": a - b, "day": day}).groupby("day")["x"].agg(["sum", "count"])
    s, c = d["sum"].to_numpy(), d["count"].to_numpy()
    rng = np.random.default_rng(seed)
    bs = [s[i].sum() / c[i].sum() for i in (rng.integers(0, len(s), len(s)) for _ in range(reps))]
    return float((a - b).mean()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))


def main() -> None:
    df = load()
    strict = pd.read_parquet(D / "exp01_features_bp_strict.parquet")[
        ["rid16", "ban"] + [c for c in pd.read_parquet(D / "exp01_features_bp_strict.parquet").columns
                            if c.startswith("bp_")]]
    strict["rid16"] = strict["rid16"].astype(str)
    strict = strict.rename(columns={c: c + "_strict" for c in strict.columns if c.startswith("bp_")})
    df = df.merge(strict, on=["rid16", "ban"], how="left")
    y = df["top3"].to_numpy()
    yr = df["year"].to_numpy()
    train = (yr >= 2016) & (yr <= 2021)
    sel = yr == 2022
    conf = yr == 2023
    exp = (yr >= 2024) & (yr <= 2025)
    pooled = conf | exp
    day = df["day"].to_numpy()

    F = SPEC["features"]
    raw, ss, bp = F["raw_choice"], F["deviation_simple"], F["deviation_behavior"]
    bps = [c + "_strict" for c in bp]
    base = ["f_v6", "f_mkt"]
    M = {"M1": base, "M2": base + raw, "M3": base + ss + bp, "M4": base + raw + ss + bp,
         "M3s": base + ss + bps, "M4s": base + raw + ss + bps}
    P = {k: fit_predict(df, v, y, train, sel)[0] for k, v in M.items()}
    res = {"note": "仕様固定後の探索分析。主判定 (Gate 2 = 逸脱固有の価値なし) を変更しない"}

    # 1. 競馬場別
    place = df["rid16"].str[8:10].map(PLACE).to_numpy()
    vrows = []
    for a, b in [("M3", "M1"), ("M4", "M1"), ("M4", "M2")]:
        d = ll_vec(y, P[a]) - ll_vec(y, P[b])
        for pl in PLACE.values():
            m = pooled & (place == pl)
            vrows.append({"compare": f"{a} vs {b}", "place": pl, "n": int(m.sum()), "delta": float(d[m].mean())})
    vdf = pd.DataFrame(vrows)
    vdf.to_csv(OUT / "exploratory_venue.csv", index=False, encoding="utf-8-sig")
    res["venue"] = {c: {"negative_places": int((g.delta < 0).sum()), "of": int(len(g)),
                        "min": float(g.delta.min()), "max": float(g.delta.max())}
                    for c, g in vdf.groupby("compare")}
    # 最も寄与の大きい競馬場を除いても残るか
    for a, b in [("M3", "M1"), ("M4", "M1")]:
        d = ll_vec(y, P[a]) - ll_vec(y, P[b])
        g = vdf[vdf["compare"] == f"{a} vs {b}"].assign(contrib=lambda x: x.delta * x.n).sort_values("contrib")
        worst = g.place.iloc[0]
        m = pooled & (place != worst)
        res["venue"][f"{a} vs {b}"]["drop_top_place"] = {"place": worst, "delta_rest": float(d[m].mean())}

    # 2. 順位
    rrows = []
    R = {k: {pn: rank_metrics(df, P[k], m) for pn, m in [("confirm_2023", conf), ("oos_2024_25", exp)]}
         for k in ["M1", "M2", "M3", "M4"]}
    for a, b in [("M3", "M1"), ("M4", "M1"), ("M4", "M2")]:
        for pn in ["confirm_2023", "oos_2024_25"]:
            ra, rb = R[a][pn], R[b][pn]
            for met in ["rho", "pair_acc"]:
                dlt, lo, hi = boot_mean_diff(ra[met].to_numpy(), rb[met].to_numpy(), ra["day"].to_numpy())
                rrows.append({"compare": f"{a} vs {b}", "period": pn, "metric": met,
                              "a": float(ra[met].mean()), "b": float(rb[met].mean()),
                              "delta": dlt, "ci_lo": lo, "ci_hi": hi})
    pd.DataFrame(rrows).to_csv(OUT / "exploratory_rank.csv", index=False, encoding="utf-8-sig")
    res["rank"] = rrows

    # 3. 厳格版 §7
    band = pd.cut(df["rank_mkt_pre"], [0, 1, 3, 6, 99], labels=["1", "2-3", "4-6", "7+"]).astype(str).to_numpy()
    strict_rows = []
    for a, b in [("M3s", "M1"), ("M4s", "M1"), ("M4s", "M2")]:
        out = {"compare": f"{a} vs {b}"}
        for pn, m in [("confirm_2023", conf), ("oos_2024", yr == 2024), ("oos_2025", yr == 2025)]:
            out[pn] = delta_boot(y[m], P[a][m], P[b][m], day[m], level=0.975)
        d = ll_vec(y, P[a]) - ll_vec(y, P[b])
        t = pd.DataFrame({"d": d[pooled], "tr": df["trainer"].to_numpy()[pooled]})
        drop = set(t.groupby("tr")["d"].sum().sort_values().index[:20])
        rest = float(t[~t.tr.isin(drop)]["d"].mean())
        bands = {bb: float(d[pooled & (band == bb)].mean()) for bb in ["1", "2-3", "4-6", "7+"]}
        c = [out["confirm_2023"]["delta"] < 0 and out["confirm_2023"]["ci_hi"] < 0,
             out["oos_2024"]["delta"] < 0 and out["oos_2025"]["delta"] < 0,
             rest < 0, sum(v < 0 for v in bands.values()) >= 3]
        out.update({"drop_top20_trainers_delta": rest, "popband_delta": bands, "conditions": c, "pass": all(c)})
        strict_rows.append(out)
    res["strict_s7"] = strict_rows

    (OUT / "exploratory_supplement.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str),
                                                     encoding="utf-8")
    print("[1] 競馬場別 (2023-25 合算, Δlogloss<0 の場数)")
    for k, v in res["venue"].items():
        print(f"  {k:<9} {v['negative_places']}/{v['of']} 場で改善  範囲 [{v['min']:+.5f}, {v['max']:+.5f}]"
              + (f"  最大寄与の{v['drop_top_place']['place']}を除くと {v['drop_top_place']['delta_rest']:+.6f}"
                 if "drop_top_place" in v else ""))
    print("\n[2] 順位 (レース内 Spearman / ペア正解率, 95%CI)")
    for r in rrows:
        print(f"  {r['compare']:<9} {r['period']:<13} {r['metric']:<9} Δ={r['delta']:+.5f} [{r['ci_lo']:+.5f},{r['ci_hi']:+.5f}]")
    print("\n[3] 厳格版 §7 (過去結果由来の文脈を外した行動モデル), 97.5%CI")
    for r in strict_rows:
        c3 = r["confirm_2023"]
        print(f"  {r['compare']:<10} 2023 Δ={c3['delta']:+.6f} [{c3['ci_lo']:+.6f},{c3['ci_hi']:+.6f}] "
              f"| 2024 {r['oos_2024']['delta']:+.6f} | 2025 {r['oos_2025']['delta']:+.6f} "
              f"| 調教師除外 {r['drop_top20_trainers_delta']:+.6f} → {'PASS' if r['pass'] else 'fail'}")


if __name__ == "__main__":
    main()
