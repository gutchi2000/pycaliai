"""A07: a04 の行データを a06 の来歴で分けて再集計 (検証済み発走前 = git の発走前版と内容一致)。
さらに v6 C1 本番モデル期 (2026-09-12 以降) と serve 修正後 (2026-11-07 以降) を別掲。
usage: python a07_live_by_provenance.py
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "out"
rng = np.random.default_rng(1)
df = pd.read_parquet(OUT / "a04_live_rows.parquet")
prov = {r["date"]: r for r in json.load(open(OUT / "a06_bundle_provenance.json", encoding="utf-8"))}
a04 = json.load(open(OUT / "a04_live_perf.json", encoding="utf-8"))
model_by_date = {m["date"]: m["model"] for m in a04["bundles"]}
df["verified_pre"] = df["date"].map(lambda d: prov[d].get("identical_to_pre_race_commit") is True)
df["model"] = df["date"].map(model_by_date)
fin = df[df.race_in_kekka & (df.pos > 0)].copy()
fin["win"] = (fin.pos == 1).astype(int)
fin["top3"] = (fin.pos <= 3).astype(int)
fin["p_mkt"] = 1 / fin["odds"]
fin["p_mkt"] /= fin.groupby("rid")["p_mkt"].transform("sum")
fin["fav"] = fin.groupby("rid")["odds"].rank(method="first") == 1


def ci(d, fn, B=2000):
    dates = d.date.unique(); gb = dict(tuple(d.groupby("date"))); v = []
    for _ in range(B):
        v.append(fn(pd.concat([gb[x] for x in rng.choice(dates, len(dates))])))
    return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]


def summ(d):
    hon = d[d.mark == "◎"]
    f = lambda x: x.loc[x.mark == "◎", "top3"].mean() - x.loc[x.fav, "top3"].mean()
    bri = lambda x: float(np.mean((x.p_win - x.win) ** 2) - np.mean((x.p_mkt - x.win) ** 2))
    bri3 = lambda x: float(np.mean((x.p_sho - x.top3) ** 2))
    return {"n_dates": int(d.date.nunique()), "n_races": int(hon.rid.nunique()),
            "hon_win": round(hon.win.mean(), 4), "hon_win_ci": ci(d, lambda x: x.loc[x.mark == "◎", "win"].mean()),
            "hon_top3": round(hon.top3.mean(), 4), "hon_top3_ci": ci(d, lambda x: x.loc[x.mark == "◎", "top3"].mean()),
            "hon_mean_p_sho": round(hon.p_sho.mean(), 4), "hon_mean_p_win": round(hon.p_win.mean(), 4),
            "fav_top3": round(d.loc[d.fav, "top3"].mean(), 4),
            "hon_minus_fav_top3": round(f(d), 4), "hon_minus_fav_top3_ci": ci(d, f),
            "brier_pwin_ai_minus_mkt": round(bri(d), 5), "brier_pwin_ai_minus_mkt_ci": ci(d, bri),
            "brier_psho": round(bri3(d), 5)}


res = {
    "all_2026": summ(fin),
    "verified_pre_race": summ(fin[fin.verified_pre]),
    "not_verified_or_changed": summ(fin[~fin.verified_pre]),
    "verified_pre_race_v6_only": summ(fin[fin.verified_pre & (fin.model == "v6")]),
    "verified_v6_C1_period_20260912+": summ(fin[fin.verified_pre & (fin.date >= "20260912")]),
    "verified_post_servefix_20261107+": summ(fin[fin.verified_pre & (fin.date >= "20261107")]),
}
json.dump(res, open(OUT / "a07_live_by_provenance.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for k, v in res.items():
    print(k, json.dumps(v, ensure_ascii=False))
