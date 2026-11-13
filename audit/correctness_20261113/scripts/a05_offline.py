"""A05: 本番 v6 (C1) を master の valid 2023 / test 2024-25 で再採点。
条件: full = master フル特徴 / serve = data/serve_feature_baseline.json の cov<40% をマスク
較正器: v6 (pl_calibrators_v6.pkl) / v6_serve (pl_calibrators_v6_serve.pkl)
指標: ◎勝率・◎3着内 (開催日クラスタ CI)・p_win / p_sho の Brier / logloss / ECE、年別。
read-only。usage: python a05_offline.py <ROOT>   (cwd=ROOT で実行し import を通す)
"""
import json, sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("E:/PyCaLiAI")
sys.path.insert(0, str(ROOT))
import pl_probs as PL  # noqa
from backtest_pl_ev import apply_encoders, all_fukusho_vec_fast  # noqa
from serve_skew_eval import load_current_serve_mask  # noqa

OUT = Path(__file__).resolve().parents[1] / "out"
RID, BAN, JYUN = "レースID(新/馬番無)", "馬番", "着順"
rng = np.random.default_rng(0)

b = joblib.load(ROOT / "models/unified_rank_v6.pkl")
model, feats, encs = b["model"], b["feature_cols"], b["encoders"]
cal_full = joblib.load(ROOT / "models/pl_calibrators_v6.pkl")["calibrators"]
cal_serve = joblib.load(ROOT / "models/pl_calibrators_v6_serve.pkl")["calibrators"]
dead_num, dead_cat, _ = load_current_serve_mask(feats, encs)

hdr = pd.read_csv(ROOT / "data/master_v2_20130105-20251228.csv", nrows=0, encoding="utf-8-sig").columns
use = list(dict.fromkeys([c for c in feats + [RID, BAN, JYUN, "split", "クラス名", "出走頭数", "芝・ダ"] if c in hdr]))
df = pd.read_csv(ROOT / "data/master_v2_20130105-20251228.csv", usecols=use, encoding="utf-8-sig", low_memory=False)
df[JYUN] = pd.to_numeric(df[JYUN], errors="coerce")
df = df.dropna(subset=[JYUN, RID, "split"])
df = df[df["split"].isin(["valid", "test"])].copy()
df["date"] = df[RID].astype(str).str[:8]
df["year"] = df["date"].str[:4]
df["cls_raw"] = df["クラス名"].astype(str)
df["n_raw"] = pd.to_numeric(df["出走頭数"], errors="coerce")
df["td_raw"] = df["芝・ダ"].astype(str)


def score(frame, serve):
    d = frame.copy()
    if serve:
        for c in dead_cat:
            d[c] = "__NaN__"
    d = apply_encoders(d, encs)
    if serve:
        for c in dead_num:
            d[c] = np.nan
    X = d[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    return model.predict(X)


df["s_full"] = score(df, False)
df["s_serve"] = score(df, True)

recs = []
for rid, g in df.groupby(RID, sort=False):
    if len(g) < 5:
        continue
    pos = g[JYUN].to_numpy()
    for cond in ["full", "serve"]:
        s = g[f"s_{cond}"].to_numpy()
        w = PL.pl_weights(s)
        pw = PL.all_tansho(w)
        ps = all_fukusho_vec_fast(w)
        order = np.argsort(-s)
        for calname, cal in [("raw", None), ("v6", cal_full), ("v6serve", cal_serve)]:
            if cal is None:
                cw, cs = pw, ps
            else:
                cw, cs = cal["tansho"].predict(pw), cal["fukusho"].predict(ps)
            recs.append({"rid": rid, "date": g["date"].iat[0], "year": g["year"].iat[0],
                         "cls": g["cls_raw"].iat[0], "n": len(g), "td": g["td_raw"].iat[0],
                         "cond": cond, "cal": calname,
                         "hon_win": int(pos[order[0]] == 1), "hon_top3": int(pos[order[0]] <= 3),
                         "pw": cw, "ps": cs, "yw": (pos == 1).astype(int), "y3": (pos <= 3).astype(int),
                         "sum_pw": float(cw.sum()), "hon_pw_tied": int((cw == cw[order[0]]).sum() > 1),
                         "pw_zero": int((cw == 0).sum())})
R = pd.DataFrame(recs)


def ece(p, y, bins=10):
    q = np.quantile(p, np.linspace(0, 1, bins + 1)); q[0], q[-1] = -1, 2
    idx = np.clip(np.searchsorted(q, p, side="right") - 1, 0, bins - 1)
    return float(sum((idx == k).mean() * abs(p[idx == k].mean() - y[idx == k].mean()) for k in range(bins) if (idx == k).any()))


def metrics(sub):
    pw = np.concatenate(sub["pw"].to_list()); yw = np.concatenate(sub["yw"].to_list())
    ps = np.concatenate(sub["ps"].to_list()); y3 = np.concatenate(sub["y3"].to_list())
    c = lambda p: np.clip(p, 1e-4, 1 - 1e-4)
    return {"n_races": int(len(sub)), "hon_win": round(sub.hon_win.mean(), 4), "hon_top3": round(sub.hon_top3.mean(), 4),
            "brier_w": round(float(np.mean((pw - yw) ** 2)), 5), "ll_w": round(float(-np.mean(yw * np.log(c(pw)) + (1 - yw) * np.log(1 - c(pw)))), 5),
            "ece_w": round(ece(pw, yw), 5), "brier_3": round(float(np.mean((ps - y3) ** 2)), 5),
            "ll_3": round(float(-np.mean(y3 * np.log(c(ps)) + (1 - y3) * np.log(1 - c(ps)))), 5),
            "ece_3": round(ece(ps, y3), 5), "mean_pw": round(float(pw.mean()), 4), "mean_yw": round(float(yw.mean()), 4),
            "mean_ps": round(float(ps.mean()), 4), "mean_y3": round(float(y3.mean()), 4),
            "sum_pw_p5_p95": [round(float(sub.sum_pw.quantile(.05)), 3), round(float(sub.sum_pw.quantile(.95)), 3)],
            "frac_hon_pw_tied": round(float(sub.hon_pw_tied.mean()), 4),
            "frac_rows_pw_zero": round(float(sub.pw_zero.sum() / len(pw)), 4)}


def ci(sub, col, B=1000):
    dates = sub["date"].unique(); gb = dict(tuple(sub.groupby("date"))); v = []
    for _ in range(B):
        s = rng.choice(dates, len(dates)); v.append(pd.concat([gb[x] for x in s])[col].mean())
    return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]


res = {"serve_mask": {"num": dead_num, "cat": dead_cat}}
for (cond, cal), sub in R.groupby(["cond", "cal"]):
    for per, ss in [("valid2023", sub[sub.year == "2023"]), ("test2024_25", sub[sub.year >= "2024"]),
                    ("test2024", sub[sub.year == "2024"]), ("test2025", sub[sub.year == "2025"])]:
        m = metrics(ss)
        if cal == "v6" and per.startswith("test2024_25"):
            m["hon_top3_ci"] = ci(ss, "hon_top3"); m["hon_win_ci"] = ci(ss, "hon_win")
        res[f"{cond}|{cal}|{per}"] = m
# 層別 (serve|v6serve|test)
t = R[(R.cond == "serve") & (R.cal == "v6serve") & (R.year >= "2024")].copy()
t["fs"] = pd.cut(t["n"], [0, 7, 11, 14, 18], labels=["<=7", "8-11", "12-14", "15-18"])
strata = {}
for col in ["cls", "fs", "td"]:
    strata[col] = [{col: str(k), "n": int(len(s)), "hon_top3": round(s.hon_top3.mean(), 3), "hon_win": round(s.hon_win.mean(), 3),
                    "ece_3": round(ece(np.concatenate(s.ps.to_list()), np.concatenate(s.y3.to_list())), 4)}
                   for k, s in t.groupby(col, observed=True) if len(s) >= 50]
res["strata_serve_v6serve_test"] = strata
json.dump(res, open(OUT / "a05_offline.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for k, v in res.items():
    if k not in ("serve_mask",):
        print(k, json.dumps(v, ensure_ascii=False))
