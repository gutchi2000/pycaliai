"""A04: 2026 本番 bundle (事前予測) を data/kekka の確定着順と突合し、印精度・確率校正を再計算。

usage: python a04_live_perf.py <ROOT>
出力: out/a04_live_perf.json, out/a04_live_rows.parquet
注意:
  - bundle の mtime が開催日 09:30 より後なら「事後再生成」と判定し、別集計する。
  - kekka 確定着順 0 (取消/除外/中止の区別なし) と bundle にいて kekka に居ない馬は分母から除外。
  - 市場比較の tansho_odds は bundle 生成時点 (前夜〜朝) の単勝オッズ。締切オッズではない。
  - 不確実性は開催日単位のクラスタ bootstrap (2000 回)。
"""
import glob, json, os, sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("E:/PyCaLiAI")
OUT = Path(__file__).resolve().parents[1] / "out"
rng = np.random.default_rng(0)

rows, meta = [], []
for bp in sorted(glob.glob(str(ROOT / "reports/cowork_input/2026*_bundle.json"))):
    date = Path(bp).name[:8]
    kp = ROOT / f"data/kekka/{date}.csv"
    if not kp.exists():
        continue
    b = json.load(open(bp, encoding="utf-8"))
    mt = datetime.fromtimestamp(os.path.getmtime(bp))
    race_start = datetime.strptime(date + "0930", "%Y%m%d%H%M")
    sp = b.get("serve_path") or {}
    meta.append({"date": date, "mtime": mt.isoformat(timespec="minutes"),
                 "pre_race": mt < race_start, "model": b.get("model"),
                 "serve_path": sp.get("version"),
                 "commit": (sp.get("git") or {}).get("commit"),
                 "py_dirty": (sp.get("git") or {}).get("py_dirty"),
                 "model_sha": ((sp.get("model") or {}).get("sha256") or "")[:12],
                 "cal": (sp.get("calibrator") or {}).get("path"),
                 "hist_max_date": (sp.get("history") or {}).get("max_date"),
                 "n_races": len(b["races"])})
    k = pd.read_csv(kp, encoding="cp932", dtype=str)
    k["rid16"] = k["レースID(新)"].str[:16]
    k["ban"] = pd.to_numeric(k["馬番"], errors="coerce").astype("Int64")
    k["pos"] = pd.to_numeric(k["確定着順"], errors="coerce")
    kk = {(r.rid16, int(r.ban)): r.pos for r in k.itertuples() if pd.notna(r.ban)}
    krace = set(k["rid16"])
    for r in b["races"]:
        rid = str(r["race_id"])[:16]
        rm = r.get("race_meta", {})
        iq = r.get("input_quality") or {}
        for h in r["horses"]:
            rows.append({
                "date": date, "rid": rid, "ban": int(h["umaban"]), "mark": h.get("mark") or "",
                "ai_rank": h.get("ai_rank"), "score": h.get("ai_score"),
                "p_win": h.get("p_win"), "p_plc": h.get("p_plc"), "p_sho": h.get("p_sho"),
                "odds": h.get("tansho_odds"), "pos": kk.get((rid, int(h["umaban"]))),
                "race_in_kekka": rid in krace, "field": rm.get("field_size"),
                "cls": rm.get("class"), "place": rm.get("place"), "course": rm.get("course"),
                "bet_eligible": iq.get("bet_eligible"),
                "pre_race": meta[-1]["pre_race"],
            })

df = pd.DataFrame(rows)
df.to_parquet(OUT / "a04_live_rows.parquet")
res = {"bundles": meta}
res["n_bundle_dates"] = len(meta)
res["n_regenerated_after_race"] = int(sum(not m["pre_race"] for m in meta))
res["regenerated_dates"] = [m["date"] for m in meta if not m["pre_race"]]
res["py_dirty_dates"] = [m["date"] for m in meta if m["py_dirty"]]
res["model_sha_by_date"] = pd.Series({m["date"]: m["model_sha"] for m in meta}).value_counts().to_dict()
res["calibrator_by_date"] = pd.Series({m["date"]: m["cal"] for m in meta}).value_counts(dropna=False).to_dict()

# 突合
res["rows_total"] = len(df)
res["races_total"] = int(df["rid"].nunique())
res["races_not_in_kekka"] = int(df.loc[~df.race_in_kekka, "rid"].nunique())
df = df[df.race_in_kekka]
res["horses_missing_in_kekka"] = int(df["pos"].isna().sum())
res["horses_pos0"] = int((df["pos"] == 0).sum())
fin = df[(df["pos"] > 0)].copy()
fin["win"] = (fin["pos"] == 1).astype(int)
fin["top3"] = (fin["pos"] <= 3).astype(int)


# 確率制約 (分母: bundle 全馬)
def constraints(d):
    g = d.groupby("rid")
    s = g["p_win"].sum()
    s3 = g["p_sho"].sum()
    out = {"sum_p_win_q": s.quantile([0, .05, .5, .95, 1]).round(3).to_dict(),
           "sum_p_sho_q": s3.quantile([0, .05, .5, .95, 1]).round(3).to_dict(),
           "frac_races_sum_pwin_outside_0.9_1.1": float(((s < .9) | (s > 1.1)).mean()),
           "rows_p_win_eq_0": int((d["p_win"] == 0).sum()),
           "frac_rows_p_win_eq_0": float((d["p_win"] == 0).mean()),
           "rows_p_win_gt_p_sho": int((d["p_win"] > d["p_sho"] + 1e-9).sum()),
           "rows_p_win_gt_p_plc": int((d["p_win"] > d["p_plc"] + 1e-9).sum()),
           "rows_p_plc_gt_p_sho": int((d["p_plc"] > d["p_sho"] + 1e-9).sum())}
    # 印順と表示確率の逆転: ◎の p_win が レース内最大でない
    def inv(gr, col):
        h = gr[gr["mark"] == "◎"]
        if h.empty:
            return np.nan
        return float(h[col].iloc[0] < gr[col].max() - 1e-9)
    def tie(gr, col):
        h = gr[gr["mark"] == "◎"]
        if h.empty:
            return np.nan
        return float(((gr[col] == h[col].iloc[0]).sum() > 1))
    out["frac_races_hon_not_max_p_win"] = float(np.nanmean([inv(gr, "p_win") for _, gr in g]))
    out["frac_races_hon_not_max_p_sho"] = float(np.nanmean([inv(gr, "p_sho") for _, gr in g]))
    out["frac_races_hon_p_win_tied"] = float(np.nanmean([tie(gr, "p_win") for _, gr in g]))
    # ai_rank 順 vs p_win 順の Kendall 的不一致 (隣接ペア)
    bad = tot = 0
    for _, gr in g:
        gg = gr.sort_values("ai_rank")
        pw = gg["p_win"].to_numpy()
        tot += len(pw) - 1
        bad += int((np.diff(pw) > 1e-9).sum())
    out["adjacent_rank_pairs_p_win_increasing"] = bad / tot if tot else None
    return out


res["constraints_all"] = constraints(df)


def ece(p, y, bins=10):
    q = np.quantile(p, np.linspace(0, 1, bins + 1))
    q[0], q[-1] = -1, 2
    idx = np.clip(np.searchsorted(q, p, side="right") - 1, 0, bins - 1)
    e, tab = 0.0, []
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        e += m.mean() * abs(p[m].mean() - y[m].mean())
        tab.append([int(m.sum()), round(float(p[m].mean()), 4), round(float(y[m].mean()), 4)])
    return e, tab


def calib(d, pcol, ycol):
    p = d[pcol].astype(float).to_numpy()
    y = d[ycol].to_numpy()
    pc = np.clip(p, 1e-4, 1 - 1e-4)
    e, tab = ece(p, y)
    return {"n": int(len(p)), "mean_p": round(float(p.mean()), 4), "mean_y": round(float(y.mean()), 4),
            "brier": round(float(np.mean((p - y) ** 2)), 5),
            "logloss": round(float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))), 5),
            "ece_q10": round(float(e), 5), "reliability_q10": tab}


def marks(d):
    hon = d[d["mark"] == "◎"]
    out = {"n_races": int(d["rid"].nunique()), "n_hon": int(len(hon)),
           "hon_win": round(float(hon["win"].mean()), 4), "hon_top3": round(float(hon["top3"].mean()), 4),
           "hon_mean_p_win": round(float(hon["p_win"].mean()), 4),
           "hon_mean_p_sho": round(float(hon["p_sho"].mean()), 4)}
    for mk in ["〇", "▲", "△"]:
        m = d[d["mark"] == mk]
        out[f"{mk}_top3"] = round(float(m["top3"].mean()), 4) if len(m) else None
    return out


def cluster_ci(d, fn, B=2000):
    dates = d["date"].unique()
    gb = {k: v for k, v in d.groupby("date")}
    vals = []
    for _ in range(B):
        s = rng.choice(dates, len(dates), replace=True)
        vals.append(fn(pd.concat([gb[x] for x in s])))
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


# 市場 (bundle 時点の朝オッズ) の確率: 1/odds をレース内で正規化
fin["p_mkt"] = 1 / fin["odds"]
fin["p_mkt"] = fin["p_mkt"] / fin.groupby("rid")["p_mkt"].transform("sum")
fin["mkt_rank"] = fin.groupby("rid")["odds"].rank(method="first")
res["hon_scratched_races"] = int(df[(df["mark"] == "◎") & ((df["pos"] == 0) | df["pos"].isna())]["rid"].nunique())

for label, sub in [("all", fin), ("pre_race_only", fin[fin.pre_race]),
                   ("regenerated_only", fin[~fin.pre_race])]:
    if sub.empty:
        continue
    r = {"marks": marks(sub), "calib_p_win": calib(sub, "p_win", "win"),
         "calib_p_sho": calib(sub, "p_sho", "top3"),
         "calib_p_plc_vs_top2": calib(sub.assign(top2=(sub.pos <= 2).astype(int)), "p_plc", "top2")}
    r["hon_top3_ci_dayclust"] = cluster_ci(sub, lambda x: x.loc[x["mark"] == "◎", "top3"].mean())
    r["hon_win_ci_dayclust"] = cluster_ci(sub, lambda x: x.loc[x["mark"] == "◎", "win"].mean())
    mk = sub.dropna(subset=["p_mkt"])
    r["market_fav_top3"] = round(float(mk.loc[mk.mkt_rank == 1, "top3"].mean()), 4)
    r["market_fav_win"] = round(float(mk.loc[mk.mkt_rank == 1, "win"].mean()), 4)
    r["n_races_with_odds"] = int(mk["rid"].nunique())
    r["calib_market_p_win"] = calib(mk, "p_mkt", "win")
    r["calib_ai_p_win_same_rows"] = calib(mk, "p_win", "win")
    # AI ◎ - 市場1番人気 の 3着内率差 (同一レース、開催日クラスタ CI)
    def diff(x):
        a = x.loc[x["mark"] == "◎", "top3"].mean()
        b_ = x.loc[x["mkt_rank"] == 1, "top3"].mean()
        return a - b_
    r["hon_minus_fav_top3"] = round(float(diff(mk)), 4)
    r["hon_minus_fav_top3_ci"] = cluster_ci(mk, diff)
    def bdiff(x):
        return float(np.mean((x.p_win - x.win) ** 2) - np.mean((x.p_mkt - x.win) ** 2))
    r["brier_ai_minus_mkt_pwin"] = round(bdiff(mk), 6)
    r["brier_ai_minus_mkt_pwin_ci"] = cluster_ci(mk, bdiff)
    res[label] = r

# 層別 (pre_race_only がほぼ全てなら all で)
fin["month"] = fin["date"].str[:6]
fin["fs_bin"] = pd.cut(fin["field"], [0, 7, 11, 14, 18], labels=["<=7", "8-11", "12-14", "15-18"])
fin["surface"] = fin["course"].str[0]
strata = {}
for col in ["month", "cls", "fs_bin", "place", "surface"]:
    t = []
    for k_, sub in fin.groupby(col, observed=True):
        hon = sub[sub.mark == "◎"]
        if len(hon) < 30:
            continue
        e, _ = ece(sub["p_sho"].to_numpy(float), sub["top3"].to_numpy())
        t.append({col: str(k_), "n_races": int(len(hon)), "hon_win": round(float(hon.win.mean()), 3),
                  "hon_top3": round(float(hon.top3.mean()), 3),
                  "mean_p_sho_hon": round(float(hon.p_sho.mean()), 3),
                  "mean_p_win_hon": round(float(hon.p_win.mean()), 3),
                  "ece_p_sho": round(float(e), 4)})
    strata[col] = t
res["strata"] = strata
json.dump(res, open(OUT / "a04_live_perf.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps({k: v for k, v in res.items() if k not in ("bundles", "strata")}, ensure_ascii=False, indent=1, default=str))
print(json.dumps(res["strata"], ensure_ascii=False))
