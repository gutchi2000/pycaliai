# -*- coding: utf-8 -*-
"""
absorption.py — L-12 「遅い資金」吸収率の年次系列と 2020 無観客開催の断点検定
==========================================================================
事前固定: docs/research/L12_SLOW_MONEY_PREREG_20261022.md (sha256 を out/ に記録)
EXP 番号なし。市場構造の記述。ROI 評価・買い目生成はしない。本番ファイルは読むだけ。

入力 (読み取りのみ):
  data/Time _series_odds/TANPUK_*.csv   単勝オッズ時系列 (区分1=途中 / 4=確定)
  data/master_v2_20130105-20251228.csv  着順・トラックコード・発走時刻
出力: analysis/slow_money/out/
  race_level.parquet       レース単位の LL / 時刻 / 票数
  yearly_series.csv        年次系列 (記述)
  flb_beta.csv             本命-大穴勾配 β (年 × スナップ)
  window_series.csv        各年の窓 W (0229-1004) の量
  break_test.json          2020 断点検定 (1 本) + placebo + bootstrap
  daily_2020.csv           2020 開催日ごとの V_pre・G_pre (日付の裏付け用、記述)
  exp16a_crosscheck.csv    EXP16A §3 との突合
実行: E:\\PyCaLiAI\\venv311\\Scripts\\python.exe analysis\\slow_money\\absorption.py
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
OUT = HERE / "out"
MASTER = BASE / "data" / "master_v2_20130105-20251228.csv"
ODIR = BASE / "data" / "Time _series_odds"
PREREG = BASE / "docs" / "research" / "L12_SLOW_MONEY_PREREG_20261022.md"

MIN_GAP_PRE = 15
JUMP_MIN, JUMP_MAX = 51, 59
WIN_LO, WIN_HI = 229, 1004                 # 窓 W (月日)
TREAT_LO, TREAT_HI = 20200229, 20201004    # 暫定無観客期間 (needs_user)
ELIGIBLE = [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2022, 2023]
SNAPS = ["eve", "am", "pre"]
N_BOOT = 2000
SEED = 20261022


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ---------------------------------------------------------------- 読み込み
def load_master() -> pd.DataFrame:
    cols = ["日付", "レースID(新/馬番無)", "馬番", "着順", "発走時刻", "トラックコード(JV)"]
    df = pd.read_csv(MASTER, encoding="utf-8-sig", dtype=str, usecols=cols)
    df["rid16"] = df["レースID(新/馬番無)"].str.replace(r"\.0$", "", regex=True).str[:16]
    df["ban"] = pd.to_numeric(df["馬番"], errors="coerce")
    df["jyun"] = pd.to_numeric(df["着順"], errors="coerce")
    df["tc"] = pd.to_numeric(df["トラックコード(JV)"], errors="coerce")
    hm = df["発走時刻"].astype(str).str.extract(r"(\d{1,2}):(\d{2})")
    df["post_min"] = pd.to_numeric(hm[0], errors="coerce") * 60 + pd.to_numeric(hm[1], errors="coerce")
    df = df.dropna(subset=["ban"])
    g = df.groupby("rid16")
    meta = pd.DataFrame({
        "is_jump": g["tc"].first().between(JUMP_MIN, JUMP_MAX),
        "post_min": g["post_min"].first(),
        "n_winner": g["jyun"].apply(lambda s: int((s == 1).sum())),
        "fin_set": df[df["jyun"].notna()].groupby("rid16")["ban"].apply(lambda s: frozenset(int(x) for x in s)),
    })
    w = df[df["jyun"] == 1].groupby("rid16")["ban"].first()
    meta["winner"] = w
    return meta


def load_tanpuk() -> tuple[pd.DataFrame, np.ndarray]:
    parts = [pd.read_csv(f, encoding="cp932", dtype=str, low_memory=False)
             for f in sorted(ODIR.glob("TANPUK_*.csv"))]
    tp = pd.concat(parts, ignore_index=True)
    c = list(tp.columns)
    tan = {}
    for x in c:
        m = re.match(r"^\s*(\d+)\s*単\s*$", str(x))
        if m:
            tan[int(m.group(1))] = x
    tp["rid16"] = tp[c[0]].astype(str).str.replace(r"\D", "", regex=True).str[:16]
    tp = tp[tp["rid16"].str.len() == 16].copy()
    tp["kb"] = pd.to_numeric(tp[c[1]], errors="coerce")
    tm = pd.to_numeric(tp[c[2]], errors="coerce")
    tp = tp[tm.notna() & tp["kb"].notna()].copy()
    tp["tm"] = tm[tm.notna()].astype(np.int64)
    tp["year"] = tp["rid16"].str[:4].astype(int)
    tp["race_mmdd"] = tp["rid16"].str[4:8].astype(int)
    tp["snap_mmdd"] = tp["tm"] // 10000
    mm, dd = tp["tm"] // 1000000, (tp["tm"] // 10000) % 100
    hh, mi = (tp["tm"] // 100) % 100, tp["tm"] % 100
    tp["ts"] = pd.to_datetime(dict(year=tp["year"], month=mm, day=dd, hour=hh, minute=mi), errors="coerce")
    tp["votes"] = pd.to_numeric(tp["単勝票数"], errors="coerce")
    O = np.full((len(tp), 18), np.nan)
    for n in range(1, 19):
        if n in tan:
            O[:, n - 1] = pd.to_numeric(tp[tan[n]], errors="coerce").to_numpy()
    tp = tp[["rid16", "kb", "tm", "ts", "year", "race_mmdd", "snap_mmdd", "votes"]].reset_index(drop=True)
    return tp, O


# ---------------------------------------------------------------- スナップ選択
def pick_snaps(tp: pd.DataFrame) -> pd.DataFrame:
    tp = tp.copy()
    tp["row"] = np.arange(len(tp))
    fin = (tp[tp.kb == 4].sort_values("ts").groupby("rid16").tail(1)
           .set_index("rid16")[["row", "ts", "votes"]].rename(columns={"row": "r_final", "ts": "ts_final", "votes": "v_final"}))
    mid = tp[tp.kb == 1].merge(fin[["ts_final"]], left_on="rid16", right_index=True, how="inner")
    mid["gap"] = (mid["ts_final"] - mid["ts"]).dt.total_seconds() / 60.0
    same = mid["snap_mmdd"] == mid["race_mmdd"]
    eve = mid[~same & (mid["snap_mmdd"] < mid["race_mmdd"])].sort_values("ts").groupby("rid16").tail(1)
    sd = mid[same].sort_values("ts")
    am = sd.groupby("rid16").head(1)
    pre = sd[sd["gap"] >= MIN_GAP_PRE].groupby("rid16").tail(1)
    pre = pre[~pre["row"].isin(am["row"])]          # am と同一行なら pre は欠損
    R = fin.copy()
    for name, d in (("eve", eve), ("am", am), ("pre", pre)):
        d = d.set_index("rid16")
        R[f"r_{name}"] = d["row"]
        R[f"gap_{name}"] = d["gap"]
        R[f"v_{name}"] = d["votes"]
        R[f"hhmm_{name}"] = d["tm"] % 10000
    return R


# ---------------------------------------------------------------- 計算
def race_table() -> pd.DataFrame:
    meta = load_master()
    tp, O = load_tanpuk()
    R = pick_snaps(tp)
    R = R.join(meta, how="inner")
    R["date"] = R.index.str[:8].astype(int)
    R["year"] = R["date"] // 10000
    R["mmdd"] = R["date"] % 10000
    funnel = {"races_tanpuk_final": int(len(R))}
    R = R[~R["is_jump"].astype(bool)]
    funnel["after_jump_excl"] = int(len(R))
    R = R[(R["n_winner"] == 1) & R["winner"].notna()]
    funnel["after_unique_winner"] = int(len(R))
    Of = O[R["r_final"].astype(int).to_numpy()]
    starters = np.isfinite(Of) & (Of > 1.0)
    R["n_start"] = starters.sum(1)
    win_idx = R["winner"].astype(int).to_numpy() - 1
    win_in = starters[np.arange(len(R)), win_idx]
    keep = (R["n_start"].to_numpy() >= 5) & win_in
    R, Of, starters, win_idx = R[keep], Of[keep], starters[keep], win_idx[keep]
    funnel["after_field5_winner_in_final"] = int(len(R))
    ar = np.arange(len(R))

    def ll_of(Os):
        inv = np.where(starters, 1.0 / Os, 0.0)
        ok = np.where(starters, np.isfinite(Os) & (Os > 1.0), True).all(1)
        pi = inv / inv.sum(1, keepdims=True)
        return -np.log(np.clip(pi[ar, win_idx], 1e-12, None)), ok, pi

    R["LL_final"], _, pi_f = ll_of(Of)
    PI = {"final": pi_f}
    for s in SNAPS:
        rr = R[f"r_{s}"]
        has = rr.notna().to_numpy()
        Os = np.full_like(Of, np.nan)
        Os[has] = O[rr[has].astype(int).to_numpy()]
        ll, ok, pi = ll_of(Os)
        okk = has & ok
        R[f"LL_{s}"] = np.where(okk, ll, np.nan)
        R[f"ok_{s}"] = okk
        PI[s] = np.where(okk[:, None], pi, np.nan)
        funnel[f"complete_coverage_{s}"] = int(okk.sum())
    R["V_pre"] = 1 - R["v_pre"] / R["v_final"]
    R["min_before_post_pre"] = R["post_min"] - ((R["hhmm_pre"] // 100) * 60 + R["hhmm_pre"] % 100)
    R["dnf"] = [len(set(np.flatnonzero(st) + 1) - fs) > 0 if isinstance(fs, frozenset) else True
                for st, fs in zip(starters, R["fin_set"])]
    R = R.drop(columns=["fin_set"])
    return R, PI, starters, win_idx, funnel


def flb_beta(pi: np.ndarray, starters: np.ndarray, win_idx: np.ndarray, rows: np.ndarray) -> tuple[float, float]:
    """条件付きロジット P(i) ∝ π_i^β の β MLE (Newton)。戻り (β, SE)。"""
    p = pi[rows]
    st = starters[rows] & np.isfinite(p) & (p > 0)
    lp = np.where(st, np.log(np.where(st, p, 1.0)), 0.0)
    lw = lp[np.arange(len(rows)), win_idx[rows]]
    b = 1.0
    for _ in range(50):
        e = np.where(st, np.exp(b * lp), 0.0)
        Z = e.sum(1)
        w = e / Z[:, None]
        m1 = (w * lp).sum(1)
        m2 = (w * lp * lp).sum(1)
        g = (lw - m1).sum()
        h = -(m2 - m1 ** 2).sum()
        step = g / h
        b -= step
        if abs(step) < 1e-10:
            break
    return float(b), float(np.sqrt(-1.0 / h))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    R, PI, starters, win_idx, funnel = race_table()
    R["G_eve"] = R["LL_eve"] - R["LL_final"]
    R["G_am"] = R["LL_am"] - R["LL_final"]
    R["G_pre"] = R["LL_pre"] - R["LL_final"]
    R.reset_index().rename(columns={"index": "rid16"}).to_parquet(OUT / "race_level.parquet", index=False)

    # ---------------- 年次系列 (記述): 共通サンプル = eve/am/pre すべて完全被覆
    allok = R["ok_eve"] & R["ok_am"] & R["ok_pre"]
    rows = []
    for y, g in R.groupby("year"):
        c = g[allok.loc[g.index]]
        p = g[g["ok_pre"]]
        rows.append({
            "year": y, "n_races_pre": len(p), "n_races_common": len(c),
            "LL_final_common": c["LL_final"].mean(), "LL_eve": c["LL_eve"].mean(),
            "LL_am": c["LL_am"].mean(), "LL_pre_common": c["LL_pre"].mean(),
            "G_eve": c["G_eve"].mean(), "G_am": c["G_am"].mean(), "G_pre_common": c["G_pre"].mean(),
            "G_pre_all": p["G_pre"].mean(), "G_pre_all_se": p["G_pre"].std() / np.sqrt(len(p)),
            "R_am": c["G_am"].mean() / c["G_eve"].mean(), "R_pre": c["G_pre"].mean() / c["G_eve"].mean(),
            "V_pre": p["V_pre"].mean(), "V_pre_median": p["V_pre"].median(),
            "votes_final_median": p["v_final"].median(),
            "gap_pre_min_median": p["gap_pre"].median(), "min_before_post_pre_median": p["min_before_post_pre"].median(),
            "gap_am_min_median": g.loc[g["ok_am"], "gap_am"].median(),
            "n_start_mean": p["n_start"].mean(),
            "use": ("descriptive_only_2024_25" if y >= 2024 else
                    "descriptive_only_2021" if y == 2021 else
                    "treated_year" if y == 2020 else "eligible"),
        })
    Y = pd.DataFrame(rows)
    Y.to_csv(OUT / "yearly_series.csv", index=False, encoding="utf-8-sig")

    # ---------------- β (記述)
    brow = []
    pos = np.arange(len(R))
    for y in sorted(R["year"].unique()):
        for s in ["eve", "am", "pre", "final"]:
            mask = (R["year"].to_numpy() == y) & (allok.to_numpy())
            b, se = flb_beta(PI[s], starters, win_idx, pos[mask])
            brow.append({"year": y, "snap": s, "beta": b, "se": se, "n": int(mask.sum())})
        inw = (R["mmdd"].between(WIN_LO, WIN_HI)).to_numpy()
        for s in ["pre", "final"]:
            mask = (R["year"].to_numpy() == y) & allok.to_numpy() & inw
            b, se = flb_beta(PI[s], starters, win_idx, pos[mask])
            brow.append({"year": y, "snap": f"{s}_windowW", "beta": b, "se": se, "n": int(mask.sum())})
    B = pd.DataFrame(brow)
    B.to_csv(OUT / "flb_beta.csv", index=False, encoding="utf-8-sig")

    # ---------------- 窓 W の系列
    P = R[R["ok_pre"]].copy()
    P["inW"] = P["mmdd"].between(WIN_LO, WIN_HI)
    W = P[P["inW"]]
    ws = W.groupby("year").agg(n=("G_pre", "size"), G_pre=("G_pre", "mean"), V_pre=("V_pre", "mean"),
                               LL_final=("LL_final", "mean"), LL_pre=("LL_pre", "mean"),
                               n_days=("date", "nunique")).reset_index()
    outside = P[~P["inW"]].groupby("year")["G_pre"].mean().rename("G_pre_outsideW")
    ws = ws.merge(outside, on="year", how="left")
    ws.to_csv(OUT / "window_series.csv", index=False, encoding="utf-8-sig")

    # ---------------- 断点検定 (1 本)
    def D_of(gmap: dict, T: int) -> float:
        fit_years = [y for y in ELIGIBLE if y != T]
        x = np.array(fit_years, float)
        yv = np.array([gmap[y] for y in fit_years])
        a, b = np.polyfit(x, yv, 1)
        return float(gmap[T] - (a * T + b))

    gmap = dict(zip(ws["year"], ws["G_pre"]))
    D2020 = D_of(gmap, 2020)
    placebo = {int(y): D_of(gmap, y) for y in ELIGIBLE}
    n_ge = sum(abs(v) >= abs(D2020) for v in placebo.values())
    p_plac = (1 + n_ge) / (1 + len(placebo))

    # 開催日ブートストラップ
    rng = np.random.default_rng(SEED)
    day_sum = W.groupby(["year", "date"])["G_pre"].agg(["sum", "size"]).reset_index()
    by_year = {y: (d["sum"].to_numpy(), d["size"].to_numpy()) for y, d in day_sum.groupby("year")}
    yrs = ELIGIBLE + [2020]
    boots = []
    for _ in range(N_BOOT):
        gm = {}
        for y in yrs:
            s, n = by_year[y]
            k = rng.integers(0, len(s), len(s))
            gm[y] = s[k].sum() / n[k].sum()
        boots.append(D_of(gm, 2020))
    boots = np.array(boots)
    ci = [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]
    verdict = "BREAK" if (abs(p_plac - 0.10) < 1e-9 and (ci[0] > 0 or ci[1] < 0)) else "NO_BREAK_DETECTED"

    # 処置窓が暫定期間と一致しているかの確認 (2020 の W = 0229-1004 = 暫定無観客期間)
    t20 = P[(P["date"] >= TREAT_LO) & (P["date"] <= TREAT_HI)]
    fit_years = [y for y in ELIGIBLE if y != 2020]
    a, b = np.polyfit(np.array(fit_years, float), np.array([gmap[y] for y in fit_years]), 1)

    # 記述: 2020 の他区間・β・V_pre の窓比較
    desc = {}
    for lab, lo, hi in (("2020_Jan_Feb28", 20200101, 20200228), ("2020_W_no_spectators", TREAT_LO, TREAT_HI),
                        ("2020_Oct10_Dec_partial", 20201010, 20201231)):
        d = P[(P["date"] >= lo) & (P["date"] <= hi)]
        desc[lab] = {"n": int(len(d)), "G_pre": float(d["G_pre"].mean()), "V_pre": float(d["V_pre"].mean())}
    ctrlW = W[W["year"].isin(fit_years)]
    desc["control_years_W_pooled"] = {"years": fit_years, "n": int(len(ctrlW)),
                                      "G_pre": float(ctrlW["G_pre"].mean()), "V_pre": float(ctrlW["V_pre"].mean())}
    for s in ["G_eve", "G_am"]:
        cc = R[allok & R["mmdd"].between(WIN_LO, WIN_HI)]
        desc[f"{s}_W_by_year"] = {int(y): float(v) for y, v in cc.groupby("year")[s].mean().items()}

    test = {
        "prereg": str(PREREG.relative_to(BASE)), "prereg_sha256": sha256(PREREG),
        "statistic": "D_T = G_pre(T,W) - linear-trend prediction from eligible years except T",
        "window_W_mmdd": [WIN_LO, WIN_HI], "treatment_dates_provisional": [TREAT_LO, TREAT_HI],
        "treatment_dates_status": "needs_user",
        "eligible_years": ELIGIBLE,
        "G_pre_W_2020": float(gmap[2020]), "trend_pred_2020": float(a * 2020 + b),
        "trend_slope_per_year": float(a),
        "D_2020": D2020, "placebo_D": placebo,
        "n_placebo_abs_ge": int(n_ge), "p_placebo_two_sided": p_plac, "p_min_attainable": 0.1,
        "bootstrap": {"unit": "meeting_day (date) within year-window", "n": N_BOOT, "seed": SEED,
                      "ci95": ci, "sd": float(boots.std())},
        "verdict": verdict,
        "n_races_2020_W": int(len(t20)),
        "descriptive_not_tested": desc,
        "funnel": funnel,
        "input_sha256": {p.name: sha256(p) for p in sorted(ODIR.glob("TANPUK_*.csv"))},
    }
    (OUT / "break_test.json").write_text(json.dumps(test, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    # ---------------- 2020 開催日ごと (日付の裏付け、記述)
    d20 = P[P["year"] == 2020].groupby("date").agg(n=("G_pre", "size"), G_pre=("G_pre", "mean"),
                                                    V_pre=("V_pre", "mean"), V_pre_med=("V_pre", "median"),
                                                    votes_final_med=("v_final", "median")).reset_index()
    d20.to_csv(OUT / "daily_2020.csv", index=False, encoding="utf-8-sig")
    # 対照: 2019 の同じ統計
    d19 = P[P["year"] == 2019].groupby("date").agg(n=("G_pre", "size"), G_pre=("G_pre", "mean"),
                                                    V_pre=("V_pre", "mean"), votes_final_med=("v_final", "median")).reset_index()
    d19.to_csv(OUT / "daily_2019.csv", index=False, encoding="utf-8-sig")

    # ---------------- EXP16A §3 突合 (2019-2023, DNF 除外)
    exp16a = {2019: (1.996264, 1.956667), 2020: (2.018685, 1.986472), 2021: (2.002010, 1.951214),
              2022: (1.948054, 1.900596), 2023: (1.977735, 1.929923)}
    xc = []
    for y, (pre_ll, q0) in exp16a.items():
        d = P[(P["year"] == y) & (~P["dnf"])]
        xc.append({"year": y, "n_mine": len(d), "LL_pre_mine": d["LL_pre"].mean(), "LL_final_mine": d["LL_final"].mean(),
                   "gap_mine": d["G_pre"].mean(), "LL_pre_exp16a": pre_ll, "LL_q0_exp16a": q0, "gap_exp16a": pre_ll - q0})
    pd.DataFrame(xc).to_csv(OUT / "exp16a_crosscheck.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(json.dumps(funnel, indent=1))
    print(Y.round(4).to_string())
    print(B.pivot(index="year", columns="snap", values="beta").round(4).to_string())
    print(ws.round(4).to_string())
    print(json.dumps({k: test[k] for k in ["G_pre_W_2020", "trend_pred_2020", "D_2020", "placebo_D",
                                           "p_placebo_two_sided", "bootstrap", "verdict"]}, indent=1, default=float))
    print(json.dumps(desc, indent=1, default=float))
    print(pd.DataFrame(xc).round(4).to_string())


if __name__ == "__main__":
    main()
