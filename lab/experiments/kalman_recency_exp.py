"""群衆は直近 1 走を重く見すぎるか (カルマン最適からのずれ) — 2026-11-14

理論
----
各走の能力指数 y (JRDB IDM) を y_t = θ_t + ε_t とみる。θ_t は走ごとのランダムウォーク
(増分分散 q)、ε_t は当日の運 (分散 σ²)。2013-2023 の同一馬・同芝ダ・間隔≤120日・距離差≤400m
の連続走のバリオグラム V(k)=2σ²+kq から σ²≈40.7, q≈6.9。この (σ², q) のもとで過去 n 走から
現在の θ を推定する最適線形推定はカルマンフィルタ (diffuse prior) で、重みは直近走に約 1/3、
以降は約 0.66 倍ずつ減衰する。理論の予言: 群衆は直近走をこの最適より重く見る
(代表性ヒューリスティック/直近バイアス)。

事前登録 (結果を見る前に固定)
-----------------------------
対象: 2013-01-01〜2025-12-31 の平地 (td∈{1,2}) 全レース。単勝確定オッズ >0 の馬を出走馬とし、
  市場勝率 p = (1/tan_odds) をレース内で正規化。
履歴: その馬の、当該レース日より前 (ymd 厳密に小さい) の平地走のうち ijou==0 かつ idm 非欠損の走、
  直近最大 10 走。芝ダは混ぜる (主検定)。
適格馬: 過去有効走 ≥3。適格馬が ≥4 頭いるレースだけを使う。
カルマン推定: σ²=40.7, q=6.9 (主)。感度 σ²∈{50, 60}。重みは履歴長 n ごとに diffuse prior で厳密計算。
信号: D = θ̂_Kalman − y_last (直近走の IDM)。レース内の適格馬で z 標準化 z=(D−mean)/sd。
群: z の全期間プール 20/40/60/80 パーセンタイルで五分位 Q1..Q5。
  Q5 = 「直近走が履歴より悪い (カルマン推定の方が高い)」→ 理論上 市場が過小評価。
  Q1 = 「直近走が履歴より良い」→ 理論上 過大評価。
主指標: 実勝/市場 = Σ(1着) / Σ p (群ごと)。CI95 は開催日クラスタ bootstrap (B=2000)。
主判定 (SIGNAL): コントラスト C = (実勝/市場 Q5) / (実勝/市場 Q1) の CI95 が 2013-19 と 2020-25 の
  両方で 1 を除外し、かつ同じ向き。理論通りなら C>1、逆向き (C<1) も同基準で記録する。
副指標:
  (a) 本命-大穴バイアス補正版: 期待勝 = Σ p × c(オッズ帯), c は期ごとに全出走馬で推定。
  (b) 条件付きロジット: 1着 ~ log p + (log p)² + z、z の係数と開催日クラスタ頑健 SE。
  (c) 朝オッズ (odds10_tan) でも同じ主指標・コントラスト。
  (d) 単勝・複勝 均等買い回収率 (控除率 約 20% のため ROI>100% は別問題)。
  (e) 市場の暗黙の重み: 過去有効走 ≥5 の馬に限り、レース内平均を引いた log p を
      y_lag1..y_lag5 (レース内平均差) に OLS → 係数比。同じ説明変数で 1 着の条件付きロジット
      (結果の示す最適重み) とカルマン重み (n=5 の上位 5 本を正規化) と比較。
      直近バイアスの判定量: lag1 の重みシェア (b1/Σb) の 市場 − 結果。
事後追加 (結果を見た後に追加、探索扱い):
  (f) 残差の条件付きロジット: 1着 ~ log p + (log p)² + y_lag1..y_lag5 (過去有効走 ≥5)。
      市場価格を条件にした各過去走の追加効果。lag1 が正・古い走が負なら市場は直近を軽視、逆なら重視。

再実行: worktree root から `python -m lab.experiments.kalman_recency_exp`
出力: reports/kalman_recency_exp.json (集計値のみ。JRDB の馬別値は出力しない)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SED = Path("E:/PyCaLiAI/data/jrdb/sed.parquet")
KEKKA = Path("E:/PyCaLiAI/data/kekka_20130105-20251228.csv")
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "kalman_recency_exp.json"

SIG2_MAIN = 40.7
Q_MAIN = 6.9
SIG2_SENS = [50.0, 60.0]
NHIST = 10
B = 2000
RNG = np.random.default_rng(20261114)
PERIODS = {"2013-19": (20130101, 20191231), "2020-25": (20200101, 20251231)}
ODDS_BANDS = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0, 1e9]


# ---------------------------------------------------------------- Kalman weights
def kalman_weights(n: int, sig2: float, q: float) -> np.ndarray:
    """diffuse prior の局所水準モデルで、n 個の観測 (古→新) から最終時点の θ を推定する線形重み。
    返り値は 新→古 の順 (w[0] = 直近走)。"""
    w = np.zeros(n)
    P = 1e9
    for t in range(n):
        if t > 0:
            P += q
        K = P / (P + sig2)
        w = (1 - K) * w
        w[t] += K
        P = (1 - K) * P
    return w[::-1].copy()


# ---------------------------------------------------------------- data
def load() -> pd.DataFrame:
    cols = ["racekey", "ymd", "ban", "ped", "dist", "td", "jyun", "ijou", "idm",
            "tan_odds", "odds10_tan"]
    d = pd.read_parquet(SED, columns=cols)
    d = d[d.td.isin([1, 2])].copy()
    d["rid"] = d.ymd.astype(str) + d.racekey.astype(str)
    return d


def build_history(d: pd.DataFrame) -> pd.DataFrame:
    v = d[(d.ijou == 0) & d.idm.notna()][["ped", "ymd", "idm"]].copy()
    # 同日に同じ馬が 2 走はない前提。念のため重複除去
    v = v.drop_duplicates(["ped", "ymd"]).sort_values(["ped", "ymd"])
    g = v.groupby("ped", sort=False)["idm"]
    v["h0"] = v["idm"]
    for k in range(1, NHIST):
        v[f"h{k}"] = g.shift(k)
    v["nh"] = g.cumcount() + 1
    v = v.drop(columns="idm").rename(columns={"ymd": "hist_ymd"})
    return v


def attach_history(d: pd.DataFrame, v: pd.DataFrame) -> pd.DataFrame:
    t = d.sort_values("ymd").reset_index(drop=True)
    v = v.sort_values("hist_ymd")
    m = pd.merge_asof(t, v, left_on="ymd", right_on="hist_ymd", by="ped",
                      allow_exact_matches=False, direction="backward")
    m["nh"] = m["nh"].fillna(0).astype(int).clip(upper=NHIST)
    return m


def kalman_estimate(m: pd.DataFrame, sig2: float, q: float) -> np.ndarray:
    H = m[[f"h{k}" for k in range(NHIST)]].to_numpy()
    nh = m["nh"].to_numpy()
    est = np.full(len(m), np.nan)
    for n in range(1, NHIST + 1):
        idx = nh == n
        if not idx.any():
            continue
        w = kalman_weights(n, sig2, q)
        est[idx] = H[idx, :n] @ w
    return est


def place_payouts() -> pd.DataFrame:
    k = pd.read_csv(KEKKA, encoding="cp932", dtype=str)
    k = k.iloc[:, [0, 5]]
    k.columns = ["rid18", "fuku"]
    k["fuku"] = pd.to_numeric(k["fuku"], errors="coerce")
    k = k[k.fuku > 0]
    return k.groupby("rid18", as_index=False)["fuku"].max()


def rid16_from_jrdb(ymd: pd.Series, rk: pd.Series) -> pd.Series:
    rk = rk.astype(str)
    place = rk.str[0:2]
    kai = rk.str[4].astype(int).map(lambda x: f"{x:02d}")
    nichi = rk.str[5].map(lambda c: f"{int(c, 16):02d}")
    r = rk.str[6:8]
    return ymd.astype(str) + place + kai + nichi + r


# ---------------------------------------------------------------- stats helpers
def cluster_sums(df: pd.DataFrame, group: str, cols: list[str]):
    """開催日 × 群 の和の配列 (ndate, ngroup, ncol)。"""
    dates = np.sort(df.ymd.unique())
    di = np.searchsorted(dates, df.ymd.to_numpy())
    groups = sorted(df[group].dropna().unique())
    gi = df[group].map({g: i for i, g in enumerate(groups)}).to_numpy()
    arr = np.zeros((len(dates), len(groups), len(cols)))
    for c, col in enumerate(cols):
        np.add.at(arr[:, :, c], (di, gi), df[col].to_numpy(dtype=float))
    return arr, groups


def boot_weights(ndate: int) -> np.ndarray:
    return RNG.multinomial(ndate, np.full(ndate, 1.0 / ndate), size=B).astype(float)


def ratio_ci(arr, num: int, den: int, W):
    tot = arr.sum(0)
    pt = tot[:, num] / tot[:, den]
    bs = np.einsum("bd,dg->bg", W, arr[:, :, num]) / np.einsum("bd,dg->bg", W, arr[:, :, den])
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    return pt, lo, hi, bs


def ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


# ---------------------------------------------------------------- conditional logit
def clogit(X: np.ndarray, y: np.ndarray, grp: np.ndarray, cl: np.ndarray, iters=30):
    """grp は連続ブロックで並んだレース番号。各レースで y の和=1 を前提。
    返り値: beta, cluster-robust SE (cluster=cl)。"""
    k = X.shape[1]
    beta = np.zeros(k)
    starts = np.r_[0, np.flatnonzero(np.diff(grp)) + 1]
    for _ in range(iters):
        u = X @ beta
        u = u - np.maximum.reduceat(u, starts)[np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(u)]))]
        e = np.exp(u)
        rs = np.add.reduceat(e, starts)
        rep = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(u)]))
        p = e / rs[rep]
        xbar = np.add.reduceat(X * p[:, None], starts)
        g = (X * y[:, None]).sum(0) - xbar.sum(0)
        Xc = X - xbar[rep]
        Hm = (Xc * p[:, None]).T @ Xc
        step = np.linalg.solve(Hm, g)
        beta += step
        if np.abs(step).max() < 1e-9:
            break
    # robust SE
    u = X @ beta
    rep = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(u)]))
    e = np.exp(u - np.maximum.reduceat(u, starts)[rep])
    p = e / np.add.reduceat(e, starts)[rep]
    xbar = np.add.reduceat(X * p[:, None], starts)
    Xc = X - xbar[rep]
    Hm = (Xc * p[:, None]).T @ Xc
    sc = Xc * (y - p)[:, None]
    cls = pd.factorize(cl)[0]
    S = np.zeros((cls.max() + 1, k))
    np.add.at(S, cls, sc)
    Hi = np.linalg.inv(Hm)
    V = Hi @ (S.T @ S) @ Hi
    return beta, np.sqrt(np.diag(V))


# ---------------------------------------------------------------- main analyses
def quintile_tables(e: pd.DataFrame, pcol: str, zcol: str, cuts, label: str, with_roi: bool):
    res = {}
    for pname, (a, b) in PERIODS.items():
        s = e[(e.ymd >= a) & (e.ymd <= b)].copy()
        s["Q"] = np.digitize(s[zcol], cuts) + 1
        # 本命-大穴バイアス補正: 期内の全出走馬 (e と同じ母集団の全頭) でオッズ帯別 c
        s["win"] = (s.jyun == 1).astype(float)
        cols = ["win", pcol, "pflb", "n", "tan_ret", "fuku_ret", "fuku_n"]
        arr, groups = cluster_sums(s, "Q", cols)
        W = boot_weights(arr.shape[0])
        pt, lo, hi, bs = ratio_ci(arr, 0, 1, W)
        pt2, lo2, hi2, bs2 = ratio_ci(arr, 0, 2, W)
        ctr = bs[:, -1] / bs[:, 0]
        ctr2 = bs2[:, -1] / bs2[:, 0]
        rows = []
        for i, g in enumerate(groups):
            row = {"Q": int(g), "n": int(arr[:, i, 3].sum()), "wins": int(arr[:, i, 0].sum()),
                   "sum_p": float(arr[:, i, 1].sum()),
                   "AE": float(pt[i]), "AE_ci": [float(lo[i]), float(hi[i])],
                   "AE_flb": float(pt2[i]), "AE_flb_ci": [float(lo2[i]), float(hi2[i])]}
            if with_roi:
                tr = np.einsum("bd,d->b", W, arr[:, i, 4]) / np.einsum("bd,d->b", W, arr[:, i, 3])
                row["win_roi"] = float(arr[:, i, 4].sum() / arr[:, i, 3].sum())
                row["win_roi_ci"] = ci(tr)
                fr = np.einsum("bd,d->b", W, arr[:, i, 5]) / np.einsum("bd,d->b", W, arr[:, i, 6])
                row["place_roi"] = float(arr[:, i, 5].sum() / arr[:, i, 6].sum())
                row["place_roi_ci"] = ci(fr)
            rows.append(row)
        med = s.groupby("Q").tan_odds.median()
        for row in rows:
            row["median_final_odds"] = float(med.loc[row["Q"]])
        res[pname] = {
            "quintiles": rows,
            "contrast_Q5_over_Q1": float(pt[-1] / pt[0]), "contrast_ci": ci(ctr),
            "contrast_flb": float(pt2[-1] / pt2[0]), "contrast_flb_ci": ci(ctr2),
            "n_dates": int(arr.shape[0]),
        }
    return res


def add_flb(e: pd.DataFrame, allr: pd.DataFrame, pcol: str, ocol: str) -> pd.Series:
    """期ごと・オッズ帯ごとの 実勝/市場 を全出走馬で推定し、p に掛けた期待勝率。"""
    out = pd.Series(np.nan, index=e.index)
    for a, b in PERIODS.values():
        s = allr[(allr.ymd >= a) & (allr.ymd <= b)]
        band = pd.cut(s[ocol], ODDS_BANDS, right=False)
        c = (s.jyun.eq(1).groupby(band, observed=False).sum() / s[pcol].groupby(band, observed=False).sum())
        m = (e.ymd >= a) & (e.ymd <= b)
        eb = pd.cut(e.loc[m, ocol], ODDS_BANDS, right=False)
        out[m] = e.loc[m, pcol] * eb.map(c).astype(float)
    return out


def implied_weights(e: pd.DataFrame, pcol: str, sig2: float, q: float):
    """過去有効走 ≥5 の馬だけでの 市場の暗黙重み と 結果の重み。"""
    lag = [f"h{k}" for k in range(5)]
    res = {}
    kw = kalman_weights(5, sig2, q)
    kw10 = kalman_weights(10, sig2, q)[:5]
    res["kalman_n5"] = (kw / kw.sum()).tolist()
    res["kalman_n10_top5_normalized"] = (kw10 / kw10.sum()).tolist()
    for pname, (a, b) in PERIODS.items():
        s = e[(e.ymd >= a) & (e.ymd <= b) & (e.nh >= 5) & e[pcol].notna()].copy()
        cnt = s.groupby("rid")["rid"].transform("size")
        s = s[cnt >= 4]
        s = s.sort_values(["rid"]).reset_index(drop=True)
        s["lp"] = np.log(s[pcol])
        dm = s[lag + ["lp"]] - s.groupby("rid")[lag + ["lp"]].transform("mean")
        X = dm[lag].to_numpy()
        yv = dm["lp"].to_numpy()
        # OLS (within) + 開催日クラスタ bootstrap
        dates = np.sort(s.ymd.unique())
        di = np.searchsorted(dates, s.ymd.to_numpy())
        XtX = np.zeros((len(dates), 5, 5))
        Xty = np.zeros((len(dates), 5))
        np.add.at(XtX, di, X[:, :, None] * X[:, None, :])
        np.add.at(Xty, di, X * yv[:, None])
        bmk = np.linalg.solve(XtX.sum(0), Xty.sum(0))
        Wb = boot_weights(len(dates))[:400]
        bb = np.array([np.linalg.solve(np.tensordot(w, XtX, 1), w @ Xty) for w in Wb])
        share_m = bb[:, 0] / bb.sum(1)
        # 結果: 1 着の条件付きロジット (勝ち馬が対象馬の中にいるレースのみ)
        win = (s.jyun == 1).astype(float)
        has = win.groupby(s.rid).transform("sum") == 1
        t = s[has.to_numpy()].reset_index(drop=True)
        Xo = (t[lag] - t.groupby("rid")[lag].transform("mean")).to_numpy()
        yo = (t.jyun == 1).astype(float).to_numpy()
        grp = pd.factorize(t.rid)[0]
        bo, se_o = clogit(Xo, yo, grp, t.ymd.to_numpy())
        # 結果のシェアの CI: 開催日 bootstrap 200 回
        tdates = np.sort(t.ymd.unique())
        rid_first = t.groupby("rid", sort=False).ymd.first()
        share_o = []
        rng = np.random.default_rng(7)
        race_codes = pd.factorize(t.rid)[0]
        race_date = t.groupby(race_codes).ymd.first().to_numpy()
        date_races = pd.Series(np.arange(len(race_date))).groupby(race_date).apply(list)
        rows_by_race = np.split(np.arange(len(t)), np.flatnonzero(np.diff(race_codes)) + 1)
        for _ in range(200):
            pick = rng.choice(date_races.index.to_numpy(), size=len(date_races), replace=True)
            rr = np.concatenate([date_races[p] for p in pick])
            idx = np.concatenate([rows_by_race[r] for r in rr])
            g2 = np.repeat(np.arange(len(rr)), [len(rows_by_race[r]) for r in rr])
            bb2, _ = clogit(Xo[idx], yo[idx], g2, g2, iters=12)
            share_o.append(bb2[0] / bb2.sum())
        share_o = np.array(share_o)
        res[pname] = {
            "n_rows": int(len(s)), "n_races": int(s.rid.nunique()),
            "market_coef": bmk.tolist(), "market_weights": (bmk / bmk.sum()).tolist(),
            "market_lag1_share_ci": ci(share_m),
            "outcome_coef": bo.tolist(), "outcome_se": se_o.tolist(),
            "outcome_weights": (bo / bo.sum()).tolist(),
            "outcome_lag1_share_ci": ci(share_o),
            "outcome_n_races": int(t.rid.nunique()),
            "market_minus_outcome_lag1_share": float(bmk[0] / bmk.sum() - bo[0] / bo.sum()),
            "market_minus_outcome_lag1_share_ci_approx": ci(
                share_m[:200] - share_o[:len(share_m[:200])]),
        }
    return res


def main():
    t0 = time.time()
    d = load()
    print(f"loaded {len(d):,} flat rows ({time.time()-t0:.0f}s)", flush=True)
    v = build_history(d)
    m = attach_history(d, v)
    m = m[(m.ymd >= 20130101) & (m.ymd <= 20251231)]
    m = m[m.tan_odds.notna() & (m.tan_odds > 0)].copy()
    m["p_fin"] = (1 / m.tan_odds) / m.groupby("rid").tan_odds.transform(lambda x: (1 / x).sum())
    ok_m = m.odds10_tan.notna() & (m.odds10_tan > 0)
    inv = (1 / m.odds10_tan).where(ok_m)
    # 朝オッズ欠損馬がいるレースは朝の分析から外す
    all_ok = ok_m.groupby(m.rid).transform("all")
    m["p_mor"] = (inv / inv.groupby(m.rid).transform("sum")).where(all_ok)
    m["n"] = 1.0
    m["tan_ret"] = np.where(m.jyun == 1, m.tan_odds, 0.0)  # 100 円あたり払戻/100
    # 複勝払戻
    pp = place_payouts()
    m["rid18"] = rid16_from_jrdb(m.ymd, m.racekey) + m.ban.map(lambda x: f"{x:02d}")
    m = m.merge(pp, on="rid18", how="left")
    m["fuku_ret"] = m.fuku.fillna(0) / 100.0
    rid16 = m.rid18.str[:16]
    has_pay = m.fuku.notna().groupby(rid16).transform("any")
    m["fuku_n"] = np.where(has_pay, 1.0, 0.0)
    m["fuku_ret"] = m.fuku_ret * m.fuku_n
    place_match_rate = float(has_pay.groupby(rid16).first().mean())
    print(f"place payout race match rate {place_match_rate:.4f}", flush=True)

    # coverage
    cov = {}
    for pname, (a, b) in PERIODS.items():
        s = m[(m.ymd >= a) & (m.ymd <= b)]
        cov[pname] = {"runners": int(len(s)), "races": int(s.rid.nunique()),
                      "share_nh_ge3": float((s.nh >= 3).mean()),
                      "share_nh_ge5": float((s.nh >= 5).mean())}

    results = {"config": {"sig2_main": SIG2_MAIN, "q": Q_MAIN, "sig2_sens": SIG2_SENS,
                          "nhist": NHIST, "B": B, "periods": PERIODS},
               "kalman_weights": {str(s2): {str(n): kalman_weights(n, s2, Q_MAIN).round(4).tolist()
                                            for n in (3, 5, 10)}
                                  for s2 in [SIG2_MAIN] + SIG2_SENS},
               "coverage": cov, "place_payout_race_match_rate": place_match_rate}

    m["y_last"] = m["h0"]
    elig = m.nh >= 3
    for s2 in [SIG2_MAIN] + SIG2_SENS:
        m[f"D_{s2}"] = kalman_estimate(m, s2, Q_MAIN) - m.y_last
    e = m[elig].copy()
    ne = e.groupby("rid").rid.transform("size")
    e = e[ne >= 4].copy()
    for s2 in [SIG2_MAIN] + SIG2_SENS:
        Dc = f"D_{s2}"
        g = e.groupby("rid")[Dc]
        e[f"z_{s2}"] = (e[Dc] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    zc = f"z_{SIG2_MAIN}"
    e = e[e[zc].notna()].copy()
    cuts = np.percentile(e[zc], [20, 40, 60, 80])
    results["zcuts_main"] = cuts.tolist()
    for pname, (a, b) in PERIODS.items():
        s = e[(e.ymd >= a) & (e.ymd <= b)]
        cov[pname]["analysis_runners"] = int(len(s))
        cov[pname]["analysis_races"] = int(s.rid.nunique())
        cov[pname]["analysis_share_of_runners"] = int(len(s)) / cov[pname]["runners"]
    print(f"analysis sample {len(e):,} runners ({time.time()-t0:.0f}s)", flush=True)

    # 最終オッズ
    e["pflb"] = add_flb(e, m, "p_fin", "tan_odds")
    results["final"] = {"main": quintile_tables(e, "p_fin", zc, cuts, "final", True)}
    for s2 in SIG2_SENS:
        zs = f"z_{s2}"
        cs = np.percentile(e[zs].dropna(), [20, 40, 60, 80])
        results["final"][f"sens_sig2_{s2}"] = quintile_tables(e[e[zs].notna()], "p_fin", zs, cs, "final", False)
    print(f"final done ({time.time()-t0:.0f}s)", flush=True)

    # 朝オッズ (朝オッズが全頭揃うレースのみ。比較のため同じレース集合で最終も再計算)
    em = e[e.p_mor.notna()].copy()
    em["odds_mor_eq"] = 1 / (em.p_mor)  # 帯分け用 (正規化後の逆数)
    mm = m[m.p_mor.notna()].copy()
    mm["odds_mor_eq"] = 1 / mm.p_mor
    em["pflb"] = add_flb(em, mm, "p_mor", "odds_mor_eq")
    results["morning"] = {"main": quintile_tables(em, "p_mor", zc, cuts, "morning", False)}
    em["pflb"] = add_flb(em, mm, "p_fin", "tan_odds")
    results["final_same_races_as_morning"] = {"main": quintile_tables(em, "p_fin", zc, cuts, "final", False)}
    print(f"morning done ({time.time()-t0:.0f}s)", flush=True)

    # 条件付きロジット: 1着 ~ log p + (log p)^2 + z
    cl = {}
    for snap, pcol, df in [("final", "p_fin", e), ("morning", "p_mor", em)]:
        cl[snap] = {}
        for pname, (a, b) in PERIODS.items():
            s = df[(df.ymd >= a) & (df.ymd <= b)].sort_values("rid").reset_index(drop=True)
            # 適格馬だけをチョイスセットにする (勝ち馬が含まれるレースのみ)
            win = (s.jyun == 1).astype(int)
            keep = (win.groupby(s.rid).transform("sum") == 1).to_numpy()
            s = s[keep].reset_index(drop=True)
            lp = np.log(s[pcol].to_numpy())
            X = np.c_[lp, lp ** 2, s[zc].to_numpy()]
            grp = pd.factorize(s.rid)[0]
            beta, se = clogit(X, (s.jyun == 1).astype(float).to_numpy(), grp, s.ymd.to_numpy())
            cl[snap][pname] = {"beta_z": float(beta[2]), "se_z": float(se[2]),
                               "ci_z": [float(beta[2] - 1.96 * se[2]), float(beta[2] + 1.96 * se[2])],
                               "beta_logp": float(beta[0]), "beta_logp2": float(beta[1]),
                               "n_races": int(grp.max() + 1)}
    results["clogit_z"] = cl
    print(f"clogit done ({time.time()-t0:.0f}s)", flush=True)

    # (f) 残差の条件付きロジット (事後追加)
    lagc = [f"h{k}" for k in range(5)]
    rc = {}
    for snap, pcol, df in [("final", "p_fin", e), ("morning", "p_mor", em)]:
        rc[snap] = {}
        for pname, (a, b) in PERIODS.items():
            s = df[(df.ymd >= a) & (df.ymd <= b) & (df.nh >= 5)]
            s = s[s.groupby("rid").rid.transform("size") >= 4]
            s = s.sort_values("rid").reset_index(drop=True)
            keep = ((s.jyun == 1).groupby(s.rid).transform("sum") == 1).to_numpy()
            s = s[keep].reset_index(drop=True)
            lp = np.log(s[pcol].to_numpy())
            L = (s[lagc] - s.groupby("rid")[lagc].transform("mean")).to_numpy()
            X = np.c_[lp, lp ** 2, L]
            grp = pd.factorize(s.rid)[0]
            beta, se = clogit(X, (s.jyun == 1).astype(float).to_numpy(), grp, s.ymd.to_numpy())
            rc[snap][pname] = {"beta_lags": beta[2:].tolist(), "se_lags": se[2:].tolist(),
                               "n_races": int(grp.max() + 1)}
    results["residual_clogit_lags_posthoc"] = rc
    print(f"residual clogit done ({time.time()-t0:.0f}s)", flush=True)

    # 暗黙の重み
    results["implied_weights_final"] = implied_weights(e, "p_fin", SIG2_MAIN, Q_MAIN)
    results["implied_weights_morning"] = implied_weights(em, "p_mor", SIG2_MAIN, Q_MAIN)
    print(f"weights done ({time.time()-t0:.0f}s)", flush=True)

    # 判定
    def verdict(block):
        r = [block[p]["contrast_ci"] for p in PERIODS]
        up = all(c[0] > 1 for c in r)
        dn = all(c[1] < 1 for c in r)
        return "SIGNAL(理論の向き)" if up else ("SIGNAL(逆向き)" if dn else "none")
    results["verdict"] = {
        "final_raw": verdict(results["final"]["main"]),
        "final_flb": verdict({p: {"contrast_ci": results["final"]["main"][p]["contrast_flb_ci"]} for p in PERIODS}),
        "morning_raw": verdict(results["morning"]["main"]),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(results["verdict"], ensure_ascii=False))
    print(f"wrote {OUT} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    sys.exit(main())
