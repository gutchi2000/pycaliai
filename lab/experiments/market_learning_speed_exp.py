# -*- coding: utf-8 -*-
"""市場の学習速度 — 市場は「自分の間違い」からどれだけ速く学ぶか (2026-11-14)

理論
----
パリミュチュエルの群衆も学習者である。各レースは「その馬 (厩舎・騎手・種牡馬・生産者・騎手×厩舎) に
ついてオッズがどれだけ外れたか」を 1 回ずつ見せる。群衆の更新が遅ければ、過去に「オッズより走った」
履歴を持つ主体は買われなさ過ぎのまま残り 実勝/市場 > 1 (市場誤差のモメンタム)。更新が行き過ぎなら
符号が逆 (リバーサル)。どちらでも両期で安定なら、定義上 市場の外にある情報。

事前登録 (確認期間の結果を見る前に固定。以後は不具合修正のみ)
------------------------------------------------------------
データ: JRDB SED 2011-01-01〜2025-12-31 の平地 (td∈{1,2}、障害は履歴からも除外)。
  出走馬 = 確定単勝オッズ >0 かつ ijou∉{1 取消, 2 除外}。頭数 n = レース内の出走馬数。
  市場勝率 p = (1/tan_odds) をレース内正規化。10 時市場 p10 = (1/odds10_tan) 同 (全出走馬に 10 時オッズが
  あるレースのみ)。中止・失格 (ijou∈{3,4}) は 1着=0, 3着内=0, 着順=n として数える。
  騎手・調教師・種牡馬・生産者は master_lite (2013-2025) を rid_s+馬番 で結合 (馬主列は使わない)。

走ごとのサプライズ (すべて確定オッズ基準。10 時市場の検定でも履歴は確定オッズ基準 = 公開されている記録):
  (a) s_win  = 1着 − p
  (b) s_top3 = 3着内 − q3。q3 = Harville の 3 着内確率 (p から) を 2011-12 の全走でロジスティック再較正
               (3着内 ~ a + b·L + c·L², L=logit(q3_Harville))。n≤3 のレースは s_top3 = 0。
  (c) s_rank = (人気順位 − 着順) / n

主体ごとの as-of 状態 (市場誤差の記憶): 指数減衰平均 + 0 への縮小。
  レース日 D の状態は 日付 < D の走だけで作る (同日の他レースは使わない)。主体の 1 開催日ぶん (n_d 走、
  サプライズ合計 s_d) をまとめて S ← λ^{n_d}·S + s_d, N ← λ^{n_d}·N + n_d, λ = 0.5^(1/半減期[走])。
  状態 = S / (N + k)。適格 = N ≥ N_min。
  変種 (主体あたり 3 本まで、計 18 本。全部報告する):
    馬 (ped, 履歴 2011〜)        : [top3, 半減期 3 走, k=2] [top3, 半減期 10 走, k=2] [rank, 半減期 3 走, k=2]   N_min=1
    調教師 (履歴 2013〜)          : [win] [top3] [rank]  半減期 200 走, k=50, N_min=30
    騎手                          : 同上
    種牡馬×芝ダ                   : 同上
    生産者                        : 同上
    騎手×調教師                   : [win] [top3] [rank]  半減期 50 走, k=20, N_min=10
  半減期・k・N_min は事前に固定した値で、データに合わせていない。

検定期間: 馬は 2013-19 / 2020-25。他の主体は master が 2013 開始なので 2013 年を助走にし 2014-19 / 2020-25。
検定母集団: 適格馬が 4 頭以上いて状態の分散が 0 でないレース。z = 適格馬内でのレース内標準化。
  五分位は検定期間全体をプールした z の 20/40/60/80 パーセンタイル。Q5 = 過去に市場を上回ってきた側。
主指標: 実勝/市場 = Σ1着 / Σp (五分位ごと)、コントラスト C = (Q5)/(Q1)。CI95 は開催日クラスタ bootstrap (B=2000)。
  本命-大穴補正版 C_adj: 期待勝 = Σ p·c(オッズ帯)、c は期・市場ごとに全出走馬で推定。
副指標: 条件付きロジット 1着 ~ log p + (log p)² + γ·z (不適格馬は z=0)、γ の開催日クラスタ頑健 SE。
  同じものを 10 時市場 (p10) でも出す。単勝 100 円均等買い回収率は文脈として五分位別に載せるだけ (探索しない)。

判定 (変種ごと、確定オッズ):
  「実在」= C と C_adj の CI95 が 2013(14)-19 と 2020-25 の両方で 1 を除外し同じ向き、かつ γ も両期で同符号かつ
  CI95 が 0 を除外。C>1 = モメンタム (市場の学習が遅い)、C<1 = リバーサル (行き過ぎ)。それ以外は none。
  10 時市場は同じ基準で別に判定 (「朝だけの現象か」を見る)。
  18 本を見るので、帰無仮説の下で 1 本が偶然通る確率は 約 18×2×0.025² ≈ 2%。
  主体の判定 = その主体の 3 変種のうち 1 本でも「実在」ならその向き (向きが割れたら「混在」)。
ゲート (gate_common.gate、valid=2023 で重み fit、test=2024-25 を 1 回): 18 変種を 1 本ずつ + 6 主体の top3 状態
  (馬は半減期 3) を同時に入れた 1 本。本丸は M0 (確定市場 → +z) と M (確定市場+v7 → +z)。CI 上端 < 0 で PASS。

市場の暗黙の学習率 (馬、記述統計・判定には使わない):
  全出走馬に前走 (平地、日付が前) があるレースで、前走のレース内中心化 log p (c_prev) と前走サプライズ s_prev から
    市場: log p_t ~ ρ·c_prev + κ_mkt·s_prev (レース固定効果つき OLS)
    結果: 1着_t ~ ρ*·c_prev + κ*·s_prev (条件付きロジット、今走の市場は使わない)
    残り: 1着_t ~ log p_t + (log p_t)² + γ·s_prev
  学習率 = κ_mkt/κ* (市場が「結果が示す最適な更新」の何割を済ませたか) と (κ/ρ) の比。s は top3 と rank、
  市場は確定と 10 時。CI は開催日クラスタ bootstrap (B=300)。κ* は市場を条件にしていないので、サプライズと
  相関する他の情報 (調教・状態) の代理分を含む = 上限の目安。

事後追加 (上の事前登録ぶんの結果を見た後に追加。探索扱いで判定には使わない。目的は交絡の点検と「学習の速さ」の記述)
  E1 馬 top3_h3 の γ は「前走着順」(1着/2-3着/4-5着/6-9着/10着以下/初出走) のダミーを入れても残るか
     (市場の癖の地図の「前走 2-3 着は買われ過ぎ」の言い換えにすぎないかの点検)。
  E2 オッズ帯 (確定 <5 / 5-9.9 / 10-29.9 / 30+ 倍) の中での C (馬 top3_h3、種牡馬×芝ダ win/top3)。
  E3 同時ロジット: log p + (log p)² + 6 主体の top3 状態 z (馬は半減期 3)。2014-19 / 2020-25、確定と 10 時。
     さらに 馬の win サプライズ状態 (半減期 10 走, k=2) を足しても種牡馬×芝ダの係数が残るか (同じ馬の持ち越しの点検)。
  E4 状態を 1 年前・2 年前の時点のものに差し替えた C と γ (種牡馬×芝ダ win/top3、調教師 win、生産者 win/rank)。
     古い状態でも同じだけ効くなら「市場がいつまでも直さない固定の癖」、効きが落ちるなら「遅れて学習している」。
  E5 馬の win サプライズ状態 (半減期 10 走, k=2) の C と γ (他主体と同じ物差しで見た馬の符号)。

再実行: worktree root から  python lab/experiments/market_learning_speed_exp.py   (約 5 分)
出力: reports/market_learning_speed_exp.json (集計値のみ。JRDB の馬別値は出さない)
      reports/mls_states.parquet (作業用。JRDB 由来の値を含むので reports/ の外へ出さない)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("NUMBA_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import pandas as pd
from numba import njit
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gate_common as gc  # noqa: E402

WT = HERE.parents[1]
OUT = WT / "reports" / "market_learning_speed_exp.json"
STATES_PQ = WT / "reports" / "mls_states.parquet"
B = 2000
B_LR = 300
SEED = 20261114
ODDS_BANDS = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0, 1e9]
PERIODS_HORSE = {"2013-19": (20130101, 20191231), "2020-25": (20200101, 20251231)}
PERIODS_ENT = {"2014-19": (20140101, 20191231), "2020-25": (20200101, 20251231)}

# (主体, キー列, 変種名, サプライズ, 半減期, k, N_min)
VARIANTS = [
    ("horse", "k_horse", "horse_top3_h3", "s_top3", 3, 2.0, 1.0),
    ("horse", "k_horse", "horse_top3_h10", "s_top3", 10, 2.0, 1.0),
    ("horse", "k_horse", "horse_rank_h3", "s_rank", 3, 2.0, 1.0),
]
for _e, _k, _hl, _kk, _nm in (("trainer", "k_trainer", 200, 50.0, 30.0), ("jockey", "k_jockey", 200, 50.0, 30.0),
                              ("sire_surface", "k_sire", 200, 50.0, 30.0), ("breeder", "k_breeder", 200, 50.0, 30.0),
                              ("jockey_trainer", "k_pair", 50, 20.0, 10.0)):
    for _s in ("win", "top3", "rank"):
        VARIANTS.append((_e, _k, f"{_e}_{_s}_h{_hl}", f"s_{_s}", _hl, _kk, _nm))
JOINT = ["horse_top3_h3", "trainer_top3_h200", "jockey_top3_h200", "sire_surface_top3_h200", "breeder_top3_h200",
         "jockey_trainer_top3_h50"]


def log(m):
    print(m, flush=True)


# ------------------------------------------------------------------ numba 部品
@njit(cache=True)
def harville_top3(p, start, end):
    out = np.ones(p.shape[0])
    for r in range(start.shape[0]):
        a, b = start[r], end[r]
        n = b - a
        if n <= 3:
            continue
        for i in range(a, b):
            pi = p[i]
            tot = pi
            for j in range(a, b):
                if j == i:
                    continue
                pj = p[j]
                dj = 1.0 - pj
                if dj <= 1e-12:
                    continue
                tot += pj * pi / dj
                for k in range(a, b):
                    if k == i or k == j:
                        continue
                    dk = dj - p[k]
                    if dk <= 1e-12:
                        continue
                    tot += pj * (p[k] / dj) * (pi / dk)
            out[i] = min(tot, 1.0)
    return out


@njit(cache=True)
def ewma_pre_state(key, n_d, s_d, lam):
    """行 = (主体, 開催日) を 主体→日付 の順に並べたもの。各行の「その日より前」の S, N を返す。"""
    m = key.shape[0]
    S_pre = np.zeros(m)
    N_pre = np.zeros(m)
    S = 0.0
    N = 0.0
    for i in range(m):
        if i == 0 or key[i] != key[i - 1]:
            S = 0.0
            N = 0.0
        S_pre[i] = S
        N_pre[i] = N
        d = lam ** n_d[i]
        S = d * S + s_d[i]
        N = d * N + n_d[i]
    return S_pre, N_pre


# ------------------------------------------------------------------ データ
def load_runs() -> pd.DataFrame:
    s = gc.load_sed(["racekey", "ymd", "ban", "ped", "td", "tosu", "jyun", "ijou", "tan_odds", "ninki", "odds10_tan"])
    s = s[s.td.isin([1, 2]) & (s.ymd <= 20251231)]
    s = s[(s.tan_odds > 0) & ~s.ijou.isin([1, 2])].copy()
    s = s.sort_values(["rid_s", "ban"]).reset_index(drop=True)
    g = s.groupby("rid_s")
    s["n"] = g["ban"].transform("size")
    inv = 1.0 / s.tan_odds
    s["p"] = inv / inv.groupby(s.rid_s).transform("sum")
    ok10 = (s.odds10_tan > 0).groupby(s.rid_s).transform("all")
    inv10 = (1.0 / s.odds10_tan).where(ok10)
    s["p10"] = inv10 / inv10.groupby(s.rid_s).transform("sum")
    dnf = s.ijou.isin([3, 4]) | (s.jyun <= 0)
    s["jy"] = np.where(dnf, s.n, s.jyun).astype(float)
    s["win"] = ((s.jy == 1) & ~dnf).astype(float)
    s["top3"] = ((s.jy <= 3) & ~dnf).astype(float)
    # Harville 3 着内 → 2011-12 で再較正
    codes = pd.factorize(s.rid_s)[0]
    start = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
    end = np.r_[start[1:], len(s)]
    q3h = harville_top3(s.p.values.astype(np.float64), start.astype(np.int64), end.astype(np.int64))
    L = np.log(np.clip(q3h, 1e-5, 1 - 1e-5) / (1 - np.clip(q3h, 1e-5, 1 - 1e-5)))
    fit = ((s.ymd <= 20121231) & (s.n >= 4)).values
    Xf = np.c_[np.ones(fit.sum()), L[fit], L[fit] ** 2]
    yf = s.top3.values[fit]

    def nll(w):
        e = Xf @ w
        return np.mean(np.logaddexp(0, e) - yf * e), Xf.T @ (1 / (1 + np.exp(-e)) - yf) / len(yf)
    w = minimize(nll, np.array([0.0, 1.0, 0.0]), jac=True, method="L-BFGS-B").x
    q3 = 1 / (1 + np.exp(-(w[0] + w[1] * L + w[2] * L ** 2)))
    q3 = np.where(s.n.values <= 3, s.top3.values, q3)
    s["q3"] = q3
    s["s_win"] = s.win - s.p
    s["s_top3"] = s.top3 - s.q3
    s["s_rank"] = (s.ninki - s.jy) / s.n
    # master_lite から主体キー
    l = gc.load_master_lite()[["レースID(新/馬番無)", "馬番", "騎手コード", "調教師コード", "種牡馬", "生産者"]]
    l = l.assign(rid_s=l["レースID(新/馬番無)"].astype(str), ban=l["馬番"].astype(int)).drop(columns=["レースID(新/馬番無)", "馬番"])
    l = l.drop_duplicates(["rid_s", "ban"])
    s = s.merge(l, on=["rid_s", "ban"], how="left")
    s["k_horse"] = pd.factorize(s.ped.astype(str))[0]

    def fz(x, mask):
        c = pd.factorize(x)[0].astype(np.int64)
        return np.where(mask.values & (c >= 0), c, -1)
    has = s["騎手コード"].notna() & s["調教師コード"].notna()
    s["k_trainer"] = fz(s["調教師コード"], s["調教師コード"].notna())
    s["k_jockey"] = fz(s["騎手コード"], s["騎手コード"].notna())
    s["k_sire"] = fz(s["種牡馬"].astype(str) + "|" + s.td.astype(str), s["種牡馬"].notna())
    s["k_breeder"] = fz(s["生産者"], s["生産者"].notna())
    s["k_pair"] = fz(s["騎手コード"].astype(str) + "|" + s["調教師コード"].astype(str), has)
    calib = {"harville_recal_w": [float(x) for x in w], "n_fit_2011_12": int(fit.sum()),
             "top3_actual_over_q3_2013_25": float(s.top3[s.ymd >= 20130101].sum() / s.q3[s.ymd >= 20130101].sum())}
    return s.drop(columns=["騎手コード", "調教師コード", "種牡馬", "生産者", "racekey"]), calib


def add_states(s: pd.DataFrame) -> pd.DataFrame:
    for ent, kcol, name, scol, hl, k, nmin in VARIANTS:
        d = s.loc[s[kcol] >= 0, [kcol, "ymd", scol]]
        a = d.groupby([kcol, "ymd"], sort=True)[scol].agg(["sum", "size"]).reset_index()
        S, N = ewma_pre_state(a[kcol].values.astype(np.int64), a["size"].values.astype(np.float64),
                              a["sum"].values.astype(np.float64), 0.5 ** (1.0 / hl))
        a[name] = np.where(N >= nmin, S / (N + k), np.nan)
        s = s.merge(a[[kcol, "ymd", name]], on=[kcol, "ymd"], how="left")
    return s


# ------------------------------------------------------------------ 検定部品
def race_z(df: pd.DataFrame, col: str) -> pd.Series:
    """適格馬 (非 NaN) 内でレース内標準化。適格 <4 頭 or sd=0 のレースは NaN。"""
    g = df.groupby("rid_s")[col]
    cnt = g.transform("count")
    mu = g.transform("mean")
    sd = g.transform("std")
    z = (df[col] - mu) / sd
    return z.where((cnt >= 4) & (sd > 1e-12))


def flb_factor(odds: np.ndarray, p: np.ndarray, win: np.ndarray) -> np.ndarray:
    band = np.digitize(odds, ODDS_BANDS[1:-1])
    A = np.bincount(band, weights=win, minlength=len(ODDS_BANDS) - 1)
    E = np.bincount(band, weights=p, minlength=len(ODDS_BANDS) - 1)
    c = np.where(E > 0, A / np.maximum(E, 1e-12), 1.0)
    return c[band]


def clogit(X, codes, win, dcodes, n_iter=30):
    """条件付きロジット (レース = 選択集合)。係数と開催日クラスタ頑健 SE。"""
    nr = codes.max() + 1
    k = X.shape[1]
    w = np.zeros(k)
    w[0] = 1.0
    nwin = np.bincount(codes, weights=win, minlength=nr)
    for _ in range(n_iter):
        lp = gc.race_logsoftmax(X @ w, codes, nr)
        pi = np.exp(lp) * nwin[codes]
        g = X.T @ (win - pi)
        M = np.stack([np.bincount(codes, weights=pi * X[:, j], minlength=nr) for j in range(k)], 1)
        H = (X * pi[:, None]).T @ X - (M / np.maximum(nwin, 1e-12)[:, None]).T @ M
        step = np.linalg.solve(H + 1e-9 * np.eye(k), g)
        w = w + step
        if np.abs(step).max() < 1e-9:
            break
    lp = gc.race_logsoftmax(X @ w, codes, nr)
    pi = np.exp(lp) * nwin[codes]
    sc = X * (win - pi)[:, None]
    nd = dcodes.max() + 1
    G = np.stack([np.bincount(dcodes, weights=sc[:, j], minlength=nd) for j in range(k)], 1)
    Hi = np.linalg.inv(H)
    V = Hi @ (G.T @ G) @ Hi * nd / max(nd - 1, 1)
    return w, np.sqrt(np.diag(V))


def boot_w(nd, b, rng):
    return rng.multinomial(nd, np.full(nd, 1.0 / nd), size=b).astype(np.float64)


def ae_block(d: pd.DataFrame, zcol: str, pcol: str, oddscol: str, cuts, rng) -> dict:
    """d: 1 期ぶんの全出走馬 (pcol 非 NaN のレースのみ)。zcol が非 NaN の馬を五分位に。"""
    win = d.win.values
    p = d[pcol].values
    c = flb_factor(d[oddscol].values, p, win)
    e = d[zcol].notna().values
    q = np.digitize(d[zcol].values[e], cuts)
    dc_all, ud = pd.factorize(d.ymd.values, sort=True)
    dc = dc_all[e]
    nd = len(ud)

    def tab(v):
        return np.bincount(dc * 5 + q, weights=v, minlength=nd * 5).reshape(nd, 5)
    A, E, Ea = tab(win[e]), tab(p[e]), tab((p * c)[e])
    Nq = tab(np.ones(e.sum()))
    R = tab((win * d.tan_odds.values)[e])
    W = boot_w(nd, B, rng)
    Ab, Eb, Eab, Nb, Rb = W @ A, W @ E, W @ Ea, W @ Nq, W @ R
    ae = A.sum(0) / E.sum(0)
    aeb = Ab / Eb
    aea = A.sum(0) / Ea.sum(0)
    aeab = Ab / Eab

    def ci(v):
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    out = {"n_horses": int(e.sum()), "n_races": int(d.rid_s[e].nunique()),
           "n_q": [int(x) for x in Nq.sum(0)], "wins_q": [int(x) for x in A.sum(0)],
           "ae_q": [float(x) for x in ae], "ae_q_ci": [ci(aeb[:, j]) for j in range(5)],
           "ae_adj_q": [float(x) for x in aea],
           "C": float(ae[4] / ae[0]), "C_ci": ci(aeb[:, 4] / aeb[:, 0]),
           "C_adj": float(aea[4] / aea[0]), "C_adj_ci": ci(aeab[:, 4] / aeab[:, 0]),
           "tan_roi_q": [float(100 * x) for x in R.sum(0) / Nq.sum(0)],
           "tan_roi_q_ci": [[100 * x for x in ci(Rb[:, j] / Nb[:, j])] for j in range(5)],
           "median_odds_q": [float(np.median(d[oddscol].values[e][q == j])) for j in range(5)]}
    # 条件付きロジット: 適格馬 ≥4 のレース全馬 (不適格は z=0)
    rr = d.rid_s.isin(d.rid_s[e].unique()).values
    x = d[rr]
    codes = pd.factorize(x.rid_s.values)[0]
    lp = np.log(x[pcol].values)
    X = np.c_[lp, lp ** 2, x[zcol].fillna(0.0).values]
    w, se = clogit(X, codes, x.win.values, pd.factorize(x.ymd.values)[0])
    out["logit_gamma"] = float(w[2])
    out["logit_gamma_ci"] = [float(w[2] - 1.96 * se[2]), float(w[2] + 1.96 * se[2])]
    out["logit_b_lp"] = float(w[0])
    out["logit_b_lp2"] = float(w[1])
    out["corr_z_logp"] = float(np.corrcoef(d[zcol].values[e], np.log(p[e]))[0, 1])
    return out


def sign_excl(ci, null):
    return 1 if ci[0] > null else (-1 if ci[1] < null else 0)


def verdict_two(pa: dict, pb: dict) -> str:
    sig = []
    for r in (pa, pb):
        a = sign_excl(r["C_ci"], 1.0)
        b_ = sign_excl(r["C_adj_ci"], 1.0)
        c = sign_excl(r["logit_gamma_ci"], 0.0)
        sig.append(a if (a != 0 and a == b_ == c) else 0)
    if sig[0] != 0 and sig[0] == sig[1]:
        return "momentum" if sig[0] > 0 else "reversal"
    return "none"


# ------------------------------------------------------------------ 学習率
def wclogit(X, codes, win, wr, n_iter=25):
    nr = codes.max() + 1
    k = X.shape[1]
    w = np.zeros(k)
    wi = wr[codes]
    nwin = np.bincount(codes, weights=win, minlength=nr)
    for _ in range(n_iter):
        lp = gc.race_logsoftmax(X @ w, codes, nr)
        pi = np.exp(lp) * nwin[codes]
        g = X.T @ (wi * (win - pi))
        M = np.stack([np.bincount(codes, weights=pi * X[:, j], minlength=nr) for j in range(k)], 1)
        H = (X * (wi * pi)[:, None]).T @ X - (M * (wr / np.maximum(nwin, 1e-12))[:, None]).T @ M
        step = np.linalg.solve(H + 1e-9 * np.eye(k), g)
        w = w + step
        if np.abs(step).max() < 1e-8:
            break
    return w


def learning_rate(s: pd.DataFrame, rng) -> dict:
    s = s.sort_values(["k_horse", "ymd"])
    same = s.k_horse.values[1:] == s.k_horse.values[:-1]
    lpc = np.log(s.p) - np.log(s.p).groupby(s.rid_s).transform("mean")
    for c, v in (("c_prev", lpc.values), ("st_prev", s.s_top3.values), ("sr_prev", s.s_rank.values)):
        x = np.full(len(s), np.nan)
        x[1:][same] = v[:-1][same]
        s[c] = x
    s = s.sort_values(["rid_s", "ban"])
    allprev = s.c_prev.notna().groupby(s.rid_s).transform("all") & (s.n >= 5)
    out = {}
    for pname, (a, b_) in PERIODS_HORSE.items():
        d = s[allprev & (s.ymd >= a) & (s.ymd <= b_)]
        res = {"n_races": int(d.rid_s.nunique()), "n_horses": int(len(d))}
        codes = pd.factorize(d.rid_s.values)[0]
        nr = codes.max() + 1
        dcode_h = pd.factorize(d.ymd.values, sort=True)[0]
        nd = dcode_h.max() + 1
        rdate = np.zeros(nr, dtype=np.int64)
        rdate[codes] = dcode_h
        W = boot_w(nd, B_LR, rng)
        for sname, scol in (("top3", "st_prev"), ("rank", "sr_prev")):
            Xr = np.c_[d.c_prev.values, d[scol].values]
            Xc = Xr - np.stack([np.bincount(codes, weights=Xr[:, j], minlength=nr) / np.bincount(codes, minlength=nr)
                                for j in range(2)], 1)[codes]
            r = {}
            # 結果 (市場なし)
            w_out = wclogit(Xc, codes, d.win.values, np.ones(nr))
            wb_out = np.array([wclogit(Xc, codes, d.win.values, W[i][rdate]) for i in range(B_LR)])
            r["outcome"] = {"rho": float(w_out[0]), "kappa": float(w_out[1]),
                            "kappa_ci": [float(np.percentile(wb_out[:, 1], 2.5)), float(np.percentile(wb_out[:, 1], 97.5))]}
            for mname, pcol in (("final", "p"), ("am10", "p10")):
                okm = d[pcol].notna().values
                y = np.log(d[pcol].values[okm])
                cm = pd.factorize(d.rid_s.values[okm])[0]
                nrm = cm.max() + 1
                cnt = np.bincount(cm, minlength=nrm)
                yc = y - (np.bincount(cm, weights=y, minlength=nrm) / cnt)[cm]
                Xm = Xr[okm]
                Xmc = Xm - np.stack([np.bincount(cm, weights=Xm[:, j], minlength=nrm) / cnt for j in range(2)], 1)[cm]
                dm = dcode_h[okm]
                # 開催日ごとの十分統計量
                XX = np.zeros((nd, 2, 2))
                Xy = np.zeros((nd, 2))
                for i in range(2):
                    Xy[:, i] = np.bincount(dm, weights=Xmc[:, i] * yc, minlength=nd)
                    for j in range(2):
                        XX[:, i, j] = np.bincount(dm, weights=Xmc[:, i] * Xmc[:, j], minlength=nd)
                bm = np.linalg.solve(XX.sum(0), Xy.sum(0))
                bb = np.array([np.linalg.solve(np.tensordot(W[i], XX, 1), W[i] @ Xy) for i in range(B_LR)])
                frac = bm[1] / w_out[1]
                fracb = bb[:, 1] / wb_out[:, 1]
                rr_ = (bm[1] / bm[0]) / (w_out[1] / w_out[0])
                rrb = (bb[:, 1] / bb[:, 0]) / (wb_out[:, 1] / wb_out[:, 0])
                # 残り (市場を条件に)
                x2 = d[okm]
                lp = np.log(x2[pcol].values)
                Xg = np.c_[lp, lp ** 2, Xmc[:, 1]]
                wg, seg = clogit(Xg, cm, x2.win.values, pd.factorize(x2.ymd.values)[0])
                r[mname] = {"rho_mkt": float(bm[0]), "kappa_mkt": float(bm[1]),
                            "kappa_mkt_ci": [float(np.percentile(bb[:, 1], 2.5)), float(np.percentile(bb[:, 1], 97.5))],
                            "learning_rate_kappa": float(frac),
                            "learning_rate_kappa_ci": [float(np.percentile(fracb, 2.5)), float(np.percentile(fracb, 97.5))],
                            "learning_rate_ratio": float(rr_),
                            "learning_rate_ratio_ci": [float(np.percentile(rrb, 2.5)), float(np.percentile(rrb, 97.5))],
                            "residual_gamma": float(wg[2]),
                            "residual_gamma_ci": [float(wg[2] - 1.96 * seg[2]), float(wg[2] + 1.96 * seg[2])]}
            res[sname] = r
        out[pname] = res
    return out


# ------------------------------------------------------------------ 事後追加 (探索)
def _state_table(s, kcol, scol, hl, k, nmin):
    d = s.loc[s[kcol] >= 0, [kcol, "ymd", scol]]
    a = d.groupby([kcol, "ymd"], sort=True)[scol].agg(["sum", "size"]).reset_index()
    S, N = ewma_pre_state(a[kcol].values.astype(np.int64), a["size"].values.astype(np.float64),
                          a["sum"].values.astype(np.float64), 0.5 ** (1.0 / hl))
    a["state"] = np.where(N >= nmin, S / (N + k), np.nan)
    return a[[kcol, "ymd", "state"]]


def _brief(x):
    return {k: x[k] for k in ("n_horses", "ae_q", "C", "C_ci", "C_adj", "C_adj_ci", "logit_gamma", "logit_gamma_ci")}


def posthoc(s_all: pd.DataFrame, rng) -> dict:
    out = {}
    vmap = {v[2]: v for v in VARIANTS}
    s = s_all.sort_values(["k_horse", "ymd"])
    same = s.k_horse.values[1:] == s.k_horse.values[:-1]
    pj = np.full(len(s), np.nan)
    pj[1:][same] = s.jy.values[:-1][same]
    s["prev_jy"] = pj
    a = _state_table(s, "k_horse", "s_win", 10, 2.0, 1.0).rename(columns={"state": "horse_win_h10"})
    s = s.merge(a, on=["k_horse", "ymd"], how="left").sort_values(["rid_s", "ban"]).reset_index(drop=True)
    s = s[s.ymd >= 20130101].copy()

    def logit_multi(d, pcol, cols):
        lp = np.log(d[pcol].values)
        X = np.c_[lp, lp ** 2, d[cols].fillna(0.0).values]
        w, se = clogit(X, pd.factorize(d.rid_s.values)[0], d.win.values, pd.factorize(d.ymd.values)[0])
        return {c: {"coef": float(w[2 + i]), "ci": [float(w[2 + i] - 1.96 * se[2 + i]), float(w[2 + i] + 1.96 * se[2 + i])]}
                for i, c in enumerate(cols)}

    # E1
    s["z_h"] = race_z(s, "horse_top3_h3")
    pj = s.prev_jy
    s["d_1"] = (pj == 1).astype(float)
    s["d_23"] = pj.between(2, 3).astype(float)
    s["d_69"] = pj.between(6, 9).astype(float)
    s["d_10p"] = (pj >= 10).astype(float)
    s["d_none"] = pj.isna().astype(float)
    e1 = {}
    dcols = ["d_1", "d_23", "d_69", "d_10p", "d_none"]
    for pname, (a_, b_) in PERIODS_HORSE.items():
        d = s[(s.ymd >= a_) & (s.ymd <= b_)]
        d = d[d.rid_s.isin(d.rid_s[d.z_h.notna()].unique())]
        e1[pname] = {"final": logit_multi(d, "p", ["z_h"] + dcols),
                     "am10": logit_multi(d[d.p10.notna()], "p10", ["z_h"] + dcols),
                     "final_dummies_only": logit_multi(d, "p", dcols)}
    out["E1_prev_finish_control"] = e1
    # E2
    e2 = {}
    bands = {"<5": (0, 5), "5-9.9": (5, 10), "10-29.9": (10, 30), "30+": (30, 1e9)}
    for name in ("horse_top3_h3", "sire_surface_win_h200", "sire_surface_top3_h200"):
        periods = PERIODS_HORSE if name.startswith("horse") else PERIODS_ENT
        lo = min(a_ for a_, _ in periods.values())
        t = s[s.ymd >= lo].copy()
        t["z"] = race_z(t, name)
        cuts = np.nanpercentile(t.z.values, [20, 40, 60, 80])
        e2[name] = {}
        for pname, (a_, b_) in periods.items():
            d = t[(t.ymd >= a_) & (t.ymd <= b_) & t.z.notna()]
            q = np.digitize(d.z.values, cuts)
            dc, ud = pd.factorize(d.ymd.values, sort=True)
            W = boot_w(len(ud), B, rng)
            r = {}
            for bn, (l_, h_) in bands.items():
                m = ((d.tan_odds.values >= l_) & (d.tan_odds.values < h_))
                st = []
                for qq in (0, 4):
                    mm = m & (q == qq)
                    st.append((np.bincount(dc[mm], weights=d.win.values[mm], minlength=len(ud)),
                               np.bincount(dc[mm], weights=d.p.values[mm], minlength=len(ud))))
                ae1 = st[0][0].sum() / st[0][1].sum()
                ae5 = st[1][0].sum() / st[1][1].sum()
                cb = ((W @ st[1][0]) / (W @ st[1][1])) / ((W @ st[0][0]) / (W @ st[0][1]))
                r[bn] = {"n_q1": int((m & (q == 0)).sum()), "n_q5": int((m & (q == 4)).sum()), "ae_q1": float(ae1), "ae_q5": float(ae5),
                         "C": float(ae5 / ae1), "C_ci": [float(np.percentile(cb, 2.5)), float(np.percentile(cb, 97.5))]}
            e2[name][pname] = r
    out["E2_within_odds_band"] = e2
    # E3
    zc = []
    for name in JOINT + ["horse_win_h10"]:
        s["zj_" + name] = race_z(s, name)
        zc.append("zj_" + name)
    e3 = {}
    for pname, (a_, b_) in PERIODS_ENT.items():
        d = s[(s.ymd >= a_) & (s.ymd <= b_)]
        e3[pname] = {"final": logit_multi(d, "p", zc[:-1]), "am10": logit_multi(d[d.p10.notna()], "p10", zc[:-1]),
                     "final_plus_horse_win": logit_multi(d, "p", zc)}
    out["E3_joint_logit"] = e3
    # E4
    e4 = {}
    s["date"] = pd.to_datetime(s.ymd.astype(str), format="%Y%m%d")
    for name in ("sire_surface_win_h200", "sire_surface_top3_h200", "trainer_win_h200", "breeder_win_h200", "breeder_rank_h200"):
        ent, kcol, _, scol, hl, k, nmin = vmap[name]
        a = _state_table(s_all, kcol, scol, hl, k, nmin)
        a["date"] = pd.to_datetime(a.ymd.astype(str), format="%Y%m%d")
        a = a.sort_values("date")
        e4[name] = {}
        for lag in (365, 730):
            t = s.loc[(s.ymd >= 20160101) & (s[kcol] >= 0), ["rid_s", "ban", "date", kcol]].copy()
            t["dlag"] = t.date - pd.Timedelta(days=lag)
            t = t.sort_values("dlag")
            mm = pd.merge_asof(t[["rid_s", "ban", "dlag", kcol]], a[[kcol, "date", "state"]], left_on="dlag", right_on="date",
                               by=kcol, direction="backward")
            u = s[s.ymd >= 20160101].merge(mm[["rid_s", "ban", "state"]], on=["rid_s", "ban"], how="left")
            u["z"] = race_z(u, "state")
            u["z0"] = race_z(u, name)
            r = {}
            for zname, zcol in (("lagged", "z"), ("current_same_period", "z0")):
                cuts = np.nanpercentile(u[zcol].values, [20, 40, 60, 80])
                r[zname] = {pn: _brief(ae_block(u[(u.ymd >= a_) & (u.ymd <= b_)], zcol, "p", "tan_odds", cuts, rng))
                            for pn, (a_, b_) in {"2016-19": (20160101, 20191231), "2020-25": (20200101, 20251231)}.items()}
            e4[name][f"lag{lag}d"] = r
    out["E4_lagged_state"] = e4
    # E5
    t = s.copy()
    t["z"] = race_z(t, "horse_win_h10")
    cuts = np.nanpercentile(t.z.values, [20, 40, 60, 80])
    e5 = {"final": {}, "am10": {}}
    for pname, (a_, b_) in PERIODS_HORSE.items():
        d = t[(t.ymd >= a_) & (t.ymd <= b_)]
        e5["final"][pname] = _brief(ae_block(d, "z", "p", "tan_odds", cuts, rng))
        e5["am10"][pname] = _brief(ae_block(d[d.p10.notna()], "z", "p10", "odds10_tan", cuts, rng))
    out["E5_horse_win_h10"] = e5
    return out


# ------------------------------------------------------------------ main
def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    s, calib = load_runs()
    log(f"[runs] {len(s):,} 走 / {s.rid_s.nunique():,} R  ({time.time() - t0:.0f}s)  較正 {calib}")
    s = add_states(s)
    log(f"[states] {len(VARIANTS)} 変種 ({time.time() - t0:.0f}s)")
    s[["rid_s", "ban", "ymd"] + [v[2] for v in VARIANTS]].to_parquet(STATES_PQ)
    res = {"generated": time.strftime("%Y-%m-%d %H:%M"), "calibration": calib, "B": B,
           "variants": {}, "entity_verdict": {}, "gates": {}}
    for ent, kcol, name, scol, hl, k, nmin in VARIANTS:
        periods = PERIODS_HORSE if ent == "horse" else PERIODS_ENT
        lo = min(a for a, _ in periods.values())
        t = s[s.ymd >= lo].copy()
        t["z"] = race_z(t, name)
        cuts = np.nanpercentile(t.z.values, [20, 40, 60, 80])
        v = {"entity": ent, "surprise": scol, "half_life_runs": hl, "k": k, "n_min": nmin,
             "coverage_eligible": float(t[name].notna().mean()), "coverage_tested": float(t.z.notna().mean()),
             "final": {}, "am10": {}}
        for pname, (a, b_) in periods.items():
            d = t[(t.ymd >= a) & (t.ymd <= b_)]
            v["final"][pname] = ae_block(d, "z", "p", "tan_odds", cuts, rng)
            d10 = d[d.p10.notna()]
            v["am10"][pname] = ae_block(d10, "z", "p10", "odds10_tan", cuts, rng)
        pn = list(periods)
        v["verdict_final"] = verdict_two(v["final"][pn[0]], v["final"][pn[1]])
        v["verdict_am10"] = verdict_two(v["am10"][pn[0]], v["am10"][pn[1]])
        res["variants"][name] = v
        f = v["final"]
        m = v["am10"]
        log(f"[{name:28s}] 確定 C {f[pn[0]]['C']:.3f} [{f[pn[0]]['C_ci'][0]:.3f},{f[pn[0]]['C_ci'][1]:.3f}] / "
            f"{f[pn[1]]['C']:.3f} [{f[pn[1]]['C_ci'][0]:.3f},{f[pn[1]]['C_ci'][1]:.3f}] γ {f[pn[0]]['logit_gamma']:+.3f}/{f[pn[1]]['logit_gamma']:+.3f}"
            f" → {v['verdict_final']} | 朝 C {m[pn[0]]['C']:.3f} [{m[pn[0]]['C_ci'][0]:.3f},{m[pn[0]]['C_ci'][1]:.3f}] / "
            f"{m[pn[1]]['C']:.3f} [{m[pn[1]]['C_ci'][0]:.3f},{m[pn[1]]['C_ci'][1]:.3f}] γ {m[pn[0]]['logit_gamma']:+.3f}/{m[pn[1]]['logit_gamma']:+.3f}"
            f" → {v['verdict_am10']}  ({time.time() - t0:.0f}s)")
    for ent in dict.fromkeys(v[0] for v in VARIANTS):
        for key in ("verdict_final", "verdict_am10"):
            vs = {x[key] for x in res["variants"].values() if x["entity"] == ent} - {"none"}
            res["entity_verdict"].setdefault(ent, {})[key] = "none" if not vs else (vs.pop() if len(vs) == 1 else "mixed")
    log(f"[entity] {res['entity_verdict']}")
    res["learning_rate_horse"] = learning_rate(s, rng)
    log(f"[learning rate] done ({time.time() - t0:.0f}s)")
    for pn_, r in res["learning_rate_horse"].items():
        for sn in ("top3", "rank"):
            for mn in ("final", "am10"):
                x = r[sn][mn]
                log(f"  {pn_} {sn:4s} {mn:5s} κ_mkt {x['kappa_mkt']:+.3f} κ* {r[sn]['outcome']['kappa']:+.3f} 学習率 {x['learning_rate_kappa']:.3f} "
                    f"[{x['learning_rate_kappa_ci'][0]:.3f},{x['learning_rate_kappa_ci'][1]:.3f}] 残り γ {x['residual_gamma']:+.3f} "
                    f"[{x['residual_gamma_ci'][0]:+.3f},{x['residual_gamma_ci'][1]:+.3f}]")
    # ゲート
    zdf = s.loc[s.ymd >= 20230101, ["rid_s", "ban"] + [v[2] for v in VARIANTS]]
    for v in VARIANTS:
        res["gates"][v[2]] = gc.gate(zdf, [v[2]], name=v[2])
    res["gates"]["joint_top3_6entities"] = gc.gate(zdf, JOINT, name="joint_top3_6entities")
    res["posthoc_exploratory"] = posthoc(s, np.random.default_rng(SEED + 1))
    log(f"[posthoc] done ({time.time() - t0:.0f}s)")
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"[done] {OUT} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    main()
