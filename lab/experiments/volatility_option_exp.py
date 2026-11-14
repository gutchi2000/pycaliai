# -*- coding: utf-8 -*-
"""
volatility_option_exp.py — ムラ馬のオプション価値 (分散は期待値とは別の情報か) (2026-11-14)
=====================================================================================
理論
----
1 走の出来 = 能力 + 運。運の大きさ (sd) は平均で IDM 約 6.4 点だが、馬ごとに違うはず (毎回走る馬 / ムラ馬)。
正規の運 (Thurstone) では、勝率は分散にも依存する:
  - 平均が下の馬 (人気薄) は、分散が大きいほど勝率が上がる (オプション価値)
  - 平均が上の馬 (本命) は、分散が大きいほど勝率が下がる。3 着内はさらに強く下がる
1 頭 1 スコアの softmax / Plackett-Luce (v7 の確率層) は全馬同じ運の形を仮定するのでこれを表せない。
馬券を買う人たちも値付けを誤っているかもしれない (ムラな人気薄を夢で買いすぎる / 堅実な本命の複勝を買い足りない)。

事前登録 (test 2024-25 を見る前に固定。ここに書いた定義・規則は結果を見てから変えない)
-----------------------------------------------------------------------------------
[馬ごとの分散 σ_h (as-of)]
  - 材料: JRDB SED の IDM。完走 (ijou==0) かつ IDM あり、芝ダ別 (td 1/2)。障害は使わない。
  - 同じ馬・同じ芝ダの連続 2 走の差 d_k = IDM_k − IDM_{k−1} (間隔 1〜365 日のみ。それ以外の差は捨てる)。
    差を取るので、ゆっくりした能力の変化 (成長・衰え) は自動で消える (軽い detrend)。
  - x_k = min(|d_k|, 30)² / 2   (30 点で頭打ち。大敗で流した走りが分散を支配しないように)
  - レース日 D の馬の σ_h² = (k·s0²_td + Σ_{日付<D} x) / (k + n)    n = D より前の差の本数
    これは逆ガンマ事前分布の事後平均と同じ形 (Bühlmann の信頼度)。s0²_td = 〜2022 の x の平均 (芝ダ別)、
    k は「次の x を二乗誤差最小で当てる」値を 〜2022 だけで求める (スカラー 1 個)。
  - n = 0 の馬は σ_h = s0 (母集団の平均)。
  - 特性かどうかの確認 (〜2022、差が 8 本以上の馬): 前半 vs 後半 (境の 1 本は捨てる) の log s の相関、
    重ならない差 (1 本おき) を交互に分けた相関、馬をまたいで x を入れ替えた帰無。前半 vs 後半の相関 < 0.05 なら理論はここで死亡と書く。

[z の定義 3 つ (これ以外は試さない。3 つとも報告する)]
  A  za = log σ_h
  B  zb = (log σ_h − レース平均) × u,  u = −(lm7 − レース平均 lm7)   (u は人気薄で正。Thurstone の一次近似)
  C  zc = log P_het − log P_hom   (どちらも Thurstone を Gauss–Hermite 40 点で厳密計算)
       平均 m_h = D より前の同じ芝ダの IDM 直近 5 走以内の平均 (無い馬はレース内の平均で埋める。m_h のある馬が 6 割未満のレースは zc = 欠損)
       P_hom: 全馬共通の sd s_c。s_c は 2016-2022 のレース 6,000 本 (乱数 seed 0) で勝ち馬の尤度を最大にする値 (スカラー 1 個)
       P_het: sd_i = sqrt(max(s_c² − s0²_td, 0) + σ_h,i²)   (追加の fit なし)
       確率は 1e-4 で下を切ってから log、zc は [−2, 2] に切る。
  ゲートは gate_common.gate を各 z 単独で 1 回ずつ (valid=2023 で重み、test=2024-25 を 1 回)。

[市場の偏りの検定 (2013-19 と 2020-25 を別々に)]
  - 市場確率 p = (1/確定単勝オッズ) をレース内で正規化 (SED tan_odds、全頭)。5 頭以上・勝ち馬 1 頭の平地戦。
  - オッズ帯: 本命 ≤2.9 / 中 3.0〜9.9 / 人気薄 10+。σ_h の三分位は「n ≥ 3 の馬」だけを (期間 × オッズ帯) の中で切る (結果は使わない)。
  - 期待値は「同じ期間・同じ市場確率 (40 分位)・同じ頭数帯」の全馬の平均で置く (本命-大穴バイアスと帯内のオッズ差を除く)。
      単勝: 実勝利 / 期待勝利 (AE_cal)、均等買い回収率とその残差 (同じ細かいオッズ帯の平均との差)
      複勝: 8 頭以上のレースで 実 3 着内 / 期待 3 着内 (AE3_cal)、複勝均等買い回収率とその残差 (払戻は data/kekka の 複勝配当)
  - 主統計 = 各オッズ帯の「σ 高 − σ 低」の差 4 つ (AE_cal, 単勝残差, AE3_cal, 複勝残差) × 3 帯 = 12 個。
    開催日クラスタ bootstrap 2,000 回の CI95。
  - 回帰版 (同じ za, zb, zc を 13 年分で): 条件付きロジット 勝ち ~ log p + z、
    3 着内は ロジスティック top3 ~ (log p の 3 次式, log 頭数, 交互作用) + z。係数と開催日クラスタ頑健 SE。
    ここでの u は市場の log p から作る (v7 は 2023 以前に無いため)。
  - 判定: 2013-19 と 2020-25 の両方で符号が同じ かつ 両方で CI95 が帰無 (0) を含まないものだけ「実在」。

[3 着内の問い]
  valid=2023 で fit、test=2024-25 で 1 回: top3 ~ (lm7 の 3 次式, log 頭数, 交互作用) に za, zb を足して対数損失が下がるか。
  市場込み版 (lk の 3 次式 + lm7) も同じ。開催日クラスタ bootstrap。8 頭以上のレース。

[判定規則]
  SIGNAL(予測)   : P ゲートが PASS (CI95 上端 < 0)
  SIGNAL(市場外) : M ゲートが PASS
  PASS が出たら 2024 / 2025 の年別符号と、偽薬 (σ_h をレース内で入れ替えた z) が none になることを確認。
  3 変種のうち 1 つだけの PASS で年別符号が割れるなら「要再確認」止まり。
  どれも PASS しなければ null (再走禁止の範囲を doc に書く)。

事後の追加 (2026-11-14、test の結果を見た後に足した。判定には使わない・test 2024-25 は使わない)
------------------------------------------------------------------------------------------
A が 4 ゲートとも PASS したので、「それは本当に分散という特性か、別のものの代理か」を 2013-19 / 2020-23 の
市場条件付きロジットだけで切り分ける (exploratory_posthoc_2013_23)。
  E0 za そのまま / E1 市場を 3 次式で制御 / E2 直近の差 1 本を除いた σ (za_old) と直近の差の上下 (last_up, last_down)
  E4 過去の差の本数 (n==0, log(1+n)) を制御 / E5 n ≥ 3 の馬だけ za を持たせる (他はレース平均)
それと as-of の総当たり自己点検 (300 行を「日付 < D の走だけ」で作り直して一致を確認)。

as-of
-----
SED の値は走った後の事実。レース日 D の特徴は日付 < D の走だけで作る (馬は 1 日 1 走なので、馬×芝ダの時系列で自分より前の累積)。
事前分布 (s0², k) と s_c は 〜2022 だけで推定。2013-22 の市場検定ではこの 4 つの定数だけが同期間由来 (個々のレース結果は使っていない)。

JRDB: IDM など 1 頭ごとの値は出力しない。JSON / doc は集計値だけ。
実行: PYTHONUTF8=1 E:/PyCaLiAI/venv311/Scripts/python.exe lab/experiments/volatility_option_exp.py
出力: reports/volatility_option_exp.json, reports/vol_features.parquet (作業用、git 管理外)
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, log_ndtr
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate_common as gc  # noqa: E402

WT = Path(__file__).resolve().parents[2]
OUT = WT / "reports" / "volatility_option_exp.json"
FEAT_PQ = WT / "reports" / "vol_features.parquet"
KEKKA = Path("E:/PyCaLiAI/data/kekka_20130105-20251228.csv")
CUT = pd.Timestamp("2022-12-31")
CLIP = 30.0
MIN_N = 3                  # 三分位に入れる最小の差の本数 (= 過去 4 走以上)
N_BOOT = 2000
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(40)
GH_W = GH_W / np.sqrt(2 * np.pi)
KMAX = 18
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:5.0f}s] {m}", flush=True)


# ------------------------------------------------------------------ 1. as-of 特徴
def load_runs() -> pd.DataFrame:
    d = gc.load_sed(["racekey", "ymd", "ban", "ped", "td", "jyun", "ijou", "tan_odds", "idm"])
    d = d[d.td.isin([1, 2])].copy()
    d["hk"] = pd.factorize(d["ped"].astype(str) + "_" + d["td"].astype(str))[0]
    d = d.sort_values(["hk", "date", "rid_s"]).reset_index(drop=True)
    ok = (d.ijou == 0) & d.idm.notna()
    v = d[ok]
    gv = v.groupby("hk", sort=False)
    dy = v.idm - gv.idm.shift(1)
    gap = (v.date - gv.date.shift(1)).dt.days
    dy = dy.where((gap >= 1) & (gap <= 365))
    d["x"] = np.nan
    d.loc[v.index, "x"] = np.minimum(dy.abs(), CLIP) ** 2 / 2.0
    xf = d.x.fillna(0.0); nf = d.x.notna().astype(int)
    d["S"] = xf.groupby(d.hk).cumsum() - xf            # 自分より前だけ
    d["n"] = nf.groupby(d.hk).cumsum() - nf
    d["dsg"] = np.nan
    d.loc[v.index, "dsg"] = dy.clip(-CLIP, CLIP)
    for src, dst in (("x", "xl"), ("dsg", "dl")):            # 自分より前で最後の差 (事後の切り分け用)
        d[dst] = d.groupby("hk", sort=False)[src].ffill()
        d[dst] = d.groupby("hk", sort=False)[dst].shift(1)
    r5 = gv.idm.rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    d["r5i"] = np.nan
    d.loc[r5.index, "r5i"] = r5
    d["r5i"] = d.groupby("hk", sort=False).r5i.ffill()
    d["m5"] = d.groupby("hk", sort=False).r5i.shift(1)   # D より前の直近 5 走以内の平均
    return d.drop(columns=["r5i"])


def split_corr(t, a, b):
    A = t[a].groupby("hk").x.mean(); B = t[b].groupby("hk").x.mean()
    j = pd.concat([A, B], axis=1, keys=["a", "b"]).dropna()
    j = j[(j.a > 0) & (j.b > 0)]
    la, lb = 0.5 * np.log(j.a), 0.5 * np.log(j.b)
    r = float(np.corrcoef(la, lb)[0, 1])
    return {"n_horses": int(len(j)), "pearson_log_s": r, "spearman": float(spearmanr(j.a, j.b)[0]),
            "spearman_brown": float(2 * r / (1 + r))}


def reliability(d) -> dict:
    t = d[(d.date <= CUT) & d.x.notna()][["hk", "td", "x"]].copy()
    t["pos"] = t.groupby("hk").cumcount(); t["nn"] = t.groupby("hk").x.transform("size")
    out = {"n_diffs_le2022": int(len(t)), "n_horse_surface_le2022": int(t.hk.nunique())}
    t = t[t.nn >= 8].copy()
    h = (t.nn - 1) // 2
    masks = {"early_vs_late": (t.pos < h, t.pos > h),
             "nonoverlap_alternate": ((t.pos % 4 == 0), (t.pos % 4 == 2))}
    for k, (a, b) in masks.items():
        out[k] = split_corr(t, a, b)
    for td, nm in ((1, "turf"), (2, "dirt")):      # 芝ダの差 (ダートの方が運が大きい) を除いた、同じ芝ダの中だけの相関
        tt = t[t.td == td]; hh = (tt.nn - 1) // 2
        out[f"early_vs_late_{nm}_only"] = split_corr(tt, tt.pos < hh, tt.pos > hh)
    rng = np.random.default_rng(0)
    perm = []
    for _ in range(200):
        t2 = t.copy()
        for td in (1, 2):
            ix = (t2.td == td).values
            t2.loc[ix, "x"] = rng.permutation(t2.loc[ix, "x"].values)
        perm.append(split_corr(t2, *masks["early_vs_late"])["pearson_log_s"])
    out["perm_null_early_vs_late"] = {"mean": float(np.mean(perm)), "p99.5": float(np.percentile(perm, 99.5)), "max": float(np.max(perm))}
    return out


def fit_prior(d) -> dict:
    t = d[(d.date <= CUT) & d.x.notna()]
    s0 = t.groupby("td").x.mean().to_dict()
    s0v = t.td.map(s0).values; S = t.S.values; n = t.n.values; x = t.x.values

    def mse(k):
        return float(((x - (k * s0v + S) / (k + n)) ** 2).mean())
    k = float(minimize_scalar(mse, bounds=(0.3, 500), method="bounded").x)
    within = float(t.groupby("hk").x.var().mul(t.groupby("hk").x.size() - 1).sum() / (len(t) - t.hk.nunique()))
    s0_all = float(t.x.mean())
    sd_sig2 = float(np.sqrt(within / k))
    return {"s0_sq": {int(a): float(b) for a, b in s0.items()}, "s0_sd": {int(a): float(np.sqrt(b)) for a, b in s0.items()},
            "k": k, "mse_k": mse(k), "mse_no_trait(k=inf)": float(((x - s0v) ** 2).mean()),
            "within_var_x": within, "implied_sd_of_true_sigma2": sd_sig2,
            "implied_cv_of_true_sigma(approx)": float(sd_sig2 / s0_all / 2)}


def calib_deciles(d, mask) -> dict:
    t = d[mask & d.x.notna() & (d.n >= MIN_N)]
    q = pd.qcut(t.sig2, 10, labels=False, duplicates="drop")
    g = t.groupby(q).agg(pred=("sig2", "mean"), real=("x", "mean"), n=("x", "size"))
    return {"n": int(len(t)), "spearman_pred_vs_next_x": float(spearmanr(t.sig2, t.x)[0]),
            "decile_pred_sd": [float(np.sqrt(v)) for v in g.pred], "decile_realised_sd": [float(np.sqrt(v)) for v in g.real]}


# ------------------------------------------------------------------ 2. Thurstone (不等分散)
def thurstone_het(m, sg):
    """m, sg: (S,K)。P_i = ∫φ(z) Π_{j≠i} Φ((m_i + σ_i z − m_j)/σ_j) dz"""
    a = m[:, :, None, None] + sg[:, :, None, None] * GH_X            # (S,K,1,G)
    L = log_ndtr((a - m[:, None, :, None]) / sg[:, None, :, None])    # (S,K,K,G)
    K = m.shape[1]
    L[:, np.arange(K), np.arange(K), :] = 0.0
    p = (np.exp(L.sum(2)) * GH_W).sum(-1)
    return p / p.sum(1, keepdims=True)


def pad(f, col, fill):
    R = int(f.rc.max()) + 1
    A = np.full((R, KMAX), fill, dtype=float)
    A[f.rc.values, f.pos.values] = f[col].values
    return A


def thurstone_frame(f, sd_col, chunk=300):
    """f: rc (レース番号), pos (レース内の位置), mfill, sd_col → 各行の勝率"""
    M = pad(f, "mfill", -1e6); SG = pad(f, sd_col, 1.0)
    P = np.empty_like(M)
    for i in range(0, len(M), chunk):
        P[i:i + chunk] = thurstone_het(M[i:i + chunk], SG[i:i + chunk])
    return P[f.rc.values, f.pos.values]


# ------------------------------------------------------------------ 3. 回帰 (クラスタ頑健)
def clogit_cluster(X, codes, n, win_idx, date_codes_race):
    w = gc._fit(X, codes, n, win_idx, 0.0)
    p = np.exp(gc.race_logsoftmax(X @ w, codes, n))
    k = X.shape[1]
    xbar = np.column_stack([np.bincount(codes, weights=p * X[:, j], minlength=n) for j in range(k)])
    g = np.zeros((n, k)); g[codes[win_idx]] = X[win_idx]
    g -= xbar
    H = (X * p[:, None]).T @ X - xbar.T @ xbar
    nd = date_codes_race.max() + 1
    G = np.column_stack([np.bincount(date_codes_race, weights=g[:, j], minlength=nd) for j in range(k)])
    Hi = np.linalg.inv(H)
    cov = Hi @ (G.T @ G) @ Hi
    return w, np.sqrt(np.diag(cov))


def logit_fit(X, y, l2=1e-6):
    w = np.zeros(X.shape[1])
    for _ in range(60):
        p = expit(X @ w)
        g = X.T @ (y - p) - l2 * w
        H = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(X.shape[1])
        step = np.linalg.solve(H, g)
        w += step
        if np.abs(step).max() < 1e-9:
            break
    return w, H


def logit_cluster(X, y, cl):
    w, H = logit_fit(X, y)
    s = X * (y - expit(X @ w))[:, None]
    nd = cl.max() + 1
    G = np.column_stack([np.bincount(cl, weights=s[:, j], minlength=nd) for j in range(X.shape[1])])
    Hi = np.linalg.inv(H)
    return w, np.sqrt(np.diag(Hi @ (G.T @ G) @ Hi))


def coef_row(w, se, j):
    return {"coef": float(w[j]), "se": float(se[j]), "z": float(w[j] / se[j]),
            "ci95": [float(w[j] - 1.96 * se[j]), float(w[j] + 1.96 * se[j])]}


def rc_center(v, codes):
    s = pd.Series(v)
    return (s - s.groupby(codes).transform("mean")).values


def top3_poly(lp, K):
    lk = np.log(K)
    return np.column_stack([np.ones(len(lp)), lp, lp ** 2, lp ** 3, lk, lp * lk])


# ------------------------------------------------------------------ 4. 市場の偏り
def market_bias(m) -> dict:
    """m: 2013-2025 の平地・単勝オッズあり・5 頭以上・勝ち馬 1 頭。p, band, per, K, win, top3, winret, plret, logsig, n"""
    res = {}
    m = m.copy()
    m["Kg"] = pd.cut(m.K, [4, 9, 13, 18], labels=False)
    m["Kg8"] = pd.cut(m.K, [7, 11, 15, 18], labels=False)
    m["pbin"] = m.groupby("per").p.transform(lambda s: pd.qcut(s, 40, labels=False, duplicates="drop"))
    key = [m.per, m.pbin, m.Kg]
    m["Ewin"] = m.groupby(key).win.transform("mean"); m["Ewinret"] = m.groupby(key).winret.transform("mean")
    m8 = m.K >= 8
    key8 = [m.per[m8], m.pbin[m8], m.Kg8[m8]]
    m["Etop3"] = np.nan; m["Eplret"] = np.nan
    m.loc[m8, "Etop3"] = m[m8].groupby(key8).top3.transform("mean")
    m.loc[m8, "Eplret"] = m[m8].groupby(key8).plret.transform("mean")
    el = m.n >= MIN_N
    m["terc"] = np.nan
    m.loc[el, "terc"] = m[el].groupby(["per", "band"]).logsig.transform(lambda s: pd.qcut(s, 3, labels=False, duplicates="drop"))
    for c in ("top3", "Etop3", "plret", "Eplret"):
        m[c + "8"] = np.where(m8, m[c], 0.0)
    m["n8"] = m8.astype(float); m["one"] = 1.0
    stats = ["one", "win", "p", "Ewin", "winret", "Ewinret", "n8", "top38", "Etop38", "plret8", "Eplret8"]
    bands = ["fav", "mid", "long"]; rng = np.random.default_rng(0)

    def derive(a):   # a[..., stat]
        o = {}
        o["n"] = a[..., 0]; o["win_rate"] = 100 * a[..., 1] / a[..., 0]
        o["AE_raw"] = a[..., 1] / a[..., 2]; o["AE_cal"] = a[..., 1] / a[..., 3]
        o["roi_win"] = 100 * a[..., 4] / a[..., 0]; o["roi_win_res"] = 100 * (a[..., 4] - a[..., 5]) / a[..., 0]
        o["n8"] = a[..., 6]; o["top3_rate"] = 100 * a[..., 7] / a[..., 6]; o["AE3_cal"] = a[..., 7] / a[..., 8]
        o["roi_place"] = 100 * a[..., 9] / a[..., 6]; o["roi_place_res"] = 100 * (a[..., 9] - a[..., 10]) / a[..., 6]
        return o

    for per in ("2013-19", "2020-25"):
        x = m[(m.per == per) & m.terc.notna()]
        dts, dc = np.unique(x.date.values, return_inverse=True)
        cell = x.band.map({b: i for i, b in enumerate(bands)}).values * 3 + x.terc.values.astype(int)
        A = np.zeros((len(dts), 9, len(stats)))
        for j, s in enumerate(stats):
            A[:, :, j] = np.bincount(dc * 9 + cell, weights=x[s].values, minlength=len(dts) * 9).reshape(len(dts), 9)
        point = derive(A.sum(0))
        I = rng.integers(0, len(dts), (N_BOOT, len(dts)))
        B = np.zeros((N_BOOT, 9, len(stats)))
        for b0 in range(0, N_BOOT, 100):
            B[b0:b0 + 100] = A[I[b0:b0 + 100]].sum(1)
        boot = derive(B)
        pr = {"n_dates": int(len(dts)), "sigma_mean_by_cell": {}, "cells": {}, "high_minus_low": {}}
        sm = x.groupby(["band", "terc"]).agg(sig=("logsig", lambda s: float(np.exp(s).mean())), odds=("tan_odds", "median"))
        for bi, bnd in enumerate(bands):
            for t in range(3):
                c = bi * 3 + t; nm = f"{bnd}|σ{'低中高'[t]}"
                pr["sigma_mean_by_cell"][nm] = {"sigma_mean": float(sm.loc[(bnd, t), "sig"]), "odds_median": float(sm.loc[(bnd, t), "odds"])}
                pr["cells"][nm] = {k: (float(v[c]) if k not in ("n", "n8") else int(v[c])) for k, v in point.items()}
                for k in ("AE_cal", "roi_win", "roi_win_res", "AE3_cal", "roi_place", "roi_place_res"):
                    pr["cells"][nm][k + "_ci95"] = [float(np.percentile(boot[k][:, c], 2.5)), float(np.percentile(boot[k][:, c], 97.5))]
            for k in ("AE_raw", "AE_cal", "roi_win_res", "AE3_cal", "roi_place_res", "roi_win", "roi_place"):
                dv = boot[k][:, bi * 3 + 2] - boot[k][:, bi * 3]
                lo, hi = float(np.percentile(dv, 2.5)), float(np.percentile(dv, 97.5))
                pr["high_minus_low"][f"{bnd}|{k}"] = {"diff": float(point[k][bi * 3 + 2] - point[k][bi * 3]), "ci95": [lo, hi],
                                                      "excl0": bool(lo > 0 or hi < 0)}
        res[per] = pr
    # 両期間で符号一致 & CI が 0 を含まない
    both = {}
    for k, a in res["2013-19"]["high_minus_low"].items():
        b = res["2020-25"]["high_minus_low"][k]
        both[k] = bool(a["excl0"] and b["excl0"] and np.sign(a["diff"]) == np.sign(b["diff"]))
    res["real_in_both_periods"] = both
    return res


def market_regression(m) -> dict:
    out = {}
    for per in ("2013-19", "2020-25"):
        x = m[m.per == per].reset_index(drop=True)
        codes = pd.factorize(x.rid_s)[0]; n = codes.max() + 1
        win_idx = np.flatnonzero(x.win.values == 1)
        lp = np.log(x.p.values)
        za = rc_center(x.logsig.values, codes)
        zb = za * (-rc_center(lp, codes))
        zc = pd.Series(x.zc.values).fillna(pd.Series(x.zc.values).groupby(codes).transform("mean")).fillna(0.0).values
        sds = {"za": za.std(), "zb": zb.std(), "zc": rc_center(zc, codes).std()}
        za_s, zb_s, zc_s = za / sds["za"], rc_center(zb, codes) / sds["zb"], rc_center(zc, codes) / sds["zc"]
        dr = pd.factorize(pd.Series(x.date.values).groupby(codes).first().sort_index().values)[0]
        pr = {"n_races": int(n), "win_clogit": {}, "top3_logit": {}}
        for nm, cols in (("A[za]", [za_s]), ("B[zb]", [zb_s]), ("A+B[za,zb]", [za_s, zb_s]), ("C[zc]", [zc_s])):
            X = np.column_stack([lp] + cols)
            w, se = clogit_cluster(X, codes, n, win_idx, dr)
            pr["win_clogit"][nm] = {"w_logp": float(w[0]), **{f"z{j}": coef_row(w, se, j) for j in range(1, X.shape[1])}}
        k8 = x.K.values >= 8
        base = top3_poly(lp[k8], x.K.values[k8]); y = x.top3.values[k8].astype(float)
        cl = pd.factorize(x.date.values[k8])[0]
        for nm, cols in (("A[za]", [za_s]), ("B[zb]", [zb_s]), ("A+B[za,zb]", [za_s, zb_s])):
            X = np.column_stack([base] + [c[k8] for c in cols])
            w, se = logit_cluster(X, y, cl)
            pr["top3_logit"][nm] = {f"z{j - 5}": coef_row(w, se, j) for j in range(6, X.shape[1])}
        out[per] = pr
    both = {}
    for fam in ("win_clogit", "top3_logit"):
        for nm in out["2013-19"][fam]:
            for zk in [k for k in out["2013-19"][fam][nm] if k.startswith("z")]:
                a, b = out["2013-19"][fam][nm][zk], out["2020-25"][fam][nm][zk]
                both[f"{fam}|{nm}|{zk}"] = bool(np.sign(a["coef"]) == np.sign(b["coef"]) and a["ci95"][0] * a["ci95"][1] > 0 and b["ci95"][0] * b["ci95"][1] > 0)
    out["real_in_both_periods"] = both
    return out


def asof_selfcheck(d, m, n_sample=300) -> dict:
    """S, n を「日付 < D の完走だけ」から作り直して一致するか"""
    rng = np.random.default_rng(7)
    smp = m[m.n >= MIN_N].sample(n_sample, random_state=7)
    hist = d[d.hk.isin(smp.hk.unique()) & (d.ijou == 0) & d.idm.notna()][["hk", "date", "idm"]]
    bad = 0; mx = 0.0
    for r in smp.itertuples():
        h = hist[(hist.hk == r.hk) & (hist.date < r.date)].sort_values("date")
        dy = np.diff(h.idm.values); gap = np.diff(h.date.values).astype("timedelta64[D]").astype(int)
        xx = (np.minimum(np.abs(dy), CLIP) ** 2 / 2.0)[(gap >= 1) & (gap <= 365)]
        e = abs(xx.sum() - r.S); mx = max(mx, e)
        bad += int(e > 1e-6 or len(xx) != r.n)
    return {"n_checked": int(n_sample), "n_mismatch": int(bad), "max_abs_err_S": float(mx)}


def explore(m, pri) -> dict:
    """事後の切り分け。2013-19 / 2020-23 の市場条件付きロジットだけ (test 2024-25 は使わない)"""
    out = {}
    k = pri["k"]
    for per, lo, hi in (("2013-19", "2013-01-01", "2019-12-31"), ("2020-23", "2020-01-01", "2023-12-31")):
        x = m[(m.date >= pd.Timestamp(lo)) & (m.date <= pd.Timestamp(hi))].reset_index(drop=True)
        codes = pd.factorize(x.rid_s)[0]; n = codes.max() + 1
        win_idx = np.flatnonzero(x.win.values == 1)
        dr = pd.factorize(pd.Series(x.date.values).groupby(codes).first().sort_index().values)[0]
        lp = np.log(x.p.values)
        st = lambda v: rc_center(v, codes) / rc_center(v, codes).std()
        za = st(x.logsig.values)
        hasl = x.xl.notna().values
        sig2_old = (k * x.s0sq.values + x.S.values - x.xl.fillna(0.0).values) / (k + x.n.values - hasl)
        za_old = st(0.5 * np.log(sig2_old))
        dl = x.dl.fillna(0.0).values / 10.0
        up, dn = np.clip(dl, 0, None), np.clip(-dl, 0, None)
        n0 = (x.n.values == 0).astype(float); ln = np.log1p(x.n.values)
        v3 = pd.Series(np.where(x.n.values >= MIN_N, x.logsig.values, np.nan))
        za_n3 = st(v3.fillna(v3.groupby(codes).transform("mean")).fillna(0.0).values)
        specs = {"E0 [za]": ([lp, za], ["logp", "za"]),
                 "E1 [logp^1..3 + za]": ([lp, lp ** 2, lp ** 3, za], ["logp", "logp2", "logp3", "za"]),
                 "E2 [za_old, last_up/10, last_down/10]": ([lp, za_old, up, dn], ["logp", "za_old", "last_up", "last_down"]),
                 "E2b [za_old]": ([lp, za_old], ["logp", "za_old"]),
                 "E4 [za, n==0, log1p(n)]": ([lp, za, n0, ln], ["logp", "za", "n0", "log1p_n"]),
                 "E5 [za (n>=3 only)]": ([lp, za_n3], ["logp", "za_n3"])}
        pr = {"n_races": int(n), "corr(za, za_old)": float(np.corrcoef(za, za_old)[0, 1])}
        for nm, (cols, names) in specs.items():
            w, se = clogit_cluster(np.column_stack(cols), codes, n, win_idx, dr)
            pr[nm] = {names[j]: coef_row(w, se, j) for j in range(len(names)) if not names[j].startswith("logp")}
        out[per] = pr
    return out


# ------------------------------------------------------------------ 5. ゲートの内訳 (年別・偽薬)
def prep(b, zcols):
    b = b.copy()
    for c in zcols:
        b[c] = pd.to_numeric(b[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        b[c] = b[c].fillna(b.groupby("rid_s")[c].transform("mean")).fillna(0.0)
        b[c] = b[c] - b.groupby("rid_s")[c].transform("mean")
        sd = b.loc[b.split == "valid", c].std()
        b[c] = b[c] / (sd if sd and sd > 0 else 1.0)
    return b


def by_year(b, zcols, bcols, l2=1e-4) -> dict:
    """gate_common.gate と同じ fit をして test のレース別 ΔNLL を年ごとに"""
    b = prep(b, zcols)
    d = b[b["lk10"].notna()] if "lk10" in bcols else b
    part = {}
    for sp in ("valid", "test"):
        x = d[d.split == sp].reset_index(drop=True)
        codes = pd.factorize(x["rid_s"])[0]
        part[sp] = (x, codes, codes.max() + 1, np.flatnonzero((x["jyun"] == 1).values))
    xv, cv, nv, wv = part["valid"]; xt, ct, nt, wt = part["test"]
    lps = []
    for cols in (bcols, bcols + zcols):
        w = gc._fit(xv[cols].values, cv, nv, wv, l2)
        lps.append(gc.race_logsoftmax(xt[cols].values @ w, ct, nt))
    diff = pd.Series(-(lps[1][wt] - lps[0][wt]), index=ct[wt]).sort_index().values
    dates = xt.groupby(ct)["date"].first().sort_index().values.astype(str)
    out = {}; rng = np.random.default_rng(0)
    for y in ("2024", "2025"):
        mk = np.char.startswith(dates, y)
        ud, dc = np.unique(dates[mk], return_inverse=True)
        s = np.bincount(dc, weights=diff[mk]); cnt = np.bincount(dc).astype(float)
        I = rng.integers(0, len(ud), (N_BOOT, len(ud)))
        v = s[I].sum(1) / cnt[I].sum(1)
        out[y] = {"n": int(mk.sum()), "d_nll": float(diff[mk].mean()), "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]}
    return out


def top3_check(b) -> dict:
    """valid で fit、test で対数損失 (nats/レース)。8 頭以上"""
    b = b.copy()
    b["K"] = b.groupby("rid_s").ban.transform("size")
    b = prep(b[b.K >= 8], ["za", "zb"])
    out = {}
    rng = np.random.default_rng(0)
    for nm, core in (("v7", "lm7"), ("market+v7", "lk")):
        def feats(x, plus):
            cols = [top3_poly(x[core].values, x.K.values)]
            if core == "lk":
                cols.append(x[["lm7"]].values)
            if plus:
                cols.append(x[plus].values)
            return np.column_stack(cols)
        v = b[b.split == "valid"]; t = b[b.split == "test"].reset_index(drop=True)
        yv = (v.jyun <= 3).values.astype(float); yt = (t.jyun <= 3).values.astype(float)
        ll = {}
        for tag, plus in (("base", None), ("A[za]", ["za"]), ("B[zb]", ["zb"]), ("A+B", ["za", "zb"])):
            w, _ = logit_fit(feats(v, plus), yv, l2=1e-4)
            p = np.clip(expit(feats(t, plus) @ w), 1e-9, 1 - 1e-9)
            ll[tag] = -(yt * np.log(p) + (1 - yt) * np.log(1 - p))
        rc = pd.factorize(t.rid_s)[0]; nr = rc.max() + 1
        dts = t.groupby(rc).date.first().sort_index().values
        ud, dc = np.unique(dts, return_inverse=True)
        cnt = np.bincount(dc).astype(float)
        I = rng.integers(0, len(ud), (N_BOOT, len(ud)))
        pr = {"n_test_races": int(nr), "ll_base_per_race": float(ll["base"].sum() / nr)}
        for tag in ("A[za]", "B[zb]", "A+B"):
            dr = np.bincount(rc, weights=ll[tag] - ll["base"], minlength=nr)
            s = np.bincount(dc, weights=dr)
            vb = s[I].sum(1) / cnt[I].sum(1)
            lo, hi = float(np.percentile(vb, 2.5)), float(np.percentile(vb, 97.5))
            pr[tag] = {"d_ll_per_race": float(dr.mean()), "ci95": [lo, hi], "verdict": "PASS" if hi < 0 else ("WORSE" if lo > 0 else "none")}
        out[nm] = pr
    return out


# ------------------------------------------------------------------ main
def main():
    res = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "script": "lab/experiments/volatility_option_exp.py"}
    d = load_runs()
    log(f"SED 平地 {len(d):,} 走、差 {int(d.x.notna().sum()):,} 本")

    # --- 特性かどうか + 事前分布 (〜2022)
    res["reliability_le2022"] = rel = reliability(d)
    log(f"信頼性 前半vs後半 r={rel['early_vs_late']['pearson_log_s']:.3f} (n={rel['early_vs_late']['n_horses']:,})  "
        f"重ならない差 r={rel['nonoverlap_alternate']['pearson_log_s']:.3f}  帰無 max={rel['perm_null_early_vs_late']['max']:.3f}")
    res["prior_le2022"] = pri = fit_prior(d)
    log(f"事前分布 s0(芝,ダ)={pri['s0_sd']}  k={pri['k']:.1f}  真の σ の CV≈{pri['implied_cv_of_true_sigma(approx)']:.3f}")
    s0v = d.td.map(pri["s0_sq"]).values
    d["sig2"] = (pri["k"] * s0v + d.S) / (pri["k"] + d.n)
    d["logsig"] = 0.5 * np.log(d.sig2)
    res["calibration_next_sqdiff"] = {"le2022(in-sample)": calib_deciles(d, d.date <= CUT),
                                      "2023-25(out-of-sample)": calib_deciles(d, (d.date > CUT) & (d.date <= pd.Timestamp("2025-12-31")))}
    co = res["calibration_next_sqdiff"]["2023-25(out-of-sample)"]
    if "--pre" in sys.argv:
        print(json.dumps(res, ensure_ascii=False, indent=1)); return
    log(f"次の差の sd (2023-25, σ̂ 十分位): 予 {co['decile_pred_sd'][0]:.2f}→{co['decile_pred_sd'][-1]:.2f}  実 {co['decile_realised_sd'][0]:.2f}→{co['decile_realised_sd'][-1]:.2f}")

    # --- 市場検定の母集団 (2013-2025)
    m = d[(d.date >= pd.Timestamp("2013-01-05")) & (d.date <= pd.Timestamp("2025-12-28")) & d.tan_odds.notna() & (d.tan_odds > 0)].copy()
    m["win"] = (m.jyun == 1).astype(float); m["top3"] = m.jyun.between(1, 3).astype(float)
    g = m.groupby("rid_s")
    m["K"] = g.ban.transform("size")
    m = m[(m.K >= 5) & (m.K <= KMAX) & (g.win.transform("sum") == 1)].copy()
    inv = 1.0 / m.tan_odds
    m["p"] = inv / inv.groupby(m.rid_s).transform("sum")
    m["per"] = np.where(m.date.dt.year <= 2019, "2013-19", "2020-25")
    m["band"] = np.where(m.tan_odds <= 2.9, "fav", np.where(m.tan_odds < 10, "mid", "long"))
    m["winret"] = m.win * m.tan_odds
    kk = pd.read_csv(KEKKA, encoding="cp932", dtype=str, usecols=["レースID(新)", "複勝配当"])
    kk["rid_s"] = kk["レースID(新)"].str[:16]; kk["ban"] = pd.to_numeric(kk["レースID(新)"].str[16:18], errors="coerce")
    kk["plret"] = pd.to_numeric(kk["複勝配当"], errors="coerce") / 100.0
    m = m.merge(kk.dropna(subset=["ban"])[["rid_s", "ban", "plret"]].drop_duplicates(["rid_s", "ban"]), on=["rid_s", "ban"], how="left")
    cov = float(m.loc[(m.top3 == 1) & (m.K >= 8), "plret"].notna().mean())
    m["plret"] = m.plret.fillna(0.0)
    m = m.sort_values(["rid_s", "ban"]).reset_index(drop=True)
    m["rc"] = pd.factorize(m.rid_s)[0]; m["pos"] = m.groupby("rc").cumcount()
    res["market_population"] = {"n_runs": int(len(m)), "n_races": int(m.rc.max() + 1), "place_payout_coverage_top3_K8": cov,
                                "share_n_ge3": float((m.n >= MIN_N).mean())}
    log(f"市場検定 {res['market_population']}")

    # --- Thurstone (C)
    has = m.m5.notna()
    m["fracm"] = has.groupby(m.rc).transform("mean")
    m["mfill"] = m.m5.fillna(m.groupby("rc").m5.transform("mean")).fillna(0.0)
    m["s0sq"] = m.td.map(pri["s0_sq"])
    ok_r = m.fracm >= 0.6
    fit_ids = m.loc[ok_r & (m.date >= pd.Timestamp("2016-01-01")) & (m.date <= CUT), "rc"].unique()
    fit_ids = np.random.default_rng(0).choice(fit_ids, 6000, replace=False)
    f = m[m.rc.isin(fit_ids)].copy()
    f["rc"] = pd.factorize(f.rc)[0]
    wmask = f.win.values == 1

    def nll_hom(s):
        f["sd_"] = s
        return float(-np.log(np.clip(thurstone_frame(f, "sd_"), 1e-12, None))[wmask].mean())
    s_c = float(minimize_scalar(nll_hom, bounds=(3.0, 40.0), method="bounded", options={"xatol": 0.02}).x)
    f["sd_"] = s_c; nh = float(-np.log(np.clip(thurstone_frame(f, "sd_"), 1e-12, None))[wmask].mean())
    f["sd_"] = np.sqrt(np.maximum(s_c ** 2 - f.s0sq, 0.0) + f.sig2)
    nhet = float(-np.log(np.clip(thurstone_frame(f, "sd_"), 1e-12, None))[wmask].mean())
    res["thurstone_fit_2016_22"] = {"s_c": s_c, "n_races": int(len(fit_ids)), "nll_hom": nh, "nll_het": nhet, "d_nll_het_minus_hom": nhet - nh}
    log(f"Thurstone s_c={s_c:.2f}  2016-22 fit 標本: 等分散 {nh:.4f} → 不等分散 {nhet:.4f} (Δ {nhet - nh:+.5f})")
    m["sd_hom"] = s_c
    m["sd_het"] = np.sqrt(np.maximum(s_c ** 2 - m.s0sq, 0.0) + m.sig2)
    ph = thurstone_frame(m, "sd_hom"); pt = thurstone_frame(m, "sd_het")
    m["zc"] = np.clip(np.log(np.clip(pt, 1e-4, None)) - np.log(np.clip(ph, 1e-4, None)), -2, 2)
    m.loc[~ok_r, "zc"] = np.nan
    log(f"zc 計算済 (欠損 {100 * m.zc.isna().mean():.1f}%, sd {m.zc.std():.4f})")
    m[["rid_s", "ban", "date", "n", "logsig", "zc", "p", "band", "per"]].to_parquet(FEAT_PQ)

    # --- 市場の偏り (2 期間)
    res["market_bias"] = mb = market_bias(m)
    for per in ("2013-19", "2020-25"):
        for k, v in mb[per]["high_minus_low"].items():
            if k.split("|")[1] in ("AE_cal", "roi_win_res", "AE3_cal", "roi_place_res"):
                log(f"  [{per}] σ高−低 {k:22s} {v['diff']:+.4f} [{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}]{' *' if v['excl0'] else ''}")
    log(f"  両期間で実在: {[k for k, v in mb['real_in_both_periods'].items() if v] or 'なし'}")
    res["market_regression"] = mr = market_regression(m)
    for per in ("2013-19", "2020-25"):
        for fam in ("win_clogit", "top3_logit"):
            for nm, v in mr[per][fam].items():
                log(f"  [{per}] {fam} {nm}: " + "  ".join(f"{zk} {c['coef']:+.4f} (z={c['z']:+.2f})" for zk, c in v.items() if zk.startswith("z")))
    log(f"  回帰版 両期間で実在: {[k for k, v in mr['real_in_both_periods'].items() if v] or 'なし'}")

    # --- ゲート (valid 2023 / test 2024-25、ここで初めて test を見る)
    base = gc.load_base()
    b = base.merge(m[["rid_s", "ban", "logsig", "zc", "n", "m5", "mfill", "sd_hom", "s0sq", "sig2", "fracm"]], on=["rid_s", "ban"], how="left")
    res["base_join"] = {"rows": int(len(b)), "sigma_coverage": float(b.logsig.notna().mean()), "share_n_ge3": float((b.n >= MIN_N).mean()),
                        "zc_coverage": float(b.zc.notna().mean())}

    def make_z(bb, col="logsig"):
        bb = bb.copy()
        bb["za"] = bb[col]
        ca = bb[col] - bb.groupby("rid_s")[col].transform("mean")
        bb["zb"] = ca * (-(bb.lm7 - bb.groupby("rid_s").lm7.transform("mean")))
        return bb
    b = make_z(b)
    lite = pd.read_parquet(gc.LITE_PQ, columns=["レースID(新/馬番無)", "馬番", "kako5_std_pos"])
    lite["rid_s"] = lite["レースID(新/馬番無)"].astype(str).str[:16]; lite["ban"] = pd.to_numeric(lite["馬番"], errors="coerce")
    bt = b[b.split == "test"].merge(lite[["rid_s", "ban", "kako5_std_pos"]].dropna(subset=["ban"]).drop_duplicates(["rid_s", "ban"]),
                                    on=["rid_s", "ban"], how="left")
    cen = lambda c: bt[c] - bt.groupby("rid_s")[c].transform("mean")
    res["diagnostics_test_no_outcome"] = {
        "corr(log_sigma, lm7) within race": float(cen("logsig").corr(cen("lm7"))),
        "corr(log_sigma, lk) within race": float(cen("logsig").corr(cen("lk"))),
        "corr(log_sigma, kako5_std_pos) within race": float(cen("logsig").corr(cen("kako5_std_pos"))),
        "corr(zc, lm7) within race": float(cen("zc").corr(cen("lm7"))),
        "sd(log_sigma) within race": float(cen("logsig").std())}
    log(f"診断 {res['diagnostics_test_no_outcome']}")

    res["gates"] = {}; res["by_year"] = {}
    for nm, col in (("A_log_sigma", "za"), ("B_log_sigma_x_underdog", "zb"), ("C_thurstone_het", "zc")):
        res["gates"][nm] = gc.gate(b[["rid_s", "ban", col]], [col], name=nm)
        res["by_year"][nm] = {gn: by_year(b, [col], bc) for gn, bc in (("P", ["lm7"]), ("M", ["lk", "lm7"]))}

    # --- 偽薬: σ_h をレース内で入れ替える
    rng = np.random.default_rng(1)
    pb = b.copy()
    pb["logsig_pl"] = pb.groupby("rid_s")["logsig"].transform(lambda v: rng.permutation(v.values))   # 同じレースの別の馬の値
    pb["sig2_pl"] = np.exp(2 * pb["logsig_pl"])
    pz = make_z(pb, "logsig_pl")
    # zc の偽薬: base のレースで sd を入れ替えて Thurstone をやり直す (m5 がそろっているレースだけ)
    q = pz[pz.sig2_pl.notna() & pz.mfill.notna()].sort_values(["rid_s", "ban"]).copy()
    q["rc"] = pd.factorize(q.rid_s)[0]; q["pos"] = q.groupby("rc").cumcount()
    q = q[q.groupby("rc").pos.transform("max") < KMAX]
    q["sd_het"] = np.sqrt(np.maximum(s_c ** 2 - q.s0sq, 0.0) + q.sig2_pl)
    q["zc_pl"] = np.clip(np.log(np.clip(thurstone_frame(q, "sd_het"), 1e-4, None)) - np.log(np.clip(thurstone_frame(q, "sd_hom"), 1e-4, None)), -2, 2)
    q.loc[q.fracm < 0.6, "zc_pl"] = np.nan
    pz = pz.merge(q[["rid_s", "ban", "zc_pl"]], on=["rid_s", "ban"], how="left")
    res["placebo_shuffle_within_race"] = {}
    for nm, col in (("A_placebo", "za"), ("B_placebo", "zb"), ("C_placebo", "zc_pl")):
        res["placebo_shuffle_within_race"][nm] = gc.gate(pz[["rid_s", "ban", col]], [col], name=nm)

    # --- 3 着内
    res["top3_logistic_valid_fit_test_eval"] = t3 = top3_check(b)
    for nm, v in t3.items():
        log(f"3着内 [{nm}] " + "  ".join(f"{k} {v[k]['d_ll_per_race']:+.5f} [{v[k]['ci95'][0]:+.5f},{v[k]['ci95'][1]:+.5f}] {v[k]['verdict']}" for k in ("A[za]", "B[zb]", "A+B")))

    # --- 事後の追加 (探索): as-of 自己点検と切り分け
    res["asof_selfcheck"] = asof_selfcheck(d, m)
    log(f"as-of 自己点検 {res['asof_selfcheck']}")
    res["exploratory_posthoc_2013_23"] = ex = explore(m, pri)
    for per, pr in ex.items():
        for nm, v in pr.items():
            if isinstance(v, dict):
                log(f"  [探索 {per}] {nm}: " + "  ".join(f"{a} {c['coef']:+.4f} (z={c['z']:+.2f})" for a, c in v.items()))

    # --- 判定
    gv = {nm: {g: r["gates"][g]["verdict"] for g in ("P", "M", "M0", "A10")} for nm, r in res["gates"].items()}
    res["verdict"] = {
        "trait_exists": bool(rel["early_vs_late"]["pearson_log_s"] >= 0.05 and rel["early_vs_late"]["pearson_log_s"] > rel["perm_null_early_vs_late"]["max"]),
        "gate_verdicts": gv,
        "signal_prediction": bool(any(v["P"] == "PASS" for v in gv.values())),
        "signal_beyond_market": bool(any(v["M"] == "PASS" for v in gv.values())),
        "market_bias_real_in_both_periods": [k for k, v in mb["real_in_both_periods"].items() if v and k.split("|")[1] in ("AE_cal", "roi_win_res", "AE3_cal", "roi_place_res")],
        "market_regression_real_in_both_periods": [k for k, v in mr["real_in_both_periods"].items() if v]}
    log(f"判定 {res['verdict']}")
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"→ {OUT}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    main()
