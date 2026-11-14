# -*- coding: utf-8 -*-
"""
random_slope_affinity_exp.py — 階層ランダム傾き (馬ごとの「条件への感度」を血統の事前分布つきで推定)  2026-11-14
====================================================================================================
理論
  能力 1 次元 (どれだけ強いか) を完全に知っても v7 に届かない。足りないのは「馬 × 条件」の相性。
  同条件の出走回数を数える疎な集計ではなく、各走の出来 (JRDB IDM) を
      idm_run = level_h + Σ_k slope_{h,k} · x_k(run) + (年齢・キャリアの共通曲線) + noise
  と置き、連続の条件軸 x_k に対する馬ごとの傾き slope_{h,k} = β_k(全体) + s_{sire,k}(種牡馬) + u_{h,k}(個体)
  を経験ベイズ (BLUP) で縮小推定する。今日の条件が普段とどれだけ違うかに傾きを掛けたものが z。

==================== 事前登録 (test を見る前に固定。変更しない) ====================
[データ] master_lite の 2013-2025 全走に SED (idm, ijou, td, bw) を rid_s+ban で結合。
  過去走として使う条件: 平地 (td∈{1,2})、ijou==0、idm あり、条件軸が全部そろう。
  idm は ≤2022 の 1% / 99% 点で両側を丸める (大差負けの外れ値対策)。
[条件軸 K=10] (≤2022 の過去走の平均・標準偏差で標準化)
  ld      log(距離/1600)
  dirt    ダート=1 / 芝=0
  wet_t   馬場状態 (良0 稍1 重2 不3) × 芝
  wet_d   馬場状態 × ダート
  field   出走頭数
  left    左回り=1 (東京・中京・新潟)、右=0、新潟芝直線=0.5
  hill    急坂 (中山・阪神・中京)=1
  hot     6-9 月=1
  kin     100 × 斤量 / 直近の馬体重 (前走馬体重。欠ける過去走だけその走の馬体重で代用。今日の行は代用しない)
  straight 最後の直線の長さ (m、場×芝ダ×内外。新潟直線は外回りと同じ 659)
[共通曲線 (全馬共通、馬ごとの傾きは持たせない)] 月齢の折れ線 (36/48/60 か月で折る) + log(1+それまでの出走数)
[推定 — 全部 ≤2022 の走だけ]
  1. β と共通曲線: 馬の固定効果つき (馬内で中心化) OLS。対象は 2011 年以降生まれ・過去走 2 以上の馬。
  2. 分散成分 σ² (走ごとの雑音)、τ_k² (個体の傾きのばらつき)、ω_k² (種牡馬の傾きのばらつき): 対角、
     馬の水準を消した周辺尤度の最尤 (馬→種牡馬の 3 階層、十分統計量から閉形式)。
  3. 種牡馬の傾きの事後分布: ≤2022 の全産駒から。評価する馬自身の ≤2022 の走は抜く (leave-one-out)。
[評価日 D の馬の傾き] その馬の D より前の走だけで閉形式リッジ。事前平均 = 種牡馬の事後平均、
  事前分散 = 種牡馬の事後分散 + diag(τ²)。2023 年以降の「D より前の自分の走」は使う (as-of)、
  β・分散成分・種牡馬の事前分布は ≤2022 のまま動かさない。
  Δx_k = x_k(今日) − その馬の過去走の x_k の平均 (過去走 0 の馬・障害戦は z=NaN、今日の kin が欠ければ Δ=0)。
  今日の馬場状態は発走前に発表されている値 (当日情報。市場は知っている)。
[変種 (3 つ、全部報告)]
  (a) z_total = Σ_k (β_k + ŝ_k + û_k) Δx_k                                   1 列  ← 判定の主役
  (b) 軸ごと: z_k = (β_k + ŝ_k + û_k) Δx_k を 9 列 (wet_t と wet_d は足して 1 列)
  (c) 種牡馬の事前分布だけ (個体の走で傾きを更新しない): z_sire = Σ_k (β_k + ŝ_k) Δx_k   1 列
[判定]
  gate_common.gate の 4 ゲート (P / M / M0 / A10)。CI95 の上端 < 0 で PASS。
  SIGNAL(予測) = (a) の P が PASS。SIGNAL(市場外) = (a) の M が PASS。
  (b)(c) だけが PASS の場合は 12 検定中の当たりなので「要追試」どまり (年別符号一致と偽薬の両方を通ったときだけ)。
  どれかが PASS したら: 2024 / 2025 の年別符号、偽薬 (馬ごとの傾きの偏差 ŝ+û を馬の間で入れ替え、β と Δx はそのまま)。
  全部 none なら null、同じ定式化の再走は禁止。
[test を見る前にやってよい確認] 2023 (valid) の走で「予測した出来の上下 (ŝ+û)·Δx」が実際の IDM の上下を当てるか。
====================================================================================

実行: (worktree root から)  PYTHONUTF8=1 E:/PyCaLiAI/venv311/Scripts/python.exe lab/experiments/random_slope_affinity_exp.py [--no-test | --posthoc]
  --no-test  valid の確認まで (ゲートを回さない)。  --posthoc  判定後の説明用の集計 (valid だけ) を JSON に追記。
出力: reports/random_slope_affinity_exp.json (集計値のみ)、reports/rs_z.parquet (作業用、JRDB 由来なので reports/ の外へ出さない)
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate_common as gc  # noqa: E402

log = gc.log
REP = gc.REP
OUT_JSON = REP / "random_slope_affinity_exp.json"
OUT_Z = REP / "rs_z.parquet"
TRAIN_END = 20221231
DIMS = ["ld", "dirt", "wet_t", "wet_d", "field", "left", "hill", "hot", "kin", "straight"]
K = len(DIMS)
OUT_DIMS = ["ld", "dirt", "wet", "field", "left", "hill", "hot", "kin", "straight"]
STRAIGHT_T = {"札幌": 266, "函館": 262, "福島": 292, "東京": 526, "中山": 310, "中京": 413, "小倉": 293}
STRAIGHT_T_IO = {("新潟", "内"): 359, ("新潟", "外"): 659, ("京都", "内"): 328, ("京都", "外"): 404,
                 ("阪神", "内"): 357, ("阪神", "外"): 474}
STRAIGHT_D = {"札幌": 264, "函館": 260, "福島": 296, "新潟": 354, "東京": 502, "中山": 308, "中京": 411,
              "京都": 329, "阪神": 353, "小倉": 291}
I_K = np.eye(K)


# ------------------------------------------------------------------ data
def build_runs() -> pd.DataFrame:
    cols = ["日付", "場所", "レースID(新/馬番無)", "血統登録番号", "馬番", "芝・ダ", "距離", "芝(内・外)", "馬場状態",
            "種牡馬", "斤量", "出走頭数", "前走馬体重", "トラックコード(JV)"]
    d = pd.read_parquet(gc.LITE_PQ, columns=cols)
    d["rid_s"] = d["レースID(新/馬番無)"].astype(str).str[:16]
    d["ban"] = pd.to_numeric(d["馬番"], errors="coerce").astype(int)
    sed = gc.load_sed(["racekey", "ymd", "ban", "td", "ijou", "idm", "bw"]).drop_duplicates(["rid_s", "ban"])
    d = d.merge(sed[["rid_s", "ban", "td", "ijou", "idm", "bw"]], on=["rid_s", "ban"], how="left")
    del sed
    d["ymd"] = d["日付"].astype(int)
    d["ped"] = d["血統登録番号"].astype(str)
    d["birth"] = d["ped"].str[:4].astype(int)
    tc = pd.to_numeric(d["トラックコード(JV)"], errors="coerce")
    d["jump"] = (tc >= 50) | (d["td"] == 3)
    place = d["場所"].astype(str).str.strip()
    dirt = (d["芝・ダ"].astype(str).str.strip() == "ダ").astype(float)
    wet = d["馬場状態"].astype(str).str.strip().map({"良": 0.0, "稍": 1.0, "重": 2.0, "不": 3.0})
    io = d["芝(内・外)"].astype(str).str.strip()
    st_t = place.map(STRAIGHT_T)
    st_io = pd.Series([STRAIGHT_T_IO.get(k, np.nan) for k in zip(place, io)], index=d.index)
    st_t = st_t.fillna(st_io)
    straight_line = (place == "新潟") & (tc == 10)
    st_t = st_t.where(~straight_line, 659.0)
    st = np.where(dirt == 1, place.map(STRAIGHT_D), st_t)
    left = place.isin(["東京", "中京", "新潟"]).astype(float).where(~straight_line, 0.5)
    kin_kg = pd.to_numeric(d["斤量"].astype(str).str.extract(r"(\d+(?:\.\d+)?)")[0], errors="coerce")
    bw_prev = pd.to_numeric(d["前走馬体重"], errors="coerce")
    bw_prev = bw_prev.where(bw_prev > 300)
    bw_own = pd.to_numeric(d["bw"], errors="coerce")
    bw_own = bw_own.where(bw_own > 300)
    d["x_ld"] = np.log(pd.to_numeric(d["距離"], errors="coerce") / 1600.0)
    d["x_dirt"] = dirt
    d["x_wet_t"] = wet * (1 - dirt)
    d["x_wet_d"] = wet * dirt
    d["x_field"] = pd.to_numeric(d["出走頭数"], errors="coerce")
    d["x_left"] = left
    d["x_hill"] = place.isin(["中山", "阪神", "中京"]).astype(float)
    d["x_hot"] = ((d["ymd"] // 100) % 100).isin([6, 7, 8, 9]).astype(float)
    d["x_kin_today"] = 100.0 * kin_kg / bw_prev                   # 今日の行で使う値 (発走前に分かる)
    d["x_kin"] = 100.0 * kin_kg / bw_prev.fillna(bw_own)          # 過去走として使う値
    d["x_straight"] = pd.to_numeric(pd.Series(st, index=d.index), errors="coerce")
    d["age_m"] = ((d["ymd"] // 10000) - d["birth"]) * 12 + ((d["ymd"] // 100) % 100)
    d = d.sort_values(["ped", "ymd", "rid_s"]).reset_index(drop=True)
    d["career"] = d.groupby("ped").cumcount()
    xc = ["x_" + k for k in DIMS]
    d["hist"] = ((~d["jump"]) & d["td"].isin([1, 2]) & (d["ijou"] == 0) & d["idm"].notna() & d[xc].notna().all(axis=1))
    keep = ["rid_s", "ban", "ped", "ymd", "birth", "種牡馬", "jump", "hist", "idm", "age_m", "career", "x_kin_today"] + xc
    return d[keep]


def nuisance_design(d: pd.DataFrame) -> np.ndarray:
    a = d["age_m"].values / 12.0
    return np.column_stack([a, np.clip(a - 3.0, 0, None), np.clip(a - 4.0, 0, None), np.clip(a - 5.0, 0, None),
                            np.log1p(d["career"].values.astype(float))])


def group_stats(X: np.ndarray, r: np.ndarray, gid: np.ndarray):
    """gid は 0..G-1 で昇順に並んでいること。馬内中心化した S=X̃'X̃, t=X̃'r̃, q=r̃'r̃, n を返す。"""
    starts = np.flatnonzero(np.r_[True, gid[1:] != gid[:-1]])
    n = np.diff(np.r_[starts, len(gid)]).astype(float)
    sx = np.add.reduceat(X, starts, axis=0)
    sr = np.add.reduceat(r, starts)
    xx = (X[:, :, None] * X[:, None, :]).reshape(len(X), -1)
    sxx = np.add.reduceat(xx, starts, axis=0).reshape(-1, K, K)
    del xx
    sxr = np.add.reduceat(X * r[:, None], starts, axis=0)
    srr = np.add.reduceat(r * r, starts)
    xb = sx / n[:, None]; rb = sr / n
    S = sxx - n[:, None, None] * xb[:, :, None] * xb[:, None, :]
    t = sxr - n[:, None] * xb * rb[:, None]
    q = srr - n * rb * rb
    return S, t, q, n


def horse_Ab(S, t, s2, T):
    """馬の傾きの個体偏差 u~N(0,T) を積分した後の、種牡馬の傾きに対する精度 A=X'V^{-1}X と b=X'V^{-1}r。"""
    Th = np.sqrt(T)
    M = I_K + (Th[None, :, None] * S * Th[None, None, :]) / s2
    STh = S * Th[None, None, :]
    Tt = t * Th[None, :]
    u = np.linalg.solve(M, Tt[:, :, None])[:, :, 0]
    W = np.linalg.solve(M, np.swapaxes(STh, 1, 2))
    A = S / s2 - (STh @ W) / s2 ** 2
    b = t / s2 - (STh @ u[:, :, None])[:, :, 0] / s2 ** 2
    return A, b, M, Tt, u


def make_negll(S, t, q, n, sire_codes, n_sire):
    """3 階層の周辺尤度 (馬の水準は中心化で消去、個体の傾き u を積分 → 種牡馬の傾き s を積分)。
    torch の自動微分で勾配つき (数値勾配だと 21 パラメータ × 5 万頭で遅すぎた)。"""
    import torch
    torch.set_num_threads(4)
    f64 = torch.float64
    S_ = torch.from_numpy(S); t_ = torch.from_numpy(t); q_ = torch.from_numpy(q); n_ = torch.from_numpy(n)
    sc = torch.from_numpy(sire_codes.astype(np.int64)); eye = torch.eye(K, dtype=f64)
    G = len(n)

    def f(theta):
        th = torch.tensor(np.asarray(theta, dtype=float), dtype=f64, requires_grad=True)
        s2 = torch.exp(th[0]); T = torch.exp(th[1:K + 1]); Om = torch.exp(th[K + 1:])
        Th = torch.sqrt(T)
        M = eye + (Th[None, :, None] * S_ * Th[None, None, :]) / s2
        L = torch.linalg.cholesky(M)
        logdet = 2 * torch.log(torch.diagonal(L, dim1=1, dim2=2)).sum(-1)
        STh = S_ * Th[None, None, :]
        Tt = t_ * Th[None, :]
        u = torch.cholesky_solve(Tt[:, :, None], L)[:, :, 0]
        W = torch.cholesky_solve(STh.transpose(1, 2), L)
        A = S_ / s2 - (STh @ W) / s2 ** 2
        b = t_ / s2 - (STh @ u[:, :, None])[:, :, 0] / s2 ** 2
        quad = q_ / s2 - (Tt * u).sum(1) / s2 ** 2
        ll = -0.5 * ((n_ - 1) * torch.log(s2) + logdet + quad)
        As = torch.zeros((n_sire, K, K), dtype=f64).index_add(0, sc, A)
        As = 0.5 * (As + As.transpose(1, 2))
        bs = torch.zeros((n_sire, K), dtype=f64).index_add(0, sc, b)
        Oh = torch.sqrt(Om)
        Ms = eye + Oh[None, :, None] * As * Oh[None, None, :]
        Ls = torch.linalg.cholesky(Ms)
        ob = bs * Oh[None, :]
        us = torch.cholesky_solve(ob[:, :, None], Ls)[:, :, 0]
        lls = -torch.log(torch.diagonal(Ls, dim1=1, dim2=2)).sum(-1) + 0.5 * (ob * us).sum(1)
        loss = -(ll.sum() + lls.sum()) / G
        loss.backward()
        return float(loss.item()), th.grad.numpy().copy()
    return f


# ------------------------------------------------------------------ main
def main(no_test: bool = False):
    t0 = time.time()
    res: dict = {"experiment": "random_slope_affinity", "date": "2026-11-14", "dims": DIMS, "train_end": TRAIN_END}
    d = build_runs()
    log(f"[runs] {len(d):,} 行  過去走として使える {int(d['hist'].sum()):,}  ({time.time() - t0:.0f}s)")

    # ---- 標準化と idm の丸め (≤2022 の過去走で決める)
    tr_hist = d["hist"] & (d["ymd"] <= TRAIN_END)
    xc = ["x_" + k for k in DIMS]
    mu = d.loc[tr_hist, xc].mean().values; sd = d.loc[tr_hist, xc].std().values
    lo, hi = np.percentile(d.loc[tr_hist, "idm"], [1, 99])
    X = (d[xc].values - mu) / sd
    kin_i = DIMS.index("kin")
    Xtoday = X.copy()
    Xtoday[:, kin_i] = (d["x_kin_today"].values - mu[kin_i]) / sd[kin_i]
    y = np.clip(d["idm"].values, lo, hi)
    Z = nuisance_design(d)
    res["standardize"] = {k: {"mean": float(m), "sd": float(s)} for k, m, s in zip(DIMS, mu, sd)}
    res["idm_clip"] = [float(lo), float(hi)]

    # ---- 1. β と共通曲線 (馬の固定効果つき OLS、≤2022、2011 年以降生まれ、過去走 2 以上)
    fit = (tr_hist & (d["birth"] >= 2011)).values
    cnt = pd.Series(fit).groupby(d["ped"].values).transform("sum").values
    fit &= cnt >= 2
    gid = pd.factorize(d.loc[fit, "ped"])[0]                      # d は ped 順なので昇順
    D = np.column_stack([X[fit], Z[fit]]); yf = y[fit]
    Dm = D - pd.DataFrame(D).groupby(gid).transform("mean").values
    ym = yf - pd.Series(yf).groupby(gid).transform("mean").values
    coef = np.linalg.lstsq(Dm, ym, rcond=None)[0]
    beta, gamma = coef[:K], coef[K:]
    log("[β] " + "  ".join(f"{k}:{b:+.3f}" for k, b in zip(DIMS, beta)))
    res["beta_global_per_sd"] = {k: float(b) for k, b in zip(DIMS, beta)}
    res["nuisance_coef"] = [float(g) for g in gamma]
    res["n_fit_runs"] = int(fit.sum()); res["n_fit_horses"] = int(gid.max() + 1)
    r = y - Z @ gamma - X @ beta                                   # 全走の残差 (係数は ≤2022 のもの)

    # ---- 2. 分散成分 (3 階層の周辺尤度)
    S, t, q, n = group_stats(X[fit], r[fit], gid)
    sire_of = d.loc[fit, ["ped", "種牡馬"]].drop_duplicates("ped")["種牡馬"].fillna("").astype(str).values
    sire_of = np.where(sire_of == "", ["__none_%d" % i for i in range(len(sire_of))], sire_of)
    sire_fit_codes, sire_fit_names = pd.factorize(sire_of)
    negll = make_negll(S, t, q, n, sire_fit_codes, len(sire_fit_names))
    th0 = np.r_[np.log(q.sum() / (n - 1).sum()), np.full(K, np.log(0.5)), np.full(K, np.log(0.1))]
    f0 = negll(th0)[0]
    te = time.time()
    opt = minimize(negll, th0, jac=True, method="L-BFGS-B", bounds=[(-14, 8)] * (2 * K + 1), options={"maxiter": 500})
    s2 = float(np.exp(opt.x[0])); T = np.exp(opt.x[1:K + 1]); Om = np.exp(opt.x[K + 1:])
    # 帰無 (傾きのばらつきなし) の尤度と比べる
    th_null = np.r_[np.log(q.sum() / (n - 1).sum()), np.full(2 * K, -14.0)]
    f_null = negll(th_null)[0]
    opt_ns = minimize(lambda v: (lambda r_: (r_[0], r_[1][:K + 1]))(negll(np.r_[v, np.full(K, -14.0)])), opt.x[:K + 1], jac=True,
                      method="L-BFGS-B", bounds=[(-14, 8)] * (K + 1), options={"maxiter": 500})
    log(f"[ML] negll/horse {f0:.4f}→{opt.fun:.4f} (傾きなし {f_null:.4f})  σ={np.sqrt(s2):.2f}  iters {opt.nit}  ({time.time() - te:.0f}s)")
    log("[τ 個体] " + "  ".join(f"{k}:{np.sqrt(v):.2f}" for k, v in zip(DIMS, T)))
    log("[ω 種牡馬] " + "  ".join(f"{k}:{np.sqrt(v):.2f}" for k, v in zip(DIMS, Om)))
    res["variance_components"] = {
        "sigma_noise": float(np.sqrt(s2)),
        "tau_horse_sd": {k: float(np.sqrt(v)) for k, v in zip(DIMS, T)},
        "omega_sire_sd": {k: float(np.sqrt(v)) for k, v in zip(DIMS, Om)},
        "negll_per_horse": {"init": float(f0), "fitted": float(opt.fun), "no_slope_variance": float(f_null)},
        "loglik_gain_total": float((f_null - opt.fun) * len(n)),
        "loglik_gain_from_sire_layer": float((opt_ns.fun - opt.fun) * len(n)), "n_horses": int(len(n)), "n_sires": int(len(sire_fit_names)),
        "converged": bool(opt.success), "note": "x は標準化済み。sd は IDM 点 / 条件 1 標準偏差"}
    del S, t, q, n

    # ---- 3. 種牡馬の傾き (≤2022 の全産駒)
    use = tr_hist.values.copy()
    cnt = pd.Series(use).groupby(d["ped"].values).transform("sum").values
    use &= cnt >= 2
    gid2, peds2 = pd.factorize(d.loc[use, "ped"])
    S2, t2, q2, n2 = group_stats(X[use], r[use], gid2)
    A2, b2, *_ = horse_Ab(S2, t2, s2, T)
    del S2, t2
    sire2 = d.loc[use, ["ped", "種牡馬"]].drop_duplicates("ped")["種牡馬"].fillna("").astype(str).values
    sire_codes, sire_names = pd.factorize(sire2)
    As = np.zeros((len(sire_names), K, K)); bs = np.zeros((len(sire_names), K))
    np.add.at(As, sire_codes, A2); np.add.at(bs, sire_codes, b2)
    none_i = np.flatnonzero(sire_names == "")
    if len(none_i):
        As[none_i] = 0; bs[none_i] = 0
    Oh = np.sqrt(Om)
    Ms = I_K + Oh[None, :, None] * As * Oh[None, None, :]
    m_s = Oh[None, :] * np.linalg.solve(Ms, (bs * Oh[None, :])[:, :, None])[:, :, 0]
    prog = np.bincount(sire_codes, minlength=len(sire_names))
    big = prog >= 100
    res["sire_prior"] = {"n_sires": int(len(sire_names)), "n_sires_100plus_progeny": int(big.sum()),
                         "posterior_mean_sd_across_big_sires": {k: float(m_s[big][:, i].std()) for i, k in enumerate(DIMS)}}
    ped2idx = pd.Series(np.arange(len(peds2)), index=peds2)
    sire2idx = pd.Series(np.arange(len(sire_names)), index=sire_names)

    # ---- 4. 評価する馬の as-of 統計 (D より前の自分の走の累積)
    base = gc.load_base()[["rid_s", "ban", "ped", "split", "date", "jyun"]]
    sub = d["ped"].isin(set(base["ped"].astype(str))).values
    ds = d.loc[sub].reset_index(drop=True)
    Xs = X[sub]; Xts = Xtoday[sub]; rs = r[sub]
    h = ds["hist"].values.astype(float)
    Xh = np.where(h[:, None] > 0, Xs, 0.0); rh = np.where(h > 0, rs, 0.0)
    contrib = np.column_stack([h, Xh, rh, (Xh[:, :, None] * Xh[:, None, :]).reshape(len(ds), -1), Xh * rh[:, None]])
    cs = np.cumsum(contrib, axis=0) - contrib                     # 自分を含まない累積
    g0 = np.flatnonzero(np.r_[True, ds["ped"].values[1:] != ds["ped"].values[:-1]])
    gidx = np.repeat(g0, np.diff(np.r_[g0, len(ds)]))
    cs = cs - cs[gidx]
    del contrib
    key = ds[["rid_s", "ban"]].copy(); key["row"] = np.arange(len(ds))
    bb = base.merge(key, on=["rid_s", "ban"], how="left")
    assert bb["row"].notna().all(), "base 行が master_lite に無い"
    rows = bb["row"].values.astype(int)
    c = cs[rows]; del cs
    nh = c[:, 0]; sx = c[:, 1:1 + K]; sr = c[:, 1 + K]
    sxx = c[:, 2 + K:2 + K + K * K].reshape(-1, K, K); sxr = c[:, 2 + K + K * K:]
    nn = np.maximum(nh, 1.0)
    xb = sx / nn[:, None]; rb = sr / nn
    Sh = sxx - nn[:, None, None] * xb[:, :, None] * xb[:, None, :]
    th = sxr - nn[:, None] * xb * rb[:, None]
    ok = (nh >= 1) & (~ds["jump"].values[rows])
    dx = Xts[rows] - xb
    dx = np.where(np.isnan(dx), 0.0, dx)
    # 事前分布: 種牡馬 (自分の ≤2022 分を抜く)
    sidx = sire2idx.reindex(ds["種牡馬"].fillna("").astype(str).values[rows]).values
    pidx = ped2idx.reindex(ds["ped"].values[rows]).values
    N = len(rows)
    d_hat = np.zeros((N, K)); m_pr = np.zeros((N, K))
    Tdiag = np.diag(T)
    for a in range(0, N, 20000):
        sl = slice(a, min(a + 20000, N))
        si = sidx[sl]; pi = pidx[sl]
        A = np.zeros((sl.stop - sl.start, K, K)); b = np.zeros((sl.stop - sl.start, K))
        hs = ~np.isnan(si)
        A[hs] = As[si[hs].astype(int)]; b[hs] = bs[si[hs].astype(int)]
        hp = hs & ~np.isnan(pi)
        A[hp] -= A2[pi[hp].astype(int)]; b[hp] -= b2[pi[hp].astype(int)]
        Mi = np.linalg.inv(I_K + Oh[None, :, None] * A * Oh[None, None, :])
        C = Oh[None, :, None] * Mi * Oh[None, None, :]
        m = (C @ b[:, :, None])[:, :, 0]
        P = C + Tdiag[None]
        rhs = (th[sl] - (Sh[sl] @ m[:, :, None])[:, :, 0]) / s2
        dd = np.linalg.solve(I_K + (Sh[sl] @ P) / s2, rhs[:, :, None])[:, :, 0]
        d_hat[sl] = m + (P @ dd[:, :, None])[:, :, 0]
        m_pr[sl] = m
    contrib_k = (beta[None, :] + d_hat) * dx
    z = pd.DataFrame({"rid_s": bb["rid_s"].values, "ban": bb["ban"].values, "split": bb["split"].values,
                      "date": bb["date"].values, "n_hist": nh})
    z["z_total"] = contrib_k.sum(1)
    for k in OUT_DIMS:
        if k == "wet":
            z["z_wet"] = contrib_k[:, DIMS.index("wet_t")] + contrib_k[:, DIMS.index("wet_d")]
        else:
            z["z_" + k] = contrib_k[:, DIMS.index(k)]
    z["z_sire"] = ((beta[None, :] + m_pr) * dx).sum(1)
    z["z_dev"] = (d_hat * dx).sum(1)            # 診断用 (個体+種牡馬の偏差だけ、β 抜き)
    z["z_sdev"] = (m_pr * dx).sum(1)            # 診断用 (種牡馬の偏差だけ)
    rng = np.random.default_rng(20261114)
    perm = np.flatnonzero(ok).copy(); shuf = perm.copy(); rng.shuffle(shuf)
    d_pl = d_hat.copy(); d_pl[perm] = d_hat[shuf]
    z["z_placebo"] = ((beta[None, :] + d_pl) * dx).sum(1)
    zc = [c_ for c_ in z.columns if c_.startswith("z_")]
    z.loc[~ok, zc] = np.nan
    z.to_parquet(OUT_Z)
    cov = {sp: float(z.loc[z.split == sp, "z_total"].notna().mean()) for sp in ("valid", "test")}
    log(f"[z] {len(z):,} 行  カバー率 valid {100 * cov['valid']:.1f}% / test {100 * cov['test']:.1f}%  ({time.time() - t0:.0f}s)")
    v = z[z.split == "valid"]
    res["coverage"] = cov
    res["z_describe_valid"] = {
        "sd_z_total": float(v["z_total"].std()), "sd_z_sire": float(v["z_sire"].std()), "sd_z_dev": float(v["z_dev"].std()),
        "sd_z_sdev": float(v["z_sdev"].std()),
        "sd_within_race_z_total": float((v["z_total"] - v.groupby("rid_s")["z_total"].transform("mean")).std()),
        "sd_by_dim": {k: float(v["z_" + k].std()) for k in OUT_DIMS},
        "median_n_hist": float(v.loc[v.z_total.notna(), "n_hist"].median())}
    log("[z sd valid] " + "  ".join(f"{k}:{s:.2f}" for k, s in res["z_describe_valid"]["sd_by_dim"].items())
        + f"  total:{res['z_describe_valid']['sd_z_total']:.2f}")

    # ---- 5. valid (2023) で IDM の上下を当てるか (test は見ない)
    hist_today = ds["hist"].values[rows]
    tgt = rs[rows] - rb                                             # その走の残差 − 過去の平均残差 (β・共通曲線は除去済み)
    diag = {}
    for sp in (["valid"] if no_test else ["valid", "test"]):
        msk = ok & hist_today & (nh >= 2) & (bb["split"].values == sp)
        tt = tgt[msk]; out = {"n": int(msk.sum()), "sd_target": float(tt.std())}
        for nm, pr in (("dev(種牡馬+個体)", z["z_dev"].values[msk]), ("sire_dev(種牡馬のみ)", z["z_sdev"].values[msk]),
                       ("placebo", ((d_pl - 0) * dx).sum(1)[msk])):
            sl_ = float(np.cov(pr, tt)[0, 1] / pr.var()) if pr.var() > 0 else float("nan")
            out[nm] = {"corr": float(np.corrcoef(pr, tt)[0, 1]), "slope": sl_, "sd_pred": float(pr.std()),
                       "mse_reduction_pct": float(100 * (1 - ((tt - pr) ** 2).mean() / (tt ** 2).mean()))}
        diag[sp] = out
        log(f"[IDM 予測 {sp}] n={out['n']:,} " + "  ".join(
            f"{nm}: r={o['corr']:+.4f} slope={o['slope']:+.2f} ΔMSE={o['mse_reduction_pct']:+.3f}%" for nm, o in out.items() if isinstance(o, dict)))
    res["idm_prediction_check"] = diag

    if no_test:
        log("[--no-test] ゲートは回さない")
        return res

    # ---- 6. ゲート (test は 1 回)
    res["gate_a_total"] = gc.gate(z, ["z_total"], name="(a) z_total")
    res["gate_b_perdim"] = gc.gate(z, ["z_" + k for k in OUT_DIMS], name="(b) 軸ごと 9 列")
    res["gate_c_sire"] = gc.gate(z, ["z_sire"], name="(c) 種牡馬の事前分布だけ")
    passes = [(g, n_) for g in ("gate_a_total", "gate_b_perdim", "gate_c_sire")
              for n_, x in res[g]["gates"].items() if x["verdict"] == "PASS"]
    res["passes"] = [list(p) for p in passes]
    if passes:
        log(f"[PASS あり] {passes} → 偽薬と年別符号")
        res["gate_placebo"] = gc.gate(z, ["z_placebo"], name="偽薬 (傾きの偏差を馬の間で入れ替え)")
        zc_map = {"gate_a_total": ["z_total"], "gate_b_perdim": ["z_" + k for k in OUT_DIMS], "gate_c_sire": ["z_sire"]}
        res["per_year"] = {f"{g}:{n_}": gate_by_year(z, zc_map[g], n_) for g, n_ in passes}
        for kk, vv in res["per_year"].items():
            log(f"[年別 {kk}] {vv}")
    a = res["gate_a_total"]["gates"]
    if a["M"]["verdict"] == "PASS":
        res["verdict"] = "SIGNAL(市場外)"
    elif a["P"]["verdict"] == "PASS":
        res["verdict"] = "SIGNAL(予測)"
    elif passes:
        res["verdict"] = "要追試 ((a) の P/M は通らず、他のゲートだけ PASS)"
    else:
        res["verdict"] = "null"
    log(f"[verdict] {res['verdict']}  ({time.time() - t0:.0f}s)")
    return res


def gate_by_year(z: pd.DataFrame, zcols: list[str], gname: str, l2: float = 1e-4) -> dict:
    """gc.gate と同じ前処理・同じ fit で、test の差を年別に分ける。"""
    b = gc.load_base().merge(z[["rid_s", "ban"] + zcols].drop_duplicates(["rid_s", "ban"]), on=["rid_s", "ban"], how="left")
    for c in zcols:
        b[c] = pd.to_numeric(b[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        b[c] = b[c].fillna(b.groupby("rid_s")[c].transform("mean")).fillna(0.0)
        b[c] = b[c] - b.groupby("rid_s")[c].transform("mean")
        sd = b.loc[b.split == "valid", c].std()
        b[c] = b[c] / (sd if sd and sd > 0 else 1.0)
    bcols = {"P": ["lm7"], "M": ["lk", "lm7"], "M0": ["lk"], "A10": ["lk10", "lm7"]}[gname]
    dd = b[b["lk10"].notna()] if "lk10" in bcols else b
    xv = dd[dd.split == "valid"].reset_index(drop=True); xt = dd[dd.split == "test"].reset_index(drop=True)
    cv = pd.factorize(xv["rid_s"])[0]; ct = pd.factorize(xt["rid_s"])[0]
    wv = np.flatnonzero((xv["jyun"] == 1).values); wt = np.flatnonzero((xt["jyun"] == 1).values)
    lp = {}
    for tag, cols in (("base", bcols), ("plus", bcols + zcols)):
        w = gc._fit(xv[cols].values, cv, cv.max() + 1, wv, l2)
        lp[tag] = gc.race_logsoftmax(xt[cols].values @ w, ct, ct.max() + 1)
    diff = -(lp["plus"][wt] - lp["base"][wt])
    yr = xt["date"].astype(str).str[:4].values[wt]
    return {y_: {"n": int((yr == y_).sum()), "d_nll": float(diff[yr == y_].mean())} for y_ in sorted(set(yr))}


def posthoc_valid_only() -> dict:
    """事後の説明用 (判定には使わない)。valid=2023 だけを使い、test は見ない。
    z が単独ではどれだけ勝ち馬を当て、v7 / 市場を入れるとどれだけ重みが残るか (= どれだけ既に織り込まれているか)。"""
    z = pd.read_parquet(OUT_Z)
    b = gc.load_base().merge(z[["rid_s", "ban", "z_total", "z_dev", "z_sire"]], on=["rid_s", "ban"], how="left")
    b = b[b.split == "valid"].reset_index(drop=True)
    out = {}
    for c in ("z_total", "z_dev", "z_sire"):
        b[c] = b[c].fillna(b.groupby("rid_s")[c].transform("mean")).fillna(0.0)
        b[c] = b[c] - b.groupby("rid_s")[c].transform("mean")
        b[c] = b[c] / b[c].std()
    codes = pd.factorize(b["rid_s"])[0]; n = codes.max() + 1; w = np.flatnonzero((b["jyun"] == 1).values)
    for c in ("lm7", "lk"):
        b[c + "_c"] = b[c] - b.groupby("rid_s")[c].transform("mean")
    out["within_race_corr"] = {f"{zc}~{c}": float(np.corrcoef(b[zc], b[c + "_c"])[0, 1])
                               for zc in ("z_total", "z_dev", "z_sire") for c in ("lm7", "lk")}
    out["nll_uniform"] = float(np.log(b.groupby("rid_s")["ban"].transform("size").values[w]).mean())

    def fit0(cols):
        X = b[cols].values

        def f(wv):
            lp = gc.race_logsoftmax(X @ wv, codes, n); pr = np.exp(lp)
            g = -(X[w].sum(0) - (pr[:, None] * X).sum(0)) / n
            return -lp[w].sum() / n, g
        o = minimize(f, np.zeros(len(cols)), jac=True, method="L-BFGS-B")
        return {"nll_valid_insample": float(o.fun), "w": {c: float(x) for c, x in zip(cols, o.x)}}
    for zc in ("z_total", "z_dev", "z_sire"):
        out[zc] = {"alone": fit0([zc]), "with_v7": fit0(["lm7", zc]), "with_market_v7": fit0(["lk", "lm7", zc])}
    out["v7_alone"] = fit0(["lm7"]); out["market_v7"] = fit0(["lk", "lm7"])
    return out


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    if "--posthoc" in sys.argv:          # 事後の説明用。valid だけ。既存 JSON に追記する
        res_ = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        res_["posthoc_valid_only"] = posthoc_valid_only()
        OUT_JSON.write_text(json.dumps(res_, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps(res_["posthoc_valid_only"], ensure_ascii=False, indent=1))
        sys.exit(0)
    no_test = "--no-test" in sys.argv
    out = main(no_test=no_test)
    if not no_test:
        OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        log(f"→ {OUT_JSON}")
