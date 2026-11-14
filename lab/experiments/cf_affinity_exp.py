# -*- coding: utf-8 -*-
"""協調フィルタリング相性 (「この馬に似た馬は、この条件で走った」) — 2026-11-14

理論
----
強さ (1 次元) を完全に知っても v7 に届かない (PREDICTABILITY_CEILING_20261114)。足りないのは
「馬 × レース条件」の相性 (多次元) だが、同条件の出走数は 1 頭あたり数走で疎。協調フィルタリングは
全馬の「条件別の出来・不出来」から条件セルの潜在因子を学び、各馬の数走からその馬の潜在ベクトルを
推定して、走ったことのないセルの出来まで予測する。

事前登録 (test を見る前に固定。以下の定義・判定は結果を見てから変えない)
-----------------------------------------------------------------------
データ: JRDB SED の平地 (td∈{1,2})。履歴に使う走は ijou==0 かつ idm 非欠損 (= 有効走)。
目的変数: y = idm (その走の出来。走った後に確定する値なので、履歴としてのみ使う)。
セル: 場 (racekey[:2]) × 芝ダ (td) × 距離 (m, そのまま)。学習期間の有効走が MIN_CELL=800 以上のセルだけ
  因子を持つ (それ未満は v=0 の中立セル。今日のレースが中立セルなら z は欠損)。
モデル (学習は 〜2022-12-31 の有効走のみ):
  y = a_馬 + g(月齢 3 か月刻み) + b_セル + u_馬 · v_セル + ε
  1) a, g, b を後退代入 (backfitting) で推定。e = y − g − b。
  2) e = a_馬 + u_馬·v_セル を ALS で解く (u にだけ ridge λ、V は全体のフロベニウスノルムを固定、
     セル数加重で中心化)。乱数 seed 固定。
ハイパーパラメータ: K∈{3,5,8} × λ∈{10,30,100,300} を「〜2019 で学習 → 2020-22 の走を as-of fold-in で予測」
  の内側検証で選ぶ (指標: 過去有効走 ≥3 の走で corr(z, e_今日 − 過去 e の平均) が最大)。選んだ (K, λ) で
  〜2022 を学習し直す。valid(2023)/test(2024-25) はこの選択に一切使わない。
fold-in (valid/test の各出走馬、レース日 D):
  その馬の ymd < D の有効走だけ (2023 年以降の走も D より前なら使う。g, b, V は 〜2022 で固定) で
  min Σ (e_i − a − u·v_{c_i})² + λ|u|² + 1e-3·a²  を解き、z = u · v_{今日のセル}。
  過去有効走が 0 の馬は欠損 (gate がレース平均で埋める)。セルが 1 種類しかない馬は自動的に z≈0。
変種 (3 つ、すべて報告):
  V1 z_cf     : 上の z (主検定)
  V2 z_cf_rel : z × 信頼度。信頼度 = 1 − λ·q'(A⁻¹)_uu q / q'q  (q = 今日のセル因子、A = ridge の正規方程式。
                事後分散/事前分散を 1 から引いたもの、0〜1)
  V3 z_cfg    : セル因子に馬場状態の因子を加える。q = v_セル + w_{芝ダ×馬場(稍 / 重・不)} (良は基準 0)。
                馬場状態は master_lite (2013-) から。学習は 2013-2022 の馬場状態が分かる有効走。
                fold-in で馬場状態不明の走は良扱い。(K, λ) は V1 と同じものを使う (選び直さない)。
比較用の基準 (変種に数えない):
  z0 明示版 : (今日とまったく同じセルの過去 e の平均 − 過去 e 全体の平均) × n_c/(n_c+3)。同セル経験なしは 0。
判定 (gate_common.gate の 4 ゲート。valid=2023 で重み fit、test=2024-25 を 1 回だけ評価):
  SIGNAL(市場外) = V1 の M ゲートが PASS (CI95 上端 < 0) かつ 2024・2025 の両年で ΔNLL < 0
                   かつ プラセボ (レース内シャッフル / 別セルの因子で予測) が PASS しない。
  SIGNAL(予測)   = V1 の P ゲートで同じ条件。
  V2/V3 だけが PASS し V1 が none の場合は「多重比較のため要追試」とし SIGNAL としない。
  それ以外は none (null)。WORSE は WORSE と書く。
記述統計 (判定に使わない): test の有効走で corr(z, e_今日 − 過去平均)、z0 との比較、同セル未経験の割合。

as-of の点検
------------
- 因子 V・g・b・W は 〜2022 の走だけで学習。valid/test 馬の u は fold-in で毎回 D より前の走から作り直す
  (ALS で学習した u は使わない)。
- 累積和は「その馬の、行より前の行」だけ (同一馬の同日 2 走は SED に無いことを確認済み)。
- 今日のセル (場・芝ダ・距離)、馬場状態は発走前に分かる。今日の idm・着順は使わない。

事後追加 (final の結果を見た後に追加。探索扱いで判定には使わない)
  内側検証 (〜2022) で、事前登録の選択規則 (全体の corr 最大) が選んだ K=8, λ=10 は「同セル未経験の走」の
  corr が格子の中で最低 (+0.019) だった。理論の中身 (未経験セルへの転移) が最も強い K=3, λ=300
  (未経験セル corr +0.068、これも 〜2022 だけで分かる) の z_cf を同じゲートに 1 回だけ通す (--stage posthoc)。

再実行: worktree root から
  python lab/experiments/cf_affinity_exp.py --stage inner   (〜2022 だけ。ハイパーパラメータ選択)
  python lab/experiments/cf_affinity_exp.py --stage final   (因子学習 → fold-in → gate)
出力: reports/cf_affinity_exp.json (集計値のみ)、reports/cf_affinity_z.parquet (作業用、馬別 z、外に出さない)
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate_common as gc  # noqa: E402

WT = Path(__file__).resolve().parents[2]
REP = WT / "reports"
OUT = REP / "cf_affinity_exp.json"
INNER = REP / "cf_affinity_inner.json"
ZPQ = REP / "cf_affinity_z.parquet"

MIN_CELL = 800
LAM_A = 1e-3
N_ITER = 20
SEED = 20261114
GRID_K = [3, 5, 8]
GRID_LAM = [10.0, 30.0, 100.0, 300.0]
Z0_KAPPA = 3.0
N_AGEB = 25


def log(m):
    print(m, flush=True)


# ------------------------------------------------------------------ data
def load_runs() -> pd.DataFrame:
    s = gc.load_sed(["racekey", "ymd", "ban", "ped", "dist", "td", "ijou", "idm"])
    s = s[s.td.isin([1, 2])].copy()
    s["ok"] = (s.ijou == 0) & s.idm.notna()
    s["cell_key"] = s.racekey.str[:2] + "_" + s.td.astype(str) + "_" + s.dist.astype(str)
    yy = pd.to_numeric(s.ped.str[:2], errors="coerce").fillna(0).astype(int)
    by = np.where(yy > 50, 1900 + yy, 2000 + yy)
    age_m = (s.ymd // 10000 - by) * 12 + (s.ymd // 100 % 100)
    s["ageb"] = np.clip((age_m - 24) // 3, 0, N_AGEB - 1).astype(int)
    lite = pd.read_parquet(gc.LITE_PQ, columns=["レースID(新/馬番無)", "馬番", "馬場状態"])
    lite["rid_s"] = lite["レースID(新/馬番無)"].astype(str).str[:16]
    lite["ban"] = pd.to_numeric(lite["馬番"], errors="coerce")
    lite = lite.dropna(subset=["ban"]).drop_duplicates(["rid_s", "ban"])
    s = s.merge(lite[["rid_s", "ban", "馬場状態"]], on=["rid_s", "ban"], how="left")
    gmap = {"良": 0, "稍": 1, "重": 2, "不": 2}
    gcls = s["馬場状態"].map(gmap)
    s["going_known"] = gcls.notna()
    gcls = gcls.fillna(0).astype(int)
    # 0 = 良/不明 (基準)、1 = 芝稍、2 = 芝重不、3 = ダ稍、4 = ダ重不
    s["gidx"] = np.where(gcls == 0, 0, gcls + (s.td.values - 1) * 2).astype(int)
    s = s.sort_values(["ped", "ymd"], kind="stable").reset_index(drop=True)
    assert not s.duplicated(["ped", "ymd"]).any()
    return s


# ------------------------------------------------------------------ factor learning (train only)
def fit_factors(tr: pd.DataFrame, K: int, lam: float, use_going: bool, verbose: bool = False) -> dict:
    """tr: 学習期間の有効走 (ped 順に並んでいる)。戻り値: セル一覧・g・b・V (・W・bg)"""
    rng = np.random.default_rng(SEED)
    cnt = tr.cell_key.value_counts()
    cells = sorted(cnt[cnt >= MIN_CELL].index)
    cmap = {k: i for i, k in enumerate(cells)}
    C = len(cells)
    c = tr.cell_key.map(cmap).fillna(C).astype(int).values      # C = 中立セル
    h = pd.factorize(tr.ped)[0]
    assert (np.diff(h) >= 0).all()
    H = h.max() + 1
    hstart = np.flatnonzero(np.r_[1, np.diff(h)])
    ab = tr.ageb.values
    gi = tr.gidx.values if use_going else np.zeros(len(tr), dtype=int)
    y = tr.idm.values.astype(float)
    n = len(y)
    # 1) backfitting: a_h, g(age), b(cell), bg(going)
    a = np.zeros(H); g = np.zeros(N_AGEB); b = np.zeros(C + 1); bg = np.zeros(5)
    mu = y.mean()
    for _ in range(25):
        r = y - mu - g[ab] - b[c] - bg[gi]
        a = np.bincount(h, r, H) / np.bincount(h, minlength=H)
        r = y - mu - a[h] - b[c] - bg[gi]
        g = np.bincount(ab, r, N_AGEB) / np.maximum(np.bincount(ab, minlength=N_AGEB), 1)
        r = y - mu - a[h] - g[ab] - bg[gi]
        b = np.bincount(c, r, C + 1) / np.maximum(np.bincount(c, minlength=C + 1), 1)
        if use_going:
            r = y - mu - a[h] - g[ab] - b[c]
            bg = np.bincount(gi, r, 5) / np.maximum(np.bincount(gi, minlength=5), 1)
            bg[0] = 0.0
    e = y - mu - g[ab] - b[c] - bg[gi]
    # 2) ALS
    d = K + 1
    V = rng.normal(0, 1, (C + 1, K)); V[C] = 0
    W = np.zeros((5, K))
    ccount = np.bincount(c, minlength=C + 1)[:C].astype(float)
    order_c = np.argsort(c, kind="stable"); cs = c[order_c]
    keep = cs < C
    order_c = order_c[keep]; cs = cs[keep]
    cstart = np.flatnonzero(np.r_[1, np.diff(cs)])
    assert len(cstart) == C
    reg = np.diag(np.r_[LAM_A, np.full(K, lam)])
    eyeK = np.eye(K)

    def normalize():
        nonlocal V, W
        V[:C] -= (V[:C] * ccount[:, None]).sum(0) / ccount.sum()
        f = np.sqrt(K * C) / np.linalg.norm(V[:C])
        V[:C] *= f; W *= f

    normalize()
    hist = []
    for it in range(N_ITER):
        Q = V[c] + W[gi]
        X = np.concatenate([np.ones((n, 1)), Q], axis=1)
        A = np.add.reduceat(X[:, :, None] * X[:, None, :], hstart, axis=0) + reg
        rhs = np.add.reduceat(X * e[:, None], hstart, axis=0)
        th = np.linalg.solve(A, rhs[:, :, None])[:, :, 0]
        ah = th[:, 0]; U = th[:, 1:]
        Uh = U[h]
        r = e - ah[h]
        rmse = float(np.sqrt(np.mean((r - (Uh * Q).sum(1)) ** 2)))
        hist.append(rmse)
        if verbose and it % 5 == 0:
            log(f"    it{it:2d} rmse {rmse:.4f}  (a のみ {np.sqrt(np.mean(r ** 2)):.4f})  |U|rms {np.sqrt((U ** 2).mean()):.3f}")
        # v-step
        rv = r - (Uh * W[gi]).sum(1)
        Uo = Uh[order_c]
        Ac = np.add.reduceat(Uo[:, :, None] * Uo[:, None, :], cstart, axis=0) + 1.0 * eyeK
        bc = np.add.reduceat(Uo * rv[order_c][:, None], cstart, axis=0)
        V[:C] = np.linalg.solve(Ac, bc[:, :, None])[:, :, 0]
        if use_going:
            rw = r - (Uh * V[c]).sum(1)
            for k in range(1, 5):
                m = gi == k
                W[k] = np.linalg.solve(Uh[m].T @ Uh[m] + eyeK, Uh[m].T @ rw[m])
        normalize()
    return {"cells": cells, "cmap": cmap, "C": C, "mu": float(mu), "g": g, "b": b, "bg": bg, "V": V, "W": W,
            "K": K, "lam": lam, "use_going": use_going, "rmse_hist": hist, "n_train": int(n), "n_horses": int(H),
            "resid_sd_a_only": float(np.sqrt(np.mean((e - a[h]) ** 2)))}


# ------------------------------------------------------------------ fold-in (as-of)
def fold_in(runs: pd.DataFrame, F: dict, d_from: int, d_to: int, placebo_cell_seed: int | None = None) -> pd.DataFrame:
    """runs (ped, ymd 順の全平地走) のうち [d_from, d_to] の行を対象に、その馬の ymd より前の有効走だけで z を作る。"""
    K, lam, C = F["K"], F["lam"], F["C"]
    tg_peds = runs.loc[(runs.ymd >= d_from) & (runs.ymd <= d_to), "ped"].unique()
    x = runs[runs.ped.isin(tg_peds) & (runs.ymd <= d_to)].reset_index(drop=True)
    n = len(x)
    c = x.cell_key.map(F["cmap"]).fillna(C).astype(int).values
    gi = x.gidx.values if F["use_going"] else np.zeros(n, dtype=int)
    ok = x.ok.values
    e = np.where(ok, x.idm.fillna(0).values - F["mu"] - F["g"][x.ageb.values] - F["b"][c] - F["bg"][gi], 0.0)
    Q = F["V"][c] + F["W"][gi]
    X = np.concatenate([np.ones((n, 1)), Q], axis=1) * ok[:, None]
    d = K + 1
    CS = np.zeros((n + 1, d, d)); np.cumsum(X[:, :, None] * X[:, None, :], axis=0, out=CS[1:])
    CR = np.zeros((n + 1, d)); np.cumsum(X * e[:, None], axis=0, out=CR[1:])
    CN = np.r_[0, np.cumsum(ok)]
    h = pd.factorize(x.ped)[0]
    first = np.flatnonzero(np.r_[1, np.diff(h)])
    start = first[h]                                    # その馬の最初の行
    idx = np.flatnonzero((x.ymd.values >= d_from) & (x.ymd.values <= d_to))
    S = CS[idx] - CS[start[idx]]                        # 行 idx より前 (同馬) の有効走だけ
    R = CR[idx] - CR[start[idx]]
    nh = CN[idx] - CN[start[idx]]
    del CS
    A = S + np.diag(np.r_[LAM_A, np.full(K, lam)])
    th = np.linalg.solve(A, R[:, :, None])[:, :, 0]
    ct = c[idx]
    if placebo_cell_seed is not None:                   # プラセボ: 今日のセルを別の (因子を持つ) セルに差し替える
        rng = np.random.default_rng(placebo_cell_seed)
        ct = np.where(ct < C, (ct + rng.integers(1, C, len(ct))) % C, ct)
    q = F["V"][ct] + F["W"][gi[idx]]
    z = (th[:, 1:] * q).sum(1)
    Ainv = np.linalg.inv(A)[:, 1:, 1:]
    qq = (q * q).sum(1)
    rel = 1.0 - lam * np.einsum("ni,nij,nj->n", q, Ainv, q) / np.where(qq > 0, qq, 1.0)
    rel = np.clip(rel, 0.0, 1.0)
    bad = (nh == 0) | (ct >= C)
    z = np.where(bad, np.nan, z); zrel = np.where(bad, np.nan, z * rel)
    mean_past = np.where(nh > 0, R[:, 0] / np.maximum(nh, 1), np.nan)
    out = x.loc[idx, ["rid_s", "ban", "ped", "ymd", "ok", "cell_key"]].copy()
    out["z"] = z; out["zrel"] = zrel; out["rel"] = np.where(bad, np.nan, rel); out["n_hist"] = nh
    out["e_today"] = np.where(ok[idx], e[idx], np.nan)
    out["mean_past"] = mean_past
    # z0: 同一セル (生のセルキー) の過去平均 − 過去全体の平均、n_c/(n_c+κ) で縮小
    grp = [x.ped.values, x.cell_key.values]
    okf = ok.astype(float)
    nc = (pd.Series(okf).groupby(grp).cumsum().values - okf)[idx]
    sc = (pd.Series(e).groupby(grp).cumsum().values - e)[idx]
    z0 = np.where(nc > 0, (sc / np.maximum(nc, 1) - mean_past) * nc / (nc + Z0_KAPPA), 0.0)
    out["z0"] = np.where(nh == 0, np.nan, z0); out["n_cell"] = nc
    return out.reset_index(drop=True)


def idm_diag(fo: pd.DataFrame, zc: str = "z") -> dict:
    """記述: 有効走・過去有効走 ≥3 で、今日の出来の上振れ t = e_今日 − 過去平均 を z がどれだけ当てるか"""
    d = fo[fo.ok & (fo.n_hist >= 3) & fo[zc].notna()].copy()
    t = (d.e_today - d.mean_past).values
    out = {"n": int(len(d)), "corr_z_t": float(np.corrcoef(d[zc], t)[0, 1]), "corr_z0_t": float(np.corrcoef(d.z0, t)[0, 1]),
           "sd_z": float(d[zc].std()), "sd_z0": float(d.z0.std()), "sd_t": float(t.std()),
           "slope_t_on_z": float(np.polyfit(d[zc], t, 1)[0]),
           "share_never_ran_cell": float((d.n_cell == 0).mean())}
    X = np.c_[np.ones(len(d)), d[zc].values, d.z0.values]
    bh = np.linalg.lstsq(X, t, rcond=None)[0]
    out["ols_t_on_z_and_z0"] = {"z": float(bh[1]), "z0": float(bh[2])}
    nv = d[d.n_cell == 0]
    if len(nv) > 100:
        out["corr_z_t_never_ran_cell"] = float(np.corrcoef(nv[zc], (nv.e_today - nv.mean_past))[0, 1])
        out["n_never_ran_cell"] = int(len(nv))
    return out


# ------------------------------------------------------------------ gate の年別 (gate_common と同じ fit)
def gate_by_year(zdf: pd.DataFrame, zcol: str) -> dict:
    b = gc.load_base().merge(zdf[["rid_s", "ban", zcol]].drop_duplicates(["rid_s", "ban"]), on=["rid_s", "ban"], how="left")
    b[zcol] = pd.to_numeric(b[zcol], errors="coerce")
    b[zcol] = b[zcol].fillna(b.groupby("rid_s")[zcol].transform("mean")).fillna(0.0)
    b[zcol] = b[zcol] - b.groupby("rid_s")[zcol].transform("mean")
    sd = b.loc[b.split == "valid", zcol].std()
    b[zcol] = b[zcol] / (sd if sd and sd > 0 else 1.0)
    res = {}
    for gname, bcols in {"P": ["lm7"], "M": ["lk", "lm7"]}.items():
        xv = b[b.split == "valid"].reset_index(drop=True); xt = b[b.split == "test"].reset_index(drop=True)
        cv = pd.factorize(xv.rid_s)[0]; ct = pd.factorize(xt.rid_s)[0]
        wv = np.flatnonzero((xv.jyun == 1).values); wt = np.flatnonzero((xt.jyun == 1).values)
        lp = {}
        for tag, cols in (("base", bcols), ("plus", bcols + [zcol])):
            w = gc._fit(xv[cols].values, cv, cv.max() + 1, wv, 1e-4)
            lp[tag] = gc.race_logsoftmax(xt[cols].values @ w, ct, ct.max() + 1)
        diff = -(lp["plus"][wt] - lp["base"][wt])
        yr = xt.date.astype(str).str[:4].values[wt]
        res[gname] = {y: {"n": int((yr == y).sum()), "d_nll": float(diff[yr == y].mean())} for y in ("2024", "2025")}
        res[gname]["all"] = float(diff.mean())
    return res


def shuffle_within_race(zdf: pd.DataFrame, zcol: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    b = gc.load_base()[["rid_s", "ban"]].merge(zdf[["rid_s", "ban", zcol]].drop_duplicates(["rid_s", "ban"]), on=["rid_s", "ban"], how="left")
    b["_r"] = rng.random(len(b))
    perm = b.sort_values(["rid_s", "_r"])[zcol].values
    b = b.sort_values(["rid_s", "ban"]).copy(); b[zcol] = perm
    return b[["rid_s", "ban", zcol]]


# ------------------------------------------------------------------ stages
def stage_inner(runs):
    """〜2022 だけを使うハイパーパラメータ選択 (〜2019 学習 → 2020-22 を as-of 予測)"""
    t0 = time.time()
    tr = runs[runs.ok & (runs.ymd <= 20191231)]
    res = []
    for K in GRID_K:
        for lam in GRID_LAM:
            F = fit_factors(tr, K, lam, use_going=False)
            fo = fold_in(runs, F, 20200101, 20221231)
            dg = idm_diag(fo)
            res.append({"K": K, "lam": lam, "rmse_last": F["rmse_hist"][-1], **dg})
            log(f"[inner] K={K} λ={lam:5.0f}  corr(z,t) {dg['corr_z_t']:+.4f}  corr(z0,t) {dg['corr_z0_t']:+.4f}"
                f"  未経験セル corr {dg.get('corr_z_t_never_ran_cell', float('nan')):+.4f}  sd_z {dg['sd_z']:.3f}  n={dg['n']:,}  ({time.time() - t0:.0f}s)")
    best = max(res, key=lambda r: r["corr_z_t"])
    out = {"grid": res, "best": {"K": best["K"], "lam": best["lam"]}}
    INNER.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"[inner] 採用 K={best['K']} λ={best['lam']} → {INNER}")
    return out


def stage_final(runs):
    t0 = time.time()
    inner = json.loads(INNER.read_text(encoding="utf-8"))
    K, lam = inner["best"]["K"], inner["best"]["lam"]
    out = {"prereg": {"MIN_CELL": MIN_CELL, "LAM_A": LAM_A, "N_ITER": N_ITER, "SEED": SEED, "Z0_KAPPA": Z0_KAPPA,
                      "grid_K": GRID_K, "grid_lam": GRID_LAM, "chosen": {"K": K, "lam": lam}},
           "inner_grid": inner["grid"]}
    tr = runs[runs.ok & (runs.ymd <= 20221231)]
    F1 = fit_factors(tr, K, lam, use_going=False, verbose=True)
    log(f"[fit V1] セル {F1['C']}  学習 {F1['n_train']:,} 走 / {F1['n_horses']:,} 頭  rmse {F1['rmse_hist'][0]:.4f}→{F1['rmse_hist'][-1]:.4f} ({time.time() - t0:.0f}s)")
    tr3 = tr[tr.going_known & (tr.ymd >= 20130101)]
    F3 = fit_factors(tr3, K, lam, use_going=True, verbose=True)
    log(f"[fit V3] セル {F3['C']}  学習 {F3['n_train']:,} 走  rmse {F3['rmse_hist'][0]:.4f}→{F3['rmse_hist'][-1]:.4f} ({time.time() - t0:.0f}s)")
    out["fit"] = {"V1": {k: F1[k] for k in ("C", "n_train", "n_horses", "resid_sd_a_only")} | {"rmse_first": F1["rmse_hist"][0], "rmse_last": F1["rmse_hist"][-1]},
                  "V3": {k: F3[k] for k in ("C", "n_train", "n_horses", "resid_sd_a_only")} | {"rmse_first": F3["rmse_hist"][0], "rmse_last": F3["rmse_hist"][-1],
                         "going_factor_norm": [float(np.linalg.norm(w)) for w in F3["W"]], "cell_factor_rms_norm": float(np.sqrt((F3["V"][:F3["C"]] ** 2).sum(1).mean()))}}
    # セル因子の構造 (集計): 第 1〜3 主成分の寄与と、芝/ダ・距離との相関
    V = F1["V"][:F1["C"]]
    sv = np.linalg.svd(V - V.mean(0), compute_uv=False)
    out["fit"]["V1"]["factor_sv_share"] = [float(x) for x in (sv ** 2 / (sv ** 2).sum())]
    cd = pd.DataFrame({"cell": F1["cells"]}); cd[["venue", "td", "dist"]] = cd.cell.str.split("_", expand=True)
    Uc, Sc, _ = np.linalg.svd(V - V.mean(0), full_matrices=False)
    for j in range(min(3, K)):
        pc = Uc[:, j] * Sc[j]
        out["fit"]["V1"][f"pc{j + 1}_corr_dirt"] = float(np.corrcoef(pc, (cd.td == "2").astype(float))[0, 1])
        out["fit"]["V1"][f"pc{j + 1}_corr_dist"] = float(np.corrcoef(pc, cd.dist.astype(float))[0, 1])

    f1 = fold_in(runs, F1, 20230101, 20251231)
    f3 = fold_in(runs, F3, 20230101, 20251231)
    fp = fold_in(runs, F1, 20230101, 20251231, placebo_cell_seed=7)
    log(f"[fold-in] {len(f1):,} 行 ({time.time() - t0:.0f}s)")
    zdf = f1[["rid_s", "ban", "ymd", "n_hist", "n_cell", "rel"]].copy()
    zdf["z_cf"] = f1.z; zdf["z_cf_rel"] = f1.zrel; zdf["z0"] = f1.z0
    zdf = zdf.merge(f3[["rid_s", "ban", "z"]].rename(columns={"z": "z_cfg"}), on=["rid_s", "ban"], how="left")
    zdf = zdf.merge(fp[["rid_s", "ban", "z"]].rename(columns={"z": "z_placebo_cell"}), on=["rid_s", "ban"], how="left")
    zdf.to_parquet(ZPQ)
    base = gc.load_base()[["rid_s", "ban", "split", "lm7", "lk"]].merge(zdf, on=["rid_s", "ban"], how="left")
    te = base[base.split == "test"]
    cw = lambda a, c: float(pd.DataFrame({"a": a - a.groupby(te.rid_s).transform("mean"), "c": c - c.groupby(te.rid_s).transform("mean")}).dropna().corr().iloc[0, 1])
    out["coverage_test"] = {"has_history": float(te.z_cf.notna().mean()), "nonzero_z": float((te.z_cf.abs() > 1e-6).mean()),
                            "never_ran_today_cell_given_history": float((te.loc[te.n_hist > 0, "n_cell"] == 0).mean()),
                            "mean_n_hist": float(te.n_hist.mean()), "mean_rel": float(te.rel.mean()),
                            "sd_z_cf": float(te.z_cf.std()), "sd_z0": float(te.z0.std()),
                            "corr_within_race": {"z_cf~lm7": cw(te.z_cf, te.lm7), "z_cf~lk": cw(te.z_cf, te.lk), "z_cf~z0": cw(te.z_cf, te.z0),
                                                 "z_cf~z_cfg": cw(te.z_cf, te.z_cfg), "z_cf~z_placebo_cell": cw(te.z_cf, te.z_placebo_cell)}}
    log(f"[coverage] {out['coverage_test']}")
    # 記述: IDM の上振れを当てるか (valid / test 別)
    out["idm_diag"] = {}
    for nm, fo in (("V1", f1), ("V3", f3), ("placebo_cell", fp)):
        for sp, (a, b_) in {"valid2023": (20230101, 20231231), "test2024_25": (20240101, 20251231)}.items():
            out["idm_diag"][f"{nm}_{sp}"] = idm_diag(fo[(fo.ymd >= a) & (fo.ymd <= b_)])
    log(f"[idm_diag] {json.dumps(out['idm_diag'], ensure_ascii=False)}")
    # ゲート
    out["gates"] = {}
    for nm, col in (("V1_z_cf", "z_cf"), ("V2_z_cf_rel", "z_cf_rel"), ("V3_z_cfg", "z_cfg"), ("baseline_z0", "z0")):
        out["gates"][nm] = gc.gate(zdf, [col], name=nm)
        out["gates"][nm]["by_year"] = gate_by_year(zdf, col)
    # プラセボ (判定の一部として事前登録)
    out["placebo"] = {"cell_mismatch": gc.gate(zdf, ["z_placebo_cell"], name="placebo_別セル"),
                      "shuffle_within_race": gc.gate(shuffle_within_race(zdf, "z_cf", 11), ["z_cf"], name="placebo_レース内シャッフル")}
    g1 = out["gates"]["V1_z_cf"]
    def sig(gn):
        g = g1["gates"][gn]; by = g1["by_year"][gn]
        ok = g["verdict"] == "PASS" and by["2024"]["d_nll"] < 0 and by["2025"]["d_nll"] < 0
        pl = any(out["placebo"][k]["gates"][gn]["verdict"] == "PASS" for k in out["placebo"])
        return bool(ok and not pl)
    sec = {nm: {gn: out["gates"][nm]["gates"][gn]["verdict"] for gn in ("P", "M", "M0", "A10")} for nm in out["gates"]}
    out["verdict"] = {"SIGNAL_market(M)": sig("M"), "SIGNAL_pred(P)": sig("P"), "verdicts": sec}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"[verdict] {json.dumps(out['verdict'], ensure_ascii=False)}")
    log(f"→ {OUT} ({time.time() - t0:.0f}s)")


def stage_posthoc(runs):
    """事後追加・探索扱い: K=3, λ=300 (内側検証で未経験セルの corr が最大) の z_cf"""
    out = json.loads(OUT.read_text(encoding="utf-8"))
    F = fit_factors(runs[runs.ok & (runs.ymd <= 20221231)], 3, 300.0, use_going=False, verbose=True)
    fo = fold_in(runs, F, 20230101, 20251231)
    zdf = fo[["rid_s", "ban"]].copy(); zdf["z_cf_k3"] = fo.z
    V = F["V"][:F["C"]]
    cd = pd.DataFrame({"cell": F["cells"]}); cd[["venue", "td", "dist"]] = cd.cell.str.split("_", expand=True)
    Uc, Sc, _ = np.linalg.svd(V - V.mean(0), full_matrices=False)
    ph = {"K": 3, "lam": 300.0, "note": "事後追加・探索扱い・判定に使わない",
          "factor_sv_share": [float(x) for x in Sc ** 2 / (Sc ** 2).sum()],
          "pc_corr_dirt": [float(np.corrcoef(Uc[:, j], (cd.td == "2").astype(float))[0, 1]) for j in range(3)],
          "pc_corr_dist": [float(np.corrcoef(Uc[:, j], cd.dist.astype(float))[0, 1]) for j in range(3)],
          "idm_diag_valid2023": idm_diag(fo[fo.ymd <= 20231231]), "idm_diag_test2024_25": idm_diag(fo[fo.ymd >= 20240101]),
          "gate": gc.gate(zdf, ["z_cf_k3"], name="posthoc_K3_λ300")}
    ph["gate"]["by_year"] = gate_by_year(zdf, "z_cf_k3")
    out["posthoc"] = ph
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    log(json.dumps({k: v for k, v in ph.items() if k != "gate"}, ensure_ascii=False))
    log(json.dumps(ph["gate"]["by_year"]))


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    stage = sys.argv[sys.argv.index("--stage") + 1] if "--stage" in sys.argv else "inner"
    R = load_runs()
    log(f"[runs] 平地 {len(R):,} 行  有効 {int(R.ok.sum()):,}  馬場状態あり {100 * R.going_known.mean():.1f}%")
    if stage == "inner":
        stage_inner(R)
    elif stage == "posthoc":
        stage_posthoc(R)
    else:
        stage_final(R)
