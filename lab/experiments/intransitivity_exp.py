# -*- coding: utf-8 -*-
"""非推移性 (競馬に「じゃんけん」はあるか) — blade–chest / 反対称双線形の直接検定  2026-11-14

理論
----
Plackett–Luce・レーティング・v7 はどれも「強さは 1 本の物差し」を仮定する (A>B, B>C なら A>C)。
脚質などの相性があれば勝敗は非推移になりうる。blade–chest モデル (Chen & Joachims 2016):
    logit P(i が j に先着) = s_i − s_j + blade_i·chest_j − blade_j·chest_i
属性ベクトル x で書けば反対称双線形 x_i^T A x_j (A = −A^T)。v7 は 1 頭ずつ採点し相手を見ないので、
相性があれば「今日の相手全員に対する相性の合計」は v7 の外にある情報になる。

事前登録 (test=2024-25 を見る前に固定。--dev で test を一切計算しない開発実行ができる)
-----------------------------------------------------------------------------------
データ: JRDB SED の平地 (td∈{1,2})。対象レース 2013-01〜2025-12。履歴は 2011-01〜、当該レース日より
  前 (ymd が厳密に小さい) の平地・完走 (ijou==0, jyun>0) の走だけ (merge_asof, allow_exact_matches=False)。
属性 x (d=8、すべて発走前に分かる as-of 値):
  epos   直近≤5 走の序盤相対位置の平均 (最初に通過順が付いたコーナーの (順位−1)/(頭数−1)、0=先頭)
  ten    直近≤5 走のテン指数の平均          agari  直近≤5 走の上がり指数の平均
  bw     直近の完走時馬体重                  age    レース年 − 生年 (血統登録番号の上 2 桁)
  female 牝=1 (master_lite の性別)           dprof  (直近≤5 走の平均距離 − 今回距離)/100
  cons   直近≤5 走の IDM の標準偏差 (2 走以上)
  標準化は 2 通り: G = 2013-22 の全体平均/sd (レース内中心化しない)、R = レース内 z 化 (出走馬の平均0)。
  欠損は標準化後 0。
推移的な強さ s: (a) IDM 版 = 直近≤5 走 IDM の平均、(b) 市場版 = −log(確定単勝オッズ)。
ペア: 同一レースの「完走・IDM 履歴あり・単勝オッズ>0」の馬の全ての組 (同着は除く)、向きは乱数。
ペアモデル (ロジスティック、切片なし):
  T   : a(s_i−s_j) + w·(x_i−x_j)                    … 推移的 (属性の主効果は強さの補正として許す)
  BIL : T + x_i^T A x_j (A 反対称、28 係数)          … 反対称双線形
  BC  : T + tanh(Bx_i)·tanh(Cx_j) − tanh(Bx_j)·tanh(Cx_i) (k=4) … 非線形 blade–chest
  学習 2013-22。L2 (相性項のみ) は λ∈{1e-5,1e-4,1e-3,1e-2} を 2013-20 学習 / 2021-22 内部検証で選び、
  2013-22 で再 fit。BC は seed 0,1 のうち内部検証が良い方。2023 は選択に使わない。
ペア検定 (「非推移性はあるか」の直接検定):
  1 ペアあたり log-loss の利得 = LL(T) − LL(T+相性)。in-sample (2013-22)、2023 (OOS)、2024-25 (test)。
  CI95 は開催日クラスタ bootstrap (B=2000)。判定: 2023 の CI 下端 > 0 なら「ペア水準で非推移性あり」。
  IDM 版と市場版 (= 市場は相性を織り込み済みか) の両方を報告。
  注: レース内中心化した属性 (R) の双線形では Σ_{j≠i} x_i^T A x_j = x_i^T A (Σ_j x_j) = 0 が恒等的に成り立ち、
  レース水準の z は作れない (ペア検定だけ診断として報告)。
レース水準の z (ゲートに掛けるのは下の 3 本だけ):
  z_i = (1/n) Σ_{j≠i} m(i,j)   (n = 単勝オッズのある出走頭数、m = 学習済みの相性項のみ。T の部分は入れない)
  V1 bil_idm : BIL・属性 G・強さ IDM 版     (= x_i^T A x̄_field)
  V2 bc_idm  : BC ・属性 R・強さ IDM 版
  V3 bil_mkt : BIL・属性 G・強さ 市場版     (市場が織り込んでいない相性)
  障害・SED に無い行は NaN (gate がレース平均=0 で埋める)。
ゲート判定: gate_common の 4 ゲート。P が PASS → SIGNAL(予測)、M が PASS → SIGNAL(市場外)。3 本とも報告。
  PASS は (i) 2024/2025 の年別で同符号、(ii) プラセボ (相手の場ベクトルを同年の別レースと入れ替え) が
  PASS しない、の両方を満たして初めて SIGNAL と呼ぶ。3 本の多重性も併記。どれも満たさなければ null。
三つ巴 (モデルを使わない統計、2013-2023 のみ):
  2 回以上対戦した 2 頭の「勝ち越し」を向きとし (五分は辺なし)、3 辺とも向きのある三つ組のうち
  循環 (A>B>C>A) の割合を数える。期待値は推移モデル u = a·s + Gumbel でレース全体の着順を 200 回
  再生成した分布 (強さ IDM 版 / 市場版)。観測の循環割合が帰無分布の 97.5% 点を超えれば非推移の証拠。
  副: 全勝 (2 戦以上で全部先着) の辺だけで同じ統計。
診断 (ゲートではない): gate と同じ土俵のペア版 — 2023 で a·Δlm7 + b·Δlk (+ c·m_ij) を fit、2024-25 で評価。

事後追加 (test を開けた後に追加。探索扱いで判定には使わない。--posthoc で単独実行し JSON の "posthoc" に追記)
  ペア水準の利得は「相性」か「ばらつきの違い」か: 属性 R・両方の強さで
    HET    : a(s_i−s_j)(1 + v·(x_i+x_j)) + w·(x_i−x_j)   … 強さの差の効き方がペアの属性で変わる (不等分散)
    HETBIL : HET + x_i^T A x_j (λ は BIL で選んだ値)
  を 2013-22 で fit し、利得 HET−T と HETBIL−HET を出す。HETBIL−HET が BIL−T と同じだけ残れば不等分散では説明できない。

再実行: worktree root から  python lab/experiments/intransitivity_exp.py [--dev]
出力: reports/intransitivity_exp.json (集計値のみ)、reports/intr_z.parquet (作業用、外へ出さない)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate_common as gc  # noqa: E402

torch.set_num_threads(4)
WT = Path(__file__).resolve().parents[2]
OUT = WT / "reports" / "intransitivity_exp.json"
ZPQ = WT / "reports" / "intr_z.parquet"
ATTR = ["epos", "ten", "agari", "bw", "age", "female", "dprof", "cons"]
D = len(ATTR)
LAMS = [1e-5, 1e-4, 1e-3, 1e-2]
K_BC = 4
N_BOOT = 2000
N_SIM = 200
DEV = "--dev" in sys.argv
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


# ------------------------------------------------------------------ 1. 馬×走の表
def build_runs() -> pd.DataFrame:
    cols = ["racekey", "ymd", "ban", "ped", "dist", "td", "tosu", "jyun", "ijou", "tan_odds", "idm",
            "ten_idx", "agari_idx", "c1", "c2", "c3", "c4", "bw"]
    sed = gc.load_sed(cols)
    sed = sed[sed.td.isin([1, 2])].copy()
    sed["ped"] = sed["ped"].astype(str)
    h = sed[(sed.ijou == 0) & (sed.jyun > 0)].copy()
    cn = h[["c1", "c2", "c3", "c4"]].where(h[["c1", "c2", "c3", "c4"]] > 0)
    first = cn.bfill(axis=1).iloc[:, 0]
    h["epos"] = ((first - 1) / (h.tosu - 1).clip(lower=1)).clip(0, 1)
    h["bwv"] = h.bw.where((h.bw > 300) & (h.bw < 700))
    h["distf"] = h.dist.astype(float)
    h = h.sort_values(["ped", "date"]).reset_index(drop=True)
    g = h.groupby("ped", sort=False)
    for src, dst in (("epos", "epos"), ("ten_idx", "ten"), ("agari_idx", "agari"), ("idm", "s_idm"), ("distf", "mdist")):
        h["h_" + dst] = g[src].rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    h["h_cons"] = g["idm"].rolling(5, min_periods=2).std().reset_index(level=0, drop=True)
    h["h_bw"] = g["bwv"].ffill()
    hist = h[["ped", "date", "h_epos", "h_ten", "h_agari", "h_s_idm", "h_mdist", "h_cons", "h_bw"]].sort_values("date")
    t = sed[(sed.ymd >= 20130101) & (sed.ymd <= 20251231)].sort_values("date")
    r = pd.merge_asof(t, hist, on="date", by="ped", allow_exact_matches=False, direction="backward")
    r["year"] = (r.ymd // 10000).astype(int)
    yy = pd.to_numeric(r.ped.str[:2], errors="coerce")
    by = np.where(yy > 50, 1900 + yy, 2000 + yy)
    r["age"] = (r.year - by).clip(2, 10).astype(float)
    lite = gc.load_master_lite()[["レースID(新/馬番無)", "馬番", "性別"]]
    lite = pd.DataFrame({"rid_s": lite["レースID(新/馬番無)"].astype(str).str[:16], "ban": lite["馬番"].astype(int),
                         "female": (lite["性別"] == "牝").astype(float)}).drop_duplicates(["rid_s", "ban"])
    r = r.merge(lite, on=["rid_s", "ban"], how="left")
    r["epos"] = r.h_epos; r["ten"] = r.h_ten; r["agari"] = r.h_agari; r["bw"] = r.h_bw
    r["dprof"] = (r.h_mdist - r.dist) / 100.0
    r["cons"] = r.h_cons
    r["s_idm"] = r.h_s_idm
    r["s_mkt"] = -np.log(r.tan_odds.where(r.tan_odds > 0))
    r["infield"] = (r.tan_odds > 0) & ~r.ijou.isin([1, 2])
    r = r.sort_values(["rid_s", "ban"]).reset_index(drop=True)
    r["rc"] = pd.factorize(r.rid_s)[0]
    keep = ["rid_s", "rc", "ban", "ped", "ymd", "year", "jyun", "ijou", "tan_odds", "infield", "s_idm", "s_mkt"] + ATTR
    return r[keep]


def standardise(R: pd.DataFrame):
    """G: 2013-22 の全体平均/sd。R: レース内 (単勝オッズのある馬) z 化。欠損→0。"""
    X = R[ATTR].values.astype(float)
    tr = (R.year <= 2022).values
    mu = np.nanmean(X[tr], 0); sd = np.nanstd(X[tr], 0)
    XG = np.nan_to_num((X - mu) / sd, nan=0.0)
    Xf = np.where(R.infield.values[:, None], X, np.nan)
    df = pd.DataFrame(Xf, columns=ATTR); rc = R.rc.values
    m = df.groupby(rc).transform("mean").values; s = df.groupby(rc).transform("std").values
    XR = (X - m) / np.where(s > 1e-6, s, np.nan)
    XR = np.nan_to_num(np.where(R.infield.values[:, None], XR, np.nan), nan=0.0)
    return XG, XR, {"mean": dict(zip(ATTR, mu.round(4).tolist())), "sd": dict(zip(ATTR, sd.round(4).tolist()))}


# ------------------------------------------------------------------ 2. ペア
def make_pairs(R: pd.DataFrame, mask: np.ndarray, rng) -> dict:
    idx = np.flatnonzero(mask)
    rc = R.rc.values[idx]
    ch = np.flatnonzero(np.diff(rc)) + 1
    st = np.r_[0, ch]; en = np.r_[ch, len(idx)]; m = en - st
    I, J = [], []
    for size in np.unique(m):
        if size < 2:
            continue
        s0 = st[m == size]; iu, ju = np.triu_indices(size, 1)
        I.append(idx[(s0[:, None] + iu[None, :]).ravel()]); J.append(idx[(s0[:, None] + ju[None, :]).ravel()])
    I = np.concatenate(I); J = np.concatenate(J)
    jy = R.jyun.values
    keep = jy[I] != jy[J]
    I, J = I[keep], J[keep]
    flip = rng.random(len(I)) < 0.5
    I2 = np.where(flip, J, I); J2 = np.where(flip, I, J)
    return {"I": I2, "J": J2, "y": (jy[I2] < jy[J2]).astype(np.float32), "year": R.year.values[I2], "ymd": R.ymd.values[I2]}


# ------------------------------------------------------------------ 3. ペアモデル (torch, full-batch L-BFGS)
class PairModel:
    def __init__(self, kind: str, seed: int = 0):
        self.kind = kind
        g = torch.Generator().manual_seed(seed)
        self.a = torch.ones(1, requires_grad=True)
        self.w = torch.zeros(D, requires_grad=True)
        self.params = [self.a, self.w]; self.pen = []
        if kind in ("het", "hetbil"):
            self.v = torch.zeros(D, requires_grad=True); self.params.append(self.v)
        if kind in ("bil", "hetbil"):
            self.U = torch.zeros(D, D, requires_grad=True); self.params.append(self.U); self.pen = [self.U]
        elif kind == "bc":
            self.B = (0.1 * torch.randn(D, K_BC, generator=g)).requires_grad_()
            self.C = (0.1 * torch.randn(D, K_BC, generator=g)).requires_grad_()
            self.params += [self.B, self.C]; self.pen = [self.B, self.C]

    def A(self):
        u = torch.triu(self.U, 1)
        return u - u.T

    def matchup(self, xi, xj):
        if self.kind in ("bil", "hetbil"):
            return ((xi @ self.A()) * xj).sum(1)
        if self.kind == "bc":
            bi, ci = torch.tanh(xi @ self.B), torch.tanh(xi @ self.C)
            bj, cj = torch.tanh(xj @ self.B), torch.tanh(xj @ self.C)
            return (bi * cj).sum(1) - (bj * ci).sum(1)
        return 0.0

    def logit(self, s, X, I, J):
        xi, xj = X[I], X[J]
        ds = self.a * (s[I] - s[J])
        if self.kind in ("het", "hetbil"):
            ds = ds * (1.0 + (xi + xj) @ self.v)
        return ds + (xi - xj) @ self.w + self.matchup(xi, xj)

    def penalty(self, lam):
        if not self.pen:
            return 0.0
        if self.kind in ("bil", "hetbil"):
            return lam * (torch.triu(self.U, 1) ** 2).sum()
        return lam * sum((p ** 2).sum() for p in self.pen)


def fit_pair(kind, s, X, I, J, y, lam=0.0, seed=0, max_iter=300) -> PairModel:
    m = PairModel(kind, seed)
    opt = torch.optim.LBFGS(m.params, lr=1.0, max_iter=max_iter, tolerance_grad=1e-7, tolerance_change=1e-10,
                            history_size=20, line_search_fn="strong_wolfe")
    bce = torch.nn.functional.binary_cross_entropy_with_logits

    def closure():
        opt.zero_grad()
        loss = bce(m.logit(s, X, I, J), y) + m.penalty(lam)
        loss.backward()
        return loss
    opt.step(closure)
    return m


@torch.no_grad()
def pair_loss(m: PairModel, s, X, I, J, y, chunk=500_000) -> np.ndarray:
    out = []
    for k in range(0, len(I), chunk):
        z = m.logit(s, X, I[k:k + chunk], J[k:k + chunk]).double()
        out.append(torch.nn.functional.binary_cross_entropy_with_logits(z, y[k:k + chunk].double(), reduction="none").numpy())
    return np.concatenate(out)


@torch.no_grad()
def pair_matchup(m: PairModel, X, I, J, chunk=500_000) -> np.ndarray:
    out = []
    for k in range(0, len(I), chunk):
        out.append(m.matchup(X[I[k:k + chunk]], X[J[k:k + chunk]]).double().numpy())
    return np.concatenate(out)


def boot_gain(diff: np.ndarray, ymd: np.ndarray, rng) -> dict:
    """diff = LL(T) − LL(model) (正 = 改善)。開催日クラスタ bootstrap。"""
    ud, dc = np.unique(ymd, return_inverse=True)
    s = np.bincount(dc, weights=diff, minlength=len(ud)); c = np.bincount(dc, minlength=len(ud)).astype(float)
    idx = rng.integers(0, len(ud), (N_BOOT, len(ud)))
    v = s[idx].sum(1) / c[idx].sum(1)
    return {"n_pairs": int(len(diff)), "n_days": int(len(ud)), "gain_per_pair": float(diff.mean()),
            "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]}


# ------------------------------------------------------------------ 4. レース水準 z
def field_z(kind, model: PairModel, X: np.ndarray, R: pd.DataFrame, perm_rc: np.ndarray | None = None) -> np.ndarray:
    """z_i = (1/n) Σ_{j≠i} m(i,j)。perm_rc を渡すと相手の場ベクトルを別レースのものに入れ替える (プラセボ)。"""
    rc = R.rc.values; nr = rc.max() + 1; inf = R.infield.values
    n = np.bincount(rc[inf], minlength=nr).astype(float); n[n == 0] = np.nan
    src = rc if perm_rc is None else perm_rc[rc]
    if kind == "bil":
        A = model.A().detach().numpy().astype(float)
        S = np.zeros((nr, D)); np.add.at(S, rc[inf], X[inf])
        z = ((X @ A) * (S / n[:, None])[src]).sum(1)
    else:
        B = model.B.detach().numpy().astype(float); C = model.C.detach().numpy().astype(float)
        bl = np.tanh(X @ B); chst = np.tanh(X @ C)
        SB = np.zeros((nr, K_BC)); SC = np.zeros((nr, K_BC))
        np.add.at(SB, rc[inf], bl[inf]); np.add.at(SC, rc[inf], chst[inf])
        z = ((bl * (SC / n[:, None])[src]).sum(1) - (chst * (SB / n[:, None])[src]).sum(1))
    return np.where(inf, z, np.nan)


def year_perm(R: pd.DataFrame, rng) -> np.ndarray:
    race = R.groupby("rc").year.first()
    perm = np.arange(len(race))
    for y in race.unique():
        ix = np.flatnonzero(race.values == y)
        perm[ix] = rng.permutation(ix)
    return perm


def gate_years(z: pd.DataFrame, col: str, l2: float = 1e-4) -> dict:
    """gate_common.gate と同じ fit を再現し、test の差を年別に出す (PASS の年別符号確認用)。"""
    b = gc.load_base().merge(z[["rid_s", "ban", col]].drop_duplicates(["rid_s", "ban"]), on=["rid_s", "ban"], how="left")
    b[col] = pd.to_numeric(b[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    b[col] = b[col].fillna(b.groupby("rid_s")[col].transform("mean")).fillna(0.0)
    b[col] = b[col] - b.groupby("rid_s")[col].transform("mean")
    sd = b.loc[b.split == "valid", col].std()
    b[col] = b[col] / (sd if sd and sd > 0 else 1.0)
    rng = np.random.default_rng(0); res = {}
    for gname, bcols in {"P": ["lm7"], "M": ["lk", "lm7"], "M0": ["lk"], "A10": ["lk10", "lm7"]}.items():
        d = b[b["lk10"].notna()] if "lk10" in bcols else b
        xv = d[d.split == "valid"].reset_index(drop=True); xt = d[d.split == "test"].reset_index(drop=True)
        cv = pd.factorize(xv.rid_s)[0]; ct = pd.factorize(xt.rid_s)[0]
        wv = np.flatnonzero((xv.jyun == 1).values); wt = np.flatnonzero((xt.jyun == 1).values)
        lp = {}
        for tag, cols in (("base", bcols), ("plus", bcols + [col])):
            w = gc._fit(xv[cols].values, cv, cv.max() + 1, wv, l2)
            lp[tag] = gc.race_logsoftmax(xt[cols].values @ w, ct, ct.max() + 1)
        diff = -(lp["plus"][wt] - lp["base"][wt]); dates = xt.date.values[wt]
        out = {"all": float(diff.mean())}
        for y in ("2024", "2025"):
            mk = np.char.startswith(dates.astype(str), y)
            ud, dc = np.unique(dates[mk], return_inverse=True)
            s = np.bincount(dc, weights=diff[mk]); c = np.bincount(dc).astype(float)
            idx = rng.integers(0, len(ud), (N_BOOT, len(ud))); v = s[idx].sum(1) / c[idx].sum(1)
            out[y] = {"d_nll": float(diff[mk].mean()), "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
                      "n": int(mk.sum())}
        res[gname] = out
    return res


# ------------------------------------------------------------------ 5. 三つ巴
def triads(R: pd.DataFrame, mask: np.ndarray, rng) -> dict:
    """2013-2023 の対戦から循環三つ組を数え、推移モデル (Gumbel) の再生成と比べる。"""
    idx = np.flatnonzero(mask)
    rc = R.rc.values[idx]
    ch = np.flatnonzero(np.diff(rc)) + 1
    st = np.r_[0, ch]; en = np.r_[ch, len(idx)]; m = en - st
    I, J = [], []
    for size in np.unique(m):
        if size < 2:
            continue
        s0 = st[m == size]; iu, ju = np.triu_indices(size, 1)
        I.append(idx[(s0[:, None] + iu[None, :]).ravel()]); J.append(idx[(s0[:, None] + ju[None, :]).ravel()])
    I = np.concatenate(I); J = np.concatenate(J)
    hid = pd.factorize(R.ped.values)[0].astype(np.int64); nh = int(hid.max()) + 1
    hi_, hj_ = hid[I], hid[J]
    swap = hi_ > hj_
    lo_row = np.where(swap, J, I); hi_row = np.where(swap, I, J)
    lo = np.minimum(hi_, hj_); hi = np.maximum(hi_, hj_)
    key = lo * nh + hi
    uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    rep = cnt[inv] >= 2
    lo_row, hi_row, inv = lo_row[rep], hi_row[rep], inv[rep]
    uk2, inv2 = np.unique(inv, return_inverse=True)
    pk = uk[uk2]; plo = (pk // nh).astype(np.int64); phi = (pk % nh).astype(np.int64); pcnt = cnt[uk2].astype(float)
    npair = len(pk)
    jy = R.jyun.values.astype(float)

    def count(ahead_lo: np.ndarray):
        """ahead_lo: 対戦行ごとに lo 側が先着なら 1、同着 0.5。→ (勝ち越し辺の三つ組数, 循環数, 全勝辺版 2 つ)"""
        wl = np.bincount(inv2, weights=ahead_lo, minlength=npair)
        out = []
        for mode in ("maj", "unan"):
            if mode == "maj":
                dom_lo = wl > pcnt / 2; dom_hi = wl < pcnt / 2
            else:
                dom_lo = wl == pcnt; dom_hi = wl == 0
            r_ = np.r_[plo[dom_lo], phi[dom_hi]]; c_ = np.r_[phi[dom_lo], plo[dom_hi]]
            Dm = sp.csr_matrix((np.ones(len(r_), dtype=np.float64), (r_, c_)), shape=(nh, nh))
            U = Dm + Dm.T
            tot = (U @ U).multiply(U).sum() / 6.0
            cyc = (Dm @ Dm).multiply(Dm.T).sum() / 3.0
            out += [int(dom_lo.sum() + dom_hi.sum()), float(tot), float(cyc)]
        return out

    obs_ahead = np.where(jy[lo_row] < jy[hi_row], 1.0, np.where(jy[lo_row] == jy[hi_row], 0.5, 0.0))
    obs = count(obs_ahead)
    res = {"n_runs": int(len(idx)), "n_pair_meetings_all": int(len(I)), "n_pairs_met_ge2": int(npair),
           "observed": {"maj": {"edges": obs[0], "triads": obs[1], "cyclic": obs[2], "cyclic_frac": obs[2] / max(obs[1], 1)},
                        "unan": {"edges": obs[3], "triads": obs[4], "cyclic": obs[5], "cyclic_frac": obs[5] / max(obs[4], 1)}},
           "null": {}}
    # 強さの尺度 a: 同じペア集合で a·Δs の 1 変数ロジスティック
    y_all = (jy[I] < jy[J]).astype(float); tie = jy[I] == jy[J]
    for sname in ("s_idm", "s_mkt"):
        s = R[sname].values.astype(float)
        ds = (s[I] - s[J])[~tie]; yy = y_all[~tie]
        f = lambda a: float(np.mean(np.logaddexp(0, a[0] * ds) - yy * a[0] * ds))  # noqa: E731
        a = float(minimize(f, [0.1 if sname == "s_idm" else 1.0], method="Nelder-Mead").x[0])
        sims = []
        for _ in range(N_SIM):
            u = np.full(len(R), np.nan)
            u[idx] = a * s[idx] + rng.gumbel(size=len(idx))
            sims.append(count((u[lo_row] > u[hi_row]).astype(float)))
        sims = np.array(sims, dtype=float)
        nul = {"a": a}
        for off, mode in ((0, "maj"), (3, "unan")):
            fr = sims[:, off + 2] / np.maximum(sims[:, off + 1], 1)
            o = res["observed"][mode]
            nul[mode] = {"triads_mean": float(sims[:, off + 1].mean()), "cyclic_mean": float(sims[:, off + 2].mean()),
                         "cyclic_frac_mean": float(fr.mean()), "cyclic_frac_sd": float(fr.std(ddof=1)),
                         "cyclic_frac_p2.5": float(np.percentile(fr, 2.5)), "cyclic_frac_p97.5": float(np.percentile(fr, 97.5)),
                         "z_obs": float((o["cyclic_frac"] - fr.mean()) / fr.std(ddof=1)),
                         "obs_exceeds_p97.5": bool(o["cyclic_frac"] > np.percentile(fr, 97.5))}
        res["null"][sname] = nul
        log(f"[triad] null {sname}: a={a:.4f}  maj 観測 {res['observed']['maj']['cyclic_frac']:.4f} vs 期待 "
            f"{nul['maj']['cyclic_frac_mean']:.4f}±{nul['maj']['cyclic_frac_sd']:.4f} (z={nul['maj']['z_obs']:+.1f})  "
            f"unan 観測 {res['observed']['unan']['cyclic_frac']:.4f} vs {nul['unan']['cyclic_frac_mean']:.4f}"
            f"±{nul['unan']['cyclic_frac_sd']:.4f} (z={nul['unan']['z_obs']:+.1f})")
    return res


# ------------------------------------------------------------------ 6. gate と同じ土俵のペア診断
def base_pair_diag(R, models, rng) -> dict:
    b = gc.load_base()
    b = b.merge(pd.DataFrame({"rid_s": R.rid_s.values, "ban": R.ban.values, "ridx": np.arange(len(R))}),
                on=["rid_s", "ban"], how="left")
    b = b[b.ridx.notna() & (b.jyun > 0)].sort_values(["rid_s", "ban"]).reset_index(drop=True)
    b["rc"] = pd.factorize(b.rid_s)[0]
    P = make_pairs(b.assign(year=b.date.str[:4].astype(int), ymd=b.date.astype(int)), np.ones(len(b), bool), rng)
    I, J, y = P["I"], P["J"], P["y"].astype(float)
    dl7 = (b.lm7.values[I] - b.lm7.values[J]); dlk = (b.lk.values[I] - b.lk.values[J])
    ri = torch.as_tensor(b.ridx.values[I].astype(np.int64)); rj = torch.as_tensor(b.ridx.values[J].astype(np.int64))
    va = P["year"] == 2023; te = ~va
    out = {"n_pairs_valid": int(va.sum()), "n_pairs_test": int(te.sum())}

    def fit_eval(F):
        f = lambda w: float(np.mean(np.logaddexp(0, F[va] @ w) - y[va] * (F[va] @ w)))  # noqa: E731
        g = lambda w: ((1 / (1 + np.exp(-(F[va] @ w))) - y[va])[:, None] * F[va]).mean(0)  # noqa: E731
        w = minimize(f, np.zeros(F.shape[1]), jac=g, method="L-BFGS-B").x
        return w, np.logaddexp(0, F @ w) - y * (F @ w)
    for bname, Fb in (("v7", dl7[:, None]), ("market+v7", np.c_[dlk, dl7])):
        _, lb = fit_eval(Fb)
        for vname, (kind, mdl, X) in models.items():
            mm = pair_matchup(mdl, X, ri, rj)
            w, lpz = fit_eval(np.c_[Fb, mm / mm[va].std()])
            r = {"coef_matchup": float(w[-1])}
            if not DEV:
                r["test"] = boot_gain((lb - lpz)[te], P["ymd"][te], rng)
            r["valid_insample_gain"] = float((lb - lpz)[va].mean())
            out[f"{bname}|{vname}"] = r
    return out


# ------------------------------------------------------------------ 事後追加 (探索)
def posthoc():
    rng = np.random.default_rng(20261114)
    res = json.load(open(OUT, encoding="utf-8"))
    R = build_runs()
    _, XR, _ = standardise(R)
    elig = (R.infield & (R.ijou == 0) & (R.jyun > 0) & R.s_idm.notna()).values
    P = make_pairs(R, elig, rng); yr = P["year"]
    sd_idm = float(np.nanstd(R.s_idm.values[(R.year <= 2022).values]))
    S = {"idm": torch.as_tensor(np.nan_to_num(R.s_idm.values / sd_idm, nan=0.0), dtype=torch.float32),
         "mkt": torch.as_tensor(np.nan_to_num(R.s_mkt.values, nan=0.0), dtype=torch.float32)}
    X = torch.as_tensor(XR, dtype=torch.float32)
    It = torch.as_tensor(P["I"].astype(np.int64)); Jt = torch.as_tensor(P["J"].astype(np.int64)); yt = torch.as_tensor(P["y"])
    tr = torch.as_tensor(yr <= 2022)
    periods = {"insample_2013_22": yr <= 2022, "oos_2023": yr == 2023, "test_2024_25": yr >= 2024}
    out = {}
    for sname in ("idm", "mkt"):
        s = S[sname]; lam = res["pairwise"][f"{sname}|R|bil"]["lam"]
        L = {}
        for kind in ("T", "bil", "het", "hetbil"):
            m = fit_pair(kind, s, X, It[tr], Jt[tr], yt[tr], lam=lam if "bil" in kind else 0.0)
            L[kind] = pair_loss(m, s, X, It, Jt, yt)
            if kind == "het":
                out[f"{sname}|v"] = dict(zip(ATTR, m.v.detach().numpy().round(5).tolist()))
        for name, (a, b) in {"bil_minus_T": ("T", "bil"), "het_minus_T": ("T", "het"), "hetbil_minus_het": ("het", "hetbil")}.items():
            out[f"{sname}|{name}"] = {k: boot_gain((L[a] - L[b])[mk], P["ymd"][mk], rng) for k, mk in periods.items()}
            log(f"[posthoc {sname}] {name}: " + "  ".join(
                f"{k} {v['gain_per_pair'] * 1e4:+.3f}e-4 [{v['ci95'][0] * 1e4:+.3f},{v['ci95'][1] * 1e4:+.3f}]"
                for k, v in out[f"{sname}|{name}"].items()))
    res["posthoc"] = out
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    log(f"saved {OUT} (posthoc)")


# ------------------------------------------------------------------ main
def main():
    rng = np.random.default_rng(20261114)
    res = {"dev": DEV, "attr": ATTR, "lams": LAMS, "k_bc": K_BC}
    R = build_runs()
    XG, XR, st = standardise(R)
    res["standardise_G"] = st
    elig = (R.infield & (R.ijou == 0) & (R.jyun > 0) & R.s_idm.notna()).values
    log(f"runs {len(R):,}  races {R.rc.nunique():,}  infield {R.infield.mean():.3f}  pair-eligible {elig.mean():.3f}")
    P = make_pairs(R, elig, rng)
    yr = P["year"]
    res["n_pairs"] = {"2013-22": int((yr <= 2022).sum()), "2023": int((yr == 2023).sum()), "2024-25": int((yr >= 2024).sum())}
    log(f"pairs {res['n_pairs']}")

    sd_idm = float(np.nanstd(R.s_idm.values[(R.year <= 2022).values]))
    S = {"idm": torch.as_tensor(np.nan_to_num(R.s_idm.values / sd_idm, nan=0.0), dtype=torch.float32),
         "mkt": torch.as_tensor(np.nan_to_num(R.s_mkt.values, nan=0.0), dtype=torch.float32)}
    XT = {"G": torch.as_tensor(XG, dtype=torch.float32), "R": torch.as_tensor(XR, dtype=torch.float32)}
    It = torch.as_tensor(P["I"].astype(np.int64)); Jt = torch.as_tensor(P["J"].astype(np.int64)); yt = torch.as_tensor(P["y"])
    tr = torch.as_tensor(yr <= 2022); tr_in = torch.as_tensor(yr <= 2020); va_in = torch.as_tensor((yr >= 2021) & (yr <= 2022))
    periods = {"insample_2013_22": yr <= 2022, "oos_2023": yr == 2023}
    if not DEV:
        periods["test_2024_25"] = yr >= 2024

    res["pairwise"] = {}; fitted = {}
    for sname in ("idm", "mkt"):
        s = S[sname]
        for xname in ("G", "R"):
            X = XT[xname]
            mT = fit_pair("T", s, X, It[tr], Jt[tr], yt[tr])
            lT = pair_loss(mT, s, X, It, Jt, yt)
            l0 = float(np.log(2))
            base = {k: float(lT[m].mean()) for k, m in periods.items()}
            log(f"[{sname}/{xname}] T: a={float(mT.a):.4f}  LL " + "  ".join(f"{k} {v:.5f}" for k, v in base.items()))
            res["pairwise"][f"{sname}|{xname}|T"] = {"a": float(mT.a), "w": dict(zip(ATTR, mT.w.detach().numpy().round(5).tolist())),
                                                     "logloss": base, "gain_vs_coinflip": {k: l0 - v for k, v in base.items()}}
            kinds = ["bil"] + (["bc"] if xname == "R" else [])
            for kind in kinds:
                sel = {}
                seeds = (0, 1) if kind == "bc" else (0,)
                for lam in LAMS:
                    for seed in seeds:
                        mi = fit_pair(kind, s, X, It[tr_in], Jt[tr_in], yt[tr_in], lam=lam, seed=seed, max_iter=150)
                        sel[(lam, seed)] = float(pair_loss(mi, s, X, It[va_in], Jt[va_in], yt[va_in]).mean())
                lam, seed = min(sel, key=sel.get)
                mM = fit_pair(kind, s, X, It[tr], Jt[tr], yt[tr], lam=lam, seed=seed)
                lM = pair_loss(mM, s, X, It, Jt, yt)
                mu = pair_matchup(mM, X, It, Jt)
                r = {"lam": lam, "seed": seed, "inner_valid_logloss": {f"{k[0]:g}|{k[1]}": v for k, v in sel.items()},
                     "a": float(mM.a), "sd_matchup_logit_2023": float(mu[yr == 2023].std()),
                     "sd_strength_logit_2023": float((float(mM.a) * (s[It] - s[Jt]).numpy())[yr == 2023].std())}
                for k, m in periods.items():
                    r[k] = boot_gain((lT - lM)[m], P["ymd"][m], rng)
                if kind == "bil":
                    A = mM.A().detach().numpy()
                    iu = np.triu_indices(D, 1); order = np.argsort(-np.abs(A[iu]))[:6]
                    r["top_A"] = [{"pair": f"{ATTR[iu[0][o]]}×{ATTR[iu[1][o]]}", "coef": float(A[iu][o])} for o in order]
                res["pairwise"][f"{sname}|{xname}|{kind}"] = r
                fitted[(sname, xname, kind)] = mM
                log(f"[{sname}/{xname}] {kind}: lam={lam:g} seed={seed}  " + "  ".join(
                    f"{k} gain {r[k]['gain_per_pair'] * 1e4:+.3f}e-4 [{r[k]['ci95'][0] * 1e4:+.3f},{r[k]['ci95'][1] * 1e4:+.3f}]"
                    for k in periods))
            del lT
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)

    # ---- レース水準 z
    variants = {"V1_bil_idm": ("bil", fitted[("idm", "G", "bil")], XG), "V2_bc_idm": ("bc", fitted[("idm", "R", "bc")], XR),
                "V3_bil_mkt": ("bil", fitted[("mkt", "G", "bil")], XG)}
    perm = year_perm(R, rng)
    Z = R[["rid_s", "ban", "year"]].copy()
    for vname, (kind, mdl, X) in variants.items():
        Z[vname] = field_z(kind, mdl, X, R)
        Z[vname + "_placebo"] = field_z(kind, mdl, X, R, perm)
    Z.to_parquet(ZPQ)
    res["z_summary"] = {v: {"sd_2023": float(Z.loc[Z.year == 2023, v].std()), "nan_rate": float(Z[v].isna().mean())} for v in variants}
    res["z_corr_2023"] = Z.loc[Z.year == 2023, list(variants)].corr().round(4).to_dict()

    # ---- 三つ巴 (2013-2023)
    res["triads_2013_2023"] = triads(R, elig & (R.year <= 2023).values, rng)

    # ---- gate と同じ土俵のペア診断
    XTn = {"V1_bil_idm": ("bil", variants["V1_bil_idm"][1], XT["G"]), "V2_bc_idm": ("bc", variants["V2_bc_idm"][1], XT["R"]),
           "V3_bil_mkt": ("bil", variants["V3_bil_mkt"][1], XT["G"])}
    res["base_pair_diag"] = base_pair_diag(R, XTn, rng)
    log(f"[base pair diag] {json.dumps(res['base_pair_diag'], ensure_ascii=False)[:1500]}")

    # ---- ゲート (test を見るのはここだけ)
    if not DEV:
        res["gates"] = {}; res["gates_placebo"] = {}; res["gates_by_year"] = {}
        zz = Z[Z.year >= 2023]
        for vname in variants:
            res["gates"][vname] = gc.gate(zz, [vname], name=vname)
            res["gates_placebo"][vname] = gc.gate(zz, [vname + "_placebo"], name=vname + "_placebo")
            res["gates_by_year"][vname] = gate_years(zz, vname)
            log(f"[by-year {vname}] {json.dumps(res['gates_by_year'][vname])}")
        any_p = [v for v in variants if res["gates"][v]["gates"]["P"]["verdict"] == "PASS"]
        any_m = [v for v in variants if res["gates"][v]["gates"]["M"]["verdict"] == "PASS"]
        res["verdict"] = {"P_pass": any_p, "M_pass": any_m}
    res["elapsed_s"] = time.time() - T0
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    log(f"saved {OUT}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    if "--posthoc" in sys.argv:
        posthoc()
    else:
        main()
