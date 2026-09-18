# -*- coding: utf-8 -*-
"""
latent_state.py — 長期能力 a と短期状態 s の2成分フィルタ (多頭数 Plackett–Luce 尤度、filtering のみ)
=====================================================================================================
観測モデル (1レースの全着順を1観測):
  走りの強さ = a + s + ノイズ(β)。EXP02 と同じ Weng–Lin PL 版 (γ=1) を、
  成分の和 (平均 a+s, 分散 var_a+var_s) に対して適用し、平均の変化と分散の縮小を
  成分の分散比で配分する (独立ガウスの和への観測と同じ Kalman 分配):
     Δa = Δeff · var_a/var_eff,   Δs = Δeff · var_s/var_eff
     var_k' = var_k − (var_k²/var_eff)(1 − f),   f = var_eff'/var_eff

状態方程式 (予測時点、前回更新からの日数 Δt):
  長期能力: a_prior = a,               var_a_prior = var_a + τa² · Δt       (ゆっくり動くランダムウォーク)
  短期状態: ρ = exp(−ln2 · Δt / HL)
            s_prior = ρ · s,             var_s_prior = ρ² var_s + qs² (1 − ρ²)  (0 へ戻る AR(1)、定常分散 qs²)
  初出走:   a ~ N(25, (25/3)²),         s ~ N(0, qs²)

識別の約束: s の長期平均は0 (事前平均0へ回帰)、s は日数で0へ戻る、a は τa² が小さく遅い、
            s の定常分散 qs² は1走あたりの a の変化より大きく取れる、a は EXP02 と同じ初期分布・尺度。

innovation (予測に使う前の、その走りの「期待からのずれ」):
  z_i = Ω_i / sqrt(Δ_i)   (Ω, Δ は Weng–Lin 更新のスコアと Fisher 情報、更新前の a+s で計算)
  正 = 相手構成から期待されたより良い着順。市場オッズは使わない。
  innovation_ewma: 更新時に  ewma ← ρ·ewma + (1−ρ)·z  (ρ は前回からの日数で上と同じ半減期)。
                   予測時点では ρ(経過日数)·ewma を出す。

同日: EXP02 と同じく、その日の全レースを前日までの状態で予測し、日の終わりにまとめて更新。
順位: master の出走行のみ (取消・除外・中止は無い)、同着は同順位、着差は使わない。
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
from analysis.mcond.exp02_dynamic_skill_dev.dyn_skill import load_runs, MU0, SIGMA0, KAPPA  # noqa: E402,F401

LN2 = np.log(2.0)


@dataclass(frozen=True)
class Hyper:
    halflife_days: float
    tau2_a: float
    qs: float
    beta: float


def pl_score(mu, var, rank, beta):
    """Weng–Lin PL (γ=1): 新しい平均・分散と、スコア Ω・Fisher 情報 Δ・c を返す。"""
    c = np.sqrt(np.sum(var + beta ** 2))
    e = np.exp((mu - mu.max()) / c)
    ge = rank[None, :] >= rank[:, None]
    S = (ge * e[None, :]).sum(axis=1)
    A = (rank[None, :] == rank[:, None]).sum(axis=1)
    M = rank[:, None] <= rank[None, :]
    Q = e[None, :] / S[:, None]
    omega = (M * (np.eye(len(mu)) - Q) / A[:, None]).sum(axis=0)
    delta = (M * Q * (1 - Q) / A[:, None]).sum(axis=0)
    new_mu = mu + var / c * omega
    new_var = var * np.maximum(1 - (var / c ** 2) * delta, KAPPA)
    return new_mu, new_var, omega, delta, c


def rank_quantile_residual(mu, var, rank, beta):
    """innovation の候補B: 期待順位分位 − 実順位分位 (期待順位はペアの PL 勝率から)。正=期待より良い。"""
    n = len(mu)
    c = np.sqrt(np.sum(var + beta ** 2))
    z = mu / c
    p_ahead = 1.0 / (1.0 + np.exp(z[:, None] - z[None, :]))     # [i, j]: j が i より前に来る確率
    np.fill_diagonal(p_ahead, 0.0)
    exp_rank = 1.0 + p_ahead.sum(axis=1)
    return (exp_rank - rank) / max(n - 1, 1)


class State:
    def __init__(self, hp: Hyper):
        self.hp = hp
        self.a, self.va, self.s, self.vs = {}, {}, {}, {}
        self.last, self.n, self.da, self.z, self.zq, self.ew = {}, {}, {}, {}, {}, {}

    def prior(self, h, day):
        hp = self.hp
        if h not in self.a:
            return dict(a=MU0, va=SIGMA0 ** 2, s=0.0, vs=hp.qs ** 2, dt=np.nan, rho=np.nan, n=0,
                        da=np.nan, z=np.nan, zq=np.nan, ew=np.nan)
        dt = (day - self.last[h]).days
        rho = float(np.exp(-LN2 * dt / hp.halflife_days))
        return dict(a=self.a[h], va=self.va[h] + hp.tau2_a * dt, s=rho * self.s[h],
                    vs=rho ** 2 * self.vs[h] + hp.qs ** 2 * (1 - rho ** 2), dt=dt, rho=rho,
                    n=self.n[h], da=self.da[h], z=self.z[h], zq=self.zq[h], ew=rho * self.ew[h])


def features(pr, mu_a, var_a, mu_s, var_s):
    n = len(mu_a)
    comb = mu_a + mu_s
    r_c = np.empty(n); r_c[(-comb).argsort(kind="stable")] = np.arange(1, n + 1)
    r_s = np.empty(n); r_s[(-mu_s).argsort(kind="stable")] = np.arange(1, n + 1)
    return {
        "persistent_ability_mu": mu_a,
        "persistent_ability_change": np.array([p["da"] for p in pr], float),
        "latent_short_state_mu": mu_s,
        "latent_short_state_sigma": np.sqrt(var_s),
        "latent_short_state_abs": np.abs(mu_s),
        "latent_state_days_since_update": np.array([p["dt"] for p in pr], float),
        "latent_state_decay_ratio": np.array([p["rho"] for p in pr], float),
        "last_performance_innovation": np.array([p["z"] for p in pr], float),
        "last_performance_innovation_rankq": np.array([p["zq"] for p in pr], float),
        "innovation_ewma": np.array([p["ew"] for p in pr], float),
        "combined_ability_state": comb,
        "combined_rank_in_race": r_c,
        "state_rank_in_race": r_s,
        "state_minus_field_mean": mu_s - mu_s.mean(),
        "_n_updates": np.array([p["n"] for p in pr], float),
    }


def run(df: pd.DataFrame, hp: Hyper, collect: bool = True):
    """日単位バッチ。返り値: (特徴 DataFrame, レースごとの勝ち馬対数尤度, 馬×走ごとの innovation 記録)"""
    st = State(hp)
    feats, ll, innov = [], [], []
    for day, dd in df.groupby("date", sort=True):
        pending = []
        for rid, g in dd.groupby("rid16", sort=False):
            hids = g["hid"].to_numpy()
            pr = [st.prior(h, day) for h in hids]
            a = np.array([p["a"] for p in pr]); va = np.array([p["va"] for p in pr])
            s = np.array([p["s"] for p in pr]); vs = np.array([p["vs"] for p in pr])
            fin = g["fin"].to_numpy(float)
            eff, veff = a + s, va + vs
            c = np.sqrt(np.sum(veff + hp.beta ** 2))
            zz = eff / c
            pw = np.exp(zz - zz.max()); pw /= pw.sum()
            ll.append((day, rid, float(np.log(max(pw[fin == fin.min()].mean(), 1e-12)))))
            if collect:
                f = features(pr, a, va, s, vs)
                f.update({"rid16": np.full(len(g), rid), "ban": g["ban"].to_numpy(), "hid": hids,
                          "date": np.full(len(g), day)})
                feats.append(pd.DataFrame(f))
            pending.append((g, pr, a, va, s, vs, fin))
        for g, pr, a, va, s, vs, fin in pending:
            hids = g["hid"].to_numpy()
            eff, veff = a + s, va + vs
            neff, nveff, om, de, _ = pl_score(eff, veff, fin, hp.beta)
            z = om / np.sqrt(np.maximum(de, 1e-12))
            zq = rank_quantile_residual(eff, veff, fin, hp.beta)
            d_eff, fshr = neff - eff, nveff / veff
            for i, h in enumerate(hids):
                na = a[i] + d_eff[i] * va[i] / veff[i]
                ns = s[i] + d_eff[i] * vs[i] / veff[i]
                st.va[h] = va[i] - (va[i] ** 2 / veff[i]) * (1 - fshr[i])
                st.vs[h] = vs[i] - (vs[i] ** 2 / veff[i]) * (1 - fshr[i])
                st.da[h] = float(na - a[i])
                st.a[h], st.s[h] = float(na), float(ns)
                # innovation_ewma: ew ← ρ·ew_前回 + (1−ρ)·z。pr["ew"] は既に ρ·ew_前回。初走は z そのもの
                if h in st.ew:
                    st.ew[h] = float(pr[i]["ew"] + (1 - pr[i]["rho"]) * z[i])
                else:
                    st.ew[h] = float(z[i])
                st.z[h], st.zq[h] = float(z[i]), float(zq[i])
                st.last[h] = day
                st.n[h] = st.n.get(h, 0) + 1
                innov.append((h, day, g["rid16"].iloc[0], float(z[i]), float(zq[i]), float(fin[i])))
    fdf = pd.concat(feats, ignore_index=True) if feats else pd.DataFrame()
    return fdf, pd.DataFrame(ll, columns=["date", "rid16", "ll_win"]), \
        pd.DataFrame(innov, columns=["hid", "date", "rid16", "z", "zq", "fin"])
