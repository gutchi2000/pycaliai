# -*- coding: utf-8 -*-
"""
dyn_skill.py — 多頭数順位からの動的能力と不確実性 (filtering のみ、smoothing なし)
==============================================================================
更新式: Weng & Lin (2011) "A Bayesian Approximation Method for Online Ranking" の
        Plackett–Luce 版 (1レースの全着順を1観測として、各馬の能力 N(mu, var) を閉形式で更新)

  c      = sqrt( Σ_i (var_i + β²) )
  e_i    = exp(mu_i / c)
  S_q    = Σ_{s: rank_s ≥ rank_q} e_s            (q 以下の着順の馬の和)
  A_q    = rank_q と同着の頭数
  Ω_i    = Σ_{q: rank_q ≤ rank_i} ( 1{q=i} − e_i/S_q ) / A_q
  Δ_i    = Σ_{q: rank_q ≤ rank_i} (e_i/S_q)(1 − e_i/S_q) / A_q
  mu_i  += var_i / c · Ω_i
  var_i *= max( 1 − γ·(var_i/c²)·Δ_i , κ )        γ = 1
  ※ γ: OpenSkill 既定の γ=sqrt(var_i)/c だと多頭数 (c≈35) で1走あたりの分散縮小が約1%しかなく、
     休養による増加と釣り合って「履歴が増えても不確実性が減らない」。評価前 (2013 のウォームアップ
     記述統計のみ確認した段階) に γ=1 へ固定した。γ=1 は PL 尤度の Laplace 近似で
     精度に Fisher 情報 Δ_i/c² を足すことに相当する。

時間変化: 予測時点で  var_prior = var_last + τ² × (前回更新からの日数)
初期分布: 全馬共通 N(μ0=25, σ0²), σ0=25/3 (年齢・性別等の階層初期値は使わない)

T2 (条件別部分プーリング):
  条件能力 = 全体 g + 芝ダ効果 s[芝|ダ] + 距離帯効果 d[≤1400|1401-1800|1801-2200|≥2201]
  効果の初期分布 N(0, (σ0/3)²)、時間ドリフトなし。レースでは条件能力 (平均・分散の和) で上の更新を行い、
  平均の変化と分散の縮小を各成分に「分散の比」で配分する (独立ガウスの和への観測と同じ Kalman 分配):
     Δmu_k = Δmu_eff · var_k/var_eff,   var_k' = var_k − (var_k²/var_eff)(1 − f)   (f = var_eff'/var_eff)
  出走の少ない条件は効果の分散が小さいまま → 全体能力へ縮約される。

同日の扱い: その日の全レースを「前日までの状態」から予測し、日の終わりにまとめて更新する。
順位の扱い: master の出走行 (着順≥1) だけを参加者とする。取消・除外・中止は master に無い (kekka では着順0)。
            同着は同順位。失格・降着は確定着順のとおり。着差は使わない。
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[3]
MASTER = BASE / "data/master_v2_20130105-20251228.csv"
MU0 = 25.0
SIGMA0 = 25.0 / 3.0
KAPPA = 1e-4
GAMMA = 1.0
OFF_SIGMA0 = SIGMA0 / 3.0
DIST_BANDS = [1400, 1800, 2200]          # ≤1400 / 1401-1800 / 1801-2200 / ≥2201


@dataclass(frozen=True)
class Hyper:
    beta: float
    tau2_per_day: float


def dist_band(d: float) -> int:
    if d != d:
        return 1
    return int(np.searchsorted(DIST_BANDS, d, side="left"))


def load_runs(max_date: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(MASTER, encoding="utf-8-sig", low_memory=False,
                     usecols=["日付", "レースID(新)", "血統登録番号", "馬番", "着順", "芝・ダ", "距離"])
    df["date"] = pd.to_datetime(df["日付"].astype(str), format="%Y%m%d")
    if max_date is not None:
        df = df[df["date"] <= pd.Timestamp(max_date)]
    df["rid16"] = df["レースID(新)"].astype(str).str[:16]
    df["ban"] = pd.to_numeric(df["馬番"], errors="coerce")
    df["fin"] = pd.to_numeric(df["着順"], errors="coerce")
    df = df.dropna(subset=["ban", "fin", "血統登録番号"])
    df = df[df["fin"] >= 1]
    df["ban"] = df["ban"].astype(int)
    df["hid"] = df["血統登録番号"].astype(str)
    df["surf"] = (df["芝・ダ"].astype(str) == "芝").astype(int)     # 1=芝, 0=ダ
    df["dband"] = pd.to_numeric(df["距離"], errors="coerce").map(dist_band)
    return df[["date", "rid16", "ban", "hid", "fin", "surf", "dband"]].sort_values(
        ["date", "rid16", "ban"]).reset_index(drop=True)


def wl_update(mu: np.ndarray, var: np.ndarray, rank: np.ndarray, beta: float):
    """Weng–Lin PL 更新。新しい (mu, var) を返す。"""
    c = np.sqrt(np.sum(var + beta ** 2))
    e = np.exp((mu - mu.max()) / c)
    ge = rank[None, :] >= rank[:, None]          # [q, s]: rank_s ≥ rank_q
    S = (ge * e[None, :]).sum(axis=1)            # S_q
    A = (rank[None, :] == rank[:, None]).sum(axis=1)  # A_q
    M = rank[:, None] <= rank[None, :]           # [q, i]: rank_q ≤ rank_i
    Q = e[None, :] / S[:, None]                  # [q, i]: e_i / S_q
    eye = np.eye(len(mu))
    omega = (M * (eye - Q) / A[:, None]).sum(axis=0)
    delta = (M * Q * (1 - Q) / A[:, None]).sum(axis=0)
    new_mu = mu + var / c * omega
    eta = GAMMA * (var / c ** 2) * delta
    new_var = var * np.maximum(1 - eta, KAPPA)
    return new_mu, new_var, c


class State:
    def __init__(self, hp: Hyper):
        self.hp = hp
        self.g_mu: dict[str, float] = {}
        self.g_var: dict[str, float] = {}
        self.last: dict[str, pd.Timestamp] = {}
        self.n: dict[str, int] = {}
        self.chg: dict[str, float] = {}
        # T2
        self.t2_g_mu: dict[str, float] = {}
        self.t2_g_var: dict[str, float] = {}
        self.off_mu: dict[tuple, float] = {}
        self.off_var: dict[tuple, float] = {}

    def prior(self, h, day):
        if h not in self.g_mu:
            return MU0, SIGMA0 ** 2, np.nan, 0, np.nan
        dt = (day - self.last[h]).days
        return (self.g_mu[h], self.g_var[h] + self.hp.tau2_per_day * dt, dt, self.n[h], self.chg[h])

    def prior_t2(self, h, day, surf, band):
        if h in self.t2_g_mu:
            dt = (day - self.last[h]).days
            gm, gv = self.t2_g_mu[h], self.t2_g_var[h] + self.hp.tau2_per_day * dt
        else:
            gm, gv = MU0, SIGMA0 ** 2
        sm = self.off_mu.get((h, "s", surf), 0.0)
        sv = self.off_var.get((h, "s", surf), OFF_SIGMA0 ** 2)
        dm = self.off_mu.get((h, "d", band), 0.0)
        dv = self.off_var.get((h, "d", band), OFF_SIGMA0 ** 2)
        return gm, gv, sm, sv, dm, dv


def race_features(hids, mu, var, beta):
    n = len(mu)
    sd = np.sqrt(var)
    c = np.sqrt(np.sum(var + beta ** 2))
    order = (-mu).argsort(kind="stable")
    rank_in = np.empty(n)
    rank_in[order] = np.arange(1, n + 1)
    tot = mu.sum()
    others_mean = (tot - mu) / max(n - 1, 1)
    top3 = np.sort(mu)[::-1][:3].mean()
    z = mu / c
    lse = z.max() + np.log(np.exp(z - z.max()).sum())
    return {
        "dyn_skill_mu": mu, "dyn_skill_sigma": sd, "dyn_skill_conservative": mu - 2 * sd,
        "dyn_skill_rank_in_race": rank_in, "dyn_skill_gap_to_top": mu.max() - mu,
        "dyn_skill_gap_to_field_mean": mu - mu.mean(),
        "field_skill_mean": np.full(n, mu.mean()), "field_skill_std": np.full(n, mu.std()),
        "field_skill_max": np.full(n, mu.max()), "field_skill_top3_mean": np.full(n, top3),
        "field_uncertainty_mean": np.full(n, sd.mean()),
        "horse_skill_minus_field": mu - others_mean,
        "horse_skill_percentile": 1.0 - (rank_in - 1) / max(n - 1, 1),
        "race_difficulty": np.full(n, lse),
        "dyn_pwin": np.exp(z - lse),
    }


def run(df: pd.DataFrame, hp: Hyper, with_t2: bool = True, collect: bool = True):
    """df: load_runs の出力。日単位バッチで予測特徴を作ってから更新する。
    返り値: (特徴 DataFrame, 次走勝ち予測の対数尤度をレースごとに集めた DataFrame)"""
    st = State(hp)
    feats, ll = [], []
    for day, dd in df.groupby("date", sort=True):
        pending = []
        for rid, g in dd.groupby("rid16", sort=False):
            hids = g["hid"].to_numpy()
            pri = [st.prior(h, day) for h in hids]
            mu = np.array([p[0] for p in pri])
            var = np.array([p[1] for p in pri])
            fin = g["fin"].to_numpy(float)
            if collect:
                f = race_features(hids, mu, var, hp.beta)
                f.update({"rid16": np.full(len(g), rid), "ban": g["ban"].to_numpy(), "hid": hids,
                          "date": np.full(len(g), day),
                          "dyn_skill_days_since_update": np.array([p[2] for p in pri], float),
                          "dyn_skill_num_updates": np.array([p[3] for p in pri], float),
                          "dyn_skill_last_change": np.array([p[4] for p in pri], float)})
            win = fin == fin.min()
            ll.append((day, rid, float(np.log(np.clip(race_features(hids, mu, var, hp.beta)["dyn_pwin"][win].mean(), 1e-12, 1)))))
            if with_t2:
                surf, band = int(g["surf"].iloc[0]), int(g["dband"].iloc[0])
                p2 = [st.prior_t2(h, day, surf, band) for h in hids]
                cm = np.array([a[0] + a[2] + a[4] for a in p2])
                cv = np.array([a[1] + a[3] + a[5] for a in p2])
                gmu = np.array([a[0] for a in p2])
                if collect:
                    order = (-cm).argsort(kind="stable")
                    rk = np.empty(len(cm))
                    rk[order] = np.arange(1, len(cm) + 1)
                    f.update({"condition_skill_mu": cm, "condition_skill_sigma": np.sqrt(cv),
                              "condition_skill_minus_global": cm - gmu, "condition_skill_rank_in_race": rk})
            if collect:
                feats.append(pd.DataFrame(f))
            pending.append((g, mu, var, fin, (surf, band, p2) if with_t2 else None))
        # ---- 日の終わりにまとめて更新 (同日の他レースの特徴には影響しない) ----
        for g, mu, var, fin, t2 in pending:
            hids = g["hid"].to_numpy()
            nm, nv, _ = wl_update(mu, var, fin, hp.beta)
            for i, h in enumerate(hids):
                st.chg[h] = float(nm[i] - mu[i])
                st.g_mu[h], st.g_var[h] = float(nm[i]), float(nv[i])
                st.last[h] = day
                st.n[h] = st.n.get(h, 0) + 1
            if t2 is not None:
                surf, band, p2 = t2
                cm = np.array([a[0] + a[2] + a[4] for a in p2])
                cv = np.array([a[1] + a[3] + a[5] for a in p2])
                ncm, ncv, _ = wl_update(cm, cv, fin, hp.beta)
                for i, h in enumerate(hids):
                    gm, gv, sm, sv, dm, dv = p2[i]
                    d_mu, fshr = ncm[i] - cm[i], ncv[i] / cv[i]
                    for kind, key, m, v in (("g", None, gm, gv), ("s", surf, sm, sv), ("d", band, dm, dv)):
                        nm_k = m + d_mu * v / cv[i]
                        nv_k = v - (v * v / cv[i]) * (1 - fshr)
                        if kind == "g":
                            st.t2_g_mu[h], st.t2_g_var[h] = nm_k, nv_k
                        else:
                            st.off_mu[(h, kind, key)], st.off_var[(h, kind, key)] = nm_k, nv_k
    fdf = pd.concat(feats, ignore_index=True) if feats else pd.DataFrame()
    return fdf, pd.DataFrame(ll, columns=["date", "rid16", "ll_win"])
