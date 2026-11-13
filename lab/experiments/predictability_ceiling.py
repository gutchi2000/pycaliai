# -*- coding: utf-8 -*-
"""
predictability_ceiling.py — 競馬の予測可能性の天井と、市場がそのどこまで来ているか (2026-11-14)
=====================================================================================
理論: 各走の出来 y = θ (その日の本当の能力) + ε (当日の運: 展開・不利・ゲート・測定誤差)。
      着順は y の順でほぼ決まる (JRDB IDM 1位が勝つのは 90%)。
      能力 θ を完全に知る「能力オラクル」でも ε は読めない → 勝ち馬予測の対数損失には下限がある。
      一様 (log K) → 朝の市場 → 確定市場 → 能力オラクル と並べれば、
        - 市場が「知りうる情報」の何割を取り込んでいるか
        - 市場がまだ取り逃がしている情報の量 (nats/レース)
      が出る。0 に近ければ、どんな新理論も市場を越えようがない。大きければ探す価値が残っている。

推定:
  1. バリオグラム: 同じ馬の k 走離れた IDM の差の分散 V(k) = 2σ² + q·k (σ²=運、q=能力の変化)。k=1..4 を直線で当て、
     切片 (nugget) から σ² を出す。2013-2023、芝ダ同じ・連続間隔 ≤120 日・距離差 ≤400m。
  2. レース内の能力の幅: τ_r² = (そのレースの完走馬の IDM 分散) − σ²。頑健版は MAD で。
  3. 検証 (物差しが正しいか): 「前走 IDM だけ知っている予想者」をモデルで再現した NLL と、
     実データで softmax(β·前走IDM) が出した NLL を同じレース集合で比べる。合わなければ σ を合うように較正。
  4. 能力オラクルの NLL = E_θ[H(P(勝ち | θ))]、P は Thurstone (正規の運) を Gauss–Hermite で厳密計算。

保守性: IDM の測定誤差・レース単位の補正誤差・展開/枠など事前に読める要素は全部 ε に入る → σ を過大に見る方向。
        したがってオラクルの NLL は「実際に到達可能な下限」より高め (= 市場が取り逃がす量は控えめに出る)。
入力: data/jrdb/sed.parquet (JRDB、git 管理外。データ値は出力しない)
実行: E:/PyCaLiAI/venv311/Scripts/python.exe lab/experiments/predictability_ceiling.py
出力: reports/predictability_ceiling.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import log_ndtr

SED = Path("E:/PyCaLiAI/data/jrdb/sed.parquet")
OUT = Path(__file__).resolve().parents[2] / "reports" / "predictability_ceiling.json"
T0 = time.time()
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(40)       # ∫ φ(z) f(z) dz = Σ w f(x) / sqrt(2π)
GH_W = GH_W / np.sqrt(2 * np.pi)
S_DRAW = 48
RNG = np.random.default_rng(0)


def log(m):
    print(f"[{time.time() - T0:5.0f}s] {m}", flush=True)


def thurstone_p(m: np.ndarray, s: float) -> np.ndarray:
    """m: (S,K) 平均、共通の運の sd s → (S,K) 勝率。P_i = ∫φ(z) Π_{j≠i} Φ((m_i−m_j)/s + z) dz"""
    d = (m[:, :, None] - m[:, None, :]) / s                         # (S,K,K)
    L = log_ndtr(d[..., None] + GH_X)                                 # (S,K,K,G)
    K = m.shape[1]
    L[:, np.arange(K), np.arange(K), :] = 0.0
    p = (np.exp(L.sum(2)) * GH_W).sum(-1)                             # (S,K)
    return p / p.sum(1, keepdims=True)


def load():
    d = pd.read_parquet(SED, columns=["racekey", "ymd", "ban", "ped", "dist", "td", "jyun", "ijou", "idm", "tan_odds", "odds10_tan"])
    d = d[d.td.isin([1, 2])].copy()
    d["date"] = pd.to_datetime(d.ymd.astype(str), format="%Y%m%d", errors="coerce")
    d["rid"] = d.ymd.astype(str) + "_" + d.racekey.astype(str)
    d["year"] = d.date.dt.year
    return d


def variogram(d):
    x = d[(d.ijou == 0) & d.idm.notna() & (d.year <= 2023)].sort_values(["ped", "date"])
    g = x.groupby("ped")
    out = {}
    for k in (1, 2, 3, 4):
        y0, yk = x.idm.values, g.idm.shift(-k).values
        ok = g.td.shift(-k).values == x.td.values
        span = (g.date.shift(-k) - x.date).dt.days.values
        dd = np.abs(g.dist.shift(-k).values - x.dist.values)
        m = ok & (span <= 120 * k) & (span > 0) & (dd <= 400) & ~np.isnan(yk)
        diff = (yk - y0)[m]
        out[k] = (float(np.var(diff)), int(m.sum()))
    ks = np.array(list(out)); V = np.array([out[k][0] for k in ks])
    b, a = np.polyfit(ks, V, 1)
    return out, float(a), float(b)


def eval_races(d):
    """2024-25、全頭に確定・朝オッズ、勝ち 1 頭、全頭に 365 日以内の前走 IDM がある 5 頭以上のレース"""
    x = d.sort_values(["ped", "date"]).copy()
    g = x.groupby("ped")
    x["prev_idm"] = g.idm.shift(1)
    x["prev_gap"] = (x.date - g.date.shift(1)).dt.days
    x.loc[(x.prev_gap > 365) | (g.ijou.shift(1) != 0), "prev_idm"] = np.nan
    e = x[x.year.between(2023, 2025)].copy()
    e = e[(e.tan_odds > 0) & (e.odds10_tan > 0)]
    r = e.groupby("rid")
    ok = (r.jyun.transform(lambda s: (s == 1).sum()) == 1) & r.prev_idm.transform(lambda s: s.notna().all()) & (r.ban.transform("size") >= 5)
    e = e[ok].sort_values(["rid", "ban"]).reset_index(drop=True)
    return e


def race_nll_softmax(e, col, beta, logit=False):
    z = beta * (np.log(1 / e[col]) if logit else e[col])
    z = z - z.groupby(e.rid).transform("max")
    lp = z - np.log(np.exp(z).groupby(e.rid).transform("sum"))
    return float(-lp[e.jyun == 1].mean())


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    d = load(); log(f"SED 平地 {len(d):,} 走")
    vg, a, b = variogram(d)
    sig2_nug = a / 2; V1 = vg[1][0]
    log("バリオグラム " + "  ".join(f"V({k})={v:.1f} (n={n:,})" for k, (v, n) in vg.items()) + f"  → nugget 2σ²={a:.1f}, q={b:.2f}/走, σ={np.sqrt(sig2_nug):.2f}")

    e = eval_races(d)
    tr, te = e[e.year == 2023], e[e.year >= 2024].reset_index(drop=True)
    log(f"評価レース 2024-25: {te.rid.nunique():,} (fit 用 2023: {tr.rid.nunique():,})")
    res = {"variogram": {str(k): v for k, v in vg.items()}, "nugget_2sig2": a, "drift_q": b, "sigma_nugget": float(np.sqrt(sig2_nug))}

    # --- 市場・一様・前走 IDM の実測 NLL (2024-25)
    K_r = te.groupby("rid").size()
    res["nll_uniform"] = float(np.log(K_r).mean())
    res["nll_mkt_final"] = race_nll_softmax(te, "tan_odds", 1.0, logit=True)
    res["nll_mkt_10am"] = race_nll_softmax(te, "odds10_tan", 1.0, logit=True)
    bp = minimize_scalar(lambda bb: race_nll_softmax(tr, "prev_idm", bb), bounds=(0.001, 1.0), method="bounded").x
    res["nll_prev_idm_emp"] = race_nll_softmax(te, "prev_idm", bp); res["beta_prev"] = float(bp)
    log(f"実測 NLL: 一様 {res['nll_uniform']:.4f}  前走IDM {res['nll_prev_idm_emp']:.4f}  朝市場 {res['nll_mkt_10am']:.4f}  確定市場 {res['nll_mkt_final']:.4f}")

    # --- レース内の能力の幅
    fin = te[(te.ijou == 0) & te.idm.notna()]
    s2 = fin.groupby("rid").idm.var()
    mad = fin.groupby("rid").idm.apply(lambda s: (1.4826 * np.median(np.abs(s - s.median()))) ** 2)
    rids = K_r.index.values; Ks = K_r.values

    def simulate(sig2, spread, n_sub=None):
        """σ² と レース内分散の系列 → (オラクル NLL, 前走予想者 NLL) をレース平均で"""
        q = max(V1 - 2 * sig2, 0.0)
        o_l, p_l = [], []
        idx = np.arange(len(rids)) if n_sub is None else RNG.choice(len(rids), n_sub, replace=False)
        for i in idx:
            K = Ks[i]; tau2 = max(spread.get(rids[i], np.nan) - sig2, 1e-6) if not np.isnan(spread.get(rids[i], np.nan)) else 1e-6
            th = RNG.normal(0, np.sqrt(tau2), (S_DRAW, K))
            pt = thurstone_p(th, np.sqrt(sig2))
            o_l.append(float(-(pt * np.log(np.clip(pt, 1e-300, None))).sum(1).mean()))
            eta2 = sig2 + q
            y = th + RNG.normal(0, np.sqrt(eta2), (S_DRAW, K))
            R = tau2 / (tau2 + eta2)
            pp = thurstone_p(R * y, np.sqrt(R * eta2 + sig2))
            p_l.append(float(-(pt * np.log(np.clip(pp, 1e-300, None))).sum(1).mean()))
        return float(np.mean(o_l)), float(np.mean(p_l))

    res["scenarios"] = {}
    s2d = s2.to_dict(); madd = mad.to_dict()
    for name, sig2, spread in (("nugget/分散", sig2_nug, s2d), ("nugget/MAD", sig2_nug, madd), ("上限V1/2/分散", V1 / 2, s2d)):
        o, p = simulate(sig2, spread)
        res["scenarios"][name] = {"sigma2": sig2, "nll_oracle": o, "nll_prev_sim": p}
        log(f"[{name}] σ={np.sqrt(sig2):.2f}  前走予想者: モデル {p:.4f} vs 実測 {res['nll_prev_idm_emp']:.4f}  → 能力オラクル {o:.4f}")

    # --- 較正: 前走予想者のモデル NLL が実測と一致する σ² (分散版、部分標本で探索 → 全レースで確定)
    target = res["nll_prev_idm_emp"]
    lo, hi = 0.3 * sig2_nug, min(V1 / 2, 3 * sig2_nug)
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        _, p = simulate(mid, s2d, n_sub=800)
        lo, hi = (mid, hi) if p < target else (lo, mid)
    sig2_cal = 0.5 * (lo + hi)
    o, p = simulate(sig2_cal, s2d)
    res["scenarios"]["較正(前走予想者を実測に合わせる)/分散"] = {"sigma2": sig2_cal, "nll_oracle": o, "nll_prev_sim": p}
    log(f"[較正] σ={np.sqrt(sig2_cal):.2f}  前走予想者 モデル {p:.4f} (実測 {target:.4f})  → 能力オラクル {o:.4f}")

    U, M, M10 = res["nll_uniform"], res["nll_mkt_final"], res["nll_mkt_10am"]
    for name, sc in res["scenarios"].items():
        O = sc["nll_oracle"]
        sc["market_captured_share"] = (U - M) / (U - O)
        sc["market_gap_nats"] = M - O
        sc["am10_to_final_share_of_gap"] = (M10 - M) / (M10 - O)
        log(f"  {name}: 知りうる情報 {U - O:.3f} nats のうち 確定市場 {100 * sc['market_captured_share']:.1f}%  取り逃がし {M - O:+.3f} nats/R  "
            f"(朝→確定で埋まるのは朝時点の残りの {100 * sc['am10_to_final_share_of_gap']:.0f}%)")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"saved {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
