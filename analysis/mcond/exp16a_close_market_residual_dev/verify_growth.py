# -*- coding: utf-8 -*-
"""
verify_growth.py — INFORMATION_TO_GROWTH_DERIVATION.md の合成例による確認
=========================================================================
実データを使わない純粋な合成シミュレーション。ROI 評価でも馬券生成でもない。
確認すること:
  C1 全資金投入・比例賭けの成長率 W* = D(p||π) + log(1-t) （恒等式）
  C2 現金(no-bet)を許すと、平均 KL < -log(1-t) でも成長率は正になりうる
  C3 pari-mutuel の自己希釈 (賭け金がプールに入る) で成長率が下がる
  C4 判断時点オッズと確定オッズのズレ (drift) で実現成長率が下がる
  C5 最小賭け金単位・端数切り捨て (breakage) の影響
出力: out/derivation_checks.json
実行: python -m analysis.mcond.exp16a_close_market_residual_dev.verify_growth
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
RNG = np.random.default_rng(20260924)
T = 0.20                      # 単勝控除率
BOUND = -np.log(1 - T)        # 0.2231 nats


def make_race(n: int, sharpness: float, market_noise: float):
    """真の p と、それを歪めた市場 π を作る"""
    a = RNG.dirichlet(np.full(n, sharpness))
    p = a / a.sum()
    z = np.log(p) + RNG.normal(0, market_noise, n)
    pi = np.exp(z - z.max())
    pi /= pi.sum()
    return p, pi


def kl(p, q):
    return float(np.sum(p * np.log(np.clip(p, 1e-12, 1) / np.clip(q, 1e-12, 1))))


def growth_full_investment(p, pi, t=T):
    """全額を比例賭け (b = p)。W = Σ p log(p (1-t)/π)"""
    o = (1 - t) / pi
    return float(np.sum(p * np.log(p * o)))


def growth_optimal_with_cash(p, pi, t=T):
    """現金を許す Kelly 最適 (b0 + Σb_i = 1, b_i >= 0)"""
    o = (1 - t) / pi
    n = len(p)

    def neg(b):
        w = b[0] + b[1:] * o
        return -float(np.sum(p * np.log(np.clip(w, 1e-12, None))))

    cons = [{"type": "eq", "fun": lambda b: b.sum() - 1.0}]
    b0 = np.full(n + 1, 1.0 / (n + 1))
    r = minimize(neg, b0, bounds=[(0, 1)] * (n + 1), constraints=cons, method="SLSQP",
                 options={"maxiter": 400, "ftol": 1e-12})
    return -float(r.fun), r.x


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"takeout": T, "bound_-ln(1-t)": BOUND, "seed": 20260924}

    # --- C1: 恒等式の確認 (全資金投入・比例賭け)
    rows = []
    for _ in range(2000):
        p, pi = make_race(RNG.integers(8, 19), 0.8, 0.6)
        w = growth_full_investment(p, pi)
        rows.append((w, kl(p, pi) - BOUND))
    a = np.array(rows)
    res["C1_full_investment_identity"] = {
        "max_abs_diff_W_minus_(KL-bound)": float(np.abs(a[:, 0] - a[:, 1]).max()),
        "mean_KL": float((a[:, 1] + BOUND).mean()),
        "share_of_races_with_KL>bound": float(((a[:, 1] + BOUND) > BOUND).mean()),
        "mean_W": float(a[:, 0].mean()),
        "note": "W = D(p||π) + log(1-t) が数値的に厳密一致することの確認",
    }

    # --- C2: 現金を許すと、平均 KL が閾値未満でも成長しうる
    Ws, Wc, KLs, edge = [], [], [], []
    for _ in range(300):
        p, pi = make_race(RNG.integers(8, 19), 0.8, 0.25)   # 市場がかなり正確 = KL 小
        Ws.append(growth_full_investment(p, pi))
        Wc.append(growth_optimal_with_cash(p, pi)[0])
        KLs.append(kl(p, pi))
        edge.append(float(np.max(p / pi)))
    res["C2_cash_option"] = {
        "mean_KL": float(np.mean(KLs)), "bound": BOUND,
        "mean_KL_below_bound": bool(np.mean(KLs) < BOUND),
        "mean_growth_full_investment": float(np.mean(Ws)),
        "mean_growth_with_cash": float(np.mean(Wc)),
        "share_races_positive_growth_with_cash": float(np.mean(np.array(Wc) > 1e-9)),
        "share_races_with_max_p_over_pi_gt_1/(1-t)": float(np.mean(np.array(edge) > 1 / (1 - T))),
        "note": "現金を持てる場合の参加条件は max_i p_i/π_i > 1/(1-t)。平均 KL と閾値の比較は停止規則にならない",
    }

    # --- C2b: 推定ノイズ付きの選択的参加 (結論を事前固定しない)
    def kelly_cash_subset(q, pi, sel, t=T):
        """信念 q で、選抜された馬 sel にだけ現金込み Kelly。戻り: bet 配分 (index 対応)"""
        o = (1 - t) / pi
        idx = np.where(sel)[0]
        if len(idx) == 0:
            return np.zeros(len(q))
        def neg(b):
            w = b[0] + np.array([b[1 + k] * o[i] for k, i in enumerate(idx)])
            full = np.full(len(q), b[0])
            for k, i in enumerate(idx):
                full[i] = b[0] + b[1 + k] * o[i]
            return -float(np.sum(q * np.log(np.clip(full, 1e-12, None))))
        b0 = np.full(len(idx) + 1, 1.0 / (len(idx) + 1))
        r = minimize(neg, b0, bounds=[(0, 1)] * (len(idx) + 1),
                     constraints=[{"type": "eq", "fun": lambda b: b.sum() - 1.0}],
                     method="SLSQP", options={"maxiter": 300, "ftol": 1e-10})
        out = np.zeros(len(q))
        for k, i in enumerate(idx):
            out[i] = max(r.x[1 + k], 0.0)
        return out

    c2b = {}
    for sd_noise in (0.0, 0.1, 0.2, 0.4, 0.8):
        g_real, entered, stake = [], [], []
        for _ in range(200):
            p, pi = make_race(RNG.integers(8, 19), 0.8, 0.45)
            z = np.log(p) + RNG.normal(0, sd_noise, len(p))      # 推定ノイズ
            q = np.exp(z - z.max()); q /= q.sum()
            sel = (q / pi) > 1 / (1 - T)                          # 信念ベースの選抜
            b = kelly_cash_subset(q, pi, sel)
            cash = max(0.0, 1.0 - b.sum())
            w = cash + b * ((1 - T) / pi)
            g_real.append(float(np.sum(p * np.log(np.clip(w, 1e-12, None)))))   # 真の p で評価
            entered.append(float(sel.any()))
            stake.append(float(b.sum()))
        c2b[f"q_noise_sd_{sd_noise}"] = {
            "mean_growth_under_true_p": float(np.mean(g_real)),
            "share_races_entered": float(np.mean(entered)),
            "mean_stake_fraction": float(np.mean(stake)),
            "share_races_growth_positive": float(np.mean(np.array(g_real) > 1e-9)),
        }
    res["C2b_selection_with_estimation_noise"] = {
        "by_noise": c2b,
        "note": "q=p+推定ノイズ で q_i/π_i 選抜 → 現金込み Kelly → 真の p で期待対数成長を評価。"
                "結論はノイズ水準に依存するので、この表をそのまま報告する",
    }

    # --- C3: pari-mutuel の自己希釈 (賭け金 s がプール P に入る)
    dil = {}
    p, pi = make_race(14, 0.8, 0.6)
    for frac in (0.0, 0.001, 0.005, 0.02, 0.05):
        o_eff = (1 - T) / np.clip(pi + frac * p, 1e-9, None)   # 自分の賭けで各馬のシェアが増える近似
        dil[f"stake_share_{frac}"] = float(np.sum(p * np.log(np.clip(p * o_eff, 1e-12, None))))
    res["C3_self_dilution"] = {"growth_by_stake_share_of_pool": dil,
                               "note": "プールに対する賭け金比率が上がるほど実現オッズが下がり成長率が落ちる"}

    # --- C4: 判断時点オッズ ≠ 確定オッズ
    drift = {}
    for sd in (0.0, 0.05, 0.10, 0.20):
        ws = []
        for _ in range(500):
            p, pi_pre = make_race(RNG.integers(8, 19), 0.8, 0.6)
            z = np.log(pi_pre) + RNG.normal(0, sd, len(pi_pre))
            pi_fin = np.exp(z - z.max()); pi_fin /= pi_fin.sum()
            b = p                                   # 判断時点の確率で比例賭け
            o_fin = (1 - T) / pi_fin                # 実際に貰えるのは確定オッズ
            ws.append(float(np.sum(p * np.log(np.clip(b * o_fin, 1e-12, None)))))
        drift[f"drift_sd_{sd}"] = float(np.mean(ws))
    res["C4_decision_vs_final_odds_symmetric_noise"] = {
        "mean_growth_by_drift_sd": drift,
        "note": "対称で情報を持たない drift は成長率を系統的には下げない (Jensen により僅かに上げることすらある)。"
                "『判断時点と確定が違う』こと自体は害の理由にならない",
    }

    # --- C4b: 市場が真値へ補正する drift (自分のエッジが締切までに消える)
    adverse = {}
    for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
        ws, edges = [], []
        for _ in range(500):
            p, pi_pre = make_race(RNG.integers(8, 19), 0.8, 0.6)
            z = (1 - lam) * np.log(pi_pre) + lam * np.log(p)   # 締切までに市場が p へ寄る
            pi_fin = np.exp(z - z.max()); pi_fin /= pi_fin.sum()
            o_fin = (1 - T) / pi_fin
            ws.append(float(np.sum(p * np.log(np.clip(p * o_fin, 1e-12, None)))))
            edges.append(float(np.max(p / pi_fin)))
        adverse[f"correction_lambda_{lam}"] = {"mean_growth": float(np.mean(ws)),
                                               "mean_max_p_over_pi_final": float(np.mean(edges))}
    res["C4b_adverse_correction_drift"] = {
        "by_lambda": adverse,
        "note": "害になるのは『自分が張る側へ不利に動く drift』。市場が締切までに真値へ寄るほど、"
                "判断時点で見えたエッジは確定オッズでは消える (lambda=1 で成長率は log(1-t) に収束)",
    }

    # --- C5: 最小単位 100 円・端数 (breakage)
    brk = {}
    for unit in (1, 100, 500):
        ws = []
        for _ in range(500):
            p, pi = make_race(RNG.integers(8, 19), 0.8, 0.6)
            b = p * 10000
            b = np.floor(b / unit) * unit
            if b.sum() <= 0:
                continue
            b = b / 10000
            rest = 1 - b.sum()
            o = np.floor(((1 - T) / pi) * 10) / 10          # 公表オッズは 0.1 刻み
            ws.append(float(np.sum(p * np.log(np.clip(rest + b * o, 1e-12, None)))))
        brk[f"unit_{unit}yen_of_10000"] = float(np.mean(ws))
    res["C5_granularity_and_breakage"] = {"mean_growth_by_unit": brk,
                                          "note": "100 円単位と 0.1 刻みオッズの切り捨ては成長率を押し下げる"}

    (OUT / "derivation_checks.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                                encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1)[:2200])


if __name__ == "__main__":
    main()
