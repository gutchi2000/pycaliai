# -*- coding: utf-8 -*-
"""
synth.py — 馬券判断 AI v1 の学習器検証用の合成データ（事前登録 docs/research/BET_DECISION_AI_LEARNER_CHECK_PREREG_20261023.md）
=============================================================================================================
実データの substrate（arm=main、2017-2022、v1 と同じ data.load_candidates）から **入力（特徴量）と候補券の並びはそのまま**
使い、**払戻だけ** を合成で置き換える。実データの payout 列は読み込み直後に捨て、一切使わない。2023 年以降は読まない
（data.py の唯一の入口が year<=2022 に絞る）。

真の当たり確率 π_i:
  基本:     π_i = (1 − 控除率[券種]) / odds_decision_i            （市場が正しい世界。期待回収率 = 1 − 控除率）
  埋め込み: 部分集合 S の券だけ π_i = boost / odds_decision_i      （boost = 1.20 / 1.10 / 1.05、対照は埋め込み無し）
  当たれば払戻 = odds_decision × 100（複勝は odds_decision = 判断時点の幅の中央）、外れは 0。

レース内の構造:
  - 単勝: 候補 3 頭 + 「その他」から 1 頭だけ勝つ（カテゴリ抽選、確率 π_tan_k と 1 − Σπ）。
  - 複勝: 候補 3 頭は単勝候補と同じ馬（全レースで確認済み）。単勝で勝った馬の複勝は必ず的中。それ以外の馬は
          条件付き確率 r_k = (π_fuku_k − π_tan_k) / (1 − π_tan_k) で独立に的中（周辺確率 = π_fuku_k を保つ）。
  - 馬連・ワイド・馬単・三連複・三連単: **券ごとの独立ベルヌーイ**（単純化。単勝・複勝の結果とも独立）。
共通乱数: 同じ合成の種では、全条件で同じ一様乱数を使う（条件間の差は π の違いだけから生じる）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.bet_decision_ai import data as D

# JRA の控除率（1 − 控除率 = 基本の期待回収率）
TAKEOUT = {"tan": 0.20, "fuku": 0.20, "umaren": 0.225, "wide": 0.225, "umatan": 0.25,
           "sanrenpuku": 0.25, "sanrentan": 0.275}
PI_CAP = 0.97            # π の上限（埋め込みで 1 を超える低オッズ券の切り詰め。切り詰め件数は報告）
TAN_SUM_CAP = 0.995      # 単勝候補 3 頭の π 合計の上限（超えたら按分）

# ---- 部分集合 S の定義（事前登録で固定）
SUBSET_BET_TYPES = ("tan", "fuku")
SUBSET_LPQ_Q = 0.60      # 学習期間 2017-20 の券種ごとの log_p_over_q の 60% 点以上（= 上位 40%）
SUBSET_MIN_FIELD = 12    # n_field_decision ≥ 12

CONDITIONS = {"control": None, "e105": 1.05, "e110": 1.10, "e120": 1.20}
SYNTH_SEEDS = [11, 22, 33]


def load_universe() -> pd.DataFrame:
    """v1 と同じ候補券（2017-2022、arm=main、決済不能レース除外）。実データの払戻・決済列は直ちに捨てる。"""
    cand = D.load_candidates(2017, D.DEV_MAX_YEAR)
    cand = cand.drop(columns=["payout", "settle_status"])
    assert cand["year"].max() <= D.DEV_MAX_YEAR
    cand = cand.sort_values(["rid16", "bet_type", "cand_rank"]).reset_index(drop=True)
    return cand


def subset_edges(cand: pd.DataFrame) -> dict:
    tr = cand[cand["year"].between(*D.TRAIN_YEARS)]
    return {bt: float(np.quantile(tr.loc[tr["bet_type"] == bt, "log_p_over_q"].to_numpy(float), SUBSET_LPQ_Q))
            for bt in SUBSET_BET_TYPES}


def subset_mask(cand: pd.DataFrame, edges: dict) -> np.ndarray:
    m = np.zeros(len(cand), bool)
    for bt in SUBSET_BET_TYPES:
        mm = (cand["bet_type"] == bt).to_numpy()
        m |= mm & (cand["log_p_over_q"].to_numpy(float) >= edges[bt])
    m &= cand["n_field_decision"].astype(float).fillna(0).to_numpy() >= SUBSET_MIN_FIELD
    return m


def true_probs(cand: pd.DataFrame, S: np.ndarray, boost: float | None) -> tuple[np.ndarray, dict]:
    odds = cand["odds_decision"].to_numpy(float)
    base = 1.0 - cand["bet_type"].map(TAKEOUT).to_numpy(float)
    er = base.copy()
    if boost is not None:
        er[S] = boost
    pi = er / odds
    n_cap = int((pi > PI_CAP).sum())
    pi = np.minimum(pi, PI_CAP)
    # 単勝候補 3 頭の合計を 1 未満に
    tan = (cand["bet_type"] == "tan").to_numpy()
    rid = cand["rid16"].to_numpy()
    tsum = pd.Series(pi[tan]).groupby(rid[tan]).transform("sum").to_numpy()
    scale = np.where(tsum > TAN_SUM_CAP, TAN_SUM_CAP / tsum, 1.0)
    n_tan_scaled_races = int(len(np.unique(rid[tan][tsum > TAN_SUM_CAP])))
    pi_t = pi[tan] * scale
    pi[tan] = pi_t
    info = {"n_pi_capped": n_cap, "n_tan_sum_scaled_races": n_tan_scaled_races,
            "design_expected_return_subset": float((pi[S] * odds[S]).mean()) if S.any() else None,
            "design_expected_return_nonsubset": float((pi[~S] * odds[~S]).mean())}
    return pi, info


def race_layout(cand: pd.DataFrame) -> dict:
    """単勝・複勝の (レース × 候補順位) 位置。cand は rid16, bet_type, cand_rank 順に並んでいる前提。"""
    rid = cand["rid16"].to_numpy()
    races, inv = np.unique(rid, return_inverse=True)
    lay = {"races": races, "inv": inv}
    for bt in ("tan", "fuku"):
        m = (cand["bet_type"] == bt).to_numpy()
        idx = np.full((len(races), 3), -1, np.int64)
        r = inv[m]
        k = cand.loc[m, "cand_rank"].to_numpy(int) - 1
        idx[r, k] = np.nonzero(m)[0]
        assert (idx >= 0).all()
        lay[bt] = idx
    # 同じ馬であることの確認
    h = cand["h1"].to_numpy()
    assert (h[lay["tan"]] == h[lay["fuku"]]).all(), "複勝候補と単勝候補の馬が一致しない"
    return lay


def draw_uniforms(n_tickets: int, n_races: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.random(n_races), rng.random(n_tickets)


def draw_outcomes(cand: pd.DataFrame, pi: np.ndarray, lay: dict, U_race: np.ndarray, V: np.ndarray) -> np.ndarray:
    """的中 0/1（券ごと）。単勝はレース内 1 頭、複勝は単勝の勝ち馬と結合、他は独立ベルヌーイ。"""
    hit = (V < pi).astype(np.int8)                     # 独立ベルヌーイ（馬連以下はこれをそのまま使う）
    ti, fi = lay["tan"], lay["fuku"]
    pt = pi[ti]                                        # [R,3]
    cum = np.cumsum(pt, 1)
    u = U_race[:, None]
    win_k = np.where(u < cum[:, 0:1], 0, np.where(u < cum[:, 1:2], 1, np.where(u < cum[:, 2:3], 2, 3)))[:, 0]
    th = np.zeros_like(pt, dtype=np.int8)
    rr = np.arange(len(pt))
    sel = win_k < 3
    th[rr[sel], win_k[sel]] = 1
    hit[ti] = th
    pf = pi[fi]
    r = np.clip((pf - pt) / np.maximum(1.0 - pt, 1e-9), 0.0, 1.0)
    fh = (V[fi] < r).astype(np.int8)
    fh[th == 1] = 1
    hit[fi] = fh
    return hit


def fuku_coupling_clip_count(cand: pd.DataFrame, pi: np.ndarray, lay: dict) -> int:
    """π_fuku < π_tan で複勝の周辺確率が設計値からずれる（π_tan に持ち上がる）券の数。"""
    return int((pi[lay["fuku"]] < pi[lay["tan"]]).sum())


def synthetic_payout(cand: pd.DataFrame, hit: np.ndarray) -> np.ndarray:
    return hit.astype(np.float64) * cand["odds_decision"].to_numpy(float) * 100.0


# ------------------------------------------------------------------ 神託（部分集合と真の確率を知っていて Kelly）
def oracle_stakes(cand: pd.DataFrame, pi: np.ndarray, S: np.ndarray, lay: dict, race_rows: np.ndarray,
                  iters: int = 3000, device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    """race_rows（レース番号の配列）について、部分集合の単勝・複勝（最大 6 券）への配分 s を
    E[log(1 + κ Σ s_i (R_i − 1))] 最大（Σs ≤ 1, s ≥ 0）で求める（単勝 1 頭勝ち＋複勝結合の 32 シナリオで厳密期待値）。
    返り値: s[len(race_rows), 6]（列 0-2 = 単勝 c1..c3、3-5 = 複勝 c1..c3）、期待対数成長[len(race_rows)]。"""
    import torch
    ti, fi = lay["tan"][race_rows], lay["fuku"][race_rows]
    odds = cand["odds_decision"].to_numpy(float)
    pt, pf = pi[ti], pi[fi]
    ot, of = odds[ti], odds[fi]
    allow = np.concatenate([S[ti], S[fi]], 1)                     # [R,6]
    r = np.clip((pf - pt) / np.maximum(1.0 - pt, 1e-9), 0.0, 1.0)
    # シナリオ: 勝ち馬 w ∈ {0,1,2,3=その他} × 複勝 3 ビット
    scen_p, scen_R = [], []
    rest = np.clip(1.0 - pt.sum(1), 0.0, 1.0)
    for w in range(4):
        pw = pt[:, w] if w < 3 else rest
        for bits in range(8):
            f = np.array([(bits >> k) & 1 for k in range(3)])
            if w < 3 and f[w] == 0:
                continue
            p = pw.copy()
            for k in range(3):
                if k == w:
                    continue
                p = p * (r[:, k] if f[k] else (1.0 - r[:, k]))
            Rt = np.zeros_like(ot)
            if w < 3:
                Rt[:, w] = ot[:, w]
            Rf = of * f[None, :]
            scen_p.append(p)
            scen_R.append(np.concatenate([Rt, Rf], 1))
    P = torch.tensor(np.stack(scen_p, 1), dtype=torch.float64, device=device)          # [R,K]
    A = torch.tensor(np.stack(scen_R, 1) - 1.0, dtype=torch.float64, device=device)    # [R,K,6]
    M = torch.tensor(allow, dtype=torch.float64, device=device)
    kap = D.KAPPA
    L = (kap ** 2 * (P[:, :, None] * A ** 2).sum((1, 2)) / 0.9).clamp(min=1e-6)        # 勾配のリプシッツ上界
    step = (1.0 / L)[:, None]

    def proj(x):  # {x ≥ 0, Σx ≤ 1, 非許可 = 0} への射影
        x = x.clamp(min=0) * M
        over = x.sum(1) > 1
        if over.any():
            y = x[over]
            srt, _ = torch.sort(y, 1, descending=True)
            css = srt.cumsum(1) - 1.0
            ind = torch.arange(1, y.shape[1] + 1, device=y.device, dtype=y.dtype)
            cond = srt - css / ind > 0
            rho = cond.sum(1).clamp(min=1)
            theta = css.gather(1, (rho - 1)[:, None]) / rho[:, None]
            x[over] = (y - theta).clamp(min=0) * M[over]
        return x

    def f_and_g(x):
        z = 1.0 + kap * (A * x[:, None, :]).sum(2)                                      # [R,K]
        val = (P * torch.log(z)).sum(1)
        g = kap * (P[:, :, None] * A / z[:, :, None]).sum(1)
        return val, g

    x = torch.zeros_like(M)
    y = x.clone()
    t = 1.0
    for _ in range(iters):   # FISTA
        _, g = f_and_g(y)
        xn = proj(y + step * g)
        tn = (1 + (1 + 4 * t * t) ** 0.5) / 2
        y = xn + ((t - 1) / tn) * (xn - x)
        x, t = xn, tn
    val, _ = f_and_g(x)
    return x.cpu().numpy(), val.cpu().numpy()


def oracle_slot_stakes(T, cand: pd.DataFrame, s6: np.ndarray, lay: dict, race_rows: np.ndarray) -> np.ndarray:
    """神託の 6 券配分を RaceTensor のスロット配置 [R,44] に置く（T.races と lay['races'] の順は同じ前提）。"""
    out = np.zeros((len(race_rows), D.NSLOT), np.float64)
    out[:, D.SLOT0["tan"]:D.SLOT0["tan"] + 3] = s6[:, :3]
    out[:, D.SLOT0["fuku"]:D.SLOT0["fuku"] + 3] = s6[:, 3:]
    return out
