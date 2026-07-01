# -*- coding: utf-8 -*-
"""
_policy_substrate.py — 三連複ベッティング・ポリシー探索の共有評価器 (workflow agents 用)
=========================================================================================
build_policy_substrate.py が焼いた data/_policy/sanpuku_substrate.pkl を読み、
任意のポリシー (= 各レースで何点買うか K) を OOS 実払戻で settle する。
すべてベクトル化。エージェントはここを import して policy builder を組むだけ。

使い方:
    import sys; sys.path.insert(0, r"E:\\PyCaLiAI\\analysis")
    import _policy_substrate as P
    sub = P.load()
    m24 = P.year_mask(sub, 2024)
    K = P.floor_K(sub, mult=0.775, kmax=25)      # ユーザーのトリガミ床ルール
    print(P.score(sub, K, mask=m24))             # 2024 OOS の成績

settle 定義 (trigami_floor_gate.py と一致):
    hit   = win_rank <= K
    cost  = K*100,  ret = pay(円/100円) if hit else 0
    trig  = hit かつ pay < cost   (当たったのにマイナス = トリガミ)
    clean = hit かつ pay >= cost  (当たって必ずプラス)
"""
from pathlib import Path
import joblib, numpy as np

BASE = Path(r"E:\PyCaLiAI")
SUB = BASE / "data/_policy/sanpuku_substrate.pkl"
TAKEOUT = 0.775


def load():
    return joblib.load(SUB)


def year_mask(sub, y):
    return sub["year"] == y


def score(sub, K, mask=None):
    """K: 各レースの購入点数 (0 or <1 = そのレース見送り). mask: 母集団 (year 等)."""
    pay = sub["pay"].astype(float); wr = sub["win_rank"]
    N = len(pay)
    m = np.ones(N, bool) if mask is None else np.asarray(mask, bool)
    Ki = np.floor(np.asarray(K, float)).astype(int)
    active = m & (Ki >= 1)
    hit = active & (wr <= Ki)
    cost = np.where(active, Ki * 100, 0).astype(float)
    ret = np.where(hit, pay, 0.0)
    trig = hit & (pay < Ki * 100)
    clean = hit & (pay >= Ki * 100)
    nb = int(active.sum()); hits = int(hit.sum()); tot = int(m.sum()); C = float(cost.sum())
    return {
        "universe": tot, "nbet": nb,
        "participation": round(nb / tot * 100, 1) if tot else 0.0,
        "hit": round(hits / nb * 100, 1) if nb else 0.0,
        "roi": round(ret.sum() / C * 100, 1) if C else 0.0,
        "trig": int(trig.sum()), "clean": int(clean.sum()),
        "trig_rate_of_hit": round(int(trig.sum()) / hits * 100, 1) if hits else 0.0,
        "avg_pts": round(float(cost[active].mean()) / 100, 2) if nb else 0.0,
        "net_yen": int(ret.sum() - C),
    }


def bootstrap_roi_ci(sub, K, mask=None, n=400, seed=0):
    """ROI と trig_rate の race-level ブートストラップ 95%CI (過学習/ノイズ判定用)."""
    pay = sub["pay"].astype(float); wr = sub["win_rank"]
    N = len(pay); m = np.ones(N, bool) if mask is None else np.asarray(mask, bool)
    Ki = np.floor(np.asarray(K, float)).astype(int)
    idx = np.where(m & (Ki >= 1))[0]
    if len(idx) == 0: return {}
    hit = wr[idx] <= Ki[idx]; cost = Ki[idx] * 100.0
    ret = np.where(hit, pay[idx], 0.0); trig = hit & (pay[idx] < cost)
    rng = np.random.default_rng(seed); rois = []; trs = []
    for _ in range(n):
        s = rng.integers(0, len(idx), len(idx))
        c = cost[s].sum()
        rois.append(ret[s].sum() / c * 100 if c else 0)
        h = int(hit[s].sum()); trs.append(int(trig[s].sum()) / h * 100 if h else 0)
    return {"roi_ci": [round(np.percentile(rois, 2.5), 1), round(np.percentile(rois, 97.5), 1)],
            "trig_rate_ci": [round(np.percentile(trs, 2.5), 1), round(np.percentile(trs, 97.5), 1)]}


# ---------------- policy builders (各レース K を返す) ----------------
def fixed_K(sub, K):
    return np.full(len(sub["pay"]), int(K))


def floor_K(sub, mult=TAKEOUT, kmax=25, kmin=1):
    """ユーザーのトリガミ床ルール: 最安組払戻(=mult/p_fav)が点数を超える最大点数。"""
    K = np.floor(mult / np.maximum(sub["p_fav"], 1e-9))
    return np.clip(K, kmin, kmax)


def mass_K(sub, m, kmax=30, kmin=1):
    """累積確率が m に届くまで買う (確率質量ターゲット)."""
    cs = np.cumsum(sub["probs_topK"], axis=1)
    K = (cs < m).sum(axis=1) + 1
    return np.clip(K, kmin, kmax)


def thresh_K(sub, t, kmax=30, kmin=1):
    """確率 t 以上の組だけ買う (確率しきい値)."""
    K = (sub["probs_topK"] >= t).sum(axis=1)
    return np.clip(K, kmin, kmax)


def apply_participation(K, mask_keep):
    """mask_keep=False のレースは見送り (K=0)."""
    K = np.asarray(K, float).copy()
    K[~np.asarray(mask_keep, bool)] = 0
    return K


def chaos_band(sub, lo=0.0, hi=1.0):
    return (sub["chaos"] >= lo) & (sub["chaos"] <= hi)
