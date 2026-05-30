"""
backtest_pl_ev.py  (vectorized)
===============================
EV ゲート付き PL backtest。numpy ベクトル化で高速。
"""
from __future__ import annotations

import io
import json
import sys
import warnings
from itertools import combinations, permutations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import pl_probs as PL

warnings.filterwarnings("ignore")
try:
    if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding.lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
except Exception:
    pass

import os
BASE       = Path(__file__).parent
MASTER_CSV = BASE / "data/master_v2_20130105-20251228.csv"
KEKKA_CSV  = BASE / "data/kekka_20130105-20251228.csv"
WIDE_PQ    = BASE / "data/wide_payouts_2016-2025.parquet"
MODEL_PKL  = Path(os.environ.get("UNIFIED_MODEL", BASE / "models/unified_rank_v1.pkl"))
CAL_PKL    = Path(os.environ.get("UNIFIED_CAL",   BASE / "models/pl_calibrators_v1.pkl"))
CURVE_PKL  = Path(os.environ.get("UNIFIED_CURVE", BASE / "data/pl_payout_curve_v1.pkl"))
OUT_JSON   = BASE / "reports/backtest_pl_ev.json"

COL_RID  = "レースID(新/馬番無)"
COL_JYUN = "着順"
COL_BAN  = "馬番"

BET = 100
EV_THRESHOLDS = [80, 90, 100, 105, 110, 120, 130, 150, 200]


def load_payouts():
    df = pd.read_csv(KEKKA_CSV, encoding="cp932", low_memory=False)
    df.columns = ["rid_horse", "ban", "ped", "jyun",
                  "tansho", "fukusho", "wakuren", "umaren",
                  "umatan", "sanrenpuku", "sanrentan"]
    df["rid"] = df["rid_horse"].astype(str).str[:16]
    df["ban"] = pd.to_numeric(df["ban"], errors="coerce").astype("Int64")
    df["jyun"] = pd.to_numeric(df["jyun"], errors="coerce").astype("Int64")
    for c in ["tansho", "fukusho", "umaren", "umatan", "sanrenpuku", "sanrentan"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    out = {}
    for rid, g in df.groupby("rid", sort=False):
        r1 = g[g["jyun"] == 1]; r2 = g[g["jyun"] == 2]; r3 = g[g["jyun"] == 3]
        if len(r1) == 0 or len(r2) == 0 or len(r3) == 0: continue
        w_row, p_row, s_row = r1.iloc[0], r2.iloc[0], r3.iloc[0]
        out[rid] = {
            "win": int(w_row["ban"]), "plc": int(p_row["ban"]), "sho": int(s_row["ban"]),
            "tansho": w_row["tansho"],
            "fuku_win": w_row["fukusho"], "fuku_plc": p_row["fukusho"], "fuku_sho": s_row["fukusho"],
            "umaren": w_row["umaren"], "umatan": w_row["umatan"],
            "sanrenpuku": w_row["sanrenpuku"], "sanrentan": w_row["sanrentan"],
        }
    return out


def load_wide_payouts():
    """wide_payouts parquet → {rid_s: [(ban_i, ban_j, pay), ...]} (最大 3 ペア)."""
    df = pd.read_parquet(WIDE_PQ)
    df["race_id"] = df["race_id"].astype(str)
    out = {}
    for _, r in df.iterrows():
        pairs = []
        for suf in ("w1", "w2", "w3"):
            bi, bj, pay = r[f"{suf}_i"], r[f"{suf}_j"], r[f"{suf}_pay"]
            if pd.isna(bi) or pd.isna(bj) or pd.isna(pay): continue
            pairs.append((int(bi), int(bj), float(pay)))
        if pairs:
            out[r["race_id"]] = pairs
    return out


def apply_encoders(df, encs):
    df = df.copy()
    for c, le in encs.items():
        if c not in df.columns: continue
        v = df[c].astype(str).fillna("__NaN__")
        known = set(le.classes_)
        v = v.where(v.isin(known), "__NaN__")
        df[c] = le.transform(v)
    return df


def lookup_pay_vec(curve, probs, key="median_pay"):
    """ベクトル化 payout lookup. key='mean_pay' (期待値正しい) or 'median_pay'."""
    bins = curve["bins"]
    if key not in curve:
        key = "median_pay"  # 後方互換
    v_tab = curve[key]
    b = np.clip(np.digitize(probs, bins) - 1, 0, len(bins) - 2)
    v = v_tab[b]
    return np.nan_to_num(v, nan=0.0)


# ============================================================
# ベクトル化 PL 計算
# ============================================================
def all_umaren_mat(w):
    """(N,N) 対称. P(umaren i,j)."""
    total = w.sum()
    # P(i→j) = wi/total * wj/(total-wi)
    denom_j = total - w[:, None]  # shape (N,1): total - wi
    P_ij = (w[:, None] / total) * (w[None, :] / denom_j)
    np.fill_diagonal(P_ij, 0)
    return P_ij + P_ij.T


def all_wide_mat(w):
    """(N,N) 対称. P(wide {i,j}) = Σ_{k ∉ {i,j}} P_sanrenpuku({i,j,k}).
    all_sanrenpuku_tensor (三連単順序付き) を対称化して使用. O(N^3), N<=18."""
    n = len(w)
    P3 = all_sanrenpuku_tensor(w)  # 三連単 (順序付き)
    # 6 順列を加算 → Psym[i,j,k] = P_sanrenpuku({i,j,k}) (i,j,k が distinct の場合; self は 0)
    Psym = np.zeros_like(P3)
    for perm in permutations(range(3)):
        Psym += np.transpose(P3, axes=perm)
    # P_wide(i,j) = Σ_k Psym[i,j,k]  (self-terms は P3 で既にゼロ化済みなので安全)
    Pw = Psym.sum(axis=2)
    np.fill_diagonal(Pw, 0)
    return Pw


def all_umatan_mat(w):
    """(N,N). P(i→j)."""
    total = w.sum()
    P = (w[:, None] / total) * (w[None, :] / (total - w[:, None]))
    np.fill_diagonal(P, 0)
    return P


def all_fukusho_vec_fast(w):
    """ベクトル化: 各馬が top3 にいる確率."""
    total = w.sum()
    n = len(w)
    # P(i=1st)
    p1 = w / total
    # P(i=2nd) = Σ_{j≠i} w_j/total * w_i/(total-w_j)
    # = w_i * Σ_{j≠i} w_j / (total * (total-w_j))
    # term_j = w_j / (total-w_j); subtract self
    term = w / (total - w)      # (N,) : w_j/(total-w_j)
    sum_term = term.sum()
    p2 = w * (sum_term - term) / total   # Σ_{j≠i} term_j * w_i / total

    # P(i=3rd) = Σ_{j≠i} Σ_{k≠i,j} w_j/total * w_k/(total-w_j) * w_i/(total-w_j-w_k)
    # 直接 O(N^2) ループ (N≤18 で高速)
    p3 = np.zeros(n)
    for j in range(n):
        pj = w[j] / total
        rem_j = total - w[j]
        for k in range(n):
            if k == j: continue
            pk = w[k] / rem_j
            rem_jk = rem_j - w[k]
            # i ≠ j,k について p_i = w_i / rem_jk
            # 全 i に対して一括加算
            contrib = pj * pk * w / rem_jk
            contrib[j] = 0; contrib[k] = 0
            p3 += contrib
    return p1 + p2 + p3


def all_sanrenpuku_tensor(w):
    """ランク 3 の三連複. returns dict {(i,j,k): p} but as vectorized array."""
    n = len(w)
    total = w.sum()
    # 三連単 p(i,j,k) = wi/T * wj/(T-wi) * wk/(T-wi-wj)
    # 三連複 = 6 * 平均 over 順列? no, sum of 6 perms.
    # 対称化した合計を直接計算:
    # ベクトル化: まず 3 連単 tensor を作る (N,N,N)
    # N=18 で 18^3 = 5832, memory OK
    W = w
    T = total
    # P3[i,j,k] = W_i/T * W_j/(T-W_i) * W_k/(T-W_i-W_j)
    denom_ij = T - W[:, None] - W[None, :]   # (N,N)
    denom_ij = np.where(denom_ij > 0, denom_ij, 1.0)
    P3 = (W[:, None, None] / T) * (W[None, :, None] / (T - W[:, None, None])) * \
         (W[None, None, :] / denom_ij[:, :, None])
    # ゼロアウト: i==j or j==k or i==k
    idx = np.arange(n)
    P3[idx, idx, :] = 0; P3[idx, :, idx] = 0; P3[:, idx, idx] = 0
    return P3  # 三連単確率 tensor (順序あり)


# ============================================================
def score_test(include_valid=False):
    """test split を score 化. include_valid=True で valid(2023) も含める (固定戦略 OOS 評価用)."""
    bundle = joblib.load(MODEL_PKL)
    model, feats, encs = bundle["model"], bundle["feature_cols"], bundle["encoders"]
    rr_mode = bundle.get("race_relative_mode")
    ca_mode = bundle.get("course_affinity_mode")
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df = df.dropna(subset=[COL_JYUN, COL_RID, "split"]).copy()
    if rr_mode:
        from race_relative_feats import add_race_relative_feats
        df = add_race_relative_feats(df, mode=rr_mode)
    if ca_mode:
        from course_affinity_feats import add_course_affinity_feats
        df = add_course_affinity_feats(df, mode=ca_mode)
    if include_valid:
        te = df[df["split"].isin(["test", "valid"])].copy()
    else:
        te = df[df["split"] == "test"].copy()
    te["year"] = te[COL_RID].astype(str).str[:4].astype(int)
    te = apply_encoders(te, encs)
    X = te[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    te["_score"] = model.predict(X)
    return te


def simulate(te, payouts, cal, curves, thresholds):
    stats = {t: {b: {"bets": 0, "hits": 0, "stake": 0, "ret": 0.0}
                 for b in ["tansho","fukusho","umaren","umatan","sanrenpuku","sanrentan"]}
             for t in thresholds}
    n_race = 0
    n_skip = 0

    for rid, g in te.groupby(COL_RID, sort=False):
        if len(g) < 3: n_skip += 1; continue
        rid_s = str(int(rid)) if isinstance(rid, (int, np.integer)) else str(rid)
        if rid_s not in payouts: n_skip += 1; continue
        po = payouts[rid_s]
        g = g.sort_values(COL_BAN).reset_index(drop=True)
        ban = g[COL_BAN].astype(int).values
        b2i = {int(b): i for i, b in enumerate(ban)}
        wi = b2i.get(po["win"]); pi_ = b2i.get(po["plc"]); si = b2i.get(po["sho"])
        if wi is None or pi_ is None or si is None: n_skip += 1; continue
        w = PL.pl_weights(g["_score"].values)
        n = len(w)
        n_race += 1

        # ========= 単勝 =========
        p_raw = w / w.sum()
        p_cal = cal["tansho"].predict(p_raw)
        pay = lookup_pay_vec(curves["tansho"], p_cal)
        ev = p_cal * pay
        hits_mask = np.zeros(n, dtype=bool); hits_mask[wi] = True
        actual = np.zeros(n); actual[wi] = float(po["tansho"]) if po["tansho"] and not pd.isna(po["tansho"]) else 0.0
        for t in thresholds:
            gate = ev >= t
            s = stats[t]["tansho"]
            nb = int(gate.sum())
            s["bets"] += nb; s["stake"] += nb * BET
            s["hits"] += int((gate & hits_mask).sum())
            s["ret"]  += float((actual * (gate & hits_mask)).sum())

        # ========= 複勝 =========
        p_raw = all_fukusho_vec_fast(w)
        p_cal = cal["fukusho"].predict(p_raw)
        pay = lookup_pay_vec(curves["fukusho"], p_cal)
        ev = p_cal * pay
        hits_mask = np.zeros(n, dtype=bool); hits_mask[wi] = True; hits_mask[pi_] = True; hits_mask[si] = True
        actual = np.zeros(n)
        for idx, k in [(wi, "fuku_win"), (pi_, "fuku_plc"), (si, "fuku_sho")]:
            if po[k] and not pd.isna(po[k]):
                actual[idx] = float(po[k])
        for t in thresholds:
            gate = ev >= t
            s = stats[t]["fukusho"]
            nb = int(gate.sum())
            s["bets"] += nb; s["stake"] += nb * BET
            s["hits"] += int((gate & hits_mask).sum())
            s["ret"]  += float((actual * (gate & hits_mask)).sum())

        # ========= 馬連 =========
        Pmat = all_umaren_mat(w)
        iu, ju = np.triu_indices(n, k=1)
        p_raw = Pmat[iu, ju]
        p_cal = cal["umaren"].predict(p_raw)
        pay = lookup_pay_vec(curves["umaren"], p_cal)
        ev = p_cal * pay
        win_pair = {wi, pi_}
        hit_arr = np.array([({int(iu[k]), int(ju[k])} == win_pair) for k in range(len(iu))])
        umaren_pay = float(po["umaren"]) if po["umaren"] and not pd.isna(po["umaren"]) else 0.0
        for t in thresholds:
            gate = ev >= t
            s = stats[t]["umaren"]
            nb = int(gate.sum())
            s["bets"] += nb; s["stake"] += nb * BET
            h = int((gate & hit_arr).sum())
            s["hits"] += h; s["ret"] += h * umaren_pay

        # ========= 馬単 =========
        Umat = all_umatan_mat(w)
        ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        mask = ii != jj
        p_raw = Umat[mask]
        p_cal = cal["umatan"].predict(p_raw)
        pay = lookup_pay_vec(curves["umatan"], p_cal)
        ev = p_cal * pay
        ii_f = ii[mask]; jj_f = jj[mask]
        hit_arr = (ii_f == wi) & (jj_f == pi_)
        umatan_pay = float(po["umatan"]) if po["umatan"] and not pd.isna(po["umatan"]) else 0.0
        for t in thresholds:
            gate = ev >= t
            s = stats[t]["umatan"]
            nb = int(gate.sum())
            s["bets"] += nb; s["stake"] += nb * BET
            h = int((gate & hit_arr).sum())
            s["hits"] += h; s["ret"] += h * umatan_pay

        # ========= 三連単 / 三連複 共通 tensor =========
        P3 = all_sanrenpuku_tensor(w)  # (N,N,N) 三連単順序付き

        # ---- 三連単 ----
        # 全 i!=j!=k の組を一括
        idx = np.arange(n)
        I, J, K = np.meshgrid(idx, idx, idx, indexing="ij")
        mask = (I != J) & (J != K) & (I != K)
        p_raw = P3[mask]
        p_cal = cal["sanrentan"].predict(p_raw)
        pay = lookup_pay_vec(curves["sanrentan"], p_cal)
        ev = p_cal * pay
        I_f = I[mask]; J_f = J[mask]; K_f = K[mask]
        hit_arr = (I_f == wi) & (J_f == pi_) & (K_f == si)
        sanrentan_pay = float(po["sanrentan"]) if po["sanrentan"] and not pd.isna(po["sanrentan"]) else 0.0
        for t in thresholds:
            gate = ev >= t
            s = stats[t]["sanrentan"]
            nb = int(gate.sum())
            s["bets"] += nb; s["stake"] += nb * BET
            h = int((gate & hit_arr).sum())
            s["hits"] += h; s["ret"] += h * sanrentan_pay

        # ---- 三連複 ----
        # combinations(range(n),3) を事前計算して i<j<k で sum of 6 perms
        tris = np.array(list(combinations(range(n), 3)))
        if len(tris) > 0:
            p_pk = np.zeros(len(tris))
            for perm in permutations(range(3)):
                p_pk += P3[tris[:, perm[0]], tris[:, perm[1]], tris[:, perm[2]]]
            p_cal = cal["sanrenpuku"].predict(p_pk)
            pay = lookup_pay_vec(curves["sanrenpuku"], p_cal)
            ev = p_cal * pay
            win_tri = {wi, pi_, si}
            hit_arr = np.array([set(t) == win_tri for t in tris])
            sp_pay = float(po["sanrenpuku"]) if po["sanrenpuku"] and not pd.isna(po["sanrenpuku"]) else 0.0
            for th in thresholds:
                gate = ev >= th
                s = stats[th]["sanrenpuku"]
                nb = int(gate.sum())
                s["bets"] += nb; s["stake"] += nb * BET
                h = int((gate & hit_arr).sum())
                s["hits"] += h; s["ret"] += h * sp_pay

        if n_race % 500 == 0:
            print(f"  ..{n_race} races done")

    print(f"  [done] races={n_race}  skipped={n_skip}")
    return stats


def main():
    print("=" * 60)
    print("backtest_pl_ev.py (vectorized)  test=2024-2025")
    print("=" * 60)

    payouts = load_payouts()
    print(f"[payouts] races={len(payouts):,}")
    cal = joblib.load(CAL_PKL)["calibrators"]
    curves = joblib.load(CURVE_PKL)["curves"]
    te = score_test()
    print(f"[test] rows={len(te):,}  races={te[COL_RID].nunique():,}")

    all_results = {}
    for label, mask in [
        ("2024", te["year"] == 2024),
        ("2025", te["year"] == 2025),
        ("2024+2025", te["year"].isin([2024, 2025])),
    ]:
        print(f"\n=== {label} ===")
        sub = te[mask]
        stats = simulate(sub, payouts, cal, curves, EV_THRESHOLDS)
        year_row = {}
        for t in EV_THRESHOLDS:
            print(f"\n[EV >= {t}]")
            bet_row = {}
            for bet, d in stats[t].items():
                if d["bets"] == 0:
                    bet_row[bet] = {"bets": 0, "hits": 0, "stake": 0, "ret": 0, "roi": 0, "hit_rate": 0}
                    print(f"  {bet:12s}  (no bets)")
                    continue
                roi = d["ret"] / d["stake"]
                hr  = d["hits"] / d["bets"]
                bet_row[bet] = {
                    "bets": d["bets"], "hits": d["hits"],
                    "stake": d["stake"], "ret": int(d["ret"]),
                    "roi": round(roi, 4), "hit_rate": round(hr, 4),
                }
                print(f"  {bet:12s}  bets={d['bets']:>7,}  hits={d['hits']:>4,} "
                      f"({hr*100:5.2f}%)  stake={d['stake']:>10,}  ret={int(d['ret']):>10,}  ROI={roi*100:7.2f}%")
            year_row[t] = bet_row
        all_results[label] = year_row

    OUT_JSON.parent.mkdir(exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "thresholds": EV_THRESHOLDS,
            "results": all_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {OUT_JSON}")


if __name__ == "__main__":
    main()
