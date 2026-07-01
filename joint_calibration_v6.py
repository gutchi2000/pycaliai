"""
joint_calibration_v6.py
=======================
問1: 「自分の同時確率(PL)は、そもそも連系で正確か？」を測る評価ハーネス。

各券種 (単勝/複勝/馬連/ワイド/馬単/三連複/三連単) の PL 同時確率を、確定着順
(的中=1 / 不的中=0) に対してキャリブレーション測定する。**オッズ不要**(着順のみで
的中判定するので全期間で測れる)。

測定対象:
  - raw PL 確率              … pl_probs.py の厳密 joint
  - Isotonic 補正後          … models/pl_calibrators_{tag}.pkl (valid=2023 で fit 済)
  両方を測ることで「Isotonic がどれだけ補正を背負っているか = 素の joint の歪みの
  大きさ」を可視化する (補正幅大 = Phase 3=PL-NLL に伸びしろ)。

指標:
  - 券種別 ECE (Expected Calibration Error) / MCE / aggregate bias (Σpred - Σactual)
  - reliability: 予測確率 bin × (平均予測, 実的中率, 件数)  ← 単勝・複勝 calibrator と同枠組み
  - Harville bias 兆候: モデル上位馬 (人気馬の代理) を含む組を PL が過大評価しているか
    → 馬連/三連複を「構成馬のうちモデル top3 に入る頭数」で層別し pred vs actual

期間: 真OOS test 2024-2025 + 参考 valid 2023。

出力: reports/joint_calibration_{tag}.json
再利用: --model で差し替え可能 → Phase 3 (PL-NLL/ListMLE) 版を同一指標で比較できる。

実行: python joint_calibration_v6.py --model v6
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import warnings
from datetime import datetime
from itertools import combinations, permutations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import pl_probs as PL
from backtest_pl_ev import (
    all_umaren_mat, all_wide_mat, all_umatan_mat,
    all_fukusho_vec_fast, all_sanrenpuku_tensor,
)

warnings.filterwarnings("ignore")
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
except Exception:
    pass

BASE       = Path(__file__).parent
MASTER_CSV = BASE / "data/master_v2_20130105-20251228.csv"

COL_RID  = "レースID(新/馬番無)"
COL_JYUN = "着順"
COL_BAN  = "馬番"

BET_TYPES = ["tansho", "fukusho", "umaren", "wide", "umatan", "sanrenpuku", "sanrentan"]

# 的中件数 / レース (Σ確率の理論値 = レースあたり期待当たり本数)
THEORY_HITS_PER_RACE = {
    "tansho": 1, "fukusho": 3, "umaren": 1, "wide": 3,
    "umatan": 1, "sanrenpuku": 1, "sanrentan": 1,
}

# 希少事象 (三連単 ~1e-4) から 単勝 (~0.5) までを覆う log 寄り固定 bin。
# 固定 bin にすることでストリーミング集計でき、券種間・モデル間で ECE を比較可能。
BINS = np.array([
    0.0, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3,
    1e-2, 2e-2, 5e-2, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0001,
])
NB = len(BINS) - 1


# ============================================================
# 集計器 (ストリーミング ECE)
# ============================================================
def new_acc():
    return {"sum_p": np.zeros(NB), "sum_y": np.zeros(NB), "cnt": np.zeros(NB)}


def acc_add(acc, p, y):
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    b = np.clip(np.digitize(p, BINS) - 1, 0, NB - 1)
    np.add.at(acc["sum_p"], b, p)
    np.add.at(acc["sum_y"], b, y)
    np.add.at(acc["cnt"], b, 1.0)


def acc_summary(acc):
    cnt = acc["cnt"]
    N = float(cnt.sum())
    if N == 0:
        return None
    with np.errstate(invalid="ignore", divide="ignore"):
        mp = np.where(cnt > 0, acc["sum_p"] / np.maximum(cnt, 1), 0.0)
        my = np.where(cnt > 0, acc["sum_y"] / np.maximum(cnt, 1), 0.0)
    diff = np.abs(mp - my)
    ece = float(np.sum((cnt / N) * diff))
    mce = float(np.max(diff[cnt > 0])) if (cnt > 0).any() else 0.0
    agg_pred = float(acc["sum_p"].sum() / N)
    agg_act = float(acc["sum_y"].sum() / N)
    bins = []
    for b in range(NB):
        if cnt[b] > 0:
            bins.append({
                "lo": float(BINS[b]), "hi": float(BINS[b + 1]),
                "n": int(cnt[b]),
                "pred": round(float(mp[b]), 6),
                "actual": round(float(my[b]), 6),
            })
    return {
        "ece": round(ece, 6), "mce": round(mce, 6), "n": int(N),
        "agg_pred": round(agg_pred, 6), "agg_actual": round(agg_act, 6),
        "bias": round(agg_pred - agg_act, 6),
        "bins": bins,
    }


# ============================================================
# Harville probe (構成馬のモデル top3 メンバー数で層別)
# ============================================================
def new_harv(n_classes):
    return {c: {"sum_p": 0.0, "sum_y": 0.0, "cnt": 0} for c in range(n_classes + 1)}


def harv_add(h, cls_arr, p, y):
    for c in np.unique(cls_arr):
        m = cls_arr == c
        h[int(c)]["sum_p"] += float(p[m].sum())
        h[int(c)]["sum_y"] += float(y[m].sum())
        h[int(c)]["cnt"]   += int(m.sum())


def harv_summary(h):
    out = {}
    for c, d in h.items():
        if d["cnt"] == 0:
            continue
        out[str(c)] = {
            "n": d["cnt"],
            "pred": round(d["sum_p"] / d["cnt"], 6),
            "actual": round(d["sum_y"] / d["cnt"], 6),
            "ratio_pred_over_actual": round(d["sum_p"] / d["sum_y"], 4) if d["sum_y"] > 0 else None,
        }
    return out


# ============================================================
def apply_encoders(df, encs):
    df = df.copy()
    for c, le in encs.items():
        if c not in df.columns:
            continue
        v = df[c].astype(str).fillna("__NaN__")
        v = v.where(v.isin(set(le.classes_)), "__NaN__")
        df[c] = le.transform(v)
    return df


def score_split(df, model, feats, encs, split_label):
    sub = df[df["split"] == split_label].copy()
    sub = apply_encoders(sub, encs)
    X = sub[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    sub["_score"] = model.predict(X)
    return sub


def clip01(x):
    return np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)


def measure_period(period_df, calibrators):
    """1 期間分を測定。returns (bet_summaries, harville_summaries, n_races)."""
    acc = {bt: {"raw": new_acc(), "cal": new_acc()} for bt in BET_TYPES}
    has_cal = {bt: (calibrators is not None and bt in calibrators) for bt in BET_TYPES}
    harv = {"umaren": new_harv(2), "sanrenpuku": new_harv(3)}
    n_races = 0

    for rid, g in period_df.groupby(COL_RID, sort=False):
        if len(g) < 5:
            continue
        g = g.sort_values(COL_BAN).reset_index(drop=True)
        jyun = g[COL_JYUN].astype(int).values
        sc = g["_score"].values
        n = len(sc)
        # 着順 1/2/3 着の局所 index
        order = np.argsort(jyun)
        win, plc, sho = int(order[0]), int(order[1]), int(order[2])
        top3 = {win, plc, sho}
        top2 = {win, plc}
        # モデル上位 (人気馬の代理) top3
        obs = np.argsort(-sc)
        model_top3 = set(int(x) for x in obs[:3])

        w = PL.pl_weights(sc)
        n_races += 1

        def emit(bt, p, y):
            p = np.asarray(p, dtype=np.float64)
            y = np.asarray(y, dtype=np.float64)
            acc_add(acc[bt]["raw"], p, y)
            if has_cal[bt]:
                acc_add(acc[bt]["cal"], clip01(calibrators[bt].predict(p)), y)

        # --- 単勝 ---
        p = w / w.sum()
        y = np.zeros(n); y[win] = 1.0
        emit("tansho", p, y)

        # --- 複勝 ---
        p = all_fukusho_vec_fast(w)
        y = np.array([1.0 if i in top3 else 0.0 for i in range(n)])
        emit("fukusho", p, y)

        # --- 馬連 ---
        M = all_umaren_mat(w)
        iu, ju = np.triu_indices(n, k=1)
        p = M[iu, ju]
        y = np.array([1.0 if {int(iu[k]), int(ju[k])} == top2 else 0.0 for k in range(len(iu))])
        emit("umaren", p, y)
        cls = np.array([(int(iu[k]) in model_top3) + (int(ju[k]) in model_top3) for k in range(len(iu))])
        harv_add(harv["umaren"], cls, p, y)

        # --- ワイド ---
        Wd = all_wide_mat(w)
        p = Wd[iu, ju]
        y = np.array([1.0 if {int(iu[k]), int(ju[k])}.issubset(top3) else 0.0 for k in range(len(iu))])
        emit("wide", p, y)

        # --- 馬単 ---
        U = all_umatan_mat(w)
        ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        m = ii != jj
        iif, jjf = ii[m], jj[m]
        p = U[m]
        y = ((iif == win) & (jjf == plc)).astype(float)
        emit("umatan", p, y)

        # --- 三連単 / 三連複 共通 tensor ---
        P3 = all_sanrenpuku_tensor(w)

        # 三連単
        I, J, K = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
        m3 = (I != J) & (J != K) & (I != K)
        If, Jf, Kf = I[m3], J[m3], K[m3]
        p = P3[m3]
        y = ((If == win) & (Jf == plc) & (Kf == sho)).astype(float)
        emit("sanrentan", p, y)

        # 三連複
        tris = np.array(list(combinations(range(n), 3)))
        if len(tris) > 0:
            p_pk = np.zeros(len(tris))
            for perm in permutations(range(3)):
                p_pk += P3[tris[:, perm[0]], tris[:, perm[1]], tris[:, perm[2]]]
            y = np.array([1.0 if set(int(x) for x in t) == top3 else 0.0 for t in tris])
            emit("sanrenpuku", p_pk, y)
            cls = np.array([sum(int(x) in model_top3 for x in t) for t in tris])
            harv_add(harv["sanrenpuku"], cls, p_pk, y)

    bet_summaries = {}
    for bt in BET_TYPES:
        bet_summaries[bt] = {
            "theory_hits_per_race": THEORY_HITS_PER_RACE[bt],
            "has_calibrator": has_cal[bt],
            "raw": acc_summary(acc[bt]["raw"]),
            "cal": acc_summary(acc[bt]["cal"]) if has_cal[bt] else None,
        }
    harv_summaries = {
        "umaren_by_modeltop3_members": harv_summary(harv["umaren"]),
        "sanrenpuku_by_modeltop3_members": harv_summary(harv["sanrenpuku"]),
    }
    return bet_summaries, harv_summaries, n_races


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="v6", help="model tag (default v6)")
    ap.add_argument("--periods", nargs="+", default=["test", "valid"],
                    help="split labels to measure (default: test valid)")
    args = ap.parse_args()
    tag = args.model

    model_pkl = BASE / f"models/unified_rank_{tag}.pkl"
    cal_pkl   = BASE / f"models/pl_calibrators_{tag}.pkl"
    out_json  = BASE / f"reports/joint_calibration_{tag}.json"

    print("=" * 72)
    print(f"joint_calibration_{tag}.py  (問1: 連系 PL 同時確率の calibration 測定)")
    print("=" * 72)

    bundle = joblib.load(model_pkl)
    model = bundle["model"]; feats = bundle["feature_cols"]; encs = bundle["encoders"]
    print(f"[model] {model_pkl.name}  feats={len(feats)}")

    calibrators = None
    if cal_pkl.exists():
        cb = joblib.load(cal_pkl)
        calibrators = cb.get("calibrators", cb)
        print(f"[cal]   {cal_pkl.name}  keys={sorted(calibrators.keys())}")
    else:
        print(f"[cal]   なし ({cal_pkl.name}) → raw のみ測定")

    print(f"[load]  {MASTER_CSV.name}")
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df = df.dropna(subset=[COL_JYUN, COL_RID, "split"]).copy()

    period_label = {"test": "test_2024_2025", "valid": "valid_2023", "train": "train_le2022"}
    results = {}
    for sp in args.periods:
        print(f"\n[score] split={sp}")
        sub = score_split(df, model, feats, encs, sp)
        print(f"  rows={len(sub):,}  races={sub[COL_RID].nunique():,}  → 測定中...")
        bet_summaries, harv_summaries, n_races = measure_period(sub, calibrators)
        results[period_label.get(sp, sp)] = {
            "n_races": n_races,
            "bet_types": bet_summaries,
            "harville": harv_summaries,
        }

    payload = {
        "model_tag": tag,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "問1: PL joint calibration (raw vs Isotonic) per bet type, true-OOS",
        "labeling": "着順1/2/3着の的中(0/1)。オッズ不要。",
        "bins": BINS.tolist(),
        "bet_types": BET_TYPES,
        "periods": results,
    }
    out_json.parent.mkdir(exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {out_json}")

    # ---- 読みやすいサマリ (報告3点) ----
    for plabel, pdata in results.items():
        print("\n" + "=" * 72)
        print(f"【{plabel}】 races={pdata['n_races']:,}")
        print(f"{'券種':<12s} {'raw_ECE':>9s} {'cal_ECE':>9s} {'raw_bias':>9s} "
              f"{'raw_agg_pred':>13s} {'agg_actual':>11s} {'n':>12s}")
        for bt in BET_TYPES:
            s = pdata["bet_types"][bt]
            r = s["raw"]; c = s["cal"]
            if r is None:
                continue
            cal_ece = f"{c['ece']:.5f}" if c else "   -   "
            print(f"{bt:<12s} {r['ece']:>9.5f} {cal_ece:>9s} {r['bias']:>9.5f} "
                  f"{r['agg_pred']:>13.6f} {r['agg_actual']:>11.6f} {r['n']:>12,}")
        print("\n Harville probe (馬連: 構成馬のうちモデルtop3に入る頭数別, raw):")
        for k, v in pdata["harville"]["umaren_by_modeltop3_members"].items():
            print(f"   top3メンバー={k}: pred={v['pred']:.5f} actual={v['actual']:.5f} "
                  f"pred/actual={v['ratio_pred_over_actual']} (n={v['n']:,})")
        print(" Harville probe (三連複: 構成馬のうちモデルtop3頭数別, raw):")
        for k, v in pdata["harville"]["sanrenpuku_by_modeltop3_members"].items():
            print(f"   top3メンバー={k}: pred={v['pred']:.6f} actual={v['actual']:.6f} "
                  f"pred/actual={v['ratio_pred_over_actual']} (n={v['n']:,})")


if __name__ == "__main__":
    main()
