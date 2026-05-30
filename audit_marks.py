"""
audit_marks.py
==============
モデルの「印 + 確率」品質を v4/v5 共通フォーマットで測定する。

v5 採用判定の絶対基準:
  - ◎ (top-1) 1着率, 連対率, 複勝圏率
  - 〇 (top-2) 1着率, 連対率, 複勝圏率
  - ▲ (top-3) 複勝圏率
  - △1 (top-4) 複勝圏率
  - △2 (top-5) 複勝圏率
  - top-3 含有率: 1着馬 ∈ top-3 (◎〇▲)
  - top-5 含有率: 1着馬 ∈ top-5 (◎〇▲△△)
  - top-3 全頭含有率: {1,2,3 着} ⊂ top-5
  - NDCG@5
  - p_tansho ECE (キャリブレーション)
  - p_fukusho ECE
  - p_umaren ECE (◎-〇 ペアのみ)

使い方:
  python audit_marks.py --model v4
  python audit_marks.py --model v5

出力:
  reports/audit_marks_<tag>.json (機械可読)
  reports/audit_marks_<tag>.log (人間可読、表形式)
"""
from __future__ import annotations
import argparse
import io
import json
import sys
import warnings
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

import pl_probs as PL
from backtest_pl_ev import (
    load_payouts, score_test,
    all_umaren_mat, all_fukusho_vec_fast,
    COL_RID, COL_BAN,
)
import backtest_pl_ev as be

warnings.filterwarnings("ignore")
try:
    if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding.lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
except Exception:
    pass

BASE = Path(__file__).parent


def ndcg_at_k(label_arr, score_arr, k=5):
    """Normalized DCG @ k. label = relevance (高いほど上位)."""
    n = len(label_arr)
    if n < k:
        k = n
    order = np.argsort(-score_arr)[:k]
    gains = (2 ** label_arr[order] - 1)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (gains * discounts).sum()
    ideal_order = np.argsort(-label_arr)[:k]
    igains = (2 ** label_arr[ideal_order] - 1)
    idcg = (igains * discounts).sum()
    return float(dcg / idcg) if idcg > 0 else 0.0


def expected_calibration_error(probs, labels, n_bins=10):
    """ECE: 予測確率を 10 ビンに分け、各ビンで |mean_pred - hit_rate| を加重平均."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(probs, bins) - 1, 0, n_bins - 1)
    total = len(probs)
    ece = 0.0
    bin_stats = []
    for b in range(n_bins):
        m = bin_idx == b
        if m.sum() == 0:
            continue
        mean_pred = probs[m].mean()
        actual = labels[m].mean()
        weight = m.sum() / total
        ece += weight * abs(mean_pred - actual)
        bin_stats.append({
            "bin_lo": float(bins[b]), "bin_hi": float(bins[b+1]),
            "n": int(m.sum()),
            "mean_pred": float(mean_pred),
            "actual_rate": float(actual),
        })
    return float(ece), bin_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="v4", help="model tag: v1/v3/v4/v5")
    args = ap.parse_args()
    tag = args.model

    be.MODEL_PKL = BASE / f"models/unified_rank_{tag}.pkl"
    be.CAL_PKL   = BASE / f"models/pl_calibrators_{tag}.pkl"
    be.CURVE_PKL = BASE / f"data/pl_payout_curve_{tag}.pkl"
    assert be.MODEL_PKL.exists(), f"{be.MODEL_PKL} not found"
    print(f"[model] tag={tag}  {be.MODEL_PKL}")

    payouts = load_payouts()
    print(f"[payouts] races={len(payouts):,}")

    te = score_test(include_valid=True)
    print(f"[test+valid] rows={len(te):,}  races={te[COL_RID].nunique():,}")

    # PL calibrator (もしあれば使う)
    cal = None
    if be.CAL_PKL.exists():
        cal = joblib.load(be.CAL_PKL)
        print(f"[cal] {be.CAL_PKL}")

    # ===== レース単位で各指標を集計 =====
    rows = []
    for rid, g in te.groupby(COL_RID, sort=False):
        if len(g) < 5:
            continue
        rid_s = str(int(rid)) if isinstance(rid, (int, np.integer)) else str(rid)
        if rid_s not in payouts:
            continue
        po = payouts[rid_s]
        g = g.sort_values(COL_BAN).reset_index(drop=True)
        ban = g[COL_BAN].astype(int).values
        b2i = {int(b): i for i, b in enumerate(ban)}
        wi = b2i.get(po["win"]); pi_ = b2i.get(po["plc"]); si = b2i.get(po["sho"])
        if wi is None or pi_ is None or si is None:
            continue
        scores = g["_score"].values
        w = PL.pl_weights(scores)
        n = len(w)
        order = np.argsort(-scores)
        # 印 (◎=0, 〇=1, ▲=2, △1=3, △2=4)
        hon = int(order[0])
        tai = int(order[1])
        san = int(order[2])
        del1 = int(order[3])
        del2 = int(order[4])
        top3 = {hon, tai, san}
        top5 = {hon, tai, san, del1, del2}
        actual_top3 = {wi, pi_, si}

        # PL 確率 (raw → calibrated if cal available)
        p_tansho_vec = PL.all_tansho(w)
        p_fukusho_vec = all_fukusho_vec_fast(w)
        Pmat = all_umaren_mat(w)
        # PL 馬連 ◎-〇
        p_umaren_hono = float(Pmat[hon, tai])
        # calibrator 適用 (cowork に渡す確率値と同じスケール)
        if cal is not None and "calibrators" in cal:
            cals = cal["calibrators"]
            p_tansho_vec = np.clip(cals["tansho"].predict(p_tansho_vec), 0.0, 1.0)
            p_fukusho_vec = np.clip(cals["fukusho"].predict(p_fukusho_vec), 0.0, 1.0)
            p_umaren_hono = float(np.clip(cals["umaren"].predict([p_umaren_hono])[0], 0.0, 1.0))
        # 馬連 ◎-〇 hit
        hit_umaren_hono = ({hon, tai} == {wi, pi_})

        # NDCG@5: relevance = 5 - 着順 (1着=4, 2着=3, ..., 5以下=0)
        jyun = pd.to_numeric(g["着順"], errors="coerce").fillna(99).astype(int).values
        rel = np.maximum(0, 5 - (jyun - 1))  # 1着→5, 2着→4, ..., 5着→1, 6位以下→0
        rel = np.clip(rel, 0, 5)
        ndcg5 = ndcg_at_k(rel.astype(float), scores, k=5)

        # 印別 hit
        hon_1st = int(hon == wi)
        hon_2nd = int(hon == pi_)
        hon_top2 = int(hon in {wi, pi_})
        hon_top3 = int(hon in actual_top3)
        tai_1st = int(tai == wi)
        tai_top2 = int(tai in {wi, pi_})
        tai_top3 = int(tai in actual_top3)
        san_top3 = int(san in actual_top3)
        del1_top3 = int(del1 in actual_top3)
        del2_top3 = int(del2 in actual_top3)

        # 1着馬が top-3 / top-5 に入る率
        winner_in_top3 = int(wi in top3)
        winner_in_top5 = int(wi in top5)
        # {1,2着} ⊂ top-5
        top2_subset_top5 = int({wi, pi_}.issubset(top5))
        # {1,2,3着} ⊂ top-5
        top3_subset_top5 = int(actual_top3.issubset(top5))

        # ECE 用: 各馬の p_tansho と 1着フラグ (馬単位レコード)
        # → race ごとにまとめて後で展開
        year = int(g["year"].iloc[0])
        rows.append({
            "rid_s": rid_s, "year": year,
            "ndcg5": ndcg5,
            "hon_1st": hon_1st, "hon_top2": hon_top2, "hon_top3": hon_top3,
            "tai_1st": tai_1st, "tai_top2": tai_top2, "tai_top3": tai_top3,
            "san_top3": san_top3, "del1_top3": del1_top3, "del2_top3": del2_top3,
            "winner_in_top3": winner_in_top3, "winner_in_top5": winner_in_top5,
            "top2_subset_top5": top2_subset_top5, "top3_subset_top5": top3_subset_top5,
            # ECE 用 (印 5 つの p)
            "p_tansho_hon": float(p_tansho_vec[hon]),
            "p_tansho_tai": float(p_tansho_vec[tai]),
            "p_fukusho_hon": float(p_fukusho_vec[hon]),
            "p_fukusho_tai": float(p_fukusho_vec[tai]),
            "p_fukusho_san": float(p_fukusho_vec[san]),
            "p_umaren_hono": p_umaren_hono,
            "hit_umaren_hono": int(hit_umaren_hono),
        })

    df = pd.DataFrame(rows)
    print(f"[build] races: {len(df):,}")

    # ===== 期間別集計 =====
    out = {"model_tag": tag, "n_races_total": len(df), "results": {}}
    for label, ys in [
        ("3年 (2023-2025)", {2023, 2024, 2025}),
        ("真OOS (2024-2025)", {2024, 2025}),
        ("2023 (valid参考)", {2023}),
        ("2024", {2024}),
        ("2025", {2025}),
    ]:
        sub = df[df["year"].isin(ys)].copy()
        R = len(sub)
        if R == 0:
            continue
        # ECE は 馬単位 (◎ + 〇 + ▲ など) で計測
        # tansho ECE: ◎の p_tansho, 1着フラグ
        ece_tansho_hon, _ = expected_calibration_error(
            sub["p_tansho_hon"].values, sub["hon_1st"].values
        )
        # fukusho ECE: ◎の p_fukusho, top3フラグ
        ece_fuku_hon, _ = expected_calibration_error(
            sub["p_fukusho_hon"].values, sub["hon_top3"].values
        )
        ece_fuku_tai, _ = expected_calibration_error(
            sub["p_fukusho_tai"].values, sub["tai_top3"].values
        )
        # 馬連 ◎-〇 ECE
        ece_umaren_hono, _ = expected_calibration_error(
            sub["p_umaren_hono"].values, sub["hit_umaren_hono"].values
        )

        res = {
            "races": R,
            "ndcg5": float(sub["ndcg5"].mean()),
            # ◎ (top-1)
            "hon_tansho_pct": float(sub["hon_1st"].mean() * 100),
            "hon_top2_pct":   float(sub["hon_top2"].mean() * 100),
            "hon_top3_pct":   float(sub["hon_top3"].mean() * 100),
            # 〇 (top-2)
            "tai_tansho_pct": float(sub["tai_1st"].mean() * 100),
            "tai_top2_pct":   float(sub["tai_top2"].mean() * 100),
            "tai_top3_pct":   float(sub["tai_top3"].mean() * 100),
            # ▲ △1 △2
            "san_top3_pct":   float(sub["san_top3"].mean() * 100),
            "del1_top3_pct":  float(sub["del1_top3"].mean() * 100),
            "del2_top3_pct":  float(sub["del2_top3"].mean() * 100),
            # 包含率
            "winner_in_top3_pct": float(sub["winner_in_top3"].mean() * 100),
            "winner_in_top5_pct": float(sub["winner_in_top5"].mean() * 100),
            "top2_subset_top5_pct": float(sub["top2_subset_top5"].mean() * 100),
            "top3_subset_top5_pct": float(sub["top3_subset_top5"].mean() * 100),
            # ECE
            "ece_tansho_hon": ece_tansho_hon,
            "ece_fuku_hon":   ece_fuku_hon,
            "ece_fuku_tai":   ece_fuku_tai,
            "ece_umaren_hono": ece_umaren_hono,
        }
        out["results"][label] = res

    # ===== 出力 =====
    out_dir = BASE / "reports"; out_dir.mkdir(exist_ok=True)
    out_json = out_dir / f"audit_marks_{tag}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    out_log = out_dir / f"audit_marks_{tag}.log"
    with open(out_log, "w", encoding="utf-8") as f:
        f.write(f"=== audit_marks {tag} ===\n")
        f.write(f"total races: {len(df):,}\n\n")
        for label, res in out["results"].items():
            f.write(f"--- {label} (R={res['races']:,}) ---\n")
            f.write(f"  NDCG@5             : {res['ndcg5']:.4f}\n")
            f.write(f"  ◎ 1着率           : {res['hon_tansho_pct']:6.2f}%\n")
            f.write(f"  ◎ 連対率          : {res['hon_top2_pct']:6.2f}%\n")
            f.write(f"  ◎ 複勝圏率        : {res['hon_top3_pct']:6.2f}%\n")
            f.write(f"  〇 1着率           : {res['tai_tansho_pct']:6.2f}%\n")
            f.write(f"  〇 連対率          : {res['tai_top2_pct']:6.2f}%\n")
            f.write(f"  〇 複勝圏率        : {res['tai_top3_pct']:6.2f}%\n")
            f.write(f"  ▲ 複勝圏率        : {res['san_top3_pct']:6.2f}%\n")
            f.write(f"  △1 複勝圏率       : {res['del1_top3_pct']:6.2f}%\n")
            f.write(f"  △2 複勝圏率       : {res['del2_top3_pct']:6.2f}%\n")
            f.write(f"  1着 ∈ top-3 率     : {res['winner_in_top3_pct']:6.2f}%\n")
            f.write(f"  1着 ∈ top-5 率     : {res['winner_in_top5_pct']:6.2f}%\n")
            f.write(f"  {{1,2着}} ⊂ top-5    : {res['top2_subset_top5_pct']:6.2f}%\n")
            f.write(f"  {{1,2,3着}} ⊂ top-5  : {res['top3_subset_top5_pct']:6.2f}%\n")
            f.write(f"  ECE 単勝(◎)       : {res['ece_tansho_hon']:.4f}\n")
            f.write(f"  ECE 複勝(◎)       : {res['ece_fuku_hon']:.4f}\n")
            f.write(f"  ECE 複勝(〇)       : {res['ece_fuku_tai']:.4f}\n")
            f.write(f"  ECE 馬連(◎-〇)    : {res['ece_umaren_hono']:.4f}\n")
            f.write("\n")

    # 標準出力に表
    print(f"\n=== audit_marks {tag} (test 2024-2025) ===")
    label = "真OOS (2024-2025)"
    if label in out["results"]:
        r = out["results"][label]
        print(f"  R={r['races']:,}  NDCG@5={r['ndcg5']:.4f}")
        print(f"  ◎ tansho={r['hon_tansho_pct']:5.2f}% top2={r['hon_top2_pct']:5.2f}% top3={r['hon_top3_pct']:5.2f}%")
        print(f"  〇 tansho={r['tai_tansho_pct']:5.2f}% top2={r['tai_top2_pct']:5.2f}% top3={r['tai_top3_pct']:5.2f}%")
        print(f"  ▲ top3={r['san_top3_pct']:5.2f}%  △1 top3={r['del1_top3_pct']:5.2f}%  △2 top3={r['del2_top3_pct']:5.2f}%")
        print(f"  win∈top3={r['winner_in_top3_pct']:.2f}%  win∈top5={r['winner_in_top5_pct']:.2f}%")
        print(f"  {{1,2}}⊂top5={r['top2_subset_top5_pct']:.2f}%  {{1,2,3}}⊂top5={r['top3_subset_top5_pct']:.2f}%")
        print(f"  ECE: tan={r['ece_tansho_hon']:.4f} fuku◎={r['ece_fuku_hon']:.4f} fuku〇={r['ece_fuku_tai']:.4f} umaren={r['ece_umaren_hono']:.4f}")

    print(f"\n[saved] {out_json}")
    print(f"[saved] {out_log}")


if __name__ == "__main__":
    main()
