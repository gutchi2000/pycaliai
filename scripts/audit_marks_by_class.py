"""
audit_marks_by_class.py
=======================
audit_marks.py のクラス別 split 版。
master_v2 の `クラス名` 列で test 期間 (2024-2025) を split し、
各クラス (未勝利 / 1勝 / 2勝 / 3勝 / OP / G3 / G2 / G1 等) ごとに
v6 印モデルの精度を測定する。

新クラス表記 (週次運用と整合) に正規化:
  500万   → 1勝
  1000万  → 2勝
  1600万  → 3勝
  ｵｰﾌﾟﾝ   → ｵｰﾌﾟﾝ
  Ｇ１/Ｇ２/Ｇ３ → G1/G2/G3
  新馬/未勝利 → そのまま

使い方:
  python scripts/audit_marks_by_class.py --model v6
  python scripts/audit_marks_by_class.py --model v5 --years 2024,2025

出力:
  reports/audit_marks_by_class_{tag}.json
  reports/audit_marks_by_class_{tag}.log
"""
from __future__ import annotations
import argparse
import io
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import pl_probs as PL  # noqa: E402
from backtest_pl_ev import (  # noqa: E402
    load_payouts, score_test,
    all_umaren_mat, all_fukusho_vec_fast,
    COL_RID, COL_BAN,
)
import backtest_pl_ev as be  # noqa: E402
from audit_marks import expected_calibration_error, ndcg_at_k  # noqa: E402

BASE = Path(__file__).parent.parent


# 旧クラス → 新クラス 正規化
CLASS_NORMALIZE = {
    "500万":   "1勝",
    "1000万":  "2勝",
    "1600万":  "3勝",
    "Ｇ１":    "G1",
    "Ｇ２":    "G2",
    "Ｇ３":    "G3",
    "ＪＧ１":  "JG1",
    "ＪＧ２":  "JG2",
    "ＪＧ３":  "JG3",
}
# 表示順序
CLASS_ORDER = ["新馬", "未勝利", "1勝", "2勝", "3勝",
               "ｵｰﾌﾟﾝ", "OP", "OP(L)", "L",
               "G3", "G2", "G1", "JG3", "JG2", "JG1", "不明"]


def normalize_class(c: str) -> str:
    c = str(c).strip()
    return CLASS_NORMALIZE.get(c, c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="v6", help="model tag: v5/v6")
    ap.add_argument("--years", default="2024,2025",
                    help="集計対象年 (カンマ区切り、デフォルト test 期間 2024,2025)")
    args = ap.parse_args()
    tag = args.model
    target_years = {int(y.strip()) for y in args.years.split(",")}

    be.MODEL_PKL = BASE / f"models/unified_rank_{tag}.pkl"
    be.CAL_PKL   = BASE / f"models/pl_calibrators_{tag}.pkl"
    be.CURVE_PKL = BASE / f"data/pl_payout_curve_{tag}.pkl"
    assert be.MODEL_PKL.exists(), f"{be.MODEL_PKL} not found"
    print(f"[model] tag={tag}  {be.MODEL_PKL}")

    payouts = load_payouts()
    print(f"[payouts] races={len(payouts):,}")

    te = score_test(include_valid=True)
    print(f"[test+valid] rows={len(te):,}  races={te[COL_RID].nunique():,}")

    # クラス列は score_test 内で encode されているため、master_v2 から
    # 生の文字列値を別途 JOIN する。
    MASTER_CSV = BASE / "data/master_v2_20130105-20251228.csv"
    raw = pd.read_csv(
        MASTER_CSV, encoding="utf-8-sig", low_memory=False,
        usecols=[COL_RID, "クラス名"],
    )
    raw[COL_RID] = raw[COL_RID].astype(str)
    raw = raw.drop_duplicates(subset=[COL_RID])
    rid2cls = dict(zip(raw[COL_RID], raw["クラス名"].astype(str)))
    print(f"[class map] {len(rid2cls):,} races")

    # 年フィルタ
    te = te[te["year"].isin(target_years)].copy()
    print(f"[filter years={sorted(target_years)}] rows={len(te):,}  races={te[COL_RID].nunique():,}")

    # PL calibrator
    cal = None
    if be.CAL_PKL.exists():
        cal = joblib.load(be.CAL_PKL)
        print(f"[cal] {be.CAL_PKL}")

    # ===== レース単位で集計 =====
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
        order = np.argsort(-scores)
        hon, tai, san = int(order[0]), int(order[1]), int(order[2])
        del1, del2 = int(order[3]), int(order[4])
        top3 = {hon, tai, san}
        top5 = {hon, tai, san, del1, del2}
        actual_top3 = {wi, pi_, si}

        p_tansho_vec = PL.all_tansho(w)
        p_fukusho_vec = all_fukusho_vec_fast(w)
        Pmat = all_umaren_mat(w)
        p_umaren_hono = float(Pmat[hon, tai])
        if cal is not None and "calibrators" in cal:
            cals = cal["calibrators"]
            p_tansho_vec = np.clip(cals["tansho"].predict(p_tansho_vec), 0.0, 1.0)
            p_fukusho_vec = np.clip(cals["fukusho"].predict(p_fukusho_vec), 0.0, 1.0)
            p_umaren_hono = float(np.clip(cals["umaren"].predict([p_umaren_hono])[0], 0.0, 1.0))
        hit_umaren_hono = ({hon, tai} == {wi, pi_})

        jyun = pd.to_numeric(g["着順"], errors="coerce").fillna(99).astype(int).values
        rel = np.maximum(0, 5 - (jyun - 1)).clip(0, 5).astype(float)
        ndcg5 = ndcg_at_k(rel, scores, k=5)

        cls_raw = rid2cls.get(rid_s, "不明")
        cls = normalize_class(cls_raw)

        rows.append({
            "rid_s": rid_s,
            "year": int(g["year"].iloc[0]),
            "cls": cls,
            "ndcg5": ndcg5,
            # 各印の 1着率
            "hon_1st":  int(hon == wi),
            "tai_1st":  int(tai == wi),
            "san_1st":  int(san == wi),
            "del1_1st": int(del1 == wi),
            "del2_1st": int(del2 == wi),
            # 各印の 連対率
            "hon_top2": int(hon in {wi, pi_}),
            "tai_top2": int(tai in {wi, pi_}),
            "san_top2": int(san in {wi, pi_}),
            "del1_top2": int(del1 in {wi, pi_}),
            "del2_top2": int(del2 in {wi, pi_}),
            # 各印の 複勝圏率
            "hon_top3":  int(hon in actual_top3),
            "tai_top3":  int(tai in actual_top3),
            "san_top3":  int(san in actual_top3),
            "del1_top3": int(del1 in actual_top3),
            "del2_top3": int(del2 in actual_top3),
            "winner_in_top3": int(wi in top3),
            "winner_in_top5": int(wi in top5),
            "p_tansho_hon": float(p_tansho_vec[hon]),
            "p_fukusho_hon": float(p_fukusho_vec[hon]),
            "p_umaren_hono": p_umaren_hono,
            "hit_umaren_hono": int(hit_umaren_hono),
        })

    df = pd.DataFrame(rows)
    print(f"[build] races: {len(df):,}")
    print(f"[classes] {df['cls'].value_counts().to_dict()}")

    # ===== クラス別集計 =====
    by_class = {}
    for cls, sub in df.groupby("cls"):
        R = len(sub)
        if R < 30:  # サンプル少なすぎは別途 "其他" にまとめても良いが、ここでは全部出す
            pass
        ece_tansho = expected_calibration_error(
            sub["p_tansho_hon"].values, sub["hon_1st"].values
        )[0] if R >= 50 else None
        ece_fuku = expected_calibration_error(
            sub["p_fukusho_hon"].values, sub["hon_top3"].values
        )[0] if R >= 50 else None
        ece_umaren = expected_calibration_error(
            sub["p_umaren_hono"].values, sub["hit_umaren_hono"].values
        )[0] if R >= 50 else None

        by_class[cls] = {
            "races": R,
            "ndcg5": float(sub["ndcg5"].mean()),
            # 1 着率
            "hon_1st_pct":  float(sub["hon_1st"].mean()  * 100),
            "tai_1st_pct":  float(sub["tai_1st"].mean()  * 100),
            "san_1st_pct":  float(sub["san_1st"].mean()  * 100),
            "del1_1st_pct": float(sub["del1_1st"].mean() * 100),
            "del2_1st_pct": float(sub["del2_1st"].mean() * 100),
            # 連対率
            "hon_top2_pct":  float(sub["hon_top2"].mean()  * 100),
            "tai_top2_pct":  float(sub["tai_top2"].mean()  * 100),
            "san_top2_pct":  float(sub["san_top2"].mean()  * 100),
            "del1_top2_pct": float(sub["del1_top2"].mean() * 100),
            "del2_top2_pct": float(sub["del2_top2"].mean() * 100),
            # 複勝圏率
            "hon_top3_pct":  float(sub["hon_top3"].mean()  * 100),
            "tai_top3_pct":  float(sub["tai_top3"].mean()  * 100),
            "san_top3_pct":  float(sub["san_top3"].mean()  * 100),
            "del1_top3_pct": float(sub["del1_top3"].mean() * 100),
            "del2_top3_pct": float(sub["del2_top3"].mean() * 100),
            "winner_in_top3_pct": float(sub["winner_in_top3"].mean() * 100),
            "winner_in_top5_pct": float(sub["winner_in_top5"].mean() * 100),
            "ece_tansho_hon": ece_tansho,
            "ece_fuku_hon":   ece_fuku,
            "ece_umaren_hono": ece_umaren,
        }

    # 表示順
    order_idx = {c: i for i, c in enumerate(CLASS_ORDER)}
    ordered = sorted(by_class.items(), key=lambda x: order_idx.get(x[0], 99))

    out = {
        "model_tag": tag,
        "target_years": sorted(target_years),
        "n_races_total": len(df),
        "by_class": dict(ordered),
    }

    out_dir = BASE / "reports"; out_dir.mkdir(exist_ok=True)
    out_json = out_dir / f"audit_marks_by_class_{tag}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # export_weekly_marks.py が読む軽量形式を data/ に書き出す
    # (bundle.json に各 race の経験 prior を埋め込むため)
    data_dir = BASE / "data"; data_dir.mkdir(exist_ok=True)
    prior_json = data_dir / f"class_prior_{tag}.json"
    prior_out = {
        "model_tag": tag,
        "target_years": sorted(target_years),
        "generated_at": pd.Timestamp.now().isoformat(),
        "n_races_total": len(df),
        "classes": {
            cls: {
                "n_samples":     r["races"],
                "hon_1st_pct":   round(r["hon_1st_pct"],   1),
                "hon_top2_pct":  round(r["hon_top2_pct"],  1),
                "hon_top3_pct":  round(r["hon_top3_pct"],  1),
                "tai_1st_pct":   round(r["tai_1st_pct"],   1),
                "tai_top2_pct":  round(r["tai_top2_pct"],  1),
                "tai_top3_pct":  round(r["tai_top3_pct"],  1),
                "san_top3_pct":  round(r["san_top3_pct"],  1),
                "del1_top3_pct": round(r["del1_top3_pct"], 1),
                "del2_top3_pct": round(r["del2_top3_pct"], 1),
            }
            for cls, r in ordered
        },
    }
    with open(prior_json, "w", encoding="utf-8") as f:
        json.dump(prior_out, f, indent=2, ensure_ascii=False)
    print(f"[saved] {prior_json}  (consumed by export_weekly_marks.py)")

    def _matrix_block(metric_key_suffix: str, title: str) -> str:
        """metric_key_suffix = '1st_pct' / 'top2_pct' / 'top3_pct'
        印 5 つ (◎ 〇 ▲ △1 △2) の値をクラス × 印 のマトリクスで出力。"""
        out_str = f"\n--- {title} ---\n"
        out_str += f"{'クラス':<8} {'R数':>5}  {'◎':>6} {'〇':>6} {'▲':>6} {'△1':>6} {'△2':>6}\n"
        out_str += "-" * 56 + "\n"
        for cls, r in ordered:
            vals = [r[f"{prefix}_{metric_key_suffix}"]
                    for prefix in ["hon", "tai", "san", "del1", "del2"]]
            out_str += (f"{cls:<8} {r['races']:>5}  "
                        + " ".join(f"{v:6.1f}" for v in vals)
                        + "\n")
        return out_str

    out_log = out_dir / f"audit_marks_by_class_{tag}.log"
    with open(out_log, "w", encoding="utf-8") as f:
        f.write(f"=== audit_marks_by_class {tag}  years={sorted(target_years)} ===\n")
        f.write(f"total races: {len(df):,}\n")
        f.write(_matrix_block("top3_pct", "複勝圏率 (%) ─ 3着以内に入る率"))
        f.write(_matrix_block("top2_pct", "連対率 (%) ─ 2着以内に入る率"))
        f.write(_matrix_block("1st_pct",  "1着率 (%)"))
        f.write("\n--- 参考: NDCG@5 / ECE ---\n")
        f.write(f"{'クラス':<8} {'R数':>5}  {'NDCG5':>6} {'ECE単◎':>7} {'ECE複◎':>7} {'ECE馬連':>7}\n")
        f.write("-" * 56 + "\n")
        for cls, r in ordered:
            def fmt(v, n=7, dec=4):
                return f"{'-':>{n}}" if v is None else f"{v:{n}.{dec}f}"
            f.write(f"{cls:<8} {r['races']:>5}  {r['ndcg5']:6.4f} "
                    f"{fmt(r['ece_tansho_hon'])} {fmt(r['ece_fuku_hon'])} {fmt(r['ece_umaren_hono'])}\n")

    # 標準出力: 複勝圏率マトリクスをメインで
    print(f"\n=== audit_marks_by_class {tag}  years={sorted(target_years)} ===")
    print(_matrix_block("top3_pct", "複勝圏率 (%) ─ 3着以内"))
    print(_matrix_block("top2_pct", "連対率 (%)"))
    print(_matrix_block("1st_pct",  "1着率 (%)"))

    print(f"\n[saved] {out_json}")
    print(f"[saved] {out_log}")


if __name__ == "__main__":
    main()
