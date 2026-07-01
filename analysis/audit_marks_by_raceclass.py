"""
audit_marks_by_raceclass.py
===========================
v6 の印精度を「クラス名」で層別して測定する。
主目的: 新馬 (過去走ゼロ → 背骨特徴が全馬 -9999 に潰れる) の印が
        「ランダム同然に壊れている」のか「血統+調教で degrade しつつ実信号あり」のかを決着。

決定的な判定軸:
  - ◎複勝圏率 (hon_top3) vs ランダム基準 mean(3/頭数)
    → 基準を有意に上回れば、過去走ナシでもモデルは実信号を抽出している (= acceptable degrade)
    → 基準と同等なら、新馬の印は無意味 (= 壊れている)

使い方:
  python analysis/audit_marks_by_raceclass.py --model v6
"""
from __future__ import annotations
import argparse
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import pl_probs as PL
from backtest_pl_ev import (
    load_payouts, apply_encoders, all_fukusho_vec_fast,
    MASTER_CSV, COL_RID, COL_BAN, COL_JYUN,
)
import backtest_pl_ev as be


def wilson_ci(k, n, z=1.96):
    """二項比率の Wilson 95% CI (low, high) を % で返す。"""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return ((center - half) * 100, (center + half) * 100)


def classify(cls: str) -> str:
    """クラス名 → 集計グループ。"""
    s = str(cls)
    if s == "新馬":
        return "新馬"
    if s == "未勝利":
        return "未勝利"
    if s.startswith("J") or "障" in s:  # JG1/JG2/JG3 等
        return "障害"
    return "古馬/3歳条件以上"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="v6")
    args = ap.parse_args()
    tag = args.model

    be.MODEL_PKL = BASE / f"models/unified_rank_{tag}.pkl"
    be.CAL_PKL = BASE / f"models/pl_calibrators_{tag}.pkl"
    assert be.MODEL_PKL.exists(), f"{be.MODEL_PKL} not found"
    print(f"[model] {be.MODEL_PKL.name}")

    payouts = load_payouts()
    print(f"[payouts] {len(payouts):,} races")

    # ---- score_test を再現しつつ、生の クラス名 を温存 ----
    bundle = joblib.load(be.MODEL_PKL)
    model, feats, encs = bundle["model"], bundle["feature_cols"], bundle["encoders"]
    rr_mode = bundle.get("race_relative_mode")
    ca_mode = bundle.get("course_affinity_mode")

    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df = df.dropna(subset=[COL_JYUN, COL_RID, "split"]).copy()
    df["_cls_raw"] = df["クラス名"].astype(str)  # 生クラス名を温存 (encode 前)
    if rr_mode:
        from race_relative_feats import add_race_relative_feats
        df = add_race_relative_feats(df, mode=rr_mode)
    if ca_mode:
        from course_affinity_feats import add_course_affinity_feats
        df = add_course_affinity_feats(df, mode=ca_mode)
    te = df[df["split"].isin(["test", "valid"])].copy()  # 2023-2025
    te["year"] = te[COL_RID].astype(str).str[:4].astype(int)
    te_enc = apply_encoders(te, encs)
    X = te_enc[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    te["_score"] = model.predict(X)
    print(f"[scored] rows={len(te):,}  races={te[COL_RID].nunique():,}")

    # ---- レース単位で印 hit を集計 ----
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
        n = len(scores)
        order = np.argsort(-scores)
        hon, tai, san = int(order[0]), int(order[1]), int(order[2])
        del1, del2 = int(order[3]), int(order[4])
        top3 = {hon, tai, san}
        top5 = {hon, tai, san, del1, del2}
        actual_top3 = {wi, pi_, si}

        rows.append({
            "year": int(g["year"].iloc[0]),
            "cls": classify(g["_cls_raw"].iloc[0]),
            "cls_raw": str(g["_cls_raw"].iloc[0]),
            "field": n,
            "rand_top3": min(1.0, 3.0 / n),         # ランダムに1頭選んで top3 入る確率
            "hon_1st": int(hon == wi),
            "hon_top2": int(hon in {wi, pi_}),
            "hon_top3": int(hon in actual_top3),
            "tai_top3": int(tai in actual_top3),
            "san_top3": int(san in actual_top3),
            "winner_in_top3": int(wi in top3),
            "winner_in_top5": int(wi in top5),
            "top3_subset_top5": int(actual_top3.issubset(top5)),
        })

    A = pd.DataFrame(rows)
    oos = A[A["year"].isin({2024, 2025})].copy()
    print(f"[OOS 2024-2025] races={len(oos):,}\n")

    # ---- グループ別サマリ ----
    groups = ["新馬", "未勝利", "古馬/3歳条件以上", "障害"]
    print("=== v6 印精度 クラス別 (真OOS 2024-2025) ===")
    hdr = (f"{'group':<16} {'R':>5} {'field':>5} "
           f"{'◎1着':>6} {'◎連対':>6} {'◎複勝':>7} {'rand':>6} {'lift':>6} "
           f"{'win∈t3':>7} {'win∈t5':>7} {'123⊂t5':>7}")
    print(hdr)
    print("-" * len(hdr))
    summary = {}
    for grp in groups + ["__ALL__"]:
        sub = oos if grp == "__ALL__" else oos[oos["cls"] == grp]
        R = len(sub)
        if R == 0:
            continue
        hon_top3 = sub["hon_top3"].mean() * 100
        rand = sub["rand_top3"].mean() * 100
        lo, hi = wilson_ci(int(sub["hon_top3"].sum()), R)
        summary[grp] = {"R": R, "hon_top3": hon_top3, "rand": rand,
                        "ci": (lo, hi)}
        name = "全クラス" if grp == "__ALL__" else grp
        print(f"{name:<16} {R:>5} {sub['field'].mean():>5.1f} "
              f"{sub['hon_1st'].mean()*100:>5.1f}% {sub['hon_top2'].mean()*100:>5.1f}% "
              f"{hon_top3:>6.1f}% {rand:>5.1f}% {hon_top3-rand:>+5.1f} "
              f"{sub['winner_in_top3'].mean()*100:>6.1f}% {sub['winner_in_top5'].mean()*100:>6.1f}% "
              f"{sub['top3_subset_top5'].mean()*100:>6.1f}%")

    # ---- 新馬の決定的判定: ◎複勝圏率 vs ランダム基準 (Wilson CI) ----
    print("\n=== 決定的判定: 新馬の ◎複勝圏率 は ランダム基準を上回るか ===")
    if "新馬" in summary:
        s = summary["新馬"]
        lo, hi = s["ci"]
        verdict = ("✅ 実信号あり (CI下限 > rand基準)" if lo > s["rand"]
                   else "⚠️ ランダムと区別不能 (CI が rand基準をまたぐ) = 壊れ寄り")
        print(f"  新馬: ◎複勝圏率 {s['hon_top3']:.1f}% "
              f"[95%CI {lo:.1f}-{hi:.1f}]  vs  ランダム基準 {s['rand']:.1f}%  R={s['R']}")
        print(f"  → {verdict}")
        if "古馬/3歳条件以上" in summary:
            o = summary["古馬/3歳条件以上"]
            print(f"  参考(条件戦): ◎複勝圏率 {o['hon_top3']:.1f}% vs rand {o['rand']:.1f}% "
                  f"(lift +{o['hon_top3']-o['rand']:.1f}pt)")
            print(f"  新馬 lift +{s['hon_top3']-s['rand']:.1f}pt / 条件戦 lift "
                  f"+{o['hon_top3']-o['rand']:.1f}pt "
                  f"→ 新馬は条件戦の {(s['hon_top3']-s['rand'])/max(o['hon_top3']-o['rand'],1e-9)*100:.0f}% の判別力")

    # ---- 新馬の cls_raw 内訳 (2歳新馬がほとんどのはず) ----
    print("\n[新馬 cls_raw 内訳]")
    print(oos[oos["cls"] == "新馬"]["cls_raw"].value_counts().to_string())


if __name__ == "__main__":
    main()
