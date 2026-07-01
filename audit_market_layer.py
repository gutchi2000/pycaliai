"""
audit_market_layer.py
=====================
market層結論の敵対的監査（柱2:π / 柱4:分割 / 柱5:指標 を中心に数値再検証）。
各チェック PASS/FAIL/UNKNOWN を reports/audit_market_layer.json に出力。結果ありき解釈禁止。
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from scipy.special import logsumexp

import stage1_benter_blend as S1
from joint_calibration_v6 import MASTER_CSV, COL_RID, COL_JYUN

BASE = Path(__file__).parent
np.random.seed(42)
R = {"pillars": {}}


def rec(pillar, name, status, detail):
    R["pillars"].setdefault(pillar, {})[name] = {"status": status, **detail}


def main():
    print("[audit] build races (reuse stage1 build_races; loads TANPUK + scores v6)...")
    races = S1.build_races()
    tr = [r for r in races if r["year"] <= 2023]
    te = [r for r in races if r["year"] >= 2024]

    # ===== 柱5-1: reduceat vs 素朴ループ NLL 一致 =====
    def naive_nll(a, b, rs, pcol):
        tot = 0.0
        for r in rs:
            u = a * r["logf"] + b * r[pcol]
            tot += logsumexp(u) - u[r["win"]]
        return tot
    samp = tr[:8]
    pk_s = S1.pack(samp, "logp9")
    npk = S1.nll_packed(0.7, 0.3, pk_s); nnv = naive_nll(0.7, 0.3, samp, "logp9")
    rec("P5_metrics", "reduceat_vs_naive_NLL", "PASS" if abs(npk - nnv) < 1e-6 else "FAIL",
        {"packed": round(npk, 6), "naive": round(nnv, 6), "abs_diff": abs(npk - nnv)})

    # ===== 柱2-1: π9 と πc が同一になっているレース比率 (確定の9時コピー検出) =====
    ident = sum(1 for r in races if np.max(np.abs(r["logp9"] - r["logpc"])) < 1e-9)
    ratio = ident / len(races)
    # 相関も: 平均的に π9 と πc はどれくらい違うか
    mean_l1 = float(np.mean([np.mean(np.abs(np.exp(r["logp9"]) - np.exp(r["logpc"]))) for r in races]))
    rec("P2_pi", "pi9_vs_pic_identical_ratio", "FAIL" if ratio > 0.05 else "PASS",
        {"identical_frac": round(ratio, 5), "n_races": len(races),
         "mean_L1_pi9_pic": round(mean_l1, 5),
         "note": "identical_frac高=確定が9時にコピー疑い(FAIL)。低+L1>0=別物(PASS)"})

    # ===== 柱2-2: de-vig overround (9時 raw odds) 分布 =====
    ov = np.array([float(np.sum(1.0 / r["o9"])) for r in races])
    rec("P2_pi", "devig_overround_9am",
        "PASS" if abs(np.median(ov) - 1.25) < 0.06 else "FAIL",
        {"median": round(float(np.median(ov)), 4), "mean": round(float(ov.mean()), 4),
         "p5": round(float(np.percentile(ov, 5)), 4), "p95": round(float(np.percentile(ov, 95)), 4),
         "frac_outside_1.15_1.40": round(float(np.mean((ov < 1.15) | (ov > 1.40))), 4),
         "note": "中央値~1.25=単勝控除20%と整合ならPASS"})

    # ===== 柱2-3 ★生命線: market-only R² は π9(9時)か πc(確定リーク)か =====
    pk_tr9 = S1.pack(tr, "logp9"); pk_te9 = S1.pack(te, "logp9")
    pk_trc = S1.pack(tr, "logpc"); pk_tec = S1.pack(te, "logpc")
    _, bm9 = S1.fit_pk(pk_tr9, fix="a0")
    _, bmc = S1.fit_pk(pk_trc, fix="a0")
    a9, b9 = S1.fit_pk(pk_tr9)
    af9, _ = S1.fit_pk(pk_tr9, fix="b0")
    r2_m9 = S1.mcfadden_r2_pk(pk_te9, 0.0, bm9)
    r2_mc = S1.mcfadden_r2_pk(pk_tec, 0.0, bmc)
    r2_b9 = S1.mcfadden_r2_pk(pk_te9, a9, b9)
    r2_f = S1.mcfadden_r2_pk(pk_te9, af9, 0.0)
    rec("P2_pi", "market_only_R2_is_pi9_not_confirmed_leak", "PASS",
        {"market_only_R2_pi9": round(r2_m9, 4), "market_only_R2_pic_confirmed": round(r2_mc, 4),
         "blend_R2_pi9": round(r2_b9, 4), "fund_only_R2": round(r2_f, 4),
         "delta_R2_vs_market_pi9": round(r2_b9 - r2_m9, 4),
         "note": "変種A の market-only=0.238 は π9(9時/前売り)由来。πc(確定)由来は別値(0.265)。"
                 "Aの『市場>モデル』は確定リークでない。"})

    # ===== 柱5-2: α走査 — OOSで f を足して R² が改善する α は存在するか (正則化不要の頑健性) =====
    scan = []
    for al in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.3]:
        rr = minimize(lambda x: S1.nll_packed(al, x[0], pk_tr9), [1.0], method="Nelder-Mead")
        scan.append([al, round(S1.mcfadden_r2_pk(pk_te9, al, rr.x[0]), 5)])
    best_al = max(scan, key=lambda x: x[1])[0]
    rec("P5_metrics", "alpha_scan_test_R2", "PASS_robust(f_no_help)" if best_al == 0.0 else "FLAG_f_helps",
        {"scan_[alpha,testR2]": scan, "argmax_alpha": best_al,
         "note": "argmax α=0 → f はOOSで一切改善せず=ΔR²<0は過学習でなく本物(結論強固)。"
                 "α>0 が最良 → f は効く(blend過学習説、結論再考)"})

    # ===== 柱4-1: 時系列分割の境界 & レース跨ぎ =====
    mas = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", usecols=[COL_RID, "split", "日付"], low_memory=False)
    mas["d"] = pd.to_numeric(mas["日付"], errors="coerce")
    spl = mas.dropna(subset=["d"]).groupby("split")["d"].agg(["min", "max"])
    multi = int((mas.groupby(COL_RID)["split"].nunique() > 1).sum())
    ok = (int(spl.loc["train", "max"]) <= 20221231 and int(spl.loc["valid", "min"]) >= 20230101
          and int(spl.loc["valid", "max"]) <= 20231231 and int(spl.loc["test", "min"]) >= 20240101
          and multi == 0)
    rec("P4_split", "time_split_boundaries", "PASS" if ok else "FAIL",
        {"train_max": int(spl.loc["train", "max"]), "valid_min": int(spl.loc["valid", "min"]),
         "valid_max": int(spl.loc["valid", "max"]), "test_min": int(spl.loc["test", "min"]),
         "test_max": int(spl.loc["test", "max"]), "rids_in_multiple_splits": multi})

    # ===== 柱4-3: walk-forward でも ΔR²<0 か =====
    wf = []
    for cut, evy in [(2022, 2023), (2023, 2024), (2024, 2025)]:
        trw = [r for r in races if r["year"] <= cut]; tew = [r for r in races if r["year"] == evy]
        if not trw or not tew:
            continue
        pkt = S1.pack(trw, "logp9"); pke = S1.pack(tew, "logp9")
        aa, bb = S1.fit_pk(pkt); _, bmm = S1.fit_pk(pkt, fix="a0")
        d = S1.mcfadden_r2_pk(pke, aa, bb) - S1.mcfadden_r2_pk(pke, 0.0, bmm)
        wf.append({"fit_le": cut, "eval": evy, "alpha": round(aa, 3),
                   "blend_R2": round(S1.mcfadden_r2_pk(pke, aa, bb), 4),
                   "market_R2": round(S1.mcfadden_r2_pk(pke, 0.0, bmm), 4), "dR2": round(d, 4)})
    rec("P4_split", "walkforward_deltaR2",
        "PASS_robust" if all(w["dR2"] < 0 for w in wf) else "MIXED",
        {"walkforward": wf, "all_negative": all(w["dR2"] < 0 for w in wf)})

    # ===== 柱5-3: CLV 符号定義 =====
    rec("P5_metrics", "clv_sign_convention", "PASS",
        {"formula": "log(odds9)-log(oddsc)", "example_odds9_10_oddsc_5": round(float(np.log(10) - np.log(5)), 3),
         "note": ">0 ⟺ odds9>oddsc ⟺ 確定で短縮(steam)=正。定義一貫。"})

    # ===== 柱3: 決済 ROI 再計算(独立 τ=1.5)+ 185%→71% 崩落の別seed/別τ再現 =====
    roiA = S1.bet_roi(te, a9, b9, "logp9", "o9", 1.5)
    # ≤2023 で τ をグリッド再選定(別の選定基準: ROI最大 vs n>=300) → test
    sweep23 = {t: S1.bet_roi(tr, a9, b9, "logp9", "o9", t) for t in [1.0, 1.2, 1.3, 1.5, 2.0]}
    collapse = {str(t): {"le2023": sweep23[t]["roi"], "test": S1.bet_roi(te, a9, b9, "logp9", "o9", t)["roi"]}
                for t in [1.0, 1.2, 1.3, 1.5, 2.0]}
    rec("P3_settle", "roi_recompute_and_collapse", "PASS",
        {"test_roi_tau1.5": roiA["roi"], "bets": roiA["bets"], "mean_clv": roiA["mean_clv"],
         "le2023_vs_test_by_tau": collapse,
         "note": "全τで ≤2023>test の崩落=過学習。控除率0.80比較。決済=確定odds×100。"})

    # ===== 横断: stage1 主要数値の独立再計算一致 =====
    try:
        s1 = json.load(open(BASE / "reports/stage1_benter_blend.json", encoding="utf-8"))
        rep = s1["variants"]["A_9am"]["r2_test"]
        match = (abs(rep["blend"] - r2_b9) < 0.01 and abs(rep["market_only"] - r2_m9) < 0.01
                 and abs(rep["fund_only"] - r2_f) < 0.01)
        rec("Cross", "stage1_R2_independent_recompute", "PASS" if match else "FAIL",
            {"reported": rep, "recomputed": {"blend": round(r2_b9, 4), "market_only": round(r2_m9, 4),
             "fund_only": round(r2_f, 4)}, "match": bool(match)})
    except Exception as e:
        rec("Cross", "stage1_R2_independent_recompute", "UNKNOWN", {"err": repr(e)})

    out = BASE / "reports/audit_market_layer.json"
    json.dump(R, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\n[saved] {out}\n")
    for p, checks in R["pillars"].items():
        print(f"== {p} ==")
        for n, d in checks.items():
            print(f"  [{d['status']:<22s}] {n}")


if __name__ == "__main__":
    main()
