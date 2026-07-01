"""
stage1b_shrinkage_alpha.py
==========================
Stage 1b: α走査FLAG を規律をもって掘る。MLE過大適合を正し、小αが本物か / test-peek
artifact か / 収益化するかを最終確定。

監査の FLAG (α≈0.3 で test ΔR²=+0.007) は α を **test 上で argmax** した値＝軽度 test-peek。
これを「train(≤2022)で β fit → valid(2023)で α 選択 → test(2024-25)で確認」の入れ子に直す。

★決定的検査: validで選んだα を test に当てて ΔR²(test) が正のまま保たれるか。
  保つ → f は本物の小増分（FLAG本物）。 消える/負 → FLAG は test-peek artifact。

3方式（全て fit/選択は ≤2023 のみ、評価 test、π9主・π確定従）:
  (1) MLE（stage1再現・対照）
  (2) L2 shrinkage（αにL2罰則、罰則強度を valid OOS R² で選択）
  (3) α grid を valid OOS R² で選択 → test 適用  ★主役

収益化ゲート: valid選択α + valid選択τ で test 賭けROI/CLV（決済=確定odds, vs控除率80%）。

出力: reports/stage1b_shrinkage_alpha.json
実行: python stage1b_shrinkage_alpha.py
"""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from scipy.optimize import minimize

import stage1_benter_blend as S1

BASE = Path(__file__).parent
np.random.seed(42)
ALPHA_GRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0, 1.3]
LAM2_GRID = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]  # per-race α² 罰則
EV_GRID = [0.9, 1.0, 1.05, 1.1, 1.2, 1.3, 1.5, 2.0]
BENTER_DR2 = 0.0178  # Benter(1994) 参考 ΔR²
MIN_TUNE_BETS = 200


def fit_beta_at_alpha(pk, alpha):
    r = minimize(lambda x: S1.nll_packed(alpha, x[0], pk), [1.0], method="Nelder-Mead")
    return float(r.x[0])


def fit_l2_alpha(pk, lam2):
    """α に L2 罰則 (per-race scaled) を掛けた二段合成 MLE。"""
    nrace = len(pk["starts"])
    pen = lam2 * nrace
    r = minimize(lambda x: S1.nll_packed(x[0], x[1], pk) + pen * x[0] ** 2,
                 [0.5, 1.0], method="Nelder-Mead")
    return float(r.x[0]), float(r.x[1])


def select_tau_on_valid(valid, a, b, pcol, ocol):
    best = (EV_GRID[0], -1e9)
    sweep = {}
    for t in EV_GRID:
        r = S1.bet_roi(valid, a, b, pcol, ocol, t)
        sweep[str(t)] = {"roi": r["roi"], "bets": r["bets"]}
        if r["bets"] >= MIN_TUNE_BETS and r["roi"] > best[1]:
            best = (t, r["roi"])
    return best[0], sweep


def analyze(races, pcol, ocol, label):
    tr = [r for r in races if r["year"] <= 2022]      # train
    vl = [r for r in races if r["year"] == 2023]       # valid (α/τ 選択)
    tv = [r for r in races if r["year"] <= 2023]       # ≤2023 (最終fit)
    te = [r for r in races if r["year"] >= 2024]
    te24 = [r for r in te if r["year"] == 2024]; te25 = [r for r in te if r["year"] == 2025]
    pk_tr = S1.pack(tr, pcol); pk_vl = S1.pack(vl, pcol)
    pk_tv = S1.pack(tv, pcol); pk_te = S1.pack(te, pcol)
    pk_te24 = S1.pack(te24, pcol); pk_te25 = S1.pack(te25, pcol)

    # market-only (β fit ≤2023) と fund-only (test 参照)
    _, bm = S1.fit_pk(pk_tv, fix="a0")
    r2_mkt_te = S1.mcfadden_r2_pk(pk_te, 0.0, bm)
    af, _ = S1.fit_pk(pk_tv, fix="b0")
    r2_fund_te = S1.mcfadden_r2_pk(pk_te, af, 0.0)

    out = {"market_only_R2_test": round(r2_mkt_te, 5),
           "fund_only_R2_test": round(r2_fund_te, 5), "methods": {}}

    def dr2_block(a, b):
        return {"alpha": round(a, 4), "beta": round(b, 4),
                "blend_R2_test": round(S1.mcfadden_r2_pk(pk_te, a, b), 5),
                "dR2_test": round(S1.mcfadden_r2_pk(pk_te, a, b) - r2_mkt_te, 5),
                "dR2_test_2024": round(S1.mcfadden_r2_pk(pk_te24, a, b) - S1.mcfadden_r2_pk(pk_te24, 0.0, bm), 5),
                "dR2_test_2025": round(S1.mcfadden_r2_pk(pk_te25, a, b) - S1.mcfadden_r2_pk(pk_te25, 0.0, bm), 5)}

    # (1) MLE on ≤2023
    a_mle, b_mle = S1.fit_pk(pk_tv)
    out["methods"]["1_MLE"] = dr2_block(a_mle, b_mle)

    # (3) α grid: β fit on TRAIN given α, select α by VALID R², apply test
    valid_curve = []
    for al in ALPHA_GRID:
        bb = fit_beta_at_alpha(pk_tr, al)
        valid_curve.append([al, round(S1.mcfadden_r2_pk(pk_vl, al, bb), 5)])
    a_star = max(valid_curve, key=lambda x: x[1])[0]
    b_star = fit_beta_at_alpha(pk_tv, a_star)  # 最終 β は ≤2023 で a_star 固定 fit
    blk3 = dr2_block(a_star, b_star)
    blk3["valid_R2_curve"] = valid_curve
    blk3["alpha_selected_on_valid"] = a_star
    out["methods"]["3_grid_validselect"] = blk3

    # (2) L2 shrinkage: λ2 を valid R² で選択
    l2_curve = []
    for lam2 in LAM2_GRID:
        aa, bb = fit_l2_alpha(pk_tr, lam2)
        l2_curve.append([lam2, round(aa, 4), round(S1.mcfadden_r2_pk(pk_vl, aa, bb), 5)])
    lam2_star = max(l2_curve, key=lambda x: x[2])[0]
    a_l2, b_l2 = fit_l2_alpha(pk_tv, lam2_star)
    blk2 = dr2_block(a_l2, b_l2)
    blk2["lam2_selected_on_valid"] = lam2_star
    blk2["l2_curve_[lam2,alpha,validR2]"] = l2_curve
    out["methods"]["2_L2_shrinkage"] = blk2

    # 収益化ゲート: grid法の (a_star,b_star) で τ を valid 選定 → test
    tau, tau_sweep = select_tau_on_valid(vl, a_star, b_star, pcol, ocol)
    roi_te = S1.bet_roi(te, a_star, b_star, pcol, ocol, tau)
    roi_24 = S1.bet_roi(te24, a_star, b_star, pcol, ocol, tau)
    roi_25 = S1.bet_roi(te25, a_star, b_star, pcol, ocol, tau)
    out["monetization_gridalpha"] = {
        "alpha": round(a_star, 4), "beta": round(b_star, 4), "tau_selected_on_valid": tau,
        "valid_tau_sweep": tau_sweep,
        "test_roi": roi_te["roi"], "test_bets": roi_te["bets"], "test_mean_clv": roi_te["mean_clv"],
        "test_2024_roi": roi_24["roi"], "test_2025_roi": roi_25["roi"],
        "by_move_test": roi_te["by_move"], "takeout_floor": 0.80}
    return out, blk3["dR2_test"], a_star, roi_te["roi"]


def main():
    print("[stage1b] build races...")
    races = S1.build_races()

    result = {"benter_ref_dR2": BENTER_DR2, "alpha_grid": ALPHA_GRID,
              "protocol": "fit β on train≤2022; select α/τ on valid 2023; confirm on test 2024-25",
              "variants": {}}
    print("\n===== π9 (当日9時, 実戦) =====")
    o9, dr9, a9, roi9 = analyze(races, "logp9", "o9", "π9")
    result["variants"]["A_pi9"] = o9
    print("\n===== π確定 (上限) =====")
    oc, drc, ac, roic = analyze(races, "logpc", "oc", "πc")
    result["variants"]["B_pic"] = oc

    # 判定
    peak_dr2 = o9["methods"]["3_grid_validselect"]["dR2_test"]
    a_sel = o9["methods"]["3_grid_validselect"]["alpha_selected_on_valid"]
    roi = o9["monetization_gridalpha"]["test_roi"]
    clv = o9["monetization_gridalpha"]["test_mean_clv"]
    if a_sel <= 0.0 or peak_dr2 <= 0.0:
        verdict = "ARTIFACT: validで選んだα ではtestΔR²が消失/負 → FLAGはtest-peek。stage1のαなしが妥当。"
    elif roi >= 1.0 and (clv or 0) > 0:
        verdict = "REAL_AND_PROFITABLE: 小αは本物かつ収益化(ROI≥100%,CLV正) → 全部ひっくり返る。要拡張。"
    else:
        verdict = "REAL_BUT_UNPROFITABLE: 小αは本物(validで選んでtestでΔR²>0保持)だが非収益化(ROI<80%)。market層を正確に閉じる。"
    result["verdict"] = verdict
    result["pi9_summary"] = {
        "valid_selected_alpha": a_sel, "test_dR2": peak_dr2,
        "ratio_to_benter": round(peak_dr2 / BENTER_DR2, 3) if peak_dr2 == peak_dr2 else None,
        "test_roi": roi, "test_clv": clv,
        "econ_note": f"単勝控除20%克服には実効的に大きな価格優位が要る。ΔR²≈{peak_dr2:.4f} はBenter(0.0178)の{100*peak_dr2/BENTER_DR2:.0f}%。"}

    outp = BASE / "reports/stage1b_shrinkage_alpha.json"
    json.dump(result, open(outp, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\n[saved] {outp}")

    # サマリ印字
    for v, blk in result["variants"].items():
        print(f"\n== {v} ==  market-only R²(test)={blk['market_only_R2_test']}  fund-only={blk['fund_only_R2_test']}")
        for mname, m in blk["methods"].items():
            print(f"  {mname:20s} α={m['alpha']:.3f} β={m['beta']:.3f}  testΔR²={m['dR2_test']:+.5f} "
                  f"(2024={m['dR2_test_2024']:+.4f} 2025={m['dR2_test_2025']:+.4f})")
        mz = blk["monetization_gridalpha"]
        print(f"  収益化(grid α={mz['alpha']:.2f}, τ*={mz['tau_selected_on_valid']}): "
              f"test ROI={mz['test_roi']*100:.1f}% (2024={mz['test_2024_roi']*100:.1f}% 2025={mz['test_2025_roi']*100:.1f}%) "
              f"n={mz['test_bets']:,} CLV={(mz['test_mean_clv'] or 0)*100:+.1f}%")
    print("\n● π9 grid: validで選んだα=", a_sel, " → test ΔR²=", peak_dr2)
    print("● 判定:", verdict)


if __name__ == "__main__":
    main()
