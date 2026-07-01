"""
market_strat_exp.py — 市場合成エッジ(ΔR²+0.007)の期間・条件層別探索
====================================================================
設計思想: モデル(f=v6)・市場合成(α,β,τ)は全期間で固定(専門化しない)。評価だけ層別し、
全体平均で薄まったエッジが特定クラスタで控除を超える塊として在るかを探す。

★罠防止(生命線): 軸は事前固定4つ(時期/開催/世代/頭数)＋2軸クロス少数のみ。
  test内 MIN_BETS=500 未満の層は判定対象外。発見(≤2023)→検証(test)分離。多重比較 Bonferroni 補正。

α*,β,τ* は stage1b と同じ入れ子(β:≤2022 / α:valid grid選択 / τ:≤2023選択)で全体決定 → 各層 test で評価。
指標: ΔR²(層) / 単勝ROI(層, bootstrap CI) / CLV(層)。baseline=全体(ΔR²+0.007/ROI70.8%)。
判定: ≤2023で控除超の層が test でも再現し Bonferroni 補正後 ROI-CI下限>控除(80%) かつ CLV>0 → 勝てるポケット。
出力: reports/market_strat_exp.json  実行: PYTHONUTF8=1 python market_strat_exp.py
"""
from __future__ import annotations
import json, warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import pl_probs as PL
import stage1_benter_blend as S1
import stage1b_shrinkage_alpha as S1B
from joint_calibration_v6 import apply_encoders, COL_RID, COL_JYUN, COL_BAN, MASTER_CSV

warnings.filterwarnings("ignore")
BASE = Path(__file__).parent
OUT_JSON = BASE / "reports/market_strat_exp.json"
rng = np.random.default_rng(42)
LOCAL = {"新潟", "小倉", "福島", "函館", "札幌"}
MIN_BETS = 500
N_BOOT = 300
TAKEOUT = 0.80

LAYERS = {
    "全体": lambda r: True,
    "夏(6-9)": lambda r: r["month"] in (6, 7, 8, 9),
    "非夏": lambda r: r["month"] not in (6, 7, 8, 9),
    "中央場": lambda r: not r["local"],
    "ローカル": lambda r: r["local"],
    "3歳限定": lambda r: r["age3"],
    "非3歳": lambda r: not r["age3"],
    "多頭14+": lambda r: r["nentry"] >= 14,
    "少頭<14": lambda r: r["nentry"] < 14,
    "夏xローカル": lambda r: r["month"] in (6, 7, 8, 9) and r["local"],
    "3歳x少頭": lambda r: r["age3"] and r["nentry"] < 14,
    "ローカルx少頭": lambda r: r["local"] and r["nentry"] < 14,
    "夏x少頭": lambda r: r["month"] in (6, 7, 8, 9) and r["nentry"] < 14,
}


def build_races_strat():
    snaps = S1.load_tanpuk_snapshots()
    b = joblib.load(BASE / "models/unified_rank_v6.pkl")
    model, feats, encs = b["model"], b["feature_cols"], b["encoders"]
    cal = joblib.load(BASE / "models/pl_calibrators_v6.pkl"); cal = cal.get("calibrators", cal)
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df = df.dropna(subset=[COL_JYUN, COL_RID, "split"]).copy()
    enc = apply_encoders(df, encs)
    X = enc[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    df["_score"] = model.predict(X)
    races = []
    for rid, g in df.groupby(COL_RID, sort=False):
        if len(g) < 5: continue
        rid16 = str(rid)[:16]
        if rid16 not in snaps: continue
        g = g.sort_values(COL_BAN).reset_index(drop=True)
        ban = g[COL_BAN].astype(int).values
        w = PL.pl_weights(g["_score"].values)
        f_all = cal["tansho"].predict(PL.all_tansho(w)) if "tansho" in cal else PL.all_tansho(w)
        f_by = {int(ban[i]): max(float(f_all[i]), 1e-9) for i in range(len(ban))}
        jy = g[COL_JYUN].astype(int).values; win_ban = int(ban[int(np.argmin(jy))])
        sp = snaps[rid16]
        common = [k for k in sp["o9"] if k in sp["oc"] and k in f_by]
        if win_ban not in common or len(common) < 5: continue
        keys = sorted(common)
        f = np.array([f_by[k] for k in keys]); p9 = S1.devig(sp["o9"], keys)
        o9 = np.array([sp["o9"][k] for k in keys]); oc = np.array([sp["oc"][k] for k in keys])
        nent = pd.to_numeric(g["出走頭数"], errors="coerce").iloc[0]
        races.append({"logf": np.log(f), "logp9": np.log(p9), "o9": o9, "oc": oc,
                      "win": keys.index(win_ban), "year": int(rid16[:4]),
                      "place": str(g["場所"].iloc[0]), "local": str(g["場所"].iloc[0]) in LOCAL,
                      "age3": str(g["年齢限定"].iloc[0]) == "３歳",
                      "nentry": int(nent) if pd.notna(nent) else len(keys), "month": int(rid16[4:6])})
    print(f"  races={len(races):,}")
    return races


def boot_roi_ci(races, a, b, tau, pct_lo):
    rois = []
    n = len(races)
    if n == 0: return None, None
    for _ in range(N_BOOT):
        samp = [races[i] for i in rng.choice(n, n, replace=True)]
        rr = S1.bet_roi(samp, a, b, "logp9", "o9", tau)
        if rr["bets"] > 0: rois.append(rr["roi"])
    if not rois: return None, None
    return round(float(np.percentile(rois, pct_lo)), 4), round(float(np.percentile(rois, 50)), 4)


def main():
    print("[build] races + strata ...")
    races = build_races_strat()
    tr = [r for r in races if r["year"] <= 2022]
    vl = [r for r in races if r["year"] == 2023]
    tv = [r for r in races if r["year"] <= 2023]
    te = [r for r in races if r["year"] >= 2024]

    # 全体 (α:valid grid選択, β:≤2023, τ:≤2023選択)  — stage1b と同規律
    pk_tr = S1.pack(tr, "logp9"); pk_vl = S1.pack(vl, "logp9"); pk_tv = S1.pack(tv, "logp9")
    _, bm = S1.fit_pk(pk_tv, fix="a0")
    curve = [(al, S1.mcfadden_r2_pk(pk_vl, al, S1B.fit_beta_at_alpha(pk_tr, al))) for al in S1B.ALPHA_GRID]
    a_star = max(curve, key=lambda x: x[1])[0]
    b_star = S1B.fit_beta_at_alpha(pk_tv, a_star)
    tau, _ = S1B.select_tau_on_valid(vl, a_star, b_star, "logp9", "o9")
    print(f"  α*={a_star} β*={b_star:.3f} τ*={tau}  (β market-only={bm:.3f})")

    n_layers = len([k for k in LAYERS if k != "全体"])
    pct_lo = 100 * (0.05 / n_layers) / 2   # Bonferroni 片側下限 percentile
    results = {"protocol": "f=v6,π9=9時. α:valid grid / β,τ:≤2023. 全体固定→層別test評価.",
               "alpha_star": a_star, "beta_star": round(b_star, 4), "tau_star": tau,
               "MIN_BETS": MIN_BETS, "n_layers_corrected": n_layers,
               "bonferroni_pct_lower": round(pct_lo, 3), "takeout": TAKEOUT, "layers": {}}

    def r2_layer(rs):
        if len(rs) < 5: return None
        pk = S1.pack(rs, "logp9")
        return round(S1.mcfadden_r2_pk(pk, a_star, b_star) - S1.mcfadden_r2_pk(pk, 0.0, bm), 5)

    pockets = []
    for name, fn in LAYERS.items():
        te_l = [r for r in te if fn(r)]; tv_l = [r for r in tv if fn(r)]
        roi_te = S1.bet_roi(te_l, a_star, b_star, "logp9", "o9", tau)
        roi_tv = S1.bet_roi(tv_l, a_star, b_star, "logp9", "o9", tau)
        ci_lo, ci_md = boot_roi_ci(te_l, a_star, b_star, tau, pct_lo)
        entry = {"n_races_test": len(te_l), "dR2_test": r2_layer(te_l),
                 "roi_test": roi_te["roi"], "bets_test": roi_te["bets"], "clv_test": roi_te["mean_clv"],
                 "roi_ci_lo_bonf": ci_lo, "roi_median_boot": ci_md,
                 "roi_le2023_discovery": roi_tv["roi"], "bets_le2023": roi_tv["bets"]}
        eligible = roi_te["bets"] >= MIN_BETS
        discovered = roi_tv["roi"] > TAKEOUT and roi_tv["bets"] >= MIN_BETS
        confirmed = eligible and (ci_lo is not None and ci_lo > TAKEOUT) and (roi_te["mean_clv"] or 0) > 0
        entry["eligible"] = bool(eligible); entry["discovered_le2023"] = bool(discovered)
        entry["confirmed_test_bonf"] = bool(confirmed and discovered)
        if entry["confirmed_test_bonf"]: pockets.append(name)
        results["layers"][name] = entry

    if pockets:
        verdict = (f"POCKET_FOUND: {pockets} が ≤2023で控除超を発見 & test で Bonferroni補正後ROI-CI下限>{TAKEOUT*100:.0f}% "
                   "かつ CLV>0 で再現。市場が荒れる勝てるポケット。最優先で深掘り。")
    else:
        any_disc = [n for n, e in results["layers"].items() if e["discovered_le2023"]]
        verdict = (f"NO_POCKET: 補正後 test で控除超を再現した層なし。≤2023で光った層={any_disc or 'なし'} は "
                   "test/補正で消失。市場は層別しても効率的=この弾も冗長。")
    results["verdict"] = verdict
    OUT_JSON.parent.mkdir(exist_ok=True)
    json.dump(results, open(OUT_JSON, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print("\n" + "=" * 96)
    print(f"全体最適: α*={a_star} β*={b_star:.3f} τ*={tau}  Bonferroni片側下限={pct_lo:.2f}%tile (n_layers={n_layers})")
    print(f"{'層':14s}{'n(test)':>8s}{'ΔR²':>9s}{'ROI_te':>8s}{'CI下限':>8s}{'CLV':>7s}{'bets':>6s}{'≤23ROI':>8s}  発見/再現")
    for name, e in results["layers"].items():
        cl = f"{e['roi_ci_lo_bonf']*100:.0f}" if e["roi_ci_lo_bonf"] is not None else "-"
        clv = f"{(e['clv_test'] or 0)*100:+.0f}" if e["clv_test"] is not None else "-"
        print(f"{name:14s}{e['n_races_test']:>8,}{(e['dR2_test'] or 0):>+9.4f}{e['roi_test']*100:>7.1f}%"
              f"{cl:>7s}%{clv:>6s}%{e['bets_test']:>6,}{e['roi_le2023_discovery']*100:>7.1f}%"
              f"  {e['discovered_le2023']}/{e['confirmed_test_bonf']}")
    print(f"\n判定: {verdict}")
    print(f"[saved] {OUT_JSON}")


if __name__ == "__main__":
    main()
