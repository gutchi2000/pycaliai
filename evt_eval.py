# -*- coding: utf-8 -*-
"""
evt_eval.py — EVT(極値統計)特徴の直交性 + 予測上乗せ + H2交互作用を厳密評価
==============================================================================
physics_eval.py と同一の BASE_FEATS / 時系列分割 / LightGBM を流用して比較可能にする。

問い (事前登録):
  H1 (上乗せ): fukusho_flag の test(2024+) AUC/logloss を baseline 超で改善するか?
              part_corr(feature, y - p_base) が死亡帯(<0.01)を超え kl_lscap(0.0235)に迫るか?
  H2 (交互作用/符号): 分散の効果は「非本命で勝率+, 本命で-」と再配分するはず。
              p_base 分位 × evt_sigma 高低 で top3率リフトの符号反転を見る。
  おまけ: 理論が最も鋭い「勝利(着順==1, argmax そのもの)」でも AUC を測る。

反証で捨てる: 上記が立たなければ「baseline が既に同じ情報を持っていた or 織り込み済み」。

出力: reports/evt_feats_eval.json + コンソール
実行: PYTHONUTF8=1 python evt_eval.py
"""
from __future__ import annotations
import io, json, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss

warnings.filterwarnings("ignore")
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
except Exception:
    pass

from physics_eval import BASE_FEATS, fit_eval, _rid16
from corr_features import add_style, COL_RID

BASE = Path(__file__).parent
MASTER = BASE / "data/master_v2_20130105-20251228.csv"
EVT = BASE / "data/evt_feats.parquet"
OUT = BASE / "reports/evt_feats_eval.json"
COL_BAN = "馬番"
COL_JYUN = "着順"

# evt_sigma(生)は NaN多/n依存なので収縮版 _s と z を使う。evt_mu は地力(baselineと重複しうる)
# が gain で判定。
EVT_FEATS = ["evt_n", "evt_mu", "evt_sigma_s", "evt_sigma_gz", "evt_sigma_rz",
             "evt_ceiling", "evt_floor", "evt_range",
             "evt_sigma_resid", "evt_sigma_resid_gz"]
# baseline 内の「平均/地力・末脚」の代理 (重複度チェック用)
OVERLAP_REFS = ["prev_hosei9", "closing_power", "kako5_avg_agari3f", "course_top3_rate"]


def load() -> pd.DataFrame:
    need = list(dict.fromkeys(
        [COL_RID, COL_BAN, COL_JYUN, "split", "fukusho_flag", "前走出走頭数",
         "前1角", "前2角", "前3角", "前4角"] + BASE_FEATS))
    need = [c for c in need if c not in ("style_rank", "pace_pressure")]
    df = pd.read_csv(MASTER, encoding="utf-8-sig",
                     usecols=lambda c: c in need, low_memory=False)
    df = add_style(df)
    df["rid16"] = df[COL_RID].map(_rid16)
    df["ban"] = pd.to_numeric(df[COL_BAN], errors="coerce").astype("Int64")
    evt = pd.read_parquet(EVT)
    df = df.merge(evt, on=["rid16", "ban"], how="left")
    df["fukusho_flag"] = pd.to_numeric(df["fukusho_flag"], errors="coerce")
    df["win"] = (pd.to_numeric(df[COL_JYUN], errors="coerce") == 1).astype("Int8")
    df = df[df["fukusho_flag"].isin([0, 1])].copy()
    for c in BASE_FEATS + EVT_FEATS + OVERLAP_REFS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def main():
    df = load()
    n = {s: int((df["split"] == s).sum()) for s in ["train", "valid", "test"]}
    cov = df.loc[df["split"] == "test", EVT_FEATS].notna().mean().mean() * 100
    print(f"[rows] train={n['train']:,} valid={n['valid']:,} test={n['test']:,}  "
          f"EVT test coverage(平均)={cov:.1f}%")

    print("\n=== H1: モデル比較 (test=2024+, fukusho_flag) ===")
    res = {}
    res["base"], p_base, m_base, te = fit_eval(df, BASE_FEATS, "baseline")
    res["base+evt"], p_evt, m_evt, _ = fit_eval(df, BASE_FEATS + EVT_FEATS, "base+evt")
    d_auc = res["base+evt"]["auc"] - res["base"]["auc"]
    d_ll = res["base+evt"]["logloss"] - res["base"]["logloss"]
    print(f"\n  ΔAUC +evt = {d_auc:+.5f}   Δlogloss = {d_ll:+.6f} (負=改善)")
    print(f"  (参考: physics +keller は ΔAUC +0.00159 / kl_lscap part_corr 0.0235)")

    # --- 直交性: part_corr(feature, y - p_base) + baseline代理との重複 ---
    print("\n=== 直交性 part_corr(feature, y - p_base) on test / baseline代理との|相関| ===")
    resid = te["fukusho_flag"].values - p_base
    ortho = {}
    for f in EVT_FEATS:
        v = te[f].astype(float)
        mask = v.notna().values
        if mask.sum() < 1000:
            continue
        pc = float(np.corrcoef(v.values[mask], resid[mask])[0, 1])
        ov = {}
        for g in OVERLAP_REFS:
            if g not in te.columns:
                continue
            gg = te[g].astype(float)
            mm = (v.notna() & gg.notna()).values
            if mm.sum() > 1000:
                ov[g] = float(np.corrcoef(v.values[mm], gg.values[mm])[0, 1])
        omax = max((abs(x) for x in ov.values()), default=0.0)
        ortho[f] = dict(part_corr=pc, max_overlap=omax, overlap=ov)
        print(f"  {f:14s} part_corr={pc:+.4f}  max|overlap|={omax:.3f}")

    # --- H2: p_base 分位 × evt_sigma 高低 で勝率リフトの符号反転 ---
    #   生σ(evt_sigma_gz)と detrend残差σ(evt_sigma_resid_gz)を比較。
    #   理論(純ノイズσ): 非本命(低p_base)で +, 本命(高p_base)で - の符号反転を期待。
    print("\n=== H2: 本命度(p_base五分位) × σ高低 → 勝率/複勝率リフト ===")
    h2 = {}
    for label, col in [("生σ", "evt_sigma_gz"), ("detrend残差σ", "evt_sigma_resid_gz")]:
        print(f"\n  [{label}] 理論: 非本命で +, 本命で - の符号反転")
        tt = pd.DataFrame({
            "p_base": p_base, "y": te["fukusho_flag"].values,
            "win": pd.to_numeric(te["win"], errors="coerce").values,
            "sig": te[col].values,
        }).dropna(subset=["sig"])
        tt["fav_q"] = pd.qcut(tt["p_base"], 5, labels=False, duplicates="drop")
        h2[col] = {}
        for q in sorted(tt["fav_q"].dropna().unique()):
            sub = tt[tt["fav_q"] == q]
            med = sub["sig"].median()
            hi = sub[sub["sig"] >= med]; lo = sub[sub["sig"] < med]
            lift3 = float(hi["y"].mean() - lo["y"].mean())
            liftw = float(hi["win"].mean() - lo["win"].mean())
            h2[col][int(q)] = dict(p_base_mean=float(sub["p_base"].mean()), n=int(len(sub)),
                                   top3_lift=lift3, win_lift=liftw)
            print(f"    fav_q={int(q)} p_base≈{sub['p_base'].mean():.3f} n={len(sub):6d}  "
                  f"複勝リフト={lift3:+.4f}  勝率リフト={liftw:+.4f}")

    # --- おまけ: 勝利(argmax)ラベルでの ablation ---
    print("\n=== おまけ: 勝利(着順==1)ラベルでの AUC ===")
    dfw = df.copy()
    dfw["fukusho_flag"] = pd.to_numeric(dfw["win"], errors="coerce")  # fit_eval はこの列をyに使う
    rw_b, pw_b, _, tew = fit_eval(dfw, BASE_FEATS, "win-baseline")
    rw_e, _, _, _ = fit_eval(dfw, BASE_FEATS + EVT_FEATS, "win-base+evt")
    d_auc_w = rw_e["auc"] - rw_b["auc"]
    print(f"  ΔAUC(win) +evt = {d_auc_w:+.5f}")

    # --- gain シェア ---
    imp = dict(zip(m_evt.feature_name(), m_evt.feature_importance(importance_type="gain")))
    tot = sum(imp.values()) or 1
    evt_imp = {k: float(imp.get(k, 0)) for k in EVT_FEATS}
    share = sum(evt_imp.values()) / tot * 100
    print(f"\n=== EVT gain シェア (base+evt 内) = {share:.2f}% ===")
    for k, v in sorted(evt_imp.items(), key=lambda x: -x[1]):
        print(f"  {k:14s} gain={v:,.0f} ({v/tot*100:.2f}%)")

    out = dict(rows=n, evt_test_cov_pct=float(cov),
               models={k: res[k] for k in res},
               delta_auc_fukusho=float(d_auc), delta_logloss_fukusho=float(d_ll),
               delta_auc_win=float(d_auc_w),
               orthogonality=ortho, h2_interaction=h2,
               evt_gain_share_pct=float(share), evt_importance=evt_imp)
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {OUT}")

    # --- 判定 ---
    print("\n=== 判定 ===")
    verdict_h1 = d_auc > 0.002
    best_pc = max((abs(o["part_corr"]) for o in ortho.values()), default=0)
    verdict_pc = best_pc > 0.01
    print(f"  H1 ΔAUC>{0.002}: {'○' if verdict_h1 else '×'} ({d_auc:+.5f})")
    print(f"  直交 best|part_corr|>0.01: {'○' if verdict_pc else '×'} ({best_pc:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
