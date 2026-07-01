# -*- coding: utf-8 -*-
"""
eval_market_feats.py — TARGET 追加4列(前走人気/前走単勝オッズ/市場取引価格/コーナー数)の限界AUC
================================================================================
master 再生成の前に「足す価値があるか」を physics_eval と同一ハーネスで先に測る。
仮説: 前走人気/オッズ＝leak-safe な「市場記憶」で、フォーム系(+0.002)を超える上乗せ。
      市場取引価格＝新馬/軽量級(市場が非効率)で限定的に効くか。
実行: PYTHONUTF8=1 python eval_market_feats.py
"""
from __future__ import annotations
import glob, io, re, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
import physics_eval as PE   # load(), fit_eval(), BASE_FEATS

# aux 列は順序固定（両ファイル同一ヘッダ）。位置で参照して符号化問題を回避。
IDX = {"rid": 7, "ban": 10, "zenso_ninki": 19, "zenso_odds": 20,
       "mkt_price": 24, "mkt_place": 25, "corner_n": 26}


def _rid16(x):
    return re.sub(r"\D", "", str(x))[:16]


def load_aux():
    frames = []
    for f in sorted(glob.glob(str(BASE / "data/aux/market_feats_*.csv"))):
        d = pd.read_csv(f, encoding="cp932", header=0, low_memory=False)
        sub = pd.DataFrame({k: d.iloc[:, i] for k, i in IDX.items()})
        frames.append(sub)
    a = pd.concat(frames, ignore_index=True)
    a["rid16"] = a["rid"].map(_rid16)
    a["ban"] = pd.to_numeric(a["ban"], errors="coerce").astype("Int64")
    a["zenso_ninki"] = pd.to_numeric(a["zenso_ninki"], errors="coerce")
    a["zenso_odds"] = pd.to_numeric(a["zenso_odds"], errors="coerce")
    a["zenso_odds_log"] = np.log(a["zenso_odds"].clip(lower=1.0))
    a["zenso_impl"] = 1.0 / a["zenso_odds"].clip(lower=1.0)           # 前走 市場含意prob
    # 市場取引価格 "2730万" → 2730
    mp = a["mkt_price"].astype(str).str.replace("万", "", regex=False).str.replace(",", "", regex=False)
    a["mkt_price_v"] = pd.to_numeric(mp, errors="coerce")
    a["mkt_price_log"] = np.log1p(a["mkt_price_v"])
    a["has_auction"] = a["mkt_price_v"].notna().astype(float)
    a["corner_n"] = pd.to_numeric(a["corner_n"], errors="coerce")
    return a.drop_duplicates(["rid16", "ban"], keep="last")


MKT_MEM = ["zenso_ninki", "zenso_odds_log", "zenso_impl"]
PRICE = ["mkt_price_log", "has_auction"]
CORNER = ["corner_n"]


def main():
    df = PE.load()                                  # master + BASE_FEATS + style + kl_/ou_(未使用)
    a = load_aux()
    df = df.merge(a[["rid16", "ban"] + MKT_MEM + PRICE + CORNER], on=["rid16", "ban"], how="left")
    te = df[df["split"] == "test"]
    for grp, fs in [("市場記憶", MKT_MEM), ("市場取引価格", PRICE), ("コーナー数", CORNER)]:
        cov = te[fs].notna().mean().mean() * 100
        print(f"  {grp:10s} test非欠損={cov:.1f}%")
    print(f"  zenso_ninki coverage(test)={te['zenso_ninki'].notna().mean()*100:.1f}%  "
          f"has_auction率(test)={te['has_auction'].mean()*100:.1f}%")

    print("\n=== test=2024+ fukusho_flag AUC ===")
    base, p_base, _, te = PE.fit_eval(df, PE.BASE_FEATS, "baseline(no odds)")
    bm, _, _, _ = PE.fit_eval(df, PE.BASE_FEATS + MKT_MEM, "+市場記憶(前走人気/odds)")
    bp, _, _, _ = PE.fit_eval(df, PE.BASE_FEATS + PRICE, "+市場取引価格")
    bc, _, _, _ = PE.fit_eval(df, PE.BASE_FEATS + CORNER, "+コーナー数")
    ball, p_all, m_all, _ = PE.fit_eval(df, PE.BASE_FEATS + MKT_MEM + PRICE + CORNER, "+全部")
    print(f"\n  ΔAUC  市場記憶={bm['auc']-base['auc']:+.5f}  取引価格={bp['auc']-base['auc']:+.5f}"
          f"  コーナー={bc['auc']-base['auc']:+.5f}  全部={ball['auc']-base['auc']:+.5f}")
    print(f"  Δlogloss 市場記憶={bm['logloss']-base['logloss']:+.6f}  全部={ball['logloss']-base['logloss']:+.6f} (負=改善)")
    print(f"  (基準: フォーム物理フルセット +both=+0.00205 / leak-safe現走市場価格=+0.034)")

    # 直交性: ベース予測統制下の part-corr
    resid = te["fukusho_flag"].values - p_base
    print("\n=== 直交性 part-corr(feature, y - p_base) on test ===")
    for f in MKT_MEM + PRICE + CORNER:
        v = te[f].astype(float); mask = v.notna().values
        if mask.sum() < 1000:
            continue
        pc = np.corrcoef(v.values[mask], resid[mask])[0, 1]
        print(f"  {f:16s} part_corr={pc:+.4f}  n={int(mask.sum())}")

    # 市場取引価格: 軽量級(出走履歴薄)で効くか層別
    print("\n=== 市場取引価格 リフト: 出走履歴の薄い馬で効くか ===")
    if "kako5_race_count" in te.columns:
        sub = te[te["has_auction"] == 1].copy()
        sub["rc"] = pd.to_numeric(sub["kako5_race_count"], errors="coerce")
        for name, mask in [("履歴薄 race_count<=2", sub["rc"] <= 2), ("履歴厚 race_count>=4", sub["rc"] >= 4)]:
            s = sub[mask]
            if len(s) < 500:
                continue
            try:
                s["pq"] = pd.qcut(s["mkt_price_v"], 4, labels=False, duplicates="drop")
                tab = s.groupby("pq")["fukusho_flag"].mean()
                print(f"  {name}: n={len(s)} top3率 価格Q0={tab.iloc[0]:.4f} → Q{tab.index[-1]}={tab.iloc[-1]:.4f}")
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
