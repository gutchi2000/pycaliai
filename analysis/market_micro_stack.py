# -*- coding: utf-8 -*-
"""
market_micro_stack.py — 市場の内部情報 (朝→発走前の動き・票数比) を積み上げた勝率モデルで、期待値上位だけ買うと控除率を超えるか (2026-11-12)
=========================================================================================================
発見 (同日): (1) 朝 9 時に売れていて発走前に売れなくなった馬は過小評価 (単勝 +10〜15pt、3 期再現)
            (2) 複勝票数/単勝票数 が大きいレースの 1 番人気は 3 着内率が高い
これらと既知の癖 (本命-大穴バイアス、頭数、R 番号、月) を 1 つの勝率モデルにまとめ、年次ウォークフォワードで
『予測勝率 × 発走前オッズ』(= 判断時点で計算できる期待値) の上位だけを買い、確定オッズで決済する。
AI (v6) のスコアは入れない (市場の内部情報だけでどこまで行くかを測る)。2023-25 だけ AI スコア併用版も見る。
基盤: data/_joint/tanpuk_all_drift.parquet (全馬の 朝/発走前/確定 単勝) + tanpuk_fav.parquet (票数)。
出力: reports/market_micro_stack.md (+ docs/research 控え)、OOS 予測 data/_joint/market_micro_oos.parquet
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "reports" / "market_micro_stack.md"
FEATS = ["l_pre", "l_9", "drift", "rk9", "rkp", "rk_chg", "n", "pi_pre", "pi_9", "rel_fav", "pool_ratio", "l_pool", "fav_pre", "fav_drift", "R", "month", "place"]


def frame() -> pd.DataFrame:
    d = pd.read_parquet(BASE / "data" / "_joint" / "tanpuk_all_drift.parquet")
    f = pd.read_parquet(BASE / "data" / "_joint" / "tanpuk_fav.parquet", columns=["rid", "pool_tan_pre", "pool_fuku_pre", "tan_pre"])
    d = d.merge(f.rename(columns={"tan_pre": "fav_pre"}), on="rid", how="left")
    d["l_pre"] = np.log(d.o_pre); d["l_9"] = np.log(d.o_win9); d["drift"] = d.l_pre - d.l_9
    g = d.groupby("rid")
    d["rk9"] = g.o_win9.rank(method="first"); d["rkp"] = g.o_pre.rank(method="first"); d["rk_chg"] = d.rkp - d.rk9
    d["pi_pre"] = (1 / d.o_pre) / g.o_pre.transform(lambda s: (1 / s).sum())
    d["pi_9"] = (1 / d.o_win9) / g.o_win9.transform(lambda s: (1 / s).sum())
    d["rel_fav"] = d.l_pre - np.log(d.fav_pre)
    d["pool_ratio"] = d.pool_fuku_pre / d.pool_tan_pre; d["l_pool"] = np.log(d.pool_tan_pre.clip(lower=1))
    d["fav_drift"] = g.drift.transform("min")
    d["R"] = d.rid.str[14:16].astype(int); d["month"] = d.rid.str[4:6].astype(int); d["place"] = d.rid.str[8:10].astype(int)
    d["date"] = d.rid.str[:8]
    return d


def walk_forward(d: pd.DataFrame, feats: list[str], first=2017) -> pd.Series:
    import lightgbm as lgb
    p = pd.Series(np.nan, index=d.index)
    for y in range(first, 2026):
        tr, te = d[d.year < y], d[d.year == y]
        m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=300, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.8, reg_lambda=10.0, verbose=-1, random_state=0)
        m.fit(tr[feats], tr.is_win)
        p.loc[te.index] = m.predict_proba(te[feats])[:, 1]
    return p


def boot(g: pd.DataFrame, col: str, seed=0):
    a = g.groupby("date")[col].agg(["sum", "size"]).values
    if len(a) < 5:
        return np.nan, np.nan
    rng = np.random.default_rng(seed); i = rng.integers(0, len(a), (3000, len(a)))
    v = a[i, 0].sum(1) / a[i, 1].sum(1)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def ev_table(d: pd.DataFrame, evcol: str, title: str, lo_odds=1.0, hi_odds=30.0) -> list[str]:
    md = [f"### {title}", "", "| 期待値の閾値 (判断時点) | n | 年あたり | 勝率 | 平均確定オッズ | 単勝回収率 [95% CI] | 年別 (17→25) |", "|---|---|---|---|---|---|---|"]
    x = d[d[evcol].notna() & (d.o_pre >= lo_odds) & (d.o_pre < hi_odds)]
    yrs = sorted(x.year.unique())
    for thr in (0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20):
        g = x[x[evcol] >= thr]
        if len(g) < 100:
            continue
        lo, hi = boot(g, "tan_pay")
        yr = " ".join(f"{g[g.year == y].tan_pay.mean():.0f}" if (g.year == y).sum() >= 20 else "-" for y in yrs)
        md.append(f"| ≥ {thr:.2f} | {len(g):,} | {len(g) / len(yrs):.0f} | {100 * g.is_win.mean():.1f}% | {g.o_fin.mean():.1f} | **{g.tan_pay.mean():.1f}** [{lo:.0f},{hi:.0f}] | {yr} |")
    md.append("")
    return md


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    d = frame()
    print(f"{len(d):,} horses", flush=True)
    base_feats = ["l_pre", "n", "pi_pre", "rkp", "rel_fav"]
    d["p_base"] = walk_forward(d, base_feats)
    d["p_full"] = walk_forward(d, FEATS)
    for c in ("p_base", "p_full"):
        s = d.groupby("rid")[c].transform("sum"); d[c + "_n"] = d[c] / s
        d["ev_" + c] = d[c + "_n"] * d.o_pre
    oos = d[d.p_full.notna()]
    from sklearn.metrics import log_loss
    ll = {k: log_loss(oos.is_win, oos[k].clip(1e-6, 1 - 1e-6)) for k in ("p_base_n", "p_full_n")}
    ll["市場 (発走前、比例正規化)"] = log_loss(oos.is_win, oos.pi_pre.clip(1e-6, 1 - 1e-6))
    pi_fin = (1 / oos.o_fin) / oos.groupby("rid").o_fin.transform(lambda s: (1 / s).sum())
    ll["市場 (確定、比例正規化)"] = log_loss(oos.is_win, pi_fin.clip(1e-6, 1 - 1e-6))

    md = [f"# 市場の内部情報を積み上げた勝率モデルと期待値買い (market_micro_stack, {pd.Timestamp.now():%Y-%m-%d})", "",
          f"全馬 {len(d):,} 頭 (2013-2025、8 頭以上)。年次ウォークフォワード (前年までで学習)、2017-2025 の 9 年が OOS。単勝を確定オッズで決済。",
          "期待値 = 正規化した予測勝率 × 発走約 28 分前の単勝オッズ (判断時点で計算できる値)。", "",
          "## 1. 勝率の当たり具合 (OOS logloss、小さいほど良い)", ""] + [f"- {k}: {v:.5f}" for k, v in ll.items()] + [""]
    md += ["## 2. 期待値上位だけ買ったときの単勝回収率", ""]
    md += ev_table(d, "ev_p_base", "基準モデル (発走前オッズと頭数だけ = 本命-大穴バイアスの補正のみ)")
    md += ev_table(d, "ev_p_full", "積み上げモデル (+ 朝オッズ・動き・人気順位の変化・票数比・R・月・場)")
    md += ev_table(d, "ev_p_full", "積み上げモデル、発走前 2.5〜15 倍に限定", 2.5, 15.0)
    md += ev_table(d, "ev_p_full", "積み上げモデル、発走前 10 倍未満に限定", 1.0, 10.0)

    # AI 併用 (2023-25)
    sc = pd.read_parquet(BASE / "data" / "_ev_grid_scores.parquet")
    sc["rid"] = sc["レースID(新/馬番無)"].astype(str).str[:16]; sc["ban"] = pd.to_numeric(sc["馬番"], errors="coerce")
    sc["ai_rk"] = sc.groupby("rid")._score.rank(ascending=False, method="first")
    t = d[d.year >= 2023].merge(sc[["rid", "ban", "ai_rk"]], on=["rid", "ban"], how="inner")
    md += ["## 3. AI の順位と重ねる (2023-25、積み上げモデルの期待値 ≥ 1.0 の馬)", "", "| 条件 | n | 勝率 | 単勝回収率 [95% CI] |", "|---|---|---|---|"]
    e = t[(t.ev_p_full >= 1.0) & (t.o_pre < 30)]
    for name, g in (("全部", e), ("AI 1 位", e[e.ai_rk == 1]), ("AI 2-3 位", e[e.ai_rk.between(2, 3)]), ("AI 4 位以下", e[e.ai_rk >= 4]),
                    ("AI 3 位以内", e[e.ai_rk <= 3])):
        if len(g) >= 30:
            lo, hi = boot(g, "tan_pay"); md.append(f"| {name} | {len(g):,} | {100 * g.is_win.mean():.1f}% | {g.tan_pay.mean():.1f} [{lo:.0f},{hi:.0f}] |")
    md.append("")
    import lightgbm as lgb
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=300, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                           reg_lambda=10.0, verbose=-1, random_state=0, importance_type="gain").fit(d[d.year < 2025][FEATS], d[d.year < 2025].is_win)
    imp = pd.Series(m.feature_importances_, index=FEATS).sort_values(ascending=False)
    md += ["## 4. 効いている特徴 (gain)", ""] + [f"- {k}: {100 * v / imp.sum():.1f}%" for k, v in imp.items()] + [""]
    OUT.write_bytes("\n".join(md).encode("utf-8"))
    d[["rid", "ban", "year", "date", "o_win9", "o_pre", "o_fin", "is_win", "top3", "tan_pay", "fuku_pay", "drift", "rk9", "rkp", "pool_ratio",
       "p_base_n", "p_full_n", "ev_p_base", "ev_p_full"]].to_parquet(BASE / "data" / "_joint" / "market_micro_oos.parquet")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
