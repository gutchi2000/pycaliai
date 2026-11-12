# -*- coding: utf-8 -*-
"""
pinned_floor_search_v2.py — 発走約 28 分前のオッズ・票数で「堅い 1 番人気の複勝」を当て分ける (2026-11-12)
=====================================================================================
v1 (朝 9 時) に対して: 判断時点を発走前スナップ (TANPUK 区分 1 の最終、発走の 26-30 分前) に進め、
複勝票数/単勝票数 (複勝に厚く入っているか)、朝からのオッズの動き、複勝 Lo/Hi を特徴に足す。
基盤: data/_joint/tanpuk_fav.parquet (analysis/build_tanpuk_fav.py)。採点は確定複勝払戻。
出力: reports/pinned_floor_search_v2.md (+ docs/research 控え)、OOS 予測 data/_joint/pinned_floor_v2_oos.parquet
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from analysis.pinned_floor_search import CAT, NUM, PERIODS, boot_ci, join_master, period_of, walk_forward  # noqa: E402

OUT = BASE / "reports" / "pinned_floor_search_v2.md"
ZONE = 1.9
MKT_FEATS = ["tan_pre", "flo_pre", "fhi_pre", "f_spread", "o2_pre", "o3_pre", "o4_pre", "flo2_pre", "flo3_pre", "o2r", "pool_tan_pre", "pool_fuku_pre",
             "pool_ratio", "drift", "drift_flo", "o_win9", "fuku_lo9", "n", "month", "min_before"]


def frame() -> pd.DataFrame:
    d = pd.read_parquet(BASE / "data" / "_joint" / "tanpuk_fav.parquet")
    d = d[(d.n >= 8) & (d.tan_pre <= ZONE)].copy()
    mk = pd.read_parquet(BASE / "data" / "_joint" / "mkt_horses.parquet", columns=["rid", "ban", "o_win9", "fuku_lo9"])
    d = d.merge(mk, on=["rid", "ban"], how="left")
    d["drift"] = np.log(d.tan_pre / d.o_win9)
    d["drift_flo"] = d.flo_pre - d.fuku_lo9
    d["pool_ratio"] = d.pool_fuku_pre / d.pool_tan_pre
    d["o2r"] = d.o2_pre / d.tan_pre
    d["f_spread"] = d.fhi_pre - d.flo_pre
    d["month"] = d.rid.str[4:6].astype(int)
    return d


def band_table(d: pd.DataFrame, col: str, title: str, cuts=(0.95, 0.9, 0.8, 0.7, 0.5)) -> list[str]:
    md = [f"### {title}", "", "| 予測の帯 (閾値は前年までの分布で決定) | n | 年あたり | 3着内率 | 的中時の平均払戻 | 回収率 [95% CI] | 年別回収率 (17→25) |", "|---|---|---|---|---|---|---|"]
    x = d[d[col].notna()].copy(); x["q"] = np.nan
    for y in sorted(x.year.unique()):
        prev = x[x.year < y][col]
        if len(prev) < 300:
            continue
        v = x.loc[x.year == y, col]
        x.loc[x.year == y, "q"] = np.searchsorted(np.sort(prev.values), v.values) / len(prev)
    x = x[x.q.notna()]
    yrs = sorted(x.year.unique())
    prev_cut = 1.01
    for c in cuts:
        for label, g in ((f"上位{round((1 - c) * 100)}% まで (累積)", x[x.q >= c]),):
            if len(g) < 30:
                continue
            lo, hi = boot_ci(g)
            yr = " ".join(f"{g[g.year == y].pay.mean():.0f}" if (g.year == y).sum() >= 8 else "-" for y in yrs)
            md.append(f"| {label} | {len(g):,} | {len(g) / len(yrs):.0f} | {100 * g.top3.mean():.1f}% | {g[g.top3 == 1].pay.mean():.0f} | **{g.pay.mean():.1f}** [{lo:.0f},{hi:.0f}] | {yr} |")
        prev_cut = c
    g = x[x.q < 0.5]; lo, hi = boot_ci(g)
    md += [f"| 下位50% | {len(g):,} | {len(g) / len(yrs):.0f} | {100 * g.top3.mean():.1f}% | {g[g.top3 == 1].pay.mean():.0f} | {g.pay.mean():.1f} [{lo:.0f},{hi:.0f}] | |", ""]
    return md


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    f = frame()
    df = join_master(f)
    df["per"] = df.year.map(period_of)
    miss = df[df.in_master == 0]
    ok = df[df.in_master == 1].copy()
    horse_feats = [c for c in NUM if c in ok.columns and c != "馬番"] + ["ban", "dist_diff", "td_change", "debut"] + CAT
    md = [f"# 堅い 1 番人気の複勝を発走前情報で当て分ける (pinned_floor_search_v2, {pd.Timestamp.now():%Y-%m-%d})", "",
          f"対象: 8 頭以上、発走約 28 分前の 1 番人気で単勝 ≤ {ZONE} 倍。{len(df):,}R (2013-2025)。複勝 1 点、確定払戻で採点。",
          f"master に行が無い馬 (競走中止等) {len(miss)}R ({100 * len(miss) / len(df):.2f}%、3着内率 {100 * miss.top3.mean():.0f}%) は採点から除外。", "",
          "## 0. 何もしない", "", "| 期 | n | 3着内率 | 回収率 |", "|---|---|---|---|"]
    for p in PERIODS:
        g = ok[ok.per == p]; md.append(f"| {p} | {len(g):,} | {100 * g.top3.mean():.1f}% | {g.pay.mean():.1f} |")
    md += ["", "## 1. ウォークフォワード (前年までで学習 → 当年、2017-2025 が全部アウトオブサンプル)", ""]
    sets = {"発走前の市場だけ": MKT_FEATS, "市場 + 馬の特徴": MKT_FEATS + horse_feats}
    for name, feats in sets.items():
        print("fit", name, flush=True)
        ok[f"p_{name}"] = walk_forward(ok, feats, "top3")
        ok[f"r_{name}"] = walk_forward(ok, feats, "pay")
        md += band_table(ok, f"p_{name}", f"3着内率モデル ({name})")
        md += band_table(ok, f"r_{name}", f"払戻回帰モデル ({name})")
    from sklearn.metrics import roc_auc_score
    oos = ok[ok["p_市場 + 馬の特徴"].notna()]
    md += ["## 2. 当て分けの強さ (2017-2025 OOS、3 着内 AUC)", "",
           f"- 発走前の単勝オッズだけ: {roc_auc_score(oos.top3, -oos.tan_pre):.4f}",
           f"- 発走前の市場モデル: {roc_auc_score(oos.top3, oos['p_発走前の市場だけ']):.4f}",
           f"- 市場 + 馬の特徴モデル: {roc_auc_score(oos.top3, oos['p_市場 + 馬の特徴']):.4f}", ""]
    # 特徴の効き (最終年モデル)
    import lightgbm as lgb
    feats = MKT_FEATS + horse_feats
    m = lgb.LGBMClassifier(n_estimators=250, learning_rate=0.02, num_leaves=6, min_child_samples=80, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.6, reg_lambda=20.0, verbose=-1, random_state=0, importance_type="gain").fit(ok[ok.year < 2025][feats], ok[ok.year < 2025].top3)
    imp = pd.Series(m.feature_importances_, index=feats).sort_values(ascending=False)
    md += ["## 3. 効いている特徴 (gain 上位 15、2013-2024 学習)", ""] + [f"- {k}: {100 * v / imp.sum():.1f}%" for k, v in imp.head(15).items()] + [""]
    OUT.write_bytes("\n".join(md).encode("utf-8"))
    ok.drop(columns=[c for c in ok.columns if str(ok[c].dtype) == "category"]).to_parquet(BASE / "data" / "_joint" / "pinned_floor_v2_oos.parquet")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
