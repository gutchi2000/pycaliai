# -*- coding: utf-8 -*-
"""
v6base.py — v6 の時点安全なスコアと、v6/市場それぞれの PL 勝率・3着内率
======================================================================
v6 スコアの供給源 (各年にとって学習に使っていないモデルのスコアだけを使う):
  2015-2023 : data/_research/mcond/oof_scores_c1master.parquet  (analysis/mcond/rebuild_oof_c1.py)
              exp_oof_scores.py の手順 (各年 Y を train≤Y-2 / early-stop Y-1 で学習した LambdaRank) を
              P0-5 修正後 (C1) の master で作り直したもの。
              ※ 旧 data/oof_scores_v6params.parquet は P0-5 リーク入り master 由来なので使わない
  2024-2025 : data/_policy/v6_scores.parquet  (本番 unified_rank_v6, train≤2022 / valid2023)
  2023 は両方にあるが、本番 v6 は 2023 を early stop/較正に使っているので OOF 側を使う。

尺度合わせ (温度 τ): スコアは年・モデルごとに尺度が違うので PL の鋭さが揃わない。
  OOF  : 2022 の OOF スコアで τ を最尤推定 (1着の条件付きロジット)
  本番 : 本番 v6 自身の検証年 2023 の本番スコアで τ を推定
  どちらもテスト年 (2024-25) を使わない。

市場: market.py の pi (snap = pre / am9) から同じ PL 式で 3着内率を出す (Harville)。
出力: data/_research/mcond/base.parquet
  rid16, ban, year, date, v6_src, v6_score, v6_pwin, v6_p3, mkt_pi_{pre,am9}, mkt_p3_{pre,am9},
  mkt_odds_pre, rank_v6, rank_mkt_pre, n_field, top3, win, fpay (確定複勝配当, 円/100円), tan_final_odds
実行: python -m analysis.mcond.v6base
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
OUT = BASE / "data" / "_research" / "mcond" / "base.parquet"
KEKKA = Path(r"E:\競馬過去走データ\kekka_20130105-20251228_v2.csv")


def pl_top3(w: np.ndarray) -> np.ndarray:
    """PL (Harville) で各馬の 3着内確率。w>0。"""
    W = w.sum()
    n = len(w)
    p1 = w / W
    t = w / (W - w)
    p2 = w * (t.sum() - t) / W
    p3 = np.zeros(n)
    for j in range(n):
        rj = W - w[j]
        for k in range(n):
            if k == j:
                continue
            rjk = rj - w[k]
            if rjk <= 0:
                continue
            c = (w[j] / W) * (w[k] / rj)
            v = c * w / rjk
            v[j] = 0.0
            v[k] = 0.0
            p3 += v
    return np.clip(p1 + p2 + p3, 1e-9, 1 - 1e-9)


def fit_tau(df: pd.DataFrame) -> float:
    """1着の条件付きロジット尤度を最大にする温度。"""
    groups = [(g["score"].to_numpy(), g["win"].to_numpy()) for _, g in df.groupby("rid16")]
    groups = [(s, w) for s, w in groups if w.sum() == 1]

    def nll(tau):
        tot = 0.0
        for s, w in groups:
            z = s / tau
            z = z - z.max()
            tot -= (z[w == 1][0] - np.log(np.exp(z).sum()))
        return tot / len(groups)
    r = minimize_scalar(nll, bounds=(0.2, 5.0), method="bounded")
    return float(r.x)


def main() -> None:
    oof = pd.read_parquet(BASE / "data/_research/mcond/oof_scores_c1master.parquet")
    oof = oof.rename(columns={"rid": "rid16"})
    oof["rid16"] = oof["rid16"].astype(str)
    oof["v6_src"] = "oof"
    prod = pd.read_parquet(BASE / "data/_policy/v6_scores.parquet").rename(columns={"rid": "rid16"})
    prod["rid16"] = prod["rid16"].astype(str)
    prod["year"] = prod["rid16"].str[:4].astype(int)
    prod["v6_src"] = "prod"

    # 結果 (着順・複勝配当・確定単勝オッズ) — 決済と目的変数のみ。特徴量には使わない
    k = pd.read_csv(KEKKA, encoding="cp932", low_memory=False,
                    usecols=["レースID(新)", "馬番", "確定着順", "単勝配当", "複勝配当"])
    k["rid16"] = k["レースID(新)"].astype(str).str[:16]
    k["ban"] = pd.to_numeric(k["馬番"], errors="coerce")
    k["fin"] = pd.to_numeric(k["確定着順"], errors="coerce")
    k["fpay"] = pd.to_numeric(k["複勝配当"], errors="coerce").fillna(0.0)
    tanstr = k["単勝配当"].astype(str)
    k["tan_final_odds"] = pd.to_numeric(tanstr.str.extract(r"\(([\d.]+)\)")[0], errors="coerce")
    k.loc[k["fin"] == 1, "tan_final_odds"] = pd.to_numeric(tanstr, errors="coerce") / 100.0
    k = k.dropna(subset=["ban"])
    k["ban"] = k["ban"].astype(int)
    res = k[["rid16", "ban", "fin", "fpay", "tan_final_odds"]]

    def attach(df):
        df = df.merge(res, on=["rid16", "ban"], how="inner")
        df = df[df["fin"].notna() & (df["fin"] >= 1)]          # 取消・除外 (着順0) は出走していない
        df["win"] = (df["fin"] == 1).astype(int)
        df["top3"] = (df["fin"] <= 3).astype(int)
        return df

    oof = attach(oof)
    prod = attach(prod)
    tau_oof = fit_tau(oof[oof.year == 2022])
    tau_prod = fit_tau(prod[prod.year == 2023])
    print(f"温度 τ: OOF(2022で推定)={tau_oof:.3f}  本番v6(2023で推定)={tau_prod:.3f}")

    ov = oof[oof.year == 2023][["rid16", "ban", "score"]].merge(
        prod[prod.year == 2023][["rid16", "ban", "score"]], on=["rid16", "ban"], suffixes=("_oof", "_prod"))
    r_in = ov.groupby("rid16").apply(lambda g: g.score_oof.corr(g.score_prod, method="spearman")).median()
    print(f"2023 重複: {len(ov):,} 行  レース内順位相関 (中央値) = {r_in:.3f}")

    oof["tau"] = tau_oof
    prod["tau"] = tau_prod
    v6 = pd.concat([oof[oof.year <= 2023], prod[prod.year >= 2024]], ignore_index=True)

    mkt = pd.read_parquet(BASE / "data/_research/mcond/market.parquet")
    mk = mkt[mkt.snap.isin(["pre", "am9"])].pivot_table(
        index=["rid16", "ban"], columns="snap", values=["pi", "odds"], aggfunc="first")
    mk.columns = [f"mkt_{a}_{b}" for a, b in mk.columns]
    mk = mk.reset_index()
    v6 = v6.merge(mk, on=["rid16", "ban"], how="left")

    out = []
    for rid, g in v6.groupby("rid16", sort=False):
        g = g.copy()
        z = g["score"].to_numpy() / g["tau"].iloc[0]
        w = np.exp(z - z.max())
        g["v6_pwin"] = w / w.sum()
        g["v6_p3"] = pl_top3(w) if len(g) >= 3 else np.nan
        for s in ("pre", "am9"):
            pi = g.get(f"mkt_pi_{s}")
            if pi is not None and pi.notna().all() and len(g) >= 3:
                p = pi.to_numpy() / pi.sum()        # 出走した馬だけで再正規化 (取消分を除く)
                g[f"mkt_pi_{s}"] = p
                g[f"mkt_p3_{s}"] = pl_top3(p)
            else:
                g[f"mkt_p3_{s}"] = np.nan
        g["n_field"] = len(g)
        g["rank_v6"] = (-g["v6_pwin"]).rank(method="first")
        g["rank_mkt_pre"] = (-g["mkt_pi_pre"]).rank(method="first") if g["mkt_pi_pre"].notna().all() else np.nan
        out.append(g)
    b = pd.concat(out, ignore_index=True)
    b["date"] = b["rid16"].str[:8]
    b = b.rename(columns={"score": "v6_score"})
    keep = ["rid16", "ban", "year", "date", "v6_src", "v6_score", "tau", "v6_pwin", "v6_p3",
            "mkt_pi_pre", "mkt_p3_pre", "mkt_odds_pre", "mkt_pi_am9", "mkt_p3_am9",
            "rank_v6", "rank_mkt_pre", "n_field", "fin", "top3", "win", "fpay", "tan_final_odds"]
    b = b[keep]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    b.to_parquet(OUT, index=False)
    print(f"saved {len(b):,} rows / {b.rid16.nunique():,} races -> {OUT}")
    for y, g in b.groupby("year"):
        ok = g["mkt_p3_pre"].notna().mean()
        print(f"  {y} races={g.rid16.nunique():>5}  v6={g.v6_src.iloc[0]}  市場pre被覆={100*ok:5.1f}%  "
              f"top3率={g.top3.mean():.3f}  Σv6_p3/レース={g.groupby('rid16').v6_p3.sum().mean():.2f}")


if __name__ == "__main__":
    main()
