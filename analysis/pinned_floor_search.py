# -*- coding: utf-8 -*-
"""
pinned_floor_search.py — 複勝の払戻下限 (100/110 円) で価格が固定される帯に、市場が織り込めない情報が残っているか (2026-11-12)
=========================================================================================================
着想: 単勝 1 倍台前半の馬の複勝は払戻が 110 円 (JRA プラス 10) か 100 円に張り付く。払戻が床に固定されている間は、
その馬の 3 着内率が 88% でも 97% でも払戻は同じ = 市場は「どの堅い馬がより堅いか」を値段に反映できない。
回収率 = 3 着内率 × 払戻 なので、この帯で 3 着内率を当て分けられれば、控除率と無関係に 100% を超えうる。

基盤: data/_joint/mkt_horses.parquet (2013-2025、朝 9 時の単勝・複勝オッズ全頭 + 確定複勝払戻、モデル不要)
      + master_v2 の発走前特徴 (朝 1 番人気の馬について)。
手順: 朝 1 番人気 & 単勝 ≤ ZONE のレース → (1) 単変量走査 (発見 13-18 / 確認 19-22 / 検証 23-25)
      (2) 年次ウォークフォワード (前年までで学習 → 当年を予測) の LightGBM / ロジスティック、予測上位 q% の回収率。
注意: master_v2 は競走中止の馬の行が無い → その馬は特徴欠損 = 学習に入れると「欠損 → 外れ」を覚える (先読み)。
      学習・採点から外し、別枠で『全部買っていた』最悪ケースを併記する。
出力: reports/pinned_floor_search.md (+ docs/research 控え)、予測は data/_joint/pinned_floor_oos.parquet
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parents[1]
MKT = BASE / "data" / "_joint" / "mkt_horses.parquet"
MASTER = BASE / "data" / "master_v2_20130105-20251228.csv"
OUT = BASE / "reports" / "pinned_floor_search.md"
ZONE = 1.9          # 朝 1 番人気の単勝がこれ以下
PERIODS = {"D 13-18": (2013, 2018), "C 19-22": (2019, 2022), "T 23-25": (2023, 2025)}

NUM = ["枠番", "馬番", "距離", "年齢", "斤量", "馬齢斤量差", "間隔", "休み明け～戦目", "前走確定着順", "前走上り3F順", "前4角", "前走出走頭数",
       "前距離", "前走馬体重", "前走馬体重増減", "前PCI", "騎手年齢", "jockey_fuku30", "jockey_fuku90", "trainer_fuku30", "trainer_fuku90",
       "horse_fuku10", "horse_fuku30", "prev_pos_rel", "closing_power", "kako5_avg_pos", "kako5_std_pos", "kako5_best_pos", "kako5_avg_ninki",
       "kako5_pos_vs_ninki", "kako5_avg_agari3f", "kako5_same_td_ratio", "kako5_same_dist_ratio", "kako5_same_place_ratio", "kako5_pos_trend",
       "kako5_race_count", "hist_same_cond_best_pos", "hist_same_cond_top3_rate", "hist_same_cond_count", "prev_hosei", "course_top3_rate",
       "jockey_win_rate", "jockey_top3_rate", "jockey_n_prev", "Ｒ"]
CAT = ["場所", "芝・ダ", "馬場状態", "天気", "クラス名", "前芝・ダ", "年齢限定", "性別限定", "重量種別", "性別", "ブリンカー", "父タイプ名"]


def market_frame() -> pd.DataFrame:
    d = pd.read_parquet(MKT)
    d = d[d.n >= 8].copy()
    d["pay"] = d.fpay.fillna(0.0)
    d = d.sort_values(["rid", "o_win9", "ban"])
    d["rk"] = d.groupby("rid").cumcount() + 1
    g = d.groupby("rid")
    piv = d[d.rk <= 5].pivot(index="rid", columns="rk", values="o_win9"); piv.columns = [f"o{c}" for c in piv.columns]
    plo = d[d.rk <= 4].pivot(index="rid", columns="rk", values="fuku_lo9"); plo.columns = [f"flo{c}" for c in plo.columns]
    agg = pd.DataFrame({"n_lt10": g.o_win9.apply(lambda s: int((s < 10).sum())), "n_lt20": g.o_win9.apply(lambda s: int((s < 20).sum())),
                        "hhi": g.pi.apply(lambda s: float((s ** 2).sum())), "n_flo_le2": g.fuku_lo9.apply(lambda s: int((s <= 2.0).sum()))})
    f = d[d.rk == 1].set_index("rid")[["year", "n", "ban", "pi", "o_win9", "fuku_lo9", "fuku_hi9", "win_overround", "top3", "pay"]]
    f = f.join(piv).join(plo).join(agg).reset_index()
    f["f_spread"] = f.fuku_hi9 - f.fuku_lo9
    f["o2_ratio"] = f.o2 / f.o_win9
    f["pi_top3"] = d[d.rk <= 3].groupby("rid").pi.sum().reindex(f.rid).values
    f["pi_23"] = f.pi_top3 - f.pi
    f["date"] = f.rid.str[:8]; f["month"] = f.rid.str[4:6].astype(int)
    return f[f.o_win9 <= ZONE].copy()


def join_master(f: pd.DataFrame) -> pd.DataFrame:
    cols = ["レースID(新/馬番無)"] + NUM + CAT
    m = pd.read_csv(MASTER, encoding="utf-8-sig", usecols=lambda c: c in set(cols), low_memory=False)
    m["rid"] = m["レースID(新/馬番無)"].astype(str).str[:16]
    m = m.rename(columns={"馬番": "ban"})
    m["ban"] = pd.to_numeric(m["ban"], errors="coerce")
    m["_in_master"] = 1
    out = f.merge(m.drop(columns=["レースID(新/馬番無)"]), on=["rid", "ban"], how="left")
    out["in_master"] = out["_in_master"].fillna(0).astype(int)
    for c in NUM:
        if c != "馬番" and c in out:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out["dist_diff"] = out["距離"] - out["前距離"]
    out["td_change"] = (out["芝・ダ"].astype(str) != out["前芝・ダ"].astype(str)).astype(float).where(out["前芝・ダ"].notna())
    out["debut"] = out["前走確定着順"].isna().astype(int)
    for c in CAT:
        out[c] = out[c].astype(str).str.strip().astype("category")
    return out


def period_of(y: int) -> str:
    for k, (a, b) in PERIODS.items():
        if a <= y <= b:
            return k
    return "?"


def boot_ci(df: pd.DataFrame, n_boot=2000, seed=0):
    g = df.groupby("date").pay.agg(["sum", "size"]).values
    if len(g) < 5:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(g), (n_boot, len(g)))
    v = g[i, 0].sum(1) / g[i, 1].sum(1)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def univariate(df: pd.DataFrame, feats: list[str], min_n=60) -> list[tuple]:
    """各特徴を 4 分位 (カテゴリは値) で切り、3 期すべてで全体平均を上回る/下回る bin を拾う。"""
    rows = []
    base = {p: df[df.per == p].pay.mean() for p in PERIODS}
    for c in feats:
        s = df[c]
        if str(s.dtype) == "category" or s.nunique() <= 8:
            b = s.astype(str)
        else:
            try:
                edges = np.unique(np.nanquantile(df[df.per == "D 13-18"][c].astype(float), [0, .25, .5, .75, 1]))
                if len(edges) < 3:
                    continue
                edges[0], edges[-1] = -np.inf, np.inf
                b = pd.cut(s.astype(float), edges).astype(str)
            except Exception:  # noqa: BLE001
                continue
        for val, g in df.groupby(b):
            if val in ("nan", "None"):
                val = "(欠損)"
            st = {p: (len(x), x.top3.mean(), x.pay.mean()) for p, x in ((p, g[g.per == p]) for p in PERIODS)}
            if any(st[p][0] < min_n for p in PERIODS):
                continue
            diffs = [st[p][2] - base[p] for p in PERIODS]
            if all(x > 0 for x in diffs) or all(x < 0 for x in diffs):
                rows.append((min(abs(x) for x in diffs) * np.sign(diffs[0]), c, val, st))
    return sorted(rows, key=lambda r: -r[0])


def walk_forward(df: pd.DataFrame, feats: list[str], target: str, first_year=2017) -> pd.Series:
    import lightgbm as lgb
    pred = pd.Series(np.nan, index=df.index)
    for y in range(first_year, 2026):
        tr = df[df.year < y]; te = df[df.year == y]
        if len(te) == 0:
            continue
        if target == "top3":
            m = lgb.LGBMClassifier(n_estimators=250, learning_rate=0.02, num_leaves=6, min_child_samples=80, subsample=0.8, subsample_freq=1,
                                   colsample_bytree=0.6, reg_lambda=20.0, verbose=-1, random_state=0)
            m.fit(tr[feats], tr.top3); pred.loc[te.index] = m.predict_proba(te[feats])[:, 1]
        else:
            m = lgb.LGBMRegressor(n_estimators=250, learning_rate=0.02, num_leaves=6, min_child_samples=80, subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.6, reg_lambda=20.0, verbose=-1, random_state=0)
            m.fit(tr[feats], tr.pay / 100.0); pred.loc[te.index] = m.predict(te[feats])
    return pred


def quantile_table(df: pd.DataFrame, col: str, title: str) -> list[str]:
    """年内順位 (当年の予測分布での分位) で切る = 閾値を未来から借りない近似として、前年までの予測分位で切る。"""
    md = [f"### {title}", "", "| 予測の帯 (各年、前年までの分布で決めた閾値) | n | 3着内率 | 平均払戻 | 回収率 [95% CI] | 年別回収率 (17→25) |", "|---|---|---|---|---|---|"]
    d = df[df[col].notna()].copy()
    d["band"] = None
    for y in sorted(d.year.unique()):
        prev = d[d.year < y][col]
        if len(prev) < 300:
            continue
        q = np.quantile(prev, [0.5, 0.7, 0.8, 0.9])
        v = d.loc[d.year == y, col]
        d.loc[d.year == y, "band"] = np.select([v >= q[3], v >= q[2], v >= q[1], v >= q[0]], ["上位10%", "上位10-20%", "上位20-30%", "上位30-50%"], "下位50%")
    d = d[d.band.notna()]
    for b in ["上位10%", "上位10-20%", "上位20-30%", "上位30-50%", "下位50%"]:
        g = d[d.band == b]
        if not len(g):
            continue
        lo, hi = boot_ci(g)
        yr = " ".join(f"{g[g.year == y].pay.mean():.0f}" if (g.year == y).sum() >= 10 else "-" for y in sorted(d.year.unique()))
        hitpay = g[g.top3 == 1].pay.mean()
        md.append(f"| {b} | {len(g):,} | {100 * g.top3.mean():.1f}% | {hitpay:.0f} | **{g.pay.mean():.1f}** [{lo:.0f},{hi:.0f}] | {yr} |")
    g = d[d.band.isin(["上位10%", "上位10-20%"])]
    lo, hi = boot_ci(g)
    md += [f"| (上位20% 計) | {len(g):,} | {100 * g.top3.mean():.1f}% | {g[g.top3 == 1].pay.mean():.0f} | **{g.pay.mean():.1f}** [{lo:.0f},{hi:.0f}] | |", ""]
    return md


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    print("[1/4] market ...", flush=True)
    f = market_frame()
    print(f"   zone n={len(f):,}", flush=True)
    print("[2/4] master join ...", flush=True)
    df = join_master(f)
    df["per"] = df.year.map(period_of)
    miss = df[df.in_master == 0]
    print(f"   master に行が無い (競走中止等): {len(miss)} ({100 * len(miss) / len(df):.2f}%)  その 3着内率 {miss.top3.mean():.3f}", flush=True)
    ok = df[df.in_master == 1].copy()

    mkt_feats = ["o_win9", "pi", "fuku_lo9", "fuku_hi9", "f_spread", "o2", "o3", "o4", "o5", "o2_ratio", "flo2", "flo3", "flo4", "n", "n_lt10", "n_lt20",
                 "n_flo_le2", "hhi", "pi_top3", "pi_23", "win_overround", "month"]
    horse_feats = [c for c in NUM if c in ok.columns and c != "馬番"] + ["ban", "dist_diff", "td_change", "debut"] + CAT
    all_feats = mkt_feats + horse_feats

    md = [f"# 複勝の払戻下限に張り付く帯の探索 (pinned_floor_search, {pd.Timestamp.now():%Y-%m-%d})", "",
          f"対象: 8 頭以上、朝 9 時の 1 番人気で単勝 ≤ {ZONE} 倍。{len(df):,}R (2013-2025)。買い目 = その馬の複勝 1 点 (確定払戻で採点)。",
          f"master に行が無い馬 (競走中止等) {len(miss)}R ({100 * len(miss) / len(df):.2f}%) は特徴が取れないので学習・採点から除外 (最悪ケースは末尾)。", ""]
    md += ["## 0. 何もしないときの回収率", "", "| 期 | n | 3着内率 | 回収率 |", "|---|---|---|---|"]
    for p in PERIODS:
        g = ok[ok.per == p]; md.append(f"| {p} | {len(g):,} | {100 * g.top3.mean():.1f}% | {g.pay.mean():.1f} |")
    md.append("")

    print("[3/4] 単変量 ...", flush=True)
    rows = univariate(ok, all_feats)
    md += ["## 1. 単変量: 3 期すべてで全体より上 / 下だった条件 (各期 60R 以上)", "", "| 特徴 | 条件 | n D/C/T | 3着内率 D/C/T | 回収率 D/C/T | 最小の差 |", "|---|---|---|---|---|---|"]
    for sc, c, val, st in rows[:18] + rows[-18:]:
        md.append(f"| {c} | {val} | {'/'.join(str(st[p][0]) for p in PERIODS)} | {'/'.join(f'{100 * st[p][1]:.0f}' for p in PERIODS)} | "
                  f"{'/'.join(f'{st[p][2]:.1f}' for p in PERIODS)} | {sc:+.1f} |")
    md.append("")

    print("[4/4] walk-forward ...", flush=True)
    for name, feats in (("市場構造だけ", mkt_feats), ("市場 + 馬の特徴", all_feats)):
        ok[f"p_{name}"] = walk_forward(ok, feats, "top3")
        ok[f"r_{name}"] = walk_forward(ok, feats, "pay")
    md += ["## 2. ウォークフォワード (前年までで学習 → 当年を予測、2017-2025 の 9 年が全部アウトオブサンプル)", ""]
    for name in ("市場構造だけ", "市場 + 馬の特徴"):
        md += quantile_table(ok, f"p_{name}", f"3着内率モデル ({name}) の予測順")
        md += quantile_table(ok, f"r_{name}", f"払戻回帰モデル ({name}) の予測順")
    from sklearn.metrics import roc_auc_score
    oos = ok[ok["p_市場 + 馬の特徴"].notna()]
    md += ["## 3. 当て分けの強さ (2017-2025 OOS、3 着内 AUC)", "",
           f"- 朝の単勝オッズだけ: {roc_auc_score(oos.top3, -oos.o_win9):.4f}",
           f"- 市場構造モデル: {roc_auc_score(oos.top3, oos['p_市場構造だけ']):.4f}",
           f"- 市場 + 馬の特徴モデル: {roc_auc_score(oos.top3, oos['p_市場 + 馬の特徴']):.4f}", ""]
    md += ["## 4. 競走中止等の最悪ケース", "",
           f"- 対象 {len(miss)}R はすべて外れ扱い。仮に上位帯に同じ比率 ({100 * len(miss) / len(df):.2f}%) で混ざっていたら、回収率はその分 (約 {100 * len(miss) / len(df):.1f}pt × 平均払戻/100) 下がる。", ""]
    OUT.write_bytes("\n".join(md).encode("utf-8"))
    ok.drop(columns=[c for c in ok.columns if str(ok[c].dtype) == "category"]).to_parquet(BASE / "data" / "_joint" / "pinned_floor_oos.parquet")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
