# -*- coding: utf-8 -*-
"""
postcovid_search.py — コロナ以降 (2020-2025) の市場で期待値のある馬券を総当たりで探す (2026-11-13)
=====================================================================================
ユーザー仮説: 2020 年前後で市場の癖が変わった。ならば探索も検証も 2020 年以降だけでやる。
設計 (後付けの当てはめを避けるため 3 段):
  発見 D = 2020-2022 / 確認 C = 2023-2024 / 最終検証 T = 2025 (探索中は見ない)
  馬 1 頭を単位に、条件 (発走前オッズ帯・朝からの動き・AI 順位・人気順位・AI と人気のずれ・票数比・頭数・クラス・芝ダ・R・1 番人気の強さ・朝→発走前の人気変動)
  の 単独 / 2 つ組 / オッズ帯×動き×AI の 3 つ組 でセルを作り、券種 4 種 (単勝・複勝・ワイド 1 番人気-X・馬連 1 番人気-X) の回収率を出す。
  D と C の両方で 100% 以上 (各 n 以上) のセルを「通過」とし、通過セルを 2025 年にそのまま買ったときの回収率で判定する。
AI 順位: 2020-22 は 〜2018 学習の作り直し (analysis/ai_scores_walkforward.py)、2023-25 は本番 v6 の OOS スコア。
出力: reports/postcovid_search.md (+ docs/research 控え)、data/_joint/postcovid_horses.parquet
"""
from __future__ import annotations

import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from analysis.easy_ticket_map import results  # noqa: E402
from analysis.market_bias_atlas import load_master, norm_class  # noqa: E402

OUT = BASE / "reports" / "postcovid_search.md"
D_Y, C_Y, T_Y = (2020, 2022), (2023, 2024), (2025, 2025)
MIN_N = {"tan": 150, "fuku": 150, "wide": 120, "umaren": 150}
TICKETS = {"tan": "単勝 X", "fuku": "複勝 X", "wide": "ワイド 1番人気-X", "umaren": "馬連 1番人気-X"}


def frame() -> pd.DataFrame:
    d = pd.read_parquet(BASE / "data" / "_joint" / "tanpuk_all_drift.parquet")
    d = d[d.year >= 2020].copy()
    d["drift"] = np.log(d.o_pre / d.o_win9)
    g = d.groupby("rid")
    d["rk9"] = g.o_win9.rank(method="first"); d["rkp"] = g.o_pre.rank(method="first")
    fav = d[d.rkp == 1].set_index("rid")
    d["fav_ban"] = d.rid.map(fav.ban); d["fav_pre"] = d.rid.map(fav.o_pre)
    fv = pd.read_parquet(BASE / "data" / "_joint" / "tanpuk_fav.parquet", columns=["rid", "pool_tan_pre", "pool_fuku_pre"])
    d["pool_ratio"] = d.rid.map(dict(zip(fv.rid, fv.pool_fuku_pre / fv.pool_tan_pre)))
    a1 = pd.read_parquet(BASE / "data" / "_ai_scores_wf_2016_2022.parquet")[["rid", "ban", "ai_rk"]]
    sc = pd.read_parquet(BASE / "data" / "_ev_grid_scores.parquet")
    sc["rid"] = sc["レースID(新/馬番無)"].astype(str).str[:16]; sc["ban"] = pd.to_numeric(sc["馬番"], errors="coerce")
    sc["ai_rk"] = sc.groupby("rid")._score.rank(ascending=False, method="first"); sc = sc[sc.rid.str[:4].astype(int) >= 2023]
    d = d.merge(pd.concat([a1, sc[["rid", "ban", "ai_rk"]]]), on=["rid", "ban"], how="left")
    m = load_master()
    race = m.groupby("rid16").agg(klass=("klass", "first")).reset_index().rename(columns={"rid16": "rid"})
    mm = pd.read_csv(BASE / "data" / "master_v2_20130105-20251228.csv", encoding="utf-8-sig", usecols=["レースID(新/馬番無)", "芝・ダ", "距離"], low_memory=False)
    mm["rid"] = mm["レースID(新/馬番無)"].astype(str).str[:16]
    race = race.merge(mm.groupby("rid").agg(surface=("芝・ダ", "first"), dist=("距離", "first")).reset_index(), on="rid", how="left")
    d = d.merge(race, on="rid", how="left")
    K = results()
    fb = d.fav_ban.values; xb = d.ban.values; rids = d.rid.values
    wide = np.full(len(d), np.nan); uma = np.full(len(d), np.nan)
    for i in range(len(d)):
        k = K.get(rids[i])
        if k is None or xb[i] == fb[i] or not np.isfinite(fb[i]):
            continue
        pair = {int(fb[i]), int(xb[i])}
        uma[i] = k["umaren"] if pair == {k["f"], k["s"]} else 0.0
        if k["wide"] is not None:
            wide[i] = k["wide"].get(frozenset(pair), 0.0)
    d["p_tan"] = d.tan_pay; d["p_fuku"] = d.fuku_pay; d["p_wide"] = wide; d["p_umaren"] = uma
    d["valid_res"] = d.rid.isin(K.keys())
    d = d[d.valid_res].copy()
    d["date"] = d.rid.str[:8]
    # ---- 条件 (カテゴリ) ----
    d["A_オッズ"] = pd.cut(d.o_pre, [0, 1.9, 2.9, 4.9, 6.9, 9.9, 19.9, 1e9], labels=["-1.9", "2-2.9", "3-4.9", "5-6.9", "7-9.9", "10-19", "20+"]).astype(str)
    d["B_朝からの動き"] = pd.cut(d.drift, [-9, -0.3, -0.12, 0.12, 0.3, 9], labels=["大きく売れた", "売れた", "横ばい", "売れず", "大きく売れず"]).astype(str)
    d["C_AI順位"] = pd.cut(d.ai_rk, [0, 1, 3, 6, 99], labels=["1位", "2-3位", "4-6位", "7位以下"]).astype(str)
    d["D_人気"] = pd.cut(d.rkp, [0, 1, 2, 3, 5, 99], labels=["1番", "2番", "3番", "4-5番", "6番以下"]).astype(str)
    d["E_AIと人気のずれ"] = pd.cut(d.ai_rk - d.rkp, [-99, -3, -1, 0, 2, 99], labels=["AIが3つ以上上", "AIが1-2上", "同じ", "AIが1-2下", "AIが3つ以上下"]).astype(str)
    q = d.drop_duplicates("rid").pool_ratio.quantile([1 / 3, 2 / 3]).values
    d["F_票数比"] = pd.cut(d.pool_ratio, [-1, q[0], q[1], 99], labels=["複勝薄", "中", "複勝厚"]).astype(str)
    d["G_頭数"] = pd.cut(d.n, [0, 10, 14, 18], labels=["8-10頭", "11-14頭", "15頭+"]).astype(str)
    kl = d.klass.map(norm_class)
    d["H_クラス"] = np.where(kl.isin(["G1", "G2", "G3", "OP"]), "OP重賞", np.where(kl.isin(["新馬", "未勝利", "1勝", "2勝", "3勝"]), kl, "他"))
    d["I_芝ダ"] = d.surface.astype(str)
    R = d.rid.str[14:16].astype(int)
    d["J_R"] = pd.cut(R, [0, 4, 8, 11, 12], labels=["1-4R", "5-8R", "9-11R", "12R"]).astype(str)
    d["K_1番人気の強さ"] = pd.cut(d.fav_pre, [0, 1.9, 2.9, 99], labels=["1.9倍以下", "2-2.9倍", "3倍以上"]).astype(str)
    d["L_人気変動"] = pd.cut(d.rkp - d.rk9, [-99, -1, 0, 1, 99], labels=["朝より人気上昇", "同じ", "1つ下がった", "2つ以上下がった"]).astype(str)
    return d


def roi(x: pd.DataFrame, col: str) -> tuple[int, float]:
    v = x[col].dropna()
    return len(v), float(v.mean()) if len(v) else np.nan


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    d = frame()
    d.to_parquet(BASE / "data" / "_joint" / "postcovid_horses.parquet")
    feats = [c for c in d.columns if len(c) > 2 and c[1] == "_" and c[0].isupper()]
    per = {"D": d[d.year.between(*D_Y)], "C": d[d.year.between(*C_Y)], "T": d[d.year.between(*T_Y)]}
    print(f"horses D={len(per['D']):,} C={len(per['C']):,} T={len(per['T']):,}  feats={len(feats)}", flush=True)
    base = {t: {p: roi(per[p], f"p_{t}")[1] for p in per} for t in TICKETS}

    # セル定義: 単独 / 2 つ組 / A×B×C
    combos = [(f,) for f in feats] + list(combinations(feats, 2)) + [("A_オッズ", "B_朝からの動き", "C_AI順位")]
    cells = []
    for cb in combos:
        gD = per["D"].groupby(list(cb), observed=True); gC = per["C"].groupby(list(cb), observed=True); gT = per["T"].groupby(list(cb), observed=True)
        for t in TICKETS:
            col = f"p_{t}"
            aD = gD[col].agg(["count", "mean"]); aC = gC[col].agg(["count", "mean"]); aT = gT[col].agg(["count", "mean"])
            j = aD.join(aC, lsuffix="_D", rsuffix="_C", how="inner").join(aT.rename(columns={"count": "count_T", "mean": "mean_T"}), how="left")
            for key, r in j.iterrows():
                key = key if isinstance(key, tuple) else (key,)
                if "nan" in key:
                    continue
                cells.append((cb, key, t, int(r.count_D), r.mean_D, int(r.count_C), r.mean_C, int(r.count_T) if np.isfinite(r.count_T) else 0, r.mean_T))
    C = pd.DataFrame(cells, columns=["feats", "vals", "ticket", "nD", "rD", "nC", "rC", "nT", "rT"])
    C["ok_n"] = [(r.nD >= MIN_N[r.ticket]) and (r.nC >= MIN_N[r.ticket] * 2 / 3) for r in C.itertuples()]
    tested = C[C.ok_n]
    surv = tested[(tested.rD >= 100) & (tested.rC >= 100)].copy()
    near = tested[(tested.rD >= 95) & (tested.rC >= 95)].copy()

    def cond(r) -> str:
        return " × ".join(f"{f[2:]}={v}" for f, v in zip(r.feats, r.vals))

    def pooled(sel: pd.DataFrame, pname: str) -> dict:
        """通過セルの買い目を重複なしでまとめて買ったときの券種別回収率。"""
        out = {}
        for t in TICKETS:
            mask = np.zeros(len(per[pname]), bool)
            for r in sel[sel.ticket == t].itertuples():
                m = np.ones(len(per[pname]), bool)
                for f, v in zip(r.feats, r.vals):
                    m &= (per[pname][f].values == v)
                mask |= m
            x = per[pname][mask]; v = x[f"p_{t}"].dropna()
            if len(v) >= 20:
                bd = x.assign(p=x[f"p_{t}"]).dropna(subset=["p"]).groupby("date").p.agg(["sum", "size"]).values
                rng = np.random.default_rng(0); i = rng.integers(0, len(bd), (3000, len(bd))); bs = bd[i, 0].sum(1) / bd[i, 1].sum(1)
                out[t] = (len(v), float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
        return out

    # 偶然で通過する数の目安: D と C の回収率の符号を独立とみなし、D で 100 以上のセルのうち C でも 100 以上になる割合を
    # 『D で 100 未満のセルが C で 100 以上になる割合』と比べる (差が無ければ D の成績に情報なし)
    hiD = tested[tested.rD >= 100]; loD = tested[tested.rD < 100]
    md = [f"# コロナ以降の市場で期待値のある馬券を探す (postcovid_search, {pd.Timestamp.now():%Y-%m-%d})", "",
          f"発見 D = {D_Y[0]}-{D_Y[1]} / 確認 C = {C_Y[0]}-{C_Y[1]} / 最終検証 T = {T_Y[0]} (探索では見ていない)。8 頭以上、同着除外、確定払戻、1 点 100 円。",
          "馬 X を 1 頭選び、単勝 X / 複勝 X / ワイド 1 番人気-X / 馬連 1 番人気-X を買う。条件 12 系統の 単独・2 つ組・(オッズ×動き×AI)。", "",
          "## 0. 何もしないときの回収率", "", "| 券種 | D | C | T |", "|---|---|---|---|"]
    md += [f"| {TICKETS[t]} | {base[t]['D']:.1f} | {base[t]['C']:.1f} | {base[t]['T']:.1f} |" for t in TICKETS]
    md += ["", "## 1. 何通り調べて、いくつ通過したか", "",
           f"- 調べたセル (n 基準を満たすもの): **{len(tested):,}**",
           f"- D で 100% 以上: {len(hiD):,} ({100 * len(hiD) / len(tested):.1f}%)。そのうち C でも 100% 以上 = **通過 {len(surv):,}** ({100 * len(surv) / max(len(hiD), 1):.1f}%)",
           f"- 参考: D で 100% 未満だったセルが C で 100% 以上になった割合 {100 * (loD.rC >= 100).mean():.1f}% (これと上の割合が同じなら、D の成績は C を何も予告していない)",
           f"- D で 100% 以上だったセルの C の回収率の平均 {hiD.rC.mean():.1f}% (全セル平均 {tested.rC.mean():.1f}%)", ""]
    md += ["## 2. 最終検証: 通過セルを 2025 年にそのまま買う", "", "| 対象 | 券種 | 2025 年の点数 | 回収率 [95% CI] |", "|---|---|---|---|"]
    for lab, sel in (("通過 (D・C とも 100% 以上)", surv), ("準通過 (D・C とも 95% 以上)", near)):
        for t, v in pooled(sel, "T").items():
            md.append(f"| {lab} | {TICKETS[t]} | {v[0]:,} | **{v[1]:.1f}** [{v[2]:.0f},{v[3]:.0f}] |")
    sT = surv[surv.nT >= 30]
    md += ["", f"- 通過セル個別の 2025 年回収率: 平均 {sT.rT.mean():.1f}%、中央値 {sT.rT.median():.1f}%、100% 以上 {int((sT.rT >= 100).sum())} / {len(sT)} セル", ""]
    md += ["## 3. 通過セル (D・C とも 100% 以上、最悪期の高い順、上位 40)", "", "| 券種 | 条件 | D (n) | C (n) | T=2025 (n) |", "|---|---|---|---|---|"]
    surv["worst"] = surv[["rD", "rC"]].min(axis=1)
    for r in surv.sort_values("worst", ascending=False).head(40).itertuples():
        md.append(f"| {TICKETS[r.ticket]} | {cond(r)} | {r.rD:.0f} ({r.nD}) | {r.rC:.0f} ({r.nC}) | {r.rT:.0f} ({r.nT}) |")
    surv3 = surv[(surv.rT >= 100) & (surv.nT >= 40)].copy()
    md += ["", f"## 4. 3 期すべて 100% 以上だったセル ({len(surv3)} 個、n の多い順)", "", "| 券種 | 条件 | D (n) | C (n) | T=2025 (n) | 6 年計 |", "|---|---|---|---|---|---|"]
    surv3["tot"] = (surv3.rD * surv3.nD + surv3.rC * surv3.nC + surv3.rT * surv3.nT) / (surv3.nD + surv3.nC + surv3.nT)
    for r in surv3.sort_values("nD", ascending=False).head(30).itertuples():
        md.append(f"| {TICKETS[r.ticket]} | {cond(r)} | {r.rD:.0f} ({r.nD}) | {r.rC:.0f} ({r.nC}) | {r.rT:.0f} ({r.nT}) | {r.tot:.1f} |")
    md.append("")
    OUT.write_bytes("\n".join(md).encode("utf-8"))
    C.assign(feats=C.feats.map("|".join), vals=C.vals.map("|".join)).to_parquet(BASE / "data" / "_joint" / "postcovid_cells.parquet")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
