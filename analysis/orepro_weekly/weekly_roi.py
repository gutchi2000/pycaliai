# -*- coding: utf-8 -*-
"""
weekly_roi.py — netkeiba「俺プロ」向け: 買い方ごとの「週の回収率の分布」(2021-2025, 歴史データ)
=====================================================================================
目的は分布の形 (上側の裾・下側の深さ・100% 超えの確率) で買い方を比べること。控除率超えの検定ではない。

データ: data/_research/bet_decision_ai/substrate.parquet (2015-2025 平地、レース×候補券、payout は 100 円あたり確定払戻、
        odds_decision は T−28 断面)。2021-2025 のみ使う。2026 の前向き台帳は使わない。
週   : 開催日を「その日以前で最も近い土曜」でまとめる (日曜・月曜祝日開催は同じ週)。
金額 : 1R 上限 1 万円、100 円格子。週の回収率 = Σ払戻 / Σ投資。

候補の買い方 (全部ルールベース、事前列挙):
  P1  現行俺プロ: 馬単 top3 + 三連複 top4 均等 (1 点 1,400 円=9,800 円/R)、gap>0.24 除外、週に gap 昇順 6R
  P1b 同じ券、ゲート通過レースをレース ID 順に 6R (oreapro_bets.py の実際の並び = 最初の開催場の 1R から)
  P2  同じ券、ゲート通過の全レース
  P3  ◎ (AI1位) 複勝 1 点 1 万円、全レース
  P4  a95_equal_v1 (帯は T−28 の AI1位 単勝オッズ、全レース、方策 JSON をそのまま読む)
  P5  馬単 top1 1 点、全レース
  P6  三連複 top4 均等、全レース
  P7  三連単 top10 均等、全レース
  P8  単勝 top1 1 点、全レース
  P9  混戦集中: field_chaos_score の週内上位 N R (N=6,12) に P1 の券
  P10 混合: ◎複勝 5,000 + 馬単 top1 5,000、全レース
  P11 馬連 top1 1 点、全レース
  P12 ワイド top1 1 点、全レース
感度: P1 の券を週 5/6/12/24/全R (gap 昇順) で買う。

実行: cd E:\\PyCaLiAI && venv311\\Scripts\\python.exe -m analysis.orepro_weekly.weekly_roi
出力: analysis/orepro_weekly/out/weekly_roi.json, weekly_roi_by_policy.csv, REPORT.md
"""
from __future__ import annotations
import json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
from shadow_a95 import band_of, equal_stakes, load_policy, sellable_kinds, waku_of  # noqa: E402

SUB = BASE / "data/_research/bet_decision_ai/substrate.parquet"
OUT = Path(__file__).resolve().parent / "out"
YEARS = (2021, 2025)
CAP, UNIT = 10_000, 100
GAP_GATE = 0.24
OREPRO_STAKE = 1400
THRESH_UP = (100, 110, 120, 150, 200, 300)
THRESH_DN = (80, 50, 30)

# bet_type 名 (substrate) ↔ a95 の kind 名
A95_BT = dict(tan="tan", fuku="fuku", umaren="umaren", umatan="umatan", wide="wide",
              wakuren="wakuren", sanpuku="sanrenpuku", sanrentan="sanrentan")


# ----------------------------------------------------------------------------- data
def week_key(date_str: str) -> str:
    d = pd.Timestamp(date_str)
    back = (d.weekday() - 5) % 7          # 土=0, 日=1, 月=2, ...
    return (d - pd.Timedelta(days=back)).strftime("%Y%m%d")


def load():
    cols = ["rid16", "year", "date", "bet_type", "cand_rank", "combo", "h1", "h2", "h3", "payout",
            "odds_decision", "n_entries_master", "field_chaos_score", "top1_p", "top2_p", "settle_status"]
    df = pd.read_parquet(SUB, columns=cols)
    df = df[(df.year >= YEARS[0]) & (df.year <= YEARS[1])].copy()
    df["bet_type"] = df.bet_type.astype(str)
    races = (df[(df.bet_type == "tan") & (df.cand_rank == 1)]
             .set_index("rid16")[["year", "date", "odds_decision", "n_entries_master", "field_chaos_score", "top1_p", "top2_p"]]
             .rename(columns={"odds_decision": "o1"}))
    races["gap"] = races.top1_p - races.top2_p
    races["week"] = races.date.map(week_key)
    # AI 上位 3 頭 (p_win 順) = 単勝候補 rank1-3
    a = df[df.bet_type == "tan"].pivot(index="rid16", columns="cand_rank", values="h1")
    races["a1"], races["a2"], races["a3"] = a[1], a[2], a[3]
    # 払戻ルックアップ: (rid, bet_type, cand_rank) と (rid, bet_type, combo)
    by_rank = {k: v for k, v in zip(zip(df.rid16, df.bet_type, df.cand_rank), df.payout)}
    by_combo = {k: v for k, v in zip(zip(df.rid16, df.bet_type, df.combo), df.payout)}
    return races.sort_values(["date", "rid16"]), by_rank, by_combo


# ----------------------------------------------------------------------------- tickets
# 1 レースの買い目 = [(bet_type, key_kind, key, stake)]; key_kind='rank'|'combo'
def tk_rank(bt, ranks, stake_each):
    return [(bt, "rank", r, stake_each) for r in ranks]


def tickets_orepro(_r):
    return tk_rank("umatan", (1, 2, 3), OREPRO_STAKE) + tk_rank("sanrenpuku", (1, 2, 3, 4), OREPRO_STAKE)


def tickets_a95(r, pol):
    n = int(r.n_entries_master)
    a1, a2, a3 = int(r.a1), int(r.a2), int(r.a3)
    band = band_of(float(r.o1), pol)
    kinds = sellable_kinds(list(pol["plan"][band]), n)
    out = []
    for k in kinds:
        bt = A95_BT[k]
        if k in ("tan", "fuku"):
            key = f"{a1:02d}"
        elif k in ("umaren", "wide"):
            key = f"{min(a1, a2):02d}-{max(a1, a2):02d}"
        elif k == "umatan":
            key = f"{a1:02d}>{a2:02d}"
        elif k == "wakuren":
            w1, w2 = waku_of(a1, n), waku_of(a2, n)
            key = f"{min(w1, w2):02d}-{max(w1, w2):02d}"
        elif k == "sanpuku":
            key = "-".join(f"{x:02d}" for x in sorted((a1, a2, a3)))
        else:
            key = f"{a1:02d}>{a2:02d}>{a3:02d}"
        out.append((bt, "combo", key, k))
    return out, kinds, band


def settle_race(tix, rid, by_rank, by_combo):
    """-> (cost, ret) または None (払戻不明の券がある場合)."""
    cost = ret = 0.0
    for bt, kind, key, stake in tix:
        pay = by_rank.get((rid, bt, key)) if kind == "rank" else by_combo.get((rid, bt, key))
        if pay is None or not np.isfinite(pay):
            return None
        cost += stake
        ret += pay / 100.0 * stake
    return cost, ret


# ----------------------------------------------------------------------------- policies
def build_race_rows(races, by_rank, by_combo, pol):
    """全レースについて、各「券セット」の (cost, ret) を前計算する。"""
    rows = []
    a95_drop_waku = 0
    for rid, r in races.iterrows():
        row = dict(rid=rid, year=int(r.year), date=r.date, week=r.week, gap=float(r.gap),
                   chaos=float(r.field_chaos_score), o1=float(r.o1))
        row["orepro"] = settle_race(tickets_orepro(r), rid, by_rank, by_combo)
        row["fuku1"] = settle_race(tk_rank("fuku", (1,), CAP), rid, by_rank, by_combo)
        row["umatan1"] = settle_race(tk_rank("umatan", (1,), CAP), rid, by_rank, by_combo)
        row["tan1"] = settle_race(tk_rank("tan", (1,), CAP), rid, by_rank, by_combo)
        row["umaren1"] = settle_race(tk_rank("umaren", (1,), CAP), rid, by_rank, by_combo)
        row["wide1"] = settle_race(tk_rank("wide", (1,), CAP), rid, by_rank, by_combo)
        row["sanpuku4"] = settle_race(tk_rank("sanrenpuku", (1, 2, 3, 4), 2500), rid, by_rank, by_combo)
        row["sanrentan10"] = settle_race(tk_rank("sanrentan", tuple(range(1, 11)), 1000), rid, by_rank, by_combo)
        row["mix_fuku_umatan"] = settle_race(tk_rank("fuku", (1,), 5000) + tk_rank("umatan", (1,), 5000), rid, by_rank, by_combo)
        # a95: 枠連の AI1-2 枠組が候補 top6 に無い (約 3%) か枠連行が無いレースは枠連を落として残りに均等配分
        if np.isfinite(r.o1) and r.o1 > 0 and int(r.n_entries_master) >= 5:
            tix, kinds, band = tickets_a95(r, pol)
            if any(k == "wakuren" and (rid, bt, key) not in by_combo for bt, _, key, k in tix):
                tix = [t for t in tix if t[3] != "wakuren"]; kinds = [k for k in kinds if k != "wakuren"]
                a95_drop_waku += 1
            st = equal_stakes(kinds, CAP, UNIT)
            tix = [(bt, kd, key, st[k]) for bt, kd, key, k in tix]
            row["a95"] = settle_race(tix, rid, by_rank, by_combo)
            row["a95_band"] = band
        else:
            row["a95"] = None; row["a95_band"] = None
        rows.append(row)
    return pd.DataFrame(rows), a95_drop_waku


def weekly_roi(sel: pd.DataFrame, col: str):
    """sel: 選んだレース行。col の (cost, ret) を週で集計 → Series(week → roi%)、races per week."""
    ok = sel[sel[col].notna()]
    cost = ok[col].map(lambda t: t[0]); ret = ok[col].map(lambda t: t[1])
    g = pd.DataFrame({"week": ok.week, "cost": cost, "ret": ret}).groupby("week").agg(cost=("cost", "sum"), ret=("ret", "sum"), n=("cost", "size"))
    g["roi"] = 100 * g.ret / g.cost
    return g


def pick_week_topn(df, col, n, by, ascending):
    """週ごとに by で並べて上位 n レースを選ぶ (col が決済可能な行のみ)。"""
    ok = df[df[col].notna()].sort_values(["week", by], ascending=[True, ascending])
    return ok.groupby("week").head(n)


def define_policies(df):
    gate = df[df.gap <= GAP_GATE]
    pols = {}
    pols["P1 俺プロ現行 (馬単3+三連複4, gap≤.24, 週6R gap昇順)"] = ("orepro", pick_week_topn(gate, "orepro", 6, "gap", True))
    pols["P1b 同券, gap≤.24, 週6R レースID順 (oreapro_bets の並び)"] = ("orepro", pick_week_topn(gate, "orepro", 6, "rid", True))
    pols["P2 同券, gap≤.24, 全レース"] = ("orepro", gate)
    pols["P3 ◎複勝1点, 全R"] = ("fuku1", df)
    pols["P4 a95_equal_v1 (帯=T-28 ◎単勝オッズ), 全R"] = ("a95", df)
    pols["P5 馬単top1 1点, 全R"] = ("umatan1", df)
    pols["P6 三連複top4均等, 全R"] = ("sanpuku4", df)
    pols["P7 三連単top10均等, 全R"] = ("sanrentan10", df)
    pols["P8 単勝top1 1点, 全R"] = ("tan1", df)
    pols["P9a 混戦上位6R/週 に P1券"] = ("orepro", pick_week_topn(df, "orepro", 6, "chaos", False))
    pols["P9b 混戦上位12R/週 に P1券"] = ("orepro", pick_week_topn(df, "orepro", 12, "chaos", False))
    pols["P10 ◎複勝5千+馬単top1 5千, 全R"] = ("mix_fuku_umatan", df)
    pols["P11 馬連top1 1点, 全R"] = ("umaren1", df)
    pols["P12 ワイドtop1 1点, 全R"] = ("wide1", df)
    return pols


def sensitivity(df):
    gate = df[df.gap <= GAP_GATE]
    out = {}
    for n in (5, 6, 12, 24, None):
        sel = gate if n is None else pick_week_topn(gate, "orepro", n, "gap", True)
        out[f"P1券 週{n if n else '全'}R"] = ("orepro", sel)
    # 参考: ゲート無し全レース
    out["P1券 ゲート無し 全R"] = ("orepro", df)
    return out


# ----------------------------------------------------------------------------- stats
def longest_below(roi: pd.Series, thr=100.0) -> int:
    best = cur = 0
    for v in roi.sort_index().values:
        cur = cur + 1 if v < thr else 0
        best = max(best, cur)
    return best


def describe(g: pd.DataFrame) -> dict:
    r = g.roi
    d = dict(weeks=int(len(r)), races_total=int(g.n.sum()), races_per_week_mean=float(g.n.mean()),
             races_per_week_min=int(g.n.min()) if len(g) else 0,
             weeks_under_5R=int((g.n < 5).sum()),
             pooled_roi=float(100 * g.ret.sum() / g.cost.sum()),
             mean=float(r.mean()), median=float(r.median()), std=float(r.std()),
             p10=float(r.quantile(.10)), p25=float(r.quantile(.25)), p75=float(r.quantile(.75)), p90=float(r.quantile(.90)),
             min=float(r.min()), max=float(r.max()),
             longest_streak_below_100=longest_below(r))
    for t in THRESH_UP:
        d[f"P_gt_{t}"] = float((r > t).mean())
    for t in THRESH_DN:
        d[f"P_lt_{t}"] = float((r < t).mean())
    return d


# ----------------------------------------------------------------------------- 俺プロ段位 pt
# ある週の実例の境界 (回収率 %) → pt。参加者分位 (上位 60%→0pt の下端、以降 10% 刻み) なので週で変わる → ×0.9/1.0/1.1 で感度。
PT_BOUNDS = (56, 63, 71, 81, 93, 115, 163)       # 区切り
PT_VALUES = (-2, -1, 0, 1, 2, 3, 4, 5)            # len(bounds)+1
PT_SCALES = (0.9, 1.0, 1.1)
PT_MIN_RACES, PT_MIN_YEN = 10, 100_000           # 7 段 (6 段以上): 週 10R 以上かつ 10 万円以上、満たさないと強制 −2pt
PT_FORCED = -2


def roi_to_pt(roi: float, scale: float = 1.0) -> int:
    b = [x * scale for x in PT_BOUNDS]
    return PT_VALUES[int(np.searchsorted(b, roi, side="right"))]


def pt_eval(g: pd.DataFrame, scale: float = 1.0) -> dict:
    """g: 週集計 (cost, ret, n, roi)。制約未達の週は強制 −2。"""
    ok = (g.n >= PT_MIN_RACES) & (g.cost >= PT_MIN_YEN)
    pts = pd.Series(np.where(ok, [roi_to_pt(r, scale) for r in g.roi], PT_FORCED), index=g.index).sort_index()
    dist = {str(v): float((pts == v).mean()) for v in PT_VALUES}
    cum52 = pts.rolling(52).sum().dropna()
    return dict(expected_pt=float(pts.mean()), weeks=int(len(pts)), forced_minus2_weeks=int((~ok).sum()), dist=dist,
                cum52_median=float(cum52.median()), cum52_p10=float(cum52.quantile(.10)), cum52_p90=float(cum52.quantile(.90)),
                cum52_min=float(cum52.min()), cum52_windows=int(len(cum52)),
                by_year_sum={y: float(s) for y, s in pts.groupby(pts.index.str[:4]).sum().items()})


def define_pt_policies(df):
    """週 10R 以上・10 万円以上を満たす設定だけ。P1 (9,800 円/R) は 10R=98,000 円で 10 万に届かないので 11R。"""
    gate = df[df.gap <= GAP_GATE]
    pols = {}
    pols["P1' 俺プロ券 gap≤.24, 週11R gap昇順"] = ("orepro", pick_week_topn(gate, "orepro", 11, "gap", True))
    pols["P1b' 俺プロ券 gap≤.24, 週11R レースID順"] = ("orepro", pick_week_topn(gate, "orepro", 11, "rid", True))
    pols["P2 俺プロ券 gap≤.24, 全レース"] = ("orepro", gate)
    pols["P3 ◎複勝1点, 全R"] = ("fuku1", df)
    pols["P4 a95_equal_v1, 全R"] = ("a95", df)
    pols["P5 馬単top1 1点, 全R"] = ("umatan1", df)
    pols["P6 三連複top4均等, 全R"] = ("sanpuku4", df)
    pols["P7 三連単top10均等, 全R"] = ("sanrentan10", df)
    pols["P8 単勝top1 1点, 全R"] = ("tan1", df)
    pols["P9a' 混戦上位11R/週 に 俺プロ券"] = ("orepro", pick_week_topn(df, "orepro", 11, "chaos", False))
    pols["P9b 混戦上位12R/週 に 俺プロ券"] = ("orepro", pick_week_topn(df, "orepro", 12, "chaos", False))
    pols["P10 ◎複勝5千+馬単top1 5千, 全R"] = ("mix_fuku_umatan", df)
    pols["P11 馬連top1 1点, 全R"] = ("umaren1", df)
    pols["P12 ワイドtop1 1点, 全R"] = ("wide1", df)
    # 1 点 1 万円券の少数 R 版 (10R で 10 万円ちょうど) も比較に入れる
    pols["P5' 馬単top1 1点, 週10R レースID順"] = ("umatan1", pick_week_topn(df, "umatan1", 10, "rid", True))
    pols["P3' ◎複勝1点, 週10R レースID順"] = ("fuku1", pick_week_topn(df, "fuku1", 10, "rid", True))
    return pols


def pt_sensitivity(df):
    gate = df[df.gap <= GAP_GATE]
    out = {}
    for n in (11, 15, 24, 36, None):
        sel = gate if n is None else pick_week_topn(gate, "orepro", n, "gap", True)
        out[f"俺プロ券 週{n if n else '全'}R gap昇順"] = ("orepro", sel)
    for n in (11, 15, 24, 36):
        out[f"俺プロ券 週{n}R レースID順"] = ("orepro", pick_week_topn(gate, "orepro", n, "rid", True))
    for n in (10, 15, 24, 36, None):
        sel = df if n is None else pick_week_topn(df, "umatan1", n, "rid", True)
        out[f"馬単top1 週{n if n else '全'}R レースID順"] = ("umatan1", sel)
    return out


def fmt(x, nd=1):
    return f"{x:.{nd}f}"


def pct(x):
    return f"{100 * x:.0f}%"


# ----------------------------------------------------------------------------- main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pol = load_policy()
    races, by_rank, by_combo = load()
    df, a95_drop_waku = build_race_rows(races, by_rank, by_combo, pol)
    n_weeks_all = df.week.nunique()
    unsettle = {c: int(df[c].isna().sum()) for c in ("orepro", "fuku1", "umatan1", "tan1", "umaren1", "wide1", "sanpuku4", "sanrentan10", "mix_fuku_umatan", "a95")}

    policies = define_policies(df)
    sens = sensitivity(df)

    results, by_year, table = {}, {}, {}
    for name, (col, sel) in {**policies, **{"[感度] " + k: v for k, v in sens.items()}}.items():
        g = weekly_roi(sel, col)
        results[name] = describe(g)
        table[name] = g.roi
        by_year[name] = {}
        for y, gy in g.groupby(g.index.str[:4]):
            by_year[name][y] = describe(gy)

    wide = pd.DataFrame(table)
    wide.index.name = "week"
    wide.round(2).to_csv(OUT / "weekly_roi_by_policy.csv", encoding="utf-8-sig")

    # 俺プロ段位 pt 評価
    pt_pols = define_pt_policies(df); pt_sens = pt_sensitivity(df)
    pt_results = {}
    for name, (col, sel) in {**pt_pols, **{"[感度] " + k: v for k, v in pt_sens.items()}}.items():
        g = weekly_roi(sel, col)
        pt_results[name] = {str(s): pt_eval(g, s) for s in PT_SCALES}
        pt_results[name]["roi"] = describe(g)
    best_exp = max(pt_pols, key=lambda k: pt_results[k]["1.0"]["expected_pt"])
    best_p10 = max(pt_pols, key=lambda k: pt_results[k]["1.0"]["cum52_p10"])

    meta = dict(years=list(YEARS), n_races=int(len(df)), n_weeks=int(n_weeks_all), cap_yen=CAP, unit_yen=UNIT, gap_gate=GAP_GATE,
                orepro_stake=OREPRO_STAKE, a95_policy=pol["policy_id"], a95_waku_dropped_races=int(a95_drop_waku),
                unsettleable_races_per_ticketset=unsettle,
                gate_pass_rate=float((df.gap <= GAP_GATE).mean()),
                substrate=str(SUB.relative_to(BASE)), odds_timing="odds_decision = T-28 断面 (発走 25-30 分前)、payout = 確定払戻")
    meta["pt"] = dict(bounds=PT_BOUNDS, values=PT_VALUES, scales=PT_SCALES, min_races=PT_MIN_RACES, min_yen=PT_MIN_YEN,
                      best_expected_pt=best_exp, best_cum52_p10=best_p10)
    json.dump(dict(meta=meta, policies=results, by_year=by_year, pt=pt_results), open(OUT / "weekly_roi.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    write_report(meta, results, by_year, policies, sens, df, pt_results, pt_pols, pt_sens, best_exp, best_p10)
    print(json.dumps(meta, ensure_ascii=False, indent=1))
    print("\n[pt] 期待pt最大:", best_exp, " / 累積52週 p10 最大:", best_p10)
    for k, v in pt_results.items():
        a = v["1.0"]
        print(f"{k:<44} E[pt] x0.9 {v['0.9']['expected_pt']:+.3f} x1.0 {a['expected_pt']:+.3f} x1.1 {v['1.1']['expected_pt']:+.3f} | cum52 med {a['cum52_median']:+.0f} p10 {a['cum52_p10']:+.0f} p90 {a['cum52_p90']:+.0f} | P(-2) {a['dist']['-2']:.2f} P(>=+3) {a['dist']['3']+a['dist']['4']+a['dist']['5']:.2f}")
    for k, v in results.items():
        print(f"{k:<58} med {v['median']:6.1f} p10 {v['p10']:5.1f} p90 {v['p90']:6.1f} P>100 {v['P_gt_100']:.2f} P>150 {v['P_gt_150']:.2f} P<50 {v['P_lt_50']:.2f} pooled {v['pooled_roi']:5.1f} R/w {v['races_per_week_mean']:.1f}")


def write_report(meta, results, by_year, policies, sens, df, pt_results, pt_pols, pt_sens, best_exp, best_p10):
    L = []
    L.append("# 俺プロ向け: 買い方ごとの週回収率の分布 (2021-2025)\n")
    L.append(f"> 生成: `python -m analysis.orepro_weekly.weekly_roi`。データ `{meta['substrate']}` (2021-2025 平地 {meta['n_races']:,}R / {meta['n_weeks']} 週)。"
             f" 1R 上限 ¥{meta['cap_yen']:,}・100 円格子。週 = 土曜起点 (日・月祝を含む)。**控除率を超えるかの検定ではなく、週単位の分布の形の比較**。\n")
    L.append("## 0. 要点\n")
    def R(name):
        return results[name]
    p1, p1b, p2, p3, p5, p9a = (R(k) for k in list(policies)[:3]) if False else (results[list(policies)[0]], results[list(policies)[1]], results[list(policies)[2]], results[list(policies)[3]], results[list(policies)[5]], results[list(policies)[9]])
    L.append(f"- 表 1a「週回収率 ≥ X% の確率」が主結果。**全レース買い**は週の分散が小さい (◎複勝 P3: p10-p90 = {p3['p10']:.0f}-{p3['p90']:.0f}、P>100 {pct(p3['P_gt_100'])}、P<50 {pct(p3['P_lt_50'])})。"
             f"**週 6R 集中** (P1) は上側の裾が厚い (P>150 {pct(p1['P_gt_150'])}、最大 {p1['max']:.0f}%) が、週の半分近くが回収 0 で P<50 {pct(p1['P_lt_50'])}、中央値 {p1['median']:.0f}%。\n"
             f"- 週 6R の選び方で分布が変わる: gap 昇順 (最も混戦の 6R、P1) は pooled {p1['pooled_roi']:.1f}% / 中央値 {p1['median']:.0f}% で、レース ID 順 (P1b) の {p1b['pooled_roi']:.1f}% / {p1b['median']:.0f}% より悪い。"
             f"混戦度上位に集中する P9a はさらに悪い (pooled {p9a['pooled_roi']:.1f}%、P<50 {pct(p9a['P_lt_50'])})。**prob-first の馬単/三連複は混戦レースでは的中が薄く、混戦集中は上側の裾を厚くしない**。\n"
             f"- 100% 超えの確率 P(週>100) で最も高いのは馬単 top1 1 点 (P5: {pct(p5['P_gt_100'])}) と P1/P1b ({pct(p1['P_gt_100'])})。P>150 以上の裾は P1/P1b/P9 (少数 R) だけが 10% 前後、全レース型は 0-7%。\n"
             f"- 俺プロの段位ポイントが週の回収率の閾値・順位で決まるなら、見るべきは pooled ROI ではなく P(週>100) / P(週>150) と下側 (P<50、最長 <100 連続週: P3 は {p3['longest_streak_below_100']} 週連続、P1 は {p1['longest_streak_below_100']} 週)。\n"
             "- 数値の注意: 帯判定 (P4) は T−28 オッズ、払戻は確定。ゲート gap は substrate の p_win (OOF softmax 2021-23 / 本番 2024-25) で、本番 bundle の尺度と厳密には一致しない。P4 の 2023-25 は帯表の選択期間と重なる in-sample。\n")
    L.append("## 1. 週回収率の分布 (全期間 2021-2025)\n")
    hdr = "| 買い方 | 週数 | R/週 | pooled ROI | 中央値 | p10 | p25 | p75 | p90 | 最小 | 最大 | P>100 | P>120 | P>150 | P>200 | P<50 | 最長<100 連続週 |"
    L.append(hdr); L.append("|" + "---|" * (hdr.count("|") - 1))
    for name in policies:
        v = results[name]
        L.append(f"| {name} | {v['weeks']} | {v['races_per_week_mean']:.1f} | {fmt(v['pooled_roi'])} | {fmt(v['median'])} | {fmt(v['p10'])} | {fmt(v['p25'])} | {fmt(v['p75'])} | {fmt(v['p90'])} | {fmt(v['min'])} | {fmt(v['max'],0)} | "
                 f"{pct(v['P_gt_100'])} | {pct(v['P_gt_120'])} | {pct(v['P_gt_150'])} | {pct(v['P_gt_200'])} | {pct(v['P_lt_50'])} | {v['longest_streak_below_100']} |")
    L.append("\n### 1a. 「週回収率 ≥ X%」の確率 (主表)\n")
    hdr = "| 買い方 | ≥80 | ≥100 | ≥110 | ≥120 | ≥150 | ≥200 | ≥300 | <50 | <30 |"
    L.append(hdr); L.append("|" + "---|" * (hdr.count("|") - 1))
    for name in policies:
        v = results[name]
        L.append(f"| {name} | {pct(1 - v['P_lt_80'])} | {pct(v['P_gt_100'])} | {pct(v['P_gt_110'])} | {pct(v['P_gt_120'])} | {pct(v['P_gt_150'])} | {pct(v['P_gt_200'])} | {pct(v['P_gt_300'])} | {pct(v['P_lt_50'])} | {pct(v['P_lt_30'])} |")
    L.append("\n## 2. 年別 (中央値 / P>100 / P>150 / P<50)\n")
    years = sorted({y for d in by_year.values() for y in d})
    hdr = "| 買い方 | " + " | ".join(years) + " |"
    L.append(hdr); L.append("|" + "---|" * (len(years) + 1))
    for name in policies:
        cells = []
        for y in years:
            v = by_year[name].get(y)
            cells.append(f"{fmt(v['median'],0)} / {pct(v['P_gt_100'])} / {pct(v['P_gt_150'])} / {pct(v['P_lt_50'])}" if v else "—")
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    L.append("\n## 3. 週に買うレース数の感度 (P1 の券: 馬単3+三連複4, gap≤0.24, gap 昇順で N R)\n")
    hdr = "| N | 週数 | R/週 | pooled ROI | 中央値 | p10 | p90 | 標準偏差 | P>100 | P>150 | P>200 | P<50 | 最長<100 連続週 |"
    L.append(hdr); L.append("|" + "---|" * (hdr.count("|") - 1))
    for k in sens:
        v = results["[感度] " + k]
        L.append(f"| {k} | {v['weeks']} | {v['races_per_week_mean']:.1f} | {fmt(v['pooled_roi'])} | {fmt(v['median'])} | {fmt(v['p10'])} | {fmt(v['p90'])} | {fmt(v['std'])} | {pct(v['P_gt_100'])} | {pct(v['P_gt_150'])} | {pct(v['P_gt_200'])} | {pct(v['P_lt_50'])} | {v['longest_streak_below_100']} |")
    L.append("\n## 4. 俺プロ 段位 pt 評価 (7 段: 週 10R 以上・10 万円以上、未達は強制 −2pt)\n")
    L.append("境界 (ある週の実例、回収率 %): " + " / ".join(f"{v:+d}pt: {lo}〜{hi}" for v, lo, hi in zip(PT_VALUES, (0,) + PT_BOUNDS, PT_BOUNDS + ("∞",)))
             + "。境界は参加者分位で週ごとに動くので ×0.9 / ×1.0 / ×1.1 の 3 通り。累積 52 週は 260 週の系列上のローリング 52 週窓 (209 窓)。"
             " 俺プロ券 (7 点 × 1,400 = 9,800 円/R) は 10R で 98,000 円と 10 万に届かないため 11R にした。\n")
    L.append(f"**期待 pt が最大**: {best_exp} ({pt_results[best_exp]['1.0']['expected_pt']:+.3f} pt/週)　　"
             f"**累積 52 週 pt の p10 が最大 (降格しにくい)**: {best_p10} (p10 {pt_results[best_p10]['1.0']['cum52_p10']:+.0f} pt)\n")
    hdr = "| 買い方 | R/週 | E[pt] ×0.9 | E[pt] ×1.0 | E[pt] ×1.1 | 52週累積 中央値 | p10 | p90 | 最小 | P(−2) | P(−1) | P(0) | P(+1) | P(+2) | P(+3) | P(+4) | P(+5) |"
    L.append(hdr); L.append("|" + "---|" * (hdr.count("|") - 1))
    for name in pt_pols:
        v = pt_results[name]; a = v["1.0"]; d = a["dist"]
        L.append(f"| {name} | {v['roi']['races_per_week_mean']:.1f} | {v['0.9']['expected_pt']:+.2f} | {a['expected_pt']:+.2f} | {v['1.1']['expected_pt']:+.2f} | "
                 f"{a['cum52_median']:+.0f} | {a['cum52_p10']:+.0f} | {a['cum52_p90']:+.0f} | {a['cum52_min']:+.0f} | "
                 + " | ".join(pct(d[str(k)]) for k in PT_VALUES) + " |")
    L.append("\n### 4a. 年ごとの累積 pt (×1.0)\n")
    years = sorted({y for n in pt_pols for y in pt_results[n]["1.0"]["by_year_sum"]})
    hdr = "| 買い方 | " + " | ".join(years) + " |"
    L.append(hdr); L.append("|" + "---|" * (len(years) + 1))
    for name in pt_pols:
        s_ = pt_results[name]["1.0"]["by_year_sum"]
        L.append(f"| {name} | " + " | ".join(f"{s_.get(y, 0):+.0f}" for y in years) + " |")
    L.append("\n### 4b. 週に買う R 数の感度 (pt 基準)\n")
    hdr = "| 設定 | R/週 | E[pt] ×0.9 | ×1.0 | ×1.1 | 52週累積 中央値 | p10 | p90 | P(−2) | P(≥+3) |"
    L.append(hdr); L.append("|" + "---|" * (hdr.count("|") - 1))
    for k in pt_sens:
        v = pt_results["[感度] " + k]; a = v["1.0"]; d = a["dist"]
        L.append(f"| {k} | {v['roi']['races_per_week_mean']:.1f} | {v['0.9']['expected_pt']:+.2f} | {a['expected_pt']:+.2f} | {v['1.1']['expected_pt']:+.2f} | "
                 f"{a['cum52_median']:+.0f} | {a['cum52_p10']:+.0f} | {a['cum52_p90']:+.0f} | {pct(d['-2'])} | {pct(d['3'] + d['4'] + d['5'])} |")
    L.append("\n### 4c. pt 評価の読み方\n")
    L.append("- 段位 pt は「週回収率の階段関数」なので、分布の中心が高く下側が浅い買い方 (◎複勝・単勝・ワイド 1 点の全レース) が期待 pt でも累積 p10 でも勝つ。"
             "回収率中央値 83% の ◎複勝は、境界 81〜93% が +2pt のため週の大半が +1〜+2 に落ち、−2 がほぼ無い。\n"
             "- 少数 R 集中 (俺プロ券 11R、混戦集中) は +4/+5 の週を作るが、P(−2) が 45〜60% で期待 pt を食われる。境界 ×1.1 (参加者が強い週) ではマイナス圏に入る設定もある。\n"
             "- 境界は週の参加者分位で動くため、この pt 評価は「この週の例の境界が全期間に当てはまる」という仮定の下の値。×0.9/×1.1 で順位がほぼ変わらないことだけを結論に使う。\n")
    L.append("\n## 5. データ上の注意\n")
    L.append(f"- オッズ: `odds_decision` は T−28 断面 (発走 25-30 分前)。a95 の帯判定だけに使い、払戻はすべて確定払戻 (100 円あたり、返還は 100)。本番の a95 は bundle 生成時 (前夜〜未明) や T−10 のオッズで帯を決めるので帯の割付がずれる。\n"
             f"- 確率の尺度: substrate の p_win は 2021-23 が v6 C1 の OOF (softmax τ=0.845)、2024-25 が本番 (τ=0.865)。gap>0.24 ゲートは本番 bundle の p_win で決めた閾値で、ここでは同じ数値をそのまま当てている (ゲート通過率 {meta['gate_pass_rate']:.1%})。\n"
             f"- 払戻不明で除外したレース (券セットごと): {meta['unsettleable_races_per_ticketset']}。同着で払戻が特定できない組などが原因。\n"
             f"- a95 の枠連: AI1-2 位の枠組が候補 top6 に無い/枠連行の無いレース {meta['a95_waku_dropped_races']}R は枠連を落として残り券種に均等配分 (本番は買う)。\n"
             "- 出走取消の返還は複勝・ワイドの 100 円返還以外は未処理 (外れ扱い=保守側)。\n"
             "- 週あたりのレース数は 2021-25 の実開催 (3 場 × 2 日 × 12R ≒ 72R、3 日開催・2 場開催あり)。俺プロの「週 5R 以上・5 万円以上」は、P1/P1b/P9a で週のゲート通過レースが 5R 未満だった週 (表の weeks_under_5R、JSON 参照) を除けば満たす。\n"
             "- in-sample 注意: P1 のゲート gap>0.24 は 2024 で決め 2025 で確認したもので、2021-23 への適用は事後。a95 の帯×券種表は 2023-25 で選んだものなので 2023-25 の P4 は in-sample。\n")
    (Path(__file__).resolve().parent / "REPORT.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
