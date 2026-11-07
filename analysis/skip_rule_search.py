# -*- coding: utf-8 -*-
"""
skip_rule_search.py — 見送り判定の徹底検討 (2026-11-08)
================================================================================
「どのレースを買わないか」を、発走前に分かるレース属性で総当たりし、
  発見期 2023-24 (offline, v6 生スコア)  →  確認期 2025 (offline)  →  前向き 2026 (実配信 bundle)
の 3 段で再現するものだけを残す。対象の買い方は「AI1位 (=◎) の複勝 100 円均等」を主、単勝を副。
offline の◎は生スコア最大 (同率なし)、オッズは確定単勝オッズ (T-10 の代理)。2026 は実配信の◎と T-10 相当の確定オッズ。

ルール候補 (単独・2 つ組):
  ◎オッズ上限 / ◎=1番人気 / 頭数下限 / クラス除外 / レース番号除外 / ◎前走着順 / ◎間隔 / AI の 1-2 位差 (margin) /
  芝ダ / 距離帯 / 月。各ルールは「参戦集合」を定義し、参戦率・◎勝率・3着内率・単勝/複勝 ROI を 3 期で出す。
選抜: 発見期 複勝 ROI 上位 (n>=300) → 確認期 2025 でも全体より +2pt 以上 → 2026 でも +0pt 以上。
出力: reports/skip_rule_search.md (+ docs/research 控え)
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from analysis.market_bias_atlas import load_ext, load_place_pay, load_master, norm_class  # noqa: E402
from analysis.guard_audit_2026 import load as load_2026  # noqa: E402

OUT = BASE / "reports" / "skip_rule_search.md"
SCORES = BASE / "data" / "_ev_grid_scores.parquet"
MIN_N = 300


def build_offline() -> pd.DataFrame:
    df = load_ext().merge(load_place_pay(), on="rid18", how="left").merge(load_master(), on=["rid16", "ban"], how="inner")
    df = df[(df.year >= 2023) & ~df.klass.astype(str).str.startswith("ＪＧ")].copy()
    sc = pd.read_parquet(SCORES)
    sc["rid16"] = sc["レースID(新/馬番無)"].astype(str).str[:16]; sc["ban"] = pd.to_numeric(sc["馬番"], errors="coerce")
    df = df.merge(sc[["rid16", "ban", "_score"]], on=["rid16", "ban"], how="inner")
    df = df.sort_values(["rid16", "ban"])
    # PL 仮定: p ∝ exp(score)
    df["p"] = np.exp(df._score - df.groupby("rid16")._score.transform("max"))
    df["p"] = df.p / df.groupby("rid16").p.transform("sum")
    df["win_ret"] = np.where(df.fin == 1, 100 * df.odds, 0.0); df["plc_ret"] = np.where(df.fin <= 3, df.fpay.fillna(0), 0.0)
    g = df.groupby("rid16")
    top = df.loc[g._score.idxmax()].set_index("rid16")
    fav = df.loc[g.odds.idxmin()].set_index("rid16")
    p2 = g.p.apply(lambda s: s.nlargest(2).iloc[-1] if len(s) > 1 else 0.0)
    out = pd.DataFrame({
        "date": top.date, "year": top.year, "R": top["Ｒ"], "field": top.field, "klass": top.klass.map(norm_class),
        "surface": top.surface, "dist": top.dist, "odds": top.odds, "p": top.p, "p2": p2.reindex(top.index).values,
        "agree": (top.ban == fav.ban), "prev": top["前走確定着順"], "kan": top["間隔"], "sex": top.sex, "age": top.age,
        "fin": top.fin, "tan": top.win_ret, "fuku": top.plc_ret,
    })
    out["period"] = np.where(out.year <= 2024, "A", "B")
    return out.reset_index()


def build_2026() -> pd.DataFrame:
    d = load_2026()
    # p2: bundle に無いので p から近似できない → margin は 2026 では欠損扱い
    out = pd.DataFrame({"rid16": d.rid, "date": d.date, "year": 2026, "R": d.R, "field": d.field, "klass": d.klass,
                        "surface": np.nan, "dist": np.nan, "odds": d.odds, "p": d.p, "p2": np.nan, "agree": d.agree,
                        "prev": pd.to_numeric(d.prev, errors="coerce"), "kan": np.nan, "sex": np.nan, "age": np.nan,
                        "fin": d.fin, "tan": d.tan, "fuku": d.fuku, "chaos_pct": d.pct})
    out["period"] = "C"
    return out


def rules(df: pd.DataFrame) -> dict[str, pd.Series]:
    r: dict[str, pd.Series] = {}
    for x in (1.5, 1.9, 2.4, 2.9, 3.4, 3.9, 4.9):
        r[f"◎オッズ≤{x}"] = df.odds <= x
    r["◎=1番人気"] = df.agree == True  # noqa: E712
    for f in (8, 10, 12, 14):
        r[f"頭数≥{f}"] = df.field >= f
    for f in (12, 14):
        r[f"頭数≤{f}"] = df.field <= f
    r["新馬を除く"] = df.klass != "新馬"
    r["新馬・未勝利を除く"] = ~df.klass.isin(["新馬", "未勝利"])
    r["重賞を除く"] = ~df.klass.isin(["G1", "G2", "G3"])
    r["OP以上を除く"] = ~df.klass.isin(["G1", "G2", "G3", "OP"])
    r["未勝利・1勝のみ"] = df.klass.isin(["未勝利", "1勝"])
    r["11Rを除く"] = df.R != 11
    r["11-12Rを除く"] = ~df.R.isin([11, 12])
    r["1-8Rのみ"] = df.R <= 8
    r["◎前走10着+を除く"] = ~(df.prev >= 10)
    r["◎前走6着+を除く"] = ~(df.prev >= 6)
    r["◎前走1-3着のみ"] = df.prev.between(1, 3)
    r["◎間隔5-9週のみ"] = df.kan.between(5, 9)
    r["◎間隔≤9週"] = df.kan <= 9
    r["◎間隔≥3週"] = df.kan >= 3
    m = df.p - df.p2
    for q in (0.05, 0.10, 0.15):
        r[f"AI1-2位差≥{q}"] = m >= q
    r["芝のみ"] = df.surface == "芝"
    r["ダのみ"] = df.surface == "ダ"
    r["距離≤1400"] = df.dist <= 1400
    r["距離1401-2000"] = df.dist.between(1401, 2000)
    r["4-9月のみ"] = df.date.str[4:6].astype(int).between(4, 9)
    return {k: v.fillna(False).astype(bool) for k, v in r.items()}


def ev(df: pd.DataFrame, m: pd.Series) -> dict:
    s = df[m]
    if len(s) == 0:
        return dict(n=0)
    return dict(n=int(len(s)), share=len(s) / len(df), win=100 * (s.fin == 1).mean(), top3=100 * (s.fin <= 3).mean(),
                tan=float(s.tan.mean()), fuku=float(s.fuku.mean()))


def boot_ci(s: pd.DataFrame, col: str, n_boot=1500, rng=np.random.default_rng(0)):
    g = s.groupby("date").agg(n=(col, "size"), v=(col, "sum"))
    if len(g) < 2:
        return (np.nan, np.nan)
    idx = rng.integers(0, len(g), (n_boot, len(g)))
    v = g.v.values[idx].sum(1) / g.n.values[idx].sum(1)
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    print("[1/3] offline 2023-25 ...", flush=True)
    off = build_offline()
    print(f"   {len(off):,} R", flush=True)
    print("[2/3] 2026 実配信 ...", flush=True)
    srv = build_2026()
    print(f"   {len(srv):,} R", flush=True)
    per = {"A": off[off.period == "A"], "B": off[off.period == "B"], "C": srv}
    base = {k: ev(v, pd.Series(True, index=v.index)) for k, v in per.items()}
    R = {k: rules(v) for k, v in per.items()}
    names = list(R["A"].keys())
    # 単独 + 2 つ組
    combos = [(n,) for n in names] + list(itertools.combinations(names, 2))
    rows = []
    for c in combos:
        rec = {"rule": " ∧ ".join(c)}
        ok = True
        for k in ("A", "B", "C"):
            m = np.logical_and.reduce([R[k][n].values for n in c])
            m = pd.Series(m, index=per[k].index)
            if k != "C" and m.sum() < MIN_N:
                ok = False; break
            if k == "C" and m.sum() < 40:
                ok = False; break
            e = ev(per[k], m)
            for kk, vv in e.items():
                rec[f"{kk}_{k}"] = vv
        if ok:
            rows.append(rec)
    df = pd.DataFrame(rows)
    for k in ("A", "B", "C"):
        df[f"dfuku_{k}"] = df[f"fuku_{k}"] - base[k]["fuku"]; df[f"dtan_{k}"] = df[f"tan_{k}"] - base[k]["tan"]
    # 選抜: A 上位 → B +2pt → C +0pt
    cand = df.sort_values("fuku_A", ascending=False)
    keep = cand[(cand.dfuku_A >= 3) & (cand.dfuku_B >= 2) & (cand.dfuku_C >= 0)]
    print(f"[3/3] {len(df)} 候補 → 3 期再現 {len(keep)}", flush=True)

    md = [f"# 見送り判定の総当たり (skip_rule_search, {pd.Timestamp.now():%Y-%m-%d})", "",
          "AI1位 (=◎) の 100 円均等買い。発見 A=2023-24 (offline) / 確認 B=2025 (offline) / 前向き C=2026 実配信。",
          "参戦集合 = ルールを満たすレース。Δ = 全レース均等買いからの差 (pt)。単独 + 2 つ組の総当たり、A で n≥300、C で n≥40。",
          "選抜 = A で複勝 Δ≥+3 ∧ B で Δ≥+2 ∧ C で Δ≥0。", "",
          "| 期 | R | ◎勝率 | 3着内 | 単勝均等 | 複勝均等 |", "|---|---|---|---|---|---|"]
    for k, b in base.items():
        md.append(f"| {k} | {b['n']:,} | {b['win']:.1f}% | {b['top3']:.1f}% | {b['tan']:.1f} | {b['fuku']:.1f} |")
    md += ["", f"## 3 期で再現したルール ({len(keep)} 件、2026 の複勝順)", "",
           "| ルール | 参戦率 A/B/C | 複勝 A → B → C (Δ) | 単勝 A → B → C | ◎勝率 C | 3着内 C | C 複勝 CI |", "|---|---|---|---|---|---|---|"]
    for _, x in keep.sort_values("fuku_C", ascending=False).head(40).iterrows():
        mC = np.logical_and.reduce([R["C"][n].values for n in x.rule.split(" ∧ ")])
        lo, hi = boot_ci(srv[pd.Series(mC, index=srv.index)], "fuku")
        md.append(f"| {x.rule} | {100*x.share_A:.0f}/{100*x.share_B:.0f}/{100*x.share_C:.0f}% | {x.fuku_A:.1f} → {x.fuku_B:.1f} → {x.fuku_C:.1f} ({x.dfuku_C:+.1f}) | "
                  f"{x.tan_A:.1f} → {x.tan_B:.1f} → {x.tan_C:.1f} | {x.win_C:.1f}% | {x.top3_C:.1f}% | [{lo:.0f},{hi:.0f}] |")
    md += ["", "## 参戦率と複勝回収率のフロンティア (3 期再現したものの中で、参戦率帯ごとの最良)", "",
           "| 参戦率 (C) | ルール | 複勝 C | 単勝 C | R/週 (2026 換算) |", "|---|---|---|---|---|"]
    kk = keep.copy(); kk["band"] = pd.cut(kk.share_C, [0, .1, .2, .3, .4, .5, .7, 1.0])
    for b, g in kk.groupby("band", observed=True):
        x = g.sort_values("fuku_C", ascending=False).iloc[0]
        md.append(f"| {b} | {x.rule} | {x.fuku_C:.1f} | {x.tan_C:.1f} | {x.n_C/59*2:.1f} |")
    md += ["", "## 単独ルールの一覧 (全部)", "", "| ルール | 参戦率 A/B/C | 複勝 A/B/C | 単勝 A/B/C |", "|---|---|---|---|"]
    for _, x in df[~df.rule.str.contains("∧")].sort_values("fuku_C", ascending=False).iterrows():
        md.append(f"| {x.rule} | {100*x.share_A:.0f}/{100*x.share_B:.0f}/{100*x.share_C:.0f}% | {x.fuku_A:.1f}/{x.fuku_B:.1f}/{x.fuku_C:.1f} | {x.tan_A:.1f}/{x.tan_B:.1f}/{x.tan_C:.1f} |")
    OUT.write_bytes("\n".join(md).encode("utf-8"))
    (BASE / "docs" / "research" / "SKIP_RULE_SEARCH_20261108.md").write_bytes("\n".join(md).encode("utf-8"))
    print(f"→ {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
