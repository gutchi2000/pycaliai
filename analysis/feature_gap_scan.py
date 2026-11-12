# -*- coding: utf-8 -*-
"""
feature_gap_scan.py — AI が見落としている「角度」を、前走の JRDB 項目ごとに順位づけする (2026-11-13)
=====================================================================================
記事の「特徴量効用値」がやっていること (= 今の特徴量に欠けている角度を探す) を、1 レースずつの再演ではなく統計でやる版。
前走の JRDB 項目 (不利・出遅・位置取り・コース取り・上昇度・馬体・気配・ペース・各指数・自己比の凹み) を帯に切り、
その帯の馬の今走 3 着内率が「同じ AI 順位・頭数の馬の平均」からどれだけずれるか (AI 比 A/E) を測る。
  A/E > 1 = AI が過小評価 (その項目を知っていれば上げるべきだった) / < 1 = 過大評価。
同じ帯について市場比 A/E (同じ確定オッズの馬の平均との比) も並べる。市場比が 1 なら「市場は知っている、AI だけ知らない」。
2 期 (2016-19 / 2020-25) の両方で同じ向きのものだけを拾う。
入力: data/jrdb/sed.parquet、analysis/excusable_flop.py が作る基盤。出力: reports/feature_gap_scan.md (+ docs/research 控え)
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
from analysis.excusable_flop import build  # noqa: E402

OUT = BASE / "reports" / "feature_gap_scan.md"
NUM = {"p_E": "前走 不利+出遅 合計", "p_E_furi": "前走 不利 (前/中/後 含む)", "p_deokure": "前走 出遅", "p_ato_furi": "前走 直線の不利", "p_ichidori": "前走 位置取り補正",
       "p_race_adj": "前走 レース補正", "p_course_dori": "前走 コース取り (1 最内〜5 大外)", "p_joushou": "前走 上昇度", "p_ten_idx": "前走 テン指数",
       "p_agari_idx": "前走 上がり指数", "p_pace_idx": "前走 ペース指数", "p_idm": "前走 IDM", "dip": "前走 IDM − 直前 3 走平均", "p_idm_minus_soten": "前走 IDM − 素点 (補正の総量)",
       "p_c4_rel": "前走 4 角位置 (頭数比)", "p_bw_diff": "前走 馬体重 − 今走までの平均", "gap": "間隔 (日)"}
CAT = {"p_batai_cd": "前走 馬体コード", "p_kehai_cd": "前走 気配コード", "p_uma_pace": "前走 馬のペース (H/M/S)", "p_race_pace": "前走 レースペース"}


def ae(v: pd.DataFrame, act: str, exp: str):
    v = v.dropna(subset=[exp])
    if len(v) < 300:
        return None
    e = v[exp].sum(); r = v[act].sum() / e; se = np.sqrt((v[exp] * (1 - v[exp])).sum()) / e
    return r, r - 1.96 * se, r + 1.96 * se, len(v)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    s = build()
    sed = pd.read_parquet(BASE / "data" / "jrdb" / "sed.parquet")
    sed["ped"] = sed.ped.astype(str).str.zfill(8)
    sed = sed.sort_values(["ped", "ymd"])
    g = sed.groupby("ped", sort=False)
    extra = pd.DataFrame({"ped": sed.ped, "ymd": sed.ymd})
    for c in ("ichidori", "race_adj", "course_dori", "joushou", "ten_idx", "agari_idx", "pace_idx", "soten", "c4", "tosu", "bw", "batai_cd", "kehai_cd", "uma_pace", "race_pace"):
        extra["p_" + c] = g[c].shift(1)
    extra["bw_mean_prev"] = g.bw.transform(lambda x: x.shift(1).expanding().mean())
    s = s.merge(extra, on=["ped", "ymd"], how="left", suffixes=("", "_x"))
    s["p_idm_minus_soten"] = s.p_idm - s.p_soten
    s["p_c4_rel"] = s.p_c4 / s.p_tosu_x if "p_tosu_x" in s else s.p_c4 / s.p_tosu
    s["p_bw_diff"] = s.p_bw - s.bw_mean_prev
    for c in ("p_ichidori", "p_race_adj"):
        s[c] = s[c].fillna(0)
    s = s[s.ai_rk.notna() & (s.year >= 2016)].copy()
    eras = {"2016-19": s.year <= 2019, "2020-25": s.year >= 2020}
    rows = []

    def add(label: str, val: str, m: pd.Series):
        r = {}
        for e, em in eras.items():
            x = s[m & em]
            r[e] = (ae(x, "top3", "exp_top3_ai"), ae(x, "top3", "exp_top3_mkt"), ae(x, "win", "exp_win"))
        if any(r[e][0] is None for e in eras):
            return
        a = [r[e][0][0] for e in eras]
        if (a[0] - 1) * (a[1] - 1) <= 0:
            return
        sig = all((r[e][0][1] > 1) or (r[e][0][2] < 1) for e in eras)
        rows.append(dict(label=label, val=val, score=min(abs(np.log(x)) for x in a) * np.sign(a[0] - 1), sig=sig, r=r))

    for c, lab in NUM.items():
        v = s[c]
        if v.nunique() <= 8:
            for val in sorted(v.dropna().unique()):
                add(lab, f"= {val:g}", v == val)
        else:
            qs = np.unique(np.nanquantile(v, [0, .1, .3, .7, .9, 1]))
            for lo, hi in zip(qs[:-1], qs[1:]):
                add(lab, f"{lo:.1f}〜{hi:.1f}", (v >= lo) & (v <= hi))
    for c, lab in CAT.items():
        for val in s[c].dropna().astype(str).unique():
            add(lab, f"= {val}", s[c].astype(str) == val)

    def fmt(t):
        return "-" if t is None else f"{t[0]:.2f} [{t[1]:.2f},{t[2]:.2f}]"

    def tab(sel, title):
        md = [f"### {title}", "", "| 前走の項目 | 帯 | n (16-19 / 20-25) | AI 比 3着内 16-19 | AI 比 3着内 20-25 | 市場比 3着内 20-25 | 市場比 勝ち 20-25 | 両期で有意 |", "|---|---|---|---|---|---|---|---|"]
        for r in sel:
            a, b = r["r"]["2016-19"], r["r"]["2020-25"]
            md.append(f"| {r['label']} | {r['val']} | {a[0][3]:,} / {b[0][3]:,} | {fmt(a[0])} | {fmt(b[0])} | {fmt(b[1])} | {fmt(b[2])} | {'○' if r['sig'] else ''} |")
        return md + [""]

    rows.sort(key=lambda r: -r["score"])
    under = [r for r in rows if r["score"] > 0][:20]; over = [r for r in rows if r["score"] < 0][-20:][::-1]
    md = [f"# AI が見落としている角度の順位表 (feature_gap_scan, {pd.Timestamp.now():%Y-%m-%d})", "",
          f"対象 {len(s):,} 走 (2016-2025、AI 順位あり、8 頭以上)。AI = 2016-22 は当時のデータだけで作り直した版、2023-25 は本番 v6。どちらも JRDB の項目は使っていない。",
          "AI 比 = その帯の馬の実際の 3 着内数 ÷ 「同じ AI 順位・頭数の馬の平均」から期待される数。**1 より大きい = AI が過小評価している角度。**",
          "市場比 = 同じ確定オッズの馬の平均に対する比。1 に近ければ市場は織り込み済みで、馬券の妙味は無い。", ""]
    md += tab(under, "1. AI が過小評価している (上げるべきだった) 上位 20")
    md += tab(over, "2. AI が過大評価している (下げるべきだった) 上位 20")
    OUT.write_bytes("\n".join(md).encode("utf-8"))
    (BASE / "docs" / "research" / "FEATURE_GAP_SCAN_20261113.md").write_bytes("\n".join(md).encode("utf-8"))
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
