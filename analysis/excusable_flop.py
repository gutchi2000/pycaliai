# -*- coding: utf-8 -*-
"""
excusable_flop.py — 「言い訳のある大敗」の次走は過小評価されているか (2026-11-13)
======================================================================
きっかけ: ユーザーが読んだ記事の「特徴量効用値」(一か所だけ凹んだ負けを再演して、運か見落としかを分ける中間指標)。
その芯 = 「異常な負けのうち、不利・出遅れという言い訳があるものは割り引くべき」を、手元のデータで直接検定する。

データ: data/jrdb/sed.parquet (JRDB 成績、2011-2026。出遅・不利・前/中/後不利は IDM 補正量 = 不利の大きさ)、
        kekka (複勝払戻)、AI 順位 (2016-22 は作り直し、2023-25 は本番 v6 の OOS)。
定義 (すべて前走の話):
  大敗 (市場) = 3 番人気以内で 6 着以下 / 大敗 (AI) = AI 3 位以内で 6 着以下 / 凹み (自己比) = その馬の直前 3 走平均より IDM が 8 以上低い
  言い訳 E = 出遅 + 不利 + 前不利 + 中不利 + 後不利 (IDM 点)。0 / 1 / 2 以上。
測るもの (今走):
  実績 / 期待 (A/E)。期待は「同じ確定オッズの馬の平均勝率」(市場) と「同じ AI 順位・頭数の馬の平均 3 着内率」(AI)。
  1 より大きければ過小評価。あわせて単勝・複勝の回収率。期間は 2013-19 / 2020-25 に分ける。
出力: reports/excusable_flop.md (+ docs/research 控え)
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "reports" / "excusable_flop.md"
ERAS = {"2013-19": (2013, 2019), "2020-25": (2020, 2025)}


def build() -> pd.DataFrame:
    s = pd.read_parquet(BASE / "data" / "jrdb" / "sed.parquet")
    s = s[s.ymd <= 20251228].copy()
    s["ped"] = s.ped.astype(str).str.zfill(8)
    for c in ("deokure", "furi", "mae_furi", "naka_furi", "ato_furi"):
        s[c] = s[c].fillna(0).clip(lower=0)
    s["E"] = s.deokure + s.furi + s.mae_furi + s.naka_furi + s.ato_furi
    s["E_furi"] = s.furi + s.mae_furi + s.naka_furi + s.ato_furi
    s = s.sort_values(["ped", "ymd"]).reset_index(drop=True)
    g = s.groupby("ped", sort=False)
    for c in ("jyun", "ninki", "tosu", "tan_odds", "E", "E_furi", "deokure", "idm", "ymd", "ijou", "ato_furi"):
        s["p_" + c] = g[c].shift(1)
    # 前走より前の 3 走の IDM 平均 (前走を含めない)
    s["base_idm"] = g.idm.transform(lambda x: x.shift(2).rolling(3, min_periods=2).mean())
    s["dip"] = s.p_idm - s.base_idm
    s["year"] = s.ymd // 10000
    s = s[(s.year >= 2013) & s.p_jyun.notna() & (s.ijou == 0) & s.tan_odds.notna() & (s.tan_odds > 0) & (s.tosu >= 8)].copy()
    s["win"] = (s.jyun == 1).astype(int); s["top3"] = (s.jyun <= 3).astype(int)
    s["tan_pay"] = s.win * s.tan_odds * 100
    # rid16 / 馬番: master_v2 (血統登録番号 下 8 桁 + 日付)。kekka マスターは 3 着内の馬しか行が無いので使わない。
    # 複勝払戻: data/_joint/mkt_horses.parquet の fpay (3 着外は 0)。
    m = pd.read_csv(BASE / "data" / "master_v2_20130105-20251228.csv", encoding="utf-8-sig", dtype=str,
                    usecols=["レースID(新/馬番無)", "馬番", "血統登録番号", "日付"])
    m["ped"] = m["血統登録番号"].str[-8:]; m["ymd"] = pd.to_numeric(m["日付"], errors="coerce")
    m = m.dropna(subset=["ymd"]); m["ymd"] = m.ymd.astype(int)
    m["rid"] = m["レースID(新/馬番無)"].str[:16]; m["kban"] = pd.to_numeric(m["馬番"], errors="coerce")
    s = s.merge(m[["ped", "ymd", "rid", "kban"]].drop_duplicates(["ped", "ymd"]), on=["ped", "ymd"], how="left")
    mk = pd.read_parquet(BASE / "data" / "_joint" / "mkt_horses.parquet", columns=["rid", "ban", "fpay"]).rename(columns={"ban": "kban"})
    mk["has_mkt"] = 1
    s = s.merge(mk, on=["rid", "kban"], how="left")
    s["fuku_pay"] = np.where(s.has_mkt == 1, s.fpay.fillna(0.0), np.nan)
    # AI 順位 (今走と前走)
    a1 = pd.read_parquet(BASE / "data" / "_ai_scores_wf_2016_2022.parquet")[["rid", "ban", "ai_rk"]]
    sc = pd.read_parquet(BASE / "data" / "_ev_grid_scores.parquet")
    sc["rid"] = sc["レースID(新/馬番無)"].astype(str).str[:16]; sc["ban"] = pd.to_numeric(sc["馬番"], errors="coerce")
    sc["ai_rk"] = sc.groupby("rid")._score.rank(ascending=False, method="first"); sc = sc[sc.rid.str[:4].astype(int) >= 2023]
    ai = pd.concat([a1, sc[["rid", "ban", "ai_rk"]]]).rename(columns={"ban": "kban"})
    s = s.merge(ai, on=["rid", "kban"], how="left")
    s = s.sort_values(["ped", "ymd"]); s["p_ai_rk"] = s.groupby("ped", sort=False).ai_rk.shift(1)
    # 期待値: 市場 (確定オッズの細帯ごとの実勝率、期間内) と AI (順位 × 頭数帯ごとの 3 着内率)
    s["era"] = np.where(s.year <= 2019, "2013-19", "2020-25")
    s["ob"] = pd.qcut(np.log(s.tan_odds), 60, duplicates="drop")
    s["exp_win"] = s.groupby(["era", "ob"], observed=True).win.transform("mean")
    s["exp_top3_mkt"] = s.groupby(["era", "ob"], observed=True).top3.transform("mean")
    s["nb"] = pd.cut(s.tosu, [0, 10, 14, 18])
    s["exp_top3_ai"] = s.groupby(["era", "ai_rk", "nb"], observed=True).top3.transform("mean").where(s.ai_rk.notna())
    s["date"] = s.ymd.astype(str)
    s["gap"] = (pd.to_datetime(s.ymd.astype(int).astype(str)) - pd.to_datetime(s.p_ymd.astype(int).astype(str))).dt.days
    return s


def ae(x: pd.DataFrame, act: str, exp: str) -> str:
    v = x.dropna(subset=[exp])
    if len(v) < 100:
        return "-"
    e = v[exp].sum(); a = v[act].sum()
    se = np.sqrt((v[exp] * (1 - v[exp])).sum()) / e
    r = a / e
    return f"{r:.3f} [{r - 1.96 * se:.2f},{r + 1.96 * se:.2f}]"


def roi(x: pd.DataFrame, col: str, rng=np.random.default_rng(0)) -> str:
    v = x.dropna(subset=[col])
    if len(v) < 100:
        return "-"
    bd = v.groupby("date")[col].agg(["sum", "size"]).values
    i = rng.integers(0, len(bd), (1500, len(bd))); b = bd[i, 0].sum(1) / bd[i, 1].sum(1)
    return f"{v[col].mean():.1f} [{np.percentile(b, 2.5):.0f},{np.percentile(b, 97.5):.0f}]"


def table(s: pd.DataFrame, groups: dict, title: str) -> list[str]:
    md = [f"### {title}", "", "| 前走の条件 | 期間 | n | 勝率 | 市場比 A/E (勝ち) | 市場比 A/E (3着内) | AI 比 A/E (3着内) | 単勝回収率 | 複勝回収率 |", "|---|---|---|---|---|---|---|---|---|"]
    for name, mask in groups.items():
        for era, (a, b) in ERAS.items():
            x = s[mask & (s.year >= a) & (s.year <= b)]
            if len(x) < 100:
                md.append(f"| {name} | {era} | {len(x)} | - | - | - | - | - | - |"); continue
            md.append(f"| {name} | {era} | {len(x):,} | {100 * x.win.mean():.1f}% | {ae(x, 'win', 'exp_win')} | {ae(x, 'top3', 'exp_top3_mkt')} | "
                      f"{ae(x, 'top3', 'exp_top3_ai')} | {roi(x, 'tan_pay')} | {roi(x, 'fuku_pay')} |")
    return md + [""]


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    s = build()
    print(f"{len(s):,} 走 (前走あり、8 頭以上、2013-2025)", flush=True)
    okp = s.p_ijou == 0
    flop_m = okp & (s.p_ninki <= 3) & (s.p_jyun >= 6)
    flop_big = okp & (s.p_ninki <= 3) & (s.p_jyun >= 10)
    flop_ai = okp & (s.p_ai_rk <= 3) & (s.p_jyun >= 6)
    dipm = okp & (s.dip <= -8)
    e0, e1, e2, e3 = s.p_E == 0, s.p_E == 1, s.p_E >= 2, s.p_E >= 3
    md = [f"# 言い訳のある大敗の次走は過小評価されているか (excusable_flop, {pd.Timestamp.now():%Y-%m-%d})", "",
          f"対象 {len(s):,} 走 (前走のある馬、8 頭以上、2013-2025)。前走の出遅・不利は JRDB の補正量 (IDM 点)。",
          "A/E = 実際の的中数 ÷ 期待数。市場比は「同じ確定オッズの馬の平均」、AI 比は「同じ AI 順位・頭数の馬の平均」に対する比。**1 より大きい = その相手に過小評価されている**。[ ] は 95% 区間。",
          "回収率は確定オッズ・確定払戻、[ ] は開催日 bootstrap 95%。", ""]
    md += table(s, {"全馬 (基準)": s.year > 0, "言い訳なし E=0": e0, "E=1 (軽い出遅など)": e1, "E≥2": e2, "E≥3": e3,
                    "不利 (前/中/後 含む) ≥2": s.p_E_furi >= 2, "出遅 ≥2": s.p_deokure >= 2, "後不利 ≥2 (直線)": s.p_ato_furi >= 2}, "1. 前走の言い訳の大きさだけで見る")
    md += table(s, {"市場の大敗 (3人気内→6着以下)": flop_m, "  うち 言い訳なし": flop_m & e0, "  うち E=1": flop_m & e1, "  うち E≥2": flop_m & e2,
                    "  うち E≥3": flop_m & e3, "市場の大敗 (3人気内→10着以下)": flop_big, "  うち E≥2 ": flop_big & e2}, "2. 人気を裏切った大敗 × 言い訳")
    md += table(s, {"AI の大敗 (AI 3位内→6着以下)": flop_ai, "  うち 言い訳なし": flop_ai & e0, "  うち E≥2": flop_ai & e2,
                    "AI も市場も裏切った (両方 3 位内→6着以下)": flop_ai & flop_m, "  うち E≥2 ": flop_ai & flop_m & e2}, "3. AI の予想を裏切った大敗 × 言い訳 (2017 年以降)")
    md += table(s, {"前走だけ凹んだ (自己平均より IDM −8 以下)": dipm, "  うち 言い訳なし": dipm & e0, "  うち E≥2": dipm & e2,
                    "前走だけ凹んだ (−15 以下)": okp & (s.dip <= -15), "  うち E≥2 ": okp & (s.dip <= -15) & e2}, "4. 一か所だけ凹んだ走り (その馬自身の近走比) × 言い訳")
    OUT.write_bytes("\n".join(md).encode("utf-8"))
    s[["ped", "ymd", "rid", "kban", "year", "tan_odds", "ninki", "jyun", "win", "top3", "tan_pay", "fuku_pay", "ai_rk", "p_ai_rk", "p_jyun", "p_ninki", "p_E",
       "p_E_furi", "p_deokure", "p_ato_furi", "p_idm", "base_idm", "dip", "gap", "exp_win", "exp_top3_mkt", "exp_top3_ai", "idm"]].to_parquet(BASE / "data" / "_joint" / "excusable_flop.parquet")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
