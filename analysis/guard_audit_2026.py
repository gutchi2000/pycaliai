# -*- coding: utf-8 -*-
"""
guard_audit_2026.py — 参戦ガード (見送り基準) は本当に正しいかを 2026 実配信で検査する (2026-11-07)
================================================================================
本線の見送り条件 (data/production_policy.json): 混戦度 percentile >= 0.667 / 出走 7 頭以下 / ◎ p_win < 0.05 / ◎オッズ欠。
検査:
  1. 実際の判定 (reports/cowork_output の race_nature=='見送り') で 参戦 vs 見送り の ◎均等買い (単勝・複勝) 回収率と的中率。
  2. 混戦度 percentile の 10 分位ごとの ◎成績 (単調に悪化しているか、0.667 の切り方が妥当か)。
  3. 閾値スイープ: 混戦度 cut / 頭数上限 / ◎p_win 下限 を動かしたときの参戦集合の回収率・レース数。
  4. 候補の追加ルール: 新馬で ◎≠1番人気 / ◎の前走 10 着以下 / 8 頭以下で ◎≠1番人気。
◎ の払戻は確定オッズ (kekka) で計算。順位付けは bundle の印をそのまま使う (同率の問題なし)。
出力: reports/guard_audit_2026.md
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from analysis.market_bias_atlas import norm_class  # noqa: E402

OUT = BASE / "reports" / "guard_audit_2026.md"
Q = json.loads((BASE / "data" / "chaos_quantiles.json").read_text(encoding="utf-8"))["quantiles"]["field_chaos_score"]


def chaos_pct(raw) -> float:
    if raw is None or not np.isfinite(raw):
        return np.nan
    return float(np.searchsorted(np.asarray(Q, float), raw, side="right") / 100.0)


def load() -> pd.DataFrame:
    dec = {}
    for p in sorted(glob.glob(str(BASE / "reports" / "cowork_output" / "2026????_bets.json"))):
        raw = json.loads(Path(p).read_text(encoding="utf-8"))
        for e in (raw["bets"] if isinstance(raw, dict) and "bets" in raw else raw):
            rid = re.sub(r"\D", "", str(e.get("race_id", "")))[:16]
            dec[rid] = (e.get("race_nature"), str(e.get("race_reason", ""))[:40], bool(e.get("bets")))
    rows = []
    for bp in sorted(glob.glob(str(BASE / "reports" / "cowork_input" / "2026????_bundle.json"))):
        d = Path(bp).name[:8]
        kp = BASE / "data" / "kekka" / f"{d}.csv"
        if not kp.exists():
            continue
        k = pd.read_csv(kp, encoding="cp932", low_memory=False)
        k["rid"] = k["レースID(新)"].astype(str).str[:16]; k["ban"] = pd.to_numeric(k["馬番"], errors="coerce")
        k["fin"] = pd.to_numeric(k["確定着順"], errors="coerce"); k["fpay"] = pd.to_numeric(k["複勝配当"], errors="coerce")
        t = k["単勝配当"].astype(str)
        k["odds"] = np.where(t.str.match(r"^\d+$"), pd.to_numeric(t, errors="coerce") / 100, pd.to_numeric(t.str.strip("()"), errors="coerce"))
        kd = {(x.rid, int(x.ban)): (x.fin, x.fpay if np.isfinite(x.fpay) else 0.0, x.odds) for x in k.itertuples() if pd.notna(x.ban)}
        b = json.loads(Path(bp).read_text(encoding="utf-8"))
        for r in (b["races"] if isinstance(b["races"], list) else list(b["races"].values())):
            rid = re.sub(r"\D", "", str(r["race_id"]))[:16]
            hs = [h for h in r["horses"] if h.get("p_win") is not None]
            hon = next((h for h in hs if h.get("mark") == "◎"), None)
            if not hon or len(hs) < 5:
                continue
            res = kd.get((rid, int(hon["umaban"])))
            if not res or not np.isfinite(res[0]) or res[0] < 1:
                continue
            rc = r.get("race_confidence") or {}; rm = r.get("race_meta") or {}
            od = [(h, float(h["tansho_odds"])) for h in hs if h.get("tansho_odds")]
            fav = min(od, key=lambda x: x[1])[0]["umaban"] if od else None
            runs = (hon.get("history") or {}).get("runs") or []
            nat = dec.get(rid, (None, "", False))
            rows.append(dict(date=d, rid=rid, R=int(rid[14:16]), chaos=rc.get("field_chaos_score"), field=len(hs),
                             p=float(hon["p_win"]), odds_am=float(hon["tansho_odds"]) if hon.get("tansho_odds") else np.nan,
                             odds=res[2], fin=res[0], fpay=res[1], agree=(fav == hon["umaban"]) if fav else np.nan,
                             klass=norm_class(rm.get("class", "")), prev=(runs[0].get("pos") if runs else None),
                             nature=nat[0], reason=nat[1], had_bets=nat[2]))
    df = pd.DataFrame(rows)
    df["pct"] = df.chaos.map(chaos_pct)
    df["win"] = (df.fin == 1).astype(int); df["top3"] = (df.fin <= 3).astype(int)
    df["tan"] = np.where(df.fin == 1, 100 * df.odds, 0.0); df["fuku"] = np.where(df.fin <= 3, df.fpay, 0.0)
    df["skipped"] = df.nature.eq("見送り")
    return df


RNG = np.random.default_rng(0)


def stat(s: pd.DataFrame, boot: int = 1500) -> dict:
    if len(s) == 0:
        return dict(n=0)
    g = s.groupby("date").agg(n=("tan", "size"), t=("tan", "sum"), f=("fuku", "sum"))
    idx = RNG.integers(0, len(g), (boot, len(g)))
    tan = g.t.values[idx].sum(1) / g.n.values[idx].sum(1); fuk = g.f.values[idx].sum(1) / g.n.values[idx].sum(1)
    return dict(n=int(len(s)), days=int(s.date.nunique()), win=100 * s.win.mean(), top3=100 * s.top3.mean(),
                tan=s.tan.mean(), tan_lo=np.percentile(tan, 2.5), tan_hi=np.percentile(tan, 97.5),
                fuku=s.fuku.mean(), fuku_lo=np.percentile(fuk, 2.5), fuku_hi=np.percentile(fuk, 97.5))


def fmt(st: dict) -> str:
    if st.get("n", 0) == 0:
        return "| 0 | | | | |"
    return (f"| {st['n']} | {st['win']:.1f}% | {st['top3']:.1f}% | {st['tan']:.1f} [{st['tan_lo']:.0f},{st['tan_hi']:.0f}] "
            f"| {st['fuku']:.1f} [{st['fuku_lo']:.0f},{st['fuku_hi']:.0f}] |")


HDR = "| 集合 | R | ◎勝率 | ◎3着内 | 単勝均等 [CI] | 複勝均等 [CI] |\n|---|---|---|---|---|---|"


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    df = load()
    md = [f"# 参戦ガード監査 (2026 実配信 {df.date.min()}〜{df.date.max()}, {len(df)} R, {df.date.nunique()} 開催日)", "",
          "◎の 100 円均等買い (確定オッズ払戻)。CI は開催日 bootstrap 95%。本線の見送り: 混戦度 percentile ≥ 0.667 / 7 頭以下 / ◎p_win < 0.05。", ""]

    md += ["## 1. 実際の判定: 参戦 vs 見送り", "", HDR]
    md.append("| 参戦 (bets あり) " + fmt(stat(df[~df.skipped & df.had_bets])))
    md.append("| 見送り " + fmt(stat(df[df.skipped])))
    md.append("| 全レース " + fmt(stat(df)))
    for key, lab in (("混沌|chaos|混戦|カオス|団子|分散|横並び", "見送り: 混戦度が理由"), ("field_size|少頭数|頭立て", "見送り: 少頭数が理由")):
        md.append(f"| {lab} " + fmt(stat(df[df.skipped & df.reason.str.contains(key)])))
    md.append("")

    md += ["## 2. 混戦度 percentile 10 分位ごとの ◎成績 (ガードの境界 0.667 は上から 1/3)", "", HDR]
    bins = [0, .1, .2, .3, .4, .5, .6, .667, .7, .8, .9, 1.01]
    df["pbin"] = pd.cut(df.pct, bins, right=False)
    for b, s in df.groupby("pbin", observed=True):
        md.append(f"| {b} " + fmt(stat(s, 300)))
    md.append("")

    md += ["## 3. 閾値スイープ (参戦集合の成績)", "", "### 3a. 混戦度の cut (この値未満を参戦、他条件は本線どおり)", "", HDR]
    base = (df.field >= 8) & (df.p >= 0.05)
    for cut in (0.4, 0.5, 0.6, 0.667, 0.75, 0.85, 1.01):
        md.append(f"| pct < {cut} " + fmt(stat(df[base & (df.pct < cut)])))
    md += ["", "### 3b. 頭数の下限 (混戦度 < 0.667 と ◎p_win ≥ 0.05 は固定)", "", HDR]
    for fmin in (6, 8, 9, 10, 12):
        md.append(f"| 出走 ≥ {fmin} 頭 " + fmt(stat(df[(df.pct < 0.667) & (df.p >= 0.05) & (df.field >= fmin)])))
    md += ["", "### 3c. ◎ p_win の下限 (混戦度 < 0.667 と 8 頭以上は固定)", "", HDR]
    for pmin in (0.0, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25):
        md.append(f"| ◎p_win ≥ {pmin:.2f} " + fmt(stat(df[(df.pct < 0.667) & (df.field >= 8) & (df.p >= pmin)])))
    md += ["", "### 3d. ◎の朝オッズ帯 (本線集合の中で)", "", HDR]
    inb = df[base & (df.pct < 0.667)]
    for lo, hi, lab in ((0, 1.9, "〜1.9"), (2, 2.9, "2.0-2.9"), (3, 4.9, "3.0-4.9"), (5, 9.9, "5.0-9.9"), (10, 999, "10+")):
        md.append(f"| ◎朝オッズ {lab} " + fmt(stat(inb[(inb.odds_am >= lo) & (inb.odds_am <= hi)], 300)))
    md.append("")

    md += ["## 4. 候補の追加ルール (本線集合の中で、該当 vs 非該当)", "", HDR]
    cand = {
        "新馬 × ◎≠1番人気": (inb.klass == "新馬") & (inb.agree == False),  # noqa: E712
        "新馬 × ◎=1番人気": (inb.klass == "新馬") & (inb.agree == True),  # noqa: E712
        "◎の前走 10 着以下": pd.to_numeric(inb.prev, errors="coerce") >= 10,
        "◎の前走 1-3 着": pd.to_numeric(inb.prev, errors="coerce").between(1, 3),
        "8-9 頭 × ◎≠1番人気": inb.field.between(8, 9) & (inb.agree == False),  # noqa: E712
        "◎≠1番人気 (全体)": inb.agree == False,  # noqa: E712
        "◎=1番人気 (全体)": inb.agree == True,  # noqa: E712
        "11R": inb.R == 11,
        "12R": inb.R == 12,
    }
    for lab, m in cand.items():
        md.append(f"| {lab} " + fmt(stat(inb[m.fillna(False)], 500)))
    md.append("| (本線集合 全体) " + fmt(stat(inb)))
    md.append("")
    md += ["## 5. 読み方", "",
           "- 1 で 見送り集合の回収率が 参戦集合より明確に低ければ、ガードは「正しい方向」。差が無ければガードは的中率を上げるだけで ROI には効いていない。",
           "- 2 で上位分位ほど単調に悪ければ percentile cut は妥当。山があれば cut の位置を見直す。",
           "- 3 は参戦レース数との交換。R が減りすぎる cut は紙上台帳の 500R に時間がかかる。",
           "- 4 は候補。該当集合が本線全体より明確に低く、かつ R が十分にあるものだけが見送り候補。"]
    OUT.write_bytes("\n".join(md).encode("utf-8"))
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
