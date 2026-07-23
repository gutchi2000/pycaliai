# -*- coding: utf-8 -*-
"""
analysis/reverify_clean_band_2026.py — クリーン帯ゲートの 2026 as-served 再検証
================================================================================
participation バイパス解消(配線)の前提検証。鉄則=[[project_unwired_roi_audit]]
「配線より最新期間での CI 付き再検証」。

データ: reports/cowork_input/*_bundle.json (2026, 実serve) × data/kekka/{date}.csv (実結果)
検定: ◎複勝/◎単勝 ROI を chaos_pct 帯 (clean≤0.33 / mid / chaotic>0.67) 別に bootstrap CI。
2025 実測 (クリーン90% vs 帯外80-82%) が 2026 で方向再現するかを見る。

実行: python -m analysis.reverify_clean_band_2026
"""
from __future__ import annotations
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]

import sys
sys.path.insert(0, str(BASE))
import compute_bets as CB  # pct() + chaos_quantiles を流用


def roi_ci(pay, n_boot=4000, seed=42):
    n = len(pay)
    if n == 0:
        return {"n": 0}
    rng = np.random.default_rng(seed)
    b = [float(np.mean(pay[rng.integers(0, n, n)])) for _ in range(n_boot)]
    return {"n": n, "roi": round(float(np.mean(pay)), 4),
            "hit": round(float(np.mean(pay > 0)), 4),
            "ci95": [round(float(np.percentile(b, 2.5)), 4),
                     round(float(np.percentile(b, 97.5)), 4)]}


def main():
    rows = []
    for bp in sorted(glob.glob(str(BASE / "reports/cowork_input/*_bundle.json"))):
        date = Path(bp).name[:8]
        if not date.startswith("2026"):
            continue
        kp = BASE / "data" / "kekka" / f"{date}.csv"
        if not kp.exists():
            continue
        k = pd.read_csv(kp, encoding="cp932", low_memory=False)
        k["rid16"] = k["レースID(新)"].astype(str).str[:16]
        k["ban"] = pd.to_numeric(k["馬番"], errors="coerce")
        k["fin"] = pd.to_numeric(k["確定着順"], errors="coerce")
        k["tan"] = pd.to_numeric(k["単勝配当"], errors="coerce")
        k["fuku"] = pd.to_numeric(k["複勝配当"], errors="coerce")
        fuku_map = {(r.rid16, int(r.ban)): r.fuku / 100.0
                    for r in k.itertuples() if pd.notna(r.fuku) and r.fuku > 0 and pd.notna(r.ban)}
        tan_map = {(r.rid16, int(r.ban)): r.tan / 100.0
                   for r in k.itertuples() if r.fin == 1 and pd.notna(r.tan) and pd.notna(r.ban)}
        raced = {r.rid16 for r in k.itertuples()}

        b = json.loads(Path(bp).read_text(encoding="utf-8"))
        races = b["races"] if isinstance(b["races"], list) else list(b["races"].values())
        for r in races:
            rid16 = "".join(ch for ch in str(r.get("race_id", "")) if ch.isdigit())[:16]
            if rid16 not in raced:
                continue
            rc = r.get("race_confidence", {})
            ch = CB.pct(rc.get("field_chaos_score"), "field_chaos_score")
            if ch is None:
                continue
            hon = next((h for h in r.get("horses", []) if h.get("mark") == "◎"), None)
            if hon is None:
                continue
            ban = int(hon.get("umaban"))
            rows.append({"date": date, "rid16": rid16, "chaos_pct": ch,
                         "fuku_pay": fuku_map.get((rid16, ban), 0.0),
                         "tan_pay": tan_map.get((rid16, ban), 0.0)})

    df = pd.DataFrame(rows)
    print(f"[reverify] 2026 as-served races={len(df)}  ({df['date'].min()}〜{df['date'].max()}, "
          f"{df['date'].nunique()}開催日)")
    bands = [("clean(≤0.33)", df["chaos_pct"] <= 0.33),
             ("mid(0.33-0.67)", (df["chaos_pct"] > 0.33) & (df["chaos_pct"] <= 0.67)),
             ("chaotic(>0.67)", df["chaos_pct"] > 0.67),
             ("帯外全体(>0.33)", df["chaos_pct"] > 0.33),
             ("全体", df["chaos_pct"] >= 0)]
    out = {}
    for lbl, m in bands:
        rf = roi_ci(df.loc[m, "fuku_pay"].values)
        rt = roi_ci(df.loc[m, "tan_pay"].values)
        out[lbl] = {"fuku": rf, "tan": rt}
        if rf.get("n"):
            print(f"  {lbl:16s}: ◎複勝 ROI={rf['roi']:.3f} 的中{rf['hit']*100:.1f}% "
                  f"CI{rf['ci95']} n={rf['n']}   ◎単勝 ROI={rt['roi']:.3f}")
    d = (out["clean(≤0.33)"]["fuku"].get("roi") or 0) - (out["帯外全体(>0.33)"]["fuku"].get("roi") or 0)
    print(f"\n  Δ(clean − 帯外) ◎複勝 = {d*100:+.2f}pt   (2025実測: 約+8〜9pt)")
    op = BASE / "reports" / "reverify_clean_band_2026.json"
    json.dump(out, open(op, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[saved] {op}")


if __name__ == "__main__":
    main()
