# -*- coding: utf-8 -*-
"""
validate_hosei_fix.py — prev_hosei 修正の効果を 2026 実データで確認する
=====================================================================
hosei エクスポートの 前走補正 充足率は 2026-04/05 で 89〜92% と学習時 (85.4%) 並み。
この期間だけを使えば「修正後の serve」を実際に作れるので、
  旧 bundle (バグあり serve) と 新 bundle (修正後 serve) を同じレースで突き合わせる。

判定: ◎の勝率 / top3 が、同じレースの市場1番人気との差でどこまで戻るか。
  バグあり serve: 市場との差 -7.46pt (勝率) / -8.13pt (top3)
  offline 正常  : 市場との差 -1.22pt / -0.60pt

使い方:
  python -m analysis.validate_hosei_fix --regen     # hosei + marks を作り直してから比較
  python -m analysis.validate_hosei_fix             # 既に作ってあるものを比較するだけ
"""
from __future__ import annotations
import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
PY = BASE / "venv311/Scripts/python.exe"
FIXED_DIR = BASE / "reports/_fixed_bundles"
DATES = ["20260418", "20260419", "20260425", "20260426", "20260502", "20260503",
         "20260509", "20260510", "20260516", "20260517", "20260523", "20260524",
         "20260530", "20260531"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def regen() -> None:
    FIXED_DIR.mkdir(parents=True, exist_ok=True)
    for d in DATES:
        csv = BASE / f"data/weekly/{d}.csv"
        if not csv.exists():
            print(f"  {d}: weekly なし → skip")
            continue
        print(f"  {d}: hosei 再生成 ...", flush=True)
        subprocess.run([str(PY), "make_weekly_hosei.py", "--csv", str(csv)],
                       cwd=BASE, capture_output=True)
        print(f"  {d}: marks 再生成 ...", flush=True)
        r = subprocess.run([str(PY), "export_weekly_marks.py", "--csv", str(csv),
                            "--model", "v6", "--out-dir", str(FIXED_DIR / "races")],
                           cwd=BASE, capture_output=True)
        # rc=2 は品質ゲート不合格 (履歴リプレイでは 出走馬分析ファイルが無い週があるため
        # 母馬/毛色/馬主 等が落ちるのは想定内)。bundle 自体は生成されるので続行する。
        agg = BASE / "reports" / f"{d}_bundle.json"
        if agg.exists():
            agg.replace(FIXED_DIR / f"{d}_bundle.json")
            print(f"    ok (rc={r.returncode}) -> {d}_bundle.json")
        else:
            out = (r.stdout or b"").decode("cp932", "replace")[-300:]
            print(f"    NG rc={r.returncode}: bundle 未生成 {out}")


def results_for(date):
    kp = BASE / "data/kekka" / f"{date}.csv"
    if not kp.exists():
        return {}
    k = pd.read_csv(kp, encoding="cp932", low_memory=False)
    k["rid16"] = k["レースID(新)"].astype(str).str[:16]
    k["ban"] = pd.to_numeric(k["馬番"], errors="coerce")
    k["fin"] = pd.to_numeric(k["確定着順"], errors="coerce")
    out = {}
    for rid, g in k.groupby("rid16"):
        g = g[g.fin.notna() & (g.fin >= 1)]
        t = g[g.fin <= 3].sort_values("fin")
        if len(t) < 3:
            continue
        out[rid] = (int(t.ban.iloc[0]), {int(x) for x in t.ban.iloc[:3]})
    return out


def score_bundle(path, res):
    b = json.loads(Path(path).read_text(encoding="utf-8"))
    races = b["races"] if isinstance(b["races"], list) else list(b["races"].values())
    rows = []
    for r in races:
        rid = "".join(c for c in str(r.get("race_id", "")) if c.isdigit())[:16]
        if rid not in res:
            continue
        hs = [h for h in r.get("horses", []) if h.get("p_win") is not None
              and h.get("umaban") is not None]
        if len(hs) < 5:
            continue
        od = [(h, float(h["tansho_odds"])) for h in hs
              if h.get("tansho_odds") not in (None, "") and float(h["tansho_odds"]) > 0]
        if not od:
            continue
        ai1 = int(max(hs, key=lambda h: float(h["p_win"]))["umaban"])
        fav = int(min(od, key=lambda t: t[1])[0]["umaban"])
        w, top3 = res[rid]
        rows.append(dict(rid=rid, ai_win=int(ai1 == w), ai_top3=int(ai1 in top3),
                         fav_win=int(fav == w), fav_top3=int(fav in top3),
                         agree=int(ai1 == fav)))
    return rows


def report(label, rows):
    if not rows:
        print(f"  {label:<22} (データなし)")
        return None
    d = pd.DataFrame(rows)
    n = len(d)
    aw, at = int(d.ai_win.sum()), int(d.ai_top3.sum())
    lo, hi = wilson(aw, n)
    print(f"  {label:<22} n={n:>4}  AI勝率={100*aw/n:>5.2f}% [{lo:.1f},{hi:.1f}]  "
          f"市場={100*d.fav_win.mean():>5.2f}%  差={100*(d.ai_win.mean()-d.fav_win.mean()):>+6.2f}pt"
          f" | AItop3={100*at/n:>5.2f}% 差={100*(d.ai_top3.mean()-d.fav_top3.mean()):>+6.2f}pt"
          f" | 一致={100*d.agree.mean():>5.1f}%")
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regen", action="store_true")
    args = ap.parse_args()
    if args.regen:
        print("=== 再生成 ===")
        regen()

    old_rows, new_rows = [], []
    for d in DATES:
        res = results_for(d)
        if not res:
            continue
        op = BASE / f"reports/cowork_input/{d}_bundle.json"
        np_ = FIXED_DIR / f"{d}_bundle.json"
        if op.exists():
            old_rows += score_bundle(op, res)
        if np_.exists():
            new_rows += score_bundle(np_, res)

    print("\n=== 2026-04-18 〜 2026-05-31 (hosei 充足率が学習時並みの期間) ===")
    o = report("旧 bundle (バグあり)", old_rows)
    n = report("新 bundle (修正後)", new_rows)
    print("\n  参考 offline 2023-25    n=10299  AI勝率=31.75%  市場=32.97%  差= -1.22pt"
          " | AItop3=63.44% 差= -0.60pt | 一致= 53.9%")

    if o is not None and n is not None:
        # 同一レースだけで対応のある比較
        m = pd.merge(o, n, on="rid", suffixes=("_o", "_n"))
        if len(m):
            dw = 100 * (m.ai_win_n.mean() - m.ai_win_o.mean())
            dt = 100 * (m.ai_top3_n.mean() - m.ai_top3_o.mean())
            b = int(((m.ai_win_n == 1) & (m.ai_win_o == 0)).sum())
            w = int(((m.ai_win_n == 0) & (m.ai_win_o == 1)).sum())
            from math import comb
            p = (sum(comb(b + w, i) for i in range(min(b, w) + 1)) / 2 ** (b + w)
                 * 2) if (b + w) else 1.0
            print(f"\n=== 同一 {len(m)}R での対応比較 ===")
            print(f"  ◎勝率  Δ={dw:+.2f}pt   (新だけ的中={b} / 旧だけ的中={w}, "
                  f"McNemar両側p={min(p,1.0):.4f})")
            print(f"  ◎top3  Δ={dt:+.2f}pt")
            print(f"  ◎が変わったレース: {100*float((m.ai_win_n != m.ai_win_o).mean()):.1f}% "
                  f"(勝敗が入れ替わった率)")


if __name__ == "__main__":
    main()
