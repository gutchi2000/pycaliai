# -*- coding: utf-8 -*-
"""
hosei_fix_attribution.py — 今の serve コードで「hosei だけ」旧(バグ版)に戻して再生成
=====================================================================================
validate_hosei_fix の結果 (旧bundle 20.73% → 新bundle 27.99%) には、4月以降に入った
他の serve 修正も混ざり得る。コードを現行のまま、data/hosei/H_{date}.csv だけを
git HEAD~1 時点 (= off-by-one 版) に差し替えて再生成し、hosei 修正単独の寄与を出す。

出力: reports/_buggy_hosei_bundles/{date}_bundle.json
実行後、H_{date}.csv は修正版に戻す。
実行: python -m analysis.hosei_fix_attribution
"""
from __future__ import annotations
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from analysis.validate_hosei_fix import DATES, PY, FIXED_DIR, results_for, score_bundle, report  # noqa: E402

BUGGY_DIR = BASE / "reports/_buggy_hosei_bundles"
REF = "aef6aa3f~1"   # off-by-one 修正コミットの直前 = バグ版 H_*.csv


def main() -> None:
    BUGGY_DIR.mkdir(parents=True, exist_ok=True)
    backup = BASE / "reports/_buggy_hosei_bundles/_fixed_hosei_backup"
    backup.mkdir(parents=True, exist_ok=True)
    for d in DATES:
        h = BASE / f"data/hosei/H_{d}.csv"
        csv = BASE / f"data/weekly/{d}.csv"
        if not csv.exists():
            continue
        old = subprocess.run(["git", "show", f"{REF}:data/hosei/H_{d}.csv"],
                             cwd=BASE, capture_output=True)
        if old.returncode != 0:
            print(f"  {d}: バグ版 H が git に無い → skip")
            continue
        if h.exists():
            shutil.copy2(h, backup / h.name)
        try:
            h.write_bytes(old.stdout)
            subprocess.run([str(PY), "export_weekly_marks.py", "--csv", str(csv),
                            "--model", "v6", "--out-dir", str(BUGGY_DIR / "races")],
                           cwd=BASE, capture_output=True)
            agg = BASE / "reports" / f"{d}_bundle.json"
            if agg.exists():
                agg.replace(BUGGY_DIR / f"{d}_bundle.json")
                print(f"  {d}: ok", flush=True)
            else:
                print(f"  {d}: bundle 未生成", flush=True)
        finally:
            if (backup / h.name).exists():
                shutil.copy2(backup / h.name, h)   # 修正版に戻す

    buggy, fixed = [], []
    for d in DATES:
        res = results_for(d)
        if not res:
            continue
        for src, acc in ((BUGGY_DIR, buggy), (FIXED_DIR, fixed)):
            p = src / f"{d}_bundle.json"
            if p.exists():
                acc += score_bundle(p, res)

    print("\n=== 同じ現行コード・違いは hosei だけ ===")
    b = report("hosei バグ版", buggy)
    f = report("hosei 修正版", fixed)
    if b is not None and f is not None:
        m = pd.merge(b, f, on="rid", suffixes=("_b", "_f"))
        from math import comb
        x = int(((m.ai_win_f == 1) & (m.ai_win_b == 0)).sum())
        y = int(((m.ai_win_f == 0) & (m.ai_win_b == 1)).sum())
        p = min(1.0, 2 * sum(comb(x + y, i) for i in range(min(x, y) + 1)) / 2 ** (x + y)) if x + y else 1.0
        print(f"\n  同一 {len(m)}R: ◎勝率 Δ={100*(m.ai_win_f.mean()-m.ai_win_b.mean()):+.2f}pt "
              f"(修正版のみ的中={x} / バグ版のみ={y}, McNemar p={p:.4f})  "
              f"◎top3 Δ={100*(m.ai_top3_f.mean()-m.ai_top3_b.mean()):+.2f}pt")


if __name__ == "__main__":
    main()
