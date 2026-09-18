# -*- coding: utf-8 -*-
"""
validate_hosei_proxy_live.py — hosei 途切れ期間 (2026-06〜09) の実レースで proxy を検証
======================================================================================
同じ現行コード・同じ週次CSVで、CB_HOSEI_PROXY=0 / 1 の2通りに印を作り直し、
実 kekka と市場1番人気に対して比べる。offline ゲートとは独立な、実データでの前向き確認。

出力: reports/_proxy_live/{off,on}/{date}_bundle.json
実行: python -m analysis.validate_hosei_proxy_live
"""
from __future__ import annotations
import glob
import os
import subprocess
import sys
from math import comb
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from analysis.validate_hosei_fix import PY, results_for, score_bundle, report  # noqa: E402

OUT = BASE / "reports/_proxy_live"


def dates():
    ds = []
    for f in sorted(glob.glob(str(BASE / "data/kekka/2026*.csv"))):
        d = Path(f).stem
        if d >= "20260601" and (BASE / f"data/weekly/{d}.csv").exists():
            ds.append(d)
    return ds


def build(d, mode):
    dst = OUT / mode / f"{d}_bundle.json"
    if dst.exists():
        return True
    env = dict(os.environ, CB_HOSEI_PROXY="1" if mode == "on" else "0",
               PYTHONIOENCODING="utf-8")
    subprocess.run([str(PY), "make_weekly_hosei.py", "--csv", f"data/weekly/{d}.csv"],
                   cwd=BASE, capture_output=True)
    subprocess.run([str(PY), "export_weekly_marks.py", "--csv", f"data/weekly/{d}.csv",
                    "--model", "v6", "--out-dir", str(OUT / mode / "races" / d)],
                   cwd=BASE, capture_output=True, env=env)
    agg = BASE / "reports" / f"{d}_bundle.json"
    if agg.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        agg.replace(dst)
        return True
    return False


def main() -> None:
    ds = dates()
    print(f"対象 {len(ds)} 開催日: {ds[0]}〜{ds[-1]}", flush=True)
    for d in ds:
        ok = [build(d, m) for m in ("off", "on")]
        print(f"  {d}: off={'ok' if ok[0] else 'NG'} on={'ok' if ok[1] else 'NG'}", flush=True)

    rows = {"served": [], "off": [], "on": []}
    for d in ds:
        res = results_for(d)
        if not res:
            continue
        srv = BASE / f"reports/cowork_input/{d}_bundle.json"
        if srv.exists():
            rows["served"] += score_bundle(srv, res)
        for m in ("off", "on"):
            p = OUT / m / f"{d}_bundle.json"
            if p.exists():
                rows[m] += score_bundle(p, res)

    print("\n=== 2026-06〜09 (TARGET 補正タイムが途切れた期間) の実レース ===")
    r_s = report("実際に配信された印", rows["served"])
    r_o = report("現行コード proxyなし", rows["off"])
    r_n = report("現行コード proxyあり", rows["on"])
    print("  参考 offline 2023-25  AI勝率 31.75% 市場 32.97% 差 -1.22pt / top3差 -0.60pt")
    if r_o is not None and r_n is not None:
        m = pd.merge(r_o, r_n, on="rid", suffixes=("_o", "_n"))
        a = int(((m.ai_win_n == 1) & (m.ai_win_o == 0)).sum())
        b = int(((m.ai_win_n == 0) & (m.ai_win_o == 1)).sum())
        p = min(1.0, 2 * sum(comb(a + b, i) for i in range(min(a, b) + 1)) / 2 ** (a + b)) if a + b else 1
        print(f"\n  同一 {len(m)}R  proxy の効果: ◎勝率 Δ={100*(m.ai_win_n.mean()-m.ai_win_o.mean()):+.2f}pt "
              f"(あり のみ的中={a} / なし のみ={b}, McNemar p={p:.4f})  "
              f"◎top3 Δ={100*(m.ai_top3_n.mean()-m.ai_top3_o.mean()):+.2f}pt")


if __name__ == "__main__":
    main()
