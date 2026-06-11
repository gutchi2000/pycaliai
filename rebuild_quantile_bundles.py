# -*- coding: utf-8 -*-
"""
rebuild_quantile_bundles.py
===========================
chaos_quantiles.json 再生成用に、過去の週次 CSV (data/weekly/*.csv) を
**現行コード** (補正リネーム + serve calibrator + pair_probs) で再エクスポートし、
新分布の bundle を reports/_requant_bundles/ に出力する。

背景 (docs/audit_20260611.md):
  data/chaos_quantiles.json は補正タイム死亡時代の bundle 12 本から作られた。
  補正復活 (2026-06-11) でモデル確信度が上がり chaos/top1 の分布がシフト、
  旧分位テーブルのままだと shape/見送り判定が歪む。
  本番 bundle (reports/cowork_input/) は歴史的成果物なので上書きしない。

使い方:
  PYTHONUTF8=1 python rebuild_quantile_bundles.py
  → 完了後: python build_chaos_quantiles.py --bundle-dir reports/_requant_bundles
"""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).parent
WEEKLY = BASE / "data" / "weekly"
OUT_ROOT = BASE / "reports" / "_requant_bundles"


def main():
    csvs = sorted(p for p in WEEKLY.glob("*.csv")
                  if p.stem.isdigit() and len(p.stem) == 8)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"[start] {len(csvs)} weekly CSV を再エクスポート")
    ok, gate_fail, err = 0, 0, 0
    for i, csv in enumerate(csvs, 1):
        date = csv.stem
        bundle = OUT_ROOT / f"{date}_bundle.json"
        if bundle.exists():
            print(f"  [{i}/{len(csvs)}] {date} skip (済)")
            ok += 1
            continue
        r = subprocess.run(
            [sys.executable, "export_weekly_marks.py",
             "--csv", str(csv), "--model", "v6",
             "--out-dir", str(OUT_ROOT / date), "--shap-topk", "0"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            cwd=BASE)
        if r.returncode == 0:
            ok += 1
            print(f"  [{i}/{len(csvs)}] {date} OK")
        elif r.returncode == 2:
            gate_fail += 1  # 品質ゲート (古い形式CSV等)。bundle は出ているが参考扱い
            print(f"  [{i}/{len(csvs)}] {date} GATE_FAIL (bundle は生成済み)")
        else:
            err += 1
            tail = (r.stderr or r.stdout or "").strip().splitlines()[-3:]
            print(f"  [{i}/{len(csvs)}] {date} ERROR: {' / '.join(tail)}")
    print(f"\n[done] ok={ok} gate_fail={gate_fail} error={err}")
    print(f"次: python build_chaos_quantiles.py --bundle-dir {OUT_ROOT}")


if __name__ == "__main__":
    main()
