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
import argparse
import concurrent.futures
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).parent
WEEKLY = BASE / "data" / "weekly"
OUT_ROOT = BASE / "reports" / "_requant_bundles"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=str(OUT_ROOT),
                    help="再生成bundleの出力先。版ごとに新しいdirを使う")
    ap.add_argument("--force", action="store_true",
                    help="既存bundleも再生成する（履歴dirの指定時は注意）")
    ap.add_argument("--jobs", type=int, default=1,
                    help="並列export数。各日付は独立した出力先を使う")
    args = ap.parse_args()
    out_root = Path(args.out_root)
    csvs = sorted(p for p in WEEKLY.glob("*.csv")
                  if p.stem.isdigit() and len(p.stem) == 8)
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[start] {len(csvs)} weekly CSV を再エクスポート")
    def export_one(item):
        i, csv = item
        date = csv.stem
        bundle = out_root / f"{date}_bundle.json"
        if bundle.exists() and not args.force:
            return "ok", f"  [{i}/{len(csvs)}] {date} skip (済)"
        r = subprocess.run(
            [sys.executable, "export_weekly_marks.py",
             "--csv", str(csv), "--model", "v6",
             "--out-dir", str(out_root / date), "--shap-topk", "0"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            cwd=BASE)
        if r.returncode == 0:
            return "ok", f"  [{i}/{len(csvs)}] {date} OK"
        if r.returncode == 2:
            return "gate", f"  [{i}/{len(csvs)}] {date} GATE_FAIL (bundle は生成済み)"
        tail = (r.stderr or r.stdout or "").strip().splitlines()[-3:]
        return "err", f"  [{i}/{len(csvs)}] {date} ERROR: {' / '.join(tail)}"

    ok, gate_fail, err = 0, 0, 0
    jobs = max(1, int(args.jobs))
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        for status, message in pool.map(export_one, enumerate(csvs, 1)):
            print(message, flush=True)
            ok += int(status == "ok")
            gate_fail += int(status == "gate")
            err += int(status == "err")
    print(f"\n[done] ok={ok} gate_fail={gate_fail} error={err}")
    print(f"次: python build_chaos_quantiles.py --bundle-dir {out_root}")


if __name__ == "__main__":
    main()
