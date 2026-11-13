# -*- coding: utf-8 -*-
"""
switch_production_model.py — 本番モデルを v6 ⇔ v7 に切り替える (方策ファイルと混戦度の基準を 1 回で揃える。2026-11-13)
=============================================================================================
本番が何のモデルかは data/production_policy.json が正本。切替で揃えないといけないもの:
  model / artifacts.rank_model / artifacts.serve_calibrator / chaos_reference (基準分布はモデルごと) / policy_id /
  model_effective_from (較正ドリフト監視がこの日より前の週を除く) / prospective.start_date (前向き cohort の起算)
成果物そのもの (pkl) は動かさない。data/artifact_pins.json は v6・v7 の両方を照合対象に持つ。

  python scripts/switch_production_model.py --to v7 --start 20261114     # v7 を本番に
  python scripts/switch_production_model.py --to v6 --start 20261121     # v6 に戻す (次の開催日を --start に)
  python scripts/switch_production_model.py --status                    # 現在の状態と照合だけ

戻すとき: 進行中の前向き cohort は数え直しになる (モデルが変わると ◎ が変わるため)。当日の bundle は作り直すこと
 (python export_weekly_marks.py --csv data/weekly/YYYYMMDD.csv)。v7 の成果物は git 管理外 (JRDB 規約) なので、
 models/production_backup/20261113_v7/ に控えがある。
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
POLICY = BASE / "data" / "production_policy.json"
CHAOS = BASE / "data" / "chaos_quantiles.json"
REFS = BASE / "data" / "chaos_refs"
MODELS = {
    "v6": dict(rank_model="models/unified_rank_v6.pkl", serve_calibrator="models/pl_calibrators_v6_serve.pkl",
               policy_id="a95-sniper-v2-odds19-20261112", label="v6 (120 特徴)"),
    "v7": dict(rank_model="models/unified_rank_v7.pkl", serve_calibrator="models/pl_calibrators_v7_serve.pkl",
               policy_id="a95-sniper-v2-odds19-v7-20261113", label="v7 (v6 + JRDB 過去走特徴 59 列)"),
}


def status() -> int:
    from production_policy import clear_policy_caches, load_policy, policy_stamp
    clear_policy_caches()
    p = load_policy()
    print(f"model={p.get('model')}  policy_id={p['policy_id']}  chaos={p['chaos_reference']['reference_id']}  "
          f"model_effective_from={p.get('model_effective_from')}  prospective.start={p['prospective']['start_date']}")
    st = policy_stamp()
    print(f"照合 OK: pins={st.get('pins_generation')}  混戦度 {p['chaos_reference']['skip_percentile']} 分位の生値 = {st['chaos_skip_raw_equivalent']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--to", choices=sorted(MODELS))
    ap.add_argument("--start", help="前向き cohort の起算日 YYYYMMDD (次の開催日)")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    if a.status or not a.to:
        return status()
    if not a.start or len(a.start) != 8 or not a.start.isdigit():
        print("--start YYYYMMDD が必要 (前向き cohort の起算日)"); return 2
    m = MODELS[a.to]
    ref = REFS / f"chaos_quantiles_{a.to}.json"
    for rel in (m["rank_model"], m["serve_calibrator"]):
        if not (BASE / rel).exists():
            print(f"成果物が無い: {rel}"); return 2
    if not ref.exists():
        print(f"混戦度の基準分布が無い: {ref}"); return 2
    d = json.loads(POLICY.read_text(encoding="utf-8"))
    old_model, old_id = d.get("model"), d["policy_id"]
    if old_model == a.to:
        print(f"既に {a.to}。変更しない"); return status()
    hist = BASE / "models" / "production_backup" / "policy_history"
    hist.mkdir(parents=True, exist_ok=True)
    shutil.copy2(POLICY, hist / f"production_policy_{datetime.now():%Y%m%dT%H%M%S}_{old_model}.json")
    q = json.loads(ref.read_text(encoding="utf-8"))
    shutil.copy2(ref, CHAOS)
    d["model"] = a.to
    d["policy_id"] = m["policy_id"]
    d["effective_from"] = f"{a.start[:4]}-{a.start[4:6]}-{a.start[6:]}"
    d["model_effective_from"] = int(a.start)
    d["artifacts"]["rank_model"] = m["rank_model"]
    d["artifacts"]["serve_calibrator"] = m["serve_calibrator"]
    d["chaos_reference"]["reference_id"] = q["reference_id"]
    pr = d["prospective"]
    if int(pr["start_date"]) != int(a.start):
        pr["previous_start_dates"] = sorted(set((pr.get("previous_start_dates") or []) + [int(pr["start_date"])]))
        pr["start_date"] = int(a.start)
    d["engine_note"] = (f"{datetime.now():%Y-%m-%d} 予測モデルを {old_model} → {a.to} ({m['label']}) に切替 (前 policy {old_id})。"
                        f"買い方 (sniper: ◎ T-10 オッズ ≤1.9 のみ参戦・◎複勝 1 点) は不変。混戦度の基準分布は {q['reference_id']}。 | 旧: " + str(d.get("engine_note", "")))[:4000]
    POLICY.write_bytes((json.dumps(d, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print(f"切替: {old_model} → {a.to}  policy_id {old_id} → {m['policy_id']}")
    return status()


if __name__ == "__main__":
    sys.exit(main())
