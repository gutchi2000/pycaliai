# -*- coding: utf-8 -*-
"""
p0_end_to_end_replay.py — P0 gate の実 production 入力による end-to-end 証明
===========================================================================
合成 fixture ではなく、**保存済みの実 weekly / bunseki / bundle** を使い、
shadow directory で 20260913 / 20260919 / 20260920 を再処理する。

**production bundle は一切書き換えない**（読むだけ。出力は shadow へ）。

出力: analysis/jump_history_only/out/p0_end_to_end_replay.json
      analysis/jump_history_only/shadow/bundle_replay/{date}_bundle.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))

import pandas as pd  # noqa: E402

from race_eligibility import (  # noqa: E402
    evaluate_race, eligibility_metadata, verify_metadata, assert_bettable,
    JumpRaceBettingError, clear_cache,
)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = Path(__file__).resolve().parent / "out"
SHADOW = Path(__file__).resolve().parent / "shadow" / "bundle_replay"
OUT.mkdir(parents=True, exist_ok=True)
SHADOW.mkdir(parents=True, exist_ok=True)

DATES = [20260913, 20260919, 20260920]
KNOWN_JUMP = {
    20260913: "2026091306040401",
    20260919: "2026091909040504",
    20260920: "2026092009040601",
}
SAMPLE_BETS = [{"馬券種": "複勝", "買い目": "3", "購入額": 1000}]


def log(m):
    print(m, flush=True)


def sha256_file(p: Path) -> str | None:
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def replay(date: int) -> dict:
    clear_cache()
    d = str(date)
    weekly = BASE / "data" / "weekly" / f"{d}.csv"
    bunseki = BASE / "data" / "bunseki" / f"{d}.csv"
    bundle_p = BASE / "reports" / "cowork_input" / f"{d}_bundle.json"

    r: dict = {
        "date": date,
        "input_hashes": {
            "weekly": sha256_file(weekly),
            "bunseki": sha256_file(bunseki),
            "bundle_production": sha256_file(bundle_p),
        },
        "known_jump_rid": KNOWN_JUMP[date],
    }

    # --- 実 production bundle を読む (書き換えない) ---
    bundle = json.loads(bundle_p.read_text(encoding="utf-8"))
    races = bundle["races"] if isinstance(bundle, dict) else bundle
    all_rids = [str(x.get("race_id"))[:16] for x in races]
    r["n_races_in_production_bundle"] = len(all_rids)

    # --- 実 weekly の全 race (bundle に入る前の母集団) ---
    from build_horse_history import parse_weekly_light
    w = parse_weekly_light(weekly)
    weekly_rids = sorted(set(w["rid16"])) if not w.empty else []
    r["n_races_in_weekly"] = len(weekly_rids)

    # --- eligibility を全 race で評価 ---
    evals = {rid: evaluate_race(rid) for rid in weekly_rids}
    jump = [rid for rid, e in evals.items() if e["is_jump"] is True]
    unknown = [rid for rid, e in evals.items() if e["determination"] == "unknown"]
    flat = [rid for rid, e in evals.items() if e["is_jump"] is False]

    r.update({
        "n_jump_races": len(jump), "jump_race_ids": jump,
        "n_flat_races": len(flat),
        "n_unknown_races": len(unknown), "unknown_race_ids": unknown,
    })

    # 判定に使った field/source
    r["jump_decision_evidence"] = {
        rid: {
            "track_code_value": evals[rid]["raw_fields"]["track_code_value"],
            "track_code_source": evals[rid]["raw_fields"]["track_code_source"],
            "flat_jump_value": evals[rid]["raw_fields"]["flat_jump_value"],
            "flat_jump_source": evals[rid]["raw_fields"]["flat_jump_source"],
            "determination": evals[rid]["determination"],
        } for rid in jump
    }
    # 平地側のソース分布（default が証拠に使われていないことの確認）
    r["flat_source_distribution"] = (
        pd.Series([evals[rid]["raw_fields"]["track_code_source"]
                   for rid in flat]).value_counts().to_dict() if flat else {})

    # --- shadow bundle を作る (production は触らない) ---
    kept, dropped = [], []
    for race in races:
        rid = str(race.get("race_id"))[:16]
        el = evaluate_race(rid)
        if el["prediction_eligible"]:
            race = dict(race)
            race["eligibility"] = eligibility_metadata(el)
            kept.append(race)
        else:
            dropped.append(rid)
    shadow_path = SHADOW / f"{d}_bundle.json"
    shadow_path.write_text(
        json.dumps({"races": kept}, ensure_ascii=False), encoding="utf-8")

    r["bundle_kept_flat_races"] = len(kept)
    r["bundle_excluded_jump_race_ids"] = dropped
    prod_flat = [rid for rid in all_rids if evaluate_race(rid)["is_jump"] is False]
    r["n_flat_in_production_bundle"] = len(prod_flat)
    r["wrongly_excluded_flat_races"] = len(
        [rid for rid in prod_flat if rid in dropped])

    # --- task schedule ---
    from t10_runner import build_schedule
    sched, _missing = build_schedule(d, races, lead_min=10)
    sched_rids = {rid for _dt, rid, _lab in sched}
    r["task_schedule_jump_races"] = len(sched_rids & set(jump))
    r["task_schedule_flat_races"] = len(sched_rids - set(jump))

    # --- compute_bets ---
    from compute_bets import compute_race_bets
    jr = next((x for x in races if str(x.get("race_id"))[:16] == KNOWN_JUMP[date]),
              None)
    if jr is not None:
        out = compute_race_bets(dict(jr))
        r["compute_bets_on_known_jump"] = {
            "n_bets": len(out.get("bets") or []),
            "race_nature": out.get("race_nature"),
            "excluded_reason": out.get("excluded_reason"),
        }
    else:
        r["compute_bets_on_known_jump"] = {"note": "bundle に該当 race なし"}

    # --- validate (shadow の一時ファイルで) ---
    import validate_cowork_bets as v
    tmp_bets = SHADOW / f"{d}_bets_shadow.json"
    tmp_bets.write_text(json.dumps(
        [{"race_id": KNOWN_JUMP[date], "race_label": "障害", "bets": SAMPLE_BETS}],
        ensure_ascii=False), encoding="utf-8")
    argv_bak = sys.argv
    try:
        sys.argv = ["validate_cowork_bets.py", "--bets", str(tmp_bets),
                    "--bundle", str(bundle_p), "--date", d]
        rc = v.main()
    finally:
        sys.argv = argv_bak
    r["validate_rejects_nonempty_bets"] = (rc != 0)
    r["validate_exit_code"] = rc

    # --- final submit ---
    try:
        assert_bettable(KNOWN_JUMP[date], SAMPLE_BETS, layer="replay_final")
        r["final_submit_refuses"] = False
    except JumpRaceBettingError:
        r["final_submit_refuses"] = True

    try:
        import masters_vote as mv
        mv.submit({"race_id": KNOWN_JUMP[date],
                   "bet_data": [{"kind": "fuku", "umaban": 3}]},
                  {"check_delay_sec": 1})
        r["masters_vote_submit_refuses"] = False
    except JumpRaceBettingError:
        r["masters_vote_submit_refuses"] = True
    except Exception as e:
        r["masters_vote_submit_refuses"] = False
        r["masters_vote_submit_error"] = str(e)[:120]

    # --- production bundle を書き換えていないことの確認 ---
    r["production_bundle_hash_after"] = sha256_file(bundle_p)
    r["production_bundle_unchanged"] = (
        r["production_bundle_hash_after"] == r["input_hashes"]["bundle_production"])
    return r


def main() -> int:
    log("=" * 74)
    log("P0 gate — 実 production 入力による end-to-end 再生 (shadow 出力)")
    log("=" * 74)

    results = [replay(d) for d in DATES]

    log(f"\n{'date':>10} {'weekly':>7} {'jump':>5} {'flat':>5} {'unk':>4} "
        f"{'bundle残':>9} {'誤除外':>7} {'task障':>7} {'task平':>7}")
    for r in results:
        log(f"{r['date']:>10} {r['n_races_in_weekly']:>7} {r['n_jump_races']:>5} "
            f"{r['n_flat_races']:>5} {r['n_unknown_races']:>4} "
            f"{r['bundle_kept_flat_races']:>9} "
            f"{r['wrongly_excluded_flat_races']:>7} "
            f"{r['task_schedule_jump_races']:>7} "
            f"{r['task_schedule_flat_races']:>7}")

    log("\n--- 期待結果の検証 ---")
    checks = []
    for r in results:
        d = r["date"]
        checks += [
            (f"{d} 既知障害が bundle から除外",
             r["known_jump_rid"] in r["bundle_excluded_jump_race_ids"]),
            (f"{d} task 登録 0 (障害)", r["task_schedule_jump_races"] == 0),
            (f"{d} compute_bets 0 点",
             r["compute_bets_on_known_jump"].get("n_bets") == 0),
            (f"{d} validate が非空買い目を拒否",
             r["validate_rejects_nonempty_bets"] is True),
            (f"{d} submit 直前で拒否", r["final_submit_refuses"] is True),
            (f"{d} masters_vote.submit で拒否",
             r["masters_vote_submit_refuses"] is True),
            (f"{d} 通常レースを誤除外していない",
             r["wrongly_excluded_flat_races"] == 0),
            (f"{d} 通常レースが task に残る",
             r["task_schedule_flat_races"] > 0),
            (f"{d} production bundle 不変",
             r["production_bundle_unchanged"] is True),
            (f"{d} default を証拠に使っていない",
             "default" not in r["flat_source_distribution"]),
        ]
    for name, ok in checks:
        log(f"  {'PASS' if ok else 'FAIL'}  {name}")
    n_fail = sum(1 for _n, ok in checks if not ok)

    payload = {"generated_at": datetime.now().isoformat(),
               "dates": DATES, "results": results,
               "checks": [{"name": n, "pass": bool(o)} for n, o in checks],
               "n_checks": len(checks), "n_fail": n_fail}
    (OUT / "p0_end_to_end_replay.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")

    log("-" * 74)
    log(f"{len(checks) - n_fail}/{len(checks)} PASS")
    log(f"保存: {OUT / 'p0_end_to_end_replay.json'}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
