# -*- coding: utf-8 -*-
"""
observation_report.py — EXP05-F前向き観測の日次/累計レポート生成 (読み取り専用)
================================================================================
2026-09-19深夜、基盤実装凍結の直前に追加。t35_shadow.ps1/jvlink_race_calendar.py/
market_snapshot.py/predict_and_store.pyのいずれも変更しない、既存の出力(ログ・
calendar JSON・odds JSON・prediction JSON)だけを読んで集計する完全な読み取り専用
ツール。収集ロジックへは一切干渉しない。

3つの計数カテゴリの定義 (ユーザー指定、厳密に区別する):
  market_observations           : そのレースについて何らかの市場snapshotが
                                   取得できた(reports/exp05fs_odds/{rid}.json の
                                   ok=true)。時間窓の有効性やM1/M3/M4の成否は問わない。
  complete_prediction_observations : M1/M3/M4まで完全計算・保存できた
                                   ({rid}_{model_hash}_rev1.json が存在、
                                   marketonlyではない)。
  valid_primary_observations    : complete_prediction_observations のうち、
                                   (a) レコードのdate/race_idが対象日と一致
                                   (b) model_hashが現在の凍結モデル(out/freeze_manifest.json
                                       のartifact_sha256_16)と一致
                                   (c) valid_for_primary=true (T-35の31-38分
                                       ウィンドウ内で取得された市場データ)
                                   の全てを満たす。spec §16の「6600レース」観測の
                                   主評価対象としてカウントしてよいのはこのカテゴリのみ。

実行:
  python -m analysis.mcond.exp05_forward_shadow.observation_report --date 20260920
  python -m analysis.mcond.exp05_forward_shadow.observation_report --cumulative
"""
from __future__ import annotations
import argparse
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
ODDS_DIR = BASE / "reports" / "exp05fs_odds"
PRED_DIR = BASE / "data" / "_research" / "mcond" / "exp05fs_predictions"
CALENDAR_DIR = BASE / "data" / "_research" / "mcond" / "exp05fs_calendar"
WEEKLY_DIR = BASE / "data" / "weekly"
LOGS_DIR = BASE / "logs"


def _current_model_hash() -> str:
    manifest = HERE / "out" / "freeze_manifest.json"
    if not manifest.exists():
        return ""
    try:
        return json.loads(manifest.read_text(encoding="utf-8")).get("artifact_sha256_16", "")
    except Exception:
        return ""


def _rid16_from_odds_filename(p: Path) -> str:
    return p.stem


def _venue_of(rid: str) -> str:
    """rid16の場コード(9-10桁目、例 06=中山/09=阪神)を返す。長さ不正なら '??'。"""
    return rid[8:10] if len(rid) == 16 else "??"


def build_report(date_str: str) -> dict:
    model_hash = _current_model_hash()
    out: dict = {"date": date_str, "generated_at": datetime.now().isoformat(timespec="seconds"),
                "current_model_hash": model_hash}

    # ---- calendarタスク ----
    scheduled_rids: set[str] = set()
    cal_path = CALENDAR_DIR / f"{date_str}.json"
    if cal_path.exists():
        try:
            cal = json.loads(cal_path.read_text(encoding="utf-8"))
            out["calendar_task_result"] = "success"
            out["calendar_generated_at"] = cal.get("generated_at")
            out["calendar_race_count"] = cal.get("record_count")
            out["calendar_source_record_version"] = cal.get("source_record_version")
            scheduled_rids = {re.sub(r"\D", "", str(r))[:16] for r in (cal.get("race_ids") or [])}
        except Exception as exc:
            out["calendar_task_result"] = f"json読込失敗: {exc}"
            out["calendar_race_count"] = None
    else:
        out["calendar_task_result"] = "JSON未生成(未実行、またはサニティチェック失敗で書込拒否)"
        out["calendar_race_count"] = None

    # ---- weekly CSV ----
    weekly = WEEKLY_DIR / f"{date_str}.csv"
    out["weekly_csv_generated_at"] = (
        datetime.fromtimestamp(weekly.stat().st_mtime).isoformat(timespec="seconds")
        if weekly.exists() else None)

    # ---- 市場snapshot (odds JSON) ----
    odds_files = sorted(ODDS_DIR.glob(f"{date_str}*.json"))
    market_ok_rids = set()
    for p in odds_files:
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if d.get("ok"):
            market_ok_rids.add(_rid16_from_odds_filename(p))
    out["market_observations"] = len(market_ok_rids)
    out["fired_task_count"] = len(odds_files)  # oddsファイルが存在する=タスクが発火し取得を試みた

    # ---- market-only 保存 ----
    day_pred_dir = PRED_DIR / date_str
    marketonly_rids = set()
    if day_pred_dir.exists():
        for p in day_pred_dir.glob("*_marketonly_rev1.json"):
            marketonly_rids.add(p.name.split("_marketonly_")[0])
    out["market_only_count"] = len(marketonly_rids)

    # ---- 完全予測保存 (M1/M3/M4) ----
    complete_rids = set()
    valid_primary_rids = set()
    retrospective_rids = set()
    invalid_reasons: Counter = Counter()
    if day_pred_dir.exists():
        for p in day_pred_dir.glob(f"*_{model_hash}_rev1.json") if model_hash else []:
            m = re.match(r"^(\d{16})_", p.name)
            if not m:
                continue
            rid = m.group(1)
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            records = d.get("records") or []
            if not records:
                continue
            complete_rids.add(rid)
            rec0 = records[0]
            date_ok = d.get("date") == date_str and rec0.get("race_id") == rid
            hash_ok = rec0.get("model_hash") == model_hash
            vfp = bool(rec0.get("valid_for_primary"))
            if rec0.get("retrospective_recovery"):
                retrospective_rids.add(rid)
            if vfp:
                if date_ok and hash_ok:
                    valid_primary_rids.add(rid)
            else:
                invalid_reasons[str(rec0.get("invalid_reason"))] += 1
    out["complete_prediction_observations"] = len(complete_rids)
    out["valid_primary_observations"] = len(valid_primary_rids)
    out["invalid_for_primary_count"] = len(complete_rids) - len(valid_primary_rids)
    out["invalid_for_primary_reasons"] = dict(invalid_reasons)
    out["retrospective_recovery_count"] = len(retrospective_rids)

    # ---- venue別内訳 (2026-09-21追加: venue06全欠測の教訓。scheduled/fired/market/
    # market_only/complete/valid_primaryをrid16[8:10](場コード)別に集計する) ----
    fired_rids = {_rid16_from_odds_filename(p) for p in odds_files}
    by_venue: dict[str, dict] = {}
    for rid in (scheduled_rids | fired_rids | market_ok_rids | marketonly_rids
               | complete_rids | valid_primary_rids):
        v = _venue_of(rid)
        by_venue.setdefault(v, {"scheduled": 0, "fired": 0, "market": 0,
                                "market_only": 0, "complete": 0, "valid_primary": 0})
    for rid in scheduled_rids:
        by_venue[_venue_of(rid)]["scheduled"] += 1
    for rid in fired_rids:
        by_venue[_venue_of(rid)]["fired"] += 1
    for rid in market_ok_rids:
        by_venue[_venue_of(rid)]["market"] += 1
    for rid in marketonly_rids:
        by_venue[_venue_of(rid)]["market_only"] += 1
    for rid in complete_rids:
        by_venue[_venue_of(rid)]["complete"] += 1
    for rid in valid_primary_rids:
        by_venue[_venue_of(rid)]["valid_primary"] += 1
    for v, c in by_venue.items():
        c["complete_prediction_rate"] = (
            round(c["complete"] / c["scheduled"], 3) if c["scheduled"] else None)
    out["by_venue"] = dict(sorted(by_venue.items()))

    # ---- missed (入力遅延で一度もタスクを作れなかったレース) ----
    missed_path = LOGS_DIR / f"exp05fs_missed_races_{date_str}.json"
    if missed_path.exists():
        try:
            missed = json.loads(missed_path.read_text(encoding="utf-8"))
            missed = missed if isinstance(missed, list) else [missed]
        except Exception:
            missed = []
    else:
        missed = []
    out["missed_count"] = len(missed)
    out["missed_reasons"] = dict(Counter(m.get("reason", "unknown") for m in missed))

    # ---- T-35個別タスク数 (目標件数。calendar優先、無ければweekly CSVの発走時刻件数) ----
    if out["calendar_race_count"] is not None:
        out["t35_task_target_count"] = out["calendar_race_count"]
    elif weekly.exists():
        try:
            import csv as _csv
            with open(weekly, encoding="cp932", errors="replace", newline="") as f:
                reader = _csv.DictReader(f)
                rid_col = next((c for c in (reader.fieldnames or []) if "レースID" in c), None)
                rids = {re.sub(r"\D", "", str(row.get(rid_col, "")))[:16]
                       for row in reader} if rid_col else set()
                out["t35_task_target_count"] = len({r for r in rids if len(r) == 16})
        except Exception:
            out["t35_task_target_count"] = None
    else:
        out["t35_task_target_count"] = None

    # ---- エラーログ ----
    def _grep_date(path: Path, needle_date: str) -> list[str]:
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return [l for l in lines if needle_date in l]

    out["jvlink_calendar_errors"] = _grep_date(LOGS_DIR / "jvlink_calendar_errors.log", date_str)
    exp05fs_errs = _grep_date(LOGS_DIR / "exp05fs_errors.log", date_str)
    out["exp05fs_errors"] = exp05fs_errs
    t35_log = LOGS_DIR / f"t35_shadow_{date_str}.log"
    anomaly_lines = []
    if t35_log.exists():
        anomaly_lines = [l for l in t35_log.read_text(encoding="utf-8", errors="replace").splitlines()
                         if "[ANOMALY]" in l or "[schedule]" in l or "[FAIL]" in l
                         or "[FIRST_MISS]" in l or "[FINAL_FAIL]" in l]
    out["powershell_task_scheduler_anomalies"] = anomaly_lines

    return out


def build_cumulative() -> dict:
    """全日付にわたるvalid_primary_observationsの累計(spec §16の6600レース向け進捗)。"""
    model_hash = _current_model_hash()
    total_valid = 0
    total_complete = 0
    total_market = 0
    per_date = {}
    if PRED_DIR.exists():
        for day_dir in sorted(PRED_DIR.iterdir()):
            if not day_dir.is_dir():
                continue
            date_str = day_dir.name
            complete = 0
            valid = 0
            for p in (day_dir.glob(f"*_{model_hash}_rev1.json") if model_hash else []):
                m = re.match(r"^(\d{16})_", p.name)
                if not m:
                    continue
                try:
                    d = json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    continue
                records = d.get("records") or []
                if not records:
                    continue
                complete += 1
                rec0 = records[0]
                if (bool(rec0.get("valid_for_primary")) and d.get("date") == date_str
                        and rec0.get("model_hash") == model_hash):
                    valid += 1
            market = len(set(_rid16_from_odds_filename(p) for p in ODDS_DIR.glob(f"{date_str}*.json")
                            if json.loads(p.read_text(encoding="utf-8")).get("ok", False)))
            if complete or valid or market:
                per_date[date_str] = {"complete": complete, "valid_primary": valid, "market": market}
            total_complete += complete
            total_valid += valid
            total_market += market
    return {"generated_at": datetime.now().isoformat(timespec="seconds"),
            "current_model_hash": model_hash,
            "cumulative_market_observations": total_market,
            "cumulative_complete_prediction_observations": total_complete,
            "cumulative_valid_primary_observations": total_valid,
            "target_valid_primary_observations": 6600,
            "per_date": per_date}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="", help="YYYYMMDD")
    ap.add_argument("--cumulative", action="store_true")
    args = ap.parse_args()

    if args.cumulative:
        print(json.dumps(build_cumulative(), ensure_ascii=False, indent=1))
        return 0
    if not args.date:
        print("使い方: --date YYYYMMDD または --cumulative")
        return 1
    print(json.dumps(build_report(args.date), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
