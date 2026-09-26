# -*- coding: utf-8 -*-
"""
forward_audit.py — EXP19 Stage 0 S0-A: forward WH の状態・最初の完全 snapshot 時刻・歴史値との parity
====================================================================================================
値を推測しない。floor (4 開催日 / 400 馬行、kg・増減・status 一致 99.5%、race coverage 99%、T−28 complete 95%) に
届かなければ「収集中」として取得済み件数と残数だけを報告する。
  forward: data/forward_bodyweight/ (attempts = 全試行、snapshots = 成功 WH の不変 JSON)
  発走時刻: collector と同じ verified calendar (data/_research/mcond/exp05fs_calendar/{date}.json)
  完全 snapshot: WH record があり、全 horse の weight_status が明示 (normal / 計量不能 / 取消等) である snapshot。
                 出走頭数との一致は各日の TARGET 出走表 (data/weekly/{date}.csv) の頭数で確認できる日だけ判定
  parity 相手: 同一 race ID + 馬番の TARGET 歴史値 (torch 形式の export)。2026 年分は未 export のため現時点は 0 件
  baba_today: 同日の公式保存値 (baba_feats.parquet) と値・venue・測定日を照合。現 parquet は 2025-12-28 までで重なり無し
出力: out/forward_parity.json
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from .loaders import BABA, BABA_TODAY, BASE, FORWARD, OUT, TORCH

CAL = BASE / "data" / "_research" / "mcond" / "exp05fs_calendar"
FLOORS = {"min_days": 4, "min_rows": 400, "value_match": 0.995, "status_match": 0.995, "race_coverage": 0.99,
          "complete_by_Tminus28": 0.95}
T_DECISION_MIN = 28


def load_calendar(date: str) -> dict:
    p = CAL / f"{date}.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text(encoding="utf-8"))
    return {rid: datetime.strptime(f"{date} {hm}", "%Y%m%d %H:%M") for rid, hm in zip(d["race_ids"], d["post_times"])}


def main():
    t0 = time.time()
    days = sorted(p.name for p in (FORWARD / "attempts").iterdir() if p.is_dir()) if (FORWARD / "attempts").exists() else []
    per_day, races_out = {}, {}
    status_counts, n_rows_snap = {}, 0
    for day in days:
        cal = load_calendar(day)
        att = []
        for f in sorted((FORWARD / "attempts" / day).glob("*.json")):
            a = json.loads(f.read_text(encoding="utf-8"))
            att.append(a)
        by_race = {}
        for a in att:
            by_race.setdefault(a["race_id"], []).append(a)
        n_complete_28 = n_with_post = 0
        for rid, lst in by_race.items():
            lst.sort(key=lambda a: a["observed_at"])
            post = cal.get(rid)
            first_ok = None
            horses = None
            for a in lst:
                if a.get("success") and a.get("snapshot"):
                    snap = json.loads((FORWARD / a["snapshot"]).read_text(encoding="utf-8"))
                    hs = snap.get("horses", [])
                    explicit = all(h.get("weight_status") not in (None, "", "unknown") for h in hs)
                    if hs and explicit and first_ok is None:
                        first_ok = a["observed_at"]
                        horses = hs
            rec = {"attempts": len(lst), "successful": sum(bool(a.get("success")) for a in lst),
                   "no_WH_record": sum(not a.get("success") for a in lst),
                   "first_attempt": lst[0]["observed_at"], "first_complete": first_ok,
                   "post": post.isoformat() if post else None}
            if post and first_ok:
                obs = datetime.fromisoformat(first_ok).replace(tzinfo=None)
                rec["first_complete_minus_post_min"] = round((obs - post).total_seconds() / 60, 1)
                first_att = datetime.fromisoformat(lst[0]["observed_at"]).replace(tzinfo=None)
                rec["first_attempt_minus_post_min"] = round((first_att - post).total_seconds() / 60, 1)
                # 最初の試行が T−28 より後なら、T−28 時点の状態は観測されていない (判定不能として数えない)
                rec["Tminus28_observable"] = bool(first_att <= post - timedelta(minutes=T_DECISION_MIN))
                if rec["Tminus28_observable"]:
                    n_with_post += 1
                    n_complete_28 += int(obs <= post - timedelta(minutes=T_DECISION_MIN))
            if horses:
                for h in horses:
                    k = f"{h.get('weight_status')}|{h.get('change_status')}"
                    status_counts[k] = status_counts.get(k, 0) + 1
                n_rows_snap += len(horses)
            races_out[rid] = rec
        per_day[day] = {"calendar_races": len(cal), "races_attempted": len(by_race),
                        "races_with_complete_snapshot": sum(1 for r in by_race if races_out[r]["first_complete"]),
                        "races_Tminus28_observable": n_with_post, "complete_by_Tminus28": n_complete_28}
    # 歴史 TARGET 値との parity: torch export の最終日と forward 期間が重なるか
    torch_last = pd.read_csv(TORCH, encoding="cp932", usecols=["日付"], dtype=str)["日付"].astype(int).max()
    torch_last = 20_000_000 + torch_last if torch_last < 1_000_000 else torch_last
    overlap_days = [d for d in days if int(d) <= torch_last]
    baba = pd.read_parquet(BABA)
    bt = json.loads(BABA_TODAY.read_text(encoding="utf-8")) if BABA_TODAY.exists() else {}
    bt_date = int(str(bt.get("date", "0")).replace("-", "")) if bt else 0
    baba_overlap = int((baba["日付"] == bt_date).sum())
    parity_pairs = 0
    rows_obs = n_rows_snap
    res = {
        "role": "forward WH の収集状況・T−28 完全性・歴史値 parity (値を推測しない)",
        "floors": FLOORS,
        "days_collected": days, "n_days": len(days), "per_day": per_day,
        "horse_rows_in_complete_snapshots": rows_obs,
        "status_counts_weight|change": status_counts,
        "races": races_out,
        "parity_with_historical_target": {
            "torch_export_last_date": int(torch_last), "forward_days_overlapping_export": overlap_days,
            "pairs_compared": parity_pairs,
            "note": "2026 年分の TARGET 馬体重 export (torch 形式) が無いので race ID+馬番の対を作れない。"
                    "forward 収集日の TARGET export を追加すれば同じスクリプトで照合する"},
        "baba_today_parity": {"baba_today_date": bt_date, "baba_feats_rows_same_date": baba_overlap,
                              "baba_feats_last_date": int(baba["日付"].max()),
                              "note": "baba_feats.parquet (PDF 由来) の最終日より後の当日値なので同日保存値が無く照合不能"},
        "status": "収集中",
        "remaining_to_floor": {"days": max(0, FLOORS["min_days"] - len(overlap_days)),
                               "parity_rows": max(0, FLOORS["min_rows"] - parity_pairs),
                               "forward_days": max(0, FLOORS["min_days"] - len(days))},
        "Tminus28": {"races_observable": sum(v["races_Tminus28_observable"] for v in per_day.values()),
                     "complete_by_Tminus28": sum(v["complete_by_Tminus28"] for v in per_day.values()),
                     "floor": FLOORS["complete_by_Tminus28"],
                     "note": "判定には T−28 以前から試行している race が必要。率は件数が揃うまで判定に使わない"},
        "gate_S0A_pass": False,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (OUT / "forward_parity.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("n_days", "horse_rows_in_complete_snapshots", "per_day", "Tminus28",
                                          "remaining_to_floor", "status")}, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
