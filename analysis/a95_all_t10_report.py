# -*- coding: utf-8 -*-
"""
a95_all_t10_report.py — T-10 オッズで決めた a95 を「本線(参戦ガード通過)」「ガード外(検証用)」
「全レース」に分けて決済・集計する。

入力は reports/cowork_output/{date}_bets.json の各レースの a95_all (compute_bets が T-10 で記録。
guard_passed=false のレースは買い目ではなく「買っていたら」の併記)。実弾ゼロ、記述のみ。
紙上台帳 shadow_a95.py (bundle 時点のオッズで全レース) とは帯の決め方だけが違う。

実行: python -m analysis.a95_all_t10_report [--since 20261003]
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
import numpy as np
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
import shadow_a95 as m  # noqa


def collect(since: str):
    wide = m.load_wide()
    rows = []
    for p in sorted((BASE / "reports" / "cowork_output").glob("2026????_bets.json")):
        date = p.name[:8]
        if date < since:
            continue
        kek = m.load_kekka(date, wide)
        if not kek:
            continue
        raw = json.loads(p.read_text(encoding="utf-8"))
        for e in (raw["bets"] if isinstance(raw, dict) and "bets" in raw else raw):
            aa = e.get("a95_all") or {}
            if not aa.get("tickets"):
                continue
            rid = re.sub(r"\D", "", str(e.get("race_id", "")))[:16]
            k = kek.get(rid)
            if not k or not k.get("valid"):
                continue
            d = {x: aa[x] for x in ("a1", "a2", "a3", "waku1", "waku2")}
            per = {t["kind"]: m.payout_per_100(t["kind"], d, k) for t in aa["tickets"]}
            if any(not np.isfinite(v) for v in per.values()):
                continue          # ワイド払戻未取込など → 集計外
            stake = sum(int(t["stake"]) for t in aa["tickets"])
            ret = sum(per[t["kind"]] / 100.0 * int(t["stake"]) for t in aa["tickets"])
            rows.append(dict(date=date, rid=rid, guard=bool(aa.get("guard_passed")),
                             band=aa.get("band", ""), stake=stake, ret=ret,
                             hit=any(v > 0 for v in per.values())))
    return rows


def _ci(rows, reps=3000):
    days = sorted({r["date"] for r in rows})
    if len(days) < 2:
        return float("nan"), float("nan")
    c = np.array([sum(r["stake"] for r in rows if r["date"] == d) for d in days], float)
    v = np.array([sum(r["ret"] for r in rows if r["date"] == d) for d in days], float)
    idx = np.random.default_rng(0).integers(0, len(days), size=(reps, len(days)))
    roi = 100 * v[idx].sum(1) / c[idx].sum(1)
    return float(np.percentile(roi, 2.5)), float(np.percentile(roi, 97.5))


def line(name, rows):
    if not rows:
        print(f"  {name:<22} (記録なし)"); return
    st = sum(r["stake"] for r in rows); rt = sum(r["ret"] for r in rows); lo, hi = _ci(rows)
    print(f"  {name:<22} R={len(rows):>5} 日={len({r['date'] for r in rows}):>3} | 投資 ¥{st:>11,} 収支 ¥{rt-st:>+12,.0f} "
          f"ROI {100*rt/st:6.1f}% [{lo:5.1f},{hi:5.1f}] (記述) | R的中 {100*np.mean([r['hit'] for r in rows]):4.1f}% "
          f"収支+R {100*np.mean([r['ret'] >= r['stake'] for r in rows]):4.1f}%")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="20261003")
    a = ap.parse_args()
    rows = collect(a.since)
    print(f"a95 (T-10 オッズで決定) 起算 {a.since}〜  ※実弾ゼロ・記述のみ。500R まで ROI で判断しない")
    line("全レース", rows)
    line("本線 (ガード通過)", [r for r in rows if r["guard"]])
    line("ガード外 (検証用)", [r for r in rows if not r["guard"]])
    return 0


if __name__ == "__main__":
    sys.exit(main())
