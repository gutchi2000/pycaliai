# -*- coding: utf-8 -*-
"""
jvlink_obs.py — 観測計画 v2.1 Phase 2 の shadow collector（32-bit 専用、JV-Link 読取のみ）
======================================================================================
本番の買い目・投票設定・latest view (reports/live_odds) には一切書かない。出力は
forward_prices（schema v2、追記専用）と jv_journal（取得ごとの時刻・rc）だけ。

stage:
  t2_candidate          発走 2 分前。0B31/0B32/0B33/0B34/0B35 を同一プロセスで連続取得（§3 E2）。
                        「T−2 決定」とは呼ばない。意思決定可能時点かは Stage 0 後に判定する。
  final_rt_candidate    RT 系 (JVRTOpen) の確定後候補。当日夜に全レースを一巡して raw を残す。
  final_stock_candidate 蓄積系 (JVOpen("RACE")) の O1〜O5。当日夜と翌日夜に取得（翌日は不変性確認）。
                        O1〜O5 以外の録（HR 払戻・SE 着順など）は読み捨て、保存も解析もしない。
`final` の認定（§2.3 の 3 条件）と区分コードの意味付けはここでは行わない（stream 別の候補 raw を残すだけ）。

★必ず 32-bit Python: py -3.12-32 jvlink_obs.py --race 2026100306040901 --stage t2_candidate --scheduled-post 10:00
  --final-rt 20261003          当日夜: 全レースの RT final 候補
  --final-stock 20261003       蓄積系 final 候補（--also-previous で直前の開催日も再取得）
  --dry                        forward_prices_dry / jvlink_fetch_journal_dry に分けて書く
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import jv_journal
import jv_records as JR

BASE = Path(__file__).resolve().parent
SPECS_ALL = ("0B31", "0B32", "0B33", "0B34", "0B35")
STOCK_KINDS = {"O1": "0B31", "O2": "0B32", "O3": "0B33", "O4": "0B34", "O5": "0B35"}


def forward_root(dry: bool) -> Path:
    from forward_prices import FORWARD_ROOT
    return FORWARD_ROOT.parent / "forward_prices_dry" if dry else FORWARD_ROOT


def market_from_captures(rid: str, captures: list[dict], fetched: str) -> dict:
    """本番 jvlink_odds.fetch_race と同じ形の market（+ trio）を、各 spec の最新録から作る。
    parse は本番で実録突合済みの関数をそのまま使う（jv_records の slot parser とは独立）。"""
    from jvlink_odds import parse_o1, parse_o2, parse_o3, parse_o4
    from jvlink_trio_odds import parse_o5
    last = {c["spec"]: (c["records"][-1]["raw"] if c["records"] else None) for c in captures}
    out = {"race_id": rid, "fetched": fetched, "tansho": {}, "fukusho": {}, "umaren": {},
           "wide": {}, "umatan": {}, "trio": {}}
    if last.get("0B31"):
        out.update(parse_o1(last["0B31"]))
    if last.get("0B32"):
        out["umaren"] = parse_o2(last["0B32"])
    if last.get("0B33"):
        out["wide"] = parse_o3(last["0B33"])
    if last.get("0B34"):
        out["umatan"] = parse_o4(last["0B34"])
    if last.get("0B35"):
        out["trio"] = parse_o5(last["0B35"])["odds"]
    tan = out["tansho"]
    over = sum(1.0 / o for o in tan.values() if o > 1.0) if tan else 0.0
    out["overround_tan"] = round(over, 3)
    out["ok"] = bool(tan) and 1.0 <= over <= 1.5 and all(c["ok"] for c in captures)
    out["reason"] = "" if out["ok"] else "; ".join(
        f"{c['spec']}:{'ok' if c['ok'] else 'NG'}" for c in captures)
    return out


def collect_rt(rid: str, stage: str, scheduled_post: str | None = None, *, dry: bool = False,
               fetch=None, root: Path | None = None, stamp: dict | None = None) -> Path:
    """1 レースの 5 spec を同一プロセスで連続取得し、schema v2 で 1 録に保存する。"""
    from forward_prices import archive_market_snapshot
    if fetch is None:
        from jvlink_odds import fetch_records_timed as fetch
    jv_journal.set_context(process="jvlink_obs", stage=stage, race_id=rid, dry=dry)
    captures = []
    for spec in SPECS_ALL:
        recs, meta = fetch(rid, spec)
        captures.append(JR.capture(spec, rid, recs, meta, stream="rt"))
    fetched = captures[0].get("fetch_started_at") or jv_journal.now_iso()
    market = market_from_captures(rid, captures, fetched)
    return archive_market_snapshot(market, stage, scheduled_post=scheduled_post, stamp=stamp,
                                   root=root or forward_root(dry), captures=captures)


def race_schedule(date_str: str) -> list[tuple[str, str]]:
    """[(rid16, 'YYYY-MM-DDTHH:MM:00')]。発走時刻は三連複 collector と同一ソース（weekly + 時刻変更）。"""
    from jvlink_trio_odds import build_schedule
    return [(rid, pt.strftime("%Y-%m-%dT%H:%M:00")) for pt, rid in build_schedule(date_str)]


def final_rt_sweep(date_str: str, *, dry: bool = False, fetch=None, root: Path | None = None,
                   pause_s: float = 1.0) -> dict:
    """当日夜: 全レースの RT final 候補を 1 レースずつ順番に取得（並走させない）。"""
    done, failed = [], []
    for rid, sp in race_schedule(date_str):
        try:
            done.append(str(collect_rt(rid, "final_rt_candidate", sp, dry=dry, fetch=fetch, root=root)))
        except Exception as exc:
            failed.append({"race_id": rid, "error": f"{type(exc).__name__}: {exc}"[:200]})
        time.sleep(pause_s)
    return {"date": date_str, "saved": len(done), "failed": failed}


def stock_route(buf: str, keep) -> str | None:
    """蓄積系の 1 録を、保存する spec か None（読み捨て）に振り分ける。見るのは種別 2 文字と
    レースキーだけ。O1〜O5 以外（HR 払戻・SE 着順・RA など）は常に None。"""
    kind = buf[:2]
    if kind in STOCK_KINDS and keep(buf[11:27]):
        return STOCK_KINDS[kind]
    return None


def read_stock(from_ts: str, keep, timeout_s: int = 1200) -> tuple[dict, dict]:
    """蓄積系 JVOpen("RACE", from_ts, 1)。keep(rid) が True の race の O1〜O5 録だけを残し、
    それ以外の録（HR/SE/RA など）は種別 2 文字だけ見て捨てる（保存も解析もしない）。"""
    import win32com.client as w
    meta = {"fetch_started_at": jv_journal.now_iso(), "stream": "stock", "from_ts": from_ts,
            "rc_init": None, "rc_open": None, "error": None}
    kept: dict[str, dict[str, list[str]]] = {}
    n_read = n_discarded = 0
    jv = w.Dispatch("JVDTLab.JVLink")
    try:
        rc = jv.JVInit(JR_SID())
        meta["rc_init"] = rc
        if rc != 0:
            return kept, meta
        r = jv.JVOpen("RACE", from_ts, 1)
        rc = r[0] if isinstance(r, tuple) else r
        dl = r[2] if isinstance(r, tuple) and len(r) > 2 else 0
        meta["rc_open"], meta["download_count"] = rc, dl
        if rc != 0:
            return kept, meta
        t0 = time.time()
        while time.time() - t0 < timeout_s and jv.JVStatus() < (dl or 0):
            time.sleep(2)
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            r = jv.JVRead(" " * 120000, 120000, " " * 256)
            size = r[0] if isinstance(r, tuple) else r
            if size == 0:
                break
            if size < 0:
                continue
            buf = r[1][:size]
            n_read += 1
            spec = stock_route(buf, keep)
            if spec:
                kept.setdefault(buf[11:27], {}).setdefault(spec, []).append(buf)
            else:
                n_discarded += 1
    except Exception as exc:
        meta["error"] = f"{type(exc).__name__}: {exc}"[:300]
    finally:
        try:
            jv.JVClose()
        except Exception:
            pass
        meta.update({"fetch_finished_at": jv_journal.now_iso(), "n_records_read": n_read,
                     "n_records_discarded_unread": n_discarded,
                     "n_records_returned": sum(len(v) for d in kept.values() for v in d.values())})
        jv_journal.write_event("STOCK", "RACE", meta)
    return kept, meta


def JR_SID() -> str:
    try:
        return (BASE / "data" / "jvlink_sid.txt").read_text(encoding="utf-8").strip().splitlines()[0] or "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def archive_stock(kept: dict, meta: dict, schedule: dict[str, str], *, dry: bool = False,
                  root: Path | None = None, stamp: dict | None = None) -> dict:
    """race ごとに stage=final_stock_candidate で 1 録。spec ごとの全録（版の履歴）を raw で残す。"""
    from forward_prices import archive_market_snapshot
    saved, failed = [], []
    for rid, by_spec in sorted(kept.items()):
        try:
            caps = [JR.capture(spec, rid, by_spec.get(spec, []),
                               {**{k: meta.get(k) for k in ("fetch_started_at", "fetch_finished_at",
                                                            "rc_init", "rc_open", "error", "from_ts")},
                                "stream": "stock"}, stream="stock") for spec in SPECS_ALL]
            market = market_from_captures(rid, caps, meta.get("fetch_finished_at") or jv_journal.now_iso())
            saved.append(str(archive_market_snapshot(market, "final_stock_candidate",
                                                     scheduled_post=schedule.get(rid), stamp=stamp,
                                                     root=root or forward_root(dry), captures=caps)))
        except Exception as exc:
            failed.append({"race_id": rid, "error": f"{type(exc).__name__}: {exc}"[:200]})
    missing = sorted(set(schedule) - set(kept))
    return {"saved": len(saved), "failed": failed, "scheduled_races_without_stock_records": missing}


def previous_race_day(date_str: str, dry: bool = False) -> str | None:
    """直前の開催日（forward store に日付フォルダがある最も近い過去日、14 日以内）。"""
    root = forward_root(dry)
    d0 = datetime.strptime(date_str, "%Y%m%d")
    for k in range(1, 15):
        d = (d0 - timedelta(days=k)).strftime("%Y%m%d")
        if (root / d).is_dir():
            return d
    return None


def final_stock(date_str: str, *, also_previous: bool = False, dry: bool = False) -> dict:
    dates = [date_str]
    prev = previous_race_day(date_str, dry) if also_previous else None
    if prev:
        dates.insert(0, prev)
    schedule = {}
    for d in dates:
        schedule.update(dict(race_schedule(d)))
    jv_journal.set_context(process="jvlink_obs", stage="final_stock_candidate", race_id=None, dry=dry)
    kept, meta = read_stock(min(dates) + "000000", keep=lambda rid: rid in schedule)
    res = archive_stock(kept, meta, schedule, dry=dry)
    return {"dates": dates, "stock_meta": {k: v for k, v in meta.items() if k != "error"},
            "error": meta.get("error"), **res}


def main() -> int:
    ap = argparse.ArgumentParser(description="観測計画 v2.1 Phase 2 shadow collector")
    ap.add_argument("--race", help="16桁 race_id")
    ap.add_argument("--stage", default="t2_candidate", choices=("t2_candidate", "final_rt_candidate"))
    ap.add_argument("--scheduled-post", default=None, help="HH:MM または ISO")
    ap.add_argument("--date", default=None, help="--race の開催日 (既定 race_id 先頭 8 桁)")
    ap.add_argument("--final-rt", help="YYYYMMDD: 全レースの RT final 候補を当日夜に一巡")
    ap.add_argument("--final-stock", help="YYYYMMDD: 蓄積系 O1〜O5 final 候補")
    ap.add_argument("--also-previous", action="store_true",
                    help="--final-stock で直前の開催日も再取得（翌日の不変性確認）")
    ap.add_argument("--list-schedule", help="YYYYMMDD: rid<TAB>HH:MM を出力（読取のみ。obs_schedule.ps1 用）")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    if args.list_schedule:
        for rid, sp in race_schedule(args.list_schedule):
            print(f"{rid}\t{sp[11:16]}")
        return 0
    if args.final_rt:
        print(json.dumps(final_rt_sweep(args.final_rt, dry=args.dry), ensure_ascii=False))
        return 0
    if args.final_stock:
        r = final_stock(args.final_stock, also_previous=args.also_previous, dry=args.dry)
        print(json.dumps({k: v for k, v in r.items() if k != "stock_meta"}, ensure_ascii=False))
        return 0 if not r.get("error") else 1
    if not args.race:
        ap.error("--race / --final-rt / --final-stock のいずれかが必要")
    rid = re.sub(r"\D", "", args.race)[:16]
    sp = args.scheduled_post
    if sp and re.fullmatch(r"\d{1,2}:\d{2}", sp):
        d = args.date or rid[:8]
        sp = f"{d[:4]}-{d[4:6]}-{d[6:8]}T{int(sp.split(':')[0]):02d}:{sp.split(':')[1]}:00"
    path = collect_rt(rid, args.stage, sp, dry=args.dry)
    print(f"[jvlink_obs] {args.stage} race={rid} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
