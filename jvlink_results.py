# -*- coding: utf-8 -*-
"""
jvlink_results.py — JV-Link 速報成績/払戻 取得（32-bit 専用ブリッジ）
=====================================================================
当日、発走後のレースの確定着順 + 払戻を JV-Link 速報成績(0B30: RA/SE/HR)から取り、
failsafe.settle が食える result dict を JSON に吐く。64-bit 本体(t10_runner / failsafe)
はこの JSON を読む（JV-Link は 32-bit COM なので bit 跨ぎ。jvlink_odds.py と同型）。

★必ず 32-bit Python: py -3.12-32 jvlink_results.py --race 2026061405030411

出力: reports/live_odds/{race_id}_result.json
  {race_id, fetched, ok, win, plc, sho, tansho, fukusho:{ban:c}, umaren, umatan,
   wide:{"i-j":c}, sanrenpuku, sanrentan}

⚠ パーサ位置は TENTATIVE（暫定）。オッズパーサ(jvlink_odds)と同様、実開催日に
   --dump-raw で 0B30 生録を保存し、当日の data/kekka/{date}.csv（確定配当）と
   突合して SE/HR の固定長オフセットを確定させること。確定までは ok=false で返し
   (failsafe 側は kekka フォールバック)、本番投入は突合後。次開催 2026-06-20/21 が検証機会。
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

BASE = Path(__file__).parent
OUT_DIR = BASE / "reports" / "live_odds"
SPEC = "0B30"        # 速報成績 (RA レース / SE 馬毎 / HR 払戻)
PARSER_VALIDATED = False   # ← dump-raw×kekka 突合で確定したら True に（本番有効化）

# jvlink_odds の COM ブリッジを再利用（DRY・同じ JVInit/JVRTOpen/JVRead）
try:
    from jvlink_odds import fetch_records, _digits
except Exception:                       # 単体テスト時など
    def fetch_records(*a, **k): return []
    def _digits(s):
        s = (s or "").strip(); return int(s) if s.isdigit() else None


def parse_se_finish(recs: list[str]) -> dict:
    """SE(馬毎成績)録 → {確定着順: 馬番}（1,2,3着）。
    ⚠ TENTATIVE: SE 録の 馬番/確定着順 のオフセットは dump-raw 突合で要確定。"""
    finish = {}
    for r in recs:
        if not r.startswith("SE"):
            continue
        # TENTATIVE offsets — 要 dump-raw 検証
        ban = _digits(r[28:30])
        chaku = _digits(r[269:271]) if len(r) > 271 else None
        if ban and chaku in (1, 2, 3):
            finish[chaku] = ban
    return finish


def parse_hr_payouts(recs: list[str]) -> dict:
    """HR(払戻)録 → 各券種の配当(/100円)。
    ⚠ TENTATIVE: HR は単勝3/複勝5/馬連/ワイド7/馬単/三連複/三連単 の連続スロット。
    オフセットは dump-raw 突合で要確定。確定まで空 dict を返す。"""
    # 確定までは未パース（PARSER_VALIDATED=False の間は使わない）
    return {}


def fetch_result(race_key: str) -> dict:
    from datetime import datetime
    fetched = datetime.now().isoformat(timespec="seconds")
    recs = fetch_records(race_key, SPEC, max_rec=400)
    heads = sorted({r[:2] for r in recs}) if recs else []
    if not recs:
        return {"race_id": race_key, "fetched": fetched, "ok": False,
                "reason": "no 0B30 record", "heads": heads}
    finish = parse_se_finish(recs)
    pay = parse_hr_payouts(recs)
    win, plc, sho = finish.get(1), finish.get(2), finish.get(3)
    ok = bool(PARSER_VALIDATED and win and plc and sho)
    return {"race_id": race_key, "fetched": fetched, "ok": ok,
            "reason": "" if ok else ("parser未確定(PARSER_VALIDATED=False)" if win else "着順パース失敗"),
            "win": win, "plc": plc, "sho": sho, "heads": heads,
            "tansho": pay.get("tansho"), "fukusho": pay.get("fukusho", {}),
            "umaren": pay.get("umaren"), "umatan": pay.get("umatan"),
            "wide": pay.get("wide", {}), "sanrenpuku": pay.get("sanrenpuku"),
            "sanrentan": pay.get("sanrentan")}


def dump_raw(race_key: str) -> None:
    """0B30 生録を reports/live_odds/raw/ に保存（SE/HR パーサ確定用）。
    開催日に1レース実行 → 当日 kekka の確定配当と突合してオフセットを決める。"""
    raw_dir = OUT_DIR / "raw"; raw_dir.mkdir(parents=True, exist_ok=True)
    recs = fetch_records(race_key, SPEC, max_rec=400)
    p = raw_dir / f"{race_key}_{SPEC}.txt"
    p.write_text("\n".join(recs), encoding="utf-8", errors="replace")
    heads = sorted({r[:2] for r in recs}) if recs else []
    print(f"[dump] {SPEC}: {len(recs)} recs (種別 {heads}) → {p.name}")


def main():
    sys.stdout = sys.stdout if hasattr(sys.stdout, "buffer") else sys.stdout
    ap = argparse.ArgumentParser()
    ap.add_argument("--race", required=True, help="16桁 race_key")
    ap.add_argument("--dump-raw", action="store_true",
                    help="0B30 生録を保存（開催日に実行→kekka突合でSE/HRパーサ確定）")
    args = ap.parse_args()
    if args.dump_raw:
        dump_raw(args.race)
    res = fetch_result(args.race)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{args.race}_result.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[jvlink_results] race={args.race} ok={res['ok']} "
          f"着順 {res.get('win')}-{res.get('plc')}-{res.get('sho')} "
          f"種別{res.get('heads')} {res.get('reason','')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
