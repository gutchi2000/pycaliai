# -*- coding: utf-8 -*-
"""
jvlink_odds.py — JV-Link 速報オッズ取得（32-bit 専用ブリッジ）
================================================================
当日 T-10 にレースの生オッズを JV-Link から取り、JSON に吐く。64-bit 本体
(compute_bets.py) はこの JSON を読む（JV-Link は 32-bit COM なので bit 跨ぎ）。

★必ず 32-bit Python で実行: py -3.12-32 jvlink_odds.py --race 2026060705030211
  64-bit では COM load 不可（-2147221021）。

実証済（2026-06-09）:
  JVInit("UNKNOWN")=0 / JVRTOpen("0B31",raceKey)=0 / JVRead→(size, O1録)
  O1 単勝: pos45 起点 stride8 bytes/頭、odds=int(rec[s:s+4])/10。人気馬で bundle と一致確認。

データ仕様(RT):
  0B31 単複枠 / 0B33 ワイド / 0B34 馬単 （馬連=不使用）
  ※ 単勝パース確定。複勝/馬単/ワイドは土曜ライブで最終検証（暫定パーサ + フォールバック）。

出力: reports/live_odds/{race_id}.json
  {race_id, fetched, ok, tansho:{ban:odds}, fukusho:{ban:[low,high]}, overround_tan}
fail-safe: overround(単勝 Σ1/odds) が [1.0,1.5] 外 / 録なし → ok=false（compute_bets 側で見送り）
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

BASE = Path(__file__).parent
OUT_DIR = BASE / "reports" / "live_odds"
SID = "UNKNOWN"  # ※継続運用は JRA-VAN に sid 登録推奨


def _digits(s):
    s = (s or "").strip()
    return int(s) if s.isdigit() else None


def fetch_records(race_key: str, spec: str, max_rec: int = 200) -> list[str]:
    """JVRTOpen→JVRead で spec の録を集める。32-bit COM。"""
    import win32com.client as w
    jv = w.Dispatch("JVDTLab.JVLink")
    if jv.JVInit(SID) != 0:
        return []
    recs = []
    try:
        if jv.JVRTOpen(spec, race_key) != 0:
            return []
        for _ in range(max_rec):
            r = jv.JVRead(" " * 120000, 120000, " " * 256)
            size = r[0] if isinstance(r, tuple) else r
            if size == 0:
                break          # 全読込完了
            if size < 0:
                continue       # ファイル境界
            buff = r[1] if isinstance(r, tuple) else ""
            recs.append(buff[:size])
    finally:
        try: jv.JVClose()
        except Exception: pass
    return recs


def parse_o1(rec: str) -> dict:
    """O1(単複枠) → {tansho:{ban:odds}, fukusho:{ban:[low,high]}}。
    単勝: pos45 起点 stride8、odds=int(rec[s:s+4])/10（実証済）。
    複勝: 単勝28頭×8=224 後の pos269 起点 stride10、low/high=int(/10)（暫定・土曜検証）。"""
    tansho, fukusho = {}, {}
    TAN0, TAN_STRIDE, N = 45, 8, 28
    for i in range(1, N + 1):
        s = TAN0 + (i - 1) * TAN_STRIDE
        v = _digits(rec[s:s + 4])
        if v and v > 0:
            tansho[i] = round(v / 10.0, 1)
    FUK0, FUK_STRIDE = TAN0 + N * TAN_STRIDE, 10   # =269（暫定）
    for i in range(1, N + 1):
        s = FUK0 + (i - 1) * FUK_STRIDE
        lo, hi = _digits(rec[s:s + 4]), _digits(rec[s + 4:s + 8])
        if lo and hi and 0 < lo <= hi:
            fukusho[i] = [round(lo / 10.0, 1), round(hi / 10.0, 1)]
    return {"tansho": tansho, "fukusho": fukusho}


def fetch_race(race_key: str) -> dict:
    from datetime import datetime
    fetched = datetime.now().isoformat(timespec="seconds")
    recs = [r for r in fetch_records(race_key, "0B31") if r.startswith("O1")]
    if not recs:
        return {"race_id": race_key, "fetched": fetched,
                "ok": False, "reason": "no O1 record"}
    o1 = parse_o1(recs[-1])   # 最新スナップ
    tan = o1["tansho"]
    over = sum(1.0 / o for o in tan.values() if o > 1.0) if tan else 0.0
    ok = bool(tan) and (1.0 <= over <= 1.5)
    # fetched は compute_bets の鮮度 fail-safe (--max-age-min) が参照する
    return {"race_id": race_key, "fetched": fetched,
            "ok": ok, "overround_tan": round(over, 3),
            "reason": "" if ok else f"overround {over:.3f} 異常 or 単勝空",
            **o1}


def dump_raw(race_key: str) -> None:
    """0B31/0B33/0B34 の生レコードを reports/live_odds/raw/ に保存する。

    用途: ワイド(0B33)/馬単(0B34) のパーサ位置確定 (audit 2026-06-11 残課題)。
    土曜のレース時間帯に 1 回実行して raw を残せば、確定オッズと突合して
    オフラインでパーサを書ける。複勝 pos269 暫定の最終検証にも使う。
    """
    raw_dir = OUT_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for spec in ("0B31", "0B33", "0B34"):
        recs = fetch_records(race_key, spec)
        p = raw_dir / f"{race_key}_{spec}.txt"
        p.write_text("\n".join(recs), encoding="utf-8", errors="replace")
        heads = sorted({r[:2] for r in recs}) if recs else []
        print(f"[dump] {spec}: {len(recs)} recs (種別 {heads}) → {p.name}")


def main():
    sys.stdout = sys.stdout if hasattr(sys.stdout, "buffer") else sys.stdout
    ap = argparse.ArgumentParser()
    ap.add_argument("--race", required=True, help="16桁 race_key（例 2026060705030211）")
    ap.add_argument("--validate", help="bundle.json パスを渡すと単勝を照合表示")
    ap.add_argument("--dump-raw", action="store_true",
                    help="0B31/0B33/0B34 の生レコードを reports/live_odds/raw/ に保存"
                         " (ワイド/馬単パーサ確定用。土曜に1回でOK)")
    args = ap.parse_args()
    if args.dump_raw:
        dump_raw(args.race)
    res = fetch_race(args.race)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{args.race}.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[jvlink_odds] race={args.race} ok={res['ok']} "
          f"単勝{len(res.get('tansho',{}))}頭 overround={res.get('overround_tan')} "
          f"{res.get('reason','')}")
    if args.validate:
        d = json.load(open(args.validate, encoding="utf-8"))
        races = d.get("races", [])
        rr = next((r for r in races if re.sub(r"\D", "", str(r.get("race_id", "")))[:16] == args.race), None)
        if rr:
            bun = {int(h["umaban"]): h.get("tansho_odds") for h in rr["horses"]}
            jv = res.get("tansho", {})
            print("  ban: JV確定 / bundleスナップ")
            for i in sorted(bun):
                print(f"   {i:2d}: {jv.get(i,'-'):>6} / {bun.get(i)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
