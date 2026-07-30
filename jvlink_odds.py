# -*- coding: utf-8 -*-
"""
jvlink_odds.py — JV-Link 速報オッズ取得（32-bit 専用ブリッジ）
================================================================
当日 T-10 にレースの生オッズを JV-Link から取り、JSON に吐く。64-bit 本体
(compute_bets.py) はこの JSON を読む（JV-Link は 32-bit COM なので bit 跨ぎ）。

★必ず 32-bit Python で実行: py -3.12-32 jvlink_odds.py --race 2026060705030211
  64-bit では COM load 不可（-2147221021）。

実証済（2026-06-09 単勝 / 2026-06-12 複勝・ワイド・馬単 raw 突合で確定）:
  JVInit("UNKNOWN")=0 / JVRTOpen("0B31",raceKey)=0 / JVRead→(size, O1録)
  O1 単勝: pos45 起点 stride8、odds=int(rec[s:s+4])/10、人気=+4(2)。bundle と全頭一致。
  O1 複勝: pos269 起点 stride12 = lo(4)+hi(4)+人気等(4)。bundle と全頭一致。
    ※旧 stride10 は 5頭ごとに1スロットずれる誤パーサだった (2026-06-12 修正)。
  O3 ワイド: pos40 起点 stride17 = 組番(4)+lo(5)+hi(5)+人気(3)、/10。153組+票数計11。
  O4 馬単  : pos40 起点 stride13 = 組番(4)+odds(6)+人気(3)、/10。306組+票数計11。

データ仕様(RT): 0B31 単複枠 / 0B33 ワイド / 0B34 馬単 （馬連=不使用）

出力: reports/live_odds/{race_id}.json
  {race_id, fetched, ok, tansho:{ban:odds}, fukusho:{ban:[low,high]},
   wide:{"i-j":[low,high]}, umatan:{"i>j":odds}, overround_tan}
fail-safe: overround(単勝 Σ1/odds) が [1.0,1.5] 外 / 録なし → ok=false（compute_bets 側で見送り）
           ワイド/馬単は取れなくても ok 判定に影響しない（compute_bets が推定にフォールバック）
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

BASE = Path(__file__).parent
OUT_DIR = BASE / "reports" / "live_odds"
# sid: data/jvlink_sid.txt があればその 1 行目を使う（2026-07-30〜: サイトに派生指標
# =T-15補正印を公開するため sid 登録方針に転換。登録が下りたら sid をこのファイルに
# 置くだけで有効化。Sid 形式: 作者ID/ソフトウェアID/ソフト名/Ver
# — developer.jra-van.jp/t/topic/30）。無ければ従来どおり "UNKNOWN"（個人利用扱い）。
try:
    SID = (Path(__file__).parent / "data" / "jvlink_sid.txt").read_text(
        encoding="utf-8").strip().splitlines()[0] or "UNKNOWN"
except Exception:
    SID = "UNKNOWN"


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
    単勝: pos45 起点 stride8 = odds(4)+人気(2)+予備(2)、/10（実証済 2026-06-09）。
    複勝: pos269 起点 stride12 = lo(4)+hi(4)+人気等(4)、/10（raw 突合で確定 2026-06-12。
          旧 stride10 は 5頭ごとに1スロットずれて別馬のオッズを返す致命バグだった）。"""
    tansho, fukusho = {}, {}
    TAN0, TAN_STRIDE, N = 45, 8, 28
    for i in range(1, N + 1):
        s = TAN0 + (i - 1) * TAN_STRIDE
        v = _digits(rec[s:s + 4])
        if v and v > 0:
            tansho[i] = round(v / 10.0, 1)
    FUK0, FUK_STRIDE = TAN0 + N * TAN_STRIDE, 12   # =269
    for i in range(1, N + 1):
        s = FUK0 + (i - 1) * FUK_STRIDE
        lo, hi = _digits(rec[s:s + 4]), _digits(rec[s + 4:s + 8])
        if lo and hi and 0 < lo <= hi:
            fukusho[i] = [round(lo / 10.0, 1), round(hi / 10.0, 1)]
    return {"tansho": tansho, "fukusho": fukusho}


def parse_o3(rec: str) -> dict:
    """O3(ワイド) → {"i-j":[low,high]}。
    pos40 起点 stride17 = 組番(4)+lo(5)+hi(5)+人気(3)、/10（raw 突合で確定 2026-06-12）。"""
    out = {}
    W0, STRIDE, NMAX = 40, 17, 153   # C(18,2)
    for k in range(NMAX):
        s = W0 + k * STRIDE
        kumi = rec[s:s + 4]
        if not kumi.strip() or not kumi.isdigit():
            continue
        i, j = int(kumi[:2]), int(kumi[2:])
        lo, hi = _digits(rec[s + 4:s + 9]), _digits(rec[s + 9:s + 14])
        if i and j and lo and hi and 0 < lo <= hi:
            out[f"{i}-{j}"] = [round(lo / 10.0, 1), round(hi / 10.0, 1)]
    return out


def parse_o4(rec: str) -> dict:
    """O4(馬単) → {"i>j":odds} (i=1着, j=2着)。
    pos40 起点 stride13 = 組番(4)+odds(6)+人気(3)、/10（raw 突合で確定 2026-06-12）。"""
    out = {}
    U0, STRIDE, NMAX = 40, 13, 306   # 18P2
    for k in range(NMAX):
        s = U0 + k * STRIDE
        kumi = rec[s:s + 4]
        if not kumi.strip() or not kumi.isdigit():
            continue
        i, j = int(kumi[:2]), int(kumi[2:])
        v = _digits(rec[s + 4:s + 10])
        if i and j and v and v > 0:
            out[f"{i}>{j}"] = round(v / 10.0, 1)
    return out


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
    # ワイド/馬単は取れなくても ok 判定に影響しない (compute_bets が推定にフォールバック)
    wide, umatan = {}, {}
    try:
        r3 = [r for r in fetch_records(race_key, "0B33") if r.startswith("O3")]
        if r3:
            wide = parse_o3(r3[-1])
        r4 = [r for r in fetch_records(race_key, "0B34") if r.startswith("O4")]
        if r4:
            umatan = parse_o4(r4[-1])
    except Exception:
        pass
    # fetched は compute_bets の鮮度 fail-safe (--max-age-min) が参照する
    return {"race_id": race_key, "fetched": fetched,
            "ok": ok, "overround_tan": round(over, 3),
            "reason": "" if ok else f"overround {over:.3f} 異常 or 単勝空",
            **o1, "wide": wide, "umatan": umatan}


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
          f"単勝{len(res.get('tansho',{}))}頭 複勝{len(res.get('fukusho',{}))}頭 "
          f"ワイド{len(res.get('wide',{}))}組 馬単{len(res.get('umatan',{}))}組 "
          f"overround={res.get('overround_tan')} {res.get('reason','')}")
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
