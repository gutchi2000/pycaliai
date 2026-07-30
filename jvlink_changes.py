# -*- coding: utf-8 -*-
"""
jvlink_changes.py — JV-Link 当日変更情報取得（32-bit 専用ブリッジ）
====================================================================
出走取消・競走除外(AV) / 騎手変更(JC) / 発走時刻変更(TC) / 馬体重(WH) を
JV-Link 速報系から取り、サイト掲載用 JSON に吐く。目的は予測でなく
**サイトの信頼性**（取消馬に◎が付いたまま公開される事故の防止）。

★必ず 32-bit Python: py -3.12-32 jvlink_changes.py 20260801
  (jvlink_odds.py と同型。JV-Link は 32-bit COM)

dataspec は 0B11〜0B16 を総当たりし、返ったレコードの先頭 2 文字
(AV/JC/TC/WH/WE/CC) でバケツ分けする。spec→種別の対応表記憶違いに
依存しない設計（未対応 spec は JVRTOpen が非 0 を返し空リストになるだけ）。

⚠ フィールドオフセットは TENTATIVE（JV-Data 仕様書の記憶ベース）。
   ヘッダ部 (種別2+区分1+作成日8+レースキー16+発表8=35) は O1 実証と同型で、
   rec[11:27] == 要求レースキー の一致を必須にしている＝ヘッダずれは検出可能。
   ボディ部は初の実イベント発生時に --dump-raw の生録と突合して確定させること。
   サニティチェック（馬番 1-28・体重 300-700 等）を通ったものだけ publish する。

出力:
  reports/live_changes/{date}.json   … フル（サニティ落ち含む診断情報つき）
  site/data/changes_{date}.json      … サイト公開用（サニティ通過分のみ）
    {date, fetched, races: {rid16: {cancels:[{umaban,name,kind}],
     jockey_changes:[{umaban, jockey_to, jockey_from}],
     time_change:{new,old}, weights:{umaban:[kg, diff_str]}}}}
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

BASE = Path(__file__).parent
OUT_DIR = BASE / "reports" / "live_changes"
SITE_DIR = BASE / "site" / "data"

# spec 総当たり対象（変更系リアルタイム。オッズ 0B31/33/34 は jvlink_odds 担当）
SPECS = ["0B11", "0B12", "0B13", "0B14", "0B15", "0B16"]
WANT_PREFIX = ("AV", "JC", "TC", "WH")   # WE(天候)/CC(コース変更) は現状パース対象外

try:
    from jvlink_odds import fetch_records, _digits
except Exception:
    def fetch_records(*a, **k): return []
    def _digits(s):
        s = (s or "").strip(); return int(s) if s.isdigit() else None


def _text(s: str) -> str:
    return (s or "").replace("　", " ").strip()


# ---------------- parsers (offsets TENTATIVE) ----------------
# 共通ヘッダ: [0:2]種別 [2:3]データ区分 [3:11]作成年月日 [11:27]レースキー [27:35]発表月日時分

def parse_av(rec: str) -> dict | None:
    """AV(出走取消・競走除外) → {umaban, name, kind}。
    データ区分: '1'=出走取消 '2'=競走除外 (TENTATIVE)。"""
    ban = _digits(rec[35:37])
    if not ban or not (1 <= ban <= 28):
        return None
    name = _text(rec[37:73])
    kind = {"1": "出走取消", "2": "競走除外"}.get(rec[2:3], "取消/除外")
    return {"umaban": ban, "name": name, "kind": kind}


def parse_jc(rec: str) -> dict | None:
    """JC(騎手変更) → {umaban, futan, jockey_to, jockey_from}。
    馬番2 + 馬名36 + 変更後(負担重量3+騎手コード5+騎手名34+見習1) + 変更前(同) TENTATIVE。"""
    ban = _digits(rec[35:37])
    if not ban or not (1 <= ban <= 28):
        return None
    futan = _digits(rec[73:76])          # 570 = 57.0kg
    to_name = _text(rec[81:115])
    from_name = _text(rec[124:158])
    if not to_name:
        return None
    return {"umaban": ban, "name": _text(rec[37:73]),
            "futan": round(futan / 10.0, 1) if futan and 400 <= futan <= 700 else None,
            "jockey_to": to_name, "jockey_from": from_name or None}


def parse_tc(rec: str) -> dict | None:
    """TC(発走時刻変更) → {new:'HH:MM', old:'HH:MM'} TENTATIVE。"""
    def hhmm(s):
        h, m = _digits(s[:2]), _digits(s[2:4])
        if h is None or m is None or not (0 <= h <= 23 and 0 <= m <= 59):
            return None
        return f"{h:02d}:{m:02d}"
    new, old = hhmm(rec[35:39]), hhmm(rec[39:43])
    if not new:
        return None
    return {"new": new, "old": old}


def parse_wh(rec: str) -> dict:
    """WH(馬体重) → {umaban: [kg, '+4'/'-2'/'±0']}。
    pos35 起点 stride45 = 馬番2+馬名36+体重3+増減符号1+増減差3 TENTATIVE。"""
    out = {}
    W0, STRIDE, N = 35, 45, 18
    for i in range(N):
        s = W0 + i * STRIDE
        if s + 45 > len(rec) + 3:
            break
        ban = _digits(rec[s:s + 2])
        kg = _digits(rec[s + 38:s + 41])
        if not ban or not (1 <= ban <= 28) or not kg or not (300 <= kg <= 700):
            continue
        sign, diff = rec[s + 41:s + 42], _digits(rec[s + 42:s + 45])
        dstr = ""
        if diff is not None and sign in ("+", "-", " ", "0"):
            dstr = "±0" if diff == 0 else f"{'-' if sign == '-' else '+'}{diff}"
        out[str(ban)] = [kg, dstr]
    return out


# ---------------- fetch & assemble ----------------

def race_ids_from_bundle(date: str) -> list[str]:
    p = BASE / "reports" / "cowork_input" / f"{date}_bundle.json"
    if not p.exists():
        return []
    import re
    d = json.loads(p.read_text(encoding="utf-8"))
    out = []
    for r in d.get("races", []):
        rid = re.sub(r"\D", "", str(r.get("race_id", "")))[:16]
        if len(rid) == 16:
            out.append(rid)
    return out


def fetch_all(race_ids: list[str], dump_dir: Path | None = None) -> dict:
    """spec × race 総当たり → {rid16: {prefix: [recs]}}。レースキー echo 不一致は捨てる。"""
    buckets: dict[str, dict[str, list[str]]] = {}
    for rid in race_ids:
        for spec in SPECS:
            recs = fetch_records(rid, spec)
            for rec in recs:
                pre = rec[:2]
                if pre not in WANT_PREFIX:
                    continue
                if rec[11:27] != rid:      # ヘッダずれ/別レース混入は不採用
                    continue
                buckets.setdefault(rid, {}).setdefault(pre, []).append(rec)
            if dump_dir and recs:
                pf = dump_dir / f"{rid}_{spec}.txt"
                pf.write_text("\n".join(recs), encoding="utf-8", errors="replace")
    return buckets


def assemble(date: str, buckets: dict) -> tuple[dict, dict]:
    """→ (full, site)。site はサニティ通過分のみ。"""
    from datetime import datetime
    fetched = datetime.now().isoformat(timespec="seconds")
    races_full, races_site = {}, {}
    for rid, by_pre in sorted(buckets.items()):
        entry: dict = {}
        cancels = [c for c in (parse_av(r) for r in by_pre.get("AV", [])) if c]
        # 同一馬の重複録は最後を採用
        cancels = list({c["umaban"]: c for c in cancels}.values())
        if cancels:
            entry["cancels"] = cancels
        jcs = [j for j in (parse_jc(r) for r in by_pre.get("JC", [])) if j]
        jcs = list({j["umaban"]: j for j in jcs}.values())
        if jcs:
            entry["jockey_changes"] = jcs
        tcs = [t for t in (parse_tc(r) for r in by_pre.get("TC", [])) if t]
        if tcs:
            entry["time_change"] = tcs[-1]
        weights = {}
        for r in by_pre.get("WH", []):
            weights.update(parse_wh(r))    # 最新録が上書き
        if weights:
            entry["weights"] = weights
        if entry:
            races_site[rid] = entry
        races_full[rid] = {"n_raw": {k: len(v) for k, v in by_pre.items()}, **entry}
    full = {"date": date, "fetched": fetched, "tentative": True, "races": races_full}
    site = {"date": date, "fetched": fetched, "races": races_site}
    return full, site


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date", help="YYYYMMDD")
    ap.add_argument("--races", help="rid16 カンマ区切り（省略時は bundle から）")
    ap.add_argument("--dump-raw", action="store_true",
                    help="生レコードを reports/live_changes/raw/ に保存（オフセット確定用）")
    args = ap.parse_args()

    rids = args.races.split(",") if args.races else race_ids_from_bundle(args.date)
    if not rids:
        print(f"[jvlink_changes] race_ids なし（bundle 未生成?） date={args.date}")
        sys.exit(1)

    dump_dir = None
    if args.dump_raw:
        dump_dir = OUT_DIR / "raw"
        dump_dir.mkdir(parents=True, exist_ok=True)

    buckets = fetch_all(rids, dump_dir)
    full, site = assemble(args.date, buckets)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{args.date}.json").write_text(
        json.dumps(full, ensure_ascii=False, indent=2), encoding="utf-8")
    SITE_DIR.mkdir(parents=True, exist_ok=True)
    (SITE_DIR / f"changes_{args.date}.json").write_text(
        json.dumps(site, ensure_ascii=False), encoding="utf-8")

    n_can = sum(len(e.get("cancels", [])) for e in site["races"].values())
    n_jc = sum(len(e.get("jockey_changes", [])) for e in site["races"].values())
    n_tc = sum(1 for e in site["races"].values() if "time_change" in e)
    n_wh = sum(len(e.get("weights", {})) for e in site["races"].values())
    print(f"[jvlink_changes] date={args.date} races={len(rids)} "
          f"取消除外{n_can} 騎手変更{n_jc} 時刻変更{n_tc} 馬体重{n_wh}頭")


if __name__ == "__main__":
    main()
