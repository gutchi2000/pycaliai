# -*- coding: utf-8 -*-
"""
jvlink_trio_odds.py — JV-Link 三連複オッズ collector（32-bit 専用ブリッジ）
==========================================================================
設計書: docs/plans/plan_trio_t10_conservative_portfolio_engine_v2.md §5.1 / §5.2 / §7

役割は **T-10 時点の全三連複組オッズ (0B35 / O5 録) を raw のまま追記専用で残す**
こと。判定・EV・資金配分は一切しない（設計書 §9: 最初のターンは Preflight と
§5.1〜5.2 のみ）。

★必ず 32-bit Python で実行: py -3.12-32 jvlink_trio_odds.py --race 2026081601010801
  (jvlink_odds.py と同型。JV-Link は 32-bit COM)

O5 レコード仕様（2026-08-18 に蓄積系 JVOpen("RACE") の実録で確定）:
  [0:2]   種別 'O5'
  [2:3]   データ区分 (蓄積確定='5')
  [3:11]  データ作成年月日
  [11:27] レースキー = 開催年4 + 月日4 + 場2 + 回次2 + 日次2 + R2  (=16桁 race_id)
  [27:35] 発表月日時分
  [35:37] 登録頭数   [37:39] 出走頭数   [39:40] 発売フラグ
  [40:]   三連複オッズ 816組 × stride15 = 組番(6) + オッズ(6) + 人気順(3)
          オッズは 1/10 倍単位 ("009890" → 989.0倍)。未発売の組は '*'/'-' 埋め、
          未使用スロットは空白（parse_o5 の docstring に実測 4 態）。
  末尾    三連複票数合計 11桁 + CRLF
  → レコード長 40 + 816*15 + 11 + 2 = 12293（実録と一致）

出力（設計書 §0.2 により reports/trio_portfolio_shadow_v2/ の外へは一切書かない）:
  raw/{date}/{rid16}/{ts}_0B35.txt      … 生録（追記専用・上書きしない）
  raw/{date}/{rid16}/{ts}_0B31.txt      … 同時点の単勝生録
  snapshots/{date}/{rid16}_{ts}.json    … パース済み断面（bundle 側の ai_* を同梱）
  manifest.jsonl                        … 全取得の追記専用台帳 (sha256/version/時刻)
  raw_stock/{date}/O5_{rid16}.txt       … 蓄積系 O5（parser 検証専用。T-10 ではない）

⚠ 蓄積系 O5 は **確定オッズ** であり、設計書 §0.2 の「確定払戻を T-10 価格として
   使わない」に従い、価格実験には絶対に使わない。用途はパーサ位置の確定だけ。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).parent
OUT_DIR = BASE / "reports" / "trio_portfolio_shadow_v2"
RAW_DIR = OUT_DIR / "raw"
SNAP_DIR = OUT_DIR / "snapshots"
STOCK_DIR = OUT_DIR / "raw_stock"
MANIFEST = OUT_DIR / "manifest.jsonl"

COLLECTOR_VERSION = "trio-collector-2026-08-18.1"
SPEC_RT = "0B35"          # 速報オッズ(三連複)
SPEC_RT_TAN = "0B31"      # 速報オッズ(単複枠) — blend 用の同時点単勝
REC_ID = "O5"

# O5 レコード固定オフセット（上記仕様）
O5_BODY0 = 40
O5_STRIDE = 15
O5_NMAX = 816             # C(18,3)
O5_HEAD = {"kind": (0, 2), "kubun": (2, 3), "made": (3, 11), "racekey": (11, 27),
           "announce": (27, 35), "toroku": (35, 37), "shusso": (37, 39),
           "hatsubai_flag": (39, 40)}

try:
    SID = (BASE / "data" / "jvlink_sid.txt").read_text(
        encoding="utf-8").strip().splitlines()[0] or "UNKNOWN"
except Exception:
    SID = "UNKNOWN"


# ============================================================
# パーサ（COM 非依存 = 64-bit からも import して単体テストできる）
# ============================================================
def _digits(s):
    s = (s or "").strip()
    return int(s) if s.isdigit() else None


def n_combos(n: int, k: int = 3) -> int:
    r = 1
    for i in range(k):
        r = r * (n - i) // (i + 1)
    return r


def parse_o5(rec: str) -> dict:
    """O5(三連複) 生録 → 構造化。

    スロットの実測 4 態（2026-08-18、蓄積 O5 576 レースで全数確認）:
      1) 組番6桁 + オッズ6桁 + 人気3桁  … 発売中。odds/10 が倍率
      2) 組番6桁 + '*'*9                … 未発売（2,130 スロット）
      3) 組番6桁 + '-'*9                … 未発売（66 スロット。'*' との違いは取消時期
                                          と推定。どちらも「買えない」= 同じ扱い）
      4) 15桁すべて空白                  … 未使用スロット
    スロットは 18頭ぶんの組番順に固定配置され、登録頭数が少ない回は
    **飛び飛びに空白が入る**（先頭から詰まってはいない）。組番キーで引くこと。
    非空白スロット = C(登録頭数,3)、うち発売中 = C(出走頭数,3) が実測で厳密成立。

    返り値:
      race_key   : 16桁 race_id
      kubun      : データ区分, announce: 発表月日時分, hatsubai_flag: 発売フラグ
      toroku/shusso: 登録頭数 / 出走頭数
      odds       : {"a-b-c": 倍率}  (a<b<c、発売中の組のみ)
      ninki      : {"a-b-c": 人気順}
      unpriced   : ['a-b-c', ...]   '*'/'-' 埋め = 取消等で未発売
      unpriced_fill: {filler文字: 件数}  未知 filler の監視用
      zero       : ['a-b-c', ...]   数字だがオッズ 0 = 発売停止
      blank      : 空白スロット数（正常な padding）
      malformed  : 上記いずれでもない異常スロット数（0 でなければパーサ異常）
    """
    h = {k: rec[a:b] for k, (a, b) in O5_HEAD.items()}
    odds, ninki, unpriced, zero = {}, {}, [], []
    fills = Counter()
    blank = malformed = 0
    for k in range(O5_NMAX):
        s = O5_BODY0 + k * O5_STRIDE
        slot = rec[s:s + O5_STRIDE]
        if len(slot) < O5_STRIDE:
            break
        kumi, body = slot[:6], slot[6:]
        if not slot.strip():
            blank += 1
            continue
        if not kumi.isdigit():
            malformed += 1
            continue
        a, b, c = int(kumi[:2]), int(kumi[2:4]), int(kumi[4:6])
        if not (0 < a < b < c):
            malformed += 1
            continue
        key = f"{a}-{b}-{c}"
        if not body[:6].isdigit():
            # '*'/'-' 等の filler = 未発売。未知 filler も「買えない」側に倒す
            # （fail-closed）が、種類は unpriced_fill に残して監視する。
            unpriced.append(key)
            fills[body[:6].strip() or "SPACE"] += 1
            continue
        v, p = _digits(body[:6]), _digits(body[6:9])
        if v is None:
            malformed += 1
            continue
        if v <= 0:
            zero.append(key)
            continue
        odds[key] = round(v / 10.0, 1)
        if p:
            ninki[key] = p
    return {"race_key": h["racekey"], "kubun": h["kubun"],
            "announce": h["announce"], "hatsubai_flag": h["hatsubai_flag"],
            "toroku": _digits(h["toroku"]), "shusso": _digits(h["shusso"]),
            "odds": odds, "ninki": ninki, "unpriced": unpriced, "zero": zero,
            "unpriced_fill": dict(fills), "blank": blank, "malformed": malformed,
            "n_slots": O5_NMAX}


def combo_key(a: int, b: int, c: int) -> str:
    x, y, z = sorted((int(a), int(b), int(c)))
    return f"{x}-{y}-{z}"


# ============================================================
# JV-Link (32-bit COM)
# ============================================================
def fetch_records(race_key: str, spec: str, max_rec: int = 400) -> list[str]:
    """JVRTOpen→JVRead。jvlink_odds.fetch_records と同型（重複を避け import 先行）。"""
    try:
        from jvlink_odds import fetch_records as _f
        return _f(race_key, spec, max_rec=max_rec)
    except Exception:
        pass
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
                break
            if size < 0:
                continue
            recs.append((r[1] if isinstance(r, tuple) else "")[:size])
    finally:
        try:
            jv.JVClose()
        except Exception:
            pass
    return recs


def fetch_stock_o5(from_ts: str, timeout_s: int = 600) -> list[str]:
    """蓄積系 JVOpen("RACE") から O5 録だけを集める（パーサ検証専用）。"""
    import win32com.client as w
    jv = w.Dispatch("JVDTLab.JVLink")
    if jv.JVInit(SID) != 0:
        return []
    out = []
    try:
        r = jv.JVOpen("RACE", from_ts, 1)
        rc = r[0] if isinstance(r, tuple) else r
        dl = r[2] if isinstance(r, tuple) and len(r) > 2 else 0
        if rc != 0:
            return []
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
            if buf.startswith(REC_ID):
                out.append(buf)
    finally:
        try:
            jv.JVClose()
        except Exception:
            pass
    return out


# ============================================================
# 追記専用ストレージ（P9: raw 不変）
# ============================================================
def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def write_append_only(path: Path, text: str) -> tuple[Path, bool]:
    """絶対に上書きしない。既存があれば連番を足した別ファイルにする。
    返り値 (実際に書いたパス, 既存衝突があったか)。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(text, encoding="utf-8", errors="replace")
        return path, False
    for i in range(1, 1000):
        alt = path.with_name(f"{path.stem}__dup{i}{path.suffix}")
        if not alt.exists():
            alt.write_text(text, encoding="utf-8", errors="replace")
            return alt, True
    raise RuntimeError(f"append-only slot exhausted: {path}")


def log_manifest(entry: dict) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ============================================================
# 発走時刻 / bundle 参照（決定時点で存在する情報のみ。P8: 未来遮断）
# ============================================================
def parse_hhmm(s: str):
    s = str(s or "").strip()
    m = re.match(r"^(\d{1,2})[:時](\d{2})", s)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.match(r"^(\d{3,4})$", s)
    if m:
        v = int(m.group(1))
        return v // 100, v % 100
    return None


def load_post_times(date_str: str) -> dict:
    """data/weekly/{date}.csv → {rid16: 'HH:MM'}（t10_runner と同一ソース）。
    発走時刻変更があれば reports/live_changes/{date}.json の time_change で上書き。"""
    out = {}
    p = BASE / "data" / "weekly" / f"{date_str}.csv"
    if p.exists():
        with open(p, encoding="cp932", errors="replace", newline="") as f:
            rd = csv.DictReader(f)
            rid_col = next((c for c in (rd.fieldnames or []) if "レースID" in c), None)
            tm_col = next((c for c in (rd.fieldnames or []) if "発走" in c), None)
            if rid_col and tm_col:
                for row in rd:
                    rid = re.sub(r"\D", "", str(row.get(rid_col, "")))[:16]
                    hm = parse_hhmm(row.get(tm_col, ""))
                    if len(rid) == 16 and hm and rid not in out:
                        out[rid] = f"{hm[0]:02d}:{hm[1]:02d}"
    ch = BASE / "reports" / "live_changes" / f"{date_str}.json"
    if ch.exists():
        try:
            d = json.loads(ch.read_text(encoding="utf-8"))
            for rid, v in (d.get("races") or {}).items():
                tc = (v or {}).get("time_change") or {}
                hm = parse_hhmm(tc.get("new", ""))
                if hm:
                    out[rid] = f"{hm[0]:02d}:{hm[1]:02d}"
        except Exception:
            pass
    return out


def lead_minutes(now: datetime, date_str: str, post_hhmm: str):
    """発走まで何分か。post 不明なら None。"""
    hm = parse_hhmm(post_hhmm)
    if hm is None:
        return None
    d = datetime.strptime(date_str, "%Y%m%d").replace(hour=hm[0], minute=hm[1])
    return (d - now).total_seconds() / 60.0


def announce_dt(announce: str, date_str: str):
    """O5 の発表月日時分 'MMDDHHMM' → datetime。年は開催日から補う。
    RT では実値が入る（例 '08161009' = 8/16 10:09）。蓄積確定は 00000000。"""
    a = (announce or "").strip()
    if len(a) != 8 or not a.isdigit() or a == "0" * 8:
        return None
    try:
        return datetime(int(date_str[:4]), int(a[:2]), int(a[2:4]),
                        int(a[4:6]), int(a[6:8]))
    except ValueError:
        return None


def bundle_race(date_str: str, rid16: str) -> dict | None:
    p = BASE / "reports" / "cowork_input" / f"{date_str}_bundle.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    for r in d.get("races", []):
        if re.sub(r"\D", "", str(r.get("race_id", "")))[:16] == rid16:
            return {"model": d.get("model"), "bundle_date": d.get("date"),
                    "race_meta": r.get("race_meta", {}),
                    "horses": [{"umaban": h.get("umaban"),
                                "ai_rank": h.get("ai_rank"),
                                "ai_score": h.get("ai_score"),
                                "p_win": h.get("p_win"),
                                "mark": h.get("mark")} for h in r.get("horses", [])]}
    return None


# ============================================================
# T-10 断面取得（§5.1 データ契約）
# ============================================================
def collect_race(rid16: str, date_str: str | None = None,
                 window=(8.0, 12.0), subdir: str | None = None) -> dict:
    """1 レースの T-10 断面を取る。subdir を渡すと保存先フォルダ名を差し替える
    （疎通テスト用。Phase 0 の集計に混ざらないようにする）。"""
    now = datetime.now()
    date_str = date_str or rid16[:8]
    folder = subdir or date_str
    ts = now.strftime("%Y%m%dT%H%M%S")
    post = load_post_times(date_str).get(rid16, "")
    lead = lead_minutes(now, date_str, post)

    recs5 = [r for r in fetch_records(rid16, SPEC_RT) if r.startswith(REC_ID)]
    recs1 = [r for r in fetch_records(rid16, SPEC_RT_TAN) if r.startswith("O1")]

    rawdir = RAW_DIR / folder / rid16
    raw_paths, dup = {}, False
    for spec, recs in ((SPEC_RT, recs5), (SPEC_RT_TAN, recs1)):
        if not recs:
            continue
        body = "\n".join(recs)
        p, d = write_append_only(rawdir / f"{ts}_{spec}.txt", body)
        raw_paths[spec] = str(p.relative_to(BASE))
        dup = dup or d
        log_manifest({"kind": "raw", "spec": spec, "race_id": rid16,
                      "fetched": now.isoformat(timespec="seconds"),
                      "path": raw_paths[spec], "sha256": _sha(body),
                      "n_records": len(recs), "collector": COLLECTOR_VERSION,
                      "dup_collision": d})

    snap = {"race_id": rid16, "date": date_str,
            "fetched": now.isoformat(timespec="seconds"),
            "collector_version": COLLECTOR_VERSION,
            "post_time": post or None, "lead_min": None if lead is None else round(lead, 2),
            "lead_window": list(window),
            "in_window": bool(lead is not None and window[0] <= lead <= window[1]),
            "raw": raw_paths, "ok": False, "reason": ""}

    if not recs5:
        snap["reason"] = f"no {REC_ID} record ({SPEC_RT})"
    else:
        o5 = parse_o5(recs5[-1])
        key_ok = o5["race_key"] == rid16
        snap.update({"trio": o5["odds"], "trio_ninki": o5["ninki"],
                     "n_trio": len(o5["odds"]), "unpriced": o5["unpriced"],
                     "zero_priced": o5["zero"], "malformed": o5["malformed"],
                     "blank_slots": o5["blank"],
                     "toroku": o5["toroku"], "shusso": o5["shusso"],
                     "hatsubai_flag": o5["hatsubai_flag"],
                     "announce": o5["announce"], "kubun": o5["kubun"],
                     "race_key_match": key_ok})
        exp = n_combos(o5["shusso"] or 0, 3)
        exp_slots = n_combos(o5["toroku"] or 0, 3)
        snap["expected_combos"] = exp
        snap["expected_slots"] = exp_slots
        snap["combo_count_match"] = bool(exp and len(o5["odds"]) == exp)
        snap["slot_count_match"] = bool(
            exp_slots and len(o5["odds"]) + len(o5["unpriced"]) + len(o5["zero"]) == exp_slots)
        # 価格の「時点」は自分の時計ではなく O5 の発表月日時分で証明する（§5.2）
        adt = announce_dt(o5["announce"], date_str)
        snap["announce_dt"] = adt.isoformat(timespec="minutes") if adt else None
        snap["announce_lead_min"] = (
            None if (adt is None or lead is None)
            else round(lead_minutes(adt, date_str, post) or 0.0, 2))
        snap["announce_in_window"] = bool(
            snap["announce_lead_min"] is not None
            and window[0] <= snap["announce_lead_min"] <= window[1])
        snap["ok"] = bool(key_ok and o5["malformed"] == 0
                          and snap["combo_count_match"] and snap["slot_count_match"])
        if not snap["ok"]:
            snap["reason"] = (f"key_match={key_ok} malformed={o5['malformed']} "
                              f"priced={len(o5['odds'])}/{exp} "
                              f"slots={len(o5['odds'])+len(o5['unpriced'])+len(o5['zero'])}"
                              f"/{exp_slots}")
    if recs1:
        try:
            from jvlink_odds import parse_o1
            snap["tansho"] = parse_o1(recs1[-1])["tansho"]
        except Exception as e:
            snap["tansho_error"] = str(e)[:120]

    b = bundle_race(date_str, rid16)
    if b:
        snap["bundle"] = b
    p, d = write_append_only(SNAP_DIR / folder / f"{rid16}_{ts}.json",
                             json.dumps(snap, ensure_ascii=False, indent=1))
    snap["_snapshot_path"] = str(p.relative_to(BASE))
    log_manifest({"kind": "snapshot", "race_id": rid16,
                  "fetched": snap["fetched"], "path": snap["_snapshot_path"],
                  "ok": snap["ok"], "in_window": snap["in_window"],
                  "lead_min": snap["lead_min"],
                  "announce_dt": snap.get("announce_dt"),
                  "announce_lead_min": snap.get("announce_lead_min"),
                  "n_trio": snap.get("n_trio"),
                  "collector": COLLECTOR_VERSION, "dup_collision": d})
    return snap


# ============================================================
# 蓄積系 O5 の dump（パーサ位置確定 = Preflight P1 の証拠）
# ============================================================
def stock_dump(from_ts: str, tag: str) -> dict:
    recs = fetch_stock_o5(from_ts)
    outdir = STOCK_DIR / tag
    n = 0
    for rec in recs:
        rid = rec[11:27]
        p, _ = write_append_only(outdir / f"O5_{rid}.txt", rec)
        n += 1
        log_manifest({"kind": "stock_o5", "race_id": rid, "path": str(p.relative_to(BASE)),
                      "sha256": _sha(rec), "len": len(rec),
                      "fetched": datetime.now().isoformat(timespec="seconds"),
                      "collector": COLLECTOR_VERSION,
                      "note": "蓄積確定オッズ。パーサ検証専用。価格実験に使用禁止"})
    return {"from": from_ts, "records": n, "dir": str(outdir)}


def build_schedule(date_str: str) -> list:
    """bundle の全レース × 出走表発走時刻 → [(post_dt, rid16)] 昇順。
    t10_runner.build_schedule と同一ソース（data/weekly + 発走時刻変更）。"""
    p = BASE / "reports" / "cowork_input" / f"{date_str}_bundle.json"
    if not p.exists():
        return []
    d = json.loads(p.read_text(encoding="utf-8"))
    post = load_post_times(date_str)
    base = datetime.strptime(date_str, "%Y%m%d")
    out = []
    for r in d.get("races", []):
        rid = re.sub(r"\D", "", str(r.get("race_id", "")))[:16]
        hm = parse_hhmm(post.get(rid, ""))
        if len(rid) == 16 and hm:
            out.append((base.replace(hour=hm[0], minute=hm[1]), rid))
    out.sort()
    return out


def watch(date_str: str, lead_min: float = 10.0, window=(8.0, 12.0)) -> int:
    """開催日に並走させる shadow collector。各レースの T-lead で 1 断面だけ取る。

    本番 t10_runner とは別プロセス・別ロック・別出力。本番には触れない。
    発走時刻変更は毎ループ読み直す（reports/live_changes/{date}.json）。
    """
    lock = OUT_DIR / f".watch_{date_str}.lock"
    if lock.exists() and (time.time() - lock.stat().st_mtime) < 6 * 3600:
        print(f"[watch] 既に起動中らしい ({lock})。二重起動を中止")
        return 1
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(str(datetime.now()), encoding="utf-8")
    done = set()
    try:
        while True:
            sched = build_schedule(date_str)
            if not sched:
                print(f"[watch] {date_str} の bundle / 発走時刻が無い")
                return 1
            now = datetime.now()
            todo = [(pt, rid) for pt, rid in sched if rid not in done]
            if not todo:
                print("[watch] 全レース完了")
                return 0
            for pt, rid in todo:
                lm = (pt - now).total_seconds() / 60.0
                if lm <= window[0]:            # 窓を過ぎた = 取らない (記録も残さない)
                    if lm < window[0]:
                        done.add(rid)
                    continue
                if lm <= lead_min:
                    s = collect_race(rid, date_str, window)
                    done.add(rid)
                    print(f"[watch] {rid} lead={s.get('lead_min')}分 ok={s['ok']} "
                          f"組={s.get('n_trio')}/{s.get('expected_combos')}")
            time.sleep(20)
    finally:
        try:
            lock.unlink()
        except Exception:
            pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--race", help="16桁 race_id → T-10 断面を取得")
    ap.add_argument("--date", help="--race 用の開催日 (既定: race_id 先頭8桁)")
    ap.add_argument("--all-races", help="YYYYMMDD: bundle の全レースを順に取得")
    ap.add_argument("--watch", help="YYYYMMDD: 開催日に並走し全レースの T-10 断面を貯める")
    ap.add_argument("--lead-min", type=float, default=10.0, help="--watch の目標分前 (既定10)")
    ap.add_argument("--smoke", action="store_true",
                    help="疎通テスト。snapshots/_smoke/ に書き Phase 0 集計に混ぜない")
    ap.add_argument("--stock-dump", help="YYYYMMDDHHMMSS: 蓄積系 O5 を dump (parser 検証専用)")
    ap.add_argument("--tag", default="", help="--stock-dump の保存タグ")
    args = ap.parse_args()

    if args.watch:
        return watch(args.watch, args.lead_min)

    if args.stock_dump:
        r = stock_dump(args.stock_dump, args.tag or args.stock_dump[:8])
        print(f"[stock] O5 {r['records']} 録 → {r['dir']}")
        return 0

    rids = []
    if args.race:
        rids = [re.sub(r"\D", "", args.race)[:16]]
    elif args.all_races:
        p = BASE / "reports" / "cowork_input" / f"{args.all_races}_bundle.json"
        d = json.loads(p.read_text(encoding="utf-8"))
        rids = [re.sub(r"\D", "", str(r["race_id"]))[:16] for r in d["races"]]
    else:
        ap.error("--race / --all-races / --stock-dump のいずれかが必要")

    for rid in rids:
        s = collect_race(rid, args.date, subdir="_smoke" if args.smoke else None)
        print(f"[trio] {rid} ok={s['ok']} lead={s.get('lead_min')}分 "
              f"(発表={s.get('announce_dt')} lead={s.get('announce_lead_min')}分) "
              f"in_window={s['in_window']} 組={s.get('n_trio')}/"
              f"{s.get('expected_combos')} 単勝{len(s.get('tansho', {}))}頭 "
              f"{s.get('reason', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
