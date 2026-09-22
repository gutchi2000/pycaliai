# -*- coding: utf-8 -*-
"""
jump_history_collector.py — 障害レース 履歴専用 collector (P1 forward-only)
===========================================================================
`data/bunseki/{date}.csv` (出走馬カード) と `data/kekka/{date}.csv` (結果) から、
**障害レースだけ**を履歴専用の append-only artifact へ収集する。

★このデータは現行 v6 の特徴計算・予測・印・買い目に **接続しない**。
  全レコードが history_only=true / prediction_eligible=false /
  bet_eligible=false / task_registration_eligible=false を持つ。

二層構造:
  raw_card layer   … 判断前情報 (bunseki 由来)。結果で上書きしない
  settled layer    … 結果確定後の情報 (kekka 由来)。別ファイルへ append

出力 (production が読まない専用 namespace):
  data/history_only/jump/raw_card/{date}.jsonl
  data/history_only/jump/settled/{date}.jsonl
  data/history_only/jump/manifest.json

実行:
  python -m analysis.jump_history_only.jump_history_collector --date 20260905
  python -m analysis.jump_history_only.jump_history_collector --all
  python -m analysis.jump_history_only.jump_history_collector --all --dry
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = Path(__file__).resolve().parents[2]
BUNSEKI = BASE / "data" / "bunseki"
KEKKA = BASE / "data" / "kekka"

STORE = BASE / "data" / "history_only" / "jump"
RAW_DIR = STORE / "raw_card"
SETTLED_DIR = STORE / "settled"
MANIFEST = STORE / "manifest.json"

SCHEMA_RAW = "jump-history-only/raw_card/1"
SCHEMA_SETTLED = "jump-history-only/settled/1"

# JRA-VAN トラックコード 51-59 = 障害 (確定判定)
JUMP_TRACK_MIN, JUMP_TRACK_MAX = 51, 59

PEDIGREE_COLS = ["種牡馬", "父タイプ名", "母名", "母父名", "母父タイプ名",
                 "父母父名", "父母父タイプ名", "母母父名", "母母父タイプ名", "生年"]

# 場コード → 場名 (race_id の 9-10 桁目)
VENUE = {"01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
         "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"}


class CollisionError(RuntimeError):
    """同一キーで内容が変わった。silent overwrite せず停止する。"""


def log(m):
    print(m, flush=True)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def content_hash(rec: dict, exclude: tuple[str, ...]) -> str:
    """レコード内容のハッシュ。captured_at 等の揮発項目は除外する。"""
    body = {k: v for k, v in sorted(rec.items()) if k not in exclude}
    return hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    """同一ディレクトリの一時ファイルへ書いてから os.replace で差し替える。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def append_jsonl(path: Path, new_recs: list[dict], key_fields: tuple[str, ...],
                 volatile: tuple[str, ...], dry: bool) -> dict:
    """append-only 書込。
      - 同一キー & 同一内容  → idempotent に skip
      - 同一キー & 内容相違  → CollisionError で停止 (silent overwrite しない)
      - 新規キー             → 追記
    """
    existing = read_jsonl(path)
    by_key = {tuple(r[k] for k in key_fields): r for r in existing}
    added, skipped, collisions = [], 0, []

    for rec in new_recs:
        k = tuple(rec[kf] for kf in key_fields)
        prev = by_key.get(k)
        if prev is None:
            added.append(rec)
            by_key[k] = rec
            continue
        if content_hash(prev, volatile) == content_hash(rec, volatile):
            skipped += 1
            continue
        collisions.append({
            "key": list(k),
            "existing_hash": content_hash(prev, volatile),
            "incoming_hash": content_hash(rec, volatile),
        })

    if collisions:
        raise CollisionError(
            f"{path.name}: {len(collisions)} 件のキー衝突 (内容が変化)。"
            f"append-only のため停止する。例: {collisions[:2]}")

    if added and not dry:
        text = "".join(json.dumps(r, ensure_ascii=False) + "\n"
                       for r in existing + added)
        atomic_write_text(path, text)

    return {"added": len(added), "skipped_idempotent": skipped,
            "total_after": len(existing) + len(added)}


# ------------------------------------------------------------------
# raw card layer
# ------------------------------------------------------------------

def _norm_date(s: str) -> int | None:
    m = re.match(r"^(\d{4})\.(\d{1,2})\.(\d{1,2})$", str(s).strip())
    if m:
        return int(f"{m.group(1)}{int(m.group(2)):02d}{int(m.group(3)):02d}")
    d = re.sub(r"\D", "", str(s))
    return int(d[:8]) if len(d) >= 8 else None


def build_raw_card(date: int) -> tuple[list[dict], dict]:
    p = BUNSEKI / f"{date}.csv"
    if not p.exists():
        return [], {"reason": "bunseki なし"}
    src_sha = sha256_file(p)
    df = pd.read_csv(p, encoding="cp932", dtype=str, on_bad_lines="skip")

    df["rid16"] = df["レースID(新)"].astype(str).str.strip().str[:16]
    df["track_code"] = pd.to_numeric(df.get("トラックコード(JV)"), errors="coerce")
    jump = df[df["track_code"].between(JUMP_TRACK_MIN, JUMP_TRACK_MAX,
                                       inclusive="both")].copy()
    if jump.empty:
        return [], {"reason": "障害レースなし", "source_sha256": src_sha}

    captured = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    recs = []
    for _, r in jump.iterrows():
        rid = str(r["rid16"])
        ped = str(r.get("血統登録番号", "")).strip()
        rec = {
            "schema_version": SCHEMA_RAW,
            # --- hard invariant フラグ ---
            "history_only": True,
            "prediction_eligible": False,
            "bet_eligible": False,
            "task_registration_eligible": False,
            # --- キー ---
            "race_id": rid,
            "ped_id": ped,
            "umaban": int(float(r["馬番"])) if pd.notna(r.get("馬番")) else None,
            # --- レース属性 ---
            "race_date": _norm_date(r.get("日付S")),
            "scheduled_post": str(r.get("発走時刻", "")).strip() or None,
            "venue": VENUE.get(rid[8:10], rid[8:10]),
            "race_number": int(rid[14:16]),
            "track_code": int(r["track_code"]),
            "jump_flag": True,
            "distance": (int(float(r["距離"]))
                         if pd.notna(r.get("距離")) and str(r["距離"]).strip() else None),
            # --- 主体コード ---
            "jockey_code": str(r.get("騎手コード", "")).strip() or None,
            "trainer_code": str(r.get("調教師コード", "")).strip() or None,
            # --- 血統 ---
            "pedigree": {c: (str(r[c]).strip() if c in jump.columns
                             and pd.notna(r.get(c)) else None)
                         for c in PEDIGREE_COLS},
            # --- provenance ---
            "captured_at": captured,
            "source_file": str(p.relative_to(BASE)).replace("\\", "/"),
            "source_sha256": src_sha,
        }
        recs.append(rec)
    return recs, {"source_sha256": src_sha,
                  "races": int(jump["rid16"].nunique()), "rows": len(recs)}


# ------------------------------------------------------------------
# settled layer
# ------------------------------------------------------------------

def build_settled(date: int, raw_recs: list[dict]) -> tuple[list[dict], dict]:
    """結果が利用可能になった後にのみ生成する。raw card は一切書き換えない。"""
    p = KEKKA / f"{date}.csv"
    if not p.exists():
        return [], {"reason": "kekka なし (結果未確定 → settled へ入れない)"}
    src_sha = sha256_file(p)
    k = pd.read_csv(p, encoding="cp932", dtype=str)
    k["rid16"] = k["レースID(新)"].astype(str).str.strip().str[:16]
    k["umaban"] = pd.to_numeric(k["馬番"], errors="coerce")
    k["finish_raw"] = k["確定着順"].astype(str).str.strip()
    k["finish"] = pd.to_numeric(k["確定着順"], errors="coerce")

    result_avail = datetime.fromtimestamp(
        p.stat().st_mtime, timezone.utc).astimezone().isoformat(timespec="seconds")
    settled_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    kmap = {(str(r.rid16), int(r.umaban)): r
            for r in k.itertuples(index=False) if pd.notna(r.umaban)}

    recs = []
    for raw in raw_recs:
        key = (raw["race_id"], raw["umaban"])
        kr = kmap.get(key)
        if kr is None:
            # card にあり結果に無い = 取消候補 (started=False)
            rec_core = {"finish_code_raw": None, "started": False,
                        "completed": False, "dnf": False, "scratched": True}
        else:
            fin = kr.finish
            if pd.notna(fin) and fin > 0:
                rec_core = {"finish_code_raw": kr.finish_raw, "started": True,
                            "completed": True, "dnf": False, "scratched": False}
            else:
                # 止(DNF) と 除外 は kekka 単独で分離できない (既確認)
                rec_core = {"finish_code_raw": kr.finish_raw, "started": True,
                            "completed": False, "dnf": None, "scratched": None}
        rec = {
            "schema_version": SCHEMA_SETTLED,
            "history_only": True,
            "prediction_eligible": False,
            "bet_eligible": False,
            "task_registration_eligible": False,
            "race_id": raw["race_id"],
            "ped_id": raw["ped_id"],
            "umaban": raw["umaban"],
            "race_date": raw["race_date"],
            **rec_core,
            "dnf_jogai_separable": False,
            "result_available_at": result_avail,
            "settled_at": settled_at,
            "result_source_file": str(p.relative_to(BASE)).replace("\\", "/"),
            "result_source_sha256": src_sha,
        }
        recs.append(rec)
    return recs, {"result_source_sha256": src_sha, "rows": len(recs)}


# ------------------------------------------------------------------

def collect(date: int, dry: bool) -> dict:
    out = {"date": date}
    raw, rmeta = build_raw_card(date)
    out["raw_meta"] = rmeta
    if not raw:
        return out
    out["raw_write"] = append_jsonl(
        RAW_DIR / f"{date}.jsonl", raw,
        key_fields=("race_id", "ped_id"),
        volatile=("captured_at",), dry=dry)

    settled, smeta = build_settled(date, raw)
    out["settled_meta"] = smeta
    if settled:
        out["settled_write"] = append_jsonl(
            SETTLED_DIR / f"{date}.jsonl", settled,
            key_fields=("race_id", "ped_id"),
            volatile=("settled_at",), dry=dry)
    return out


def update_manifest(results: list[dict], dry: bool) -> dict:
    prev = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    # 実際にストアへ存在する日付を正とする (単発実行でも全体を保つ)
    on_disk = {int(p.stem) for p in RAW_DIR.glob("*.jsonl") if p.stem.isdigit()}
    on_disk |= {r["date"] for r in results if r.get("raw_write")}
    dates = sorted(on_disk)
    cov_start = str(dates[0]) if dates else prev.get("jump_history_coverage_start")
    man = {
        "schema_version_raw": SCHEMA_RAW,
        "schema_version_settled": SCHEMA_SETTLED,
        "jump_history_coverage_start": cov_start,
        "known_unrecoverable_gap": {
            "period": "2026-03-07 〜 収集開始日の前日",
            "jump_races": 47,
            "rows": 582,
            "reason": "data/weekly が障害を含まず、共有 JV-Link ストアでの "
                      "過去 backfill は隔離不可のため永久中止 "
                      "(判断分岐 B、2026-09-22)",
            "policy": "推測補完しない / 馬名 join しない / 0 埋めしない",
        },
        "collected_dates": dates,
        "updated_at": datetime.now(timezone.utc).astimezone().isoformat(
            timespec="seconds"),
    }
    if not dry:
        atomic_write_text(MANIFEST, json.dumps(man, ensure_ascii=False, indent=2))
    return man


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", type=int)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    if a.all:
        dates = sorted(int(p.stem) for p in BUNSEKI.glob("*.csv")
                       if p.stem.isdigit())
    elif a.date:
        dates = [a.date]
    else:
        ap.error("--date か --all が必要")

    log(f"収集対象: {dates}  (dry={a.dry})")
    results = []
    for d in dates:
        try:
            r = collect(d, a.dry)
        except CollisionError as e:
            log(f"  {d}: !! COLLISION → 停止: {e}")
            raise
        results.append(r)
        rm, sw = r.get("raw_meta", {}), r.get("settled_write")
        log(f"  {d}: raw={r.get('raw_write')} "
            f"({rm.get('races', 0)}R/{rm.get('rows', 0)}行) settled={sw}"
            f"{'  ' + r.get('settled_meta', {}).get('reason', '') if r.get('settled_meta', {}).get('reason') else ''}")

    man = update_manifest(results, a.dry)
    log(f"\nmanifest: coverage_start={man['jump_history_coverage_start']} "
        f"collected={man['collected_dates']}")
    if a.dry:
        log("(dry-run: ファイルは書いていない)")


if __name__ == "__main__":
    main()
