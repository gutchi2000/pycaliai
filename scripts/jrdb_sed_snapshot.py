# -*- coding: utf-8 -*-
"""
jrdb_sed_snapshot.py — JRDB 成績 (SED) を日付付きで控える。追記専用 (2026-11-19)
==========================================================================
背景: v7/v8 の過去走特徴 (j1_*) は data/jrdb/sed.parquet から引く。sed.parquet は毎週作り直され、
JRDB 側も同じ開催日の成績ファイルを差し替える (開催当日の夕方に速報版、翌週木曜に確定版。
速報版は IDM・ペース指数などが空)。後から作り直した sed.parquet で過去の配信日を再現すると、
その日には無かった値が混ざって前向き検証 (v8 判定 CW-24、JRDB 市場残差 CW-25、CW-16) が成り立たない。

このスクリプトは 3 種類を data/jrdb/snapshots/Sed/ に控える。どれも既存ファイルは上書きしない
(新しい名前で排他作成。同じ内容なら何もしない)。
  files/{元の名前}__{sha256 先頭 12}.zip   成績ファイルの実物。中身ごとに 1 本 (同じ名前でも中身が変われば別の 1 本)
      (a) ローカルの data/jrdb/raw/Sed/*.zip (sed.parquet の材料)
      (b) JRDB サーバー上の直近 --recheck-days 日の日別ファイル (速報版→確定版の差し替えを拾う。raw には書かない)
  parquet/sed_{取得時刻}.parquet           その時点の data/jrdb/sed.parquet (=配信が実際に読んだ表)。中身が変わったときだけ
  manifest.jsonl                           1 行 1 件の記録 (取得時刻・sha256・由来・Last-Modified・差し替えか)

ある開催日 D の配信が読んだ表 = D の 9:00 より前に取った最後の parquet。`asof_parquet("YYYYMMDD")` で引く。
2026-11-14 に手で取った data/jrdb/snapshots/sed_20261114.parquet は動かさず、manifest に legacy として載せる。

data/jrdb/raw と sed.parquet には触れない (読むだけ)。データは git 管理外 (JRDB 規約で再配布禁止)。
自動実行: タスク PyCaLiAI_JRDB_Snapshot (金土 19:40・土日 07:30) の scripts/jrdb_snapshot.py capture が、
         jrdb_refresh.py (sed.parquet の更新) の直後に呼ぶ。
手動 (root から): python scripts/jrdb_sed_snapshot.py capture [--no-recheck] [--notify]
                  python scripts/jrdb_sed_snapshot.py status
                  python scripts/jrdb_sed_snapshot.py asof 20261121
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("PYCALIAI_DATA_ROOT", "E:/PyCaLiAI"))
RAW = ROOT / "data/jrdb/raw/Sed"
SED_PQ = ROOT / "data/jrdb/sed.parquet"
SNAP = ROOT / "data/jrdb/snapshots/Sed"
FILES = SNAP / "files"
PQ = SNAP / "parquet"
MANIFEST = SNAP / "manifest.jsonl"
LEGACY = ROOT / "data/jrdb/snapshots/sed_20261114.parquet"
LEGACY_AT = "2026-11-14T07:45:00"        # 手で取ったときのファイル時刻
PARSER = Path(__file__).resolve().parent / "jrdb_parse.py"
PRE_RACE_HOUR = 9                         # 開催日の 9:00 より前に取ったものをその日の「配信時点」とする
DAILY = re.compile(r"SED(\d{6})\.zip$", re.I)


def sha256(path=None, data=None):
    h = hashlib.sha256()
    if data is not None:
        h.update(data)
        return h.hexdigest()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifest():
    if not MANIFEST.exists():
        return []
    return [json.loads(x) for x in MANIFEST.read_text(encoding="utf-8").splitlines() if x.strip()]


def append(rec):
    SNAP.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_new(dst: Path, data: bytes = None, src: Path = None) -> bool:
    """dst を排他作成する。既にあれば何もしない (False)。途中で落ちても半端なファイルを残さない。"""
    if dst.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    if data is None:
        data = src.read_bytes()
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    try:
        os.link(tmp, dst)                     # dst が既にあれば失敗する (上書きしない)
    except FileExistsError:
        tmp.unlink()
        return False
    except OSError:                           # ハードリンク不可の環境: 存在確認つきで改名
        if dst.exists():
            tmp.unlink()
            return False
        os.rename(tmp, dst)
        return True
    tmp.unlink()
    return True


def race_date(name):
    m = DAILY.search(name)
    if not m:
        return None
    yy = int(m.group(1)[:2])
    return dt.datetime.strptime(("19" if yy >= 90 else "20") + m.group(1), "%Y%m%d").date()


def stored_files(ms):
    """{元の名前: [sha, ...]} 控え済みの成績ファイル (古い順)"""
    out = {}
    for m in ms:
        if m.get("kind") == "file":
            out.setdefault(m["name"], []).append(m["sha256"])
    return out


def keep_file(ms_files, name, data, now, origin, last_modified=None, file_mtime=None):
    sha = sha256(data=data)
    have = ms_files.get(name, [])
    if sha in have:
        return None
    fname = f"{Path(name).stem}__{sha[:12]}{Path(name).suffix}"
    write_new(FILES / fname, data=data)
    d = race_date(name)
    rec = {"kind": "file", "name": name, "race_date": d.strftime("%Y%m%d") if d else None, "origin": origin,
           "captured_at": now.isoformat(timespec="seconds"), "sha256": sha, "bytes": len(data),
           "file": f"files/{fname}", "last_modified": last_modified, "file_mtime": file_mtime, "revised": bool(have)}
    append(rec)
    ms_files.setdefault(name, []).append(sha)
    return rec


def raw_fingerprint():
    """sed.parquet の材料 (raw/Sed の zip 群) と変換コードの指紋"""
    h = hashlib.sha256()
    for p in sorted(RAW.glob("*.zip")):
        h.update(p.name.encode() + b"\0" + sha256(p).encode() + b"\n")
    if PARSER.exists():
        h.update(b"parser\0" + sha256(PARSER).encode())
    return h.hexdigest()


def pq_summary(path):
    import pandas as pd
    y = pd.read_parquet(path, columns=["ymd"]).ymd
    return {"rows": int(len(y)), "max_ymd": int(y.max())}


def cmd_capture(a):
    now = dt.datetime.now()
    ms = read_manifest()
    files = stored_files(ms)
    if LEGACY.exists() and not any(m.get("kind") == "parquet" and m.get("legacy") for m in ms):
        rec = {"kind": "parquet", "legacy": True, "captured_at": LEGACY_AT, "sha256": sha256(LEGACY),
               "bytes": LEGACY.stat().st_size, "file": "../sed_20261114.parquet", "raw_fingerprint": None,
               "note": "2026-11-14 に手で取った控え。場所は動かさない", **pq_summary(LEGACY)}
        append(rec)
        ms.append(rec)
        print(f"legacy を記録: sed_20261114.parquet (最終 {rec['max_ymd']})")
    # (a) ローカルの材料
    n_raw = 0
    for p in sorted(RAW.glob("*.zip")):
        if keep_file(files, p.name, p.read_bytes(), now, "local_raw",
                     file_mtime=dt.datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")):
            n_raw += 1
    # (b) サーバー上の直近の日別ファイル
    n_srv = n_rev = 0
    srv_err = None
    if not a.no_recheck:
        if now.weekday() >= 5 and 8 <= now.hour < 19:
            print("レース当日 08-19 時は JRDB の取得制限があるので、サーバーの再確認は飛ばす")
        else:
            try:
                sys.path.insert(0, str(Path(__file__).resolve().parent))
                import jrdb_fetch as jfe
                s = jfe.session()
                _, st, links = jfe.list_links(s, "Sed")
                if st != 200:
                    raise RuntimeError(f"Sed 一覧 HTTP {st}")
                since = now.date() - dt.timedelta(days=a.recheck_days)
                for link in links:
                    name = link.rsplit("/", 1)[-1]
                    d = race_date(name)
                    if d is None or d < since:
                        continue
                    r = s.get(link, timeout=120)
                    time.sleep(1)
                    if r.status_code != 200:
                        print(f"{name}: HTTP {r.status_code}")
                        continue
                    rec = keep_file(files, name, r.content, now, "server", r.headers.get("Last-Modified"))
                    if rec:
                        n_srv += 1
                        n_rev += rec["revised"]
                        print(f"保存 {rec['file']}  Last-Modified {rec['last_modified']}  差し替え={rec['revised']}")
            except Exception as e:  # noqa: BLE001  サーバーの再確認が失敗しても、ローカル分と parquet の控えは続ける
                srv_err = f"{type(e).__name__}: {e}"
                print(f"サーバーの再確認に失敗: {srv_err}")
    # (c) 配信が読む sed.parquet
    pq_new = None
    if SED_PQ.exists():
        sha = sha256(SED_PQ)
        fp = raw_fingerprint()
        last = [m for m in ms if m.get("kind") == "parquet"]
        if not any(m["sha256"] == sha for m in last) and not (last and last[-1].get("raw_fingerprint") == fp):
            fname = f"sed_{now.strftime('%Y%m%dT%H%M%S')}.parquet"
            if write_new(PQ / fname, src=SED_PQ):
                if sha256(PQ / fname) != sha:        # 写している間に sed.parquet が書き換わった
                    print("sed.parquet が写している間に変わったので、この回の parquet は記録しない (次の回で取る)")
                else:
                    pq_new = {"kind": "parquet", "captured_at": now.isoformat(timespec="seconds"), "sha256": sha,
                              "bytes": SED_PQ.stat().st_size, "file": f"parquet/{fname}", "raw_fingerprint": fp,
                              "sed_parquet_mtime": dt.datetime.fromtimestamp(SED_PQ.stat().st_mtime).isoformat(timespec="seconds"),
                              **pq_summary(SED_PQ)}
                    append(pq_new)
    print(f"sed snapshot: 材料 {n_raw} 本 / サーバー {n_srv} 本 (うち差し替え {n_rev}) / "
          f"parquet {'新規 ' + pq_new['file'] + ' 最終 ' + str(pq_new['max_ymd']) if pq_new else '変化なし'}")
    if srv_err and a.notify:
        try:
            import requests
            url = json.loads((ROOT / "notify_config.json").read_text(encoding="utf-8")).get("discord_webhook")
            if url:
                requests.post(url, json={"content": f"⚠️ JRDB 成績 (SED) の控え: サーバーの再確認に失敗 ({srv_err[:120]})"}, timeout=30)
        except Exception as e:  # noqa: BLE001
            print(f"notify failed: {type(e).__name__}")
    return 0


def asof_parquet(race_ymd: str, hour: int = PRE_RACE_HOUR):
    """開催日 race_ymd (YYYYMMDD) の hour 時より前に取った最後の sed.parquet の控え (Path)。無ければ None。"""
    cut = dt.datetime.strptime(race_ymd, "%Y%m%d").replace(hour=hour)
    best = None
    for m in read_manifest():
        if m.get("kind") == "parquet" and dt.datetime.fromisoformat(m["captured_at"]) < cut:
            if best is None or m["captured_at"] >= best["captured_at"]:
                best = m
    return (SNAP / best["file"]).resolve() if best else None


def cmd_asof(a):
    p = asof_parquet(a.date)
    print(p if p else f"{a.date} の {PRE_RACE_HOUR}:00 より前の控えはありません")
    return 0 if p else 1


def cmd_status(a):
    ms = read_manifest()
    fs = [m for m in ms if m.get("kind") == "file"]
    ps = [m for m in ms if m.get("kind") == "parquet"]
    rev = [m for m in fs if m.get("revised")]
    print(f"成績ファイル {len(fs)} 本 (うち同じ名前の差し替え {len(rev)} 本) / parquet {len(ps)} 本")
    for m in ps[-5:]:
        print(f"  {m['captured_at']}  最終 {m['max_ymd']}  {m['rows']:,} 行  {m['file']}")
    for m in rev[-5:]:
        print(f"  差し替え {m['name']}  {m['captured_at']}  Last-Modified {m.get('last_modified')}")
    return 0


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--recheck-days", type=int, default=42, help="サーバーで取り直して照合する日別ファイルの範囲 (既定 42 日)")
    c.add_argument("--no-recheck", action="store_true", help="サーバーに取りに行かない (ローカルの材料と parquet だけ)")
    c.add_argument("--notify", action="store_true", help="サーバーの再確認に失敗したとき私用 Discord へ")
    c.set_defaults(fn=cmd_capture)
    s = sub.add_parser("asof")
    s.add_argument("date", help="YYYYMMDD")
    s.set_defaults(fn=cmd_asof)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
