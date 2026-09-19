# -*- coding: utf-8 -*-
"""
jvlink_race_calendar.py — JV-Link「RACE」蓄積系dataspecから、weekly CSV(TARGET出走表)に
一切依存せず当日・将来日のレースID+発走時刻を取得する (spec: T-35収集をweekly特徴生成より
優先し、JV-Linkの当日レース情報からT-35収集タスクを先に登録できるようにする)。

★必ず 32-bit Python で実行: py -3.12-32 -m analysis.mcond.exp05_forward_shadow.jvlink_race_calendar

============================================================================================
【2026-09-19の訂正の経緯 — 重要】
`jvlink_race_day_probe.py`は当初(2026-09-19朝)、`JVOpen("RACE", 対象日+000000, 1)`で
「対象日自体」をfromtimeに使い、未来日でJVOpen自体がrc=-1になることから
「JV-Linkは将来日程を取得できない」と結論していた。これは誤りだった。

蓄積系dataspecのfromtimeは「データ作成年月日時分」(=JRA-VANがレコードを発表/更新した
日時)でのフィルタであり、「レース開催日」そのもののフィルタではない。JRAは通常レース
開催の数日前に番組(RAレコード)を先行発表しているため、fromtimeに「対象日そのもの」を
使うと「対象日以降に発表された分」しか返らず、まだ何も発表されていない未来日では
当然rc=-1(該当データなし)になる。正しくは、fromtimeに「対象日より十分前の日付」
(例: 対象日の14日前)を指定して問い合わせれば、対象日の番組が既に発表済みであれば
正しく取得できる。

2026-09-19実機で以下を実証済み(このファイルのフィールドオフセットは全て以下の
実測で検証済み、JRA-VAN公式ドキュメントは未参照):
  - `JVOpen("RACE", "20260101000000", 1)` で2026-09-21(月・祝、中山・阪神)の
    全24レース(発走時刻含む)を取得できた。データ作成年月日=2026-09-17
    (開催4日前に先行発表されていた)。
  - `JVOpen("RACE", "20260801000000", 1)` で2026-08-29〜2026-09-19の7開催日
    (211レース)の既知の発走時刻(data/weekly/*.csvから取得)と全件突合し、
    211/211件でレースIDが一致、204/211件で発走時刻も完全一致。残り7件は
    ±1分のズレのみ(オフセット誤りではなく、TARGET側と JV-Link側の発走時刻
    改定タイミングの差と推測される正常な実世界の差異)。
  - フィールドオフセットは"RA"レコードの先頭2文字(レコード種別ID)を除いた
    絶対位置(buf[0]='R', buf[1]='A' を含めた絶対インデックス)で:
      buf[2]      : データ区分 (1文字)
      buf[3:11]   : データ作成年月日 (YYYYMMDD)
      buf[11:19]  : 開催年月日 (YYYYMMDD) ← レース実施日、これでフィルタする
      buf[19:21]  : 場コード (JRA公式10場コード: 01札幌 02函館 03福島 04新潟
                    05東京 06中山 07中京 08京都 09阪神 10小倉。今回06/09を実証)
      buf[21:23]  : 開催回
      buf[23:25]  : 開催日目
      buf[25:27]  : レース番号
      buf[734:738]: 発走時刻 (HHMM、4桁)
    rid16 = 開催年月日(8) + 場コード(2) + 開催回(2) + 開催日目(2) + レース番号(2)
    は本番の既存rid16フォーマット(t10_runner._rid16等)と完全一致する。

【運用上の注意】
  - このオフセットは2026-09-19時点のJV-Data仕様に対する実機検証結果であり、
    JRA-VAN側の仕様変更があれば無効化しうる(公式仕様書を未参照のため)。
    実際の登録前には毎回 `--verify-against-weekly` で既知の週と突合することを推奨。
  - fromtimeは対象日から遡って`--lookback-days`(既定14日、実測4日の3.5倍の余裕)
    分だけ過去に設定する。将来的にJRAの先行発表がもっと早い/遅い場合は要調整。
  - 発走時刻はJV-Linkの「予定」であり本番のT-35実行時にも別途鮮度検証
    (market_snapshot.py の valid_for_primary 判定)が効くため、本モジュールの
    値をそのまま信頼しきる設計にはしていない(あくまで「タスク登録を何時に
    仕掛けるか」の情報源であり、実際のオッズ取得時刻の正しさは別途保証される)。
  - t35_shadow.ps1 からは常に非致命的に呼ばれる(失敗しても既存のweekly CSV
    ベースのCase A/B/C判定に安全にフォールバックする、非干渉設計)。

実行:
  py -3.12-32 -m analysis.mcond.exp05_forward_shadow.jvlink_race_calendar --date 20260921
  py -3.12-32 -m analysis.mcond.exp05_forward_shadow.jvlink_race_calendar --date 20260921 \
      --lookback-days 14 --verify-against-weekly 20260919
出力: 標準出力に "rid16\tHH:MM" を1行1レースで出す (market_snapshot.py --list-schedule と
      同一フォーマット、t35_shadow.ps1側で差し替え可能にするため)。
"""
from __future__ import annotations
import argparse
import sys
import time
from datetime import datetime, timedelta

BASE = None  # このファイルはBASE/dataのSID探索を使わない(32bit環境で依存最小に保つため)
try:
    from pathlib import Path
    SID = (Path(__file__).resolve().parents[3] / "data" / "jvlink_sid.txt").read_text(
        encoding="utf-8").strip().splitlines()[0] or "UNKNOWN"
except Exception:
    SID = "UNKNOWN"

VENUE_CODES = {"01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
               "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"}


def fetch_races(target_date: str, lookback_days: int = 14, timeout_s: float = 90.0) -> tuple[list[dict], str]:
    """(races, error) を返す。races は成功時 [{"rid16":..,"post":"HH:MM","venue":..}]、
    失敗時は空リスト+errorに理由。例外は投げない(呼び出し側で非干渉に扱えるように)。"""
    try:
        import win32com.client as w
    except Exception as exc:
        return [], f"win32com import失敗: {exc}"

    try:
        jv = w.Dispatch("JVDTLab.JVLink")
    except Exception as exc:
        return [], f"JVLink Dispatch失敗: {exc}"

    try:
        if jv.JVInit(SID) != 0:
            return [], "JVInit失敗"
    except Exception as exc:
        return [], f"JVInit例外: {exc}"

    try:
        target_dt = datetime.strptime(target_date, "%Y%m%d")
    except Exception as exc:
        return [], f"date形式不正: {exc}"
    from_dt = target_dt - timedelta(days=lookback_days)
    from_ts = from_dt.strftime("%Y%m%d") + "000000"

    races: list[dict] = []
    try:
        try:
            r = jv.JVOpen("RACE", from_ts, 1)
        except Exception as exc:
            return [], f"JVOpen例外: {exc}"
        rc = r[0] if isinstance(r, tuple) else r
        if rc != 0:
            return [], f"JVOpen失敗 rc={rc} (from={from_ts})"

        t0 = time.time()
        while time.time() - t0 < timeout_s:
            try:
                rr = jv.JVRead(" " * 120000, 120000, " " * 256)
            except Exception as exc:
                return races, f"JVRead例外(途中まで取得済み): {exc}"
            size = rr[0] if isinstance(rr, tuple) else rr
            if size == 0:
                break
            if size < 0:
                continue
            buf = rr[1][:size] if isinstance(rr, tuple) else ""
            if buf[:2] != "RA" or len(buf) < 738:
                continue
            meet_date = buf[11:19]
            if meet_date != target_date:
                continue
            venue = buf[19:21]
            kaiji = buf[21:23]
            nichime = buf[23:25]
            race_no = buf[25:27]
            hhmm = buf[734:738]
            if not (hhmm.isdigit() and len(hhmm) == 4):
                continue
            rid16 = f"{meet_date}{venue}{kaiji}{nichime}{race_no}"
            post = f"{hhmm[:2]}:{hhmm[2:]}"
            races.append({"rid16": rid16, "post": post, "venue": VENUE_CODES.get(venue, venue)})
    finally:
        try:
            jv.JVClose()
        except Exception:
            pass

    # 重複除去 (同一レースの更新レコードが複数回流れてくることがある。後勝ちでなく
    # rid16単位でユニーク化、最初に見つかった値を採用)
    seen = {}
    for r in races:
        seen.setdefault(r["rid16"], r)
    out = sorted(seen.values(), key=lambda r: (r["post"], r["rid16"]))

    sanity_err = _sanity_check(out)
    if sanity_err:
        return [], sanity_err
    return out, ""


def _sanity_check(races: list[dict]) -> str:
    """抽出結果の最低限の妥当性チェック(win32com不要、pure Python。単体テスト対象)。
    異常なしなら空文字列、異常ありならその理由を返す。

    2026-09-19夜追加、経緯: 実機検証中、全く同じコマンドをPowerShellの子プロセスとして
    実行した場合にのみ発走時刻が全レース"00:00"という明らかに異常な(しかしrc=0・例外無しで
    返ってくる)結果が再現した(原因はJVLink COM側の何らかの環境依存の不具合と推測されるが
    未特定、少なくともスタックした別プロセスの並行アクセスが原因ではないことは確認済み)。
    rc=0で例外も出ないため呼び出し側からは正常応答と区別がつかない。このプロジェクトの
    「既知の正解値との突合でしか信用しない」という方針に従い、返す前に最低限の妥当性を
    自己検査し、異常なら空リスト+エラーとして返す(判定ロジック側は空+エラー=クエリ失敗
    として扱い、既存のweekly CSVベースの安全網へ自動フォールバックする設計なので、
    ここで弾くことが安全側に働く)。"""
    if len(races) < 3:
        return ""
    distinct_posts = {r["post"] for r in races}
    if len(distinct_posts) <= 1:
        return (f"サニティチェック失敗: {len(races)}件のレース全てが同一発走時刻"
               f"({next(iter(distinct_posts), '?')})→抽出異常の疑いのため破棄")
    if "00:00" in distinct_posts:
        return "サニティチェック失敗: 発走時刻00:00(JRAでは実在しない時刻)を検出→抽出異常の疑いのため破棄"
    return ""


def verify_against_weekly(date_str: str, races: list[dict]) -> dict:
    """data/weekly/{date_str}.csv が既にあれば突合し、一致率を返す(診断用、
    判定ロジックには使わない)。"""
    import csv as _csv
    import re as _re
    from pathlib import Path as _P
    path = _P(__file__).resolve().parents[3] / "data" / "weekly" / f"{date_str}.csv"
    if not path.exists():
        return {"available": False}
    known = {}
    with open(path, encoding="cp932", errors="replace", newline="") as f:
        reader = _csv.DictReader(f)
        rid_col = next((c for c in (reader.fieldnames or []) if "レースID" in c), None)
        tm_col = next((c for c in (reader.fieldnames or []) if "発走" in c), None)
        if rid_col and tm_col:
            for row in reader:
                rid = _re.sub(r"\D", "", str(row.get(rid_col, "")))[:16]
                if len(rid) == 16 and rid not in known:
                    s = str(row.get(tm_col, "") or "").strip()
                    m = _re.match(r"^(\d{1,2})[:時](\d{2})", s)
                    if m:
                        known[rid] = f"{int(m.group(1)):02d}:{int(m.group(2)):02d}"
    got = {r["rid16"]: r["post"] for r in races}
    exact = sum(1 for k, v in known.items() if got.get(k) == v)
    return {"available": True, "known_count": len(known), "found_count": len(got),
            "exact_time_matches": exact, "missing_from_jvlink": sorted(set(known) - set(got))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="YYYYMMDD (開催年月日)")
    ap.add_argument("--lookback-days", type=int, default=14)
    ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument("--verify-against-weekly", default="",
                    help="診断用: 指定日のdata/weekly/{date}.csvと突合した結果をstderrへ")
    args = ap.parse_args()

    races, err = fetch_races(args.date, args.lookback_days, args.timeout)
    if err:
        print(f"[jvlink_race_calendar] error: {err}", file=sys.stderr)
    for r in races:
        print(f"{r['rid16']}\t{r['post']}")

    if args.verify_against_weekly:
        v = verify_against_weekly(args.verify_against_weekly, races)
        print(f"[jvlink_race_calendar][verify] {v}", file=sys.stderr)

    return 0 if races else (1 if err else 0)


if __name__ == "__main__":
    raise SystemExit(main())
