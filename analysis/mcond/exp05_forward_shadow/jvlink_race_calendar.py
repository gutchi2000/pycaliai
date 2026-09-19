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

【2026-09-19夜、PowerShell経由破損の切り分けのため--output モード追加】
PowerShellの子プロセスとして実行すると発走時刻が壊れる不具合の切り分けのため、
Task Schedulerから直接32bit Pythonを起動する(PowerShellを介さない)経路で実測する。
`--output <path>`を指定すると、標準出力へのtab区切り出力の代わりに以下を行う:
  - サニティチェック通過時のみ、指定JSONスキーマ(下記)を一時ファイルへ書き、
    os.replace()でatomicに本ファイルへrenameする(部分書き込みが本ファイルとして
    見えることを防ぐ)。
  - サニティチェック失敗時は本ファイルを一切作らない(古い本ファイルも残っていれば
    そのままにする=上書きしない、呼び出し側が「新しい判定結果が無い」と正しく
    区別できるようにするため)。
  - 標準出力は使わない(PowerShellがstdoutをJSON変換する経路を経由しないことを
    保証するため)。エラー詳細は`--errlog`(既定`logs/jvlink_calendar_errors.log`)へ
    Python自身が直接追記する(シェルのリダイレクトに頼らない)。
  - exit code: 0=JSON書き込み成功、1=取得失敗またはサニティチェック失敗(JSON未書込)。

出力JSONスキーマ (file_hashはrace_ids+post_timesの正規化表現のSHA256、後から
読む側が再計算して一致確認することで改ざん・破損を検知できるようにするため):
  target_date, generated_at, source, python_executable, record_count,
  race_ids, post_times, source_record_version, sanity_check_result, file_hash

実行 (標準出力モード、従来通り):
  py -3.12-32 -m analysis.mcond.exp05_forward_shadow.jvlink_race_calendar --date 20260921
  py -3.12-32 -m analysis.mcond.exp05_forward_shadow.jvlink_race_calendar --date 20260921 \
      --lookback-days 14 --verify-against-weekly 20260919
出力: 標準出力に "rid16\tHH:MM" を1行1レースで出す (market_snapshot.py --list-schedule と
      同一フォーマット、t35_shadow.ps1側で差し替え可能にするため)。

実行 (ファイル出力モード、Task Scheduler直接実行での実測用):
  py -3.12-32 -m analysis.mcond.exp05_forward_shadow.jvlink_race_calendar --date 20260921 \
      --output logs/jvlink_calendar_20260921.json
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

SOURCE_RECORD_VERSION = "RA_offsets_v1_20260919"  # フィールドオフセット実証時点のバージョン印

BASE = None  # このファイルはBASE/dataのSID探索を使わない(32bit環境で依存最小に保つため)
try:
    SID = (Path(__file__).resolve().parents[3] / "data" / "jvlink_sid.txt").read_text(
        encoding="utf-8").strip().splitlines()[0] or "UNKNOWN"
except Exception:
    SID = "UNKNOWN"

VENUE_CODES = {"01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
               "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"}
DEFAULT_ERRLOG = Path(__file__).resolve().parents[3] / "logs" / "jvlink_calendar_errors.log"


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

    sanity_err = _sanity_check(out, target_date)
    if sanity_err:
        return [], sanity_err
    return out, ""


def _sanity_check(races: list[dict], target_date: str = "") -> str:
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
    ここで弾くことが安全側に働く)。

    2026-09-19深夜追加: 日付不一致・レースID重複・非合理的な発走時刻の3チェックを追加
    (ユーザー指摘: Task Scheduler直接実行での実測に先立ち、個別タスク登録前の防御を
    強化する)。"""
    if not races:
        return ""

    rids = [r["rid16"] for r in races]
    if len(set(rids)) != len(rids):
        dups = sorted({x for x in rids if rids.count(x) > 1})
        return f"サニティチェック失敗: レースIDが重複している ({dups[:5]})→抽出異常の疑いのため破棄"

    if target_date:
        wrong_date = [r["rid16"] for r in races if r["rid16"][:8] != target_date]
        if wrong_date:
            return (f"サニティチェック失敗: 対象日{target_date}と異なる日付のレースIDが混入 "
                   f"({wrong_date[:5]})→抽出異常の疑いのため破棄")

    for r in races:
        try:
            hh = int(r["post"][:2])
        except Exception:
            return f"サニティチェック失敗: 発走時刻の形式が不正 ({r['post']!r})→抽出異常の疑いのため破棄"
        if not (7 <= hh <= 21):
            return (f"サニティチェック失敗: 発走時刻{r['post']}がJRAの実運用範囲外(07:00-21:59)"
                   f"→抽出異常の疑いのため破棄")

    if len(races) >= 3:
        distinct_posts = {r["post"] for r in races}
        if len(distinct_posts) <= 1:
            return (f"サニティチェック失敗: {len(races)}件のレース全てが同一発走時刻"
                   f"({next(iter(distinct_posts), '?')})→抽出異常の疑いのため破棄")
        if "00:00" in distinct_posts:
            return "サニティチェック失敗: 発走時刻00:00(JRAでは実在しない時刻)を検出→抽出異常の疑いのため破棄"
    return ""


def _canonical_hash(race_ids: list[str], post_times: list[str]) -> str:
    """race_ids+post_timesから再現可能なSHA256(順序に依存しないよう事前にソートされた
    リストを前提とする)。JSON読み込み側が再計算して file_hash と突合し、
    改ざん・部分書き込み・想定外の手編集を検知できるようにするため。"""
    canon = json.dumps({"race_ids": race_ids, "post_times": post_times},
                       ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def write_calendar_json(target_date: str, races: list[dict], output_path: Path) -> dict:
    """成功時のcalendar JSONを構築し、一時ファイル書き込み→atomic renameで
    output_path へ確定させる。呼び出し前にサニティチェック通過が前提
    (この関数自体はチェックを行わない、fetch_races()の結果をそのまま渡す用途)。"""
    race_ids = [r["rid16"] for r in races]
    post_times = [r["post"] for r in races]
    doc = {
        "target_date": target_date,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": "jvlink_race_calendar",
        "python_executable": sys.executable,
        "record_count": len(races),
        "race_ids": race_ids,
        "post_times": post_times,
        "source_record_version": SOURCE_RECORD_VERSION,
        "sanity_check_result": {"passed": True, "record_count": len(races),
                                "distinct_post_times": len(set(post_times))},
        "file_hash": _canonical_hash(race_ids, post_times),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp_path, output_path)  # atomic rename (同一ボリューム内)
    return doc


def verify_calendar_json(doc: dict, max_age_minutes: float = 180.0) -> str:
    """読み込んだcalendar JSONの鮮度・整合性を再検証する(win32com不要、pure Python。
    t35_shadow.ps1側がJSONを使う前に必ず通すことを想定)。異常なしなら空文字列。"""
    for k in ("target_date", "generated_at", "race_ids", "post_times", "file_hash"):
        if k not in doc:
            return f"calendar JSON検証失敗: 必須キー'{k}'が無い"
    race_ids = doc["race_ids"]
    post_times = doc["post_times"]
    if len(race_ids) != len(post_times):
        return "calendar JSON検証失敗: race_idsとpost_timesの件数不一致"
    recomputed = _canonical_hash(race_ids, post_times)
    if recomputed != doc["file_hash"]:
        return "calendar JSON検証失敗: file_hash不一致(内容が破損または改変されている疑い)"
    try:
        gen = datetime.fromisoformat(doc["generated_at"])
    except Exception:
        return f"calendar JSON検証失敗: generated_at形式不正 ({doc.get('generated_at')!r})"
    age_min = (datetime.now() - gen).total_seconds() / 60.0
    if age_min > max_age_minutes:
        return f"calendar JSON検証失敗: 生成から{age_min:.1f}分経過(上限{max_age_minutes}分)→古い可能性、前日/別日フォールバック禁止"
    if age_min < -5:
        return f"calendar JSON検証失敗: generated_atが未来時刻 ({doc['generated_at']})"
    pairs = list(zip(race_ids, post_times))
    err = _sanity_check([{"rid16": r, "post": p} for r, p in pairs], doc.get("target_date", ""))
    if err:
        return f"calendar JSON検証失敗: 再検証で異常検出 ({err})"
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


def _log_error(errlog_path: Path, message: str) -> None:
    """シェルのリダイレクトに頼らずPython自身がエラーログへ直接追記する
    (Task Schedulerの直接Action実行にはstdout/stderrリダイレクト機構が無いため)。"""
    try:
        errlog_path.parent.mkdir(parents=True, exist_ok=True)
        with open(errlog_path, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat(timespec='seconds')} {message}\n")
    except Exception:
        pass  # ログ自体の失敗でexit codeの意味を変えない


DEFAULT_CALENDAR_DIR = Path(__file__).resolve().parents[3] / "data" / "_research" / "mcond" / "exp05fs_calendar"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="",
                    help="YYYYMMDD (開催年月日)。省略時は実行時点の今日"
                        "(Task SchedulerのActionを固定コマンドラインのままにできるようにするため)")
    ap.add_argument("--lookback-days", type=int, default=14)
    ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument("--verify-against-weekly", default="",
                    help="診断用: 指定日のdata/weekly/{date}.csvと突合した結果をstderrへ")
    ap.add_argument("--output", default="",
                    help="指定するとファイル出力モード: サニティチェック通過時のみ"
                        "指定パスへcalendar JSONをatomicに書く。標準出力は使わない。"
                        "省略時は--calendar-dir/{date}.jsonを自動使用。")
    ap.add_argument("--calendar-dir", default=str(DEFAULT_CALENDAR_DIR),
                    help="--outputを省略した場合の出力先ディレクトリ (ファイル出力モードで使用)")
    ap.add_argument("--errlog", default=str(DEFAULT_ERRLOG),
                    help="ファイル出力モード時のエラーログ先 (Python自身が直接追記)")
    ap.add_argument("--verify-file", default="",
                    help="指定するとcalendar JSON検証モード: win32com不要。指定パスのJSONを"
                        "読み、--date(必須、期待する対象日)と一致し鮮度・ハッシュが正しければ"
                        "rid16\\tHH:MM形式で標準出力へ出しexit 0。異常ならexit 1、標準出力は空。")
    ap.add_argument("--max-age-minutes", type=float, default=600.0,
                    help="--verify-fileで許容する生成からの経過分数 (既定600分=10時間、"
                        "カレンダー生成8:20〜観測終了18:00をカバーしつつ前日以前の"
                        "取り残しJSONは弾く)")
    args = ap.parse_args()

    if args.verify_file:
        try:
            doc = json.loads(Path(args.verify_file).read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[jvlink_race_calendar][verify_file] 読込失敗: {exc}", file=sys.stderr)
            return 1
        if not args.date:
            print("[jvlink_race_calendar][verify_file] --dateで期待対象日を指定すること", file=sys.stderr)
            return 1
        if doc.get("target_date") != args.date:
            print(f"[jvlink_race_calendar][verify_file] target_date不一致 "
                 f"(JSON={doc.get('target_date')!r} 期待={args.date!r})→前日/別日JSONへのフォールバック拒否",
                 file=sys.stderr)
            return 1
        err = verify_calendar_json(doc, args.max_age_minutes)
        if err:
            print(f"[jvlink_race_calendar][verify_file] {err}", file=sys.stderr)
            return 1
        for rid, post in zip(doc["race_ids"], doc["post_times"]):
            print(f"{rid}\t{post}")
        return 0

    date_str = args.date or datetime.now().strftime("%Y%m%d")
    races, err = fetch_races(date_str, args.lookback_days, args.timeout)

    if args.output or not args.date:
        # --dateを省略した(=Task Scheduler固定コマンドラインからの日次自動実行)場合は
        # 常にファイル出力モードとして扱う(標準出力に頼る経路をそもそも使わせない)。
        output_path = Path(args.output) if args.output else (Path(args.calendar_dir) / f"{date_str}.json")
        errlog_path = Path(args.errlog)
        if err or not races:
            _log_error(errlog_path, f"[jvlink_race_calendar] date={date_str} "
                                    f"取得失敗、JSON未書込 (output={output_path}): {err or '0件'}")
            return 1
        try:
            write_calendar_json(date_str, races, output_path)
        except Exception as exc:
            _log_error(errlog_path, f"[jvlink_race_calendar] date={date_str} "
                                    f"JSON書き込み失敗: {exc}")
            return 1
        return 0

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
