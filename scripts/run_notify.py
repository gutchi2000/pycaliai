# -*- coding: utf-8 -*-
"""
scripts/run_notify.py — 週次フローの出力を保存し、WARNING 以上 / serve canary 行だけ Discord へ転送する (2026-11-04)
=====================================================================================================
背景: export_weekly_marks.py の serve canary (絶対 canary / baseline 鮮度 / artifact pins / 品質ゲート) は
コンソールにしか出ず、9/27 の baseline 巻き戻り事故は 5 週間誰も気づかなかった。
WARNING 以上の行を notify_config.json の Discord webhook (t10_runner / kaiko_warm と同じ) へ流す。
通知は best-effort で、子プロセスの exit code はそのまま返す (週次フローの fail-hard を壊さない)。

使い方:
  実行モード: python scripts/run_notify.py --tag "Phase A export 20261107" --log logs/export_20261107.log -- python export_weekly_marks.py --csv ...
              → コマンドを実行し、出力をコンソール + ログに流し、終了後に警告行を通知。exit code は子のもの。
  走査モード: python scripts/run_notify.py --tag "Phase C weekly_post 20261108" --scan logs/weekly_post_20261108.log --rc 1
              → 既存ログ (Tee-Object 出力、UTF-8/UTF-16 自動判別) を走査して通知。exit 0。
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from kaiko_warm import discord  # notify_config.json の discord_webhook を読む (鍵はこのスクリプトに置かない)
except Exception:  # noqa: BLE001
    def discord(text: str) -> bool:  # type: ignore
        print("  [notify] kaiko_warm.discord を import できず通知スキップ")
        return False

# 通知対象 (いずれかに一致)
ALERT = re.compile(
    r"\[(WARNING|ERROR|CRITICAL)\]"          # Python logging
    r"|^\s*(WARNING|警告|ERROR|エラー)\s*[:：]"  # PowerShell Write-Warning / Write-Error
    r"|canary|artifact pins|PolicyError|品質ゲート|Traceback|fail-hard|\bFAIL\b"
    r"|serve recover\] .*失敗|baseline が古い|train_cov が無い",
    re.IGNORECASE,
)
# 毎週必ず出る定常 WARNING (通知しない)
IGNORE = re.compile(
    r"週次CSVに不足の \d+ 列を補完"
    r"|(Performance|Future|User|Deprecation|Runtime)Warning"
    r"|warnings\.warn"
    r"|kako5 履歴未取得"
    r"|予想外の列を無視",
)
MAX_LINES = 25
MAX_CHARS = 220


def _pick(lines: list[str]) -> list[str]:
    out = []
    for ln in lines:
        s = ln.rstrip("\r\n")
        if not s.strip():
            continue
        if ALERT.search(s) and not IGNORE.search(s):
            out.append(s[:MAX_CHARS])
    return out


def _read_log(path: Path) -> list[str]:
    b = path.read_bytes()
    if b.startswith(b"\xff\xfe") or b.startswith(b"\xfe\xff"):
        return b.decode("utf-16", errors="replace").splitlines()
    for enc in ("utf-8-sig", "cp932"):
        try:
            return b.decode(enc).splitlines()
        except UnicodeDecodeError:
            continue
    return b.decode("utf-8", errors="replace").splitlines()


def notify(tag: str, rc: int | None, hits: list[str], log_path: Path | None) -> None:
    if not hits and (rc in (0, None)):
        print(f"  [notify] {tag}: 警告なし (通知しません)")
        return
    head = f"**⚠️ {tag}**" + (f" exit={rc}" if rc not in (0, None) else "")
    body = "\n".join(hits[:MAX_LINES]) if hits else "(警告行なし。exit code のみ異常)"
    more = f"\n…他 {len(hits) - MAX_LINES} 行" if len(hits) > MAX_LINES else ""
    tail = f"\n全文: {log_path}" if log_path else ""
    ok = discord(f"{head}\n```\n{body}{more}\n```{tail}")
    print(f"  [notify] {tag}: {len(hits)} 行 → Discord {'sent' if ok else 'skip'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--log", default=None, help="実行モード: 出力の保存先")
    ap.add_argument("--scan", default=None, help="走査モード: 既存ログを走査")
    ap.add_argument("--rc", type=int, default=None, help="走査モード: 元コマンドの exit code")
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="-- の後に実行するコマンド")
    a = ap.parse_args()

    if a.scan:
        p = Path(a.scan)
        if not p.exists():
            print(f"  [notify] ログが無い: {p}")
            return 0
        notify(a.tag, a.rc, _pick(_read_log(p)), p.resolve())
        return 0

    cmd = [c for c in a.cmd if c != "--"]
    if not cmd:
        ap.error("実行するコマンドを -- の後に指定するか --scan を使う")
    # 素の "python" は親と同じインタプリタに差し替える (2026-11-06)。
    # Windows の venv の python.exe は基底インタプリタを起動する中継で、子の "python" は
    # CreateProcess が「親の実体 exe のディレクトリ」を最初に探すため、venv ではなく
    # 基底 Python (Store 版 3.11、pyarrow/shap 無し) が動いていた。11/07 Phase A で
    # export が parquet を読めず全頭「馬同定 unknown」→ 品質ゲート不合格になった原因。
    if Path(cmd[0]).name.lower() in ("python", "python.exe", "python3", "python3.exe"):
        cmd[0] = sys.executable
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    log_path = Path(a.log) if a.log else None
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                          text=True, encoding="utf-8", errors="replace", bufsize=1) as proc:
        assert proc.stdout is not None
        fh = open(log_path, "w", encoding="utf-8", newline="\n") if log_path else None
        try:
            for ln in proc.stdout:
                sys.stdout.write(ln)
                sys.stdout.flush()
                lines.append(ln)
                if fh:
                    fh.write(ln)
        finally:
            if fh:
                fh.close()
        rc = proc.wait()
    try:
        notify(a.tag, rc, _pick(lines), log_path.resolve() if log_path else None)
    except Exception as e:  # noqa: BLE001  通知失敗で週次を止めない
        print(f"  [notify] 通知処理で例外 (無視): {e}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
