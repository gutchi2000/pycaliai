# -*- coding: utf-8 -*-
"""
t10_runner.py — 当日 T-10 自動馬券ライン（スケジューラ本体）
================================================================
土日の当日に 1 回起動しておくと、bundle の全レースについて発走 lead 分前
(デフォルト T-10) に自動で:

  1. jvlink_odds.py (32-bit, py -3.12-32) → reports/live_odds/{rid16}.json
  2. compute_bets.py --race {rid16} --live-odds-dir --apply
       → reports/cowork_output/{date}_bets.json へ当該レースのみ in-place merge
  3. validate_cowork_bets.py --apply (見送りガード)
  4. compute_bets(🎫 prob-first) と gutchi_brain(🧠 俺のブレイン) の買い目を併記表示
     + ビープ（投票は人間が IPAT。ブレインは同一ライブオッズ/realized bias で算出した
       比較用ラインで、bets.json には書かない = 見送りガードも通さない）

を実行する。NiceGUI (ローカル) は cowork_output を ui.timer で随時読むので
画面にもそのまま反映される。HF への push は本ランナーでは行わない（手動）。

発走時刻: data/weekly/{date}.csv の「発走時刻」列（TARGET 出走表）。
fail-safe: オッズ欠損/鮮度NG/overround 異常は compute_bets 側で見送りになる。

実行:
  venv311\\Scripts\\python.exe t10_runner.py                # 当日( bundle 自動検出)
  venv311\\Scripts\\python.exe t10_runner.py 20260614       # 日付指定
  venv311\\Scripts\\python.exe t10_runner.py 20260614 --dry # 計算のみ・書込なし
  venv311\\Scripts\\python.exe t10_runner.py 20260607 --once 2026060705030211 --max-age-min 99999
                                                            # 1レース即時処理（動作テスト用）

ルーチン運用 (タスクスケジューラ "PyCaLiAI_T10" → t10.ps1 -Routine が呼ぶ):
  --wait-bundle : bundle 未生成なら 2 分間隔でポーリングして待つ (Phase A 完了待ち)。
                  デッドライン (--wait-deadline, 既定 15:00) 超過で諦めて終了。
                  日付未指定なら「今日」を対象にする (最新 bundle 検出だと翌日分を
                  拾い得るため、ルーチンでは必ず当日に固定)。
  多重起動ガード: reports/live_odds/.t10_lock (12h 未満) があれば起動拒否 (--force-lock で無視)。
"""
from __future__ import annotations
import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

BASE = Path(__file__).parent
PY32 = ["py", "-3.12-32"]          # JV-Link は 32-bit COM
LIVE_DIR = BASE / "reports" / "live_odds"
LOCK = LIVE_DIR / ".t10_lock"
NOTIFY_CFG = BASE / "notify_config.json"   # gitignore 対象 (webhook URL は秘匿)
POLL_SEC = 15
BUNDLE_POLL_SEC = 120

sys.stdout.reconfigure(encoding="utf-8")


def _webhook_url() -> str:
    """Discord webhook URL。環境変数 PYCALIAI_DISCORD_WEBHOOK > notify_config.json。"""
    url = os.environ.get("PYCALIAI_DISCORD_WEBHOOK", "").strip()
    if url:
        return url
    try:
        cfg = json.loads(NOTIFY_CFG.read_text(encoding="utf-8"))
        return str(cfg.get("discord_webhook", "") or "").strip()
    except (OSError, ValueError):
        return ""


def notify(text: str) -> bool:
    """Discord webhook へ送信。未設定/失敗は非致命 (本処理は止めない)。"""
    url = _webhook_url()
    if not url:
        return False
    import urllib.request
    body = json.dumps({"content": text[:1990]}).encode("utf-8")  # Discord 上限 2000 字
    # User-Agent 必須: Python-urllib デフォルト UA は Cloudflare に 403 で弾かれる
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json",
                                 "User-Agent": "PyCaLiAI-T10 (private bot)"})
    try:
        urllib.request.urlopen(req, timeout=10)
        return True
    except Exception as e:
        print(f"  [notify] Discord 送信失敗 (非致命): {e}")
        return False


# ---------------------------------------------------------------
# Discord Bot 受信 (予算再計算コマンド)。webhook は送信専用なので、
# 受信には bot_token + channel_id が必要 (notify_config.json)。
# 未設定なら一方向通知のみで動く。
# ---------------------------------------------------------------
def _bot_cfg() -> tuple[str, str]:
    try:
        cfg = json.loads(NOTIFY_CFG.read_text(encoding="utf-8"))
        return (str(cfg.get("bot_token", "") or "").strip(),
                str(cfg.get("channel_id", "") or "").strip())
    except (OSError, ValueError):
        return "", ""


def _discord_get_messages(token: str, channel_id: str, after_id: str | None):
    """チャンネルの新着メッセージ (古い順) を返す。失敗は [] (非致命)。"""
    import urllib.request
    url = f"https://discord.com/api/v10/channels/{channel_id}/messages?limit=20"
    if after_id:
        url += f"&after={after_id}"
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bot {token}",
        "User-Agent": "PyCaLiAI-T10 (private bot)"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            msgs = json.loads(r.read().decode("utf-8"))
        return sorted(msgs, key=lambda m: int(m["id"]))  # 古い順
    except Exception as e:
        print(f"  [bot] メッセージ取得失敗 (非致命): {e}")
        return []


def parse_budget_command(text: str) -> int | None:
    """「金額2000円」「2000円」「¥2000」「2000」「3,000円」→ int。該当なしは None。"""
    t = (text or "").strip().replace(",", "").replace("，", "")
    m = re.search(r"(\d{3,6})\s*円", t)
    if not m:
        m = re.fullmatch(r"[¥\\]?\s*(\d{3,6})", t)
    if not m:
        return None
    v = int(m.group(1))
    return v if 500 <= v <= 100000 else None


class BotPoller:
    """T-10 ループ内で Discord 返信をポーリングし、予算コマンドを拾う。"""

    def __init__(self):
        self.token, self.channel = _bot_cfg()
        self.enabled = bool(self.token and self.channel)
        self.last_id: str | None = None
        if self.enabled:
            # 起動以前のメッセージには反応しない (最新 id を起点にする)
            msgs = _discord_get_messages(self.token, self.channel, None)
            if msgs:
                self.last_id = msgs[-1]["id"]
            print(f"[bot] 予算コマンド受付 ON (channel={self.channel})")
        else:
            print("[bot] bot_token/channel_id 未設定 → 一方向通知のみ "
                  "(返信での予算再計算は無効)")

    def poll_budgets(self) -> list[int]:
        """新着の人間メッセージから予算コマンドを抽出。"""
        if not self.enabled:
            return []
        msgs = _discord_get_messages(self.token, self.channel, self.last_id)
        budgets = []
        for m in msgs:
            self.last_id = m["id"]
            if m.get("author", {}).get("bot"):
                continue   # 自分の webhook 投稿等は無視
            b = parse_budget_command(m.get("content", ""))
            if b is not None:
                budgets.append(b)
        return budgets


def acquire_lock(force: bool) -> bool:
    """多重起動ガード (手動起動とスケジュールタスクの同時実行防止)。"""
    try:
        if LOCK.exists():
            age_h = (time.time() - LOCK.stat().st_mtime) / 3600.0
            if age_h < 12 and not force:
                print(f"[ERROR] 別の t10_runner が稼働中の可能性 ({LOCK} が {age_h:.1f}h 前)。"
                      "多重起動を中止。強制するなら --force-lock")
                return False
        LIVE_DIR.mkdir(parents=True, exist_ok=True)
        LOCK.write_text(f"{os.getpid()} {datetime.now().isoformat()}", encoding="utf-8")
    except OSError:
        pass   # ロックが書けなくても本処理は止めない
    return True


def release_lock():
    try:
        LOCK.unlink(missing_ok=True)
    except OSError:
        pass


def _rid16(x) -> str:
    return re.sub(r"\D", "", str(x))[:16]


def beep():
    try:
        import winsound
        winsound.Beep(880, 250); winsound.Beep(1320, 250)
    except Exception:
        print("\a", end="")


# --- keep-awake: 処理中は PC をアイドルスリープさせない (SetThreadExecutionState) ---
_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


def keep_awake(on: bool):
    """on=True で「実行中はスリープ禁止」。レース処理の前後で呼ぶ。"""
    try:
        import ctypes
        flags = (_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED) if on else _ES_CONTINUOUS
        ctypes.windll.kernel32.SetThreadExecutionState(flags)
    except Exception:
        pass   # 非 Windows / 失敗は非致命


def build_schedule(date_str: str, races: list, lead_min: float):
    """bundle races + 発走時刻 → (sched[(post_dt, rid16, label)], missing)。
    --list-schedule とループ両方で使う単一ソース。"""
    post = load_post_times(date_str)
    base_day = datetime.strptime(date_str, "%Y%m%d")
    sched, missing = [], []
    for r in races:
        rid = _rid16(r.get("race_id", ""))
        rm = r.get("race_meta", {})
        label = f"{rm.get('place','')} {rm.get('course','')} {rm.get('class','')}".strip()
        hm = parse_hhmm(post.get(rid, ""))
        if hm is None:
            missing.append((rid, label)); continue
        sched.append((base_day.replace(hour=hm[0], minute=hm[1], second=0), rid, label))
    sched.sort()
    return sched, missing


def latest_bundle_date() -> str | None:
    files = glob.glob(str(BASE / "reports" / "cowork_input" / "*_bundle.json"))
    dates = sorted({Path(f).name.split("_")[0] for f in files
                    if re.match(r"^\d{8}_", Path(f).name)}, reverse=True)
    return dates[0] if dates else None


def parse_hhmm(s: str) -> tuple[int, int] | None:
    """'15:40' / '1540' / '15:40:00' / '15時40分' → (15, 40)"""
    s = str(s or "").strip()
    m = re.match(r"^(\d{1,2})[:時](\d{2})", s)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.match(r"^(\d{3,4})$", s)
    if m:
        v = int(m.group(1))
        return v // 100, v % 100
    return None


def load_post_times(date_str: str) -> dict[str, str]:
    """data/weekly/{date}.csv → {rid16: '15:40'} (cp932, TARGET 出走表)"""
    path = BASE / "data" / "weekly" / f"{date_str}.csv"
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    with open(path, encoding="cp932", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        rid_col = next((c for c in (reader.fieldnames or []) if "レースID" in c), None)
        tm_col = next((c for c in (reader.fieldnames or []) if "発走" in c), None)
        if not rid_col or not tm_col:
            return {}
        for row in reader:
            rid = _rid16(row.get(rid_col, ""))
            if len(rid) == 16 and rid not in out:
                hm = parse_hhmm(row.get(tm_col, ""))
                if hm:
                    out[rid] = f"{hm[0]:02d}:{hm[1]:02d}"
    return out


def run_cmd(cmd: list[str], timeout: int = 180) -> tuple[int, str]:
    env = dict(os.environ, PYTHONUTF8="1")
    try:
        r = subprocess.run(cmd, cwd=str(BASE), env=env, timeout=timeout,
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired:
        return -1, f"timeout {timeout}s: {' '.join(cmd)}"
    except FileNotFoundError as e:
        return -1, str(e)


def brain_tickets(bundle: Path, rid16: str, date_str: str,
                  live_dir: Path | None, max_age_min: float,
                  budget: int | None = None) -> dict | None:
    """gutchi_brain (俺のブレイン) で同一レースの買い目を算出 (併走比較用)。
    compute_bets と同じライブオッズ・realized bias を食わせ、bets.json には書かない。
    返値: {"tickets":[...], "note":str, "miokuri":bool} / None(レース不在・bundle不能)。"""
    import copy
    try:
        d = json.loads(bundle.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    race = next((r for r in d.get("races", [])
                 if _rid16(r.get("race_id", "")) == rid16), None)
    if race is None:
        return None
    race = copy.deepcopy(race)   # bundle を破壊しない

    # realized bias (土→日クロスデイ): show_on_date == 今日 の時だけ一次シグナルに使う
    sat_bias, note = None, ""
    rb_path = BASE / "data" / "realized_bias.json"
    if rb_path.exists():
        try:
            rb = json.loads(rb_path.read_text(encoding="utf-8"))
            if rb.get("show_on_date") == date_str and rb.get("venues"):
                sat_bias = rb["venues"]
                note = f" +{rb.get('label', '実現')}"
        except (OSError, ValueError):
            pass

    # ライブオッズ (compute_bets.load_live_odds と同一ソース) を horses にマージ
    scratched: set = set()
    if live_dir is not None:
        from compute_bets import load_live_odds
        live, why = load_live_odds(live_dir, rid16, max_age_min)
        if live is None:      # compute_bets と同じ fail-safe 見送り
            return {"tickets": [], "note": why, "miokuri": True}
        live_tan = {int(k): v for k, v in (live.get("tansho") or {}).items()}
        live_fuku = {int(k): v for k, v in (live.get("fukusho") or {}).items()}
        for h in race.get("horses", []):
            try:
                b = int(h.get("umaban"))
            except (TypeError, ValueError):
                continue
            if b in live_tan:
                h["tansho_odds"] = live_tan[b]
            if b in live_fuku:
                h["fuku_odds_low"], h["fuku_odds_high"] = live_fuku[b]
        if live_tan:          # 実オッズが取れた頭のみ生存 = ライブに居ない馬は取消扱い
            for h in race.get("horses", []):
                try:
                    b = int(h.get("umaban"))
                except (TypeError, ValueError):
                    continue
                if b not in live_tan:
                    scratched.add(h.get("umaban"))
        note = " [T-10オッズ]" + note

    import gutchi_brain
    tickets = gutchi_brain.build_tickets(
        race, sat_bias=sat_bias, scratched=scratched,
        budget=int(budget) if budget else gutchi_brain.BUDGET)
    return {"tickets": tickets, "note": note.strip(), "miokuri": not tickets}


def render_brain(brain: dict | None) -> tuple[list[str], list[str]]:
    """brain 結果 → (コンソール行, Discord行)。brain=None は空。"""
    if brain is None:
        return [], []
    if brain.get("miokuri"):
        note = f" ({brain['note']})" if brain.get("note") else ""
        return ([f"  🧠 俺のブレイン: 見送り{note}"],
                [f"🧠 **俺のブレイン: 見送り**{note}"])
    ts = brain["tickets"]
    tot = sum(t["購入額"] for t in ts)
    note = f" {brain['note']}" if brain.get("note") else ""
    con = [f"  🧠 俺のブレイン {len(ts)}点 ¥{tot:,}{note}"]
    dis = [f"🧠 **俺のブレイン {len(ts)}点 ¥{tot:,}**{note}"]
    for t in ts:
        con.append(f"     {t['馬券種']:3s} {t['買い目']:10s} ¥{t['購入額']:>5,}  {t.get('理由','')}")
        dis.append(f"{t['馬券種']} `{t['買い目']}` ¥{t['購入額']:,}  {t.get('理由','')}")
    return con, dis


def show_race_bets(date_str: str, rid16: str, brain: dict | None = None):
    """apply 後の bets.json (🎫 prob-first) を読み戻し、🧠 俺のブレインと併記表示 + Discord 通知。"""
    path = BASE / "reports" / "cowork_output" / f"{date_str}_bets.json"
    if not path.exists():
        return
    raw = json.loads(path.read_text(encoding="utf-8"))
    races = raw["bets"] if isinstance(raw, dict) and "bets" in raw else raw
    e = next((r for r in races if _rid16(r.get("race_id", "")) == rid16), None)
    if e is None:
        return
    hosei_line = ""
    if e.get("hosei_marks"):
        from compute_bets import fmt_hosei
        hosei_line = fmt_hosei(e["hosei_marks"])

    cb: list[str] = []   # Discord 行 (compute_bets = prob-first 側)
    has_bets = bool(e.get("bets"))
    if has_bets:
        tot = sum(b["購入額"] for b in e["bets"])
        head = (f"🎫 {e.get('race_label','')} [{e.get('race_nature','')}] "
                f"prob-first {len(e['bets'])}点 ¥{tot:,}")
        print(f"  {head}")
        cb.append(f"**{head}**")
        for b in e["bets"]:
            print(f"     {b['馬券種']:3s} {b['買い目']:8s} ¥{b['購入額']:>5,}  {b.get('理由','')}")
            cb.append(f"{b['馬券種']} `{b['買い目']}` ¥{b['購入額']:,}  {b.get('理由','')}")
        if hosei_line:
            print(f"     {hosei_line}")
            cb.append(hosei_line)
        print(f"     → {e.get('race_reason','')}")
        cb.append(f"_{e.get('race_reason','')}_")
    else:
        print(f"  — {e.get('race_label','')} [見送り] {e.get('race_reason','')}")
        cb.append(f"🎫 {e.get('race_label','')} **prob-first 見送り** {e.get('race_reason','')}")
        if hosei_line:
            print(f"     {hosei_line}")
            cb.append(hosei_line)

    con_b, dis_b = render_brain(brain)
    for l in con_b:
        print(l)

    # Discord: 🎫 と 🧠 を 1 通に統合。2000 字上限に近い時だけ 2 通に分割。
    if dis_b:
        combined = "\n".join(cb + [""] + dis_b)
        if len(combined) <= 1900:
            notify(combined)
        else:
            notify("\n".join(cb))
            notify("\n".join(dis_b))
    else:
        notify("\n".join(cb))
    if has_bets:
        beep()


def ensure_plan(date_str: str) -> None:
    """枠プラン(reports/bet_plan/{date}.json)が無ければ build_bet_plan.py で生成。
    失敗しても続行(compute_bets は --plan 無しの従来挙動にフォールバック)。"""
    plan = BASE / "reports" / "bet_plan" / f"{date_str}.json"
    if plan.exists():
        print(f"[plan] 既存 {plan.name} を使用")
        return
    try:
        rc, _ = run_cmd([sys.executable, "build_bet_plan.py", date_str])
        print(f"[plan] build_bet_plan {date_str} (exit {rc}) → "
              f"{'生成OK' if plan.exists() else '生成失敗→--plan無しで継続'}")
    except Exception as e:
        print(f"[plan] 生成スキップ ({e})→--plan無しで継続")


def process_race(date_str: str, bundle: Path, rid16: str, label: str,
                 max_age_min: float, dry: bool, budget: int | None = None) -> bool:
    """T-10 処理 1 レース分。True=完了 (見送り含む)。budget=予算再計算 (Discord コマンド)。"""
    now = datetime.now().strftime("%H:%M:%S")
    tag = f" (予算¥{budget:,} 再計算)" if budget else ""
    print(f"\n[{now}] ▶ T-10 処理開始: {label} ({rid16}){tag}")

    # 1. JV-Link ライブオッズ (32-bit)。失敗しても compute_bets が fail-safe 見送りにする
    rc, out = run_cmd([*PY32, "jvlink_odds.py", "--race", rid16])
    line = next((l for l in out.splitlines() if "[jvlink_odds]" in l), out.strip()[-200:])
    print(f"  [1/3] jvlink_odds (exit {rc}) {line}")

    # 2. compute_bets (当該レースのみ、ライブ必須モード)
    cmd = [sys.executable, "compute_bets.py", "--bundle", str(bundle),
           "--live-odds-dir", str(LIVE_DIR), "--max-age-min", str(max_age_min),
           "--race", rid16]
    if budget:
        cmd += ["--budget", str(int(budget))]    # Discord 手動再計算: 枠予算を上書き
    else:
        _plan = BASE / "reports" / "bet_plan" / f"{date_str}.json"
        if _plan.exists():
            cmd += ["--plan", str(_plan)]         # 枠プラン: per-race予算 + プロ条件floor(force_floor)
    if not dry:
        cmd.append("--apply")
    rc, out = run_cmd(cmd)
    print(f"  [2/3] compute_bets (exit {rc})")
    if rc != 0:
        print("       " + out.strip().replace("\n", "\n       ")[-500:])
        return True   # 再試行しない (fail-safe 見送り扱い)

    # gutchi_brain (俺のブレイン) 併走: compute_bets と同じライブオッズ/realized bias で比較買い目
    brain = None
    try:
        brain = brain_tickets(bundle, rid16, date_str, LIVE_DIR, max_age_min, budget)
    except Exception as ex:
        print(f"  [brain] gutchi_brain 失敗 (非致命): {ex}")

    if dry:
        # dry は compute_bets の出力 + 🧠 ブレインをそのまま見せる
        for l in out.splitlines():
            if l.strip():
                print("       " + l)
        for l in render_brain(brain)[0]:
            print(l)
        return True

    # 3. 見送りガード (書込後は必ず通す契約。ブレインは bets.json 非書込なので対象外)
    rc, out = run_cmd([sys.executable, "validate_cowork_bets.py",
                       "--date", date_str, "--apply"])
    print(f"  [3/3] validate_cowork_bets (exit {rc})")
    if rc == 1:
        print("       ⚠ ガード実行不能: " + out.strip()[-300:])

    show_race_bets(date_str, rid16, brain=brain)
    return True


def main():
    ap = argparse.ArgumentParser(description="T-10 自動馬券ライン")
    ap.add_argument("date", nargs="?", default=None, help="YYYYMMDD (省略時は最新 bundle)")
    ap.add_argument("--lead-min", type=float, default=10.0,
                    help="発走の何分前に処理するか (default 10 = T-10)")
    ap.add_argument("--max-age-min", type=float, default=20.0,
                    help="ライブオッズ許容鮮度 (分) → compute_bets へ渡す")
    ap.add_argument("--dry", action="store_true", help="計算のみ。bets.json へ書き込まない")
    ap.add_argument("--once", default=None,
                    help="rid16 を指定して 1 レースだけ処理 (レース毎タスク / テスト用)")
    ap.add_argument("--poll-until", default=None,
                    help="--once 後、この時刻 HH:MM まで Discord 予算返信を受け付ける"
                         " (レース毎タスクが発走時刻を渡す)。省略時は即終了")
    ap.add_argument("--list-schedule", action="store_true",
                    help="rid<TAB>発走HH:MM<TAB>label を出力して終了 (t10.ps1 -Schedule 用)")
    ap.add_argument("--wait-bundle", action="store_true",
                    help="bundle 未生成なら 2 分間隔で待つ (ルーチン用。日付未指定なら今日)")
    ap.add_argument("--wait-deadline", default="15:00",
                    help="--wait-bundle の諦め時刻 HH:MM (default 15:00)")
    ap.add_argument("--force-lock", action="store_true",
                    help="多重起動ガード (.t10_lock) を無視して起動")
    ap.add_argument("--test-notify", action="store_true",
                    help="Discord 通知のテスト送信だけして終了")
    args = ap.parse_args()

    if args.test_notify:
        if not _webhook_url():
            print("[notify] 未設定。notify_config.json の discord_webhook に "
                  "webhook URL を貼るか、環境変数 PYCALIAI_DISCORD_WEBHOOK を設定。")
            return 1
        ok = notify("✅ PyCaLiAI T-10 ライン: 通知テスト成功。土日はこのチャンネルに買い目が届きます。")
        print(f"[notify] テスト送信 {'成功' if ok else '失敗'}")
        return 0 if ok else 1

    if args.wait_bundle and not args.date:
        # ルーチンは必ず「今日」(最新 bundle 検出だと先行生成済みの翌日分を拾い得る)
        date_str = datetime.now().strftime("%Y%m%d")
    else:
        date_str = args.date or latest_bundle_date()
    if not date_str:
        print("[ERROR] bundle が見つからない (reports/cowork_input/)"); return 1
    bundle = BASE / "reports" / "cowork_input" / f"{date_str}_bundle.json"

    if not bundle.exists() and args.wait_bundle:
        hm = parse_hhmm(args.wait_deadline) or (15, 0)
        deadline = datetime.now().replace(hour=hm[0], minute=hm[1], second=0)
        print(f"[wait] {bundle.name} 未生成 → Phase A 完了を待機 "
              f"({BUNDLE_POLL_SEC}s 間隔, {deadline:%H:%M} まで)")
        while not bundle.exists():
            if datetime.now() >= deadline:
                print(f"[ERROR] {deadline:%H:%M} までに bundle が生成されず → 諦めて終了。"
                      "(開催日でない / Phase A 未実行)")
                notify(f"⚠ T-10 ライン {date_str}: {deadline:%H:%M} までに bundle 未生成のため"
                       "終了 (Phase A 未実行?)")
                return 1
            time.sleep(BUNDLE_POLL_SEC)
        print(f"[wait] bundle 検出 → 開始")
    if not bundle.exists():
        print(f"[ERROR] {bundle} が無い (Phase A を先に実行)"); return 1

    ensure_plan(date_str)   # 枠プラン(勝負/準勝負/消化 + プロ条件 10R∧¥100k)を用意

    d = json.loads(bundle.read_text(encoding="utf-8"))
    races = d.get("races", [])

    # --list-schedule: t10.ps1 -Schedule がレース毎タスク登録に使う (rid<TAB>HH:MM<TAB>label)
    if args.list_schedule:
        sched, _ = build_schedule(date_str, races, args.lead_min)
        for pt, rid, label in sched:
            print(f"{rid}\t{pt:%H:%M}\t{label}")
        return 0

    # --once: 1 レース処理 (レース毎タスク本体)。--poll-until まで予算返信を受付。
    if args.once:
        rid = _rid16(args.once)
        rm = next((r.get("race_meta", {}) for r in races
                   if _rid16(r.get("race_id", "")) == rid), {})
        label = f"{rm.get('place','')}{rm.get('R','') or ''} {rm.get('course','')}".strip() or rid
        keep_awake(True)
        try:
            process_race(date_str, bundle, rid, label, args.max_age_min, args.dry)
            # 発走時刻まで予算返信 (「2000円」) を受け付けて再計算
            until_hm = parse_hhmm(args.poll_until) if args.poll_until else None
            if until_hm and not args.dry:
                until = datetime.now().replace(hour=until_hm[0], minute=until_hm[1], second=0)
                poller = BotPoller()
                if poller.enabled and datetime.now() < until:
                    print(f"[bot] 予算返信を {until:%H:%M} まで受付")
                    while datetime.now() < until:
                        for b in poller.poll_budgets():
                            notify(f"🔁 {label} を予算 ¥{b:,} で再計算します…")
                            process_race(date_str, bundle, rid, label,
                                         args.max_age_min, args.dry, budget=b)
                        time.sleep(POLL_SEC)
        finally:
            keep_awake(False)
        return 0

    # スケジュール構築 (1本ループ運用 -Routine 用。レース毎タスク運用では使わない)
    sched, missing = build_schedule(date_str, races, args.lead_min)

    if missing:
        print(f"[WARN] 発走時刻不明 {len(missing)}R (data/weekly/{date_str}.csv に無い) → スキップ:")
        for rid, label in missing:
            print(f"   {rid} {label}")
    if not sched:
        print("[ERROR] スケジュール対象 0 レース"); return 1

    lead = timedelta(minutes=args.lead_min)
    print(f"\n========== T-10 自動馬券ライン  {date_str}  "
          f"{len(sched)}R (T-{args.lead_min:.0f}, 鮮度{args.max_age_min:.0f}分"
          f"{', DRY' if args.dry else ''}) ==========")
    for pt, rid, label in sched:
        print(f"  {pt:%H:%M} 発走 → {(pt-lead):%H:%M} 処理  {label}")
    print("  (Ctrl+C で停止。投票は人間が IPAT で行う)\n")
    if not args.dry:
        notify(f"🏇 T-10 ライン起動 {date_str}: {len(sched)}R をスケジュール "
               f"(T-{args.lead_min:.0f}, {sched[0][0]:%H:%M}〜{sched[-1][0]:%H:%M})")

    if not acquire_lock(args.force_lock):
        return 1
    poller = BotPoller() if not args.dry else None
    last_post = sched[-1][0]
    keep_awake(True)   # 1本ループ運用中はアイドルスリープ禁止 (凍結=取りこぼし防止)
    try:
        done: set[str] = set()
        last_proc: tuple | None = None   # 直近処理レース (rid, label, post_dt)
        while True:
            now = datetime.now()
            for pt, rid, label in sched:
                if rid in done:
                    continue
                if now >= pt:
                    print(f"[{now:%H:%M:%S}] ✗ {label} は発走済み ({pt:%H:%M}) → スキップ")
                    done.add(rid)
                elif now >= pt - lead:
                    process_race(date_str, bundle, rid, label, args.max_age_min, args.dry)
                    done.add(rid)
                    last_proc = (rid, label, pt)
            # Discord 返信の予算コマンド (「2000円」等) → 直近レースを新予算で再計算
            if poller:
                for b in poller.poll_budgets():
                    if last_proc is None:
                        notify("⚠ まだ処理済みレースがありません (最初の T-10 をお待ちください)")
                        continue
                    rid, label, pt = last_proc
                    if datetime.now() >= pt:
                        notify(f"⚠ {label} は発走済みのため再計算できません")
                        continue
                    notify(f"🔁 {label} を予算 ¥{b:,} で再計算します…")
                    process_race(date_str, bundle, rid, label,
                                 args.max_age_min, args.dry, budget=b)
            # 全レース処理済みでも最終発走までは返信を受け付ける
            if len(done) >= len(sched) and now >= last_post:
                break
            time.sleep(POLL_SEC)
    finally:
        release_lock()
        keep_awake(False)
    print(f"\n========== 全 {len(sched)}R 処理完了 ==========")
    if not args.dry:
        notify(f"🏁 T-10 ライン {date_str}: 全 {len(sched)}R 処理完了")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[停止] Ctrl+C")
        raise SystemExit(130)
