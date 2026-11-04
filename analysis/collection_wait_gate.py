# -*- coding: utf-8 -*-
"""
collection_wait_gate.py — 収集待ち台帳 (docs/research/COLLECTION_WAIT_REGISTRY.md) の Gate 到達・異常を機械判定する
===========================================================================================================
2026-11-04 新設。台帳の「現在量」は人が覚えて更新する運用だったので、Phase C (weekly_post.ps1) で毎週
各行の現在量を正本ファイルから数え直し、以下のときだけ Discord に通知する (台帳の更新規則 5 に従い、
単なるデータ増加では通知しない):
  * Gate 到達 (必要量に初めて達した)
  * 現在量の減少 / 正本ファイルの消失 (壊れた・巻き戻った)
  * 停滞 (開催日が増えているのに 10 日以上カウントが増えない = collector 停止)
スナップショットは data/collection_wait_status.json (履歴 30 件) と docs/research/COLLECTION_WAIT_STATUS.md
(自動生成の現況表、台帳本体は人が書く) に残す。

実行: python -m analysis.collection_wait_gate [--notify] [--quiet]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "scripts"))

STATUS_JSON = BASE / "data" / "collection_wait_status.json"
STATUS_MD = BASE / "docs" / "research" / "COLLECTION_WAIT_STATUS.md"
def _a95_start() -> str:
    """CW-10/11 の起算日 = data/production_policy.json の prospective.start_date (案 B で版が変わるたびに更新される)。"""
    try:
        d = json.loads((BASE / "data" / "production_policy.json").read_text(encoding="utf-8"))
        return str(d["prospective"]["start_date"])
    except Exception:  # noqa: BLE001
        return "20261024"


A95_START = _a95_start()        # CW-10/11 再起算日 (production_policy.json 由来)
STALE_DAYS = 10


def _latest_weekly_date() -> str | None:
    ds = sorted(p.stem for p in (BASE / "data" / "weekly").glob("2026????.csv"))
    return ds[-1] if ds else None


def _kekka_exists(date: str) -> bool:
    return (BASE / "data" / "kekka" / f"{date}.csv").exists()


# ---------------------------------------------------------------- 各行のカウンタ
def cw01_exp19() -> dict:
    p = BASE / "analysis/mcond/exp19_bodyweight_track_condition_dev/out/forward_parity.json"
    if not p.exists():
        return {"id": "CW-01", "name": "EXP19 当日馬体重 W経路 (T−28 完全率)", "missing": str(p)}
    d = json.loads(p.read_text(encoding="utf-8"))
    return {"id": "CW-01", "name": "EXP19 当日馬体重 W経路 (T−28 完全率)", "current": int(d.get("n_days") or 0),
            "target": 4, "unit": "開催日", "status": str(d.get("status", "")),
            "latest": max((str(x) for x in (d.get("days_collected") or [])), default=None), "source": str(p.relative_to(BASE))}


def cw02_jump() -> dict:
    p = BASE / "data/history_only/jump/manifest.json"
    if not p.exists():
        return {"id": "CW-02", "name": "障害履歴 forward-only collector", "missing": str(p)}
    d = json.loads(p.read_text(encoding="utf-8"))
    dates = [str(x) for x in (d.get("collected_dates") or [])]
    return {"id": "CW-02", "name": "障害履歴 forward-only collector", "current": len(dates), "target": None,
            "unit": "開催日", "latest": max(dates, default=None), "source": str(p.relative_to(BASE))}


def cw03_exp05f() -> dict:
    try:
        from analysis.mcond.exp05_forward_shadow.observation_report import build_cumulative
        c = build_cumulative()
        per = c.get("per_date") or {}
        return {"id": "CW-03", "name": "EXP05-F T−35 前向き市場観測 (valid primary)",
                "current": int(c.get("cumulative_valid_primary_observations") or 0),
                "target": int(c.get("target_valid_primary_observations") or 6600), "unit": "race",
                "latest": max(per.keys(), default=None), "market": int(c.get("cumulative_market_observations") or 0),
                "source": "analysis.mcond.exp05_forward_shadow.observation_report --cumulative"}
    except Exception as e:  # noqa: BLE001
        return {"id": "CW-03", "name": "EXP05-F T−35 前向き市場観測 (valid primary)", "missing": f"build_cumulative 失敗: {e}"}


def cw09_trio() -> dict:
    # 三連複 500R は Stage 0 通過窓の初日以降だけを数える契約で、カウントは analysis.obs_guard の管轄
    # (無断の性能経路を静的検査が検出する)。本スクリプトは数えず、状態だけを載せる。
    return {"id": "CW-09", "name": "三連複 500R (§5.3 初回棄却専用)", "current": None, "target": 500, "unit": "race",
            "latest": None, "note": "Stage 0 未通過 (CW-07/08 Dry 継続)。カウントは obs_guard 管轄、ここでは数えない",
            "source": "analysis/obs_guard.py"}


def cw10_shadow_a95() -> dict:
    d = BASE / "data/shadow_ledger/a95_equal_v1/settlements"
    if not d.exists():
        return {"id": "CW-10", "name": "shadow a95_equal_v1 (紙上台帳・全レース)", "missing": str(d)}
    n = 0
    dates = []
    for p in sorted(d.glob("2026????.json")):
        if p.stem < A95_START:
            continue
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        k = sum(1 for r in (j.get("races") or []) if r.get("valid") and r.get("result") is not None)
        if k:
            n += k
            dates.append(p.stem)
    return {"id": "CW-10", "name": "shadow a95_equal_v1 (紙上台帳・全レース)", "current": n, "target": 500, "unit": "race",
            "latest": dates[-1] if dates else None, "since": A95_START, "source": "data/shadow_ledger/a95_equal_v1/settlements"}


def cw11_live_a95() -> dict:
    n = 0
    dates = []
    for p in sorted((BASE / "reports" / "cowork_output").glob("2026????_bets.json")):
        date = p.name[:8]
        if date < A95_START or not _kekka_exists(date):
            continue
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        bets = raw["bets"] if isinstance(raw, dict) and "bets" in raw else raw
        k = sum(1 for e in bets if (e.get("a95_all") or {}).get("guard_passed") and e.get("bets"))
        if k:
            n += k
            dates.append(date)
    return {"id": "CW-11", "name": "本線 a95_equal_v1 前向き cohort (参戦ガード通過・決済済)", "current": n, "target": 500,
            "unit": "race", "latest": dates[-1] if dates else None, "since": A95_START,
            "source": "reports/cowork_output/*_bets.json (a95_all.guard_passed) × data/kekka"}


COUNTERS = [cw01_exp19, cw02_jump, cw03_exp05f, cw09_trio, cw10_shadow_a95, cw11_live_a95]


# ---------------------------------------------------------------- 判定
def judge(rows: list[dict], prev: dict | None, latest_weekly: str | None) -> list[str]:
    alerts = []
    prev_rows = {r["id"]: r for r in (prev or {}).get("rows", [])}
    for r in rows:
        pid = r["id"]
        if r.get("missing"):
            alerts.append(f"{pid} 正本が無い/読めない: {r['missing']}")
            continue
        cur, tgt = r.get("current"), r.get("target")
        p = prev_rows.get(pid)
        if p and p.get("current") is not None and cur is not None and cur < p["current"]:
            alerts.append(f"{pid} 現在量が減少 {p['current']} → {cur} ({r['name']})")
        if tgt and cur is not None and cur >= tgt and not (p and (p.get("current") or 0) >= tgt):
            alerts.append(f"🎯 {pid} Gate 到達 {cur}/{tgt} {r.get('unit','')} — 台帳の「次の判断」を実行する ({r['name']})")
        if tgt and cur is not None and 0 < cur < tgt and r.get("latest") and latest_weekly:
            try:
                dl = dt.datetime.strptime(r["latest"], "%Y%m%d").date()
                dw = dt.datetime.strptime(latest_weekly, "%Y%m%d").date()
                if (dw - dl).days > STALE_DAYS:
                    alerts.append(f"{pid} 停滞: 最終 {r['latest']} だが開催は {latest_weekly} まで進んでいる ({r['name']})")
            except ValueError:
                pass
    return alerts


def render_md(rows: list[dict], alerts: list[str], now: str) -> str:
    out = [f"# 収集待ち 現況 (自動生成 {now})", "",
           "`python -m analysis.collection_wait_gate` が Phase C で生成。台帳本体は `COLLECTION_WAIT_REGISTRY.md`（人が書く）。",
           "通知は Gate 到達・減少・消失・停滞のときだけ。", "",
           "| ID | 対象 | 現在 | 必要 | 進捗 | 最終データ日 | 正本 |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("missing"):
            out.append(f"| {r['id']} | {r['name']} | — | — | **正本なし** | — | {r['missing']} |")
            continue
        cur, tgt = r.get("current"), r.get("target")
        prog = (f"{100*cur/tgt:.1f}%" if (tgt and cur is not None) else ("(Gate なし)" if not tgt else r.get("note", "—")))
        out.append(f"| {r['id']} | {r['name']} | {cur if cur is not None else '—'} {r.get('unit','') if cur is not None else ''} | {tgt or '—'} | {prog} | {r.get('latest') or '—'} | `{r.get('source','')}` |")
    out += ["", "## 今回の通知", ""] + ([f"- {a}" for a in alerts] or ["- なし"]) + [""]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--notify", action="store_true", help="到達/異常があれば Discord へ")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    rows = []
    for fn in COUNTERS:
        try:
            rows.append(fn())
        except Exception as e:  # noqa: BLE001
            rows.append({"id": fn.__name__.split("_")[0].upper().replace("CW", "CW-"), "name": fn.__name__, "missing": f"例外 {e}"})
    prev = json.loads(STATUS_JSON.read_text(encoding="utf-8")) if STATUS_JSON.exists() else None
    latest_weekly = _latest_weekly_date()
    alerts = judge(rows, prev, latest_weekly)
    snap = {"generated_at": now, "latest_weekly": latest_weekly, "rows": rows, "alerts": alerts,
            "history": ((prev or {}).get("history") or [])[-29:] + [{"at": now, "counts": {r["id"]: r.get("current") for r in rows}}]}
    STATUS_JSON.write_bytes(json.dumps(snap, ensure_ascii=False, indent=1).encode("utf-8"))
    STATUS_MD.write_bytes(render_md(rows, alerts, now).encode("utf-8"))
    if not a.quiet:
        for r in rows:
            if r.get("missing"):
                print(f"  {r['id']:<6} 正本なし: {r['missing']}")
            else:
                tgt = r.get("target")
                print(f"  {r['id']:<6} {r.get('current')}{('/' + str(tgt)) if tgt else ''} {r.get('unit','')}  最終 {r.get('latest')}  {r['name']}")
        print(f"[collection_wait_gate] alerts={len(alerts)} → {STATUS_MD.relative_to(BASE)}")
        for x in alerts:
            print("   ! " + x)
    if alerts and a.notify:
        try:
            from kaiko_warm import discord
            discord("**📋 収集待ち台帳 Gate 判定**\n" + "\n".join(f"- {x}" for x in alerts) +
                    f"\n現況: docs/research/COLLECTION_WAIT_STATUS.md ({now})")
        except Exception as e:  # noqa: BLE001
            print(f"  [notify] 失敗 (無視): {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
