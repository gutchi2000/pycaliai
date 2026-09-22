# -*- coding: utf-8 -*-
"""
preregistration_gate.py — PyCaLiAI_JumpHistory タスク登録前の関門
=================================================================
ユーザー指定の登録前 Gate をすべて機械的に検査する。
1 つでも FAIL ならタスクを登録しない。

実行: venv311/Scripts/python.exe -m analysis.jump_history_only.preregistration_gate
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
PY = str(BASE / "venv311" / "Scripts" / "python.exe")
PS1 = BASE / "analysis" / "jump_history_only" / "jump_history_collect.ps1"

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}"
          + (f"  — {detail}" if detail else ""), flush=True)


def run(args, **kw):
    return subprocess.run(args, capture_output=True, text=True,
                          errors="replace", cwd=str(BASE), timeout=900, **kw)


def main() -> int:
    print("=" * 70)
    print("PyCaLiAI_JumpHistory 登録前 Gate")
    print("=" * 70)

    # 1. PowerShell syntax
    r = run(["powershell", "-NoProfile", "-Command",
             f"$e=$null; [void][System.Management.Automation.Language.Parser]::"
             f"ParseFile('{PS1}', [ref]$null, [ref]$e); "
             f"if($e.Count){{$e|%{{$_.Message}}; exit 1}} else {{exit 0}}"])
    check("PowerShell syntax", r.returncode == 0, (r.stdout or r.stderr)[:120])

    # 2. 既存 8 日で idempotency
    r = run([PY, "-m", "analysis.jump_history_only.jump_history_collector", "--all"])
    idem = r.returncode == 0 and "'added': 0" in r.stdout
    check("既存8日で idempotency (added=0)", idem,
          f"exit={r.returncode}")

    # 3. collision で全体停止
    from analysis.jump_history_only.jump_history_collector import (
        CollisionError, append_jsonl, build_raw_card, read_jsonl)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "c.jsonl"
        raw, _ = build_raw_card(20260905)
        append_jsonl(p, raw, ("race_id", "ped_id"), ("captured_at",), dry=False)
        mut = [dict(x) for x in raw]
        mut[0]["distance"] = (mut[0]["distance"] or 0) + 1
        stopped = False
        try:
            append_jsonl(p, mut, ("race_id", "ped_id"), ("captured_at",), dry=False)
        except CollisionError:
            stopped = True
        intact = len(read_jsonl(p)) == len(raw)
        check("collision で全体停止・部分書込なし", stopped and intact,
              f"raised={stopped} rows_intact={intact}")

    # 4. 非開催日 no-input -> exit 0
    from analysis.jump_history_only.jump_history_collector import is_race_day
    nonrace = 20260923
    ok, why = is_race_day(nonrace)
    r = run([PY, "-m", "analysis.jump_history_only.jump_history_collector",
             "--date", str(nonrace)])
    check("非開催日 no-input は exit 0", (not ok) and r.returncode == 0,
          f"is_race_day={ok}({why}) exit={r.returncode}")

    # 5. 開催日 no-input -> 非 0 (MISSING_BUNSEKI_EXPORT)
    #    bunseki を一時退避して実際に確かめる (必ず元へ戻す)
    d = 20260905
    src = BASE / "data" / "bunseki" / f"{d}.csv"
    bak = src.with_suffix(".csv.pregate_bak")
    moved = False
    try:
        shutil.move(str(src), str(bak))
        moved = True
        import race_eligibility as re_mod
        re_mod.clear_cache()
        r = run([PY, "-m", "analysis.jump_history_only.jump_history_collector",
                 "--date", str(d)])
        check("開催日 no-input は FAIL (exit 3)", r.returncode == 3,
              f"exit={r.returncode}")
    finally:
        if moved:
            shutil.move(str(bak), str(src))
            import race_eligibility as re_mod
            re_mod.clear_cache()
    check("bunseki 原本を復元した", src.exists() and not bak.exists())

    # 6. pending settlement
    from analysis.jump_history_only.jump_history_collector import pending_settlements
    pend = pending_settlements()
    ok6 = all(not (BASE / "data" / "kekka" / f"{x}.csv").exists() for x in pend)
    check("pending settlement が正しく列挙される", ok6,
          f"pending={pend}")

    # 7. 通常 prediction/bet 経路へ混入 0
    from race_eligibility import evaluate_race
    jump_rids = set()
    for p in sorted((BASE / "data" / "history_only" / "jump" / "raw_card").glob("*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                jump_rids.add(json.loads(line)["race_id"])
    leaked = []
    for bp in sorted((BASE / "reports" / "cowork_input").glob("*_bundle.json")):
        try:
            j = json.loads(bp.read_text(encoding="utf-8"))
        except Exception:
            continue
        for race in j.get("races", []):
            rid = str(race.get("race_id", ""))[:16]
            if rid in jump_rids:
                leaked.append((bp.name, rid))
    known = {"2026091306040401", "2026091909040504", "2026092009040601"}
    new_leak = {rid for _n, rid in leaked} - known
    check("通常 prediction 経路へ新規混入 0", not new_leak,
          f"既知={len(known)} 新規={sorted(new_leak)}")
    all_jump_blocked = all(not evaluate_race(r)["bet_eligible"] for r in jump_rids)
    check("収集済み障害 race が全て bet 不可", all_jump_blocked,
          f"n={len(jump_rids)}")

    # 8/9. P0 テスト + full suite
    r = run([PY, "-m", "pytest", "tests/test_jump_race_p0_gate.py", "-q"])
    check("P0 障害除外テスト", r.returncode == 0, r.stdout.strip()[-60:])
    r = run([PY, "-m", "pytest",
             "analysis/jump_history_only/test_jump_history_invariants.py",
             "tests/", "-q"])
    check("full test suite", r.returncode == 0, r.stdout.strip()[-80:])

    # 10. タスク定義の固定値
    ps1_txt = PS1.read_text(encoding="utf-8-sig")
    check("WorkingDirectory 固定", "Set-Location 'E:\\PyCaLiAI'" in ps1_txt)
    check("Python interpreter 固定",
          "venv311\\Scripts\\python.exe" in ps1_txt)
    check("ログ出力先固定", "logs" in ps1_txt and "jump_history_" in ps1_txt)
    check("error log 出力先固定", "jump_history_error.log" in ps1_txt)

    print("-" * 70)
    n_fail = sum(1 for _n, ok, _d in results if not ok)
    print(f"{len(results) - n_fail}/{len(results)} PASS")
    if n_fail:
        print("★ FAIL があるためタスクを登録してはいけない")
    else:
        print("★ 全 Gate PASS — タスク登録に進んでよい")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
