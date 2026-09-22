# -*- coding: utf-8 -*-
"""
p0_transactional_fail_test.py — unknown 時の transactional fail を実ファイルで検証
=================================================================================
production を書き換えず、shadow directory で A/B/C の 3 シナリオを確認する。

A. bunseki 全欠損 → 非 0 終了・bundle 未生成・既存 bundle hash 不変・task 0
B. bunseki 部分欠損 (1 race だけ除く) → 部分 bundle を公開しない
C. 正常 → 通常 race のみ bundle・障害は history-only・exit 0・atomic publish

出力: analysis/jump_history_only/out/p0_transactional_fail_test.json
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = Path(__file__).resolve().parent / "out"
SHADOW = Path(__file__).resolve().parent / "shadow" / "transactional"
OUT.mkdir(parents=True, exist_ok=True)

DATE = "20260913"
JUMP_RID = "2026091306040401"
DROP_ONE_RID = "2026091306040405"      # B で evidence から抜く通常 race
PY = str(BASE / "venv311" / "Scripts" / "python.exe")

results: list[dict] = []


def log(m):
    print(m, flush=True)


def sha256(p: Path) -> str | None:
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def scheduled_rids() -> list[str]:
    from build_horse_history import parse_weekly_light
    w = parse_weekly_light(BASE / "data" / "weekly" / f"{DATE}.csv")
    return sorted(set(w["rid16"]))


def run_gate(date: str) -> tuple[int, dict]:
    """coverage gate を子プロセスで実行し exit code を実測する。"""
    r = subprocess.run([PY, "-m", "eligibility_coverage_gate", "--date", date],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=str(BASE), timeout=600)
    return r.returncode, {"stdout": r.stdout, "stderr": r.stderr[-500:]}


def with_bunseki(tmp_csv: Path | None):
    """data/bunseki/{DATE}.csv を一時的に差し替える context。必ず復元する。"""
    import contextlib

    @contextlib.contextmanager
    def _cm():
        real = BASE / "data" / "bunseki" / f"{DATE}.csv"
        bak = real.with_suffix(".csv.txfail_bak")
        moved = False
        try:
            if real.exists():
                shutil.move(str(real), str(bak))
                moved = True
            if tmp_csv is not None:
                shutil.copy2(str(tmp_csv), str(real))
            import race_eligibility as m
            m.clear_cache()
            yield
        finally:
            if real.exists() and tmp_csv is not None:
                real.unlink()
            if moved:
                shutil.move(str(bak), str(real))
            import race_eligibility as m
            m.clear_cache()
    return _cm()


def make_partial_bunseki(dst: Path, drop_rid: str) -> int:
    """実 bunseki のコピーから 1 race 分の行を除く。"""
    src = BASE / "data" / "bunseki" / f"{DATE}.csv"
    dropped = 0
    with open(src, encoding="cp932", errors="replace", newline="") as f:
        rows = list(csv.reader(f))
    hdr = rows[0]
    try:
        ridx = hdr.index("レースID(新)")
    except ValueError:
        raise RuntimeError("レースID(新) 列が無い")
    keep = [hdr]
    for row in rows[1:]:
        if len(row) > ridx and str(row[ridx]).strip()[:16] == drop_rid:
            dropped += 1
            continue
        keep.append(row)
    with open(dst, "w", encoding="cp932", errors="replace", newline="") as f:
        csv.writer(f).writerows(keep)
    return dropped


def check(name: str, ok: bool, detail: str = ""):
    results.append({"name": name, "pass": bool(ok), "detail": detail})
    log(f"    {'PASS' if ok else 'FAIL'}  {name}" + (f" — {detail}" if detail else ""))


def n_scheduled_tasks() -> int:
    try:
        r = subprocess.run(["schtasks", "/query", "/fo", "LIST"],
                           capture_output=True, text=True, timeout=60,
                           errors="replace")
        return sum(1 for rid in [JUMP_RID] if rid in r.stdout)
    except Exception:
        return -1


def scenario_a() -> dict:
    log("\n[A] bunseki 全欠損 (実 weekly あり)")
    bundle_p = BASE / "reports" / "cowork_input" / f"{DATE}_bundle.json"
    before = sha256(bundle_p)          # sentinel
    rids = scheduled_rids()
    with with_bunseki(None):
        rc, io = run_gate(DATE)
        from eligibility_coverage_gate import check_coverage
        res = check_coverage(DATE, rids)
    after = sha256(bundle_p)

    check("A: exit 非 0", rc != 0, f"exit={rc}")
    check("A: ELIGIBILITY_EVIDENCE_INCOMPLETE",
          "ELIGIBILITY_EVIDENCE_INCOMPLETE" in io["stdout"])
    check("A: 全 race が evidence 欠落",
          len(res.missing_rids) == len(rids) and res.observed == 0,
          f"missing={len(res.missing_rids)}/{len(rids)} observed={res.observed}")
    check("A: track_code_source が全て missing",
          all(x["track_code_source"] == "missing" for x in res.bad_source_rids),
          f"n_bad_source={len(res.bad_source_rids)}")
    check("A: bundle を生成しない (既存 hash 不変)", before == after,
          "sentinel hash 一致" if before == after else "★上書きされた")
    check("A: task 登録 0 (障害 rid のタスク無し)", n_scheduled_tasks() == 0)
    check("A: error log に記録",
          (BASE / "logs" / "eligibility_gate_error.log").exists())
    # --- A2: 朝の実態 (結果系の bias も無い) では determination=unknown ---
    import race_eligibility as m
    real_bias = BASE / "data" / "bias" / f"{DATE}.csv"
    bias_bak = real_bias.with_suffix(".csv.txfail_bak")
    moved_bias = False
    try:
        if real_bias.exists():
            shutil.move(str(real_bias), str(bias_bak))
            moved_bias = True
        with with_bunseki(None):
            from eligibility_coverage_gate import check_coverage as _cc
            res2 = _cc(DATE, rids)
    finally:
        if moved_bias:
            shutil.move(str(bias_bak), str(real_bias))
        m.clear_cache()
    # 収集済みの障害 race は history-only store が証拠になるため unknown にならない
    _known_jump = {r for r in rids if r == JUMP_RID}
    check("A2: bias も無い朝は (収集済み障害を除く) 全 race が unknown",
          set(res2.unknown_rids) == set(rids) - _known_jump,
          f"unknown={len(res2.unknown_rids)}/{len(rids)} "
          f"(収集済み障害 {len(_known_jump)} 件は store が証拠)")
    check("A2: それでも gate は FAIL する (公開しない)",
          not res2.ok and res2.exit_code != 0,
          f"ok={res2.ok} exit={res2.exit_code}")
    check("A2: bias 原本を復元した", real_bias.exists())

    return {"exit_code": rc, "expected": res.expected,
            "observed": res.observed, "missing": res.missing,
            "n_unknown": len(res.unknown_rids),
            "n_bad_source": len(res.bad_source_rids),
            "A2_n_unknown": len(res2.unknown_rids),
            "bundle_sha_before": before, "bundle_sha_after": after,
            "files": res.files}


def scenario_b() -> dict:
    log("\n[B] bunseki 部分欠損 (1 race だけ除く)")
    bundle_p = BASE / "reports" / "cowork_input" / f"{DATE}_bundle.json"
    before = sha256(bundle_p)
    rids = scheduled_rids()
    with tempfile.TemporaryDirectory() as td:
        part = Path(td) / "partial.csv"
        n_dropped = make_partial_bunseki(part, DROP_ONE_RID)
        with with_bunseki(part):
            rc, io = run_gate(DATE)
            from eligibility_coverage_gate import check_coverage
            res = check_coverage(DATE, rids)
    after = sha256(bundle_p)

    check("B: exit 非 0", rc != 0, f"exit={rc}")
    check("B: missing race_id を正確に報告",
          res.missing_rids == [DROP_ONE_RID],
          f"missing={res.missing_rids}")
    check("B: 部分 bundle を公開しない (hash 不変)", before == after)
    check("B: expected/observed が一致しない",
          res.expected == len(rids) and res.observed == len(rids) - 1,
          f"expected={res.expected} observed={res.observed}")
    check("B: task 登録 0", n_scheduled_tasks() == 0)
    return {"exit_code": rc, "dropped_rows": n_dropped,
            "expected": res.expected, "observed": res.observed,
            "missing_rids": res.missing_rids,
            "bundle_sha_before": before, "bundle_sha_after": after}


def scenario_c() -> dict:
    log("\n[C] 正常 (実 weekly + 完全 bunseki)")
    from eligibility_coverage_gate import check_coverage, AtomicPublish
    import race_eligibility as m
    m.clear_cache()
    rids = scheduled_rids()
    rc, io = run_gate(DATE)
    res = check_coverage(DATE, rids)

    check("C: exit 0", rc == 0, f"exit={rc}")
    check("C: coverage 100%", res.expected == res.observed and res.missing == 0,
          f"expected={res.expected} observed={res.observed}")
    check("C: unknown 0", not res.unknown_rids)
    check("C: 重複 0", not res.duplicate_rids)
    check("C: 不許可 source 0", not res.bad_source_rids)
    check("C: flat/jump 不一致 0", not res.disagreement_rids)
    check("C: 障害は bundle へ入らない", JUMP_RID in res.jump_rids)
    check("C: 通常 race のみが flat", len(res.flat_rids) == len(rids) - 1,
          f"flat={len(res.flat_rids)} / {len(rids)-1}")
    check("C: 障害は history-only に保存済み",
          (BASE / "data" / "history_only" / "jump" / "raw_card"
           / f"{DATE}.jsonl").exists())

    # atomic publish 自体の挙動 (shadow の空ディレクトリで)
    SHADOW.mkdir(parents=True, exist_ok=True)
    od = SHADOW / "out"
    bp = SHADOW / "b.json"
    bp.write_text('{"sentinel":1}', encoding="utf-8")
    sent = sha256(bp)
    pub = AtomicPublish(od, bp)
    pub.__enter__()
    (pub.tmp_dir / "x.json").write_text("{}", encoding="utf-8")
    pub.tmp_bundle.write_text('{"new":1}', encoding="utf-8")
    pub.abort()
    check("C: abort で既存 bundle 不変", sha256(bp) == sent)
    check("C: abort で temp が消える", not pub.tmp_dir.exists())

    pub2 = AtomicPublish(od, bp)
    pub2.__enter__()
    (pub2.tmp_dir / "x.json").write_text("{}", encoding="utf-8")
    pub2.tmp_bundle.write_text('{"new":2}', encoding="utf-8")
    pub2.commit()
    check("C: commit で bundle が差し替わる",
          json.loads(bp.read_text(encoding="utf-8")).get("new") == 2)
    check("C: commit で個別 JSON が本番へ移る", (od / "x.json").exists())
    shutil.rmtree(SHADOW, ignore_errors=True)

    return {"exit_code": rc, "expected": res.expected,
            "observed": res.observed, "n_flat": len(res.flat_rids),
            "n_jump": len(res.jump_rids)}


def main() -> int:
    log("=" * 74)
    log("P0 transactional fail — 実ファイル回帰テスト (production 不変)")
    log("=" * 74)
    a = scenario_a()
    b = scenario_b()
    c = scenario_c()

    n_fail = sum(1 for r in results if not r["pass"])
    log("-" * 74)
    log(f"{len(results) - n_fail}/{len(results)} PASS")

    payload = {"generated_at": datetime.now().isoformat(),
               "date": DATE, "jump_rid": JUMP_RID,
               "dropped_rid_for_B": DROP_ONE_RID,
               "scenario_A": a, "scenario_B": b, "scenario_C": c,
               "checks": results, "n_fail": n_fail}
    (OUT / "p0_transactional_fail_test.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    log(f"保存: {OUT / 'p0_transactional_fail_test.json'}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
