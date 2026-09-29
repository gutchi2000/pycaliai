# -*- coding: utf-8 -*-
"""
obs_guard.py — 観測計画 v2.1: 500R 到達前に性能・ROI・帯選択を実行させないための入口 guard と静的検査
==================================================================================================
1. guard: §5.3 / §6 / §7 の性能・回収率・帯選択・候補選択の集計は、計算の最初に
   `assert_performance_allowed(stream)` を呼ぶ。有効 race が 500 未満なら PermissionError で止まる。
   件数は forward store から数える（呼出側が件数を渡して通す経路は無い）。

   有効 race の数え方は label-free な部分だけ（§5.3 の有効 race 定義のうち、発売中・全組完全・
   race_key 一致・決定時点の録と final 候補の両方が存在）。同着の除外は結果を要するためここでは数えない。
   したがってこの件数は §5.3 の有効 race の上限であり、上限が 500 未満なら確実に未到達。

2. 静的検査 `find_unguarded_modules()`: repo の追跡 .py のうち
   (a) 観測 stage（trio_t10 / t2_candidate / final_rt_candidate / final_stock_candidate）を参照し、かつ
   (b) 結果・払戻・着順系を import する（generate_results 等）か、それらの path/列名の文字列を持つ
   module が、`assert_performance_allowed(...)` を呼んでいなければ違反。guard の呼出で `root=` を
   差し替えるのも違反（本番 store 以外を数えて通す経路を塞ぐ）。docstring と、FORBIDDEN/MARKER/BLOCK を
   名に含む遮断リスト定数の中の文字列は数えない。tests/ と本 module は対象外。

python -m analysis.obs_guard --check      違反があれば一覧を出して exit 1
"""
from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
import warnings
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from forward_prices import FORWARD_ROOT, canonical_stage, read_snapshot  # noqa: E402

MIN_RACES_FOR_PERFORMANCE = 500
DECISION_STAGE = {"trio": "trio_t10", "umatan": "t10", "stake": "t10"}
SPEC = {"trio": "0B35", "umatan": "0B34", "stake": None}
FINAL_STAGES = {"final_rt_candidate", "final_stock_candidate"}

OBS_STAGE_RE = re.compile(r"\b(trio_t10|t2_candidate|final_rt_candidate|final_stock_candidate|final_\w+_candidate)\b")
OUTCOME_MODULES = {"generate_results", "update_live_results", "canonical_settlement", "settle_masters_vote",
                   "jvlink_results", "backtest_pl_ev", "evaluate_wide_residual_forward"}
OUTCOME_NAMES = {"load_kekka_all", "get_winner", "get_race_kk", "parse_wide_kekka", "load_payouts",
                 "settle_bets", "normalize_result", "load_raw_results"}
OUTCOME_MODULE_RE = re.compile(r"(kekka|payout|haraimodoshi|result)", re.IGNORECASE)
OUTCOME_STRING_RE = re.compile(r"(kekka|payout|haraimodoshi|wide_payouts|live_results|払戻|着順|確定着)", re.IGNORECASE)
BLOCKLIST_NAME_RE = re.compile(r"(FORBIDDEN|MARKER|BLOCK)", re.IGNORECASE)
GUARD = "assert_performance_allowed"
SELF = "analysis/obs_guard.py"


# ---------------------------------------------------------------- 1. guard
def _ok_capture(rec: dict, spec: str | None) -> bool:
    caps = rec.get("jv_captures") or []
    if spec is None:
        return bool(caps) and all(c.get("ok") for c in caps)
    return any(c.get("spec") == spec and c.get("ok") for c in caps)


def count_label_free_valid_races(stream: str, root: Path = FORWARD_ROOT) -> int:
    if stream not in DECISION_STAGE:
        raise ValueError(f"unknown stream {stream!r}")
    dec, fin = set(), set()
    for p in Path(root).glob("*/*.json.gz"):
        if p.parent.name.startswith("_"):
            continue
        rec = read_snapshot(p)
        st = canonical_stage(rec.get("stage"))
        if st == DECISION_STAGE[stream] and _ok_capture(rec, SPEC[stream]):
            dec.add(rec.get("race_id"))
        elif st in FINAL_STAGES and _ok_capture(rec, SPEC[stream]):
            fin.add(rec.get("race_id"))
    return len(dec & fin)


def _require(n: int, stream: str) -> int:
    if n < MIN_RACES_FOR_PERFORMANCE:
        raise PermissionError(
            f"observation plan v2.1: {stream} has {n} label-free valid races (< {MIN_RACES_FOR_PERFORMANCE}). "
            "Performance, ROI, band or candidate selection must not run before 500R.")
    return n


def assert_performance_allowed(stream: str, root: Path = FORWARD_ROOT) -> int:
    """性能・ROI・帯選択の集計の入口。件数は store から数える（外から件数を渡す引数は無い）。"""
    return _require(count_label_free_valid_races(stream, root), stream)


# ---------------------------------------------------------------- 2. static check
def _docstring_ids(tree) -> set[int]:
    ids = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.body:
            first = n.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                ids.add(id(first.value))
    return ids


def _blocklist_ids(tree) -> set[int]:
    ids = set()
    for n in ast.walk(tree):
        targets = n.targets if isinstance(n, ast.Assign) else [n.target] if isinstance(n, ast.AnnAssign) else []
        if any(isinstance(t, ast.Name) and BLOCKLIST_NAME_RE.search(t.id) for t in targets) and n.value is not None:
            ids |= {id(c) for c in ast.walk(n.value) if isinstance(c, ast.Constant)}
    return ids


def _data_like(s: str) -> bool:
    """path・列名・識別子として使われる文字列だけを数える（空白を含む文章は数えない。
    ただし path らしいもの＝区切り文字や拡張子付きは空白があっても数える）。"""
    return (not re.search(r"\s", s)) or bool(re.search(r"[/\\]|\.(csv|json|parquet|gz|txt)\b", s))


def analyze_source(src: str) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tree = ast.parse(src)
    skip = _docstring_ids(tree) | _blocklist_ids(tree)
    strings = [n.value for n in ast.walk(tree)
               if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in skip]
    mods, names = set(), set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            mods.add(n.module or "")
            names |= {a.name for a in n.names}
    mods |= {m.split(".")[-1] for m in mods}
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and (getattr(n.func, "id", None) == GUARD or getattr(n.func, "attr", None) == GUARD)]
    dynamic = any(isinstance(n, ast.Call) and (getattr(n.func, "id", None) == "__import__"
                                               or getattr(n.func, "attr", None) == "import_module")
                  and any(isinstance(a, ast.Constant) and isinstance(a.value, str)
                          and (a.value.split(".")[-1] in OUTCOME_MODULES or OUTCOME_MODULE_RE.search(a.value))
                          for a in n.args) for n in ast.walk(tree))
    outcome_import = bool(mods & OUTCOME_MODULES or names & OUTCOME_NAMES
                          or any(OUTCOME_MODULE_RE.search(m) for m in mods if m) or dynamic)
    return {"obs_stage_refs": sorted({m.group(1) for s in strings for m in OBS_STAGE_RE.finditer(s)}),
            "outcome_import": outcome_import,
            "outcome_strings": sorted({s[:40] for s in strings if OUTCOME_STRING_RE.search(s) and _data_like(s)})[:5],
            "guard_calls": len(calls),
            "guard_root_override": any(k.arg == "root" for c in calls for k in c.keywords)}


def violation(info: dict) -> str | None:
    if not info["obs_stage_refs"] or not (info["outcome_import"] or info["outcome_strings"]):
        return None
    if info["guard_calls"] == 0:
        return "references observation stages and outcome/payout sources without assert_performance_allowed()"
    if info["guard_root_override"]:
        return "assert_performance_allowed() called with root= (store override is not allowed)"
    return None


def tracked_python_files(base: Path | None = None) -> list[Path]:
    base = BASE if base is None else base
    try:
        out = subprocess.run(["git", "-c", "safe.directory=*", "ls-files", "-z", "--", "*.py"], cwd=base,
                             capture_output=True, check=True).stdout.decode("utf-8")
        files = [base / p for p in out.split("\0") if p]
    except Exception:
        files = [p for p in base.rglob("*.py")
                 if not any(part in (".git", "__pycache__", "node_modules") or part.startswith("venv")
                            for part in p.relative_to(base).parts)]
    return [p for p in files if p.exists()]


def find_unguarded_modules(base: Path | None = None, files: list[Path] | None = None) -> list[dict]:
    base = BASE if base is None else base
    out = []
    for p in (files if files is not None else tracked_python_files(base)):
        rel = p.relative_to(base).as_posix()
        if rel == SELF or rel.startswith("tests/"):
            continue
        try:
            info = analyze_source(p.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError, ValueError):
            continue
        why = violation(info)
        if why:
            out.append({"module": rel, "why": why, **info})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="静的検査を実行し違反があれば exit 1")
    args = ap.parse_args()
    if not args.check:
        ap.error("--check を指定")
    bad = find_unguarded_modules()
    for b in bad:
        print(f"[obs_guard] VIOLATION {b['module']}: {b['why']} (stages={b['obs_stage_refs']})")
    print(f"[obs_guard] checked tracked python files; violations={len(bad)}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
