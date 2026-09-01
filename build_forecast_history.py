# -*- coding: utf-8 -*-
"""build_forecast_history.py — Phase 6A: append-only, immutable forecast ledger.

Captures PyCaLiAI's own (mark, p_win, p_sho) + race-state (chaos, member_level) for
every race in a date's already-built, already-scrubbed site/data/{date}.json, as a
permanent, write-once-per-race record ("what the model knew/predicted at that
moment"). Source is deliberately the already-public site/data/{date}.json, not the
raw weekly bundle or any cowork_output/live_odds/wide_residual_shadow file — those
sit in the betting layer, are mutated in place across race day, and partly carry
market-derived values (e.g. T-10 hosei_marks blend p_win with 市場π). None of that
is read here, by construction.

Idempotent / immutable: if a race's ledger file already exists, it is left
untouched. Re-running this after a later site rebuild (e.g. Sunday's Phase C, which
adds `result` to site/data/{date}.json) can never overwrite an earlier forecast
snapshot — a settled result must not contaminate the historical record.

--- Provenance (added after a review that flagged the original single-timestamp
design as unable to distinguish "this was genuinely captured before the race" from
"this was backfilled today by reading an old archived file") -----------------------

Every record carries:
  captured_at              wall-clock time THIS script wrote the record. Not a claim
                            about when the forecast was originally computed.
  provenance                "live_generation"          — this run was explicitly
                                                           invoked (--live) as part of
                                                           the real, same-day Phase-A
                                                           pipeline.
                             "archived_bundle_backfill" — the default. This run read
                                                           an already-existing
                                                           site/data/{date}.json at
                                                           some later point; the
                                                           forecast content is
                                                           accurate (verified below),
                                                           but this record is NOT
                                                           proof the Ledger itself was
                                                           locked before the race.
  source_generated_at       Best-effort estimate of when the underlying forecast
                            content actually first became public, OR null if it
                            cannot be established (never guessed/inferred without a
                            checkable basis — see source_generated_at_basis).
  source_generated_at_basis How source_generated_at was derived:
                              "git_first_commit_content_verified" — found this file's
                                  earliest git commit, AND confirmed every race's
                                  mark/p_win/p_sho in that commit is byte-identical to
                                  the current content (i.e. nothing has changed since
                                  that commit) — so that commit's timestamp is a
                                  trustworthy proxy for "when this forecast became
                                  public."
                              "content_diverged_from_earliest_known_commit" — a git
                                  history exists but the earliest commit's content for
                                  this race differs from current — cannot vouch for
                                  when the CURRENT content originated, so
                                  source_generated_at is left null rather than
                                  guessed.
                              "no_git_history" — site/data/{date}.json isn't tracked
                                  in git (or git isn't available) — null.

Output: site/data/forecast_history/{date}/{race_id}.json (one file per race).
Living under site/data/ means it ships with the existing static-site deploy
(Dockerfile COPYs the whole data/ tree) with no separate sync step.

Usage:
  python build_forecast_history.py 20260830
  python build_forecast_history.py 20260830 --dry     # report only, write nothing
  python build_forecast_history.py 20260830 --live    # mark provenance=live_generation
                                                        # (ONLY when genuinely invoked
                                                        # same-day, right after Phase A —
                                                        # never pass this for a manual/
                                                        # later run)
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(__file__).parent
SITE_DATA = BASE / "site" / "data"
HISTORY_DIR = SITE_DATA / "forecast_history"


def load_model_version(date_str: str) -> str | None:
    """モデル版数はスクレイプ前の週次バンドル(reports/cowork_input)から取る。
    site/data/{date}.json 自体にはモデル版数フィールドが無いため。"""
    bundle_path = BASE / "reports" / "cowork_input" / f"{date_str}_bundle.json"
    try:
        d = json.loads(bundle_path.read_text(encoding="utf-8"))
        return d.get("model")
    except (OSError, ValueError):
        return None


def git_first_commit(date_str: str) -> tuple[str | None, dict | None]:
    """site/data/{date}.json の最古コミット時刻と、そのコミット時点の内容を返す。
    履歴が無い/git失敗時は (None, None)。"""
    rel = f"site/data/{date_str}.json"
    try:
        log = subprocess.run(
            ["git", "log", "--follow", "--format=%H %aI", "--", rel],
            cwd=BASE, capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=30, check=True,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None, None
    lines = [l for l in (log.stdout or "").strip().splitlines() if l.strip()]
    if not lines:
        return None, None
    first_hash, first_iso = lines[-1].split(" ", 1)  # git log は新しい順 → 最終行が最古
    try:
        show = subprocess.run(
            ["git", "show", f"{first_hash}:{rel}"],
            cwd=BASE, capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=30, check=True,
        )
        first_day = json.loads(show.stdout or "")
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError, TypeError):
        return first_iso, None
    return first_iso, first_day


def _mark_tuple(r: dict) -> list:
    return [(h.get("umaban"), h.get("mark"), h.get("p_win"), h.get("p_sho"))
            for h in (r.get("horses") or [])]


def resolve_source_provenance(rid: str, r_current: dict, first_iso: str | None,
                               first_day: dict | None) -> tuple[str | None, str]:
    if first_iso is None:
        return None, "no_git_history"
    r_old = None
    if first_day:
        r_old = next((x for x in first_day.get("races", []) if x.get("race_id") == rid), None)
    if r_old is not None and _mark_tuple(r_current) == _mark_tuple(r_old):
        return first_iso, "git_first_commit_content_verified"
    return None, "content_diverged_from_earliest_known_commit"


def horse_snapshot(h: dict) -> dict:
    """公開済みフィールドのみを明示的に許可リスト方式でコピー。上流の内部行は一切読まない。"""
    return {
        "umaban": h.get("umaban"),
        "name": h.get("name"),
        "mark": h.get("mark") or "",
        "p_win": h.get("p_win"),
        "p_sho": h.get("p_sho"),
    }


def race_snapshot(r: dict, model_version: str | None, captured_at: str, provenance: str,
                   source_generated_at: str | None, source_basis: str) -> dict:
    conf = r.get("confidence") or {}
    ml = r.get("member_level") or {}
    return {
        "race_id": r.get("race_id"),
        "provenance": provenance,
        "captured_at": captured_at,
        "source_generated_at": source_generated_at,
        "source_generated_at_basis": source_basis,
        "model_version": model_version,
        "race_state": {
            "chaos": conf.get("field_chaos_score"),
            "member_level": {"tier": ml.get("tier"), "label": ml.get("label")} if ml.get("tier") else None,
        },
        "horses": [horse_snapshot(h) for h in (r.get("horses") or [])],
    }


# 万一にも紛れ込んではいけない禁止フィールド (公開JSON生成のscrub_public()と同じ精神の最終防衛線)。
_FORBIDDEN_KEYS = {
    "odds", "fuku_low", "fuku_high", "ev_tan", "ev_fuku", "vs_market",
    "training", "taiju", "taiju_diff", "umaren_odds", "market_p", "hosei_marks",
}


def assert_public_safe(obj) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in _FORBIDDEN_KEYS:
                raise ValueError(f"forecast_history: forbidden key '{k}' would be written — aborting")
            assert_public_safe(v)
    elif isinstance(obj, list):
        for v in obj:
            assert_public_safe(v)


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 6A forecast ledger (write-once per race)")
    ap.add_argument("date", help="YYYYMMDD")
    ap.add_argument("--dry", action="store_true", help="report only, write nothing")
    ap.add_argument("--live", action="store_true",
                     help="mark provenance=live_generation — ONLY for a genuine same-day "
                          "Phase-A invocation, never for a manual/later run")
    args = ap.parse_args()
    date_str = args.date

    src = SITE_DATA / f"{date_str}.json"
    if not src.exists():
        print(f"[ERROR] {src} が無い (先に build_site.py / weekly Phase A を実行)")
        return 1
    day = json.loads(src.read_text(encoding="utf-8"))
    races = day.get("races", [])
    if not races:
        print(f"[ERROR] {src} に races が無い")
        return 1

    model_version = load_model_version(date_str)
    captured_at = datetime.now().isoformat(timespec="seconds")
    provenance = "live_generation" if args.live else "archived_bundle_backfill"
    first_iso, first_day = git_first_commit(date_str)
    out_dir = HISTORY_DIR / date_str

    written, skipped = 0, 0
    basis_counts: dict[str, int] = {}
    for r in races:
        rid = r.get("race_id")
        if not rid:
            continue
        out_path = out_dir / f"{rid}.json"
        if out_path.exists():
            skipped += 1
            continue
        src_at, src_basis = resolve_source_provenance(rid, r, first_iso, first_day)
        basis_counts[src_basis] = basis_counts.get(src_basis, 0) + 1
        snap = race_snapshot(r, model_version, captured_at, provenance, src_at, src_basis)
        assert_public_safe(snap)
        if args.dry:
            print(f"[dry] would write {out_path.relative_to(BASE)}  provenance={provenance} "
                  f"source_generated_at={src_at} ({src_basis})")
        else:
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")
        written += 1

    # インデックス (この日にレコード済みの race_id 一覧)。フロントが日次1回のfetchで
    # 「このレースにFORECASTタブを出すか」を判定するための軽量マニフェスト。
    # ディレクトリの再スキャンから毎回作り直すだけなので、スナップショット本体と違い
    # 不変である必要はない (中身はどのファイルが存在するかの事実そのもの)。
    if not args.dry and out_dir.exists():
        ids = sorted(p.stem for p in out_dir.glob("*.json") if p.stem != "_index")
        (out_dir / "_index.json").write_text(
            json.dumps({"date": date_str, "race_ids": ids}, ensure_ascii=False), encoding="utf-8")

    print(f"[forecast_history] {date_str}: {written} 件新規{'(dry)' if args.dry else '書込'} / "
          f"{skipped} 件は既存のためスキップ(immutable) / provenance={provenance}")
    for basis, n in basis_counts.items():
        print(f"    source_generated_at_basis={basis}: {n} 件")
    return 0


if __name__ == "__main__":
    sys.exit(main())
