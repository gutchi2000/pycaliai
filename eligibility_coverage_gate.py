# -*- coding: utf-8 -*-
"""
eligibility_coverage_gate.py — bundle 生成前の完全性 Gate（transactional fail）
==============================================================================
fail-closed とは「空の成果物を公開すること」ではない。
**処理を非 0 で終了させ、直前の正常成果物をそのまま残すこと**である。

当日の「予定 race 集合」(weekly/calendar) と
「eligibility evidence 集合」(bunseki) を突き合わせ、1 件でも欠ければ
その日の bundle 生成・push・task 登録を**すべて中止**する。

必須条件:
  - scheduled race coverage = 100%
  - race_id 重複 = 0
  - authoritative track code coverage = 100%
  - 許可済み source のみ (default / missing / 未指定は不可)
  - unknown race = 0
  - flat/jump classification disagreement = 0

使い方:
    from eligibility_coverage_gate import check_coverage, GateResult
    res = check_coverage(date_str, scheduled_rids)
    if not res.ok:
        res.report(); return res.exit_code      # bundle を作らない
"""
from __future__ import annotations

import hashlib
import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).parent
LOG_DIR = BASE / "logs"
ERROR_LOG = LOG_DIR / "eligibility_gate_error.log"

EXIT_EVIDENCE_INCOMPLETE = 3
ERR_CODE = "ELIGIBILITY_EVIDENCE_INCOMPLETE"

# 証拠として許可する provenance
ALLOWED_SOURCES = {"raw_jv", "bunseki", "weekly_explicit", "calendar"}

logger = logging.getLogger(__name__)


def _sha256(p: Path) -> str | None:
    if not p or not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


@dataclass
class GateResult:
    ok: bool
    date: str
    error_code: str | None = None
    exit_code: int = 0
    expected: int = 0
    observed: int = 0
    missing: int = 0
    unknown_rids: list[str] = field(default_factory=list)
    missing_rids: list[str] = field(default_factory=list)
    duplicate_rids: list[str] = field(default_factory=list)
    bad_source_rids: list[dict] = field(default_factory=list)
    disagreement_rids: list[dict] = field(default_factory=list)
    files: dict = field(default_factory=dict)
    jump_rids: list[str] = field(default_factory=list)
    flat_rids: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "ok": self.ok, "date": self.date, "error_code": self.error_code,
            "exit_code": self.exit_code,
            "expected_races": self.expected, "observed_races": self.observed,
            "missing_races": self.missing,
            "unknown_race_ids": self.unknown_rids,
            "missing_race_ids": self.missing_rids,
            "duplicate_race_ids": self.duplicate_rids,
            "bad_source_races": self.bad_source_rids,
            "classification_disagreements": self.disagreement_rids,
            "files": self.files,
            "n_jump": len(self.jump_rids), "n_flat": len(self.flat_rids),
        }

    def report(self) -> None:
        print("=" * 72, flush=True)
        print(f"✗ {self.error_code} — {self.date} の bundle を生成しない", flush=True)
        print("=" * 72, flush=True)
        print(f"  expected race : {self.expected}", flush=True)
        print(f"  observed race : {self.observed}", flush=True)
        print(f"  missing race  : {self.missing}", flush=True)
        if self.unknown_rids:
            print(f"  unknown race_id ({len(self.unknown_rids)}): "
                  f"{self.unknown_rids}", flush=True)
        if self.missing_rids:
            print(f"  evidence 欠落 race_id ({len(self.missing_rids)}): "
                  f"{self.missing_rids}", flush=True)
        if self.duplicate_rids:
            print(f"  重複 race_id: {self.duplicate_rids}", flush=True)
        if self.bad_source_rids:
            print(f"  不許可 source: {self.bad_source_rids[:5]}", flush=True)
        if self.disagreement_rids:
            print(f"  flat/jump 不一致: {self.disagreement_rids}", flush=True)
        for k, v in self.files.items():
            print(f"  {k}: {v}", flush=True)
        print("  → production bundle は作成も上書きもしない。"
              "site/cowork 出力・task 登録も行わない。", flush=True)
        print("  → 復旧: TARGET「出走馬分析」を export し data/_inbox へ置いて "
              "place_weekly.py を実行すること。", flush=True)
        self._write_error_log()

    def _write_error_log(self) -> None:
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            rec = dict(self.as_dict(),
                       at=datetime.now().isoformat(timespec="seconds"))
            with open(ERROR_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:                               # pragma: no cover
            logger.warning("error log 書込失敗: %s", e)


def check_coverage(date_str: str, scheduled_rids,
                   weekly_path: Path | None = None) -> GateResult:
    """予定 race 集合 vs eligibility evidence 集合。"""
    from race_eligibility import evaluate_race, clear_cache
    clear_cache()

    rids = [str(r)[:16] for r in scheduled_rids]
    dup = [r for r, c in Counter(rids).items() if c > 1]
    uniq = sorted(set(rids))

    bunseki_p = BASE / "data" / "bunseki" / f"{date_str}.csv"
    weekly_p = weekly_path or (BASE / "data" / "weekly" / f"{date_str}.csv")
    bias_p = BASE / "data" / "bias" / f"{date_str}.csv"
    files = {
        "bunseki_path": str(bunseki_p),
        "bunseki_exists": bunseki_p.exists(),
        "bunseki_sha256": _sha256(bunseki_p),
        "weekly_path": str(weekly_p),
        "weekly_sha256": _sha256(weekly_p),
        "bias_sha256": _sha256(bias_p),
    }

    unknown, missing, bad_src, disagree = [], [], [], []
    jump, flat = [], []

    for rid in uniq:
        el = evaluate_race(rid)
        ev = el["raw_fields"]
        src = ev["track_code_source"]

        if el["determination"] == "unknown":
            unknown.append(rid)
            missing.append(rid)
            continue
        if el["determination"] == "conflict":
            disagree.append({"race_id": rid,
                             "track_code": ev["track_code_value"],
                             "flat_jump": ev["flat_jump_value"]})
            continue
        # authoritative track code が無い (平・障 だけで判定された) 場合も不足扱い
        if ev["track_code_value"] is None or src not in ALLOWED_SOURCES:
            bad_src.append({"race_id": rid, "track_code_source": src,
                            "track_code_value": ev["track_code_value"]})
            missing.append(rid)
            continue
        (jump if el["is_jump"] else flat).append(rid)

    ok = not (unknown or missing or dup or bad_src or disagree)
    res = GateResult(
        ok=ok, date=date_str,
        error_code=None if ok else ERR_CODE,
        exit_code=0 if ok else EXIT_EVIDENCE_INCOMPLETE,
        expected=len(uniq), observed=len(jump) + len(flat),
        missing=len(set(missing)),
        unknown_rids=unknown, missing_rids=sorted(set(missing)),
        duplicate_rids=dup, bad_source_rids=bad_src,
        disagreement_rids=disagree, files=files,
        jump_rids=jump, flat_rids=flat,
    )
    return res


# ------------------------------------------------------------------
# atomic publish
# ------------------------------------------------------------------

class AtomicPublish:
    """temp へ全成果物を作り、全 Gate PASS 後にだけ本番へ差し替える。

        with AtomicPublish(out_dir, bundle_path) as pub:
            ... pub.tmp_dir / f"{rid}.json" へ書く ...
            ... pub.tmp_bundle へ書く ...
            if gate_errors: raise GateFailure()
            pub.commit()
        # commit されなければ temp は破棄され、本番は無傷
    """

    def __init__(self, out_dir: Path, bundle_path: Path):
        self.out_dir = Path(out_dir)
        self.bundle_path = Path(bundle_path)
        self.tmp_dir = self.out_dir.parent / f".{self.out_dir.name}__tmp"
        self.tmp_bundle = self.bundle_path.with_name(
            f".{self.bundle_path.name}__tmp")
        self.committed = False

    def __enter__(self) -> "AtomicPublish":
        import shutil
        if self.tmp_dir.exists():
            shutil.rmtree(self.tmp_dir, ignore_errors=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        return self

    def commit(self) -> None:
        """bundle を atomic に差し替え、個別 JSON を本番ディレクトリへ移す。"""
        import os
        import shutil
        self.out_dir.mkdir(parents=True, exist_ok=True)
        for p in sorted(self.tmp_dir.glob("*.json")):
            os.replace(str(p), str(self.out_dir / p.name))
        if self.tmp_bundle.exists():
            os.replace(str(self.tmp_bundle), str(self.bundle_path))
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        self.committed = True

    def abort(self) -> None:
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        try:
            if self.tmp_bundle.exists():
                self.tmp_bundle.unlink()
        except OSError:
            pass

    def __exit__(self, exc_type, exc, tb) -> bool:
        if not self.committed:
            self.abort()
        return False


def main() -> int:
    """単体実行: python eligibility_coverage_gate.py --date YYYYMMDD"""
    import argparse
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--weekly")
    a = ap.parse_args()

    from build_horse_history import parse_weekly_light
    wp = Path(a.weekly) if a.weekly else (
        BASE / "data" / "weekly" / f"{a.date}.csv")
    w = parse_weekly_light(wp)
    rids = sorted(set(w["rid16"])) if not w.empty else []

    res = check_coverage(a.date, rids, weekly_path=wp)
    if not res.ok:
        res.report()
        return res.exit_code
    print(f"✓ eligibility coverage OK: {a.date} "
          f"expected={res.expected} flat={len(res.flat_rids)} "
          f"jump={len(res.jump_rids)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
