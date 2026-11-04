# -*- coding: utf-8 -*-
"""
serve_path.py — 予測経路 (serve path) の版の記録 (単一ソース)
================================================================
bundle を作った予測経路を後から特定できるように、版 ID と入力資産の hash を
bundle 最上位の `serve_path` に刻む。下流 (compute_bets の run stamp、
shadow_a95 の decisions、production_policy の stamp) は同じ版 ID を写す。

  版 ID   : serve_history_feats.SERVE_PATH_VERSION (挙動を変えたら必ず上げる)
  旧記録 : 版 ID の無い bundle・決定記録は LEGACY_VERSION (sp-20261004-0) とみなす。
           後から書き換えず、この対応で読む (docs/research/SERVE_REPRODUCIBILITY_FIX_SPEC_20261004.md §7)。

記録項目: 版 ID、git commit、履歴 parquet の sha256・行数・最大日付・列構成、
直近集計の順序キー、馬同定の方式、名前の正規化、補正タイム推定の有無、
モデル・較正器・騎手/調教師コード表 (serve_code_maps.json) の sha256。
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

BASE = Path(__file__).resolve().parent
LEGACY_VERSION = "sp-20261004-0"


def sha256_file(path: Path) -> str | None:
    path = Path(path)
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_commit(base: Path = BASE) -> dict:
    """HEAD と、追跡ファイルに未 commit の変更があるか。git が使えなければ None。"""
    def _git(*args: str) -> str | None:
        try:
            r = subprocess.run(["git", "-c", "safe.directory=*", *args], cwd=str(base),
                               capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.SubprocessError):
            return None
        return r.stdout.strip() if r.returncode == 0 else None

    head = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain", "-uno", "--", "*.py")
    return {"commit": head, "py_dirty": (None if dirty is None else bool(dirty))}


def _artifact(path: Path, base: Path) -> dict:
    path = Path(path)
    try:
        rel = path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        rel = str(path)
    return {"path": rel, "sha256": sha256_file(path)}


def build_record(base: Path = BASE, *, model_path: Path | None = None,
                 calibrator_path: Path | None = None, hosei_proxy: bool = False,
                 history_stats: dict | None = None) -> dict:
    """bundle 最上位に置く `serve_path` レコードを作る。"""
    import serve_history_feats as shf

    base = Path(base)
    # 実際に読んだファイルを hash する (fill_history_features の統計に経路が載る)。
    # 履歴特徴が失敗して統計が無いときは既定の置き場所を記録する。
    stats = history_stats or {}
    hist_path = Path(stats.get("hist_path") or (base / "data" / "_horse_history.parquet"))
    maps_path = Path(stats.get("maps_path") or (base / "data" / "serve_code_maps.json"))
    hist = _artifact(hist_path, base)
    hist["loaded"] = bool(history_stats)
    if history_stats:
        hist["rows"] = history_stats.get("hist_rows")
        hist["max_date"] = history_stats.get("hist_max_date")
        hist["columns"] = history_stats.get("hist_columns")
    rec = {
        "version": shf.SERVE_PATH_VERSION,
        "git": git_commit(base),
        "history": hist,
        "rolling_order_key": shf.ROLLING_ORDER_KEY,
        "identity_mode": shf.IDENTITY_MODE,
        "name_normalization": shf.NAME_NORMALIZATION,
        "hosei_proxy": bool(hosei_proxy),
        "code_maps": _artifact(maps_path, base),
    }
    if model_path is not None:
        rec["model"] = _artifact(model_path, base)
    if calibrator_path is not None:
        rec["calibrator"] = _artifact(calibrator_path, base)
    return rec


def record_sha256(rec: dict | None) -> str | None:
    """serve_path レコードの正規化 JSON の sha256 (run stamp から bundle の記録へ辿る鍵)。"""
    if not rec:
        return None
    raw = json.dumps(rec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def version_of(bundle: dict | None) -> str:
    """bundle (または serve_path レコード) の版 ID。記録の無い旧 bundle は LEGACY_VERSION。"""
    if not isinstance(bundle, dict):
        return LEGACY_VERSION
    rec = bundle.get("serve_path") if "serve_path" in bundle else bundle
    if isinstance(rec, dict) and rec.get("version"):
        return str(rec["version"])
    return LEGACY_VERSION


def stamp_fields(bundle: dict | None) -> dict:
    """compute_bets の run stamp / shadow_a95 の decisions へ写す 2 項目。"""
    rec = bundle.get("serve_path") if isinstance(bundle, dict) else None
    return {"serve_path_version": version_of(bundle),
            "serve_path_sha256": record_sha256(rec) if isinstance(rec, dict) else None}
