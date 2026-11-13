# -*- coding: utf-8 -*-
"""
bundle_provenance.py — 公開・採点に使う bundle の来歴 (2026-11-13 監査 P1-5 の修正)
======================================================================================
監査で分かったこと: reports/cowork_input/{date}_bundle.json は発走後にも作り直されることがあり
(0517: ◎ 9R、0620/0921: ◎ 4R が発走後に変化)、サイトは毎回いまある bundle から過去日を
作り直すため、過去の成績が後から書き換わっていた。モデル sha を刻んでいる bundle は 2/61 件。

ここでは 3 つを提供する:
  1. freeze: 開催日の最初の発走より前に作られた bundle を reports/cowork_input_frozen/ に
     写す。発走前なら最新版で置き換え、最初の発走を過ぎたら二度と書き換えない。
     live の reports/cowork_input/ は従来どおり (T-10 などが読む)。
  2. resolve_for_scoring: サイトの採点・表示に使う bundle。凍結版があればそれ、無ければ live に
     「未検証」の印を付けて返す。
  3. model_era: その日の予測を作ったモデルの世代 (v5 / v6-C0 / v6-C1 / 不明)。
     bundle の serve_path.model.sha256 があればそれで、無い旧 bundle は記録済みの切替日で決める
     (docs/research/PROD_MODEL_GENERATION_REGRESSION_20261021.md)。
"""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent
LIVE_DIR = BASE / "reports" / "cowork_input"
FROZEN_DIR = BASE / "reports" / "cowork_input_frozen"
SNAP_DIR = BASE / "reports" / "serve_path_snapshots"

# モデル sha → 世代 (artifact_pins と退行報告の記録)
MODEL_SHA_ERA = {
    "db2ffe65ee27fb9014dcd78724851e77d292fcaf5d77aa33e40b83f3b186d0d5": "v6-C1",
    "0a040c4e": "v6-C0",   # 先頭一致で照合
}
# sha を刻んでいない旧 bundle 用。開催日 (YYYYMMDD) の範囲 → 世代。
# v6 本番化 0517 / C1 切替 0911 (0912 開催から) / 0920 10:42 に C0 へ退行 (0921 開催から) /
# 1022 に C1 再配置 (1024 開催から)。0920 開催の bundle は 0919 23:13 作成で C1 側。
DATE_ERAS = [
    ("00000000", "20260516", "v5"),
    ("20260517", "20260911", "v6-C0"),
    ("20260912", "20260920", "v6-C1"),
    ("20260921", "20261023", "v6-C0"),
    ("20261024", "99999999", "v6-C1"),
]
CURRENT_ERA = "v6-C1"


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def model_era(date_str: str, bundle: dict | None) -> str:
    b = bundle or {}
    if str(b.get("model") or "").lower() == "v5":
        return "v5"
    sha = (((b.get("serve_path") or {}).get("model") or {}).get("sha256") or "")
    for k, era in MODEL_SHA_ERA.items():
        if sha and sha.startswith(k):
            return era
    if sha:
        return "unknown"
    for lo, hi, era in DATE_ERAS:
        if lo <= date_str <= hi:
            return era
    return "unknown"


def first_post_time(date_str: str, post_times=None) -> datetime:
    """その開催日の最初の発走時刻。分からなければ 09:30 (JRA の第 1R はそれより後)。"""
    best = None
    for t in post_times or []:
        s = str(t).strip().replace("：", ":")
        if not s or s.lower() == "nan":
            continue
        try:
            hh, mm = s.split(":")[:2]
            dt = datetime.strptime(f"{date_str}{int(hh):02d}{int(mm):02d}", "%Y%m%d%H%M")
        except Exception:
            continue
        best = dt if best is None or dt < best else best
    return best or datetime.strptime(date_str + "0930", "%Y%m%d%H%M")


def freeze_if_pre_race(date_str: str, bundle_path: Path, first_post: datetime,
                       now: datetime | None = None, frozen_dir: Path | None = None) -> dict:
    """発走前なら凍結版を作る/更新する。発走後は何もしない (凍結版は不変)。"""
    now = now or datetime.now()
    fdir = Path(frozen_dir or FROZEN_DIR)
    dst = fdir / f"{date_str}_bundle.json"
    side = fdir / f"{date_str}_bundle.provenance.json"
    if now >= first_post:
        return {"frozen": False, "reason": "after_first_post",
                "first_post": first_post.isoformat(timespec="minutes"), "frozen_exists": dst.exists()}
    fdir.mkdir(parents=True, exist_ok=True)
    raw = Path(bundle_path).read_bytes()
    tmp = dst.with_suffix(".json.tmp")
    tmp.write_bytes(raw)
    tmp.replace(dst)
    meta = {"date": date_str, "frozen_at": now.isoformat(timespec="seconds"),
            "first_post": first_post.isoformat(timespec="minutes"), "sha256": _sha256_bytes(raw),
            "source": "export_weekly_marks (pre-race)", "verified_pre_race": True}
    side.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"frozen": True, **meta}


def resolve_for_scoring(date_str: str, live_dir: Path | None = None,
                        frozen_dir: Path | None = None) -> tuple[Path | None, dict]:
    """採点・公開に使う bundle と来歴。凍結版があれば優先。"""
    ldir, fdir = Path(live_dir or LIVE_DIR), Path(frozen_dir or FROZEN_DIR)
    f = fdir / f"{date_str}_bundle.json"
    side = fdir / f"{date_str}_bundle.provenance.json"
    live = ldir / f"{date_str}_bundle.json"
    if f.exists():
        meta = {}
        if side.exists():
            try:
                meta = json.loads(side.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        meta.setdefault("verified_pre_race", True)
        meta["source_path"] = "frozen"
        meta["differs_from_live"] = live.exists() and live.read_bytes() != f.read_bytes()
        return f, meta
    if live.exists():
        return live, {"verified_pre_race": False, "source_path": "live",
                      "reason": "発走前の版が残っていない (凍結導入前・git に発走前の commit が無い)"}
    return None, {}


def snapshot_serve_inputs(date_str: str, history_path: Path, git_info: dict,
                          base: Path | None = None, snap_dir: Path | None = None) -> dict:
    """過去日を作り直せるよう、履歴 parquet と未 commit の .py 差分を保存する。"""
    import subprocess
    base = Path(base or BASE)
    sdir = Path(snap_dir or SNAP_DIR)
    out: dict = {}
    hp = Path(history_path)
    if hp.is_file():
        raw = hp.read_bytes()
        sha = _sha256_bytes(raw)
        dst = sdir / "history" / f"{sha[:16]}.parquet"
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(hp, dst)
        out["history_snapshot"] = dst.relative_to(base).as_posix() if dst.is_relative_to(base) else str(dst)
    if git_info.get("py_dirty"):
        try:
            r = subprocess.run(["git", "-c", "safe.directory=*", "diff", "HEAD", "--", "*.py"],
                               cwd=str(base), capture_output=True, timeout=60)
            patch = r.stdout
            files = subprocess.run(["git", "-c", "safe.directory=*", "diff", "--name-only", "HEAD", "--", "*.py"],
                                   cwd=str(base), capture_output=True, text=True, timeout=60).stdout.split()
            psha = _sha256_bytes(patch)
            dst = sdir / "py_diff" / f"{date_str}_{psha[:16]}.patch"
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                dst.write_bytes(patch)
            out.update({"py_diff_sha256": psha, "py_diff_files": files,
                        "py_diff_snapshot": dst.relative_to(base).as_posix() if dst.is_relative_to(base) else str(dst)})
        except Exception as e:  # noqa: BLE001
            out["py_diff_error"] = f"{type(e).__name__}: {e}"
    return out
