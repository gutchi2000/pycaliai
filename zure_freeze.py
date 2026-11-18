"""ズレ警告の前向き確認用 凍結保存（2026-11-18、仕様 docs/research/ZURE_VALIDATION_SPEC_20261118.md）。

Phase A（その開催日の予想を公開する直前）に、公開されるのと同じ値を 1 回だけ書き、以後は書き換えない。

  reports/zure_freeze/{date}.json          公開サイトに出る値だけ（PW・順位・同率フラグ、モデル順位、馬レベル、
                                           スピード指数、出世予測、bundle の sha256、パラメータとコードの版）。git に入れてよい
  reports/zure_freeze/{date}.shap.json.gz  全馬の pred_contrib（特徴ごとの寄与）。JRDB 由来の特徴の寄与を含むので
                                           ローカル専用（.gitignore）。{date}.json に sha256 を刻む
  reports/zure_freeze/_contrib/{date}.json.gz  export_weekly_marks.py が毎回書く作業用ファイル（凍結物ではない）

  python zure_freeze.py --date 20261121                 # Phase A から（発走前でなければ何も書かない）
  python zure_freeze.py --date 20261115 --allow-late --out-dir <tmp>   # テスト用（本番の場所には書けない）

書き込みは排他作成（既にあれば何もしない）で、書いた後は読み取り専用にする。
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import stat
import subprocess
import sys
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent
FREEZE_DIR = BASE / "reports" / "zure_freeze"
CONTRIB_DIR = FREEZE_DIR / "_contrib"
VERSION = "zf-20261118-1"
GRADED = {"G1", "G2", "G3"}
CODE_FILES = ("zure_freeze.py", "power_rating.py", "speed_index.py", "build_site.py",
              "export_weekly_marks.py", "export_marks_json.py", "analysis/zure_validation.py")


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha_file(p: Path) -> str | None:
    return _sha(p.read_bytes()) if p.exists() else None


# ------------------------------------------------------------------ export から呼ぶ
def write_contrib_sidecar(date: str, bundle_path, feats: list[str], sink: dict, tag: str) -> Path:
    """export_weekly_marks.py が bundle を公開した直後に呼ぶ。作業用なので毎回上書きしてよい。"""
    cdir = Path(os.environ.get("CB_ZURE_CONTRIB_DIR") or CONTRIB_DIR)   # テスト用の置き場所
    cdir.mkdir(parents=True, exist_ok=True)
    errors = sink.get("_errors") or []
    races = {k: v for k, v in sink.items() if k != "_errors"}
    doc = {"date": date, "model": tag, "bundle_sha256": _sha_file(Path(bundle_path)),
           "feature_cols": feats, "contrib_layout": "feature_cols の順 + 末尾に定数項",
           "written_at": datetime.now().isoformat(timespec="seconds"), "errors": errors, "races": races}
    p = cdir / f"{date}.json.gz"
    tmp = p.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False)
    os.replace(tmp, p)
    return p


# ------------------------------------------------------------------ 凍結
def _git_info() -> dict:
    def run(*a):
        r = subprocess.run(["git", *a], cwd=BASE, capture_output=True, text=True, encoding="utf-8")
        return r.stdout.strip() if r.returncode == 0 else None
    dirty = run("status", "--porcelain", "--", *CODE_FILES)
    return {"commit": run("rev-parse", "HEAD"), "code_files_dirty": bool(dirty),
            "dirty_files": [ln[3:] for ln in (dirty or "").splitlines()],
            "code_sha256": {f: _sha_file(BASE / f) for f in CODE_FILES}}


def _rank_desc(vals: dict[int, float]) -> dict[int, int]:
    xs = sorted(vals.values(), reverse=True)
    return {u: 1 + sum(1 for x in xs if x > v) for u, v in vals.items()}


def _first_post(date: str) -> tuple[datetime, str]:
    prov = BASE / "reports" / "cowork_input_frozen" / f"{date}_bundle.provenance.json"
    if prov.exists():
        try:
            return datetime.fromisoformat(json.loads(prov.read_text(encoding="utf-8"))["first_post"]), "frozen_provenance"
        except Exception:  # noqa: BLE001
            pass
    from bundle_provenance import first_post_time
    import build_site
    race_extra, _ = build_site.parse_weekly(date)
    times = [v.get("start_time") for v in race_extra.values()]
    return first_post_time(date, times), ("weekly_csv" if any(times) else "default_0930")


def _pick_bundle(date: str) -> tuple[Path | None, str]:
    frozen = BASE / "reports" / "cowork_input_frozen" / f"{date}_bundle.json"
    live = BASE / "reports" / "cowork_input" / f"{date}_bundle.json"
    if frozen.exists():
        return frozen, "cowork_input_frozen"
    if live.exists():
        return live, "cowork_input"
    return None, "missing"


def _ratings_csv() -> dict:
    import csv
    out = {}
    p = BASE / "reports" / "power_rating" / "ratings.csv"
    if not p.exists():
        return out
    with open(p, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            out[row["馬名"]] = row
    return out


def build_freeze(date: str, bundle_override: Path | None = None,
                 contrib_dir: Path = CONTRIB_DIR) -> tuple[dict, dict | None]:
    """凍結する中身（公開できる値）と、SHAP の中身（ローカル専用）を作る。書き込みはしない。"""
    import build_site
    from analysis.zure_validation import load_shusse, shusse_pg12
    import pandas as pd

    bpath, bsrc = (Path(bundle_override), "override(test)") if bundle_override else _pick_bundle(date)
    if bpath is None:
        raise FileNotFoundError(f"bundle が無い: {date}")
    braw = bpath.read_bytes()
    bundle = json.loads(braw.decode("utf-8"))
    live = BASE / "reports" / "cowork_input" / f"{date}_bundle.json"

    pt = build_site._power_tables()
    pmeta_p = BASE / "reports" / "power_rating" / "meta.json"
    pmeta = json.loads(pmeta_p.read_text(encoding="utf-8")) if pmeta_p.exists() else {}
    hist_has_day = date in pt["hist"]
    ratings = _ratings_csv()
    speed = build_site.load_speed_day(date)
    speed_p = BASE / "reports" / "speed_index" / f"day_{date}.json"
    by_ns, by_n = load_shusse()
    from analysis.zure_validation import SHUSSE_PRED
    dt = pd.Timestamp(date)

    races, post_race_pw = [], []
    for r in bundle.get("races", []):
        meta = r.get("race_meta") or {}
        rid = str(r.get("race_id"))[:16]
        course = meta.get("course") or ""
        surf = course[:1]
        rno = int(rid[-2:]) if rid[-2:].isdigit() else None
        klass = meta.get("class") or ""
        hs = r.get("horses") or []
        rows = []
        for h in hs:
            u = int(h["umaban"])
            name = h.get("horse_name") or ""
            score = build_site.power_score(date, name, surf)
            rr = ratings.get(name) or {}
            raw = None
            if hist_has_day:
                v = pt["hist"][date].get(name)
                raw = v[1] if v and v[0] == surf else None
            elif rr.get(f"rating_{surf}"):
                raw = float(rr[f"rating_{surf}"])
                last = rr.get(f"最終_{surf}") or ""
                if last and last.replace("-", "") >= date:
                    post_race_pw.append(f"{rid}:{u}")
            sp = speed.get((meta.get("place"), rno, u)) if rno else None
            lv = build_site.horse_level(h.get("history"))
            pg = shusse_pg12(by_ns, by_n, name, (h.get("pedigree") or {}).get("sire"), dt)
            rows.append({"umaban": u, "horse_name": name, "ai_rank": h.get("ai_rank"), "p_win": h.get("p_win"),
                         "mark": h.get("mark") or "", "pw": score, "pw_raw": raw,
                         "level": (lv or {}).get("score"), "speed": (sp or {}).get("v"),
                         "speed_rank_site": (sp or {}).get("rk"), "pg12": pg})
        pw_rk = _rank_desc({x["umaban"]: x["pw"] for x in rows if x["pw"] is not None})
        lv_rk = _rank_desc({x["umaban"]: x["level"] for x in rows if x["level"] is not None})
        sp_rk = _rank_desc({x["umaban"]: x["speed"] for x in rows if x["speed"] is not None})
        n_top = sum(1 for v in pw_rk.values() if v == 1)
        pws = sorted((x["pw"] for x in rows if x["pw"] is not None), reverse=True)
        margin = round(pws[0] - pws[1], 3) if len(pws) >= 2 else None
        for x in rows:
            u = x["umaban"]
            x["pw_rank"] = pw_rk.get(u)
            x["pw_top_tied"] = bool(pw_rk.get(u) == 1 and n_top > 1)
            x["level_rank"] = lv_rk.get(u)
            x["speed_rank"] = sp_rk.get(u)
            low = (x["ai_rank"] or 0) >= 6
            # 凍結時点の出走表での判定（参考）。採点は発売前取消を除いて付け直す（仕様どおり）
            x["flags_at_freeze"] = {
                "D1": bool(low and (x["level_rank"] or 99) <= 3),
                "D3p": bool(low and x["pw_rank"] == 1 and n_top == 1),
                "D4": bool(low and x["speed_rank"] == 1),
                "D5": bool(low and klass in GRADED and (x["pg12"] or 0) >= 0.5),
            }
        races.append({"race_id": rid, "place": meta.get("place"), "rno": rno, "course": course, "class": klass,
                      "field_size": meta.get("field_size"), "pw_top_count": n_top, "pw_margin_1_2": margin,
                      "horses": rows})

    # SHAP（pred_contrib）: export が書いた作業用ファイルが、この bundle と同じ版のものかを確かめる
    shap_doc, shap_status = None, "missing"
    cp = Path(contrib_dir) / f"{date}.json.gz"
    if cp.exists():
        with gzip.open(cp, "rt", encoding="utf-8") as f:
            c = json.load(f)
        if c.get("bundle_sha256") == _sha(braw):
            shap_doc, shap_status = c, "ok"
        elif c.get("bundle_sha256") == _sha_file(live):
            shap_doc, shap_status = c, "ok_live_bundle_differs_from_frozen"
        else:
            shap_status = "bundle_sha_mismatch"

    doc = {
        "version": VERSION, "date": date,
        "bundle": {"path": f"{bsrc}/{bpath.name}", "sha256": _sha(braw), "live_sha256": _sha_file(live),
                   "model": bundle.get("model"),
                   "serve_path_version": (bundle.get("serve_path") or {}).get("version"),
                   "model_sha256": ((bundle.get("serve_path") or {}).get("model") or {}).get("sha256")},
        "power_rating": {"meta_generated_at": pmeta.get("generated_at"), "asof": pmeta.get("asof"),
                         "params": pmeta.get("params"), "anchors": pmeta.get("anchors"),
                         "source": "history.csv R前" if hist_has_day else "ratings.csv（最新）",
                         "post_race_contamination": post_race_pw,
                         "ratings_sha256": _sha_file(BASE / "reports" / "power_rating" / "ratings.csv")},
        "speed_index": {"day_file_sha256": _sha_file(speed_p),
                        "version": (json.loads(speed_p.read_text(encoding="utf-8")).get("version")
                                    if speed_p.exists() else None)},
        "shusse": {"predictions": SHUSSE_PRED.name, "sha256": _sha_file(SHUSSE_PRED),
                   "rule": "base_date < 開催日 かつ base_date + 365 日 > 開催日 の最新の基準点の p_graded12"},
        "shap": {"status": shap_status},
        "code": _git_info(),
        "races": races,
    }
    return doc, shap_doc


def _write_once(path: Path, data: bytes) -> bool:
    """排他作成。既にあれば何もしない（False）。書いたら読み取り専用にする。"""
    try:
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0))
    except FileExistsError:
        return False
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    os.chmod(path, stat.S_IREAD)
    return True


def freeze(date: str, out_dir: Path = FREEZE_DIR, now: datetime | None = None, allow_late: bool = False,
           bundle_override: Path | None = None, contrib_dir: Path = CONTRIB_DIR) -> dict:
    out_dir = Path(out_dir)
    if (allow_late or bundle_override) and out_dir.resolve() == FREEZE_DIR.resolve():
        raise ValueError("--allow-late / --bundle はテスト用。本番の凍結場所には使えない（--out-dir に別の場所を指定する）")
    jpath, spath = out_dir / f"{date}.json", out_dir / f"{date}.shap.json.gz"
    if jpath.exists():
        return {"written": False, "reason": "already_frozen", "path": str(jpath)}
    now = now or datetime.now()
    fp, fp_src = _first_post(date)
    pre_race = now < fp
    if not pre_race and not allow_late:
        return {"written": False, "reason": "after_first_post", "first_post": fp.isoformat(timespec="minutes")}
    doc, shap_doc = build_freeze(date, bundle_override, contrib_dir)
    doc.update({"frozen_at": now.isoformat(timespec="seconds"), "first_post": fp.isoformat(timespec="minutes"),
                "first_post_source": fp_src, "pre_race": pre_race,
                "note": None if pre_race else "テスト用（--allow-late）。発走後に作ったので前向き確認には使わない"})
    out_dir.mkdir(parents=True, exist_ok=True)
    if shap_doc is not None:
        sbytes = gzip.compress(json.dumps(shap_doc, ensure_ascii=False).encode("utf-8"), mtime=0)
        wrote_s = _write_once(spath, sbytes)
        doc["shap"].update({"file": spath.name, "sha256": _sha(sbytes) if wrote_s else _sha_file(spath),
                            "races": len(shap_doc.get("races") or {}), "errors": len(shap_doc.get("errors") or []),
                            "model": shap_doc.get("model"), "features": len(shap_doc.get("feature_cols") or []),
                            "written_now": wrote_s, "local_only": True})
    jbytes = json.dumps(doc, ensure_ascii=False, indent=1).encode("utf-8")
    if not _write_once(jpath, jbytes):
        return {"written": False, "reason": "already_frozen", "path": str(jpath)}
    return {"written": True, "path": str(jpath), "sha256": _sha(jbytes), "pre_race": pre_race,
            "first_post": doc["first_post"], "shap": doc["shap"]["status"],
            "races": len(doc["races"]), "pw_source": doc["power_rating"]["source"],
            "post_race_contamination": len(doc["power_rating"]["post_race_contamination"])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--out-dir", default=str(FREEZE_DIR))
    ap.add_argument("--allow-late", action="store_true", help="テスト用。発走後でも書く（本番の場所には不可）")
    ap.add_argument("--bundle", default=None, help="テスト用。使う bundle を指定（本番の場所には不可）")
    ap.add_argument("--contrib-dir", default=str(CONTRIB_DIR), help="pred_contrib の作業用ファイルの場所")
    a = ap.parse_args()
    try:
        res = freeze(a.date, Path(a.out_dir), allow_late=a.allow_late,
                     bundle_override=Path(a.bundle) if a.bundle else None, contrib_dir=Path(a.contrib_dir))
    except Exception as e:  # noqa: BLE001
        print(f"[zure_freeze] ERROR {type(e).__name__}: {e}")
        return 1
    print("[zure_freeze] " + json.dumps(res, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
