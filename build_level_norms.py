# -*- coding: utf-8 -*-
"""
build_level_norms.py — 「馬レベル」正規化基準を生成 (公開データのみ / ZI非依存)
================================================================================
level_metric.horse_level_raw の生スコアを母集団分位で 0〜100 に正規化する
アンカーと、メンバーレベル(=上位3頭の平均レベル)のクラス別分布を作る。

出力: data/level_norms.json
  {
    "generated_at": "...",
    "n_horses": N,                              # レベル算出できた頭数
    "raw_anchors": [q0,q10,...,q100],           # 生スコア→0-100 の補間アンカー(11点)
    "classes": { "未勝利": {"n":..,"p20":..,"p40":..,"p60":..,"p80":..,"mean":..}, ...},
    "global":  {"p20":.., ...}
  }

ソース: site/data/*.json (build 済バンドル= history.runs がクリーンに入っている)。
実行:  python build_level_norms.py [--force]
ensure() を build_site.py が毎回呼ぶ(バンドルより古い時のみ再生成)。
"""
from __future__ import annotations
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

import level_metric as LM

BASE = Path(__file__).parent
SITE_DATA = BASE / "site" / "data"
OUT = BASE / "data" / "level_norms.json"
SKIP_FILES = {"manifest", "baba_today", "realized_bias", "results"}
MIN_CLASS_N = 15


def raw_to_100(raw: float, anchors) -> float:
    """生スコア → 0-100 (分位アンカーで線形補間)。"""
    return float(np.interp(raw, anchors, np.linspace(0, 100, len(anchors))))


def _pcts(vals) -> dict:
    a = np.asarray(vals, float)
    return {"n": int(len(a)), "mean": round(float(a.mean()), 1),
            "p20": round(float(np.percentile(a, 20)), 1),
            "p40": round(float(np.percentile(a, 40)), 1),
            "p60": round(float(np.percentile(a, 60)), 1),
            "p80": round(float(np.percentile(a, 80)), 1)}


def _is_stale() -> bool:
    if not OUT.exists():
        return True
    om = OUT.stat().st_mtime
    return any(p.stat().st_mtime > om for p in SITE_DATA.glob("2026*.json"))


def build() -> dict:
    raws: list[float] = []
    per_race: list[tuple] = []   # (class_group, [raw...])
    for f in sorted(SITE_DATA.glob("*.json")):
        if f.stem in SKIP_FILES:
            continue
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        for r in d.get("races", []):
            field = []
            for h in r.get("horses", []):
                raw = LM.horse_level_raw(h.get("history"))
                if raw is not None:
                    raws.append(raw)
                    field.append(raw)
            per_race.append((LM.class_group(r.get("klass")), field))

    if not raws:
        raise RuntimeError("no rated horses")
    anchors = [round(float(np.percentile(raws, q)), 2)
               for q in range(0, 101, 10)]

    # メンバーレベル = 上位3頭の level100 平均。クラス別分布。
    by_class: dict[str, list] = {}
    all_top3: list[float] = []
    for cg, field in per_race:
        if cg is None or len(field) < 3:
            continue
        top3 = sorted((raw_to_100(x, anchors) for x in field), reverse=True)[:3]
        t3 = float(np.mean(top3))
        by_class.setdefault(cg, []).append(t3)
        all_top3.append(t3)

    classes = {g: _pcts(v) for g, v in by_class.items() if len(v) >= MIN_CLASS_N}
    out = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "n_horses": len(raws),
        "raw_anchors": anchors,
        "classes": classes,
        "global": _pcts(all_top3) if all_top3 else {},
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[level_norms saved] {OUT}  horses={len(raws)}  classes={list(classes)}")
    return out


def ensure(force: bool = False, quiet: bool = False):
    if not force and not _is_stale():
        if not quiet:
            print(f"[level_norms] fresh, skip ({OUT.name})")
        return None
    return build()


if __name__ == "__main__":
    ensure(force="--force" in sys.argv)
