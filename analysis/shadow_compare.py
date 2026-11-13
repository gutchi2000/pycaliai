# -*- coding: utf-8 -*-
"""
shadow_compare.py — 候補モデルの紙上 bundle と本番 (凍結版) を、同じレースの結果で比べる。

使い方: python analysis/shadow_compare.py --shadow-dir reports/shadow_c2 --since 20261114
  本番側は reports/cowork_input_frozen/ (発走前の凍結版) のみ。凍結版の無い日は数えない。
  影側は bundle の mtime が当日 09:30 より前のものだけ (事後に作った紙上 bundle は数えない)。
  取消 (kekka 単勝欄が空の着順 0) の ◎ はそのレースを除外。中止・発売後除外は圏外。
  差はレース単位の対応のある差、CI は開催日クラスタ bootstrap 2000 回。
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def kekka(date: str) -> dict:
    p = ROOT / "data" / "kekka" / f"{date}.csv"
    if not p.exists():
        return {}
    raw = p.read_bytes()
    for enc in ("cp932", "utf-8-sig"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    out: dict = {}
    rd = csv.reader(io.StringIO(text))
    next(rd, None)
    for row in rd:
        if len(row) < 9:
            continue
        try:
            rid, uma, pos = row[7][:16], int(row[4]), int(row[6])
        except ValueError:
            continue
        r = out.setdefault(rid, {"pos": {}, "scratched": set()})
        r["pos"][uma] = pos
        if pos == 0 and not row[8].strip():
            r["scratched"].add(uma)
    return out


def hon(bundle: dict) -> dict:
    return {str(r["race_id"])[:16]: next((h["umaban"] for h in r["horses"] if h.get("mark") == "◎"), None)
            for r in bundle.get("races", [])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shadow-dir", required=True)
    ap.add_argument("--base-dir", default=None,
                    help="比べる相手。省略時は本番の凍結版 (reports/cowork_input_frozen)。"
                         "影同士 (例: reports/shadow_v6) を指定したときは相手も発走前に作った bundle だけを数える")
    ap.add_argument("--since", default="20261114")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    sdir = ROOT / a.shadow_dir if not Path(a.shadow_dir).is_absolute() else Path(a.shadow_dir)
    bdir = None if a.base_dir is None else (ROOT / a.base_dir if not Path(a.base_dir).is_absolute() else Path(a.base_dir))

    def pre_race(path: Path, d: str) -> bool:
        return datetime.fromtimestamp(os.path.getmtime(path)) < datetime.strptime(d + "0930", "%Y%m%d%H%M")
    rows = []
    for sp in sorted(sdir.glob("*_bundle.json")):
        d = sp.name[:8]
        if d < a.since:
            continue
        if not pre_race(sp, d):
            continue
        fp = (bdir or ROOT / "reports" / "cowork_input_frozen") / f"{d}_bundle.json"
        if not fp.exists() or (bdir is not None and not pre_race(fp, d)):
            continue
        res = kekka(d)
        hs, hp = hon(json.loads(sp.read_text(encoding="utf-8"))), hon(json.loads(fp.read_text(encoding="utf-8")))
        for rid in set(hs) & set(hp):
            r = res.get(rid)
            if not r or hs[rid] is None or hp[rid] is None:
                continue
            if hs[rid] in r["scratched"] or hp[rid] in r["scratched"]:
                continue
            ps, pp = r["pos"].get(hs[rid], 0), r["pos"].get(hp[rid], 0)
            rows.append((d, float(ps == 1), float(0 < ps <= 3), float(pp == 1), float(0 < pp <= 3)))
    if not rows:
        print("比較できるレースがまだ無い")
        return 0
    arr = np.array([r[1:] for r in rows])
    dates = np.array([r[0] for r in rows])
    ud = np.unique(dates)
    rng = np.random.default_rng(0)
    idx = {d: np.where(dates == d)[0] for d in ud}

    def ci(col_a, col_b):
        v = []
        for _ in range(a.boot):
            s = np.concatenate([idx[x] for x in rng.choice(ud, len(ud))])
            v.append(arr[s, col_a].mean() - arr[s, col_b].mean())
        return [round(float(np.percentile(v, 2.5)) * 100, 2), round(float(np.percentile(v, 97.5)) * 100, 2)]

    out = {"n_races": len(rows), "n_days": int(len(ud)), "from": str(ud[0]), "to": str(ud[-1]),
           "shadow": {"hon_win_pct": round(arr[:, 0].mean() * 100, 2), "hon_top3_pct": round(arr[:, 1].mean() * 100, 2)},
           "base": str(bdir.relative_to(ROOT) if bdir and bdir.is_relative_to(ROOT) else (bdir or "reports/cowork_input_frozen")),
           "production": {"hon_win_pct": round(arr[:, 2].mean() * 100, 2), "hon_top3_pct": round(arr[:, 3].mean() * 100, 2)},
           "diff_top3_pt": round((arr[:, 1].mean() - arr[:, 3].mean()) * 100, 2), "diff_top3_ci95": ci(1, 3),
           "diff_win_pt": round((arr[:, 0].mean() - arr[:, 2].mean()) * 100, 2), "diff_win_ci95": ci(0, 2)}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
