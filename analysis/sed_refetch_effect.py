# -*- coding: utf-8 -*-
"""
sed_refetch_effect.py — SED の取り直し (速報版→確定版、2026-11-19) の前後で、配信の予測がどれだけ変わったか (分析のみ)
=====================================================================================================
修正前の sed.parquet (控え data/jrdb/snapshots/Sed/parquet/ の 1 本) と修正後の表で、同じ開催日の予測を作り直して比べる。
作り直しは analysis/sed_prelim_impact.py と同じ子プロセス (--out-dir 付き、本番の出力先・凍結版・serve_path の控えには書かない)。
成績の表だけを差し替え、他の入力は同じ。対象は当時配信したレース (bundle が無い日は全レース)。

実行 (root から):
  python -m analysis.sed_refetch_effect --dates 20261114 20261115 --old <修正前の parquet> --new <修正後の parquet> [--model v8]
  (11/21 以降は、Phase A の後に --dates 20261121 で同じ比較ができる。--old は `python scripts/jrdb_sed_snapshot.py asof` 等で選ぶ)
出力: docs/research/SED_REFETCH_EFFECT_20261119.json (--out で変更可)。作業データは data/jrdb/analysis/sed_refetch_effect/
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from analysis import sed_prelim_impact as spi  # noqa: E402

WORK = spi.ROOT / "data/jrdb/analysis/sed_refetch_effect"


def run(model, arm, pq, d):
    od = WORK / f"{model}_{arm}"
    od.mkdir(parents=True, exist_ok=True)
    if (od / f"{d}_bundle.json").exists():
        return od / f"{d}_bundle.json"
    served = spi.SERVED / f"{d}_bundle.json"
    code = spi._CHILD.format(base=str(spi.ROOT), sed=str(pq), served=str(served), date=int(d),
                             csv=str(spi.ROOT / "data/weekly" / f"{d}.csv"), out=str(od / d), model=model)
    if not served.exists():      # 配信の bundle がまだ無い日は、対象レースを絞らない
        code = code.replace("el['prediction_eligible'] = re.sub(r'\\D', '', str(rid))[:16] in _served", "pass")
        code = code.replace("_b = json.load(open(r'" + str(served) + "', encoding='utf-8'))", "_b = {'races': []}")
    env = dict(os.environ, CB_HOSEI_PROXY="1", CB_SERVE_RECOVER="1", CB_ZURE_CONTRIB="0", PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable, "-c", code], cwd=str(spi.ROOT), capture_output=True, env=env, text=True, encoding="utf-8", errors="replace")
    (od / f"{d}.log").write_text((r.stdout or "") + (r.stderr or ""), encoding="utf-8")
    jl = [l.split("[jrdb]")[-1].strip() for l in (r.stdout + r.stderr).splitlines() if "[jrdb]" in l]
    print(f"{model} {arm} {d} rc={r.returncode} {jl[-1] if jl else ''}", flush=True)
    return od / f"{d}_bundle.json"


def compare(pairs):
    rows, per = [], {}
    for d, (pb, pa) in pairs.items():
        rb, ra = spi.races(pb), spi.races(pa)
        c = {"races": 0, "ai1_changed": 0, "hon_changed": 0, "any_mark_changed": 0, "marked_set_changed": 0, "changed_races": []}
        for rid in sorted(set(rb) & set(ra)):
            hb, ha = rb[rid][0], ra[rid][0]
            common = sorted(set(hb) & set(ha))
            if len(common) < 5:
                continue
            c["races"] += 1
            for u in common:
                rows.append({"date": d, "rid": rid, "dp": (float(ha[u]["p_win"]) - float(hb[u]["p_win"])) * 100,
                             "rank_b": hb[u].get("ai_rank"), "rank_a": ha[u].get("ai_rank")})
            top = lambda hs: max(common, key=lambda u: float(hs[u]["p_win"]))  # noqa: E731
            hon = lambda hs: next((u for u in common if spi.mark_of(hs[u]) == "◎"), None)  # noqa: E731
            mb = {u: spi.mark_of(hb[u]) for u in common if spi.mark_of(hb[u])}
            ma = {u: spi.mark_of(ha[u]) for u in common if spi.mark_of(ha[u])}
            c["ai1_changed"] += top(hb) != top(ha)
            c["hon_changed"] += hon(hb) != hon(ha)
            c["any_mark_changed"] += mb != ma
            c["marked_set_changed"] += set(mb) != set(ma)
            if mb != ma or top(hb) != top(ha):
                c["changed_races"].append(rid)
        per[d] = c
    df = pd.DataFrame(rows)
    a = df.dp.abs()
    rk = df.dropna(subset=["rank_b", "rank_a"])
    return {"horses": int(len(df)), "share_changed": float((a > 1e-9).mean()),
            "share_abs_ge": {str(t): float((a >= t).mean()) for t in (0.1, 0.5, 1.0, 2.0)},
            "abs_dp_max_pt": float(a.max()), "abs_dp_mean_changed_pt": float(a[a > 1e-9].mean()) if (a > 1e-9).any() else 0.0,
            "dp_quantiles_pt": {str(q): float(np.quantile(df.dp, q)) for q in (0.01, 0.05, 0.5, 0.95, 0.99)},
            "rank_changed_share": float((rk.rank_b != rk.rank_a).mean()), "per_date": per}


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", nargs="+", required=True)
    ap.add_argument("--old", required=True, help="修正前の sed.parquet (控え)")
    ap.add_argument("--new", required=True, help="修正後の sed.parquet")
    ap.add_argument("--model", default="v8")
    ap.add_argument("--out", default=str(BASE / "docs/research/SED_REFETCH_EFFECT_20261119.json"))
    a = ap.parse_args()
    pairs = {d: (run(a.model, "old", Path(a.old), d), run(a.model, "new", Path(a.new), d)) for d in a.dates}
    pairs = {d: v for d, v in pairs.items() if v[0].exists() and v[1].exists()}
    res = {"model": a.model, "old": Path(a.old).name, "new": Path(a.new).name, "dates": list(pairs), **compare(pairs)}
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "per_date"}, ensure_ascii=False))
    for d, c in res["per_date"].items():
        print(d, {k: v for k, v in c.items() if k != "changed_races"}, "変わったレース", c["changed_races"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
