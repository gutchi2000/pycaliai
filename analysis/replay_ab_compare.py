# -*- coding: utf-8 -*-
"""
replay_ab_compare.py — replay_serve_2026 の 2 つの作業ディレクトリ (A/B) を同一レース集合で比べる (2026-11-04)
================================================================================
用途: serve 側のコード変更 (例: CB_SERVE_RECOVER=0 と 1) が印の精度に与える差を、同じ週次 CSV・同じ kekka で
再生した bundle 同士で測る。指標は AI1位 (p_win 最大) の 1 着率 / 3 着内率、◎ (mark=◎) の 3 着内率、
AI1位の一致率、開催日ブートストラップの 95% CI。品質ゲート不合格日 (入力欠落) は別枠で出す。

実行: python -m analysis.replay_ab_compare --a <dirA> --b <dirB> [--since 20260101] [--until 20261231]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from analysis.replay_serve_2026 import _kekka, _races  # noqa: E402


def _top(hs):
    return int(max(hs, key=lambda h: float(h["p_win"]))["umaban"])


def _hon(hs):
    for h in hs:
        if h.get("mark") == "◎":
            return int(h["umaban"])
    return None


def load(dirp: Path, since: str, until: str):
    out = {}
    for p in sorted(dirp.glob("2026????_bundle.json")):
        d = p.name[:8]
        if not (since <= d <= until):
            continue
        lg = dirp / f"{d}.log"
        gate_ng = lg.exists() and "品質ゲート不合格" in lg.read_text(encoding="utf-8", errors="replace")
        out[d] = (_races(p), gate_ng)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--since", default="20260101"); ap.add_argument("--until", default="20261231")
    ap.add_argument("--label-a", default="A"); ap.add_argument("--label-b", default="B")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    A, B = load(Path(a.a), a.since, a.until), load(Path(a.b), a.since, a.until)
    rows = []
    for d in sorted(set(A) & set(B)):
        if not (BASE / "data" / "kekka" / f"{d}.csv").exists():
            continue
        kek = _kekka(d)
        ra, ga = A[d]; rb, gb = B[d]
        for rid in set(ra) & set(rb) & set(kek):
            top3, win = kek[rid]
            ta, tb = _top(ra[rid]), _top(rb[rid]); ha, hb = _hon(ra[rid]), _hon(rb[rid])
            rows.append(dict(date=d, gate_ng=bool(ga or gb), same=int(ta == tb),
                             a_win=int(ta in win), b_win=int(tb in win), a_top3=int(ta in top3), b_top3=int(tb in top3),
                             a_hon3=int(ha in top3) if ha else np.nan, b_hon3=int(hb in top3) if hb else np.nan))
    df = pd.DataFrame(rows)
    if df.empty:
        print("比較できるレースなし (両方の作業ディレクトリに bundle があり kekka もある日が無い)"); return 1
    rng = np.random.default_rng(0)

    def ci(g, x, y):
        per = g.groupby("date").agg(n=(x, "count"), x=(x, "sum"), y=(y, "sum"))
        if len(per) < 2:
            return float("nan"), float("nan")
        idx = rng.integers(0, len(per), size=(4000, len(per)))
        diff = 100 * (per.y.values[idx].sum(1) - per.x.values[idx].sum(1)) / per.n.values[idx].sum(1)
        return float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))

    def block(name, g):
        if g.empty:
            print(f"  {name}: n=0"); return
        print(f"  {name}: {len(g)} R / {g.date.nunique()} 日 | AI1位同一 {100*g.same.mean():.1f}%")
        for lab, x, y in (("AI1位 1着率", "a_win", "b_win"), ("AI1位 3着内率", "a_top3", "b_top3"), ("◎ 3着内率", "a_hon3", "b_hon3")):
            lo, hi = ci(g, x, y)
            print(f"    {lab:<12} {a.label_a} {100*g[x].mean():5.1f}%  {a.label_b} {100*g[y].mean():5.1f}%  差 {100*(g[y].mean()-g[x].mean()):+5.2f}pt  CI95 [{lo:+.2f}, {hi:+.2f}]")

    print(f"A={a.a}\nB={a.b}\n同一レース集合・同じ kekka・開催日 bootstrap 4000 回")
    block("[全体]", df)
    block("[品質ゲート合格日のみ]", df[~df.gate_ng])
    if df.gate_ng.any():
        block("[入力欠落日 (参考)]", df[df.gate_ng])
    out = {"a": a.a, "b": a.b, "n_races": int(len(df)), "n_days": int(df.date.nunique()),
           "all": {k: float(df[k].mean()) for k in ("same", "a_win", "b_win", "a_top3", "b_top3", "a_hon3", "b_hon3")}}
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
