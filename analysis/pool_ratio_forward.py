# -*- coding: utf-8 -*-
"""
pool_ratio_forward.py — 「複勝票数 / 単勝票数 が大きいレースの堅い 1 番人気」を 2026 の前向き記録で数える (CW-13, 2026-11-12)
====================================================================================================
根拠 (docs/research/PINNED_FLOOR_SEARCH_20261112.md): 2013-2025 の発走約 28 分前スナップで、1 番人気 単勝 ≤1.9 倍のうち
複勝票数/単勝票数 ≥ 2.19 (2013-16 の 75% 点、以後固定) のレースは複勝回収率 96.1% (2017-25 の 9 年すべて 93〜98%)、
3 着内率 89% (全体 81%)。黒字ではないが、sniper の現行条件 (≤1.9 だけ) の 90% 前後を 5pt 上回る。
本スクリプトは data/forward_prices の T-35 / T-20 / T-10 記録 (schema v2、票数計つき、2026-10-03〜) から同じ条件を数えるだけ。
買い目・本線には一切触れない。Gate: T-10 で条件を満たし決済できたレース 100 → 複勝回収率と 3 着内率を 1 回だけ判定。
出力: reports/pool_ratio_forward.md、data/_joint/forward_pool_ratio_2026.parquet。--json で件数だけ。
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
import shadow_a95 as sa  # noqa: E402

FWD = BASE / "data" / "forward_prices"
OUT = BASE / "reports" / "pool_ratio_forward.md"
START = "20261003"            # schema v2 (票数計つき) の開始
RATIO_MIN = 2.19              # 2013-16 の 75% 点 (固定)
ODDS_MAX = 1.9
TARGET = 100
STAGES = ("exp05fs_t35", "t20", "t10")


def collect() -> pd.DataFrame:
    wide = sa.load_wide()
    rows = []
    for dd in sorted(glob.glob(str(FWD / "2026*"))):
        date = Path(dd).name
        if date < START:
            continue
        kek = {k: v for k, v in sa.load_kekka(date, wide).items() if v.get("valid")}
        for stage in STAGES:
            seen = {}
            for f in sorted(glob.glob(f"{dd}/*_{stage}_*.json.gz")):
                seen[Path(f).name[:16]] = f
            for rid, f in seen.items():
                try:
                    r = json.loads(gzip.open(f, "rt", encoding="utf-8").read())
                except (OSError, ValueError):
                    continue
                m = r.get("market") or {}
                if not m.get("ok"):
                    continue
                vt = None
                for c in r.get("jv_captures") or []:
                    if c.get("kind") == "O1" and c.get("records"):
                        vt = c["records"][-1].get("votes_total")
                if not vt or not vt.get("tansho") or not vt.get("fukusho"):
                    continue
                tan = {int(k): float(v) for k, v in (m.get("tansho") or {}).items() if v}
                if len(tan) < 8:
                    continue
                fav = min(tan, key=lambda b: (tan[b], b))
                k = kek.get(rid)
                pay = sa.payout_per_100("fuku", dict(a1=fav, a2=0, a3=0, waku1=0, waku2=0), k) if k else np.nan
                rows.append(dict(date=date, rid=rid, stage=stage, fav=fav, tan=tan[fav], n=len(tan), ratio=vt["fukusho"] / vt["tansho"],
                                 settled=k is not None, pay=pay))
    return pd.DataFrame(rows)


def summarize(d: pd.DataFrame) -> dict:
    out = {}
    for st in STAGES:
        z = d[(d.stage == st) & (d.tan <= ODDS_MAX) & d.settled]
        a = z[z.ratio >= RATIO_MIN]; b = z[z.ratio < RATIO_MIN]
        out[st] = dict(n_zone=len(z), n_hit_rule=len(a), roi_rule=float(a.pay.mean()) if len(a) else None, top3_rule=float((a.pay > 0).mean()) if len(a) else None,
                       n_rest=len(b), roi_rest=float(b.pay.mean()) if len(b) else None, top3_rest=float((b.pay > 0).mean()) if len(b) else None)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    d = collect()
    s = summarize(d) if len(d) else {}
    if a.json:
        print(json.dumps(dict(current=(s.get("t10") or {}).get("n_hit_rule", 0), target=TARGET, dates=int(d.date.nunique()) if len(d) else 0, stages=s), ensure_ascii=False))
        return 0
    if len(d):
        d.to_parquet(BASE / "data" / "_joint" / "forward_pool_ratio_2026.parquet")
    md = [f"# 複勝票数/単勝票数 の前向き確認 (CW-13, {pd.Timestamp.now():%Y-%m-%d})", "",
          f"条件: 8 頭以上、その時点の 1 番人気が単勝 ≤{ODDS_MAX} 倍、複勝票数/単勝票数 ≥ {RATIO_MIN}。その馬の複勝 1 点を確定払戻で採点 (記述のみ、買い目ではない)。",
          f"開催日 {d.date.nunique() if len(d) else 0} 日 ({START}〜)。Gate = T-10 で条件を満たし決済できたレース {TARGET}。", "",
          "| 時点 | 対象 (≤1.9) | 条件を満たす | 3着内率 | 複勝回収率 | 満たさない | 3着内率 | 複勝回収率 |", "|---|---|---|---|---|---|---|---|"]
    f = lambda v, p=1: "-" if v is None else (f"{100 * v:.0f}%" if p == 0 else f"{v:.1f}")  # noqa: E731
    for st in STAGES:
        x = s.get(st) or {}
        md.append(f"| {st} | {x.get('n_zone', 0)} | {x.get('n_hit_rule', 0)} | {f(x.get('top3_rule'), 0)} | {f(x.get('roi_rule'))} | {x.get('n_rest', 0)} | {f(x.get('top3_rest'), 0)} | {f(x.get('roi_rest'))} |")
    OUT.write_bytes("\n".join(md).encode("utf-8"))
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
