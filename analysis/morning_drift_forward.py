# -*- coding: utf-8 -*-
"""
morning_drift_forward.py — 「朝 9 時に売れていて発走前に売れなくなった馬」を 2026 年の前向き記録で数える (CW-14, 2026-11-13)
=================================================================================================
根拠: docs/research/POSTCOVID_SEARCH_20261113.md。過去 13 年 (TANPUK 時系列) で、
  セル A = 発走前の単勝 3〜4.9 倍 かつ 朝 9 時より 35% 以上オッズが伸びた馬 の単勝が 2020-25 の 6 年すべて 100% 超 (計 109.5% [92,127])。
  広い条件 = 発走前 3〜6.9 倍 かつ 朝比 +13% 以上 (レース内で最も伸びた 1 頭) の単勝は 85〜88% (基準 80%)。
本スクリプトは data/forward_prices の am9 (朝 9 時、scripts/am9_snapshot.py) と t20 / exp05fs_t35 / t10 の記録から同じ条件を数え、
確定払戻で採点するだけ。買い目・本線には一切触れない。
Gate A: 広い条件 300 点 (約 4 か月) → 同じオッズ帯の全馬より単勝回収率が高いか (信号そのものの存否)。
Gate B: セル A 100 点 (約 14 か月) → 単勝回収率と 95% CI を 1 回だけ記述。
出力: reports/morning_drift_forward.md、--json で件数。
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
import shadow_a95 as sa  # noqa: E402

FWD = BASE / "data" / "forward_prices"
OUT = BASE / "reports" / "morning_drift_forward.md"
START = "20261114"
DEC_STAGES = ("t20", "exp05fs_t35", "t10")      # 過去検証の発走 26〜30 分前に最も近いのは t20 と t35 の間。主判定は t20
TARGET_A, TARGET_B = 300, 100


def _load(f: str) -> dict | None:
    try:
        return json.loads(gzip.open(f, "rt", encoding="utf-8").read())
    except (OSError, ValueError):
        return None


def _am9_ok(r: dict, date: str) -> bool:
    """朝 9 時の記録として有効か: 当日の 08:30〜09:45 に取得したものだけ (後日の手動取得や遅延を混ぜない)。"""
    try:
        t = datetime.fromisoformat(str(r.get("observed_at"))[:19])
    except ValueError:
        return False
    return t.strftime("%Y%m%d") == date and "08:30" <= t.strftime("%H:%M") <= "09:45"


def collect() -> pd.DataFrame:
    wide = sa.load_wide()
    rows = []
    for dd in sorted(glob.glob(str(FWD / "2026*"))):
        date = Path(dd).name
        if date < START:
            continue
        kek = {k: v for k, v in sa.load_kekka(date, wide).items() if v.get("valid")}
        am = {}
        for f in sorted(glob.glob(f"{dd}/*_am9_*.json.gz")):
            r = _load(f)
            if r and (r.get("market") or {}).get("ok") and _am9_ok(r, date):
                am[Path(f).name[:16]] = {int(k): float(v) for k, v in r["market"]["tansho"].items() if v}
        for stage in DEC_STAGES:
            seen = {Path(f).name[:16]: f for f in sorted(glob.glob(f"{dd}/*_{stage}_*.json.gz"))}
            for rid, f in seen.items():
                if rid not in am:
                    continue
                r = _load(f)
                m = (r or {}).get("market") or {}
                if not m.get("ok"):
                    continue
                dec = {int(k): float(v) for k, v in (m.get("tansho") or {}).items() if v}
                if len(dec) < 8:
                    continue
                fav = min(dec, key=lambda b: (dec[b], b))
                k = kek.get(rid)
                for ban, o in dec.items():
                    o9 = am[rid].get(ban)
                    if not o9:
                        continue
                    sel = dict(a1=ban, a2=fav, a3=0, waku1=0, waku2=0)
                    rows.append(dict(date=date, rid=rid, stage=stage, ban=ban, o_dec=o, o_am9=o9, drift=float(np.log(o / o9)), is_fav=ban == fav,
                                     settled=k is not None,
                                     tan=sa.payout_per_100("tan", sel, k) if k else np.nan,
                                     fuku=sa.payout_per_100("fuku", sel, k) if k else np.nan,
                                     wide=(sa.payout_per_100("wide", sel, k) if (k and ban != fav) else np.nan),
                                     umaren=(sa.payout_per_100("umaren", sel, k) if (k and ban != fav) else np.nan)))
    return pd.DataFrame(rows)


def cells(d: pd.DataFrame, stage: str) -> dict:
    z = d[(d.stage == stage) & d.settled]
    if not len(z):
        return {}
    band = z[(z.o_dec >= 3) & (z.o_dec < 5)]
    a = band[band.drift > 0.3]
    w = z[(z.o_dec >= 3) & (z.o_dec < 7) & (z.drift > 0.12)]
    w = w.sort_values(["rid", "drift"]).drop_duplicates("rid", keep="last")
    base_w = z[(z.o_dec >= 3) & (z.o_dec < 7)]
    f = lambda x, c: None if not len(x) or x[c].notna().sum() == 0 else float(x[c].mean())  # noqa: E731
    return dict(n_races=int(z.rid.nunique()), cell_a=dict(n=len(a), tan=f(a, "tan"), fuku=f(a, "fuku"), wide=f(a, "wide"), umaren=f(a, "umaren")),
                band_3_49=dict(n=len(band), tan=f(band, "tan"), fuku=f(band, "fuku")),
                wide_rule=dict(n=len(w), tan=f(w, "tan"), fuku=f(w, "fuku"), wide=f(w, "wide")),
                band_3_69=dict(n=len(base_w), tan=f(base_w, "tan"), fuku=f(base_w, "fuku")))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    d = collect()
    s = {st: cells(d, st) for st in DEC_STAGES} if len(d) else {}
    main_st = s.get("t20") or {}
    if a.json:
        print(json.dumps(dict(gate_a=(main_st.get("wide_rule") or {}).get("n", 0), target_a=TARGET_A, gate_b=(main_st.get("cell_a") or {}).get("n", 0),
                              target_b=TARGET_B, dates=int(d.date.nunique()) if len(d) else 0, latest=str(d.date.max()) if len(d) else None, stages=s), ensure_ascii=False))
        return 0
    g = lambda v: "-" if v is None else f"{v:.1f}"  # noqa: E731
    md = [f"# 朝 9 時 → 発走前の動きの前向き確認 (CW-14, {pd.Timestamp.now():%Y-%m-%d})", "",
          f"朝 9 時の記録がある開催日 {d.date.nunique() if len(d) else 0} 日 ({START}〜)。採点は確定払戻、記述のみ (買い目ではない)。",
          f"Gate A = 広い条件 {TARGET_A} 点 / Gate B = セル A {TARGET_B} 点 (どちらも t20 時点で数える)。", "",
          "| 判断時点 | セル A (3〜4.9 倍・朝比 +35%) n | 単勝 | 複勝 | ワイド 1人気-X | 同じ帯の全馬 単勝 | 広い条件 n | 単勝 | 3〜6.9 倍の全馬 単勝 |", "|---|---|---|---|---|---|---|---|---|"]
    for st in DEC_STAGES:
        x = s.get(st) or {}
        ca, b1, wr, b2 = x.get("cell_a") or {}, x.get("band_3_49") or {}, x.get("wide_rule") or {}, x.get("band_3_69") or {}
        md.append(f"| {st} | {ca.get('n', 0)} | {g(ca.get('tan'))} | {g(ca.get('fuku'))} | {g(ca.get('wide'))} | {g(b1.get('tan'))} | {wr.get('n', 0)} | {g(wr.get('tan'))} | {g(b2.get('tan'))} |")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_bytes("\n".join(md).encode("utf-8"))
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
