# -*- coding: utf-8 -*-
"""
kyi_precheck.py — v8 の Phase A の前に、その開催日の JRDB 前日 KYI が品質ゲートを通るかを見る (読むだけ)
==========================================================================================
export_weekly_marks.py と同じ KYI ファイル (jrdb_kyi_features.serve_sources: 発走前の控えの最新 → 無ければアーカイブ) を読み、
k8_* の素の列ごとに 埋まり・値の種類数・ゲートの測り方での充足 を出す。
ゲートの測り方 = export_weekly_marks.feature_coverage (全頭同じ値なら 0%、CONST_OK_COLS の列だけは埋まり率)。
基準は data/serve_feature_baseline.json。その日の bundle があれば、配信したレースに絞って測る。

  python scripts/kyi_precheck.py 20261121

判定 (exit code): 0 = 通る見込み / 1 = 止まる見込み (下の「止まる」行を見る) / 2 = KYI が無い
「止まる」の条件はゲートと同じ: 基準 40% 以上の列が、今回 20% 未満 かつ 基準の 4 割未満。
ほかに KYI の IDM の埋まりが 90% 未満なら、export がその日の v8 を諦めて v7 で作り直す。
"""
from __future__ import annotations

import json
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
import jrdb_kyi_features as jk  # noqa: E402
from export_weekly_marks import CONST_OK_COLS, feature_coverage  # noqa: E402


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    if len(sys.argv) < 2 or not re.fullmatch(r"\d{8}", sys.argv[1]):
        print(__doc__)
        return 2
    d = sys.argv[1]
    kp, _ = jk.serve_sources(d)
    if kp is None:
        print(f"[止まる] KYI{d[2:]} が無い。python scripts/jrdb_snapshot.py capture (平日 or 週末 19:00 以降)")
        return 2
    with zipfile.ZipFile(kp) as z:
        rows = []
        for nm in z.namelist():
            rows += jk._parse_kyi_bytes(z.read(nm))
    k = jk.kyi_frame(rows)
    scope = "KYI の全レース"
    bp = BASE / "reports/cowork_input" / f"{d}_bundle.json"
    if bp.exists():
        b = json.loads(bp.read_text(encoding="utf-8"))
        rs = b["races"] if isinstance(b["races"], list) else list(b["races"].values())
        rk = set(jk.rid16_to_racekey(pd.Series([re.sub(r"\D", "", str(r.get("race_id", "")))[:16] for r in rs])))
        if rk:
            k = k[k["racekey"].isin(rk)]
            scope = f"配信した {len(rk)} レース (bundle の model={b.get('model')})"
    base = json.loads((BASE / "data/serve_feature_baseline.json").read_text(encoding="utf-8")).get("baseline_cov", {})
    print(f"{d}  {kp.name}  {len(rows)} 行  範囲: {scope}  {len(k)} 頭")
    idm = float(k["idm"].notna().mean()) if len(k) else 0.0
    bad = []
    print(f"{'列':16s} {'埋まり':>6s} {'種類':>4s} {'ゲート':>6s} {'基準':>6s}")
    for c in [c for c in jk.KYI_NUM if c not in ("d_jun", "a_jun", "g_jun")]:
        f = f"k8_{c}"
        s = k[c].astype(float)
        g = feature_coverage(s, allow_constant=(f in CONST_OK_COLS))
        e = base.get(f)
        stop = e is not None and e >= 0.40 and g < 0.20 and g < e * 0.40
        bad += [f] if stop else []
        print(f"{f:16s} {s.notna().mean() * 100:5.0f}% {s.dropna().nunique():4d} {g * 100:5.0f}% "
              f"{'' if e is None else f'{e * 100:5.0f}%':>6s}{'  ← 止まる' if stop else ''}")
    print(f"KYI の IDM の埋まり {idm * 100:.0f}% (90% 未満なら v8 を諦めて v7)")
    if bad or idm < 0.90:
        print(f"[止まる見込み] {', '.join(bad) or 'IDM の埋まり不足'}")
        return 1
    print("[通る見込み] KYI 由来の k8_* はゲートで止まらない (z/順位・外厩の列と他の特徴は export のログで確認)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
