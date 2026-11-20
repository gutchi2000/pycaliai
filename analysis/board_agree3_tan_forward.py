"""CW-27: ボード一致 3 の ◎単勝 1 点（帯なし）の前向き確認（2026-11-21〜）。

2026-11-21 ユーザー指示「◎単勝の前向き検証は裏で n=100〜150 貯まるまで」。
再構成の探索（docs/research/BOARD_GO_RECON_EXPLORATION_20261120.md §10c）で 167〜201R・100〜103% だった買い方を、
発走前に記録した値だけで数える:
  - 対象: reports/cowork_output/{date}_bets.json の各レースの board_combo（T-10 で compute_bets が記録）で
    agree == 3（◎が速度・PW・推定人気の 3 つとも 1 位。ゲージは発走前の site/data/{date}.json）。GO 不問・帯なし。
  - 買い目: ◎単勝 100 円。払戻は data/kekka/{date}.csv の確定払戻（build_site.parse_kekka / settle_bet）。
  - 取消（返還）と決済不能は数えない。
サイトの買い目（board_combo の 27 セル）とは別の、記述だけの台帳。本線・表示には何もしない。

python -m analysis.board_agree3_tan_forward [--since 20261121]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

SINCE = "20261121"
CHECKPOINTS = (100, 150)   # 100 で中間の確認、150 で判定（ユーザー指示の 100〜150）


def rows(since: str = SINCE) -> list[dict]:
    import build_site as B
    out = []
    wide = None
    for p in sorted((BASE / "reports" / "cowork_output").glob("2026????_bets.json")):
        d = p.name[:8]
        if d < since or not (BASE / "data" / "kekka" / f"{d}.csv").exists():
            continue
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            continue
        races = raw.get("bets", []) if isinstance(raw, dict) else raw
        if wide is None:
            wide = B.parse_wide_kekka()
        res_all = B.parse_kekka(d, wide)
        for e in races:
            c = (e or {}).get("board_combo") or {}
            if c.get("stage") != "T-10" or c.get("agree") != 3 or not c.get("hon"):
                continue
            rid = str(e.get("race_id"))[:16]
            res = res_all.get(rid)
            if not res:
                continue
            s = B.settle_bet("単勝", str(c["hon"]), 100.0, res)
            if not s.get("settled") or s.get("refunded"):
                continue
            out.append({"date": d, "rid": rid, "hon": c["hon"], "go": c.get("go"), "band": c.get("band_code"),
                        "ret": float(s["received"]), "win": bool(s["is_win"])})
    return out


def summary(since: str = SINCE, nboot: int = 4000) -> dict:
    rs = rows(since)
    n = len(rs)
    if n == 0:
        return {"n": 0, "since": since}
    ret = np.array([r["ret"] for r in rs])
    rng = np.random.default_rng(0)
    bs = ret[rng.integers(0, n, (nboot, n))].mean(1)
    im = int(np.argmax(ret))
    return {"n": n, "since": since, "latest": rs[-1]["date"], "roi": round(float(ret.mean()), 1),
            "ci": [round(float(np.percentile(bs, 2.5)), 1), round(float(np.percentile(bs, 97.5)), 1)],
            "roi_drop_max": round(float((ret.sum() - ret[im]) / max(n - 1, 1)), 1) if n > 1 else None,
            "hit": round(100 * float(np.mean([r["win"] for r in rs])), 1),
            "by_band": {b: sum(1 for r in rs if r["band"] == b) for b in sorted({r["band"] for r in rs if r["band"]})}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=SINCE)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(summary(a.since), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
