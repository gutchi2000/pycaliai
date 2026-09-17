# -*- coding: utf-8 -*-
"""
serve_gap_controls.py — serve 劣化 (-9.65pt) が母集団差で説明できるか潰す
=========================================================================
対照は「同じレースの市場1番人気」。母集団が難しくなったなら市場も同じだけ落ちるはず。
さらに 頭数 / クラス で層別し、比較可能な部分集合でも差が残るかを見る。

実行: python -m analysis.serve_gap_controls
"""
from __future__ import annotations
import glob
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def load_offline():
    races = joblib.load(BASE / "data/_policy/bet_substrate.pkl")
    out = []
    for r in races:
        o = np.where(np.isfinite(r["odds9"]) & (r["odds9"] > 0), r["odds9"], 9999.0)
        ai1 = int(r["ban"][int(np.argmax(r["p"]))])
        fav = int(r["ban"][int(np.argmin(o))])
        top3 = {r["first"], r["second"], r["third"]}
        out.append(dict(src="offline23-25", n=r["n"],
                        ai_win=int(ai1 == r["first"]), ai_top3=int(ai1 in top3),
                        fav_win=int(fav == r["first"]), fav_top3=int(fav in top3),
                        agree=int(ai1 == fav), fav_odds=float(np.min(o))))
    return pd.DataFrame(out)


def load_serve():
    out = []
    for bp in sorted(glob.glob(str(BASE / "reports/cowork_input/*_bundle.json"))):
        date = Path(bp).name[:8]
        kp = BASE / "data" / "kekka" / f"{date}.csv"
        if not date.startswith("2026") or not kp.exists():
            continue
        k = pd.read_csv(kp, encoding="cp932", low_memory=False)
        k["rid16"] = k["レースID(新)"].astype(str).str[:16]
        k["ban"] = pd.to_numeric(k["馬番"], errors="coerce")
        k["fin"] = pd.to_numeric(k["確定着順"], errors="coerce")
        fin = {}
        for rid, g in k.groupby("rid16"):
            g = g[g.fin.notna() & (g.fin >= 1)]
            t = g[g.fin <= 3].sort_values("fin")
            if len(t) < 3:
                continue
            fin[rid] = (int(t.ban.iloc[0]), {int(x) for x in t.ban.iloc[:3]})
        b = json.loads(Path(bp).read_text(encoding="utf-8"))
        races = b["races"] if isinstance(b["races"], list) else list(b["races"].values())
        for r in races:
            rid = "".join(c for c in str(r.get("race_id", "")) if c.isdigit())[:16]
            if rid not in fin:
                continue
            hs = [h for h in r.get("horses", []) if h.get("p_win") is not None
                  and h.get("umaban") is not None]
            if len(hs) < 5:
                continue
            od = [(h, float(h.get("tansho_odds"))) for h in hs
                  if h.get("tansho_odds") not in (None, "") and float(h.get("tansho_odds")) > 0]
            if not od:
                continue
            ai1 = int(max(hs, key=lambda h: float(h["p_win"]))["umaban"])
            fav_h, fav_o = min(od, key=lambda t: t[1])
            fav = int(fav_h["umaban"])
            w, top3 = fin[rid]
            rm = r.get("race_meta", {}) or {}
            out.append(dict(src="serve2026", n=len(hs),
                            ai_win=int(ai1 == w), ai_top3=int(ai1 in top3),
                            fav_win=int(fav == w), fav_top3=int(fav in top3),
                            agree=int(ai1 == fav), fav_odds=fav_o,
                            cls=str(rm.get("class") or rm.get("race_class")
                                    or rm.get("grade") or "")))
    return pd.DataFrame(out)


def row(label, g):
    n = len(g)
    aw, at = g.ai_win.sum(), g.ai_top3.sum()
    fw, ft = g.fav_win.sum(), g.fav_top3.sum()
    lo, hi = wilson(aw, n)
    print(f"  {label:<22} n={n:>5}  AI勝率={100*aw/n:>5.2f}% [{lo:.1f},{hi:.1f}]  "
          f"市場={100*fw/n:>5.2f}%  差={100*(aw-fw)/n:>+6.2f}pt | "
          f"AItop3={100*at/n:>5.2f}% 市場top3={100*ft/n:>5.2f}% "
          f"差={100*(at-ft)/n:>+6.2f}pt | 一致={100*g.agree.mean():>5.1f}%")


def main() -> None:
    off, srv = load_offline(), load_serve()
    print("★ 対照 = 同じレースの市場1番人気。母集団が難しくなっただけなら「差」は変わらないはず。\n")
    print("=== 全体 ===")
    row("offline 2023-25", off)
    row("serve 2026", srv)

    print("\n=== 頭数で層別 ===")
    for lo_, hi_, lb in [(5, 10, "〜10頭"), (11, 13, "11-13頭"),
                         (14, 16, "14-16頭"), (17, 30, "17頭〜")]:
        a = off[(off.n >= lo_) & (off.n <= hi_)]
        b = srv[(srv.n >= lo_) & (srv.n <= hi_)]
        if len(a) > 200 and len(b) > 100:
            row(f"offline {lb}", a)
            row(f"serve   {lb}", b)

    print("\n=== 1番人気オッズ帯で層別 (レースの堅さを揃える) ===")
    for lo_, hi_, lb in [(0, 2.0, "1人気<2倍"), (2.0, 3.0, "2-3倍"),
                         (3.0, 5.0, "3-5倍"), (5.0, 99, "5倍〜")]:
        a = off[(off.fav_odds >= lo_) & (off.fav_odds < hi_)]
        b = srv[(srv.fav_odds >= lo_) & (srv.fav_odds < hi_)]
        if len(a) > 200 and len(b) > 100:
            row(f"offline {lb}", a)
            row(f"serve   {lb}", b)

    print("\n=== 結論の読み方 ===")
    d_off = 100 * (off.ai_win.mean() - off.fav_win.mean())
    d_srv = 100 * (srv.ai_win.mean() - srv.fav_win.mean())
    print(f"  「AI - 市場」勝率差: offline {d_off:+.2f}pt  →  serve {d_srv:+.2f}pt  "
          f"(劣化 {d_srv-d_off:+.2f}pt)")
    d_off3 = 100 * (off.ai_top3.mean() - off.fav_top3.mean())
    d_srv3 = 100 * (srv.ai_top3.mean() - srv.fav_top3.mean())
    print(f"  「AI - 市場」top3差 : offline {d_off3:+.2f}pt  →  serve {d_srv3:+.2f}pt  "
          f"(劣化 {d_srv3-d_off3:+.2f}pt)")
    print(f"  AI と市場の1位一致率: offline {100*off.agree.mean():.1f}%  →  "
          f"serve {100*srv.agree.mean():.1f}%")


if __name__ == "__main__":
    main()
