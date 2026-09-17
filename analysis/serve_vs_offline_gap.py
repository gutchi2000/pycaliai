# -*- coding: utf-8 -*-
"""
serve_vs_offline_gap.py — offline OOS の印と 2026 実serve の印を同じ物差しで比べる
==============================================================================
deep_bet_search の構造は offline(2023-25) で ROI 101.7% だったのに、
2026 as-served では 67.3% / 的中率が 9.43%→4.23% に半減した。

年の違いなのか、serve 経路の劣化なのかを切り分ける。
物差しは「同じレースでの市場1番人気」。市場の1番人気勝率はどの年もほぼ 32% で安定
しているので、これを基準線にすれば期間効果を除いてモデル側の劣化が見える。

実行: python -m analysis.serve_vs_offline_gap
"""
from __future__ import annotations
import glob
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]


def offline():
    races = joblib.load(BASE / "data/_policy/bet_substrate.pkl")
    out = []
    for r in races:
        o = np.where(np.isfinite(r["odds9"]) & (r["odds9"] > 0), r["odds9"], 9999.0)
        ai1 = int(r["ban"][int(np.argmax(r["p"]))])
        fav = int(r["ban"][int(np.argmin(o))])
        top3 = {r["first"], r["second"], r["third"]}
        out.append(dict(year=r["date"][:4],
                        ai_win=int(ai1 == r["first"]), ai_top3=int(ai1 in top3),
                        fav_win=int(fav == r["first"]), fav_top3=int(fav in top3),
                        agree=int(ai1 == fav)))
    return pd.DataFrame(out)


def serve():
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
            ai1 = int(max(hs, key=lambda h: float(h["p_win"]))["umaban"])
            hon = next((int(h["umaban"]) for h in hs if h.get("mark") == "◎"), ai1)
            od = [(h, h.get("tansho_odds") or h.get("odds") or h.get("tan_odds"))
                  for h in hs]
            od = [(h, float(o)) for h, o in od if o not in (None, "") and float(o) > 0]
            fav = int(min(od, key=lambda t: t[1])[0]["umaban"]) if od else None
            w, top3 = fin[rid]
            out.append(dict(year="2026s", ai_win=int(ai1 == w), ai_top3=int(ai1 in top3),
                            hon_win=int(hon == w), hon_top3=int(hon in top3),
                            fav_win=(int(fav == w) if fav else np.nan),
                            fav_top3=(int(fav in top3) if fav else np.nan),
                            agree=(int(ai1 == fav) if fav else np.nan)))
    return pd.DataFrame(out)


def main() -> None:
    off = offline()
    srv = serve()
    print("=== offline OOS (v6 を後から流した 2023-25) ===")
    for y, g in off.groupby("year"):
        print(f"  {y} n={len(g):>5}  AI1位 勝率={100*g.ai_win.mean():>5.2f}% "
              f"top3={100*g.ai_top3.mean():>5.2f}%   "
              f"市場1人気 勝率={100*g.fav_win.mean():>5.2f}% top3={100*g.fav_top3.mean():>5.2f}%  "
              f"一致率={100*g.agree.mean():>5.1f}%")
    g = off
    print(f"  通算 n={len(g):>5}  AI1位 勝率={100*g.ai_win.mean():>5.2f}% "
          f"top3={100*g.ai_top3.mean():>5.2f}%   "
          f"市場1人気 勝率={100*g.fav_win.mean():>5.2f}% top3={100*g.fav_top3.mean():>5.2f}%")

    print("\n=== 2026 as-served (実際に朝配信していた bundle) ===")
    s = srv
    fav_ok = s.dropna(subset=["fav_win"])
    print(f"  n={len(s):>5}  AI1位 勝率={100*s.ai_win.mean():>5.2f}% "
          f"top3={100*s.ai_top3.mean():>5.2f}%")
    print(f"        ◎(印)  勝率={100*s.hon_win.mean():>5.2f}% top3={100*s.hon_top3.mean():>5.2f}%")
    if len(fav_ok):
        print(f"  市場1人気 (bundle内オッズ, n={len(fav_ok)}) 勝率={100*fav_ok.fav_win.mean():>5.2f}% "
              f"top3={100*fav_ok.fav_top3.mean():>5.2f}%  一致率={100*fav_ok.agree.mean():.1f}%")
    else:
        print("  bundle 内に単勝オッズが無く市場ベースラインは取れない")

    print("\n=== 判定 ===")
    a = 100 * off.ai_win.mean()
    b = 100 * s.ai_win.mean()
    print(f"  AI 1位の勝率: offline {a:.2f}%  →  2026 serve {b:.2f}%   Δ={b-a:+.2f}pt")
    a3 = 100 * off.ai_top3.mean()
    b3 = 100 * s.ai_top3.mean()
    print(f"  AI 1位の top3: offline {a3:.2f}%  →  2026 serve {b3:.2f}%   Δ={b3-a3:+.2f}pt")
    print("  ※市場1番人気の勝率は年によらず ~32% で安定。offline の AI1位がそれと同等なのに")
    print("    serve で大きく下回るなら、期間差ではなく serve 経路の劣化を意味する。")


if __name__ == "__main__":
    main()
