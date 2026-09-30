# 2026 serve の印品質を「serve 修正前 / 後」で分けて同じ物差し(市場1番人気)で比べる
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd
BASE = Path(r"E:\PyCaLiAI")
rows = []
for bp in sorted(glob.glob(str(BASE / "reports/cowork_input/*_bundle.json"))):
    date = Path(bp).name[:8]
    kp = BASE / "data" / "kekka" / f"{date}.csv"
    if not date.startswith("2026") or not kp.exists():
        continue
    k = pd.read_csv(kp, encoding="cp932", low_memory=False)
    k["rid16"] = k["レースID(新)"].astype(str).str[:16]
    for c, n in [("馬番", "ban"), ("確定着順", "fin"), ("馬単", "umatan")]:
        k[n] = pd.to_numeric(k[c], errors="coerce")
    fin = {}
    for rid, g in k.groupby("rid16"):
        g = g[g.fin.notna() & (g.fin >= 1)]
        t = g[g.fin <= 3].sort_values("fin")
        if len(t) < 3: continue
        w1 = g[g.fin == 1]; w2 = g[g.fin == 2]
        if len(w1) != 1 or len(w2) != 1: continue
        fin[rid] = (int(t.ban.iloc[0]), int(w2.ban.iloc[0]), {int(x) for x in t.ban.iloc[:3]},
                    float(w1.umatan.iloc[0]) if pd.notna(w1.umatan.iloc[0]) else np.nan)
    b = json.loads(Path(bp).read_text(encoding="utf-8"))
    races = b["races"] if isinstance(b["races"], list) else list(b["races"].values())
    for r in races:
        rid = "".join(c for c in str(r.get("race_id", "")) if c.isdigit())[:16]
        if rid not in fin: continue
        hs = [h for h in r.get("horses", []) if h.get("p_win") is not None and h.get("umaban") is not None]
        if len(hs) < 5: continue
        order = sorted(hs, key=lambda h: -float(h["p_win"]))
        ai1, ai2 = int(order[0]["umaban"]), int(order[1]["umaban"])
        od = [(h, h.get("tansho_odds") or h.get("odds") or h.get("tan_odds")) for h in hs]
        od = [(h, float(o)) for h, o in od if o not in (None, "") and float(o) > 0]
        fav = int(min(od, key=lambda t: t[1])[0]["umaban"]) if od else None
        w, s, top3, ut = fin[rid]
        rows.append(dict(date=date, ai_win=int(ai1 == w), ai_top3=int(ai1 in top3),
                         fav_win=(int(fav == w) if fav else np.nan), fav_top3=(int(fav in top3) if fav else np.nan),
                         agree=(int(ai1 == fav) if fav else np.nan),
                         umatan_hit=int(ai1 == w and ai2 == s),
                         umatan_ret=(ut if (ai1 == w and ai2 == s and np.isfinite(ut)) else 0.0)))
df = pd.DataFrame(rows)
def rep(name, d):
    if len(d) == 0: print(f"{name}: n=0"); return
    print(f"{name:<34} n={len(d):>5} days={d.date.nunique():>2} | AI1着 {100*d.ai_win.mean():5.1f}% 市場1人気 {100*d.fav_win.mean():5.1f}% Δ{100*(d.ai_win.mean()-d.fav_win.mean()):+5.1f} "
          f"| AI top3 {100*d.ai_top3.mean():5.1f}% 市場 {100*d.fav_top3.mean():5.1f}% Δ{100*(d.ai_top3.mean()-d.fav_top3.mean()):+5.1f} "
          f"| 一致 {100*d.agree.mean():4.1f}% | 馬単1点 的中 {100*d.umatan_hit.mean():4.2f}% ROI {d.umatan_ret.sum()/len(d):6.1f}%")
print("offline 2023-25 参考: AI1着 31.75 / 市場 32.97 / top3 63.44 vs 64.05 / 一致 53.9 / 馬単 9.43% 101.7%")
rep("2026 全期間", df)
rep("〜0919 (serve修正前)", df[df.date <= "20260919"])
rep("0920〜 (芝ダ正規化後)", df[df.date >= "20260920"])
rep("0906〜0919 (直前3週)", df[(df.date >= "20260906") & (df.date <= "20260919")])
rep("0801〜0919", df[(df.date >= "20260801") & (df.date <= "20260919")])
print("\n日別 (0906〜):")
for d, g in df[df.date >= "20260906"].groupby("date"):
    print(f"  {d} n={len(g):>3} AI1着 {100*g.ai_win.mean():5.1f} 市場 {100*g.fav_win.mean():5.1f} top3 {100*g.ai_top3.mean():5.1f}/{100*g.fav_top3.mean():5.1f} 一致 {100*g.agree.mean():4.1f} 馬単hit {int(g.umatan_hit.sum())}")
