# -*- coding: utf-8 -*-
"""serve_fuku_validate.py — v6 bundle の◎複勝が「実 serve(2026)」で本当に何%当たり何%回収か。
bundle(reports/cowork_input/*_bundle.json)の◎(umaban,p_win) を 2026実結果(data/kekka/2026*.csv の
確定着順・複勝配当)に突合し、p_win 信頼度カバレッジ別の 的中率/回収率を出す。offline射影でなく実serve。"""
import json, glob, re
from pathlib import Path
import numpy as np, pandas as pd

BASE = Path(r"E:\PyCaLiAI")
def rid16(x): return re.sub(r"\D", "", str(x))[:16]

# 1) bundle の◎ (rid16, umaban, p_win, date)
recs = []
for f in glob.glob(str(BASE / "reports/cowork_input/*_bundle.json")):
    d = json.load(open(f, encoding="utf-8"))
    for r in d.get("races", []):
        rid = rid16(r.get("race_id", ""))
        h = next((h for h in r.get("horses", []) if h.get("mark") == "◎"
                  and h.get("p_win") is not None and h.get("umaban") is not None), None)
        if h:
            recs.append((rid, int(h["umaban"]), float(h["p_win"]), rid[:8]))
bun = pd.DataFrame(recs, columns=["rid", "uma", "pw", "date"]).drop_duplicates(["rid", "uma"])

# 2) 2026 kekka: (rid16, umaban) -> (確定着順, 複勝配当)
res = {}
for f in glob.glob(str(BASE / "data/kekka/2026*.csv")):
    try:
        df = pd.read_csv(f, encoding="cp932")
    except Exception:
        continue
    if "確定着順" not in df.columns or "複勝配当" not in df.columns:
        continue
    for _, row in df.iterrows():
        try:
            uma = int(row["馬番"]); rid = rid16(row["レースID(新)"])
        except Exception:
            continue
        try:
            chk = int(row["確定着順"])
        except Exception:
            chk = 99
        fp = row.get("複勝配当")
        try:
            fp = float(str(fp).replace(",", "")) if pd.notna(fp) and str(fp).strip() not in ("", "-", "nan") else 0.0
        except Exception:
            fp = 0.0
        res[(rid, uma)] = (chk, fp)

# 3) 突合
rows = []
for _, r in bun.iterrows():
    rr = res.get((r["rid"], r["uma"]))
    if rr is None:
        continue
    chk, fp = rr
    hit = 1 if chk <= 3 else 0
    rows.append((r["pw"], hit, fp if hit else 0.0, r["date"]))
df = pd.DataFrame(rows, columns=["pw", "hit", "pay", "date"])
nweeks = pd.to_datetime(df["date"], format="%Y%m%d").dt.strftime("%G-W%V").nunique()
print(f"bundle ◎ = {len(bun):,}  /  2026 kekka 突合成功 = {len(df):,}  /  実週数 = {nweeks}")
if len(df) < 30:
    print("突合数が少なすぎ。join key を要確認。"); raise SystemExit
d = df.sort_values("pw", ascending=False).reset_index(drop=True); N = len(d)
print(f"\n=== 実 serve(2026) ◎複勝 信頼度選別フロンティア  (全{N}R / {nweeks}週) ===")
print(f"{'cover':>6} {'R数':>5} {'週R':>5} {'的中率':>7} {'回収率':>7} {'pw≥':>6}")
for cov in [100, 60, 50, 40, 30, 25, 20, 15, 10]:
    k = max(1, int(round(N * cov / 100))); s = d.iloc[:k]
    print(f"{cov:>5}% {k:>5} {k/nweeks:>5.1f} {s.hit.mean()*100:>6.1f}% "
          f"{s.pay.sum()/(len(s)*100)*100:>6.1f}% {s.pw.iloc[-1]:>6.3f}")
