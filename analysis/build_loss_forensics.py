# -*- coding: utf-8 -*-
"""
build_loss_forensics.py — 全ベット台帳 × bundle(印/確率/オッズ) × kekka(着順/枠/払戻) を
1行/ベット + 1行/レース に結合し、損失の検死(原因追求)用データセットを作る。
出力: data/_forensic/bet_forensics.csv, race_forensics.csv
ユーザー仮説: 印通り来たか / 来たけどトリガミか / 穴に寄せすぎか / 枠が違えば獲れたか / 馬場
"""
import sys, io, json, glob, re, os
import pandas as pd, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
BASE = "E:/PyCaLiAI"
os.makedirs(f"{BASE}/data/_forensic", exist_ok=True)

def rid16(x): return re.sub(r"\D", "", str(x))[:16]

# ---------- 1. bundles ----------
bundles = {}  # rid -> race dict
for f in glob.glob(f"{BASE}/reports/cowork_input/*_bundle.json"):
    d = json.load(open(f, encoding="utf-8"))
    for r in d.get("races", []):
        rid = rid16(r.get("race_id") or r.get("race_meta", {}).get("race_id", ""))
        if len(rid) != 16: continue
        horses = {}
        marks = {}
        for h in r.get("horses", []):
            ub = h.get("umaban")
            if ub is None: continue
            ub = int(ub)
            horses[ub] = {
                "mark": h.get("mark", ""), "ai_rank": h.get("ai_rank"),
                "p_win": h.get("p_win"), "p_sho": h.get("p_sho"),
                "odds": h.get("tansho_odds"), "ai_vs_market": h.get("ai_vs_market"),
            }
            if h.get("mark"): marks.setdefault(h["mark"], ub)
        bundles[rid] = {"meta": r.get("race_meta", {}), "conf": r.get("race_confidence", {}),
                        "horses": horses, "marks": marks}
print(f"[bundle] races={len(bundles)}")

# ---------- 2. kekka ----------
def load_kekka(f):
    return pd.read_csv(f, encoding="cp932", low_memory=False)
kk_files = glob.glob(f"{BASE}/data/kekka/202*.csv") + [f"{BASE}/data/kekka/kekka_20260418-20260517.csv"]
seen = set(); frames = []
for f in kk_files:
    if not os.path.exists(f): continue
    try:
        df = load_kekka(f)
        if "レースID(新)" not in df.columns: continue
        frames.append(df)
    except Exception as e:
        print("kekka skip", f, str(e)[:40])
kall = pd.concat(frames, ignore_index=True)
kall["rid"] = kall["レースID(新)"].map(rid16)
kall["chaku"] = pd.to_numeric(kall["確定着順"], errors="coerce")
kall = kall.dropna(subset=["rid"]).drop_duplicates(subset=["rid", "馬番"])
races_k = {}
for rid, g in kall.groupby("rid"):
    g = g.copy()
    finish = {int(b): (int(c) if pd.notna(c) else None) for b, c in zip(g["馬番"], g["chaku"]) if pd.notna(b)}
    draw = {int(b): (int(w) if pd.notna(w) else None) for b, w in zip(g["馬番"], g["枠番"]) if pd.notna(b)}
    def pos(p):
        m = g[g["chaku"] == p]
        return int(m["馬番"].iloc[0]) if len(m) else None
    def num(col):
        v = pd.to_numeric(g[col], errors="coerce").dropna()
        return float(v.iloc[0]) if len(v) else None
    fuk = {int(b): (float(re.sub(r"[()]", "", str(x))) if str(x).strip() not in ("", "nan") and not str(x).startswith("(") else None)
           for b, x in zip(g["馬番"], g["複勝配当"])}
    tan = {}
    for b, x in zip(g["馬番"], g["単勝配当"]):
        s = str(x)
        if s.strip() not in ("", "nan") and not s.startswith("("):
            try: tan[int(b)] = float(s)
            except: pass
    races_k[rid] = {"finish": finish, "draw": draw, "n": len(finish),
                    "win": pos(1), "plc": pos(2), "sho": pos(3),
                    "fuku": fuk, "tan": tan,
                    "umaren": num("馬連"), "umatan": num("馬単")}
print(f"[kekka] races={len(races_k)}")

# ---------- 3. bets ----------
bets = json.load(open(f"{BASE}/data/cowork_results.json", encoding="utf-8"))["bets"]
print(f"[bets] {len(bets)}")

def parse_legs(sel):
    """買い目 -> 組のリスト(各組=馬番tuple)。単勝/複勝は単一。"""
    out = []
    for c in str(sel).replace("→", "-").replace("＞", "-").replace(">", "-").split(","):
        nums = [int(x) for x in re.findall(r"\d+", c)]
        if nums: out.append(tuple(nums))
    return out

rows = []
for b in bets:
    rid = rid16(b["race_id"])
    bu = bundles.get(rid); kk = races_k.get(rid)
    legs = parse_legs(b.get("買い目", ""))
    picks = sorted(set(x for leg in legs for x in leg))
    stake = b.get("購入額", 0); ret = b.get("払戻", 0) or 0; hit = int(b.get("的中", 0) or 0)
    row = {"date": b["date"], "era": "T10" if b["date"] in ("20260613", "20260614") else "manual",
           "rid": rid, "場所": b.get("場所"), "R": b.get("R"), "btype": b.get("馬券種"),
           "買い目": b.get("買い目"), "stake": stake, "ret": ret, "hit": hit, "npts": len(legs),
           "torigami": int(hit == 1 and ret < stake), "pnl": ret - stake}
    # 印・確率・オッズ（買い目の馬の代表＝最良p_winの馬）
    if bu:
        pm = []
        for ub in picks:
            h = bu["horses"].get(ub, {})
            pm.append((ub, h.get("p_win") or 0, h.get("odds"), h.get("mark", ""), h.get("ai_vs_market")))
        if pm:
            pm.sort(key=lambda t: -(t[1] or 0))
            top = pm[0]
            row["pick_best_pwin"] = top[1]; row["pick_best_odds"] = top[2]
            row["pick_best_mark"] = top[3]; row["pick_best_avm"] = top[4]
            row["pick_mean_odds"] = np.mean([t[2] for t in pm if t[2]]) if any(t[2] for t in pm) else None
            row["pick_max_odds"] = max([t[2] for t in pm if t[2]], default=None)
            row["pick_marks"] = "".join(sorted(set(t[3] for t in pm if t[3])))
            row["n_unmarked_picks"] = sum(1 for t in pm if not t[3])
        row["field_chaos"] = bu["conf"].get("field_chaos_score")
        row["ai_mkt_agree"] = bu["conf"].get("ai_market_agreement")
        row["course"] = bu["meta"].get("course"); row["class"] = bu["meta"].get("class")
        row["field_size"] = bu["meta"].get("field_size")
    # 着順・枠・勝ち馬
    if kk:
        fin = kk["finish"]
        # 印の着順
        for mk, key in [("◎", "hon"), ("〇", "tai"), ("○", "tai"), ("▲", "san")]:
            if bu and mk in bu["marks"]:
                ub = bu["marks"][mk]
                row[f"{key}_uma"] = ub; row[f"{key}_chaku"] = fin.get(ub)
        row["winner"] = kk["win"]
        if bu and kk["win"] is not None:
            wh = bu["horses"].get(kk["win"], {})
            row["winner_mark"] = wh.get("mark", "") or "無"
            row["winner_odds"] = wh.get("odds"); row["winner_pwin"] = wh.get("p_win")
            row["winner_ai_rank"] = wh.get("ai_rank")
            row["winner_draw"] = kk["draw"].get(kk["win"])
        # 買い目の馬の最高着順（来てたか）
        pick_chaku = [fin.get(ub) for ub in picks if fin.get(ub) is not None]
        row["pick_best_chaku"] = min(pick_chaku) if pick_chaku else None
        row["pick_draws"] = ",".join(str(kk["draw"].get(ub)) for ub in picks)
    rows.append(row)

bf = pd.DataFrame(rows)
bf.to_csv(f"{BASE}/data/_forensic/bet_forensics.csv", index=False, encoding="utf-8-sig")
print(f"[out] bet_forensics.csv rows={len(bf)} join_bundle={bf['pick_best_pwin'].notna().mean():.2f} join_kekka={bf['winner'].notna().mean():.2f}")

# ---------- 4. 速報サニティ ----------
print("\n=== サニティ ===")
print("era別:", bf.groupby("era").agg(n=("stake","size"), roi=("ret", lambda s: round(s.sum()/bf.loc[s.index,"stake"].sum()*100,1)), hit=("hit","mean")).to_dict("index"))
print("トリガミ件数:", int(bf["torigami"].sum()), "/ 的中", int(bf["hit"].sum()), "件中")
print("トリガミ損失:", int((bf[bf.torigami==1]["stake"]-bf[bf.torigami==1]["ret"]).sum()), "円相当の取りこぼし(的中したのに赤)")
