# -*- coding: utf-8 -*-
"""
trigami_floor_gate.py — ユーザー定義「トリガミ床」ルールを AI に組み込んで OOS 実測
=====================================================================================
ルール (言語化済):
  三連複フォメを N 点買うとき、「一番人気の組(=最安=最確の組)の払戻 > 総点数コスト」
  なら買う、下回るなら買わない/削る。最安が床を超える ⟹ どの当たりも必ずプラス
  ⟹ トリガミ(当たってるのにマイナス)が構造的に消える。儲けでなく「的中の尊厳」の条件。

AI への翻訳 (serve 可能な量だけ使う):
  最安の組 = モデル三連複確率が最大の組 p_fav。
  その market 払戻 ≈ 0.775 / p_fav (0.775 = 三連複還元率)。
  ゲート: 0.775 / p_fav >= N(点数)  ⟺  当たれば必ず床を超える。

答え合わせ:
  ゲートで通したレースを 2024-2025 OOS の実払戻 po["sanrenpuku"] で settle し、
  実際にトリガミ(hit かつ payout < cost)が消えたか・代償(参加減/ROI)を実測する。

  ※ p は v6 PL の生確率。market != model のズレは「残ったトリガミ」として正直に出る。
  ※ 三連複オッズの過去全グリッドは無いので model-implied を market proxy に使う(serve一致)。
"""
import sys, io, os, re
from itertools import combinations, permutations
from pathlib import Path
import joblib, numpy as np, pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
BASE = Path(r"E:\PyCaLiAI"); sys.path.insert(0, str(BASE))
import pl_probs as PL
from backtest_pl_ev import all_sanrenpuku_tensor, load_payouts, COL_RID, COL_BAN, COL_JYUN

TAKEOUT = 0.775          # 三連複還元率 (控除率 22.5%)
def _rid16(x): return re.sub(r"\D", "", str(x))[:16]

# ---------------- load v6 + master test + payouts ----------------
print("[load] v6 model / master(test) / payouts ...")
b = joblib.load(BASE / "models/unified_rank_v6.pkl")
model, feats, encs = b["model"], b["feature_cols"], b["encoders"]
df = pd.read_csv(BASE / "data/master_v2_20130105-20251228.csv", encoding="utf-8-sig", low_memory=False)
df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
df = df.dropna(subset=[COL_JYUN, COL_RID, "split"]); df = df[df["split"] == "test"].copy()
df["year"] = df[COL_RID].astype(str).str[:4].astype(int)
for c, le in encs.items():
    if c in df.columns:
        v = df[c].astype(str).fillna("__NaN__")
        df[c] = le.transform(v.where(v.isin(set(le.classes_)), "__NaN__"))
X = df[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
df["_score"] = model.predict(X)
pays = load_payouts()
print(f"  races(test)={df[COL_RID].nunique():,}  payouts={len(pays):,}")

# ---------------- per-race records ----------------
# 各レース: 全三連複組(index)の model 確率 + prob-first 順 + 勝ち組 index + 実払戻
races = {2024: [], 2025: []}
for rid, g in df.groupby(COL_RID, sort=False):
    if len(g) < 5: continue
    rid16 = _rid16(rid); po = pays.get(rid16)
    if po is None: continue
    sp = po.get("sanrenpuku")
    if sp is None or not (sp == sp) or sp <= 0: continue
    g = g.sort_values(COL_BAN).reset_index(drop=True)
    ban = g[COL_BAN].astype(int).values; n = len(ban)
    b2i = {int(ban[i]): i for i in range(n)}
    if not all(k in b2i for k in (po["win"], po["plc"], po["sho"])): continue
    w = PL.pl_weights(g["_score"].values)
    P3 = all_sanrenpuku_tensor(w)
    combos = list(combinations(range(n), 3))
    pk = np.zeros(len(combos))
    ci = np.array(combos)
    for a, bb, c in permutations(range(3)):
        pk += P3[ci[:, a], ci[:, bb], ci[:, c]]
    order = np.argsort(-pk)                       # prob-first 順 (combo index)
    win_combo = frozenset((b2i[po["win"]], b2i[po["plc"]], b2i[po["sho"]]))
    races[int(g["year"].iloc[0])].append({
        "rid": rid16, "combos": combos, "pk": pk, "order": order,
        "p_fav": float(pk[order[0]]), "win": win_combo, "pay": float(sp),
    })
print(f"  usable 2024={len(races[2024])}R  2025={len(races[2025])}R")

# ---------------- settle helpers ----------------
def settle(recs, pick_fn):
    """pick_fn(rec) -> list[combo-index] (買う組). None/[] は見送り。"""
    cost = ret = hits = trig = clean = nbet = miokuri = 0
    clean_gain = 0.0
    for r in recs:
        picks = pick_fn(r)
        if not picks:
            miokuri += 1; continue
        K = len(picks); c = K * 100
        hit = r["win"] in {frozenset(r["combos"][i]) for i in picks}
        pay = r["pay"] if hit else 0.0
        cost += c; ret += pay; nbet += 1
        if hit:
            hits += 1
            if pay < c: trig += 1                 # トリガミ: 当たったのにマイナス
            else: clean += 1; clean_gain += (pay - c)
    return {
        "nbet": nbet, "miokuri": miokuri,
        "hit": round(hits / nbet * 100, 1) if nbet else 0,
        "roi": round(ret / cost * 100, 1) if cost else 0,
        "trig": trig, "clean": clean,
        "trig_rate_of_hit": round(trig / hits * 100, 1) if hits else 0,
        "avg_clean_gain": int(clean_gain / clean) if clean else 0,
        "avg_pts": round(cost / 100 / nbet, 1) if nbet else 0,
    }

def probfirst(rec, K):           return list(rec["order"][:K])
def gate_pass(rec, K):           return (TAKEOUT / rec["p_fav"]) >= K
def probfirst_gated(rec, K):     return list(rec["order"][:K]) if gate_pass(rec, K) else []
def adaptive(rec, kmax=25, kmin=1):
    kstar = int(TAKEOUT / rec["p_fav"])          # 床を超える最大点数
    kstar = max(kmin, min(kmax, kstar))
    return list(rec["order"][:kstar])

# ---------------- report ----------------
def block(title, recs):
    tot = len(recs)
    print(f"\n{'='*84}\n{title}  (対象 {tot}R, 三連複 実払戻 settle)\n{'='*84}")
    print(f"{'戦略':<26}{'参加R':>7}{'見送':>6}{'的中%':>7}{'ROI%':>7}"
          f"{'ﾄﾘｶﾞﾐ数':>8}{'当ﾄﾘ率%':>8}{'平均勝額':>9}{'平均点':>7}")
    def row(name, fn):
        s = settle(recs, fn)
        print(f"{name:<26}{s['nbet']:>7}{s['miokuri']:>6}{s['hit']:>7}{s['roi']:>7}"
              f"{s['trig']:>8}{s['trig_rate_of_hit']:>8}{s['avg_clean_gain']:>9}{s['avg_pts']:>7}")
        return s
    print("-- ① 素の prob-first (ゲート無し / 全レース参加) --")
    base = {K: row(f"prob-first {K}点", lambda r, K=K: probfirst(r, K)) for K in (25, 12, 8, 4)}
    print("-- ② トリガミ床ゲート (床を割るレースは参加しない) --")
    gated = {K: row(f"床ゲート {K}点", lambda r, K=K: probfirst_gated(r, K)) for K in (25, 12, 8, 4)}
    print("-- ③ 適応点数 (床を超える最大点数を自動で買う = 君のルールの正しい使い方) --")
    row("適応 (max25)", lambda r: adaptive(r, 25))
    row("適応 (max12)", lambda r: adaptive(r, 12))
    return base, gated

block("2025 OOS", races[2025])
block("2024 OOS", races[2024])
block("2024+2025 OOS", races[2024] + races[2025])
print("\n[読み] ②③ で『当ﾄﾘ率%(当たったのにマイナスの割合)』が①からどれだけ落ちたか= 君の"
      "ルールの効き目。ROI は控除率の壁で改善しない前提。参加R の減り= 払う代償。")

# ---- 可視化用 正確内訳 ----
allr = races[2024] + races[2025]
def breakdown(fn):
    clean=trig=miss=0
    for r in allr:
        picks=fn(r)
        if not picks: miss+=1; continue   # 適応は常に買うので見送りは出ない
        K=len(picks); hit=r["win"] in {frozenset(r["combos"][i]) for i in picks}
        if not hit: miss+=1
        elif r["pay"] < K*100: trig+=1
        else: clean+=1
    return clean,trig,miss
print("\n[VIZ] total_races", len(allr))
print("[VIZ] before_25pt", breakdown(lambda r: probfirst(r,25)))
print("[VIZ] adaptive25", breakdown(lambda r: adaptive(r,25)))
