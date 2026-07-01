# -*- coding: utf-8 -*-
"""
build_policy_substrate.py — ベッティング・ポリシー探索の土台を 1 回だけ焼く
============================================================================
trigami_floor_gate.py と同じ v6 推論を 1 回回し、各レースについて
「ポリシー探索に必要な最小量」だけを抽出して data/_policy/sanpuku_substrate.pkl に保存。
これにより探索エージェントは 515MB master も model も再ロードせず、
小さな pkl + ベクトル化 scorer (analysis/_policy_substrate.py) だけで全ポリシーを評価できる。

各レース substrate:
  year       : 2024 / 2025 (どちらも真OOS, 学習は2022まで)
  win_rank   : 三連複 prob-first 順で勝ち組が何番目か(1始まり, >KMAX は 9999=圏外)
  pay        : 三連複 実払戻 (100円あたり円)
  p_fav      : prob-first 最確組の確率 (= トリガミ床判定に使う最安組)
  chaos      : レース混戦度 (p_win 正規化エントロピー 0-1)
  n_horses   : 出走頭数
  probs_topK : 上位 KMAX 組の確率 (降順, 0 padding) ← mass/threshold ポリシー用
"""
import sys, io, re
from itertools import combinations, permutations
from pathlib import Path
import joblib, numpy as np, pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
BASE = Path(r"E:\PyCaLiAI"); sys.path.insert(0, str(BASE))
import pl_probs as PL
from backtest_pl_ev import all_sanrenpuku_tensor, load_payouts, COL_RID, COL_BAN, COL_JYUN

KMAX = 40
TAKEOUT = 0.775
def _rid16(x): return re.sub(r"\D", "", str(x))[:16]

print("[load] v6 / master(test) / payouts ...")
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

rows = {"year": [], "win_rank": [], "pay": [], "p_fav": [], "chaos": [], "n_horses": []}
probs_topK = []
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
    pw = np.asarray(PL.all_tansho(w), float); pw = pw / pw.sum()
    chaos = float(-(pw * np.log(np.clip(pw, 1e-12, 1))).sum() / np.log(n)) if n > 1 else 0.0
    P3 = all_sanrenpuku_tensor(w)
    combos = list(combinations(range(n), 3)); ci = np.array(combos)
    pk = np.zeros(len(combos))
    for a, bb, c in permutations(range(3)):
        pk += P3[ci[:, a], ci[:, bb], ci[:, c]]
    order = np.argsort(-pk)
    win_combo = frozenset((b2i[po["win"]], b2i[po["plc"]], b2i[po["sho"]]))
    rank = {frozenset(combos[idx]): r for r, idx in enumerate(order[:KMAX])}
    wr = rank.get(win_combo); win_rank = (wr + 1) if wr is not None else 9999
    top = pk[order[:KMAX]]
    if len(top) < KMAX: top = np.concatenate([top, np.zeros(KMAX - len(top))])
    rows["year"].append(int(g["year"].iloc[0])); rows["win_rank"].append(win_rank)
    rows["pay"].append(float(sp)); rows["p_fav"].append(float(pk[order[0]]))
    rows["chaos"].append(chaos); rows["n_horses"].append(int(n))
    probs_topK.append(top)

sub = {k: np.asarray(v) for k, v in rows.items()}
sub["probs_topK"] = np.vstack(probs_topK)
sub["KMAX"] = KMAX; sub["TAKEOUT"] = TAKEOUT
out = BASE / "data/_policy/sanpuku_substrate.pkl"
out.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(sub, out)
n24 = int((sub["year"] == 2024).sum()); n25 = int((sub["year"] == 2025).sum())
print(f"[out] {out}  N={len(sub['pay'])}  2024={n24}  2025={n25}")
print(f"  win_rank<=25 率={float((sub['win_rank']<=25).mean()):.3f}  圏外(9999)={int((sub['win_rank']==9999).sum())}")
