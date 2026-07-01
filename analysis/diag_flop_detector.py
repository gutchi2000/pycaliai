# -*- coding: utf-8 -*-
"""
diag_flop_detector.py
=====================
Q9  低entropy×高gap ゲートで ◎複勝率/ROI/残存率 はどう変わるか
Q10 entropy/gap/頭数/馬場 だけで ◎飛び を何%検知できるか (ROC-AUC/PR-AUC)
Q11 ◎飛びモデル(entropy/gap/std/頭数/馬場) の識別力
Q12 飛び確率上位X% を除外したら ROI はどうなるか
honest OOS: flop模型は train=2024 / test=2025。ゲート閾値も2024で決め2025で評価。
入力: reports/_diag_upset_oos.parquet (diag_upset_decomposition.py が生成)
"""
import sys, io
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
import lightgbm as lgb

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
BASE = Path(r"E:\PyCaLiAI")

df = pd.read_parquet(BASE / "reports/_diag_upset_oos.parquet")
df["baba_n"] = pd.to_numeric(df["baba"], errors="coerce").fillna(3)
df["flop"] = 1 - df["hon_top3"]          # 1 = ◎飛び(複勝圏外)
tr = df[df.year == 2024].copy()
te = df[df.year == 2025].copy()
print(f"train(2024)={len(tr):,} flop率={tr.flop.mean()*100:.1f}% / test(2025)={len(te):,} flop率={te.flop.mean()*100:.1f}%\n")

def roi(sub, col, stake=100):
    return np.nan if len(sub)==0 else round(sub[col].sum()/(stake*len(sub))*100, 1)
def line(sub, label, base_n):
    return dict(gate=label, 残存=f"{len(sub)/base_n*100:.0f}%", n=len(sub),
        complement率=round(sub.hon_top3.mean()*100,1),
        単ROI=roi(sub,'tan_ret'), 複ROI=roi(sub,'fuk_ret'),
        馬連ROI=roi(sub,'umaren_hono_ret'), 流しROI=roi(sub,'naga_ret',400))

# ===================== Q9 =====================
print("="*72); print("Q9  低entropy × 高gap ゲート (閾値=2024分位, 評価=2025)"); print("="*72)
e = {q: tr.entropy.quantile(q) for q in (.25,.40,.50)}
g = {q: tr.gap_12.quantile(q) for q in (.50,.60,.75)}
N = len(te)
gates = [
    ("無し(全2025)", te),
    ("ent<=Q50 & gap>=Q50", te[(te.entropy<=e[.50]) & (te.gap_12>=g[.50])]),
    ("ent<=Q40 & gap>=Q60", te[(te.entropy<=e[.40]) & (te.gap_12>=g[.60])]),
    ("ent<=Q25 & gap>=Q75", te[(te.entropy<=e[.25]) & (te.gap_12>=g[.75])]),
    ("(参考)gap>=Q75のみ",  te[te.gap_12>=g[.75]]),
    ("(参考)ent<=Q25のみ",  te[te.entropy<=e[.25]]),
]
print(pd.DataFrame([line(s,l,N) for l,s in gates]).to_string(index=False))

# ===================== Q10 / Q11 =====================
def evaluate(name, feats, model):
    Xtr, ytr = tr[feats].values, tr.flop.values
    Xte, yte = te[feats].values, te.flop.values
    if isinstance(model, LogisticRegression):
        sc = StandardScaler().fit(Xtr)
        model.fit(sc.transform(Xtr), ytr)
        p = model.predict_proba(sc.transform(Xte))[:,1]
    else:
        model.fit(Xtr, ytr)
        p = model.predict_proba(Xte)[:,1]
    auc = roc_auc_score(yte, p); ap = average_precision_score(yte, p)
    print(f"\n[{name}]  feats={feats}")
    print(f"  ROC-AUC={auc:.4f}  PR-AUC={ap:.4f}  (base flop率={yte.mean()*100:.1f}%)")
    # 操作点: 飛び確率上位k%をフラグ → 再現率/精度
    order = np.argsort(-p)
    print("  上位k%をフラグ:  捕捉(recall)  精度(precision)  flop率")
    for k in (0.10,0.20,0.30,0.40):
        m = order[:int(len(p)*k)]
        flagged = yte[m]
        rec = flagged.sum()/yte.sum(); prec = flagged.mean()
        print(f"    top{int(k*100):>2}%   {rec*100:5.1f}%       {prec*100:5.1f}%      (全体{yte.mean()*100:.1f}%)")
    return p

print("\n"+"="*72); print("Q10  entropy/gap/頭数/馬場 のみで ◎飛び検知 (Logistic, OOS)"); print("="*72)
feats4 = ["entropy","gap_12","n","baba_n"]
for fcol in feats4:
    a = roc_auc_score(te.flop, te[fcol]); a = max(a, 1-a)
    print(f"  単変量 {fcol:8s} ROC-AUC={a:.4f}")
evaluate("Q10 Logistic 4feat", feats4, LogisticRegression(max_iter=1000))

print("\n"+"="*72); print("Q11  ◎飛びモデル entropy/gap/std/頭数/馬場 (LightGBM, OOS)"); print("="*72)
feats5 = ["entropy","gap_12","score_std","n","baba_n"]
lgbm = lgb.LGBMClassifier(n_estimators=200, num_leaves=15, learning_rate=0.05,
                          min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
                          random_state=42, verbose=-1)
p_te = evaluate("Q11 LightGBM 5feat", feats5, lgbm)
imp = pd.Series(lgbm.feature_importances_, index=feats5).sort_values(ascending=False)
print("  feature importance:", {k: int(v) for k,v in imp.items()})

# ===================== Q12 =====================
print("\n"+"="*72); print("Q12  飛び確率上位X%を除外したら ROI はどうなるか (2025, LightGBMのp)"); print("="*72)
te = te.copy(); te["pflop"] = p_te
base_n = len(te)
rows = []
for frac in (0.0, 0.10, 0.20, 0.30, 0.40):
    thr = np.quantile(te.pflop, 1-frac) if frac>0 else np.inf
    keep = te[te.pflop < thr] if frac>0 else te
    drop = te[te.pflop >= thr] if frac>0 else te.iloc[:0]
    rows.append(dict(除外=f"top{int(frac*100)}%", 残存=f"{len(keep)/base_n*100:.0f}%", n=len(keep),
        complement率=round(keep.hon_top3.mean()*100,1),
        除外側complement率=(round(drop.hon_top3.mean()*100,1) if len(drop) else np.nan),
        単ROI=roi(keep,'tan_ret'), 複ROI=roi(keep,'fuk_ret'),
        馬連ROI=roi(keep,'umaren_hono_ret'), 流しROI=roi(keep,'naga_ret',400)))
print(pd.DataFrame(rows).to_string(index=False))
print("\n[読み] complement率=◎複勝圏率。除外側complement率が低い=飛びを正しく除外。"
      " ROIが控除率(~80%)を超えれば参加ゲートが収益化、横ばいなら市場priced。")
