# -*- coding: utf-8 -*-
"""
learn_bet_v3.py — 「オッズの動き(資金フロー)は静的価格を超えるか」決定的検証 (leak-safe)
================================================================================
今日の結論: v6の120公開特徴+NNでも市場価格(9時)を OOS で超えない(効率的市場)。
残った唯一の合法な"市場が織り込みきれてない可能性"= オッズ軌跡/OU(資金フロー)。
これは既に leak-safe(9時以前のみ)で特徴化済(odds_traj/odds_ou)だが v6 は未使用。

検証: 市場価格(9時) baseline に対し、money-flow特徴を足して OOS2025 で
      logloss/AUC が改善するか(=価格を超える増分情報があるか)。
      + 価格のみGBM vs 価格+動きGBM の限界寄与。EV選抜ROIも。

leak規約: v6=train〜2022/valid2023。価値学習=2024 / 評価=2025。
          価格=9時オッズ。動き特徴=9時以前前売りのみ(odds_traj_leakcheck PASS済)。
実行: PYTHONUTF8=1 python learn_bet_v3.py
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
BASE = Path(__file__).parent
CACHE = BASE / "data/_xai/bet_dyn_rows.parquet"
FIT_YEAR, EVAL_YEAR = 2024, 2025


def build():
    import joblib, pl_probs as PL
    from audit_ev_bin_roi import load_tanpuk, _rid16, MASTER_CSV, COL_RID, COL_JYUN, COL_BAN
    from backtest_pl_ev import load_payouts
    b = joblib.load(BASE/"models/unified_rank_v6.pkl"); model,feats,encs=b["model"],b["feature_cols"],b["encoders"]
    print("[load] master"); df=pd.read_csv(MASTER_CSV,encoding="utf-8-sig",low_memory=False)
    df[COL_JYUN]=pd.to_numeric(df[COL_JYUN],errors="coerce"); df=df.dropna(subset=[COL_JYUN,COL_RID,"split"])
    df=df[df["split"]=="test"].copy()
    for c,le in encs.items():
        if c in df.columns:
            v=df[c].astype(str).fillna("__NaN__"); df[c]=le.transform(v.where(v.isin(set(le.classes_)),"__NaN__"))
    Xf=df[feats].apply(pd.to_numeric,errors="coerce"); df["_score"]=model.predict(Xf.fillna(-9999).values)
    tanpuk=load_tanpuk(); payouts=load_payouts()
    rows=[]
    for rid,g in df.groupby(COL_RID,sort=False):
        if len(g)<5: continue
        rid16=_rid16(rid); yr=int(rid16[:4])
        if yr not in (FIT_YEAR,EVAL_YEAR): continue
        sp=tanpuk.get(rid16); po=payouts.get(rid16)
        if not sp: continue
        w=PL.pl_weights(g["_score"].values); pw=w/w.sum()
        ban=g[COL_BAN].astype(int).values
        for r in range(len(ban)):
            o9=sp["tan9"].get(int(ban[r]))
            if o9 is None or o9<=1.0: continue
            won=1 if (po is not None and int(ban[r])==po["win"]) else 0
            tp=float(po["tansho"]) if (won and po and po.get("tansho") and not pd.isna(po["tansho"])) else 0.0
            rows.append(dict(year=yr,rid=rid16,ban=int(ban[r]),odds=float(o9),p_win=float(pw[r]),won=won,tanpay=tp))
    R=pd.DataFrame(rows)
    tj=pd.read_parquet(BASE/"data/odds_traj_feats.parquet")
    ou=pd.read_parquet(BASE/"data/odds_ou_feats.parquet")
    R=R.merge(tj,on=["rid16","ban"],how="left",suffixes=("","_t")) if "rid16" in R else R.merge(
        tj.rename(columns={"rid16":"rid"}),on=["rid","ban"],how="left")
    R=R.merge(ou.rename(columns={"rid16":"rid"}),on=["rid","ban"],how="left")
    CACHE.parent.mkdir(exist_ok=True); R.to_parquet(CACHE)
    print(f"[cache] {len(R):,} rows → {CACHE}")
    return R


if __name__=="__main__":
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, roc_auc_score
    import lightgbm as lgb
    R = pd.read_parquet(CACHE) if CACHE.exists() else build()
    R["q_raw"]=1.0/R["odds"]; R["q_mkt"]=R["q_raw"]/R.groupby("rid")["q_raw"].transform("sum")
    def lg(p,e=1e-6): p=np.clip(p,e,1-e); return np.log(p/(1-p))
    DYN=["traj_nsnap","traj_has","traj_logchg","traj_range","traj_absmax","traj_revers","traj_rankshift",
         "ou_n","ou_has","ou_net","ou_pathlen","ou_s2n","ou_informed","ou_termvel","ou_termaccel",
         "ou_theta","ou_sigma","ou_equilgap","ou_lastlog","ou_firstlog"]
    DYN=[c for c in DYN if c in R.columns]
    cov=R["traj_has"].fillna(0).mean() if "traj_has" in R else 0
    tr=R[R.year==FIT_YEAR].copy(); ev=R[R.year==EVAL_YEAR].copy()
    base_ll=log_loss(ev.won,np.clip(ev.q_mkt,1e-6,1-1e-6)); base_auc=roc_auc_score(ev.won,ev.q_mkt)
    print(f"rows={len(R):,} dyn特徴={len(DYN)} 軌跡カバー率={cov*100:.0f}%")
    print(f"fit2024={len(tr):,} eval2025={len(ev):,}\n市場(9時)単独: logloss={base_ll:.4f} AUC={base_auc:.4f}\n")

    res={}
    # Benter p_win+price
    lr=LogisticRegression(max_iter=2000).fit(np.column_stack([lg(tr.p_win),lg(tr.q_mkt)]),tr.won)
    p=lr.predict_proba(np.column_stack([lg(ev.p_win),lg(ev.q_mkt)]))[:,1]; res["Benter(p_win+価格)"]=(p,)
    # GBM price+pwin (no dynamics)
    base_cols=["p_win","q_mkt"]
    g1=lgb.train(dict(objective="binary",learning_rate=0.03,num_leaves=31,min_data_in_leaf=300,
        feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,verbose=-1),
        lgb.Dataset(tr[base_cols],tr.won),num_boost_round=400)
    res["GBM(p_win+価格)"]=(g1.predict(ev[base_cols]),)
    # GBM price+pwin+DYNAMICS
    cols2=base_cols+DYN
    g2=lgb.train(dict(objective="binary",learning_rate=0.03,num_leaves=63,min_data_in_leaf=300,
        feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,verbose=-1),
        lgb.Dataset(tr[cols2].fillna(-9999),tr.won),num_boost_round=600)
    res["GBM(p_win+価格+動き)"]=(g2.predict(ev[cols2].fillna(-9999)),)
    # GBM price+DYNAMICS only (no model prob) : 資金フロー単独で価格を超えるか
    cols3=["q_mkt"]+DYN
    g3=lgb.train(dict(objective="binary",learning_rate=0.03,num_leaves=63,min_data_in_leaf=300,
        feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,verbose=-1),
        lgb.Dataset(tr[cols3].fillna(-9999),tr.won),num_boost_round=600)
    res["GBM(価格+動き)"]=(g3.predict(ev[cols3].fillna(-9999)),)

    print(f"{'モデル':26s}{'logloss':>9s}{'vs市場':>9s}{'AUC':>8s}")
    print(f"{'市場(9時)単独':26s}{base_ll:>9.4f}{'--':>9s}{base_auc:>8.4f}")
    for k,(p,) in res.items():
        print(f"{k:26s}{log_loss(ev.won,p):>9.4f}{base_ll-log_loss(ev.won,p):>+9.4f}{roc_auc_score(ev.won,p):>8.4f}")
    print("(vs市場>0 = 価格を超える増分情報あり=資金フローにエッジ)")

    # feature importance of dynamics in g2
    imp=pd.Series(g2.feature_importance(importance_type="gain"),index=cols2).sort_values(ascending=False)
    print("\n動き特徴の寄与(GBM gain, 上位):")
    print(imp[[c for c in imp.index if c in DYN]].head(8).round(0).to_string())

    # EV ROI for the dynamics model
    def roi(EVdf,pcol):
        E=EVdf.copy(); E["_pb"]=E[pcol]/E.groupby("rid")[pcol].transform("sum"); E["_ev"]=E["_pb"]*E["odds"]
        out=[]
        for t in [1.0,1.1,1.2]:
            s=E[E["_ev"]>=t]; n=len(s)
            if n==0: out.append(f"EV≥{t}:n0"); continue
            r=s.tanpay.values; roi=r.mean()/100; se=r.std(ddof=1)/np.sqrt(n)/100 if n>1 else 0
            out.append(f"EV≥{t}:n{n} ROI{roi*100:.0f}%[{(roi-1.96*se)*100:.0f},{(roi+1.96*se)*100:.0f}]")
        return " ".join(out)
    evd=ev.copy(); evd["_p2"]=res["GBM(p_win+価格+動き)"][0]; evd["_p3"]=res["GBM(価格+動き)"][0]
    print("\n単勝EV選抜ROI(2025 OOS):")
    print("  GBM(p_win+価格+動き):",roi(evd,"_p2"))
    print("  GBM(価格+動き)      :",roi(evd,"_p3"))
