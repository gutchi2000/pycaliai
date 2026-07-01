# -*- coding: utf-8 -*-
"""
ablation_aptitude_asof.py — 距離適性/会場/血統 を as-of(leak-safe) で足して v6 を再学習し ΔAUC/◎top3 を測る
==========================================================================================================
ユーザー仮説「過去走60%は過多。距離適性・会場・血統を詰めれば伸びる」を本番v6土俵で決着。
設計:
  - baseline = v6 の実 feature_cols(120) を v6 best_params/label/sample_weight で再学習
  - augmented = baseline + as-of leak-safe 新特徴9本 を同一params/seedで再学習
  - test(2024+2025)で ◎複勝(hon_top3)/◎単勝/AUC(fukusho_flag)/ECE_high を比較
リーク安全(v8 affinityの自レース込みleak反省):
  - 馬個体特徴: groupby(馬,条件) を date順 cumulative、当該行はshift/減算で除外 (馬は1日1走=同日leak無)
  - 血統特徴: (種牡馬,条件,日付) 日次集計 → cumsum − 当該日 = 厳密に「過去日のみ」(同日他馬もleakさせない)
実行: PYTHONUTF8=1 ./venv311/Scripts/python.exe analysis/ablation_aptitude_asof.py
出力: reports/_lever_reverify/ablation_aptitude.log (tee) + reports/ablation_aptitude.json
"""
from __future__ import annotations
import sys, io, json, warnings
from itertools import groupby
from pathlib import Path
import joblib, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
BASE = Path(r"E:\PyCaLiAI"); sys.path.insert(0, str(BASE))
import pl_probs as PL

MASTER = BASE/"data/master_v2_20130105-20251228.csv"
KEKKA  = BASE/"data/kekka_20130105-20251228.csv"
V6     = BASE/"models/unified_rank_v6.pkl"
COL_RID="レースID(新/馬番無)"; COL_JYUN="着順"; COL_BAN="馬番"
SEED=42
DIST_BINS=[0,1300,1600,2000,2400,9999]  # sprint/mile/mid/long/extended

print("[load] master + v6 ...")
b=joblib.load(V6); FEATS6=list(b["feature_cols"]); ENCS=b["encoders"]; P=b["optuna_best_params"]; ALPHA=b.get("sample_weight_alpha",0.0)
df=pd.read_csv(MASTER, encoding="utf-8-sig", low_memory=False)
df[COL_JYUN]=pd.to_numeric(df[COL_JYUN],errors="coerce")
df=df.dropna(subset=[COL_JYUN,COL_RID,"split"]).copy()
df["label"]=np.clip(6-df[COL_JYUN].astype(int),0,5).astype(int)
df["_date"]=pd.to_numeric(df["日付"],errors="coerce").fillna(0).astype(np.int64)
df["_jyun"]=df[COL_JYUN].astype(int); df["_t3"]=(df["_jyun"]<=3).astype(np.int64)
df["_dist"]=pd.to_numeric(df["距離"],errors="coerce")
df["_band"]=pd.cut(df["_dist"],DIST_BINS,labels=False).astype("Int64").astype(str)
print(f"  rows={len(df):,}")

# =========================================================================
# as-of leak-safe 新特徴 (当該行/当該日を厳密に除外)
# =========================================================================
print("[build] as-of leak-safe aptitude features ...")
df=df.sort_values(["血統登録番号","_date"]).reset_index(drop=True)
NEW=[]
def horse_asof(keycols, prefix):
    """groupby(馬,keycols) date順 cumulative。当該行除外(shift/減算)。馬は1日1走=同日leak無。全ベクトル化。"""
    gk=["血統登録番号"]+keycols
    grp=df.groupby(gk, sort=False)
    prior_n=grp.cumcount().astype(float)                      # 当該より前の本数(0始まり=過去本数)
    prior_t3=grp["_t3"].cumsum()-df["_t3"]                    # 当該除く top3 数
    df[f"{prefix}_t3rate"]=np.where(prior_n>0, prior_t3/prior_n.replace(0,np.nan), np.nan)
    df[f"{prefix}_runs"]=prior_n
    df["_sh_tmp"]=grp["_jyun"].shift(1)                       # 当該除外した直前までの着順
    df[f"{prefix}_best"]=df.groupby(gk, sort=False)["_sh_tmp"].cummin()  # 当該除く expanding min (cython)
    df.drop(columns=["_sh_tmp"], inplace=True)
    NEW.extend([f"{prefix}_t3rate",f"{prefix}_runs",f"{prefix}_best"])

horse_asof(["_band"], "asof_dist")        # 距離適性(同距離帯, 全キャリアas-of)
horse_asof(["場所"], "asof_venue")         # 会場適性(同場所)
df=df.sort_values(["血統登録番号","_date"]).reset_index(drop=True)  # re-sort (apply乱す保険)

def ped_asof(pedcol, condcol, name):
    """(血統,条件,日付)日次集計→cumsum−当該日=厳密に過去日のみ(同日他馬もleak無)。"""
    sub=df[[pedcol,condcol,"_date","_t3"]].copy()
    daily=sub.groupby([pedcol,condcol,"_date"],sort=False).agg(n=("_t3","size"),t3=("_t3","sum")).reset_index()
    daily=daily.sort_values([pedcol,condcol,"_date"])
    gg=daily.groupby([pedcol,condcol],sort=False)
    daily["cum_n"]=gg["n"].cumsum(); daily["cum_t3"]=gg["t3"].cumsum()
    daily["bn"]=daily["cum_n"]-daily["n"]; daily["bt3"]=daily["cum_t3"]-daily["t3"]
    daily[name]=np.where(daily["bn"]>0, daily["bt3"]/daily["bn"].replace(0,np.nan), np.nan)
    out=df.merge(daily[[pedcol,condcol,"_date",name]], on=[pedcol,condcol,"_date"], how="left")
    df[name]=out[name].values; NEW.append(name)

ped_asof("種牡馬","芝・ダ","asof_sire_surf_t3")    # 種牡馬×馬場 産駒top3率(過去日のみ)
ped_asof("種牡馬","_band","asof_sire_dist_t3")     # 種牡馬×距離帯
ped_asof("母父馬","芝・ダ","asof_bmsire_surf_t3")  # 母父×馬場
print(f"  new feats({len(NEW)}): {NEW}")

# =========================================================================
# encode (v6 encoders) + split + sample weight
# =========================================================================
for c,le in ENCS.items():
    if c in df.columns:
        v=df[c].astype(str).fillna("__NaN__"); df[c]=le.transform(v.where(v.isin(set(le.classes_)),"__NaN__"))
# winner単勝 sample weight (race-level, v6再現)
kk=pd.read_csv(KEKKA, encoding="cp932", low_memory=False)
kk.columns=["rid_horse","ban","ped","jyun","tansho","fukusho","wakuren","umaren","umatan","sanrenpuku","sanrentan"]
kk["rid_s"]=kk["rid_horse"].astype(str).str[:16]; kk["jyun"]=pd.to_numeric(kk["jyun"],errors="coerce"); kk["tansho"]=pd.to_numeric(kk["tansho"],errors="coerce")
win=kk[kk["jyun"]==1].drop_duplicates("rid_s"); win_pay=dict(zip(win["rid_s"],win["tansho"]))
df["rid_s"]=df[COL_RID].astype(str).str[:16]; df["winner_tansho"]=df["rid_s"].map(win_pay).fillna(100.0)

df["year"]=df[COL_RID].astype(str).str[:4].astype(int)
tr=df[df["split"]=="train"]; vl=df[df["split"]=="valid"]; te=df[df["split"]=="test"]
print(f"  train={len(tr):,} valid={len(vl):,} test={len(te):,}")

PARAMS=dict(objective="lambdarank", lambdarank_truncation_level=5, metric="ndcg", eval_at=[1,3,5],
            learning_rate=P["lr"], num_leaves=P["num_leaves"], max_depth=P["max_depth"],
            min_data_in_leaf=P["min_data_in_leaf"], feature_fraction=P["ff"], bagging_fraction=P["bf"],
            bagging_freq=5, lambda_l1=P["l1"], lambda_l2=P["l2"], verbose=-1, n_jobs=-1, seed=SEED,
            deterministic=True, force_col_wise=True)

def ds(d, feats):
    d=d.sort_values(COL_RID).reset_index(drop=True)
    X=d[feats].apply(pd.to_numeric,errors="coerce").fillna(-9999).values
    y=d["label"].values.astype(int)
    g=np.array([len(list(gr)) for _,gr in groupby(d[COL_RID])])
    w=(1.0+ALPHA*np.log1p(d["winner_tansho"].values/100.0)).astype(float)
    return lgb.Dataset(X,label=y,group=g,weight=w,free_raw_data=False), d

def train(feats, tag):
    print(f"\n[train {tag}] n_feat={len(feats)}")
    dtr,_=ds(tr,feats); dvl,_=ds(vl,feats)
    m=lgb.train(PARAMS, dtr, num_boost_round=2000, valid_sets=[dvl],
                callbacks=[lgb.early_stopping(150,verbose=False)])
    print(f"  best_iter={m.best_iteration} trees={m.num_trees()}")
    return m

def evalu(m, feats, tag):
    d=te.copy(); X=d[feats].apply(pd.to_numeric,errors="coerce").fillna(-9999).values
    d["_score"]=m.predict(X)
    out={}
    for yr,sub in [("2024",d[d["year"]==2024]),("2025",d[d["year"]==2025]),("24+25",d)]:
        nr=0; t1=0; t3=0
        for rid,g in sub.groupby(COL_RID,sort=False):
            if len(g)<3: continue
            sc=g["_score"].values; jy=g[COL_JYUN].astype(int).values
            order=np.argsort(-sc); hon=int(order[0]); nr+=1
            asort=np.argsort(jy); top3={int(asort[0]),int(asort[1]),int(asort[2])}
            if hon==int(asort[0]): t1+=1
            if hon in top3: t3+=1
        fuk=pd.to_numeric(sub["fukusho_flag"],errors="coerce").fillna(0).values
        auc=roc_auc_score(fuk, sub["_score"].values) if len(set(fuk))>1 else float("nan")
        out[yr]=dict(n=nr, tansho=round(t1/nr*100,2) if nr else 0, hon_top3=round(t3/nr*100,2) if nr else 0, auc=round(auc,5))
    # ECE_high (PL p_win, threshold 0.10) on 24+25
    ph=[]; ah=[]
    for rid,g in d.groupby(COL_RID,sort=False):
        if len(g)<5: continue
        w=PL.pl_weights(g["_score"].values); pwin=PL.all_tansho(w); jy=g[COL_JYUN].astype(int).values
        mask=pwin>=0.10
        if mask.sum()>0: ph.extend(pwin[mask].tolist()); ah.extend((jy[mask]==1).astype(int).tolist())
    ece=abs(np.mean(ph)-np.mean(ah)) if len(ph)>100 else float("nan")
    out["ece_high"]=round(float(ece),5)
    print(f"  [{tag}] AUC(24+25)={out['24+25']['auc']}  ◎top3={out['24+25']['hon_top3']}%  ◎単={out['24+25']['tansho']}%  ECE={out['ece_high']}")
    return out

print("\n"+"="*78+"\nABLATION: baseline(v6 120feat) vs augmented(+as-of 距離/会場/血統 9feat)\n"+"="*78)
m_base=train(FEATS6,"baseline")
r_base=evalu(m_base,FEATS6,"baseline")
AUG=FEATS6+NEW
m_aug=train(AUG,"augmented")
r_aug=evalu(m_aug,AUG,"augmented")

# new feats gain share in augmented model
gain=np.array(m_aug.feature_importance("gain"),float); G=gain.sum()
gmap=dict(zip(AUG,gain))
new_gain=sum(gmap.get(f,0) for f in NEW)/G*100
print(f"\n[aug model] 新9特徴の合計gain占有 = {new_gain:.2f}%  (使われてさえいるか)")
for f in sorted(NEW,key=lambda x:-gmap.get(x,0)):
    print(f"    {gmap.get(f,0)/G*100:6.3f}%  {f}")

print("\n"+"="*78+"\nΔ (augmented − baseline) on test 2024+2025\n"+"="*78)
dauc=r_aug["24+25"]["auc"]-r_base["24+25"]["auc"]
dtop3=r_aug["24+25"]["hon_top3"]-r_base["24+25"]["hon_top3"]
dtan=r_aug["24+25"]["tansho"]-r_base["24+25"]["tansho"]
dece=r_aug["ece_high"]-r_base["ece_high"]
print(f"  ΔAUC      = {dauc:+.5f}   (base {r_base['24+25']['auc']} → aug {r_aug['24+25']['auc']})")
print(f"  Δ◎top3    = {dtop3:+.2f}pt (base {r_base['24+25']['hon_top3']}% → aug {r_aug['24+25']['hon_top3']}%)")
print(f"  Δ◎単勝    = {dtan:+.2f}pt")
print(f"  ΔECE_high = {dece:+.5f} (負=改善)")
print(f"  年別◎top3: 2024 {r_base['2024']['hon_top3']}→{r_aug['2024']['hon_top3']} / 2025 {r_base['2025']['hon_top3']}→{r_aug['2025']['hon_top3']}")
print("\n[判定基準] v6採用ゲート: ΔAUC>+0.001 程度かつ◎top3が方向一致で上振れ、でなければ『過去走60%は妥当=詰めても死』確定。")

out={"alpha":ALPHA,"params":P,"new_feats":NEW,"new_gain_pct":round(new_gain,2),
     "baseline":r_base,"augmented":r_aug,
     "delta":{"auc":round(dauc,5),"hon_top3":round(dtop3,2),"tansho":round(dtan,2),"ece_high":round(dece,5)}}
(BASE/"reports/ablation_aptitude.json").write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding="utf-8")
print(f"\n[saved] reports/ablation_aptitude.json")
