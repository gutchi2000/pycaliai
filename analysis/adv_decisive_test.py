# -*- coding: utf-8 -*-
"""
adv_decisive_test.py — the honest baseline.
The screen measured delta_auc of feat vs LGBM[v6_score] (a 1-D scalar).
But all 8 survivors are transforms of raw inputs ALREADY in v6 feature_cols.
A scalar score cannot recombine them; a fresh GBM with the raw inputs can.
Test: delta of each survivor vs LGBM[score + raw_v6_inputs_it_derives_from].
If ~0, the 'signal' is just in-race normalization of kako5_avg_pos that the
model could already extract — i.e. redundant, not new.
"""
import sys, io, json, warnings
from pathlib import Path
import joblib, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

BASE = Path(r"E:\PyCaLiAI")
COL_RID = "レースID(新/馬番無)"
RAW = ["kako5_avg_pos","kako5_best_pos","kako5_std_pos","kako5_race_count","kako5_pos_trend","出走頭数","フルゲート頭数"]

SURV = {
 "kako5_form_favored_rank":"-(df.groupby('レースID(新/馬番無)')['kako5_avg_pos'].rank(method='average') - 1) / (df['出走頭数'].clip(lower=2) - 1)",
 "kako5_fieldz_shrunk_by_count":"(-(df['kako5_avg_pos'] - df.groupby('レースID(新/馬番無)')['kako5_avg_pos'].transform('mean')) / df.groupby('レースID(新/馬番無)')['kako5_avg_pos'].transform('std')) * (df['kako5_race_count'] / (df['kako5_race_count'] + 1))",
 "kako5_pos_z_in_race":"(df['kako5_avg_pos'] - df.groupby('レースID(新/馬番無)')['kako5_avg_pos'].transform('mean')) / df.groupby('レースID(新/馬番無)')['kako5_avg_pos'].transform('std')",
 "kako5_avgpos_field_z":"-(df['kako5_avg_pos'] - df.groupby('レースID(新/馬番無)')['kako5_avg_pos'].transform('mean')) / df.groupby('レースID(新/馬番無)')['kako5_avg_pos'].transform('std')",
 "kako5_seasoned_gate_x_fieldz":"(df['kako5_race_count'] >= 4).astype(float) * (-(df['kako5_avg_pos'] - df.groupby('レースID(新/馬番無)')['kako5_avg_pos'].transform('mean')) / df.groupby('レースID(新/馬番無)')['kako5_avg_pos'].transform('std'))",
 "kako5_bestpos_field_frac":"(df['kako5_best_pos'] - 1) / (df['出走頭数'].clip(lower=2) - 1)",
 "kako5_avg_minus_best_gap":"df['kako5_avg_pos'] - df['kako5_best_pos']",
 "gate_crowding":"df['出走頭数'] / df['フルゲート頭数']",
}

b = joblib.load(BASE/"models/unified_rank_v6.pkl")
model, feats, encs = b["model"], b["feature_cols"], b["encoders"]
df = pd.read_csv(BASE/"data/master_v2_20130105-20251228.csv", encoding="utf-8-sig", low_memory=False)
df = df.dropna(subset=["split"])
dt = df[df["split"]=="test"].copy().reset_index(drop=True)

de = dt.copy()
for c, le in encs.items():
    if c in de.columns:
        v = de[c].astype(str).fillna("__NaN__"); known=set(le.classes_)
        de[c] = le.transform(v.where(v.isin(known), "__NaN__"))
X = de[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
sc = model.predict(X)
y = pd.to_numeric(dt["fukusho_flag"], errors="coerce").values.astype(float)
yr = pd.to_datetime(dt["date_dt"], errors="coerce").dt.year.values
rids = dt[COL_RID].values
valid = np.isfinite(sc)&np.isfinite(y)

raw_arr = {c: pd.to_numeric(dt[c], errors="coerce").values.astype(float) for c in RAW}
fv = {}
for nm, expr in SURV.items():
    val = eval(expr, {"df": dt, "pd": pd, "np": np})
    v = pd.to_numeric(pd.Series(np.asarray(val, dtype="float64"), index=dt.index), errors="coerce").replace([np.inf,-np.inf], np.nan)
    fv[nm] = v.values.astype(float)

def auc(cols, fitm, evalm, seed=0):
    Xa = np.nan_to_num(np.column_stack(cols), nan=0.0, posinf=0.0, neginf=0.0)
    dtr = lgb.Dataset(Xa[fitm], label=y[fitm])
    p = dict(objective="binary", metric="auc", num_leaves=15, learning_rate=0.05, min_child_samples=200, verbosity=-1, seed=seed, n_jobs=4)
    bst = lgb.train(p, dtr, num_boost_round=120)
    return roc_auc_score(y[evalm], bst.predict(Xa[evalm]))

chrono_fit = valid&(yr==2024); chrono_eval = valid&(yr==2025)
def rsplit(seed):
    uniq=pd.unique(rids[valid]); rng=np.random.RandomState(seed); perm=rng.permutation(len(uniq)); h=len(uniq)//2
    fs=set(uniq[perm[:h]]); isf=np.array([r in fs for r in rids])&valid; return isf, valid&~isf

splits = {"chrono(24>25)":(chrono_fit,chrono_eval)}
for s in (42,7,123): splits[f"rand{s}"]=rsplit(s)

raw_cols = [raw_arr[c] for c in RAW]

print("=== DECISIVE: delta vs LGBM[score] (orig screen) AND vs LGBM[score + raw v6 inputs] ===\n")
print(f"{'feature':<32} | screen-style Δ(vs score) | HONEST Δ(vs score+raw_inputs)")
print("-"*95)
res=[]
for nm in SURV:
    v=fv[nm]
    screen_d=[]; honest_d=[]
    for label,(fm,em) in splits.items():
        b_score = auc([sc], fm, em)
        b_full  = auc([sc]+raw_cols, fm, em)
        screen_d.append(auc([sc,v], fm, em)-b_score)
        honest_d.append(auc([sc]+raw_cols+[v], fm, em)-b_full)
    rec=dict(name=nm, screen_mean=round(float(np.mean(screen_d)),5),
             honest_mean=round(float(np.mean(honest_d)),5),
             honest_min=round(float(np.min(honest_d)),5),
             honest_max=round(float(np.max(honest_d)),5))
    res.append(rec)
    print(f"{nm:<32} | {np.mean(screen_d):+.5f}               | {np.mean(honest_d):+.5f}  [{np.min(honest_d):+.5f},{np.max(honest_d):+.5f}]")

# baseline lift: does score+raw beat score alone? (shows v6 already underexploits its own inputs?)
print("\n=== sanity: AUC[score] vs AUC[score+raw_inputs] (no engineered feat) ===")
for label,(fm,em) in splits.items():
    print(f"  {label:<14} score={auc([sc],fm,em):.5f}  score+raw={auc([sc]+raw_cols,fm,em):.5f}")

Path(BASE/"data/_adv_decisive.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print("\n[done]")
