# -*- coding: utf-8 -*-
"""
adv_reverify_survivors.py — adversarial re-verification of the 8 'survivor' feats.
Tests: leak-safety (handled statically), delta_auc under (a) original random 50/50,
(b) chronological 2024-fit/2025-eval, (c) per-era random splits, (d) multi-seed,
plus collinearity vs the actual v6 feature_cols.
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

SURVIVORS = {
 "kako5_form_favored_rank":"-(df.groupby('レースID(新/馬番無)')['kako5_avg_pos'].rank(method='average') - 1) / (df['出走頭数'].clip(lower=2) - 1)",
 "kako5_fieldz_shrunk_by_count":"(-(df['kako5_avg_pos'] - df.groupby('レースID(新/馬番無)')['kako5_avg_pos'].transform('mean')) / df.groupby('レースID(新/馬番無)')['kako5_avg_pos'].transform('std')) * (df['kako5_race_count'] / (df['kako5_race_count'] + 1))",
 "kako5_pos_z_in_race":"(df['kako5_avg_pos'] - df.groupby('レースID(新/馬番無)')['kako5_avg_pos'].transform('mean')) / df.groupby('レースID(新/馬番無)')['kako5_avg_pos'].transform('std')",
 "kako5_avgpos_field_z":"-(df['kako5_avg_pos'] - df.groupby('レースID(新/馬番無)')['kako5_avg_pos'].transform('mean')) / df.groupby('レースID(新/馬番無)')['kako5_avg_pos'].transform('std')",
 "kako5_seasoned_gate_x_fieldz":"(df['kako5_race_count'] >= 4).astype(float) * (-(df['kako5_avg_pos'] - df.groupby('レースID(新/馬番無)')['kako5_avg_pos'].transform('mean')) / df.groupby('レースID(新/馬番無)')['kako5_avg_pos'].transform('std'))",
 "kako5_bestpos_field_frac":"(df['kako5_best_pos'] - 1) / (df['出走頭数'].clip(lower=2) - 1)",
 "kako5_avg_minus_best_gap":"df['kako5_avg_pos'] - df['kako5_best_pos']",
 "gate_crowding":"df['出走頭数'] / df['フルゲート頭数']",
}

print("[load] model / master ...")
b = joblib.load(BASE / "models/unified_rank_v6.pkl")
model, feats, encs = b["model"], b["feature_cols"], b["encoders"]

df = pd.read_csv(BASE / "data/master_v2_20130105-20251228.csv", encoding="utf-8-sig", low_memory=False)
df = df.dropna(subset=["split"])
df_test = df[df["split"] == "test"].copy().reset_index(drop=True)
print(f"  test rows={len(df_test)}")

# v6 score
df_enc = df_test.copy()
for c, le in encs.items():
    if c in df_enc.columns:
        v = df_enc[c].astype(str).fillna("__NaN__")
        known = set(le.classes_)
        df_enc[c] = le.transform(v.where(v.isin(known), "__NaN__"))
X = df_enc[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
df_test["v6_score"] = model.predict(X)
sc = df_test["v6_score"].values.astype(float)

y = pd.to_numeric(df_test["fukusho_flag"], errors="coerce")
yf = y.values.astype(float)
yr = pd.to_datetime(df_test["date_dt"], errors="coerce").dt.year.values
rids = df_test[COL_RID].values

base_auc_full = roc_auc_score(yf[np.isfinite(yf)], sc[np.isfinite(yf)])
print(f"  baseline v6 AUC (full test) = {base_auc_full:.5f}")

# build feats
fv = {}
for nm, expr in SURVIVORS.items():
    val = eval(expr, {"df": df_test, "pd": pd, "np": np})
    v = pd.to_numeric(pd.Series(np.asarray(val, dtype="float64"), index=df_test.index), errors="coerce")
    v = v.replace([np.inf, -np.inf], np.nan)
    fv[nm] = v.values.astype(float)
    print(f"  built {nm}: cov={np.isfinite(fv[nm]).mean():.3f} uni_auc={roc_auc_score(yf[np.isfinite(yf)&np.isfinite(fv[nm])], fv[nm][np.isfinite(yf)&np.isfinite(fv[nm])]):.4f}")

def lgbm_auc(cols, fit_mask, eval_mask, seed=0):
    Xall = np.column_stack(cols)
    Xall = np.nan_to_num(Xall, nan=0.0, posinf=0.0, neginf=0.0)
    dtr = lgb.Dataset(Xall[fit_mask], label=yf[fit_mask])
    params = dict(objective="binary", metric="auc", num_leaves=15, learning_rate=0.05,
                  min_child_samples=200, verbosity=-1, seed=seed, n_jobs=4)
    bst = lgb.train(params, dtr, num_boost_round=120)
    return roc_auc_score(yf[eval_mask], bst.predict(Xall[eval_mask]))

valid = np.isfinite(sc) & np.isfinite(yf)

def random_split(seed, restrict=None):
    sel = valid.copy()
    if restrict is not None:
        sel &= (yr == restrict)
    uniq = pd.unique(rids[sel])
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(uniq)); half = len(uniq)//2
    fitset = set(uniq[perm[:half]])
    isfit = np.array([r in fitset for r in rids]) & sel
    return isfit, sel & ~isfit

# chronological
chrono_fit = valid & (yr == 2024)
chrono_eval = valid & (yr == 2025)

print("\n=== delta_auc under multiple regimes (vs LGBM[score]) ===")
out = []
# precompute baselines
b_chrono = lgbm_auc([sc], chrono_fit, chrono_eval)
splits_rand = [random_split(s) for s in (42, 1, 7, 123, 2026)]
b_rand = [lgbm_auc([sc], f, e, seed=s) for (f,e),s in zip(splits_rand,(0,1,2,3,4))]
b_2024 = []; b_2025 = []
sp24 = [random_split(s, 2024) for s in (42,7)]
sp25 = [random_split(s, 2025) for s in (42,7)]
for (f,e) in sp24: b_2024.append(lgbm_auc([sc], f, e))
for (f,e) in sp25: b_2025.append(lgbm_auc([sc], f, e))

print(f"  base AUC: chrono(24->25)={b_chrono:.5f}  rand(mean)={np.mean(b_rand):.5f}  2024rand(mean)={np.mean(b_2024):.5f}  2025rand(mean)={np.mean(b_2025):.5f}")

for nm in SURVIVORS:
    v = fv[nm]
    d_chrono = lgbm_auc([sc, v], chrono_fit, chrono_eval) - b_chrono
    d_rand = [lgbm_auc([sc, v], f, e, seed=s)-bb for (f,e),s,bb in zip(splits_rand,(0,1,2,3,4),b_rand)]
    d24 = [lgbm_auc([sc, v], f, e)-bb for (f,e),bb in zip(sp24,b_2024)]
    d25 = [lgbm_auc([sc, v], f, e)-bb for (f,e),bb in zip(sp25,b_2025)]
    rec = dict(name=nm, d_chrono=round(d_chrono,5),
               d_rand_mean=round(float(np.mean(d_rand)),5), d_rand_min=round(float(np.min(d_rand)),5),
               d_rand_max=round(float(np.max(d_rand)),5),
               d_2024=round(float(np.mean(d24)),5), d_2025=round(float(np.mean(d25)),5))
    out.append(rec)
    print(f"  {nm:<32} chrono={d_chrono:+.5f}  rand[{np.min(d_rand):+.5f},{np.max(d_rand):+.5f}]mean={np.mean(d_rand):+.5f}  2024={np.mean(d24):+.5f} 2025={np.mean(d25):+.5f}")

# collinearity vs v6 feature_cols (numeric) AND vs v6_score
print("\n=== collinearity (max |corr| vs v6 feature_cols + v6_score) ===")
Xnum = df_enc[feats].apply(pd.to_numeric, errors="coerce")
coll = []
for nm in SURVIVORS:
    v = pd.Series(fv[nm])
    best = ("v6_score", abs(np.corrcoef(np.nan_to_num(fv[nm]), sc)[0,1]))
    for c in feats:
        col = Xnum[c]
        m = v.notna() & col.notna()
        if m.sum() < 1000 or col[m].nunique() < 2: continue
        r = abs(np.corrcoef(v[m], col[m])[0,1])
        if r > best[1]: best = (c, r)
    coll.append(dict(name=nm, max_corr_feat=best[0], max_corr=round(float(best[1]),4)))
    print(f"  {nm:<32} max|corr|={best[1]:.4f} with {best[0]}")

# cross-survivor collinearity (they're mostly the same kako5_avg_pos transform)
print("\n=== cross-survivor |corr| matrix (key pairs) ===")
names = list(SURVIVORS)
for i in range(len(names)):
    for j in range(i+1, len(names)):
        a, bb = fv[names[i]], fv[names[j]]
        m = np.isfinite(a)&np.isfinite(bb)
        if m.sum()<1000: continue
        r = abs(np.corrcoef(a[m], bb[m])[0,1])
        if r > 0.8:
            print(f"  {names[i]:<30} ~ {names[j]:<30} |r|={r:.3f}")

Path(BASE/"data/_adv_reverify.json").write_text(json.dumps(dict(base_auc_full=base_auc_full, deltas=out, collinearity=coll), ensure_ascii=False, indent=1), encoding="utf-8")
print("\n[done] wrote data/_adv_reverify.json")
