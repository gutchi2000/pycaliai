# -*- coding: utf-8 -*-
"""
feat_exam_300.py — 300特徴 leak-safe 検定官
============================================
新baseline = [v6_score, kako5_avgpos_field_z]
各特徴 expr を頑健構築 -> part_corr(feat,fukusho | baseline) + univariate AUC
|part_corr|上位20で 増分AUC = AUC(LGBM[base,feat]) - AUC(LGBM[base])
survivors = part_corr>0.01 & delta_auc>0
"""
import sys, io, json, warnings, re
from pathlib import Path
import numpy as np, pandas as pd, joblib
warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
BASE = Path(r"E:\PyCaLiAI")
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

RID = "レースID(新/馬番無)"
TGT = "fukusho_flag"

# ----- leak ガード: これらの語を含む列を参照する特徴は除外 -----
LEAK_TOKENS = ["fukusho_flag", "roi_target", "確定着順", "着順", "_JYUN", "着 順"]
# 注: 前走確定着順 / 前走着差 / kako5系 は過去走由来でleak-safe(着順を含むが「前走/過去」)。
#     ただし今走の「確定着順」「fukusho_flag」「roi_target」は禁止。
LEAK_FORBIDDEN_EXACT = ["fukusho_flag", "roi_target"]

def expr_is_leaky(expr: str) -> str:
    """今走の結果列を参照していたら理由を返す。前走/過去走由来はOK。"""
    for tok in LEAK_FORBIDDEN_EXACT:
        if tok in expr:
            return f"refs {tok}"
    # 今走の確定着順(前走でない)を参照していないか
    # '前走確定着順' はOK。素の '確定着順' (前置詞なし) は今走 -> leak
    for m in re.finditer(r"(.{0,3})確定着順", expr):
        prefix = m.group(1)
        if "前走" not in prefix and "kako" not in prefix:
            return "refs 今走確定着順"
    return ""

print("[load] v6 model + master ...")
b = joblib.load(BASE / "models/unified_rank_v6.pkl")
model, feats, encs = b["model"], b["feature_cols"], b["encoders"]

df = pd.read_csv(BASE / "data/master_v2_20130105-20251228.csv", encoding="utf-8-sig", low_memory=False)
df = df.dropna(subset=[RID, "split", TGT]).copy()
df[TGT] = pd.to_numeric(df[TGT], errors="coerce")
df = df.dropna(subset=[TGT]).copy()
df["year"] = df[RID].astype(str).str[:4]
print(f"  rows={len(df)}  splits={df['split'].value_counts().to_dict()}")

# ---- v6 score (categorical を encode した別行列で predict; 元dfは生のまま保持) ----
Xenc = df.copy()
for c, le in encs.items():
    if c in Xenc.columns:
        v = Xenc[c].astype(str).fillna("__NaN__")
        Xenc[c] = le.transform(v.where(v.isin(set(le.classes_)), "__NaN__"))
Xmat = Xenc[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
df["v6_score"] = model.predict(Xmat)
del Xenc, Xmat
print("  v6_score computed")

# ---- 既知survivor: kako5_avgpos_field_z ----
g = df.groupby(RID)["kako5_avg_pos"]
df["kako5z"] = ((df["kako5_avg_pos"] - g.transform("mean")) / (g.transform("std") + 1e-9))
df["kako5z"] = df["kako5z"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

# ---- train/test mask ----
m_tr = (df["split"] == "train").values
m_te = (df["split"] == "test").values
print(f"  train={m_tr.sum()}  test={m_te.sum()}")

y_te = df.loc[m_te, TGT].values.astype(int)
base_cols = ["v6_score", "kako5z"]

# ---- baseline test AUC (univariate v6_score + kako5z は LGBM 2列) ----
def lgbm_auc(train_X, train_y, test_X, test_y, seed=42):
    dtr = lgb.Dataset(train_X, label=train_y)
    params = dict(objective="binary", metric="auc", num_leaves=31, learning_rate=0.05,
                  feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1,
                  min_data_in_leaf=200, verbose=-1, seed=seed)
    bst = lgb.train(params, dtr, num_boost_round=200)
    return roc_auc_score(test_y, bst.predict(test_X))

Xb_tr = df.loc[m_tr, base_cols].values
Xb_te = df.loc[m_te, base_cols].values
yb_tr = df.loc[m_tr, TGT].values.astype(int)
baseline_auc = lgbm_auc(Xb_tr, yb_tr, Xb_te, y_te)
print(f"\n*** baseline test fukusho AUC [v6_score, kako5z] = {baseline_auc:.4f}\n")

# residualize helper (on test set): resid of a vs [v6_score,kako5z] by OLS
Bte = np.column_stack([np.ones(m_te.sum()), df.loc[m_te, "v6_score"].values, df.loc[m_te, "kako5z"].values])
# precompute pseudo-inverse projector
BtB_inv = np.linalg.pinv(Bte.T @ Bte)
def residualize_te(vec):
    coef = BtB_inv @ (Bte.T @ vec)
    return vec - Bte @ coef

# residual of fukusho target (constant across features)
y_resid = residualize_te(df.loc[m_te, TGT].values.astype(float))
y_resid_std = y_resid.std()

# ---- feature spec JSON ----
SPEC = json.loads(Path(BASE / "analysis/feat_spec_300.json").read_text(encoding="utf-8"))
print(f"[spec] {len(SPEC)} features loaded")

def robust(s):
    s = pd.to_numeric(s, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)
    return s

def rz(series, rid):
    """レース内z. 999/0等のsentinelは呼び出し側でmask済み前提。"""
    gg = series.groupby(rid)
    return (series - gg.transform("mean")) / (gg.transform("std").replace(0, np.nan))

# 評価環境に渡す名前空間
def build_feature(expr, df):
    """expr を複数戦略で構築。戻り: pd.Series(index=df.index) or raise."""
    rid = df[RID]
    ns = {"df": df, "np": np, "pd": pd, "rid": rid,
          "rz": lambda x: rz(x, rid)}
    # 戦略1: 'feat=' を含む -> exec して feat を取り出す
    if "feat=" in expr or "feat =" in expr:
        # コメント (# 以降) は安全のため除去しない (文字列内 # 稀) が、行末コメントは exec で問題になる場合あり
        loc = dict(ns)
        exec(expr, {"np": np, "pd": pd}, loc)
        feat = loc["feat"]
        return pd.Series(np.asarray(feat, dtype="float64"), index=df.index)
    # 戦略2: 純粋な eval 可能式
    try:
        val = eval(expr, {"np": np, "pd": pd}, ns)
        return pd.Series(np.asarray(val, dtype="float64"), index=df.index)
    except Exception:
        raise

results = []
built = 0
failed = 0
fail_log = []

for spec in SPEC:
    name = spec["name"]
    expr = spec["expr"]
    # leak check
    leak = expr_is_leaky(expr)
    if leak:
        failed += 1
        fail_log.append((name, "LEAK:" + leak))
        results.append({"name": name, "part_corr": 0.0, "uni_auc": None,
                        "built": False, "_err": "leak:" + leak})
        continue
    # try build
    try:
        feat = build_feature(expr, df)
    except Exception as e:
        failed += 1
        fail_log.append((name, "BUILD:" + repr(e)[:120]))
        results.append({"name": name, "part_corr": 0.0, "uni_auc": None,
                        "built": False, "_err": "build:" + repr(e)[:80]})
        continue
    feat = robust(feat).fillna(0.0)
    if feat.nunique() <= 1:
        failed += 1
        fail_log.append((name, "CONST"))
        results.append({"name": name, "part_corr": 0.0, "uni_auc": None,
                        "built": False, "_err": "constant"})
        continue
    built += 1
    fv_te = feat.values[m_te]
    # univariate AUC
    try:
        ua = roc_auc_score(y_te, fv_te)
        ua = max(ua, 1 - ua)  # 単調方向不問
    except Exception:
        ua = None
    # part_corr
    fr = residualize_te(fv_te.astype(float))
    denom = fr.std() * y_resid_std
    pc = float((fr * y_resid).mean() / denom) if denom > 1e-12 else 0.0
    results.append({"name": name, "part_corr": pc, "uni_auc": (round(ua, 4) if ua is not None else None),
                    "built": True, "_fv_te": fv_te, "_fv_tr": feat.values[m_tr]})

print(f"\n[build] built={built}  failed={failed}")
for nm, why in fail_log[:60]:
    print(f"   FAIL {nm:<40} {why}")

# ---- |part_corr| 降順 ----
results_sorted = sorted(results, key=lambda r: -abs(r["part_corr"]))

# ---- 上位20 で 増分AUC ----
TOPN = 20
top = [r for r in results_sorted if r["built"]][:TOPN]
print(f"\n[delta AUC] computing for top {len(top)} by |part_corr| ...")
for r in top:
    Xf_tr = np.column_stack([Xb_tr, r["_fv_tr"]])
    Xf_te = np.column_stack([Xb_te, r["_fv_te"]])
    try:
        auc_f = lgbm_auc(Xf_tr, yb_tr, Xf_te, y_te)
        r["delta_auc"] = round(auc_f - baseline_auc, 5)
    except Exception as e:
        r["delta_auc"] = None
    print(f"   {r['name']:<40} pc={r['part_corr']:+.4f} uni={r['uni_auc']} dAUC={r['delta_auc']}")

# ---- survivors ----
survivors = [r["name"] for r in top
             if r.get("delta_auc") is not None and abs(r["part_corr"]) > 0.01 and r["delta_auc"] > 0]

# ---- ranked output (全特徴, |part_corr|降順) ----
ranked = []
for r in results_sorted:
    entry = {"name": r["name"], "part_corr": round(r["part_corr"], 5), "built": r["built"]}
    if r.get("uni_auc") is not None:
        entry["uni_auc"] = r["uni_auc"]
    if "delta_auc" in r and r["delta_auc"] is not None:
        entry["delta_auc"] = r["delta_auc"]
    ranked.append(entry)

out = {"built": built, "failed": failed, "baseline_auc": round(baseline_auc, 5),
       "survivors": survivors, "ranked": ranked}
Path(BASE / "analysis/feat_exam_300_result.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\n[survivors] {len(survivors)}: {survivors}")
print("[saved] analysis/feat_exam_300_result.json")
