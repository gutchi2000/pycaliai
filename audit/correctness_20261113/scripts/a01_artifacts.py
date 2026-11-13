"""A01: 本番成果物の識別情報と中身 (read-only)。

python -m audit.correctness_20261113.scripts.a01_artifacts  (cwd = E:/PyCaLiAI)
出力: audit/correctness_20261113/out/a01_artifacts.json
"""
import hashlib, json, sys
from pathlib import Path
import joblib
import numpy as np

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("E:/PyCaLiAI")
OUT = Path(__file__).resolve().parents[1] / "out"
OUT.mkdir(exist_ok=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


res = {"root": str(ROOT)}
files = ["models/unified_rank_v6.pkl", "models/pl_calibrators_v6.pkl",
         "models/pl_calibrators_v6_serve.pkl", "models/prev_hosei_proxy.pkl",
         "data/pl_payout_curve_v6.pkl", "data/serve_feature_baseline.json",
         "data/_horse_history.parquet", "data/class_prior_v6.json",
         "data/master_v2_20130105-20251228.csv", "data/serve_code_maps.json",
         "data/chaos_quantiles.json"]
pins = json.load(open(ROOT / "data/artifact_pins.json", encoding="utf-8"))["pins"]
res["files"] = {}
for f in files:
    p = ROOT / f
    if not p.exists():
        res["files"][f] = None
        continue
    s = sha(p)
    res["files"][f] = {"sha256": s, "size": p.stat().st_size,
                       "mtime": p.stat().st_mtime,
                       "pinned": pins.get(f), "pin_match": (pins.get(f) == s) if f in pins else None}

b = joblib.load(ROOT / "models/unified_rank_v6.pkl")
m = b["model"]
feats = b["feature_cols"]
res["model"] = {k: (v if isinstance(v, (str, int, float, dict, list, type(None))) else str(type(v)))
                for k, v in b.items() if k not in ("model", "encoders", "feature_cols")}
res["model"]["n_feats"] = len(feats)
res["model"]["feature_cols"] = feats
res["model"]["cat_cols_encoded"] = sorted(b["encoders"].keys())
res["model"]["best_iteration"] = m.best_iteration
res["model"]["num_trees"] = m.num_trees()
gain = m.feature_importance(importance_type="gain")
order = np.argsort(-gain)
res["model"]["gain_top40"] = [[feats[i], round(float(gain[i] / gain.sum() * 100), 3)] for i in order[:40]]
res["model"]["params"] = {k: v for k, v in m.params.items()} if hasattr(m, "params") else None

for name in ["pl_calibrators_v6.pkl", "pl_calibrators_v6_serve.pkl"]:
    c = joblib.load(ROOT / "models" / name)
    meta = {k: v for k, v in c.items() if k != "calibrators"}
    cals = c["calibrators"]
    meta["keys"] = sorted(cals.keys())
    # isotonic step 数と x 範囲
    meta["iso"] = {}
    for k, iso in cals.items():
        xs = getattr(iso, "X_thresholds_", None)
        ys = getattr(iso, "y_thresholds_", None)
        meta["iso"][k] = {"n_thresholds": None if xs is None else int(len(xs)),
                          "x_min": None if xs is None else float(xs.min()),
                          "x_max": None if xs is None else float(xs.max()),
                          "y_max": None if ys is None else float(ys.max()),
                          "n_distinct_y": None if ys is None else int(len(np.unique(ys)))}
    res[name] = json.loads(json.dumps(meta, default=str))

# serve calibrator のマスク vs 現行 baseline の <40%
bl = json.load(open(ROOT / "data/serve_feature_baseline.json", encoding="utf-8"))
bc = bl.get("baseline_cov", {})
cur_dead = sorted([f for f in feats if bc.get(f, 0.0) < 0.40])
c = joblib.load(ROOT / "models/pl_calibrators_v6_serve.pkl")
fit_dead = sorted(set(c.get("serve_mask_numeric", [])) | set(c.get("serve_mask_cat", [])))
res["serve_mask_sync"] = {
    "baseline_weeks": bl.get("weeks"), "baseline_generated": bl.get("generated_at"),
    "calibrator_fit_baseline_weeks": c.get("serve_baseline_weeks"),
    "n_dead_now": len(cur_dead), "n_dead_at_fit": len(fit_dead),
    "dead_now_not_masked_at_fit": sorted(set(cur_dead) - set(fit_dead)),
    "masked_at_fit_but_alive_now": sorted(set(fit_dead) - set(cur_dead)),
}
json.dump(res, open(OUT / "a01_artifacts.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps({k: res[k] for k in ["files", "serve_mask_sync"]}, ensure_ascii=False, indent=1, default=str))
print("n_feats", len(feats))
print(feats)
print("gain_top40", res["model"]["gain_top40"])
print({k: v for k, v in res["model"].items() if k not in ("feature_cols", "gain_top40", "cat_cols_encoded", "params")})
