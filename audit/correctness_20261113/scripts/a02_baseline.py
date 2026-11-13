"""A02: serve baseline の coverage / train_cov / gain を主要特徴について表示。"""
import json, sys
from pathlib import Path
ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("E:/PyCaLiAI")
b = json.load(open(ROOT / "data/serve_feature_baseline.json", encoding="utf-8"))
print(list(b.keys()))
print("weeks", b.get("weeks"), "dead_gain_pct", b.get("dead_gain_pct"))
rows = []
for k, v in b["baseline_cov"].items():
    rows.append((k, v, b.get("train_cov", {}).get(k), b.get("gain_pct", {}).get(k)))
rows.sort(key=lambda r: -(r[3] or 0))
print("feature, serve_cov, train_cov, gain%")
for r in rows:
    if (r[1] is not None and r[2] is not None and r[1] < 0.7 * r[2]) or r[1] < 0.4:
        print("DEGRADED", r)
for r in rows[:30]:
    print(r)
