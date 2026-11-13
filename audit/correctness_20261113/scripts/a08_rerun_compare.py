"""A08: 20261108 の再実行 bundle (A, B) と公開 bundle を比較 (決定性・再現性)。
usage: python a08_rerun_compare.py <published_bundle> <rerunA_bundle> <rerunB_bundle>
"""
import json, sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "out"
pub, ra, rb = (json.load(open(p, encoding="utf-8")) for p in sys.argv[1:4])
FIELDS = ["mark", "ai_rank", "ai_score", "p_win", "p_plc", "p_sho"]


def idx(b):
    return {(str(r["race_id"]), h["umaban"]): h for r in b["races"] for h in r["horses"]}


def cmp(x, y):
    X, Y = idx(x), idx(y)
    keys = set(X) & set(Y)
    out = {"n_common": len(keys), "only_left": len(set(X) - set(Y)), "only_right": len(set(Y) - set(X))}
    for f in FIELDS:
        diff = [k for k in keys if X[k].get(f) != Y[k].get(f)]
        out[f"diff_{f}"] = len(diff)
        if f in ("ai_score", "p_win", "p_sho") and diff:
            out[f"maxabs_{f}"] = round(max(abs((X[k][f] or 0) - (Y[k][f] or 0)) for k in diff), 4)
    hx = {str(r["race_id"]): next((h["umaban"] for h in r["horses"] if h["mark"] == "◎"), None) for r in x["races"]}
    hy = {str(r["race_id"]): next((h["umaban"] for h in r["horses"] if h["mark"] == "◎"), None) for r in y["races"]}
    out["races_hon_changed"] = sum(hx[k] != hy[k] for k in set(hx) & set(hy))
    out["serve_path_left"] = {k: x["serve_path"].get(k) for k in ("version",)} | {"hist_max": x["serve_path"]["history"].get("max_date"), "hist_sha": x["serve_path"]["history"].get("sha256", "")[:12], "commit": x["serve_path"]["git"].get("commit", "")[:10]}
    out["serve_path_right"] = {k: y["serve_path"].get(k) for k in ("version",)} | {"hist_max": y["serve_path"]["history"].get("max_date"), "hist_sha": y["serve_path"]["history"].get("sha256", "")[:12], "commit": y["serve_path"]["git"].get("commit", "")[:10]}
    return out


res = {"rerunA_vs_rerunB": cmp(ra, rb), "published_vs_rerunA": cmp(pub, ra)}
json.dump(res, open(OUT / "a08_rerun_compare.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
