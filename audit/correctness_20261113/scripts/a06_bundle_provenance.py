"""A06: 各 2026 bundle の現行ファイルを、git 上の「発走前 (当日 09:30 より前) 最後のコミット版」と比較。
mtime は checkout でも変わるため、内容の一致で事後書換えを判定する。
指標: 内容一致 / ◎が変わったレース数 / p_win の最大差。read-only (git show のみ)。
usage: python a06_bundle_provenance.py <ROOT>
"""
import glob, json, subprocess, sys
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("E:/PyCaLiAI")
OUT = Path(__file__).resolve().parents[1] / "out"


def git(*a):
    return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True).stdout


def hon_map(b):
    m = {}
    for r in b.get("races", []):
        h = [x for x in r["horses"] if x.get("mark") == "◎"]
        m[str(r["race_id"])[:16]] = (h[0]["umaban"] if h else None,
                                     {x["umaban"]: x.get("p_win") for x in r["horses"]})
    return m


rows = []
for bp in sorted(glob.glob(str(ROOT / "reports/cowork_input/2026*_bundle.json"))):
    d = Path(bp).name[:8]
    rel = f"reports/cowork_input/{d}_bundle.json"
    until = f"{d[:4]}-{d[4:6]}-{d[6:]} 09:30"
    pre = git("log", "-1", "--format=%H", f"--until={until}", "--", rel).decode().strip()
    post = git("log", "--format=%H %ci %s", f"--since={until}", "--", rel).decode().strip().splitlines()
    cur_raw = Path(bp).read_bytes()
    row = {"date": d, "pre_race_commit": pre[:10] or None, "n_commits_after_race_start": len(post),
           "commits_after": [p[:80] for p in post][:4]}
    if pre:
        old_raw = git("show", f"{pre}:{rel}")
        row["identical_to_pre_race_commit"] = old_raw == cur_raw
        if old_raw != cur_raw:
            try:
                old, cur = hon_map(json.loads(old_raw)), hon_map(json.loads(cur_raw))
                common = set(old) & set(cur)
                row["races_pre"], row["races_now"] = len(old), len(cur)
                row["races_hon_changed"] = sum(old[k][0] != cur[k][0] for k in common)
                diffs = [abs((cur[k][1].get(u) or 0) - (v or 0)) for k in common for u, v in old[k][1].items()]
                row["max_abs_p_win_diff"] = round(max(diffs), 4) if diffs else None
                row["races_added_after"] = len(set(cur) - set(old))
            except Exception as e:
                row["error"] = str(e)
    rows.append(row)
json.dump(rows, open(OUT / "a06_bundle_provenance.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
n_nopre = sum(r["pre_race_commit"] is None for r in rows)
n_changed = sum(r.get("identical_to_pre_race_commit") is False for r in rows)
print(f"dates={len(rows)} no_pre_race_commit={n_nopre} changed_after_race_start={n_changed}")
for r in rows:
    if r["pre_race_commit"] is None or r.get("identical_to_pre_race_commit") is False:
        print(json.dumps(r, ensure_ascii=False))
