"""A09: 配信の学習定義合わせ (2026-11-13 修正) の前後で、同じ入力から作った bundle を比べる。
old = HEAD (14a419a6) のコード、new = 修正後。モデル・較正器・履歴・入力は同一。
指標: ◎の変化レース数、◎勝率・3着内 (取消◎は除外)、勝ち馬の log p_win、3着内の Brier。開催日クラスタ CI。
usage: python a09_replay_old_new.py <replay_dir> <ROOT>
"""
import csv, io, json, sys
from pathlib import Path
import numpy as np

S, ROOT = Path(sys.argv[1]), Path(sys.argv[2])
OUT = Path(__file__).resolve().parents[1] / "out"
rng = np.random.default_rng(3)


def kekka(d):
    p = ROOT / "data" / "kekka" / f"{d}.csv"
    rd = csv.reader(io.StringIO(p.read_bytes().decode("cp932")))
    next(rd)
    out = {}
    for row in rd:
        try:
            rid, u, pos = row[7][:16], int(row[4]), int(row[6])
        except (ValueError, IndexError):
            continue
        r = out.setdefault(rid, {"pos": {}, "scr": set()})
        r["pos"][u] = pos
        if pos == 0 and not row[8].strip():
            r["scr"].add(u)
    return out


def races(b):
    return {str(r["race_id"])[:16]: {h["umaban"]: h for h in r["horses"]} for r in b["races"]}


rows, n_changed, n_common, dates_used = [], 0, 0, []
for ob in sorted((S / "out_old").glob("*_bundle.json")):
    d = ob.name[:8]
    nb = S / "out_new" / ob.name
    if not nb.exists():
        continue
    o, n = races(json.load(open(ob, encoding="utf-8"))), races(json.load(open(nb, encoding="utf-8")))
    res = kekka(d)
    dates_used.append(d)
    for rid in set(o) & set(n):
        n_common += 1
        ho = next((u for u, h in o[rid].items() if h.get("mark") == "◎"), None)
        hn = next((u for u, h in n[rid].items() if h.get("mark") == "◎"), None)
        n_changed += int(ho != hn)
        r = res.get(rid)
        if not r or ho is None or hn is None or ho in r["scr"] or hn in r["scr"]:
            continue
        def stats(hs, h):
            pos = r["pos"].get(h, 0)
            win = [u for u, p in r["pos"].items() if p == 1]
            pw = sum((hs.get(u) or {}).get("p_win") or 0 for u in win)
            br = np.mean([((hs[u].get("p_sho") or 0) - float(0 < r["pos"].get(u, 0) <= 3)) ** 2
                          for u in hs if u not in r["scr"] and u in r["pos"]])
            return float(pos == 1), float(0 < pos <= 3), -np.log(max(pw, 1e-4)), br
        rows.append((d, *stats(o[rid], ho), *stats(n[rid], hn)))

A = np.array([r[1:] for r in rows]); D = np.array([r[0] for r in rows]); ud = np.unique(D)
idx = {x: np.where(D == x)[0] for x in ud}


def ci(i_new, i_old, B=2000):
    v = []
    for _ in range(B):
        s = np.concatenate([idx[x] for x in rng.choice(ud, len(ud))])
        v.append(A[s, i_new].mean() - A[s, i_old].mean())
    return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]


names = ["hon_win", "hon_top3", "nll_winner", "brier_top3"]
res = {"dates": dates_used, "n_races_common": n_common, "n_races_hon_changed": n_changed,
       "n_races_scored": int(len(A)),
       "old": {k: round(float(A[:, i].mean()), 4) for i, k in enumerate(names)},
       "new": {k: round(float(A[:, i + 4].mean()), 4) for i, k in enumerate(names)},
       "new_minus_old": {k: {"diff": round(float(A[:, i + 4].mean() - A[:, i].mean()), 4), "ci95": ci(i + 4, i)}
                         for i, k in enumerate(names)}}
json.dump(res, open(OUT / "a09_replay_old_new.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
