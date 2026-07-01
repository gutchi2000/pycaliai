"""
settle_k_compare.py
===================
馬単/三連複の点数(k)を変えて 2026 ライブ(11週末)を同条件で精算・比較。
同じレース・同じ gap ゲート・同じ prob-first 順位、違いは k のカットのみ。
コストは実在 combo 数で頭打ち(小頭数で過大計上しない=どの案も公平)。

実行: python settle_k_compare.py
"""
from __future__ import annotations
import io, json, sys
from math import comb
from pathlib import Path
import numpy as np
import pl_probs as PL
from backtest_pl_ev import all_umatan_mat, all_sanrenpuku_tensor
from oreapro_bets import GAP_GATE
from settle_oreapro import load_kekka

try:
    if not isinstance(sys.stdout, io.TextIOWrapper) or (sys.stdout.encoding or "").lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
except Exception:
    pass

BASE = Path(__file__).parent
DATES = ["20260426", "20260516", "20260517", "20260523", "20260524",
         "20260530", "20260531", "20260606", "20260607", "20260613", "20260614"]
CONFIGS = [(3, 4), (6, 10), (10, 30), (17, 45)]   # (馬単点, 三連複点)


def race_ranks(horses):
    """ai_score から gap, 馬番→local index, 馬単行列, 三連複テンソルを返す."""
    hs = [h for h in horses if isinstance(h.get("ai_score"), (int, float)) and h["ai_score"] == h["ai_score"]]
    if len(hs) < 3:
        return None
    uma = [int(h["umaban"]) for h in hs]
    sc = np.array([float(h["ai_score"]) for h in hs])
    w = PL.pl_weights(sc); p = w / w.sum(); ps = np.sort(p)[::-1]
    return {"gap": float(ps[0] - ps[1]), "uma2i": {u: i for i, u in enumerate(uma)},
            "w": w, "n": len(w)}


def winning_ranks(rr, res):
    """winning 馬単pair / 三連複trio の prob 順位(0-index)を返す."""
    u2i = rr["uma2i"]
    try:
        lwin, lplc, lsho = u2i[res["win"]], u2i[res["plc"]], u2i[res["sho"]]
    except KeyError:
        return None
    w = rr["w"]; n = rr["n"]
    U = all_umatan_mat(w)
    um_rank = int((U[~np.eye(n, dtype=bool)] > U[lwin, lplc]).sum())
    P3 = all_sanrenpuku_tensor(w)
    from itertools import combinations, permutations
    tris = np.array(list(combinations(range(n), 3)))
    ptri = np.zeros(len(tris))
    for a, b, c in permutations(range(3)):
        ptri += P3[tris[:, a], tris[:, b], tris[:, c]]
    wt = tuple(sorted((lwin, lplc, lsho)))
    wm = (tris[:, 0] == wt[0]) & (tris[:, 1] == wt[1]) & (tris[:, 2] == wt[2])
    if not wm.any():
        return None
    sp_rank = int((ptri > ptri[wm][0]).sum())
    return um_rank, sp_rank, n


def main():
    # 集計器: config -> [cost, ret, um_hit, sp_hit, hitR, nR]
    agg = {cfg: [0, 0.0, 0, 0, 0, 0] for cfg in CONFIGS}
    races = 0
    for date in DATES:
        bundle = json.load(open(BASE / f"reports/cowork_input/{date}_bundle.json", encoding="utf-8"))
        kek = load_kekka(date)
        for r in bundle["races"]:
            rr = race_ranks(r.get("horses", []))
            if rr is None or rr["gap"] > GAP_GATE:
                continue
            res = kek.get(str(r["race_id"])[:16])
            if res is None:
                continue
            wr = winning_ranks(rr, res)
            if wr is None:
                continue
            um_rank, sp_rank, n = wr
            n_pairs = n * (n - 1); n_tri = comb(n, 3)
            races += 1
            for cfg in CONFIGS:
                k_um, k_sp = cfg
                cu = min(k_um, n_pairs); cs = min(k_sp, n_tri)
                um_hit = um_rank < cu
                sp_hit = sp_rank < cs
                a = agg[cfg]
                a[0] += (cu + cs) * 100                       # cost (100円/点)
                a[1] += (res["umatan"] if um_hit else 0) + (res["sanpuku"] if sp_hit else 0)
                a[2] += um_hit; a[3] += sp_hit
                a[4] += (um_hit or sp_hit); a[5] += 1

    print("=" * 84)
    print(f"2026 ライブ k 比較 (同一{races}R, 同ゲート, prob-first). コストは実在combo数で頭打ち")
    print("=" * 84)
    print(f"{'馬単点':>5s} {'三連複点':>7s} {'投資':>11s} {'払戻':>12s} {'ROI':>7s} "
          f"{'馬単的中%':>9s} {'三連複的中%':>11s} {'R的中%':>7s}")
    for cfg in CONFIGS:
        k_um, k_sp = cfg; a = agg[cfg]
        roi = a[1] / a[0] * 100 if a[0] else 0
        print(f"{k_um:>5d} {k_sp:>7d} {a[0]:>11,} {a[1]:>12,.0f} {roi:>6.1f}% "
              f"{a[2]/a[5]*100:>8.1f}% {a[3]/a[5]*100:>10.1f}% {a[4]/a[5]*100:>6.1f}%")
    print("\n(注: 馬単17/三連複45 ≈ 軸全流し相当=単勝1点と同確率帯。三連複30は中頭数で全combo近くを買う)")


if __name__ == "__main__":
    main()
