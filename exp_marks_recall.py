"""
exp_marks_recall.py — 〇▲△△ の印精度を「セット含有率(recall)」目的で上げられるか検証。

現v6は NDCG(=順序) 最適化。だが馬連/ワイド/3連複は「実1-2-3着が自分の5印に入ってるか」
= セット含有率が効く。これは過去の設定スイープが一度も測ってない指標。
同一特徴・同一params で、学習目的だけ変えた3アームを「セット含有率＋印別複勝圏率」で比較する:
  A baseline : lambdarank × clip6   (現v6, 順序最適化)
  B L_top3   : lambdarank × top3傾斜 (top3領域を均等に重視)
  C bin_top3 : binary    × (着<=3)  (複勝圏membershipを直接予測 → そのprobで印)
判定: B or C が top3_subset_top5 / △複勝圏率を baseline 比で明確に(>+1.0pt)上回れば
      「2-5印は recall 目的で改善余地あり」。横ばいなら 2-5 も天井。
リークなし(特徴/分割不変・設定のみ)。 出力: reports/exp_marks_recall.json
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np

from exp_v6_settings import (
    load_master, load_winner_pay, fit_encoders, apply_encoders, train, metrics,
    LEAK_COLS, COL_RID, COL_JYUN, V6_ALPHA,
)

BASE = Path(__file__).parent
OUT_JSON = BASE / "reports" / "exp_marks_recall.json"


def recall_extra(df, scorecol):
    """セット含有率 + 印別(◎〇▲△1△2) 勝率/連対率/複勝圏率。order[:5] を ◎〇▲△1△2 とみなす。"""
    te = df[df["split"] == "test"]
    n = 0
    sho = np.zeros(5); ren = np.zeros(5); win = np.zeros(5)
    w3 = w5 = t2s5 = t3s5 = t3s4 = t3s3 = 0
    cap3 = 0.0  # 実top3のうち自分のtop5に入った数(0-3)
    for rid, g in te.groupby(COL_RID, sort=False):
        if len(g) < 5:
            continue
        s = g[scorecol].values.astype(float); jy = g[COL_JYUN].astype(int).values
        n += 1
        order = np.argsort(-s)
        top3m = set(int(x) for x in order[:3]); top4m = set(int(x) for x in order[:4])
        top5m = set(int(x) for x in order[:5])
        sij = np.argsort(jy); wi, pi_, si = int(sij[0]), int(sij[1]), int(sij[2])
        actual3 = {wi, pi_, si}
        for i in range(5):
            h = int(order[i])
            if jy[h] <= 3: sho[i] += 1
            if jy[h] <= 2: ren[i] += 1
            if jy[h] == 1: win[i] += 1
        if wi in top3m: w3 += 1
        if wi in top5m: w5 += 1
        if {wi, pi_}.issubset(top5m): t2s5 += 1
        if actual3.issubset(top5m): t3s5 += 1
        if actual3.issubset(top4m): t3s4 += 1
        if actual3 == top3m: t3s3 += 1
        cap3 += len(actual3 & top5m)
    r = lambda x: round(100.0 * x / n, 2)
    return {
        "n_races": n,
        "mark_fukusho_pct": {"◎": r(sho[0]), "〇": r(sho[1]), "▲": r(sho[2]), "△1": r(sho[3]), "△2": r(sho[4])},
        "mark_renta_pct":   {"◎": r(ren[0]), "〇": r(ren[1]), "▲": r(ren[2]), "△1": r(ren[3]), "△2": r(ren[4])},
        "mark_win_pct":     {"◎": r(win[0]), "〇": r(win[1])},
        "winner_in_top3_pct": r(w3), "winner_in_top5_pct": r(w5),
        "top2_subset_top5_pct": r(t2s5),
        "top3_subset_top5_pct": r(t3s5),   # ← 5印BOX三連複の的中上限
        "top3_subset_top4_pct": r(t3s4),   # ← 4頭BOX
        "top3_subset_top3_pct": r(t3s3),   # ← ◎〇▲ 3連複ピタリ
        "avg_top3_captured_in_top5": round(cap3 / n, 4),
    }


def main():
    T0 = time.time(); OUT_JSON.parent.mkdir(exist_ok=True)
    df = load_master()
    df["lbl_bintop3"] = (df[COL_JYUN].astype(int) <= 3).astype(int)   # 複勝圏 membership
    win_pay = load_winner_pay()
    encs = fit_encoders(df[df["split"] == "train"])
    df = apply_encoders(df, encs)
    base_feats = [c for c in df.columns if c not in LEAK_COLS and c != "rid_s" and c != "_dt"
                  and not c.startswith("lbl_")]
    print(f"  base_feats={len(base_feats)}", flush=True)
    tr = df[df["split"] == "train"]; vl = df[df["split"] == "valid"]

    # (obj, label, alpha, ovr, itm, grp)
    arms = {
        "A_baseline_NDCG": ("lambdarank", "lbl_clip6",   V6_ALPHA, {}, 1.1, True),
        "B_L_top3":        ("lambdarank", "lbl_top3",    V6_ALPHA, {}, 1.1, True),
        "C_bin_top3":      ("binary",     "lbl_bintop3", V6_ALPHA, {}, 1.5, False),
    }
    results = {"protocol": "同一特徴120/同一params/学習目的のみ変更/train≤2022→valid2023→test2024-25/市場非経由",
               "metric_focus": "セット含有率(top3_subset_top5) + 印別複勝圏率",
               "verdict_rule": "B or C が top3_subset_top5 を baseline 比 >+1.0pt 上回れば recall 改善余地あり。横ばい→2-5も天井。",
               "arms": {}}

    for name, (obj, lbl, alpha, ovr, itm, grp) in arms.items():
        t0 = time.time()
        model = train(tr, vl, base_feats, obj, lbl, alpha, ovr, itm, grp, win_pay)
        col = f"_s_{name}"
        Xall = df[base_feats].apply(__import__("pandas").to_numeric, errors="coerce").fillna(-9999).values
        df[col] = model.predict(Xall)
        m_std = metrics(df, col)             # AUC/pairwise/hon_top3/ndcg5/winner_in_top5
        m_rec = recall_extra(df, col)        # セット含有率 + 印別
        results["arms"][name] = {"obj": obj, "label": lbl, "best_iter": model.best_iteration,
                                 "order_metrics": m_std, "recall_metrics": m_rec}
        print(f"[{name:16s}] {time.time()-t0:.0f}s it={model.best_iteration} "
              f"◎top3={m_std['hon_top3']*100:.2f}% AUC={m_std['auc_win']} "
              f"top3⊆top5={m_rec['top3_subset_top5_pct']}% △1={m_rec['mark_fukusho_pct']['△1']}% "
              f"△2={m_rec['mark_fukusho_pct']['△2']}%", flush=True)

    # baseline 比デルタ
    A = results["arms"]["A_baseline_NDCG"]
    deltas = {}
    for name, d in results["arms"].items():
        if name == "A_baseline_NDCG":
            continue
        dr = d["recall_metrics"]; ar = A["recall_metrics"]
        deltas[name] = {
            "d_top3_subset_top5": round(dr["top3_subset_top5_pct"] - ar["top3_subset_top5_pct"], 2),
            "d_top3_subset_top4": round(dr["top3_subset_top4_pct"] - ar["top3_subset_top4_pct"], 2),
            "d_winner_in_top5":   round(dr["winner_in_top5_pct"] - ar["winner_in_top5_pct"], 2),
            "d_mark_△1_fuku":     round(dr["mark_fukusho_pct"]["△1"] - ar["mark_fukusho_pct"]["△1"], 2),
            "d_mark_△2_fuku":     round(dr["mark_fukusho_pct"]["△2"] - ar["mark_fukusho_pct"]["△2"], 2),
            "d_hon_top3(◎順序の犠牲)": round((d["order_metrics"]["hon_top3"] - A["order_metrics"]["hon_top3"]) * 100, 2),
        }
    results["deltas_vs_baseline"] = deltas
    best = max(deltas.values(), key=lambda v: v["d_top3_subset_top5"]) if deltas else {}
    improved = any(v["d_top3_subset_top5"] >= 1.0 for v in deltas.values())
    results["verdict"] = ("RECALL_IMPROVEMENT_FOUND: 学習目的を recall 寄りにすると 2-5印のセット含有率が改善。"
                          if improved else
                          "CEILING_2to5_CONFIRMED: 目的を recall に変えても top3_subset_top5 横ばい。"
                          "〇▲△△ の精度も情報量の天井で、学習目的では破れない。")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 78)
    print(f"{'arm':18s}{'◎top3':>8s}{'top3⊆5':>9s}{'top3⊆4':>9s}{'△1複':>7s}{'△2複':>7s}{'win∈5':>8s}")
    for name, d in results["arms"].items():
        o = d["order_metrics"]; rc = d["recall_metrics"]
        print(f"{name:18s}{o['hon_top3']*100:>7.2f}%{rc['top3_subset_top5_pct']:>8.2f}%"
              f"{rc['top3_subset_top4_pct']:>8.2f}%{rc['mark_fukusho_pct']['△1']:>6.1f}%"
              f"{rc['mark_fukusho_pct']['△2']:>6.1f}%{rc['winner_in_top5_pct']:>7.2f}%")
    print(f"\n判定: {results['verdict']}")
    print(f"[saved] {OUT_JSON}  ({time.time()-T0:.0f}s)")


if __name__ == "__main__":
    main()
