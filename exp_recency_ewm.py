"""
exp_recency_ewm.py — 【新角度】近走重み付け(EWMA)の多走集計。

現状の多走集計(kako5_avg_pos 等)は全部フラット平均=5走前の凡走も前走も等加重。
さらに RPCI/PCI/着差 等の質的特徴は「前走1走だけ」。
ここで主要signalを 血統登録番号でJOIN→日付順→EWMA(近走ほど重い, halflife可変)で
多走集計し直す。前走列(=過去走の値)を EWMA するのでリークなし。
  A baseline : 現120特徴
  B +ewm     : 120 + EWMA多走集計(着順/上り/RPCI/着差/補正T/PCI)
判定(全死亡実験と同じ基準): ΔAUC≥+0.008 & Δpairwise≥+0.005 → 初の本物の増分。
出力: reports/exp_recency_ewm.json
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pandas as pd

from exp_v6_settings import (
    load_master, load_winner_pay, fit_encoders, apply_encoders, train, metrics,
    LEAK_COLS, COL_RID, COL_JYUN, V6_ALPHA,
)
from exp_marks_recall import recall_extra

BASE = Path(__file__).parent
OUT_JSON = BASE / "reports" / "exp_recency_ewm.json"
COL_PED = "血統登録番号"
HALFLIFE = 1.8   # 近走をどれだけ重くするか(=約2走で半減)


def add_ewm(df, halflife=HALFLIFE):
    """前走系signalを 血統登録番号×日付順で EWMA(近走重み)。前走値=過去走なのでリークなし。"""
    df = df.copy()
    df["_d"] = pd.to_numeric(df["日付"], errors="coerce")
    df = df.sort_values([COL_PED, "_d", COL_RID])
    sigs = [("pos", "前走確定着順"), ("agari", "前走上り3F"), ("rpci", "前走RPCI"),
            ("chakusa", "前走着差タイム"), ("hosei", "prev_hosei"), ("pci", "前PCI"),
            ("avg1f", "前走平均1Fタイム")]
    cols = []
    gp = df.groupby(COL_PED, sort=False)
    for tag, src in sigs:
        if src not in df.columns:
            continue
        s = pd.to_numeric(df[src], errors="coerce")
        c = f"ewm_{tag}"
        df[c] = gp[src].transform(lambda x: pd.to_numeric(x, errors="coerce")
                                  .ewm(halflife=halflife, min_periods=1).mean())
        # 前走値との差 (近走が EWMA より良いか=上昇トレンド)
        df[f"ewm_{tag}_vs_prev"] = s - df[c]
        cols += [c, f"ewm_{tag}_vs_prev"]
    df = df.sort_index().drop(columns=["_d"])
    return df, cols


def evalblock(df, col):
    m = metrics(df, col)
    r = recall_extra(df, col)
    m["top3_subset_top5"] = r["top3_subset_top5_pct"]
    m["top3_subset_top4"] = r["top3_subset_top4_pct"]
    return m


def main():
    T0 = time.time(); OUT_JSON.parent.mkdir(exist_ok=True)
    df = load_master()
    df, ewm_cols = add_ewm(df)
    win_pay = load_winner_pay()
    encs = fit_encoders(df[df["split"] == "train"])
    df = apply_encoders(df, encs)
    base_feats = [c for c in df.columns if c not in LEAK_COLS and c != "rid_s" and c != "_dt"
                  and not c.startswith("lbl_") and not c.startswith("ewm_")]
    print(f"  base_feats={len(base_feats)}  ewm_feats={len(ewm_cols)} halflife={HALFLIFE}", flush=True)
    cov = {c: round(df.loc[df.split == "test", c].notna().mean() * 100, 1) for c in ewm_cols}

    tr = df[df["split"] == "train"]; vl = df[df["split"] == "valid"]
    arms = {"A_baseline": base_feats, "B_ewm": base_feats + ewm_cols}
    results = {"protocol": "新角度=近走重みEWMA多走集計(着順/上り/RPCI/着差/補正T/PCI/1F)。"
                           "前走値を血統登録番号×日付でEWMA(リークなし)。train≤2022→valid2023→test2024-25",
               "halflife": HALFLIFE, "ewm_cols": ewm_cols, "coverage_test": cov,
               "verdict_rule": "ΔAUC≥+0.008 & Δpairwise≥+0.005 → 初の本物の増分。横ばい→減衰も天井。",
               "arms": {}}

    for name, feats in arms.items():
        t0 = time.time()
        model = train(tr, vl, feats, "lambdarank", "lbl_clip6", V6_ALPHA, {}, 1.1, True, win_pay)
        col = f"_s_{name}"
        Xall = df[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
        df[col] = model.predict(Xall)
        m = evalblock(df, col)
        gi = pd.Series(model.feature_importance("gain"), index=feats)
        ewm_rank = ({c: int(gi.sort_values(ascending=False).index.get_loc(c) + 1) for c in ewm_cols}
                    if name == "B_ewm" else {})
        ewm_gain = ({c: round(float(gi.get(c, 0.0)), 1) for c in ewm_cols} if name == "B_ewm" else {})
        results["arms"][name] = {"n_feats": len(feats), "best_iter": model.best_iteration,
                                 "metrics": m, "ewm_rank_of_total": ewm_rank, "ewm_gain": ewm_gain}
        print(f"[{name:11s}] {time.time()-t0:.0f}s it={model.best_iteration} AUC={m['auc_win']} "
              f"pair={m['pairwise_acc']} ◎top3={m['hon_top3']*100:.2f}% top3⊆5={m['top3_subset_top5']}%", flush=True)

    A = results["arms"]["A_baseline"]["metrics"]; B = results["arms"]["B_ewm"]["metrics"]
    d = {"dAUC": round(B["auc_win"] - A["auc_win"], 5),
         "dpairwise": round(B["pairwise_acc"] - A["pairwise_acc"], 5),
         "dlogloss": round(B["logloss_win"] - A["logloss_win"], 5),
         "dtop3_pt": round((B["hon_top3"] - A["hon_top3"]) * 100, 3),
         "d_top3_subset_top5": round(B["top3_subset_top5"] - A["top3_subset_top5"], 3)}
    d["WIN"] = bool(d["dAUC"] >= 0.008 and d["dpairwise"] >= 0.005)
    results["delta"] = d
    results["verdict"] = ("INCREMENT_FOUND: 近走重みEWMAで初めて事前基準クリア。要精緻化(halflife最適化等)。"
                          if d["WIN"] else
                          f"CEILING: EWMA多走でも ΔAUC={d['dAUC']:+.5f}/Δpair={d['dpairwise']:+.5f} 基準未達。"
                          "近走重み付けも per-horse 既存集計に冗長。")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 72)
    print(f"baseline : AUC={A['auc_win']} pair={A['pairwise_acc']} ◎top3={A['hon_top3']*100:.2f}%")
    print(f"+ewm     : AUC={B['auc_win']} pair={B['pairwise_acc']} ◎top3={B['hon_top3']*100:.2f}%")
    print(f"Δ        : dAUC={d['dAUC']:+.5f} dpair={d['dpairwise']:+.5f} dtop3={d['dtop3_pt']:+.3f}pt WIN={d['WIN']}")
    pr = results["arms"]["B_ewm"]["ewm_rank_of_total"]; pg = results["arms"]["B_ewm"]["ewm_gain"]
    print(f"\nEWMA特徴の gain 順位(全{results['arms']['B_ewm']['n_feats']}特徴中):")
    for c in sorted(pg, key=lambda x: -pg[x])[:8]:
        print(f"  {c:20s} rank={pr[c]:>3}  gain={pg[c]}")
    print(f"\n判定: {results['verdict']}")
    print(f"[saved] {OUT_JSON}  ({time.time()-T0:.0f}s)")


if __name__ == "__main__":
    main()
