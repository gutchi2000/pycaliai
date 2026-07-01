"""
exp_pace_interaction.py — 【新角度】馬同士の相互作用＝展開(ペース構成)を初投入。

これまでの死んだ実験は全部「馬を独立評価→ランク→PL(IIA)」で、馬同士の相互作用を一度も入れてない。
ここでは各馬の【過去の脚質】(前走前方度 prev_pos_rel / 追込力 closing_power / 前走RPCI / 上り順)を
【そのレースの全馬で集約】し、「逃げ馬何頭/単騎逃げ/ペース圧×脚質フィット」= 展開の相互作用特徴を作る。
全入力は前走(=出走前に既知)なのでリークなし。現フィールドの結果は一切不使用。

A baseline : 現120特徴
B +pace    : 120 + 展開構成10特徴
判定(全死亡実験と同じ事前固定基準): ΔAUC≥+0.008 & Δpairwise≥+0.005 → 初の本物の増分。
横ばい→展開も per-horse 特徴/市場に吸収済で天井。 出力: reports/exp_pace_interaction.json
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
OUT_JSON = BASE / "reports" / "exp_pace_interaction_v2.json"


def add_pace_comp(df):
    """【v2】展開構成。脚質proxyを kako5_avg_pos(5走平均位置,80%) + 前4角(90%) に強化。
    全部 前走/過去スタイルを現フィールドで集約=リークなし(現レース結果は不使用)。"""
    df = df.copy()
    ka = pd.to_numeric(df.get("kako5_avg_pos"), errors="coerce")        # 1=前…18=後 (5走平均位置)
    n_prev = pd.to_numeric(df.get("前走出走頭数"), errors="coerce").clip(lower=2)
    zen4 = pd.to_numeric(df.get("前4角"), errors="coerce") / n_prev     # 0=前…1=後 (前走最終角, 90%)
    cp = pd.to_numeric(df.get("closing_power"), errors="coerce")
    rpci = pd.to_numeric(df.get("前走RPCI"), errors="coerce")
    df["_ka"] = ka; df["_zen4"] = zen4; df["_rpci"] = rpci
    g = df.groupby(COL_RID)
    n = g[COL_RID].transform("size").astype(float)

    df["pc_ka_rank"]    = g["_ka"].rank(method="average") / n           # 0=フィールドで最も前で走る馬
    df["pc_zen4_rank"]  = g["_zen4"].rank(method="average") / n
    df["pc_ka_vs_field"] = ka - g["_ka"].transform("mean")             # 平均脚質より前か後か

    is_front = (df["pc_ka_rank"] <= 0.25).astype(float)                # 前付け脚質(前方25%)
    df["_isf"] = is_front
    n_front = df.groupby(COL_RID)["_isf"].transform("sum")
    df["pc_n_front"]       = n_front                                    # 前付け頭数(ペース圧)
    df["pc_front_density"] = n_front / n
    df["pc_front_congest"] = np.where(is_front == 1, np.maximum(n_front - 1, 0), 0.0)

    # 単騎逃げ: レース内 最前(min ka)の馬で、2番目に前の馬との差>=2 (=楽に行ける)
    ka_min = g["_ka"].transform("min")
    ka_2nd = g["_ka"].transform(lambda s: (np.sort(s.dropna().values)[1]
                                           if s.notna().sum() >= 2 else np.nan))
    df["pc_is_lone_front"] = ((ka == ka_min) & ((ka_2nd - ka_min) >= 2.0)).astype(float)

    df["pc_closer_fit"]    = df["pc_ka_rank"] * df["pc_front_density"]  # 後方脚質×ペース圧=差し展開利
    df["pc_field_rpci"]    = g["_rpci"].transform("mean")               # フィールド平均RPCI(速い展開?)
    df["pc_rpci_vs_field"] = rpci - df["pc_field_rpci"]

    pace_cols = ["pc_ka_rank", "pc_zen4_rank", "pc_ka_vs_field", "pc_n_front", "pc_front_density",
                 "pc_front_congest", "pc_is_lone_front", "pc_closer_fit", "pc_field_rpci",
                 "pc_rpci_vs_field"]
    df = df.drop(columns=["_ka", "_zen4", "_rpci", "_isf"])
    return df, pace_cols


def evalblock(df, col):
    m = metrics(df, col)
    r = recall_extra(df, col)
    m["top3_subset_top5"] = r["top3_subset_top5_pct"]
    m["top3_subset_top4"] = r["top3_subset_top4_pct"]
    return m


def main():
    T0 = time.time(); OUT_JSON.parent.mkdir(exist_ok=True)
    df = load_master()
    df, pace_cols = add_pace_comp(df)
    win_pay = load_winner_pay()
    encs = fit_encoders(df[df["split"] == "train"])
    df = apply_encoders(df, encs)
    base_feats = [c for c in df.columns if c not in LEAK_COLS and c != "rid_s" and c != "_dt"
                  and not c.startswith("lbl_") and not c.startswith("pc_")]
    print(f"  base_feats={len(base_feats)}  pace_feats={len(pace_cols)}", flush=True)

    # 展開特徴のカバレッジ
    cov = {c: round(df.loc[df.split == "test", c].notna().mean() * 100, 1) for c in pace_cols}

    tr = df[df["split"] == "train"]; vl = df[df["split"] == "valid"]
    arms = {"A_baseline": base_feats, "B_pace": base_feats + pace_cols}
    results = {"protocol": "新角度=展開(ペース構成)相互作用。前走スタイルを現フィールド集約(リークなし)。"
                           "train≤2022→valid2023→test2024-25/市場非経由",
               "pace_cols": pace_cols, "coverage_test": cov,
               "verdict_rule": "ΔAUC≥+0.008 & Δpairwise≥+0.005 → 初の本物の増分。横ばい→展開も天井。",
               "arms": {}}

    for name, feats in arms.items():
        t0 = time.time()
        model = train(tr, vl, feats, "lambdarank", "lbl_clip6", V6_ALPHA, {}, 1.1, True, win_pay)
        col = f"_s_{name}"
        Xall = df[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
        df[col] = model.predict(Xall)
        m = evalblock(df, col)
        gainimp = pd.Series(model.feature_importance("gain"), index=feats)
        pace_gain = {c: round(float(gainimp.get(c, 0.0)), 1) for c in (pace_cols if name == "B_pace" else [])}
        pace_rank = {c: int((gainimp.sort_values(ascending=False).index.get_loc(c) + 1))
                     for c in pace_cols} if name == "B_pace" else {}
        results["arms"][name] = {"n_feats": len(feats), "best_iter": model.best_iteration,
                                 "metrics": m, "pace_gain": pace_gain, "pace_rank_of_total": pace_rank}
        print(f"[{name:11s}] {time.time()-t0:.0f}s it={model.best_iteration} AUC={m['auc_win']} "
              f"pair={m['pairwise_acc']} ◎top3={m['hon_top3']*100:.2f}% "
              f"top3⊆5={m['top3_subset_top5']}%", flush=True)

    A = results["arms"]["A_baseline"]["metrics"]; B = results["arms"]["B_pace"]["metrics"]
    d = {"dAUC": round(B["auc_win"] - A["auc_win"], 5),
         "dpairwise": round(B["pairwise_acc"] - A["pairwise_acc"], 5),
         "dlogloss": round(B["logloss_win"] - A["logloss_win"], 5),
         "dtop3_pt": round((B["hon_top3"] - A["hon_top3"]) * 100, 3),
         "d_top3_subset_top5": round(B["top3_subset_top5"] - A["top3_subset_top5"], 3),
         "d_top3_subset_top4": round(B["top3_subset_top4"] - A["top3_subset_top4"], 3)}
    d["WIN"] = bool(d["dAUC"] >= 0.008 and d["dpairwise"] >= 0.005)
    results["delta"] = d
    results["verdict"] = ("INCREMENT_FOUND: 展開(相互作用)で初めて事前基準クリア。新角度に signal あり。要精緻化。"
                          if d["WIN"] else
                          f"CEILING: 展開特徴でも ΔAUC={d['dAUC']:+.5f}/Δpair={d['dpairwise']:+.5f} で基準未達。"
                          "相互作用も per-horse 特徴に吸収済。ただし dtop3/top3⊆top5 の符号は要確認。")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 72)
    print(f"baseline : AUC={A['auc_win']} pair={A['pairwise_acc']} ◎top3={A['hon_top3']*100:.2f}% "
          f"top3⊆5={A['top3_subset_top5']}%")
    print(f"+pace    : AUC={B['auc_win']} pair={B['pairwise_acc']} ◎top3={B['hon_top3']*100:.2f}% "
          f"top3⊆5={B['top3_subset_top5']}%")
    print(f"Δ        : dAUC={d['dAUC']:+.5f}  dpair={d['dpairwise']:+.5f}  dtop3={d['dtop3_pt']:+.3f}pt  "
          f"d(top3⊆5)={d['d_top3_subset_top5']:+.3f}pt  WIN={d['WIN']}")
    # 展開特徴の重要度(モデルが使ったか)
    pr = results["arms"]["B_pace"]["pace_rank_of_total"]
    pg = results["arms"]["B_pace"]["pace_gain"]
    print("\n展開特徴の gain 順位(全%d特徴中) / gain:" % results["arms"]["B_pace"]["n_feats"])
    for c in sorted(pg, key=lambda x: -pg[x])[:10]:
        print(f"  {c:22s} rank={pr[c]:>3}  gain={pg[c]}")
    print(f"\n判定: {results['verdict']}")
    print(f"[saved] {OUT_JSON}  ({time.time()-T0:.0f}s)")


if __name__ == "__main__":
    main()
