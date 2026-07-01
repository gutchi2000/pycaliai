"""
exp_recency_sweep.py — 近走重みEWMA の halflife スイープ + valid/test 両確認。
exp_recency_ewm(halflife=1.8) で初の全指標プラス(+0.23pt ◎top3)。本物かノイズかを:
  - halflife ∈ {0.7,1.2,1.8,3.0,6.0} で最適値を探す
  - valid(2023) と test(2024-25) の両方で効くか (片方だけ=過学習/ノイズ)
で見極める。valid で選んだ halflife が test でも効けば本物。
出力: reports/exp_recency_sweep.json
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pandas as pd

from exp_v6_settings import (load_master, load_winner_pay, fit_encoders, apply_encoders,
                             train, metrics, LEAK_COLS, COL_RID, V6_ALPHA)
from exp_recency_ewm import add_ewm

OUT_JSON = Path(__file__).parent / "reports" / "exp_recency_sweep.json"


def eval_vt(df, col):
    m_te = metrics(df, col)
    dfv = df[df["split"] == "valid"].assign(split="test")
    m_va = metrics(dfv, col)
    return ({"auc": m_va["auc_win"], "pair": m_va["pairwise_acc"], "top3": m_va["hon_top3"]},
            {"auc": m_te["auc_win"], "pair": m_te["pairwise_acc"], "top3": m_te["hon_top3"]})


def fit_eval(dfb, feats, win_pay):
    tr = dfb[dfb["split"] == "train"]; vl = dfb[dfb["split"] == "valid"]
    model = train(tr, vl, feats, "lambdarank", "lbl_clip6", V6_ALPHA, {}, 1.1, True, win_pay)
    col = "_s"
    dfb[col] = model.predict(dfb[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values)
    return eval_vt(dfb, col)


def main():
    T0 = time.time()
    df = load_master(); win_pay = load_winner_pay()
    encs = fit_encoders(df[df["split"] == "train"]); df = apply_encoders(df, encs)
    base_feats = [c for c in df.columns if c not in LEAK_COLS and c != "rid_s" and c != "_dt"
                  and not c.startswith("lbl_") and not c.startswith("ewm_")]
    print(f"  base_feats={len(base_feats)}", flush=True)

    res = {"halflives": [0.7, 1.2, 1.8, 3.0, 6.0], "rows": {}}
    # baseline
    va, te = fit_eval(df.copy(), base_feats, win_pay)
    res["baseline"] = {"valid": va, "test": te}
    print(f"[baseline    ] valid top3={va['top3']*100:.2f}% AUC={va['auc']} | "
          f"test top3={te['top3']*100:.2f}% AUC={te['auc']}", flush=True)

    for hl in res["halflives"]:
        t0 = time.time()
        dfh, ewm_cols = add_ewm(df, halflife=hl)
        va, te = fit_eval(dfh, base_feats + ewm_cols, win_pay)
        res["rows"][str(hl)] = {
            "valid": va, "test": te,
            "d_valid_top3_pt": round((va["top3"] - res["baseline"]["valid"]["top3"]) * 100, 3),
            "d_test_top3_pt": round((te["top3"] - res["baseline"]["test"]["top3"]) * 100, 3),
            "d_valid_auc": round(va["auc"] - res["baseline"]["valid"]["auc"], 5),
            "d_test_auc": round(te["auc"] - res["baseline"]["test"]["auc"], 5),
            "d_test_pair": round(te["pair"] - res["baseline"]["test"]["pair"], 5),
        }
        r = res["rows"][str(hl)]
        print(f"[hl={hl:<4}] {time.time()-t0:.0f}s  valid Δtop3={r['d_valid_top3_pt']:+.3f}pt "
              f"Δauc={r['d_valid_auc']:+.5f} | test Δtop3={r['d_test_top3_pt']:+.3f}pt "
              f"Δauc={r['d_test_auc']:+.5f} Δpair={r['d_test_pair']:+.5f}", flush=True)

    # valid で最良の halflife を選び、test で効くか
    best_hl = max(res["rows"], key=lambda k: res["rows"][k]["d_valid_top3_pt"])
    bt = res["rows"][best_hl]
    res["best_hl_by_valid"] = best_hl
    consistent = (bt["d_valid_top3_pt"] > 0 and bt["d_test_top3_pt"] > 0
                  and bt["d_test_auc"] > 0 and bt["d_test_pair"] > 0)
    res["verdict"] = (f"REAL_SMALL_GAIN: valid最良 hl={best_hl} が test でも全指標プラス "
                      f"(test Δtop3={bt['d_test_top3_pt']:+.3f}pt Δauc={bt['d_test_auc']:+.5f})。"
                      "ノイズでなく一貫した小改善=v6に折込む価値あり。"
                      if consistent else
                      f"NOISE/INCONSISTENT: valid最良 hl={best_hl} は test で符号が揃わず "
                      f"(test Δtop3={bt['d_test_top3_pt']:+.3f}pt)。近走重みは過学習/ノイズ。")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("\n" + "=" * 76)
    print(f"{'halflife':>10}{'valid Δtop3':>14}{'test Δtop3':>13}{'test Δauc':>12}{'test Δpair':>12}")
    for hl in res["halflives"]:
        r = res["rows"][str(hl)]
        print(f"{hl:>10}{r['d_valid_top3_pt']:>+13.3f}{r['d_test_top3_pt']:>+12.3f}"
              f"{r['d_test_auc']:>+12.5f}{r['d_test_pair']:>+12.5f}")
    print(f"\nvalid最良 halflife={best_hl}")
    print(f"判定: {res['verdict']}")
    print(f"[saved] {OUT_JSON}  ({time.time()-T0:.0f}s)")


if __name__ == "__main__":
    main()
