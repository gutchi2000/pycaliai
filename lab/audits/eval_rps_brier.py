"""
eval_rps_brier.py — v6 の確率予測を「適正スコアリングルール」で診断する。

NDCG は順序しか見ず較正に盲目。ここでは EV 目的に効く指標を test 2024-25 で測る:
  - RPS (Ranked Probability Score): 着順を順序尺度 {1着 / 連対 / 複勝圏 / 着外} の
    累積分布で評価。2着予想が「3着外れ」と「10着外れ」を区別して罰する (Epstein 1969)。
  - Brier score: 勝 / 連対 / 複勝 の各2値に対する適正スコア。
  - Murphy 分解: Brier = 信頼性(reliability) - 分解能(resolution) + 不確実性(uncertainty)。
    「較正のズレ(rel)」だけを取り出して、どこで過信/過小が起きてるか診断する (Murphy 1973)。

採点経路は本番 v6 を再利用 (unified_rank_v6 → PL 閉形式 → pl_calibrators_v6)。
p_win/p_sho は本番較正器、p_plc(連対) は calibrator 不在のため valid(2023) で isotonic を自前 fit。
"""
import os
import json
from pathlib import Path

# ★ backtest_pl_ev は import 時に MODEL_PKL/CAL_PKL を env から読むので、import 前に v6 を指定
os.environ.setdefault("UNIFIED_MODEL", str(Path(__file__).parent / "models/unified_rank_v6.pkl"))
os.environ.setdefault("UNIFIED_CAL",   str(Path(__file__).parent / "models/pl_calibrators_v6.pkl"))

import numpy as np
import joblib
from sklearn.isotonic import IsotonicRegression

import pl_probs as PL
from backtest_pl_ev import score_test, all_fukusho_vec_fast, CAL_PKL, COL_JYUN, COL_RID, COL_BAN

BASE = Path(__file__).parent
OUT_JSON = BASE / "reports" / "eval_rps_brier.json"


def all_renta_vec(w):
    """各馬の P(連対=2着以内) を PL 閉形式で。p_place_at(pos=1)+p_place_at(pos=2)."""
    n = len(w)
    return np.array([PL.p_place_at(w, i, 1) + PL.p_place_at(w, i, 2) for i in range(n)])


def collect(te):
    """race ごとに raw 周辺確率 (p_win/p_plc/p_sho) と着順カテゴリを集める。"""
    rows = {"p_win_raw": [], "p_plc_raw": [], "p_sho_raw": [],
            "y_win": [], "y_plc": [], "y_sho": [], "jyun": [], "field": []}
    extra = {"class": [], "year": []}
    has_class = "クラス" in te.columns
    for rid, g in te.groupby(COL_RID, sort=False):
        if len(g) < 3:
            continue
        g = g.sort_values(COL_BAN)
        s = g["_score"].values.astype(float)
        j = g[COL_JYUN].values.astype(float)
        w = PL.pl_weights(s)
        p_win = w / w.sum()
        p_plc = all_renta_vec(w)
        p_sho = all_fukusho_vec_fast(w)
        n = len(w)
        rows["p_win_raw"].append(p_win); rows["p_plc_raw"].append(p_plc); rows["p_sho_raw"].append(p_sho)
        rows["y_win"].append((j <= 1).astype(float))
        rows["y_plc"].append((j <= 2).astype(float))
        rows["y_sho"].append((j <= 3).astype(float))
        rows["jyun"].append(j); rows["field"].append(np.full(n, n))
        extra["class"].append(np.array([str(g["クラス"].iloc[0])] * n) if has_class else np.full(n, "NA"))
        extra["year"].append(np.full(n, int(str(rid)[:4])))
    out = {k: np.concatenate(v) for k, v in rows.items()}
    out["class"] = np.concatenate(extra["class"]); out["year"] = np.concatenate(extra["year"])
    return out


def murphy(p, y, bins=10):
    """Brier の Murphy 分解。返り値: brier, reliability(↓良), resolution(↑良), uncertainty, ece."""
    p = np.clip(p, 0, 1); y = y.astype(float)
    obar = float(y.mean()); unc = obar * (1 - obar)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rel = res = 0.0; ece = 0.0
    for k in range(bins):
        m = idx == k
        nk = int(m.sum())
        if nk == 0:
            continue
        pbar = float(p[m].mean()); obar_k = float(y[m].mean())
        rel += nk * (pbar - obar_k) ** 2
        res += nk * (obar_k - obar) ** 2
        ece += nk * abs(pbar - obar_k)
    rel /= len(p); res /= len(p); ece /= len(p)
    brier = float(((p - y) ** 2).mean())
    return dict(brier=round(brier, 5), reliability=round(rel, 6), resolution=round(res, 5),
                uncertainty=round(unc, 5), ece=round(ece, 5), n=int(len(p)))


def rps_finish(p_win, p_plc, p_sho, y_win, y_plc, y_sho):
    """着順4カテゴリ {1着/連対/複勝圏/着外} の累積RPS。F_pred=[p_win,p_plc,p_sho]を単調化して比較。"""
    # 累積を単調に (較正で順序が崩れる場合の保険)
    f1 = p_win
    f2 = np.maximum(f1, np.minimum(p_plc, 1.0))
    f3 = np.maximum(f2, np.minimum(p_sho, 1.0))
    a1, a2, a3 = y_win, y_plc, y_sho
    rps = ((f1 - a1) ** 2 + (f2 - a2) ** 2 + (f3 - a3) ** 2) / 3.0
    return float(rps.mean())


def main():
    OUT_JSON.parent.mkdir(exist_ok=True)
    print(f"[model] {os.environ['UNIFIED_MODEL']}")
    print(f"[cal  ] {os.environ['UNIFIED_CAL']}")
    cal = joblib.load(CAL_PKL)["calibrators"]

    print("scoring valid+test ...")
    te_all = score_test(include_valid=True)
    te_all["year"] = te_all[COL_RID].astype(str).str[:4].astype(int)
    vl = te_all[te_all["split"] == "valid"].copy()
    te = te_all[te_all["split"] == "test"].copy()
    print(f"  valid races horses={len(vl)}  test horses={len(te)}")

    D_vl = collect(vl)
    D_te = collect(te)

    # p_plc(連対) calibrator を valid で自前 fit (本番 pkl に無いため)
    iso_plc = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    iso_plc.fit(D_vl["p_plc_raw"], D_vl["y_plc"])

    # 較正適用 (test)
    cal_win = cal["tansho"].predict(D_te["p_win_raw"])
    cal_plc = iso_plc.predict(D_te["p_plc_raw"])
    cal_sho = cal["fukusho"].predict(D_te["p_sho_raw"])

    def block(p_win, p_plc, p_sho, tag):
        return {
            "RPS_finish": round(rps_finish(p_win, p_plc, p_sho,
                                           D_te["y_win"], D_te["y_plc"], D_te["y_sho"]), 6),
            "Brier_win":  murphy(p_win, D_te["y_win"]),
            "Brier_renta": murphy(p_plc, D_te["y_plc"]),
            "Brier_sho":  murphy(p_sho, D_te["y_sho"]),
        }

    res = {
        "protocol": "v6本番採点 / test2024-25 / RPS(着順4cat累積)+Brier+Murphy分解 / 市場非経由",
        "model": os.environ["UNIFIED_MODEL"], "cal": os.environ["UNIFIED_CAL"],
        "n_test_horses": int(len(D_te["y_win"])),
        "base_rate": {"win": round(float(D_te["y_win"].mean()), 4),
                       "renta": round(float(D_te["y_plc"].mean()), 4),
                       "sho": round(float(D_te["y_sho"].mean()), 4)},
        "RAW_周辺確率":  block(D_te["p_win_raw"], D_te["p_plc_raw"], D_te["p_sho_raw"], "raw"),
        "CALIBRATED_v6": block(cal_win, cal_plc, cal_sho, "cal"),
    }

    # どこで較正が崩れるか: p_win ビン別 reliability (CAL)
    res["by_pwin_bin_reliability_CAL"] = {}
    edges = [0, 0.02, 0.05, 0.10, 0.20, 0.40, 1.01]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (cal_win >= lo) & (cal_win < hi)
        if m.sum() < 50:
            continue
        res["by_pwin_bin_reliability_CAL"][f"[{lo},{hi})"] = {
            "n": int(m.sum()),
            "p_pred_mean": round(float(cal_win[m].mean()), 4),
            "obs_win_rate": round(float(D_te["y_win"][m].mean()), 4),
            "gap(pred-obs)": round(float(cal_win[m].mean() - D_te["y_win"][m].mean()), 4),
        }

    # クラス別 (あれば) RPS / Brier_win reliability
    if not (D_te["class"] == "NA").all():
        res["by_class"] = {}
        for c in np.unique(D_te["class"]):
            m = D_te["class"] == c
            if m.sum() < 200:
                continue
            res["by_class"][c] = {
                "n": int(m.sum()),
                "RPS": round(rps_finish(cal_win[m], cal_plc[m], cal_sho[m],
                                        D_te["y_win"][m], D_te["y_plc"][m], D_te["y_sho"][m]), 6),
                "Brier_win_reliability": murphy(cal_win[m], D_te["y_win"][m])["reliability"],
            }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    # サマリ表示
    print("\n===== v6 適正スコアリング診断 (test 2024-25) =====")
    print(f"n_test_horses={res['n_test_horses']}  base(win/連対/複)={res['base_rate']}")
    for tag in ["RAW_周辺確率", "CALIBRATED_v6"]:
        b = res[tag]
        print(f"\n[{tag}]  RPS_finish={b['RPS_finish']}")
        for k in ["Brier_win", "Brier_renta", "Brier_sho"]:
            d = b[k]
            print(f"  {k:12s} Brier={d['brier']:.5f}  reliability(較正ズレ↓)={d['reliability']:.6f}  "
                  f"resolution(分解能↑)={d['resolution']:.5f}  ECE={d['ece']:.5f}")
    print("\n[p_win ビン別 較正(CAL)]  pred vs 実勝率")
    for k, v in res["by_pwin_bin_reliability_CAL"].items():
        print(f"  {k:12s} n={v['n']:>6} pred={v['p_pred_mean']:.4f} obs={v['obs_win_rate']:.4f} "
              f"gap={v['gap(pred-obs)']:+.4f}")
    print(f"\n[saved] {OUT_JSON}")


if __name__ == "__main__":
    main()
