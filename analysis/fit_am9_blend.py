"""
fit_am9_blend.py
================
朝の補正印のための blend 係数 λ を、本番 v7 の確率で fit する (2026-11-14)。

補正印 = レース内で  u_i = log(p_win_i) + λ·log(π_i)  の降順に付け直した印。
  p_win = v7 の配信用較正 PL 単勝確率 (bundle の p_win と同じ作り方: pl_calibrators_v7_serve)
  π     = de-vig 市場単勝確率。fit は当日 9 時前後のスナップ (data/_joint/mkt_horses.parquet)

analysis/fit_t10_blend.py (v6 用、data/t10_blend.json) と同じ手順:
  valid=2023 で ◎3着内を直接最大化する λ を選び、test=2024-25 で 1 回評価 (paired bootstrap)。
違いは土台が v7 であること、出力先が data/am9_blend.json であること、v6 の T-10 用係数には触れないこと。

入力 (どちらも git 管理外。v7 スコアは JRDB 由来なので出力には集計値だけを書く):
  reports/gate_base.parquet (lab/experiments/gate_common.py --build が作る。valid+test の v7 生スコア)
  data/_joint/mkt_horses.parquet (9 時の de-vig π と 3 着内ラベル)
  models/pl_calibrators_v7_serve.pkl
実行: venv311/Scripts/python.exe -m analysis.fit_am9_blend
      (データが別の場所にあるときは環境変数 PYCALIAI_DATA_ROOT=E:/PyCaLiAI)
出力: data/am9_blend.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
ROOT = Path(os.environ.get("PYCALIAI_DATA_ROOT", BASE))
sys.path.insert(0, str(BASE))
import pl_probs as PL  # noqa: E402
from analysis.fit_t10_blend import LAMBDA_GRID, paired_boot_delta, pick_hits, top5_mean_hits, wilson_ci  # noqa: E402

OUT = BASE / "data" / "am9_blend.json"


def load_races() -> list[dict]:
    sc = pd.read_parquet(BASE / "reports" / "gate_base.parquet", columns=["rid_s", "ban", "date", "s_v7", "jyun"])
    mk = pd.read_parquet(ROOT / "data" / "_joint" / "mkt_horses.parquet", columns=["rid", "ban", "pi"])
    mk["rid_s"] = mk["rid"].astype(str).str[:16]
    df = sc.merge(mk[["rid_s", "ban", "pi"]], on=["rid_s", "ban"], how="inner")
    cal = joblib.load(ROOT / "models" / "pl_calibrators_v7_serve.pkl")
    iso_tan = cal.get("calibrators", cal)["tansho"]
    n_sc = sc.rid_s.nunique()
    races = []
    for rid, g in df.groupby("rid_s", sort=False):
        if len(g) < 5 or (g["pi"] <= 0).any():   # 9 時オッズの無い馬 (取消等) は落ちる。5 頭未満は除外
            continue
        g = g.sort_values("ban")
        f = np.maximum(iso_tan.predict(PL.all_tansho(PL.pl_weights(g["s_v7"].values))), 1e-9)
        races.append({"year": int(str(g["date"].iloc[0])[:4]), "logf": np.log(f), "logpi": np.log(g["pi"].values),
                      "top3": (g["jyun"].values <= 3).astype(int), "win": (g["jyun"].values == 1).astype(int)})
    print(f"[races] {len(races):,} / {n_sc:,} レース (9 時オッズと結合できたもの)")
    return races


def win_hits(races, lam):
    out = np.empty(len(races), dtype=int)
    for i, r in enumerate(races):
        u = r["logpi"] if np.isinf(lam) else r["logf"] + lam * r["logpi"]
        out[i] = r["win"][int(np.argmax(u))]
    return out


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    races = load_races()
    valid = [r for r in races if r["year"] == 2023]
    test = [r for r in races if r["year"] >= 2024]
    curve = {lam: float(pick_hits(valid, lam).mean()) for lam in LAMBDA_GRID}
    for lam, v in curve.items():
        print(f"  valid λ={lam:>5}: ◎3着内 {100 * v:.2f}%")
    finite = {k: v for k, v in curve.items() if np.isfinite(k)}
    lam_star = max(finite, key=finite.get)
    h0, hm, hb = pick_hits(test, 0.0), pick_hits(test, np.inf), pick_hits(test, lam_star)
    n = len(test)
    d_lo, d_hi = paired_boot_delta(hb, h0); m_lo, m_hi = paired_boot_delta(hb, hm)
    w0, wb = win_hits(test, 0.0), win_hits(test, lam_star); w_lo, w_hi = paired_boot_delta(wb, w0)
    print(f"→ λ* = {lam_star}")
    for name, h in (("v7 のみ", h0), ("9 時市場のみ", hm), (f"blend λ={lam_star}", hb)):
        lo, hi = wilson_ci(int(h.sum()), n)
        print(f"  test {name:<14}: ◎3着内 {100 * h.mean():.2f}% [{100 * lo:.2f},{100 * hi:.2f}]")
    print(f"  Δ(blend−v7) {100 * (hb.mean() - h0.mean()):+.2f}pt [{100 * d_lo:+.2f},{100 * d_hi:+.2f}]"
          f"  Δ(blend−市場) {100 * (hb.mean() - hm.mean()):+.2f}pt [{100 * m_lo:+.2f},{100 * m_hi:+.2f}]"
          f"  ◎1着 {100 * w0.mean():.2f}→{100 * wb.mean():.2f}% [{100 * w_lo:+.2f},{100 * w_hi:+.2f}]")
    by_year = {str(y): {"n": int(sum(r["year"] == y for r in test)),
                        "top3_v7": float(pick_hits([r for r in test if r["year"] == y], 0.0).mean()),
                        "top3_blend": float(pick_hits([r for r in test if r["year"] == y], lam_star).mean())} for y in (2024, 2025)}
    out = {"lambda": float(lam_star), "model": "v7", "fitted_on": "valid_2023_top3_of_top_pick",
           "odds_snapshot": "当日 9 時前後 (data/_joint/mkt_horses.parquet)",
           "valid_curve": {str(k): round(v, 5) for k, v in curve.items()},
           "test": {"n_races": n, "top3_v7": round(float(h0.mean()), 5), "top3_market": round(float(hm.mean()), 5),
                    "top3_blend": round(float(hb.mean()), 5),
                    "delta_blend_vs_v7_ci95": [round(d_lo, 5), round(d_hi, 5)],
                    "delta_blend_vs_market_ci95": [round(m_lo, 5), round(m_hi, 5)],
                    "win_v7": round(float(w0.mean()), 5), "win_blend": round(float(wb.mean()), 5),
                    "delta_win_ci95": [round(w_lo, 5), round(w_hi, 5)],
                    "top5_mean_hits_v7": round(top5_mean_hits(test, 0.0), 4), "top5_mean_hits_blend": round(top5_mean_hits(test, lam_star), 4),
                    "by_year": by_year}}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[out] {OUT}")


if __name__ == "__main__":
    main()
