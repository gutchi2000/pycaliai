"""
fit_t10_blend.py
================
T-10 補正印のための blend 係数 λ を fit する。

補正印 = レース内で  u_i = log(p_win_i) + λ·log(π_i)  の降順に付け直した印。
  p_win = v6 較正 PL 単勝確率 (bundle の p_win と同じ serve calibrator)
  π     = de-vig 市場単勝確率 (T-10 ライブオッズから計算。fit は 9時スナップ)

目的は「精度」(◎top3) であり ROI ではない。順位だけが欲しいので
blend は 1 パラメータ λ (= 市場の混合比 β/α) に落ちる。

先行知見 (stage1_benter_blend.py):
  - ≤2023 fit の (α,β) MLE は OOS 尤度で market-only に負けた (α 過大)
  - 9時 arm と確定 arm の係数はほぼ同じ → 9時 fit の λ は T-10 に転移可
→ 本スクリプトは尤度でなく ◎top3 を直接 valid(2023) で最適化し、
  test(2024-25) で paired bootstrap CI 付き OOS 評価する。

データ (全てキャッシュ済み、515MB master 不要):
  data/_ev_grid_scores.parquet   v6 生スコア (2023-2025, rid+馬番)
  data/_joint/mkt_horses.parquet 9時 de-vig π + top3 ラベル (rid16+ban)

出力: data/t10_blend.json  (λ, valid/test 指標, ベースライン)
実行: venv311/Scripts/python.exe -m analysis.fit_t10_blend
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
import pl_probs as PL  # noqa: E402

np.random.seed(42)

LAMBDA_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.25, 1.5,
               2.0, 2.5, 3.0, 4.0, 6.0, 10.0, np.inf]
N_BOOT = 2000


def load_races() -> list[dict]:
    """v6 スコア × 9時市場 π × top3 ラベルを rid16+馬番で JOIN したレース群。"""
    sc = pd.read_parquet(BASE / "data/_ev_grid_scores.parquet")
    mk = pd.read_parquet(BASE / "data/_joint/mkt_horses.parquet",
                         columns=["rid", "ban", "pi", "top3"])
    sc["rid16"] = sc["レースID(新/馬番無)"].astype(str).str[:16]
    mk = mk.rename(columns={"rid": "rid16"})
    mk["rid16"] = mk["rid16"].astype(str)

    cal = joblib.load(BASE / "models/pl_calibrators_v6_serve.pkl")
    cal = cal.get("calibrators", cal)
    iso_tan = cal["tansho"]

    df = sc.merge(mk, left_on=["rid16", "馬番"], right_on=["rid16", "ban"],
                  how="inner")
    print(f"[join] score rows={len(sc):,}  mkt rows={len(mk):,}  "
          f"joined={len(df):,}")

    races = []
    for (rid16, year), g in df.groupby(["rid16", "year"], sort=False):
        if len(g) < 5 or (g["pi"] <= 0).any():
            continue
        g = g.sort_values("馬番")
        w = PL.pl_weights(g["_score"].values)
        f = iso_tan.predict(PL.all_tansho(w))
        f = np.maximum(f, 1e-9)
        races.append({
            "year": int(year),
            "logf": np.log(f),
            "logpi": np.log(g["pi"].values),
            "top3": g["top3"].values.astype(int),
        })
    print(f"[races] {len(races):,} レース "
          f"(2023={sum(1 for r in races if r['year'] == 2023):,} / "
          f"2024={sum(1 for r in races if r['year'] == 2024):,} / "
          f"2025={sum(1 for r in races if r['year'] == 2025):,})")
    return races


def pick_hits(races: list[dict], lam: float) -> np.ndarray:
    """各レースで blend 順1位 (=補正◎) を選び、その馬の top3 ラベルを返す。"""
    hits = np.empty(len(races), dtype=int)
    for i, r in enumerate(races):
        u = r["logpi"] if np.isinf(lam) else r["logf"] + lam * r["logpi"]
        hits[i] = r["top3"][int(np.argmax(u))]
    return hits


def top5_mean_hits(races: list[dict], lam: float) -> float:
    """印5頭 (blend 上位5) のうち3着内に来た頭数の平均。"""
    tot = 0.0
    for r in races:
        u = r["logpi"] if np.isinf(lam) else r["logf"] + lam * r["logpi"]
        idx = np.argsort(-u)[:5]
        tot += r["top3"][idx].sum()
    return tot / len(races)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def paired_boot_delta(h_a: np.ndarray, h_b: np.ndarray,
                      n_boot: int = N_BOOT) -> tuple[float, float]:
    """Δ = mean(h_a) - mean(h_b) の paired bootstrap 95% CI。"""
    n = len(h_a)
    rng = np.random.default_rng(42)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas[b] = h_a[idx].mean() - h_b[idx].mean()
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def main() -> None:
    races = load_races()
    valid = [r for r in races if r["year"] == 2023]
    test = [r for r in races if r["year"] >= 2024]

    # --- valid 2023 で λ 選択 (◎top3 最大化) ---
    print("\n[valid 2023] λ sweep (◎top3):")
    curve = {}
    for lam in LAMBDA_GRID:
        rate = pick_hits(valid, lam).mean()
        curve[lam] = rate
        tag = {0.0: " (v6のみ)", np.inf: " (市場のみ)"}.get(lam, "")
        print(f"  λ={lam:>5}: {rate * 100:.2f}%{tag}")
    finite = {k: v for k, v in curve.items() if np.isfinite(k)}
    lam_star = max(finite, key=finite.get)
    print(f"  → λ* = {lam_star} (valid ◎top3 {finite[lam_star] * 100:.2f}%)")

    # --- test 2024-25 で OOS 評価 ---
    h_v6 = pick_hits(test, 0.0)
    h_mkt = pick_hits(test, np.inf)
    h_bl = pick_hits(test, lam_star)
    n = len(test)
    print(f"\n[test 2024-25] n={n:,} レース (◎top3):")
    rows = [("v6のみ (λ=0)", h_v6), ("市場のみ (λ=∞)", h_mkt),
            (f"blend (λ={lam_star})", h_bl)]
    for name, h in rows:
        lo, hi = wilson_ci(int(h.sum()), n)
        print(f"  {name:<16}: {h.mean() * 100:.2f}%  "
              f"CI95 [{lo * 100:.2f}, {hi * 100:.2f}]")
    d_lo, d_hi = paired_boot_delta(h_bl, h_v6)
    print(f"  Δ(blend−v6)  = {(h_bl.mean() - h_v6.mean()) * 100:+.2f}pt  "
          f"CI95 [{d_lo * 100:+.2f}, {d_hi * 100:+.2f}]")
    dm_lo, dm_hi = paired_boot_delta(h_bl, h_mkt)
    print(f"  Δ(blend−市場) = {(h_bl.mean() - h_mkt.mean()) * 100:+.2f}pt  "
          f"CI95 [{dm_lo * 100:+.2f}, {dm_hi * 100:+.2f}]")

    t5_v6 = top5_mean_hits(test, 0.0)
    t5_bl = top5_mean_hits(test, lam_star)
    print(f"  印5頭の3着内平均: v6 {t5_v6:.3f} → blend {t5_bl:.3f} 頭")

    out = {
        "lambda": float(lam_star),
        "fitted_on": "valid_2023_top3_of_top_pick",
        "odds_snapshot": "9am (stage1で確定armと係数ほぼ同一を確認済み→T-10転移可)",
        "valid_curve": {str(k): round(v, 5) for k, v in curve.items()},
        "test": {
            "n_races": n,
            "top3_v6": round(float(h_v6.mean()), 5),
            "top3_market": round(float(h_mkt.mean()), 5),
            "top3_blend": round(float(h_bl.mean()), 5),
            "delta_blend_vs_v6_ci95": [round(d_lo, 5), round(d_hi, 5)],
            "delta_blend_vs_market_ci95": [round(dm_lo, 5), round(dm_hi, 5)],
            "top5_mean_hits_v6": round(t5_v6, 4),
            "top5_mean_hits_blend": round(t5_bl, 4),
        },
    }
    path = BASE / "data/t10_blend.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(f"\n[out] {path}")


if __name__ == "__main__":
    main()
