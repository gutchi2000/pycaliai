# -*- coding: utf-8 -*-
"""
analysis/exotics_ev_market_test.py — exotics 実市場オッズでの EV 銘柄選抜 初検定
================================================================================
SettleAI Phase 1 (2026-07-31)。前提監査(2026-07-30)で確定した穴:
「EV銘柄選抜は有害」(単勝で -13pt, project_ev_selection_harmful_probfirst) は
exotics では curve 自己参照 EV (backtest_ev_grid) でしか検定されておらず、
**実市場オッズに対しては一度も検定されていない**。

ここで初めて埋める:
- ワイド: reports/live_odds の T-10 実オッズ (lo-hi mid) × bundle 埋込 calibrated
  PL joint 確率 (pair_probs) × data/kekka/wide_kekka.csv の実確定配当。
- 馬単: 2026 年の確定配当データが存在しない (kekka は単複のみ) ため**検定不能**。
  TARGET から馬単払戻をエクスポートしてもらえたら同枠組みで追検定する。

検定設計 (レースを標本単位、点推定でなく CI で判定):
  候補 = 印馬10ペアのうち T-10 ワイド実オッズがあるもの (本番と同じ土俵)。
  ルール比較 (同一候補集合・флат¥100/点):
    A. prob-first: p 上位 k 点          (現行本番の思想)
    B. EV-first:   settle_wide(o)×p 上位 k 点 (検定対象=EV選抜)
    C. EV gate:    A の選択のうち EV>=1.0 のみ (EV閾値ゲート)
  k=2,3。判定は ROI(B)−ROI(A) の race-level bootstrap CI95。
  A/B の選択が同一のレースでは差はゼロなので、divergence率も報告する
  (低ければ検出力が無い、という事実ごと報告する)。

実行: python -m analysis.exotics_ev_market_test [--base E:\\PyCaLiAI]
出力: reports/exotics_ev_market_test.json
禁止事項: この検定は「EV選抜の是非」の判定のみに使う。ドリフト方向での
銘柄選別・steam逆張りへの転用は封印済み (両方向エッジゼロ実証)。
"""
from __future__ import annotations
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
BASE = REPO

sys.path.insert(0, str(REPO))
from compute_bets import settle_wide  # noqa: E402

from analysis.measure_settle_drift import load_wide_kekka  # noqa: E402
import analysis.measure_settle_drift as _msd  # noqa: E402


def collect(base: Path):
    """レースごとの (候補ペア, 確率, T-10オッズ, 実配当) を組み立てる。"""
    _msd.BASE = base  # load_wide_kekka のデータ元
    wk = load_wide_kekka()
    bundles = {}
    races = []
    for f in sorted(glob.glob(str(base / "reports/live_odds/*.json"))):
        j = json.load(open(f, encoding="utf-8"))
        if not j.get("ok"):
            continue
        rid = str(j["race_id"])[:16]
        date = rid[:8]
        if date not in bundles:
            p = base / "reports" / "cowork_input" / f"{date}_bundle.json"
            if p.exists():
                b = json.load(open(p, encoding="utf-8"))
                bundles[date] = {str(r["race_id"])[:16]: r for r in b["races"]}
            else:
                bundles[date] = {}
        race = bundles[date].get(rid)
        if not race:
            continue
        pp = race.get("pair_probs") or {}
        wide = j.get("wide") or {}
        pays = wk.get((date, rid[8:10], int(rid[14:16])))
        if pays is None:
            continue  # 結果未取得レースは除外 (的中0と混同しない)
        cands = []
        for key, d in pp.items():
            p_wide = d.get("wide")
            lohi = wide.get(key)
            if not p_wide or not lohi:
                continue
            mid = (lohi[0] + lohi[1]) / 2
            if mid <= 1.0:
                continue
            i, jn = map(int, key.split("-"))
            pay = pays.get((min(i, jn), max(i, jn)), 0.0)  # 倍率 (100円あたり/100)
            cands.append({"pair": key, "p": float(p_wide), "o": float(mid),
                          "ev": settle_wide(float(mid)) * float(p_wide),
                          "ev_raw": float(mid) * float(p_wide),
                          "pay": float(pay)})
        if len(cands) >= 3:
            races.append(cands)
    return races


def simulate(races, k, ev_key="ev"):
    """レースごとに A(prob top-k) / B(EV top-k) / C(A∩EV>=1) を賭ける。
    returns: per-race (stakeA, retA, stakeB, retB, stakeC, retC, diverged)"""
    out = []
    for cands in races:
        if len(cands) < k:
            continue
        by_p = sorted(cands, key=lambda c: -c["p"])[:k]
        by_ev = sorted(cands, key=lambda c: -c[ev_key])[:k]
        gate = [c for c in by_p if c[ev_key] >= 1.0]
        row = {
            "sA": 100 * len(by_p), "rA": 100 * sum(c["pay"] for c in by_p),
            "sB": 100 * len(by_ev), "rB": 100 * sum(c["pay"] for c in by_ev),
            "sC": 100 * len(gate), "rC": 100 * sum(c["pay"] for c in gate),
            "div": int({c["pair"] for c in by_p} != {c["pair"] for c in by_ev}),
        }
        out.append(row)
    return out


def summarize(rows, label, nboot=3000, seed=42):
    rng = np.random.default_rng(seed)
    arr = {k: np.array([r[k] for r in rows], dtype=float)
           for k in ("sA", "rA", "sB", "rB", "sC", "rC", "div")}
    n = len(rows)

    def roi(ret, stake, idx=None):
        if idx is not None:
            ret, stake = ret[idx], stake[idx]
        s = stake.sum()
        return float(ret.sum() / s) if s > 0 else float("nan")

    res = {"n_races": n, "divergence_rate": round(float(arr["div"].mean()), 3)}
    for tag, (rk, sk) in {"A_prob": ("rA", "sA"), "B_ev": ("rB", "sB"),
                          "C_evgate": ("rC", "sC")}.items():
        boot = [roi(arr[rk], arr[sk], rng.integers(0, n, n)) for _ in range(nboot)]
        boot = [b for b in boot if not np.isnan(b)]
        res[tag] = {"roi": round(roi(arr[rk], arr[sk]), 4),
                    "stake": int(arr[sk].sum()),
                    "roi_ci95": [round(float(np.percentile(boot, 2.5)), 4),
                                 round(float(np.percentile(boot, 97.5)), 4)]}
    # paired diff (同一レース resample): B−A と C−A
    for tag, rk, sk in (("diff_B_minus_A", "rB", "sB"), ("diff_C_minus_A", "rC", "sC")):
        dboot = []
        for _ in range(nboot):
            idx = rng.integers(0, n, n)
            a, b = roi(arr["rA"], arr["sA"], idx), roi(arr[rk], arr[sk], idx)
            if not (np.isnan(a) or np.isnan(b)):
                dboot.append(b - a)
        base_roi = res["B_ev" if rk == "rB" else "C_evgate"]["roi"]
        res[tag] = {"point": round(base_roi - res["A_prob"]["roi"], 4),
                    "ci95": [round(float(np.percentile(dboot, 2.5)), 4),
                             round(float(np.percentile(dboot, 97.5)), 4)]}
    print(f"[{label}] n={n} div={res['divergence_rate']*100:.0f}%  "
          f"A(prob)={res['A_prob']['roi']*100:.1f}% CI{res['A_prob']['roi_ci95']}  "
          f"B(EV)={res['B_ev']['roi']*100:.1f}% CI{res['B_ev']['roi_ci95']}  "
          f"C(gate)={res['C_evgate']['roi']*100:.1f}% (stake {res['C_evgate']['stake']})  "
          f"B-A={res['diff_B_minus_A']['point']*100:+.1f}pt CI{res['diff_B_minus_A']['ci95']}  "
          f"C-A={res['diff_C_minus_A']['point']*100:+.1f}pt CI{res['diff_C_minus_A']['ci95']}")
    return res


def main(base: Path):
    races = collect(base)
    print(f"検定対象レース: {len(races)} (live_odds×bundle×wide_kekka 三重join)")
    out = {"n_races": len(races), "note": "wide のみ。馬単は2026確定配当データ無しで検定不能",
           "results": {}}
    for k in (2, 3):
        out["results"][f"wide_k{k}"] = summarize(simulate(races, k), f"wide k={k}")
    # settle補正なしの素EVランキングでも同じか (頑健性)
    out["results"]["wide_k2_rawev"] = summarize(
        simulate(races, 2, ev_key="ev_raw"), "wide k=2 rawEV")
    # 本番同等 ODDS_CAP=50 (compute_bets): EV-first の大穴依存を遮断した現実運用土俵
    capped = [[c for c in cs if c["o"] <= 50.0] for cs in races]
    capped = [cs for cs in capped if len(cs) >= 3]
    for k in (2, 3):
        out["results"][f"wide_k{k}_cap50"] = summarize(
            simulate(capped, k), f"wide k={k} cap50")
    op = REPO / "reports" / "exotics_ev_market_test.json"
    op.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(op, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[saved] {op}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=None)
    a = ap.parse_args()
    main(Path(a.base) if a.base else BASE)
