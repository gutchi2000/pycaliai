"""勝率の再キャリブレーションで EV ルールが生き返るかの検証 (分析専用、本番経路には不干渉)。

事前に固定した設計 (検証側の結果を見てから変えない):
  - データ: analysis.ev_threshold_backtest.load_rows と同じ (凍結 bundle × kekka、56 開催日)。
  - 分割: 開催日を日付順に並べ、前半 28 日 = fit (2026-05-02〜08-15)、後半 28 日 = 検証 (08-16〜11-15)。
  - 単勝: 前半の (p_win, 1 着か) に isotonic 回帰 (単調増加、出力は [0.001, 0.999] に制限、範囲外は端の値)。
      主 (A): isotonic の出力をそのまま使う。
      副 (B): A をレースごとに合計 1 へ正規化。
  - 複勝: 前半の (p_sho, 複勝が払い戻されたか) に同じ isotonic。7 頭以下は 2 着までなので
      的中の定義は「複勝払戻がある」。正規化はしない。
  - 市場: 単勝 = 最終単勝オッズの逆数をレースごとに正規化。複勝 = その市場勝率から Harville で
      払戻対象 (8 頭以上 3 着内 / 7 頭以下 2 着内) に入る確率。
  - 評価 (検証側のみ): 確率帯別の予測と実績の差、ECE、Brier、LogLoss (確率は全予測器とも
      [0.001, 0.999] に制限して計算)。差の CI はレース単位の対応付きブートストラップ。
  - EV ルール (検証側のみ): 閾値 1.0/1.1/1.15/1.2、単勝の全該当と上位 1 頭、複勝の全該当と上位 1 頭。
      オッズは前回と同じ (単勝 = 最終、複勝 = export 時点の下限)。

使い方:
  python -m analysis.ev_recalibration_check --data-root E:/PyCaLiAI --out-json <path>
"""
from __future__ import annotations

import argparse
import json
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from analysis import ev_threshold_backtest as bt

THRESHOLDS = bt.THRESHOLDS
N_FIT_DATES = 28
P_LO, P_HI = 0.001, 0.999
BANDS = [0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0001]
N_BOOT = 4000
RNG = np.random.default_rng(20261118)


def harville_in_top(q: np.ndarray, k: int) -> np.ndarray:
    """市場勝率 q から各馬が k 着以内に入る確率 (Harville)。k は 2 か 3。"""
    n = len(q)
    out = q.copy()
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            p2 = q[j] * q[i] / (1 - q[j])            # j 1 着, i 2 着
            out[i] += p2
            if k >= 3:
                for m in range(n):
                    if m in (i, j):
                        continue
                    out[i] += q[m] * q[j] / (1 - q[m]) * q[i] / (1 - q[m] - q[j])   # m, j, i
    return out


def add_market(df: pd.DataFrame) -> pd.DataFrame:
    inv = 1.0 / df["o_fin"]
    df["q_win"] = inv / inv.groupby(df["rid"]).transform("sum")
    df["n_start"] = df.groupby("rid")["uma"].transform("size")
    q_plc = np.zeros(len(df))
    for rid, idx in df.groupby("rid").indices.items():
        q = df["q_win"].to_numpy()[idx]
        k = 3 if df["n_start"].to_numpy()[idx[0]] >= 8 else 2
        q_plc[idx] = harville_in_top(q, k)
    df["q_plc"] = q_plc
    return df


def clip(p):
    return np.clip(np.asarray(p, float), P_LO, P_HI)


def metrics(df: pd.DataFrame, pcol: str, ycol: str) -> dict:
    p, y = clip(df[pcol]), df[ycol].to_numpy().astype(float)
    brier = (p - y) ** 2
    ll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    bands = []
    ece = 0.0
    cut = pd.cut(df[pcol], BANDS, right=False)
    for b, g in df.groupby(cut, observed=True):
        pp = clip(g[pcol]).mean()
        aa = g[ycol].mean()
        bands.append({"band": f"{b.left:.2f}-{min(b.right, 1):.2f}", "n": int(len(g)),
                      "pred": float(pp), "actual": float(aa), "diff": float(pp - aa)})
        ece += len(g) / len(df) * abs(pp - aa)
    return {"brier": float(brier.mean()), "logloss": float(ll.mean()), "ece": float(ece), "bands": bands}


def paired_metric_diff(df: pd.DataFrame, a: str, b: str, ycol: str) -> dict:
    """a − b の Brier / LogLoss 差 (負 = a の方が良い)。レース単位ブートストラップ。"""
    y = df[ycol].to_numpy().astype(float)
    out = {}
    for name, f in (("brier", lambda p: (p - y) ** 2),
                    ("logloss", lambda p: -(y * np.log(p) + (1 - y) * np.log(1 - p)))):
        d = f(clip(df[a])) - f(clip(df[b]))
        uniq, inv = np.unique(df["rid"].to_numpy(), return_inverse=True)
        s = np.bincount(inv, weights=d)
        c = np.bincount(inv)
        idx = RNG.integers(0, len(uniq), size=(N_BOOT, len(uniq)))
        boots = s[idx].sum(1) / c[idx].sum(1)
        out[name] = {"diff": float(d.mean()), "ci95": [float(np.percentile(boots, 2.5)),
                                                        float(np.percentile(boots, 97.5))],
                     "p_a_better": float((boots < 0).mean())}
    return out


def odds_band_table(df: pd.DataFrame, cols: list[str], ycol: str) -> list[dict]:
    cut = pd.cut(df["o_fin"], [0, 2, 3, 5, 10, 20, 50, 100, 1e9])
    rows = []
    for b, g in df.groupby(cut, observed=True):
        r = {"odds_band": f"{b.left:g}-{b.right:g}", "n": int(len(g)), "actual": float(g[ycol].mean())}
        for c in cols:
            r[c] = float(clip(g[c]).mean())
        rows.append(r)
    return rows


def ev_block(df: pd.DataFrame, pw: str, ps: str | None) -> dict:
    recs = df.to_dict("records")
    for r in recs:
        r["p_win"] = r[pw]
        r["p_sho"] = r[ps] if ps else None
    out = {}
    kinds = ["tan_all", "tan_top1"] + (["fuku_all", "fuku_top1"] if ps else [])
    for k in kinds:
        for t in THRESHOLDS:
            out[f"{k}@{t}"] = bt.summarize(bt.select(recs, k, t))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="E:/PyCaLiAI")
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    rows, info = bt.load_rows(Path(a.data_root))
    df = pd.DataFrame(rows)
    df = df[df["o_fin"].notna() & ~df["no_finish"]].copy()
    df["win"] = (df["tan_ret"] > 0).astype(int)
    df["plc"] = (df["fuku_ret"] > 0).astype(int)
    df = add_market(df)

    dates = sorted(df["date"].unique())
    fit_dates, val_dates = dates[:N_FIT_DATES], dates[N_FIT_DATES:]
    fit, val = df[df["date"].isin(fit_dates)].copy(), df[df["date"].isin(val_dates)].copy()

    iso_w = IsotonicRegression(y_min=P_LO, y_max=P_HI, increasing=True, out_of_bounds="clip")
    iso_w.fit(fit["p_win"], fit["win"])
    fit_p = fit[fit["p_sho"].notna()]
    iso_p = IsotonicRegression(y_min=P_LO, y_max=P_HI, increasing=True, out_of_bounds="clip")
    iso_p.fit(fit_p["p_sho"], fit_p["plc"])

    for d in (fit, val):
        d["pw_A"] = iso_w.predict(d["p_win"])
        d["pw_B"] = d["pw_A"] / d.groupby("rid")["pw_A"].transform("sum")
        d["ps_A"] = np.where(d["p_sho"].notna(), iso_p.predict(d["p_sho"].fillna(0)), np.nan)

    vp = val[val["p_sho"].notna()]
    out = {
        "design": {"fit_dates": [fit_dates[0], fit_dates[-1], len(fit_dates)],
                   "val_dates": [val_dates[0], val_dates[-1], len(val_dates)],
                   "p_clip": [P_LO, P_HI]},
        "info": {"fit_races": int(fit["rid"].nunique()), "val_races": int(val["rid"].nunique()),
                 "fit_horses": int(len(fit)), "val_horses": int(len(val)),
                 "fit_eras": fit.drop_duplicates("rid")["era"].value_counts().to_dict(),
                 "val_eras": val.drop_duplicates("rid")["era"].value_counts().to_dict(),
                 "dates_excluded": info["dates_no_frozen"]},
        "iso_win_knots": {"x": [float(x) for x in iso_w.X_thresholds_], "y": [float(y) for y in iso_w.y_thresholds_]},
        "iso_plc_knots": {"x": [float(x) for x in iso_p.X_thresholds_], "y": [float(y) for y in iso_p.y_thresholds_]},
        "val_win": {c: metrics(val, c, "win") for c in ("p_win", "pw_A", "pw_B", "q_win")},
        "val_plc": {c: metrics(vp, c, "plc") for c in ("p_sho", "ps_A", "q_plc")},
        "val_win_diff": {f"{a}-q_win": paired_metric_diff(val, a, "q_win", "win") for a in ("p_win", "pw_A", "pw_B")},
        "val_plc_diff": {f"{a}-q_plc": paired_metric_diff(vp, a, "q_plc", "plc") for a in ("p_sho", "ps_A")},
        "val_win_by_odds": odds_band_table(val, ["p_win", "pw_A", "pw_B", "q_win"], "win"),
        "fit_insample_win": {c: {k: v for k, v in metrics(fit, c, "win").items() if k != "bands"}
                             for c in ("p_win", "pw_A", "q_win")},
        "val_ev": {
            "raw": ev_block(val, "p_win", "p_sho"),
            "recal_A": ev_block(val, "pw_A", "ps_A"),
            "recal_B": ev_block(val, "pw_B", None),
        },
        "val_baseline": {
            "tan_all_horses": bt.summarize([{**r, "ret": r["tan_ret"]} for r in val.to_dict("records")]),
            "fuku_all_horses": bt.summarize([{**r, "ret": r["fuku_ret"]} for r in vp.to_dict("records")
                                             if r["fo_low"] is not None and not pd.isna(r["fo_low"])]),
        },
    }
    Path(a.out_json).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out["design"]), json.dumps(out["info"], ensure_ascii=False))
    for k, v in out["val_win"].items():
        print("win", k, {x: round(v[x], 5) for x in ("brier", "logloss", "ece")})
    for k, v in out["val_plc"].items():
        print("plc", k, {x: round(v[x], 5) for x in ("brier", "logloss", "ece")})
    for name, blk in out["val_ev"].items():
        for key, s in blk.items():
            if s.get("n_bets"):
                print(f"{name:8s} {key:14s} bets={s['n_bets']:5d} races={s['n_races']:4d} hit={s['hit_rate']*100:5.1f}% "
                      f"roi={s['roi']*100:6.1f}% CI[{s['roi_ci95'][0]*100:.0f},{s['roi_ci95'][1]*100:.0f}]")
            else:
                print(f"{name:8s} {key:14s} bets=0")


if __name__ == "__main__":
    main()
