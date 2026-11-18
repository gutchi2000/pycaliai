"""目的関数の検証 (評価): ランク学習の勝率が市場より劣るのは目的関数のせいか (分析専用)。

事前に固定した設計 (後半の結果を見てから変えない):
  - 特徴: analysis.objective_check_replay が凍結 bundle の各開催日を現在の配信経路 (v6) で作り直して
      保存した X (v6 と同じエンコード・-9999 補完)。v6 のスコアも同じ X から。
  - 比べる確率 (すべて採点対象の馬の集合でレース内正規化):
      現行 (再生)   : 再生 bundle の p_win (v6 スコア → PL → 配信用 isotonic 較正器)。主な比較基準。
      現行 (凍結版) : 公開時点の凍結 bundle の p_win (世代混合、参考)。
      温度          : softmax(v6 スコア / T)。T は前半 (〜08-15) の勝ち馬 NLL 最小で fit (0.2〜5)。
      M1            : log-loss (binary) で学習したモデル (analysis.objective_check_train)。主な新モデル。
      M2            : レース内 softmax (条件付きロジット) で学習したモデル。副。
      市場 (最終)   : 最終単勝オッズの逆数の正規化。
      市場 (T-10)   : T-10 単勝オッズの逆数の正規化 (analysis.t10_market_blend_check.t10_odds と同じ取り方)。
  - 採点対象の馬: 前回までと同じ (最終単勝オッズあり・中止等でない・凍結 p_win あり) で、再生の X があり、
      T-10 オッズがある馬。1 頭でも欠けるレースは外す。
  - 分割: 前半 = 〜2026-08-15 (温度の fit のみ)、後半 = 2026-08-16〜 (評価、1 回だけ)。
  - 指標: 馬ごとの 2 値 LogLoss (確率は [0.001, 0.999] に制限)、Brier、ECE、勝ち馬の対数尤度。
      差の CI はレース単位の対応付きブートストラップ (4,000 回)。比較: 各確率 − 市場 (最終)、
      − 市場 (T-10)、− 現行 (再生)。
  - 補助 (オフライン): test 2024-25 で M1・M2・温度 (valid 2023 で fit) と v6 (PL + isotonic) の差。

使い方: python -m analysis.objective_check_eval --work reports/objective_check --out-json <path>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from analysis import ev_threshold_backtest as bt
from analysis.ev_recalibration_check import metrics, paired_metric_diff
from analysis.market_blend_check import paired_winner_nll
from analysis.t10_market_blend_check import post_times, t10_odds

SPLIT = "20260815"
ROOT = Path("E:/PyCaLiAI")
N_BOOT = 4000
RNG = np.random.default_rng(20261119)


def norm(df, col):
    return df[col] / df.groupby("rid")[col].transform("sum")


def softmax_by(df, col, T=1.0):
    z = df[col] / T
    z = z - z.groupby(df["rid"]).transform("max")
    e = np.exp(z)
    return e / e.groupby(df["rid"]).transform("sum")


def winner_nll_df(df, col, floor=1e-12):
    w = df[df["win"] == 1]
    s = pd.Series(-np.log(np.clip(w[col].to_numpy(), floor, None)), index=w.index).groupby(w["rid"]).mean()
    return float(s.mean())


def winner_tail(df, col) -> dict:
    """勝ち馬に付いた確率の下端 (記述用)。"""
    w = df[df["win"] == 1][col].to_numpy()
    return {"n_winners": int(len(w)), "n_below_0.001": int((w < 0.001).sum()), "n_below_0.005": int((w < 0.005).sum()),
            "min": float(w.min())}


def paired_winner_nll_raw(df, a, b) -> dict:
    """下限処理なし (1e-12) の勝ち馬 NLL の差 (a − b)、レース単位ブートストラップ。"""
    w = df[df["win"] == 1]
    da = pd.Series(-np.log(np.clip(w[a].to_numpy(), 1e-12, None)), index=w.index).groupby(w["rid"]).mean()
    db = pd.Series(-np.log(np.clip(w[b].to_numpy(), 1e-12, None)), index=w.index).groupby(w["rid"]).mean()
    d = (da - db).to_numpy()
    idx = RNG.integers(0, len(d), size=(N_BOOT, len(d)))
    return {"diff": float(d.mean()), "ci95": [float(np.percentile(d[idx].mean(1), 2.5)), float(np.percentile(d[idx].mean(1), 97.5))]}


def load_replay(work: Path) -> pd.DataFrame:
    feats = json.loads((work / "feats" / "_feature_cols.json").read_text(encoding="utf-8"))
    frames = []
    for p in sorted((work / "feats").glob("2026????.parquet")):
        d = pd.read_parquet(p)
        d["date"] = p.stem
        b = work / "bundles" / f"{p.stem}_bundle.json"
        pw = {}
        if b.exists():
            for r in json.loads(b.read_text(encoding="utf-8"))["races"]:
                for h in r["horses"]:
                    pw[(str(r["race_id"])[:16], int(h["umaban"]))] = h.get("p_win")
        d["rid"] = d["rid"].astype(str).str[:16]
        d["p_replay"] = [pw.get((r, int(u))) for r, u in zip(d["rid"], d["umaban"])]
        frames.append(d)
    return pd.concat(frames, ignore_index=True), feats


def boot_pair(a: np.ndarray, b: np.ndarray, rid: np.ndarray) -> dict:
    d = a - b
    uniq, inv = np.unique(rid, return_inverse=True)
    s, c = np.bincount(inv, weights=d), np.bincount(inv)
    idx = RNG.integers(0, len(uniq), size=(N_BOOT, len(uniq)))
    boots = s[idx].sum(1) / c[idx].sum(1)
    return {"diff": float(d.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}


def offline_supplement(work: Path) -> dict:
    z = np.load(work / "offline_test_probs.npz", allow_pickle=True)
    y, starts, counts, rid = z["y"], z["starts"], z["counts"], z["rid"]
    ysum = np.repeat(np.add.reduceat(y, starts), counts)
    out = {}
    for k in ("m1", "m2", "v6_T"):
        res = {}
        for name, f in (("binary_logloss", lambda p: -(y * np.log(np.clip(p, 1e-3, 1 - 1e-3)) + (1 - y) * np.log(1 - np.clip(p, 1e-3, 1 - 1e-3)))),
                        ("brier", lambda p: (p - y) ** 2)):
            res[name] = boot_pair(f(z[k]), f(z["v6_pl_iso"]), rid)
        # 勝ち馬の対数尤度 (レース単位)
        wl = lambda p: np.where(y > 0, -y * np.log(np.clip(p, 1e-12, None)) / np.where(ysum > 0, ysum, 1), 0.0)  # noqa: E731
        ra, rb = np.add.reduceat(wl(z[k]), starts), np.add.reduceat(wl(z["v6_pl_iso"]), starts)
        ok = np.add.reduceat(y, starts) > 0
        dd = (ra - rb)[ok]
        idx = RNG.integers(0, len(dd), size=(N_BOOT, len(dd)))
        res["winner_nll"] = {"diff": float(dd.mean()), "ci95": [float(np.percentile(dd[idx].mean(1), 2.5)),
                                                               float(np.percentile(dd[idx].mean(1), 97.5))]}
        out[f"{k}-v6_pl_iso"] = res
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="reports/objective_check")
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    work = Path(a.work).resolve()

    rows, info = bt.load_rows(ROOT)
    fz = pd.DataFrame(rows)
    fz = fz[fz["o_fin"].notna() & ~fz["no_finish"] & fz["p_win"].notna()].copy()
    rp, feats = load_replay(work)

    # 再生側の全馬でモデル確率を出してから、採点対象に絞る
    X = rp[[f"f{i}" for i in range(len(feats))]].to_numpy()
    m1 = lgb.Booster(model_file=str(work / "models" / "m1.txt"))
    m2 = lgb.Booster(model_file=str(work / "models" / "m2.txt"))
    rp["m1_raw"] = m1.predict(X)
    rp["m2_score"] = m2.predict(X)
    keep_cols = ["rid", "umaban", "score", "m1_raw", "m2_score", "p_replay"]
    df = fz.merge(rp[keep_cols], left_on=["rid", "uma"], right_on=["rid", "umaban"], how="left")
    miss = df.groupby("rid")[["score", "p_replay"]].apply(lambda g: g.isna().any().any())
    stat = {"frozen_races": int(fz["rid"].nunique()), "races_missing_replay": int(miss.sum())}

    posts = post_times(ROOT, sorted(df["date"].unique()))
    odds, src, st10 = t10_odds(ROOT, sorted(df["rid"].unique()), posts)
    df["o10"] = [odds.get(r, {}).get(u) for r, u in zip(df["rid"], df["uma"])]
    miss10 = df.groupby("rid")["o10"].apply(lambda s: s.isna().any())
    bad = set(miss[miss].index) | set(miss10[miss10].index)
    stat["races_missing_t10"] = int(len(set(miss10[miss10].index) - set(miss[miss].index)))
    df = df[~df["rid"].isin(bad)].copy()
    df["win"] = (df["tan_ret"] > 0).astype(int)

    df["p_cur_replay"] = norm(df, "p_replay")
    df["p_cur_frozen"] = norm(df, "p_win")
    df["p_m1"] = norm(df, "m1_raw")
    df["p_m2"] = softmax_by(df, "m2_score")
    inv = 1.0 / df["o_fin"]
    df["q_fin"] = inv / inv.groupby(df["rid"]).transform("sum")
    inv = 1.0 / df["o10"]
    df["q_t10"] = inv / inv.groupby(df["rid"]).transform("sum")

    fit, val = df[df["date"] <= SPLIT].copy(), df[df["date"] > SPLIT].copy()
    T = float(minimize_scalar(lambda t: winner_nll_df(fit.assign(_p=softmax_by(fit, "score", t)), "_p"),
                              bounds=(0.2, 5.0), method="bounded").x)
    for d in (fit, val):
        d["p_T"] = softmax_by(d, "score", T)
    print("T =", T, "fit races", fit["rid"].nunique(), "val races", val["rid"].nunique(), json.dumps(stat), flush=True)

    cols = ["p_cur_replay", "p_cur_frozen", "p_T", "p_m1", "p_m2", "q_fin", "q_t10"]
    out = {"design": {"split": SPLIT, "T_bounds": [0.2, 5.0], "T": T}, "stat": stat,
           "coverage": {h: {"races": int(d["rid"].nunique()), "horses": int(len(d)),
                            "dates": [d["date"].min(), d["date"].max(), int(d["date"].nunique())],
                            "eras": d.drop_duplicates("rid")["era"].value_counts().to_dict()}
                        for h, d in (("fit", fit), ("val", val))}}
    out["fit_reference"] = {c: {k: v for k, v in metrics(fit, c, "win").items() if k != "bands"} | {"winner_nll": winner_nll_df(fit, c)} for c in cols}
    # ---- 後半 (1 回だけ) ----
    # winner_nll = 確率の下限 0.001 (差の表 paired_winner_nll と同じ)、winner_nll_raw = 下限なし
    out["val"] = {c: metrics(val, c, "win") | {"winner_nll": winner_nll_df(val, c, 0.001),
                                                "winner_nll_raw": winner_nll_df(val, c), "winner_tail": winner_tail(val, c)}
                  for c in cols}
    out["val_diff"] = {}
    for base in ("q_fin", "q_t10", "p_cur_replay"):
        for c in cols:
            if c == base:
                continue
            out["val_diff"][f"{c}-{base}"] = paired_metric_diff(val, c, base, "win") | {
                "winner_nll": paired_winner_nll(val, c, base), "winner_nll_raw": paired_winner_nll_raw(val, c, base)}
    out["offline_test2024_25"] = offline_supplement(work)
    Path(a.out_json).write_text(json.dumps(out, ensure_ascii=False, default=str), encoding="utf-8")
    for c in cols:
        m = out["val"][c]
        print(f"val {c:14s} LL={m['logloss']:.5f} Brier={m['brier']:.5f} ECE={m['ece']:.4f} wNLL={m['winner_nll']:.4f} "
              f"wNLLraw={m['winner_nll_raw']:.4f} tail={m['winner_tail']}")
    for k, v in out["val_diff"].items():
        print(f"{k:28s} LL {v['logloss']['diff']:+.5f} [{v['logloss']['ci95'][0]:+.5f},{v['logloss']['ci95'][1]:+.5f}] "
              f"Brier {v['brier']['diff']:+.6f} [{v['brier']['ci95'][0]:+.6f},{v['brier']['ci95'][1]:+.6f}] "
              f"wNLL {v['winner_nll']['diff']:+.4f} [{v['winner_nll']['ci95'][0]:+.4f},{v['winner_nll']['ci95'][1]:+.4f}] "
              f"raw {v['winner_nll_raw']['diff']:+.4f} [{v['winner_nll_raw']['ci95'][0]:+.4f},{v['winner_nll_raw']['ci95'][1]:+.4f}]")
    print(json.dumps(out["offline_test2024_25"]))


if __name__ == "__main__":
    main()
