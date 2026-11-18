"""市場を土台に、モデルの修正だけを学習する (残差学習) の検証 (分析専用、探索扱い)。

形: logit(p) = logit(q_T10) + f(x)。f は前半 (〜2026-08-15) だけで学習・調整し、後半 (08-16〜) で 1 回だけ評価する。

事前に固定した設計 (後半の結果を見てから変えない):
  - 土台 q_T10: T-10 単勝オッズ (発走 5〜15 分前、analysis.t10_market_blend_check.t10_odds と同じ) の逆数を
      レース内 (採点対象の馬) で正規化。最終オッズは土台にも特徴にも使わない (最終オッズは評価と EV の払戻計算だけ)。
  - 採点対象: analysis.objective_check_eval と同じ (最終オッズあり・中止等でない・凍結 p_win あり・再生 X あり・
      T-10 オッズあり。欠けるレースは外す) に加え、公開サイトの日別 JSON にそのレースがあること。
  - 特徴 x (配信時点で持っているものだけ。人気・妙味・vs_market などオッズ由来の表示項目は使わない):
      馬: d_model = log(現行 v6 の勝率 (再生、レース内正規化)) − log q_T10、v6 生スコアのレース内 z、
          スピード指数 (予測) のレース内 z、d_est = log(推定オッズの逆数の正規化) − log q_T10、
          強さスコア PW のレース内 z、馬レベル score のレース内 z、
          展開予想の序盤・4 角の予想位置 (0=前〜1=後)、差し届く確率 × 4 角位置、ハイペース確率 × 序盤位置、
          log q_T10 と (log q_T10)^2 (市場の本命・大穴の偏りの補正)。
      レース: CHAOS 分位、頭数/18、芝か、新馬・未勝利・OP 以上のフラグ。
      欠損は 0 (z 系) または 0.5 (位置) で埋め、欠損フラグを足す。連続量は前半の平均・標準偏差で標準化。
  - モデル (3 つ、主 = R1):
      R0 (参照): f = log q_T10 と (log q_T10)^2 だけのロジスティック (市場の形の補正だけでどこまで行くか)。
      R1 (主): 全特徴の L2 正則化ロジスティック (オフセット logit q_T10)。λ ∈ {1e-4, 1e-3, 1e-2, 1e-1, 1} を
          前半の開催日単位 5 分割 CV の 2 値 LogLoss で選ぶ。
      R2 (副): 全特徴の LightGBM (init_score = logit q_T10、binary、num_leaves 4、learning_rate 0.02、
          min_data_in_leaf 200、feature_fraction 0.8、lambda_l2 10)。回数は同じ 5 分割 CV で早期停止 (50、上限 2,000)。
      R0 も R1 と同じ方法で λ を選ぶ。
  - 確率: 1 / (1 + exp(−(logit q_T10 + f))) をレース内で正規化。
  - 評価 (後半のみ、1 回): (1) T-10 市場との LogLoss・Brier・勝ち馬 NLL の差、(2) 最終市場との差
      (CI はレース単位の対応付きブートストラップ 4,000 回)、(3) EV = 補正後の確率 × 最終単勝オッズ、
      閾値 1.0 / 1.1 / 1.15、単勝の全該当と最上位 1 頭、回収率と 95%CI (払戻は確定配当)。
      参考: q_T10 そのまま × 最終オッズの EV、全馬均等。

使い方: python -m analysis.market_residual_check --work reports/objective_check --out-json <path>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from analysis import ev_threshold_backtest as bt
from analysis.ev_recalibration_check import metrics, paired_metric_diff
from analysis.market_blend_check import paired_winner_nll
from analysis.objective_check_eval import load_replay, norm
from analysis.t10_market_blend_check import post_times, t10_odds

SPLIT = "20260815"
ROOT = Path("E:/PyCaLiAI")
LAMBDAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
EV_THRESHOLDS = [1.0, 1.1, 1.15]
N_FOLDS = 5
LGB_PARAMS = {"objective": "binary", "num_leaves": 4, "learning_rate": 0.02, "min_data_in_leaf": 200,
              "feature_fraction": 0.8, "lambda_l2": 10.0, "verbose": -1, "seed": 42, "deterministic": True,
              "force_col_wise": True}


def zin(df, col):
    g = df.groupby("rid")[col]
    sd = g.transform("std").replace(0, np.nan)
    return ((df[col] - g.transform("mean")) / sd).fillna(0.0)


def site_features(dates: list[str]) -> pd.DataFrame:
    rows = []
    for d in dates:
        p = ROOT / "site" / "data" / f"{d}.json"
        if not p.exists():
            continue
        o = json.loads(p.read_text(encoding="utf-8"))
        for r in o.get("races", []):
            rid = str(r.get("race_id", ""))[:16]
            tk = r.get("tenkai") or {}
            tkh = {int(x[0]): x for x in (tk.get("h") or [])}
            pace = tk.get("pace") or [None, None, None]
            jd = r.get("judgment") or {}
            course, klass = str(r.get("course") or ""), str(r.get("klass") or "")
            for h in r.get("horses", []):
                u = int(h["umaban"])
                sp = h.get("speed") or {}
                lv = h.get("level") or {}
                t = tkh.get(u)
                rows.append({"rid": rid, "uma": u,
                             "speed_v": sp.get("v"), "est_odds": sp.get("eo"), "pw": h.get("pw"),
                             "level": lv.get("score") if isinstance(lv, dict) else None,
                             "tk_early": t[1] if t else None, "tk_c4": t[2] if t else None,
                             "late": tk.get("late"), "pace_high": pace[2] if len(pace) > 2 else None,
                             "field": r.get("field_size"),   # CHAOS は凍結 bundle の chaos_pct (load_rows) を使う
                             "turf": 1.0 if course.startswith("芝") else 0.0,
                             "shinba": 1.0 if klass == "新馬" else 0.0, "misho": 1.0 if klass == "未勝利" else 0.0,
                             "opplus": 1.0 if klass not in ("新馬", "未勝利", "1勝", "2勝", "3勝") else 0.0})
    return pd.DataFrame(rows)


def build_X(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    X = pd.DataFrame(index=df.index)
    lq = np.log(df["q"])
    X["d_model"] = np.log(df["m"]) - lq
    X["score_z"] = zin(df, "score")
    for c, name in (("speed_v", "speed_z"), ("pw", "pw_z"), ("level", "level_z")):
        X[name + "_na"] = df[c].isna().astype(float)
        X[name] = zin(df.assign(**{c: df[c].astype(float)}), c).where(df[c].notna(), 0.0)
    eo = df["est_odds"].astype(float)
    X["d_est_na"] = eo.isna().astype(float)
    inv = (1.0 / eo).where(eo > 0)
    pe = inv / inv.groupby(df["rid"]).transform("sum")
    X["d_est"] = (np.log(pe) - lq).where(eo.notna() & (eo > 0), 0.0).fillna(0.0)
    X["tk_na"] = df["tk_early"].isna().astype(float)
    X["tk_early"] = df["tk_early"].astype(float).fillna(0.5)
    X["tk_c4"] = df["tk_c4"].astype(float).fillna(0.5)
    X["late_x_c4"] = df["late"].astype(float).fillna(0.0) * X["tk_c4"]
    X["high_x_early"] = df["pace_high"].astype(float).fillna(0.0) * X["tk_early"]
    X["lq"] = lq
    X["lq2"] = lq ** 2
    X["chaos_na"] = df["chaos"].isna().astype(float)
    X["chaos"] = df["chaos"].astype(float).fillna(0.5)
    X["field"] = df["field"].astype(float).fillna(df["n_start"]) / 18.0
    for c in ("turf", "shinba", "misho", "opplus"):
        X[c] = df[c].astype(float).fillna(0.0)
    return X, list(X.columns)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_logit(X, y, off, lam):
    n, k = X.shape
    Xa = np.hstack([np.ones((n, 1)), X])

    def fg(b):
        z = off + Xa @ b
        p = 1 / (1 + np.exp(-z))
        loss = np.mean(np.logaddexp(0, z) - y * z) + 0.5 * lam * np.sum(b[1:] ** 2)
        g = Xa.T @ (p - y) / n
        g[1:] += lam * b[1:]
        return loss, g
    r = minimize(fg, np.zeros(k + 1), jac=True, method="L-BFGS-B", options={"maxiter": 2000})
    return r.x


def predict_logit(b, X, off):
    return 1 / (1 + np.exp(-(off + np.hstack([np.ones((len(X), 1)), X]) @ b)))


def race_norm(p, rid):
    s = pd.Series(p).groupby(rid).transform("sum").to_numpy()
    return p / s


def bin_ll(p, y):
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))


def date_folds(dates: np.ndarray) -> list[np.ndarray]:
    ud = np.array(sorted(set(dates)))
    return [np.isin(dates, ud[i::N_FOLDS]) for i in range(N_FOLDS)]


def cv_lambda(X, y, off, rid, dates) -> tuple[float, dict]:
    folds = date_folds(dates)
    scores = {}
    for lam in LAMBDAS:
        ll = []
        for te in folds:
            b = fit_logit(X[~te], y[~te], off[~te], lam)
            p = race_norm(predict_logit(b, X[te], off[te]), rid[te])
            ll.append(bin_ll(p, y[te]) * te.sum())
        scores[lam] = float(np.sum(ll) / len(y))
    return min(scores, key=scores.get), scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="reports/objective_check")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--fit-only", action="store_true", help="前半の学習までで止める (後半は計算しない、動作確認用)")
    a = ap.parse_args()
    work = Path(a.work).resolve()

    rows, _ = bt.load_rows(ROOT)
    fz = pd.DataFrame(rows)
    fz = fz[fz["o_fin"].notna() & ~fz["no_finish"] & fz["p_win"].notna()].copy()
    rp, _ = load_replay(work)
    df = fz.merge(rp[["rid", "umaban", "score", "p_replay"]], left_on=["rid", "uma"], right_on=["rid", "umaban"], how="left")
    posts = post_times(ROOT, sorted(df["date"].unique()))
    odds, _, _ = t10_odds(ROOT, sorted(df["rid"].unique()), posts)
    df["o10"] = [odds.get(r, {}).get(u) for r, u in zip(df["rid"], df["uma"])]
    sf = site_features(sorted(df["date"].unique()))
    df = df.merge(sf, on=["rid", "uma"], how="left", indicator="_site")
    bad = df.groupby("rid").apply(lambda g: bool(g[["score", "p_replay", "o10"]].isna().any().any()
                                                 or (g["_site"] != "both").any()))
    stat = {"frozen_races": int(fz["rid"].nunique()), "dropped_races": int(bad.sum())}
    df = df[~df["rid"].isin(set(bad[bad].index))].copy().reset_index(drop=True)
    df["win"] = (df["tan_ret"] > 0).astype(int)
    inv = 1.0 / df["o10"]
    df["q"] = inv / inv.groupby(df["rid"]).transform("sum")
    invf = 1.0 / df["o_fin"]
    df["q_fin"] = invf / invf.groupby(df["rid"]).transform("sum")
    df["m"] = norm(df, "p_replay").clip(lower=1e-4)
    df["n_start"] = df.groupby("rid")["uma"].transform("size")
    Xdf, cols = build_X(df)

    is_fit = (df["date"] <= SPLIT).to_numpy()
    mu, sd = Xdf[is_fit].mean(), Xdf[is_fit].std().replace(0, 1.0)
    binary_cols = [c for c in cols if c.endswith("_na") or c in ("turf", "shinba", "misho", "opplus")]
    Xs = Xdf.copy()
    for c in cols:
        if c not in binary_cols:
            Xs[c] = (Xdf[c] - mu[c]) / sd[c]
    y = df["win"].to_numpy(float)
    off = logit(df["q"].to_numpy())
    rid = df["rid"].to_numpy()
    dates = df["date"].to_numpy()
    F, V = is_fit, ~is_fit

    out = {"design": {"split": SPLIT, "lambdas": LAMBDAS, "lgb_params": LGB_PARAMS, "features": cols},
           "stat": stat,
           "coverage": {h: {"races": int(df.loc[m, "rid"].nunique()), "horses": int(m.sum()),
                            "dates": [df.loc[m, "date"].min(), df.loc[m, "date"].max(), int(df.loc[m, "date"].nunique())],
                            "eras": df.loc[m].drop_duplicates("rid")["era"].value_counts().to_dict()}
                        for h, m in (("fit", F), ("val", V))},
           "feature_missing_rate": {c: float(Xdf[c].mean()) for c in cols if c.endswith("_na")}}

    # ---- R0 / R1 (前半だけで λ 選択と学習) ----
    fitted = {}
    for name, use in (("R0", ["lq", "lq2"]), ("R1", cols)):
        Xm = Xs[use].to_numpy(float)
        lam, cv = cv_lambda(Xm[F], y[F], off[F], rid[F], dates[F])
        b = fit_logit(Xm[F], y[F], off[F], lam)
        fitted[name] = race_norm(predict_logit(b, Xm, off), rid)
        out[name] = {"lambda": lam, "cv_logloss": cv, "coef": dict(zip(["intercept"] + use, map(float, b)))}
        print(name, "lambda", lam, "cv", cv, flush=True)
    # ---- R2 (LightGBM、前半の CV で回数) ----
    XF = Xs[cols].to_numpy(float)
    folds = date_folds(dates[F])
    idxF = np.where(F)[0]
    cvres = lgb.cv(LGB_PARAMS, lgb.Dataset(XF[F], label=y[F], init_score=off[F]), num_boost_round=2000,
                   folds=[(np.where(~te)[0], np.where(te)[0]) for te in folds],
                   callbacks=[lgb.early_stopping(50, verbose=False)], stratified=False)
    key = [k for k in cvres if k.endswith("-mean")][0]
    n_round = len(cvres[key])
    gbm = lgb.train(LGB_PARAMS, lgb.Dataset(XF[F], label=y[F], init_score=off[F]), num_boost_round=n_round)
    fitted["R2"] = race_norm(1 / (1 + np.exp(-(off + gbm.predict(XF, raw_score=True)))), rid)
    out["R2"] = {"n_round": n_round, "cv_" + key: float(cvres[key][-1]),
                 "gain": dict(sorted(zip(cols, map(float, gbm.feature_importance("gain"))), key=lambda t: -t[1])[:12])}
    print("R2 rounds", n_round, flush=True)
    for k, p in fitted.items():
        df[f"p_{k}"] = p
    out["fit_insample"] = {k: bin_ll(df.loc[F, f"p_{k}"].to_numpy(), y[F]) for k in fitted} | {"q": bin_ll(df.loc[F, "q"].to_numpy(), y[F])}

    if a.fit_only:
        print(json.dumps({k: out[k] for k in ("coverage", "stat", "feature_missing_rate", "fit_insample")}, ensure_ascii=False, default=str))
        print(json.dumps({k: {"lambda": out[k]["lambda"], "cv": out[k]["cv_logloss"]} for k in ("R0", "R1")}), out["R2"])
        return
    # ---- 後半 (1 回だけ) ----
    val = df[V].copy()
    pcols = ["p_R0", "p_R1", "p_R2", "q", "q_fin"]
    out["val"] = {c: {k: v for k, v in metrics(val, c, "win").items()} for c in pcols}
    out["val_diff"] = {}
    for base in ("q", "q_fin"):
        for c in ("p_R0", "p_R1", "p_R2") + (("q",) if base == "q_fin" else ()):
            out["val_diff"][f"{c}-{base}"] = paired_metric_diff(val, c, base, "win") | {"winner_nll": paired_winner_nll(val, c, base)}
    ev = {}
    for c in ("p_R0", "p_R1", "p_R2", "q"):
        recs = val.to_dict("records")
        for r in recs:
            r["p_win"] = r[c]
        for k in ("tan_all", "tan_top1"):
            for t in EV_THRESHOLDS:
                ev[f"{c}|{k}@{t}"] = bt.summarize(bt.select(recs, k, t))
    out["val_ev"] = ev
    out["val_baseline_all_horses"] = bt.summarize([{**r, "ret": r["tan_ret"]} for r in val.to_dict("records")])
    Path(a.out_json).write_text(json.dumps(out, ensure_ascii=False, default=str), encoding="utf-8")

    print(json.dumps(out["coverage"], ensure_ascii=False), json.dumps(stat))
    for c in pcols:
        m = out["val"][c]
        print(f"val {c:6s} LL={m['logloss']:.5f} Brier={m['brier']:.5f} ECE={m['ece']:.4f}")
    for k, v in out["val_diff"].items():
        print(f"{k:12s} LL {v['logloss']['diff']:+.5f} [{v['logloss']['ci95'][0]:+.5f},{v['logloss']['ci95'][1]:+.5f}] "
              f"wNLL {v['winner_nll']['diff']:+.4f} [{v['winner_nll']['ci95'][0]:+.4f},{v['winner_nll']['ci95'][1]:+.4f}]")
    for k, s in ev.items():
        if s.get("n_bets"):
            print(f"{k:22s} bets={s['n_bets']:5d} R={s['n_races']:4d} hit={s['hit_rate']*100:5.1f}% roi={s['roi']*100:6.1f}% "
                  f"CI[{s['roi_ci95'][0]*100:.0f},{s['roi_ci95'][1]*100:.0f}] ex10={s['roi_ex_top10']*100:.1f}%")
        else:
            print(k, "bets=0")
    print("baseline", out["val_baseline_all_horses"]["roi"])


if __name__ == "__main__":
    main()
