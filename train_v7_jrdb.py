# -*- coding: utf-8 -*-
"""
train_v7_jrdb.py — v7 候補: 本番 v6 と同じ作り方に、JRDB の過去走特徴 (j1_*、59 列) だけを足して学習する (2026-11-13)
=================================================================================================
変えるのは特徴だけ。ラベル (6−着順 を 0〜5 に丸め)、学習窓 (〜2022)、重み (1 + α·log1p(勝ち馬単勝/100))、
ハイパーパラメータ、回数 (515)、エンコーダ、乱数 seed は本番 v6 (C1) の bundle からそのまま引く。
  R  = v6 の 120 特徴 (同じコードで再学習。本番 v6 と一致するかの確認用)
  V7 = R + jrdb_features.FEATURES
評価: test (2024-25) の ◎1着率・◎3着内率、開催日 paired bootstrap。seed 42 / 1 / 7。
保存: models/unified_rank_v7.pkl (seed 42。bundle に jrdb_mode を刻む → 各スクリプトが JRDB 特徴を足す合図)。
      本番の切替はしない (data/production_policy.json・artifact_pins.json は触らない)。
実行: PYTHONUTF8=1 ./venv311/Scripts/python.exe train_v7_jrdb.py
出力: reports/train_v7_jrdb.json (JRDB のデータ値は書かない)
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
import backtest_pl_ev as BT  # noqa: E402
import jrdb_features as jf  # noqa: E402
import optuna_v6_marks as ovm  # noqa: E402

MASTER = BASE / "data" / "master_v2_20130105-20251228.csv"
V6 = BASE / "models" / "unified_rank_v6.pkl"
OUT = BASE / "models" / "unified_rank_v7.pkl"
REPORT = BASE / "reports" / "train_v7_jrdb.json"
SEEDS = [42, 1, 7]


def marks_eval(te: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
    """レースごとの ◎ (スコア最大) の 1 着・3 着内。"""
    d = te[[ovm.COL_RID, ovm.COL_JYUN]].copy(); d["s"] = score; d["date"] = te["日付"].values
    i = d.groupby(ovm.COL_RID).s.idxmax()
    t = d.loc[i]
    return pd.DataFrame({"date": t.date.values, "win": (t[ovm.COL_JYUN] == 1).values.astype(float), "top3": (t[ovm.COL_JYUN] <= 3).values.astype(float)}, index=t[ovm.COL_RID].values)


def paired(a: pd.DataFrame, b: pd.DataFrame, col: str, n_boot=4000, seed=0):
    j = a[[col, "date"]].join(b[[col]], rsuffix="_b", how="inner")
    g = j.groupby("date").agg(d=(col, lambda s: 0.0), n=(col, "size"))
    diff = (j[col + "_b"] - j[col]).groupby(j.date).sum().values; n = j.groupby("date").size().values
    rng = np.random.default_rng(seed); i = rng.integers(0, len(n), (n_boot, len(n)))
    v = diff[i].sum(1) / n[i].sum(1)
    return float(100 * diff.sum() / n.sum()), float(100 * np.percentile(v, 2.5)), float(100 * np.percentile(v, 97.5))


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    t0 = time.time()
    b = joblib.load(V6)
    feats_r, encs = list(b["feature_cols"]), b["encoders"]
    params = dict(b["base_lgbm_params"]); rounds = int(b["num_boost_round"]); alpha = float(b["sample_weight_alpha"])
    df = pd.read_csv(MASTER, encoding="utf-8-sig", low_memory=False)
    df[ovm.COL_JYUN] = pd.to_numeric(df[ovm.COL_JYUN], errors="coerce")
    df = df.dropna(subset=[ovm.COL_JYUN, ovm.COL_RID, "split"]).copy()
    df["label"] = np.clip(6 - df[ovm.COL_JYUN].astype(int), 0, 5).astype(int)
    df = jf.add_jrdb_feats(df)                       # エンコード前 (血統登録番号・日付の生値で引く)
    print(f"master {len(df):,} 行  前走 IDM あり {100 * jf.coverage(df):.1f}%  ({time.time() - t0:.0f}s)", flush=True)
    win_pay = ovm.load_winner_tansho_pay()
    enc = BT.apply_encoders(df, encs)
    tr = enc[enc["split"] == "train"].copy(); te = enc[enc["split"] == "test"].copy()
    tr["winner_tansho"] = tr[ovm.COL_RID].astype(str).map(win_pay).fillna(100.0)
    feats_v7 = feats_r + jf.FEATURES
    Xte = {"R": te[feats_r].apply(pd.to_numeric, errors="coerce").fillna(-9999).values,
           "V7": te[feats_v7].apply(pd.to_numeric, errors="coerce").fillna(-9999).values}
    prod = marks_eval(te, b["model"].predict(Xte["R"]))
    res = {"n_test_races": int(len(prod)), "prod_v6": {"win": float(100 * prod.win.mean()), "top3": float(100 * prod.top3.mean())}, "seeds": {}}
    print(f"本番 v6: ◎1着 {res['prod_v6']['win']:.2f}%  ◎3着内 {res['prod_v6']['top3']:.2f}%  ({len(prod):,}R)", flush=True)
    models = {}
    for seed in SEEDS:
        p = dict(params); p["seed"] = seed
        out = {}
        for arm, feats in (("R", feats_r), ("V7", feats_v7)):
            ds, _, _ = ovm.make_dataset(tr, feats, alpha=alpha)
            w = ds.get_weight(); assert (np.abs(w - 1.0) > 1e-9).mean() > 0.5, "重みが効いていない"
            m = lgb.train(p, ds, num_boost_round=rounds)
            ev = marks_eval(te, m.predict(Xte[arm]))
            out[arm] = ev; models[(seed, arm)] = m
            print(f"  seed {seed} {arm}: ◎1着 {100 * ev.win.mean():.2f}%  ◎3着内 {100 * ev.top3.mean():.2f}%  ({time.time() - t0:.0f}s)", flush=True)
        d3 = paired(out["R"], out["V7"], "top3"); d1 = paired(out["R"], out["V7"], "win"); dp = paired(prod, out["V7"], "top3")
        res["seeds"][seed] = {"R": {"win": float(100 * out["R"].win.mean()), "top3": float(100 * out["R"].top3.mean())},
                              "V7": {"win": float(100 * out["V7"].win.mean()), "top3": float(100 * out["V7"].top3.mean())},
                              "delta_top3_vs_R": d3, "delta_win_vs_R": d1, "delta_top3_vs_prod": dp}
        print(f"  seed {seed}: V7−R ◎3着内 {d3[0]:+.2f}pt [{d3[1]:+.2f},{d3[2]:+.2f}]  ◎1着 {d1[0]:+.2f}pt [{d1[1]:+.2f},{d1[2]:+.2f}]  | V7−本番 {dp[0]:+.2f}pt [{dp[1]:+.2f},{dp[2]:+.2f}]", flush=True)
    # seed 42 の R が本番 v6 と同じ ◎ を選ぶか (再現性の確認)
    r42 = marks_eval(te, models[(42, "R")].predict(Xte["R"]))
    res["repro_R42_same_top_pick"] = float((r42.join(prod, rsuffix="_p").eval("top3 == top3_p & win == win_p")).mean())
    m7 = models[(42, "V7")]
    imp = pd.Series(m7.feature_importance("gain"), index=feats_v7); tot = imp.sum()
    res["jrdb_gain_share"] = float(imp[jf.FEATURES].sum() / tot)
    res["top_gain"] = [(k if not k.startswith(jf.JRDB_FEATURE_PREFIX) else k, float(v / tot)) for k, v in imp.sort_values(ascending=False).head(15).items()]
    p42 = dict(params); p42["seed"] = 42
    joblib.dump({"model": m7, "feature_cols": feats_v7, "encoders": encs, "seed": 42, "variant": "v7_jrdb_j1_candidate",
                 "jrdb_mode": jf.MODE, "base_lgbm_params": p42, "num_boost_round": rounds, "sample_weight_alpha": alpha,
                 "parent": "unified_rank_v6 (C1_v2venv311_weightfixed)",
                 "note": "2026-11-13 v6 の作り方に JRDB 過去走特徴 59 列だけを足した候補。本番切替はしていない。"}, OUT)
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"R42 と本番 v6 の ◎ 成績一致率 {100 * res['repro_R42_same_top_pick']:.2f}%  JRDB 特徴の gain 比 {100 * res['jrdb_gain_share']:.1f}%")
    print("gain 上位:", ", ".join(f"{k} {100 * v:.1f}%" for k, v in res["top_gain"][:10]))
    print(f"saved {OUT}  ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
