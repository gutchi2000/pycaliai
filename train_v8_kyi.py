# -*- coding: utf-8 -*-
"""
train_v8_kyi.py — v8 候補: v7 (v6 + JRDB 過去走 j1_*) に、JRDB の前日データ KYI と外厩 (k8_*) を足して学習する (2026-11-15)
==========================================================================================================
作り方は v7 と同じ (ラベル・学習窓 〜2022・重み・ハイパーパラメータ・回数・エンコーダは本番 v6 (C1) の bundle から)。変えるのは特徴だけ。
  V7  = v6 の 120 特徴 + jrdb_features.FEATURES (v7 と同じ。seed ごとに同じ条件で学習し直して比較の土台にする)
  V8K = V7 + KYI の前日指数・展開予想 (k8_* のうち外厩を除く)
  V8  = V7 + k8_* すべて (KYI + 外厩)
評価: test (2024-25) の ◎1着率・◎3着内率、開催日 paired bootstrap。seed 42 / 1 / 7。
      seed 42 の各アームのスコアを valid+test について reports/v8_scores.parquet に保存 (市場ゲート用、git 管理外)。
保存: models/unified_rank_v8.pkl (seed 42 の V8)。本番の切替はしない (production_policy.json・artifact_pins.json に触れない)。
実行: PYTHONUTF8=1 E:/PyCaLiAI/venv311/Scripts/python.exe train_v8_kyi.py
出力: reports/train_v8_kyi.json (JRDB のデータ値は書かない)
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
import jrdb_kyi_features as kf  # noqa: E402
import optuna_v6_marks as ovm  # noqa: E402
from train_v7_jrdb import marks_eval, paired  # noqa: E402

MASTER = BASE / "data" / "master_v2_20130105-20251228.csv"
V6 = BASE / "models" / "unified_rank_v6.pkl"
if V6.stat().st_size < 10_000:          # worktree では LFS の参照ファイルのまま (実体は本体の checkout にある)
    V6 = Path("E:/PyCaLiAI/models/unified_rank_v6.pkl")
OUT = BASE / "models" / "unified_rank_v8.pkl"
REPORT = BASE / "reports" / "train_v8_kyi.json"
SCORES = BASE / "reports" / "v8_scores.parquet"
SEEDS = [42, 1, 7]
GK = ["k8_gk_has", "k8_gk_fac", "k8_gk_days", "k8_gk_weeks"]


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
    df = jf.add_jrdb_feats(df)
    df = kf.add_master_feats(df)
    cov = df.assign(y=df[ovm.COL_RID].astype(str).str[:4]).groupby("y").apply(lambda x: kf.coverage(x)).to_dict()
    print(f"master {len(df):,} 行  KYI/外厩 の埋まり {json.dumps({k: {a: round(v, 3) for a, v in d.items()} for k, d in cov.items()})}  ({time.time() - t0:.0f}s)", flush=True)
    enc = BT.apply_encoders(df, encs)
    tr = enc[enc["split"] == "train"].copy(); te = enc[enc["split"] == "test"].copy(); va = enc[enc["split"] == "valid"].copy()
    tr["winner_tansho"] = tr[ovm.COL_RID].astype(str).map(ovm.load_winner_tansho_pay()).fillna(100.0)
    f7 = feats_r + jf.FEATURES
    arms = {"V7": f7, "V8K": f7 + [c for c in kf.FEATURES if c not in GK], "V8": f7 + kf.FEATURES}

    def X(d, feats):
        return d[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values

    res = {"n_test_races": int(te[ovm.COL_RID].nunique()), "coverage": cov, "seeds": {}}
    sc = pd.concat([va, te])[[ovm.COL_RID, "馬番", "split"]].copy()
    sc["rid_s"] = sc[ovm.COL_RID].astype(str).str[:16]; sc["ban"] = pd.to_numeric(sc["馬番"], errors="coerce").astype(int)
    sc = sc[["rid_s", "ban", "split"]].reset_index(drop=True)
    vt = pd.concat([va, te])
    models = {}
    for seed in SEEDS:
        p = dict(params); p["seed"] = seed
        out = {}
        for arm, feats in arms.items():
            ds, _, _ = ovm.make_dataset(tr, feats, alpha=alpha)
            w = ds.get_weight(); assert (np.abs(w - 1.0) > 1e-9).mean() > 0.5, "重みが効いていない"
            m = lgb.train(p, ds, num_boost_round=rounds)
            out[arm] = marks_eval(te, m.predict(X(te, feats)))
            models[(seed, arm)] = m
            if seed == 42:
                sc[f"s_{arm}"] = m.predict(X(vt, feats))
            print(f"  seed {seed} {arm}: ◎1着 {100 * out[arm].win.mean():.2f}%  ◎3着内 {100 * out[arm].top3.mean():.2f}%  ({time.time() - t0:.0f}s)", flush=True)
        r = {a: {"win": float(100 * out[a].win.mean()), "top3": float(100 * out[a].top3.mean())} for a in arms}
        for a, bb in (("V8K", "V7"), ("V8", "V7"), ("V8", "V8K")):
            r[f"{a}-{bb}"] = {"top3": paired(out[bb], out[a], "top3"), "win": paired(out[bb], out[a], "win")}
            d3, d1 = r[f"{a}-{bb}"]["top3"], r[f"{a}-{bb}"]["win"]
            print(f"  seed {seed}: {a}−{bb} ◎3着内 {d3[0]:+.2f}pt [{d3[1]:+.2f},{d3[2]:+.2f}]  ◎1着 {d1[0]:+.2f}pt [{d1[1]:+.2f},{d1[2]:+.2f}]", flush=True)
        res["seeds"][seed] = r
    m8 = models[(42, "V8")]
    imp = pd.Series(m8.feature_importance("gain"), index=arms["V8"]); tot = imp.sum()
    res["gain_share"] = {"j1": float(imp[jf.FEATURES].sum() / tot), "k8_kyi": float(imp[[c for c in kf.FEATURES if c not in GK]].sum() / tot),
                         "k8_gaikyu": float(imp[GK].sum() / tot)}
    res["top_gain"] = [(k, float(v / tot)) for k, v in imp.sort_values(ascending=False).head(20).items()]
    p42 = dict(params); p42["seed"] = 42
    joblib.dump({"model": m8, "feature_cols": arms["V8"], "encoders": encs, "seed": 42, "variant": "v8_kyi_gaikyu_candidate",
                 "jrdb_mode": jf.MODE, "kyi_mode": kf.MODE, "base_lgbm_params": p42, "num_boost_round": rounds,
                 "sample_weight_alpha": alpha, "parent": "unified_rank_v7 (v6 C1 + j1_v1)",
                 "note": "2026-11-15 v7 に JRDB 前日 KYI と外厩 (k8_*) を足した候補。本番切替はしていない。配信経路は未実装。"}, OUT)
    sc.to_parquet(SCORES, index=False)
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("gain 比", {k: round(100 * v, 1) for k, v in res["gain_share"].items()})
    print("gain 上位:", ", ".join(f"{k} {100 * v:.1f}%" for k, v in res["top_gain"][:12]))
    print(f"saved {OUT}  ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
