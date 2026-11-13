# -*- coding: utf-8 -*-
"""
C2 候補 vs 本番 v6 (C1) — 配信と同じ情報条件での比較 (valid 2023 / test 2024-25)。

条件:
  v6_trained : 本番 master のまま (学習時と同じ = 同日先行レース込みの騎手成績)。参考値。
  v6_served  : 騎手・調教師成績を「開催週末の初日より前」(実際に配信される値) に差し替え。
               配信で死んでいる特徴 (serve baseline cov<40%) はマスク。← 本番の実力に近い条件
  c2a / c2b  : C2 master (配信と同じ定義)。c2a は死んでいる特徴をマスク、c2b はそもそも使わない。
確率は較正前の PL (較正器はどのモデルも作り直しになるため同条件で比べる)。
差は同じレース集合の対応のある差、CI は開催日単位のクラスタ bootstrap (2000 回)。

実行: PYTHONUTF8=1 E:/PyCaLiAI/venv311/Scripts/python.exe analysis/c2_feature_fix/eval_c2.py
出力: E:/PyCaLiAI/candidates/c2_feature_fix/eval_c2.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
import pl_probs as PL  # noqa: E402
from backtest_pl_ev import all_fukusho_vec_fast, apply_encoders  # noqa: E402

ROOT = Path(r"E:\PyCaLiAI")
CAND = ROOT / "candidates/c2_feature_fix"
RID, BAN, JYUN = "レースID(新/馬番無)", "馬番", "着順"
FUKU = ["jockey_fuku30", "jockey_fuku90", "trainer_fuku30", "trainer_fuku90"]
rng = np.random.default_rng(7)

base_cov = json.load(open(ROOT / "data/serve_feature_baseline.json", encoding="utf-8"))["baseline_cov"]
prod = joblib.load(ROOT / "models/unified_rank_v6.pkl")
c2 = pd.read_parquet(CAND / "data/master_c2.parquet")
c2 = c2[c2["split"].isin(["valid", "test"])].copy()
c2[JYUN] = pd.to_numeric(c2[JYUN], errors="coerce")
c2 = c2.dropna(subset=[JYUN])
need = sorted(set(prod["feature_cols"]) | {RID, BAN, JYUN, "split"})
pm = pd.read_csv(ROOT / "data/master_v2_20130105-20251228.csv", encoding="utf-8-sig", low_memory=False,
                 usecols=lambda c: c in need)
pm[JYUN] = pd.to_numeric(pm[JYUN], errors="coerce")
pm = pm[pm["split"].isin(["valid", "test"])].dropna(subset=[JYUN])
pm = pm.set_index([RID, BAN]).loc[pd.MultiIndex.from_frame(c2[[RID, BAN]])].reset_index()
assert len(pm) == len(c2)
served = pm.copy()
for c in FUKU:
    served[c] = c2[c].to_numpy()


def score(bundle, frame, mask_dead=True):
    feats, encs = bundle["feature_cols"], bundle["encoders"]
    d = frame.copy()
    dead = [c for c in feats if base_cov.get(c, 0.0) < 0.40] if mask_dead else []
    for c in dead:
        if c in encs:
            d[c] = "__NaN__"
    d = apply_encoders(d, encs)
    for c in dead:
        if c not in encs:
            d[c] = np.nan
    X = d[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    return bundle["model"].predict(X)


conds = {"v6_trained": score(prod, pm), "v6_served": score(prod, served)}
for v in ("c2a", "c2b"):
    for s in (42, 1):
        b = joblib.load(CAND / f"models/unified_rank_{v}_seed{s}.pkl")
        conds[f"{v}_s{s}"] = score(b, c2)

frame = c2[[RID, BAN, JYUN]].copy()
frame["date"] = frame[RID].astype(str).str[:8]
for k, v in conds.items():
    frame[k] = v

rows = []
for rid, g in frame.groupby(RID, sort=False):
    if len(g) < 5:
        continue
    pos = g[JYUN].to_numpy()
    yw, y3 = (pos == 1).astype(float), (pos <= 3).astype(float)
    rec = {"rid": rid, "date": g["date"].iat[0]}
    for k in conds:
        s = g[k].to_numpy()
        w = PL.pl_weights(s)
        pw, ps = PL.all_tansho(w), all_fukusho_vec_fast(w)
        h = int(np.argmax(s))
        rec[f"{k}|hon_win"] = float(pos[h] == 1)
        rec[f"{k}|hon_top3"] = float(pos[h] <= 3)
        rec[f"{k}|ll_w"] = float(-np.log(max(pw[yw == 1].sum(), 1e-6)))       # 勝ち馬の対数尤度
        rec[f"{k}|brier_3"] = float(np.mean((ps - y3) ** 2))
    rows.append(rec)
R = pd.DataFrame(rows)
R["period"] = np.where(R["date"] < "20240101", "valid2023", "test2024_25")
# seed 平均 (指標の平均)
for v in ("c2a", "c2b"):
    for m in ("hon_win", "hon_top3", "ll_w", "brier_3"):
        R[f"{v}_mean|{m}"] = (R[f"{v}_s42|{m}"] + R[f"{v}_s1|{m}"]) / 2


def cluster_ci(sub, col, B=2000):
    by = sub.groupby("date")[col].agg(["sum", "count"])
    s, n = by["sum"].to_numpy(), by["count"].to_numpy()
    idx = rng.integers(0, len(by), size=(B, len(by)))
    vals = s[idx].sum(1) / n[idx].sum(1)
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


res = {}
names = ["v6_trained", "v6_served", "c2a_s42", "c2a_s1", "c2b_s42", "c2b_s1", "c2a_mean", "c2b_mean"]
for per, sub in R.groupby("period"):
    out = {"n_races": int(len(sub))}
    for k in names:
        out[k] = {m: round(float(sub[f"{k}|{m}"].mean()), 4) for m in ("hon_win", "hon_top3", "ll_w", "brier_3")}
    for k in ["v6_trained", "c2a_mean", "c2b_mean"]:
        d = {}
        for m in ("hon_top3", "hon_win", "ll_w"):
            col = f"__d_{k}_{m}"
            sub = sub.assign(**{col: sub[f"{k}|{m}"] - sub[f"v6_served|{m}"]})
            d[m] = {"diff": round(float(sub[col].mean()), 4), "ci95": cluster_ci(sub, col)}
        out[f"{k}_minus_v6_served"] = d
    res[per] = out
json.dump(res, open(CAND / "eval_c2.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
