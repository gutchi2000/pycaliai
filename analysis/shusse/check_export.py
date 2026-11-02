# -*- coding: utf-8 -*-
"""
analysis/shusse/check_export.py — 出世予測 AI の答え合わせ（確定した馬の予測 vs 結果）をサイト用に書き出す
======================================================================================================
predictions_latest.parquet のうち **確定行**（基準日 + 365 日 ≤ asof、12 か月窓が閉じた予測）を使い、

    out_v2/live/site/check_index.json     デビュー年 × 基準点ごとの指標・混同行列・較正ビン・外し/拾いの例
    out_v2/live/site/check_{debut_year}.json  馬ごとの確定行（基準点ごとの予測と実績）

を書く。build_site.py が site/data/shusse/ にコピーし、site/shusse_check.html が読む。
的中・外れの判定はページ側（narrate.py と同じ規則: 走らず→離脱、走った→予測クラスと実クラスの差 0/1/2+）。
市場オッズ・人気は含まない。

実行:
    ./venv311/Scripts/python.exe -m analysis.shusse.check_export
（site_export.build() の最後からも呼ばれる）
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.shusse.build_career import COL_HORSE
from analysis.shusse.narrate import LEVEL_NAME, load_trainer_names
from analysis.shusse.train_v2 import auc, qwk

OUT = Path(__file__).resolve().parent / "out_v2"
LIVE = OUT / "live"
DST = LIVE / "site"
KS = (1, 2, 3, 5)
BINS = [0.0, 0.02, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.0001]
N_EXAMPLES = 8


def _r(x, nd=3):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


def _i(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else int(x)


def _metrics(g: pd.DataFrame, val_k: dict | None) -> dict:
    ran = g[g["actual_ran_12m"] == 1]
    y = ran["actual_level"].astype(int).values
    pred = ran["pred_level"].astype(int).values
    conf = np.zeros((5, 5), dtype=int)                     # conf[pred][actual] 走った馬のみ
    for p, a in zip(pred, y):
        conf[p][a] += 1
    drop_by_pred = g[g["actual_ran_12m"] == 0]["pred_level"].astype(int).value_counts().reindex(range(5), fill_value=0)
    # 較正: P(重賞掲示板 12m) のビン → 実際に重賞掲示板に載った率
    cal = []
    cut = pd.cut(g["p_graded12"], BINS, right=False, include_lowest=True)
    for iv, d in g.groupby(cut, observed=False):
        if len(d) == 0:
            continue
        cal.append({"lo": float(iv.left), "hi": min(1.0, float(iv.right)), "n": int(len(d)),
                    "mean_p": _r(d["p_graded12"].mean()), "rate": _r(d["actual_graded12"].mean())})
    by_pred = {}
    for lv in range(5):
        d = g[g["pred_level"] == lv]
        dr = d[d["actual_ran_12m"] == 1]
        by_pred[str(lv)] = {"n": int(len(d)), "dropout": int((d["actual_ran_12m"] == 0).sum()),
                            "exact": int((dr["actual_level"] == lv).sum()), "within1": int(((dr["actual_level"] - lv).abs() <= 1).sum()),
                            "graded12_rate": _r(d["actual_graded12"].mean())}
    m = {
        "n": int(len(g)), "n_ran": int(len(ran)), "dropout_rate": _r(1 - len(ran) / max(1, len(g))), "graded12_rate": _r(g["actual_graded12"].mean()),
        "qwk": _r(qwk(y, pred)) if len(ran) > 10 else None,
        "exact_rate": _r((pred == y).mean()) if len(ran) else None,
        "within1_rate": _r((np.abs(pred - y) <= 1).mean()) if len(ran) else None,
        "g12_auc": _r(auc(g["actual_graded12"].values, g["p_graded12"].values)) if g["actual_graded12"].nunique() > 1 else None,
        "d_auc": _r(auc((g["actual_ran_12m"] == 0).astype(int).values, g["p_dropout"].values)) if g["actual_ran_12m"].nunique() > 1 else None,
        "confusion": conf.tolist(), "dropout_by_pred": [int(v) for v in drop_by_pred.values],
        "calibration": cal, "by_pred": by_pred,
    }
    if val_k:                                               # predict_live の CI 付き値を優先して併記
        m["qwk_ci"] = [_r(x) for x in val_k.get("A_qwk_ci95", [])] or None
        m["g12_auc_ci"] = [_r(x) for x in val_k.get("G12_auc_ci95", [])] or None
        m["rule_qwk"] = _r(val_k.get("rule_qwk"))
    return m


def _examples(g: pd.DataFrame, tnames: dict) -> dict:
    """外した例（高い確率を付けたのに重賞掲示板に届かず/離脱）と拾った例（低い確率から重賞掲示板）。"""
    def row(r):
        return {"id": int(getattr(r, COL_HORSE)), "name": r.馬名, "k": int(r.k), "pg12": _r(r.p_graded12), "pd": _r(r.p_dropout),
                "lvl": int(r.pred_level), "ran": int(r.actual_ran_12m), "alvl": _i(r.actual_level), "g12": int(r.actual_graded12),
                "summary": r.summary, "trainer": tnames.get(int(r.調教師コード))}
    miss = g[(g["actual_graded12"] == 0) & (g["actual_ran_12m"] == 1)].sort_values("p_graded12", ascending=False).head(N_EXAMPLES)
    drop = g[(g["actual_ran_12m"] == 0)].sort_values("p_graded12", ascending=False).head(N_EXAMPLES)
    hit = g[g["actual_graded12"] == 1].sort_values("p_graded12", ascending=True).head(N_EXAMPLES)
    return {"miss": [row(r) for r in miss.itertuples(index=False)], "dropout": [row(r) for r in drop.itertuples(index=False)],
            "surprise": [row(r) for r in hit.itertuples(index=False)]}


def build(write: bool = True) -> dict:
    t0 = time.time()
    pred = pd.read_parquet(LIVE / "predictions_latest.parquet")
    val = json.loads((LIVE / "live_validation.json").read_text(encoding="utf-8")) if (LIVE / "live_validation.json").exists() else {}
    tnames = load_trainer_names()
    asof = pd.Timestamp(pred["asof"].max())
    c = pred[pred["confirmed"]].copy()
    c["actual_level"] = c["actual_level"].astype(float)

    years = sorted(int(y) for y in c["debut_year"].unique())
    index = {"model": "shusse_v2", "asof": asof.strftime("%Y-%m-%d"), "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
             "levels": {str(i): LEVEL_NAME[i] for i in range(5)}, "ks": list(KS), "years": years[::-1], "bins": BINS[:-1] + [1.0],
             "n_rows": int(len(c)), "n_horses": int(c[COL_HORSE].nunique()),
             "note": ("確定 = 基準日 + 365 日 ≤ asof。confusion[pred][actual] は走った馬のみ、dropout_by_pred は予測クラス別の離脱数。"
                      "calibration は P(重賞掲示板 12m) のビンごとの実現率。examples.miss = 走ったが重賞掲示板に届かなかった馬を予測確率の高い順、"
                      "dropout = 離脱した馬を同順、surprise = 重賞掲示板に載った馬を予測確率の低い順。市場情報は含まない。"),
             "by_year": {}}
    files: dict[str, dict] = {}
    for y in years:
        gy = c[c["debut_year"] == y]
        vy = val.get("by_cohort_k", {}).get(str(y), {})
        index["by_year"][str(y)] = {"n_horses": int(gy[COL_HORSE].nunique()),
                                    "k": {str(k): _metrics(gy[gy["k"] == k], vy.get(f"k{k}")) for k in KS if (gy["k"] == k).any()},
                                    "examples": {str(k): _examples(gy[gy["k"] == k], tnames) for k in KS if (gy["k"] == k).any()}}
        horses = []
        for hid, d in gy.sort_values("k").groupby(COL_HORSE, sort=False):
            last = d.iloc[-1]
            rows = [[int(r.k), pd.Timestamp(r.base_date).strftime("%Y-%m-%d"), _r(r.p_graded12), _r(r.p_dropout), int(r.pred_level),
                     _r(r.percentile_graded12, 1), int(r.actual_ran_12m), _i(r.actual_level), int(r.actual_graded12),
                     (int(r.actual_graded24) if bool(r.confirmed24) and not pd.isna(r.actual_graded24) else None)]
                    for r in d.itertuples(index=False)]
            horses.append({"id": int(hid), "name": last["馬名"], "sex": last["性別"], "sire": (None if pd.isna(last["父"]) else last["父"]),
                           "trainer": tnames.get(int(last["調教師コード"])), "dy": int(y), "n": int(last["n_starts_total"]),
                           "summary": last["summary"], "rows": rows})
        horses.sort(key=lambda h: -max(r[2] for r in h["rows"]))
        files[f"check_{y}.json"] = {"asof": index["asof"], "year": y, "n": len(horses),
                                    "cols": ["k", "base", "pg12", "pd", "lvl", "pct", "ran", "alvl", "g12", "g24"], "horses": horses}
    files["check_index.json"] = index
    if write:
        DST.mkdir(parents=True, exist_ok=True)
        tot = 0
        for name, obj in files.items():
            fp = DST / name
            fp.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str), encoding="utf-8")
            tot += fp.stat().st_size
        sizes = ", ".join(f"{n} {(DST / n).stat().st_size / 1e6:.2f}MB" for n in files)
        print(f"[check_export] {len(c):,} confirmed rows / {index['n_horses']:,} horses / {tot / 1e6:.2f} MB ({sizes}) / {time.time() - t0:.1f}s")
    return files


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    build()
