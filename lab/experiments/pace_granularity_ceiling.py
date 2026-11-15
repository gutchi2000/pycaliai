# -*- coding: utf-8 -*-
"""
pace_granularity_ceiling.py — 展開を「もっと細かく」学習したときの上限を先に測る (2026-11-15)
=====================================================================================
問い
----
「馬の展開をもっと細かく学習できないか」。11/14 の展開の順問題 (`PACE_FORWARD_PROBLEM_20261114.md`) は
ペースを JRDB の 3 区分 (S/M/H) で扱い、オラクル (実際のペースを入れる) でも上乗せは 0.0005〜0.001 nats/R だった。
同文書 §7 で未検証として残ったのは「ペースの連続量」「今日の位置取りそのもの」「道中の不利・コース取り」。
これらを細かく学習する価値があるかは、まず「完全に分かったら (オラクル) いくら効くか」で決まる。
上限が小さければ、どれだけ細かく学習しても取り出せる量はそれ以下なので、作り込む前に止められる。

これは診断 (採用候補ではない)。オラクル列は今日のレースの事後情報で、予想には使えない。
上限が大きかった集合についてだけ、発走前の情報でその列がどれだけ予測できるか (R²) と、
予測値で置き換えた z (「現実に取り出せる分」) を同じゲートで測る。

データ
------
JRDB SED (平地、取消・除外を除く)。馬の過去走量は `pace_forward_problem_exp.build_runs()` をそのまま使う (as-of)。
追加で過去走の 出遅れ率・4角コース取り平均・1角→4角の位置の上げ幅・上がり指数 を as-of で作る (細かい脚質)。

モデル
------
LightGBM 2 値 (目的 = 1 着)、2013-2020 で学習、2021-22 で早期終了。2023-25 に予測し、log-odds を z とする。
  pre   : 発走前の脚質量だけ (ten3 tenmax5 pf5 lead5 pidx3 idm5 y5 nprev ddist 距離 芝ダ 場 頭数 馬番/頭数
          + 細かい脚質 deok5 yc5 gain5 agari3)
  R     : pre + レース単位の今日の実際 (ペース区分 q, レースペース指数 racep_idx, 今日のテン指数のレース平均,
          最初のコーナーで前 15% にいた頭数)            … ペースを連続量で知ったら
  P     : pre + 馬単位の今日の実際の前半 (最初のコーナー相対位置, 先頭か, 普段との差, 出遅れ)  … 位置取りを知ったら
  T     : pre + 道中 (4角コース取り, 不利 前/中/後/計)   … 道中の細部を知ったら
  RP / RPT : 組み合わせ
ゲート: gate_common.gate を各 z で 1 回ずつ。ΔNLL の差 (z_X − z_pre) を「その細部の上限」と読む。

読み方 (事前に決める)
---------------------
- ある集合の M ゲート (確定市場 + v7 → +z) の上乗せが z_pre より 0.002 nats/R 以上大きくなければ、
  その細部は「完全に分かっても効かない」= 細かく学習する価値なしと書く。
- 0.002 以上なら、発走前の予測でその列をどこまで再現できるか (R²) と、予測値で置き換えた z_X_hat を測る。
  z_X_hat が M で PASS しない限り、配線の候補にはしない。
- P・T には「今日調子が良い馬は前に行ける / 不利を受けにくい」という結果の漏れが混ざるので、上限は過大側に出る。

出力: reports/pace_granularity_ceiling.json (集計値のみ。JRDB の馬別値は出さない)
再実行: worktree root から  E:/PyCaLiAI/venv311/Scripts/python.exe lab/experiments/pace_granularity_ceiling.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate_common as gc  # noqa: E402
import pace_forward_problem_exp as pfx  # noqa: E402

T0 = time.time()
OUT = gc.REP / "pace_granularity_ceiling.json"


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def add_fine(d: pd.DataFrame) -> pd.DataFrame:
    ex = gc.load_sed(["racekey", "ymd", "ban", "deokure", "furi", "mae_furi", "naka_furi", "ato_furi",
                      "yon_course", "agari_idx"])
    ex = ex[["rid_s", "ban", "deokure", "furi", "mae_furi", "naka_furi", "ato_furi", "yon_course", "agari_idx"]]
    d = d.merge(ex.drop_duplicates(["rid_s", "ban"]), on=["rid_s", "ban"], how="left")
    d.loc[d.yon_course <= 0, "yon_course"] = np.nan
    d.loc[d.agari_idx <= -90, "agari_idx"] = np.nan
    d.loc[d.agari_idx > 60, "agari_idx"] = np.nan
    for c in ("deokure", "furi", "mae_furi", "naka_furi", "ato_furi"):
        d[c] = d[c].fillna(0.0)
    c4 = d.c4.where(d.c4 > 0)
    d["pf4"] = (c4 - 1) / (d.n_start - 1).clip(lower=1)
    d["gain"] = d.pf - d.pf4                      # 正 = 1角→4角で前へ上げた
    d["deok_flag"] = (d.deokure > 0).astype(float)
    d = d.sort_values(["ped", "ymd"]).reset_index(drop=True)
    d["deokv"] = d.deok_flag.where(d.valid_run); d["ycv"] = d.yon_course.where(d.valid_run)
    d["gainv"] = d.gain.where(d.valid_run); d["agv"] = d.agari_idx.where(d.valid_run)
    g = d.groupby("ped", sort=False)
    lag = {k: np.column_stack([g[k].shift(i).values for i in range(1, 6)]) for k in ("deokv", "ycv", "gainv", "agv")}
    with np.errstate(all="ignore"):
        import warnings
        warnings.simplefilter("ignore")
        d["deok5"] = np.nanmean(lag["deokv"], axis=1)
        d["yc5"] = np.nanmean(lag["ycv"], axis=1)
        d["gain5"] = np.nanmean(lag["gainv"], axis=1)
        d["agari3"] = np.nanmean(lag["agv"][:, :3], axis=1)
    # 今日の実際 (オラクル列)
    d["ten_today_mean"] = d.ten_idx.groupby(d.rid_s).transform("mean")
    d["n_front_today"] = (d.pf <= 0.15).astype(float).groupby(d.rid_s).transform("sum")
    d["pf_dev"] = d.pf - d.pf5
    d["ban_rel"] = d.ban / d.n_start
    d["win"] = (d.jyun == 1).astype(int)
    return d


PRE = ["ten3", "tenmax5", "pf5", "lead5", "pidx3", "idm5", "y5", "nprev", "ddist", "dist", "td", "venue", "n_start",
       "ban_rel", "deok5", "yc5", "gain5", "agari3"]
ORA = {
    "R": ["q", "racep_idx", "ten_today_mean", "n_front_today"],
    "P": ["pf", "lead", "pf_dev", "deokure"],
    "T": ["yon_course", "furi", "mae_furi", "naka_furi", "ato_furi"],
}
SETS = {"pre": [], "R": ["R"], "P": ["P"], "T": ["T"], "RP": ["R", "P"], "RPT": ["R", "P", "T"]}
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=0)


def fit_predict(d, feats, target="win", params=PARAMS):
    tr = d[(d.year >= 2013) & (d.year <= 2020)]
    va = d[(d.year >= 2021) & (d.year <= 2022)]
    te = d[d.year >= 2023]
    dtr = lgb.Dataset(tr[feats], tr[target]); dva = lgb.Dataset(va[feats], va[target])
    m = lgb.train(params, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    return m, m.predict(te[feats], raw_score=True), te


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    d = pfx.build_runs()
    d = add_fine(d)
    d = d[d.year >= 2013].reset_index(drop=True)
    log(f"runs {len(d):,} races {d.rid_s.nunique():,}")
    res = {"experiment": "pace_granularity_ceiling", "date": "2026-11-15", "sets": {}, "gates": {}}
    z = d.loc[d.year >= 2023, ["rid_s", "ban"]].reset_index(drop=True)
    models = {}
    for name, groups in SETS.items():
        feats = PRE + [c for gname in groups for c in ORA[gname]]
        m, zz, te = fit_predict(d, feats)
        models[name] = (m, feats)
        z[f"z_{name}"] = zz
        gain = pd.Series(m.feature_importance("gain"), index=feats)
        res["sets"][name] = {"features": feats, "best_iter": int(m.best_iteration),
                             "gain_share_oracle": float(gain[[c for c in feats if c not in PRE]].sum() / gain.sum()) if groups else 0.0}
        log(f"[model {name}] iters {m.best_iteration} oracle gain share {res['sets'][name]['gain_share_oracle']:.2f}")
    for name in SETS:
        res["gates"][name] = pfx.short(gc.gate(z, [f"z_{name}"], name=name))

    # 上限 = z_X − z_pre の M ゲート差
    base_m = res["gates"]["pre"]["M"]["d_nll"]
    res["ceiling_vs_pre_M"] = {k: res["gates"][k]["M"]["d_nll"] - base_m for k in SETS if k != "pre"}
    log("[ceiling M − pre] " + json.dumps({k: round(v, 5) for k, v in res["ceiling_vs_pre_M"].items()}))

    # 予測可能性: 上限が 0.002 を超えた集合のオラクル列を発走前の情報で予測し、置き換えた z を測る
    te = d[d.year >= 2023].reset_index(drop=True)
    # レース内の相手関係 (発走前に分かる)
    d["ten3_rank"] = d.groupby("rid_s")["ten3"].rank(ascending=False, pct=True)
    d["pf5_rank"] = d.groupby("rid_s")["pf5"].rank(pct=True)
    d["n_lead_others"] = (d.lead5 >= 0.4).astype(float).groupby(d.rid_s).transform("sum") - (d.lead5 >= 0.4).astype(float)
    d["ten3_top_others"] = d.groupby("rid_s")["ten3"].transform("max")
    PREF = PRE + ["ten3_rank", "pf5_rank", "n_lead_others", "ten3_top_others"]
    res["predictability"] = {}
    for gname in ("R", "P", "T"):
        if res["ceiling_vs_pre_M"][gname] > -0.002:
            continue
        te = d[d.year >= 2023].reset_index(drop=True)
        rep = te.copy()
        r2 = {}
        for c in ORA[gname]:
            ok = d[c].notna()
            reg = dict(PARAMS, objective="regression")
            dd = d[ok]
            m, pred, tte = fit_predict(dd, PREF, target=c, params=reg)
            t = (tte.year >= 2024).values
            y = tte[c].values[t]
            r2[c] = float(1 - np.mean((y - pred[t]) ** 2) / np.var(y))
            mp = dict(zip(zip(tte.rid_s, tte.ban), pred))
            rep[c] = [mp.get(k, np.nan) for k in zip(rep.rid_s, rep.ban)]
        m, feats = models[gname]
        z[f"z_{gname}_hat"] = m.predict(rep[feats], raw_score=True)
        res["predictability"][gname] = {"r2_test2024_25": r2,
                                        "gate": pfx.short(gc.gate(z, [f"z_{gname}_hat"], name=f"{gname}_hat"))}
        log(f"[predict {gname}] R2 {json.dumps({k: round(v, 3) for k, v in r2.items()})}")

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"wrote {OUT}")


if __name__ == "__main__":
    main()
