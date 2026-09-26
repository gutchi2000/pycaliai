# -*- coding: utf-8 -*-
"""
power_data.py — EXP19 Stage 0: 検出力・floor・placebo 構造用の race 配列 (結果ラベルを含めない)
==============================================================================================
主母集団の定義 (EXP16A 正式 set + 全 starter が torch 行と pre/terminal 単勝オッズを持つ) を 2016-2023 に適用し、
race ごとに starter (terminal 単勝 > 1.0、馬番順) の配列を作る:
  log m_pre / log m_term (比例 de-vig)、terminal 単勝オッズ (決済用)、s_clean (R0-clean-nobw OOF の 5 seed 平均)、
  W 設計行列 (主 7 + 欠損指示子 3)、WP 8 列、race 属性 (年・暦日・競馬場・開催回・月・芝ダ・頭数・年齢構成帯・WP 可否)
**着順・勝馬は入れない** (floor/power は合成 winner だけを使う)。
出力: data/_research/mcond/exp19/power_arrays.npz、out/power_data_meta.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import features as F
from .loaders import BASE, OUT, RESEARCH, load_market, load_torch_struct, sha256_file

YEARS = list(range(2016, 2024))
REF16 = BASE / "data" / "_research" / "mcond" / "exp16a" / "official_rids_by_year.json"
SCORES = RESEARCH / "scores"
ARR = RESEARCH / "power_arrays.npz"
FEAT = RESEARCH / "features_2013_2023.parquet"


def age_band(ages: np.ndarray) -> str:
    a = set(int(x) for x in ages if x == x)
    if a == {2}:
        return "2yo"
    if a == {3}:
        return "3yo"
    if min(a) <= 3:
        return "3yo_plus_mixed"
    return "4yo_plus"


def field_band(n: int) -> str:
    return "5-8" if n <= 8 else ("9-12" if n <= 12 else ("13-15" if n <= 15 else "16-18"))


def main():
    t0 = time.time()
    ref = json.loads(REF16.read_text(encoding="utf-8"))
    feat = pd.read_parquet(FEAT)
    feat["key"] = feat["rid16"] + "_" + feat["ban"].astype(str)
    fx = feat.set_index("key")
    wd = F.design_w(fx)
    wcols = list(wd.columns)
    # s_clean = 5 seed 平均の OOF score (年 Y の予測)
    sc = {}
    for Y in YEARS:
        fs = sorted(SCORES.glob(f"N1_Y{Y}_s*.npz"))
        assert len(fs) == 5, (Y, len(fs))
        keys, vals = None, []
        for f in fs:
            z = np.load(f, allow_pickle=True)
            if keys is None:
                keys = z["key"]
            assert np.array_equal(keys, z["key"])
            vals.append(z["score"])
        for k, v in zip(keys, np.mean(vals, axis=0)):
            sc[str(k)] = float(v)
    idx, Wm = load_market(range(2016, 2024))
    t = load_torch_struct()
    kaisai = t.drop_duplicates("rid16").set_index("rid16")["開催"].astype(str)
    recs, miss = [], {"no_market": 0, "pre_incomplete": 0, "missing_feature_row": 0, "missing_s_clean": 0}
    for Y in YEARS:
        for rid in sorted(ref[str(Y)]):
            if rid not in idx.index:
                miss["no_market"] += 1; continue
            r = idx.loc[rid]
            wt = Wm[int(r["term_i"])]
            bans = np.flatnonzero(np.isfinite(wt) & (wt > 1.0)) + 1
            if r["pre_i"] < 0:
                miss["pre_incomplete"] += 1; continue
            wpre = Wm[int(r["pre_i"])][bans - 1]
            if not np.all(np.isfinite(wpre) & (wpre > 1.0)):
                miss["pre_incomplete"] += 1; continue
            keys = [f"{rid}_{b}" for b in bans]
            if not all(k in fx.index for k in keys):
                miss["missing_feature_row"] += 1; continue
            if not all(k in sc for k in keys):
                miss["missing_s_clean"] += 1; continue
            recs.append((Y, rid, bans, wpre, wt[bans - 1], keys))
    cnt = np.array([len(x[2]) for x in recs])
    off = np.concatenate([[0], np.cumsum(cnt)])
    keys = [k for x in recs for k in x[5]]
    inv_pre = np.concatenate([1 / x[3] for x in recs])
    inv_term = np.concatenate([1 / x[4] for x in recs])
    rr = np.repeat(np.arange(len(recs)), cnt)

    def norm_log(inv):
        s = np.add.reduceat(inv, off[:-1])
        return np.log(inv) - np.log(s[rr])
    sub = fx.loc[keys]
    race_attr = []
    for x in recs:
        g = fx.loc[x[5]]
        race_attr.append({"year": x[0], "rid": x[1], "day": int(x[1][:8]), "venue": str(g["場所"].iloc[0]),
                          "kaisai": str(kaisai.get(x[1], "")), "month": int(x[1][4:6]),
                          "surface": str(g["surface"].iloc[0]), "n": len(x[2]),
                          "age_band": age_band(pd.to_numeric(g["年齢"], errors="coerce").to_numpy()),
                          "field_band": field_band(len(x[2])),
                          "wp_ok": bool(g["p_available"].iloc[0] is True or g["p_available"].iloc[0] == True)
                          and bool(g["all_measured_race"].iloc[0])})
    A = pd.DataFrame(race_attr)
    np.savez_compressed(
        ARR, off=off, log_m_pre=norm_log(inv_pre), log_m_term=norm_log(inv_term),
        odds_term=np.concatenate([x[4] for x in recs]), s_clean=np.array([sc[k] for k in keys]),
        W=wd.loc[keys].to_numpy(dtype=float), WP=sub[F.WP_COLS].to_numpy(dtype=float),
        cushion_z=sub["cushion_z"].to_numpy(dtype=float), moist_gp_z=sub["moist_gp_z"].to_numpy(dtype=float),
        year=A["year"].to_numpy(), day=A["day"].to_numpy(), rid=A["rid"].to_numpy(), venue=A["venue"].to_numpy(),
        kaisai=A["kaisai"].to_numpy(), month=A["month"].to_numpy(), surface=A["surface"].to_numpy(),
        n=A["n"].to_numpy(), age_band=A["age_band"].to_numpy(), field_band=A["field_band"].to_numpy(),
        wp_ok=A["wp_ok"].to_numpy(), w_cols=np.array(wcols), wp_cols=np.array(F.WP_COLS),
        ban=np.concatenate([x[2] for x in recs]))
    meta = {"races": len(recs), "rows": int(off[-1]), "by_year": A.groupby("year").size().to_dict(),
            "wp_ok_by_year": A.groupby("year")["wp_ok"].mean().round(4).to_dict(), "excluded": miss,
            "w_cols": wcols, "wp_cols": F.WP_COLS, "s_clean": "mean of 5 seed OOF scores (R0-clean-nobw)",
            "arrays_sha256": sha256_file(ARR), "no_outcome_columns": True, "elapsed_sec": round(time.time() - t0, 1)}
    (OUT / "power_data_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str),
                                              encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
