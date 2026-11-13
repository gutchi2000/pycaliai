# -*- coding: utf-8 -*-
"""
volatility_option_verify.py — ムラ馬 (log σ_h) の M ゲート通過を、既知の癖の代理でないか確かめる (2026-11-14、事後検証)
対照に入れるもの (どれも発走前に分かる): キャリアの深さ log(1+n)・n==0、前走の相対着順、前走が自己平均からどれだけ下振れたか (直近の IDM 差)。
土台 [確定市場 + v7 + 対照] に log σ_h を足して NLL が下がるか。valid=2023 で fit、test=2024-25、開催日 bootstrap。
実行: E:/PyCaLiAI/venv311/Scripts/python.exe lab/experiments/volatility_option_verify.py
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate_common as gc

def run(b, base_cols, ctrl, z, label, rng):
    out = {}
    for sp in ("valid", "test"):
        x = b[b.split == sp].reset_index(drop=True); c = pd.factorize(x.rid_s)[0]
        out[sp] = (x, c, c.max() + 1, np.flatnonzero((x.jyun == 1).values))
    xv, cv, nv, wv = out["valid"]; xt, ct, nt, wt = out["test"]
    nll = {}
    for tag, cols in (("base", base_cols + ctrl), ("plus", base_cols + ctrl + z)):
        w = gc._fit(xv[cols].values, cv, nv, wv, 1e-4)
        lp = gc.race_logsoftmax(xt[cols].values @ w, ct, nt)
        nll[tag] = pd.Series(-lp[wt], index=ct[wt]).sort_index(); wz = w
    diff = (nll["plus"] - nll["base"]).values
    dates = xt.groupby(ct)["date"].first().sort_index().values
    ud, dc = np.unique(dates, return_inverse=True)
    s = np.bincount(dc, weights=diff, minlength=len(ud)); n = np.bincount(dc, minlength=len(ud)).astype(float)
    I = rng.integers(0, len(ud), (2000, len(ud))); v = s[I].sum(1) / n[I].sum(1)
    yr = pd.Series(dates).str[:4].values
    r = {"d_nll": float(diff.mean()), "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
         "w_z": float(wz[-1]), "by_year": {y: float(diff[yr == y].mean()) for y in sorted(set(yr))}}
    print(f"{label:46s} ΔNLL {r['d_nll']:+.5f} [{r['ci95'][0]:+.5f},{r['ci95'][1]:+.5f}]  w={r['w_z']:+.3f}  年別 {r['by_year']}", flush=True)
    return r

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    b = gc.load_base()
    f = pd.read_parquet(gc.REP / "vol_features.parquet")[["rid_s", "ban", "n", "logsig"]]
    b = b.merge(f, on=["rid_s", "ban"], how="left")
    s = gc.load_sed(["racekey", "ymd", "ban", "ped", "td", "jyun", "tosu", "ijou", "idm"])
    s = s[s.td.isin([1, 2])].sort_values(["ped", "date"]); g = s.groupby("ped")
    s["prev_rel"] = g.jyun.shift(1) / g.tosu.shift(1)
    s["prev_bad"] = (g.jyun.shift(1) >= 6).astype(float)
    s["last_dev"] = g.idm.shift(1) - g.idm.transform(lambda x: x.shift(1).expanding().mean().shift(0))   # 前走 IDM − それまでの平均 (前走を含む平均との差)
    s["ncar"] = g.cumcount()
    b = b.merge(s[["rid_s", "ban", "prev_rel", "prev_bad", "last_dev", "ncar"]].drop_duplicates(["rid_s", "ban"]), on=["rid_s", "ban"], how="left")
    b["lognc"] = np.log1p(b["ncar"]); b["debut"] = (b["ncar"].fillna(0) == 0).astype(float); b["n0"] = (b["n"].fillna(0) == 0).astype(float)
    b["logn"] = np.log1p(b["n"])
    cols = ["logsig", "lognc", "debut", "n0", "logn", "prev_rel", "prev_bad", "last_dev"]
    for c in cols:
        b[c] = pd.to_numeric(b[c], errors="coerce")
        b[c] = b[c].fillna(b.groupby("rid_s")[c].transform("mean")).fillna(0.0)
        b[c] = b[c] - b.groupby("rid_s")[c].transform("mean")
        sd = b.loc[b.split == "valid", c].std(); b[c] = b[c] / (sd if sd > 0 else 1)
    rng = np.random.default_rng(0); res = {}
    M = ["lk", "lm7"]
    res["M_no_ctrl"] = run(b, M, [], ["logsig"], "M: 対照なし (再現)", rng)
    res["M_career"] = run(b, M, ["lognc", "debut", "n0", "logn"], ["logsig"], "M: + キャリアの深さ", rng)
    res["M_prev"] = run(b, M, ["prev_rel", "prev_bad", "last_dev"], ["logsig"], "M: + 前走着順・前走の下振れ", rng)
    res["M_all"] = run(b, M, ["lognc", "debut", "n0", "logn", "prev_rel", "prev_bad", "last_dev"], ["logsig"], "M: + 全対照", rng)
    res["M_ctrl_only_career"] = run(b, M, [], ["lognc", "debut", "n0", "logn"], "M: キャリアの深さだけを z に", rng)
    res["M_ctrl_only_prev"] = run(b, M, [], ["prev_rel", "prev_bad", "last_dev"], "M: 前走系だけを z に", rng)
    res["P_all"] = run(b, ["lm7"], ["lognc", "debut", "n0", "logn", "prev_rel", "prev_bad", "last_dev"], ["logsig"], "P: + 全対照", rng)
    # n>=3 の馬だけで σ に情報がある部分: n<3 は 0 (レース平均) に置く
    b2 = b.copy(); raw = pd.read_parquet(gc.REP / "vol_features.parquet")[["rid_s", "ban", "n"]].rename(columns={"n": "n_raw"})
    b2 = b2.merge(raw, on=["rid_s", "ban"], how="left"); b2.loc[b2.n_raw.fillna(0) < 3, "logsig"] = 0.0
    res["M_all_n3"] = run(b2, M, ["lognc", "debut", "n0", "logn", "prev_rel", "prev_bad", "last_dev"], ["logsig"], "M: + 全対照、σ は差 3 本以上の馬だけ", rng)
    (gc.REP / "volatility_option_verify.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

if __name__ == "__main__":
    main()
