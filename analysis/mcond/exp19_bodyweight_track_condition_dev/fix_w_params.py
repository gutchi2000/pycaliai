# -*- coding: utf-8 -*-
"""
fix_w_params.py — EXP19 Stage 0: MAD floor・winsorize 幅を**分布だけ**から固定する (SPEC §4.1)
===========================================================================================
結果列を読まない (torch 構造 loader だけ)。使う行は 2013-2018 の measured・履歴 >= 2 の行だけ
(評価年 2019-2023 の分布も見ない)。規則は実行前に本 docstring で固定した:

  MAD floor   = 体重の記録刻み (2 kg) と、≤2018 の生 MAD 分布の 10 パーセンタイルの大きい方
                (MAD = 0 で z が発散するのを防ぐための下限。結果との関係は見ない)
  winsorize   = 上の floor を入れた |bw_robust_z5| の ≤2018 分布の 99.5 パーセンタイルを 0.5 刻みで切り上げた値
                (bw_sex_age_z にも同じ幅を使う)
  履歴必要数  = 2 (SPEC で既定。変更しない)
出力: out/w_param_fixing.json
"""
from __future__ import annotations

import json
import math
import time

import numpy as np

from . import features as F
from .loaders import OUT, load_torch_struct, loader_sha256

RULE_GRANULARITY_KG = 2.0
RULE_MAD_PCTL = 10
RULE_WINSOR_PCTL = 99.5
RULE_WINSOR_STEP = 0.5
FIT_MAX_YEAR = 2018


def main():
    t0 = time.time()
    t = load_torch_struct()
    t["measured"] = t["kg"].between(F.MEASURED_LO, F.MEASURED_HI)
    hist = F.history_table(t)
    h = F.asof_history(t, hist)
    sel = (t["year"] <= FIT_MAX_YEAR) & t["measured"] & (h["h_n"].fillna(0) >= F.MIN_HIST)
    mad = h.loc[sel, "h_mad5"].to_numpy()
    mad_q = {str(q): float(np.percentile(mad, q)) for q in (1, 5, 10, 25, 50, 75, 90, 99)}
    floor = max(RULE_GRANULARITY_KG, float(np.percentile(mad, RULE_MAD_PCTL)))
    z = (t.loc[sel, "kg"] - h.loc[sel, "h_med5"]) / (F.MAD_SCALE * np.maximum(mad, floor))
    az = np.abs(z.to_numpy())
    q995 = float(np.percentile(az, RULE_WINSOR_PCTL))
    winsor = math.ceil(q995 / RULE_WINSOR_STEP) * RULE_WINSOR_STEP
    res = {
        "role": "MAD floor と winsorize 幅を分布だけで固定 (結果列不使用、2013-2018 のみ)",
        "rules": {"mad_floor": f"max({RULE_GRANULARITY_KG} kg granularity, p{RULE_MAD_PCTL} of raw MAD)",
                  "winsor": f"ceil(p{RULE_WINSOR_PCTL} of |z| / {RULE_WINSOR_STEP}) * {RULE_WINSOR_STEP}",
                  "min_history": F.MIN_HIST, "max_history": F.MAX_HIST, "mad_scale": F.MAD_SCALE},
        "n_rows_used": int(sel.sum()), "years_used": [2013, FIT_MAX_YEAR],
        "raw_mad_quantiles_kg": mad_q, "share_raw_mad_zero": float(np.mean(mad == 0)),
        "share_raw_mad_below_floor": float(np.mean(mad < floor)),
        "abs_z_quantiles": {str(q): float(np.percentile(az, q)) for q in (50, 90, 99, 99.5, 99.9)},
        "MAD_FLOOR_KG": floor, "WINSOR_Z": winsor,
        "share_clipped_at_winsor": float(np.mean(az > winsor)),
        "loader_sha256": loader_sha256(), "elapsed_sec": round(time.time() - t0, 1),
    }
    (OUT / "w_param_fixing.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
