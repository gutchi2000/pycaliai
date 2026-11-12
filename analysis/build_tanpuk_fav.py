# -*- coding: utf-8 -*-
"""
build_tanpuk_fav.py — 時系列オッズ (TANPUK) から「発走前スナップ時点の 1 番人気」の単勝・複勝 Lo/Hi と確定値を 1 レース 1 行にする (2026-11-12)
入力: data/Time _series_odds/TANPUK_*.csv (2011-2025)。区分 1 = 途中 / 4 = 確定。
pre = 当日の区分 1 のうち、確定時刻の 15 分以上前で最も遅いもの (発走の約 26-30 分前、analysis/mcond/market.py と同じ定義)。
出力: data/_joint/tanpuk_fav.parquet
  rid, year, date, n, ban (pre の 1 番人気), tan_pre, flo_pre, fhi_pre, tan_fin, flo_fin, fhi_fin, min_before,
  o2_pre..o4_pre, flo2_pre, flo3_pre, pool_tan_pre, pool_fuku_pre, pool_tan_fin, pool_fuku_fin, top3, pay (確定複勝払戻、外れ 0)
"""
from __future__ import annotations

import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
ODIR = BASE / "data" / "Time _series_odds"
MKT = BASE / "data" / "_joint" / "mkt_horses.parquet"
OUT = BASE / "data" / "_joint" / "tanpuk_fav.parquet"
NON_DIGIT = re.compile(r"[^0-9]")


def main() -> int:
    files = sorted(glob.glob(str(ODIR / "TANPUK_*.csv")))
    tp = pd.concat([pd.read_csv(f, encoding="cp932", low_memory=False) for f in files], ignore_index=True)
    c = list(tp.columns)
    RID, KB, TM, NF, PT, PF = c[0], c[1], c[2], c[3], c[4], c[5]
    tp["rid"] = tp[RID].astype(str).map(lambda s: NON_DIGIT.sub("", s)[:16])
    tp = tp[tp.rid.str.len() == 16].copy()
    tp["year"] = tp.rid.str[:4].astype(int)
    tp = tp[tp.year >= 2013]
    tm = pd.to_numeric(tp[TM], errors="coerce").fillna(0).astype(np.int64)
    tp["ts"] = pd.to_datetime(dict(year=tp.year, month=tm // 1000000, day=(tm // 10000) % 100, hour=(tm // 100) % 100, minute=tm % 100), errors="coerce")
    tp["kb"] = pd.to_numeric(tp[KB], errors="coerce")
    tp["race_day"] = pd.to_datetime(tp.rid.str[:8], format="%Y%m%d", errors="coerce")
    fin = tp[tp.kb == 4].drop_duplicates("rid", keep="last").set_index("rid")
    mid = tp[(tp.kb == 1) & (tp.ts.dt.normalize() == tp.race_day)].copy()
    mid = mid.join(fin[["ts"]].rename(columns={"ts": "ts_fin"}), on="rid")
    mid["gap"] = (mid.ts_fin - mid.ts).dt.total_seconds() / 60
    mid = mid[mid.gap >= 15].sort_values(["rid", "ts"]).drop_duplicates("rid", keep="last").set_index("rid")
    rids = mid.index.intersection(fin.index)
    mid, fin = mid.loc[rids], fin.loc[rids]

    def mat(df, suffix):
        cols = [f"{i}{suffix}" for i in range(1, 19)]
        m = df.reindex(columns=[x for x in df.columns if NON_DIGIT.sub("", str(x)) and str(x).strip().endswith(suffix) and str(x).strip()[:-len(suffix)].isdigit()])
        m.columns = [int(str(x).strip()[:-len(suffix)]) for x in m.columns]
        return m.apply(pd.to_numeric, errors="coerce").reindex(columns=range(1, 19)).values.astype(float)

    T, L, H = mat(mid, "単"), mat(mid, "複Lo"), mat(mid, "複Hi")
    Tf, Lf, Hf = mat(fin, "単"), mat(fin, "複Lo"), mat(fin, "複Hi")
    Tm = np.where((T > 0), T, np.inf)
    order = np.argsort(Tm, axis=1, kind="stable")
    r = np.arange(len(rids))
    fav = order[:, 0]
    out = pd.DataFrame({"rid": rids, "ban": fav + 1, "tan_pre": T[r, fav], "flo_pre": L[r, fav], "fhi_pre": H[r, fav],
                        "tan_fin": Tf[r, fav], "flo_fin": Lf[r, fav], "fhi_fin": Hf[r, fav], "min_before": mid.gap.values,
                        "o2_pre": T[r, order[:, 1]], "o3_pre": T[r, order[:, 2]], "o4_pre": T[r, order[:, 3]],
                        "flo2_pre": L[r, order[:, 1]], "flo3_pre": L[r, order[:, 2]],
                        "pool_tan_pre": pd.to_numeric(mid[PT], errors="coerce").values, "pool_fuku_pre": pd.to_numeric(mid[PF], errors="coerce").values,
                        "pool_tan_fin": pd.to_numeric(fin[PT], errors="coerce").values, "pool_fuku_fin": pd.to_numeric(fin[PF], errors="coerce").values,
                        "n_pre": (T > 0).sum(1)})
    mk = pd.read_parquet(MKT, columns=["rid", "ban", "n", "top3", "fpay"])
    out = out.merge(mk, on=["rid", "ban"], how="inner")
    out["pay"] = out.fpay.fillna(0.0)
    out["year"] = out.rid.str[:4].astype(int); out["date"] = out.rid.str[:8]
    out = out[np.isfinite(out.tan_pre) & (out.tan_pre > 0)]
    out.drop(columns=["fpay"]).to_parquet(OUT)
    print(f"{len(out):,} R, {out.year.min()}-{out.year.max()}, min_before median {out.min_before.median():.0f} 分")
    print(out.groupby("year").size().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
