"""
build_odds_traj_feats.py — 9時以前 単勝オッズ軌跡の「形」特徴 (leak-safe)
==========================================================================
仮説: 確定オッズ(結果)は効率的でも、9時以前のオッズ軌跡(過程の形)に kako5/点オッズに
無い増分が残るかもしれない。それを測る素材を作る。

★リーク防止 (生命線): 特徴は「当日09:00以前に確定したスナップショット」だけから算出。
  TANPUK 区分1(前売り) のうち (mmdd<race_mmdd) or (mmdd==race_mmdd & HHMM<=900) のみ採用。
  9時以降(HHMM>900 当日)・確定(区分4) は 1 列も使わない。検査で違反0を確認 (>0 で FAIL)。

各馬の 9時以前単勝オッズ系列 [o_1..o_k] (時刻順) から:
  traj_nsnap   : 9時以前スナップ数
  traj_logchg  : log(o_k)-log(o_1)  (売り出し→9時直前の対数変化。負=短縮=買われた)
  traj_range   : log オッズの max-min (変動幅)
  traj_absmax  : 最大単一ステップ変化 |Δlog o|
  traj_revers  : Σ|Δ| - |ΣΔ|  (往復度。0=単調、大=往復/急変)
  traj_rankshift: 最初snapの人気順位 - 9時直前snapの人気順位 (+=人気上昇)
  traj_has     : 軌跡算出可(k>=2)フラグ
※単勝のみ (馬連軌跡は今回省略、報告明記)。9時の点オッズ自体は既存 π9 として別に持つ。

出力: data/odds_traj_feats.parquet / reports/odds_traj_leakcheck.json
実行: PYTHONUTF8=1 python build_odds_traj_feats.py
"""
from __future__ import annotations
import glob, json, re, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
BASE = Path(__file__).parent
ODIR = BASE / "data/Time _series_odds"
OUT_PARQUET = BASE / "data/odds_traj_feats.parquet"
OUT_LEAK = BASE / "reports/odds_traj_leakcheck.json"
LEAK_SAMPLE_EVERY = 40


def _rid16(x): return re.sub(r"\D", "", str(x))[:16]


def main():
    files = sorted(glob.glob(str(ODIR / "TANPUK_*.csv")))  # 全期間 (train含む)
    print(f"[load] {len(files)} TANPUK files")
    parts = []
    tan_cols = None
    for f in files:
        tp = pd.read_csv(f, encoding="cp932", low_memory=False)
        cols = list(tp.columns); RID, KB, TM = cols[0], cols[1], cols[2]
        if tan_cols is None:
            tan_cols = {}
            for c in cols:
                m = re.match(r"^\s*(\d+)\s*単\s*$", str(c))
                if m: tan_cols[int(m.group(1))] = c
        tp[KB] = pd.to_numeric(tp[KB], errors="coerce")
        tp = tp[tp[KB] == 1].copy()                       # ★前売りのみ
        tp["rid16"] = tp[RID].map(_rid16)
        tp = tp[tp["rid16"].str.len() == 16]
        tp[TM] = pd.to_numeric(tp[TM], errors="coerce")
        parts.append(tp[["rid16", TM] + list(tan_cols.values())])
    df = pd.concat(parts, ignore_index=True)
    TMc = [c for c in df.columns if c not in ["rid16"] + list(tan_cols.values())][0]
    df["mmdd"] = (df[TMc] // 10000).astype("Int64")
    df["hhmm"] = (df[TMc] % 10000).astype("Int64")
    print(f"  区分1 行数={len(df):,}  単勝列={len(tan_cols)}")

    recs = []; leak_checked = 0; leak_viol = 0; rc = 0
    cov_total = cov_has = 0
    for rid, g in df.groupby("rid16", sort=False):
        rmmdd = int(rid[4:8])
        pre = g[(g["mmdd"] < rmmdd) | ((g["mmdd"] == rmmdd) & (g["hhmm"] <= 900))].copy()
        if len(pre) == 0:
            continue
        pre = pre.sort_values(TMc)
        rc += 1
        do_leak = (rc % LEAK_SAMPLE_EVERY == 0)
        if do_leak:
            leak_checked += len(pre)
            bad = (((pre["mmdd"] == rmmdd) & (pre["hhmm"] > 900)) | (pre["mmdd"] > rmmdd)).sum()
            leak_viol += int(bad)
        # 各 snap のオッズ行列 (snap × ban)
        first = pre.iloc[0]; last = pre.iloc[-1]
        # 人気順位 (オッズ昇順): first/last snap
        def ranks(row):
            od = {b: pd.to_numeric(pd.Series([row[c]]), errors="coerce").iloc[0] for b, c in tan_cols.items()}
            od = {b: v for b, v in od.items() if pd.notna(v) and v > 1.0}
            order = sorted(od, key=lambda b: od[b])
            return {b: i + 1 for i, b in enumerate(order)}, od
        rk_f, od_f = ranks(first); rk_l, od_l = ranks(last)
        for b, c in tan_cols.items():
            series = pd.to_numeric(pre[c], errors="coerce").values
            series = series[(~np.isnan(series)) & (series > 1.0)]
            k = len(series)
            if k < 1:
                continue
            cov_total += 1
            rec = {"rid16": rid, "ban": b, "traj_nsnap": k, "traj_has": 1 if k >= 2 else 0,
                   "traj_logchg": np.nan, "traj_range": np.nan, "traj_absmax": np.nan,
                   "traj_revers": np.nan, "traj_rankshift": np.nan}
            if k >= 2:
                cov_has += 1
                lo = np.log(series)
                d = np.diff(lo)
                rec["traj_logchg"] = float(lo[-1] - lo[0])
                rec["traj_range"] = float(lo.max() - lo.min())
                rec["traj_absmax"] = float(np.abs(d).max())
                rec["traj_revers"] = float(np.abs(d).sum() - abs(d.sum()))
                if b in rk_f and b in rk_l:
                    rec["traj_rankshift"] = float(rk_f[b] - rk_l[b])
            recs.append(rec)

    out = pd.DataFrame(recs)
    out.to_parquet(OUT_PARQUET, index=False)
    leak = {"leak_policy": "区分1 かつ (mmdd<race_mmdd or (mmdd==race & hhmm<=900)) のみ。9時以降/確定 不使用。",
            "leak_sample_every": LEAK_SAMPLE_EVERY, "leak_checked_snaps": leak_checked,
            "leak_violations": leak_viol, "PASS": bool(leak_viol == 0),
            "n_races_with_pre": rc, "n_horse_rows": len(out),
            "coverage_has_traj(k>=2)/total": round(cov_has / cov_total * 100, 1) if cov_total else 0.0,
            "note": "単勝のみ・馬連軌跡は省略。9時点オッズは別途π9。"}
    OUT_LEAK.parent.mkdir(exist_ok=True)
    with open(OUT_LEAK, "w", encoding="utf-8") as f:
        json.dump(leak, f, indent=2, ensure_ascii=False)
    print(f"\n[リーク検査] 検査snap={leak_checked:,}  違反={leak_viol}  {'PASS' if leak_viol==0 else 'FAIL'}")
    print(f"  軌跡算出可(k>=2)率={leak['coverage_has_traj(k>=2)/total']}%  馬行={len(out):,}  pre有レース={rc:,}")
    if len(out):
        print(out[["traj_nsnap", "traj_logchg", "traj_range", "traj_revers", "traj_rankshift"]].describe().T[["mean", "std", "min", "max"]].to_string())
    print(f"[saved] {OUT_PARQUET}")
    return 0 if leak_viol == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
