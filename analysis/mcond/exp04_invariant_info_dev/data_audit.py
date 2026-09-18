# -*- coding: utf-8 -*-
"""
data_audit.py — Gate 0 データ監査: C1候補 (v6の120特徴) の年別被覆率から除外列を機械的に決める
================================================================================================
除外規則 (評価前に固定、結果を見て閾値を変えない):
  R1 恒常欠損   : 2016-2025 通算の被覆率 < 5%  (収集自体ができていない列)
  R2 期間内シフト: 2016-2025 の年別被覆率の最大-最小 > 30pt  (収集仕様が期間途中で変わった列)
  R3 識別子     : 名称が「レースID」「血統登録番号」を含む列 (集約統計量ではなく生キー)
  R4 上書き列   : 馬主(最新/仮想) (EXP01/EXP02の監査で当時値ではないと確認済み、再確認もここで行う)
出力: out/data_audit.json, out/coverage_by_year.csv
実行: python -m analysis.mcond.exp04_invariant_info_dev.data_audit
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
MASTER = BASE / "data/master_v2_20130105-20251228.csv"

R1_MIN_COV = 0.05
R2_MAX_SHIFT = 0.30


def main() -> None:
    OUT.mkdir(exist_ok=True)
    b = joblib.load(BASE / "models/unified_rank_v6.pkl")
    feats = b["feature_cols"]
    print(f"v6 特徴数: {len(feats)}")

    df = pd.read_csv(MASTER, encoding="utf-8-sig", low_memory=False, usecols=["日付"] + feats)
    df["year"] = df["日付"].astype(str).str[:4].astype(int)
    df = df[(df.year >= 2016) & (df.year <= 2025)]

    def cov(s: pd.Series) -> float:
        if not pd.api.types.is_numeric_dtype(s):
            ss = s.astype(str)
            valid = s.notna() & (ss != "") & (ss != "nan")
        else:
            valid = s.notna()
        return float(valid.mean())

    rows = []
    by_year = {}
    for c in feats:
        yc = df.groupby("year")[c].apply(cov)
        by_year[c] = yc.to_dict()
        total = cov(df[c])
        shift = float(yc.max() - yc.min())
        rows.append({"feature": c, "total_coverage": total, "year_min": float(yc.min()),
                     "year_max": float(yc.max()), "shift": shift})
    cy = pd.DataFrame(rows).sort_values("shift", ascending=False)
    cy.to_csv(OUT / "coverage_by_year.csv", index=False, encoding="utf-8-sig")
    byyear_df = pd.DataFrame(by_year).T
    byyear_df.index.name = "feature"
    byyear_df.to_csv(OUT / "coverage_by_year_detail.csv", encoding="utf-8-sig")

    excl = {}
    for r in rows:
        c = r["feature"]
        reasons = []
        if r["total_coverage"] < R1_MIN_COV:
            reasons.append(f"R1恒常欠損(通算被覆{r['total_coverage']:.3f}<{R1_MIN_COV})")
        if r["shift"] > R2_MAX_SHIFT:
            reasons.append(f"R2期間内シフト(年別被覆差{r['shift']:.3f}>{R2_MAX_SHIFT})")
        if ("レースID" in c) or ("血統登録番号" in c):
            reasons.append("R3識別子")
        if c == "馬主(最新/仮想)":
            reasons.append("R4上書き列(EXP01/EXP02監査で確認済み)")
        if reasons:
            excl[c] = reasons

    kept = [c for c in feats if c not in excl]
    result = {"n_v6_features": len(feats), "n_excluded": len(excl), "n_kept": len(kept),
              "excluded": excl, "kept": kept,
              "rules": {"R1_min_total_coverage": R1_MIN_COV, "R2_max_year_shift": R2_MAX_SHIFT}}
    (OUT / "data_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n除外: {len(excl)} / {len(feats)}")
    for c, rs in excl.items():
        print(f"  {c:<28} {', '.join(rs)}")
    print(f"\nC1採用: {len(kept)} 特徴")


if __name__ == "__main__":
    main()
