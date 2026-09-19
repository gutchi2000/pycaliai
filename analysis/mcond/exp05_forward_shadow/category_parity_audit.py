# -*- coding: utf-8 -*-
"""
category_parity_audit.py — v6 encoderを持つ全カテゴリ列のunknown_rate監査 (spec §3)
========================================================================================
export_weekly_marks.py と全く同じ経路 (predict_weekly.parse_csv → _SERVE_RENAME →
_CATEGORY_NORMALIZE) を複数週分回し、各カテゴリ列について
  unknown_rate = (encoder語彙に無い非欠損値の数) / (非欠損値の数)
を正規化前後で計算する。未知値を意味確認なしに既知カテゴリへ寄せることはしない
(このスクリプトは測定のみ、_CATEGORY_NORMALIZEへの追加はしない)。

出力: analysis/mcond/exp05_forward_shadow/CATEGORY_PARITY.csv
実行: python -m analysis.mcond.exp05_forward_shadow.category_parity_audit
"""
from __future__ import annotations
import sys
from pathlib import Path

import joblib
import pandas as pd

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
from predict_weekly import parse_csv  # noqa: E402
from category_normalize import NORMALIZERS  # noqa: E402  (正本、export_weekly_marks.pyと共有)

HERE = Path(__file__).resolve().parent
OUT = HERE / "CATEGORY_PARITY.csv"

_SERVE_RENAME = {
    "R": "Ｒ", "前走補正": "prev_hosei", "前走補9": "prev_hosei9",
    "trn_hanro_4f": "trnH_Time1", "trn_hanro_3f": "trnH_Time2",
    "trn_hanro_2f": "trnH_Time3", "trn_hanro_1f": "trnH_Time4",
    "trn_hanro_lap1": "trnH_Lap1", "trn_hanro_lap2": "trnH_Lap2",
    "trn_hanro_lap3": "trnH_Lap3", "trn_hanro_lap4": "trnH_Lap4",
    "trn_hanro_days": "trnH_days_ago",
    "trn_wc_5f": "trnW_5F", "trn_wc_4f": "trnW_4F", "trn_wc_3f": "trnW_3F",
    "trn_wc_lap1": "trnW_Lap1", "trn_wc_lap2": "trnW_Lap2",
    "trn_wc_lap3": "trnW_Lap3", "trn_wc_days": "trnW_days_ago",
}
# 高基数・時間とともに新規カテゴリが正当に増え続ける列 (新種牡馬のデビュー等)。
# 生産者は predict_weekly.py 自身が列非存在時に定数0を代入する既知の仕様
# (欠損の正規化とは別問題、CATEGORY_PARITY.mdに記載)。
HIGH_CARDINALITY_EXPECTED_DRIFT = {"種牡馬", "父タイプ名", "母父馬", "母父タイプ名",
                                   "馬主(最新/仮想)", "生産者", "騎手コード", "調教師コード"}


def sample_weeks(n: int = 10) -> list[str]:
    weekly = sorted(p.stem for p in (BASE / "data/weekly").glob("2026*.csv"))
    if len(weekly) <= n:
        return weekly
    step = len(weekly) / n
    return [weekly[int(i * step)] for i in range(n)]


def main() -> None:
    bundle = joblib.load(BASE / "models/unified_rank_v6.pkl")
    encs = bundle["encoders"]

    weeks = sample_weeks()
    print(f"対象週 ({len(weeks)}): {weeks}")

    counts_by_col_raw: dict[str, "pd.Series"] = {}
    for wk in weeks:
        path = BASE / "data/weekly" / f"{wk}.csv"
        try:
            df = parse_csv(path)
        except Exception as e:
            print(f"  {wk}: parse失敗 {e}")
            continue
        rename = {k: v for k, v in _SERVE_RENAME.items() if k in df.columns and v not in df.columns}
        df = df.rename(columns=rename)
        for col in encs:
            if col not in df.columns:
                continue
            s = df[col].astype(str)
            vc = s.value_counts()
            if col in counts_by_col_raw:
                counts_by_col_raw[col] = counts_by_col_raw[col].add(vc, fill_value=0)
            else:
                counts_by_col_raw[col] = vc
        print(f"  {wk}: {len(df)}頭 集計済み")

    rows = []
    for col, vc in counts_by_col_raw.items():
        classes = set(encs[col].classes_)
        norm_fn = NORMALIZERS.get(col)
        total_nonnull = vc[vc.index != "nan"].sum() if "nan" in vc.index else vc.sum()
        unknown_before_n = 0
        unknown_after_n = 0
        for raw_val, n in vc.items():
            if raw_val in ("nan", "__NaN__", ""):
                continue  # 実欠損 (encoder classes_ 側にも __NaN__ が用意されている前提)
            normalized = norm_fn(raw_val) if norm_fn else raw_val
            known_before = raw_val in classes
            known_after = normalized in classes
            if not known_before:
                unknown_before_n += int(n)
            if not known_after:
                unknown_after_n += int(n)
            rows.append({
                "feature_name": col, "raw_value": raw_val, "normalized_value": normalized,
                "count": int(n), "known_before": known_before, "known_after": known_after,
                "normalization_rule": ("NORMALIZERS(変化あり)" if norm_fn and normalized != raw_val else
                                       "恒等(規則なし)" if known_before else "未対応(未知のまま)"),
                "expected_drift_column": col in HIGH_CARDINALITY_EXPECTED_DRIFT,
            })
        denom = int(total_nonnull) if total_nonnull else 1
        for r in rows:
            if r["feature_name"] == col:
                r["unknown_rate_before_col"] = unknown_before_n / denom
                r["unknown_rate_after_col"] = unknown_after_n / denom

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT, index=False, encoding="utf-8-sig")

    summary = out_df.groupby("feature_name").agg(
        unknown_rate_before=("unknown_rate_before_col", "first"),
        unknown_rate_after=("unknown_rate_after_col", "first"),
        expected_drift=("expected_drift_column", "first")).reset_index()
    summary["delta"] = summary["unknown_rate_before"] - summary["unknown_rate_after"]
    flagged = summary[(summary["unknown_rate_after"] > 0.01) & (~summary["expected_drift"])]

    print(f"\n保存: {OUT.relative_to(BASE)} ({len(out_df)}行)")
    print("\n=== 列別 unknown_rate (正規化前→後) ===")
    print(summary.sort_values("unknown_rate_before", ascending=False).to_string(index=False))
    if len(flagged):
        print(f"\n【要確認】正規化後もunknown_rate>1%かつ高基数(新カテゴリ想定)列でない: "
              f"{flagged['feature_name'].tolist()}")
    else:
        print("\n正規化後、高基数列(種牡馬等、新規カテゴリの継続出現が正常)を除き"
              "unknown_rate>1%の列なし。")


if __name__ == "__main__":
    main()
