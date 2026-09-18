# -*- coding: utf-8 -*-
"""
features.py — C1(v6由来105特徴の数値化) + C2(EXP01生の選択) + C3(EXP02能力+EXP03経験) の候補行列
====================================================================================================
C1の型付け規則 (機械的、結果を見て変えない):
  - 数値dtype                                    → そのまま使う
  - object dtype かつ pd.to_numeric 変換後の被覆率が raw の notna 率の 50% 以上を保つ
                                                  → 数値変換して使う (前走着差タイム・斤量等)
  - object dtype かつ pd.to_numeric 変換後の被覆率が 10% 未満                        → 除外 (R6, 前走走破タイムのみ該当)
  - object dtype かつ 学習期間 (2016-2021) 内での unique 値数 <= 15                  → one-hotダミー化 (未知カテゴリは全0)
  - object dtype かつ unique 値数 > 15                                              → 除外 (R5, 高基数の生カテゴリ)
出力: rid16, ban をキーに、候補特徴の辞書 {name: np.ndarray} と、どのCグループに属するかの表を返す。
実行: python -m analysis.mcond.exp04_invariant_info_dev.features  (単体で作って parquet に保存)
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
from grade_feats import class_name_to_ord  # noqa: E402  (EXP01/EXP03と同じ既存の序数エンコード)
HERE = Path(__file__).resolve().parent
D = BASE / "data/_research/mcond"
OUT = HERE / "out"
MASTER = BASE / "data/master_v2_20130105-20251228.csv"

C2_COLS = ["raw_log_int", "raw_dist_chg", "raw_venue_chg", "raw_surface_chg", "raw_cls_chg",
           "raw_jockey_same", "raw_jq_delta", "raw_jt_pair"]
C3_FROM_EXP02 = ["dyn_skill_mu", "horse_skill_minus_field"]
C3_FROM_EXP03 = ["raw_career_runs", "raw_days_since"]

NUMERIC_COERCE_KEEP_RATE = 0.50
NUMERIC_COERCE_MIN_ABS = 0.10
MAX_ONEHOT_CARDINALITY = 15


def _type_and_encode_c1(kept_cols: list[str], train_mask_by_rid: dict) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(MASTER, encoding="utf-8-sig", low_memory=False,
                     usecols=["レースID(新)", "馬番", "日付"] + kept_cols)
    df["rid16"] = df["レースID(新)"].astype(str).str[:16]
    df["ban"] = pd.to_numeric(df["馬番"], errors="coerce")
    df = df.dropna(subset=["ban"]).copy()
    df["ban"] = df["ban"].astype(int)
    df["year"] = df["日付"].astype(str).str[:4].astype(int)
    train = (df.year >= 2016) & (df.year <= 2021)

    kept_num, kept_onehot, excluded = [], {}, {}
    out = pd.DataFrame({"rid16": df["rid16"], "ban": df["ban"]})
    for c in kept_cols:
        s = df[c]
        if c == "クラス名":
            # R7: 生カテゴリ(17水準, one-hotだとMAX_ONEHOT_CARDINALITY超で除外される)より、
            # EXP01/EXP03と同じ既存の序数エンコード(高いほど格上)を1本の数値特徴として使う方が
            # 効率的かつ情報の欠落が少ない。この置換は評価(Gate1以降)を一度も見る前に決めた。
            out["c1__クラス名_ord"] = s.map(class_name_to_ord)
            kept_num.append("クラス名_ord(=class_name_to_ord)")
            continue
        if pd.api.types.is_numeric_dtype(s):
            out[f"c1__{c}"] = pd.to_numeric(s, errors="coerce")
            kept_num.append(c)
            continue
        raw_notna = s.notna().mean()
        num = pd.to_numeric(s, errors="coerce")
        num_notna = num.notna().mean()
        # R6 (数値のはずだが変換不能, 例: 前走走破タイム) は build() 側で事前に列ごと除外済み。
        # ここでの num_notna が低いのは「元々カテゴリで数値ではない」場合がほとんどなので、
        # 数値変換を諦めて基数でone-hot/除外(R5)を判定する一本の分岐にする。
        if num_notna >= NUMERIC_COERCE_MIN_ABS and num_notna >= NUMERIC_COERCE_KEEP_RATE * raw_notna:
            out[f"c1__{c}"] = num
            kept_num.append(c)
            continue
        uniq_train = s[train].dropna().unique()
        if len(uniq_train) <= MAX_ONEHOT_CARDINALITY:
            cats = sorted(str(x) for x in uniq_train)
            for cat in cats[1:]:  # 先頭カテゴリを参照水準として1つ落とす (完全共線性回避)
                out[f"c1__{c}__{cat}"] = (s.astype(str) == cat).astype(float)
            kept_onehot[c] = cats
        else:
            excluded[c] = f"R5高基数({len(uniq_train)}水準)"
    meta = {"kept_numeric": kept_num, "kept_onehot": kept_onehot, "excluded": excluded}
    return out.drop(columns=["rid16", "ban"]).assign(rid16=out.rid16, ban=out.ban), meta


def build(save: bool = True) -> tuple[pd.DataFrame, dict]:
    audit = json.loads((OUT / "data_audit.json").read_text(encoding="utf-8"))
    kept_cols = [c for c in audit["kept"] if c != "前走走破タイム"]  # R6 (手動確認, DATA_AUDIT.md参照)
    manual_excl = {"前走走破タイム": "R6数値変換不能(手動確認, coerced_notna=0.000, DATA_AUDIT.md)"}

    c1, meta = _type_and_encode_c1(kept_cols, {})
    meta["excluded"].update(manual_excl)
    c1["rid16"] = c1["rid16"].astype(str)

    b = pd.read_parquet(D / "base.parquet")[["rid16", "ban", "year"]]
    b["rid16"] = b["rid16"].astype(str)
    c2 = pd.read_parquet(D / "exp01_features.parquet")[["rid16", "ban"] + C2_COLS]
    c2["rid16"] = c2["rid16"].astype(str)
    c3a = pd.read_parquet(D / "exp02_features.parquet")[["rid16", "ban"] + C3_FROM_EXP02]
    c3a["rid16"] = c3a["rid16"].astype(str)
    c3b = pd.read_parquet(D / "exp03_features.parquet")[["rid16", "ban"] + C3_FROM_EXP03]
    c3b["rid16"] = c3b["rid16"].astype(str)

    df = b.merge(c1, on=["rid16", "ban"], how="left").merge(c2, on=["rid16", "ban"], how="left") \
          .merge(c3a, on=["rid16", "ban"], how="left").merge(c3b, on=["rid16", "ban"], how="left")

    c1_cols = [c for c in df.columns if c.startswith("c1__")]
    groups = {"C1": c1_cols, "C2": C2_COLS, "C3": C3_FROM_EXP02 + C3_FROM_EXP03}
    meta["groups"] = groups
    meta["n_candidates"] = sum(len(v) for v in groups.values())
    if save:
        df.to_parquet(D / "exp04_candidates.parquet", index=False)
        (OUT / "feature_typing.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str),
                                                 encoding="utf-8")
    return df, meta


if __name__ == "__main__":
    df, meta = build()
    print(f"C1(数値化後): {len(meta['groups']['C1'])} 列  C2: {len(meta['groups']['C2'])}  "
          f"C3: {len(meta['groups']['C3'])}  合計候補: {meta['n_candidates']}")
    print(f"C1除外: {len(meta['excluded'])}")
    for k, v in meta["excluded"].items():
        print(f"  {k}: {v}")
    print(f"rows={len(df):,}")
    na_rate = df[meta['groups']['C1'] + meta['groups']['C2'] + meta['groups']['C3']].isna().mean()
    print("欠損率トップ10:")
    print(na_rate.sort_values(ascending=False).head(10).round(3).to_string())
