"""
gen_payout_table.py
人気構成別の配当中央値テーブルを生成する。

Stage 1 EV ゲートの基盤データ。

入力:
  - data/kekka_20160105_20251228_v2.csv (cp932)
  - data/master_kako5.csv               (距離・芝ダ取得)

出力:
  - data/payout_table.parquet
    columns: 馬券種, 人気パターン, 場所, 芝ダ, 距離区分, 頭数区分,
             median, mean, p25, p75, sample_n

人気パターン定義:
  - 単勝: pop_1着 (1, 2, 3, "4-6", "7+")
  - 複勝: pop_着内 同上
  - 馬連: pop_min, pop_max を tuple sort して 5x5=25 マス
  - 馬単: pop_1着, pop_2着 順序付き 25 マス
  - 三連複: (pop_min, pop_mid, pop_max) ソート済 35 マス
  - 三連単: (pop_1, pop_2, pop_3) 順序付き 125 マス

レース条件区分:
  - 場所: 中央 10 場
  - 芝ダ: "芝", "ダ"
  - 距離区分: "短"(<=1400), "マイル"(1500-1700), "中"(1800-2200), "長"(2400+)
  - 頭数区分: "少"(<=10), "中"(11-14), "多"(15+)

使い方:
    python gen_payout_table.py
"""
from __future__ import annotations
import io, re, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

BASE       = Path(__file__).parent
KEKKA_CSV  = BASE / "data/kekka_20160105_20251228_v2.csv"
MASTER_CSV = BASE / "data/master_kako5.csv"
OUT_PARQUET = BASE / "data/payout_table.parquet"


def pop_bucket(p: int | float) -> str:
    """人気を 5 区分に縮約。"""
    if pd.isna(p):
        return "?"
    p = int(p)
    if p == 1: return "1"
    if p == 2: return "2"
    if p == 3: return "3"
    if p <= 6: return "4-6"
    return "7+"


def dist_bucket(d) -> str:
    if pd.isna(d):
        return "?"
    d = int(d)
    if d <= 1400: return "短"
    if d <= 1700: return "マイル"
    if d <= 2200: return "中"
    return "長"


def field_bucket(n: int) -> str:
    if n <= 10: return "少"
    if n <= 14: return "中"
    return "多"


# ---------------------------------------------------------
# 1. kekka 読み込み
# ---------------------------------------------------------
print("kekka 読み込み...")
kk = pd.read_csv(KEKKA_CSV, encoding="cp932", dtype=str)
kk.columns = ["日付","場所","R","枠番","馬番","馬名","確定着順","レースID新",
              "単勝配当","複勝配当","枠連","馬連","馬単","三連複","三連単"]
kk["umaban"] = pd.to_numeric(kk["馬番"], errors="coerce").astype("Int64")
kk["jyun"]   = pd.to_numeric(kk["確定着順"], errors="coerce")
kk["race_id"] = kk["レースID新"].astype(str).str[:16]

# 単勝オッズ抽出
def parse_tan(row) -> float:
    s = str(row["単勝配当"]).strip()
    if not s or s == "nan":
        return np.nan
    if s.startswith("("):
        # 単勝オッズ表示 "(3.7)"
        m = re.match(r"\(([\d.]+)\)", s)
        return float(m.group(1)) if m else np.nan
    # 1着馬: 配当金額 → /100 でオッズ復元
    try:
        return float(s) / 100.0
    except Exception:
        return np.nan
kk["tan_odds"] = kk.apply(parse_tan, axis=1)

# 各レース内で人気ランキング (オッズ昇順 = 1人気)
kk["popularity"] = kk.groupby("race_id")["tan_odds"].rank(method="min", ascending=True).astype("Int64")

# 各レースの頭数
kk["field_size"] = kk.groupby("race_id")["umaban"].transform("count")

# ---------------------------------------------------------
# 2. master 読み込み (距離・芝ダ取得)
# ---------------------------------------------------------
print("master 読み込み (距離取得用)...")
master = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False,
                    usecols=["レースID(新/馬番無)", "場所", "芝・ダ", "距離"])
master = master.rename(columns={"レースID(新/馬番無)":"race_id", "芝・ダ":"td", "距離":"distance"})
master["race_id"] = master["race_id"].astype(str)
master = master.drop_duplicates(subset="race_id")
print(f"  master: {len(master):,} R")

# kekka に master 情報をマージ
kk = kk.merge(master[["race_id","td","distance"]], on="race_id", how="left")
kk["td_bucket"]   = kk["td"].fillna("?")
kk["dist_bucket"] = kk["distance"].apply(dist_bucket)
kk["field_bucket"]= kk["field_size"].apply(field_bucket)


# ---------------------------------------------------------
# 3. レース単位での 1-3 着馬の人気を取得
# ---------------------------------------------------------
print("レース単位の人気構成を集計...")
race_summary = []
for rid, grp in kk.groupby("race_id"):
    grp = grp.dropna(subset=["jyun"]).sort_values("jyun")
    if len(grp) < 3: continue
    p1 = grp.iloc[0]; p2 = grp.iloc[1]; p3 = grp.iloc[2]
    if pd.isna(p1["popularity"]) or pd.isna(p2["popularity"]) or pd.isna(p3["popularity"]):
        continue
    rec = {
        "race_id": rid,
        "place": p1["場所"],
        "td": p1["td_bucket"],
        "dist_bucket": p1["dist_bucket"],
        "field_bucket": p1["field_bucket"],
        "pop_1": int(p1["popularity"]),
        "pop_2": int(p2["popularity"]),
        "pop_3": int(p3["popularity"]),
        # 配当（最初の非nan値）
        "tan":    pd.to_numeric(p1["単勝配当"], errors="coerce"),
        "fuku_1": pd.to_numeric(p1["複勝配当"], errors="coerce"),
        "fuku_2": pd.to_numeric(p2["複勝配当"], errors="coerce"),
        "fuku_3": pd.to_numeric(p3["複勝配当"], errors="coerce"),
        "umaren": pd.to_numeric(p1["馬連"], errors="coerce"),
        "umatan": pd.to_numeric(p1["馬単"], errors="coerce"),
        "sanren": pd.to_numeric(p1["三連複"], errors="coerce"),
        "santan": pd.to_numeric(p1["三連単"], errors="coerce"),
    }
    race_summary.append(rec)

rs = pd.DataFrame(race_summary)
print(f"  集計対象: {len(rs):,} R")

# 人気バケット
rs["pop_1_b"] = rs["pop_1"].apply(pop_bucket)
rs["pop_2_b"] = rs["pop_2"].apply(pop_bucket)
rs["pop_3_b"] = rs["pop_3"].apply(pop_bucket)


# ---------------------------------------------------------
# 4. 馬券種別 集計
# ---------------------------------------------------------
def agg(df, group_cols, val_col):
    grp = df.groupby(group_cols)[val_col]
    out = grp.agg(median="median", mean="mean",
                   p25=lambda x: x.quantile(0.25),
                   p75=lambda x: x.quantile(0.75),
                   sample_n="count").reset_index()
    return out

print("配当集計中...")

# --- 単勝 ---
tbl_tan = agg(rs.dropna(subset=["tan"]),
              ["pop_1_b","place","td","dist_bucket","field_bucket"], "tan")
tbl_tan["馬券種"] = "単勝"
tbl_tan["人気パターン"] = tbl_tan["pop_1_b"]

# --- 複勝（着内3頭分まとめて評価）---
fuku_long = pd.concat([
    rs[["pop_1_b","place","td","dist_bucket","field_bucket","fuku_1"]].rename(columns={"pop_1_b":"pop_b","fuku_1":"pay"}),
    rs[["pop_2_b","place","td","dist_bucket","field_bucket","fuku_2"]].rename(columns={"pop_2_b":"pop_b","fuku_2":"pay"}),
    rs[["pop_3_b","place","td","dist_bucket","field_bucket","fuku_3"]].rename(columns={"pop_3_b":"pop_b","fuku_3":"pay"}),
], ignore_index=True).dropna(subset=["pay"])
tbl_fuku = agg(fuku_long, ["pop_b","place","td","dist_bucket","field_bucket"], "pay")
tbl_fuku["馬券種"] = "複勝"
tbl_fuku["人気パターン"] = tbl_fuku["pop_b"]

# --- 馬連（順序なし、min/max）---
def umaren_pat(row):
    a, b = sorted([row["pop_1_b"], row["pop_2_b"]])
    return f"{a}-{b}"
rs["umaren_pat"] = rs.apply(umaren_pat, axis=1)
tbl_umaren = agg(rs.dropna(subset=["umaren"]),
                  ["umaren_pat","place","td","dist_bucket","field_bucket"], "umaren")
tbl_umaren["馬券種"] = "馬連"
tbl_umaren["人気パターン"] = tbl_umaren["umaren_pat"]

# --- 馬単（順序付き）---
def umatan_pat(row):
    return f"{row['pop_1_b']}-{row['pop_2_b']}"
rs["umatan_pat"] = rs.apply(umatan_pat, axis=1)
tbl_umatan = agg(rs.dropna(subset=["umatan"]),
                  ["umatan_pat","place","td","dist_bucket","field_bucket"], "umatan")
tbl_umatan["馬券種"] = "馬単"
tbl_umatan["人気パターン"] = tbl_umatan["umatan_pat"]

# --- 三連複（順序なし）---
def sanren_pat(row):
    s = sorted([row["pop_1_b"], row["pop_2_b"], row["pop_3_b"]])
    return "-".join(s)
rs["sanren_pat"] = rs.apply(sanren_pat, axis=1)
tbl_sanren = agg(rs.dropna(subset=["sanren"]),
                  ["sanren_pat","place","td","dist_bucket","field_bucket"], "sanren")
tbl_sanren["馬券種"] = "三連複"
tbl_sanren["人気パターン"] = tbl_sanren["sanren_pat"]

# --- 三連単（順序付き）---
def santan_pat(row):
    return f"{row['pop_1_b']}-{row['pop_2_b']}-{row['pop_3_b']}"
rs["santan_pat"] = rs.apply(santan_pat, axis=1)
tbl_santan = agg(rs.dropna(subset=["santan"]),
                  ["santan_pat","place","td","dist_bucket","field_bucket"], "santan")
tbl_santan["馬券種"] = "三連単"
tbl_santan["人気パターン"] = tbl_santan["santan_pat"]


# ---------------------------------------------------------
# 5. 統合 & 出力
# ---------------------------------------------------------
all_tbl = pd.concat([
    tbl_tan[   ["馬券種","人気パターン","place","td","dist_bucket","field_bucket","median","mean","p25","p75","sample_n"]],
    tbl_fuku[  ["馬券種","人気パターン","place","td","dist_bucket","field_bucket","median","mean","p25","p75","sample_n"]],
    tbl_umaren[["馬券種","人気パターン","place","td","dist_bucket","field_bucket","median","mean","p25","p75","sample_n"]],
    tbl_umatan[["馬券種","人気パターン","place","td","dist_bucket","field_bucket","median","mean","p25","p75","sample_n"]],
    tbl_sanren[["馬券種","人気パターン","place","td","dist_bucket","field_bucket","median","mean","p25","p75","sample_n"]],
    tbl_santan[["馬券種","人気パターン","place","td","dist_bucket","field_bucket","median","mean","p25","p75","sample_n"]],
], ignore_index=True)
all_tbl = all_tbl.rename(columns={"place":"場所", "td":"芝ダ"})

# サンプル数 5 未満は除外（信頼度低い）
before = len(all_tbl)
all_tbl = all_tbl[all_tbl["sample_n"] >= 5].copy()
print(f"  集計行: {before:,} → {len(all_tbl):,} (sample_n>=5)")

# parquet 出力
OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
all_tbl.to_parquet(OUT_PARQUET, index=False)
print(f"\n書出し: {OUT_PARQUET}")
print(f"  サイズ: {OUT_PARQUET.stat().st_size/1024:.1f} KB")

# サンプル表示
print("\n--- サンプル: 三連単 ---")
print(all_tbl[all_tbl["馬券種"]=="三連単"].head(10).to_string(index=False))
print("\n--- サンプル: 馬連 ---")
print(all_tbl[all_tbl["馬券種"]=="馬連"].head(10).to_string(index=False))
print("\n--- サンプル: 単勝 ---")
print(all_tbl[all_tbl["馬券種"]=="単勝"].head(10).to_string(index=False))
