"""A03: 学習 master の整合性 (split・ラベル・同着・取消/中止・重複・頭数)。read-only。

usage: python a03_master.py <ROOT>
出力: out/a03_master.json
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("E:/PyCaLiAI")
OUT = Path(__file__).resolve().parents[1] / "out"
CAND = ROOT / "candidates/c1_migration_v2/data/master_v2_20130105-20251228.csv"
LIVE = ROOT / "data/master_v2_20130105-20251228.csv"
RID, BAN, JYUN = "レースID(新/馬番無)", "馬番", "着順"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


res = {"cand_sha": sha(CAND), "live_sha": sha(LIVE)}
res["same_master"] = res["cand_sha"] == res["live_sha"]
cols = [RID, BAN, JYUN, "split", "日付", "馬名", "出走頭数", "fukusho_flag", "レースID(新)",
        "馬主(最新/仮想)"]
hdr = pd.read_csv(CAND, nrows=0, encoding="utf-8-sig").columns.tolist()
use = [c for c in cols if c in hdr]
res["missing_cols"] = [c for c in cols if c not in hdr]
df = pd.read_csv(CAND, usecols=use, encoding="utf-8-sig", low_memory=False)
res["n_rows"] = len(df)
res["n_races"] = int(df[RID].nunique())

# 日付
if "日付" in df.columns:
    d = pd.to_numeric(df["日付"], errors="coerce")
    if d.max() < 1e7:  # yymmdd 形式など
        d = d + 20000000
    df["_d"] = d
else:
    df["_d"] = pd.to_numeric(df[RID].astype(str).str[:8], errors="coerce")
df["_rid8"] = pd.to_numeric(df[RID].astype(str).str[:8], errors="coerce")
res["date_vs_rid8_mismatch"] = int((df["_d"] != df["_rid8"]).sum())
sp = df.groupby("split")["_rid8"].agg(["min", "max", "count"])
res["split_ranges"] = sp.reset_index().astype(str).to_dict("records")
res["split_na_rows"] = int(df["split"].isna().sum())
# レースが split をまたぐか
nsplit = df.groupby(RID)["split"].nunique()
res["races_spanning_splits"] = int((nsplit > 1).sum())
# 期待境界
exp = np.where(df["_rid8"] <= 20221231, "train", np.where(df["_rid8"] <= 20231231, "valid", "test"))
res["split_vs_date_rule_mismatch"] = int((exp != df["split"].astype(str)).sum())

# 着順
j = df[JYUN].astype(str).str.strip()
jn = pd.to_numeric(j, errors="coerce")
res["jyun_non_numeric_values"] = j[jn.isna()].value_counts().head(20).to_dict()
res["jyun_non_numeric_rows"] = int(jn.isna().sum())
res["jyun_zero_rows"] = int((jn == 0).sum())
df["_j"] = jn
# 中止/失格/降着 の痕跡 (数値でない値は学習・較正で dropna → レースから馬ごと消える)
# 同着: レース内で同一着順の重複
fin = df[df["_j"].notna() & (df["_j"] > 0)]
dup = fin.groupby([RID, "_j"]).size()
dh = dup[dup > 1].reset_index()
res["dead_heat_groups"] = int(len(dh))
res["dead_heat_by_position"] = dh["_j"].value_counts().sort_index().head(5).astype(int).to_dict()
res["dead_heat_races_top3"] = int(dh[dh["_j"] <= 3][RID].nunique())
# 着順が 1 から始まらない / 欠番レース (除外馬を落とした後に連番が崩れる等)
g = fin.groupby(RID)["_j"]
res["races_without_winner"] = int((g.min() != 1).sum())
res["races_lt3_finishers"] = int((g.size() < 3).sum())
# 重複キー
res["dup_race_umaban"] = int(df.duplicated([RID, BAN]).sum())
res["dup_race_horse"] = int(df.duplicated([RID, "馬名"]).sum()) if "馬名" in df.columns else None
# fukusho_flag の定義
if "fukusho_flag" in df.columns:
    ff = pd.to_numeric(df["fukusho_flag"], errors="coerce")
    res["fukusho_flag_vs_top3_mismatch"] = int(((ff == 1) != (df["_j"] <= 3)).sum())
    res["fukusho_flag_rate"] = float(ff.mean())
# 頭数: 複勝は 7 頭以下で 2 着払い。学習/較正のラベルは top3
fs = fin.groupby(RID).size()
res["finisher_field_size_dist"] = fs.value_counts().sort_index().astype(int).to_dict()
res["races_le7_finishers"] = int((fs <= 7).sum())
if "出走頭数" in df.columns:
    sz = pd.to_numeric(df.groupby(RID)["出走頭数"].first(), errors="coerce")
    res["races_shusso_le7"] = int((sz <= 7).sum())
    res["races_shusso_le7_by_split"] = df.groupby(RID).agg(s=("split", "first"), n=("出走頭数", "first")) \
        .assign(n=lambda x: pd.to_numeric(x.n, errors="coerce")).query("n<=7")["s"].value_counts().to_dict()
# 年別件数
res["races_by_year"] = df.groupby(df["_rid8"] // 10000)[RID].nunique().astype(int).to_dict()
json.dump(res, open(OUT / "a03_master.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
