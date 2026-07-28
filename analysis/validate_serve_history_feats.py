# -*- coding: utf-8 -*-
"""
validate_serve_history_feats.py
===============================
serve skew 残差回収 (serve_history_feats.py) の検証。test=2024-25 で:

  [1] パリティ: serve 経路の再計算値 (馬名JOIN + as-of) が master_v2 の
      学習側格納値と一致するか (特徴ごとに一致率 / MAE)。
      騎手コードは master の実コードを override で与え、名前マップ誤差と分離。
  [2] 影響: diag_serve_full_impact.py の枠組みで ◎複勝率を
      baseline(フル特徴) / serve旧(dead12マスク) / serve新(再計算値で置換) で比較。
      期待: serve新 ≈ baseline (悪化しないこと、+1pt 規模の回収)。

実行: PYTHONUTF8=1 python -m analysis.validate_serve_history_feats
"""
import sys, io, json
from pathlib import Path
import joblib, numpy as np, pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
BASE = Path(r"E:\PyCaLiAI"); sys.path.insert(0, str(BASE))
from backtest_pl_ev import apply_encoders, COL_RID, COL_BAN
from serve_history_feats import fill_history_features, NUM_FEATS, CAT_FEATS

b = joblib.load(BASE / "models/unified_rank_v6.pkl")
model, feats, encs = b["model"], b["feature_cols"], b["encoders"]
catset = set(encs.keys())

# serve 旧 (現状 dead のまま) のマスク対象 = 今回回収する 12
DEAD12 = NUM_FEATS + CAT_FEATS

print("[load] master test=2024-25")
df = pd.read_csv(BASE / "data/master_v2_20130105-20251228.csv",
                 encoding="utf-8-sig", low_memory=False)
df = df.dropna(subset=[COL_RID, "split"])
df = df[df["split"] == "test"].copy().reset_index(drop=True)
df["着順"] = pd.to_numeric(df["着順"], errors="coerce")
print(f"  {len(df):,} 行 / {df[COL_RID].nunique():,} R")

# ------------------------------------------------------------
# serve 相当のフレームを作って再計算
# ------------------------------------------------------------
sv = df[["馬名", "種牡馬", "年齢", "日付", "場所", "芝・ダ", "距離"]].copy()
stats = fill_history_features(
    sv, base=BASE,
    jockey_code_override=df["騎手コード"],
    trainer_code_override=df["調教師コード"],
)
print(f"\n[resolve] hit={stats['hit']:,} new={stats['new']:,} "
      f"ambiguous={stats['ambiguous']:,} bad_date={stats['bad_date']:,}")

# ------------------------------------------------------------
# [1] パリティ
# ------------------------------------------------------------
print(f"\n{'='*72}\n[1] パリティ (serve 再計算 vs master 学習値, test 全行)\n{'='*72}")
print(f"{'feature':<28}{'両方あり':>9}{'一致率':>8}{'MAE':>9}{'serveのみNaN':>13}{'masterのみNaN':>14}")
parity = {}
for c in NUM_FEATS:
    a = pd.to_numeric(df[c], errors="coerce")   # 学習値
    s = pd.to_numeric(sv[c], errors="coerce")   # serve 再計算
    both = a.notna() & s.notna()
    agree = float((np.abs(a[both] - s[both]) < 1e-6).mean()) if both.any() else np.nan
    mae = float(np.abs(a[both] - s[both]).mean()) if both.any() else np.nan
    only_s_nan = int((a.notna() & s.isna()).sum())
    only_a_nan = int((a.isna() & s.notna()).sum())
    parity[c] = {"agree": agree, "mae": mae,
                 "serve_nan_master_val": only_s_nan,
                 "master_nan_serve_val": only_a_nan}
    print(f"{c:<28}{int(both.sum()):>9,}{agree*100:>7.1f}%{mae:>9.4f}"
          f"{only_s_nan:>13,}{only_a_nan:>14,}")

# ------------------------------------------------------------
# [2] 影響 (◎複勝率)
# ------------------------------------------------------------
print(f"\n{'='*72}\n[2] ◎複勝率影響 (test 2024-25)\n{'='*72}")

def score_frame(mode: str) -> pd.DataFrame:
    d = df.copy()
    if mode == "serve_old":       # 現状: 12特徴が -9999/__NaN__ に潰れる
        for c in DEAD12:
            d[c] = "__NaN__" if c in catset else np.nan
    elif mode == "serve_new":     # 今回: 再計算値で置換
        for c in NUM_FEATS:
            d[c] = pd.to_numeric(sv[c], errors="coerce").values
        for c in CAT_FEATS:
            d[c] = sv[c].values   # str コード or "__NaN__"
    d = apply_encoders(d, encs)
    X = d[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    o = df[[COL_RID, COL_BAN, "着順"]].copy()
    o["_score"] = model.predict(X)
    return o

def top3_rate(sc: pd.DataFrame):
    hit = 0; nR = 0
    for rid, g in sc.groupby(COL_RID, sort=False):
        if len(g) < 5:
            continue
        hon = g.loc[g["_score"].idxmax()]
        if pd.notna(hon["着順"]):
            hit += int(hon["着順"] <= 3); nR += 1
    return hit / nR * 100, nR

res = {}
for mode, label in [("baseline", "baseline(フル特徴)"),
                    ("serve_old", "serve旧(12特徴dead)"),
                    ("serve_new", "serve新(再計算)")]:
    r, nR = top3_rate(score_frame(mode))
    res[mode] = r
    print(f"  {label:<24} ◎複勝率 {r:.2f}%  (nR={nR:,})")

d_new = res["serve_new"] - res["serve_old"]
d_gap = res["baseline"] - res["serve_new"]
print(f"\n  回収: serve旧→serve新 {d_new:+.2f}pt / baseline との残差 {d_gap:.2f}pt")

out = {"parity": parity, "resolve": {k: stats[k] for k in ("hit", "new", "ambiguous")},
       "top3": res, "recovered_pt": round(d_new, 3), "residual_pt": round(d_gap, 3)}
(BASE / "reports" / "validate_serve_history_feats.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n[saved] reports/validate_serve_history_feats.json")
