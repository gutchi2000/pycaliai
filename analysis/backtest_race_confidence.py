"""
backtest_race_confidence.py
「AIレース信頼度」指標がROIと相関するかを検証するバックテスト。

対象: 2024-2025 の OOS（除外: 東京・小倉）
評価: HALO 三連単フォーメーション（本番ロジック = Hybrid / Stage 0）の
      レース別実ROIを、各信頼度指標および合成スコアと突合する。

出力:
  - reports/race_confidence_backtest.csv   レース別の生データ
  - 標準出力に相関分析＋バケット別ROI表

使い方:
    python backtest_race_confidence.py
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

BASE           = Path(__file__).parent
PRED_CSV       = BASE / "reports/ensemble_predictions.csv"
MASTER_CSV     = BASE / "data/master_kako5.csv"
KEKKA_CSV      = BASE / "data/kekka_20160105_20251228_v2.csv"
ORDER_MODEL    = BASE / "models/order_model_v1.pkl"
PLAYBOOK_JSON  = BASE / "data/halo_formation_playbook.json"
OUT_CSV        = BASE / "reports/race_confidence_backtest.csv"

BUDGET         = 9600
YEARS_TARGET   = {2024, 2025}
EXCLUDE_PLACES = {"東京", "小倉"}

# =========================================================
# 1. モデル読み込み
# =========================================================
print("着順モデル読み込み...")
order_obj = joblib.load(ORDER_MODEL)
o_model = order_obj["model"]
o_feats = order_obj["features"]
o_encs  = order_obj["encoders"]
print(f"  order_model features: {len(o_feats)}")

from utils import parse_time_str
TIME_COLS = ["前走走破タイム", "前走着差タイム"]


def _prep(df: pd.DataFrame, encs: dict, feats: list) -> pd.DataFrame:
    df = df.copy()
    for col, le in encs.items():
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("__NaN__")
            known = set(le.classes_)
            df[col] = df[col].apply(lambda x: x if x in known else "__unknown__")
            if "__unknown__" not in le.classes_:
                le.classes_ = np.append(le.classes_, "__unknown__")
            df[col] = le.transform(df[col])
    for col in TIME_COLS:
        if col in df.columns:
            df[col] = parse_time_str(df[col])
    for f in feats:
        if f not in df.columns:
            df[f] = np.nan
    return df


def predict_order_proba(race_df: pd.DataFrame) -> dict[int, float]:
    """各馬の1着確率を返す（order_model_v1 class 0 = 1着）。"""
    df = _prep(race_df, o_encs, o_feats)
    proba = o_model.predict_proba(df[o_feats])
    return {int(ub): float(proba[i, 0])
            for i, ub in enumerate(df["馬番"].values)
            if pd.notna(ub)}


# =========================================================
# 2. playbook 読み込み
# =========================================================
print("playbook 読み込み...")
playbook_cells: dict[str, dict] = {}
if PLAYBOOK_JSON.exists():
    with open(PLAYBOOK_JSON, encoding="utf-8") as f:
        playbook_cells = json.load(f).get("cells", {})
print(f"  playbook cells: {len(playbook_cells)}")


def _field_bucket(n: int) -> str:
    if n <= 10: return "少"
    if n <= 14: return "中"
    return "多"


def cell_oos_roi(shiba_da: str, field_size: int) -> float:
    key = f"{shiba_da}_{_field_bucket(field_size)}"
    cell = playbook_cells.get(key, {})
    return float(cell.get("oos_roi", 100.0))


# =========================================================
# 3. データ読み込み
# =========================================================
print("ensemble_predictions 読み込み...")
pred = pd.read_csv(PRED_CSV, encoding="utf-8-sig")
pred = pred.rename(columns={"レースID(新/馬番無)": "race_id", "馬番": "umaban"})
pred["race_id"] = pred["race_id"].astype(str)
pred["umaban"]  = pd.to_numeric(pred["umaban"], errors="coerce").astype("Int64")
pred["mark"]    = pred["mark"].astype(str).replace({"nan": ""})
pred["year"]    = pred["race_id"].str[:4].astype(int)
pred = pred[pred["year"].isin(YEARS_TARGET)].copy()
print(f"  予測: {pred['race_id'].nunique():,} R")

print("master_kako5 読み込み中（時間かかります）...")
master = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
master["race_id"] = master["レースID(新/馬番無)"].astype(str)
master["馬番"]    = pd.to_numeric(master["馬番"], errors="coerce").astype("Int64")
master["year"]    = master["race_id"].str[:4].astype(int)
master = master[master["year"].isin(YEARS_TARGET)].copy()
print(f"  master: {master['race_id'].nunique():,} R / {len(master):,} 行")
master_grp = {rid: g for rid, g in master.groupby("race_id")}

print("kekka 読み込み...")
kk = pd.read_csv(KEKKA_CSV, encoding="cp932", dtype=str)
kk["race_id"]    = kk["レースID(新)"].astype(str).str[:16]
kk["umaban"]     = pd.to_numeric(kk["馬番"], errors="coerce").astype("Int64")
kk["jyun"]       = pd.to_numeric(kk["確定着順"], errors="coerce")
kk["santan_pay"] = pd.to_numeric(kk["３連単"], errors="coerce")
kk["place"]      = kk["場所"].astype(str)
kk_grp = {rid: g for rid, g in kk.groupby("race_id")}
print(f"  kekka: {len(kk_grp):,} R")


# =========================================================
# 4. レース構築 + 指標計算
# =========================================================
print("レース構築 + 指標計算中...")

def _shiba_da(mgrp: pd.DataFrame) -> str:
    """master_kako5 に芝ダ列があるなら採用、無ければ距離から推定。"""
    if mgrp is None or len(mgrp) == 0:
        return "芝"
    for col in ["芝・ダ", "芝ダ", "馬場"]:
        if col in mgrp.columns:
            v = str(mgrp.iloc[0].get(col, "")).strip()
            if v.startswith("ダ"): return "ダ"
            if v.startswith("芝"): return "芝"
    return "芝"  # fallback


def _build_combos(first, second, third):
    combos = set()
    for f in first:
        for s in second:
            for t in third:
                if len({f, s, t}) == 3:
                    combos.add((f, s, t))
    return list(combos)


def _score_pattern_combos(marks, scores):
    """sim_halo_formation_v2.py logic_score と同じ式。スコア差ルールによる買い目。"""
    h  = marks.get("◎"); o  = marks.get("◯")
    h3 = marks.get("▲"); h4 = marks.get("△"); h5 = marks.get("×")
    s_hon = scores.get(h, 0); s_tai = scores.get(o, 0)
    s_sab = scores.get(h3, 0) if h3 else 0
    s_del = scores.get(h4, 0) if h4 else 0
    gap_12   = (s_hon - s_tai) * 100
    gap_top4 = ((s_hon - s_del) if s_del > 0 else (s_hon - s_sab)) * 100
    if gap_12 >= 10:
        first  = [h]
        second = [x for x in [o, h3]              if x]
        third  = [x for x in [o, h3, h4, h5]      if x]
    elif gap_12 <= 5 and gap_top4 <= 15:
        first  = [h, o]
        second = [x for x in [h, o, h3]           if x]
        third  = [x for x in [h, o, h3, h4]       if x]
    else:
        first  = [h, o]
        second = [x for x in [h, o, h3, h4]       if x]
        third  = [x for x in [h, o, h3, h4, h5]   if x]
    return _build_combos(first, second, third)


def _hybrid_combos(marks, scores, pw):
    """Stage 0 本番 Hybrid: スコア差ベース + AI高信頼で◎突出を上書き。"""
    h = marks.get("◎"); o = marks.get("◯")
    combos = _score_pattern_combos(marks, scores)
    if pw and h in pw and o in pw:
        pw_h = pw[h]; pw_o = pw[o]
        if pw_h >= 0.50 and pw_h >= pw_o * 2.0:
            h3 = marks.get("▲"); h4 = marks.get("△"); h5 = marks.get("×")
            first  = [h]
            second = [x for x in [o, h3] if x]
            third  = [x for x in [o, h3, h4, h5] if x]
            combos = _build_combos(first, second, third)
    return combos


rows = []
n_proc = 0
for rid, p_grp in pred.groupby("race_id"):
    n_proc += 1
    if n_proc % 500 == 0:
        print(f"  ...{n_proc:,} races")

    kgrp = kk_grp.get(rid)
    if kgrp is None or kgrp.empty:
        continue
    place = kgrp.iloc[0]["place"]
    if place in EXCLUDE_PLACES:
        continue
    top3 = kgrp.dropna(subset=["jyun"]).sort_values("jyun").head(3)
    if len(top3) < 3:
        continue
    first  = int(top3.iloc[0]["umaban"])
    second = int(top3.iloc[1]["umaban"])
    third  = int(top3.iloc[2]["umaban"])
    pay_series = top3["santan_pay"].dropna()
    if pay_series.empty or pay_series.iloc[0] <= 0:
        continue
    santan_pay = float(pay_series.iloc[0])

    # マーク + スコア
    marks: dict[str, int] = {}
    scores: dict[int, float] = {}
    for _, r in p_grp.iterrows():
        ub = int(r["umaban"]) if pd.notna(r["umaban"]) else None
        if ub is None: continue
        scores[ub] = float(r["ensemble_prob"]) if pd.notna(r["ensemble_prob"]) else 0.0
        if r["mark"] in ("◎","◯","▲","△","×"):
            marks[r["mark"]] = ub
    if "◎" not in marks or "◯" not in marks:
        continue

    # AI 推論
    mgrp = master_grp.get(rid)
    pw_order = {}
    if mgrp is not None and len(mgrp) >= 5:
        try:
            pw_order = predict_order_proba(mgrp)
        except Exception:
            pass

    h  = marks["◎"]; o  = marks["◯"]
    h3 = marks.get("▲"); h4 = marks.get("△")
    s_hon = scores.get(h, 0); s_tai = scores.get(o, 0)
    s_sab = scores.get(h3, 0) if h3 else 0
    s_del = scores.get(h4, 0) if h4 else 0

    gap_12   = (s_hon - s_tai) * 100
    gap_top4 = ((s_hon - s_del) if s_del > 0 else (s_hon - s_sab)) * 100
    pw_hon   = float(pw_order.get(h, 0.0)) if pw_order else 0.0
    pw_tai   = float(pw_order.get(o, 0.0)) if pw_order else 0.0
    pw_ratio = (pw_hon / pw_tai) if pw_tai > 0 else (pw_hon * 100 if pw_hon > 0 else 0.0)

    field_size = len(kgrp)
    shiba_da   = _shiba_da(mgrp)
    cell_roi   = cell_oos_roi(shiba_da, field_size)

    # 実 HALO ROI（Stage 0 本番 Hybrid）
    combos = _hybrid_combos(marks, scores, pw_order)
    if not combos:
        continue
    n = len(combos)
    per = max(100, (BUDGET // n // 100) * 100)
    bet = per * n
    target = (first, second, third)
    hit = target in set(combos)
    ret = (per * santan_pay / 100.0) if hit else 0.0

    rows.append({
        "race_id":    rid,
        "place":      place,
        "shiba_da":   shiba_da,
        "field_size": field_size,
        "field_bucket": _field_bucket(field_size),
        "s_hon":      s_hon,
        "s_tai":      s_tai,
        "gap_12":     gap_12,
        "gap_top4":   gap_top4,
        "pw_hon":     pw_hon,
        "pw_tai":     pw_tai,
        "pw_ratio":   pw_ratio,
        "cell_roi":   cell_roi,
        "n_combos":   n,
        "bet":        bet,
        "ret":        ret,
        "hit":        int(hit),
        "roi":        ret / bet * 100 if bet else 0.0,
    })

df_races = pd.DataFrame(rows)
print(f"\n  構築完了: {len(df_races):,} レース")
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df_races.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print(f"  出力: {OUT_CSV.name}")

# =========================================================
# 5. 相関分析（スピアマン順位相関で単調性を評価）
# =========================================================
print("\n=== 指標単体 × HALO ROI（スピアマン順位相関）===")
print("  ※ 相関係数 > 0: 指標が高いほどROIが高い傾向 → 使える")
indicator_cols = ["gap_12", "gap_top4", "pw_hon", "pw_ratio", "cell_roi"]
for col in indicator_cols:
    x = df_races[col].values
    y = df_races["roi"].values
    # スピアマン相関（手計算で scipy 不要化）
    rx = pd.Series(x).rank()
    ry = pd.Series(y).rank()
    rho = rx.corr(ry)
    print(f"  {col:10s}  spearman ρ = {rho:+.4f}")

# =========================================================
# 6. 合成スコア（仮の線形重み）+ バケット分析
# =========================================================
print("\n=== 合成「信頼度スコア」バケット別ROI ===")
print("  formula: 0.3×gap_12 + 30×pw_hon + 0.4×cell_roi")
print("  （仮の重み。後工程でチューニング可能）")

# 標準化しなくても各指標のスケールは概ね合わせている:
#   gap_12: 0〜30pt 想定 (×0.3 → 0〜9)
#   pw_hon: 0〜0.7 想定 (×30  → 0〜21)
#   cell_roi: 60〜160 想定 (×0.4 → 24〜64)
df_races["conf"] = (
    0.3  * df_races["gap_12"].clip(0, 30)
    + 30 * df_races["pw_hon"]
    + 0.4 * df_races["cell_roi"].clip(60, 160)
)

# 5 分位
df_races["bucket"] = pd.qcut(
    df_races["conf"], q=5, labels=["🚫見送り", "⚠️薄め", "➖標準", "⚡勝負", "🔥超勝負"],
    duplicates="drop",
)

agg = df_races.groupby("bucket", observed=True).agg(
    n=("race_id", "count"),
    bet=("bet", "sum"),
    ret=("ret", "sum"),
    hits=("hit", "sum"),
    avg_conf=("conf", "mean"),
).reset_index()
agg["roi"]    = agg["ret"] / agg["bet"] * 100
agg["hrate"]  = agg["hits"] / agg["n"] * 100

print(f"  {'bucket':12s}  {'n':>5} {'avg_conf':>9} {'hrate':>7} {'ROI':>7}")
for _, r in agg.iterrows():
    print(f"  {str(r['bucket']):12s}  {int(r['n']):>5} "
          f"{r['avg_conf']:>9.2f} {r['hrate']:>6.2f}% {r['roi']:>6.2f}%")

# 単調性チェック
rois_by_order = agg.sort_values("avg_conf")["roi"].values
monotone_inc = all(rois_by_order[i] <= rois_by_order[i+1] + 5  # 5pt ノイズ許容
                    for i in range(len(rois_by_order) - 1))
print(f"\n  単調増加（5ptノイズ許容）: {'✅ YES' if monotone_inc else '❌ NO'}")
print(f"  🔥超勝負 − 🚫見送り のROI差: "
      f"{rois_by_order[-1] - rois_by_order[0]:+.2f}pt")

# =========================================================
# 7. 層別分析（芝ダ × 頭数）
# =========================================================
print("\n=== 層別分析: 芝ダ × 頭数ビン ===")
for (sd, fb), grp in df_races.groupby(["shiba_da", "field_bucket"]):
    if len(grp) < 50:
        continue
    tb = grp["bet"].sum(); tr = grp["ret"].sum()
    roi = tr / tb * 100 if tb else 0
    top_bucket = grp[grp["bucket"] == "🔥超勝負"]
    bot_bucket = grp[grp["bucket"] == "🚫見送り"]
    top_roi = (top_bucket["ret"].sum() / top_bucket["bet"].sum() * 100) if len(top_bucket) else np.nan
    bot_roi = (bot_bucket["ret"].sum() / bot_bucket["bet"].sum() * 100) if len(bot_bucket) else np.nan
    print(f"  {sd}_{fb} n={len(grp):>4}  全体ROI={roi:>6.2f}%  "
          f"🔥={top_roi:>6.2f}%  🚫={bot_roi:>6.2f}%  "
          f"lift={top_roi - bot_roi:+.2f}pt")

print("\n完了。詳細は reports/race_confidence_backtest.csv を参照。")
