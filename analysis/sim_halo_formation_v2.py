"""
sim_halo_formation_v2.py
HALO 三連単フォーメーション 完全版バックテスト
  A) 旧 ◎◯2頭軸マルチ24点
  B) スコア差ルール フォーメーション
  C) 着順予測モデル (order_model_v1.pkl) フォーメーション ← 本番ロジック

データ源:
  - data/master_kako5.csv          (全特徴量)
  - reports/ensemble_predictions.csv (mark, ensemble_prob)
  - data/kekka_*.csv               (確定着順 + 三連単払戻)

使い方:
    python sim_halo_formation_v2.py
"""
from __future__ import annotations
import io, sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

BASE        = Path(__file__).parent
PRED_CSV    = BASE / "reports/ensemble_predictions.csv"
MASTER_CSV  = BASE / "data/master_kako5.csv"
KEKKA_CSV   = BASE / "data/kekka_20160105_20251228_v2.csv"
ORDER_MODEL = BASE / "models/order_model_v1.pkl"
WIN_MODEL   = BASE / "models/lgbm_win_v1.pkl"

BUDGET         = 9600
YEARS_TARGET   = {2024, 2025}
EXCLUDE_PLACES = {"東京", "小倉"}

# =========================================================
# 1. 着順モデル読み込み
# =========================================================
print("着順モデル読み込み...")
order_obj = joblib.load(ORDER_MODEL)
o_model = order_obj["model"]
o_feats = order_obj["features"]
o_encs  = order_obj["encoders"]
print(f"  order_model features: {len(o_feats)}")

print("lgbm_win 読み込み...")
win_obj   = joblib.load(WIN_MODEL)
w_model   = win_obj["model"]
w_feats   = win_obj["feature_cols"]
w_encs    = win_obj["encoders"]
print(f"  lgbm_win features: {len(w_feats)}")

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
    df = _prep(race_df, o_encs, o_feats)
    proba = o_model.predict_proba(df[o_feats])
    return {int(ub): float(proba[i, 0])
            for i, ub in enumerate(df["馬番"].values)
            if pd.notna(ub)}


def predict_win_proba(race_df: pd.DataFrame) -> dict[int, float]:
    """lgbm_win_v1 (binary 1着) で predict"""
    df = _prep(race_df, w_encs, w_feats)
    proba = w_model.predict_proba(df[w_feats])
    return {int(ub): float(proba[i, 1])  # class 1 = 1着
            for i, ub in enumerate(df["馬番"].values)
            if pd.notna(ub)}


# =========================================================
# 2. データ読み込み
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
# 3. レース構築
# =========================================================
print("レース構築 + 着順モデル推論中...")
races = []
n_proc = 0
for rid, p_grp in pred.groupby("race_id"):
    n_proc += 1
    if n_proc % 500 == 0:
        print(f"  ...{n_proc:,}/6909")
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

    # 着順モデル + lgbm_win 推論（master_kako5から特徴拾う）
    mgrp = master_grp.get(rid)
    pw_order = {}
    pw_win   = {}
    if mgrp is not None and len(mgrp) >= 5:
        try: pw_order = predict_order_proba(mgrp)
        except Exception: pass
        try: pw_win = predict_win_proba(mgrp)
        except Exception: pass

    races.append({
        "race_id": rid, "place": place,
        "marks": marks, "scores": scores,
        "pw_order": pw_order, "pw_win": pw_win,
        "first": first, "second": second, "third": third,
        "santan_pay": santan_pay,
    })

print(f"  構築完了: {len(races):,} R")
n_o = sum(1 for r in races if r["pw_order"])
n_w = sum(1 for r in races if r["pw_win"])
print(f"  order_model 推論: {n_o:,} R / lgbm_win 推論: {n_w:,} R")


# =========================================================
# 4. ロジック定義
# =========================================================
def logic_old(r):
    marks = r["marks"]
    h = marks.get("◎"); o = marks.get("◯")
    if h is None or o is None:
        return [], "旧24点"
    others = [marks[m] for m in ["▲","△","×"] if m in marks]
    combos = set()
    for f, s in [(h,o),(o,h)]:
        for t in others:
            if t in (f,s): continue
            combos.add((f,s,t))
        for mid in others:
            for t in [h,o]+others:
                if t in (f,mid) or mid==f: continue
                combos.add((f,mid,t))
    return list(combos), "旧24点"


def _score_pattern(marks, scores):
    h = marks.get("◎"); o = marks.get("◯")
    h3 = marks.get("▲"); h4 = marks.get("△"); h5 = marks.get("×")
    s_hon = scores.get(h, 0); s_tai = scores.get(o, 0)
    s_sab = scores.get(h3, 0) if h3 else 0
    s_del = scores.get(h4, 0) if h4 else 0
    gap_12 = (s_hon - s_tai) * 100
    gap_top4 = ((s_hon - s_del) if s_del>0 else (s_hon - s_sab)) * 100
    if gap_12 >= 10:
        return ("◎突出", [h], [x for x in [o,h3] if x], [x for x in [o,h3,h4,h5] if x])
    elif gap_12 <= 5 and gap_top4 <= 15:
        return ("◎◯拮抗", [h,o], [x for x in [h,o,h3] if x], [x for x in [h,o,h3,h4] if x])
    else:
        return ("標準", [h,o], [x for x in [h,o,h3,h4] if x], [x for x in [h,o,h3,h4,h5] if x])


def _model_pattern(marks, pw):
    h = marks.get("◎"); o = marks.get("◯")
    h3 = marks.get("▲"); h4 = marks.get("△"); h5 = marks.get("×")
    pw_h = pw.get(h, 0); pw_o = pw.get(o, 0)
    if pw_h >= pw_o * 2.0:
        return ("◎突出(AI)", [h], [x for x in [o,h3] if x], [x for x in [o,h3,h4,h5] if x])
    elif pw_o >= pw_h * 0.7:
        return ("◎◯拮抗(AI)", [h,o], [x for x in [h,o,h3] if x], [x for x in [h,o,h3,h4,h5] if x])
    else:
        return ("標準(AI)", [h,o], [x for x in [h,o,h3,h4] if x], [x for x in [h,o,h3,h4,h5] if x])


def _build(first, second, third):
    combos = set()
    for f in first:
        for s in second:
            for t in third:
                if len({f,s,t}) == 3:
                    combos.add((f,s,t))
    return list(combos)


def logic_score(r):
    marks = r["marks"]; scores = r["scores"]
    h = marks.get("◎"); o = marks.get("◯")
    if h is None or o is None: return [], "skip"
    pat, f, s, t = _score_pattern(marks, scores)
    return _build(f, s, t), pat


def logic_order_model(r):
    marks = r["marks"]; scores = r["scores"]; pw = r["pw_order"]
    h = marks.get("◎"); o = marks.get("◯")
    if h is None or o is None: return [], "skip"
    if not pw or h not in pw or o not in pw:
        pat, f, s, t = _score_pattern(marks, scores)
        return _build(f, s, t), pat + "[fb]"
    pat, f, s, t = _model_pattern(marks, pw)
    return _build(f, s, t), pat


def logic_win_model(r):
    marks = r["marks"]; scores = r["scores"]; pw = r["pw_win"]
    h = marks.get("◎"); o = marks.get("◯")
    if h is None or o is None: return [], "skip"
    if not pw or h not in pw or o not in pw:
        pat, f, s, t = _score_pattern(marks, scores)
        return _build(f, s, t), pat + "[fb]"
    pat, f, s, t = _model_pattern(marks, pw)
    return _build(f, s, t), pat


def logic_hybrid(r):
    """Stage 0 本番ロジック: スコア差を主軸＋AI高信頼の◎突出のみ補強。
    条件: pw_hon ≥ 0.50 かつ pw_hon ≥ pw_tai × 2.0 のときのみ AI 採用。"""
    marks = r["marks"]; scores = r["scores"]; pw = r["pw_order"]
    h = marks.get("◎"); o = marks.get("◯")
    if h is None or o is None: return [], "skip"
    # スコア差で第一決定
    pat, f, s, t = _score_pattern(marks, scores)
    pat += " [Hybrid]"
    # AI 高信頼ケースのみ ◎突出に上書き
    if pw and h in pw and o in pw:
        pw_h = pw[h]; pw_o = pw[o]
        if pw_h >= 0.50 and pw_h >= pw_o * 2.0:
            h3 = marks.get("▲"); h4 = marks.get("△"); h5 = marks.get("×")
            f = [h]
            s = [x for x in [o, h3] if x]
            t = [x for x in [o, h3, h4, h5] if x]
            pat = "◎突出(AI高信頼) [Hybrid]"
    return _build(f, s, t), pat


# =========================================================
# 5. シミュレーション
# =========================================================
def simulate(name, fn):
    tb = tr = hits = nR = 0
    pat_stats: dict[str, dict] = {}
    for r in races:
        combos, pat = fn(r)
        if not combos: continue
        n = len(combos)
        per = max(100, (BUDGET // n // 100) * 100)
        target = (r["first"], r["second"], r["third"])
        hit = target in set(combos)
        tb += per * n
        if hit:
            tr += per * r["santan_pay"] / 100.0
            hits += 1
        nR += 1
        d = pat_stats.setdefault(pat, {"R":0,"hit":0,"bet":0,"ret":0,"pts":0})
        d["R"] += 1; d["hit"] += 1 if hit else 0
        d["bet"] += per*n
        d["ret"] += (per*r["santan_pay"]/100.0) if hit else 0
        d["pts"] += n
    roi = tr/tb*100 if tb else 0
    hr = hits/nR*100 if nR else 0
    return {"name":name,"R":nR,"hits":hits,"hrate":hr,"bet":tb,"ret":tr,"roi":roi,"patterns":pat_stats}


def show(r):
    print(f"\n【{r['name']}】")
    print(f"  対象R={r['R']:,} 的中={r['hits']:,}({r['hrate']:.2f}%) "
          f"投資¥{r['bet']:,.0f} 払戻¥{r['ret']:,.0f} ROI={r['roi']:.2f}%")
    for pat, d in sorted(r["patterns"].items(), key=lambda x: -x[1]["R"]):
        roi = d["ret"]/d["bet"]*100 if d["bet"] else 0
        hr = d["hit"]/d["R"]*100 if d["R"] else 0
        ppr = d["pts"]/d["R"]
        print(f"    {pat:18s} R={d['R']:>5,} 的中={d['hit']:>4}({hr:>5.2f}%) "
              f"点数={ppr:>4.1f} ROI={roi:>6.2f}% 収支={d['ret']-d['bet']:+,.0f}")


print("\n=== シミュレーション実行 ===")
r_old = simulate("A) 旧24点マルチ",            logic_old)
r_scr = simulate("B) スコア差ルール",           logic_score)
r_mdl = simulate("C) order_model_v1",         logic_order_model)
r_win = simulate("D) lgbm_win_v1 (binary)",   logic_win_model)
r_hyb = simulate("E) Hybrid (Stage0 本番)",   logic_hybrid)

show(r_old)
show(r_scr)
show(r_mdl)
show(r_win)
show(r_hyb)

print("\n==========================================================")
print(f"A) 旧24点マルチ:           ROI {r_old['roi']:.2f}%")
print(f"B) スコア差ルール:          ROI {r_scr['roi']:.2f}%  ({r_scr['roi']-r_old['roi']:+.2f}pt)")
print(f"C) order_model_v1:        ROI {r_mdl['roi']:.2f}%  ({r_mdl['roi']-r_old['roi']:+.2f}pt)")
print(f"D) lgbm_win_v1:           ROI {r_win['roi']:.2f}%  ({r_win['roi']-r_old['roi']:+.2f}pt)")
print(f"E) Hybrid (Stage0 本番):  ROI {r_hyb['roi']:.2f}%  ({r_hyb['roi']-r_old['roi']:+.2f}pt)")
