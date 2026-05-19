"""
diag_trifecta_divergence.py
sim_halo_v3 (PRED_CSV 由来) と backtest_v2 (現行モデル直接予測) で
同じ 2025 レースに対する trifecta_model_v1 top3 を比較する。
"""
from __future__ import annotations
import sys
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import joblib

# streamlit スタブ
class _StStub:
    def __init__(self):
        self.cache_data = self._pt
        self.cache_resource = self._pt
        self.cache = self._pt
        self.session_state = {}
    def _pt(self, *a, **k):
        if len(a) == 1 and callable(a[0]) and not k:
            return a[0]
        def w(fn): return fn
        return w
    def __getattr__(self, n):
        m = MagicMock(name=f"st.{n}")
        setattr(self, n, m); return m
sys.modules["streamlit"] = _StStub()
sys.modules.setdefault("shap", MagicMock())

BASE = Path(r"E:\PyCaLiAI")
sys.path.insert(0, str(BASE))

PRED_CSV = BASE / "reports/ensemble_predictions.csv"
MASTER   = BASE / "data/master_20130105-20251228.csv"
MODEL    = BASE / "models/trifecta_model_v1.pkl"

from train_trifecta_model import add_race_features, pl_combo_probs, FEATURE_COLS
import backtest as bt_old

# === sim 経由の予測を使うパス ===
def combos_from_pred_csv(rid: str):
    pred = pd.read_csv(PRED_CSV, encoding="utf-8-sig")
    pred.columns = ["race_id", "umaban", "horse_name", "ensemble_prob", "mark", "fukusho_flag"]
    pred["race_id"] = pred["race_id"].astype(str)
    p_grp = pred[pred["race_id"] == rid]
    if p_grp.empty:
        return None, None, "no pred"
    rows = []
    for _, r in p_grp.iterrows():
        ub = int(r["umaban"]) if pd.notna(r["umaban"]) else None
        if ub is None: continue
        ep = float(r["ensemble_prob"]) if pd.notna(r["ensemble_prob"]) else 0.0
        mk = str(r["mark"]) if pd.notna(r["mark"]) else ""
        rows.append({"race_id": rid, "umaban": ub, "ensemble_prob": ep,
                     "mark": mk, "jyun": float("nan")})
    rdf = pd.DataFrame(rows)
    rdf = add_race_features(rdf)
    obj = joblib.load(MODEL)
    model = obj["model"]
    X = pd.DataFrame([{f: float(r.get(f, 0)) for f in FEATURE_COLS}
                      for _, r in rdf.iterrows()]).fillna(0)
    scores = model.predict(X)
    score_map = {int(rdf.iloc[i]["umaban"]): float(scores[i]) for i in range(len(rdf))}
    combos = pl_combo_probs(score_map, top_n=min(8, len(score_map)))[:3]
    probs_map = dict(zip(rdf["umaban"].astype(int), rdf["ensemble_prob"].astype(float)))
    return combos, probs_map, "ok"


# === backtest_v2 経由の予測を使うパス ===
def combos_from_live_model(rid: str):
    df = pd.read_csv(MASTER, encoding="utf-8-sig", low_memory=False)
    df["日付"] = pd.to_numeric(df["日付"], errors="coerce")
    df = df[(df["日付"] >= 20250101) & (df["日付"] < 20260101)].copy()
    COL = "レースID(新/馬番無)"
    race_df = df[df[COL].astype(str) == rid].copy()
    if race_df.empty:
        return None, None, "no race"
    race_df["prob"] = bt_old.ensemble_predict_batch(race_df)
    race_df = bt_old.assign_marks_df(race_df)
    # 準備
    rdf = race_df.copy()
    rdf["race_id"] = rid
    rdf["jyun"] = float("nan")
    rdf["ensemble_prob"] = pd.to_numeric(rdf["prob"], errors="coerce").fillna(0)
    rdf = add_race_features(rdf)
    obj = joblib.load(MODEL)
    model = obj["model"]
    X = pd.DataFrame([{f: float(r.get(f, 0)) for f in FEATURE_COLS}
                      for _, r in rdf.iterrows()]).fillna(0)
    scores = model.predict(X)
    umabans = [int(r["馬番"]) for _, r in race_df.iterrows() if pd.notna(r.get("馬番"))]
    score_map = dict(zip(umabans, [float(s) for s in scores]))
    combos = pl_combo_probs(score_map, top_n=min(8, len(score_map)))[:3]
    probs_map = dict(zip(umabans, race_df["prob"].astype(float)))
    return combos, probs_map, "ok"


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    # PRED_CSV から 2025 のレース ID を取る
    pred = pd.read_csv(PRED_CSV, encoding="utf-8-sig")
    pred.columns = ["race_id", "umaban", "horse_name", "ensemble_prob", "mark", "fukusho_flag"]
    pred["race_id"] = pred["race_id"].astype(str)
    # 2025 レース (race_id は YYYYMMDD から始まる)
    r2025 = [rid for rid in pred["race_id"].unique() if rid.startswith("2025")]
    print(f"2025 レース数 in PRED_CSV: {len(r2025)}")
    for rid in r2025[:3]:
        print(f"\n=== レース {rid} ===")
        s_combos, s_probs, _ = combos_from_pred_csv(rid)
        l_combos, l_probs, _ = combos_from_live_model(rid)
        print(f"sim 経由 top3 combos: {[c for c,_ in s_combos] if s_combos else None}")
        print(f"live 経由 top3 combos: {[c for c,_ in l_combos] if l_combos else None}")
        if s_probs and l_probs:
            # 共通 umaban で prob 比較
            common = sorted(set(s_probs) & set(l_probs))
            print(f"{'馬番':>4s} {'sim prob':>10s} {'live prob':>10s} {'diff':>10s}")
            for ub in common:
                sp = s_probs[ub]; lp = l_probs[ub]
                print(f"{ub:>4d} {sp:>10.4f} {lp:>10.4f} {lp-sp:>+10.4f}")
