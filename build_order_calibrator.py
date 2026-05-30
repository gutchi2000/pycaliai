"""
build_order_calibrator.py
order_model_v1 の p_win / p_fuku を 2024 データで isotonic 校正する。
出力: models/order_calibrator_v1.pkl = {"p_win": IsotonicRegression, "p_fuku": IsotonicRegression}
"""
from __future__ import annotations
import sys, random
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np, pandas as pd

class _St:
    def __init__(self):
        self.cache_data=self._pt; self.cache_resource=self._pt; self.cache=self._pt; self.session_state={}
    def _pt(self,*a,**k):
        if len(a)==1 and callable(a[0]) and not k: return a[0]
        def w(fn): return fn
        return w
    def __getattr__(self,n):
        m=MagicMock(); setattr(self,n,m); return m
sys.modules['streamlit']=_St()
sys.modules.setdefault('shap', MagicMock())

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

import backtest as bt
import app
from app import _predict_order_proba


def build_actual_maps():
    """kekka_dict から (rid, 馬番) → (is_win, is_fuku) を作る。
    1着: 馬単 key の先頭。2/3着: 三連単 key。"""
    kd = bt.load_kekka(BASE / "data/kekka_20130105-20251228.csv")
    win_map, top3_map = {}, {}
    for rid, entry in kd.items():
        # 馬単の任意 1 key から 1-2 着
        umatan = next(iter(entry.get("馬単", {})), None)
        santan = next(iter(entry.get("三連単", {})), None)
        if santan:
            parts = santan.split("-")
            if len(parts) == 3:
                try:
                    a, b, c = int(parts[0]), int(parts[1]), int(parts[2])
                    win_map[(rid, a)] = 1
                    top3_map[(rid, a)] = 1
                    top3_map[(rid, b)] = 1
                    top3_map[(rid, c)] = 1
                    # 明示的に他の馬は 0 としない — in-race で後で補完
                except Exception:
                    pass
        elif umatan:
            parts = umatan.split("-")
            if len(parts) == 2:
                try:
                    a = int(parts[0])
                    win_map[(rid, a)] = 1
                    top3_map[(rid, a)] = 1
                except Exception:
                    pass
    return win_map, top3_map


def main():
    print("[1/4] kekka 実結果読込...")
    win_pos, top3_pos = build_actual_maps()
    print(f"  勝ち馬エントリ: {len(win_pos)}, 複勝圏: {len(top3_pos)}")

    print("[2/4] master 2024 読込...")
    df = pd.read_csv(BASE / "data/master_20130105-20251228.csv",
                     encoding="utf-8-sig", low_memory=False)
    df["日付"] = pd.to_numeric(df["日付"], errors="coerce")
    df = df[(df["日付"] >= 20240101) & (df["日付"] < 20250101)].copy()
    COL = "レースID(新/馬番無)"
    rids = sorted(df[COL].astype(str).unique())
    random.seed(0)
    random.shuffle(rids)
    rids = rids[:2500]
    print(f"  対象レース: {len(rids)}")

    pw, pf, hw, hf = [], [], [], []
    for i, rid in enumerate(rids):
        if i % 300 == 0:
            print(f"  {i}/{len(rids)}")
        rd = df[df[COL].astype(str) == rid].copy()
        if len(rd) < 5:
            continue
        rd["prob"] = bt.ensemble_predict_batch(rd)
        op = _predict_order_proba(rd)
        if op is None:
            continue
        for _, r in op.iterrows():
            try:
                ban = int(r["馬番"])
            except Exception:
                continue
            key = (rid, ban)
            # in-race actual: 勝ち=1 なら 1, それ以外は 0
            is_win = 1 if win_pos.get(key) == 1 else 0
            is_fuku = 1 if top3_pos.get(key) == 1 else 0
            pw.append(float(r["p_win"]))
            pf.append(float(r["p_win"]) + float(r["p_place23"]))
            hw.append(is_win)
            hf.append(is_fuku)

    pw = np.array(pw); pf = np.array(pf); hw = np.array(hw); hf = np.array(hf)
    print(f"[3/4] サンプル n={len(pw)}  actual win_rate={hw.mean():.3f} fuku_rate={hf.mean():.3f}")
    print(f"  p_win  mean={pw.mean():.3f} p50={np.median(pw):.3f} p90={np.quantile(pw,0.9):.3f}")
    print(f"  p_fuku mean={pf.mean():.3f} p50={np.median(pf):.3f} p90={np.quantile(pf,0.9):.3f}")

    # bin ごとの校正前ギャップ
    for lo, hi in [(0,.05),(.05,.1),(.1,.15),(.15,.25),(.25,.4),(.4,1.0)]:
        m = (pw >= lo) & (pw < hi)
        if m.sum() > 20:
            print(f"  p_win {lo:.2f}-{hi:.2f} n={m.sum():5d} pred={pw[m].mean():.3f} actual={hw[m].mean():.3f}")

    print("[4/4] isotonic 校正...")
    from sklearn.isotonic import IsotonicRegression
    import joblib
    iso_win  = IsotonicRegression(out_of_bounds="clip").fit(pw, hw)
    iso_fuku = IsotonicRegression(out_of_bounds="clip").fit(pf, hf)

    # 校正後チェック
    pw_c  = iso_win.transform(pw)
    pf_c  = iso_fuku.transform(pf)
    print(f"  校正後 p_win  mean={pw_c.mean():.3f}")
    print(f"  校正後 p_fuku mean={pf_c.mean():.3f}")

    out = BASE / "models/order_calibrator_v1.pkl"
    joblib.dump({"p_win": iso_win, "p_fuku": iso_fuku,
                 "n_samples": int(len(pw)),
                 "train_year": 2024,
                 "actual_win_rate": float(hw.mean()),
                 "actual_fuku_rate": float(hf.mean())}, out)
    print(f"✓ 保存: {out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
