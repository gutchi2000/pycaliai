# -*- coding: utf-8 -*-
"""
calibration_drift_check.py — 現行較正器 (models/pl_calibrators_v6_serve.pkl) の 2026 実配信 ECE を監視する (2026-11-04)
==================================================================================================
較正器は valid 2023 で fit したまま。calibrator_shadow_eval が Phase A で「実際に配信した生スコア」を保存し
Phase C で kekka と突合している (reports/calibrator_shadow_scores_settled.parquet) ので、その直近 N 週に
現行較正器を当て直して ECE (単勝 = 1着, 複勝 = 3着内) を測り、fit 時の参照値
から離れたときだけ Discord へ出す。参照値は同じ settled parquet の「直近 window 週より前の全週」で測った
ECE (全頭プール、同一定義)。calibrators_v6_serve_eval.json の ece_*_hon は本命馬だけの値で母集団が違うため参考表示のみ。
再 fit は自動では行わない (build_pl_calibrators_serve.py + artifact_pins 更新 + 混戦度ゲート閾値の
百分位再導出が必要: memory project_calibrator_shifts_chaos_gate)。

判定: 直近 window 週 (既定 8) の ECE_fukusho が max(参照×1.5, 参照+0.02) を超え、かつ頭数 >= 1500 → DRIFT。
実行: python -m analysis.calibration_drift_check [--window 8] [--notify]
出力: reports/calibration_drift_latest.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "scripts"))

SETTLED = BASE / "reports" / "calibrator_shadow_scores_settled.parquet"
REF = BASE / "reports" / "calibrators_v6_serve_eval.json"
CUR_CAL = BASE / "models" / "pl_calibrators_v6_serve.pkl"
OUT = BASE / "reports" / "calibration_drift_latest.json"
MIN_HORSES = 1500


def _ece(p, y, bins: int = 10) -> float:
    p = np.asarray(p, float); y = np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(tot)


def _ref_values() -> dict:
    if not REF.exists():
        return {}
    e = json.loads(REF.read_text(encoding="utf-8"))
    sc = (e.get("serve_cal") or {}).get("test_2024_2025") or {}
    # キー名は eval スクリプトの版で揺れるので候補を順に探す
    def pick(*names):
        for n in names:
            if n in sc and isinstance(sc[n], (int, float)):
                return float(sc[n])
            for k, v in sc.items():
                if isinstance(v, dict) and n in v and isinstance(v[n], (int, float)):
                    return float(v[n])
        return None
    return {"ece_tansho": pick("ece_tansho", "ece_tan", "ece_win"),
            "ece_fukusho": pick("ece_fukusho", "ece_fuk", "ece_place", "ece_sho")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=8, help="直近 N 週")
    ap.add_argument("--notify", action="store_true")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    if not SETTLED.exists() or not CUR_CAL.exists():
        print(f"[drift] 入力なし (settled={SETTLED.exists()} cal={CUR_CAL.exists()}) → 判定しない")
        return 0
    from analysis.calibrator_shadow_eval import _apply_calibrators  # 本番と同じ適用関数
    settled = pd.read_parquet(SETTLED)
    settled = settled[settled["jyun"].notna()] if "jyun" in settled else settled
    weeks = sorted(settled["week"].astype(str).unique())
    recent_weeks = weeks[-a.window:]
    sub = settled[settled["week"].astype(str).isin(recent_weeks)].copy()
    cur = joblib.load(CUR_CAL)["calibrators"]
    res = _apply_calibrators(sub, cur, cur)
    n = int(len(res))
    out = {"generated_at": dt.datetime.now().isoformat(timespec="seconds"), "weeks": recent_weeks, "n_horses": n,
           "n_races": int(res.groupby(["week", "rid"]).ngroups) if n else 0}
    if n == 0:
        print("[drift] 直近週に決済済みデータなし"); OUT.write_bytes(json.dumps(out, ensure_ascii=False, indent=1).encode()); return 0
    ece_t = _ece(res["p_tan_cur"], res["y_1st"]); ece_f = _ece(res["p_fuk_cur"], res["y_top3"])
    # 参照 = 直近 window より前の全週 (同じ較正器・同じ母集団定義)。8 週未満なら参照なし (判定しない)
    base_weeks = weeks[:-a.window]
    ref = {"weeks": len(base_weeks), "info_hon_only": _ref_values()}
    if len(base_weeks) >= 8:
        bsub = settled[settled["week"].astype(str).isin(base_weeks)].copy()
        bres = _apply_calibrators(bsub, cur, cur)
        if len(bres):
            ref.update({"n_horses": int(len(bres)), "ece_tansho": _ece(bres["p_tan_cur"], bres["y_1st"]),
                        "ece_fukusho": _ece(bres["p_fuk_cur"], bres["y_top3"])})
    out.update({"ece_tansho": ece_t, "ece_fukusho": ece_f, "ref": ref,
                "mean_p_fuk": float(res["p_fuk_cur"].mean()), "rate_top3": float(res["y_top3"].mean()),
                "mean_p_tan": float(res["p_tan_cur"].mean()), "rate_1st": float(res["y_1st"].mean())})
    rf = ref.get("ece_fukusho")
    thr = max(rf * 1.5, rf + 0.02) if rf is not None else None
    drift = bool(thr is not None and n >= MIN_HORSES and ece_f > thr)
    out.update({"threshold_fukusho": thr, "drift": drift, "enough_sample": n >= MIN_HORSES})
    OUT.write_bytes(json.dumps(out, ensure_ascii=False, indent=1).encode("utf-8"))
    print(f"[drift] 直近 {len(recent_weeks)} 週 {out['n_races']}R {n} 頭 | ECE 単勝 {ece_t:.4f} 複勝 {ece_f:.4f} "
          f"| 参照(前 {ref.get('weeks')} 週) 複勝 {rf if rf is None else f'{rf:.4f}'} 閾値 {thr if thr is None else f'{thr:.4f}'} "
          f"| 平均p複 {out['mean_p_fuk']:.3f} vs 実 {out['rate_top3']:.3f} → {'DRIFT' if drift else 'OK'}")
    if drift and a.notify:
        try:
            from kaiko_warm import discord
            discord(f"**📐 較正ドリフト検知** 直近 {len(recent_weeks)} 週 ({recent_weeks[0]}〜{recent_weeks[-1]}, {out['n_races']}R)\n"
                    f"ECE 複勝 {ece_f:.4f} > 閾値 {thr:.4f} (前 {ref.get('weeks')} 週の参照 {rf:.4f})、平均 p複 {out['mean_p_fuk']:.3f} vs 実 3着内率 {out['rate_top3']:.3f}\n"
                    "再 fit は手動: build_pl_calibrators_serve.py → artifact_pins.json → 混戦度ゲート閾値の百分位再導出")
        except Exception as e:  # noqa: BLE001
            print(f"  [notify] 失敗 (無視): {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
