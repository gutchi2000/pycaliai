# -*- coding: utf-8 -*-
"""
predict_and_store.py — 凍結モデルで予測しappend-onlyで保存する (spec §12)
============================================================================
market_snapshot.py の process_race() から呼ばれる。同一 race_id×horse_id×model_hash の
レコードは上書きしない (修正が必要な場合は新しい revision として追記、主評価は
期限内に生成された最初の revision だけを使う)。
"""
from __future__ import annotations
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
from analysis.mcond.v6base import pl_top3  # noqa: E402
from analysis.mcond.exp05_forward_shadow import freeze_model as FM  # noqa: E402
from analysis.mcond.exp05_market_residual_dev import models as M  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT_DIR = BASE / "data" / "_research" / "mcond" / "exp05fs_predictions"
FEATURES_DIR = BASE / "data" / "_research" / "mcond" / "exp05fs_features"
FROZEN_PATH = HERE / "out" / "frozen_model.joblib"
SPEC = json.loads((HERE / "spec.json").read_text(encoding="utf-8")) if (HERE / "spec.json").exists() else {}

_frozen_cache = None


def _frozen():
    global _frozen_cache
    if _frozen_cache is None:
        _frozen_cache = joblib.load(FROZEN_PATH)
    return _frozen_cache


def _artifact_hash() -> str:
    manifest = json.loads((HERE / "out" / "freeze_manifest.json").read_text(encoding="utf-8"))
    return manifest["artifact_sha256_16"]


def _load_bundle_race(date_str: str, rid: str) -> dict:
    import masters_vote as mv
    return mv.load_bundle_race(date_str, rid)


def _devig_market(tansho: dict, active_bans: list[int]) -> dict[int, float]:
    raw = {int(b): 1.0 / float(tansho[str(b)]) for b in active_bans if str(b) in tansho
          and float(tansho[str(b)]) > 0}
    tot = sum(raw.values())
    if tot <= 0:
        return {}
    return {b: v / tot for b, v in raw.items()}


def _revision_path(date_str: str, rid: str, model_hash: str, rev: int) -> Path:
    d = OUT_DIR / date_str
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{rid}_{model_hash}_rev{rev}.json"


def _existing_revisions(date_str: str, rid: str, model_hash: str) -> list[Path]:
    d = OUT_DIR / date_str
    if not d.exists():
        return []
    return sorted(d.glob(f"{rid}_{model_hash}_rev*.json"))


def store_prediction(date_str: str, rid: str, market_result: dict) -> Path:
    frozen = _frozen()
    model_hash = _artifact_hash()

    race = _load_bundle_race(date_str, rid)
    horses = race.get("horses") or []
    active = [h for h in horses if isinstance(h.get("ai_score"), (int, float))]
    if len(active) < 3:
        raise ValueError(f"有効な馬が3頭未満 ({len(active)})")

    feat_path = FEATURES_DIR / f"{date_str}.parquet"
    if not feat_path.exists():
        raise FileNotFoundError(f"特徴量snapshot未生成: {feat_path} "
                                "(先に feature_snapshot.py --date を実行すること)")
    feats = pd.read_parquet(feat_path)
    feats = feats[feats["rid16"] == rid].set_index("ban")

    scores = np.array([h["ai_score"] for h in active], dtype=float)
    bans = [int(h["umaban"]) for h in active]
    tau = frozen["tau_2026_estimate"]
    z = scores / tau
    w = np.exp(z - z.max())
    v6_pwin = w / w.sum()
    v6_p3 = pl_top3(w) if len(w) >= 3 else np.full(len(w), np.nan)
    rank_v6 = pd.Series(-scores).rank(method="first").to_numpy()

    market_ok = bool(market_result.get("ok")) and bool(market_result.get("valid_for_primary"))
    market_p = {}
    if market_ok:
        tansho = market_result["market"].get("tansho") or {}
        market_p = _devig_market(tansho, bans)

    cal_top3 = frozen["calibrator_top3"].predict(v6_p3)
    cal_top3 = np.where(np.isnan(v6_p3), np.nan, cal_top3)

    mkt_arr = np.array([market_p.get(b, np.nan) for b in bans])
    EPS = 1e-6
    f_mkt = np.log(np.clip(mkt_arr, EPS, 1 - EPS) / (1 - np.clip(mkt_arr, EPS, 1 - EPS)))
    lp_cal = np.log(np.clip(cal_top3, EPS, 1 - EPS) / (1 - np.clip(cal_top3, EPS, 1 - EPS)))

    mx = scores.max()
    sd = scores.std()
    sorted_desc = np.sort(scores)[::-1]
    second = sorted_desc[1] if len(sorted_desc) > 1 else sorted_desc[0]
    score_gap_to_top = scores - mx
    score_gap_to_second = scores - second
    score_pct = pd.Series(scores).rank(pct=True).to_numpy()
    field_disp = np.full(len(scores), sd)

    X_m1 = np.column_stack([lp_cal, f_mkt])
    X_m3 = np.column_stack([lp_cal, f_mkt, scores, rank_v6, score_gap_to_top, score_gap_to_second,
                            score_pct, field_disp])
    p_m1 = FM.predict_linear(frozen["M1"], X_m1)
    p_m3 = FM.predict_linear(frozen["M3"], X_m3)

    f_serve_cols = frozen["F_serve_cols"]
    fmiss = [c for c in f_serve_cols if c not in feats.columns]
    Xf = pd.DataFrame(index=bans, columns=f_serve_cols, dtype=float)
    for b in bans:
        if b in feats.index:
            row = feats.loc[b]
            for c in f_serve_cols:
                Xf.loc[b, c] = row[c] if c in row.index else np.nan
    beta = np.array(frozen["M4"]["beta"])
    mu4, sd4 = np.array(frozen["M4"]["mu"]), np.array(frozen["M4"]["sd"])
    Xz = (Xf.to_numpy(dtype=float) - mu4) / sd4
    Xz = np.where(np.isnan(Xz), 0.0, Xz)
    offset3 = np.log(np.clip(p_m3, EPS, 1 - EPS) / (1 - np.clip(p_m3, EPS, 1 - EPS)))
    z4 = offset3 + Xz @ beta
    p_m4 = 1.0 / (1.0 + np.exp(-np.clip(z4, -30, 30)))

    now = datetime.now().isoformat(timespec="seconds")
    records = []
    for i, h in enumerate(active):
        b = bans[i]
        edge_m1 = p_m1[i] / mkt_arr[i] if market_ok and mkt_arr[i] == mkt_arr[i] else None
        edge_m3 = p_m3[i] / mkt_arr[i] if market_ok and mkt_arr[i] == mkt_arr[i] else None
        edge_m4 = p_m4[i] / mkt_arr[i] if market_ok and mkt_arr[i] == mkt_arr[i] else None
        r1 = bool(edge_m4 is not None and edge_m4 >= 1.15)
        rec = {
            "race_id": rid, "horse_id": b, "umaban": b, "horse_name": h.get("horse_name"),
            "race_date": date_str, "scheduled_start_time": market_result.get("scheduled_post"),
            "prediction_time": now, "snapshot_time": market_result.get("market", {}).get("fetched"),
            "model_version": "exp05fs_v1", "model_hash": model_hash,
            "feature_schema_hash": "exp05fs_v1",
            "source_data_hash": frozen.get("feature_typing_sha256_16"),
            "v6_raw_score": float(h["ai_score"]), "v6_rank": float(rank_v6[i]),
            "v6_calibrated_probability": float(cal_top3[i]) if cal_top3[i] == cal_top3[i] else None,
            "market_probability": float(mkt_arr[i]) if mkt_arr[i] == mkt_arr[i] else None,
            "M1_probability": float(p_m1[i]), "M3_probability": float(p_m3[i]),
            "M4_probability": float(p_m4[i]),
            "edge_M1": edge_m1, "edge_M3": edge_m3, "edge_M4": edge_m4,
            "R1_virtual_action": "bet_100yen_fukusho" if r1 else "no_bet",
            "R2_virtual_action": None,  # レース単位で後段に埋める (最大p1頭)
            "virtual_stake": 100 if r1 else 0,
            "valid_for_primary": bool(market_result.get("valid_for_primary")),
            "invalid_reason": market_result.get("why") if not market_result.get("valid_for_primary") else None,
            "feature_missing_columns": fmiss,
        }
        records.append(rec)

    top_i = int(np.argmax(p_m4))
    records[top_i]["R2_virtual_action"] = "bet_100yen_fukusho_race_top1"
    for i, r in enumerate(records):
        if i != top_i:
            r["R2_virtual_action"] = "no_bet"

    model_hash_tag = model_hash
    existing = _existing_revisions(date_str, rid, model_hash_tag)
    rev = len(existing) + 1
    path = _revision_path(date_str, rid, model_hash_tag, rev)
    path.write_text(json.dumps({"race_id": rid, "date": date_str, "revision": rev,
                                "saved_at": now, "records": records}, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    print(f"  [predict_and_store] {len(records)}頭 -> {path.relative_to(BASE)} (revision {rev})")
    return path
