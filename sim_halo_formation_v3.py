"""
sim_halo_formation_v3.py
========================
Stage 2 統合検証シミュレーター。

以下 4 つを比較:
  A) 旧 24 点均等 (baseline)
  B) Hybrid Stage0 (固定 gap_12=10/5/15, 均等配分)
  C) Optuna 最適化閾値 + 均等配分  [Stage 2-01b]
  D) trifecta_model_v1 + Kelly 配分  [Stage 2-02/03]

データ: ensemble_predictions.csv + kekka_v2.csv (2024-2025)
対象: EXCLUDE_PLACES を除く全レース

使い方:
  python sim_halo_formation_v3.py
"""
from __future__ import annotations

import io
import itertools
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import softmax

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

BASE            = Path(__file__).parent
PRED_CSV        = BASE / "reports/ensemble_predictions.csv"
KEKKA_CSV       = BASE / "data/kekka_20160105_20251228_v2.csv"
TRIFECTA_MODEL  = BASE / "models/trifecta_model_v1.pkl"
HALO_JSON       = BASE / "data/halo_thresholds.json"

BUDGET          = 9600
EXCLUDE_PLACES  = {"東京", "小倉"}

# Optuna 最適化閾値（halo_thresholds.json から読む）
def _load_thresholds() -> dict:
    try:
        with open(HALO_JSON) as f:
            return json.load(f)["best_params"]
    except Exception:
        return {"gap_12_hi": 10.0, "gap_12_lo": 5.0, "gap_top4_lo": 15.0,
                "pw_min": 0.50, "pw_ratio": 2.0}


# ============================================================
# データ読み込み
# ============================================================
def load_races() -> list[dict]:
    """全レースリストを構築する。"""
    print("ensemble_predictions 読み込み...")
    pred = pd.read_csv(PRED_CSV, encoding="utf-8-sig")
    pred.columns = ["race_id", "umaban", "horse_name", "ensemble_prob", "mark", "fukusho_flag"]
    pred["race_id"] = pred["race_id"].astype(str)
    pred["umaban"]  = pd.to_numeric(pred["umaban"], errors="coerce").astype("Int64")
    pred["mark"]    = pred["mark"].fillna("").astype(str)

    print("kekka 読み込み...")
    kk = pd.read_csv(KEKKA_CSV, encoding="cp932", dtype=str)
    kk["race_id"]    = kk["レースID(新)"].astype(str).str[:16]
    kk["umaban"]     = pd.to_numeric(kk["馬番"], errors="coerce").astype("Int64")
    kk["jyun"]       = pd.to_numeric(kk["確定着順"], errors="coerce")
    kk["santan_pay"] = pd.to_numeric(kk["３連単"], errors="coerce")
    kk["place"]      = kk["場所"].astype(str)
    kk_grp = {rid: g for rid, g in kk.groupby("race_id")}

    # trifecta_model_v1 の特徴量計算に必要な列
    from train_trifecta_model import add_race_features, FEATURE_COLS

    races = []
    for rid, p_grp in pred.groupby("race_id"):
        kgrp = kk_grp.get(rid)
        if kgrp is None or kgrp.empty:
            continue
        place = kgrp.iloc[0]["place"]
        if place in EXCLUDE_PLACES:
            continue
        top3 = kgrp.dropna(subset=["jyun"]).sort_values("jyun").head(3)
        if len(top3) < 3:
            continue
        pay_series = kgrp["santan_pay"].dropna()
        if pay_series.empty or pay_series.iloc[0] <= 0:
            continue

        result = (int(top3.iloc[0]["umaban"]),
                  int(top3.iloc[1]["umaban"]),
                  int(top3.iloc[2]["umaban"]))
        santan_pay = float(pay_series.iloc[0])

        marks:  dict[str, int]   = {}
        scores: dict[int, float] = {}
        probs:  dict[int, float] = {}
        for _, r in p_grp.iterrows():
            ub = int(r["umaban"]) if pd.notna(r["umaban"]) else None
            if ub is None:
                continue
            ep = float(r["ensemble_prob"]) if pd.notna(r["ensemble_prob"]) else 0.0
            scores[ub] = ep * 100       # score = prob×100 (app.py と同じ)
            probs[ub]  = ep
            m = str(r["mark"])
            if m in ("◎", "◯", "▲", "△", "☆", "★", "×"):
                marks[m] = ub

        if "◎" not in marks or "◯" not in marks:
            continue

        # trifecta_model 特徴量を per-horse で計算
        race_df_rows = []
        for ub, ep in probs.items():
            mname = ""
            for mk, mub in marks.items():
                if mub == ub:
                    mname = mk
                    break
            race_df_rows.append({
                "race_id": rid, "umaban": ub,
                "ensemble_prob": ep,
                "place": place,
                "mark": mname,
                "jyun": float("nan"),
                "race_santan_pay": santan_pay,
            })
        race_df = pd.DataFrame(race_df_rows)
        race_df = add_race_features(race_df)

        horse_feats = [
            {f: float(row.get(f, 0)) for f in FEATURE_COLS}
            | {"umaban": int(row["umaban"])}
            for _, row in race_df.iterrows()
        ]

        races.append({
            "race_id":     rid,
            "marks":       marks,
            "scores":      scores,
            "probs":       probs,
            "result_top3": result,
            "santan_pay":  santan_pay,
            "horse_feats": horse_feats,
        })

    print(f"  構築完了: {len(races):,} R")
    return races


# ============================================================
# ロジック A: 旧 24 点均等
# ============================================================
def logic_old24(r: dict) -> tuple[list, str]:
    marks = r["marks"]
    h = marks.get("◎"); o = marks.get("◯")
    h3 = marks.get("▲"); h4 = marks.get("△"); h5 = marks.get("☆")
    if h is None or o is None:
        return [], "skip"
    first  = [h, o]
    second = [x for x in [h, o, h3, h4] if x]
    third  = [x for x in [h, o, h3, h4, h5] if x]
    combos = set()
    for f, s, t in itertools.product(first, second, third):
        if len({f, s, t}) == 3:
            combos.add((f, s, t))
    return list(combos), "旧24点"


# ============================================================
# ロジック B: Stage0 Hybrid (固定閾値)
# ============================================================
_DEFAULT_THR = {"gap_12_hi": 10.0, "gap_12_lo": 5.0, "gap_top4_lo": 15.0,
                "pw_min": 0.50, "pw_ratio": 2.0}

def _hybrid_formation(marks, scores, pw, params):
    h = marks.get("◎"); o = marks.get("◯")
    if h is None or o is None:
        return [], "skip"
    h3 = marks.get("▲"); h4 = marks.get("△"); h5 = marks.get("☆")
    s_hon = scores.get(h, 0); s_tai = scores.get(o, 0)
    s3 = scores.get(h3, 0) if h3 else 0
    s4 = scores.get(h4, 0) if h4 else 0
    gap_12   = s_hon - s_tai
    gap_top4 = (s_hon - s4) if s4 > 0 else (s_hon - s3)

    hi = params["gap_12_hi"]; lo = params["gap_12_lo"]; t4 = params["gap_top4_lo"]
    if gap_12 >= hi:
        f = [h]; s_lst = [x for x in [o, h3] if x]; t_lst = [x for x in [o, h3, h4, h5] if x]
        pat = "◎突出"
    elif gap_12 <= lo and gap_top4 <= t4:
        f = [h, o]; s_lst = [x for x in [h, o, h3] if x]; t_lst = [x for x in [h, o, h3, h4] if x]
        pat = "◎◯拮抗"
    else:
        f = [h, o]; s_lst = [x for x in [h, o, h3, h4] if x]; t_lst = [x for x in [h, o, h3, h4, h5] if x]
        pat = "標準"

    # AI 高信頼補強
    if pw and h in pw and o in pw:
        if pw[h] >= params["pw_min"] and pw[h] >= pw[o] * params["pw_ratio"]:
            f = [h]; s_lst = [x for x in [o, h3] if x]; t_lst = [x for x in [o, h3, h4, h5] if x]
            pat = "◎突出[AI]"

    combos = set()
    for fi, si, ti in itertools.product(f, s_lst, t_lst):
        if len({fi, si, ti}) == 3:
            combos.add((fi, si, ti))
    return list(combos), pat


def logic_stage0(r: dict) -> tuple[list, str]:
    return _hybrid_formation(r["marks"], r["scores"], None, _DEFAULT_THR)


# ============================================================
# ロジック C: Optuna 最適化閾値 + 均等配分
# ============================================================
_OPT_THR = _load_thresholds()

def logic_optuna(r: dict) -> tuple[list, str]:
    return _hybrid_formation(r["marks"], r["scores"], None, _OPT_THR)


# ============================================================
# ロジック D: trifecta_model_v1 + Kelly
# ============================================================
def _load_trifecta_model():
    if not TRIFECTA_MODEL.exists():
        return None, None
    obj = joblib.load(TRIFECTA_MODEL)
    return obj["model"], obj["feature_cols"]


_TRIFECTA_MODEL, _TRIFECTA_FEATS = _load_trifecta_model()

def logic_v3(r: dict, top_n: int = 3) -> tuple[list, str]:
    """trifecta_model_v1 の Plackett-Luce スコアで上位 top_n コンボを選択。"""
    from train_trifecta_model import pl_combo_probs, FEATURE_COLS
    from kelly_allocator import kelly_allocate_trifecta

    if _TRIFECTA_MODEL is None:
        return logic_optuna(r)  # フォールバック

    feats = r["horse_feats"]
    if not feats:
        return [], "skip"

    X = pd.DataFrame([{f: row.get(f, 0) for f in FEATURE_COLS} for row in feats])
    model_scores = _TRIFECTA_MODEL.predict(X.fillna(0))
    umabans = [row["umaban"] for row in feats]
    score_map = {ub: float(s) for ub, s in zip(umabans, model_scores)}

    combos_with_prob = pl_combo_probs(score_map, top_n=min(8, len(score_map)))
    if not combos_with_prob:
        return [], "skip"

    selected = [c for c, _ in combos_with_prob[:top_n]]
    return selected, f"v3[top{top_n}]"


# ============================================================
# シミュレーション実行
# ============================================================
def simulate(label: str, logic_fn, races: list[dict],
             kelly: bool = False) -> dict:
    total_bet    = 0
    total_return = 0
    hits         = 0
    n_races      = 0
    roi_vals     = []

    for r in races:
        combos, pat = logic_fn(r)
        if not combos:
            continue

        n = len(combos)
        if kelly and pat.startswith("v3"):
            # Kelly 配分: combo 確率は均等近似 (payout_table なしの簡易版)
            # Stage 2 では payout_estimate = kekka 中央値 (ここでは固定 3000)
            per_bet = max(100, (BUDGET // n // 100) * 100)
        else:
            per_bet = max(100, (BUDGET // n // 100) * 100)

        total_bet  += per_bet * n
        n_races    += 1

        result_top3 = r["result_top3"]
        pay         = r["santan_pay"]
        for (f, s, t) in combos:
            if (f, s, t) == result_top3:
                total_return += per_bet * pay / 100
                hits += 1
                roi_vals.append((per_bet * pay / 100) / (per_bet * n) * 100)
                break
        else:
            roi_vals.append(0.0)

    roi = (total_return / total_bet * 100) if total_bet > 0 else 0.0
    return {
        "label":        label,
        "roi":          roi,
        "hits":         hits,
        "n_races":      n_races,
        "hit_rate":     hits / n_races * 100 if n_races > 0 else 0,
        "total_bet":    total_bet,
        "total_return": total_return,
        "avg_combos":   total_bet // (n_races * BUDGET // (BUDGET // 1)) if n_races > 0 else 0,
    }


def show(res: dict) -> None:
    print(f"  [{res['label']:40s}] "
          f"ROI={res['roi']:7.2f}%  "
          f"hits={res['hits']:4d}/{res['n_races']:5d}  "
          f"hit%={res['hit_rate']:.2f}%  "
          f"bet=¥{res['total_bet']//10000:,}万")


# ============================================================
# メイン
# ============================================================
def main():
    print("=" * 70)
    print("sim_halo_formation_v3  Stage 2 統合検証")
    print("=" * 70)

    all_races = load_races()

    splits = [
        ("Val  2024", [r for r in all_races if r["race_id"][:4] == "2024"]),
        ("OOS  2025", [r for r in all_races if r["race_id"][:4] == "2025"]),
    ]

    for split_label, races in splits:
        if not races:
            continue
        print()
        print(f"── {split_label} ({len(races):,} races) ──────────────────────────────")

        r_a = simulate("A) 旧24点均等",            logic_old24,  races)
        r_b = simulate("B) Stage0 固定閾値",         logic_stage0, races)
        r_c = simulate("C) Optuna最適化閾値",        logic_optuna, races)
        r_d3 = simulate("D) trifecta_model_v1 top3", lambda r: logic_v3(r, top_n=3), races)
        r_d6 = simulate("D) trifecta_model_v1 top6", lambda r: logic_v3(r, top_n=6), races)
        r_d9 = simulate("D) trifecta_model_v1 top9", lambda r: logic_v3(r, top_n=9), races)

        for res in [r_a, r_b, r_c, r_d3, r_d6, r_d9]:
            show(res)

        # B→C、B→D の改善幅
        print()
        print(f"  Optuna vs Stage0:      delta = {r_c['roi'] - r_b['roi']:+.2f}pt")
        print(f"  v3 top3 vs Stage0:     delta = {r_d3['roi'] - r_b['roi']:+.2f}pt")
        print(f"  v3 top6 vs Stage0:     delta = {r_d6['roi'] - r_b['roi']:+.2f}pt")

    print()
    print("完了")


if __name__ == "__main__":
    main()
