"""
grid_search_playbook.py
========================
Stage 2-05: HALO フォーメーション playbook 多次元生成。

目的:
  trifecta_model_v1 のレース条件別最適パラメータを探索し、
  data/halo_formation_playbook.json に保存する。

条件セル (18 マス):
  - 芝ダ        : 2 (芝 / ダ)
  - field_bucket: 3 (少 <=10 / 中 11-14 / 多 15+)
  - ◎信頼度     : 3 (low prob_z<0.8 / mid 0.8-1.5 / high >=1.5)

各セルで探索するポリシー:
  - top_n   ∈ {3, 4, 5, 6, 9}         # Plackett-Luce 上位何点を買うか
  - ev_gate ∈ {0.0, 0.85, 0.95, 1.05}  # 期待 ROI がこれ未満ならレーススキップ
  - NO_BET (このセルは買わない)

データ分割:
  Validation : 2024 (チューニング)
  Test OOS   : 2025 (最終評価)

出力:
  data/halo_formation_playbook.json
  - cells[cell_key] = {top_n, ev_gate, val_roi, val_n, oos_roi, oos_n, note}

使い方:
  python grid_search_playbook.py
"""
from __future__ import annotations

import io
import itertools
import json
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import softmax

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

BASE           = Path(__file__).parent
PRED_CSV       = BASE / "reports/ensemble_predictions.csv"
KEKKA_CSV      = BASE / "data/kekka_20160105_20251228_v2.csv"
MASTER_CSV     = BASE / "data/master_kako5.csv"
TRIFECTA_MODEL = BASE / "models/trifecta_model_v1.pkl"
PAYOUT_PARQUET = BASE / "data/payout_table.parquet"
OUT_JSON       = BASE / "data/halo_formation_playbook.json"

BUDGET         = 9600
EXCLUDE_PLACES = {"東京", "小倉"}

TOP_N_CAND   = [3, 5]               # overfit 抑制のため候補を絞る
EV_GATE_CAND = [0.0, 0.90, 1.00]
MIN_SAMPLES_PER_CELL = 50
MIN_VAL_DELTA_TO_DEVIATE = 30.0     # baseline(top3,gate=0) を +30pt 超えた時のみ deviate
NO_BET_THRESHOLD = 100.0            # baseline val ROI < 100% の cell はスキップ


# ============================================================
# ユーティリティ
# ============================================================
def dist_bucket(d) -> str:
    try:
        d = int(d)
    except Exception:
        return "?"
    if d <= 1400: return "短"
    if d <= 1700: return "マイル"
    if d <= 2200: return "中"
    return "長"


def field_bucket(n: int) -> str:
    if n <= 10: return "少"
    if n <= 14: return "中"
    return "多"


def pop_bucket(p) -> str:
    if pd.isna(p): return "?"
    p = int(p)
    if p == 1: return "1"
    if p == 2: return "2"
    if p == 3: return "3"
    if p <= 6: return "4-6"
    return "7+"


def confidence_bucket(prob_z: float) -> str:
    if prob_z < 0.8:  return "low"
    if prob_z < 1.5:  return "mid"
    return "high"


# ============================================================
# データ読み込み（レース構築）
# ============================================================
def load_races() -> list[dict]:
    """全レースを構築。場所・芝ダ・距離・頭数・人気マップ付き。"""
    print("ensemble_predictions 読み込み...")
    pred = pd.read_csv(PRED_CSV, encoding="utf-8-sig")
    pred.columns = ["race_id", "umaban", "horse_name", "ensemble_prob", "mark", "fukusho_flag"]
    pred["race_id"]       = pred["race_id"].astype(str)
    pred["umaban"]        = pd.to_numeric(pred["umaban"], errors="coerce").astype("Int64")
    pred["mark"]          = pred["mark"].fillna("").astype(str)

    print("kekka 読み込み...")
    kk = pd.read_csv(KEKKA_CSV, encoding="cp932", dtype=str)
    kk["race_id"]    = kk["レースID(新)"].astype(str).str[:16]
    kk["umaban"]     = pd.to_numeric(kk["馬番"], errors="coerce").astype("Int64")
    kk["jyun"]       = pd.to_numeric(kk["確定着順"], errors="coerce")
    kk["santan_pay"] = pd.to_numeric(kk["３連単"], errors="coerce")
    kk["place"]      = kk["場所"].astype(str)

    # 単勝オッズ: 1着=配当/100、それ以外=(N.N) 形式
    def _parse_tan(s: str) -> float:
        if not s or str(s) == "nan":
            return float("nan")
        s = str(s).strip()
        if s.startswith("("):
            m = re.match(r"\(([\d.]+)\)", s)
            return float(m.group(1)) if m else float("nan")
        try:
            return float(s) / 100.0
        except Exception:
            return float("nan")
    kk["tan_odds"] = kk["単勝配当"].apply(_parse_tan)
    kk["popularity"] = (
        kk.groupby("race_id")["tan_odds"]
          .rank(method="min", ascending=True).astype("Int64")
    )
    kk_grp = {rid: g for rid, g in kk.groupby("race_id")}

    print("master_kako5 読み込み (芝ダ/距離取得)...")
    master = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    master["race_id"] = master["レースID(新/馬番無)"].astype(str)
    master_meta = master.groupby("race_id").agg(
        shiba_da=("芝・ダ", "first"),
        kyori=("距離", "first"),
    ).to_dict("index")

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

        # 人気マップ (popularity 列そのまま利用)
        pop_map: dict[int, int] = {}
        for _, rr in kgrp.iterrows():
            if pd.isna(rr["umaban"]) or pd.isna(rr["popularity"]):
                continue
            pop_map[int(rr["umaban"])] = int(rr["popularity"])

        result = (int(top3.iloc[0]["umaban"]),
                  int(top3.iloc[1]["umaban"]),
                  int(top3.iloc[2]["umaban"]))
        santan_pay = float(pay_series.iloc[0])

        marks: dict[str, int] = {}
        probs: dict[int, float] = {}
        for _, r in p_grp.iterrows():
            ub = int(r["umaban"]) if pd.notna(r["umaban"]) else None
            if ub is None:
                continue
            probs[ub] = float(r["ensemble_prob"]) if pd.notna(r["ensemble_prob"]) else 0.0
            m = str(r["mark"])
            if m in ("◎", "◯", "▲", "△", "☆", "★", "×"):
                marks[m] = ub

        if "◎" not in marks or "◯" not in marks:
            continue

        # 特徴量計算
        race_df_rows = []
        for ub, ep in probs.items():
            mname = next((mk for mk, mub in marks.items() if mub == ub), "")
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

        # ◎の prob_z を信頼度指標に
        hon_ub = marks["◎"]
        hon_row = race_df[race_df["umaban"] == hon_ub]
        honprob_z = float(hon_row.iloc[0]["prob_z"]) if not hon_row.empty else 0.0

        meta = master_meta.get(rid, {})
        shiba_da = str(meta.get("shiba_da", "")) or "?"
        try:
            kyori_i = int(float(meta.get("kyori", 1600)))
        except Exception:
            kyori_i = 1600

        races.append({
            "race_id":      rid,
            "place":        place,
            "shiba_da":     shiba_da,
            "kyori":        kyori_i,
            "field_size":   len(probs),
            "dist_bucket":  dist_bucket(kyori_i),
            "field_bucket": field_bucket(len(probs)),
            "conf_bucket":  confidence_bucket(honprob_z),
            "honprob_z":    honprob_z,
            "pop_map":      pop_map,
            "result_top3":  result,
            "santan_pay":   santan_pay,
            "horse_feats":  horse_feats,
        })

    print(f"  構築完了: {len(races):,} R")
    return races


# ============================================================
# payout_table 参照（三連単のみ）
# ============================================================
class PayoutLookup:
    """三連単 payout_table の EV 参照ヘルパ。"""

    def __init__(self, parquet_path: Path):
        df = pd.read_parquet(parquet_path)
        df = df[df["馬券種"] == "三連単"].copy()
        # key: (人気パターン, 場所, 芝ダ, dist_bucket, field_bucket) -> median
        self.exact: dict = {}
        for _, r in df.iterrows():
            key = (str(r["人気パターン"]), str(r["場所"]), str(r["芝ダ"]),
                   str(r["dist_bucket"]), str(r["field_bucket"]))
            self.exact[key] = float(r["median"])

        # フォールバック用: 人気パターン単独の全会場平均
        self.fallback_by_pop: dict[str, float] = (
            df.groupby("人気パターン")["median"].median().to_dict()
        )
        self.fallback_by_pop = {str(k): float(v) for k, v in self.fallback_by_pop.items()}
        self.global_median = float(df["median"].median())

    @staticmethod
    def pop_pattern(pops: tuple[int, int, int]) -> str:
        """三連単人気パターン文字列 (順序付き)。"""
        return "-".join(pop_bucket(p) for p in pops)

    def lookup(self, place: str, shiba_da: str, dist_b: str,
               field_b: str, pops: tuple[int, int, int]) -> float:
        key = (self.pop_pattern(pops), place, shiba_da, dist_b, field_b)
        if key in self.exact:
            return self.exact[key]
        fb = self.fallback_by_pop.get(self.pop_pattern(pops))
        if fb is not None:
            return fb
        return self.global_median


# ============================================================
# trifecta_model_v1 推論
# ============================================================
def _load_trifecta_model():
    obj = joblib.load(TRIFECTA_MODEL)
    return obj["model"], obj["feature_cols"]


def build_race_predictions(races: list[dict]) -> None:
    """各 race に trifecta_model_v1 の combos (top 9) と PL 確率を付与。"""
    from train_trifecta_model import pl_combo_probs, FEATURE_COLS
    model, _ = _load_trifecta_model()

    print("trifecta_model 推論中...")
    for r in races:
        feats = r["horse_feats"]
        if not feats:
            r["combos_all"] = []
            continue
        X = pd.DataFrame([{f: row.get(f, 0) for f in FEATURE_COLS} for row in feats])
        pred = model.predict(X.fillna(0))
        umabans = [row["umaban"] for row in feats]
        score_map = {ub: float(s) for ub, s in zip(umabans, pred)}
        combos = pl_combo_probs(score_map, top_n=min(8, len(score_map)))
        r["combos_all"] = combos[:9]  # 上位 9 点まで保持


# ============================================================
# ポリシー評価
# ============================================================
def evaluate_policy(races: list[dict], payout: PayoutLookup,
                    top_n: int, ev_gate: float) -> tuple[float, int, int, int]:
    """指定ポリシーで races を評価。
    Returns: (roi_pct, played_races, hits, total_bet)
    """
    total_bet    = 0
    total_return = 0
    hits         = 0
    played       = 0

    for r in races:
        combos = r.get("combos_all", [])
        if not combos:
            continue
        selected = combos[:top_n]
        if not selected:
            continue

        # 期待 ROI を計算してゲート
        expected_return = 0.0
        for combo, prob in selected:
            pops = tuple(r["pop_map"].get(ub, 7) for ub in combo)
            pay = payout.lookup(r["place"], r["shiba_da"],
                                r["dist_bucket"], r["field_bucket"], pops)
            expected_return += prob * pay
        per_bet = BUDGET / len(selected)
        exp_roi = expected_return / BUDGET if BUDGET > 0 else 0.0

        if exp_roi < ev_gate:
            continue  # 期待 ROI 不足 → スキップ

        # 実際の投票
        per_bet_100 = max(100, int(per_bet // 100) * 100)
        total_bet += per_bet_100 * len(selected)
        played    += 1
        for combo, _ in selected:
            if combo == r["result_top3"]:
                total_return += per_bet_100 * r["santan_pay"] / 100
                hits += 1
                break

    roi = (total_return / total_bet * 100) if total_bet > 0 else 0.0
    return roi, played, hits, total_bet


# ============================================================
# セル探索
# ============================================================
def cell_key(r: dict) -> tuple[str, str]:
    # 粗く: (芝ダ, field_bucket) のみの 6 マス
    return (r["shiba_da"], r["field_bucket"])


def grid_search(val_races: list[dict], test_races: list[dict],
                payout: PayoutLookup) -> dict:
    # セル別にレース分割
    cells_val  = defaultdict(list)
    cells_test = defaultdict(list)
    for r in val_races:  cells_val[cell_key(r)].append(r)
    for r in test_races: cells_test[cell_key(r)].append(r)

    # ベースライン (全レース top3, gate=0)
    bl_val_roi,  _, _, _  = evaluate_policy(val_races,  payout, 3, 0.0)
    bl_test_roi, _, _, _  = evaluate_policy(test_races, payout, 3, 0.0)
    print(f"\nベースライン (全レース top3 gate=0):")
    print(f"  val  ROI: {bl_val_roi:6.2f}%  ({len(val_races):,} R)")
    print(f"  test ROI: {bl_test_roi:6.2f}%  ({len(test_races):,} R)")

    playbook = {}
    all_cells = sorted(set(cells_val.keys()) | set(cells_test.keys()))

    print(f"\n{'cell':<12} {'n_val':>6} {'policy':<18} metrics")
    print("-" * 80)

    for cell in all_cells:
        vr = cells_val[cell]
        tr = cells_test[cell]
        cell_str = "/".join(cell)

        if len(vr) < MIN_SAMPLES_PER_CELL:
            playbook["_".join(cell)] = {
                "top_n": 3, "ev_gate": 0.0,
                "val_roi": None, "val_n": len(vr),
                "oos_roi": None, "oos_n": len(tr),
                "note": f"sample_n<{MIN_SAMPLES_PER_CELL}, fallback to baseline(top3,gate=0)",
            }
            continue

        # セル別ベースライン (top3, gate=0) を基準
        base_val_roi, base_val_n, base_val_hits, _ = evaluate_policy(vr, payout, 3, 0.0)
        base_oos_roi, base_oos_n, base_oos_hits, _ = evaluate_policy(tr, payout, 3, 0.0)

        # NO_BET 判定: ベースライン val < NO_BET_THRESHOLD
        if base_val_roi < NO_BET_THRESHOLD:
            playbook["_".join(cell)] = {
                "top_n": 0, "ev_gate": 999.0,
                "val_roi": round(base_val_roi, 2), "val_n": base_val_n,
                "oos_roi": round(base_oos_roi, 2), "oos_n": base_oos_n,
                "note": f"NO_BET (base val_roi {base_val_roi:.1f} < {NO_BET_THRESHOLD})",
            }
            print(f"{cell_str:<12} {len(vr):>6} {'NO_BET':<18} "
                  f"base_val={base_val_roi:>6.1f}% base_oos={base_oos_roi:>6.1f}%")
            continue

        # ベースライン超えのポリシーを探索
        best_alt = None
        for top_n, ev_gate in itertools.product(TOP_N_CAND, EV_GATE_CAND):
            if top_n == 3 and ev_gate == 0.0:
                continue  # baseline はスキップ
            val_roi, val_played, val_hits, _ = evaluate_policy(vr, payout, top_n, ev_gate)
            if val_played < 30:
                continue
            if best_alt is None or val_roi > best_alt["val_roi"]:
                best_alt = {"top_n": top_n, "ev_gate": ev_gate,
                            "val_roi": val_roi, "val_n": val_played, "val_hits": val_hits}

        # 保守的条件: alt が base を +MIN_VAL_DELTA_TO_DEVIATE 超えたら採用
        deviate = (best_alt is not None
                   and best_alt["val_roi"] - base_val_roi >= MIN_VAL_DELTA_TO_DEVIATE)

        if deviate:
            oos_roi, oos_played, oos_hits, _ = evaluate_policy(
                tr, payout, best_alt["top_n"], best_alt["ev_gate"])
            playbook["_".join(cell)] = {
                "top_n":   best_alt["top_n"],
                "ev_gate": best_alt["ev_gate"],
                "val_roi": round(best_alt["val_roi"], 2),
                "val_n":   best_alt["val_n"],
                "val_hits": best_alt["val_hits"],
                "oos_roi": round(oos_roi, 2),
                "oos_n":   oos_played,
                "oos_hits": oos_hits,
                "note":    f"deviated from baseline (+{best_alt['val_roi']-base_val_roi:.1f}pt val)",
            }
            pol_str = f"top{best_alt['top_n']} gate{best_alt['ev_gate']}"
            print(f"{cell_str:<12} {len(vr):>6} {pol_str:<18} "
                  f"val={best_alt['val_roi']:>6.1f}%  oos={oos_roi:>6.1f}%  "
                  f"(base val={base_val_roi:.1f})")
        else:
            playbook["_".join(cell)] = {
                "top_n": 3, "ev_gate": 0.0,
                "val_roi": round(base_val_roi, 2), "val_n": base_val_n,
                "val_hits": base_val_hits,
                "oos_roi": round(base_oos_roi, 2), "oos_n": base_oos_n,
                "oos_hits": base_oos_hits,
                "note":    "baseline (top3,gate=0) - alt not significantly better",
            }
            print(f"{cell_str:<12} {len(vr):>6} {'top3 gate0.0 (base)':<18} "
                  f"val={base_val_roi:>6.1f}%  oos={base_oos_roi:>6.1f}%")

    return {
        "baseline": {
            "val_roi":  round(bl_val_roi, 2),
            "test_roi": round(bl_test_roi, 2),
            "n_val":    len(val_races),
            "n_test":   len(test_races),
        },
        "cells": playbook,
    }


# ============================================================
# メイン
# ============================================================
def main():
    print("=" * 70)
    print("grid_search_playbook: HALO 条件別ポリシー探索")
    print("=" * 70)

    all_races = load_races()

    print("\npayout_table 読み込み...")
    payout = PayoutLookup(PAYOUT_PARQUET)
    print(f"  lookup keys: {len(payout.exact):,}  pop fallback: {len(payout.fallback_by_pop)}")

    val_races  = [r for r in all_races if r["race_id"][:4] == "2024"]
    test_races = [r for r in all_races if r["race_id"][:4] == "2025"]
    print(f"\nval=2024 {len(val_races):,} R / test=2025 {len(test_races):,} R")

    build_race_predictions(all_races)

    result = grid_search(val_races, test_races, payout)

    # 統合 OOS 評価 (playbook 適用)
    print("\n" + "=" * 70)
    print("playbook 適用時の OOS 統合 ROI 計測...")
    total_bet    = 0
    total_return = 0
    hits         = 0
    n_played     = 0
    for r in test_races:
        cell = "_".join(cell_key(r))
        p = result["cells"].get(cell)
        if p is None or p["top_n"] == 0:
            continue
        top_n   = p["top_n"]
        ev_gate = p["ev_gate"]
        combos  = r.get("combos_all", [])
        if not combos:
            continue
        selected = combos[:top_n]
        if not selected:
            continue
        expected = 0.0
        for combo, prob in selected:
            pops = tuple(r["pop_map"].get(ub, 7) for ub in combo)
            pay = payout.lookup(r["place"], r["shiba_da"],
                                r["dist_bucket"], r["field_bucket"], pops)
            expected += prob * pay
        if expected / BUDGET < ev_gate:
            continue
        per_bet_100 = max(100, int(BUDGET // len(selected) // 100) * 100)
        total_bet += per_bet_100 * len(selected)
        n_played  += 1
        for combo, _ in selected:
            if combo == r["result_top3"]:
                total_return += per_bet_100 * r["santan_pay"] / 100
                hits += 1
                break

    oos_integrated_roi = (total_return / total_bet * 100) if total_bet > 0 else 0.0
    result["integrated_oos"] = {
        "roi":       round(oos_integrated_roi, 2),
        "n_played":  n_played,
        "hits":      hits,
        "total_bet": total_bet,
        "baseline_roi": result["baseline"]["test_roi"],
        "delta":     round(oos_integrated_roi - result["baseline"]["test_roi"], 2),
    }
    bl = result["baseline"]["test_roi"]
    print(f"\n統合 OOS ROI: {oos_integrated_roi:.2f}%  "
          f"({hits}/{n_played}R  総額¥{total_bet:,})")
    print(f"vs ベースライン ({bl:.2f}%):  delta = {oos_integrated_roi - bl:+.2f}pt")

    # 保存
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n保存: {OUT_JSON}")


if __name__ == "__main__":
    main()
