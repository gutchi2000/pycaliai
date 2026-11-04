"""
precompute_buylist.py
週末出走表 CSV から買い目 + EV を一括計算し、parquet に書き出す。

Stage 1 中核ツール。Streamlit 側は parquet を読むだけにすることで、
画面切替の待ち時間を実質ゼロにする。

依存:
  - predict_weekly.py の parse_csv / ensemble_predict / assign_marks / get_bets
  - ev_gate.py の compute_ev_*  + pass_ev_gate
  - data/payout_table.parquet  (gen_payout_table.py で生成)

使い方:
    python precompute_buylist.py --csv data/weekly/20260419.csv
    python precompute_buylist.py --csv data/weekly/20260419.csv --budget 10000

出力:
    reports/buylist_horses_YYYYMMDD.parquet  (馬単位)
    reports/buylist_bets_YYYYMMDD.parquet    (買い目 × EV 単位)
"""
from __future__ import annotations
import argparse
import io
import json
import logging
import sys
import time
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# UTF-8 出力（Windows）
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# 既存パイプライン再利用
from predict_weekly import (  # noqa: E402
    BASE_DIR, LGBM_PATH, CAT_PATH, STRATEGY_JSON,
    parse_csv, ensemble_predict, assign_marks, get_bets,
    compute_value_scores, _predict_order_proba,
    EXCLUDE_PLACES, EXCLUDE_CLASSES, CLASS_NORMALIZE,
)

# EV ゲート
from ev_gate import (  # noqa: E402
    make_race_meta,
    compute_ev_tansho, compute_ev_fuku,
    compute_ev_umaren, compute_ev_umatan,
    compute_ev_sanren, compute_ev_santan,
    pass_ev_gate, _pop_bucket,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPORTS_DIR = BASE_DIR / "reports"


# ============================================================
# 買い目文字列 → tuple へパース
# ============================================================
def _parse_uma_combo(s: str) -> list[tuple[int, ...]]:
    """ "1-3 / 1-5" → [(1,3), (1,5)]  /  "1-3-7" → [(1,3,7)] """
    out = []
    if not s:
        return out
    for part in str(s).split("/"):
        part = part.strip()
        if not part:
            continue
        # 矢印 (→) は順序付き、ハイフン (-) は順序なし
        for sep in ("→", "-"):
            if sep in part:
                try:
                    nums = tuple(int(x) for x in part.split(sep))
                    out.append(nums)
                except ValueError:
                    pass
                break
        else:
            try:
                out.append((int(part),))
            except ValueError:
                pass
    return out


def _parse_single_horses(s: str) -> list[int]:
    """単勝・複勝の買い目「1」「1, 3」「1/3」を [1] や [1,3] にパース。"""
    if not s:
        return []
    s = str(s).replace(",", " ").replace("/", " ").replace("-", " ")
    out = []
    for tok in s.split():
        try:
            out.append(int(tok))
        except ValueError:
            pass
    return out


# ============================================================
# レース文脈 → EV 計算
# ============================================================
def _expand_bets_to_lines(
    race_meta_db: dict,
    place: str, td: str, dist: int, field_size: int,
    bets: dict,
    pop_map: dict[int, int],
    p_win_map: dict[int, float],
    p_fuku_map: dict[int, float],
    p_place23_map: dict[int, float],
) -> list[dict]:
    """1 レースの bets dict を 1 行 = 1 買い目に展開し、EV を付与。

    Args:
        pop_map: 馬番 → 人気
        p_win_map, p_fuku_map, p_place23_map: 馬番 → 確率
    """
    meta = make_race_meta(place, td, dist, field_size)
    lines: list[dict] = []

    base_row = {
        "場所": place, "td": td, "距離": dist, "頭数": field_size,
        "dist_b": meta["dist_b"], "field_b": meta["field_b"],
    }

    def _add(plan: str, bet_type: str, combo_str: str,
             amount: int, ev: float, payout: float, p_hit: float, n_combos: int):
        lines.append({**base_row,
            "plan": plan, "bet_type": bet_type,
            "combo": combo_str, "n_combos": n_combos,
            "amount": int(amount), "p_hit": round(float(p_hit), 5),
            "payout_est": round(float(payout), 1),
            "ev": round(float(ev), 4),
            "gate_pass": bool(pass_ev_gate(bet_type, ev)),
        })

    # ── HALO 三連単 ────────────────────────────────────
    if bets.get("HALO_戦略対象"):
        combos = _parse_uma_combo(bets.get("HALO_三連単_買い目", ""))
        n = len(combos)
        per_bet = (bets.get("HALO_三連単_購入額", 0) // n) if n > 0 else 0
        for combo in combos:
            if len(combo) != 3:
                continue
            f, s, t = combo
            p1 = p_win_map.get(f, 0.0)
            # 三連単 p_hit は単純化として 1 着 prob × 残り 2 着の条件 (粗い)
            # 厳密には Plackett-Luce 必要だが Stage 1 は p_win × p_place_others / N で近似
            # （Stage 2 で trifecta_model_v1 に置き換え）
            p_2nd = p_place23_map.get(s, 0.0)
            p_3rd = p_place23_map.get(t, 0.0)
            # 2-3 着確率は「3 着以内に来る」 ÷ 2 で近似（重複排除）
            p_hit_est = p1 * (p_2nd / 2.0) * (p_3rd / 2.0)
            ev, pay = compute_ev_santan(p_hit_est,
                pop_map.get(f, 7), pop_map.get(s, 7), pop_map.get(t, 7), meta)
            _add("HALO", "三連単", f"{f}→{s}→{t}", per_bet, ev, pay, p_hit_est, n)

    # ── HAHO 三連複 ────────────────────────────────────
    if bets.get("HAHO_戦略対象"):
        combos = _parse_uma_combo(bets.get("HAHO_三連複_買い目", ""))
        n = len(combos)
        per_bet = (bets.get("HAHO_三連複_購入額", 0) // n) if n > 0 else 0
        for combo in combos:
            if len(combo) != 3:
                continue
            a, b, c = combo
            # 三連複 p_hit ≈ Σ permutations(p_1着 × p_2着 × p_3着) / 6
            # Stage 1 簡易: 3 馬とも「3 着以内」確率の積 / 2
            p_a = p_win_map.get(a, 0.0) + p_place23_map.get(a, 0.0)
            p_b = p_win_map.get(b, 0.0) + p_place23_map.get(b, 0.0)
            p_c = p_win_map.get(c, 0.0) + p_place23_map.get(c, 0.0)
            # 3 馬とも 3 着以内（独立近似）
            p_hit_est = p_a * p_b * p_c / 6.0
            ev, pay = compute_ev_sanren(p_hit_est,
                pop_map.get(a, 7), pop_map.get(b, 7), pop_map.get(c, 7), meta)
            _add("HAHO", "三連複", f"{a}-{b}-{c}", per_bet, ev, pay, p_hit_est, n)

    # ── STANDARD 単勝 ──────────────────────────────────
    # 注意: predict_weekly の STANDARD 改修以降、買い目は「1 / 9」のように
    # 複数馬を併記し、購入額はその合計が入っている。
    # 1 行 = 1 馬 として展開するため、合計を候補数で割る。
    if bets.get("STANDARD_戦略対象"):
        tan_horses = _parse_single_horses(bets.get("STANDARD_単勝_買い目", ""))
        tan_total  = bets.get("STANDARD_単勝_購入額", 0)
        tan_each   = (tan_total // len(tan_horses)) if tan_horses else 0
        for h in tan_horses:
            p = p_win_map.get(h, 0.0)
            ev, pay = compute_ev_tansho(p, pop_map.get(h, 7), meta)
            _add("STANDARD", "単勝", str(h), tan_each, ev, pay, p, 1)

        # ── STANDARD 複勝 ──────────────────────────────
        fuku_horses = _parse_single_horses(bets.get("STANDARD_複勝_買い目", ""))
        fuku_total  = bets.get("STANDARD_複勝_購入額", 0)
        fuku_each   = (fuku_total // len(fuku_horses)) if fuku_horses else 0
        for h in fuku_horses:
            p = p_fuku_map.get(h, p_win_map.get(h, 0.0) + p_place23_map.get(h, 0.0))
            ev, pay = compute_ev_fuku(p, pop_map.get(h, 7), meta)
            _add("STANDARD", "複勝", str(h), fuku_each, ev, pay, p, 1)

        # ── STANDARD 馬連 ──────────────────────────────
        uma_combos = [c for c in _parse_uma_combo(bets.get("STANDARD_馬連_買い目", ""))
                      if len(c) == 2]
        uma_total  = bets.get("STANDARD_馬連_購入額", 0)
        uma_each   = (uma_total // len(uma_combos)) if uma_combos else 0
        for combo in uma_combos:
            a, b = combo
            # 馬連 p_hit ≈ 「両者 3 着以内」÷ 3（粗い近似）
            p_a = p_win_map.get(a, 0.0) + p_place23_map.get(a, 0.0)
            p_b = p_win_map.get(b, 0.0) + p_place23_map.get(b, 0.0)
            p_hit_est = p_a * p_b / 3.0
            ev, pay = compute_ev_umaren(p_hit_est,
                pop_map.get(a, 7), pop_map.get(b, 7), meta)
            _add("STANDARD", "馬連", f"{a}-{b}", uma_each, ev, pay, p_hit_est, 1)

    # ── TRIPLE 三連複 + 複勝 ────────────────────────────
    if bets.get("TRIPLE_戦略対象"):
        for combo in _parse_uma_combo(bets.get("TRIPLE_三連複_買い目", "")):
            if len(combo) != 3:
                continue
            a, b, c = combo
            p_a = p_win_map.get(a, 0.0) + p_place23_map.get(a, 0.0)
            p_b = p_win_map.get(b, 0.0) + p_place23_map.get(b, 0.0)
            p_c = p_win_map.get(c, 0.0) + p_place23_map.get(c, 0.0)
            p_hit_est = p_a * p_b * p_c / 6.0
            ev, pay = compute_ev_sanren(p_hit_est,
                pop_map.get(a, 7), pop_map.get(b, 7), pop_map.get(c, 7), meta)
            _add("TRIPLE", "三連複", f"{a}-{b}-{c}",
                 bets.get("TRIPLE_三連複_購入額", 0), ev, pay, p_hit_est, 1)

        for h in _parse_single_horses(bets.get("TRIPLE_複勝_買い目", "")):
            p = p_fuku_map.get(h, p_win_map.get(h, 0.0) + p_place23_map.get(h, 0.0))
            ev, pay = compute_ev_fuku(p, pop_map.get(h, 7), meta)
            _add("TRIPLE", "複勝", str(h),
                 bets.get("TRIPLE_複勝_購入額", 0), ev, pay, p, 1)

    return lines


# ============================================================
# メイン
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="週末買い目を precompute → parquet 出力")
    parser.add_argument("--csv", required=True, help="入力 CSV パス (例: data/weekly/20260419.csv)")
    parser.add_argument("--budget", type=int, default=10000, help="1R 予算 (default: 10000)")
    parser.add_argument("--out-dir", default="", help="出力ディレクトリ (default: reports/)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        logger.error(f"CSV が見つかりません: {csv_path}")
        sys.exit(1)

    out_dir = Path(args.out_dir) if args.out_dir else REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    date_str = csv_path.stem  # "20260419"

    # ── モデル & 戦略 読み込み ──────────────────────────
    t0 = time.time()
    logger.info("モデル読み込み中...")
    lgbm_obj = joblib.load(LGBM_PATH)
    cat_obj  = joblib.load(CAT_PATH)
    with open(STRATEGY_JSON, encoding="utf-8") as f:
        strategy = json.load(f)

    # ── CSV パース ─────────────────────────────────────
    logger.info(f"CSV パース: {csv_path}")
    df = parse_csv(csv_path)
    race_id_col = "レースID(新/馬番無)"
    n_races = df[race_id_col].nunique()
    logger.info(f"  → {n_races} R / {len(df)} 頭")

    horses_rows: list[dict] = []
    bets_rows:   list[dict] = []

    # ── レースごとに予測 + 買い目 + EV ─────────────────
    for race_id, race_df in df.groupby(race_id_col):
        race_df = race_df.copy().reset_index(drop=True)
        meta    = race_df.iloc[0]
        place   = str(meta.get("場所", ""))
        cls_raw = str(meta.get("クラス名", ""))
        td      = str(meta.get("芝・ダ", meta.get("芝・ダート", "")))
        try:
            dist = int(float(meta.get("距離", 0)))
        except Exception:
            dist = 0
        date    = str(meta.get("日付S", ""))
        r_num   = str(meta.get("R", ""))
        hassou  = str(meta.get("発走時刻", ""))
        race_name = str(meta.get("レース名", ""))
        field_size = len(race_df)

        # 予測
        try:
            race_df["prob"]  = ensemble_predict(race_df, lgbm_obj, cat_obj)
            race_df          = assign_marks(race_df)
            race_df["score"] = (race_df["prob"] * 100).round(1)
        except Exception as e:
            logger.warning(f"予測失敗 {race_id}: {e}")
            continue

        # 人気（単勝オッズ昇順）
        tansho = pd.to_numeric(race_df.get("単勝", pd.Series(dtype=float)), errors="coerce")
        race_df["popularity"] = tansho.rank(method="min", ascending=True).astype("Int64")

        # order_model から p_win, p_place23 を取得（無ければ prob を流用）
        order_proba = _predict_order_proba(race_df)
        if order_proba is not None:
            race_df["p_win"]      = order_proba["p_win"].values
            race_df["p_place23"]  = order_proba["p_place23"].values
            race_df["p_fuku"]     = race_df["p_win"] + race_df["p_place23"]
        else:
            # フォールバック: prob を p_win とみなし、複勝率は約 2.5x
            race_df["p_win"]     = race_df["prob"]
            race_df["p_place23"] = race_df["prob"] * 1.5
            race_df["p_fuku"]    = race_df["prob"] * 2.5

        # Value Model
        try:
            race_df["value_score"], race_df["cal_prob"] = compute_value_scores(race_df, od_odds=None)
        except Exception:
            race_df["value_score"] = np.nan
            race_df["cal_prob"]    = np.nan

        # 買い目生成（既存の get_bets を使用）
        if place in EXCLUDE_PLACES or cls_raw in EXCLUDE_CLASSES:
            bets = get_bets(race_df, "", "", {}, 0)
        else:
            bets = get_bets(race_df, place, cls_raw, strategy, args.budget)

        # ── 馬単位 行 ──────────────────────────────────
        for _, row in race_df.sort_values("馬番").iterrows():
            horses_rows.append({
                "日付":   date,
                "場所":   place,
                "R":      r_num,
                "クラス": cls_raw,
                "td":     td,
                "距離":   dist,
                "頭数":   field_size,
                "発走時刻": hassou,
                "レース名": race_name,
                "race_id": race_id,
                "馬番": int(row["馬番"]) if pd.notna(row["馬番"]) else None,
                "馬名": str(row.get("馬名", row.get("馬名S", ""))),
                "騎手": str(row.get("騎手", "")),
                "印":   str(row.get("mark", "")),
                "prob": float(row.get("prob", 0.0)),
                "score": float(row.get("score", 0.0)),
                "p_win":     float(row.get("p_win", 0.0)),
                "p_place23": float(row.get("p_place23", 0.0)),
                "p_fuku":    float(row.get("p_fuku", 0.0)),
                "popularity": int(row["popularity"]) if pd.notna(row["popularity"]) else None,
                "単勝オッズ": float(tansho.iloc[row.name]) if pd.notna(tansho.iloc[row.name]) else None,
                "value_score": float(row["value_score"]) if pd.notna(row.get("value_score")) else None,
                "cal_prob":   float(row["cal_prob"])    if pd.notna(row.get("cal_prob"))    else None,
            })

        # ── 買い目単位 行 ─────────────────────────────
        pop_map      = {int(r["馬番"]): int(r["popularity"]) for _, r in race_df.iterrows()
                        if pd.notna(r["馬番"]) and pd.notna(r["popularity"])}
        p_win_map    = {int(r["馬番"]): float(r["p_win"])     for _, r in race_df.iterrows() if pd.notna(r["馬番"])}
        p_fuku_map   = {int(r["馬番"]): float(r["p_fuku"])    for _, r in race_df.iterrows() if pd.notna(r["馬番"])}
        p_place_map  = {int(r["馬番"]): float(r["p_place23"]) for _, r in race_df.iterrows() if pd.notna(r["馬番"])}

        lines = _expand_bets_to_lines(
            race_meta_db=None,
            place=place, td=td, dist=dist, field_size=field_size,
            bets=bets,
            pop_map=pop_map, p_win_map=p_win_map,
            p_fuku_map=p_fuku_map, p_place23_map=p_place_map,
        )
        for ln in lines:
            ln.update({
                "日付": date, "R": r_num, "クラス": cls_raw,
                "発走時刻": hassou, "レース名": race_name, "race_id": race_id,
            })
            bets_rows.append(ln)

    # ── DataFrame & 出力 ────────────────────────────────
    horses_df = pd.DataFrame(horses_rows)
    bets_df   = pd.DataFrame(bets_rows)

    horses_path = out_dir / f"buylist_horses_{date_str}.parquet"
    bets_path   = out_dir / f"buylist_bets_{date_str}.parquet"
    horses_df.to_parquet(horses_path, index=False)
    bets_df.to_parquet(bets_path, index=False)

    # ── サマリ ──────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"precompute 完了 ({elapsed:.1f} 秒)")
    print(f"  対象: {n_races} R / {len(horses_df)} 頭")
    print(f"  買い目行数: {len(bets_df)}")
    if not bets_df.empty:
        for plan in ["HAHO", "HALO", "STANDARD", "TRIPLE"]:
            sub = bets_df[bets_df["plan"] == plan]
            if sub.empty:
                continue
            n_lines = len(sub)
            n_pass  = int(sub["gate_pass"].sum())
            mean_ev = sub["ev"].mean()
            sum_amt = int(sub["amount"].sum())
            print(f"    {plan:10s}: {n_lines:4d} 行 (gate通過 {n_pass:3d}, "
                  f"平均EV {mean_ev:.3f}, 投資額 ¥{sum_amt:,})")
    print(f"  出力 (馬): {horses_path}  ({horses_path.stat().st_size/1024:.1f} KB)")
    print(f"  出力 (買): {bets_path}    ({bets_path.stat().st_size/1024:.1f} KB)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
