"""
export_marks_json.py
====================
Cowork 連携用: PyCaLiAI モデルから「印 + PL 確率 + メタ情報」を JSON 出力する。

JSON スキーマ:
{
  "race_id": "2025122806050811",
  "race_meta": {
    "date": "2025-12-28",
    "place": "中山",
    "course": "芝1600",
    "field_size": 16,
    "class": "G1"
  },
  "horses": [
    {
      "umaban": 4,
      "horse_name": "...",
      "mark": "◎",
      "ai_rank": 1,
      "ai_score": 0.847,
      "p_win": 0.32,
      "p_plc": 0.55,
      "p_sho": 0.71,
      "tansho_odds": 2.4,
      "fuku_odds_low": 1.3,
      "fuku_odds_high": 1.7,
      "ai_vs_market": "fair"
    },
    ...
  ],
  "race_confidence": {
    "top1_dominance": 0.42,
    "top2_concentration": 0.65,
    "field_chaos_score": 0.31,
    "ai_market_agreement": 0.88
  }
}

使い方:
  # 単一レース
  python export_marks_json.py --model v5 --race-id 2025122806050811

  # 年単位で全 race を出力 (1 race / 1 file)
  python export_marks_json.py --model v5 --year 2025 --out-dir reports/marks_v5/

  # 期間指定
  python export_marks_json.py --model v5 --year-from 2024 --year-to 2025 \
      --out-dir reports/marks_v5/
"""
from __future__ import annotations
import argparse
import io
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import pl_probs as PL
from backtest_pl_ev import (
    all_umaren_mat, all_fukusho_vec_fast,
    COL_RID, COL_BAN,
)
import backtest_pl_ev as be
import betting_judgment as bj

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = Path(__file__).parent
MARKS = ["◎", "〇", "▲", "△", "△"]  # 5 つ. 4位と5位は同じ△ (cowork 側で順位で識別)


def load_odds_for_year(year):
    """年単位でオッズ CSV をロードして {(rid_s, ban): tansho_odds} を返す."""
    candidates = [
        BASE / f"data/odds/odds_{year}0105-{year}1228.csv",
        BASE / f"data/odds/odds_{year}0106-{year}1228.csv",
        BASE / f"data/odds/odds_{year}0105-{year+1}1228.csv",
    ]
    odds_files = [f for f in candidates if f.exists()]
    if not odds_files:
        return {}, {}
    odds_df = pd.concat([pd.read_csv(f, encoding="cp932") for f in odds_files],
                        ignore_index=True)
    odds_df = odds_df.rename(columns={
        "レースID(新)": "rid", "馬番": "ban",
        "単勝オッズ": "tansho_odds",
        "複勝オッズ下限": "fuku_low", "複勝オッズ上限": "fuku_high",
    })
    odds_df["rid_s"] = odds_df["rid"].astype(str).str[:16]
    odds_df["ban"] = pd.to_numeric(odds_df["ban"], errors="coerce").astype("Int64")
    odds_df["tansho_odds"] = pd.to_numeric(odds_df["tansho_odds"], errors="coerce")
    odds_df["fuku_low"] = pd.to_numeric(odds_df["fuku_low"], errors="coerce")
    odds_df["fuku_high"] = pd.to_numeric(odds_df["fuku_high"], errors="coerce")
    tansho_idx = odds_df.set_index(["rid_s", "ban"])["tansho_odds"].to_dict()
    fuku_idx = {}
    for _, r in odds_df.iterrows():
        try:
            fuku_idx[(r["rid_s"], int(r["ban"]))] = (
                float(r["fuku_low"]) if not pd.isna(r["fuku_low"]) else None,
                float(r["fuku_high"]) if not pd.isna(r["fuku_high"]) else None,
            )
        except Exception:
            pass
    return tansho_idx, fuku_idx


def race_confidence(p_win_vec, p_plc_vec, ai_rank_order, market_rank_by_ban=None):
    """レース信頼度の各種指標を計算."""
    p_sorted = np.sort(p_win_vec)[::-1]
    top1 = float(p_sorted[0]) if len(p_sorted) >= 1 else 0.0
    top2 = float(p_sorted[1]) if len(p_sorted) >= 2 else 0.0
    top1_dom = max(0.0, min(1.0, top1 - top2))  # ◎-〇 確率差 (clamped)
    top2_conc = max(0.0, min(1.0, top1 + top2))  # 上位 2 馬の確率合計 (calibrator 後に >1 になりうるため clamp)
    # field_chaos: エントロピーベース (高 = カオス、低 = 固い)
    # calibrator 適用後は Σ p_win ≠ 1.0 になり得るので、正規化してから entropy 計算
    p = p_win_vec[p_win_vec > 1e-9]
    if len(p) > 0 and p.sum() > 0:
        p_norm = p / p.sum()
        entropy = -np.sum(p_norm * np.log(p_norm))
        max_entropy = np.log(len(p_norm))
        chaos = float(entropy / max_entropy) if max_entropy > 0 else 0.0
        chaos = max(0.0, min(1.0, chaos))  # numerical safety clamp [0, 1]
    else:
        chaos = 0.0
    # AI vs 市場: spearman 相関
    market_corr = None
    if market_rank_by_ban is not None and len(market_rank_by_ban) == len(ai_rank_order):
        ai_rank = pd.Series(ai_rank_order).rank()
        market_rank = pd.Series(market_rank_by_ban).rank()
        market_corr = float(ai_rank.corr(market_rank, method="spearman"))
    return {
        "top1_dominance": round(top1_dom, 4),
        "top2_concentration": round(top2_conc, 4),
        "field_chaos_score": round(chaos, 4),
        "ai_market_agreement": round(market_corr, 4) if market_corr is not None else None,
    }


def horse_record(g_row, ban, mark, ai_rank, ai_score, p_win, p_plc, p_sho,
                 tansho_odds, fuku_odds):
    """1 馬分のレコード."""
    horse_name = g_row.get("馬名", "")
    if pd.isna(horse_name):
        horse_name = ""
    # ai_vs_market: AI 確率 vs 単勝オッズベース市場確率
    ai_vs_market = "unknown"
    if tansho_odds and tansho_odds > 0 and p_win is not None:
        # 市場の implied probability (控除率を考慮しない単純化)
        market_p = 1.0 / tansho_odds
        if p_win >= market_p * 1.20:
            ai_vs_market = "under"  # 市場より AI が高評価 = 過小評価されてる馬
        elif p_win <= market_p * 0.80:
            ai_vs_market = "over"   # 市場より AI が低評価 = 過大評価されてる馬
        else:
            ai_vs_market = "fair"
    rec = {
        "umaban": int(ban),
        "horse_name": str(horse_name),
        "mark": mark,
        "ai_rank": ai_rank,
        "ai_score": round(float(ai_score), 4),
        "p_win": round(float(p_win), 4) if p_win is not None else None,
        "p_plc": round(float(p_plc), 4) if p_plc is not None else None,
        "p_sho": round(float(p_sho), 4) if p_sho is not None else None,
        "tansho_odds": round(float(tansho_odds), 2) if tansho_odds else None,
        "fuku_odds_low": round(fuku_odds[0], 2) if fuku_odds and fuku_odds[0] else None,
        "fuku_odds_high": round(fuku_odds[1], 2) if fuku_odds and fuku_odds[1] else None,
        "ai_vs_market": ai_vs_market,
    }
    return rec


def race_meta(g, class_prior_map: dict | None = None):
    """レースメタ情報. class_prior_map={class_name: prior_dict} を渡すと
    そのレースの class に対応する経験 prior (◎〇▲△△ の複勝圏率等) を埋め込む。
    """
    row = g.iloc[0]
    def _g(k, default=""):
        v = row.get(k, default)
        return "" if pd.isna(v) else str(v)
    cls = _g("クラス名")
    meta = {
        "date": _g("日付"),
        "place": _g("場所"),
        "course": f"{_g('芝・ダ')}{_g('距離')}",
        "field_size": int(len(g)),
        "class": cls,
        "race_name": _g("レース名"),
    }
    if class_prior_map and cls:
        prior = class_prior_map.get(cls)
        if prior:
            meta["class_prior"] = prior
    return meta


def export_race(rid, g_orig, model, feats, encs, tansho_idx, fuku_idx,
                calibrators=None, umaren_idx=None, class_prior_map=None,
                shap_explainer=None, shap_topk=0, shap_marked_only=True):
    """1 レース分の JSON を返す.

    umaren_idx: optional dict {(rid_s, i, j): float} (i<j 対称) — bundle に
                 'umaren_matrix': {"i-j": odds} を埋め込む。
    class_prior_map: optional dict {class_name: prior_dict} → race_meta に
                 class_prior フィールドを埋め込み (Cowork が prior 参照可能に)。
    shap_explainer: optional shap.TreeExplainer (marks_shap.build_explainer)。
                 渡されて shap_topk>0 なら各馬 (印馬のみ/全馬) に "why" (top-k 寄与) を付与。
    shap_topk: 付与する SHAP 寄与の数 (0 で無効)。
    shap_marked_only: True なら印が付いた馬 (◎〇▲△△) だけに "why" を付与。
    """
    rid_s = str(rid).split(".")[0]  # remove .0 if float
    if isinstance(rid, float) and rid.is_integer():
        rid_s = str(int(rid))

    # 推論用エンコード
    g = g_orig.copy().sort_values(COL_BAN).reset_index(drop=True)
    g_enc = g.copy()
    for c, le in encs.items():
        if c not in g_enc.columns: continue
        v = g_enc[c].astype(str).fillna("__NaN__")
        known = set(le.classes_)
        v = v.where(v.isin(known), "__NaN__")
        g_enc[c] = le.transform(v)

    X = g_enc[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    scores = model.predict(X)
    n = len(scores)
    w = PL.pl_weights(scores)
    p_win_vec = PL.all_tansho(w)
    p_plc_vec = np.zeros(n)  # 連対率: 1 or 2 着
    for i in range(n):
        # 連対率 = P(1着) + P(2着)
        # P(2着) = Σ_j≠i (w_j / total) * (w_i / (total - w_j))
        total = w.sum()
        p2 = sum(w[j] / total * w[i] / (total - w[j]) for j in range(n) if j != i)
        p_plc_vec[i] = float(p_win_vec[i] + p2)
    p_sho_vec = all_fukusho_vec_fast(w)  # 複勝率: top-3

    # キャリブレーター適用 (利用可能なら)
    if calibrators is not None:
        if "tansho" in calibrators:
            p_win_vec = calibrators["tansho"].predict(p_win_vec)
        if "fukusho" in calibrators:
            p_sho_vec = calibrators["fukusho"].predict(p_sho_vec)

    order = np.argsort(-scores)
    ai_rank_by_idx = np.zeros(n, dtype=int)
    for rank, idx in enumerate(order):
        ai_rank_by_idx[idx] = rank + 1  # 1-indexed
    mark_by_idx = ["" for _ in range(n)]
    for rank, idx in enumerate(order[:5]):
        mark_by_idx[idx] = MARKS[rank]

    # 市場ランク (オッズ昇順 = 人気順)
    bans = g[COL_BAN].astype(int).values
    market_odds_by_idx = np.array([tansho_idx.get((rid_s, int(b)), np.nan)
                                    for b in bans])
    market_rank_by_idx = pd.Series(market_odds_by_idx).rank().values
    ai_rank_order = ai_rank_by_idx.copy()

    # SHAP 寄与 (印の根拠)。explainer 未指定 or topk<=0 なら None のまま。
    contribs = None
    if shap_explainer is not None and shap_topk and shap_topk > 0:
        try:
            from marks_shap import race_contribs
            contribs = race_contribs(shap_explainer, X, feats, g, topk=shap_topk)
        except Exception:
            contribs = None  # SHAP 失敗時も bundle 生成は継続

    horses = []
    for i in range(n):
        ban = int(bans[i])
        tansho = tansho_idx.get((rid_s, ban))
        fuku = fuku_idx.get((rid_s, ban))
        rec = horse_record(
            g.iloc[i], ban,
            mark=mark_by_idx[i],
            ai_rank=int(ai_rank_by_idx[i]),
            ai_score=float(scores[i]),
            p_win=p_win_vec[i],
            p_plc=p_plc_vec[i],
            p_sho=p_sho_vec[i],
            tansho_odds=tansho if tansho and not pd.isna(tansho) else None,
            fuku_odds=fuku,
        )
        if contribs is not None and (not shap_marked_only or mark_by_idx[i] != ""):
            rec["why"] = contribs[i]
        horses.append(rec)

    # umaban 順にソート
    horses.sort(key=lambda r: r["umaban"])

    # race confidence
    valid_mask = ~np.isnan(market_odds_by_idx)
    if valid_mask.sum() >= 3:
        conf = race_confidence(p_win_vec, p_plc_vec, ai_rank_order,
                               market_rank_by_ban=market_rank_by_idx)
    else:
        conf = race_confidence(p_win_vec, p_plc_vec, ai_rank_order)

    payload = {
        "race_id": rid_s,
        "race_meta": race_meta(g, class_prior_map=class_prior_map),
        "horses": horses,
        "race_confidence": conf,
        # 買い方判定 (堅さ × 妙味馬の有無 → 買い方 + 妙味馬リスト)。
        # Cowork が方針/券種ヒント/妙味馬の判断材料に使う。閾値は betting_judgment.py。
        "buy_judgment": bj.build_judgment(conf, horses),
    }

    # 馬連オッズ matrix (OD CSV 由来)
    if umaren_idx:
        bans_set = sorted({int(b) for b in bans})
        matrix = {}
        for ii, a in enumerate(bans_set):
            for b in bans_set[ii + 1:]:
                v = umaren_idx.get((rid_s, a, b))
                if v is not None and v > 0:
                    matrix[f"{a}-{b}"] = round(float(v), 1)
        if matrix:
            payload["umaren_matrix"] = matrix

    # --- 印馬ペア C(5,2)=10 組の正確な PL joint 確率 (calibrated) ---
    # compute_bets のワイド EV が独立積近似 p_sho_i×p_sho_j で +21〜27% 系統過大
    # だった問題 (docs/audit_20260611.md 🔴) の解消用。PL の厳密 joint を
    # calibrator (wide/umaren/umatan) に通した値を埋め込み、compute_bets は
    # これを読むだけにする。キーは umaren_matrix と同じ "a-b" (a<b の馬番)。
    # umatan のみ方向があるので {"a→b": p, "b→a": p} のネスト。
    top5_idx = [int(ix) for ix in order[:5]]
    pair_probs = {}
    for ai in range(len(top5_idx)):
        for bi in range(ai + 1, len(top5_idx)):
            i, j = top5_idx[ai], top5_idx[bi]
            p_wide_ = float(PL.p_wide(w, i, j))
            p_umaren_ = float(PL.p_umaren(w, i, j))
            p_ut_ij = float(PL.p_umatan(w, i, j))
            p_ut_ji = float(PL.p_umatan(w, j, i))
            if calibrators is not None:
                if "wide" in calibrators:
                    p_wide_ = float(calibrators["wide"].predict([p_wide_])[0])
                if "umaren" in calibrators:
                    p_umaren_ = float(calibrators["umaren"].predict([p_umaren_])[0])
                if "umatan" in calibrators:
                    p_ut_ij = float(calibrators["umatan"].predict([p_ut_ij])[0])
                    p_ut_ji = float(calibrators["umatan"].predict([p_ut_ji])[0])
            ban_i, ban_j = int(bans[i]), int(bans[j])
            a, b = sorted((ban_i, ban_j))
            ut = {f"{ban_i}→{ban_j}": round(p_ut_ij, 5),
                  f"{ban_j}→{ban_i}": round(p_ut_ji, 5)}
            pair_probs[f"{a}-{b}"] = {
                "wide": round(p_wide_, 5),
                "umaren": round(p_umaren_, 5),
                "umatan": ut,
            }
    if pair_probs:
        payload["pair_probs"] = pair_probs

    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="v5", help="model tag: v4 or v5")
    ap.add_argument("--race-id", default=None, help="特定 race_id (1 件出力, stdout)")
    ap.add_argument("--year", type=int, default=None, help="年単位出力")
    ap.add_argument("--year-from", type=int, default=None)
    ap.add_argument("--year-to", type=int, default=None)
    ap.add_argument("--out-dir", default=None, help="バッチ出力先ディレクトリ")
    args = ap.parse_args()

    tag = args.model
    be.MODEL_PKL = BASE / f"models/unified_rank_{tag}.pkl"
    be.CAL_PKL   = BASE / f"models/pl_calibrators_{tag}.pkl"
    assert be.MODEL_PKL.exists(), f"{be.MODEL_PKL} not found"
    print(f"[model] tag={tag}  {be.MODEL_PKL}", file=sys.stderr)

    bundle = joblib.load(be.MODEL_PKL)
    model = bundle["model"]
    feats = bundle["feature_cols"]
    encs  = bundle["encoders"]

    calibrators = None
    if be.CAL_PKL.exists():
        cal_bundle = joblib.load(be.CAL_PKL)
        calibrators = cal_bundle.get("calibrators", cal_bundle)
        print(f"[cal] {be.CAL_PKL}", file=sys.stderr)

    # 期間決定
    if args.race_id:
        years = None  # race_id を直接指定
    elif args.year:
        years = {args.year}
    elif args.year_from and args.year_to:
        years = set(range(args.year_from, args.year_to + 1))
    else:
        print("ERROR: --race-id or --year or --year-from/--year-to を指定してください",
              file=sys.stderr)
        sys.exit(1)

    # データ読み込み
    print(f"[load] master_v2", file=sys.stderr)
    df = pd.read_csv(BASE / "data/master_v2_20130105-20251228.csv",
                     encoding="utf-8-sig", low_memory=False)
    df["着順"] = pd.to_numeric(df["着順"], errors="coerce")
    df["year"] = df[COL_RID].astype(str).str[:4].astype(int)

    if args.race_id:
        df_target = df[df[COL_RID].astype(str) == args.race_id]
        if len(df_target) == 0:
            df_target = df[df[COL_RID].astype(str).str.startswith(args.race_id[:16])]
        if len(df_target) == 0:
            print(f"ERROR: race_id={args.race_id} not found", file=sys.stderr)
            sys.exit(1)
        years_in_data = {df_target["year"].iloc[0]}
    else:
        df_target = df[df["year"].isin(years)]
        years_in_data = years

    print(f"[data] target rows={len(df_target):,}  races={df_target[COL_RID].nunique():,}",
          file=sys.stderr)

    # オッズ
    tansho_all = {}; fuku_all = {}
    for y in years_in_data:
        ti, fi = load_odds_for_year(y)
        tansho_all.update(ti); fuku_all.update(fi)
    print(f"[odds] {len(tansho_all):,} 馬の単勝オッズ取得", file=sys.stderr)

    # 出力
    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        n_done = 0
        for rid, g in df_target.groupby(COL_RID, sort=False):
            if len(g) < 5: continue
            try:
                payload = export_race(rid, g, model, feats, encs,
                                       tansho_all, fuku_all, calibrators)
            except Exception as e:
                print(f"  ERROR rid={rid}: {e}", file=sys.stderr)
                continue
            rid_s = payload["race_id"]
            with open(out_dir / f"{rid_s}.json", "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            n_done += 1
            if n_done % 500 == 0:
                print(f"  ..{n_done:,} races", file=sys.stderr)
        print(f"[done] {n_done:,} JSON files saved to {out_dir}", file=sys.stderr)
    else:
        # 単一 race を stdout に出す
        rid, g = next(iter(df_target.groupby(COL_RID, sort=False)))
        payload = export_race(rid, g, model, feats, encs,
                               tansho_all, fuku_all, calibrators)
        print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
