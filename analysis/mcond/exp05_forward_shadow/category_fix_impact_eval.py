# -*- coding: utf-8 -*-
"""
category_fix_impact_eval.py — カテゴリ正規化の予測変化・成績指標への影響評価
========================================================================================
spec (EXP05-F 最終確認) §1-2。完了済みレース(data/kekka/に結果がある週)について、
旧処理(正規化なし)と新処理(category_normalize適用)でv6生スコアを再計算し、
  1) ai_score変化量・順位変更・◎変更・top3構成変更 (芝/ダート/内外/暫定表記別)
  2) 実際の結果(win_flag/top3_flag)に対する ◎的中率・top3的中率・NDCG@3・logloss・Brier
の新旧差を比較する。「完全な過去週次replay」ではなく、現在保持している
data/weekly/*.csv (2026年分のみ) の完了済み週によるcontrolled replayである制約を明記する。

実行: python -m analysis.mcond.exp05_forward_shadow.category_fix_impact_eval
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
from predict_weekly import parse_csv  # noqa: E402
from backtest_pl_ev import apply_encoders  # noqa: E402
from category_normalize import normalize_categorical  # noqa: E402
from analysis.mcond.exp05_forward_shadow.live_history import (  # noqa: E402
    _kekka_finish_map, ensure_date_column)

HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / "category_fix_impact.json"

_SERVE_RENAME = {
    "R": "Ｒ", "前走補正": "prev_hosei", "前走補9": "prev_hosei9",
    "trn_hanro_4f": "trnH_Time1", "trn_hanro_3f": "trnH_Time2",
    "trn_hanro_2f": "trnH_Time3", "trn_hanro_1f": "trnH_Time4",
    "trn_hanro_lap1": "trnH_Lap1", "trn_hanro_lap2": "trnH_Lap2",
    "trn_hanro_lap3": "trnH_Lap3", "trn_hanro_lap4": "trnH_Lap4",
    "trn_hanro_days": "trnH_days_ago",
    "trn_wc_5f": "trnW_5F", "trn_wc_4f": "trnW_4F", "trn_wc_3f": "trnW_3F",
    "trn_wc_lap1": "trnW_Lap1", "trn_wc_lap2": "trnW_Lap2",
    "trn_wc_lap3": "trnW_Lap3", "trn_wc_days": "trnW_days_ago",
}


def completed_weeks(n: int = 12) -> list[str]:
    kekka_dates = sorted(p.stem for p in (BASE / "data/kekka").glob("2026*.csv"))
    weekly_dates = set(p.stem for p in (BASE / "data/weekly").glob("2026*.csv"))
    dates = [d for d in kekka_dates if d in weekly_dates]
    if len(dates) <= n:
        return dates
    step = len(dates) / n
    return [dates[int(i * step)] for i in range(n)]


def _fill_missing_feats(df: pd.DataFrame, feats: list[str], encs: dict) -> pd.DataFrame:
    """export_weekly_marks.py の「feats に含まれるが週次CSVにない列を補完」と同じ処理。"""
    df = df.copy()
    cat_cols = set(encs.keys())
    for c in feats:
        if c not in df.columns:
            df[c] = "__NaN__" if c in cat_cols else np.nan
    return df


def score(df: pd.DataFrame, model, feats: list[str], encs: dict) -> np.ndarray:
    df = _fill_missing_feats(df, feats, encs)
    te = apply_encoders(df, encs)
    X = te[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    return model.predict(X)


def ndcg_at_k(order_by_score_desc: np.ndarray, fin: np.ndarray, k: int = 3) -> float:
    """理想順位=着順昇順。relevance=1/着順 (簡易)。"""
    rel = 1.0 / np.clip(fin, 1, None)
    dcg = sum(rel[order_by_score_desc[i]] / np.log2(i + 2) for i in range(min(k, len(order_by_score_desc))))
    ideal_order = np.argsort(-rel)
    idcg = sum(rel[ideal_order[i]] / np.log2(i + 2) for i in range(min(k, len(ideal_order))))
    return dcg / idcg if idcg > 0 else np.nan


def main() -> None:
    bundle = joblib.load(BASE / "models/unified_rank_v6.pkl")
    model, feats, encs = bundle["model"], bundle["feature_cols"], bundle["encoders"]

    weeks = completed_weeks()
    print(f"controlled replay対象週 ({len(weeks)}, 完了済みのみ): {weeks}")
    print("制約: data/weekly/には2026年分の週次CSVしか保持されていないため、"
          "2023-2025を含む完全な過去replayではない (SERVE_PARITY.mdと同じデータ制約)。")

    all_rows = []
    race_level_rows = []
    for wk in weeks:
        wpath = BASE / "data/weekly" / f"{wk}.csv"
        try:
            df = parse_csv(wpath)
        except Exception as e:
            print(f"  {wk}: parse失敗 {e}")
            continue
        rename = {k: v for k, v in _SERVE_RENAME.items() if k in df.columns and v not in df.columns}
        df = df.rename(columns=rename)
        finmap = _kekka_finish_map(wk)
        if not finmap:
            continue

        df_old = df.copy()
        df_new = normalize_categorical(df.copy())
        # 正規化で実際に変わった行だけ「変化フラグ」を残す (今回の4→8列修正で触れた列)
        touched_cols = ["芝・ダ", "前芝・ダ", "芝(内・外)", "馬場状態", "前走馬場状態", "天気",
                        "前走競走種別", "重量種別"]
        changed_mask = pd.Series(False, index=df.index)
        for c in touched_cols:
            if c in df.columns:
                changed_mask |= (df_old[c].astype(str) != df_new[c].astype(str))

        s_old = score(df_old, model, feats, encs)
        s_new = score(df_new, model, feats, encs)

        rid_col = "レースID(新/馬番無)" if "レースID(新/馬番無)" in df.columns else "レースID(新)"
        rid16 = df[rid_col].astype(str).str[:16]
        ban = pd.to_numeric(df["馬番"], errors="coerce")
        surf = df.get("芝・ダ", pd.Series("", index=df.index)).astype(str)
        naigai_touched = df.get("芝(内・外)", pd.Series("", index=df.index)).astype(str) != ""
        teiban_touched = pd.Series(False, index=df.index)
        for c in ["馬場状態", "天気", "前走馬場状態"]:
            if c in df.columns:
                teiban_touched |= df[c].astype(str).str.contains(r"\(暫定\)", na=False)

        for i in df.index:
            b = ban.loc[i]
            if pd.isna(b):
                continue
            fin = finmap.get((rid16.loc[i], int(b)))
            all_rows.append({
                "week": wk, "rid16": rid16.loc[i], "ban": int(b),
                "score_old": float(s_old[df.index.get_loc(i)]), "score_new": float(s_new[df.index.get_loc(i)]),
                "changed": bool(changed_mask.loc[i]), "surface_raw": surf.loc[i],
                "naigai_touched": bool(naigai_touched.loc[i]), "teiban_touched": bool(teiban_touched.loc[i]),
                "fin": fin, "win": (fin == 1) if fin is not None else None,
                "top3": (fin is not None and 1 <= fin <= 3),
            })

        for rid, g in df.groupby(rid_col):
            r16 = str(rid)[:16]
            idx = g.index
            gi = [df.index.get_loc(i) for i in idx]
            so, sn = s_old[gi], s_new[gi]
            fins = np.array([finmap.get((r16, int(ban.loc[i]))) for i in idx], dtype=float)
            if np.isnan(fins).all():
                continue
            top1_old = idx[int(np.argmax(so))]
            top1_new = idx[int(np.argmax(sn))]
            order_old = np.argsort(-so)
            order_new = np.argsort(-sn)
            top3_old = set(idx[order_old[:3]])
            top3_new = set(idx[order_new[:3]])
            race_level_rows.append({
                "week": wk, "rid16": r16,
                "top1_changed": top1_old != top1_new,
                "top3_set_changed": top3_old != top3_new,
                "any_score_changed": bool(changed_mask.loc[idx].any()),
                "top1_old_fin": finmap.get((r16, int(ban.loc[top1_old]))),
                "top1_new_fin": finmap.get((r16, int(ban.loc[top1_new]))),
                "ndcg3_old": ndcg_at_k(order_old, np.nan_to_num(fins, nan=99), 3) if not np.isnan(fins).all() else np.nan,
                "ndcg3_new": ndcg_at_k(order_new, np.nan_to_num(fins, nan=99), 3) if not np.isnan(fins).all() else np.nan,
            })
        print(f"  {wk}: {len(df)}頭 {df[rid_col].nunique()}レース処理済み")

    horse_df = pd.DataFrame(all_rows)
    race_df = pd.DataFrame(race_level_rows)
    horse_df.to_csv(HERE / "out" / "category_fix_impact_horse_level.csv", index=False, encoding="utf-8-sig")
    race_df.to_csv(HERE / "out" / "category_fix_impact_race_level.csv", index=False, encoding="utf-8-sig")

    horse_df["delta"] = horse_df["score_new"] - horse_df["score_old"]
    changed = horse_df[horse_df["changed"]]
    unchanged = horse_df[~horse_df["changed"]]

    def marks_metrics(sub, tag):
        sub = sub.dropna(subset=["fin"])
        if len(sub) == 0:
            return {}
        return {
            f"{tag}_n": int(len(sub)),
            f"{tag}_win_rate": float(sub["win"].mean()),
            f"{tag}_top3_rate": float(sub["top3"].mean()),
        }

    race_df_c = race_df.dropna(subset=["top1_old_fin", "top1_new_fin"])
    summary = {
        "weeks": weeks,
        "constraint": "data/weekly/は2026年分のみ保持。2023-2025を含む完全replayではないcontrolled replay",
        "n_horse_rows": int(len(horse_df)), "n_races": int(len(race_df)),
        "n_rows_category_changed": int(changed.shape[0]),
        "score_delta_abs_mean_changed_rows": float(changed["delta"].abs().mean()) if len(changed) else None,
        "score_delta_abs_mean_unchanged_rows": float(unchanged["delta"].abs().mean()) if len(unchanged) else None,
        "n_races_top1_changed": int(race_df["top1_changed"].sum()),
        "n_races_top3set_changed": int(race_df["top3_set_changed"].sum()),
        "n_races_any_category_touched": int(race_df["any_score_changed"].sum()),
        "top1_win_rate_old": float((race_df_c["top1_old_fin"] == 1).mean()),
        "top1_win_rate_new": float((race_df_c["top1_new_fin"] == 1).mean()),
        "top1_top3_rate_old": float((race_df_c["top1_old_fin"] <= 3).mean()),
        "top1_top3_rate_new": float((race_df_c["top1_new_fin"] <= 3).mean()),
        "ndcg3_mean_old": float(race_df["ndcg3_old"].mean()),
        "ndcg3_mean_new": float(race_df["ndcg3_new"].mean()),
        "by_surface": {},
    }
    for surf_val, tag in [("ダート", "dirt"), ("芝", "turf")]:
        sub_c = horse_df[(horse_df.surface_raw == surf_val) & horse_df.changed]
        sub_u = horse_df[(horse_df.surface_raw == surf_val) & ~horse_df.changed]
        summary["by_surface"][tag] = {
            "n_changed": int(len(sub_c)), "n_unchanged": int(len(sub_u)),
            **marks_metrics(sub_c, "changed"), **marks_metrics(sub_u, "unchanged"),
        }
    summary["naigai_touched_rows"] = int(horse_df["naigai_touched"].sum())
    summary["teiban_touched_rows"] = int(horse_df["teiban_touched"].sum())

    HERE.joinpath("out").mkdir(exist_ok=True)
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n保存: {OUT.relative_to(BASE)}")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
