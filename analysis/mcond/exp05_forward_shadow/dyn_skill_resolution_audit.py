# -*- coding: utf-8 -*-
"""
dyn_skill_resolution_audit.py — 動的能力の馬identity解決率を正しく分解する (spec §6)
========================================================================================
serve_history_feats._HistoryIndex (本番の既存資産、data/_horse_history.parquet ベース、
名前+種牡馬+生年での曖昧回避つき解決) を再利用し、対象週の各馬を
  hit(=履歴あり・解決成功) / new(=履歴なし、正当な初出走) / ambiguous(=同名複数候補で未解決)
に分類する。live_history.py の単純な馬名一致 (name_to_hid_2025) がこの正しい分類と
どれだけ一致するかを突き合わせ、「履歴を持つ馬の解決成功率」(全馬一致率ではない)を報告する。
実行: python -m analysis.mcond.exp05_forward_shadow.dyn_skill_resolution_audit --date 20260919
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
from serve_history_feats import _load, _HistoryIndex, _clean_name  # noqa: E402
from analysis.mcond.exp05_forward_shadow import live_history as LH  # noqa: E402
from analysis.mcond.exp05_forward_shadow.feature_snapshot import load_and_prepare  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / "dyn_skill_resolution.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    args = ap.parse_args()
    date_str = args.date

    df = load_and_prepare(date_str)  # export_weekly_marks と同じ _SERVE_RENAME + fill_history_features
    idx, maps, meta = _load(BASE)
    print(f"_horse_history.parquet: {meta['n_rows']:,}行 max_date={meta['max_date']}")

    name2hid_simple = LH.name_to_hid_2025()  # live_history.py の現行(単純)解決

    names = df.get("馬名", pd.Series([""] * len(df))).map(_clean_name)
    sires = df.get("種牡馬", pd.Series([""] * len(df))).map(_clean_name)
    ages = pd.to_numeric(df.get("年齢"), errors="coerce")
    # _HistoryIndex.resolve は birth_year を期待。年齢と対象レース年から概算する
    year = int(date_str[:4])
    birth_years = (year - ages).where(ages.notna())

    rows = []
    for i in df.index:
        name = names.loc[i]
        sire = sires.loc[i]
        by = birth_years.loc[i]
        by_arg = int(by) if pd.notna(by) else None
        ent, status = idx.resolve(name, sire, by_arg)
        simple_hid = name2hid_simple.get(str(df.get("馬名", pd.Series(dtype=str)).get(i, "")))
        rows.append({
            "horse_name": str(df.get("馬名", pd.Series(dtype=str)).get(i, "")),
            "history_status": status,  # hit/new/ambiguous
            "history_ped_id": ent["ped_id"] if ent else None,
            "simple_name_resolved": simple_hid is not None,
            "simple_matches_history": (ent is not None and simple_hid == str(ent["ped_id"])),
        })
    res = pd.DataFrame(rows)

    def classify(r):
        if r["history_status"] == "new":
            return "正当な初出走馬"
        if r["history_status"] == "ambiguous":
            return "重複候補があり安全のため未解決"
        # history_status == "hit"
        if r["simple_matches_history"]:
            return "正常に血統登録番号相当で一致(history_index一致)"
        if r["simple_name_resolved"]:
            return "過去履歴はあるが単純名前解決がhistory_indexと不一致"
        return "過去履歴はあるがID解決に失敗(単純名前解決)"

    res["classification"] = res.apply(classify, axis=1)
    counts = res["classification"].value_counts().to_dict()

    has_history = res[res["history_status"] == "hit"]
    resolved_among_history_naive = (has_history["simple_matches_history"].mean()
                                    if len(has_history) else float("nan"))

    # live_history.py 現行 (resolve_idents, _HistoryIndex使用) での解決率。
    # build_2026_log/build_chain_target が両方ともこの関数を使うようになったため、
    # 定義上 hit 判定された馬は必ず同じ ped_id で chain 側にも現れる (100%が期待値)。
    # それでも実際に確認する (将来 resolve_idents の実装が分岐したときの回帰検知)。
    current_ident, current_status = LH.resolve_idents(df, int(date_str[:4]))
    current_resolved_among_history = float(
        (current_status == "hit").sum() / len(has_history)) if len(has_history) else float("nan")

    summary = {
        "date": date_str, "n_horses": int(len(res)),
        "classification_counts": counts,
        "n_with_history": int(len(has_history)),
        "n_new_legitimate": int((res["history_status"] == "new").sum()),
        "n_ambiguous": int((res["history_status"] == "ambiguous").sum()),
        "resolution_rate_among_horses_with_history_naive_v2": float(resolved_among_history_naive),
        "resolution_rate_among_horses_with_history_current_v3": current_resolved_among_history,
        "naive_all_horse_match_rate": float(res["simple_name_resolved"].mean()),
        "note": "v2(単純馬名一致)からv3(_HistoryIndex、種牡馬+生年)へ2026-09-19に刷新。"
                "current_v3はlive_history.pyが実際に使っている現行の解決率。",
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
