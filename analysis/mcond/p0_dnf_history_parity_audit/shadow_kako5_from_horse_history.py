# -*- coding: utf-8 -*-
"""
item9: Category B(kako5_race_count・same_td/dist/place_ratio)のlegacy-compatible
shadow patchプロトタイプ（本番へは一切適用しない）。

現状: `parse_kako5.build_from_kako5()`はTARGET週次kako5 CSV(固定5列横持ち)を
読むため、直近5枠のうちDNF/取消/除外があってもそれより前の実走で埋め直す
手段がない(データソース自体が5列しか保持していないため、コードだけでは
解決できない)。

本shadowは、course/jockey系(`serve_history_feats.py`)が既に使っている
`data/_horse_history.parquet`(全キャリア深度を持つ、5列に限定されない)を
kako5計算にも使うことで、training契約(現行の未修正定義=DNFは母集団から
除外、生存者だけで直近5走を構成)と**真にparityする**shadow実装を試作する。

意味定義specの新定義(DNFをwindow slotとして数える)は混入させない——
あくまで現行training契約(post-dropna母集団)への整合のみを目的とする。

本番コード(`parse_kako5.py`)・`data/_horse_history.parquet`は変更しない。
"""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(BASE))
from parse_kako5 import _compute_features, _safe_int, _safe_float, KAKO5_COLS  # noqa: E402
from parse_kako5 import build_from_kako5  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "out"
HORSE_HISTORY = BASE / "data" / "_horse_history.parquet"
KAKO5_DIR = BASE / "data" / "kako5"

_SURF = {"芝": "T", "ダ": "D", "ダート": "D"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_kako5_from_horse_history_training_contract(hist: pd.DataFrame, ped_id: int,
                                                       race_date: int, place: str,
                                                       surface: str, dist) -> dict:
    """training契約(post-dropna母集団=DNFを除外した生存者のみ)に真にparityする
    kako5計算。`data/_horse_history.parquet`は全キャリア深度を持つため、
    DNFが挟まっていても実際に「直近5"生存者"走」まで正しく遡れる
    (TARGET固定5列フォーマットの制約を受けない)。"""
    h = hist[(hist["ped_id"] == ped_id) & (hist["date"] < race_date) & hist["pos"].notna()]
    h = h.sort_values("date").tail(5)

    past_races = []
    for _, pr in h.iloc[::-1].iterrows():
        past_races.append({
            "着順": _safe_int(pr["pos"]), "人気": None, "上り3F": None,
            "TD": _SURF.get(str(pr["surface"]), ""), "距離": _safe_float(pr["dist"]),
            "場所": str(pr["place"]),
        })
    cur_td = _SURF.get(str(surface), "")
    return _compute_features(past_races, current_td=cur_td, current_dist=_safe_float(dist),
                              current_place=str(place))


def main():
    print("[1] data/_horse_history.parquet読み込み(既存ファイル、変更しない)...")
    hist = pd.read_parquet(HORSE_HISTORY)
    hist_hash = sha256_file(HORSE_HISTORY)
    print(f"    行数={len(hist):,}  sha256={hist_hash[:16]}...")

    print("[2] pos=NaN(2026年DNF)を持つ馬のうち、後続レースがある実例を抽出...")
    dnf_horses = hist[hist["pos"].isna()]["ped_id"].unique()
    print(f"    DNF歴を持つ馬(2026年分) = {len(dnf_horses)}")

    examples = []
    for ped_id in dnf_horses:
        h = hist[hist["ped_id"] == ped_id].sort_values("date")
        dnf_dates = h[h["pos"].isna()]["date"].tolist()
        later = h[h["date"] > max(dnf_dates)]
        if len(later) == 0:
            continue
        target = later.iloc[0]
        examples.append({
            "ped_id": int(ped_id), "name": str(target["name"]),
            "target_race_id": str(target["race_id"]), "target_date": int(target["date"]),
            "target_place": str(target["place"]), "target_surface": str(target["surface"]),
            "target_dist": float(target["dist"]),
        })
    print(f"    後続レースがある実例 = {len(examples)}")

    print("[3] 各実例について、shadow(training契約parity)版kako5を計算...")
    results = []
    for ex in examples:
        shadow_val = build_kako5_from_horse_history_training_contract(
            hist, ex["ped_id"], ex["target_date"], ex["target_place"],
            ex["target_surface"], ex["target_dist"])
        row = {**ex, "shadow_training_contract_parity": {
            c: (None if pd.isna(shadow_val.get(c)) else float(shadow_val.get(c))) for c in KAKO5_COLS
        }}
        results.append(row)

    out = {
        "purpose": "legacy-compatible shadow patchプロトタイプ(本番未適用)。"
                   "training契約(post-dropna母集団)へのparityのみを目的とし、"
                   "DNF_SEMANTIC_SPEC.mdの新定義は混入させていない。",
        "horse_history_parquet_sha256": hist_hash,
        "horse_history_snapshot_date_range": [int(hist["date"].min()), int(hist["date"].max())],
        "n_dnf_horses_2026": int(len(dnf_horses)),
        "n_examples_with_later_race": len(examples),
        "examples": results,
        "architectural_finding": (
            "現行build_from_kako5()はTARGET週次kako5 CSV(固定5列横持ち)を読むため、"
            "直近5枠にDNF/取消/除外が挟まっていても、それより前の実走で埋め直す手段が"
            "コードレベルで存在しない(データソース自体が5列を超える履歴を保持しない)。"
            "したがって真の意味でのtraining契約parityは、build_from_kako5()自体への"
            "コード修正だけでは達成できない——course/jockey系が既に使っている"
            "data/_horse_history.parquet(全キャリア深度)へデータソースを切り替える"
            "ことで初めて達成可能である。本shadowはこの切り替えのプロトタイプ。"
        ),
        "not_applied_to_production": True,
    }
    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / "shadow_kako5_legacy_patch_prototype.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "examples"}, ensure_ascii=False, indent=1))
    print(f"n_examples={len(results)}, showing first 3:")
    for r in results[:3]:
        print(json.dumps(r, ensure_ascii=False, indent=1, default=str))
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
