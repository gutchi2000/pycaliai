# -*- coding: utf-8 -*-
"""
item8: Category B(kako5_race_count・same_td/dist/place_ratio、4特徴)の
本番影響を、実際の2026年serve入力(実際のdata/kako5/{date}.csvファイル)を使い、
結果ラベルなしで測定する（読み取り専用）。

結果・ROIは一切見ない。current serve値(実際にparse_kako5.build_from_kako5()
が計算した値)と、同じ実データ(そのkako5 CSV行が実際に記録している最大5走の
生履歴)にtraining-contract互換の計算(意味定義specではなく、現行training
契約——止をwindow slotとして数える版——を適用した場合の値)を比較する。

実行: venv311\\Scripts\\python.exe -m analysis.mcond.p0_dnf_history_parity_audit.category_b_real_production_impact
"""
from __future__ import annotations
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(BASE))
from parse_kako5 import build_from_kako5, _compute_features, _safe_int, _safe_float, KAKO5_COLS  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "out"
KAKO5_DIR = BASE / "data" / "kako5"
STOP_CODES = {"止", "外", "消"}
B_FEATURES = ["kako5_race_count", "kako5_same_td_ratio", "kako5_same_dist_ratio", "kako5_same_place_ratio"]

RACE_OFFSETS = [(14, 18, 19, 21), (26, 30, 31, 33), (38, 42, 43, 45), (50, 54, 55, 57), (62, 66, 67, 69)]
TD_OFFSETS = [15, 27, 39, 51, 63]
DIST_OFFSETS = [16, 28, 40, 52, 64]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def training_contract_kako5(raw_slots: list[dict | None], current_td: str, current_dist, current_place: str) -> dict:
    """現行training契約(止をwindow slotとして数える、着順・上り3Fは欠損)で
    kako5を再計算する。raw_slots[i]は実際にCSVへ記録されている値そのもの
    (新しい順)、Noneはそもそも記録がない空スロット(取消・除外等で欠番、
    もしくは単純にキャリアが浅い)。"""
    past_races = []
    for slot in raw_slots:
        if slot is None:
            continue  # 記録なし(スロット自体が存在しない)はwindowへ含めない
        code = slot["pos_raw"]
        is_stop = code in STOP_CODES
        if code == "0" or code == "":
            continue  # TARGET側で「対象外」を示す0埋め、空スロットと同義
        past_races.append({
            "着順": None if is_stop else _safe_int(code),
            "人気": _safe_int(slot.get("ninki")),
            "上り3F": None if is_stop else _safe_float(slot.get("agari")),
            "TD": slot.get("td", ""),
            "距離": _safe_float(slot.get("dist")),
            "場所": slot.get("place", ""),
        })
    return _compute_features(past_races, current_td=current_td, current_dist=current_dist,
                              current_place=current_place)


def parse_kako5_file_raw(path: Path) -> list[dict]:
    """kako5 CSVを生のまま読み、各データ行についてヘッダ情報+5スロットの
    生コード(pos_raw、止/外/消/数値/0を区別したまま)を抽出する。"""
    rows_out = []
    current_header = None
    with open(path, encoding="cp932", errors="replace") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) <= 1:
                continue
            if len(row) == 19 and row[0] and len(row[0]) >= 10 and row[0][:4].isdigit():
                current_header = {"race_id": row[0], "place": row[3], "td_raw": row[8],
                                   "dist": _safe_float(row[9])}
                continue
            if len(row) == 72 and row[0] in ("枠番",):
                continue
            if len(row) == 72 and row[0].isdigit() and current_header:
                ban = _safe_int(row[2])
                if ban is None:
                    continue
                td_map = {"芝": "T", "ダ": "D", "ダート": "D", "T": "T", "D": "D"}
                slots = []
                any_stop = False
                for i, (place_i, pos_i, ninki_i, agari_i) in enumerate(RACE_OFFSETS):
                    pos_raw = row[pos_i].strip() if pos_i < len(row) else ""
                    if pos_raw in STOP_CODES:
                        any_stop = True
                    if pos_raw in ("", "0"):
                        slots.append(None)
                        continue
                    slots.append({
                        "pos_raw": pos_raw, "ninki": row[ninki_i] if ninki_i < len(row) else "",
                        "agari": row[agari_i] if agari_i < len(row) else "",
                        "td": row[TD_OFFSETS[i]] if TD_OFFSETS[i] < len(row) else "",
                        "dist": row[DIST_OFFSETS[i]] if DIST_OFFSETS[i] < len(row) else "",
                        "place": row[place_i] if place_i < len(row) else "",
                    })
                rows_out.append({
                    "race_id": current_header["race_id"], "ban": ban,
                    "current_place": current_header["place"],
                    "current_td": td_map.get(current_header["td_raw"], ""),
                    "current_dist": current_header["dist"],
                    "slots": slots, "any_stop_in_window": any_stop,
                })
    return rows_out


def main():
    files = sorted(KAKO5_DIR.glob("2026*.csv"))
    print(f"対象ファイル数: {len(files)}")
    all_rows = []
    for f in files:
        try:
            rows = parse_kako5_file_raw(f)
        except Exception as e:
            print(f"  [skip] {f.name}: {e}")
            continue
        for r in rows:
            r["source_file"] = f.name
        all_rows.extend(rows)
    print(f"総行数: {len(all_rows)}")

    stop_rows = [r for r in all_rows if r["any_stop_in_window"]]
    print(f"直近5走windowに止/外/消を含む実データ行: {len(stop_rows)}")

    results = []
    # ファイルごとにbuild_from_kako5()を1回だけ実行してキャッシュ
    serve_cache: dict[str, pd.DataFrame] = {}
    for fname in {r["source_file"] for r in stop_rows}:
        df = build_from_kako5(KAKO5_DIR / fname)
        serve_cache[fname] = df.set_index(["レースID(新)", "馬番"])

    for r in stop_rows:
        key = (r["race_id"], r["ban"])
        serve_df = serve_cache.get(r["source_file"])
        if serve_df is None or key not in serve_df.index:
            continue
        serve_val = serve_df.loc[key]
        if isinstance(serve_val, pd.DataFrame):
            serve_val = serve_val.iloc[0]
        train_val = training_contract_kako5(r["slots"], r["current_td"], r["current_dist"], r["current_place"])

        row_result = {"race_id": r["race_id"], "ban": r["ban"], "source_file": r["source_file"]}
        for c in B_FEATURES:
            sv = serve_val.get(c, np.nan)
            tv = train_val.get(c, np.nan)
            sv = float(sv) if pd.notna(sv) else None
            tv = float(tv) if pd.notna(tv) else None
            diff = None if (sv is None or tv is None) else sv - tv
            row_result[c] = {"serve": sv, "training_contract": tv, "diff": diff}
        results.append(row_result)

    n_affected = len(results)
    summary = {"n_files_scanned": len(files), "n_total_kako5_rows": len(all_rows),
               "n_rows_with_stop_in_window": len(stop_rows),
               "n_rows_comparable": n_affected,
               "file_hashes": {f.name: sha256_file(f) for f in files}}
    for c in B_FEATURES:
        diffs = [r[c]["diff"] for r in results if r[c]["diff"] is not None]
        n_mismatch = sum(1 for d in diffs if abs(d) > 1e-9)
        summary[c] = {
            "n_compared": len(diffs), "n_mismatch": n_mismatch,
            "mismatch_rate": round(n_mismatch / len(diffs), 4) if diffs else None,
            "mean_abs_diff": float(np.mean([abs(d) for d in diffs])) if diffs else None,
            "max_abs_diff": float(np.max([abs(d) for d in diffs])) if diffs else None,
        }

    out = {"summary": summary, "detail": results, "note": "結果・ROIは見ていない(◎変更・順位変更・raw score差のみ、"
           "着順は本スクリプトが読む対象データに存在しないため参照していない)"}
    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / "category_b_real_production_impact.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1, default=str))
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
