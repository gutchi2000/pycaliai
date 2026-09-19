# -*- coding: utf-8 -*-
"""
feature_parity_audit.py — F-serve 117特徴の完全監査 (spec §2)、F-forward の確定
==================================================================================
2026-09-19実データ (data/weekly/20260919.csv, 287頭24レース) で
feature_snapshot.build() を実行した結果から、117特徴それぞれを分類しCSVへ出力する。
実行: python -m analysis.mcond.exp05_forward_shadow.feature_parity_audit
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
from analysis.mcond.exp05_forward_shadow import feature_snapshot as FS  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "FEATURE_PARITY.csv"

TEST_DATE = "20260919"

ALWAYS_MISSING = {"c1__前走日付", "c1__前好走", "c1__限定", "c1__ブリンカー"}
LIVE_STATEFUL = {"raw_jq_delta", "raw_jt_pair"}  # chain(騎手/調教師コード)ベース、馬identity不要
LIVE_HORSE_CHAIN = {"raw_log_int", "raw_dist_chg", "raw_venue_chg", "raw_surface_chg",
                    "raw_cls_chg", "raw_jockey_same"}  # 馬名一致chainが必要 (前走が無い馬はNaN)
LIVE_DYN_SKILL = {"dyn_skill_mu", "horse_skill_minus_field", "raw_career_runs"}
LIVE_SIMPLE = {"raw_days_since"}


def classify(col: str, train_avail: bool, live_avail: bool, missing_rate_live: float) -> tuple[str, str]:
    if not train_avail:
        return "対象外", "F-serve(=学習時候補)に無い"
    if col in ALWAYS_MISSING:
        return "データ源なし", "週次CSVに列自体が存在しない(2026-09-19実データで確認)"
    if not live_avail:
        return "実装不足だが追加可能", "現状NaNだが理論上は追加実装で埋まる可能性"
    if missing_rate_live > 0.15:
        return "部分的に再現可能(行単位欠損あり)", f"前走が無い馬・新規状態解決に失敗した馬でNaN (欠損率{missing_rate_live:.1%})"
    return "ライブで同一定義を再現可能", ""


def main() -> None:
    feats, meta = FS.build(TEST_DATE, save=True)
    feature_lists = json.loads(
        (BASE / "analysis/mcond/exp05_market_residual_dev/out/feature_lists.json").read_text(encoding="utf-8"))
    F_serve = feature_lists["F_serve"]

    rows = []
    for col in F_serve:
        block = "C1" if col.startswith("c1__") else ("C2" if col.startswith("raw_") and col not in
                                                       LIVE_DYN_SKILL else
                                                       ("C3" if col in LIVE_DYN_SKILL | LIVE_SIMPLE else "C2"))
        found_key = col.replace("c1__", "c1__") if col.startswith("c1__") else col
        train_avail = True
        live_col_present = col in feats.columns
        if not live_col_present:
            status, reason = "対象外", "feature_snapshot出力に無い(実装漏れ、要確認)"
            missing_rate = 1.0
        else:
            missing_rate = float(feats[col].isna().mean())
            live_avail_flag = meta["column_found"].get(found_key, missing_rate < 1.0)
            status, reason = classify(col, train_avail, bool(live_avail_flag), missing_rate)

        generation_time = ("週次バッチ(feature_snapshot.py, 発走より十分前)" if block == "C1" or
                           col in LIVE_HORSE_CHAIN | LIVE_DYN_SKILL | LIVE_SIMPLE else
                           "レース単位(市場snapshotと同時)" if col in LIVE_STATEFUL else "週次バッチ")
        time_safe = True  # 全列: 対象レース自身の結果・市場は不使用 (chainはas_of_date以前の確定結果のみ)
        rows.append({
            "feature_name": col, "feature_block": block,
            "training_available": True, "live_available": status == "ライブで同一定義を再現可能" or
            status == "部分的に再現可能(行単位欠損あり)",
            "live_source": ("weekly_csv+frozen_encode" if block == "C1" else
                            "live_history.chain(騎手/調教師コード)" if col in LIVE_STATEFUL else
                            "live_history.chain(馬名identity)" if col in LIVE_HORSE_CHAIN else
                            "live_history.dyn_skill_live+horse identity" if col in LIVE_DYN_SKILL else
                            "weekly_csv(間隔)" if col in LIVE_SIMPLE else "unknown"),
            "generation_time": generation_time, "time_safe": time_safe,
            "missing_rate_train": None,  # 研究側(exp04)は別途DATA_AUDIT.md参照、ここではlive側のみ記録
            "missing_rate_live": round(missing_rate, 4),
            "current_handling": status,
            "action": ("除外(F-forwardに含めない)" if status in ("データ源なし", "対象外") else "採用"),
            "reason": reason,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")

    n_adopt = (df["action"] == "採用").sum()
    n_exclude = len(df) - n_adopt
    print(f"F-serve {len(df)}列 → F-forward採用 {n_adopt} / 除外 {n_exclude}")
    print(df[df["action"] != "採用"][["feature_name", "current_handling", "reason"]].to_string(index=False))
    print(f"\n保存: {OUT.relative_to(BASE)}")


if __name__ == "__main__":
    main()
