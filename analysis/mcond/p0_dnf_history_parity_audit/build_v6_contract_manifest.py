# -*- coding: utf-8 -*-
"""item4: current v6の契約をmanifest化する（読み取り専用）。"""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

BASE = Path(__file__).resolve().parent.parent.parent.parent
MASTER_V2 = BASE / "data" / "master_v2_20130105-20251228.csv"
MODEL_PKL = BASE / "models" / "unified_rank_v6.pkl"
HORSE_HISTORY = BASE / "data" / "_horse_history.parquet"
KAKO5_MODULE = BASE / "parse_kako5.py"
SERVE_HIST_MODULE = BASE / "serve_history_feats.py"
BUILD_HORSE_HISTORY_MODULE = BASE / "build_horse_history.py"

OUT_DIR = Path(__file__).resolve().parent / "out"

HIST6 = ["course_n_prev", "course_win_rate", "course_top3_rate",
          "jockey_n_prev", "jockey_win_rate", "jockey_top3_rate"]
KAKO5_13 = ["kako5_avg_pos", "kako5_std_pos", "kako5_best_pos", "kako5_avg_agari3f",
            "kako5_best_agari3f", "kako5_same_td_ratio", "kako5_same_dist_ratio",
            "kako5_same_place_ratio", "kako5_pos_trend", "kako5_race_count",
            "kako5_expected_good_count", "kako5_hidden_good_count", "kako5_same_cond_best_pos"]


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main():
    bundle = joblib.load(MODEL_PKL)
    model, feats, encs, cat_cols = bundle["model"], bundle["feature_cols"], bundle["encoders"], bundle["cat_cols"]

    # feature schema hash: 特徴名+順序を固定文字列化してhash
    feature_schema_str = json.dumps(feats, ensure_ascii=False)
    feature_schema_hash = sha256_bytes(feature_schema_str.encode("utf-8"))

    # encoder hash: 各カテゴリ列のclasses_を固定文字列化してhash
    enc_repr = {c: list(le.classes_) for c, le in sorted(encs.items())}
    encoder_hash = sha256_bytes(json.dumps(enc_repr, ensure_ascii=False, sort_keys=True).encode("utf-8"))

    model_hash = sha256_file(MODEL_PKL)
    master_v2_hash = sha256_file(MASTER_V2)
    horse_history_hash = sha256_file(HORSE_HISTORY)

    manifest = {
        "created_at": "2026-09-22",
        "model_path": str(MODEL_PKL.relative_to(BASE)),
        "model_pkl_sha256": model_hash,
        "training_master_path": str(MASTER_V2.relative_to(BASE)),
        "training_master_sha256": master_v2_hash,
        "training_master_rows": None,
        "feature_schema_hash": feature_schema_hash,
        "feature_schema_n_features": len(feats),
        "feature_schema_list": feats,
        "encoder_hash": encoder_hash,
        "encoder_cat_cols": cat_cols,
        "model_num_trees": model.num_trees(),
        "seed": bundle.get("seed"),
        "optuna_best_params": bundle.get("optuna_best_params"),
        "sample_weight_alpha": bundle.get("sample_weight_alpha"),
        "label_scheme": bundle.get("label_scheme"),
        "live_serve_horse_history_parquet_sha256_snapshot": horse_history_hash,
        "live_serve_horse_history_note": (
            "data/_horse_history.parquet は週次で再生成される可変ファイルのため、"
            "このsha256は本監査実行時点のスナップショットに過ぎない(model契約の"
            "固定要素ではない)。"
        ),
        "19_features_definitions": {},
        "parity_verdict_by_feature_group": {},
    }

    for c in HIST6:
        manifest["19_features_definitions"][c] = {
            "training_definition": "build_master_v2.py:compute_history_features() — "
                "post-dropna(626,774行)母集団でexpanding cumcount。DNF(止)は母集団から"
                "欠落するため恒久的に過小カウント(実測: 3,401-3,854/626,774行)",
            "live_serve_definition_2013_2025": "serve_history_feats.compute_row_feats() が "
                "data/_horse_history.parquet(build_horse_history.py生成)を参照。"
                "2013-2025分はmaster_v2をそのまま再利用するため training と"
                "全く同一の欠落母集団",
            "live_serve_definition_2026": "build_horse_history.py:load_2026_history() が "
                "data/kekka/{date}.csv×data/weekly/{date}.csvをdropna無しでjoin。"
                "止はpos=NaN行として残るが、外/消も同じ'0'コードに潰れるため区別不能",
            "desired_semantic_definition": "DNF_SEMANTIC_SPEC.md: 止は分母に含める"
                "(出走経験1回)、外/消は含めない",
            "train_vs_serve_code_level_parity_given_same_input": "MATCH（本監査"
                "train_serve_same_input_parity.pyで実測確認、全8シナリオ一致）",
            "train_vs_serve_actual_parity_2013_2025": "PARITY（同一ファイルを参照する"
                "ため定義上train=serveが保証される。ただし両方とも意味的に誤り）",
            "train_vs_serve_actual_parity_2026": "UNCONFIRMED（データソースが異なり、"
                "外/消の誤混入リスクが構造的に存在するが実発生率は未計測）",
        }
    for c in KAKO5_13:
        is_ratio_or_count = c in ("kako5_race_count", "kako5_same_td_ratio",
                                    "kako5_same_dist_ratio", "kako5_same_place_ratio")
        manifest["19_features_definitions"][c] = {
            "training_definition": "parse_kako5.py:build_from_master() — "
                "post-dropna(626,774行)母集団の位置ベース直近5行window。DNFは"
                "window内に一切現れないため、本来より古い走まで遡って5走を構成する"
                "(window drift、実測: 最大1.8%行・kako5_avg_agari3f)",
            "live_serve_definition": "parse_kako5.py:build_from_kako5()（訓練用"
                "build_from_master()とは別関数） — TARGET週次kako5 CSV(固定5列"
                "横持ち)を読む。止/外/消いずれも `_safe_int()`失敗により該当"
                "スロットを無条件でcontinue(スキップ)、代替を探さない",
            "desired_semantic_definition": "DNF_SEMANTIC_SPEC.md: 止はwindow slot"
                "へ含める(着順/上り3Fは欠損、TD/距離/場所は実値)、外/消は含めない",
            "train_vs_serve_code_level_parity_given_same_input": (
                "MISMATCH（本監査で実測確認: DNFがwindow内にある場合、train相当は"
                "DNFをslot消費として数えるがserveは数えない）" if is_ratio_or_count else
                "MATCH（本監査で実測確認: 着順・上り3Fベースの集約はいずれの実装も"
                "DNF自身の値を除外するため、同一の有効レース集合が与えられれば"
                "一致する）"
            ),
            "train_vs_serve_actual_parity": (
                "SKEW CONFIRMED（B: 実際のserve skew。同一の正しい入力を与えても"
                "train相当とserveのコードロジック自体が異なる値を返す）" if is_ratio_or_count else
                "UNCONFIRMED（コードの数式は一致するが、実際にtrainとserveへ渡る"
                "『直近5走』の実レース集合が異なる可能性がある—trainはwindow drift、"
                "serveはTARGET側5列固定フォーマットの打ち切り—どちらも入力データの"
                "構成が食い違いうるため、実データでの数値一致は別途未検証）"
            ),
        }

    manifest["parity_verdict_by_feature_group"] = {
        "course_jockey_6features_2013_2025": "A（train=serveだが両方意味的に誤り。"
            "同一ファイルmaster_v2を参照するため定義上のparityは保証される）",
        "course_jockey_6features_2026": "D（2026経路だけ別定義、実発生規模は未計測につきE寄り）",
        "kako5_race_count_and_3ratios_4features": "B（実際のserve skew、同一正入力でも"
            "コードロジックが異なる値を返すことを実測確認）",
        "kako5_remaining_9features": "E（コード式は一致するが実際に投入される直近5走の"
            "レース集合が食い違う可能性があり未確認）",
        "current_production_bug_scope": "『production bug』と呼ぶのはBまたは現在liveへ"
            "影響するAのみ: course_jockey_6features(2013-2025分、A)とkako5_race_count/"
            "same_td_dist_place_ratio(4特徴、B)。kako5の残り9特徴(E)とcourse系2026分(D)は"
            "現時点でproduction bugと断定しない",
    }

    manifest["do_not_feed_corrected_features_to_current_v6"] = (
        "現在のunified_rank_v6.pklはバグを含む特徴分布で学習されているため、"
        "corrected features(DNF_SEMANTIC_SPEC.md準拠)を現行モデルへ直接入力しない。"
        "学習時と異なる意味の値になり、モデルが未知の分布として誤って解釈するリスクがある。"
    )

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / "v6_contract_manifest.json"
    out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("feature_schema_list", "19_features_definitions")},
                      ensure_ascii=False, indent=1))
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
