# -*- coding: utf-8 -*-
"""
add_train_cov_baseline.py — serve canary の基準に「学習時の充足率」を足す
========================================================================
既存 canary の欠陥: baseline_cov は「最近の健全とみなした serve 週」から採る。
すでに劣化している状態を基準にしてしまうと、劣化はそのまま正常として素通りする
(実例: prev_hosei は学習時 85.4% だが baseline が 46.3% になっており、floor 20% を
 一度も割らないので一度も鳴らなかった)。

学習時 (train split) の充足率を絶対基準として baseline JSON に埋める。
実行: python -m analysis.add_train_cov_baseline
"""
from __future__ import annotations
import json
import sys
import warnings
from pathlib import Path

import joblib
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
warnings.filterwarnings("ignore")

import backtest_pl_ev as be                       # noqa: E402
from backtest_pl_ev import apply_encoders, COL_JYUN, COL_RID  # noqa: E402


def main() -> None:
    be.MODEL_PKL = BASE / "models/unified_rank_v6.pkl"
    bundle = joblib.load(be.MODEL_PKL)
    feats, encs = bundle["feature_cols"], bundle["encoders"]
    gains = bundle["model"].feature_importance(importance_type="gain")
    gain_pct = {f: round(100.0 * float(g) / float(gains.sum()), 4)
                for f, g in zip(feats, gains)}

    df = pd.read_csv(be.MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df = df.dropna(subset=[COL_JYUN, COL_RID, "split"]).copy()
    for key, mod, fn in [("race_relative_mode", "race_relative_feats", "add_race_relative_feats"),
                         ("course_affinity_mode", "course_affinity_feats", "add_course_affinity_feats"),
                         ("grade_feats_mode", "grade_feats", "add_grade_feats")]:
        if bundle.get(key):
            df = getattr(__import__(mod), fn)(df, mode=bundle[key])
    tr = apply_encoders(df[df["split"] == "train"].copy(), encs)[feats]
    cov = tr.apply(pd.to_numeric, errors="coerce").notna().mean()

    p = BASE / "data/serve_feature_baseline.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    d["train_cov"] = {f: round(float(cov[f]), 4) for f in feats}
    d["gain_pct"] = gain_pct
    d["train_cov_note"] = (
        "学習 split での特徴充足率。baseline_cov (最近の serve 週) は劣化済みの値を"
        "正常と誤認する自己参照の穴があるため、絶対基準としてこちらを併記する。"
        "analysis/add_train_cov_baseline.py で再生成。")
    p.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"train_cov を {len(feats)} 特徴ぶん書き込み -> {p}")

    print("\n=== serve が学習より大きく落ちている高gain特徴 ===")
    bc = d.get("baseline_cov", {})
    rows = [(f, float(cov[f]), bc.get(f), gain_pct[f]) for f in feats
            if bc.get(f) is not None and gain_pct[f] >= 0.3
            and bc[f] < 0.7 * float(cov[f])]
    for f, t, s, g in sorted(rows, key=lambda r: -r[3]):
        print(f"  {f:<24} train={t:.3f}  serve={s:.3f}  gain={g:.2f}%")


if __name__ == "__main__":
    main()
