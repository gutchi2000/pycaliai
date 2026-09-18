# -*- coding: utf-8 -*-
"""
build.py — ハイパーパラメータ選択 (train の着順尤度のみ) → T1/T2 の時点安全な特徴を生成
=========================================================================================
1. spec.json の格子 (β × τ²) それぞれで T1 を 2013-2025 に流し、2016-2021 のレースの
   次走勝ち予測対数尤度の平均が最大の組を選ぶ。v6・市場・ROI は一切使わない。
2. 選んだ組で T1+T2 を流し、全出走の特徴を保存。
出力: data/_research/mcond/exp02_features.parquet
      out/hyper_selection.csv, out/build_meta.json
実行: python -m analysis.mcond.exp02_dynamic_skill_dev.build
"""
from __future__ import annotations
import itertools
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
from analysis.mcond.exp02_dynamic_skill_dev.dyn_skill import load_runs, run, Hyper  # noqa: E402

HERE = Path(__file__).resolve().parent
OUTD = HERE / "out"
FEAT = BASE / "data/_research/mcond/exp02_features.parquet"
SPEC = json.loads((HERE / "spec.json").read_text(encoding="utf-8"))


def main() -> None:
    OUTD.mkdir(exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=BASE, capture_output=True, text=True).stdout.strip()
    df = load_runs()
    print(f"runs={len(df):,} races={df.rid16.nunique():,}  commit={commit[:10]}")
    grid = SPEC["hyperparameter_selection"]["grid"]
    rows = []
    for b, t in itertools.product(grid["beta"], grid["tau2_per_day"]):
        _, ll = run(df, Hyper(b, t), with_t2=False, collect=False)
        yr = ll["date"].dt.year
        m = (yr >= 2016) & (yr <= 2021)
        rows.append({"beta": b, "tau2_per_day": t, "ll_win_train": float(ll.loc[m, "ll_win"].mean()),
                     "n_races": int(m.sum())})
        print(f"  beta={b:.4f} tau2={t:<6} ll_win(2016-21)={rows[-1]['ll_win_train']:.5f}", flush=True)
    sel = pd.DataFrame(rows).sort_values("ll_win_train", ascending=False)
    sel.to_csv(OUTD / "hyper_selection.csv", index=False, encoding="utf-8-sig")
    best = sel.iloc[0]
    hp = Hyper(float(best.beta), float(best.tau2_per_day))
    print(f"選択: beta={hp.beta:.4f} tau2_per_day={hp.tau2_per_day}")
    f, ll = run(df, hp, with_t2=True, collect=True)
    FEAT.parent.mkdir(parents=True, exist_ok=True)
    f.to_parquet(FEAT, index=False)
    meta = {"commit": commit, "selected": {"beta": hp.beta, "tau2_per_day": hp.tau2_per_day},
            "grid_result": rows, "n_rows": int(len(f))}
    (OUTD / "build_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str),
                                          encoding="utf-8")
    print(f"saved {len(f):,} rows -> {FEAT}")


if __name__ == "__main__":
    main()
