# -*- coding: utf-8 -*-
"""
audit.py — Gate 0 データ監査 (事実確認。結果は out/gate0_audit.json)
====================================================================
1. 馬主・調教師が過去行に最新値で上書きされていないか
   キャリア中に値が変わる馬の割合。上書きなら「変わる馬」がほぼゼロになる。
   上書きの疑いがあれば、その列は時点整合性なしとして使わない。
2. 前走連結の整合 (直前行 == master の 前走日付) の年別成立率
3. 市場オッズの取得時点 (pre / am9 の 確定までの分)
4. v6 スコア: OOF と本番の 2023 重複での順位相関、各年の供給源
5. 2024-25 の既往利用回数 (reports/*.json と *.py の参照数)
実行: python -m analysis.mcond.exp01_choice_dev.audit
"""
from __future__ import annotations
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[3]
OUTD = Path(__file__).resolve().parent / "out"


def main() -> None:
    OUTD.mkdir(exist_ok=True)
    res = {}
    m = pd.read_csv(BASE / "data/master_v2_20130105-20251228.csv", encoding="utf-8-sig",
                    low_memory=False, usecols=["日付", "血統登録番号", "調教師コード", "馬主(最新/仮想)",
                                               "騎手コード", "着順"])
    m = m[pd.to_numeric(m["着順"], errors="coerce") >= 1]
    g = m.groupby("血統登録番号")
    n_runs = g.size()
    multi = n_runs[n_runs >= 5].index
    mm = m[m["血統登録番号"].isin(multi)]
    gg = mm.groupby("血統登録番号")
    res["owner_change_rate"] = float((gg["馬主(最新/仮想)"].nunique() > 1).mean())
    res["trainer_change_rate"] = float((gg["調教師コード"].nunique() > 1).mean())
    res["jockey_change_rate"] = float((gg["騎手コード"].nunique() > 1).mean())
    res["n_horses_5plus_runs"] = int(len(multi))
    # 調教師が変わった馬で、変化が「一度きりで以後戻らない」割合 (転厩らしさ)
    def monotone_switch(s):
        vals = s.tolist()
        changes = sum(1 for a, b in zip(vals, vals[1:]) if a != b)
        return changes
    ch = gg["調教師コード"].apply(monotone_switch)
    res["trainer_switch_count_dist"] = ch.value_counts().sort_index().head(6).to_dict()
    res["trainer_switch_count_dist"] = {int(k): int(v) for k, v in res["trainer_switch_count_dist"].items()}

    f = pd.read_parquet(BASE / "data/_research/mcond/exp01_features.parquet")
    tot = m.assign(year=m["日付"].astype(str).str[:4].astype(int)).groupby("year").size()
    ok = f.groupby(f["date"].dt.year).size()
    res["chain_ok_rate_by_year"] = {int(y): round(float(ok.get(y, 0) / tot[y]), 4) for y in tot.index}

    mk = pd.read_parquet(BASE / "data/_research/mcond/market.parquet")
    for s in ("pre", "am9"):
        x = mk[mk.snap == s].drop_duplicates("rid16")
        res[f"market_{s}_min_before_final"] = {
            "median": float(x.min_before_final.median()), "p01": float(x.min_before_final.quantile(.01)),
            "p99": float(x.min_before_final.quantile(.99)), "races": int(len(x))}
        res[f"market_{s}_overround_median"] = float(x.race_implied_sum.median())
    res["market_pre_is_before_T10"] = bool(
        mk[mk.snap == "pre"].min_before_final.min() >= 15)

    b = pd.read_parquet(BASE / "data/_research/mcond/base.parquet")
    res["v6_source_by_year"] = b.groupby("year")["v6_src"].first().to_dict()
    res["v6_source_by_year"] = {int(k): v for k, v in res["v6_source_by_year"].items()}
    res["v6_tau_by_source"] = b.groupby("v6_src")["tau"].first().to_dict()

    n_json = sum(1 for p in glob.glob(str(BASE / "reports/*.json"))
                 if re.search(r"2024-25|test2024|2024-2025", Path(p).read_text(encoding="utf-8", errors="ignore")))
    n_py = 0
    for p in glob.glob(str(BASE / "**/*.py"), recursive=True):
        if "venv" in p or "worktrees" in p:
            continue
        try:
            if re.search(r"2024-25|test2024|2024-2025", Path(p).read_text(encoding="utf-8", errors="ignore")):
                n_py += 1
        except Exception:
            pass
    res["prior_use_2024_25"] = {"report_json": n_json, "py_files": n_py,
                                "status": "再利用済み期間 (探索的OOSとしてのみ扱う)"}

    (OUTD / "gate0_audit.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
