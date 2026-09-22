# -*- coding: utf-8 -*-
"""
build_contract.py — clean feature contract を base_train (2016-2021) だけで実測して凍結する
============================================================================================
出力: out/feature_contract.json
  features.clean   : R0-clean / R1 / R1-noctx の正式入力
  features.prodref : R0-prodref (v6 120列そのまま、参考値)
  v6_dead          : v6 変換後 base_train 被覆率 0% (v6 では全行 -9999 定数) → 両モデルから除外
  race_constant    : base_train で全レース内一定の列 (placebo で相手側を自レース値に上書き)
実行: python -m analysis.mcond.exp15_race_as_set_dev.build_contract
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import common as C


def main():
    feats, cats = C.v6_meta()
    df = C.load_rows(force=True)
    tr = df[df["period"] == "train"]
    spec_fc = C.SPEC["feature_contract"]
    excl_id = spec_fc["exclude_identifier_or_asof_violation"]
    never = set(spec_fc["never_included"])
    assert not (never & set(feats)), "v6 feature_cols に never_included 列がある"

    cov, card = {}, {}
    for c in feats:
        if c in cats:
            cov[c] = 1.0
            card[c] = int(tr[c].astype(str).nunique())
        else:
            v = pd.to_numeric(tr[c], errors="coerce")
            cov[c] = float(v.notna().mean())
            card[c] = int(v.nunique())
    dead = [c for c in feats if c not in cats and cov[c] == 0.0]

    clean = [c for c in feats if c not in excl_id and c not in dead]
    # 一意識別子ガード (spec): カーディナリティ > base_train 行数の 1%
    lim = 0.01 * len(tr)
    over = {c: card[c] for c in clean if card[c] > lim}
    bad = sorted(set(over) - set(spec_fc["unique_id_allowlist"]))
    assert not bad, f"一意識別子ガード停止 (allowlist 外): {bad}"

    # race 内一定列 (base_train)
    g = tr.groupby("rid16")
    const = {}
    for c in clean:
        nun = g[c].nunique(dropna=False)
        const[c] = float((nun <= 1).mean())
    race_constant = [c for c, f in const.items() if f == 1.0]

    cg = C.class_group(tr["クラス名"])
    cmap = (pd.DataFrame({"クラス名": tr["クラス名"].astype(str), "group": cg})
            .drop_duplicates().sort_values("クラス名"))

    out = {
        "built_from": "base_train 2016-01-01..2021-12-31 のみ",
        "row_hash_2016_2023": C.row_hash(df),
        "n_rows": {k: int((df["period"] == k).sum()) for k in C.PERIODS},
        "n_races": {k: int(df.loc[df["period"] == k, "rid16"].nunique()) for k in C.PERIODS},
        "date_range": [int(df["date"].min()), int(df["date"].max())],
        "v6_feature_count": len(feats),
        "excluded_identifier_or_asof": excl_id,
        "v6_dead_zero_coverage": dead,
        "features": {"clean": clean, "prodref": feats},
        "cat_cols": [c for c in cats],
        "n_clean": len(clean),
        "n_clean_cat": len([c for c in clean if c in cats]),
        "coverage_base_train": cov,
        "cardinality_base_train": card,
        "unique_id_guard": {"limit": lim, "over_limit": over,
                            "allowlist": spec_fc["unique_id_allowlist"]},
        "race_constant_fraction": const,
        "race_constant": race_constant,
        "class_group_map": dict(zip(cmap["クラス名"], cmap["group"])),
    }
    C.dump(out, "feature_contract.json")
    print(f"rows {out['n_rows']} races {out['n_races']} range {out['date_range']}")
    print(f"dead(v6 -9999定数) = {dead}")
    print(f"clean = {len(clean)} 列 (cat {out['n_clean_cat']})")
    print(f"card > 1% ({lim:.0f}) : {over}")
    print(f"race_constant ({len(race_constant)}) = {race_constant}")
    near = {c: f for c, f in const.items() if 0.99 <= f < 1.0}
    print(f"near-constant (0.99..1) = {near}")
    print("partial coverage (<0.5):", {c: round(v, 3) for c, v in cov.items() if v < 0.5})


if __name__ == "__main__":
    main()
