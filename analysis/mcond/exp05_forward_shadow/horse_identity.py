# -*- coding: utf-8 -*-
"""
horse_identity.py — 馬名 → 2025年末時点の状態 (前向き特徴生成の土台)
======================================================================
週次CSV (data/weekly/*.csv) は 血統登録番号 を含まず 馬名 しか無いため、EXP02/EXP03 の
状態 (dyn_skill の mu/var、career_runs、last_date) を 2025年末で確定させ、馬名でキャッチ
できるようにする。EXP02/EXP03 のコード自体は変更しない (wl_update, State, load_runs を
そのままimportして再利用、更新ループだけ薄く複製 — dyn_skill.run() は最終状態を返さない
ため)。

既知の限界 (MODEL_FREEZE.md に明記):
  - 馬名の重複 (引退馬と新馬の偶発的同名) は無視する。実務上は稀。
  - この状態は「2025年末までの履歴」で固定。2026年に入ってから既に走ったレースの結果は
    dyn_skill_mu / raw_career_runs / raw_days_since の更新に反映されない (v1の既知の制約)。
出力: out/horse_state_2025.json (馬名 -> {hid, last_date, career_runs, dyn_mu, dyn_var})
実行: python -m analysis.mcond.exp05_forward_shadow.horse_identity
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
from analysis.mcond.exp02_dynamic_skill_dev.dyn_skill import (  # noqa: E402
    load_runs, Hyper, State, wl_update, race_features)

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
MASTER = BASE / "data/master_v2_20130105-20251228.csv"
HP = Hyper(beta=2.0833, tau2_per_day=0.005)  # analysis/mcond/exp02_dynamic_skill_dev/out/build_meta.json の "selected" をそのまま流用 (再選択しない)


def _final_dyn_state(df: pd.DataFrame, hp: Hyper) -> State:
    """dyn_skill.run() のT1更新ループだけを複製 (wl_update/State は import して再利用)。
    最終状態 st を得るためだけの薄いラッパ (dyn_skill.py 自体は変更しない)。"""
    st = State(hp)
    for day, dd in df.groupby("date", sort=True):
        pending = []
        for rid, g in dd.groupby("rid16", sort=False):
            hids = g["hid"].to_numpy()
            pri = [st.prior(h, day) for h in hids]
            mu = np.array([p[0] for p in pri])
            var = np.array([p[1] for p in pri])
            fin = g["fin"].to_numpy(float)
            pending.append((hids, mu, var, fin))
        for hids, mu, var, fin in pending:
            nm, nv, _ = wl_update(mu, var, fin, hp.beta)
            for i, h in enumerate(hids):
                st.chg[h] = float(nm[i] - mu[i])
                st.g_mu[h], st.g_var[h] = float(nm[i]), float(nv[i])
                st.last[h] = day
                st.n[h] = st.n.get(h, 0) + 1
    return st


def build() -> dict:
    OUT.mkdir(exist_ok=True)
    df = load_runs()  # 2013-2025 全期間 (max_date指定なし)
    print(f"dyn_skill 状態再構築: {len(df):,} 行 / {df.hid.nunique():,} 頭")
    st = _final_dyn_state(df, HP)

    # 血統登録番号 -> 最新の馬名 (master の最終出走行)
    m = pd.read_csv(MASTER, encoding="utf-8-sig", low_memory=False,
                    usecols=["日付", "血統登録番号", "馬名"])
    m["date"] = pd.to_datetime(m["日付"].astype(str), format="%Y%m%d", errors="coerce")
    m = m.dropna(subset=["血統登録番号", "馬名", "date"])
    last_name = m.sort_values("date").groupby(m["血統登録番号"].astype(str))["馬名"].last()
    hid_to_name = last_name.to_dict()

    state: dict[str, dict] = {}
    dup = 0
    for h in st.g_mu:
        name = hid_to_name.get(h)
        if name is None:
            continue
        entry = {"hid": h, "last_date": st.last[h].strftime("%Y-%m-%d"),
                 "career_runs": int(st.n.get(h, 0)), "dyn_mu": float(st.g_mu[h]),
                 "dyn_var": float(st.g_var[h])}
        if name in state:
            dup += 1
            # 同名複数頭は直近出走(last_date)を優先する (引退馬より現役馬の状態が実用上重要)
            if entry["last_date"] > state[name]["last_date"]:
                state[name] = entry
        else:
            state[name] = entry

    meta = {"n_horses": len(state), "n_name_collisions": dup,
           "hyper": {"beta": HP.beta, "tau2_per_day": HP.tau2_per_day},
           "source_max_date": df["date"].max().strftime("%Y-%m-%d"),
           "note": "2025年末までの状態。2026年の既走分は未反映 (README.md参照)"}
    (OUT / "horse_state_2025.json").write_text(
        json.dumps({"meta": meta, "horses": state}, ensure_ascii=False), encoding="utf-8")
    print(f"保存: {len(state):,} 頭 (同名衝突 {dup} 件) -> out/horse_state_2025.json")
    return {"meta": meta, "horses": state}


if __name__ == "__main__":
    build()
