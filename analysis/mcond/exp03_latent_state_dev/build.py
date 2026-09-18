# -*- coding: utf-8 -*-
"""
build.py — EXP03 の特徴生成 (spec.json どおり)
===============================================
1. ハイパーパラメータ: 格子 (半減期 × τa² × qs × β) それぞれで 2013-2025 を流し、
   2016-2021 の次走勝ち予測対数尤度 (a+s の softmax) が最大の組を選ぶ。v6・市場・ROI 不使用。
   参考として qs≈0 (状態なし) の行も計算するが選択対象にしない。
2. innovation の定義: 選んだ組で、候補A (標準化スコア残差 Ω/√Δ) と 候補B (期待順位分位 − 実順位分位)
   の「持続性」= 60日以内の連続2走での順位相関 (2016-2021) を比べ、大きい方を特徴に使う。
3. 生の対照特徴 (時点安全): 出走回数・直近180日の出走回数・前走からの日数・休養/短間隔フラグ・
   前走着順・前走着順分位・前走着差・直近3走の着順分位平均
4. 既存 EWMA: exp_recency_ewm.py と同じ定義 (前走値の走数ベース EWMA, 半減期1.8走, 14列)
出力: data/_research/mcond/exp03_features.parquet, out/hyper_selection.csv, out/innovation_choice.json,
      out/build_meta.json
"""
from __future__ import annotations
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
from analysis.mcond.exp03_latent_state_dev.latent_state import load_runs, run, Hyper  # noqa: E402

HERE = Path(__file__).resolve().parent
OUTD = HERE / "out"
FEAT = BASE / "data/_research/mcond/exp03_features.parquet"
SPEC = json.loads((HERE / "spec.json").read_text(encoding="utf-8"))
MASTER = BASE / "data/master_v2_20130105-20251228.csv"
EWM_SIGS = [("pos", "前走確定着順"), ("agari", "前走上り3F"), ("rpci", "前走RPCI"), ("chakusa", "前走着差タイム"),
            ("hosei", "prev_hosei"), ("pci", "前PCI"), ("avg1f", "前走平均1Fタイム")]


def ewma_from_log(iv: pd.DataFrame, f: pd.DataFrame, col: str, halflife: float):
    """innovation の記録から、各レース直前の (前回 innovation, 経過日数で減衰させた EWMA) を作る。
    ew ← ρ·ew_前回 + (1−ρ)·z (初走は z)、予測時点では ρ(経過日数)·ew。対象レース自身の z は使わない。"""
    ln2 = np.log(2.0)
    iv = iv.sort_values(["hid", "date"])
    last_z, ew_at = {}, {}
    ew, lastd = {}, {}
    for h, d, rid, z in zip(iv["hid"], iv["date"], iv["rid16"], iv[col]):
        if h in ew:
            rho = np.exp(-ln2 * (d - lastd[h]).days / halflife)
            ew_at[(rid, h)] = (last_z[h], rho * ew[h])
            ew[h] = rho * ew[h] + (1 - rho) * z
        else:
            ew_at[(rid, h)] = (np.nan, np.nan)
            ew[h] = z
        last_z[h], lastd[h] = z, d
    v = [ew_at.get((r, h), (np.nan, np.nan)) for r, h in zip(f["rid16"], f["hid"])]
    return np.array([x[0] for x in v], float), np.array([x[1] for x in v], float)


def raw_and_ewm() -> pd.DataFrame:
    cols = ["日付", "レースID(新)", "血統登録番号", "馬番", "着順", "出走頭数", "前走日付", "前走出走頭数"] + \
           [s for _, s in EWM_SIGS]
    m = pd.read_csv(MASTER, encoding="utf-8-sig", low_memory=False, usecols=cols)
    m["date"] = pd.to_datetime(m["日付"].astype(str), format="%Y%m%d")
    m["rid16"] = m["レースID(新)"].astype(str).str[:16]
    m["ban"] = pd.to_numeric(m["馬番"], errors="coerce")
    m["fin"] = pd.to_numeric(m["着順"], errors="coerce")
    m = m.dropna(subset=["ban", "血統登録番号"])
    m = m[m["fin"] >= 1].copy()
    m["ban"] = m["ban"].astype(int)
    m["hid"] = m["血統登録番号"].astype(str)
    m = m.sort_values(["hid", "date", "rid16"]).reset_index(drop=True)
    g = m.groupby("hid", sort=False)
    # 生の対照 (前走値 = 対象レース前に確定している公知情報 / master の過去行のみ)
    m["raw_career_runs"] = g.cumcount()
    zd = pd.to_numeric(m["前走日付"], errors="coerce")
    zts = pd.to_datetime((zd + 20000000).astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    m["raw_days_since"] = (m["date"] - zts).dt.days
    m["raw_layoff_flag"] = (m["raw_days_since"] > 180).astype(float)
    m["raw_short_flag"] = (m["raw_days_since"] <= 14).astype(float)
    m["raw_prev_fin"] = pd.to_numeric(m["前走確定着順"], errors="coerce")
    m["raw_prev_fin_q"] = m["raw_prev_fin"] / pd.to_numeric(m["前走出走頭数"], errors="coerce")
    m["raw_prev_margin"] = pd.to_numeric(m["前走着差タイム"], errors="coerce")
    m["finq"] = m["fin"] / pd.to_numeric(m["出走頭数"], errors="coerce")
    m["raw_last3_finq_mean"] = g["finq"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    # 直近180日の出走回数 (対象日より前)
    cnt = np.zeros(len(m))
    for _, idx in g.indices.items():
        d = m["date"].to_numpy()[idx].astype("datetime64[D]").astype(np.int64)
        lo = np.searchsorted(d, d - 180, side="left")
        cnt[idx] = np.arange(len(idx)) - lo
    m["raw_runs_180d"] = cnt
    # 既存 EWMA (exp_recency_ewm.py と同じ: 前走値を馬ごと走数ベースで EWMA, halflife=1.8走)
    for tag, src in EWM_SIGS:
        s = pd.to_numeric(m[src], errors="coerce")
        m[f"ewm_{tag}"] = s.groupby(m["hid"]).transform(lambda x: x.ewm(halflife=1.8, min_periods=1).mean())
        m[f"ewm_{tag}_vs_prev"] = s - m[f"ewm_{tag}"]
    keep = ["rid16", "ban"] + [c for c in m.columns if c.startswith(("raw_", "ewm_"))]
    return m[keep]


def main() -> None:
    OUTD.mkdir(exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=BASE, capture_output=True, text=True).stdout.strip()
    df = load_runs()
    grid = SPEC["hyperparameter_selection"]["grid"]
    rows = []
    for hl, ta, qs, b in itertools.product(grid["halflife_days"], grid["tau2_a"], grid["qs"], grid["beta"]):
        _, ll, _ = run(df, Hyper(hl, ta, qs, b), collect=False)
        m = (ll.date.dt.year >= 2016) & (ll.date.dt.year <= 2021)
        rows.append({"halflife_days": hl, "tau2_a": ta, "qs": qs, "beta": b,
                     "ll_win_train": float(ll.loc[m, "ll_win"].mean()), "selectable": True})
        print(f"  HL={hl:<4} τa²={ta:<7} qs={qs:<4} β={b:<7} ll={rows[-1]['ll_win_train']:.5f}", flush=True)
    for ta, b in itertools.product(grid["tau2_a"], grid["beta"]):
        _, ll, _ = run(df, Hyper(90, ta, 1e-3, b), collect=False)
        m = (ll.date.dt.year >= 2016) & (ll.date.dt.year <= 2021)
        rows.append({"halflife_days": 90, "tau2_a": ta, "qs": 1e-3, "beta": b,
                     "ll_win_train": float(ll.loc[m, "ll_win"].mean()), "selectable": False})
        print(f"  [参考: 状態なし] τa²={ta:<7} β={b:<7} ll={rows[-1]['ll_win_train']:.5f}", flush=True)
    sel = pd.DataFrame(rows)
    sel.to_csv(OUTD / "hyper_selection.csv", index=False, encoding="utf-8-sig")
    best = sel[sel.selectable].sort_values("ll_win_train", ascending=False).iloc[0]
    hp = Hyper(float(best.halflife_days), float(best.tau2_a), float(best.qs), float(best.beta))
    print(f"選択: {hp}")

    f, ll, iv = run(df, hp, collect=True)
    # innovation 定義の選択 (train の持続性のみ)
    iv = iv.sort_values(["hid", "date"])
    for c in ["z", "zq"]:
        iv[f"next_{c}"] = iv.groupby("hid")[c].shift(-1)
    iv["gap"] = (iv.groupby("hid")["date"].shift(-1) - iv["date"]).dt.days
    t = iv[(iv.date.dt.year >= 2016) & (iv.date.dt.year <= 2021) & (iv.gap <= 60)]
    pers = {c: float(spearmanr(t[c], t[f"next_{c}"], nan_policy="omit").correlation) for c in ["z", "zq"]}
    choice = "A" if pers["z"] >= pers["zq"] else "B"
    print(f"innovation 持続性 (60日以内の連続2走の順位相関): A={pers['z']:.4f} B={pers['zq']:.4f} → {choice}")
    # 採用した定義で「前回 innovation」と「innovation_ewma」を記録から作り直す (A・B 共通の経路)
    lz_a, ew_a = ewma_from_log(iv, f, "z", hp.halflife_days)
    agree = np.allclose(ew_a, f["innovation_ewma"].to_numpy(float), equal_nan=True, atol=1e-9)
    print(f"  記録からの再計算がエンジンの値と一致 (A): {agree}")
    assert agree, "innovation_ewma の再計算がエンジンと一致しない"
    lz, ew = ewma_from_log(iv, f, "z" if choice == "A" else "zq", hp.halflife_days)
    f["last_performance_innovation"], f["innovation_ewma"] = lz, ew
    f = f.drop(columns=["last_performance_innovation_rankq"])
    (OUTD / "innovation_choice.json").write_text(json.dumps(
        {"persistence_spearman": pers, "choice": choice, "n_pairs": int(len(t))}, ensure_ascii=False, indent=1),
        encoding="utf-8")

    raw = raw_and_ewm()
    f["rid16"] = f["rid16"].astype(str)
    out = f.merge(raw, on=["rid16", "ban"], how="left")
    FEAT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(FEAT, index=False)
    (OUTD / "build_meta.json").write_text(json.dumps(
        {"commit": commit, "selected": hp.__dict__, "n_rows": int(len(out))}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"saved {len(out):,} rows -> {FEAT}")


if __name__ == "__main__":
    main()
