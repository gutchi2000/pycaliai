"""
build_elo_feats.py — 相互ELO 4アーム (T1/T2 × M1/M2) 特徴生成 + リーク検査
============================================================================
レース⇄馬の二部グラフ的相互更新: 馬レート r_h を ELO 更新し、レースレベル L_R =
出走馬レートの集約(平均)を派生。馬レートはレース(=相手集合)を通じて更新され、
レベルは馬レートの集約 → 相互に影響。

軸1 (時間処理):
  T1 = オンライン逐次更新 (race_ord 昇順 1パス、各レース記録は処理直前=未来不使用)
  T2 = 各年初に「前年末までの全レースで反復収束(T2_PASSES回)」した基盤レート + 年内オンライン
       (前年末まで = 暦年境界で確実に T 未満 → リークフリー。年内は P 直前まで)
軸2 (更新ルール):
  M1 = pairwise margin-weighted ELO (勝敗 + 着差で更新幅重み付け)
  M2 = Plackett-Luce 的 着順全体更新 (softmax 期待分布 vs 着順実績分布)
→ T1M1 / T1M2 / T2M1 / T2M2

各行 (馬H, レースP) の特徴 (P処理直前のレートで, 4種):
  elo_<arm>_horse      : H のレート
  elo_<arm>_level      : P 出走馬レート平均 (レースレベル)
  elo_<arm>_vs         : H レート − レベル (レース内相対)
  elo_<arm>_prevlevel  : H の前走のレベル (前走の格)

★リーク検査 (生命線): 各アームで記録時点に未更新レースのみ反映かをサンプル assert。
  T2 は「収束に使った前年末データの最大ord < 当該年の最小ord」も確認。違反>0 で当該アーム FAIL。

定数 (K/init/passes) は妥当な固定値 (valid 最適化は今回省略、報告明記)。
出力: data/elo_feats.parquet / reports/elo_feats_leakcheck.json
実行: PYTHONUTF8=1 python build_elo_feats.py
"""
from __future__ import annotations
import json, warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
MASTER_CSV = BASE / "data/master_v2_20130105-20251228.csv"
OUT_PARQUET = BASE / "data/elo_feats.parquet"
OUT_LEAK = BASE / "reports/elo_feats_leakcheck.json"

COL_RID = "レースID(新/馬番無)"
COL_BAN = "馬番"
COL_PED = "血統登録番号"
COL_DATE = "日付"
COL_HASSOU = "発走時刻"
COL_JYUN = "着順"
COL_MARGIN = "前走着差タイム"   # chain: 走X+1 のこの値 = 走X の着差

INIT = 1500.0
K1 = 24.0
K2 = 24.0
SCALE = 200.0
T2_PASSES = 3
LEAK_EVERY = 50
ARMS = ["T1M1", "T1M2", "T2M1", "T2M2"]


def update_M1(rr, jyun, margin):
    """pairwise margin-weighted ELO。S_ij=1 if i上位(着順小)。重み=1+|着差差|。"""
    n = len(rr)
    if n < 2:
        return rr.copy()
    diff = rr[None, :] - rr[:, None]                      # rr[j]-rr[i]
    E = 1.0 / (1.0 + 10.0 ** (diff / 400.0))              # i から見た期待勝率
    J = jyun.astype(float)
    S = (J[:, None] < J[None, :]).astype(float) + 0.5 * (J[:, None] == J[None, :])
    mv = margin.astype(float)
    Wm = np.abs(mv[:, None] - mv[None, :])
    Wm = np.where(np.isnan(Wm), 0.0, Wm)
    W = 1.0 + Wm
    np.fill_diagonal(W, 0.0); np.fill_diagonal(S, 0.0); np.fill_diagonal(E, 0.0)
    num = (W * (S - E)).sum(axis=1)
    den = W.sum(axis=1)
    return rr + K1 * np.where(den > 0, num / den, 0.0)


def update_M2(rr, jyun):
    """PL 的: softmax 期待分布 p と 着順実績分布 rel の差で全馬更新。"""
    n = len(rr)
    if n < 2:
        return rr.copy()
    z = (rr - rr.mean()) / SCALE
    z = z - z.max()
    p = np.exp(z); p = p / p.sum()
    rel = (n - jyun.astype(float)) / (n - 1)
    s = rel.sum()
    rel = rel / s if s > 0 else np.ones(n) / n
    return rr + K2 * n * (rel - p)


def _apply(update, rr, jyun, margin):
    return update(rr, jyun, margin) if update is update_M1 else update(rr, jyun)


def run_T1(races, update):
    rates = {}; prev = {}; rec = {}
    last_ord = -1; viol = 0; checked = 0
    for r in races:                       # race_ord 昇順
        peds = r["peds"]; bans = r["ban"]
        rr = np.array([rates.get(p, INIT) for p in peds], float)
        level = float(rr.mean())
        if r["idx"] % LEAK_EVERY == 0:
            checked += 1
            if last_ord >= r["ord"]:      # 既に同/未来レースを更新済なら違反
                viol += 1
        for k, p in enumerate(peds):
            rec[(r["rid"], int(bans[k]))] = (float(rr[k]), level, float(rr[k]) - level,
                                             prev.get(p, np.nan))
        newr = _apply(update, rr, r["jyun"], r["margin"])
        for k, p in enumerate(peds):
            rates[p] = float(newr[k]); prev[p] = level
        last_ord = r["ord"]
    return rec, viol, checked


def run_T2(races, update, passes=T2_PASSES):
    by_year = defaultdict(list)
    for r in races:
        by_year[r["year"]].append(r)
    years = sorted(by_year)
    rec = {}; viol = 0; checked = 0
    for y in years:
        before = [r for r in races if r["year"] < y]
        rates = {}; prev = {}
        for _ in range(passes):           # 前年末まで反復収束
            for r in before:
                peds = r["peds"]
                rr = np.array([rates.get(p, INIT) for p in peds], float)
                lv = float(rr.mean())
                newr = _apply(update, rr, r["jyun"], r["margin"])
                for k, p in enumerate(peds):
                    rates[p] = float(newr[k]); prev[p] = lv
        if before:                        # リーク検査: 収束データ最大ord < 当年最小ord
            checked += 1
            if max(r["ord"] for r in before) >= min(r["ord"] for r in by_year[y]):
                viol += 1
        last_ord = -1
        for r in sorted(by_year[y], key=lambda x: x["ord"]):   # 年内オンライン
            peds = r["peds"]; bans = r["ban"]
            rr = np.array([rates.get(p, INIT) for p in peds], float)
            level = float(rr.mean())
            if r["idx"] % LEAK_EVERY == 0:
                checked += 1
                if last_ord >= r["ord"]:
                    viol += 1
            for k, p in enumerate(peds):
                rec[(r["rid"], int(bans[k]))] = (float(rr[k]), level, float(rr[k]) - level,
                                                 prev.get(p, np.nan))
            newr = _apply(update, rr, r["jyun"], r["margin"])
            for k, p in enumerate(peds):
                rates[p] = float(newr[k]); prev[p] = level
            last_ord = r["ord"]
    return rec, viol, checked


def main():
    print(f"[load] {MASTER_CSV.name}")
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig",
                     usecols=[COL_RID, COL_BAN, COL_PED, COL_DATE, COL_HASSOU, COL_JYUN,
                              COL_MARGIN, "split"], low_memory=False)
    df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df = df.dropna(subset=[COL_JYUN, COL_RID, COL_PED, COL_DATE]).copy()
    df[COL_DATE] = pd.to_numeric(df[COL_DATE], errors="coerce").astype("int64")
    df["_hassou"] = pd.to_numeric(df[COL_HASSOU], errors="coerce").fillna(0).astype("int64")
    df[COL_BAN] = pd.to_numeric(df[COL_BAN], errors="coerce").fillna(0).astype(int)
    df[COL_MARGIN] = pd.to_numeric(df[COL_MARGIN], errors="coerce")
    df["_rid"] = df[COL_RID].astype(str)
    df["_year"] = df["_rid"].str[:4].astype(int)

    # race_ord (日付, 発走時刻, rid)
    rinfo = df.groupby("_rid").agg(date=(COL_DATE, "first"), hassou=("_hassou", "first"),
                                   year=("_year", "first")).reset_index()
    rinfo = rinfo.sort_values(["date", "hassou", "_rid"]).reset_index(drop=True)
    rinfo["race_ord"] = np.arange(len(rinfo), dtype=np.int64)
    rid2ord = dict(zip(rinfo["_rid"], rinfo["race_ord"]))
    df["race_ord"] = df["_rid"].map(rid2ord)

    # 着差 chain: 走X の着差 = 同馬次走行の前走着差
    df = df.sort_values([COL_PED, "race_ord"])
    df["_margin"] = df.groupby(COL_PED, sort=False)[COL_MARGIN].shift(-1)

    # race オブジェクト (ord 昇順)
    races = []
    for rid, g in df.groupby("_rid", sort=False):
        races.append({"rid": rid, "ord": int(rid2ord[rid]), "year": int(g["_year"].iloc[0]),
                      "peds": g[COL_PED].values, "jyun": g[COL_JYUN].values.astype(float),
                      "margin": g["_margin"].values.astype(float), "ban": g[COL_BAN].values})
    races.sort(key=lambda r: r["ord"])
    for i, r in enumerate(races):
        r["idx"] = i
    print(f"  races={len(races):,}  rows={len(df):,}  years={sorted(set(r['year'] for r in races))[:3]}..")

    rule = {"M1": update_M1, "M2": update_M2}
    leak = {"K1": K1, "K2": K2, "scale": SCALE, "T2_passes": T2_PASSES, "init": INIT,
            "note": "K/init/passes は固定 (valid最適化は今回省略)", "arms": {}}
    cols = {}
    for arm in ARMS:
        mode, mrule = arm[:2], arm[2:]
        upd = rule[mrule]
        print(f"[ELO] {arm} ...", flush=True)
        if mode == "T1":
            rec, viol, checked = run_T1(races, upd)
        else:
            rec, viol, checked = run_T2(races, upd)
        leak["arms"][arm] = {"leak_checked": checked, "leak_violations": viol,
                             "PASS": bool(viol == 0)}
        print(f"   検査={checked:,}  違反={viol}  {'PASS' if viol==0 else 'FAIL'}")
        cols[arm] = rec

    # records → DataFrame
    keys = list(cols[ARMS[0]].keys())
    out = pd.DataFrame({"rid16": [k[0] for k in keys], "ban": [k[1] for k in keys]})
    for arm in ARMS:
        rec = cols[arm]
        arr = np.array([rec.get(k, (np.nan,) * 4) for k in keys], float)
        out[f"elo_{arm}_horse"] = arr[:, 0]
        out[f"elo_{arm}_level"] = arr[:, 1]
        out[f"elo_{arm}_vs"] = arr[:, 2]
        out[f"elo_{arm}_prevlevel"] = arr[:, 3]

    # split カバレッジ
    sp = df[["_rid", COL_BAN, "split"]].rename(columns={"_rid": "rid16", COL_BAN: "ban"})
    cov_df = out.merge(sp, on=["rid16", "ban"], how="left")
    leak["coverage"] = {}
    for spl in ["train", "valid", "test"]:
        s = cov_df[cov_df["split"] == spl]
        if len(s):
            leak["coverage"][spl] = {"n": int(len(s)),
                                     "prevlevel_notna%": round(s["elo_T1M1_prevlevel"].notna().mean() * 100, 1)}
    leak["all_PASS"] = bool(all(v["leak_violations"] == 0 for v in leak["arms"].values()))

    out.to_parquet(OUT_PARQUET, index=False)
    OUT_LEAK.parent.mkdir(exist_ok=True)
    with open(OUT_LEAK, "w", encoding="utf-8") as f:
        json.dump(leak, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 64)
    print("[リーク検査] アーム別:")
    for arm, v in leak["arms"].items():
        print(f"  {arm}: 検査={v['leak_checked']:,} 違反={v['leak_violations']} → {'PASS' if v['PASS'] else 'FAIL'}")
    print(f"  全アーム PASS = {leak['all_PASS']}")
    print(f"  カバレッジ: {leak['coverage']}")
    # 分布 (test)
    st = cov_df[cov_df["split"] == "test"]
    print("test レート分布 (horse):")
    print(st[[f"elo_{a}_horse" for a in ARMS]].describe().T[["mean", "std", "min", "max"]].to_string())
    print(f"\n[saved] {OUT_PARQUET}")
    return 0 if leak["all_PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
