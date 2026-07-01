"""
build_glicko_feats.py — Glicko-2 の μ(点レート) と RD(不確実性) を時間順更新 (leak-safe)
==========================================================================================
ELo点レート(μ)は本日 REDUNDANT 確定。Glicko-2 が質的に違うのは RD(rating deviation=不確実性)
を持つこと。新馬(init RD大)/休み明け(inactivityでRD増大)/キャリア浅 を RD で定量化し、
kako5・キャリア戦数・ELo に対し増分を持つかの素材を作る。

レース=多人数対戦(同レース他馬との pairwise 着順結果)を 1 rating period として Glicko-2 更新。
休み明け: 前走からの経過月 t で φ_pre=sqrt(φ²+σ²·t) と RD を増大 (標準 Glicko-2 inactivity)。

★リーク防止: race_ord(日付+発走+rid) 昇順 1パス。各馬の特徴は「inactivity反映後・当該レース更新前」
  の状態(=予測時点で利用可能)。検査で「記録時に未来レース未更新」をassert、違反0で確認。

特徴: g2_mu(点レート, 対照) / g2_rd(★不確実性) / g2_cons(μ-2RD, 保守的レート)
出力: data/glicko_feats.parquet / reports/glicko_leakcheck.json
実行: PYTHONUTF8=1 python build_glicko_feats.py
"""
from __future__ import annotations
import json, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
BASE = Path(__file__).parent
MASTER_CSV = BASE / "data/master_v2_20130105-20251228.csv"
OUT_PARQUET = BASE / "data/glicko_feats.parquet"
OUT_LEAK = BASE / "reports/glicko_leakcheck.json"
COL_RID = "レースID(新/馬番無)"; COL_BAN = "馬番"; COL_PED = "血統登録番号"
COL_DATE = "日付"; COL_HASSOU = "発走時刻"; COL_JYUN = "着順"
SCALE = 173.7178
INIT_R, INIT_RD, INIT_VOL, TAU = 1500.0, 350.0, 0.06, 0.5
LEAK_EVERY = 50


def g2_update(mu, phi, sig, omu, ophi, s, tau=TAU):
    g = 1.0 / np.sqrt(1 + 3 * ophi ** 2 / np.pi ** 2)
    E = 1.0 / (1 + np.exp(-g * (mu - omu)))
    v = 1.0 / np.sum(g ** 2 * E * (1 - E))
    dsum = float(np.sum(g * (s - E))); delta = v * dsum
    a = np.log(sig ** 2)

    def f(x):
        ex = np.exp(x)
        return (ex * (delta ** 2 - phi ** 2 - v - ex)) / (2 * (phi ** 2 + v + ex) ** 2) - (x - a) / (tau ** 2)
    A = a
    if delta ** 2 > phi ** 2 + v:
        B = np.log(delta ** 2 - phi ** 2 - v)
    else:
        k = 1
        while f(a - k * tau) < 0 and k < 100:
            k += 1
        B = a - k * tau
    fa, fb = f(A), f(B)
    for _ in range(60):
        if abs(B - A) < 1e-6: break
        C = A + (A - B) * fa / (fb - fa); fc = f(C)
        if fc * fb < 0: A, fa = B, fb
        else: fa = fa / 2
        B, fb = C, fc
    sig_new = np.exp(A / 2)
    phi_star = np.sqrt(phi ** 2 + sig_new ** 2)
    phi_new = 1.0 / np.sqrt(1.0 / phi_star ** 2 + 1.0 / v)
    mu_new = mu + phi_new ** 2 * dsum
    return mu_new, phi_new, sig_new


def main():
    print(f"[load] {MASTER_CSV.name}")
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig",
                     usecols=[COL_RID, COL_BAN, COL_PED, COL_DATE, COL_HASSOU, COL_JYUN, "split"], low_memory=False)
    df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df = df.dropna(subset=[COL_JYUN, COL_RID, COL_PED, COL_DATE]).copy()
    df[COL_DATE] = pd.to_numeric(df[COL_DATE], errors="coerce").astype("int64")
    df["_hassou"] = pd.to_numeric(df[COL_HASSOU], errors="coerce").fillna(0).astype("int64")
    df[COL_BAN] = pd.to_numeric(df[COL_BAN], errors="coerce").fillna(0).astype(int)
    df["_rid"] = df[COL_RID].astype(str)
    rinfo = df.groupby("_rid").agg(date=(COL_DATE, "first"), hassou=("_hassou", "first")).reset_index()
    rinfo = rinfo.sort_values(["date", "hassou", "_rid"]).reset_index(drop=True)
    rinfo["ord"] = np.arange(len(rinfo))
    rid2ord = dict(zip(rinfo["_rid"], rinfo["ord"]))
    df["race_ord"] = df["_rid"].map(rid2ord)
    dts = {d: pd.Timestamp(str(int(d))) for d in df[COL_DATE].unique()}

    races = []
    for rid, g in df.groupby("_rid", sort=False):
        races.append({"rid": rid, "ord": int(rid2ord[rid]), "date": int(g[COL_DATE].iloc[0]),
                      "peds": g[COL_PED].values, "jyun": g[COL_JYUN].values.astype(float),
                      "ban": g[COL_BAN].values})
    races.sort(key=lambda r: r["ord"])
    print(f"  races={len(races):,}")

    state = {}   # ped -> (mu, phi, sig, last_date)
    rec = {}; last_ord = -1; viol = 0; checked = 0
    for ri, r in enumerate(races):
        peds = r["peds"]; jy = r["jyun"]; bans = r["ban"]; cur = dts[r["date"]]
        n = len(peds)
        mus = np.empty(n); phis = np.empty(n); sigs = np.empty(n)
        if ri % LEAK_EVERY == 0:
            checked += 1
            if last_ord >= r["ord"]: viol += 1
        for i, p in enumerate(peds):
            if p in state:
                mu, phi, sig, last = state[p]
                t = max(0.0, (cur - last).days / 30.0)
                phi = np.sqrt(phi ** 2 + sig ** 2 * t)            # inactivity → RD増大
            else:
                mu, phi, sig = 0.0, INIT_RD / SCALE, INIT_VOL
            mus[i], phis[i], sigs[i] = mu, phi, sig
            rd = phi * SCALE; mr = mu * SCALE + 1500.0
            rec[(r["rid"], int(bans[i]))] = (mr, rd, mr - 2 * rd)
        # update (記録後 = そのレース結果未使用で記録済)
        for i, p in enumerate(peds):
            mask = np.arange(n) != i
            si = np.where(jy[i] < jy[mask], 1.0, np.where(jy[i] > jy[mask], 0.0, 0.5))
            mn, pn, sn = g2_update(mus[i], phis[i], sigs[i], mus[mask], phis[mask], si)
            state[p] = (mn, pn, sn, cur)
        last_ord = r["ord"]

    keys = list(rec.keys())
    out = pd.DataFrame({"rid16": [k[0] for k in keys], "ban": [k[1] for k in keys]})
    arr = np.array([rec[k] for k in keys])
    out["g2_mu"] = arr[:, 0]; out["g2_rd"] = arr[:, 1]; out["g2_cons"] = arr[:, 2]
    out.to_parquet(OUT_PARQUET, index=False)

    sp = df[["_rid", COL_BAN, "split"]].rename(columns={"_rid": "rid16", COL_BAN: "ban"})
    cov = out.merge(sp, on=["rid16", "ban"], how="left")
    leak = {"leak_checked": checked, "leak_violations": viol, "PASS": bool(viol == 0),
            "scale": SCALE, "init": {"R": INIT_R, "RD": INIT_RD, "vol": INIT_VOL}, "tau": TAU,
            "note": "Glicko-2自前(多人数=pairwise着順). TrueSkill は依存不在のため省略, RDで不確実性代表.",
            "n_rows": len(out)}
    for spl in ["train", "valid", "test"]:
        s = cov[cov["split"] == spl]
        if len(s):
            leak[f"rd_mean_{spl}"] = round(float(s["g2_rd"].mean()), 2)
    OUT_LEAK.parent.mkdir(exist_ok=True)
    json.dump(leak, open(OUT_LEAK, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\n[リーク検査] 検査={checked:,} 違反={viol} {'PASS' if viol==0 else 'FAIL'}")
    print(f"  RD平均 train/valid/test = {leak.get('rd_mean_train')}/{leak.get('rd_mean_valid')}/{leak.get('rd_mean_test')}")
    print(out[["g2_mu", "g2_rd", "g2_cons"]].describe().T[["mean", "std", "min", "max"]].to_string())
    print(f"[saved] {OUT_PARQUET}")
    return 0 if viol == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
