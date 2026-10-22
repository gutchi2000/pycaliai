# -*- coding: utf-8 -*-
"""
旧文脈マネー残差 Stage 0 (label-free)
=====================================
閉鎖基準は docs/research/SCRATCH_MONEY_STAGE0_CRITERIA_20261022.md に計算前に固定済み
(sha256 は criteria_sha256_before_compute.txt)。この script は TANPUK だけを読む。
master / kekka / 着順 / 払戻は一切読まない。

実行 (root から): venv311\\Scripts\\python.exe analysis/scratch_money_stage0/stage0.py
出力: analysis/scratch_money_stage0/out/{stage0_result.json, races.csv}
"""
from __future__ import annotations

import glob
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
BASE = Path(__file__).resolve().parents[2]
ODIR = BASE / "data" / "Time _series_odds"
OUT = Path(__file__).resolve().parent / "out"
CRIT = BASE / "docs" / "research" / "SCRATCH_MONEY_STAGE0_CRITERIA_20261022.md"
FLOOR = 0.005
NREQ_CONST = 15.68
SEED = 20261022
NBOOT = 2000
BANDS = [(5, 8), (9, 12), (13, 15), (16, 18)]


def band_of(n: int) -> int:
    for i, (lo, hi) in enumerate(BANDS):
        if lo <= n <= hi:
            return i
    return -1


def load() -> pd.DataFrame:
    fs = [f for f in sorted(glob.glob(str(ODIR / "TANPUK_20[12]*.csv"))) if "2011" not in f]
    d = pd.concat([pd.read_csv(f, encoding="cp932", dtype=str) for f in fs], ignore_index=True)
    d = d.drop_duplicates()
    c = list(d.columns)
    RID, KB, TM, N, TV = c[0], c[1], c[2], c[3], c[4]
    tan = sorted([x for x in c if re.match(r"^\d+単$", x)], key=lambda x: int(x[:-1]))
    out = pd.DataFrame({
        "rid": d[RID].str.strip(),
        "kb": pd.to_numeric(d[KB].str.strip()).astype(int),
        "tm": pd.to_numeric(d[TM].str.strip()).astype(np.int64),
        "n": pd.to_numeric(d[N].str.strip()).astype(int),
        "votes": pd.to_numeric(d[TV].str.strip(), errors="coerce"),
    })
    O = d[tan].apply(lambda s: pd.to_numeric(s.str.strip(), errors="coerce")).to_numpy(float)
    out["odds"] = list(np.nan_to_num(O, nan=0.0))
    out["year"] = out.rid.str[:4].astype(int)
    out = out[(out.year >= 2016) & (out.year <= 2025)]
    out["same"] = (out.tm // 10000) == out.rid.str[4:8].astype(int)
    return out.reset_index(drop=True)


def kl(p: np.ndarray, q: np.ndarray) -> float:
    m = p > 0
    return float(np.sum(p[m] * np.log(p[m] / q[m])))


def norm_inv(o: np.ndarray, mask: np.ndarray) -> np.ndarray:
    x = np.where(mask, 1.0 / np.where(o > 0, o, 1.0), 0.0)
    return x / x.sum()


def build(d: pd.DataFrame):
    races, horses = [], []
    n_anom = 0
    for rid, g in d.groupby("rid", sort=False):
        g = g.sort_values("tm")
        k1 = g[g.kb == 1]
        k4 = g[g.kb == 4]
        if len(k4) != 1 or len(k1) == 0:
            continue
        snaps = []
        prev = k1[~k1.same]
        same = k1[k1.same]
        if len(prev):
            snaps.append(("E", prev.iloc[-1]))
        if len(same):
            snaps.append(("M", same.iloc[0]))
            if len(same) >= 2:
                snaps.append(("P", same.iloc[-1]))
        snaps.append(("F", k4.iloc[0]))
        labels = [s[0] for s in snaps]
        O = np.vstack([s[1].odds for s in snaps])          # (S, 18)
        V = np.array([s[1].votes for s in snaps], float)
        nreg = int(k4.iloc[0].n)
        pres = O[:, :nreg] > 0
        presF = pres[-1]
        scr_int = {}          # interval index -> list of ban
        presale = 0
        anomaly = False
        for b in range(nreg):
            col = pres[:, b]
            if not col.any():
                presale += 1
                continue
            # monotone: present ... present, absent ... absent ?
            first_abs = np.argmin(col) if not col.all() else None
            if col.all():
                continue
            if not col[0]:
                anomaly = True        # 最初は不在で後に在
                horses.append(dict(rid=rid, ban=b + 1, kind="anomaly_late_appear"))
                continue
            if col[first_abs:].any():
                anomaly = True        # 不在→在の復活
                horses.append(dict(rid=rid, ban=b + 1, kind="anomaly_reappear"))
                continue
            k = first_abs - 1         # 区間 k: snap k -> snap k+1
            scr_int.setdefault(k, []).append(b + 1)
        if anomaly:
            n_anom += 1
        year = int(rid[:4])
        nF = int(presF.sum())
        base = dict(rid=rid, date=rid[:8], year=year, nreg=nreg, nF=nF, band=band_of(nF),
                    presale=presale, anomaly=anomaly, snaps="".join(labels),
                    vF=V[-1])
        for i, lab in enumerate(labels[:-1]):
            base[f"v{lab}"] = V[i]
        piF = norm_inv(O[-1, :nreg], presF)
        if anomaly:
            base["kind"] = "anomaly"
            races.append(base)
            continue
        if len(scr_int) == 0:
            base["kind"] = "control"
            for i, lab in enumerate(labels[:-1]):
                a = norm_inv(O[i, :nreg], presF)
                base[f"Dnat_{lab}"] = kl(piF, a)
            races.append(base)
            continue
        for k, bans in scr_int.items():
            A, B = labels[k], labels[k + 1]
            for b in bans:
                piA_full = norm_inv(O[k, :nreg], pres[k])
                horses.append(dict(rid=rid, ban=b, kind="scratch", interval=f"{A}->{B}",
                                   piA=float(piA_full[b - 1]), sA=V[k] / V[-1],
                                   sB=V[k + 1] / V[-1], year=year))
        if len(scr_int) > 1:
            base["kind"] = "scratch_multi_interval"
            base["intervals"] = ",".join(f"{labels[k]}->{labels[k+1]}" for k in sorted(scr_int))
            races.append(base)
            continue
        k = next(iter(scr_int))
        A, B = labels[k], labels[k + 1]
        a = norm_inv(O[k, :nreg], presF)              # IIA: A を F 残存馬へ比例再配分
        piA_full = norm_inv(O[k, :nreg], pres[k])
        base.update(kind="scratch", interval=f"{A}->{B}", A=A, B=B,
                    n_scr=len(scr_int[k]),
                    scr_piA=float(sum(piA_full[b - 1] for b in scr_int[k])),
                    sA=V[k] / V[-1], sB=V[k + 1] / V[-1],
                    D_scr=kl(piF, a))
        races.append(base)
    return pd.DataFrame(races), pd.DataFrame(horses)


def wquant(x, w, q):
    o = np.argsort(x)
    x, w = x[o], w[o]
    cw = np.cumsum(w) / w.sum()
    return float(x[np.searchsorted(cw, q)])


def qs(x):
    x = np.asarray(x, float)
    return {k: float(np.quantile(x, v)) for k, v in
            [("p05", .05), ("p25", .25), ("median", .5), ("p75", .75), ("p95", .95)]} | \
           {"mean": float(x.mean()), "n": int(len(x))}


def main():
    sha_before = (Path(__file__).resolve().parent / "criteria_sha256_before_compute.txt").read_text().split()[0]
    sha_now = hashlib.sha256(CRIT.read_bytes()).hexdigest()
    assert sha_now == sha_before, "閉鎖基準ファイルが計算前から変更されている"
    d = load()
    R, H = build(d)
    OUT.mkdir(parents=True, exist_ok=True)
    R.to_csv(OUT / "races.csv", index=False, encoding="utf-8")
    H.to_csv(OUT / "horses.csv", index=False, encoding="utf-8")

    res = {"criteria_file": str(CRIT.relative_to(BASE)), "criteria_sha256": sha_now,
           "period": "2016-01-01..2025-12-31", "inputs": "TANPUK only (no master/kekka/labels)"}
    res["races_total"] = int(len(R))
    res["race_kind_counts"] = R.kind.value_counts().to_dict()
    res["snap_patterns"] = R.snaps.value_counts().to_dict()
    hs = H[H.kind == "scratch"]
    res["horses_presale_scratch"] = int(R.presale.sum())
    res["horses_anomaly"] = H[H.kind != "scratch"].kind.value_counts().to_dict()
    # 2. 区間別件数 / s 分布
    cnt = {}
    for itv, g in hs.groupby("interval"):
        cnt[itv] = {"horses": int(len(g)), "races": int(g.rid.nunique()),
                    "by_year_horses": g.groupby("year").size().to_dict(),
                    "s_A": qs(g.sA), "s_B": qs(g.sB), "scratched_piA": qs(g.piA)}
    res["scratch_by_interval"] = cnt
    ctl = R[R.kind == "control"]
    res["pool_share_all_control_races"] = {
        lab: qs((ctl[f"v{lab}"] / ctl.vF).dropna()) for lab in ["E", "M", "P"]}

    # 3. 乖離 vs 自然ドリフト
    S = R[(R.kind == "scratch") & (R.band >= 0)].copy()
    ctl = ctl[ctl.band >= 0]
    # 非取消の層別 D_nat (long)
    nat = []
    for lab in ["E", "M", "P"]:
        x = ctl[["rid", "date", "year", "band", f"Dnat_{lab}"]].dropna().rename(columns={f"Dnat_{lab}": "D"})
        x["A"] = lab
        nat.append(x)
    nat = pd.concat(nat, ignore_index=True)
    key = ["A", "band", "year"]
    nat_st = nat.groupby(key).D.agg(["mean", "median", "size"]).rename(
        columns={"mean": "m_nat", "median": "med_nat", "size": "n_nat"})
    S = S.merge(nat_st, left_on=key, right_index=True, how="left")
    res["scratch_races_without_control_stratum"] = int(S.m_nat.isna().sum())
    S = S.dropna(subset=["m_nat"])
    S["excess"] = S.D_scr - S.m_nat
    S["K_r"] = S.excess / (1.0 - S.sB.clip(upper=0.999999))
    S["early"] = S.A.isin(["E", "M"])
    S.loc[~S.early, "K_r"] = np.nan   # 遅い取消は s_B=1 で K 未定義 (K は早い取消だけから推定)
    S.to_csv(OUT / "scratch_races_main.csv", index=False, encoding="utf-8")

    def matched_nat(sub):
        w_st = sub.groupby(key).size().rename("n_scr")
        x = nat.merge(w_st, left_on=key, right_index=True, how="inner")
        x = x.merge(nat_st[["n_nat"]], left_on=key, right_index=True)
        w = (x.n_scr / x.n_nat).to_numpy()
        return x.D.to_numpy(), w

    def block(sub, name):
        Dn, w = matched_nat(sub)
        med_nat = wquant(Dn, w, 0.5)
        return {"subset": name, "n_races": int(len(sub)),
                "D_scr": qs(sub.D_scr), "D_nat_matched_median": med_nat,
                "D_nat_matched_mean": float(np.average(Dn, weights=w)),
                "rho_median_ratio": float(sub.D_scr.median() / med_nat),
                "excess_mean": float(sub.excess.mean()),
                "excess_median": float(sub.excess.median()),
                "K_mean": float(sub.K_r.mean()) if sub.K_r.notna().any() else None,
                "s_B": qs(sub.sB), "s_A": qs(sub.sA)}

    E = S[S.early]
    L = S[~S.early]
    res["divergence"] = {
        "early(E->M,M->P)": block(E, "early"),
        "E->M": block(S[S.interval == "E->M"], "E->M"),
        "M->P": block(S[S.interval == "M->P"], "M->P"),
        "late(P->F)": block(L, "late"),
    }
    # 感度 (判定に使わない): 取消馬の pre 確率三分位
    sens = {}
    if len(E) > 30:
        t = np.quantile(E.scr_piA, [1 / 3, 2 / 3])
        for nm, m in [("low", E.scr_piA <= t[0]), ("mid", (E.scr_piA > t[0]) & (E.scr_piA <= t[1])),
                      ("high", E.scr_piA > t[1])]:
            sens[f"early_scrpiA_{nm}"] = block(E[m], nm) | {"scr_piA_range": qs(E[m].scr_piA)}
        top = E[E.scr_piA >= 0.2]
        if len(top) >= 5:
            sens["early_scrpiA_ge0.2"] = block(top, "piA>=0.2")
    res["sensitivity_not_for_decision"] = sens

    # K_hi: 開催日クラスタ bootstrap (取消・非取消とも日付で再抽出)
    rng = np.random.default_rng(SEED)
    dates = np.array(sorted(set(E.date) | set(nat[nat.A.isin(["E", "M"])].date)))
    di = {x: i for i, x in enumerate(dates)}
    st_keys = pd.MultiIndex.from_frame(E[key]).unique()
    sk = {k: i for i, k in enumerate(st_keys)}
    natE = nat[nat.set_index(key).index.isin(st_keys)]
    n_d = natE.date.map(di).to_numpy()
    n_s = np.array([sk[k] for k in zip(natE.A, natE.band, natE.year)])
    n_D = natE.D.to_numpy()
    e_d = E.date.map(di).to_numpy()
    e_s = np.array([sk[k] for k in zip(E.A, E.band, E.year)])
    e_D = E.D_scr.to_numpy()
    e_inv = 1.0 / (1.0 - E.sB.to_numpy())
    boots = []
    for _ in range(NBOOT):
        m = np.bincount(rng.integers(0, len(dates), len(dates)), minlength=len(dates)).astype(float)
        wn = m[n_d]
        sumD = np.bincount(n_s, weights=wn * n_D, minlength=len(st_keys))
        sumW = np.bincount(n_s, weights=wn, minlength=len(st_keys))
        with np.errstate(invalid="ignore", divide="ignore"):
            mn = sumD / sumW
        we = m[e_d]
        ok = np.isfinite(mn[e_s]) & (we > 0)
        Kr = (e_D - mn[e_s]) * e_inv
        boots.append(np.sum((we * Kr)[ok]) / np.sum(we[ok]))
    boots = np.array(boots)
    K = float(E.K_r.mean())
    K_hi = float(np.quantile(boots, 0.975))
    K_lo = float(np.quantile(boots, 0.025))

    # 上界
    S["D_ub"] = np.where(S.early, S.sB * K_hi, K_hi)
    D_ub_bar = float(S.D_ub.mean())
    n_all_races = int(len(R[R.kind != "anomaly"]))
    n_avail = int(len(S))
    n_avail_1623 = int((S.year <= 2023).sum())
    nreq_floor = NREQ_CONST / FLOOR
    nreq_ub = NREQ_CONST / D_ub_bar if D_ub_bar > 0 else float("inf")
    rho_early = res["divergence"]["early(E->M,M->P)"]["rho_median_ratio"]
    crit = {
        "C1_rho_lt_2": {"value": rho_early, "closes": rho_early < 2.0},
        "C2_Khi_lt_floor": {"value": K_hi, "closes": K_hi < FLOOR},
        "C3a_nreq_floor_gt_navail": {"n_req": nreq_floor, "n_avail": n_avail, "closes": nreq_floor > n_avail},
        "C3b_nreq_ub_gt_navail": {"n_req": nreq_ub, "n_avail": n_avail, "closes": nreq_ub > n_avail},
    }
    closed = any(v["closes"] for v in crit.values())
    res["bound"] = {
        "K_point": K, "K_boot_CI95": [K_lo, K_hi], "nboot": NBOOT, "seed": SEED,
        "n_early_races": int(len(E)), "n_late_races": int(len(L)),
        "D_ub_bar_per_scratch_race": D_ub_bar,
        "D_ub_early_mean": float(S[S.early].D_ub.mean()) if S.early.any() else None,
        "D_ub_late_mean": K_hi,
        "scratch_race_share_of_all": n_avail / n_all_races,
        "D_ub_bar_all_races": D_ub_bar * n_avail / n_all_races,
        "n_all_races": n_all_races, "n_avail_2016_2025": n_avail, "n_avail_2016_2023": n_avail_1623,
        "floor": FLOOR, "n_req_floor": nreq_floor, "n_req_at_D_ub_bar": nreq_ub,
    }
    res["criteria"] = crit
    res["decision"] = "CLOSE" if closed else "STAGE1_WORTH_REVIEW"
    (OUT / "stage0_result.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float),
                                            encoding="utf-8")
    print(json.dumps({k: res[k] for k in ["race_kind_counts", "bound", "criteria", "decision"]},
                     ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
