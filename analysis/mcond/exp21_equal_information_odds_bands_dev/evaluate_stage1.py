# -*- coding: utf-8 -*-
"""
evaluate_stage1.py — EXP21 Stage 1 (spec v0.3-frozen): 単勝・複勝・馬連の帯 ROI、G1 / G2
==========================================================================================
v0.3-frozen の規則だけで判定する (結果を見て規則・帯・期間・券種を変えない)。v0.3-frozen commit 後にだけ実行する。
  期間    discovery 2013-2018 / evaluation 2019-2023。2024/2025 は読まない
  価格層  D0 = terminal で帯決定・terminal 決済 / D1 = historical_pre_snapshot (約 T−28) で帯決定・terminal 決済
  帯      primary = 等期待的中質量 10 帯 (主)。20 帯・等 ticket-count 10 帯・固定帯は副解析 (救済に使わない)
  基準    単勝・馬連: 各帯の label-free 較正 null = 帯内 ticket の 1/overround (race の terminal 価格だけ) の平均。
          法定控除率は参考値。複勝: 当該期間・価格層の全 ticket プール ROI
  払戻    公式払戻 (kekka master)。同着 race は券種ごとに除外して件数を報告 (単勝: 1 着同着、馬連: 1・2 着いずれかの同着、
          複勝: 払戻対象数が places(n) と違う race)。取消馬の ticket は元々 eligible 外 (返還)
  推論    年層化・暦日 cluster bootstrap、B = 10,000、seed 20260928 (同じ期間の全券種・全価格層で同じ再標本を使う)
  G1      単勝・馬連: discovery D0 と evaluation D0 の帯 uplift (ROI − null) の符号一致 >= 8/10 かつ帯 ROI の Spearman >= 0.70
          複勝: 帯 ROI の Spearman >= 0.70 が主判定。全 ticket プール ROI との差の符号一致 (>= 8/10) は副 (等級に使わない)
  G2      G1 PASS 券種だけ。discovery D0 で「uplift > 0 かつ bootstrap CI95 下限 > 0」の帯順位を一度だけ選び、その和集合 S を
          evaluation D1 で一度だけ検定する。evaluation D1 の uplift が discovery D0 の S の uplift の 50% 以上残り、
          かつ evaluation D1 uplift の CI95 下限 > 0 なら PASS。S が空なら NOT_APPLICABLE。ROI > 1.0 は別に報告
          (G2 PASS を利益 edge と解釈しない)
出力: out/stage1_results.json
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import bands as B
from . import loaders as L

TYPES = ["tansho", "fukusho", "umaren"]
PERIODS = {"discovery": "discovery_2013_2018", "evaluation": "evaluation_2019_2023"}
LAYERS = ["D0", "D1"]
BOOT_B = 10000
BOOT_SEED = 20260928
STATUTORY = {"tansho": 0.80, "fukusho": 0.80, "umaren": 0.775}
G1_MIN_SIGNS = 8
G1_MIN_RHO = 0.70
G2_RETAIN = 0.50


# ---------------------------------------------------------------- 判定 (pure)
def g1_decide(t: str, roi_disc, roi_eval, base_disc, base_eval) -> dict:
    roi_disc, roi_eval = np.asarray(roi_disc, float), np.asarray(roi_eval, float)
    up_d, up_e = roi_disc - np.asarray(base_disc, float), roi_eval - np.asarray(base_eval, float)
    signs = int(np.sum(np.sign(up_d) == np.sign(up_e)))
    rho = float(spearmanr(roi_disc, roi_eval).correlation)
    if t == "fukusho":
        return {"grade": "PASS" if rho >= G1_MIN_RHO else "FAIL", "spearman": rho, "sign_agreement_secondary": signs,
                "rule": "fukusho: Spearman >= 0.70 (primary); sign agreement vs pool ROI is secondary"}
    ok = signs >= G1_MIN_SIGNS and rho >= G1_MIN_RHO
    return {"grade": "PASS" if ok else "FAIL", "spearman": rho, "sign_agreement": signs,
            "rule": "sign agreement of uplift vs calibrated null >= 8/10 AND Spearman >= 0.70"}


def g2_select(up_disc, up_disc_ci_lower) -> list[int]:
    return [int(b) for b in range(len(up_disc)) if up_disc[b] > 0 and up_disc_ci_lower[b] > 0]


def g2_decide(sel, up_disc_S, up_eval_S, up_eval_S_ci_lower) -> dict:
    if not sel:
        return {"grade": "NOT_APPLICABLE", "reason": "no primary band preselected in discovery D0"}
    ok = (up_disc_S > 0 and up_eval_S >= G2_RETAIN * up_disc_S and up_eval_S_ci_lower > 0)
    return {"grade": "PASS" if ok else "FAIL", "selected_bands": sel, "uplift_discovery_D0": up_disc_S,
            "uplift_evaluation_D1": up_eval_S, "retention": (up_eval_S / up_disc_S) if up_disc_S else None,
            "uplift_evaluation_D1_ci95_lower": up_eval_S_ci_lower}


# ---------------------------------------------------------------- 払戻 (結果)
def payout_tables():
    k = L.load_kekka_master(20231231)
    k["jyun"] = pd.to_numeric(k["確定着順"], errors="coerce")
    for c in ("単勝配当", "複勝配当", "馬連"):
        k[c + "_n"] = pd.to_numeric(k[c].astype(str).str.replace(",", ""), errors="coerce")
    tbl = {}
    for rid, g in k.groupby("rid16"):
        first = g.loc[g.jyun == 1, "ban"].astype(int).tolist()
        second = g.loc[g.jyun == 2, "ban"].astype(int).tolist()
        tan = {int(b): float(p) for b, p in zip(g.loc[g.jyun == 1, "ban"], g.loc[g.jyun == 1, "単勝配当_n"]) if p == p}
        fuku = {int(b): float(p) for b, p in zip(g["ban"], g["複勝配当_n"]) if p == p}
        um = g["馬連_n"].dropna()
        tbl[rid] = {"first": first, "second": second, "tan": tan, "fuku": fuku,
                    "umaren": float(um.iloc[0]) if len(um) else np.nan}
    return tbl


def ticket_payouts(t, rid_list, d, tbl):
    """ticket ごとの払戻 (円 / 100 円)。除外 race の ticket は mask=False"""
    n = len(d["key"])
    pay = np.zeros(n)
    keep = np.ones(n, bool)
    reason = {"no_result": 0, "dead_heat_or_irregular": 0}
    race = d["race"]
    starts = np.flatnonzero(np.r_[True, race[1:] != race[:-1]])
    ends = np.r_[starts[1:], n]
    for s, e in zip(starts, ends):
        rid = str(rid_list[race[s]])
        r = tbl.get(rid)
        if r is None:
            keep[s:e] = False
            reason["no_result"] += 1
            continue
        a, b = d["a"][s:e], d["b"][s:e]
        if t == "tansho":
            if len(r["first"]) != 1 or r["first"][0] not in r["tan"]:
                keep[s:e] = False; reason["dead_heat_or_irregular"] += 1; continue
            pay[s:e] = np.where(a == r["first"][0], r["tan"][r["first"][0]], 0.0)
        elif t == "umaren":
            if len(r["first"]) != 1 or len(r["second"]) != 1 or not r["umaren"] == r["umaren"]:
                keep[s:e] = False; reason["dead_heat_or_irregular"] += 1; continue
            x, y = sorted((r["first"][0], r["second"][0]))
            pay[s:e] = np.where((a == x) & (b == y), r["umaren"], 0.0)
        else:
            k = B.places(e - s)
            if len(r["fuku"]) != k:
                keep[s:e] = False; reason["dead_heat_or_irregular"] += 1; continue
            pay[s:e] = np.array([r["fuku"].get(int(x), 0.0) for x in a])
    return pay / 100.0, keep, reason


# ---------------------------------------------------------------- bootstrap (年層化・暦日)
class DayBoot:
    def __init__(self, days, years, seed=BOOT_SEED, b=BOOT_B):
        self.udays = np.unique(days)
        years_of = self.udays // 10000
        rng = np.random.default_rng(seed)
        self.W = np.zeros((b, len(self.udays)))
        for y in np.unique(years_of):
            idx = np.flatnonzero(years_of == y)
            pick = rng.integers(0, len(idx), size=(b, len(idx)))
            rows = np.repeat(np.arange(b), len(idx))
            np.add.at(self.W, (rows, idx[pick.ravel()]), 1.0)

    def ratio_ci(self, days, num, den):
        di = np.searchsorted(self.udays, days)
        # 欠けた日を隣の日へ無言で対応させない (searchsorted は非一致でも位置を返す)
        assert np.array_equal(self.udays[np.minimum(di, len(self.udays) - 1)], days), "DayBoot に無い暦日がある"
        sn = np.bincount(di, weights=num, minlength=len(self.udays))
        sd = np.bincount(di, weights=den, minlength=len(self.udays))
        t = (self.W @ sn) / np.maximum(self.W @ sd, 1e-300)
        return t

    def ci(self, t):
        return [float(np.quantile(t, 0.025)), float(np.quantile(t, 0.975))]


def band_table(t, key, q, pay, null, days, bands, nb, boot: DayBoot, pool_boot=None, pool_point=None):
    rows, up_boot = [], []
    for b in range(nb):
        m = bands == b
        N = int(m.sum())
        if N == 0:
            rows.append({"band": b, "tickets": 0}); up_boot.append(None); continue
        roi = float(pay[m].sum() / N)
        roi_bt = boot.ratio_ci(days[m], pay[m], np.ones(N))
        if t == "fukusho":
            base = pool_point
            up_bt = roi_bt - pool_boot
        else:
            base = float(null[m].mean())
            up_bt = boot.ratio_ci(days[m], pay[m] - null[m], np.ones(N))
        rows.append({"band": b, "odds_min": float(key[m].min()), "odds_max": float(key[m].max()), "tickets": N,
                     "races_days": None, "hits": int((pay[m] > 0).sum()), "roi": roi, "roi_ci95": boot.ci(roi_bt),
                     "base": base, "uplift": roi - base, "uplift_ci95": boot.ci(up_bt),
                     "statutory_reference": STATUTORY[t], "roi_minus_statutory": roi - STATUTORY[t],
                     "expected_hit_mass": float(q[m].sum())})
        up_boot.append(up_bt)
    return rows, up_boot


def concentration(pay, days, m):
    out = {}
    p = pay[m]
    N = int(m.sum())
    o = np.sort(p)[::-1]
    for k in (1, 3, 10):
        out[f"drop_top{k}_tickets_roi"] = float((p.sum() - o[:k].sum()) / max(N - k, 1))
    dsum = pd.Series(p).groupby(days[m]).sum().sort_values(ascending=False)
    dcnt = pd.Series(np.ones(N)).groupby(days[m]).sum()
    for k in (1, 3):
        top = dsum.index[:k]
        out[f"drop_top{k}_days_roi"] = float((p.sum() - dsum.iloc[:k].sum()) / max(N - dcnt.loc[top].sum(), 1))
    return out


def main():
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    t0 = time.time()
    spec = json.loads((L.HERE / "spec.json").read_text(encoding="utf-8"))
    assert spec["version"] == "0.3-frozen", "spec v0.3-frozen の凍結前は Stage 1 を実行しない"
    rid_list = np.load(L.RESEARCH / "rid_list.npy", allow_pickle=True)
    tbl = payout_tables()
    res = {"spec_version": spec["version"], "boot": {"B": BOOT_B, "seed": BOOT_SEED, "cluster": "calendar day, year-stratified"},
           "types": {}}
    boots = {}
    for t in TYPES:
        R = {}
        for pk, pname in PERIODS.items():
            for lay in LAYERS:
                z = np.load(L.RESEARCH / f"tickets_{t}_{lay}_{pname}.npz")
                d = {k: z[k] for k in z.files}
                pay, keep, reason = ticket_payouts(t, rid_list, d, tbl)
                d = {k: v[keep] for k, v in d.items()}
                pay = pay[keep]
                if pk not in boots:
                    boots[pk] = DayBoot(d["day"], d["year"])
                bt = boots[pk]
                # 券種・価格層間で暦日集合が完全一致すること (共通再標本の前提)
                assert np.array_equal(np.unique(d["day"]), bt.udays), f"{t} {pk} {lay}: 暦日集合が他券種と不一致"
                pool_point = float(pay.mean())
                pool_boot = bt.ratio_ci(d["day"], pay, np.ones(len(pay)))
                bands = {"primary_mass10": (B.mass_bands(d["key"], d["q"], 10), 10),
                         "primary_mass20_sensitivity": (B.mass_bands(d["key"], d["q"], 20), 20),
                         "secondary_count10": (B.count_bands(d["key"], 10), 10),
                         "fixed": (B.fixed_bands(d["key"]), len(B.FIXED_EDGES) - 1)}
                out = {"excluded_races": reason, "tickets": int(len(pay)), "races": int(len(np.unique(d["race"]))),
                       "days": int(len(np.unique(d["day"]))), "pool_roi": pool_point, "pool_roi_ci95": bt.ci(pool_boot),
                       "tables": {}, "_boot": {}}
                for name, (bb, nb) in bands.items():
                    rows, upb = band_table(t, d["key"], d["q"], pay, d["null"], d["day"], bb, nb, bt, pool_boot, pool_point)
                    ins = B.summarize(d["key"], d["q"], d["race"], d["day"], bb, nb, d["pay"])
                    for r_, s_ in zip(rows, ins):
                        if r_["tickets"]:
                            r_["races_days"] = [s_["races"], s_["days"]]
                            r_["insufficient_label_free"] = s_["insufficient"]
                    out["tables"][name] = rows
                    if name == "primary_mass10":
                        out["_boot"]["up"] = upb
                        out["_bands"] = bb
                        out["concentration_primary"] = [concentration(pay, d["day"], bb == b) if (bb == b).any() else None
                                                        for b in range(10)]
                R[f"{pk}|{lay}"] = out
                R[f"{pk}|{lay}"]["_d"] = d
                R[f"{pk}|{lay}"]["_pay"] = pay
                print(f"[{t} {pk} {lay}] tickets={len(pay)} pool_roi={pool_point:.4f} ({time.time()-t0:.0f}s)", flush=True)
        # G1 (D0 discovery vs D0 evaluation, primary 10)
        pdisc, peval = R["discovery|D0"]["tables"]["primary_mass10"], R["evaluation|D0"]["tables"]["primary_mass10"]
        g1 = g1_decide(t, [r["roi"] for r in pdisc], [r["roi"] for r in peval],
                       [r["base"] for r in pdisc], [r["base"] for r in peval])
        s20d, s20e = R["discovery|D0"]["tables"]["primary_mass20_sensitivity"], R["evaluation|D0"]["tables"]["primary_mass20_sensitivity"]
        g1["sensitivity_20_bins_spearman"] = float(spearmanr([r["roi"] for r in s20d], [r["roi"] for r in s20e]).correlation)
        # G2
        g2 = {"grade": "NOT_RUN", "reason": "G1 not PASS"}
        if g1["grade"] == "PASS":
            sel = g2_select([r["uplift"] for r in pdisc], [r["uplift_ci95"][0] for r in pdisc])
            if sel:
                dd, pay_d = R["discovery|D0"]["_d"], R["discovery|D0"]["_pay"]
                md = np.isin(R["discovery|D0"]["_bands"], sel)
                de, pay_e = R["evaluation|D1"]["_d"], R["evaluation|D1"]["_pay"]
                me = np.isin(R["evaluation|D1"]["_bands"], sel)
                bt = boots["evaluation"]
                if t == "fukusho":
                    up_d = float(pay_d[md].mean() - R["discovery|D0"]["pool_roi"])
                    pool_b = bt.ratio_ci(de["day"], pay_e, np.ones(len(pay_e)))
                    up_e = float(pay_e[me].mean() - R["evaluation|D1"]["pool_roi"])
                    up_e_b = bt.ratio_ci(de["day"][me], pay_e[me], np.ones(int(me.sum()))) - pool_b
                else:
                    up_d = float((pay_d[md] - dd["null"][md]).mean())
                    up_e = float((pay_e[me] - de["null"][me]).mean())
                    up_e_b = bt.ratio_ci(de["day"][me], pay_e[me] - de["null"][me], np.ones(int(me.sum())))
                g2 = g2_decide(sel, up_d, up_e, bt.ci(up_e_b)[0])
                g2["evaluation_D1_roi_S"] = float(pay_e[me].mean())
                g2["evaluation_D1_roi_S_above_1"] = bool(pay_e[me].mean() > 1.0)
                g2["tickets_S_evaluation_D1"] = int(me.sum())
            else:
                g2 = g2_decide(sel, 0.0, 0.0, 0.0)
        # D0/D1 差と遷移 (evaluation)
        e0, e1 = R["evaluation|D0"], R["evaluation|D1"]
        d0d1 = [{"band": b, "roi_D1_minus_D0": (e1["tables"]["primary_mass10"][b]["roi"] - e0["tables"]["primary_mass10"][b]["roi"])}
                for b in range(10)]
        key0 = {(int(r), int(a), int(b_)): bb for r, a, b_, bb in zip(e0["_d"]["race"], e0["_d"]["a"], e0["_d"]["b"], e0["_bands"])}
        trans = np.zeros((10, 10), int)
        for r, a, b_, bb in zip(e1["_d"]["race"], e1["_d"]["a"], e1["_d"]["b"], e1["_bands"]):
            k0 = key0.get((int(r), int(a), int(b_)))
            if k0 is not None:
                trans[bb, k0] += 1
        for v in R.values():
            for k in ("_d", "_pay", "_boot", "_bands"):
                v.pop(k, None)
        res["types"][t] = {"layers": R, "G1": g1, "G2": g2, "evaluation_D1_minus_D0": d0d1,
                           "evaluation_transition_D1band_to_D0band": trans.tolist()}
        print(f"[{t}] G1={g1['grade']} rho={g1['spearman']:.3f} G2={g2['grade']}", flush=True)
    res["elapsed_sec"] = round(time.time() - t0, 1)
    (L.OUT / "stage1_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"[saved] out/stage1_results.json ({res['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
