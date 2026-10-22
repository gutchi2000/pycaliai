# -*- coding: utf-8 -*-
"""
min_budget.py — O-1 最小運用予算 B*（判定基準: docs/research/MIN_BUDGET_BSTAR_PREREG_20261022.md）

a95_equal_v1（凍結）を 1R 予算 B で運用したときの ROI(B) − ROI(10000) を、
(a) 100 円格子の丸め由来 と (b) 券種落ち由来 に分解し、開催日ブロック bootstrap で CI を出す。
配分は本番関数 a95_engine.build をそのまま呼ぶ（方策・本番ファイルは読むだけ）。
substrate と払戻判定は analysis/a95_chaos_gate_check.py と同じ（2023-25、9時オッズで帯）。

実行（root から）: venv311\\Scripts\\python.exe -m analysis.min_budget.min_budget
出力: analysis/min_budget/out/results.json, per_race.parquet
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
import a95_engine  # noqa: E402
import shadow_a95 as m  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
BUDGETS = [500, 1000, 2000, 3000, 5000, 10000]
EXTRA = [100, 200, 300, 400]          # 記述のみ（判定外）
REF = 10000
REPS, SEED = 3000, 20261022
YEAR_R, PASS_PREV = 3400, 6658 / 10261
GATE = 1.0
COLS = ["tan", "fuku", "umaren", "umatan", "wide", "wakuren", "sanpuku", "sanrentan"]


def load_rows(pol):
    rows = []
    for r in joblib.load(BASE / "data/_policy/bet_substrate.pkl"):
        p, ban, o = np.asarray(r["p"], float), r["ban"], np.asarray(r["odds9"], float)
        n = len(p)
        if n < 5 or not np.isfinite(o).any():
            continue
        order = np.argsort(-p, kind="stable")
        i1, i2, i3 = order[:3]
        o1 = float(o[i1]) if np.isfinite(o[i1]) and o[i1] > 0 else np.nan
        if not np.isfinite(o1):
            continue
        pp = p / p.sum()
        H = float(-(pp[pp > 0] * np.log(pp[pp > 0])).sum() / np.log(n))
        a1, a2, a3 = int(ban[i1]), int(ban[i2]), int(ban[i3])
        f, s, t = r["first"], r["second"], r["third"]
        w1, w2 = int(r["waku"][i1]), int(r["waku"][i2])
        wide = {frozenset((x, y)): v for x, y, v in (r.get("wide") or ())}
        per = dict(tan=(r["tan_pay"] if a1 == f else 0.0), fuku=r["fuku_pay"].get(a1, 0.0),
                   umaren=(r["umaren"] if {a1, a2} == {f, s} else 0.0),
                   umatan=(r["umatan"] if (a1, a2) == (f, s) else 0.0),
                   wide=float(wide.get(frozenset((a1, a2)), 0.0)),
                   wakuren=(r["wakuren"] if {w1, w2} == {int(r["waku_first"]), int(r["waku_second"])} else 0.0),
                   sanpuku=(r["sanpuku"] if {a1, a2, a3} == {f, s, t} else 0.0),
                   sanrentan=(r["sanrentan"] if (a1, a2, a3) == (f, s, t) else 0.0))
        per = {k: (float(v) if np.isfinite(v) else 0.0) for k, v in per.items()}
        horses = [dict(umaban=int(ban[i]), p_win=float(p[i]),
                      tansho_odds=(float(o[i]) if np.isfinite(o[i]) else None)) for i in range(n)]
        rows.append(dict(date=r["date"], year=r["date"][:4], n=n, H=H, p1=float(p[i1]),
                         band=m.band_of(o1, pol), horses=horses, a1=a1, a2=a2, **per))
    df = pd.DataFrame(rows)
    thr = df.H.quantile(0.667)
    df["keep"] = (df.H < thr) & (df.n > 7) & (df.p1 >= 0.05)
    return df


def allocate(df, pol, B):
    """本番 a95_engine.build で配分。戻り: cost, ret(格子), ret_cont(残った券種へ連続均等), 落ち件数 など。"""
    cost, ret, cont, drop, nonuni, kept_n = [], [], [], [], [], []
    for r in df.itertuples():
        full = m.sellable_kinds(list(pol["plan"][r.band]), r.n)
        try:
            b = a95_engine.build(r.horses, budget=B, policy=pol)
            assert b["a1"] == r.a1 and b["a2"] == r.a2 and b["band"] == r.band
            st = {t["kind"]: t["stake"] for t in b["tickets"]}
        except a95_engine.A95Error:
            st = {}
        kinds = list(st)
        c = float(sum(st.values()))
        cost.append(c)
        ret.append(sum(getattr(r, k) / 100.0 * v for k, v in st.items()))
        cont.append(c * np.mean([getattr(r, k) / 100.0 for k in kinds]) if kinds else 0.0)
        drop.append(len(full) - len(kinds))
        nonuni.append(len(set(st.values())) > 1)
        kept_n.append(len(kinds))
    out = pd.DataFrame(dict(cost=cost, ret=ret, cont=cont, drop=drop, nonuni=nonuni, kinds=kept_n), index=df.index)
    return out


def day_matrix(df, alloc, idx):
    """開催日 × 量 の集計（bootstrap 用）。"""
    d = df.loc[idx, ["date"]].copy()
    for B, a in alloc.items():
        d[f"c{B}"] = a.loc[idx, "cost"]
        d[f"r{B}"] = a.loc[idx, "ret"]
        d[f"k{B}"] = a.loc[idx, "cont"]
    return d.groupby("date").sum()


def boot_counts(n_days, reps, seed, size=None):
    rng = np.random.default_rng(seed)
    size = size or n_days
    return np.stack([np.bincount(rng.integers(0, n_days, size), minlength=n_days) for _ in range(reps)]).astype(float)


def analyse(df, alloc, idx, budgets, W=None):
    g = day_matrix(df, alloc, idx)
    if W is None:
        W = boot_counts(len(g), REPS, SEED)
    roi = lambda r, c: 100.0 * r / c
    res = {}
    ref_r, ref_c, ref_k = g[f"r{REF}"].values, g[f"c{REF}"].values, g[f"k{REF}"].values
    R0 = roi(ref_r.sum(), ref_c.sum()); K0 = roi(ref_k.sum(), ref_c.sum())
    bR0 = roi(W @ ref_r, W @ ref_c); bK0 = roi(W @ ref_k, W @ ref_c)
    for B in budgets:
        r, c, k = g[f"r{B}"].values, g[f"c{B}"].values, g[f"k{B}"].values
        R, K = roi(r.sum(), c.sum()), roi(k.sum(), c.sum())
        bR, bK = roi(W @ r, W @ c), roi(W @ k, W @ c)
        d, a, bb = bR - bR0, (bR - bK) - (bR0 - bK0), bK - bK0
        q = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
        ret_sorted = np.sort(alloc[B].loc[idx, "ret"].values)[::-1]
        res[B] = dict(
            n=int(len(idx)), roi=R, roi_ci=q(bR), roi_cont=K,
            delta=R - R0, delta_ci=q(d), abs_delta_hi=float(max(abs(q(d)[0]), abs(q(d)[1]))),
            round_part=(R - K) - (R0 - K0), round_ci=q(a),
            drop_part=K - K0, drop_ci=q(bb),
            races_with_drop=int((alloc[B].loc[idx, "drop"] > 0).sum()),
            races_unbuyable=int((alloc[B].loc[idx, "kinds"] == 0).sum()),
            share_nonuniform=float(alloc[B].loc[idx, "nonuni"].mean()),
            ex10=100.0 * (ret_sorted.sum() - ret_sorted[:10].sum()) / c.sum(),
            hit=float((alloc[B].loc[idx, "ret"] > 0).mean() * 100),
        )
    return res, g


def annual(g, B, roi_pt, roi_ex10, n_year_races, days_per_year, seed):
    pnl_day = g[f"r{B}"].values - g[f"c{B}"].values
    rc = (g[f"c{B}"].values / B)  # レース数（買えたレース）
    W = boot_counts(len(g), REPS, seed, size=days_per_year)
    ypnl = W @ pnl_day
    yR = W @ rc
    return dict(
        year_races=n_year_races, turnover=n_year_races * B,
        exp_pnl_insample=n_year_races * B * (roi_pt / 100 - 1),
        exp_pnl_ex10=n_year_races * B * (roi_ex10 / 100 - 1),
        boot_days=days_per_year, boot_mean_races=float(yR.mean()),
        p5=float(np.percentile(ypnl, 5)), p50=float(np.percentile(ypnl, 50)), p95=float(np.percentile(ypnl, 95)),
        p_plus=float((ypnl > 0).mean() * 100),
    )


def main():
    pol = m.load_policy()
    df = load_rows(pol)
    allB = sorted(set(BUDGETS + EXTRA))
    alloc = {B: allocate(df, pol, B) for B in allB}
    keep_idx = df.index[df.keep]
    all_idx = df.index
    pass_rate = len(keep_idx) / len(df)
    days_per_year = int(round(df.groupby("year").date.nunique().mean()))
    out = dict(meta=dict(n_all=len(df), n_keep=len(keep_idx), pass_rate=pass_rate, pass_rate_prev=PASS_PREV,
                         years=sorted(df.year.unique().tolist()), reps=REPS, seed=SEED, ref=REF,
                         days_per_year=days_per_year, policy_sha256=pol["policy_sha256"]))
    for name, idx in (("keep", keep_idx), ("all", all_idx)):
        res, g = analyse(df, alloc, idx, allB)
        out[name] = {str(B): v for B, v in res.items()}
        out[f"{name}_by_year"] = {}
        for y in sorted(df.year.unique()):
            iy = idx[df.loc[idx, "year"] == y]
            ry, _ = analyse(df, alloc, iy, BUDGETS)
            out[f"{name}_by_year"][y] = {str(B): dict(roi=v["roi"], delta=v["delta"], delta_ci=v["delta_ci"],
                                                      round_part=v["round_part"], n=v["n"]) for B, v in ry.items()}
        if name == "keep":
            ny = int(round(YEAR_R * PASS_PREV))
            out["annual_keep"] = {str(B): annual(g, B, res[B]["roi"], res[B]["ex10"], ny, days_per_year, SEED + B)
                                  for B in BUDGETS}
    # 判定
    ok = {B: (out["keep"][str(B)]["abs_delta_hi"] <= GATE and out["keep"][str(B)]["races_with_drop"] == 0
              and out["all"][str(B)]["races_with_drop"] == 0) for B in BUDGETS}
    bstar = REF
    for B in BUDGETS:
        if all(ok[x] for x in BUDGETS if x >= B):
            bstar = B
            break
    out["verdict"] = dict(pass_by_B={str(B): bool(v) for B, v in ok.items()}, bstar=bstar)
    # 券種別の配分表（帯 × B）
    tab = {}
    for band, kinds in pol["plan"].items():
        tab[band] = {}
        for nk_label, kk in (("9頭以上", kinds), ("8頭以下", m.sellable_kinds(list(kinds), 8))):
            tab[band][nk_label] = {str(B): m.equal_stakes(kk, B, 100) for B in BUDGETS}
    out["stake_table"] = tab
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    pr = df[["date", "year", "n", "band", "keep"]].copy()
    for B in allB:
        pr[f"cost{B}"] = alloc[B].cost.values
        pr[f"ret{B}"] = alloc[B].ret.values
    pr.to_parquet(OUT / "per_race.parquet")

    print(f"母集団 全 {len(df)}R / ガード通過近似 {len(keep_idx)}R ({100*pass_rate:.1f}%)  年平均開催日 {days_per_year}")
    for name in ("keep", "all"):
        print(f"\n== {name} ==  B: ROI [CI] | Δ [CI] | 丸め [CI] | 落ち | 落ちR 不能R 非均等%")
        for B in allB:
            v = out[name][str(B)]
            tag = "" if B in BUDGETS else " (記述)"
            print(f" {B:>6}: {v['roi']:6.2f} [{v['roi_ci'][0]:6.1f},{v['roi_ci'][1]:6.1f}] | {v['delta']:+6.2f} "
                  f"[{v['delta_ci'][0]:+6.2f},{v['delta_ci'][1]:+6.2f}] | {v['round_part']:+6.2f} "
                  f"[{v['round_ci'][0]:+6.2f},{v['round_ci'][1]:+6.2f}] | {v['drop_part']:+6.2f} | "
                  f"{v['races_with_drop']:>5} {v['races_unbuyable']:>5} {100*v['share_nonuniform']:5.1f}  ex10 {v['ex10']:6.1f}{tag}")
        for y, ry in out[f"{name}_by_year"].items():
            print(f"   {y}: " + " ".join(f"{B}:{ry[str(B)]['delta']:+.2f}[{ry[str(B)]['delta_ci'][0]:+.2f},{ry[str(B)]['delta_ci'][1]:+.2f}]" for B in BUDGETS))
    print("\n== 年間（ガード通過近似, 年 %dR） ==" % int(round(YEAR_R * PASS_PREV)))
    for B in BUDGETS:
        a = out["annual_keep"][str(B)]
        print(f" {B:>6}: 売上 ¥{a['turnover']:>12,.0f} 期待収支 in-sample ¥{a['exp_pnl_insample']:>+12,.0f} / 上位10除外 ¥{a['exp_pnl_ex10']:>+12,.0f} | "
              f"年収支 5% ¥{a['p5']:>+12,.0f} 50% ¥{a['p50']:>+12,.0f} 95% ¥{a['p95']:>+12,.0f} P(+) {a['p_plus']:.0f}% (bootR {a['boot_mean_races']:.0f})")
    print("\n判定:", out["verdict"])


if __name__ == "__main__":
    main()
