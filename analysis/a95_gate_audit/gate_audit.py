# -*- coding: utf-8 -*-
"""
gate_audit.py — a95_equal_v1 の 500R 判定ゲート監査 (M-1, label-free)
=====================================================================
宣言: docs/research/A95_GATE_AUDIT_PREREG_20261022.md (計算前に固定)
結果: docs/research/A95_GATE_AUDIT_20261022.md

読むもの (すべて読み取りのみ):
  data/_policy/bet_substrate.pkl          2023-25 OOS (v6 p_win + 9時単勝オッズ + 全券種払戻)
  data/_research/mcond/market.parquet     historical_pre_snapshot (snap=="pre", 確定の約35分前 = 発走約28分前)
  data/chaos_quantiles.json               本番の混戦度参照分布 (percentile 変換)
  data/shadow_policies/a95_equal_v1.json  凍結方策 (shadow_a95.load_policy で sha 検査)
  前向き (2026-10-03 以降):
    data/shadow_ledger/a95_equal_v1/decisions/*.json   → band 列だけ
    reports/cowork_output/*_bets.json                  → races[].a95.band / a95_all.band だけ
  ※ settlements・kekka・着順・払戻は前向き分について一切開かない。

やること:
  1. レース的中 (凍結 plan の採用券種のどれかが当たる、枠連は 9 頭以上のみ) の ICC と deff
     (開催日 / 開催日×場 / 週) を、全レース×9時帯 (CW-10 近似) と
     参戦ガード通過×pre 帯 (CW-11 近似) で推定。
  2. 帯構成比と期待的中率 (4 母集団: 全/ガード × 9時/pre)、床 19.7% の再現と CW-11 母集団での類似値。
  3. 前向き band 構成比 (label-free)。
  4. 500R 棄却確率 (真値 23.3/19.7/15%、iid と deff 補正)。

実行 (root から): python -m analysis.a95_gate_audit.gate_audit
"""
from __future__ import annotations
import bisect, json, math, sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
import shadow_a95 as m  # noqa: E402  (読み取りのみ: load_policy / band_of / sellable_kinds)

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(parents=True, exist_ok=True)
pol = m.load_policy()
LABELS = [b[2] for b in pol["bands"]]
KINDS = m.KINDS
N_GATE = int(pol["registration"]["min_valid_races"])
FLOOR = float(pol["registration"]["reject_if_hit_upper_below"])
Z = 1.645
RNG = np.random.default_rng(20261022)

# ------------------------------------------------------------------ chaos (本番と同じ定義)
QREF = json.loads((BASE / "data" / "chaos_quantiles.json").read_text(encoding="utf-8"))["quantiles"]["field_chaos_score"]


def chaos_raw(p: np.ndarray) -> float:
    """export_marks_json.race_confidence と同じ: p>1e-9 を正規化した entropy / log(件数)。"""
    q = p[p > 1e-9]
    if len(q) == 0 or q.sum() <= 0:
        return 0.0
    q = q / q.sum()
    h = float(-(q * np.log(q)).sum())
    mx = math.log(len(q))
    return max(0.0, min(1.0, h / mx)) if mx > 0 else 0.0


def chaos_pct_ref(x: float) -> float:
    """production_policy.percentile と同じ線形補間 (参照分布 serve34-forward-policy-20260825)。"""
    v = QREF
    if x <= v[0]:
        return 0.0
    if x >= v[-1]:
        return 1.0
    i = bisect.bisect_right(v, x)
    lo, hi = v[i - 1], v[i]
    return ((i - 1) + (0.0 if hi == lo else (x - lo) / (hi - lo))) / 100.0


# ------------------------------------------------------------------ substrate
def build() -> pd.DataFrame:
    races = joblib.load(BASE / "data/_policy/bet_substrate.pkl")
    mk = pd.read_parquet(BASE / "data/_research/mcond/market.parquet",
                         columns=["rid16", "ban", "snap", "odds", "min_before_final"])
    fin_ = mk[mk.snap == "final"]
    fin = {(r, int(b)): float(o) for r, b, o in zip(fin_.rid16, fin_.ban, fin_.odds)}
    mk = mk[mk.snap == "pre"]
    pre = {(r, int(b)): float(o) for r, b, o in zip(mk.rid16, mk.ban, mk.odds)}
    rows = []
    for r in races:
        p = np.asarray(r["p"], float)
        ban = r["ban"]
        n = len(p)
        if n < 5:
            continue
        order = np.argsort(-p)
        i1, i2, i3 = order[:3]
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
        o9 = float(r["odds9"][i1])
        o9 = o9 if np.isfinite(o9) and o9 > 0 else np.nan
        opre = pre.get((r["rid"], a1), np.nan)
        opre = opre if np.isfinite(opre) and opre > 1.0 else np.nan
        ofin = fin.get((r["rid"], a1), np.nan)
        ofin = ofin if np.isfinite(ofin) and ofin > 1.0 else np.nan
        cr = chaos_raw(p)
        rows.append(dict(rid=r["rid"], date=r["date"], year=r["date"][:4], place=r["rid"][8:10], n=n,
                         p1=float(p[i1]), chaos=cr, chaos_pct=chaos_pct_ref(cr), o9=o9, opre=opre, ofin=ofin,
                         **{f"pay_{k}": v for k, v in per.items()}))
    df = pd.DataFrame(rows)
    df["dp"] = df.date + "_" + df.place
    dt = pd.to_datetime(df.date)
    iso = dt.dt.isocalendar()
    df["week"] = iso.year.astype(str) + "W" + iso.week.astype(str).str.zfill(2)
    df["band9"] = [m.band_of(o, pol) if np.isfinite(o) else None for o in df.o9]
    df["bandpre"] = [m.band_of(o, pol) if np.isfinite(o) else None for o in df.opre]
    df["bandfin"] = [m.band_of(o, pol) if np.isfinite(o) else None for o in df.ofin]
    # 参戦ガード (本番 hard_skip の近似; 障害は substrate に無い)
    q667 = df.chaos.quantile(0.667)
    base_ok = (df.n > 7) & (df.p1 >= 0.05)
    df["guard_ref"] = base_ok & (df.chaos_pct + 1e-12 < 0.667)     # 本番の参照分布でそのまま
    df["guard_q"] = base_ok & (df.chaos < q667)                     # 母集団内 66.7% 点 (参戦率を本番設計に揃える)
    df.attrs["q667_raw"] = float(q667)
    return df


def hit_with(df: pd.DataFrame, band_col: str, plan: dict | None = None) -> pd.Series:
    plan = plan or pol["plan"]
    out = np.zeros(len(df), bool)
    valid = np.zeros(len(df), bool)
    for j, (b, n) in enumerate(zip(df[band_col].values, df.n.values)):
        if b is None or (isinstance(b, float) and np.isnan(b)):
            continue
        kinds = m.sellable_kinds(list(plan.get(b, [])), int(n))
        if not kinds:
            continue
        valid[j] = True
        out[j] = any(df[f"pay_{k}"].values[j] > 0 for k in kinds)
    return pd.Series(np.where(valid, out.astype(float), np.nan), index=df.index)


# ------------------------------------------------------------------ ICC / deff
def anova_icc(y: np.ndarray, g: np.ndarray) -> float:
    """一元配置 ANOVA 推定量 (不均衡クラスタ、二値でも可)。"""
    d = pd.DataFrame(dict(y=y, g=g))
    gs = d.groupby("g").y.agg(["sum", "count", "mean"])
    N, k = len(d), len(gs)
    ybar = d.y.mean()
    msb = float((gs["count"] * (gs["mean"] - ybar) ** 2).sum() / (k - 1))
    msw = float(((d.y - d.g.map(gs["mean"])) ** 2).sum() / (N - k))
    m0 = (N - float((gs["count"] ** 2).sum()) / N) / (k - 1)
    return (msb - msw) / (msb + (m0 - 1) * msw)


def deff_ratio(y: np.ndarray, g: np.ndarray) -> tuple[float, float]:
    """クラスタ標本の比推定量 (線形化) の design effect。deff = s_e^2 / (mbar p(1-p))。
    500R をこのクラスタ構成 (同じ平均サイズ) で集めたときの分散 / iid 二項分散。"""
    d = pd.DataFrame(dict(y=y, g=g))
    gs = d.groupby("g").y.agg(["sum", "count"])
    p = gs["sum"].sum() / gs["count"].sum()
    e = gs["sum"] - p * gs["count"]
    k = len(gs)
    s2 = float((e ** 2).sum() / (k - 1))
    mbar = float(gs["count"].mean())
    return s2 / (mbar * p * (1 - p)), mbar


def cluster_boot_ci(y, g, fn, reps=1000):
    d = pd.DataFrame(dict(y=y, g=g))
    groups = [x.y.values for _, x in d.groupby("g")]
    keys = np.arange(len(groups))
    vals = []
    for _ in range(reps):
        idx = RNG.choice(keys, size=len(keys), replace=True)
        yy = np.concatenate([groups[i] for i in idx])
        gg = np.concatenate([np.full(len(groups[i]), j) for j, i in enumerate(idx)])
        vals.append(fn(yy, gg))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def window_deff(sub: pd.DataFrame, ycol: str, n_win: int = N_GATE) -> dict:
    """連続する開催日を前から積み、ちょうど 500R の非重複窓を作ったときの的中率のばらつき (季節・年ドリフト込み)。"""
    s = sub.sort_values(["date", "place", "rid"])
    y = s[ycol].values
    k = len(y) // n_win
    hs = np.array([y[i * n_win:(i + 1) * n_win].mean() for i in range(k)])
    p = y.mean()
    v_iid = p * (1 - p) / n_win
    v = hs.var(ddof=1)
    lo = v * (k - 1) / stats.chi2.ppf(0.975, k - 1)
    hi = v * (k - 1) / stats.chi2.ppf(0.025, k - 1)
    return dict(n_windows=int(k), sd_window=float(np.sqrt(v)), sd_iid=float(np.sqrt(v_iid)),
                deff=float(v / v_iid), deff_ci95=[float(lo / v_iid), float(hi / v_iid)],
                window_hits=[round(float(x), 4) for x in hs])


def icc_block(sub: pd.DataFrame, ycol: str) -> dict:
    s = sub[sub[ycol].notna()]
    y = s[ycol].values.astype(float)
    out = dict(n=int(len(s)), hit=float(y.mean()))
    for lev in ("date", "dp", "week"):
        g = s[lev].values
        icc = anova_icc(y, g)
        de, mbar = deff_ratio(y, g)
        icc_ci = cluster_boot_ci(y, g, anova_icc, reps=600)
        de_ci = cluster_boot_ci(y, g, lambda a, b: deff_ratio(a, b)[0], reps=600)
        out[lev] = dict(icc=round(icc, 5), icc_ci95=[round(x, 5) for x in icc_ci],
                        mbar=round(mbar, 2), deff_formula=round(1 + (mbar - 1) * icc, 4),
                        deff=round(de, 4), deff_ci95=[round(x, 4) for x in de_ci],
                        n_clusters=int(s[lev].nunique()))
    # band を条件づけた残差の ICC (帯構成の日内偏りを除いたもの、参考)
    bcol = "band9" if ycol.endswith("9") else "bandpre"
    resid = y - s.groupby(bcol)[ycol].transform("mean").values
    out["resid_after_band_icc_date"] = round(anova_icc(resid, s["date"].values), 5)
    out["window500"] = window_deff(s, ycol)
    s2 = s.copy()
    s2["_dm"] = s2[ycol] - s2.groupby("year")[ycol].transform("mean") + s2[ycol].mean()
    out["window500_year_demeaned"] = window_deff(s2, "_dm")
    ym = s.date.str[:6].values
    out["month_icc"] = round(anova_icc(y, ym), 5)
    out["month_deff_ratio"] = round(deff_ratio(y, ym)[0], 4)
    tab = pd.crosstab(s.year, s[ycol])
    chi2, pv, _, _ = stats.chi2_contingency(tab.values)
    out["year_hit"] = {k: round(float(v), 4) for k, v in s.groupby("year")[ycol].mean().items()}
    out["year_homogeneity_chi2_p"] = round(float(pv), 5)
    return out


# ------------------------------------------------------------------ 床の再現 (alloc_95_design と同じ手順の的中率)
SPLITS = [(["2023"], ["2024", "2025"]), (["2025"], ["2023", "2024"]),
          (["2023", "2024"], ["2025"]), (["2024", "2025"], ["2023"])]


def plan_from(fit: pd.DataFrame, band_col: str = "band9", thr: float = 95.0) -> dict:
    plan = {}
    for b in LABELS:
        g = fit[fit[band_col] == b]
        if len(g) < 30:
            plan[b] = []
            continue
        plan[b] = [k for k in KINDS if g[f"pay_{k}"].mean() >= thr]
    return plan


def oos_hits(df_all: pd.DataFrame, eval_mask: pd.Series, eval_band: str, wakuren_rule: bool) -> list[dict]:
    res = []
    for fy, ey in SPLITS:
        plan = plan_from(df_all[df_all.year.isin(fy)], "band9")
        ev = df_all[eval_mask & df_all.year.isin(ey)]
        if wakuren_rule:
            h = hit_with(ev, eval_band, plan)
        else:  # alloc_95_design は枠連 8 頭以下も残す (払戻 0 で外れ扱い)
            h = pd.Series([(any(ev[f"pay_{k}"].values[j] > 0 for k in plan.get(b, [])) if plan.get(b) else np.nan)
                           if b is not None else np.nan for j, b in enumerate(ev[eval_band].values)], index=ev.index)
        res.append(dict(fit="+".join(fy), eval="+".join(ey), n=int(h.notna().sum()), hit=round(float(h.mean()), 4)))
    return res


# ------------------------------------------------------------------ 検出力
def thr_obs(deff: float, n: int = N_GATE) -> float:
    """h + Z*sqrt(deff*h(1-h)/n) = FLOOR を満たす観測的中率 h (これ未満で棄却)。"""
    lo, hi = 0.0, FLOOR
    for _ in range(100):
        mid = (lo + hi) / 2
        if mid + Z * math.sqrt(deff * mid * (1 - mid) / n) < FLOOR:
            lo = mid
        else:
            hi = mid
    return lo


def p_reject(p_true: float, deff_data: float, deff_rule: float, n: int = N_GATE) -> dict:
    """凍結規則 (deff_rule=1) / 補正規則 で、観測的中率の分散が deff_data × 二項分散のときの棄却確率。
    deff_data=1 は正確な二項、それ以外は正規近似。"""
    h_star = thr_obs(deff_rule, n)
    # 棄却は ub < FLOOR。ub は h の単調増加なので h < h_star ⇔ 棄却 (格子上は X/n で判定)
    if abs(deff_data - 1.0) < 1e-12:
        xs = np.arange(n + 1)
        hs = xs / n
        ub = hs + Z * np.sqrt(deff_rule * hs * (1 - hs) / n)
        pr = float(stats.binom.pmf(xs[ub < FLOOR], n, p_true).sum())
        return dict(h_star=round(h_star, 4), p_reject=round(pr, 4), method="exact_binomial")
    sd = math.sqrt(deff_data * p_true * (1 - p_true) / n)
    pr_norm = float(stats.norm.cdf((h_star - p_true) / sd))
    return dict(h_star=round(h_star, 4), p_reject=round(pr_norm, 4), method="normal_approx")


# ------------------------------------------------------------------ 前向き band 構成 (label-free)
def forward_bands(start: str = "20261003") -> dict:
    out = dict(cw10_bundle={}, cw11_t10={}, t10_all={}, dates=[])
    dec_dir = BASE / "data/shadow_ledger/a95_equal_v1/decisions"
    for p in sorted(dec_dir.glob("*.json")):
        if p.stem < start:
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        out["dates"].append(p.stem)
        for r in d["races"]:
            if r.get("valid"):
                out["cw10_bundle"][r["band"]] = out["cw10_bundle"].get(r["band"], 0) + 1
    for p in sorted((BASE / "reports/cowork_output").glob("*_bets.json")):
        ds = p.name[:8]
        if not ds.isdigit() or ds < start:
            continue
        b = json.loads(p.read_text(encoding="utf-8"))
        races = b.get("bets") or []
        if isinstance(races, dict):
            races = list(races.values())
        for r in races:
            a = r.get("a95") if isinstance(r, dict) else None
            if a and r.get("bets") and a.get("band"):
                out["cw11_t10"][a["band"]] = out["cw11_t10"].get(a["band"], 0) + 1
            aa = r.get("a95_all") if isinstance(r, dict) else None
            if aa and aa.get("band"):
                out["t10_all"][aa["band"]] = out["t10_all"].get(aa["band"], 0) + 1
    return out


def shares(counts: dict) -> dict:
    tot = sum(counts.values())
    return {b: (round(counts.get(b, 0) / tot, 4) if tot else None) for b in LABELS} | {"n": int(tot)}


# ------------------------------------------------------------------ main
def main() -> None:
    df = build()
    df["hit9"] = hit_with(df, "band9")
    df["hitpre"] = hit_with(df, "bandpre")
    df["hitfin"] = hit_with(df, "bandfin")
    res: dict = dict(policy_id=pol["policy_id"], policy_sha256=pol["policy_sha256"], n_gate=N_GATE, floor=FLOOR,
                     substrate_races=int(len(df)), q667_raw_within=df.attrs["q667_raw"],
                     guard_rates=dict(guard_ref=float(df.guard_ref.mean()), guard_q=float(df.guard_q.mean())))

    pops = {
        "A_all_x_9h (CW-10 設計母集団)": (pd.Series(True, index=df.index), "band9", "hit9"),
        "B_all_x_pre": (pd.Series(True, index=df.index), "bandpre", "hitpre"),
        "C_guardq_x_9h": (df.guard_q, "band9", "hit9"),
        "D_guardq_x_pre (CW-11 近似)": (df.guard_q, "bandpre", "hitpre"),
        "E_guardref_x_pre (CW-11 近似, 本番参照分布)": (df.guard_ref, "bandpre", "hitpre"),
        "F_guardq_x_final (感度: 帯を確定オッズで。発走後の値なので決定には使えない)": (df.guard_q, "bandfin", "hitfin"),
    }
    # 帯別の的中率 (設計母集団 A) — 構成調整の基準
    A = df[df.band9.notna()]
    hA = A.groupby("band9").hit9.mean().reindex(LABELS)
    comp = {}
    for name, (mask, bcol, ycol) in pops.items():
        s = df[mask & df[bcol].notna() & df[ycol].notna()]
        cnt = s[bcol].value_counts().reindex(LABELS).fillna(0)
        w = cnt / cnt.sum()
        hb = s.groupby(bcol)[ycol].mean().reindex(LABELS)
        comp[name] = dict(
            n=int(len(s)), share=w.round(4).to_dict(), hit_by_band=hb.round(4).to_dict(),
            hit_realized=round(float(s[ycol].mean()), 4),
            hit_expected_from_A_bandrates=round(float((w * hA).sum()), 4),
            by_year={y: round(float(g[ycol].mean()), 4) for y, g in s.groupby("year")})
    res["composition"] = comp
    res["band_agreement_9h_vs_pre"] = dict(
        same_band=round(float((df.band9 == df.bandpre).mean()), 4),
        crosstab=pd.crosstab(df.band9, df.bandpre).reindex(index=LABELS, columns=LABELS).fillna(0).astype(int).to_dict())

    # 床の再現と CW-11 類似値
    res["floor_reproduction"] = dict(
        A_all_9h_as_alloc95=oos_hits(df, pd.Series(True, index=df.index), "band9", wakuren_rule=False),
        A_all_9h_with_wakuren_rule=oos_hits(df, pd.Series(True, index=df.index), "band9", wakuren_rule=True),
        D_guardq_pre=oos_hits(df, df.guard_q, "bandpre", wakuren_rule=True),
        E_guardref_pre=oos_hits(df, df.guard_ref, "bandpre", wakuren_rule=True),
    )

    # ICC / deff
    icc = {}
    for name, (mask, bcol, ycol) in pops.items():
        if name[0] in "BCF":
            continue
        icc[name] = icc_block(df[mask & df[bcol].notna()], ycol)
    res["icc"] = icc

    # 検出力
    cw10 = icc["A_all_x_9h (CW-10 設計母集団)"]
    cw11 = icc["D_guardq_x_pre (CW-11 近似)"]
    power = {}
    for lab, blk in (("CW-10", cw10), ("CW-11", cw11)):
        dclu = max(blk["date"]["deff"], blk["dp"]["deff"])
        dwin = blk["window500"]["deff"]
        scen = {"iid": 1.0, "cluster_est(max day,dp)": dclu, "window500_est": dwin, "stress_1.5": 1.5, "stress_2.0": 2.0}
        rows = {}
        for pt in (0.233, 0.197, 0.15):
            r = {}
            for sn, dd in scen.items():
                r[f"frozen_rule|data={sn}"] = p_reject(pt, dd, 1.0)
                if dd > 1.0:
                    r[f"corrected_rule|data={sn}"] = p_reject(pt, dd, dd)
            rows[f"{pt:.3f}"] = r
        power[lab] = dict(deff_cluster=round(dclu, 4), deff_window=round(dwin, 4),
                          threshold_obs={sn: round(thr_obs(max(dd, 1.0)), 4) for sn, dd in scen.items()},
                          reject_iff_hits_le=int(max(x for x in range(N_GATE + 1)
                                                     if x / N_GATE + Z * math.sqrt((x / N_GATE) * (1 - x / N_GATE) / N_GATE) < FLOOR)),
                          table=rows)
    res["power"] = power

    # 前向き band 構成 (label-free)
    fb = forward_bands()
    res["forward_band_composition_label_free"] = dict(
        note="2026-10-03〜10-18 は本番モデル C0 混成期間 (判定に数えない、記述のみ)。band 列以外は読んでいない。",
        dates=fb["dates"], cw10_bundle=shares(fb["cw10_bundle"]), cw11_t10=shares(fb["cw11_t10"]),
        t10_all=shares(fb["t10_all"]),
        expected_hit_from_A_bandrates=dict(
            cw10_bundle=(round(float(sum(shares(fb["cw10_bundle"])[b] * hA[b] for b in LABELS)), 4) if fb["cw10_bundle"] else None),
            cw11_t10=(round(float(sum(shares(fb["cw11_t10"])[b] * hA[b] for b in LABELS)), 4) if fb["cw11_t10"] else None),
            cw11_t10_using_D_bandrates=(round(float(sum(shares(fb["cw11_t10"])[b] * comp["D_guardq_x_pre (CW-11 近似)"]["hit_by_band"][b] for b in LABELS)), 4) if fb["cw11_t10"] else None),
        ))

    def gof(counts, ref_share):
        obs = np.array([counts.get(b, 0) for b in LABELS], float)
        exp = np.array([ref_share[b] for b in LABELS], float) * obs.sum()
        # 期待度数が小さい帯 (8-15倍, 15倍〜) をまとめる
        o2 = np.r_[obs[:4], obs[4:].sum()]; e2 = np.r_[exp[:4], exp[4:].sum()]
        return round(float(stats.chisquare(o2, e2 * o2.sum() / e2.sum()).pvalue), 5)
    res["forward_band_composition_label_free"]["gof_p"] = dict(
        cw10_bundle_vs_A_9h=gof(fb["cw10_bundle"], comp["A_all_x_9h (CW-10 設計母集団)"]["share"]),
        cw11_t10_vs_D_pre=gof(fb["cw11_t10"], comp["D_guardq_x_pre (CW-11 近似)"]["share"]),
        cw11_t10_vs_F_final=gof(fb["cw11_t10"], comp["F_guardq_x_final (感度: 帯を確定オッズで。発走後の値なので決定には使えない)"]["share"]),
        t10_all_vs_B_pre=gof(fb["t10_all"], comp["B_all_x_pre"]["share"]),
    )
    (OUT / "gate_audit.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    for _st in (sys.stdout, sys.stderr):
        if hasattr(_st, "reconfigure"):
            _st.reconfigure(encoding="utf-8", errors="replace")
    main()
