# -*- coding: utf-8 -*-
"""
parimutuel_equilibrium_exp.py — パリミュチュエル均衡理論で本命-大穴バイアス (FLB) の強さがどこで変わるかを測り、
3 つの理論 (Ottaviani–Sørensen / Shin 型インサイダー / 危険愛好・素人参加) を見分ける (2026-11-14)

実行 (リポジトリ root から):  python -m lab.experiments.parimutuel_equilibrium_exp
出力: reports/parimutuel_equilibrium_exp.json

======================================================================================
事前登録 (結果を見る前に固定。2026-11-14)
======================================================================================
対象: JRA 平地 (トラックコード 51 以上 = 障害 は除外)、8 頭以上、2013-2025。単勝確定オッズの逆数の和が 1.20〜1.33、
      勝ち馬がちょうど 1 頭 (同着レースは除外)。

FLB の測り方 (主指標):
  β = 条件付きロジット  P(i 勝ち) = p_i^β / Σ_j p_j^β  の β (p = 確定単勝オッズの逆数を 1 に正規化した市場確率)。
  logit 表記の  logit(p_true) = α + β·logit(p_mkt)  と同じ傾き。β > 1 = 本命が売れ不足 (FLB)、β = 1 = 市場が較正済み。
  副指標: 発走約 28 分前 (T-28) の市場確率で同じ β (β_pre)。ROI: 確定オッズ 1 番人気 ≤1.9 倍の単勝・複勝、1.0〜1.4 倍の複勝。
  ※ 確定オッズは「評価される値段」として使う (締切後の値。結果は含まない)。

レジーム代理変数 (いずれも結果が出る前に決まる。★ = 締切時点で確定する値 = T-28 時点では未知だが着順情報は含まない):
  POOL   ★ log(確定単勝票数) − 直前 365 日 (当日を含まない) の全レースの log 確定単勝票数 中央値。
           = 参加者数 / 資金規模の相対値。最初の 365 日 (2013) は欠損として POOL 分析から外す。
  LATE   ★ log(確定単勝票数 / T-28 単勝票数) を T-28 の分数 (min_before) で 2013-16 に線形回帰した残差。
           = 締切前約 30 分に入った資金の厚さ (Shin 型の「遅い資金」)。
  LINFO  ★ log KL(p_fin ‖ p_pre)  = 最後の約 30 分で市場の確率分布がどれだけ書き換わったか (遅い資金の情報量)。
  EINFO    log KL(p_pre ‖ p_9)    = 朝 9 時 → T-28 の書き換え量 (発走前に観測可能、実用版)。
  GRADE    G1 / 重賞 (G2・G3・重賞) / OP (OP・OP(L)・ｵｰﾌﾟﾝ・3勝・1600万) / 条件 (1勝・2勝・500万・1000万) / 未勝利 / 新馬
  G1DAY    G1 が行われた日の、G1 以外のレース (= 大一番の日の素人流入、レースの格と分離)
  FIELD    頭数 8-11 / 12-15 / 16-18 (固定)
  連続量 (POOL, LATE, LINFO, EINFO) の 3 分位の境界は 2013-16 のレースで固定 (以後の年もその境界で切る)。

期間: P1 = 2013-2019 (コロナ前)、P2 = 2020-2025 (コロナ後)。同じ比較を両期で別々に。
CI: 開催日クラスタ bootstrap (B=400)、95% パーセンタイル。
判定: レジーム効果が「本物」= 両期で同符号かつ両期とも CI が 0 を跨がない。

理論ごとの予測符号 (上位 3 分位 − 下位 3 分位 の β 差、または G1 − 平場):
                      POOL   LATE   LINFO  G1/G1DAY   新馬(情報非対称)
  O–S (情報均衡)        +      (+)    +       +(N大)    +
  Shin (インサイダー)    −      +      +       −(希釈)   +
  危険愛好/素人          +      0      0       +         −(素人少ない)
  → POOL と G1/G1DAY の符号で Shin と (O–S, 素人) が分かれ、新馬 と LATE/LINFO で O–S と 素人 が分かれる。

同時モデル (主分析の補助): β_r = θ0 + θ·z_r  (z = POOL, LATE, LINFO, EINFO の 2013-16 標準化値, G1, 重賞, 新馬, 未勝利, G1DAY, log頭数標準化)
  を期間ごとに最尤推定。互いに相関する代理変数 (G1 と POOL など) の偏効果を見る。
======================================================================================
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"E:/PyCaLiAI")
OUTDIR = Path(__file__).resolve().parents[2] / "reports"
DRIFT = BASE / "data/_joint/tanpuk_all_drift.parquet"
FAV = BASE / "data/_joint/tanpuk_fav.parquet"
MASTER = BASE / "data/master_v2_20130105-20251228.csv"
KEKKA = BASE / "data/kekka_20130105-20251228.csv"
import os
B = int(os.environ.get("PE_B", "400"))
SEED = 20261114
PERIODS = {"P1_2013_19": (2013, 2019), "P2_2020_25": (2020, 2025)}


# ----------------------------------------------------------------------------- data
def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    d = pd.read_parquet(DRIFT)
    tf = pd.read_parquet(FAV, columns=["rid", "pool_tan_pre", "pool_tan_fin", "min_before"])
    m = pd.read_csv(MASTER, encoding="utf-8-sig", usecols=["レースID(新/馬番無)", "クラス名", "トラックコード(JV)", "日付"],
                    dtype={"レースID(新/馬番無)": str})
    m = m.drop_duplicates("レースID(新/馬番無)").rename(columns={"レースID(新/馬番無)": "rid", "クラス名": "cls", "トラックコード(JV)": "trk", "日付": "date"})
    m["rid"] = m.rid.str.zfill(16)
    k = pd.read_csv(KEKKA, encoding="cp932", usecols=[0, 4], dtype=str)
    k.columns = ["rh", "tan"]
    k["rid"] = k.rh.str[:16]; k["ban"] = k.rh.str[16:18].astype(int)
    k["tan_pay"] = pd.to_numeric(k.tan, errors="coerce")  # 勝ち馬だけ数値、他は (オッズ)
    k = k.dropna(subset=["tan_pay"])[["rid", "ban", "tan_pay"]].drop_duplicates(["rid", "ban"])

    d = d.drop(columns=["tan_pay"]).merge(k, on=["rid", "ban"], how="left")
    d["tan_pay"] = d.tan_pay.fillna(0.0)
    d = d.sort_values(["rid", "ban"]).reset_index(drop=True)
    g = d.groupby("rid")
    r = pd.DataFrame({"k": g.size(), "w": g.is_win.sum(), "ovr": g.o_fin.apply(lambda x: (1 / x).sum()),
                      "year": g.year.first()})
    r = r.join(tf.set_index("rid")).join(m.set_index("rid")[["cls", "trk", "date"]])
    r = r[(r.k >= 8) & (r.w == 1) & r.ovr.between(1.20, 1.33) & (r.trk < 51) & r.pool_tan_pre.gt(0) & r.cls.notna()]
    d = d[d.rid.isin(r.index)].reset_index(drop=True)
    return d, r


def build_regimes(d: pd.DataFrame, r: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = d.copy()
    for c, s in (("o_fin", "pf"), ("o_pre", "pp"), ("o_win9", "p9")):
        inv = 1 / d[c]
        d[s] = inv / inv.groupby(d.rid).transform("sum")
    d["kl_late"] = d.pf * np.log(d.pf / d.pp)
    d["kl_early"] = d.pp * np.log(d.pp / d.p9)
    agg = d.groupby("rid").agg(kl_late=("kl_late", "sum"), kl_early=("kl_early", "sum"))
    r = r.join(agg)
    r["date"] = r.index.str[:8]
    r["dt"] = pd.to_datetime(r.date, format="%Y%m%d")
    # POOL: 直前 365 日 (当日除く) の中央値との差
    r["lpool"] = np.log(r.pool_tan_fin)
    daily = r.groupby("dt").lpool.apply(list).sort_index()
    dates = daily.index.values
    allv = np.concatenate(daily.values); cnt = np.cumsum([0] + [len(v) for v in daily.values])
    ref = {}
    for i, dt in enumerate(dates):
        lo = np.searchsorted(dates, dt - np.timedelta64(365, "D"), side="left")
        if dt - dates[0] < np.timedelta64(365, "D"):
            ref[dt] = np.nan; continue
        ref[dt] = np.median(allv[cnt[lo]:cnt[i]])
    r["POOL"] = r.lpool - r.dt.map(ref)
    # LATE: log(fin/pre) を min_before で 2013-16 回帰した残差
    r["late_raw"] = np.log(r.pool_tan_fin / r.pool_tan_pre)
    cal = r[r.year <= 2016]
    b1, b0 = np.polyfit(cal.min_before, cal.late_raw, 1)
    r["LATE"] = r.late_raw - (b0 + b1 * r.min_before)
    r["LINFO"] = np.log(r.kl_late.clip(lower=1e-6))
    r["EINFO"] = np.log(r.kl_early.clip(lower=1e-6))
    gmap = {"Ｇ１": "G1", "Ｇ２": "重賞", "Ｇ３": "重賞", "重賞": "重賞", "ｵｰﾌﾟﾝ": "OP", "OP(L)": "OP", "3勝": "OP", "1600万": "OP",
            "1勝": "条件", "2勝": "条件", "500万": "条件", "1000万": "条件", "未勝利": "未勝利", "新馬": "新馬"}
    r["GRADE"] = r.cls.map(gmap)
    r = r[r.GRADE.notna()]
    g1days = set(r.loc[r.GRADE == "G1", "date"])
    r["G1DAY"] = np.where(r.GRADE.isin(["G1", "重賞"]), "対象外(重賞)", np.where(r.date.isin(g1days), "G1の日", "G1なしの日"))
    r["FIELD"] = pd.cut(r.k, [7, 11, 15, 18], labels=["8-11", "12-15", "16-18"]).astype(str)
    cuts = {}
    for c in ("POOL", "LATE", "LINFO", "EINFO"):
        q = r.loc[(r.year <= 2016) & r[c].notna(), c].quantile([1 / 3, 2 / 3]).values
        cuts[c] = [float(q[0]), float(q[1])]
        r[c + "_T"] = pd.cut(r[c], [-np.inf, q[0], q[1], np.inf], labels=["T1低", "T2中", "T3高"]).astype(str).where(r[c].notna())
    r.attrs["cuts"] = cuts
    r.attrs["late_fit"] = [float(b0), float(b1)]
    d = d[d.rid.isin(r.index)].reset_index(drop=True)
    return d, r


# ----------------------------------------------------------------------------- conditional logit
class CL:
    """レース単位で並んだ馬の配列。β_r = Z_r·θ の条件付きロジット (重み付き) を Newton で解く。"""

    def __init__(self, d: pd.DataFrame, r: pd.DataFrame, pcol: str):
        d = d[d.rid.isin(r.index)]
        self.rids = r.index.values
        pos = pd.Series(np.arange(len(self.rids)), index=self.rids)
        d = d.assign(ri=d.rid.map(pos).values).sort_values(["ri", "ban"])
        self.ri = d.ri.values
        self.x = np.log(d[pcol].values)
        self.win = d.is_win.values.astype(float)
        self.starts = np.r_[0, np.flatnonzero(np.diff(self.ri)) + 1]
        self.xw = np.add.reduceat(self.x * self.win, self.starts)

    def fit(self, Z: np.ndarray, w: np.ndarray, theta0=None, iters=30):
        K = Z.shape[1]
        th = np.zeros(K) if theta0 is None else theta0.copy()
        if theta0 is None:
            th[0] = 1.0
        for _ in range(iters):
            b = Z @ th
            u = b[self.ri] * self.x
            umax = np.maximum.reduceat(u, self.starts)
            e = np.exp(u - umax[self.ri])
            s0 = np.add.reduceat(e, self.starts)
            s1 = np.add.reduceat(e * self.x, self.starts)
            s2 = np.add.reduceat(e * self.x * self.x, self.starts)
            m1 = s1 / s0; var = s2 / s0 - m1 ** 2
            gr = Z.T @ (w * (self.xw - m1))
            H = (Z * (w * var)[:, None]).T @ Z
            step = np.linalg.solve(H + 1e-9 * np.eye(K), gr)
            th = th + step
            if np.max(np.abs(step)) < 1e-6:
                break
        return th


def cluster_weights(dates: np.ndarray, rng: np.random.Generator):
    u, inv = np.unique(dates, return_inverse=True)
    pick = rng.integers(0, len(u), len(u))
    cnt = np.bincount(pick, minlength=len(u)).astype(float)
    return cnt[inv]


def ci(a):
    a = np.asarray(a)
    return [round(float(np.nanpercentile(a, 2.5)), 4), round(float(np.nanpercentile(a, 97.5)), 4)]


# ----------------------------------------------------------------------------- analyses
def regime_beta(cl_f: CL, cl_p: CL, r: pd.DataFrame, col: str, levels: list[str], rng, ref=None):
    """各レベルの β_fin, β_pre (レベルダミーの同時推定 = レベル別に別々に推定するのと同値)。"""
    rr = r.loc[cl_f.rids]
    lv = rr[col].values
    ok = np.isin(lv, levels)
    Z = np.stack([(lv == L).astype(float) for L in levels], 1)
    dates = rr.date.values
    w0 = ok.astype(float)
    tf = cl_f.fit(Z, w0); tp = cl_p.fit(Z, w0)
    bf, bp = [], []
    for _ in range(B):
        w = cluster_weights(dates, rng) * w0
        bf.append(cl_f.fit(Z, w, tf, iters=8)); bp.append(cl_p.fit(Z, w, tp, iters=8))
    bf, bp = np.array(bf), np.array(bp)
    out = {"levels": {}}
    for i, L in enumerate(levels):
        out["levels"][L] = {"n_races": int((lv == L).sum()), "beta_fin": round(float(tf[i]), 4), "ci_fin": ci(bf[:, i]),
                            "beta_pre": round(float(tp[i]), 4), "ci_pre": ci(bp[:, i]),
                            "late_minus_pre": round(float(tf[i] - tp[i]), 4), "ci_late_minus_pre": ci(bf[:, i] - bp[:, i])}
    if ref is not None:
        hi, lo = ref
        i, j = levels.index(hi), levels.index(lo)
        out["contrast"] = {"def": f"{hi} − {lo}", "d_beta_fin": round(float(tf[i] - tf[j]), 4), "ci_fin": ci(bf[:, i] - bf[:, j]),
                           "d_beta_pre": round(float(tp[i] - tp[j]), 4), "ci_pre": ci(bp[:, i] - bp[:, j])}
    return out


def joint_model(cl: CL, r: pd.DataFrame, std: dict, rng):
    rr = r.loc[cl.rids]
    cols, Z = ["const"], [np.ones(len(rr))]
    for c in ("POOL", "LATE", "LINFO", "EINFO"):
        cols.append(c); Z.append(((rr[c] - std[c][0]) / std[c][1]).values)
    cols.append("logN"); Z.append(((np.log(rr.k) - std["logN"][0]) / std["logN"][1]).values)
    for g in ("G1", "重賞", "新馬", "未勝利"):
        cols.append("GRADE=" + g); Z.append((rr.GRADE == g).astype(float).values)
    cols.append("G1DAY"); Z.append((rr.G1DAY == "G1の日").astype(float).values)
    Z = np.stack(Z, 1)
    ok = np.isfinite(Z).all(1)
    Z = np.where(np.isfinite(Z), Z, 0.0)
    w0 = ok.astype(float)
    th = cl.fit(Z, w0)
    bs = np.array([cl.fit(Z, cluster_weights(rr.date.values, rng) * w0, th, iters=8) for _ in range(B)])
    return {"n_races": int(ok.sum()), "coef": {c: {"theta": round(float(th[i]), 4), "ci": ci(bs[:, i])} for i, c in enumerate(cols)}}


def fav_roi(d: pd.DataFrame, r: pd.DataFrame, col: str, levels: list[str], rng):
    fav = d.sort_values(["rid", "o_fin", "ban"]).drop_duplicates("rid")
    fav = fav.join(r[[col, "date"]], on="rid")
    res = {}
    for L in levels:
        s = fav[fav[col] == L]
        cells = {"win_fav_le1.9": (s[s.o_fin <= 1.9], "tan_pay"), "place_fav_le1.9": (s[s.o_fin <= 1.9], "fuku_pay"),
                 "place_fav_1.0-1.4": (s[s.o_fin <= 1.4], "fuku_pay")}
        res[L] = {}
        for name, (x, pc) in cells.items():
            if len(x) < 30:
                res[L][name] = {"n": int(len(x))}; continue
            g = x.groupby("date")[pc].agg(["sum", "size"])
            sv, nv = g["sum"].values, g["size"].values
            bs = []
            for _ in range(B):
                p = rng.integers(0, len(g), len(g))
                bs.append(sv[p].sum() / nv[p].sum())
            res[L][name] = {"n": int(len(x)), "roi": round(float(sv.sum() / nv.sum()), 2), "ci": [round(float(np.percentile(bs, 2.5)), 1), round(float(np.percentile(bs, 97.5)), 1)],
                            "hit": round(float((x[pc] > 0).mean()), 3)}
    return res


def flb_gap(d: pd.DataFrame, r: pd.DataFrame, col: str, levels: list[str], rng):
    """モデルを使わない FLB 指標: 単勝回収率 (確定 <5 倍) − 単勝回収率 (確定 ≥50 倍)。正 = 本命有利 = FLB。"""
    x = d[["rid", "o_fin", "tan_pay"]].join(r[[col, "date"]], on="rid")
    x["band"] = np.where(x.o_fin < 5, "fav", np.where(x.o_fin >= 50, "long", "mid"))
    res = {}
    for L in levels:
        s = x[(x[col] == L) & (x.band != "mid")]
        g = s.pivot_table(index="date", columns="band", values="tan_pay", aggfunc=["sum", "size"], fill_value=0)
        if g.shape[1] < 4:
            continue
        S, N = g["sum"][["fav", "long"]].values, g["size"][["fav", "long"]].values
        f = lambda idx: S[idx, 0].sum() / max(N[idx, 0].sum(), 1) - S[idx, 1].sum() / max(N[idx, 1].sum(), 1)
        allidx = np.arange(len(g))
        bs = [f(rng.integers(0, len(g), len(g))) for _ in range(B)]
        res[L] = {"roi_fav_lt5": round(float(S[:, 0].sum() / N[:, 0].sum()), 2), "roi_long_ge50": round(float(S[:, 1].sum() / N[:, 1].sum()), 2),
                  "gap": round(float(f(allidx)), 2), "ci": [round(float(np.percentile(bs, 2.5)), 1), round(float(np.percentile(bs, 97.5)), 1)],
                  "n_fav": int(N[:, 0].sum()), "n_long": int(N[:, 1].sum())}
    return res


def main() -> int:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    d, r = load()
    d, r = build_regimes(d, r)
    print(f"races {len(r):,} horses {len(d):,}  ({time.time()-t0:.0f}s)", flush=True)
    sanity = {"winner_tan_pay_missing": float((d.loc[d.is_win == 1, "tan_pay"] <= 0).mean()),
              "win_roi_all": float(d.tan_pay.sum() / len(d)), "place_roi_all": float(d.fuku_pay.sum() / len(d)),
              "tan_pay_vs_odds_median_ratio": float((d.loc[d.is_win == 1, "tan_pay"] / (100 * d.loc[d.is_win == 1, "o_fin"])).median())}
    print(sanity, flush=True)
    std = {c: [float(r.loc[r.year <= 2016, c].mean()), float(r.loc[r.year <= 2016, c].std())] for c in ("POOL", "LATE", "LINFO", "EINFO")}
    lk = np.log(r.loc[r.year <= 2016, "k"]); std["logN"] = [float(lk.mean()), float(lk.std())]
    out = {"preregistered_cuts_2013_16": r.attrs["cuts"], "late_fit_b0_b1": r.attrs["late_fit"], "std_2013_16": std,
           "n_races_total": int(len(r)), "sanity": sanity, "periods": {}}
    # 代理変数の相関 (2013-16)
    cc = r.loc[r.year <= 2016, ["POOL", "LATE", "LINFO", "EINFO", "k"]].assign(G1=(r.GRADE == "G1").astype(float)).corr(method="spearman")
    out["proxy_spearman_2013_16"] = cc.round(3).to_dict()
    specs = [("POOL_T", ["T1低", "T2中", "T3高"], ("T3高", "T1低")),
             ("LATE_T", ["T1低", "T2中", "T3高"], ("T3高", "T1低")),
             ("LINFO_T", ["T1低", "T2中", "T3高"], ("T3高", "T1低")),
             ("EINFO_T", ["T1低", "T2中", "T3高"], ("T3高", "T1低")),
             ("GRADE", ["G1", "重賞", "OP", "条件", "未勝利", "新馬"], ("G1", "条件")),
             ("G1DAY", ["G1の日", "G1なしの日"], ("G1の日", "G1なしの日")),
             ("FIELD", ["8-11", "12-15", "16-18"], ("16-18", "8-11"))]
    for pname, (y0, y1) in PERIODS.items():
        rp = r[(r.year >= y0) & (r.year <= y1)]
        dp = d[d.rid.isin(rp.index)]
        clf, clp = CL(dp, rp, "pf"), CL(dp, rp, "pp")
        po = {"n_races": int(len(rp))}
        th = clf.fit(np.ones((len(rp), 1)), np.ones(len(rp))); thp = clp.fit(np.ones((len(rp), 1)), np.ones(len(rp)))
        bs = [(clf.fit(np.ones((len(rp), 1)), w, th, 8)[0], clp.fit(np.ones((len(rp), 1)), w, thp, 8)[0])
              for w in (cluster_weights(rp.loc[clf.rids].date.values, rng) for _ in range(B))]
        bs = np.array(bs)
        po["overall"] = {"beta_fin": round(float(th[0]), 4), "ci_fin": ci(bs[:, 0]), "beta_pre": round(float(thp[0]), 4), "ci_pre": ci(bs[:, 1])}
        print(pname, po["overall"], flush=True)
        po["regimes"] = {}
        for col, lv, ref in specs:
            po["regimes"][col] = regime_beta(clf, clp, rp, col, lv, rng, ref)
            po["regimes"][col]["roi"] = fav_roi(dp, rp, col, lv, rng)
            po["regimes"][col]["flb_gap"] = flb_gap(dp, rp, col, lv, rng)
            print(pname, col, po["regimes"][col].get("contrast"), f"({time.time()-t0:.0f}s)", flush=True)
        po["joint_fin"] = joint_model(clf, rp, std, rng)
        po["joint_pre"] = joint_model(clp, rp, std, rng)
        print(pname, "joint", {k: v["theta"] for k, v in po["joint_fin"]["coef"].items()}, flush=True)
        out["periods"][pname] = po
    # 実用: 両期とも ROI ≥ 100 のセルがあるか
    hits = []
    for col, lv, _ in specs:
        for L in lv:
            for cell in ("win_fav_le1.9", "place_fav_le1.9", "place_fav_1.0-1.4"):
                a = out["periods"]["P1_2013_19"]["regimes"][col]["roi"][L].get(cell, {})
                b = out["periods"]["P2_2020_25"]["regimes"][col]["roi"][L].get(cell, {})
                if "roi" in a and "roi" in b:
                    hits.append({"regime": col, "level": L, "cell": cell, "P1": a["roi"], "P1_ci": a["ci"], "n1": a["n"],
                                 "P2": b["roi"], "P2_ci": b["ci"], "n2": b["n"], "min": min(a["roi"], b["roi"])})
    hits.sort(key=lambda h: -h["min"])
    out["roi_ranking_by_worse_period"] = hits[:15]
    # 年別の β_fin (全体) と G1 の β (安定性の確認)
    yr = {}
    for y in sorted(r.year.unique()):
        ry = r[r.year == y]; cy = CL(d[d.rid.isin(ry.index)], ry, "pf"); cp = CL(d[d.rid.isin(ry.index)], ry, "pp")
        yr[int(y)] = {"beta_fin": round(float(cy.fit(np.ones((len(ry), 1)), np.ones(len(ry)))[0]), 4),
                      "beta_pre": round(float(cp.fit(np.ones((len(ry), 1)), np.ones(len(ry)))[0]), 4), "n": int(len(ry))}
    out["beta_by_year"] = yr
    out["any_cell_ge100_both_periods"] = any(h["min"] >= 100 for h in hits)
    OUTDIR.mkdir(exist_ok=True)
    (OUTDIR / "parimutuel_equilibrium_exp.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"done {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
