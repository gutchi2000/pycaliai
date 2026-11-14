# -*- coding: utf-8 -*-
"""
late_money_decomp_exp.py — 遅い資金の中身の分解 (2026-11-14)
=====================================================================
問い: 10 時 → 締切のあいだに市場へ入る情報 (勝ち馬 NLL で約 0.079 nats/レース) は何でできているか。

実行 (worktree の root から):
    PYTHONUTF8=1 E:/PyCaLiAI/venv311/Scripts/python.exe lab/experiments/late_money_decomp_exp.py
出力: reports/late_money_decomp_exp.json (集計値だけ) / reports/late_panel.parquet (作業用。JRDB 値を含むので reports の外へ出さない)

---------------------------------------------------------------- 事前登録 (test を見る前に固定)
■ 時点 (何時に手に入るか)
  v7      : 前日まで (v7 は前日までの情報だけで作る)
  prev23  : 前日 23:00 前後の単勝オッズ (TANPUK 時系列 区分 1 の前日行)
  am9     : 当日 8:54〜9:11 の単勝オッズ (TANPUK 区分 1 の当日 1 本目)
  j10     : JRDB SED の「10 時単勝オッズ」(アーカイブ値。記録時刻は未検証)
  bw      : 当日馬体重 (発走の約 70 分前に発表。値は JRDB SED の bw、前走馬体重は master)
  pre28   : 発走 25〜30 分前の単勝オッズ (TANPUK 区分 1 の当日 2 本目)
  tyb17   : JRDB TYB の単勝オッズ (odds_time。発走の中央値 17 分前)
  close   : 確定単勝オッズ
  各時点の市場の log 勝率 = そのレースで実際に走った馬の 1/オッズ を正規化したもの
  (取消馬を後から除く点は全時点に共通の小さな先読み)。

■ 対象レース
  ALL : gate_base の valid(2023)+test(2024-25) のうち、上の 6 時点のオッズが出走全馬に揃い、
        TYB の odds_time が発走の 5〜60 分前のレース。
  S11 : ALL のうち発走 11:00 以降 (10 時オッズが発走の 60 分以上前 = 時点の順序が必ず
        am9 < j10 < pre28 < tyb17 < close になる)。発走が 10 時台以前のレースでは j10 が pre28 より
        後ろに来るので、Q2 (分解)・Q3 (予測)・Q4 (朝の補正印) の見出しは S11、Q1 の時間表は ALL と S11 の両方。
        (この線引きはオッズ同士の距離だけを見て決めた。着順は見ていない)

■ 当てはめ
  すべて条件付きロジット (gc._fit, l2=1e-4) を valid 2023 で fit、test 2024-25 で 1 回評価。
  指標 = 勝ち馬 NLL (nats/レース)、◎ (確率最大。同率は馬番の小さい方) の 1 着率・3 着内率。
  CI は開催日クラスタ bootstrap 2000 回の 95%。

■ Q1 時間表 : 各時点 s について raw (1/オッズそのまま)、[lk_s] (べき乗較正)、[lk_s + v7]。
   「遅い」の判定 H1: G = NLL[j10] − NLL[close] のうち pre28 以降の分 (NLL[pre28] − NLL[close]) の割合。
   CI 下端 > 50% なら「最後の 28 分に集中」、CI 上端 < 50% なら「28 分より前」、他は「どちらとも言えない」。
■ Q2 分解 (S11 が見出し) : G = NLL[j10] − NLL[close] を
     (a) 公開成績への追いつき = NLL[j10] − NLL[j10+v7]
     (b) 馬体重             = NLL[j10+v7] − NLL[j10+v7+bw]     bw = {増減, |増減|, 増減², 体重, 前走体重なし}
     (c) 残り               = NLL[j10+v7+bw] − NLL[close]
   の 3 つに分け、割合 a/G, b/G, c/G と CI を出す。順序を逆 (bw を先) にした値も併記。
   H2: (c) の CI 下端 > 50% → 「大半は v7 にも馬体重にも無い当日/非公開の情報」。
       (a) の CI 下端 > 50% → 「大半は公開成績への追いつき」。他は「混合」。
   補助: 1 頭ごとの Δ = lk_close − lk_j10 を (lm7 − lk_j10) に回帰した傾きと R² (valid で fit、test で評価)。
■ Q3 遅い資金の予測 (S11) : 目的変数 = レース内で平均を引いた Δ。10 時までの情報だけで
     L0 [lk_j10, lk_j10²] / L1 +(lm7−lk_j10) / L2 +(j10−am9 の動き, am9−prev23 の動き)
     L3 +騎手・調教師の癖 (2013-2022 の Δ のレース内偏差の縮小平均、k=200。test より前で凍結)
     GBM (LightGBM 400 本, lr 0.03, leaves 15, min_child 200, 4 thread) = L3 の列 + 10 時人気順位・頭数・発走時刻
   test の R² (開催日 bootstrap CI) と、「締切までに売れる」(Δ ≥ +0.15) の AUC。
   H3: 最良モデルの test R² ≥ 0.05 →「予測できる」、< 0.02 →「ほぼ予測できない」、あいだは「弱い」。
   馬券への利用は検討しない (払戻は確定オッズ。steam/drift 選別は既に 2 回、控除率どおりで閉鎖済み)。回収率セル探索はしない。
■ Q4 朝の補正印 (S11 が見出し、ALL 併記) : v7 / +prev23 / +am9 / +j10 / +pre28 / +tyb17 / +close の ◎1着・◎3着内。
   H4: [j10+v7] の ◎3着内 − v7 の CI 下端 > 0 → 朝の補正印は有効。
   朝で取れる割合 = (X[j10+v7] − X[v7]) / (X[tyb17+v7] − X[v7])、X = ◎3着内 と NLL の両方 (T−10 は pre28 と close のあいだ。
   tyb17 を T−10 の代わりに使う)。参考に v6 でも同じ表 (以前の T−10 補正印の測定は v6: 61.7→65.1%)。
   追加: [j10 + v7 + 朝の動き + 騎手/調教師の癖] が [j10 + v7] を上回るか (NLL 差の CI 上端 < 0 で PASS)。
■ 引用のみ・やり直さない: EXP19 (当日馬体重は T−28 市場 + 特徴の上に Δ −0.00017 nats、CI は 0 を跨ぐ)。
   パドック/TYB 指数が締切市場を超えるかは別の実験の担当。ここでは TYB のオッズと時刻だけを使う。
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate_common as gc  # noqa: E402

ROOT = Path("E:/PyCaLiAI")
REP = gc.REP
OUT_JSON = REP / "late_money_decomp_exp.json"
PANEL = REP / "late_panel.parquet"
TANPUK = ROOT / "data" / "Time _series_odds" / "TANPUK_20210105-20251228.csv"
TYB = ROOT / "data" / "jrdb" / "tyb.parquet"
N_BOOT = 2000
L2 = 1e-4
STAGES = ["prev23", "am9", "j10", "pre28", "tyb17", "close"]
BW = ["bw_d", "bw_abs", "bw_sq", "bw_lv", "bw_noprev"]
log = gc.log


def hm_to_min(s):
    s = pd.to_numeric(s, errors="coerce")
    return (s // 100) * 60 + s % 100


# ------------------------------------------------------------------ panel
def load_tanpuk():
    tp = pd.read_csv(TANPUK, encoding="cp932", low_memory=False)
    c = list(tp.columns)
    tp["rid_s"] = tp[c[0]].astype(str).str.replace(r"[^0-9]", "", regex=True).str[:16]
    tp = tp[tp.rid_s.str[:4].isin(["2023", "2024", "2025"])].copy()
    tm = pd.to_numeric(tp[c[2]], errors="coerce").fillna(0).astype(np.int64)
    tp["md"] = tm // 10000
    tp["minute"] = ((tm % 10000) // 100) * 60 + (tm % 100)
    tp["kb"] = pd.to_numeric(tp[c[1]], errors="coerce")
    tp["rmd"] = tp.rid_s.str[4:8].astype(int)
    tp["pool"] = pd.to_numeric(tp[c[4]], errors="coerce")
    tcols = {int(str(x).strip()[:-1]): x for x in c if str(x).strip().endswith("単") and str(x).strip()[:-1].isdigit()}
    k1 = tp[tp.kb == 1]
    same = k1[k1.md == k1.rmd].sort_values(["rid_s", "minute"])
    rows = {"prev23": k1[k1.md != k1.rmd].sort_values(["rid_s", "md", "minute"]).drop_duplicates("rid_s", keep="last"),
            "am9": same.drop_duplicates("rid_s", keep="first"),
            "pre28": same.drop_duplicates("rid_s", keep="last"),
            "fin": tp[tp.kb == 4].drop_duplicates("rid_s", keep="last")}
    # am9 と pre28 が同じ行になるレース (当日 1 本しか無い) は pre28 を欠損扱い
    one = same.groupby("rid_s").size()
    rows["pre28"] = rows["pre28"][~rows["pre28"].rid_s.isin(one[one < 2].index)]
    longs, meta = {}, {}
    for k, r in rows.items():
        m = r.set_index("rid_s")[[tcols[i] for i in sorted(tcols)]].apply(pd.to_numeric, errors="coerce")
        m.columns = sorted(tcols)
        s = m.stack().rename(f"o_{k}").reset_index()
        s.columns = ["rid_s", "ban", f"o_{k}"]
        longs[k] = s
        meta[k] = r.set_index("rid_s")[["minute", "pool"]].add_suffix(f"_{k}")
    return longs, pd.concat(meta.values(), axis=1)


def norm_lk(d, col):
    """レース内の全馬が正のオッズを持つレースだけ log 勝率にする。"""
    ok = d.groupby("rid_s")[col].transform(lambda s: s.notna().all() and (s > 0).all())
    inv = 1.0 / d[col].where(ok)
    return np.log(inv / inv.groupby(d["rid_s"]).transform("sum"))


def jt_encoding(lite):
    """2013-2022 の Δ (確定 − 10 時、レース内偏差) を騎手・調教師ごとに縮小平均 (k=200)。valid/test より前で凍結。"""
    sed = gc.load_sed(["racekey", "ymd", "ban", "tan_odds", "odds10_tan"])
    sed = sed[(sed.ymd >= 20130101) & (sed.ymd <= 20221231) & (sed.tan_odds > 0) & (sed.odds10_tan > 0)].copy()
    for c, o in (("lkf", "tan_odds"), ("lk0", "odds10_tan")):
        inv = 1.0 / sed[o]
        sed[c] = np.log(inv / inv.groupby(sed["rid_s"]).transform("sum"))
    sed["dl"] = sed.lkf - sed.lk0
    sed["dl"] = sed.dl - sed.groupby("rid_s").dl.transform("mean")
    sed = sed.merge(lite[["rid_s", "ban", "jk", "tr"]], on=["rid_s", "ban"], how="inner")
    out = {}
    for k in ("jk", "tr"):
        g = sed.groupby(k).dl.agg(["sum", "size"])
        out[k] = (g["sum"] / (g["size"] + 200.0)).rename(f"{k}_enc")
    return out, int(len(sed))


def build_panel():
    b = gc.load_base()
    lite = gc.load_master_lite()[["レースID(新/馬番無)", "馬番", "発走時刻", "前走馬体重", "騎手コード", "調教師コード"]].copy()
    lite["rid_s"] = lite["レースID(新/馬番無)"].astype(str).str[:16]
    lite["ban"] = pd.to_numeric(lite["馬番"], errors="coerce")
    lite = lite.dropna(subset=["ban"])
    lite["ban"] = lite["ban"].astype(int)
    h = lite["発走時刻"].astype(str).str.split(":", expand=True)
    lite["post"] = pd.to_numeric(h[0], errors="coerce") * 60 + pd.to_numeric(h[1], errors="coerce")
    lite = lite.rename(columns={"前走馬体重": "bw_prev", "騎手コード": "jk", "調教師コード": "tr"}).drop_duplicates(["rid_s", "ban"])
    enc, n_enc = jt_encoding(lite)
    d = b.merge(lite[["rid_s", "ban", "post", "bw_prev", "jk", "tr"]], on=["rid_s", "ban"], how="left")
    d = d.join(enc["jk"], on="jk").join(enc["tr"], on="tr")
    longs, meta = load_tanpuk()
    for k in ("prev23", "am9", "pre28", "fin"):
        d = d.merge(longs[k], on=["rid_s", "ban"], how="left")
    d = d.join(meta, on="rid_s")
    sed = gc.load_sed(["racekey", "ymd", "ban", "bw"])
    sed = sed[sed.ymd >= 20230101]
    tyb = pd.read_parquet(TYB, columns=["racekey", "ban", "tan_odds", "odds_time"]).rename(columns={"tan_odds": "o_tyb17"})
    sed = sed.merge(tyb, on=["racekey", "ban"], how="left")
    sed["tyb_min"] = hm_to_min(sed["odds_time"])
    d = d.merge(sed[["rid_s", "ban", "bw", "o_tyb17", "tyb_min"]].drop_duplicates(["rid_s", "ban"]), on=["rid_s", "ban"], how="left")
    d = d.sort_values(["rid_s", "ban"]).reset_index(drop=True)
    d["lk_prev23"] = norm_lk(d, "o_prev23")
    d["lk_am9"] = norm_lk(d, "o_am9")
    d["lk_j10"] = d["lk10"]
    d["lk_pre28"] = norm_lk(d, "o_pre28")
    d["lk_tyb17"] = norm_lk(d, "o_tyb17")
    d["lk_close"] = d["lk"]
    d["tyb_gap"] = d["post"] - d.groupby("rid_s")["tyb_min"].transform("first")
    d["pre_gap"] = d["post"] - d["minute_pre28"]
    att = {"base_races": {s: int(d[d.split == s].rid_s.nunique()) for s in ("valid", "test")}}
    for s in STAGES:
        att[f"has_{s}"] = int(d[d[f"lk_{s}"].notna()].rid_s.nunique())
    ok = d[[f"lk_{s}" for s in STAGES]].notna().all(axis=1) & d["tyb_gap"].between(5, 60) & d["post"].notna()
    ok = ok.groupby(d["rid_s"]).transform("all")
    d = d[ok].reset_index(drop=True)
    att["common_races"] = {s: int(d[d.split == s].rid_s.nunique()) for s in ("valid", "test")}
    att["jt_encoding_rows_2013_2022"] = n_enc
    # 馬体重 (発走の約 70 分前)
    d["bw_d"] = d["bw"] - d["bw_prev"]
    d["bw_abs"] = d["bw_d"].abs()
    d["bw_sq"] = d["bw_d"] ** 2
    d["bw_lv"] = d["bw"]
    d["bw_noprev"] = d["bw_prev"].isna().astype(float)
    # 10 時までの情報
    d["d7"] = d["lm7"] - d["lk_j10"]
    d["m910"] = d["lk_j10"] - d["lk_am9"]
    d["m239"] = d["lk_am9"] - d["lk_prev23"]
    d["lk10sq"] = d["lk_j10"] ** 2
    d["n_run"] = d.groupby("rid_s")["ban"].transform("size")
    d["rank10"] = d.groupby("rid_s")["odds10"].rank(method="average") / d["n_run"]
    d["delta"] = d["lk_close"] - d["lk_j10"]
    d["delta_late"] = d["lk_close"] - d["lk_pre28"]
    d["s11"] = d["post"] >= 660
    d.to_parquet(PANEL)
    return d, att


# ------------------------------------------------------------------ helpers
def zstd(d, cols):
    """レース内平均を引く (NaN はレース平均 → 0)、valid の sd で割る。"""
    out = d.copy()
    for c in cols:
        x = pd.to_numeric(out[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        x = x.fillna(x.groupby(out["rid_s"]).transform("mean")).fillna(0.0)
        x = x - x.groupby(out["rid_s"]).transform("mean")
        sd = x[out.split == "valid"].std()
        out["z_" + c] = x / (sd if sd and sd > 0 else 1.0)
    return out


class Block:
    """valid で fit、test の 1 レースごとの (nll, ◎1着, ◎3着内) を持つ。"""

    def __init__(self, d):
        self.v = d[d.split == "valid"].reset_index(drop=True)
        self.t = d[d.split == "test"].reset_index(drop=True)
        self.cv = pd.factorize(self.v.rid_s)[0]
        self.ct = pd.factorize(self.t.rid_s)[0]
        self.nv, self.nt = self.cv.max() + 1, self.ct.max() + 1
        self.wv = np.flatnonzero((self.v.jyun == 1).values)
        self.wt = np.flatnonzero((self.t.jyun == 1).values)
        first = np.r_[0, np.flatnonzero(np.diff(self.ct)) + 1]
        dates = self.t.date.values[first]
        self.ud, self.dc = np.unique(dates, return_inverse=True)
        rng = np.random.default_rng(0)
        self.I = rng.integers(0, len(self.ud), (N_BOOT, len(self.ud)))
        self.cnt = np.bincount(self.dc, minlength=len(self.ud)).astype(float)
        self.cntB = self.cnt[self.I].sum(1)
        self.M, self.W = {}, {}
        self.M["uniform"] = self._score(np.zeros(len(self.t)))

    def _score(self, x):
        lp = gc.race_logsoftmax(x, self.ct, self.nt)
        nll = -lp[self.wt]
        order = np.lexsort((self.t.ban.values, -lp, self.ct))
        top = order[np.r_[0, np.flatnonzero(np.diff(self.ct[order])) + 1]]
        jy = self.t.jyun.values[top]
        return nll, (jy == 1).astype(float), (jy <= 3).astype(float)

    def add(self, name, cols, fit=True):
        if fit:
            w = gc._fit(self.v[cols].values.astype(float), self.cv, self.nv, self.wv, L2)
        else:
            w = np.ones(len(cols))
        self.W[name] = {c: float(x) for c, x in zip(cols, w)}
        self.M[name] = self._score(self.t[cols].values.astype(float) @ w)

    # --- bootstrap (開催日クラスタ)
    def _bm(self, a):
        s = np.bincount(self.dc, weights=a, minlength=len(self.ud))
        return s[self.I].sum(1) / self.cntB

    def stat(self, f, *arrs):
        """f(各配列の平均…) の点推定と CI95"""
        pt = float(f(*[a.mean() for a in arrs]))
        bs = f(*[self._bm(a) for a in arrs])
        return {"est": pt, "ci95": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]}

    def row(self, name):
        nll, w1, t3 = self.M[name]
        return {"nll": self.stat(lambda x: x, nll), "hon_win": self.stat(lambda x: 100 * x, w1),
                "hon_top3": self.stat(lambda x: 100 * x, t3), "w": self.W.get(name)}

    def diff(self, a, b, k=0, scale=1.0):
        """M[a] − M[b] (k: 0=nll 1=◎1着 2=◎3着内)"""
        return self.stat(lambda x, y: scale * (x - y), self.M[a][k], self.M[b][k])


def r2_boot(y, pred, dates, seed=0):
    se = (y - pred) ** 2
    st = y ** 2
    ud, dc = np.unique(dates, return_inverse=True)
    a = np.bincount(dc, weights=se, minlength=len(ud))
    b = np.bincount(dc, weights=st, minlength=len(ud))
    I = np.random.default_rng(seed).integers(0, len(ud), (N_BOOT, len(ud)))
    bs = 1 - a[I].sum(1) / b[I].sum(1)
    return {"est": float(1 - se.sum() / st.sum()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}


def demean(d, col):
    x = d[col].astype(float)
    x = x.fillna(x.groupby(d["rid_s"]).transform("mean")).fillna(0.0)
    return (x - x.groupby(d["rid_s"]).transform("mean")).values


# ------------------------------------------------------------------ analyses
def run_block(d, tag):
    log(f"\n===== {tag}: valid {d[d.split == 'valid'].rid_s.nunique():,}R / test {d[d.split == 'test'].rid_s.nunique():,}R =====")
    d = zstd(d, BW + ["m910", "m239", "jk_enc", "tr_enc"])
    zbw = ["z_" + c for c in BW]
    zmorn = ["z_m910", "z_m239", "z_jk_enc", "z_tr_enc"]
    B = Block(d)
    res = {"n_valid_races": int(B.nv), "n_test_races": int(B.nt), "n_test_days": int(len(B.ud))}
    B.add("v7", ["lm7"])
    B.add("v6", ["lm6"])
    for s in STAGES:
        B.add(f"raw_{s}", [f"lk_{s}"], fit=False)
        B.add(f"mk_{s}", [f"lk_{s}"])
        B.add(f"bl_{s}", [f"lk_{s}", "lm7"])
    for s in ("j10", "tyb17", "close"):
        B.add(f"bl6_{s}", [f"lk_{s}", "lm6"])
    B.add("j10_bw", ["lk_j10"] + zbw)
    B.add("j10_v7_bw", ["lk_j10", "lm7"] + zbw)
    B.add("pre28_v7_bw", ["lk_pre28", "lm7"] + zbw)
    B.add("close_v7_bw", ["lk_close", "lm7"] + zbw)
    B.add("j10_v7_morn", ["lk_j10", "lm7"] + zmorn)
    B.add("am9_v7_morn", ["lk_am9", "lm7", "z_m239", "z_jk_enc", "z_tr_enc"])

    # ---- Q1 時間表
    tl = {"uniform": B.row("uniform"), "v7": B.row("v7")}
    for s in STAGES:
        tl[s] = {"raw": B.row(f"raw_{s}"), "market": B.row(f"mk_{s}"), "market_v7": B.row(f"bl_{s}"),
                 "v7_adds_nll": B.diff(f"bl_{s}", f"mk_{s}"),
                 "v7_adds_top3": B.diff(f"bl_{s}", f"mk_{s}", 2, 100.0),
                 "gap_to_close_nll": B.diff(f"mk_{s}", "mk_close")}
    res["timeline"] = tl
    for s in STAGES:
        r = tl[s]
        log(f"[{tag}] {s:7s} raw {r['raw']['nll']['est']:.4f}  市場 {r['market']['nll']['est']:.4f}  +v7 {r['market_v7']['nll']['est']:.4f}"
            f"  v7上乗せ {r['v7_adds_nll']['est']:+.4f} [{r['v7_adds_nll']['ci95'][0]:+.4f},{r['v7_adds_nll']['ci95'][1]:+.4f}]"
            f"  ◎3着内 市場 {r['market']['hon_top3']['est']:.2f} +v7 {r['market_v7']['hon_top3']['est']:.2f}")
    n = {s: B.M[f"mk_{s}"][0] for s in STAGES}
    G = lambda a, b: a - b  # noqa: E731
    seg = {"G_j10_to_close": B.stat(G, n["j10"], n["close"]),
           "G_raw_j10_to_close": B.stat(G, B.M["raw_j10"][0], B.M["raw_close"][0]),
           "prev23_to_am9": B.stat(G, n["prev23"], n["am9"]), "am9_to_j10": B.stat(G, n["am9"], n["j10"]),
           "j10_to_pre28": B.stat(G, n["j10"], n["pre28"]), "pre28_to_tyb17": B.stat(G, n["pre28"], n["tyb17"]),
           "tyb17_to_close": B.stat(G, n["tyb17"], n["close"]), "pre28_to_close": B.stat(G, n["pre28"], n["close"]),
           "share_after_pre28": B.stat(lambda a, b, c: (b - c) / (a - c), n["j10"], n["pre28"], n["close"]),
           "share_after_tyb17": B.stat(lambda a, b, c: (b - c) / (a - c), n["j10"], n["tyb17"], n["close"]),
           "uniform_to_close": B.stat(G, B.M["uniform"][0], n["close"]),
           "share_of_total_after_j10": B.stat(lambda u, a, c: (a - c) / (u - c), B.M["uniform"][0], n["j10"], n["close"])}
    sh = seg["share_after_pre28"]
    seg["H1_verdict"] = ("最後の28分に集中" if sh["ci95"][0] > 0.5 else "28分より前" if sh["ci95"][1] < 0.5 else "どちらとも言えない")
    res["segments"] = seg
    log(f"[{tag}] G(j10→close) {seg['G_j10_to_close']['est']:.4f}  うち pre28 以降 {100 * sh['est']:.1f}% "
        f"[{100 * sh['ci95'][0]:.1f},{100 * sh['ci95'][1]:.1f}] → {seg['H1_verdict']}")

    # ---- Q2 分解
    A, Bm, C, Ab, D = (B.M[k][0] for k in ("mk_j10", "bl_j10", "j10_v7_bw", "j10_bw", "mk_close"))
    dec = {"G": B.stat(G, A, D),
           "a_catchup_nats": B.stat(G, A, Bm), "b_bw_nats": B.stat(G, Bm, C), "c_rest_nats": B.stat(G, C, D),
           "a_share": B.stat(lambda a, b, d_: (a - b) / (a - d_), A, Bm, D),
           "b_share": B.stat(lambda b, c, a, d_: (b - c) / (a - d_), Bm, C, A, D),
           "c_share": B.stat(lambda c, a, d_: (c - d_) / (a - d_), C, A, D),
           "alt_order_bw_first": {"b_share": B.stat(lambda a, ab, d_: (a - ab) / (a - d_), A, Ab, D),
                                  "a_share": B.stat(lambda ab, c, a, d_: (ab - c) / (a - d_), Ab, C, A, D)},
           "v7_left_at_close_nats": B.diff("bl_close", "mk_close"),
           "bw_on_pre28_v7_nats": B.diff("pre28_v7_bw", "bl_pre28"),
           "bw_on_close_v7_nats": B.diff("close_v7_bw", "bl_close"),
           "w_j10_v7": B.W["bl_j10"], "w_j10_v7_bw": B.W["j10_v7_bw"]}
    dec["H2_verdict"] = ("大半は v7 にも馬体重にも無い当日/非公開の情報" if dec["c_share"]["ci95"][0] > 0.5
                         else "大半は公開成績への追いつき" if dec["a_share"]["ci95"][0] > 0.5 else "混合")
    # 補助: Δ の回帰
    v, t = d[d.split == "valid"], d[d.split == "test"]
    yv, yt = demean(v, "delta"), demean(t, "delta")
    xv, xt = demean(v, "d7"), demean(t, "d7")
    slope = float((xv * yv).sum() / (xv * xv).sum())
    dec["delta_on_d7"] = {"slope_valid": slope, "slope_test": float((xt * yt).sum() / (xt * xt).sum()),
                          "r2_test": r2_boot(yt, slope * xt, t.date.values),
                          "sd_delta_test": float(yt.std()), "sd_d7_test": float(xt.std())}
    Xv = np.column_stack([demean(v, c) for c in BW]); Xt = np.column_stack([demean(t, c) for c in BW])
    sdv = Xv.std(0); sdv[sdv == 0] = 1
    wb = np.linalg.lstsq(Xv / sdv, yv, rcond=None)[0]
    dec["delta_on_bw_r2_test"] = r2_boot(yt, (Xt / sdv) @ wb, t.date.values)
    # 補足 (事前登録したモデルから導ける量だけ): v7 を入れたあとの区間ごとの情報 = v7 に無い情報がいつ入るか
    bl = {s: B.M[f"bl_{s}"][0] for s in STAGES}
    dec["supp_non_v7_info_by_segment"] = {
        "j10_to_close": B.stat(G, bl["j10"], bl["close"]), "j10_to_pre28": B.stat(G, bl["j10"], bl["pre28"]),
        "pre28_to_tyb17": B.stat(G, bl["pre28"], bl["tyb17"]), "tyb17_to_close": B.stat(G, bl["tyb17"], bl["close"]),
        "share_after_tyb17": B.stat(lambda a_, b_, c_: (b_ - c_) / (a_ - c_), bl["j10"], bl["tyb17"], bl["close"]),
        "share_after_pre28": B.stat(lambda a_, b_, c_: (b_ - c_) / (a_ - c_), bl["j10"], bl["pre28"], bl["close"])}
    dec["supp_catchup_share_by_segment"] = {
        f"{s0}_to_{s1}": B.stat(lambda m0, m1, b0, b1: 1 - (b0 - b1) / (m0 - m1), n[s0], n[s1], bl[s0], bl[s1])
        for s0, s1 in (("j10", "pre28"), ("pre28", "close"), ("tyb17", "close"))}
    R0, RD = B.M["raw_j10"][0], B.M["raw_close"][0]
    dec["supp_raw_G_split"] = {"G_raw": B.stat(G, R0, RD),
                               "shape_calibration_share": B.stat(lambda r0, a_, rd: (r0 - a_) / (r0 - rd), R0, A, RD),
                               "a_share": B.stat(lambda a_, b_, r0, rd: (a_ - b_) / (r0 - rd), A, Bm, R0, RD),
                               "b_share": B.stat(lambda b_, c_, r0, rd: (b_ - c_) / (r0 - rd), Bm, C, R0, RD),
                               "c_share": B.stat(lambda c_, r0, rd: (c_ - rd) / (r0 - rd), C, R0, RD)}
    res["decomposition"] = dec
    sv = dec["supp_non_v7_info_by_segment"]
    log(f"[{tag}] v7 込みの情報の入り方: j10→close {sv['j10_to_close']['est']:.4f} = j10→pre28 {sv['j10_to_pre28']['est']:.4f} + pre28→tyb17 {sv['pre28_to_tyb17']['est']:.4f}"
        f" + tyb17→close {sv['tyb17_to_close']['est']:.4f}  (最後の17分 {100 * sv['share_after_tyb17']['est']:.1f}% [{100 * sv['share_after_tyb17']['ci95'][0]:.1f},{100 * sv['share_after_tyb17']['ci95'][1]:.1f}])")
    rs = dec["supp_raw_G_split"]
    log(f"[{tag}] raw G {rs['G_raw']['est']:.4f}: 形の較正 {100 * rs['shape_calibration_share']['est']:.1f}% / 追いつき {100 * rs['a_share']['est']:.1f}% / 馬体重 {100 * rs['b_share']['est']:.1f}% / 残り {100 * rs['c_share']['est']:.1f}%"
        f"   市場のべき乗 β: " + " ".join(f"{s}={B.W['mk_' + s]['lk_' + s]:.3f}" for s in STAGES))
    log(f"[{tag}] 分解 G={dec['G']['est']:.4f}: (a) 追いつき {100 * dec['a_share']['est']:.1f}% [{100 * dec['a_share']['ci95'][0]:.1f},{100 * dec['a_share']['ci95'][1]:.1f}]"
        f"  (b) 馬体重 {100 * dec['b_share']['est']:.1f}% [{100 * dec['b_share']['ci95'][0]:.1f},{100 * dec['b_share']['ci95'][1]:.1f}]"
        f"  (c) 残り {100 * dec['c_share']['est']:.1f}% [{100 * dec['c_share']['ci95'][0]:.1f},{100 * dec['c_share']['ci95'][1]:.1f}] → {dec['H2_verdict']}")
    log(f"[{tag}] Δ〜(v7−10時) 傾き {slope:.3f}  R² {dec['delta_on_d7']['r2_test']['est']:.4f}   Δ〜馬体重 R² {dec['delta_on_bw_r2_test']['est']:.4f}")

    # ---- Q4 朝の補正印
    mm = {"v7": B.row("v7"), "v6": B.row("v6")}
    for s in STAGES:
        mm[f"v7+{s}"] = B.row(f"bl_{s}")
        mm[f"v7+{s}"]["d_top3_vs_v7"] = B.diff(f"bl_{s}", "v7", 2, 100.0)
        mm[f"v7+{s}"]["d_win_vs_v7"] = B.diff(f"bl_{s}", "v7", 1, 100.0)
        mm[f"v7+{s}"]["d_nll_vs_v7"] = B.diff(f"bl_{s}", "v7", 0)
    for s in ("j10", "tyb17", "close"):
        mm[f"v6+{s}"] = B.row(f"bl6_{s}")
        mm[f"v6+{s}"]["d_top3_vs_v6"] = B.diff(f"bl6_{s}", "v6", 2, 100.0)
    frac = lambda a, b, c: (b - a) / (c - a)  # noqa: E731
    mm["frac_of_t17_gain_in_morning"] = {
        "top3_j10": B.stat(frac, B.M["v7"][2], B.M["bl_j10"][2], B.M["bl_tyb17"][2]),
        "top3_am9": B.stat(frac, B.M["v7"][2], B.M["bl_am9"][2], B.M["bl_tyb17"][2]),
        "nll_j10": B.stat(frac, B.M["v7"][0], B.M["bl_j10"][0], B.M["bl_tyb17"][0]),
        "nll_am9": B.stat(frac, B.M["v7"][0], B.M["bl_am9"][0], B.M["bl_tyb17"][0]),
        "nll_j10_vs_pre28": B.stat(frac, B.M["v7"][0], B.M["bl_j10"][0], B.M["bl_pre28"][0]),
        "nll_j10_vs_close": B.stat(frac, B.M["v7"][0], B.M["bl_j10"][0], B.M["bl_close"][0]),
        "top3_j10_vs_close": B.stat(frac, B.M["v7"][2], B.M["bl_j10"][2], B.M["bl_close"][2])}
    mm["H4_verdict"] = "朝の補正印は有効" if mm["v7+j10"]["d_top3_vs_v7"]["ci95"][0] > 0 else "有効と言えない"
    mm["morning_plus"] = {"j10_v7_morn": B.row("j10_v7_morn"), "d_nll_vs_j10_v7": B.diff("j10_v7_morn", "bl_j10"),
                          "d_top3_vs_j10_v7": B.diff("j10_v7_morn", "bl_j10", 2, 100.0),
                          "am9_v7_morn": B.row("am9_v7_morn"), "d_nll_am9_vs_am9_v7": B.diff("am9_v7_morn", "bl_am9")}
    mm["morning_plus"]["verdict"] = "PASS" if mm["morning_plus"]["d_nll_vs_j10_v7"]["ci95"][1] < 0 else "none"
    res["morning_mark"] = mm
    for k in ["v7"] + [f"v7+{s}" for s in STAGES]:
        r = mm[k]
        log(f"[{tag}] {k:10s} ◎1着 {r['hon_win']['est']:.2f} [{r['hon_win']['ci95'][0]:.2f},{r['hon_win']['ci95'][1]:.2f}]"
            f"  ◎3着内 {r['hon_top3']['est']:.2f} [{r['hon_top3']['ci95'][0]:.2f},{r['hon_top3']['ci95'][1]:.2f}]"
            + (f"  差 {r['d_top3_vs_v7']['est']:+.2f}pt [{r['d_top3_vs_v7']['ci95'][0]:+.2f},{r['d_top3_vs_v7']['ci95'][1]:+.2f}]" if k != "v7" else ""))
    f = mm["frac_of_t17_gain_in_morning"]
    log(f"[{tag}] 朝で取れる割合 (対 tyb17): ◎3着内 j10 {100 * f['top3_j10']['est']:.0f}% [{100 * f['top3_j10']['ci95'][0]:.0f},{100 * f['top3_j10']['ci95'][1]:.0f}]"
        f"  NLL j10 {100 * f['nll_j10']['est']:.0f}% [{100 * f['nll_j10']['ci95'][0]:.0f},{100 * f['nll_j10']['ci95'][1]:.0f}]"
        f"  NLL am9 {100 * f['nll_am9']['est']:.0f}%   {mm['H4_verdict']}")
    log(f"[{tag}] 朝の動き+騎手/調教師の癖を足す: ΔNLL {mm['morning_plus']['d_nll_vs_j10_v7']['est']:+.5f} "
        f"[{mm['morning_plus']['d_nll_vs_j10_v7']['ci95'][0]:+.5f},{mm['morning_plus']['d_nll_vs_j10_v7']['ci95'][1]:+.5f}] {mm['morning_plus']['verdict']}")
    return res


def run_predict(d, tag):
    """Q3: 10 時の情報で Δ (締切までの動き) を当てられるか"""
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    v, t = d[d.split == "valid"].reset_index(drop=True), d[d.split == "test"].reset_index(drop=True)
    yv, yt = demean(v, "delta"), demean(t, "delta")
    sets = {"L0_lk10": ["lk_j10", "lk10sq"], "L1_+v7差": ["lk_j10", "lk10sq", "d7"],
            "L2_+朝の動き": ["lk_j10", "lk10sq", "d7", "m910", "m239"],
            "L3_+騎手調教師": ["lk_j10", "lk10sq", "d7", "m910", "m239", "jk_enc", "tr_enc"]}
    out = {"n_test_horses": int(len(t)), "sd_delta_test": float(yt.std()),
           "share_bet_down_ge_0.15": float((t.delta >= 0.15).mean()), "models": {}}
    lab = (t.delta.values >= 0.15).astype(int)
    best = -9.0
    for name, cols in sets.items():
        Xv = np.column_stack([demean(v, c) for c in cols]); Xt = np.column_stack([demean(t, c) for c in cols])
        sd = Xv.std(0); sd[sd == 0] = 1
        w = np.linalg.lstsq(Xv / sd, yv, rcond=None)[0]
        p = (Xt / sd) @ w
        r = r2_boot(yt, p, t.date.values)
        out["models"][name] = {"r2_test": r, "auc_bet_down": float(roc_auc_score(lab, p)), "coef_std": {c: float(x) for c, x in zip(cols, w)}}
        best = max(best, r["est"])
        log(f"[{tag}] Δ予測 {name:14s} R² {r['est']:.4f} [{r['ci95'][0]:.4f},{r['ci95'][1]:.4f}]  AUC(売れる) {out['models'][name]['auc_bet_down']:.3f}")
    gcols = sets["L3_+騎手調教師"] + ["rank10", "n_run", "post"]
    m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=200, subsample=0.8,
                          subsample_freq=1, colsample_bytree=0.8, n_jobs=4, random_state=0, verbose=-1)
    m.fit(v[gcols].astype(float), yv)
    p = m.predict(t[gcols].astype(float))
    p = p - pd.Series(p).groupby(t.rid_s.values).transform("mean").values
    r = r2_boot(yt, p, t.date.values)
    out["models"]["GBM"] = {"r2_test": r, "auc_bet_down": float(roc_auc_score(lab, p)),
                            "gain_share": {c: float(g) for c, g in zip(gcols, m.booster_.feature_importance("gain") / m.booster_.feature_importance("gain").sum())}}
    best = max(best, r["est"])
    log(f"[{tag}] Δ予測 GBM            R² {r['est']:.4f} [{r['ci95'][0]:.4f},{r['ci95'][1]:.4f}]  AUC(売れる) {out['models']['GBM']['auc_bet_down']:.3f}")
    out["best_r2"] = float(best)
    # ---- 事後の探索 E1 (事前登録の外。1 回目の実行で Δ の R² が大きかったので追加):
    #   10 時に予測した締切までの動き Δ^ は、[j10+v7] に無い着順の情報を持つか。valid の Δ^ は開催日 5 分割の out-of-fold。
    ud = np.sort(v.date.unique()); fold = pd.Series(np.arange(len(ud)) % 5, index=ud)
    fv = v.date.map(fold).values
    oof = np.zeros(len(v))
    for k in range(5):
        mk = lgb.LGBMRegressor(**m.get_params())
        mk.fit(v.loc[fv != k, gcols].astype(float), yv[fv != k])
        oof[fv == k] = mk.predict(v.loc[fv == k, gcols].astype(float))
    dd = pd.concat([v.assign(dhat=oof), t.assign(dhat=m.predict(t[gcols].astype(float)))], ignore_index=True)
    dd["pred_close"] = dd["lk_j10"] + dd["dhat"]
    dd = zstd(dd, ["dhat"])
    B = Block(dd)
    B.add("mk_j10", ["lk_j10"]); B.add("mk_close", ["lk_close"]); B.add("bl_j10", ["lk_j10", "lm7"])
    B.add("pred_close_raw", ["pred_close"], fit=False); B.add("pred_close", ["pred_close"])
    B.add("j10_v7_dhat", ["lk_j10", "lm7", "z_dhat"])
    out["E1_posthoc"] = {"oof_r2_valid": float(1 - ((yv - (oof - pd.Series(oof).groupby(v.rid_s.values).transform("mean").values)) ** 2).sum() / (yv ** 2).sum()),
                         "pred_close_raw": B.row("pred_close_raw"), "pred_close": B.row("pred_close"), "j10_v7_dhat": B.row("j10_v7_dhat"),
                         "d_nll_dhat_vs_j10_v7": B.diff("j10_v7_dhat", "bl_j10"), "d_top3_dhat_vs_j10_v7": B.diff("j10_v7_dhat", "bl_j10", 2, 100.0),
                         "d_nll_predclose_vs_j10_v7": B.diff("pred_close", "bl_j10"),
                         "share_of_G_recovered_by_pred_close": B.stat(lambda a_, p_, c_: (a_ - p_) / (a_ - c_), B.M["mk_j10"][0], B.M["pred_close"][0], B.M["mk_close"][0])}
    e = out["E1_posthoc"]
    log(f"[{tag}] E1(事後) 予測締切オッズ NLL {e['pred_close']['nll']['est']:.4f} (G の {100 * e['share_of_G_recovered_by_pred_close']['est']:.1f}% "
        f"[{100 * e['share_of_G_recovered_by_pred_close']['ci95'][0]:.1f},{100 * e['share_of_G_recovered_by_pred_close']['ci95'][1]:.1f}])"
        f"  [j10+v7+Δ^] − [j10+v7] ΔNLL {e['d_nll_dhat_vs_j10_v7']['est']:+.5f} [{e['d_nll_dhat_vs_j10_v7']['ci95'][0]:+.5f},{e['d_nll_dhat_vs_j10_v7']['ci95'][1]:+.5f}]"
        f"  ◎3着内 {e['d_top3_dhat_vs_j10_v7']['est']:+.2f}pt [{e['d_top3_dhat_vs_j10_v7']['ci95'][0]:+.2f},{e['d_top3_dhat_vs_j10_v7']['ci95'][1]:+.2f}]")
    out["H3_verdict"] = "予測できる" if best >= 0.05 else ("ほぼ予測できない" if best < 0.02 else "弱い")
    # 参考: T−28 の時点で最後の 28 分の動きを当てられるか
    d2 = d.assign(lkpsq=d.lk_pre28 ** 2, d7p=d.lm7 - d.lk_pre28, mp10=d.lk_pre28 - d.lk_j10)
    v2, t2 = d2[d2.split == "valid"], d2[d2.split == "test"]
    cols = ["lk_pre28", "lkpsq", "d7p", "mp10", "m910"]
    Xv = np.column_stack([demean(v2, c) for c in cols]); Xt = np.column_stack([demean(t2, c) for c in cols])
    sd = Xv.std(0); sd[sd == 0] = 1
    y2v, y2t = demean(v2, "delta_late"), demean(t2, "delta_late")
    w = np.linalg.lstsq(Xv / sd, y2v, rcond=None)[0]
    out["late28_linear"] = {"r2_test": r2_boot(y2t, (Xt / sd) @ w, t2.date.values), "sd_delta_late_test": float(y2t.std()),
                            "coef_std": {c: float(x) for c, x in zip(cols, w)}}
    log(f"[{tag}] 参考 T−28→締切 の動き R² {out['late28_linear']['r2_test']['est']:.4f}   {out['H3_verdict']}")
    return out


def main():
    t0 = time.time()
    d, att = build_panel()
    log(f"[panel] {att}  ({time.time() - t0:.0f}s)")
    r1 = d.drop_duplicates("rid_s")
    tim = {"tyb_gap_min": {k: float(x) for k, x in r1.tyb_gap.describe()[["min", "25%", "50%", "75%", "max"]].items()},
           "pre28_gap_min": {k: float(x) for k, x in r1.pre_gap.describe()[["min", "25%", "50%", "75%", "max"]].items()},
           "am9_clock_min": {k: float(x) for k, x in r1.minute_am9.describe()[["min", "50%", "max"]].items()},
           "win_pool_share_of_final_median": {k: float((r1[f"pool_{k}"] / r1["pool_fin"]).median()) for k in ("prev23", "am9", "pre28")},
           "win_pool_share_of_final_mean": {k: float((r1[f"pool_{k}"] / r1["pool_fin"]).mean()) for k in ("prev23", "am9", "pre28")},
           "s11_races": {s: int(r1[(r1.split == s) & r1.s11].shape[0]) for s in ("valid", "test")},
           "mean_abs_log_odds_gap_to_close": {s: float((d[f"lk_{s}"] - d["lk_close"]).abs().mean()) for s in STAGES[:-1]},
           "mean_abs_log_odds_gap_to_close_s11": {s: float((d.loc[d.s11, f"lk_{s}"] - d.loc[d.s11, "lk_close"]).abs().mean()) for s in STAGES[:-1]}}
    log(f"[timing] {json.dumps(tim, ensure_ascii=False)}")
    out = {"experiment": "late_money_decomp", "date": "2026-11-14", "attrition": att, "timing": tim,
           "ALL": run_block(d, "ALL"), "S11": run_block(d[d.s11].reset_index(drop=True), "S11")}
    out["S11"]["late_money_predictability"] = run_predict(d[d.s11].reset_index(drop=True), "S11")
    out["ALL"]["late_money_predictability"] = run_predict(d, "ALL")
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"\n→ {OUT_JSON}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    main()
