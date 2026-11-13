# -*- coding: utf-8 -*-
"""
intervention_signal_exp.py — 陣営の介入シグナル理論 (2026-11-14)
=================================================================
理論: 厩舎が馬に「手を入れた」(初ブリンカー・ハミ替え・馬具の着脱・去勢・芝ダ替わり・大幅な距離変更・連闘) という事実は、
      陣営の私的な診断 (「能力はあるが特定の問題があり、これで直る」) か苦し紛れのどちらかを表す。
      前日までに公表されるものと、当日のパドックでしか見えないものがある。市場がそれを過大/過小評価しているかを測る。

■ 事前登録 (結果を見る前に固定。この docstring を書いた時点で見たのは列の存在率・コード頻度・コード表だけ)

シグナル 12 本 (これ以外は作らない。事後のセル分けはしない)。「前走」= その馬の D より前の直近の平地出走 (取消・除外は除く)。
  公表組 (前日までに分かる)
   1 BL_FIRST   初ブリンカー: 今回ブリンカー着用、過去の全出走 (障害含む) で一度も無し、過去 1 走以上  [前日: 出馬表のブリンカー表示]
   2 BL_REON    ブリンカー再装着: 今回着用・前走なし・それ以前に着用歴あり                              [前日]
   3 BL_OFF     ブリンカー外し: 前走着用・今回なし                                                      [前日]
   4 GELD_FIRST 去勢明け初戦: 今回 セ、master 上の前走が 牡                                             [出馬投票時]
   5 SURF_FIRST 初の芝ダ替わり: 過去の平地出走がすべて一方の面、今回が初めて他方、過去 1 走以上         [出馬投票時]
   6 DIST_SHORT 前走より 400m 以上短縮                                                                  [出馬投票時]
   7 DIST_LONG  前走より 400m 以上延長                                                                  [出馬投票時]
   8 RENTO      連闘: 前の出走 (障害含む) から 8 日以内                                                 [出馬投票時]
  パドック組 (当日の馬を見ないと分からない)
   9 BIT_CHG    ハミ替え: JRDB SKB の「ハミ」コードが前走と違う (両方記録あり)                          [当日パドック。SKB の公開はレース後]
  10 GEAR_ADD   馬具追加: レース用馬具セット G のうち前走に無く今回あるものが 1 つ以上                  [同上]
  11 GEAR_OFF   馬具外し: G のうち前走にあって今回無いものが 1 つ以上                                   [同上]
  12 TYB_CHG    JRDB 直前情報 (TYB) の馬具変更フラグ >=1                                               [当日パドック時点で実際に配信される唯一の版]
  G = 馬具種別 2 (その他馬具) からブリンカー(001, 別シグナル) と パドック用・ほぼ全馬が着ける物
      (008 メンコ / 009 ガムチェーン / 019 チェーンシャンク / 020 パドックブリンカー / 068 ネックストラップ / 070 ホライゾネット(パドック)) を除いた
      {002,011,012,018,021,022,031,032,036,038,039,069,071,074,079,081,082,083,096,099}
  事前に外したもの: バンテージの着脱 (着用率 48% で前後肢の区別なし、12 本の枠に収めるため)、蹄鉄の変更 (記録率 1% 未満)。
  経歴が欠ける馬: SKB/SED は 2011-01〜。生年 2008 以前の馬は 2010 年以前の出走が見えないので BL_FIRST / BL_REON / SURF_FIRST を欠損 (集計外) にする。

テスト 1 (市場テスト、主検定)
  対象 = master_lite の平地 2013-2025、全出走馬に確定単勝オッズがあり勝ち馬がいるレース。
  市場確率 p = (1/オッズ) のレース内正規化。本命-大穴バイアスを 1 パラメータで除く: q ∝ p^γ、γ は 2013-2022 のレースだけで条件付きロジット fit (10 時オッズは別の γ10)。
  統計量 = シグナル馬の 実際の勝ち数 / Σq (= actual/market)。期間は 2013-19 と 2020-25。CI95 は開催日クラスタ bootstrap (2000 回)。
  p 値 = bootstrap の標準誤差による正規近似 (両側、帰無 = 1)。生の p に対する比と 均等買い単勝回収率は文脈として併記するだけ。
  判定: 12 本それぞれ、両期間で符号が同じなら p_sig = max(p_前期, p_後期)、違えば 1。12 本に Benjamini-Hochberg (q=0.05)。
        「実在」= BH で棄却 かつ 両期間の CI95 がともに 1 を含まない。確定市場と 10 時市場は別々の族として同じ規則を当てる。
テスト 2 (前走の出来への依存、分け方は 1 つだけ)
  前走 3 着以内 (好走後) と 前走 6 着以下 (凡走後)。前走の異常 (中止等) は両方から除く。
  統計量 = 差の差 DiD = [actual/market(シグナル,好走後) / actual/market(シグナル,凡走後)] / [同じ比を非シグナル馬で]。帰無 = 1。
  判定はテスト 1 と同じ (両期間同符号 → max p → BH q=0.05 → 両期間 CI が 1 を外す)。確定市場のみ。
テスト 3 (ゲート)
  指標 (0/1、欠損はレース平均) をそのまま z 列にして gc.gate へ。公表組 8 列と パドック組 4 列を別々に 1 回ずつ。
  重みは valid=2023 で fit、test=2024-25 は 1 回だけ。本丸は M (確定市場 + v7 → +z)。CI95 の上端 < 0 で PASS。
総合判定
  「市場の外にある介入シグナルがある」と言えるのは、テスト 1 で実在のシグナルがあるか、ゲート M が PASS のときだけ。
  どちらも無ければ null (確定市場に織り込み済み)。10 時市場だけで出るものは「朝だけの偏り」と書く。回収率のセル探しはしない。

■ 事後追加 (事前登録の結果を見た後に足した頑健性確認。判定は上の事前登録の規則のまま、ここは「その結果は帰無モデルの作りのせいか」を見るだけ)
  R1 柔軟な本命-大穴補正: q3 ∝ exp(a·lnp + b·lnp² + c·lnp³) (2013-2022 で fit)。1 パラメータの γ では 2〜8% 帯が 5〜8% 過小に残ったため。
     12 本すべてを q3 で測り直し、同じ FDR 規則を当てる (robust_flexible_flb)。払戻均等買い (1/オッズ比例) の回収率も文脈として付ける。
  R3 前走着順の構成をそろえた比: 非シグナル馬で 前走 3 着以内は actual/market 0.95〜0.96、6 着以下は 1.06〜1.07 と分かった
     (市場の癖の地図 2026-11-07 で既知の「前走着順」の癖)。介入した馬は凡走後に偏る (例: 400m 以上短縮の 85% が前走 6 着以下) ので、
     層 (前走なし / 1-3 着 / 4-5 着 / 6 着以下 / 前走異常) ごとの非シグナル馬の actual/market(q3) を期待値に掛けて測り直す (robust_prev_finish_std)。
     「介入の効果」と「前走で負けた馬が過小評価される既知の癖」を分けるため。同じ FDR 規則を当てる。
  R2 事前登録の規則で残ったシグナルだけ: 4 つの年ブロック (2013-16 / 2017-19 / 2020-22 / 2023-25) の actual/market(q3) と、その 1 列だけのゲート。

出力: reports/intervention_signal_exp.json (集計値のみ)、reports/intv_panel.parquet (作業用、JRDB 由来なので外へ出さない)
実行: PYTHONUTF8=1 E:/PyCaLiAI/venv311/Scripts/python.exe lab/experiments/intervention_signal_exp.py
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
from scipy.optimize import minimize_scalar
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate_common as gc  # noqa: E402

WT = Path(__file__).resolve().parents[2]
REP = WT / "reports"
SKB = gc.ROOT / "data" / "jrdb" / "skb.parquet"
TYB = gc.ROOT / "data" / "jrdb" / "tyb.parquet"
OUT_JSON = REP / "intervention_signal_exp.json"
PANEL_PQ = REP / "intv_panel.parquet"

PUBLIC = ["BL_FIRST", "BL_REON", "BL_OFF", "GELD_FIRST", "SURF_FIRST", "DIST_SHORT", "DIST_LONG", "RENTO"]
PADDOCK = ["BIT_CHG", "GEAR_ADD", "GEAR_OFF", "TYB_CHG"]
SIGNALS = PUBLIC + PADDOCK
GEAR = ["002", "011", "012", "018", "021", "022", "031", "032", "036", "038", "039", "069", "071", "074",
        "079", "081", "082", "083", "096", "099"]
PERIODS = {"2013-19": (2013, 2019), "2020-25": (2020, 2025)}
N_BOOT = 2000
Q_FDR = 0.05
log = gc.log


# ------------------------------------------------------------------ panel
def build_panel() -> pd.DataFrame:
    t0 = time.time()
    sed = gc.load_sed(["racekey", "ymd", "ban", "ped", "dist", "td", "jyun", "ijou", "tan_odds", "odds10_tan"])
    sed = sed[sed["ymd"] <= 20251231].copy()
    bc = [f"bagu_{i}" for i in range(1, 9)]
    skb = pd.read_parquet(SKB, columns=["racekey", "ban", "hami"] + bc)
    bl = np.zeros(len(skb), bool)
    mask = np.zeros(len(skb), np.int64)
    for c in bc:
        v = skb[c].values
        bl |= (v == "001")
        for k, code in enumerate(GEAR):
            mask |= np.where(v == code, 1 << k, 0)
    skb["bl"] = bl
    skb["gmask"] = mask
    skb["has_skb"] = True
    sed = sed.merge(skb[["racekey", "ban", "hami", "bl", "gmask", "has_skb"]], on=["racekey", "ban"], how="left")
    tyb = pd.read_parquet(TYB, columns=["racekey", "ban", "bagu_change"])
    sed = sed.merge(tyb.drop_duplicates(["racekey", "ban"]), on=["racekey", "ban"], how="left")
    del skb, tyb
    log(f"[panel] SED {len(sed):,} 行  SKB 結合率 {sed.has_skb.notna().mean():.4f}  TYB 結合率 {sed.bagu_change.notna().mean():.4f}")
    log(f"[panel] ijou 分布 {sed.ijou.value_counts(dropna=False).to_dict()}")
    sed["started"] = ~sed["ijou"].isin([1, 2])          # 1 取消 / 2 除外 は出走していない
    h = sed[sed["started"]].sort_values(["ped", "ymd", "racekey"]).reset_index(drop=True)
    assert not h.duplicated(["ped", "ymd"]).any(), "同じ馬が同じ日に 2 回出走している"
    h["bl"] = h["bl"].fillna(False).astype(bool)
    h["gmask"] = h["gmask"].fillna(0).astype(np.int64)
    g = h.groupby("ped", sort=False)
    h["n_prev_any"] = g.cumcount()
    h["prev_date_any"] = g["date"].shift()
    h["bl_before"] = g["bl"].cumsum() - h["bl"].astype(int)          # これより前の全出走でのブリンカー着用回数
    # 平地だけの連鎖
    f = h[h["td"].isin([1, 2])].copy()
    gf = f.groupby("ped", sort=False)
    f["n_prev_flat"] = gf.cumcount()
    for c in ("ymd", "dist", "bl", "gmask", "hami", "jyun", "ijou"):
        f[f"p_{c}"] = gf[c].shift()
    assert (f["p_ymd"].isna() | (f["p_ymd"] < f["ymd"])).all()
    f["is_turf"] = (f["td"] == 1).astype(int)
    f["turf_before"] = gf["is_turf"].cumsum() - f["is_turf"]
    f["dirt_before"] = f["n_prev_flat"] - f["turf_before"]
    has_prev = f["n_prev_flat"] > 0
    yy = pd.to_numeric(f["ped"].str[:2], errors="coerce")
    career_ok = (yy >= 9) & (yy <= 30)                                # 生年 2009 以降 = SKB/SED で初出走から見えている
    p_bl = f["p_bl"].fillna(False).astype(bool)
    sig = pd.DataFrame(index=f.index)
    sig["BL_FIRST"] = np.where(career_ok, (f["bl"] & (f["bl_before"] == 0) & (f["n_prev_any"] > 0)).astype(float), np.nan)
    sig["BL_REON"] = np.where(career_ok, (f["bl"] & has_prev & ~p_bl & (f["bl_before"] > 0)).astype(float), np.nan)
    sig["BL_OFF"] = (has_prev & p_bl & ~f["bl"]).astype(float)
    switch = ((f["is_turf"] == 1) & (f["turf_before"] == 0) & (f["dirt_before"] > 0)) | \
             ((f["is_turf"] == 0) & (f["dirt_before"] == 0) & (f["turf_before"] > 0))
    sig["SURF_FIRST"] = np.where(career_ok, switch.astype(float), np.nan)
    dd = f["dist"] - f["p_dist"]
    sig["DIST_SHORT"] = (has_prev & (dd <= -400)).astype(float)
    sig["DIST_LONG"] = (has_prev & (dd >= 400)).astype(float)
    gap = (f["date"] - f["prev_date_any"]).dt.days
    sig["RENTO"] = (gap <= 8).astype(float)
    both_h = has_prev & f["hami"].notna() & f["p_hami"].notna()
    sig["BIT_CHG"] = np.where(both_h | ~has_prev, (both_h & (f["hami"] != f["p_hami"])).astype(float), np.nan)
    pm = f["p_gmask"].fillna(0).astype(np.int64)
    sig["GEAR_ADD"] = (has_prev & ((f["gmask"] & ~pm) != 0)).astype(float)
    sig["GEAR_OFF"] = (has_prev & ((pm & ~f["gmask"]) != 0)).astype(float)
    sig["TYB_CHG"] = np.where(f["bagu_change"].notna(), (f["bagu_change"] >= 1).astype(float), np.nan)
    f = pd.concat([f, sig], axis=1)
    f["pstrat"] = np.where(~has_prev, 0, np.where(f["p_ijou"] != 0, 4, np.where(f["p_jyun"] <= 3, 1, np.where(f["p_jyun"] <= 5, 2, 3))))
    f["prev_good"] = np.where(has_prev & (f["p_ijou"] == 0) & f["p_jyun"].between(1, 3), 1.0,
                              np.where(has_prev & (f["p_ijou"] == 0) & (f["p_jyun"] >= 6), 0.0, np.nan))
    # master (平地 2013-2025) を母集団にする。去勢は master の性別から
    m = gc.load_master_lite()[["日付", "レースID(新/馬番無)", "馬番", "血統登録番号", "性別"]].copy()
    m["rid_s"] = m["レースID(新/馬番無)"].astype(str).str[:16]
    m["ban"] = pd.to_numeric(m["馬番"], errors="coerce")
    m = m.dropna(subset=["ban"]).sort_values(["血統登録番号", "日付"])
    m["p_sex"] = m.groupby("血統登録番号")["性別"].shift()
    m["GELD_FIRST"] = np.where(m["p_sex"].isna(), np.nan, ((m["性別"] == "セ") & (m["p_sex"] == "牡")).astype(float))
    m.loc[m["p_sex"].isna() & (m["日付"] >= 20140101), "GELD_FIRST"] = 0.0     # 初出走は去勢明けではない (2013 年は前走が master 外の可能性)
    log(f"[panel] master の性別遷移: 牡→セ {int((m.GELD_FIRST == 1).sum()):,} 件 / セ の出走 {int((m['性別'] == 'セ').sum()):,}")
    keep = ["rid_s", "ban", "ped", "ymd", "date", "td", "jyun", "ijou", "tan_odds", "odds10_tan", "prev_good", "pstrat"] + \
           [s for s in SIGNALS if s != "GELD_FIRST"]
    pan = m[["rid_s", "ban", "GELD_FIRST"]].merge(f[keep], on=["rid_s", "ban"], how="inner")
    log(f"[panel] master {len(m):,} 行 → SED 平地出走と結合 {len(pan):,} 行 ({time.time() - t0:.0f}s)")
    # 市場確率
    pan["year"] = pan["ymd"] // 10000
    pan["win"] = (pan["jyun"] == 1) & (pan["ijou"].isin([0, 5, 6]) | pan["ijou"].isna())
    gr = pan.groupby("rid_s")
    ok = gr["tan_odds"].transform(lambda s: s.notna().all() and (s > 0).all()) & (gr["win"].transform("sum") >= 1) \
        & (gr["ban"].transform("size") >= 5)
    pan = pan[ok].sort_values(["rid_s", "ban"]).reset_index(drop=True)
    inv = 1.0 / pan["tan_odds"]
    pan["p"] = inv / inv.groupby(pan["rid_s"]).transform("sum")
    ok10 = pan.groupby("rid_s")["odds10_tan"].transform(lambda s: s.notna().all() and (s > 0).all())
    inv10 = 1.0 / pan["odds10_tan"].where(ok10)
    pan["p10"] = inv10 / inv10.groupby(pan["rid_s"]).transform("sum")
    log(f"[panel] 使えるレース {pan.rid_s.nunique():,}  行 {len(pan):,}  10時オッズあり {pan.p10.notna().mean():.3f}")
    return pan


def fit_gamma(pan: pd.DataFrame, col: str) -> float:
    d = pan[(pan["year"] <= 2022) & pan[col].notna()]
    codes = pd.factorize(d["rid_s"])[0]
    n = codes.max() + 1
    lp = np.log(d[col].values)
    w = d["win"].values
    f = lambda gma: -gc.race_logsoftmax(gma * lp, codes, n)[w].sum() / n
    return float(minimize_scalar(f, bounds=(0.5, 1.5), method="bounded").x)


def apply_gamma(pan, col, gma):
    ok = pan[col].notna().values
    out = np.full(len(pan), np.nan)
    d = pan[ok]
    codes = pd.factorize(d["rid_s"])[0]
    out[ok] = np.exp(gc.race_logsoftmax(gma * np.log(d[col].values), codes, codes.max() + 1))
    return out


# ------------------------------------------------------------------ stats
def _by_date(dc, nd, mask, w):
    return np.bincount(dc[mask], weights=w[mask], minlength=nd)


def ratio_stat(d: pd.DataFrame, mask: np.ndarray, ecol: str, rng) -> dict:
    """d: 1 期間のパネル。mask: シグナル馬。actual/expected と開催日 bootstrap。"""
    use = mask & d[ecol].notna().values
    ud, dc = np.unique(d["ymd"].values, return_inverse=True)
    nd = len(ud)
    a = _by_date(dc, nd, use, d["win"].values.astype(float))
    e = _by_date(dc, nd, use, d[ecol].fillna(0).values)
    A, E = a.sum(), e.sum()
    I = rng.integers(0, nd, (N_BOOT, nd))
    rb = a[I].sum(1) / np.maximum(e[I].sum(1), 1e-9)
    r = A / E if E > 0 else np.nan
    se = rb.std(ddof=1)
    z = (r - 1) / se if se > 0 else 0.0
    return {"n": int(use.sum()), "wins": float(A), "expected": float(E), "ratio": float(r),
            "ci95": [float(np.percentile(rb, 2.5)), float(np.percentile(rb, 97.5))], "se": float(se),
            "p": float(2 * norm.sf(abs(z)))}


def did_stat(d: pd.DataFrame, sig: np.ndarray, rng) -> dict:
    ud, dc = np.unique(d["ymd"].values, return_inverse=True)
    nd = len(ud)
    win = d["win"].values.astype(float)
    q = d["q"].values
    pg = d["prev_good"].values
    cells = {}
    for sname, sm in (("S", sig == 1), ("N", sig == 0)):
        for gname, gm in (("good", pg == 1), ("bad", pg == 0)):
            mk = sm & gm
            cells[(sname, gname)] = (_by_date(dc, nd, mk, win), _by_date(dc, nd, mk, q), int(mk.sum()))
    I = rng.integers(0, nd, (N_BOOT, nd))

    def r(key, idx=None):
        a, e, _ = cells[key]
        if idx is None:
            return max(a.sum(), 0.5) / e.sum()
        return np.maximum(a[idx].sum(1), 0.5) / np.maximum(e[idx].sum(1), 1e-9)
    did = (r(("S", "good")) / r(("S", "bad"))) / (r(("N", "good")) / r(("N", "bad")))
    db = (r(("S", "good"), I) / r(("S", "bad"), I)) / (r(("N", "good"), I) / r(("N", "bad"), I))
    se = np.log(db).std(ddof=1)
    z = np.log(did) / se if se > 0 else 0.0
    return {"n_sig_good": cells[("S", "good")][2], "n_sig_bad": cells[("S", "bad")][2],
            "ratio_sig_good": float(r(("S", "good"))), "ratio_sig_bad": float(r(("S", "bad"))),
            "ratio_non_good": float(r(("N", "good"))), "ratio_non_bad": float(r(("N", "bad"))),
            "did": float(did), "ci95": [float(np.percentile(db, 2.5)), float(np.percentile(db, 97.5))],
            "p": float(2 * norm.sf(abs(z)))}


def std_ratio_stat(d: pd.DataFrame, v: np.ndarray, ecol: str, rng) -> dict:
    """前走着順の層ごとに、非シグナル馬の actual/expected を期待値へ掛けてそろえた比 (事後追加 R3)"""
    ud, dc = np.unique(d["ymd"].values, return_inverse=True)
    nd = len(ud)
    st = d["pstrat"].values.astype(int)
    win = d["win"].values.astype(float)
    e = d[ecol].values
    K = 5
    arr = {}
    for nm, mk in (("s", v == 1), ("n", v == 0)):
        idx = dc[mk] * K + st[mk]
        arr[nm + "a"] = np.bincount(idx, weights=win[mk], minlength=nd * K).reshape(nd, K)
        arr[nm + "e"] = np.bincount(idx, weights=e[mk], minlength=nd * K).reshape(nd, K)

    def stat(sa, se, na, ne):
        fac = np.where(ne > 0, na / np.maximum(ne, 1e-9), 1.0)
        return sa.sum(-1) / np.maximum((se * fac).sum(-1), 1e-9)
    r = float(stat(arr["sa"].sum(0), arr["se"].sum(0), arr["na"].sum(0), arr["ne"].sum(0)))
    I = rng.integers(0, nd, (N_BOOT, nd))
    C = np.zeros((N_BOOT, nd))
    np.add.at(C, (np.repeat(np.arange(N_BOOT), nd), I.ravel()), 1.0)
    rb = stat(C @ arr["sa"], C @ arr["se"], C @ arr["na"], C @ arr["ne"])
    se_ = rb.std(ddof=1)
    z = (r - 1) / se_ if se_ > 0 else 0.0
    return {"n": int((v == 1).sum()), "ratio": r, "ci95": [float(np.percentile(rb, 2.5)), float(np.percentile(rb, 97.5))],
            "p": float(2 * norm.sf(abs(z))),
            "non_signal_ratio_by_stratum": {nm: float(x) for nm, x in zip(("no_prev", "prev_1_3", "prev_4_5", "prev_6plus", "prev_abnormal"),
                                                                           arr["na"].sum(0) / np.maximum(arr["ne"].sum(0), 1e-9))},
            "signal_share_by_stratum": {nm: float(x) for nm, x in zip(("no_prev", "prev_1_3", "prev_4_5", "prev_6plus", "prev_abnormal"),
                                                                       np.bincount(st[v == 1], minlength=K) / max((v == 1).sum(), 1))}}


def bh(pvals: dict, q: float) -> dict:
    names = sorted(pvals, key=lambda k: pvals[k])
    m = len(names)
    adj, prev = {}, 1.0
    for i in range(m - 1, -1, -1):
        prev = min(prev, pvals[names[i]] * m / (i + 1))
        adj[names[i]] = prev
    return {k: {"p_sig": float(pvals[k]), "p_bh": float(adj[k]), "reject": bool(adj[k] <= q)} for k in pvals}


def family(per_sig: dict, stat_key: str, null: float = 1.0) -> dict:
    """per_sig[signal][period] = {stat_key, ci95, p}。両期間同符号 → max p → BH → 両 CI が null を外す"""
    ps, same, both_ci = {}, {}, {}
    for s, pp in per_sig.items():
        vals = [pp[k][stat_key] for k in PERIODS]
        sgn = [np.sign(v - null) for v in vals]
        same[s] = bool(sgn[0] == sgn[1] and sgn[0] != 0 and not any(np.isnan(vals)))
        ps[s] = max(pp[k]["p"] for k in PERIODS) if same[s] else 1.0
        both_ci[s] = bool(all((pp[k]["ci95"][0] > null) or (pp[k]["ci95"][1] < null) for k in PERIODS))
    out = bh(ps, Q_FDR)
    for s in out:
        out[s]["same_sign"] = same[s]
        out[s]["both_ci_exclude_null"] = both_ci[s]
        out[s]["real"] = bool(out[s]["reject"] and same[s] and both_ci[s])
    return out


# ------------------------------------------------------------------ main
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    t0 = time.time()
    REP.mkdir(exist_ok=True)
    pan = build_panel()
    g_fin, g_10 = fit_gamma(pan, "p"), fit_gamma(pan, "p10")
    pan["q"] = apply_gamma(pan, "p", g_fin)
    pan["q10"] = apply_gamma(pan, "p10", g_10)
    log(f"[flb] γ(確定)={g_fin:.4f}  γ(10時)={g_10:.4f}  (2013-2022 で fit)")
    pan.to_parquet(PANEL_PQ)
    rng = np.random.default_rng(20261114)
    res = {"generated": time.strftime("%Y-%m-%d %H:%M"), "signals": SIGNALS, "public": PUBLIC, "paddock": PADDOCK,
           "gear_set": GEAR, "flb_gamma": {"final": g_fin, "am10": g_10}, "n_boot": N_BOOT, "q_fdr": Q_FDR,
           "panel": {"races": int(pan.rid_s.nunique()), "rows": int(len(pan))}, "test1": {}, "test2": {}, "baseline": {}}
    per = {k: pan[(pan.year >= a) & (pan.year <= b)].reset_index(drop=True) for k, (a, b) in PERIODS.items()}
    for k, d in per.items():
        allm = np.ones(len(d), bool)
        res["baseline"][k] = {"races": int(d.rid_s.nunique()), "rows": int(len(d)),
                              "all_final_adj": ratio_stat(d, allm, "q", rng), "all_final_raw": ratio_stat(d, allm, "p", rng),
                              "all_am10_adj": ratio_stat(d, allm, "q10", rng),
                              "roi_all": float(100 * (d["tan_odds"] * d["win"]).sum() / len(d))}
    t1f, t1a, t2 = {}, {}, {}
    for s in SIGNALS:
        res["test1"][s] = {}
        t1f[s], t1a[s], t2[s] = {}, {}, {}
        for k, d in per.items():
            v = d[s].values
            mk = v == 1
            defined = ~np.isnan(v)
            fin = ratio_stat(d, mk, "q", rng)
            raw = ratio_stat(d, mk, "p", rng)
            am = ratio_stat(d, mk, "q10", rng)
            non = ratio_stat(d, defined & ~mk, "q", rng)
            roi = float(100 * (d.loc[mk, "tan_odds"] * d.loc[mk, "win"]).sum() / max(mk.sum(), 1))
            res["test1"][s][k] = {"n": int(mk.sum()), "share": float(mk.sum() / max(defined.sum(), 1)),
                                  "mean_market_p": float(d.loc[mk, "p"].mean()) if mk.any() else None,
                                  "final_adj": fin, "final_raw": raw, "am10_adj": am, "non_signal_final_adj": non,
                                  "flat_win_roi": roi}
            t1f[s][k], t1a[s][k] = fin, am
            t2[s][k] = did_stat(d, v, rng)
            log(f"[t1] {s:10s} {k} n={mk.sum():>6,}  確定 {fin['ratio']:.3f} [{fin['ci95'][0]:.3f},{fin['ci95'][1]:.3f}] p={fin['p']:.3f}"
                f"  10時 {am['ratio']:.3f} [{am['ci95'][0]:.3f},{am['ci95'][1]:.3f}] p={am['p']:.3f}  ROI {roi:.1f}%"
                f"  | DiD {t2[s][k]['did']:.3f} [{t2[s][k]['ci95'][0]:.3f},{t2[s][k]['ci95'][1]:.3f}]")
        res["test2"][s] = t2[s]
    res["fdr_final"] = family(t1f, "ratio")
    res["fdr_am10"] = family(t1a, "ratio")
    res["fdr_test2"] = family(t2, "did")
    for nm in ("fdr_final", "fdr_am10", "fdr_test2"):
        log(f"[{nm}] " + "  ".join(f"{s}:{v['p_bh']:.3f}{'*' if v['real'] else ''}" for s, v in res[nm].items()))
    # ゲート
    z = pan[pan.year >= 2023][["rid_s", "ban"] + SIGNALS].copy()
    z["ban"] = z["ban"].astype(int)
    base = gc.load_base()[["rid_s", "ban"]]
    cov = base.merge(z, on=["rid_s", "ban"], how="left")
    res["gate_join_coverage"] = float(cov["BL_OFF"].notna().mean())
    log(f"[gate] base {len(base):,} 行のうち指標あり {res['gate_join_coverage']:.4f}")
    res["gate_public"] = gc.gate(z, PUBLIC, name="介入・公表組")
    res["gate_paddock"] = gc.gate(z, PADDOCK, name="介入・パドック組")
    # ---------------- 事後追加の頑健性確認 (判定には使わない)
    rng2 = np.random.default_rng(777)
    tr = pan[pan.year <= 2022]
    lp_all = np.log(pan["p"].values)
    X_all = np.column_stack([lp_all, lp_all ** 2 / 10.0, lp_all ** 3 / 100.0])
    ctr = pd.factorize(tr["rid_s"])[0]
    w3 = gc._fit(X_all[(pan.year <= 2022).values], ctr, ctr.max() + 1, np.flatnonzero(tr["win"].values), 0.0)
    call = pd.factorize(pan["rid_s"])[0]
    pan["q3"] = np.exp(gc.race_logsoftmax(X_all @ w3, call, call.max() + 1))
    bins = [0, 0.01, 0.02, 0.04, 0.08, 0.15, 0.25, 0.4, 1.0]
    cal = {}
    for k, (a, b) in PERIODS.items():
        d = pan[(pan.year >= a) & (pan.year <= b)]
        gb = d.groupby(pd.cut(d["q"], bins), observed=True)
        cal[k] = {str(i): {"n": int(len(x)), "ae_gamma": float(x["win"].sum() / x["q"].sum()),
                           "ae_cubic": float(x["win"].sum() / x["q3"].sum())} for i, x in gb}
        log(f"[R1] 較正 {k}: " + "  ".join(f"{i}:{v['ae_gamma']:.3f}/{v['ae_cubic']:.3f}" for i, v in cal[k].items()))
    rob, per2 = {}, {k: pan[(pan.year >= a) & (pan.year <= b)].reset_index(drop=True) for k, (a, b) in PERIODS.items()}
    for s in SIGNALS:
        rob[s] = {}
        for k, d in per2.items():
            mk = d[s].values == 1
            st = ratio_stat(d, mk, "q3", rng2)
            st["dutch_roi"] = float(100 * d.loc[mk, "win"].sum() / (1.0 / d.loc[mk, "tan_odds"]).sum())
            rob[s][k] = st
        log(f"[R1] {s:10s} " + "  ".join(f"{k} {rob[s][k]['ratio']:.3f} [{rob[s][k]['ci95'][0]:.3f},{rob[s][k]['ci95'][1]:.3f}] p={rob[s][k]['p']:.3f}"
                                         f" 払戻均等ROI {rob[s][k]['dutch_roi']:.1f}%" for k in PERIODS))
    res["robust_flexible_flb"] = {"w_cubic": [float(x) for x in w3], "calibration_by_q_bin": cal, "per_signal": rob,
                                  "fdr": family(rob, "ratio")}
    log("[R1 fdr] " + "  ".join(f"{s}:{v['p_bh']:.3f}{'*' if v['real'] else ''}" for s, v in res["robust_flexible_flb"]["fdr"].items()))
    rob3 = {}
    for s in SIGNALS:
        rob3[s] = {k: std_ratio_stat(d, d[s].values, "q3", rng2) for k, d in per2.items()}
        log(f"[R3] {s:10s} " + "  ".join(f"{k} {rob3[s][k]['ratio']:.3f} [{rob3[s][k]['ci95'][0]:.3f},{rob3[s][k]['ci95'][1]:.3f}] p={rob3[s][k]['p']:.3f}"
                                         f" (6着以下の割合 {100 * rob3[s][k]['signal_share_by_stratum']['prev_6plus']:.0f}%)" for k in PERIODS))
    res["robust_prev_finish_std"] = {"per_signal": rob3, "fdr": family(rob3, "ratio")}
    log("[R3 fdr] " + "  ".join(f"{s}:{v['p_bh']:.3f}{'*' if v['real'] else ''}" for s, v in res["robust_prev_finish_std"]["fdr"].items()))
    log("[R3] 非シグナル馬の層別 actual/market (BIT_CHG 基準): " + json.dumps({k: {a: round(b, 3) for a, b in rob3["BIT_CHG"][k]["non_signal_ratio_by_stratum"].items()} for k in PERIODS}))
    surv = [s for s, v in res["fdr_final"].items() if v["real"]]
    res["robust_survivors"] = {}
    for s in surv:
        blocks = {}
        for nm, (a, b) in {"2013-16": (2013, 2016), "2017-19": (2017, 2019), "2020-22": (2020, 2022), "2023-25": (2023, 2025)}.items():
            d = pan[(pan.year >= a) & (pan.year <= b)].reset_index(drop=True)
            blocks[nm] = ratio_stat(d, d[s].values == 1, "q3", rng2)
            blocks[nm]["prev_finish_std"] = std_ratio_stat(d, d[s].values, "q3", rng2)
        log(f"[R2] {s} 年ブロック: " + "  ".join(f"{nm} {v['ratio']:.3f} [{v['ci95'][0]:.3f},{v['ci95'][1]:.3f}] → 前走着順そろえ {v['prev_finish_std']['ratio']:.3f}"
                                                 f" [{v['prev_finish_std']['ci95'][0]:.3f},{v['prev_finish_std']['ci95'][1]:.3f}]" for nm, v in blocks.items()))
        res["robust_survivors"][s] = {"blocks_q3": blocks, "gate_single": gc.gate(z, [s], name=f"事後・{s} 単独")}
    res["elapsed_sec"] = round(time.time() - t0, 1)
    OUT_JSON.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"[done] {OUT_JSON} ({res['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
