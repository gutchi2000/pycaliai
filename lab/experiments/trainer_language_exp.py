# -*- coding: utf-8 -*-
"""
trainer_language_exp.py — 厩舎の言語を翻訳する (調教は「その厩舎の普段との差」で意味を持つか) 2026-11-14
==========================================================================================================
理論: 速い追い切りの意味は厩舎ごとに違う。普段やらない厩舎が鋭い時計を出せば「勝負仕上げ」、いつも速い厩舎の同じ時計は
何も語らない。市場は新聞に載る生の時計を読む。厩舎ごとの「方言」は 1 厩舎あたり数千本の観測が要るので機械の方が読めるはず。

実行 (worktree の root から):  PYTHONUTF8=1 E:/PyCaLiAI/venv311/Scripts/python.exe lab/experiments/trainer_language_exp.py
出力: reports/trainer_language_exp.json (集計値のみ) / reports/trl_workouts.parquet (作業用キャッシュ)

━━━━ 事前登録 (test 2024-25 を見る前に確定。ここから下の定義・判定は結果を見て変えない) ━━━━

[調教の取り方]  master の trnH_* は「レース前日までの最後の 1 本」で、前日・前々日の軽いキャンター (4F 67 秒前後) が
  4 割を占める (発走 1〜2 日前の 4F 中央値 67 秒、3〜4 日前だけ 55〜64 秒)。追い切りを見る理論なので、生の調教ファイル
  (data/training/H-20150401-20260313.csv 坂路 517 万本 / W-… ウッド 70 万本、読み取りのみ) から取り直す。
  - 最終追い切り (坂路)   = レース D の 2〜5 日前 (実日数) の坂路のうち 4F (Time1) が最速の 1 本。同タイムは D に近い方。
  - 最終追い切り (ウッド) = 同じ窓のウッド (C/D コース) のうち 4F が最速の 1 本 (2021-07〜)。
  - 有効範囲: 坂路 4F 45〜90 秒・Lap 10.5〜25 秒、ウッド 4F 45〜90 秒・Lap1 10〜25 秒 (0 / 99.9 / 999.9 は捨てる)。
  - 馬名 + 日付で結ぶ (master と同じ)。調教日は必ず D より前。

[厩舎の普段 = as-of]  キー (調教師コード × 調教場 [美浦/栗東] × 美浦坂路の改修前後 [2023-09-28 再開で分ける]) ごとに、
  「レース日が D より前で D−730 日以降」のその厩舎の出走馬の最終追い切りだけで平均 μ と標準偏差 σ を出す。
  50 本未満は欠損。同日の他の出走馬は使わない。z は ±4 で切る。

[変種 (3 つ、全部報告)]
  (a) zA = −(Lap1 − μ_Lap1) / σ_Lap1        坂路最終追い切りのラスト 1F が「その厩舎の普段」より何 σ 鋭いか。1 列。
  (b) ベクトル 7 列:
        zH_lap1 = (a) と同じ
        zH_t4   = −(4F − μ)/σ                  全体時計
        zH_acc  = +((Lap2 − Lap1) − μ)/σ        加速ラップの形
        zH_dd   = +(追い切りが D の何日前か − μ)/σ  キーにレース曜日 (土/日/他) を足す。σ の下限 0.25
        hH_lap1 = −(Lap1 − その馬自身の過去の最終追い切り Lap1 の平均 [同じ場・改修区分、2 本以上、1500 日以内]) / σ_Lap1(厩舎)
        cH      = (今回 坂路の最終追い切りが有る 1/0) − (その厩舎の普段の坂路追い切り率)。美浦坂路閉鎖 (2023-05-29〜09-27) に
                  窓がかかる美浦厩舎の出走は履歴にも入れず欠損。
        zW_lap1 = −(ウッド Lap1 − μ)/σ          ウッドの厩舎相対 (キーは調教師 × 場)
  (c) zC = 厩舎別の反応を入れた「市場からのズレ」の予測値。〜2022 だけで fit:
        目的変数 s = (1 着 1/0) − p_mkt。p_mkt は SED の確定単勝オッズをレース内で正規化し、本命-大穴バイアスを
        1 パラメータ (p ∝ q^a、a は fit 期間の条件付きロジット) で補正した値。
        説明変数 F = (b) の坂路 6 列 (zW は 2022 以前の履歴が 1 年半しか無いので入れない)。欠損は 0 (= 普段通り)、レース内で中心化、
        〜2022 の SD で割る。
        s = F·(β + u_t)。β は全体の最小二乗、u_t は厩舎 t のランダム傾き (リッジ: u_t = (F_t'F_t + λI)^-1 F_t'(s_t − F_tβ))。
        λ は {300, 1e3, 3e3, 1e4, 3e4, 1e5, ∞(=全厩舎共通)} から「〜2019 で fit → 2020-22 の二乗誤差」が最小のものを選び、
        〜2022 全部で fit し直す。2023 以降は係数を凍結 (新しい厩舎は β だけ)。

[対照 (変種ではない)]  raw = 厩舎で割らない最終追い切り Lap1 (符号反転) ／ placebo = (a) の μ, σ を同じ年・同じ場の
  無作為な別の出走馬 (= 別の厩舎) のものに差し替えた z。(a) が効いても placebo も同じだけ効くなら「方言」ではなく
  「追い切りの取り方」の効果。

[ゲート] gate_common.gate (valid=2023 で重み fit、test=2024-25 を 1 回)。P: v7→v7+z、M: 確定市場+v7→+z (本丸)、
  M0: 確定市場→+z、A10: 10 時市場+v7→+z。ΔNLL の開催日クラスタ bootstrap CI95 の上端 < 0 で PASS。
  判定: P が PASS → SIGNAL(予測)、M が PASS → SIGNAL(市場外)。どれかが PASS したら 2024 / 2025 別の符号と placebo を確認し、
  年で符号が割れる・placebo と区別できない場合は「未確認」に落とす。3 変種 × 4 ゲートで多重なので、単発の PASS は
  それだけでは採用根拠にしない。

[市場の癖の検定 (a)]  2015-19 と 2020-25 に分ける。
  主検定: 期間ごとの条件付きロジット  P(勝ち) ∝ exp(a·log q + b·zA)  の b (zA は欠損をレース平均で埋めて中心化)。
          開催日クラスタ bootstrap (重み付き再 fit 300 回) の CI95。
          「実在」と呼ぶのは b の符号が両期間で一致し、両方の CI が 0 を含まないときだけ。
  記述: zA の固定 5 区分 (正規分布の 5 分位点 ±0.2533, ±0.8416) ごとの 実勝利数 / 市場期待勝利数 (生と FLB 補正後)、
        単勝均等買いの回収率、最上位区分 − 最下位区分の差と CI (開催日 bootstrap 2000 回)。区分の探索はしない。

[方言は在るか]  2016-2022 の最終追い切りについて、厩舎が説明する分散の割合 (一元配置の級内相関 ICC)。調教場×年の平均を
  引いた後の値も出す。厩舎平均の偶数年/奇数年の相関 (再現性)。厩舎別の反応の有無は (c) の λ 選択 (∞ が選ばれれば「無い」)。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[事後追加 (1 回目の実行で全結果を見た後に足した記述。判定には使わない。上の定義・乱数・結果は変えていない)]
  市場の癖の記述表で最上位区分 (zA > 0.84) の A/E が両期間とも 1 を上回って見えたので、その読み方を決めるために 2 つだけ足した:
  (i)  同じ表を placebo (別の厩舎の μ, σ で割った z) で出す → 最上位区分の上振れが「厩舎相対」固有か、生の時計でも同じか。
  (ii) 坂路の最終追い切りが有る馬 / 無い馬の A/E (FLB 補正後) → 区分に関係ない水準差か。
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate_common as gc  # noqa: E402

warnings.filterwarnings("ignore")
REP = gc.REP
WK_PQ = REP / "trl_workouts.parquet"
OUT_JSON = REP / "trainer_language_exp.json"
H_RAW = gc.ROOT / "data" / "training" / "H-20150401-20260313.csv"
W_RAW = gc.ROOT / "data" / "training" / "W-20150401-20260313.csv"
WINDOW = 730
MIN_N = 50
ZCLIP = 4.0
MIHO_REOPEN = np.datetime64("2023-09-28")
MIHO_CLOSE = np.datetime64("2023-05-29")
LAMBDAS = [300.0, 1e3, 3e3, 1e4, 3e4, 1e5, np.inf]
BIN_EDGES = [-np.inf, -0.8416, -0.2533, 0.2533, 0.8416, np.inf]
log = gc.log


# ------------------------------------------------------------------ 1. 調教の取り直し
def build_workouts(lite: pd.DataFrame) -> pd.DataFrame:
    if WK_PQ.exists():
        return pd.read_parquet(WK_PQ)
    t0 = time.time()
    r = lite[["馬名", "rdt"]].drop_duplicates().sort_values("rdt").reset_index(drop=True)
    r["race_dt"] = r["rdt"]
    f32 = {c: "float32" for c in ("Time1", "Lap1", "Lap2", "4F", "5F")}
    h = pd.read_csv(H_RAW, encoding="cp932", usecols=["場所", "年月日", "馬名", "Time1", "Lap2", "Lap1"], dtype=f32)
    h = h[h.Time1.between(45, 90) & h.Lap1.between(10.5, 25) & h.Lap2.between(10.5, 25)]
    h["dt"] = pd.to_datetime(h["年月日"].astype(str), format="%Y%m%d")
    h["mi"] = (h["場所"] == "美浦").astype("int8")
    h = h.drop(columns=["場所", "年月日"]).sort_values("dt")
    m = pd.merge_asof(h, r, left_on="dt", right_on="rdt", by="馬名", direction="forward", allow_exact_matches=False)
    del h
    m["dd"] = (m["race_dt"] - m["dt"]).dt.days
    m = m[m.dd.between(1, 14)]
    fin = (m[m.dd.between(2, 5)].sort_values(["Time1", "dd"]).drop_duplicates(["馬名", "race_dt"])
           [["馬名", "race_dt", "Time1", "Lap1", "Lap2", "dd", "mi"]]
           .rename(columns={"Time1": "Hf_t4", "Lap1": "Hf_lap1", "Lap2": "Hf_lap2", "dd": "Hf_dd", "mi": "Hf_mi"}))
    cnt = m.groupby(["馬名", "race_dt"]).size().rename("H_n14").reset_index()
    del m
    w = pd.read_csv(W_RAW, encoding="cp932", usecols=["場所", "年月日", "馬名", "4F", "Lap1"], dtype=f32)
    w = w[w["4F"].between(45, 90) & w.Lap1.between(10, 25)]
    w["dt"] = pd.to_datetime(w["年月日"].astype(str), format="%Y%m%d")
    w["mi"] = (w["場所"] == "美浦").astype("int8")
    w = w.drop(columns=["場所", "年月日"]).sort_values("dt")
    mw = pd.merge_asof(w, r, left_on="dt", right_on="rdt", by="馬名", direction="forward", allow_exact_matches=False)
    mw["dd"] = (mw["race_dt"] - mw["dt"]).dt.days
    wfin = (mw[mw.dd.between(2, 5)].sort_values(["4F", "dd"]).drop_duplicates(["馬名", "race_dt"])
            [["馬名", "race_dt", "4F", "Lap1", "mi"]].rename(columns={"4F": "Wf_4f", "Lap1": "Wf_lap1", "mi": "Wf_mi"}))
    out = fin.merge(wfin, on=["馬名", "race_dt"], how="outer").merge(cnt, on=["馬名", "race_dt"], how="outer")
    out = out.rename(columns={"race_dt": "rdt"})
    out.to_parquet(WK_PQ)
    log(f"[workouts] {len(out):,} (馬, レース) 行 → {WK_PQ.name} ({time.time() - t0:.0f}s)")
    return out


# ------------------------------------------------------------------ 2. as-of の「普段」
def asof_norm(days: np.ndarray, key: np.ndarray, val: np.ndarray, window: int = WINDOW, min_n: int = MIN_N):
    """各行について「同じ key で、日付が自分より前かつ window 日以内」の val の平均・SD・本数。val が NaN の行は履歴に入らない。"""
    n_all = len(days)
    mu = np.full(n_all, np.nan); sd = np.full(n_all, np.nan); nn = np.zeros(n_all, dtype=np.int64)
    order = np.lexsort((days, key))
    k = key[order]; dy = days[order]; v = val[order].astype(np.float64)
    bounds = np.flatnonzero(np.r_[True, k[1:] != k[:-1], True])
    for s, e in zip(bounds[:-1], bounds[1:]):
        if k[s] < 0:
            continue
        dg = dy[s:e]; vg = v[s:e]; okg = ~np.isnan(vg)
        if okg.sum() < min_n:
            continue
        dv = dg[okg]; vv = vg[okg]; c0 = vv.mean(); vv = vv - c0
        c1 = np.r_[0.0, np.cumsum(vv)]; c2 = np.r_[0.0, np.cumsum(vv ** 2)]
        hi = np.searchsorted(dv, dg, side="left")
        lo = np.searchsorted(dv, dg - window, side="left")
        n = hi - lo
        nz = np.maximum(n, 1)
        m = (c1[hi] - c1[lo]) / nz
        var = np.maximum((c2[hi] - c2[lo]) / nz - m ** 2, 0.0) * nz / np.maximum(nz - 1, 1)
        good = n >= min_n
        idx = order[s:e]
        mu[idx] = np.where(good, m + c0, np.nan); sd[idx] = np.where(good, np.sqrt(var), np.nan); nn[idx] = n
    return mu, sd, nn


def build_features() -> pd.DataFrame:
    cols = ["日付", "レースID(新/馬番無)", "馬番", "馬名", "血統登録番号", "調教師コード", "着順", "split"]
    lite = pd.read_parquet(gc.LITE_PQ, columns=cols)
    lite["rdt"] = pd.to_datetime(lite["日付"].astype(str), format="%Y%m%d")
    wk = build_workouts(lite)
    d = lite[lite["日付"] >= 20150401].merge(wk, on=["馬名", "rdt"], how="left").reset_index(drop=True)
    d["rid_s"] = d["レースID(新/馬番無)"].astype(str).str[:16]
    d["ban"] = d["馬番"].astype(int)
    d["year"] = d["日付"] // 10000
    d["win"] = (d["着順"] == 1).astype(float)
    days = d["rdt"].values.astype("datetime64[D]").astype(np.int64)
    tr = d["調教師コード"].values.astype(np.int64)
    has_h = d["Hf_lap1"].notna().values
    mi = d["Hf_mi"].fillna(0).values.astype(np.int64)
    wdt = d["rdt"].values.astype("datetime64[D]") - d["Hf_dd"].fillna(0).values.astype("timedelta64[D]")
    regime = ((mi == 1) & (wdt >= MIHO_REOPEN)).astype(np.int64)
    key_h = np.where(has_h, tr * 4 + mi * 2 + regime, -1)

    def z_of(val, key, sign, sd_floor=0.05, min_n=MIN_N, window=WINDOW):
        mu, sd, nn = asof_norm(days, key, val, window=window, min_n=min_n)
        z = sign * (val - mu) / np.maximum(sd, sd_floor)
        z[~np.isfinite(z)] = np.nan
        return np.clip(z, -ZCLIP, ZCLIP), mu, sd

    lap1 = d["Hf_lap1"].values.astype(float)
    d["zH_lap1"], mu_l, sd_l = z_of(lap1, key_h, -1.0)
    d["mu_lap1"], d["sd_lap1"] = mu_l, sd_l
    d["zH_t4"], _, _ = z_of(d["Hf_t4"].values.astype(float), key_h, -1.0)
    d["Hf_acc"] = d["Hf_lap2"] - d["Hf_lap1"]
    d["zH_acc"], _, _ = z_of(d["Hf_acc"].values.astype(float), key_h, +1.0)
    dow = d["rdt"].dt.dayofweek.values
    rd3 = np.where(dow == 5, 0, np.where(dow == 6, 1, 2))
    d["zH_dd"], _, _ = z_of(d["Hf_dd"].values.astype(float), np.where(has_h, key_h * 3 + rd3, -1), +1.0, sd_floor=0.25)
    # 馬自身の過去
    pedc = pd.factorize(d["血統登録番号"].astype(str))[0].astype(np.int64)
    own_mu, _, _ = asof_norm(days, np.where(has_h, pedc * 4 + mi * 2 + regime, -1), lap1, window=1500, min_n=2)
    hh = -(lap1 - own_mu) / np.maximum(sd_l, 0.05)
    hh[~np.isfinite(hh)] = np.nan
    d["hH_lap1"] = np.clip(hh, -ZCLIP, ZCLIP)
    # 坂路追い切りの有無 (厩舎の習慣との差)。厩舎の所属 (美浦/栗東) は公開情報 = その厩舎の坂路追い切りの場の多数決
    tr_mi = d.loc[has_h].groupby("調教師コード")["Hf_mi"].mean()
    d["tr_miho"] = (d["調教師コード"].map(tr_mi).fillna(0) > 0.5).astype(int)
    rdt64 = d["rdt"].values.astype("datetime64[D]")
    closed = (d["tr_miho"].values == 1) & (rdt64 >= MIHO_CLOSE + np.timedelta64(2, "D")) & (rdt64 <= MIHO_REOPEN + np.timedelta64(4, "D"))
    hv = has_h.astype(float)
    hv[(rdt64 < np.datetime64("2015-04-07")) | closed] = np.nan
    rate, _, _ = asof_norm(days, tr, hv)
    d["has_h"] = hv
    d["cH"] = hv - rate
    # ウッド
    has_w = d["Wf_lap1"].notna().values
    key_w = np.where(has_w, tr * 2 + d["Wf_mi"].fillna(0).values.astype(np.int64), -1)
    d["zW_lap1"], _, _ = z_of(d["Wf_lap1"].values.astype(float), key_w, -1.0)
    # 対照
    d["raw_lap1"] = -lap1
    rng = np.random.default_rng(20261114)
    ok = d["zH_lap1"].notna().values
    strata = d["year"].values * 4 + mi * 2 + regime
    pl = np.full(len(d), np.nan)
    for s in np.unique(strata[ok]):
        ix = np.flatnonzero(ok & (strata == s))
        pm = rng.permutation(ix)
        pl[ix] = -(lap1[ix] - mu_l[pm]) / np.maximum(sd_l[pm], 0.05)
    d["z_placebo"] = np.clip(pl, -ZCLIP, ZCLIP)
    return d


# ------------------------------------------------------------------ 3. 方言は在るか
def icc(v: pd.Series, g: pd.Series) -> float:
    df = pd.DataFrame({"v": v.values, "g": g.values}).dropna()
    grp = df.groupby("g")["v"]; ni = grp.size(); k = len(ni); n = ni.sum()
    means = grp.mean(); gm = df.v.mean()
    ssb = float((ni * (means - gm) ** 2).sum()); ssw = float(((df.v - df.g.map(means)) ** 2).sum())
    msb = ssb / (k - 1); msw = ssw / (n - k); n0 = (n - (ni ** 2).sum() / n) / (k - 1)
    sb = max((msb - msw) / n0, 0.0)
    return float(sb / (sb + msw))


def dialect_stats(d: pd.DataFrame) -> dict:
    x = d[(d.year >= 2016) & (d.year <= 2022)].copy()
    x["cy"] = x["tr_miho"] * 10000 + x["year"]
    out = {"period": "2016-2022", "n_runs": int(len(x)), "n_trainers": int(x["調教師コード"].nunique()),
           "share_with_hanro_final": float(x["has_h"].mean()), "vars": {}}
    for c, lab in (("Hf_lap1", "最終追い切りラスト1F"), ("Hf_t4", "最終追い切り4F"), ("Hf_acc", "加速 (Lap2−Lap1)"),
                   ("Hf_dd", "何日前か"), ("has_h", "坂路で最終追い切りをしたか")):
        v = x[c].astype(float)
        vd = v - v.groupby(x["cy"]).transform("mean")
        tm = pd.DataFrame({"v": vd, "t": x["調教師コード"], "odd": x.year % 2}).dropna()
        a = tm[tm.odd == 0].groupby("t")["v"].agg(["mean", "size"]); b = tm[tm.odd == 1].groupby("t")["v"].agg(["mean", "size"])
        j = a.join(b, lsuffix="_e", rsuffix="_o", how="inner"); j = j[(j.size_e >= 100) & (j.size_o >= 100)]
        tmean = tm.groupby("t")["v"].agg(["mean", "size"]); tmean = tmean[tmean["size"] >= 200]["mean"]
        out["vars"][c] = {"label": lab, "sd_total": float(v.std()), "icc_trainer": icc(v, x["調教師コード"]),
                          "icc_trainer_within_center_year": icc(vd, x["調教師コード"]),
                          "split_half_r_even_odd_years": float(j.mean_e.corr(j.mean_o)), "n_trainers_split_half": int(len(j)),
                          "trainer_mean_p10_p50_p90_dev": [float(tmean.quantile(q)) for q in (0.1, 0.5, 0.9)]}
    return out


# ------------------------------------------------------------------ 条件付きロジット (レース重み付き)
def fit_cl(X, codes, n, win_idx, wr=None, w0=None):
    if wr is None:
        wr = np.ones(n)
    tot = wr.sum(); wrow = wr[codes]; ww = wrow[win_idx]

    def f(w):
        lp = gc.race_logsoftmax(X @ w, codes, n); p = np.exp(lp)
        g = -((ww[:, None] * X[win_idx]).sum(0) - ((wrow * p)[:, None] * X).sum(0)) / tot
        return -(ww * lp[win_idx]).sum() / tot, g
    if w0 is None:
        w0 = np.zeros(X.shape[1]); w0[0] = 1.0
    r = minimize(f, w0, jac=True, method="L-BFGS-B")
    return r.x, float(r.fun)


def market_frame(d: pd.DataFrame) -> pd.DataFrame:
    sed = gc.load_sed(["racekey", "ymd", "ban", "tan_odds"])
    sed = sed[(sed.ymd >= 20150401) & (sed.ymd <= 20251231) & (sed.tan_odds > 0)]
    inv = 1.0 / sed["tan_odds"]
    sed["q"] = inv / inv.groupby(sed["rid_s"]).transform("sum")
    m = d.merge(sed[["rid_s", "ban", "tan_odds", "q"]].drop_duplicates(["rid_s", "ban"]), on=["rid_s", "ban"], how="left")
    g = m.groupby("rid_s")
    ok = (g["q"].transform(lambda s: s.notna().all()) & (g["win"].transform("sum") == 1) & (g["ban"].transform("size") >= 5))
    m = m[ok].sort_values(["rid_s", "ban"]).reset_index(drop=True)
    m["lq"] = np.log(m["q"])
    return m


def center_fill(m: pd.DataFrame, c: str) -> np.ndarray:
    v = m[c].astype(float)
    v = v.fillna(v.groupby(m["rid_s"]).transform("mean")).fillna(0.0)
    return (v - v.groupby(m["rid_s"]).transform("mean")).values


def market_bias(m: pd.DataFrame, col: str = "zH_lap1", tag: str = "") -> dict:
    res = {}
    rng = np.random.default_rng(7)
    for pname, lo, hi in (("2015-19", 2015, 2019), ("2020-25", 2020, 2025)):
        x = m[(m.year >= lo) & (m.year <= hi)].reset_index(drop=True)
        codes = pd.factorize(x["rid_s"])[0]; n = codes.max() + 1; win_idx = np.flatnonzero(x["win"].values == 1)
        zc = center_fill(x, col)
        X0 = x[["lq"]].values; X1 = np.column_stack([x["lq"].values, zc])
        w0, nll0 = fit_cl(X0, codes, n, win_idx); w1, nll1 = fit_cl(X1, codes, n, win_idx)
        rdate = x.groupby(codes)["日付"].first().sort_index().values
        ud, dc = np.unique(rdate, return_inverse=True)
        bs = []
        for _ in range(300):
            cnt = np.bincount(rng.integers(0, len(ud), len(ud)), minlength=len(ud)).astype(float)
            wb, _ = fit_cl(X1, codes, n, win_idx, wr=cnt[dc], w0=w1)
            bs.append(wb[1])
        ci_b = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        # 記述: 固定 5 区分
        x["p_adj"] = np.exp(gc.race_logsoftmax(w0[0] * x["lq"].values, codes, n))
        y = x[x[col].notna()].copy()
        y["bin"] = pd.cut(y[col], BIN_EDGES, labels=False)
        y["pay"] = y["win"] * y["tan_odds"]
        di = pd.Series(dc[codes[y.index.values]], index=y.index)
        nd = len(ud)
        agg = {}
        for k in ("win", "q", "p_adj", "pay"):
            a = np.zeros((nd, 5)); np.add.at(a, (di.values, y["bin"].values.astype(int)), y[k].values); agg[k] = a
        cn = np.zeros((nd, 5)); np.add.at(cn, (di.values, y["bin"].values.astype(int)), 1.0)
        I = rng.integers(0, nd, (2000, nd))
        W = np.stack([np.bincount(i, minlength=nd) for i in I]).astype(float)   # (2000, nd)
        S = {k: W @ a for k, a in agg.items()}; SN = W @ cn
        ae_adj_b = S["win"] / S["p_adj"]; roi_b = 100 * S["pay"] / SN
        bins = []
        for b in range(5):
            bins.append({"bin": b + 1, "n": int(cn[:, b].sum()), "wins": int(agg["win"][:, b].sum()),
                         "mean_odds_implied_p": float(agg["q"][:, b].sum() / cn[:, b].sum()),
                         "AE_raw": float(agg["win"][:, b].sum() / agg["q"][:, b].sum()),
                         "AE_flb_adj": float(agg["win"][:, b].sum() / agg["p_adj"][:, b].sum()),
                         "AE_flb_adj_ci95": [float(np.percentile(ae_adj_b[:, b], 2.5)), float(np.percentile(ae_adj_b[:, b], 97.5))],
                         "roi_win_flat": float(100 * agg["pay"][:, b].sum() / cn[:, b].sum()),
                         "roi_ci95": [float(np.percentile(roi_b[:, b], 2.5)), float(np.percentile(roi_b[:, b], 97.5))]})
        dae = ae_adj_b[:, 4] - ae_adj_b[:, 0]; droi = roi_b[:, 4] - roi_b[:, 0]
        # 事後 (ii): z が有る馬 (= 坂路の最終追い切りが有り厩舎履歴 50 本以上) / 無い馬の A/E
        hz = x[col].notna().values.astype(int); dall = dc[codes]
        aw = np.zeros((nd, 2)); ap = np.zeros((nd, 2))
        np.add.at(aw, (dall, hz), x["win"].values); np.add.at(ap, (dall, hz), x["p_adj"].values)
        ae2 = (W @ aw) / (W @ ap)
        have = {lab: {"AE_flb_adj": float(aw[:, k].sum() / ap[:, k].sum()),
                      "ci95": [float(np.percentile(ae2[:, k], 2.5)), float(np.percentile(ae2[:, k], 97.5))]}
                for k, lab in ((1, "z_defined"), (0, "z_missing"))}
        res[pname] = {"n_races": int(n), "n_runners_with_z": int(len(y)), "flb_exponent_a": float(w0[0]),
                      "b_zA": float(w1[1]), "b_ci95": ci_b, "b_excludes_0": bool(ci_b[0] > 0 or ci_b[1] < 0),
                      "d_nll_in_sample": float(nll1 - nll0), "bins": bins,
                      "top_minus_bottom_AE_adj": float(bins[4]["AE_flb_adj"] - bins[0]["AE_flb_adj"]),
                      "top_minus_bottom_AE_adj_ci95": [float(np.percentile(dae, 2.5)), float(np.percentile(dae, 97.5))],
                      "top_minus_bottom_roi": float(bins[4]["roi_win_flat"] - bins[0]["roi_win_flat"]),
                      "top_minus_bottom_roi_ci95": [float(np.percentile(droi, 2.5)), float(np.percentile(droi, 97.5))],
                      "posthoc_AE_by_z_defined": have}
        log(f"[bias{tag} {pname}] races {n:,}  b(zA) {w1[1]:+.4f} [{ci_b[0]:+.4f},{ci_b[1]:+.4f}]  FLB a={w0[0]:.3f}  "
            f"top−bottom A/E {res[pname]['top_minus_bottom_AE_adj']:+.3f}  ROI差 {res[pname]['top_minus_bottom_roi']:+.1f}pt")
        for b in bins:
            log(f"     bin{b['bin']} n={b['n']:,} A/E生 {b['AE_raw']:.3f} A/E補正 {b['AE_flb_adj']:.3f} "
                f"[{b['AE_flb_adj_ci95'][0]:.3f},{b['AE_flb_adj_ci95'][1]:.3f}] 単回収 {b['roi_win_flat']:.1f}% "
                f"[{b['roi_ci95'][0]:.1f},{b['roi_ci95'][1]:.1f}]")
        log(f"     z 有り A/E {have['z_defined']['AE_flb_adj']:.3f} [{have['z_defined']['ci95'][0]:.3f},{have['z_defined']['ci95'][1]:.3f}]"
            f"  z 無し A/E {have['z_missing']['AE_flb_adj']:.3f} [{have['z_missing']['ci95'][0]:.3f},{have['z_missing']['ci95'][1]:.3f}]")
    p1, p2 = res["2015-19"], res["2020-25"]
    res["real"] = bool(p1["b_excludes_0"] and p2["b_excludes_0"] and np.sign(p1["b_zA"]) == np.sign(p2["b_zA"]))
    return res


# ------------------------------------------------------------------ 5. 階層モデル (c)
HCOLS = ["zH_lap1", "zH_t4", "zH_acc", "zH_dd", "hH_lap1", "cH"]


def hier_model(m: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    F_all = np.column_stack([center_fill(m.assign(**{c: m[c].fillna(0.0)}), c) for c in HCOLS])
    yr = m["year"].values; tr = m["調教師コード"].values
    codes = pd.factorize(m["rid_s"])[0]

    def surprise(fit_mask):
        x = m[fit_mask]; cf = pd.factorize(x["rid_s"])[0]
        w, _ = fit_cl(x[["lq"]].values, cf, cf.max() + 1, np.flatnonzero(x["win"].values == 1))
        p = np.exp(gc.race_logsoftmax(w[0] * m["lq"].values, codes, codes.max() + 1))
        return m["win"].values - p, float(w[0])

    def fit(mask, s, scale, lam):
        F = F_all[mask] / scale; y = s[mask]
        beta = np.linalg.lstsq(F, y, rcond=None)[0]
        u = {}
        if np.isfinite(lam):
            r = y - F @ beta; t = tr[mask]
            o = np.argsort(t, kind="stable"); ts = t[o]; bnd = np.flatnonzero(np.r_[True, ts[1:] != ts[:-1], True])
            eye = np.eye(F.shape[1])
            for a, b in zip(bnd[:-1], bnd[1:]):
                Ft = F[o[a:b]]
                u[int(ts[a])] = np.linalg.solve(Ft.T @ Ft + lam * eye, Ft.T @ r[o[a:b]])
        return beta, u

    def predict(mask, scale, beta, u):
        F = F_all[mask] / scale
        pred = F @ beta
        if u:
            U = np.array([u.get(int(t), np.zeros(F.shape[1])) for t in tr[mask]])
            pred = pred + (F * U).sum(1)
        return pred

    info = {"features": HCOLS, "lambdas": [("inf" if not np.isfinite(l) else l) for l in LAMBDAS]}
    # λ 選択: 〜2019 fit → 2020-22
    a_mask = yr <= 2019; b_mask = (yr >= 2020) & (yr <= 2022)
    s_cv, a_cv = surprise(a_mask)
    sc = F_all[a_mask].std(0); sc[sc == 0] = 1.0
    mse0 = float((s_cv[b_mask] ** 2).mean())
    cv = {}
    for lam in LAMBDAS:
        beta, u = fit(a_mask, s_cv, sc, lam)
        pr = predict(b_mask, sc, beta, u)
        cv["inf" if not np.isfinite(lam) else str(int(lam))] = {
            "oos_r2_x1e4": float(1e4 * (1 - ((s_cv[b_mask] - pr) ** 2).mean() / mse0)),
            "corr_pred_surprise": float(np.corrcoef(pr, s_cv[b_mask])[0, 1])}
    best = max(cv, key=lambda k: cv[k]["oos_r2_x1e4"])
    lam_best = np.inf if best == "inf" else float(best)
    info["cv_2020_22"] = cv; info["lambda_chosen"] = best
    info["global_beats_zero_in_cv"] = bool(cv["inf"]["oos_r2_x1e4"] > 0)
    info["trainer_slopes_beat_global_in_cv"] = bool(best != "inf")
    # 本 fit: 〜2022
    f_mask = yr <= 2022
    s_f, a_f = surprise(f_mask)
    sc = F_all[f_mask].std(0); sc[sc == 0] = 1.0
    beta, u = fit(f_mask, s_f, sc, lam_best)
    info["flb_exponent_fit"] = a_f
    info["beta_global_per_sd"] = {c: float(b) for c, b in zip(HCOLS, beta)}
    info["n_fit_rows"] = int(f_mask.sum()); info["n_trainers_with_slopes"] = int(len(u))
    if u:
        U = np.array(list(u.values()))
        info["trainer_slope_sd"] = {c: float(s) for c, s in zip(HCOLS, U.std(0))}
    o_mask = yr >= 2023
    zc = pd.DataFrame({"rid_s": m.loc[o_mask, "rid_s"].values, "ban": m.loc[o_mask, "ban"].values,
                       "zC": predict(o_mask, sc, beta, u)})
    # 対照用: 全厩舎共通 (λ=∞) の予測も作っておく (厩舎別の傾きが効いたのか、共通の傾きだけで足りるのかを切り分ける)
    beta_g, _ = fit(f_mask, s_f, sc, np.inf)
    zc["zC_global"] = predict(o_mask, sc, beta_g, {})
    log(f"[hier] CV(〜2019→2020-22) OOS R²×1e4: " + "  ".join(f"λ={k}:{v['oos_r2_x1e4']:+.3f}" for k, v in cv.items())
        + f"  → λ={best}")
    return zc, info


# ------------------------------------------------------------------ 6. 年別
def gate_by_year(z: pd.DataFrame, zcols: list[str], bcols: list[str], l2: float = 1e-4) -> dict:
    """gate_common.gate と同じ手順 (valid で fit) で、test を 2024 / 2025 に分けて ΔNLL を出す。"""
    b = gc.load_base().merge(z[["rid_s", "ban"] + zcols].drop_duplicates(["rid_s", "ban"]), on=["rid_s", "ban"], how="left")
    for c in zcols:
        b[c] = pd.to_numeric(b[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        b[c] = b[c].fillna(b.groupby("rid_s")[c].transform("mean")).fillna(0.0)
        b[c] = b[c] - b.groupby("rid_s")[c].transform("mean")
        sd = b.loc[b.split == "valid", c].std()
        b[c] = b[c] / (sd if sd and sd > 0 else 1.0)
    if "lk10" in bcols:
        b = b[b["lk10"].notna()]
    xv = b[b.split == "valid"].reset_index(drop=True); xt = b[b.split == "test"].reset_index(drop=True)
    cv = pd.factorize(xv["rid_s"])[0]; ct = pd.factorize(xt["rid_s"])[0]
    wv = np.flatnonzero((xv["jyun"] == 1).values); wt = np.flatnonzero((xt["jyun"] == 1).values)
    lp = {}
    for tag, cols in (("base", bcols), ("plus", bcols + zcols)):
        w = gc._fit(xv[cols].values, cv, cv.max() + 1, wv, l2)
        lp[tag] = gc.race_logsoftmax(xt[cols].values @ w, ct, ct.max() + 1)
    diff = -(lp["plus"][wt] - lp["base"][wt]); yy = xt["date"].astype(str).str[:4].values[wt]; dt = xt["date"].values[wt]
    rng = np.random.default_rng(3); out = {}
    for y in ("2024", "2025"):
        dfy = pd.DataFrame({"d": diff[yy == y], "date": dt[yy == y]})
        g = dfy.groupby("date")["d"].agg(["sum", "size"]); s = g["sum"].values; c = g["size"].values.astype(float)
        I = rng.integers(0, len(s), (2000, len(s))); v = s[I].sum(1) / c[I].sum(1)
        out[y] = {"n": int(len(dfy)), "d_nll": float(dfy.d.mean()), "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]}
    return out


# ------------------------------------------------------------------ main
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    t0 = time.time()
    res = {"experiment": "trainer_language", "date": "2026-11-14",
           "prereg": {"window_days": WINDOW, "min_history": MIN_N, "final_workout_window_days_before": [2, 5],
                      "variants": ["a: zH_lap1", "b: 7-vector", "c: hierarchical predicted surprise"],
                      "bias_rule": "b の符号一致かつ両期間の CI95 が 0 を含まない",
                      "gate_rule": "CI95 上端 < 0 で PASS。P→SIGNAL(予測)、M→SIGNAL(市場外)。PASS は年別符号と placebo で確認"}}
    d = build_features()
    log(f"[features] {len(d):,} 行 (2015-04〜2025)  ({time.time() - t0:.0f}s)")
    cov = d.groupby("year")[["Hf_lap1", "zH_lap1", "hH_lap1", "cH", "Wf_lap1", "zW_lap1"]].agg(lambda s: float(s.notna().mean()))
    log("[coverage]\n" + cov.round(3).to_string())
    res["coverage_by_year"] = {str(y): {c: float(v) for c, v in r.items()} for y, r in cov.iterrows()}
    zs = d[d["zH_lap1"].notna()]
    res["z_sanity"] = {"zA_mean": float(zs.zH_lap1.mean()), "zA_sd": float(zs.zH_lap1.std()),
                       "corr_zA_raw_lap1": float(zs.zH_lap1.corr(zs.raw_lap1)), "corr_zA_placebo": float(zs.zH_lap1.corr(zs.z_placebo)),
                       "corr_zA_zH_t4": float(zs.zH_lap1.corr(zs.zH_t4)), "corr_zA_hH": float(zs.zH_lap1.corr(zs.hH_lap1))}
    log(f"[z] {res['z_sanity']}")

    res["dialect"] = dialect_stats(d)
    for c, v in res["dialect"]["vars"].items():
        log(f"[dialect] {v['label']:20s} ICC {v['icc_trainer']:.3f} (場×年を引いた後 {v['icc_trainer_within_center_year']:.3f})  "
            f"偶奇年の厩舎平均の相関 {v['split_half_r_even_odd_years']:.3f} (n={v['n_trainers_split_half']})")

    m = market_frame(d)
    log(f"[market] {m.rid_s.nunique():,} レース / {len(m):,} 頭 ({time.time() - t0:.0f}s)")
    res["market_bias_a"] = market_bias(m)
    log(f"[bias] 実在判定 = {res['market_bias_a']['real']}  ({time.time() - t0:.0f}s)")
    res["posthoc_market_bias_placebo"] = market_bias(m, col="z_placebo", tag=" 事後placebo")

    zc, res["hier"] = hier_model(m)
    log(f"[hier] {res['hier']['beta_global_per_sd']}  ({time.time() - t0:.0f}s)")

    zdf = d.loc[d.year >= 2023, ["rid_s", "ban", "zH_lap1", "zH_t4", "zH_acc", "zH_dd", "hH_lap1", "cH", "zW_lap1",
                                 "raw_lap1", "z_placebo"]].merge(zc, on=["rid_s", "ban"], how="left")
    B = ["zH_lap1", "zH_t4", "zH_acc", "zH_dd", "hH_lap1", "cH", "zW_lap1"]
    runs = {"a_trainer_rel_lap1": ["zH_lap1"], "b_vector7": B, "c_hier_surprise": ["zC"],
            "ctrl_raw_lap1": ["raw_lap1"], "ctrl_placebo_other_trainer_norm": ["z_placebo"], "ctrl_c_global_slopes": ["zC_global"]}
    res["gates"] = {}
    specs = {"P": ["lm7"], "M": ["lk", "lm7"], "M0": ["lk"], "A10": ["lk10", "lm7"]}
    for name, zc_ in runs.items():
        g = gc.gate(zdf, zc_, name=name)
        passed = [k for k, v in g["gates"].items() if v["verdict"] == "PASS"]
        if passed:
            g["by_year"] = {k: gate_by_year(zdf, zc_, specs[k]) for k in passed}
            for k in passed:
                log(f"   [年別 {name} {k}] " + "  ".join(f"{y}: {v['d_nll']:+.5f} [{v['ci95'][0]:+.5f},{v['ci95'][1]:+.5f}]"
                                                       for y, v in g["by_year"][k].items()))
        res["gates"][name] = g
    v = {}
    for name in ("a_trainer_rel_lap1", "b_vector7", "c_hier_surprise"):
        gg = res["gates"][name]["gates"]
        v[name] = {"P": gg["P"]["verdict"], "M": gg["M"]["verdict"], "M0": gg["M0"]["verdict"], "A10": gg["A10"]["verdict"]}
    res["verdict_table"] = v
    res["elapsed_sec"] = float(time.time() - t0)
    OUT_JSON.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"[done] {OUT_JSON}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
