# -*- coding: utf-8 -*-
"""
tyb_paddock_gate_exp.py — 当日のパドック・直前情報 (JRDB TYB) は締切市場の外にあるか (2026-11-14)
=================================================================================================
実行 (worktree の root から):
    PYTHONUTF8=1 E:/PyCaLiAI/venv311/Scripts/python.exe lab/experiments/tyb_paddock_gate_exp.py
出力: reports/tyb_paddock_gate_exp.json (集計のみ) / reports/tyb_z.parquet (作業用、JRDB 由来・持ち出し禁止)

■ 理論
  v7 (前日までの情報) ≒ 10 時市場。10 時から締切までに 0.079 nats/レース の情報が入る (馬体重・パドック・直前の金)。
  馬体重は EXP19 で「T−28 市場に何も足さない」と閉鎖済み (再走しない)。残る問いはパドック:
  JRDB のスタッフが発走 30〜15 分前に 1 頭ずつ馬体・気配・脚元を見て付ける評価は、締切オッズに織り込み済みか。

■ 入力がいつ分かるか (TYB の odds_time は発走の中央値 17 分前 = TYB の作成時刻)
  F1 前日系        idm, jockey_idx, info_idx                       前日夜〜当日朝 (TYB に転記されているだけ)
  F2 パドック系    paddock_idx, paddock_mark, batai_cd, kehai_cd,  発走 30〜15 分前 (TYB 配信は約 17 分前)
                   ashi_info, bagu_change
  F3 市場混じり    odds_idx, odds_mark, sogo_idx, chokuzen_mark    同上。ただし作成時点のオッズを含む (対照)
  F4 SKB 当日欄    paddock_comment, ashi_comment (句), bagu_1..8,  見たのは発走前だが、このファイルの公開はレース後。
                   hami, bandage, soe, kotsuryu, ashi_sogo_1..3    = パドック観察者が知りうることの上限 (後知恵混入の恐れあり)
                   ※ bagu_comment / race_comment / tokki はレース内容を含むので使わない

■ 事前登録 (test=2024-25 を見る前に固定。ここから下は結果を見て変えない)
  学習・符号化はすべて 〜2022-12-31。valid=2023 は gate が合成重みを fit、test=2024-25 は 1 回だけ。
  カテゴリ符号 TE(code) = log((勝ち数 + 20) / (Σ 1/頭数 + 20))  (〜2022 で fit、未知コードは 0)。
    batai_cd / kehai_cd は TYB には 2021-07 以降しか無いので、〜2022 の fit に限り SED の同名列で補う
    (重なる期間の一致率 99.8%)。2023 以降の z は TYB の値だけを使う。
  欠損の扱い: paddock_mark / odds_mark / chokuzen_mark の空白 = 「印なし」という 1 つのコード (観察なしではない。
    印は全レースで 3〜8 頭に付く)。paddock_idx / odds_idx の 0 = 加点なし。ashi_info / bagu_change の 0 = 特記なし。
  変種 (各族 3 つ以内、全部報告):
    a (主) : 族の列をそのまま z に (数値は生値、コードは TE)。重みは gate が valid=2023 で fit
    b      : LightGBM (binary, 勝ち)。特徴 = 族の列 + レース内偏差・順位 + 頭数。〜2021 で学習、2022 で early stopping。
             ハイパラ固定 (lr 0.05, leaves 15, min_data 300, ff 0.8, bag 0.8, l2 10, 最大 1500 round, patience 50)。z = raw score
    c      : b と同じだが init_score = 確定市場の logit (SED 確定単勝) = 市場の残差だけを学習。z = 木の寄与分。
             F2 / F4 / F1+F2 のみ (M ゲートを直接狙う変種)
  族: F1 (a,b) / F2 (a,b,c) / F3 (a,b) / F4 (a,b,c) / NM = F1+F2 (a,b,c) / ALL = F1+F2+F3 (a,b)
    F4a = 3 列 (パドック句の TE 和、脚元句の TE 和、馬具等コードの TE 和。語彙は 〜2022 の頻度上位、句 = 「。」区切り)
  ゲート: gc.gate の P / M / M0 / A10 (公式値)。加えて参考ゲート T17 = [TYB 作成時点の単勝オッズ + v7] → [+ z]
    (TYB の tan_odds をレース内正規化。gc.gate と同じ手順を本スクリプトで再現して計算、公式 4 ゲートとの一致を assert)。
  判定:
    SIGNAL(予測)   = 主変種 a の P が PASS (CI95 上端 < 0)
    SIGNAL(市場外) = 主変種 a の M が PASS、かつ 2024・2025 の両年で ΔNLL < 0、かつ レース内シャッフルの偽薬が PASS しない
    a は none で b/c だけ M が PASS → 「要再確認 (多重比較)」止まり。SIGNAL とは呼ばない
    それ以外 = none (WORSE は WORSE と書く)
  F2 の副次分析 (探索、判定には使わない): 年別 (2024 / 2025)、パドック印が 6 頭以上に付いたレース / 5 頭以下、
    新馬 / 新馬以外。重みは全レースの valid で fit したものを使い、test の該当レースだけで ΔNLL と CI を出す。
  M が PASS した変種すべてに: 年別の符号、偽薬 (レース内シャッフル、seed 固定 1 回)、v7 の上乗せ (−0.00085) との比。

■ 事後診断 (事前登録の外。1 回目の実行で F1b / F2b / NMb の M が PASS し、z の重みが負だったので追加。判定は変えない)
  M2 = [確定市場 lk, lk^2, lk^3, v7] → [+ z]。M の PASS が「市場の非線形 (人気-不人気の歪み) の代理」なら M2 で消える。
  あわせて M ゲートでの z の重みの符号を記録する (負 = その評価が高い馬は締切時点で買われすぎ)。

■ 断り
  TYB parquet は後日取得した保存ファイル。発走前に配信された値と同じか (後から直されていないか) は未検証。
  JRDB の値・コメント文は JSON / doc に 1 頭単位で出さない (集計のみ)。
"""
from __future__ import annotations

import json
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate_common as gc  # noqa: E402

warnings.filterwarnings("ignore")
TYB = gc.ROOT / "data" / "jrdb" / "tyb.parquet"
SKB = gc.ROOT / "data" / "jrdb" / "skb.parquet"
OUT_JSON = gc.REP / "tyb_paddock_gate_exp.json"
OUT_Z = gc.REP / "tyb_z.parquet"
TRAIN_END, ES_START = 20221231, 20220101
TE_K = 20.0
V7_MARKET_GAIN = -0.00085
LGB_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=300, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, num_threads=4, seed=0, verbose=-1)
F1_NUM, F1_CAT = ["idm", "jockey_idx", "info_idx"], []
F2_NUM, F2_CAT = ["paddock_idx"], ["paddock_mark", "batai", "kehai", "ashi_info", "bagu_change"]
F3_NUM, F3_CAT = ["odds_idx", "sogo_idx"], ["odds_mark", "chokuzen_mark"]
log = gc.log


# ------------------------------------------------------------------ データ
def load_tyb_frame() -> pd.DataFrame:
    t = pd.read_parquet(TYB, columns=["racekey", "ban", "idm", "jockey_idx", "info_idx", "odds_idx", "paddock_idx", "sogo_idx",
                                      "bagu_change", "ashi_info", "torikeshi", "tan_odds", "odds_time", "odds_mark",
                                      "paddock_mark", "chokuzen_mark", "batai_cd", "kehai_cd"])
    s = gc.load_sed(["racekey", "ymd", "ban", "jyun", "ijou", "tan_odds", "batai_cd", "kehai_cd", "hassou"])
    s = s.rename(columns={"tan_odds": "odds_fin", "batai_cd": "batai_sed", "kehai_cd": "kehai_sed"})
    d = t.merge(s, on=["racekey", "ban"], how="inner")
    d["ymd"] = d["ymd"].astype(int)
    for c in ("paddock_mark", "odds_mark", "chokuzen_mark"):
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0).astype(int)
    pre = d["ymd"] <= TRAIN_END
    for c in ("batai", "kehai"):
        a = pd.to_numeric(d[f"{c}_cd"], errors="coerce"); b = pd.to_numeric(d[f"{c}_sed"], errors="coerce")
        d[c] = a.where(~(pre & a.isna()), b).fillna(-1).astype(int)   # 〜2022 だけ SED で補う。2023 以降は TYB のみ
    d["n"] = d.groupby("racekey")["ban"].transform("size")
    d["win"] = (d["jyun"] == 1).astype(float)
    d["started"] = ~d["ijou"].isin([1, 2]) & d["jyun"].notna() & (d["torikeshi"] == 0)
    return d.sort_values(["ymd", "racekey", "ban"]).reset_index(drop=True)


def fit_te(d: pd.DataFrame, col: str) -> dict:
    tr = d[(d.ymd <= TRAIN_END) & d.started]
    g = tr.groupby(col).agg(w=("win", "sum"), e=("n", lambda x: (1.0 / x).sum()))
    return {int(k): float(np.log((r.w + TE_K) / (r.e + TE_K))) for k, r in g.iterrows()}


def lgb_matrix(d: pd.DataFrame, num: list[str], cat: list[str]) -> pd.DataFrame:
    X = pd.DataFrame({"n": d["n"].values}, index=d.index)
    g = d.groupby("racekey")
    for c in num:
        X[c] = d[c].values
        X[c + "_d"] = (d[c] - g[c].transform("mean")).values
        X[c + "_r"] = g[c].rank(ascending=False, method="average").values
    for c in cat:
        X[c] = d[c].astype("category")
    return X


def fit_lgb(X, d: pd.DataFrame, offset: bool, tag: str):
    """〜2021 で学習、2022 で early stopping。offset=True は確定市場の logit を init_score に。全行の raw score を返す。"""
    import lightgbm as lgb
    ok = d.started.values.copy()
    init = None
    if offset:
        inv = (1.0 / d["odds_fin"].where(d["odds_fin"] > 0)).where(d.started)
        p = (inv / inv.groupby(d["racekey"]).transform("sum")).clip(1e-5, 1 - 1e-5)
        init = np.log(p / (1 - p)).values
        ok &= np.isfinite(init)
    tr = ok & (d.ymd.values < ES_START); es = ok & (d.ymd.values >= ES_START) & (d.ymd.values <= TRAIN_END)
    take = (lambda m: X[m]) if sp.issparse(X) else (lambda m: X.loc[m])
    y = d["win"].values
    dtr = lgb.Dataset(take(tr), y[tr], init_score=init[tr] if offset else None, free_raw_data=False)
    des = lgb.Dataset(take(es), y[es], init_score=init[es] if offset else None, reference=dtr)
    bst = lgb.train(LGB_PARAMS, dtr, num_boost_round=1500, valid_sets=[des],
                    callbacks=[lgb.early_stopping(50, verbose=False)])
    it = bst.best_iteration or 0
    log(f"[lgb {tag}] train {tr.sum():,} es {es.sum():,} best_iter {it} es_logloss {bst.best_score['valid_0']['binary_logloss']:.5f}")
    z = bst.predict(X, raw_score=True, num_iteration=it) if it > 0 else np.zeros(len(d))
    return z, {"best_iter": int(it), "es_logloss": float(bst.best_score["valid_0"]["binary_logloss"]),
               "n_train": int(tr.sum()), "n_es": int(es.sum())}


# ------------------------------------------------------------------ F4 (SKB)
def build_f4(d: pd.DataFrame):
    from sklearn.feature_extraction.text import CountVectorizer
    k = pd.read_parquet(SKB, columns=["racekey", "ban", "paddock_comment", "ashi_comment"] + [f"bagu_{i}" for i in range(1, 9)]
                        + ["ashi_sogo_1", "ashi_sogo_2", "ashi_sogo_3", "hami", "bandage", "soe", "kotsuryu"])
    k = d[["racekey", "ban"]].merge(k, on=["racekey", "ban"], how="left")
    tr = ((d.ymd <= TRAIN_END) & d.started).values
    tok = lambda s: [w for w in re.split(r"[。\s　]+", s) if w]
    mats, names = {}, {}
    for col, kmax in (("paddock_comment", 200), ("ashi_comment", 100)):
        txt = k[col].fillna("").astype(str)
        cv = CountVectorizer(tokenizer=tok, lowercase=False, binary=True, max_features=kmax, min_df=300, token_pattern=None)
        cv.fit(txt[tr]); mats[col] = cv.transform(txt).tocsr().astype(np.float32); names[col] = len(cv.vocabulary_)
    assert len(k) == len(d)
    eq = pd.Series("", index=k.index)
    for c in [f"bagu_{i}" for i in range(1, 9)]:
        eq = eq + k[c].map(lambda v: f"b{v} " if isinstance(v, str) else "")
    for c in ("ashi_sogo_1", "ashi_sogo_2", "ashi_sogo_3", "hami", "bandage", "soe", "kotsuryu"):
        eq = eq + k[c].map(lambda v, p=c[:2]: f"{p}{v} " if isinstance(v, str) else "")
    cv = CountVectorizer(tokenizer=str.split, lowercase=False, binary=True, min_df=300, token_pattern=None)
    cv.fit(eq[tr]); mats["equip"] = cv.transform(eq).tocsr().astype(np.float32); names["equip"] = len(cv.vocabulary_)
    filled = {"paddock_comment": k["paddock_comment"].fillna("").str.strip().ne("").values,
              "ashi_comment": k["ashi_comment"].fillna("").str.strip().ne("").values}
    w = d["win"].values[tr]; e = (1.0 / d["n"].values[tr])
    zs = {}
    for g, M in mats.items():
        W = np.asarray(M[tr].T @ w).ravel(); E = np.asarray(M[tr].T @ e).ravel()
        zs[g] = np.asarray(M @ np.log((W + TE_K) / (E + TE_K))).ravel()
    Xs = sp.hstack([sp.csr_matrix(d[["n"]].values.astype(np.float32))] + list(mats.values())).tocsr()
    return zs, Xs, names, filled


# ------------------------------------------------------------------ gate の再現 (レース別の差を得る)
def detail(base: pd.DataFrame, z: pd.DataFrame, zcols: list[str], bcols: list[str], l2: float = 1e-4) -> pd.DataFrame:
    b = base.merge(z[["rid_s", "ban"] + zcols].drop_duplicates(["rid_s", "ban"]), on=["rid_s", "ban"], how="left")
    for c in zcols:
        b[c] = pd.to_numeric(b[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        b[c] = b[c].fillna(b.groupby("rid_s")[c].transform("mean")).fillna(0.0)
        b[c] = b[c] - b.groupby("rid_s")[c].transform("mean")
        sd = b.loc[b.split == "valid", c].std()
        b[c] = b[c] / (sd if sd and sd > 0 else 1.0)
    d = b
    for c in bcols:
        d = d[d[c].notna()]
    part = {}
    for spl in ("valid", "test"):
        x = d[d.split == spl].reset_index(drop=True)
        codes = pd.factorize(x["rid_s"])[0]
        part[spl] = (x, codes, codes.max() + 1, np.flatnonzero((x["jyun"] == 1).values))
    xv, cv, nv, wv = part["valid"]; xt, ct, nt, wt = part["test"]
    lps = {}
    for tag, cols in (("base", bcols), ("plus", bcols + zcols)):
        w = gc._fit(xv[cols].values, cv, nv, wv, l2)
        lps[tag] = gc.race_logsoftmax(xt[cols].values @ w, ct, nt)
    r = pd.DataFrame({"rid_s": xt["rid_s"].values[wt], "date": xt["date"].values[wt],
                      "nb": -lps["base"][wt], "np": -lps["plus"][wt]}, index=ct[wt]).sort_index()
    r["diff"] = r["np"] - r["nb"]
    for tag in ("base", "plus"):
        t = xt.assign(lp=lps[tag]); top = t.loc[t.groupby(ct)["lp"].idxmax()]
        r[f"top_jyun_{tag}"] = top["jyun"].values
    return r


def boot(r: pd.DataFrame, seed: int = 0) -> dict:
    if len(r) == 0:
        return {"n": 0}
    ud, dc = np.unique(r["date"].values, return_inverse=True)
    s = np.bincount(dc, weights=r["diff"].values, minlength=len(ud)); cnt = np.bincount(dc, minlength=len(ud)).astype(float)
    I = np.random.default_rng(seed).integers(0, len(ud), (gc.N_BOOT, len(ud)))
    v = s[I].sum(1) / cnt[I].sum(1)
    ci = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {"n": int(len(r)), "d_nll": float(r["diff"].mean()), "ci95": ci, "se": float(v.std()),
            "verdict": "PASS" if ci[1] < 0 else ("WORSE" if ci[0] > 0 else "none"),
            "hon_win_base": float(100 * (r.top_jyun_base == 1).mean()), "hon_win_plus": float(100 * (r.top_jyun_plus == 1).mean()),
            "hon_top3_base": float(100 * (r.top_jyun_base <= 3).mean()), "hon_top3_plus": float(100 * (r.top_jyun_plus <= 3).mean())}


def shuffle_in_race(z: pd.DataFrame, zcols: list[str], seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    o = z.copy(); o["_u"] = rng.random(len(o))
    perm = o.sort_values(["rid_s", "_u"])[zcols].values
    o = o.sort_values(["rid_s", "ban"]); o[zcols] = perm
    return o.drop(columns="_u")


def fmt(g):
    return f"{g['d_nll']:+.5f} [{g['ci95'][0]:+.5f},{g['ci95'][1]:+.5f}] {g['verdict']}"


# ------------------------------------------------------------------ main
def main():
    t0 = time.time()
    d = load_tyb_frame()
    log(f"[data] TYB×SED {len(d):,} 行  〜2022 {int((d.ymd <= TRAIN_END).sum()):,}  ({time.time() - t0:.0f}s)")
    base = gc.load_base()
    res = {"generated": time.strftime("%Y-%m-%d %H:%M"), "preregistered": "script docstring", "families": {}, "notes": {}}

    # ---- タイミングと保存ファイルの素性 (集計のみ)
    mins = lambda x: (pd.to_numeric(x, errors="coerce") // 100) * 60 + pd.to_numeric(x, errors="coerce") % 100
    rr = d[d.ymd >= 20230101].assign(lead=lambda x: mins(x.hassou) - mins(x.odds_time)).groupby("racekey")["lead"].first()
    x = d[(d.ymd >= 20230101) & (d.tan_odds > 0) & (d.odds_fin > 0)]
    ov = d[(d.batai_cd.notna()) & (d.batai_sed.notna())]
    res["timing"] = {
        "tyb_odds_lead_min_2023on": {"median": float(rr.median()), "p10": float(rr.quantile(.1)), "p90": float(rr.quantile(.9)),
                                     "n_races": int(rr.notna().sum())},
        "tyb_odds_equals_final_share": float((x.tan_odds == x.odds_fin).mean()),
        "tyb_vs_final_mean_abs_logdiff": float(np.abs(np.log(x.tan_odds / x.odds_fin)).mean()),
        "tyb_vs_sed_batai_agree": float((pd.to_numeric(ov.batai_cd) == pd.to_numeric(ov.batai_sed)).mean()),
        "tyb_vs_sed_kehai_agree": float((pd.to_numeric(ov.kehai_cd, errors="coerce") == pd.to_numeric(ov.kehai_sed, errors="coerce")).mean()),
        "n_overlap": int(len(ov))}
    log(f"[timing] {res['timing']}")

    # ---- base へ結合、カバレッジ
    key = d[["rid_s", "ban", "racekey", "ymd", "n", "tan_odds"] + F1_NUM + F2_NUM + F2_CAT + F3_NUM + F3_CAT]
    bz = base.merge(key, on=["rid_s", "ban"], how="left")
    bz["year"] = bz["date"].str[:4]
    bz["venue"] = bz["rid_s"].str[8:10]
    cov = bz.groupby("year").agg(
        horses=("ban", "size"), races=("rid_s", "nunique"), tyb_joined=("racekey", lambda s: s.notna().mean()),
        paddock_mark_1to6=("paddock_mark", lambda s: s.between(1, 6).mean()), paddock_mark_9=("paddock_mark", lambda s: (s == 9).mean()),
        paddock_idx_nonzero=("paddock_idx", lambda s: (s != 0).mean()), batai_filled=("batai", lambda s: (s > 0).mean()),
        kehai_filled=("kehai", lambda s: (s > 0).mean()), ashi_info_nonzero=("ashi_info", lambda s: (s != 0).mean()),
        bagu_change_nonzero=("bagu_change", lambda s: (s != 0).mean()), info_idx_nonzero=("info_idx", lambda s: (s != 0).mean()),
        odds_mark=("odds_mark", lambda s: (s > 0).mean()), chokuzen_mark=("chokuzen_mark", lambda s: (s > 0).mean()))
    res["coverage_by_year"] = json.loads(cov.round(4).to_json(orient="index"))
    res["coverage_by_venue_paddock_mark_1to6"] = {k: round(float(v), 4) for k, v in
                                                 bz.groupby("venue")["paddock_mark"].apply(lambda s: s.between(1, 6).mean()).items()}
    nmark = bz.groupby("rid_s")["paddock_mark"].apply(lambda s: int(s.between(1, 6).sum()))
    res["paddock_marked_horses_per_race"] = {int(k): int(v) for k, v in nmark.value_counts().sort_index().items()}
    res["races_with_any_paddock_mark"] = float((nmark > 0).mean())
    log(cov.round(3).to_string())

    # ---- 記述: コード別の 実勝ち数 / 市場期待 (test のみ、集計)
    te_rows = bz[bz.split == "test"].assign(p=lambda x: np.exp(x.lk), w=lambda x: (x.jyun == 1).astype(float))
    ae = {}
    for c in ("paddock_mark", "batai", "kehai", "ashi_info", "bagu_change", "chokuzen_mark", "odds_mark"):
        g = te_rows.groupby(c).agg(n=("w", "size"), wins=("w", "sum"), exp_mkt=("p", "sum"))
        g = g[g.n >= 300]
        g["a_over_e"] = g.wins / g.exp_mkt
        g["ae_lo"] = (g.wins - 1.96 * np.sqrt(g.wins)) / g.exp_mkt; g["ae_hi"] = (g.wins + 1.96 * np.sqrt(g.wins)) / g.exp_mkt
        ae[c] = json.loads(g.round(3).to_json(orient="index"))
    res["actual_over_final_market_expected_test"] = ae
    # 同じ表を 〜2022 と 2023 でも (SED の確定単勝で市場期待を作る)。test で 1 を外れたセルが過去にも外れているかの確認用
    h = d[d.started & (d.odds_fin > 0)].copy()
    h["p"] = 1.0 / h["odds_fin"]; h["p"] = h["p"] / h.groupby("racekey")["p"].transform("sum")
    h["era"] = np.where(h.ymd <= 20161231, "2011-16", np.where(h.ymd <= TRAIN_END, "2017-22", np.where(h.ymd <= 20231231, "2023", "2024-")))
    aeh = {}
    for c in ("paddock_mark", "kehai"):
        g = h.groupby(["era", c]).agg(n=("win", "size"), wins=("win", "sum"), exp_mkt=("p", "sum"))
        g = g[g.n >= 300]
        g["a_over_e"] = g.wins / g.exp_mkt
        g["ae_lo"] = (g.wins - 1.96 * np.sqrt(g.wins)) / g.exp_mkt; g["ae_hi"] = (g.wins + 1.96 * np.sqrt(g.wins)) / g.exp_mkt
        aeh[c] = {f"{e}|{k}": json.loads(r.round(3).to_json()) for (e, k), r in g.iterrows()}
        if c == "paddock_mark":
            log(g.round(3).to_string())
    res["actual_over_final_market_expected_by_era"] = aeh

    # ---- 参考ゲート T17 用の市場
    inv = 1.0 / bz["tan_odds"].where(bz["tan_odds"] > 0)
    okr = inv.notna().groupby(bz["rid_s"]).transform("all")
    base17 = base.copy()
    base17["lk17"] = np.log(inv.where(okr) / inv.where(okr).groupby(bz["rid_s"]).transform("sum")).values
    res["notes"]["t17_market_coverage_test"] = float(base17.loc[base17.split == "test", "lk17"].notna().mean())
    # 事後診断 M2 用: 市場の非線形 (lk^2, lk^3) を土台に入れる。レース内中心化して valid の sd で割る
    baseM2 = base.copy()
    for pw in (2, 3):
        v = baseM2["lk"] ** pw
        v = v - v.groupby(baseM2["rid_s"]).transform("mean")
        baseM2[f"lk{pw}"] = v / v[baseM2.split == "valid"].std()

    # ---- TE と z の組み立て
    te = {c: fit_te(d, c) for c in F2_CAT + F3_CAT}
    res["notes"]["te_codes"] = {c: len(v) for c, v in te.items()}
    for c in F2_CAT + F3_CAT:
        d["te_" + c] = d[c].map(te[c]).fillna(0.0)
    zsets = {}          # name -> (zcols)
    cols_a = {"F1": F1_NUM, "F2": F2_NUM + ["te_" + c for c in F2_CAT], "F3": F3_NUM + ["te_" + c for c in F3_CAT]}
    cols_a["NM"] = cols_a["F1"] + cols_a["F2"]; cols_a["ALL"] = cols_a["NM"] + cols_a["F3"]
    for f, cs in cols_a.items():
        zsets[f + "a"] = cs
    fam = {"F1": (F1_NUM, F1_CAT), "F2": (F2_NUM, F2_CAT), "F3": (F3_NUM, F3_CAT),
           "NM": (F1_NUM + F2_NUM, F2_CAT), "ALL": (F1_NUM + F2_NUM + F3_NUM, F2_CAT + F3_CAT)}
    lgb_info = {}
    for f, (num, cat) in fam.items():
        X = lgb_matrix(d, num, cat)
        d[f"z_{f}b"], lgb_info[f + "b"] = fit_lgb(X, d, False, f + "b"); zsets[f + "b"] = [f"z_{f}b"]
        if f in ("F2", "NM"):
            d[f"z_{f}c"], lgb_info[f + "c"] = fit_lgb(X, d, True, f + "c"); zsets[f + "c"] = [f"z_{f}c"]
    log(f"[z] TYB 系 完了 ({time.time() - t0:.0f}s)")
    zs4, X4, names4, filled4 = build_f4(d)
    for g, v in zs4.items():
        d["z4_" + g] = v
    zsets["F4a"] = ["z4_paddock_comment", "z4_ashi_comment", "z4_equip"]
    d["z_F4b"], lgb_info["F4b"] = fit_lgb(X4, d, False, "F4b"); zsets["F4b"] = ["z_F4b"]
    d["z_F4c"], lgb_info["F4c"] = fit_lgb(X4, d, True, "F4c"); zsets["F4c"] = ["z_F4c"]
    res["lgb"] = lgb_info
    res["notes"]["f4_vocab_sizes"] = names4
    in_test = ((d.ymd >= 20240101) & d.rid_s.isin(base.rid_s)).values
    res["notes"]["f4_filled_test"] = {k: float(v[in_test].mean()) for k, v in filled4.items()}
    allz = sorted({c for cs in zsets.values() for c in cs})
    Z = d.loc[d.rid_s.isin(base.rid_s), ["rid_s", "ban"] + allz].reset_index(drop=True)
    Z.to_parquet(OUT_Z)
    log(f"[z] 全 z 完了 {len(Z):,} 行 → {OUT_Z} ({time.time() - t0:.0f}s)")

    # ---- 副次分析用のレース属性
    lite = pd.read_parquet(gc.LITE_PQ, columns=["レースID(新/馬番無)", "クラス名", "split"])
    lite = lite[lite.split.isin(["valid", "test"])]
    cls = lite.assign(rid_s=lite["レースID(新/馬番無)"].astype(str).str[:16]).drop_duplicates("rid_s").set_index("rid_s")["クラス名"]
    rattr = pd.DataFrame({"nmark": nmark}).join(cls.rename("cls"))
    rattr["shinba"] = rattr["cls"].astype(str).str.contains("新馬")

    # ---- ゲート
    order = ["F1a", "F1b", "F2a", "F2b", "F2c", "F3a", "F3b", "F4a", "F4b", "F4c", "NMa", "NMb", "NMc", "ALLa", "ALLb"]
    summary = []
    for name in order:
        zc = zsets[name]
        g = gc.gate(Z, zc, name=name)
        r17 = detail(base17, Z, zc, ["lk17", "lm7"]); t17 = boot(r17)
        rM = detail(base, Z, zc, ["lk", "lm7"])
        assert abs(rM["diff"].mean() - g["gates"]["M"]["d_nll"]) < 1e-7, (name, rM["diff"].mean(), g["gates"]["M"]["d_nll"])
        log(f"[gate {name}] T17 n={t17['n']:,}  ΔNLL {fmt(t17)}  ◎3着内 {t17['hon_top3_base']:.2f}→{t17['hon_top3_plus']:.2f}%")
        rM2 = detail(baseM2, Z, zc, ["lk", "lk2", "lk3", "lm7"]); m2 = boot(rM2)
        m2["nll_base"] = float(rM2["nb"].mean()); m2["base_gain_vs_M"] = float(rM2["nb"].mean() - rM["nb"].mean())
        zw = {c: g["gates"]["M"]["w_plus"][c] for c in zc}
        log(f"[gate {name}] M2(事後診断: 土台に lk^2,lk^3) ΔNLL {fmt(m2)}  / M での z の重み {({c: round(w, 4) for c, w in zw.items()})}")
        ent = {"zcols": zc, "gates": g["gates"], "coverage_test": g["coverage_test"], "T17": t17, "M2_posthoc": m2,
               "z_weight_in_M": zw}
        # 年別 (全変種、M / P / A10 / T17)
        rP = detail(base, Z, zc, ["lm7"]); rA = detail(base, Z, zc, ["lk10", "lm7"])
        ent["by_year"] = {gn: {y: boot(r[r.date.str[:4] == y]) for y in ("2024", "2025")}
                          for gn, r in (("P", rP), ("M", rM), ("A10", rA), ("T17", r17))}
        need_placebo = name.startswith("F2") or g["gates"]["M"]["verdict"] == "PASS" or g["gates"]["P"]["verdict"] == "PASS"
        if need_placebo:
            Zs = shuffle_in_race(Z, zc)
            ent["placebo"] = {gn: boot(detail(base, Zs, zc, bc)) for gn, bc in (("P", ["lm7"]), ("M", ["lk", "lm7"]))}
            log(f"[gate {name}] 偽薬 P {fmt(ent['placebo']['P'])} / M {fmt(ent['placebo']['M'])}")
        if name.startswith("F2") or name.startswith("NM"):
            sub = {}
            for gn, r in (("M", rM), ("T17", r17), ("A10", rA), ("P", rP)):
                a = r.join(rattr, on="rid_s")
                sub[gn] = {"nmark_ge6": boot(a[a.nmark >= 6]), "nmark_le5": boot(a[a.nmark <= 5]),
                           "shinba": boot(a[a.shinba]), "non_shinba": boot(a[~a.shinba])}
            ent["subsets"] = sub
            log(f"[gate {name}] M 副次: 印6頭以上 {fmt(sub['M']['nmark_ge6'])} / 5頭以下 {fmt(sub['M']['nmark_le5'])} / "
                f"新馬 {fmt(sub['M']['shinba'])} (n={sub['M']['shinba']['n']}) / 新馬以外 {fmt(sub['M']['non_shinba'])}")
        m = g["gates"]["M"]
        se_m = (m["ci95"][1] - m["ci95"][0]) / 3.92
        sd_eff = se_m * np.sqrt(m["n_test_races"])
        ent["forward_n"] = {"sd_eff_per_race": float(sd_eff),
                            "races_to_detect_observed": (float(((1.96 + 0.84) * sd_eff / abs(m["d_nll"])) ** 2) if m["d_nll"] < 0 else None),
                            "races_to_detect_v7_size": float(((1.96 + 0.84) * sd_eff / abs(V7_MARKET_GAIN)) ** 2)}
        ent["ratio_to_v7_market_gain"] = float(m["d_nll"] / V7_MARKET_GAIN)
        res["families"][name] = ent
        summary.append((name, g["gates"], t17))

    # ---- 判定 (事前登録どおり)
    verdicts = {}
    for f in ("F1", "F2", "F3", "F4", "NM", "ALL"):
        a = res["families"][f + "a"]
        v = []
        if a["gates"]["P"]["verdict"] == "PASS":
            v.append("SIGNAL(予測)")
        if a["gates"]["M"]["verdict"] == "PASS":
            yrs = all(a["by_year"]["M"][y]["d_nll"] < 0 for y in ("2024", "2025"))
            plc = a.get("placebo", {}).get("M", {}).get("verdict") != "PASS"
            v.append("SIGNAL(市場外)" if (yrs and plc) else "M PASS だが年別/偽薬で不成立")
        elif any(res["families"].get(f + s, {}).get("gates", {}).get("M", {}).get("verdict") == "PASS" for s in ("b", "c")):
            v.append("要再確認(多重比較: b/c のみ M PASS)")
        if a["gates"]["M"]["verdict"] == "WORSE":
            v.append("M WORSE")
        verdicts[f] = v or ["none"]
    res["verdicts"] = verdicts
    log("\n=== まとめ (ΔNLL nats/レース, 負=改善) ===")
    for name, g, t17 in summary:
        log(f"{name:5s} P {fmt(g['P'])} | M {fmt(g['M'])} | M2 {fmt(res['families'][name]['M2_posthoc'])} | M0 {fmt(g['M0'])} | A10 {fmt(g['A10'])} | T17 {fmt(t17)}"
            f" | ◎3着内 P {g['P']['hon_top3_base']:.2f}→{g['P']['hon_top3_plus']:.2f} A10 {g['A10']['hon_top3_base']:.2f}→{g['A10']['hon_top3_plus']:.2f}")
    log(f"判定: {verdicts}")
    OUT_JSON.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"→ {OUT_JSON} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    main()
