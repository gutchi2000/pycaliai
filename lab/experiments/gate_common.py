# -*- coding: utf-8 -*-
"""
gate_common.py — 新理論を同じ物差しで測るための共通部品 (2026-11-14)
=====================================================================
どの理論も「発走前に分かる 1 頭ごとの数値 z」を作り、次の 4 つのゲートで測る。
学習済みの本番 v7 (〜2022 学習) を土台に、valid=2023 で重みを fit、test=2024-25 で 1 回だけ評価。

  P   予測ゲート : [v7]            → [v7 + z]            z は v7 が知らないことを足すか
  M   市場ゲート : [確定市場 + v7]  → [確定市場 + v7 + z]  z は締切市場の外にあるか (本丸)
  M0  市場単独   : [確定市場]       → [確定市場 + z]
  A10 朝ゲート   : [10時市場 + v7]  → [10時市場 + v7 + z]  朝の時点の予想を良くするか

指標は勝ち馬の NLL (nats/レース) の差と開催日クラスタ bootstrap の CI95。負 = 改善。
CI の上端が 0 未満なら PASS。P と A10 は ◎(確率最大) の 1 着率・3 着内率も出す。

使い方 (worktree の root から E:/PyCaLiAI/venv311/Scripts/python.exe で):
    from lab.experiments import gate_common as gc      # もしくは sys.path に lab/experiments を足して import gate_common
    base = gc.load_base()            # valid+test の 1 頭 1 行。列は下の BASE_COLS
    lite = gc.load_master_lite()     # 2013-2025 全走の軽量 master (特徴づくり用、結果列 着順 を含むので as-of に注意)
    res = gc.gate(z_df, ["z1", "z2"], name="my_theory")   # z_df: rid_s, ban, z1, z2 (base に無い行は無視、NaN はレース平均で埋める)

初回だけ: python lab/experiments/gate_common.py --build   (reports/gate_base.parquet と reports/master_lite.parquet を作る)
JRDB 由来の値 (v7 スコア・10時オッズ) を含むので reports/ から外へ出さない (reports/ は git 管理外)。
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

warnings.filterwarnings("ignore")
ROOT = Path("E:/PyCaLiAI")
WT = Path(__file__).resolve().parents[2]
REP = WT / "reports"
BASE_PQ = REP / "gate_base.parquet"
LITE_PQ = REP / "master_lite.parquet"
MASTER = ROOT / "data" / "master_v2_20130105-20251228.csv"
SED = ROOT / "data" / "jrdb" / "sed.parquet"
EXT = Path(r"E:\競馬過去走データ\raw_data\kekka_1986_2025_enhanced.csv")
PLACE_CODE = {"札幌": "01", "函館": "02", "福島": "03", "新潟": "04", "東京": "05", "中山": "06",
              "中京": "07", "京都": "08", "阪神": "09", "小倉": "10"}
LITE_COLS = ["日付", "場所", "Ｒ", "レースID(新)", "レースID(新/馬番無)", "血統登録番号", "枠番", "馬番", "馬名", "芝・ダ", "距離",
             "コース区分", "芝(内・外)", "馬場状態", "天気", "クラス名", "トラックコード(JV)", "発走時刻", "前走レースID(新)", "前走日付",
             "種牡馬", "父タイプ名", "母父馬", "母父タイプ名", "騎手コード", "調教師コード", "生産者", "性別", "年齢", "斤量", "ブリンカー",
             "間隔", "休み明け～戦目", "着順", "出走頭数", "split", "前走馬体重", "前走馬体重増減", "kako5_avg_pos", "kako5_std_pos",
             "trnH_Time1", "trnH_Time2", "trnH_Time3", "trnH_Time4", "trnH_Lap1", "trnH_Lap2", "trnH_Lap3", "trnH_Lap4", "trnH_days_ago",
             "trnW_5F", "trnW_4F", "trnW_3F", "trnW_Lap1", "trnW_Lap2", "trnW_Lap3", "trnW_days_ago"]
N_BOOT = 2000


def log(m):
    print(m, flush=True)


# ------------------------------------------------------------------ JRDB キー
def jrdb_rid(ymd: pd.Series, racekey: pd.Series) -> pd.Series:
    """JRDB racekey (場2+年2+回1+日1(16進)+R2) と ymd → 16 桁のレース ID (YYYYMMDD+場2+回2+日2+R2 = master のレースID(新/馬番無))"""
    rk = racekey.astype(str)
    return (ymd.astype(str) + rk.str[0:2] + "0" + rk.str[4] + rk.str[5].map(lambda c: "%02d" % int(c, 16)) + rk.str[6:8])


def load_sed(columns=None) -> pd.DataFrame:
    """JRDB SED (1 走 1 行、2011-01〜)。rid_s (16 桁) と date を付ける。td: 1=芝 2=ダ 3=障害。ijou==0 が通常の完走。"""
    d = pd.read_parquet(SED, columns=columns)
    if {"ymd", "racekey"} <= set(d.columns):
        d["rid_s"] = jrdb_rid(d["ymd"], d["racekey"])
        d["date"] = pd.to_datetime(d["ymd"].astype(str), format="%Y%m%d", errors="coerce")
    return d


# ------------------------------------------------------------------ build
def _load_final_odds():
    use = ["年", "月", "日", "回次", "場所", "日次", "レース番号", "馬番", "単勝オッズ"]
    d = pd.read_csv(EXT, encoding="cp932", usecols=use, dtype=str, low_memory=False)
    for c in ("年", "月", "日", "回次", "日次", "レース番号", "馬番"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["年", "月", "日", "回次", "日次", "レース番号", "馬番"])
    yy = d["年"].astype(int)
    d["year"] = np.where(yy < 100, 2000 + yy, yy)
    d = d[d.year >= 2023]
    d["pcode"] = d["場所"].astype(str).str.strip().map(PLACE_CODE)
    d = d.dropna(subset=["pcode"])
    d["rid_s"] = (d.year.astype(str) + d["月"].astype(int).map("{:02d}".format) + d["日"].astype(int).map("{:02d}".format)
                  + d.pcode + d["回次"].astype(int).map("{:02d}".format) + d["日次"].astype(int).map("{:02d}".format)
                  + d["レース番号"].astype(int).map("{:02d}".format))
    d["ban"] = d["馬番"].astype(int)
    d["odds"] = pd.to_numeric(d["単勝オッズ"], errors="coerce")
    return d[["rid_s", "ban", "odds"]].drop_duplicates(["rid_s", "ban"])


def race_logsoftmax(x, codes, n):
    mx = np.full(n, -np.inf); np.maximum.at(mx, codes, x)
    e = np.exp(x - mx[codes]); s = np.zeros(n); np.add.at(s, codes, e)
    return x - mx[codes] - np.log(s[codes])


def build():
    import joblib
    sys.path.insert(0, str(ROOT))
    import backtest_pl_ev as BT
    import jrdb_features as jf
    import optuna_v6_marks as ovm
    t0 = time.time()
    REP.mkdir(exist_ok=True)
    df = pd.read_csv(MASTER, encoding="utf-8-sig", low_memory=False)
    df[LITE_COLS].to_parquet(LITE_PQ)
    log(f"[lite] {len(df):,} 行 → {LITE_PQ} ({time.time() - t0:.0f}s)")
    df[ovm.COL_JYUN] = pd.to_numeric(df[ovm.COL_JYUN], errors="coerce")
    df = df.dropna(subset=[ovm.COL_JYUN, ovm.COL_RID, "split"])
    df = df[df["split"].isin(["valid", "test"])].copy()
    df = jf.add_jrdb_feats(df)
    out = pd.DataFrame({"rid_s": df[ovm.COL_RID].astype(str).str[:16].values,
                        "ban": pd.to_numeric(df["馬番"], errors="coerce").values,
                        "ped": df["血統登録番号"].astype(str).values,
                        "date": df["日付"].astype(str).str[:8].values, "split": df["split"].values,
                        "jyun": df[ovm.COL_JYUN].values})
    for tag in ("v6", "v7"):
        b = joblib.load(ROOT / "models" / f"unified_rank_{tag}.pkl")
        enc = BT.apply_encoders(df, b["encoders"])
        X = enc[list(b["feature_cols"])].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
        out[f"s_{tag}"] = b["model"].predict(X)
    out = out.dropna(subset=["ban"]); out["ban"] = out["ban"].astype(int)
    out = out.merge(_load_final_odds(), on=["rid_s", "ban"], how="left")
    sed = load_sed(["racekey", "ymd", "ban", "odds10_tan"])
    out = out.merge(sed[["rid_s", "ban", "odds10_tan"]].rename(columns={"odds10_tan": "odds10"}).drop_duplicates(["rid_s", "ban"]),
                    on=["rid_s", "ban"], how="left")
    g = out.groupby("rid_s")
    ok = (g["odds"].transform(lambda s: s.notna().all() and (s > 0).all())
          & (g["jyun"].transform(lambda s: (s == 1).sum()) == 1) & (g["ban"].transform("size") >= 5))
    out = out[ok].sort_values(["rid_s", "ban"]).reset_index(drop=True)
    inv = 1.0 / out["odds"]; out["lk"] = np.log(inv / inv.groupby(out["rid_s"]).transform("sum"))
    ok10 = out.groupby("rid_s")["odds10"].transform(lambda s: s.notna().all() and (s > 0).all())
    inv10 = 1.0 / out["odds10"].where(ok10)
    out["lk10"] = np.log(inv10 / inv10.groupby(out["rid_s"]).transform("sum"))
    v = out[out.split == "valid"]; cv = pd.factorize(v["rid_s"])[0]; nv = cv.max() + 1; wv = (v["jyun"] == 1).values
    codes = pd.factorize(out["rid_s"])[0]; n = codes.max() + 1
    for tag in ("v6", "v7"):
        f = lambda b: -race_logsoftmax(b[0] * v[f"s_{tag}"].values, cv, nv)[wv].mean()
        beta = float(minimize(f, [1.0], method="Nelder-Mead").x[0])
        out[f"lm{tag[1]}"] = race_logsoftmax(beta * out[f"s_{tag}"].values, codes, n)
    out.to_parquet(BASE_PQ)
    log(f"[base] races valid {out[out.split == 'valid'].rid_s.nunique():,} / test {out[out.split == 'test'].rid_s.nunique():,}"
        f"  10時オッズあり {100 * out.lk10.notna().mean():.1f}% → {BASE_PQ} ({time.time() - t0:.0f}s)")


def load_base() -> pd.DataFrame:
    """列: rid_s, ban, ped, date(YYYYMMDD), split(valid/test), jyun, s_v6, s_v7, odds(確定単勝), odds10(10時単勝),
           lk(確定市場の log 勝率), lk10(10時市場、欠けるレースは NaN), lm6, lm7 (v6/v7 の log 勝率)"""
    return pd.read_parquet(BASE_PQ)


def load_master_lite() -> pd.DataFrame:
    return pd.read_parquet(LITE_PQ)


# ------------------------------------------------------------------ gate
def _fit(X, codes, n, win_idx, l2):
    def f(w):
        lp = race_logsoftmax(X @ w, codes, n); p = np.exp(lp)
        g = -(X[win_idx].sum(0) - (p[:, None] * X).sum(0)) / n + 2 * l2 * w
        return -lp[win_idx].sum() / n + l2 * (w ** 2).sum(), g
    w0 = np.zeros(X.shape[1]); w0[0] = 1.0
    return minimize(f, w0, jac=True, method="L-BFGS-B").x


def gate(z: pd.DataFrame, zcols: list[str], name: str = "z", l2: float = 1e-4, verbose: bool = True) -> dict:
    """z: rid_s, ban, zcols。NaN はレース内平均 (全欠損レースは 0) で埋め、valid の sd で割る。"""
    b = load_base().merge(z[["rid_s", "ban"] + zcols].drop_duplicates(["rid_s", "ban"]), on=["rid_s", "ban"], how="left")
    cov = {c: float(b.loc[b.split == "test", c].notna().mean()) for c in zcols}
    for c in zcols:
        b[c] = pd.to_numeric(b[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        b[c] = b[c].fillna(b.groupby("rid_s")[c].transform("mean")).fillna(0.0)
        b[c] = b[c] - b.groupby("rid_s")[c].transform("mean")
        sd = b.loc[b.split == "valid", c].std()
        b[c] = b[c] / (sd if sd and sd > 0 else 1.0)
    res = {"name": name, "zcols": zcols, "coverage_test": cov, "gates": {}}
    specs = {"P": ["lm7"], "M": ["lk", "lm7"], "M0": ["lk"], "A10": ["lk10", "lm7"]}
    rng = np.random.default_rng(0)
    for gname, bcols in specs.items():
        d = b[b["lk10"].notna()] if "lk10" in bcols else b
        part = {}
        for sp in ("valid", "test"):
            x = d[d.split == sp].reset_index(drop=True)
            codes = pd.factorize(x["rid_s"])[0]
            part[sp] = (x, codes, codes.max() + 1, np.flatnonzero((x["jyun"] == 1).values))
        xv, cv, nv, wv = part["valid"]; xt, ct, nt, wt = part["test"]
        out = {}
        for tag, cols in (("base", bcols), ("plus", bcols + zcols)):
            w = _fit(xv[cols].values, cv, nv, wv, l2)
            lp = race_logsoftmax(xt[cols].values @ w, ct, nt)
            out[tag] = (w, lp)
        nb = pd.Series(-out["base"][1][wt], index=ct[wt]).sort_index(); npl = pd.Series(-out["plus"][1][wt], index=ct[wt]).sort_index()
        diff = (npl - nb).values
        dates = xt.groupby(ct)["date"].first().sort_index().values
        ud, dc = np.unique(dates, return_inverse=True)
        s = np.bincount(dc, weights=diff, minlength=len(ud)); cnt = np.bincount(dc, minlength=len(ud)).astype(float)
        I = rng.integers(0, len(ud), (N_BOOT, len(ud)))
        v = s[I].sum(1) / cnt[I].sum(1)
        g = {"n_test_races": int(nt), "nll_base": float(nb.mean()), "nll_plus": float(npl.mean()),
             "d_nll": float(diff.mean()), "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
             "w_plus": {c: float(x) for c, x in zip(bcols + zcols, out["plus"][0])}}
        g["verdict"] = "PASS" if g["ci95"][1] < 0 else ("WORSE" if g["ci95"][0] > 0 else "none")
        if gname in ("P", "A10"):
            for tag in ("base", "plus"):
                t = xt.assign(lp=out[tag][1]); top = t.loc[t.groupby(ct)["lp"].idxmax()]
                g[f"hon_win_{tag}"] = float(100 * (top.jyun == 1).mean()); g[f"hon_top3_{tag}"] = float(100 * (top.jyun <= 3).mean())
        res["gates"][gname] = g
        if verbose:
            extra = (f"  ◎1着 {g['hon_win_base']:.2f}→{g['hon_win_plus']:.2f}%  ◎3着内 {g['hon_top3_base']:.2f}→{g['hon_top3_plus']:.2f}%"
                     if gname in ("P", "A10") else "")
            log(f"[gate {name}] {gname:3s} n={nt:,}  ΔNLL {g['d_nll']:+.5f} [{g['ci95'][0]:+.5f},{g['ci95'][1]:+.5f}] {g['verdict']}{extra}")
    return res


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    if "--build" in sys.argv:
        build()
    b = load_base()
    # 自己点検: 無意味な z (乱数) は none、市場そのものを z にすると P が大きく PASS になるはず
    rng = np.random.default_rng(1)
    gate(b.assign(z=rng.normal(size=len(b))), ["z"], name="乱数(none のはず)")
    gate(b.assign(z=b["lk"]), ["z"], name="確定市場を z に(P が PASS のはず)")
