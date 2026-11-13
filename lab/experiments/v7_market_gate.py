# -*- coding: utf-8 -*-
"""
v7_market_gate.py — 「JRDB は市場が知らない情報か」の最初のゲート (2026-11-14)
=================================================================================
理論: パリミュチュエルで控除を越えるには、締切時点の市場が持っていない情報が要る。
      120 特徴の v6 は確定オッズに足しても勝ち馬の情報をほぼ足さなかった (a≈0.02、WIN_MODEL_EXP_20261113)。
      v7 は初めて入った「新しい情報源」(JRDB 過去走 59 列) を持つ。これが市場の外にあるなら a が 0 から離れ、
      合成の NLL が市場単独より有意に下がるはず。JRDB も織り込み済みなら v6 と同じく a≈0。

手順 (本番の pkl をそのまま使う。再学習しない):
  1. master (valid=2023 / test=2024-25) に JRDB 特徴を足し、本番 v6 と v7 の生スコアを出す
  2. 勝率 = レース内 softmax(β·s)、β は valid の勝ち馬尤度で推定
  3. 市場 = 確定単勝オッズの逆数をレース内正規化 (kekka_1986_2025_enhanced.csv、読み取りのみ)
  4. log p = a·log p_model + b·log p_mkt (条件付きロジット) を valid で fit、test で市場単独との NLL 差
  5. 開催日クラスタ bootstrap。v7 と v6 の差も同じ resample で

事前に決めた判定 (test):
  PASS  = v7 合成の NLL が市場単独より下がり CI が 0 を跨がない、かつ v7−v6 の NLL 差も CI が 0 を跨がない
  FAIL  = どちらかが CI 跨ぎ → JRDB 過去走も締切市場に織り込み済み
実行 (どこからでも): E:/PyCaLiAI/venv311/Scripts/python.exe lab/experiments/v7_market_gate.py
出力: reports/v7_market_gate.json (JRDB のデータ値は書かない)
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize

warnings.filterwarnings("ignore")
ROOT = Path("E:/PyCaLiAI")
sys.path.insert(0, str(ROOT))
import backtest_pl_ev as BT  # noqa: E402
import jrdb_features as jf  # noqa: E402
import optuna_v6_marks as ovm  # noqa: E402

MASTER = ROOT / "data" / "master_v2_20130105-20251228.csv"
EXT = Path(r"E:\競馬過去走データ\raw_data\kekka_1986_2025_enhanced.csv")
OUT = Path(__file__).resolve().parents[2] / "reports" / "v7_market_gate.json"
PLACE_CODE = {"札幌": "01", "函館": "02", "福島": "03", "新潟": "04", "東京": "05", "中山": "06",
              "中京": "07", "京都": "08", "阪神": "09", "小倉": "10"}
N_BOOT = 2000


def log(m):
    print(m, flush=True)


def load_odds():
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


def scores():
    t0 = time.time()
    df = pd.read_csv(MASTER, encoding="utf-8-sig", low_memory=False)
    df[ovm.COL_JYUN] = pd.to_numeric(df[ovm.COL_JYUN], errors="coerce")
    df = df.dropna(subset=[ovm.COL_JYUN, ovm.COL_RID, "split"])
    df = df[df["split"].isin(["valid", "test"])].copy()
    df = jf.add_jrdb_feats(df)
    log(f"[master] valid+test {len(df):,} 行  JRDB 前走あり {100 * jf.coverage(df):.1f}%  ({time.time() - t0:.0f}s)")
    out = pd.DataFrame({"rid_s": df[ovm.COL_RID].astype(str).str[:16].values,
                        "ban": pd.to_numeric(df["馬番"], errors="coerce").values,
                        "jyun": df[ovm.COL_JYUN].values, "split": df["split"].values,
                        "date": df["日付"].astype(str).values})
    for tag in ("v6", "v7"):
        b = joblib.load(ROOT / "models" / f"unified_rank_{tag}.pkl")
        enc = BT.apply_encoders(df, b["encoders"])
        X = enc[list(b["feature_cols"])].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
        out[f"s_{tag}"] = b["model"].predict(X)
        log(f"[score] {tag} {len(b['feature_cols'])} 特徴 ({time.time() - t0:.0f}s)")
    return out


def race_logsoftmax(x, codes, n):
    mx = np.full(n, -np.inf); np.maximum.at(mx, codes, x)
    e = np.exp(x - mx[codes]); s = np.zeros(n); np.add.at(s, codes, e)
    return x - mx[codes] - np.log(s[codes])


def fit_beta(s, codes, n, win):
    f = lambda b: -race_logsoftmax(b[0] * s, codes, n)[win].mean()
    return float(minimize(f, [1.0], method="Nelder-Mead").x[0])


def fit_blend(lm, lk, codes, n, win):
    f = lambda w: -race_logsoftmax(w[0] * lm + w[1] * lk, codes, n)[win].mean()
    r = minimize(f, [0.1, 1.0], method="Nelder-Mead", options={"xatol": 1e-5, "fatol": 1e-8})
    return float(r.x[0]), float(r.x[1])


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    p = scores()
    p = p.dropna(subset=["ban"]); p["ban"] = p["ban"].astype(int)
    p = p.merge(load_odds(), on=["rid_s", "ban"], how="left")
    g = p.groupby("rid_s")
    ok = (g["odds"].transform(lambda s: s.notna().all() and (s > 0).all())
          & (g["jyun"].transform(lambda s: (s == 1).sum()) == 1) & (g["ban"].transform("size") >= 5))
    n0 = p.rid_s.nunique(); p = p[ok].sort_values(["rid_s", "ban"]).reset_index(drop=True)
    log(f"[data] races {n0:,} → {p.rid_s.nunique():,} (全頭オッズ・1着 1 頭・5 頭以上)")
    inv = 1.0 / p["odds"]; p["lk"] = np.log(inv / inv.groupby(p["rid_s"]).transform("sum"))
    p["win"] = p["jyun"] == 1

    res = {"n_races": {}, "beta": {}, "blend": {}, "test": {}}
    parts = {}
    for sp in ("valid", "test"):
        d = p[p.split == sp].reset_index(drop=True)
        codes = pd.factorize(d["rid_s"])[0]; parts[sp] = (d, codes, codes.max() + 1)
        res["n_races"][sp] = int(codes.max() + 1)
    dv, cv, nv = parts["valid"]; dt, ct, nt = parts["test"]
    wv, wt = dv["win"].values, dt["win"].values
    lp = {"mkt": dt["lk"].values}
    for tag in ("v6", "v7"):
        beta = fit_beta(dv[f"s_{tag}"].values, cv, nv, wv); res["beta"][tag] = beta
        lm_v = race_logsoftmax(beta * dv[f"s_{tag}"].values, cv, nv)
        a, b = fit_blend(lm_v, dv["lk"].values, cv, nv, wv); res["blend"][tag] = {"a": a, "b": b}
        lm_t = race_logsoftmax(beta * dt[f"s_{tag}"].values, ct, nt)
        lp[tag] = lm_t
        lp[f"{tag}+mkt"] = race_logsoftmax(a * lm_t + b * dt["lk"].values, ct, nt)
        log(f"[fit] {tag}: β={beta:.3f}  合成 a={a:.4f} b={b:.4f}")

    # レースごとの勝ち馬 NLL → 開催日クラスタ bootstrap
    win_idx = np.where(wt)[0]; race_of = ct[win_idx]
    nll = {k: pd.Series(-v[win_idx], index=race_of).sort_index() for k, v in lp.items()}
    rdate = dt.groupby(ct)["date"].first().sort_index()
    dates = rdate.loc[nll["mkt"].index].values
    ud, dcode = np.unique(dates, return_inverse=True)
    def by_date(s):
        return np.bincount(dcode, weights=s.values, minlength=len(ud))
    cnt = np.bincount(dcode, minlength=len(ud)).astype(float)
    rng = np.random.default_rng(0); I = rng.integers(0, len(ud), (N_BOOT, len(ud)))
    def ci(diff):
        s = by_date(diff); v = s[I].sum(1) / cnt[I].sum(1)
        return [float(diff.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    for k in lp:
        res["test"][f"nll_{k}"] = float(nll[k].mean())
    comps = {"v6単独−市場": ("v6", "mkt"), "v7単独−市場": ("v7", "mkt"), "v7単独−v6単独": ("v7", "v6"),
             "v6+市場−市場": ("v6+mkt", "mkt"), "v7+市場−市場": ("v7+mkt", "mkt"), "v7+市場−v6+市場": ("v7+mkt", "v6+mkt")}
    res["diff"] = {}
    for name, (x, y) in comps.items():
        res["diff"][name] = ci(nll[x] - nll[y])
        m, lo, hi = res["diff"][name]
        log(f"[test] {name}: ΔNLL {m:+.5f} [{lo:+.5f},{hi:+.5f}]")
    # 年ごとの再現 (判定には使わない、記述のみ)
    yr = pd.Series(dates).str[:4].values
    res["by_year"] = {}
    for y in sorted(set(yr)):
        m = yr == y
        res["by_year"][y] = {k: float((nll[a].values[m] - nll[b].values[m]).mean())
                             for k, (a, b) in (("v7+市場−市場", ("v7+mkt", "mkt")), ("v6+市場−市場", ("v6+mkt", "mkt")),
                                               ("v7+市場−v6+市場", ("v7+mkt", "v6+mkt")))}
        log(f"[year {y}] n={int(m.sum())} " + "  ".join(f"{k} {v:+.5f}" for k, v in res["by_year"][y].items()))
    d7 = res["diff"]["v7+市場−市場"]; d76 = res["diff"]["v7+市場−v6+市場"]
    res["verdict"] = "PASS" if (d7[2] < 0 and d76[2] < 0) else "FAIL"
    log(f"[verdict] {res['verdict']}  (n test races {res['n_races']['test']:,})")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"saved {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
