"""残差学習の診断版: 2024-25 のオフラインデータで「最終市場を土台に、モデルの修正が効く条件はどこか」(分析専用・診断)。

問い: 2026 凍結データの残差学習 (analysis/market_residual_check.py、土台 = T-10 市場) と同じ形
  logit(p) = logit(q) + f(x)、p はレース内正規化
を、標本の大きい 2024-25 のオフラインデータで、土台 q = 最終単勝オッズの逆数のレース内正規化 として当て、
修正が効く / 効かない条件 (ポケット) を条件別に分解する。
**最終市場は購入時点に存在しないので、ここでの結果は単独では買い方の根拠にならない (診断)。**
EV・回収率は計算しない。

事前に固定した設計 (2025 の結果を見てから変えない):
  - 期間: fit = 2024 年 (λ・回数の選択は開催日単位 5 分割 CV、2024 の OOF 予測も作る)、評価 = 2025 年 (1 回)。
      2025 で学習して 2024 を見る逆向きはしない。
  - 採点対象: master_v2 の valid/test (障害なし)、全頭に最終単勝オッズ > 0、1 着がちょうど 1 頭、5 頭以上
      (lab/experiments/v7_market_gate.py と同じ)。最終オッズは kekka_1986_2025_enhanced.csv。
  - 特徴 (2026 版と同じ名前・同じ作り方。late_x_c4 は 2024-25 の届くか予測を作っていないので無し):
      d_model = log m_v6 − log q (m_v6 = softmax(β·s_v6)、β は 2023 で勝ち馬尤度 fit)、score_z (s_v6 のレース内 z)、
      speed_z (speed_index の予測指数 pred のレース内 z)、d_est = log(推定オッズ qhat の正規化) − log q、
      pw_z (power_rating の R前 のレース内 z)、level_z (level_metric.horse_level_raw を runs の過去 5 走から)、
      tk_early / tk_c4 (tenkai の位置モデルの予測 0=前〜1=後)、high_x_early (ハイペース確率 × 序盤位置。
      確率は tenkai の pace_model と meta の cuts/σ)、lq, lq2、CHAOS (m_v6 の正規化エントロピーの 2024 内分位)、
      頭数/18、芝、新馬・未勝利・OP 以上。欠損は 0 (z 系) / 0.5 (位置) と欠損フラグ。連続量は 2024 の平均・SD で標準化。
      ※ speed / est_odds / tenkai のモデルは 〜2022 学習 (2023 早期停止) なので 2024-25 は学習外。
        PW は各レース前の値。level は過去走の着順・人気だけ。
  - モデル: R0 = lq, lq2 だけ (市場の形の補正 = 本命-大穴の偏り)、R1 (主) = 全特徴の L2 ロジスティック、
      R2 (副) = LightGBM (init_score = logit q、2026 版と同じハイパラ)、
      R3 (副・v8 前向き確認の手がかり) = R1 + d_model_v7, score_z_v7 (JRDB 過去走入りの v7。JRDB は後からの改訂で
      水増しの可能性あり、V7_MARKET_GATE_20261114.md)。
  - 指標: 2 値 LogLoss (確率は [0.001, 0.999] に制限、2026 版と同じ)、勝ち馬 NLL (nats/R)。
      差の CI は開催日クラスタのブートストラップ 2,000 回。
  - 条件別の分解 (事前に固定した区分、比較 = R1−q、R1−R0、R0−q、R2−q、R3−R1):
      レース単位 (勝ち馬 NLL の差): クラス、芝ダ、頭数、距離、CHAOS 5 分位、1 番人気の最終オッズ帯、
        AI1位 (m_v6 最大) と 1 番人気の一致。
      馬単位 (2 値 LogLoss の差、1 頭あたり): 人気順の帯、前走からの間隔。
      ポケット候補の条件 (2025 で判定): 差 < 0 かつ CI 上端 < 0、かつ 2024 の OOF でも差 < 0。
      多重比較: 比較ごとに区分全体で Benjamini-Hochberg (q = 0.10) を付記 (p 値はブートストラップの両側)。

使い方:
  python -m analysis.market_residual_offline_2425 build --cache <dir>     # データ組み立て (10 分前後)
  python -m analysis.market_residual_offline_2425 run   --cache <dir> --out-json <path>
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path("E:/PyCaLiAI")
EXT = Path(r"E:\競馬過去走データ\raw_data\kekka_1986_2025_enhanced.csv")
PLACE_CODE = {"札幌": "01", "函館": "02", "福島": "03", "新潟": "04", "東京": "05", "中山": "06",
              "中京": "07", "京都": "08", "阪神": "09", "小倉": "10"}
CODE_PLACE = {v: k for k, v in PLACE_CODE.items()}
LAMBDAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
N_FOLDS = 5
N_BOOT = 2000
LGB_PARAMS = {"objective": "binary", "num_leaves": 4, "learning_rate": 0.02, "min_data_in_leaf": 200,
              "feature_fraction": 0.8, "lambda_l2": 10.0, "verbose": -1, "seed": 42, "deterministic": True,
              "force_col_wise": True}


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


# ================================================================ build
def load_odds() -> pd.DataFrame:
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


def model_scores() -> pd.DataFrame:
    import joblib
    sys.path.insert(0, str(ROOT))
    import backtest_pl_ev as BT
    import jrdb_features as jf
    import optuna_v6_marks as ovm
    df = pd.read_csv(ROOT / "data" / "master_v2_20130105-20251228.csv", encoding="utf-8-sig", low_memory=False)
    df[ovm.COL_JYUN] = pd.to_numeric(df[ovm.COL_JYUN], errors="coerce")
    df = df.dropna(subset=[ovm.COL_JYUN, ovm.COL_RID, "split"])
    df = df[df["split"].isin(["valid", "test"])].copy()
    df = jf.add_jrdb_feats(df)
    out = pd.DataFrame({"rid_s": df[ovm.COL_RID].astype(str).str[:16].values,
                        "ban": pd.to_numeric(df["馬番"], errors="coerce").values,
                        "jyun": df[ovm.COL_JYUN].values, "split": df["split"].values,
                        "horse": df["馬名"].astype(str).str.strip().values})
    for tag in ("v6", "v7"):
        b = joblib.load(ROOT / "models" / f"unified_rank_{tag}.pkl")
        enc = BT.apply_encoders(df, b["encoders"])
        X = enc[list(b["feature_cols"])].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
        out[f"s_{tag}"] = b["model"].predict(X)
        log(f"score {tag}: {len(b['feature_cols'])} 特徴")
    return out


def level_raw_table(runs: pd.DataFrame) -> pd.Series:
    """level_metric.horse_level_raw と同じ式を、各走の「それより前の 5 走」(着順・人気) から。"""
    from level_metric import _POS_SCORE, _RECENCY_W
    r = runs.sort_values(["horse_id", "date", "race_key"])[["race_key", "horse_id", "finish", "ninki"]].copy()
    pos = r["finish"].astype(float)
    base = pos.map(lambda p: _POS_SCORE.get(int(p), max(12.0, 59.0 - (int(p) - 5) * 6.0)) if p == p and p > 0 else np.nan)
    nk = r["ninki"].astype(float)
    beat = np.where(nk > 0, np.clip(nk - pos, -7, 7) * 2.2, 0.0)
    r["rating"] = np.clip(base + beat, 0, 118).where(base.notna())
    g = r.groupby("horse_id")["rating"]
    lags = [g.shift(k) for k in range(1, 6)]
    L = np.column_stack(lags)
    W = np.array(_RECENCY_W)
    # 直近から「評価のある走」を詰めて重み付けする（horse_level_raw は欠けた走を飛ばす）
    vals = np.full(len(r), np.nan)
    for i in range(len(r)):
        row = L[i][~np.isnan(L[i])]
        if len(row):
            w = W[:len(row)]
            vals[i] = 0.7 * float((w * row).sum() / w.sum()) + 0.3 * float(row[:3].max())
    return pd.Series(vals, index=pd.MultiIndex.from_arrays([r["race_key"], r["horse_id"]]))


def tenkai_feats(cache: Path, keep_keys: set) -> tuple[pd.DataFrame, pd.Series]:
    import lightgbm as lgb
    sys.path.insert(0, str(ROOT))
    import speed_index as si
    import tenkai as tk
    work = cache / "tk"
    work.mkdir(parents=True, exist_ok=True)
    for f in ("pos_model_e.txt", "pos_model_l.txt", "pace_model.txt", "meta.json",
              "_corners_a_v1.parquet", "_corners_b_v1.parquet"):
        if not (work / f).exists():
            shutil.copy2(ROOT / "reports/tenkai" / f, work / f)
    tk.OUT_DIR = work                                  # 本番の reports/tenkai には書かない
    tk.SI_DIR = si.OUT_DIR = ROOT / "reports" / "speed_index"
    meta = json.loads((work / "meta.json").read_text(encoding="utf-8"))
    r = tk.runs_with_corners()
    x = tk.pos_features(r)
    x = x[x["race_key"].isin(keep_keys)].copy()
    me = lgb.Booster(model_file=str(work / "pos_model_e.txt"))
    ml = lgb.Booster(model_file=str(work / "pos_model_l.txt"))
    mp = lgb.Booster(model_file=str(work / "pace_model.txt"))
    x["pe"] = me.predict(x[tk.POS_FEATS])
    x["pl"] = ml.predict(x[tk.POS_FEATS])
    P = tk.pace_frame(x)
    mu = mp.predict(P[tk.PACE_FEATS])
    surf = x.groupby("race_key")["surface"].first().reindex(P.index).to_numpy()
    pr = tk.pace_probs(mu, surf, meta["cuts"], meta["sigma"])
    return x[["race_key", "horse_id", "pe", "pl"]], pd.Series(pr[:, 2], index=P.index)


def cmd_build(a) -> None:
    cache = Path(a.cache)
    cache.mkdir(parents=True, exist_ok=True)
    log("model scores …")
    S = model_scores()
    S = S.dropna(subset=["ban"])
    S["ban"] = S["ban"].astype(int)
    S = S.merge(load_odds(), on=["rid_s", "ban"], how="left")
    g = S.groupby("rid_s")
    ok = (g["odds"].transform(lambda s: s.notna().all() and (s > 0).all())
          & (g["jyun"].transform(lambda s: (s == 1).sum()) == 1) & (g["ban"].transform("size") >= 5))
    n0 = S["rid_s"].nunique()
    S = S[ok].copy()
    log(f"races {n0} → {S['rid_s'].nunique()}")
    S["date"] = pd.to_datetime(S["rid_s"].str[:8], format="%Y%m%d")
    S["place"] = S["rid_s"].str[8:10].map(CODE_PLACE)
    S["rno"] = S["rid_s"].str[14:16].astype(int)

    log("runs / speed / market / level …")
    runs = pd.read_parquet(ROOT / "reports/speed_index/runs.parquet",
                           columns=["race_key", "date", "place", "rno", "umaban", "horse_id", "klass", "surface", "dist",
                                    "field", "finish", "ninki"])
    rk = runs[["race_key", "date", "place", "rno", "umaban", "horse_id"]].drop_duplicates(["date", "place", "rno", "umaban"])
    S = S.merge(rk, left_on=["date", "place", "rno", "ban"], right_on=["date", "place", "rno", "umaban"], how="left")
    S = S.drop(columns=["umaban"])
    # レース属性（runs の最初の行）。runs に無いレースは race_key 単位で埋まらない → 後で落とす
    S["race_key"] = S.groupby("rid_s")["race_key"].transform(lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan)
    ra = runs.groupby("race_key").agg(klass=("klass", "first"), surface=("surface", "first"), dist=("dist", "first"))
    S = S.join(ra, on="race_key")
    # 前走からの日数
    rr = runs.sort_values(["horse_id", "date"])
    rr["prev_date"] = rr.groupby("horse_id")["date"].shift(1)
    S = S.merge(rr[["race_key", "horse_id", "prev_date"]], on=["race_key", "horse_id"], how="left")
    S["days"] = (S["date"] - S["prev_date"]).dt.days
    pred = pd.read_parquet(ROOT / "reports/speed_index/pred.parquet", columns=["race_key", "horse_id", "pred"])
    S = S.merge(pred.drop_duplicates(["race_key", "horse_id"]), on=["race_key", "horse_id"], how="left")
    mk = pd.read_parquet(ROOT / "reports/speed_index/market_pred.parquet", columns=["race_key", "umaban", "qhat"])
    S = S.merge(mk.drop_duplicates(["race_key", "umaban"]), left_on=["race_key", "ban"], right_on=["race_key", "umaban"],
                how="left").drop(columns=["umaban"])
    lv = level_raw_table(runs[runs["date"] >= "2015-01-01"]).rename("level")
    S = S.join(lv, on=["race_key", "horse_id"])
    hist = pd.read_csv(ROOT / "reports/power_rating/history.csv", usecols=["race_id", "馬名", "R前"])
    hist["rid_s"] = hist["race_id"].astype(str).str[:16]
    hist = hist.rename(columns={"馬名": "horse", "R前": "pw"}).drop_duplicates(["rid_s", "horse"])
    S = S.merge(hist[["rid_s", "horse", "pw"]], on=["rid_s", "horse"], how="left")

    log("tenkai …")
    keys = set(S.loc[S["split"] == "test", "race_key"].dropna())
    pos, phigh = tenkai_feats(cache, keys)
    S = S.merge(pos.drop_duplicates(["race_key", "horse_id"]), on=["race_key", "horse_id"], how="left")
    S["pace_high"] = S["race_key"].map(phigh)
    S.to_parquet(cache / "horses.parquet")
    t = S[S["split"] == "test"]
    log(f"saved {len(S)} rows; test races {t['rid_s'].nunique()}; missing rate: "
        + ", ".join(f"{c} {t[c].isna().mean():.3f}" for c in ("race_key", "pred", "qhat", "level", "pw", "pe", "pace_high", "days")))


# ================================================================ run
def race_logsoftmax(x, codes, n):
    mx = np.full(n, -np.inf)
    np.maximum.at(mx, codes, x)
    e = np.exp(x - mx[codes])
    s = np.zeros(n)
    np.add.at(s, codes, e)
    return x - mx[codes] - np.log(s[codes])


def fit_beta(s, codes, n, win):
    f = lambda b: -race_logsoftmax(b[0] * s, codes, n)[win].mean()  # noqa: E731
    return float(minimize(f, [1.0], method="Nelder-Mead").x[0])


def zin(df, col):
    g = df.groupby("rid_s")[col]
    sd = g.transform("std").replace(0, np.nan)
    return ((df[col] - g.transform("mean")) / sd).fillna(0.0)


def build_X(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    lq = np.log(df["q"])
    X["d_model"] = np.log(df["m6"]) - lq
    X["score_z"] = zin(df, "s_v6")
    for c, name in (("pred", "speed_z"), ("pw", "pw_z"), ("level", "level_z")):
        X[name + "_na"] = df[c].isna().astype(float)
        X[name] = zin(df.assign(**{c: df[c].astype(float)}), c).where(df[c].notna(), 0.0)
    qh = df["qhat"].astype(float)
    X["d_est_na"] = qh.isna().astype(float)
    pe = qh / qh.groupby(df["rid_s"]).transform("sum")
    X["d_est"] = (np.log(pe) - lq).where(qh.notna() & (qh > 0), 0.0).fillna(0.0)
    X["tk_na"] = df["pe"].isna().astype(float)
    X["tk_early"] = df["pe"].astype(float).fillna(0.5)
    X["tk_c4"] = df["pl"].astype(float).fillna(0.5)
    X["high_x_early"] = df["pace_high"].astype(float).fillna(0.0) * X["tk_early"]
    X["lq"] = lq
    X["lq2"] = lq ** 2
    X["chaos"] = df["chaos"].astype(float)
    X["field"] = df["n"].astype(float) / 18.0
    k = df["klass"].astype(str)
    X["turf"] = (df["surface"] == "芝").astype(float)
    X["shinba"] = (k == "新馬").astype(float)
    X["misho"] = (k == "未勝利").astype(float)
    X["opplus"] = (~k.isin(["新馬", "未勝利", "1勝", "2勝", "3勝"])).astype(float)
    return X


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_logit(X, y, off, lam):
    n, k = X.shape
    Xa = np.hstack([np.ones((n, 1)), X])

    def fg(b):
        z = off + Xa @ b
        p = 1 / (1 + np.exp(-z))
        loss = np.mean(np.logaddexp(0, z) - y * z) + 0.5 * lam * np.sum(b[1:] ** 2)
        g = Xa.T @ (p - y) / n
        g[1:] += lam * b[1:]
        return loss, g
    return minimize(fg, np.zeros(k + 1), jac=True, method="L-BFGS-B", options={"maxiter": 3000}).x


def predict_logit(b, X, off):
    return 1 / (1 + np.exp(-(off + np.hstack([np.ones((len(X), 1)), X]) @ b)))


def race_norm(p, rid):
    return p / pd.Series(p).groupby(rid).transform("sum").to_numpy()


def bin_ll_vec(p, y):
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def date_folds(dates):
    ud = np.array(sorted(set(dates)))
    return [np.isin(dates, ud[i::N_FOLDS]) for i in range(N_FOLDS)]


def fit_r_linear(Xm, y, off, rid, dates, F):
    """2024 で λ を CV 選択 → 2024 全体で学習。2024 の OOF 予測も返す。"""
    folds = date_folds(dates[F])
    idx = np.where(F)[0]
    scores, oof_by_lam = {}, {}
    for lam in LAMBDAS:
        oof = np.zeros(len(idx))
        for te in folds:
            b = fit_logit(Xm[idx[~te]], y[idx[~te]], off[idx[~te]], lam)
            oof[te] = race_norm(predict_logit(b, Xm[idx[te]], off[idx[te]]), rid[idx[te]])
        scores[lam] = float(bin_ll_vec(oof, y[idx]).mean())
        oof_by_lam[lam] = oof
    lam = min(scores, key=scores.get)
    b = fit_logit(Xm[F], y[F], off[F], lam)
    p = race_norm(predict_logit(b, Xm, off), rid)
    p_oof = p.copy()
    p_oof[idx] = oof_by_lam[lam]
    return lam, scores, b, p, p_oof


def fit_r_lgb(Xm, y, off, rid, dates, F):
    import lightgbm as lgb
    folds = date_folds(dates[F])
    idx = np.where(F)[0]
    cvres = lgb.cv(LGB_PARAMS, lgb.Dataset(Xm[F], label=y[F], init_score=off[F]), num_boost_round=2000,
                   folds=[(np.where(~te)[0], np.where(te)[0]) for te in folds],
                   callbacks=[lgb.early_stopping(50, verbose=False)], stratified=False)
    key = [k for k in cvres if k.endswith("-mean")][0]
    n_round = len(cvres[key])
    oof = np.zeros(len(idx))
    for te in folds:
        m = lgb.train(LGB_PARAMS, lgb.Dataset(Xm[idx[~te]], label=y[idx[~te]], init_score=off[idx[~te]]), n_round)
        oof[te] = race_norm(1 / (1 + np.exp(-(off[idx[te]] + m.predict(Xm[idx[te]], raw_score=True)))), rid[idx[te]])
    gbm = lgb.train(LGB_PARAMS, lgb.Dataset(Xm[F], label=y[F], init_score=off[F]), n_round)
    p = race_norm(1 / (1 + np.exp(-(off + gbm.predict(Xm, raw_score=True)))), rid)
    p_oof = p.copy()
    p_oof[idx] = oof
    return n_round, gbm, p, p_oof


class Boot:
    """開催日クラスタのブートストラップ（同じ resample をすべての区分・比較で使う）。"""

    def __init__(self, dates, seed=0):
        self.ud, self.code = np.unique(dates, return_inverse=True)
        self.I = np.random.default_rng(seed).integers(0, len(self.ud), (N_BOOT, len(self.ud)))

    def ci(self, diff, mask=None):
        w = np.ones_like(diff) if mask is None else mask.astype(float)
        if w.sum() == 0:
            return None
        s = np.bincount(self.code, weights=diff * w, minlength=len(self.ud))
        c = np.bincount(self.code, weights=w, minlength=len(self.ud))
        v = s[self.I].sum(1) / np.maximum(c[self.I].sum(1), 1e-9)
        m = float((diff * w).sum() / w.sum())
        p = float(min(1.0, 2 * min((v >= 0).mean(), (v <= 0).mean())))
        return {"n": int(w.sum()), "diff": m, "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))], "p": p}


def bh(pvals, q=0.10):
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    thr = q * (np.arange(1, n + 1)) / n
    passed = p[order] <= thr
    k = np.where(passed)[0].max() + 1 if passed.any() else 0
    out = np.zeros(n, bool)
    out[order[:k]] = True
    return out


def race_cells(R: pd.DataFrame, chaos_edges) -> dict:
    k = R["klass"].astype(str)
    kc = np.select([k == "新馬", k == "未勝利", k == "1勝", k == "2勝", k == "3勝", k.isin(["OP", "L"])],
                   ["新馬", "未勝利", "1勝", "2勝", "3勝", "OP・L"], "重賞")
    cells = {}
    for v in ("新馬", "未勝利", "1勝", "2勝", "3勝", "OP・L", "重賞"):
        cells[f"クラス:{v}"] = kc == v
    for v in ("芝", "ダ"):
        cells[f"芝ダ:{v}"] = (R["surface"] == v).to_numpy()
    n = R["n"].to_numpy()
    for lab, lo, hi in (("〜10頭", 0, 10), ("11〜13頭", 11, 13), ("14〜16頭", 14, 16), ("17〜18頭", 17, 18)):
        cells[f"頭数:{lab}"] = (n >= lo) & (n <= hi)
    d = R["dist"].to_numpy()
    for lab, lo, hi in (("〜1400m", 0, 1400), ("1500〜1800m", 1401, 1800), ("1900〜2200m", 1801, 2200), ("2300m〜", 2201, 9999)):
        cells[f"距離:{lab}"] = (d >= lo) & (d <= hi)
    ch = R["chaos"].to_numpy()
    q = np.searchsorted(chaos_edges, ch, side="right")
    for i in range(5):
        cells[f"CHAOS:Q{i+1}{'(堅い)' if i == 0 else '(荒れ)' if i == 4 else ''}"] = q == i
    fo = R["fav_odds"].to_numpy()
    for lab, lo, hi in (("〜1.9倍", 0, 1.95), ("2.0〜2.9倍", 1.95, 2.95), ("3.0〜4.9倍", 2.95, 4.95), ("5.0倍〜", 4.95, 1e9)):
        cells[f"1番人気オッズ:{lab}"] = (fo >= lo) & (fo < hi)
    ag = R["ai_eq_fav"].to_numpy()
    cells["AI1位と1番人気:一致"] = ag
    cells["AI1位と1番人気:不一致"] = ~ag
    return cells


def horse_cells(H: pd.DataFrame) -> dict:
    cells = {}
    rk = H["pop_rank"].to_numpy()
    for lab, lo, hi in (("1番人気", 1, 1), ("2〜3番人気", 2, 3), ("4〜6番人気", 4, 6), ("7〜9番人気", 7, 9), ("10番人気〜", 10, 99)):
        cells[f"人気:{lab}"] = (rk >= lo) & (rk <= hi)
    dd = H["days"].to_numpy()
    cells["間隔:初出走"] = np.isnan(dd)
    for lab, lo, hi in (("〜5週", 0, 35), ("6〜9週", 36, 63), ("10〜25週", 64, 175), ("26週〜", 176, 99999)):
        cells[f"間隔:{lab}"] = (dd >= lo) & (dd <= hi)
    return cells


def prepare(cache: Path, out: dict):
    S = pd.read_parquet(cache / "horses.parquet")
    S = S.sort_values(["rid_s", "ban"]).reset_index(drop=True)
    S["win"] = (S["jyun"] == 1).astype(int)

    # ---- β（2023 で fit）
    betas = {}
    V23 = S[S["split"] == "valid"].reset_index(drop=True)
    c23 = pd.factorize(V23["rid_s"])[0]
    for tag in ("v6", "v7"):
        betas[tag] = fit_beta(V23[f"s_{tag}"].to_numpy(), c23, c23.max() + 1, V23["win"].to_numpy().astype(bool))
    out["beta"] = betas

    df = S[(S["split"] == "test") & S["race_key"].notna()].reset_index(drop=True)
    out["dropped_no_runs_key_races"] = int(S.loc[(S["split"] == "test") & S["race_key"].isna(), "rid_s"].nunique())
    # runs.parquet は完走した馬だけ → runs に行の無い馬（競走中止・タイム無し）は必ず負ける。
    # その馬の特徴が欠損＝「完走しなかった」の印になり先読みになる（初回実行で判明、係数 −0.52）ので、そういう馬を含むレースを外す。
    leak = df.groupby("rid_s")["horse_id"].transform(lambda s: s.isna().any())
    out["dropped_nonfinisher_races"] = int(df.loc[leak, "rid_s"].nunique())
    df = df[~leak].reset_index(drop=True)
    codes = pd.factorize(df["rid_s"])[0]
    nR = codes.max() + 1
    for tag in ("v6", "v7"):
        df[f"m{tag[1]}"] = np.exp(race_logsoftmax(betas[tag] * df[f"s_{tag}"].to_numpy(), codes, nR))
    inv = 1.0 / df["odds"]
    df["q"] = inv / inv.groupby(df["rid_s"]).transform("sum")
    df["n"] = df.groupby("rid_s")["ban"].transform("size")
    ent = -(df["m6"] * np.log(df["m6"])).groupby(df["rid_s"]).transform("sum") / np.log(df["n"])
    df["year"] = df["date"].dt.year
    F = (df["year"] == 2024).to_numpy()
    V = (df["year"] == 2025).to_numpy()
    rent = ent.groupby(df["rid_s"]).first()
    ryear = df.groupby("rid_s")["year"].first()
    fit_ent = np.sort(rent[ryear == 2024].to_numpy())
    df["chaos"] = np.searchsorted(fit_ent, ent.to_numpy(), side="right") / len(fit_ent)
    chaos_edges = np.array([0.2, 0.4, 0.6, 0.8])
    df["pop_rank"] = df.groupby("rid_s")["odds"].rank(method="first")

    X = build_X(df)
    cols = list(X.columns)
    mu, sd = X[F].mean(), X[F].std().replace(0, 1.0)
    binary = [c for c in cols if c.endswith("_na") or c in ("turf", "shinba", "misho", "opplus")]
    Xs = X.copy()
    for c in cols:
        if c not in binary:
            Xs[c] = (X[c] - mu[c]) / sd[c]
    # v7 の追加特徴
    Xs["d_model_v7"] = np.log(df["m7"]) - np.log(df["q"])
    Xs["score_z_v7"] = zin(df, "s_v7")
    for c in ("d_model_v7", "score_z_v7"):
        Xs[c] = (Xs[c] - Xs.loc[F, c].mean()) / Xs.loc[F, c].std()
    y = df["win"].to_numpy(float)
    off = logit(df["q"].to_numpy())
    rid = df["rid_s"].to_numpy()
    dates = df["date"].dt.strftime("%Y%m%d").to_numpy()
    out["coverage"] = {nm: {"races": int(df.loc[m, "rid_s"].nunique()), "horses": int(m.sum()),
                            "dates": int(df.loc[m, "date"].nunique())} for nm, m in (("fit_2024", F), ("eval_2025", V))}
    out["missing_rate_2024_25"] = {c: round(float(X[c].mean()), 4) for c in cols if c.endswith("_na")}
    return df, Xs, cols, F, V, y, off, rid, dates, chaos_edges


def cmd_run(a) -> None:
    out = {}
    df, Xs, cols, F, V, y, off, rid, dates, chaos_edges = prepare(Path(a.cache), out)
    P = {"q": df["q"].to_numpy()}
    POOF = {"q": df["q"].to_numpy()}
    for name, use in (("R0", ["lq", "lq2"]), ("R1", cols), ("R3", cols + ["d_model_v7", "score_z_v7"])):
        lam, cv, b, p, poof = fit_r_linear(Xs[use].to_numpy(float), y, off, rid, dates, F)
        P[name], POOF[name] = p, poof
        out[name] = {"lambda": lam, "cv_logloss": cv, "coef": dict(sorted(zip(use, map(float, b[1:])), key=lambda t: -abs(t[1])))}
        log(f"{name} λ={lam} cv={cv[lam]:.5f}")
    n_round, gbm, p, poof = fit_r_lgb(Xs[cols].to_numpy(float), y, off, rid, dates, F)
    P["R2"], POOF["R2"] = p, poof
    out["R2"] = {"n_round": n_round,
                 "gain": dict(sorted(zip(cols, map(float, gbm.feature_importance("gain"))), key=lambda t: -t[1])[:12])}
    log(f"R2 rounds {n_round}")

    # ---- 指標（馬ごとの 2 値 LL、レースごとの勝ち馬 NLL）
    widx = np.where(y == 1)[0]
    wrace = rid[widx]
    R = df.iloc[widx][["rid_s", "date", "year", "klass", "surface", "dist", "n", "chaos"]].reset_index(drop=True)
    fav = df.loc[df["pop_rank"] == 1, ["rid_s", "odds", "m6"]].set_index("rid_s")
    aitop = df.loc[df.groupby("rid_s")["m6"].idxmax(), ["rid_s", "ban"]].set_index("rid_s")["ban"]
    favban = df.loc[df["pop_rank"] == 1].set_index("rid_s")["ban"]
    R["fav_odds"] = R["rid_s"].map(fav["odds"]).to_numpy()
    R["ai_eq_fav"] = (R["rid_s"].map(aitop) == R["rid_s"].map(favban)).to_numpy()
    out["race_share"] = {"ai_eq_fav_2025": float(R.loc[R["year"] == 2025, "ai_eq_fav"].mean())}

    def wnll(p):
        return -np.log(np.clip(p[widx], 1e-12, 1))

    comps = {"R1−q": ("R1", "q"), "R1−R0": ("R1", "R0"), "R0−q": ("R0", "q"), "R2−q": ("R2", "q"),
             "R3−q": ("R3", "q"), "R3−R1": ("R3", "R1")}
    rv = (R["year"] == 2025).to_numpy()
    rf = (R["year"] == 2024).to_numpy()
    bootV = Boot(R.loc[rv, "date"].dt.strftime("%Y%m%d").to_numpy(), 0)
    bootF = Boot(R.loc[rf, "date"].dt.strftime("%Y%m%d").to_numpy(), 1)
    hbootV = Boot(dates[V], 2)
    hbootF = Boot(dates[F], 3)

    # 全体
    overall = {}
    for nm, (a_, b_) in comps.items():
        dw = wnll(P[a_]) - wnll(P[b_])
        dwo = wnll(POOF[a_]) - wnll(POOF[b_])
        dl = bin_ll_vec(P[a_], y) - bin_ll_vec(P[b_], y)
        dlo = bin_ll_vec(POOF[a_], y) - bin_ll_vec(POOF[b_], y)
        overall[nm] = {"eval_2025": {"winner_nll": bootV.ci(dw[rv]), "logloss": hbootV.ci(dl[V])},
                       "oof_2024": {"winner_nll": bootF.ci(dwo[rf]), "logloss": hbootF.ci(dlo[F])}}
    out["overall"] = overall
    out["levels_2025"] = {k: {"logloss": float(bin_ll_vec(P[k][V], y[V]).mean()),
                              "winner_nll": float(wnll(P[k])[rv].mean())} for k in P}

    # 区分
    rc = race_cells(R, chaos_edges)
    Hc = horse_cells(df)
    cells_out = {}
    for nm, (a_, b_) in comps.items():
        dw, dwo = wnll(P[a_]) - wnll(P[b_]), wnll(POOF[a_]) - wnll(POOF[b_])
        dl, dlo = bin_ll_vec(P[a_], y) - bin_ll_vec(P[b_], y), bin_ll_vec(POOF[a_], y) - bin_ll_vec(POOF[b_], y)
        rows = {}
        for cn, m in rc.items():
            rows[cn] = {"unit": "race_winner_nll", "eval_2025": bootV.ci(dw[rv], m[rv]), "oof_2024": bootF.ci(dwo[rf], m[rf])}
        for cn, m in Hc.items():
            rows[cn] = {"unit": "horse_logloss", "eval_2025": hbootV.ci(dl[V], m[V]), "oof_2024": hbootF.ci(dlo[F], m[F])}
        keys = [k for k in rows if rows[k]["eval_2025"]]
        flags = bh([rows[k]["eval_2025"]["p"] for k in keys])
        for k, f in zip(keys, flags):
            e, o = rows[k]["eval_2025"], rows[k]["oof_2024"]
            rows[k]["bh_q10"] = bool(f)
            rows[k]["pocket"] = bool(e["diff"] < 0 and e["ci95"][1] < 0 and o and o["diff"] < 0)
            rows[k]["harm"] = bool(e["diff"] > 0 and e["ci95"][0] > 0 and o and o["diff"] > 0)
        cells_out[nm] = rows
    out["cells"] = cells_out
    out["n_cells_per_comp"] = len(rc) + len(Hc)
    out["features"] = cols
    out["chaos_note"] = "CHAOS = m_v6（softmax β）の正規化エントロピーを 2024 年のレース分布で分位化。配信の chaos_pct（PL+isotonic、v8 基準表）とは別物"
    Path(a.out_json).write_text(json.dumps(out, ensure_ascii=False, default=str, indent=1), encoding="utf-8")
    # 要約
    for nm, v in overall.items():
        e, o = v["eval_2025"], v["oof_2024"]
        print(f"{nm:7s} 2025 wNLL {e['winner_nll']['diff']:+.5f} [{e['winner_nll']['ci95'][0]:+.5f},{e['winner_nll']['ci95'][1]:+.5f}]"
              f"  LL {e['logloss']['diff']:+.6f} [{e['logloss']['ci95'][0]:+.6f},{e['logloss']['ci95'][1]:+.6f}]"
              f" | 2024OOF wNLL {o['winner_nll']['diff']:+.5f}")
    for nm, rows in cells_out.items():
        pk = [k for k, r in rows.items() if r.get("pocket")]
        hm = [k for k, r in rows.items() if r.get("harm")]
        print(nm, "pocket:", pk, "harm:", hm)


GROUPS = {"v6（d_model, score_z）": ["d_model", "score_z"],
          "スピード指数（speed_z）": ["speed_z", "speed_z_na"],
          "馬レベル（level_z）": ["level_z", "level_z_na"],
          "PW（pw_z）": ["pw_z", "pw_z_na"],
          "推定オッズ（d_est）": ["d_est", "d_est_na"],
          "展開（tk_early, tk_c4, high_x_early）": ["tk_na", "tk_early", "tk_c4", "high_x_early"],
          "市場の形（lq, lq2）": ["lq", "lq2"],
          "レース条件（chaos, 頭数, 芝, クラス）": ["chaos", "field", "turf", "shinba", "misho", "opplus"]}


def cmd_extra(a) -> None:
    """2025 を新たに見ない追加の診断: (1) 特徴グループの寄与（2024 の OOF だけ）、
    (2) 区分間の不均一性の検定と 2024/2025 の区分推定値の相関（run の結果 JSON から）。"""
    from scipy.stats import chi2, pearsonr, spearmanr
    cache = Path(a.cache)
    out = {}
    df, Xs, cols, F, V, y, off, rid, dates, _ = prepare(cache, {})
    widx = np.where(y == 1)[0]
    wf = widx[F[widx]]
    bootF = Boot(dates[wf], 1)

    def oof_wnll(use):
        _, _, _, _, poof = fit_r_linear(Xs[use].to_numpy(float), y, off, rid, dates, F)
        return -np.log(np.clip(poof[wf], 1e-12, 1))
    base_q = -np.log(df["q"].to_numpy()[wf])
    full = oof_wnll(cols)
    r0 = oof_wnll(["lq", "lq2"])
    out["oof_2024_full_minus_q"] = bootF.ci(full - base_q)
    abl = {}
    for g, use in GROUPS.items():
        drop = oof_wnll([c for c in cols if c not in use])
        only = oof_wnll(sorted(set(["lq", "lq2"] + use)))
        abl[g] = {"drop_minus_full": bootF.ci(drop - full), "only_plus_shape_minus_R0": bootF.ci(only - r0)}
        log(f"{g}: drop−full {abl[g]['drop_minus_full']['diff']*1000:+.2f}  only−R0 {abl[g]['only_plus_shape_minus_R0']['diff']*1000:+.2f}")
    out["ablation_oof_2024"] = abl

    res = json.loads(Path(a.result).read_text(encoding="utf-8"))
    het, corr = {}, {}
    for comp in ("R1−q", "R1−R0"):
        rows = res["cells"][comp]
        het[comp] = {}
        for per in ("eval_2025", "oof_2024"):
            dims = {}
            for k, r in rows.items():
                dims.setdefault(k.split(":")[0], []).append(r[per])
            het[comp][per] = {}
            for dname, cs in dims.items():
                d = np.array([c["diff"] for c in cs])
                se = np.array([(c["ci95"][1] - c["ci95"][0]) / 3.92 for c in cs])
                w = 1 / se ** 2
                pooled = (w * d).sum() / w.sum()
                Q = float((w * (d - pooled) ** 2).sum())
                het[comp][per][dname] = {"k": len(cs), "Q": round(Q, 2), "p": round(float(1 - chi2.cdf(Q, len(cs) - 1)), 3)}
        a25 = [r["eval_2025"]["diff"] for r in rows.values()]
        a24 = [r["oof_2024"]["diff"] for r in rows.values()]
        unit = [r["unit"] for r in rows.values()]
        corr[comp] = {}
        for u in ("race_winner_nll", "horse_logloss"):
            m = [i for i, x in enumerate(unit) if x == u]
            corr[comp][u] = {"n_cells": len(m), "pearson": round(float(pearsonr([a24[i] for i in m], [a25[i] for i in m])[0]), 3),
                             "spearman": round(float(spearmanr([a24[i] for i in m], [a25[i] for i in m])[0]), 3)}
    out["heterogeneity"] = het
    out["cell_corr_2024_vs_2025"] = corr
    Path(a.out_json).write_text(json.dumps(out, ensure_ascii=False, default=str, indent=1), encoding="utf-8")
    print(json.dumps({"het": het, "corr": corr}, ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "run", "extra"])
    ap.add_argument("--result", default=None, help="extra: run の結果 JSON")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    {"build": cmd_build, "run": cmd_run, "extra": cmd_extra}[a.cmd](a)


if __name__ == "__main__":
    main()
