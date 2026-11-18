# -*- coding: utf-8 -*-
"""
jrdb_block_audit.py — JRDB の情報ブロックごとに「確定市場への上乗せ」を測る監査 (2026-11-19、分析のみ)
=====================================================================================================
問い: 目的関数の検証 (OBJECTIVE_CHECK_20261119) で、勝率が市場に劣る原因は特徴量の情報量と分かった。
      11/14 の市場ゲートで上乗せを示したのは JRDB 過去走特徴 (v7) だけ。では JRDB のどの情報ブロックが、
      確定単勝オッズの市場に対して追加の情報を持つか。

設計 (test を見る前にここへ固定。探索扱い = 多重比較を補正しても残るかを併記し、採否は v8 の前向きデータで再確認する):
  土台     市場の勝率 p_mkt = 確定単勝オッズの逆数をレース内で正規化 (kekka_1986_2025_enhanced.csv、読み取りのみ)
  残差モデル LightGBM 2 値 (1 着か)。init_score = logit(p_mkt) を入れ、市場で説明できない部分だけを学習する。
           特徴 = ブロックの列 (+ 数値列のレース内 z) + 出走頭数。ハイパラは固定 (下の PARAMS)、seed 42。
  期間     学習 2016-01〜2022-12 (外厩データが 2016 から) / 早期停止とゲートの重み = 2023 (valid) / 評価 = 2024-25 (test、1 回だけ)
  ゲート   条件付きロジット (レース内 softmax) を valid で fit、test で評価。
           単独 (S):  [log p_mkt]          → [log p_mkt, z_B]
           対比 (I):  [log p_mkt, z_N]     → [log p_mkt, z_N, z_{N+B}]   z_N = JRDB 以外の全特徴 (v6 の 120 特徴) の残差、
                                                                      z_{N+B} = 同じ特徴にブロックを足して学習し直した残差
           参照 (N):  [log p_mkt]          → [log p_mkt, z_N]          JRDB 以外の全特徴そのものの上乗せ
  指標     主: 1 頭ごとの 2 値 LogLoss (勝率を [0.001, 0.999] に制限)。副: 勝ち馬の負の対数尤度 (nats/レース、過去のゲートと同じ物差し)。
           差 = plus − base (負 = 改善)。CI は開催日単位の bootstrap 10,000 回 (95%、および族全体の Bonferroni 補正区間)。
           p 値は bootstrap 分布から両側で出し、族 (S・I・参照の全行) で Holm 補正。年別 (2024 / 2025) の向きは記述のみ。
  偽薬     全 JRDB の残差をレース内でシャッフル / 乱数を z にして同じゲートを通す (改善しないはず)。
  対象     master_v2 の各行 (完走扱いの馬) で、全頭の確定オッズあり・1 着がちょうど 1 頭・5 頭以上のレース。

規約: JRDB のデータ・指数は加工しても第三者が見られる場所に出さない。この文書と JSON に書くのは集計 (ΔLogLoss・CI・カバレッジ) だけ。
      中間ファイルは reports/jrdb_block_audit_*.parquet (git 管理外)。

実行 (worktree の root から):
  E:/PyCaLiAI/venv311/Scripts/python.exe -m analysis.jrdb_block_audit build   # 特徴づくり (約 5 分)
  E:/PyCaLiAI/venv311/Scripts/python.exe -m analysis.jrdb_block_audit fit     # 残差モデルの学習 (約 30〜60 分)
  E:/PyCaLiAI/venv311/Scripts/python.exe -m analysis.jrdb_block_audit dev     # valid の中だけで配線を確認 (2023 前半 fit / 後半)
  E:/PyCaLiAI/venv311/Scripts/python.exe -m analysis.jrdb_block_audit eval    # test を 1 回だけ評価 (出力があれば止まる)
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

warnings.filterwarnings("ignore")
WT = Path(__file__).resolve().parents[1]
ROOT = Path("E:/PyCaLiAI")
sys.path.insert(0, str(WT))
REP = WT / "reports"
FEAT_PQ = REP / "jrdb_block_audit_feats.parquet"
RESID_PQ = REP / "jrdb_block_audit_resid.parquet"
FIT_JSON = REP / "jrdb_block_audit_fit.json"
OUT_JSON = WT / "docs" / "research" / "JRDB_BLOCK_AUDIT_20261119.json"
MASTER = ROOT / "data" / "master_v2_20130105-20251228.csv"
JRDB_DIR = ROOT / "data" / "jrdb"
EXT = Path(r"E:\競馬過去走データ\raw_data\kekka_1986_2025_enhanced.csv")
V6 = WT / "models" / "unified_rank_v6.pkl"
if not V6.exists() or V6.stat().st_size < 10_000:
    V6 = ROOT / "models" / "unified_rank_v6.pkl"
PLACE_CODE = {"札幌": "01", "函館": "02", "福島": "03", "新潟": "04", "東京": "05", "中山": "06",
              "中京": "07", "京都": "08", "阪神": "09", "小倉": "10"}
TRAIN_FROM = 20160101
N_BOOT = 10_000
PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_data_in_leaf=500, feature_fraction=0.7,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, max_bin=255, num_threads=14, seed=42, verbose=-1)
MAX_ROUNDS = 4000
EARLY = 100

# ------------------------------------------------------------------ ブロック定義
# num = 連続量 (レース内 z も足す)、code = 区分・順位など (そのまま)。報告書には一般名 (label) だけを出す。
J1_K = (1, 2, 3)


def _j1(cols):
    return [f"j1_{c}_{k}" for k in J1_K for c in cols]


BLOCKS = {
    # --- 過去走 (成績データ。対象レースの 5 日以上前の走だけ、v7 と同じ作り方)
    "P_ABIL": dict(src="過去走", label="過去走の能力評価（走破の数値評価とその補正前の値、近 5 走の平均と最大）",
                   num=_j1(["idm", "soten"]) + ["j1_idm_mean5", "j1_idm_max5"], code=[]),
    "P_TROUBLE": dict(src="過去走", label="過去走の不利・出遅れ・位置取り・コース取りの記録",
                      num=_j1(["furi", "mae_furi", "naka_furi", "ato_furi"]),
                      code=_j1(["deokure", "ichidori", "course_dori", "yon_course"])),
    "P_PACE": dict(src="過去走", label="過去走のペース・区間（テン・上がり）の数値評価",
                   num=_j1(["ten_idx", "agari_idx", "pace_idx", "racep_idx"]), code=_j1(["uma_pace_n", "race_pace_n"])),
    "P_COND": dict(src="過去走", label="過去走時の馬体・気配・上昇度の区分", num=[], code=_j1(["batai", "kehai", "joushou"])),
    # --- 前日 (KYI、発走前日に配布。v8 と同じ作り方)
    "K_ABIL": dict(src="前日", label="前日の能力評価（数値・レース内の偏差と順位）", num=["k8_idm"], code=["k8_idm_z", "k8_idm_rk"]),
    "K_SOGO": dict(src="前日", label="前日の合成評価（能力・騎手・情報を合わせた数値）", num=["k8_sogo_idx"],
                   code=["k8_sogo_idx_z", "k8_sogo_idx_rk"]),
    "K_JOCKEY": dict(src="前日", label="前日の騎手評価", num=["k8_jk_idx"], code=["k8_jk_idx_z", "k8_jk_idx_rk"]),
    "K_INFO": dict(src="前日", label="前日の陣営・取材情報の評価", num=["k8_info_idx"], code=["k8_info_idx_z", "k8_info_idx_rk"]),
    "K_TRAIN": dict(src="前日", label="前日の調教評価", num=["k8_chokyo_idx"], code=[]),
    "K_STABLE": dict(src="前日", label="前日の厩舎評価（数値と区分）", num=["k8_kyusha_idx"], code=["k8_kyusha_hyoka"]),
    "K_UPSET": dict(src="前日", label="前日の穴馬評価（好走の可能性の数値）", num=["k8_gekiso"], code=[]),
    "K_APT": dict(src="前日", label="前日の適性・状態の区分（脚質・距離・上昇度・間隔・道悪・馬具）", num=[],
                  code=["k8_style", "k8_kyori", "k8_joushou", "k8_rot", "k8_omo", "k8_blinker"]),
    "K_PACE": dict(src="前日", label="前日の展開予想（位置・ペース・上がりの予想）", num=["k8_ten", "k8_pidx", "k8_agari", "k8_posidx"],
                   code=["k8_pace_q", "k8_d_sa", "k8_d_io", "k8_a_sa", "k8_a_io", "k8_g_sa", "k8_g_io", "k8_tenkai",
                         "k8_d_jun_rel", "k8_a_jun_rel", "k8_g_jun_rel"]),
    # --- 外厩 (前日、2016〜)
    "GAIKYU": dict(src="外厩", label="外厩（放牧・調整先の施設と期間）", num=[],
                   code=["k8_gk_has", "k8_gk_fac", "k8_gk_days", "k8_gk_weeks"]),
}
GROUPS = {
    "G_PAST": dict(label="過去走ブロックすべて（v7 の追加分）", blocks=["P_ABIL", "P_TROUBLE", "P_PACE", "P_COND"]),
    "G_PREDAY": dict(label="前日ブロックすべて（外厩を除く）",
                     blocks=["K_ABIL", "K_SOGO", "K_JOCKEY", "K_INFO", "K_TRAIN", "K_STABLE", "K_UPSET", "K_APT", "K_PACE"]),
    "G_ALL": dict(label="JRDB すべて（過去走＋前日＋外厩）", blocks=list(BLOCKS)),
}
# JRDB 以外 (v6 の 120 特徴) の中の参照ブロック。単独の上乗せだけを測る (JRDB の同系統と並べて読むため)。
NREF = {
    "N_PED": dict(label="（参照・JRDB 以外）血統（種牡馬・母父・系統・生産者）",
                  cols=["種牡馬", "父タイプ名", "母馬", "母父馬", "母父タイプ名", "毛色", "生産者"]),
    "N_JT": dict(label="（参照・JRDB 以外）騎手・調教師（コード・年齢・直近成績）",
                 cols=["騎手コード", "調教師コード", "騎手年齢", "調教師年齢", "jockey_fuku30", "jockey_fuku90", "trainer_fuku30",
                       "trainer_fuku90", "jockey_n_prev", "jockey_win_rate", "jockey_top3_rate"]),
    "N_TRN": dict(label="（参照・JRDB 以外）調教タイム（坂路・ウッド）",
                  cols=["trnH_Time1", "trnH_Time2", "trnH_Time3", "trnH_Time4", "trnH_Lap1", "trnH_Lap2", "trnH_Lap3", "trnH_Lap4",
                        "trnH_days_ago", "trnW_5F", "trnW_4F", "trnW_3F", "trnW_Lap1", "trnW_Lap2", "trnW_Lap3", "trnW_days_ago"]),
}


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def block_cols(names):
    cols = []
    for n in names:
        b = BLOCKS[n]
        cols += b["num"] + [f"{c}__rz" for c in b["num"]] + b["code"]
    return cols


# ------------------------------------------------------------------ build
def load_odds(min_year: int) -> pd.DataFrame:
    use = ["年", "月", "日", "回次", "場所", "日次", "レース番号", "馬番", "単勝オッズ"]
    d = pd.read_csv(EXT, encoding="cp932", usecols=use, dtype=str, low_memory=False)
    for c in ("年", "月", "日", "回次", "日次", "レース番号", "馬番"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["年", "月", "日", "回次", "日次", "レース番号", "馬番"])
    yy = d["年"].astype(int)
    d["year"] = np.where(yy < 100, 2000 + yy, yy)
    d = d[d.year >= min_year]
    d["pcode"] = d["場所"].astype(str).str.strip().map(PLACE_CODE)
    d = d.dropna(subset=["pcode"])
    d["rid_s"] = (d.year.astype(str) + d["月"].astype(int).map("{:02d}".format) + d["日"].astype(int).map("{:02d}".format)
                  + d.pcode + d["回次"].astype(int).map("{:02d}".format) + d["日次"].astype(int).map("{:02d}".format)
                  + d["レース番号"].astype(int).map("{:02d}".format))
    d["ban"] = d["馬番"].astype(int)
    d["odds"] = pd.to_numeric(d["単勝オッズ"], errors="coerce")
    return d[["rid_s", "ban", "odds"]].drop_duplicates(["rid_s", "ban"])


def build():
    import joblib
    import backtest_pl_ev as BT
    import jrdb_features as jf
    import jrdb_kyi_features as kf
    jf.SED = JRDB_DIR / "sed.parquet"
    kf.KYI_PQ, kf.GK_PQ, kf.FAC_JSON = JRDB_DIR / "kyi.parquet", JRDB_DIR / "gaikyu.parquet", JRDB_DIR / "gaikyu_fac_map.json"
    t0 = time.time()
    b = joblib.load(V6)
    fcols, encs = list(b["feature_cols"]), b["encoders"]
    need = sorted(set(fcols) | set(encs) | {"日付", "レースID(新/馬番無)", "血統登録番号", "馬番", "着順", "split"})
    df = pd.read_csv(MASTER, encoding="utf-8-sig", usecols=lambda c: c in need, low_memory=False)
    df["着順"] = pd.to_numeric(df["着順"], errors="coerce")
    df = df.dropna(subset=["着順", "レースID(新/馬番無)", "split"])
    df = df[pd.to_numeric(df["日付"], errors="coerce") >= TRAIN_FROM].copy()
    log(f"[master] {len(df):,} 行 (2016〜)  ({time.time() - t0:.0f}s)")
    df = jf.add_jrdb_feats(df, history=jf.load_history(jf.SED))
    df = kf.add_master_feats(df)
    log(f"[jrdb] 特徴を付与 ({time.time() - t0:.0f}s)")
    enc = BT.apply_encoders(df, encs)
    out = pd.DataFrame({"rid_s": df["レースID(新/馬番無)"].astype(str).str[:16].values,
                        "ban": pd.to_numeric(df["馬番"], errors="coerce").values,
                        "date": df["日付"].astype(str).str[:8].values, "split": df["split"].values,
                        "jyun": df["着順"].values})
    for i, c in enumerate(fcols):
        out[f"n{i:03d}"] = pd.to_numeric(enc[c], errors="coerce").values
    jcols = sorted({c for blk in BLOCKS.values() for c in blk["num"] + blk["code"]})
    for c in jcols:
        out[c] = pd.to_numeric(df[c], errors="coerce").values
    out = out.dropna(subset=["ban"]); out["ban"] = out["ban"].astype(int)
    out = out.merge(load_odds(2016), on=["rid_s", "ban"], how="left")
    g = out.groupby("rid_s")
    ok = (g["odds"].transform(lambda s: s.notna().all() and (s > 0).all())
          & (g["jyun"].transform(lambda s: (s == 1).sum()) == 1) & (g["ban"].transform("size") >= 5))
    n0 = out.rid_s.nunique()
    out = out[ok].sort_values(["rid_s", "ban"]).reset_index(drop=True)
    log(f"[races] {n0:,} → {out.rid_s.nunique():,} (全頭オッズ・1 着 1 頭・5 頭以上)")
    inv = 1.0 / out["odds"]
    out["p_mkt"] = inv / inv.groupby(out["rid_s"]).transform("sum")
    out["lk"] = np.log(out["p_mkt"])
    out["n_run"] = out.groupby("rid_s")["ban"].transform("size").astype(float)
    for blk in BLOCKS.values():
        for c in blk["num"]:
            gg = out.groupby("rid_s")[c]
            out[f"{c}__rz"] = (out[c] - gg.transform("mean")) / gg.transform("std").replace(0, np.nan)
    meta = {"fcols": fcols}
    REP.mkdir(exist_ok=True)
    out.to_parquet(FEAT_PQ, index=False)
    (REP / "jrdb_block_audit_meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    log(f"[saved] {FEAT_PQ}  {len(out):,} 行 ({time.time() - t0:.0f}s)")


# ------------------------------------------------------------------ fit
def model_specs(ncols):
    """(名前, 特徴列) の一覧。N = JRDB 以外の全特徴、B = ブロック単独、N+B = 対比用。"""
    specs = {"CTX": ["n_run"], "N": ncols + ["n_run"]}
    for name in list(BLOCKS) + list(GROUPS):
        cols = block_cols(GROUPS[name]["blocks"] if name in GROUPS else [name])
        specs[name] = cols + ["n_run"]
        specs[f"N+{name}"] = ncols + cols + ["n_run"]
    meta = json.loads((REP / "jrdb_block_audit_meta.json").read_text(encoding="utf-8"))
    idx = {c: f"n{i:03d}" for i, c in enumerate(meta["fcols"])}
    for name, r in NREF.items():
        specs[name] = [idx[c] for c in r["cols"]] + ["n_run"]
    return specs


def fit(only=None):
    import lightgbm as lgb
    t0 = time.time()
    d = pd.read_parquet(FEAT_PQ)
    ncols = [c for c in d.columns if c.startswith("n") and c[1:].isdigit()]
    specs = model_specs(ncols)
    p = d["p_mkt"].clip(1e-4, 1 - 1e-4)
    d["init"] = np.log(p / (1 - p))
    d["y"] = (d["jyun"] == 1).astype(int)
    tr = d[d.split == "train"]; va = d[d.split == "valid"]
    vt = d[d.split.isin(["valid", "test"])]
    res = pd.read_parquet(RESID_PQ) if (RESID_PQ.exists() and only) else vt[["rid_s", "ban", "date", "split", "jyun", "lk"]].reset_index(drop=True)
    summ = json.loads(FIT_JSON.read_text(encoding="utf-8")) if (FIT_JSON.exists() and only) else {}
    summ["_rows"] = {"train": int(len(tr)), "valid": int(len(va)), "test": int((d.split == "test").sum()),
                     "train_races": int(tr.rid_s.nunique())}
    for name, cols in specs.items():
        if only and name not in only:
            continue
        dtr = lgb.Dataset(tr[cols].values.astype(np.float32), tr["y"].values, init_score=tr["init"].values, free_raw_data=True)
        dva = lgb.Dataset(va[cols].values.astype(np.float32), va["y"].values, init_score=va["init"].values, reference=dtr)
        m = lgb.train(PARAMS, dtr, num_boost_round=MAX_ROUNDS, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(EARLY, verbose=False)])
        res[f"z_{name}"] = m.predict(vt[cols].values.astype(np.float32), num_iteration=m.best_iteration, raw_score=True)
        imp = pd.Series(m.feature_importance("gain"), index=cols)
        jshare = float(imp[[c for c in cols if not (c.startswith("n") and c[1:].isdigit()) and c != "n_run"]].sum() / max(imp.sum(), 1e-12))
        summ[name] = {"n_feats": len(cols), "best_iter": int(m.best_iteration),
                      "valid_logloss": float(m.best_score["valid_0"]["binary_logloss"]), "jrdb_gain_share": jshare}
        log(f"[fit] {name:12s} 特徴 {len(cols):3d}  回数 {m.best_iteration:4d}  valid logloss {summ[name]['valid_logloss']:.5f}"
            f"  JRDB gain {100 * jshare:.1f}%  ({time.time() - t0:.0f}s)")
        res.to_parquet(RESID_PQ, index=False)
        FIT_JSON.write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    # 先にベースラインの valid logloss (init だけ) を記録
    yv = va["y"].values; pv = 1 / (1 + np.exp(-va["init"].values))
    summ["_valid_logloss_init_only"] = float(-np.mean(yv * np.log(pv) + (1 - yv) * np.log(1 - pv)))
    FIT_JSON.write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"[saved] {RESID_PQ}  ({time.time() - t0:.0f}s)")


# ------------------------------------------------------------------ gate
def race_logsoftmax(x, codes, n):
    mx = np.full(n, -np.inf); np.maximum.at(mx, codes, x)
    e = np.exp(x - mx[codes]); s = np.zeros(n); np.add.at(s, codes, e)
    return x - mx[codes] - np.log(s[codes])


def fit_w(X, codes, n, win_idx, l2=1e-4):
    def f(w):
        lp = race_logsoftmax(X @ w, codes, n); p = np.exp(lp)
        g = -(X[win_idx].sum(0) - (p[:, None] * X).sum(0)) / n + 2 * l2 * w
        return -lp[win_idx].sum() / n + l2 * (w ** 2).sum(), g
    w0 = np.zeros(X.shape[1]); w0[0] = 1.0
    return minimize(f, w0, jac=True, method="L-BFGS-B").x


class Part:
    def __init__(self, x: pd.DataFrame):
        self.x = x.reset_index(drop=True)
        self.codes = pd.factorize(self.x["rid_s"])[0]
        self.n = int(self.codes.max() + 1)
        self.y = (self.x["jyun"] == 1).values
        self.win = np.flatnonzero(self.y)
        self.rdate = self.x.groupby(self.codes)["date"].first().sort_index().values
        self.ud, self.dc = np.unique(self.rdate, return_inverse=True)


def per_race_metrics(part: Part, lp):
    """レースごとの (2 値 LogLoss の合計, 頭数, 勝ち馬 NLL)。"""
    p = np.clip(np.exp(lp), 1e-3, 1 - 1e-3)
    ll = -(part.y * np.log(p) + (~part.y) * np.log(1 - p))
    ll_r = np.bincount(part.codes, weights=ll, minlength=part.n)
    cnt_r = np.bincount(part.codes, minlength=part.n).astype(float)
    nll_r = np.zeros(part.n); nll_r[part.codes[part.win]] = -lp[part.win]
    return ll_r, cnt_r, nll_r


def gate_pair(pv: Part, pt: Part, base_cols, plus_cols, zt_override=None):
    """valid で重み、test で評価。戻り: base/plus の test 上の log 勝率、重み。"""
    out = {}
    for tag, cols in (("base", base_cols), ("plus", plus_cols)):
        w = fit_w(pv.x[cols].values, pv.codes, pv.n, pv.win)
        Xt = pt.x[cols].values.copy()
        if tag == "plus" and zt_override is not None:
            Xt[:, -1] = zt_override
        out[tag] = (w, race_logsoftmax(Xt @ w, pt.codes, pt.n))
    return out


def boot_stats(pt: Part, d_ll_r, cnt_r, d_nll_r, I, z_bonf):
    """開催日 bootstrap: LogLoss は頭数加重 (全馬平均の差)、勝ち馬 NLL はレース平均の差。"""
    s_ll = np.bincount(pt.dc, weights=d_ll_r, minlength=len(pt.ud)); s_c = np.bincount(pt.dc, weights=cnt_r, minlength=len(pt.ud))
    s_n = np.bincount(pt.dc, weights=d_nll_r, minlength=len(pt.ud)); s_r = np.bincount(pt.dc, minlength=len(pt.ud)).astype(float)
    out = {}
    for key, num, den in (("d_logloss", s_ll, s_c), ("d_nll", s_n, s_r)):
        v = num[I].sum(1) / den[I].sum(1)
        m = float(num.sum() / den.sum())
        p_two = float(min(1.0, 2 * min((v >= 0).mean(), (v <= 0).mean())))
        p_two = max(p_two, 1.0 / len(v))
        lo_b, hi_b = np.percentile(v, [100 * z_bonf / 2, 100 * (1 - z_bonf / 2)])
        out[key] = {"mean": m, "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
                    "ci_bonf": [float(lo_b), float(hi_b)], "p_boot": p_two, "boot_sd": float(v.std())}
    return out


def evaluate(pv: Part, pt: Part, rows, tag_split: str):
    """rows: [(行名, 種別, base_cols, plus_cols)]。"""
    n_family = len(rows)
    alpha_b = 0.05 / n_family
    rng = np.random.default_rng(20261119)
    I = rng.integers(0, len(pt.ud), (N_BOOT, len(pt.ud)))
    res = {}
    yr = pd.Series(pt.rdate).str[:4].values
    for name, kind, bcols, pcols in rows:
        g = gate_pair(pv, pt, bcols, pcols)
        llb, cnt, nlb = per_race_metrics(pt, g["base"][1]); llp, _, nlp = per_race_metrics(pt, g["plus"][1])
        st = boot_stats(pt, llp - llb, cnt, nlp - nlb, I, alpha_b)
        st["kind"] = kind
        st["logloss_base"] = float(llb.sum() / cnt.sum()); st["logloss_plus"] = float(llp.sum() / cnt.sum())
        st["nll_base"] = float(nlb.mean()); st["nll_plus"] = float(nlp.mean())
        st["w_plus"] = [float(x) for x in g["plus"][0]]
        st["by_year"] = {}
        for y in sorted(set(yr)):
            m = yr == y
            st["by_year"][y] = {"d_logloss": float((llp[m] - llb[m]).sum() / cnt[m].sum()), "d_nll": float((nlp[m] - nlb[m]).mean()),
                                "n_races": int(m.sum())}
        res[name] = st
        a = st["d_logloss"]; b = st["d_nll"]
        log(f"[{tag_split}] {name:14s} {kind:4s} ΔLogLoss {a['mean'] * 1e4:+7.2f}e-4 [{a['ci95'][0] * 1e4:+7.2f},{a['ci95'][1] * 1e4:+7.2f}]"
            f"  ΔNLL {b['mean']:+.5f} [{b['ci95'][0]:+.5f},{b['ci95'][1]:+.5f}]  p={a['p_boot']:.4f}")
    # Holm (主指標 LogLoss の p)
    names = list(res); ps = np.array([res[k]["d_logloss"]["p_boot"] for k in names])
    order = np.argsort(ps); adj = np.empty(len(ps)); run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i])); adj[i] = run
    for k, a in zip(names, adj):
        res[k]["d_logloss"]["p_holm"] = float(a)
        ci = res[k]["d_logloss"]["ci95"]; cb = res[k]["d_logloss"]["ci_bonf"]
        res[k]["verdict"] = ("改善(補正後も)" if cb[1] < 0 else "改善(未補正のみ)" if ci[1] < 0 else
                             "悪化" if ci[0] > 0 else "差なし")
    return res


def rows_for(d: pd.DataFrame):
    rows = [("N", "参照", ["lk"], ["lk", "z_N"])]
    for name in NREF:
        rows.append((name, "参照", ["lk"], ["lk", f"z_{name}"]))
    for name in list(BLOCKS) + list(GROUPS):
        rows.append((f"S:{name}", "単独", ["lk"], ["lk", f"z_{name}"]))
    for name in list(BLOCKS) + list(GROUPS):
        rows.append((f"I:{name}", "対比", ["lk", "z_N"], ["lk", "z_N", f"z_N+{name}"]))
    rows.append(("S:CTX", "対照", ["lk"], ["lk", "z_CTX"]))
    return [r for r in rows if all(c in d.columns for c in r[3])]


def placebo(pv: Part, pt: Part):
    rng = np.random.default_rng(7)
    out = {}
    # 1) 全 JRDB 残差をレース内でシャッフル (valid の重みは本物で fit、test だけ並べ替え)
    zt = pt.x["z_G_ALL"].values.copy()
    perm = pt.x.groupby(pt.codes)["z_G_ALL"].transform(lambda s: rng.permutation(s.values)).values
    I = np.random.default_rng(20261119).integers(0, len(pt.ud), (N_BOOT, len(pt.ud)))
    g = gate_pair(pv, pt, ["lk"], ["lk", "z_G_ALL"], zt_override=perm)
    llb, cnt, nlb = per_race_metrics(pt, g["base"][1]); llp, _, nlp = per_race_metrics(pt, g["plus"][1])
    out["shuffle_G_ALL"] = boot_stats(pt, llp - llb, cnt, nlp - nlb, I, 0.05)
    # 2) 乱数
    pv.x["z_rand"] = rng.normal(size=len(pv.x)); pt.x["z_rand"] = rng.normal(size=len(pt.x))
    g = gate_pair(pv, pt, ["lk"], ["lk", "z_rand"])
    llb, cnt, nlb = per_race_metrics(pt, g["base"][1]); llp, _, nlp = per_race_metrics(pt, g["plus"][1])
    out["random"] = boot_stats(pt, llp - llb, cnt, nlp - nlb, I, 0.05)
    _ = zt
    for k, v in out.items():
        a = v["d_logloss"]
        log(f"[placebo] {k}: ΔLogLoss {a['mean'] * 1e4:+.2f}e-4 [{a['ci95'][0] * 1e4:+.2f},{a['ci95'][1] * 1e4:+.2f}]")
    return out


def coverage(d: pd.DataFrame):
    t = d[d.split == "test"]
    cov = {}
    for name, b in BLOCKS.items():
        cols = b["num"] + b["code"]
        cov[name] = float(t[cols].notna().any(axis=1).mean())
    cov["GAIKYU_has_record"] = float((t["k8_gk_has"] == 1).mean())
    return cov


def dev():
    """valid (2023) の中だけで配線を確認する。前半 (〜06-30) で重み、後半で評価。test は読まない。"""
    r = pd.read_parquet(RESID_PQ)
    v = r[r.split == "valid"]
    pv = Part(v[v.date <= "20230630"]); pt = Part(v[v.date > "20230630"])
    global N_BOOT
    N_BOOT = 2000
    res = evaluate(pv, pt, rows_for(r), "dev")
    placebo(pv, pt)
    return res


def run_eval(force=False):
    if OUT_JSON.exists() and not force:
        raise SystemExit(f"{OUT_JSON} がある。test の評価は 1 回だけ (やり直すなら理由を記録して --force)")
    r = pd.read_parquet(RESID_PQ)
    pv = Part(r[r.split == "valid"]); pt = Part(r[r.split == "test"])
    rows = rows_for(r)
    res = {"design": {"train": "2016-01〜2022-12", "valid": "2023 (早期停止とゲートの重み)", "test": "2024-25 (1 回だけ)",
                      "n_boot": N_BOOT, "family_size": len(rows), "params": PARAMS,
                      "market": "確定単勝オッズの逆数をレース内正規化", "primary": "2 値 LogLoss (勝率を [0.001,0.999] に制限、全馬平均)",
                      "secondary": "勝ち馬 NLL (nats/レース)"},
           "n_races": {"valid": pv.n, "test": pt.n}, "n_horses": {"valid": int(len(pv.x)), "test": int(len(pt.x))},
           "labels": {**{k: v["label"] for k, v in BLOCKS.items()}, **{k: v["label"] for k, v in GROUPS.items()},
                      **{k: v["label"] for k, v in NREF.items()}, "N": "JRDB 以外の全特徴（v6 の 120 特徴）"},
           "sources": {k: v["src"] for k, v in BLOCKS.items()},
           "fit": json.loads(FIT_JSON.read_text(encoding="utf-8"))}
    d = pd.read_parquet(FEAT_PQ, columns=["split"] + sorted({c for b in BLOCKS.values() for c in b["num"] + b["code"]}))
    res["coverage_test"] = coverage(d)
    res["results"] = evaluate(pv, pt, rows, "test")
    res["placebo"] = placebo(pv, pt)
    OUT_JSON.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"[saved] {OUT_JSON}")


def freeze(names=("N", "N+G_ALL")):
    """前向き再確認用: 指定の残差モデルを同じ設定で学習し直して保存し、保存済みの残差と一致するか確かめる。
    ゲートの重み (valid で fit) も一緒に凍結する。test の成績は計算しない。保存先は reports/ (git 管理外、JRDB 由来)。"""
    import joblib
    import lightgbm as lgb
    d = pd.read_parquet(FEAT_PQ)
    ncols = [c for c in d.columns if c.startswith("n") and c[1:].isdigit()]
    specs = model_specs(ncols)
    p = d["p_mkt"].clip(1e-4, 1 - 1e-4)
    d["init"] = np.log(p / (1 - p)); d["y"] = (d["jyun"] == 1).astype(int)
    tr = d[d.split == "train"]; va = d[d.split == "valid"]
    r = pd.read_parquet(RESID_PQ)
    rv = r[r.split == "valid"].reset_index(drop=True)
    out = {"models": {}, "feature_cols": {}, "params": PARAMS}
    for name in names:
        cols = specs[name]
        dtr = lgb.Dataset(tr[cols].values.astype(np.float32), tr["y"].values, init_score=tr["init"].values)
        dva = lgb.Dataset(va[cols].values.astype(np.float32), va["y"].values, init_score=va["init"].values, reference=dtr)
        m = lgb.train(PARAMS, dtr, num_boost_round=MAX_ROUNDS, valid_sets=[dva], callbacks=[lgb.early_stopping(EARLY, verbose=False)])
        z = m.predict(va[cols].values.astype(np.float32), num_iteration=m.best_iteration, raw_score=True)
        diff = float(np.abs(z - rv[f"z_{name}"].values).max())
        log(f"[freeze] {name}: 回数 {m.best_iteration}  保存済み残差との最大差 {diff:.2e}")
        out["models"][name] = m.model_to_string(num_iteration=m.best_iteration)
        out["feature_cols"][name] = cols
    pv = Part(rv)
    out["gate_w"] = {"base[lk,z_N]": [float(x) for x in fit_w(pv.x[["lk", "z_N"]].values, pv.codes, pv.n, pv.win)],
                     "plus[lk,z_N,z_N+G_ALL]": [float(x) for x in fit_w(pv.x[["lk", "z_N", "z_N+G_ALL"]].values, pv.codes, pv.n, pv.win)]}
    out["fcols_v6"] = json.loads((REP / "jrdb_block_audit_meta.json").read_text(encoding="utf-8"))["fcols"]
    joblib.dump(out, REP / "jrdb_block_audit_frozen.pkl")
    log(f"[freeze] gate 重み {out['gate_w']}  → {REP / 'jrdb_block_audit_frozen.pkl'}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "build":
        build()
    elif cmd == "fit":
        fit(sys.argv[2].split(",") if len(sys.argv) > 2 else None)
    elif cmd == "dev":
        dev()
    elif cmd == "eval":
        run_eval("--force" in sys.argv)
    elif cmd == "freeze":
        freeze()
    else:
        print(__doc__)
