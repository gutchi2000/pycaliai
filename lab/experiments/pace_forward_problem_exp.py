# -*- coding: utf-8 -*-
"""
pace_forward_problem_exp.py — 展開の順問題 (出走メンバー → 今日のペース → 馬ごとのペース応答) 2026-11-14
=====================================================================================================
理論
----
レースは連成系。ペースは「誰が出ているか」(テンの速い馬が何頭いるか) で決まり、各馬はペースに違う
応答をする (ハイペースなら差し馬、スロー/単騎逃げなら前の馬)。v7 (LambdaRank) は 1 頭ずつ採点し
相手を見ないので、相手関係から来るこの項は原理的に持てない。順問題を 2 段で解く:
  Step 1  出走馬の過去走 (as-of) だけから今日のレースペース (JRDB の H/M/S) を予測する。
  Step 2  馬ごとのペース応答を過去データから学び、予測ペースと掛けて 1 頭ごとの z にする。

既知の死亡ルート (再走しない): lab/experiments/exp_pace_interaction.py (2026-06)。master 列
(kako5_avg_pos, 前4角, closing_power, RPCI) の展開構成 10 特徴を LightGBM に足して再学習 → 増分なし。
今回の新しい点: JRDB のテン指数/ペース指数、明示的なペース予測と予測力の実測、馬別のペース応答、
再学習でなく共通ゲート (gate_common) での評価。

データと as-of
--------------
JRDB SED (2011-01〜、平地 td∈{1,2}、出走した馬 = ijou∉{1取消,2除外})。SED の値は全てその走の事後情報なので、
日付 D のレースの特徴には「同じ馬の D より前の走」だけを使う (馬ごとに日付順に並べ shift。同日 2 走は無い)。
今日のレースの SED からは「誰が何番で出ているか・距離・芝ダ・場・頭数」だけを使う (発走前に分かる)。
馬場状態は当日情報なので使わない。クラス名は master_lite から。
モデルと表はすべて 2013-2022 だけで fit (LightGBM は 2013-2020 で学習・2021-22 で早期終了の本数を決め、
その本数で 2013-2022 を学習し直す)。2023 = valid (ゲートが重みを fit)、2024-25 = test (1 回だけ)。

馬ごとの過去走量 (有効走 = ijou==0 かつ ten_idx・idm あり、直近 5 出走のうち有効なもの)
  ten3   = 直近 3 走の ten_idx 平均        tenmax5 = 直近 5 走の ten_idx 最大
  pf5    = 直近 5 走の「最初のコーナー通過順位の相対値」(fc-1)/(頭数-1) の平均 (0=先頭, 1=最後方)
           fc = c1,c2,c3,c4 のうち最初に 0 でないもの
  lead5  = 直近 5 走で fc==1 だった割合     pidx3 = 直近 3 走の pace_idx 平均
  y      = idm − pace(ペース補正) − ichidori(位置取り補正)  … JRDB がペース不利を補った分を外した「実際の走り」
  y5     = 直近 5 走の y 平均
  r      = (y − y5) − そのレースの完走馬の (y − y5) の平均  … 「自分の普段比」を「レースの他馬の普段比」で引いた残差
           (ペースの主効果・馬場差の誤差はレース内で消える。残るのは 脚質×ペース の相互作用)
  q      = レースペース S/M/H → −1/0/+1 (JRDB race_pace)

Step 1 ペース予測 (レース単位、LightGBM 3 クラス H/M/S)
  条件だけ (cond): 距離, 芝ダ, 場, 頭数, クラス, 月
  条件+メンバー構成 (full): 上に加えて ten3 の最大/2 番目/上位 3 平均/平均/SD/1-2 位差、tenmax5 の最大/上位 3 平均、
    pf5 の最小/2 番目/下位 3 平均/平均、pf5≤0.1・≤0.2 の頭数、lead5 の合計/最大/≥0.4・≥0.2 の頭数、
    pidx3 の最大/上位 3 平均/平均、メンバーが過去に走ったレースの q 平均・racep_idx 平均、idm 平均/最大、
    過去走なしの割合、距離短縮 (前走−今回) の平均と 200m 以上短縮の頭数
  予測力は 2023 と 2024-25 で: 多クラス logloss (学習期間の構成比を出すだけの基準との差)、AUC(H 対 他)、AUC(S 対 他)、
  q̂=P(H)−P(S) と実際の q の R²。racep_idx の回帰 R² も参考に出す。
  ★Step 1 の死亡基準 (事前登録): 2024-25 で full の AUC(H) も AUC(S) も cond より +0.01 以上高くなければ、
    「メンバー構成からペースは予測できない」= 理論は Step 1 で死亡と書く (ゲートは記録のため実行する)。

Step 2 事前登録する z (3 本、これ以外は作らない)
  z1 (集団の応答): 脚質群 sg = pf5 の四分位 (切れ目は 2013-22)。2013-22 で (芝ダ, sg, 実際のペース) ごとの r 平均を取り、
      芝ダごとに二重中心化した相互作用表 T_int[芝ダ, sg, ペース] を作る。
      z1 = Σ_p P̂_full(p) × T_int[今日の芝ダ, 馬の sg, p]      (過去走なしの馬は欠損)
  z2 (馬別の応答): 馬の過去走 (今日より前すべて) での r の q への傾き b_raw = Sxy/Sxx (馬内中心化) を、
      脚質群の傾き b_style[芝ダ, sg] (2013-22 で fit) へ縮約: b_h = b_style + Sxx/(Sxx+k) × (b_raw − b_style)。
      k = σ²/τ² は 2013-22 の馬ごとの集計からモーメント法で 1 回だけ推定 (τ²≤0 なら k=∞ = 馬別成分なし)。
      z2 = b_h × (q̂_full − 馬の過去走の q 平均)
  z3 (単騎逃げ): 馬単位の LightGBM 2 値。目的 = 「最初のコーナー先頭 (fc==1) かつ レースペースが H でない」。
      full = 自分の量 (ten3, tenmax5, pf5, lead5, pidx3, 過去走数, 馬番, 馬番/頭数, 距離増減) + 条件
             + 相手との関係 (ten3/tenmax5/lead5 の「自分 − 他馬の最大」, pf5 の「自分 − 他馬の最小」,
               他馬のうち lead5≥0.4 の頭数・pf5≤0.15 の頭数, ten3 と pf5 のレース内順位, 上位 3 平均 ten3, 他馬の lead5 合計)
      solo = 自分の量 + 条件 だけ
      z3 = P̂_full − P̂_solo   (相手関係が動かした単騎逃げ確率。馬単体の逃げ癖は v7 が既に持つので引く)
  ゲート実行: z1, z2, z3 を別々に、そして 3 本同時 (joint)。計 4 回。

判定規則 (事前登録)
  各ゲートは gate_common.gate の PASS/none/WORSE (ΔNLL の開催日 bootstrap CI95 の上端 < 0 で PASS)。
  P が PASS → SIGNAL(予測) 候補、M が PASS → SIGNAL(市場外) 候補。
  候補を確定とするのは (i) 2024 年と 2025 年の ΔNLL が両方とも負、(ii) 偽薬で同じゲートが PASS しない、の両方を満たすときだけ。
    偽薬 A: z をレース内で馬に無作為に振り直す (none のはず)
    偽薬 B: ペース予測を同じ split の別レースのものと入れ替えて z1/z2 を作り直す (メンバー構成の情報を壊す)
    偽薬 C (z3): solo の確率そのもの (相手関係なし)
  4 回 × 4 ゲート = 16 検定なので、PASS が 1 つだけで年別・偽薬を通らなければ「偶然の範囲」と書く。
  それ以外は null (再走禁止の範囲を doc に書く)。
診断 (事前登録の変種ではない・判定に使わない):
  cond 版 (ペース予測を条件だけのモデルに替えた z1/z2) と、オラクル (今日の実際のペースを入れた z1/z2 = 事後情報、
  「ペースが完全に読めたら」の上限。予想には使えない) を同じゲートで測り、どこで情報が失われるかを見る。

出力: reports/pace_forward_problem_exp.json (集計値のみ。JRDB の馬別値は出さない)、
      reports/pace_z.parquet (作業用・git 管理外)
再実行: worktree root から  python lab/experiments/pace_forward_problem_exp.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate_common as gc  # noqa: E402

WT = Path(__file__).resolve().parents[2]
OUT = WT / "reports" / "pace_forward_problem_exp.json"
ZPQ = WT / "reports" / "pace_z.parquet"
T0 = time.time()
SEED = 20261114
PACES = ["S", "M", "H"]          # クラス番号 0,1,2
QMAP = {"S": -1.0, "M": 0.0, "H": 1.0}


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


# ------------------------------------------------------------------ 1. 馬ごとの as-of 履歴
def build_runs() -> pd.DataFrame:
    cols = ["racekey", "ymd", "ban", "ped", "dist", "td", "tosu", "jyun", "ijou", "idm", "pace", "ichidori",
            "race_pace", "ten_idx", "pace_idx", "racep_idx", "c1", "c2", "c3", "c4"]
    d = gc.load_sed(cols)
    d = d[d.td.isin([1, 2]) & ~d.ijou.isin([1, 2]) & (d.ymd <= 20251231)].copy()
    for c in ("ten_idx", "pace_idx", "racep_idx"):
        d.loc[d[c] <= -90, c] = np.nan
    d.loc[d.ten_idx > 40, "ten_idx"] = np.nan
    d.loc[d.idm <= -90, "idm"] = np.nan
    d["venue"] = d.racekey.str[:2].astype(int)
    d["n_start"] = d.groupby("rid_s")["ban"].transform("size")
    c = d[["c1", "c2", "c3", "c4"]].fillna(0).values
    idx = (c > 0).argmax(1)
    fc = c[np.arange(len(c)), idx]
    fc = np.where(fc > 0, fc, np.nan)
    d["fc"] = fc
    d["pf"] = (d.fc - 1) / (d.n_start - 1).clip(lower=1)
    d["lead"] = np.where(np.isnan(fc), np.nan, (fc == 1).astype(float))
    d["q"] = d.race_pace.map(QMAP)
    d["y"] = d.idm - d.pace.fillna(0) - d.ichidori.fillna(0)
    valid = (d.ijou == 0) & d.ten_idx.notna() & d.idm.notna()
    d["valid_run"] = valid
    hist_src = {"ten": "ten_idx", "pfv": "pf", "leadv": "lead", "pidx": "pace_idx", "qv": "q", "rpv": "racep_idx",
                "yv": "y", "idmv": "idm"}
    for k, src in hist_src.items():
        d[k] = d[src].where(valid)
    d = d.sort_values(["ped", "ymd"]).reset_index(drop=True)
    g = d.groupby("ped", sort=False)
    lag = {k: np.column_stack([g[k].shift(i).values for i in range(1, 6)]) for k in hist_src}
    with np.errstate(all="ignore"):
        import warnings
        warnings.simplefilter("ignore")
        d["ten3"] = np.nanmean(lag["ten"][:, :3], axis=1)
        d["tenmax5"] = np.nanmax(lag["ten"], axis=1)
        d["pf5"] = np.nanmean(lag["pfv"], axis=1)
        d["lead5"] = np.nanmean(lag["leadv"], axis=1)
        d["pidx3"] = np.nanmean(lag["pidx"][:, :3], axis=1)
        d["q5"] = np.nanmean(lag["qv"], axis=1)
        d["rp5"] = np.nanmean(lag["rpv"], axis=1)
        d["y5"] = np.nanmean(lag["yv"], axis=1)
        d["idm5"] = np.nanmean(lag["idmv"], axis=1)
    d["nprev"] = g["valid_run"].cumsum() - d["valid_run"].astype(int)
    d["ddist"] = g["dist"].shift(1) - d["dist"]
    # 残差 r (その走の事後量。後のレースの履歴としてだけ使う)
    dev = (d.y - d.y5).where(d.valid_run)
    d["r"] = dev - dev.groupby(d.rid_s).transform("mean")
    # 馬別の傾き用の累積和 (今日の走を含まない = 今日より前の走だけ)
    m = d.r.notna() & d.q.notna()
    for name, v in (("s_n", m.astype(float)), ("s_q", d.q.where(m, 0.0)), ("s_r", d.r.where(m, 0.0)),
                    ("s_qr", (d.q * d.r).where(m, 0.0)), ("s_qq", (d.q ** 2).where(m, 0.0)),
                    ("s_rr", (d.r ** 2).where(m, 0.0))):
        cs = v.groupby(d.ped, sort=False).cumsum()
        d[name + "_incl"] = cs            # 今日の走を含む (EB の k 推定で ≤2022 の最終行に使う)
        d[name] = cs - v                  # 今日より前だけ
    d["year"] = d.ymd // 10000
    return d


# ------------------------------------------------------------------ 2. レース単位の特徴
def topk(d, col, k, largest=True):
    s = d[["rid_s", col]].dropna().sort_values(["rid_s", col], ascending=[True, not largest])
    s["k"] = s.groupby("rid_s").cumcount()
    return s[s.k < k].pivot(index="rid_s", columns="k", values=col).reindex(columns=range(k))


def build_race_frame(d: pd.DataFrame, lite: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby("rid_s")
    R = g.agg(ymd=("ymd", "first"), year=("year", "first"), dist=("dist", "first"), td=("td", "first"),
              venue=("venue", "first"), n_start=("n_start", "first"), race_pace=("race_pace", "first"),
              racep_idx=("racep_idx", "first"),
              ten3_mean=("ten3", "mean"), ten3_sd=("ten3", "std"), pf5_mean=("pf5", "mean"),
              lead5_sum=("lead5", "sum"), lead5_max=("lead5", "max"), pidx3_mean=("pidx3", "mean"),
              q5_mean=("q5", "mean"), rp5_mean=("rp5", "mean"), idm5_mean=("idm5", "mean"), idm5_max=("idm5", "max"),
              ddist_mean=("ddist", "mean"))
    R["month"] = (R.ymd // 100) % 100
    t = topk(d, "ten3", 3); R["ten3_1"] = t[0]; R["ten3_2"] = t[1]; R["ten3_top3"] = t.mean(1); R["ten3_gap"] = t[0] - t[1]
    t = topk(d, "tenmax5", 3); R["tenmax5_1"] = t[0]; R["tenmax5_top3"] = t.mean(1)
    t = topk(d, "pf5", 3, largest=False); R["pf5_1"] = t[0]; R["pf5_2"] = t[1]; R["pf5_low3"] = t.mean(1)
    t = topk(d, "pidx3", 3); R["pidx3_1"] = t[0]; R["pidx3_top3"] = t.mean(1)
    R["n_pf10"] = (d.pf5 <= 0.10).groupby(d.rid_s).sum(); R["n_pf20"] = (d.pf5 <= 0.20).groupby(d.rid_s).sum()
    R["n_lead40"] = (d.lead5 >= 0.4).groupby(d.rid_s).sum(); R["n_lead20"] = (d.lead5 >= 0.2).groupby(d.rid_s).sum()
    R["nohist_share"] = (d.nprev == 0).groupby(d.rid_s).mean()
    R["n_short200"] = (d.ddist >= 200).groupby(d.rid_s).sum()
    cls = lite.drop_duplicates("rid_s").set_index("rid_s")["クラス名"]
    R["cls"] = cls.reindex(R.index)
    return R.reset_index()


COND = ["dist", "td", "venue", "n_start", "cls_code", "month"]
FIELD = ["ten3_1", "ten3_2", "ten3_top3", "ten3_mean", "ten3_sd", "ten3_gap", "tenmax5_1", "tenmax5_top3",
         "pf5_1", "pf5_2", "pf5_low3", "pf5_mean", "n_pf10", "n_pf20", "lead5_sum", "lead5_max", "n_lead40", "n_lead20",
         "pidx3_1", "pidx3_top3", "pidx3_mean", "q5_mean", "rp5_mean", "idm5_mean", "idm5_max", "nohist_share",
         "ddist_mean", "n_short200"]


def lgb_fit(Xa, ya, Xb, yb, Xall, yall, params, cat):
    """2013-20 で学習・2021-22 で本数決定 → 2013-22 で学習し直す"""
    import lightgbm as lgb
    p = dict(learning_rate=0.05, num_threads=4, verbose=-1, seed=SEED, feature_fraction=0.8, bagging_fraction=0.8,
             bagging_freq=1, lambda_l2=1.0, **params)
    m = lgb.train(p, lgb.Dataset(Xa, ya, categorical_feature=cat), 3000,
                  valid_sets=[lgb.Dataset(Xb, yb, categorical_feature=cat)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    nb = max(20, int(m.best_iteration * 1.1))
    return lgb.train(p, lgb.Dataset(Xall, yall, categorical_feature=cat), nb), nb


def pace_models(R: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    from sklearn.metrics import log_loss, r2_score, roc_auc_score
    R = R[R.year >= 2013].copy()
    cats = sorted(R.loc[R.year <= 2022, "cls"].dropna().unique())
    R["cls_code"] = pd.Categorical(R.cls, categories=cats).codes
    lab = R.race_pace.map({p: i for i, p in enumerate(PACES)})
    R["lab"] = lab
    ok = R.lab.notna()
    a = ok & (R.year <= 2020); b = ok & R.year.between(2021, 2022); tr = ok & (R.year <= 2022)
    skill = {"n_train_races": int(tr.sum())}
    prior = R.loc[tr, "lab"].value_counts(normalize=True).sort_index().values
    skill["train_class_share_SMH"] = [float(x) for x in prior]
    for tag, feats in (("cond", COND), ("full", COND + FIELD)):
        cat = ["venue", "cls_code"]
        m, nb = lgb_fit(R.loc[a, feats], R.loc[a, "lab"], R.loc[b, feats], R.loc[b, "lab"], R.loc[tr, feats], R.loc[tr, "lab"],
                        dict(objective="multiclass", num_class=3, num_leaves=15, min_data_in_leaf=60), cat)
        P = m.predict(R[feats])
        for i, p in enumerate(PACES):
            R[f"p{p}_{tag}"] = P[:, i]
        okr = R.racep_idx.notna()
        mr, nbr = lgb_fit(R.loc[a & okr, feats], R.loc[a & okr, "racep_idx"], R.loc[b & okr, feats], R.loc[b & okr, "racep_idx"],
                          R.loc[tr & okr, feats], R.loc[tr & okr, "racep_idx"],
                          dict(objective="regression", num_leaves=15, min_data_in_leaf=60), cat)
        R[f"rp_hat_{tag}"] = mr.predict(R[feats])
        skill[tag] = {"n_rounds_cls": nb, "n_rounds_reg": nbr}
        if tag == "full":
            imp = pd.Series(m.feature_importance("gain"), index=feats)
            skill["full_gain_share_top10"] = {k: float(v) for k, v in (imp / imp.sum()).sort_values(ascending=False).head(10).items()}
            skill["full_gain_share_field_total"] = float(imp[FIELD].sum() / imp.sum())
        for per, msk in (("2021-22(早期終了に使用)", ok & R.year.between(2021, 2022)), ("valid2023", ok & (R.year == 2023)),
                         ("test2024-25", ok & R.year.between(2024, 2025))):
            x = R[msk]; Pm = x[[f"p{p}_{tag}" for p in PACES]].values; yv = x.lab.values.astype(int)
            qhat = Pm[:, 2] - Pm[:, 0]; qv = yv - 1.0
            xr = x[x.racep_idx.notna()]
            skill[tag][per] = {
                "n_races": int(len(x)),
                "logloss": float(log_loss(yv, Pm, labels=[0, 1, 2])),
                "logloss_prior": float(log_loss(yv, np.tile(prior, (len(x), 1)), labels=[0, 1, 2])),
                "auc_H": float(roc_auc_score(yv == 2, Pm[:, 2])), "auc_S": float(roc_auc_score(yv == 0, Pm[:, 0])),
                "r2_q": float(r2_score(qv, qhat)), "corr_q": float(np.corrcoef(qv, qhat)[0, 1]),
                "acc": float((Pm.argmax(1) == yv).mean()),
                "r2_racep_idx": float(r2_score(xr.racep_idx, xr[f"rp_hat_{tag}"]))}
    for per in ("valid2023", "test2024-25"):
        skill.setdefault("field_increment", {})[per] = {
            k: skill["full"][per][k] - skill["cond"][per][k] for k in ("logloss", "auc_H", "auc_S", "r2_q", "r2_racep_idx")}
    inc = skill["field_increment"]["test2024-25"]
    skill["step1_alive"] = bool(inc["auc_H"] >= 0.01 or inc["auc_S"] >= 0.01)
    return R, skill


# ------------------------------------------------------------------ 3. 応答 (集団の表・馬別の傾き)
def response_tables(d: pd.DataFrame) -> dict:
    tr = d[(d.year >= 2013) & (d.year <= 2022)]
    cuts = [float(x) for x in tr.pf5.dropna().quantile([0.25, 0.5, 0.75]).values]
    d["sg"] = np.where(d.pf5.isna(), np.nan, np.searchsorted(cuts, d.pf5.fillna(0), side="right").astype(float))
    tr = d[(d.year >= 2013) & (d.year <= 2022) & d.r.notna() & d.sg.notna() & d.race_pace.notna()]
    T = {}; tab_out = {}; bstyle = {}
    for td, x in tr.groupby("td"):
        cell = x.groupby(["sg", "race_pace"])["r"].agg(["mean", "size"]).reset_index()
        grand = x.r.mean(); ms = x.groupby("sg")["r"].mean(); mp = x.groupby("race_pace")["r"].mean()
        for _, c in cell.iterrows():
            v = c["mean"] - ms[c.sg] - mp[c.race_pace] + grand
            T[(int(td), int(c.sg), c.race_pace)] = float(v)
            tab_out[f"td{int(td)}_sg{int(c.sg)}_{c.race_pace}"] = {"T_int": float(v), "mean_r": float(c["mean"]), "n": int(c["size"])}
        for sg, xx in x.groupby("sg"):
            bstyle[(int(td), int(sg))] = float(np.cov(xx.q, xx.r)[0, 1] / xx.q.var())
    # EB の k: 2013-22 の各馬の最終行 (その走を含む累積)
    last = d[(d.year <= 2022)].groupby("ped", sort=False).tail(1)
    n = last.s_n_incl; sxx = last.s_qq_incl - last.s_q_incl ** 2 / n; sxy = last.s_qr_incl - last.s_q_incl * last.s_r_incl / n
    srr = last.s_rr_incl - last.s_r_incl ** 2 / n
    ok = (n >= 3) & (sxx > 1e-9) & last.sg.notna()
    braw = sxy[ok] / sxx[ok]
    sig2 = float(((srr[ok] - sxy[ok] ** 2 / sxx[ok]).clip(lower=0)).sum() / (n[ok] - 2).sum())
    bs = np.array([bstyle.get((int(t), int(s)), 0.0) for t, s in zip(last.td[ok], last.sg[ok])])
    w = sxx[ok].values
    tau2 = float((w * ((braw.values - bs) ** 2 - sig2 / w)).sum() / w.sum())
    k = float(sig2 / tau2) if tau2 > 0 else float("inf")
    info = {"pf5_quartile_cuts": cuts, "T_table": tab_out,
            "b_style": {f"td{a}_sg{b}": v for (a, b), v in bstyle.items()},
            "eb": {"n_horses": int(ok.sum()), "sigma2": sig2, "tau2": tau2, "k": (k if np.isfinite(k) else "inf"),
                   "median_Sxx": float(np.median(w)), "median_weight_on_own_slope": float(np.median(w / (w + k))) if np.isfinite(k) else 0.0}}
    return {"T": T, "bstyle": bstyle, "k": k, "info": info}


def make_pace_z(x: pd.DataFrame, resp: dict, ps, pm, ph, suffix: str) -> None:
    """x: 2023-25 の馬行。ps/pm/ph = そのレースの S/M/H 確率 (馬行に展開済みの配列)。z1/z2 を x に書く。"""
    T, bstyle, k = resp["T"], resp["bstyle"], resp["k"]
    td = x.td.values.astype(int); sg = x.sg.values
    z1 = np.full(len(x), np.nan); bs = np.full(len(x), np.nan)
    for (t, s) in {(a, b) for (a, b, _) in T}:
        mk = (td == t) & (sg == s)
        z1[mk] = ps[mk] * T[(t, s, "S")] + pm[mk] * T[(t, s, "M")] + ph[mk] * T[(t, s, "H")]
        bs[mk] = bstyle[(t, s)]
    n = x.s_n.values; sxx = x.s_qq.values - np.where(n > 0, x.s_q.values ** 2 / np.maximum(n, 1), 0)
    sxy = x.s_qr.values - np.where(n > 0, x.s_q.values * x.s_r.values / np.maximum(n, 1), 0)
    with np.errstate(all="ignore"):
        braw = np.where(sxx > 1e-9, sxy / np.where(sxx > 1e-9, sxx, 1), 0.0)
        wgt = np.where(sxx > 1e-9, sxx / (sxx + k), 0.0) if np.isfinite(k) else np.zeros(len(x))
        bh = bs + wgt * (braw - bs)
        qbar = np.where(n > 0, x.s_q.values / np.maximum(n, 1), np.nan)
    x["z1" + suffix] = z1
    x["z2" + suffix] = bh * ((ph - ps) - qbar)


# ------------------------------------------------------------------ 4. 単騎逃げ
def lone_lead(d: pd.DataFrame, R: pd.DataFrame) -> dict:
    from sklearn.metrics import roc_auc_score
    x = d[d.year >= 2013].merge(R[["rid_s", "cls_code"]], on="rid_s", how="left")
    x["ban_rel"] = x.ban / x.n_start
    for col, largest in (("ten3", True), ("tenmax5", True), ("lead5", True), ("pf5", False)):
        t = topk(x, col, 2, largest)
        t1 = x.rid_s.map(t[0]); t2 = x.rid_s.map(t[1])
        other = np.where(x[col].notna() & (x[col] == t1), t2, t1)      # 自分が 1 位なら 2 位、そうでなければ 1 位
        x[f"{col}_vs"] = x[col] - other
        x[f"{col}_rank"] = x.groupby("rid_s")[col].rank(ascending=not largest, method="min")
    x["n_lead40_oth"] = x.groupby("rid_s")["lead5"].transform(lambda s: (s >= 0.4).sum()) - (x.lead5 >= 0.4).astype(int)
    x["n_pf15_oth"] = x.groupby("rid_s")["pf5"].transform(lambda s: (s <= 0.15).sum()) - (x.pf5 <= 0.15).astype(int)
    x["ten3_top3"] = x.rid_s.map(R.set_index("rid_s")["ten3_top3"])
    x["lead5_oth"] = x.groupby("rid_s")["lead5"].transform("sum") - x.lead5.fillna(0)
    own = ["ten3", "tenmax5", "pf5", "lead5", "pidx3", "nprev", "ban", "ban_rel", "ddist"]
    cond = ["dist", "td", "venue", "n_start", "cls_code"]
    rel = ["ten3_vs", "tenmax5_vs", "lead5_vs", "pf5_vs", "n_lead40_oth", "n_pf15_oth", "ten3_rank", "pf5_rank", "ten3_top3", "lead5_oth"]
    x["lone"] = np.where(x.fc.notna() & x.race_pace.notna(), ((x.fc == 1) & (x.race_pace != "H")).astype(float), np.nan)
    ok = x.lone.notna()
    a = ok & (x.year <= 2020); b = ok & x.year.between(2021, 2022); tr = ok & (x.year <= 2022)
    info = {"base_rate_train": float(x.loc[tr, "lone"].mean())}
    for tag, feats in (("solo", own + cond), ("full", own + cond + rel)):
        m, nb = lgb_fit(x.loc[a, feats], x.loc[a, "lone"], x.loc[b, feats], x.loc[b, "lone"], x.loc[tr, feats], x.loc[tr, "lone"],
                        dict(objective="binary", num_leaves=31, min_data_in_leaf=200), ["venue", "cls_code"])
        x[f"pl_{tag}"] = m.predict(x[feats])
        info[tag] = {"n_rounds": nb}
        for per, msk in (("valid2023", ok & (x.year == 2023)), ("test2024-25", ok & x.year.between(2024, 2025))):
            xx = x[msk]
            # レース内で最大確率の馬が実際に単騎逃げした割合も出す
            top = xx.loc[xx.groupby("rid_s")[f"pl_{tag}"].idxmax()]
            info[tag][per] = {"auc": float(roc_auc_score(xx.lone, xx[f"pl_{tag}"])), "top_pick_hit": float(top.lone.mean()),
                              "base_rate": float(xx.lone.mean()), "n": int(len(xx))}
    out = x.loc[x.year >= 2023, ["rid_s", "ban", "pl_full", "pl_solo"]].copy()
    out["z3"] = out.pl_full - out.pl_solo
    return {"z": out, "info": info}


# ------------------------------------------------------------------ 5. ゲートの年別内訳 (gate_common と同じ手順を再現)
def gate_years(z: pd.DataFrame, zcols: list[str]) -> dict:
    b = gc.load_base().merge(z[["rid_s", "ban"] + zcols].drop_duplicates(["rid_s", "ban"]), on=["rid_s", "ban"], how="left")
    for c in zcols:
        b[c] = pd.to_numeric(b[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        b[c] = b[c].fillna(b.groupby("rid_s")[c].transform("mean")).fillna(0.0)
        b[c] = b[c] - b.groupby("rid_s")[c].transform("mean")
        sd = b.loc[b.split == "valid", c].std()
        b[c] = b[c] / (sd if sd and sd > 0 else 1.0)
    res = {}
    for gname, bcols in {"P": ["lm7"], "M": ["lk", "lm7"], "M0": ["lk"], "A10": ["lk10", "lm7"]}.items():
        dd = b[b["lk10"].notna()] if "lk10" in bcols else b
        xv = dd[dd.split == "valid"].reset_index(drop=True); xt = dd[dd.split == "test"].reset_index(drop=True)
        cv = pd.factorize(xv.rid_s)[0]; ct = pd.factorize(xt.rid_s)[0]
        wv = np.flatnonzero((xv.jyun == 1).values); wt = np.flatnonzero((xt.jyun == 1).values)
        lp = {}
        for tag, cols in (("base", bcols), ("plus", bcols + zcols)):
            w = gc._fit(xv[cols].values, cv, cv.max() + 1, wv, 1e-4)
            lp[tag] = gc.race_logsoftmax(xt[cols].values @ w, ct, ct.max() + 1)
        diff = -(lp["plus"][wt] - lp["base"][wt]); yr = xt.date.str[:4].values[wt]
        res[gname] = {"all": float(diff.mean()), "2024": float(diff[yr == "2024"].mean()), "2025": float(diff[yr == "2025"].mean())}
    return res


def short(g: dict) -> dict:
    o = {}
    for k, v in g["gates"].items():
        o[k] = {"d_nll": v["d_nll"], "ci95": v["ci95"], "verdict": v["verdict"]}
        if "hon_top3_base" in v:
            o[k]["hon_top3"] = [v["hon_top3_base"], v["hon_top3_plus"]]; o[k]["hon_win"] = [v["hon_win_base"], v["hon_win_plus"]]
    return o


# ------------------------------------------------------------------ main
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    res = {"experiment": "pace_forward_problem", "date": "2026-11-14"}
    d = build_runs()
    log(f"runs {len(d):,} (平地・出走) races {d.rid_s.nunique():,}")
    lite = gc.load_master_lite()[["レースID(新/馬番無)", "クラス名"]].rename(columns={"レースID(新/馬番無)": "rid_s"})
    lite["rid_s"] = lite.rid_s.astype(str).str[:16]
    R = build_race_frame(d, lite)
    R, skill = pace_models(R)
    res["step1_pace_forecast"] = skill
    for tag in ("cond", "full"):
        for per in ("valid2023", "test2024-25"):
            s = skill[tag][per]
            log(f"[pace {tag:4s} {per}] logloss {s['logloss']:.4f} (prior {s['logloss_prior']:.4f}) AUC_H {s['auc_H']:.3f} AUC_S {s['auc_S']:.3f}"
                f" R2q {s['r2_q']:.3f} acc {s['acc']:.3f} R2racep {s['r2_racep_idx']:.3f}")
    log(f"[pace] field increment test {skill['field_increment']['test2024-25']}  step1_alive={skill['step1_alive']}")

    resp = response_tables(d)
    res["step2_response"] = resp["info"]
    log(f"[resp] cuts {resp['info']['pf5_quartile_cuts']} EB {resp['info']['eb']}")
    log("[resp] T_int " + json.dumps({k: round(v['T_int'], 2) for k, v in resp['info']['T_table'].items()}, ensure_ascii=False))
    log("[resp] b_style " + json.dumps({k: round(v, 2) for k, v in resp['info']['b_style'].items()}))

    # 2023-25 の馬行に予測を展開
    x = d[d.year >= 2023].merge(R[["rid_s"] + [f"p{p}_{t}" for t in ("cond", "full") for p in PACES]], on="rid_s", how="left")
    make_pace_z(x, resp, x.pS_full.values, x.pM_full.values, x.pH_full.values, "")
    make_pace_z(x, resp, x.pS_cond.values, x.pM_cond.values, x.pH_cond.values, "_cond")
    # 偽薬 B: 同じ年区分 (valid / test) の別レースのペース予測を当てる
    rng = np.random.default_rng(SEED)
    Rr = R[R.year >= 2023][["rid_s", "year", "pS_full", "pM_full", "pH_full"]].copy()
    Rr["sp"] = np.where(Rr.year == 2023, 0, 1)
    sh = Rr.copy()
    for sp in (0, 1):
        ix = np.flatnonzero((Rr.sp == sp).values); pm_ = rng.permutation(ix)
        sh.iloc[ix, 2:5] = Rr.iloc[pm_, 2:5].values
    xs = x[["rid_s"]].merge(sh[["rid_s", "pS_full", "pM_full", "pH_full"]], on="rid_s", how="left")
    make_pace_z(x, resp, xs.pS_full.values, xs.pM_full.values, xs.pH_full.values, "_shuf")
    # オラクル (事後情報・診断のみ)
    oh = {p: (x.race_pace == p).astype(float).where(x.race_pace.notna()).values for p in PACES}
    make_pace_z(x, resp, oh["S"], oh["M"], oh["H"], "_oracle")
    ll = lone_lead(d, R)
    res["step2_lone_lead_model"] = ll["info"]
    log(f"[lone] {ll['info']}")
    x = x.merge(ll["z"], on=["rid_s", "ban"], how="left")
    # 偽薬 A: レース内で振り直し
    for c in ("z1", "z2", "z3"):
        x[c + "_perm"] = x.groupby("rid_s")[c].transform(lambda s: rng.permutation(s.values))
    zc = [c for c in x.columns if c.startswith(("z1", "z2", "z3"))] + ["pl_solo"]
    x[["rid_s", "ban"] + zc].to_parquet(ZPQ)
    # 応答の予測力 (診断): z と実際の残差 r の相関 (レース内)
    diag = {}
    for per, msk in (("valid2023", x.year == 2023), ("test2024-25", x.year >= 2024)):
        xx = x[msk & x.r.notna()]
        diag[per] = {c: float(xx[[c, "r"]].dropna().corr().iloc[0, 1]) for c in ("z1", "z2", "z3", "z1_cond", "z1_oracle", "z2_oracle")}
    res["response_corr_with_realised_r"] = diag
    log(f"[diag] corr(z, r) {diag}")

    # ---- ゲート (事前登録の 3 本 + joint)
    main_runs = {"z1": ["z1"], "z2": ["z2"], "z3": ["z3"], "joint": ["z1", "z2", "z3"]}
    res["gates"] = {}; res["gates_by_year"] = {}
    for name, cols in main_runs.items():
        res["gates"][name] = gc.gate(x, cols, name=name)
        res["gates_by_year"][name] = gate_years(x, cols)
        log(f"  年別 {name}: " + json.dumps({k: {a: round(b, 5) for a, b in v.items()} for k, v in res['gates_by_year'][name].items()}))
    # ---- 偽薬・診断
    side = {"placeboA_z1_perm": ["z1_perm"], "placeboA_z2_perm": ["z2_perm"], "placeboA_z3_perm": ["z3_perm"],
            "placeboB_z1_shufpace": ["z1_shuf"], "placeboB_z2_shufpace": ["z2_shuf"], "placeboC_z3_solo": ["pl_solo"],
            "diag_z1_cond": ["z1_cond"], "diag_z2_cond": ["z2_cond"],
            "oracle_z1(事後情報)": ["z1_oracle"], "oracle_z2(事後情報)": ["z2_oracle"]}
    res["side"] = {}
    for name, cols in side.items():
        res["side"][name] = short(gc.gate(x, cols, name=name))

    # ---- 判定
    verdict = {}
    plc = {"z1": ["placeboA_z1_perm", "placeboB_z1_shufpace"], "z2": ["placeboA_z2_perm", "placeboB_z2_shufpace"],
           "z3": ["placeboA_z3_perm", "placeboC_z3_solo"], "joint": []}
    for name in main_runs:
        v = {}
        for gname, label in (("P", "SIGNAL(予測)"), ("M", "SIGNAL(市場外)")):
            g = res["gates"][name]["gates"][gname]; by = res["gates_by_year"][name][gname]
            if g["verdict"] != "PASS":
                v[gname] = g["verdict"]; continue
            yr_ok = by["2024"] < 0 and by["2025"] < 0
            pl_ok = all(res["side"][p][gname]["verdict"] != "PASS" for p in plc[name])
            v[gname] = label if (yr_ok and pl_ok) else f"PASS だが未確定 (年別 {yr_ok}, 偽薬 {pl_ok})"
        verdict[name] = v
    conf = [f"{n}:{g}" for n, v in verdict.items() for g, s in v.items() if s.startswith("SIGNAL")]
    verdict["overall"] = ("SIGNAL " + ", ".join(conf)) if conf else "null"
    verdict["step1"] = "alive" if skill["step1_alive"] else "dead (メンバー構成はペース予測を条件より良くしない)"
    res["verdict"] = verdict
    log(f"[verdict] {json.dumps(verdict, ensure_ascii=False)}")
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"→ {OUT}")


if __name__ == "__main__":
    main()
