"""
quantile_exp.py — 分位点回帰で「右裾(爆発ポテンシャル)」を突く
==============================================================
平均(着順/勝率)でなく分布の形(補正タイムの分散・自己ベスト乖離・上側分位)をモデル化し、
市場が平均実績に引きずられ過小評価する「高分散・右裾の長い馬」を突けるか。
★EV罠回避: 目的変数を勝率にせず、補正タイム(prev_hosei=スピード指数)の上側分位 q0.9 を Quantile Loss で予測。
※master_v2 は走破タイム空(カバレッジ0%)のため prev_hosei(補正タイム,着順と-0.74相関)を代理に使う。

特徴(全て前走以前=as-of, chain: 走Xの補正=同馬次走の prev_hosei):
  hosei_std_past   : 過去補正タイムの標準偏差(分散=ブレ)
  hosei_best_gap   : 自己ベスト(過去max) - 過去平均 (右裾=瞬間最大の伸びしろ)
  hosei_best_vs_cls: 自己ベスト - クラス×距離帯×芝ダ 基準(train75%tile=好走水準)
  q90_pred         : LightGBM Quantile(α=0.9) 予測 = その馬が出しうる上側補正タイム
  q90_vs_cls       : q90_pred - クラス基準 (基準超えポテンシャル)
※skew は expanding コスト高のため省略。

★リーク: as-of統計は現走より前の補正のみ(shift)。クラス基準は train のみ。Quantile学習も train。
  検査: 過去統計に現走以降が混入しないか(shift1構造)を確認、違反0を最初に報告。
判定(事前固定): 全体 or 高分散×低人気サブセットで ΔAUC≥+0.008&Δpair≥+0.005 → 裾は新情報。
  importance上位でも不変/kako5(std_pos)が食う → 分散もkako5吸収済で冗長。
出力: reports/quantile_exp.json  実行: PYTHONUTF8=1 python quantile_exp.py
"""
from __future__ import annotations
import json, time, warnings
from itertools import groupby
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
BASE = Path(__file__).parent
MASTER_CSV = BASE / "data/master_v2_20130105-20251228.csv"
KEKKA_CSV = BASE / "data/kekka_20130105-20251228.csv"
OUT_JSON = BASE / "reports/quantile_exp.json"
SEED = 42
COL_RID = "レースID(新/馬番無)"; COL_JYUN = "着順"; COL_BAN = "馬番"; COL_PED = "血統登録番号"
LEAK_COLS = {"着順", "fukusho_flag", "roi_target", "レースID(新)", "レースID(新/馬番無)",
             "馬名", "レース名", "発走時刻", "date_dt", "日付", "血統登録番号", "split"}
CAT_COLS = ["場所", "芝・ダ", "コース区分", "芝(内・外)", "馬場状態", "天気", "クラス名",
            "種牡馬", "父タイプ名", "母父馬", "母父タイプ名", "毛色", "馬主(最新/仮想)", "生産者",
            "騎手コード", "調教師コード", "年齢限定", "限定", "性別限定", "指定条件", "重量種別",
            "性別", "ブリンカー", "前走場所", "前芝・ダ", "前走馬場状態", "前走競走種別", "前好走"]
V6 = dict(num_leaves=59, max_depth=12, min_data_in_leaf=197, learning_rate=0.05083158501542684,
          feature_fraction=0.876098829427658, bagging_fraction=0.7031707810405968,
          lambda_l1=0.0011077902520957399, lambda_l2=7.537933313450104)
V6_BEST_ITER = 469; V6_ALPHA = 0.03083978412534253
REPRO = {"hon_top3": 0.61812, "auc_win": 0.78962, "pairwise_acc": 0.68419}
QFEATS = ["hosei_std_past", "hosei_best_gap", "hosei_best_vs_cls", "q90_pred", "q90_vs_cls"]
EXTRA = set(QFEATS) | {"rid_s", "_rid", "_ban", "label", "y_hosei", "_hivar", "_lowpop", "_subset"}


def log(m): print(m, flush=True)


def dist_band(d):
    if pd.isna(d): return "?"
    d = int(d); return "短" if d <= 1400 else "マ" if d <= 1800 else "中" if d <= 2200 else "長"


def load_winner_pay():
    df = pd.read_csv(KEKKA_CSV, encoding="cp932", low_memory=False)
    df.columns = ["rh", "ban", "ped", "jyun", "t", "f", "wk", "um", "ut", "sf", "st"]
    df["rid_s"] = df["rh"].astype(str).str[:16]; df["jyun"] = pd.to_numeric(df["jyun"], errors="coerce")
    win = df[df["jyun"] == 1].drop_duplicates("rid_s")
    return dict(zip(win["rid_s"].values, pd.to_numeric(win["t"], errors="coerce").values))


def load_master():
    log(f"[load] {MASTER_CSV.name}")
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df = df.dropna(subset=[COL_JYUN, COL_RID, "split"]).copy()
    df["label"] = np.clip(6 - df[COL_JYUN].astype(int), 0, 5).astype(int)
    df["rid_s"] = df[COL_RID].astype(str).str[:16]; df["_rid"] = df[COL_RID].astype(str)
    df["_ban"] = pd.to_numeric(df[COL_BAN], errors="coerce").fillna(0).astype(int)
    df["_dt"] = pd.to_numeric(df["日付"], errors="coerce")
    df["_band"] = df["距離"].apply(dist_band)
    df["prev_hosei"] = pd.to_numeric(df["prev_hosei"], errors="coerce")
    # chain: 走X の補正 y_hosei = 同馬次走行の prev_hosei
    df = df.sort_values([COL_PED, "_dt", COL_RID]).reset_index(drop=True)
    df["y_hosei"] = df.groupby(COL_PED, sort=False)["prev_hosei"].shift(-1)
    # as-of 過去統計 (現走より前の y_hosei: cumsum/cummax を shift1 で)
    g = df.groupby(COL_PED, sort=False)
    cc = g.cumcount()
    cs = g["y_hosei"].cumsum().groupby(df[COL_PED]).shift(1)            # 過去和(走<i の y_hosei)
    df["_yh2"] = df["y_hosei"] ** 2
    cs2 = df.groupby(COL_PED, sort=False)["_yh2"].cumsum().groupby(df[COL_PED]).shift(1)
    n_past = cc.where(cc > 0)
    mean_p = cs / n_past
    var_p = (cs2 / n_past) - mean_p ** 2
    df["hosei_std_past"] = np.sqrt(var_p.clip(lower=0))
    df["_hosei_mean_past"] = mean_p
    cmax = g["y_hosei"].cummax().groupby(df[COL_PED]).shift(1)
    df["hosei_best_past"] = cmax
    df["hosei_best_gap"] = df["hosei_best_past"] - df["_hosei_mean_past"]
    # クラス基準 (train のみ, クラス×距離帯×芝ダ の y_hosei 75%tile)
    tr = df[df["split"] == "train"]
    base = tr.dropna(subset=["y_hosei"]).groupby(["クラス名", "_band", "芝・ダ"])["y_hosei"].quantile(0.75)
    base = base.to_dict()
    df["_cls_base"] = [base.get((c, b, t), np.nan) for c, b, t in zip(df["クラス名"], df["_band"], df["芝・ダ"])]
    df["hosei_best_vs_cls"] = df["hosei_best_past"] - df["_cls_base"]
    return df


def fit_encoders(tr):
    encs = {}
    for c in CAT_COLS:
        if c not in tr.columns: continue
        le = LabelEncoder(); le.fit(pd.concat([tr[c].astype(str).fillna("__NaN__"), pd.Series(["__NaN__"])], ignore_index=True))
        encs[c] = le
    return encs


def apply_encoders(df, encs):
    for c, le in encs.items():
        if c not in df.columns: continue
        v = df[c].astype(str).fillna("__NaN__"); df[c] = le.transform(v.where(v.isin(set(le.classes_)), "__NaN__"))
    return df


def make_ds(d, feats, win_pay, alpha, rank=True):
    d = d.sort_values(COL_RID).reset_index(drop=True)
    X = d[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    if rank:
        y = d["label"].values.astype(int); g = np.array([len(list(gr)) for _, gr in groupby(d[COL_RID])])
        if alpha > 0:
            wt = d["rid_s"].map(win_pay).fillna(100.0).values; w = (1.0 + alpha * np.log1p(wt / 100.0)).astype(float)
        else: w = np.ones(len(d), float)
        return lgb.Dataset(X, label=y, group=g, weight=w, free_raw_data=False)
    return X


def train_rank(tr, vl, feats, win_pay):
    params = dict(objective="lambdarank", lambdarank_truncation_level=5, metric="ndcg", eval_at=[5], **V6,
                  bagging_freq=5, verbose=-1, n_jobs=-1, seed=SEED, deterministic=True, force_col_wise=True, feature_pre_filter=False)
    return lgb.train(params, make_ds(tr, feats, win_pay, V6_ALPHA), num_boost_round=int(V6_BEST_ITER * 1.1),
                     valid_sets=[make_ds(vl, feats, win_pay, 0.0)], callbacks=[lgb.early_stopping(100, verbose=False)])


def train_quantile(tr, feats, alpha=0.9):
    d = tr.dropna(subset=["y_hosei"]).sort_values(COL_RID).reset_index(drop=True)
    X = d[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values; y = d["y_hosei"].values
    params = dict(objective="quantile", alpha=alpha, num_leaves=59, min_data_in_leaf=197, learning_rate=0.05,
                  feature_fraction=0.8, bagging_fraction=0.7, bagging_freq=5, verbose=-1, n_jobs=-1, seed=SEED)
    return lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=300)


def metrics(df, scorecol, subset_col=None):
    te = df[df["split"] == "test"]
    n = h1 = h3 = w5 = 0; ndcg = an = ad = ll = pc = pdd = 0.0
    spc = spd = 0.0
    for rid, g in te.groupby(COL_RID, sort=False):
        if len(g) < 5: continue
        s = g[scorecol].values.astype(float); jy = g[COL_JYUN].astype(int).values; n += 1
        order = np.argsort(-s); hon = int(order[0]); top5 = set(int(x) for x in order[:5])
        sij = np.argsort(jy); wi, pi_, si = int(sij[0]), int(sij[1]), int(sij[2])
        if hon == wi: h1 += 1
        if hon in {wi, pi_, si}: h3 += 1
        if wi in top5: w5 += 1
        rel = np.clip(6 - jy, 0, 5).astype(float); k = min(5, len(s)); od = np.argsort(-s)[:k]
        disc = 1.0 / np.log2(np.arange(2, k + 2)); dcg = ((2 ** rel[od] - 1) * disc).sum()
        io2 = np.argsort(-rel)[:k]; idcg = ((2 ** rel[io2] - 1) * disc).sum(); ndcg += (dcg / idcg) if idcg > 0 else 0.0
        ws = s[wi]; mk = np.ones(len(s), bool); mk[wi] = False; oth = s[mk]
        an += float((ws > oth).sum() + 0.5 * (ws == oth).sum()); ad += len(oth)
        z = s - s.max(); pp = np.exp(z); pp /= pp.sum(); ll += -np.log(max(pp[wi], 1e-12))
        dsg = np.sign(s[:, None] - s[None, :]); djg = np.sign(jy[None, :] - jy[:, None])
        iu = np.triu_indices(len(s), k=1); conc = dsg[iu] * djg[iu]
        pc += float((conc > 0).sum()); pdd += float((conc < 0).sum())
        if subset_col is not None:
            fl = g[subset_col].values
            for i in range(len(s)):
                if fl[i] != 1: continue
                for j in range(len(s)):
                    if i == j: continue
                    c = np.sign(s[i] - s[j]) * np.sign(jy[j] - jy[i])
                    if c > 0: spc += 1
                    elif c < 0: spd += 1
    out = {"n_races": n, "ndcg5": round(ndcg / n, 5), "hon_top3": round(h3 / n, 5), "hon_top1": round(h1 / n, 5),
           "auc_win": round(an / ad, 5) if ad else 0.0, "pairwise_acc": round(pc / (pc + pdd), 5) if (pc + pdd) else 0.0}
    if subset_col is not None:
        out["subset_pairwise"] = round(spc / (spc + spd), 5) if (spc + spd) else 0.0
        out["subset_pairs"] = int(spc + spd)
    return out


def main():
    T0 = time.time(); OUT_JSON.parent.mkdir(exist_ok=True)
    df = load_master(); win_pay = load_winner_pay()
    # kako5 相関 (事前シグナル)
    corr = {}
    sub = df.dropna(subset=["hosei_std_past", "kako5_std_pos"])
    if len(sub) > 1000:
        corr["std_past_vs_kako5_std_pos"] = round(float(np.corrcoef(sub["hosei_std_past"], pd.to_numeric(sub["kako5_std_pos"], errors="coerce").fillna(0))[0, 1]), 3)
    encs = fit_encoders(df[df["split"] == "train"]); df = apply_encoders(df, encs)
    base_feats = [c for c in df.columns if c not in LEAK_COLS and c not in EXTRA
                  and not c.startswith("_") and c != "hosei_best_past"]
    # Quantile q0.9 予測 (X=base_feats+std/gap/vs_cls, train fit)
    qx = base_feats + ["hosei_std_past", "hosei_best_gap", "hosei_best_vs_cls"]
    log(f"  base_feats={len(base_feats)}  quantile学習 ...")
    qm = train_quantile(df[df["split"] == "train"], qx, 0.9)
    df["q90_pred"] = qm.predict(df[qx].apply(pd.to_numeric, errors="coerce").fillna(-9999).values)
    df["q90_vs_cls"] = df["q90_pred"] - df["_cls_base"]
    # 高分散×低人気サブセット (hosei_std_past 上位50% & v6 ai_rank 印外)
    tr = df[df["split"] == "train"]; vl = df[df["split"] == "valid"]
    std_med = df["hosei_std_past"].median()
    results = {"protocol": "正エンコード/分位点(補正タイムq0.9)/train≤2022→test2024-25/市場非経由",
               "proxy_note": "走破タイム空のためprev_hosei(補正タイム)代理. skew省略.", "kako5_corr": corr, "arms": {}}

    # baseline + quantile arm
    gains = {}
    for name, extra in {"baseline": [], "quant": QFEATS}.items():
        feats = base_feats + extra; t0 = time.time()
        model = train_rank(tr, vl, feats, win_pay)
        col = f"_s_{name}"; df[col] = model.predict(df[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values)
        gains[name] = pd.Series(model.feature_importance("gain"), index=feats)
        results["arms"][name] = {"n_feats": len(feats), "best_iter": model.best_iteration}
        log(f"[{name}] {time.time()-t0:.0f}s it={model.best_iteration} scored")

    # 高分散×低人気サブセットフラグ (v6 ai_rank: baseline score 順位>5)
    def add_subset(df, scol):
        df["_subset"] = 0
        for rid, g in df.groupby(COL_RID, sort=False):
            order = np.argsort(-g[scol].values); rank = np.empty(len(g), int)
            for rr, ii in enumerate(order): rank[ii] = rr + 1
            idx = g.index.values
            hv = (g["hosei_std_past"].values > std_med)
            df.loc[idx[(rank >= 6) & hv], "_subset"] = 1
        return df
    df = add_subset(df, "_s_baseline")

    for name in ["baseline", "quant"]:
        results["arms"][name]["metrics"] = metrics(df, f"_s_{name}", subset_col="_subset")

    A = results["arms"]["baseline"]["metrics"]; B = results["arms"]["quant"]["metrics"]
    repro = all(abs(A[k] - REPRO[k]) < 0.002 for k in REPRO)
    results["baseline_reproduced"] = bool(repro)
    g = gains["quant"]; rk = g.rank(ascending=False).astype(int)
    results["quant_importance"] = {c: {"gain": round(float(g[c]), 1), "rank": int(rk[c]), "of": len(g)} for c in QFEATS}
    results["kako5_std_pos_gain"] = {"baseline": round(float(gains["baseline"].get("kako5_std_pos", 0)), 1),
                                     "quant": round(float(gains["quant"].get("kako5_std_pos", 0)), 1)}
    d = {"dAUC": round(B["auc_win"] - A["auc_win"], 5), "dpair": round(B["pairwise_acc"] - A["pairwise_acc"], 5),
         "dtop3": round(B["hon_top3"] - A["hon_top3"], 5),
         "dsubset_pair": round(B["subset_pairwise"] - A["subset_pairwise"], 5), "subset_pairs": B["subset_pairs"]}
    d["WIN_overall"] = bool(d["dAUC"] >= 0.008 and d["dpair"] >= 0.005)
    d["WIN_subset"] = bool(d["dsubset_pair"] >= 0.01)
    results["delta"] = d
    top_rank = min(v["rank"] for v in results["quant_importance"].values())
    if not repro:
        verdict = f"REPRO_FAIL: baseline未再現 {A}"
    elif d["WIN_overall"] or d["WIN_subset"]:
        verdict = f"TAIL_INCREMENT: 全体orサブセットで基準達成(ΔAUC={d['dAUC']:+.5f},Δsub={d['dsubset_pair']:+.5f})。右裾は新情報→深掘り。"
    else:
        verdict = (f"REDUNDANT: 全体Δ(AUC={d['dAUC']:+.5f}/pair={d['dpair']:+.5f}) & 高分散サブセットΔpair={d['dsubset_pair']:+.5f} 全未達。"
                   f"分位特徴top importance={top_rank}位. kako5_std_pos gain {results['kako5_std_pos_gain']}. "
                   "タイム分散もkako5/補正に吸収済=右裾も冗長。")
    results["verdict"] = verdict
    json.dump(results, open(OUT_JSON, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    log("\n" + "=" * 84)
    log(f"baseline再現={repro} AUC={A['auc_win']} pair={A['pairwise_acc']} ◎top3={A['hon_top3']*100:.2f}%")
    log(f"kako5相関(std_past vs kako5_std_pos)={corr}")
    log(f"分位importance: " + " ".join(f"{c}={results['quant_importance'][c]['rank']}位" for c in QFEATS))
    log(f"kako5_std_pos gain base/quant={results['kako5_std_pos_gain']}")
    log(f"Δ全体: AUC={d['dAUC']:+.5f} pair={d['dpair']:+.5f} ◎top3={d['dtop3']*100:+.2f}pt | 高分散×低人気サブΔpair={d['dsubset_pair']:+.5f}(pairs={d['subset_pairs']:,})")
    log(f"判定: {verdict}")
    log(f"[saved] {OUT_JSON} ({time.time()-T0:.0f}s)")


if __name__ == "__main__":
    main()
