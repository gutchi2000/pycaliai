"""
exp_window.py — 学習期間(時間窓)の最適化: concept drift 除去 vs データ量減少の綱引き
====================================================================================
評価は全arm共通 test=2024-25 / valid=2023 early-stop に固定。学習窓(train≤2022)だけ動かす。
特徴・分割境界・v6 best params は不変、正エンコード。リークなし(窓変更のみ)。

arm:
  W_all   : train 2013-2022 (現行v6 baseline, 61.81%/AUC0.78962/pair0.68419 再現確認)
  W_8y    : train 2015-2022
  W_6y    : train 2017-2022
  W_4y    : train 2019-2022
  D_mild  : train全期間 + 時間減衰 weight (半減期4年)
  D_strong: train全期間 + 時間減衰 weight (半減期2年)
(W_10y=2013-2022 は train が10年ぴったりで W_all と同一のため省略)
減衰は v6 の winner_tansho weight に 0.5^((2022-year)/halflife) を乗算。

判定(事前固定, AUC/pairwise主体・◎top3は振れるので参考):
  最良窓がW_allをΔAUC≥+0.005で安定上回り単調改善 → drift支配(期間短縮に伸びしろ)。
  窓縮小で悪化(W_all最良) → データ量支配(期間短縮は逆効果)。途中ピーク→最適点。D_*>W_*なら減衰が正解。
出力: reports/exp_window.json  実行: PYTHONUTF8=1 python exp_window.py
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
OUT_JSON = BASE / "reports/exp_window.json"
SEED = 42
COL_RID = "レースID(新/馬番無)"; COL_JYUN = "着順"; COL_BAN = "馬番"
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
# arm: (year_from, halflife)  year_from=学習窓開始年, halflife=None なら減衰なし
ARMS = {"W_all": (2013, None), "W_8y": (2015, None), "W_6y": (2017, None), "W_4y": (2019, None),
        "D_mild": (2013, 4.0), "D_strong": (2013, 2.0)}


def log(m): print(m, flush=True)


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
    df["rid_s"] = df[COL_RID].astype(str).str[:16]
    df["_year"] = df["rid_s"].str[:4].astype(int)
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


def make_ds(d, feats, win_pay, alpha, halflife):
    d = d.sort_values(COL_RID).reset_index(drop=True)
    X = d[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    y = d["label"].values.astype(int); g = np.array([len(list(gr)) for _, gr in groupby(d[COL_RID])])
    if alpha > 0:
        wt = d["rid_s"].map(win_pay).fillna(100.0).values; w = (1.0 + alpha * np.log1p(wt / 100.0)).astype(float)
    else:
        w = np.ones(len(d), float)
    if halflife is not None:
        w = w * (0.5 ** ((2022 - d["_year"].values) / halflife))    # 直近ほど重い
    return lgb.Dataset(X, label=y, group=g, weight=w, free_raw_data=False)


def train(tr, vl, feats, win_pay, halflife):
    params = dict(objective="lambdarank", lambdarank_truncation_level=5, metric="ndcg", eval_at=[5], **V6,
                  bagging_freq=5, verbose=-1, n_jobs=-1, seed=SEED, deterministic=True, force_col_wise=True, feature_pre_filter=False)
    return lgb.train(params, make_ds(tr, feats, win_pay, V6_ALPHA, halflife), num_boost_round=int(V6_BEST_ITER * 1.1),
                     valid_sets=[make_ds(vl, feats, win_pay, 0.0, None)], callbacks=[lgb.early_stopping(100, verbose=False)])


def metrics(df, scorecol):
    te = df[df["split"] == "test"]
    n = h1 = h3 = w5 = 0; ndcg = an = ad = ll = pc = pdd = 0.0
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
    return {"n_races": n, "ndcg5": round(ndcg / n, 5), "hon_top3": round(h3 / n, 5), "hon_top1": round(h1 / n, 5),
            "auc_win": round(an / ad, 5) if ad else 0.0, "logloss_win": round(ll / n, 5),
            "pairwise_acc": round(pc / (pc + pdd), 5) if (pc + pdd) else 0.0}


def main():
    T0 = time.time(); OUT_JSON.parent.mkdir(exist_ok=True)
    df = load_master(); win_pay = load_winner_pay()
    encs = fit_encoders(df[df["split"] == "train"]); df = apply_encoders(df, encs)
    base_feats = [c for c in df.columns if c not in LEAK_COLS and c not in {"label", "rid_s", "_year"}]
    log(f"  base_feats={len(base_feats)}")
    tr_all = df[df["split"] == "train"]; vl = df[df["split"] == "valid"]
    results = {"protocol": "test2024-25/valid2023固定, 学習窓のみ変更, 正エンコード, v6params固定",
               "arms": {}}
    for name, (yfrom, hl) in ARMS.items():
        tr = tr_all[tr_all["_year"] >= yfrom]
        t0 = time.time()
        model = train(tr, vl, base_feats, win_pay, hl)
        col = f"_s_{name}"; df[col] = model.predict(df[base_feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values)
        m = metrics(df, col)
        results["arms"][name] = {"year_from": yfrom, "halflife": hl, "n_train": int(len(tr)),
                                 "best_iter": model.best_iteration, "metrics": m}
        log(f"[{name:9s}] {time.time()-t0:.0f}s train={len(tr):,} it={model.best_iteration} "
            f"AUC={m['auc_win']} pair={m['pairwise_acc']} ◎top3={m['hon_top3']*100:.2f}% logloss={m['logloss_win']}")

    A = results["arms"]["W_all"]["metrics"]
    repro = all(abs(A[k] - REPRO[k]) < 0.002 for k in REPRO)
    results["baseline_reproduced"] = bool(repro)
    deltas = {}
    for name in ARMS:
        if name == "W_all": continue
        m = results["arms"][name]["metrics"]
        deltas[name] = {"dAUC": round(m["auc_win"] - A["auc_win"], 5), "dpair": round(m["pairwise_acc"] - A["pairwise_acc"], 5),
                        "dlogloss": round(m["logloss_win"] - A["logloss_win"], 5), "dtop3": round(m["hon_top3"] - A["hon_top3"], 5)}
    results["deltas_vs_W_all"] = deltas
    # 判定 (AUC/pair 主体)
    best = max(results["arms"].items(), key=lambda kv: kv[1]["metrics"]["auc_win"])
    best_name = best[0]; best_dAUC = round(best[1]["metrics"]["auc_win"] - A["auc_win"], 5)
    win_arms = ["W_8y", "W_6y", "W_4y"]
    win_aucs = [results["arms"][w]["metrics"]["auc_win"] for w in win_arms]
    monotone = all(win_aucs[i] >= win_aucs[i + 1] - 0.001 for i in range(len(win_aucs) - 1))  # 縮むほど改善?
    if best_name != "W_all" and best_dAUC >= 0.005:
        verdict = f"DRIFT_DOMINANT: 最良={best_name} が W_all を ΔAUC={best_dAUC:+.5f} 上回る。concept drift支配=学習期間短縮に伸びしろ。"
    elif best_name == "W_all":
        verdict = "DATA_DOMINANT: W_all が最良(全arm未改善)。古いデータも効いている=期間短縮は逆効果で確定。"
    else:
        verdict = (f"PEAK_or_NOISE: 最良={best_name} だが ΔAUC={best_dAUC:+.5f}(<+0.005)。"
                   "改善は閾値未満=実質W_allと同等(データ量とdriftが拮抗、明確な伸びしろなし)。")
    results["verdict"] = verdict; results["best_arm"] = best_name; results["best_dAUC_vs_W_all"] = best_dAUC
    json.dump(results, open(OUT_JSON, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    log("\n" + "=" * 84)
    log(f"baseline(W_all)再現={repro}  AUC={A['auc_win']} pair={A['pairwise_acc']} ◎top3={A['hon_top3']*100:.2f}%")
    log(f"{'arm':9s}{'n_train':>9s}{'AUC':>9s}{'pair':>9s}{'◎top3':>8s}{'logloss':>9s}{'ΔAUC':>9s}{'Δpair':>9s}")
    for name in ARMS:
        m = results["arms"][name]["metrics"]; d = deltas.get(name, {"dAUC": 0, "dpair": 0})
        log(f"{name:9s}{results['arms'][name]['n_train']:>9,}{m['auc_win']:>9.5f}{m['pairwise_acc']:>9.5f}"
            f"{m['hon_top3']*100:>7.2f}%{m['logloss_win']:>9.4f}{d['dAUC']:>+9.5f}{d['dpair']:>+9.5f}")
    log(f"\n判定: {verdict}")
    log(f"[saved] {OUT_JSON} ({time.time()-T0:.0f}s)")


if __name__ == "__main__":
    main()
