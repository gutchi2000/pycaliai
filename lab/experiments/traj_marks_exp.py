"""
traj_marks_exp.py — (a) オッズ軌跡特徴の予測精度 A/B (正エンコード・市場非経由)
================================================================================
baseline=正v6(120, ◎top3=61.81%/AUC=0.78962/pair=0.68419 再現確認) vs +オッズ軌跡特徴(9時以前)。
多指標 AUC/log-loss/pairwise/◎top3/NDCG + importance + カバレッジ。
軌跡特徴は build_odds_traj_feats.py 生成(リーク検査済=9時以降不使用)を流用。
判定(事前固定): ΔAUC≥+0.008 & Δpairwise≥+0.005 → 軌跡は新情報。未達→点オッズ/kako5に吸収済。
出力: reports/traj_marks_exp.json  実行: PYTHONUTF8=1 python traj_marks_exp.py
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
TRAJ_PQ = BASE / "data/odds_traj_feats.parquet"
OUT_JSON = BASE / "reports/traj_marks_exp.json"
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
TRAJ_COLS = ["traj_nsnap", "traj_has", "traj_logchg", "traj_range", "traj_absmax", "traj_revers", "traj_rankshift"]
EXTRA = set(TRAJ_COLS) | {"rid_s", "_rid", "_ban", "label"}


def log(m): print(m, flush=True)


def load_winner_pay():
    df = pd.read_csv(KEKKA_CSV, encoding="cp932", low_memory=False)
    df.columns = ["rid_horse", "ban", "ped", "jyun", "t", "f", "wk", "um", "ut", "sf", "st"]
    df["rid_s"] = df["rid_horse"].astype(str).str[:16]; df["jyun"] = pd.to_numeric(df["jyun"], errors="coerce")
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
    tj = pd.read_parquet(TRAJ_PQ); tj["rid16"] = tj["rid16"].astype(str)
    tj["ban"] = pd.to_numeric(tj["ban"], errors="coerce").fillna(0).astype(int)
    df = df.merge(tj, left_on=["_rid", "_ban"], right_on=["rid16", "ban"], how="left")
    df = df.drop(columns=[c for c in ["rid16", "ban"] if c in df.columns], errors="ignore")
    cov = round(df.loc[df.split == "test", "traj_has"].fillna(0).astype(bool).mean() * 100, 1)
    cov_any = round(df.loc[df.split == "test", "traj_nsnap"].notna().mean() * 100, 1)
    log(f"  test 軌跡(k>=2)カバレッジ={cov}%  snapあり={cov_any}%")
    return df, cov, cov_any


def fit_encoders(tr):
    encs = {}
    for c in CAT_COLS:
        if c not in tr.columns: continue
        le = LabelEncoder()
        le.fit(pd.concat([tr[c].astype(str).fillna("__NaN__"), pd.Series(["__NaN__"])], ignore_index=True))
        encs[c] = le
    return encs


def apply_encoders(df, encs):
    for c, le in encs.items():
        if c not in df.columns: continue
        v = df[c].astype(str).fillna("__NaN__")
        df[c] = le.transform(v.where(v.isin(set(le.classes_)), "__NaN__"))
    return df


def make_ds(d, feats, win_pay, alpha):
    d = d.sort_values(COL_RID).reset_index(drop=True)
    X = d[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    y = d["label"].values.astype(int)
    g = np.array([len(list(gr)) for _, gr in groupby(d[COL_RID])])
    if alpha > 0:
        wt = d["rid_s"].map(win_pay).fillna(100.0).values; w = (1.0 + alpha * np.log1p(wt / 100.0)).astype(float)
    else:
        w = np.ones(len(d), float)
    return lgb.Dataset(X, label=y, group=g, weight=w, free_raw_data=False)


def train(tr, vl, feats, win_pay):
    params = dict(objective="lambdarank", lambdarank_truncation_level=5, metric="ndcg", eval_at=[5],
                  **V6, bagging_freq=5, verbose=-1, n_jobs=-1, seed=SEED, deterministic=True,
                  force_col_wise=True, feature_pre_filter=False)
    return lgb.train(params, make_ds(tr, feats, win_pay, V6_ALPHA), num_boost_round=int(V6_BEST_ITER * 1.1),
                     valid_sets=[make_ds(vl, feats, win_pay, 0.0)], callbacks=[lgb.early_stopping(100, verbose=False)])


def metrics(df, scorecol):
    te = df[df["split"] == "test"]
    n = h1 = h3 = w5 = 0; ndcg = 0.0; an = ad = 0.0; ll = 0.0; pc = pdd = 0.0
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
            "winner_in_top5": round(w5 / n, 5), "auc_win": round(an / ad, 5) if ad else 0.0,
            "logloss_win": round(ll / n, 5), "pairwise_acc": round(pc / (pc + pdd), 5) if (pc + pdd) else 0.0}


def main():
    T0 = time.time(); OUT_JSON.parent.mkdir(exist_ok=True)
    df, cov, cov_any = load_master(); win_pay = load_winner_pay()
    encs = fit_encoders(df[df["split"] == "train"]); df = apply_encoders(df, encs)
    base_feats = [c for c in df.columns if c not in LEAK_COLS and c not in EXTRA]
    log(f"  base_feats={len(base_feats)}")
    tr = df[df["split"] == "train"]; vl = df[df["split"] == "valid"]
    results = {"protocol": "正エンコード/軌跡特徴(9時以前)/train≤2022→valid2023→test2024-25/市場非経由",
               "traj_coverage_test_k2": cov, "traj_coverage_any": cov_any,
               "verdict_rule": "ΔAUC≥+0.008 & Δpairwise≥+0.005 → 軌跡は新情報", "arms": {}}
    gains = {}
    for name, extra in {"baseline": [], "traj": TRAJ_COLS}.items():
        feats = base_feats + extra; t0 = time.time()
        model = train(tr, vl, feats, win_pay)
        col = f"_s_{name}"; df[col] = model.predict(df[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values)
        m = metrics(df, col); gains[name] = pd.Series(model.feature_importance("gain"), index=feats)
        results["arms"][name] = {"n_feats": len(feats), "best_iter": model.best_iteration, "metrics": m}
        log(f"[{name:8s}] {time.time()-t0:.0f}s it={model.best_iteration} AUC={m['auc_win']} "
            f"pair={m['pairwise_acc']} ◎top3={m['hon_top3']*100:.2f}% logloss={m['logloss_win']}")
    A = results["arms"]["baseline"]["metrics"]; B = results["arms"]["traj"]["metrics"]
    repro_ok = all(abs(A[k] - REPRO[k]) < 0.002 for k in REPRO)
    results["baseline_reproduced"] = bool(repro_ok)
    g = gains["traj"]; rk = g.rank(ascending=False).astype(int)
    results["traj_importance"] = {c: {"gain": round(float(g[c]), 1), "rank": int(rk[c]), "of": len(g)}
                                  for c in TRAJ_COLS}
    d = {"dAUC": round(B["auc_win"] - A["auc_win"], 5), "dpairwise": round(B["pairwise_acc"] - A["pairwise_acc"], 5),
         "dlogloss": round(B["logloss_win"] - A["logloss_win"], 5), "dtop3": round(B["hon_top3"] - A["hon_top3"], 5)}
    d["WIN"] = bool(d["dAUC"] >= 0.008 and d["dpairwise"] >= 0.005)
    results["delta_vs_baseline"] = d
    if not repro_ok:
        verdict = f"REPRO_FAIL: baseline未再現 ({A})"
    elif d["WIN"]:
        verdict = f"INCREMENT: 軌跡で ΔAUC={d['dAUC']:+.5f}/Δpair={d['dpairwise']:+.5f} 事前基準達成。過程の形は新情報→(b)市場増分へ。"
    else:
        verdict = (f"REDUNDANT_(a): 軌跡で ΔAUC={d['dAUC']:+.5f} Δpair={d['dpairwise']:+.5f} (未達)。"
                   "予測精度では点オッズ/kako5に吸収済。※(b)市場ΔR²は別途。")
    results["verdict"] = verdict
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log(f"\nbaseline再現={repro_ok}  AUC={A['auc_win']} pair={A['pairwise_acc']} ◎top3={A['hon_top3']*100:.2f}%")
    log(f"traj importance: " + " ".join(f"{c}={results['traj_importance'][c]['rank']}位" for c in TRAJ_COLS))
    log(f"Δ: AUC={d['dAUC']:+.5f} pair={d['dpairwise']:+.5f} logloss={d['dlogloss']:+.5f} top3={d['dtop3']*100:+.2f}pt WIN={d['WIN']}")
    log(f"判定: {verdict}")
    log(f"[saved] {OUT_JSON} ({time.time()-T0:.0f}s)")


if __name__ == "__main__":
    main()
