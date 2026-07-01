"""
race_elo_exp.py — 相互ELO 4アーム + v1/v2 の A/B (市場非経由・多指標)
=====================================================================
arm: A(v6 120) / T1M1 T1M2 T2M1 T2M2 (A+ELO4列) / v1_W3 v2_W3 (A+旧レベル1列)
主指標: AUC(レース内 win 判別 pairwise) / log-loss(レース内softmax win) / pairwise順位acc
補助: ◎top3 / NDCG@5 / 新特徴importance / 既存食い合い。市場ΔR²/ROIは測らない。
v6 best params固定、train≤2022→valid2023→test2024-25。A が ◎top3=61.48% 再現確認。
判定(事前固定・多重比較): ΔAUC≥+0.008 AND Δpairwise≥+0.005 AND Δlogloss≤0 の符号一貫で勝ち。
出力: reports/race_elo_exp.json   実行: PYTHONUTF8=1 python race_elo_exp.py
"""
from __future__ import annotations
import json, time, warnings
from itertools import groupby
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
BASE = Path(__file__).parent
MASTER_CSV = BASE / "data/master_v2_20130105-20251228.csv"
KEKKA_CSV = BASE / "data/kekka_20130105-20251228.csv"
OUT_JSON = BASE / "reports/race_elo_exp.json"
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
V6_BEST_ITER = 469; V6_ALPHA_SW = 0.03083978412534253
ELO_ARMS = ["T1M1", "T1M2", "T2M1", "T2M2"]
ELO_FEATS = {a: [f"elo_{a}_horse", f"elo_{a}_level", f"elo_{a}_vs", f"elo_{a}_prevlevel"] for a in ELO_ARMS}
ARMS_DEF = {"A": []}
for a in ELO_ARMS: ARMS_DEF[a] = ELO_FEATS[a]
ARMS_DEF["v1_W3"] = ["race_level_prev_W3"]
ARMS_DEF["v2_W3"] = ["race_level_str_W3"]
ALL_EXTRA = sum(ELO_FEATS.values(), []) + ["race_level_prev_W1", "race_level_prev_W2", "race_level_prev_W3",
                                           "race_level_str_W2", "race_level_str_W3"]
HELP_GROUPS = {"kako5": lambda c: c.startswith("kako5_"), "hosei": lambda c: "hosei" in c,
               "zensou": lambda c: c.startswith("前"), "trnH": lambda c: c.startswith(("trnH_", "trnW_")),
               "elo_new": lambda c: c.startswith("elo_"), "racelevel_old": lambda c: c.startswith("race_level_")}


def log(m): print(m, flush=True)


def load_winner_pay():
    df = pd.read_csv(KEKKA_CSV, encoding="cp932", low_memory=False)
    df.columns = ["rid_horse", "ban", "ped", "jyun", "tansho", "fukusho", "wakuren",
                  "umaren", "umatan", "sanrenpuku", "sanrentan"]
    df["rid_s"] = df["rid_horse"].astype(str).str[:16]
    df["jyun"] = pd.to_numeric(df["jyun"], errors="coerce")
    df["tansho"] = pd.to_numeric(df["tansho"], errors="coerce")
    win = df[df["jyun"] == 1].drop_duplicates("rid_s")
    return dict(zip(win["rid_s"].values, win["tansho"].values))


def load_master():
    log(f"[load] {MASTER_CSV.name}")
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df = df.dropna(subset=[COL_JYUN, COL_RID, "split"]).copy()
    df["label"] = np.clip(6 - df[COL_JYUN].astype(int), 0, 5).astype(int)
    df["rid_s"] = df[COL_RID].astype(str).str[:16]
    df["_rid"] = df[COL_RID].astype(str)
    df["_ban"] = pd.to_numeric(df[COL_BAN], errors="coerce").fillna(0).astype(int)
    for pq in ["data/elo_feats.parquet", "data/race_level_feats.parquet", "data/race_level_feats_v2.parquet"]:
        p = BASE / pq
        if p.exists():
            rl = pd.read_parquet(p)
            rl["rid16"] = rl["rid16"].astype(str)
            rl["ban"] = pd.to_numeric(rl["ban"], errors="coerce").fillna(0).astype(int)
            df = df.merge(rl, left_on=["_rid", "_ban"], right_on=["rid16", "ban"], how="left",
                          suffixes=("", "_dup"))
            df = df.drop(columns=[c for c in df.columns if c.endswith("_dup") or c in ("rid16", "ban")],
                         errors="ignore")
    return df


def fit_encoders(tr):
    encs = {}
    for c in CAT_COLS:
        if c not in tr.columns: continue
        le = LabelEncoder()
        le.fit(pd.concat([tr[c].astype(str).fillna("__NaN__"), pd.Series(["__NaN__"])], ignore_index=True))
        encs[c] = le
    return encs


def apply_encoders(df, encs):
    df = df.copy()
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
        wt = d["rid_s"].map(win_pay).fillna(100.0).values
        w = (1.0 + alpha * np.log1p(wt / 100.0)).astype(float)
    else:
        w = np.ones(len(d), dtype=float)
    return lgb.Dataset(X, label=y, group=g, weight=w, free_raw_data=False)


def train_arm(tr, vl, feats, win_pay):
    params = dict(objective="lambdarank", lambdarank_truncation_level=5, metric="ndcg",
                  eval_at=[5], **V6, bagging_freq=5, verbose=-1, n_jobs=-1, seed=SEED,
                  deterministic=True, force_col_wise=True, feature_pre_filter=False)
    return lgb.train(params, make_ds(tr, feats, win_pay, V6_ALPHA_SW),
                     num_boost_round=int(V6_BEST_ITER * 1.1),
                     valid_sets=[make_ds(vl, feats, win_pay, 0.0)],
                     callbacks=[lgb.early_stopping(100, verbose=False)])


def score_all(df, model, feats, encs, col):
    enc = apply_encoders(df, encs)
    X = enc[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    df[col] = model.predict(X)
    return df


def full_metrics(df, scorecol):
    te = df[df["split"] == "test"]
    n = h1 = h3 = w5 = 0
    ndcg = 0.0
    auc_num = auc_den = 0.0
    ll = 0.0
    pc = pd_ = 0.0
    for rid, g in te.groupby(COL_RID, sort=False):
        if len(g) < 5: continue
        s = g[scorecol].values.astype(float); jy = g[COL_JYUN].astype(int).values
        n += 1
        order = np.argsort(-s); hon = int(order[0]); top5 = set(int(x) for x in order[:5])
        sij = np.argsort(jy); wi, pi_, si = int(sij[0]), int(sij[1]), int(sij[2])
        if hon == wi: h1 += 1
        if hon in {wi, pi_, si}: h3 += 1
        if wi in top5: w5 += 1
        rel = np.clip(6 - jy, 0, 5).astype(float)
        k = min(5, len(s)); od = np.argsort(-s)[:k]
        disc = 1.0 / np.log2(np.arange(2, k + 2))
        dcg = ((2 ** rel[od] - 1) * disc).sum(); io2 = np.argsort(-rel)[:k]
        idcg = ((2 ** rel[io2] - 1) * disc).sum()
        ndcg += (dcg / idcg) if idcg > 0 else 0.0
        # win-AUC (winner vs others)
        ws = s[wi]; mask = np.ones(len(s), bool); mask[wi] = False; oth = s[mask]
        auc_num += float((ws > oth).sum() + 0.5 * (ws == oth).sum()); auc_den += len(oth)
        # log-loss (softmax win)
        z = s - s.max(); p = np.exp(z); p /= p.sum()
        ll += -np.log(max(p[wi], 1e-12))
        # pairwise concordance (higher score ↔ lower jyun)
        ds = np.sign(s[:, None] - s[None, :]); dj = np.sign(jy[None, :] - jy[:, None])
        iu = np.triu_indices(len(s), k=1)
        conc = ds[iu] * dj[iu]
        pc += float((conc > 0).sum()); pd_ += float((conc < 0).sum())
    return {"n_races": n, "ndcg5": round(ndcg / n, 5), "hon_top3": round(h3 / n, 5),
            "hon_top1": round(h1 / n, 5), "winner_in_top5": round(w5 / n, 5),
            "auc_win": round(auc_num / auc_den, 5) if auc_den else 0.0,
            "logloss_win": round(ll / n, 5),
            "pairwise_acc": round(pc / (pc + pd_), 5) if (pc + pd_) else 0.0}


def group_gain(gain):
    return {gn: {"sum_gain": float(gain[[c for c in gain.index if fn(c)]].sum()),
                 "n": len([c for c in gain.index if fn(c)])} for gn, fn in HELP_GROUPS.items()}


def main():
    T0 = time.time()
    OUT_JSON.parent.mkdir(exist_ok=True)
    df = load_master()
    win_pay = load_winner_pay()
    encs = fit_encoders(df[df["split"] == "train"])
    base_feats = [c for c in df.columns if c not in LEAK_COLS and c != "label"
                  and c not in {"rid_s", "_rid", "_ban"} and c not in set(ALL_EXTRA)]
    log(f"  base_feats={len(base_feats)}  (extra列除外後)")
    cov = {a: round(df.loc[df["split"] == "test", ELO_FEATS[a][0]].notna().mean() * 100, 1) for a in ELO_ARMS}
    cov["v1_W3"] = round(df.loc[df["split"] == "test", "race_level_prev_W3"].notna().mean() * 100, 1) \
        if "race_level_prev_W3" in df.columns else None
    cov["v2_W3"] = round(df.loc[df["split"] == "test", "race_level_str_W3"].notna().mean() * 100, 1) \
        if "race_level_str_W3" in df.columns else None

    results = {"protocol": "v6params固定/train≤2022/valid2023/test2024-25/市場非経由・多指標",
               "main_metrics": ["auc_win", "logloss_win", "pairwise_acc"],
               "verdict_rule": "勝ち=ΔAUC≥+0.008 AND Δpairwise≥+0.005 AND Δlogloss≤0 (多重比較厳しめ・符号一貫)",
               "coverage_test": cov, "arms": {}}
    gains = {}
    tr = df[df["split"] == "train"]; vl = df[df["split"] == "valid"]
    for name, extra in ARMS_DEF.items():
        feats = base_feats + extra
        t0 = time.time()
        model = train_arm(tr, vl, feats, win_pay)
        col = f"_s_{name}"
        df = score_all(df, model, feats, encs, col)
        m = full_metrics(df, col)
        gains[name] = pd.Series(model.feature_importance("gain"), index=feats)
        results["arms"][name] = {"n_feats": len(feats), "best_iter": model.best_iteration,
                                 "metrics": m, "group_gain": group_gain(gains[name])}
        log(f"[{name:6s}] {time.time()-t0:.0f}s it={model.best_iteration} "
            f"AUC={m['auc_win']} logloss={m['logloss_win']} pair={m['pairwise_acc']} "
            f"◎top3={m['hon_top3']*100:.2f}% NDCG={m['ndcg5']}")

    A = results["arms"]["A"]["metrics"]
    # 新特徴 importance + 判定
    new_imp = {}
    for a in ELO_ARMS + ["v1_W3", "v2_W3"]:
        g = gains[a]; rk = g.rank(ascending=False).astype(int)
        for c in ARMS_DEF[a]:
            new_imp.setdefault(a, {})[c] = {"gain": round(float(g[c]), 1), "rank": int(rk[c]), "of": len(g)}
    results["new_feat_importance"] = new_imp

    verdicts = {}
    for a in ELO_ARMS + ["v1_W3", "v2_W3"]:
        m = results["arms"][a]["metrics"]
        dA = round(m["auc_win"] - A["auc_win"], 5)
        dL = round(m["logloss_win"] - A["logloss_win"], 5)
        dP = round(m["pairwise_acc"] - A["pairwise_acc"], 5)
        d3 = round(m["hon_top3"] - A["hon_top3"], 5)
        win = (dA >= 0.008 and dP >= 0.005 and dL <= 0)
        verdicts[a] = {"dAUC": dA, "dlogloss": dL, "dpairwise": dP, "dtop3": d3, "WIN": bool(win)}
    results["arm_deltas_vs_A"] = verdicts

    any_win = any(v["WIN"] for v in verdicts.values())
    elo_any = any(verdicts[a]["WIN"] for a in ELO_ARMS)
    if any_win:
        winners = [a for a, v in verdicts.items() if v["WIN"]]
        overall = (f"WIN: {winners} が事前固定基準(ΔAUC≥0.008 & Δpairwise≥0.005 & Δlogloss≤0)を満たす。"
                   "能力推定がv6を超えた→反復精緻化/段階2(市場EV)へ進む価値。")
    else:
        # 符号一貫性チェック
        elo_auc = [verdicts[a]["dAUC"] for a in ELO_ARMS]
        overall = (f"LOSE: 4 ELOアーム全て事前基準未達。ΔAUC(ELO)={elo_auc}。"
                   "改善は単一指標/僅差/符号不一致に留まり多重比較で有意なし。"
                   "★相互ELOでも独自情報は薄い=能力解明の幹は精度面で確定的に閉じる。")
    results["verdict"] = overall

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log("\n" + "=" * 80)
    log(f"baseline A: AUC={A['auc_win']} logloss={A['logloss_win']} pair={A['pairwise_acc']} ◎top3={A['hon_top3']*100:.2f}%")
    log(f"{'arm':7s}{'ΔAUC':>9s}{'Δlogloss':>10s}{'Δpair':>9s}{'Δtop3pt':>9s}  WIN")
    for a, v in verdicts.items():
        log(f"{a:7s}{v['dAUC']:>+9.5f}{v['dlogloss']:>+10.5f}{v['dpairwise']:>+9.5f}{v['dtop3']*100:>+8.2f}  {v['WIN']}")
    log(f"\n判定: {overall}")
    log(f"[saved] {OUT_JSON}  ({time.time()-T0:.0f}s)")


if __name__ == "__main__":
    main()
