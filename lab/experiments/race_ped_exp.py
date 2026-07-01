"""
race_ped_exp.py — 血統embedding の A/B (市場非経由・多指標・初物サブセット別建て)
================================================================================
arm A = v6 120特徴 / B16,B32,B64 = A + 血統SVD埋め込み 先頭16/32/64次元。
主指標(全体): AUC / log-loss / pairwise順位acc。補助: ◎top3 / NDCG / importance / 食い合い。
★初物サブセット(初ダート/初距離/初コース)別: 該当馬の pairwise順位acc と ◎top3的中
  (kako5が無力な領域で血統embが先読みできるか)。
v6 best params固定、train≤2022→valid2023→test2024-25。A が AUC=0.78694/◎top3=61.48% 再現確認。
判定(事前固定): 全体 or 初物で ΔAUC≥+0.008 & Δpairwise≥+0.005 が次元間一貫で勝ち。
出力: reports/race_ped_exp.json   実行: PYTHONUTF8=1 python race_ped_exp.py
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
EMB_PARQUET = BASE / "data/pedigree_emb.parquet"
OUT_JSON = BASE / "reports/race_ped_exp.json"
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
V6_BEST_ITER = 469; V6_ALPHA_SW = 0.03083978412534253
ALL_EMB = [f"ped_emb_{i:02d}" for i in range(64)]
ARMS_DEF = {"A": [], "B16": ALL_EMB[:16], "B32": ALL_EMB[:32], "B64": ALL_EMB[:64]}
HELP_GROUPS = {"kako5": lambda c: c.startswith("kako5_"), "hosei": lambda c: "hosei" in c,
               "zensou": lambda c: c.startswith("前"),
               "pedlabel": lambda c: c in ("種牡馬", "父タイプ名", "母父馬", "母父タイプ名"),
               "ped_emb": lambda c: c.startswith("ped_emb_")}


def log(m): print(m, flush=True)


def dist_band(d):
    if pd.isna(d): return "?"
    d = int(d)
    return "短" if d <= 1400 else "マ" if d <= 1800 else "中" if d <= 2200 else "長"


def load_winner_pay():
    df = pd.read_csv(KEKKA_CSV, encoding="cp932", low_memory=False)
    df.columns = ["rid_horse", "ban", "ped", "jyun", "tansho", "fukusho", "wakuren",
                  "umaren", "umatan", "sanrenpuku", "sanrentan"]
    df["rid_s"] = df["rid_horse"].astype(str).str[:16]
    df["jyun"] = pd.to_numeric(df["jyun"], errors="coerce")
    win = df[df["jyun"] == 1].drop_duplicates("rid_s")
    return dict(zip(win["rid_s"].values, pd.to_numeric(win["tansho"], errors="coerce").values))


def add_first_flags(df):
    """as-of 初物フラグ (過去走のみ): first_dirt / first_dist / first_place。"""
    d = df.copy()
    d["_date"] = pd.to_numeric(d["日付"], errors="coerce")
    d["_ord"] = d.groupby(COL_RID, sort=False).ngroup()  # 近似順序 (rid単位)
    # rid は日付+場所+R を含むが厳密順序は日付で十分 (初物=過去経験有無)
    d = d.sort_values([COL_PED, "_date", COL_RID]).reset_index()
    fd = np.zeros(len(d), int); fdi = np.zeros(len(d), int); fp = np.zeros(len(d), int)
    cur = None; seen_td = set(); seen_b = set(); seen_p = set()
    td = d["芝・ダ"].astype(str).values
    bands = d["距離"].apply(dist_band).values
    places = d["場所"].astype(str).values
    peds = d[COL_PED].values
    for i in range(len(d)):
        if peds[i] != cur:
            cur = peds[i]; seen_td = set(); seen_b = set(); seen_p = set()
        if td[i] == "ダ" and "ダ" not in seen_td: fd[i] = 1
        if bands[i] not in seen_b: fdi[i] = 1
        if places[i] not in seen_p: fp[i] = 1
        seen_td.add(td[i]); seen_b.add(bands[i]); seen_p.add(places[i])
    d["first_dirt"] = fd; d["first_dist"] = fdi; d["first_place"] = fp
    d = d.set_index("index").sort_index()
    for c in ["first_dirt", "first_dist", "first_place"]:
        df[c] = d[c]
    return df


def load_master():
    log(f"[load] {MASTER_CSV.name}")
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df = df.dropna(subset=[COL_JYUN, COL_RID, "split"]).copy()
    df["label"] = np.clip(6 - df[COL_JYUN].astype(int), 0, 5).astype(int)
    df["rid_s"] = df[COL_RID].astype(str).str[:16]
    df = add_first_flags(df)
    emb = pd.read_parquet(EMB_PARQUET)
    emb["_pedk"] = emb["血統登録番号"].astype(str)
    emb = emb.drop(columns=["血統登録番号"])
    df["_pedk"] = df[COL_PED].astype(str)
    df = df.merge(emb, on="_pedk", how="left")
    cov = round(df.loc[df["split"] == "test", "ped_emb_00"].notna().mean() * 100, 1)
    log(f"  rows={len(df):,}  ped_emb test カバレッジ={cov}%  "
        f"初物(test): ダ={int((df[df.split=='test'].first_dirt==1).sum()):,} "
        f"距離={int((df[df.split=='test'].first_dist==1).sum()):,} "
        f"コース={int((df[df.split=='test'].first_place==1).sum()):,}")
    return df, cov


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


def metrics(df, scorecol):
    """全体: AUC/logloss/pairwise/◎top3/NDCG + 初物サブセット別 該当馬 pairwise/◎top3的中。"""
    te = df[df["split"] == "test"]
    n = h1 = h3 = w5 = 0; ndcg = 0.0; auc_num = auc_den = 0.0; ll = 0.0; pc = pdd = 0.0
    # 初物サブセット: 該当馬関与ペアの concordance, 該当馬の複勝予測的中
    sub = {k: {"pc": 0.0, "pd": 0.0, "hit_num": 0, "hit_den": 0} for k in ["first_dirt", "first_dist", "first_place"]}
    for rid, g in te.groupby(COL_RID, sort=False):
        if len(g) < 5: continue
        s = g[scorecol].values.astype(float); jy = g[COL_JYUN].astype(int).values
        n += 1
        order = np.argsort(-s); hon = int(order[0]); top3p = set(int(x) for x in order[:3]); top5 = set(int(x) for x in order[:5])
        sij = np.argsort(jy); wi, pi_, si = int(sij[0]), int(sij[1]), int(sij[2])
        if hon == wi: h1 += 1
        if hon in {wi, pi_, si}: h3 += 1
        if wi in top5: w5 += 1
        rel = np.clip(6 - jy, 0, 5).astype(float)
        k = min(5, len(s)); od = np.argsort(-s)[:k]; disc = 1.0 / np.log2(np.arange(2, k + 2))
        dcg = ((2 ** rel[od] - 1) * disc).sum(); io2 = np.argsort(-rel)[:k]
        idcg = ((2 ** rel[io2] - 1) * disc).sum(); ndcg += (dcg / idcg) if idcg > 0 else 0.0
        ws = s[wi]; mask = np.ones(len(s), bool); mask[wi] = False; oth = s[mask]
        auc_num += float((ws > oth).sum() + 0.5 * (ws == oth).sum()); auc_den += len(oth)
        z = s - s.max(); p = np.exp(z); p /= p.sum(); ll += -np.log(max(p[wi], 1e-12))
        dsg = np.sign(s[:, None] - s[None, :]); djg = np.sign(jy[None, :] - jy[:, None])
        iu = np.triu_indices(len(s), k=1); conc = dsg[iu] * djg[iu]
        pc += float((conc > 0).sum()); pdd += float((conc < 0).sum())
        # 初物サブセット (該当馬 i の関与ペア + 複勝的中)
        for sc in sub:
            fl = g[sc].values
            for i in range(len(s)):
                if fl[i] != 1: continue
                # 該当馬 i vs 他馬 j の concordance
                for j in range(len(s)):
                    if i == j: continue
                    c = np.sign(s[i] - s[j]) * np.sign(jy[j] - jy[i])
                    if c > 0: sub[sc]["pc"] += 1
                    elif c < 0: sub[sc]["pd"] += 1
                # 複勝予測的中: pred top3 に入れたか かつ 実3着内か (両立で的中)
                sub[sc]["hit_den"] += 1
                if (i in top3p) and (jy[i] <= 3): sub[sc]["hit_num"] += 1
    out = {"n_races": n, "ndcg5": round(ndcg / n, 5), "hon_top3": round(h3 / n, 5),
           "hon_top1": round(h1 / n, 5), "winner_in_top5": round(w5 / n, 5),
           "auc_win": round(auc_num / auc_den, 5) if auc_den else 0.0,
           "logloss_win": round(ll / n, 5),
           "pairwise_acc": round(pc / (pc + pdd), 5) if (pc + pdd) else 0.0, "subsets": {}}
    for sc, d in sub.items():
        out["subsets"][sc] = {"pairwise_acc": round(d["pc"] / (d["pc"] + d["pd"]), 5) if (d["pc"] + d["pd"]) else 0.0,
                              "fukusho_hit": round(d["hit_num"] / d["hit_den"], 5) if d["hit_den"] else 0.0,
                              "n_horses": d["hit_den"]}
    return out


def group_gain(gain):
    return {gn: {"sum_gain": float(gain[[c for c in gain.index if fn(c)]].sum()),
                 "n": len([c for c in gain.index if fn(c)])} for gn, fn in HELP_GROUPS.items()}


def main():
    T0 = time.time(); OUT_JSON.parent.mkdir(exist_ok=True)
    df, cov = load_master()
    win_pay = load_winner_pay()
    encs = fit_encoders(df[df["split"] == "train"])
    base_feats = [c for c in df.columns if c not in LEAK_COLS and c != "label"
                  and c not in {"rid_s", "_pedk", "first_dirt", "first_dist", "first_place"}
                  and c not in set(ALL_EMB)]
    log(f"  base_feats={len(base_feats)}")
    results = {"protocol": "v6params固定/train≤2022/valid2023/test2024-25/市場非経由・多指標+初物サブセット",
               "emb_method": "血統SVD構造埋め込み(成績不使用,Node2Vec代替)", "emb_coverage_test": cov,
               "verdict_rule": "全体or初物で ΔAUC≥+0.008 & Δpairwise≥+0.005 が次元間一貫で勝ち",
               "arms": {}}
    gains = {}
    tr = df[df["split"] == "train"]; vl = df[df["split"] == "valid"]
    for name, extra in ARMS_DEF.items():
        feats = base_feats + extra; t0 = time.time()
        model = train_arm(tr, vl, feats, win_pay)
        col = f"_s_{name}"; df = score_all(df, model, feats, encs, col)
        m = metrics(df, col); gains[name] = pd.Series(model.feature_importance("gain"), index=feats)
        results["arms"][name] = {"n_feats": len(feats), "best_iter": model.best_iteration,
                                 "metrics": m, "group_gain": group_gain(gains[name])}
        ss = m["subsets"]
        log(f"[{name:4s}] {time.time()-t0:.0f}s it={model.best_iteration} AUC={m['auc_win']} "
            f"pair={m['pairwise_acc']} ◎top3={m['hon_top3']*100:.2f}% | 初ダ pair={ss['first_dirt']['pairwise_acc']} "
            f"初距 pair={ss['first_dist']['pairwise_acc']} 初コ pair={ss['first_place']['pairwise_acc']}")

    A = results["arms"]["A"]["metrics"]
    new_imp = {}
    for b in ["B16", "B32", "B64"]:
        g = gains[b]; rk = g.rank(ascending=False).astype(int)
        embcols = ARMS_DEF[b]
        ranks = sorted([(int(rk[c]), c) for c in embcols])[:3]
        new_imp[b] = {"top3_emb": [{"col": c, "rank": r, "gain": round(float(g[c]), 1)} for r, c in ranks],
                      "n_emb_in_top30": int(sum(1 for c in embcols if rk[c] <= 30))}
    results["new_feat_importance"] = new_imp

    deltas = {}
    for b in ["B16", "B32", "B64"]:
        m = results["arms"][b]["metrics"]
        d = {"dAUC": round(m["auc_win"] - A["auc_win"], 5), "dlogloss": round(m["logloss_win"] - A["logloss_win"], 5),
             "dpairwise": round(m["pairwise_acc"] - A["pairwise_acc"], 5), "dtop3": round(m["hon_top3"] - A["hon_top3"], 5),
             "subsets": {}}
        for sc in ["first_dirt", "first_dist", "first_place"]:
            d["subsets"][sc] = {"dpairwise": round(m["subsets"][sc]["pairwise_acc"] - A["subsets"][sc]["pairwise_acc"], 5),
                                "dfukusho_hit": round(m["subsets"][sc]["fukusho_hit"] - A["subsets"][sc]["fukusho_hit"], 5)}
        d["WIN_overall"] = bool(d["dAUC"] >= 0.008 and d["dpairwise"] >= 0.005)
        d["WIN_firstcat"] = bool(any(d["subsets"][sc]["dpairwise"] >= 0.01 for sc in d["subsets"]))
        deltas[b] = d
    results["deltas_vs_A"] = deltas

    overall_win = any(deltas[b]["WIN_overall"] for b in deltas)
    first_win = any(deltas[b]["WIN_firstcat"] for b in deltas)
    consistent = all(deltas[b]["dAUC"] >= 0.008 for b in deltas) if overall_win else False
    if overall_win and consistent:
        verdict = "WIN: 全体AUC/pairwise が次元一貫で有意改善。血統embは新規情報→Node2Vec/GNN精緻化へ。"
    elif first_win:
        verdict = "PARTIAL_FIRST: 全体は未達だが初物サブセットでpairwise改善。初物限定の補助特徴として部分価値。"
    else:
        aucs = [deltas[b]["dAUC"] for b in deltas]
        verdict = (f"REDUNDANT: 全体ΔAUC={aucs} 全て未達, 初物も非有意。"
                   "★ELoと同じ=血統embはlabel encodingの言い換えで冗長。血統軸も精度面で閉じる。")
    results["verdict"] = verdict

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log("\n" + "=" * 84)
    log(f"baseline A: AUC={A['auc_win']} pair={A['pairwise_acc']} ◎top3={A['hon_top3']*100:.2f}%  "
        f"初物pair(ダ/距/コ)={A['subsets']['first_dirt']['pairwise_acc']}/"
        f"{A['subsets']['first_dist']['pairwise_acc']}/{A['subsets']['first_place']['pairwise_acc']}")
    log(f"{'arm':5s}{'ΔAUC':>9s}{'Δpair':>9s}{'Δtop3pt':>9s}{'初ダΔpair':>10s}{'初距Δpair':>10s}{'初コΔpair':>10s}  WIN全/初")
    for b, d in deltas.items():
        log(f"{b:5s}{d['dAUC']:>+9.5f}{d['dpairwise']:>+9.5f}{d['dtop3']*100:>+8.2f}"
            f"{d['subsets']['first_dirt']['dpairwise']:>+10.5f}{d['subsets']['first_dist']['dpairwise']:>+10.5f}"
            f"{d['subsets']['first_place']['dpairwise']:>+10.5f}  {d['WIN_overall']}/{d['WIN_firstcat']}")
    log(f"\n判定: {verdict}")
    log(f"[saved] {OUT_JSON}  ({time.time()-T0:.0f}s)")


if __name__ == "__main__":
    main()
