"""
race_level_exp.py — レースレベル特徴の A/B (市場非経由・印精度のみ)
==================================================================
arm A  = v6 120特徴 (baseline 再学習, v6 best params + sample_weight 固定)
arm B-W1/W2/W3 = A + race_level_prev_W{1,2,3} (1列, 前走レベル)

v6 best params 固定。train≤2022 学習 → valid2023 early stop → test2024-25 評価。
指標: NDCG@5 / ◎top1 / ◎top3 / win∈top5 / 新特徴importance(gain) /
      既存 kako5・前走・hosei の importance が食われたか / カバレッジ。
※市場 ΔR²/ROI は測らない (市場非経由の能力推定精度のみ)。
※近走平均レベルは今回未実装 (前走レベル1列のみ・MVP)。

前提: race_level_feats.parquet が生成済 (リーク検査 PASS)。
出力: reports/race_level_exp.json
実行: PYTHONUTF8=1 python race_level_exp.py
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
PARQUET = BASE / "data/race_level_feats.parquet"
OUT_JSON = BASE / "reports/race_level_exp.json"
SEED = 42

COL_RID = "レースID(新/馬番無)"
COL_JYUN = "着順"
COL_BAN = "馬番"

LEAK_COLS = {"着順", "fukusho_flag", "roi_target", "レースID(新)", "レースID(新/馬番無)",
             "馬名", "レース名", "発走時刻", "date_dt", "日付", "血統登録番号", "split"}
CAT_COLS = ["場所", "芝・ダ", "コース区分", "芝(内・外)", "馬場状態", "天気", "クラス名",
            "種牡馬", "父タイプ名", "母父馬", "母父タイプ名", "毛色", "馬主(最新/仮想)", "生産者",
            "騎手コード", "調教師コード", "年齢限定", "限定", "性別限定", "指定条件", "重量種別",
            "性別", "ブリンカー", "前走場所", "前芝・ダ", "前走馬場状態", "前走競走種別", "前好走"]

V6 = dict(num_leaves=59, max_depth=12, min_data_in_leaf=197, learning_rate=0.05083158501542684,
          feature_fraction=0.876098829427658, bagging_fraction=0.7031707810405968,
          lambda_l1=0.0011077902520957399, lambda_l2=7.537933313450104)
V6_BEST_ITER = 469
V6_ALPHA_SW = 0.03083978412534253

NEW_BY_WINDOW = {"W1": ["race_level_prev_W1"], "W2": ["race_level_prev_W2"], "W3": ["race_level_prev_W3"]}
ALL_NEW = ["race_level_prev_W1", "race_level_prev_W2", "race_level_prev_W3"]

HELP_GROUPS = {
    "kako5": lambda c: c.startswith("kako5_"),
    "hist": lambda c: c.startswith("hist_"),
    "hosei": lambda c: "hosei" in c,
    "zensou": lambda c: c.startswith("前"),
    "trnH": lambda c: c.startswith("trnH_") or c.startswith("trnW_"),
    "race_level_new": lambda c: c.startswith("race_level_"),
}


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
    rl = pd.read_parquet(PARQUET)
    rl["rid16"] = rl["rid16"].astype(str)
    rl["ban"] = pd.to_numeric(rl["ban"], errors="coerce").fillna(0).astype(int)
    df = df.merge(rl, left_on=["_rid", "_ban"], right_on=["rid16", "ban"], how="left")
    cov = {sp: round(df.loc[df["split"] == sp, "race_level_prev_W3"].notna().mean() * 100, 1)
           for sp in ["train", "valid", "test"]}
    log(f"  rows={len(df):,}  level W3 カバレッジ={cov}")
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


def mark_metrics(df, scorecol):
    te = df[df["split"] == "test"]
    n = h1 = h3 = w5 = 0
    ndcg = 0.0
    for rid, g in te.groupby(COL_RID, sort=False):
        if len(g) < 5: continue
        s = g[scorecol].values; jy = g[COL_JYUN].astype(int).values
        n += 1
        order = np.argsort(-s); hon = int(order[0]); top5 = set(int(x) for x in order[:5])
        sij = np.argsort(jy); wi, pi_, si = int(sij[0]), int(sij[1]), int(sij[2])
        if hon == wi: h1 += 1
        if hon in {wi, pi_, si}: h3 += 1
        if wi in top5: w5 += 1
        rel = np.clip(6 - jy, 0, 5).astype(float)
        k = min(5, len(s)); od = np.argsort(-s)[:k]
        disc = 1.0 / np.log2(np.arange(2, k + 2))
        dcg = ((2 ** rel[od] - 1) * disc).sum()
        io2 = np.argsort(-rel)[:k]
        idcg = ((2 ** rel[io2] - 1) * disc).sum()
        ndcg += (dcg / idcg) if idcg > 0 else 0.0
    return {"n_races": n, "ndcg5": round(ndcg / n, 4), "hon_top1": round(h1 / n, 4),
            "hon_top3": round(h3 / n, 4), "winner_in_top5": round(w5 / n, 4)}


def group_gain(gain):
    out = {}
    for gname, fn in HELP_GROUPS.items():
        cols = [c for c in gain.index if fn(c)]
        out[gname] = {"sum_gain": float(gain[cols].sum()), "n": len(cols)}
    return out


def eval_arm(name, df, feats, encs, win_pay, results):
    log(f"\n[{name}] train (feats={len(feats)}) ...")
    tr = df[df["split"] == "train"]; vl = df[df["split"] == "valid"]
    t0 = time.time()
    model = train_arm(tr, vl, feats, win_pay)
    col = f"_score_{name}"
    df = score_all(df, model, feats, encs, col)
    marks = mark_metrics(df, col)
    log(f"  {time.time()-t0:.0f}s best_iter={model.best_iteration}  "
        f"NDCG@5={marks['ndcg5']} ◎top1={marks['hon_top1']*100:.2f}% "
        f"◎top3={marks['hon_top3']*100:.2f}% win∈top5={marks['winner_in_top5']*100:.2f}%")
    gain = pd.Series(model.feature_importance("gain"), index=feats)
    results[name] = {"n_feats": len(feats), "best_iter": model.best_iteration,
                     "marks_test": marks, "group_gain": group_gain(gain)}
    return df, gain, results


def main():
    T0 = time.time()
    OUT_JSON.parent.mkdir(exist_ok=True)
    results = {"protocol": "v6params固定 / train≤2022 / valid2023 early-stop / test2024-25 / 市場非経由",
               "note": "前走レベルのみ(近走平均は未実装MVP)"}
    df, cov = load_master()
    results["coverage_W3"] = cov
    win_pay = load_winner_pay()
    encs = fit_encoders(df[df["split"] == "train"])

    base_feats = [c for c in df.columns if c not in LEAK_COLS and c != "label"
                  and c not in {"rid_s", "_rid", "_ban", "rid16", "ban"} and c not in set(ALL_NEW)]
    log(f"  base_feats={len(base_feats)}")

    # arm A
    df, gainA, results = eval_arm("A", df, base_feats, encs, win_pay, results)

    # arm B per window
    new_imp = {}
    for win, cols in NEW_BY_WINDOW.items():
        feats_B = base_feats + cols
        df, gainB, results = eval_arm(f"B_{win}", df, feats_B, encs, win_pay, results)
        rankB = gainB.rank(ascending=False).astype(int)
        for c in cols:
            new_imp[c] = {"gain": float(gainB[c]), "rank": int(rankB[c]), "of": len(feats_B)}
        # group_gain shift vs A
        ggA = results["A"]["group_gain"]; ggB = results[f"B_{win}"]["group_gain"]
        results[f"B_{win}"]["group_gain_shift_vs_A"] = {
            k: {"A": ggA[k]["sum_gain"], "B": ggB[k]["sum_gain"],
                "pct": round(100 * (ggB[k]["sum_gain"] - ggA[k]["sum_gain"]) / ggA[k]["sum_gain"], 1)
                if ggA[k]["sum_gain"] else None} for k in ggA}
    results["new_feat_importance"] = new_imp

    # 判定
    a3 = results["A"]["marks_test"]["hon_top3"]
    best_win, best_d = None, -9
    for win in NEW_BY_WINDOW:
        d = results[f"B_{win}"]["marks_test"]["hon_top3"] - a3
        if d > best_d:
            best_d, best_win = d, win
    top_rank = min((v["rank"] for v in new_imp.values()), default=9999)
    if best_d >= 0.005 and top_rank <= 40:
        verdict = (f"EFFECTIVE: B_{best_win} の◎top3が A比 {best_d*100:+.2f}pt 上昇かつ新特徴importance上位"
                   f"({top_rank}位)。レースレベル逆算は能力推定に効く。反復整合版・窓最適化へ進む価値。")
    elif abs(best_d) < 0.003:
        verdict = (f"NO_GAIN: 最良窓でも◎top3 A比 {best_d*100:+.2f}pt(誤差内)。"
                   "粗いレベルスコアでは増分なし。指標/集約法の見直しか、能力解明軸も厳しい。")
    else:
        verdict = f"WEAK: 最良 B_{best_win} ◎top3 A比 {best_d*100:+.2f}pt。弱い動き、importanceと併せ要判断。"
    results["verdict"] = verdict
    results["summary"] = {
        "marks_top3_A": a3,
        "marks_top3_B": {win: results[f"B_{win}"]["marks_test"]["hon_top3"] for win in NEW_BY_WINDOW},
        "ndcg5_A": results["A"]["marks_test"]["ndcg5"],
        "ndcg5_B": {win: results[f"B_{win}"]["marks_test"]["ndcg5"] for win in NEW_BY_WINDOW},
        "best_window": best_win, "best_top3_delta": round(best_d, 5), "top_new_feat_rank": top_rank,
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log("\n" + "=" * 72)
    log("=== 判定 ===")
    log(f"  ◎top3: A={a3*100:.2f}%  " + "  ".join(
        f"B_{w}={results[f'B_{w}']['marks_test']['hon_top3']*100:.2f}%" for w in NEW_BY_WINDOW))
    log(f"  NDCG@5: A={results['A']['marks_test']['ndcg5']}  " + "  ".join(
        f"B_{w}={results[f'B_{w}']['marks_test']['ndcg5']}" for w in NEW_BY_WINDOW))
    log(f"  新特徴importance: " + "  ".join(f"{c}={v['rank']}位" for c, v in new_imp.items()))
    log(f"  判定: {verdict}")
    log(f"[saved] {OUT_JSON}  ({time.time()-T0:.0f}s)")


if __name__ == "__main__":
    main()
