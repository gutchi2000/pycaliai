"""
exp_v6_settings.py — v6 最適性監査 (特徴固定・設定だけ動かす)
=============================================================
「◎top3=61.48%/AUC=0.78694 は設定の甘さでなく情報量の天井」を検証。
v6 の 120特徴は固定し、設定軸だけスイープして多指標が動くかを事前固定基準で判定。

軸 (baseline=現行v6: lambdarank, label=clip(6-着順,0,5), α=0.0308, V6 best params):
  軸1 ハイパラ : H_wide(高表現力) / H_reg(強正則化)
  軸2 目的関数 : O_xendcg(rank_xendcg) / O_binary(二値win) / O_regr(回帰)
  軸3 ラベル   : L_binary(1着/否) / L_top3(clip(4-着順,0,3)) / L_margin(着差連続rel)
  軸4 weight/iter: W_none(α=0) / W_strong(α=0.5) / I_long(iter×2.5)
  軸5 特徴削減 : F_trim(baseline gain 下位20列を除外)
多指標: AUC / log-loss / pairwise / ◎top3 / NDCG@5。train≤2022→valid2023→test2024-25。
判定(事前固定): ある設定で ΔAUC≥+0.008 & Δpairwise≥+0.005 → v6に設定の甘さ。全滅→情報量の天井確定。
リークなし(設定のみ・特徴/分割不変)。 出力: reports/exp_v6_settings.json
実行: PYTHONUTF8=1 python exp_v6_settings.py
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
OUT_JSON = BASE / "reports/exp_v6_settings.json"
SEED = 42
COL_RID = "レースID(新/馬番無)"; COL_JYUN = "着順"; COL_BAN = "馬番"; COL_PED = "血統登録番号"
COL_MARGIN = "前走着差タイム"
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


def log(m): print(m, flush=True)


def load_winner_pay():
    df = pd.read_csv(KEKKA_CSV, encoding="cp932", low_memory=False)
    df.columns = ["rid_horse", "ban", "ped", "jyun", "tansho", "fukusho", "wakuren",
                  "umaren", "umatan", "sanrenpuku", "sanrentan"]
    df["rid_s"] = df["rid_horse"].astype(str).str[:16]
    df["jyun"] = pd.to_numeric(df["jyun"], errors="coerce")
    win = df[df["jyun"] == 1].drop_duplicates("rid_s")
    return dict(zip(win["rid_s"].values, pd.to_numeric(win["tansho"], errors="coerce").values))


def load_master():
    log(f"[load] {MASTER_CSV.name}")
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df = df.dropna(subset=[COL_JYUN, COL_RID, "split"]).copy()
    df["rid_s"] = df[COL_RID].astype(str).str[:16]
    jy = df[COL_JYUN].astype(int)
    df["lbl_clip6"] = np.clip(6 - jy, 0, 5).astype(int)
    df["lbl_binwin"] = (jy == 1).astype(int)
    df["lbl_top3"] = np.clip(4 - jy, 0, 3).astype(int)
    # 着差連続rel (chain: 走X着差 = 同馬次走の前走着差)
    df["_dt"] = pd.to_numeric(df["日付"], errors="coerce")
    df = df.sort_values([COL_PED, "_dt", COL_RID])
    mg = pd.to_numeric(df[COL_MARGIN], errors="coerce").groupby(df[COL_PED]).shift(-1)
    relm = np.clip((3.0 - mg) / 3.0 * 5.0, 0, 5)
    df["lbl_margin"] = relm.fillna(df["lbl_clip6"]).round().astype(int)
    df = df.sort_index()
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


def make_ds(d, feats, label_col, alpha, win_pay, grp):
    d = d.sort_values(COL_RID).reset_index(drop=True)
    X = d[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    y = d[label_col].values
    g = np.array([len(list(gr)) for _, gr in groupby(d[COL_RID])]) if grp else None
    if alpha > 0:
        wt = d["rid_s"].map(win_pay).fillna(100.0).values
        w = (1.0 + alpha * np.log1p(wt / 100.0)).astype(float)
    else:
        w = np.ones(len(d), float)
    return lgb.Dataset(X, label=y, group=g, weight=w, free_raw_data=False)


def train(tr, vl, feats, obj, label_col, alpha, ovr, itm, grp, win_pay):
    p = dict(V6); p.update(ovr)
    params = dict(objective=obj, **p, bagging_freq=5, verbose=-1, n_jobs=-1, seed=SEED,
                  deterministic=True, force_col_wise=True, feature_pre_filter=False)
    if obj in ("lambdarank", "rank_xendcg"):
        params.update(metric="ndcg", eval_at=[5], lambdarank_truncation_level=5)
    elif obj == "binary":
        params.update(metric="auc")
    else:
        params.update(metric="l2")
    dtr = make_ds(tr, feats, label_col, alpha, win_pay, grp)
    dvl = make_ds(vl, feats, label_col, 0.0, win_pay, grp)
    return lgb.train(params, dtr, num_boost_round=int(V6_BEST_ITER * itm),
                     valid_sets=[dvl], callbacks=[lgb.early_stopping(100, verbose=False)])


def metrics(df, scorecol):
    te = df[df["split"] == "test"]
    n = h1 = h3 = w5 = 0; ndcg = 0.0; auc_num = auc_den = 0.0; ll = 0.0; pc = pdd = 0.0
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
        k = min(5, len(s)); od = np.argsort(-s)[:k]; disc = 1.0 / np.log2(np.arange(2, k + 2))
        dcg = ((2 ** rel[od] - 1) * disc).sum(); io2 = np.argsort(-rel)[:k]
        idcg = ((2 ** rel[io2] - 1) * disc).sum(); ndcg += (dcg / idcg) if idcg > 0 else 0.0
        ws = s[wi]; mask = np.ones(len(s), bool); mask[wi] = False; oth = s[mask]
        auc_num += float((ws > oth).sum() + 0.5 * (ws == oth).sum()); auc_den += len(oth)
        z = s - s.max(); p = np.exp(z); p /= p.sum(); ll += -np.log(max(p[wi], 1e-12))
        dsg = np.sign(s[:, None] - s[None, :]); djg = np.sign(jy[None, :] - jy[:, None])
        iu = np.triu_indices(len(s), k=1); conc = dsg[iu] * djg[iu]
        pc += float((conc > 0).sum()); pdd += float((conc < 0).sum())
    return {"n_races": n, "ndcg5": round(ndcg / n, 5), "hon_top3": round(h3 / n, 5),
            "hon_top1": round(h1 / n, 5), "winner_in_top5": round(w5 / n, 5),
            "auc_win": round(auc_num / auc_den, 5) if auc_den else 0.0,
            "logloss_win": round(ll / n, 5),
            "pairwise_acc": round(pc / (pc + pdd), 5) if (pc + pdd) else 0.0}


def main():
    T0 = time.time(); OUT_JSON.parent.mkdir(exist_ok=True)
    df = load_master()
    win_pay = load_winner_pay()
    encs = fit_encoders(df[df["split"] == "train"])
    df = apply_encoders(df, encs)
    base_feats = [c for c in df.columns if c not in LEAK_COLS and c != "rid_s" and c != "_dt"
                  and not c.startswith("lbl_")]
    log(f"  base_feats={len(base_feats)}")
    tr = df[df["split"] == "train"]; vl = df[df["split"] == "valid"]

    # ARM 定義 (obj, label, alpha, ovr, itm, grp)
    arms = {
        "baseline": ("lambdarank", "lbl_clip6", V6_ALPHA, {}, 1.1, True),
        "H_wide":   ("lambdarank", "lbl_clip6", V6_ALPHA,
                     dict(num_leaves=255, max_depth=-1, min_data_in_leaf=30, lambda_l1=0.001, lambda_l2=0.001, feature_fraction=1.0), 1.1, True),
        "H_reg":    ("lambdarank", "lbl_clip6", V6_ALPHA,
                     dict(num_leaves=31, max_depth=6, min_data_in_leaf=400, lambda_l1=5.0, lambda_l2=5.0, feature_fraction=0.6), 1.1, True),
        "O_xendcg": ("rank_xendcg", "lbl_clip6", V6_ALPHA, {}, 1.1, True),
        "O_binary": ("binary", "lbl_binwin", V6_ALPHA, {}, 1.5, False),
        "O_regr":   ("regression", "lbl_clip6", V6_ALPHA, {}, 1.5, False),
        "L_binary": ("lambdarank", "lbl_binwin", V6_ALPHA, {}, 1.1, True),
        "L_top3":   ("lambdarank", "lbl_top3", V6_ALPHA, {}, 1.1, True),
        "L_margin": ("lambdarank", "lbl_margin", V6_ALPHA, {}, 1.1, True),
        "W_none":   ("lambdarank", "lbl_clip6", 0.0, {}, 1.1, True),
        "W_strong": ("lambdarank", "lbl_clip6", 0.5, {}, 1.1, True),
        "I_long":   ("lambdarank", "lbl_clip6", V6_ALPHA, {}, 2.5, True),
    }
    results = {"protocol": "特徴固定120/設定のみ変更/train≤2022/valid2023/test2024-25/市場非経由",
               "verdict_rule": "ある設定で ΔAUC≥+0.008 & Δpairwise≥+0.005 → 設定の甘さ。全滅→情報量の天井。",
               "arms": {}}

    base_gain = None
    for name, (obj, lbl, alpha, ovr, itm, grp) in arms.items():
        t0 = time.time()
        model = train(tr, vl, base_feats, obj, lbl, alpha, ovr, itm, grp, win_pay)
        col = f"_s_{name}"
        Xall = df[base_feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
        df[col] = model.predict(Xall)
        m = metrics(df, col)
        results["arms"][name] = {"obj": obj, "label": lbl, "alpha": alpha, "best_iter": model.best_iteration, "metrics": m}
        if name == "baseline":
            base_gain = pd.Series(model.feature_importance("gain"), index=base_feats)
        log(f"[{name:9s}] {time.time()-t0:.0f}s it={model.best_iteration} AUC={m['auc_win']} "
            f"logloss={m['logloss_win']} pair={m['pairwise_acc']} ◎top3={m['hon_top3']*100:.2f}% NDCG={m['ndcg5']}")

    # 軸5 F_trim: baseline gain 下位20列除外
    trim_drop = list(base_gain.sort_values().index[:20])
    feats_trim = [c for c in base_feats if c not in trim_drop]
    t0 = time.time()
    model = train(tr, vl, feats_trim, "lambdarank", "lbl_clip6", V6_ALPHA, {}, 1.1, True, win_pay)
    Xt = df[feats_trim].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    df["_s_F_trim"] = model.predict(Xt)
    m = metrics(df, "_s_F_trim")
    results["arms"]["F_trim"] = {"obj": "lambdarank", "n_feats": len(feats_trim),
                                 "dropped_low_gain": trim_drop, "best_iter": model.best_iteration, "metrics": m}
    log(f"[{'F_trim':9s}] {time.time()-t0:.0f}s it={model.best_iteration} AUC={m['auc_win']} "
        f"pair={m['pairwise_acc']} ◎top3={m['hon_top3']*100:.2f}%  (除外20列例: {trim_drop[:5]})")

    # 判定
    A = results["arms"]["baseline"]["metrics"]
    deltas = {}; any_win = False
    for name, d in results["arms"].items():
        if name == "baseline": continue
        m = d["metrics"]
        dA = round(m["auc_win"] - A["auc_win"], 5); dP = round(m["pairwise_acc"] - A["pairwise_acc"], 5)
        dL = round(m["logloss_win"] - A["logloss_win"], 5); d3 = round(m["hon_top3"] - A["hon_top3"], 5)
        win = (dA >= 0.008 and dP >= 0.005)
        any_win = any_win or win
        deltas[name] = {"dAUC": dA, "dpairwise": dP, "dlogloss": dL, "dtop3": d3, "WIN": bool(win)}
    results["deltas_vs_baseline"] = deltas

    obj_arms = ["O_xendcg", "O_binary", "O_regr"]
    obj_win = any(deltas[a]["WIN"] for a in obj_arms)
    if any_win and obj_win and not any(deltas[a]["WIN"] for a in deltas if a not in obj_arms):
        verdict = ("OBJECTIVE_MISMATCH: 目的関数変更のみ有意改善。NDCG最適化がAUC/能力推定に最善でなかった。"
                   f"勝ち={[a for a in obj_arms if deltas[a]['WIN']]}")
    elif any_win:
        winners = [a for a, v in deltas.items() if v["WIN"]]
        verdict = f"SUBOPTIMAL: 設定変更 {winners} が ΔAUC≥0.008 & Δpairwise≥0.005。v6に設定の甘さあり。"
    else:
        aucs = {a: deltas[a]["dAUC"] for a in deltas}
        verdict = ("CEILING_CONFIRMED: 全12+1設定変更で ΔAUC<0.008 or Δpairwise<0.005。"
                   "ハイパラ/目的関数/ラベル/weight/特徴削減のどれを動かしても有意改善なし。"
                   "★62%/AUC0.787 は設定の甘さでなく情報量の天井で確定。")
    results["verdict"] = verdict

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log("\n" + "=" * 82)
    log(f"baseline: AUC={A['auc_win']} logloss={A['logloss_win']} pair={A['pairwise_acc']} ◎top3={A['hon_top3']*100:.2f}% NDCG={A['ndcg5']}")
    log(f"{'arm':10s}{'ΔAUC':>9s}{'Δpair':>9s}{'Δlogloss':>10s}{'Δtop3pt':>9s}  WIN")
    for a, v in deltas.items():
        log(f"{a:10s}{v['dAUC']:>+9.5f}{v['dpairwise']:>+9.5f}{v['dlogloss']:>+10.5f}{v['dtop3']*100:>+8.2f}  {v['WIN']}")
    log(f"\n判定: {verdict}")
    log(f"[saved] {OUT_JSON}  ({time.time()-T0:.0f}s)")


if __name__ == "__main__":
    main()
