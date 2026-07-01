"""
exp_recheck.py — 能力解明4実験の再測定 (正しい CAT_COLS エンコード版)
=====================================================================
前回 race_level_exp/race_elo_exp/race_ped_exp は学習時に CAT_COLS を未エンコード(-9999定数化)
していた (exp_v6_settings.py で発覚)。本スクリプトは exp_v6_settings と同じ「df 全体を学習前に
label-encode」する正しい処理で、既存の特徴 parquet を流用して全 arm を測り直す。

★baseline 再現ゲート: arm A が ◎top3≈61.81% / AUC≈0.78962 / pair≈0.68419 を再現するか最初に確認。

arm: baseline / ped_B16,B32,B64(+初物サブセット) / elo_T1M1,T1M2,T2M1,T2M2 / lv_v1_W3 / lv_v2_W3
特徴 parquet は生成時にリーク検査済 (成績/未来不使用) を流用。時系列分割不変。
エンコードは train のカテゴリで fit → valid/test に適用 (未知=__NaN__、test情報をfitに使わない)。
多指標: AUC/log-loss/pairwise/◎top3/NDCG + importance(血統label gain含む) + 食い合い + カバレッジ。
判定(事前固定): 全体 or 初物で ΔAUC≥+0.008 & Δpairwise≥+0.005 が次元/アーム間一貫で本物の増分。
出力: reports/exp_recheck.json  実行: PYTHONUTF8=1 python exp_recheck.py
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
OUT_JSON = BASE / "reports/exp_recheck.json"
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

ALL_EMB = [f"ped_emb_{i:02d}" for i in range(64)]
ELO_ARMS = ["T1M1", "T1M2", "T2M1", "T2M2"]
ELO_FEATS = {a: [f"elo_{a}_horse", f"elo_{a}_level", f"elo_{a}_vs", f"elo_{a}_prevlevel"] for a in ELO_ARMS}
LV_ALL = ["race_level_prev_W1", "race_level_prev_W2", "race_level_prev_W3", "race_level_str_W2", "race_level_str_W3"]
EXTRA = set(ALL_EMB) | set(sum(ELO_FEATS.values(), [])) | set(LV_ALL) | {
    "first_dirt", "first_dist", "first_place", "rid_s", "_rid", "_ban", "_pedk", "_dt", "label"}
HELP_GROUPS = {"kako5": lambda c: c.startswith("kako5_"), "hosei": lambda c: "hosei" in c,
               "zensou": lambda c: c.startswith("前"), "trnH": lambda c: c.startswith(("trnH_", "trnW_")),
               "pedlabel": lambda c: c in ("種牡馬", "父タイプ名", "母父馬", "母父タイプ名"),
               "ped_emb": lambda c: c.startswith("ped_emb_"), "elo": lambda c: c.startswith("elo_"),
               "racelevel": lambda c: c.startswith("race_level_")}

ARMS_DEF = {"baseline": [], "ped_B16": ALL_EMB[:16], "ped_B32": ALL_EMB[:32], "ped_B64": ALL_EMB[:64]}
for a in ELO_ARMS: ARMS_DEF[f"elo_{a}"] = ELO_FEATS[a]
ARMS_DEF["lv_v1_W3"] = ["race_level_prev_W3"]; ARMS_DEF["lv_v2_W3"] = ["race_level_str_W3"]
PED_ARMS = ["ped_B16", "ped_B32", "ped_B64"]


def log(m): print(m, flush=True)


def dist_band(d):
    if pd.isna(d): return "?"
    d = int(d); return "短" if d <= 1400 else "マ" if d <= 1800 else "中" if d <= 2200 else "長"


def load_winner_pay():
    df = pd.read_csv(KEKKA_CSV, encoding="cp932", low_memory=False)
    df.columns = ["rid_horse", "ban", "ped", "jyun", "tansho", "fukusho", "wakuren",
                  "umaren", "umatan", "sanrenpuku", "sanrentan"]
    df["rid_s"] = df["rid_horse"].astype(str).str[:16]; df["jyun"] = pd.to_numeric(df["jyun"], errors="coerce")
    win = df[df["jyun"] == 1].drop_duplicates("rid_s")
    return dict(zip(win["rid_s"].values, pd.to_numeric(win["tansho"], errors="coerce").values))


def add_first_flags(df):
    d = df.copy(); d["_dd"] = pd.to_numeric(d["日付"], errors="coerce")
    d = d.sort_values([COL_PED, "_dd", COL_RID]).reset_index()
    fd = np.zeros(len(d), int); fi = np.zeros(len(d), int); fp = np.zeros(len(d), int)
    cur = None; std = set(); sb = set(); sp = set()
    td = d["芝・ダ"].astype(str).values; bd = d["距離"].apply(dist_band).values; pl = d["場所"].astype(str).values
    pe = d[COL_PED].values
    for i in range(len(d)):
        if pe[i] != cur: cur = pe[i]; std, sb, sp = set(), set(), set()
        if td[i] == "ダ" and "ダ" not in std: fd[i] = 1
        if bd[i] not in sb: fi[i] = 1
        if pl[i] not in sp: fp[i] = 1
        std.add(td[i]); sb.add(bd[i]); sp.add(pl[i])
    d["first_dirt"] = fd; d["first_dist"] = fi; d["first_place"] = fp
    d = d.set_index("index").sort_index()
    for c in ["first_dirt", "first_dist", "first_place"]:
        df[c] = d[c]
    return df


def merge_pq(df, path, key_cols, val_prefix=None):
    p = BASE / path
    if not p.exists():
        log(f"  !! parquet 不在: {path}"); return df
    rl = pd.read_parquet(p)
    if "rid16" in rl.columns:
        rl["rid16"] = rl["rid16"].astype(str); rl["ban"] = pd.to_numeric(rl["ban"], errors="coerce").fillna(0).astype(int)
        df = df.merge(rl, left_on=["_rid", "_ban"], right_on=["rid16", "ban"], how="left", suffixes=("", "_d"))
        df = df.drop(columns=[c for c in df.columns if c.endswith("_d") or c in ("rid16", "ban")], errors="ignore")
    else:  # pedigree (血統登録番号)
        rl["_pedk"] = rl["血統登録番号"].astype(str); rl = rl.drop(columns=["血統登録番号"])
        df = df.merge(rl, on="_pedk", how="left")
    return df


def load_master():
    log(f"[load] {MASTER_CSV.name}")
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df = df.dropna(subset=[COL_JYUN, COL_RID, "split"]).copy()
    df["label"] = np.clip(6 - df[COL_JYUN].astype(int), 0, 5).astype(int)
    df["rid_s"] = df[COL_RID].astype(str).str[:16]; df["_rid"] = df[COL_RID].astype(str)
    df["_ban"] = pd.to_numeric(df[COL_BAN], errors="coerce").fillna(0).astype(int)
    df["_pedk"] = df[COL_PED].astype(str)
    df = add_first_flags(df)                      # ★エンコード前 (生カテゴリ使用)
    df = merge_pq(df, "data/pedigree_emb.parquet", None)
    df = merge_pq(df, "data/elo_feats.parquet", None)
    df = merge_pq(df, "data/race_level_feats.parquet", None)
    df = merge_pq(df, "data/race_level_feats_v2.parquet", None)
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


def metrics(df, scorecol, with_subset=False):
    te = df[df["split"] == "test"]
    n = h1 = h3 = w5 = 0; ndcg = 0.0; auc_num = auc_den = 0.0; ll = 0.0; pc = pdd = 0.0
    sub = {k: {"pc": 0.0, "pd": 0.0, "hn": 0, "hd": 0} for k in ["first_dirt", "first_dist", "first_place"]} if with_subset else {}
    for rid, g in te.groupby(COL_RID, sort=False):
        if len(g) < 5: continue
        s = g[scorecol].values.astype(float); jy = g[COL_JYUN].astype(int).values; n += 1
        order = np.argsort(-s); hon = int(order[0]); top3p = set(int(x) for x in order[:3]); top5 = set(int(x) for x in order[:5])
        sij = np.argsort(jy); wi, pi_, si = int(sij[0]), int(sij[1]), int(sij[2])
        if hon == wi: h1 += 1
        if hon in {wi, pi_, si}: h3 += 1
        if wi in top5: w5 += 1
        rel = np.clip(6 - jy, 0, 5).astype(float); k = min(5, len(s)); od = np.argsort(-s)[:k]
        disc = 1.0 / np.log2(np.arange(2, k + 2)); dcg = ((2 ** rel[od] - 1) * disc).sum()
        io2 = np.argsort(-rel)[:k]; idcg = ((2 ** rel[io2] - 1) * disc).sum(); ndcg += (dcg / idcg) if idcg > 0 else 0.0
        ws = s[wi]; mk = np.ones(len(s), bool); mk[wi] = False; oth = s[mk]
        auc_num += float((ws > oth).sum() + 0.5 * (ws == oth).sum()); auc_den += len(oth)
        z = s - s.max(); pp = np.exp(z); pp /= pp.sum(); ll += -np.log(max(pp[wi], 1e-12))
        dsg = np.sign(s[:, None] - s[None, :]); djg = np.sign(jy[None, :] - jy[:, None])
        iu = np.triu_indices(len(s), k=1); conc = dsg[iu] * djg[iu]
        pc += float((conc > 0).sum()); pdd += float((conc < 0).sum())
        if with_subset:
            for sc in sub:
                fl = g[sc].values
                for i in range(len(s)):
                    if fl[i] != 1: continue
                    for j in range(len(s)):
                        if i == j: continue
                        c = np.sign(s[i] - s[j]) * np.sign(jy[j] - jy[i])
                        if c > 0: sub[sc]["pc"] += 1
                        elif c < 0: sub[sc]["pd"] += 1
                    sub[sc]["hd"] += 1
                    if (i in top3p) and (jy[i] <= 3): sub[sc]["hn"] += 1
    out = {"n_races": n, "ndcg5": round(ndcg / n, 5), "hon_top3": round(h3 / n, 5), "hon_top1": round(h1 / n, 5),
           "winner_in_top5": round(w5 / n, 5), "auc_win": round(auc_num / auc_den, 5) if auc_den else 0.0,
           "logloss_win": round(ll / n, 5), "pairwise_acc": round(pc / (pc + pdd), 5) if (pc + pdd) else 0.0}
    if with_subset:
        out["subsets"] = {sc: {"pairwise_acc": round(d["pc"] / (d["pc"] + d["pd"]), 5) if (d["pc"] + d["pd"]) else 0.0,
                               "fukusho_hit": round(d["hn"] / d["hd"], 5) if d["hd"] else 0.0, "n_horses": d["hd"]}
                          for sc, d in sub.items()}
    return out


def group_gain(gain):
    return {gn: {"sum_gain": round(float(gain[[c for c in gain.index if fn(c)]].sum()), 1),
                 "n": len([c for c in gain.index if fn(c)])} for gn, fn in HELP_GROUPS.items()}


def main():
    T0 = time.time(); OUT_JSON.parent.mkdir(exist_ok=True)
    df = load_master(); win_pay = load_winner_pay()
    encs = fit_encoders(df[df["split"] == "train"])
    df = apply_encoders(df, encs)                 # ★全体を正しくエンコード
    base_feats = [c for c in df.columns if c not in LEAK_COLS and c not in EXTRA]
    log(f"  base_feats={len(base_feats)}  (正エンコード)")
    cov = {"ped_emb": round(df.loc[df.split == "test", "ped_emb_00"].notna().mean() * 100, 1),
           "elo": round(df.loc[df.split == "test", "elo_T1M1_horse"].notna().mean() * 100, 1),
           "lv_v1": round(df.loc[df.split == "test", "race_level_prev_W3"].notna().mean() * 100, 1),
           "lv_v2": round(df.loc[df.split == "test", "race_level_str_W3"].notna().mean() * 100, 1)}
    results = {"protocol": "正CAT_COLSエンコード/特徴parquet流用/train≤2022→valid2023→test2024-25/市場非経由",
               "repro_target": REPRO, "coverage_test": cov,
               "verdict_rule": "全体or初物で ΔAUC≥+0.008 & Δpairwise≥+0.005 が一貫で本物の増分", "arms": {}}
    tr = df[df["split"] == "train"]; vl = df[df["split"] == "valid"]
    gains = {}
    for name, extra in ARMS_DEF.items():
        feats = base_feats + extra; t0 = time.time()
        model = train(tr, vl, feats, win_pay)
        col = f"_s_{name}"; Xall = df[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
        df[col] = model.predict(Xall)
        m = metrics(df, col, with_subset=(name in PED_ARMS or name == "baseline"))
        gains[name] = pd.Series(model.feature_importance("gain"), index=feats)
        results["arms"][name] = {"n_feats": len(feats), "best_iter": model.best_iteration,
                                 "metrics": m, "group_gain": group_gain(gains[name])}
        extra_ss = ""
        if "subsets" in m:
            extra_ss = " | 初ダpair=%.4f" % m["subsets"]["first_dirt"]["pairwise_acc"]
        log(f"[{name:9s}] {time.time()-t0:.0f}s it={model.best_iteration} AUC={m['auc_win']} "
            f"pair={m['pairwise_acc']} ◎top3={m['hon_top3']*100:.2f}%{extra_ss}")

    # baseline 再現ゲート
    A = results["arms"]["baseline"]["metrics"]
    repro_ok = (abs(A["hon_top3"] - REPRO["hon_top3"]) < 0.002 and abs(A["auc_win"] - REPRO["auc_win"]) < 0.002
                and abs(A["pairwise_acc"] - REPRO["pairwise_acc"]) < 0.002)
    results["baseline_reproduced"] = bool(repro_ok)
    results["pedlabel_gain_baseline"] = results["arms"]["baseline"]["group_gain"]["pedlabel"]

    deltas = {}; any_overall = False; any_first = False
    for name in ARMS_DEF:
        if name == "baseline": continue
        m = results["arms"][name]["metrics"]
        d = {"dAUC": round(m["auc_win"] - A["auc_win"], 5), "dpairwise": round(m["pairwise_acc"] - A["pairwise_acc"], 5),
             "dlogloss": round(m["logloss_win"] - A["logloss_win"], 5), "dtop3": round(m["hon_top3"] - A["hon_top3"], 5)}
        d["WIN_overall"] = bool(d["dAUC"] >= 0.008 and d["dpairwise"] >= 0.005)
        if "subsets" in m:
            d["subsets"] = {sc: round(m["subsets"][sc]["pairwise_acc"] - A["subsets"][sc]["pairwise_acc"], 5)
                            for sc in m["subsets"]}
            d["WIN_first"] = bool(any(v >= 0.01 for v in d["subsets"].values()))
            any_first = any_first or d["WIN_first"]
        any_overall = any_overall or d["WIN_overall"]
        deltas[name] = d
    results["deltas_vs_baseline"] = deltas

    if not repro_ok:
        verdict = f"REPRO_FAIL: baseline が目標未再現 (top3={A['hon_top3']}, auc={A['auc_win']}, pair={A['pairwise_acc']})。エンコードにまだ問題。"
    elif any_overall or any_first:
        winners = [n for n, d in deltas.items() if d.get("WIN_overall") or d.get("WIN_first")]
        verdict = f"INCREMENT_FOUND: {winners} が事前基準を満たす。正baselineで初めて効いた=前回バグで見落とし。要精緻化。"
    else:
        pl = results["pedlabel_gain_baseline"]
        verdict = (f"REDUNDANT_CONFIRMED: 正エンコードbaselineでも全arm ΔAUC<0.008 or Δpair<0.005。"
                   f"能力解明軸(レベル/ELO/血統emb)は冗長で確定的に閉じる。"
                   f"※血統label gain(正エンコード)={pl['sum_gain']} (前回バグでは0)。")
    results["verdict"] = verdict

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log("\n" + "=" * 86)
    log(f"baseline: AUC={A['auc_win']} pair={A['pairwise_acc']} ◎top3={A['hon_top3']*100:.2f}%  再現={repro_ok}")
    log(f"血統label gain(正エンコード)={results['pedlabel_gain_baseline']['sum_gain']} (前回バグ=0)")
    log(f"{'arm':10s}{'ΔAUC':>9s}{'Δpair':>9s}{'Δtop3pt':>9s}  WIN(全/初)")
    for name, d in deltas.items():
        wf = d.get("WIN_first", "-")
        log(f"{name:10s}{d['dAUC']:>+9.5f}{d['dpairwise']:>+9.5f}{d['dtop3']*100:>+8.2f}  {d['WIN_overall']}/{wf}")
    log(f"\n判定: {verdict}")
    log(f"[saved] {OUT_JSON}  ({time.time()-T0:.0f}s)")


if __name__ == "__main__":
    main()
