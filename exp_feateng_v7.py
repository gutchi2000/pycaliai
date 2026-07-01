# -*- coding: utf-8 -*-
"""
exp_feateng_v7.py — 再・特徴量エンジニアリング（フルv6本番スタックでの直交特徴アブレーション）
===========================================================================================
背景: 2026-06-16 の全FEサーベイで判明したこと
  - 予測層はほぼ収束。死因はほぼ全部「kako5 背骨との冗長」(高importance順位なのに正味リフト0)。
  - 唯一ベンチに座る本物 = 物理生存者 kl_lscap/kl_surge/ou_informed。ただし検証は physics_eval.py の
    37特徴・2クラス分類器どまりで、★フル120特徴の v6 LambdaRank 本番スタックでは未検証★。
  - 生き残る特徴は「直交 by construction」のものだけ。→ 新規も直交を狙い撃ちする。

本スクリプトがやること:
  1. exp_recheck.py と同一の v6 再現ハーネス(V6 params/市場重み/LambdaRank/分割不変)で baseline 再現。
  2. 物理 parquet(keller/ou) をフルスタックにマージし、生存者を本番土俵で再検証。
  3. 新規・手作り直交特徴を engineer_new_feats() で生成(全て leak-safe):
       - クラス変動: class_ord / prev_class_ord(馬ごと自己結合 shift) / class_delta / class_drop / class_up
       - 条件変化:   dist_change / surface_switch / going_delta / weight_change / field_change
     ※全て前走列 or 厳密に過去の自己レースのみ参照。当日結果/未来不使用。
  4. 事前固定の判定: 全体 ΔAUC≥+0.008 & Δpairwise≥+0.005 を満たすアームのみ「本物の増分」。
  5. 各新規特徴の as-of リーク健全性を軽くアサート(prev_class は厳密に過去日付)。

出力: reports/exp_feateng_v7.json
実行: PYTHONUTF8=1 python exp_feateng_v7.py
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
KELLER_PQ = BASE / "data/keller_pace_feats.parquet"
OU_PQ = BASE / "data/odds_ou_feats.parquet"
OUT_JSON = BASE / "reports/exp_feateng_v7.json"
SEED = 42
COL_RID = "レースID(新/馬番無)"; COL_JYUN = "着順"; COL_BAN = "馬番"; COL_PED = "血統登録番号"
COL_DATE = "日付"
LEAK_COLS = {"着順", "fukusho_flag", "roi_target", "レースID(新)", "レースID(新/馬番無)",
             "馬名", "レース名", "発走時刻", "date_dt", "日付", "血統登録番号", "split"}
CAT_COLS = ["場所", "芝・ダ", "コース区分", "芝(内・外)", "馬場状態", "天気", "クラス名",
            "種牡馬", "父タイプ名", "母父馬", "母父タイプ名", "毛色", "馬主(最新/仮想)", "生産者",
            "騎手コード", "調教師コード", "年齢限定", "限定", "性別限定", "指定条件", "重量種別",
            "性別", "ブリンカー", "前走場所", "前芝・ダ", "前走馬場状態", "前走競走種別", "前好走"]
# exp_recheck.py と同一の v6 ハイパラ（baseline 厳密再現のため）
V6 = dict(num_leaves=59, max_depth=12, min_data_in_leaf=197, learning_rate=0.05083158501542684,
          feature_fraction=0.876098829427658, bagging_fraction=0.7031707810405968,
          lambda_l1=0.0011077902520957399, lambda_l2=7.537933313450104)
V6_BEST_ITER = 469; V6_ALPHA = 0.03083978412534253
REPRO = {"hon_top3": 0.61812, "auc_win": 0.78962, "pairwise_acc": 0.68419}

# ---- 物理生存者（サーベイ runway #1） ----
KELLER_ALL = ["kl_reserve", "kl_burn", "kl_draft", "kl_lscap",
              "kl_pace_fit", "kl_surge", "kl_solo_front", "kl_pace_hardness"]
OU_KEEP = ["ou_net", "ou_s2n", "ou_informed", "ou_termvel", "ou_n"]
PHYS_SURV = ["kl_lscap", "kl_surge", "ou_informed"]          # 3生存者(part_corr 0.024/0.024/0.033)

# ---- 新規・手作り直交特徴 ----
NEW_CLASS = ["class_ord", "prev_class_ord", "class_delta", "class_drop", "class_up"]
NEW_CTX = ["dist_change", "surface_switch", "going_delta", "weight_change", "field_change"]
NEW_ALL = NEW_CLASS + NEW_CTX

# 学習特徴から除く補助列
EXTRA = set(KELLER_ALL) | set(OU_KEEP) | set(NEW_ALL) | {
    "rid16", "ban", "rid_s", "_rid", "_ban", "_pedk", "_dd", "label", "ou_has"}

ARMS_DEF = {
    "baseline": [],
    "phys_surv": PHYS_SURV,
    "phys_keller": KELLER_ALL,
    "phys_ou": OU_KEEP,
    "phys_all": KELLER_ALL + OU_KEEP,
    "new_class": NEW_CLASS,
    "new_ctx": NEW_CTX,
    "new_all": NEW_ALL,
    "combined": PHYS_SURV + NEW_ALL,
}

HELP_GROUPS = {
    "kako5": lambda c: c.startswith("kako5_"), "hosei": lambda c: "hosei" in c,
    "zensou": lambda c: c.startswith("前"), "trnH": lambda c: c.startswith(("trnH_", "trnW_")),
    "keller": lambda c: c.startswith("kl_"), "ou": lambda c: c.startswith("ou_"),
    "newclass": lambda c: c in NEW_CLASS, "newctx": lambda c: c in NEW_CTX,
}

# JRA 平地クラス序数（障害は NaN）
CLASS_MAP = {
    "新馬": 0, "未勝利": 0,
    "500万": 1, "1勝": 1,
    "1000万": 2, "2勝": 2,
    "1600万": 3, "3勝": 3,
    "ｵｰﾌﾟﾝ": 4, "オープン": 4, "OP(L)": 4, "OP": 4,
    "重賞": 5, "Ｇ３": 5,
    "Ｇ２": 6, "Ｇ１": 7,
}
GOING_MAP = {"良": 0, "稍": 1, "稍重": 1, "重": 2, "不": 3, "不良": 3}


def log(m): print(m, flush=True)


def load_winner_pay():
    df = pd.read_csv(KEKKA_CSV, encoding="cp932", low_memory=False)
    df.columns = ["rid_horse", "ban", "ped", "jyun", "tansho", "fukusho", "wakuren",
                  "umaren", "umatan", "sanrenpuku", "sanrentan"]
    df["rid_s"] = df["rid_horse"].astype(str).str[:16]
    df["jyun"] = pd.to_numeric(df["jyun"], errors="coerce")
    win = df[df["jyun"] == 1].drop_duplicates("rid_s")
    return dict(zip(win["rid_s"].values, pd.to_numeric(win["tansho"], errors="coerce").values))


def engineer_new_feats(df: pd.DataFrame) -> pd.DataFrame:
    """★新規直交特徴（エンコード前・生カテゴリ使用）。全て leak-safe。"""
    d = df
    # --- クラス序数（現在） ---
    d["class_ord"] = d["クラス名"].astype(str).str.strip().map(CLASS_MAP).astype("float")
    # --- 前走クラス序数: 馬ごと日付ソートで直前レースの class_ord を取る（厳密に過去） ---
    dd = pd.to_numeric(d[COL_DATE], errors="coerce")
    tmp = d[[COL_PED]].copy()
    tmp["_dd"] = dd
    tmp["_rid"] = d[COL_RID].astype(str)
    tmp["_cls"] = d["class_ord"].values
    tmp = tmp.sort_values([COL_PED, "_dd", "_rid"], kind="mergesort")
    tmp["prev_class_ord"] = tmp.groupby(COL_PED, sort=False)["_cls"].shift(1)
    tmp["_prev_dd"] = tmp.groupby(COL_PED, sort=False)["_dd"].shift(1)
    d["prev_class_ord"] = tmp["prev_class_ord"].reindex(d.index)
    # as-of リーク健全性: 直前日付は厳密に過去(<= 当日)。当日同馬重複は実質皆無だが <= で許容。
    leak_bad = int((tmp["_prev_dd"] > tmp["_dd"]).sum())
    d.attrs["class_leak_violations"] = leak_bad
    # --- クラス変動 ---
    d["class_delta"] = d["class_ord"] - d["prev_class_ord"]            # >0=格上挑戦, <0=格下げ
    d["class_drop"] = (d["class_delta"] < 0).astype("float")
    d["class_up"] = (d["class_delta"] > 0).astype("float")
    d.loc[d["class_delta"].isna(), ["class_drop", "class_up"]] = np.nan
    # --- 条件変化（全て前走列から、leak-safe） ---
    dist = pd.to_numeric(d.get("距離"), errors="coerce")
    pdist = pd.to_numeric(d.get("前距離"), errors="coerce")
    d["dist_change"] = dist - pdist
    sd = d["芝・ダ"].astype(str).str.strip()
    psd = d["前芝・ダ"].astype(str).str.strip()
    d["surface_switch"] = ((sd != psd) & psd.ne("nan") & psd.ne("")).astype("float")
    g = d["馬場状態"].astype(str).str.strip().map(GOING_MAP)
    pg = d["前走馬場状態"].astype(str).str.strip().map(GOING_MAP)
    d["going_delta"] = g - pg
    w = pd.to_numeric(d.get("斤量"), errors="coerce")
    pw = pd.to_numeric(d.get("前走斤量"), errors="coerce")
    d["weight_change"] = w - pw
    f = pd.to_numeric(d.get("出走頭数"), errors="coerce")
    pf = pd.to_numeric(d.get("前走出走頭数"), errors="coerce")
    d["field_change"] = f - pf
    return d


def _rid16(s):
    return s.astype(str).str.replace(r"\D", "", regex=True).str[:16]


def load_master():
    log(f"[load] {MASTER_CSV.name}")
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df = df.dropna(subset=[COL_JYUN, COL_RID, "split"]).copy().reset_index(drop=True)
    df["label"] = np.clip(6 - df[COL_JYUN].astype(int), 0, 5).astype(int)
    df["rid_s"] = df[COL_RID].astype(str).str[:16]
    df["rid16"] = _rid16(df[COL_RID])
    df["ban"] = pd.to_numeric(df[COL_BAN], errors="coerce").fillna(0).astype(int)
    # 新規特徴（エンコード前）
    df = engineer_new_feats(df)
    # 物理 parquet マージ
    kl = pd.read_parquet(KELLER_PQ).drop(columns=["split"], errors="ignore")
    kl["rid16"] = kl["rid16"].astype(str); kl["ban"] = pd.to_numeric(kl["ban"], errors="coerce").fillna(0).astype(int)
    df = df.merge(kl[["rid16", "ban"] + KELLER_ALL], on=["rid16", "ban"], how="left")
    ou = pd.read_parquet(OU_PQ)
    ou["rid16"] = ou["rid16"].astype(str); ou["ban"] = pd.to_numeric(ou["ban"], errors="coerce").fillna(0).astype(int)
    df = df.merge(ou[["rid16", "ban"] + OU_KEEP], on=["rid16", "ban"], how="left")
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
        wt = d["rid_s"].map(win_pay).fillna(100.0).values
        w = (1.0 + alpha * np.log1p(wt / 100.0)).astype(float)
    else:
        w = np.ones(len(d), float)
    return lgb.Dataset(X, label=y, group=g, weight=w, free_raw_data=False)


def train(tr, vl, feats, win_pay):
    params = dict(objective="lambdarank", lambdarank_truncation_level=5, metric="ndcg", eval_at=[5],
                  **V6, bagging_freq=5, verbose=-1, n_jobs=-1, seed=SEED, deterministic=True,
                  force_col_wise=True, feature_pre_filter=False)
    return lgb.train(params, make_ds(tr, feats, win_pay, V6_ALPHA),
                     num_boost_round=int(V6_BEST_ITER * 1.1),
                     valid_sets=[make_ds(vl, feats, win_pay, 0.0)],
                     callbacks=[lgb.early_stopping(100, verbose=False)])


def metrics(df, scorecol):
    te = df[df["split"] == "test"]
    n = h1 = h3 = w5 = 0; ndcg = 0.0; auc_num = auc_den = 0.0; ll = 0.0; pc = pdd = 0.0
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
        io2 = np.argsort(-rel)[:k]; idcg = ((2 ** rel[io2] - 1) * disc).sum()
        ndcg += (dcg / idcg) if idcg > 0 else 0.0
        ws = s[wi]; mk = np.ones(len(s), bool); mk[wi] = False; oth = s[mk]
        auc_num += float((ws > oth).sum() + 0.5 * (ws == oth).sum()); auc_den += len(oth)
        z = s - s.max(); pp = np.exp(z); pp /= pp.sum(); ll += -np.log(max(pp[wi], 1e-12))
        dsg = np.sign(s[:, None] - s[None, :]); djg = np.sign(jy[None, :] - jy[:, None])
        iu = np.triu_indices(len(s), k=1); conc = dsg[iu] * djg[iu]
        pc += float((conc > 0).sum()); pdd += float((conc < 0).sum())
    return {"n_races": n, "ndcg5": round(ndcg / n, 5), "hon_top3": round(h3 / n, 5),
            "hon_top1": round(h1 / n, 5), "winner_in_top5": round(w5 / n, 5),
            "auc_win": round(auc_num / auc_den, 5) if auc_den else 0.0,
            "logloss_win": round(ll / n, 5),
            "pairwise_acc": round(pc / (pc + pdd), 5) if (pc + pdd) else 0.0}


def group_gain(gain):
    return {gn: {"sum_gain": round(float(gain[[c for c in gain.index if fn(c)]].sum()), 1),
                 "n": len([c for c in gain.index if fn(c)])} for gn, fn in HELP_GROUPS.items()}


def main():
    T0 = time.time(); OUT_JSON.parent.mkdir(exist_ok=True)
    df = load_master(); win_pay = load_winner_pay()
    class_leak = int(df.attrs.get("class_leak_violations", -1))
    encs = fit_encoders(df[df["split"] == "train"])
    df = apply_encoders(df, encs)
    base_feats = [c for c in df.columns if c not in LEAK_COLS and c not in EXTRA]
    log(f"  base_feats={len(base_feats)}  prev_class as-of違反={class_leak}")

    # カバレッジ（test 非欠損率）
    cov = {}
    for c in PHYS_SURV + NEW_ALL:
        cov[c] = round(df.loc[df.split == "test", c].notna().mean() * 100, 1)

    results = {"protocol": "v6再現/keller+ou parquet流用+新規直交特徴/train≤2022→valid2023→test2024-25/市場非経由",
               "repro_target": REPRO, "class_prev_leak_violations": class_leak,
               "coverage_test_pct": cov,
               "verdict_rule": "全体 ΔAUC≥+0.008 & Δpairwise≥+0.005 で本物の増分", "arms": {}}
    tr = df[df["split"] == "train"]; vl = df[df["split"] == "valid"]
    gains = {}
    for name, extra in ARMS_DEF.items():
        feats = base_feats + extra; t0 = time.time()
        model = train(tr, vl, feats, win_pay)
        col = f"_s_{name}"
        Xall = df[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
        df[col] = model.predict(Xall)
        m = metrics(df, col)
        gains[name] = pd.Series(model.feature_importance("gain"), index=feats)
        results["arms"][name] = {"n_feats": len(feats), "best_iter": model.best_iteration,
                                 "metrics": m, "group_gain": group_gain(gains[name])}
        log(f"[{name:11s}] {time.time()-t0:.0f}s it={model.best_iteration} AUC={m['auc_win']} "
            f"pair={m['pairwise_acc']} ◎top3={m['hon_top3']*100:.2f}%")

    A = results["arms"]["baseline"]["metrics"]
    repro_ok = (abs(A["hon_top3"] - REPRO["hon_top3"]) < 0.003 and abs(A["auc_win"] - REPRO["auc_win"]) < 0.003
                and abs(A["pairwise_acc"] - REPRO["pairwise_acc"]) < 0.003)
    results["baseline_reproduced"] = bool(repro_ok)

    deltas = {}; any_win = False
    for name in ARMS_DEF:
        if name == "baseline": continue
        m = results["arms"][name]["metrics"]
        d = {"dAUC": round(m["auc_win"] - A["auc_win"], 5), "dpairwise": round(m["pairwise_acc"] - A["pairwise_acc"], 5),
             "dlogloss": round(m["logloss_win"] - A["logloss_win"], 5), "dtop3pt": round((m["hon_top3"] - A["hon_top3"]) * 100, 3),
             "dndcg5": round(m["ndcg5"] - A["ndcg5"], 5)}
        d["WIN"] = bool(d["dAUC"] >= 0.008 and d["dpairwise"] >= 0.005)
        any_win = any_win or d["WIN"]
        # その arm の新規/物理特徴 gain（食い合いの可視化）
        gg = results["arms"][name]["group_gain"]
        d["added_gain"] = {k: gg[k]["sum_gain"] for k in ("keller", "ou", "newclass", "newctx")
                           if gg[k]["n"] > 0 and gg[k]["sum_gain"] > 0}
        deltas[name] = d
    results["deltas_vs_baseline"] = deltas

    if not repro_ok:
        verdict = (f"REPRO_FAIL: baseline 未再現 (top3={A['hon_top3']} auc={A['auc_win']} pair={A['pairwise_acc']})。")
    elif class_leak > 0:
        verdict = f"LEAK_GUARD: prev_class as-of違反={class_leak} > 0。新規クラス特徴は無効。要修正。"
    elif any_win:
        winners = [n for n, d in deltas.items() if d["WIN"]]
        verdict = f"INCREMENT_FOUND: {winners} が事前ゲート(ΔAUC≥0.008&Δpair≥0.005)を満たす。本番土俵で本物。要精緻化→v7候補。"
    else:
        # 最大ΔAUCのarmを参考表示
        best = max(deltas.items(), key=lambda kv: kv[1]["dAUC"])
        verdict = (f"NO_INCREMENT: フルv6スタックでは全arm ΔAUC<0.008 or Δpair<0.005。"
                   f"物理生存者/新規直交特徴も120特徴の前では冗長 or 微小。"
                   f"最大={best[0]} ΔAUC={best[1]['dAUC']:+.5f} Δpair={best[1]['dpairwise']:+.5f}。")
    results["verdict"] = verdict

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log("\n" + "=" * 88)
    log(f"baseline: AUC={A['auc_win']} pair={A['pairwise_acc']} ◎top3={A['hon_top3']*100:.2f}%  再現={repro_ok}")
    log(f"{'arm':12s}{'ΔAUC':>9s}{'Δpair':>9s}{'Δtop3pt':>9s}{'Δndcg5':>9s}  WIN  added_gain")
    for name, d in deltas.items():
        log(f"{name:12s}{d['dAUC']:>+9.5f}{d['dpairwise']:>+9.5f}{d['dtop3pt']:>+9.3f}{d['dndcg5']:>+9.5f}  "
            f"{str(d['WIN']):5s} {d['added_gain']}")
    log(f"\n判定: {verdict}")
    log(f"[saved] {OUT_JSON}  ({time.time()-T0:.0f}s)")


if __name__ == "__main__":
    main()
