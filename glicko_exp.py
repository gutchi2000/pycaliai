"""
glicko_exp.py — Glicko-2 μ/RD の A/B (正エンコード・市場非経由・不確実性サブセット)
================================================================================
arm: A(baseline) / +μ(g2_mu,g2_cons) / +RD(g2_rd,g2_rd_rel ★本命) / +μ+RD。
★σ/RD(不確実性)が kako5・キャリア戦数・ELo に対し増分を持つか。μは ELo 冗長の再確認(対照)。
多指標 AUC/log-loss/pairwise/◎top3 + importance(g2_rd 順位/kako5_race_count を食うか)。
★サブセット別建て: 初ダート/初距離/休み明け初戦/キャリア浅(kako5_race_count<=1) で RD が効くか。
baseline=正v6(61.81%/0.78962/0.68419 再現確認)。TrueSkill は依存不在で省略(Glicko-2 RDで代表)。
判定(事前固定): σ系が全体orサブセットで ΔAUC≥+0.008 & Δpair≥+0.005 一貫 → 不確実性は新情報。
出力: reports/glicko_exp.json  実行: PYTHONUTF8=1 python glicko_exp.py
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
GL_PQ = BASE / "data/glicko_feats.parquet"
OUT_JSON = BASE / "reports/glicko_exp.json"
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
G2_MU = ["g2_mu", "g2_cons"]; G2_RD = ["g2_rd", "g2_rd_rel"]
ALL_G2 = ["g2_mu", "g2_cons", "g2_rd", "g2_rd_rel"]
ARMS_DEF = {"baseline": [], "mu": G2_MU, "rd": G2_RD, "mu_rd": G2_MU + G2_RD}
SUBSETS = ["first_dirt", "first_dist", "rest1", "shallow"]
EXTRA = set(ALL_G2) | set(SUBSETS) | {"rid_s", "_rid", "_ban", "label"}
HELP_GROUPS = {"kako5": lambda c: c.startswith("kako5_"), "career": lambda c: c == "kako5_race_count",
               "hosei": lambda c: "hosei" in c, "zensou": lambda c: c.startswith("前"),
               "g2_rd": lambda c: c in ("g2_rd", "g2_rd_rel"), "g2_mu": lambda c: c in ("g2_mu", "g2_cons")}


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


def add_first(df):
    d = df.copy(); d["_dd"] = pd.to_numeric(d["日付"], errors="coerce")
    d = d.sort_values([COL_PED, "_dd", COL_RID]).reset_index()
    fd = np.zeros(len(d), int); fi = np.zeros(len(d), int)
    cur = None; std = set(); sb = set()
    td = d["芝・ダ"].astype(str).values; bd = d["距離"].apply(dist_band).values; pe = d[COL_PED].values
    for i in range(len(d)):
        if pe[i] != cur: cur = pe[i]; std, sb = set(), set()
        if td[i] == "ダ" and "ダ" not in std: fd[i] = 1
        if bd[i] not in sb: fi[i] = 1
        std.add(td[i]); sb.add(bd[i])
    d["first_dirt"] = fd; d["first_dist"] = fi
    d = d.set_index("index").sort_index()
    df["first_dirt"] = d["first_dirt"]; df["first_dist"] = d["first_dist"]
    return df


def load_master():
    log(f"[load] {MASTER_CSV.name}")
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df = df.dropna(subset=[COL_JYUN, COL_RID, "split"]).copy()
    df["label"] = np.clip(6 - df[COL_JYUN].astype(int), 0, 5).astype(int)
    df["rid_s"] = df[COL_RID].astype(str).str[:16]; df["_rid"] = df[COL_RID].astype(str)
    df["_ban"] = pd.to_numeric(df[COL_BAN], errors="coerce").fillna(0).astype(int)
    df["rest1"] = (pd.to_numeric(df["休み明け～戦目"], errors="coerce") == 1).astype(int)
    df["shallow"] = (pd.to_numeric(df["kako5_race_count"], errors="coerce") <= 1).astype(int)
    df = add_first(df)
    gl = pd.read_parquet(GL_PQ); gl["rid16"] = gl["rid16"].astype(str)
    gl["ban"] = pd.to_numeric(gl["ban"], errors="coerce").fillna(0).astype(int)
    df = df.merge(gl, left_on=["_rid", "_ban"], right_on=["rid16", "ban"], how="left")
    df = df.drop(columns=[c for c in ["rid16", "ban"] if c in df.columns], errors="ignore")
    # g2_rd_rel: レース内 z
    g = df.groupby(COL_RID)["g2_rd"]
    df["g2_rd_rel"] = ((df["g2_rd"] - g.transform("mean")) / g.transform("std").replace(0, np.nan)).fillna(0.0)
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


def make_ds(d, feats, win_pay, alpha):
    d = d.sort_values(COL_RID).reset_index(drop=True)
    X = d[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    y = d["label"].values.astype(int); g = np.array([len(list(gr)) for _, gr in groupby(d[COL_RID])])
    if alpha > 0:
        wt = d["rid_s"].map(win_pay).fillna(100.0).values; w = (1.0 + alpha * np.log1p(wt / 100.0)).astype(float)
    else: w = np.ones(len(d), float)
    return lgb.Dataset(X, label=y, group=g, weight=w, free_raw_data=False)


def train(tr, vl, feats, win_pay):
    params = dict(objective="lambdarank", lambdarank_truncation_level=5, metric="ndcg", eval_at=[5], **V6,
                  bagging_freq=5, verbose=-1, n_jobs=-1, seed=SEED, deterministic=True, force_col_wise=True, feature_pre_filter=False)
    return lgb.train(params, make_ds(tr, feats, win_pay, V6_ALPHA), num_boost_round=int(V6_BEST_ITER * 1.1),
                     valid_sets=[make_ds(vl, feats, win_pay, 0.0)], callbacks=[lgb.early_stopping(100, verbose=False)])


def metrics(df, scorecol):
    te = df[df["split"] == "test"]
    n = h1 = h3 = w5 = 0; ndcg = an = ad = ll = pc = pdd = 0.0
    sub = {k: {"pc": 0.0, "pd": 0.0} for k in SUBSETS}
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
        for sc in SUBSETS:
            fl = g[sc].values
            for i in range(len(s)):
                if fl[i] != 1: continue
                for j in range(len(s)):
                    if i == j: continue
                    c = np.sign(s[i] - s[j]) * np.sign(jy[j] - jy[i])
                    if c > 0: sub[sc]["pc"] += 1
                    elif c < 0: sub[sc]["pd"] += 1
    out = {"n_races": n, "ndcg5": round(ndcg / n, 5), "hon_top3": round(h3 / n, 5), "hon_top1": round(h1 / n, 5),
           "auc_win": round(an / ad, 5) if ad else 0.0, "logloss_win": round(ll / n, 5),
           "pairwise_acc": round(pc / (pc + pdd), 5) if (pc + pdd) else 0.0,
           "subset_pairwise": {sc: round(d["pc"] / (d["pc"] + d["pd"]), 5) if (d["pc"] + d["pd"]) else 0.0 for sc, d in sub.items()}}
    return out


def group_gain(gain):
    return {gn: {"sum_gain": round(float(gain[[c for c in gain.index if fn(c)]].sum()), 1), "n": len([c for c in gain.index if fn(c)])} for gn, fn in HELP_GROUPS.items()}


def main():
    T0 = time.time(); OUT_JSON.parent.mkdir(exist_ok=True)
    df = load_master(); win_pay = load_winner_pay()
    encs = fit_encoders(df[df["split"] == "train"]); df = apply_encoders(df, encs)
    base_feats = [c for c in df.columns if c not in LEAK_COLS and c not in EXTRA]
    log(f"  base_feats={len(base_feats)}")
    tr = df[df["split"] == "train"]; vl = df[df["split"] == "valid"]
    results = {"protocol": "正エンコード/Glicko-2 μ・RD/train≤2022→valid2023→test2024-25/市場非経由",
               "note": "TrueSkill省略(依存不在)・Glicko-2 RDで不確実性代表", "arms": {}}
    gains = {}
    for name, extra in ARMS_DEF.items():
        feats = base_feats + extra; t0 = time.time()
        model = train(tr, vl, feats, win_pay)
        col = f"_s_{name}"; df[col] = model.predict(df[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values)
        m = metrics(df, col); gains[name] = pd.Series(model.feature_importance("gain"), index=feats)
        results["arms"][name] = {"n_feats": len(feats), "best_iter": model.best_iteration, "metrics": m,
                                 "group_gain": group_gain(gains[name])}
        ss = m["subset_pairwise"]
        log(f"[{name:8s}] {time.time()-t0:.0f}s it={model.best_iteration} AUC={m['auc_win']} pair={m['pairwise_acc']} "
            f"◎top3={m['hon_top3']*100:.2f}% | 休明pair={ss['rest1']} 浅pair={ss['shallow']} 初ダpair={ss['first_dirt']}")

    A = results["arms"]["baseline"]["metrics"]
    repro = all(abs(A[k] - REPRO[k]) < 0.002 for k in REPRO)
    results["baseline_reproduced"] = bool(repro)
    # g2_rd importance (rd arm)
    grd = gains["rd"]; rkd = grd.rank(ascending=False).astype(int)
    results["g2_rd_importance"] = {c: {"gain": round(float(grd[c]), 1), "rank": int(rkd[c]), "of": len(grd)} for c in G2_RD}
    deltas = {}; any_win = False
    for name in ["mu", "rd", "mu_rd"]:
        m = results["arms"][name]["metrics"]
        d = {"dAUC": round(m["auc_win"] - A["auc_win"], 5), "dpair": round(m["pairwise_acc"] - A["pairwise_acc"], 5),
             "dtop3": round(m["hon_top3"] - A["hon_top3"], 5),
             "subset_dpair": {sc: round(m["subset_pairwise"][sc] - A["subset_pairwise"][sc], 5) for sc in SUBSETS}}
        d["WIN_overall"] = bool(d["dAUC"] >= 0.008 and d["dpair"] >= 0.005)
        d["WIN_subset"] = bool(any(v >= 0.01 for v in d["subset_dpair"].values()))
        if name in ("rd", "mu_rd") and (d["WIN_overall"] or d["WIN_subset"]): any_win = True
        deltas[name] = d
    results["deltas_vs_baseline"] = deltas
    if not repro:
        verdict = f"REPRO_FAIL: baseline未再現 {A}"
    elif any_win:
        verdict = "UNCERTAINTY_INCREMENT: σ/RD系が全体orサブセットで事前基準達成。不確実性は新情報→深掘り。"
    elif deltas["mu"]["WIN_overall"]:
        verdict = "MU_ONLY(=ELo冗長再確認): μで動くがσ無効。ELo再走に過ぎず、不確実性は増分なし。"
    else:
        cg = results["arms"]["mu_rd"]["group_gain"]
        verdict = (f"REDUNDANT: σ/RD系も全体・サブセット全て未達。g2_rd importance={results['g2_rd_importance']}。"
                   "RDはキャリア戦数(kako5_race_count)の言い換えで冗長＝不確実性軸も閉じる。")
    results["verdict"] = verdict
    json.dump(results, open(OUT_JSON, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    log("\n" + "=" * 88)
    log(f"baseline再現={repro} AUC={A['auc_win']} pair={A['pairwise_acc']} ◎top3={A['hon_top3']*100:.2f}%")
    log(f"g2_rd importance: {results['g2_rd_importance']}")
    log(f"{'arm':7s}{'ΔAUC':>9s}{'Δpair':>9s}{'Δtop3':>8s} 全/サブ  | サブセットΔpair(初ダ/初距/休明/浅)")
    for name, d in deltas.items():
        sd = d["subset_dpair"]
        log(f"{name:7s}{d['dAUC']:>+9.5f}{d['dpair']:>+9.5f}{d['dtop3']*100:>+7.2f} {d['WIN_overall']}/{d['WIN_subset']}  | "
            f"{sd['first_dirt']:+.4f}/{sd['first_dist']:+.4f}/{sd['rest1']:+.4f}/{sd['shallow']:+.4f}")
    log(f"\n判定: {verdict}")
    log(f"[saved] {OUT_JSON} ({time.time()-T0:.0f}s)")


if __name__ == "__main__":
    main()
