# -*- coding: utf-8 -*-
"""
exp_havoc_m1.py — Step 1: M1 2体結合の「波乱読み出し」が havoc_v1 を超えるか
================================================================================
仮説(ユーザー): havoc_v1 は v6 が潰した後の周辺分布スカラー(_entropy/_v6_std/_p_win_*)
しか見ていない。逃げ2頭の共倒れ・潰し合い・脚質クラスタ・枠順相互作用 という「潰す前の
配置(=第2モーメント/分散を生む構造)」は、その周辺分布からは復元不能。
→ M1 2体結合(gate1, 実証済み・墓場唯一の生存者)で配置を joint に戻し、その
   「結合エネルギー」と「ジョイント分布のエントロピー変化 ΔH」を波乱読み出しに使えば、
   havoc_v1 が見れない分散情報を足せるはず。

設計(apples-to-apples):
  - ターゲット: is_havoc = (1着オッズ≥5倍 = 払戻≥500) … havoc_v1 と同一
  - B1(baseline) = havoc_v1 の特徴(12スカラー + レース条件カテゴリ) を再現
  - B2(+M1)      = B1 + M1 由来のレース単位 結合特徴
  - 学習器・ハイパラは両アーム共通(havoc_v1.best_params 固定)=「容量固定・特徴のみ変化」
  - 分割: train≤2022 / valid2023(early stop) / test2024-25
  - β: reports/gate1_corr_calibration.json (train≤2022 MLE, leak-safe) を流用

M1 レース特徴(全て leak-safe: 脚質=前走通過順, 枠=前売り既知, β=train fit):
  m1_corr_abs_mean / m1_corr_mean / m1_corr_min   … 全ペア結合の強さ・向き・最強潰し
  m1_n_bothfront                                  … 両前ペア数(共倒れ構造の量)
  m1_top_corr_mean / m1_top_corr_min / m1_bothfront_top … 上位4頭(本命圏)の配置
  m1_umaren_dH                                    … ★M1補正で馬連joint分布のエントロピーが
                                                     どれだけ動くか(=周辺に無い分散)

判定: TEST で AUC/AP/logloss/ECE/十分位リフト を B1 vs B2 比較。
  決定的診断(exp_pace_interaction 同型): M1特徴が gain を取るのに TEST 改善が無ければ
  「配置を貪欲に使うが汎化ゼロ」が波乱検出でも再確定=隙間は閉じる。
  逆に AUC/AP/リフトが意味ある幅で上がれば、ユーザーの隙間仮説が当たり。

出力: reports/exp_havoc_m1.json   実行: PYTHONUTF8=1 python exp_havoc_m1.py
"""
from __future__ import annotations
import io, json, sys, warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import pl_probs as PL
from corr_features import add_style, PAIR_FEATURE_NAMES

BASE = Path(__file__).parent
MASTER_CSV = BASE / "data/master_v2_20130105-20251228.csv"
KEKKA_CSV = BASE / "data/kekka_20130105-20251228.csv"
V6_PKL = BASE / "models/unified_rank_v6.pkl"
HAVOC_PKL = BASE / "models/havoc_v1.pkl"
BETA_JSON = BASE / "reports/gate1_corr_calibration.json"
OUT_JSON = BASE / "reports/exp_havoc_m1.json"
SEED = 42
COL_RID = "レースID(新/馬番無)"; COL_JYUN = "着順"; COL_BAN = "馬番"; COL_WAKU = "枠番"
HAVOC_PAY_THRESHOLD = 500

BASE_SCALARS = ["_n_horses", "_v6_top1", "_v6_top3_mean", "_v6_top5_mean", "_v6_max",
                "_v6_min", "_v6_std", "_v6_range", "_entropy", "_p_win_1st",
                "_p_win_3rd", "_p_win_top3_sum"]
BASE_CATS = ["場所", "芝・ダ", "距離", "馬場状態", "天気", "クラス名",
             "コース区分", "年齢限定", "限定", "性別限定", "重量種別"]
M1_FEATS = ["m1_corr_abs_mean", "m1_corr_mean", "m1_corr_min", "m1_n_bothfront",
            "m1_top_corr_mean", "m1_top_corr_min", "m1_bothfront_top", "m1_umaren_dH"]


def log(m): print(m, flush=True)


def load_beta():
    b = json.loads(BETA_JSON.read_text(encoding="utf-8"))["beta"]
    return np.array([b[k] for k in PAIR_FEATURE_NAMES], dtype=float)


def load_havoc_labels():
    df = pd.read_csv(KEKKA_CSV, encoding="cp932", low_memory=False)
    df.columns = ["rid_horse", "ban", "ped", "jyun", "tansho", "fukusho",
                  "wakuren", "umaren", "umatan", "sanrenpuku", "sanrentan"]
    df["rid"] = df["rid_horse"].astype(str).str[:16]
    df["jyun"] = pd.to_numeric(df["jyun"], errors="coerce")
    df["tansho"] = pd.to_numeric(df["tansho"], errors="coerce")
    win = df[df["jyun"] == 1].drop_duplicates("rid")
    out = {}
    for r in win.itertuples():
        if pd.isna(r.tansho) or r.tansho <= 0: continue
        out[r.rid] = int(r.tansho >= HAVOC_PAY_THRESHOLD)
    return out


def corr_matrix(style, waku, pace, beta):
    """全ペア (i,j) の log-correction corr[i,j] = β·pair_features をベクトル化。
    style NaN→0.5 (pair_features と同じ中立化)。"""
    n = len(style)
    s = np.where(np.isnan(style), 0.5, style).astype(float)
    si = s[:, None]; sj = s[None, :]
    both_front = ((si < 0.30) & (sj < 0.30)).astype(float)
    both_closer = ((si >= 0.70) & (sj >= 0.70)).astype(float)
    front_closer = ((np.minimum(si, sj) < 0.30) & (np.maximum(si, sj) >= 0.55)).astype(float)
    style_gap = np.abs(si - sj)
    bf_pace = both_front * float(pace)
    w = np.where(np.isnan(waku), 0.0, waku).astype(float)
    draw_gap = np.abs(w[:, None] - w[None, :]) / 8.0
    # β 順: both_front, both_closer, front_closer, style_gap, both_front_x_pace, draw_gap
    corr = (beta[0] * both_front + beta[1] * both_closer + beta[2] * front_closer
            + beta[3] * style_gap + beta[4] * bf_pace + beta[5] * draw_gap)
    np.fill_diagonal(corr, 0.0)
    return corr, both_front


def harville_pair_q0(p_win, iu, ju):
    """周辺勝率 p_win から Harville の馬連(順不同上位2)確率を全ペアで。"""
    pa = p_win[iu]; pb = p_win[ju]
    h = pa * pb / np.clip(1 - pa, 1e-9, None) + pb * pa / np.clip(1 - pb, 1e-9, None)
    s = h.sum()
    return h / s if s > 0 else np.full(len(h), 1.0 / max(len(h), 1))


def m1_race_features(scores, style, waku, pace, beta):
    n = len(scores)
    if n < 3:
        return {k: np.nan for k in M1_FEATS}
    corr, both_front = corr_matrix(style, waku, pace, beta)
    iu, ju = np.triu_indices(n, 1)
    cu = corr[iu, ju]
    # 全ペア結合
    abs_mean = float(np.abs(cu).mean())
    mean = float(cu.mean())
    cmin = float(cu.min())
    n_bf = float(both_front[iu, ju].sum())
    # 上位4頭(本命圏)の配置
    order = np.argsort(-scores)
    top = order[:min(4, n)]
    if len(top) >= 2:
        tg = [(a, b) for a in top for b in top if a < b]
        tc = np.array([corr[a, b] for a, b in tg]) if tg else np.array([0.0])
        top_corr_mean = float(tc.mean()); top_corr_min = float(tc.min())
        bf_top = float(sum(both_front[a, b] for a, b in tg))
    else:
        top_corr_mean = top_corr_min = bf_top = 0.0
    # ★ ΔH: M1 補正で馬連 joint のエントロピーがどれだけ動くか
    w = PL.pl_weights(scores)
    p_win = PL.all_tansho(w)
    q0 = harville_pair_q0(p_win, iu, ju)
    q1 = q0 * np.exp(cu); ssum = q1.sum()
    q1 = q1 / ssum if ssum > 0 else q0
    H0 = -(q0 * np.log(np.clip(q0, 1e-12, 1))).sum()
    H1 = -(q1 * np.log(np.clip(q1, 1e-12, 1))).sum()
    dH = float(H1 - H0)
    return {"m1_corr_abs_mean": abs_mean, "m1_corr_mean": mean, "m1_corr_min": cmin,
            "m1_n_bothfront": n_bf, "m1_top_corr_mean": top_corr_mean,
            "m1_top_corr_min": top_corr_min, "m1_bothfront_top": bf_top, "m1_umaren_dH": dH}


def apply_encoders(df, encs):
    df = df.copy()
    for c, le in encs.items():
        if c not in df.columns: continue
        v = df[c].astype(str).fillna("__NaN__")
        df[c] = v.where(v.isin(set(le.classes_)), "__NaN__")
        df[c] = le.transform(df[c])
    return df


def build_race_df(beta):
    log(f"[load v6] {V6_PKL.name}")
    v6 = joblib.load(V6_PKL)
    vm, vf, ve = v6["model"], v6["feature_cols"], v6["encoders"]
    log(f"[load] {MASTER_CSV.name}")
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df = df.dropna(subset=[COL_JYUN, COL_RID, "split"]).copy()
    df = add_style(df)                       # style_rank / pace_pressure
    # v6 score
    de = apply_encoders(df, ve)
    X = de[vf].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    df["_v6_score"] = vm.predict(X)
    df["_waku"] = pd.to_numeric(df.get(COL_WAKU), errors="coerce")
    log(f"  rows={len(df):,}  races={df[COL_RID].nunique():,}  脚質cov={df['has_style'].mean()*100:.1f}%")

    rows = []
    for rid, g in df.groupby(COL_RID, sort=False):
        g = g.reset_index(drop=True)
        scores = g["_v6_score"].values.astype(float)
        if np.any(np.isnan(scores)) or len(g) < 3:
            continue
        w = PL.pl_weights(scores); p_win = PL.all_tansho(w)
        ent = -(p_win * np.log(np.clip(p_win, 1e-10, 1))).sum()
        ss = np.sort(scores)[::-1]; ps = np.sort(p_win)[::-1]
        row = {COL_RID: rid, "_n_horses": len(g),
               "_v6_top1": float(ss[0]), "_v6_top3_mean": float(ss[:3].mean()),
               "_v6_top5_mean": float(ss[:5].mean()), "_v6_max": float(scores.max()),
               "_v6_min": float(scores.min()), "_v6_std": float(scores.std()),
               "_v6_range": float(scores.max() - scores.min()), "_entropy": float(ent),
               "_p_win_1st": float(ps[0]), "_p_win_3rd": float(ps[2]) if len(ps) >= 3 else float(ps[-1]),
               "_p_win_top3_sum": float(ps[:3].sum())}
        pace = float(g["pace_pressure"].iloc[0]) if "pace_pressure" in g.columns else 0.5
        row.update(m1_race_features(scores, g["style_rank"].values, g["_waku"].values, pace, beta))
        for c in BASE_CATS + ["split"]:
            if c in g.columns: row[c] = g.iloc[0][c]
        rows.append(row)
    rdf = pd.DataFrame(rows)
    labels = load_havoc_labels()
    rdf["rid_s"] = rdf[COL_RID].astype(str).str[:16]
    rdf["is_havoc"] = rdf["rid_s"].map(labels)
    rdf = rdf.dropna(subset=["is_havoc"]).copy()
    rdf["is_havoc"] = rdf["is_havoc"].astype(int)
    return rdf


def ece(p, y, bins=10):
    p = np.asarray(p); y = np.asarray(y)
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[0] -= 1e-9; edges[-1] += 1e-9
    e = 0.0
    for i in range(bins):
        m = (p > edges[i]) & (p <= edges[i + 1])
        if m.sum() == 0: continue
        e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def decile_lift(p, y, bins=10):
    p = np.asarray(p); y = np.asarray(y)
    q = pd.qcut(pd.Series(p).rank(method="first"), bins, labels=False)
    tab = pd.DataFrame({"p": p, "y": y, "q": q}).groupby("q")["y"].agg(["mean", "size"])
    rates = tab["mean"].values
    sp_rho = float(np.corrcoef(np.arange(len(rates)), rates)[0, 1])
    return {"bin_rates": [round(float(x), 4) for x in rates],
            "spread_top_bottom": round(float(rates[-1] - rates[0]), 4),
            "rank_corr": round(sp_rho, 4)}


def train_eval(tr, vl, te, feats, params, best_iter):
    Xtr = tr[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    Xvl = vl[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    Xte = te[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    ytr, yvl, yte = tr["is_havoc"].values, vl["is_havoc"].values, te["is_havoc"].values
    dtr = lgb.Dataset(Xtr, label=ytr, free_raw_data=False)
    dvl = lgb.Dataset(Xvl, label=yvl, free_raw_data=False)
    model = lgb.train(params, dtr, num_boost_round=int(best_iter * 1.2),
                      valid_sets=[dvl], callbacks=[lgb.early_stopping(100, verbose=False)])
    pte = model.predict(Xte)
    m = {"auc": round(roc_auc_score(yte, pte), 5), "ap": round(average_precision_score(yte, pte), 5),
         "logloss": round(log_loss(yte, np.clip(pte, 1e-6, 1 - 1e-6)), 5),
         "ece": round(ece(pte, yte), 5), "best_iter": model.best_iteration,
         "lift": decile_lift(pte, yte)}
    gain = pd.Series(model.feature_importance("gain"), index=feats)
    return m, gain, pte


def main():
    OUT_JSON.parent.mkdir(exist_ok=True)
    beta = load_beta()
    log(f"[beta] {dict(zip(PAIR_FEATURE_NAMES, beta.round(4)))}")
    rdf = build_race_df(beta)
    for sp in ["train", "valid", "test"]:
        sub = rdf[rdf.split == sp]
        log(f"  {sp}: {len(sub):,}  havoc率={sub['is_havoc'].mean()*100:.2f}%")
    # M1 特徴カバレッジ
    cov = {c: round(rdf.loc[rdf.split == "test", c].notna().mean() * 100, 1) for c in M1_FEATS}
    log(f"  M1 cov(test)={cov}")

    cats = [c for c in BASE_CATS if c in rdf.columns]
    encs = {}
    tr0 = rdf[rdf.split == "train"]
    for c in cats:
        le = LabelEncoder()
        le.fit(pd.concat([tr0[c].astype(str).fillna("__NaN__"), pd.Series(["__NaN__"])], ignore_index=True))
        encs[c] = le
    rdf = apply_encoders(rdf, encs)
    tr = rdf[rdf.split == "train"].copy(); vl = rdf[rdf.split == "valid"].copy(); te = rdf[rdf.split == "test"].copy()

    # havoc_v1 の best_params を流用(両アーム共通=容量固定)
    hv = joblib.load(HAVOC_PKL)
    bp = hv.get("best_params", {})
    best_iter = int(hv.get("best_iter", 200))
    params = {"objective": "binary", "metric": "auc",
              "learning_rate": bp.get("lr", 0.05), "num_leaves": bp.get("num_leaves", 31),
              "max_depth": bp.get("max_depth", -1), "min_data_in_leaf": bp.get("min_data_in_leaf", 100),
              "feature_fraction": bp.get("ff", 0.8), "bagging_fraction": bp.get("bf", 0.8),
              "bagging_freq": 5, "lambda_l1": bp.get("l1", 0.1), "lambda_l2": bp.get("l2", 0.1),
              "verbose": -1, "n_jobs": -1, "seed": SEED, "deterministic": True,
              "force_col_wise": True, "feature_pre_filter": False}

    feats_base = BASE_SCALARS + cats
    feats_m1 = feats_base + M1_FEATS
    log("\n[train B1 baseline=havoc_v1再現]")
    m1res, g1, p1 = train_eval(tr, vl, te, feats_base, params, best_iter)
    log(f"  B1 test AUC={m1res['auc']} AP={m1res['ap']} logloss={m1res['logloss']} ECE={m1res['ece']}")
    log("[train B2 = +M1 結合特徴]")
    m2res, g2, p2 = train_eval(tr, vl, te, feats_m1, params, best_iter)
    log(f"  B2 test AUC={m2res['auc']} AP={m2res['ap']} logloss={m2res['logloss']} ECE={m2res['ece']}")

    m1_gain = {c: round(float(g2.get(c, 0)), 1) for c in M1_FEATS}
    m1_gain_share = round(sum(m1_gain.values()) / float(g2.sum()) * 100, 2)
    d = {"dAUC": round(m2res["auc"] - m1res["auc"], 5), "dAP": round(m2res["ap"] - m1res["ap"], 5),
         "dlogloss": round(m2res["logloss"] - m1res["logloss"], 5),
         "dspread": round(m2res["lift"]["spread_top_bottom"] - m1res["lift"]["spread_top_bottom"], 4)}
    # 判定: 波乱検出AUCで意味ある改善(>= +0.005)かつ AP も改善
    win = bool(d["dAUC"] >= 0.005 and d["dAP"] >= 0.003)
    top_m1 = sorted(m1_gain.items(), key=lambda x: -x[1])[:3]
    if win:
        verdict = (f"GAP_FOUND: M1結合特徴が波乱検出を改善(ΔAUC={d['dAUC']:+.5f}/ΔAP={d['dAP']:+.5f})。"
                   f"周辺分布が潰す配置情報が第2モーメントに効く=ユーザー仮説が当たり。Set Transformerへ進む価値。")
    elif m1_gain_share >= 5.0 and d["dAUC"] < 0.005:
        verdict = (f"GREEDY_BUT_NULL: M1特徴はgain {m1_gain_share}%を貪欲に使う(上位{top_m1})のに"
                   f"TEST改善 ΔAUC={d['dAUC']:+.5f}/ΔAP={d['dAP']:+.5f}=ほぼ0。exp_pace_interaction同型="
                   f"配置情報は波乱検出でも汎化ゼロ。隙間は閉じる。Set Transformerは期待値極低。")
    else:
        verdict = (f"NULL: M1特徴 gain {m1_gain_share}% / ΔAUC={d['dAUC']:+.5f}。havoc_v1のスカラーで"
                   f"既に飽和。M1読み出しは波乱検出に新規寄与なし。")

    out = {"target": "is_havoc (1着オッズ≥5)", "beta_source": "gate1_corr_calibration.json (train≤2022 MLE)",
           "beta": dict(zip(PAIR_FEATURE_NAMES, [round(float(b), 4) for b in beta])),
           "n": {sp: int((rdf.split == sp).sum()) for sp in ["train", "valid", "test"]},
           "havoc_rate_test": round(float(te["is_havoc"].mean()), 4),
           "m1_coverage_test": cov, "params_source": "havoc_v1.best_params (両アーム共通)",
           "B1_baseline": m1res, "B2_plus_m1": m2res, "delta": d,
           "m1_gain": m1_gain, "m1_gain_share_pct": m1_gain_share, "verdict": verdict}
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    log("\n" + "=" * 84)
    log(f"havoc率(test)={out['havoc_rate_test']*100:.1f}%  n_test={out['n']['test']:,}")
    log(f"B1 baseline : AUC={m1res['auc']} AP={m1res['ap']} ECE={m1res['ece']} 十分位spread={m1res['lift']['spread_top_bottom']}")
    log(f"B2 +M1      : AUC={m2res['auc']} AP={m2res['ap']} ECE={m2res['ece']} 十分位spread={m2res['lift']['spread_top_bottom']}")
    log(f"Δ           : AUC={d['dAUC']:+.5f}  AP={d['dAP']:+.5f}  spread={d['dspread']:+.4f}")
    log(f"M1 gain share={m1_gain_share}%  上位={top_m1}")
    log(f"\n判定: {verdict}")
    log(f"[saved] {OUT_JSON}")


if __name__ == "__main__":
    main()
