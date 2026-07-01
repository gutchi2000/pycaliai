"""
exp_trn_relative.py
===================
坂路相対化特徴 (b自己過去比 / d推移 / c同日正規化) を第一段 f に足すと、市場二段合成の
ΔR² が現状 +0.007 から動くか / 印精度が動くか / importance で新特徴が効くかを入れ子protocolで測る。

同一 v6 パラメータ (alpha_sw=0.0308 含む) で 3 本を比較:
  ANCHOR : 本番 unified_rank_v6.pkl + pl_calibrators_v6.pkl (現状 +0.007 の再現確認)
  A      : v6 と同手順で再学習した baseline (新特徴なし)         ← clean A/B の対照
  B1     : A + 坂路相対特徴 (欠損 sentinel -9999)               ← 主役
  B2     : A + 坂路相対特徴 (欠損 train中央値補完 + *_isna flag)

各 arm の f = model → race内PL → tansho Isotonic(valid2023 fit) → 単勝確率。
市場 π9(当日9時 de-vig) と softmax(α logf + β logπ) を
  β: train≤2022 fit / α,τ: valid2023 選択 / test2024-25 確認   (stage1b と同一)
で測る。

出力: reports/exp_trn_relative.json
実行: python exp_trn_relative.py
"""
from __future__ import annotations
import gc, io, json, sys, time, warnings
from itertools import groupby
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
# stdout の utf-8 化は実行時 PYTHONUTF8=1 に委譲 (依存モジュールの二重ラップ回避)

import pl_probs as PL
import stage1_benter_blend as S1
import stage1b_shrinkage_alpha as S1B
from trn_relative_feats import add_trn_relative_feats, _parse_date

BASE       = Path(__file__).parent
MASTER_CSV = BASE / "data/master_v2_20130105-20251228.csv"
KEKKA_CSV  = BASE / "data/kekka_20130105-20251228.csv"
OUT_JSON   = BASE / "reports/exp_trn_relative.json"
SEED = 42

COL_RID  = "レースID(新/馬番無)"
COL_JYUN = "着順"
COL_BAN  = "馬番"

LEAK_COLS = {"着順", "fukusho_flag", "roi_target", "レースID(新)", "レースID(新/馬番無)",
             "馬名", "レース名", "発走時刻", "date_dt", "日付", "血統登録番号", "split"}
CAT_COLS = ["場所", "芝・ダ", "コース区分", "芝(内・外)", "馬場状態", "天気", "クラス名",
            "種牡馬", "父タイプ名", "母父馬", "母父タイプ名", "毛色", "馬主(最新/仮想)", "生産者",
            "騎手コード", "調教師コード", "年齢限定", "限定", "性別限定", "指定条件", "重量種別",
            "性別", "ブリンカー", "前走場所", "前芝・ダ", "前走馬場状態", "前走競走種別", "前好走"]

# v6 best params (reports/optuna_v6_marks.json)
V6 = dict(num_leaves=59, max_depth=12, min_data_in_leaf=197, learning_rate=0.05083158501542684,
          feature_fraction=0.876098829427658, bagging_fraction=0.7031707810405968,
          lambda_l1=0.0011077902520957399, lambda_l2=7.537933313450104)
V6_BEST_ITER = 469
V6_ALPHA_SW = 0.03083978412534253   # sample_weight alpha
HELP_FEAT_GROUPS = {
    "kako5":  lambda c: c.startswith("kako5_"),
    "hosei":  lambda c: "hosei" in c,
    "zensou": lambda c: c.startswith("前"),
    "trnH_raw": lambda c: c.startswith("trnH_") and not any(c.startswith(p) for p in
                  ("trnH_t1_", "trnH_t4_", "trnH_self", "trnH_rel")),
    "trn_new": lambda c: c.startswith(("trnH_t1_", "trnH_t4_", "trnH_self", "trnH_rel")),
    "hist":   lambda c: c.startswith("hist_") or c.startswith("course_") or c.startswith("horse_"),
}


def log(m): print(m, flush=True)


# ---------------- data ----------------
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
    df["_rdate"] = _parse_date(df["日付"])
    df = df.dropna(subset=["_rdate"]).reset_index(drop=True)
    df["rid_s"] = df[COL_RID].astype(str).str[:16]
    log(f"  rows={len(df):,}  train={ (df['split']=='train').sum():,} "
        f"valid={(df['split']=='valid').sum():,} test={(df['split']=='test').sum():,}")
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
        v = v.where(v.isin(set(le.classes_)), "__NaN__")
        df[c] = le.transform(v)
    return df


# ---------------- model ----------------
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
    m = lgb.train(params, make_ds(tr, feats, win_pay, V6_ALPHA_SW),
                  num_boost_round=int(V6_BEST_ITER * 1.1),
                  valid_sets=[make_ds(vl, feats, win_pay, 0.0)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    return m


def score_all(df, model, feats, encs, col):
    enc = apply_encoders(df, encs)
    X = enc[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    df[col] = model.predict(X)
    return df


# ---------------- f (PL + tansho calibrator) ----------------
def fit_tansho_cal(df, scorecol):
    vl = df[df["split"] == "valid"]
    ps, ys = [], []
    for rid, g in vl.groupby(COL_RID, sort=False):
        if len(g) < 3: continue
        w = PL.pl_weights(g[scorecol].values)
        p = PL.all_tansho(w)
        jy = g[COL_JYUN].astype(int).values
        win = int(np.argmin(jy))
        ps.extend(p.tolist()); ys.extend([1 if i == win else 0 for i in range(len(w))])
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(np.asarray(ps), np.asarray(ys))
    return iso


def build_races(df, scorecol, cal, snaps):
    races = []
    for rid, g in df.groupby(COL_RID, sort=False):
        if len(g) < 5: continue
        rid16 = str(rid)[:16]
        if rid16 not in snaps: continue
        g = g.sort_values(COL_BAN).reset_index(drop=True)
        ban = g[COL_BAN].astype(int).values
        w = PL.pl_weights(g[scorecol].values)
        f_all = cal.predict(PL.all_tansho(w)) if cal is not None else PL.all_tansho(w)
        f_by_ban = {int(ban[i]): max(float(f_all[i]), 1e-9) for i in range(len(ban))}
        jy = g[COL_JYUN].astype(int).values
        win_ban = int(ban[int(np.argmin(jy))])
        sp = snaps[rid16]
        common = [k for k in sp["o9"] if k in sp["oc"] and k in f_by_ban]
        if win_ban not in common or len(common) < 5: continue
        keys = sorted(common)
        f = np.array([f_by_ban[k] for k in keys])
        p9 = S1.devig(sp["o9"], keys); pc = S1.devig(sp["oc"], keys)
        o9 = np.array([sp["o9"][k] for k in keys]); oc = np.array([sp["oc"][k] for k in keys])
        wi = keys.index(win_ban)
        races.append({"logf": np.log(f), "logp9": np.log(p9), "logpc": np.log(pc),
                      "o9": o9, "oc": oc, "win": wi, "year": int(rid16[:4]),
                      "split": g["split"].iloc[0]})
    return races


# ---------------- 印精度 (test 2024-25) ----------------
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


def importance_table(model, feats):
    gain = pd.Series(model.feature_importance("gain"), index=feats)
    return gain


def group_gain(gain):
    out = {}
    for gname, fn in HELP_FEAT_GROUPS.items():
        cols = [c for c in gain.index if fn(c)]
        out[gname] = {"sum_gain": float(gain[cols].sum()), "n": len(cols)}
    return out


# ---------------- 1 arm 評価 ----------------
def eval_arm(name, df, feats, encs, win_pay, snaps, results):
    log(f"\n[{name}] train (feats={len(feats)}) ...")
    tr = df[df["split"] == "train"]; vl = df[df["split"] == "valid"]
    t0 = time.time()
    model = train_arm(tr, vl, feats, win_pay)
    scorecol = f"_score_{name}"
    df = score_all(df, model, feats, encs, scorecol)
    log(f"  trained+scored {time.time()-t0:.0f}s  best_iter={model.best_iteration}")

    marks = mark_metrics(df, scorecol)
    log(f"  印精度(test {marks['n_races']:,}R): NDCG@5={marks['ndcg5']} "
        f"◎top1={marks['hon_top1']*100:.2f}% ◎top3={marks['hon_top3']*100:.2f}% "
        f"win∈top5={marks['winner_in_top5']*100:.2f}%")

    cal = fit_tansho_cal(df, scorecol)
    races = build_races(df, scorecol, cal, snaps)
    log(f"  blend races={len(races):,}  → π9 入れ子protocol ...")
    out9, dr9, a9, roi9 = S1B.analyze(races, "logp9", "o9", "π9")

    blk = out9["methods"]["3_grid_validselect"]
    mz = out9["monetization_gridalpha"]
    log(f"  ΔR²(test,π9, valid選α={blk['alpha_selected_on_valid']}) = {blk['dR2_test']:+.5f} "
        f"(2024={blk['dR2_test_2024']:+.4f} 2025={blk['dR2_test_2025']:+.4f})")
    log(f"  単勝ROI(test, τ*={mz['tau_selected_on_valid']}) = {mz['test_roi']*100:.1f}% "
        f"(n={mz['test_bets']:,} CLV={(mz['test_mean_clv'] or 0)*100:+.1f}%)")

    gain = importance_table(model, feats)
    results[name] = {
        "n_feats": len(feats), "best_iter": model.best_iteration,
        "marks_test": marks,
        "blend_pi9": {
            "market_only_R2_test": out9["market_only_R2_test"],
            "alpha_selected_on_valid": blk["alpha_selected_on_valid"],
            "dR2_test": blk["dR2_test"], "dR2_test_2024": blk["dR2_test_2024"],
            "dR2_test_2025": blk["dR2_test_2025"], "blend_R2_test": blk["blend_R2_test"],
            "valid_R2_curve": blk["valid_R2_curve"],
            "roi_test": mz["test_roi"], "roi_2024": mz["test_2024_roi"], "roi_2025": mz["test_2025_roi"],
            "tau_selected_on_valid": mz["tau_selected_on_valid"], "test_bets": mz["test_bets"],
            "mean_clv": mz["test_mean_clv"], "by_move": mz["by_move_test"],
        },
        "group_gain": group_gain(gain),
    }
    return df, gain, results


def main():
    T0 = time.time()
    OUT_JSON.parent.mkdir(exist_ok=True)
    results = {"protocol": "β:train≤2022 / α,τ:valid2023 / confirm:test2024-25  (stage1b と同一)",
               "v6_params": V6, "v6_alpha_sw": V6_ALPHA_SW, "reference_dR2": 0.00689}

    df = load_master()
    win_pay = load_winner_pay()

    log("[load] TANPUK 単勝 9時/確定 snapshot ...")
    snaps = S1.load_tanpuk_snapshots()
    log(f"  snapshot races={len(snaps):,}")

    # 新特徴 (sentinel) を付与し、新特徴名を取得
    df, new_feats_sentinel = add_trn_relative_feats(df, missing="sentinel")

    # baseline feats = v6 と同じ列集合 (新特徴・leak・label を除く)
    base_feats = [c for c in df.columns if c not in LEAK_COLS and c != "label"
                  and c != "_rdate" and c != "rid_s" and c not in set(new_feats_sentinel)]
    encs = fit_encoders(df[df["split"] == "train"])
    log(f"  base_feats={len(base_feats)}  new_feats={len(new_feats_sentinel)}")

    # ---- ANCHOR: 本番 v6 ----
    log("\n[ANCHOR] 本番 unified_rank_v6.pkl + pl_calibrators_v6.pkl で +0.007 再現 ...")
    bv6 = joblib.load(BASE / "models/unified_rank_v6.pkl")
    cal6 = joblib.load(BASE / "models/pl_calibrators_v6.pkl"); cal6 = cal6.get("calibrators", cal6)
    enc6 = apply_encoders(df, bv6["encoders"])
    Xv6 = enc6[bv6["feature_cols"]].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    df["_score_ANCHOR"] = bv6["model"].predict(Xv6)
    races6 = build_races(df, "_score_ANCHOR", cal6["tansho"], snaps)
    o9, _, _, _ = S1B.analyze(races6, "logp9", "o9", "π9")
    b6 = o9["methods"]["3_grid_validselect"]; m6 = o9["monetization_gridalpha"]
    results["ANCHOR_prod_v6"] = {
        "marks_test": mark_metrics(df, "_score_ANCHOR"),
        "blend_pi9": {"alpha_selected_on_valid": b6["alpha_selected_on_valid"],
                      "dR2_test": b6["dR2_test"], "dR2_test_2024": b6["dR2_test_2024"],
                      "dR2_test_2025": b6["dR2_test_2025"], "roi_test": m6["test_roi"],
                      "mean_clv": m6["test_mean_clv"], "test_bets": m6["test_bets"]}}
    log(f"  ANCHOR ΔR²(test,π9)={b6['dR2_test']:+.5f} (α*={b6['alpha_selected_on_valid']}) "
        f"ROI={m6['test_roi']*100:.1f}%  ← 参照 +0.00689")
    del enc6, Xv6, races6, bv6, cal6, o9
    gc.collect()

    # ---- A: baseline 再学習 ----
    df, gainA, results = eval_arm("A", df, base_feats, encs, win_pay, snaps, results)

    # ---- B1: + 坂路相対 (sentinel) ----
    feats_B1 = base_feats + new_feats_sentinel
    df, gainB1, results = eval_arm("B1", df, feats_B1, encs, win_pay, snaps, results)

    # 新特徴の gain 順位 (B1)
    rankB1 = gainB1.rank(ascending=False).astype(int)
    new_imp = {f: {"gain": float(gainB1[f]), "rank": int(rankB1[f])} for f in new_feats_sentinel}
    results["B1"]["new_feat_importance"] = dict(sorted(new_imp.items(), key=lambda kv: kv[1]["rank"]))
    # kako5/hosei/前 が A→B1 で食われたか
    gg_A = results["A"]["group_gain"]; gg_B = results["B1"]["group_gain"]
    results["B1"]["group_gain_shift_vs_A"] = {
        k: {"A": gg_A[k]["sum_gain"], "B1": gg_B[k]["sum_gain"],
            "pct_change": round(100 * (gg_B[k]["sum_gain"] - gg_A[k]["sum_gain"]) /
                                gg_A[k]["sum_gain"], 1) if gg_A[k]["sum_gain"] else None}
        for k in gg_A}

    # ---- B2: + 坂路相対 (median 補完 + isna flag) ----
    del df, gainA, gainB1
    gc.collect()
    df2 = load_master()
    train_mask = df2["split"] == "train"
    df2, new_feats_med = add_trn_relative_feats(df2, missing="median", train_mask=train_mask)
    base_feats2 = [c for c in df2.columns if c not in LEAK_COLS and c != "label"
                   and c != "_rdate" and c != "rid_s" and c not in set(new_feats_med)]
    encs2 = fit_encoders(df2[df2["split"] == "train"])
    feats_B2 = base_feats2 + new_feats_med
    df2, gainB2, results = eval_arm("B2", df2, feats_B2, encs2, win_pay, snaps, results)

    # ---- 判定 ----
    drA = results["A"]["blend_pi9"]["dR2_test"]
    drB1 = results["B1"]["blend_pi9"]["dR2_test"]
    drB2 = results["B2"]["blend_pi9"]["dR2_test"]
    top_new_rank = min(v["rank"] for v in new_imp.values())
    delta = drB1 - drA
    if delta >= 0.005 and top_new_rank <= 30:
        verdict = f"REAL_INCREMENT: ΔR² が A比 {delta:+.4f} 上昇かつ新特徴 importance 上位({top_new_rank}位)。調教相対化は本物の増分。"
    elif abs(delta) < 0.002:
        verdict = f"EXHAUSTED: ΔR² ほぼ不変 (A比 {delta:+.4f})。調教情報は既に kako5/前走/hosei 等で間接吸収済み。汲み切り確定。"
    else:
        marks_up = results["B1"]["marks_test"]["hon_top3"] - results["A"]["marks_test"]["hon_top3"]
        verdict = (f"MIXED: ΔR² 変化 {delta:+.4f} (印◎top3 変化 {marks_up*100:+.2f}pt)。"
                   "印は動いても市場超えには非寄与の可能性。")
    results["verdict"] = verdict
    results["summary"] = {
        "dR2_ANCHOR": results["ANCHOR_prod_v6"]["blend_pi9"]["dR2_test"],
        "dR2_A": drA, "dR2_B1_sentinel": drB1, "dR2_B2_median": drB2,
        "dR2_delta_B1_minus_A": round(delta, 5),
        "top_new_feat_rank": top_new_rank,
        "marks_top3_A": results["A"]["marks_test"]["hon_top3"],
        "marks_top3_B1": results["B1"]["marks_test"]["hon_top3"],
        "roi_A": results["A"]["blend_pi9"]["roi_test"],
        "roi_B1": results["B1"]["blend_pi9"]["roi_test"],
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log("\n" + "=" * 72)
    log("=== 判定 ===")
    log(f"  ΔR²: ANCHOR={results['summary']['dR2_ANCHOR']:+.5f}  A={drA:+.5f}  "
        f"B1={drB1:+.5f}  B2={drB2:+.5f}  (B1-A={delta:+.5f})")
    log(f"  ◎top3: A={results['summary']['marks_top3_A']*100:.2f}%  "
        f"B1={results['summary']['marks_top3_B1']*100:.2f}%")
    log(f"  新特徴 最上位 importance 順位: {top_new_rank}")
    log(f"  判定: {verdict}")
    log(f"[saved] {OUT_JSON}  ({time.time()-T0:.0f}s)")


if __name__ == "__main__":
    main()
