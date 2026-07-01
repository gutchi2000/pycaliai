# -*- coding: utf-8 -*-
"""
train_v5000.py — 全leak-safe異種ファミリ統合 + ファミリ別 honest ablation + v_5000
================================================================================
「真剣に・いろいろ」= 背骨の焼き直しでなく、情報源が構造的に違うファミリを総動員し、
各ファミリが本番 v6 LambdaRank フルスタックで増分を持つかを決定的に測る。

ハーネス = exp_feateng_v7 の v6 再現(同ハイパラ/市場重み/分割/metrics)。

統合ファミリ(全て leak-check 済 parquet, 当日不使用):
  base(120) / elo(16) / glicko(3) / odds_traj(7) / odds_ou(13) / evt(11) /
  keller(8) / race_level(5) / pedigree_emb(64) / serious新規(69) / feats_1000(1039)

出力:
  reports/unified_rank_v_5000_eval.json   ← ★ファミリ別 ΔAUC/Δpair(本命)
  models/unified_rank_v_5000.pkl
実行: PYTHONUTF8=1 ./venv311/Scripts/python.exe train_v5000.py
"""
from __future__ import annotations
import gc, io, json, sys, time, warnings
from itertools import combinations, groupby
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from exp_feateng_v7 import (
    V6, V6_ALPHA, V6_BEST_ITER, REPRO, CAT_COLS, LEAK_COLS,
    metrics, load_winner_pay, fit_encoders, apply_encoders,
)

BASE = Path(__file__).parent
MASTER_CSV = BASE / "data/master_v2_20130105-20251228.csv"
OUT_MODEL = BASE / "models/unified_rank_v_5000.pkl"
OUT_EVAL = BASE / "reports/unified_rank_v_5000_eval.json"
SEED = 42
COL_RID = "レースID(新/馬番無)"; COL_JYUN = "着順"; COL_BAN = "馬番"; COL_PED = "血統登録番号"

# per-row (rid16+ban) で結合する parquet ファミリ
ROW_FAMILIES = {
    "elo": "data/elo_feats.parquet",
    "glicko": "data/glicko_feats.parquet",
    "odds_traj": "data/odds_traj_feats.parquet",
    "odds_ou": "data/odds_ou_feats.parquet",
    "evt": "data/evt_feats.parquet",
    "keller": "data/keller_pace_feats.parquet",
    "race_level": "data/race_level_feats.parquet",
    "race_level_v2": "data/race_level_feats_v2.parquet",
    "serious": "data/feats_serious.parquet",
    "feats_1000": "data/feats_1000.parquet",
}
PED_FAMILY = ("pedigree", "data/pedigree_emb.parquet")


def log(m): print(m, flush=True)
def _rid16(s): return s.astype(str).str.replace(r"\D", "", regex=True).str[:16]
def safe_div(a, b): return (a / np.where(b == 0, np.nan, b))


def load():
    log(f"[load] {MASTER_CSV.name}")
    t0 = time.time()
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df = df.dropna(subset=[COL_JYUN, COL_RID, "split"]).copy().reset_index(drop=True)
    df["label"] = np.clip(6 - df[COL_JYUN].astype(int), 0, 5).astype(int)
    df["rid_s"] = df[COL_RID].astype(str).str[:16]
    df["rid16"] = _rid16(df[COL_RID])
    df["ban"] = pd.to_numeric(df[COL_BAN], errors="coerce").fillna(0).astype(int)
    log(f"  rows={len(df):,}  ({time.time()-t0:.0f}s)")

    base_feats = [c for c in df.columns if c not in LEAK_COLS and c not in {"label", "rid_s", "rid16", "ban"}]
    encs = fit_encoders(df[df["split"] == "train"])
    df = apply_encoders(df, encs)

    DROP_META = {"rid16", "ban", "split", "着順", "fukusho_flag", "roi_target", "label",
                 "血統登録番号", "レースID(新)", "レースID(新/馬番無)", "馬番", "日付",
                 "date_dt", "rid_s", "_pedk", "発走時刻", "馬名"}
    fam_cols = {"base": base_feats}
    for fam, path in ROW_FAMILIES.items():
        g = pd.read_parquet(BASE / path)
        g["rid16"] = g["rid16"].astype(str)
        g["ban"] = pd.to_numeric(g["ban"], errors="coerce").fillna(0).astype(int)
        cols = [c for c in g.columns if c not in DROP_META]
        # 衝突回避(serious と feats_1000 などはprefixで一意のはず)
        df = df.merge(g[["rid16", "ban"] + cols], on=["rid16", "ban"], how="left")
        fam_cols[fam] = cols
        log(f"  merged {fam:12s} +{len(cols)}")
    # pedigree (血統登録番号)
    pname, ppath = PED_FAMILY
    gp = pd.read_parquet(BASE / ppath)
    pcols = [c for c in gp.columns if c != COL_PED and c not in DROP_META]
    gp[COL_PED] = gp[COL_PED].astype(str)
    df["_pedk"] = df[COL_PED].astype(str)
    df = df.merge(gp.rename(columns={COL_PED: "_pedk"}), on="_pedk", how="left")
    fam_cols[pname] = pcols
    log(f"  merged {pname:12s} +{len(pcols)}")

    # serious を S1..S7 に細分(prefix)
    for s in ["S1", "S2", "S3", "S4", "S5", "S6", "S7"]:
        sub = [c for c in fam_cols["serious"] if c.startswith(f"s{s[1]}_")]
        if sub: fam_cols[s] = sub

    # 全特徴 float32
    all_distinct = []
    for fam, cols in fam_cols.items():
        if fam in ("S1", "S2", "S3", "S4", "S5", "S6", "S7"):
            continue  # serious に含まれるので二重カウントしない
        all_distinct += cols
    all_distinct = list(dict.fromkeys(all_distinct))  # 重複除去・順序保持
    for c in all_distinct:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    log(f"  all_distinct={len(all_distinct)}  ({time.time()-t0:.0f}s)")

    import os
    frac = os.environ.get("V5000_FRAC")
    if frac:
        df = (df.groupby("split", group_keys=False)
                .apply(lambda x: x.sample(frac=float(frac), random_state=SEED)).reset_index(drop=True))
        log(f"  [SMOKE] frac={frac} → rows={len(df):,}")
    return df, fam_cols, all_distinct, encs


def make_ds(d, X, win_pay, alpha):
    y = d["label"].to_numpy(np.int32)
    grp = np.array([len(list(gr)) for _, gr in groupby(d[COL_RID])])
    if alpha > 0:
        wt = d["rid_s"].map(win_pay).fillna(100.0).to_numpy(np.float64)
        w = (1.0 + alpha * np.log1p(wt / 100.0)).astype(np.float32)
    else:
        w = np.ones(len(d), np.float32)
    return lgb.Dataset(X, label=y, group=grp, weight=w, free_raw_data=True)


def matrix(df, feats):
    return df[feats].fillna(-9999.0).to_numpy(np.float32)


def train_eval(df, feats, win_pay, rounds, scorecol):
    """feats で学習し test metrics を返す。レース順ソート済み df を仮定。"""
    tr = df[df["split"] == "train"]; vl = df[df["split"] == "valid"]
    params = dict(objective="lambdarank", lambdarank_truncation_level=5, metric="ndcg",
                  eval_at=[5], **V6, bagging_freq=5, verbose=-1, n_jobs=-1, seed=SEED,
                  deterministic=True, force_col_wise=True, feature_pre_filter=False)
    Xtr = matrix(tr, feats)
    ds_tr = make_ds(tr, Xtr, win_pay, V6_ALPHA); del Xtr; gc.collect()
    Xvl = matrix(vl, feats)
    ds_vl = make_ds(vl, Xvl, win_pay, 0.0); del Xvl; gc.collect()
    model = lgb.train(params, ds_tr, num_boost_round=rounds, valid_sets=[ds_vl],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
    Xall = matrix(df, feats)
    df[scorecol] = model.predict(Xall); del Xall; gc.collect()
    return model, metrics(df, scorecol)


def main():
    import os
    SMOKE = bool(os.environ.get("V5000_SMOKE"))
    T0 = time.time(); OUT_MODEL.parent.mkdir(exist_ok=True); OUT_EVAL.parent.mkdir(exist_ok=True)
    df, fam_cols, all_distinct, encs = load()
    df = df.sort_values(COL_RID, kind="mergesort").reset_index(drop=True)
    win_pay = load_winner_pay()
    base = fam_cols["base"]
    R = 60 if SMOKE else int(V6_BEST_ITER * 1.1)

    result = {"protocol": "v6再現ハーネス / 全leak-safe異種ファミリ統合 / train≤2022→valid2023→test2024-25",
              "families": {k: len(v) for k, v in fam_cols.items()},
              "repro_target": REPRO, "ablation": {}, "elapsed_sec": None}

    # ---- baseline ----
    log("\n[baseline]")
    t0 = time.time()
    m_base, mb = train_eval(df, base, win_pay, R, "_s_base")
    repro_ok = (abs(mb["hon_top3"]-REPRO["hon_top3"]) < 0.004 and abs(mb["auc_win"]-REPRO["auc_win"]) < 0.004
                and abs(mb["pairwise_acc"]-REPRO["pairwise_acc"]) < 0.004)
    result["baseline_reproduced"] = bool(repro_ok)
    result["baseline_metrics"] = mb
    log(f"  {time.time()-t0:.0f}s 再現={'OK' if repro_ok else 'NG'} "
        f"◎top3={mb['hon_top3']*100:.2f}% AUC={mb['auc_win']} pair={mb['pairwise_acc']}")

    # ---- ファミリ別 honest ablation (base + 各ファミリ, full-stack) ----
    AB = (["elo", "serious", "feats_1000"] if SMOKE else
          ["elo", "glicko", "odds_traj", "odds_ou", "evt", "keller", "race_level",
           "race_level_v2", "pedigree", "serious", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "feats_1000"])
    for fam in AB:
        cols = fam_cols.get(fam, [])
        if not cols: continue
        feats = base + [c for c in cols if c not in base]
        t0 = time.time()
        _, m = train_eval(df, feats, win_pay, R, f"_s_{fam}")
        d_auc = round(m["auc_win"] - mb["auc_win"], 5)
        d_pair = round(m["pairwise_acc"] - mb["pairwise_acc"], 5)
        d_top3 = round(m["hon_top3"] - mb["hon_top3"], 5)
        gate = bool(d_auc >= 0.008 and d_pair >= 0.005)
        result["ablation"][fam] = {"n_added": len(cols), "metrics": m,
                                   "d_auc": d_auc, "d_pair": d_pair, "d_top3": d_top3, "gate_pass": gate}
        log(f"  [{fam:13s}] +{len(cols):4d} {time.time()-t0:.0f}s "
            f"ΔAUC={d_auc:+.5f} Δpair={d_pair:+.5f} Δ◎top3={d_top3*100:+.2f}pt {'★PASS' if gate else ''}")
        json.dump(result, open(OUT_EVAL, "w", encoding="utf-8"), ensure_ascii=False, indent=2)  # 逐次保存

    # ---- all_serious = base + 全異種ファミリ(クロス/feats_1000無し) ----
    serious_fams = ["elo", "glicko", "odds_traj", "odds_ou", "evt", "keller",
                    "race_level", "race_level_v2", "pedigree", "serious"]
    feats_serious = list(dict.fromkeys(base + [c for f in serious_fams for c in fam_cols[f]]))
    log(f"\n[all_serious] {len(feats_serious)} feats")
    t0 = time.time()
    m_ser, ms = train_eval(df, feats_serious, win_pay, R, "_s_serious")
    result["all_serious"] = {"n_feats": len(feats_serious), "metrics": ms,
                             "d_auc": round(ms["auc_win"]-mb["auc_win"], 5),
                             "d_pair": round(ms["pairwise_acc"]-mb["pairwise_acc"], 5),
                             "d_top3": round(ms["hon_top3"]-mb["hon_top3"], 5)}
    log(f"  {time.time()-t0:.0f}s ◎top3={ms['hon_top3']*100:.2f}% AUC={ms['auc_win']} "
        f"ΔAUC={result['all_serious']['d_auc']:+.5f}")
    json.dump(result, open(OUT_EVAL, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # ---- v_5000 = all_distinct + クロスファミリ層 ----
    log("\n[v_5000] クロスファミリ層を生成 ...")
    ANCHORS = [c for c in [
        # 背骨(少数) + 各異種ファミリの代表を満遍なく
        "kako5_avg_pos", "kako5_best_pos", "kako5_avg_ninki", "kako5_pos_vs_ninki",
        "kako5_pos_trend", "closing_power", "prev_hosei", "prev_hosei9", "horse_fuku30",
        "jockey_fuku30", "jockey_fuku90", "trainer_fuku30", "course_win_rate",
        "前走確定着順", "前走上り3F", "前PCI", "斤量体重比",
        "elo_T2M1_vs", "elo_T2M1_horse", "elo_T1M1_level", "elo_T2M2_vs", "elo_T1M2_vs",
        "elo_T2M2_horse", "elo_T2M1_prevlevel",
        "g2_mu", "g2_rd", "g2_cons",
        "traj_logchg", "traj_rankshift", "traj_range", "traj_revers",
        "ou_informed", "ou_net", "ou_termvel", "ou_s2n", "ou_theta",
        "evt_sigma", "evt_ceiling", "evt_sigma_resid", "evt_mu", "evt_range",
        "kl_lscap", "kl_surge", "kl_pace_fit", "kl_reserve", "kl_draft", "kl_pace_hardness",
        "race_level_prev_W1", "race_level_str_W2",
        "ped_emb_00", "ped_emb_01", "ped_emb_02", "ped_emb_03", "ped_emb_04", "ped_emb_05",
        "ped_emb_06", "ped_emb_07", "ped_emb_08", "ped_emb_09", "ped_emb_10", "ped_emb_11",
        "s1_prev_hosei_best", "s1_prev_hosei_mean", "s1_prev_hosei9_best", "s1_hosei_vs_bucket",
        "s2_style_early", "s2_style_move", "s2_fit_closer", "s2_race_pace_press",
        "s3_sire_surfdist_top3_asof", "s3_msire_surf_top3_asof", "s3_sire_top3_asof", "s3_sire_win_asof",
        "s4_jky_course_top3_asof", "s4_combo_top3_asof", "s4_jky_overall_top3_asof", "s4_trn_course_top3_asof",
        "s5_career_runs", "s5_is_fresh", "s6_k5pos_field_std", "s6_hosei_field_std",
        "s6_field_entropy", "s6_hf30_gap_to_best", "s7_best_class_ever", "s7_class_vs_best",
    ] if c in df.columns]
    log(f"  anchors={len(ANCHORS)}")

    feats_v5000 = None
    try:
        cross_names = []
        A = {c: df[c].to_numpy(np.float32) for c in ANCHORS}
        for a, b in combinations(ANCHORS, 2):                 # 積(全総当たり)
            nm = f"x_{a}__X__{b}"
            df[nm] = (A[a] * A[b]).astype("float32"); cross_names.append(nm)
        for a, b in combinations(ANCHORS[:42], 2):            # 比(前半42アンカー)
            nm = f"x_{a}__OV__{b}"
            df[nm] = safe_div(A[a], A[b]).astype("float32"); cross_names.append(nm)
        feats_v5000 = list(dict.fromkeys(all_distinct + cross_names))
        log(f"  cross={len(cross_names)}  v5000_total={len(feats_v5000)}")
        t0 = time.time()
        m_v5000, mv = train_eval(df, feats_v5000, win_pay, 60 if SMOKE else 2000, "_s_v5000")
        result["v_5000"] = {"n_feats": len(feats_v5000), "n_cross": len(cross_names),
                            "best_iter": m_v5000.best_iteration, "metrics": mv,
                            "d_auc": round(mv["auc_win"]-mb["auc_win"], 5),
                            "d_pair": round(mv["pairwise_acc"]-mb["pairwise_acc"], 5),
                            "d_top3": round(mv["hon_top3"]-mb["hon_top3"], 5),
                            "gate_pass": bool((mv["auc_win"]-mb["auc_win"]) >= 0.008 and
                                              (mv["pairwise_acc"]-mb["pairwise_acc"]) >= 0.005)}
        log(f"  {time.time()-t0:.0f}s it={m_v5000.best_iteration} ◎top3={mv['hon_top3']*100:.2f}% "
            f"AUC={mv['auc_win']} ΔAUC={result['v_5000']['d_auc']:+.5f}")
        joblib.dump({"model": m_v5000, "feature_cols": feats_v5000, "base_feats": base,
                     "fam_cols": fam_cols, "cross_names": cross_names, "encoders": encs,
                     "cat_cols": CAT_COLS, "metrics_test": mv,
                     "description": "unified_rank_v_5000: 全leak-safe異種ファミリ+クロス層. v6本番とは別系統の研究モデル。"},
                    OUT_MODEL)
        log(f"  [saved] {OUT_MODEL}")
    except MemoryError:
        result["v_5000"] = {"status": "OOM_skipped",
                            "n_feats_target": (len(feats_v5000) if feats_v5000 else len(all_distinct)),
                            "note": "メモリ不足でv_5000フル学習はスキップ。all_serious+ablationが本命結果。"}
        log("  ⚠ MemoryError → v_5000スキップ(all_serious+ablationは取得済)")
        gc.collect()

    # ---- サマリ ----
    result["elapsed_sec"] = round(time.time() - T0, 1)
    json.dump(result, open(OUT_EVAL, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    log(f"\n[saved] {OUT_EVAL}")
    log("\n" + "=" * 70)
    log(f"baseline再現: {'OK' if repro_ok else 'NG'}  ◎top3={mb['hon_top3']*100:.2f}% AUC={mb['auc_win']}")
    log("--- ファミリ別 honest ablation (base+各ファミリ, ΔはAUC/pair/◎top3) ---")
    for fam, a in result["ablation"].items():
        log(f"  {fam:13s} +{a['n_added']:4d}  ΔAUC={a['d_auc']:+.5f}  Δpair={a['d_pair']:+.5f}  "
            f"Δ◎top3={a['d_top3']*100:+.2f}pt  {'★PASS' if a['gate_pass'] else '未達'}")
    if "d_auc" in result.get("all_serious", {}):
        a = result["all_serious"]; log(f"  {'all_serious':13s} ({a['n_feats']})  ΔAUC={a['d_auc']:+.5f} Δpair={a['d_pair']:+.5f}")
    if "metrics" in result.get("v_5000", {}):
        a = result["v_5000"]; log(f"  {'v_5000':13s} ({a['n_feats']})  ΔAUC={a['d_auc']:+.5f} Δpair={a['d_pair']:+.5f} {'★PASS' if a['gate_pass'] else '未達'}")
    log("=" * 70)


if __name__ == "__main__":
    main()
