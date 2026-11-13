# -*- coding: utf-8 -*-
"""
C2 候補 (2026-11-13 監査の学習特徴修正) — data/master は上書きしない。

監査 audit/correctness_20261113/AUDIT_REPORT.md §C の学習側の不具合を直した master を
candidates/c2_feature_fix/ に作り、v6 本番と同じハイパーパラメータ・alpha・515 ラウンドで学習する。

修正 (本番 master = C1 からの差分だけを作り直す。他の列は C1 と同一):
  1. jockey/trainer_fuku30/90: 「同日の先行レース込み」→ 配信と同じ「その開催週末より前」
     (土日・連休の連続開催日を 1 ブロックとし、ブロック初日より前の騎乗だけを使う)。
     配信 (serve_history_feats) は date < 当日 だが、日曜 bundle は土曜夜に作られ履歴は前週末までしか
     無い (_horse_history.parquet は Phase C でしか更新されない) ので、実際に配信される値はこれ。
  2. trnH_days_ago / trnW_days_ago: YYYYMMDD の整数差 → 実日数。
  3. kako5_avg_agari3f / kako5_best_agari3f / kako5_hidden_good_count: 各過去走の「前走上り3F」
     (=その 1 つ前の走) を使っていた 1 走ずれ → 各過去走自身の上がり (次走行の 前走上り3F、
     前走日付が一致するときだけ)。
  4. 特徴から除外: 馬主(最新/仮想) (未来値の上書き)、斤量体重比 (当日馬体重)、
     前走レースID(新)/(新/馬番無) (配信で常に欠損・日付の代理)、母馬 (数値化で全行欠損)。
  variant b はさらに配信で恒常的に死んでいる特徴 (serve baseline cov<40%) も学習から外す。

出力: E:/PyCaLiAI/candidates/c2_feature_fix/
  data/master_c2.parquet, models/unified_rank_c2{a,b}_seed{42,1}.pkl, manifest.json
実行: PYTHONUTF8=1 E:/PyCaLiAI/venv311/Scripts/python.exe analysis/c2_feature_fix/build_c2_candidate.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
DATA_ROOT = Path(r"E:\PyCaLiAI")
CAND = DATA_ROOT / "candidates" / "c2_feature_fix"
(CAND / "data").mkdir(parents=True, exist_ok=True)
(CAND / "models").mkdir(parents=True, exist_ok=True)

import optuna_v6_marks as ovm  # noqa: E402
from build_dataset import normalize_date, convert_finish  # noqa: E402

ovm.MASTER_CSV = DATA_ROOT / "data/master_v2_20130105-20251228.csv"
ovm.KEKKA_CSV = DATA_ROOT / "data/kekka_20130105-20251228.csv"
RID, BAN = "レースID(新/馬番無)", "馬番"
SEEDS = [42, 1]
t0 = time.time()


def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def log(msg):
    print(f"[{time.time()-t0:7.0f}s] {msg}", flush=True)


def weekend_block(dates: pd.Series) -> pd.Series:
    """開催日 (YYYYMMDD int) → 連続開催ブロックの初日。間隔 2 日以内を同じブロックとする。"""
    u = np.array(sorted(pd.unique(dates.dropna().astype(int))))
    dt = pd.to_datetime(u.astype(str), format="%Y%m%d")
    start = np.empty(len(u), dtype=np.int64)
    s = u[0]
    for i in range(len(u)):
        if i > 0 and (dt[i] - dt[i - 1]).days > 2:
            s = u[i]
        start[i] = s
    return dates.astype(int).map(dict(zip(u, start)))


# ---------------------------------------------------------------- 1. 騎手・調教師
log("STEP1 全母集団 (DNF 含む) を再構築")


def load_cp932_or_utf8(path):
    try:
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp932", low_memory=False)


JOIN_KEY = ["レースID(新)", "馬番"]
lgbm_df = load_cp932_or_utf8(DATA_ROOT / "data/lgbm_20130105-20251228.csv")
cat_df = load_cp932_or_utf8(DATA_ROOT / "data/cat_20130105-20251228.csv")
add_df = load_cp932_or_utf8(DATA_ROOT / "data/add_20130105-20251228.csv")
full = lgbm_df.merge(cat_df[JOIN_KEY + ["騎手コード", "調教師コード"]], on=JOIN_KEY, how="left")
add2 = add_df[JOIN_KEY + ["着順"]].copy()
add2["着順"] = convert_finish(add2["着順"])
full = full.merge(add2, on=JOIN_KEY, how="left")
full["日付"] = normalize_date(full["日付"])
full["fukusho_flag"] = (full["着順"] <= 3).astype("Int8")
assert len(full) == 631_965, len(full)
full = full.sort_values(["日付", "発走時刻", RID, BAN]).reset_index(drop=True)
full["_block"] = weekend_block(pd.to_numeric(full["日付"], errors="coerce"))
log(f"  universe={len(full):,}  blocks={full['_block'].nunique():,}")

FUKU = ["jockey_fuku30", "jockey_fuku90", "trainer_fuku30", "trainer_fuku90"]
for code_col, prefix in [("騎手コード", "jockey"), ("調教師コード", "trainer")]:
    for window in (30, 90):
        col = f"{prefix}_fuku{window}"
        v = (full.groupby(code_col, sort=False)["fukusho_flag"]
             .transform(lambda x: x.shift(1).rolling(window, min_periods=5).mean()))
        full[col] = v
        # (主体, 開催ブロック) の先頭行 = ブロック前までの騎乗だけの窓 → ブロック内全騎乗に配る
        first = full.groupby([code_col, "_block"], sort=False)[col].transform("first")
        # transform("first") は NaN を飛ばすので、先頭行の値そのものを位置で取る
        pos = full.groupby([code_col, "_block"], sort=False).cumcount()
        head = full.loc[pos == 0, [code_col, "_block", col]].rename(columns={col: "__h"})
        full[col] = full[[code_col, "_block"]].merge(head, on=[code_col, "_block"], how="left")["__h"].values
        del first
survivors = full[full["着順"].notna()][[RID, BAN] + FUKU].copy()
assert len(survivors) == 626_774, len(survivors)
del lgbm_df, cat_df, add_df, add2

# ---------------------------------------------------------------- 2. master 差分
log("STEP2 本番 master を読み、差分列を作り直す")
prod_master_path = DATA_ROOT / "data/master_v2_20130105-20251228.csv"
master = pd.read_csv(prod_master_path, encoding="utf-8-sig", low_memory=False)
n0 = len(master)
old = master[[RID, BAN] + FUKU].copy()
master = master.drop(columns=FUKU).merge(survivors, on=[RID, BAN], how="left")
assert len(master) == n0
changes = {}
for c in FUKU:
    a = old[c].to_numpy(float)
    b = master.set_index([RID, BAN]).loc[pd.MultiIndex.from_frame(old[[RID, BAN]]), c].to_numpy(float)
    both = ~np.isnan(a) & ~np.isnan(b)
    changes[c] = {"changed": int((np.abs(a[both] - b[both]) > 1e-9).sum()), "compared": int(both.sum())}
log(f"  jockey/trainer 変化: {changes}")

# 2b. 調教 days_ago → 実日数
d_int = pd.to_numeric(master["日付"], errors="coerce").astype("Int64")
race_dt = pd.to_datetime(d_int.astype(str), format="%Y%m%d", errors="coerce")
for c in ["trnH_days_ago", "trnW_days_ago"]:
    diff = pd.to_numeric(master[c], errors="coerce")
    trn_int = (d_int - diff.round().astype("Int64"))
    trn_dt = pd.to_datetime(trn_int.astype(str), format="%Y%m%d", errors="coerce")
    real = (race_dt - trn_dt).dt.days.astype(float)
    bad = diff.notna() & real.isna()
    changes[c] = {"changed": int((diff.notna() & (real != diff)).sum()), "nonnull": int(diff.notna().sum()),
                  "unparseable": int(bad.sum())}
    master[c] = real
log(f"  days_ago: {changes['trnH_days_ago']} / {changes['trnW_days_ago']}")

# 2c. kako5 上がり (1 走ずれの修正)
master["_d"] = d_int.astype(float)
master["_pd"] = pd.to_numeric(master["前走日付"], errors="coerce") + 20000000  # yymmdd → yyyymmdd
master = master.sort_values(["血統登録番号", "_d"]).reset_index(drop=True)
g = master.groupby("血統登録番号", sort=False)
nxt_agari = g["前走上り3F"].shift(-1)
nxt_prev_date = g["_pd"].shift(-1)
own_agari = pd.to_numeric(nxt_agari, errors="coerce").where(nxt_prev_date == master["_d"])
master["_own_agari"] = own_agari
master["_pos"] = pd.to_numeric(master["着順"], errors="coerce")
master["_hidden"] = ((master["_pos"] > 5) & (master["_own_agari"] < 34.5)).astype(float)
g = master.groupby("血統登録番号", sort=False)
prev_agari = g["_own_agari"].shift(1)
prev_hidden = g["_hidden"].shift(1)
master["_pa"] = prev_agari
master["_ph"] = prev_hidden
g = master.groupby("血統登録番号", sort=False)
avg = g["_pa"].transform(lambda x: x.rolling(5, min_periods=1).mean())
best = g["_pa"].transform(lambda x: x.rolling(5, min_periods=1).min())
hid = g["_ph"].transform(lambda x: x.rolling(5, min_periods=1).sum())
has_past = g.cumcount() > 0
for c, newv in [("kako5_avg_agari3f", avg), ("kako5_best_agari3f", best)]:
    oldv = pd.to_numeric(master[c], errors="coerce")
    master[c] = newv.where(has_past)
    changes[c] = {"changed": int((oldv.notna() & (oldv - master[c]).abs().gt(1e-9)).sum()),
                  "nonnull_new": int(master[c].notna().sum()), "nonnull_old": int(oldv.notna().sum())}
oldv = pd.to_numeric(master["kako5_hidden_good_count"], errors="coerce")
master["kako5_hidden_good_count"] = hid.where(has_past & oldv.notna())
changes["kako5_hidden_good_count"] = {"changed": int((oldv - master["kako5_hidden_good_count"]).abs().gt(1e-9).sum())}
log(f"  kako5 agari: { {k: changes[k] for k in ['kako5_avg_agari3f', 'kako5_best_agari3f', 'kako5_hidden_good_count']} }")
master = master.drop(columns=["_d", "_pd", "_own_agari", "_pos", "_hidden", "_pa", "_ph"])
master = master.sort_values([RID, BAN]).reset_index(drop=True)
master.to_parquet(CAND / "data/master_c2.parquet", index=False)
log(f"  [saved] master_c2.parquet rows={len(master):,}")

# ---------------------------------------------------------------- 3. 学習
prod = joblib.load(DATA_ROOT / "models/unified_rank_v6.pkl")
FEATS_V6 = prod["feature_cols"]
params0 = dict(prod["base_lgbm_params"])
NUM_ROUND = int(prod["num_boost_round"])
ALPHA = float(prod["sample_weight_alpha"])
LEAK_DROP = ["馬主(最新/仮想)", "斤量体重比", "前走レースID(新)", "前走レースID(新/馬番無)", "母馬"]
base_cov = json.load(open(DATA_ROOT / "data/serve_feature_baseline.json", encoding="utf-8"))["baseline_cov"]
SERVE_DEAD = [c for c in FEATS_V6 if base_cov.get(c, 0.0) < 0.40 and c not in LEAK_DROP]
VARIANTS = {"c2a": [c for c in FEATS_V6 if c not in LEAK_DROP],
            "c2b": [c for c in FEATS_V6 if c not in LEAK_DROP and c not in SERVE_DEAD]}

from sklearn.preprocessing import LabelEncoder  # noqa: E402

df = master.copy()
df[ovm.COL_JYUN] = pd.to_numeric(df[ovm.COL_JYUN], errors="coerce")
df = df.dropna(subset=[ovm.COL_JYUN, RID, "split"]).copy()
df["label"] = np.clip(6 - df[ovm.COL_JYUN].astype(int), 0, 5).astype(int)
tr = df[df["split"] == "train"].copy()
encs = {}
for c in ovm.CAT_COLS:
    if c not in tr.columns:
        continue
    le = LabelEncoder()
    vals = tr[c].astype(str).fillna("__NaN__")
    le.fit(pd.concat([vals, pd.Series(["__NaN__"])], ignore_index=True))
    encs[c] = le
for c, le in encs.items():
    v = tr[c].astype(str).fillna("__NaN__")
    tr[c] = le.transform(v.where(v.isin(set(le.classes_)), "__NaN__"))
win_pay = ovm.load_winner_tansho_pay()
tr["rid_s"] = tr[RID].astype(str)
tr["winner_tansho"] = tr["rid_s"].map(win_pay).fillna(100.0)
log(f"STEP3 学習 train={len(tr):,}  round={NUM_ROUND} alpha={ALPHA:.4f}  serve_dead={len(SERVE_DEAD)}")

outputs = {}
for vname, feats in VARIANTS.items():
    for seed in SEEDS:
        p = dict(params0)
        p["seed"] = seed
        ds, _, _ = ovm.make_dataset(tr, feats, alpha=ALPHA)
        w = ds.get_weight()
        assert (np.abs(w - 1.0) > 1e-9).mean() > 0.5, "sample weight が効いていない"
        model = lgb.train(p, ds, num_boost_round=NUM_ROUND)
        out = CAND / "models" / f"unified_rank_{vname}_seed{seed}.pkl"
        joblib.dump({"model": model, "feature_cols": feats, "encoders": encs, "seed": seed,
                     "variant": f"{vname}_feature_fix_20261113",
                     "base_lgbm_params": p, "num_boost_round": NUM_ROUND, "sample_weight_alpha": ALPHA,
                     # 配信 (export_weekly_marks) がこの 2 つで定義を切り替える
                     "trn_days_unit": "days", "kako5_agari_window": "own",
                     "jockey_window": "weekend_block_start",
                     "dropped_features": sorted(set(FEATS_V6) - set(feats)),
                     "master": "candidates/c2_feature_fix/data/master_c2.parquet"}, out)
        outputs[f"{vname}_seed{seed}"] = {"path": str(out), "n_feats": len(feats), "sha256": sha256_file(out)}
        log(f"  [saved] {out.name} feats={len(feats)}")

manifest = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "source_master_sha256": sha256_file(prod_master_path),
            "changes": changes, "leak_drop": LEAK_DROP, "serve_dead_dropped_in_c2b": SERVE_DEAD,
            "hyperparams_from": "models/unified_rank_v6.pkl (C1)", "num_boost_round": NUM_ROUND,
            "alpha": ALPHA, "models": outputs}
json.dump(manifest, open(CAND / "manifest.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=str)
log("done")
