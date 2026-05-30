"""
train_catboost_marks_diag.py
============================
診断: 「血統・騎手などの高カーディナリティ・カテゴリを native categorical で
扱うと test 印精度がどれだけ上がるか」を CatBoost で測る。

v6 (unified_rank_v6, LGBM LambdaRank) との唯一の差分:
  - v6 は CAT_COLS を LabelEncoder で整数化し、連続値として LGBM に渡す
  - 本スクリプトは CAT_COLS を string のまま CatBoost の cat_features に渡す
    (ordered boosting により目的統計量が構造的にリークフリー)

それ以外 (master_v2 / 時系列split / feats / label=clip(6-着順,0,5)) は v6 と同一。

リークフリー保証:
  - train<=2022 / valid=2023 / test=2024-25 の時系列split厳守
  - 血統登録番号は v6 同様 feats から除外 (個体記憶を防ぐ)
  - 目的由来の特徴量は一切追加しない
  - CatBoost ordered TS はレース内/未来情報を使わない

比較基準 (evaluate_transformer_v6_stack.json α=1.0, test 2024-25, 6908 races):
  ◎top1=30.24%  ◎top2=?  ◎top3=62.03%  win∈top5=78.20%

実行:
  python train_catboost_marks_diag.py            # full
  python train_catboost_marks_diag.py --smoke    # 高速スモーク (pipeline検証)
"""
from __future__ import annotations
import io
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

BASE       = Path(__file__).parent
MASTER_CSV = BASE / "data/master_v2_20130105-20251228.csv"

COL_RID  = "レースID(新/馬番無)"
COL_JYUN = "着順"
SEED     = 42
SMOKE    = "--smoke" in sys.argv

# v6 と同一
LEAK_COLS = {
    "着順", "fukusho_flag", "roi_target",
    "レースID(新)", "レースID(新/馬番無)",
    "馬名", "レース名", "発走時刻", "date_dt", "日付",
    "血統登録番号", "split",
}
CAT_COLS = [
    "場所", "芝・ダ", "コース区分", "芝(内・外)", "馬場状態", "天気",
    "クラス名", "種牡馬", "父タイプ名", "母父馬", "母父タイプ名", "毛色",
    "馬主(最新/仮想)", "生産者", "騎手コード", "調教師コード",
    "年齢限定", "限定", "性別限定", "指定条件", "重量種別", "性別",
    "ブリンカー", "前走場所", "前芝・ダ", "前走馬場状態", "前走競走種別", "前好走",
]

# v6 の test 2024-25 実測 (比較基準)
V6_REF = {"hon_top1": 0.3024, "hon_top2": None, "hon_top3": 0.6203, "winner_in_top5": 0.7820}


def load_prep():
    print(f"[load] {MASTER_CSV}")
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df = df.dropna(subset=[COL_JYUN, COL_RID, "split"]).copy()
    df["label"] = np.clip(6 - df[COL_JYUN].astype(int), 0, 5).astype(int)

    feats = [c for c in df.columns if c not in LEAK_COLS and c != "label"]
    cat_in_feats = [c for c in CAT_COLS if c in feats]
    num_feats = [c for c in feats if c not in cat_in_feats]

    # カテゴリは string のまま (CatBoost native)。NaN → "__NaN__"
    for c in cat_in_feats:
        df[c] = df[c].astype(str).fillna("__NaN__")
    # 数値は numeric化 (CatBoost が NaN を native 処理)
    for c in num_feats:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    print(f"  feats={len(feats)}  (cat={len(cat_in_feats)}, num={len(num_feats)})")
    return df, feats, cat_in_feats


def make_pool(d, feats, cat_in_feats):
    d = d.sort_values(COL_RID).reset_index(drop=True)
    grp = pd.factorize(d[COL_RID])[0]  # sort後なので同一RIDは連続 → contiguous group
    return Pool(d[feats], label=d["label"].values, group_id=grp, cat_features=cat_in_feats)


def evaluate_marks(te_scored):
    """v6 と同じ印精度: ◎=最高スコア馬。len>=5 のレースのみ。"""
    n_race = 0
    hon_top1 = hon_top2 = hon_top3 = winner_in_top5 = 0
    for rid, g in te_scored.groupby(COL_RID, sort=False):
        if len(g) < 5:
            continue
        scores = g["_score"].values
        jyun = g[COL_JYUN].astype(int).values
        n_race += 1
        order = np.argsort(-scores)
        hon = int(order[0])
        top5_pred = set(int(x) for x in order[:5])
        sij = np.argsort(jyun)
        wi, pi_, si = int(sij[0]), int(sij[1]), int(sij[2])
        if hon == wi:
            hon_top1 += 1
        if hon in {wi, pi_}:
            hon_top2 += 1
        if hon in {wi, pi_, si}:
            hon_top3 += 1
        if wi in top5_pred:
            winner_in_top5 += 1
    return {
        "n_race": n_race,
        "hon_top1": hon_top1 / n_race,
        "hon_top2": hon_top2 / n_race,
        "hon_top3": hon_top3 / n_race,
        "winner_in_top5": winner_in_top5 / n_race,
    }


def main():
    t0 = time.time()
    df, feats, cat_in_feats = load_prep()

    tr = df[df["split"] == "train"].copy()
    vl = df[df["split"] == "valid"].copy()
    te = df[df["split"] == "test"].copy()

    if SMOKE:
        # 高速 pipeline 検証: train を race 単位で subsample
        rids = tr[COL_RID].drop_duplicates().sample(2000, random_state=SEED)
        tr = tr[tr[COL_RID].isin(rids)].copy()
        print("[smoke] train を 2000 races に縮小")

    print(f"  train={len(tr):,} ({tr[COL_RID].nunique():,}R)  "
          f"valid={len(vl):,} ({vl[COL_RID].nunique():,}R)  "
          f"test={len(te):,} ({te[COL_RID].nunique():,}R)")

    pool_tr = make_pool(tr, feats, cat_in_feats)
    pool_vl = make_pool(vl, feats, cat_in_feats)

    iters = 80 if SMOKE else 2000
    model = CatBoostRanker(
        loss_function="YetiRank",
        iterations=iters,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3.0,
        random_seed=SEED,
        eval_metric="NDCG:top=5",
        od_type="Iter",
        od_wait=150,
        thread_count=-1,
        task_type="CPU",
        verbose=100,
    )
    print(f"[fit] CatBoostRanker YetiRank depth=8 iters={iters} ...")
    model.fit(pool_tr, eval_set=pool_vl, use_best_model=True)
    print(f"  best_iter={model.get_best_iteration()}")

    te_s = te.sort_values(COL_RID).reset_index(drop=True)
    te_s["_score"] = model.predict(Pool(te_s[feats], cat_features=cat_in_feats))
    m = evaluate_marks(te_s)

    print("\n" + "=" * 64)
    print(f"=== CatBoost(native cat) vs v6 LGBM  test 2024-25 ({m['n_race']:,} races) ===")
    print("=" * 64)
    print(f"{'指標':16s} {'CatBoost':>10s} {'v6 LGBM':>10s} {'差':>8s}")
    for k in ["hon_top1", "hon_top2", "hon_top3", "winner_in_top5"]:
        cb = m[k] * 100
        ref = V6_REF[k]
        if ref is None:
            print(f"{k:16s} {cb:9.2f}% {'?':>10s}")
        else:
            print(f"{k:16s} {cb:9.2f}% {ref*100:9.2f}% {cb-ref*100:+7.2f}pt")
    print("=" * 64)
    print(f"[done] {time.time()-t0:.0f}s")
    if not SMOKE:
        model.save_model(str(BASE / "models/catboost_marks_diag_v1.cbm"))
        print(f"[saved] models/catboost_marks_diag_v1.cbm")


if __name__ == "__main__":
    main()
