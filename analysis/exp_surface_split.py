# -*- coding: utf-8 -*-
"""
exp_surface_split.py — 芝/ダート分割 vs プール、交絡排除&リーク無しで決着
==========================================================================
問い: 「芝とダートは違う競技。モデルを分けた方が良くない？」

設計(夏専用モデル実験と同じ交絡排除):
  M_pool : 全 train(≤2022) で学習           → 芝test と ダtest 両方で評価
  M_turf : 芝 train のみで学習               → 芝test で評価
  M_dirt : ダ train のみで学習               → ダtest で評価
  3モデルは **同一params・同一feats・同一encoder**。違うのは「学習データの馬場だけ」。
  → M_turf が M_pool を芝testで上回るか? M_dirt が ダtestで上回るか?
  差が運かどうかは **同一レースのペア・ブートストラップ** で ◎top3 差の95%CIを出す。

リーク対策:
  - train≤2022 / valid=2023 / test=2024-25 (master_v2 の split 列のみ)。test は学習に一切不使用
  - encoder は train で fit (train_unified_rank.fit_encoders を流用)
  - 馬場マスクは encode 前の生の "芝・ダ"(発走前条件) で作成。helper 列は feats に入れない
  - ラベルは着順のみ由来。特徴は本番 LEAK_COLS 除外済みの select_features
  - race-relative 特徴はレース内 z/rank (時系列の他レースを参照しない)

実行: PYTHONUTF8=1 ./venv311/Scripts/python.exe analysis/exp_surface_split.py
出力: reports/exp_surface_split.json
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(r"E:\PyCaLiAI"); sys.path.insert(0, str(BASE))
import train_unified_rank as TR

COL_RID, COL_JYUN, COL_BAN = TR.COL_RID, TR.COL_JYUN, TR.COL_BAN
SURF_COL = "芝・ダ"


def surf_mask(frame):
    s = frame[SURF_COL].astype(str)
    return s.str.startswith("芝").values, s.str.startswith("ダ").values


def per_race_top3(df, model, feats):
    """rid -> ◎(score最上位馬)が3着以内か 0/1。スコアは特徴のみ、判定は着順のみ。"""
    X = df[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    d = df.assign(_score=model.predict(X))
    out = {}
    for rid, g in d.groupby(COL_RID, sort=False):
        if len(g) < 3:
            continue
        top_ban = int(g[COL_BAN].values[np.argmax(g["_score"].values)])
        top3 = set(int(x) for x in g.sort_values(COL_JYUN)[COL_BAN].astype(int).values[:3])
        out[str(rid)] = int(top_ban in top3)
    return out


def paired_boot(hits_a, hits_b, nboot=3000, seed=11):
    """同一レース集合で a,b の ◎top3 率と差の95%CI(レースクラスタ・ペアboot)。"""
    rids = sorted(set(hits_a) & set(hits_b))
    a = np.array([hits_a[r] for r in rids], float)
    b = np.array([hits_b[r] for r in rids], float)
    ra, rb = a.mean(), b.mean()
    rng = np.random.default_rng(seed); m = len(rids); diffs = np.empty(nboot)
    for i in range(nboot):
        idx = rng.integers(0, m, m)
        diffs[i] = a[idx].mean() - b[idx].mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"n_races": m, "rate_a": float(ra), "rate_b": float(rb),
            "diff": float(ra - rb), "ci95": [float(lo), float(hi)],
            "significant": bool(lo > 0 or hi < 0)}


def main():
    print("=" * 70)
    print("芝/ダート分割 vs プール  (交絡排除・リーク無し・同一レシピ)")
    print("=" * 70)
    df = TR.load_data()
    tr, vl, te = TR.split_df(df)

    encs = TR.fit_encoders(tr)                    # train(≤2022) でのみ fit
    trE = TR.apply_encoders(tr, encs)
    vlE = TR.apply_encoders(vl, encs)
    teE = TR.apply_encoders(te, encs)
    feats = TR.select_features(trE)
    feats = [f for f in feats if f not in ("_surf", "year")]
    print(f"feats={len(feats)}  (例 surface列含む: {'芝・ダ' in feats})")

    tr_t, tr_d = surf_mask(tr)
    vl_t, vl_d = surf_mask(vl)
    te_t, te_d = surf_mask(te)
    print(f"train 芝={tr_t.sum():,} ダ={tr_d.sum():,} | valid 芝={vl_t.sum():,} ダ={vl_d.sum():,} | "
          f"test 芝={te_t.sum():,} ダ={te_d.sum():,}")
    uniq = tr[SURF_COL].astype(str).value_counts().to_dict()
    print(f"surface 生値: {uniq}")

    print("\n[train M_pool] 全train ...")
    M_pool = TR.train(trE, vlE, feats)
    print("[train M_turf] 芝train ...")
    M_turf = TR.train(trE[tr_t], vlE[vl_t], feats)
    print("[train M_dirt] ダtrain ...")
    M_dirt = TR.train(trE[tr_d], vlE[vl_d], feats)

    teE_t, teE_d = teE[te_t], teE[te_d]
    # ◎top3 per-race
    pool_on_t = per_race_top3(teE_t, M_pool, feats)
    turf_on_t = per_race_top3(teE_t, M_turf, feats)
    pool_on_d = per_race_top3(teE_d, M_pool, feats)
    dirt_on_d = per_race_top3(teE_d, M_dirt, feats)

    print("\n=== 芝test: M_turf vs M_pool (◎top3, ペアboot) ===")
    turf_cmp = paired_boot(turf_on_t, pool_on_t)
    print(f"  M_turf={turf_cmp['rate_a']*100:.2f}%  M_pool={turf_cmp['rate_b']*100:.2f}%  "
          f"diff={turf_cmp['diff']*100:+.2f}pt  CI95=[{turf_cmp['ci95'][0]*100:+.2f},{turf_cmp['ci95'][1]*100:+.2f}]  "
          f"{'★有意' if turf_cmp['significant'] else '差は運の範囲'}")

    print("=== ダtest: M_dirt vs M_pool (◎top3, ペアboot) ===")
    dirt_cmp = paired_boot(dirt_on_d, pool_on_d)
    print(f"  M_dirt={dirt_cmp['rate_a']*100:.2f}%  M_pool={dirt_cmp['rate_b']*100:.2f}%  "
          f"diff={dirt_cmp['diff']*100:+.2f}pt  CI95=[{dirt_cmp['ci95'][0]*100:+.2f},{dirt_cmp['ci95'][1]*100:+.2f}]  "
          f"{'★有意' if dirt_cmp['significant'] else '差は運の範囲'}")

    # 参考: 単勝/三連単 も
    ref = {
        "turf_test": {"M_turf": TR.evaluate(teE_t, M_turf, feats, "芝test M_turf"),
                      "M_pool": TR.evaluate(teE_t, M_pool, feats, "芝test M_pool")},
        "dirt_test": {"M_dirt": TR.evaluate(teE_d, M_dirt, feats, "ダtest M_dirt"),
                      "M_pool": TR.evaluate(teE_d, M_pool, feats, "ダtest M_pool")},
    }

    out = {
        "design": "M_pool/M_turf/M_dirt 同一レシピ(params/feats/encoder)。違いは学習データの馬場のみ。test=2024-25未使用。",
        "counts": {"train_turf": int(tr_t.sum()), "train_dirt": int(tr_d.sum()),
                   "test_turf_races": turf_cmp["n_races"], "test_dirt_races": dirt_cmp["n_races"]},
        "turf_test_top3": turf_cmp, "dirt_test_top3": dirt_cmp, "reference_rates": ref,
        "verdict_rule": "diff>0 かつ CI下限>0 で『分割が有意に勝つ』。そうでなければ分割は効かない(プール同等以上)。",
    }
    json.dump(out, open(BASE / "reports/exp_surface_split.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=float)
    print("\n[saved] reports/exp_surface_split.json")


if __name__ == "__main__":
    main()
