"""
build_pedigree_emb.py — 血統グラフ構造の埋め込み (成績不使用 = リークなし)
==========================================================================
gensim(Node2Vec) が wheel ビルド不可のため、構造埋め込みを scipy/sklearn で実装:
  各馬を (種牡馬, 母父馬, 母馬) への incidence (親 one-hot 連結) で疎ベクトル化し、
  TruncatedSVD で低次元化。親を共有する馬 (同父=半兄弟, 同母父, 同母=全兄弟) が近い
  埋め込みになり、血統の近縁・系統構造を捉える (DeepWalk≈PPMI-SVD の隣接1次・簡略版)。
  ※効けば次段で Node2Vec(高次walk)/GNN へ精緻化する切り分け。

★リーク防止: 入力は血統の接続構造 (誰の子か) のみ。着順・タイム等の成績は 1 列も使わない
  (usecols に成績を含めない)。血統は出生時に確定し時系列と無関係 → 構造的にリーク不能。
  検査: 入力列が {血統登録番号, 種牡馬, 母父馬, 母馬} のみであることを assert (成績列0)。

出力: data/pedigree_emb.parquet (血統登録番号, ped_emb_00..63)
       reports/pedigree_emb_leakcheck.json
実行: PYTHONUTF8=1 python build_pedigree_emb.py
"""
from __future__ import annotations
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, hstack
from sklearn.decomposition import TruncatedSVD

warnings.filterwarnings("ignore")
BASE = Path(__file__).parent
MASTER_CSV = BASE / "data/master_v2_20130105-20251228.csv"
OUT_PARQUET = BASE / "data/pedigree_emb.parquet"
OUT_LEAK = BASE / "reports/pedigree_emb_leakcheck.json"

USECOLS = ["血統登録番号", "種牡馬", "母父馬", "母馬"]   # ★成績列を含めない
EMB_DIM = 64
SEED = 42


def _onehot(series: pd.Series):
    cats = series.astype("category")
    codes = cats.cat.codes.values
    n = len(codes); k = len(cats.cat.categories)
    valid = codes >= 0
    m = coo_matrix((np.ones(valid.sum()), (np.arange(n)[valid], codes[valid])),
                   shape=(n, max(k, 1)))
    return m.tocsr(), k


def main():
    # ★リーク検査: 読み込む列に成績由来が無いこと
    leak_bad = [c for c in USECOLS if any(s in c for s in ["着", "順", "タイム", "賞金", "配当", "fukusho", "roi"])]
    assert not leak_bad, f"成績由来列が混入: {leak_bad}"
    print(f"[leak] 入力列={USECOLS} (成績列0) → 構造的リーク不能")

    print(f"[load] {MASTER_CSV.name} usecols={USECOLS}")
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", usecols=USECOLS, low_memory=False)
    h = df.drop_duplicates("血統登録番号").reset_index(drop=True)
    peds = h["血統登録番号"].astype(str).values
    print(f"  unique馬={len(h):,}  種牡馬={h['種牡馬'].nunique():,}  母父={h['母父馬'].nunique():,}  母={h['母馬'].nunique():,}")

    S, ks = _onehot(h["種牡馬"]); MS, kms = _onehot(h["母父馬"]); D, kd = _onehot(h["母馬"])
    X = hstack([S, MS, D]).tocsr()
    print(f"  incidence: {X.shape[0]:,} 馬 × {X.shape[1]:,} 親ノード (sire {ks}+msire {kms}+dam {kd})")

    svd = TruncatedSVD(n_components=EMB_DIM, random_state=SEED)
    emb = svd.fit_transform(X)
    ev = float(svd.explained_variance_ratio_.sum())
    print(f"  TruncatedSVD({EMB_DIM}) 累積寄与率={ev:.3f}")

    out = pd.DataFrame(emb, columns=[f"ped_emb_{i:02d}" for i in range(EMB_DIM)])
    out.insert(0, "血統登録番号", peds)
    out.to_parquet(OUT_PARQUET, index=False)

    leak = {"input_cols": USECOLS, "uses_results": False, "leak_violations": 0, "PASS": True,
            "method": "TruncatedSVD on (sire|msire|dam) one-hot incidence (gensim不可のためNode2Vec代替・構造1次)",
            "emb_dim": EMB_DIM, "n_horses": int(len(h)),
            "explained_variance_ratio_sum": round(ev, 4),
            "note": "成績不使用=構造のみ。test馬の血統も接続構造だけ。リーク不能。"}
    OUT_LEAK.parent.mkdir(exist_ok=True)
    with open(OUT_LEAK, "w", encoding="utf-8") as f:
        json.dump(leak, f, indent=2, ensure_ascii=False)

    print(f"\n[リーク検査] 入力に成績列0 → 違反=0 PASS")
    print(f"[saved] {OUT_PARQUET}  ({len(out):,}馬 × {EMB_DIM}次元)")
    print(f"[saved] {OUT_LEAK}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
