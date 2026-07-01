"""
eval_v7.py
==========
2025年(v6:≤2022学習, v7:≤2023学習 → どちらもOOS)で exotic を比較。
  v6フル         = offline 上限
  v6-serve(12死) = ライブ相当のv6 (serve skew)
  v7            = serve不安定12特徴を除外した新モデル (skew無し=full即serve)

v7 が v6-serve を上回れば、本番投入候補。

実行: PYTHONUTF8=1 python eval_v7.py
"""
from __future__ import annotations
import io, sys
from itertools import combinations, permutations
from pathlib import Path
import joblib, numpy as np, pandas as pd
import pl_probs as PL
from backtest_pl_ev import (apply_encoders, load_payouts, all_umatan_mat,
                            all_sanrenpuku_tensor, COL_RID, COL_BAN)

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
except Exception:
    pass

BASE = Path(__file__).parent
MASTER = BASE / "data/master_v2_20130105-20251228.csv"
CONFIGS = [(3, 4), (6, 10)]
SERVE12_NUM_PREFIX = ("hist_same_",)
SERVE12_NUM = {"course_n_prev", "course_win_rate", "course_top3_rate",
               "jockey_n_prev", "jockey_win_rate", "jockey_top3_rate"}
SERVE12_CAT = {"騎手コード", "調教師コード"}


def settle(sc_df, pays):
    agg = {c: [0, 0.0, 0, 0, 0] for c in CONFIGS}   # cost,ret,um,sp,nR
    for rid, g in sc_df.groupby(COL_RID, sort=False):
        pr = pays.get(str(rid)[:16])
        if pr is None:
            continue
        g = g.sort_values(COL_BAN)
        ban = pd.to_numeric(g[COL_BAN], errors="coerce").values
        s = g["_score"].values
        ok = ~np.isnan(ban)
        if ok.sum() < 3:
            continue
        ban = ban[ok].astype(int); s = s[ok]; n = len(s)
        b2i = {int(x): i for i, x in enumerate(ban)}
        try:
            lwin, lplc, lsho = b2i[int(pr["win"])], b2i[int(pr["plc"])], b2i[int(pr["sho"])]
        except (KeyError, ValueError, TypeError):
            continue
        if not (pr["umatan"] == pr["umatan"] and pr["sanrenpuku"] == pr["sanrenpuku"]):
            continue
        if not pr["umatan"] or not pr["sanrenpuku"]:
            continue
        w = PL.pl_weights(s)
        U = all_umatan_mat(w)
        um_rank = int((U[~np.eye(n, dtype=bool)] > U[lwin, lplc]).sum())
        P3 = all_sanrenpuku_tensor(w)
        tris = np.array(list(combinations(range(n), 3)))
        ptri = np.zeros(len(tris))
        for a, b, c in permutations(range(3)):
            ptri += P3[tris[:, a], tris[:, b], tris[:, c]]
        wt = tuple(sorted((lwin, lplc, lsho)))
        wm = (tris[:, 0] == wt[0]) & (tris[:, 1] == wt[1]) & (tris[:, 2] == wt[2])
        if not wm.any():
            continue
        sp_rank = int((ptri > ptri[wm][0]).sum())
        for cfg in CONFIGS:
            ku, ks = cfg; a = agg[cfg]
            uh = um_rank < ku; sh = sp_rank < ks
            a[0] += (ku + ks) * 100
            a[1] += (pr["umatan"] if uh else 0) + (pr["sanrenpuku"] if sh else 0)
            a[2] += uh; a[3] += sh; a[4] += 1
    return agg


def main():
    pays = load_payouts()
    df = pd.read_csv(MASTER, encoding="utf-8-sig", low_memory=False)
    df = df.dropna(subset=[COL_RID]).copy()
    df["year"] = df[COL_RID].astype(str).str[:4].astype(int)
    df = df[df["year"] == 2025].copy()
    print(f"[data] 2025 rows={len(df):,} races={df[COL_RID].nunique():,}")

    v6 = joblib.load(BASE / "models/unified_rank_v6.pkl")
    v7 = joblib.load(BASE / "models/unified_rank_v7.pkl")

    def score(model, feats, encs, mask_num=(), mask_cat=()):
        d = df.copy()
        for c in mask_cat:
            if c in d.columns:
                d[c] = "__NaN__"
        d = apply_encoders(d, encs)
        for c in mask_num:
            if c in d.columns:
                d[c] = np.nan
        X = d[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
        out = df[[COL_RID, COL_BAN]].copy()
        out["_score"] = model.predict(X)
        return out

    v6_num = [c for c in v6["feature_cols"] if c.startswith(SERVE12_NUM_PREFIX) or c in SERVE12_NUM]
    v6_cat = [c for c in v6["feature_cols"] if c in SERVE12_CAT]

    scenarios = {
        "v6フル(offline上限)": score(v6["model"], v6["feature_cols"], v6["encoders"]),
        "v6-serve(12死=ライブ相当)": score(v6["model"], v6["feature_cols"], v6["encoders"], v6_num, v6_cat),
        "v7(12特徴除外/skew無)": score(v7["model"], v7["feature_cols"], v7["encoders"]),
    }

    print(f"\n{'='*82}")
    print("2025 OOS exotic 比較 (馬単+三連複, prob-first)")
    print(f"{'='*82}")
    print(f"{'モデル':<26}{'config':>8}{'ROI':>8}{'馬単的中':>9}{'三連複的中':>11}")
    for name, sc in scenarios.items():
        agg = settle(sc, pays)
        for ci, cfg in enumerate(CONFIGS):
            a = agg[cfg]
            roi = a[1] / a[0] * 100 if a[0] else 0
            tag = name if ci == 0 else ""
            print(f"{tag:<26}{str(cfg):>8}{roi:>7.1f}%{a[2]/a[4]*100:>8.1f}%{a[3]/a[4]*100:>10.1f}%")
        print("  " + "-" * 76)


if __name__ == "__main__":
    main()
