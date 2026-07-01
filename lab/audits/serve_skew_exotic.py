"""
serve_skew_exotic.py
====================
serve skew が exotic(馬単/三連複)戦略の ROI/的中をどれだけ落とすかを、
master_v2(2024-25, 正解あり)で定量化。serve_skew_eval.py の特徴マスク手法を
exotic 精算に拡張。

シナリオ:
  baseline(全特徴)   = offline 上限 (= 前回バックテストの ~82%/22%)
  現状serve(12死)    = 2026-06-11修正後に本番で死ぬ12特徴をマスク
  監査時serve(30死)  = 修正前(=ライブ11週のほぼ全期間)で死ぬ30特徴をマスク

baseline 82% → 30死 で ~69% に落ちれば、ライブ劣化の犯人は serve skew(drift でなく)。
12死 の値が「6/11修正後にどこまで戻るか」。

実行: PYTHONUTF8=1 python serve_skew_exotic.py
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
    if not isinstance(sys.stdout, io.TextIOWrapper) or (sys.stdout.encoding or "").lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
except Exception:
    pass

BASE = Path(__file__).parent
MASTER = BASE / "data/master_v2_20130105-20251228.csv"
MODEL = BASE / "models/unified_rank_v6.pkl"
CONFIGS = [(3, 4), (6, 10)]

DEAD_PREFIX_30 = ("trnH_", "trnW_", "hist_same_")
DEAD_EXACT_30 = {"course_n_prev", "course_win_rate", "course_top3_rate",
                 "jockey_n_prev", "jockey_win_rate", "jockey_top3_rate", "prev_hosei", "prev_hosei9"}
DEAD_CAT = {"騎手コード", "調教師コード"}
SERVE12_PREFIX = ("hist_same_",)
SERVE12_EXACT = {"course_n_prev", "course_win_rate", "course_top3_rate",
                 "jockey_n_prev", "jockey_win_rate", "jockey_top3_rate"}


def main():
    b = joblib.load(MODEL); model, feats, encs = b["model"], b["feature_cols"], b["encoders"]
    dead30_num = [c for c in feats if c.startswith(DEAD_PREFIX_30) or c in DEAD_EXACT_30]
    dead30_cat = [c for c in feats if c in DEAD_CAT]
    serve12_num = [c for c in feats if c.startswith(SERVE12_PREFIX) or c in SERVE12_EXACT]
    serve12_cat = [c for c in feats if c in DEAD_CAT]
    print(f"[feats] total={len(feats)}  30死={len(dead30_num)+len(dead30_cat)}  12死={len(serve12_num)+len(serve12_cat)}")
    print(f"[30死] {sorted(dead30_num)+sorted(dead30_cat)}")

    print(f"[load] {MASTER.name}")
    df = pd.read_csv(MASTER, encoding="utf-8-sig", low_memory=False)
    df = df.dropna(subset=[COL_RID, "split"]).copy()
    df = df[df["split"] == "test"].copy()
    pays = load_payouts()
    print(f"[data] test rows={len(df):,} races={df[COL_RID].nunique():,}")

    def score(mask_num, mask_cat):
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

    scenarios = {
        "baseline(全特徴)": ([], []),
        "現状serve(12死)": (serve12_num, serve12_cat),
        "監査時serve(30死)": (dead30_num, dead30_cat),
    }

    print(f"\n{'='*86}")
    print("serve skew が exotic に与える影響 (master_v2 test=2024-25, 同レース)")
    print(f"{'='*86}")
    print(f"{'シナリオ':<20}{'config':>8}{'ROI':>8}{'馬単的中':>9}{'三連複的中':>11}{'R的中':>8}")
    for name, (mn, mc) in scenarios.items():
        sc = score(mn, mc)
        agg = {c: [0, 0.0, 0, 0, 0, 0] for c in CONFIGS}   # cost,ret,um,sp,hitR,nR
        for rid, g in sc.groupby(COL_RID, sort=False):
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
            for a, bb, c in permutations(range(3)):
                ptri += P3[tris[:, a], tris[:, bb], tris[:, c]]
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
                a[2] += uh; a[3] += sh; a[4] += (uh or sh); a[5] += 1
        for ci, cfg in enumerate(CONFIGS):
            a = agg[cfg]
            roi = a[1] / a[0] * 100 if a[0] else 0
            tag = name if ci == 0 else ""
            print(f"{tag:<20}{str(cfg):>8}{roi:>7.1f}%{a[2]/a[5]*100:>8.1f}%{a[3]/a[5]*100:>10.1f}%{a[4]/a[5]*100:>7.1f}%")
        print("  " + "-" * 80)


if __name__ == "__main__":
    main()
