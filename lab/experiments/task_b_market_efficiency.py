"""
task_b_market_efficiency.py
===========================
Task B: 全期間 payout 市場効率テスト（前売りオッズ不要・leak-safe）。

leak規約: 馬券の選択は「モデル確率(raw / Stern補正後) bin × 構成層(構成馬のうちモデル
top3に入る頭数 0/1/2/3)」のみで層別。**オッズは選択に一切使わない**。payout(確定配当)は
実現リターンの計算だけに使う → leak-safe。

見たいこと:
  (i) 同じモデル確率 bin の中で、チャート独占目(top3が2〜3頭)の実現ROIが穴絡み(0〜1頭)より
      系統的に低いか = 市場がチャートを過剰人気(過小配当)にしているか。
  (ii) 控除率(≈80%, ROI>0.80)を超えるセルが存在するか。

券種/期間:
  - 馬連・三連複: 全期間 2013-2025
  - ワイド: 2016-2025 (payout parquet の範囲)
  - 三連単: test 2024-25 のみ (全期間は計算コスト大のため限定。チャート信号は三連複と同型)
  各券種で raw(λ=1) と corrected(λ*) の両方で binning。

注意: full 期間は train(2013-22) も含み v6 スコアは in-sample。市場効率(実現payout)の性質
  自体は不変だが bin 分離は楽観方向。test_2024_2025(OOS) も併記して構造の頑健性を見る。

出力: reports/joint_market_efficiency_fullperiod.json
実行: python task_b_market_efficiency.py --model v6 --lam 0.881
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime
from itertools import combinations, permutations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from backtest_pl_ev import (
    load_payouts, load_wide_payouts,
    all_umaren_mat, all_wide_mat, all_sanrenpuku_tensor,
)
import pl_stern as ST
from joint_calibration_v6 import BINS, apply_encoders, COL_RID, COL_JYUN, COL_BAN, MASTER_CSV

warnings.filterwarnings("ignore")
BASE = Path(__file__).parent
np.random.seed(42)
NB = len(BINS) - 1
UNIT = 100.0  # 1点 100円


def dig(p):
    return np.clip(np.digitize(np.asarray(p, dtype=np.float64), BINS) - 1, 0, NB - 1)


NCOMP = {"umaren": 3, "wide": 3, "sanrenpuku": 4, "sanrentan": 4}
PERIODS = ["full", "test"]
# 三連単は full をやらない (test のみ)
FULL_BETS = ["umaren", "wide", "sanrenpuku"]

_TRIS = {}
def tris_for(n):
    if n not in _TRIS:
        _TRIS[n] = np.array(list(combinations(range(n), 3))) if n >= 3 else np.zeros((0, 3), int)
    return _TRIS[n]


def new_grid(nc):
    return {"n": np.zeros((NB, nc)), "ret": np.zeros((NB, nc)), "hits": np.zeros((NB, nc))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="v6")
    ap.add_argument("--lam", type=float, default=0.881, help="Stern λ (Task A の λ*)")
    args = ap.parse_args()
    tag = args.model; lam = args.lam

    print("=" * 64)
    print(f"task_b_market_efficiency.py  model={tag}  λ*={lam}")
    print("=" * 64)

    bundle = joblib.load(BASE / f"models/unified_rank_{tag}.pkl")
    model, feats, encs = bundle["model"], bundle["feature_cols"], bundle["encoders"]

    print("[payouts] load")
    payouts = load_payouts()
    wide_map = load_wide_payouts()
    print(f"  win/連系 payout races={len(payouts):,}  wide races={len(wide_map):,}")

    print(f"[load] {MASTER_CSV.name}")
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df = df.dropna(subset=[COL_JYUN, COL_RID, "split"]).copy()
    df = apply_encoders(df, encs)
    X = df[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    df["_score"] = model.predict(X)
    print(f"  scored rows={len(df):,}  races={df[COL_RID].nunique():,}")

    cells = {bt: {v: {pp: new_grid(NCOMP[bt]) for pp in PERIODS} for v in ["raw", "cor"]}
             for bt in NCOMP}

    n_race = 0; n_skip = 0
    for rid, g in df.groupby(COL_RID, sort=False):
        if len(g) < 5:
            n_skip += 1; continue
        rid_s = str(int(rid)) if isinstance(rid, (int, np.integer)) else str(rid)
        po = payouts.get(rid_s)
        if po is None:
            n_skip += 1; continue
        g = g.sort_values(COL_BAN).reset_index(drop=True)
        ban = g[COL_BAN].astype(int).values
        b2i = {int(b): i for i, b in enumerate(ban)}
        wi = b2i.get(po["win"]); pi = b2i.get(po["plc"]); si = b2i.get(po["sho"])
        if wi is None or pi is None or si is None:
            n_skip += 1; continue
        sc = g["_score"].values.astype(np.float64)
        w = np.exp(sc - sc.max()); v = w ** lam
        n = len(w)
        in3 = np.zeros(n, dtype=bool); in3[np.argsort(-sc)[:3]] = True
        is_test = (g["split"].iloc[0] == "test")
        per = ["full", "test"] if is_test else ["full"]
        per_test = ["test"] if is_test else []

        iu, ju = np.triu_indices(n, k=1)
        comp_pair = (in3[iu].astype(int) + in3[ju].astype(int))

        # ---------- 馬連 (full) ----------
        Mr = all_umaren_mat(w); Mc = ST.all_umaren_mat_stern(w, v)
        for vr, M in (("raw", Mr), ("cor", Mc)):
            pb = dig(M[iu, ju])
            for pp in per:
                np.add.at(cells["umaren"][vr][pp]["n"], (pb, comp_pair), 1.0)
        a, b = min(wi, pi), max(wi, pi)
        cwp = int(in3[wi]) + int(in3[pi])
        pay = float(po["umaren"]) if (po["umaren"] and not pd.isna(po["umaren"])) else 0.0
        for vr, M in (("raw", Mr), ("cor", Mc)):
            pbw = int(dig(np.array([M[a, b]]))[0])
            for pp in per:
                cells["umaren"][vr][pp]["ret"][pbw, cwp] += pay
                cells["umaren"][vr][pp]["hits"][pbw, cwp] += 1

        # ---------- ワイド (full, 2016+ のみ payout 有) ----------
        wlist = wide_map.get(rid_s)
        if wlist:
            Wr = all_wide_mat(w); Wc = ST.all_wide_mat_stern(w, v)
            for vr, W in (("raw", Wr), ("cor", Wc)):
                pb = dig(W[iu, ju])
                for pp in per:
                    np.add.at(cells["wide"][vr][pp]["n"], (pb, comp_pair), 1.0)
            for (ui, uj, wpay) in wlist:
                ii = b2i.get(int(ui)); jj = b2i.get(int(uj))
                if ii is None or jj is None:
                    continue
                cwd = int(in3[ii]) + int(in3[jj])
                for vr, W in (("raw", Wr), ("cor", Wc)):
                    pbw = int(dig(np.array([W[ii, jj]]))[0])
                    for pp in per:
                        cells["wide"][vr][pp]["ret"][pbw, cwd] += float(wpay)
                        cells["wide"][vr][pp]["hits"][pbw, cwd] += 1

        # ---------- 三連複 (full) + 三連単 (test のみ) ----------
        P3r = all_sanrenpuku_tensor(w)
        P3c = ST.all_sanrenpuku_tensor_stern(w, v)
        tris = tris_for(n)
        if len(tris) > 0:
            comp_tri = in3[tris].sum(axis=1)
            def puku(P3):
                pk = np.zeros(len(tris))
                for prm in permutations(range(3)):
                    pk += P3[tris[:, prm[0]], tris[:, prm[1]], tris[:, prm[2]]]
                return pk
            for vr, P3 in (("raw", P3r), ("cor", P3c)):
                pb = dig(puku(P3))
                for pp in per:
                    np.add.at(cells["sanrenpuku"][vr][pp]["n"], (pb, comp_tri), 1.0)
            compw = int(in3[wi]) + int(in3[pi]) + int(in3[si])
            paypk = float(po["sanrenpuku"]) if (po["sanrenpuku"] and not pd.isna(po["sanrenpuku"])) else 0.0
            for vr, P3 in (("raw", P3r), ("cor", P3c)):
                prk = sum(P3[t[0], t[1], t[2]] for t in permutations((wi, pi, si)))
                pbw = int(dig(np.array([prk]))[0])
                for pp in per:
                    cells["sanrenpuku"][vr][pp]["ret"][pbw, compw] += paypk
                    cells["sanrenpuku"][vr][pp]["hits"][pbw, compw] += 1

        # 三連単: test のみ (compute 抑制)
        if is_test:
            I, J, K = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
            m3 = (I != J) & (J != K) & (I != K)
            If, Jf, Kf = I[m3], J[m3], K[m3]
            comp_o = in3[If].astype(int) + in3[Jf].astype(int) + in3[Kf].astype(int)
            for vr, P3 in (("raw", P3r), ("cor", P3c)):
                pb = dig(P3[m3])
                np.add.at(cells["sanrentan"][vr]["test"]["n"], (pb, comp_o), 1.0)
            comps = int(in3[wi]) + int(in3[pi]) + int(in3[si])
            payst = float(po["sanrentan"]) if (po["sanrentan"] and not pd.isna(po["sanrentan"])) else 0.0
            for vr, P3 in (("raw", P3r), ("cor", P3c)):
                pbw = int(dig(np.array([P3[wi, pi, si]]))[0])
                cells["sanrentan"][vr]["test"]["ret"][pbw, comps] += payst
                cells["sanrentan"][vr]["test"]["hits"][pbw, comps] += 1

        n_race += 1
        if n_race % 8000 == 0:
            print(f"  ..{n_race:,} races")

    print(f"  [done] races={n_race:,}  skipped={n_skip:,}")

    # ---------- 集計 ----------
    def summarize(grid):
        n = grid["n"]; ret = grid["ret"]; hits = grid["hits"]
        nc = n.shape[1]
        by_comp = {}
        for c in range(nc):
            N = float(n[:, c].sum())
            if N == 0:
                continue
            R = float(ret[:, c].sum()); H = float(hits[:, c].sum())
            by_comp[str(c)] = {"n_combos": int(N), "hits": int(H),
                               "hit_rate": round(H / N, 6),
                               "realized_roi": round(R / (UNIT * N), 4)}
        # prob bin × comp (n>=200 のみ)
        by_cell = []
        for bi in range(NB):
            for c in range(nc):
                N = n[bi, c]
                if N >= 200:
                    R = ret[bi, c]
                    by_cell.append({
                        "prob_lo": float(BINS[bi]), "prob_hi": float(BINS[bi + 1]),
                        "comp": c, "n": int(N), "hits": int(hits[bi, c]),
                        "realized_roi": round(float(R) / (UNIT * N), 4),
                    })
        return {"by_composition": by_comp, "by_probbin_comp": by_cell}

    out_payload = {"model_tag": tag, "lambda": lam,
                   "generated_at": datetime.now().isoformat(timespec="seconds"),
                   "unit": UNIT, "n_races": n_race,
                   "leak_safe": "selection by (model prob bin × composition) only; odds NOT used; payout for realized return only",
                   "composition_meaning": "number of constituent horses in model top3 (0..2 for pairs, 0..3 for triples)",
                   "note_full_period": "full=2013-2025 incl in-sample train scores; test=2024-25 OOS. sanrentan: test only (compute).",
                   "results": {}}
    for bt in NCOMP:
        out_payload["results"][bt] = {}
        for vr in ["raw", "cor"]:
            out_payload["results"][bt][vr] = {}
            for pp in PERIODS:
                if bt == "sanrentan" and pp == "full":
                    continue
                if cells[bt][vr][pp]["n"].sum() == 0:
                    continue
                out_payload["results"][bt][vr][pp] = summarize(cells[bt][vr][pp])

    out = BASE / "reports/joint_market_efficiency_fullperiod.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2, ensure_ascii=False)
    print(f"[saved] {out}")

    # ---------- 読みやすいサマリ (raw, full / sanrentan は test) ----------
    print("\n実現ROI by 構成層 (構成馬のモデルtop3頭数). raw binning:")
    for bt in NCOMP:
        pp = "test" if bt == "sanrentan" else "full"
        if pp not in out_payload["results"][bt]["raw"]:
            continue
        bc = out_payload["results"][bt]["raw"][pp]["by_composition"]
        period_tag = "(test OOS)" if bt == "sanrentan" else "(full 2013-25)"
        print(f"\n  [{bt}] {period_tag}  構成層: realized ROI (hit% / n)")
        for c in sorted(bc.keys()):
            d = bc[c]
            print(f"    top3={c}頭: ROI {d['realized_roi']*100:6.1f}%  "
                  f"(hit {d['hit_rate']*100:5.2f}% / n={d['n_combos']:,})")

    print("\n控除率超え (ROI>0.80) のセル (prob bin × comp, n>=200, raw):")
    found = False
    for bt in NCOMP:
        for pp in PERIODS:
            if pp not in out_payload["results"][bt]["raw"]:
                continue
            for cell in out_payload["results"][bt]["raw"][pp]["by_probbin_comp"]:
                if cell["realized_roi"] > 0.80:
                    found = True
                    print(f"  {bt}/{pp}: p∈[{cell['prob_lo']:.4f},{cell['prob_hi']:.4f}) "
                          f"comp={cell['comp']} ROI={cell['realized_roi']*100:.1f}% n={cell['n']:,}")
    if not found:
        print("  なし (全セル ROI<=0.80)")


if __name__ == "__main__":
    main()
