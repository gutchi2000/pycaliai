"""
gate_q2_umaren_priceedge.py
===========================
真の問2: 前売り馬連オッズ × v6較正PL確率 の「価格ズレ」で FLB をハーベストできるか、
13年・leak-safe でバックテスト。

意思決定 = 当日9時(区分1) 馬連odds。自分の確率 = v6較正PL p_ij (+ gate1 β補正の変種)。
EV_ij = p_ij × odds_ij(当日9時)。EV>閾値 のペアを buy（価格ズレ＝過小配当の本命集中を拾い穴を外す）。
決済 = 確定(区分4) 馬連odds（pari-mutuel: 前売りで賭けても配当は確定）。ユニット・フラット。

leak規約: 意思決定は前売り(区分1)のみ。確定(区分4)は決済とCLVのみ。
  EV閾値の選定は train+valid(≤2023)のみ。test(2024-25)は評価のみ（test閾値最適化禁止）。
対照(control): prob-only top-K（価格を見ない）。価格情報の寄与を分離。
CLV診断: 当日9時odds vs 確定odds（pari-mutuelなので先行指標）。

出力: reports/gate_q2_umaren_priceedge.json
実行: python gate_q2_umaren_priceedge.py
"""
from __future__ import annotations
import glob, json, re, sys
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import pl_probs as PL
from corr_features import add_style
from joint_calibration_v6 import apply_encoders, COL_RID, COL_JYUN, COL_BAN, MASTER_CSV

BASE = Path(__file__).parent
np.random.seed(42)
ODIR = BASE / "data" / "Time _series_odds"
EV_GRID = [1.0, 1.05, 1.1, 1.2, 1.3, 1.5, 2.0]
KS = [3, 5, 7]
MIN_TUNE_BETS = 500  # ≤2023 で閾値を選ぶ際の最小ベット数


# ---------- 馬連 時系列オッズの読み込み & snapshot 抽出 ----------
def load_umaren_snapshots():
    files = sorted(glob.glob(str(ODIR / "UMAREN_*.csv")))
    dfs = []
    for f in files:
        d = pd.read_csv(f, encoding="cp932", low_memory=False)
        dfs.append(d)
    um = pd.concat(dfs, ignore_index=True)
    cols = list(um.columns)
    RID, KB, TM, TOU = cols[0], cols[1], cols[2], cols[3]
    pair_cols = {}
    for c in cols[5:]:
        m = re.search(r"(\d+)\s*-\s*(\d+)", str(c))
        if m:
            pair_cols[(int(m.group(1)), int(m.group(2)))] = c
    um["rid16"] = um[RID].astype(str).str.replace(r"\D", "", regex=True).str[:16]
    um[KB] = pd.to_numeric(um[KB], errors="coerce")
    um[TM] = pd.to_numeric(um[TM], errors="coerce")
    um = um[um["rid16"].str.len() == 16]
    um = um[um["rid16"].str[:4].astype(int) >= 2013]
    pair_items = list(pair_cols.items())  # [((i,j), colname), ...]

    def extract(row, tou):
        out = {}
        for (i, j), c in pair_items:
            if i <= tou and j <= tou:
                v = row[c]
                v = pd.to_numeric(v, errors="coerce") if not isinstance(v, (int, float, np.floating)) else v
                if pd.notna(v) and v > 1.0:
                    out[(i, j)] = float(v)
        return out

    snaps = {}
    for rid, g in um.groupby("rid16", sort=False):
        try:
            tou = int(pd.to_numeric(g[TOU], errors="coerce").dropna().iloc[0])
        except Exception:
            continue
        race_mmdd = int(rid[4:8])
        g1 = g[g[KB] == 1]
        g4 = g[g[KB] == 4]
        rec = {"tou": tou}
        if len(g1) > 0:
            hh = (g1[TM] % 10000)
            mmdd = (g1[TM] // 10000)
            # 当日9時: race日(mmdd一致)優先で HHMM 最寄0900
            same = g1[mmdd == race_mmdd]
            base = same if len(same) > 0 else g1
            am9 = base.loc[(hh.loc[base.index] - 900).abs().idxmin()]
            last = g1.loc[g1[TM].idxmax()]  # 直前 = 月日時分 最大
            rec["am9"] = extract(am9, tou)
            rec["last"] = extract(last, tou)
            rec["am9_time"] = int(am9[TM]); rec["last_time"] = int(last[TM])
        if len(g4) > 0:
            fin = g4.loc[g4[TM].idxmax()]
            rec["fin"] = extract(fin, tou)
        if "am9" in rec and "fin" in rec:
            snaps[rid] = rec
    return snaps


# ---------- v6 確率 ----------
def score_master():
    b = joblib.load(BASE / "models/unified_rank_v6.pkl")
    model, feats, encs = b["model"], b["feature_cols"], b["encoders"]
    cal = joblib.load(BASE / "models/pl_calibrators_v6.pkl")
    cal = cal.get("calibrators", cal)
    beta_d = json.load(open(BASE / "reports/gate1_corr_calibration.json", encoding="utf-8"))["beta"]
    from corr_features import PAIR_FEATURE_NAMES
    beta = np.array([beta_d[n] for n in PAIR_FEATURE_NAMES])
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df = df.dropna(subset=[COL_JYUN, COL_RID, "split"]).copy()
    df = add_style(df)
    enc = apply_encoders(df, encs)
    X = enc[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    df["_score"] = model.predict(X)
    return df, cal, beta


def build_race(g, cal, beta):
    """master 1レース → 馬番ペア→(p_raw, p_beta), winning pair, 頭数。"""
    from backtest_pl_ev import all_umaren_mat
    from gate1_eval1 import pair_feat_tensor
    g = g.sort_values(COL_BAN).reset_index(drop=True)
    ban = g[COL_BAN].astype(int).values
    n = len(ban)
    w = PL.pl_weights(g["_score"].values)
    M = all_umaren_mat(w)
    iu, ju = np.triu_indices(n, 1)
    praw = M[iu, ju]
    praw_c = cal["umaren"].predict(praw) if "umaren" in cal else praw
    # β 補正
    style = g["style_rank"].values.astype(float)
    waku = pd.to_numeric(g["枠番"], errors="coerce").fillna(0).values
    g_rid = str(g[COL_RID].iloc[0])
    pace = float(g["pace_pressure"].iloc[0]) if "pace_pressure" in g.columns else 0.33
    F = pair_feat_tensor(style, waku, pace, n)
    corr = (F @ beta)[iu, ju]
    pb = praw * np.exp(corr); pb = pb / pb.sum() * praw.sum()
    pb_c = cal["umaren"].predict(pb) if "umaren" in cal else pb
    jyun = g[COL_JYUN].astype(int).values
    order = np.argsort(jyun)
    _b1, _b2 = int(ban[order[0]]), int(ban[order[1]])
    win_pair = (min(_b1, _b2), max(_b1, _b2))  # 馬番タプル (他のペアキーと型を揃える)
    p_raw = {}; p_beta = {}
    for k in range(len(iu)):
        key = (int(ban[iu[k]]), int(ban[ju[k]]))
        key = (min(key), max(key))
        p_raw[key] = float(praw_c[k]); p_beta[key] = float(pb_c[k])
    return p_raw, p_beta, win_pair, n


# ---------- バックテスト ----------
def run():
    print("[load] UMAREN 時系列オッズ (3ファイル結合, snapshot抽出)...")
    snaps = load_umaren_snapshots()
    print(f"  races with 当日9時+確定: {len(snaps):,}")
    print("[score] v6 PL on master...")
    df, cal, beta = score_master()

    # レースごとに 意思決定材料を作る
    races = {}
    for rid, g in df.groupby(COL_RID, sort=False):
        if len(g) < 5:
            continue
        rid16 = str(rid)[:16]
        if rid16 not in snaps:
            continue
        p_raw, p_beta, win_pair, n = build_race(g, cal, beta)
        sp = snaps[rid16]
        if win_pair not in sp.get("fin", {}):
            # 確定オッズに winning pair が無い→決済不能（同着等）→スキップ
            continue
        races[rid16] = {"p_raw": p_raw, "p_beta": p_beta, "win_pair": win_pair,
                        "am9": sp["am9"], "last": sp.get("last", sp["am9"]),
                        "fin": sp["fin"], "year": int(rid16[:4]),
                        "split": g["split"].iloc[0]}
    print(f"  backtest対象レース: {len(races):,}")

    def settle(rd, sel_pairs):
        """sel_pairs(馬番frozenset集合) を確定oddsでフラット決済。"""
        stake = len(sel_pairs) * 100.0
        ret = 0.0; hits = 0
        clv_num = []
        for pr in sel_pairs:
            if pr == rd["win_pair"]:
                ret += rd["fin"].get(pr, 0.0) * 100.0  # 確定odds×100 = 配当
                hits += 1
            # CLV: 当日9時 vs 確定 (我々が触れたペアのオッズ変化)
            o9 = rd["am9"].get(pr); of = rd["fin"].get(pr)
            if o9 and of and of > 0:
                clv_num.append(o9 / of - 1.0)  # >0 = 当日9時の方が高odds=締まった=賢い金
        return stake, ret, hits, clv_num

    def select_ev(rd, pkey, oddskey, thr):
        p = rd[pkey]; od = rd[oddskey]
        sel = set()
        for pr, o in od.items():
            if pr in p and p[pr] * o >= thr:
                sel.add(pr)
        return sel

    def select_topk(rd, pkey, k):
        p = rd[pkey]; od = rd["am9"]
        # odds がある(=実在の)ペアのみ候補に
        cand = [(pr, p[pr]) for pr in od.keys() if pr in p]
        cand.sort(key=lambda x: -x[1])
        return set(pr for pr, _ in cand[:k])

    def agg(selector, period_filter):
        stake = ret = hits = bets = 0.0; clv = []
        for rid, rd in races.items():
            if not period_filter(rd):
                continue
            sel = selector(rd)
            if not sel:
                continue
            s, r, h, c = settle(rd, sel)
            stake += s; ret += r; hits += h; bets += len(sel); clv += c
        roi = ret / stake if stake else 0.0
        return {"roi": round(roi, 4), "bets": int(bets),
                "hit_rate": round(hits / bets, 4) if bets else 0.0,
                "n_clv": len(clv), "mean_clv": round(float(np.mean(clv)), 4) if clv else None}

    le2023 = lambda rd: rd["year"] <= 2023
    test = lambda rd: rd["year"] >= 2024
    y2024 = lambda rd: rd["year"] == 2024
    y2025 = lambda rd: rd["year"] == 2025

    out = {"ev_grid": EV_GRID, "ks": KS, "n_races": len(races),
           "note": "decision=当日9時 馬連odds; settle=確定odds; tune τ on ≤2023, eval test only.",
           "sweeps": {}, "control": {}, "chosen": {}, "oracle": {}}

    # EV sweep: pversion × decision point × τ, on ≤2023 と test
    for pver, pkey in [("raw", "p_raw"), ("beta", "p_beta")]:
        for dpt, okey in [("am9", "am9"), ("last", "last")]:
            key = f"{pver}_{dpt}"
            out["sweeps"][key] = {"le2023": {}, "test": {}, "test_2024": {}, "test_2025": {}}
            for thr in EV_GRID:
                sel = lambda rd, t=thr, pk=pkey, ok=okey: select_ev(rd, pk, ok, t)
                out["sweeps"][key]["le2023"][str(thr)] = agg(sel, le2023)
                out["sweeps"][key]["test"][str(thr)] = agg(sel, test)
                out["sweeps"][key]["test_2024"][str(thr)] = agg(sel, y2024)
                out["sweeps"][key]["test_2025"][str(thr)] = agg(sel, y2025)
            # ≤2023 で τ* 選定 (n>=MIN_TUNE_BETS の中で ROI 最大) → test 適用
            cand = [(thr, out["sweeps"][key]["le2023"][str(thr)]) for thr in EV_GRID
                    if out["sweeps"][key]["le2023"][str(thr)]["bets"] >= MIN_TUNE_BETS]
            if cand:
                tstar = max(cand, key=lambda x: x[1]["roi"])[0]
                out["chosen"][key] = {"tau_star": tstar,
                                      "le2023": out["sweeps"][key]["le2023"][str(tstar)],
                                      "test": out["sweeps"][key]["test"][str(tstar)],
                                      "test_2024": out["sweeps"][key]["test_2024"][str(tstar)],
                                      "test_2025": out["sweeps"][key]["test_2025"][str(tstar)]}

    # control: prob-only top-K（価格無視）, raw, test
    for k in KS:
        sel = lambda rd, kk=k: select_topk(rd, "p_raw", kk)
        out["control"][f"raw_topK{k}"] = {"le2023": agg(sel, le2023), "test": agg(sel, test)}

    # oracle: 確定oddsでEV選択 (上限の目安, leak)
    for thr in [1.0, 1.2, 1.5]:
        sel = lambda rd, t=thr: select_ev(rd, "p_raw", "fin", t)
        out["oracle"][f"fin_ev_{thr}"] = {"test": agg(sel, test)}

    outp = BASE / "reports/gate_q2_umaren_priceedge.json"
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[saved] {outp}")

    # ---- サマリ ----
    print("\n=== EV(価格ズレ)選択: ≤2023で選んだτ* を test 適用 ===")
    print(f"{'p×決定点':<14s} {'τ*':>5s} {'≤2023 ROI':>10s} {'test ROI':>9s} {'2024':>7s} {'2025':>7s} {'test n':>8s} {'CLV':>7s}")
    for key, c in out["chosen"].items():
        print(f"{key:<14s} {c['tau_star']:>5.2f} {c['le2023']['roi']*100:>9.1f}% "
              f"{c['test']['roi']*100:>8.1f}% {c['test_2024']['roi']*100:>6.1f}% "
              f"{c['test_2025']['roi']*100:>6.1f}% {c['test']['bets']:>8,} "
              f"{(c['test']['mean_clv'] if c['test']['mean_clv'] is not None else 0)*100:>+6.1f}%")
    print("\n=== 対照 prob-only top-K (test) ===")
    for k in KS:
        d = out["control"][f"raw_topK{k}"]["test"]
        print(f"  topK={k}: ROI {d['roi']*100:.1f}%  hit {d['hit_rate']*100:.1f}%  n={d['bets']:,}")
    print("\n=== oracle (確定odds EV選択, 上限目安) test ===")
    for thr in [1.0, 1.2, 1.5]:
        d = out["oracle"][f"fin_ev_{thr}"]["test"]
        print(f"  fin_EV>{thr}: ROI {d['roi']*100:.1f}%  n={d['bets']:,}")
    print("\n控除率ライン: 馬連 77.5%")


if __name__ == "__main__":
    run()
