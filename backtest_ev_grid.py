# -*- coding: utf-8 -*-
"""
backtest_ev_grid.py — 期待値買いの精査: 会場×芝ダ×距離帯×券種×EV閾値 ROIグリッド
================================================================================
質問: 「フォメ/流し関係なく、期待値買い(単勝EV≥1.3, 馬連≥1.4, 馬単≥1.5...)で勝てるか?
       会場×距離×券種で相性の良い/悪いセルはあるか?」

正しくやらないと必ず嘘が出る分析(多重比較/forking paths)。なので:
  1. v6本番モデルで valid(2023) と test(2024-25) を別々に集計
  2. セル採用 = valid と test の両方で ROI≥100% かつ 最低レース数を満たす(OOS一貫性)
  3. レースクラスタ・ブートストラップで「100%超」が偶然でないか95%CI検定
  4. 多重比較: 検定セル数と「偶然なら何セル黒字に見えるはず」を併記

EV = p_cal(calibrated PL prob) × mean_pay(payout curve)  ＝ compute_bets/backtest_pl_ev と同一定義。
  選択=このEV、採点=kekka実配当(単位100円)。馬連等は全組合せオッズが歴史的に無いので
  payout-curve EVで選択し実配当で採点(プロジェクト既存手法)。

土台: backtest_pl_ev.py の関数を再利用(確率・EV・払戻ロジックのバグ混入を避ける)。
出力: reports/ev_grid_result.json / reports/ev_grid_summary.txt
実行: PYTHONUTF8=1 ./venv311/Scripts/python.exe backtest_ev_grid.py
"""
from __future__ import annotations
import io, json, os, sys, warnings
from itertools import combinations, permutations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import backtest_pl_ev as BT   # 確率/EV/払戻ロジック再利用
import pl_probs as PL

BASE = Path(__file__).parent
MASTER_CSV = BASE / "data/master_v2_20130105-20251228.csv"
V6_MODEL = BASE / "models/unified_rank_v6.pkl"
V6_CAL = BASE / "models/pl_calibrators_v6.pkl"
V6_CURVE = BASE / "data/pl_payout_curve_v6.pkl"
SCORE_CACHE = BASE / "data/_ev_grid_scores.parquet"
OUT_JSON = BASE / "reports/ev_grid_result.json"
OUT_TXT = BASE / "reports/ev_grid_summary.txt"

COL_RID = "レースID(新/馬番無)"; COL_JYUN = "着順"; COL_BAN = "馬番"
COL_VENUE = "場所"; COL_SURF = "芝・ダ"; COL_DIST = "距離"

BET = 100
SWEEP = [100, 110, 120, 130, 140, 150, 175, 200]          # EV閾値スイープ(×100=%, 130=1.3倍)
# ユーザー指定プリセット(券種ごとの実運用閾値)。*_mkt = 実オッズEV(単勝1.3, 複勝)
PRESET = {"tansho": 130, "fukusho": 110, "umaren": 140, "umatan": 150,
          "wide": 130, "sanrenpuku": 160, "sanrentan": 200,
          "tansho_mkt": 130, "fukusho_mkt": 110}
BETS7 = ["tansho", "fukusho", "umaren", "umatan", "wide", "sanrenpuku", "sanrentan"]
BETS_MKT = ["tansho_mkt", "fukusho_mkt"]          # 実オッズ(data/odds)ベースの value bet
ALL_BETS = BETS7 + BETS_MKT
ODDS_GLOB = "data/odds/odds_*.csv"
MIN_RACES = 30            # セル採用の最低レース数(valid と test 各々)
NBOOT = 2000
PAYKEY = "mean_pay"       # 期待値として正しい(median_payは過小)


def log(m): print(m, flush=True)


def dist_band(d):
    d = pd.to_numeric(d, errors="coerce")
    return pd.cut(d, [0, 1399, 1599, 1999, 9999],
                  labels=["Sprint", "Miler", "Middle", "Long"]).astype(str)


def load_odds():
    """全頭の確定オッズ {rid16: {ban: (tansho, fuku_lo, fuku_hi)}}. レースID(新)[:16]=rid16。"""
    import glob
    out = {}
    for f in sorted(glob.glob(str(BASE / ODDS_GLOB))):
        df = pd.read_csv(f, encoding="cp932", low_memory=False)
        rid = df["レースID(新)"].astype(str).str[:16]
        ban = pd.to_numeric(df["馬番"], errors="coerce")
        tan = pd.to_numeric(df["単勝オッズ"], errors="coerce")
        flo = pd.to_numeric(df["複勝オッズ下限"], errors="coerce")
        fhi = pd.to_numeric(df["複勝オッズ上限"], errors="coerce")
        for r, b, t, lo, hi in zip(rid.values, ban.values, tan.values, flo.values, fhi.values):
            if b != b:
                continue
            out.setdefault(r, {})[int(b)] = (float(t) if t == t else np.nan,
                                             float(lo) if lo == lo else np.nan,
                                             float(hi) if hi == hi else np.nan)
    return out


def score_all():
    """v6で valid+test をスコア化。スライス列(venue/surf/dist)は encode 前に退避。"""
    if SCORE_CACHE.exists():
        log(f"[cache] {SCORE_CACHE.name}")
        return pd.read_parquet(SCORE_CACHE)
    log("[score] v6 で valid+test をスコア化 ...")
    b = joblib.load(V6_MODEL)
    model, feats, encs = b["model"], b["feature_cols"], b["encoders"]
    rr = b.get("race_relative_mode"); ca = b.get("course_affinity_mode")
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df = df.dropna(subset=[COL_JYUN, COL_RID, "split"]).copy()
    df = df[df["split"].isin(["valid", "test"])].copy()
    if rr:
        from race_relative_feats import add_race_relative_feats
        df = add_race_relative_feats(df, mode=rr)
    if ca:
        from course_affinity_feats import add_course_affinity_feats
        df = add_course_affinity_feats(df, mode=ca)
    # スライス列退避(encode で整数化される前)
    out_slice = pd.DataFrame({
        "_venue": df[COL_VENUE].astype(str).values,
        "_surf": df[COL_SURF].astype(str).values,
        "_band": dist_band(df[COL_DIST]).values,
    }, index=df.index)
    df_enc = BT.apply_encoders(df, encs)
    X = df_enc[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    df["_score"] = model.predict(X)
    df["_venue"] = out_slice["_venue"]; df["_surf"] = out_slice["_surf"]; df["_band"] = out_slice["_band"]
    df["year"] = df[COL_RID].astype(str).str[:4].astype(int)
    keep = [COL_RID, COL_BAN, COL_JYUN, "split", "year", "_venue", "_surf", "_band", "_score"]
    sc = df[keep].copy()
    sc.to_parquet(SCORE_CACHE, index=False)
    log(f"  rows={len(sc):,}  races={sc[COL_RID].nunique():,}  [saved cache]")
    return sc


def simulate(te, payouts, widepays, cal, curves, oddsmap):
    """1期間(valid or test)を回す。
    返り値:
      agg[(cell,bet,thr)] = dict(bets,hits,stake,ret,sumsq)   ※cell='ALL' と '会場|芝ダ|距離帯'
      perrace[(cell,bet)] = list[(stake,ret)]  ※PRESET閾値での race集計(bootstrap用)
    """
    has_wide = ("wide" in cal) and ("wide" in curves)
    agg = {}
    perrace = {}

    def acc(cell, bet, thr, nb, nh, ret, sumsq):
        k = (cell, bet, thr)
        d = agg.get(k)
        if d is None:
            d = {"bets": 0, "hits": 0, "stake": 0, "ret": 0.0, "sumsq": 0.0}; agg[k] = d
        d["bets"] += nb; d["hits"] += nh; d["stake"] += nb * BET; d["ret"] += ret; d["sumsq"] += sumsq

    def acc_race(cell, bet, stake, ret):
        if stake <= 0: return
        for c in (cell, "ALL"):
            perrace.setdefault((c, bet), []).append((stake, ret))

    n_race = 0; n_skip = 0
    for rid, g in te.groupby(COL_RID, sort=False):
        if len(g) < 3: n_skip += 1; continue
        rid_s = str(int(rid)) if isinstance(rid, (int, np.integer)) else str(rid)
        if rid_s not in payouts: n_skip += 1; continue
        po = payouts[rid_s]
        g = g.sort_values(COL_BAN).reset_index(drop=True)
        ban = g[COL_BAN].astype(int).values
        b2i = {int(b): i for i, b in enumerate(ban)}
        wi = b2i.get(po["win"]); pi_ = b2i.get(po["plc"]); si = b2i.get(po["sho"])
        if wi is None or pi_ is None or si is None: n_skip += 1; continue
        w = PL.pl_weights(g["_score"].values); n = len(w)
        cell = f"{g['_venue'].iloc[0]}|{g['_surf'].iloc[0]}|{g['_band'].iloc[0]}"
        n_race += 1

        def run(bet, ev, hit_arr, pay_arr):
            """ev/hit_arr/pay_arr 同形。pay_arr=各候補の実配当(当たり時)。"""
            for thr in SWEEP:
                gate = ev >= thr
                nb = int(gate.sum())
                if nb == 0:
                    continue
                hm = gate & hit_arr
                ret = float((pay_arr * hm).sum())
                sumsq = float(((pay_arr * hm) ** 2).sum())
                nh = int(hm.sum())
                acc(cell, bet, thr, nb, nh, ret, sumsq)
                acc("ALL", bet, thr, nb, nh, ret, sumsq)
            # PRESET: race集計
            thr = PRESET[bet]; gate = ev >= thr
            if gate.sum() > 0:
                hm = gate & hit_arr
                acc_race(cell, bet, int(gate.sum()) * BET, float((pay_arr * hm).sum()))

        # 単勝
        p = cal["tansho"].predict(w / w.sum())
        ev = p * BT.lookup_pay_vec(curves["tansho"], p, PAYKEY)
        hit = np.zeros(n, bool); hit[wi] = True
        pay = np.zeros(n); pay[wi] = _f(po["tansho"])
        run("tansho", ev, hit, pay)

        # 複勝
        p = cal["fukusho"].predict(BT.all_fukusho_vec_fast(w))
        ev = p * BT.lookup_pay_vec(curves["fukusho"], p, PAYKEY)
        hit = np.zeros(n, bool); hit[[wi, pi_, si]] = True
        pay = np.zeros(n)
        for idx, k in [(wi, "fuku_win"), (pi_, "fuku_plc"), (si, "fuku_sho")]:
            pay[idx] = _f(po[k])
        run("fukusho", ev, hit, pay)

        # ===== 実オッズ value bet (単勝/複勝) =====
        od = oddsmap.get(rid_s)
        if od:
            tan = np.array([od.get(int(b), (np.nan, np.nan, np.nan))[0] for b in ban])
            flo = np.array([od.get(int(b), (np.nan, np.nan, np.nan))[1] for b in ban])
            fhi = np.array([od.get(int(b), (np.nan, np.nan, np.nan))[2] for b in ban])
            mid = (flo + fhi) / 2.0
            # 単勝_mkt: EV(×100) = p_win × 実単勝オッズ × 100
            p_win = cal["tansho"].predict(w / w.sum())
            ev = p_win * tan * 100.0
            hit = np.zeros(n, bool); hit[wi] = True
            pay = np.zeros(n); pay[wi] = _f(po["tansho"])
            run("tansho_mkt", np.nan_to_num(ev, nan=-1.0), hit, pay)
            # 複勝_mkt: EV = p_sho × 実複勝中央オッズ × 100, 採点=実複勝配当
            p_sho = cal["fukusho"].predict(BT.all_fukusho_vec_fast(w))
            ev = p_sho * mid * 100.0
            hit = np.zeros(n, bool); hit[[wi, pi_, si]] = True
            pay = np.zeros(n)
            for idx, k in [(wi, "fuku_win"), (pi_, "fuku_plc"), (si, "fuku_sho")]:
                pay[idx] = _f(po[k])
            run("fukusho_mkt", np.nan_to_num(ev, nan=-1.0), hit, pay)

        # 馬連
        Pm = BT.all_umaren_mat(w); iu, ju = np.triu_indices(n, k=1)
        p = cal["umaren"].predict(Pm[iu, ju])
        ev = p * BT.lookup_pay_vec(curves["umaren"], p, PAYKEY)
        wp = {wi, pi_}
        hit = np.array([{int(iu[k]), int(ju[k])} == wp for k in range(len(iu))])
        pay = np.where(hit, _f(po["umaren"]), 0.0)
        run("umaren", ev, hit, pay)

        # 馬単
        Um = BT.all_umatan_mat(w); ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        m = ii != jj; p = cal["umatan"].predict(Um[m])
        ev = p * BT.lookup_pay_vec(curves["umatan"], p, PAYKEY)
        hit = (ii[m] == wi) & (jj[m] == pi_)
        pay = np.where(hit, _f(po["umatan"]), 0.0)
        run("umatan", ev, hit, pay)

        # ワイド
        if has_wide:
            Wm = BT.all_wide_mat(w)
            p = cal["wide"].predict(Wm[iu, ju])
            ev = p * BT.lookup_pay_vec(curves["wide"], p, PAYKEY)
            wpairs = widepays.get(rid_s, [])
            paymap = {}
            for (bi, bj, pv) in wpairs:
                a, bb = b2i.get(bi), b2i.get(bj)
                if a is not None and bb is not None:
                    paymap[frozenset((a, bb))] = pv
            hit = np.zeros(len(iu), bool); pay = np.zeros(len(iu))
            for k in range(len(iu)):
                fs = frozenset((int(iu[k]), int(ju[k])))
                if fs in paymap:
                    hit[k] = True; pay[k] = paymap[fs]
            run("wide", ev, hit, pay)

        # 三連単/三連複
        P3 = BT.all_sanrenpuku_tensor(w); idx = np.arange(n)
        I, J, K = np.meshgrid(idx, idx, idx, indexing="ij")
        m3 = (I != J) & (J != K) & (I != K)
        p = cal["sanrentan"].predict(P3[m3])
        ev = p * BT.lookup_pay_vec(curves["sanrentan"], p, PAYKEY)
        hit = (I[m3] == wi) & (J[m3] == pi_) & (K[m3] == si)
        pay = np.where(hit, _f(po["sanrentan"]), 0.0)
        run("sanrentan", ev, hit, pay)

        tris = np.array(list(combinations(range(n), 3)))
        if len(tris):
            pk = np.zeros(len(tris))
            for perm in permutations(range(3)):
                pk += P3[tris[:, perm[0]], tris[:, perm[1]], tris[:, perm[2]]]
            p = cal["sanrenpuku"].predict(pk)
            ev = p * BT.lookup_pay_vec(curves["sanrenpuku"], p, PAYKEY)
            wt = {wi, pi_, si}
            hit = np.array([set(t) == wt for t in tris])
            pay = np.where(hit, _f(po["sanrenpuku"]), 0.0)
            run("sanrenpuku", ev, hit, pay)

        if n_race % 1000 == 0:
            log(f"    ..{n_race} races")
    log(f"  [done] races={n_race} skipped={n_skip} wide={'on' if has_wide else 'OFF'}")
    return agg, perrace


def _f(x):
    try:
        v = float(x); return v if v == v else 0.0
    except Exception:
        return 0.0


def boot_ci(pairs, nboot=NBOOT, seed=42):
    """レースクラスタ bootstrap. pairs=list[(stake,ret)] per race. 戻り: (roi, lo, hi)."""
    if not pairs:
        return (0.0, 0.0, 0.0)
    st = np.array([p[0] for p in pairs], float); rt = np.array([p[1] for p in pairs], float)
    roi = rt.sum() / st.sum() if st.sum() > 0 else 0.0
    rng = np.random.default_rng(seed); m = len(pairs); rois = np.empty(nboot)
    for b in range(nboot):
        idx = rng.integers(0, m, m)
        s = st[idx].sum()
        rois[b] = rt[idx].sum() / s if s > 0 else 0.0
    return (float(roi), float(np.percentile(rois, 2.5)), float(np.percentile(rois, 97.5)))


def main():
    OUT_JSON.parent.mkdir(exist_ok=True)
    payouts = BT.load_payouts(); log(f"[payouts] {len(payouts):,} races")
    try:
        widepays = BT.load_wide_payouts(); log(f"[wide] {len(widepays):,} races")
    except Exception as e:
        widepays = {}; log(f"[wide] load failed: {e}")
    cal = joblib.load(V6_CAL)["calibrators"]
    curves = joblib.load(V6_CURVE)["curves"]
    log(f"[cal] keys={list(cal.keys())}")
    oddsmap = load_odds(); log(f"[odds] {len(oddsmap):,} races with all-horse odds")
    sc = score_all()

    smoke = os.environ.get("EV_GRID_SMOKE")
    if smoke:
        keep_rids = sc[COL_RID].drop_duplicates().sample(frac=float(smoke), random_state=42)
        sc = sc[sc[COL_RID].isin(keep_rids)].copy()
        log(f"[SMOKE] frac={smoke} races={sc[COL_RID].nunique():,}")

    periods = {}
    for name, mask in [("valid", sc["split"] == "valid"), ("test", sc["year"].isin([2024, 2025]))]:
        log(f"\n=== simulate {name} ===")
        agg, perrace = simulate(sc[mask], payouts, widepays, cal, curves, oddsmap)
        periods[name] = {"agg": agg, "perrace": perrace}

    # ---------- 1) 全体(ALL)券種別: 期待値買いそのものの成否 ----------
    overall = {}
    for bet in ALL_BETS:
        row = {}
        for thr in SWEEP:
            v = periods["valid"]["agg"].get(("ALL", bet, thr))
            t = periods["test"]["agg"].get(("ALL", bet, thr))
            row[thr] = {
                "valid_roi": round(v["ret"] / v["stake"], 4) if v and v["stake"] else None,
                "valid_bets": v["bets"] if v else 0,
                "test_roi": round(t["ret"] / t["stake"], 4) if t and t["stake"] else None,
                "test_bets": t["bets"] if t else 0,
            }
        overall[bet] = row

    # ---------- 2) セルグリッド: valid∩test 両黒字 + bootstrap ----------
    cells = sorted({k[0] for k in periods["test"]["agg"] if k[0] != "ALL"})
    survivors = []; tested = 0
    grid = {}
    for cell in cells:
        for bet in ALL_BETS:
            thr = PRESET[bet]
            v = periods["valid"]["agg"].get((cell, bet, thr))
            t = periods["test"]["agg"].get((cell, bet, thr))
            if not v or not t: continue
            vr = v["ret"] / v["stake"] if v["stake"] else 0.0
            tr = t["ret"] / t["stake"] if t["stake"] else 0.0
            # レース数(perrace の長さ)
            v_races = len(periods["valid"]["perrace"].get((cell, bet), []))
            t_races = len(periods["test"]["perrace"].get((cell, bet), []))
            if v_races < MIN_RACES or t_races < MIN_RACES:
                continue
            tested += 1
            entry = {"cell": cell, "bet": bet, "preset_ev": thr,
                     "valid_roi": round(vr, 4), "valid_races": v_races, "valid_bets": v["bets"],
                     "test_roi": round(tr, 4), "test_races": t_races, "test_bets": t["bets"]}
            grid[f"{cell}#{bet}"] = entry
            if vr >= 1.0 and tr >= 1.0:
                roi, lo, hi = boot_ci(periods["test"]["perrace"][(cell, bet)])
                entry["test_boot_roi"] = round(roi, 4)
                entry["test_ci95"] = [round(lo, 4), round(hi, 4)]
                entry["significant"] = bool(lo > 1.0)
                survivors.append(entry)

    survivors.sort(key=lambda e: e["test_roi"], reverse=True)

    # ---------- 3) ALL券種の preset bootstrap(全体EV買いの結論) ----------
    overall_boot = {}
    for bet in ALL_BETS:
        pr = periods["test"]["perrace"].get(("ALL", bet), [])
        roi, lo, hi = boot_ci(pr) if pr else (0, 0, 0)
        vpr = periods["valid"]["perrace"].get(("ALL", bet), [])
        vroi = (sum(r for _, r in vpr) / sum(s for s, _ in vpr)) if vpr else 0.0
        overall_boot[bet] = {"preset_ev": PRESET[bet], "valid_roi": round(vroi, 4),
                             "test_roi": round(roi, 4), "test_ci95": [round(lo, 4), round(hi, 4)],
                             "test_races": len(pr), "significant_over_breakeven": bool(lo > 1.0)}

    # ---------- 多重比較 ----------
    n_signif = sum(1 for e in survivors if e.get("significant"))
    expected_fp_05 = round(0.05 * tested, 1)   # α=5%片側でも目安
    result = {
        "protocol": "v6本番 / valid(2023)・test(2024-25)別 / EV=p_cal×mean_pay(curve) / 採点=kekka実配当 / レースクラスタbootstrap",
        "ev_definition": "EV(×100) = calibrated_PL_prob × payout_curve_mean_pay。SWEEP=%閾値。PRESET=ユーザー指定券種別閾値。",
        "preset_thresholds": PRESET, "min_races_per_period": MIN_RACES,
        "overall_by_bet_threshold": overall,
        "overall_preset_bootstrap": overall_boot,
        "cells_tested": tested,
        "survivors_valid_and_test_profitable": survivors,
        "n_survivors": len(survivors),
        "n_survivors_significant_ci": n_signif,
        "expected_false_positives_at_5pct": expected_fp_05,
        "multiple_testing_note": f"{tested}セル検定。偶然でも約{expected_fp_05}セルがα=5%で黒字に見えうる。生存{len(survivors)}・CI有意{n_signif}と比較せよ。",
    }
    json.dump(result, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    log(f"\n[saved] {OUT_JSON}")

    # ---------- サマリ出力 ----------
    lines = []
    def P(s): lines.append(s); log(s)
    P("=" * 78)
    P("期待値買い 精査サマリ (v6 / valid=2023 / test=2024-25 / フォメ無視・単純EV閾値買い)")
    P("=" * 78)
    P("\n■ 全体(全会場全距離) 券種別 ROI ― 期待値買いそのものの成否")
    P(f"{'券種':10s} {'EV閾値':>6s} | {'valid ROI':>10s} {'bets':>8s} | {'test ROI':>10s} {'bets':>8s}")
    for bet in ALL_BETS:
        for thr in SWEEP:
            r = overall[bet][thr]
            vr = f"{r['valid_roi']*100:.1f}%" if r['valid_roi'] is not None else "-"
            tr = f"{r['test_roi']*100:.1f}%" if r['test_roi'] is not None else "-"
            star = " <preset" if thr == PRESET[bet] else ""
            P(f"{bet:10s} {thr:>6d} | {vr:>10s} {r['valid_bets']:>8,} | {tr:>10s} {r['test_bets']:>8,}{star}")
    P("\n■ プリセット閾値での全体ROI + test 95%CI(レースクラスタbootstrap)")
    for bet in ALL_BETS:
        b = overall_boot[bet]
        P(f"  {bet:10s} EV≥{b['preset_ev']:>3d}  valid={b['valid_roi']*100:6.1f}%  "
          f"test={b['test_roi']*100:6.1f}%  CI95=[{b['test_ci95'][0]*100:.1f}%,{b['test_ci95'][1]*100:.1f}%]  "
          f"races={b['test_races']:,}  {'★控除超有意' if b['significant_over_breakeven'] else '床以下'}")
    P(f"\n■ セルグリッド(会場×芝ダ×距離帯×券種, preset閾値): 検定{tested}セル")
    P(f"  valid∩test 両方ROI≥100% の生存セル: {len(survivors)}")
    P(f"  うち test 95%CI下限>100% (偶然でない): {n_signif}")
    P(f"  多重比較の目安: 偶然でも約 {expected_fp_05} セルが黒字に見えうる")
    if survivors:
        P(f"\n  --- 生存セル(test ROI降順, 上位20) ---")
        P(f"  {'セル(会場|芝ダ|距離帯)':28s} {'券種':10s} {'vROI':>7s} {'tROI':>7s} {'CI95下限':>8s} {'tR数':>5s} 有意")
        for e in survivors[:20]:
            ci = e.get("test_ci95", [0, 0])
            P(f"  {e['cell']:28s} {e['bet']:10s} {e['valid_roi']*100:6.1f}% {e['test_roi']*100:6.1f}% "
              f"{ci[0]*100:7.1f}% {e['test_races']:>5d} {'★' if e.get('significant') else ''}")
    else:
        P("\n  生存セル: なし。期待値買いで控除率を超えるセルは(OOS一貫性下で)存在しない。")
    P("=" * 78)
    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
    log(f"[saved] {OUT_TXT}")


if __name__ == "__main__":
    main()
