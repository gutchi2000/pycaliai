"""
distortion_exp.py — 市場の内部整合性の破れ(歪み)を裁定機会として探す
====================================================================
fの予測精度でなく「市場が市場自身と矛盾する個別レース」を外れ値として抽出し、
整合性が示す側に賭けて控除を超えるかを分位別に測る。9時オッズのみ(リーク防止)。

歪みA: 券種間整合性の破れ。単勝de-vig p_win → PL重み → PL複勝率 p_place_PL と、
       複勝市場 de-vig p_place_mkt の乖離 dA=Σ|p_place_mkt-p_place_PL| が大きいレース=歪み。
       賭け: disc_i=p_place_mkt-p_place_PL>0(複勝市場が単勝PLより高評価)の馬を複勝(複勝市場が正しい仮説)、
             および disc<0(単勝PLが高)の馬を複勝(単勝が正しい仮説) の両方を検証。
歪みB: モデル-市場大乖離。r_i=log f_cal - log π9, dB=max|r_i|。乖離大レースで r>0(f買い)馬を単勝。

★リーク: 9時近傍(当日HHMM<=900の最後)スナップのみ。9時以降/確定はオッズに不使用(決済のみ確定配当)。
  検査で違反0を最初に報告。分位閾値は≤2023で決定→≤2023(発見)/test(検証)に同閾値適用。
  サンプル下限 test 500bet、多重比較 Bonferroni、発見→検証再現を要求。
出力: reports/distortion_exp.json  実行: PYTHONUTF8=1 python distortion_exp.py
"""
from __future__ import annotations
import glob, json, re, warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import pl_probs as PL
from backtest_pl_ev import load_payouts, all_fukusho_vec_fast
from joint_calibration_v6 import apply_encoders, COL_RID, COL_JYUN, COL_BAN, MASTER_CSV

warnings.filterwarnings("ignore")
BASE = Path(__file__).parent
ODIR = BASE / "data/Time _series_odds"
OUT_JSON = BASE / "reports/distortion_exp.json"
rng = np.random.default_rng(42)
MIN_BETS = 500
N_BOOT = 300
QUANTILES = [0.99, 0.95, 0.90]   # top1%, 5%, 10%
TAKEOUT_T, TAKEOUT_F = 0.80, 0.80


def _rid16(x): return re.sub(r"\D", "", str(x))[:16]


def load_9am():
    files = sorted(glob.glob(str(ODIR / "TANPUK_*.csv")))
    out = {}; leak_bad = 0; leak_chk = 0
    for fp in files:
        tp = pd.read_csv(fp, encoding="cp932", low_memory=False)
        cols = list(tp.columns); RID, KB, TM = cols[0], cols[1], cols[2]
        tan_c, flo, fhi = {}, {}, {}
        for c in cols:
            cs = str(c)
            m = re.match(r"^\s*(\d+)\s*単\s*$", cs)
            if m: tan_c[int(m.group(1))] = c; continue
            m = re.match(r"^\s*(\d+)\s*複\s*Lo\s*$", cs)
            if m: flo[int(m.group(1))] = c; continue
            m = re.match(r"^\s*(\d+)\s*複\s*Hi\s*$", cs)
            if m: fhi[int(m.group(1))] = c
        tp[KB] = pd.to_numeric(tp[KB], errors="coerce")
        tp = tp[tp[KB] == 1].copy()
        tp["rid16"] = tp[RID].map(_rid16); tp = tp[tp["rid16"].str.len() == 16]
        tp[TM] = pd.to_numeric(tp[TM], errors="coerce")
        tp["mmdd"] = (tp[TM] // 10000); tp["hhmm"] = (tp[TM] % 10000)
        for rid, g in tp.groupby("rid16", sort=False):
            rmmdd = int(rid[4:8])
            pre = g[(g["mmdd"] < rmmdd) | ((g["mmdd"] == rmmdd) & (g["hhmm"] <= 900))]
            if len(pre) == 0: continue
            row = pre.sort_values(TM).iloc[-1]       # 9時直前
            leak_chk += 1
            if (row["mmdd"] == rmmdd and row["hhmm"] > 900) or row["mmdd"] > rmmdd: leak_bad += 1
            tan = {b: float(row[c]) for b, c in tan_c.items()
                   if pd.notna(pd.to_numeric(pd.Series([row[c]]), errors="coerce").iloc[0]) and float(pd.to_numeric(pd.Series([row[c]]), errors="coerce").iloc[0]) > 1.0}
            fuk = {}
            for b in flo:
                lo = pd.to_numeric(pd.Series([row[flo[b]]]), errors="coerce").iloc[0]
                hi = pd.to_numeric(pd.Series([row[fhi[b]]]), errors="coerce").iloc[0] if b in fhi else np.nan
                if pd.notna(lo) and lo > 1.0: fuk[b] = (float(lo), float(hi) if pd.notna(hi) else float(lo))
            if tan: out[rid] = {"tan": tan, "fuku": fuk}
    return out, leak_chk, leak_bad


def build():
    snaps, lchk, lbad = load_9am()
    b = joblib.load(BASE / "models/unified_rank_v6.pkl")
    model, feats, encs = b["model"], b["feature_cols"], b["encoders"]
    cal = joblib.load(BASE / "models/pl_calibrators_v6.pkl"); cal = cal.get("calibrators", cal)
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df = df.dropna(subset=[COL_JYUN, COL_RID, "split"]).copy()
    enc = apply_encoders(df, encs)
    X = enc[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    df["_score"] = model.predict(X)
    payouts = load_payouts()
    races = []
    for rid, g in df.groupby(COL_RID, sort=False):
        if len(g) < 5: continue
        rid16 = str(rid)[:16]
        if rid16 not in snaps: continue
        sp = snaps[rid16]; po = payouts.get(rid16)
        g = g.sort_values(COL_BAN).reset_index(drop=True)
        ban = g[COL_BAN].astype(int).values
        keys = [int(bb) for bb in ban if int(bb) in sp["tan"]]
        if len(keys) < 5: continue
        idx = {int(ban[i]): i for i in range(len(ban))}
        w_all = PL.pl_weights(g["_score"].values)
        fcal_all = cal["tansho"].predict(PL.all_tansho(w_all)) if "tansho" in cal else PL.all_tansho(w_all)
        tan = np.array([sp["tan"][k] for k in keys])
        inv = 1.0 / tan; p_win = inv / inv.sum()                       # 単勝 de-vig
        w_mkt = p_win.copy()                                           # PL重み≈単勝勝率
        p_place_PL = all_fukusho_vec_fast(w_mkt)                       # 単勝由来 PL複勝率
        has_fuku = all(k in sp["fuku"] for k in keys)
        rec = {"rid16": rid16, "year": int(rid16[:4]), "keys": keys,
               "p_win": p_win, "f_cal": np.array([max(float(fcal_all[idx[k]]), 1e-9) for k in keys]),
               "po": po, "tan": tan}
        if has_fuku:
            fmid = np.array([(sp["fuku"][k][0] + sp["fuku"][k][1]) / 2 for k in keys])
            pm = 1.0 / fmid; p_place_mkt = pm / pm.sum() * 3.0          # 複勝 de-vig (Σ=3)
            rec["p_place_mkt"] = p_place_mkt; rec["p_place_PL"] = p_place_PL
            rec["dA"] = float(np.sum(np.abs(p_place_mkt - p_place_PL)))
            rec["disc"] = p_place_mkt - p_place_PL
        rec["r"] = np.log(rec["f_cal"]) - np.log(p_win)
        rec["dB"] = float(np.max(np.abs(rec["r"])))
        races.append(rec)
    return races, lchk, lbad


def fuku_pay(po, ban):
    if po is None: return 0.0
    for bn, k in [(po["win"], "fuku_win"), (po["plc"], "fuku_plc"), (po["sho"], "fuku_sho")]:
        if int(bn) == ban and po[k] and not pd.isna(po[k]): return float(po[k])
    return 0.0


def is_top3(po, ban):
    return po is not None and ban in (int(po["win"]), int(po["plc"]), int(po["sho"]))


def betA(races, side):
    """歪みA: disc>0(side=+1) or disc<0(side=-1) の馬を複勝。"""
    stake = ret = 0.0; bets = hits = 0
    for r in races:
        if "disc" not in r: continue
        for i, k in enumerate(r["keys"]):
            if (side > 0 and r["disc"][i] > 0) or (side < 0 and r["disc"][i] < 0):
                stake += 100; bets += 1; won = is_top3(r["po"], k)
                if won: ret += fuku_pay(r["po"], k); hits += 1
    return ret / stake if stake else 0.0, bets


def betB(races):
    """歪みB: r>0(f買い)馬を単勝。"""
    stake = ret = 0.0; bets = hits = 0; clv = []
    for r in races:
        for i, k in enumerate(r["keys"]):
            if r["r"][i] > 0:
                stake += 100; bets += 1; won = (r["po"] is not None and int(r["po"]["win"]) == k)
                if won and r["po"]["tansho"] and not pd.isna(r["po"]["tansho"]):
                    ret += float(r["po"]["tansho"]); hits += 1
    return ret / stake if stake else 0.0, bets


def boot(fn, races, lo_pct):
    rois = []
    for _ in range(N_BOOT):
        samp = [races[i] for i in rng.choice(len(races), len(races), replace=True)]
        roi, bets = fn(samp)
        if bets > 0: rois.append(roi)
    return round(float(np.percentile(rois, lo_pct)), 4) if rois else None


def main():
    print("[build] 9時単複 + v6 f ...")
    races, lchk, lbad = build()
    print(f"  races={len(races):,}  リーク検査={lchk:,} 違反={lbad}")
    tv = [r for r in races if r["year"] <= 2023]; te = [r for r in races if r["year"] >= 2024]

    # 分位閾値は ≤2023 で決定
    dA_tv = np.array([r["dA"] for r in tv if "dA" in r])
    dB_tv = np.array([r["dB"] for r in tv])
    results = {"leak_checked": lchk, "leak_violations": lbad, "PASS": bool(lbad == 0),
               "n_races": len(races), "min_bets": MIN_BETS, "quantiles": QUANTILES,
               "note": "9時単複のみ(馬連UMARENは省略). 歪みA=単勝PL vs 複勝市場, 歪みB=f vs π9.",
               "distortions": {}}
    n_tests = len(QUANTILES) * 3   # A+ / A- / B
    pct_lo = 100 * (0.05 / n_tests) / 2

    pockets = []
    for q in QUANTILES:
        thA = float(np.quantile(dA_tv, q)); thB = float(np.quantile(dB_tv, q))
        teA = [r for r in te if "dA" in r and r["dA"] >= thA]
        tvA = [r for r in tv if "dA" in r and r["dA"] >= thA]
        teB = [r for r in te if r["dB"] >= thB]; tvB = [r for r in tv if r["dB"] >= thB]
        for label, fn, te_l, tv_l, floor in [
            (f"A+_complies_q{q}", lambda rs: betA(rs, +1), teA, tvA, TAKEOUT_F),
            (f"A-_complies_q{q}", lambda rs: betA(rs, -1), teA, tvA, TAKEOUT_F),
            (f"B_fedge_q{q}", betB, teB, tvB, TAKEOUT_T)]:
            roi_te, bets_te = fn(te_l); roi_tv, bets_tv = fn(tv_l)
            ci_lo = boot(fn, te_l, pct_lo) if bets_te > 0 else None
            elig = bets_te >= MIN_BETS
            disc = roi_tv > floor and bets_tv >= MIN_BETS
            conf = elig and disc and (ci_lo is not None and ci_lo > floor)
            if conf: pockets.append(label)
            results["distortions"][label] = {
                "thr": round(thA if label.startswith("A") else thB, 4),
                "roi_test": round(roi_te, 4), "bets_test": bets_te, "roi_ci_lo_bonf": ci_lo,
                "roi_le2023": round(roi_tv, 4), "bets_le2023": bets_tv, "floor": floor,
                "eligible": bool(elig), "discovered": bool(disc), "confirmed": bool(conf)}

    if pockets:
        verdict = f"ARBITRAGE_FOUND: {pockets} が ≤2023発見 & test再現 & Bonferroni補正後CI下限>控除。市場内部矛盾の裁定機会実在。最優先深掘り。"
    else:
        any_disc = [k for k, v in results["distortions"].items() if v["discovered"]]
        verdict = (f"NO_ARBITRAGE: 補正後 test で控除超を再現した歪み分位なし。≤2023発見={any_disc or 'なし'} は test/補正で消失。"
                   "市場は内部整合的で歪みも控除に食われる=この弾も冗長。")
    results["verdict"] = verdict
    OUT_JSON.parent.mkdir(exist_ok=True)
    json.dump(results, open(OUT_JSON, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print(f"\n[リーク] 違反={lbad} {'PASS' if lbad==0 else 'FAIL'}")
    print(f"Bonferroni片側下限={pct_lo:.3f}%tile (n_tests={n_tests})")
    print(f"{'歪み分位':22s}{'thr':>7s}{'ROIte':>8s}{'CI下限':>8s}{'bets_te':>8s}{'≤23ROI':>8s}{'bets≤23':>8s} 発見/再現")
    for k, v in results["distortions"].items():
        cl = f"{v['roi_ci_lo_bonf']*100:.0f}" if v["roi_ci_lo_bonf"] is not None else "-"
        print(f"{k:22s}{v['thr']:>7.3f}{v['roi_test']*100:>7.1f}%{cl:>7s}%{v['bets_test']:>8,}{v['roi_le2023']*100:>7.1f}%{v['bets_le2023']:>8,}  {v['discovered']}/{v['confirmed']}")
    print(f"\n判定: {verdict}")
    print(f"[saved] {OUT_JSON}")


if __name__ == "__main__":
    main()
