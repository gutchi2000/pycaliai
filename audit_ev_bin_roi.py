"""
audit_ev_bin_roi.py — EV-bin 別 実現ROIカーブ (leak-safe, 診断のみ)
====================================================================
問い: EV シグナルは本物 (高EVほど高ROI) か、罠 (高EVほど低ROI = 低人気過大評価由来) か。
      EV の大きさ別に実現ROIがどう動くかを直接証拠として出す。

leak規約: EV判定=前売り9時(区分1, 09:00近傍)。決済=確定配当(kekka)。CLV=log(o9)-log(o確定)。
          test 2024-25 のみ。対象=単勝(TANPUK)/複勝(TANPUK)/馬連(UMAREN)。
          ワイド/馬単/三連複/三連単は9時時系列オッズ無しのため対象外。

出すもの (券種別):
  1. EV-bin ごとの 件数 / 平均EV / 実現ROI / 的中率 / 平均オッズ / 平均CLV
  2. EV-bin別ROIが単調増加(本物)か右肩下がり(罠)か
  3. オッズ帯(人気帯)層別: 同じ高EVでも人気馬由来か低人気由来か
  4. EV較正曲線: 予測EV(bin平均) vs 実現ROI (EV=1.3で実現1.3倍に近いか)
  5. 実現ROI が控除率(単複0.80/馬連0.775)を超える EV-bin の有無・件数・CLV

出力: reports/audit_ev_bin_roi.json
実行: PYTHONUTF8=1 python audit_ev_bin_roi.py
"""
from __future__ import annotations
import glob, json, re
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import pl_probs as PL
from backtest_pl_ev import all_umaren_mat, all_fukusho_vec_fast, load_payouts

BASE = Path(__file__).parent
MASTER_CSV = BASE / "data/master_v2_20130105-20251228.csv"
ODIR = BASE / "data/Time _series_odds"
OUT_JSON = BASE / "reports/audit_ev_bin_roi.json"

COL_RID = "レースID(新/馬番無)"
COL_JYUN = "着順"
COL_BAN = "馬番"

EV_BINS = [0.0, 0.8, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0, 1e18]
EV_LAB = ["<0.8", "0.8-0.9", "0.9-1.0", "1.0-1.1", "1.1-1.3", "1.3-1.5", "1.5-2.0", "2.0+"]
FAV_BINS = [0.0, 3.0, 7.0, 15.0, 50.0, 1e18]
FAV_LAB = ["<3", "3-7", "7-15", "15-50", "50+"]
TAKEOUT_FLOOR = {"tansho": 0.80, "fukusho": 0.80, "umaren": 0.775}


def _rid16(x):
    return re.sub(r"\D", "", str(x))[:16]


def ev_bin(ev):
    return EV_LAB[int(np.clip(np.digitize([ev], EV_BINS)[0] - 1, 0, len(EV_LAB) - 1))]


def fav_bin(o):
    return FAV_LAB[int(np.clip(np.digitize([o], FAV_BINS)[0] - 1, 0, len(FAV_LAB) - 1))]


def _pick(grp, kb_col, tm_col, race_mmdd, kb_val, mode):
    """kb_val=区分。mode='9am'→09:00近傍 / mode='final'→最大TM(確定)。"""
    g = grp[pd.to_numeric(grp[kb_col], errors="coerce") == kb_val]
    if len(g) == 0:
        return None
    tm = pd.to_numeric(g[tm_col], errors="coerce")
    if mode == "9am":
        hh = tm % 10000
        mmdd = tm // 10000
        same = g[mmdd == race_mmdd]
        base = same if len(same) > 0 else g
        return base.loc[(hh.loc[base.index] - 900).abs().idxmin()]
    return g.loc[tm.idxmax()]


def load_tanpuk():
    """{rid16: {"tan9":{ban:o}, "tanf":{ban:o}, "fuku9":{ban:(lo,hi)}}}."""
    files = sorted(glob.glob(str(ODIR / "TANPUK_*.csv")))[-1:]
    tp = pd.concat([pd.read_csv(f, encoding="cp932", low_memory=False) for f in files],
                   ignore_index=True)
    cols = list(tp.columns)
    RID, KB, TM = cols[0], cols[1], cols[2]
    tan_c, flo_c, fhi_c = {}, {}, {}
    for c in cols:
        cs = str(c)
        m = re.match(r"^\s*(\d+)\s*単\s*$", cs)
        if m: tan_c[int(m.group(1))] = c; continue
        m = re.match(r"^\s*(\d+)\s*複\s*Lo\s*$", cs)
        if m: flo_c[int(m.group(1))] = c; continue
        m = re.match(r"^\s*(\d+)\s*複\s*Hi\s*$", cs)
        if m: fhi_c[int(m.group(1))] = c
    tp["rid16"] = tp[RID].map(_rid16)
    tp = tp[(tp["rid16"].str.len() == 16) & (tp["rid16"].str[:4].astype(int) >= 2024)]
    out = {}
    for rid, g in tp.groupby("rid16", sort=False):
        mmdd = int(rid[4:8])
        r9 = _pick(g, KB, TM, mmdd, 1, "9am")
        rf = _pick(g, KB, TM, mmdd, 4, "final")
        if r9 is None:
            continue
        def vals(row, cmap):
            d = {}
            if row is None: return d
            for ban, c in cmap.items():
                v = pd.to_numeric(pd.Series([row[c]]), errors="coerce").iloc[0]
                if pd.notna(v) and v > 1.0:
                    d[ban] = float(v)
            return d
        tan9 = vals(r9, tan_c)
        tanf = vals(rf, tan_c)
        fuku9 = {}
        for ban in flo_c:
            lo = pd.to_numeric(pd.Series([r9[flo_c[ban]]]), errors="coerce").iloc[0]
            hi = pd.to_numeric(pd.Series([r9[fhi_c[ban]]]), errors="coerce").iloc[0] if ban in fhi_c else np.nan
            if pd.notna(lo) and lo > 1.0:
                fuku9[ban] = (float(lo), float(hi) if pd.notna(hi) else float(lo))
        if tan9:
            out[rid] = {"tan9": tan9, "tanf": tanf, "fuku9": fuku9}
    return out


def load_umaren():
    """{rid16: {"um9":{(i,j):o}, "umf":{(i,j):o}}}."""
    files = sorted(glob.glob(str(ODIR / "UMAREN_*.csv")))[-1:]
    um = pd.concat([pd.read_csv(f, encoding="cp932", low_memory=False) for f in files],
                   ignore_index=True)
    cols = list(um.columns)
    RID, KB, TM = cols[0], cols[1], cols[2]
    pair_c = {}
    for c in cols:
        m = re.search(r"(\d+)\s*-\s*(\d+)", str(c))
        if m:
            i, j = int(m.group(1)), int(m.group(2))
            if i != j:
                pair_c[(min(i, j), max(i, j))] = c
    um["rid16"] = um[RID].map(_rid16)
    um = um[(um["rid16"].str.len() == 16) & (um["rid16"].str[:4].astype(int) >= 2024)]
    out = {}
    for rid, g in um.groupby("rid16", sort=False):
        mmdd = int(rid[4:8])
        r9 = _pick(g, KB, TM, mmdd, 1, "9am")
        rf = _pick(g, KB, TM, mmdd, 4, "final")
        if r9 is None:
            continue
        def vals(row):
            d = {}
            if row is None: return d
            for (i, j), c in pair_c.items():
                v = pd.to_numeric(pd.Series([row[c]]), errors="coerce").iloc[0]
                if pd.notna(v) and v > 1.0:
                    d[(i, j)] = float(v)
            return d
        d9 = vals(r9)
        if d9:
            out[rid] = {"um9": d9, "umf": vals(rf)}
    return out


def main():
    print("[load] v6 model + calibrators")
    b = joblib.load(BASE / "models/unified_rank_v6.pkl")
    model, feats, encs = b["model"], b["feature_cols"], b["encoders"]
    cal = joblib.load(BASE / "models/pl_calibrators_v6.pkl"); cal = cal.get("calibrators", cal)

    print("[load] master (test 2024-25)")
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df = df.dropna(subset=[COL_JYUN, COL_RID, "split"])
    df = df[df["split"] == "test"].copy()
    for c, le in encs.items():
        if c not in df.columns: continue
        v = df[c].astype(str).fillna("__NaN__")
        df[c] = le.transform(v.where(v.isin(set(le.classes_)), "__NaN__"))
    X = df[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    df["_score"] = model.predict(X)

    print("[load] payouts + 9時/確定 odds")
    payouts = load_payouts()
    tanpuk = load_tanpuk()
    umaren = load_umaren()
    print(f"  単複={len(tanpuk):,}  馬連={len(umaren):,}")

    def newcell():
        return {"n": 0, "sev": 0.0, "stake": 0.0, "ret": 0.0, "hit": 0,
                "so9": 0.0, "sclv": 0.0, "nclv": 0}
    # agg[bet][evlab]   と  aggx[bet][(evlab,favlab)]
    agg = {b: defaultdict(newcell) for b in ["tansho", "fukusho", "umaren"]}
    aggx = {b: defaultdict(newcell) for b in ["tansho", "fukusho", "umaren"]}

    def add(bet, evlab, favlab, ev, o9, of, won, pay):
        for store, key in ((agg[bet], evlab), (aggx[bet], (evlab, favlab))):
            c = store[key]
            c["n"] += 1; c["sev"] += ev; c["stake"] += 100.0; c["ret"] += pay
            c["hit"] += int(won); c["so9"] += o9
            if of and of > 1.0 and o9 > 1.0:
                c["sclv"] += float(np.log(o9) - np.log(of)); c["nclv"] += 1

    n_races = 0
    for rid, g in df.groupby(COL_RID, sort=False):
        if len(g) < 5: continue
        rid16 = _rid16(rid)
        po = payouts.get(rid16)
        g = g.sort_values(COL_BAN).reset_index(drop=True)
        ban = g[COL_BAN].astype(int).values
        n = len(ban)
        w = PL.pl_weights(g["_score"].values)
        n_races += 1
        sp = tanpuk.get(rid16)
        um = umaren.get(rid16)

        # 単勝
        if sp:
            p_cal = cal["tansho"].predict(PL.all_tansho(w))
            for i in range(n):
                o9 = sp["tan9"].get(int(ban[i]))
                if o9 is None: continue
                ev = p_cal[i] * o9
                won = (po is not None and int(ban[i]) == po["win"])
                pay = float(po["tansho"]) if (won and po and po["tansho"] and not pd.isna(po["tansho"])) else 0.0
                of = sp["tanf"].get(int(ban[i]))
                add("tansho", ev_bin(ev), fav_bin(o9), ev, o9, of, won, pay)

        # 複勝 (人気帯は単勝9時オッズで定義)
        if sp and sp["fuku9"]:
            p_cal = cal["fukusho"].predict(all_fukusho_vec_fast(w))
            top3 = {po["win"], po["plc"], po["sho"]} if po else set()
            fpay = {}
            if po:
                for bn, k in [(po["win"], "fuku_win"), (po["plc"], "fuku_plc"), (po["sho"], "fuku_sho")]:
                    if po[k] and not pd.isna(po[k]): fpay[int(bn)] = float(po[k])
            for i in range(n):
                fo = sp["fuku9"].get(int(ban[i]))
                if fo is None: continue
                odds = (fo[0] + fo[1]) / 2.0
                ev = p_cal[i] * odds
                won = (po is not None and int(ban[i]) in top3)
                pay = fpay.get(int(ban[i]), 0.0) if won else 0.0
                tan_o9 = sp["tan9"].get(int(ban[i]), np.nan)
                fb = fav_bin(tan_o9) if not np.isnan(tan_o9) else "?"
                add("fukusho", ev_bin(ev), fb, ev, odds, None, won, pay)

        # 馬連 (人気帯はペア2頭の単勝9時オッズの大きい方=人気薄い側)
        if um:
            Pmat = all_umaren_mat(w)
            iu, ju = np.triu_indices(n, k=1)
            p_cal = cal["umaren"].predict(Pmat[iu, ju])
            win_pair = {po["win"], po["plc"]} if po else set()
            um_pay = float(po["umaren"]) if (po and po["umaren"] and not pd.isna(po["umaren"])) else 0.0
            tan9 = sp["tan9"] if sp else {}
            for k in range(len(iu)):
                i, j = int(iu[k]), int(ju[k])
                bi, bj = int(ban[i]), int(ban[j])
                kp = (min(bi, bj), max(bi, bj))
                o9 = um["um9"].get(kp)
                if o9 is None: continue
                ev = p_cal[k] * o9
                won = (po is not None and {bi, bj} == win_pair)
                pay = um_pay if won else 0.0
                of = um["umf"].get(kp)
                oi, oj = tan9.get(bi, np.nan), tan9.get(bj, np.nan)
                fb = fav_bin(np.nanmax([oi, oj])) if not (np.isnan(oi) and np.isnan(oj)) else "?"
                add("umaren", ev_bin(ev), fb, ev, o9, of, won, pay)

    # ---------- finalize ----------
    def fin(cell):
        n = cell["n"]
        return {"n": n, "mean_ev": round(cell["sev"] / n, 3) if n else 0.0,
                "roi": round(cell["ret"] / cell["stake"], 4) if cell["stake"] else 0.0,
                "hit_rate": round(cell["hit"] / n, 4) if n else 0.0,
                "mean_odds9": round(cell["so9"] / n, 2) if n else 0.0,
                "mean_clv": round(cell["sclv"] / cell["nclv"], 4) if cell["nclv"] else None}

    result = {"leak_policy": "EV=前売り9時 / 決済=確定配当 / CLV=log(o9)-log(o確定) / test2024-25",
              "ev_bins": EV_LAB, "fav_bins": FAV_LAB, "takeout_floor": TAKEOUT_FLOOR,
              "n_test_races": n_races, "by_ev_bin": {}, "by_ev_x_fav": {}, "calibration_curve": {},
              "above_takeout_bins": {}}

    for bet in ["tansho", "fukusho", "umaren"]:
        result["by_ev_bin"][bet] = {lab: fin(agg[bet][lab]) for lab in EV_LAB if agg[bet][lab]["n"] > 0}
        # EV較正曲線 (mean_ev, roi)
        result["calibration_curve"][bet] = [
            {"ev_bin": lab, "mean_ev": fin(agg[bet][lab])["mean_ev"], "roi": fin(agg[bet][lab])["roi"],
             "n": agg[bet][lab]["n"]} for lab in EV_LAB if agg[bet][lab]["n"] > 0]
        # 控除超 bin
        floor = TAKEOUT_FLOOR[bet]
        above = []
        for lab in EV_LAB:
            f = fin(agg[bet][lab])
            if agg[bet][lab]["n"] > 0 and f["roi"] >= floor:
                above.append({"ev_bin": lab, "n": f["n"], "roi": f["roi"], "mean_clv": f["mean_clv"]})
        result["above_takeout_bins"][bet] = above
        # EV×人気帯 (EV>=1.3 帯のみ抽出して保存、全クロスも)
        result["by_ev_x_fav"][bet] = {}
        for (evlab, favlab), c in aggx[bet].items():
            if c["n"] >= 20:
                result["by_ev_x_fav"][bet][f"{evlab}|{favlab}"] = fin(c)

    # 単調性判定 (EV>=1.0 の bin 列で roi が増加か)
    verdict = {}
    for bet in ["tansho", "fukusho", "umaren"]:
        seq = [(lab, fin(agg[bet][lab])["roi"], agg[bet][lab]["n"])
               for lab in ["1.0-1.1", "1.1-1.3", "1.3-1.5", "1.5-2.0", "2.0+"]
               if agg[bet][lab]["n"] >= 20]
        if len(seq) >= 2:
            rois = [r for _, r, _ in seq]
            trend = rois[-1] - rois[0]
            mono_up = all(rois[i] <= rois[i + 1] + 0.02 for i in range(len(rois) - 1))
            verdict[bet] = {"ev_ge1_roi_seq": [(l, r) for l, r, _ in seq],
                            "roi_top_minus_bottom": round(trend, 4),
                            "monotone_up": mono_up}
    result["verdict_monotonicity"] = verdict

    OUT_JSON.parent.mkdir(exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # ---------- console ----------
    print("\n" + "=" * 92)
    print(f"EV-bin 別 実現ROIカーブ (leak-safe 9時)  test races={n_races:,}")
    print("=" * 92)
    for bet in ["tansho", "fukusho", "umaren"]:
        floor = TAKEOUT_FLOOR[bet]
        print(f"\n【{bet}】控除率床={floor*100:.1f}%")
        print(f"  {'EV-bin':9s} {'n':>8s} {'meanEV':>7s} {'ROI':>8s} {'hit':>7s} {'平均odds9':>9s} {'CLV':>8s}")
        for lab in EV_LAB:
            if agg[bet][lab]["n"] == 0: continue
            f = fin(agg[bet][lab])
            flag = " ★控除超" if f["roi"] >= floor else ""
            clv = f"{f['mean_clv']*100:+.1f}%" if f["mean_clv"] is not None else "  -  "
            print(f"  {lab:9s} {f['n']:>8,} {f['mean_ev']:>7.2f} {f['roi']*100:>7.1f}% "
                  f"{f['hit_rate']*100:>6.1f}% {f['mean_odds9']:>9.1f} {clv:>8s}{flag}")
        v = verdict.get(bet, {})
        if v:
            print(f"  → EV>=1.0 帯 ROI: {[(l, f'{r*100:.0f}%') for l, r in v['ev_ge1_roi_seq']]}  "
                  f"top-bottom={v['roi_top_minus_bottom']*100:+.1f}pt  単調増={v['monotone_up']}")

    # 人気帯クロス (高EV帯で人気/低人気の差)
    print("\n--- EV>=1.3 帯 × 人気帯(単勝9時オッズ) の実現ROI ---")
    for bet in ["tansho", "fukusho", "umaren"]:
        print(f"  [{bet}]")
        for evlab in ["1.3-1.5", "1.5-2.0", "2.0+"]:
            for favlab in FAV_LAB:
                c = aggx[bet].get((evlab, favlab))
                if c and c["n"] >= 20:
                    f = fin(c)
                    print(f"    EV {evlab:8s} 人気{favlab:6s}: n={f['n']:>6,} ROI={f['roi']*100:6.1f}% "
                          f"hit={f['hit_rate']*100:5.1f}% meanEV={f['mean_ev']:.2f}")
    print(f"\n[saved] {OUT_JSON}")


if __name__ == "__main__":
    main()
