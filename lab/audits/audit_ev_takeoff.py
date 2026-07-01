"""
audit_ev_takeoff.py — EV追い「取りこぼし」診断 (leak-safe, 監査のみ)
====================================================================
問い: 現状の買い目生成 (印先行 = ◎〇▲△ で絞ってから組む) は、
      「市場が付けた前売り9時オッズの中で、v6 較正済み確率が EV>1 と判定する目」を
      取りこぼしていないか。取りこぼしの件数・分布・実現ROI を事実で出す。

leak規約: EV 判定に使うオッズは **前売り当日9時** (TANPUK/UMAREN の 区分=1, 当日09:00 近傍)。
          決済のみ確定配当 (kekka)。test 2024-2025 のみ。

対象券種 (9時時系列オッズが手元にある 3 種):
  - 単勝   : TANPUK "N単"      × cal[tansho]
  - 複勝   : TANPUK "N複Lo/Hi" × cal[fukusho]   (EV = p_sho × (Lo+Hi)/2)
  - 馬連   : UMAREN "iJ-jJ"    × cal[umaren]
  ワイド/馬単/三連複/三連単は 9時時系列オッズが無いため対象外 (報告に明記)。

印の定義: モデル score 降順 rank。◎=1 〇=2 ▲=3 △=4,5 (= 印 rank<=5)、印外 = rank>=6。
取りこぼし = 「EV>1 かつ 印外(rank>=6) が絡む目」(現状 Cowork は印中心で組むため構造的に拾えない)。

出力: reports/audit_ev_takeoff.json
実行: PYTHONUTF8=1 python audit_ev_takeoff.py
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
OUT_JSON = BASE / "reports/audit_ev_takeoff.json"

COL_RID = "レースID(新/馬番無)"
COL_JYUN = "着順"
COL_BAN = "馬番"
EV_MIN = 1.0


def _rid16(x):
    return re.sub(r"\D", "", str(x))[:16]


def _pick_9am(grp, kb_col, tm_col, race_mmdd):
    """区分=1 (前売り) の中から当日 09:00 近傍の行を返す。無ければ None。"""
    g1 = grp[pd.to_numeric(grp[kb_col], errors="coerce") == 1]
    if len(g1) == 0:
        return None
    tm = pd.to_numeric(g1[tm_col], errors="coerce")
    hh = tm % 10000
    mmdd = tm // 10000
    same = g1[mmdd == race_mmdd]
    base = same if len(same) > 0 else g1
    hh_base = hh.loc[base.index]
    return base.loc[(hh_base - 900).abs().idxmin()]


def load_tanpuk_9am():
    """{rid16: {"tan": {ban: odds}, "fuku": {ban: (lo,hi)}}} (前売り9時)。"""
    files = sorted(glob.glob(str(ODIR / "TANPUK_*.csv")))[-1:]  # 最新=2021-2025 (test 2024-25 を含む)
    tp = pd.concat([pd.read_csv(f, encoding="cp932", low_memory=False) for f in files],
                   ignore_index=True)
    cols = list(tp.columns)
    RID, KB, TM, TOU = cols[0], cols[1], cols[2], cols[3]
    tan_cols, fuku_lo, fuku_hi = {}, {}, {}
    for c in cols:
        cs = str(c)
        m = re.match(r"^\s*(\d+)\s*単\s*$", cs)
        if m:
            tan_cols[int(m.group(1))] = c; continue
        m = re.match(r"^\s*(\d+)\s*複\s*Lo\s*$", cs)
        if m:
            fuku_lo[int(m.group(1))] = c; continue
        m = re.match(r"^\s*(\d+)\s*複\s*Hi\s*$", cs)
        if m:
            fuku_hi[int(m.group(1))] = c
    tp["rid16"] = tp[RID].map(_rid16)
    tp = tp[tp["rid16"].str.len() == 16]
    tp = tp[tp["rid16"].str[:4].astype(int) >= 2024]   # test 期間に限定 (高速化)
    out = {}
    for rid, g in tp.groupby("rid16", sort=False):
        race_mmdd = int(rid[4:8])
        row = _pick_9am(g, KB, TM, race_mmdd)
        if row is None:
            continue
        tan, fuku = {}, {}
        for ban, c in tan_cols.items():
            v = pd.to_numeric(pd.Series([row[c]]), errors="coerce").iloc[0]
            if pd.notna(v) and v > 1.0:
                tan[ban] = float(v)
        for ban in fuku_lo:
            lo = pd.to_numeric(pd.Series([row[fuku_lo[ban]]]), errors="coerce").iloc[0]
            hi = pd.to_numeric(pd.Series([row[fuku_hi.get(ban)]]), errors="coerce").iloc[0] \
                if ban in fuku_hi else np.nan
            if pd.notna(lo) and lo > 1.0:
                fuku[ban] = (float(lo), float(hi) if pd.notna(hi) else float(lo))
        if tan:
            out[rid] = {"tan": tan, "fuku": fuku}
    return out


def load_umaren_9am():
    """{rid16: {(i,j): odds}} (i<j, 前売り9時)。"""
    files = sorted(glob.glob(str(ODIR / "UMAREN_*.csv")))[-1:]  # 最新=2021-2025 (test 2024-25 を含む)
    um = pd.concat([pd.read_csv(f, encoding="cp932", low_memory=False) for f in files],
                   ignore_index=True)
    cols = list(um.columns)
    RID, KB, TM = cols[0], cols[1], cols[2]
    pair_cols = {}
    for c in cols:
        m = re.search(r"(\d+)\s*-\s*(\d+)", str(c))
        if m:
            i, j = int(m.group(1)), int(m.group(2))
            if i != j:
                pair_cols[(min(i, j), max(i, j))] = c
    um["rid16"] = um[RID].map(_rid16)
    um = um[um["rid16"].str.len() == 16]
    um = um[um["rid16"].str[:4].astype(int) >= 2024]
    out = {}
    for rid, g in um.groupby("rid16", sort=False):
        race_mmdd = int(rid[4:8])
        row = _pick_9am(g, KB, TM, race_mmdd)
        if row is None:
            continue
        d = {}
        for (i, j), c in pair_cols.items():
            v = pd.to_numeric(pd.Series([row[c]]), errors="coerce").iloc[0]
            if pd.notna(v) and v > 1.0:
                d[(i, j)] = float(v)
        if d:
            out[rid] = d
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
        if c not in df.columns:
            continue
        v = df[c].astype(str).fillna("__NaN__")
        df[c] = le.transform(v.where(v.isin(set(le.classes_)), "__NaN__"))
    X = df[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    df["_score"] = model.predict(X)

    print("[load] payouts (確定, 決済用)")
    payouts = load_payouts()
    print("[load] 前売り9時 odds (TANPUK 単複 / UMAREN 馬連)")
    tanpuk = load_tanpuk_9am()
    umaren = load_umaren_9am()
    print(f"  9時 races: 単複={len(tanpuk):,}  馬連={len(umaren):,}")

    # 集計器: 券種 → {rankclass: {"n_evgt1":, "stake":, "ret":, "hit":}}
    def newcell():
        return defaultdict(lambda: {"n": 0, "stake": 0.0, "ret": 0.0, "hit": 0})
    agg = {"tansho": newcell(), "fukusho": newcell(), "umaren": newcell()}
    n_races = 0

    for rid, g in df.groupby(COL_RID, sort=False):
        if len(g) < 5:
            continue
        rid16 = _rid16(rid)
        po = payouts.get(rid16)
        g = g.sort_values(COL_BAN).reset_index(drop=True)
        ban = g[COL_BAN].astype(int).values
        n = len(ban)
        scores = g["_score"].values
        # rank (1=◎): score 降順
        order = np.argsort(-scores)
        rank_by_idx = np.empty(n, dtype=int)
        for r, idx in enumerate(order):
            rank_by_idx[idx] = r + 1
        b2i = {int(ban[i]): i for i in range(n)}

        w = PL.pl_weights(scores)
        n_races += 1

        # ---------- 単勝 ----------
        sp = tanpuk.get(rid16)
        if sp:
            p_cal = cal["tansho"].predict(PL.all_tansho(w))
            win_ban = po["win"] if po else None
            for i in range(n):
                o = sp["tan"].get(int(ban[i]))
                if o is None:
                    continue
                ev = p_cal[i] * o
                if ev >= EV_MIN:
                    rc = "◎" if rank_by_idx[i] == 1 else ("印" if rank_by_idx[i] <= 5 else "印外")
                    won = (po is not None and int(ban[i]) == win_ban)
                    pay = float(po["tansho"]) if (won and po["tansho"] and not pd.isna(po["tansho"])) else 0.0
                    for key in (rc, "ALL"):
                        c = agg["tansho"][key]
                        c["n"] += 1; c["stake"] += 100.0; c["ret"] += pay; c["hit"] += int(won)

        # ---------- 複勝 ----------
        if sp and sp["fuku"]:
            p_cal = cal["fukusho"].predict(all_fukusho_vec_fast(w))
            top3 = {po["win"], po["plc"], po["sho"]} if po else set()
            fuku_pay = {}
            if po:
                for bn, k in [(po["win"], "fuku_win"), (po["plc"], "fuku_plc"), (po["sho"], "fuku_sho")]:
                    if po[k] and not pd.isna(po[k]):
                        fuku_pay[int(bn)] = float(po[k])
            for i in range(n):
                fo = sp["fuku"].get(int(ban[i]))
                if fo is None:
                    continue
                odds = (fo[0] + fo[1]) / 2.0
                ev = p_cal[i] * odds
                if ev >= EV_MIN:
                    rc = "◎" if rank_by_idx[i] == 1 else ("印" if rank_by_idx[i] <= 5 else "印外")
                    won = (po is not None and int(ban[i]) in top3)
                    pay = fuku_pay.get(int(ban[i]), 0.0) if won else 0.0
                    for key in (rc, "ALL"):
                        c = agg["fukusho"][key]
                        c["n"] += 1; c["stake"] += 100.0; c["ret"] += pay; c["hit"] += int(won)

        # ---------- 馬連 ----------
        um = umaren.get(rid16)
        if um:
            Pmat = all_umaren_mat(w)
            iu, ju = np.triu_indices(n, k=1)
            p_raw = Pmat[iu, ju]
            p_cal = cal["umaren"].predict(p_raw)
            win_pair = {po["win"], po["plc"]} if po else set()
            um_pay = float(po["umaren"]) if (po and po["umaren"] and not pd.isna(po["umaren"])) else 0.0
            for k in range(len(iu)):
                i, j = int(iu[k]), int(ju[k])
                bi, bj = int(ban[i]), int(ban[j])
                key_pair = (min(bi, bj), max(bi, bj))
                o = um.get(key_pair)
                if o is None:
                    continue
                ev = p_cal[k] * o
                if ev >= EV_MIN:
                    ri, rj = rank_by_idx[i], rank_by_idx[j]
                    if ri <= 5 and rj <= 5:
                        rc = "両印"
                    elif ri >= 6 and rj >= 6:
                        rc = "両印外"
                    else:
                        rc = "片印外"
                    won = (po is not None and {bi, bj} == win_pair)
                    pay = um_pay if won else 0.0
                    for key in (rc, "ALL"):
                        c = agg["umaren"][key]
                        c["n"] += 1; c["stake"] += 100.0; c["ret"] += pay; c["hit"] += int(won)

    # ---------- 集計出力 ----------
    def finalize(cell):
        out = {}
        for k, d in cell.items():
            out[k] = {"n_evgt1": d["n"], "hit": d["hit"],
                      "hit_rate": round(d["hit"] / d["n"], 4) if d["n"] else 0.0,
                      "roi": round(d["ret"] / d["stake"], 4) if d["stake"] else 0.0}
        return out

    result = {
        "leak_policy": "EV判定=前売り9時(区分1,09:00近傍) / 決済=確定配当 / test 2024-25",
        "ev_min": EV_MIN,
        "mark_def": "rank<=1:◎  rank<=5:印(◎〇▲△△)  rank>=6:印外",
        "n_test_races_scored": n_races,
        "covered_bet_types": ["tansho", "fukusho", "umaren"],
        "uncovered_no_9am_odds": ["wide", "umatan", "sanrenpuku", "sanrentan", "wakuren"],
        "results": {bt: finalize(agg[bt]) for bt in agg},
    }
    OUT_JSON.parent.mkdir(exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # ---------- コンソール ----------
    print("\n" + "=" * 72)
    print(f"EV>1 取りこぼし診断 (leak-safe 9時)  test races={n_races:,}")
    print("=" * 72)
    for bt in ["tansho", "fukusho", "umaren"]:
        r = result["results"][bt]
        allc = r.get("ALL", {})
        print(f"\n【{bt}】 EV>1 総数={allc.get('n_evgt1',0):,}  "
              f"実現ROI(9時EV>1を確定決済)={allc.get('roi',0)*100:.1f}%  hit={allc.get('hit_rate',0)*100:.1f}%")
        for key in (["◎", "印", "印外"] if bt != "umaren" else ["両印", "片印外", "両印外"]):
            d = r.get(key)
            if d:
                print(f"    {key:6s}: EV>1 {d['n_evgt1']:>6,} 件  実現ROI {d['roi']*100:6.1f}%  hit {d['hit_rate']*100:5.1f}%")
    print(f"\n[saved] {OUT_JSON}")


if __name__ == "__main__":
    main()
