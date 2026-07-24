# -*- coding: utf-8 -*-
"""
sim_gakusei.py — 学生大会(夏競馬)トーナメント・シミュレータ
============================================================
ルール再現: 初期100万pt / 確定オッズ決済 / 96R以上+50万pt以上 / 複利(次レースから可)。
窓: 2024-08-25〜09-22 (275R) / 2025-08-30〜09-21 (264R) の同季節ホールドアウト。
モデル: v6 (≤2022学習) = 両窓とも完全OOS。

賞3軸: 最終pt / 的中率(プラス収支R÷投票R) / 最高オッズ的中。

build: 基質生成 → data/gakusei_sim_races.parquet
sim  : 戦略総当たり + 日次ブートストラップで最終pt分布
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).parent
SUB = BASE / "data" / "gakusei_sim_races.parquet"
WINDOWS = {2024: ("20240825", "20240922"), 2025: ("20250830", "20250921")}


def cmd_build():
    import joblib
    import pl_probs as PL
    from joint_calibration_v6 import apply_encoders, COL_RID, COL_JYUN, COL_BAN, MASTER_CSV
    b = joblib.load(BASE / "models/unified_rank_v6.pkl")
    model, feats, encs = b["model"], b["feature_cols"], b["encoders"]
    cal = joblib.load(BASE / "models/pl_calibrators_v6.pkl")
    cal = cal.get("calibrators", cal)

    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df = df.dropna(subset=[COL_JYUN, COL_RID]).copy()
    df["_rid"] = df[COL_RID].astype(str).str.strip()
    df["_date"] = df["_rid"].str[:8]
    m = pd.Series(False, index=df.index)
    for y, (lo, hi) in WINDOWS.items():
        m |= (df["_date"] >= lo) & (df["_date"] <= hi)
    df = df[m].copy()
    print(f"window rows={len(df):,} races={df['_rid'].nunique()}")
    enc = apply_encoders(df, encs)
    X = enc[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    df["_s"] = model.predict(X)

    k = pd.read_csv(BASE / "data/kekka_20130105-20251228.csv", encoding="cp932",
                    usecols=["レースID(新)", "馬番", "確定着順", "単勝配当", "複勝配当", "馬連", "馬単", "３連複"],
                    low_memory=False)
    k["rid"] = k["レースID(新)"].astype(str).str[:16]
    k["ban"] = pd.to_numeric(k["馬番"], errors="coerce")
    k["f"] = pd.to_numeric(k["確定着順"], errors="coerce")
    tanmap = {(r.rid, int(r.ban)): pd.to_numeric(r.単勝配当, errors="coerce") / 100
              for r in k.itertuples() if r.f == 1 and pd.notna(r.ban)}
    fukmap = {(r.rid, int(r.ban)): pd.to_numeric(r.複勝配当, errors="coerce") / 100
              for r in k.itertuples() if pd.notna(r.ban) and r.f <= 3}
    kw = k[k["f"] == 1]
    ummap = dict(zip(kw["rid"], pd.to_numeric(kw["馬連"], errors="coerce") / 100))
    utmap = dict(zip(kw["rid"], pd.to_numeric(kw["馬単"], errors="coerce") / 100))
    trimap = dict(zip(kw["rid"], pd.to_numeric(kw["３連複"], errors="coerce") / 100))
    secmap = {r.rid: int(r.ban) for r in k.itertuples() if r.f == 2 and pd.notna(r.ban)}
    w = pd.read_parquet(BASE / "data/wide_payouts_2016-2025.parquet")
    w["rid"] = w["race_id"].astype(str).str[:16]
    wmap = {}
    for r in w.itertuples():
        dd = {}
        for a in ["w1", "w2", "w3"]:
            i, j, p = getattr(r, a + "_i"), getattr(r, a + "_j"), getattr(r, a + "_pay")
            if pd.notna(p):
                dd[(min(int(i), int(j)), max(int(i), int(j)))] = float(p) / 100
        wmap[r.rid] = dd

    rows = []
    for rid, g in df.groupby("_rid", sort=False):
        if len(g) < 6:
            continue
        g = g.sort_values("_s", ascending=False)
        sc = g["_s"].values
        praw = PL.all_tansho(PL.pl_weights(sc))
        p = cal["tansho"].predict(np.clip(praw, 1e-9, 1 - 1e-9)) if "tansho" in cal else praw
        p = np.clip(p, 1e-9, None)
        ban = g[COL_BAN].astype(int).values
        fin = g[COL_JYUN].values
        rid16 = rid[:16]
        ent = -(praw * np.log(np.clip(praw, 1e-12, 1))).sum() / np.log(len(praw))
        top4 = ban[:4]
        f4 = fin[:4]
        # ワイド/馬連の r1-r2 と、三連複 box4 (C(4,3)=4点) の的中/払戻
        pr12 = (min(top4[0], top4[1]), max(top4[0], top4[1]))
        wd12 = wmap.get(rid16, {}).get(pr12, 0.0) if (f4[0] <= 3 and f4[1] <= 3) else 0.0
        um12 = (ummap.get(rid16, 0.0) or 0.0) if (f4[0] <= 2 and f4[1] <= 2) else 0.0
        top3fin = set(ban[i] for i in range(len(ban)) if fin[i] <= 3)
        tri_hit = len(top3fin & set(top4)) >= 3 and len(top3fin) == 3
        tri_pay = (trimap.get(rid16, 0.0) or 0.0) if tri_hit else 0.0
        # 馬単3点 formation: r1→{r2,r3,r4}
        ut_hit = (int(fin[0]) == 1) and (secmap.get(rid16) in set(int(b) for b in top4[1:4]))
        ut_pay = (utmap.get(rid16, 0.0) or 0.0) if ut_hit else 0.0
        rows.append({
            "rid": rid16, "date": rid[:8], "year": int(rid[:4]),
            "n": len(g), "p1": p[0] / p.sum(), "p23": (praw[1] + praw[2]) / (1 - praw[0]),
            "chaos": ent,
            "hon_fin": int(fin[0]),
            "tan1": tanmap.get((rid16, int(ban[0])), 0.0) or 0.0,
            "fuku1": fukmap.get((rid16, int(ban[0])), 0.0) or 0.0,
            "fuku2": fukmap.get((rid16, int(ban[1])), 0.0) or 0.0,
            "wd12": wd12, "um12": um12,
            "tri_hit": tri_hit, "tri_pay": tri_pay,
            "ut_hit": ut_hit, "ut_pay": ut_pay,
            "gap": float(praw[0] - praw[1]),
        })
    out = pd.DataFrame(rows)
    out.to_parquet(SUB, index=False)
    print(f"[build] {len(out)} races -> {SUB}")
    for y, g in out.groupby("year"):
        print(f"  {y}: {len(g)}R {g['date'].nunique()}日  ◎勝率{(g.hon_fin==1).mean()*100:.1f}% "
              f"三連複box4的中{g.tri_hit.mean()*100:.1f}% 平均三連複配当(的中時){g[g.tri_hit].tri_pay.mean():.1f}倍")


def race_bet(row, stake, menu, tail_frac, p23_med):
    """1レースの (投票額, 払戻, hit最高オッズ) を返す。"""
    tail = int(stake * tail_frac / 100) * 100
    base = stake - tail
    ret = 0.0
    mx = 0.0
    if menu == "fuku":
        ret += base * row.fuku1
        if row.fuku1 > 0: mx = max(mx, row.fuku1)
    elif menu == "wide":
        ret += base * row.wd12
        if row.wd12 > 0: mx = max(mx, row.wd12)
    else:  # fuku_wide_gate: 相手強ならワイド半分、相手弱なら複勝のみ
        if row.p23 >= p23_med:
            h = int(base / 200) * 100
            ret += h * row.fuku1 + (base - h) * row.wd12
            if row.fuku1 > 0: mx = max(mx, row.fuku1)
            if row.wd12 > 0: mx = max(mx, row.wd12)
        else:
            ret += base * row.fuku1
            if row.fuku1 > 0: mx = max(mx, row.fuku1)
    if tail > 0:
        per = int(tail / 4 / 100) * 100  # box4 = 4点 (的中は高々1点)
        ret += per * row.tri_pay
        if row.tri_pay > 0: mx = max(mx, row.tri_pay)
        stake_eff = base + per * 4
    else:
        stake_eff = base
    return stake_eff, ret, mx


def run_tournament(days, cfg, p23_med):
    """days: [ (date, DataFrame) ] 日次リスト。cfg: dict。→ 結果メトリクス。"""
    bank = 1_000_000
    total_stake = 0
    n_voted = 0
    n_plus = 0
    mx_all = 0.0
    for date, g in days:
        g = g.sort_values("chaos").head(cfg["nsel"])
        for row in g.itertuples():
            stake = max(1000, int(bank * cfg["f"] / 100) * 100)
            stake = min(stake, bank)
            if stake < 500 or bank <= 0:
                continue
            se, ret, mx = race_bet(row, stake, cfg["menu"], cfg["tail"], p23_med)
            bank = bank - se + int(ret)
            total_stake += se
            n_voted += 1
            if ret > se: n_plus += 1
            mx_all = max(mx_all, mx)
    ok = (n_voted >= 96) and (total_stake >= 500_000)
    return {"final": bank, "voted": n_voted, "staked": total_stake,
            "hitrate": n_plus / max(n_voted, 1), "maxodds": mx_all, "eligible": ok}


def cmd_sim():
    d = pd.read_parquet(SUB)
    for c in ["tan1", "fuku1", "fuku2", "wd12", "um12", "tri_pay"]:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
    p23_med = float(d["p23"].median())
    grid = []
    for nsel in [11, 13, 16]:
        for menu in ["fuku", "wide", "fuku_wide_gate"]:
            for tail in [0.0, 0.2, 0.4]:
                for f in [0.7, 1.2, 2.0]:
                    grid.append({"nsel": nsel, "menu": menu, "tail": tail, "f": f / 100})
    rng = np.random.default_rng(42)
    out = []
    for cfg in grid:
        res = {}
        for y in [2024, 2025]:
            g = d[d["year"] == y]
            days = [(dt, gg) for dt, gg in g.groupby("date")]
            base = run_tournament(days, cfg, p23_med)
            # 日次ブートストラップで分布
            finals = []
            for _ in range(300):
                idx = rng.integers(0, len(days), len(days))
                finals.append(run_tournament([days[i] for i in idx], cfg, p23_med)["final"])
            finals = np.array(finals)
            base["p_gt_1m"] = float((finals > 1_000_000).mean())
            base["p_gt_1m2"] = float((finals > 1_200_000).mean())
            base["fin_p90"] = float(np.percentile(finals, 90))
            res[y] = base
        out.append({"cfg": cfg, **{f"{k}_{y}": v for y in [2024, 2025] for k, v in res[y].items()}})
    o = pd.DataFrame([{**r["cfg"],
                       "fin24": r["final_2024"], "fin25": r["final_2025"],
                       "hit24": round(r["hitrate_2024"], 3), "hit25": round(r["hitrate_2025"], 3),
                       "P1M_24": r["p_gt_1m_2024"], "P1M_25": r["p_gt_1m_2025"],
                       "p90_24": int(r["fin_p90_2024"]), "p90_25": int(r["fin_p90_2025"]),
                       "vote24": r["voted_2024"], "stak24": r["staked_2024"],
                       "elig": r["eligible_2024"] and r["eligible_2025"],
                       "mx24": round(r["maxodds_2024"], 1), "mx25": round(r["maxodds_2025"], 1)}
                      for r in out])
    o = o[o["elig"]]
    o["fin_min"] = o[["fin24", "fin25"]].min(axis=1)
    o["P1M_min"] = o[["P1M_24", "P1M_25"]].min(axis=1)
    print("=== 最終pt 両年min 上位10 ===")
    print(o.sort_values("fin_min", ascending=False).head(10).to_string(index=False))
    print("\n=== P(>100万) 両年min 上位6 ===")
    print(o.sort_values("P1M_min", ascending=False).head(6).to_string(index=False))
    print("\n=== 的中率 両年min 上位6 (スポンサー賞軸) ===")
    o["hit_min"] = o[["hit24", "hit25"]].min(axis=1)
    print(o.sort_values("hit_min", ascending=False).head(6)[
        ["nsel", "menu", "tail", "f", "hit24", "hit25", "fin24", "fin25"]].to_string(index=False))
    o.to_csv(BASE / "reports" / "gakusei_sim_grid.csv", index=False)
    print("[saved] reports/gakusei_sim_grid.csv")


def run_dynamic(days, p23_med, base_f=0.007, nsel=11, end_days=2, target=1_100_000,
                end_inst="wide", end_f=0.15):
    """二段階: 序盤=◎複勝flat温存 / 終盤=目標未達なら end_inst に end_f×bank を張る。"""
    bank = 1_000_000
    total_stake = 0; n_voted = 0; n_plus = 0; mx_all = 0.0
    nd = len(days)
    for di, (date, g) in enumerate(days):
        endgame = (di >= nd - end_days) and (bank < target)
        g = g.sort_values("chaos").head(nsel)
        for row in g.itertuples():
            if endgame:
                stake = max(1000, int(bank * end_f / 100) * 100)
            else:
                stake = max(1000, int(bank * base_f / 100) * 100)
            stake = min(stake, bank)
            if stake < 500 or bank <= 0:
                continue
            menu = end_inst if endgame else "fuku"
            tail = 0.5 if (endgame and end_inst == "trio") else 0.0
            if endgame and end_inst == "trio":
                se, ret, mx = race_bet(row, stake, "fuku", 0.5, p23_med)
            else:
                se, ret, mx = race_bet(row, stake, menu, 0.0, p23_med)
            bank = bank - se + int(ret)
            total_stake += se; n_voted += 1
            if ret > se: n_plus += 1
            mx_all = max(mx_all, mx)
    ok = (n_voted >= 96) and (total_stake >= 500_000)
    return {"final": bank, "voted": n_voted, "staked": total_stake,
            "hitrate": n_plus / max(n_voted, 1), "maxodds": mx_all, "eligible": ok}


def cmd_sim2():
    d = pd.read_parquet(SUB)
    for c in ["tan1", "fuku1", "fuku2", "wd12", "um12", "tri_pay"]:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
    p23_med = float(d["p23"].median())
    rng = np.random.default_rng(42)
    rows = []
    for end_inst in ["wide", "trio"]:
        for end_f in [0.08, 0.15, 0.25]:
            for target in [1_050_000, 1_150_000]:
                for y in [2024, 2025]:
                    g = d[d["year"] == y]
                    days = [(dt, gg) for dt, gg in g.groupby("date")]
                    finals, hits = [], []
                    for _ in range(400):
                        idx = rng.integers(0, len(days), len(days))
                        r = run_dynamic([days[i] for i in idx], p23_med,
                                        end_inst=end_inst, end_f=end_f, target=target)
                        finals.append(r["final"]); hits.append(r["hitrate"])
                    finals = np.array(finals)
                    rows.append({"inst": end_inst, "end_f": end_f, "target": target // 1000, "year": y,
                                 "E_fin": int(finals.mean()),
                                 "P>1.0M": round(float((finals > 1_000_000).mean()), 3),
                                 "P>1.15M": round(float((finals > 1_150_000).mean()), 3),
                                 "P>1.3M": round(float((finals > 1_300_000).mean()), 3),
                                 "p90": int(np.percentile(finals, 90)),
                                 "p10": int(np.percentile(finals, 10)),
                                 "hit": round(float(np.mean(hits)), 3)})
    o = pd.DataFrame(rows)
    piv = o.pivot_table(index=["inst", "end_f", "target"],
                        columns="year",
                        values=["E_fin", "P>1.0M", "P>1.15M", "P>1.3M", "p90", "p10", "hit"])
    print(piv.round(3).to_string())
    o.to_csv(BASE / "reports" / "gakusei_sim_dynamic.csv", index=False)
    print("[saved] reports/gakusei_sim_dynamic.csv")


def run_exotic(days, sel, f=0.025):
    """先行確定戦略: 馬単3点+三連複4点=7点均等, f×bank/R, 複利。sel(g)->参加サブセット。"""
    bank = 1_000_000
    total_stake = 0; n_voted = 0; n_plus = 0; mx_all = 0.0
    for date, g in days:
        g = sel(g)
        for row in g.itertuples():
            stake = max(700, int(bank * f / 700) * 700)  # 7点均等(100pt単位)
            stake = min(stake, bank - bank % 100)
            per = int(stake / 7 / 100) * 100
            if per < 100 or bank <= 0:
                continue
            se = per * 7
            ret = per * (row.ut_pay + row.tri_pay)
            bank = bank - se + int(ret)
            total_stake += se; n_voted += 1
            if ret > se: n_plus += 1
            for p_ in (row.ut_pay, row.tri_pay):
                if p_ > 0: mx_all = max(mx_all, p_)
    ok = (n_voted >= 96) and (total_stake >= 500_000)
    return {"final": bank, "voted": n_voted, "staked": total_stake,
            "hitrate": n_plus / max(n_voted, 1), "maxodds": mx_all, "eligible": ok}


def cmd_sim3():
    d = pd.read_parquet(SUB)
    for c in ["ut_pay", "tri_pay", "fuku1", "wd12"]:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
    p23_terc = d["p23"].quantile([0.33, 0.67]).values
    gap90 = d["gap"].quantile(0.90)
    SELS = {
        "全R":               lambda g: g,
        "chalk10%除外(先行)": lambda g: g[g["gap"] < gap90],
        "相手強tercile":      lambda g: g[g["p23"] >= p23_terc[1]],
        "相手中強(下1/3除外)": lambda g: g[g["p23"] >= p23_terc[0]],
        "chalk除外∧相手中強":  lambda g: g[(g["gap"] < gap90) & (g["p23"] >= p23_terc[0])],
    }
    rng = np.random.default_rng(42)
    rows = []
    for sname, sel in SELS.items():
        for f in [0.02, 0.025, 0.035]:
            for y in [2024, 2025]:
                g = d[d["year"] == y]
                days = [(dt, gg) for dt, gg in g.groupby("date")]
                finals = []
                elig = 0
                for _ in range(500):
                    idx = rng.integers(0, len(days), len(days))
                    r = run_exotic([days[i] for i in idx], sel, f=f)
                    finals.append(r["final"]); elig += r["eligible"]
                base = run_exotic(days, sel, f=f)
                finals = np.array(finals)
                rows.append({"sel": sname, "f": f, "year": y,
                             "voted": base["voted"], "E": int(finals.mean()),
                             "P>1.06M(入賞)": round(float((finals > 1_060_000).mean()), 3),
                             "P>1.29M(表彰台)": round(float((finals > 1_290_000).mean()), 3),
                             "p90": int(np.percentile(finals, 90)),
                             "p50": int(np.percentile(finals, 50)),
                             "elig%": round(elig / 500, 2)})
    o = pd.DataFrame(rows)
    piv = o.pivot_table(index=["sel", "f"], columns="year",
                        values=["voted", "E", "P>1.06M(入賞)", "P>1.29M(表彰台)", "p50", "p90", "elig%"])
    print(piv.round(3).to_string())
    o.to_csv(BASE / "reports" / "gakusei_sim_exotic.csv", index=False)
    print("[saved] reports/gakusei_sim_exotic.csv")


def run_twophase(days, p23_terc_hi, ph1_days=4, ph1_n=24, ph1_stake=5300,
                 ph2_n=8, ph2_f=0.025, shadow_go=0.90):
    """フェーズ1: 複勝flatで96R+50万pt完走 + シャドー記録。
    フェーズ2: シャドーROI最良の道具(wide/exotic)が閾値超なら攻撃、未満なら店じまい。"""
    bank = 1_000_000
    total_stake = 0; n_voted = 0; n_plus = 0; mx_all = 0.0
    sh = {"wide": [0.0, 0.0], "exotic": [0.0, 0.0]}   # [stake, ret] 仮想
    inst = None
    for di, (date, g) in enumerate(days):
        if di < ph1_days:
            gg = g.sort_values("chaos").head(ph1_n)
            for row in gg.itertuples():
                stake = min(ph1_stake, bank)
                if stake < 100 or bank <= 0:
                    continue
                ret = stake * row.fuku1
                bank = bank - stake + int(ret)
                total_stake += stake; n_voted += 1
                if ret > stake: n_plus += 1
                if row.fuku1 > 0: mx_all = max(mx_all, row.fuku1)
            # シャドー: 相手強top ph2_n で wide / exotic
            ga = g[g["p23"] >= p23_terc_hi].sort_values("p23", ascending=False).head(ph2_n)
            for row in ga.itertuples():
                sh["wide"][0] += 1.0; sh["wide"][1] += row.wd12
                sh["exotic"][0] += 7.0; sh["exotic"][1] += row.ut_pay + row.tri_pay
        else:
            if inst is None:
                rois = {k: (v[1] / v[0] if v[0] else 0.0) for k, v in sh.items()}
                best = max(rois, key=rois.get)
                inst = best if rois[best] >= shadow_go else "stop"
            if inst == "stop":
                continue
            ga = g[g["p23"] >= p23_terc_hi].sort_values("p23", ascending=False).head(ph2_n)
            for row in ga.itertuples():
                stake = max(700, int(bank * ph2_f / 700) * 700)
                stake = min(stake, bank - bank % 100)
                if stake < 700 or bank <= 0:
                    continue
                if inst == "wide":
                    se = stake - stake % 100
                    ret = se * row.wd12
                    if row.wd12 > 0: mx_all = max(mx_all, row.wd12)
                else:
                    per = int(stake / 7 / 100) * 100
                    if per < 100: continue
                    se = per * 7
                    ret = per * (row.ut_pay + row.tri_pay)
                    for p_ in (row.ut_pay, row.tri_pay):
                        if p_ > 0: mx_all = max(mx_all, p_)
                bank = bank - se + int(ret)
                total_stake += se; n_voted += 1
                if ret > se: n_plus += 1
    ok = (n_voted >= 96) and (total_stake >= 500_000)
    return {"final": bank, "voted": n_voted, "staked": total_stake,
            "hitrate": n_plus / max(n_voted, 1), "maxodds": mx_all,
            "eligible": ok, "inst": inst or "n/a"}


def cmd_sim4():
    d = pd.read_parquet(SUB)
    for c in ["ut_pay", "tri_pay", "fuku1", "wd12"]:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
    terc_hi = float(d["p23"].quantile(0.67))
    rng = np.random.default_rng(42)
    TH = [("P>1.02M", 1_020_000), ("P>1.06M", 1_060_000),
          ("P>1.18M", 1_180_000), ("P>1.29M", 1_290_000)]
    rows = []
    for ph2_f in [0.02, 0.03, 0.045]:
        for ph2_n in [6, 8, 10]:
            for go in [0.85, 0.95]:
                for y in [2024, 2025]:
                    g = d[d["year"] == y]
                    days = [(dt, gg) for dt, gg in g.groupby("date")]
                    finals, hits, insts = [], [], []
                    elig = 0
                    for _ in range(600):
                        # フェーズ1の4日は固定(実カレンダー)、フェーズ2日をブートストラップ
                        d2 = days[:4] + [days[4:][i] for i in rng.integers(0, len(days) - 4, len(days) - 4)]
                        r = run_twophase(d2, terc_hi, ph2_f=ph2_f, ph2_n=ph2_n, shadow_go=go)
                        finals.append(r["final"]); hits.append(r["hitrate"])
                        insts.append(r["inst"]); elig += r["eligible"]
                    fn = np.array(finals)
                    from collections import Counter
                    rec = {"ph2_f": ph2_f, "ph2_n": ph2_n, "go": go, "year": y,
                           "med": int(np.percentile(fn, 50)),
                           "p90": int(np.percentile(fn, 90)),
                           "hit": round(float(np.mean(hits)), 2),
                           "inst": Counter(insts).most_common(1)[0][0],
                           "elig%": round(elig / 600, 2)}
                    for lab, t in TH:
                        rec[lab] = round(float((fn > t).mean()), 3)
                    rows.append(rec)
    o = pd.DataFrame(rows)
    piv = o.pivot_table(index=["ph2_f", "ph2_n", "go"], columns="year",
                        values=["med", "P>1.02M", "P>1.06M", "P>1.29M", "hit", "elig%"],
                        aggfunc="first")
    print(piv.round(3).to_string())
    o.to_csv(BASE / "reports" / "gakusei_sim_twophase.csv", index=False)
    print("[saved] reports/gakusei_sim_twophase.csv")




def run_gate5(days, gap90, f=0.02, gate_days=2, k=11):
    """★最終戦略: 初週末(2日)=wide/exotic半々で実測 → 3日目以降=実測ROI優位の道具に全集中。
    選別=chalk10%除外∧相手信頼p23降順top-k/日。f=資金比/R。"""
    def bet_wide(row, bank, ff):
        sw = min(int(bank * ff / 100) * 100, bank)
        return (0, 0.0) if sw < 100 else (sw, sw * row.wd12)
    def bet_exo(row, bank, ff):
        # 馬連1点(r1-r2)+三連複box4 = 5点均等。馬単3点から振替 (2026-07-23):
        # 大標本9,526R同一選別で馬連82.7%>馬単78.4%、ゲートP(賞金圏)+4-5pt両年改善。
        pe = int(bank * ff / 500) * 100
        return (0, 0.0) if (pe < 100 or pe * 5 > bank) else (pe * 5, pe * (row.um12 + row.tri_pay))
    bank = 1_000_000; ts = 0; nv = 0; npl = 0
    perf = {"wide": [0.0, 0.0], "exo": [0.0, 0.0]}; inst = None
    for di, (date, g) in enumerate(days):
        g = g[g["gap"] < gap90].sort_values("p23", ascending=False).head(k)
        for row in g.itertuples():
            if bank <= 0: break
            if di < gate_days:
                sw, rw = bet_wide(row, bank, f / 2); se_, re_ = bet_exo(row, bank, f / 2)
                se = sw + se_; ret = rw + re_
                perf["wide"][0] += sw; perf["wide"][1] += rw
                perf["exo"][0] += se_; perf["exo"][1] += re_
            else:
                if inst is None:
                    r_ = {kk: (v[1] / v[0] if v[0] else 0) for kk, v in perf.items()}
                    inst = max(r_, key=r_.get)
                se, ret = (bet_wide(row, bank, f) if inst == "wide" else bet_exo(row, bank, f))
            if se == 0: continue
            bank = bank - se + int(ret); ts += se; nv += 1
            if ret > se: npl += 1
    return {"final": bank, "voted": nv, "staked": ts, "hitrate": npl / max(nv, 1),
            "eligible": (nv >= 96 and ts >= 500_000), "inst": inst}


def cmd_sim5():
    """最終戦略の再現実行。実測(2026-07-23): 馬連版アーム gate_f2 P(>1.02M)=18.9%/33.0% (2024/2025),
    gate_f3=17.2%/29.6% P(表彰台)6.1%/6.0%。単独道具は外れ年0%=関門式が唯一の regime 頑健解。"""
    d = pd.read_parquet(SUB)
    for c in ["ut_pay", "tri_pay", "fuku1", "wd12"]:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
    gap90 = d["gap"].quantile(0.90)
    rng = np.random.default_rng(42)
    for f in [0.02, 0.03]:
        for y in [2024, 2025]:
            g = d[d["year"] == y]; days = [(dt, gg) for dt, gg in g.groupby("date")]
            finals = []
            for _ in range(800):
                sched = [days[i] for i in rng.integers(0, len(days), len(days))]
                finals.append(run_gate5(sched, gap90, f=f)["final"])
            fn = np.array(finals)
            print(f"gate f={f} {y}: med={int(np.percentile(fn,50)):,} "
                  f"P>1.02M={float((fn>1_020_000).mean()):.3f} "
                  f"P>1.29M={float((fn>1_290_000).mean()):.3f}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    {"build": cmd_build, "sim": cmd_sim, "sim2": cmd_sim2,
     "sim3": cmd_sim3, "sim4": cmd_sim4, "sim5": cmd_sim5}[cmd]()
