# -*- coding: utf-8 -*-
"""
baselines.py — 仕様 §5 の比較相手 B0〜B3（B4 は policy.py の市場版判断 AI）
==========================================================================
すべて「買い目帳」book（evaluate.py の形式）を返す。母集団は load_candidates() のレース（決済不能 16R 除外済み）。

B0 null   : ◎（p_win 最大 = 単勝候補 1 位）の複勝を 1 点、全レース、予算 B 全額
B1 a95    : 凍結方策 data/shadow_policies/a95_equal_v1.json を a95_engine.build で流用。
            順序 = v6 p_win 降順（base.parquet の v6_pwin、substrate の p_model と同じ尺度）、
            帯 = AI1位の T−28 単勝オッズ（substrate の odds_decision = base の mkt_odds_pre）。
            **a95 の参戦ガード（混戦度・頭数・◎p_win・◎オッズ・障害）は仕様 §5 に無いので掛けず、全レースに適用**
            （紙上台帳 CW-10 と同じ扱い）。枠連券は substrate の sub_wakuren 行（2021 以降）で決済し、
            枠連の判断時点オッズ・払戻の無いレース（2017-2020 全部と 2021 以降の一部）は枠連を「発売なし」と
            同じ扱いで外し、残りの券種へ均等配分し直す（a95 の sellable_kinds と同じ考え方）。
            枠番は master_v2 の実枠番。a95 の他の券が substrate の候補（モデル上位 K）に無い場合は件数を記録して落とす。
B2 市場人気: 判断時点（T−28）の単勝 1 番人気（全馬の T−28 単勝オッズ最小、同値は馬番の小さい方）の複勝を 1 点。
            払戻は base.parquet の fpay（substrate の複勝払戻と全行一致を確認済み）。1 番人気が完走馬に居ない場合:
            確定断面で単勝オッズが無い（取消・除外）→ 返還（払戻 100）、ある（競走中止）→ 0。
B3 確率順 : 各レースで p_model 最大の候補券（券種横断）を 1 点。学習なし。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.bet_decision_ai import data as D


def _single_ticket_book(rows: pd.DataFrame, B: int, name: str) -> pd.DataFrame:
    return pd.DataFrame({"rid16": rows["rid16"].to_numpy(), "date": rows["date"].astype(str).to_numpy(),
                         "bet_type": rows["bet_type"].to_numpy(), "combo": rows["combo"].to_numpy(),
                         "stake": np.full(len(rows), B, np.int64), "payout100": rows["payout"].to_numpy(float),
                         "p_model": rows["p_model"].to_numpy(float), "lpq": rows["log_p_over_q"].to_numpy(float),
                         "policy": name})


def b0_null(cand: pd.DataFrame, B: int) -> pd.DataFrame:
    a1 = cand[(cand["bet_type"] == "tan") & (cand["cand_rank"] == 1)][["rid16", "h1"]]
    f = cand[cand["bet_type"] == "fuku"].merge(a1, on=["rid16", "h1"], how="inner")
    assert f["rid16"].is_unique
    miss = set(a1["rid16"]) - set(f["rid16"])
    assert not miss, f"◎の複勝が候補に無いレース {len(miss)}"
    return _single_ticket_book(f, B, "B0_null")


def b3_prob_first(cand: pd.DataFrame, B: int) -> pd.DataFrame:
    i = cand.groupby("rid16")["p_model"].idxmax()
    return _single_ticket_book(cand.loc[i], B, "B3_prob_first")


def b2_market_fav(cand: pd.DataFrame, base_mkt: pd.DataFrame, tan_snap: dict, B: int) -> tuple[pd.DataFrame, dict]:
    rids = cand["rid16"].unique()
    dates = cand.groupby("rid16")["date"].first()
    bm = {rid: g for rid, g in base_mkt.groupby("rid16", sort=False)}
    rows, info = [], {"fav_not_finisher_scratch": 0, "fav_not_finisher_dnf": 0, "no_snapshot": 0}
    for rid in rids:
        snap = tan_snap.get(rid)
        if snap is None:
            info["no_snapshot"] += 1
            continue
        pre, fin = snap
        ok = np.flatnonzero(pre > 1.0)
        fav = int(ok[np.argmin(pre[ok])]) + 1          # argmin は同値で最初（馬番の小さい方）
        g = bm[rid]
        hit = g[g["ban"] == fav]
        if len(hit):
            pay = float(hit["fpay"].iloc[0])
        else:
            started = fin is not None and fin[fav - 1] > 1.0
            pay = 0.0 if started else 100.0
            info["fav_not_finisher_dnf" if started else "fav_not_finisher_scratch"] += 1
        rows.append((rid, str(dates[rid]), "fuku", f"{fav:02d}", B, pay, np.nan, np.nan, "B2_market_fav"))
    return pd.DataFrame(rows, columns=["rid16", "date", "bet_type", "combo", "stake", "payout100", "p_model", "lpq",
                                       "policy"]), info


A95_KIND = {"tan": "tan", "fuku": "fuku", "umaren": "umaren", "umatan": "umatan", "wide": "wide",
            "sanpuku": "sanrenpuku", "sanrentan": "sanrentan", "wakuren": "wakuren"}


def b1_a95(cand: pd.DataFrame, base_mkt: pd.DataFrame, wak: pd.DataFrame, waku_map: dict, B: int,
           ) -> tuple[pd.DataFrame, dict]:
    import a95_engine
    from shadow_a95 import equal_stakes, load_policy
    pol = load_policy()
    unit = int(pol["unit_yen"])
    rids = cand["rid16"].unique()
    dates = cand.groupby("rid16")["date"].first()
    cg = {rid: g for rid, g in cand.groupby("rid16", sort=False)}
    bm = {rid: g for rid, g in base_mkt.groupby("rid16", sort=False)}
    wg = {rid: g for rid, g in wak.groupby("rid16", sort=False)}
    info = {"races": 0, "a95_error": 0, "wakuren_planned": 0, "wakuren_dropped_no_data": 0,
            "ticket_not_in_candidates": 0, "band_counts": {}, "odds_mismatch": 0}
    out = []
    for rid in rids:
        g = cg[rid]
        b = bm[rid]
        horses = [{"umaban": int(r.ban), "p_win": float(r.v6_pwin), "tansho_odds": float(r.mkt_odds_pre)}
                  for r in b.itertuples(index=False)]
        try:
            res = a95_engine.build(horses, budget=B, policy=pol)
        except a95_engine.A95Error:
            info["a95_error"] += 1
            continue
        info["races"] += 1
        info["band_counts"][res["band"]] = info["band_counts"].get(res["band"], 0) + 1
        a1, a2, a3 = res["a1"], res["a2"], res["a3"]
        t1 = g[(g["bet_type"] == "tan") & (g["cand_rank"] == 1)]
        if int(t1["h1"].iloc[0]) != a1 or abs(float(t1["odds_decision"].iloc[0]) - res["o1"]) > 1e-6:
            info["odds_mismatch"] += 1
        kinds = [t["kind"] for t in res["tickets"]]
        rows = {}
        for k in kinds:
            bt = A95_KIND[k]
            if bt == "wakuren":
                info["wakuren_planned"] += 1
                wm = waku_map.get(rid, {})
                w = wg.get(rid)
                r = None
                if w is not None and a1 in wm and a2 in wm:
                    wa, wb = sorted((wm[a1], wm[a2]))
                    hh = w[(w["h1"] == wa) & (w["h2"] == wb)]
                    r = hh.iloc[0] if len(hh) else None
                if r is None:
                    info["wakuren_dropped_no_data"] += 1
                    continue
                rows[k] = (bt, r["combo"], float(r["payout"]), np.nan, np.nan)
                continue
            gg = g[g["bet_type"] == bt]
            if bt in ("tan", "fuku"):
                hh = gg[gg["h1"] == a1]
            elif bt in ("umaren", "wide"):
                x, y = sorted((a1, a2))
                hh = gg[(gg["h1"] == x) & (gg["h2"] == y)]
            elif bt == "umatan":
                hh = gg[(gg["h1"] == a1) & (gg["h2"] == a2)]
            elif bt == "sanrenpuku":
                x, y, z = sorted((a1, a2, a3))
                hh = gg[(gg["h1"] == x) & (gg["h2"] == y) & (gg["h3"] == z)]
            else:
                hh = gg[(gg["h1"] == a1) & (gg["h2"] == a2) & (gg["h3"] == a3)]
            if not len(hh):
                info["ticket_not_in_candidates"] += 1
                continue
            r = hh.iloc[0]
            rows[k] = (bt, r["combo"], float(r["payout"]), float(r["p_model"]), float(r["log_p_over_q"]))
        keep = [k for k in kinds if k in rows]
        if not keep:
            continue
        cap = B // unit * unit
        if len(keep) != len(kinds):
            st = equal_stakes(keep, cap, unit)
        else:
            st = {t["kind"]: int(t["stake"]) for t in res["tickets"]}
        for k in keep:
            bt, combo, pay, p, lpq = rows[k]
            out.append((rid, str(dates[rid]), bt, combo, int(st[k]), pay, p, lpq, "B1_a95"))
    book = pd.DataFrame(out, columns=["rid16", "date", "bet_type", "combo", "stake", "payout100", "p_model", "lpq",
                                      "policy"])
    return book, info
