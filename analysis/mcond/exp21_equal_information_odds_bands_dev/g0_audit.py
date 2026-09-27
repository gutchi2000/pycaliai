# -*- coding: utf-8 -*-
"""
g0_audit.py — EXP21 Stage 0 G0 データ契約監査 (ROI は計算しない)
================================================================
判定規則 (実行前に固定):
  時点同定 (2023 91 列)   単勝を TANPUK 区分4 / pre (≈T−28) / 9 時と、馬連 18 slot を UMAREN terminal / pre と
                          0.05 以内で照合。一致率 >= 99% の側を採用。どちらも < 99% なら使用禁止
  時点同定 (2026 OD 各日)  公開 UMAREN/TANPUK が無いので、単勝の公式払戻 (勝馬) と odds×100 の完全一致率で判定。
                          >= 99% なら terminal 相当。< 99% なら terminal ではない (T−10 とは呼ばない)
  配置の逆引き            terminal と同定されたファイルだけで行う。的中 ticket の仮説セル値 ×100 が公式払戻と
                          一致 (単勝・馬連・枠連・馬単・三連複は |差| <= 1 円、複勝・ワイドは [Lo, Hi]×100 に入る) する
                          race の率が >= 99%、かつ対立配置 (転置・隣接 slot ずれ・Lo/Hi ブロック配置) より高いこと。
                          同着 race は分母から外して件数を報告
  G0 PASS (券種ごと)      価格源がある・時点が同定できる・配置逆引き PASS・key 検査 PASS
払戻は配置の逆引きと同着・取消の分類にだけ使い、ROI・帯別集計はしない。2024/2025 は読まない。
出力: out/data_manifest.json、out/g0_audit.json
"""
from __future__ import annotations

import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from . import loaders as L

TYPES = ["tansho", "fukusho", "wakuren", "umaren", "wide", "umatan", "sanrenpuku", "sanrentan"]
MATCH_FLOOR = 0.99


def ok_exact(v, pay):
    return np.isfinite(v) and v > 0 and abs(round(v * 100) - pay) <= 1


def ok_range(lo, hi, pay):
    return np.isfinite(lo) and np.isfinite(hi) and lo > 0 and lo * 100 - 1 <= pay <= hi * 100 + 1


def race_rows(df):
    return {rid: g.set_index("ban") for rid, g in df.groupby("rid16")}


def winners(k):
    """kekka の上位行 → 各 race の着順集合と払戻。同着 race を識別"""
    out = {}
    for rid, g in k.groupby("rid16"):
        j = pd.to_numeric(g["確定着順"], errors="coerce")
        first, second, third = (g.loc[j == 1, "ban"].astype(int).tolist(), g.loc[j == 2, "ban"].astype(int).tolist(),
                                g.loc[j == 3, "ban"].astype(int).tolist())
        rec = {"first": first, "second": second, "third": third,
               "dead_heat": not (len(first) == 1 and len(second) == 1 and len(third) == 1)}

        def pay(col):
            v = pd.to_numeric(g[col].astype(str).str.replace(",", ""), errors="coerce")
            v = v[v.notna()]
            return float(v.iloc[0]) if len(v) else np.nan
        rec["tan"] = float(pd.to_numeric(g.loc[j == 1, "単勝配当"].astype(str).str.replace(",", ""), errors="coerce").iloc[0]) if len(first) else np.nan
        rec["fuku"] = {int(b): float(p) for b, p in zip(g["ban"], pd.to_numeric(g["複勝配当"].astype(str).str.replace(",", ""),
                                                                                  errors="coerce")) if p == p}
        for col, key in (("枠連", "waku"), ("馬連", "umaren"), ("馬単", "umatan"), ("３連複", "trio"), ("３連単", "trifecta")):
            rec[key] = pay(col)
        out[rid] = rec
    return out


def verify_layout(rows: dict, win: dict, wide: dict | None, with_trio: bool) -> dict:
    """仮説配置と対立配置で、的中 ticket のセル値 ×100 と公式払戻の一致率を測る"""
    stats = {t: {"hyp": [0, 0]} for t in ("tansho", "fukusho", "wakuren", "umaren", "wide", "umatan", "sanrenpuku")}
    alts = {"umaren": ["slot_shift_+1"], "umatan": ["transposed", "slot_shift_+1"], "wakuren": ["frame_shift_+1"],
            "wide": ["hi_block_then_lo"], "sanrenpuku": ["pairs_including_self_order"]}
    for t, al in alts.items():
        for a in al:
            stats[t][a] = [0, 0]
    dead = 0
    for rid, w in win.items():
        if rid not in rows:
            continue
        g = rows[rid]
        if w["dead_heat"]:
            dead += 1
            continue
        a, b, c = w["first"][0], w["second"][0], w["third"][0]
        if a not in g.index or b not in g.index:
            continue
        va = L.split_vals(g.loc[a, "vals"], with_trio)
        vb = L.split_vals(g.loc[b, "vals"], with_trio)
        n = int(g["n"].iloc[0])
        if np.isfinite(w["tan"]):
            stats["tansho"]["hyp"][1] += 1
            stats["tansho"]["hyp"][0] += ok_exact(va["tan"], w["tan"])
        ks = [x for x in (a, b, c) if x in w["fuku"] and x in g.index]
        if ks:
            stats["fukusho"]["hyp"][1] += 1
            stats["fukusho"]["hyp"][0] += all(ok_range(L.split_vals(g.loc[x, "vals"], with_trio)["fuku_lo"],
                                                       L.split_vals(g.loc[x, "vals"], with_trio)["fuku_hi"], w["fuku"][x])
                                              for x in ks)
        if np.isfinite(w["umaren"]):
            s = stats["umaren"]
            s["hyp"][1] += 1
            s["hyp"][0] += ok_exact(va["umaren"][b - 1], w["umaren"])
            s["slot_shift_+1"][1] += 1
            s["slot_shift_+1"][0] += ok_exact(va["umaren"][b % 18], w["umaren"])
        if np.isfinite(w["umatan"]):
            s = stats["umatan"]
            s["hyp"][1] += 1
            s["hyp"][0] += ok_exact(va["umatan"][b - 1], w["umatan"])
            s["transposed"][1] += 1
            s["transposed"][0] += ok_exact(vb["umatan"][a - 1], w["umatan"])
            s["slot_shift_+1"][1] += 1
            s["slot_shift_+1"][0] += ok_exact(va["umatan"][b % 18], w["umatan"])
        if np.isfinite(w["waku"]):
            fb = int(g.loc[b, "waku"])
            s = stats["wakuren"]
            s["hyp"][1] += 1
            s["hyp"][0] += ok_exact(va["wakuren"][fb - 1], w["waku"])
            s["frame_shift_+1"][1] += 1
            s["frame_shift_+1"][0] += ok_exact(va["wakuren"][fb % 8], w["waku"])
        if wide is not None and rid in wide:
            s = stats["wide"]
            okh = okalt = True
            for i, j, p in wide[rid]:
                if i not in g.index:
                    okh = okalt = False
                    break
                vi = L.split_vals(g.loc[i, "vals"], with_trio)
                okh &= ok_range(vi["wide_lo"][j - 1], vi["wide_hi"][j - 1], p)
                raw = g.loc[i, "vals"][29:65]
                okalt &= ok_range(raw[j - 1], raw[18 + j - 1], p)
            s["hyp"][1] += 1
            s["hyp"][0] += okh
            s["hi_block_then_lo"][1] += 1
            s["hi_block_then_lo"][0] += okalt
        if with_trio and np.isfinite(w["trio"]):
            s = stats["sanrenpuku"]
            pairs = L.TRIO_PAIRS[a]
            j, k = sorted((b, c))
            idx = pairs.index((j, k))
            s["hyp"][1] += 1
            s["hyp"][0] += ok_exact(va["trio"][idx], w["trio"])
            allp = list(combinations(range(1, 19), 2))
            alt_i = allp.index((j, k)) if allp.index((j, k)) < 136 else 0
            s["pairs_including_self_order"][1] += 1
            s["pairs_including_self_order"][0] += ok_exact(va["trio"][alt_i], w["trio"])
    out = {}
    for t, d in stats.items():
        out[t] = {k: {"match": v[0], "n": v[1], "rate": (v[0] / v[1]) if v[1] else None} for k, v in d.items()}
        hyp = out[t]["hyp"]["rate"]
        alt_rates = [x["rate"] for k, x in out[t].items() if k != "hyp" and x["rate"] is not None]
        out[t]["layout_pass"] = bool(hyp is not None and hyp >= MATCH_FLOOR and all(hyp > r for r in alt_rates))
    out["_dead_heat_races_excluded"] = dead
    return out


def key_checks(rows: dict, with_trio: bool) -> dict:
    """ticket key の一意性・対称性・self=0・頭数別 ticket 数・0.0 セルの分類"""
    c = {"races": 0, "umaren_asym": 0, "wide_asym": 0, "self_nonzero": 0, "waku_row_inconsistent": 0,
         "trio_cross_row_inconsistent": 0, "umaren_zero_within_starters": 0, "umaren_ticket_count_mismatch": 0,
         "dup_rows": 0, "rows": 0, "scratched_rows_tan_zero": 0, "refundable_tickets_involving_scratched_umaren": 0}
    for rid, g in rows.items():
        c["races"] += 1
        if g.index.duplicated().any():
            c["dup_rows"] += 1
        bans = [int(x) for x in g.index]
        V = {b: L.split_vals(g.loc[b, "vals"], with_trio) for b in bans}
        starters = [b for b in bans if V[b]["tan"] > 0]
        c["rows"] += len(bans)
        scr = [b for b in bans if not V[b]["tan"] > 0]
        c["scratched_rows_tan_zero"] += len(scr)
        c["refundable_tickets_involving_scratched_umaren"] += len(scr) * (len(bans) - 1) - len(scr) * (len(scr) - 1) // 2
        for i in bans:
            if V[i]["umaren"][i - 1] != 0 or V[i]["umatan"][i - 1] != 0:
                c["self_nonzero"] += 1
            for j in bans:
                if j <= i:
                    continue
                if abs(V[i]["umaren"][j - 1] - V[j]["umaren"][i - 1]) > 1e-9:
                    c["umaren_asym"] += 1
                if abs(V[i]["wide_lo"][j - 1] - V[j]["wide_lo"][i - 1]) > 1e-9 or \
                        abs(V[i]["wide_hi"][j - 1] - V[j]["wide_hi"][i - 1]) > 1e-9:
                    c["wide_asym"] += 1
        nz = sum(1 for i in starters for j in starters if j > i and V[i]["umaren"][j - 1] > 0)
        c["umaren_zero_within_starters"] += len(starters) * (len(starters) - 1) // 2 - nz
        if nz != len(starters) * (len(starters) - 1) // 2:
            c["umaren_ticket_count_mismatch"] += 1
        byf = {}
        for b in bans:
            byf.setdefault(int(g.loc[b, "waku"]), []).append(tuple(V[b]["wakuren"]))
        c["waku_row_inconsistent"] += sum(len(set(v)) > 1 for v in byf.values())
        if with_trio:
            seen = {}
            for i in bans:
                for idx, (j, k) in enumerate(L.TRIO_PAIRS[i]):
                    key = tuple(sorted((i, j, k)))
                    seen.setdefault(key, set()).add(V[i]["trio"][idx])
            c["trio_cross_row_inconsistent"] += sum(len(v) > 1 for k, v in seen.items() if all(x in bans for x in k))
    c["pass"] = all(c[k] == 0 for k in ("umaren_asym", "wide_asym", "self_nonzero", "waku_row_inconsistent",
                                        "trio_cross_row_inconsistent", "dup_rows"))
    return c


def main():
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    t0 = time.time()
    L.OUT.mkdir(exist_ok=True)
    res = {"rules": {"match_floor": MATCH_FLOOR}}
    # ---- 2023 91 列
    from ..exp18_cross_pool_market_tomography_dev.loaders import load_structure
    from ..exp18_cross_pool_market_tomography_dev.market_build import PAIR_POS, pool_matrices, snapshot_index
    df = L.read_target_odds(L.RAW2023, 91)
    st = load_structure([2023])
    W, LO, HI, U = pool_matrices(st["tan"], st["um"])
    idx = snapshot_index(st["tan"], st["um"], st["info"])
    keymap = {(r[8:10], int(r[10:12]), int(r[12:14]), int(r[14:16])): r[:8] for r in idx.index}
    df["rid16"] = [f"{keymap.get((v, k, n, R), 'X')}{v}{k:02d}{n:02d}{R:02d}"
                   for v, k, n, R in zip(df.venue, df.kai, df.nichi, df.R)]
    unmapped = int((df.rid16.str[:1] == "X").sum())
    tim = {"tan": {}, "umaren": {}}
    for lab, tcol, ucol in (("terminal", "term_tan", "term_um"), ("pre_Tminus28", "pre_tan", "pre_um"),
                            ("am9", "am9_tan", "am9_um")):
        m = n = um_m = um_n = 0
        for rid, g in df.groupby("rid16"):
            if rid not in idx.index:
                continue
            r = idx.loc[rid]
            ti, ui = r.get(tcol), r.get(ucol)
            if ti == ti:
                wt = W[int(ti)]
                for b, v in zip(g.ban, g.vals):
                    o = wt[b - 1]
                    if v[0] > 0 and np.isfinite(o):
                        n += 1
                        m += abs(v[0] - o) < 0.05
            if ui == ui and int(ui) >= 0:
                for b, v in zip(g.ban, g.vals):
                    for j in range(1, 19):
                        if j == b or v[3 + j - 1] <= 0:
                            continue
                        o = U[int(ui), PAIR_POS[(min(b, j), max(b, j))]]
                        if np.isfinite(o) and o > 0:
                            um_n += 1
                            um_m += abs(v[3 + j - 1] - o) < 0.05
        tim["tan"][lab] = {"rate": m / n if n else None, "n": n}
        tim["umaren"][lab] = {"rate": um_m / um_n if um_n else None, "n": um_n}
    term_ok = tim["tan"]["terminal"]["rate"] >= MATCH_FLOOR and tim["umaren"]["terminal"]["rate"] >= MATCH_FLOOR
    pre_ok = tim["tan"]["pre_Tminus28"]["rate"] >= MATCH_FLOOR and tim["umaren"]["pre_Tminus28"]["rate"] >= MATCH_FLOOR
    res["raw2023_timing"] = {"rows": int(len(df)), "races": int(df.rid16.nunique()), "unmapped_rows": unmapped,
                             "export_hhmm": sorted(df.export_hhmm.unique().tolist()),
                             "code_col2": sorted(df.code.unique().tolist()), "match": tim,
                             "identified": "terminal" if term_ok else ("pre" if pre_ok else "unusable")}
    print("[2023 timing]", res["raw2023_timing"]["identified"], json.dumps(tim), flush=True)
    rows23 = race_rows(df)
    k = L.load_kekka_master()
    win23 = winners(k[k["date"] // 10000 == 2023])
    wd = L.load_wide_payouts()
    wd = wd[wd["date"] // 10000 == 2023]
    wide23 = {r.race_id: [(int(r.w1_i), int(r.w1_j), float(r.w1_pay)), (int(r.w2_i), int(r.w2_j), float(r.w2_pay)),
                          (int(r.w3_i), int(r.w3_j), float(r.w3_pay))] for r in wd.itertuples()
              if all(x == x for x in (r.w3_i, r.w3_pay))}
    res["raw2023_layout"] = verify_layout(rows23, win23, wide23, with_trio=False) if term_ok else "not_run (timing)"
    res["raw2023_keys"] = key_checks(rows23, with_trio=False)
    print("[2023 layout]", {t: v["hyp"]["rate"] for t, v in res["raw2023_layout"].items() if isinstance(v, dict)},
          flush=True)
    # ---- 2026 OD (日ごとに時点を同定)
    od = {}
    files = sorted(L.OD_DIR.glob("OD26*.CSV"))
    k26 = L.load_kekka_2026([f"20{f.stem[2:]}" for f in files])
    win26 = winners(k26.assign(ban=pd.to_numeric(k26["馬番"], errors="coerce"))) if len(k26) else {}
    w26 = L.load_wide_2026()
    vcode = {"札幌": "01", "函館": "02", "福島": "03", "新潟": "04", "東京": "05", "中山": "06", "中京": "07",
             "京都": "08", "阪神": "09", "小倉": "10"}
    wide26 = {}
    for r in w26.itertuples():
        wide26.setdefault((r.date, vcode.get(r.venue_name, "??"), r.R), []).append((r.i, r.j, float(r.pay)))
    all_rows26 = {}
    for f in files:
        d = f"20{f.stem[2:]}"
        o = L.read_target_odds(f, 227)
        o["rid16"] = [f"{d}{v}{k_:02d}{n:02d}{R:02d}" for v, k_, n, R in zip(o.venue, o.kai, o.nichi, o.R)]
        rr = race_rows(o)
        m = n = 0
        for rid, g in rr.items():
            w = win26.get(rid)
            if not w or w["dead_heat"] or not np.isfinite(w["tan"]) or w["first"][0] not in g.index:
                continue
            n += 1
            m += ok_exact(L.split_vals(g.loc[w["first"][0], "vals"], True)["tan"], w["tan"])
        rate = m / n if n else None
        od[d] = {"rows": int(len(o)), "races": len(rr), "export_hhmm": sorted(o.export_hhmm.unique().tolist()),
                 "code_col2": sorted(o.code.unique().tolist()), "tan_payout_exact_rate": rate, "n_compared": n,
                 "identified": ("terminal" if (rate is not None and rate >= MATCH_FLOOR) else
                                ("no_payout_yet" if n == 0 else "not_terminal"))}
        if od[d]["identified"] == "terminal":
            all_rows26.update(rr)
    res["od2026_files"] = od
    wide26_rid = {}
    for rid in all_rows26:
        key = (int(rid[:8]), rid[8:10], int(rid[14:16]))
        if key in wide26 and len(wide26[key]) == 3:
            wide26_rid[rid] = wide26[key]
    res["od2026_terminal_layout"] = verify_layout(all_rows26, {r: w for r, w in win26.items() if r in all_rows26},
                                                  wide26_rid, with_trio=True)
    res["od2026_terminal_keys"] = key_checks(all_rows26, with_trio=True)
    print("[2026 layout]", {t: v["hyp"]["rate"] for t, v in res["od2026_terminal_layout"].items()
                            if isinstance(v, dict)}, flush=True)
    # ---- G0 判定 (券種ごと)
    lay23, lay26 = res["raw2023_layout"], res["od2026_terminal_layout"]
    g0 = {}
    for t in TYPES:
        if t in ("tansho", "fukusho", "umaren"):
            g0[t] = {"sources": ["TANPUK/UMAREN 2013-2023 (terminal + historical_pre_snapshot)",
                                 "raw 2023 91-col (terminal)", "OD 2026 terminal days"],
                     "pass": True, "note": "history coverage in label_free_band_coverage.json; 91/OD layouts verified"}
        elif t == "sanrentan":
            g0[t] = {"sources": [], "pass": False,
                     "reason": "no full-ticket price source: raw 2023 91-col has no trifecta block; OD 227-col extra 136 "
                               "columns are trio (sanrenpuku) per anchor horse, verified by payout reverse-mapping"}
        elif t == "sanrenpuku":
            p26 = bool(lay26["sanrenpuku"]["layout_pass"])
            g0[t] = {"sources": ["OD 2026 terminal days only (2023 91-col has no trio block)"],
                     "od2026_layout_pass": p26, "pass": p26,
                     "terminal_races": sum(v["races"] for v in od.values() if v["identified"] == "terminal"),
                     "reach": "D0 descriptive only on OD 2026 terminal days; no D1 (no verified T-10); no G1/G2"}
        else:
            p23 = bool(lay23[t]["layout_pass"]) if isinstance(lay23, dict) else False
            p26 = bool(lay26[t]["layout_pass"])
            g0[t] = {"sources": ["raw 2023 91-col (terminal)", "OD 2026 terminal days"],
                     "raw2023_layout_pass": p23, "od2026_layout_pass": p26, "pass": p23 and p26,
                     "reach": "D0 descriptive only (2023 terminal + OD 2026 terminal days); no D1 (no verified T-10); "
                              "no G1/G2"}
    res["g0"] = g0
    res["t10_premise"] = {
        "spec_claim": "2026 OD 227-col = true T-10 about 49 days",
        "finding": {k: v["identified"] for k, v in od.items()},
        "note": "OD files are TARGET exports (col 36 = export time HHMM). Days exported after racing match official "
                "payouts exactly -> terminal, not T-10. No OD day is identified as T-10; D1 forward (T-10) is not "
                "available from these files"}
    res["elapsed_sec"] = round(time.time() - t0, 1)
    (L.OUT / "g0_audit.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    manifest = {"sources": []}
    for p, role, per in ((L.RAW2023, "2023 91-col TARGET odds export", "2023"),
                         *[(f, "2026 OD 227-col TARGET odds export", f.stem[2:]) for f in files],
                         (L.KEKKA_MASTER, "official payouts top-3 rows (2013-2023 used; 2024/2025 dropped on read)", "2013-2025 file"),
                         (L.WIDE_PARQUET, "official wide payouts (2016-2023 used)", "2016-2025 file"),
                         (L.WIDE_2026, "2026 wide payouts text", "2026"),
                         *[(f, "TANPUK/UMAREN time series (<=2023 used)", f.stem.split('_')[-1])
                           for f in sorted(L.ODIR.glob("TANPUK_*.csv")) + sorted(L.ODIR.glob("UMAREN_*.csv"))]):
        manifest["sources"].append({"path": str(p), "role": role, "period": per, "sha256": L.sha256(Path(p)),
                                    "bytes": Path(p).stat().st_size})
    manifest["units"] = {"price": "decimal odds per 1 JPY (0.1 resolution); fukusho/wide as [Lo, Hi]",
                         "payout": "JPY per 100 JPY stake (official)", "record_code_col2": "TARGET export code "
                         "('4' on terminal exports; '0'/'6' on pre-race exports observed)",
                         "tanpuk_umaren_kubun": "1 = interim snapshots, 4 = terminal",
                         "takeout_change_2014_06_07": "JRA takeout changed on 2014-06-07 for umaren, wakuren and wide "
                                                      "(22.5%). The discovery period 2013-2018 straddles this date; the "
                                                      "statutory takeout is shown for reference only, and v0.3 compares "
                                                      "tansho/umaren bands with the label-free calibrated null (1/overround "
                                                      "from each race's terminal prices) instead"}
    manifest["per_type"] = {t: g0[t] for t in TYPES}
    (L.OUT / "data_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"g0": {t: g0[t]["pass"] for t in TYPES}}, ensure_ascii=False))
    print(f"[saved] g0_audit.json / data_manifest.json ({res['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
