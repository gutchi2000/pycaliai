"""券種別に「戦える土俵」があるかの探索 (分析専用、本番経路には不干渉)。

位置づけ: 探索。対象期間 (2026-05-02〜11-15 の凍結 bundle) は過去の分析で何度も使っているので、
良い数字が出ても確定扱いにしない。閾値はすべて事前に固定し、事後に選ばない。

1. 複勝 (T-10): EV = bundle の p_sho × T-10 の複勝オッズ下限。閾値 1.0 / 1.1 / 1.15、
   EV ≥ 閾値の全馬と、EV 最大の 1 頭 (EV ≥ 閾値のとき)。T-10 オッズの取り方は
   analysis.t10_market_blend_check.t10_odds と同じ (発走 5〜15 分前の値だけ)。
2. 馬単・馬連 (・ワイド): モデルの組み合わせ確率 = bundle の p_win (レース内正規化) から
   Plackett-Luce (= Harville と同じ式)。市場の組み合わせ確率 = 最終単勝オッズの逆数の正規化から Harville。
   比 (モデル ÷ 市場) が 1.2 / 1.5 / 2.0 以上の組を 100 円ずつ買う。
   ワイドは「2 頭とも 3 着以内」の確率 (7 頭以下でもワイドは 3 着まで払われることを確認済み)。
3. 比較: 各券種の全通り買い (複勝は T-10 オッズのある全馬)。

決済は build_site と同じ規則: 取消 (発売前) の馬は対象から外す、最終オッズのある着順 0 の馬
(中止/発売後除外、区別不能) を含む券は集計外、同着は JRA ルールで的中組を列挙
(build_site._winning_combos / _combo_pay)。ワイドは data/kekka/wide_kekka.csv の払戻。
組み合わせの対象は、取消以外の全馬に最終単勝オッズと p_win があるレースだけ。

使い方:
  python -m analysis.bet_type_terrain --data-root E:/PyCaLiAI --out-json <path>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import build_site
from analysis import ev_threshold_backtest as bt
from analysis.t10_market_blend_check import post_times, t10_odds

PLACE_THRESHOLDS = [1.0, 1.1, 1.15]
RATIO_THRESHOLDS = [1.2, 1.5, 2.0]
SPLIT = "20260815"     # 参考の前後半 (前回までと同じ境界)


def harville_pairs(p: np.ndarray) -> np.ndarray:
    """P[i, j] = i が 1 着・j が 2 着の確率 (Harville / PL)。"""
    P = np.outer(p, p) / (1.0 - p)[:, None]
    np.fill_diagonal(P, 0.0)
    return P


def wide_pairs(p: np.ndarray) -> np.ndarray:
    """W[i, j] = i と j がともに 3 着以内の確率 (PL)。"""
    n = len(p)
    a = p[:, None, None]
    b = p[None, :, None] / (1.0 - p[:, None, None])
    c = p[None, None, :] / np.clip(1.0 - p[:, None, None] - p[None, :, None], 1e-12, None)
    T = a * b * c
    idx = np.arange(n)
    T[idx, idx, :] = 0
    T[idx, :, idx] = 0
    T[:, idx, idx] = 0
    W = T.sum(2) + T.sum(1) + T.sum(0)   # i が j より前に来る 3 通りの位置
    W = W + W.T
    np.fill_diagonal(W, 0.0)
    return W


def combo_payout(res: dict, kind: str, combo: tuple, win_sets: dict) -> float | None:
    """100 円あたりの払戻。None = 集計外 (中止/発売後除外の馬を含む、ワイド払戻未取込)。"""
    unknown = {int(u) for u in res.get("no_finish") or []}
    if set(combo) & unknown:
        return None
    if kind == "wide":
        wide = (res.get("pays") or {}).get("wide") or {}
        if not wide:
            return None
        return float(wide.get(f"{min(combo)}-{max(combo)}") or 0)
    ordered = kind == "umatan"
    if kind not in win_sets:
        win_sets[kind] = build_site._winning_combos(res, 2, ordered)
    ws = win_sets[kind]
    if combo in ws:
        return float(build_site._combo_pay(res, kind, combo) or 0)
    return 0.0


def load_combo_races(root: Path, dates: list[str]):
    frozen = root / "reports" / "cowork_input_frozen"
    build_site.KEKKA_DIR = root / "data" / "kekka"
    wide_data = build_site.parse_wide_kekka()
    out, skipped = [], {"no_result": 0, "missing_odds_or_p": 0}
    for d in dates:
        b = json.loads((frozen / f"{d}_bundle.json").read_text(encoding="utf-8"))
        era = bt.build_site_model_era(d, b)
        res_all = build_site.parse_kekka(d, wide_data)
        fin_all = bt.final_win_odds(d, root / "data" / "kekka")
        for race in b.get("races", []):
            rid = str(race.get("race_id", ""))[:16]
            res = res_all.get(rid)
            if not res or not res.get("top3"):
                skipped["no_result"] += 1
                continue
            scratched = {int(u) for u in res.get("scratched") or []}
            fin = fin_all.get(rid, {})
            hs = [h for h in race.get("horses", []) if int(h.get("umaban", 0)) not in scratched]
            uma = [int(h["umaban"]) for h in hs]
            pw = [h.get("p_win") for h in hs]
            ow = [fin.get(u) for u in uma]
            if len(uma) < 2 or any(x is None for x in pw) or any(x is None or x <= 0 for x in ow):
                skipped["missing_odds_or_p"] += 1
                continue
            m = np.array(pw, float)
            m = m / m.sum()
            q = 1.0 / np.array(ow, float)
            q = q / q.sum()
            out.append({"date": d, "rid": rid, "era": era, "uma": uma, "m": m, "q": q, "res": res})
    return out, skipped


def combo_bets(races: list[dict]) -> dict[str, list[dict]]:
    """券種ごとに全組の (rid, date, ratio, model_p, market_p, ret, no_finish)。"""
    out = {"umatan": [], "umaren": [], "wide": []}
    for r in races:
        uma, m, q, res = r["uma"], r["m"], r["q"], r["res"]
        Pm, Pq = harville_pairs(m), harville_pairs(q)
        Wm, Wq = wide_pairs(m), wide_pairs(q)
        ws: dict = {}
        n = len(uma)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                c = (uma[i], uma[j])
                pay = combo_payout(res, "umatan", c, ws)
                out["umatan"].append(_rec(r, Pm[i, j], Pq[i, j], pay))
                if i < j:
                    cs = tuple(sorted(c))
                    pay = combo_payout(res, "umaren", cs, ws)
                    out["umaren"].append(_rec(r, Pm[i, j] + Pm[j, i], Pq[i, j] + Pq[j, i], pay))
                    pay = combo_payout(res, "wide", cs, ws)
                    out["wide"].append(_rec(r, Wm[i, j], Wq[i, j], pay))
    return out


def _rec(r, pm, pq, pay):
    return {"rid": r["rid"], "date": r["date"], "era": r["era"], "pm": float(pm), "pq": float(pq),
            "ratio": float(pm / pq) if pq > 0 else np.inf,
            "ret": 0.0 if pay is None else pay, "no_finish": pay is None}


def describe(bets: list[dict]) -> dict:
    s = bt.summarize(bets)
    live = [b for b in bets if not b["no_finish"]]
    if live:
        s["mean_market_p"] = float(np.mean([b["pq"] for b in live]))
        s["mean_model_p"] = float(np.mean([b["pm"] for b in live]))
        s["median_implied_odds"] = float(np.median([0.8 / b["pq"] for b in live]))   # 控除 20% を仮定した目安
    s["n_unsettled"] = int(len(bets) - len(live))
    return s


def by_half(bets: list[dict]) -> dict:
    return {"fit_half(〜08-15)": describe([b for b in bets if b["date"] <= SPLIT]),
            "val_half(08-16〜)": describe([b for b in bets if b["date"] > SPLIT])}


ODDS_BANDS = [0, 10, 30, 100, 300, 1000, np.inf]   # 市場確率から見た配当の目安 (0.8 / 市場確率、倍)


def band_table(bets: list[dict]) -> list[dict]:
    """参考 (記述のみ): 配当の目安の帯ごとに、全通り買いと比ルールの回収率を並べる。
    同じ帯の中で比ルールが全通りより良ければ「人気薄を選ぶ効果」以外の何かがある。"""
    rows = []
    for lo, hi in zip(ODDS_BANDS[:-1], ODDS_BANDS[1:]):
        sub = [b for b in bets if lo <= 0.8 / b["pq"] < hi]
        r = {"band": f"{lo:g}-{hi:g}", "all": _brief(sub)}
        for t in RATIO_THRESHOLDS:
            r[f"ratio>={t}"] = _brief([b for b in sub if b["ratio"] >= t])
        rows.append(r)
    return rows


def _brief(bets: list[dict]) -> dict:
    s = bt.summarize(bets)
    return {k: s.get(k) for k in ("n_bets", "n_races", "hit_rate", "roi", "roi_ci95")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="E:/PyCaLiAI")
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    root = Path(a.data_root)
    out: dict = {"design": {"place_thresholds": PLACE_THRESHOLDS, "ratio_thresholds": RATIO_THRESHOLDS,
                            "position": "exploratory"}}

    # ---- 1. 複勝 (T-10) ----
    rows, info = bt.load_rows(root)
    out["info"] = {"dates_used": info["dates_used"], "dates_excluded": info["dates_no_frozen"]}
    df = pd.DataFrame(rows)
    df = df[df["p_sho"].notna()].copy()
    rids = sorted(df["rid"].unique())
    posts = post_times(root, sorted(df["date"].unique()))
    fodds, src, stat = t10_odds(root, rids, posts, key="fukusho")
    df["fo10"] = [fodds.get(r, {}).get(u) for r, u in zip(df["rid"], df["uma"])]
    have = df[df["fo10"].notna()].copy()
    n_field = df.groupby("rid")["uma"].transform("size")
    have["small_field"] = n_field.loc[have.index] <= 7
    hr = have.drop_duplicates("rid")
    out["place_coverage"] = {
        "races_frozen": int(df["rid"].nunique()), "races_with_t10_place": int(hr["rid"].nunique()),
        "horses_with_t10_place": int(len(have)),
        "dates": [hr["date"].min(), hr["date"].max(), int(hr["date"].nunique())],
        "by_half_races": {"fit": int((hr["date"] <= SPLIT).sum()), "val": int((hr["date"] > SPLIT).sum())},
        "sources": {k: int(v) for k, v in stat.items()},
        "eras": hr["era"].value_counts().to_dict(),
        "small_field_races(<=7)": int(hr[hr["small_field"]]["rid"].nunique()),
    }
    recs = have.to_dict("records")
    for r in recs:
        r["p_sho"], r["fo_low"] = r["p_sho"], r["fo10"]
    place = {}
    for kind in ("fuku_all", "fuku_top1"):
        for t in PLACE_THRESHOLDS:
            sel = bt.select(recs, kind, t)
            place[f"{kind}@{t}"] = {"all": bt.summarize(sel),
                                    "fit_half": bt.summarize([b for b in sel if b["date"] <= SPLIT]),
                                    "val_half": bt.summarize([b for b in sel if b["date"] > SPLIT])}
    allp = [{**r, "ret": r["fuku_ret"]} for r in recs]
    place["all_horses"] = {"all": bt.summarize(allp),
                           "fit_half": bt.summarize([b for b in allp if b["date"] <= SPLIT]),
                           "val_half": bt.summarize([b for b in allp if b["date"] > SPLIT])}
    out["place_t10"] = place
    print("place coverage", json.dumps(out["place_coverage"], ensure_ascii=False))
    for k, v in place.items():
        s = v["all"]
        if s.get("n_bets"):
            print(f"place {k:16s} bets={s['n_bets']:5d} R={s['n_races']:4d} hit={s['hit_rate']*100:5.1f}% "
                  f"roi={s['roi']*100:6.1f}% CI[{s['roi_ci95'][0]*100:.0f},{s['roi_ci95'][1]*100:.0f}] "
                  f"ex10={s['roi_ex_top10']*100:.1f}% | fit {v['fit_half'].get('roi', 0)*100:.1f}% val {v['val_half'].get('roi', 0)*100:.1f}%")

    # ---- 2/3. 馬単・馬連・ワイド ----
    races, skipped = load_combo_races(root, info["dates_used"])
    cb = combo_bets(races)
    out["combo_coverage"] = {"races": len(races), "skipped": skipped,
                             "eras": pd.Series([r["era"] for r in races]).value_counts().to_dict(),
                             "wide_races_with_payout": int(sum(1 for r in races if (r["res"].get("pays") or {}).get("wide")))}
    combos = {}
    for kind, bets in cb.items():
        k = {"all_combos": {"all": describe(bets), **by_half(bets)}}
        for t in RATIO_THRESHOLDS:
            sel = [b for b in bets if b["ratio"] >= t]
            k[f"ratio>={t}"] = {"all": describe(sel), **by_half(sel)}
        k["by_odds_band"] = band_table(bets)
        combos[kind] = k
    out["combos"] = combos
    print("combo coverage", json.dumps(out["combo_coverage"], ensure_ascii=False))
    for kind, k in combos.items():
        for name, v in k.items():
            if name == "by_odds_band":
                for r in v:
                    f = lambda x: "—" if not x.get("n_bets") else f"{x['n_bets']}:{x['roi']*100:.0f}%"
                    print(f"   {kind} band {r['band']:10s} all {f(r['all'])}  " + "  ".join(f"r{t} {f(r[f'ratio>={t}'])}" for t in RATIO_THRESHOLDS))
                continue
            s = v["all"]
            print(f"{kind:7s} {name:12s} bets={s['n_bets']:7d} R={s['n_races']:4d} hit={s['hit_rate']*100:5.2f}% "
                  f"roi={s['roi']*100:6.1f}% CI[{s['roi_ci95'][0]*100:.0f},{s['roi_ci95'][1]*100:.0f}] "
                  f"ex10={s['roi_ex_top10']*100:.1f}% medodds={s.get('median_implied_odds', 0):.0f} "
                  f"| fit {v['fit_half(〜08-15)']['roi']*100:.1f}% val {v['val_half(08-16〜)']['roi']*100:.1f}%")
    Path(a.out_json).write_text(json.dumps(out, ensure_ascii=False, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
