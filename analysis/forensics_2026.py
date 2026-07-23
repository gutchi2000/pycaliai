# -*- coding: utf-8 -*-
"""
analysis/forensics_2026.py — 2026実運用 確定395R のガチ検死
============================================================
当たってるレース/当たってないレースの傾向を機構まで分解する。

分解軸:
  1. 芝ダROI差(+37pt)の機構: ◎飛び率か / 的中配当の薄さか / 配分過剰か
  2. ◎着順(1着/2-3着/4着以下)別の回収 — 2026版 ◎飛び検死
  3. 券種×サーフェスのROI行列
  4. チョーク度: 勝ち馬単勝配当の分布差 (芝 vs ダ)
  5. トリガミ率(的中したのに赤字)
  6. 交絡統制: surf効果が chaos/nature/距離/点数 を統制して生き残るか (OLS+bootstrap)

実行: python -m analysis.forensics_2026
"""
from __future__ import annotations
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(42)


def load_races():
    r = json.load(open(BASE / "data/cowork_results.json", encoding="utf-8"))
    rows = []
    for v in (r["races"].values() if isinstance(r["races"], dict) else r["races"]):
        if v.get("決着") != "確定" or not v.get("総投資"):
            continue
        m = re.search(r"独走([\d.]+)/集中([\d.]+)/混戦([\d.]+)/市場([+-]?[\d.]+)", v.get("race_reason", ""))
        sm = re.search(r"(芝|ダート)(\d+)", v.get("race_label", ""))
        rows.append({
            "rid16": re.sub(r"\D", "", str(v["race_id"]))[:16],
            "date": v["date"], "place": v.get("場所", ""),
            "nature": v.get("race_nature", ""), "inv": v["総投資"], "ret": v.get("総払戻", 0),
            "pts": v.get("点数", 0), "hitpts": v.get("的中点数", 0),
            "chaos": float(m.group(3)) if m else np.nan,
            "dom": float(m.group(1)) if m else np.nan,
            "surf": sm.group(1) if sm else "?",
            "dist": int(sm.group(2)) if sm else 0,
        })
    bets = pd.DataFrame(r["bets"])
    bets["rid16"] = bets["race_id"].astype(str).str.replace(r"\D", "", regex=True).str[:16]
    return pd.DataFrame(rows), bets


def join_bundle_kekka(d):
    """◎umaban/p_win (bundle) と ◎着順・勝ち馬配当 (kekka) を結合。"""
    hon_map, kcache = {}, {}
    for date in d["date"].unique():
        bp = BASE / "reports" / "cowork_input" / f"{date}_bundle.json"
        if not bp.exists():
            continue
        b = json.loads(bp.read_text(encoding="utf-8"))
        for r in (b["races"] if isinstance(b["races"], list) else list(b["races"].values())):
            rid = re.sub(r"\D", "", str(r.get("race_id")))[:16]
            hon = next((h for h in r.get("horses", []) if h.get("mark") == "◎"), None)
            if hon:
                hon_map[rid] = (int(hon["umaban"]), float(hon.get("p_win") or 0))
    fin, wpay = {}, {}
    for date in d["date"].unique():
        kp = BASE / "data" / "kekka" / f"{date}.csv"
        if not kp.exists():
            continue
        k = pd.read_csv(kp, encoding="cp932", low_memory=False)
        k["rid16"] = k["レースID(新)"].astype(str).str[:16]
        k["ban"] = pd.to_numeric(k["馬番"], errors="coerce")
        k["f"] = pd.to_numeric(k["確定着順"], errors="coerce")
        k["tan"] = pd.to_numeric(k["単勝配当"], errors="coerce")
        for r in k.itertuples():
            if pd.notna(r.ban):
                fin[(r.rid16, int(r.ban))] = r.f
            if r.f == 1 and pd.notna(r.tan):
                wpay[r.rid16] = r.tan / 100
    d["hon_ban"] = d["rid16"].map(lambda x: hon_map.get(x, (None, None))[0])
    d["hon_pwin"] = d["rid16"].map(lambda x: hon_map.get(x, (None, None))[1])
    d["hon_fin"] = d.apply(lambda r: fin.get((r["rid16"], r["hon_ban"])) if r["hon_ban"] else np.nan, axis=1)
    d["win_pay"] = d["rid16"].map(wpay)
    return d


def roi_ci(g, n_boot=2000):
    inv = g["inv"].values.astype(float)
    ret = g["ret"].values.astype(float)
    n = len(g)
    if n < 5:
        return None
    b = [ret[i].sum() / inv[i].sum() for i in rng.integers(0, n, (n_boot, n))]
    return [round(float(np.percentile(b, 2.5)) * 100, 1), round(float(np.percentile(b, 97.5)) * 100, 1)]


def main():
    d, bets = load_races()
    d = join_bundle_kekka(d)
    d["roi"] = d["ret"] / d["inv"]
    d["hit"] = d["ret"] > 0
    print(f"確定 {len(d)}R  ◎結合={d['hon_fin'].notna().mean()*100:.0f}%  勝ち配当結合={d['win_pay'].notna().mean()*100:.0f}%")

    print("\n===== 1. ◎着順の分解 (2026版 ◎飛び検死) =====")
    d["hon_res"] = pd.cut(d["hon_fin"], [0, 1, 3, 99], labels=["1着", "2-3着", "4着以下(飛び)"])
    for s in ["芝", "ダート"]:
        g = d[d["surf"] == s]
        dist = g["hon_res"].value_counts(normalize=True).sort_index()
        print(f"  {s}: " + "  ".join(f"{k}={v*100:.1f}%" for k, v in dist.items()))
    print("  ◎結果別 ROI (全体):")
    for k, g in d.groupby("hon_res", observed=True):
        print(f"    ◎{k}: n={len(g)} ROI={g['ret'].sum()/g['inv'].sum()*100:.1f}% CI={roi_ci(g)}")
    print("  ◎結果別 ROI (サーフェス別):")
    for s in ["芝", "ダート"]:
        for k, g in d[d["surf"] == s].groupby("hon_res", observed=True):
            if len(g) >= 10:
                print(f"    {s}×◎{k}: n={len(g)} ROI={g['ret'].sum()/g['inv'].sum()*100:.1f}%")

    print("\n===== 2. チョーク度 (勝ち馬単勝配当の分布) =====")
    for s in ["芝", "ダート"]:
        g = d[(d["surf"] == s) & d["win_pay"].notna()]
        print(f"  {s}: 勝ち馬単勝 中央{g['win_pay'].median():.1f}倍 平均{g['win_pay'].mean():.1f}倍 "
              f"1番人気圏(<3倍)率={(g['win_pay']<3).mean()*100:.0f}%  荒れ(>10倍)率={(g['win_pay']>10).mean()*100:.0f}%")

    print("\n===== 3. 券種×サーフェス ROI行列 (bet-level) =====")
    bets2 = bets.merge(d[["rid16", "surf"]], on="rid16", how="inner")
    bets2["決着OK"] = bets2["決着"] == "確定"
    bets2 = bets2[bets2["決着OK"]]
    piv = bets2.groupby(["馬券種", "surf"]).apply(
        lambda g: pd.Series({"n": len(g), "ROI%": g["払戻"].sum() / g["購入額"].sum() * 100 if g["購入額"].sum() else 0}),
        include_groups=False).round(1)
    print(piv.unstack().to_string())

    print("\n===== 4. 的中の質 (的中レースのみ) =====")
    for s in ["芝", "ダート"]:
        g = d[(d["surf"] == s) & d["hit"]]
        mult = g["ret"] / g["inv"]
        print(f"  {s}: 的中{len(g)}R  的中時回収倍率 中央{mult.median():.2f} 平均{mult.mean():.2f} "
              f"トリガミ率(的中でも赤字)={(g['ret']<g['inv']).mean()*100:.0f}%")

    print("\n===== 5. 配分 (サーフェス別の買い方) =====")
    for s in ["芝", "ダート"]:
        g = d[d["surf"] == s]
        print(f"  {s}: 平均点数{g['pts'].mean():.1f} 平均投資¥{g['inv'].mean():,.0f} "
              f"1点あたり¥{(g['inv']/g['pts'].clip(lower=1)).mean():,.0f}")

    print("\n===== 6. 交絡統制 (OLS: roi ~ surf + chaos + dom + log_dist + pts + nature) =====")
    dd = d.dropna(subset=["chaos", "dom"]).copy()
    dd["is_dirt"] = (dd["surf"] == "ダート").astype(float)
    dd["log_dist"] = np.log(dd["dist"].clip(lower=1000))
    X = pd.get_dummies(dd["nature"], prefix="nt", drop_first=True).astype(float)
    X["is_dirt"] = dd["is_dirt"]; X["chaos"] = dd["chaos"]; X["dom"] = dd["dom"]
    X["log_dist"] = dd["log_dist"]; X["pts"] = dd["pts"].astype(float)
    X["const"] = 1.0
    y = dd["roi"].values
    Xv = X.values
    coef = np.linalg.lstsq(Xv, y, rcond=None)[0]
    ci = X.columns.get_loc("is_dirt")
    boots = []
    n = len(dd)
    for _ in range(2000):
        idx = rng.integers(0, n, n)
        try:
            boots.append(np.linalg.lstsq(Xv[idx], y[idx], rcond=None)[0][ci])
        except Exception:
            pass
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"  is_dirt 係数 (交絡統制後): {coef[ci]*100:+.1f}pt  CI95[{lo*100:+.1f}, {hi*100:+.1f}]"
          f"  {'★ゼロ除外' if hi < 0 or lo > 0 else '(ゼロ含む=未証明)'}")

    print("\n===== 7. 多重性の正直な申告 =====")
    print("  検定セル数 ~50 (前回断面) → 期待偽陽性 ~2-3。芝ダは事前自然二分で多重性軽いが、")
    print("  ◎飛び/チョーク/配分の機構分解が一致した方向を指すかで判断すること。")


if __name__ == "__main__":
    main()
