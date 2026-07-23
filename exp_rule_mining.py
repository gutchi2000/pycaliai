# -*- coding: utf-8 -*-
"""
exp_rule_mining.py — 「圧勝→次走」条件ルールの閾値スイープ実測
================================================================
ユーザー仮説: 「新馬戦を0.3秒差で勝ったら次の1勝クラスは(ほぼ)勝つ」型の
決定的条件を、着差閾値を刻んで探す。

規律 (蜃気楼防止):
  - 発見フェーズ = ≤2023 のみでスイープ (winrate / 単勝ROI + CI)
  - 確認フェーズ = 発見期で有望なセルだけ 2024-25 で開封
  - 決済 = kekka 単勝配当 (勝ち馬のみ収録で単勝ROIは計算可能)

実行: python exp_rule_mining.py
出力: reports/rule_mining_margin.json
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).parent
OUT = BASE / "reports" / "rule_mining_margin.json"
RID = "レースID(新/馬番無)"
PRID = "前走レースID(新/馬番無)"

USE = [RID, PRID, "クラス名", "着順", "前走確定着順", "前走着差タイム",
       "前走上り3F順", "前走出走頭数"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def roi_ci(pay, n_boot=3000, seed=42):
    """pay = 1ベットあたり回収 (的中=配当/100, 外れ=0)。"""
    n = len(pay)
    if n < 3:
        return None
    rng = np.random.default_rng(seed)
    b = [pay[rng.integers(0, n, n)].mean() for _ in range(n_boot)]
    return [round(float(np.percentile(b, 2.5)), 3), round(float(np.percentile(b, 97.5)), 3)]


def cell(df, mask):
    g = df[mask]
    n = len(g)
    if n == 0:
        return {"n": 0}
    wins = int((g["着順"] == 1).sum())
    pay = np.where(g["着順"] == 1, g["_pay"], 0.0)
    lo, hi = wilson(wins, n)
    return {"n": n, "wins": wins, "winrate": round(wins / n, 4),
            "win_ci": [round(lo, 3), round(hi, 3)],
            "roi": round(float(pay.mean()), 4), "roi_ci": roi_ci(pay)}


def main():
    print("[load] master (usecols)...")
    df = pd.read_csv(BASE / "data/master_v2_20130105-20251228.csv",
                     encoding="utf-8-sig", usecols=USE, low_memory=False)
    for c in ["着順", "前走確定着順", "前走着差タイム", "前走上り3F順", "前走出走頭数"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # PRID は NaN 混入で float64 化し int64 索引と噛み合わない → 両方文字列キー化
    # ID列は英字混じり(地方/海外)ありの object → 素の文字列キーで結合
    df["_rid"] = df[RID].astype(str).str.strip()
    df["_prid"] = df[PRID].astype(str).str.strip()
    df["year"] = df["_rid"].str[:4].astype(int)
    cls_map = df.drop_duplicates("_rid").set_index("_rid")["クラス名"]
    df["前走クラス"] = df["_prid"].map(cls_map)
    # 決済: kekka 単勝配当
    k = pd.read_csv(BASE / "data/kekka_20130105-20251228.csv", encoding="cp932",
                    usecols=["レースID(新)", "確定着順", "単勝配当"], low_memory=False)
    k = k[pd.to_numeric(k["確定着順"], errors="coerce") == 1].copy()
    k["rid16"] = k["レースID(新)"].astype(str).str[:16]
    k["_p"] = pd.to_numeric(k["単勝配当"], errors="coerce") / 100.0
    pay_map = k.drop_duplicates("rid16").set_index("rid16")["_p"]
    df["_pay"] = df["_rid"].map(pay_map)
    df = df.dropna(subset=["着順"])
    print(f"  rows={len(df):,}  pay_coverage={df['_pay'].notna().mean():.3f}")

    disc = df["year"] <= 2023          # 発見
    conf = df["year"] >= 2024          # 確認 (有望セルのみ開封)
    prev_win = df["前走確定着順"] == 1
    margin = df["前走着差タイム"]

    # ---- スイープ対象の遷移 ----
    TRANS = [
        ("新馬", ["未勝利"]),               # 新馬負け→未勝利 は別枠、まず勝ち上がり遷移:
        ("新馬", ["1勝", "500万"]),
        ("未勝利", ["1勝", "500万"]),
        ("1勝", ["2勝", "1000万"]),
        ("500万", ["2勝", "1000万"]),
        (None, None),                      # 全遷移 (前走圧勝のみ条件)
    ]
    XS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]

    out = {"discovery_le2023": {}, "note": "margin<=-X (負=前に出た秒数), 決済=kekka単勝配当"}
    print("\n===== 発見フェーズ (≤2023) =====")
    for pc, cc in TRANS:
        base = disc & prev_win
        label = f"{pc or 'ANY'}勝ち→{'/'.join(cc) if cc else 'ANY'}"
        if pc:
            base = base & (df["前走クラス"] == pc)
        if cc:
            base = base & (df["クラス名"].isin(cc))
        rows = {}
        for X in XS:
            r = cell(df, base & (margin <= -X + 1e-9))
            rows[f"{X:.1f}"] = r
            if r["n"] > 0:
                print(f"  {label} 着差>={X:.1f}s: n={r['n']:5d} 勝率={r.get('winrate',0)*100:5.1f}% "
                      f"CI[{r['win_ci'][0]*100:.1f},{r['win_ci'][1]*100:.1f}] "
                      f"ROI={r.get('roi',0):.3f} CI={r.get('roi_ci')}")
        out["discovery_le2023"][label] = rows
        print()

    # ---- 追加軸: 圧勝 + 上り最速 (発見のみ) ----
    print("===== 圧勝+上り最速 (≤2023) =====")
    extra = {}
    for X in [0.2, 0.3, 0.5]:
        m = disc & prev_win & (margin <= -X) & (df["前走上り3F順"] == 1)
        r = cell(df, m)
        extra[f"margin{X}_agari1"] = r
        print(f"  前走1着&着差>={X}s&上り最速: n={r['n']} 勝率={r.get('winrate',0)*100:.1f}% ROI={r.get('roi')}")
    out["discovery_extra"] = extra

    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[saved] {OUT}")


def cmd_cross():
    """着差×次走9時人気バンドのクロス。発見=≤2023 / 確認=2024-25 は
    発見ROI CI下限>1.0 のセルのみ開封 (事前登録)。決済=確定オッズ。"""
    from stage1_benter_blend import load_tanpuk_snapshots
    snaps = load_tanpuk_snapshots()
    print(f"  snapshots: {len(snaps):,}")

    use = USE + ["馬番"]
    df = pd.read_csv(BASE / "data/master_v2_20130105-20251228.csv",
                     encoding="utf-8-sig", usecols=use, low_memory=False)
    for c in ["着順", "前走確定着順", "前走着差タイム", "馬番"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["_rid"] = df[RID].astype(str).str.strip()
    df["_prid"] = df[PRID].astype(str).str.strip()
    df["year"] = df["_rid"].str[:4].astype(int)
    cls_map = df.drop_duplicates("_rid").set_index("_rid")["クラス名"]
    df["前走クラス"] = df["_prid"].map(cls_map)

    sub = df[(df["前走確定着順"] == 1) & df["前走着差タイム"].notna() & df["馬番"].notna()].copy()
    print(f"  prev-win rows: {len(sub):,}")
    o9v, ocv, rank9 = [], [], []
    for rid16, ban in zip(sub["_rid"].str[:16], sub["馬番"].astype(int)):
        sp = snaps.get(rid16)
        if sp is None or ban not in sp["o9"] or ban not in sp["oc"]:
            o9v.append(np.nan); ocv.append(np.nan); rank9.append(np.nan)
            continue
        o9v.append(sp["o9"][ban]); ocv.append(sp["oc"][ban])
        rank9.append(1 + sum(1 for v in sp["o9"].values() if v < sp["o9"][ban]))
    sub["o9"], sub["oc"], sub["rank9"] = o9v, ocv, rank9
    sub = sub.dropna(subset=["o9", "oc", "rank9"])
    print(f"  with odds: {len(sub):,}")
    sub["_payoc"] = np.where(sub["着順"] == 1, sub["oc"], 0.0)

    disc = sub["year"] <= 2023
    conf = sub["year"] >= 2024
    margin = sub["前走着差タイム"]
    BANDS = [("人気1", sub["rank9"] == 1), ("人気2-3", sub["rank9"].between(2, 3)),
             ("人気4-6", sub["rank9"].between(4, 6)), ("人気7+", sub["rank9"] >= 7)]
    XS = [0.2, 0.3, 0.5, 0.8]

    def cell2(mask):
        g = sub[mask]
        n = len(g)
        if n == 0:
            return {"n": 0}
        wins = int((g["着順"] == 1).sum())
        pay = g["_payoc"].values
        return {"n": n, "wins": wins, "winrate": round(wins / n, 4),
                "roi": round(float(pay.mean()), 4), "roi_ci": roi_ci(pay)}

    out = {"discovery": {}, "confirm_unveiled": {}}
    qualify = []
    print("\n===== 発見 (≤2023): 圧勝×次走9時人気 =====")
    for X in XS:
        for bl, bm in BANDS:
            r = cell2(disc & (margin <= -X) & bm)
            key = f"着差>={X}s×{bl}"
            out["discovery"][key] = r
            if r["n"] > 0:
                lo = (r.get("roi_ci") or [0])[0]
                star = " ★資格" if (r["n"] >= 100 and lo > 1.0) else ""
                print(f"  {key}: n={r['n']:5d} 勝率={r['winrate']*100:5.1f}% "
                      f"ROI={r['roi']:.3f} CI={r.get('roi_ci')}{star}")
                if r["n"] >= 100 and lo > 1.0:
                    qualify.append((key, X, bl, bm))
    print(f"\n  確認開封資格セル: {len(qualify)}")
    for key, X, bl, bm in qualify:
        r = cell2(conf & (margin <= -X) & bm)
        out["confirm_unveiled"][key] = r
        print(f"  [確認2024-25] {key}: {r}")

    prev = json.load(open(OUT, encoding="utf-8")) if OUT.exists() else {}
    prev["cross_margin_x_rank"] = out
    json.dump(prev, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "cross":
        cmd_cross()
    else:
        main()
