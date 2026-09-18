# -*- coding: utf-8 -*-
"""
serve_gap_diagnosis.py — offline と本番の ◎勝率差 (約6pt) を要因に分解する
===========================================================================
本番処理は変更しない。読むだけ + 一時的な別名での再生成のみ。

物差し: 「◎の勝率 − 同じレースの確定1番人気の勝率」(AI−市場の差)。
        期間・対象レースで難しさが変わっても、市場も同じだけ難しくなるので相殺される。
◎の定義: bundle は ai_rank==1 (生スコア最大)。offline は v6 スコア最大。
        ※本番 bundle の p_win (isotonic 較正後) は上位が同値に潰れることがある (17.9%) ので使わない。

段階 (2026 は同一レース集合で比較):
  S  : 実際に配信された印 (その時点の本番モデル・コード。9/11 まで旧モデル)
  P  : 旧モデル (models/production_backup/20260911_pre_c1) × 今のコード で再生成
  R  : C1 モデル × 今のコード で再生成 (reports/_fixed_bundles, reports/_proxy_live/off)
  O  : offline 2024-25 (C1 本番モデル・学習時と同じ特徴)
  Ow : O を 2026 の対象レース構成 (クラス×頭数帯) に重み付け

分解:
  S→P : serve 時の特徴生成・入力データ (コード版) の違い
  P→R : 学習版・モデル版の違い
  R vs Ow : 期間の違い + (2026 の serve 特徴と offline 特徴の違い)。この2つは同一レースで
            揃えられないので分離できない (2026 には offline 特徴を作れる master が無い)
  O vs Ow : 対象レース集合の違い
  R の 4-5月 vs 6-9月 : 入力特徴の欠損 (TARGET 補正タイムが 5/31 で途切れる) の自然実験

実行:
  python -m analysis.mcond.serve_gap_diagnosis --regen-prec1   # P を作る (約20分)
  python -m analysis.mcond.serve_gap_diagnosis                 # 集計のみ
出力: analysis/mcond/serve_gap_out/serve_gap.json, serve_gap_steps.csv
"""
from __future__ import annotations
import argparse
import glob
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
OUT = Path(__file__).resolve().parent / "serve_gap_out"
PY = BASE / "venv311/Scripts/python.exe"
BK = BASE / "models/production_backup/20260911_pre_c1"
TAG = "v6prec1diag"
PREC1_DIR = BASE / "reports/_diag_prec1"
CLASS_RANK = {"新馬": "新馬", "未勝利": "未勝利", "1勝": "1勝", "500万": "1勝", "2勝": "2勝",
              "1000万": "2勝", "3勝": "3勝", "1600万": "3勝"}


def cls_bin(c):
    c = str(c)
    for k, v in CLASS_RANK.items():
        if k in c:
            return v
    return "OP以上"


def fs_bin(n):
    return "≤10" if n <= 10 else ("11-14" if n <= 14 else "15+")


def dates_2026():
    ds = []
    for p in sorted(glob.glob(str(BASE / "reports/cowork_input/2026*_bundle.json"))):
        d = Path(p).name[:8]
        if (BASE / f"data/kekka/{d}.csv").exists() and (BASE / f"data/weekly/{d}.csv").exists():
            ds.append(d)
    return ds


def regen_prec1(ds):
    """旧モデル一式を一時的な別名で置いて今のコードで再生成。終わったら必ず消す。"""
    tmp = [(BK / "models/unified_rank_v6.pkl", BASE / f"models/unified_rank_{TAG}.pkl"),
           (BK / "models/pl_calibrators_v6.pkl", BASE / f"models/pl_calibrators_{TAG}.pkl"),
           (BK / "models/pl_calibrators_v6_serve.pkl", BASE / f"models/pl_calibrators_{TAG}_serve.pkl")]
    for src, dst in tmp:
        assert not dst.exists(), f"{dst} が既に存在する (上書きしない)"
    try:
        for src, dst in tmp:
            shutil.copy2(src, dst)
        PREC1_DIR.mkdir(parents=True, exist_ok=True)
        for d in ds:
            if (PREC1_DIR / f"{d}_bundle.json").exists():
                continue
            subprocess.run([str(PY), "make_weekly_hosei.py", "--csv", f"data/weekly/{d}.csv"],
                           cwd=BASE, capture_output=True)
            subprocess.run([str(PY), "export_weekly_marks.py", "--csv", f"data/weekly/{d}.csv",
                            "--model", TAG, "--out-dir", str(PREC1_DIR / "races" / d)],
                           cwd=BASE, capture_output=True)
            agg = PREC1_DIR / "races" / f"{d}_bundle.json"
            if agg.exists():
                agg.replace(PREC1_DIR / f"{d}_bundle.json")
            print(f"  {d}: {'ok' if (PREC1_DIR / f'{d}_bundle.json').exists() else 'NG'}", flush=True)
    finally:
        for _, dst in tmp:
            if dst.exists():
                dst.unlink()


def kekka_2026(d):
    k = pd.read_csv(BASE / f"data/kekka/{d}.csv", encoding="cp932", low_memory=False)
    k["rid16"] = k["レースID(新)"].astype(str).str[:16]
    k["ban"] = pd.to_numeric(k["馬番"], errors="coerce")
    k["fin"] = pd.to_numeric(k["確定着順"], errors="coerce")
    s = k["単勝配当"].astype(str)
    k["fodds"] = pd.to_numeric(s.str.extract(r"\(([\d.]+)\)")[0], errors="coerce")
    k.loc[k["fin"] == 1, "fodds"] = pd.to_numeric(s, errors="coerce") / 100.0
    k = k[k["fin"].notna() & (k["fin"] >= 1) & k["ban"].notna()]
    out = {}
    for rid, g in k.groupby("rid16"):
        if g["fodds"].isna().any() or (g["fin"] == 1).sum() != 1:
            continue
        w = int(g.loc[g["fin"] == 1, "ban"].iloc[0])
        fav = int(g.loc[g["fodds"].idxmin(), "ban"])
        out[rid] = (w, fav, len(g))
    return out


def score_bundle(path, res, label):
    b = json.loads(Path(path).read_text(encoding="utf-8"))
    races = b["races"] if isinstance(b["races"], list) else list(b["races"].values())
    rows = []
    for r in races:
        rid = "".join(c for c in str(r.get("race_id", "")) if c.isdigit())[:16]
        if rid not in res:
            continue
        hs = [h for h in r.get("horses", []) if h.get("ai_rank") is not None and h.get("umaban") is not None]
        if len(hs) < 5:
            continue
        hon = int(min(hs, key=lambda h: float(h["ai_rank"]))["umaban"])
        pw = [h for h in hs if h.get("p_win") is not None]
        tie = False
        if len(pw) >= 2:
            ps = sorted((float(h["p_win"]) for h in pw), reverse=True)
            tie = abs(ps[0] - ps[1]) < 1e-12
        w, fav, n = res[rid]
        rm = r.get("race_meta", {}) or {}
        rows.append(dict(src=label, rid=rid, month=rid[4:6], ai_win=int(hon == w), fav_win=int(fav == w),
                         agree=int(hon == fav), tie=int(tie), cls=cls_bin(rm.get("class", "")),
                         fsb=fs_bin(n)))
    return rows


def gap_stats(d, reps=2000, seed=0, weights=None):
    x = (d["ai_win"] - d["fav_win"]).to_numpy(float)
    w = np.ones(len(x)) if weights is None else np.asarray(weights, float)
    day = d["rid"].str[:8].to_numpy()
    g = pd.DataFrame({"xw": x * w, "w": w, "day": day}).groupby("day")[["xw", "w"]].sum()
    a, b = g["xw"].to_numpy(), g["w"].to_numpy()
    rng = np.random.default_rng(seed)
    bs = np.array([a[i].sum() / b[i].sum() for i in (rng.integers(0, len(a), len(a)) for _ in range(reps))])
    return {"n": int(len(d)), "ai_win": float(np.average(d["ai_win"], weights=w)),
            "fav_win": float(np.average(d["fav_win"], weights=w)),
            "gap_pt": float(100 * a.sum() / b.sum()),
            "ci95": [float(100 * np.quantile(bs, .025)), float(100 * np.quantile(bs, .975))],
            "agree": float(np.average(d["agree"], weights=w))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regen-prec1", action="store_true")
    args = ap.parse_args()
    ds = dates_2026()
    if args.regen_prec1:
        print(f"旧モデル × 今のコード で再生成: {len(ds)} 開催日")
        regen_prec1(ds)

    rows = []
    for d in ds:
        res = kekka_2026(d)
        rows += score_bundle(BASE / f"reports/cowork_input/{d}_bundle.json", res, "S")
        rp = (BASE / f"reports/_fixed_bundles/{d}_bundle.json") if d <= "20260531" \
            else (BASE / f"reports/_proxy_live/off/{d}_bundle.json")
        if rp.exists():
            rows += score_bundle(rp, res, "R")
        pp = PREC1_DIR / f"{d}_bundle.json"
        if pp.exists():
            rows += score_bundle(pp, res, "P")
    d26 = pd.DataFrame(rows)
    common = set.intersection(*[set(d26[d26.src == s].rid) for s in d26.src.unique()])
    d26c = d26[d26.rid.isin(common)]

    # offline (C1 本番モデル, 2024-25)
    b = pd.read_parquet(BASE / "data/_research/mcond/base.parquet")
    b = b[b.year.isin([2024, 2025])]
    m = pd.read_csv(BASE / "data/master_v2_20130105-20251228.csv", encoding="utf-8-sig",
                    usecols=["レースID(新)", "クラス名"], low_memory=False)
    m["rid16"] = m["レースID(新)"].astype(str).str[:16]
    cm = m.drop_duplicates("rid16").set_index("rid16")["クラス名"]
    offl = []
    for rid, g in b.groupby("rid16"):
        if g["tan_final_odds"].isna().any() or g["win"].sum() != 1:
            continue
        hon = g.loc[g["v6_score"].idxmax()]
        fav = g.loc[g["tan_final_odds"].idxmin()]
        offl.append(dict(src="O", rid=rid, month=rid[4:6], ai_win=int(hon["win"]), fav_win=int(fav["win"]),
                         agree=int(hon["ban"] == fav["ban"]), tie=0, cls=cls_bin(cm.get(rid, "")),
                         fsb=fs_bin(len(g))))
    off = pd.DataFrame(offl)

    # 2026 の対象レース構成に重み付け
    tgt = d26c[d26c.src == "S"].groupby(["cls", "fsb"]).size()
    tgt = tgt / tgt.sum()
    src = off.groupby(["cls", "fsb"]).size()
    src = src / src.sum()
    wmap = (tgt / src).fillna(0.0)
    ow = off.apply(lambda r: wmap.get((r["cls"], r["fsb"]), 0.0), axis=1).to_numpy()

    steps = {}
    steps["O  offline 2024-25 (C1)"] = gap_stats(off)
    steps["Ow offline を2026構成に重み付け"] = gap_stats(off, weights=ow)
    for s, lab in [("S", "S  実際に配信された印"), ("P", "P  旧モデル×今のコード"), ("R", "R  C1×今のコード")]:
        x = d26c[d26c.src == s]
        if len(x):
            steps[lab] = gap_stats(x)
    for s in ["R", "S"]:
        x = d26c[d26c.src == s]
        if len(x):
            steps[f"{s} 4-5月 (補正タイムあり)"] = gap_stats(x[x.month.isin(["04", "05"])])
            steps[f"{s} 6-9月 (補正タイム途切れ)"] = gap_stats(x[~x.month.isin(["04", "05"])])
    s = d26c[d26c.src == "S"]
    steps["S p_win同値レースの割合"] = {"tie_rate": float(s["tie"].mean()), "n": int(len(s))}
    for y in [2024, 2025]:
        steps[f"O {y}"] = gap_stats(off[off.rid.str[:4] == str(y)])

    OUT.mkdir(exist_ok=True)
    (OUT / "serve_gap.json").write_text(json.dumps(
        {"common_races_2026": len(common), "dates": ds, "steps": steps}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"2026 共通レース {len(common)} R ({len(ds)} 開催日)\n")
    print(f"{'段階':<34}{'n':>6}{'◎勝率':>8}{'1人気勝率':>10}{'差(pt)':>9}  95%CI           一致率")
    flat = []
    for k, v in steps.items():
        if "gap_pt" in v:
            print(f"{k:<34}{v['n']:>6}{100*v['ai_win']:>8.2f}{100*v['fav_win']:>10.2f}"
                  f"{v['gap_pt']:>+9.2f}  [{v['ci95'][0]:+.1f},{v['ci95'][1]:+.1f}]  {100*v['agree']:.1f}%")
            flat.append({"step": k, **{kk: vv for kk, vv in v.items() if kk != "ci95"},
                         "ci_lo": v["ci95"][0], "ci_hi": v["ci95"][1]})
        else:
            print(f"{k:<34} {v}")
    pd.DataFrame(flat).to_csv(OUT / "serve_gap_steps.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
