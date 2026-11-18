"""購入時点 (T-10) の市場に対して、モデルを混ぜると当たりが良くなるか (分析専用、本番経路には不干渉)。

事前に固定した設計 (検証側の結果を見てから変えない):
  - データ: analysis.ev_threshold_backtest.load_rows (凍結 bundle × kekka)。採点対象は前回
      (market_blend_check) と同じ馬 (最終単勝オッズあり・中止等でない・p_win あり)。
  - T-10 の単勝オッズ (レースごとに次の順で 1 つ):
      1. data/forward_prices/{date}/{rid}_t10_*.json.gz (追記専用の保存)。発走予定時刻の 5〜15 分前に
         観測された ok なスナップショットのうち最も遅いもの。
      2. 無ければ reports/live_odds/{rid}.json (T-10 実行時の最新版)。週次 CSV の発走時刻の
         5〜15 分前に取得されたものだけ (08-29 以降は発走後に上書きされているので自然に外れる)。
      採点対象の馬のうち 1 頭でも T-10 の単勝オッズが無いレースは丸ごと外す。
  - 分割: 前回と同じ境界。fit = 〜2026-08-15、検証 = 2026-08-16〜。検証は 1 回だけ。
  - 市場 q: T-10 単勝オッズの逆数をレース内 (採点対象の馬) で正規化。
  - モデル m: bundle の p_win を同じ馬の集合でレース内正規化。
  - ブレンド: 幾何 p ∝ q^(1−w)·m^w、w = 0.05 / 0.10 / 0.15 / 0.25 (11/14 の自由 fit で最適が約 0.06
      だったことによる格子)。fit 側の LogLoss 最小の 1 個を選ぶ。
  - 検証: 選んだブレンド・T-10 市場のみ・モデルのみを LogLoss (主) / Brier / 勝ち馬対数尤度で比較。
      参考として最終オッズの市場も並べる (判断には使わない)。CI はレース単位の対応付きブートストラップ。
  - EV ルールの条件: 検証でブレンドの LogLoss の点推定が T-10 市場より小さいときだけ。
      EV = ブレンド確率 × T-10 単勝オッズ、閾値 1.0 / 1.1 / 1.15、単勝の全該当と上位 1 頭、検証側のみ。
      払戻は確定配当 (実際に買えば受け取る額)。

使い方:
  python -m analysis.t10_market_blend_check --data-root E:/PyCaLiAI --out-json <path>
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import ev_threshold_backtest as bt
from analysis.ev_recalibration_check import metrics, paired_metric_diff
from analysis.market_blend_check import blend, paired_winner_nll

WEIGHTS = [0.05, 0.10, 0.15, 0.25]
EV_THRESHOLDS = [1.0, 1.1, 1.15]
SPLIT = "20260815"
LEAD_MIN, LEAD_MAX = 5.0, 15.0


def _ts(x) -> datetime:
    return datetime.fromisoformat(str(x)[:19])


def post_times(root: Path, dates: list[str]) -> dict[str, datetime]:
    out = {}
    for d in dates:
        p = root / "data" / "weekly" / f"{d}.csv"
        if not p.exists():
            continue
        rd = csv.reader(io.StringIO(p.read_bytes().decode("cp932", errors="replace")))
        head = next(rd)
        i_id, i_t = head.index("レースID(新)"), head.index("発走時刻")
        for r in rd:
            if len(r) > i_t and r[i_id] and r[i_t].strip():
                t = r[i_t].strip().replace(":", "").zfill(4)
                try:
                    out[r[i_id][:16]] = datetime.strptime(d + t, "%Y%m%d%H%M")
                except ValueError:      # 週次 CSV の形式違い (列ずれ) の行は使わない
                    continue
    return out


def t10_odds(root: Path, rids: list[str], posts: dict[str, datetime]) -> tuple[dict, dict, Counter]:
    """rid → {馬番: T-10 単勝オッズ}, rid → {source, lead_min}, 集計。"""
    odds, meta, stat = {}, {}, Counter()
    fp_root = root / "data" / "forward_prices"
    for rid in rids:
        best = None
        for f in (fp_root / rid[:8]).glob(f"{rid}_t10_*.json.gz") if (fp_root / rid[:8]).exists() else []:
            d = json.load(gzip.open(f, "rt", encoding="utf-8"))
            mk = d.get("market") or {}
            if not d.get("scheduled_post") or not mk.get("ok") or not mk.get("tansho"):
                continue
            lead = (_ts(d["scheduled_post"]) - _ts(d["observed_at"])).total_seconds() / 60
            if LEAD_MIN <= lead <= LEAD_MAX and (best is None or lead < best[1]):
                best = (mk["tansho"], lead, "forward_prices")
        if best is None:
            f = root / "reports" / "live_odds" / f"{rid}.json"
            if f.exists() and rid in posts:
                j = json.loads(f.read_text(encoding="utf-8"))
                if j.get("ok") and j.get("tansho"):
                    lead = (posts[rid] - _ts(j["fetched"])).total_seconds() / 60
                    if LEAD_MIN <= lead <= LEAD_MAX:
                        best = (j["tansho"], lead, "live_odds")
        if best is None:
            stat["no_t10"] += 1
            continue
        odds[rid] = {int(k): float(v) for k, v in best[0].items() if v and float(v) > 0}
        meta[rid] = {"source": best[2], "lead_min": best[1]}
        stat[best[2]] += 1
    return odds, meta, stat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="E:/PyCaLiAI")
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    root = Path(a.data_root)
    rows, info = bt.load_rows(root)
    df = pd.DataFrame(rows)
    df = df[df["o_fin"].notna() & ~df["no_finish"] & df["p_win"].notna()].copy()
    races_all = sorted(df["rid"].unique())
    posts = post_times(root, sorted(df["date"].unique()))
    odds, src, stat = t10_odds(root, races_all, posts)

    df["o10"] = [odds.get(r, {}).get(u) for r, u in zip(df["rid"], df["uma"])]
    miss = df.groupby("rid")["o10"].apply(lambda s: s.isna().any())
    drop_missing_horse = [r for r in odds if miss.get(r, True)]
    keep = [r for r in odds if not miss.get(r, True)]
    stat["dropped_missing_horse_odds"] = len(drop_missing_horse)
    df = df[df["rid"].isin(keep)].copy()
    df["o10"] = df["o10"].astype(float)
    df["win"] = (df["tan_ret"] > 0).astype(int)
    inv = 1.0 / df["o10"]
    df["q"] = inv / inv.groupby(df["rid"]).transform("sum")
    invf = 1.0 / df["o_fin"]
    df["q_fin"] = invf / invf.groupby(df["rid"]).transform("sum")
    df["m"] = df["p_win"] / df.groupby("rid")["p_win"].transform("sum")

    fit, val = df[df["date"] <= SPLIT].copy(), df[df["date"] > SPLIT].copy()
    cov = {}
    for name, part in (("fit", fit), ("val", val)):
        rr = part.drop_duplicates("rid")
        cov[name] = {"dates": [rr["date"].min(), rr["date"].max(), int(rr["date"].nunique())],
                     "races": int(len(rr)), "horses": int(len(part)),
                     "eras": rr["era"].value_counts().to_dict(),
                     "sources": Counter(src[r]["source"] for r in rr["rid"]),
                     "lead_min_median": float(np.median([src[r]["lead_min"] for r in rr["rid"]]))}
    all_dates = sorted(set(d for d in pd.DataFrame(rows)["date"]))
    base_races = pd.DataFrame(rows).drop_duplicates("rid")
    cov["frozen_races_by_half"] = {"fit": int((base_races["date"] <= SPLIT).sum()),
                                   "val": int((base_races["date"] > SPLIT).sum())}
    cov["race_source_stat"] = dict(stat)
    print(json.dumps(cov, ensure_ascii=False, default=dict))

    fit_scores = []
    for w in WEIGHTS:
        col = f"b_{w}"
        fit[col] = blend(fit, "geo", w)
        mt = metrics(fit, col, "win")
        fit_scores.append({"w": w, "logloss": mt["logloss"], "brier": mt["brier"]})
    fit_ref = {c: {x: metrics(fit, c, "win")[x] for x in ("logloss", "brier", "ece")} for c in ("q", "m", "q_fin")}
    bw = min(fit_scores, key=lambda s: s["logloss"])["w"]
    print("fit:", json.dumps(fit_scores), "chosen w:", bw, "ref:", json.dumps(fit_ref))

    # ---- 検証 (1 回だけ) ----
    val["blend"] = blend(val, "geo", bw)
    vm = {c: metrics(val, c, "win") for c in ("blend", "q", "m", "q_fin")}
    vdiff = {f"{c}-q": paired_metric_diff(val, c, "q", "win") for c in ("blend", "m", "q_fin")}
    vwin = {f"{c}-q": paired_winner_nll(val, c, "q") for c in ("blend", "m", "q_fin")}
    beats = vm["blend"]["logloss"] < vm["q"]["logloss"]

    ev = None
    if beats:
        recs = val.to_dict("records")
        for r in recs:
            r["p_win"], r["o_exp"] = r["blend"], r["o10"]
        ev = {f"{k}@{t}": bt.summarize(bt.select(recs, f"tan_exp_{k}", t))
              for k in ("all", "top1") for t in EV_THRESHOLDS}
    baseline = bt.summarize([{**r, "ret": r["tan_ret"]} for r in val.to_dict("records")])

    out = {
        "design": {"split": SPLIT, "weights": WEIGHTS, "lead_window_min": [LEAD_MIN, LEAD_MAX],
                   "ev_gate": "val logloss(blend) < logloss(T-10 market), point estimate",
                   "ev_thresholds": EV_THRESHOLDS},
        "coverage": cov, "fit_scores": fit_scores, "fit_ref": fit_ref, "chosen_w": bw,
        "val_metrics": {c: {k: v for k, v in m.items()} for c, m in vm.items()},
        "val_diff": vdiff, "val_winner_nll": vwin, "blend_beats_t10_market": bool(beats),
        "val_ev": ev, "val_baseline_all_horses": baseline,
    }
    Path(a.out_json).write_text(json.dumps(out, ensure_ascii=False, default=dict), encoding="utf-8")
    for c in ("blend", "q", "m", "q_fin"):
        print("val", c, {x: round(vm[c][x], 5) for x in ("logloss", "brier", "ece")})
    print(json.dumps(vdiff))
    print(json.dumps(vwin))
    print("beats:", beats)
    if ev:
        for k, s in ev.items():
            print(k, s.get("n_bets"), s.get("n_races"), round(s.get("hit_rate", 0) * 100, 1),
                  round(s.get("roi", 0) * 100, 1), [round(x * 100) for x in s.get("roi_ci95", [0, 0])],
                  round((s.get("roi_ex_top10") or 0) * 100, 1))
    print("baseline", round(baseline["roi"] * 100, 1))


if __name__ == "__main__":
    main()
