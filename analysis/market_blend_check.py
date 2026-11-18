"""モデルは市場に対して追加の情報を持っているか (市場 × モデルのブレンド検証、分析専用)。

事前に固定した設計 (検証側の結果を見てから変えない):
  - データ: analysis.ev_threshold_backtest.load_rows と同じ (凍結 bundle × kekka)。
      ev_recalibration_check と同じく、最終単勝オッズの無い馬と着順 0 で最終オッズのある馬
      (中止/発売後除外) は採点から外す。
  - 分割: 開催日を日付順に並べ、前半 28 日 = fit (05-02〜08-15)、後半 28 日 = 検証 (08-16〜11-15)。
  - 市場 q: 最終単勝オッズの逆数をレース内 (採点対象の馬) で正規化。
  - モデル m: bundle の p_win を同じ馬の集合でレース内正規化 (合計 1 にそろえる)。
  - ブレンド候補 (6 個):
      幾何: p ∝ q^(1−w) · m^w をレース内正規化、w = 0.25 / 0.5 / 0.75
      算術: p = (1−w)·q + w·m、w = 0.25 / 0.5 / 0.75
    fit 側の LogLoss (馬ごとの 1 着か / 確率は [0.001, 0.999] に制限) が最小の 1 個を選ぶ。
  - 検証: 選んだブレンド・市場のみ・モデルのみを LogLoss / Brier で比較 (主 = LogLoss)。
      差の CI はレース単位の対応付きブートストラップ (4,000 回)。副指標として勝ち馬の
      対数尤度 (レースごとの −log p(勝ち馬)) も出す。
  - EV ルールを計算する条件: 検証側でブレンドの LogLoss の点推定が市場より小さいこと
      (CI が 0 をまたぐかは別に報告)。満たさなければ EV ルールは計算しない。
      計算する場合は EV = ブレンド確率 × 最終単勝オッズ、閾値 1.0/1.1/1.15/1.2、
      単勝の全該当と上位 1 頭、検証側のみ。
  - 参考 (探索): 検証側 (と fit 側) で、LogLoss の市場との差 (モデル − 市場、ブレンド − 市場) を
      クラス・芝ダ・頭数・CHAOS 帯・モデル世代で分けて出す。採否の判断には使わない。

使い方:
  python -m analysis.market_blend_check --data-root E:/PyCaLiAI --out-json <path>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import ev_threshold_backtest as bt
from analysis.ev_recalibration_check import clip, metrics, paired_metric_diff, N_FIT_DATES

THRESHOLDS = bt.THRESHOLDS
WEIGHTS = [0.25, 0.5, 0.75]
N_BOOT = 4000
RNG = np.random.default_rng(20261119)


def race_meta_map(data_root: Path, dates: list[str]) -> dict[str, dict]:
    out = {}
    for d in dates:
        b = json.loads((data_root / "reports" / "cowork_input_frozen" / f"{d}_bundle.json").read_text(encoding="utf-8"))
        for r in b.get("races", []):
            m = r.get("race_meta") or {}
            out[str(r.get("race_id", ""))[:16]] = {
                "cls": str(m.get("class") or ""), "course": str(m.get("course") or ""),
                "field_size": m.get("field_size")}
    return out


def class_group(c: str) -> str:
    if c.startswith("ＪＧ") or c.startswith("JG"):
        return "障害重賞"
    if c in ("新馬", "未勝利", "1勝", "2勝", "3勝"):
        return c
    return "OP以上"


def field_group(n) -> str:
    if n is None or pd.isna(n):
        return "不明"
    n = int(n)
    return "〜10頭" if n <= 10 else ("11〜14頭" if n <= 14 else "15頭〜")


def chaos_group(c) -> str:
    if c is None or pd.isna(c):
        return "なし(〜05-31)"
    return "0〜1/3" if c < 1 / 3 else ("1/3〜0.667" if c < 0.667 else "0.667〜")


def blend(df: pd.DataFrame, kind: str, w: float) -> np.ndarray:
    q, m = df["q"].to_numpy(), df["m"].to_numpy()
    if kind == "arith":
        return (1 - w) * q + w * m
    lg = (1 - w) * np.log(np.clip(q, 1e-12, None)) + w * np.log(np.clip(m, 1e-12, None))
    x = np.exp(lg)
    return x / pd.Series(x, index=df.index).groupby(df["rid"]).transform("sum").to_numpy()


def ll_vec(p, y):
    p = clip(p)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def winner_nll(df: pd.DataFrame, col: str) -> pd.Series:
    """レースごとの −log p(勝ち馬) (勝ち馬が採点対象にいるレースだけ)。"""
    w = df[df["win"] == 1]
    return pd.Series(-np.log(clip(w[col])), index=w.index).groupby(w["rid"]).sum()


def paired_winner_nll(df: pd.DataFrame, a: str, b: str) -> dict:
    da, db = winner_nll(df, a), winner_nll(df, b)
    d = (da - db).to_numpy()
    idx = RNG.integers(0, len(d), size=(N_BOOT, len(d)))
    boots = d[idx].mean(1)
    return {"n_races": int(len(d)), "a": float(da.mean()), "b": float(db.mean()), "diff": float(d.mean()),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}


def subgroup_table(df: pd.DataFrame, gcol: str, cols: list[str]) -> list[dict]:
    rows = []
    y = df["win"].to_numpy().astype(float)
    llq = ll_vec(df["q"], y)
    for g, sub in df.groupby(gcol, observed=True):
        mask = (df[gcol] == g).to_numpy()
        r = {"group": str(g), "n_races": int(sub["rid"].nunique()), "n_horses": int(len(sub)),
             "ll_market": float(llq[mask].mean())}
        rid = sub["rid"].to_numpy()
        uniq, inv = np.unique(rid, return_inverse=True)
        idx = RNG.integers(0, len(uniq), size=(N_BOOT, len(uniq)))
        cnt = np.bincount(inv)
        for c in cols:
            d = ll_vec(sub[c], sub["win"].to_numpy().astype(float)) - llq[mask]
            s = np.bincount(inv, weights=d)
            boots = s[idx].sum(1) / cnt[idx].sum(1)
            r[f"{c}-q"] = {"diff": float(d.mean()), "ci95": [float(np.percentile(boots, 2.5)),
                                                              float(np.percentile(boots, 97.5))]}
        rows.append(r)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="E:/PyCaLiAI")
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    root = Path(a.data_root)
    rows, info = bt.load_rows(root)
    df = pd.DataFrame(rows)
    n0 = len(df)
    df = df[df["o_fin"].notna() & ~df["no_finish"] & df["p_win"].notna()].copy()
    df["win"] = (df["tan_ret"] > 0).astype(int)
    inv = 1.0 / df["o_fin"]
    df["q"] = inv / inv.groupby(df["rid"]).transform("sum")
    df["m"] = df["p_win"] / df.groupby("rid")["p_win"].transform("sum")

    meta = race_meta_map(root, info["dates_used"])
    df["cls_g"] = df["rid"].map(lambda r: class_group(meta.get(r, {}).get("cls", "")))
    df["surf"] = df["rid"].map(lambda r: {"芝": "芝", "ダ": "ダート", "障": "障害"}.get(meta.get(r, {}).get("course", "")[:1], "不明"))
    df["field_g"] = df["rid"].map(lambda r: field_group(meta.get(r, {}).get("field_size")))
    df["chaos_g"] = df["chaos"].map(chaos_group)

    dates = sorted(df["date"].unique())
    fit_dates, val_dates = dates[:N_FIT_DATES], dates[N_FIT_DATES:]
    fit, val = df[df["date"].isin(fit_dates)].copy(), df[df["date"].isin(val_dates)].copy()

    cands = [(k, w) for k in ("geo", "arith") for w in WEIGHTS]
    fit_scores = []
    for k, w in cands:
        col = f"b_{k}_{w}"
        fit[col] = blend(fit, k, w)
        mt = metrics(fit, col, "win")
        fit_scores.append({"kind": k, "w": w, "logloss": mt["logloss"], "brier": mt["brier"]})
    fit_ref = {c: {x: metrics(fit, c, "win")[x] for x in ("logloss", "brier", "ece")} for c in ("q", "m")}
    best = min(fit_scores, key=lambda s: s["logloss"])
    bk, bw = best["kind"], best["w"]
    print("fit:", json.dumps(fit_scores), "best:", bk, bw, "ref:", json.dumps(fit_ref))

    # ---- 検証 (1 回だけ) ----
    val["blend"] = blend(val, bk, bw)
    fit["blend"] = fit[f"b_{bk}_{bw}"]
    vm = {c: {x: v for x, v in metrics(val, c, "win").items()} for c in ("blend", "q", "m")}
    vdiff = {f"{c}-q": paired_metric_diff(val, c, "q", "win") for c in ("blend", "m")}
    vwin = {f"{c}-q": paired_winner_nll(val, c, "q") for c in ("blend", "m")}
    beats = vm["blend"]["logloss"] < vm["q"]["logloss"]

    ev = None
    if beats:
        recs = val.to_dict("records")
        for r in recs:
            r["p_win"] = r["blend"]
        ev = {f"{k}@{t}": bt.summarize(bt.select(recs, k, t)) for k in ("tan_all", "tan_top1") for t in THRESHOLDS}
    baseline = bt.summarize([{**r, "ret": r["tan_ret"]} for r in val.to_dict("records")])

    explo = {}
    for name, d in (("val", val), ("fit", fit)):
        explo[name] = {g: subgroup_table(d, g, ["m", "blend"]) for g in ("cls_g", "surf", "field_g", "chaos_g", "era")}

    out = {
        "design": {"fit_dates": [fit_dates[0], fit_dates[-1], len(fit_dates)],
                   "val_dates": [val_dates[0], val_dates[-1], len(val_dates)],
                   "candidates": [f"{k}_{w}" for k, w in cands], "ev_gate": "val logloss(blend) < logloss(market), point estimate"},
        "info": {"rows_loaded": n0, "rows_scored": int(len(df)),
                 "fit_races": int(fit["rid"].nunique()), "val_races": int(val["rid"].nunique()),
                 "fit_horses": int(len(fit)), "val_horses": int(len(val)),
                 "fit_eras": fit.drop_duplicates("rid")["era"].value_counts().to_dict(),
                 "val_eras": val.drop_duplicates("rid")["era"].value_counts().to_dict(),
                 "dates_excluded": info["dates_no_frozen"]},
        "fit_scores": fit_scores, "fit_ref": fit_ref, "chosen": {"kind": bk, "w": bw},
        "val_metrics": vm, "val_diff": vdiff, "val_winner_nll": vwin, "blend_beats_market": bool(beats),
        "val_ev": ev, "val_baseline_all_horses": baseline, "exploratory": explo,
    }
    Path(a.out_json).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    for c in ("blend", "q", "m"):
        print("val", c, {x: round(vm[c][x], 5) for x in ("logloss", "brier", "ece")})
    print(json.dumps(vdiff), json.dumps(vwin))
    print("beats:", beats)
    if ev:
        for k, s in ev.items():
            print(k, s.get("n_bets"), s.get("n_races"), round(s.get("hit_rate", 0) * 100, 1),
                  round(s.get("roi", 0) * 100, 1), [round(x * 100) for x in s.get("roi_ci95", [0, 0])])


if __name__ == "__main__":
    main()
