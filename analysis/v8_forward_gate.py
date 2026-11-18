"""v8 の前向き判定 (CW-21 系) の集計: モデル単独と市場の差、自由 fit したブレンドの重み (分析専用、本番経路には不干渉)。

使い方 (v8 のデータが溜まったらそのまま流す):
  python -m analysis.v8_forward_gate                                  # 既定: v8、2026-11-21〜今日
  python -m analysis.v8_forward_gate --eras v8 --since 20261121 --out-json reports/v8_forward_gate/latest.json
  # 動作確認 (v6/v7 の凍結データで):
  python -m analysis.v8_forward_gate --eras v6-C0,v6-C1,v7 --since 20260613 --until 20261115 --force-final

入力:
  reports/cowork_input_frozen/{date}_bundle.json (+ .provenance.json の verified_pre_race) — モデルの勝率 p_win
  reports/zure_freeze/{date}.json があれば、記録された bundle の sha256 と凍結 bundle が一致するかを確認 (不一致は報告、除外はしない)
  data/kekka/{date}.csv — 着順・確定配当・最終単勝オッズ
  data/forward_prices/{date}/*_t10_*.json.gz (無ければ reports/live_odds) — T-10 単勝オッズ (発走 5〜15 分前の値だけ)

事前に決めた判定 (2026-11-19、結果を見る前に固定):
  段階: 採点できたレースが 200R 未満 = 収集中 (数字は出すが読まない)、200R 以上 = 中間確認 (判定しない)、
        1,000R 以上 = 本判定。
  (1) モデル単独: 馬ごとの 2 値 LogLoss と勝ち馬 NLL (nats/R) を、T-10 市場・最終市場と比べる (レース単位
      ブートストラップ 4,000 回の 95%CI)。確率はすべて採点対象の馬でレース内正規化、[0.001, 0.999] に制限。
  (2) ブレンド: log p = a·log p_model + b·log q_T10 (レース内 softmax)。(a, b) は**開催日順で前半の日**の
      勝ち馬 NLL 最小で自由 fit し、**後半の日**で評価する (日数で半分、端数は前半に入れない)。
      全期間で fit した (a, b) も記述として出す (判定には使わない)。
  判定 (本判定のときだけ): 後半の日で「ブレンド − T-10 市場」の勝ち馬 NLL の 95%CI 上限 < 0 なら
      **BLEND** (ブレンド側で買い方を設計する)、それ以外は **MODEL_ONLY** (モデル側の改善のみ)。
      最終市場との差は診断 (購入時点に無いので判定に使わない)。
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from analysis import ev_threshold_backtest as bt
from analysis.t10_market_blend_check import post_times, t10_odds

ROOT = Path("E:/PyCaLiAI")
INTERIM_RACES, FINAL_RACES = 200, 1000
N_BOOT = 4000
P_LO, P_HI = 0.001, 0.999


def load(eras: set[str], since: str, until: str) -> tuple[pd.DataFrame, dict]:
    bt.START, bt.END = since, until               # load_rows はこの範囲の凍結 bundle を読む
    rows, info = bt.load_rows(ROOT)
    df = pd.DataFrame(rows)
    if df.empty:
        return df, info
    df = df[df["era"].isin(eras)].copy()
    return df, info


def zure_check(dates: list[str]) -> dict:
    out = {"checked": 0, "match": 0, "mismatch": [], "no_freeze": 0}
    for d in dates:
        zf = ROOT / "reports" / "zure_freeze" / f"{d}.json"
        bp = ROOT / "reports" / "cowork_input_frozen" / f"{d}_bundle.json"
        if not zf.exists():
            out["no_freeze"] += 1
            continue
        out["checked"] += 1
        rec = json.loads(zf.read_text(encoding="utf-8"))
        want = rec.get("bundle_sha256") or ((rec.get("bundle") or {}).get("sha256"))
        have = hashlib.sha256(bp.read_bytes()).hexdigest()
        if want == have:
            out["match"] += 1
        else:
            out["mismatch"].append(d)
    return out


def clip(p):
    return np.clip(np.asarray(p, float), P_LO, P_HI)


def softmax_by(rid: np.ndarray, z: np.ndarray) -> np.ndarray:
    s = pd.Series(z)
    z = z - s.groupby(rid).transform("max").to_numpy()
    e = np.exp(z)
    return e / pd.Series(e).groupby(rid).transform("sum").to_numpy()


def winner_nll_by_race(df: pd.DataFrame, col: str) -> pd.Series:
    w = df[df["win"] == 1]
    return pd.Series(-np.log(clip(w[col].to_numpy())), index=w.index).groupby(w["rid"]).mean()


def boot_ci(d: np.ndarray, rng) -> list[float]:
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    m = d[idx].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def compare(df: pd.DataFrame, a: str, b: str, rng) -> dict:
    """a − b。2 値 LogLoss (レース平均をレース単位で) と勝ち馬 NLL。負 = a が良い。"""
    y = df["win"].to_numpy(float)
    ll = lambda p: -(y * np.log(clip(p)) + (1 - y) * np.log(1 - clip(p)))  # noqa: E731
    d = pd.Series(ll(df[a]) - ll(df[b])).groupby(df["rid"].to_numpy())
    s, c = d.sum().to_numpy(), d.size().to_numpy()
    idx = rng.integers(0, len(s), size=(N_BOOT, len(s)))
    boots = s[idx].sum(1) / c[idx].sum(1)
    wa, wb = winner_nll_by_race(df, a), winner_nll_by_race(df, b)
    dw = (wa - wb).to_numpy()
    return {"logloss_diff": float(s.sum() / c.sum()),
            "logloss_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "winner_nll_diff": float(dw.mean()), "winner_nll_ci95": boot_ci(dw, rng), "n_races": int(len(dw))}


def fit_blend(df: pd.DataFrame) -> dict:
    rid = df["rid"].to_numpy()
    lm, lq = np.log(clip(df["m"])), np.log(clip(df["q10"]))
    win = df["win"].to_numpy(float)
    nw = pd.Series(win).groupby(rid).transform("sum").to_numpy()
    n_races_with_winner = int((pd.Series(win).groupby(rid).sum() > 0).sum())

    def nll(ab):
        """レースごとの −Σ y·log p / Σ y (同着は等分) の平均。"""
        p = softmax_by(rid, ab[0] * lm + ab[1] * lq)
        per_row = win * np.log(np.clip(p, 1e-12, None)) / np.where(nw > 0, nw, 1)
        return float(-per_row.sum() / n_races_with_winner)
    r = minimize(nll, x0=np.array([0.0, 1.0]), method="Nelder-Mead",
                 options={"xatol": 1e-4, "fatol": 1e-7, "maxiter": 2000})
    return {"a_model": float(r.x[0]), "b_market": float(r.x[1]), "ok": bool(r.success)}


def apply_blend(df: pd.DataFrame, ab: dict) -> np.ndarray:
    rid = df["rid"].to_numpy()
    return softmax_by(rid, ab["a_model"] * np.log(clip(df["m"])) + ab["b_market"] * np.log(clip(df["q10"])))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eras", default="v8", help="凍結 bundle の世代 (bundle_provenance.model_era)。カンマ区切り")
    ap.add_argument("--since", default="20261121")
    ap.add_argument("--until", default=dt.date.today().strftime("%Y%m%d"))
    ap.add_argument("--force-final", action="store_true", help="動作確認用: レース数に関係なく本判定の形で出す")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--seed", type=int, default=20261119)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    eras = {e.strip() for e in a.eras.split(",") if e.strip()}

    df, info = load(eras, a.since, a.until)
    out = {"args": vars(a), "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
           "dates_excluded_no_prerace_bundle": info.get("dates_no_frozen", [])}
    if df.empty:
        out["stage"] = "NO_DATA"
        print(json.dumps(out, ensure_ascii=False, indent=1))
        if a.out_json:
            Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out_json).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        return
    out["zure_freeze"] = zure_check(sorted(df["date"].unique()))
    df = df[df["o_fin"].notna() & ~df["no_finish"] & df["p_win"].notna()].copy()
    posts = post_times(ROOT, sorted(df["date"].unique()))
    odds, _, st = t10_odds(ROOT, sorted(df["rid"].unique()), posts)
    df["o10"] = [odds.get(r, {}).get(u) for r, u in zip(df["rid"], df["uma"])]
    bad = df.groupby("rid")["o10"].apply(lambda s: s.isna().any())
    out["races_without_t10"] = int(bad.sum())
    df = df[~df["rid"].isin(set(bad[bad].index))].copy()
    df["win"] = (df["tan_ret"] > 0).astype(int)
    for col, src in (("m", "p_win"), ("q10", "o10"), ("qfin", "o_fin")):
        v = df[src].astype(float) if col == "m" else 1.0 / df[src].astype(float)
        v = v.clip(lower=1e-6) if col == "m" else v
        df[col] = v / v.groupby(df["rid"]).transform("sum")

    n_races = int(df["rid"].nunique())
    stage = ("FINAL" if (n_races >= FINAL_RACES or a.force_final) else
             "INTERIM" if n_races >= INTERIM_RACES else "COLLECTING")
    dates = sorted(df["date"].unique())
    out.update({"stage": stage, "n_races": n_races, "n_horses": int(len(df)),
                "dates": [dates[0], dates[-1], len(dates)],
                "eras": df.drop_duplicates("rid")["era"].value_counts().to_dict()})

    # (1) モデル単独 vs 市場 (全レース)
    y = df["win"].to_numpy(float)
    ll = lambda p: float(np.mean(-(y * np.log(clip(p)) + (1 - y) * np.log(1 - clip(p)))))  # noqa: E731
    out["logloss"] = {c: ll(df[c]) for c in ("m", "q10", "qfin")}
    out["winner_nll"] = {c: float(winner_nll_by_race(df, c).mean()) for c in ("m", "q10", "qfin")}
    out["model_vs_market"] = {"m-q10": compare(df, "m", "q10", rng), "m-qfin": compare(df, "m", "qfin", rng),
                              "q10-qfin": compare(df, "q10", "qfin", rng)}

    # (2) ブレンド: 前半の日で fit、後半の日で評価
    half = len(dates) // 2
    fit_d, ev_d = set(dates[:half]), set(dates[half:])
    fit, ev = df[df["date"].isin(fit_d)].copy(), df[df["date"].isin(ev_d)].copy()
    blend = {"fit_dates": [min(fit_d), max(fit_d), len(fit_d)] if fit_d else None,
             "eval_dates": [min(ev_d), max(ev_d), len(ev_d)], "fit_races": int(fit["rid"].nunique()),
             "eval_races": int(ev["rid"].nunique())}
    if len(fit) and len(ev):
        ab = fit_blend(fit)
        ev["blend"] = apply_blend(ev, ab)
        blend["weights_fit_half"] = ab
        blend["eval"] = {"blend-q10": compare(ev, "blend", "q10", rng), "blend-qfin": compare(ev, "blend", "qfin", rng),
                         "m-q10": compare(ev, "m", "q10", rng)}
    blend["weights_all_dates_descriptive"] = fit_blend(df)
    out["blend"] = blend

    if stage == "FINAL" and "eval" in blend:
        hi = blend["eval"]["blend-q10"]["winner_nll_ci95"][1]
        out["decision"] = "BLEND" if hi < 0 else "MODEL_ONLY"
        out["decision_rule"] = "後半の日の (ブレンド − T-10 市場) 勝ち馬 NLL の 95%CI 上限 < 0 なら BLEND"
    else:
        out["decision"] = None

    if a.out_json:
        Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out_json).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    f = lambda d, k: f"{d[k + '_diff']:+.5f} [{d[k + '_ci95'][0]:+.5f}, {d[k + '_ci95'][1]:+.5f}]"  # noqa: E731
    print(f"段階: {stage}  ({n_races}R / {len(dates)} 開催日 {dates[0]}〜{dates[-1]}、世代 {out['eras']})")
    print(f"zure_freeze 照合: {out['zure_freeze']}  T-10 欠けで除外 {out['races_without_t10']}R")
    print(f"LogLoss  モデル {out['logloss']['m']:.5f} / T-10 市場 {out['logloss']['q10']:.5f} / 最終市場 {out['logloss']['qfin']:.5f}")
    for k, v in out["model_vs_market"].items():
        print(f"  {k:9s} LogLoss {f(v, 'logloss')}  勝ち馬NLL {f(v, 'winner_nll')}")
    if "eval" in blend:
        w = blend["weights_fit_half"]
        print(f"ブレンド (前半 {blend['fit_races']}R で fit): a_model={w['a_model']:.3f} b_market={w['b_market']:.3f} → 後半 {blend['eval_races']}R")
        for k, v in blend["eval"].items():
            print(f"  {k:10s} LogLoss {f(v, 'logloss')}  勝ち馬NLL {f(v, 'winner_nll')}")
    wa = blend["weights_all_dates_descriptive"]
    print(f"全期間で fit した重み (記述): a_model={wa['a_model']:.3f} b_market={wa['b_market']:.3f}")
    print(f"判定: {out['decision'] or '（本判定の段階ではない）'}")


if __name__ == "__main__":
    main()
