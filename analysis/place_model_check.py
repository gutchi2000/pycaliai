"""複勝専用モデルの検証 (分析専用、探索扱い)。

問い: 今の複勝率 (v6 スコア → PL の 3 着以内確率 → 配信用 isotonic) より、「3 着以内」を直接 log-loss で
学習した専用モデルの方が当たるか。

事前に固定した設計 (後半の結果を見てから変えない):
  - 学習 (train サブコマンド): v6 と同じ 120 特徴・エンコード・master (〜2022 で学習、2023 で早期停止)。
      目的関数 = binary log-loss、目的変数 = 着順 1〜3 (同着含む)。ハイパラは v6 から lambdarank 固有のものを
      除いてそのまま、標本重みなし、早期停止 100 回・上限 5,000 回。確率は生の出力 (レース内の正規化なし)。
      補助として test 2024-25 で現行の作り方 (v6 → PL → pl_calibrators_v6 の複勝 isotonic) と比べる。
  - 評価 (eval サブコマンド): 2026 の X は analysis.objective_check_replay で再生したもの (全モデル同じ X)。
      現行 = 再生 bundle の p_sho (v6 → PL → pl_calibrators_v6_serve の複勝)。参考に凍結版 p_sho。
      採点対象: 凍結 bundle の馬のうち、中止等でない・再生 X あり・T-10 の複勝オッズ (下限) ありの馬。
      1 頭でも欠けるレースは外す。**出走 7 頭以下のレース (複勝は 2 着まで) は主評価から外す** (どちらの確率も
      3 着以内の確率なので、払戻条件と合わない)。外したレース数は報告する。
      分割: 前半 〜08-15 / 後半 08-16〜。専用モデルは 2026 のデータで何も fit しないので、前半は参考値。
      後半を 1 回だけ評価する。
  - 主指標: 3 着以内の 2 値 LogLoss (確率は [0.001, 0.999] に制限) の差 (専用 − 現行)、レース単位の
      対応付きブートストラップ 4,000 回。副: Brier、ECE。
  - 副次: EV = 複勝率 × T-10 の複勝オッズ下限 (最終の複勝オッズは手元に無い: kekka には 3 着内の馬の
      確定配当しか無いため、買う馬を選ぶのに使えない)。閾値 1.0 / 1.1、全該当と最上位 1 頭、払戻は確定複勝配当。
      専用モデルと現行の両方で出す。比較: 全馬均等。

使い方:
  python -m analysis.place_model_check train --out reports/objective_check
  python -m analysis.place_model_check eval  --work reports/objective_check --out-json <path>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from analysis import ev_threshold_backtest as bt
from analysis.ev_recalibration_check import metrics, paired_metric_diff
from analysis.objective_check_train import RANK_ONLY, X_of, binary_ll, groups_of, load_split
from analysis.t10_market_blend_check import post_times, t10_odds

ROOT = Path("E:/PyCaLiAI")
SPLIT = "20260815"
EV_THRESHOLDS = [1.0, 1.1]
N_BOOT = 4000
RNG = np.random.default_rng(20261119)


def cmd_train(a) -> None:
    out = Path(a.out).resolve()
    (out / "models").mkdir(parents=True, exist_ok=True)
    pkl, feats, parts, _ = load_split()
    tr, vl, te = parts["train"], parts["valid"], parts["test"]
    import optuna_v6_marks as ovm
    y = {k: (d[ovm.COL_JYUN].astype(int).between(1, 3)).astype(float).to_numpy() for k, d in parts.items()}
    Xtr, Xvl, Xte = X_of(tr, feats), X_of(vl, feats), X_of(te, feats)
    params = {k: v for k, v in pkl["model"].params.items() if k not in RANK_ONLY}
    params.update(objective="binary", metric="binary_logloss")
    m = lgb.train(params, lgb.Dataset(Xtr, label=y["train"]), num_boost_round=5000,
                  valid_sets=[lgb.Dataset(Xvl, label=y["valid"])],
                  callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)])
    m.save_model(str(out / "models" / "place_p1.txt"))
    # 補助: test 2024-25 で現行の作り方と比べる
    import pl_probs as PL
    from backtest_pl_ev import all_fukusho_vec_fast
    cal = joblib.load(ROOT / "models" / "pl_calibrators_v6.pkl")["calibrators"]["fukusho"]
    s6 = pkl["model"].predict(Xte)
    starts, counts = groups_of(te)
    cur = np.empty(len(te))
    for a0, n in zip(starts, counts):
        cur[a0:a0 + n] = all_fukusho_vec_fast(PL.pl_weights(s6[a0:a0 + n]))
    cur = cal.predict(cur)
    p1 = m.predict(Xte, num_iteration=m.best_iteration)
    yt = y["test"]
    f = lambda p: -(yt * np.log(np.clip(p, 1e-3, 1 - 1e-3)) + (1 - yt) * np.log(1 - np.clip(p, 1e-3, 1 - 1e-3)))  # noqa: E731
    d = f(p1) - f(cur)
    rid = te[ovm.COL_RID].astype(str).to_numpy()
    uniq, inv = np.unique(rid, return_inverse=True)
    s, c = np.bincount(inv, weights=d), np.bincount(inv)
    idx = RNG.integers(0, len(uniq), size=(N_BOOT, len(uniq)))
    boots = s[idx].sum(1) / c[idx].sum(1)
    summ = {"best_iteration": m.best_iteration, "params": params,
            "test2024_25": {"p1_logloss": binary_ll(p1, yt), "current_logloss": binary_ll(cur, yt),
                            "diff": float(d.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                            "p1_brier": float(((p1 - yt) ** 2).mean()), "current_brier": float(((cur - yt) ** 2).mean()),
                            "races": int(len(uniq))}}
    (out / "place_train_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(summ["test2024_25"]), "best_iter", m.best_iteration)


def load_replay_place(work: Path) -> pd.DataFrame:
    frames = []
    for p in sorted((work / "feats").glob("2026????.parquet")):
        d = pd.read_parquet(p)
        d["date"] = p.stem
        ps = {}
        b = work / "bundles" / f"{p.stem}_bundle.json"
        if b.exists():
            for r in json.loads(b.read_text(encoding="utf-8"))["races"]:
                for h in r["horses"]:
                    ps[(str(r["race_id"])[:16], int(h["umaban"]))] = h.get("p_sho")
        d["rid"] = d["rid"].astype(str).str[:16]
        d["p_sho_replay"] = [ps.get((r, int(u))) for r, u in zip(d["rid"], d["umaban"])]
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def cmd_eval(a) -> None:
    work = Path(a.work).resolve()
    feats = json.loads((work / "feats" / "_feature_cols.json").read_text(encoding="utf-8"))
    rows, _ = bt.load_rows(ROOT)
    fz = pd.DataFrame(rows)
    fz["n_run"] = fz.groupby("rid")["uma"].transform("size")      # 出走頭数 (取消を除き、中止等は含む)
    fz = fz[~fz["no_finish"]].copy()
    rp = load_replay_place(work)
    m = lgb.Booster(model_file=str(work / "models" / "place_p1.txt"))
    rp["p1"] = m.predict(rp[[f"f{i}" for i in range(len(feats))]].to_numpy())
    df = fz.merge(rp[["rid", "umaban", "p1", "p_sho_replay"]], left_on=["rid", "uma"], right_on=["rid", "umaban"], how="left")
    posts = post_times(ROOT, sorted(df["date"].unique()))
    fo, _, _ = t10_odds(ROOT, sorted(df["rid"].unique()), posts, key="fukusho")
    df["fo10"] = [fo.get(r, {}).get(u) for r, u in zip(df["rid"], df["uma"])]
    bad = df.groupby("rid")[["p1", "p_sho_replay", "fo10", "p_sho"]].apply(lambda g: bool(g.isna().any().any()))
    df = df[~df["rid"].isin(set(bad[bad].index))].copy()
    small = df[df["n_run"] <= 7]["rid"].nunique()
    df = df[df["n_run"] >= 8].copy()
    # 3 着以内 (同着含む): build_site の top3 と同じ。8 頭以上では複勝払戻あり ⇔ 3 着以内
    df["top3"] = (df["fuku_ret"] > 0).astype(int)
    df = df.rename(columns={"p_sho": "p_sho_frozen"})
    fit, val = df[df["date"] <= SPLIT].copy(), df[df["date"] > SPLIT].copy()
    cols = ["p1", "p_sho_replay", "p_sho_frozen"]
    out = {"design": {"split": SPLIT, "ev_thresholds": EV_THRESHOLDS},
           "stat": {"dropped_missing": int(bad.sum()), "dropped_small_field_races": int(small)},
           "coverage": {h: {"races": int(d["rid"].nunique()), "horses": int(len(d)),
                            "dates": [d["date"].min(), d["date"].max(), int(d["date"].nunique())],
                            "eras": d.drop_duplicates("rid")["era"].value_counts().to_dict()}
                        for h, d in (("fit_reference", fit), ("val", val))},
           "fit_reference": {c: {k: v for k, v in metrics(fit, c, "top3").items() if k != "bands"} for c in cols}}
    # ---- 後半 (1 回だけ) ----
    out["val"] = {c: metrics(val, c, "top3") for c in cols}
    out["val_diff"] = {f"{c}-p_sho_replay": paired_metric_diff(val, c, "p_sho_replay", "top3") for c in ("p1", "p_sho_frozen")}
    ev = {}
    for c in ("p1", "p_sho_replay"):
        recs = val.to_dict("records")
        for r in recs:
            r["p_sho"], r["fo_low"] = r[c], r["fo10"]
        for k in ("fuku_all", "fuku_top1"):
            for t in EV_THRESHOLDS:
                ev[f"{c}|{k}@{t}"] = bt.summarize(bt.select(recs, k, t))
    out["val_ev"] = ev
    out["val_baseline_all_horses"] = bt.summarize([{**r, "ret": r["fuku_ret"]} for r in val.to_dict("records")])
    out["train_summary"] = json.loads((work / "place_train_summary.json").read_text(encoding="utf-8"))
    Path(a.out_json).write_text(json.dumps(out, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(out["stat"]), json.dumps(out["coverage"], ensure_ascii=False))
    for c in cols:
        mm = out["val"][c]
        print(f"val {c:14s} LL={mm['logloss']:.5f} Brier={mm['brier']:.5f} ECE={mm['ece']:.4f}")
    for k, v in out["val_diff"].items():
        print(k, f"LL {v['logloss']['diff']:+.5f} [{v['logloss']['ci95'][0]:+.5f},{v['logloss']['ci95'][1]:+.5f}]",
              f"Brier {v['brier']['diff']:+.6f} [{v['brier']['ci95'][0]:+.6f},{v['brier']['ci95'][1]:+.6f}]")
    for k, s in ev.items():
        print(f"{k:24s} bets={s.get('n_bets',0):5d} R={s.get('n_races',0):4d} hit={s.get('hit_rate',0)*100:5.1f}% "
              f"roi={s.get('roi',0)*100:6.1f}% CI[{s.get('roi_ci95',[0,0])[0]*100:.0f},{s.get('roi_ci95',[0,0])[1]*100:.0f}] "
              f"ex10={(s.get('roi_ex_top10') or 0)*100:.1f}%")
    print("baseline", out["val_baseline_all_horses"]["roi"])


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("train"); t.add_argument("--out", required=True)
    e = sp.add_parser("eval"); e.add_argument("--work", default="reports/objective_check"); e.add_argument("--out-json", required=True)
    a = ap.parse_args()
    {"train": cmd_train, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
