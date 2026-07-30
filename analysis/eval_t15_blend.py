"""
eval_t15_blend.py — 補正印 blend の「オッズ鮮度」別 ◎top3 実測
================================================================
サイトに T-15 で補正印を出す計画の精度見積り。歴史スナップは
前日23時 / 当日9時 / 発走35-40分前ごろ(区分1最終) / 確定(区分4) の4点しか
無いので、T-15 は「区分1最終 〜 確定」の間として挟み撃ちで答える。

アーム (λ=1.5 固定 = data/t10_blend.json の本番値):
  v6のみ / 9時π (既存実証65.1%の再現sanity) / 区分1最終π / 確定π / 市場のみ(確定)

実行: venv311/Scripts/python.exe -m analysis.eval_t15_blend
出力: reports/eval_t15_blend.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
import pl_probs as PL  # noqa: E402
from analysis.fit_t10_blend import wilson_ci, paired_boot_delta  # noqa: E402

np.random.seed(42)
LAM = 1.5
TANPUK = BASE / "data/Time _series_odds/TANPUK_20210105-20251228.csv"


def load_tanpuk_arms() -> pd.DataFrame:
    """rid16 ごとに 区分1最終 / 確定 の単勝オッズ {ban:odds} と時刻を取る。"""
    df = pd.read_csv(TANPUK, encoding="cp932", low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    rid_col, kbn_col, t_col = df.columns[0], df.columns[1], df.columns[2]
    odds_cols = [f"{i}単" for i in range(1, 19)]
    df["rid16"] = df[rid_col].astype(str).str.replace(r"\D", "", regex=True).str[:16]
    df["_t"] = pd.to_numeric(df[t_col], errors="coerce")

    rows = []
    for rid16, g in df.groupby("rid16", sort=False):
        g1 = g[g[kbn_col] == 1].sort_values("_t")
        g4 = g[g[kbn_col] == 4]
        if g1.empty or g4.empty:
            continue
        # 当日の区分1のうち最終 (9時スナップと同一日中でより遅いもの)
        last1, fin = g1.iloc[-1], g4.iloc[-1]
        rec = {"rid16": rid16, "t_last1": last1["_t"], "t_fin": fin["_t"]}
        for i in range(1, 19):
            c = odds_cols[i - 1]
            if c in g.columns:
                rec[f"o1_{i}"] = pd.to_numeric(last1[c], errors="coerce")
                rec[f"o4_{i}"] = pd.to_numeric(fin[c], errors="coerce")
        rows.append(rec)
    out = pd.DataFrame(rows)
    print(f"[tanpuk] races={len(out):,}")
    return out


def mmddhhmm_delta_min(a: float, b: float) -> float:
    """同日前提の分差 (b-a)。日跨ぎ/欠損は nan。"""
    if not (np.isfinite(a) and np.isfinite(b)):
        return np.nan
    a, b = int(a), int(b)
    if a // 10000 != b // 10000:
        return np.nan
    am, bm = (a // 100) % 100 * 60 + a % 100, (b // 100) % 100 * 60 + b % 100
    return bm - am


def main() -> None:
    sc = pd.read_parquet(BASE / "data/_ev_grid_scores.parquet")
    mk = pd.read_parquet(BASE / "data/_joint/mkt_horses.parquet",
                         columns=["rid", "ban", "pi", "top3"])
    sc["rid16"] = sc["レースID(新/馬番無)"].astype(str).str[:16]
    mk = mk.rename(columns={"rid": "rid16"})
    mk["rid16"] = mk["rid16"].astype(str)
    cal = joblib.load(BASE / "models/pl_calibrators_v6_serve.pkl")
    cal = cal.get("calibrators", cal)
    iso_tan = cal["tansho"]
    tp = load_tanpuk_arms().set_index("rid16")

    df = sc.merge(mk, left_on=["rid16", "馬番"], right_on=["rid16", "ban"],
                  how="inner")
    races, lead_mins = [], []
    for (rid16, year), g in df.groupby(["rid16", "year"], sort=False):
        if year < 2024 or len(g) < 5 or (g["pi"] <= 0).any():
            continue
        if rid16 not in tp.index:
            continue
        row = tp.loc[rid16]
        g = g.sort_values("馬番")
        bans = g["馬番"].astype(int).values
        o_last1 = np.array([row.get(f"o1_{b}", np.nan) for b in bans], dtype=float)
        o_fin = np.array([row.get(f"o4_{b}", np.nan) for b in bans], dtype=float)
        if not (np.isfinite(o_last1).all() and (o_last1 > 0).all()
                and np.isfinite(o_fin).all() and (o_fin > 0).all()):
            continue
        w = PL.pl_weights(g["_score"].values)
        f = np.maximum(iso_tan.predict(PL.all_tansho(w)), 1e-9)
        pi_l1 = (1 / o_last1) / (1 / o_last1).sum()
        pi_fin = (1 / o_fin) / (1 / o_fin).sum()
        races.append({
            "logf": np.log(f),
            "logpi9": np.log(g["pi"].values),
            "logpil1": np.log(pi_l1),
            "logpifin": np.log(pi_fin),
            "top3": g["top3"].values.astype(int),
        })
        lead_mins.append(mmddhhmm_delta_min(row["t_last1"], row["t_fin"]))

    n = len(races)
    lead = np.array(lead_mins, dtype=float)
    lead = lead[np.isfinite(lead)]
    print(f"[races] test2024-25 n={n:,}")
    print(f"[lead] 区分1最終→確定の分差: median={np.median(lead):.0f}min "
          f"p25={np.percentile(lead,25):.0f} p75={np.percentile(lead,75):.0f}")

    def hits(key: str | None) -> np.ndarray:
        h = np.empty(n, dtype=int)
        for i, r in enumerate(races):
            u = r["logf"] if key is None else r["logf"] + LAM * r[key]
            h[i] = r["top3"][int(np.argmax(u))]
        return h

    def hits_mktonly() -> np.ndarray:
        return np.array([r["top3"][int(np.argmax(r["logpifin"]))] for r in races])

    arms = {
        "v6のみ": hits(None),
        "blend 9時π (既存65.1%再現)": hits("logpi9"),
        "blend 区分1最終π (発走~35分前)": hits("logpil1"),
        "blend 確定π (T-0上限)": hits("logpifin"),
        "市場のみ (確定)": hits_mktonly(),
    }
    out = {"n_races": n, "lambda": LAM,
           "lead_min_median": float(np.median(lead)),
           "arms": {}}
    print(f"\n[test 2024-25, λ={LAM}] ◎top3:")
    base = arms["v6のみ"]
    for name, h in arms.items():
        lo, hi = wilson_ci(int(h.sum()), n)
        d_lo, d_hi = paired_boot_delta(h, base)
        print(f"  {name:<28}: {h.mean()*100:.2f}%  CI[{lo*100:.2f},{hi*100:.2f}]"
              f"  Δvs_v6 {(h.mean()-base.mean())*100:+.2f}pt CI[{d_lo*100:+.2f},{d_hi*100:+.2f}]")
        out["arms"][name] = {"top3": round(float(h.mean()), 5),
                             "ci": [round(lo, 5), round(hi, 5)],
                             "delta_vs_v6_ci": [round(d_lo, 5), round(d_hi, 5)]}
    (BASE / "reports/eval_t15_blend.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n[out] reports/eval_t15_blend.json")


if __name__ == "__main__":
    main()
