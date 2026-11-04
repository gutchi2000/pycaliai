"""EV>1 の馬を全レース・頭数無制限で 単勝100円+複勝100円 均等買いしたら (2026)。

ユーザー質問 (2026-11-04): 「全レース期待値1越えの馬を単勝100円複勝100円で買ってたら？
1Rに何頭いてもいい。今年で計算できる？」

データ:
  - reports/cowork_input/{date}_bundle.json : v6 の p_win / p_plc / p_sho と
    朝オッズ (tansho_odds, fuku_odds_low/high = TARGET 出走表エクスポート時点)。2026-04-18〜
  - data/forward_prices/{date}/{rid}_t10_*.json.gz : T-10 の JV-Link 単勝・複勝オッズ。2026-08-29〜
  - data/kekka/{date}.csv : 確定着順・単勝配当・複勝配当。単勝配当列の "(x.x)" は
    非勝利馬の確定単勝オッズ (全頭) なので「確定オッズ基準」も参考値として出せる。

EV 定義:
  単勝 EV = p_win × 単勝オッズ
  複勝 EV = p_sho × 複勝オッズ下限 (頭数 7 以下は複勝が 2 着までなので p_plc を使う)
買い方 (2 ルール):
  rule=tan   : 単勝 EV>1 の馬に 単勝100 + 複勝100 (質問の文面どおり)
  rule=each  : 単勝 EV>1 なら単勝100、複勝 EV>1 なら複勝100 (券種ごとに独立判定)
取消 (確定着順 0) は返還扱い (投資に数え、払戻 = 元金)。

実行: python -m analysis.ev1_flat_tanfuku  → reports/ev1_flat_tanfuku_2026.{csv,json,md}
"""
from __future__ import annotations

import glob
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = ROOT / "reports" / "ev1_flat_tanfuku_2026_horses.csv"
OUT_JSON = ROOT / "reports" / "ev1_flat_tanfuku_2026.json"
OUT_MD = ROOT / "reports" / "ev1_flat_tanfuku_2026.md"

EV_BANDS = [(1.0, 1.1), (1.1, 1.25), (1.25, 1.5), (1.5, 2.0), (2.0, 99.0)]


def _f(v):
    try:
        if v is None:
            return None
        x = float(str(v).replace(",", "").strip())
        return x if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _parse_tan_cell(v):
    """単勝配当セル → (払戻 per100 or 0, 確定単勝オッズ or None)。"(3.0)" は非勝利馬の確定オッズ。"""
    s = str(v).strip()
    if s in ("", "nan", "None"):
        return 0.0, None
    if s.startswith("("):
        return 0.0, _f(s.strip("()"))
    pay = _f(s)
    return (pay or 0.0), (pay / 100.0 if pay else None)


def _load_t10(date: str) -> dict[str, dict]:
    """rid16 → 最新 T-10 market payload。"""
    out: dict[str, tuple[str, dict]] = {}
    for f in glob.glob(str(ROOT / "data" / "forward_prices" / date / "*_t10_*.json.gz")):
        try:
            d = json.load(gzip.open(f, "rt", encoding="utf-8"))
        except Exception:
            continue
        m = d.get("market") or {}
        if not m.get("ok"):
            continue
        rid = str(d.get("race_id") or m.get("race_id"))[:16]
        obs = str(d.get("observed_at") or "")
        if rid not in out or obs > out[rid][0]:
            out[rid] = (obs, m)
    return {k: v[1] for k, v in out.items()}


def build_rows() -> pd.DataFrame:
    rows = []
    n_race_no_kekka = 0
    for bf in sorted(glob.glob(str(ROOT / "reports" / "cowork_input" / "2026*_bundle.json"))):
        date = os.path.basename(bf)[:8]
        kf = ROOT / "data" / "kekka" / f"{date}.csv"
        if not kf.exists():
            continue
        kk = pd.read_csv(kf, encoding="cp932")
        kk["rid16"] = kk["レースID(新)"].astype(str).str[:16]
        kk["uma"] = pd.to_numeric(kk["馬番"], errors="coerce")
        t10 = _load_t10(date)
        b = json.load(open(bf, encoding="utf-8"))
        for race in b.get("races", []):
            rid = str(race["race_id"])[:16]
            sub = kk[kk["rid16"] == rid]
            if sub.empty:
                n_race_no_kekka += 1
                continue
            meta = race.get("race_meta") or {}
            fs = len(race.get("horses") or [])
            mk = t10.get(rid) or {}
            t_tan = mk.get("tansho") or {}
            t_fuku = mk.get("fukusho") or {}
            for h in race.get("horses") or []:
                u = int(h["umaban"])
                row = sub[sub["uma"] == u]
                if row.empty:
                    continue
                r0 = row.iloc[0]
                pos = _f(r0["確定着順"])
                pos = int(pos) if pos is not None else None
                tan_pay, final_tan = _parse_tan_cell(r0["単勝配当"])
                fuku_pay = _f(r0["複勝配当"]) or 0.0
                if pos != 1:
                    tan_pay = 0.0
                if pos not in (1, 2, 3):
                    fuku_pay = 0.0
                tf = t_fuku.get(str(u))
                rows.append({
                    "date": date, "month": date[4:6], "rid": rid,
                    "place": meta.get("place"), "klass": meta.get("class"),
                    "field_size": fs, "umaban": u, "name": h.get("horse_name"),
                    "mark": h.get("mark"),
                    "p_win": _f(h.get("p_win")), "p_plc": _f(h.get("p_plc")), "p_sho": _f(h.get("p_sho")),
                    "am_tan": _f(h.get("tansho_odds")),
                    "am_fuku_lo": _f(h.get("fuku_odds_low")), "am_fuku_hi": _f(h.get("fuku_odds_high")),
                    "t10_tan": _f(t_tan.get(str(u))),
                    "t10_fuku_lo": _f(tf[0]) if isinstance(tf, list) and tf else None,
                    "t10_fuku_hi": _f(tf[1]) if isinstance(tf, list) and len(tf) > 1 else None,
                    "final_tan": final_tan,
                    "pos": pos, "cancelled": pos == 0,
                    "tan_pay": tan_pay, "fuku_pay": fuku_pay,
                })
    df = pd.DataFrame(rows)
    df.attrs["n_race_no_kekka"] = n_race_no_kekka
    return df


# ---------------------------------------------------------------- evaluation
def _p_fuku(df: pd.DataFrame) -> pd.Series:
    """複勝の該当確率: 頭数 8 以上は p_sho (3着内)、7 以下は p_plc (2着内)。"""
    return np.where(df["field_size"] <= 7, df["p_plc"], df["p_sho"])


def _settle(sel: pd.DataFrame, buy_tan: pd.Series, buy_fuku: pd.Series) -> pd.DataFrame:
    s = sel.copy()
    s["buy_tan"] = buy_tan.astype(bool)
    s["buy_fuku"] = buy_fuku.astype(bool)
    s = s[s["buy_tan"] | s["buy_fuku"]].copy()
    canc = s["cancelled"]
    s["inv"] = 100 * s["buy_tan"] + 100 * s["buy_fuku"]
    s["ret"] = np.where(canc, s["inv"],
                        s["buy_tan"] * s["tan_pay"] + s["buy_fuku"] * s["fuku_pay"])
    s["hit_tan"] = s["buy_tan"] & (s["pos"] == 1)
    s["hit_fuku"] = s["buy_fuku"] & (s["fuku_pay"] > 0)
    return s


def _roi_ci(s: pd.DataFrame, n_boot: int = 3000, seed: int = 7) -> tuple[float, float]:
    """レース単位クラスタ bootstrap の ROI 95% CI。"""
    g = s.groupby("rid")[["inv", "ret"]].sum()
    if len(g) < 5:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    inv = g["inv"].to_numpy(); ret = g["ret"].to_numpy()
    idx = rng.integers(0, len(g), size=(n_boot, len(g)))
    rois = ret[idx].sum(axis=1) / inv[idx].sum(axis=1) * 100
    return (float(np.percentile(rois, 2.5)), float(np.percentile(rois, 97.5)))


def _summ(s: pd.DataFrame) -> dict:
    inv = float(s["inv"].sum()); ret = float(s["ret"].sum())
    out = {
        "n_horse_bets": int(len(s)), "n_races": int(s["rid"].nunique()),
        "n_days": int(s["date"].nunique()),
        "n_tan": int(s["buy_tan"].sum()), "n_fuku": int(s["buy_fuku"].sum()),
        "invest": int(inv), "return": int(round(ret)), "pnl": int(round(ret - inv)),
        "roi": round(ret / inv * 100, 1) if inv else None,
        "tan_hit_rate": round(float(s.loc[s["buy_tan"], "hit_tan"].mean() * 100), 1) if s["buy_tan"].any() else None,
        "fuku_hit_rate": round(float(s.loc[s["buy_fuku"], "hit_fuku"].mean() * 100), 1) if s["buy_fuku"].any() else None,
    }
    if s["buy_tan"].any():
        t = s[s["buy_tan"]]
        out["tan_roi"] = round(float(np.where(t["cancelled"], 100, t["tan_pay"]).sum() / (100 * len(t)) * 100), 1)
    if s["buy_fuku"].any():
        f = s[s["buy_fuku"]]
        out["fuku_roi"] = round(float(np.where(f["cancelled"], 100, f["fuku_pay"]).sum() / (100 * len(f)) * 100), 1)
    lo, hi = _roi_ci(s)
    out["roi_ci95"] = [round(lo, 1), round(hi, 1)]
    return out


def evaluate(df: pd.DataFrame, tan_col: str, fuku_col: str, rule: str) -> dict:
    d = df[df[tan_col].notna() & df["p_win"].notna()].copy()
    d["ev_tan"] = d["p_win"] * d[tan_col]
    pf = _p_fuku(d)
    d["ev_fuku"] = np.where(d[fuku_col].notna(), pf * d[fuku_col].fillna(0), np.nan)
    if rule == "tan":
        q = d["ev_tan"] > 1.0
        s = _settle(d, q, q)
    elif rule == "each":
        s = _settle(d, d["ev_tan"] > 1.0, d["ev_fuku"] > 1.0)
    elif rule == "all":  # 全馬ベースライン
        s = _settle(d, pd.Series(True, index=d.index), pd.Series(True, index=d.index))
    else:
        raise ValueError(rule)
    res = _summ(s)
    res["universe_races"] = int(d["rid"].nunique())
    res["universe_horses"] = int(len(d))
    res["bets_per_race_with_bet"] = round(len(s) / s["rid"].nunique(), 2) if len(s) else 0
    res["by_month"] = {m: _summ(g) for m, g in s.groupby("month")}
    if rule in ("tan", "each"):
        bands = {}
        key = "ev_tan"
        for lo, hi in EV_BANDS:
            g = s[(s[key] >= lo) & (s[key] < hi) & s["buy_tan"]]
            if len(g):
                gg = g.copy(); gg["buy_fuku"] = False
                gg["inv"] = 100; gg["ret"] = np.where(gg["cancelled"], 100, gg["tan_pay"])
                bands[f"{lo}-{hi}"] = {"n": int(len(gg)), "tan_roi": round(float(gg["ret"].sum() / gg["inv"].sum() * 100), 1),
                                       "hit": round(float((gg["pos"] == 1).mean() * 100), 1)}
        res["tan_by_ev_band"] = bands
        # 人気帯 (そのオッズ基準の単勝オッズで)
        ob = {}
        for lo, hi, lab in [(0, 3, "<3倍"), (3, 10, "3-10倍"), (10, 30, "10-30倍"), (30, 100, "30-100倍"), (100, 1e9, "100倍+")]:
            g = s[(s[tan_col] >= lo) & (s[tan_col] < hi)]
            if len(g):
                ob[lab] = {k: v for k, v in _summ(g).items() if k in ("n_horse_bets", "invest", "return", "roi", "tan_hit_rate", "fuku_hit_rate")}
        res["by_odds_band"] = ob
    return res, s


def main() -> int:
    df = build_rows()
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"rows={len(df)} races={df['rid'].nunique()} days={df['date'].nunique()} "
          f"no_kekka_races={df.attrs.get('n_race_no_kekka')}")
    print(f"T-10 odds coverage: horses={df['t10_tan'].notna().sum()} races={df.loc[df['t10_tan'].notna(),'rid'].nunique()}")
    print(f"final odds coverage: {df['final_tan'].notna().sum()}")

    results = {"generated_from": {"bundles": sorted(df['date'].unique().tolist())}}
    variants = [
        ("am_tan_rule",   "am_tan",    "am_fuku_lo",  "tan",  "朝オッズ基準 / 単勝EV>1 → 単+複"),
        ("am_each_rule",  "am_tan",    "am_fuku_lo",  "each", "朝オッズ基準 / 券種ごと独立判定"),
        ("am_all",        "am_tan",    "am_fuku_lo",  "all",  "朝オッズ期間 全馬 単+複 (ベースライン)"),
        ("t10_tan_rule",  "t10_tan",   "t10_fuku_lo", "tan",  "T-10オッズ基準 / 単勝EV>1 → 単+複"),
        ("t10_each_rule", "t10_tan",   "t10_fuku_lo", "each", "T-10オッズ基準 / 券種ごと独立判定"),
        ("t10_all",       "t10_tan",   "t10_fuku_lo", "all",  "T-10期間 全馬 単+複 (ベースライン)"),
        ("final_tan_rule","final_tan", "am_fuku_lo",  "tan",  "確定単勝オッズ基準 (事後・参考) / 単勝EV>1 → 単+複"),
    ]
    settled = {}
    for key, tc, fc, rule, label in variants:
        r, s = evaluate(df, tc, fc, rule)
        r["label"] = label
        results[key] = r
        settled[key] = s
        print(f"\n== {label}")
        print({k: v for k, v in r.items() if k not in ("by_month", "tan_by_ev_band", "by_odds_band", "label")})
        if "tan_by_ev_band" in r:
            print(" EV帯:", r["tan_by_ev_band"])
            print(" オッズ帯:", r["by_odds_band"])
        print(" 月別:", {m: (v["n_horse_bets"], v["roi"]) for m, v in r["by_month"].items()})

    # 同一期間比較 (T-10 期間に朝オッズ基準を適用)
    t10_dates = set(df.loc[df["t10_tan"].notna(), "date"])
    r, _ = evaluate(df[df["date"].isin(t10_dates)], "am_tan", "am_fuku_lo", "tan")
    r["label"] = "朝オッズ基準を T-10 期間 (8/29〜) に限定"
    results["am_tan_rule_t10period"] = r
    print("\n== 同一期間比較 (朝オッズ基準, 8/29〜):", {k: v for k, v in r.items() if k in ("n_horse_bets", "n_races", "invest", "return", "roi", "roi_ci95")})

    json.dump(results, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    # 買い目明細 (T-10 基準) も残す
    settled["t10_tan_rule"].to_csv(OUT_CSV.with_name("ev1_flat_tanfuku_2026_t10_bets.csv"), index=False, encoding="utf-8-sig")
    settled["am_tan_rule"].to_csv(OUT_CSV.with_name("ev1_flat_tanfuku_2026_am_bets.csv"), index=False, encoding="utf-8-sig")
    print(f"\nwrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
