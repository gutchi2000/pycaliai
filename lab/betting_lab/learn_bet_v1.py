# -*- coding: utf-8 -*-
"""
learn_bet_v1.py — 馬券最適化を「学習問題」として解く第一版 (leak-safe)
================================================================================
問い: 固定戦略(◎単/複, UMAMIビン)は全部 ROI~85% で控除率(20%)を超えない。
      → では AI確率×市場オッズを **学習でブレンド** して +EV の部分集合だけ賭けたら
        控除率を超えられるか? (Benter 型)

設計:
  - ベースモデル = unified_rank_v6 (train〜2022 / valid 2023) ⇒ 2024-25 は OOS
  - p_win/p_sho = v6 PL → calibrator (校正済み確率)
  - 学習データ = master test の 2024 のみ / 評価 = 2025 のみ (完全 OOS)
  - 市場 = 9時オッズ (前売り, leak なし) / 決済 = 確定配当
  - 学習器:
      (A) Benter ロジット: y(勝/複) ~ logit(p_model) + logit(q_market_norm)
          → ブレンド確率 → EV=p_blend*odds → EV>=t のみ賭ける (閾値スイープ)
      (B) LightGBM: リッチ特徴で勝/複を学習 → 同様に EV 選抜
  - 比較対象: 固定◎ / UMAMI-A+ (exp_umami_vs_marks と同じ土俵)

初回実行は master を読んで per-horse 行を data/_xai/bet_horse_rows.parquet にキャッシュ。
以後はキャッシュから学習のみ(高速)。

実行: PYTHONUTF8=1 python learn_bet_v1.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = Path(__file__).parent
CACHE = BASE / "data/_xai/bet_horse_rows.parquet"
FIT_YEAR, EVAL_YEAR = 2024, 2025


# ----------------------------------------------------------------------------
def build_cache():
    import joblib
    import pl_probs as PL
    from backtest_pl_ev import all_fukusho_vec_fast, load_payouts
    from audit_ev_bin_roi import (load_tanpuk, _rid16, MASTER_CSV,
                                  COL_RID, COL_JYUN, COL_BAN)

    print("[load] v6 model + calibrators")
    b = joblib.load(BASE / "models/unified_rank_v6.pkl")
    model, feats, encs = b["model"], b["feature_cols"], b["encoders"]
    cal = joblib.load(BASE / "models/pl_calibrators_v6.pkl"); cal = cal.get("calibrators", cal)

    print("[load] master (test 2024-25)")
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df = df.dropna(subset=[COL_JYUN, COL_RID, "split"])
    df = df[df["split"] == "test"].copy()
    for c, le in encs.items():
        if c not in df.columns:
            continue
        v = df[c].astype(str).fillna("__NaN__")
        df[c] = le.transform(v.where(v.isin(set(le.classes_)), "__NaN__"))
    X = df[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    df["_score"] = model.predict(X)

    print("[load] payouts + 9時 odds")
    payouts = load_payouts()
    tanpuk = load_tanpuk()

    out = []
    for rid, g in df.groupby(COL_RID, sort=False):
        if len(g) < 5:
            continue
        rid16 = _rid16(rid)
        year = int(rid16[:4])
        if year not in (FIT_YEAR, EVAL_YEAR):
            continue
        sp = tanpuk.get(rid16)
        if not sp:
            continue
        po = payouts.get(rid16)
        g = g.sort_values(COL_BAN).reset_index(drop=True)
        ban = g[COL_BAN].astype(int).values
        n = len(ban)
        w = PL.pl_weights(g["_score"].values)
        p_win = cal["tansho"].predict(PL.all_tansho(w))
        p_sho = cal["fukusho"].predict(all_fukusho_vec_fast(w))
        win_rank = (-p_win).argsort().argsort() + 1   # 1=最有力(◎)
        top3 = {po["win"], po["plc"], po["sho"]} if po else set()
        fpay = {}
        if po:
            for bn, k in [(po["win"], "fuku_win"), (po["plc"], "fuku_plc"), (po["sho"], "fuku_sho")]:
                if po.get(k) and not pd.isna(po[k]):
                    fpay[int(bn)] = float(po[k])
        for i in range(n):
            o9 = sp["tan9"].get(int(ban[i]))
            if o9 is None or o9 <= 1.0:
                continue
            won = 1 if (po is not None and int(ban[i]) == po["win"]) else 0
            tanpay = float(po["tansho"]) if (won and po and po.get("tansho") and not pd.isna(po["tansho"])) else 0.0
            in3 = 1 if (po is not None and int(ban[i]) in top3) else 0
            fukupay = fpay.get(int(ban[i]), 0.0) if in3 else 0.0
            fo = sp["fuku9"].get(int(ban[i])) if sp.get("fuku9") else None
            place_odds = ((fo[0] + fo[1]) / 2.0) if fo else np.nan
            out.append(dict(year=year, rid=rid16, ban=int(ban[i]), n=n,
                            p_win=float(p_win[i]), p_sho=float(p_sho[i]),
                            odds=float(o9), place_odds=place_odds,
                            win_rank=int(win_rank[i]),
                            won=won, tanpay=tanpay, in3=in3, fukupay=fukupay))
    R = pd.DataFrame(out)
    CACHE.parent.mkdir(exist_ok=True)
    R.to_parquet(CACHE)
    print(f"[cache] saved {len(R):,} rows → {CACHE}")
    return R


# ----------------------------------------------------------------------------
def add_market(R):
    """市場 implied prob を race 内で正規化 (overround 除去)。"""
    R = R.copy()
    R["q_raw"] = 1.0 / R["odds"]
    R["q_mkt"] = R["q_raw"] / R.groupby("rid")["q_raw"].transform("sum")
    R["q_place_raw"] = 1.0 / R["place_odds"]
    return R


def _logit(p, eps=1e-5):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def renorm(R, col):
    """race 内で col を合計1に正規化した新 series を返す。"""
    s = R.groupby("rid")[col].transform("sum")
    return R[col] / s.replace(0, np.nan)


def ci95(rets):
    rets = np.asarray(rets, float)
    n = len(rets)
    if n < 2:
        return (np.nan, np.nan)
    roi = rets.mean() / 100
    se = rets.std(ddof=1) / np.sqrt(n) / 100
    return (roi - 1.96 * se, roi + 1.96 * se)


def report(name, sel, paycol):
    """sel: bool index on EVAL rows. paycol: 'tanpay'/'fukupay'."""
    g = EV[sel]
    n = len(g)
    if n == 0:
        print(f"  {name:30s} n=0")
        return
    rets = g[paycol].values
    roi = rets.mean() / 100
    hit = (rets > 0).mean()
    lo, hi = ci95(rets)
    print(f"  {name:30s} n={n:>6,} hit={hit*100:>5.1f}% ROI={roi*100:>6.1f}% "
          f"CI[{lo*100:>5.1f},{hi*100:>5.1f}] odds中={g['odds'].median():>5.1f}")


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    if CACHE.exists():
        R = pd.read_parquet(CACHE)
        print(f"[cache] loaded {len(R):,} rows")
    else:
        R = build_cache()
    R = add_market(R)
    fit = R[R.year == FIT_YEAR].copy()
    EV = R[R.year == EVAL_YEAR].copy()
    print(f"fit({FIT_YEAR})={len(fit):,}  eval({EVAL_YEAR})={len(EV):,}  "
          f"races eval={EV.rid.nunique():,}")

    from sklearn.linear_model import LogisticRegression

    # ============ (A) Benter ロジット ============
    for tgt, paycol, pcol, lab in [("won", "tanpay", "p_win", "単勝"),
                                   ("in3", "fukupay", "p_sho", "複勝")]:
        Xtr = np.column_stack([_logit(fit[pcol]), _logit(fit["q_mkt"])])
        clf = LogisticRegression(max_iter=1000)
        clf.fit(Xtr, fit[tgt])
        cm, cq = clf.coef_[0]
        wsum = abs(cm) + abs(cq)
        print(f"\n=== (A) Benter {lab}: 学習ブレンド係数 ===")
        print(f"   model={cm:+.3f} market={cq:+.3f}  →相対重み model {abs(cm)/wsum*100:.0f}% / market {abs(cq)/wsum*100:.0f}%")
        Xev = np.column_stack([_logit(EV[pcol]), _logit(EV["q_mkt"])])
        EV["_praw"] = clf.predict_proba(Xev)[:, 1]
        EV["_pblend"] = renorm(EV, "_praw")
        odds_use = EV["odds"] if tgt == "won" else EV["place_odds"]
        EV["_ev"] = EV["_pblend"] * odds_use
        print(f"--- {lab}: Benter EV 閾値スイープ (2025 OOS) ---")
        report(f"{lab} 固定◎ (rank1)", EV["win_rank"] == 1, paycol)
        for t in [1.0, 1.05, 1.1, 1.2, 1.3]:
            report(f"{lab} Benter EV>={t}", EV["_ev"] >= t, paycol)

    # ============ (B) LightGBM ============
    try:
        import lightgbm as lgb
        feats = ["p_win", "p_sho", "odds", "q_mkt", "win_rank", "n"]
        fit2 = fit.dropna(subset=feats)
        EV2 = EV.copy()
        for tgt, paycol, lab, odcol in [("won", "tanpay", "単勝", "odds"),
                                        ("in3", "fukupay", "複勝", "place_odds")]:
            dtr = lgb.Dataset(fit2[feats], label=fit2[tgt])
            params = dict(objective="binary", learning_rate=0.05, num_leaves=31,
                          min_data_in_leaf=200, feature_fraction=0.8,
                          bagging_fraction=0.8, bagging_freq=1, verbose=-1)
            gbm = lgb.train(params, dtr, num_boost_round=300)
            EV2["_praw"] = gbm.predict(EV2[feats])
            EV2["_pblend"] = renorm(EV2, "_praw")
            odds_use = EV2[odcol]
            EV2["_ev"] = EV2["_pblend"] * odds_use
            # write back to EV for report()
            globals()["EV"] = EV2
            print(f"\n=== (B) LightGBM {lab} EV 閾値スイープ (2025 OOS) ===")
            for t in [1.0, 1.05, 1.1, 1.2, 1.3]:
                report(f"{lab} GBM EV>={t}", EV2["_ev"] >= t, paycol)
    except Exception as e:
        print("LightGBM skip:", e)
