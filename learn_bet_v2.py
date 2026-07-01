# -*- coding: utf-8 -*-
"""
learn_bet_v2.py — 「特徴 vs 市場」決定的検証 + NN/GBM/Benter 横並び (leak-safe)
================================================================================
v1 の手抜き: 学習器に p_win/odds しか与えてない＝価格と冗長で新情報ゼロ。
v2: v6 の **全特徴 + 市場価格** を GBM と NN(MLP) に学習させ、
    2025 OOS で「特徴は市場価格を超える予測力を持つか」を logloss/AUC で
    市場単独・v6p_win 単独・Benter と **直接対決**。
    → 効率的市場(エッジ無し)なら full-feature でも price-only を OOS で超えない。
       超えれば エッジ実在 → EV選抜ROIに出る。これが「決めつけ」でなく検証。

leak規約: ベース v6=train〜2022/valid2023。価値モデルの学習=2024のみ / 評価=2025のみ。
          市場=9時オッズ(前売り) / 決済=確定単勝配当。

初回 master 読込→ data/_xai/bet_rich_rows.parquet にキャッシュ。
実行: PYTHONUTF8=1 python learn_bet_v2.py
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

BASE = Path(__file__).parent
CACHE = BASE / "data/_xai/bet_rich_rows.parquet"
META  = BASE / "data/_xai/bet_rich_meta.json"
FIT_YEAR, EVAL_YEAR = 2024, 2025


def build_cache():
    import joblib
    import pl_probs as PL
    from audit_ev_bin_roi import load_tanpuk, _rid16, MASTER_CSV, COL_RID, COL_JYUN, COL_BAN
    from backtest_pl_ev import load_payouts
    print("[load] v6 model")
    b = joblib.load(BASE / "models/unified_rank_v6.pkl")
    model, feats, encs = b["model"], b["feature_cols"], b["encoders"]
    cal = joblib.load(BASE / "models/pl_calibrators_v6.pkl"); cal = cal.get("calibrators", cal)
    print("[load] master")
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df = df.dropna(subset=[COL_JYUN, COL_RID, "split"])
    df = df[df["split"] == "test"].copy()
    df_enc = df.copy()
    for c, le in encs.items():
        if c not in df_enc.columns: continue
        v = df_enc[c].astype(str).fillna("__NaN__")
        df_enc[c] = le.transform(v.where(v.isin(set(le.classes_)), "__NaN__"))
    Xf = df_enc[feats].apply(pd.to_numeric, errors="coerce")
    df_enc["_score"] = model.predict(Xf.fillna(-9999).values)
    tanpuk = load_tanpuk(); payouts = load_payouts()
    rows = []
    feat_block = Xf.values
    fi = {f: k for k, f in enumerate(feats)}
    idx = 0
    for rid, g in df_enc.groupby(COL_RID, sort=False):
        if len(g) < 5: continue
        rid16 = _rid16(rid); year = int(rid16[:4])
        if year not in (FIT_YEAR, EVAL_YEAR): continue
        sp = tanpuk.get(rid16)
        if not sp: continue
        po = payouts.get(rid16)
        gi = g.index
        w = PL.pl_weights(g["_score"].values); pwin = w / w.sum()
        ban = g[COL_BAN].astype(int).values
        for r, gidx in enumerate(gi):
            o9 = sp["tan9"].get(int(ban[r]))
            if o9 is None or o9 <= 1.0: continue
            won = 1 if (po is not None and int(ban[r]) == po["win"]) else 0
            tanpay = float(po["tansho"]) if (won and po and po.get("tansho") and not pd.isna(po["tansho"])) else 0.0
            rec = dict(year=year, rid=rid16, odds=float(o9), p_win=float(pwin[r]),
                       won=won, tanpay=tanpay)
            featrow = df_enc.loc[gidx, feats]
            rows.append((rec, featrow.values))
    meta = {"feats": feats}
    base = pd.DataFrame([r[0] for r in rows])
    F = pd.DataFrame([r[1] for r in rows], columns=feats).apply(pd.to_numeric, errors="coerce")
    out = pd.concat([base.reset_index(drop=True), F.reset_index(drop=True)], axis=1)
    CACHE.parent.mkdir(exist_ok=True)
    out.to_parquet(CACHE); META.write_text(json.dumps(meta))
    print(f"[cache] {len(out):,} rows, {len(feats)} feats → {CACHE}")
    return out, feats


def market_norm(R):
    R = R.copy()
    R["q_raw"] = 1.0 / R["odds"]
    R["q_mkt"] = R["q_raw"] / R.groupby("rid")["q_raw"].transform("sum")
    return R


def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps); return np.log(p/(1-p))


def renorm(R, col):
    s = R.groupby("rid")[col].transform("sum"); return R[col]/s.replace(0, np.nan)


def roi_sweep(EV, pcol):
    out = []
    EVx = EV.copy()
    EVx["_pb"] = renorm(EVx, pcol)
    EVx["_ev"] = EVx["_pb"] * EVx["odds"]
    for t in [1.0, 1.1, 1.2]:
        s = EVx[EVx["_ev"] >= t]
        n = len(s)
        if n == 0: out.append((t, 0, 0, 0, 0, 0)); continue
        rets = s["tanpay"].values; roi = rets.mean()/100
        se = rets.std(ddof=1)/np.sqrt(n)/100 if n > 1 else 0
        out.append((t, n, (rets>0).mean(), roi, roi-1.96*se, roi+1.96*se))
    return out


if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.neural_network import MLPClassifier

    if CACHE.exists():
        R = pd.read_parquet(CACHE); feats = json.loads(META.read_text())["feats"]
        print(f"[cache] {len(R):,} rows, {len(feats)} feats")
    else:
        R, feats = build_cache()
    R = market_norm(R)
    tr = R[R.year == FIT_YEAR].copy(); ev = R[R.year == EVAL_YEAR].copy()
    print(f"fit2024={len(tr):,} eval2025={len(ev):,} races={ev.rid.nunique():,} 勝率={ev.won.mean():.3f}")
    base_ll = log_loss(ev.won, np.clip(ev.q_mkt, 1e-6, 1-1e-6))
    print(f"\n市場単独 baseline: OOS logloss={base_ll:.4f}  AUC={roc_auc_score(ev.won, ev.q_mkt):.4f}")

    res = {}
    # --- B1 v6 p_win only ---
    res["B1 v6p_win単独"] = (log_loss(ev.won, np.clip(ev.p_win,1e-6,1-1e-6)), roc_auc_score(ev.won, ev.p_win), ev.assign(_p=ev.p_win))
    # --- B2 Benter (p_win + price) ---
    Xtr = np.column_stack([_logit(tr.p_win), _logit(tr.q_mkt)]); Xev = np.column_stack([_logit(ev.p_win), _logit(ev.q_mkt)])
    lr = LogisticRegression(max_iter=2000).fit(Xtr, tr.won); p = lr.predict_proba(Xev)[:,1]
    print(f"  Benter係数: p_win={lr.coef_[0][0]:+.3f} market={lr.coef_[0][1]:+.3f}")
    res["B2 Benter(p_win+価格)"] = (log_loss(ev.won,p), roc_auc_score(ev.won,p), ev.assign(_p=p))
    # --- M1 GBM(全特徴+価格) ---
    import lightgbm as lgb
    trX = tr[feats].fillna(-9999).copy(); trX["mkt_logit"]=_logit(tr.q_mkt)
    evX = ev[feats].fillna(-9999).copy(); evX["mkt_logit"]=_logit(ev.q_mkt)
    gbm = lgb.train(dict(objective="binary",learning_rate=0.03,num_leaves=63,min_data_in_leaf=300,
                         feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,verbose=-1),
                    lgb.Dataset(trX, tr.won), num_boost_round=500)
    p = gbm.predict(evX)
    res["M1 GBM(全特徴+価格)"] = (log_loss(ev.won,p), roc_auc_score(ev.won,p), ev.assign(_p=p))
    # --- M1b GBM 特徴のみ(価格なし) : 特徴単独の力 ---
    gbm2 = lgb.train(dict(objective="binary",learning_rate=0.03,num_leaves=63,min_data_in_leaf=300,
                          feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,verbose=-1),
                     lgb.Dataset(tr[feats].fillna(-9999), tr.won), num_boost_round=500)
    p = gbm2.predict(ev[feats].fillna(-9999))
    res["M1b GBM(特徴のみ)"] = (log_loss(ev.won,p), roc_auc_score(ev.won,p), ev.assign(_p=p))
    # --- M2 NN/MLP(全特徴+価格) ---
    imp = SimpleImputer(strategy="median"); sc = StandardScaler()
    trM = sc.fit_transform(imp.fit_transform(trX)); evM = sc.transform(imp.transform(evX))
    mlp = MLPClassifier(hidden_layer_sizes=(128,64), alpha=1e-3, max_iter=60,
                        early_stopping=True, random_state=0).fit(trM, tr.won)
    p = mlp.predict_proba(evM)[:,1]
    res["M2 NN/MLP(全特徴+価格)"] = (log_loss(ev.won,p), roc_auc_score(ev.won,p), ev.assign(_p=p))

    print("\n================ OOS 2025: 予測力 (市場を超えるか) ================")
    print(f"  {'モデル':24s}{'logloss':>10s}{'vs市場':>9s}{'AUC':>8s}")
    print(f"  {'市場単独':24s}{base_ll:>10.4f}{'--':>9s}{roc_auc_score(ev.won,ev.q_mkt):>8.4f}")
    for k,(ll,auc,_) in res.items():
        print(f"  {k:24s}{ll:>10.4f}{(base_ll-ll):>+9.4f}{auc:>8.4f}")
    print("  (vs市場>0 = 市場より良い予測 = エッジの候補)")

    print("\n================ OOS 2025: 単勝 EV選抜 ROI ================")
    for k,(_,_,EVdf) in res.items():
        sw = roi_sweep(EVdf, "_p")
        cells=" ".join(f"EV≥{t}:n{n} ROI{roi*100:.0f}%[{lo*100:.0f},{hi*100:.0f}]" for (t,n,hit,roi,lo,hi) in sw)
        print(f"  {k:24s} {cells}")
