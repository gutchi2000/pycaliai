# -*- coding: utf-8 -*-
"""
exp_deepvalue_t10.py — T-10 Deep Value Net (棄権学習つき ◎単勝 意思決定ネット)
================================================================================
目標(事前登録): OOS(test=2024-25)で「選別◎単勝」ROI の bootstrap CI 下限 > 1.00。
  ※ in-sample や点推定の 100%超えは合格でない。

設計:
  - 候補 = 各レースの v6較正PL単勝確率 top-1 (◎)。ネットは「賭ける/見送る」を学習。
  - serve情報 = 当日9時オッズのみ (o9)。決済 = 確定オッズ (oc)。leak規約は stage1 と同一。
  - 損失 = 直接効用: -E[ b * (win*oc - 1) ] + λ_bet * E[b]   (期待利益最大化 + 賭けコスト
    → 棄権デフォルト)。変種で log 効用も。
  - 防御: (a) オッズ入力へのガウスノイズ注入 (見かけの妙味を罰する)
          (b) edge 特徴のシュリンケージ (c) валid=2023 profit で early stop
          (d) 判定は test bootstrap CI 下限のみ。
分割: Train ≤2022 / Valid 2023 / Test 2024-25 (プロジェクト標準)。

使い方:
  python exp_deepvalue_t10.py build      # 基質キャッシュ生成 (stage1のbuild_races流用, 重い)
  python exp_deepvalue_t10.py baseline   # ◎単勝フラット買いの床を実測
  python exp_deepvalue_t10.py train      # ネット学習 + OOS判定
"""
from __future__ import annotations
import json, pickle, sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).parent
CACHE = BASE / "data" / "deepvalue_races.pkl"
OUT = BASE / "reports" / "deepvalue_t10.json"
SEED = 42


# ---------------------------------------------------------------- substrate
def cmd_build():
    from stage1_benter_blend import build_races
    races = build_races()
    slim = []
    for r in races:
        slim.append({
            "logf": np.asarray(r["logf"], dtype=np.float32),
            "logp9": np.asarray(r["logp9"], dtype=np.float32),
            "o9": np.asarray(r["o9"], dtype=np.float32),
            "oc": np.asarray(r["oc"], dtype=np.float32),
            "win": int(r["win"]), "year": int(r["year"]),
        })
    CACHE.parent.mkdir(exist_ok=True)
    with open(CACHE, "wb") as f:
        pickle.dump(slim, f)
    print(f"[build] cached {len(slim):,} races -> {CACHE}")


CACHE2 = BASE / "data" / "deepvalue_races_oof.pkl"


def cmd_build2():
    """基質 v2: ≤2023 は OOF スコア (expanding-window, as-of) + OOF較正チェーン、
    2024-25 は本番 v6 + 本番 calibrator (=本番サービングと同一)。in-sample 毒抜き版。"""
    import joblib
    import pandas as pd
    from sklearn.isotonic import IsotonicRegression
    import pl_probs as PL
    from stage1_benter_blend import load_tanpuk_snapshots, devig
    from joint_calibration_v6 import apply_encoders, COL_RID, COL_JYUN, COL_BAN, MASTER_CSV

    snaps = load_tanpuk_snapshots()
    print(f"  snapshots: {len(snaps):,}")
    oof = pd.read_parquet(BASE / "data/oof_scores_v6params.parquet")
    oof_map = {}
    for rid, g in oof.groupby("rid", sort=False):
        oof_map[str(rid)] = dict(zip(g["ban"].astype(int), g["score"].astype(float)))
    print(f"  OOF races: {len(oof_map):,} ({oof['year'].min()}-{oof['year'].max()})")

    b = joblib.load(BASE / "models/unified_rank_v6.pkl")
    model, feats, encs = b["model"], b["feature_cols"], b["encoders"]
    cal = joblib.load(BASE / "models/pl_calibrators_v6.pkl")
    cal = cal.get("calibrators", cal)
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df = df.dropna(subset=[COL_JYUN, COL_RID]).copy()
    df["year"] = df[COL_RID].astype(str).str[:4].astype(int)
    enc = apply_encoders(df, encs)
    m24 = enc["year"] >= 2024
    X24 = enc.loc[m24, feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    enc.loc[m24, "_score_v6"] = model.predict(X24)
    print("  v6 scored 2024-25")

    # レース組み立て (年昇順 → OOF較正チェーン)
    rows = []
    for rid, g in df.groupby(COL_RID, sort=False):
        if len(g) < 5:
            continue
        rid16 = str(rid)[:16]
        if rid16 not in snaps:
            continue
        y = int(rid16[:4])
        g = g.sort_values(COL_BAN)
        ban = g[COL_BAN].astype(int).values
        jyun = pd.to_numeric(g[COL_JYUN], errors="coerce").values
        if y >= 2024:
            sc = enc.loc[g.index, "_score_v6"].values
        else:
            om = oof_map.get(rid16) or oof_map.get(str(rid))
            if om is None:
                continue
            sc = np.array([om.get(int(bn), np.nan) for bn in ban])
            if np.isnan(sc).any():
                continue
        rows.append((rid16, y, ban, sc, jyun))
    rows.sort(key=lambda r: r[0])
    print(f"  assembled: {len(rows):,} races")

    # OOF較正チェーン: iso は「その年より前の OOF (praw, win)」だけで fit (自年 leak なし)
    slim, hist_p, hist_w = [], [], []
    iso, iso_year = None, -1
    for rid16, y, ban, sc, jyun in rows:
        w = PL.pl_weights(sc)
        praw = np.clip(PL.all_tansho(w), 1e-9, 1 - 1e-9)
        if y >= 2024:
            f_all = cal["tansho"].predict(praw) if "tansho" in cal else praw
        else:
            if y != iso_year:
                if len(hist_p) > 5000:
                    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6)
                    iso.fit(np.array(hist_p), np.array(hist_w))
                iso_year = y
            f_all = iso.predict(praw) if iso is not None else praw
        f_all = np.clip(f_all, 1e-9, None)
        sp = snaps[rid16]
        win_ban = int(ban[int(np.nanargmin(jyun))])
        bl = list(ban.astype(int))
        f_by = dict(zip(bl, f_all))
        praw_by = dict(zip(bl, praw))
        common = [k for k in sp["o9"] if k in sp["oc"] and k in f_by]
        if win_ban not in common or len(common) < 5:
            continue
        keys = sorted(common)
        f = np.array([f_by[k] for k in keys])
        p9 = devig(sp["o9"], keys)
        slim.append({
            "logf": np.log(f).astype(np.float32),
            "logp9": np.log(p9).astype(np.float32),
            "o9": np.array([sp["o9"][k] for k in keys], dtype=np.float32),
            "oc": np.array([sp["oc"][k] for k in keys], dtype=np.float32),
            "win": keys.index(win_ban), "year": y,
        })
        if y < 2024:
            for k in bl:
                hist_p.append(float(praw_by[k]))
                hist_w.append(1.0 if k == win_ban else 0.0)
    with open(CACHE2, "wb") as fh:
        pickle.dump(slim, fh)
    yrs = np.array([r["year"] for r in slim])
    print(f"[build2] cached {len(slim):,} races -> {CACHE2}  "
          f"(oof<=2023: {(yrs <= 2023).sum():,} / v6 2024-25: {(yrs >= 2024).sum():,})")


def load_races(oof=False):
    with open(CACHE2 if oof else CACHE, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------- features
def race_features(r):
    """◎(=argmax logf) 1点の意思決定に使う race-level 特徴 (serve=9時のみ)。"""
    logf, logp9, o9 = r["logf"], r["logp9"], r["o9"]
    n = len(logf)
    i = int(np.argmax(logf))                      # ◎
    f = np.exp(logf); p9 = np.exp(logp9)
    ordf = np.argsort(-logf)
    gap12_f = float(logf[ordf[0]] - logf[ordf[1]])
    ordp = np.argsort(-logp9)
    gap12_p = float(logp9[ordp[0]] - logp9[ordp[1]])
    mkt_rank = int(np.where(ordp == i)[0][0]) + 1  # ◎の市場人気順位
    ent_p = float(-(p9 * logp9).sum())
    ent_f = float(-(f * logf).sum())
    edge = float(logf[i] - logp9[i])               # ★核: モデル-市場の対数乖離
    x = np.array([
        logf[i], logp9[i], edge,
        np.log(o9[i]),
        gap12_f, gap12_p,
        ent_p, ent_f, ent_f - ent_p,
        np.log(n),
        mkt_rank,
        float(f[i]), float(p9[i]), float(f[i] - p9[i]),
    ], dtype=np.float32)
    return x, i


FEAT_NAMES = ["logf", "logp9", "edge", "log_o9", "gap12_f", "gap12_p",
              "ent_p", "ent_f", "ent_diff", "log_n", "mkt_rank",
              "f", "p9", "f_minus_p9"]


def build_matrix(races):
    X, prof, oc9, win, year = [], [], [], [], []
    for r in races:
        x, i = race_features(r)
        X.append(x)
        w = 1.0 if i == r["win"] else 0.0
        prof.append(w * r["oc"][i] - 1.0)          # 100円あたり利益(決済=確定)
        oc9.append(r["o9"][i])
        win.append(w)
        year.append(r["year"])
    return (np.stack(X), np.array(prof, dtype=np.float32),
            np.array(oc9, dtype=np.float32), np.array(win, dtype=np.float32),
            np.array(year))


def splits(year):
    return year <= 2022, year == 2023, year >= 2024


def roi_ci(prof, mask, n_boot=4000, seed=SEED):
    """選択集合の ROI = (Σ(prof)+n)/n = 平均回収率。bootstrap CI。"""
    p = prof[mask]
    n = len(p)
    if n == 0:
        return {"roi": None, "n": 0}
    rng = np.random.default_rng(seed)
    rois = [(p[rng.integers(0, n, n)] + 1.0).mean() for _ in range(n_boot)]
    return {"roi": round(float((p + 1.0).mean()), 4), "n": int(n),
            "hit": round(float((p > 0).mean()), 4),
            "ci95": [round(float(np.percentile(rois, 2.5)), 4),
                     round(float(np.percentile(rois, 97.5)), 4)]}


# ---------------------------------------------------------------- baseline
def cmd_baseline(oof=False):
    races = load_races(oof)
    X, prof, o9, win, year = build_matrix(races)
    tr, va, te = splits(year)
    out = {"flat_all": {}, "by_year_test": {}, "by_mkt_rank_test": {}}
    for name, m in [("train<=2022", tr), ("valid2023", va), ("test2024-25", te)]:
        out["flat_all"][name] = roi_ci(prof, m)
        print(f"◎単勝フラット {name}: {out['flat_all'][name]}")
    for y in [2024, 2025]:
        out["by_year_test"][y] = roi_ci(prof, year == y)
    mkt_rank = X[:, FEAT_NAMES.index("mkt_rank")]
    for r_ in [1, 2, 3]:
        out["by_mkt_rank_test"][f"rank{r_}"] = roi_ci(prof, te & (mkt_rank == r_))
        print(f"  test ◎が市場{r_}番人気: {out['by_mkt_rank_test'][f'rank{r_}']}")
    out["by_mkt_rank_test"]["rank4+"] = roi_ci(prof, te & (mkt_rank >= 4))
    print(f"  test ◎が市場4番人気以下: {out['by_mkt_rank_test']['rank4+']}")
    OUT.parent.mkdir(exist_ok=True)
    prev = json.load(open(OUT, encoding="utf-8")) if OUT.exists() else {}
    prev["baseline"] = out
    json.dump(prev, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[saved] {OUT}")


# ---------------------------------------------------------------- net
def cmd_train(cfg_override=None, oof=False):
    import torch
    import torch.nn as nn
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device={dev}")

    cfg = {
        "hidden": [64, 32], "dropout": 0.2,
        "lr": 1e-3, "epochs": 300, "batch": 4096,
        "lam_bet": 0.05,          # 賭けコスト(棄権デフォルト圧)
        "odds_noise": 0.10,       # log-odds 入力ノイズ σ
        "edge_shrink": 0.7,       # edge 特徴のシュリンケージ
        "patience": 30,
    }
    if cfg_override:
        cfg.update(cfg_override)

    races = load_races(oof)
    X, prof, o9, win, year = build_matrix(races)
    tr, va, te = splits(year)

    # edge シュリンケージ (核特徴を素で信じない)
    ei = FEAT_NAMES.index("edge")
    X[:, ei] *= cfg["edge_shrink"]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
    Xn = (X - mu) / sd

    def t(a):
        return torch.tensor(a, device=dev)

    Xt, pt = t(Xn), t(prof)
    noise_cols = [FEAT_NAMES.index(c) for c in ("logp9", "log_o9", "edge", "f_minus_p9")]

    layers, d = [], X.shape[1]
    for h in cfg["hidden"]:
        layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(cfg["dropout"])]
        d = h
    layers += [nn.Linear(d, 1)]
    net = nn.Sequential(*layers).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])

    idx_tr = np.where(tr)[0]
    # 棄権全振り(利益0)を正当なベースラインとする: それを上回る選別だけ採用
    best_va, best_state, bad = 0.0, [p.detach().clone() for p in net.parameters()], 0
    for ep in range(cfg["epochs"]):
        net.train()
        perm = np.random.permutation(idx_tr)
        for s in range(0, len(perm), cfg["batch"]):
            bi = perm[s:s + cfg["batch"]]
            xb = Xt[bi].clone()
            nz = torch.randn(len(bi), len(noise_cols), device=dev) * cfg["odds_noise"]
            xb[:, noise_cols] += nz
            b = torch.sigmoid(net(xb).squeeze(-1))
            loss = -(b * pt[bi]).mean() + cfg["lam_bet"] * b.mean()
            opt.zero_grad(); loss.backward(); opt.step()
        # valid: 実利益 (hard 決定 b>0.5)
        net.eval()
        with torch.no_grad():
            bv = torch.sigmoid(net(Xt).squeeze(-1)).cpu().numpy()
        sel_va = va & (bv > 0.5)
        pv = float(prof[sel_va].sum()) if sel_va.sum() > 0 else 0.0
        if pv > best_va:
            best_va, best_state, bad = pv, [p.detach().clone() for p in net.parameters()], 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    for p, s in zip(net.parameters(), best_state):
        p.data.copy_(s)
    net.eval()
    with torch.no_grad():
        b = torch.sigmoid(net(Xt).squeeze(-1)).cpu().numpy()

    res = {"cfg": cfg, "epochs_ran": ep + 1, "best_valid_profit": round(best_va, 1)}
    for name, m in [("train<=2022", tr), ("valid2023", va), ("test2024-25", te)]:
        sel = m & (b > 0.5)
        r = roi_ci(prof, sel)
        r["bet_frac"] = round(float(sel.sum() / max(m.sum(), 1)), 4)
        res[name] = r
        print(f"  {name}: {r}")
    # 診断: b の順位づけは情報を持ってるか (b上位q%のROI, valid/test)
    res["b_topq"] = {}
    for name, m in [("valid2023", va), ("test2024-25", te)]:
        rows = {}
        bm = b[m]; pm = prof[m]
        for q in [0.05, 0.10, 0.20, 0.50]:
            k = max(int(len(bm) * q), 1)
            idx = np.argsort(-bm)[:k]
            rows[f"top{int(q*100)}%"] = {"roi": round(float((pm[idx] + 1).mean()), 4), "n": k}
        res["b_topq"][name] = rows
        print(f"  b分位 {name}: {rows}")
    ok = (res["test2024-25"].get("ci95") or [0])[0] > 1.0
    res["gate_pass_ci_low_gt_1"] = bool(ok)
    print(f"  ★事前登録ゲート(test CI下限>1.00): {'PASS' if ok else 'FAIL'}")

    prev = json.load(open(OUT, encoding="utf-8")) if OUT.exists() else {}
    prev.setdefault("runs", []).append(res)
    json.dump(prev, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[saved] {OUT}")
    return res


# ---------------------------------------------------------------- all-horse net
def build_matrix_all(races, odds_cap=50.0):
    """全馬候補の per-horse 行列。serve=9時のみ。"""
    X, prof, year, rid_i = [], [], [], []
    for ri, r in enumerate(races):
        logf, logp9, o9 = r["logf"], r["logp9"], r["o9"]
        n = len(logf)
        f = np.exp(logf); p9 = np.exp(logp9)
        ordf = np.argsort(-logf); ordp = np.argsort(-logp9)
        rank_f = np.empty(n, dtype=np.float32); rank_f[ordf] = np.arange(1, n + 1)
        rank_p = np.empty(n, dtype=np.float32); rank_p[ordp] = np.arange(1, n + 1)
        ent_p = float(-(p9 * logp9).sum())
        best_f = float(logf.max())
        for i in range(n):
            if o9[i] > odds_cap:
                continue
            X.append([logf[i], logp9[i], logf[i] - logp9[i], np.log(o9[i]),
                      rank_f[i], rank_p[i], rank_p[i] - rank_f[i],
                      best_f - logf[i], ent_p, np.log(n),
                      f[i], p9[i], f[i] - p9[i]])
            prof.append((1.0 if i == r["win"] else 0.0) * r["oc"][i] - 1.0)
            year.append(r["year"]); rid_i.append(ri)
    return (np.array(X, dtype=np.float32), np.array(prof, dtype=np.float32),
            np.array(year), np.array(rid_i))


ALL_FEATS = ["logf", "logp9", "edge", "log_o9", "rank_f", "rank_p", "rank_gap",
             "gap_to_top", "ent_p", "log_n", "f", "p9", "f_minus_p9"]


def _train_one_rank(Xn, prof, tr, cfg, seed):
    """収束まで学習して b を返す (選別は外側で quantile 方式)。early stop は train loss."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed); np.random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xt = torch.tensor(Xn, device=dev); pt = torch.tensor(prof, device=dev)
    noise_cols = [ALL_FEATS.index(c) for c in ("logp9", "log_o9", "edge", "f_minus_p9")]
    layers, d = [], Xn.shape[1]
    for h in cfg["hidden"]:
        layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(cfg["dropout"])]
        d = h
    layers += [nn.Linear(d, 1)]
    net = nn.Sequential(*layers).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    idx_tr = np.where(tr)[0]
    best_tl, bad = np.inf, 0
    for ep in range(cfg["epochs"]):
        net.train()
        perm = np.random.permutation(idx_tr)
        tl = 0.0; nb = 0
        for s in range(0, len(perm), cfg["batch"]):
            bi = perm[s:s + cfg["batch"]]
            xb = Xt[bi].clone()
            xb[:, noise_cols] += torch.randn(len(bi), len(noise_cols), device=dev) * cfg["odds_noise"]
            b = torch.sigmoid(net(xb).squeeze(-1))
            loss = -(b * pt[bi]).mean() + cfg["lam_bet"] * b.mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tl += float(loss) * len(bi); nb += len(bi)
        tl /= nb
        if tl < best_tl - 1e-5:
            best_tl, bad = tl, 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    net.eval()
    import torch as _t
    with _t.no_grad():
        b = _t.sigmoid(net(Xt).squeeze(-1)).cpu().numpy()
    return b


def cmd_sweep2():
    """quantile 選別版スイープ。(cfg, q) の選択= valid CI下限のみ。test は最良1点を1回開封。"""
    races = load_races(oof=True)
    X, prof, year, rid_i = build_matrix_all(races)
    tr, va, te = splits(year)
    print(f"[sweep2] rows={len(X):,} (tr={tr.sum():,} va={va.sum():,} te={te.sum():,})")
    ei = ALL_FEATS.index("edge")
    base = {"hidden": [64, 32], "dropout": 0.2, "lr": 1e-3, "epochs": 80,
            "batch": 8192, "patience": 15}
    grid = []
    for noise in [0.05, 0.15]:
        for shrink in [0.5, 1.0]:
            for lam in [0.0, 0.05]:
                grid.append(dict(base, odds_noise=noise, edge_shrink=shrink, lam_bet=lam))
    QS = [0.005, 0.01, 0.02, 0.05, 0.10]
    results = []
    for gi, cfg in enumerate(grid):
        Xs = X.copy(); Xs[:, ei] *= cfg["edge_shrink"]
        mu, sd = Xs[tr].mean(0), Xs[tr].std(0) + 1e-8
        Xn = (Xs - mu) / sd
        b = np.mean([_train_one_rank(Xn, prof, tr, cfg, sd_) for sd_ in [0, 1, 2]], axis=0)
        row = {"cfg": {k: cfg[k] for k in ("odds_noise", "edge_shrink", "lam_bet")},
               "valid_q": {}, "_b": b}
        bv, pv = b[va], prof[va]
        ordv = np.argsort(-bv)
        for q in QS:
            k = max(int(len(bv) * q), 20)
            sel = ordv[:k]
            r = roi_ci_arr(pv[sel])
            row["valid_q"][str(q)] = r
        results.append(row)
        print(f"  [{gi+1}/{len(grid)}] {row['cfg']} valid: " +
              " ".join(f"q{q}={row['valid_q'][str(q)]['roi']:.3f}[{row['valid_q'][str(q)]['ci95'][0]:.2f}]"
                       for q in QS), flush=True)
    # 選択 = valid CI下限最大の (cfg, q)
    best, bq, blow = None, None, -np.inf
    for row in results:
        for q, r in row["valid_q"].items():
            if r["ci95"][0] > blow:
                best, bq, blow = row, q, r["ci95"][0]
    b = best.pop("_b")
    for r in results:
        r.pop("_b", None)
    bt, pt_ = b[te], prof[te]
    k = max(int(len(bt) * float(bq)), 20)
    sel = np.argsort(-bt)[:k]
    test = roi_ci_arr(pt_[sel])
    ok = test["ci95"][0] > 1.0
    print(f"\n★best cfg={best['cfg']} q={bq} (valid CI低={blow:.3f})")
    print(f"★test(single unveil): {test}")
    print(f"★事前登録ゲート(test CI下限>1.00): {'PASS' if ok else 'FAIL'}")
    prev = json.load(open(OUT, encoding="utf-8")) if OUT.exists() else {}
    prev["sweep2"] = {"grid": results, "best_cfg": best["cfg"], "best_q": bq,
                      "test_unveil": test, "gate_pass": bool(ok)}
    json.dump(prev, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[saved] {OUT}")


def roi_ci_arr(p, n_boot=2000, seed=SEED):
    n = len(p)
    if n == 0:
        return {"roi": None, "n": 0, "ci95": [0, 0]}
    rng = np.random.default_rng(seed)
    rois = [(p[rng.integers(0, n, n)] + 1.0).mean() for _ in range(n_boot)]
    return {"roi": round(float((p + 1.0).mean()), 4), "n": int(n),
            "hit": round(float((p > 0).mean()), 4),
            "ci95": [round(float(np.percentile(rois, 2.5)), 4),
                     round(float(np.percentile(rois, 97.5)), 4)]}


def _train_one(Xn, prof, tr, va, te, cfg, seed):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed); np.random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xt = torch.tensor(Xn, device=dev); pt = torch.tensor(prof, device=dev)
    noise_cols = [ALL_FEATS.index(c) for c in ("logp9", "log_o9", "edge", "f_minus_p9")]
    layers, d = [], Xn.shape[1]
    for h in cfg["hidden"]:
        layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(cfg["dropout"])]
        d = h
    layers += [nn.Linear(d, 1)]
    net = nn.Sequential(*layers).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    idx_tr = np.where(tr)[0]
    best_va, best_state, bad = 0.0, [p.detach().clone() for p in net.parameters()], 0
    for ep in range(cfg["epochs"]):
        net.train()
        perm = np.random.permutation(idx_tr)
        for s in range(0, len(perm), cfg["batch"]):
            bi = perm[s:s + cfg["batch"]]
            xb = Xt[bi].clone()
            xb[:, noise_cols] += torch.randn(len(bi), len(noise_cols), device=dev) * cfg["odds_noise"]
            b = torch.sigmoid(net(xb).squeeze(-1))
            if cfg["loss"] == "logu":
                gain = torch.clamp(b * cfg["stake"] * pt[bi], min=-0.95)
                loss = -torch.log1p(gain).mean() + cfg["lam_bet"] * b.mean()
            else:
                loss = -(b * pt[bi]).mean() + cfg["lam_bet"] * b.mean()
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            bv = torch.sigmoid(net(Xt).squeeze(-1)).cpu().numpy()
        sel = va & (bv > 0.5)
        pv = float(prof[sel].sum()) if sel.sum() > 0 else 0.0
        if pv > best_va:
            best_va, best_state, bad = pv, [p.detach().clone() for p in net.parameters()], 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    for p, s in zip(net.parameters(), best_state):
        p.data.copy_(s)
    net.eval()
    import torch as _t
    with _t.no_grad():
        b = _t.sigmoid(net(Xt).squeeze(-1)).cpu().numpy()
    return b, best_va


def cmd_sweep_all():
    """全馬候補ネットの設定スイープ。選択=validのみ、testは最良1設定だけ最後に開封。"""
    races = load_races(oof=True)
    X, prof, year, rid_i = build_matrix_all(races)
    tr, va, te = splits(year)
    print(f"[sweep_all] rows={len(X):,} (tr={tr.sum():,} va={va.sum():,} te={te.sum():,})")
    ei = ALL_FEATS.index("edge")
    base = {"hidden": [64, 32], "dropout": 0.2, "lr": 1e-3, "epochs": 200,
            "batch": 8192, "patience": 25, "stake": 0.05}
    grid = []
    for lam in [0.02, 0.05, 0.10]:
        for noise in [0.05, 0.15]:
            for shrink in [0.5, 0.8]:
                for loss in ["profit", "logu"]:
                    grid.append(dict(base, lam_bet=lam, odds_noise=noise,
                                     edge_shrink=shrink, loss=loss))
    results = []
    for gi, cfg in enumerate(grid):
        Xs = X.copy(); Xs[:, ei] *= cfg["edge_shrink"]
        mu, sd = Xs[tr].mean(0), Xs[tr].std(0) + 1e-8
        Xn = (Xs - mu) / sd
        bs = []
        for seed in [0, 1, 2]:
            b, bv = _train_one(Xn, prof, tr, va, te, cfg, seed)
            bs.append(b)
        b = np.mean(bs, axis=0)          # seed アンサンブル
        sel_va = va & (b > 0.5)
        va_profit = float(prof[sel_va].sum()) if sel_va.sum() else 0.0
        va_roi = roi_ci(prof, sel_va, n_boot=500)
        results.append({"cfg": {k: cfg[k] for k in ("lam_bet", "odds_noise", "edge_shrink", "loss")},
                        "valid_profit": round(va_profit, 1), "valid": va_roi, "_b": b})
        print(f"  [{gi+1}/{len(grid)}] {results[-1]['cfg']} -> valid_profit={va_profit:.1f} "
              f"roi={va_roi.get('roi')} n={va_roi.get('n')}", flush=True)
    # ★ valid_profit 最良の1設定のみ test 開封
    best = max(results, key=lambda r: r["valid_profit"])
    b = best.pop("_b")
    for r in results:
        r.pop("_b", None)
    sel_te = te & (b > 0.5)
    test = roi_ci(prof, sel_te)
    ok = (test.get("ci95") or [0])[0] > 1.0
    print(f"\n★best cfg={best['cfg']} valid_profit={best['valid_profit']}")
    print(f"★test(single unveil): {test}")
    print(f"★事前登録ゲート(test CI下限>1.00): {'PASS' if ok else 'FAIL'}")
    prev = json.load(open(OUT, encoding="utf-8")) if OUT.exists() else {}
    prev["sweep_all"] = {"grid": results, "best": best, "test_unveil": test,
                         "gate_pass": bool(ok)}
    json.dump(prev, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    {"build": cmd_build, "build2": cmd_build2,
     "baseline": cmd_baseline, "train": cmd_train,
     "baseline2": lambda: cmd_baseline(oof=True),
     "train2": lambda: cmd_train(oof=True),
     "sweep_all": cmd_sweep_all, "sweep2": cmd_sweep2}[cmd]()
