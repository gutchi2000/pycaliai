"""tenkai のペース区切り（cuts）の再計算と新旧比較（2026-11-19、分析のみ・本番経路に不干渉）。

背景: tenkai.py の cuts は「2022 年までの tenkai 自身のペース値を芝ダ別に三等分した点」のまま。
その物差しで見るとハイの割合が 2023 年 33% → 2026 年 24% と下がった（docs/research/TENKAI_ACCURACY_2026_20261118.md §3.1）。
ここでは、ペースモデル（pace_model.txt）と位置モデルは**そのまま**にして、確率への変換（cuts・σ・平均のずれ）だけを
最近のデータで作り直したときに、ペース 3 区分の当たり方がどう変わるかを、同じ正解で比べる。

入力（読むだけ）
  {root}/reports/speed_index/runs.parquet, {root}/reports/tenkai/{pos_model_e,pace_model}.txt, meta.json, day_*.json
  {root}/data/jrdb/raw/Sed/*.zip（正解 B/C、analysis.tenkai_accuracy_2026 の関数を流用）
出力
  --cache 以下に中間 parquet、--out に結果 JSON（標準出力にも要約）

使い方
  python -m analysis.tenkai_pace_cuts build --root E:/PyCaLiAI --cache <dir>     # 学習と同じ作り方で各レースの μ（数分）
  python -m analysis.tenkai_pace_cuts eval  --root E:/PyCaLiAI --cache <dir> --out <json>
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

PACE = ["スロー", "平均", "ハイ"]


# ---------------------------------------------------------------- build: 学習と同じ作り方で μ を出す
def cmd_build(a) -> None:
    import lightgbm as lgb
    import speed_index as si
    import tenkai as tk
    import shutil
    root = Path(a.root)
    cache = Path(a.cache)
    cache.mkdir(parents=True, exist_ok=True)
    # 通過順キャッシュの作り直しが本番の reports/tenkai に書かないよう、作業用の写しを使う
    work = cache / "tk"
    work.mkdir(exist_ok=True)
    for f in ("pos_model_e.txt", "pace_model.txt", "_corners_a_v1.parquet", "_corners_b_v1.parquet"):
        if (root / "reports/tenkai" / f).exists() and not (work / f).exists():
            shutil.copy2(root / "reports/tenkai" / f, work / f)
    tk.OUT_DIR = work
    tk.SI_DIR = si.OUT_DIR = root / "reports" / "speed_index"
    r = tk.runs_with_corners()
    x = tk.pos_features(r)
    full = x.groupby("race_key")["horse_id"].transform("size") >= np.maximum(5, 0.8 * x["field"])
    x = x[full & x["e_rel"].notna()].copy()
    me = lgb.Booster(model_file=str(tk.OUT_DIR / "pos_model_e.txt"))
    mp = lgb.Booster(model_file=str(tk.OUT_DIR / "pace_model.txt"))
    x["pe"] = me.predict(x[tk.POS_FEATS])
    races = x.groupby("race_key").agg(pace=("pace", "first"), date=("date", "first"), surface=("surface", "first"),
                                      dist=("dist", "first"), field=("field", "first"), klass=("klass", "first"),
                                      n=("horse_id", "size"))
    sub = x[x["date"] >= pd.Timestamp(a.mu_from)]
    P = tk.pace_frame(sub)
    races["mu"] = pd.Series(mp.predict(P[tk.PACE_FEATS]), index=P.index)
    races.to_parquet(cache / "races_train_style.parquet")
    print(f"races {len(races)} / mu {races['mu'].notna().sum()} → {cache / 'races_train_style.parquet'}")


# ---------------------------------------------------------------- 配信済み（2026）の μ を確率から逆算
def served_mu(root: Path, meta: dict) -> pd.DataFrame:
    cuts, sig = meta["cuts"], meta["sigma"]
    rows = []
    for p in sorted((root / "reports" / "tenkai").glob("day_*.json")):
        o = json.loads(p.read_text(encoding="utf-8"))
        for r in o["races"]:
            rows.append({"race_key": r["race_key"], "date": o["date"], "ps": r["pace"]["slow"],
                         "pm": r["pace"]["mid"], "ph": r["pace"]["high"]})
    D = pd.DataFrame(rows)
    return D


def invert_mu(D: pd.DataFrame, surf: pd.Series, cuts: dict, sig: float) -> np.ndarray:
    c1 = surf.map(lambda s: cuts[s][0]).to_numpy()
    c2 = surf.map(lambda s: cuts[s][1]).to_numpy()
    ps = np.clip(D["ps"].to_numpy(), 1e-4, 1 - 1e-4)
    ph = np.clip(D["ph"].to_numpy(), 1e-4, 1 - 1e-4)
    m1 = c1 - sig * norm.ppf(ps)
    m2 = c2 - sig * norm.ppf(1 - ph)
    w1 = np.abs(ps - 0.5) < np.abs(ph - 0.5)          # 0.5 に近い方（丸めの影響が小さい）を使う
    return np.where(w1, m1, m2)


# ---------------------------------------------------------------- 確率・指標
def probs(mu, surf, cuts, sig):
    """sig はスカラーか、各レースの配列。"""
    c1 = np.array([cuts[s][0] for s in surf])
    c2 = np.array([cuts[s][1] for s in surf])
    ps = norm.cdf((c1 - mu) / sig)
    ph = 1 - norm.cdf((c2 - mu) / sig)
    return np.column_stack([ps, 1 - ps - ph, ph])


def label(v, surf, cuts):
    c1 = np.array([cuts[s][0] for s in surf])
    c2 = np.array([cuts[s][1] for s in surf])
    v = np.asarray(v, dtype=float)
    out = np.where(v < c1, 0, np.where(v < c2, 1, 2)).astype(float)
    out[np.isnan(v)] = np.nan
    return out


def terciles(df, col="pace", qs=(1 / 3, 2 / 3)):
    return {s: [float(df.loc[df["surface"] == s, col].quantile(q)) for q in qs] for s in ("芝", "ダ")}


def wilson(k, n, z=1.96):
    if n == 0:
        return [np.nan, np.nan]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(c - h, 4), round(c + h, 4)]


def brier_rows(P, y):
    Y = np.eye(3)[y.astype(int)]
    return np.sum((P - Y) ** 2, axis=1)


def auc(y, s):
    y, s = np.asarray(y), np.asarray(s)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    rk = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((rk[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def block(P, y) -> dict:
    ok = ~np.isnan(y)
    P, y = P[ok], y[ok].astype(int)
    n = len(y)
    pred = P.argmax(1)
    hit = pred == y
    br = brier_rows(P, y)
    freq = np.bincount(y, minlength=3) / n
    ll = -np.mean(np.log(np.clip(P[np.arange(n), y], 1e-9, 1)))
    out = {"n": n, "acc": round(float(hit.mean()), 4), "acc_ci": wilson(int(hit.sum()), n),
           "brier": round(float(br.mean()), 4), "brier_unif": round(float(brier_rows(np.full_like(P, 1 / 3), y).mean()), 4),
           "logloss": round(float(ll), 4),
           "actual_dist": [round(float(v), 3) for v in freq],
           "pred_dist": [round(float(v), 3) for v in np.bincount(pred, minlength=3) / n],
           "mean_prob": [round(float(v), 3) for v in P.mean(0)],
           "big_miss": round(float((((pred == 0) & (y == 2)) | ((pred == 2) & (y == 0))).mean()), 4)}
    out["bss"] = round(1 - out["brier"] / out["brier_unif"], 4)
    # 確度の言葉（表示と同じ: 平均が最大 → はっきりしない、他は 50% 以上 → 公算が大きい、未満 → 寄り）
    top = P.max(1)
    words = {}
    for nm, m in (("スロー公算(≥50%)", (pred == 0) & (top >= .5)), ("スロー寄り", (pred == 0) & (top < .5)),
                  ("ハイ公算(≥50%)", (pred == 2) & (top >= .5)), ("ハイ寄り", (pred == 2) & (top < .5)),
                  ("ハイ≥55%", (pred == 2) & (top >= .55)), ("平均が最大", pred == 1)):
        k = int(m.sum())
        if k:
            words[nm] = {"n": k, "share": round(k / n, 3), "pred": round(float(top[m].mean()), 3),
                         "hit": round(float(hit[m].mean()), 3), "hit_ci": wilson(int(hit[m].sum()), k),
                         "actual": [round(float(v), 3) for v in np.bincount(y[m], minlength=3) / k]}
        else:
            words[nm] = {"n": 0}
    out["words"] = words
    # 「平均」の情報量
    pm = P[:, 1]
    out["mid"] = {"max": round(float(pm.max()), 3), "p95": round(float(np.percentile(pm, 95)), 3),
                  "sd": round(float(pm.std()), 4), "auc": round(auc(y == 1, pm), 3),
                  "auc_slow": round(auc(y == 0, P[:, 0]), 3), "auc_high": round(auc(y == 2, P[:, 2]), 3)}
    # 較正（帯ごと）
    cal = []
    for k, nm in enumerate(PACE):
        pc = P[:, k]
        for lo, hi in ((0, .25), (.25, .4), (.4, .55), (.55, 1.01)):
            mm = (pc >= lo) & (pc < hi)
            if mm.sum() >= 30:
                cal.append([nm, f"{int(lo*100)}-{min(int(hi*100),100)}", int(mm.sum()),
                            round(float(pc[mm].mean()), 3), round(float((y[mm] == k).mean()), 3)])
    out["calib"] = cal
    out["_br"] = br
    out["_hit"] = hit
    out["_ok"] = ok
    return out


def paired(b_new: dict, b_old: dict, rng, n_boot=2000) -> dict:
    """同じレース集合での new − old（Brier と的中）。レース単位のブートストラップ。"""
    d_br = b_new["_br"] - b_old["_br"]
    d_hit = b_new["_hit"].astype(float) - b_old["_hit"].astype(float)
    idx = np.arange(len(d_br))
    bs_b, bs_h = [], []
    for _ in range(n_boot):
        s = rng.choice(idx, len(idx))
        bs_b.append(d_br[s].mean())
        bs_h.append(d_hit[s].mean())
    return {"d_brier": round(float(d_br.mean()), 4), "d_brier_ci": [round(float(np.percentile(bs_b, q)), 4) for q in (2.5, 97.5)],
            "d_acc": round(float(d_hit.mean()), 4), "d_acc_ci": [round(float(np.percentile(bs_h, q)), 4) for q in (2.5, 97.5)]}


def strip(d):
    if isinstance(d, dict):
        return {k: strip(v) for k, v in d.items() if not k.startswith("_")}
    return d


def mid_cap(cuts, sig):
    """一様な σ のとき「平均」の確率がとれる最大値 = 2Φ(w/2σ) − 1（μ が区切りの真ん中のとき）。"""
    return {s: round(float(2 * norm.cdf((cuts[s][1] - cuts[s][0]) / (2 * sig)) - 1), 3) for s in cuts}


# ---------------------------------------------------------------- 変換の作り方（窓のデータから）
def fit_variants(W: pd.DataFrame, old: dict) -> dict:
    """W = 窓のレース（pace と mu あり）。返り値: 名前 → (cuts, sigma or per-surface dict, bias dict)。"""
    newc = terciles(W)
    res = W["pace"] - W["mu"]
    sig_all = float(res.std())
    sig_s = {s: float(res[W["surface"] == s].std()) for s in ("芝", "ダ")}
    bias = {s: float(res[W["surface"] == s].mean()) for s in ("芝", "ダ")}
    zero = {"芝": 0.0, "ダ": 0.0}
    return {
        "V0 旧（≤2022 三等分, σ=2023）": (old["cuts"], old["sigma"], zero),
        "V1 cuts だけ新": (newc, old["sigma"], zero),
        "V2 cuts+σ 新": (newc, sig_all, zero),
        "V3 cuts+σ+平均のずれ 新": (newc, sig_all, bias),
        "V4 V3+σ 芝ダ別": (newc, sig_s, bias),
    }, {"cuts": newc, "sigma": sig_all, "sigma_surface": sig_s, "bias": bias,
        "resid_mean_all": float(res.mean()), "n": int(len(W))}


def apply_variant(mu, surf, v):
    cuts, sig, bias = v
    mu2 = mu + np.array([bias[s] for s in surf])
    sg = np.array([sig[s] for s in surf]) if isinstance(sig, dict) else sig
    return probs(mu2, surf, cuts, sg)


def evaluate(E: pd.DataFrame, variants: dict, truths: dict, rng) -> dict:
    out = {}
    surf = E["surface"].to_numpy()
    mu = E["mu"].to_numpy()
    for tname, tcol in truths.items():
        y_all = E[tcol].to_numpy(dtype=float)
        ok = ~np.isnan(y_all)
        blocks = {}
        for vname, v in variants.items():
            P = apply_variant(mu[ok], surf[ok], v)
            blocks[vname] = block(P, y_all[ok])
        base = blocks[next(iter(variants))]
        for vname in list(blocks)[1:]:
            blocks[vname]["vs_V0"] = paired(blocks[vname], base, rng)
        out[tname] = strip(blocks)
    return out


def cmd_eval(a) -> None:
    from analysis.tenkai_accuracy_2026 import parse_raw, race_truth
    root, cache = Path(a.root), Path(a.cache)
    rng = np.random.default_rng(7)
    meta = json.loads((root / "reports/tenkai/meta.json").read_text(encoding="utf-8"))
    old = {"cuts": meta["cuts"], "sigma": meta["sigma"]}
    res = {"meta_version": meta["version"], "old": old}

    X = pd.read_parquet(cache / "races_train_style.parquet")
    X = X[X["pace"].notna() & (X["pace"] != 0)].copy()
    X["year"] = X["date"].dt.year

    # 正解 B/C（JRDB SED 2023〜）
    rc = cache / "truth_races.parquet"
    if rc.exists():
        R = pd.read_parquet(rc)
    else:
        S0, L = parse_raw(root / "data/jrdb/raw/Sed")
        R, _ = race_truth(S0, L)
        R.to_parquet(rc)
    R = R[["pace_b", "pace_b_cls", "pace_c", "pace_c_cls", "jrdb_cls", "N"]]

    # ---- 0. 再現確認（≤2022 の三等分が meta と同じか、2024-25 の的中が meta と同じか）
    chk_cuts = terciles(X[X["year"] <= 2022])
    T = X[X["year"].between(2024, 2025) & X["mu"].notna()]
    Pv = probs(T["mu"].to_numpy(), T["surface"].to_numpy(), old["cuts"], old["sigma"])
    yv = label(T["pace"], T["surface"].to_numpy(), old["cuts"])
    res["repro"] = {"cuts_le2022": chk_cuts, "meta_cuts": old["cuts"], "test_2024_25_n": int(len(T)),
                    "test_2024_25_acc": round(float((Pv.argmax(1) == yv).mean()), 4), "meta_pace_acc": meta["test"]["pace_acc"]}

    # ---- 1. 物差しの年次推移（tenkai 自身の値、旧 cuts で区分）と μ の推移
    yr = []
    for y, g in X[X["year"] >= 2013].groupby("year"):
        lab = label(g["pace"], g["surface"].to_numpy(), old["cuts"])
        row = {"year": int(y), "n": int(len(g)), "pace_mean": round(float(g["pace"].mean()), 2),
               "pace_median": round(float(g["pace"].median()), 2), "pace_sd": round(float(g["pace"].std()), 2),
               "act_dist_oldcuts": [round(float((lab == k).mean()), 3) for k in range(3)]}
        for s in ("芝", "ダ"):
            gs = g[g["surface"] == s]
            row[f"median_{s}"] = round(float(gs["pace"].median()), 2)
        gm = g[g["mu"].notna()]
        if len(gm):
            row["mu_mean"] = round(float(gm["mu"].mean()), 2)
            row["resid_mean"] = round(float((gm["pace"] - gm["mu"]).mean()), 2)
            row["resid_sd"] = round(float((gm["pace"] - gm["mu"]).std()), 2)
            row["spearman_mu"] = round(float(gm["mu"].corr(gm["pace"], method="spearman")), 3)
            P0 = probs(gm["mu"].to_numpy(), gm["surface"].to_numpy(), old["cuts"], old["sigma"])
            row["pred_dist_oldcuts"] = [round(float(v), 3) for v in np.bincount(P0.argmax(1), minlength=3) / len(gm)]
        yr.append(row)
    res["yearly"] = yr

    # ---- 2. 歩進検証 A: 窓 2023-24 で作る → 2025 で評価（tenkai 自身の値は全頭そろう）
    W = X[X["year"].between(2023, 2024) & X["mu"].notna()]
    variants, fit = fit_variants(W, old)
    E = X[(X["year"] == 2025) & X["mu"].notna()].join(R, how="left")
    s = E["surface"].to_numpy()
    E["a_old"] = label(E["pace"], s, old["cuts"])
    E["a_win"] = label(E["pace"], s, fit["cuts"])
    E["a_own"] = label(E["pace"], s, terciles(E))     # 2025 自身の三等分（後知恵の物差し）
    truths = {"A_旧区切り(≤2022)": "a_old", "A_窓区切り(2023-24)": "a_win", "A_2025自身の三等分": "a_own",
              "B_JRDB全頭再現(2023-25三等分)": "pace_b_cls", "C_ラップ": "pace_c_cls"}
    res["wf_2025"] = {"fit": fit, "mid_cap": {k: mid_cap(v[0], v[1] if not isinstance(v[1], dict) else min(v[1].values()))
                                              for k, v in variants.items()},
                      "eval": evaluate(E, variants, truths, rng)}

    # ---- 3. 前向き B: 窓 2023-25 で作る → 2026 配信済み（04-18〜11-14）で評価
    fits26 = {}
    for wname, wm in (("2023-25", X["year"].between(2023, 2025)), ("2024-25", X["year"].between(2024, 2025)),
                      ("2025", X["year"] == 2025)):
        fits26[wname] = fit_variants(X[wm & X["mu"].notna()], old)
    D = served_mu(root, meta)
    D = D.set_index("race_key").join(R, how="inner")
    D = D[D["N"].notna()]
    # 芝ダは SED から（R に surface を入れていないので runs / X から引く）
    rs = pd.read_parquet(root / "reports/speed_index/runs.parquet", columns=["race_key", "surface", "pace", "field"])
    rr = rs.groupby("race_key").agg(surface=("surface", "first"), pace=("pace", "first"), field=("field", "first"),
                                    nr=("pace", "size"))
    D = D.join(rr, how="left")
    sed_surf = pd.read_parquet(rc)["surface"] if "surface" in pd.read_parquet(rc).columns else None
    if sed_surf is not None:
        D["surface"] = D["surface"].fillna(sed_surf.reindex(D.index))
    D = D[D["surface"].isin(["芝", "ダ"])].copy()
    D["mu"] = invert_mu(D, D["surface"], old["cuts"], old["sigma"])
    back = probs(D["mu"].to_numpy(), D["surface"].to_numpy(), old["cuts"], old["sigma"])
    res["served_inversion_maxerr"] = round(float(np.abs(back - D[["ps", "pm", "ph"]].to_numpy()).max()), 4)
    D["cov"] = D["nr"] / D["field"]
    s = D["surface"].to_numpy()
    paceA = D["pace"].where((D["cov"] >= 0.6) & (D["pace"] != 0))
    D["a_old"] = label(paceA, s, old["cuts"])
    D["a_new"] = label(paceA, s, fits26["2023-25"][1]["cuts"])
    truths26 = {"B_JRDB全頭再現(2023-25三等分)": "pace_b_cls", "C_ラップ": "pace_c_cls",
                "A_旧区切り(≤2022, 復元6割以上)": "a_old", "A_新区切り(2023-25, 復元6割以上)": "a_new"}
    res["fwd_2026"] = {"races": int(len(D)), "dates": [D["date"].min(), D["date"].max()]}
    for wname, (vars_, fit_) in fits26.items():
        res["fwd_2026"][wname] = {"fit": fit_, "eval": evaluate(D, vars_, truths26 if wname == "2023-25" else
                                                              {k: v for k, v in truths26.items() if k.startswith("B")},
                                                              rng)}
    # 2026 の物差しの分布（A は復元 6 割以上だけ）
    res["fwd_2026"]["A_dist_oldcuts"] = [round(float((D["a_old"] == k).mean() / D["a_old"].notna().mean()), 3) for k in range(3)]
    res["fwd_2026"]["A_dist_newcuts"] = [round(float((D["a_new"] == k).mean() / D["a_new"].notna().mean()), 3) for k in range(3)]

    # ---- 4. 番手・届くか: cuts を使わないので変わらない（念のため、配信の位置予測と μ 以外を確認）
    res["note_pos"] = "位置（e/l）と届くか（late）は cuts を使わない（tenkai.score_days: pe/pl は pos_model、late は close_model に mu を入れる）"

    # ---- 5. 平均を広げる参考（定義を変える案）: 25/50/25 の区切りで、物差しも同じ区切りにしたとき
    W = X[X["year"].between(2023, 2025) & X["mu"].notna()]
    wide = terciles(W, qs=(0.25, 0.75))
    resd = W["pace"] - W["mu"]
    bias = {s_: float(resd[W["surface"] == s_].mean()) for s_ in ("芝", "ダ")}
    vw = {"W 25/50/25（2023-25）": (wide, float(resd.std()), bias)}
    Bq = pd.read_parquet(rc)
    Bq = Bq[Bq["date"] < "2026-01-01"]
    bw = {s_: Bq.loc[Bq["surface"] == s_, "pace_b"].quantile([.25, .75]).tolist() for s_ in ("芝", "ダ")}
    D["b_w"] = label(D["pace_b"], s, bw)
    D["a_w"] = label(paceA, s, wide)
    res["wide_band"] = {"cuts": wide, "mid_cap": mid_cap(wide, float(resd.std())),
                        "eval": evaluate(D, vw, {"B 25/50/25": "b_w", "A 25/50/25（復元6割以上）": "a_w"}, rng)}

    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"repro": res["repro"], "inv_err": res["served_inversion_maxerr"]}, ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "eval"])
    ap.add_argument("--root", default="E:/PyCaLiAI")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--mu-from", default="2013-01-01")
    a = ap.parse_args()
    (cmd_build if a.cmd == "build" else cmd_eval)(a)


if __name__ == "__main__":
    main()
