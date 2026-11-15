"""展開予想 (2026-11-16)。

各馬の「序盤の位置（最初のコーナー）」と「4 角の位置」をレース内の相対位置（0 = 先頭, 1 = 最後方）で予測し、
そこからレースのペース（スロー / 平均 / ハイ）を確率で出す。stride の「AI 展開シミュレーター」を参考に、
仕組みは自前（過去走の通過順・前半の速さ・枠・頭数・距離。オッズ・JRDB は使わない）。

入力
  reports/speed_index/runs.parquet  … speed_index.py build が作る全走（前半の点数 f_pts、レースのペース pace、4 角 c4）
  通過順 1〜3 角 … 結果 CSV（enhanced / TARGET）と 2026 年の週次出走表の「前1角〜前3角」
出力
  reports/tenkai/  (.gitignore)  pos_model_e.txt / pos_model_l.txt / pace_model.txt / meta.json / day_{date}.json
  site/data/tenkai_meta.json     … 解説用の検証値（build_site とサイトが読む）

使い方
  python tenkai.py train            # 学習 〜2022 / 確認 2023 / 検証 2024-25（数分）
  python tenkai.py score-day 20261115
  python tenkai.py score-days 20260418 20261115
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

import speed_index as si

BASE = Path(__file__).resolve().parent
OUT_DIR = BASE / "reports" / "tenkai"
SI_DIR = si.OUT_DIR
META_SITE = BASE / "site" / "data" / "tenkai_meta.json"
CORNER_VERSION = 1
log = logging.getLogger("tenkai")

PLACES = ["札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"]
PLACE_N = {p: i for i, p in enumerate(PLACES)}


# ---------------------------------------------------------------- 通過順の読み込み
def _raw_weekly(path: Path) -> pd.DataFrame:
    """週次出走表 CSV を、調教 JOIN などをせずに行だけ読む（weekly_parse.parse_csv の行分けと同じ）。"""
    import weekly_parse as wp
    for enc in ("cp932", "shift_jis", "utf-8"):
        try:
            text = path.read_bytes().decode(enc)
            break
        except Exception:      # noqa: BLE001
            continue
    widths = {33: "HORSE_COLS_33", 44: "HORSE_COLS_44", 46: "HORSE_COLS_46", 48: "HORSE_COLS_48",
              49: "HORSE_COLS_49", 99: "HORSE_COLS_99"}
    rows, cur = [], None
    for line in text.splitlines():
        cols = line.split(",")
        if cols[0] in ("レースID(新)", "枠番", "番", ""):
            continue
        if len(cols) == 19:
            cur = dict(zip(wp.RACE_COLS, cols))
        elif cur and len(cols) in widths and hasattr(wp, widths[len(cols)]):
            h = dict(zip(getattr(wp, widths[len(cols)]), cols))
            h.update(cur)
            rows.append(h)
    return pd.DataFrame(rows).rename(columns=wp.COLUMN_MAP)


def load_corners() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(A) 日付・場・R・馬番 → 1〜3 角、(B) 日付・馬名 → 1〜3 角（2026 年の週次出走表の前走欄）。"""
    cache_a = OUT_DIR / f"_corners_a_v{CORNER_VERSION}.parquet"
    cache_b = OUT_DIR / f"_corners_b_v{CORNER_VERSION}.parquet"
    wfiles = si._weekly_files()
    newest = max([si.SRC_ENH.stat().st_mtime] + [p.stat().st_mtime for p in si.SRC_TARGET if p.exists()]
                 + [p.stat().st_mtime for p in wfiles])
    if cache_a.exists() and cache_b.exists() and cache_a.stat().st_mtime > newest:
        return pd.read_parquet(cache_a), pd.read_parquet(cache_b)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log.info("通過順を読み込み中（enhanced）…")
    d = pd.read_csv(si.SRC_ENH, encoding="cp932", low_memory=False,
                    usecols=["年", "月", "日", "場所", "レース番号", "馬番", "通過順1", "通過順2", "通過順3"])
    d = d[d["年"] >= si.FIRST_YEAR - 2000]
    A = [pd.DataFrame({
        "date": pd.to_datetime(dict(year=d["年"] + 2000, month=d["月"], day=d["日"])),
        "place": d["場所"].astype(str).str.strip(), "rno": d["レース番号"].astype(int),
        "umaban": d["馬番"].astype(int),
        "c1": pd.to_numeric(d["通過順1"], errors="coerce"), "c2": pd.to_numeric(d["通過順2"], errors="coerce"),
        "c3": pd.to_numeric(d["通過順3"], errors="coerce")})]
    for p in si.SRC_TARGET:
        if not p.exists():
            continue
        t = pd.read_csv(p, encoding="cp932", low_memory=False, dtype=str)
        t.columns = [c.strip() for c in t.columns]
        rid = t["レースID(旧)"].astype(str).str.zfill(10)
        A.append(pd.DataFrame({
            "date": pd.to_datetime("20" + t["日付"].astype(str).str.zfill(6), format="%Y%m%d", errors="coerce"),
            "place": t["場所"].astype(str).str.strip(), "rno": pd.to_numeric(rid.str[6:8], errors="coerce"),
            "umaban": pd.to_numeric(t["馬番"], errors="coerce"),
            "c1": pd.to_numeric(t["1角"], errors="coerce"), "c2": pd.to_numeric(t["2角"], errors="coerce"),
            "c3": pd.to_numeric(t["3角"], errors="coerce")}))
    A = pd.concat(A, ignore_index=True).dropna(subset=["date", "rno", "umaban"])
    A["rno"] = A["rno"].astype(int)
    A["umaban"] = A["umaban"].astype(int)
    A = A.drop_duplicates(["date", "place", "rno", "umaban"], keep="last")
    log.info("通過順を読み込み中（週次出走表 %d 件）…", len(wfiles))
    B = []
    for p in wfiles:
        try:
            w = _raw_weekly(p)
        except Exception as e:      # noqa: BLE001
            log.warning("weekly %s 読めない: %s", p.name, e)
            continue
        if w.empty or "前走月" not in w:
            continue
        dt = pd.to_datetime(p.stem, format="%Y%m%d")
        mo = pd.to_numeric(w["前走月"], errors="coerce")
        dy = pd.to_numeric(w["前走日"], errors="coerce")
        yr = np.where(mo > dt.month, dt.year - 1, dt.year)
        pdt = pd.to_datetime(dict(year=yr, month=mo.fillna(1), day=dy.fillna(1)), errors="coerce").where(mo.notna())
        B.append(pd.DataFrame({
            "date": pdt, "horse": w["馬名"].astype(str).str.strip(),
            "c1": pd.to_numeric(w.get("前1角"), errors="coerce"), "c2": pd.to_numeric(w.get("前2角"), errors="coerce"),
            "c3": pd.to_numeric(w.get("前3角"), errors="coerce")}))
    B = pd.concat(B, ignore_index=True) if B else pd.DataFrame(columns=["date", "horse", "c1", "c2", "c3"])
    B = B[B["date"].notna()].drop_duplicates(["date", "horse"], keep="first")
    A.to_parquet(cache_a)
    B.to_parquet(cache_b)
    return A, B


def runs_with_corners() -> pd.DataFrame:
    runs = pd.read_parquet(SI_DIR / "runs.parquet")
    A, B = load_corners()
    r = runs.merge(A, on=["date", "place", "rno", "umaban"], how="left")
    miss = r["c1"].isna() & r["c2"].isna() & r["c3"].isna()
    if miss.any():
        fill = r.loc[miss, ["date", "horse"]].merge(B, on=["date", "horse"], how="left")
        for c in ("c1", "c2", "c3"):
            r.loc[miss, c] = fill[c].to_numpy()
    # 結果 CSV は「そのコースに無いコーナー」を 0 で持つ（1200m の 1・2 角など）。0 は欠損として扱う。
    for c in ("c1", "c2", "c3", "c4"):
        r[c] = r[c].where(r[c] > 0)
    # 最初のコーナー（1 角が無い距離は 2 角、3 角の順）。無ければ 4 角。
    early = r["c1"].fillna(r["c2"]).fillna(r["c3"]).fillna(r["c4"])
    den = np.maximum(r["field"] - 1, 1)
    r["e_rel"] = ((early - 1) / den).clip(0, 1)
    r["l_rel"] = ((r["c4"] - 1) / den).clip(0, 1)
    r["f_rate"] = r["f_pts"] / np.maximum(r["dist"] - 600, 200) * 1000
    return r


# ---------------------------------------------------------------- 特徴
POS_FEATS = ["e1", "e2", "e3", "e_mean5", "e_min5", "e_std5", "e_same", "e_simdist", "l1", "l_mean3",
             "f1", "f_mean3", "f_max3", "n_hist", "days", "dist_ratio", "surf_chg",
             "umaban_rel", "field", "dist", "surface_n", "place_n", "cls", "age", "sex_f", "weight",
             "r_e_mean5_rk", "r_e_mean5_dev", "r_e1_rk", "r_f_mean3_rk", "r_f_mean3_dev", "r_n_front"]
PACE_FEATS = ["field", "dist", "surface_n", "place_n", "cls", "n_front", "n_front2", "fr_top1", "fr_top2",
              "fr_top3", "fr_mean", "fr_std", "pe_min", "pe_n02", "hist_frac", "e_mean_front"]


def pos_features(x: pd.DataFrame) -> pd.DataFrame:
    """各走について、それより前の走（通過順の分かる走）だけから作る特徴。"""
    x = x.sort_values(["horse_id", "date", "race_key"]).copy()
    is_hist = x["e_rel"].notna().to_numpy()
    hid = x["horse_id"].to_numpy()
    h = x[is_hist].reset_index(drop=True)
    start = pd.Series(np.arange(len(h)), index=h["horse_id"]).groupby(level=0).min()
    csum = pd.Series(is_hist.astype(int), index=x.index).groupby(hid).cumsum().to_numpy()
    pos = csum - is_hist.astype(int)
    st = pd.Series(hid).map(start).to_numpy()
    lag = {}
    for k in range(1, 6):
        ok = (~pd.isna(st)) & (pos - k >= 0)
        take = np.where(ok, st + pos - k, 0).astype(int)
        s = h.iloc[take].set_index(x.index)
        lag[k] = s.where(pd.Series(ok, index=x.index), np.nan)
    E = np.column_stack([lag[k]["e_rel"] for k in range(1, 6)])
    L = np.column_stack([lag[k]["l_rel"] for k in range(1, 4)])
    F = np.column_stack([lag[k]["f_rate"] for k in range(1, 4)])
    S = np.column_stack([(lag[k]["surface"] == x["surface"]).to_numpy() for k in range(1, 6)])
    D = np.column_stack([(lag[k]["dist"] - x["dist"]).abs().le(200).to_numpy() for k in range(1, 6)])
    with np.errstate(all="ignore"):
        x["e1"], x["e2"], x["e3"] = E[:, 0], E[:, 1], E[:, 2]
        x["e_mean5"] = np.nanmean(E, axis=1)
        x["e_min5"] = np.nanmin(E, axis=1)
        x["e_std5"] = np.nanstd(E, axis=1)
        x["e_same"] = np.nanmean(np.where(S, E, np.nan), axis=1)
        x["e_simdist"] = np.nanmean(np.where(S & D, E, np.nan), axis=1)
        x["l1"] = L[:, 0]
        x["l_mean3"] = np.nanmean(L, axis=1)
        x["f1"] = F[:, 0]
        x["f_mean3"] = np.nanmean(F, axis=1)
        x["f_max3"] = np.nanmax(F, axis=1)
    x["n_hist"] = pos
    x["days"] = (x["date"] - lag[1]["date"]).dt.days
    x["dist_ratio"] = x["dist"] / lag[1]["dist"]
    x["surf_chg"] = (x["surface"] != lag[1]["surface"]).astype(float).where(lag[1]["e_rel"].notna())
    x["umaban_rel"] = (x["umaban"] - 1) / np.maximum(x["field"] - 1, 1)
    x["surface_n"] = (x["surface"] == "芝").astype(int)
    x["place_n"] = x["place"].map(PLACE_N).fillna(-1)
    x["cls"] = x["klass"].map(si.CLASS_RANK)
    x["sex_f"] = (x["sex"] == "牝").astype(int)
    # レースの中での相対（序盤の位置は相手次第）
    g = x.groupby("race_key")
    n = g["horse_id"].transform("size")
    x["r_e_mean5_rk"] = g["e_mean5"].rank(method="average") / n
    x["r_e_mean5_dev"] = x["e_mean5"] - g["e_mean5"].transform("median")
    x["r_e1_rk"] = g["e1"].rank(method="average") / n
    x["r_f_mean3_rk"] = g["f_mean3"].rank(method="average", ascending=False) / n
    x["r_f_mean3_dev"] = x["f_mean3"] - g["f_mean3"].transform("median")
    x["r_n_front"] = g["e_mean5"].transform(lambda s: (s < 0.2).sum())
    return x


def pace_frame(x: pd.DataFrame) -> pd.DataFrame:
    """レース単位の特徴（予測した序盤位置 pe と過去の前半の速さから）。"""
    def agg(g):
        fr = g["f_mean3"].dropna().sort_values(ascending=False)
        front = g.nsmallest(3, "pe")
        return pd.Series({
            "field": g["field"].iloc[0], "dist": g["dist"].iloc[0], "surface_n": g["surface_n"].iloc[0],
            "place_n": g["place_n"].iloc[0], "cls": g["cls"].iloc[0],
            "n_front": float((g["e_mean5"] < 0.15).sum()), "n_front2": float((g["e_min5"] < 0.05).sum()),
            "fr_top1": fr.iloc[0] if len(fr) > 0 else np.nan, "fr_top2": fr.iloc[1] if len(fr) > 1 else np.nan,
            "fr_top3": fr.iloc[2] if len(fr) > 2 else np.nan,
            "fr_mean": fr.mean() if len(fr) else np.nan, "fr_std": fr.std() if len(fr) > 1 else np.nan,
            "pe_min": g["pe"].min(), "pe_n02": float((g["pe"] < 0.2).sum()),
            "hist_frac": g["e1"].notna().mean(), "e_mean_front": front["f_mean3"].mean(),
        })
    return x.groupby("race_key").apply(agg, include_groups=False)


# ---------------------------------------------------------------- 学習・検証
def _spearman_in_race(df, a, b):
    cs = []
    for _, g in df.groupby("race_key"):
        g = g[[a, b]].dropna()
        if len(g) >= 5:
            c = g[a].rank().corr(g[b].rank())
            if not math.isnan(c):
                cs.append(c)
    return float(np.mean(cs)) if cs else float("nan"), len(cs)


def _style_rank(q):
    """レース内の順位（0〜1、小さいほど前）から予想脚質。逃げ = 先頭 1 割、先行 〜35%、差し 〜70%、残りは追込。"""
    q = np.asarray(q, dtype=float)
    return np.select([q <= 0.1, q <= 0.35, q <= 0.70], ["逃げ", "先行", "差し"], "追込")


def cmd_train(args) -> None:
    import lightgbm as lgb
    r = runs_with_corners()
    log.info("全走 %d、通過順あり %.1f%%", len(r), 100 * r["e_rel"].notna().mean())
    x = pos_features(r)
    # ほぼ頭数分そろったレース（取消・競走中止・タイム不明で数頭欠けるのは許す）
    full = x.groupby("race_key")["horse_id"].transform("size") >= np.maximum(5, 0.8 * x["field"])
    x = x[full & x["e_rel"].notna()].copy()
    yr = x["date"].dt.year
    tr, va, te = yr <= 2022, yr == 2023, yr.between(2024, 2025)
    params = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=7)
    models = {}
    for tgt in ("e_rel", "l_rel"):
        m = x[tgt].notna()
        dtr = lgb.Dataset(x.loc[tr & m, POS_FEATS], x.loc[tr & m, tgt])
        dva = lgb.Dataset(x.loc[va & m, POS_FEATS], x.loc[va & m, tgt])
        b = lgb.train(params, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
        models[tgt] = b
        b.save_model(str(OUT_DIR / f"pos_model_{tgt[0]}.txt"))
        log.info("%s: best_iter %d", tgt, b.best_iteration)
    x["pe"] = models["e_rel"].predict(x[POS_FEATS])
    x["pl"] = models["l_rel"].predict(x[POS_FEATS])

    # 位置の検証（2024-25）。比べる相手は「前 5 走の平均位置」（過去走の無い馬は真ん中）。
    T = x[te].assign(b=x.loc[te, "e_mean5"].fillna(0.5))
    allh = T[T.groupby("race_key")["e_mean5"].transform(lambda s: s.notna().all())]   # 全馬に過去走があるレース
    sp_e, n_e = _spearman_in_race(allh, "pe", "e_rel")
    sp_base, _ = _spearman_in_race(allh, "b", "e_rel")
    sp_l, _ = _spearman_in_race(allh, "pl", "l_rel")
    n_all = int(T["race_key"].nunique())

    def _lead(col):
        L = T.loc[T.groupby("race_key")[col].idxmin()]
        top3 = (L["e_rel"] * np.maximum(L["field"] - 1, 1) + 1) <= 3 + 1e-9
        return float((L["e_rel"] <= 1e-9).mean()), float(top3.mean())
    lead_hit, lead_top3 = _lead("pe")
    base_hit, base_top3 = _lead("b")
    rk = T.groupby("race_key")["pe"].rank(method="first") / T.groupby("race_key")["pe"].transform("size")
    ak = T.groupby("race_key")["e_rel"].rank(method="first") / T.groupby("race_key")["e_rel"].transform("size")
    style_acc = float((_style_rank(rk) == _style_rank(ak)).mean())
    log.info("位置 2024-25: 序盤 Spearman %.3f（前5走の平均だけ %.3f）、4角 %.3f、逃げ的中 %.1f%%（前5走平均だけ %.1f%%）",
             sp_e, sp_base, sp_l, 100 * lead_hit, 100 * base_hit)

    # ペース
    P = pace_frame(x)
    pace = x.groupby("race_key").agg(pace=("pace", "first"), date=("date", "first"), surface=("surface", "first"))
    P = P.join(pace)
    P = P[P["pace"].notna() & (P["pace"] != 0)]
    py = P["date"].dt.year
    ptr, pva, pte = py <= 2022, py == 2023, py.between(2024, 2025)
    cuts = {s: [float(P.loc[ptr & (P["surface"] == s), "pace"].quantile(q)) for q in (1 / 3, 2 / 3)]
            for s in ("芝", "ダ")}
    pp = dict(objective="regression", learning_rate=0.03, num_leaves=31, min_data_in_leaf=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=7)
    pm = lgb.train(pp, lgb.Dataset(P.loc[ptr, PACE_FEATS], P.loc[ptr, "pace"]), 3000,
                   valid_sets=[lgb.Dataset(P.loc[pva, PACE_FEATS], P.loc[pva, "pace"])],
                   callbacks=[lgb.early_stopping(100, verbose=False)])
    pm.save_model(str(OUT_DIR / "pace_model.txt"))
    P["mu"] = pm.predict(P[PACE_FEATS])
    sigma = float((P.loc[pva, "pace"] - P.loc[pva, "mu"]).std())
    probs = pace_probs(P["mu"].to_numpy(), P["surface"].to_numpy(), cuts, sigma)
    P["p_slow"], P["p_mid"], P["p_high"] = probs[:, 0], probs[:, 1], probs[:, 2]
    lab = lambda v, s: np.where(v < np.array([cuts[t][0] for t in s]), 0, np.where(v < np.array([cuts[t][1] for t in s]), 1, 2))
    P["a_cls"] = lab(P["pace"].to_numpy(), P["surface"].to_numpy())
    P["p_cls"] = probs.argmax(1)
    Q = P[pte]
    acc = float((Q["a_cls"] == Q["p_cls"]).mean())
    corr = float(Q["mu"].corr(Q["pace"], method="spearman"))
    # 確率の当たり具合（予測した確率の帯ごとの実際の割合）
    conf = []
    for k, nm in enumerate(("スロー", "平均", "ハイ")):
        pc = Q[["p_slow", "p_mid", "p_high"]].to_numpy()[:, k]
        hit = (Q["a_cls"] == k).to_numpy()
        for lo, hi in ((0, .25), (.25, .4), (.4, .55), (.55, 1.01)):
            mm = (pc >= lo) & (pc < hi)
            if mm.sum() >= 50:
                conf.append({"cls": nm, "band": f"{int(lo*100)}〜{min(int(hi*100),100)}%", "n": int(mm.sum()),
                             "pred": round(float(pc[mm].mean()), 3), "actual": round(float(hit[mm].mean()), 3)})
    hi_sel = Q[Q["p_cls"] == 2]
    sl_sel = Q[Q["p_cls"] == 0]
    log.info("ペース 2024-25: 3 区分の的中 %.1f%%（でたらめ 33%%）、順位相関 %.3f、σ=%.3f", 100 * acc, corr, sigma)

    meta = {
        "version": "tk-20261116", "sigma": sigma, "cuts": cuts,
        "test": {"period": "2024-01〜2025-12", "races": n_all, "races_allhist": int(n_e),
                 "pos_spearman": round(sp_e, 3), "pos_spearman_base": round(sp_base, 3), "l_spearman": round(sp_l, 3),
                 "lead_hit": round(lead_hit, 3), "lead_top3": round(lead_top3, 3),
                 "lead_hit_base": round(base_hit, 3), "lead_top3_base": round(base_top3, 3),
                 "style_acc": round(style_acc, 3), "pace_races": int(len(Q)), "pace_acc": round(acc, 3),
                 "pace_corr": round(corr, 3),
                 "pace_when_high": {"n": int(len(hi_sel)), "actual_high": round(float((hi_sel["a_cls"] == 2).mean()), 3),
                                    "actual_slow": round(float((hi_sel["a_cls"] == 0).mean()), 3)},
                 "pace_when_slow": {"n": int(len(sl_sel)), "actual_slow": round(float((sl_sel["a_cls"] == 0).mean()), 3),
                                    "actual_high": round(float((sl_sel["a_cls"] == 2).mean()), 3)},
                 "pace_calib": conf},
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    META_SITE.write_bytes((json.dumps(meta, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))
    print(json.dumps(meta["test"], ensure_ascii=False, indent=1))


def pace_probs(mu, surf, cuts, sigma):
    from math import erf, sqrt
    Phi = np.vectorize(lambda z: 0.5 * (1 + erf(z / sqrt(2))))
    c1 = np.array([cuts[s][0] for s in surf])
    c2 = np.array([cuts[s][1] for s in surf])
    ps = Phi((c1 - mu) / sigma)
    ph = 1 - Phi((c2 - mu) / sigma)
    return np.column_stack([ps, 1 - ps - ph, ph])


# ---------------------------------------------------------------- 当日
def score_days(dates: list[str]) -> dict[str, dict]:
    import lightgbm as lgb
    meta = json.loads((OUT_DIR / "meta.json").read_text(encoding="utf-8"))
    me = lgb.Booster(model_file=str(OUT_DIR / "pos_model_e.txt"))
    ml = lgb.Booster(model_file=str(OUT_DIR / "pos_model_l.txt"))
    mp = lgb.Booster(model_file=str(OUT_DIR / "pace_model.txt"))
    r = runs_with_corners()
    dates = sorted(dates)
    out = {}
    ids = r.sort_values("date").drop_duplicates("horse", keep="last").set_index("horse")["horse_id"]
    for d in dates:
        E = si.day_entries(d)
        E["horse_id"] = E["horse"].map(ids).fillna("N:" + E["horse"])
        for c in ("e_rel", "l_rel", "f_rate"):
            E[c] = np.nan
        dt = pd.Timestamp(d)
        base = r[r["date"] < dt]
        allx = pos_features(pd.concat([base[base["horse_id"].isin(E["horse_id"])], E.assign(_today=True)],
                                      ignore_index=True))
        t = allx[allx["_today"].fillna(False).astype(bool)].copy()
        t["pe"] = me.predict(t[POS_FEATS])
        t["pl"] = ml.predict(t[POS_FEATS])
        P = pace_frame(t)
        P["mu"] = mp.predict(P[PACE_FEATS])
        surf = t.groupby("race_key")["surface"].first().reindex(P.index).to_numpy()
        pr = pace_probs(P["mu"].to_numpy(), surf, meta["cuts"], meta["sigma"])
        races = []
        for (rk, g) in t.sort_values(["place", "rno", "umaban"]).groupby("race_key", sort=False):
            i = P.index.get_loc(rk)
            g = g.assign(e_rank=g["pe"].rank(method="first").astype(int), l_rank=g["pl"].rank(method="first").astype(int))
            g = g.assign(st=_style_rank(g["e_rank"] / len(g)))
            hs = [{"umaban": int(h.umaban), "e": round(float(h.pe), 3), "l": round(float(h.pl), 3),
                   "er": int(h.e_rank), "lr": int(h.l_rank), "st": str(h.st),
                   "nh": int(h.n_hist)} for h in g.itertuples()]
            races.append({"race_key": rk, "place": g["place"].iloc[0], "rno": int(g["rno"].iloc[0]),
                          "pace": {"slow": round(float(pr[i, 0]), 3), "mid": round(float(pr[i, 1]), 3),
                                   "high": round(float(pr[i, 2]), 3)},
                          "horses": hs})
        out[d] = {"date": d, "version": meta["version"], "races": races}
    return out


def cmd_score(args) -> None:
    lo, hi = args.start, args.end or args.start
    ds = [f.stem for f in si._weekly_files() if lo <= f.stem <= hi]
    res = score_days(ds)
    for d, o in res.items():
        (OUT_DIR / f"day_{d}.json").write_text(json.dumps(o, ensure_ascii=False), encoding="utf-8")
    log.info("tenkai %d 日 → %s", len(res), OUT_DIR)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser(description="展開予想（序盤・4角の位置とペース）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("train").set_defaults(fn=cmd_train)
    s = sub.add_parser("score-day")
    s.add_argument("start")
    s.set_defaults(fn=cmd_score, end=None)
    s = sub.add_parser("score-days")
    s.add_argument("start")
    s.add_argument("end", nargs="?")
    s.set_defaults(fn=cmd_score)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    args.fn(args)


if __name__ == "__main__":
    main()
