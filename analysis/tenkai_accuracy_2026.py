"""tenkai（展開予想）の 2026 年前向き精度の検証（2026-11-18、分析のみ・本番経路に不干渉）。

入力（すべて保存済み、読むだけ）
  reports/tenkai/day_{date}.json   … サイト・文章に出した tenkai の出力（2026-04-18〜11-15）
  reports/tenkai/meta.json / course_late_rate.json
  data/jrdb/raw/Sed/*.zip           … JRDB 成績 SED（全頭の着順・走破・上がり3F・1〜4角）と SRB（ラップ）
  reports/speed_index/runs.parquet  … tenkai 自身のペース定義の値（2026-02-23 以降は再出走馬だけの復元＝欠けあり）
  site/data/{date}.json             … クラス・コース・頭数
出力
  標準出力の表 + --out の JSON

使い方
  python -m analysis.tenkai_accuracy_2026 --root E:/PyCaLiAI --out <path.json>
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

PLACE = {"01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京", "06": "中山", "07": "中京",
         "08": "京都", "09": "阪神", "10": "小倉"}
PACE = ["スロー", "平均", "ハイ"]
STYLE = ["逃げ", "先行", "差し", "追込"]


# ---------------------------------------------------------------- JRDB SED / SRB
def _num(b: bytes):
    s = b.decode("ascii", "ignore").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def parse_raw(sed_dir: Path, years=("23", "24", "25", "26")):
    sed, srb = [], []
    for zp in sorted(glob.glob(str(sed_dir / "*.zip"))):
        with zipfile.ZipFile(zp) as z:
            for name in z.namelist():
                up = name.upper()
                if not up.endswith(".TXT") or up[3:5] not in years:
                    continue
                raw = z.read(name)
                if up.startswith("SED"):
                    for ln in raw.split(b"\n"):
                        ln = ln.rstrip(b"\r")
                        if len(ln) != 374:
                            continue
                        t = ln[143:147].decode("ascii", "ignore").strip()
                        tm = (int(t[0]) * 60 + int(t[1:]) / 10) if t.isdigit() and len(t) == 4 and int(t) > 0 else np.nan
                        sed.append((ln[0:8].decode(), int(_num(ln[8:10])), ln[18:26].decode(), _num(ln[62:66]),
                                    _num(ln[66:67]), _num(ln[130:132]), _num(ln[140:142]), _num(ln[142:143]), tm,
                                    _num(ln[261:264]) / 10, _num(ln[308:310]), _num(ln[310:312]), _num(ln[312:314]),
                                    _num(ln[314:316]), ln[221:222].decode("ascii", "ignore").strip(),
                                    ln[10:18].decode("ascii", "ignore").strip()))
                elif up.startswith("SRB"):
                    for ln in raw.split(b"\n"):
                        ln = ln.rstrip(b"\r")
                        if len(ln) != 850:
                            continue
                        laps = [_num(ln[8 + 3 * i:11 + 3 * i]) / 10 for i in range(18)]
                        laps = [x for x in laps if x == x and x > 0]
                        srb.append((ln[0:8].decode(), laps))
    S = pd.DataFrame(sed, columns=["rk", "umaban", "ymd", "dist", "td", "tosu", "finish", "ijou", "time", "l3",
                                   "c1", "c2", "c3", "c4", "jrdb_pace", "ped"])
    S = S.drop_duplicates(["rk", "umaban"], keep="last")
    L = pd.DataFrame(srb, columns=["rk", "laps"]).drop_duplicates("rk", keep="last")
    S["place"] = S["rk"].str[:2].map(PLACE)
    S["rno"] = S["rk"].str[6:8].astype(int)
    S["date"] = pd.to_datetime(S["ymd"], format="%Y%m%d", errors="coerce")
    S["surface"] = S["td"].map({1.0: "芝", 2.0: "ダ"})
    S = S[S["surface"].notna() & S["place"].notna() & S["date"].notna()].copy()
    S["race_key"] = S["ymd"] + S["place"] + S["rno"].astype(str).str.zfill(2)
    return S, L


def race_truth(S: pd.DataFrame, L: pd.DataFrame) -> pd.DataFrame:
    """レース単位の正解: 走った頭数、ペース B（馬の 前半/上がり の速さの差）、ペース C（ラップの前後半差）、JRDB の H/M/S、
    勝ち馬の 4 角位置。"""
    S = S.copy()
    S["ran"] = ~S["ijou"].isin([1, 2])                     # 取消・除外を除く
    S["N"] = S.groupby("race_key")["ran"].transform("sum")
    v = S["ran"] & (S["ijou"] == 0) & (S["finish"] > 0) & (S["time"] > 0) & S["l3"].between(30, 45) & (S["dist"] > 600)
    S["valid"] = v
    S["nv"] = S.groupby("race_key")["valid"].transform("sum")
    S["half"] = v & (S["finish"] <= np.maximum(3, np.ceil(S["nv"] / 2)))
    S["f"] = (S["time"] - S["l3"]) / (S["dist"] - 600)
    S["a"] = S["l3"] / 600
    S["ck"] = S["place"] + S["surface"] + S["dist"].astype(int).astype(str)
    hist = S[S["half"] & (S["date"] < "2026-01-01")]
    std = hist.groupby("ck").agg(f_std=("f", "median"), a_std=("a", "median"), n_std=("f", "size"),
                                 races=("race_key", "nunique"))
    std = std[std["races"] >= 15]
    S = S.join(std[["f_std", "a_std"]], on="ck")
    S["pb"] = 100 * ((S["f_std"] - S["f"]) / S["f_std"] - (S["a_std"] - S["a"]) / S["a_std"])
    R = S.groupby("race_key").agg(date=("date", "first"), place=("place", "first"), rno=("rno", "first"),
                                  surface=("surface", "first"), dist=("dist", "first"), ck=("ck", "first"),
                                  N=("N", "first"), jrdb_pace=("jrdb_pace", "first"), rk=("rk", "first"))
    R["pace_b"] = S[S["half"]].groupby("race_key")["pb"].median()
    R["n_half"] = S[S["half"]].groupby("race_key").size()
    # 勝ち馬の 4 角（同着は先頭の 1 頭）
    w = S[(S["finish"] == 1) & S["ran"] & (S["c4"] > 0)].sort_values("c4").drop_duplicates("race_key")
    R["win_c4"] = w.set_index("race_key")["c4"]
    R["late_y"] = (((R["win_c4"] - 1) / np.maximum(R["N"] - 1, 1)) > 0.5).astype(float).where(R["win_c4"].notna())
    # ラップ: 前半 600m の時刻（最初のラップは端数）と最後の 3 ラップ
    lap = L.set_index("rk")["laps"]
    R["laps"] = R["rk"].map(lap)

    def _fb(row):
        laps, d = row["laps"], row["dist"]
        if not isinstance(laps, list) or len(laps) < 4:
            return np.nan
        first = d - 200 * (len(laps) - 1)
        if not (0 < first <= 200):
            return np.nan
        seg = [first] + [200] * (len(laps) - 1)
        cum_d, cum_t, t600 = 0.0, 0.0, np.nan
        for s, t in zip(seg, laps):
            if cum_d + s >= 600:
                t600 = cum_t + t * (600 - cum_d) / s
                break
            cum_d += s
            cum_t += t
        return sum(laps[-3:]) - t600
    R["lap_diff"] = R.apply(_fb, axis=1)          # 後半 3F − 前半 3F（秒）。大きい = 前が速く後ろが遅い = ハイ
    old = R[R["date"] < "2026-01-01"]
    lstd = old.groupby("ck")["lap_diff"].agg(["median", "size"])
    lstd = lstd[lstd["size"] >= 15]["median"]
    R["pace_c"] = R["lap_diff"] - R["ck"].map(lstd)
    # 3 区分（芝ダ別、2023-25 の三等分）
    old = R[R["date"] < "2026-01-01"]
    for col in ("pace_b", "pace_c"):
        cuts = {s: old.loc[old["surface"] == s, col].quantile([1 / 3, 2 / 3]).tolist() for s in ("芝", "ダ")}
        c1 = R["surface"].map(lambda s: cuts[s][0])
        c2 = R["surface"].map(lambda s: cuts[s][1])
        R[col + "_cls"] = np.where(R[col].isna(), np.nan, np.where(R[col] < c1, 0, np.where(R[col] < c2, 1, 2)))
    R["jrdb_cls"] = R["jrdb_pace"].map({"S": 0, "M": 1, "H": 2})
    return R.drop(columns=["laps"]), S


# ---------------------------------------------------------------- tenkai の出力
def load_tenkai(tk_dir: Path, site_dir: Path, start: str, end: str):
    races, horses = [], []
    for p in sorted(tk_dir.glob("day_*.json")):
        d = p.stem[4:]
        if not (start <= d <= end):
            continue
        o = json.loads(p.read_text(encoding="utf-8"))
        meta = {}
        sp = site_dir / f"{d}.json"
        if sp.exists():
            sj = json.loads(sp.read_text(encoding="utf-8"))
            for r in sj.get("races", []):
                meta[(r.get("place"), int(r.get("rno")))] = r
        for r in o["races"]:
            m = meta.get((r["place"], int(r["rno"])), {})
            pc = r["pace"]
            races.append({"race_key": r["race_key"], "date": d, "place": r["place"], "rno": int(r["rno"]),
                          "p_slow": pc["slow"], "p_mid": pc["mid"], "p_high": pc["high"], "late_p": r.get("late"),
                          "klass": m.get("klass"), "course": m.get("course"), "site_field": m.get("field_size"),
                          "n_tk": len(r["horses"]), "nh0": float(np.mean([h["nh"] == 0 for h in r["horses"]]))})
            for h in r["horses"]:
                horses.append({"race_key": r["race_key"], "umaban": h["umaban"], "pe": h["e"], "pl": h["l"],
                               "nh": h["nh"]})
    return pd.DataFrame(races), pd.DataFrame(horses)


# ---------------------------------------------------------------- 指標
def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def brier3(P, y):
    Y = np.eye(3)[y.astype(int)]
    return float(np.mean(np.sum((P - Y) ** 2, axis=1)))


def auc(y, s):
    y, s = np.asarray(y), np.asarray(s)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    r = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def pace_block(T: pd.DataFrame, truth: str, boot=0, rng=None) -> dict:
    t = T[T[truth].notna()]
    if len(t) == 0:
        return {"n": 0}
    P = t[["p_slow", "p_mid", "p_high"]].to_numpy()
    y = t[truth].to_numpy().astype(int)
    pred = P.argmax(1)
    hit = (pred == y)
    freq = np.bincount(y, minlength=3) / len(y)
    out = {"n": int(len(t)), "acc": float(hit.mean()), "acc_ci": wilson(int(hit.sum()), len(t)),
           "brier": brier3(P, y), "brier_unif": brier3(np.full_like(P, 1 / 3), y),
           "brier_freq": brier3(np.tile(freq, (len(y), 1)), y),
           "acc_majority": float(freq.max()),
           "actual_dist": [round(float(x), 3) for x in freq],
           "pred_dist": [round(float(x), 3) for x in np.bincount(pred, minlength=3) / len(pred)],
           "big_miss": float(((pred == 0) & (y == 2) | (pred == 2) & (y == 0)).mean()),
           "confusion": pd.crosstab(pd.Series(pred, name="pred"), pd.Series(y, name="act")).reindex(
               index=[0, 1, 2], columns=[0, 1, 2], fill_value=0).to_numpy().tolist()}
    out["bss_unif"] = 1 - out["brier"] / out["brier_unif"]
    if boot:
        bs = []
        idx = np.arange(len(t))
        for _ in range(boot):
            s = rng.choice(idx, len(idx))
            bs.append(brier3(P[s], y[s]) - brier3(np.full_like(P[s], 1 / 3), y[s]))
        out["brier_minus_unif_ci"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    return out


def pos_block(H: pd.DataFrame, pred: str, act: str) -> dict:
    h = H[H[pred].notna() & H[act].notna()]
    cs, mae, maen, n = [], [], [], 0
    for _, g in h.groupby("race_key"):
        if len(g) < 5:
            continue
        pr = g[pred].rank(method="first")
        ar = g[act].rank(method="average")
        c = pr.corr(ar)
        if c == c:
            cs.append(c)
        mae.append(float((pr - g[act]).abs().mean()))
        maen.append(float((pr - g[act]).abs().mean() / max(len(g) - 1, 1)))
        n += 1
    cs = np.array(cs)
    return {"races": n, "spearman": float(cs.mean()) if len(cs) else np.nan,
            "spearman_se": float(cs.std(ddof=1) / math.sqrt(len(cs))) if len(cs) > 1 else np.nan,
            "mae": float(np.mean(mae)) if mae else np.nan, "mae_rel": float(np.mean(maen)) if maen else np.nan}


def style_of(q):
    q = np.asarray(q, dtype=float)
    return np.select([q <= 0.1, q <= 0.35, q <= 0.70], [0, 1, 2], 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="E:/PyCaLiAI")
    ap.add_argument("--start", default="20260418")
    ap.add_argument("--end", default="20261115")
    ap.add_argument("--out", default=None)
    ap.add_argument("--cache", default=None)
    a = ap.parse_args()
    root = Path(a.root)
    rng = np.random.default_rng(7)
    cache = Path(a.cache) if a.cache else None
    if cache and cache.with_suffix(".races.parquet").exists():
        R = pd.read_parquet(cache.with_suffix(".races.parquet"))
        S = pd.read_parquet(cache.with_suffix(".horses.parquet"))
    else:
        S0, L = parse_raw(root / "data/jrdb/raw/Sed")
        R, S = race_truth(S0, L)
        if cache:
            R.to_parquet(cache.with_suffix(".races.parquet"))
            S.drop(columns=[c for c in S.columns if S[c].dtype == object and c not in
                            ("race_key", "place", "surface", "ck", "jrdb_pace", "ped", "rk", "ymd")]).to_parquet(
                cache.with_suffix(".horses.parquet"))
    res = {}

    # ---- 0. 正解 B（馬の前半/上がり）と tenkai 自身の定義（runs.parquet）の一致
    runs = pd.read_parquet(root / "reports/speed_index/runs.parquet",
                           columns=["race_key", "date", "pace", "field", "finish", "fsize", "surface"])
    meta = json.loads((root / "reports/tenkai/meta.json").read_text(encoding="utf-8"))
    cuts = meta["cuts"]
    rp = runs.groupby("race_key").agg(pace_a=("pace", "first"), field=("field", "first"), n_runs=("pace", "size"),
                                      date=("date", "first"), surface=("surface", "first"))
    rp = rp[rp["pace_a"] != 0]
    c1 = rp["surface"].map(lambda s: cuts.get(s, [np.nan, np.nan])[0])
    c2 = rp["surface"].map(lambda s: cuts.get(s, [np.nan, np.nan])[1])
    rp["pace_a_cls"] = np.where(rp["pace_a"] < c1, 0, np.where(rp["pace_a"] < c2, 1, 2))
    rp["cov"] = rp["n_runs"] / rp["field"]
    J = R.join(rp[["pace_a", "pace_a_cls", "cov"]], how="left")
    chk = {}
    for nm, m in (("2024-25", J["date"].dt.year.between(2024, 2025)),
                  ("2026 tenkai期間・runs 6割以上", (J["date"] >= a.start) & (J["cov"] >= 0.6))):
        q = J[m & J["pace_a"].notna() & J["pace_b"].notna()]
        qc = q[q["pace_c"].notna()]
        chk[nm] = {"n": int(len(q)), "spearman_B_vs_A": float(q["pace_b"].corr(q["pace_a"], method="spearman")),
                   "cls_agree_B_vs_A": float((q["pace_b_cls"] == q["pace_a_cls"]).mean()),
                   "spearman_C_vs_A": float(qc["pace_c"].corr(qc["pace_a"], method="spearman")),
                   "cls_agree_C_vs_A": float((qc["pace_c_cls"] == qc["pace_a_cls"]).mean()),
                   "n_jrdb": int(q["jrdb_cls"].notna().sum()),
                   "cls_agree_jrdb_vs_A": float((q["jrdb_cls"] == q["pace_a_cls"])[q["jrdb_cls"].notna()].mean())}
    res["truth_check"] = chk

    # ---- tenkai 出力と結合
    TR, TH = load_tenkai(root / "reports/tenkai", root / "site/data", a.start, a.end)
    res["tenkai_files"] = {"days": int(TR["date"].nunique()), "races": int(len(TR)),
                           "first": TR["date"].min(), "last": TR["date"].max()}
    T = TR.join(J, on="race_key", rsuffix="_t", how="left")
    T = T[T["N"].notna()].copy()                     # SED のあるレース
    T["surface"] = T["surface"].fillna(T["course"].str[0].map({"芝": "芝", "ダ": "ダ"}))
    res["eval_set"] = {"days": int(T["date"].nunique()), "races": int(len(T)), "first": T["date"].min(),
                       "last": T["date"].max(),
                       "no_sed_days": sorted(set(TR["date"]) - set(T["date"]))}

    # 条件
    def kcls(k):
        k = str(k or "")
        for key, lab in (("新馬", "新馬"), ("未勝利", "未勝利"), ("1勝", "1勝"), ("2勝", "2勝"), ("3勝", "3勝")):
            if key in k:
                return lab
        if any(x in k for x in ("G1", "G2", "G3", "GI", "重賞", "Ｇ")):
            return "重賞"
        return "OP・L" if k else "不明"
    T["cls"] = T["klass"].map(kcls)
    T["fbin"] = pd.cut(T["N"], [0, 10, 13, 16, 18], labels=["〜10頭", "11〜13頭", "14〜16頭", "17〜18頭"])
    T["dbin"] = pd.cut(T["dist"], [0, 1400, 1800, 2200, 4000], labels=["〜1400", "1500〜1800", "1900〜2200", "2300〜"])
    T["hbin"] = pd.cut(T["nh0"], [-0.01, 0.0, 0.5, 1.0], labels=["全馬に過去走", "過去走なし 1〜5割", "過去走なし 5割超"])
    T["ptop"] = T[["p_slow", "p_mid", "p_high"]].max(axis=1)
    T["plab"] = T[["p_slow", "p_mid", "p_high"]].to_numpy().argmax(1)
    T["sure"] = np.where(T["plab"] == 1, "はっきりしない（平均寄り）",
                         np.where(T["ptop"] >= 0.5, "公算が大きい", "寄り"))

    # ---- 1. ペース
    pace = {}
    for truth, nm in (("pace_b_cls", "B 馬の前半/上がり（tenkai 定義の全頭再現）"), ("pace_c_cls", "C ラップ前後半差"),
                      ("jrdb_cls", "D JRDB の H/M/S（参考・別定義）")):
        pace[nm] = pace_block(T, truth, boot=1000, rng=rng)
    tA = T[(T["cov"] >= 0.6)]
    pace["A tenkai 定義そのもの（runs.parquet、復元 6 割以上のレースだけ）"] = pace_block(tA, "pace_a_cls")
    ok = T["pace_b"].notna()
    pace["spearman_(phigh-pslow)_vs_B"] = float((T.loc[ok, "p_high"] - T.loc[ok, "p_slow"]).corr(T.loc[ok, "pace_b"], method="spearman"))
    okc = T["pace_c"].notna()
    pace["spearman_(phigh-pslow)_vs_C"] = float((T.loc[okc, "p_high"] - T.loc[okc, "p_slow"]).corr(T.loc[okc, "pace_c"], method="spearman"))
    # 確度の言葉ごと
    sure = {}
    for s, g in T.groupby("sure"):
        sure[s] = {k: pace_block(g, k) for k in ("pace_b_cls", "pace_c_cls")}
    pace["by_sure_word"] = sure
    # 較正（B）
    cal = []
    tb = T[T["pace_b_cls"].notna()]
    for k, nm in enumerate(PACE):
        pc = tb[["p_slow", "p_mid", "p_high"]].to_numpy()[:, k]
        hit = (tb["pace_b_cls"] == k).to_numpy()
        for lo, hi in ((0, .25), (.25, .4), (.4, .55), (.55, 1.01)):
            mm = (pc >= lo) & (pc < hi)
            if mm.sum() >= 30:
                cal.append({"cls": nm, "band": f"{int(lo*100)}-{min(int(hi*100),100)}%", "n": int(mm.sum()),
                            "pred": round(float(pc[mm].mean()), 3), "actual": round(float(hit[mm].mean()), 3)})
    pace["calibration_B"] = cal
    res["pace"] = pace

    # ---- 2. 番手
    Sx = S[S["date"] >= pd.Timestamp(a.start)].copy()
    for c in ("c1", "c2", "c3", "c4"):
        Sx[c] = Sx[c].where(Sx[c] > 0)
    Sx["early"] = Sx["c1"].fillna(Sx["c2"]).fillna(Sx["c3"]).fillna(Sx["c4"])
    H = TH.merge(Sx[Sx["ran"]][["race_key", "umaban", "early", "c4", "N", "finish", "ped", "date"]],
                 on=["race_key", "umaban"], how="inner")
    # 素朴な基準: 前走の序盤位置 / 前 5 走平均（JRDB SED 2023〜の as-of、相対位置）
    Sh = S[S["ran"]].copy()
    for c in ("c1", "c2", "c3", "c4"):
        Sh[c] = Sh[c].where(Sh[c] > 0)
    Sh["e_rel"] = ((Sh["c1"].fillna(Sh["c2"]).fillna(Sh["c3"]).fillna(Sh["c4"]) - 1) / np.maximum(Sh["N"] - 1, 1))
    Sh["l_rel"] = (Sh["c4"] - 1) / np.maximum(Sh["N"] - 1, 1)
    Sh = Sh[Sh["e_rel"].notna() & (Sh["ped"] != "")].sort_values(["ped", "date", "race_key"])
    g = Sh.groupby("ped")
    Sh["b_e1"] = g["e_rel"].shift(1)
    Sh["b_e5"] = g["e_rel"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    Sh["b_l3"] = g["l_rel"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    H = H.merge(Sh[["race_key", "umaban", "b_e1", "b_e5", "b_l3"]], on=["race_key", "umaban"], how="left")
    for c in ("b_e1", "b_e5", "b_l3"):
        H[c + "_f"] = H[c].fillna(0.5)
    H = H.join(T.set_index("race_key")[["cls", "fbin", "dbin", "surface", "hbin"]], on="race_key", how="inner")
    pos = {"horses": int(len(H)), "races": int(H["race_key"].nunique()),
           "序盤 tenkai": pos_block(H, "pe", "early"), "序盤 前走そのまま": pos_block(H, "b_e1_f", "early"),
           "序盤 前5走平均": pos_block(H, "b_e5_f", "early"),
           "4角 tenkai": pos_block(H, "pl", "c4"), "4角 前3走平均": pos_block(H, "b_l3_f", "c4")}
    # 逃げ・先行勢・脚質区分
    H["pr_e"] = H.groupby("race_key")["pe"].rank(method="first")
    H["n_r"] = H.groupby("race_key")["pe"].transform("size")
    H["ar_e"] = H.groupby("race_key")["early"].rank(method="min")
    lead = H[H["pr_e"] == 1]
    pos["予想1番手が実際に先頭"] = float((lead["ar_e"] == 1).mean())
    pos["予想1番手が3番手以内"] = float((lead["ar_e"] <= 3).mean())
    front2 = H[H["pr_e"] <= 2]
    pos["先行勢(予想上位2頭)が実際に3番手以内"] = float((front2["ar_e"] <= 3).mean())
    pos["先行勢2頭のどちらかが先頭"] = float(front2.groupby("race_key")["ar_e"].min().eq(1).mean())
    H["st_p"] = style_of(H["pr_e"] / H["n_r"])
    H["st_a"] = style_of(H["ar_e"] / H["n_r"])
    pos["脚質区分の一致"] = float((H["st_p"] == H["st_a"]).mean())
    pos["脚質区分の2段以上ずれ"] = float(((H["st_p"] - H["st_a"]).abs() >= 2).mean())
    pos["脚質 混同表(行=予想 逃げ先行差し追込, 列=実際)"] = pd.crosstab(H["st_p"], H["st_a"]).reindex(
        index=range(4), columns=range(4), fill_value=0).to_numpy().tolist()
    # 過去走の有無
    pos["序盤 MAE(番手) 過去走なしの馬"] = float((H.loc[H["nh"] == 0, "pr_e"] - H.loc[H["nh"] == 0, "early"]).abs().mean())
    pos["序盤 MAE(番手) 過去走ありの馬"] = float((H.loc[H["nh"] > 0, "pr_e"] - H.loc[H["nh"] > 0, "early"]).abs().mean())
    res["position"] = pos

    # ---- 3. 届くか
    cr = json.loads((root / "reports/tenkai/course_late_rate.json").read_text(encoding="utf-8"))
    T["course_rate"] = (T["place"] + T["surface"] + T["dist"].astype(int).astype(str)).map(cr["course"]).fillna(
        T["surface"].map(cr["surface"]))
    tl = T[T["late_y"].notna() & T["late_p"].notna()]
    y, p = tl["late_y"].to_numpy(), tl["late_p"].to_numpy()
    ll = lambda yy, pp: float(-np.mean(yy * np.log(np.clip(pp, 1e-6, 1)) + (1 - yy) * np.log(np.clip(1 - pp, 1e-6, 1))))
    late = {"n": int(len(tl)), "actual_rate": float(y.mean()), "pred_mean": float(p.mean()),
            "auc": auc(y, p), "auc_course": auc(y, tl["course_rate"].to_numpy()),
            "brier": float(np.mean((p - y) ** 2)), "brier_course": float(np.mean((tl["course_rate"] - y) ** 2)),
            "brier_const2325": float(np.mean((0.114 - y) ** 2)),
            "logloss": ll(y, p), "logloss_course": ll(y, tl["course_rate"].to_numpy()),
            "logloss_const2325": ll(y, np.full(len(y), 0.114))}
    edges = meta["close"]["edges"]
    words = ["届きにくい", "やや届きにくい", "ふつう", "やや届きやすい", "届きやすい"]
    tl = tl.assign(word=[words[int(np.searchsorted(edges, v, side="left"))] if v == v else None for v in tl["late_p"]])
    # _late_word: b 進めるのは p > edge のとき → searchsorted(side='left')
    bands = []
    for w in words:
        g = tl[tl["word"] == w]
        k = int(g["late_y"].sum())
        bands.append({"word": w, "n": int(len(g)), "pred": float(g["late_p"].mean()) if len(g) else np.nan,
                      "actual": float(g["late_y"].mean()) if len(g) else np.nan, "ci": wilson(k, len(g))})
    late["bands"] = bands
    # 馬ごと: 予想 4 角が後ろ半分の馬の 3 着内率（言葉ごと）
    H["pr_l"] = H.groupby("race_key")["pl"].rank(method="first") / H["n_r"]
    H2 = H.join(tl.set_index("race_key")["word"], on="race_key", how="inner")
    hb = []
    for w in words:
        g = H2[H2["word"] == w]
        lt, fr = g[g["pr_l"] > 0.5], g[g["pr_l"] <= 1 / 3]
        hb.append({"word": w, "late_n": int(len(lt)), "late_top3": float((lt["finish"] <= 3).mean()) if len(lt) else np.nan,
                   "front_top3": float((fr["finish"] <= 3).mean()) if len(fr) else np.nan})
    late["horse_bands"] = hb
    res["late"] = late

    # ---- 4. 条件別
    cond = {}
    for col in ("fbin", "cls", "surface", "dbin", "hbin"):
        rows = []
        for key, g in T.groupby(col, observed=True):
            pb = pace_block(g, "pace_b_cls")
            pc = pace_block(g, "pace_c_cls")
            hg = H[H[col] == key]
            gl = g[g["late_y"].notna() & g["late_p"].notna()]
            rows.append({"key": str(key), "races": int(len(g)),
                         "pace_acc_B": pb.get("acc"), "pace_acc_B_ci": pb.get("acc_ci"),
                         "pace_bss_B": pb.get("bss_unif"), "pace_acc_C": pc.get("acc"), "pace_bss_C": pc.get("bss_unif"),
                         "early_sp": pos_block(hg, "pe", "early")["spearman"],
                         "early_sp_base5": pos_block(hg, "b_e5_f", "early")["spearman"],
                         "early_mae_rel": pos_block(hg, "pe", "early")["mae_rel"],
                         "c4_sp": pos_block(hg, "pl", "c4")["spearman"],
                         "late_n": int(len(gl)), "late_rate": float(gl["late_y"].mean()) if len(gl) else np.nan,
                         "late_auc": auc(gl["late_y"], gl["late_p"]) if len(gl) >= 30 else np.nan,
                         "late_auc_course": auc(gl["late_y"], gl["course_rate"]) if len(gl) >= 30 else np.nan,
                         "late_pred": float(gl["late_p"].mean()) if len(gl) else np.nan})
        cond[col] = rows
    res["by_condition"] = cond
    txt = json.dumps(res, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    if a.out:
        Path(a.out).write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
