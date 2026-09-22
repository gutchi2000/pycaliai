# -*- coding: utf-8 -*-
"""
EXP15 共通部品 — データ読み込み・clean 契約・race テンソル化・指標・τ・bootstrap
==============================================================================
spec.json が正本。本番 (compute_bets / export_weekly_marks / models/) には一切触れない。
2024-01-01 以降の行は chunk 単位で捨て、保持しない (spec.periods.load_filter)。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

EXP = Path(__file__).resolve().parent
BASE = EXP.parents[2]
OUT = EXP / "out"
CACHE = BASE / "data" / "_research" / "mcond" / "exp15"
MASTER = BASE / "data" / "master_v2_20130105-20251228.csv"
KEKKA = BASE / "data" / "kekka_20130105-20251228.csv"
V6 = BASE / "models" / "unified_rank_v6.pkl"
SPEC = json.loads((EXP / "spec.json").read_text(encoding="utf-8"))

COL_RID = "レースID(新/馬番無)"
COL_JYUN = "着順"
COL_DATE = "日付"
D_MIN, D_MAX = 20160101, 20231231

PERIODS = {  # yyyymmdd 閉区間
    "train": (20160101, 20211231),
    "sel": (20220101, 20221231),
    "dev": (20230101, 20231231),
}


def v6_meta() -> tuple[list[str], list[str]]:
    import joblib
    b = joblib.load(V6)
    return list(b["feature_cols"]), list(b["cat_cols"])


# ---------------------------------------------------------------- 読み込み
def load_rows(force: bool = False) -> pd.DataFrame:
    """master_v2 の 2016-2023 行 (文字列のまま)。2024 以降は chunk 単位で捨てる。"""
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / "rows_2016_2023.parquet"
    if cache.exists() and not force:
        df = pd.read_parquet(cache)
    else:
        feats, _ = v6_meta()
        cols = list(dict.fromkeys([COL_DATE, COL_RID, COL_JYUN, "血統登録番号", *feats]))
        keep = []
        for ch in pd.read_csv(MASTER, encoding="utf-8-sig", dtype=str, usecols=cols,
                              chunksize=100_000):
            d = pd.to_numeric(ch[COL_DATE], errors="coerce")
            keep.append(ch[(d >= D_MIN) & (d <= D_MAX)])
            del ch
        df = pd.concat(keep, ignore_index=True)
        df.to_parquet(cache, index=False)
    df = df.copy()
    df["date"] = pd.to_numeric(df[COL_DATE], errors="coerce").astype("Int64")
    assert int(df["date"].max()) <= D_MAX and int(df["date"].min()) >= D_MIN, "T12 違反"
    # v6 と同じ母集団: 着順が数値・race id あり
    df["jyun"] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df = df.dropna(subset=["jyun", COL_RID]).copy()
    df["jyun"] = df["jyun"].astype(int)
    df["rid16"] = df[COL_RID].astype(str).str.replace(r"\.0$", "", regex=True).str[:16]
    df["meeting_day"] = df["rid16"].str[:10]
    df["ban"] = pd.to_numeric(df["馬番"], errors="coerce").astype(int)
    df["period"] = None
    for k, (a, b) in PERIODS.items():
        df.loc[(df["date"] >= a) & (df["date"] <= b), "period"] = k
    assert df["period"].notna().all()
    df = df.sort_values(["rid16", "ban"]).reset_index(drop=True)
    df["rel"] = np.clip(6 - df["jyun"], 0, 5).astype(int)
    df["win"] = (df["jyun"] == 1).astype(int)
    df["top3"] = (df["jyun"] <= 3).astype(int)
    return df


def row_hash(df: pd.DataFrame) -> str:
    key = (df["rid16"] + "_" + df["ban"].astype(str)).sort_values()
    return hashlib.sha256("\n".join(key).encode()).hexdigest()


# ---------------------------------------------------------------- 契約
def load_contract() -> dict:
    return json.loads((OUT / "feature_contract.json").read_text(encoding="utf-8"))


def class_group(s: pd.Series) -> pd.Series:
    t = s.fillna("").astype(str)
    out = pd.Series("OP以上", index=s.index)
    out[t.str.contains("新馬")] = "新馬"
    out[t.str.contains("未勝利")] = "未勝利"
    out[t.str.contains("1勝|500万|５００万|１勝")] = "1勝"
    out[t.str.contains("2勝|1000万|１０００万|２勝")] = "2勝"
    out[t.str.contains("3勝|1600万|１６００万|３勝")] = "3勝"
    return out


class Encoded:
    """contract を当てた数値行列。R0 用 (整数/欠損 -9999) と NN 用 (埋め込み idx + 標準化数値)。"""

    def __init__(self, df: pd.DataFrame, contract: dict, which: str = "clean"):
        cols = contract["features"][which]
        cats = [c for c in cols if c in contract["cat_cols"]]
        nums = [c for c in cols if c not in contract["cat_cols"]]
        tr = df["period"] == "train"
        self.cols, self.cats, self.nums = cols, cats, nums
        # --- カテゴリ: v6 と同じ astype(str) 語彙を base_train で fit (T8)
        self.vocab = {}
        cat_codes = np.zeros((len(df), len(cats)), dtype=np.int64)
        for j, c in enumerate(cats):
            v = df[c].astype(str)
            classes = np.array(sorted(set(v[tr]) | {"__NaN__"}))
            self.vocab[c] = classes
            v = v.where(v.isin(set(classes)), "__NaN__")
            cat_codes[:, j] = np.searchsorted(classes, v.to_numpy())
        # --- 数値: v6 と同じ pd.to_numeric(coerce)
        num = np.column_stack([pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
                               for c in nums]) if nums else np.zeros((len(df), 0))
        self.cat_codes, self.num_raw = cat_codes, num
        # R0 行列 (v6 と同じ: カテゴリ整数・数値・欠損 -9999)。列順は cols
        X = np.zeros((len(df), len(cols)), dtype=np.float64)
        ci = {c: j for j, c in enumerate(cats)}
        ni = {c: j for j, c in enumerate(nums)}
        for k, c in enumerate(cols):
            X[:, k] = cat_codes[:, ci[c]] if c in ci else num[:, ni[c]]
        X[np.isnan(X)] = -9999.0
        self.X_r0 = X
        # NN 数値: base_train の中央値・IQR (T8)
        trn = num[tr.to_numpy()]
        med = np.nanmedian(trn, axis=0)
        q75, q25 = np.nanpercentile(trn, 75, axis=0), np.nanpercentile(trn, 25, axis=0)
        iqr = np.where((q75 - q25) > 0, q75 - q25, np.nanstd(trn, axis=0))
        iqr = np.where(iqr > 0, iqr, 1.0)
        miss_cols = np.where(np.isnan(trn).any(axis=0))[0]
        z = (np.where(np.isnan(num), med, num) - med) / iqr
        z = np.clip(z, -10, 10)
        miss = np.isnan(num[:, miss_cols]).astype(np.float32)
        self.num_nn = np.hstack([z, miss]).astype(np.float32)
        self.fit_stats = {"median": med, "iqr": iqr, "miss_cols": miss_cols,
                          "fit_rows": int(tr.sum())}
        self.card = [len(self.vocab[c]) for c in cats]


# ---------------------------------------------------------------- race テンソル
def race_index(df: pd.DataFrame, mask: np.ndarray) -> list[np.ndarray]:
    """mask 行を race (rid16) ごとの行番号配列に。df は rid16, ban でソート済み前提。"""
    sub = np.where(mask)[0]
    rid = df["rid16"].to_numpy()[sub]
    cut = np.flatnonzero(rid[1:] != rid[:-1]) + 1
    return np.split(sub, cut)


# ---------------------------------------------------------------- 確率・指標
def softmax_race(s: np.ndarray, races: list[np.ndarray], tau: float) -> np.ndarray:
    p = np.empty_like(s, dtype=float)
    for idx in races:
        z = s[idx] / tau
        z = z - z.max()
        e = np.exp(z)
        p[idx] = e / e.sum()
    return p


def fit_tau_padded(S: np.ndarray, W: np.ndarray, M: np.ndarray) -> tuple[float, float]:
    """padded (R,H) スコアで 1着条件付きロジット最尤の温度 τ と、その τ での平均 NLL。
    単独勝ちレースのみ使う (analysis.mcond.v6base.fit_tau と同じ目的関数)。"""
    W = (W * M).astype(float)
    keep = W.sum(1) == 1
    S, W, M = S[keep], W[keep], M[keep]
    S0 = np.where(M, S, 0.0)
    sw = (S0 * W).sum(1)

    def nll(tau):
        z = np.where(M, S0 / tau, -np.inf)
        mx = z.max(1)
        lse = mx + np.log(np.exp(z - mx[:, None]).sum(1))
        return float((lse - sw / tau).mean())

    r = minimize_scalar(nll, bounds=(0.02, 50.0), method="bounded", options={"xatol": 1e-5})
    return float(r.x), float(r.fun)


def fit_tau(s: np.ndarray, win: np.ndarray, races: list[np.ndarray]) -> float:
    H = max(len(i) for i in races)
    S = np.zeros((len(races), H))
    W = np.zeros((len(races), H))
    M = np.zeros((len(races), H), dtype=bool)
    for r, i in enumerate(races):
        S[r, :len(i)] = s[i]
        W[r, :len(i)] = win[i]
        M[r, :len(i)] = True
    return fit_tau_padded(S, W, M)[0]


def race_metrics(df: pd.DataFrame, races: list[np.ndarray], s: np.ndarray,
                 p: np.ndarray) -> pd.DataFrame:
    """race 単位の指標表 (1行=1レース)。win 系は単独勝ちのレースのみ値を持つ。"""
    win = df["win"].to_numpy()
    rel = df["rel"].to_numpy()
    top3 = df["top3"].to_numpy()
    ban = df["ban"].to_numpy()
    rows = []
    disc = 1.0 / np.log2(np.arange(2, 20))
    for idx in races:
        n = len(idx)
        w = win[idx]
        pi = p[idx]
        single = w.sum() == 1
        ll = -np.log(max(pi[w == 1][0], 1e-12)) if single else np.nan
        br = float(((pi - w) ** 2).sum()) if single else np.nan
        # スコア同点は馬番昇順 (lexsort: 最後のキーが主キー)
        order = np.lexsort((ban[idx], -s[idx]))
        g = (2.0 ** rel[idx] - 1)
        ideal = np.sort(g)[::-1]
        nd = {}
        for k in (3, 5):
            kk = min(k, n)
            dcg = (g[order[:kk]] * disc[:kk]).sum()
            idcg = (ideal[:kk] * disc[:kk]).sum()
            nd[k] = dcg / idcg if idcg > 0 else np.nan
        rows.append((idx[0], n, single, ll, br, nd[3], nd[5], int(top3[idx][order[0]])))
    out = pd.DataFrame(rows, columns=["row0", "n", "single_win", "ll", "brier",
                                      "ndcg3", "ndcg5", "hon_top3"])
    out["rid16"] = df["rid16"].to_numpy()[out["row0"]]
    out["meeting_day"] = df["meeting_day"].to_numpy()[out["row0"]]
    return out


def ece10(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    idx = np.clip((p * bins).astype(int), 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def boot_delta(d: np.ndarray, day: np.ndarray, reps: int = 3000, seed: int = 20260923,
               level: float = 0.95) -> dict:
    """race 単位の差 d を meeting_day で resample (ratio of sums)。"""
    ok = ~np.isnan(d)
    dd = pd.DataFrame({"d": d[ok], "day": day[ok]}).groupby("day")["d"].agg(["sum", "count"])
    s, c = dd["sum"].to_numpy(), dd["count"].to_numpy()
    rng = np.random.default_rng(seed)
    n = len(s)
    bs = np.empty(reps)
    for k in range(reps):
        i = rng.integers(0, n, n)
        bs[k] = s[i].sum() / c[i].sum()
    a = (1 - level) / 2
    return {"delta": float(d[ok].mean()), "ci_lo": float(np.quantile(bs, a)),
            "ci_hi": float(np.quantile(bs, 1 - a)), "se": float(bs.std(ddof=1)),
            "n_races": int(ok.sum()), "n_days": int(n)}


def boot_ece_delta(pa: np.ndarray, pb: np.ndarray, y: np.ndarray, day_row: np.ndarray,
                   reps: int = 1000, seed: int = 20260923) -> dict:
    days = np.unique(day_row)
    pos = {d: np.where(day_row == d)[0] for d in days}
    rng = np.random.default_rng(seed)
    bs = np.empty(reps)
    for k in range(reps):
        pick = rng.choice(days, len(days))
        ii = np.concatenate([pos[d] for d in pick])
        bs[k] = ece10(pa[ii], y[ii]) - ece10(pb[ii], y[ii])
    return {"delta": ece10(pa, y) - ece10(pb, y), "ci_lo": float(np.quantile(bs, 0.025)),
            "ci_hi": float(np.quantile(bs, 0.975))}


def dump(obj, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name

    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    p.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=conv), encoding="utf-8")
    return p
