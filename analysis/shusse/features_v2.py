# -*- coding: utf-8 -*-
"""
analysis/shusse/features_v2.py — 出世予測 AI v2: 目的変数・as-of 特徴（学習と推論で共用）
=========================================================================================
基準点 = 平地 k 走目終了時（k ∈ {1,2,3,5}）。特徴は基準日（k 走目当日）までの情報のみ。

目的変数（基準日から 365 日の窓 (base, base+365]）:
  ran_12m      窓内に 1 走以上したか（0 = 離脱）
  y_level      走った馬のみ: 窓内の最高到達クラス L0〜L4
               L4 = 重賞(race_level≥6) で 5 着以内（掲示板）。重賞 6 着以下は L3 扱い
               L0 未勝利圏 / L1 1勝 / L2 2勝 / L3 3勝/OP/L（+重賞着外）
  y_level_v1   v1 定義（出走ベース L4、離脱は L0）。比較用
  y_graded24   基準日から 730 日以内に重賞で 5 着以内（窓が閉じた馬のみ定義）

推論（predict_live.py）での再利用:
  tables = load_asof_tables(dir, k); spec = json.load(model_spec_k{k}.json)
  X = build_features(starts, horses, k, tables, priors=spec["te_priors"])
  starts は build_career_v2.py の starts_v2.parquet と同じ列を持つこと。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.shusse.build_career import COL_HORSE, DATA_END, TARGET_LEVEL_OF_RACE

N_LEVELS = 5
MAIN4 = {"東京", "中山", "京都", "阪神"}
KS = (1, 2, 3, 5)
SPLITS = {"train": (2013, 2019), "valid": (2020, 2021), "test": (2022, 2022), "test2023_partial": (2023, 2023)}
CAT_COLS = ["sex", "sire_type", "bms_type"]
TE_ENTITIES = {"trn": "trainer_debut", "sire": "sire", "brd": "breeder"}
TE_M = 20.0
GRADED_BOARD_POS = 5          # 掲示板 = 5 着以内
TARGET_LEVEL_NAME = {0: "未勝利圏", 1: "1勝クラス", 2: "2勝クラス", 3: "3勝/OP", 4: "重賞掲示板"}

# 推論時に落とすメタ列（特徴ではない）
META_COLS = {COL_HORSE, "馬名", "base_date", "debut_year", "split", "trainer_debut", "sire", "breeder",
             "ran_12m", "n_starts_12m", "y_level", "y_level_v1", "y_graded24", "h12_complete", "h24_complete",
             "rule_level", "rule_score", "y_dropout", "y_graded12"}


# --------------------------------------------------------------------------- targets
def start_level_v2(race_level: pd.Series, pos: pd.Series) -> pd.Series:
    """1 走の到達レベル（新定義）。重賞は 5 着以内で L4、着外は L3。"""
    lvl = race_level.map(TARGET_LEVEL_OF_RACE).astype(int)
    graded = race_level >= 6
    return lvl.where(~graded, np.where(pos <= GRADED_BOARD_POS, 4, 3)).astype(int)


def build_targets(starts: pd.DataFrame, k: int, data_end: pd.Timestamp = DATA_END) -> pd.DataFrame:
    """k 走目基準の目的変数（馬 1 行）。k 走目が存在する馬のみ。"""
    base = starts.loc[starts["k"] == k, [COL_HORSE, "date"]].rename(columns={"date": "base_date"})
    s = starts[starts["k"] > k][[COL_HORSE, "date", "race_level", "pos"]].merge(base, on=COL_HORSE, how="inner")
    s = s[s["date"] > s["base_date"]]
    s["lvl_v2"] = start_level_v2(s["race_level"], s["pos"])
    s["lvl_v1"] = s["race_level"].map(TARGET_LEVEL_OF_RACE).astype(int)
    s["board"] = ((s["race_level"] >= 6) & (s["pos"] <= GRADED_BOARD_POS)).astype(int)
    w12 = s[s["date"] <= s["base_date"] + pd.Timedelta(days=365)]
    w24 = s[s["date"] <= s["base_date"] + pd.Timedelta(days=730)]
    t = base.set_index(COL_HORSE)
    g12 = w12.groupby(COL_HORSE)
    t["n_starts_12m"] = g12.size()
    t["y_level"] = g12["lvl_v2"].max()
    t["y_level_v1"] = g12["lvl_v1"].max()
    t["y_graded24"] = w24.groupby(COL_HORSE)["board"].max()
    t["n_starts_12m"] = t["n_starts_12m"].fillna(0).astype(int)
    t["ran_12m"] = (t["n_starts_12m"] > 0).astype(int)
    t["y_level_v1"] = t["y_level_v1"].fillna(0).astype(int)        # v1: 離脱は L0
    t["y_graded24"] = t["y_graded24"].fillna(0).astype(int)
    t["h12_complete"] = t["base_date"] + pd.Timedelta(days=365) <= data_end
    t["h24_complete"] = t["base_date"] + pd.Timedelta(days=730) <= data_end
    t.loc[~t["h24_complete"], "y_graded24"] = np.nan
    t["y_dropout"] = 1 - t["ran_12m"]
    t["y_graded12"] = ((t["ran_12m"] == 1) & (t["y_level"] == 4)).astype(int)
    return t.reset_index()


def eligible_mask(horses: pd.DataFrame, targets: pd.DataFrame, k: int) -> pd.Series:
    """対象: 真のデビュー・デビュー年齢 2〜3・平地 k 走以上・12 か月窓がデータ末尾内。"""
    h = horses.set_index(COL_HORSE)
    t = targets.set_index(COL_HORSE)
    idx = t.index
    ok = h.loc[idx, "true_debut"].values & h.loc[idx, "debut_age"].isin([2, 3]).values & (h.loc[idx, "n_starts"].values >= k) & t["h12_complete"].values
    return pd.Series(ok, index=idx)


# --------------------------------------------------------------------------- as-of TE
def build_asof_tables(horses: pd.DataFrame, targets: pd.DataFrame, k: int) -> dict[str, pd.DataFrame]:
    """entity ごとの as-of イベント表。
    結果確定日 known = base_date_k + 365 に、その馬の (ran, lvl, l4, l3p) を累積する。
    deb 表はデビュー日ごとの累積頭数（規模）。学習・推論で同じ表を使う（parquet 保存）。"""
    h = horses[horses["true_debut"]][[COL_HORSE, "debut_date", *TE_ENTITIES.values()]]
    t = targets[targets["h12_complete"]][[COL_HORSE, "base_date", "ran_12m", "y_level"]]
    hist = h.merge(t, on=COL_HORSE, how="inner")
    hist["known"] = hist["base_date"] + pd.Timedelta(days=365)
    hist["ran"] = hist["ran_12m"].astype(float)
    lv = hist["y_level"].where(hist["ran_12m"] == 1)
    hist["lvl"] = lv.fillna(0.0).astype(float)
    hist["l4"] = (lv >= 4).fillna(False).astype(float)
    hist["l3p"] = (lv >= 3).fillna(False).astype(float)
    tables = {}
    for prefix, ent in TE_ENTITIES.items():
        e = hist.dropna(subset=[ent]).rename(columns={ent: "ent"})
        e["ent"] = e["ent"].astype(str)
        ev = e.sort_values(["known", COL_HORSE]).copy()
        ev["cnt"] = 1.0
        for c in ("cnt", "ran", "lvl", "l4", "l3p"):
            ev[f"cum_{c}"] = ev.groupby("ent")[c].cumsum()
        ev = ev[["ent", "known", "cum_cnt", "cum_ran", "cum_lvl", "cum_l4", "cum_l3p"]].rename(columns={"known": "t"})
        # 同日に複数頭が確定する entity がある。(ent, t) で 1 行（最後の累積）に潰す
        # （merge_asof 前の sort が非安定だと途中の累積行を拾う）
        ev = ev.groupby(["ent", "t"], as_index=False).last()
        hh = horses[horses["true_debut"]].dropna(subset=[ent])[[ent, "debut_date"]].rename(columns={ent: "ent", "debut_date": "t"})
        hh["ent"] = hh["ent"].astype(str)
        deb = hh.sort_values("t").copy()
        deb["cum_deb"] = deb.groupby("ent").cumcount() + 1.0
        deb = deb.groupby(["ent", "t"], as_index=False)["cum_deb"].max()
        tables[f"{prefix}_ev"] = ev.reset_index(drop=True)
        tables[f"{prefix}_deb"] = deb.reset_index(drop=True)
    return tables


def save_asof_tables(tables: dict[str, pd.DataFrame], out_dir: Path, k: int) -> None:
    for name, df in tables.items():
        df.to_parquet(out_dir / f"asof_{name}_k{k}.parquet", index=False)


def load_asof_tables(out_dir: Path, k: int) -> dict[str, pd.DataFrame]:
    tables = {}
    for prefix in TE_ENTITIES:
        for kind in ("ev", "deb"):
            tables[f"{prefix}_{kind}"] = pd.read_parquet(out_dir / f"asof_{prefix}_{kind}_k{k}.parquet")
    return tables


def te_priors(targets: pd.DataFrame, horse_ids: pd.Series) -> dict[str, float]:
    """縮約の事前分布（train コホートの全体率）。"""
    t = targets[targets[COL_HORSE].isin(horse_ids)]
    ran = t[t["ran_12m"] == 1]
    return {"ran": float(t["ran_12m"].mean()), "lvl": float(ran["y_level"].mean()),
            "l4": float((ran["y_level"] >= 4).mean()), "l3p": float((ran["y_level"] >= 3).mean())}


def add_asof_te(X: pd.DataFrame, tables: dict[str, pd.DataFrame], priors: dict[str, float], m: float = TE_M) -> pd.DataFrame:
    """known < base_date（同日を含めない）の過去馬だけで集計した as-of TE を付ける。"""
    X = X.copy()
    for prefix, ent in TE_ENTITIES.items():
        q = X[[COL_HORSE, ent, "base_date"]].rename(columns={ent: "ent", "base_date": "t"}).copy()
        q["ent"] = q["ent"].astype(str)
        q["t"] = pd.to_datetime(q["t"])
        q = q.sort_values("t")
        ev = tables[f"{prefix}_ev"].copy()
        deb = tables[f"{prefix}_deb"].copy()
        ev["ent"] = ev["ent"].astype(str)
        deb["ent"] = deb["ent"].astype(str)
        a = pd.merge_asof(q, ev.sort_values("t", kind="stable"), on="t", by="ent", allow_exact_matches=False, direction="backward").set_index(COL_HORSE)
        b = pd.merge_asof(q, deb.sort_values("t", kind="stable"), on="t", by="ent", allow_exact_matches=False, direction="backward").set_index(COL_HORSE)
        n = a["cum_cnt"].fillna(0)
        nr = a["cum_ran"].fillna(0)
        idx = X[COL_HORSE]
        X[f"{prefix}_n_known"] = n.reindex(idx).values
        X[f"{prefix}_te_ran"] = ((nr + m * priors["ran"]) / (n + m)).reindex(idx).values
        X[f"{prefix}_te_lvl"] = ((a["cum_lvl"].fillna(0) + m * priors["lvl"]) / (nr + m)).reindex(idx).values
        X[f"{prefix}_te_l4"] = ((a["cum_l4"].fillna(0) + m * priors["l4"]) / (nr + m)).reindex(idx).values
        X[f"{prefix}_te_l3p"] = ((a["cum_l3p"].fillna(0) + m * priors["l3p"]) / (nr + m)).reindex(idx).values
        X[f"{prefix}_n_debut"] = b["cum_deb"].fillna(0).reindex(idx).values
    return X


# --------------------------------------------------------------------------- rule baseline
def promotion_level(race_level: int, won: int) -> int:
    """規則: k 走目のクラスと勝敗から「次に走るクラス」。"""
    cur = TARGET_LEVEL_OF_RACE[int(race_level)]
    if not won:
        return cur
    if race_level <= 1:
        return 1
    if race_level == 2:
        return 2
    if race_level in (3, 4):
        return 3
    return 4


# --------------------------------------------------------------------------- features
def build_features(starts: pd.DataFrame, horses: pd.DataFrame, k: int, tables: dict[str, pd.DataFrame],
                   priors: dict[str, float], horse_ids: pd.Series | None = None) -> pd.DataFrame:
    """k 走目基準の特徴行列（馬 1 行）。基準日以前（k 走目まで）の行しか見ない。
    horse_ids を与えるとその馬だけ。目的変数は付けない（train_v2 側で build_targets を merge）。"""
    s = starts[starts["k"] <= k]
    if horse_ids is not None:
        s = s[s[COL_HORSE].isin(horse_ids)]
    has_k = s.loc[s["k"] == k, COL_HORSE]
    s = s[s[COL_HORSE].isin(has_k)].copy()
    s["turf"] = (s["芝・ダ"].astype(str).str.strip() == "芝").astype(int)
    s["main4"] = s["場所"].astype(str).str.strip().isin(MAIN4).astype(int)
    wide: dict[str, pd.Series] = {}
    for j in range(1, k + 1):
        r = s[s["k"] == j].set_index(COL_HORSE)
        wide[f"pos_{j}"] = r["pos"]
        wide[f"relpos_{j}"] = r["rel_pos"]
        wide[f"field_{j}"] = r["field"]
        wide[f"lvl_{j}"] = r["race_level"]
        wide[f"win_{j}"] = r["win"]
        wide[f"dist_{j}"] = r["距離"]
        wide[f"turf_{j}"] = r["turf"]
        wide[f"main4_{j}"] = r["main4"]
        wide[f"jky_top3_{j}"] = r["jky_top3_2y"]          # 騎手一般の as-of 3 着内率（直近 2 年、当日除く）
        wide[f"jky_n_{j}"] = r["jky_n_2y"]
        wide[f"jky_pair_top3_{j}"] = r["jockey_top3_rate"]  # master 既存: 馬×騎手ペアの as-of 率（初コンビは NaN）
        wide[f"jky_age_{j}"] = r["騎手年齢"]
        wide[f"margin_{j}"] = r["margin"]
        wide[f"agari_{j}"] = r["agari3f"]
        wide[f"agari_rank_{j}"] = r["agari_rank"]
        wide[f"weight_{j}"] = r["weight"]
        if j >= 2:
            wide[f"gap_days_{j}"] = r["days_since_prev"]
    X = pd.DataFrame(wide)
    rk = s[s["k"] == k].set_index(COL_HORSE).reindex(X.index)
    r1 = s[s["k"] == 1].set_index(COL_HORSE).reindex(X.index)
    js = list(range(1, k + 1))
    X["base_date"] = rk["date"]
    X["age_base"] = rk["年齢"]
    X["trainer_age"] = rk["調教師年齢"]
    X["debut_month"] = r1["date"].dt.month
    X["debut_age"] = r1["年齢"]
    X["n_wins"] = X[[f"win_{j}" for j in js]].sum(axis=1)
    X["n_top3"] = (X[[f"pos_{j}" for j in js]] <= 3).sum(axis=1)
    X["best_pos"] = X[[f"pos_{j}" for j in js]].min(axis=1)
    X["mean_relpos"] = X[[f"relpos_{j}" for j in js]].mean(axis=1)
    X["max_lvl"] = X[[f"lvl_{j}" for j in js]].max(axis=1)
    X["best_margin"] = X[[f"margin_{j}" for j in js]].min(axis=1)
    X["mean_margin"] = X[[f"margin_{j}" for j in js]].mean(axis=1)
    X["best_agari_rank"] = X[[f"agari_rank_{j}" for j in js]].min(axis=1)
    X["mean_agari_rank"] = X[[f"agari_rank_{j}" for j in js]].mean(axis=1)
    X["dist_mean"] = X[[f"dist_{j}" for j in js]].mean(axis=1)
    X["turf_share"] = X[[f"turf_{j}" for j in js]].mean(axis=1)
    if k >= 2:
        X["trend_pos"] = X[f"pos_{k}"] - X["pos_1"]
        X["trend_margin"] = X[f"margin_{k}"] - X["margin_1"]
        X["weight_change"] = X[f"weight_{k}"] - X["weight_1"]
        X["days_debut_to_k"] = (rk["date"] - r1["date"]).dt.days
    h = horses.set_index(COL_HORSE).reindex(X.index)
    X["馬名"] = h["馬名"]
    X["debut_year"] = h["debut_year"]
    X["sex"] = h["sex"]
    X["sire_type"] = h["sire_type"]
    X["bms_type"] = h["bms_type"]
    X["debut_dayage"] = h["debut_dayage"]
    X["trainer_debut"] = h["trainer_debut"]
    X["sire"] = h["sire"]
    X["breeder"] = h["breeder"]
    X["rule_level"] = [promotion_level(int(l), int(w)) for l, w in zip(X[f"lvl_{k}"], X[f"win_{k}"])]
    X["rule_score"] = (X["rule_level"] * 10 + X["n_wins"] * 3 + (3 - X["best_pos"].clip(upper=3)) + (1 - X["mean_relpos"]))
    X = X.reset_index()
    X = add_asof_te(X, tables, priors)
    return X


def feature_cols(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if c not in META_COLS]


def feature_groups(cols: list[str]) -> dict[str, list[str]]:
    """アブレーション用の群分け。"""
    jky = [c for c in cols if c.startswith("jky_")]
    ped = [c for c in cols if c in ("sire_type", "bms_type", "trainer_age") or c.split("_")[0] in TE_ENTITIES]
    body = [c for c in cols if c.startswith("weight") or c == "debut_dayage"]
    perf = [c for c in cols if c not in jky + ped + body]
    return {"perf": perf, "jockey": jky, "pedigree_stable": ped, "body": body}


def prep_matrix(X: pd.DataFrame, feats: list[str], cats: dict[str, list[str]] | None = None):
    """LightGBM 用行列。cats は {col: [categories]}（学習時に固定し spec に保存）。"""
    M = X[feats].copy()
    if cats is None:
        cats = {c: sorted(M[c].dropna().astype(str).unique().tolist()) for c in CAT_COLS if c in feats}
    for c, levels in cats.items():
        if c in M.columns:
            M[c] = pd.Categorical(M[c].astype(str).where(M[c].notna(), None), categories=levels)
    return M, cats


# --------------------------------------------------------------------------- labels
_PER_START = {"pos": "{j}走目の着順", "relpos": "{j}走目の相対着順(0=1着)", "field": "{j}走目の頭数", "lvl": "{j}走目のクラス",
              "win": "{j}走目の勝利", "dist": "{j}走目の距離", "turf": "{j}走目が芝", "main4": "{j}走目が主場(東京/中山/京都/阪神)",
              "jky_top3": "{j}走目の騎手の3着内率(直近2年)", "jky_n": "{j}走目の騎手の騎乗数(直近2年)",
              "jky_pair_top3": "{j}走目の騎手とのコンビ3着内率", "jky_age": "{j}走目の騎手年齢", "margin": "{j}走目の勝ち馬との着差(秒)",
              "agari": "{j}走目の上り3F(秒)", "agari_rank": "{j}走目の上り3F順位", "weight": "{j}走目の馬体重", "gap_days": "{j}走目までの間隔(日)"}
_COMMON = {"age_base": "基準日時点の年齢", "trainer_age": "調教師年齢", "debut_month": "デビュー月", "debut_age": "デビュー時年齢",
           "n_wins": "勝利数", "n_top3": "3着内回数", "best_pos": "最高着順", "mean_relpos": "平均相対着順", "max_lvl": "出走した最高クラス",
           "best_margin": "最良の着差(秒)", "mean_margin": "平均着差(秒)", "best_agari_rank": "最良の上り3F順位", "mean_agari_rank": "平均上り3F順位",
           "dist_mean": "平均距離", "turf_share": "芝出走比率", "trend_pos": "着順の推移(最新−初戦)", "trend_margin": "着差の推移(最新−初戦)",
           "weight_change": "馬体重の増減(最新−初戦)", "days_debut_to_k": "デビューから基準日までの日数", "debut_dayage": "デビュー時の日齢",
           "sex": "性別", "sire_type": "父系タイプ", "bms_type": "母父系タイプ"}
_TE = {"n_known": "{e}の過去馬数(結果確定済)", "te_ran": "{e}の過去馬の継続出走率", "te_lvl": "{e}の過去馬の平均到達レベル",
       "te_l4": "{e}の過去馬の重賞掲示板率", "te_l3p": "{e}の過去馬のOP以上到達率", "n_debut": "{e}のデビュー頭数(基準日まで)"}
_ENT_JA = {"trn": "調教師", "sire": "種牡馬", "brd": "生産者"}


def feature_labels(cols: list[str]) -> dict[str, str]:
    out = {}
    for c in cols:
        if c in _COMMON:
            out[c] = _COMMON[c]
            continue
        head, _, tail = c.rpartition("_")
        if head in _PER_START and tail.isdigit():
            out[c] = _PER_START[head].format(j=tail)
            continue
        p, _, rest = c.partition("_")
        if p in _ENT_JA and rest in _TE:
            out[c] = _TE[rest].format(e=_ENT_JA[p])
            continue
        out[c] = c
    return out


def save_json(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (pd.Timestamp,)):
        return str(o.date())
    return str(o)
