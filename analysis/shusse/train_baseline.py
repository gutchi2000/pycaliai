# -*- coding: utf-8 -*-
"""
analysis/shusse/train_baseline.py — 出世予測 AI: ベースライン学習・評価・読み物カード
=================================================================================
問題設定（docs/research/SHUSSE_AI_DESIGN_20261030.md 参照）:
  基準点 = 平地 3 走目終了時。
  主タスク A: 基準日から 365 日以内に出走した最高クラス（5 段順序カテゴリ）
      L0 未勝利圏(新馬/未勝利のみ or 出走なし) / L1 1勝 / L2 2勝 / L3 3勝/OP/L / L4 重賞(G3+)
  副タスク B: 基準日から 730 日以内に重賞（G3 以上）へ出走するか（2 値）
  特徴は基準日以前の情報のみ（as-of）。市場オッズは使わない。

分割はデビュー年コホート: train 2013-2019 / valid 2020-2021 / test 2022。

実行（root から）:
    venv311\\Scripts\\python.exe -m analysis.shusse.train_baseline

出力（analysis/shusse/out/）:
    dataset.parquet          基準点 1 行/馬 の特徴 + 目的変数
    metrics.json             規則 / LightGBM の成績（valid, test）
    feature_importance_A.csv / feature_importance_B.csv
    calibration_B.csv        2 値タスクの信頼度図（test, 10 分位）
    pred_test.parquet        test 世代の予測
    cards.md / cards.json    読み物カード（test 世代の数頭）
    lgbm_A.txt / lgbm_B.txt  モデル
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, cohen_kappa_score, log_loss, roc_auc_score

from analysis.shusse.build_career import (
    COL_HORSE, DATA_END, RACE_LEVEL_NAME, TARGET_LEVEL_NAME, TARGET_LEVEL_OF_RACE,
)

OUT = Path(__file__).resolve().parent / "out"
SEED = 20261030
N_LEVELS = 5
SPLITS = {"train": (2013, 2019), "valid": (2020, 2021), "test": (2022, 2022)}
MAIN4 = {"東京", "中山", "京都", "阪神"}

# 読み物カードに載せたい test 世代（2022 デビュー）の馬。無ければスキップ。
CARD_NAMES = ["リバティアイランド", "ソールオリエンス", "タスティエーラ", "ドゥレッツァ",
              "ドゥラエレーデ", "ハーパー", "ベラジオオペラ", "スキルヴィング", "ペリファーニア"]


# --------------------------------------------------------------------------- dataset
def promotion_level(race_level: int, won: int) -> int:
    """規則ベースライン: 3 走目のクラスと勝敗から「次に走るクラス」を返す。"""
    cur = TARGET_LEVEL_OF_RACE[race_level]
    if not won:
        return cur
    if race_level <= 1:      # 新馬/未勝利 勝ち → 1勝
        return 1
    if race_level == 2:      # 1勝 勝ち → 2勝
        return 2
    if race_level in (3, 4): # 2勝/3勝 勝ち → 3勝/OP
        return 3
    return 4                 # OP/重賞 勝ち → 重賞


def build_dataset(starts: pd.DataFrame, career: pd.DataFrame) -> pd.DataFrame:
    c = career[career["true_debut"] & career["debut_age"].isin([2, 3]) & (career["n_starts"] >= 3)
               & career["horizon12_complete"]].copy()
    c = c[c["debut_year"].between(2013, 2022)]
    s = starts[starts[COL_HORSE].isin(c[COL_HORSE]) & (starts["k"] <= 3)].copy()
    s["surface_turf"] = (s["芝・ダ"].astype(str).str.strip() == "芝").astype(int)
    s["main4"] = s["場所"].isin(MAIN4).astype(int)
    s["month"] = s["date"].dt.month

    wide = {}
    for k in (1, 2, 3):
        r = s[s["k"] == k].set_index(COL_HORSE)
        wide[f"pos_{k}"] = r["pos"]
        wide[f"relpos_{k}"] = r["rel_pos"]
        wide[f"field_{k}"] = r["field"]
        wide[f"lvl_{k}"] = r["race_level"]
        wide[f"win_{k}"] = r["win"]
        wide[f"dist_{k}"] = r["距離"]
        wide[f"turf_{k}"] = r["surface_turf"]
        wide[f"main4_{k}"] = r["main4"]
        wide[f"jky_top3_{k}"] = r["jockey_top3_rate"]
        wide[f"jky_age_{k}"] = r["騎手年齢"]
        if k <= 2:
            # 着差は k+1 走目の行から持ち上げたもの。k=3 は 4 走目の存在と相関するので使わない
            wide[f"margin_{k}"] = r["margin_this"]
        if k >= 2:
            wide[f"gap_days_{k}"] = r["days_since_prev"]
    X = pd.DataFrame(wide)
    r3 = s[s["k"] == 3].set_index(COL_HORSE)
    r1 = s[s["k"] == 1].set_index(COL_HORSE)
    X["base_date"] = r3["date"]
    X["age_base"] = r3["年齢"]
    X["trainer"] = r3["調教師コード"]
    X["trainer_age"] = r3["調教師年齢"]
    X["debut_month"] = r1["month"]
    X["debut_age"] = r1["年齢"]
    X["days_debut_to_3rd"] = (r3["date"] - r1["date"]).dt.days
    X["n_wins3"] = X[["win_1", "win_2", "win_3"]].sum(axis=1)
    X["n_top3_3"] = (X[["pos_1", "pos_2", "pos_3"]] <= 3).sum(axis=1)
    X["best_pos3"] = X[["pos_1", "pos_2", "pos_3"]].min(axis=1)
    X["mean_relpos3"] = X[["relpos_1", "relpos_2", "relpos_3"]].mean(axis=1)
    X["max_lvl3"] = X[["lvl_1", "lvl_2", "lvl_3"]].max(axis=1)
    X["best_margin12"] = X[["margin_1", "margin_2"]].min(axis=1)
    X["trend_pos"] = X["pos_3"] - X["pos_1"]
    X["dist_mean3"] = X[["dist_1", "dist_2", "dist_3"]].mean(axis=1)
    X["turf_share3"] = X[["turf_1", "turf_2", "turf_3"]].mean(axis=1)
    X["year_base"] = X["base_date"].dt.year

    c = c.set_index(COL_HORSE)
    X["馬名"] = c["馬名"]
    X["debut_year"] = c["debut_year"]
    X["sex"] = c["sex"]
    X["sire"] = c["sire"]
    X["trainer_debut"] = c["trainer_debut"]
    X["sire_type"] = c["sire_type"]
    X["bms_type"] = c["bms_type"]
    X["breeder"] = c["breeder"]
    X["y_level"] = c["level_12m_after_3rd"].astype(int)
    X["y_graded24"] = c["graded_24m_after_3rd"].astype(int)
    X["h24_complete"] = c["horizon24_complete"]
    X["n_starts_12m"] = c["n_starts_12m_after_3rd"]
    X["rule_level"] = [promotion_level(int(l), int(w)) for l, w in zip(X["lvl_3"], X["win_3"])]
    X["rule_score"] = (X["rule_level"] * 10 + X["n_wins3"] * 3 + (3 - X["best_pos3"].clip(upper=3))
                       + (1 - X["mean_relpos3"]))
    X["split"] = "none"
    for name, (a, b) in SPLITS.items():
        X.loc[X["debut_year"].between(a, b), "split"] = name
    return X.reset_index()


def add_asof_te(X: pd.DataFrame, career: pd.DataFrame, entity: str, prefix: str, m: float = 20.0) -> pd.DataFrame:
    """as-of ターゲットエンコーディング。
    entity の過去馬のうち「結果が確定した日 (= 3 走目 + 365 日) ≤ 基準日」のものだけで
    平均到達レベル・重賞(L4)率・OP以上(L3+)率を集計し、全体率へ縮約する。
    さらに基準日より前にデビューした同 entity の馬数（規模）も付ける。"""
    hist = career[career["true_debut"] & career["third_start_date"].notna()].copy()
    hist["known_date"] = hist["third_start_date"] + pd.Timedelta(days=365)
    hist = hist[[entity, "known_date", "level_12m_after_3rd", "debut_date"]].dropna(subset=[entity])
    hist = hist.rename(columns={entity: "ent"})
    hist["lvl"] = hist["level_12m_after_3rd"].astype(float)
    hist["l4"] = (hist["lvl"] >= 4).astype(float)
    hist["l3p"] = (hist["lvl"] >= 3).astype(float)
    ev = hist.sort_values(["known_date"]).copy()
    ev["cnt"] = 1.0
    for col in ("cnt", "lvl", "l4", "l3p"):
        ev[f"cum_{col}"] = ev.groupby("ent")[col].cumsum()
    ev = ev[["ent", "known_date", "cum_cnt", "cum_lvl", "cum_l4", "cum_l3p"]].rename(columns={"known_date": "t"})
    deb = hist.sort_values("debut_date").copy()
    deb["cum_deb"] = deb.groupby("ent").cumcount() + 1
    deb = deb[["ent", "debut_date", "cum_deb"]].rename(columns={"debut_date": "t"})

    q = X[[COL_HORSE, entity, "base_date"]].rename(columns={entity: "ent", "base_date": "t"}).copy()
    q["ent"] = q["ent"].astype(object)
    q = q.sort_values("t")
    ev["ent"] = ev["ent"].astype(object)
    deb["ent"] = deb["ent"].astype(object)
    # known_date ≤ base_date（同日は含めない: allow_exact_matches=False で厳密に前）
    a = pd.merge_asof(q, ev.sort_values("t"), on="t", by="ent", allow_exact_matches=False, direction="backward")
    b = pd.merge_asof(q, deb.sort_values("t"), on="t", by="ent", allow_exact_matches=False, direction="backward")
    a = a.set_index(COL_HORSE)
    b = b.set_index(COL_HORSE)
    # 事前分布は train 期間の全体率（学習データからのみ算出）
    tr = X[X["split"] == "train"]
    g_lvl, g_l4, g_l3p = tr["y_level"].mean(), (tr["y_level"] >= 4).mean(), (tr["y_level"] >= 3).mean()
    n = a["cum_cnt"].fillna(0)
    out = pd.DataFrame(index=X[COL_HORSE])
    out[f"{prefix}_n_known"] = n.reindex(out.index).values
    out[f"{prefix}_te_lvl"] = ((a["cum_lvl"].fillna(0) + m * g_lvl) / (n + m)).reindex(out.index).values
    out[f"{prefix}_te_l4"] = ((a["cum_l4"].fillna(0) + m * g_l4) / (n + m)).reindex(out.index).values
    out[f"{prefix}_te_l3p"] = ((a["cum_l3p"].fillna(0) + m * g_l3p) / (n + m)).reindex(out.index).values
    out[f"{prefix}_n_debut"] = b["cum_deb"].fillna(0).reindex(out.index).values
    return X.merge(out.reset_index(), on=COL_HORSE, how="left")


CAT_COLS = ["sex", "sire_type", "bms_type"]
DROP_COLS = {COL_HORSE, "馬名", "base_date", "trainer", "trainer_debut", "sire", "breeder", "y_level", "y_graded24",
             "h24_complete", "n_starts_12m", "rule_level", "rule_score", "split", "debut_year", "year_base"}


def feature_cols(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if c not in DROP_COLS]


def prep_matrix(X: pd.DataFrame, feats: list[str], cats: dict | None = None):
    M = X[feats].copy()
    if cats is None:
        cats = {c: pd.CategoricalDtype(sorted(M[c].dropna().astype(str).unique())) for c in CAT_COLS}
    for c in CAT_COLS:
        M[c] = M[c].astype(str).astype(cats[c])
    return M, cats


# --------------------------------------------------------------------------- metrics
def qwk(y, p) -> float:
    return float(cohen_kappa_score(y, p, weights="quadratic"))


def cum_auc(y_level: np.ndarray, prob: np.ndarray) -> dict:
    """P(level ≥ k) の AUC（k=1..4）。prob は (n, 5) の確率行列。"""
    out = {}
    for k in range(1, N_LEVELS):
        yk = (y_level >= k).astype(int)
        pk = prob[:, k:].sum(axis=1)
        out[f"auc_ge_L{k}"] = float(roc_auc_score(yk, pk)) if 0 < yk.mean() < 1 else np.nan
    return out


def cum_auc_score(y_level: np.ndarray, score: np.ndarray) -> dict:
    out = {}
    for k in range(1, N_LEVELS):
        yk = (y_level >= k).astype(int)
        out[f"auc_ge_L{k}"] = float(roc_auc_score(yk, score)) if 0 < yk.mean() < 1 else np.nan
    return out


def fit_cutpoints(expected: np.ndarray, y: np.ndarray) -> list[float]:
    """期待レベル E[L] を順序カテゴリへ丸める閾値を QWK 最大化で座標降下的に決める（valid で fit）。"""
    cuts = [0.5, 1.5, 2.5, 3.5]
    grid = np.arange(0.05, 4.0, 0.05)
    best = qwk(y, np.digitize(expected, cuts))
    for _ in range(3):
        for i in range(4):
            for v in grid:
                lo = cuts[i - 1] if i > 0 else -np.inf
                hi = cuts[i + 1] if i < 3 else np.inf
                if not (lo < v < hi):
                    continue
                trial = cuts.copy()
                trial[i] = float(v)
                s = qwk(y, np.digitize(expected, trial))
                if s > best + 1e-9:
                    best, cuts = s, trial
    return cuts


def bootstrap_ci(fn, *arrays, n: int = 300, seed: int = SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    N = len(arrays[0])
    vals = []
    for _ in range(n):
        idx = rng.integers(0, N, N)
        try:
            vals.append(fn(*[a[idx] for a in arrays]))
        except ValueError:
            continue
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    q = pd.qcut(p, bins, labels=False, duplicates="drop")
    t = pd.DataFrame({"y": y, "p": p, "bin": q}).groupby("bin").agg(n=("y", "size"), pred_mean=("p", "mean"), obs_rate=("y", "mean"))
    t["gap"] = t["obs_rate"] - t["pred_mean"]
    return t.reset_index()


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    t = calibration_table(y, p, bins)
    return float((t["n"] / t["n"].sum() * t["gap"].abs()).sum())


# --------------------------------------------------------------------------- main
def main() -> None:
    starts = pd.read_parquet(OUT / "starts.parquet")
    career = pd.read_parquet(OUT / "career.parquet")
    X = build_dataset(starts, career)
    X = add_asof_te(X, career, "trainer_debut", "trn")
    X = add_asof_te(X, career, "sire", "sire")
    X = add_asof_te(X, career, "breeder", "brd")
    X.to_parquet(OUT / "dataset.parquet", index=False)
    feats = feature_cols(X)
    print("rows", len(X), "feats", len(feats))
    print(X.groupby("split")["y_level"].value_counts(normalize=True).unstack().round(3))

    tr, va, te = (X[X["split"] == s].reset_index(drop=True) for s in ("train", "valid", "test"))
    Mtr, cats = prep_matrix(tr, feats)
    Mva, _ = prep_matrix(va, feats, cats)
    Mte, _ = prep_matrix(te, feats, cats)
    metrics: dict = {"n": {s: int(len(d)) for s, d in zip(("train", "valid", "test"), (tr, va, te))},
                     "level_dist_test": te["y_level"].value_counts(normalize=True).sort_index().round(4).to_dict(),
                     "features": feats}

    # ---- A: 規則ベースライン
    for name, d in (("valid", va), ("test", te)):
        y = d["y_level"].values
        metrics[f"rule_A_{name}"] = {"qwk": qwk(y, d["rule_level"].values), **cum_auc_score(y, d["rule_score"].values)}

    # ---- A: LightGBM multiclass
    params_A = dict(objective="multiclass", num_class=N_LEVELS, learning_rate=0.03, num_leaves=31,
                    min_data_in_leaf=60, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                    lambda_l2=1.0, verbose=-1, seed=SEED)
    dtr = lgb.Dataset(Mtr, tr["y_level"], categorical_feature=CAT_COLS)
    dva = lgb.Dataset(Mva, va["y_level"], categorical_feature=CAT_COLS, reference=dtr)
    mA = lgb.train(params_A, dtr, num_boost_round=3000, valid_sets=[dva],
                   callbacks=[lgb.early_stopping(100, verbose=False)])
    mA.save_model(str(OUT / "lgbm_A.txt"))
    metrics["lgbm_A_best_iter"] = int(mA.best_iteration)
    Pva = mA.predict(Mva, num_iteration=mA.best_iteration)
    Pte = mA.predict(Mte, num_iteration=mA.best_iteration)
    lv = np.arange(N_LEVELS)
    cuts = fit_cutpoints(Pva @ lv, va["y_level"].values)
    metrics["lgbm_A_cutpoints"] = cuts
    for name, d, P in (("valid", va, Pva), ("test", te, Pte)):
        y = d["y_level"].values
        pred_argmax = P.argmax(axis=1)
        pred_cut = np.digitize(P @ lv, cuts)
        res = {"qwk_argmax": qwk(y, pred_argmax), "qwk_expected_cut": qwk(y, pred_cut),
               "multi_logloss": float(log_loss(y, P, labels=lv)), **cum_auc(y, P)}
        if name == "test":
            res["qwk_expected_cut_ci95"] = bootstrap_ci(qwk, y, pred_cut)
            res["auc_ge_L4_ci95"] = bootstrap_ci(lambda a, b: roc_auc_score((a >= 4).astype(int), b), y, P[:, 4])
            res["rule_qwk_ci95"] = bootstrap_ci(qwk, y, d["rule_level"].values)
            res["rule_auc_ge_L4_ci95"] = bootstrap_ci(lambda a, b: roc_auc_score((a >= 4).astype(int), b), y, d["rule_score"].values)
            # 混同行列（期待値丸め）
            metrics["confusion_test_expected_cut"] = pd.crosstab(pd.Series(y, name="actual"), pd.Series(pred_cut, name="pred")).reindex(index=lv, columns=lv).fillna(0).astype(int).values.tolist()
        metrics[f"lgbm_A_{name}"] = res
    imp = pd.DataFrame({"feature": feats, "gain": mA.feature_importance("gain"), "split": mA.feature_importance("split")}).sort_values("gain", ascending=False)
    imp["gain_share"] = (imp["gain"] / imp["gain"].sum()).round(4)
    imp.to_csv(OUT / "feature_importance_A.csv", index=False, encoding="utf-8-sig")

    # ---- B: 2 値（重賞 24 か月）。horizon が完了した馬のみ
    trB, vaB, teB = (d[d["h24_complete"]].reset_index(drop=True) for d in (tr, va, te))
    MtrB, _ = prep_matrix(trB, feats, cats)
    MvaB, _ = prep_matrix(vaB, feats, cats)
    MteB, _ = prep_matrix(teB, feats, cats)
    # 規則ベースライン B: train の (rule_level, n_wins3) 別経験率を確率として使う
    tab = trB.groupby(["rule_level", "n_wins3"])["y_graded24"].agg(["mean", "size"]).reset_index()
    gB = trB["y_graded24"].mean()
    tab["p_rule"] = (tab["mean"] * tab["size"] + 20 * gB) / (tab["size"] + 20)
    for name, d in (("valid", vaB), ("test", teB)):
        pr = d.merge(tab[["rule_level", "n_wins3", "p_rule"]], on=["rule_level", "n_wins3"], how="left")["p_rule"].fillna(gB).values
        y = d["y_graded24"].values
        metrics[f"rule_B_{name}"] = {"auc_score": float(roc_auc_score(y, d["rule_score"].values)),
                                     "auc_table_prob": float(roc_auc_score(y, pr)),
                                     "brier": float(brier_score_loss(y, pr)), "ece10": ece(y, pr), "base_rate": float(y.mean())}
        if name == "test":
            d = d.assign(p_rule_B=pr)
            teB = d
    params_B = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_data_in_leaf=60,
                    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=SEED)
    dtrB = lgb.Dataset(MtrB, trB["y_graded24"], categorical_feature=CAT_COLS)
    dvaB = lgb.Dataset(MvaB, vaB["y_graded24"], categorical_feature=CAT_COLS, reference=dtrB)
    mB = lgb.train(params_B, dtrB, num_boost_round=3000, valid_sets=[dvaB], callbacks=[lgb.early_stopping(100, verbose=False)])
    mB.save_model(str(OUT / "lgbm_B.txt"))
    metrics["lgbm_B_best_iter"] = int(mB.best_iteration)
    for name, d, M in (("valid", vaB, MvaB), ("test", teB, MteB)):
        y = d["y_graded24"].values
        p = mB.predict(M, num_iteration=mB.best_iteration)
        res = {"auc": float(roc_auc_score(y, p)), "brier": float(brier_score_loss(y, p)), "logloss": float(log_loss(y, p)),
               "ece10": ece(y, p), "base_rate": float(y.mean())}
        if name == "test":
            res["auc_ci95"] = bootstrap_ci(roc_auc_score, y, p)
            calibration_table(y, p).to_csv(OUT / "calibration_B.csv", index=False, encoding="utf-8-sig")
            teB = teB.assign(p_lgbm_B=p)
        metrics[f"lgbm_B_{name}"] = res
    impB = pd.DataFrame({"feature": feats, "gain": mB.feature_importance("gain")}).sort_values("gain", ascending=False)
    impB["gain_share"] = (impB["gain"] / impB["gain"].sum()).round(4)
    impB.to_csv(OUT / "feature_importance_B.csv", index=False, encoding="utf-8-sig")

    # ---- test 予測の保存
    pred = te[[COL_HORSE, "馬名", "base_date", "debut_year", "y_level", "y_graded24", "h24_complete", "rule_level", "rule_score"]].copy()
    for k in range(N_LEVELS):
        pred[f"p_L{k}"] = Pte[:, k]
    pred["expected_level"] = Pte @ lv
    pred["pred_level"] = np.digitize(pred["expected_level"], cuts)
    pred = pred.merge(teB[[COL_HORSE, "p_lgbm_B", "p_rule_B"]], on=COL_HORSE, how="left")
    pred.to_parquet(OUT / "pred_test.parquet", index=False)

    # ---- 読み物カード
    cards = make_cards(pred, X, starts, career)
    (OUT / "cards.json").write_text(json.dumps(cards, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (OUT / "cards.md").write_text(render_cards(cards), encoding="utf-8")

    (OUT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in metrics.items() if k not in ("features", "confusion_test_expected_cut")}, ensure_ascii=False, indent=1, default=float))
    print(imp.head(20).to_string(index=False))
    print(impB.head(12).to_string(index=False))


# --------------------------------------------------------------------------- cards
def make_cards(pred: pd.DataFrame, X: pd.DataFrame, starts: pd.DataFrame, career: pd.DataFrame) -> list[dict]:
    pick = pred[pred["馬名"].isin(CARD_NAMES)].copy()
    pick["why"] = "有名馬"
    rest = pred[~pred[COL_HORSE].isin(pick[COL_HORSE])]
    # 予測を裏切った出世（P(L4) が低いのに重賞到達）/ 期待外れ（P(L4) 高いのに未勝利圏）/ 的中した大物
    up = rest[rest["y_level"] == 4].nsmallest(2, "p_L4").assign(why="予測外の出世（P(重賞) 最小で重賞到達）")
    bust = rest[rest["y_level"] == 0].nlargest(2, "p_L4").assign(why="期待外れ（P(重賞) 最大で未勝利圏）")
    hit = rest[rest["y_level"] == 4].nlargest(1, "p_L4").assign(why="的中した大物（P(重賞) 最大で重賞到達）")
    pick = pd.concat([pick, up, bust, hit])
    car = career.set_index(COL_HORSE)
    cards = []
    for _, r in pick.iterrows():
        hid = r[COL_HORSE]
        s = starts[starts[COL_HORSE] == hid].sort_values("k")
        first3 = s[s["k"] <= 3]
        after = s[(s["date"] > r["base_date"]) & (s["date"] <= r["base_date"] + pd.Timedelta(days=365))]
        best_after = after.sort_values(["race_level", "pos"], ascending=[False, True]).head(1)
        cards.append({
            "馬名": r["馬名"], "why": r["why"], "debut_year": int(r["debut_year"]),
            "base_date": str(pd.Timestamp(r["base_date"]).date()),
            "first3": [{"date": str(x.date.date()), "class": x.クラス名, "race": str(x.レース名).strip(), "pos": int(x.pos),
                        "field": int(x.field), "margin": (None if pd.isna(x.margin_this) else float(x.margin_this))}
                       for x in first3.itertuples()],
            "pred_probs": {TARGET_LEVEL_NAME[k]: round(float(r[f"p_L{k}"]), 3) for k in range(N_LEVELS)},
            "expected_level": round(float(r["expected_level"]), 2),
            "pred_level": TARGET_LEVEL_NAME[int(r["pred_level"])],
            "rule_level": TARGET_LEVEL_NAME[int(r["rule_level"])],
            "actual_level": TARGET_LEVEL_NAME[int(r["y_level"])],
            "actual_best_race_12m": (None if best_after.empty else
                                     f"{best_after.iloc[0]['クラス名']} {str(best_after.iloc[0]['レース名']).strip()} {int(best_after.iloc[0]['pos'])}着 ({best_after.iloc[0]['date'].date()})"),
            "n_starts_12m": int(len(after)),
            "p_graded24_lgbm": (None if pd.isna(r.get("p_lgbm_B")) else round(float(r["p_lgbm_B"]), 3)),
            "p_graded24_rule": (None if pd.isna(r.get("p_rule_B")) else round(float(r["p_rule_B"]), 3)),
            "actual_graded24": (bool(r["y_graded24"]) if r["h24_complete"] else None),
            "career_max": RACE_LEVEL_NAME[int(car.loc[hid, "max_race_level"])],
            "career_starts": int(car.loc[hid, "n_starts"]),
        })
    return cards


def render_cards(cards: list[dict]) -> str:
    lines = ["# 出世予測 AI — 読み物カード（test 世代 = 2022 デビュー、3 走目時点の予測 vs 実際）", ""]
    for c in cards:
        lines.append(f"## {c['馬名']}（{c['why']}）")
        lines.append(f"- 基準日: {c['base_date']}（3 走目終了時）")
        for i, f in enumerate(c["first3"], 1):
            mg = "" if f["margin"] is None else ("、着差 0.0s" if abs(f["margin"]) < 0.05 else f"、着差 {f['margin']:+.1f}s")
            lines.append(f"  - {i}走目 {f['date']} {f['class']} {f['race']} {f['pos']}着/{f['field']}頭{mg}")
        pp = " / ".join(f"{k} {v:.0%}" for k, v in c["pred_probs"].items())
        lines.append(f"- 予測（12 か月後の最高クラス）: **{c['pred_level']}**（期待レベル {c['expected_level']}）  確率: {pp}")
        lines.append(f"- 規則ベースライン: {c['rule_level']}")
        lines.append(f"- 実際: **{c['actual_level']}** — {c['actual_best_race_12m'] or '12 か月以内の出走なし'}（12 か月で {c['n_starts_12m']} 走）")
        if c["p_graded24_lgbm"] is not None:
            lines.append(f"- 2 年以内に重賞出走: 予測 {c['p_graded24_lgbm']:.0%}（規則 {c['p_graded24_rule']:.0%}） → 実際 {'到達' if c['actual_graded24'] else '未到達'}")
        lines.append(f"- 通算: {c['career_starts']} 走、最高 {c['career_max']}")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
