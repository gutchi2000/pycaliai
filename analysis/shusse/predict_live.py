# -*- coding: utf-8 -*-
"""
analysis/shusse/predict_live.py — 出世予測 AI v2: 現役馬への適用
===================================================================
2023〜2026 デビュー（真のデビュー・年齢 2〜3・平地 1 走以上）の馬に、到達している基準点
k ∈ {1,2,3,5} ごとに学習済み 3 ヘッド（D 離脱 / A 到達クラス / B 重賞掲示板 24m）を適用する。
最新の基準点が本命の予測、過去の基準点（k=1,2,3）は連載の「更新履歴」用に残す。

前提: analysis.shusse.extend_2026 を実行済み（out_v2/live/starts_ext.parquet, horses_ext.parquet）。
実行（root から）:
    venv311\\Scripts\\python.exe -m analysis.shusse.predict_live [--asof YYYY-MM-DD]

出力（analysis/shusse/out_v2/live/）:
    predictions_{asof}.parquet / predictions_latest.parquet   馬×k の全行（確定分は実績ラベル付き）
    live_validation.json      確定分（主に 2023 デビュー）の前向き検証: AUC / QWK を test 2022 と並べる
    live_summary.json         頭数・k 別件数・確定/未確定・欠損率・前向き検証の要約
    live_top.md               デビュー年別 P(重賞掲示板 12m) 上位 30 頭（未確定・最新基準点のみ）
    live_sanity.md            2024〜2025 デビューの重賞勝ち馬の k=3 / k=1 時点予測（目視妥当性）
    shusse_live.json          サイト用（未確定・最新基準点の馬 1 行 + 更新履歴、小数 3 桁）

確定 / 未確定: 基準日 + 365 日 ≤ asof なら「確定」（12 か月窓が閉じている）。asof の既定は取り込んだ
データの最新日（kekka の最終日）。市場オッズ・人気は特徴に入れていない（features_v2 と同じ）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import unicodedata
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from analysis.shusse.build_career import COL_HORSE, RACE_LEVEL_NAME
from analysis.shusse.features_v2 import (
    KS, N_LEVELS, TARGET_LEVEL_NAME, build_features, build_targets, load_asof_tables, prep_matrix, save_json,
)
from analysis.shusse.train_v2 import LV, auc, boot_ci, qwk

BASE = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "out_v2"
LIVE = OUT / "live"
MODELS = OUT / "models"
CODE_MAPS = BASE / "data" / "serve_code_maps.json"
DEBUT_YEARS = (2023, 2024, 2025, 2026)
HEADS = ("A", "D", "B")
TOP_N_MD = 30
GENERIC_PREFIX = ("1勝", "2勝", "3勝", "500万", "1000万", "1600万", "オープン", "未勝利", "新馬", "OP")


# --------------------------------------------------------------------------- helpers
def _nfkc(s) -> str:
    return unicodedata.normalize("NFKC", str(s)).strip()


def race_label(race_name, class_norm, race_level: int) -> str:
    """1 走の短い表記。新馬/未勝利はクラス名、特別戦は「レース名(クラス)」、条件戦はクラス名。"""
    lvl_name = RACE_LEVEL_NAME[int(race_level)]
    if race_level <= 1:
        return _nfkc(class_norm)
    rn = _nfkc(race_name) if race_name is not None else ""
    rn = rn.replace("･", "・")
    changed = True
    while changed and rn:   # 末尾の記号・クラス表記を剥がす（例 '黄菊賞・1勝*' → '黄菊賞'）
        changed = False
        for suf in ("*", "・1勝", "・2勝", "・3勝", "・牝", "・若", "G1", "G2", "G3", "(L)", "H", "・"):
            if rn.endswith(suf) and len(rn) > len(suf):
                rn = rn[: -len(suf)].rstrip("・ ")
                changed = True
    if not rn or rn.startswith(GENERIC_PREFIX) or rn.endswith("クラス"):
        return lvl_name
    return f"{rn}({lvl_name})"


def career_summary(starts: pd.DataFrame, k: int) -> pd.Series:
    """馬ごとの k 走までの要約文字列（例 新馬3着→未勝利1着→山吹賞(1勝)1着）。"""
    s = starts[starts["k"] <= k].sort_values([COL_HORSE, "k"])
    lab = [f"{race_label(rn, cn, rl)}{int(p)}着" for rn, cn, rl, p in zip(s["レース名"], s["class_norm"], s["race_level"], s["pos"])]
    return pd.Series(lab, index=s[COL_HORSE].values).groupby(level=0).agg("→".join)


def load_k(k: int):
    spec = json.loads((OUT / f"model_spec_k{k}.json").read_text(encoding="utf-8"))
    tables = load_asof_tables(OUT, k)
    boosters = {h: lgb.Booster(model_file=str(MODELS / f"lgbm_{h}_k{k}.txt")) for h in HEADS}
    return spec, tables, boosters


def top_contrib(m: lgb.Booster, M: pd.DataFrame, feats: list[str], labels: dict[str, str], cls: int | None = None, n: int = 3) -> list[list]:
    """各行の寄与上位 n（符号付き）を [[日本語ラベル, 寄与], ...] で返す。multiclass は cls のブロック。"""
    C = m.predict(M, pred_contrib=True)
    F = len(feats)
    if cls is not None:
        C = C[:, cls * (F + 1):(cls + 1) * (F + 1)]
    C = C[:, :F]
    out = []
    for row in C:
        idx = np.argsort(-np.abs(row))[:n]
        out.append([[labels.get(feats[i], feats[i]), round(float(row[i]), 4)] for i in idx])
    return out


def trainer_names() -> dict[int, str]:
    maps = json.loads(CODE_MAPS.read_text(encoding="utf-8"))
    return {int(v): k for k, v in maps["trainer"].items()}


# --------------------------------------------------------------------------- per k
def run_k(starts: pd.DataFrame, horses: pd.DataFrame, k: int, asof: pd.Timestamp, labels: dict[str, str], log: dict) -> pd.DataFrame:
    t0 = time.time()
    spec, tables, mb = load_k(k)
    feats, cats, priors = spec["features"], spec["categorical"], spec["te_priors"]
    cuts = spec["heads"]["A"]["cutpoints"]
    h = horses[horses["true_debut"] & horses["debut_age"].isin([2, 3]) & horses["debut_year"].isin(DEBUT_YEARS) & (horses["n_starts"] >= k)]
    ids = h[COL_HORSE]
    X = build_features(starts, horses, k, tables, priors, horse_ids=ids)
    missing = sorted(set(feats) - set(X.columns))
    assert not missing, f"k={k}: features missing from build_features: {missing}"
    M, _ = prep_matrix(X, feats, cats)
    P = mb["A"].predict(M)
    pD = mb["D"].predict(M)
    pB = mb["B"].predict(M)
    exp_lvl = P @ LV
    tg = build_targets(starts, k, data_end=asof).set_index(COL_HORSE).reindex(X[COL_HORSE])
    hx = horses.set_index(COL_HORSE).reindex(X[COL_HORSE])
    summ = career_summary(starts[starts[COL_HORSE].isin(X[COL_HORSE])], k).reindex(X[COL_HORSE])
    out = pd.DataFrame({
        COL_HORSE: X[COL_HORSE].values,
        "馬名": X["馬名"].values,
        "debut_year": X["debut_year"].astype(int).values,
        "debut_date": hx["debut_date"].values,
        "性別": X["sex"].values,
        "父": X["sire"].values,
        "調教師コード": X["trainer_debut"].values,
        "k": k,
        "base_date": pd.to_datetime(X["base_date"]).values,
        "asof": asof,
        "n_starts_total": hx["n_starts"].astype(int).values,
        "summary": summ.values,
        "p_dropout": pD,
        **{f"p_L{c}": P[:, c] for c in range(N_LEVELS)},
        "expected_level": exp_lvl,
        "pred_level": np.digitize(exp_lvl, cuts),
        "p_graded12": (1 - pD) * P[:, 4],
        "p_graded24": pB,
        "p_L3plus": P[:, 3] + P[:, 4],
        "rule_level": X["rule_level"].values,
        "rule_score": X["rule_score"].values,
        "top3_A_L4": [json.dumps(v, ensure_ascii=False) for v in top_contrib(mb["A"], M, feats, labels, cls=4)],
        "top3_D": [json.dumps(v, ensure_ascii=False) for v in top_contrib(mb["D"], M, feats, labels)],
        "top3_B": [json.dumps(v, ensure_ascii=False) for v in top_contrib(mb["B"], M, feats, labels)],
        "confirmed": (pd.to_datetime(X["base_date"]) + pd.Timedelta(days=365) <= asof).values,
        "confirmed24": (pd.to_datetime(X["base_date"]) + pd.Timedelta(days=730) <= asof).values,
        "actual_ran_12m": tg["ran_12m"].values,
        "actual_n_starts_12m": tg["n_starts_12m"].values,
        "actual_level": tg["y_level"].values,
        "actual_graded12": tg["y_graded12"].values,
        "actual_graded24": tg["y_graded24"].values,
        "sire_type_missing": X["sire_type"].isna().values,
        "bms_type_missing": X["bms_type"].isna().values,
        "margin_k_missing": X[f"margin_{k}"].isna().values,
        "weight_k_missing": X[f"weight_{k}"].isna().values,
        "debut_dayage_missing": X["debut_dayage"].isna().values,
    })
    # 未確定分は実績を伏せる（窓が閉じていない）
    for c in ("actual_ran_12m", "actual_n_starts_12m", "actual_level", "actual_graded12"):
        out.loc[~out["confirmed"], c] = np.nan
    out.loc[~out["confirmed24"], "actual_graded24"] = np.nan
    out["actual_level"] = out["actual_level"].where(out["actual_ran_12m"] == 1)
    out["pred_level_name"] = out["pred_level"].map(TARGET_LEVEL_NAME)
    out["percentile_graded12"] = out.groupby("debut_year")["p_graded12"].rank(pct=True).mul(100).round(1)
    log[f"k{k}"] = {
        "n": int(len(out)), "n_confirmed": int(out["confirmed"].sum()), "n_unconfirmed": int((~out["confirmed"]).sum()),
        "by_debut_year": {int(y): int(n) for y, n in out["debut_year"].value_counts().sort_index().items()},
        "missing_rate": {c: round(float(out[c].mean()), 4) for c in ("sire_type_missing", "bms_type_missing", "margin_k_missing", "weight_k_missing", "debut_dayage_missing")},
        "n_features": len(feats), "elapsed_sec": round(time.time() - t0, 1),
    }
    print(f"[k={k}] n={len(out)} confirmed={out['confirmed'].sum()} missing={log[f'k{k}']['missing_rate']} {log[f'k{k}']['elapsed_sec']}s", flush=True)
    return out


# --------------------------------------------------------------------------- validation
def _cum_auc_L4(y, P4):
    return auc((y >= 4).astype(int), P4)


def validate(pred: pd.DataFrame, metrics_v2: dict) -> dict:
    """確定分の前向き検証。test 2022（学習時）と同じ指標を並べる。"""
    res: dict = {"definition": "confirmed = base_date + 365 <= asof; QWK は valid で決めた閾値による期待レベル丸め（走った馬のみ）",
                 "by_cohort_k": {}}
    c = pred[pred["confirmed"]]
    for (y, k), d in c.groupby(["debut_year", "k"]):
        if len(d) < 100:
            continue
        spec = json.loads((OUT / f"model_spec_k{k}.json").read_text(encoding="utf-8"))
        cuts = spec["heads"]["A"]["cutpoints"]
        yd = d["actual_ran_12m"].eq(0).astype(int).values
        g12 = d["actual_graded12"].astype(int).values
        ran = d["actual_ran_12m"].eq(1).values
        yl = d.loc[ran, "actual_level"].astype(int).values
        P = d.loc[ran, [f"p_L{i}" for i in range(N_LEVELS)]].values
        pred_cut = np.digitize(P @ LV, cuts)
        r = {"n": int(len(d)), "n_ran": int(ran.sum()), "dropout_rate": float(yd.mean()), "graded12_rate": float(g12.mean()),
             "level_dist_given_ran": {int(a): round(float(b), 4) for a, b in pd.Series(yl).value_counts(normalize=True).sort_index().items()},
             "G12_auc": auc(g12, d["p_graded12"].values), "G12_auc_ci95": boot_ci(auc, g12, d["p_graded12"].values),
             "D_auc": auc(yd, d["p_dropout"].values),
             "A_qwk_expected_cut": qwk(yl, pred_cut), "A_qwk_ci95": boot_ci(qwk, yl, pred_cut),
             "A_qwk_argmax": qwk(yl, P.argmax(axis=1)),
             "A_auc_L4": _cum_auc_L4(yl, P[:, 4]), "A_auc_L4_ci95": boot_ci(_cum_auc_L4, yl, P[:, 4]),
             "A_auc_ge_L1": auc((yl >= 1).astype(int), P[:, 1:].sum(axis=1)),
             "rule_qwk": qwk(yl, d.loc[ran, "rule_level"].astype(int).values),
             "rule_auc_L4_score": _cum_auc_L4(yl, d.loc[ran, "rule_score"].values)}
        d24 = d[d["confirmed24"] & d["actual_graded24"].notna()]
        if len(d24) >= 100 and d24["actual_graded24"].nunique() > 1:
            r["B_n"] = int(len(d24))
            r["B_auc"] = auc(d24["actual_graded24"].astype(int).values, d24["p_graded24"].values)
        res["by_cohort_k"].setdefault(str(int(y)), {})[f"k{k}"] = r
    ref = {}
    for k in KS:
        mk = metrics_v2.get(f"k{k}", {})
        ref[f"k{k}"] = {
            "test2022": {"G12_auc": mk.get("G12", {}).get("lgbm_test", {}).get("auc"),
                         "G12_auc_ci95": mk.get("G12", {}).get("lgbm_test", {}).get("auc_ci95"),
                         "A_qwk_expected_cut": mk.get("A", {}).get("lgbm_test", {}).get("qwk_expected_cut"),
                         "A_qwk_ci95": mk.get("A", {}).get("lgbm_test", {}).get("qwk_expected_cut_ci95"),
                         "A_auc_L4": mk.get("A", {}).get("lgbm_test", {}).get("auc_ge_L4"),
                         "D_auc": mk.get("D", {}).get("lgbm_test", {}).get("auc"),
                         "B_auc": mk.get("B", {}).get("lgbm_test", {}).get("auc"),
                         "rule_qwk": mk.get("A", {}).get("rule_test", {}).get("qwk")},
            "test2023_partial_at_training": {"A_qwk_expected_cut": mk.get("A", {}).get("lgbm_test2023_partial", {}).get("qwk_expected_cut"),
                                             "G12_auc": mk.get("G12", {}).get("lgbm_test2023_partial", {}).get("auc"),
                                             "n": mk.get("n", {}).get("test2023_partial")}}
    res["reference_from_training"] = ref
    return res


# --------------------------------------------------------------------------- reports
def _fmt_reasons(js: str) -> str:
    return " / ".join(f"{a}({'+' if b > 0 else ''}{b:.2f})" for a, b in json.loads(js))


def write_top_md(pred: pd.DataFrame, asof: pd.Timestamp, tnames: dict[int, str]) -> str:
    lines = [f"# 出世予測 AI v2 — 現役馬 P(重賞掲示板 12m) 上位（asof {asof.date()}）", "",
             "未確定（基準日 + 365 日 > asof）かつ各馬の最新基準点のみ。P(重賞) = (1 − P(離脱)) · P(L4 | 走る)。",
             "根拠 = L4 クラス確率への SHAP 寄与上位 3（符号付き、log-odds）。", ""]
    live = pred[~pred["confirmed"] & pred["is_latest"]]
    for y, d in live.groupby("debut_year"):
        d = d.sort_values("p_graded12", ascending=False).head(TOP_N_MD)
        lines += [f"## {int(y)} デビュー（未確定 {int((~pred['confirmed'] & pred['is_latest'] & (pred['debut_year'] == y)).sum())} 頭）", "",
                  "| # | 馬名 | 性 | 父 | 調教師 | k | 経過 | P(重賞) | P(離脱) | 予測クラス | 根拠上位 3 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for i, r in enumerate(d.itertuples(index=False), 1):
            tn = tnames.get(int(r.調教師コード), "—")
            lines.append(f"| {i} | {r.馬名} | {r.性別} | {r.父} | {tn} | {r.k} | {r.summary} | {r.p_graded12:.3f} | {r.p_dropout:.3f} | {r.pred_level_name} | {_fmt_reasons(r.top3_A_L4)} |")
        lines.append("")
    return "\n".join(lines)


def write_sanity_md(pred: pd.DataFrame, starts: pd.DataFrame, horses: pd.DataFrame, asof: pd.Timestamp) -> tuple[str, dict]:
    """2024〜2025 デビューの重賞勝ち馬について、k=3（と k=1）時点の予測を並べる。"""
    hs = horses.set_index(COL_HORSE)
    gw = starts[(starts["race_level"] >= 6) & (starts["pos"] == 1)].copy()
    gw["debut_year"] = hs["debut_year"].reindex(gw[COL_HORSE]).values
    gw["true_debut"] = hs["true_debut"].reindex(gw[COL_HORSE]).values
    gw = gw[gw["true_debut"] & gw["debut_year"].isin([2024, 2025])].sort_values("date")
    first_win = gw.drop_duplicates(COL_HORSE, keep="first")
    p3 = pred[pred["k"] == 3].set_index(COL_HORSE)
    p1 = pred[pred["k"] == 1].set_index(COL_HORSE)
    lines = [f"# 妥当性チェック — 2024〜2025 デビューの重賞勝ち馬の k=3 / k=1 時点予測（asof {asof.date()}）", "",
             "重賞 1 着（2025〜2026 の走、master + 2026 取り込み）を 1 頭 1 行。P(重賞) は基準日時点の (1 − P(離脱)) · P(L4 | 走る)。",
             "分位 = 同デビュー年の全対象馬の中での P(重賞) の百分位（k=3）。「既知」= 基準日（3 走目）の時点で既にその重賞を勝っていた。", ""]
    stats: dict = {}
    for y in (2024, 2025):
        d = first_win[first_win["debut_year"] == y]
        rows = []
        for r in d.itertuples(index=False):
            hid = getattr(r, COL_HORSE) if hasattr(r, COL_HORSE) else r[4]
            a = p3.loc[hid] if hid in p3.index else None
            b = p1.loc[hid] if hid in p1.index else None
            rows.append({"馬名": r.馬名, "重賞": race_label(r.レース名, r.class_norm, r.race_level), "勝利日": str(r.date.date()),
                         "k3_base": None if a is None else str(pd.Timestamp(a["base_date"]).date()),
                         "k3_p": None if a is None else float(a["p_graded12"]), "k3_pct": None if a is None else float(a["percentile_graded12"]),
                         "k3_pred": None if a is None else a["pred_level_name"], "k3_known": None if a is None else bool(pd.Timestamp(a["base_date"]) >= r.date),
                         "k1_p": None if b is None else float(b["p_graded12"]), "k1_pct": None if b is None else float(b["percentile_graded12"])})
        df = pd.DataFrame(rows)
        n_k3 = int(df["k3_p"].notna().sum()) if len(df) else 0
        unk = df[(df["k3_known"] == False)] if len(df) else df
        stats[str(y)] = {"graded_winners": int(len(df)), "with_k3_pred": n_k3,
                         "k3_not_yet_won_at_base": int(len(unk)),
                         "k3_pct_median_not_yet_won": None if not len(unk) else float(unk["k3_pct"].median()),
                         "k3_pct_ge90_share_not_yet_won": None if not len(unk) else float((unk["k3_pct"] >= 90).mean()),
                         "k1_pct_median": None if not len(df) or df["k1_pct"].isna().all() else float(df["k1_pct"].median())}
        lines += [f"## {y} デビュー（重賞勝ち馬 {len(df)} 頭、k=3 予測あり {n_k3}、基準日時点で未勝利の重賞 {len(unk)} 頭: 分位中央値 {stats[str(y)]['k3_pct_median_not_yet_won']}, 上位 10% 内 {stats[str(y)]['k3_pct_ge90_share_not_yet_won']}）", "",
                  "| 馬名 | 重賞 | 勝利日 | k=3 基準日 | k=3 P(重賞) | k=3 分位 | k=3 予測 | 既知 | k=1 P(重賞) | k=1 分位 |", "|---|---|---|---|---|---|---|---|---|---|"]
        for r in df.head(40).itertuples(index=False):
            f = lambda v, fmt: "—" if v is None or (isinstance(v, float) and np.isnan(v)) else fmt.format(v)
            lines.append(f"| {r.馬名} | {r.重賞} | {r.勝利日} | {f(r.k3_base, '{}')} | {f(r.k3_p, '{:.3f}')} | {f(r.k3_pct, '{:.1f}')} | {f(r.k3_pred, '{}')} | {'既知' if r.k3_known else ('' if r.k3_known is None else '未')} | {f(r.k1_p, '{:.3f}')} | {f(r.k1_pct, '{:.1f}')} |")
        lines.append("")
    return "\n".join(lines), stats


def site_json(pred: pd.DataFrame, asof: pd.Timestamp, tnames: dict[int, str]) -> dict:
    """サイト用: 未確定・最新基準点の馬 1 行 + 過去基準点の履歴。"""
    live = pred[~pred["confirmed"] & pred["is_latest"]].sort_values("p_graded12", ascending=False)
    hist = pred[pred[COL_HORSE].isin(live[COL_HORSE])].sort_values([COL_HORSE, "k"])
    hmap: dict[int, list] = {}
    for r in hist.itertuples(index=False):
        # [k, base_date, p_graded12, p_dropout, pred_level]
        hmap.setdefault(int(getattr(r, COL_HORSE)), []).append(
            [int(r.k), str(pd.Timestamp(r.base_date).date()), round(float(r.p_graded12), 3), round(float(r.p_dropout), 3), int(r.pred_level)])
    rows = []
    for r in live.itertuples(index=False):
        hid = int(getattr(r, COL_HORSE))
        rows.append({
            "ped_id": hid, "name": r.馬名, "debut_year": int(r.debut_year), "sex": r.性別, "sire": r.父,
            "trainer_code": int(r.調教師コード), "trainer": tnames.get(int(r.調教師コード)),
            "n_starts": int(r.n_starts_total), "k": int(r.k), "base_date": str(pd.Timestamp(r.base_date).date()),
            "summary": r.summary, "p_dropout": round(float(r.p_dropout), 3),
            "p_level": [round(float(getattr(r, f"p_L{i}")), 3) for i in range(N_LEVELS)],
            "expected_level": round(float(r.expected_level), 2), "pred_level": int(r.pred_level),
            "p_graded12": round(float(r.p_graded12), 3), "p_graded24": round(float(r.p_graded24), 3), "p_L3plus": round(float(r.p_L3plus), 3),
            "percentile_graded12": float(r.percentile_graded12), "rule_level": int(r.rule_level),
            "reasons_L4": [[a, round(b, 2)] for a, b in json.loads(r.top3_A_L4)],
            "reasons_dropout": [a for a, _ in json.loads(r.top3_D)],
            "history": hmap.get(hid, []),
        })
    return {"model": "shusse_v2", "asof": str(asof.date()), "generated_at": pd.Timestamp.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "levels": TARGET_LEVEL_NAME, "n_horses": len(rows),
            "note": "p_graded12 = (1-p_dropout)*p_level[4]; 未確定（12か月窓が閉じていない）馬のみ。history = [k, base_date, p_graded12, p_dropout, pred_level]。市場オッズ不使用。",
            "horses": rows}


# --------------------------------------------------------------------------- main
def main(asof_arg: str | None = None) -> None:
    T0 = time.time()
    starts = pd.read_parquet(LIVE / "starts_ext.parquet")
    horses = pd.read_parquet(LIVE / "horses_ext.parquet")
    asof = pd.Timestamp(asof_arg) if asof_arg else starts["date"].max()
    labels = json.loads((OUT / "feature_labels.json").read_text(encoding="utf-8"))
    metrics_v2 = json.loads((OUT / "metrics_v2.json").read_text(encoding="utf-8"))
    tnames = trainer_names()
    log: dict = {"asof": str(asof.date()), "data_end_ext": str(starts["date"].max().date()), "debut_years": list(DEBUT_YEARS)}
    preds = [run_k(starts, horses, k, asof, labels, log) for k in KS]
    pred = pd.concat(preds, ignore_index=True)
    # 最新の基準点: その馬が到達している最大の k
    kmax = pred.groupby(COL_HORSE)["k"].transform("max")
    pred["is_latest"] = pred["k"] == kmax
    pred = pred.sort_values([COL_HORSE, "k"]).reset_index(drop=True)
    tag = asof.strftime("%Y%m%d")
    pred.to_parquet(LIVE / f"predictions_{tag}.parquet", index=False)
    pred.to_parquet(LIVE / "predictions_latest.parquet", index=False)

    val = validate(pred, metrics_v2)
    save_json(val, LIVE / "live_validation.json")
    (LIVE / "live_top.md").write_text(write_top_md(pred, asof, tnames), encoding="utf-8")
    sanity_md, sanity_stats = write_sanity_md(pred, starts, horses, asof)
    (LIVE / "live_sanity.md").write_text(sanity_md, encoding="utf-8")
    sj = site_json(pred, asof, tnames)
    (LIVE / "shusse_live.json").write_text(json.dumps(sj, ensure_ascii=False, separators=(",", ":"), default=str), encoding="utf-8")

    latest = pred[pred["is_latest"]]
    log["horses"] = int(latest[COL_HORSE].nunique())
    log["rows_all_k"] = int(len(pred))
    log["latest_k_counts"] = {int(k): int(v) for k, v in latest["k"].value_counts().sort_index().items()}
    log["latest_by_debut_year"] = {int(y): {"n": int(len(d)), "confirmed": int(d["confirmed"].sum()), "unconfirmed": int((~d["confirmed"]).sum())}
                                   for y, d in latest.groupby("debut_year")}
    log["site_json_horses"] = sj["n_horses"]
    log["site_json_bytes"] = (LIVE / "shusse_live.json").stat().st_size
    log["forward_validation"] = val["by_cohort_k"]
    log["reference_test2022"] = val["reference_from_training"]
    log["sanity_graded_winners"] = sanity_stats
    log["elapsed_sec"] = round(time.time() - T0, 1)
    save_json(log, LIVE / "live_summary.json")
    print(json.dumps({k: v for k, v in log.items() if k not in ("forward_validation", "reference_test2022")}, ensure_ascii=False, indent=1, default=str))
    for y, kk in val["by_cohort_k"].items():
        for k, r in kk.items():
            ref = val["reference_from_training"][k]["test2022"]
            print(f"  debut {y} {k}: n={r['n']} G12 AUC {r['G12_auc']:.3f} {r['G12_auc_ci95']} (test2022 {ref['G12_auc']:.3f}) | "
                  f"QWK {r['A_qwk_expected_cut']:.3f} {r['A_qwk_ci95']} (test2022 {ref['A_qwk_expected_cut']:.3f}, rule {r['rule_qwk']:.3f}) | "
                  f"AUC L4 {r['A_auc_L4']:.3f} (test {ref['A_auc_L4']:.3f}) | D AUC {r['D_auc']:.3f} (test {ref['D_auc']:.3f})"
                  + (f" | B AUC {r['B_auc']:.3f} (test {ref['B_auc']:.3f})" if "B_auc" in r else ""))
    print(f"total {log['elapsed_sec']}s")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default=None, help="推論日 YYYY-MM-DD（既定: 取り込んだデータの最新日）")
    a = ap.parse_args()
    main(a.asof)
