# -*- coding: utf-8 -*-
"""
analysis/shusse/site_export.py — 出世予測 AI v2 のサイト用軽量 JSON
=====================================================================
predict_live.py の出力（predictions_latest.parquet / live_validation.json / starts_ext.parquet）から
公開サイト（site/shusse.html）が読む 1 ファイルを作る。

    analysis/shusse/out_v2/live/site/index.json     メタ・件数・検証数値（軽量、ページ初期表示用）
    analysis/shusse/out_v2/live/site/{debut_year}.json  その年デビュー馬の予測行 + 文章カード
        → build_site.py が site/data/shusse/ にコピー（鮮度が新しいときだけ）
  デビュー年ごとに分けるのは 1 ファイル 4MB 超を避けるため（タブ切替時に遅延読込）。

shusse_live.json（report_v2.py が読む完全版）とは別ファイル。こちらは
  - 未確定（12 か月窓が閉じていない）かつ各馬の最新基準点の 1 行だけ
  - 根拠は SHAP 上位 3 を narrate.direction_phrase で日本語句に変換済み（label+数値は持たない）
  - デビュー年ごとの上位 TOP_CARDS 頭だけ narrate.make_card の文章カード（text / runs）を同梱
  - 検証数値（test 2022 と 2023 デビュー世代の前向き）を validation に同梱
市場オッズ・人気は一切含まない（学習にも使っていない）。

実行:
    ./venv311/Scripts/python.exe -m analysis.shusse.site_export
    （predict_live.main() の最後からも呼ばれる）
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.shusse.build_career import COL_HORSE
from analysis.shusse.narrate import (
    LEVEL_NAME, _HEAD_GOOD, direction_phrase, load_trainer_names, make_card,
)

OUT = Path(__file__).resolve().parent / "out_v2"
LIVE = OUT / "live"
DST = LIVE / "site"
TOP_CARDS = 30           # デビュー年ごとの文章カード数
SHOW_YEARS = (2023, 2024, 2025, 2026)


def _r(x, nd=3):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


def _reasons(js: str, labels: dict[str, str], inv: dict[str, str], head: str, n: int = 3) -> list[str]:
    """predict_live の top3（[ラベル, 寄与] の JSON）→ 向き付き日本語句。"""
    out: list[str] = []
    try:
        pairs = json.loads(js) if isinstance(js, str) else (js or [])
    except Exception:
        return out
    for lab, c in pairs[:n]:
        f = lab if lab in labels else inv.get(lab, lab)
        phrase, _ = direction_phrase(f, float(c), labels, _HEAD_GOOD.get(head, True))
        if phrase and phrase not in out:
            out.append(phrase)
    return out


def _validation(val: dict) -> dict:
    """test 2022（学習時）と 2023 デビュー世代（前向き）の要約。"""
    ref = val.get("reference_from_training", {})
    fwd = val.get("by_cohort_k", {}).get("2023", {})
    rows = {}
    for k in ("k1", "k2", "k3", "k5"):
        t = ref.get(k, {}).get("test2022", {})
        f = fwd.get(k, {})
        rows[k] = {
            "test2022": {"qwk": _r(t.get("A_qwk_expected_cut")), "rule_qwk": _r(t.get("rule_qwk")),
                         "g12_auc": _r(t.get("G12_auc")), "d_auc": _r(t.get("D_auc"))},
            "forward2023": {"n": f.get("n"), "qwk": _r(f.get("A_qwk_expected_cut")), "qwk_ci": [_r(x) for x in f.get("A_qwk_ci95", [])] or None,
                            "rule_qwk": _r(f.get("rule_qwk")), "g12_auc": _r(f.get("G12_auc")),
                            "g12_auc_ci": [_r(x) for x in f.get("G12_auc_ci95", [])] or None, "d_auc": _r(f.get("D_auc")),
                            "graded12_rate": _r(f.get("graded12_rate")), "dropout_rate": _r(f.get("dropout_rate"))},
        }
    return rows


def build(write: bool = True) -> dict:
    t0 = time.time()
    pred = pd.read_parquet(LIVE / "predictions_latest.parquet")
    starts = pd.read_parquet(LIVE / "starts_ext.parquet")
    labels = json.loads((OUT / "feature_labels.json").read_text(encoding="utf-8"))
    inv = {v: k for k, v in labels.items()}
    val = json.loads((LIVE / "live_validation.json").read_text(encoding="utf-8")) if (LIVE / "live_validation.json").exists() else {}
    tnames = load_trainer_names()
    asof = pd.Timestamp(pred["asof"].max())

    latest = pred[pred["is_latest"]]
    live = latest[~latest["confirmed"] & latest["debut_year"].isin(SHOW_YEARS)].copy()
    live = live.sort_values("p_graded12", ascending=False)

    # 履歴（過去の基準点）: [k, base_date, p_graded12, p_dropout, pred_level]
    hist = pred[pred[COL_HORSE].isin(live[COL_HORSE])].sort_values([COL_HORSE, "k"])
    hmap: dict[int, list] = {}
    for r in hist.itertuples(index=False):
        hmap.setdefault(int(getattr(r, COL_HORSE)), []).append(
            [int(r.k), pd.Timestamp(r.base_date).strftime("%Y-%m-%d"), _r(r.p_graded12), _r(r.p_dropout), int(r.pred_level)])

    # デビュー年内の順位（P(重賞掲示板 12m) 降順）
    live["rank_in_year"] = live.groupby("debut_year")["p_graded12"].rank(ascending=False, method="first").astype(int)

    horses = []
    for r in live.itertuples(index=False):
        hid = int(getattr(r, COL_HORSE))
        pl = [_r(getattr(r, f"p_L{i}")) for i in range(5)]
        horses.append({
            "id": hid, "name": r.馬名, "dy": int(r.debut_year), "sex": r.性別, "sire": (None if pd.isna(r.父) else r.父),
            "trainer": tnames.get(int(r.調教師コード)), "n": int(r.n_starts_total), "k": int(r.k),
            "base": pd.Timestamp(r.base_date).strftime("%Y-%m-%d"), "summary": r.summary,
            "pd": _r(r.p_dropout), "pl": pl, "pg12": _r(r.p_graded12), "pg24": _r(r.p_graded24),
            "pct": _r(r.percentile_graded12, 1), "lvl": int(r.pred_level), "rank": int(r.rank_in_year),
            "why": _reasons(r.top3_A_L4, labels, inv, "top3_A_L4"),
            "hist": hmap.get(hid, []),
        })

    # 文章カード（デビュー年ごとの上位 TOP_CARDS）
    cards: dict[str, dict] = {}
    for y, g in live.groupby("debut_year"):
        top = g.sort_values("p_graded12", ascending=False).head(TOP_CARDS)
        for i, (_, row) in enumerate(top.iterrows()):
            sh = starts[starts[COL_HORSE] == row[COL_HORSE]]
            c = make_card(row, sh, labels, k=int(row["k"]), cohort=g["p_graded12"], why=None, trainer_names=tnames)
            cards[str(int(row[COL_HORSE]))] = {
                "text": c["text"],
                "runs": [{"date": d["date"], "label": d["label"], "pos": d["pos"], "field": d["field"],
                          "margin": (None if d["margin"] is None else max(0.0, d["margin"])),
                          "place": d["place"], "surface": d["surface"], "dist": d["dist"]} for d in c["runs"]],
            }

    by_year = {int(y): int(n) for y, n in live["debut_year"].value_counts().sort_index().items()}
    n_latest_by_year = {int(y): int(n) for y, n in latest[latest["debut_year"].isin(SHOW_YEARS)]["debut_year"].value_counts().sort_index().items()}
    meta = {
        "model": "shusse_v2", "asof": asof.strftime("%Y-%m-%d"),
        "data_end": pd.Timestamp(starts["date"].max()).strftime("%Y-%m-%d"),
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "levels": {str(i): LEVEL_NAME[i] for i in range(5)},
        "ks": [1, 2, 3, 5], "top_cards": TOP_CARDS,
        "years": sorted(by_year, reverse=True),
        "n_horses": len(horses), "n_by_year": by_year, "n_tracked_by_year": n_latest_by_year,
        "validation": _validation(val),
        "note": ("pg12 = (1 − pd) × pl[4]。未確定（基準日 + 365 日 > asof）で各馬の最新基準点のみ。"
                 "hist = [k, 基準日, pg12, pd, 予測クラス]。市場オッズ・人気は不使用。馬券の予想ではありません。"),
    }
    files: dict[str, dict] = {"index.json": meta}
    for y in by_year:
        hy = [h for h in horses if h["dy"] == y]
        ids = {str(h["id"]) for h in hy}
        files[f"{y}.json"] = {"asof": meta["asof"], "year": y, "n": len(hy), "horses": hy,
                              "cards": {k: v for k, v in cards.items() if k in ids}}
    if write:
        DST.mkdir(parents=True, exist_ok=True)
        for old in DST.glob("*.json"):
            if old.name not in files:
                old.unlink()
        tot = 0
        for name, obj in files.items():
            fp = DST / name
            fp.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str), encoding="utf-8")
            tot += fp.stat().st_size
        sizes = ", ".join(f"{n} {(DST / n).stat().st_size / 1e6:.2f}MB" for n in files)
        print(f"[site_export] {DST.relative_to(OUT.parents[1])}/: {len(horses):,} horses / {len(cards)} cards / "
              f"{tot / 1e6:.2f} MB total ({sizes}) / {time.time() - t0:.1f}s")
    return files


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    build()
