# -*- coding: utf-8 -*-
"""
build_horse_career.py — 各馬の「指数推移」(全キャリア) を site/data/career_{date}.json に生成
================================================================================
他サイトの馬指数(MI)推移チャート相当を、**公開事実だけ**(着順/確定オッズ→人気導出)で作る。
ZI/補正タイム等の TARGET 外部指数には依存しない (JRA-VAN 再掲問題を回避)。

指数の定義 (level_metric.py と同一思想):
  1走スコア   = run_rating(着順, 人気)          # 人気は確定単勝オッズの race 内順位から導出
  時点レベル  = その時点までの直近5走を horse_level_raw と同じ重みで合成
  表示値      = data/level_norms.json の分位アンカーで 0-100 正規化
  ランキング  = 対象日直前 365 日以内に出走した現役馬の中での時点レベル順位

ソース:
  E:/競馬過去走データ/kekka_20130105-20251228_v2.csv   全頭・確定オッズ入り (cp932)
  data/kekka/{YYYYMMDD}.csv                            2026 週次 (cp932, 同形式)
  → data/_career_results.parquet にキャッシュ (ソース更新時のみ再パース)

出力: site/data/career_{date}.json
  {"date": "...", "horses": {馬名: {"cur":73, "best":78, "rank":225, "n_rank":9999,
                                    "points": [["2023-11-18", 69, pos, ninki], ...]}}}

実行: python build_horse_career.py 20260725 [20260726 ...]   # 単体バックフィル
      build_site.py が最新数日分を自動で呼ぶ。
"""
from __future__ import annotations
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import level_metric as LM

ROOT = Path(__file__).parent
V2_CSV = Path("E:/競馬過去走データ/kekka_20130105-20251228_v2.csv")
WEEKLY_DIR = ROOT / "data" / "kekka"
CACHE = ROOT / "data" / "_career_results.parquet"
NORMS_PATH = ROOT / "data" / "level_norms.json"
SITE_DATA = ROOT / "site" / "data"

MAX_YEARS = 8          # 同名再使用リスク回避のため過去8年に限定 (現役馬のキャリアは収まる)
MAX_POINTS = 40        # チャート点数上限 (直近優先)
ACTIVE_DAYS = 365      # ランキング母集団 = この日数以内に出走


# ---------------------------------------------------------------- 結果ロード
def _parse_odds(s: pd.Series) -> pd.Series:
    """単勝配当列 → 確定単勝オッズ。勝ち馬=払戻(円)/100、他馬='(x.x)'。"""
    t = s.astype(str).str.strip()
    inpar = t.str.startswith("(")
    odds = pd.to_numeric(t.str.strip("()"), errors="coerce")
    odds = odds.where(inpar, odds / 100.0)
    return odds


def _load_one(path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path, encoding="cp932",
                         usecols=["日付", "場所", "Ｒ", "馬名", "確定着順", "単勝配当"],
                         dtype={"日付": str})
    except Exception as e:
        print(f"[career skip] {path.name}: {e}")
        return None
    df["pos"] = pd.to_numeric(df["確定着順"], errors="coerce")
    df = df.dropna(subset=["pos"])
    df = df[df["pos"] > 0]
    ymd = df["日付"].str.strip().str.zfill(6)
    df["date"] = pd.to_datetime("20" + ymd, format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date"])
    df["odds"] = _parse_odds(df["単勝配当"])
    return df[["date", "場所", "Ｒ", "馬名", "pos", "odds"]]


def load_results(force: bool = False) -> pd.DataFrame:
    """全結果 (date, name, pos, ninki)。parquet キャッシュ付き。"""
    srcs = [V2_CSV] + sorted(WEEKLY_DIR.glob("[0-9]" * 8 + ".csv"))
    srcs = [p for p in srcs if p.exists()]
    if CACHE.exists() and not force:
        cm = CACHE.stat().st_mtime
        if all(p.stat().st_mtime <= cm for p in srcs):
            return pd.read_parquet(CACHE)
    parts = [d for p in srcs if (d := _load_one(p)) is not None]
    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(subset=["date", "場所", "Ｒ", "馬名"], keep="last")
    # 人気 = レース内の確定オッズ昇順順位 (同オッズ同人気)
    df["ninki"] = (df.groupby(["date", "場所", "Ｒ"])["odds"]
                     .rank(method="min", ascending=True))
    df["pos"] = df["pos"].astype(int)
    df["ninki"] = df["ninki"].fillna(0).astype(int)
    out = (df[["date", "馬名", "pos", "ninki"]]
           .rename(columns={"馬名": "name"})
           .sort_values(["name", "date"])
           .reset_index(drop=True))
    out.to_parquet(CACHE)
    print(f"[career cache] {CACHE.name}  rows={len(out)}  "
          f"({out['date'].min().date()}〜{out['date'].max().date()})")
    return out


# ---------------------------------------------------------------- 指数計算
def _rate(pos: np.ndarray, ninki: np.ndarray) -> np.ndarray:
    """run_rating のベクトル版 (level_metric._POS_SCORE と同一式)。"""
    base = np.where(pos == 1, 100.0,
            np.where(pos == 2, 86.0,
             np.where(pos == 3, 75.0,
              np.where(pos == 4, 66.0,
               np.where(pos == 5, 59.0,
                np.maximum(12.0, 59.0 - (pos - 5) * 6.0))))))
    beat = np.where(ninki > 0, np.clip(ninki - pos, -7, 7) * 2.2, 0.0)
    return np.clip(base + beat, 0.0, 118.0)


_W = np.array(LM._RECENCY_W)


def _level_raw(ratings: np.ndarray) -> float:
    """直近走が先頭の rating 列 → horse_level_raw と同一の合成生スコア。"""
    r = ratings[:5]
    w = _W[: len(r)]
    wmean = float((w * r).sum() / w.sum())
    best3 = float(r[:3].max())
    return 0.7 * wmean + 0.3 * best3


def _to100(raw, anchors) -> float:
    return float(np.interp(raw, anchors, np.linspace(0, 100, len(anchors))))


def build_for_date(date_str: str, names: set[str], results: pd.DataFrame,
                   anchors) -> dict:
    """カード日 date_str 時点の career json payload を作る (当日以降の結果は見ない)。"""
    asof = pd.Timestamp(datetime.strptime(date_str, "%Y%m%d"))
    lo = asof - timedelta(days=MAX_YEARS * 365)
    df = results[(results["date"] < asof) & (results["date"] >= lo)].copy()
    df["rating"] = _rate(df["pos"].to_numpy(), df["ninki"].to_numpy())

    # ランキング母集団: 直近 ACTIVE_DAYS 以内に出走した馬の時点レベル
    active_cut = asof - timedelta(days=ACTIVE_DAYS)
    last_run = df.groupby("name")["date"].max()
    active = set(last_run[last_run >= active_cut].index)
    levels: dict[str, float] = {}
    for name, g in df[df["name"].isin(active)].groupby("name", sort=False):
        levels[name] = _to100(_level_raw(g["rating"].to_numpy()[::-1]), anchors)
    ranked = sorted(levels.values(), reverse=True)
    n_rank = len(ranked)

    horses = {}
    for name in sorted(names):
        g = df[df["name"] == name]
        if g.empty:
            continue
        ratings = g["rating"].to_numpy()
        # 各走時点のレベル (その走までの直近5走合成) → チャートの点
        pts = []
        for i in range(len(g)):
            lv = _to100(_level_raw(ratings[max(0, i - 4): i + 1][::-1]), anchors)
            row = g.iloc[i]
            pts.append([row["date"].strftime("%Y-%m-%d"), round(lv),
                        int(row["pos"]), int(row["ninki"])])
        pts = pts[-MAX_POINTS:]
        cur = pts[-1][1]
        best = max(p[1] for p in pts)
        rank = (1 + sum(1 for v in ranked if v > levels[name])
                if name in levels else None)
        horses[name] = {"cur": cur, "best": best, "rank": rank,
                        "n_rank": n_rank, "points": pts}
    return {"date": date_str,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "n_rank": n_rank, "horses": horses}


# ---------------------------------------------------------------- エントリ
def _names_from_day_json(date_str: str) -> set[str]:
    p = SITE_DATA / f"{date_str}.json"
    if not p.exists():
        return set()
    d = json.loads(p.read_text(encoding="utf-8"))
    return {h.get("name") for r in d.get("races", []) for h in r.get("horses", [])
            if h.get("name")}


def build_careers(date_strs: list[str], names_by_date: dict[str, set] | None = None):
    anchors = json.loads(NORMS_PATH.read_text(encoding="utf-8"))["raw_anchors"]
    results = load_results()
    for ds in date_strs:
        names = (names_by_date or {}).get(ds) or _names_from_day_json(ds)
        if not names:
            print(f"[career skip] {ds}: no horses")
            continue
        payload = build_for_date(ds, names, results, anchors)
        out = SITE_DATA / f"career_{ds}.json"
        out.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                       encoding="utf-8")
        print(f"[career saved] {out.name}  horses={len(payload['horses'])}/{len(names)}"
              f"  rank_pool={payload['n_rank']}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not args:
        # 引数なし: manifest の最新2日
        mf = json.loads((SITE_DATA / "manifest.json").read_text(encoding="utf-8"))
        args = [e["date"] for e in mf.get("dates", [])[:2]]
    build_careers(args)
