# -*- coding: utf-8 -*-
"""
speed_index.py — PyCaLiAI スピード指数（実走指数 / 予測指数、100 基準）
================================================================================
走破タイムを「物差し（基準タイム）」と比べ、斤量・馬場差・ペース・競馬場の癖を補正して
1 本の数字にした能力指数。考え方は stride の BT 値（https://stride.keiba.bigtime.world/about/bt）
を参考にし、数値はすべてこのリポジトリのデータから決めている。

  指数 100 = 古馬 1〜2 勝クラスの基準タイムどおりに走った水準
  1 pt    = 1600m で 0.1 秒（距離に比例。短距離は下限あり＝ SPP_FLOOR_DIST）

入力は JRA 公表の結果（走破タイム・上がり 3F・着順・斤量・馬場状態）だけ。オッズ・人気・
TARGET の補正タイム/ZI・JRDB の値は使わない（サイト公開の規約上も、指数の独立性の上でも）。

手順（実走指数）:
  1. 斤量補正     t' = t − K_W × (斤量 − 57) × 距離/1000      （前半 70% / 上がり 30% に配分）
  2. 基準タイム   競馬場×芝ダ×距離×内外（小倉芝・中京芝は夏 6-8 月と冬で別）ごとに、
                  直前 5 年の良馬場・古馬 1〜2 勝クラスの「2 着〜上位 33%」の t' の中央値。
                  標本が足りないコースは全クラスをクラス差で 1〜2 勝相当に換算して作る。
  3. 素点         100 + (基準 − t') / spp(距離)
  4. 馬場差       同じ日・場・芝ダ・馬場状態のレースごとに「上位 3 頭の素点 − クラス標準」を
                  取り、日単位で縮小推定（レース数が少ない日は馬場状態ごとの平均に寄せる）。3 回反復。
  5. ペース       レースの前半と上がりの速さの差が大きいとき、スローは上がり・ハイは前半の
                  出来を混ぜる（BT のペース補正ブレンドと同じ考え方、混ぜる割合は上限つき）。
  6. 場・馬場の癖 同じ馬（4 歳以上）の 30〜120 日間隔の連続出走の指数差から、
                  競馬場×芝ダ と 芝ダ×馬場状態 のずれを最小二乗で推定して差し引く。
  補正の強さ（K_W・馬場差の縮小・ペースの係数）は「同じ馬の次走の指数とどれだけ一致するか」
  （再現性）が最大になるよう `tune` で決める。

予測指数: 過去の実走指数・間隔・距離/芝ダ替わり・斤量・年齢から、今回の実走指数を
  LightGBM で予測（学習 〜2022 / 調整 2023 / 検証 2024〜）。オッズは使わない。

データ源（読み取りのみ）:
  E:/競馬過去走データ/raw_data/kekka_1986_2025_enhanced.csv   2011-01〜2025-11（全頭の走破タイム）
  E:/競馬過去走データ/raw_data/kekka_20251206_20260222.csv     TARGET 成績形式
  data/speed_src/*.csv                                         追加の TARGET 成績形式（2026〜）

使い方:
  python speed_index.py build            # 実走指数を作る → reports/speed_index/runs.parquet
  python speed_index.py tune             # 補正の強さの格子探索（2016-2021 の再現性）
  python speed_index.py report           # 再現性・クラス別の水準・前向きの的中を表示
  python speed_index.py predict-train    # 予測指数モデルを学習 + 検証
出力は reports/speed_index/（.gitignore 済み。公開サイトへは build_site 側が指数だけを渡す）。
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
import unicodedata
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).parent
OUT_DIR = BASE / "reports" / "speed_index"
SRC_ENH = Path("E:/競馬過去走データ/raw_data/kekka_1986_2025_enhanced.csv")
SRC_TARGET = [Path("E:/競馬過去走データ/raw_data/kekka_20251206_20260222.csv")]
SRC_EXTRA_DIR = BASE / "data" / "speed_src"
FIRST_YEAR = 2011
LOAD_VERSION = 3   # load_runs の中身を変えたら上げる（キャッシュ無効化）

log = logging.getLogger("speed_index")

CLASS_BY_CODE = {15: "新馬", 7: "未勝利", 23: "1勝", 43: "2勝", 67: "3勝", 131: "OP", 115: "L",
                 147: "G3", 163: "G3", 179: "G2", 195: "G1"}
CLASS_ORDER = ["新馬", "未勝利", "1勝", "2勝", "3勝", "OP", "L", "G3", "G2", "G1"]
PLACE_CODE = {"01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京", "06": "中山",
              "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"}
SEASONAL = {("小倉", "芝"), ("中京", "芝")}


@dataclass(frozen=True)
class Params:
    k_w: float = 0.07            # 斤量 1kg あたりの秒数（1000m あたり）
    spp_base: float = 0.10       # 1600m で 1pt = 0.1 秒
    spp_floor_dist: int = 1400   # これより短い距離は spp をこの距離の値で止める
    var_lambda: float = 1.5      # 馬場差の縮小（レース数換算の事前の重み）
    var_iters: int = 3
    pace_a: float = 0.45         # 上がりの出来（同じレースの中央値との差）を混ぜる割合
    pace_b: float = 0.15         # 前半の出来（同上）を混ぜる割合
    pace_g: float = 0.35         # スロー（前半が上がりより遅い）レースの水準の引き上げ
    pace_tau: float = 0.0        # 引き上げを始める前後半差（pt/1000m）
    course_ridge: float = 50.0   # 場・馬場の癖の推定の ridge
    min_base_n: int = 25


# --------------------------------------------------------------------------- load
def _z2h(s):
    return unicodedata.normalize("NFKC", str(s)) if s is not None else s


def _load_enhanced() -> pd.DataFrame:
    usecols = ["年", "月", "日", "場所", "レース番号", "レース名", "クラスコード", "芝・ダ", "コースコード",
               "距離", "馬場状態", "馬名", "性別", "年齢", "斤量", "頭数", "馬番", "確定着順", "異常コード",
               "着差タイム", "人気順", "走破タイム", "通過順1", "通過順2", "通過順3", "通過順4",
               "上がり3Fタイム", "血統登録番号", "騎手コード"]
    d = pd.read_csv(SRC_ENH, encoding="cp932", usecols=usecols, low_memory=False)
    d = d[d["年"] >= FIRST_YEAR - 2000].copy()
    out = pd.DataFrame({
        "date": pd.to_datetime(dict(year=d["年"] + 2000, month=d["月"], day=d["日"])),
        "place": d["場所"].astype(str).str.strip(),
        "rno": d["レース番号"].astype(int),
        "race_name": d["レース名"].astype(str).str.strip(),
        "klass": d["クラスコード"].map(CLASS_BY_CODE),
        "surface": d["芝・ダ"].astype(str).str.strip(),
        "course_code": d["コースコード"].astype(int),
        "dist": d["距離"].astype(int),
        "going": d["馬場状態"].astype(str).str.strip(),
        "horse": d["馬名"].astype(str).str.strip(),
        "horse_id": d["血統登録番号"].astype(str).str.strip(),
        "sex": d["性別"].astype(str).str.strip(),
        "age": d["年齢"].astype(int),
        "weight": d["斤量"].astype(float),
        "field": d["頭数"].astype(int),
        "umaban": d["馬番"].astype(int),
        "finish": d["確定着順"].astype(int),
        "abnormal": d["異常コード"].astype(int),
        "margin": d["着差タイム"].astype(float),
        "ninki": d["人気順"].astype(int),
        "time": d["走破タイム"].astype(float),
        "agari": d["上がり3Fタイム"].astype(float),
        "c4": pd.to_numeric(d["通過順4"], errors="coerce"),
        "jockey": d["騎手コード"].astype(str),
        "src": "enh",
    })
    return out


def _parse_target(path: Path) -> pd.DataFrame:
    t = pd.read_csv(path, encoding="cp932", low_memory=False, dtype=str)
    t.columns = [c.strip() for c in t.columns]
    rid = t["レースID(旧)"].astype(str).str.zfill(10)
    ymd = t["日付"].astype(str).str.zfill(6)
    tt = t["走破タイム"].astype(str).str.strip()
    # 1194 → 1:19.4 → 79.4 秒
    num = pd.to_numeric(tt, errors="coerce")
    secs = (num // 1000) * 60 + (num % 1000) / 10.0
    fin = pd.to_numeric(t["着順"].map(_z2h), errors="coerce")
    klass = t["クラス名"].map(lambda s: _z2h(s).strip() if isinstance(s, str) else s).map(
        {"新馬": "新馬", "未勝利": "未勝利", "1勝": "1勝", "2勝": "2勝", "3勝": "3勝", "ｵｰﾌﾟﾝ": "OP",
         "オープン": "OP", "OP(L)": "L", "G3": "G3", "G2": "G2", "G1": "G1"})
    surf = t["芝・ダ"].astype(str).str.strip()
    out = pd.DataFrame({
        "date": pd.to_datetime("20" + ymd, format="%Y%m%d", errors="coerce"),
        "place": t["場所"].astype(str).str.strip(),
        "rno": pd.to_numeric(rid.str[6:8], errors="coerce"),
        "race_name": t["レース名"].astype(str).str.strip(),
        "klass": klass,
        "surface": surf,
        "course_code": np.where(surf == "ダ", 1, -1),   # 芝の内外は後で推定
        "dist": pd.to_numeric(t["距離"], errors="coerce"),
        "going": t["馬場状態"].astype(str).str.strip(),
        "horse": t["馬名"].astype(str).str.strip(),
        "horse_id": t["血統登録番号"].astype(str).str.strip(),
        "sex": t["性別"].astype(str).str.strip(),
        "age": pd.to_numeric(t["年齢"], errors="coerce"),
        "weight": pd.to_numeric(t["斤量"].astype(str).str.extract(r"(\d+(?:\.\d)?)")[0], errors="coerce"),
        "field": pd.to_numeric(t["出走頭数"], errors="coerce"),
        "umaban": pd.to_numeric(t["馬番"], errors="coerce"),
        "finish": fin,
        "abnormal": np.where(fin.isna(), 1, 0),
        "margin": np.nan,
        "ninki": np.nan,
        "time": secs,
        "agari": pd.to_numeric(t["上り3F"], errors="coerce"),
        "c4": pd.to_numeric(t["4角"], errors="coerce"),
        "jockey": t["騎手コード"].astype(str),
        "src": path.stem,
    })
    if "外回り" in t.columns:   # 追加エクスポートで外回り列があれば使う
        outer = t["外回り"].astype(str).str.strip().isin(["外", "1", "○", "外回り"])
        out.loc[surf == "芝", "course_code"] = np.where(outer[surf == "芝"], 8, 0)
    # 着差（勝ち馬との秒差）は走破タイムから
    key = ["date", "place", "rno"]
    out["margin"] = out["time"] - out.groupby(key)["time"].transform("min")
    return out.dropna(subset=["date", "rno", "dist", "time"])


WEEKLY_DIRS = [BASE / "data" / "weekly", Path("E:/PyCaLiAI/data/weekly")]
OUTER_TRACK = {12, 13, 18, 19}     # JV トラックコード: 芝 外回り / 内→外


def _weekly_files() -> list[Path]:
    seen = {}
    for d in WEEKLY_DIRS:
        if d.exists():
            for p in d.glob("2026[0-9][0-9][0-9][0-9].csv"):
                seen.setdefault(p.name, p)
    return [seen[k] for k in sorted(seen)]


def _norm_class(s) -> str | None:
    s = _z2h(s or "").replace(" ", "")
    for k in ("G1", "G2", "G3"):
        if k in s and "J" not in s:
            return k
    if "(L)" in s or s.endswith("L"):
        return "L"
    if "新馬" in s:
        return "新馬"
    if "未勝利" in s:
        return "未勝利"
    for k in ("1勝", "2勝", "3勝"):
        if k in s:
            return k
    if "ｵｰﾌﾟﾝ" in s or "オープン" in s or "OP" in s:
        return "OP"
    return None


def load_weekly_reconstructed() -> pd.DataFrame:
    """2026 年の週次出走表（TARGET 出馬表 CSV）から、各馬の「前走」を復元する。
    前走の走破タイム = 前走 Ave-3F × (前走距離 − 600)/600 + 前走上り 3F（TARGET の Ave-3F の定義）。
    前走のレース番号・クラス・外回りは、前走日の出走表を馬名で引いて決める。
    次に走った馬の分しか取れない（直近週の馬場差は少ない頭数から推定される）。"""
    from weekly_parse import parse_csv
    ent, prev = [], []
    for p in _weekly_files():
        try:
            w = parse_csv(p)
        except Exception as e:      # noqa: BLE001
            log.warning("weekly %s 読めない: %s", p.name, e)
            continue
        dt = pd.to_datetime(p.stem, format="%Y%m%d")
        surf = w["芝・ダ"].astype(str).str[0].map({"芝": "芝", "ダ": "ダ"})
        e = pd.DataFrame({
            "date": dt, "place": w["場所"].astype(str).str.strip(), "rno": pd.to_numeric(w["R"], errors="coerce"),
            "race_name": w["レース名"].astype(str).str.strip(), "klass": w["クラス名"].map(_norm_class),
            "surface": surf, "dist": pd.to_numeric(w["距離"], errors="coerce"),
            "course_code": np.where(surf == "ダ", 1, np.where(pd.to_numeric(w.get("トラックコード(JV)"), errors="coerce")
                                                          .isin(OUTER_TRACK), 8, 0)),
            "horse": w["馬名"].astype(str).str.strip(), "sex": w["性別"].astype(str).str.strip(),
            "age": pd.to_numeric(w["年齢"], errors="coerce"), "weight_e": pd.to_numeric(w["斤量"], errors="coerce"),
            "umaban": pd.to_numeric(w["馬番"], errors="coerce"), "jump": w["芝・ダ"].astype(str).str.contains("障"),
        })
        ent.append(e)
        mo = pd.to_numeric(w["前走月"], errors="coerce")
        dy = pd.to_numeric(w["前走日"], errors="coerce")
        yr = np.where(mo > dt.month, dt.year - 1, dt.year)
        pdt = pd.to_datetime(dict(year=yr, month=mo.fillna(1), day=dy.fillna(1)), errors="coerce").where(mo.notna())
        pv = pd.DataFrame({
            "date": pdt, "horse": w["馬名"].astype(str).str.strip(),
            "going": w["前走馬場状態"].astype(str).str.strip(), "weight": pd.to_numeric(w["前走斤量"], errors="coerce"),
            "finish": pd.to_numeric(w["前走確定着順"], errors="coerce"), "margin": pd.to_numeric(w["前走着差タイム"], errors="coerce"),
            "c4": pd.to_numeric(w["前4角"], errors="coerce"), "field": pd.to_numeric(w["前走出走頭数"], errors="coerce"),
            "ave3f": pd.to_numeric(w["前走Ave-3F"], errors="coerce"), "agari": pd.to_numeric(w["前走上り3F"], errors="coerce"),
            "pdist": pd.to_numeric(w["前距離"], errors="coerce"), "ninki": pd.to_numeric(w["前走人気"], errors="coerce"),
        })
        prev.append(pv[pv["date"].notna() & (pv["date"].dt.year >= 2026)])
    E = pd.concat(ent, ignore_index=True)
    P = pd.concat(prev, ignore_index=True).drop_duplicates(["date", "horse"], keep="first")
    R = P.merge(E, on=["date", "horse"], how="inner")
    R = R[(~R["jump"]) & R["surface"].notna() & (R["pdist"] == R["dist"])].copy()
    R["time"] = R["ave3f"] * (R["dist"] - 600) / 600 + R["agari"]
    R["abnormal"] = np.where(R["finish"] > 0, 0, 1)
    R["horse_id"] = ""
    R["jockey"] = ""
    R["src"] = "weekly"
    cols = ["date", "place", "rno", "race_name", "klass", "surface", "course_code", "dist", "going", "horse",
            "horse_id", "sex", "age", "weight", "field", "umaban", "finish", "abnormal", "margin", "ninki", "time",
            "agari", "c4", "jockey", "src"]
    R = R.dropna(subset=["rno", "time", "finish"]).drop_duplicates(["date", "horse"])
    return R[cols]


def load_runs() -> pd.DataFrame:
    parts = [_load_enhanced()]
    tgt = list(SRC_TARGET)
    if SRC_EXTRA_DIR.exists():
        tgt += sorted(SRC_EXTRA_DIR.glob("*.csv"))
    for p in tgt:
        if p.exists():
            parts.append(_parse_target(p))
    d = pd.concat(parts, ignore_index=True)
    d = d.drop_duplicates(["date", "place", "rno", "umaban"], keep="last")
    # 直接のデータが無い期間は週次出走表からの復元で埋める（2026-02-23〜）
    for name in ("weekly_parse", "training_join"):
        logging.getLogger(name).setLevel(logging.WARNING)
    w = load_weekly_reconstructed()
    w = w[w["date"] > d["date"].max()]
    if len(w):
        ids = d.sort_values("date").drop_duplicates("horse", keep="last").set_index("horse")["horse_id"]
        w["horse_id"] = w["horse"].map(ids).fillna("N:" + w["horse"])
        log.info("週次出走表から復元 %d 走（%s〜%s）", len(w), w["date"].min().date(), w["date"].max().date())
        d = pd.concat([d, w], ignore_index=True)
    # 障害を除く（enhanced のコースコード 2/3、名前、TARGET の芝・ダ=障）
    jump = d["course_code"].isin([2, 3]) | d["race_name"].str.contains("障害|ジャンプ|ＪＧ|JG", na=False) \
        | ~d["surface"].isin(["芝", "ダ"])
    d = d[~jump].copy()
    d["rno"] = d["rno"].astype(int)
    d["dist"] = d["dist"].astype(int)
    d["race_key"] = d["date"].dt.strftime("%Y%m%d") + d["place"] + d["rno"].astype(str).str.zfill(2)
    d["year"] = d["date"].dt.year
    d["month"] = d["date"].dt.month
    # レースの年齢区分: 2歳 / 3歳 / 古馬（3歳以上・4歳以上）
    amin = d.groupby("race_key")["age"].transform("min")
    amax = d.groupby("race_key")["age"].transform("max")
    d["age_grp"] = np.where(amax <= 2, "2歳", np.where(amax <= 3, "3歳", "古馬"))
    d.loc[(amin >= 3) & (amax > 3), "age_grp"] = "古馬"
    # 週次復元の芝の内外は JV トラックコード頼みで欠けることがある。過去に存在しない
    # （場・距離・内外）の組なら -1 にして、過去の構成から決め直す。
    known = set(map(tuple, d.loc[d["src"] != "weekly", ["place", "dist", "course_code"]].drop_duplicates().to_numpy()))
    wk = (d["src"] == "weekly") & (d["surface"] == "芝")
    bad = wk & ~pd.Series([(a, b, c) in known for a, b, c in zip(d["place"], d["dist"], d["course_code"])], index=d.index)
    d.loc[bad, "course_code"] = -1
    _resolve_course_code(d)
    d["valid"] = (d["abnormal"] == 0) & (d["finish"] > 0) & (d["time"] > 0) & d["agari"].between(30, 45)
    d["fsize"] = d.groupby("race_key")["valid"].transform("sum")
    return d.sort_values(["date", "place", "rno", "finish"]).reset_index(drop=True)


def _resolve_course_code(d: pd.DataFrame) -> None:
    """TARGET 形式（内外不明 = -1）の芝の内外を、同じ場・距離の過去の内外構成から決める。
    1 種類しかないコースはそれ。内外両方あるコース（京都芝1400/1600・新潟芝2000）は、
    そのレースの勝ちタイムに近い方の過去中央値を採る。"""
    known = d[d["course_code"] >= 0]
    combos = known[known["surface"] == "芝"].groupby(["place", "dist"])["course_code"].agg(
        lambda s: sorted(s.unique().tolist()))
    unk = d.index[d["course_code"] < 0]
    if len(unk) == 0:
        return
    win = known[(known["finish"] == 1) & (known["going"] == "良")].groupby(
        ["place", "surface", "dist", "course_code"])["time"].median()
    for rk, idx in d.loc[unk].groupby("race_key").groups.items():
        r = d.loc[idx[0]]
        opts = combos.get((r["place"], r["dist"]), [0])
        if len(opts) == 1:
            d.loc[idx, "course_code"] = opts[0]
            continue
        wt = d.loc[idx, "time"].min()
        best = min(opts, key=lambda c: abs(win.get((r["place"], "芝", r["dist"], c), 1e9) - wt))
        d.loc[idx, "course_code"] = best


# --------------------------------------------------------------------------- figures
def spp(dist, p: Params):
    return p.spp_base * np.maximum(dist, p.spp_floor_dist) / 1600.0


def _course_key(d: pd.DataFrame) -> pd.Series:
    seas = np.where(d["month"].between(6, 8), "夏", "冬")
    tag = [s if (pl, sf) in SEASONAL else "" for pl, sf, s in zip(d["place"], d["surface"], seas)]
    return d["place"] + d["surface"] + d["dist"].astype(str) + "_" + d["course_code"].astype(str) + pd.Series(tag, index=d.index)


def _ref_mask(d: pd.DataFrame) -> pd.Series:
    """基準タイムの標本: 良・古馬 1〜2 勝・2 着〜上位 33%・着差 2 秒以内。"""
    top = np.maximum(2, np.ceil(d["fsize"] * 0.33))
    return d["valid"] & (d["going"] == "良") & d["klass"].isin(["1勝", "2勝"]) & (d["age_grp"] == "古馬") \
        & (d["finish"] >= 2) & (d["finish"] <= top) & (d["margin"].fillna(0) <= 2.0)


def _placed_mask(d: pd.DataFrame) -> pd.Series:
    top = np.maximum(2, np.ceil(d["fsize"] * 0.33))
    return d["valid"] & (d["going"] == "良") & (d["finish"] >= 2) & (d["finish"] <= top) \
        & (d["margin"].fillna(0) <= 2.0)


def compute_figures(d: pd.DataFrame, p: Params, years=None) -> pd.DataFrame:
    """実走指数を計算して列を足した DataFrame を返す。years を渡すとその年だけ計算する（tune 用）。"""
    d = d.copy()
    d["ckey"] = _course_key(d)
    wcorr = p.k_w * (d["weight"] - 57.0) * d["dist"] / 1000.0
    d["t_adj"] = d["time"] - wcorr
    d["f_adj"] = (d["time"] - d["agari"]) - 0.7 * wcorr
    d["a_adj"] = d["agari"] - 0.3 * wcorr
    d["spp"] = spp(d["dist"], p)

    ref = _ref_mask(d)
    placed = _placed_mask(d)
    target_years = sorted(d["year"].unique()) if years is None else sorted(years)
    base_t = pd.Series(np.nan, index=d.index)
    base_f = pd.Series(np.nan, index=d.index)
    base_a = pd.Series(np.nan, index=d.index)
    for y in target_years:
        lo, hi = (y - 5, y - 1) if y - 5 >= FIRST_YEAR else (FIRST_YEAR, FIRST_YEAR + 4)
        win = d["year"].between(lo, hi)
        r = d[win & ref]
        g = r.groupby("ckey")
        med = g[["t_adj", "f_adj", "a_adj"]].median()
        n = g.size()
        dense = med[n >= p.min_base_n]
        # クラス差（pt）: 標本の多いコースで、各クラスの placed 馬が基準から何 pt 速いか
        pl = d[win & placed & d["ckey"].isin(dense.index)]
        coff = ((dense["t_adj"].reindex(pl["ckey"]).to_numpy() - pl["t_adj"].to_numpy()) / pl["spp"].to_numpy())
        class_off = pd.Series(coff, index=pl.index).groupby([pl["klass"], pl["age_grp"]]).median()
        # 標本の足りないコースは全クラスを 1〜2 勝相当に換算
        sp = d[win & placed & ~d["ckey"].isin(dense.index)]
        if len(sp):
            off = class_off.reindex(pd.MultiIndex.from_arrays([sp["klass"], sp["age_grp"]])).fillna(0).to_numpy()
            conv = sp[["t_adj", "f_adj", "a_adj"]].copy()
            conv["t_adj"] += off * sp["spp"].to_numpy()
            frac_f = (sp["dist"] - 600) / sp["dist"]
            conv["f_adj"] += off * sp["spp"].to_numpy() * frac_f.to_numpy()
            conv["a_adj"] += off * sp["spp"].to_numpy() * (1 - frac_f.to_numpy())
            sparse = conv.groupby(sp["ckey"]).median()[conv.groupby(sp["ckey"]).size() >= 5]
            table = pd.concat([dense, sparse[~sparse.index.isin(dense.index)]])
        else:
            table = dense
        m = d["year"] == y
        base_t[m] = d.loc[m, "ckey"].map(table["t_adj"])
        base_f[m] = d.loc[m, "ckey"].map(table["f_adj"])
        base_a[m] = d.loc[m, "ckey"].map(table["a_adj"])
    d["base_t"], d["base_f"], d["base_a"] = base_t, base_f, base_a
    if years is not None:
        d = d[d["year"].isin(years)].copy()
    d = d[d["base_t"].notna() & d["valid"]].copy()

    # 素点（前半 + 上がり = 合計）
    d["f_pts"] = (d["base_f"] - d["f_adj"]) / d["spp"]
    d["a_pts"] = (d["base_a"] - d["a_adj"]) / d["spp"]
    d["raw"] = 100.0 + (d["base_t"] - d["t_adj"]) / d["spp"]

    # 馬場差（日・場・芝ダ・馬場状態ごと、縮小推定・反復）
    d["vkey"] = d["date"].dt.strftime("%Y%m%d") + d["place"] + d["surface"] + d["going"]
    d["par_key"] = d["klass"].astype(str) + d["age_grp"]
    top3 = d["finish"] <= 3
    V = pd.Series(0.0, index=d.index)
    for _ in range(p.var_iters):
        fig = d["raw"] - V
        pm = top3 & (d["year"] <= 2022)          # クラス標準は検証期間（2023〜）を見ずに作る
        if pm.sum() < 1000:
            pm = top3
        par = fig[pm].groupby(d.loc[pm, "par_key"]).median()
        rr = d[top3].assign(fig=fig[top3]).groupby("race_key").agg(
            fig=("fig", "median"), par_key=("par_key", "first"), vkey=("vkey", "first"),
            sg=("surface", "first"), going=("going", "first"))
        rr["dev"] = rr["fig"] - rr["par_key"].map(par)
        # 今の V を足し戻した「生の」ずれ
        vprev = V.groupby(d["race_key"]).first()
        rr["dev"] = rr["dev"] + rr.index.map(vprev).fillna(0).to_numpy()
        prior = rr.groupby(["sg", "going"])["dev"].mean()
        day = rr.groupby("vkey").agg(s=("dev", "sum"), n=("dev", "size"), sg=("sg", "first"), going=("going", "first"))
        pri = prior.reindex(pd.MultiIndex.from_arrays([day["sg"], day["going"]])).to_numpy()
        day["V"] = (day["s"] + p.var_lambda * pri) / (day["n"] + p.var_lambda)
        V = d["vkey"].map(day["V"]).fillna(0.0)
    d["variant"] = V
    fr = (d["dist"] - 600) / d["dist"]
    d["f_pts"] -= V * fr
    d["a_pts"] -= V * (1 - fr)
    d["fig0"] = d["raw"] - V

    # ペース補正（前半と上がりの「1000m あたりの出来」の差）
    d["f_rate"] = d["f_pts"] / (d["dist"] - 600) * 1000
    d["a_rate"] = d["a_pts"] / 600 * 1000
    half = d["finish"] <= np.maximum(3, np.ceil(d["fsize"] / 2))
    pace = (d.loc[half, "f_rate"] - d.loc[half, "a_rate"]).groupby(d.loc[half, "race_key"]).median()
    d["pace"] = d["race_key"].map(pace).fillna(0.0)
    # 2026-11-15: BT 型の「上がり/前半の点数を混ぜる」は再現性を下げた（tune で係数 0 が最良）。
    # 代わりに、同じレースの中での上がり・前半の速さ（中央値との差）を足し、スローの
    # レースは水準を引き上げる形にした（2016-21 で決め、2022-25 で レース内相関 0.381→0.408）。
    a_dev = d["a_pts"] - d.groupby("race_key")["a_pts"].transform("median")
    f_dev = d["f_pts"] - d.groupby("race_key")["f_pts"].transform("median")
    adj = p.pace_a * a_dev + p.pace_b * f_dev + p.pace_g * np.maximum(0, -d["pace"] - p.pace_tau)
    d["pace_adj"] = adj
    d["fig1"] = d["fig0"] + adj
    d["fig"] = d["fig1"]
    return d


def fit_course_offsets(d: pd.DataFrame, p: Params, fit_years=None) -> dict:
    """同じ馬（4 歳以上）の 30〜120 日間隔の連続出走の差から、場×芝ダ と 芝ダ×馬場状態 のずれを推定。"""
    x = d[(d["age"] >= 4)].sort_values(["horse_id", "date"])
    if fit_years is not None:
        x = x[x["year"].isin(fit_years)]
    x = x.assign(g1=x["place"] + x["surface"], g2=x["surface"] + x["going"])
    prev = x.groupby("horse_id").shift(1)
    gap = (x["date"] - prev["date"]).dt.days
    ok = gap.between(30, 120) & prev["fig"].notna() & (x["margin"].fillna(0) <= 3) & (prev["margin"].fillna(0) <= 3)
    a, b = x[ok], prev[ok]
    groups = sorted(set(a["g1"]) | set(b["g1"])) + sorted(set(a["g2"]) | set(b["g2"]))
    gi = {g: i for i, g in enumerate(groups)}
    n = len(a)
    y = (a["fig"] - b["fig"]).to_numpy()
    from scipy.sparse import coo_matrix
    rows = np.r_[np.arange(n), np.arange(n), np.arange(n), np.arange(n)]
    cols = np.r_[a["g1"].map(gi), b["g1"].map(gi), a["g2"].map(gi), b["g2"].map(gi)]
    vals = np.r_[np.ones(n), -np.ones(n), np.ones(n), -np.ones(n)]
    X = coo_matrix((vals, (rows, cols)), shape=(n, len(groups))).tocsr()
    A = (X.T @ X).toarray() + p.course_ridge * np.eye(len(groups))
    coef = np.linalg.solve(A, X.T @ (y - y.mean()))
    off = dict(zip(groups, coef))
    # 中心: 中央 4 場（東京・中山・京都・阪神）と良馬場の平均を 0 に
    for sf in ("芝", "ダ"):
        c = np.mean([off.get(pl + sf, 0) for pl in ("東京", "中山", "京都", "阪神")])
        for pl in PLACE_CODE.values():
            if pl + sf in off:
                off[pl + sf] -= c
        g0 = off.get(sf + "良", 0)
        for gname in ("良", "稍", "重", "不"):
            if sf + gname in off:
                off[sf + gname] -= g0
    return {k: float(v) for k, v in off.items()}


def apply_course_offsets(d: pd.DataFrame, off: dict) -> pd.DataFrame:
    d = d.copy()
    o = (d["place"] + d["surface"]).map(off).fillna(0) + (d["surface"] + d["going"]).map(off).fillna(0)
    d["course_adj"] = -o
    d["fig"] = d["fig1"] - o
    return d


# --------------------------------------------------------------------------- evaluation
def reliability(d: pd.DataFrame, col="fig", years=None, min_gap=14, max_gap=90) -> dict:
    """同じ馬の連続出走（同じ芝ダ・3 歳以上・大敗除く）の指数の相関＝再現性。"""
    x = d[(d["age"] >= 3)].sort_values(["horse_id", "date"])
    prev = x.groupby("horse_id").shift(1)
    gap = (x["date"] - prev["date"]).dt.days
    ok = gap.between(min_gap, max_gap) & (x["surface"] == prev["surface"]) & prev[col].notna() \
        & (x["margin"].fillna(0) <= 3) & (prev["margin"].fillna(0) <= 3)
    if years is not None:
        ok &= x["year"].isin(years)
    a, b = x.loc[ok, col], prev.loc[ok, col]
    # 年・クラスで中心化した残差相関も（クラスの格の差だけで相関が出るのを除く）
    return {"n": int(ok.sum()), "r": float(np.corrcoef(a, b)[0, 1]),
            "mad": float(np.median(np.abs(a.to_numpy() - b.to_numpy())))}


def _with_prev(d: pd.DataFrame, col: str) -> pd.DataFrame:
    x = d.sort_values(["horse_id", "date"])
    prev = x.groupby("horse_id").shift(1)
    gap = (x["date"] - prev["date"]).dt.days
    return x.assign(prev_fig=np.where(gap.le(180) & (x["surface"] == prev["surface"]), prev[col], np.nan))


def race_corr(d: pd.DataFrame, col: str, years) -> dict:
    """前走（同じ芝ダ・180 日以内）の指数と今回の着順のレース内順位相関（Spearman）の平均。
    クラスの格の差が効かない「同じレースの中での並べ方」の正しさ。tune の目的関数。"""
    x = _with_prev(d, col)
    x = x[x["year"].isin(years) & x["prev_fig"].notna()]
    n = x.groupby("race_key")["prev_fig"].transform("size")
    x = x[n >= 6]
    r1 = x.groupby("race_key")["prev_fig"].rank(ascending=False)
    r2 = x.groupby("race_key")["finish"].rank()
    dd = (r1 - r2) ** 2
    m = x.groupby("race_key").size()
    s = dd.groupby(x["race_key"]).sum()
    rho = 1 - 6 * s / (m * (m ** 2 - 1))
    return {"races": int(len(rho)), "rho": float(rho.mean())}


def within_race_hit(d: pd.DataFrame, col: str, years) -> dict:
    """各馬の前走（同じ芝ダ・180 日以内）の指数でレース内 1 位の馬の勝率・3着内率。"""
    x = d.sort_values(["horse_id", "date"])
    prev = x.groupby("horse_id").shift(1)
    gap = (x["date"] - prev["date"]).dt.days
    x = x.assign(prev_fig=np.where(gap.le(180) & (x["surface"] == prev["surface"]), prev[col], np.nan))
    x = x[x["year"].isin(years)]
    cov = x.groupby("race_key")["prev_fig"].apply(lambda s: s.notna().mean())
    races = cov[cov >= 0.7].index
    y = x[x["race_key"].isin(races)].copy()
    y["rk"] = y.groupby("race_key")["prev_fig"].rank(ascending=False, method="first")
    top = y[y["rk"] == 1]
    return {"races": int(len(top)), "win": float((top["finish"] == 1).mean()),
            "top3": float((top["finish"] <= 3).mean())}


# --------------------------------------------------------------------------- commands
def _cache_runs() -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = OUT_DIR / f"_runs_src_v{LOAD_VERSION}.parquet"
    srcs = [SRC_ENH, *SRC_TARGET] + (sorted(SRC_EXTRA_DIR.glob("*.csv")) if SRC_EXTRA_DIR.exists() else [])         + _weekly_files()
    mt = max(p.stat().st_mtime for p in srcs if p.exists())
    if cache.exists() and cache.stat().st_mtime > mt:
        return pd.read_parquet(cache)
    d = load_runs()
    d.to_parquet(cache)
    return d


def load_params() -> Params:
    pf = OUT_DIR / "params.json"
    if pf.exists():
        return Params(**json.loads(pf.read_text(encoding="utf-8")))
    return Params()


def cmd_build(args) -> None:
    p = load_params()
    d = _cache_runs()
    log.info("runs %d  %s〜%s", len(d), d["date"].min().date(), d["date"].max().date())
    f = compute_figures(d, p)
    off = fit_course_offsets(f, p, fit_years=range(2014, 2023))
    f = apply_course_offsets(f, off)
    # 中心合わせ: 芝・ダートそれぞれ、基準タイムの標本（良・古馬1〜2勝・2着〜上位33%）の中央値を 100 に
    ref = _ref_mask(f)
    shift = {}
    for sf in ("芝", "ダ"):
        m = ref & (f["surface"] == sf) & f["year"].between(2014, 2022)
        shift[sf] = float(100.0 - f.loc[m, "fig"].median())
        f.loc[f["surface"] == sf, "fig"] += shift[sf]
    log.info("中心合わせ %s", shift)
    off["_center_shift"] = shift
    f["fig"] = f["fig"].clip(20, 170)
    (OUT_DIR / "course_offsets.json").write_text(json.dumps(off, ensure_ascii=False, indent=1), encoding="utf-8")
    keep = ["date", "year", "month", "race_key", "place", "rno", "race_name", "klass", "age_grp", "surface", "course_code",
            "dist", "going", "horse", "horse_id", "sex", "age", "weight", "field", "fsize", "umaban", "finish",
            "margin", "ninki", "c4", "jockey", "variant", "pace", "pace_adj", "course_adj", "f_pts", "a_pts",
            "fig0", "fig"]
    f[keep].to_parquet(OUT_DIR / "runs.parquet")
    log.info("figures %d → %s", len(f), OUT_DIR / "runs.parquet")
    _report(f)


def _report(f: pd.DataFrame) -> None:
    for col in ("fig0", "fig1", "fig"):
        if col in f:
            print(f"再現性 {col:5s} 2016-21 {reliability(f, col, range(2016, 2022))}  2022-25 {reliability(f, col, range(2022, 2026))}")
            print(f"  レース内相関 {col:5s} 2016-21 {race_corr(f, col, range(2016, 2022))}  2022-25 {race_corr(f, col, range(2022, 2026))}")
    ok = f["finish"] == 1
    lv = f[ok & (f["going"] == "良")].groupby(["klass"])["fig"].median().reindex(CLASS_ORDER)
    print("勝ち馬の指数（中央値, 良）:", {k: round(v, 1) for k, v in lv.items() if pd.notna(v)})
    lv2 = f[ok].groupby(["surface", "klass"])["fig"].median().unstack().reindex(columns=CLASS_ORDER)
    print(lv2.round(1).to_string())
    # 距離帯ごとの格の差（G1 勝ち − 1勝 勝ち）が距離で揃っているか
    f2 = f[ok].assign(db=pd.cut(f.loc[ok, "dist"], [0, 1300, 1500, 1700, 2100, 2500, 4000]))
    gap = f2.groupby(["surface", "db"], observed=True).apply(
        lambda g: g.loc[g["klass"].isin(["3勝"]), "fig"].median() - g.loc[g["klass"] == "1勝", "fig"].median())
    print("3勝勝ち − 1勝勝ち（距離帯ごと）:", gap.round(1).to_dict())
    print("前走指数1位の的中（2024-25）:", within_race_hit(f, "fig", [2024, 2025]))
    print("前走指数1位の的中（2016-21）:", within_race_hit(f, "fig", list(range(2016, 2022))))


def cmd_report(args) -> None:
    f = pd.read_parquet(OUT_DIR / "runs.parquet")
    _report(f)


def cmd_tune(args) -> None:
    d = _cache_runs()
    years = list(range(2016, 2022))
    base = load_params()
    grid = {
        "k_w": [0.0, 0.04, 0.07, 0.10, 0.14],
        "var_lambda": [0.5, 1.5, 3.0, 6.0],
        "spp_floor_dist": [1000, 1200, 1400, 1600],
        "pace_a": [0.3, 0.45, 0.6],
        "pace_b": [0.0, 0.15, 0.3],
        "pace_g": [0.2, 0.35, 0.5],
        "pace_tau": [0.0, 4.0],
    }
    best = base
    # 座標降下: 1 つずつ動かして再現性（2016-21 の連続出走の相関）が最大の値を採る
    def score(pp: Params) -> float:
        f = compute_figures(d, pp, years=[2015] + years)
        return race_corr(f, "fig", years)["rho"]
    cur = score(best)
    log.info("start r=%.4f %s", cur, best)
    for rnd in range(2):
        for k, vals in grid.items():
            res = {}
            for v in vals:
                pp = replace(best, **{k: v})
                res[v] = cur if getattr(best, k) == v else score(pp)
                log.info("  %s=%s r=%.4f", k, v, res[v])
            v_best = max(res, key=res.get)
            if res[v_best] > cur + 1e-4:
                best, cur = replace(best, **{k: v_best}), res[v_best]
            log.info("round %d %s → %s (r=%.4f)", rnd, k, getattr(best, k), cur)
    (OUT_DIR / "params.json").write_text(json.dumps(asdict(best), indent=1), encoding="utf-8")
    log.info("best r=%.4f %s", cur, best)


# --------------------------------------------------------------------------- 予測指数
CLASS_RANK = {c: i for i, c in enumerate(CLASS_ORDER)}
PRED_FEATS = ["l1", "l2", "l3", "l4", "l5", "l1_same", "best3_same", "mean3_same", "wmean5", "std5",
              "n_runs", "n_same", "days", "dist_diff", "dist_ratio_last", "surf_chg", "cls_now", "cls_diff",
              "l1_finish_rel", "l1_margin", "l1_pace", "l1_c4_rel", "age", "sex_f", "month", "age_month",
              "going_n", "surface_n", "field", "dist", "l1_going_n", "same_dist_best", "trend3"]
GOING_N = {"良": 0, "稍": 1, "重": 2, "不": 3}


def pred_features(f: pd.DataFrame) -> pd.DataFrame:
    """各走について、その走より前の実走指数だけから作る特徴（未来は見ない）。"""
    x = f.sort_values(["horse_id", "date", "race_key"]).copy()
    x["cls"] = x["klass"].map(CLASS_RANK)
    x["fin_rel"] = (x["finish"] - 1) / np.maximum(x["fsize"] - 1, 1)
    x["c4_rel"] = (x["c4"] - 1) / np.maximum(x["fsize"] - 1, 1)
    # 過去走（ラグ）は実走指数のある行だけから取る。指数の無い行（予測したい当日の行・取消・
    # タイム不明）は後の行のラグにならない。自分自身はラグに入らない（pos は自分より前の数）。
    is_hist = x["fig"].notna().to_numpy()
    hid = x["horse_id"].to_numpy()
    h = x[is_hist].reset_index(drop=True)
    start = pd.Series(np.arange(len(h)), index=h["horse_id"]).groupby(level=0).min()
    csum = pd.Series(is_hist.astype(int), index=x.index).groupby(hid).cumsum().to_numpy()
    pos = csum - is_hist.astype(int)                    # この行より前の履歴の数
    st = pd.Series(hid).map(start).to_numpy()
    lag = {}
    for k in range(1, 6):
        idx = st + pos - k
        ok = (~pd.isna(st)) & (pos - k >= 0)
        take = np.where(ok, idx, 0).astype(int)
        s = h.iloc[take].set_index(x.index) if len(h) else x.iloc[:0].reindex(x.index)
        s = s.where(pd.Series(ok, index=x.index), np.nan) if len(h) else s
        lag[k] = s
        x[f"l{k}"] = s["fig"]
    x["n_runs"] = pos
    x["days"] = (x["date"] - lag[1]["date"]).dt.days
    x["dist_diff"] = x["dist"] - lag[1]["dist"]
    x["dist_ratio_last"] = x["dist"] / lag[1]["dist"]
    x["surf_chg"] = (x["surface"] != lag[1]["surface"]).astype(float).where(lag[1]["fig"].notna())
    x["cls_now"] = x["cls"]
    x["cls_diff"] = x["cls"] - lag[1]["cls"]
    x["l1_finish_rel"] = lag[1]["fin_rel"]
    x["l1_margin"] = lag[1]["margin"].clip(upper=5)
    x["l1_pace"] = lag[1]["pace"]
    x["l1_c4_rel"] = lag[1]["c4_rel"]
    x["l1_going_n"] = lag[1]["going"].map(GOING_N)
    L = np.column_stack([x[f"l{k}"] for k in range(1, 6)])
    S = np.column_stack([(lag[k]["surface"] == x["surface"]).to_numpy() for k in range(1, 6)])
    D = np.column_stack([(lag[k]["dist"] - x["dist"]).abs().le(200).to_numpy() for k in range(1, 6)])
    Ls = np.where(S, L, np.nan)
    with np.errstate(all="ignore"):
        x["l1_same"] = Ls[:, 0]
        x["best3_same"] = np.nanmax(Ls[:, :3], axis=1)
        x["mean3_same"] = np.nanmean(Ls[:, :3], axis=1)
        wts = np.array([0.35, 0.25, 0.18, 0.12, 0.10])
        m = ~np.isnan(L)
        x["wmean5"] = np.nansum(np.nan_to_num(L) * wts, axis=1) / np.where(m.any(1), (m * wts).sum(1), np.nan)
        x["std5"] = np.nanstd(L, axis=1)
        x["n_same"] = S.sum(1) * 1.0
        x["same_dist_best"] = np.nanmax(np.where(S & D, L, np.nan), axis=1)
        x["trend3"] = L[:, 0] - np.nanmean(L[:, 1:3], axis=1)
    x["sex_f"] = (x["sex"] == "牝").astype(int)
    x["age_month"] = x["age"] * 12 + x["month"]
    x["going_n"] = x["going"].map(GOING_N)
    x["surface_n"] = (x["surface"] == "芝").astype(int)
    return x


def _wt_pts(df: pd.DataFrame, p: Params) -> pd.Series:
    """今回の斤量の効き（pt）。指数は 57kg 換算なので、予測を今回の斤量に戻すときに引く。"""
    return p.k_w * (df["weight"] - 57.0) * df["dist"] / 1000.0 / spp(df["dist"], p)


def _rank_table(df: pd.DataFrame, score: str) -> pd.DataFrame:
    df = df.copy()
    df["rk"] = df.groupby("race_key")[score].rank(ascending=False, method="first")
    df["rk_b"] = pd.cut(df["rk"], [0, 1, 2, 3, 4, 5, 9, 99], labels=["1位", "2位", "3位", "4位", "5位", "6-9位", "10位-"])
    t = df.groupby("rk_b", observed=True).agg(n=("finish", "size"), win=("finish", lambda s: (s == 1).mean()),
                                               ren=("finish", lambda s: (s <= 2).mean()),
                                               top3=("finish", lambda s: (s <= 3).mean()))
    return t


def _gap_table(df: pd.DataFrame, score: str) -> pd.DataFrame:
    s = df.sort_values(score, ascending=False).groupby("race_key")
    top = s.head(1).set_index("race_key")
    second = s.nth(1).set_index("race_key")[score]
    top["gap"] = top[score] - second.reindex(top.index)
    top["gb"] = pd.cut(top["gap"], [-0.01, 1, 2, 3, 5, 8, 999], labels=["0-1", "1-2", "2-3", "3-5", "5-8", "8+"])
    return top.groupby("gb", observed=True).agg(races=("finish", "size"), win=("finish", lambda s: (s == 1).mean()),
                                                ren=("finish", lambda s: (s <= 2).mean()),
                                                top3=("finish", lambda s: (s <= 3).mean()))


def cmd_predict_train(args) -> None:
    import lightgbm as lgb
    p = load_params()
    f = pd.read_parquet(OUT_DIR / "runs.parquet")
    x = pred_features(f)
    have = x["l1"].notna()
    tr = have & x["year"].between(2013, 2022)
    va = have & (x["year"] == 2023)
    params = dict(objective="huber", alpha=8.0, learning_rate=0.03, num_leaves=63, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1)
    dtr = lgb.Dataset(x.loc[tr, PRED_FEATS], x.loc[tr, "fig"])
    dva = lgb.Dataset(x.loc[va, PRED_FEATS], x.loc[va, "fig"])
    m = lgb.train(params, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
    m.save_model(str(OUT_DIR / "pred_model.txt"))
    x["pred"] = np.where(have, m.predict(x[PRED_FEATS]), np.nan)
    x["pred_w"] = x["pred"] - _wt_pts(x, p)          # 今回の斤量込み
    out = x[["race_key", "date", "year", "horse_id", "horse", "klass", "surface", "dist", "finish", "fsize",
             "ninki", "fig", "pred", "pred_w", "n_runs"]]
    out.to_parquet(OUT_DIR / "pred.parquet")
    imp = pd.Series(m.feature_importance("gain"), index=PRED_FEATS).sort_values(ascending=False)
    print("gain 上位:", (imp / imp.sum()).head(12).round(3).to_dict())
    _pred_eval(out)


def _pred_eval(out: pd.DataFrame) -> None:
    for name, yrs in (("調整 2023", [2023]), ("検証 2024-26", [2024, 2025, 2026])):
        y = out[out["year"].isin(yrs) & (out["klass"] != "新馬")].copy()
        cov = y.groupby("race_key")["pred_w"].transform(lambda s: s.notna().mean())
        y = y[cov >= 0.5]
        y["score"] = y["pred_w"].fillna(-1e9)
        print(f"\n== {name}: {y['race_key'].nunique()} レース（新馬除く・予測のある馬が半数以上）")
        print("予測指数の順位別:\n" + (_rank_table(y, "score") * [1, 100, 100, 100]).round(1).to_string())
        print("1位と2位の差別:\n" + (_gap_table(y[y["pred_w"].notna()], "score") * [1, 100, 100, 100]).round(1).to_string())
        if y["ninki"].notna().any():
            y["pop"] = -y["ninki"]
            print("（参考）1番人気:", (_rank_table(y.dropna(subset=["ninki"]), "pop").iloc[0, 1:] * 100).round(1).to_dict())
        ok = y["pred"].notna() & y["fig"].notna()
        print("予測と実走の相関 r=%.3f  MAE=%.2f" % (np.corrcoef(y.loc[ok, "pred"], y.loc[ok, "fig"])[0, 1],
                                                 (y.loc[ok, "pred"] - y.loc[ok, "fig"]).abs().mean()))


def day_entries(date: str) -> pd.DataFrame:
    """その日の出走表（障害除く）。"""
    from weekly_parse import parse_csv
    for name in ("weekly_parse", "training_join"):
        logging.getLogger(name).setLevel(logging.WARNING)
    path = next((d / f"{date}.csv" for d in WEEKLY_DIRS if (d / f"{date}.csv").exists()), None)
    if path is None:
        raise SystemExit(f"出走表が無い: {date}.csv")
    w = parse_csv(path)
    dt = pd.to_datetime(date, format="%Y%m%d")
    surf = w["芝・ダ"].astype(str).str[0]
    e = pd.DataFrame({
        "date": dt, "year": dt.year, "month": dt.month, "place": w["場所"].astype(str).str.strip(),
        "rno": pd.to_numeric(w["R"], errors="coerce").astype(int), "race_name": w["レース名"].astype(str).str.strip(),
        "klass": w["クラス名"].map(_norm_class), "surface": surf, "dist": pd.to_numeric(w["距離"], errors="coerce"),
        "going": w["馬場状態"].astype(str).str.replace("(暫定)", "", regex=False).str.strip().str[0],
        "horse": w["馬名"].astype(str).str.strip(), "sex": w["性別"].astype(str).str.strip(),
        "age": pd.to_numeric(w["年齢"], errors="coerce"), "weight": pd.to_numeric(w["斤量"], errors="coerce"),
        "umaban": pd.to_numeric(w["馬番"], errors="coerce"),
    })
    e = e[e["surface"].isin(["芝", "ダ"])].copy()
    e["race_key"] = date + e["place"] + e["rno"].astype(str).str.zfill(2)
    # 出走表 CSV は同じ馬が 2 行入っている日がある（4/12 など）。残すと、まとめて計算したときに
    # 片方の行の「直近の走」にその日自身の結果が入る（先読み）ので 1 頭 1 行にする。
    e = e.drop_duplicates(["race_key", "umaban"]).drop_duplicates(["date", "horse"])
    e["field"] = e.groupby("race_key")["horse"].transform("size")
    e["fsize"] = e["field"]
    for c in ("finish", "c4", "fig", "margin", "pace"):
        e[c] = np.nan
    return e


def score_days(dates: list[str]) -> dict[str, dict]:
    """複数の開催日の予測指数（今回の斤量込み）と各馬の近 5 走の実走指数を、特徴を 1 回だけ作って計算する。
    各日の予測は、その日より前の走だけから作る（その日以降の結果は特徴に入らない）。"""
    import lightgbm as lgb
    p = load_params()
    runs = pd.read_parquet(OUT_DIR / "runs.parquet")
    model = lgb.Booster(model_file=str(OUT_DIR / "pred_model.txt"))
    dates = sorted(dates)
    E = pd.concat([day_entries(d) for d in dates], ignore_index=True)
    first = E["date"].min()
    # 対象期間の実走は出走表の行に載せ替える（同じ日の同じ馬が二重にならないように）
    act = runs[runs["date"] >= first].drop_duplicates(["date", "horse"])
    cols = ["fig", "finish", "c4", "margin", "pace", "fsize", "going", "weight"]
    E = E.drop(columns=["fig", "finish", "c4", "margin", "pace", "fsize"]).merge(
        act[["date", "horse"] + cols].rename(columns={"going": "going_a", "weight": "weight_a"}),
        on=["date", "horse"], how="left")
    # 走った後の行は確定の馬場状態・斤量で（1 日ずつ計算したときの履歴と同じにする）
    E["going"] = E["going_a"].fillna(E["going"])
    E["weight"] = E["weight_a"].fillna(E["weight"])
    E = E.drop(columns=["going_a", "weight_a"])
    E["fsize"] = E["fsize"].fillna(E["field"])
    ids = runs.sort_values("date").drop_duplicates("horse", keep="last").set_index("horse")["horse_id"]
    E["horse_id"] = E["horse"].map(ids).fillna("N:" + E["horse"])
    base = runs[runs["date"] < first]
    allx = pred_features(pd.concat([base, E.assign(_today=True)], ignore_index=True))
    t = allx[allx["_today"].fillna(False).astype(bool)].copy()
    have = t["l1"].notna()
    t["pred"] = np.nan
    if have.any():
        t.loc[have, "pred"] = model.predict(t.loc[have, PRED_FEATS])
    t["pred_w"] = t["pred"] - _wt_pts(t, p)
    t["rank"] = t.groupby("race_key")["pred_w"].rank(ascending=False, method="min")
    H = pd.concat([base, t[t["fig"].notna()]], ignore_index=True).sort_values("date")
    out = {}
    for d in dates:
        dt = pd.Timestamp(d)
        td = t[t["date"] == dt]
        hist = H[(H["date"] < dt) & H["horse_id"].isin(td["horse_id"])].groupby("horse_id").tail(5)
        recent = {hid: [{"date": r.date.strftime("%Y%m%d"), "place": r.place, "surface": r.surface,
                         "dist": int(r.dist), "klass": r.klass,
                         "finish": int(r.finish) if pd.notna(r.finish) else None,
                         "fig": round(float(r.fig), 1)} for r in g.itertuples()][::-1]
                  for hid, g in hist.groupby("horse_id")}
        races = []
        for rk, g in td.sort_values(["place", "rno", "umaban"]).groupby("race_key", sort=False):
            r0 = g.iloc[0]
            hs = [{"umaban": int(r.umaban), "name": r.horse,
                   "pred": None if pd.isna(r.pred_w) else round(float(r.pred_w), 1),
                   "rank": None if pd.isna(r.rank) else int(r.rank),
                   "n_runs": int(r.n_runs), "recent": recent.get(r.horse_id, [])} for r in g.itertuples()]
            races.append({"race_key": rk, "place": r0.place, "rno": int(r0.rno), "race_name": r0.race_name,
                          "klass": r0.klass, "surface": r0.surface, "dist": int(r0.dist), "horses": hs})
        out[d] = {"date": d, "version": "si-20261115", "races": races}
    return out


def score_day(date: str) -> dict:
    return score_days([date])[date]


def cmd_score_days(args) -> None:
    """score-days 20260418 20261115 → 期間内の出走表がある全開催日を一括で作る。"""
    lo, hi = args.start, args.end or args.start
    ds = [f.stem for f in _weekly_files() if lo <= f.stem <= hi]
    res = score_days(ds)
    for d, out in res.items():
        (OUT_DIR / f"day_{d}.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    log.info("score-days %d 日 → %s", len(res), OUT_DIR)


def cmd_score_day(args) -> None:
    out = score_day(args.date)
    (OUT_DIR / f"day_{args.date}.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    for r in out["races"][: args.show]:
        hs = sorted(r["horses"], key=lambda h: -(h["pred"] if h["pred"] is not None else -1e9))
        print()
        print(f"{r['place']}{r['rno']}R {r['race_name']} {r['surface']}{r['dist']} {r['klass']}")
        for h in hs:
            rc = " ".join(f"{x['fig']:.0f}" for x in h["recent"][:5])
            pr = "  —  " if h["pred"] is None else f"{h['pred']:5.1f}"
            print(f"  {h['rank'] or '-':>2} {h['umaban']:>2} {h['name']:<12} 予測 {pr}   近走 {rc}")


SITE_META = BASE / "site" / "data" / "speed_meta.json"


def _kekka_finish(dates: list[str]) -> pd.DataFrame:
    ks = []
    for d in dates:
        f = next((x / f"{d}.csv" for x in (BASE / "data" / "kekka", Path("E:/PyCaLiAI/data/kekka"))
                  if (x / f"{d}.csv").exists()), None)
        if f is None:
            continue
        k = pd.read_csv(f, encoding="cp932", usecols=["場所", "Ｒ", "馬番", "確定着順"])
        k["date"] = d
        ks.append(k)
    if not ks:
        return pd.DataFrame(columns=["date", "place", "rno", "umaban", "finish"])
    K = pd.concat(ks).rename(columns={"場所": "place", "Ｒ": "rno", "馬番": "umaban", "確定着順": "finish"})
    K["finish"] = pd.to_numeric(K["finish"], errors="coerce")
    return K[K["finish"] > 0]


def forward_frame() -> pd.DataFrame:
    """day_*.json（各日その日より前の情報だけで作った予測指数）と実際の着順を突き合わせる。"""
    rows = []
    for f in sorted(OUT_DIR.glob("day_2026*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        for r in d["races"]:
            for h in r["horses"]:
                rows.append((d["date"], r["place"], r["rno"], r["klass"], h["umaban"], h["pred"]))
    D = pd.DataFrame(rows, columns=["date", "place", "rno", "klass", "umaban", "pred"])
    K = _kekka_finish(sorted(D["date"].unique()))
    y = D.merge(K, on=["date", "place", "rno", "umaban"], how="inner")
    y["race_key"] = y["date"] + y["place"] + y["rno"].astype(str)
    y = y[y["klass"] != "新馬"]
    cov = y.groupby("race_key")["pred"].transform(lambda s: s.notna().mean())
    y = y[cov >= 0.5].copy()
    y["score"] = y["pred"].fillna(-1e9)
    return y


def _tables(y: pd.DataFrame) -> dict:
    rt = _rank_table(y, "score")
    gt = _gap_table(y[y["score"] > -1e8], "score")
    conv = lambda t, key: [{key: str(i), "n": int(r.iloc[0]), "win": round(r["win"] * 100, 1),
                            "ren": round(r["ren"] * 100, 1), "top3": round(r["top3"] * 100, 1)} for i, r in t.iterrows()]
    return {"races": int(y["race_key"].nunique()), "by_rank": conv(rt, "rank"), "by_gap": conv(gt, "gap")}


def cmd_export_meta(args) -> None:
    """サイトの解説ページ（speed.html）が読む数字を site/data/speed_meta.json に書く。集計値だけ。"""
    f = pd.read_parquet(OUT_DIR / "runs.parquet")
    w = f[(f["finish"] == 1) & (f["going"] == "良") & f["year"].between(2016, 2025)]
    lv = w.groupby(["surface", "klass"])["fig"].median().unstack().reindex(columns=CLASS_ORDER)
    q = w.groupby(["surface", "klass"])["fig"].quantile(0.75).unstack().reindex(columns=CLASS_ORDER)
    levels = {sf: [{"klass": k, "median": round(float(lv.loc[sf, k]), 1), "p75": round(float(q.loc[sf, k]), 1)}
                   for k in CLASS_ORDER if pd.notna(lv.loc[sf, k])] for sf in ("芝", "ダ")}
    pred = pd.read_parquet(OUT_DIR / "pred.parquet")
    t = pred[pred["year"].isin([2024, 2025]) & (pred["klass"] != "新馬")].copy()
    cov = t.groupby("race_key")["pred_w"].transform(lambda s: s.notna().mean())
    t = t[cov >= 0.5]
    t["score"] = t["pred_w"].fillna(-1e9)
    p = load_params()
    meta = {
        "version": "si-20261115",
        "built_at": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "params": {"k_w": p.k_w, "spp_base": p.spp_base, "pace_a": p.pace_a, "pace_b": p.pace_b, "pace_g": p.pace_g},
        "levels": levels,
        "test_2024_25": _tables(t),
        "forward_2026": _tables(forward_frame()),
        "race_corr": {"fig0": race_corr(f, "fig0", range(2022, 2026))["rho"], "fig": race_corr(f, "fig", range(2022, 2026))["rho"]},
    }
    SITE_META.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("test_2024_25", "forward_2026")}, ensure_ascii=False)[:1500])


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    sub.add_parser("tune")
    sub.add_parser("report")
    sub.add_parser("predict-train")
    sd = sub.add_parser("score-day")
    sd.add_argument("date")
    sd.add_argument("--show", type=int, default=3)
    sub.add_parser("export-meta")
    sds = sub.add_parser("score-days")
    sds.add_argument("start")
    sds.add_argument("end", nargs="?")
    a = ap.parse_args(argv)
    {"build": cmd_build, "tune": cmd_tune, "report": cmd_report, "predict-train": cmd_predict_train,
     "score-day": cmd_score_day, "score-days": cmd_score_days,
     "export-meta": cmd_export_meta}[a.cmd](a)


if __name__ == "__main__":
    main()
