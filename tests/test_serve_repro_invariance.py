# -*- coding: utf-8 -*-
"""配信再現性の不変性検査 T1〜T6 (合成データ、hermetic)。

仕様: docs/research/SERVE_REPRODUCIBILITY_FIX_SPEC_20261004.md §5
  T1  後日の履歴追加に対する不変   : 履歴を date < d で切り落とした版と全履歴版で特徴が完全一致
  T1r 同上 (騎手・調教師の直近 4 特徴だけ)
  T2  入力行の並べ替えに対する不変 : 履歴の行 (種 5 通り) と出走表の馬の順を並べ替えても完全一致
  T3  同定の day-start 限定        : 切り落とし版と全履歴版で、馬ごとの同定状態と個体 ID が完全一致
  T5  同順位の不在                 : 順序キー K=(date, post_time, race_id, umaban) の重複はエラー
  T6  無言 fallback の不在         : post_time / umaban 列の無い履歴はエラーで停止
実データ (2026 年の開催日) での同じ検査は analysis/serve_repro_invariance.py が行い、
reports/serve_repro/ に記録する。

注 (現行コードでの実測): T2 は現行コードでも通る。_HistoryIndex.__init__ が履歴を
(ped_id, date) — 1 走を一意に指す全順序 — で先に並べ直すため、parquet の行順は結果に
届かない。D1 (不安定ソート) は配列そのものが変わるとき (履歴の追加 = T1r) に現れる。
"""
from __future__ import annotations

import inspect
import json

import numpy as np
import pandas as pd
import pytest

import serve_history_feats as shf

D = 20261003                 # 予測する開催日
JOCKEY, TRAINER = 501, 601
BULK_JOCKEY, BULK_TRAINER = 502, 602   # 不安定ソートが現れる大きさの同日行を持つ主体
ROLL4 = ["jockey_fuku30", "jockey_fuku90", "trainer_fuku30", "trainer_fuku90"]
QUOTED_SIRE = '"American Pharoah"'


def _hhmm(k: int) -> int:
    """k 番目のレースの発走時刻 (HHMM 整数、09:30 から 30 分刻み)。"""
    minutes = 9 * 60 + 30 + 30 * k
    return (minutes // 60) * 100 + minutes % 60


def _rows_for_day(date: int, n: int, first_ped: int, pos_cycle: list[int], venue: str = "05") -> list[dict]:
    """騎手 JOCKEY・調教師 TRAINER が 1 日に n レース騎乗した行 (各行は別の馬・別のレース)。"""
    out = []
    for k in range(n):
        rno = k + 1
        out.append(dict(
            ped_id=first_ped + k, name=f"ウマ{first_ped + k}", sire="ダミー父", birth_year=2022,
            date=date, place="東京", surface="芝", dist=1600.0,
            pos=float(pos_cycle[k % len(pos_cycle)]),
            jockey_code=float(JOCKEY), trainer_code=float(TRAINER),
            race_id=f"{date}{venue}0101{rno:02d}", post_time=_hhmm(k), umaban=1 + (k % 9),
            src="master" if date < 20260101 else "kekka2026"))
    return out


def _bulk_rows() -> list[dict]:
    """騎手 BULK_JOCKEY の行: B 日 120 行・C 日 20 行・予測日以後 200 行 (馬 ID は乱数、種固定)。

    窓 30/90 の端が B 日の 120 行の途中に落ちる。B 日の中の並びを決めるのは日付だけの
    不安定ソートなので、予測日以後の行が増えて配列が変わると、窓に入る B 日の行が入れ替わる
    (現行コードで実測: 種 0 で騎手直近 30 走複勝率 0.233 → 0.267)。
    """
    rng = np.random.default_rng(0)
    out = []
    for date, n, venue in ((20260906, 120, "06"), (20260913, 20, "06"),
                           (D, 100, "07"), (20261010, 100, "07")):
        for k in range(n):
            out.append(dict(
                ped_id=int(rng.integers(10**8, 10**9)), name=f"バルク{date}{k}", sire="ダミー父",
                birth_year=2022, date=date, place="中山", surface="ダ", dist=1200.0,
                pos=float(rng.integers(1, 15)),
                jockey_code=float(BULK_JOCKEY), trainer_code=float(BULK_TRAINER),
                race_id=f"{date}{venue}{1 + k // 12:02d}{1 + (k // 12) % 9:02d}{1 + k % 12:02d}",
                post_time=_hhmm(k % 12), umaban=1 + k // 12,
                src="kekka2026"))
    return out


def _history() -> pd.DataFrame:
    rows: list[dict] = _bulk_rows()
    # 直近集計の窓 (30 行) の端が「同じ日付の 12 行」の途中に落ちる構成:
    #   A 日 8 行 → B 日 12 行 → C 日 25 行。窓 30 = C の 25 行 + B の 5 行。
    #   B の 12 行は着順がばらつくので、B のどの 5 行を含めるかで値が変わる。
    rows += _rows_for_day(20260830, 8, 10000, [1, 5, 9])
    rows += _rows_for_day(20260906, 12, 11000, [1, 8, 2, 9, 3, 10, 11, 12, 1, 13, 14, 2])
    rows += _rows_for_day(20260913, 25, 12000, [4, 6, 1, 7, 3])
    # 予測日以後の行 (全履歴版にだけある)
    rows += _rows_for_day(D, 10, 13000, [2, 9])
    rows += _rows_for_day(20261010, 20, 14000, [1, 12, 5])
    # D2: 2025 年 (master 由来) は父名が引用符なし、予測日より後の 2026 行 (出走表由来) は引用符つき
    rows.append(dict(ped_id=20001, name="クォートホース", sire="American Pharoah", birth_year=2022,
                     date=20250601, place="東京", surface="芝", dist=1600.0, pos=2.0,
                     jockey_code=float(JOCKEY), trainer_code=float(TRAINER),
                     race_id="2025060105010105", post_time=1200, umaban=3, src="master"))
    rows.append(dict(ped_id=20001, name="クォートホース", sire=QUOTED_SIRE, birth_year=2022,
                     date=20261010, place="東京", surface="芝", dist=1600.0, pos=1.0,
                     jockey_code=float(JOCKEY), trainer_code=float(TRAINER),
                     race_id="2026101005020107", post_time=1300, umaban=4, src="kekka2026"))
    # 予測日より後にしか行の無い馬 (予測日朝の時点では未出走)
    rows.append(dict(ped_id=-30001, name="ミライバ", sire="ダミー父", birth_year=2023,
                     date=20261010, place="東京", surface="芝", dist=1400.0, pos=1.0,
                     jockey_code=float(JOCKEY), trainer_code=float(TRAINER),
                     race_id="2026101005020103", post_time=1100, umaban=5, src="kekka2026"))
    # 普通に解決する馬 (予測日より前に 3 走)
    for j, (dt, pos) in enumerate(((20250510, 3.0), (20250720, 1.0), (20260913, 5.0))):
        rows.append(dict(ped_id=20002, name="フツウバ", sire="ダミー父", birth_year=2021,
                         date=dt, place="東京", surface="芝", dist=1800.0, pos=pos,
                         jockey_code=float(JOCKEY), trainer_code=float(TRAINER),
                         race_id=f"{dt}05010111", post_time=1500, umaban=6 + j, src="master"))
    h = pd.DataFrame(rows)
    h["birth_year"] = h["birth_year"].astype("Int32")
    h["date"] = h["date"].astype(np.int32)
    h["ped_id"] = h["ped_id"].astype(np.int64)
    return h.sort_values(["ped_id", "date"]).reset_index(drop=True)


def _weekly() -> pd.DataFrame:
    names = ["クォートホース", "ミライバ", "フツウバ", "シンバ", "バルク騎乗"]
    sires = [QUOTED_SIRE, "ダミー父", "ダミー父", "ダミー父", "ダミー父"]
    ages = [4, 3, 5, 2, 3]
    jockeys = ["騎手J", "騎手J", "騎手J", "騎手J", "騎手K"]
    trainers = ["調教師T", "調教師T", "調教師T", "調教師T", "調教師U"]
    return pd.DataFrame({"馬名": names, "種牡馬": sires, "年齢": ages, "日付": str(D),
                         "場所": "東京", "芝・ダ": "芝", "距離": 1600,
                         "騎手": jockeys, "調教師": trainers})


def _base(tmp_path, hist: pd.DataFrame, tag: str):
    base = tmp_path / tag
    (base / "data").mkdir(parents=True)
    hist.to_parquet(base / "data" / "_horse_history.parquet", index=False)
    (base / "data" / "serve_code_maps.json").write_text(
        json.dumps({"jockey": {"騎手J": JOCKEY, "騎手K": BULK_JOCKEY},
                    "trainer": {"調教師T": TRAINER, "調教師U": BULK_TRAINER}}, ensure_ascii=False),
        encoding="utf-8")
    return base


def _features(base, weekly: pd.DataFrame) -> pd.DataFrame:
    df = weekly.copy()
    shf.fill_history_features(df, base=base)
    return df.set_index("馬名")[shf.NUM_FEATS + shf.CAT_FEATS].sort_index()


def _same(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    """完全一致 (欠損同士は一致扱い)。"""
    a, b = a.sort_index(), b.sort_index()
    if list(a.index) != list(b.index) or list(a.columns) != list(b.columns):
        return False
    for c in a.columns:
        x, y = a[c].to_numpy(), b[c].to_numpy()
        if a[c].dtype == object or b[c].dtype == object:
            if not np.array_equal(x.astype(str), y.astype(str)):
                return False
        elif not np.array_equal(x.astype(float), y.astype(float), equal_nan=True):
            return False
    return True


def _truncated(hist: pd.DataFrame) -> pd.DataFrame:
    return hist[hist["date"] < D].reset_index(drop=True)


def _identify(idx, name: str, sire: str, birth_year: int, race_date: int):
    """(同定状態, 個体 ID)。as-of 版の resolve は race_date を取る。"""
    params = inspect.signature(idx.resolve).parameters
    if "race_date" in params:
        ent, status = idx.resolve(name, sire, birth_year, race_date=race_date)
    else:
        ent, status = idx.resolve(name, sire, birth_year)
    return status, (None if ent is None else int(ent["ped_id"]))


def _clean(s: str) -> str:
    return shf._clean_name(s)


@pytest.mark.xfail(strict=True, reason="D2: 馬同定が全期間の最終行の父名を使う (spec §1)")
def test_T1_history_append_invariance_all_features(tmp_path):
    hist = _history()
    full = _features(_base(tmp_path, hist, "full"), _weekly())
    trunc = _features(_base(tmp_path, _truncated(hist), "trunc"), _weekly())
    assert _same(full, trunc)


def test_T1r_history_append_invariance_rolling(tmp_path):
    hist = _history()
    full = _features(_base(tmp_path, hist, "full"), _weekly())[ROLL4]
    trunc = _features(_base(tmp_path, _truncated(hist), "trunc"), _weekly())[ROLL4]
    assert _same(full, trunc)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_T2_row_order_invariance(tmp_path, seed):
    hist = _history()
    ref = _features(_base(tmp_path, hist, "ref"), _weekly())
    rng = np.random.default_rng(seed)
    shuffled = hist.iloc[rng.permutation(len(hist))].reset_index(drop=True)
    weekly = _weekly()
    weekly = weekly.iloc[rng.permutation(len(weekly))].reset_index(drop=True)
    got = _features(_base(tmp_path, shuffled, f"shuf{seed}"), weekly)
    assert _same(ref, got)


@pytest.mark.xfail(strict=True, reason="D2: 予測日より後の行が同定の候補・照合属性に使われる (spec §1)")
def test_T3_identity_is_day_start_limited():
    hist = _history()
    full_idx = shf._HistoryIndex(hist)
    trunc_idx = shf._HistoryIndex(_truncated(hist))
    w = _weekly()
    mism = []
    for _, r in w.iterrows():
        by = int(D // 10000 - int(r["年齢"]))
        a = _identify(full_idx, _clean(r["馬名"]), _clean(r["種牡馬"]), by, D)
        b = _identify(trunc_idx, _clean(r["馬名"]), _clean(r["種牡馬"]), by, D)
        if a != b:
            mism.append((r["馬名"], a, b))
    assert mism == []


def test_T5_duplicate_order_key_is_rejected():
    hist = _history()
    shf.validate_history(hist)                     # 重複なしは通る
    dup = pd.concat([hist, hist.iloc[[0]]], ignore_index=True)
    with pytest.raises(shf.HistorySchemaError):
        shf.validate_history(dup)


@pytest.mark.parametrize("drop", ["post_time", "umaban"])
def test_T6_missing_order_columns_stop_with_error(tmp_path, drop):
    hist = _history().drop(columns=[drop])
    base = _base(tmp_path, hist, f"no_{drop}")
    with pytest.raises(shf.HistorySchemaError):
        shf.fill_history_features(_weekly(), base=base)


def test_T6_missing_post_time_value_stops_with_error(tmp_path):
    hist = _history()
    hist.loc[3, "post_time"] = np.nan
    base = _base(tmp_path, hist, "nan_post_time")
    with pytest.raises(shf.HistorySchemaError):
        shf.fill_history_features(_weekly(), base=base)
