# -*- coding: utf-8 -*-
"""
eligible_races.py
==================
EXP10 Stage 1. 対象馬(◎)選択と、DNF・除外・取消の母集団分離。

対象・除外規則(ユーザー指定、spec.jsonに凍結):
  1. 評価単位 = 1レースにつき、事前時点のraw v6 scoreが最大の馬を1頭。
  2. 較正後確率・馬番順は一切使わない(選択は生スコアのみ)。
  3. 同点(浮動小数点タイ)は、race_id + 血統登録番号(結果非依存の馬個体キー)の
     決定論的ハッシュで固定順に解決する(結果・確定オッズは一切使わない)。
  4. 取消・除外・中止でmaster_v2に行自体が存在しない(=v6スコアが計算されない)馬は
     主解析の対象馬候補から自動的に除外される(母集団がmaster_v2の生存者に限定される
     ため、構造的に除外)。これらの頭数はDATA_AUDIT.md §1で別途カウント済み。
  5. 頭数3未満のレースは除外(EXP09のeligible_races.pyと同じ基準を踏襲、単勝分布が
     退化するのを避けるため)。

母集団: `backtest_pl_ev.score_test(include_valid=True)`が返す行(model=unified_rank_v6、
2022年末までの学習で固定、split="valid"(2023)/"test"(2024-2025)のraw scoreのみ使用、
較正器は一切使わない)。
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"E:\PyCaLiAI")
sys.path.insert(0, str(BASE))

import backtest_pl_ev as be  # noqa: E402

be.MODEL_PKL = BASE / "models/unified_rank_v6.pkl"
# CAL_PKL/CURVE_PKLはscore_test()が使わない(rawスコアのみ計算)ため変更不要。

from backtest_pl_ev import COL_RID, COL_BAN, COL_JYUN  # noqa: E402

MIN_FIELD_SIZE = 3
TIE_BREAK_SEED = 20261010  # EXP09のTIE_BREAK_SEEDとは独立の固定値
COL_KETTO = "血統登録番号"


def _tie_break_key(race_id: str, ketto: str) -> int:
    """race_id + 血統登録番号(結果・オッズ非依存)の決定論的ハッシュ。値が小さい方を選ぶ。"""
    h = hashlib.blake2b(f"{TIE_BREAK_SEED}:{race_id}:{ketto}".encode("utf-8"), digest_size=8)
    return int.from_bytes(h.digest(), "big")


def score_population(years: list[int]) -> pd.DataFrame:
    """unified_rank_v6のraw scoreをvalid(2023)/test(2024-2025)について取得し、
    指定年でフィルタする。較正器は一切ロードしない(生スコアのみ)。"""
    te = be.score_test(include_valid=True)
    te = te[te["year"].isin(years)].copy()
    return te


def build_target_rows(years: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """指定年について、レースごとの対象馬(◎)を確定する。

    戻り値:
      targets: 1レース1行。rid, year, n, focal_ban, focal_ketto, focal_score,
               tie_break_used, n_tied
      race_horses: 全馬全行(対象馬選択の再現・順位分布計算用)。
               rid, year, n, 馬番, 血統登録番号, score, is_focal, 着順(数値)
    """
    te = score_population(years)
    target_rows = []
    horse_rows = []
    n_races_total = 0
    n_excluded_small_field = 0
    tie_break_count = 0

    for rid, g in te.groupby(COL_RID, sort=False):
        n_races_total += 1
        n = len(g)
        if n < MIN_FIELD_SIZE:
            n_excluded_small_field += 1
            continue
        g = g.reset_index(drop=True)
        scores = g["_score"].values.astype(float)
        max_score = scores.max()
        tied_mask = np.isclose(scores, max_score, rtol=0, atol=1e-9)
        n_tied = int(tied_mask.sum())
        tie_break_used = n_tied > 1
        if tie_break_used:
            tie_break_count += 1
            cand_idx = np.where(tied_mask)[0]
            keys = [
                (_tie_break_key(str(rid), str(g.loc[i, COL_KETTO])), i) for i in cand_idx
            ]
            keys.sort()
            focal_pos = keys[0][1]
        else:
            focal_pos = int(np.argmax(scores))

        year = int(g["year"].iloc[0])
        finish = pd.to_numeric(g[COL_JYUN], errors="coerce").values

        target_rows.append(dict(
            rid=str(rid), year=year, n=n,
            focal_ban=int(g.loc[focal_pos, COL_BAN]),
            focal_ketto=str(g.loc[focal_pos, COL_KETTO]),
            focal_score=float(scores[focal_pos]),
            focal_finish=(float(finish[focal_pos]) if not np.isnan(finish[focal_pos]) else np.nan),
            tie_break_used=tie_break_used, n_tied=n_tied,
        ))
        for pos in range(n):
            horse_rows.append(dict(
                rid=str(rid), year=year, n=n, pos_in_group=pos,
                ban=int(g.loc[pos, COL_BAN]), ketto=str(g.loc[pos, COL_KETTO]),
                score=float(scores[pos]), is_focal=(pos == focal_pos),
                finish=(float(finish[pos]) if not np.isnan(finish[pos]) else np.nan),
            ))

    targets = pd.DataFrame(target_rows)
    race_horses = pd.DataFrame(horse_rows)
    meta = dict(
        n_races_total_in_population=n_races_total,
        n_excluded_small_field=n_excluded_small_field,
        n_eligible_races=len(targets),
        n_tie_break_invoked=tie_break_count,
    )
    return targets, race_horses, meta


if __name__ == "__main__":
    # backtest_pl_ev import時点で既にsys.stdoutがutf-8 TextIOWrapperへ差し替え済み。
    targets, race_horses, meta = build_target_rows([2023])
    print("[2023 development population, 主解析のみ・構造的件数]")
    for k, v in meta.items():
        print(f"  {k}: {v:,}")
    print(f"  focal_finish が NaN(対象馬自身が数値着順を持たない、通常は起こらないはずの検査): "
          f"{targets['focal_finish'].isna().sum()}")
    print(targets.describe(include="all").to_string())
