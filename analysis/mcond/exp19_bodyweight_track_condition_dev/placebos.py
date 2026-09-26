# -*- coding: utf-8 -*-
"""
placebos.py — EXP19 SPEC §8 の placebo 構造 (Stage 0 では構造と境界だけを実装・検査し、実データで回さない)
==========================================================================================================
入力は power_data の race 配列 (off, W, WP, 属性)。
  P1_IDENTITY_WITHIN_RACE  同一 race 内で W block の行を馬 identity 間で置換 (値 multiset・市場・馬場・頭数を保持)
  P2_TIME_SHIFT            同競馬場×芝ダ×年齢構成帯×頭数帯の別 race (自身除外) から W block を借りる。
                           donor は受け手以上の頭数を持つ race に限り、donor の馬番順の先頭 n 行を受け手の馬番順に当てる。
                           該当 donor が無い race は自身を保持し件数を報告
  P3_TRACK_SHIFT           WP だけ: 同競馬場×芝ダ×月の**別開催回** (年・回が異なる) の venue-day から P を借りて WP を作り直す。
                           同一開催回の日は借りない。W・市場・結果は保持
  P4_MISSINGNESS_ONLY      W の値を FILL に置き換え、status・欠損指示子だけを残す
判定式は gate_grade.placebo_exceeded (real < quantile(placebo, 0.025))。
"""
from __future__ import annotations

import numpy as np

from . import features as F


def p1_permute(W: np.ndarray, off: np.ndarray, rng) -> np.ndarray:
    race = np.repeat(np.arange(len(off) - 1), np.diff(off))
    order = np.lexsort((rng.random(len(W)), race))
    return W[order]


def cells(attrs: dict, keys) -> np.ndarray:
    return np.array(["|".join(str(attrs[k][i]) for k in keys) for i in range(len(attrs[keys[0]]))])


def p2_time_shift(W: np.ndarray, off: np.ndarray, attrs: dict, rng):
    """戻り: 置換後 W、自身保持の race 数"""
    cell = cells(attrs, ["venue", "surface", "age_band", "field_band"])
    n = np.diff(off)
    members = {}
    for r, c in enumerate(cell):
        members.setdefault(c, []).append(r)
    out = W.copy()
    kept_self = 0
    for r in range(len(n)):
        cand = [d for d in members[cell[r]] if d != r and n[d] >= n[r]]
        if not cand:
            kept_self += 1
            continue
        d = cand[int(rng.integers(len(cand)))]
        out[off[r]:off[r + 1]] = W[off[d]:off[d] + n[r]]
    return out, kept_self


def kaisai_round(kaisai: str) -> str:
    """torch `開催` (例 '5中8' = 5 回中山 8 日目) → 回"""
    s = str(kaisai)
    digits = ""
    for ch in s:
        if ch.isdigit():
            digits += ch
        else:
            break
    return digits


def p3_track_shift(p_race: dict, attrs: dict, rng):
    """race ごとの P (cushion_z, moist_gp_z, moist_gradient_z, track_extreme_z) を別開催回の race から借りる。
    donor = 同 venue × surface × 月、(年, 回) が異なり、P が利用可能な race。戻り: 置換後 P、自身保持数"""
    R = len(attrs["venue"])
    rnd = np.array([f"{attrs['year'][i]}-{kaisai_round(attrs['kaisai'][i])}" for i in range(R)])
    cell = cells(attrs, ["venue", "surface", "month"])
    ok = np.asarray(attrs["wp_ok"], bool)
    members = {}
    for r in range(R):
        if ok[r]:
            members.setdefault(cell[r], []).append(r)
    out = {k: v.copy() for k, v in p_race.items()}
    kept_self = 0
    donor = np.arange(R)
    for r in range(R):
        cand = [d for d in members.get(cell[r], []) if rnd[d] != rnd[r]]
        if not cand:
            kept_self += 1
            continue
        d = cand[int(rng.integers(len(cand)))]
        donor[r] = d
        for k in out:
            out[k][r] = p_race[k][d]
    return out, kept_self, donor, rnd


def p4_missingness_only(Wdesign, w_cols) -> np.ndarray:
    """値列を FILL に置き換え、status (bw_status_not_measured) と欠損指示子 (miss_*) だけを残す"""
    out = np.array(Wdesign, float, copy=True)
    for j, c in enumerate(w_cols):
        if c == "bw_status_not_measured" or c.startswith("miss_"):
            continue
        out[:, j] = F.FILL
    return out
