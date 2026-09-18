# -*- coding: utf-8 -*-
"""
environments.py — E1(年度) / E2(競馬場) / E3(芝ダ×距離帯) の環境ラベル生成
=================================================================================
最小環境サイズ規則 (評価前に固定): TRAIN期間 (2016-2021) のレース数が MIN_RACES 未満の
水準は、そのレース数が最小の水準から順に、軸ごとに1つ設けた固定の "other" 区分へ統合する。
統合の要否・対象はTRAIN期間の実測値だけで決め、confirm/exploratory期間の分布は見ない。
E3の距離帯は EXP02 (`analysis.mcond.exp02_dynamic_skill_dev.dyn_skill.dist_band`) の
既存区分 (≤1400/1401-1800/1801-2200/≥2201) をそのまま流用する (仕様「既存仕様があれば使う」)。
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
from analysis.mcond.exp02_dynamic_skill_dev.dyn_skill import dist_band  # noqa: E402

MIN_RACES = 1000
PLACE = {"01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京", "06": "中山",
         "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"}
DBAND_NAME = {0: "d_le1400", 1: "d_1401_1800", 2: "d_1801_2200", 3: "d_ge2201"}


def build(rid16: pd.Series, surface: pd.Series, distance: pd.Series, year: pd.Series,
         train_mask) -> tuple[pd.DataFrame, list]:
    """rid16, surface('芝'/'ダ'等), distance(数値), year, train_mask(bool array)
    を受け取り (e1/e2/e3(統合後)の列を持つDataFrame, 統合情報リスト) を返す。
    ★DataFrame.attrs は merge 等の操作で失われることがある (project_leak_guard の既知の穴)
      ため、統合情報は attrs ではなく戻り値のタプルで別途渡す。"""
    e1 = year.astype(int)
    e2 = rid16.astype(str).str[8:10].map(PLACE).fillna("他")
    db = distance.map(dist_band)
    surf01 = (surface.astype(str) == "芝").map({True: "芝", False: "ダ"})
    e3 = surf01 + "_" + db.map(DBAND_NAME)

    def merge_small(s: pd.Series, axis_name: str) -> tuple[pd.Series, dict]:
        cnt = s[train_mask].value_counts()
        small = set(cnt[cnt < MIN_RACES].index)
        info = {"axis": axis_name, "levels_total": int(cnt.shape[0]),
                "levels_merged": len(small), "merged_levels": sorted(small),
                "counts_train": cnt.to_dict()}
        out = s.where(~s.isin(small), other=f"{axis_name}_other")
        return out, info

    e1m, i1 = merge_small(e1.astype(str), "e1")
    e2m, i2 = merge_small(e2, "e2")
    e3m, i3 = merge_small(e3, "e3")
    df = pd.DataFrame({"e1_year": e1m, "e2_course": e2m, "e3_surfdist": e3m})
    return df, [i1, i2, i3]
