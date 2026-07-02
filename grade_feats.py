"""
grade_feats.py — 格文脈特徴 v1 (unified_rank_v11 用)
====================================================
「AI印は過去走の格・対戦相手の質を見れない」構造盲点 (エコロアルバ事件) への配線。
出自: analysis/_tmp_grade_feats_gate.py の格13特徴 (ΔAUC(複勝)+0.0130 実証済み) を
serve 可能な形に再定義したもの。

設計原則 (serve skew 再発防止):
  - 特徴は「直近5走窓」に限定する。serve 側の供給源 data/kako5/*.csv が過去5走
    しか持たないため、学習だけキャリア全体で定義すると hist_ 系と同じ serve 死を再発する。
  - クラス序数化は本モジュールの2関数に一本化:
      class_name_to_ord() … 学習側 (master の クラス名 列)
      race_name_to_ord()  … serve 側 (kako5/weekly のレース名文字列)
  - 検証系 (backtest_pl_ev.score_test / build_pl_calibrators.load_valid) は bundle の
    grade_feats_mode フラグ経由で add_grade_feats() を通る (race_relative_mode と同じ流儀)。

leak 規約: 全特徴は sort(血統登録番号, date_dt, rid; mergesort) + groupby shift/rolling で
現走行を含まない。今走クラス (gf_cur_crank) のみ出走表時点で既知の今走情報。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PED = "血統登録番号"
RID = "レースID(新/馬番無)"
POS = "着順"

# 学習側: master の クラス名 → 序数 (高いほど格上)。
# analysis/_tmp_grade_feats_gate.py の CLASS_RANK と同一 (オープン/OP 表記揺れのみ追加)。
CLASS_RANK = {
    "新馬": 0, "未勝利": 1,
    "500万": 2, "1勝": 2,
    "1000万": 3, "2勝": 3,
    "1600万": 4, "3勝": 4,
    "ｵｰﾌﾟﾝ": 5, "オープン": 5, "OP": 5,
    "OP(L)": 6,
    "Ｇ３": 7, "重賞": 7, "Ｇ２": 8, "Ｇ１": 9,
}
# 障害 (マップ外 → NaN のまま)
JUMP = {"ＪＧ１", "ＪＧ２", "ＪＧ３"}

GRADE_FEATS_V1 = [
    "gf_cur_crank",
    "gf_prev_crank", "gf_prev2_crank", "gf_prev3_crank",
    "gf_prev2_pos", "gf_prev3_pos",
    "gf_max_class5", "gf_maxclass5_vs_cur",
    "gf_drop_vs_prev", "gf_drop_vs_prev2",
    "gf_classdrop_flag",
    "gf_n_graded5", "gf_graded_top3_5",
    "gf_wgt_x_classdrop", "gf_wgt_x_classup",
]


def class_name_to_ord(name) -> float:
    """master の クラス名 → 序数。障害/未知は NaN。"""
    return CLASS_RANK.get(str(name).strip(), np.nan)


def race_name_to_ord(name) -> float:
    """serve 側: kako5/weekly のレース名文字列 → 序数。

    TARGET のレース名はクラスサフィックス付き (実測 2026-07-03):
      未勝利 / 新馬・牝 / １勝ｸﾗｽ* / 唐戸特別･1勝 / 青梅特Ｈ･2勝 /
      京都金杯ＨG3 / 有馬記念G1 / 京葉Ｓ(L) / 障害未勝利 / 加納宿(地方=NaN)
    """
    if name is None:
        return np.nan
    s = str(name).strip().replace("*", "")
    if not s:
        return np.nan
    if "障" in s or "ＪＧ" in s or "JG" in s:
        return np.nan
    for pat, v in (("G1", 9), ("Ｇ１", 9), ("G2", 8), ("Ｇ２", 8),
                   ("G3", 7), ("Ｇ３", 7)):
        if pat in s:
            return float(v)
    if "(L)" in s or "（L）" in s:
        return 6.0
    if "新馬" in s:
        return 0.0
    if "未勝利" in s:
        return 1.0
    for pat, v in (("1勝", 2), ("１勝", 2), ("500万", 2),
                   ("2勝", 3), ("２勝", 3), ("1000万", 3),
                   ("3勝", 4), ("３勝", 4), ("1600万", 4)):
        if pat in s:
            return float(v)
    if "ｵｰﾌﾟﾝ" in s or "オープン" in s:
        return 5.0
    return np.nan  # 地方交流・マーカー無し OP 特別 等


def _roll5(series: pd.Series, key: np.ndarray, how: str) -> pd.Series:
    """グループ内 shift(1)+rolling(5) 集約 (現走除外の直近5走窓)。"""
    g = series.groupby(key)
    if how == "max":
        return g.transform(lambda x: x.shift(1).rolling(5, min_periods=1).max())
    if how == "sum":
        return g.transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum())
    raise ValueError(how)


def add_grade_feats(df: pd.DataFrame, mode: str = "v1") -> pd.DataFrame:
    """master 縦持ち df に格文脈15特徴を追加して返す (leak-safe as-of)。

    必要列: 血統登録番号, date_dt, レースID(新/馬番無), クラス名, 着順, 斤量, 前走斤量
    返り値 df.attrs["grade_leak_violations"] にグループ内日付逆行の件数を記録。
    """
    if mode != "v1":
        raise ValueError(f"unknown grade_feats mode: {mode}")

    s = df[[PED, "date_dt", RID, "クラス名", POS, "斤量", "前走斤量"]].copy()
    s["_dd"] = pd.to_datetime(s["date_dt"], errors="coerce")
    s["_rid_s"] = s[RID].astype(str)
    s = s.sort_values([PED, "_dd", "_rid_s"], kind="mergesort")
    key = s[PED].values

    crank = s["クラス名"].astype(str).str.strip().map(CLASS_RANK)
    pos = pd.to_numeric(s[POS], errors="coerce")

    gc, gp = crank.groupby(key), pos.groupby(key)
    prev_c, prev2_c, prev3_c = gc.shift(1), gc.shift(2), gc.shift(3)
    prev_p, prev2_p, prev3_p = gp.shift(1), gp.shift(2), gp.shift(3)

    max5 = _roll5(crank, key, "max")
    graded = (crank >= 7).astype(float)
    graded_top3 = ((crank >= 7) & (pos <= 3)).astype(float)
    n_graded5 = _roll5(graded, key, "sum")
    graded_top3_5 = _roll5(graded_top3, key, "sum")

    wchg = (pd.to_numeric(s["斤量"], errors="coerce")
            - pd.to_numeric(s["前走斤量"], errors="coerce"))
    drop1, drop2 = prev_c - crank, prev2_c - crank

    feats = pd.DataFrame({
        "gf_cur_crank": crank,
        "gf_prev_crank": prev_c,
        "gf_prev2_crank": prev2_c,
        "gf_prev3_crank": prev3_c,
        "gf_prev2_pos": prev2_p,
        "gf_prev3_pos": prev3_p,
        "gf_max_class5": max5,
        "gf_maxclass5_vs_cur": max5 - crank,
        "gf_drop_vs_prev": drop1,
        "gf_drop_vs_prev2": drop2,
        "gf_classdrop_flag": (((prev_c > crank) & (prev_p >= 6))
                              | ((prev2_c > crank) & (prev2_p >= 6))).astype(float),
        "gf_n_graded5": n_graded5,
        "gf_graded_top3_5": graded_top3_5,
        "gf_wgt_x_classdrop": wchg * drop1,
        "gf_wgt_x_classup": wchg * (crank > prev_c).astype(float),
    }, index=s.index)

    viol = int((s["_dd"].groupby(key).diff().dt.total_seconds() < 0).sum())

    out = pd.concat([df, feats.reindex(df.index)], axis=1)
    out.attrs["grade_leak_violations"] = viol
    return out


if __name__ == "__main__":
    # 簡易セルフテスト: 1頭の合成履歴で shift/rolling の as-of 性を確認
    toy = pd.DataFrame({
        PED: ["A"] * 4,
        "date_dt": ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"],
        RID: ["1", "2", "3", "4"],
        "クラス名": ["新馬", "未勝利", "Ｇ３", "1勝"],
        POS: [5, 1, 8, 2],
        "斤量": [54, 54, 56, 53],
        "前走斤量": [np.nan, 54, 54, 56],
    })
    r = add_grade_feats(toy)
    row = r.iloc[3]  # 4走目: G3で8着凡走→1勝クラスへ格下げ、斤量-3
    assert row["gf_prev_crank"] == 7 and row["gf_cur_crank"] == 2
    assert row["gf_classdrop_flag"] == 1.0, "格上凡走→格下げフラグ"
    assert row["gf_max_class5"] == 7 and row["gf_n_graded5"] == 1
    assert row["gf_graded_top3_5"] == 0
    assert row["gf_wgt_x_classdrop"] == (53 - 56) * (7 - 2)
    assert r.attrs["grade_leak_violations"] == 0
    assert race_name_to_ord("唐戸特別･1勝") == 2
    assert race_name_to_ord("京都金杯ＨG3") == 7
    assert race_name_to_ord("有馬記念G1") == 9
    assert race_name_to_ord("京葉Ｓ(L)") == 6
    assert race_name_to_ord("未勝利・牝*") == 1
    assert np.isnan(race_name_to_ord("加納宿"))
    assert np.isnan(race_name_to_ord("障害未勝利"))
    print("self-test OK")
