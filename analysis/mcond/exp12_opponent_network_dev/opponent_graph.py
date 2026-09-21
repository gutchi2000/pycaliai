# -*- coding: utf-8 -*-
"""
opponent_graph.py
===================
EXP12 Stage1. O3: 時点安全な対戦相手ネットワーク特徴。

設計（ユーザー指定、spec.jsonに凍結予定）:
  - node key = 血統登録番号(hid)を**文字列のまま**使用。数値変換禁止
    （先頭ゼロ消失防止。現行データに先頭ゼロは無いことを実測済みだが、
    将来のデータでも安全なようコード上は一切int変換しない）。
  - グラフは対象日tの**日初時点固定**。同日内の先行レース結果も使わない。
  - edgeは対象日より前のJRAレースだけ（地方・海外はmaster_v2に構造的に
    不在=自動的に除外される）。
  - target raceの出走馬同士の当日edgeは追加しない（設計上、当日のレースは
    「まだ対象馬が参加していない未来のレース」として扱われるため、日初
    バッチ更新の前に特徴量を計算する=当日のレースは自動的にグラフへ
    含まれない）。
  - 対戦相手の能力は「対象日tの前までに判明した最新値」（day-batched
    current_ability辞書、O(1)ルックアップ）。対戦相手の対象日"後"の成績は
    一切使わない。
  - external_history_gap: 前走日付(TARGET記録)と、master_v2上でのこの馬の
    直前行の日付が一致しない場合に1を立てる（EXP01が確立した検証パターンを
    踏襲、地方・海外レースを挟んだ場合に発生）。

対戦相手の基礎能力（3方式、選択規則は事前固定）:
  - O3b（主方式・固定）: EXP02動的能力(dyn_skill_mu、
    data/_research/mcond/exp02_features.parquet)
  - O3a（感度分析）: 既存ELO(elo_T1M1_horse、data/elo_feats.parquet)
  - O3c（感度分析）: v6 as-of能力proxy(model.predict()の生スコア、
    train期間はin-sample)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"E:\PyCaLiAI")
MASTER_CSV = BASE / "data/master_v2_20130105-20251228.csv"
EXP02_PARQUET = BASE / "data/_research/mcond/exp02_features.parquet"
ELO_PARQUET = BASE / "data/elo_feats.parquet"

COL_RID = "レースID(新/馬番無)"
COL_BAN = "馬番"
COL_HID = "血統登録番号"
COL_DATE = "日付"
COL_PREV_DATE = "前走日付"
COL_FINISH = "着順"

LATER_PROVED_STRONG_THRESHOLD = 3.0  # dyn_skill_mu尺度(初期値25、sigma≈8.33)上の閾値、設計値として明記
RECENCY_HALFLIFE_RACES = 5.0  # 直近Nレース分で重みが半減する指数減衰(直近優先)


def load_master_base() -> pd.DataFrame:
    """hidは文字列のまま保持する(数値変換禁止)。"""
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig",
                      usecols=[COL_RID, COL_BAN, COL_HID, COL_DATE, COL_PREV_DATE, COL_FINISH],
                      dtype={COL_HID: str, COL_RID: str}, low_memory=False)
    df[COL_FINISH] = pd.to_numeric(df[COL_FINISH], errors="coerce")
    df = df.dropna(subset=[COL_FINISH]).copy()
    df["date_int"] = df[COL_DATE].astype(int)  # YYYYMMDD、日付比較専用(ソート・比較のみ、hidには使わない)
    return df


def add_external_history_gap(df: pd.DataFrame) -> pd.DataFrame:
    """前走日付(TARGET記録)と、master_v2上でのこの馬の直前行の日付が
    一致しなければexternal_history_gap=1(EXP01established pattern踏襲)。"""
    df = df.sort_values([COL_HID, "date_int"], kind="mergesort").copy()
    df["_master_prev_date_int"] = df.groupby(COL_HID)["date_int"].shift(1)
    # 前走日付はYYMMDD(6桁, 20XX年のXX)形式、date_intはYYYYMMDD(8桁)なので
    # 下6桁同士を比較する(世紀またぎは対象期間2013-2025では発生しない)。
    prev_date_6digit = pd.to_numeric(df[COL_PREV_DATE], errors="coerce")
    master_prev_6digit = (df["_master_prev_date_int"] % 1000000)
    df["external_history_gap"] = (
        prev_date_6digit.notna()
        & df["_master_prev_date_int"].notna()
        & (prev_date_6digit.astype("Int64") != master_prev_6digit.astype("Int64"))
    ).astype(int)
    # 前走日付が存在するのにmasterに直前行が無い(デビュー戦扱いになってしまっている)場合も検出
    df.loc[prev_date_6digit.notna() & df["_master_prev_date_int"].isna(), "external_history_gap"] = 1
    return df.drop(columns=["_master_prev_date_int"])


def load_ability_source(mode: str) -> pd.DataFrame:
    """[hid, rid16, ban, date_int, ability] を返す。hidは文字列のまま。"""
    if mode == "o3b_exp02":
        df = pd.read_parquet(EXP02_PARQUET, columns=["hid", "rid16", "ban", "date", "dyn_skill_mu"])
        df = df.rename(columns={"rid16": COL_RID, "dyn_skill_mu": "ability"})
        df["date_int"] = df["date"].dt.strftime("%Y%m%d").astype(int)
        return df[["hid", COL_RID, "ban", "date_int", "ability"]]
    if mode == "o3a_elo":
        elo = pd.read_parquet(ELO_PARQUET, columns=["rid16", "ban", "elo_T1M1_horse"])
        elo = elo.rename(columns={"rid16": COL_RID, "elo_T1M1_horse": "ability"})
        base = load_master_base()[[COL_RID, COL_BAN, COL_HID, "date_int"]]
        merged = base.merge(elo, left_on=[COL_RID, COL_BAN], right_on=[COL_RID, "ban"], how="inner")
        return merged.rename(columns={COL_HID: "hid"})[["hid", COL_RID, "ban", "date_int", "ability"]]
    if mode == "o3c_v6":
        return _load_v6_asof_ability()
    raise ValueError(f"unknown mode: {mode}")


def _load_v6_asof_ability() -> pd.DataFrame:
    sys.path.insert(0, str(BASE))
    import joblib
    bundle = joblib.load(BASE / "models/unified_rank_v6.pkl")
    model, feats, encs = bundle["model"], bundle["feature_cols"], bundle["encoders"]
    rr_mode = bundle.get("race_relative_mode")
    ca_mode = bundle.get("course_affinity_mode")
    gf_mode = bundle.get("grade_feats_mode")
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", dtype={COL_HID: str, COL_RID: str}, low_memory=False)
    df[COL_FINISH] = pd.to_numeric(df[COL_FINISH], errors="coerce")
    df = df.dropna(subset=[COL_FINISH, COL_RID, "split"]).copy()
    if rr_mode:
        from race_relative_feats import add_race_relative_feats
        df = add_race_relative_feats(df, mode=rr_mode)
    if ca_mode:
        from course_affinity_feats import add_course_affinity_feats
        df = add_course_affinity_feats(df, mode=ca_mode)
    if gf_mode:
        from grade_feats import add_grade_feats
        df = add_grade_feats(df, mode=gf_mode)
    from backtest_pl_ev import apply_encoders
    df = apply_encoders(df, encs)
    X = df[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    ability = model.predict(X)
    out = pd.DataFrame({
        "hid": df[COL_HID].values, COL_RID: df[COL_RID].values, "ban": df[COL_BAN].values,
        "date_int": df[COL_DATE].astype(int).values, "ability": ability,
    })
    return out


def build_opponent_features(
    mode: str = "o3b_exp02", output_years: list[int] | None = None,
    max_build_year: int | None = None, collect_raw: bool = False,
):
    """day-batched as-ofアルゴリズムで10特徴量+external_history_gapを計算する。

    重要: グラフ構築(current_ability/opponent_historyの更新)は
    `max_build_year`まで(またはoutput_yearsの範囲)の**全履歴**を使う。
    `output_years`は特徴量を**出力する行**の絞り込みであり、グラフ構築
    自体は絞り込まない(絞り込むと対象年より前の対戦相手履歴が消えて
    しまい、意味のある特徴が作れなくなるため)。

    アルゴリズム:
      1. 日付昇順で日ごとにブロック化。
      2. 各日Dの処理前、current_ability[hid]・opponent_history[hid]は
         Dより前の情報のみを反映している(日初時点固定)。
      3. Dの各行について、opponent_history[hid]（過去の一意対戦相手集合、
         各相手の遭遇時能力・勝敗）とcurrent_ability(対戦相手の現在値)から
         10特徴量を計算する(output_yearsに該当する行のみ結果に残す)。
      4. Dの全行を処理し終えてから、Dのレース結果をopponent_history・
         current_abilityへ反映する(次の日以降に使われる)。
    """
    base = load_master_base()
    base = add_external_history_gap(base)
    if max_build_year is not None:
        base = base[base["date_int"] // 10000 <= max_build_year]
    output_year_set = set(output_years) if output_years is not None else None
    ability = load_ability_source(mode)
    return run_day_batched_algorithm(base, ability, output_year_set, collect_raw=collect_raw)


def run_day_batched_algorithm(
    base: pd.DataFrame, ability: pd.DataFrame, output_year_set: set[int] | None,
    collect_raw: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """collect_raw=Trueの場合、(df, raw_opponents)を返す。raw_opponentsは
    {(rid16,hid): [(opp_hid, ability_at_encounter, current_ability, beaten, lost, last_date_int), ...]}
    で、placebo検定(stratumプール構築)に使う。"""
    """コア・アルゴリズム本体。base/abilityを直接渡せるようにし、合成テスト・
    削除不変性テストで再利用しやすくする。

    base: [レースID(新/馬番無), 馬番, 血統登録番号, date_int, 着順,
           external_history_gap] を持つDataFrame(load_master_base()+
           add_external_history_gap()の出力と同じスキーマ)。
    ability: [hid, レースID(新/馬番無), ban, date_int, ability]。
    """
    ability_by_hid_rid = {(h, r): a for h, r, a in zip(ability["hid"], ability[COL_RID], ability["ability"])}

    base = base.sort_values("date_int", kind="mergesort").reset_index(drop=True)
    race_groups = base.groupby(COL_RID, sort=False)
    race_members: dict[str, list[tuple]] = {}
    for rid, g in race_groups:
        race_members[rid] = list(zip(g[COL_HID], g[COL_BAN], g[COL_FINISH]))

    current_ability: dict[str, float] = {}
    # opponent_history[hid] = {opp_hid: [ability_at_encounter, beaten_flag, lost_flag, last_date_int]}
    opponent_history: dict[str, dict[str, list]] = {}

    out_rows = []
    raw_opponents: dict[tuple, list] = {}
    dates = base["date_int"].values
    n = len(base)
    pos = 0
    while pos < n:
        d = dates[pos]
        end = pos
        while end < n and dates[end] == d:
            end += 1

        day_slice = base.iloc[pos:end]
        need_output = output_year_set is None or (int(d) // 10000) in output_year_set
        if need_output:
            for idx, row in day_slice.iterrows():
                hid = row[COL_HID]
                hist = opponent_history.get(hid, {})
                feats = _compute_features(hist, current_ability)
                feats["rid16"] = row[COL_RID]
                feats["hid"] = hid
                feats["ban"] = row[COL_BAN]
                feats["external_history_gap"] = row["external_history_gap"]
                out_rows.append(feats)
                if collect_raw:
                    raw_list = [
                        (o, v[0], current_ability.get(o, np.nan), v[1], v[2], v[3])
                        for o, v in hist.items()
                    ]
                    raw_opponents[(row[COL_RID], hid)] = raw_list

        # このDの結果をグラフへ反映(次の日以降に使う)
        rids_today = day_slice[COL_RID].unique()
        for rid in rids_today:
            members = race_members[rid]
            if len(members) < 2:
                continue
            for i in range(len(members)):
                hid_i, ban_i, fin_i = members[i]
                ability_i = ability_by_hid_rid.get((hid_i, rid))
                if ability_i is not None:
                    current_ability[hid_i] = ability_i
                for j in range(len(members)):
                    if i == j:
                        continue
                    hid_j, ban_j, fin_j = members[j]
                    ability_j_at_encounter = ability_by_hid_rid.get((hid_j, rid))
                    if ability_j_at_encounter is None:
                        continue
                    h = opponent_history.setdefault(hid_i, {})
                    entry = h.get(hid_j)
                    beat = 1 if fin_i < fin_j else 0
                    lost = 1 if fin_i > fin_j else 0
                    if entry is None:
                        h[hid_j] = [ability_j_at_encounter, beat, lost, d]
                    else:
                        entry[1] = max(entry[1], beat)
                        entry[2] = max(entry[2], lost)
                        entry[0] = ability_j_at_encounter  # 最新の遭遇時能力で更新
                        entry[3] = d
        pos = end

    result = pd.DataFrame(out_rows)
    if collect_raw:
        return result, raw_opponents
    return result


def _compute_features(hist: dict[str, list], current_ability: dict[str, float]) -> dict:
    if not hist:
        return dict(
            unique_opponent_count=0, opponent_current_strength_mean=np.nan,
            opponent_current_strength_max=np.nan, opponent_current_strength_top3_mean=np.nan,
            beaten_opponent_strength_mean=np.nan, lost_to_opponent_strength_mean=np.nan,
            strongest_beaten_opponent=np.nan, recency_weighted_opponent_strength=np.nan,
            opponent_strength_dispersion=np.nan, opponent_later_proved_strong_count=0,
        )
    opp_ids = list(hist.keys())
    n = len(opp_ids)
    cur = np.array([current_ability.get(o, np.nan) for o in opp_ids], dtype=float)
    at_enc = np.array([hist[o][0] for o in opp_ids], dtype=float)
    beaten = np.array([hist[o][1] for o in opp_ids], dtype=bool)
    lost = np.array([hist[o][2] for o in opp_ids], dtype=bool)
    last_date = np.array([hist[o][3] for o in opp_ids], dtype=float)

    valid = ~np.isnan(cur)
    cur_v = cur[valid]
    unique_opponent_count = n

    if len(cur_v) == 0:
        strength_mean = strength_max = strength_top3 = dispersion = np.nan
    else:
        strength_mean = float(np.mean(cur_v))
        strength_max = float(np.max(cur_v))
        top3 = np.sort(cur_v)[::-1][:3]
        strength_top3 = float(np.mean(top3))
        dispersion = float(np.std(cur_v)) if len(cur_v) > 1 else 0.0

    beaten_v = cur[valid & beaten]
    lost_v = cur[valid & lost]
    beaten_mean = float(np.mean(beaten_v)) if len(beaten_v) > 0 else np.nan
    lost_mean = float(np.mean(lost_v)) if len(lost_v) > 0 else np.nan
    strongest_beaten = float(np.max(beaten_v)) if len(beaten_v) > 0 else np.nan

    # 直近優先の指数減衰重み: 対戦相手ごとの「最後に対戦した日」の新しさで重み付け
    if len(cur_v) > 0:
        recency_rank = pd.Series(last_date[valid]).rank(method="average").values  # 古い順に1..k
        k = len(recency_rank)
        weights = 0.5 ** ((k - recency_rank) / RECENCY_HALFLIFE_RACES)
        recency_weighted = float(np.average(cur_v, weights=weights))
    else:
        recency_weighted = np.nan

    proved_strong = int(np.sum(valid & (cur - at_enc > LATER_PROVED_STRONG_THRESHOLD)))

    return dict(
        unique_opponent_count=unique_opponent_count,
        opponent_current_strength_mean=strength_mean,
        opponent_current_strength_max=strength_max,
        opponent_current_strength_top3_mean=strength_top3,
        beaten_opponent_strength_mean=beaten_mean,
        lost_to_opponent_strength_mean=lost_mean,
        strongest_beaten_opponent=strongest_beaten,
        recency_weighted_opponent_strength=recency_weighted,
        opponent_strength_dispersion=dispersion,
        opponent_later_proved_strong_count=proved_strong,
    )


if __name__ == "__main__":
    import time
    t0 = time.time()
    feats = build_opponent_features(mode="o3b_exp02", output_years=[2023], max_build_year=2023)
    print(f"[opponent_graph] 2023年出力(2013-2023の全履歴でグラフ構築): "
          f"{len(feats):,}行、{time.time()-t0:.1f}秒")
    print(feats.describe().to_string())
    out_dir = Path(__file__).parent / "out"
    out_dir.mkdir(exist_ok=True)
    feats.to_parquet(out_dir / "o3_features_2023_o3b.parquet")
    print(f"[saved] {out_dir / 'o3_features_2023_o3b.parquet'}")
