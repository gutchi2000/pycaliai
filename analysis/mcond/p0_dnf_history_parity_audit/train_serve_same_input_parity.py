# -*- coding: utf-8 -*-
"""
item3: 同一入力によるtrain/serve parity検証（読み取り専用）。

同じas-of日時・同じ馬・同じ履歴を、(a) training builder相当のロジック
（DNF_SEMANTIC_SPEC.md準拠、母集団=止込み・外消除外で単純集計）と
(b) 実際のlive serve関数（`serve_history_feats.compute_row_feats()`、
course/jockey/hist_same/horse_fuku用。kako5は別関数
`parse_kako5.build_from_kako5()`）へ**そのまま**入力し、19特徴の値を比較する。

目的: 「実データでの現状の食い違い」ではなく、「同じ正しい入力を与えたとき、
コード自体(compute_row_feats/build_from_kako5)が同じ値を計算するか」を見る。
これにより、本番で観測される食い違いが (A) 上流データソースの共有バグに
起因するのか (B) コードロジック自体が異なる(真のserve skew)のかを分離する。

シナリオ(最低限、指示通り):
  1. 過去DNFなし
  2. 過去DNF1回
  3. 過去DNF複数回
  4. 騎手継続
  5. 騎手変更
  6. 同一競馬場
  7. 競馬場変更
  8. kako5窓の1〜5走目にDNF(各位置)
  9. 取消・除外履歴(scratchがentに混入した場合の挙動、上流データ品質への依存を確認)

実行: venv311\\Scripts\\python.exe -m analysis.mcond.p0_dnf_history_parity_audit.train_serve_same_input_parity
"""
from __future__ import annotations
import csv
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(BASE))
from serve_history_feats import compute_row_feats  # noqa: E402
from parse_kako5 import build_from_kako5, KAKO5_COLS  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "out"


# ============================================================
# PART 1: course/jockey系6特徴
# ============================================================
def make_ent(rows: list[dict]) -> dict:
    """rows: [{"date":YYYYMMDD, "place":str, "surface":"芝"|"ダ", "dist":float,
    "pos": float|nan, "jockey": float}, ...] を日付昇順で ent 形式へ変換。
    pos=nanの行はDNF(止)を表す(意味定義spec: 着順欠損・場所/サーフェス/距離は実値)。"""
    rows = sorted(rows, key=lambda r: r["date"])
    return {
        "date": np.array([r["date"] for r in rows], dtype=np.int32),
        "place": np.array([r["place"] for r in rows], dtype=object),
        "surface": np.array([r["surface"] for r in rows], dtype=object),
        "dist": np.array([r["dist"] for r in rows], dtype=np.float64),
        "pos": np.array([r["pos"] for r in rows], dtype=np.float64),
        "jockey": np.array([r["jockey"] for r in rows], dtype=np.float64),
    }


def training_equivalent_course_jockey(ent: dict, race_date: int, place: str,
                                       surface: str, dist: float, jockey_code: float) -> dict:
    """DNF_SEMANTIC_SPEC.md準拠の「あるべきtraining値」を、entと全く同じ
    (既にscratch除外済み・DNFはpos=nanで残存)配列から素朴に計算する
    (build_master_v2.compute_history_features()と同一の算術、単一クエリ版)。"""
    m = ent["date"] < race_date
    past = {k: ent[k][m] for k in ent}
    band = _dist_band(dist)

    def dist_band_arr(d):
        return np.array([_dist_band(x) for x in d])

    out = {}
    if len(past["date"]):
        bands = dist_band_arr(past["dist"])
        sel = (past["place"] == place) & (past["surface"] == surface) & (bands == band)
    else:
        sel = np.zeros(0, dtype=bool)
    n = int(sel.sum())
    out["course_n_prev"] = float(n)
    if n > 0:
        pos = past["pos"][sel]
        out["course_win_rate"] = float(np.nansum(pos == 1)) / n
        out["course_top3_rate"] = float(np.nansum(pos <= 3)) / n
    else:
        out["course_win_rate"] = np.nan
        out["course_top3_rate"] = np.nan

    if len(past["date"]):
        selj = past["jockey"] == jockey_code
        nj = int(selj.sum())
    else:
        nj = 0
    out["jockey_n_prev"] = float(nj)
    if nj > 0:
        posj = past["pos"][selj]
        out["jockey_win_rate"] = float(np.nansum(posj == 1)) / nj
        out["jockey_top3_rate"] = float(np.nansum(posj <= 3)) / nj
    else:
        out["jockey_win_rate"] = np.nan
        out["jockey_top3_rate"] = np.nan
    return out


def _dist_band(d):
    if pd.isna(d):
        return "?"
    d = int(d)
    if d <= 1400:
        return "短"
    if d <= 1700:
        return "マ"
    if d <= 2200:
        return "中"
    return "長"


HIST6 = ["course_n_prev", "course_win_rate", "course_top3_rate",
          "jockey_n_prev", "jockey_win_rate", "jockey_top3_rate"]


def run_course_jockey_scenarios() -> list[dict]:
    target = {"race_date": 20240301, "place": "東京", "surface": "ダ", "dist": 1600.0, "jockey": 100.0}
    scenarios = {
        "1_no_dnf": [
            {"date": 20240101, "place": "東京", "surface": "ダ", "dist": 1600.0, "pos": 3.0, "jockey": 100.0},
            {"date": 20240201, "place": "東京", "surface": "ダ", "dist": 1600.0, "pos": 1.0, "jockey": 100.0},
        ],
        "2_one_dnf": [
            {"date": 20240101, "place": "東京", "surface": "ダ", "dist": 1600.0, "pos": 3.0, "jockey": 100.0},
            {"date": 20240201, "place": "東京", "surface": "ダ", "dist": 1600.0, "pos": np.nan, "jockey": 100.0},
        ],
        "3_multi_dnf": [
            {"date": 20231101, "place": "東京", "surface": "ダ", "dist": 1600.0, "pos": np.nan, "jockey": 100.0},
            {"date": 20231201, "place": "東京", "surface": "ダ", "dist": 1600.0, "pos": np.nan, "jockey": 100.0},
            {"date": 20240201, "place": "東京", "surface": "ダ", "dist": 1600.0, "pos": 2.0, "jockey": 100.0},
        ],
        "4_jockey_continuity": [
            {"date": 20240101, "place": "東京", "surface": "ダ", "dist": 1600.0, "pos": np.nan, "jockey": 100.0},
            {"date": 20240201, "place": "東京", "surface": "ダ", "dist": 1600.0, "pos": 4.0, "jockey": 100.0},
        ],
        "5_jockey_change": [
            {"date": 20240101, "place": "東京", "surface": "ダ", "dist": 1600.0, "pos": np.nan, "jockey": 999.0},
            {"date": 20240201, "place": "東京", "surface": "ダ", "dist": 1600.0, "pos": 4.0, "jockey": 100.0},
        ],
        "6_same_venue": [
            {"date": 20240101, "place": "東京", "surface": "ダ", "dist": 1600.0, "pos": np.nan, "jockey": 100.0},
        ],
        "7_venue_change": [
            {"date": 20240101, "place": "中山", "surface": "ダ", "dist": 1600.0, "pos": np.nan, "jockey": 100.0},
        ],
        "9_scratch_present_in_ent": [
            # 意味定義spec上は外/消はentに含まれるべきではない。上流builderが
            # 誤って含めた場合にcompute_row_feats自身が区別できるかを確認する目的。
            {"date": 20240101, "place": "東京", "surface": "ダ", "dist": 1600.0, "pos": np.nan, "jockey": 100.0},  # 想定: 実は「外」
            {"date": 20240201, "place": "東京", "surface": "ダ", "dist": 1600.0, "pos": 5.0, "jockey": 100.0},
        ],
    }
    results = []
    for name, rows in scenarios.items():
        ent = make_ent(rows)
        serve_val = compute_row_feats(ent, target["race_date"], target["place"],
                                       target["surface"], target["dist"], target["jockey"])
        train_val = training_equivalent_course_jockey(
            ent, target["race_date"], target["place"], target["surface"],
            target["dist"], target["jockey"])
        row = {"scenario": name}
        for c in HIST6:
            sv = serve_val.get(c)
            tv = train_val.get(c)
            match = (pd.isna(sv) and pd.isna(tv)) or (
                not pd.isna(sv) and not pd.isna(tv) and abs(sv - tv) < 1e-9)
            row[c] = {"serve": sv, "train_equivalent": tv, "match": bool(match)}
        results.append(row)
    return results


# ============================================================
# PART 2: kako5系13特徴
# ============================================================
def make_kako5_csv_row(race_id: str, place: str, td: str, dist: float, ban: int,
                        past_races: list[dict | None]) -> tuple[list[str], list[str]]:
    """past_races: 新しい順(1走前が先頭)、最大5件。各要素は
    {"place":str,"pos":int|"止"|"外"|"消"|0,"ninki":int,"agari":float,"td":str,"dist":float}
    または None(空スロット)。TARGET週次kako5 CSVと同じ72列レイアウトで
    ヘッダ行+データ行のCSV文字列2行を返す。"""
    header = [""] * 19
    header[0] = race_id
    header[3] = place
    header[8] = td
    header[9] = str(dist)

    row = [""] * 72
    row[0] = "1"  # 枠番
    row[2] = str(ban)  # 馬番
    race_offsets = [(14, 18, 19, 21), (26, 30, 31, 33), (38, 42, 43, 45),
                    (50, 54, 55, 57), (62, 66, 67, 69)]
    td_offsets = [15, 27, 39, 51, 63]
    dist_offsets = [16, 28, 40, 52, 64]
    for i in range(5):
        pr = past_races[i] if i < len(past_races) else None
        place_i, pos_i, ninki_i, agari_i = race_offsets[i]
        if pr is None:
            row[pos_i] = "0"
            continue
        row[place_i] = pr.get("place", "")
        row[pos_i] = str(pr.get("pos", "0"))
        row[ninki_i] = str(pr.get("ninki", ""))
        row[agari_i] = str(pr.get("agari", ""))
        row[td_offsets[i]] = pr.get("td", "")
        row[dist_offsets[i]] = str(pr.get("dist", ""))
    return header, row


def training_equivalent_kako5(past_races_chrono_desc: list[dict | None], current_td: str,
                               current_dist: float, current_place: str) -> dict:
    """DNF_SEMANTIC_SPEC.md準拠のkako5値を、parse_kako5._compute_features()を
    そのままimportして計算する(ロジック複製ではなくimport)。DNFスロットは
    着順=None・上り3F=None・TD/距離/場所=実値として渡す。Noneスロット(スロット
    自体がない=取消/除外/空)は past_races から除外する。"""
    from parse_kako5 import _compute_features
    past_races = []
    for pr in past_races_chrono_desc:
        if pr is None:
            continue
        is_dnf = pr.get("is_dnf", False)
        past_races.append({
            "着順": None if is_dnf else pr.get("pos"),
            "人気": pr.get("ninki"),
            "上り3F": None if is_dnf else pr.get("agari"),
            "TD": pr.get("td", ""),
            "距離": pr.get("dist"),
            "場所": pr.get("place", ""),
        })
    return _compute_features(past_races, current_td=current_td,
                              current_dist=current_dist, current_place=current_place)


def run_kako5_scenarios() -> list[dict]:
    current = {"place": "東京", "td": "D", "dist": 1600.0}
    base_race = lambda pos, place="東京", td="D", dist=1600.0, agari=36.0, ninki=3: {
        "place": place, "pos": pos, "ninki": ninki, "agari": agari, "td": td, "dist": dist}

    scenarios = {
        "1_no_dnf": [base_race(3), base_race(1)],
        "2_one_dnf": [base_race("止"), base_race(2)],
        "3_multi_dnf": [base_race("止"), base_race("止"), base_race(4)],
        "8_dnf_at_slot1": [base_race("止"), base_race(2), base_race(3), base_race(4), base_race(5)],
        "8_dnf_at_slot3": [base_race(1), base_race(2), base_race("止"), base_race(4), base_race(5)],
        "8_dnf_at_slot5": [base_race(1), base_race(2), base_race(3), base_race(4), base_race("止")],
        "9_scratch_history": [base_race("外"), base_race(2)],
        "10_dnf_different_condition": [
            base_race("止", place="中山", td="芝", dist=2000.0),  # DNFだけ条件が違う
            base_race(2, place="東京", td="D", dist=1600.0),
            base_race(3, place="東京", td="D", dist=1600.0),
        ],
    }

    results = []
    for name, races_new_to_old in scenarios.items():
        # --- serve側: 実際のbuild_from_kako5()へ合成CSVを食わせる ---
        race_id = "2024030108010101"
        past_for_csv = []
        for r in races_new_to_old:
            past_for_csv.append({
                "place": r["place"], "pos": r["pos"], "ninki": r["ninki"],
                "agari": r["agari"], "td": r["td"], "dist": r["dist"],
            })
        header, datarow = make_kako5_csv_row(race_id, current["place"], current["td"],
                                              current["dist"], 1, past_for_csv)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False,
                                          encoding="cp932", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerow(datarow)
            tmp_path = Path(f.name)
        try:
            serve_df = build_from_kako5(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)
        serve_val = serve_df.iloc[0].to_dict() if len(serve_df) else {c: np.nan for c in KAKO5_COLS}

        # --- training側: _compute_features()へ意味定義spec準拠の入力を渡す ---
        train_input = []
        for r in races_new_to_old:
            is_dnf = r["pos"] in ("止",)
            is_scratch = r["pos"] in ("外", "消")
            if is_scratch:
                continue  # scratchはwindow slotへ含めない(意味定義spec)
            train_input.append({
                "place": r["place"], "pos": None if is_dnf else r["pos"],
                "ninki": r["ninki"], "agari": r["agari"], "td": r["td"],
                "dist": r["dist"], "is_dnf": is_dnf,
            })
        train_val = training_equivalent_kako5(train_input, current["td"], current["dist"], current["place"])

        row = {"scenario": name}
        for c in KAKO5_COLS:
            sv = serve_val.get(c, np.nan)
            tv = train_val.get(c, np.nan)
            sv = float(sv) if pd.notna(sv) else np.nan
            tv = float(tv) if pd.notna(tv) else np.nan
            match = (pd.isna(sv) and pd.isna(tv)) or (
                not pd.isna(sv) and not pd.isna(tv) and abs(sv - tv) < 1e-6)
            row[c] = {"serve": sv, "train_equivalent": tv, "match": bool(match)}
        results.append(row)
    return results


def main():
    print("=== PART1: course/jockey同一入力parity ===")
    cj_results = run_course_jockey_scenarios()
    for r in cj_results:
        mismatches = [c for c in HIST6 if not r[c]["match"]]
        print(f"  {r['scenario']}: {'ALL MATCH' if not mismatches else f'MISMATCH in {mismatches}'}")

    print("\n=== PART2: kako5同一入力parity ===")
    k5_results = run_kako5_scenarios()
    for r in k5_results:
        mismatches = [c for c in KAKO5_COLS if not r[c]["match"]]
        print(f"  {r['scenario']}: {'ALL MATCH' if not mismatches else f'MISMATCH in {mismatches}'}")

    out = {"course_jockey": cj_results, "kako5": k5_results}
    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / "same_input_parity.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
