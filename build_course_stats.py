"""
build_course_stats.py
=====================
master_v2_*.csv (~390MB) からコース別の集計統計だけ抜き出して、
data/course_stats.json (小さい、~数百KB) に保存する。

これを HF Spaces にも同期することで、nicegui_app.py のコース分析タブが
master_v2 を読まずに過去成績テーブルを表示できる。

使い方:
    python build_course_stats.py            # data/master_v2_*.csv の最新を使用
    python build_course_stats.py --master data/master_v2_xxx.csv
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
DEFAULT_OUT = DATA_DIR / "course_stats.json"


def parse_course_label(surface: str, distance: int) -> list[str]:
    """master の (surface='ダ'/'芝', distance) から bundle の course 文字列候補を返す。

    bundle.json の race_meta.course は 'ダート1800' '芝1600' のように
    full name で書かれているため、両方ヒットするキー候補を出す。
    """
    keys = []
    if surface == "ダ":
        keys.append(f"ダート{distance}")
        keys.append(f"ダ{distance}")
    elif surface == "芝":
        keys.append(f"芝{distance}")
    return keys


def stats_breakdown(sub: pd.DataFrame, col: str, values: list,
                     label_fn=None) -> list[dict]:
    out = []
    for v in values:
        grp = sub[sub[col] == v]
        total = len(grp)
        if total == 0:
            continue
        wins = int((grp["着順"] == 1).sum())
        top2 = int((grp["着順"] <= 2).sum())
        top3 = int((grp["着順"] <= 3).sum())
        label = label_fn(v) if label_fn else str(v)
        out.append({
            "label": label,
            "n_1": wins,
            "n_2": top2 - wins,
            "n_3": top3 - top2,
            "n_out": total - top3,
            "n_total": total,
            "win_rate":    round(wins / total * 100, 2),
            "rentai_rate": round(top2 / total * 100, 2),
            "fuku_rate":   round(top3 / total * 100, 2),
        })
    return out


def classify_kyaku(pos4: float, field_size: float) -> str:
    """前走 4 角通過位置から脚質を推定 (field_size 相対)。"""
    if pd.isna(pos4) or pd.isna(field_size) or field_size < 1:
        return "不明"
    p = float(pos4) / float(field_size)
    if p <= 0.20:
        return "逃げ"
    if p <= 0.45:
        return "先行"
    if p <= 0.70:
        return "差し"
    return "追込"


def compute_stats_for_group(sub: pd.DataFrame, place: str,
                              surface: str, distance: int) -> dict:
    """(place, surface, distance) の sub df から全条件別統計を算出。"""
    n_races = sub["日付"].nunique() if "日付" in sub.columns else len(sub) // 14

    waku = stats_breakdown(sub, "枠番", list(range(1, 9)),
                             label_fn=str)
    uma_values = sorted(sub["馬番"].dropna().unique().astype(int))
    uma_values = [v for v in uma_values if 1 <= v <= 18]
    uma = stats_breakdown(sub, "馬番", uma_values, label_fn=str)

    age = stats_breakdown(sub, "年齢", [3, 4, 5, 6, 7],
                            label_fn=lambda v: f"{v}歳")
    grp_8 = sub[sub["年齢"] >= 8]
    total = len(grp_8)
    if total > 0:
        wins = int((grp_8["着順"] == 1).sum())
        top2 = int((grp_8["着順"] <= 2).sum())
        top3 = int((grp_8["着順"] <= 3).sum())
        age.append({
            "label": "8歳~",
            "n_1": wins, "n_2": top2 - wins, "n_3": top3 - top2,
            "n_out": total - top3, "n_total": total,
            "win_rate":    round(wins / total * 100, 2),
            "rentai_rate": round(top2 / total * 100, 2),
            "fuku_rate":   round(top3 / total * 100, 2),
        })

    sex = stats_breakdown(sub, "性別", ["牡", "牝", "セ"],
                            label_fn=lambda v: {"牡":"牡馬","牝":"牝馬","セ":"セン馬"}.get(v, v))

    # 脚質別好走 (前走 4 角通過位置から各馬の脚質を推定)
    # master_v2 の 前4角 は「前走の 4 角通過位置」なので、馬の脚質を直接示す
    # (この race の通過位置ではないが、馬の脚質はレース間で大体一貫しているので
    #  近似として有効)
    kyaku: list[dict] = []
    if {"前4角", "出走頭数"}.issubset(sub.columns):
        sub2 = sub.copy()
        sub2["_kyaku"] = [
            classify_kyaku(p, f)
            for p, f in zip(sub2["前4角"], sub2["出走頭数"])
        ]
        kyaku = stats_breakdown(sub2, "_kyaku",
                                  ["逃げ", "先行", "差し", "追込"])

    return {
        "place": place,
        "surface": surface,
        "distance": int(distance),
        "n_races": int(n_races),
        "n_starts": int(len(sub)),
        "waku": waku,
        "uma": uma,
        "age": age,
        "sex": sex,
        "kyaku": kyaku,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default=None,
                     help="master_v2 CSV path (default: latest match)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--min-starts", type=int, default=100,
                     help="exclude (place,surface,distance) with fewer rows")
    args = ap.parse_args()

    # locate master csv
    if args.master:
        master_path = Path(args.master)
    else:
        candidates = sorted(DATA_DIR.glob("master_v2_*.csv"))
        if not candidates:
            print("ERROR: no master_v2_*.csv found in data/")
            return 1
        master_path = candidates[-1]
    if not master_path.exists():
        print(f"ERROR: {master_path} not found")
        return 1
    print(f"reading {master_path} ...")

    usecols = ["日付", "場所", "枠番", "馬番", "着順",
                "芝・ダ", "距離", "年齢", "性別",
                "前4角", "出走頭数"]
    df = pd.read_csv(master_path, encoding="utf-8-sig",
                      usecols=usecols, dtype=str,
                      low_memory=False, on_bad_lines="skip")
    df["日付"] = pd.to_numeric(df["日付"], errors="coerce")
    df["枠番"] = pd.to_numeric(df["枠番"], errors="coerce")
    df["馬番"] = pd.to_numeric(df["馬番"], errors="coerce")
    df["着順"] = pd.to_numeric(df["着順"], errors="coerce")
    df["距離"] = pd.to_numeric(df["距離"], errors="coerce")
    df["年齢"] = pd.to_numeric(df["年齢"], errors="coerce")
    if "前4角" in df.columns:
        df["前4角"] = pd.to_numeric(df["前4角"], errors="coerce")
    if "出走頭数" in df.columns:
        df["出走頭数"] = pd.to_numeric(df["出走頭数"], errors="coerce")
    df = df.dropna(subset=["着順", "距離"]).copy()
    print(f"  rows: {len(df):,}")

    out: dict[str, dict] = {}
    n_skip = 0
    for (place, surface, dist), sub in df.groupby(
            ["場所", "芝・ダ", "距離"], dropna=True):
        if len(sub) < args.min_starts:
            n_skip += 1
            continue
        if not isinstance(place, str) or not isinstance(surface, str):
            continue
        if not str(place).strip() or not str(surface).strip():
            continue
        dist_int = int(dist)
        stats = compute_stats_for_group(sub, place, surface, dist_int)

        # Both lookup keys for bundle's course strings
        for course_key in parse_course_label(surface, dist_int):
            key = f"{place}|{course_key}"
            out[key] = stats

    print(f"  combinations: {len(out)} (skipped {n_skip} with <{args.min_starts} rows)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    size_kb = out_path.stat().st_size / 1024
    print(f"  wrote {out_path}  ({size_kb:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
