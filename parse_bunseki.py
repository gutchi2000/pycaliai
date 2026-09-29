"""
parse_bunseki.py
=================
data/bunseki/{YYYYMMDD}.csv （TARGET「出走馬分析」エクスポート）をパースする。

2026-09 発見: serve で長年埋まらなかった特徴（馬齢斤量差・前走馬体重(増減)・
前走出走頭数・前走場所・母馬・毛色・生産者・馬主・騎手/調教師年齢・
トラックコード(JV)）の TARGET ネイティブな供給源。枠発表前でも取得可能
（枠番/馬番は「仮」「未」になるが、他の列は施行前から確定値が入る）。

このファイルはヘッダ行あり（先頭列 "No."）で、他の週次ファイル(S/K/T)と違い
列数がユーザー側のTARGET項目選択で変動しうる。そのため位置ではなく列名で読む。

2026-09-05 検証済み（血統登録番号キーで master_v2 と突合、262 頭）:
  毛色 100%(262/262) / 生産者 100%(262/262) / 母馬 100%(262/262) /
  馬主 98.9%(259/262、残差はオーナー変更という自然な理由）。
  前走場所・前走出走頭数・前走馬体重・前走馬体重増減は「一覧」形式ファイルで
  kekka 実績と 100% 一致済み（前走場所/前走出走頭数=53/53、前走馬体重系=17/17）。
  ★馬名だけでの突合は同名の別馬を誤合成するので絶対に使わない
    （血統登録番号 or 種牡馬+生年での disambiguation が必須）。この理由で
    apply_bunseki() の 馬名 JOIN は「同一週内」限定にとどめる(全履歴横断はしない)。

`apply_bunseki(df, date_str)` で predict_weekly.parse_csv に配線済み（2026-09-05〜）。
data/bunseki/{date_str}.csv が無い週は何もせず fail-open（既存の定数フォールバック
のまま、後方互換）。再学習は不要——v6 は元々これらの特徴を学習済みで、
壊れていたのは serve 配信側だけ（本番未確認、初回運用は注意深く監視すること）。

`芝(内・外)` は bunseki 非依存。build_shiba_naigai_lookup.py が master_v2 から
場所×芝・ダ×距離 の静的変換表(data/shiba_naigai_lookup.json)を作り、
apply_shiba_naigai() で適用する（27 組み合わせ中 24 が決定的、レース単位で
内外が変わる 3 組み合わせ=京都芝1400/1600・新潟芝2000 は意図的に対象外）。

未解決のまま:
  前走競走種別(前ｸﾗｽ)はテキストラベル("未勝利"等)。master_v2 の
  "前走競走種別"は自己整合性チェックの結果、勝ち星クラスでなく
  年齢条件区分（2歳/3歳/3歳以上等）と判明——別概念のため対応せず。
  前走トラックコード(JV) は対応列が見当たらず未対応。

実行: python -c "from parse_bunseki import load_bunseki; print(load_bunseki('20260905'))"
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).parent
BUNSEKI_DIR = BASE / "data" / "bunseki"

# 出力する特徴名 -> bunseki 側の実列名
# (母馬/毛色/生産者/馬主 は master_v2 列名と定義一致を未検証。合わせて注記)
COL_MAP = {
    "馬齢斤量差":        "馬齢斤量差",
    "前走馬体重":        "前馬体重",
    "前走馬体重増減":     "前馬体重増減",
    "前走出走頭数":       "前頭数",
    "前走場所":          "前場所",
    "母馬":             "母名",
    "毛色":             "毛色",
    "生産者":            "生産者",
    "馬主(最新/仮想)":     "馬主",       # ★定義一致未検証
    "トラックコード(JV)":  "トラックコード(JV)",
}

AGE_RE = re.compile(r"\((\d+)歳\)")


def _extract_age(s) -> float:
    """'2005年 3月19日(21歳)' -> 21.0"""
    if pd.isna(s):
        return np.nan
    m = AGE_RE.search(str(s))
    return float(m.group(1)) if m else np.nan


def _clean_numeric(s) -> float:
    """'+8' '- 6' '0' ' 55 ' -> float。空文字/NaNはNaN。"""
    if pd.isna(s):
        return np.nan
    s = str(s).strip().replace(" ", "")
    if s == "":
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def load_bunseki(date_str: str, base: Path = BASE) -> pd.DataFrame:
    """data/bunseki/{date_str}.csv を読み、COL_MAP の特徴 + キー列を返す。

    キー: 馬名, レースID(新), 血統登録番号
    枠番/馬番は施行前だと "仮"/"未" のことがあるため、キーには使わない。
    """
    path = base / "data" / "bunseki" / f"{date_str}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} が無い（date_str={date_str}）")

    df = pd.read_csv(path, encoding="cp932", low_memory=False)
    missing_src = [v for v in COL_MAP.values() if v not in df.columns]
    if missing_src:
        raise ValueError(
            f"bunseki ファイルに期待した列が無い: {missing_src}. "
            f"TARGET側のエクスポート項目設定を確認すること。"
        )

    out = pd.DataFrame()
    out["馬名"] = df["馬名"].astype(str).str.strip()
    out["レースID(新)"] = df.get("レースID(新)", pd.Series(dtype=object)).astype(str)
    out["血統登録番号"] = df.get("血統登録番号", pd.Series(dtype=object)).astype(str)

    for feat, src in COL_MAP.items():
        col = df[src]
        if feat in ("馬齢斤量差", "前走馬体重", "前走馬体重増減",
                    "前走出走頭数", "トラックコード(JV)"):
            out[feat] = col.map(_clean_numeric)
        else:
            out[feat] = col.astype(str).str.strip().replace({"": np.nan, "nan": np.nan})

    if "騎手誕生日(歳)" in df.columns:
        out["騎手年齢"] = df["騎手誕生日(歳)"].map(_extract_age)
    if "調教師誕生日(歳)" in df.columns:
        out["調教師年齢"] = df["調教師誕生日(歳)"].map(_extract_age)

    return out


SHIBA_NAIGAI_JSON = BASE / "data" / "shiba_naigai_lookup.json"
_shiba_naigai_cache: dict | None = None


def _load_shiba_naigai() -> dict:
    global _shiba_naigai_cache
    if _shiba_naigai_cache is None:
        if SHIBA_NAIGAI_JSON.exists():
            _shiba_naigai_cache = json.loads(SHIBA_NAIGAI_JSON.read_text(encoding="utf-8"))
        else:
            _shiba_naigai_cache = {}
    return _shiba_naigai_cache


def apply_shiba_naigai(df: pd.DataFrame) -> pd.DataFrame:
    """場所×芝・ダ×距離 の静的変換表から 芝(内・外) を埋める（決定的な組み合わせのみ）。
    bunseki 非依存。build_shiba_naigai_lookup.py で作った表を使う。"""
    lookup = _load_shiba_naigai()
    if not lookup or "場所" not in df.columns or "距離" not in df.columns:
        return df
    df = df.copy()
    td_col = "芝・ダ" if "芝・ダ" in df.columns else "芝ダ"
    if td_col not in df.columns:
        return df
    dist_num = pd.to_numeric(df["距離"], errors="coerce")
    key = df["場所"].astype(str) + "|" + df[td_col].astype(str) + "|" + dist_num.astype("Int64").astype(str)
    resolved = key.map(lookup)
    if "芝(内・外)" not in df.columns:
        df["芝(内・外)"] = resolved
    else:
        df["芝(内・外)"] = df["芝(内・外)"].where(df["芝(内・外)"].notna() & (df["芝(内・外)"] != ""), resolved)
    return df


def apply_bunseki(df: pd.DataFrame, date_str: str, base: Path = BASE) -> pd.DataFrame:
    """週次 df (parse_csv 済み) に bunseki 特徴を 馬名 キーで JOIN する。

    data/bunseki/{date_str}.csv が無ければ何もせず df をそのまま返す(fail-open、
    後方互換)。同一週内の馬名重複は極めて稀という前提（全履歴横断の突合とは違い
    衝突リスクは低い）。"""
    path = base / "data" / "bunseki" / f"{date_str}.csv"
    if not path.exists() or "馬名" not in df.columns:
        return df
    try:
        b = load_bunseki(date_str, base)
    except Exception:
        return df

    df = df.copy()
    key = df["馬名"].astype(str).str.strip()
    b = b.drop_duplicates("馬名")
    b_indexed = b.set_index("馬名")

    for feat in list(COL_MAP.keys()) + ["騎手年齢", "調教師年齢"]:
        if feat not in b_indexed.columns:
            continue
        mapped = key.map(b_indexed[feat])
        if feat not in df.columns:
            df[feat] = mapped
        else:
            # bunseki値が取れた行だけ既存の定数フォールバックを上書きする
            df[feat] = mapped.where(mapped.notna(), df[feat])

    df = apply_shiba_naigai(df)
    return df


def coverage_report(date_str: str, base: Path = BASE) -> None:
    """簡易カバレッジ表示（動作確認用）。"""
    df = load_bunseki(date_str, base)
    n = len(df)
    print(f"[bunseki {date_str}] {n} 行")
    for c in df.columns:
        if c in ("馬名", "レースID(新)", "血統登録番号"):
            continue
        cov = df[c].notna().mean() * 100
        print(f"  {c:16s} coverage={cov:5.1f}%")


if __name__ == "__main__":
    import sys
    coverage_report(sys.argv[1] if len(sys.argv) > 1 else "20260905")
