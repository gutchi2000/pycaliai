"""
result_code_audit.py — EXP13 Stage 0: 結果コードの実測監査（読み取り専用）
==========================================================================
外部kekkaファイル(E:\\競馬過去走データ\\raw_data\\kekka_2010_2025_fix_raceid_v2__keyed.csv)
の生「着順」列を実測し、年別・芝ダート障害別に以下を集計する:
  - 正常完走 / 中止(止) / 除外(外) / 取消(消) / 降着相当(丸数字) / コード不明
  - 各コードについて「出走頭数==頭数か」「単勝オッズの有無」「走破タイムの有無」
    を突合し、started/finishedの二値を経験的に確定する

[[project_kekka_ext_data_quirks]]の罠(着順の全角/半角混在)に注意。本ファイルの
分類関数はTARGETの文字コード(止/外/消/丸数字)を直接判定するため全角/半角変換は
不要(数値着順のみ全角/半角混在の対象、コード文字は元々1バイト運用ではない)。

実行: venv311\\Scripts\\python.exe -m analysis.mcond.exp13_nonfinish_risk_dev.result_code_audit
出力: analysis/mcond/exp13_nonfinish_risk_dev/out/result_code_audit.json
"""
from __future__ import annotations
import json
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "out"
KEKKA_PATH = Path(r"E:\競馬過去走データ\raw_data\kekka_2010_2025_fix_raceid_v2__keyed.csv")

COLS = ["日付", "レース名", "クラス名", "着順", "頭数", "出走頭数",
        "単勝オッズ", "走破タイム", "芝・ダ", "race_id16", "年齢"]


def classify_code(code: str) -> str:
    """着順列の生コードを分類する。TARGET/JV-Link由来の実測値:
    数値=正常完走扱い(降着分含む)、'止'=競走中止(started, unfinished)、
    '外'=競走除外(not started)、'消'=出走取消(not started)、
    丸数字(①-⑮等)=降着相当(started, finished, 着順のみ事後修正)。
    それ以外は '不明' として個別確認対象にする(理由コードを断定しない)。"""
    if code == "止":
        return "chuushi_mid_race_stop"
    if code == "外":
        return "jogai_exclude_prerace"
    if code == "消":
        return "torikeshi_withdraw_prerace"
    if isinstance(code, str) and len(code) == 1 and 0x2460 <= ord(code) <= 0x2473:
        return "circled_number_demoted_finish"
    try:
        float(code)
        return "numeric_finish"
    except (TypeError, ValueError):
        return "unknown_code"


def load_kekka() -> pd.DataFrame:
    df = pd.read_csv(KEKKA_PATH, encoding="utf-8-sig", usecols=COLS, dtype=str)
    yy = df["日付"].str[:2].astype(int)
    df["year"] = yy.apply(lambda x: 2000 + x if x < 50 else 1900 + x)
    df["code_class"] = df["着順"].apply(classify_code)
    df["is_jump"] = df["レース名"].str.contains("障害", na=False)
    df["surface"] = df["芝・ダ"]
    df.loc[df["is_jump"], "surface"] = "障害"
    df["is_debut"] = df["クラス名"].str.contains("新馬", na=False)
    df["age_i"] = pd.to_numeric(df["年齢"], errors="coerce")
    df["n_entered"] = pd.to_numeric(df["頭数"], errors="coerce")
    df["n_started"] = pd.to_numeric(df["出走頭数"], errors="coerce")
    df["has_odds"] = df["単勝オッズ"].notna()
    df["has_time"] = df["走破タイム"].notna() & (df["走破タイム"] != "----")
    df["started"] = df["code_class"].isin(
        ["numeric_finish", "chuushi_mid_race_stop", "circled_number_demoted_finish"])
    df["positive"] = df["code_class"] == "chuushi_mid_race_stop"
    return df


def audit(df: pd.DataFrame) -> dict:
    out: dict = {"source": str(KEKKA_PATH), "total_rows": int(len(df)),
                "year_range": [int(df["year"].min()), int(df["year"].max())]}

    out["code_class_counts"] = df["code_class"].value_counts().to_dict()

    # 各コードクラスの「started/finished」経験的な確定根拠
    evidence = {}
    for cls, g in df.groupby("code_class"):
        evidence[cls] = {
            "n": int(len(g)),
            "started_eq_entered_rate": round(float((g["n_started"] == g["n_entered"]).mean()), 4),
            "has_odds_rate": round(float(g["has_odds"].mean()), 4),
            "has_time_rate": round(float(g["has_time"].mean()), 4),
        }
    out["code_class_evidence"] = evidence

    # 年別 x surface別: unique races / started / positive / rate
    g = df.groupby(["year", "surface"]).agg(
        rows=("着順", "size"), unique_races=("race_id16", "nunique"),
        started=("started", "sum"), positive=("positive", "sum"),
    ).reset_index()
    g["positive_rate_pct"] = (g["positive"] / g["started"] * 100).round(4)
    out["by_year_surface"] = g.to_dict(orient="records")

    surf_totals = df.groupby("surface").agg(
        rows=("着順", "size"), unique_races=("race_id16", "nunique"),
        started=("started", "sum"), positive=("positive", "sum"),
    )
    surf_totals["positive_rate_pct"] = (surf_totals["positive"] / surf_totals["started"] * 100).round(4)
    out["by_surface_total"] = surf_totals.to_dict(orient="index")

    # 平地限定: 新馬 vs 非新馬、2歳 vs 3歳以上
    flat = df[~df["is_jump"]]
    debut = flat.groupby("is_debut").agg(started=("started", "sum"), positive=("positive", "sum"))
    debut["rate_pct"] = (debut["positive"] / debut["started"] * 100).round(4)
    out["flat_by_debut"] = debut.to_dict(orient="index")

    flat = flat.copy()
    flat["age_grp"] = flat["age_i"].apply(lambda a: "2yo" if a == 2 else ("3yo_plus" if a >= 3 else "unknown"))
    age = flat.groupby("age_grp").agg(started=("started", "sum"), positive=("positive", "sum"))
    age["rate_pct"] = (age["positive"] / age["started"] * 100).round(4)
    out["flat_by_age_group"] = age.to_dict(orient="index")

    return out


def main() -> int:
    df = load_kekka()
    result = audit(df)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "result_code_audit.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"[result_code_audit] -> {out_path.relative_to(BASE)}")
    print(f"  code_class_counts: {result['code_class_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
