"""
market_provenance_audit.py — 市場オッズprovenance横断監査（読み取り専用）
==========================================================================
2026-09-22、EXP05/EXP07/EXP09の「historical_pre_snapshotが存在する」という
過去報告と、EXP13 Gate0Aの「2023年coverage 0%」という結論の矛盾を検証する。

結論（実データで確定）: EXP13 Gate0Aは誤りだった。`data/Time _series_odds/
TANPUK_*.csv`（2011-2025年、区分1=前売り時系列・区分4=確定）という、
EXP13監査時に発見できなかった生ソースが存在し、`analysis/mcond/market.py`→
`v6base.py`→`base.parquet`という既存パイプラインを通じてEXP01/04/05/06/07/09
が共通利用している。この「pre」スナップショットは実測で発走のおよそ
26-30分前（中央値28分、EXP07 DATA_AUDIT.md §5.4の既存監査と一致）であり、
確定オッズではない。止馬（DNF）についても同等のcoverageがあることを
本スクリプトで確認する。

出力:
  docs/research/market_provenance_manifests/*.json （artifactごとのmanifest）
  標準出力にyear×started/DNF別のcoverage表

実行: venv311\\Scripts\\python.exe -m analysis.market_provenance_audit
（`data/`・外部データのいずれも変更しない。既存artifactを読むのみ）
"""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
ODIR = BASE / "data" / "Time _series_odds"
MASTER_V2 = BASE / "data" / "master_v2_20130105-20251228.csv"
KEKKA_EXT = Path(r"E:\競馬過去走データ\raw_data\kekka_2010_2025_fix_raceid_v2__keyed.csv")
OUT_DIR = BASE / "docs" / "research" / "market_provenance_manifests"

# EXP07 DATA_AUDIT.md §5.4 が確定した用語 (2026-09-20夜、ユーザー指示による正式名称)。
# 「T-10」「T-28」等のT-記法ではなくこちらを使う。
HISTORICAL_PRE_SNAPSHOT_WINDOW = (26.0, 30.0)  # 分、発走時刻基準の実測レンジ


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def classify_finish_code(code) -> str:
    if code == "止":
        return "chuushi_dnf"
    if code == "外":
        return "jogai_exclude_prerace"
    if code == "消":
        return "torikeshi_withdraw_prerace"
    if isinstance(code, str) and len(code) == 1 and 0x2460 <= ord(code) <= 0x2473:
        return "circled_demoted_finish"
    try:
        float(code)
        return "numeric_finish"
    except (TypeError, ValueError):
        return "unknown"


def load_kekka_labels(year_min: int, year_max: int) -> pd.DataFrame:
    """外部kekkaファイルから started/DNF ラベルを構築する (race_id列を使う。
    race_id16はEXP13当初のバグで別スキーマ、[[project_kekka_ext_data_quirks]]参照)。"""
    kk = pd.read_csv(KEKKA_EXT, encoding="utf-8-sig",
                     usecols=["日付", "着順", "race_id", "umaban", "レース名"], dtype=str)
    yy = kk["日付"].str[:2].astype(int)
    kk["year"] = yy.apply(lambda x: 2000 + x if x < 50 else 1900 + x)
    kk = kk[(kk["year"] >= year_min) & (kk["year"] <= year_max)].copy()
    kk["is_jump"] = kk["レース名"].str.contains("障害", na=False)
    kk = kk[~kk["is_jump"]].copy()
    kk["cls"] = kk["着順"].apply(classify_finish_code)
    kk["started"] = kk["cls"].isin(["numeric_finish", "chuushi_dnf", "circled_demoted_finish"])
    kk["is_dnf"] = kk["cls"] == "chuushi_dnf"
    kk["ban_i"] = pd.to_numeric(kk["umaban"], errors="coerce")
    kk = kk.dropna(subset=["ban_i"])
    kk["ban_i"] = kk["ban_i"].astype(int)
    kk["rid16"] = kk["race_id"].astype(str)
    return kk[kk["rid16"].str.len() == 16].copy()


def load_tanpuk_interim(year_min: int, year_max: int) -> pd.DataFrame:
    """TANPUK区分1(前売り)のうち、同日かつ最新の1件を馬単位で抽出する。"""
    files = sorted(ODIR.glob("TANPUK_*.csv"))
    frames = [pd.read_csv(f, encoding="cp932", low_memory=False) for f in files]
    tp = pd.concat(frames, ignore_index=True)
    c = list(tp.columns)
    RID, KB, TM = c[0], c[1], c[2]
    tan_cols = {}
    for x in c:
        m = re.match(r"^\s*(\d+)\s*単\s*$", str(x))
        if m:
            tan_cols[int(m.group(1))] = x
    tp["rid16"] = tp[RID].astype(str).str.replace(r"\D", "", regex=True).str[:16]
    tp = tp[tp["rid16"].str.len() == 16].copy()
    tp["year"] = tp["rid16"].str[:4].astype(int)
    tp = tp[(tp["year"] >= year_min) & (tp["year"] <= year_max)].copy()
    tp[KB] = pd.to_numeric(tp[KB], errors="coerce")
    tp[TM] = pd.to_numeric(tp[TM], errors="coerce")
    tp = tp.dropna(subset=[KB, TM])
    tp[TM] = tp[TM].astype(np.int64)
    tp["snap_mmdd"] = tp[TM] // 10000
    tp["race_mmdd"] = tp["rid16"].str[4:8].astype(int)
    tp["sameday"] = tp["snap_mmdd"] == tp["race_mmdd"]

    long = tp.melt(id_vars=["rid16", "year", KB, TM, "sameday"],
                   value_vars=list(tan_cols.values()), var_name="col", value_name="odds")
    long["ban"] = long["col"].str.extract(r"(\d+)").astype(int)
    long["odds"] = pd.to_numeric(long["odds"], errors="coerce")
    long = long.dropna(subset=["odds"])
    long = long[long["odds"] > 0]

    interim = long[(long[KB] == 1) & (long["sameday"])].copy()
    interim = interim.sort_values(TM).groupby(["rid16", "ban"]).tail(1)
    interim = interim.rename(columns={TM: "snap_mmddhhmm"})
    return interim


def load_post_times() -> pd.DataFrame:
    m2 = pd.read_csv(MASTER_V2, encoding="utf-8-sig",
                     usecols=["レースID(新/馬番無)", "発走時刻", "日付"], dtype=str)
    m2 = m2.drop_duplicates("レースID(新/馬番無)").rename(columns={"レースID(新/馬番無)": "rid16"})
    h = m2["発走時刻"].str.split(":").str[0].astype(float)
    mi = m2["発走時刻"].str.split(":").str[1].astype(float)
    date_dt = pd.to_datetime(m2["日付"], format="%Y%m%d", errors="coerce")
    m2["post_ts"] = date_dt + pd.to_timedelta(h, unit="h") + pd.to_timedelta(mi, unit="m")
    return m2[["rid16", "post_ts"]]


def _to_ts(mmddhhmm: np.ndarray, year: np.ndarray) -> pd.Series:
    mm = mmddhhmm // 1000000
    dd = (mmddhhmm // 10000) % 100
    hh = (mmddhhmm // 100) % 100
    mi = mmddhhmm % 100
    return pd.to_datetime(dict(year=year, month=mm, day=dd, hour=hh, minute=mi), errors="coerce")


def bucket_minutes(m: float) -> str:
    """分類はあくまでTANPUK由来のhistorical_pre_snapshotの実測分布を区間分けする
    ものであり、31-38分帯であっても forward_t35 (2026-09-19以降のEXP05-F実収集、
    別データソース) そのものではない点に注意 ([[用語の統一]]参照)。"""
    if pd.isna(m):
        return "no_snapshot"
    if 31 <= m <= 38:
        return "w31_38_min_to_post"
    lo, hi = HISTORICAL_PRE_SNAPSHOT_WINDOW
    if lo <= m <= hi:
        return "w26_30_historical_pre_snapshot"
    if m > 38:
        return "other_pre_post_gt38"
    if 0 <= m < lo:
        return "other_pre_post_lt26"
    return "post_race"


def build_coverage_table(year_min: int = 2023, year_max: int = 2025) -> pd.DataFrame:
    kk = load_kekka_labels(year_min, year_max)
    interim = load_tanpuk_interim(year_min, year_max).reset_index(drop=True)
    # 注意: _to_ts() は既定RangeIndexのSeriesを返す。interim側のindexは groupby+tail(1)
    # 由来で非連番のため、reset_index せずに列代入すると pandas がindexで整列し
    # 大半がNaNへ化ける (2026-09-22発見・修正: 本スクリプト自身のバグ、詳細は
    # docs/research/MARKET_DATA_PROVENANCE_AUDIT_20260921.md 参照)。
    interim["snap_ts"] = _to_ts(interim["snap_mmddhhmm"].to_numpy(), interim["year"].to_numpy()).to_numpy()
    post = load_post_times()
    interim = interim.merge(post, on="rid16", how="left")
    interim["minutes_to_post"] = (interim["post_ts"] - interim["snap_ts"]).dt.total_seconds() / 60.0

    full = kk.merge(interim[["rid16", "ban", "minutes_to_post"]],
                    left_on=["rid16", "ban_i"], right_on=["rid16", "ban"], how="left")
    full["bucket"] = full["minutes_to_post"].apply(bucket_minutes)
    return full


def write_manifest(name: str, path: Path, date_min: str, date_max: str, snapshot_rule: str,
                   timestamp_provenance: str, mtp_median, mtp_min, mtp_max,
                   final_odds: bool, safe_for_decision_time: bool) -> dict:
    manifest = {
        "source": str(path.relative_to(BASE)) if path.is_absolute() and BASE in path.parents
                  else str(path),
        "date_min": date_min,
        "date_max": date_max,
        "snapshot_rule": snapshot_rule,
        "timestamp_provenance": timestamp_provenance,
        "minutes_to_post_median": mtp_median,
        "minutes_to_post_min": mtp_min,
        "minutes_to_post_max": mtp_max,
        "final_odds": final_odds,
        "safe_for_decision_time": safe_for_decision_time,
        "sha256": sha256_file(path) if path.exists() else None,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{name}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return manifest


def main() -> int:
    full = build_coverage_table(2023, 2025)
    print("=== coverage by year x started/DNF (flat only, race_id-joined) ===")
    tab = full[full["started"]].groupby(["year", "is_dnf"])["bucket"].value_counts().unstack(fill_value=0)
    print(tab)
    print()
    for year in (2023, 2024, 2025):
        for dnf in (False, True):
            sub = full[(full["year"] == year) & (full["started"]) & (full["is_dnf"] == dnf)]
            n = len(sub)
            cov = int((sub["bucket"] == "w26_30_historical_pre_snapshot").sum())
            print(f"{year} {'DNF' if dnf else 'finisher'}: n={n} w26_30_coverage={cov} ({cov/n*100:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
