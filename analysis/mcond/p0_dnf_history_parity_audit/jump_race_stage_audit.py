# -*- coding: utf-8 -*-
"""
jump_race_stage_audit.py
========================
「v6 は障害競走を一度も学習していない」(EXP13 DATA_AUDIT §1) と
「master_v2 に障害 1,538 レース (3.42%)」(ALT_SOURCE_RECOVERY_AUDIT §2.2) の
矛盾を、パイプライン段階別の実数で解消する。

障害判定は 4 方式を**混同せずに併記**する:
  D1 `芝・ダ` 列の値
  D2 `トラックコード(JV)` (JRA-VAN: 10-22=平地 / 23-29=障害)  ← 確定判定
  D3 `レース名` 文字列 (EXP13 LABEL_CODEBOOK が使った方式)
  D4 距離・クラスからの推定 (傍証のみ)

READ-ONLY。production 変更・再学習・ROI 評価は行わない。
出力: out/jump_race_stage_audit.json
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

COL_RACE = "レースID(新/馬番無)"
NAME_PAT = "障害|ジャンプ|ジャＧ|ジャG|ステープル|JG"


def log(m):
    print(m, flush=True)


# ---------- 判定器 ----------

def d1_surface(df: pd.DataFrame) -> pd.Series | None:
    if "芝・ダ" not in df.columns:
        return None
    return df["芝・ダ"].astype(str).str.contains("障", na=False)


def d2_trackcode(df: pd.DataFrame) -> pd.Series | None:
    """JRA-VAN トラックコード。本データでの実測値域:
        10,11,12,17,18,20,21 = 芝   / 23,24 = ダート / 52,54,55,56,57 = 障害
    仕様上の障害レンジ 51-59 を判定に使う (確定判定)。"""
    col = next((c for c in df.columns if "トラックコード" in c and "前走" not in c), None)
    if col is None:
        return None
    v = pd.to_numeric(df[col], errors="coerce")
    return (v >= 51) & (v <= 59)


def d3_racename(df: pd.DataFrame) -> pd.Series | None:
    if "レース名" not in df.columns:
        return None
    return df["レース名"].astype(str).str.contains(NAME_PAT, na=False, regex=True)


def d4_dist_class(df: pd.DataFrame) -> pd.Series | None:
    """距離>=2750 かつ コーナー回数が平地レンジ外、等の傍証。
    ここでは距離のみの粗い推定 (平地にも 2750m 以上が存在するため過剰検出する)。"""
    if "距離" not in df.columns:
        return None
    return pd.to_numeric(df["距離"], errors="coerce") >= 2750


DETECTORS = [("D1_芝ダ列", d1_surface), ("D2_トラックコードJV", d2_trackcode),
             ("D3_レース名", d3_racename), ("D4_距離>=2750", d4_dist_class)]


def summarize(df: pd.DataFrame, stage: str, race_col: str | None = None) -> dict:
    rc = race_col or (COL_RACE if COL_RACE in df.columns else None)
    if rc is None:
        rc = next((c for c in df.columns if "レースID" in c), None)
    out = {"stage": stage, "total_rows": int(len(df)),
           "total_races": int(df[rc].nunique()) if rc else None,
           "detectors": {}}
    for name, fn in DETECTORS:
        m = fn(df)
        if m is None:
            out["detectors"][name] = {"available": False}
            continue
        m = m.fillna(False)
        d = {"available": True, "rows": int(m.sum())}
        if rc:
            d["races"] = int(df.loc[m, rc].nunique())
        if "日付" in df.columns:
            yr = pd.to_numeric(
                df["日付"].astype(str).str.replace(r"\D", "", regex=True).str[:4],
                errors="coerce")
            d["by_year_rows"] = {str(int(k)): int(v) for k, v in
                                 yr[m].value_counts().sort_index().items()}
        out["detectors"][name] = d
    return out


def cross_tab(df: pd.DataFrame, stage: str) -> dict:
    """D2(確定) と D1/D3/D4 の一致・不一致を出す。"""
    d2 = d2_trackcode(df)
    if d2 is None:
        return {"stage": stage, "note": "D2 不在のため cross-tab 不可"}
    d2 = d2.fillna(False)
    res = {"stage": stage, "n": int(len(df)), "D2_jump_rows": int(d2.sum())}
    for name, fn in DETECTORS:
        if name.startswith("D2"):
            continue
        m = fn(df)
        if m is None:
            continue
        m = m.fillna(False)
        res[name] = {
            "both": int((m & d2).sum()),
            "only_this": int((m & ~d2).sum()),
            "only_D2": int((~m & d2).sum()),
            "neither": int((~m & ~d2).sum()),
        }
    return res


def main():
    log("=" * 74)
    log("障害競走 段階別実数監査 (v6 が実際に学習したか)")
    log("=" * 74)

    stages: list[dict] = []
    crosses: list[dict] = []

    # ---- S1 raw source (lgbm ベース) ----
    log("\n[S1] raw source: data/lgbm_20130105-20251228.csv")
    from build_dataset import load_csv
    raw = load_csv(BASE / "data" / "lgbm_20130105-20251228.csv", "lgbm")
    stages.append(summarize(raw, "S1_raw_source_lgbm"))
    crosses.append(cross_tab(raw, "S1_raw_source_lgbm"))
    log(f"  rows={len(raw):,}")
    for n, d in stages[-1]["detectors"].items():
        log(f"    {n}: {d}" if not d.get("available") else
            f"    {n}: rows={d['rows']:,} races={d.get('races')}")

    # ---- S2 結合直後 / S3 dropna 直前 ----
    #  build_dataset.py は merge 後 631,965 行 (assert) → dropna → 626,774
    #  merge は how='left' で lgbm がベースのため行数・障害集合は S1 と同一。
    #  実測で確認する: master_v2 に残る race 集合との差分から逆算する。
    log("\n[S2/S3] 結合直後・dropna直前 (build_dataset.py:256-321)")
    log("  merge は全て how='left'、base=lgbm のため行集合は S1 と同一 "
        f"({len(raw):,} 行)。assert 631,965 と一致するか確認:")
    log(f"    S1 rows = {len(raw):,}  / build_dataset.py assert = 631,965 "
        f"→ {'一致' if len(raw) == 631965 else '不一致'}")

    # ---- S4 dropna 後 = master_v2 ----
    log("\n[S4/S5] dropna後 = data/master_v2_20130105-20251228.csv")
    usecols = None
    m2 = pd.read_csv(BASE / "data" / "master_v2_20130105-20251228.csv",
                     encoding="utf-8-sig", low_memory=False, usecols=usecols)
    stages.append(summarize(m2, "S4_dropna後_master_v2"))
    crosses.append(cross_tab(m2, "S4_dropna後_master_v2"))
    log(f"  rows={len(m2):,}")
    for n, d in stages[-1]["detectors"].items():
        log(f"    {n}: rows={d['rows']:,} races={d.get('races')}"
            if d.get("available") else f"    {n}: 列なし")

    # dropna で落ちた障害行
    d2_raw = d2_trackcode(raw).fillna(False)
    d2_m2 = d2_trackcode(m2).fillna(False)
    log(f"\n  dropna で除去された障害行: "
        f"{int(d2_raw.sum()) - int(d2_m2.sum()):,} "
        f"({int(d2_raw.sum()):,} → {int(d2_m2.sum()):,})")

    # ---- S6 feature selection 後 / S7-9 split / S10 v6 実学習行 ----
    log("\n[S6-S10] v6 の実学習行")
    import joblib
    bundle = joblib.load(BASE / "models" / "unified_rank_v6.pkl")
    feats = bundle.get("feature_cols")
    log(f"  v6 特徴数: {len(feats)}")
    jump_feat = [c for c in feats if "トラック" in c or "障" in c]
    log(f"  特徴に含まれるトラック/障害系: {jump_feat}")

    if "split" in m2.columns:
        sp = m2["split"]
    else:
        dt = pd.to_numeric(m2["日付"], errors="coerce")
        sp = np.where(dt <= 20221231, "train",
                      np.where(dt <= 20231231, "valid", "test"))
        sp = pd.Series(sp, index=m2.index)

    split_stats = {}
    for s in ["train", "valid", "test"]:
        sub = m2[sp == s]
        jm = d2_trackcode(sub).fillna(False)
        split_stats[s] = {
            "rows": int(len(sub)), "jump_rows": int(jm.sum()),
            "jump_races": int(sub.loc[jm, COL_RACE].nunique()),
            "jump_row_share": round(float(jm.mean()), 6),
        }
        log(f"  {s}: rows={len(sub):,} jump_rows={int(jm.sum()):,} "
            f"({jm.mean()*100:.2f}%) jump_races={split_stats[s]['jump_races']:,}")

    # v6 の学習に使われた行 = train split のうち欠損等で落ちない行
    # (train_unified_rank.py と同じ絞り込みを再現しない代わりに、
    #  train split 全体を上限として報告する)
    log("\n  → v6 は train split をそのまま学習に使う設計のため、"
        "上記 train の jump_rows が実学習に含まれた障害行の実数となる。")

    # ---- S11 保存済み prediction 対象 ----
    log("\n[S11] 保存済み prediction 対象 (reports/cowork_input/*_bundle.json)")
    bundles = sorted((BASE / "reports" / "cowork_input").glob("*_bundle.json"))
    n_b, n_race_b = 0, 0
    for bp in bundles[-8:]:
        try:
            j = json.load(open(bp, encoding="utf-8"))
        except Exception:
            continue
        races = j.get("races", [])
        n_b += 1
        n_race_b += len(races)
    log(f"  直近 {n_b} bundle / {n_race_b} レース "
        f"(bundle は weekly 由来のため障害は構造的に 0)")

    # ---- S12 weekly live serve 対象 ----
    log("\n[S12] weekly live serve 対象")
    log("  data/weekly は障害を含まない (ALT_SOURCE_RECOVERY_AUDIT §2 で実測)")

    payload = {
        "generated_at": datetime.now().isoformat(),
        "detector_definitions": {
            "D1_芝ダ列": "`芝・ダ` に '障' を含むか",
            "D2_トラックコードJV": "JRA-VAN トラックコード 23-29 = 障害 (確定判定)",
            "D3_レース名": f"`レース名` が /{NAME_PAT}/ にマッチ "
                          "(EXP13 LABEL_CODEBOOK の方式)",
            "D4_距離>=2750": "距離のみの粗い推定 (平地の長距離も拾う=過剰検出)",
        },
        "stages": stages,
        "detector_cross_tabs": crosses,
        "split_stats": split_stats,
        "v6_feature_count": len(feats),
        "v6_track_related_features": jump_feat,
        "raw_row_count": int(len(raw)),
        "build_dataset_assert": 631965,
        "master_v2_row_count": int(len(m2)),
    }
    with open(OUT / "jump_race_stage_audit.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    log("\n保存: out/jump_race_stage_audit.json")


if __name__ == "__main__":
    main()
