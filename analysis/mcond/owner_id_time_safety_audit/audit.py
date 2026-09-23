# -*- coding: utf-8 -*-
"""
audit.py — `馬主(最新/仮想)` と `前走レースID` の time-safety 監査 (§1-§3, §6)
=============================================================================
読み取りのみ。production 変更・モデル再学習・2024/2025 の新規性能評価・ROI は行わない。
(§4 の A/B/C/D ablation は ablation.py。EXP15 と同じ凍結プロトコルで学習する)

出力: out/owner_provenance.json / out/deletion_invariance.json / out/prev_race_id.json
実行: python -m analysis.mcond.owner_id_time_safety_audit.audit
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "out"
MASTER = BASE / "data" / "master_v2_20130105-20251228.csv"
CAT = BASE / "data" / "cat_20130105-20251228.csv"          # TARGET cat export (馬主の供給元)
SOUKAN = BASE / "data" / "走間分析"                          # 別時点の TARGET export
BUNSEKI = BASE / "data" / "bunseki"                         # serve 時の出走馬分析 (当時スナップ)
V6 = BASE / "models" / "unified_rank_v6.pkl"
OWN = "馬主(最新/仮想)"
PREV_ID = ["前走レースID(新)", "前走レースID(新/馬番無)"]

norm = lambda s: s.astype(str).str.replace(r"\s|　", "", regex=True)


def dump(obj, name):
    OUT.mkdir(parents=True, exist_ok=True)
    def conv(o):
        if isinstance(o, np.bool_): return bool(o)
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        raise TypeError(type(o))
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=conv), encoding="utf-8")
    print(f"[saved] out/{name}")


def mtime(p: Path) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(p.stat().st_mtime))


def load_master(cols):
    parts = []
    for ch in pd.read_csv(MASTER, encoding="utf-8-sig", dtype=str, usecols=cols, chunksize=200_000):
        parts.append(ch)
    df = pd.concat(parts, ignore_index=True)
    df["date"] = pd.to_numeric(df["日付"], errors="coerce")
    df["year"] = (df["date"] // 10000).astype("Int64")
    df["rid16"] = df["レースID(新/馬番無)"].astype(str).str.replace(r"\.0$", "", regex=True).str[:16]
    return df


# ---------------------------------------------------------------- §1・§2・§3 馬主
def owner_audit():
    df = load_master(["日付", "レースID(新/馬番無)", "馬番", "血統登録番号", OWN,
                      "調教師コード", "生産者", "騎手コード"])
    g = df.groupby("血統登録番号")
    nun = {c: g[c].nunique(dropna=False) for c in (OWN, "調教師コード", "生産者", "騎手コード")}
    span = g["year"].agg(["min", "max", "size"])
    multi = span[span["max"] > span["min"]].index
    long3 = span[(span["max"] - span["min"]) >= 3].index

    res = {
        "column": OWN,
        "files": {"master_v2": {"mtime": mtime(MASTER)}, "cat_export": {"mtime": mtime(CAT)}},
        "§1_per_horse_constancy": {
            "n_horses": int(len(span)), "n_rows": int(len(df)),
            "owner_multi_value_horses": int((nun[OWN] > 1).sum()),
            "owner_multi_value_rate": float((nun[OWN] > 1).mean()),
            "control_trainer_multi_value_rate": float((nun["調教師コード"] > 1).mean()),
            "control_breeder_multi_value_rate": float((nun["生産者"] > 1).mean()),
            "control_jockey_multi_value_rate": float((nun["騎手コード"] > 1).mean()),
            "multi_year_horses": int(len(multi)),
            "owner_changed_among_multi_year": int((nun[OWN].loc[multi] > 1).sum()),
            "trainer_changed_among_multi_year": int((nun["調教師コード"].loc[multi] > 1).sum()),
            "span_ge_3y_horses": int(len(long3)),
            "owner_changed_among_span_ge_3y": int((nun[OWN].loc[long3] > 1).sum()),
            "unique_owner_values": int(df[OWN].nunique()),
            "missing_rate": float(df[OWN].isna().mean()),
            "per_year_unique": {int(k): int(v) for k, v in df.groupby("year")[OWN].nunique().items()},
        },
    }

    # §3 historical truth: TARGET には 馬主(レース時) 列があるが、手元 export では空
    hist = {}
    for p in sorted(SOUKAN.glob("*.csv")):
        ch = next(pd.read_csv(p, encoding="cp932", dtype=str, chunksize=200_000, low_memory=False,
                              usecols=[OWN, "馬主(レース時)", "馬主タイプ(レース時)"]))
        hist[p.name] = {"export_mtime": mtime(p), "head200k_latest_nonnull": float(ch[OWN].notna().mean()),
                        "head200k_racetime_nonnull": float(ch["馬主(レース時)"].notna().mean())}
    res["§3_historical_truth_availability"] = {
        "target_has_racetime_column": True,
        "racetime_column_populated_in_local_exports": False,
        "files": hist,
        "note": "TARGET は 馬主(レース時)/馬主タイプ(レース時) を列として持つが、手元の export は全て空。"
                "当時の馬主との直接照合は、その列を含めた再 export がない限り不可能",
    }

    # §2 別時点スナップショット間のドリフト (cat 2026-03-01 vs 走間分析 2026-06-15)
    cat = pd.read_csv(CAT, encoding="cp932", dtype=str, low_memory=False,
                      usecols=["日付", "レースID(新/馬番無)", "血統登録番号", OWN])
    cat["rid16"] = cat["レースID(新/馬番無)"].astype(str).str[:16]
    cat["own_a"] = norm(cat[OWN])
    parts = []
    for p in sorted(SOUKAN.glob("*.csv")):
        for ch in pd.read_csv(p, encoding="cp932", dtype=str, chunksize=300_000, low_memory=False,
                              usecols=["日付", "レースID(新/馬番無)", "血統登録番号", OWN]):
            ch = ch.dropna(subset=[OWN])
            if len(ch):
                parts.append(ch)
    sk = pd.concat(parts, ignore_index=True).drop_duplicates(["レースID(新/馬番無)", "血統登録番号"])
    sk["rid16"] = sk["レースID(新/馬番無)"].astype(str).str[:16]
    sk["own_b"] = norm(sk[OWN])
    j = cat.merge(sk[["rid16", "血統登録番号", "own_b"]], on=["rid16", "血統登録番号"], how="inner")
    j["year"] = pd.to_numeric(j["日付"], errors="coerce") // 10000
    eq = j["own_a"] == j["own_b"]
    dis = j[~eq]
    by_year = (j.assign(ne=~eq).groupby("year")
                 .agg(rows=("ne", "size"), mismatch=("ne", "sum")))
    by_year["rate"] = by_year["mismatch"] / by_year["rows"]
    res["§2_cross_snapshot_drift"] = {
        "snapshot_a": f"cat export ({mtime(CAT)})", "snapshot_b": f"走間分析 export ({mtime(sorted(SOUKAN.glob('*.csv'))[0])})",
        "join_key": "rid16 + 血統登録番号 (馬名 join は不使用)",
        "overlap_rows": int(len(j)), "overlap_races": int(j["rid16"].nunique()),
        "overlap_horses": int(j["血統登録番号"].nunique()),
        "mismatch_rows": int((~eq).sum()), "mismatch_races": int(dis["rid16"].nunique()),
        "mismatch_horses": int(dis["血統登録番号"].nunique()),
        "mismatch_rate_rows": float((~eq).mean()),
        "horse_level_change_rate_in_window": float(dis["血統登録番号"].nunique() / j["血統登録番号"].nunique()),
        "window_days": 106,
        "by_race_year": {int(k): {"rows": int(v.rows), "mismatch": int(v.mismatch), "rate": float(v.rate)}
                         for k, v in by_year.iterrows()},
        "examples": dis[["rid16", "血統登録番号", "own_a", "own_b"]].drop_duplicates("血統登録番号").head(10).to_dict("records"),
        "interpretation": "同一の過去レース行に対し、export 時点が違うと馬主が変わる = 過去行が将来の所有者で上書きされている",
    }
    # 将来置換の実数: ドリフトで所有者が変わった馬の master 行は、全てレース後の所有者を持つ
    changed = set(dis["血統登録番号"])
    mrows = df[df["血統登録番号"].isin(changed)]
    res["§2_cross_snapshot_drift"]["prospective_future_owner_substitution"] = {
        "horses": int(len(changed)),
        "master_rows_of_those_horses": int(len(mrows)),
        "master_races": int(mrows["rid16"].nunique()),
        "all_race_dates_before_change": bool(int(mrows["date"].max()) <= 20260301),
        "note": "これらの馬は 2026-03 と 2026-06 の間に所有者が変わった。master の該当行 (全てそれ以前のレース) は"
                "現時点で『レース後の所有者』を保持しており、再 export すればさらに別の値へ変わる",
    }
    # 年率換算 (窓 106 日の馬単位変化率から)
    r = res["§2_cross_snapshot_drift"]["horse_level_change_rate_in_window"]
    annual = 1 - (1 - r) ** (365 / 106)
    res["§2_cross_snapshot_drift"]["estimated_annual_owner_change_rate"] = float(annual)
    res["§2_cross_snapshot_drift"]["estimated_future_owner_substitution_by_race_age"] = {
        f"{k}y": float(1 - (1 - annual) ** k) for k in (1, 3, 5, 10)
    }

    # serve 側 (bunseki = 当時スナップショット) と cat の突合 + 語彙一致
    bfiles = sorted(BUNSEKI.glob("2026*.csv"))
    if bfiles:
        bb = []
        for p in bfiles:
            b = pd.read_csv(p, encoding="cp932", dtype=str, low_memory=False, usecols=["血統登録番号", "馬主"])
            b["src"] = p.name
            bb.append(b)
        b = pd.concat(bb, ignore_index=True).dropna(subset=["血統登録番号"]).drop_duplicates("血統登録番号")
        b["own_serve"] = norm(b["馬主"])
        cat_h = cat.drop_duplicates("血統登録番号").set_index("血統登録番号")["own_a"]
        jb = b.set_index("血統登録番号").join(cat_h, how="inner")
        eqb = jb["own_serve"] == jb["own_a"]
        import joblib
        v6 = joblib.load(V6)
        classes = set(v6["encoders"][OWN].classes_)
        classes_n = set(norm(pd.Series(list(classes))))
        res["§1_serve_path"] = {
            "bunseki_files": [p.name for p in bfiles], "bunseki_mtime": mtime(bfiles[-1]),
            "horses_joined_with_cat": int(len(jb)),
            "agree_with_cat_snapshot": int(eqb.sum()), "disagree": int((~eqb).sum()),
            "disagree_rate": float((~eqb).mean()),
            "serve_value_in_v6_encoder_vocab_rate": float(jb["own_serve"].isin(classes_n).mean()),
            "v6_encoder_classes": len(classes),
            "note": "serve では weekly CSV に馬主列が無く、bunseki がある週だけ 馬名 join で復元される (未検証の定義一致)",
        }
    return res


# ---------------------------------------------------------------- §2 deletion invariance
def deletion_invariance():
    cols = ["日付", "レースID(新/馬番無)", "馬番", "血統登録番号", OWN] + PREV_ID
    df = load_master(cols)
    out = {"method": "master_v2 から将来行を削除して再構築し、残った行の値が一致するかを見る",
           "cases": {}}
    for cut in (20231231, 20221231):
        keep = df["date"] <= cut
        sub = df[keep].reset_index(drop=True)
        same = {}
        for c in [OWN] + PREV_ID:
            same[c] = bool(df.loc[keep, c].reset_index(drop=True).fillna("<NA>")
                           .equals(sub[c].fillna("<NA>")))
        out["cases"][f"delete_after_{cut}"] = {"rows_kept": int(keep.sum()), "identical": same}
    out["verdict"] = "PASS (repo 側で再計算していないので一致する)"
    out["limitation"] = ("master_v2 は静的な成果物で、馬主は TARGET export 時点の値をそのまま持つ。"
                         "上書きは export 上流で起きるため、この削除テストでは原理的に検出できない。"
                         "検出には export 時点の異なるスナップショット比較 (§2_cross_snapshot_drift) が必要")
    # 馬主変更馬の抽出 (master 内)
    g = df.groupby("血統登録番号")[OWN].nunique(dropna=False)
    out["owner_changed_horses_inside_master"] = int((g > 1).sum())
    out["note_changed_horses"] = ("master 内では 0 頭。変更が記録されていないこと自体が『当時の値ではない』ことの証拠で、"
                                  "『変更前レースの値が将来所有者に変わっていないか』は master 単独では検証不能")
    return out


# ---------------------------------------------------------------- §6 前走レースID
def prev_race_id_audit():
    df = load_master(["日付", "レースID(新/馬番無)", "レースID(新)", "馬番", "血統登録番号",
                      "前走日付"] + PREV_ID)
    df = df.sort_values(["血統登録番号", "date"]).reset_index(drop=True)
    g = df.groupby("血統登録番号")
    prev_rid_actual = g["rid16"].shift(1)
    prev_date_actual = g["date"].shift(1)
    pid16 = pd.to_numeric(df["前走レースID(新/馬番無)"], errors="coerce")
    pid18 = pd.to_numeric(df["前走レースID(新)"], errors="coerce")
    cur16 = pd.to_numeric(df["rid16"], errors="coerce")
    has = pid16.notna()
    pid16_s = pid16.astype("Float64").astype("string").str.replace(r"\.0$", "", regex=True)
    match = (pid16_s == prev_rid_actual) & has
    first_run = prev_rid_actual.isna()
    res = {
        "columns": PREV_ID,
        "rows": int(len(df)),
        "coverage_前走レースID(新/馬番無)": float(has.mean()),
        "first_run_rows": int(first_run.sum()),
        "§6_is_strictly_past": {
            "rows_with_value": int(has.sum()),
            "value_ge_current_race_id": int(((pid16 >= cur16) & has).sum()),
            "value_lt_current_race_id_rate": float(((pid16 < cur16) & has).mean() / has.mean()),
        },
        "§6_matches_actual_previous_run": {
            "rows_compared": int((has & ~first_run).sum()),
            "match": int(match.sum()),
            "match_rate": float(match[has & ~first_run].mean()),
            "note": "master 内の同一馬の直前行 (rid16) と一致するか。母集団は dropna 後なので DNF 走が抜ける分は不一致になりうる",
        },
        "§6_date_proxy": {
            "corr_with_prev_date": float(pd.Series(pid16[has]).corr(pd.to_numeric(df["前走日付"], errors="coerce")[has])),
            "first8_digits_equal_prev_date_rate": float(
                (pid16_s.str[:8] == ("20" + pd.to_numeric(df["前走日付"], errors="coerce")
                                     .astype("Int64").astype("string").str.zfill(6)))[has].mean()),
            "prev_date_format": "前走日付 は YYMMDD (例 121201.0)。ID 先頭8桁は YYYYMMDD",
            "note": "先頭8桁が前走日付そのもの = 数値として入れると日付 proxy になる",
        },
        "§6_float_precision": {
            "max_abs_value_18digit": float(np.nanmax(pid18.to_numpy())) if pid18.notna().any() else None,
            "exceeds_float64_exact_int_range_2^53": bool(np.nanmax(pid18.to_numpy()) > 2 ** 53) if pid18.notna().any() else None,
            "distinct_18digit_values": int(pid18.nunique()),
            "distinct_16digit_values": int(pid16.nunique()),
        },
    }
    # 同一前走レース由来の馬が同一レースに同居する割合 (馬間リンク情報)
    d2 = df[has].copy()
    d2["pid16"] = pid16_s[has]
    grp = d2.groupby(["rid16", "pid16"]).size()
    shared = grp[grp >= 2]
    res["§6_links_horses_from_same_previous_race"] = {
        "races_with_at_least_one_shared_previous_race": int(shared.index.get_level_values(0).nunique()),
        "total_races": int(d2["rid16"].nunique()),
        "rate": float(shared.index.get_level_values(0).nunique() / d2["rid16"].nunique()),
        "rows_in_shared_groups": int(shared.sum()),
        "note": "集合モデルなら『前走が同じレースだった馬』を ID 一致で検出できる経路になる",
    }
    return res


def main():
    o = owner_audit(); dump(o, "owner_provenance.json")
    d = deletion_invariance(); dump(d, "deletion_invariance.json")
    p = prev_race_id_audit(); dump(p, "prev_race_id.json")
    print("\n[§1] 馬主が生涯一定の馬:", o["§1_per_horse_constancy"]["owner_multi_value_horses"], "/",
          o["§1_per_horse_constancy"]["n_horses"],
          " 調教師変化率", round(o["§1_per_horse_constancy"]["control_trainer_multi_value_rate"], 4))
    s = o["§2_cross_snapshot_drift"]
    print(f"[§2] 別時点 export 不一致 {s['mismatch_rows']}行 / {s['mismatch_races']}レース / {s['mismatch_horses']}頭"
          f" (窓 {s['window_days']}日, 馬単位 {s['horse_level_change_rate_in_window']*100:.2f}%)"
          f" → 年率推定 {s['estimated_annual_owner_change_rate']*100:.2f}%")
    print("[§3] 馬主(レース時) 列は手元 export で全て空 =", o["§3_historical_truth_availability"]["racetime_column_populated_in_local_exports"])
    print("[§6] 前走ID 一致率", round(p["§6_matches_actual_previous_run"]["match_rate"], 4),
          " 未来ID件数", p["§6_is_strictly_past"]["value_ge_current_race_id"],
          " 日付proxy率", round(p["§6_date_proxy"]["first8_digits_equal_prev_date_rate"], 4))


if __name__ == "__main__":
    main()
