# -*- coding: utf-8 -*-
"""
phase_a_bunseki_audit.py — P1 forward-only collector の Phase A 検証
====================================================================
保存済み `data/bunseki/` 8 日だけを使い、collector の前提が成り立つかを測る。

READ-ONLY。原本 (data/bunseki, data/kekka) は一切変更しない。
production の特徴計算・予測・印・買い目には接続しない。

出力: analysis/jump_history_only/out/phase_a_bunseki_audit.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(parents=True, exist_ok=True)

BUNSEKI = BASE / "data" / "bunseki"
KEKKA = BASE / "data" / "kekka"

# 必須 fixture (Phase A 項目3)
FIXTURE_DATES = [20260905, 20260906]


def log(m):
    print(m, flush=True)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_bunseki(d: int) -> pd.DataFrame:
    p = BUNSEKI / f"{d}.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, encoding="cp932", dtype=str, on_bad_lines="skip")
    df["rid16"] = df["レースID(新)"].astype(str).str.strip().str[:16]
    df["umaban"] = pd.to_numeric(df["馬番"], errors="coerce")
    df["track_code"] = pd.to_numeric(df.get("トラックコード(JV)"), errors="coerce")
    df["jump_flag"] = df["track_code"].between(51, 59, inclusive="both")
    return df


def load_kekka(d: int) -> pd.DataFrame:
    p = KEKKA / f"{d}.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, encoding="cp932", dtype=str)
    df["rid16"] = df["レースID(新)"].astype(str).str.strip().str[:16]
    df["umaban"] = pd.to_numeric(df["馬番"], errors="coerce")
    df["finish_raw"] = df["確定着順"].astype(str).str.strip()
    df["finish"] = pd.to_numeric(df["確定着順"], errors="coerce")
    return df


def audit_date(d: int) -> dict:
    b, k = load_bunseki(d), load_kekka(d)
    r = {"date": d, "bunseki_exists": not b.empty, "kekka_exists": not k.empty}
    if b.empty:
        return r

    bp = BUNSEKI / f"{d}.csv"
    r["bunseki_sha256"] = sha256_file(bp)
    r["bunseki_mtime"] = datetime.fromtimestamp(
        bp.stat().st_mtime).isoformat(timespec="seconds")
    r["bunseki_rows"] = int(len(b))
    r["bunseki_races"] = int(b["rid16"].nunique())
    r["bunseki_jump_races"] = int(b.loc[b["jump_flag"], "rid16"].nunique())
    r["bunseki_jump_rows"] = int(b["jump_flag"].sum())
    r["track_code_values"] = {str(int(k_)): int(v) for k_, v in
                              b["track_code"].value_counts().sort_index().items()
                              if pd.notna(k_)}

    # --- 必須 ID coverage ---
    for name, col in [("ped_id", "血統登録番号"), ("jockey_code", "騎手コード"),
                      ("trainer_code", "調教師コード")]:
        if col in b.columns:
            v = b[col].astype(str).str.strip()
            ok = v.ne("") & v.ne("nan") & v.ne("0")
            r[f"{name}_coverage"] = round(float(ok.mean()), 6)
            r[f"{name}_missing"] = int((~ok).sum())
        else:
            r[f"{name}_coverage"] = None

    # --- duplicate ---
    key = b[["rid16", "umaban"]].dropna()
    r["duplicate_rows"] = int(len(key) - len(key.drop_duplicates()))
    r["duplicate_rate"] = round(r["duplicate_rows"] / max(len(key), 1), 6)
    ped = b["血統登録番号"].astype(str).str.strip() if "血統登録番号" in b else pd.Series(dtype=str)
    pk = pd.DataFrame({"rid16": b["rid16"], "ped": ped}).dropna()
    r["duplicate_rows_by_ped"] = int(len(pk) - len(pk.drop_duplicates()))

    if k.empty:
        return r

    kp = KEKKA / f"{d}.csv"
    r["kekka_sha256"] = sha256_file(kp)
    r["kekka_rows"] = int(len(k))
    r["kekka_races"] = int(k["rid16"].nunique())

    # --- race_id 一致率 (kekka 基準) ---
    br, kr = set(b["rid16"]), set(k["rid16"])
    r["race_recovery_rate"] = round(len(br & kr) / max(len(kr), 1), 6)
    r["races_in_kekka_not_bunseki"] = sorted(kr - br)
    r["races_in_bunseki_not_kekka"] = sorted(br - kr)

    # --- horse-row 回収率 / 馬番一致 (race_id + 馬番、馬名不使用) ---
    bk = set(zip(b["rid16"], b["umaban"]))
    kk = set(zip(k["rid16"], k["umaban"]))
    r["kekka_rows_matched_in_bunseki"] = len(bk & kk)
    r["horse_row_recovery_rate"] = round(len(bk & kk) / max(len(kk), 1), 6)
    r["bunseki_only_rows"] = len(bk - kk)   # card にあり結果に無い = 取消候補
    r["kekka_only_rows"] = len(kk - bk)

    # --- finish code coverage / 4 状態の判別可能性 ---
    kj = k.merge(b[["rid16", "umaban", "jump_flag"]], on=["rid16", "umaban"],
                 how="left", indicator=True)
    states = Counter()
    for _, row in kj.iterrows():
        if pd.notna(row["finish"]) and row["finish"] > 0:
            states["completed"] += 1
        else:
            states["non_completed(DNF or 除外, 判別不能)"] += 1
    states["scratched候補(card のみ・結果に無い)"] = len(bk - kk)
    r["state_counts"] = dict(states)
    r["finish_raw_values"] = {k_: int(v) for k_, v in
                              k["finish_raw"].value_counts().items()}
    r["kekka_join_rate"] = round(
        float((kj["_merge"] == "both").mean()), 6)
    return r


def main():
    log("=" * 74)
    log("Phase A — bunseki collector 検証 (保存済み 8 日)")
    log("=" * 74)

    dates = sorted(int(p.stem) for p in BUNSEKI.glob("*.csv")
                   if p.stem.isdigit())
    log(f"\nbunseki 保存日: {dates}")

    per_date = [audit_date(d) for d in dates]

    log(f"\n{'date':>10} {'b_rows':>7} {'b_race':>7} {'jump':>5} "
        f"{'race回収':>8} {'row回収':>8} {'ped':>6} {'jky':>6} {'trn':>6} {'dup':>4}")
    for r in per_date:
        if not r.get("bunseki_exists"):
            continue
        log(f"{r['date']:>10} {r['bunseki_rows']:>7} {r['bunseki_races']:>7} "
            f"{r['bunseki_jump_races']:>5} "
            f"{r.get('race_recovery_rate', float('nan')):>8.4f} "
            f"{r.get('horse_row_recovery_rate', float('nan')):>8.4f} "
            f"{r.get('ped_id_coverage', 0):>6.3f} "
            f"{r.get('jockey_code_coverage', 0):>6.3f} "
            f"{r.get('trainer_code_coverage', 0):>6.3f} "
            f"{r.get('duplicate_rows', 0):>4}")

    # ---- fixture 検証 ----
    log("\n--- 必須 fixture 検証 (20260905 / 20260906) ---")
    fixture = {}
    for d in FIXTURE_DATES:
        r = next((x for x in per_date if x["date"] == d), None)
        checks = {}
        if r is None or not r.get("bunseki_exists"):
            checks["bunseki_exists"] = False
        else:
            checks["bunseki_exists"] = True
            checks["kekka_exists"] = bool(r.get("kekka_exists"))
            checks["has_jump_race"] = r["bunseki_jump_races"] > 0
            checks["all_starters_present"] = (
                r.get("horse_row_recovery_rate", 0) == 1.0)
            checks["joinable_by_rid_umaban"] = (
                r.get("kekka_join_rate", 0) == 1.0)
            checks["no_name_join_needed"] = True   # 全 join を rid+馬番で実施
            checks["ped_id_full"] = r.get("ped_id_coverage") == 1.0
            checks["jockey_code_full"] = r.get("jockey_code_coverage") == 1.0
            checks["trainer_code_full"] = r.get("trainer_code_coverage") == 1.0
            checks["no_duplicates"] = r.get("duplicate_rows", 1) == 0
            st = r.get("state_counts", {})
            checks["can_distinguish_completed"] = "completed" in st
            checks["can_distinguish_scratch_candidate"] = (
                "scratched候補(card のみ・結果に無い)" in st)
            # 取消 と DNF の分離は kekka 単独では不可 (既確認)
            checks["can_separate_dnf_from_jogai"] = False
        fixture[str(d)] = checks
        allpass = all(v for k_, v in checks.items()
                      if k_ != "can_separate_dnf_from_jogai")
        log(f"  {d}: {'PASS' if allpass else 'CHECK'} {checks}")

    payload = {
        "generated_at": datetime.now().isoformat(),
        "scope": "Phase A — 保存済み bunseki のみ。production 未接続。",
        "bunseki_dates": dates,
        "per_date": per_date,
        "fixture_checks": fixture,
        "generation_path_audit": {
            "generator": "TARGET GUI の「出走馬分析」エクスポート（手動）",
            "intake": "data/_inbox/ へ投入 → place_weekly.py が "
                      "data/bunseki/{date}.csv へ振り分け",
            "detection": "ヘッダ行あり(先頭列 'No.') かつ "
                         "'馬齢斤量差'/'前場所' 列の有無で判定 "
                         "(place_weekly.py:80-96, 237-239)",
            "automatic": False,
            "scheduler_registered": False,
            "scheduler_evidence": "Get-ScheduledTask に bunseki/place_weekly "
                                  "関連タスクは存在しない "
                                  "(PyCaLiAI_Baba/T10/T20_Site/EXP05FS_* のみ)",
            "invocation": "weekly_nicegui.ps1 の Phase A/C 冒頭 Step 0 で "
                          "place_weekly.py が呼ばれる (-SkipIntake で無効化)",
            "generation_time": "TARGET からユーザーがエクスポートした時刻。"
                               "ファイル mtime が実際の保存時刻 (per_date に記録)",
            "includes_all_races": "実測で判定 (per_date.race_recovery_rate)",
            "jump_included_guaranteed": "★保証されていない。TARGET 側の項目選択・"
                                        "出力条件に依存し、設定が変われば "
                                        "障害が落ちうる。collector 側で "
                                        "jump_flag を毎回検査する必要がある",
            "overwrite_or_dated": "日付別保存 (data/bunseki/{date}.csv)。"
                                  "ただし同一日付を再投入すると "
                                  "place_weekly.py は上書きする",
            "failure_log": "place_weekly.py は標準出力へログするのみ。"
                           "専用の失敗ログ・通知は無い",
        },
        "state_distinguishability": {
            "completed": "kekka 確定着順 > 0 で判定可能",
            "non_completed": "kekka 確定着順 = 0/空 で判定可能だが "
                             "**DNF(止) と 除外 は分離できない**",
            "scratched": "bunseki(card) にあり kekka(結果) に無い行として "
                         "**候補**を検出できる",
            "day_changes_route": "site/data/changes_{date}.json は 8 日すべて "
                                 "races={} で空。現時点で取消の独立ソースに "
                                 "ならない",
        },
    }
    with open(OUT / "phase_a_bunseki_audit.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    log(f"\n保存: {OUT / 'phase_a_bunseki_audit.json'}")


if __name__ == "__main__":
    main()
