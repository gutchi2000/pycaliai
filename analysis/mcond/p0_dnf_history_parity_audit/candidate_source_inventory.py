# -*- coding: utf-8 -*-
"""
candidate_source_inventory.py
=============================
欠落集合(out/missing_set_manifest.json)を復元しうる候補データソースを
リポジトリ内から全探索し、ファイル名ではなく **実際の列・期間・粒度・
更新時刻** を確認する。

READ-ONLY。ファイルは読むだけで一切書き換えない。
出力: out/candidate_source_inventory.json
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

EXT = Path(r"E:\競馬過去走データ")

# 必須列の判定キーワード (日本語/英語両方)
REQUIRED = {
    "race_id":        ["レースid", "race_id", "rid", "レースｉｄ"],
    "race_date":      ["日付", "年月日", "date", "開催日"],
    "ped_id":         ["血統登録番号", "ped_id", "血統登録", "馬id", "horse_id"],
    "finish_status":  ["着順", "確定着順", "pos", "finish", "結果"],
    "surface":        ["芝・ダ", "芝ダ", "surface", "トラック", "芝・ダート"],
    "distance":       ["距離", "dist"],
    "venue":          ["場所", "競馬場", "place", "venue", "開催場"],
    "jockey":         ["騎手", "jockey"],
    "trainer":        ["調教師", "厩舎", "trainer"],
    "sire":           ["父", "種牡馬", "sire"],
    "race_class":     ["クラス", "条件", "class", "grade", "レース名"],
    "umaban":         ["馬番", "umaban"],
    "horse_name":     ["馬名", "name", "horse"],
}


def log(m):
    print(m, flush=True)


def match_required(cols: list[str]) -> dict:
    low = [str(c).lower().replace(" ", "") for c in cols]
    out = {}
    for key, kws in REQUIRED.items():
        hit = None
        for c_orig, c in zip(cols, low):
            for kw in kws:
                if kw in c:
                    hit = str(c_orig)
                    break
            if hit:
                break
        out[key] = hit
    return out


def peek_csv(p: Path, max_rows=300) -> dict | None:
    for enc in ("cp932", "utf-8-sig", "utf-8"):
        try:
            df = pd.read_csv(p, encoding=enc, dtype=str, nrows=max_rows,
                             on_bad_lines="skip")
            if df.shape[1] <= 1:
                continue
            return {"cols": list(df.columns), "encoding": enc,
                    "sample_shape": list(df.shape)}
        except Exception:
            continue
    # ヘッダ無しの可能性
    for enc in ("cp932", "utf-8-sig"):
        try:
            df = pd.read_csv(p, encoding=enc, dtype=str, nrows=5,
                             header=None, on_bad_lines="skip")
            if df.shape[1] > 1:
                return {"cols": [f"<noheader:{df.shape[1]}cols>"],
                        "encoding": enc, "sample_shape": list(df.shape),
                        "headerless": True,
                        "first_row": df.iloc[0].tolist()[:20]}
        except Exception:
            continue
    return None


def peek_parquet(p: Path) -> dict | None:
    try:
        import pyarrow.parquet as pq
        f = pq.ParquetFile(p)
        return {"cols": list(f.schema.names), "n_rows": f.metadata.num_rows}
    except Exception as e:
        return {"error": str(e)[:200]}


def peek_json(p: Path) -> dict | None:
    try:
        if p.stat().st_size > 60_000_000:
            return {"skipped": "too large", "size_mb": round(p.stat().st_size / 1e6, 1)}
        with open(p, encoding="utf-8") as f:
            j = json.load(f)
    except Exception as e:
        return {"error": str(e)[:200]}
    if isinstance(j, dict):
        keys = list(j.keys())[:25]
        # ネストの中に race 配列がある形を探る
        probe = {}
        for k in keys[:25]:
            v = j[k]
            if isinstance(v, list) and v and isinstance(v[0], dict):
                probe[k] = list(v[0].keys())[:30]
            elif isinstance(v, dict):
                probe[k] = f"<dict:{list(v.keys())[:8]}>"
        return {"top_keys": keys, "list_of_dict_fields": probe}
    if isinstance(j, list) and j and isinstance(j[0], dict):
        return {"root": "list", "fields": list(j[0].keys())[:30], "n": len(j)}
    return {"type": str(type(j))}


def scan(path: Path, label: str, limit_files=6) -> list[dict]:
    """ディレクトリまたは単一ファイルを検査する。"""
    entries: list[dict] = []
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = []
        for pat in ("*.csv", "*.parquet", "*.json", "*.pkl"):
            files.extend(sorted(path.glob(pat))[-limit_files:])
        if not files:
            sub = [d for d in path.iterdir() if d.is_dir()][:3]
            for s in sub:
                for pat in ("*.csv", "*.parquet", "*.json"):
                    files.extend(sorted(s.glob(pat))[-2:])
    else:
        return [{"label": label, "path": str(path), "exists": False}]

    for p in files[: limit_files * 2]:
        try:
            st = p.stat()
        except Exception:
            continue
        e = {
            "label": label,
            "path": str(p.relative_to(BASE)) if str(p).startswith(str(BASE)) else str(p),
            "size_mb": round(st.st_size / 1e6, 3),
            "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
        }
        suf = p.suffix.lower()
        info = None
        if suf == ".csv":
            info = peek_csv(p)
        elif suf == ".parquet":
            info = peek_parquet(p)
        elif suf == ".json":
            info = peek_json(p)
        elif suf == ".pkl":
            info = {"note": "pickle (未展開)"}
        if info:
            e.update(info)
            if "cols" in info and info["cols"]:
                e["required_match"] = match_required(info["cols"])
                e["n_required_hit"] = sum(
                    1 for v in e["required_match"].values() if v)
        entries.append(e)
    return entries


def main():
    log("=" * 70)
    log("CANDIDATE SOURCE INVENTORY (read-only)")
    log("=" * 70)

    targets = [
        (BASE / "data" / "kako5", "kako5 CSV (TARGET 過去5走)"),
        (BASE / "data" / "bunseki", "TARGET 出走馬分析"),
        (BASE / "data" / "hosei", "補正タイム週次"),
        (BASE / "data" / "target_review", "TARGET review 出力"),
        (BASE / "data" / "uma_review", "馬 review 出力"),
        (BASE / "data" / "target_shisu", "TARGET 指数"),
        (BASE / "data" / "tyaku", "着度数"),
        (BASE / "data" / "odds", "オッズ保存物"),
        (BASE / "data" / "forward_prices", "forward collection 成果物"),
        (BASE / "data" / "Time _series_odds", "時系列オッズ"),
        (BASE / "data" / "win5", "WIN5"),
        (BASE / "data" / "training", "調教週次"),
        (BASE / "data" / "baba", "馬場"),
        (BASE / "data" / "bias", "バイアス"),
        (BASE / "data" / "aux", "aux"),
        (BASE / "data" / "_research", "research artifacts"),
        (BASE / "data" / "_forensic", "forensic artifacts"),
        (BASE / "data" / "archive", "archive"),
        (BASE / "data" / "_inbox", "inbox (未仕分け)"),
        (BASE / "data" / "騎手_調教師コード", "騎手/調教師コード"),
        (BASE / "data" / "走間分析", "走間分析"),
        (BASE / "data" / "タイム分析", "タイム分析"),
        (BASE / "data" / "pycali_history.parquet", "pycali_history parquet"),
        (BASE / "data" / "pycali_history.csv", "pycali_history csv"),
        (BASE / "data" / "_career_results.parquet", "career results parquet"),
        (BASE / "data" / "horse_pedigree.json", "horse pedigree json"),
        (BASE / "data" / "live_results_2026.csv", "live results 2026"),
        (BASE / "data" / "kekka_20160105_20251228_v2.csv", "kekka v2 master"),
        (BASE / "data" / "payout_table.parquet", "payout table"),
        (BASE / "data" / "results.json", "results.json"),
        (BASE / "data" / "cowork_results.json", "cowork results"),
        (BASE / "reports" / "cowork_input", "bundle JSON (cowork_input)"),
        (BASE / "reports" / "cowork_output", "cowork output"),
        (BASE / "reports" / "forward_prices", "reports/forward_prices"),
        (BASE / "reports" / "site_odds", "reports/site_odds"),
        (BASE / "site" / "data", "静的サイト data"),
        (EXT, "外部: 競馬過去走データ (root)"),
        (EXT / "2026", "外部: 2026 レース別"),
    ]

    all_entries: list[dict] = []
    for p, label in targets:
        try:
            es = scan(p, label)
        except Exception as ex:
            es = [{"label": label, "path": str(p), "error": str(ex)[:200]}]
        all_entries.extend(es)
        best = max((e.get("n_required_hit", 0) for e in es), default=0)
        log(f"  {label:38s} files={len(es):2d}  best_required_hit={best}/13")

    # 必須列ヒット数の高い順に要約
    ranked = sorted([e for e in all_entries if e.get("n_required_hit")],
                    key=lambda e: -e["n_required_hit"])[:40]

    log("\n--- 必須列ヒット数 上位 ---")
    for e in ranked[:25]:
        hits = [k for k, v in e.get("required_match", {}).items() if v]
        log(f"  {e['n_required_hit']:2d}/13  {e['path']}")
        log(f"          {','.join(hits)}")

    with open(OUT / "candidate_source_inventory.json", "w", encoding="utf-8") as f:
        json.dump({"generated_at": datetime.now().isoformat(),
                   "n_entries": len(all_entries),
                   "entries": all_entries,
                   "ranked_by_required_hit": ranked},
                  f, ensure_ascii=False, indent=2, default=str)
    log(f"\n保存: out/candidate_source_inventory.json ({len(all_entries)} entries)")


if __name__ == "__main__":
    main()
