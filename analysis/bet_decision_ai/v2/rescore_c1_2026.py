# -*- coding: utf-8 -*-
"""
rescore_c1_2026.py — 2026-01〜08-22 の C1 生スコアを「当時の入力」から作り直す（研究用）
=======================================================================================
仕様: docs/research/BET_DECISION_AI_V2_SPEC.md §2（2026 年 1〜8 月の C1 スコアは保存されていない）

方式: analysis/replay_serve_2026.py と同じく、本番の export_weekly_marks.py を子プロセスで
そのまま呼び（＝本番と同じ特徴生成・同じ C1 モデル）、出力先だけを data/_research 配下にする。
  - モデル: models/unified_rank_v6.pkl（export_weekly_marks が data/artifact_pins.json と sha256 照合。
    C1 = db2ffe65… でなければ子プロセスが exit 1 で止まる）
  - 入力: data/weekly/{date}.csv（当時の出走表）。補正タイム推定 CB_HOSEI_PROXY=1（本番の週次フローの設定）
  - 馬履歴 data/_horse_history.parquet は開催日より前の行だけに物理的に切ってから使う（as-of）
  - eligibility coverage gate（過去日は証拠ファイルが無い）は素通し。障害判定はこのスクリプトでは
    しない（全レースを採点し、障害除外は build_substrate_v2 が行う）
  - 品質ゲート不合格の日も作業ディレクトリへ書き出す（不合格はログで記録し substrate に日単位のフラグを残す）
  - 生スコアは丸め前の値を ai_score_full として bundle に追加で書く（horse_record を包むだけ。計算は同じ）
本番の bundle・reports・models・data（data/_research 以外）には書かない。

出力（すべて data/_research/bet_decision_ai/v2/c1_2026/ 配下）
  {date}/                    レース単位 JSON（export_weekly_marks の通常出力）
  {date}_bundle.json         bundle
  {date}.log                 子プロセスのログ
  ../c1_2026_scores.parquet  rid16, ban, date, score（全日の結合。build_substrate_v2 が読む）
  ../c1_2026_manifest.json   日ごとの rc・品質ゲート・欠損特徴の記録、モデル sha256

実行: python -m analysis.bet_decision_ai.v2.rescore_c1_2026 [--since 20260101] [--until 20260822] [--force]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
WORK = BASE / "data" / "_research" / "bet_decision_ai" / "v2" / "c1_2026"
SCORES_OUT = WORK.parent / "c1_2026_scores.parquet"
MANIFEST = WORK.parent / "c1_2026_manifest.json"
O1_RAW = Path(r"E:\競馬過去走データ\TAN_PUKU_WAKU_20210105-20260922.txt")
MODEL = BASE / "models" / "unified_rank_v6.pkl"
PINS = BASE / "data" / "artifact_pins.json"

_CHILD = r"""
import sys, runpy
sys.path.insert(0, r'{base}')
import eligibility_coverage_gate as g
class _Pass:
    ok = True; expected = 0; flat_rids = []; jump_rids = []
    def as_dict(self): return {{"research_rescore_c1_2026": True}}
g.check_coverage = lambda *a, **k: _Pass()
g.AtomicPublish.abort = g.AtomicPublish.commit      # 品質ゲート不合格でも作業ディレクトリへは書き出す
import race_eligibility as re_
_orig = re_.evaluate_race
def _ev(rid, *a, **k):
    el = dict(_orig(rid, *a, **k))
    el['prediction_eligible'] = True                  # 全レース採点 (障害除外は substrate 側)
    return el
re_.evaluate_race = _ev
import export_marks_json as emj
_hr = emj.horse_record
def _hr2(g_row, ban, mark, ai_rank, ai_score, *a, **k):
    rec = _hr(g_row, ban, mark, ai_rank, ai_score, *a, **k)
    rec['ai_score_full'] = float(ai_score)           # 丸め前の生スコア (計算は本番と同じ)
    return rec
emj.horse_record = _hr2
import pandas as _pd
_rp = _pd.read_parquet
def _read(path, *a, **k):
    df = _rp(path, *a, **k)
    if str(path).endswith('_horse_history.parquet'):
        df = df[df['date'].astype(int) < {date}].copy()      # 開催日より後の行を物理的に落とす (as-of)
    return df
_pd.read_parquet = _read
sys.argv = ['export_weekly_marks.py', '--csv', r'{csv}', '--shap-topk', '0', '--out-dir', r'{out}']
runpy.run_path(r'{base}\export_weekly_marks.py', run_name='__main__')
"""


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 22), b""):
            h.update(ch)
    return h.hexdigest()


def o1_dates(since: str, until: str) -> list[str]:
    """O1 原本に載っている 2026 年の開催日（レース ID 先頭 8 桁）。原本は読み取りのみ。"""
    ds = set()
    with open(O1_RAW, "r", encoding="cp932", newline="") as fh:
        for line in fh:
            if line[:2] == "O1" and len(line.rstrip("\r\n")) == 960:
                d = line[11:19]
                if d[:4] == "2026" and since <= d <= until:
                    ds.add(d)
    return sorted(ds)


def parse_log(txt: str) -> dict:
    out = {"gate_fail": "品質ゲート不合格" in txt, "pins_ok": "artifact pins OK" in txt}
    m = re.search(r"週次CSVに不足の (\d+) 列を補完", txt)
    out["n_missing_cols_filled"] = int(m.group(1)) if m else 0
    m = re.search(r"\[serve history\] 解決 (\d+)/(\d+) 頭", txt)
    out["history_resolved"] = [int(m.group(1)), int(m.group(2))] if m else None
    m = re.search(r"\[done\] ([\d,]+) JSON \((\d+) skip\)", txt)
    out["n_race_json"] = int(m.group(1).replace(",", "")) if m else None
    out["n_skip"] = int(m.group(2)) if m else None
    m = re.search(r"\[serve canary absolute\] dead gain=([\d.]+)% \((\d+)特徴", txt)
    out["dead_gain_pct"] = float(m.group(1)) if m else None
    out["n_dead_feats"] = int(m.group(2)) if m else None
    m = re.search(r"\[serve canary\] 監視 (\d+) 特徴 / 無言死 (\d+)", txt)
    out["n_silent_dead"] = int(m.group(2)) if m else None
    out["kako5_missing"] = "kako5 CSV なし" in txt
    m = re.search(r"パース完了（障害除外済）: (\d+)頭 / (\d+)レース", txt)
    out["parsed_races"] = int(m.group(2)) if m else None
    out["gate_lines"] = [ln.strip()[:300] for ln in txt.splitlines()
                         if ("ゲート" in ln or "gate" in ln.lower()) and ("ERROR" in ln or "WARNING" in ln)][:12]
    return out


def run(since: str, until: str, force: bool) -> dict:
    WORK.mkdir(parents=True, exist_ok=True)
    pins = json.loads(PINS.read_text(encoding="utf-8"))
    model_sha = sha256(MODEL)
    want = pins["pins"]["models/unified_rank_v6.pkl"]
    if model_sha != want:
        raise SystemExit(f"model sha {model_sha} != pinned C1 {want}")
    days = o1_dates(since, until)
    rec: dict = {}
    for d in days:
        csv = BASE / "data" / "weekly" / f"{d}.csv"
        bpath = WORK / f"{d}_bundle.json"
        lpath = WORK / f"{d}.log"
        if not csv.exists():
            rec[d] = {"rc": None, "status": "no_weekly_csv"}
            print(f"{d} no weekly csv", flush=True)
            continue
        if bpath.exists() and lpath.exists() and not force:
            info = parse_log(lpath.read_text(encoding="utf-8", errors="replace"))
            rec[d] = {"rc": 0, "status": "cached", **info}
            continue
        code = _CHILD.format(base=str(BASE), csv=str(csv), out=str(WORK / d), date=int(d))
        env = dict(os.environ, CB_HOSEI_PROXY="1", PYTHONIOENCODING="utf-8")
        t0 = datetime.now()
        r = subprocess.run([sys.executable, "-c", code], cwd=str(BASE), capture_output=True, env=env,
                           text=True, encoding="utf-8", errors="replace")
        txt = (r.stdout or "") + (r.stderr or "")
        lpath.write_text(txt, encoding="utf-8")
        info = parse_log(txt)
        rec[d] = {"rc": r.returncode, "status": "ok" if bpath.exists() else "no_bundle",
                  "sec": round((datetime.now() - t0).total_seconds(), 1), "csv_sha256": sha256(csv), **info}
        print(f"{d} rc={r.returncode} bundle={'yes' if bpath.exists() else 'NO'} "
              f"gate_fail={info['gate_fail']} races={info['n_race_json']} {rec[d]['sec']}s", flush=True)

    rows = []
    for d in days:
        bpath = WORK / f"{d}_bundle.json"
        if not bpath.exists():
            continue
        b = json.loads(bpath.read_text(encoding="utf-8"))
        rs = b["races"] if isinstance(b["races"], list) else list(b["races"].values())
        for r in rs:
            rid = re.sub(r"\D", "", str(r.get("race_id", "")))[:16]
            for h in r.get("horses", []):
                sc = h.get("ai_score_full", h.get("ai_score"))
                rows.append({"rid16": rid, "ban": int(h["umaban"]), "date": d, "score": float(sc),
                             "score_full_precision": "ai_score_full" in h,
                             "p_win_serve_cal": h.get("p_win")})
    sc = pd.DataFrame(rows)
    sc.to_parquet(SCORES_OUT, index=False)
    man = {"generated_at": datetime.now().isoformat(timespec="seconds"), "model_sha256": model_sha,
           "pins_generation": pins.get("generation"), "since": since, "until": until,
           "env": {"CB_HOSEI_PROXY": "1"}, "horse_history_truncated_lt_date": True,
           "n_days_o1": len(days), "n_rows": int(len(sc)),
           "n_races": int(sc["rid16"].nunique()) if len(sc) else 0, "days": rec}
    MANIFEST.write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {len(sc):,} rows / {man['n_races']:,} races -> {SCORES_OUT}")
    return man


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="20260101")
    ap.add_argument("--until", default="20260822")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    run(a.since, a.until, a.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
