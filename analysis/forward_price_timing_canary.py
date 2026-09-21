#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""forward_prices timing canary — 発走予定時刻との整合性を検査する (読み取り専用)。

2026-09-21、T-10の-5,228分異常 (2026090601020601, 2026-09-06発走のレースが
2026-09-10深夜に再取得され append-only store へ混入) の調査で追加。同種の異常が
T-20/vote(T-4)にも見つかった (いずれも同一の手動/デバッグ実行由来と推定、
札幌2回6日開催 2026090601020601-12 を対象に 2026-09-10〜11 頃に複数stageで
繰り返し取得されている)。

`data/forward_prices/` は append-only 不変ストアであり、異常レコードも含め
一切上書き・削除しない (docs/forward_price_protocol.md 「latest viewだけを残し、
観測履歴を上書きすること」禁止と同じ精神)。本モジュールは既存ファイルを
一切変更せず、各レコードの `minutes_to_post = scheduled_post - observed_at` を
stage別の想定ウィンドウと比較して `timing_valid` を判定し、別ファイルへ
隔離レポートとして書き出すだけの分析ツールである。

将来の価格形成モデル研究がこのストアを学習データとして使う際は、
`load_timing_valid_records()` で timing_valid=True のみを取り込むこと
(NEXT_RESEARCH_START_CONDITIONS_20260921.md §1.3 の要求に対応)。

実行:
  python -m analysis.forward_price_timing_canary --scan
      → 全stage全件を走査しreports/forward_price_timing_canary.jsonへ書き出す
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

JST = timezone(timedelta(hours=9))

from forward_prices import FORWARD_ROOT  # noqa: E402

REPORT_PATH = BASE / "reports" / "forward_price_timing_canary.json"

# stage別の想定ウィンドウ (分、scheduled_post - observed_at)。実運用の実測分布
# (2026-08-29〜2026-09-21) から: t10 median=9.98/範囲9.96-10.00、close は常に-1.00、
# t20/voteは仕様上の目標値 (t20_site_bets.py: 発走20分前 / masters_vote: 発走4分前)
# に運用ジッター許容幅を持たせた。exp05fs_t35 は market_snapshot.py の
# WINDOW_MIN=(31,38) より広く取り、ここでは「明白な異常 (別日混入等)」だけを
# 粗く捕える用途に限定する (primary判定の正確な31-38分窓判定はmarket_snapshot.py
# 側の valid_for_primary が既に担っている、二重の厳密化はしない)。
STAGE_WINDOWS_MIN: dict[str, tuple[float, float]] = {
    "t10": (0.0, 25.0),
    "close": (-15.0, 2.0),
    "t20": (5.0, 45.0),
    "vote": (0.0, 20.0),
    "exp05fs_t35": (15.0, 55.0),
    "manual": (float("-inf"), float("inf")),  # 手動実行は目標窓を定義しない
}


def minutes_to_post(scheduled_post: str | None, observed_at: str | None) -> float | None:
    if not scheduled_post or not observed_at:
        return None
    try:
        sp = datetime.fromisoformat(str(scheduled_post))
        oa = datetime.fromisoformat(str(observed_at))
    except Exception:
        return None
    # 記録によりtz付き/tz無しが混在する (JST前提の裸timestampが大半)。素朴な
    # naive-aware比較エラーを避けるため、tz無しはJSTとして扱い統一する。
    if sp.tzinfo is None:
        sp = sp.replace(tzinfo=JST)
    if oa.tzinfo is None:
        oa = oa.replace(tzinfo=JST)
    return (sp - oa).total_seconds() / 60.0


def classify_timing(stage: str, scheduled_post: str | None, observed_at: str | None
                     ) -> tuple[bool, str, float | None]:
    """(timing_valid, reason, minutes_to_post) を返す。stage未知/scheduled_post欠落は
    「判定不能」を意味し timing_valid=False・理由を明示する (fail-closed: 不明を
    有効とみなさない)。"""
    m = minutes_to_post(scheduled_post, observed_at)
    if scheduled_post is None:
        return False, "scheduled_post欠落のため判定不能", None
    if m is None:
        return False, "observed_atまたはscheduled_postのparse失敗", None
    window = STAGE_WINDOWS_MIN.get(stage)
    if window is None:
        return False, f"未知のstage: {stage}", m
    lo, hi = window
    if lo <= m <= hi:
        return True, "", m
    return False, f"想定ウィンドウ外 (stage={stage} 想定[{lo},{hi}]分 実測{m:.2f}分)", m


def _iter_records(root: Path = FORWARD_ROOT):
    for p in sorted(root.rglob("*.json.gz")):
        try:
            with gzip.open(p, "rt", encoding="utf-8") as f:
                d = json.load(f)
        except Exception as exc:
            yield p, None, f"読込失敗: {exc}"
            continue
        yield p, d, None


def scan(root: Path = FORWARD_ROOT) -> dict:
    """data/forward_prices/ 配下 (market_snapshotレコードのみ、decisionは対象外)
    を全走査し、stage別・日付別のtiming_valid集計と異常レコード一覧を返す。
    一切のファイル変更を行わない (読み取り専用)。"""
    by_stage: dict[str, dict] = {}
    anomalies: list[dict] = []
    dup_check: dict[tuple[str, str], list[str]] = {}
    read_errors: list[str] = []

    for p, d, err in _iter_records(root):
        if err is not None:
            read_errors.append(f"{p}: {err}")
            continue
        if d.get("record_type") != "market_snapshot":
            continue
        stage = d.get("stage", "?")
        rid = d.get("race_id", "?")
        sp = d.get("scheduled_post")
        oa = d.get("observed_at")
        valid, reason, m = classify_timing(stage, sp, oa)

        st = by_stage.setdefault(stage, {"n": 0, "timing_valid": 0, "timing_invalid": 0})
        st["n"] += 1
        st["timing_valid" if valid else "timing_invalid"] += 1
        if not valid:
            anomalies.append({
                "file": str(p.relative_to(root)), "stage": stage, "race_id": rid,
                "scheduled_post": sp, "observed_at": oa,
                "minutes_to_post": round(m, 2) if m is not None else None,
                "reason": reason,
            })
        dup_check.setdefault((stage, rid), []).append(str(p.relative_to(root)))

    duplicates = [{"stage": k[0], "race_id": k[1], "files": v}
                  for k, v in sorted(dup_check.items()) if len(v) > 1]

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "root": str(root),
        "stage_windows_min": STAGE_WINDOWS_MIN,
        "by_stage": by_stage,
        "anomaly_count": len(anomalies),
        "anomalies": anomalies,
        "duplicate_race_stage_count": len(duplicates),
        "duplicates": duplicates,
        "read_errors": read_errors,
    }


_GATE_MARKER = "_timing_gate_passed"


def load_timing_valid_records(stage: str, root: Path = FORWARD_ROOT):
    """将来の価格形成研究向け: 指定stageの timing_valid=True レコードだけを
    読み込んで返す (学習データからwindow外レコードを除外するための唯一の入口として
    使うことを想定。フィルタを各研究スクリプトで再実装しないこと)。

    返す各dictにはこの関数を通過した印 (`_timing_gate_passed=True`) を刻む。
    `require_timing_gate()` と組み合わせ、将来の価格形成研究の入力コードが
    `data/forward_prices/` を直接globして (=このgateを経由せず) window外レコードを
    紛れ込ませていないことを、実行時に強制できるようにする (2026-09-22追加、
    ユーザー指摘対応: 「推奨関数」であるだけでは将来のコードが直接読むのを防げない
    という指摘に対する、テストで検証可能な明示的Gate)。"""
    out = []
    for p, d, err in _iter_records(root):
        if err is not None or d is None:
            continue
        if d.get("record_type") != "market_snapshot" or d.get("stage") != stage:
            continue
        valid, _reason, _m = classify_timing(stage, d.get("scheduled_post"), d.get("observed_at"))
        if valid:
            stamped = dict(d)
            stamped[_GATE_MARKER] = True
            out.append(stamped)
    return out


def require_timing_gate(records: list[dict]) -> None:
    """価格形成研究の学習データ組成コードは、モデルへ渡す直前に必ずこれを呼ぶこと。

    `load_timing_valid_records()` を経由していない (=生の `data/forward_prices/`
    をglob等で直接読んでwindow外レコードが混じっている可能性がある) レコードが
    1件でも見つかれば ValueError で止める。「推奨関数として提供する」だけでは
    将来のコードがそれを無視して直接読むのを防げない、というユーザー指摘への
    対応 (2026-09-22)。価格形成研究の入力パイプラインは、このgateを通過しない
    限りモデルへデータを渡せない設計にすること。"""
    for i, r in enumerate(records):
        if not r.get(_GATE_MARKER):
            raise ValueError(
                f"timing gate 未通過のレコードが混入 (index={i}, "
                f"race_id={r.get('race_id')!r}). load_timing_valid_records() を "
                f"経由せず data/forward_prices/ を直接読んでいないか確認すること。"
                f"window外(timing_valid=false)レコードが価格形成モデルの学習データへ"
                f"混入する事故を防ぐための必須gateです。")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="store_true")
    args = ap.parse_args()
    if not args.scan:
        print("使い方: --scan")
        return 1
    report = scan()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[timing_canary] {REPORT_PATH} "
          f"(異常 {report['anomaly_count']} 件 / 重複race×stage {report['duplicate_race_stage_count']} 件)")
    for stage, st in sorted(report["by_stage"].items()):
        print(f"  {stage}: n={st['n']} valid={st['timing_valid']} invalid={st['timing_invalid']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
