# -*- coding: utf-8 -*-
"""
report_card_miokuri.py — 見送り判断の成績（買っていたら？）
=============================================================
build_report_card.py から呼ばれ、report_card.json の "miokuri" ブロックを返す。

本線（a95）は参戦ガードで見送るレースにも、T-10 オッズで「買っていたら」の買い目を
reports/cowork_output/{date}_bets.json の各レース a95_all（guard_passed=false）に記録している。
ここでは results のあるレースを 参戦（guard_passed=true＝実際にサイトに表示した買い目）と
見送り（guard_passed=false＝表示していない仮想買い目）に分け、レース的中率（Wilson 95% CI）と
回収率（レース単位クラスタブートストラップ 95% CI）を集計する。

出力は集計値のみ。1R ごとのオッズ・払戻・買い目は一切出さない（公開サイト用）。
決済ロジックは analysis/a95_all_t10_report.py と同じ（shadow_a95.load_kekka / payout_per_100 を再利用）。
kekka が無い開催日は飛ばして skipped_dates に記録する。
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import shadow_a95 as sa  # noqa: E402  (root モジュール: kekka 読込と払戻計算)

COWORK_OUT = ROOT / "reports" / "cowork_output"
A95_SINCE = "20261024"      # a95 の再起算日（[[project_prod_model_c0_regression_20260920]]）
MIN_N_VERDICT = 20          # これ未満なら判定文を出さない
BOOT_REPS = 4000


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def collect(since: str = A95_SINCE) -> tuple[list[dict], list[str], list[str], int]:
    """rows(1 行 = 1 レース), used_dates, skipped_dates, n_excluded_races"""
    wide = sa.load_wide()
    rows: list[dict] = []
    used: list[str] = []
    skipped: list[str] = []
    excluded = 0
    for p in sorted(COWORK_OUT.glob("2026????_bets.json")):
        date = p.name[:8]
        if date < since:
            continue
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            skipped.append(date)
            continue
        entries = raw["bets"] if isinstance(raw, dict) and "bets" in raw else raw
        entries = [e for e in entries if isinstance(e, dict) and (e.get("a95_all") or {}).get("tickets")]
        if not entries:
            continue                      # a95_all 未記録の日は対象外（skip でもない）
        kek = sa.load_kekka(date, wide)
        if not kek:
            skipped.append(date)          # 結果未取込
            continue
        day_used = False
        for e in entries:
            aa = e["a95_all"]
            rid = re.sub(r"\D", "", str(e.get("race_id", "")))[:16]
            k = kek.get(rid)
            if not k or not k.get("valid"):
                excluded += 1
                continue
            d = {x: aa.get(x) for x in ("a1", "a2", "a3", "waku1", "waku2")}
            try:
                per = {t["kind"]: sa.payout_per_100(t["kind"], d, k) for t in aa["tickets"]}
            except (KeyError, ValueError):
                excluded += 1
                continue
            if any(not np.isfinite(v) for v in per.values()):
                excluded += 1             # ワイド払戻未取込など
                continue
            stake = float(sum(int(t["stake"]) for t in aa["tickets"]))
            ret = float(sum(per[t["kind"]] / 100.0 * int(t["stake"]) for t in aa["tickets"]))
            rows.append(dict(date=date, rid=rid, guard=bool(aa.get("guard_passed")),
                             stake=stake, ret=ret, hit=any(v > 0 for v in per.values())))
            day_used = True
        if day_used:
            used.append(date)
    return rows, used, skipped, excluded


def roi_boot_ci(rows: list[dict], reps: int = BOOT_REPS, seed: int = 0) -> tuple[float | None, float | None]:
    """レース単位（1R の全券種をひとかたまり）のクラスタブートストラップ 95% CI。"""
    n = len(rows)
    if n < 5:
        return (None, None)
    c = np.array([r["stake"] for r in rows], float)
    v = np.array([r["ret"] for r in rows], float)
    idx = np.random.default_rng(seed).integers(0, n, size=(reps, n))
    roi = 100.0 * v[idx].sum(1) / c[idx].sum(1)
    return (round(float(np.percentile(roi, 2.5)), 1), round(float(np.percentile(roi, 97.5)), 1))


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    hit = sum(1 for r in rows if r["hit"])
    stake = sum(r["stake"] for r in rows)
    ret = sum(r["ret"] for r in rows)
    hlo, hhi = wilson(hit, n)
    rlo, rhi = roi_boot_ci(rows)
    return {
        "n_races": n,
        "n_days": len({r["date"] for r in rows}),
        "hit": hit,
        "hit_rate": round(hit / n * 100, 1) if n else None,
        "hit_lo": round(hlo * 100, 1), "hit_hi": round(hhi * 100, 1),
        "stake": int(stake),
        "roi": round(ret / stake * 100, 1) if stake else None,
        "roi_lo": rlo, "roi_hi": rhi,
        "plus_rate": round(sum(1 for r in rows if r["ret"] >= r["stake"]) / n * 100, 1) if n else None,
    }


def verdict(sanse: dict, miokuri: dict) -> dict:
    """数字から機械的に一文を作る。code は JS 側の色分け用。"""
    if miokuri["n_races"] < MIN_N_VERDICT or sanse["n_races"] < MIN_N_VERDICT or miokuri["roi"] is None or sanse["roi"] is None:
        return {"code": "insufficient",
                "text": f"見送り {miokuri['n_races']}R・参戦 {sanse['n_races']}R ではまだ件数が少なく、見送り判断の良し悪しは判定できません。"}
    overlap = (miokuri["roi_lo"] is not None and sanse["roi_lo"] is not None
               and not (miokuri["roi_hi"] < sanse["roi_lo"] or sanse["roi_hi"] < miokuri["roi_lo"]))
    diff = round(sanse["roi"] - miokuri["roi"], 1)
    if miokuri["roi"] < sanse["roi"]:
        text = (f"見送ったレースを買っていた場合の回収率は {miokuri['roi']}% で、参戦したレースの {sanse['roi']}% より "
                f"{abs(diff)} ポイント低く、見送りは正しかった方向です。")
        code = "good"
    else:
        text = (f"見送ったレースを買っていた場合の回収率は {miokuri['roi']}% で、参戦したレースの {sanse['roi']}% を "
                f"{abs(diff)} ポイント上回っており、見送りで取り逃している可能性があります。")
        code = "leak"
    if overlap:
        text += "ただし両者の 95% 信頼区間は重なっており、偶然の範囲内で説明できる差です。"
        code += "_uncertain"
    return {"code": code, "text": text, "roi_diff": diff, "ci_overlap": overlap}


def build_miokuri(since: str = A95_SINCE) -> dict:
    rows, used, skipped, excluded = collect(since)
    sanse = summarize([r for r in rows if r["guard"]])
    miok = summarize([r for r in rows if not r["guard"]])
    both = summarize(rows)
    return {
        "since": since,
        "period": {"from": used[0] if used else None, "to": used[-1] if used else None, "n_days": len(used)},
        "skipped_dates": skipped,
        "n_excluded_races": excluded,
        "sanse": sanse,        # 参戦（ガード通過＝サイトに表示した買い目）
        "miokuri": miok,       # 見送り（ガードで見送り＝買っていたらの仮想買い目）
        "all": both,           # 参考: 全レース参戦した場合
        "verdict": verdict(sanse, miok),
        "note": ("参戦ガード（混戦度・頭数・◎の勝率・◎のオッズ・障害）で見送ったレースにも、発走 10 分前のオッズで同じ方式の買い目を"
                 "記録しておき、結果が出た後に「買っていたら」を集計したもの。回収率の区間はレース単位のブートストラップ 95%、"
                 "的中率の区間は Wilson 95%。表示専用・実弾ではない。払戻額・オッズそのものは掲載しない。"),
    }


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print(json.dumps(build_miokuri(), ensure_ascii=False, indent=1))
