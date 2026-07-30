# -*- coding: utf-8 -*-
"""
publish_hosei.py — T-15 補正印(オッズblend)のサイト反映 (64-bit)
==================================================================
t15.ps1 から呼ばれる。jvlink_odds.py (32-bit) が書いた reports/live_odds/{rid}.json
の単勝オッズと bundle の p_win を blend し、site/data/changes_{date}.json の
races[rid].hosei に補正印を書き込む (changes オーバーレイと同じ配管で配信)。

blend は compute_bets.hosei_marks と同式: u = log(p_win) + λ·log(de-vig π)。
λ = data/t10_blend.json。精度実測 (analysis/eval_t15_blend.py, test2024-25 6,858R):
  ◎top3 = 65.2〜65.5% (発走35分前スナップ65.21% ≤ T-15 ≤ 確定65.79% の挟み撃ち)。

実行: python publish_hosei.py --date 20260801 --race 2026080101010101
"""
from __future__ import annotations
import argparse, json, math
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).parent
HOSEI_MARK5 = ["◎", "〇", "▲", "△", "△"]


def _num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def blend_lambda() -> float | None:
    try:
        return float(json.loads(
            (BASE / "data/t10_blend.json").read_text(encoding="utf-8"))["lambda"])
    except Exception:
        return None


def hosei_for_race(date: str, rid: str) -> list[dict] | None:
    lam = blend_lambda()
    if lam is None:
        return None
    lp = BASE / "reports" / "live_odds" / f"{rid}.json"
    if not lp.exists():
        return None
    live = json.loads(lp.read_text(encoding="utf-8"))
    if not live.get("ok") or not live.get("tansho"):
        return None          # overround 異常等はオッズ側 fail-safe に従い出さない
    tansho = {int(k): float(v) for k, v in live["tansho"].items() if _num(v)}

    bp = BASE / "reports" / "cowork_input" / f"{date}_bundle.json"
    if not bp.exists():
        return None
    import re
    bundle = json.loads(bp.read_text(encoding="utf-8"))
    race = next((r for r in bundle.get("races", [])
                 if re.sub(r"\D", "", str(r.get("race_id", "")))[:16] == rid), None)
    if race is None:
        return None

    rows = []
    for h in race.get("horses", []):
        b, p = _num(h.get("umaban")), _num(h.get("p_win"))
        o = tansho.get(int(b)) if b is not None else None
        if b is None or p is None or o is None or p <= 0 or o <= 0:
            continue
        rows.append((int(b), p, 1.0 / o, h.get("horse_name") or "", h.get("mark") or ""))
    if len(rows) < 5:
        return None
    s_inv = sum(r[2] for r in rows)
    rows.sort(key=lambda r: -(math.log(r[1]) + lam * math.log(r[2] / s_inv)))
    return [{"mark": HOSEI_MARK5[i], "umaban": r[0], "name": r[3],
             "chg": r[4] != HOSEI_MARK5[i]} for i, r in enumerate(rows[:5])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--race", required=True, help="rid16")
    args = ap.parse_args()

    marks = hosei_for_race(args.date, args.race)
    if not marks:
        print(f"[publish_hosei] {args.race}: 補正印なし (オッズ未取得/λ無し/5頭未満)")
        return

    sp = BASE / "site" / "data" / f"changes_{args.date}.json"
    if sp.exists():
        ch = json.loads(sp.read_text(encoding="utf-8"))
    else:
        ch = {"date": args.date, "races": {}}
    ch["fetched"] = datetime.now().isoformat(timespec="seconds")
    entry = ch.setdefault("races", {}).setdefault(args.race, {})
    entry["hosei"] = {"asof": datetime.now().strftime("%H:%M"), "marks": marks}
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps(ch, ensure_ascii=False), encoding="utf-8")
    line = " ".join(f"{m['mark']}{m['umaban']}{m['name']}{'*' if m['chg'] else ''}"
                    for m in marks)
    print(f"[publish_hosei] {args.race}: {line}")


if __name__ == "__main__":
    main()
