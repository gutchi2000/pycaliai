# -*- coding: utf-8 -*-
"""
morning_hosei.py — 朝の補正印 (v7 の確率 × 朝 9 時の単勝オッズ) を私用 Discord とローカルに出す (2026-11-14)
=====================================================================================
T-10 補正印 (compute_bets.hosei_marks) と同じ式を、朝 9 時のオッズで全レースに 1 回だけ当てる。
    u = log(p_win) + λ·log(π)    p_win = bundle の較正済み単勝確率、π = 朝 9 時オッズの逆数をレース内で正規化
    λ = data/am9_blend.json (analysis/fit_am9_blend.py、v7・valid 2023 で fit)。無ければ data/t10_blend.json。
過去検証 (test 2024-25): ◎3着内 v7 62.30% → 65.24% (+2.94pt [+2.00,+3.90])、朝の市場単独 64.40%。
T-10 補正印の上げ幅が、朝の時点でほぼ取れる (docs/research/LATE_MONEY_DECOMP_20261114.md)。

出し先は私用だけ:
  - reports/morning_hosei/{date}.json  (git 管理外)
  - 私用 Discord (notify_config.json の discord_webhook)。--discord を付けたときだけ送る。
**公開サイト・公開 Discord・bundle・買い目には書かない。** 朝 9 時のオッズは JV-Link 経由で、
JRA-VAN 投稿ガイドライン (JV-Link から取得したデータは投稿不可) により、オッズを混ぜた印のサイト掲載は 2026-07-31 に停止している。
表示専用。参戦判定・買い目・公開印には影響しない。

入力: reports/cowork_input/{date}_bundle.json、reports/am9_odds/{date}/{rid}.json (scripts/am9_snapshot.py)
実行: python scripts/morning_hosei.py [--date YYYYMMDD] [--discord]
      am9_snapshot.py の最後から自動で呼ばれる (失敗しても収集には影響しない)。
      データが別の場所にあるとき (worktree からの確認など) は環境変数 PYCALIAI_DATA_ROOT=E:/PyCaLiAI。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import sys
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("PYCALIAI_DATA_ROOT", BASE))
MARK5 = ["◎", "〇", "▲", "△", "△"]
MIN_COVER = 0.8          # 朝オッズのある馬がこの割合未満のレースは出さない


def _num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def blend_lambda() -> tuple[float | None, str]:
    for name in ("am9_blend.json", "t10_blend.json"):
        for base in (BASE, ROOT):
            try:
                return float(json.loads((base / "data" / name).read_text(encoding="utf-8"))["lambda"]), name
            except Exception:  # noqa: BLE001
                continue
    return None, ""


def marks_for_race(race: dict, tansho: dict[int, float], lam: float) -> list[dict] | None:
    """blend 順の上位 5 頭。朝オッズか p_win の無い馬は対象外。5 頭未満・カバー不足なら None。"""
    horses = race.get("horses", [])
    rows, n_odds = [], 0
    for h in horses:
        b, p = _num(h.get("umaban")), _num(h.get("p_win"))
        o = tansho.get(int(b)) if b is not None else None
        if b is None or o is None or o <= 0:
            continue
        n_odds += 1
        if p is None or p <= 0:          # bundle の p_win は小数 4 桁。0.0000 の大穴は順位付けの対象外 (T-10 補正印と同じ)
            continue
        rows.append((int(b), p, 1.0 / o, h.get("horse_name") or "", h.get("mark") or ""))
    if len(rows) < 5 or n_odds < MIN_COVER * len(horses):
        return None
    s_inv = sum(r[2] for r in rows)
    rows.sort(key=lambda r: -(math.log(r[1]) + lam * math.log(r[2] / s_inv)))
    return [{"mark": MARK5[i], "umaban": r[0], "horse_name": r[3], "orig_mark": r[4]} for i, r in enumerate(rows[:5])]


def build(date: str) -> dict:
    lam, src = blend_lambda()
    if lam is None:
        raise SystemExit("[morning_hosei] 混合係数が無い (data/am9_blend.json / t10_blend.json)")
    bp = ROOT / "reports" / "cowork_input" / f"{date}_bundle.json"
    if not bp.exists():
        raise SystemExit(f"[morning_hosei] bundle が無い: {bp}")
    bundle = json.loads(bp.read_text(encoding="utf-8"))
    odir = ROOT / "reports" / "am9_odds" / date
    out = {"date": date, "generated": dt.datetime.now().isoformat(timespec="seconds"), "lambda": lam, "lambda_source": src,
           "model": bundle.get("model"), "races": {}, "skipped": {}}
    for race in bundle.get("races", []):
        rid = re.sub(r"\D", "", str(race.get("race_id", "")))[:16]
        op = odir / f"{rid}.json"
        if not op.exists():
            out["skipped"][rid] = "朝オッズ無し"
            continue
        try:
            live = json.loads(op.read_text(encoding="utf-8"))
        except ValueError:
            out["skipped"][rid] = "朝オッズ読めず"
            continue
        if not live.get("ok") or not live.get("tansho"):
            out["skipped"][rid] = "朝オッズ異常"          # overround 異常などは収集側の判定に従う
            continue
        tansho = {int(k): float(v) for k, v in live["tansho"].items() if _num(v)}
        marks = marks_for_race(race, tansho, lam)
        if not marks:
            out["skipped"][rid] = "頭数/カバー不足"
            continue
        meta = race.get("race_meta") or {}
        hon0 = next((h for h in race.get("horses", []) if h.get("mark") == "◎"), None)
        out["races"][rid] = {"place": meta.get("place", ""), "rno": int(rid[14:16]), "asof": str(live.get("fetched", ""))[11:16],
                             "orig_hon": hon0.get("umaban") if hon0 else None,
                             "hon_changed": bool(hon0 is None or hon0.get("umaban") != marks[0]["umaban"]), "marks": marks}
    return out


def fmt_lines(res: dict) -> list[str]:
    """会場ごとにまとめた表示行。元の印から変わった馬に * を付ける。オッズの値は出さない。"""
    races = sorted(res["races"].items(), key=lambda kv: (kv[1]["place"], kv[1]["rno"]))
    n, chg = len(races), sum(r["hon_changed"] for _, r in races)
    lines = [f"**🌅 朝の補正印 {res['date']}**（v7 × 朝 9 時オッズ、{n}R 中 ◎が変わったのは {chg}R。私用・表示専用）"]
    place = None
    for _, r in races:
        if r["place"] != place:
            place = r["place"]; lines.append(f"__{place}__")
        body = " ".join(f"{m['mark']}{m['umaban']}{m['horse_name']}{'*' if m['orig_mark'] != m['mark'] else ''}" for m in r["marks"][:3])
        lines.append(f"{r['rno']:>2}R {body}")
    if res["skipped"]:
        lines.append(f"（出せなかったレース {len(res['skipped'])}）")
    return lines


def chunks(lines: list[str], limit: int = 1900) -> list[str]:
    out, cur = [], ""
    for ln in lines:
        if cur and len(cur) + len(ln) + 1 > limit:
            out.append(cur); cur = ""
        cur = f"{cur}\n{ln}" if cur else ln
    if cur:
        out.append(cur)
    return out


def discord_private(msgs: list[str]) -> bool:
    """私用サーバー (discord_webhook) だけに送る。公開用の webhook は読まない。"""
    try:
        url = str(json.loads((ROOT / "notify_config.json").read_text(encoding="utf-8")).get("discord_webhook") or "").strip()
    except (OSError, ValueError):
        url = ""
    if not url:
        print("[morning_hosei] discord_webhook 未設定 → 送らない")
        return False
    try:
        for m in msgs:
            req = urllib.request.Request(url, data=json.dumps({"content": m}).encode("utf-8"),
                                         headers={"Content-Type": "application/json", "User-Agent": "pycaliai-morning-hosei"})
            urllib.request.urlopen(req, timeout=15).read()
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[morning_hosei] discord 送信失敗: {type(e).__name__}")
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.date.today().strftime("%Y%m%d"))
    ap.add_argument("--discord", action="store_true", help="私用 Discord に送る (付けなければ表示と保存だけ)")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    res = build(a.date)
    if not res["races"]:
        print(f"[morning_hosei] {a.date}: 出せるレースが無い (skipped {len(res['skipped'])})")
        return 1
    op = BASE / "reports" / "morning_hosei" / f"{a.date}.json"
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = fmt_lines(res)
    print("\n".join(lines))
    print(f"[morning_hosei] saved {op}  λ={res['lambda']} ({res['lambda_source']})")
    if a.discord:
        sent = discord_private(chunks(lines))
        print(f"[morning_hosei] discord {'送信済み' if sent else '未送信'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
