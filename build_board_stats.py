"""デイリーボード用の集計 (2026-11-16).

公開済み site/data/{date}.json だけから、結果の出たレースについて
「◎ と 速度1位 / PW1位 / 推定1番人気 が何個一致したか」ごとの ◎ の成績と、
難易度 (固い / 標準 / 荒れ) ごとの ◎ の成績を集計して site/data/board_stats.json に書く。
build_site.py の末尾で毎ビルド呼ばれる。LLM・オッズは使わない。

python build_board_stats.py
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SITE_DATA = ROOT / "site" / "data"
OUT = SITE_DATA / "board_stats.json"
GAUGE_OUT = SITE_DATA / "gauge_ref.json"
GAUGES = ("top1_dominance", "top2_concentration", "field_chaos_score", "ai_market_agreement")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def tops(r: dict) -> dict | None:
    """◎・速度1位・PW1位・推定1番人気 の馬番。どれかが無いレースは None 扱いの項目を持つ。"""
    hs = [h for h in r.get("horses", []) if not h.get("scratched")]
    hon = next((h["umaban"] for h in hs if h.get("mark") == "◎"), None)
    sp = [h for h in hs if (h.get("speed") or {}).get("v") is not None]
    pw = [h for h in hs if h.get("pw") is not None]
    er = [h for h in hs if (h.get("speed") or {}).get("er") is not None]
    return {
        "hon": hon,
        "si": max(sp, key=lambda h: h["speed"]["v"])["umaban"] if sp else None,
        "pw": max(pw, key=lambda h: h["pw"])["umaban"] if pw else None,
        "er": min(er, key=lambda h: h["speed"]["er"])["umaban"] if er else None,
    }


def _row(key, a):
    n, w, t3 = a
    lo, hi = wilson(t3, n)
    return {"k": key, "n": n, "win": round(w / n, 4) if n else None,
            "top3": round(t3 / n, 4) if n else None, "top3_lo": round(lo, 4), "top3_hi": round(hi, 4)}


def build() -> dict:
    gv = {k: [] for k in GAUGES}          # 全レースの値（位置＝百分位の物差し）
    gres = {k: [] for k in GAUGES}        # 結果のあるレースの (値, ◎1着, ◎3着内)
    lvl = {}                              # (印, 馬レベル) → [頭数, AI予測3着内の合計, 実際の3着内]
    agree = {k: [0, 0, 0] for k in range(4)}
    hard = {k: [0, 0, 0] for k in ("固い", "標準", "荒れ")}
    dates = []
    for f in sorted(SITE_DATA.glob("20[0-9][0-9][0-1][0-9][0-3][0-9].json")):
        d = json.loads(f.read_bytes())
        used = False
        for r in d.get("races", []):
            res = r.get("result") or {}
            order = res.get("order")
            conf = r.get("confidence") or {}
            for k in GAUGES:
                if conf.get(k) is not None:
                    gv[k].append(float(conf[k]))
            if not order:
                continue
            for h in r.get("horses", []):
                lv = (h.get("level") or {}).get("tier")
                if h.get("scratched") or not lv or h.get("p_sho") is None:
                    continue
                mk = {"○": "〇"}.get(h.get("mark") or "", h.get("mark") or "無")
                pos = order.get(str(h.get("umaban")))
                a = lvl.setdefault(f"{mk}|{lv}", [0, 0.0, 0])
                a[0] += 1
                a[1] += float(h["p_sho"])
                a[2] += int(pos is not None and pos <= 3)
            t = tops(r)
            if t["hon"] is not None:
                p0 = order.get(str(t["hon"]))
                for k in GAUGES:
                    if conf.get(k) is not None:
                        gres[k].append((float(conf[k]), int(p0 == 1), int(p0 is not None and p0 <= 3)))
            if t["hon"] is None:
                continue
            pos = order.get(str(t["hon"]))
            win = int(pos == 1)
            top3 = int(pos is not None and pos <= 3)
            hd = (r.get("judgment") or {}).get("hardness")
            if hd in hard:
                a = hard[hd]; a[0] += 1; a[1] += win; a[2] += top3
            if None in (t["si"], t["pw"], t["er"]):
                continue
            k = (t["si"] == t["hon"]) + (t["pw"] == t["hon"]) + (t["er"] == t["hon"])
            a = agree[k]; a[0] += 1; a[1] += win; a[2] += top3
            used = True
        if used:
            dates.append(f.stem)
    out = {
        "built_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "from": dates[0] if dates else None, "to": dates[-1] if dates else None,
        "n_days": len(dates),
        "n_races": sum(a[0] for a in agree.values()),
        "by_agree": [_row(k, agree[k]) for k in range(4)],
        "by_hardness": [_row(k, hard[k]) for k in ("固い", "標準", "荒れ")],
    }
    OUT.write_bytes((json.dumps(out, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))
    build_gauge_ref(gv, gres, dates, lvl)
    print(f"board_stats: {out['n_races']} races / {out['n_days']} days -> {OUT.name}")
    return out


def build_gauge_ref(gv: dict, gres: dict, dates: list, lvl: dict | None = None) -> None:
    """レース上部の 4 つのゲージを「今年のレースの中での位置（百分位）」で出すための物差し。
    q: 0〜100% 点の値（101 個）、bands: 位置 5 等分ごとの◎の 1 着率・3 着内率（結果のあるレースだけ）。"""
    import bisect
    out = {"built_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
           "from": dates[0] if dates else None, "to": dates[-1] if dates else None, "gauges": {}}
    for k in GAUGES:
        v = sorted(gv[k])
        if len(v) < 50:
            continue
        q = [round(v[min(len(v) - 1, int(round(i / 100 * (len(v) - 1))))], 5) for i in range(101)]

        def pct(x):
            # 同じ値が並ぶとき（勝率が同点で 0 など）は、その値の中央の位置にする
            lo = bisect.bisect_left(v, x)
            hi = bisect.bisect_right(v, x)
            return (lo + hi) / 2 / len(v)
        bands = [[0, 0, 0] for _ in range(5)]
        for x, w, t3 in gres[k]:
            b = min(4, int(pct(x) * 5))
            bands[b][0] += 1
            bands[b][1] += w
            bands[b][2] += t3
        out["gauges"][k] = {
            "n": len(v), "q": q,
            "bands": [{"n": n, "win": round(w / n, 3) if n else None, "top3": round(t / n, 3) if n else None}
                      for n, w, t in bands],
        }
    # 印 × 馬レベル: AI の予測 3 着内率と実際（印と馬レベルが食い違うときの注意書きに使う）
    out["level_mark"] = {k: {"n": n, "pred": round(s / n, 3), "act": round(t / n, 3)}
                         for k, (n, s, t) in sorted((lvl or {}).items()) if n >= 50}
    GAUGE_OUT.write_bytes((json.dumps(out, ensure_ascii=False) + "\n").encode("utf-8"))
    print("gauge_ref: " + ", ".join(f"{k}={g['n']}" for k, g in out["gauges"].items()))


if __name__ == "__main__":
    build()
