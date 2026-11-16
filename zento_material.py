"""全頭短評（Cowork の keiba-narrative スキル (C)）に渡す素材 (2026-11-16)。

build_site.py が、結果の出ていない開催日を作るたびに、公開用の site/data/{date}.json から
reports/cowork_input/{date}_zento.json を書く。中身はサイトに出している数字だけ（オッズ・EV・
JRDB の値は入れない）。Cowork はこれを読んで全レース全頭に 1〜2 文の短評を書き、
{date}_bets.json の各レースに "zento": [{"umaban", "horse_name", "comment"}] として返す。
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "reports" / "cowork_input"
PACE_NAMES = ["スロー", "平均", "ハイ"]


def _pct_rank(vals: dict, key_desc=True) -> dict:
    xs = sorted(((v, k) for k, v in vals.items() if v is not None), reverse=key_desc)
    return {k: i + 1 for i, (v, k) in enumerate(xs)}


def _late_word(p, edges):
    if p is None:
        return None
    b = 0
    while b < len(edges) and p > edges[b]:
        b += 1
    return ["届きにくい", "やや届きにくい", "ふつう", "やや届きやすい", "届きやすい"][b]


def race_material(r: dict, close_edges: list) -> dict:
    hs = [h for h in r.get("horses", []) if not h.get("scratched")]
    ai = _pct_rank({h["umaban"]: h.get("ai_score") for h in hs})
    pw = _pct_rank({h["umaban"]: h.get("pw") for h in hs})
    tk = r.get("tenkai") or {}
    pos = {row[0]: row for row in tk.get("h", [])}
    pace = None
    if tk.get("pace"):
        v = [x or 0 for x in tk["pace"]]
        pace = {"予想": PACE_NAMES[v.index(max(v))], "スロー": round(v[0] * 100), "平均": round(v[1] * 100),
                "ハイ": round(v[2] * 100)}
    ml = r.get("member_level") or {}
    horses = []
    for h in sorted(hs, key=lambda x: ai.get(x["umaban"], 99)):
        u = h["umaban"]
        sp = h.get("speed") or {}
        p = pos.get(u)
        hist = (h.get("history") or {}).get("runs") or []
        horses.append({
            "馬番": u, "馬名": h.get("name"), "印": h.get("mark") or "", "性齢": f"{h.get('sex', '')}{h.get('age', '')}",
            "騎手": h.get("jockey"), "斤量": h.get("kinryo"),
            "AI順位": ai.get(u), "AI勝率%": None if h.get("p_win") is None else round(h["p_win"] * 100, 1),
            "AI3着内%": None if h.get("p_sho") is None else round(h["p_sho"] * 100),
            "スピード指数(予測)": sp.get("v"), "スピード指数順位": sp.get("rk"),
            "強さスコア": h.get("pw"), "強さスコア順位": pw.get(u),
            "推定人気": sp.get("er"),
            "馬レベル": (h.get("level") or {}).get("tier"),
            "妙味グレード": (h.get("umami") or {}).get("grade"),
            "ふだんの脚質": h.get("style"),
            "予想脚質": p[5] if p else None, "予想序盤番手": p[3] if p else None, "予想4角番手": p[4] if p else None,
            "近走スピード指数": [{"日付": x[0], "コース": x[1], "クラス": x[2], "着順": x[3], "指数": x[4]}
                         for x in (sp.get("runs") or [])[:3]],
            "近走": [{"前": x.get("n_ago"), "場": x.get("place"), "芝ダ距離": f"{x.get('td', '')}{x.get('dist', '')}",
                    "馬場": x.get("track"), "着順": x.get("pos"), "人気": x.get("ninki"),
                    "レース": x.get("race_name"), "脚質": x.get("style"), "間隔週": x.get("interval_weeks")}
                   for x in hist[:3]],
        })
    return {
        "race_id": r["race_id"],
        "race_label": f"{r.get('place', '')}{r.get('rno', '')}R {r.get('course', '')} {r.get('klass', '')} {r.get('race_name', '')}".strip(),
        "頭数": len(hs),
        "難しさ": (r.get("judgment") or {}).get("hardness"),
        "メンバーレベル": ml.get("label"),
        "ペース予想%": pace,
        "差し追込は届くか": _late_word(tk.get("late"), close_edges),
        "horses": horses,
    }


def write(day: dict) -> Path | None:
    """結果の出ていない開催日だけ書く（結果のある日は短評を作り直す意味がない）。"""
    if any(r.get("result") for r in day.get("races", [])):
        return None
    edges = [0.054, 0.094, 0.129, 0.173]
    meta = ROOT / "site" / "data" / "tenkai_meta.json"
    if meta.exists():
        try:
            edges = json.loads(meta.read_text(encoding="utf-8")).get("close", {}).get("edges") or edges
        except Exception:
            pass
    out = {
        "date": day["date"],
        "note": "全頭短評の素材。公開サイトに出している数字だけ（オッズ・JRDB の値は無し）。"
                "AI順位は AI のスコア順、スピード指数は 100 が古馬 1〜2 勝クラスの基準、強さスコアは芝ダ別に現役最上位 100。",
        "races": [race_material(r, edges) for r in day["races"]],
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / f"{day['date']}_zento.json"
    p.write_bytes((json.dumps(out, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))
    return p


if __name__ == "__main__":
    import sys
    d = sys.argv[1]
    day = json.loads((ROOT / "site" / "data" / f"{d}.json").read_bytes())
    print(write(day))
