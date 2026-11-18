"""全頭短評（Cowork の keiba-narrative スキル (C)）に渡す素材 (2026-11-16)。

build_site.py が、結果の出ていない開催日を作るたびに、公開用の site/data/{date}.json から
reports/cowork_input/{date}_zento.json を書く。中身はサイトに出している数字だけ（オッズ・EV・
JRDB の値は入れない）。Cowork はこれを読んで全レース全頭に 1〜2 文の短評を書き、
{date}_bets.json の各レースに "zento": [{"umaban", "horse_name", "comment"}] として返す。

2026-11-18: advisor・Grade Scope もペース・展開はここの展開予想（tenkai.py）だけを根拠に書く
（以前は過去走の決め手から推測して、展開タブと食い違った: 11/15 エリザベス女王杯）。
レースごとに「展開予想あり」「ペース予想%」（スロー/平均/ハイの確率と最大のラベル、確度の言葉は
サイトの展開タブと同じ）、「差し追込は届くか」、「先行勢」（序盤予想の上位 2 頭＝展開タブの逃げ候補）。
馬ごとの「展開予想の脚質区分」は、このレースのメンバー内で予想した序盤の位置を
逃げ 1 割・先行 〜35%・差し 〜70%・残り追込で区切ったもの（出走表の「ふだんの脚質」とは別物）。
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


def pace_summary(pace_probs) -> dict | None:
    """[スロー, 平均, ハイ] の確率 → 予想ラベルと確度の言葉（site/js/app.js の展開タブ・worker/tools.js と同じ決め方）。

    「公算が大きい」はスローで 50% 以上のときだけ。ハイは 50% 以上でも「ハイペース寄り」まで
    （2026 年実績: スローの公算 64% 的中、ハイの公算は 47% で逆の端 19%＝「寄り」の 44% と変わらない。
    docs/research/TENKAI_ACCURACY_2026_20261118.md）。平均が最大は情報なし（実際 30/39/31%）。"""
    if not pace_probs or all(x is None for x in pace_probs):
        return None
    v = [x or 0 for x in pace_probs]
    top = v.index(max(v))
    sure = ("はっきりしない（平均寄り）" if top == 1
            else "スローペースの公算が大きい" if top == 0 and v[top] >= 0.5 else f"{PACE_NAMES[top]}ペース寄り")
    return {"予想": PACE_NAMES[top], "確度": sure, "スロー": round(v[0] * 100), "平均": round(v[1] * 100),
            "ハイ": round(v[2] * 100)}


def style_band(rank: int, n: int) -> str:
    """tenkai.py _style_rank と同じ区切り（序盤予想の順位 / 頭数）。"""
    q = rank / max(n, 1)
    return "逃げ" if q <= 0.1 else "先行" if q <= 0.35 else "差し" if q <= 0.70 else "追込"


def tenkai_positions(tk: dict, umabans: list[int]) -> dict:
    """取消を除いた出走馬だけで序盤・4角の予想番手を付け直す（展開タブと同じ）。
    → {馬番: (序盤番手, 4角番手, 脚質区分)}。"""
    rows = [row for row in tk.get("h", []) if row[0] in set(umabans)]
    if not rows:
        return {}
    er = {row[0]: i + 1 for i, row in enumerate(sorted(rows, key=lambda x: (x[1], x[0])))}
    lr = {row[0]: i + 1 for i, row in enumerate(sorted(rows, key=lambda x: (x[2], x[0])))}
    n = len(rows)
    return {u: (er[u], lr[u], style_band(er[u], n)) for u in er}


def positions_unreliable(r: dict, tk: dict, umabans: list[int]) -> bool:
    """新馬戦、または過去走（通過順の分かる過去走＝tenkai の nh）の無い馬が出走の半数を超えるレース。
    各馬の予想番手はほぼでたらめ（2026 年実績: 序盤の順位相関 0.10、全体は 0.50）なので素材に出さない。"""
    if "新馬" in str(r.get("klass") or ""):
        return True
    live = set(umabans)
    nh = [row[6] if len(row) > 6 else None for row in tk.get("h", []) if row[0] in live]
    return bool(nh) and sum(1 for x in nh if not x) > len(nh) / 2


def race_material(r: dict, close_edges: list) -> dict:
    hs = [h for h in r.get("horses", []) if not h.get("scratched")]
    ai = _pct_rank({h["umaban"]: h.get("ai_score") for h in hs})
    pw = _pct_rank({h["umaban"]: h.get("pw") for h in hs})
    tk = r.get("tenkai") or {}
    pos = tenkai_positions(tk, [h["umaban"] for h in hs])
    pace = pace_summary(tk.get("pace"))
    # 新馬・過去走の無い馬が半数超: 予想番手・展開予想の脚質区分・先行勢は出さない（ペースと届くかは出す）
    unreliable = bool(pos) and positions_unreliable(r, tk, [h["umaban"] for h in hs])
    if unreliable:
        pos = {}
    name_by = {h["umaban"]: h.get("name") for h in hs}
    front = [{"馬番": u, "馬名": name_by.get(u)} for u, p in sorted(pos.items(), key=lambda x: x[1][0])[:2]]
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
            "展開予想の脚質区分": p[2] if p else None, "予想序盤番手": p[0] if p else None, "予想4角番手": p[1] if p else None,
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
        "展開予想あり": pace is not None,
        "ペース予想%": pace,
        # 平均が最大＝ペースの情報なし。ペースを根拠に有利不利を書かない
        "ペースを根拠にしてよい": bool(pace) and pace["予想"] != "平均",
        "差し追込は届くか": _late_word(tk.get("late"), close_edges),
        "先行勢": front,
        # "参考外" のとき各馬の予想番手・展開予想の脚質区分・先行勢は空（新馬・過去走の無い馬が半数超）
        "予想位置": "参考外（新馬・過去走の無い馬が半数超）" if unreliable else ("あり" if pos else "なし"),
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
                "AI順位は AI のスコア順、スピード指数は 100 が古馬 1〜2 勝クラスの基準、強さスコアは芝ダ別に現役最上位 100。"
                "ペース・展開・脚質の有利不利は「ペース予想%」「差し追込は届くか」「先行勢」と各馬の予想番手だけを根拠にする"
                "（advisor・Grade Scope・全頭短評すべて）。「展開予想あり」が false のレースはペースに触れない。"
                "「展開予想の脚質区分」はこのレースで予想した序盤の位置の区分で、「ふだんの脚質」（出走表）とは別。"
                "「ペースを根拠にしてよい」が false（平均が最大＝読めない、または展開予想なし）のレースは、"
                "ペースを根拠に有利不利を書かない。「予想位置」が参考外のレース（新馬・過去走の無い馬が半数超）は"
                "予想番手・展開予想の脚質区分・先行勢を出していないので、馬ごとの位置取りを根拠にしない。"
                "「公算が大きい」はスローだけ（ハイは最大でも「寄り」）。",
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
