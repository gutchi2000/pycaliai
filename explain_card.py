# -*- coding: utf-8 -*-
"""
explain_card.py — build_site のレース・ビューモデルから「分析カード」を生成(ライブ本番用)
================================================================================
2026 週次 bundle は ELO/Glicko parquet の被覆外(rid16 が未来日)。build_site の
ビューモデルには p_win/p_sho/odds/ev/印/脚質(style)/SHAP(why)/買い判定(judgment)/馬場 が
既に揃っている。ELO/Glicko/位置取りは data/explain_horse_ratings.json(馬名→最新as-of)を
引いて復元し、vs/レースレベルは当該レース出走馬の elo から field 平均で再構成する。
被覆外の馬(2026初出走/新馬)のみ None=graceful。辞書は build_explain_ratings.py で再生成。

  card_from_viewmodel(race_vm, with_llm=False) -> card dict (explain.html と同スキーマ)
  write_day_explain(day_vm, outdir, with_llm) -> site/data/explain/{date}.json
  write_explain_manifest(outdir)

build_site.py から import して使う(重い依存なし・自己完結)。
"""
from __future__ import annotations
import json
import re
from pathlib import Path

MARK_ORDER = {"◎": 0, "〇": 1, "○": 1, "▲": 2, "△": 3}

# 馬名→最新as-ofレーティング(build_explain_ratings.py 生成)。被覆外は graceful に None。
_RATINGS = None


def _ratings():
    global _RATINGS
    if _RATINGS is None:
        try:
            p = Path(__file__).parent / "data/explain_horse_ratings.json"
            _RATINGS = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            _RATINGS = {}
    return _RATINGS


def _split_course(course):
    """'芝1200'/'ダ1800' -> ('芝',1200)。"""
    if not course:
        return (None, None)
    m = re.match(r"\s*([芝ダ障]+)\s*(\d+)", str(course))
    if m:
        return (m.group(1), int(m.group(2)))
    return (str(course), None)


def _ollama(facts, timeout=90):
    import os
    if os.environ.get("EXPLAIN_NO_LLM"):
        return None
    payload = {"model": os.environ.get("PYCALI_MODEL", "qwen2.5:7b"),
               "system": ("あなたはJRA競馬の予想解説者。与えられた数値・事実だけを使い創作厳禁。"
                          "◎は最上位評価馬で『人気薄/穴』と書くな。各馬1〜2文、勝率と脚質とAI注目点に触れる。"
                          "EVは1.0が損益分岐。特選穴馬は与えた馬のみ。煽らず淡々と。英単語禁止。出力は本文のみ。"),
               "prompt": facts, "stream": False, "options": {"num_ctx": 4096, "temperature": 0.4}}
    url = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
    try:
        import requests
        return (requests.post(url, json=payload, timeout=timeout).json().get("response") or "").strip() or None
    except Exception:
        return None


def card_from_viewmodel(race, with_llm=False):
    surf, dist = _split_course(race.get("course"))
    hv = list(race.get("horses", []))
    n = len(hv)
    by_rank = sorted(hv, key=lambda h: (h.get("ai_rank") or 99))

    # 展開: 脚質(style)から逃げ先行頭数
    n_front = sum(1 for h in hv if h.get("style") in ("逃げ", "先行"))
    if n_front >= 4:
        pace = "ハイペース想定(逃げ先行多数)→差し・追込有利"
    elif n_front <= 1:
        pace = "スローペース想定(逃げ先行手薄)→前残り有利"
    else:
        pace = "平均ペース想定"

    # 馬名→as-ofレーティング。vs/レースレベルは「2026出走馬のelo」からfield平均で再構成。
    R = _ratings()
    field_elos = [rt["elo"] for h in by_rank
                  if (rt := R.get(h.get("name") or "")) and rt.get("elo") is not None]
    mean_elo = (sum(field_elos) / len(field_elos)) if len(field_elos) >= 2 else None
    race_level = round(mean_elo) if mean_elo is not None else None

    horses = []
    for h in by_rank:
        pw = h.get("p_win") or 0.0
        ps = h.get("p_sho") or 0.0
        odds = h.get("odds")
        yoso = round(1.0 / pw, 1) if pw > 0 else None
        kijun = round(1.1 / pw, 1) if pw > 0 else None
        ev = h.get("ev_tan")
        if ev is None and odds and pw:
            ev = round(odds * pw, 2)
        rt = R.get(h.get("name") or "") or {}
        elo_v = rt.get("elo")
        vs = round(elo_v - mean_elo, 1) if (elo_v is not None and mean_elo is not None) else None
        se, sl = rt.get("se"), rt.get("sl")
        pos1 = int(round(se * (n - 1) + 1)) if se is not None else None
        pos4 = int(round(sl * (n - 1) + 1)) if sl is not None else None
        horses.append({
            "ban": h.get("umaban"), "name": h.get("name", ""),
            "mark": h.get("mark", "") or "",
            "p_win_pct": round(pw * 100, 1), "p_top3_pct": round(ps * 100, 1),
            "yoso_odds": yoso, "kijun_odds": kijun, "real_odds": odds,
            "ev": ev, "buy": bool(odds and kijun and odds >= kijun),
            "elo": round(elo_v) if elo_v is not None else None, "elo_vs_field": vs,
            "glicko_mu": round(rt["mu"]) if rt.get("mu") is not None else None,
            "glicko_rd": round(rt["rd"]) if rt.get("rd") is not None else None,
            "kyaku": h.get("style") or "不明", "pos_1c": pos1, "pos_4c": pos4,
            "why": h.get("why") or [],
        })

    # 印上位3頭コメント = 勝率 + SHAP(AI注目点)
    comments = []
    for h, hv_row in zip(horses[:3], by_rank[:3]):
        labels = [w.get("label") for w in (hv_row.get("why") or [])[:3] if w.get("label")]
        why_txt = ("AI注目: " + " / ".join(labels)) if labels else ""
        rtg = ""
        if h["elo"] is not None:
            vs = h["elo_vs_field"]
            vs_txt = (f"（平均比{vs:+}）" if vs is not None else "")
            conf = ""
            if h["glicko_rd"] is not None:
                conf = "・実績安定" if h["glicko_rd"] < 80 else "・評価に幅(浅/休明)"
            rtg = f"対戦RTG{h['elo']}{vs_txt}{conf}。"
        comments.append(f"{h['mark']}{h['name']}: {rtg}AI勝率{h['p_win_pct']}%・複勝圏{h['p_top3_pct']}%"
                        f"（予想{h['yoso_odds']}倍/脚質{h['kyaku']}）。{why_txt}")

    # 特選穴馬 = 買い判定の value_horses から中穴妙味
    ana = None
    vhs = (race.get("judgment") or {}).get("value_horses") or []
    cand = [v for v in vhs if v.get("umami_tan") or v.get("ev_tan")]
    if cand:
        v = max(cand, key=lambda x: (x.get("ev_tan") or 0))
        ana = (f"{v.get('horse_name')}: 単勝EV{v.get('ev_tan')}・妙味{v.get('umami_grade', '–')}"
               f"。AIが人気以上に評価。※大穴は構造的に弱いため中穴妙味として少額・抑え推奨。")

    jud = race.get("judgment") or {}
    place = race.get("place") or ""
    rno = race.get("rno")
    # レース名が空のとき(未勝利/新馬等)は「会場名+〇R」を見出しにする。
    rname = race.get("race_name") or (f"{place}{rno}R" if place and rno else (place or "レース"))
    card = {
        "race": {
            "rid": race.get("race_id"), "name": rname,
            "venue": race.get("place"), "surface": surf, "distance": dist,
            "class": race.get("klass"), "field_size": n,
            "race_level_elo": race_level,
            "baba": (f"{race.get('baba','')} {race.get('weather','')}").strip() or None,
            "pace": pace, "n_front_runners": n_front,
            "judgment": {"headline": jud.get("headline"), "category": jud.get("category"),
                         "hardness": jud.get("hardness"), "kenshu_hint": jud.get("kenshu_hint")},
        },
        "horses": horses, "comments_top3": comments, "ana_pick": ana,
        "llm_narrative": None, "result_for_demo": None,
    }

    if with_llm:
        fl = [f"レース: {card['race']['name']} {race.get('place')}{race.get('course')} "
              f"{race.get('klass')} {n}頭。馬場: {card['race']['baba']}。展開: {pace}。"]
        for h, hv_row in zip(horses[:3], by_rank[:3]):
            labels = [w.get("label") for w in (hv_row.get("why") or [])[:3] if w.get("label")]
            fl.append(f"{h['mark']}{h['name']}: 勝率{h['p_win_pct']}% 複勝圏{h['p_top3_pct']}% "
                      f"脚質{h['kyaku']} 実{h['real_odds']}倍 EV{h['ev']} AI注目[{'/'.join(labels)}]。")
        if ana:
            fl.append("特選穴馬データ: " + ana.split("※")[0])
        card["llm_narrative"] = _ollama("\n".join(fl) + "\n\n上記の事実だけで◎〇▲短評と穴一言を。")
    return card


def write_day_explain(day, outdir, with_llm=False):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cards = []
    for race in day.get("races", []):
        try:
            cards.append(card_from_viewmodel(race, with_llm=with_llm))
        except Exception:
            continue
    payload = {"date": day.get("date"), "n_races": len(cards),
               "races": sorted(cards, key=lambda c: str(c["race"]["rid"]))}
    (outdir / f"{day.get('date')}.json").write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return len(cards)


def write_explain_manifest(outdir):
    outdir = Path(outdir)
    if not outdir.exists():
        return
    dates = sorted([p.stem for p in outdir.glob("*.json") if p.stem != "manifest"], reverse=True)
    (outdir / "manifest.json").write_text(
        json.dumps({"dates": dates}, ensure_ascii=False), encoding="utf-8")
