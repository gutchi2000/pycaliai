# -*- coding: utf-8 -*-
"""board_combo.py — サイト公開用の買い目「GO × ボード一致 × ◎オッズ帯」(単一ソース)。

2026-11-21 ユーザー指示「買い目の配線は俺がさっき言った組み合わせで」「T-10, T-20 でやってほしい」。
方策の正本は data/shadow_policies/board_combo_v1.json（27 セル）。
出どころ: docs/research/BOARD_GO_RECON_EXPLORATION_20261120.md §11 の表で回収率 90% 以上だった
「集合 × 券種 × ◎単勝オッズ帯」（15倍〜 の列は除く）。探索・再構成値で選んだもので、§12 の前後半の
確認では選んだ期間の外で 65〜83% に落ちた（＝期待は控除率なみ）。real_money は false。

1 レースの買い目:
  - 判定の材料: bundle の buy_judgment（category=go か・hardness）、印（◎〇▲△△）、
    公開サイト JSON の 3 ゲージ（速度 speed.v 最大・PW 最大・推定人気 speed.er 最小。board.js と同じ並び順の扱い）、
    ◎の単勝オッズ（呼び出し側が渡す T-10 / T-20 のライブ値）。
  - レースが当てはまるセルの券種をすべて買う。同じ券種は 1 回だけ。
    ワイド・馬連 = ◎-〇▲△△ の 4 点、三連複 = ◎-(〇▲△△ から 2 頭) の 6 点、単勝・複勝 = ◎ 1 点。
  - どのセルにも入らなければ見送り。
本線 (a95 sniper の T-10 ◎複勝) とは独立。こちらは bets に入れず、bets.json の各レースに
"board_combo" として併記し、サイトの TACT 欄だけに出す。
"""
from __future__ import annotations

import hashlib
import json
import math
from itertools import combinations
from pathlib import Path

BASE = Path(__file__).resolve().parent
POLICY_PATH = BASE / "data" / "shadow_policies" / "board_combo_v1.json"
PARTNER = ("○", "▲", "△1", "△2")
TYPE_ORDER = {"単勝": 0, "複勝": 1, "ワイド": 2, "馬連": 3, "三連複": 4}
BAND_CODE = {"<2倍": "A", "2-3倍": "B", "3-5倍": "C", "5-8倍": "D", "8-15倍": "E", "15倍〜": "F"}


def _f(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def load_policy(path: Path = POLICY_PATH) -> dict:
    pol = json.loads(path.read_text(encoding="utf-8"))
    body = json.dumps(pol["cells"], ensure_ascii=False, sort_keys=True)
    if hashlib.sha256(body.encode("utf-8")).hexdigest() != pol["cells_sha256"]:
        raise ValueError("board_combo_v1.json の cells が sha256 と一致しない（手で書き換えた？）")
    return pol


def in_set(name: str, go: bool, agree: int, hard: str | None, gset: str) -> bool:
    if name == "GO∧一致2∧固い":
        return go and agree == 2 and hard == "固い"
    if name == "GO∧一致3∧標準":
        return go and agree == 3 and hard == "標準"
    if name == "GO∧一致2以上∧固い":
        return go and agree >= 2 and hard == "固い"
    if name == "一致3（GO不問）":
        return agree == 3
    if name == "GO∧一致3":
        return go and agree == 3
    if name == "GO∧PWだけ一致":
        return go and gset == "PW"
    raise ValueError(f"未知の集合 {name}")


def band_of(odds: float | None, bands: list) -> str | None:
    if odds is None:
        return None
    for lo, hi, lb in bands:
        if odds >= lo and (hi is None or odds < hi):
            return lb
    return None


def gauges(site_race: dict | None) -> dict | None:
    """公開サイト JSON の 1 レース → {si, pw, er} の馬番（ボードと同じ: 先頭から見て厳密に良いときだけ入れ替え）。"""
    if not site_race:
        return None
    hs = site_race.get("horses") or []
    sp = [h for h in hs if (h.get("speed") or {}).get("v") is not None]
    pw = [h for h in hs if h.get("pw") is not None]
    er = [h for h in hs if (h.get("speed") or {}).get("er") is not None]
    if not (sp and pw and er):
        return None
    si1 = sp[0]
    for h in sp[1:]:
        if h["speed"]["v"] > si1["speed"]["v"]:
            si1 = h
    pw1 = pw[0]
    for h in pw[1:]:
        if h["pw"] > pw1["pw"]:
            pw1 = h
    er1 = er[0]
    for h in er[1:]:
        if h["speed"]["er"] < er1["speed"]["er"]:
            er1 = h
    return {"si": int(si1["umaban"]), "pw": int(pw1["umaban"]), "er": int(er1["umaban"])}


def load_site_day(date_str: str, site_dir: Path | None = None) -> dict:
    """site/data/{date}.json → {rid16: race}。無ければ {}。"""
    p = (site_dir or BASE / "site" / "data") / f"{date_str}.json"
    try:
        j = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {str(r.get("race_id", ""))[:16]: r for r in j.get("races") or []}


def marks_of(horses: list[dict]) -> tuple[int, dict] | None:
    mk: dict[str, list[int]] = {}
    for h in horses:
        m = h.get("mark") or ""
        try:
            mk.setdefault(m, []).append(int(h["umaban"]))
        except (TypeError, ValueError, KeyError):
            continue
    if len(mk.get("◎", [])) != 1 or len(mk.get("〇", [])) != 1 or len(mk.get("▲", [])) != 1 \
            or len(mk.get("△", [])) != 2:
        return None
    return mk["◎"][0], {"○": mk["〇"][0], "▲": mk["▲"][0], "△1": mk["△"][0], "△2": mk["△"][1]}


def selections(btype: str, hon: int, part: dict) -> list[str]:
    if btype in ("単勝", "複勝"):
        return [str(hon)]
    if btype in ("ワイド", "馬連"):
        return [f"{min(hon, part[lb])}-{max(hon, part[lb])}" for lb in PARTNER]
    if btype == "三連複":
        return ["-".join(str(x) for x in sorted((hon, part[a], part[b]))) for a, b in combinations(PARTNER, 2)]
    raise ValueError(btype)


def build(race: dict, hon_odds_live: float | None, site_race: dict | None,
          stage: str, policy: dict | None = None) -> dict:
    """bundle の 1 レース → {"bets": [...公開用 {type, selection, reason}], "skip": bool, "skip_reason", 来歴}。

    stage: "T-10" / "T-20"（理由文と来歴に使う）。hon_odds_live: ◎の単勝オッズ（ライブ値）。
    オッズ生値は戻り値に入れない（帯コードだけ）。
    """
    pol = policy or load_policy()
    out = {"policy_id": pol["policy_id"], "cells_sha256": pol["cells_sha256"], "stage": stage,
           "bets": [], "skip": True}
    j = race.get("buy_judgment") or {}
    if not j.get("category"):
        out["skip_reason"] = "GO 判定が無いため見送り"
        return out
    mm = marks_of(race.get("horses") or [])
    if mm is None:
        out["skip_reason"] = "印（◎〇▲△△）が揃わないため見送り"
        return out
    hon, part = mm
    g = gauges(site_race)
    if g is None:
        out["skip_reason"] = "ボードの指数（速度・PW・推定人気）が揃わないため見送り"
        return out
    go = j.get("category") == "go"
    hard = j.get("hardness")
    agree = int(g["si"] == hon) + int(g["pw"] == hon) + int(g["er"] == hon)
    gset = ("人気" if g["er"] == hon else "") + ("PW" if g["pw"] == hon else "") + ("速度" if g["si"] == hon else "")
    # 前向き確認 (CW-27: 一致 3 の ◎単勝、帯なし) 用に、買い目の有無にかかわらず判定材料を残す
    out.update(hon=hon, go=go, hardness=hard, agree=agree, gauge_set=gset)
    band = band_of(_f(hon_odds_live), pol["bands"])
    if band is None:
        out["skip_reason"] = "◎の単勝オッズが取れないため見送り"
        return out
    hit = [c for c in pol["cells"] if c["band"] == band and in_set(c["set"], go, agree, hard, gset)]
    out.update(band_code=BAND_CODE.get(band, "?"), cells=[f"{c['set']}|{c['type']}|{c['band']}" for c in hit])
    types = sorted({c["type"] for c in hit}, key=lambda t: TYPE_ORDER.get(t, 9))
    if not types:
        out["skip_reason"] = "ボード×GO の対象外のため見送り"
        return out
    lbl = f"ボード一致{agree}・帯{out['band_code']}（{stage}）"
    for t in types:
        for sel in selections(t, hon, part):
            out["bets"].append({"type": t, "selection": sel,
                                "reason": f"◎{'1点' if t in ('単勝', '複勝') else '流し'}・{lbl}"})
    out["skip"] = False
    out.pop("skip_reason", None)
    return out


def build_live(race: dict, tansho: dict | None, date_str: str, stage: str,
               site_dir: Path | None = None) -> dict:
    """T-10 / T-20 用: 障害の除外・予測入力の品質ゲートを通してから build。

    tansho: {馬番(int|str): 単勝オッズ} のライブ値。site JSON は site/data/{date}.json（Phase A で作られる）。
    呼び出し側は例外を握りつぶして本線に影響させないこと。
    """
    from race_eligibility import EligibilityMetadataError, evaluate_race, verify_metadata
    rid = str(race.get("race_id") or (race.get("race_meta") or {}).get("race_id") or "")[:16]
    base = {"policy_id": load_policy()["policy_id"], "stage": stage, "bets": [], "skip": True}
    try:
        el = verify_metadata(race.get("eligibility"), rid)
    except EligibilityMetadataError:
        el = evaluate_race(rid)
    if not el["bet_eligible"]:
        return {**base, "skip_reason": "対象外レースのため見送り"}
    from input_quality import race_skip_reason
    from production_policy import load_policy as _pp
    if race_skip_reason(race, _pp().get("serve_path_version")):
        return {**base, "skip_reason": "予測入力の品質ゲートで見送り"}
    tan = {int(k): v for k, v in (tansho or {}).items()}
    mm = marks_of(race.get("horses") or [])
    hon_odds = tan.get(mm[0]) if mm else None
    site_race = load_site_day(date_str, site_dir).get(rid)
    return build(race, hon_odds, site_race, stage)
