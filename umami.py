# -*- coding: utf-8 -*-
"""
umami.py — UMAMI（馬味）スコア: 「馬券的にどれだけ美味しいか」の単一指標
=======================================================================
生 EV (p×オッズ) は「モデルが市場と喧嘩している度合い」であり、喧嘩の大半は
モデルの間違い (audit_ev_bin_roi: EV2.0+ の実現ROI 64%、単勝50倍超帯は
ROI 30-61% / CLV -54〜-67%)。生 EV を「美味しさ」として表示するのは罠。

UMAMI = **実測テーブルで補正した期待回収率 (xROI)**:
    reports/audit_ev_bin_roi.json の by_ev_x_fav
    (EVビン × 単勝オッズ帯 → test 2024-25 の実現ROI, leak-safe 9時オッズ)
    を参照テーブルにして、その馬・その券種の「過去に同じ状況だった馬券の
    実際の回収率」を返す。1.0=損益分岐、0.80=控除率中立。

ゲート (ユーザー要件: 妙味があっても明らかに来ない馬は出さない):
    - 来ない馬: calibrated 確率の下限 (単勝 p_win / 複勝 p_sho)
    - 大穴帯: 単勝オッズ 50 倍超は実測最悪セグメントとして常時ゲート
    ゲート落ちは gated=True + 理由付きで返す (表示側で「罠」扱い)。

依存: stdlib のみ (betting_judgment と同方針)。
使用者: betting_judgment.extract_value_horses / nicegui_app 全頭分析タブ。
再fit: audit_ev_bin_roi.json を再生成したら自動で追従 (キャッシュは起動毎)。
"""
from __future__ import annotations

import functools
import json
from pathlib import Path

BASE = Path(__file__).parent
AUDIT_JSON = BASE / "reports" / "audit_ev_bin_roi.json"

# --- ゲート閾値 (調整はここだけ) ---
P_WIN_FLOOR = 0.04    # 単勝 UMAMI: 勝率これ未満は「来ない馬」
P_SHO_FLOOR = 0.12    # 複勝 UMAMI: 複勝率これ未満は「来ない馬」
ODDS_HARD_CAP = 50.0  # 単勝オッズこれ超は実測最悪帯 (ROI 30-61%) → 常時ゲート
MIN_CELL_N = 300      # セルの標本がこれ未満なら EV ビン周辺値へフォールバック

# audit_ev_bin_roi.json のビン定義 (生成側と一致させること)
EV_EDGES = [0.8, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0]
EV_LABELS = ["<0.8", "0.8-0.9", "0.9-1.0", "1.0-1.1",
             "1.1-1.3", "1.3-1.5", "1.5-2.0", "2.0+"]
FAV_EDGES = [3.0, 7.0, 15.0, 50.0]
FAV_LABELS = ["<3", "3-7", "7-15", "15-50", "50+"]

# UMAMI グレード (xROI 基準。0.80=控除率中立)
GRADE_EDGES = [(0.85, "S"), (0.80, "A"), (0.72, "B")]


@functools.lru_cache(maxsize=1)
def _tables() -> dict:
    """audit_ev_bin_roi.json → {kind: {"x": by_ev_x_fav, "ev": by_ev_bin}}"""
    if not AUDIT_JSON.exists():
        return {}
    try:
        d = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
        out = {}
        for kind in ("tansho", "fukusho", "umaren"):
            out[kind] = {
                "x": (d.get("by_ev_x_fav") or {}).get(kind, {}),
                "ev": (d.get("by_ev_bin") or {}).get(kind, {}),
            }
        return out
    except Exception:
        return {}


def _bin_label(value: float, edges: list[float], labels: list[str]) -> str:
    for i, e in enumerate(edges):
        if value < e:
            return labels[i]
    return labels[-1]


def umami(kind: str, p: float | None, odds: float | None,
          tansho_odds: float | None = None) -> dict:
    """1 つの (券種, calibrated確率, オッズ) の UMAMI を返す。

    kind: 'tansho' / 'fukusho' (umaren はテーブルあり、必要時に追加)
    p:    calibrated 確率 (単勝=p_win, 複勝=p_sho)
    odds: その券種のオッズ (複勝は (low+high)/2)
    tansho_odds: 人気帯の層別キー。複勝でも「その馬の単勝オッズ」を渡す
                 (audit テーブルの fav 帯が単勝オッズ基準のため)。省略時 odds。

    返り値: {
      xroi:  実測補正後の期待回収率 (None=データ不足),
      ev:    生 EV (参考表示用),
      grade: 'S'/'A'/'B'/'C' (gated 時は '罠'),
      gated: bool, gate_reason: str,
      cell_n: 参照セルの標本数,
    }
    """
    p = float(p or 0)
    odds = float(odds or 0)
    band_odds = float(tansho_odds if tansho_odds else odds)
    ev = p * odds

    res = {"xroi": None, "ev": round(ev, 2), "grade": "C",
           "gated": False, "gate_reason": "", "cell_n": 0}

    if p <= 0 or odds <= 0:
        res.update(gated=True, gate_reason="確率/オッズ欠損")
        return res

    # --- ゲート: 明らかに来ない馬 (妙味があっても出さない) ---
    floor = P_WIN_FLOOR if kind == "tansho" else P_SHO_FLOOR
    label = "勝率" if kind == "tansho" else "複勝率"
    if p < floor:
        res.update(gated=True,
                   gate_reason=f"{label}{p*100:.1f}%<{floor*100:.0f}% (来る見込み薄)")
    elif band_odds > ODDS_HARD_CAP:
        res.update(gated=True,
                   gate_reason=f"単勝{band_odds:.0f}倍の大穴帯 (実測ROI 30-61%の罠)")

    # --- xROI: EVビン × オッズ帯 の実測 ROI ---
    t = _tables().get(kind)
    if not t:
        return res
    key = f"{_bin_label(ev, EV_EDGES, EV_LABELS)}|{_bin_label(band_odds, FAV_EDGES, FAV_LABELS)}"
    cell = t["x"].get(key)
    if cell and cell.get("n", 0) >= MIN_CELL_N:
        res["xroi"] = round(float(cell["roi"]), 3)
        res["cell_n"] = int(cell["n"])
    else:
        # セル標本不足 → EV ビン単独の実測 ROI にフォールバック
        evcell = t["ev"].get(_bin_label(ev, EV_EDGES, EV_LABELS))
        if evcell:
            res["xroi"] = round(float(evcell["roi"]), 3)
            res["cell_n"] = int(evcell.get("n", 0))

    # --- グレード ---
    if res["gated"]:
        res["grade"] = "罠"
    elif res["xroi"] is not None:
        res["grade"] = "C"
        for edge, g in GRADE_EDGES:
            if res["xroi"] >= edge:
                res["grade"] = g
                break
    return res


def umami_for_horse(h: dict) -> dict:
    """bundle の horse dict から 単勝/複勝 両方の UMAMI を計算。
    返り値: {"tan": umami_dict, "fuku": umami_dict}
    """
    tan_odds = h.get("tansho_odds")
    lo, hi = h.get("fuku_odds_low") or 0, h.get("fuku_odds_high") or 0
    fmid = (lo + hi) / 2.0 if (lo and hi) else None
    return {
        "tan": umami("tansho", h.get("p_win"), tan_odds, tansho_odds=tan_odds),
        "fuku": umami("fukusho", h.get("p_sho"), fmid, tansho_odds=tan_odds),
    }


def umami_total(h: dict, um: dict | None = None) -> dict:
    """UMAMI総合 = 単勝/複勝のうちゲートを通った側の最良 xROI
    (= この馬で一番マシな買い方をしたときの実測期待回収率)。

    返り値: {"xroi": float|None, "side": "単勝"/"複勝"/None,
             "grade": str, "um": umami_for_horse の生 dict}
    両側 gated → xroi=None, grade="罠"。
    """
    um = um or umami_for_horse(h)
    cands = []
    if not um["tan"]["gated"] and um["tan"]["xroi"] is not None:
        cands.append((um["tan"]["xroi"], "単勝", um["tan"]["grade"]))
    if not um["fuku"]["gated"] and um["fuku"]["xroi"] is not None:
        cands.append((um["fuku"]["xroi"], "複勝", um["fuku"]["grade"]))
    if not cands:
        return {"xroi": None, "side": None, "grade": "罠", "um": um}
    xroi, side, grade = max(cands, key=lambda c: c[0])
    return {"xroi": xroi, "side": side, "grade": grade, "um": um}


_GRADE_NOTE = {
    "S": "控除率(80%)を超える数少ない帯",
    "A": "控除率相当で負けにくい帯",
    "B": "平凡な帯 (控除率未満)",
    "C": "実測の悪い帯。見送り寄り",
}


def explain(h: dict, total: dict | None = None) -> str:
    """UMAMI 判定の理由を 1〜2 文の自然な日本語で返す (テーブルの「理由」列用)。

    構成: なぜその判定か = ①ゲート理由 (罠) または
          ②人気帯×AI評価の市場乖離 + 同条件帯の実測回収率 + グレード注釈。
    """
    total = total or umami_total(h)
    um = total["um"]
    tan_odds = h.get("tansho_odds")

    # 罠: ゲート理由をそのまま (単勝側優先)
    if total["side"] is None:
        reason = um["tan"]["gate_reason"] or um["fuku"]["gate_reason"]
        return reason or "確率/オッズ欠損で評価不能"

    side = total["side"]
    cell = um["tan"] if side == "単勝" else um["fuku"]
    p = (h.get("p_win") if side == "単勝" else h.get("p_sho")) or 0

    # 市場との乖離 (単勝オッズの implied 確率と AI の比較)
    mk = ""
    if side == "単勝" and tan_odds:
        imp = 1.0 / float(tan_odds)
        if p >= imp * 1.2:
            mk = f"AI勝率{p*100:.0f}%は市場({imp*100:.0f}%)より割安。"
        elif p <= imp * 0.8:
            mk = f"AI勝率{p*100:.0f}%は市場({imp*100:.0f}%)より割高。"
        else:
            mk = f"AI勝率{p*100:.0f}%は市場とほぼ一致。"
    elif side == "複勝":
        mk = f"AI複勝率{p*100:.0f}%。"

    band = ""
    if tan_odds:
        o = float(tan_odds)
        band = ("本命級" if o < 3 else "中人気" if o < 7 else
                "やや人気薄" if o < 15 else "人気薄") + f"({o:.1f}倍)の{side}。"

    note = _GRADE_NOTE.get(total["grade"], "")
    return (f"{band}{mk}同条件帯の実測回収{cell['xroi']*100:.0f}%"
            f"＝{note}" if cell["xroi"] is not None else f"{band}{mk}実測データ不足")
