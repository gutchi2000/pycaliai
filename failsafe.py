# -*- coding: utf-8 -*-
"""
failsafe.py — T-10 当日フェイルセーフ（資金ガード）の頭脳
==========================================================
推奨買い目を当日中に決済し、当日P&L/ROI を積み上げ、しきい値で
  #1 資金・下振れ: 当日損失/DD 上限 → halt(以降見送り) / 減額
  #5 資金・上振れ: 当日ROI>100%(=net黒字) → 利確サイン(買い目は送るが非推奨)
を判定する。結果ソース(JV-Link 0B30 等)に依存しない純ロジック=単体テスト可。

state: reports/live_odds/{date}_failsafe.json （当日のみ。日跨ぎでリセット）
config: failsafe_config.json （無ければ既定値。ユーザーが実バンクロールで調整）
"""
from __future__ import annotations
import json, re
from pathlib import Path

BASE = Path(__file__).parent
CFG_PATH = BASE / "failsafe_config.json"

DEFAULT_CFG = {
    # #1 下振れ: 当日の純損失がこの額(円)を割ったら halt（以降のレースを見送り)
    "day_loss_halt_yen": 30000,
    # #1 下振れ: 当日純損失がこの額を割ったら以降の購入額を stake_down_factor 倍に減額
    "day_loss_stakedown_yen": 15000,
    "stake_down_factor": 0.5,
    # #5 上振れ: 当日ROIがこの%を超えたら利確サイン（買い目は送るが「非推奨」マーク）
    "profit_take_roi_pct": 100.0,
    # 利確/halt は「ソフト」: 人間が最終判断。買い目自体は出し続ける（halt時のみ空）
    "halt_is_hard": False,
}


def load_cfg() -> dict:
    cfg = dict(DEFAULT_CFG)
    try:
        cfg.update(json.loads(CFG_PATH.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        pass
    return cfg


# ---------------------------------------------------------------
# 決済: 1ベット (馬券種, 買い目, 購入額) × レース結果 → (hit, payout円)
#   result = {win, plc, sho: 馬番,
#             tansho: 単勝配当(/100), fukusho:{ban:配当}, umaren, umatan,
#             wide:{frozenset({i,j}):配当}, sanrenpuku, sanrentan}
#   配当は「100円あたりの払戻額」。payout = 購入額/100 * 配当。
# ---------------------------------------------------------------
def _legs(sel: str):
    """買い目 → 組のリスト。'4→8'/'5-15'/'1,2,3' 等を正規化。"""
    out = []
    for c in str(sel).replace("→", "-").replace("＞", "-").replace(">", "-").replace("－", "-").split(","):
        nums = [int(x) for x in re.findall(r"\d+", c)]
        if nums:
            out.append(nums)
    return out


def settle(bet: dict, result: dict) -> tuple[bool, float]:
    kind = bet.get("馬券種"); stake = float(bet.get("購入額", 0) or 0)
    legs = _legs(bet.get("買い目", ""))
    if not legs or stake <= 0 or not result:
        return False, 0.0
    win, plc, sho = result.get("win"), result.get("plc"), result.get("sho")
    top3 = {x for x in (win, plc, sho) if x is not None}
    per = stake / max(len(legs), 1)   # 複数組は等分賭けと見なす
    pay = 0.0; hit = False

    def add(cond, haitou):
        nonlocal pay, hit
        if cond and haitou and haitou == haitou:   # not NaN
            pay += per / 100.0 * float(haitou); hit = True

    for leg in legs:
        if kind == "単勝":
            add(leg[0] == win, result.get("tansho"))
        elif kind == "複勝":
            add(leg[0] in top3, (result.get("fukusho") or {}).get(leg[0]))
        elif kind == "ワイド" and len(leg) >= 2:
            wd = result.get("wide") or {}
            add(leg[0] in top3 and leg[1] in top3, wd.get(frozenset((leg[0], leg[1]))))
        elif kind == "馬連" and len(leg) >= 2:
            add({leg[0], leg[1]} == {win, plc}, result.get("umaren"))
        elif kind == "馬単" and len(leg) >= 2:
            add((leg[0], leg[1]) == (win, plc), result.get("umatan"))
        elif kind == "三連複" and len(leg) >= 3:
            add(set(leg[:3]) == top3, result.get("sanrenpuku"))
        elif kind == "三連単" and len(leg) >= 3:
            add((leg[0], leg[1], leg[2]) == (win, plc, sho), result.get("sanrentan"))
    return hit, round(pay, 1)


# ---------------------------------------------------------------
# 当日 state の積み上げ + しきい値判定
# ---------------------------------------------------------------
def _state_path(date_str: str) -> Path:
    return BASE / "reports" / "live_odds" / f"{date_str}_failsafe.json"


def load_state(date_str: str) -> dict:
    p = _state_path(date_str)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"date": date_str, "settled_rids": [], "stake": 0.0, "ret": 0.0,
                "peak_pnl": 0.0, "min_pnl": 0.0}


def save_state(date_str: str, st: dict) -> None:
    p = _state_path(date_str)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")


def settle_race_into_state(st: dict, rid16: str, race_bets: list[dict], result: dict) -> dict:
    """1レースの推奨買い目を結果で決済し state に積む（同一 rid は二重計上しない）。"""
    if rid16 in st.get("settled_rids", []):
        return st
    for b in race_bets:
        _, pay = settle(b, result)
        st["stake"] += float(b.get("購入額", 0) or 0)
        st["ret"] += pay
    st.setdefault("settled_rids", []).append(rid16)
    pnl = st["ret"] - st["stake"]
    st["peak_pnl"] = max(st.get("peak_pnl", 0.0), pnl)
    st["min_pnl"] = min(st.get("min_pnl", 0.0), pnl)
    return st


def evaluate(st: dict, cfg: dict | None = None) -> dict:
    """当日 state → {mode, roi, pnl, dd, message}。
    mode: 'normal' | 'profit_take'(#5) | 'stakedown'(#1) | 'halt'(#1)。"""
    cfg = cfg or load_cfg()
    stake, ret = st.get("stake", 0.0), st.get("ret", 0.0)
    pnl = ret - stake
    roi = (ret / stake * 100.0) if stake > 0 else 0.0
    dd = st.get("peak_pnl", 0.0) - pnl   # ピークからの落ち込み
    mode, msg = "normal", ""
    if pnl <= -abs(cfg["day_loss_halt_yen"]):
        mode = "halt"
        msg = (f"🛑 当日損失 ¥{-pnl:,.0f} が上限¥{cfg['day_loss_halt_yen']:,}超 → "
               f"本日の購入を停止推奨(以降見送り)。ROI {roi:.0f}%")
    elif roi >= cfg["profit_take_roi_pct"] and stake > 0:
        mode = "profit_take"
        msg = (f"✅ 当日ROI {roi:.0f}% (収支 +¥{pnl:,.0f}) ＝黒字。"
               f"これ以上やる必要なし。買い目は送るが非推奨—続けるかは任せます。")
    elif pnl <= -abs(cfg["day_loss_stakedown_yen"]):
        mode = "stakedown"
        msg = (f"⚠ 当日損失 ¥{-pnl:,.0f} → 以降の購入額を×{cfg['stake_down_factor']}に減額推奨。"
               f"ROI {roi:.0f}%")
    return {"mode": mode, "roi": round(roi, 1), "pnl": round(pnl, 0),
            "dd": round(dd, 0), "stake": round(stake, 0), "ret": round(ret, 0),
            "n_races": len(st.get("settled_rids", [])), "message": msg}


# kekka CSV → result dict （テスト/フォールバック用。本番は JV-Link 0B30）
def result_from_kekka_row(g) -> dict:
    """g = 1レース分の kekka DataFrame (cp932)。"""
    import pandas as pd
    g = g.copy(); g["_c"] = pd.to_numeric(g["確定着順"], errors="coerce")
    def ban(pos):
        m = g[g["_c"] == pos]
        return int(m["馬番"].iloc[0]) if len(m) else None
    def num(col):
        if col not in g.columns: return None
        v = pd.to_numeric(g[col].astype(str).str.replace(r"[()]", "", regex=True), errors="coerce").dropna()
        return float(v.iloc[0]) if len(v) else None
    fuk = {}
    for b, x in zip(g["馬番"], g["複勝配当"]):
        s = str(x)
        if s.strip() not in ("", "nan") and not s.startswith("("):
            try: fuk[int(b)] = float(s)
            except ValueError: pass
    tan = None
    for b, x in zip(g["馬番"], g["単勝配当"]):
        s = str(x)
        if s.strip() not in ("", "nan") and not s.startswith("("):
            try: tan = float(s); break
            except ValueError: pass
    return {"win": ban(1), "plc": ban(2), "sho": ban(3), "tansho": tan, "fukusho": fuk,
            "umaren": num("馬連"), "umatan": num("馬単"),
            "sanrenpuku": num("３連複"), "sanrentan": num("３連単"), "wide": {}}
