"""
betting_judgment.py
===================
「買い方判定」ロジック (レースの堅さ × 妙味馬の有無 → 買い方) の単一ソース。

NiceGUI 表示 (nicegui_app.py) と bundle 埋め込み (export_marks_json.py) の
両方がこのモジュールを使うことで、閾値・定義のズレを防ぐ。

設計方針:
  - stdlib のみ依存 (json / bisect / functools / pathlib)。pandas 等は使わない。
  - レース単位値 (混戦度) と 馬単位値 (勝率/オッズ/妙味) は呼び出し側が渡す
    race_confidence(dict) と horses(list[dict]) から読む。
  - 閾値はすべてモジュール冒頭の定数。実データ (audit_marks / cowork_results)
    が溜まったらここだけ触れば全体の判定が動く。

★ 触らない範囲: EV 計算・印ロジック・既存の総合判定 4 指標パネルには手を入れない。
   buy_judgment は「妙味馬リストを出す」+「買い方の濃淡を示す」までで、
   最終取捨 (券種/点数/金額) は Cowork (LLM) と人間が行う。
"""
from __future__ import annotations

import bisect
import functools
import json
from pathlib import Path

BASE = Path(__file__).parent
CHAOS_Q_JSON = BASE / "data" / "chaos_quantiles.json"

# ============================================================
# 閾値 (仮。実データで調整可能。ここだけ触れば全体に効く)
# ============================================================
# --- レースの堅さ (混戦度パーセンタイル 0-1 で判定) ---
CHAOS_HARD_MAX = 0.30    # 混戦度 <= これ → 固い
CHAOS_ROUGH_MIN = 0.70   # 混戦度 >= これ → 荒れ
#                          その間 → 標準

# --- 妙味馬 (買い候補) の条件 ---
VALUE_EV_MIN = 1.10      # 割安側 (単勝 or 複勝) の EV 下限
VALUE_PWIN_MIN = 0.05    # 勝率下限 (テール除外。これ未満は高 EV でも候補外)
VALUE_BAND = 0.20        # vs市場 ±バンド (model_p >= market_p*(1+band) で「割安」)
#                          既存 ai_vs_market (export_marks_json) と同じ ±20%


# ============================================================
# 混戦度: 生値 → 過去分布パーセンタイル(0-1)
# (nicegui_app._to_pct と同じロジック。chaos_quantiles.json 共有)
# ============================================================
@functools.lru_cache(maxsize=1)
def _quantile_tables() -> dict:
    if not CHAOS_Q_JSON.exists():
        return {}
    try:
        return json.loads(CHAOS_Q_JSON.read_text(encoding="utf-8")).get("quantiles", {})
    except Exception:
        return {}


def chaos_to_pct(raw: float, key: str = "field_chaos_score") -> float:
    """混戦度の生値 (正規化エントロピー等、値域が狭い) を過去分布の
    パーセンタイル(0-1)に変換。テーブル欠如時は生値返し。"""
    t = _quantile_tables().get(key)
    if not t or len(t) < 2:
        return float(raw or 0)
    if raw <= t[0]:
        return 0.0
    if raw >= t[-1]:
        return 1.0
    i = bisect.bisect_right(t, raw)
    lo, hi = t[i - 1], t[i]
    frac = 0.0 if hi == lo else (raw - lo) / (hi - lo)
    return (i - 1 + frac) / (len(t) - 1)


# ============================================================
# レースの堅さ
# ============================================================
def classify_hardness(chaos_pct: float | None) -> str:
    """混戦度パーセンタイル → '固い' / '標準' / '荒れ'"""
    if chaos_pct is None:
        return "標準"
    if chaos_pct <= CHAOS_HARD_MAX:
        return "固い"
    if chaos_pct >= CHAOS_ROUGH_MIN:
        return "荒れ"
    return "標準"


# ============================================================
# vs市場 (割安/適正/割高) — 単勝も複勝も同じ ±20% ロジック
# ============================================================
def vs_market_status(model_p: float | None, odds: float | None) -> str:
    """モデル確率 vs 市場 implied(=1/odds)。
    → 'under'(割安/妙味) / 'over'(割高) / 'fair'(適正) / 'unknown'"""
    if not odds or odds <= 0 or not model_p:
        return "unknown"
    market_p = 1.0 / odds
    if model_p >= market_p * (1.0 + VALUE_BAND):
        return "under"
    if model_p <= market_p * (1.0 - VALUE_BAND):
        return "over"
    return "fair"


# ============================================================
# 妙味馬 (買い候補) 抽出
# ============================================================
def _fuku_mid(h: dict) -> float:
    lo = h.get("fuku_odds_low") or 0
    hi = h.get("fuku_odds_high") or 0
    return (lo + hi) / 2.0


def extract_value_horses(horses: list[dict]) -> list[dict]:
    """妙味馬 = 次を全部満たす馬:
        - 単勝妙味=割安 または 複勝妙味=割安
        - 該当する方の EV >= VALUE_EV_MIN
        - 勝率 >= VALUE_PWIN_MIN (テール除外)
    → EV 降順の list[dict]。各要素は umaban/horse_name/p_win/ev_tan/ev_fuku/
      tan_value/fuku_value/sides を持つ。
    """
    out = []
    for h in horses or []:
        p_win = h.get("p_win") or 0
        if p_win < VALUE_PWIN_MIN:
            continue  # テール除外
        tan = h.get("tansho_odds") or 0
        ev_tan = p_win * tan
        fmid = _fuku_mid(h)
        ev_fuku = (h.get("p_sho") or 0) * fmid

        # 単勝妙味: bundle 既存 ai_vs_market を優先 (無ければ計算)
        tan_status = h.get("ai_vs_market") or vs_market_status(p_win, tan)
        # 複勝妙味: 同ロジックを p_sho vs 1/複勝中央値 で
        fuku_status = vs_market_status(h.get("p_sho"), fmid if fmid > 0 else None)

        tan_value = (tan_status == "under" and ev_tan >= VALUE_EV_MIN)
        fuku_value = (fuku_status == "under" and ev_fuku >= VALUE_EV_MIN)
        if not (tan_value or fuku_value):
            continue

        sides = []
        if tan_value:
            sides.append("単勝")
        if fuku_value:
            sides.append("複勝")

        out.append({
            "umaban": h.get("umaban"),
            "horse_name": h.get("horse_name"),
            "p_win": round(float(p_win), 4),
            "ev_tan": round(float(ev_tan), 2),
            "ev_fuku": round(float(ev_fuku), 2),
            "tan_value": tan_value,
            "fuku_value": fuku_value,
            "sides": sides,
        })

    # 割安側のうち良い方の EV で降順ソート
    def _best_ev(v):
        cands = []
        if v["tan_value"]:
            cands.append(v["ev_tan"])
        if v["fuku_value"]:
            cands.append(v["ev_fuku"])
        return max(cands) if cands else 0.0

    out.sort(key=_best_ev, reverse=True)
    return out


# ============================================================
# 買い方判定 (堅さ × 妙味の有無 → 買い方)
# ============================================================
# category は nicegui の _CAT_COLOR と共有 (go=緑/caution=黄/avoid=グレー/danger=赤)
def decide_strategy(hardness: str, has_value: bool) -> dict:
    """(堅さ, 妙味有無) → 買い方判定 dict。
    headline / category / detail / kenshu_hint(券種ヒント) / waku_tag(回顧タグ)。
    """
    if hardness == "固い" and has_value:
        return {
            "headline": "妙味本線（絞って厚く）", "category": "go",
            "detail": "妙味馬を本線に。固いレースなので少点数で厚く。",
            "kenshu_hint": "単勝/複勝/馬連で少点数（絞って厚く）",
            "waku_tag": "妙味枠",
        }
    if hardness == "固い" and not has_value:
        return {
            "headline": "原則見送り（参加可）", "category": "caution",
            "detail": "妙味馬なし。やるなら本命の複勝1点を回収底上げと割り切り、小さく。",
            "kenshu_hint": "本命の複勝1点を小さく（参加なら）",
            "waku_tag": "参加枠",
        }
    if hardness == "標準" and has_value:
        return {
            "headline": "妙味狙い", "category": "go",
            "detail": "妙味馬を中心に通常点数で。",
            "kenshu_hint": "妙味馬中心・通常点数",
            "waku_tag": "妙味枠",
        }
    if hardness == "標準" and not has_value:
        return {
            "headline": "見送り寄り", "category": "avoid",
            "detail": "妙味馬なし・標準。無理に張らない。",
            "kenshu_hint": "原則見送り（買うなら最小限）",
            "waku_tag": None,
        }
    if hardness == "荒れ" and has_value:
        return {
            "headline": "広く薄く", "category": "caution",
            "detail": "妙味馬を複勝/ワイドで複数、1点を小さく（テールは除外済み）。",
            "kenshu_hint": "妙味馬を複勝/ワイドで複数・1点小",
            "waku_tag": "妙味枠",
        }
    # 荒れ × 妙味なし
    return {
        "headline": "見送り", "category": "danger",
        "detail": "荒れ・妙味なし。見送り。",
        "kenshu_hint": "見送り",
        "waku_tag": None,
    }


def build_judgment(race_confidence: dict | None,
                   horses: list[dict] | None) -> dict:
    """レース 1 件分の買い方判定 dict を組み立てる (bundle 埋込 / 表示 共通)。"""
    raw_chaos = (race_confidence or {}).get("field_chaos_score") or 0
    chaos_pct = chaos_to_pct(raw_chaos)
    hardness = classify_hardness(chaos_pct)
    value_horses = extract_value_horses(horses or [])
    has_value = len(value_horses) > 0
    strat = decide_strategy(hardness, has_value)
    return {
        "hardness": hardness,
        "chaos_pct": round(chaos_pct, 3),
        "has_value": has_value,
        "headline": strat["headline"],
        "category": strat["category"],
        "detail": strat["detail"],
        "kenshu_hint": strat["kenshu_hint"],
        "waku_tag": strat["waku_tag"],
        "value_horses": value_horses,
    }
