# -*- coding: utf-8 -*-
"""
gutchi_brain.py — 決定論ポリシーエンジン (docs/gutchi_brain_policy.md の券種決定木)。
================================================================================
入力 : bundle の 1レース (horses=印/ai_score/p_win/tansho_odds/fuku_odds/history)
        + sat_bias (土曜の実現バイアス realized_bias["venues"]) で 枠×脚質 再重み
出力 : 買い目 list [{"馬券種","買い目","購入額","理由"}]

決定木 (2026-06-29, 6/21 函館12R ヒアリングで較正):
  0. 見送り: 新馬 / 超混戦(団子が印以外まで続く) / ◎が逆風で代替も無し
  1. バイアス再重み: 前残り日は 前脚質↑/後↓、内有利日は 内枠↑/外↓ で ai_score を±
  2. ◎非分離(◎-〇 gap小) + 団子 → 上位N頭BOX(ワイド=安全/三連複=上振れ)、軸ナシ
  3. ◎が抜けてる →
       相手なし(◎-〇 > SECOND_WALL)     → 複勝1点 (◎妙味帯のみ単複)
       2頭(◎-〇近い)                     → 馬連+ワイド (2頭圧倒的なら ◎単勝+複勝両頭の保険)
       3頭+ (▲が BOX_WALL 内)            → 馬連流し + ワイドBOX(上位3) + 三連複1点
                                            ◎逆風なら 三連複カット + ワイド流し広め(軽め)
  少頭数(≤5)×◎確勝 → 三連複 + 三連単box
取消: scratched で除外。 馬連=近い壁(WALL) / ワイドBOX=広い壁(BOX_WALL)。
"""
from __future__ import annotations
import statistics as _st
from itertools import combinations

# ---- パラメタ (6/21 函館 較正) ----
WALL = 15          # 馬連/三連複の「近い相手」壁(pt)
BOX_WALL = 22      # ワイドBOX/トリオ判定の壁(pt)。馬連より広い (函3R ▲19pt下でもbox)
NO_AXIS_GAP = 7    # ◎-〇 gap がこれ以下 = ◎非分離 → 軸立てず上位N頭BOX (函1/2R)
SECOND_WALL = 22   # ◎-〇 gap がこれ超 = 相手なし → 複勝1点 (函9R 35pt)
ANA_FUKU = 10.0    # 相手の単オッズがこれ超は穴 → 馬連でなく複勝(妙味複勝)
ANA_ODDS = 20.0    # これ超は穴すぎ = 相手に数えない
MARKET_TRUST = 3.0 # これ以下の人気馬は AI gap 無視で相手に残す
FUKU_FLOOR = 1.5   # 単複の複勝倍率床
TANSEN_LO, TANSEN_HI = 5.0, 7.5   # ◎単勝add の妙味帯(オッズ)
SMALL_FIELD = 5    # 少頭数 (三連単解禁)
BUDGET = 10000


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def ai_index(horses):
    sc = {h["umaban"]: (_num(h.get("ai_score")) or 0.0) for h in horses}
    lo, hi = min(sc.values()), max(sc.values())
    if hi <= lo:
        return {u: 0.0 for u in sc}
    return {u: 100.0 * (v - lo) / (hi - lo) for u, v in sc.items()}


def fuku_mid(h):
    lo, hi = _num(h.get("fuku_odds_low")), _num(h.get("fuku_odds_high"))
    return (lo + hi) / 2.0 if (lo and hi) else None


STYLE_MAP = {"逃げ": "逃げ", "先行": "先行", "中団": "差し", "差し": "差し",
             "ﾏｸﾘ": "差し", "マクリ": "差し", "後方": "追込", "追込": "追込"}
STYLE_ORDER = ["逃げ", "先行", "差し", "追込"]


def classify_style(history):      # build_site と同じ (新schema=style / 旧=weight_change)
    runs = (history or {}).get("runs") or []
    score = {}
    for u in runs:
        st = (STYLE_MAP.get(str(u.get("style", "")).strip())
              or STYLE_MAP.get(str(u.get("weight_change", "")).strip()))
        if not st:
            continue
        score[st] = score.get(st, 0) + max(1, 6 - (u.get("n_ago") or 5))
    if not score:
        return None
    best = max(score.values())
    for st in STYLE_ORDER:
        if score.get(st) == best:
            return st
    return None


def leg(h):                       # 脚質 前(逃先)/後(差追)。classify_style 基準(avg_pos でない)
    st = classify_style(h.get("history"))
    return None if st is None else ("前" if st in ("逃げ", "先行") else "後")


def waku(u, fs):                  # 内/中/外
    nd = (u - 1) / (fs - 1) if fs and fs > 1 else 0.5
    return "内" if nd <= 0.33 else "外" if nd >= 0.67 else "中"


def apply_bias(horses, vb, fs):
    """土曜バイアス vb で 枠×脚質 適合を ai_score に ±。AI主・bias従(0.3σ/0.15σ)。"""
    if not vb:
        return
    sc = [_num(h.get("ai_score")) or 0 for h in horses]
    sd = _st.pstdev(sc) or 1.0
    front = vb.get("front_rate", 0) >= 0.6
    wk = vb.get("waku")
    for h in horses:
        d = 0.0
        lg = leg(h)
        if front and lg == "前":
            d += 0.30 * sd
        elif front and lg == "後":
            d -= 0.30 * sd
        w = waku(h["umaban"], fs)
        if wk == "内":
            d += 0.15 * sd if w == "内" else (-0.15 * sd if w == "外" else 0)
        elif wk == "外":
            d += 0.15 * sd if w == "外" else (-0.15 * sd if w == "内" else 0)
        h["ai_score"] = (_num(h.get("ai_score")) or 0) + d


def _t(kind, sel, yen, why):
    return {"馬券種": kind, "買い目": sel, "購入額": int(yen), "理由": why}


def _pair(a, b):
    return f"{min(a, b)}-{max(a, b)}"


def _box_wide(members, per, why):
    return [_t("ワイド", _pair(members[i], members[j]), per, why)
            for i in range(len(members)) for j in range(i + 1, len(members))]


def _box_trio(members, per, why):
    return [_t("三連複", "-".join(map(str, sorted(c))), per, why)
            for c in combinations(members, 3)]


def build_tickets(race, sat_bias=None, scratched=None, budget=BUDGET):
    scratched = scratched or set()
    horses = [h for h in race.get("horses", [])
              if _num(h.get("tansho_odds")) and h.get("umaban") not in scratched]
    if not horses:
        return []
    m = race.get("race_meta", {})
    klass = m.get("class", "") or ""
    fs = m.get("field_size") or len(horses)
    surf = "芝" if (m.get("course", "") or "")[:1] == "芝" else "ダ"

    # === #5 見送り: 新馬 ===
    if "新馬" in klass:
        return []

    # === #1 バイアス ===
    # ★構造(混戦/pair/trio)は「生AI指数」で判定する。bias でスコアは動かさない
    #   (動かすと函1/3/8R の混戦判定が壊れる)。bias は逆風フラグ/相手の強調にだけ使う。
    vb = (sat_bias or {}).get(f"{m.get('place')}|{surf}") if sat_bias else None

    idx = ai_index(horses)
    ranked = sorted(horses, key=lambda h: -idx[h["umaban"]])
    a_h = next((h for h in horses if h.get("mark") == "◎"), ranked[0])
    a = a_h["umaban"]
    others = [h for h in ranked if h["umaban"] != a]
    g_ow = idx[a] - idx[others[0]["umaban"]] if others else 99

    od = lambda h: _num(h.get("tansho_odds")) or 999

    # クラスター: 連続落差で2種 (混戦/見送り用=タイト8pt, ワイドbox用=広い BOX_WALL)
    def grow(thr):
        c = [ranked[0]["umaban"]]
        for prev, cur in zip(ranked, ranked[1:]):
            if idx[prev["umaban"]] - idx[cur["umaban"]] <= thr:
                c.append(cur["umaban"])
            else:
                break
        return c
    clS = grow(13)           # 混戦/超混戦見送り判定 (◎が抜けてないか・印の塊)
    clB = grow(BOX_WALL)     # ワイドBOX/トリオの相手 (馬連より広い壁)
    gyaku = bool(vb) and vb.get("front_rate", 0) >= 0.6 and leg(a_h) == "後"  # 逆風(◎後脚質 on 前残り日)
    hod0 = {h["umaban"]: (od(h)) for h in others}

    # === 少頭数(≤5)×◎確勝級 → 三連複 + 三連単box ===
    if fs <= SMALL_FIELD and (_num(a_h.get("p_win")) or 0) >= 0.30:
        t3 = sorted([h["umaban"] for h in sorted(horses, key=lambda h: -(_num(h.get("p_win")) or 0))[:3]])
        return [_t("三連複", "-".join(map(str, t3)), 4000, "少頭数×◎確勝の本線"),
                _t("三連単", f"{t3[0]}-{t3[1]}-{t3[2]}(box)", 6000, "少頭数の一撃box")]

    chaos = _num((race.get("race_confidence") or {}).get("field_chaos_score")) or 0.5

    # === #4/#5 ◎非分離(◎-〇 gap小) + 塊 → 混戦。chaos≥0.92(印以外も競合)=見送り / 未満=上位N頭BOX ===
    if g_ow <= NO_AXIS_GAP and len(clS) >= 3:
        if chaos >= 0.92:                     # 超混戦(函8R 0.94)→見送り
            return []
        N = sorted(clS[:5])
        nw = len(N) * (len(N) - 1) // 2
        nt = len(N) * (len(N) - 1) * (len(N) - 2) // 6
        if nw + nt <= budget // 1000:
            return _box_wide(N, 1000, f"上位{len(N)}頭混戦ワイドBOX") + _box_trio(N, 1000, "混戦三連複BOX(上振れ)")
        per = max(100, (budget // max(1, nw)) // 100 * 100)
        return _box_wide(N, per, f"上位{len(N)}頭混戦ワイドBOX(安全)")

    # === #3 逆風(◎後 on 前残り日) → ◎-印上位ワイド流し広め + ◎-tight馬連 (三連複切り・軽め) ===
    if gyaku and sum(1 for u in hod0 if hod0[u] <= ANA_ODDS) >= 2:
        rel4 = [h["umaban"] for h in others if hod0[h["umaban"]] <= ANA_ODDS][:4]
        bets = [_t("ワイド", _pair(a, r), 2000, "逆風◎軸ワイド流し") for r in rel4]
        for r in rel4[:2]:                     # ◎-〇▲ 馬連 (印上位2頭)
            bets.append(_t("馬連", _pair(a, r), 1000, "逆風◎-〇▲馬連"))
        return bets

    # === ◎が抜けてる ===
    second = others[0] if others else None
    # 相手なし (◎-〇 が SECOND_WALL 超 & 〇が市場信頼も無し) → 複勝1点 / 単複
    if not (second and (g_ow <= SECOND_WALL or od(second) <= MARKET_TRUST)):
        oa, fm = od(a_h), fuku_mid(a_h)
        if fm and fm >= FUKU_FLOOR and TANSEN_LO <= oa <= TANSEN_HI:
            return [_t("単勝", str(a), 3000, f"◎妙味(複{fm:.1f})"), _t("複勝", str(a), 7000, "◎複勝")]
        return [_t("複勝", str(a), 10000, "相手なし・◎複勝1点")]

    hod = {h["umaban"]: od(h) for h in others}
    box_rel = [h["umaban"] for h in others                                    # ワイドbox相手(広い壁 or 市場信頼)
               if (idx[a] - idx[h["umaban"]] <= BOX_WALL or hod[h["umaban"]] <= MARKET_TRUST)
               and hod[h["umaban"]] <= ANA_ODDS][:2]
    tight_rel = [u for u in box_rel if idx[a] - idx[u] <= WALL or hod[u] <= MARKET_TRUST][:2]
    if not box_rel:                                                           # 相手が穴すぎ → ◎複勝1点
        return [_t("複勝", str(a), 10000, "相手穴すぎ・◎複勝1点")]
    b0 = box_rel[0]

    # === #2 上位2頭圧倒 (◎〇とも高指数&人気, ▲が落ちる) → 馬連+ワイドに ◎単+複両頭の保険 ===
    if b0 and idx[a] >= 85 and idx[b0] >= 85 and od(a_h) <= 5 and hod[b0] <= 6 \
            and (len(box_rel) < 2 or idx[box_rel[1]] < 80):
        return [_t("単勝", str(a), 1000, "圧倒2頭・◎単"), _t("複勝", str(a), 3000, "◎複(0回避)"),
                _t("複勝", str(b0), 2000, "〇複(0回避)"), _t("ワイド", _pair(a, b0), 3000, "2頭ワイド"),
                _t("馬連", _pair(a, b0), 1000, "2頭馬連")]

    # === 2頭 (box相手1頭) → 馬連+ワイド (相手も穴なら妙味複勝×2) ===
    if len(box_rel) <= 1:
        b = b0
        if od(a_h) > ANA_FUKU and hod[b] > ANA_FUKU:
            return [_t("複勝", str(a), 4500, "穴◎妙味複勝"), _t("複勝", str(b), 3500, "穴〇妙味複勝"),
                    _t("ワイド", _pair(a, b), 2000, "2頭upside")]
        return [_t("馬連", _pair(a, b), 4000, "2頭・一撃"), _t("ワイド", _pair(a, b), 6000, "2頭・安定")]

    # === 3頭+ → ワイドBOX(上位3) + 馬連流し(◎-tight) + 三連複1点 ===
    box = sorted([a] + box_rel)
    bets = _box_wide(box, 2000, "上位3頭ワイドBOX(基軸)")
    for r in tight_rel:
        bets.append(_t("馬連", _pair(a, r), 1000, "◎軸流し"))
    bets.append(_t("三連複", "-".join(map(str, box)), 2000, "3頭1点"))
    return bets


def _main():
    import json, sys
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    b = json.load(open(args[0], encoding="utf-8"))
    races = b["races"]; races = races if isinstance(races, list) else list(races.values())
    for r in races:
        ts = build_tickets(r)
        m = r.get("race_meta", {})
        lab = f"{m.get('place')}{str(r.get('race_id'))[-2:]}R"
        print(f"\n== {lab} {m.get('course')} {m.get('field_size')}頭 ==" + ("  見送り" if not ts else ""))
        for t in ts:
            print(f"   {t['馬券種']:4s}{t['買い目']:12s} ¥{t['購入額']:>5,} {t['理由']}")


if __name__ == "__main__":
    _main()
