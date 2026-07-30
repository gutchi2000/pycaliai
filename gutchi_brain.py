# -*- coding: utf-8 -*-
"""
gutchi_brain.py — 決定論ポリシーエンジン (docs/gutchi_brain_policy.md の券種決定木)。
================================================================================
入力 : bundle の 1レース (horses=印/ai_score/p_win/tansho_odds/fuku_odds/history)
        + sat_bias (土曜の実現バイアス realized_bias["venues"]) は 逆風フラグ/記号ゲートに使用
出力 : 買い目 list [{"馬券種","買い目","購入額","理由"}]

決定木 (2026-06-29 較正 / 2026-07-31 監査修正):
  0. 見送り: 新馬 / 超混戦 chaos>=0.92 (★真の参加ゲート=構造分岐に関係なく冒頭で判定)
  1. バイアス: スコア再重み(旧 apply_bias)は撤去済み。クロスデイ馬場バイアスの馬券効果は
     実測ゼロ ([[project_crossday_trackbias_dead]])。生き残りは
       - 逆風フラグ (◎後脚質 on 前残り日) → 券種形を軽く (三連複切り・ワイド広め)
       - バイアス記号ゲート (bias_mode=True 時のみ、★レース限定参加)
     の2つ。クラスター/構造判定は常に生AI指数で行う。
  2. ◎非分離(◎-〇 gap小) + 団子 → 上位N頭BOX(ワイド=安全/三連複=上振れ)、軸ナシ
  3. ◎が抜けてる →
       相手なし(◎-〇 > SECOND_WALL)     → 複勝1点 (◎妙味帯のみ単複)
       2頭(◎-〇近い)                     → 馬連+ワイド (2頭圧倒的なら ◎単勝+複勝両頭の保険)
       3頭+ (▲が BOX_WALL 内)            → 馬連流し + ワイドBOX(上位3) + 三連複1点
                                            ◎逆風なら 三連複カット + ワイド流し広め(軽め)
  少頭数(≤5)×◎確勝 → 三連複(本線=◎込み + 上位4頭BOX)  ※三連単は禁止則で不使用
  L3 規律層 (_discipline): トリガミ床(三連複の適応点数削り) + chalk-cap(堅い回は12点まで)
取消: scratched で除外。 馬連=近い壁(WALL) / ワイドBOX=広い壁(BOX_WALL)。
"""
from __future__ import annotations
from itertools import combinations, permutations

BRAIN_VERSION = "0.2.0"   # 2026-07-31: chaos真見送り/未賭バグ修正/L3規律/apply_bias撤去

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
# ⚠ WALL/BOX_WALL/NO_AXIS_GAP/SECOND_WALL は per-race min-max 正規化指数上の絶対カット
#   =スコアの散らばりで同じ着順順位でも券種形が変わる(2026-07-06 監査#4)。散らばり=確信の
#   代理でもあるため v0.2 では仕様として維持。動かす時は simulate_brain_day で決済検証必須。

# ---- L3 規律層 ([[project_trigami_floor_rule_validated]] の生き残り2規律) ----
CHAOS_SKIP = 0.92    # 参加ゲート: field_chaos_score >= これ = 超混戦 → 真の見送り
CHALK_CHAOS = 0.86   # chalk-cap: chaos < これ(堅い) なら総点数を CHALK_CAP に制限
CHALK_CAP = 12
TRIO_TAKEOUT = 0.775 # 三連複還元率。トリガミ床 kstar = ⌊0.775/p_fav⌋


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


def _fav_axes(vb):
    """Step0: realized_bias vb → (fav_leg, fav_side)。fav_leg∈{前,後,None}, fav_side∈{内,外,None}。"""
    fr = vb.get("front_rate")
    fav_leg = ("前" if (fr is not None and fr >= 0.6)
               else "後" if (fr is not None and fr <= 0.40)
               else None)                       # 0.4-0.6 = 脚質中立
    wk = vb.get("waku")
    return fav_leg, (wk if wk in ("内", "外") else None)


def _waku_fav(u, fs, fav_side):
    """Step1: 枠が有利側か。ユーザー閾値 内≦0.5 / 外≧0.5(馬番÷頭数)。"""
    if not fav_side:
        return False
    pos = (u - 1) / (fs - 1) if fs and fs > 1 else 0.5
    return pos <= 0.5 if fav_side == "内" else pos >= 0.5


def is_star(h, fs, fav_leg, fav_side):
    """Step2 ★: 立ってる軸を「全部」満たす。両立場(内/外×前残り)=脚質∧枠 /
    前残りのみ(枠フラット)=脚質だけ / 枠のみ(脚質中立)=枠だけ。バイアス無=★不成立。"""
    if fav_leg is None and fav_side is None:
        return False
    if fav_leg is not None and leg(h) != fav_leg:
        return False
    if fav_side is not None and not _waku_fav(h["umaban"], fs, fav_side):
        return False
    return True


def bias_symbol(horses, a_h, vb, fs):
    """Step3: レース記号 ★/⚠/無印 と ⚠時の振替軸馬番(★の〇▲)。ほぼ◎で決まる。"""
    if not vb:
        return "無印", None
    fav_leg, fav_side = _fav_axes(vb)
    if fav_leg is None and fav_side is None:
        return "無印", None                       # 場が均等(前提なし)
    if is_star(a_h, fs, fav_leg, fav_side):
        return "★", None                          # ◎が★ → ◎本線・軸OK
    lg = leg(a_h)
    pos = (a_h["umaban"] - 1) / (fs - 1) if fs and fs > 1 else 0.5
    leg_bad = bool(fav_leg) and lg is not None and lg != fav_leg
    waku_bad = (fav_side == "内" and pos > 0.5) or (fav_side == "外" and pos < 0.5)
    if leg_bad or waku_bad:                        # ◎が後/枠逆 → ⚠(★の〇▲へ軸振替)
        for h in sorted(horses, key=lambda x: -(_num(x.get("ai_score")) or 0)):
            if h.get("mark") in ("〇", "▲") and is_star(h, fs, fav_leg, fav_side):
                return "⚠", h["umaban"]
        return "⚠", None                          # 受け皿(★の〇▲)なし
    return "無印", None                            # ◎が中/脚質不明 → AI印のまま軽め


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


def _trio_prob(trio, p):
    """Harville: 正規化 p_win から三連複組(3頭が上位3着)の確率 = 6順列の和。"""
    s = 0.0
    for x, y, z in permutations(trio):
        d1 = 1.0 - p[x]
        d2 = d1 - p[y]
        if d1 <= 1e-9 or d2 <= 1e-9:
            continue
        s += p[x] * (p[y] / d1) * (p[z] / d2)
    return s


def _discipline(bets, horses, chaos):
    """L3 規律層 (2026-07-31 配線)。
    (a) トリガミ床(適応点数版): 三連複ブロックを kstar=⌊0.775/p_fav⌋ 点まで削る。
        p_fav=最有力組の model 確率、0.775/p_fav=その組の market 払戻 proxy。
        「最有力組が当たれば必ずブロックの元が取れる」点数に絞る=見送りでなく点数削り
        (analysis/trigami_floor_gate.py の適応版と同型。model-implied proxy=serve一致)。
    (b) chalk-cap: 堅いレース(chaos<0.86)は総点数 12 まで。低確率トリオ→末尾ワイドの順に落とす。
    """
    trios = [b for b in bets if b["馬券種"] == "三連複"]
    if not trios and (chaos >= CHALK_CHAOS or len(bets) <= CHALK_CAP):
        return bets
    tot = sum((_num(h.get("p_win")) or 0.0) for h in horses) or 1.0
    p = {h["umaban"]: (_num(h.get("p_win")) or 0.0) / tot for h in horses}
    pr = {}
    for b in trios:
        trio = [int(x) for x in b["買い目"].split("-")]
        pr[id(b)] = _trio_prob(trio, p) if all(u in p for u in trio) else 0.0
    if len(trios) > 1:
        p_fav = max(pr.values())
        kstar = max(1, int(TRIO_TAKEOUT / p_fav)) if p_fav > 0 else len(trios)
        if kstar < len(trios):
            keep = {id(b) for b in sorted(trios, key=lambda b: -pr[id(b)])[:kstar]}
            bets = [b for b in bets if b["馬券種"] != "三連複" or id(b) in keep]
    if chaos < CHALK_CHAOS and len(bets) > CHALK_CAP:
        drop = len(bets) - CHALK_CAP
        order = sorted([b for b in bets if b["馬券種"] == "三連複"],
                       key=lambda b: pr.get(id(b), 0.0))
        order += [b for b in reversed(bets) if b["馬券種"] == "ワイド"]
        kill = {id(b) for b in order[:drop]}
        bets = [b for b in bets if id(b) not in kill]
    return bets


def build_tickets(race, sat_bias=None, scratched=None, budget=BUDGET, bias_mode=False,
                  chaos_gate=True, discipline=True):
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

    chaos = _num((race.get("race_confidence") or {}).get("field_chaos_score")) or 0.5
    # === 参加ゲート (2026-07-31 監査#3): 超混戦は構造分岐に関係なく真の見送り ===
    #   旧実装は ◎非分離枝の内側にネストし、◎が抜けた高chaosレースでは発火しなかった。
    if chaos_gate and chaos >= CHAOS_SKIP:
        return []

    # === #1 バイアス ===
    # ★構造(混戦/pair/trio)は「生AI指数」で判定する。スコア再重み(旧 apply_bias)は
    #   クロスデイ馬場バイアスの馬券効果が実測ゼロにつき撤去(2026-07-31)。
    #   bias は 逆風フラグ / bias_mode 記号ゲート にだけ使う。
    vb = (sat_bias or {}).get(f"{m.get('place')}|{surf}") if sat_bias else None

    idx = ai_index(horses)
    ranked = sorted(horses, key=lambda h: -idx[h["umaban"]])
    a_h = next((h for h in horses if h.get("mark") == "◎"), ranked[0])

    # === バイアス記号(opt-in): ★(◎が有利脚質×有利枠)のレースだけ参加。⚠/無印/バイアス元なし=見送り ===
    if bias_mode and (not vb or bias_symbol(horses, a_h, vb, fs)[0] != "★"):
        return []

    bets = _shape(race, horses, m, fs, idx, ranked, a_h, vb, budget)
    if discipline and bets:
        bets = _discipline(bets, horses, chaos)
    return bets


def _shape(race, horses, m, fs, idx, ranked, a_h, vb, budget):
    """券種決定木の本体 (参加ゲート通過後)。"""
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

    # === 少頭数(≤5)×◎確勝級 → 三連複(本線=◎込み + 上位4頭BOX)。三連単は禁止則で不使用 ===
    if fs <= SMALL_FIELD and (_num(a_h.get("p_win")) or 0) >= 0.30:
        # ◎を必ず本線に含める (旧: p_win top3 で◎不在化しえた=監査#4)
        top = [a] + [h["umaban"] for h in
                     sorted(horses, key=lambda h: -(_num(h.get("p_win")) or 0))
                     if h["umaban"] != a]
        t3 = sorted(top[:3])
        extra = [sorted(c) for c in combinations(sorted(top[:4]), 3) if sorted(c) != t3]
        if not extra:                                             # 3頭立て: 本線1点に全額 (旧: ¥6k未賭)
            return [_t("三連複", "-".join(map(str, t3)), budget, "少頭数×◎確勝の本線")]
        hon = budget * 4 // 10                                    # 本線に4割
        per = max(100, (budget - hon) // len(extra) // 100 * 100)
        bets = [_t("三連複", "-".join(map(str, t3)), hon, "少頭数×◎確勝の本線")]
        bets += [_t("三連複", "-".join(map(str, c)), per, "上位4頭BOX(取りこぼし)") for c in extra]
        return bets

    # === #4/#5 ◎非分離(◎-〇 gap小) + 塊 → 混戦 → 上位N頭BOX (超混戦は冒頭ゲートで見送り済) ===
    if g_ow <= NO_AXIS_GAP and len(clS) >= 3:
        N = sorted(clS[:5])
        nw = len(N) * (len(N) - 1) // 2
        nt = len(N) * (len(N) - 1) * (len(N) - 2) // 6
        if nw + nt <= budget // 1000:
            per = max(100, (budget // (nw + nt)) // 100 * 100)    # 予算を使い切る (旧: 固定¥1000でN=3時¥6k未賭)
            return (_box_wide(N, per, f"上位{len(N)}頭混戦ワイドBOX")
                    + _box_trio(N, per, "混戦三連複BOX(上振れ)"))
        per = max(100, (budget // max(1, nw)) // 100 * 100)
        return _box_wide(N, per, f"上位{len(N)}頭混戦ワイドBOX(安全)")

    # === #3 逆風(◎後 on 前残り日) → ◎-印上位ワイド流し広め + ◎-tight馬連 (三連複切り・軽め) ===
    if gyaku and sum(1 for u in hod0 if hod0[u] <= ANA_ODDS) >= 2:
        rel4 = [h["umaban"] for h in others if hod0[h["umaban"]] <= ANA_ODDS][:4]
        n_u = min(2, len(rel4))
        unit = budget // (2 * len(rel4) + n_u)                    # ワイド:馬連=2:1 で予算按分 (旧: 相手<4で¥4-6k未賭)
        w_amt = max(100, 2 * unit // 100 * 100)
        u_amt = max(100, unit // 100 * 100)
        rest = budget - w_amt * len(rel4) - u_amt * n_u
        bets = [_t("ワイド", _pair(a, r),
                   w_amt + (rest // 100 * 100 if i == 0 and rest >= 100 else 0),
                   "逆風◎軸ワイド流し") for i, r in enumerate(rel4)]
        for r in rel4[:n_u]:                   # ◎-〇▲ 馬連 (印上位2頭)
            bets.append(_t("馬連", _pair(a, r), u_amt, "逆風◎-〇▲馬連"))
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
    # tight 相手が減った分はワイド基軸に載せて予算を使い切る (旧: tight=0 で¥2k未賭)
    w_per = max(100, (budget - 1000 * len(tight_rel) - 2000) // 3 // 100 * 100)
    rest = budget - 1000 * len(tight_rel) - 2000 - w_per * 3
    bets = _box_wide(box, w_per, "上位3頭ワイドBOX(基軸)")
    if rest >= 100:
        bets[0]["購入額"] += rest // 100 * 100
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
