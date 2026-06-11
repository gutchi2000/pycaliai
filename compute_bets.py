# -*- coding: utf-8 -*-
"""
compute_bets.py — 馬券構築（confidence 駆動・カード連動）。docs/compute_bets_spec.md 実装。
========================================================================
方針（2026-06-09 改訂）: EV を主決定から「配分の重み（従）」に降格。
NiceGUI の 4 カード（市場一致 / ◎独走度 / 上位2頭集中 / 混戦度）と **同じパーセンタイル＋しきい値**
で「買いの形」を決め、その形で **おいしい馬（value_horses）+ ◎〇▲△** の馬券を提示する。
人間はカードを見ながら最終 見送り/狙い を判断する前提。

形（shape）:
  本命勝負  : ◎独走(>=0.75) & 本線濃厚(top2>=0.75) & 固い(chaos<=0.50) → 馬単8点(◎〇→◎〇▲△△)+単複◎
  ◎軸      : ◎やや優位(top1>=0.50)                                   → 単勝◎+馬連◎軸流し+ワイド+複勝
  広め流し  : 拮抗(top1<0.25) or 分散(top2<0.40)                       → 馬連box+ワイドbox+複勝
  カオス薄  : カオス(chaos_pct>0.75, raw<0.92)                         → ワイド◎軸 薄く+複勝
  標準     : 上記以外                                                 → 単勝◎+馬連◎-〇+ワイド◎軸+複勝
  穴overlay : 市場乖離(market<0.30) かつ value_horses 有 → その馬の単勝/ワイドを上乗せ

見送り(hard, §0): raw chaos>=0.92 / field<=7 / ◎オッズ無 / p_win(◎)<0.05。

実行: PYTHONUTF8=1 python compute_bets.py --bundle reports/cowork_input/20260607_bundle.json --dry

⚠️ 未実装 (2026-06-11 監査時点 / docs/audit_20260611.md 参照):
  - 出力は stdout print のみ。spec §書込契約 (cowork_output/{date}_bets.json への in-place merge) は未実装。
  - --dry は引数定義のみで未参照 (print のみなので dry/本実行の区別が無い)。
  - 入力2 (JV-Link T-10 ライブオッズ / jvlink_odds.py の reports/live_odds/) は未配線。
    EV は bundle 埋込オッズ (= 土曜朝 TARGET スナップ) で計算しており、T-10 時点とはズレる。
  - ✅解消済(2026-06-11): ワイド/馬連/馬単の確率は bundle 埋込の pair_probs
    (calibrated PL joint, export_marks_json が印馬10ペアに付与) を優先使用。
    pair_probs の無い旧 bundle のみ独立積近似 (+21〜27% 過大) にフォールバック。
  T-10 運用に投入する前に残りの配線 (ライブオッズ/書込/--dry) が必要。
"""
from __future__ import annotations
import argparse, bisect, io, json, sys
from pathlib import Path

BASE = Path(__file__).parent
BUDGET, MIN_BET, MAX_BET = 10000, 500, 7000
CHAOS_Q = BASE / "data" / "chaos_quantiles.json"

# カード閾値（nicegui_app と一致）
TH_TOP1_GO, TH_TOP1_OK = 0.75, 0.50      # ◎独走 / ◎やや優位
TH_TOP2_GO, TH_TOP2_OK, TH_TOP2_LOW = 0.75, 0.50, 0.40  # 本線濃厚 / やや本線 / 分散
TH_CHAOS_HARD, TH_CHAOS_MID = 0.75, 0.50  # カオス / 混戦（パーセンタイル）
TH_MARKET_ANABA = 0.30                    # 市場乖離→妙味
CHAOS_RAW_SKIP = 0.92                     # §0 hard 見送り（生値）

_QT = None
def _qtab():
    global _QT
    if _QT is None:
        try:
            _QT = json.loads(CHAOS_Q.read_text(encoding="utf-8")).get("quantiles", {})
        except Exception:
            _QT = {}
    return _QT

def pct(raw, key):
    """生値→過去分布パーセンタイル(0-1)。テーブル欠如時は生値（nicegui _to_pct と同一）。"""
    t = _qtab().get(key)
    if not t or len(t) < 2:
        return float(raw or 0)
    raw = float(raw or 0)
    if raw <= t[0]: return 0.0
    if raw >= t[-1]: return 1.0
    i = bisect.bisect_right(t, raw)
    lo, hi = t[i - 1], t[i]
    frac = 0.0 if hi == lo else (raw - lo) / (hi - lo)
    return (i - 1 + frac) / (len(t) - 1)


def _num(x):
    try:
        v = float(x); return v if v == v else None
    except Exception:
        return None


def amount_for_ev(ev):
    if ev >= 1.50: return 5500
    if ev >= 1.20: return 3750
    if ev >= 1.00: return 2250
    if ev >= 0.85: return 1500
    return 900


def allocate(weights, budget=BUDGET, mn=MIN_BET, mx=MAX_BET):
    n = len(weights)
    if n == 0: return []
    s = sum(weights) or 1.0
    amts = [min(mx, max(mn, int(round(budget * w / s / 100)) * 100)) for w in weights]
    for _ in range(6000):
        d = budget - sum(amts)
        if d == 0: break
        step = 100 if d > 0 else -100
        order = sorted(range(n), key=lambda i: (-weights[i] if d > 0 else weights[i]))
        for i in order:
            na = amts[i] + step
            if mn <= na <= mx:
                amts[i] = na; break
        else:
            break
    return amts


def compute_race_bets(race: dict) -> dict:
    rm = race.get("race_meta", {})
    rc = race.get("race_confidence", {})
    bj = race.get("buy_judgment", {})
    horses = race.get("horses", [])
    field = _num(rm.get("field_size")) or len(horses)
    rid = str(race.get("race_id") or rm.get("race_id") or "")
    label = f"{rm.get('place','')}{rm.get('course','')} {rm.get('race_name','')}".strip()
    by_ban = {int(h["umaban"]): h for h in horses if _num(h.get("umaban")) is not None}

    def fld(b, k):
        return _num(by_ban.get(b, {}).get(k)) if b is not None else None

    marks = {}
    for h in horses:
        m = h.get("mark", "")
        if m in ("◎", "〇", "○", "▲", "△"):
            marks.setdefault("○" if m == "〇" else m, []).append(int(h["umaban"]))
    hon = (marks.get("◎") or [None])[0]
    tai = (marks.get("○") or [None])[0]
    san = (marks.get("▲") or [None])[0]
    osae = marks.get("△", [])

    chaos_raw = _num(rc.get("field_chaos_score")) or 0.0
    pwin_hon, tan_hon = fld(hon, "p_win"), fld(hon, "tansho_odds")

    # ---- §0 hard 見送り ----
    skip = None
    if chaos_raw >= CHAOS_RAW_SKIP: skip = f"極度カオス(chaos {chaos_raw:.3f}>=0.92)"
    elif field <= 7: skip = "少頭数(<=7)"
    elif hon is None or tan_hon is None: skip = "本命オッズ未取得"
    elif pwin_hon is not None and pwin_hon < 0.05: skip = "本命勝率<0.05"
    if skip:
        return {"race_id": rid, "race_label": label, "race_nature": "見送り",
                "race_reason": f"{skip} のため見送り。", "bets": []}

    # ---- カード値（パーセンタイル + market 生値）----
    top1 = pct(rc.get("top1_dominance"), "top1_dominance")
    top2 = pct(rc.get("top2_concentration"), "top2_concentration")
    chaos = pct(rc.get("field_chaos_score"), "field_chaos_score")
    market = _num(rc.get("ai_market_agreement")) or 0.0
    anaba = market < TH_MARKET_ANABA
    value_bans = [int(v["umaban"]) for v in bj.get("value_horses", []) if _num(v.get("umaban")) is not None]

    # ---- 形（shape）決定 ----
    if hon and tai and top1 >= TH_TOP1_GO and top2 >= TH_TOP2_GO and chaos <= TH_CHAOS_MID:
        shape = "本命勝負"
    elif hon and top1 >= TH_TOP1_OK:
        shape = "◎軸"
    elif top1 < 0.25 or top2 < TH_TOP2_LOW:
        shape = "広め流し"
    elif chaos > TH_CHAOS_HARD:
        shape = "カオス薄"
    else:
        shape = "標準"

    # ---- EV ヘルパ → 候補 (kind, sel, ev, weight_boost) ----
    um = race.get("umaren_matrix", {})
    def umaren(i, j):
        a, b = sorted((i, j)); return _num(um.get(f"{a}-{b}"))
    # bundle 埋込の正確な PL joint 確率 (calibrated, export_marks_json が印馬10ペアに付与)。
    # あればワイド/馬連/馬単の確率に優先使用。無い旧 bundle は従来近似にフォールバック
    # (ワイド独立積は +21〜27% 系統過大 → docs/audit_20260611.md 🔴)。
    pp = race.get("pair_probs", {}) or {}
    def pair_p(i, j, kind):
        d = pp.get(f"{min(i, j)}-{max(i, j)}")
        if not d:
            return None
        if kind == "umatan":
            return _num((d.get("umatan") or {}).get(f"{i}→{j}"))
        return _num(d.get(kind))
    cands = []  # [kind, sel, bans, odds, ev, ish, boost]
    # オッズ上限（モデルのテール過大評価が EV 経由で大穴を生むのを遮断）
    ODDS_CAP = {"馬連": 100.0, "ワイド": 50.0, "馬単": 200.0}
    def push(kind, sel, bans, odds, ev, boost=1.0):
        if not odds or ev is None: return
        cap = ODDS_CAP.get(kind)
        if cap and odds > cap: return
        # 同一 (kind, sel) の重複統合: 穴 overlay (value_horses) と shape 側が
        # 同じ買い目 (例 vb==◎ の単勝/複勝) を生成し得る。dedup しないと同一馬券に
        # 2 行・別金額が出て露出が最大 2×MAX_BET になり、点数上限 cap も浪費する。
        # boost は強い方を採用。
        for c in cands:
            if c[0] == kind and c[1] == sel:
                if boost > c[6]:
                    c[6] = boost
                return
        cands.append([kind, sel, tuple(bans), odds, ev, (hon in bans), boost])
    def c_tan(b, boost=1.0):
        o, pw = fld(b, "tansho_odds"), fld(b, "p_win")
        if o and pw: push("単勝", str(b), (b,), o, pw * o, boost)
    def c_fuku(b, boost=1.0):
        lo, hi, ps = fld(b, "fuku_odds_low"), fld(b, "fuku_odds_high"), fld(b, "p_sho")
        if lo and hi and ps: push("複勝", str(b), (b,), (lo + hi) / 2, ps * (lo + hi) / 2, boost)
    def c_umaren(i, j, boost=1.0):
        o = umaren(i, j)
        if not o:
            return
        p = pair_p(i, j, "umaren")
        if p is None:  # 旧 bundle 互換: Harville 風近似
            pi, pj = fld(i, "p_win"), fld(j, "p_win")
            if not (pi and pj and 0 < pi < 1 and 0 < pj < 1):
                return
            p = pi * pj / (1 - pi) + pj * pi / (1 - pj)
        push("馬連", f"{min(i,j)}-{max(i,j)}", (i, j), o, o * p, boost)
    def c_wide(i, j, boost=1.0):
        o = umaren(i, j)
        if not o:
            return
        p = pair_p(i, j, "wide")
        if p is None:  # 旧 bundle 互換: 独立積近似 (+21〜27% 系統過大)
            si, sj = fld(i, "p_sho"), fld(j, "p_sho")
            if not (si and sj):
                return
            p = si * sj
        push("ワイド", f"{min(i,j)}-{max(i,j)}", (i, j), o / 3.0, (o / 3.0) * p, boost)
    def c_umatan(i, j, boost=1.0):  # i→j (i 1着, j 2着)。馬単odds ≈ 馬連 × (pi+pj)/pi
        o, pi, pj = umaren(i, j), fld(i, "p_win"), fld(j, "p_win")
        if not (o and pi and pj and 0 < pi < 1):
            return
        ut = o * (pi + pj) / pi
        p = pair_p(i, j, "umatan")
        if p is None:  # 旧 bundle 互換
            p = pi * pj / (1 - pi)
        push("馬単", f"{i}→{j}", (i, j), ut, ut * p, boost)

    rel = [x for x in (tai, san, *osae) if x]            # 相手(〇▲△)
    box4 = [b for b in (hon, tai, san, *osae) if b][:4]  # 上位4

    # ---- 形ごとに候補生成（おいしい馬 boost）----
    # ★馬連は全廃（実績ROI最弱56%）。◎独走系は 馬単＋単勝、◎弱は ワイド流し。
    if shape == "本命勝負" and hon and tai:
        tgt = [b for b in (hon, tai, san, *osae) if b]    # ◎〇▲△△
        for a in (hon, tai):
            for t in tgt:
                if a != t: c_umatan(a, t, 1.3)            # 馬単8点 formation
        c_tan(hon, 1.6); c_fuku(hon, 1.1)                 # 単勝◎ 厚
    elif shape == "◎軸" and hon:
        c_tan(hon, 1.6)                                   # 単勝◎（最重視・実績120%）
        if (fld(hon, "tansho_odds") or 99) <= 15:         # ◎が実本命の時だけ馬単
            for r in rel: c_umatan(hon, r, 1.2)           # 馬単 ◎→相手（◎1着固定流し）
        for r in (tai, san, *osae[:1]):
            if r: c_wide(hon, r, 1.1)                      # 人気薄◎は単勝＋ワイドに寄せる
        c_fuku(hon, 1.0)
    elif shape == "広め流し":                              # ◎弱 → ワイド流しのみ（馬単/馬連なし）
        for r in rel:
            if hon: c_wide(hon, r, 1.2)
        if tai:
            for r in [x for x in (hon, san, *osae) if x and x != tai]:
                c_wide(tai, r, 1.0)
        if hon: c_fuku(hon, 1.1)
    elif shape == "カオス薄":
        if hon:
            for r in rel: c_wide(hon, r, 1.0)
            c_fuku(hon, 1.2)
    else:  # 標準
        if hon:
            c_tan(hon, 1.4); c_fuku(hon, 1.1)
            for r in (tai, san, *osae):
                if r: c_wide(hon, r, 1.1)

    # ---- 穴 overlay: おいしい馬（value_horses）を上乗せ ----
    if anaba:
        for vb in value_bans:
            c_tan(vb, 1.3); c_wide(vb, hon, 1.2) if hon and vb != hon else None
            c_fuku(vb, 1.1)

    # ---- ソフトEVフロア（明確な-EVのみ除外。本命勝負の馬単は formation 全採用）----
    floor = 0.70 if shape == "本命勝負" else 0.80
    chosen = [c for c in cands if c[4] >= floor or (shape == "本命勝負" and c[0] == "馬単")]
    # ◎必須
    if hon and not any(c[5] for c in chosen):
        hc = [c for c in cands if c[5]]
        if hc: chosen.append(max(hc, key=lambda c: c[4]))
    if not chosen and cands:
        chosen = [max(cands, key=lambda c: c[4])]
    # 点数上限（馬単formationは8、その他は6目安）
    cap = 8 if shape == "本命勝負" else 6
    chosen = sorted(chosen, key=lambda c: -(c[4] * c[6]))[:cap]

    # ---- 配分（EV×boost 重み、キャップ厳守）----
    base = [amount_for_ev(c[4]) * c[6] for c in chosen]
    amts = allocate(base)
    waku = bj.get("waku_tag") or "参加枠"
    bets = []
    for c, amt in zip(chosen, amts):
        kind, sel, bans, odds, ev, ish, boost = c
        role = "おいしい" if (set(bans) & set(value_bans)) else ("◎絡み" if ish else "相手")
        bets.append({"馬券種": kind, "買い目": sel, "購入額": int(amt), "枠タグ": waku,
                     "理由": f"{role}（{kind} {odds:.1f}倍）"})

    hon_name = by_ban.get(hon, {}).get("horse_name", "本命")
    rr = f"◎{hon_name}。{shape}（独走{top1:.2f}/集中{top2:.2f}/混戦{chaos:.2f}/市場{market:+.2f}）で {len(bets)}点。"
    return {"race_id": rid, "race_label": label, "race_nature": shape, "race_reason": rr,
            "confidence": {"top1_pct": round(top1, 3), "top2_pct": round(top2, 3),
                           "chaos_pct": round(chaos, 3), "market": round(market, 3)},
            "bets": bets}


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    d = json.load(open(args.bundle, encoding="utf-8"))
    races = d.get("races", d if isinstance(d, list) else [])
    out = [compute_race_bets(r) for r in races]
    n_bet = sum(1 for e in out if e["bets"]); tot = sum(b["購入額"] for e in out for b in e["bets"])
    shapes = {}
    for e in out: shapes[e["race_nature"]] = shapes.get(e["race_nature"], 0) + 1
    print(f"[compute_bets] races={len(out)} 買い={n_bet} 見送り={len(out)-n_bet} 総額¥{tot:,}  形:{shapes}")
    print("=" * 92)
    for e in out:
        if e["bets"]:
            s = sum(b['購入額'] for b in e['bets'])
            print(f"\n● {e['race_label']} [{e['race_nature']}] {len(e['bets'])}点 ¥{s:,}  {e['race_reason'].split('。',1)[-1]}")
            for b in e["bets"]:
                print(f"   {b['馬券種']:3s} {b['買い目']:8s} ¥{b['購入額']:>5,}  {b['理由']}")
        else:
            print(f"\n— {e['race_label']} [見送り] {e['race_reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
