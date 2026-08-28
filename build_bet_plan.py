# -*- coding: utf-8 -*-
"""
build_bet_plan.py — 前日の「枠」アロケータ (勝負 / 準勝負 / 消化)
================================================================================
bundle の race_confidence から、レースを信頼度で3枠に自動編成し各レース予算を割る。
T-10 ライン(compute_bets)が各レースの枠予算内で買い目を生成する(買い目/金額は直前)。

ルール(ユーザー指定, 2026-06-19):
  禁止: 馬単 / 三連単 / 穴推奨の馬連 / 高EVだけを理由にした馬連  (compute_bets 側で強制)
  見送り: field_chaos>=0.92 ∨ field_size<=7 ∨ ◎p_win<0.05  (validate_cowork_bets と一致)
  勝負枠 : 信頼度上位、1日2〜3R、1R ¥10,000 固定
  準勝負 : 次点、1日2〜4R、1R ¥5,000〜8,000 (信頼度でスケール)
  消化枠 : 残りで floor(10R∧¥100k/週=1日5R∧¥50k目安)を満たすぶん、1R ¥1,000〜3,000
  順序   : 勝負→準勝負→消化 の順で floor を充足

信頼度 = 0.45*上位2集中 + 0.30*◎独走 + 0.25*(1-混戦度)   (今日のOOS実証で堅い側が負け少)

出力: reports/bet_plan/{date}.json + 標準出力サマリ
実行: PYTHONUTF8=1 ./venv311/Scripts/python.exe build_bet_plan.py [date] [--weekend]
"""
from __future__ import annotations
import io, json, sys
from pathlib import Path

from production_policy import hard_skip_reasons, load_policy, policy_stamp

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

BASE = Path(__file__).parent
BUNDLE_DIR = BASE / "reports" / "cowork_input"
OUTDIR = BASE / "reports" / "bet_plan"

_POLICY = load_policy()
CHAOS_SKIP = float(_POLICY["chaos_reference"]["skip_percentile"])
FIELD_SIZE_SKIP = int(_POLICY["hard_skip"]["field_size_max"])
PWIN_SKIP = float(_POLICY["hard_skip"]["hon_p_win_min"])
# 枠パラメタ
SHOBU_MAX = 3;  SHOBU_YEN = 10000
JUN_MAX = 4;    JUN_YEN_HI = 8000; JUN_YEN_LO = 5000
SHOUKA_YEN_HI = 3000; SHOUKA_YEN_LO = 1000
# floor: プロ条件「10R以上∧¥100,000以上」を1日で確実に満たす(--weekend で2倍=2日合算運用時)
DAY_MIN_RACES = 10; DAY_MIN_YEN = 100000


def log(m): print(m, flush=True)


def conf_score(rc):
    d = rc.get("top1_dominance") or 0.0
    c = rc.get("top2_concentration") or 0.0
    ch = rc.get("field_chaos_score")
    ch = 0.5 if ch is None else ch
    return round(0.45 * c + 0.30 * d + 0.25 * (1 - ch), 4)


# ◎前走圧勝ボーナス (2026-07-23 実測: 前走1着の着差が大きいほど次走勝率が単調増
# 0.0s=12.7% → 0.3s=16.1% → 1.0s=31.5% [新馬→1勝, 他遷移も同傾向, n=4万+]。
# ROIは市場織込みで不変のため賭け金レバーでなく「枠の信頼度」レバーとしてのみ使う)
MARGIN_BONUS_W = 0.08


def load_margins(date):
    """weekly CSV から (rid16, 馬番) → (前走確定着順, 前走着差タイム)。無ければ空。"""
    try:
        from predict_weekly import parse_csv
        import pandas as pd
        p = BASE / "data" / "weekly" / f"{date}.csv"
        if not p.exists():
            return {}
        df = parse_csv(p)
        out = {}
        for _, r in df.iterrows():
            rid16 = str(r.get("レースID(新)", ""))[:16]
            ban = int(float(r.get("馬番", 0) or 0))
            fin = float(r.get("前走確定着順") or 99) if str(r.get("前走確定着順", "")).strip() else 99.0
            mg = r.get("前走着差タイム")
            try:
                mg = float(mg)
            except Exception:
                mg = None
            out[(rid16, ban)] = (fin, mg)
        return out
    except Exception as e:
        log(f"  (margin読込スキップ: {e})")
        return {}


def margin_bonus(margins, rid, horses):
    """◎馬が前走1着かつ着差圧勝なら conf ボーナス (0〜MARGIN_BONUS_W)。"""
    rid16 = str(rid)[:16]
    for h in horses:
        if h.get("mark") == "◎":
            fin, mg = margins.get((rid16, int(h.get("umaban", 0))), (99.0, None))
            if fin == 1.0 and mg is not None and mg < 0:
                return round(MARGIN_BONUS_W * min(-mg, 1.0), 4), mg
            return 0.0, mg if fin == 1.0 else None
    return 0.0, None


def load_races(date):
    p = BUNDLE_DIR / f"{date}_bundle.json"
    b = json.loads(p.read_text(encoding="utf-8"))
    races = b["races"]
    races = races if isinstance(races, list) else list(races.values())
    out = []
    for r in races:
        m = r.get("race_meta", {}); rc = r.get("race_confidence", {})
        horses = r.get("horses", [])
        hon = next((h for h in horses if h.get("mark") == "◎"), None)
        pwin_top = (hon or {}).get("p_win") or 0.0
        course = m.get("course", "")
        surf = course[:1] if course else "?"
        out.append({
            "rid": r.get("race_id"), "place": m.get("place", ""),
            "rno": int(str(r.get("race_id", "0"))[-2:]) if str(r.get("race_id", "")).strip()[-2:].isdigit() else None,
            "course": course, "klass": m.get("class", ""),
            "race_name": m.get("race_name") or "", "field_size": m.get("field_size"),
            "conf": conf_score(rc), "dom": rc.get("top1_dominance"),
            "concentration": rc.get("top2_concentration"), "chaos": rc.get("field_chaos_score"),
            "market": rc.get("ai_market_agreement"), "pwin_top": round(pwin_top, 4),
            "hon": ({"mark": "◎", "p_win": (hon or {}).get("p_win"),
                     "tansho_odds": (hon or {}).get("tansho_odds")} if hon else None),
            "judgment": (r.get("buy_judgment", {}) or {}).get("headline", ""),
        })
    return out


def miokuri_reason(r):
    reasons = hard_skip_reasons(
        {"field_size": r.get("field_size")},
        {"field_chaos_score": r.get("chaos")}, r.get("hon"))
    return " / ".join(reasons) if reasons else None


def jun_yen(rank, n):
    if n <= 1: return JUN_YEN_HI
    t = rank / (n - 1)           # 0(高conf)→1(低)
    y = JUN_YEN_HI - t * (JUN_YEN_HI - JUN_YEN_LO)
    return int(round(y / 500) * 500)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    weekend = "--weekend" in sys.argv
    date = args[0] if args else None
    if not date:
        bs = sorted(BUNDLE_DIR.glob("*_bundle.json"))
        date = bs[-1].name[:8] if bs else None
    if not date:
        log("bundle が無い"); return
    mult = 2 if weekend else 1
    min_races = DAY_MIN_RACES * mult; min_yen = DAY_MIN_YEN * mult

    races = load_races(date)
    # ◎前走圧勝ボーナス (的中率レバー。weekly CSV 無ければ 0 のまま)
    margins = load_margins(date)
    if margins:
        p = BUNDLE_DIR / f"{date}_bundle.json"
        b = json.loads(p.read_text(encoding="utf-8"))
        braces = b["races"] if isinstance(b["races"], list) else list(b["races"].values())
        hmap = {str(r.get("race_id")): r.get("horses", []) for r in braces}
        for r in races:
            bonus, mg = margin_bonus(margins, r["rid"], hmap.get(str(r["rid"]), []))
            r["margin_bonus"] = bonus; r["prev_margin"] = mg
            r["conf"] = round(r["conf"] + bonus, 4)
        nb = sum(1 for r in races if r.get("margin_bonus"))
        log(f"◎前走圧勝ボーナス適用: {nb}R (重み{MARGIN_BONUS_W}×圧勝秒clip1.0)")
    else:
        for r in races:
            r["margin_bonus"] = 0.0; r["prev_margin"] = None
    # 見送り判定
    live, miokuri = [], []
    for r in races:
        mr = miokuri_reason(r)
        if mr:
            r["tier"] = "見送り"; r["miokuri"] = mr; r["budget"] = 0; miokuri.append(r)
        else:
            live.append(r)
    live.sort(key=lambda x: x["conf"], reverse=True)

    # 勝負 → 準勝負 → 消化
    shobu = live[:SHOBU_MAX]
    jun = live[SHOBU_MAX:SHOBU_MAX + JUN_MAX]
    rest = live[SHOBU_MAX + JUN_MAX:]
    for r in shobu:
        r["tier"] = "勝負"; r["budget"] = SHOBU_YEN
    for i, r in enumerate(jun):
        r["tier"] = "準勝負"; r["budget"] = jun_yen(i, len(jun))
    # 消化: floor (races & yen) を満たすまで上位confから確保
    total_r = len(shobu) + len(jun)
    total_y = sum(r["budget"] for r in shobu + jun)
    shouka = []
    for r in rest:
        need_more = (total_r < min_races) or (total_y < min_yen)
        if not need_more:
            r["tier"] = "対象外"; r["budget"] = 0
            continue
        # 不足金額が小さければ下限、大きければ上限寄り
        gap = max(0, min_yen - total_y)
        amt = SHOUKA_YEN_HI if gap >= SHOUKA_YEN_HI else max(SHOUKA_YEN_LO, int(round(gap / 1000) * 1000) or SHOUKA_YEN_LO)
        r["tier"] = "消化"; r["budget"] = amt; shouka.append(r)
        total_r += 1; total_y += amt

    plan = {"date": date, "weekend": weekend, "policy": policy_stamp(),
            "floor": {"min_races": min_races, "min_yen": min_yen},
            "rules": {"禁止": ["馬単", "三連単", "穴推奨の馬連", "高EVだけの馬連"],
                       "見送り": [f"chaos percentile≥{CHAOS_SKIP}",
                                    f"頭数≤{FIELD_SIZE_SKIP}",
                                    "◎単勝オッズ欠損", f"◎p_win<{PWIN_SKIP}"]},
            "tiers": {
                "勝負": [_row(r) for r in shobu], "準勝負": [_row(r) for r in jun],
                "消化": [_row(r) for r in shouka], "見送り": [_row(r) for r in miokuri]},
            "totals": {"bet_races": total_r, "bet_yen": total_y,
                        "floor_met": bool(total_r >= min_races and total_y >= min_yen)}}
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / f"{date}.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    # サマリ
    log("=" * 78)
    log(f"枠プラン {date}  floor: {min_races}R∧¥{min_yen:,} {'(週末)' if weekend else '(1日)'}")
    log(f"禁止: 馬単/三連単/穴馬連/EV単独馬連   見送り: "
        f"chaos pct≥{CHAOS_SKIP}∨頭数≤{FIELD_SIZE_SKIP}∨◎odds欠損∨◎<{PWIN_SKIP}")
    log("=" * 78)
    for tier, rows in [("勝負", shobu), ("準勝負", jun), ("消化", shouka)]:
        sub = sum(r["budget"] for r in rows)
        log(f"\n【{tier}枠】{len(rows)}R / ¥{sub:,}")
        for r in rows:
            mb = f"/◎圧勝+{r['margin_bonus']:.2f}" if r.get("margin_bonus") else ""
            log(f"  {r['place']}{r['rno']}R {r['course']} {r['klass']} {r['field_size']}頭  "
                f"¥{r['budget']:,}  conf {r['conf']}(独走{r['dom']:.2f}/集中{r['concentration']:.2f}/混戦{r['chaos']:.2f}/市場{r['market'] if r['market'] is None else round(r['market'],2)}{mb})")
    if miokuri:
        log(f"\n【見送り】{len(miokuri)}R")
        for r in miokuri:
            log(f"  {r['place']}{r['rno']}R {r['course']} {r['klass']}  ← {r['miokuri']}")
    t = plan["totals"]
    log("\n" + "=" * 78)
    log(f"合計参加 {t['bet_races']}R / ¥{t['bet_yen']:,}  floor達成={'✅' if t['floor_met'] else '❌'}")
    log(f"[saved] {OUTDIR / (date + '.json')}")


def _row(r):
    return {k: r.get(k) for k in ("rid", "place", "rno", "course", "klass", "race_name",
                                   "field_size", "tier", "budget", "conf", "dom",
                                   "concentration", "chaos", "market", "pwin_top",
                                   "margin_bonus", "prev_margin",
                                   "miokuri", "judgment")}


if __name__ == "__main__":
    main()
