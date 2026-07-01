# -*- coding: utf-8 -*-
"""◎〇▲△△ 印馬の週次レース回顧(Web結果版).

bundle.json(印・確率) と reports/web_results/{date}.json(Webで調べた1-3着) を
突合し、印馬がどう走ったかを全レース回顧する。kekka.csv が無い段階用。

入力:
  reports/cowork_input/{date}_bundle.json   ... 印・確率・オッズ
  reports/web_results/{date}.json           ... {place: {raceno: [[umaban,name]x3]}}

usage:
  python analysis/review_marks_join.py                 # 最新土日を自動検出
  python analysis/review_marks_join.py 20260606 20260607
"""
import sys
import json
import glob
import os

sys.stdout.reconfigure(encoding="utf-8")
BASE = r"E:\PyCaLiAI"
INDIR = rf"{BASE}\reports\cowork_input"
RESDIR = rf"{BASE}\reports\web_results"

MARK_ORDER = {"◎": 0, "〇": 1, "○": 1, "▲": 2, "△": 3}
NORM = {"○": "〇"}


def latest_weekend():
    files = glob.glob(rf"{INDIR}\*_bundle.json")
    dates = sorted({os.path.basename(f).split("_")[0] for f in files}, reverse=True)
    return list(reversed(dates[:2]))


def race_no(rid):
    return int(str(rid)[-2:])


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def pos_of(umaban, top3):
    """top3 = [[uma,name],...] → 1/2/3 か None(着外)."""
    for i, (u, _name) in enumerate(top3, 1):
        if int(u) == int(umaban):
            return i
    return None


def winner_odds_band(bundle_race, win_uma):
    """勝ち馬の単勝オッズ帯で波乱/順当を判定."""
    for h in bundle_race["horses"]:
        if int(h["umaban"]) == int(win_uma):
            o = h.get("tansho_odds")
            if o is None:
                return "?", None
            if o < 5:
                return "順当", o
            if o < 10:
                return "中波", o
            return "波乱", o
    return "?(穴/印無)", None


def review_day(date, agg):
    bundle = load_json(rf"{INDIR}\{date}_bundle.json")
    res = load_json(rf"{RESDIR}\{date}.json")
    if not bundle:
        print(f"## {date}: bundle なし\n")
        return
    if not res:
        print(f"## {date}: web_results なし → reports/web_results/{date}.json を作成してください\n")
        return
    results = res["results"]
    races = sorted(bundle["races"],
                   key=lambda r: (r["race_meta"].get("place", ""), race_no(r["race_id"])))
    print(f"\n# {date}  (model {bundle.get('model')})  対象 {len(races)}R")
    print(f"_結果ソース: {res.get('source', '?')}_\n")

    for r in races:
        m = r["race_meta"]
        place, rno = m.get("place"), race_no(r["race_id"])
        top3 = results.get(place, {}).get(str(rno))
        marked = sorted([h for h in r["horses"] if h.get("mark")],
                        key=lambda h: MARK_ORDER.get(h["mark"], 9))
        rc = r.get("race_confidence", {})
        chaos = rc.get("field_chaos_score")
        head = (f"## {place}{rno}R {m.get('course')} {m.get('class')} "
                f"({m.get('field_size')}頭)"
                + (f"  chaos{chaos:.2f}" if chaos is not None else ""))
        if not top3:
            print(head + "  ※結果未取得\n")
            continue

        # 結果行(印を付与)
        umk = {int(h["umaban"]): NORM.get(h["mark"], h["mark"]) for h in marked}
        band, wo = winner_odds_band(r, top3[0][0])
        res_cells = []
        for i, (u, nm) in enumerate(top3, 1):
            mk = umk.get(int(u), "")
            res_cells.append(f"{i}着 {u}{mk} {nm}")
        print(head)
        print(f"  結果: " + " / ".join(res_cells) + f"  [勝馬{('単' + str(wo) + '倍') if wo else ''}/{band}]")

        # 印別の着順
        hit_marks = []
        for h in marked:
            mk = NORM.get(h["mark"], h["mark"])
            p = pos_of(h["umaban"], top3)
            ps = f"{p}着" if p else "着外"
            if p:
                hit_marks.append((mk, p))
            o = h.get("tansho_odds")
            print(f"    {mk} {h['umaban']:>2} {h['horse_name']}  単{o if o is not None else '?'}倍"
                  f"  rank{h['ai_rank']} vs:{h.get('ai_vs_market')}  → {ps}")
            # 集計
            a = agg["mark"].setdefault(mk, {"n": 0, "win": 0, "ren": 0, "fuku": 0})
            a["n"] += 1
            if p == 1:
                a["win"] += 1
            if p and p <= 2:
                a["ren"] += 1
            if p:
                a["fuku"] += 1
        # レース単位集計
        agg["races"] += 1
        hon = next((h for h in marked if NORM.get(h["mark"], h["mark"]) == "◎"), None)
        if hon:
            hp = pos_of(hon["umaban"], top3)
            tag = {1: "◎勝ち", 2: "◎連対", 3: "◎複勝"}.get(hp, "◎消し")
            if hp:
                agg["hon_in3"] += 1
            agg["hon_n"] += 1
        else:
            tag = "◎なし"
        marks_in3 = len(hit_marks)
        agg["marks_in3_total"] += marks_in3
        if marks_in3 >= 3:
            agg["races_3plus"] += 1
        if marks_in3 == 0:
            agg["races_0"] += 1
        # 上位3着がすべて印 = 印BOX完全的中
        all_marked = all(int(u) in umk for u, _ in top3)
        flag = "  ★印で1-3着独占" if all_marked else ""
        print(f"    => {tag} / 印の複勝圏 {marks_in3}/5{flag}\n")


def print_summary(agg):
    print("\n# ===== 週次サマリ =====")
    print(f"対象レース {agg['races']}R\n")
    print("## 印別 成績(複勝圏=3着内)")
    print("| 印 | 頭数 | 勝 | 連対 | 複勝圏 | 勝率 | 連対率 | 複勝率 |")
    print("|---|---|---|---|---|---|---|---|")
    for mk in ["◎", "〇", "▲", "△"]:
        a = agg["mark"].get(mk)
        if not a or a["n"] == 0:
            continue
        n = a["n"]
        print(f"| {mk} | {n} | {a['win']} | {a['ren']} | {a['fuku']} | "
              f"{a['win']/n*100:.1f}% | {a['ren']/n*100:.1f}% | {a['fuku']/n*100:.1f}% |")
    if agg["hon_n"]:
        print(f"\n◎ 複勝圏率: {agg['hon_in3']}/{agg['hon_n']} = {agg['hon_in3']/agg['hon_n']*100:.1f}%")
    print(f"印の複勝圏 平均: {agg['marks_in3_total']/agg['races']:.2f}頭/R")
    print(f"印3頭以上が複勝圏のレース: {agg['races_3plus']}R / 印が1頭も来なかったレース: {agg['races_0']}R")


def main(dates):
    agg = {"mark": {}, "races": 0, "hon_in3": 0, "hon_n": 0,
           "marks_in3_total": 0, "races_3plus": 0, "races_0": 0}
    print("# ◎〇▲△△ 週次レース回顧（Web結果版）")
    for d in dates:
        review_day(d, agg)
    print_summary(agg)


if __name__ == "__main__":
    args = sys.argv[1:]
    dates = args if args else latest_weekend()
    main(dates)
