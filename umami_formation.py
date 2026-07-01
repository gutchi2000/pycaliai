# -*- coding: utf-8 -*-
"""
umami_formation.py — UMAMIテーブルから三連複フォメを機械生成 (ユーザー仕様)
=========================================================================
見送り: S格0頭 → 買わない
通常フォメ: 1列=◎+S全 / 2列=S全+A全 / 3列=罠以外全
Sボックス: S>=SBOX_TH 頭 → S全頭の三連複ボックス
25点上限: 超過は3列を低xROI順(同値はp_sho低い順)に削る
入力: reports/cowork_input/{date}_bundle.json
実行: python umami_formation.py [--date 20260614] [--sbox 5]
"""
from __future__ import annotations
import argparse, io, json, sys
from itertools import combinations, product
from pathlib import Path
from umami import umami_total

try: sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
except Exception: pass
BASE = Path(__file__).parent
CAP = 25

def build_tris(c1, c2, c3):
    t = set()
    for a in c1:
        for b in c2:
            for c in c3:
                if len({a, b, c}) == 3:
                    t.add(frozenset((a, b, c)))
    return t

def _sbox(box, hon, by):
    """{◎}∪S ボックス。25超なら弱いSから削る。returns (box, tris, trimmed)."""
    box = sorted(set(box)); trimmed = []
    tris = {frozenset(c) for c in combinations(box, 3)}
    weak = sorted([b for b in box if b != hon], key=lambda b: (by[b]["xroi"], by[b]["p"]))
    i = 0
    while len(tris) > CAP and i < len(weak) and len(box) > 3:
        box.remove(weak[i]); trimmed.append(f"S{weak[i]}"); i += 1
        tris = {frozenset(c) for c in combinations(box, 3)}
    return box, tris, trimmed

def formation(race, sbox_th):
    horses = race["horses"]
    by = {}
    for h in horses:
        t = umami_total(h)
        by[int(h["umaban"])] = {"ban": int(h["umaban"]), "mark": h.get("mark") or "",
            "grade": t["grade"], "xroi": (t["xroi"] if t["xroi"] is not None else -1.0),
            "p": float(h.get("p_sho") or 0), "rank": h.get("ai_rank")}
    S = sorted([b for b, x in by.items() if x["grade"] == "S"])
    A = sorted([b for b, x in by.items() if x["grade"] == "A"])
    nontrap = sorted([b for b, x in by.items() if x["grade"] != "罠"])
    hon = next((b for b, x in by.items() if "◎" in x["mark"]), None)
    if hon is None:  # ◎印が無ければ ai_rank==1 を本命扱い
        hon = next((b for b, x in by.items() if x["rank"] == 1), None)

    if not S:
        return {"adopt": False, "reason": "S格0頭→見送り", "nS": 0}

    # 方針2: S>=閾値 → 積極的 {◎}∪S ボックス (戦略選択)
    if len(S) >= sbox_th:
        box, tris, trimmed = _sbox(([hon] if hon else []) + S, hon, by)
        return {"adopt": True, "mode": f"Sボックス(S{len(S)})", "nS": len(S), "nA": len(A),
                "hon": hon, "c1": box, "c2": [], "c3": [], "tris": list(tris),
                "trimmed": trimmed, "pts": len(tris)}

    # 通常フォメ + 25点カスケード (Phase A→B→C)
    c1 = sorted(set(([hon] if hon else []) + S))
    c2 = sorted(set(S + A))
    c3 = list(nontrap)
    core = set(([hon] if hon else []) + S + A)
    trimmed = []
    tris = build_tris(c1, c2, c3)
    # Phase A: 3列の非中核(B/C)を低xROI順に削る
    for b in sorted([x for x in c3 if x not in core], key=lambda b: (by[b]["xroi"], by[b]["p"])):
        if len(tris) <= CAP: break
        c3.remove(b); trimmed.append(f"3列{b}"); tris = build_tris(c1, c2, c3)
    # Phase B: A格を低xROI順に削る (◎は絶対残す)
    for b in sorted([x for x in A if x != hon], key=lambda b: (by[b]["xroi"], by[b]["p"])):
        if len(tris) <= CAP: break
        if b in c2: c2.remove(b)
        if b in c3: c3.remove(b)
        trimmed.append(f"A{b}"); tris = build_tris(c1, c2, c3)
    # Phase C: ◎+Sだけで超 → {◎}∪S ボックス退避
    if len(tris) > CAP:
        box, tris2, tr2 = _sbox(([hon] if hon else []) + S, hon, by)
        return {"adopt": True, "mode": "Sボックス退避", "nS": len(S), "nA": len(A), "hon": hon,
                "c1": box, "c2": [], "c3": [], "tris": list(tris2),
                "trimmed": trimmed + tr2, "pts": len(tris2)}
    return {"adopt": True, "mode": "通常フォメ", "nS": len(S), "nA": len(A), "hon": hon,
            "c1": c1, "c2": c2, "c3": sorted(c3), "tris": list(tris),
            "trimmed": trimmed, "pts": len(tris)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="20260614")
    ap.add_argument("--sbox", type=int, default=5, help="S>=これでSボックス(暫定)")
    ap.add_argument("--show", type=int, default=8)
    args = ap.parse_args()
    d = json.load(open(BASE / f"reports/cowork_input/{args.date}_bundle.json", encoding="utf-8"))
    races = d["races"]
    print(f"[bundle] {args.date}  {len(races)}R  / Sボックス閾値=S{args.sbox}以上 / 上限{CAP}点\n")

    adopt = miokuri = total_pts = max_pts = over25 = 0
    shown = 0
    for r in races:
        f = formation(r, args.sbox)
        name = (r.get("race_meta", {}) or {}).get("race_name") or r["race_id"]
        if not f["adopt"]:
            miokuri += 1
            if shown < args.show:
                print(f"[見送り] {r['race_id']} {name}  ({f['reason']})"); shown += 1
            continue
        adopt += 1; total_pts += f["pts"]
        max_pts = max(max_pts, f["pts"]); over25 += (f["pts"] > CAP)
        if shown < args.show:
            tk = ", ".join("-".join(map(str, sorted(t))) for t in sorted(f["tris"], key=lambda s: sorted(s))[:8])
            tr = f"  削り{f['trimmed']}" if f["trimmed"] else ""
            print(f"[採用] {r['race_id']} {name}  ◎{f['hon']} {f['mode']} S{f['nS']}/A{f['nA']} → {f['pts']}点{tr}")
            print(f"        1列{f['c1']} 2列{f['c2']} 3列{f['c3']}")
            print(f"        買い目(先頭8): {tk}{' ...' if f['pts']>8 else ''}")
            shown += 1
    print(f"\n{'='*60}\n採用 {adopt}R / 見送り {miokuri}R / 平均 {total_pts/max(adopt,1):.1f}点"
          f" / 最大 {max_pts}点 / 25点超過 {over25}R / 採用時総点 {total_pts}点")
    print(f"  → 25点キャップ {'✅ 全レース25以下' if over25 == 0 else f'❌ {over25}R が超過(要修正)'}")

if __name__ == "__main__":
    main()
