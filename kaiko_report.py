# -*- coding: utf-8 -*-
"""
kaiko_report.py — 週次「回顧」レポート（月曜 kekka 置きしだい / weekly_post.ps1 フック）
====================================================================================
予測→賭け→結果のループを閉じる。kekka 配置後に:
  1. ◎〇▲ の信頼度（複勝率/勝率）を当日 + クラス別で突合
  2. 券種別 P/L（投資/払戻/ROI/的中率）を突合 → compute_bets の実績フィードバック材料
  3. 自信を持って外した印（◎が着外）を抽出 → watch / 要因メモ
出力: reports/kaiko/{date}.json + 人間可読サマリ（NiceGUI/監査が読む）
実行: PYTHONUTF8=1 python kaiko_report.py --date 20260531
"""
from __future__ import annotations
import argparse, io, json, re, sys
from pathlib import Path
import pandas as pd

BASE = Path(__file__).parent
OUT_DIR = BASE / "reports" / "kaiko"


def _rid16(x): return re.sub(r"\D", "", str(x))[:16]
def _num(x):
    try: v = float(x); return v if v == v else None
    except Exception: return None


def load(date: str):
    bundle = json.load(open(BASE / f"reports/cowork_input/{date}_bundle.json", encoding="utf-8"))
    betsf = BASE / f"reports/cowork_output/{date}_bets.json"
    bets = json.load(open(betsf, encoding="utf-8")) if betsf.exists() else []
    bets = bets.get("bets", []) if isinstance(bets, dict) else (bets if isinstance(bets, list) else [])
    kpath = BASE / f"data/kekka/{date}.csv"
    cols = ["日付", "場所", "R", "枠番", "馬番", "馬名", "着順", "レースID",
            "単勝配当", "複勝配当", "枠連", "馬連", "馬単", "三連複", "三連単"]
    if not kpath.exists():   # kekka未配置でも review(定性)だけは回せるよう空で続行
        k = pd.DataFrame(columns=cols + ["rid16"])
        return bundle, bets, k
    k = pd.read_csv(kpath, encoding="cp932", low_memory=False)
    k.columns = cols[:len(k.columns)]
    k["rid16"] = k["レースID"].map(_rid16)
    for c in ["馬番", "着順", "単勝配当", "複勝配当", "馬連", "馬単"]:
        k[c] = pd.to_numeric(k[c], errors="coerce")
    return bundle, bets, k


def load_review(date: str):
    """reports/review/*full_target_import.csv（ヘッダ無 4列: 場R/馬名/レースID18桁/コメント）
    から該当日(rid[:8])の行を抽出。コメントから印・着順・次走方針を構造化。"""
    import glob
    rows = []
    for f in glob.glob(str(BASE / "reports/review/*full_target_import.csv")):
        d = pd.read_csv(f, encoding="cp932", header=None,
                        names=["race", "name", "rid", "comment"], low_memory=False)
        d["rid"] = d["rid"].astype(str)
        rows.append(d[d["rid"].str[:8] == date])
    if not rows or sum(len(r) for r in rows) == 0:
        return None
    d = pd.concat(rows, ignore_index=True)
    d["comment"] = d["comment"].astype(str)
    d["rid16"] = d["rid"].str[:16]
    d["ban"] = pd.to_numeric(d["rid"].str[16:18], errors="coerce")
    d["mark"] = d["comment"].str.extract(r"AI([◎〇○▲△])")[0]
    d["chaku"] = pd.to_numeric(d["comment"].str.extract(r"で(\d+)着")[0], errors="coerce")
    d["jisou"] = d["comment"].str.split("→").str[-1].str.strip()
    return d


def load_wide(date: str):
    """wide_kekka.csv（ヘッダ無）から該当日の {(場所,R): {frozenset(i,j): payout}} を作る。"""
    p = BASE / "data/kekka/wide_kekka.csv"
    if not p.exists(): return {}
    w = pd.read_csv(p, encoding="cp932", header=None, low_memory=False)
    y, mo, d = int(date[:4]), int(date[4:6]), int(date[6:8])
    w = w[(w[0] == y) & (w[1] == mo) & (w[2] == d)]
    out = {}
    for _, r in w.iterrows():
        place, R, paycol = str(r[3]), int(r[4]), str(r[9])
        pays = {}
        for m in re.finditer(r"(\d+)-(\d+)\s*\\?(\d+)", paycol):
            pays[frozenset((int(m.group(1)), int(m.group(2))))] = int(m.group(3))
        if pays: out[(place, R)] = pays
    return out


def race_result(kr: pd.DataFrame):
    """1レースの着順・払戻を辞書化。"""
    finish = {int(r["着順"]): int(r["馬番"]) for _, r in kr.iterrows()
              if _num(r["着順"]) and _num(r["馬番"])}
    tansho = {int(r["馬番"]): _num(r["単勝配当"]) for _, r in kr.iterrows()
              if _num(r["単勝配当"]) and r["単勝配当"] > 0}
    fukusho = {int(r["馬番"]): _num(r["複勝配当"]) for _, r in kr.iterrows()
               if _num(r["複勝配当"]) and r["複勝配当"] > 0}
    umaren = next((_num(r["馬連"]) for _, r in kr.iterrows() if _num(r["馬連"]) and r["馬連"] > 0), None)
    umatan = next((_num(r["馬単"]) for _, r in kr.iterrows() if _num(r["馬単"]) and r["馬単"] > 0), None)
    return finish, tansho, fukusho, umaren, umatan


def settle(bet, finish, tansho, fukusho, umaren, umatan):
    """1 bet の (hit, payout円)。stake=購入額。ワイドは kekka に無いので None（別途）。"""
    kind = bet.get("馬券種"); sel = str(bet.get("買い目", "")); stake = _num(bet.get("購入額")) or 0
    top3 = {finish.get(1), finish.get(2), finish.get(3)}
    if kind == "単勝":
        b = int(sel) if sel.isdigit() else None
        return (b == finish.get(1), stake / 100 * (tansho.get(b, 0) if b else 0))
    if kind == "複勝":
        b = int(sel) if sel.isdigit() else None
        return (b in top3, stake / 100 * (fukusho.get(b, 0) if b else 0))
    if kind == "馬連":
        ps = set(int(x) for x in re.findall(r"\d+", sel))
        hit = ps == {finish.get(1), finish.get(2)}
        return (hit, stake / 100 * (umaren or 0) if hit else 0.0)
    if kind == "馬単":
        m = re.findall(r"\d+", sel)
        hit = len(m) == 2 and finish.get(1) == int(m[0]) and finish.get(2) == int(m[1])
        return (hit, stake / 100 * (umatan or 0) if hit else 0.0)
    return (None, None)   # ワイド/三連は別ソース


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    ap = argparse.ArgumentParser(); ap.add_argument("--date", required=True); args = ap.parse_args()
    bundle, bets, k = load(args.date)
    marks = {r.get("race_id") or r.get("race_meta", {}).get("race_id"):
             {"cls": r.get("race_meta", {}).get("class", ""),
              "m": {h.get("mark"): int(h["umaban"]) for h in r["horses"] if h.get("mark") in ("◎", "〇", "○", "▲")}}
             for r in bundle["races"]}

    # --- 印信頼度 ---
    rel = {"◎": [0, 0, 0], "〇": [0, 0, 0], "▲": [0, 0, 0]}  # [n, top3, win]
    by_cls = {}
    for rid, info in marks.items():
        kr = k[k["rid16"] == _rid16(rid)]
        if kr.empty: continue
        finish, *_ = race_result(kr)
        pos = {b: j for j, b in finish.items()}
        for mk, key in [(info["m"].get("◎"), "◎"), (info["m"].get("〇") or info["m"].get("○"), "〇"), (info["m"].get("▲"), "▲")]:
            if mk is None or mk not in pos: continue
            p = pos[mk]; rel[key][0] += 1; rel[key][1] += p <= 3; rel[key][2] += p == 1
            if key == "◎":
                c = by_cls.setdefault(info["cls"], [0, 0]); c[0] += 1; c[1] += p <= 3

    # --- 券種別 P/L（ワイドは wide_kekka 結線）---
    pl = {}
    wide_map = load_wide(args.date)
    for race in bets:
        kr = k[k["rid16"] == _rid16(race.get("race_id", ""))]
        if kr.empty: continue
        place, R = str(kr["場所"].iloc[0]), int(kr["R"].iloc[0])
        fin, tan, fuk, um, ut = race_result(kr)
        wides = wide_map.get((place, R), {})
        for b in race.get("bets", []):
            kind = b.get("馬券種"); stake = _num(b.get("購入額")) or 0
            if kind == "ワイド":
                ps = frozenset(int(x) for x in re.findall(r"\d+", str(b.get("買い目", ""))))
                pay = wides.get(ps)
                d = pl.setdefault("ワイド" if wides else "ワイド(未突合)", [0, 0, 0, 0])
                d[0] += 1; d[1] += stake
                if wides and pay: d[2] += stake / 100 * pay; d[3] += 1
                continue
            hit, pay = settle(b, fin, tan, fuk, um, ut)
            if hit is None:
                d = pl.setdefault(kind + "(未突合)", [0, 0, 0, 0]); d[0] += 1; d[1] += stake; continue
            d = pl.setdefault(kind, [0, 0, 0, 0])
            d[0] += 1; d[1] += stake; d[2] += pay; d[3] += int(bool(hit))

    # --- 出力 ---
    rep = {"date": args.date,
           "marks": {key: {"n": v[0], "top3_rate": round(v[1] / v[0], 3) if v[0] else None,
                           "win_rate": round(v[2] / v[0], 3) if v[0] else None} for key, v in rel.items()},
           "marks_by_class": {c: {"n": v[0], "top3_rate": round(v[1] / v[0], 3) if v[0] else None}
                              for c, v in by_cls.items()},
           "bets_pl": {kind: {"n": v[0], "invest": int(v[1]), "return": int(v[2]),
                              "roi": round(v[2] / v[1], 3) if v[1] else None,
                              "hit_rate": round(v[3] / v[0], 3) if v[0] else None}
                       for kind, v in pl.items()}}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{args.date}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"=== 回顧 {args.date} ===")
    for key in ("◎", "〇", "▲"):
        m = rep["marks"][key]
        if m["n"]: print(f"  {key}: 複勝率 {m['top3_rate']*100:.1f}% / 勝率 {m['win_rate']*100:.1f}%  (n={m['n']})")
    print("  [クラス別 ◎複勝率]")
    for c, v in sorted(rep["marks_by_class"].items(), key=lambda x: x[1]["top3_rate"] or 0):
        if v["n"] >= 2: print(f"     {c}: {v['top3_rate']*100:.1f}% (n={v['n']})")
    print("  [券種別 P/L]（compute_bets 実績フィードバック材料）")
    tot_i = tot_r = 0
    for kind, v in rep["bets_pl"].items():
        if "未突合" in kind:
            print(f"     {kind}: n={v['n']} 投資¥{v['invest']:,}（kekkaに払戻列なし→wide_kekka要）"); continue
        tot_i += v["invest"]; tot_r += v["return"]
        print(f"     {kind}: ROI {v['roi']*100:.0f}% 的中{v['hit_rate']*100:.0f}% (n={v['n']}, ¥{v['invest']:,}→¥{v['return']:,})")
    if tot_i: print(f"     -- 突合分合計: ROI {tot_r/tot_i*100:.0f}% (¥{tot_i:,}→¥{tot_r:,})")

    # --- 結果コメント回顧（定性: 敗因 + 次走方針 watch-list）---
    rv = load_review(args.date)
    if rv is not None and len(rv):
        hon = rv[rv["mark"] == "◎"]
        print(f"\n  [結果コメント回顧] {len(rv)}頭 / ◎ {len(hon)}頭")
        if len(hon):
            print(f"     ◎ 複勝率(コメント着順) {(hon['chaku']<=3).mean()*100:.0f}% / 勝率 {(hon['chaku']==1).mean()*100:.0f}%")
        WATCH = r"見直|短縮|延長|妙味|穴|巻き返|展開向|条件替|変わり身"
        wl = rv[rv["mark"].notna() & rv["jisou"].str.contains(WATCH, regex=True, na=False)]
        rep["review"] = {"n_horses": int(len(rv)),
            "watchlist": [{"race": r["race"], "name": r["name"], "mark": r["mark"],
                           "chaku": int(r["chaku"]) if pd.notna(r["chaku"]) else None,
                           "jisou": r["jisou"]} for _, r in wl.iterrows()]}
        print(f"     watch-list（次走 妙味/見直し）= {len(wl)}頭。例:")
        for _, r in wl.head(5).iterrows():
            print(f"       {r['mark']} {r['name']}({r['race']}) {int(r['chaku']) if pd.notna(r['chaku']) else '?'}着 → {str(r['jisou'])[:42]}")
        (OUT_DIR / f"{args.date}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print(f"\n  [結果コメント回顧] reports/review/ に {args.date} のコメント無し（full_target_import を置けば定性回顧+watch-list）")
    print(f"[saved] reports/kaiko/{args.date}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
