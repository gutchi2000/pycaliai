# -*- coding: utf-8 -*-
"""
simulate_brain_day.py — gutchi_brain.build_tickets を当日確定配当で決済する専用ハーネス。
================================================================================
simulate_plan_day.py は compute_bets を決済する(=prob-first 線)。こちらは **俺のブレイン
(gutchi_brain)自身の買い目**を決済する。ブレイン単独のバックテストは今まで存在しなかった
(記憶の「6/28 144%」は compute_bets 帰属/ golden 無し)。これで初めて数字が出る。

決済ソース: data/kekka/{date}.csv(15列: 単複馬連馬単３連複) + data/kekka/wide_kekka.csv(ワイド)
バイアス   : 前日(土)の実現バイアスを build_realized_bias.build() で生成し sat_bias に渡す
             (ドクトリン=土の結果→日のカードのクロスデイ。前日 kekka が無い日は None)
使い方:
  PYTHONUTF8=1 ./venv311/Scripts/python.exe simulate_brain_day.py            # 全16開催
  PYTHONUTF8=1 ./venv311/Scripts/python.exe simulate_brain_day.py 20260628   # 単日
  ...--gate                # バイアス記号ゲート版(bias_mode=True)で回す
  ...--no-chaos-gate       # chaos>=0.92 の真見送り(2026-07-31配線)を切る (アブレーション)
  ...--no-discipline       # L3規律層(トリガミ床+chalk-cap)を切る (アブレーション)
  ...--strict              # 厳選モード(chaos<0.90∧dom1>=0.10のみ参加, opt-in, 前向き検証用)

判定はROIでなく トリガミ率/的中率/最大DD を主指標に、CI と power/MDE を併記する
(2026-07-30 前提監査の教訓: 非有意=効果なし ではない)。
"""
from __future__ import annotations
import datetime as _dt
import io
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import pandas as pd

import gutchi_brain as GB
import build_realized_bias as BRB

BASE = Path(__file__).parent
# bundle と kekka の両方が揃う開催(2026-05〜07)
DEFAULT_DATES = [
    "20260516", "20260517", "20260523", "20260524", "20260530", "20260531",
    "20260606", "20260607", "20260613", "20260614", "20260620", "20260621",
    "20260627", "20260628", "20260704", "20260705",
]


def rid16(x):
    return re.sub(r"\D", "", str(x))[:16]


def _n(x):
    try:
        s = str(x).replace(",", "").strip()
        return float(s) if s not in ("", "nan") else 0.0
    except Exception:
        return 0.0


def load_kekka(date):
    df = pd.read_csv(BASE / "data" / "kekka" / f"{date}.csv", encoding="cp932", low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    df["rid16"] = df["レースID(新)"].map(rid16)
    df["着"] = pd.to_numeric(df["確定着順"], errors="coerce")
    df["馬番"] = pd.to_numeric(df["馬番"], errors="coerce")
    out = {}
    for rid, g in df.groupby("rid16"):
        g1, g2, g3 = g[g["着"] == 1], g[g["着"] == 2], g[g["着"] == 3]
        if len(g1) == 0 or len(g2) == 0 or len(g3) == 0:
            continue
        w, p, s = int(g1.iloc[0]["馬番"]), int(g2.iloc[0]["馬番"]), int(g3.iloc[0]["馬番"])
        fuku = {int(r["馬番"]): _n(r["複勝配当"]) for _, r in g.iterrows() if r["着"] in (1, 2, 3)}
        out[rid] = {
            "win": w, "plc": p, "sho": s,
            "place": g1.iloc[0]["場所"], "R": int(g1.iloc[0]["Ｒ"]),
            "tansho": _n(g1.iloc[0]["単勝配当"]), "fuku": fuku,
            "umaren": _n(g1.iloc[0]["馬連"]), "trio": _n(g1.iloc[0]["３連複"]),
        }
    return out


def load_wide(date):
    df = pd.read_csv(BASE / "data" / "kekka" / "wide_kekka.csv", encoding="cp932",
                     header=None, low_memory=False)
    y, m, d = int(date[:4]), int(date[4:6]), int(date[6:8])
    out = {}
    for _, r in df.iterrows():
        try:
            if int(r[0]) != y or int(r[1]) != m or int(r[2]) != d:
                continue
        except Exception:
            continue
        place, R, pairs = str(r[3]), int(r[4]), {}
        for part in str(r[9]).split("/"):
            mm = re.search(r"(\d+)\s*-\s*(\d+)\s*\\?\s*([\d,]+)", part)
            if mm:
                pairs[frozenset((int(mm.group(1)), int(mm.group(2))))] = _n(mm.group(3))
        if pairs:
            out[(place, R)] = pairs
    return out


def prior_bias(date):
    """前日(土)の 174列バイアスCSV data/bias/{prev}.csv から会場×面バイアス(venues)を生成。
    無ければ None(=クロスデイ元なし=見送り)。※ kekka(15列)は通過順が無く脚質を出せないので不可。"""
    d = _dt.date(int(date[:4]), int(date[4:6]), int(date[6:8])) - _dt.timedelta(days=1)
    prev = d.strftime("%Y%m%d")
    p = BASE / "data" / "bias" / f"{prev}.csv"
    if not p.exists():
        return None
    try:
        return BRB.build(str(p), date=prev).get("venues") or None
    except Exception:
        return None


def settle(bet, ko, wide_pairs):
    kind, sel, amt = bet["馬券種"], bet["買い目"], bet["購入額"]
    unit = amt / 100.0
    if kind == "単勝":
        return ko["tansho"] * unit if int(sel) == ko["win"] else 0.0
    if kind == "複勝":
        b = int(sel)
        return ko["fuku"].get(b, 0.0) * unit if b in (ko["win"], ko["plc"], ko["sho"]) else 0.0
    if kind == "馬連":
        i, j = (int(x) for x in sel.split("-"))
        return ko["umaren"] * unit if {i, j} == {ko["win"], ko["plc"]} else 0.0
    if kind == "ワイド":
        i, j = (int(x) for x in sel.split("-"))
        return (wide_pairs or {}).get(frozenset((i, j)), 0.0) * unit
    if kind == "三連複":
        trio = set(int(x) for x in sel.split("-"))
        return ko["trio"] * unit if trio == {ko["win"], ko["plc"], ko["sho"]} else 0.0
    return 0.0


def run_day(date, gate, bykind, tally, recs, bt_kw):
    bundle = json.loads((BASE / "reports" / "cowork_input" / f"{date}_bundle.json").read_text(encoding="utf-8"))
    races = bundle["races"]
    races = races if isinstance(races, list) else list(races.values())
    kekka, wide = load_kekka(date), load_wide(date)
    sat_bias = prior_bias(date)

    d_stake = d_ret = 0
    n_bet = n_miokuri = n_settled = 0
    for r in races:
        kw = dict(sat_bias=sat_bias, budget=GB.BUDGET, **bt_kw)
        if gate:
            kw["bias_mode"] = True
            # 記号の内訳を集計(★/⚠/無印)
            hs = [h for h in r.get("horses", []) if GB._num(h.get("tansho_odds"))]
            if hs and "新馬" not in (r.get("race_meta", {}).get("class", "") or ""):
                m = r.get("race_meta", {})
                surf = "芝" if (m.get("course", "") or "")[:1] == "芝" else "ダ"
                vb = (sat_bias or {}).get(f"{m.get('place')}|{surf}")
                idx = GB.ai_index(hs)
                a_h = next((h for h in hs if h.get("mark") == "◎"),
                           max(hs, key=lambda h: idx[h["umaban"]]))
                fsz = m.get("field_size") or len(hs)
                sym = GB.bias_symbol(hs, a_h, vb, fsz)[0] if vb else "無印"
                tally.setdefault("sym", defaultdict(int))[sym] += 1
        tickets = GB.build_tickets(r, **kw)
        if not tickets:
            n_miokuri += 1
            continue
        ko = kekka.get(rid16(r.get("race_id")))
        if ko is None:            # 結果欠損(取消レース等)は決済対象外
            continue
        n_bet += 1
        n_settled += 1
        wp = wide.get((ko["place"], ko["R"]))
        r_stake = r_ret = 0.0
        for b in tickets:
            pay = settle(b, ko, wp)
            r_stake += b["購入額"]
            r_ret += pay
            k = bykind[b["馬券種"]]
            k[0] += b["購入額"]; k[1] += pay; k[2] += 1; k[3] += 1 if pay > 0 else 0
        d_stake += r_stake
        d_ret += r_ret
        recs.append({"date": date, "stake": r_stake, "ret": r_ret, "pts": len(tickets)})
    roi = d_ret / d_stake * 100 if d_stake else 0
    tally["stake"] += d_stake; tally["ret"] += d_ret
    tally["bet"] += n_bet; tally["miokuri"] += n_miokuri
    tally["races"] += n_bet + n_miokuri
    print(f"{date}  参加{n_bet:>2d}/{n_bet+n_miokuri:>2d}  見送り{n_miokuri:>2d}  "
          f"投資¥{d_stake:>7,}  払戻¥{int(d_ret):>7,}  収支¥{int(d_ret-d_stake):>+8,}  ROI {roi:>6.1f}%")
    return roi


def race_stats(recs):
    """レース粒度の主指標: 的中率 / トリガミ率(的中中) / 最大DD / ROI CI95(bootstrap) / MDE。"""
    if not recs:
        return None
    st = np.array([r["stake"] for r in recs])
    rt = np.array([r["ret"] for r in recs])
    hit = rt > 0
    trig = hit & (rt < st)                     # 当たったのにマイナス
    net = rt - st
    cum = np.cumsum(net)
    peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]
    maxdd = float((peak - cum).max())
    rng = np.random.default_rng(42)
    n = len(recs)
    idx = rng.integers(0, n, (4000, n))
    boots = rt[idx].sum(axis=1) / np.where(st[idx].sum(axis=1) > 0, st[idx].sum(axis=1), 1)
    lo, hi = np.percentile(boots, [2.5, 97.5]) * 100
    se = float(boots.std() * 100)
    return {
        "n": n, "hit": float(hit.mean() * 100),
        "trig_of_hit": float(trig.sum() / hit.sum() * 100) if hit.sum() else 0.0,
        "clean": int((hit & (rt >= st)).sum()), "trig": int(trig.sum()),
        "maxdd": maxdd, "ci": (float(lo), float(hi)), "se": se, "mde": 2.8 * se,
    }


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    gate = "--gate" in sys.argv
    bt_kw = {}
    if "--no-chaos-gate" in sys.argv:
        bt_kw["chaos_gate"] = False
    if "--no-discipline" in sys.argv:
        bt_kw["discipline"] = False
    if "--strict" in sys.argv:
        bt_kw["strict"] = True
    dates = args or DEFAULT_DATES

    bykind = defaultdict(lambda: [0, 0.0, 0, 0])   # stake, ret, n, hit
    tally = {"stake": 0, "ret": 0.0, "bet": 0, "miokuri": 0, "races": 0}
    recs = []
    print("=" * 96)
    mode = "バイアス記号ゲートON" if gate else "現行"
    for k, lab in (("chaos_gate", "chaos真見送りOFF"), ("discipline", "L3規律OFF")):
        if bt_kw.get(k) is False:
            mode += f" / {lab}"
    if bt_kw.get("strict"):
        mode += f" / 厳選(chaos<{GB.STRICT_CHAOS}∧dom1>={GB.STRICT_DOM1})"
    print(f"gutchi_brain v{GB.BRAIN_VERSION} 決済バックテスト  [{mode}]  "
          f"選択=9時bundleオッズ / 採点=確定配当")
    print("=" * 96)
    for dt in dates:
        try:
            run_day(dt, gate, bykind, tally, recs, bt_kw)
        except FileNotFoundError as e:
            print(f"{dt}  SKIP ({e.filename})")

    print("-" * 96)
    st, rt = tally["stake"], tally["ret"]
    part = tally["bet"] / tally["races"] * 100 if tally["races"] else 0
    print(f"\n【券種別】")
    print(f"  {'種':4s} {'投資':>9s} {'払戻':>9s} {'収支':>10s} {'ROI':>7s}  {'的中/点':>9s}")
    for kind in ("単勝", "複勝", "馬連", "ワイド", "三連複"):
        if kind not in bykind:
            continue
        s, r, n, h = bykind[kind]
        print(f"  {kind:4s} ¥{int(s):>8,} ¥{int(r):>8,} ¥{int(r-s):>+9,} "
              f"{r/s*100 if s else 0:>6.1f}%  {h:>3d}/{n:<3d}")
    print("\n【合計】")
    print(f"  開催 {len([d for d in dates])} 日  レース {tally['races']}  "
          f"参加 {tally['bet']}  見送り {tally['miokuri']}  参加率 {part:.0f}%")
    print(f"  投資 ¥{int(st):,}  払戻 ¥{int(rt):,}  収支 ¥{int(rt-st):+,}  "
          f"**ROI {rt/st*100 if st else 0:.1f}%**")
    s = race_stats(recs)
    if s:
        print(f"\n【主指標 (レース粒度, n={s['n']})】")
        print(f"  的中率(何か当たった) {s['hit']:.1f}%   トリガミ率(的中中) {s['trig_of_hit']:.1f}% "
              f"(トリガミ{s['trig']} / クリーン{s['clean']})")
        print(f"  最大DD ¥{int(s['maxdd']):,} (通算収支カーブ)")
        print(f"  ROI CI95 [{s['ci'][0]:.1f}, {s['ci'][1]:.1f}]%  SE {s['se']:.1f}pt  "
              f"→ MDE(80%power/α5%)≈{s['mde']:.1f}pt")
        print(f"  ※ 差が MDE 未満の改善/悪化は本標本では検出不能 (非有意=効果なし ではない)")
    print("=" * 96)


if __name__ == "__main__":
    main()
