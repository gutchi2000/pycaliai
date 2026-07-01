# -*- coding: utf-8 -*-
"""
wide_holdout_2026.py — ワイドAI 最良config を 2026実データでホールドアウト + トリガミ分解
================================================================================
wide_lab.py の最良 OOS config = ◎軸 / 2点 / 堅いレース(field_chaos_score<=0.7943)。
= 実質「◎-〇, ◎-▲ のワイド2点を、堅い回だけ買う」。これを 2026(4-6月) の bundle pair_probs
(calibrated p_wide) + wide_kekka.csv 実配当 で検証(完全未使用データ=真のホールドアウト)。

トリガミ分解: 的中したのに race 収支がマイナス(小配当)の回を数える=なぜ86%止まりかの解剖。
実行: PYTHONUTF8=1 ./venv311/Scripts/python.exe wide_holdout_2026.py [chaos_max]
"""
from __future__ import annotations
import io, json, re, sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import pandas as pd

BASE = Path(__file__).parent
CHAOS_MAX = float(sys.argv[1]) if len(sys.argv) > 1 else 0.7943
POOL = 3; CAP = 2; BET = 100


def _n(x):
    try: return float(str(x).replace(",", "")) if str(x).strip() not in ("", "nan") else 0.0
    except Exception: return 0.0


def load_wide(date):
    y, m, d = int(date[:4]), int(date[4:6]), int(date[6:8])
    df = pd.read_csv(BASE/"data/kekka/wide_kekka.csv", encoding="cp932", header=None, low_memory=False)
    out = {}
    for _, r in df.iterrows():
        try:
            if int(r[0]) != y or int(r[1]) != m or int(r[2]) != d: continue
        except Exception:
            continue
        place=str(r[3]); R=int(r[4]); pairs={}
        for part in str(r[9]).split("/"):
            mm=re.search(r"(\d+)\s*-\s*(\d+)\s*\\?\s*([\d,]+)", part)
            if mm: pairs[frozenset((int(mm.group(1)),int(mm.group(2))))]=_n(mm.group(3))
        if pairs: out[(place,R)]=pairs
    return out


def main():
    bundles = sorted((BASE/"reports/cowork_input").glob("*_bundle.json"))
    tot_st=tot_rt=0; n_race=0; n_hit=0; n_torigami=0; per_day={}
    print("="*86)
    print(f"ワイドAI ホールドアウト 2026  [◎軸 {CAP}点 / 堅いレース chaos<={CHAOS_MAX} / flat¥{BET}]")
    print("="*86)
    for bf in bundles:
        date = bf.name[:8]
        wide = load_wide(date)
        if not wide: continue
        b = json.loads(bf.read_text(encoding="utf-8"))
        d_st=d_rt=d_n=d_hit=0
        for race in b["races"]:
            rc = race.get("race_confidence", {})
            chaos = rc.get("field_chaos_score")
            if chaos is None or chaos > CHAOS_MAX: continue        # 堅いレースのみ
            rm = race.get("race_meta", {})
            place = rm.get("place"); rid = str(race.get("race_id",""))
            R = int(rid[-2:]) if rid[-2:].isdigit() else None
            wk = wide.get((place, R))
            if wk is None: continue
            horses = race.get("horses", [])
            hon = next((h for h in horses if h.get("mark")=="◎"), None)
            if not hon: continue
            hon_ban = hon.get("umaban")
            aite = [h.get("umaban") for h in sorted(horses, key=lambda x:(x.get("ai_rank") or 99))
                    if (h.get("ai_rank") or 99) <= POOL and h.get("umaban") != hon_ban]
            pp = race.get("pair_probs", {})
            cands = []
            for x in aite:
                a, c = sorted((int(hon_ban), int(x)))
                key = f"{a}-{c}"
                pw = (pp.get(key, {}) or {}).get("wide")
                if pw is not None:
                    cands.append((pw, frozenset((int(hon_ban), int(x)))))
            cands.sort(key=lambda z: -z[0])
            sel = cands[:CAP]
            if not sel: continue
            stake = len(sel)*BET; ret = sum(wk.get(fs, 0.0)*(BET/100.0) for _pw, fs in sel)
            hit = ret > 0
            n_race+=1; d_n+=1; tot_st+=stake; tot_rt+=ret; d_st+=stake; d_rt+=ret
            if hit: n_hit+=1; d_hit+=1
            if hit and ret < stake: n_torigami+=1     # 的中but race赤(トリガミ)
        if d_n: per_day[date]=(d_n, d_st, d_rt, d_hit)

    print(f"\n{'日付':10} {'R数':>4} {'投資':>9} {'払戻':>9} {'ROI':>7} {'的中R':>6}")
    for date,(dn,ds,dr,dh) in sorted(per_day.items()):
        print(f"{date:10} {dn:>4} ¥{int(ds):>7,} ¥{int(dr):>7,} {dr/ds*100 if ds else 0:>6.1f}% {dh:>4}/{dn}")
    print("-"*86)
    roi = tot_rt/tot_st*100 if tot_st else 0
    print(f"{'合計':10} {n_race:>4} ¥{int(tot_st):>7,} ¥{int(tot_rt):>7,} {roi:>6.1f}% {n_hit}/{n_race}")
    print(f"\n的中率(レース) {n_hit/n_race*100 if n_race else 0:.1f}%  "
          f"トリガミ(的中but赤字)回 {n_torigami} ({n_torigami/max(n_hit,1)*100:.1f}% of 的中)")
    print(f"収支 ¥{int(tot_rt-tot_st):+,}  ROI {roi:.1f}%  ({'✅黒字' if roi>=100 else '控除床以下'})")
    print("="*86)


if __name__ == "__main__":
    main()
