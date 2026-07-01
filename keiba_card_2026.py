# -*- coding: utf-8 -*-
"""
keiba_card_2026.py — 競馬新聞風 多指数カード(2026ライブ版, bundle履歴から5軸)
================================================================================
2026は score cache/ELO parquet 未被覆。bundle の per-horse history(過去走) から5軸を組む。
各指数が独立に ◎〇▲△ を付与。総合=PyCaLi指数(v6 ai_score)。

  スピード/距離 : 過去走 best上り3F(速い) + 同距離好走率(same_dist_ratio)
  スタミナ/展開 : 同コース種別率(same_td_ratio) + 脚質×レースペース適性
  勝負根性     : 人気以上の好走(平均 ninki-pos) + 終い脚(best agari)   [proxy]
  気性(落ち着き): 着順の安定(std小) + 出遅れ少(deogure_count小)         [proxy]
  人気実績     : -単勝オッズ + -best_pos + 経験n_runs + p_sho

実行: PYTHONUTF8=1 ./venv311/Scripts/python.exe keiba_card_2026.py <rid or 宝塚記念>
"""
from __future__ import annotations
import glob, io, json, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
BASE = Path(__file__).parent
FRONT = {"逃げ", "先行"}


def z(a):
    a=np.asarray(a,float); m=np.nanmean(a); s=np.nanstd(a)
    return np.zeros_like(a) if (not np.isfinite(s) or s==0) else np.nan_to_num((a-m)/s,nan=0.0)


def axis(*c): return np.mean([z(x) for x in c], axis=0)


def find_race(q):
    for f in sorted(glob.glob(str(BASE/"reports/cowork_input/*_bundle.json"))):
        b=json.load(open(f,encoding="utf-8"))
        for r in b["races"]:
            rid=str(r.get("race_id","")); nm=str(r.get("race_meta",{}).get("race_name",""))
            if q==rid or (q in nm and nm):
                return f[-20:-12], r
    return None, None


def horse_feats(h):
    hist=h.get("history") or {}; runs=hist.get("runs") or []
    ag=[rr.get("agari3f") for rr in runs if rr.get("agari3f")]
    pos=[rr.get("pos") for rr in runs if rr.get("pos")]
    nin=[rr.get("ninki") for rr in runs if rr.get("ninki")]
    sty=[rr.get("style") for rr in runs if rr.get("style")]
    style=sty[0] if sty else None
    return {
        "best_agari": min(ag) if ag else np.nan,
        "same_dist": hist.get("same_dist_ratio", np.nan),
        "same_td": hist.get("same_td_ratio", np.nan),
        "overperf": np.nanmean([n-p for n,p in zip(nin,pos)]) if (nin and pos) else np.nan,
        "pos_std": np.nanstd(pos) if len(pos)>=2 else np.nan,
        "deogure": hist.get("deogure_count", np.nan),
        "best_pos": hist.get("best_pos", np.nan),
        "n_runs": hist.get("n_runs", 0) or 0,
        "style": style, "odds": h.get("tansho_odds"), "p_sho": h.get("p_sho", np.nan),
        "ai_score": h.get("ai_score", np.nan), "mark": h.get("mark","") or "",
        "ban": h.get("umaban"), "name": h.get("horse_name",""),
    }


def main():
    q = sys.argv[1] if len(sys.argv)>1 else "宝塚記念"
    date, race = find_race(q)
    if race is None:
        print(f"レース '{q}' が bundle に無い"); return
    rm=race.get("race_meta",{})
    F=[horse_feats(h) for h in race.get("horses",[])]
    n=len(F)
    style_front = np.array([1.0 if (f["style"] in FRONT) else 0.0 for f in F])
    n_front = style_front.sum()
    pace_press = n_front/max(n,1)
    # 脚質×ペース適性: 先行馬は高ペースで不利, 差し/追込は有利
    fit = np.array([(-pace_press if f["style"] in FRONT else (pace_press if f["style"] in ("差し","追込") else 0.0)) for f in F])

    g=lambda k:[f[k] for f in F]
    A1=axis([-x if x==x else np.nan for x in g("best_agari")], g("same_dist"))
    A2=axis(g("same_td"), fit)
    A3=axis(g("overperf"), [-x if x==x else np.nan for x in g("best_agari")])
    A4=axis([-x if x==x else np.nan for x in g("pos_std")], [-x if x==x else np.nan for x in g("deogure")])
    A5=axis([-(x or np.nan) for x in g("odds")], [-(x if x==x else np.nan) for x in g("best_pos")],
            g("n_runs"), g("p_sho"))

    def amark(v):  # 各ステータス: ◎1/〇1/▲1/△(4-8位,最大5頭)/・(9位〜)。◎〇▲重複なし
        r=pd.Series(v).rank(ascending=False,method="first")
        def mk(x):
            x=int(x)
            return "◎" if x==1 else "〇" if x==2 else "▲" if x==3 else "△" if x<=8 else "・"
        return [mk(x) for x in r]
    df=pd.DataFrame({"ban":g("ban"),"name":g("name"),"mark":g("mark"),
                     "sr":pd.Series(g("ai_score")).rank(ascending=False,method="first")})
    df["速"]=amark(A1); df["スタ"]=amark(A2); df["根"]=amark(A3); df["気"]=amark(A4); df["実"]=amark(A5)
    df=df.sort_values("sr").reset_index(drop=True)

    print("="*88)
    print(f"競馬新聞風 多指数カード  {rm.get('place')}{rm.get('course')} {rm.get('race_name')} ({date}, {n}頭)")
    print("各指数が独立に◎〇▲△を付与 / 総合=PyCaLi指数(v6)。2026=bundle履歴から算出")
    print("="*88)
    print(f"{'総合':4}{'馬番':>3} {'馬名':16}{'スピード':6}{'スタミナ':6}{'根性':5}{'気性':5}{'人気実績':7}")
    for _,r in df.iterrows():
        nm=(str(r['name'])[:15]+"…") if len(str(r['name']))>16 else str(r['name'])
        print(f"{(r['mark'] or '・'):4}{int(r['ban']):>3} {nm:16}{r['速']:6}{r['スタ']:6}{r['根']:5}{r['気']:5}{r['実']:7}")
    print("="*88)
    print("※根性・気性は直接データ無の近接proxy(過去走の人気以上好走/着順安定から)。")

    # 結果(過去日なら)
    kf=BASE/f"data/kekka/{date}.csv"
    if kf.exists():
        k=pd.read_csv(kf,encoding="cp932",low_memory=False); k.columns=[str(c).strip() for c in k.columns]
        k["rid16"]=k["レースID(新)"].astype(str).str[:16]; k["着"]=pd.to_numeric(k["確定着順"],errors="coerce")
        rr=k[k["rid16"]==str(race.get("race_id"))[:16]].sort_values("着").head(3)
        if len(rr):
            print("確定: "+" ".join(f"{int(x['着'])}着{int(x['馬番'])}{x['馬名']}" for _,x in rr.iterrows()))


if __name__=="__main__":
    main()
