# -*- coding: utf-8 -*-
"""
keiba_card.py — 競馬新聞風 能力5角形カード (PyCaLi指数 + 5軸能力指数)
================================================================================
印=PyCaLi指数(v6)を総合の軸に、競馬新聞風の5軸を★5段階(レース内相対)で表示。
全て既存資産の流用(新規学習なし)。ROIはv6が最良で他は超えない=「見せる/読む」用途。

  1 スピード+距離適性 : -prev_hosei9, -前走上り3F, kako5_same_dist_ratio, s3_sire_surfdist   [強]
  2 スタミナ+展開     : 前走RPCI, 脚質×ペース適性(s2_fit_front+closer)                       [中]
  3 勝負根性         : closing_power, -kako5_pos_vs_ninki(人気以上), -前走上り3F順           [proxy]
  4 気性(落ち着き)    : -kako5_std_pos, -g2_rd, ブリンカー無=加点                            [proxy]
  5 人気+実績        : -単勝オッズ, elo, -kako5_best_pos, kako5_race_count                  [強]

★ = レース内 相対(上位20%=★5 … 下位20%=★1)。
実行: PYTHONUTF8=1 ./venv311/Scripts/python.exe keiba_card.py [rid]
"""
from __future__ import annotations
import glob, io, sys
from pathlib import Path
import joblib, numpy as np, pandas as pd
import backtest_pl_ev as BT
import pl_probs as PL

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
BASE = Path(__file__).parent
SCORE_CACHE = BASE/"data/_ev_grid_scores.parquet"
MASTER = BASE/"data/master_v2_20130105-20251228.csv"
V6_CAL = BASE/"models/pl_calibrators_v6.pkl"
COL_RID="レースID(新/馬番無)"; COL_BAN="馬番"


def z(a):
    a = np.asarray(a, float); m = np.nanmean(a); s = np.nanstd(a)
    if not np.isfinite(s) or s == 0: return np.zeros_like(a)
    return np.nan_to_num((a - m) / s, nan=0.0)


def axis(*comps):
    """各成分(向きを揃え高いほど良)をz化し平均。全NaN成分は無視。"""
    zs = [z(c) for c in comps]
    return np.mean(zs, axis=0)


def stars(vals):
    """レース内 相対で★1-5 (上位20%=5)。"""
    s = pd.Series(vals); r = s.rank(ascending=False, method="min")  # 1=best
    n = len(s)
    q = np.ceil(r / n * 5).clip(1, 5)
    return (6 - q).astype(int)   # rank上位→★大


def col(g, c):
    return pd.to_numeric(g[c], errors="coerce").values if c in g else np.full(len(g), np.nan)


def main():
    rid = sys.argv[1] if len(sys.argv) > 1 else "2025060805030208"
    sc = pd.read_parquet(SCORE_CACHE); sc[COL_RID]=sc[COL_RID].astype(str)
    g = sc[sc[COL_RID]==rid].copy()
    if g.empty:
        te=sc[(sc.get("year")==2025)&(sc["_venue"]=="東京")&(sc["_surf"]=="芝")]
        rid=te[COL_RID].value_counts().index[0]; g=sc[sc[COL_RID]==rid].copy()
    rid16="".join(ch for ch in rid if ch.isdigit())[:16]
    g["ban"]=pd.to_numeric(g[COL_BAN],errors="coerce").fillna(0).astype(int); g["rid16"]=rid16
    for nm,cs in [("elo_feats",["elo_T2M1_horse"]),("glicko_feats",["g2_rd"]),
                  ("feats_serious",["s2_fit_front","s2_fit_closer","s3_sire_surfdist_top3_asof"])]:
        d=pd.read_parquet(BASE/f"data/{nm}.parquet"); d["rid16"]=d["rid16"].astype(str)
        d["ban"]=pd.to_numeric(d["ban"],errors="coerce").fillna(0).astype(int)
        g=g.merge(d[["rid16","ban"]+cs],on=["rid16","ban"],how="left")
    MCOLS=[COL_RID,COL_BAN,"馬名","prev_hosei9","前走上り3F","前走上り3F順","前走RPCI",
           "closing_power","kako5_same_dist_ratio","kako5_pos_vs_ninki","kako5_std_pos",
           "kako5_best_pos","kako5_race_count","ブリンカー"]
    md=pd.read_csv(MASTER,encoding="utf-8-sig",low_memory=False,usecols=lambda c:c in MCOLS)
    md[COL_RID]=md[COL_RID].astype(str); md["ban"]=pd.to_numeric(md[COL_BAN],errors="coerce").fillna(0).astype(int)
    g=g.merge(md[md[COL_RID]==rid].drop(columns=[COL_RID]),on="ban",how="left",suffixes=("","_m"))
    # odds
    odds={}
    for f in sorted(glob.glob(str(BASE/"data/odds/odds_*.csv"))):
        df=pd.read_csv(f,encoding="cp932",low_memory=False); df["r16"]=df["レースID(新)"].astype(str).str[:16]
        sub=df[df["r16"]==rid16]
        if len(sub): odds={int(b):float(o) for b,o in zip(pd.to_numeric(sub["馬番"],errors="coerce"),pd.to_numeric(sub["単勝オッズ"],errors="coerce")) if b==b}; break
    g=g.sort_values("ban").reset_index(drop=True); n=len(g); ban=g["ban"].values
    odds_arr=np.array([odds.get(int(b),np.nan) for b in ban])
    nobl=np.array([0.0 if (isinstance(x,str) and x.strip() not in("","nan")) else 1.0 for x in g.get("ブリンカー",[np.nan]*n)])

    A1=axis(-col(g,"prev_hosei9"), -col(g,"前走上り3F"), col(g,"kako5_same_dist_ratio"), col(g,"s3_sire_surfdist_top3_asof"))
    A2=axis(col(g,"前走RPCI"), col(g,"s2_fit_front")+np.nan_to_num(col(g,"s2_fit_closer")))
    A3=axis(col(g,"closing_power"), -col(g,"kako5_pos_vs_ninki"), -col(g,"前走上り3F順"))
    A4=axis(-col(g,"kako5_std_pos"), -col(g,"g2_rd"), nobl)
    A5=axis(-odds_arr, col(g,"elo_T2M1_horse"), -col(g,"kako5_best_pos"), col(g,"kako5_race_count"))

    M5=["◎","〇","▲","△","△"]
    def amark_total(vals):   # 総合(PyCaLi): ◎〇▲△△ top5 (従来通り)
        r=pd.Series(vals).rank(ascending=False,method="first")
        return [M5[int(x)-1] if x<=5 else "・" for x in r]
    def amark(vals):         # 各ステータス: ◎1/〇1/▲1/△(4-8位,最大5頭)/・。◎〇▲重複なし
        r=pd.Series(vals).rank(ascending=False,method="first")
        def mk(x):
            x=int(x)
            return "◎" if x==1 else "〇" if x==2 else "▲" if x==3 else "△" if x<=8 else "・"
        return [mk(x) for x in r]
    df=pd.DataFrame({"ban":ban,"name":g.get("馬名",pd.Series([f"{b}番" for b in ban]))})
    df["PyCaLi_r"]=pd.Series(g["_score"].values).rank(ascending=False,method="first").astype(int)
    df["総合"]=amark_total(g["_score"].values)
    df["速"]=amark(A1); df["スタ"]=amark(A2); df["根"]=amark(A3); df["気"]=amark(A4); df["実"]=amark(A5)
    df=df.sort_values("PyCaLi_r").reset_index(drop=True)
    print("="*86)
    print(f"競馬新聞風 多指数カード  race={rid}")
    print("各指数が独立に ◎〇▲△ を付与 (総合=PyCaLi指数v6)")
    print("="*86)
    print(f"{'総合':4}{'馬番':>3} {'馬名':15}{'スピード':6}{'スタミナ':6}{'根性':5}{'気性':5}{'人気実績':7}")
    for _,r in df.iterrows():
        nm=(str(r['name'])[:14]+"…") if len(str(r['name']))>15 else str(r['name'])
        print(f"{r['総合']:4}{int(r['ban']):>3} {nm:15}{r['速']:6}{r['スタ']:6}{r['根']:5}{r['気']:5}{r['実']:7}")
    print("="*86)
    print("指数: スピード=補正/上り/距離適性 ｜ スタミナ=RPCI/脚質×展開 ｜ 根性=終い脚/人気以上(proxy)")
    print("      気性=着順安定/評価確信/ブリンカー(proxy) ｜ 人気実績=オッズ/ELO/最高着順/経験")
    print("※3根性・4気性は直接データ無の近接proxy。総合(印)=v6が担い、ROIは他指数で超えない。")


if __name__=="__main__":
    main()
