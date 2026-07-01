# -*- coding: utf-8 -*-
"""
multi_index.py — 多指数レイヤー (PyCaLi指数を主軸に、別角度の指数を並列表示)
================================================================================
印=PyCaLi指数(v6 score)一本だが、説明用に複数の指数で馬を順位付けして見せる。
全て既存資産の流用(新規学習なし)。ROIはv6を超えない(検証済)が、説明/表示に効く。

指数(各レース内 順位 1=良):
  PyCaLi指数 : v6 score (=現行の印)            higher良
  速度指数   : prev_hosei9 (前走補正タイム)     lower良(速い)
  対戦指数   : elo_T2M1_horse (ELOレート)       higher良
  調教指数   : trnH_Lap1 (坂路 終い200m)        lower良(速い)
  血統指数   : s3_sire_surfdist_top3_asof       higher良
  市場指数   : 単勝オッズ                        lower良(人気)
  複勝特化   : p_sho (校正 top-3 確率)           higher良

実行: PYTHONUTF8=1 ./venv311/Scripts/python.exe multi_index.py [rid]
"""
from __future__ import annotations
import glob, io, sys
from pathlib import Path
import joblib, numpy as np, pandas as pd
import backtest_pl_ev as BT
import pl_probs as PL

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
BASE = Path(__file__).parent
SCORE_CACHE = BASE / "data/_ev_grid_scores.parquet"
MASTER = BASE / "data/master_v2_20130105-20251228.csv"
V6_CAL = BASE / "models/pl_calibrators_v6.pkl"
COL_RID = "レースID(新/馬番無)"; COL_BAN = "馬番"


def rank_series(vals, ascending):
    """1=良 の順位。NaNは最後(順位末尾)。"""
    s = pd.Series(vals, dtype="float64")
    r = s.rank(ascending=ascending, method="min", na_option="bottom")
    return r.astype("Int64")


def load_odds_for(rid16):
    for f in sorted(glob.glob(str(BASE/"data/odds/odds_*.csv"))):
        df = pd.read_csv(f, encoding="cp932", low_memory=False)
        df["rid16"] = df["レースID(新)"].astype(str).str[:16]
        sub = df[df["rid16"] == rid16]
        if len(sub):
            return {int(b): float(o) for b, o in zip(pd.to_numeric(sub["馬番"], errors="coerce"),
                                                     pd.to_numeric(sub["単勝オッズ"], errors="coerce")) if b == b}
    return {}


def main():
    rid_arg = sys.argv[1] if len(sys.argv) > 1 else "2025060805030208"
    sc = pd.read_parquet(SCORE_CACHE); sc[COL_RID] = sc[COL_RID].astype(str)
    g = sc[sc[COL_RID] == rid_arg].copy()
    if g.empty:
        te = sc[(sc.get("year") == 2025) & (sc["_venue"] == "東京") & (sc["_surf"] == "芝")]
        rid_arg = te[COL_RID].value_counts().index[0]; g = sc[sc[COL_RID] == rid_arg].copy()
    rid16 = "".join(ch for ch in rid_arg if ch.isdigit())[:16]
    g["ban"] = pd.to_numeric(g[COL_BAN], errors="coerce").fillna(0).astype(int)
    g["rid16"] = rid16

    # ELO / 血統 join
    elo = pd.read_parquet(BASE/"data/elo_feats.parquet"); elo["rid16"]=elo["rid16"].astype(str)
    elo["ban"]=pd.to_numeric(elo["ban"],errors="coerce").fillna(0).astype(int)
    g = g.merge(elo[["rid16","ban","elo_T2M1_horse"]], on=["rid16","ban"], how="left")
    ser = pd.read_parquet(BASE/"data/feats_serious.parquet"); ser["rid16"]=ser["rid16"].astype(str)
    ser["ban"]=pd.to_numeric(ser["ban"],errors="coerce").fillna(0).astype(int)
    g = g.merge(ser[["rid16","ban","s3_sire_surfdist_top3_asof"]], on=["rid16","ban"], how="left")

    # master: 馬名/補正/調教
    mdf = pd.read_csv(MASTER, encoding="utf-8-sig", low_memory=False,
                      usecols=lambda c: c in [COL_RID,COL_BAN,"馬名","prev_hosei9","trnH_Lap1"])
    mdf[COL_RID]=mdf[COL_RID].astype(str); mdf["ban"]=pd.to_numeric(mdf[COL_BAN],errors="coerce").fillna(0).astype(int)
    md = mdf[mdf[COL_RID]==rid_arg]
    name = dict(zip(md["ban"], md["馬名"]))
    hosei = dict(zip(md["ban"], pd.to_numeric(md["prev_hosei9"],errors="coerce")))
    trn = dict(zip(md["ban"], pd.to_numeric(md["trnH_Lap1"],errors="coerce")))

    # p_sho (複勝特化)
    cal = joblib.load(V6_CAL)["calibrators"]
    g = g.sort_values("ban").reset_index(drop=True)
    w = PL.pl_weights(g["_score"].values)
    p_sho = cal["fukusho"].predict(BT.all_fukusho_vec_fast(w))
    odds = load_odds_for(rid16)

    n = len(g); ban = g["ban"].values
    idx = pd.DataFrame({"ban": ban, "name": [name.get(int(b),f"{b}番") for b in ban]})
    idx["PyCaLi"] = rank_series(g["_score"].values, ascending=False)
    idx["速度"]   = rank_series([hosei.get(int(b),np.nan) for b in ban], ascending=True)
    idx["対戦"]   = rank_series(g["elo_T2M1_horse"].values, ascending=False)
    idx["調教"]   = rank_series([trn.get(int(b),np.nan) for b in ban], ascending=True)
    idx["血統"]   = rank_series(g["s3_sire_surfdist_top3_asof"].values, ascending=False)
    idx["市場"]   = rank_series([odds.get(int(b),np.nan) for b in ban], ascending=True)
    idx["複勝"]   = rank_series(p_sho, ascending=False)
    idx = idx.sort_values("PyCaLi").reset_index(drop=True)

    MK = {1:"◎",2:"〇",3:"▲",4:"△",5:"△"}
    cols = ["PyCaLi","速度","対戦","調教","血統","市場","複勝"]
    print("="*84)
    print(f"多指数レイヤー  race={rid_arg}  (各列=レース内順位 1=最良)")
    print("="*84)
    print(f"{'印':3}{'馬番':>3} {'馬名':14} " + " ".join(f"{c:>4}" for c in cols))
    for _, r in idx.head(10).iterrows():
        mk = MK.get(int(r["PyCaLi"]), "")
        nm = (str(r["name"])[:13]+"…") if len(str(r["name"]))>14 else str(r["name"])
        print(f"{mk:3}{int(r['ban']):>3} {nm:14} " + " ".join(
            f"{'-' if pd.isna(r[c]) else int(r[c]):>4}" for c in cols))
    # 一致/不一致サマリ
    top = idx.iloc[0]  # ◎
    agree = [c for c in cols[1:] if not pd.isna(top[c]) and int(top[c])<=2]
    print("\n【◎の多指数評価】" + (f" {top['name']} は " + "・".join(agree) + " でも上位(鉄板度高)"
          if agree else f" {top['name']} は他指数では上位薄=PyCaLi突出型"))
    # 指数が割れている馬(妙味候補: 複勝or対戦で上位だがPyCaLi/市場で下)
    sleeper = idx[(idx["PyCaLi"]>5) & ((idx["複勝"]<=4) | (idx["対戦"]<=3))].head(3)
    if len(sleeper):
        print("【指数割れ(別レンズ上位)】" + " / ".join(
            f"{int(s['ban'])}{s['name']}(PyCaLi{int(s['PyCaLi'])}/複勝{'-' if pd.isna(s['複勝']) else int(s['複勝'])}/対戦{'-' if pd.isna(s['対戦']) else int(s['対戦'])})"
            for _, s in sleeper.iterrows()))
    print("="*84)
    print("※指数はレンズ。ROIはPyCaLi(v6)が最良で他は超えない(検証済)。表示/説明用。")


if __name__ == "__main__":
    main()
