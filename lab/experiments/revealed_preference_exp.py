# -*- coding: utf-8 -*-
"""
revealed_preference_exp.py — 「行動が漏らす私的情報」(顕示選好) 理論の検定 (2026-11-14)
=================================================================================
理論: 市場の外にある情報の多くは、厩舎・騎手の頭の中にある。それは言葉では出てこないが、
      「選ぶ」という行動には漏れる。市場が騎手の名前や乗り替わりは見ていても、
      「同じレースで何を捨てて何を選んだか」という構造までは値段に入れていないなら、そこに歪みが残る。

T1 騎手の選択: 前走で馬 A に乗った騎手 J が、同じレースで別の馬 B に乗っている。
    → B は「J が A より選んだ馬」(chosen)、A は「J に捨てられた馬」(abandoned)。
    J は両方の馬の感触を知っている数少ない人間。
T2 厩舎の二頭出し: 同じ調教師の馬が 2 頭以上いるレースで、as-of 勝率の高い騎手が乗る方 (stable_pick) と
    もう一方 (stable_other)。厩舎は自分の馬の状態を一番よく知っている。
T3 騎手の格上げ: 前走より as-of 勝率が大きく高い騎手に替わった馬 (upgrade)。陣営が勝負に来ている合図。

測り方 (モデル不要、市場 vs 結果だけ):
  - 市場勝率 = 確定単勝オッズ逆数のレース内正規化。各群で 実勝数 / 市場勝率の和 (= 1 なら市場は正しい)。
  - 単勝・複勝の均等買い回収率と、同じオッズ帯の全馬の回収率からの残差 (オッズ帯構成を除く)。
  - 期間は 2013-19 / 2020-25 (コロナ前後で必ず切る、feedback) の 2 期で同じ向きかを見る。
  - CI は開催日クラスタ bootstrap。
事前の判定: 2 期とも「実勝/市場」の CI が 1 を跨がない、かつ同じ向き → 市場の外の情報 (SIGNAL)。
           回収率が 100% を超えるかは別問題 (控除 20%)。
実行: E:/PyCaLiAI/venv311/Scripts/python.exe lab/experiments/revealed_preference_exp.py
出力: reports/revealed_preference_exp.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("E:/PyCaLiAI")
MASTER = ROOT / "data" / "master_v2_20130105-20251228.csv"
KEKKA = ROOT / "data" / "kekka_20130105-20251228.csv"
EXT = Path(r"E:\競馬過去走データ\raw_data\kekka_1986_2025_enhanced.csv")
OUT = Path(__file__).resolve().parents[2] / "reports" / "revealed_preference_exp.json"
PLACE_CODE = {"札幌": "01", "函館": "02", "福島": "03", "新潟": "04", "東京": "05", "中山": "06",
              "中京": "07", "京都": "08", "阪神": "09", "小倉": "10"}
BANDS = [1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0, 1e9]
N_BOOT = 1000
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:5.0f}s] {m}", flush=True)


def load_odds():
    use = ["年", "月", "日", "回次", "場所", "日次", "レース番号", "馬番", "単勝オッズ"]
    d = pd.read_csv(EXT, encoding="cp932", usecols=use, dtype=str, low_memory=False)
    for c in ("年", "月", "日", "回次", "日次", "レース番号", "馬番"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["年", "月", "日", "回次", "日次", "レース番号", "馬番"])
    yy = d["年"].astype(int)
    d["year"] = np.where(yy < 100, 2000 + yy, yy)
    d = d[d.year >= 2013]
    d["pcode"] = d["場所"].astype(str).str.strip().map(PLACE_CODE)
    d = d.dropna(subset=["pcode"])
    d["rid"] = (d.year.astype(str) + d["月"].astype(int).map("{:02d}".format) + d["日"].astype(int).map("{:02d}".format)
                + d.pcode + d["回次"].astype(int).map("{:02d}".format) + d["日次"].astype(int).map("{:02d}".format)
                + d["レース番号"].astype(int).map("{:02d}".format))
    d["ban"] = d["馬番"].astype(int)
    d["odds"] = pd.to_numeric(d["単勝オッズ"], errors="coerce")
    return d[["rid", "ban", "odds"]].drop_duplicates(["rid", "ban"])


def load_pay():
    k = pd.read_csv(KEKKA, encoding="cp932", low_memory=False)
    k.columns = ["rid_horse", "ban", "ped", "jyun", "tansho", "fukusho", "wakuren",
                 "umaren", "umatan", "sanrenpuku", "sanrentan"]
    k["rid"] = k["rid_horse"].astype(str).str[:16]
    k["ban"] = pd.to_numeric(k["ban"], errors="coerce")
    k = k.dropna(subset=["ban"]); k["ban"] = k["ban"].astype(int)
    k["jyun_k"] = pd.to_numeric(k["jyun"], errors="coerce")
    k["tansho"] = np.where(k["jyun_k"] == 1, pd.to_numeric(k["tansho"], errors="coerce"), 0.0)
    k["fukusho"] = pd.to_numeric(k["fukusho"], errors="coerce")
    return k[["rid", "ban", "tansho", "fukusho"]].drop_duplicates(["rid", "ban"])


def load_master():
    use = ["日付", "レースID(新)", "レースID(新/馬番無)", "馬番", "確定着順", "騎手コード", "調教師コード", "前走レースID(新)"]
    m = pd.read_csv(MASTER, encoding="utf-8-sig", usecols=lambda c: c in use or c == "着順", dtype=str, low_memory=False)
    jcol = "確定着順" if "確定着順" in m.columns else "着順"
    m["jyun"] = pd.to_numeric(m[jcol], errors="coerce")
    m["rid"] = m["レースID(新/馬番無)"].astype(str).str[:16]
    m["ban"] = pd.to_numeric(m["馬番"], errors="coerce")
    m = m.dropna(subset=["ban", "jyun"]); m["ban"] = m["ban"].astype(int)
    m["date"] = pd.to_datetime(m["日付"].astype(str).str[:8], format="%Y%m%d", errors="coerce")
    m["jk"] = m["騎手コード"].astype(str).str.strip()
    m["tr"] = m["調教師コード"].astype(str).str.strip()
    m["rid18"] = m["レースID(新)"].astype(str).str.strip()
    m["prev18"] = m["前走レースID(新)"].astype(str).str.strip()
    return m[["rid", "ban", "date", "jyun", "jk", "tr", "rid18", "prev18"]]


def asof_jockey_winrate(m):
    """騎手の as-of 勝率 (前日までの全成績、Beta(1,12) 平滑)。同日は含めない。"""
    d = m[["date", "jk", "jyun"]].copy(); d["w"] = (d.jyun == 1).astype(float); d["n"] = 1.0
    daily = d.groupby(["jk", "date"])[["w", "n"]].sum().sort_index()
    cum = daily.groupby(level=0).cumsum() - daily          # 前日までの累積
    rate = (cum["w"] + 1.0) / (cum["n"] + 13.0)
    return m.join(rate.rename("jk_rate"), on=["jk", "date"])["jk_rate"].values


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    m = load_master(); log(f"master {len(m):,}")
    m["jk_rate"] = asof_jockey_winrate(m)
    prev = m.set_index("rid18")[["jk", "jk_rate"]].rename(columns={"jk": "prev_jk", "jk_rate": "prev_jk_rate_then"})
    prev = prev[~prev.index.duplicated()]
    m = m.join(prev, on="prev18")
    # 前走騎手の「今日の」as-of 勝率 (乗り替わりの格上げ判定は同じ日付の物差しで比べる)
    m = m.merge(m[["date", "jk", "jk_rate"]].drop_duplicates(["date", "jk"]).rename(columns={"jk": "prev_jk", "jk_rate": "prev_jk_rate_now"}),
                on=["date", "prev_jk"], how="left")
    m = m.merge(load_odds(), on=["rid", "ban"], how="left").merge(load_pay(), on=["rid", "ban"], how="left")
    g = m.groupby("rid")
    ok = (g["odds"].transform(lambda s: s.notna().all() and (s > 0).all())
          & (g["jyun"].transform(lambda s: (s == 1).sum()) == 1) & (g["ban"].transform("size") >= 5))
    m = m[ok].copy(); log(f"races {m.rid.nunique():,} horses {len(m):,}")
    inv = 1.0 / m["odds"]; m["p_mkt"] = inv / inv.groupby(m["rid"]).transform("sum")
    m["win"] = (m.jyun == 1).astype(float)
    m["tansho"] = m["tansho"].fillna(0.0); m["fukusho"] = m["fukusho"].fillna(0.0)
    m["band"] = pd.cut(m["odds"], BANDS, right=False, labels=False)
    m["year"] = m["date"].dt.year
    m["period"] = np.where(m.year <= 2019, "2013-19", "2020-25")

    # --- T1 騎手の選択
    riders = m.groupby("rid")["jk"].apply(set)
    m["prev_jk_in_race"] = [ (pj in riders.get(r, ())) for r, pj in zip(m.rid, m.prev_jk) ]
    switched = m.prev_jk.notna() & (m.prev_jk != m.jk) & (m.prev_jk != "nan")
    m["abandoned"] = switched & m.prev_jk_in_race
    # chosen: 今乗っている騎手 J が、同じレースにいる別の馬 A の前走騎手でもある (J は A を降りてこちらを選んだ)
    ab = m[m.abandoned][["rid", "prev_jk"]].drop_duplicates()
    ab["chosen"] = True
    m = m.merge(ab.rename(columns={"prev_jk": "jk"}), on=["rid", "jk"], how="left")
    m["chosen"] = m["chosen"].fillna(False).astype(bool)
    # --- T2 厩舎の二頭出し
    ntr = m.groupby(["rid", "tr"])["ban"].transform("size")
    multi = ntr >= 2
    best = m[multi].groupby(["rid", "tr"])["jk_rate"].transform("max")
    m["stable_pick"] = False; m["stable_other"] = False
    idx = m.index[multi]
    top = m.loc[idx, "jk_rate"].values >= best.values - 1e-12
    uniq = m.loc[idx].assign(top=top).groupby(["rid", "tr"])["top"].transform("sum").values == 1  # 同率は除外
    m.loc[idx[top & uniq], "stable_pick"] = True
    m.loc[idx[(~top) & uniq], "stable_other"] = True
    # --- T3 騎手の格上げ / 格下げ (同じ日の as-of 勝率で比べる)
    d_rate = m["jk_rate"] - m["prev_jk_rate_now"]
    m["upgrade"] = switched & (d_rate >= 0.05)
    m["downgrade"] = switched & (d_rate <= -0.05)

    # オッズ帯ごとの基準回収率 (期間別)
    base = m.groupby(["period", "band"]).agg(bt=("tansho", "mean"), bf=("fukusho", "mean"))
    m = m.join(base, on=["period", "band"])
    m["rt"] = m["tansho"] - m["bt"]; m["rf"] = m["fukusho"] - m["bf"]

    groups = ["chosen", "abandoned", "stable_pick", "stable_other", "upgrade", "downgrade"]
    res = {"n_races": int(m.rid.nunique()), "groups": {}}
    rng = np.random.default_rng(0)
    for per in ("2013-19", "2020-25"):
        mp = m[m.period == per]
        dates = mp["date"].values; ud, dc = np.unique(dates, return_inverse=True)
        I = rng.integers(0, len(ud), (N_BOOT, len(ud)))
        for gname in groups:
            s = mp[mp[gname]]
            if len(s) < 200:
                continue
            dci = dc[np.flatnonzero(mp[gname].values)]
            def boot(num, den):
                a = np.bincount(dci, weights=num, minlength=len(ud)); b = np.bincount(dci, weights=den, minlength=len(ud))
                v = a[I].sum(1) / np.maximum(b[I].sum(1), 1e-9)
                return [float(num.sum() / den.sum()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            one = np.ones(len(s))
            r = {"n": int(len(s)), "mean_odds_med": float(s.odds.median()),
                 "win_over_mkt": boot(s.win.values, s.p_mkt.values),
                 "roi_tan": boot(s.tansho.values, 100 * one), "roi_fuku": boot(s.fukusho.values, 100 * one),
                 "resid_tan_pt": boot(s.rt.values, 100 * one), "resid_fuku_pt": boot(s.rf.values, 100 * one)}
            r["roi_tan"] = [100 * x for x in r["roi_tan"]]; r["roi_fuku"] = [100 * x for x in r["roi_fuku"]]
            r["resid_tan_pt"] = [100 * x for x in r["resid_tan_pt"]]; r["resid_fuku_pt"] = [100 * x for x in r["resid_fuku_pt"]]
            res["groups"].setdefault(gname, {})[per] = r
            w = r["win_over_mkt"]
            log(f"{per} {gname:13s} n={r['n']:6,d} 実勝/市場 {w[0]:.3f} [{w[1]:.3f},{w[2]:.3f}]  "
                f"単 {r['roi_tan'][0]:5.1f}% (帯残差 {r['resid_tan_pt'][0]:+5.1f}pt [{r['resid_tan_pt'][1]:+.1f},{r['resid_tan_pt'][2]:+.1f}])  "
                f"複 {r['roi_fuku'][0]:5.1f}% (帯残差 {r['resid_fuku_pt'][0]:+5.1f}pt [{r['resid_fuku_pt'][1]:+.1f},{r['resid_fuku_pt'][2]:+.1f}])")
    verdict = {}
    for gname, per in res["groups"].items():
        if len(per) < 2:
            continue
        a, b = per["2013-19"]["win_over_mkt"], per["2020-25"]["win_over_mkt"]
        same = (a[1] > 1 and b[1] > 1) or (a[2] < 1 and b[2] < 1)
        verdict[gname] = "SIGNAL" if same else "none"
    res["verdict"] = verdict
    log(f"verdict {verdict}")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"saved {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
