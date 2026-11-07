# -*- coding: utf-8 -*-
"""
market_bias_atlas.py — 市場の癖の地図 (2026-11-07)
================================================================================
モデルを一切使わず「市場 (確定単勝オッズ) vs 結果 (払戻)」だけで、発走前に分かる属性ごとに
100 円均等買い (単勝・複勝) の回収率を測る。狙いは「市場が系統的に買いすぎ / 買わなさすぎる属性」。

入力 (すべて読み取り):
  E:\\競馬過去走データ\\raw_data\\kekka_2010_2025_enhanced_from_jv.csv  全頭の確定単勝オッズ・人気・騎手名・頭数・枠 等
  data/kekka_20130105-20251228.csv                                       上位 3 頭の複勝配当 (レースID 18 桁)
  data/master_v2_20130105-20251228.csv                                   間隔 / 休み明け / 前走着順 / 前距離 / 前芝ダ / Ｒ / 天気 / 限定
設計:
  * 発見期 2013-2020、確認期 2021-2025。発見期で目についたものが確認期でも同じ向きかを見る。
  * 「本命-大穴バイアス」はオッズ帯そのものの癖なので、各属性の回収率からオッズ帯構成で期待される回収率を引いた
    残差 (pt) を「その属性に固有の癖」とする。
  * CI は開催日クラスタ bootstrap (2,000 回)。確認期の残差 ≠ 0 を両側検定し BH-FDR (q<0.10) で選ぶ。
  * 順位付けをしないので同率タイブレークの先読み漏れ (project_bet_substrate_tie_leak) は起きない。
出力: reports/market_bias_atlas.md / .json
実行: python -m analysis.market_bias_atlas [--boot 2000]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
EXT = Path(r"E:\競馬過去走データ\raw_data\kekka_1986_2025_enhanced.csv")   # 全頭 (単勝オッズ・人気順・騎手名・頭数)。from_jv 版はレースIDが float で壊れている
KEKKA = BASE / "data" / "kekka_20130105-20251228.csv"
MASTER = BASE / "data" / "master_v2_20130105-20251228.csv"
OUT_MD = BASE / "reports" / "market_bias_atlas.md"
OUT_JSON = BASE / "reports" / "market_bias_atlas.json"
DISC = (2013, 2020)
CONF = (2021, 2025)
MIN_N = 400          # セグメント最小ベット数 (期ごと)


def log(msg: str) -> None:
    print(msg, flush=True)


# ------------------------------------------------------------------ 読み込み
def _col(cols, *keys):
    for k in keys:
        for c in cols:
            if k == c or (k in c and c.startswith("raw__")):
                return c
    raise KeyError(keys)


PLACE_CODE = {"札幌": "01", "函館": "02", "福島": "03", "新潟": "04", "東京": "05", "中山": "06", "中京": "07", "京都": "08", "阪神": "09", "小倉": "10"}


def load_ext() -> pd.DataFrame:
    use = ["年", "月", "日", "回次", "場所", "日次", "レース番号", "芝・ダ", "距離", "性別", "年齢", "騎手名", "斤量", "頭数", "馬番",
           "確定着順", "異常コード", "人気順", "調教師", "単勝オッズ", "クラスコード"]
    df = pd.read_csv(EXT, encoding="cp932", usecols=use, dtype=str, low_memory=False)
    df = df.rename(columns={"芝・ダ": "surface", "距離": "dist", "性別": "sex", "年齢": "age", "騎手名": "jockey", "斤量": "kin",
                            "頭数": "field", "馬番": "ban", "確定着順": "fin", "異常コード": "abn", "人気順": "pop", "調教師": "trainer",
                            "単勝オッズ": "odds", "場所": "place"})
    for c in ("年", "月", "日", "回次", "日次", "レース番号"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["年", "月", "日", "回次", "日次", "レース番号"])
    yy = df["年"].astype(int)
    df["year"] = np.where(yy < 100, 2000 + yy, yy)
    df = df[(df.year >= DISC[0]) & (df.year <= CONF[1])].copy()
    df["date"] = df.year.astype(str) + df["月"].astype(int).map("{:02d}".format) + df["日"].astype(int).map("{:02d}".format)
    df["pcode"] = df.place.astype(str).str.strip().map(PLACE_CODE)
    df = df.dropna(subset=["pcode"])
    df["rid16"] = (df.date + df.pcode + df["回次"].astype(int).map("{:02d}".format) + df["日次"].astype(int).map("{:02d}".format)
                   + df["レース番号"].astype(int).map("{:02d}".format))
    for c in ("odds", "pop", "fin", "field", "ban", "age", "kin", "dist", "abn"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["rid18"] = df.rid16 + df.ban.fillna(0).astype(int).map("{:02d}".format)
    df = df[df.odds.notna() & (df.odds > 0) & df.fin.notna() & (df.fin >= 1) & (df.abn.fillna(0) == 0)]
    df = df[df.surface.isin(["芝", "ダ"])]
    df["jockey"] = df["jockey"].astype(str).str.strip()
    df["trainer"] = df["trainer"].astype(str).str.strip()
    df["Ｒ"] = df["レース番号"].astype(int)
    return df


def load_place_pay() -> pd.DataFrame:
    k = pd.read_csv(KEKKA, encoding="cp932", usecols=["レースID(新)", "複勝配当"], dtype=str)
    k["rid18"] = k["レースID(新)"].astype(str).str[:18]
    k["fpay"] = pd.to_numeric(k["複勝配当"], errors="coerce")
    return k[["rid18", "fpay"]].dropna()


def load_master() -> pd.DataFrame:
    cols = ["レースID(新/馬番無)", "馬番", "間隔", "休み明け～戦目", "前走確定着順", "前距離", "前芝・ダ",
            "天気", "馬場状態", "性別限定", "年齢限定", "クラス名", "枠番", "ブリンカー", "前走馬体重増減"]
    m = pd.read_csv(MASTER, encoding="utf-8-sig", usecols=cols, low_memory=False)
    m = m.rename(columns={"レースID(新/馬番無)": "rid16", "馬番": "ban"})
    m["rid16"] = m["rid16"].astype(str).str[:16]
    m["ban"] = pd.to_numeric(m["ban"], errors="coerce")
    for c in ("間隔", "休み明け～戦目", "前走確定着順", "前距離", "枠番", "前走馬体重増減"):
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m = m.rename(columns={"クラス名": "klass", "枠番": "waku", "ブリンカー": "blinker", "前走馬体重増減": "wdiff"})
    m["天気"] = m["天気"].astype(str).str.strip()
    return m


# ------------------------------------------------------------------ セグメント定義
def odds_band(o):
    return pd.cut(o, [0, 1.4, 1.9, 2.9, 4.9, 9.9, 19.9, 49.9, 99.9, 1e9],
                  labels=["1.0-1.4", "1.5-1.9", "2.0-2.9", "3.0-4.9", "5.0-9.9", "10-19", "20-49", "50-99", "100+"]).astype(str)


def norm_class(k: str) -> str:
    k = str(k)
    return {"500万": "1勝", "1000万": "2勝", "1600万": "3勝", "ｵｰﾌﾟﾝ": "OP", "OP(L)": "OP", "重賞": "OP",
            "Ｇ３": "G3", "Ｇ２": "G2", "Ｇ１": "G1"}.get(k, k)


def build_segments(df: pd.DataFrame) -> dict[str, pd.Series]:
    s: dict[str, pd.Series] = {}
    s["オッズ帯"] = odds_band(df.odds)
    s["人気順"] = pd.cut(df["pop"], [0, 1, 2, 3, 5, 8, 99], labels=["1", "2", "3", "4-5", "6-8", "9+"]).astype(str)
    s["頭数"] = pd.cut(df.field, [0, 8, 12, 15, 18], labels=["≤8", "9-12", "13-15", "16-18"]).astype(str)
    s["クラス"] = df.klass.map(norm_class)
    dband = pd.cut(df.dist, [0, 1400, 1800, 2200, 4000], labels=["短", "マ", "中", "長"]).astype(str)
    s["芝ダ×距離"] = df.surface + dband
    s["競馬場"] = df.place.astype(str).str.strip()
    s["月"] = df.date.str[4:6]
    s["レース番号"] = pd.cut(df["Ｒ"], [0, 4, 8, 10, 11, 12], labels=["1-4R", "5-8R", "9-10R", "11R", "12R(最終)"]).astype(str)
    s["G1開催日"] = np.where(df.g1day, "G1の日", "通常日")
    s["性別"] = df.sex.astype(str).str.strip()
    s["牝馬限定"] = np.where(df["性別限定"].astype(str).str.contains("牝"), "牝限定", "混合")
    s["年齢"] = pd.cut(df.age, [0, 2, 3, 4, 5, 20], labels=["2歳", "3歳", "4歳", "5歳", "6歳+"]).astype(str)
    s["斤量"] = pd.cut(df.kin, [0, 53, 55, 57, 70], labels=["≤53", "54-55", "56-57", "58+"]).astype(str)
    s["騎手(上位15)"] = np.where(df.jockey.isin(df.attrs["top_jockeys"]), df.jockey, "その他")
    s["調教師(上位15)"] = np.where(df.trainer.isin(df.attrs["top_trainers"]), df.trainer, "その他")
    s["間隔"] = pd.cut(df["間隔"], [0, 2, 4, 9, 19, 999], labels=["1-2週", "3-4週", "5-9週", "10-19週", "20週+"]).astype(str)
    s["休み明け～戦目"] = pd.cut(df["休み明け～戦目"], [0, 1, 2, 3, 99], labels=["1(休み明け)", "2", "3", "4+"]).astype(str)
    s["前走着順"] = pd.cut(df["前走確定着順"], [0, 1, 3, 5, 9, 99], labels=["1着", "2-3着", "4-5着", "6-9着", "10着+"]).astype(str)
    dd = df.dist - df["前距離"]
    s["距離変更"] = np.where(dd <= -200, "短縮", np.where(dd >= 200, "延長", "同距離"))
    s["芝ダ替わり"] = np.where(df["前芝・ダ"].notna() & (df["前芝・ダ"] != df.surface), "替わり", "同じ")
    s["枠"] = pd.cut(df.waku, [0, 2, 4, 6, 8], labels=["1-2枠", "3-4枠", "5-6枠", "7-8枠"]).astype(str)
    s["ブリンカー"] = np.where(df.blinker.astype(str).str.strip().isin(["B", "1"]), "B", "なし")
    s["前走馬体重増減"] = pd.cut(df.wdiff, [-99, -9, -3, 2, 8, 99], labels=["-9kg以下", "-8〜-3", "-2〜+2", "+3〜+8", "+9kg以上"]).astype(str)
    s["馬場"] = df["馬場状態"].astype(str).str.strip()
    s["天気"] = df["天気"].astype(str).str.replace(" ", "")
    for k in s:
        s[k] = pd.Series(np.asarray(s[k]).astype(str), index=df.index).replace({"nan": "不明", "None": "不明"})
    return s


# ------------------------------------------------------------------ 集計
def expected_by_band(df: pd.DataFrame, period: pd.Series, col_ret: str) -> pd.Series:
    """オッズ帯 (20 分位) ごとの回収率 → 各馬の期待払戻 (100 円あたり)。本命-大穴バイアスの成分。"""
    sub = df[period]
    q = pd.qcut(sub.odds.rank(method="first"), 20, labels=False)
    roi = sub.groupby(q)[col_ret].mean()          # 100 円あたり払戻
    band_all = pd.qcut(df.odds.rank(method="first"), 20, labels=False)
    # 期の分位境界で全体を切ると期間外にずれるので、期ごとに求めた境界を使う
    edges = np.quantile(sub.odds, np.linspace(0, 1, 21))
    edges[0], edges[-1] = 0, 1e9
    b = pd.cut(df.odds, edges, labels=False, include_lowest=True)
    return pd.Series(b, index=df.index).map(roi).astype(float)


def boot_ci(stake_d: np.ndarray, ret_d: np.ndarray, exp_d: np.ndarray, n_boot: int, rng) -> dict:
    """開催日ごとの (stake, return, expected) ベクトルから ROI と残差の bootstrap CI / p 値。"""
    n = len(stake_d)
    if n < 2 or stake_d.sum() == 0:
        return {"roi_lo": np.nan, "roi_hi": np.nan, "res_lo": np.nan, "res_hi": np.nan, "p_res": np.nan}
    idx = rng.integers(0, n, size=(n_boot, n))
    st = stake_d[idx].sum(1); rt = ret_d[idx].sum(1); ex = exp_d[idx].sum(1)
    roi = 100 * rt / st; res = 100 * (rt - ex) / st
    p = 2 * min((res <= 0).mean(), (res >= 0).mean())
    return {"roi_lo": float(np.percentile(roi, 2.5)), "roi_hi": float(np.percentile(roi, 97.5)),
            "res_lo": float(np.percentile(res, 2.5)), "res_hi": float(np.percentile(res, 97.5)), "p_res": float(min(1.0, p))}


def bh(pvals: list[float], q: float = 0.10) -> list[bool]:
    p = np.asarray(pvals, float); n = len(p); order = np.argsort(p); ranked = p[order] * n / (np.arange(n) + 1)
    thr = np.zeros(n, bool)
    ok = ranked <= q
    if ok.any():
        kmax = np.max(np.where(ok)[0]); thr[order[:kmax + 1]] = True
    return thr.tolist()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    rng = np.random.default_rng(0)

    log("[1/5] 全頭オッズ (外部, 読み取りのみ) ...")
    df = load_ext()
    log(f"   {len(df):,} 頭 / {df.rid16.nunique():,} R / {df.year.min()}-{df.year.max()} (異常コード 0・芝ダのみ)")
    log("[2/5] 複勝配当 ...")
    fp = load_place_pay()
    df = df.merge(fp, on="rid18", how="left")
    df["win_ret"] = np.where(df.fin == 1, 100 * df.odds, 0.0)
    df["plc_ret"] = np.where(df.fin <= 3, df.fpay.fillna(0.0), 0.0)
    log("[3/5] 属性 (master_v2) ...")
    m = load_master()
    df = df.merge(m, on=["rid16", "ban"], how="inner")      # master に無い行 (障害・地方等) は落とす
    df = df[~df.klass.astype(str).str.startswith("ＪＧ")]
    log(f"   master 結合後: {len(df):,} 頭 / {df.rid16.nunique():,} R / 間隔 {df['間隔'].notna().mean()*100:.1f}%")
    g1_dates = set(df.loc[df.klass.map(norm_class) == "G1", "date"].unique())
    df["g1day"] = df.date.isin(g1_dates)
    disc = (df.year >= DISC[0]) & (df.year <= DISC[1])
    conf = (df.year >= CONF[0]) & (df.year <= CONF[1])
    topj = df[disc].jockey.value_counts().head(15).index.tolist()
    topt = df[disc].trainer.value_counts().head(15).index.tolist()
    df.attrs["top_jockeys"], df.attrs["top_trainers"] = topj, topt

    log("[4/5] 期待払戻 (オッズ帯 20 分位) と集計 ...")
    df["exp_win_d"] = expected_by_band(df, disc, "win_ret"); df["exp_win_c"] = expected_by_band(df, conf, "win_ret")
    df["exp_plc_d"] = expected_by_band(df, disc, "plc_ret"); df["exp_plc_c"] = expected_by_band(df, conf, "plc_ret")
    segs = build_segments(df)
    base = {}
    for nm, mask in (("発見期", disc), ("確認期", conf)):
        sub = df[mask]
        base[nm] = {"n": int(len(sub)), "races": int(sub.rid16.nunique()), "win_roi": float(100 * sub.win_ret.sum() / (100 * len(sub))),
                    "plc_roi": float(100 * sub.plc_ret.sum() / (100 * len(sub)))}
    log(f"   全体: 発見期 単勝 {base['発見期']['win_roi']:.1f}% 複勝 {base['発見期']['plc_roi']:.1f}% / 確認期 単勝 {base['確認期']['win_roi']:.1f}% 複勝 {base['確認期']['plc_roi']:.1f}%")

    rows = []
    for dim, lab in segs.items():
        for cat in sorted(lab.unique()):
            if cat == "不明":
                continue
            r = {"dim": dim, "cat": cat}
            ok = True
            for pname, mask, ew, ep in (("d", disc, "exp_win_d", "exp_plc_d"), ("c", conf, "exp_win_c", "exp_plc_c")):
                sel = mask & (lab == cat)
                sub = df[sel]
                n = len(sub)
                r[f"n_{pname}"] = int(n)
                if n < MIN_N:
                    ok = False
                    continue
                for bt, rc, ec in (("win", "win_ret", ew), ("plc", "plc_ret", ep)):
                    g = sub.groupby("date").agg(st=(rc, "size"), rt=(rc, "sum"), ex=(ec, "sum"))
                    st = 100 * g.st.values.astype(float); rt = g.rt.values; ex = g.ex.values
                    r[f"{bt}_roi_{pname}"] = float(100 * rt.sum() / st.sum())
                    r[f"{bt}_res_{pname}"] = float(100 * (rt.sum() - ex.sum()) / st.sum())
                    ci = boot_ci(st, rt, ex, a.boot if pname == "c" else max(300, a.boot // 4), rng)
                    r[f"{bt}_roi_lo_{pname}"], r[f"{bt}_roi_hi_{pname}"] = ci["roi_lo"], ci["roi_hi"]
                    r[f"{bt}_res_lo_{pname}"], r[f"{bt}_res_hi_{pname}"] = ci["res_lo"], ci["res_hi"]
                    r[f"{bt}_p_{pname}"] = ci["p_res"]
            if ok:
                rows.append(r)
        log(f"   {dim}: {sum(1 for x in rows if x['dim'] == dim)} セグメント")

    for bt in ("win", "plc"):
        ps = [x.get(f"{bt}_p_c", 1.0) for x in rows]
        flags = bh(ps, 0.10)
        for x, f in zip(rows, flags):
            same_sign = np.sign(x.get(f"{bt}_res_d", 0)) == np.sign(x.get(f"{bt}_res_c", 0))
            x[f"{bt}_flag"] = bool(f and same_sign and abs(x.get(f"{bt}_res_c", 0)) >= 2.0 and abs(x.get(f"{bt}_res_d", 0)) >= 1.0)

    log("[5/5] 出力 ...")
    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_bytes(json.dumps({"base": base, "disc": DISC, "conf": CONF, "min_n": MIN_N, "boot": a.boot,
                                     "top_jockeys": topj, "top_trainers": topt, "rows": rows}, ensure_ascii=False, indent=1).encode("utf-8"))
    md = [f"# 市場の癖の地図 (market_bias_atlas, {pd.Timestamp.now():%Y-%m-%d})", "",
          "モデル不要。全頭の確定単勝オッズ × 払戻だけ。各セグメントで 100 円均等買いの回収率 (単勝 / 複勝)。",
          f"発見期 {DISC[0]}-{DISC[1]} / 確認期 {CONF[0]}-{CONF[1]}。残差 = 回収率 − オッズ帯構成から期待される回収率 (本命-大穴バイアスを除いた、その属性固有の癖)。",
          "CI は開催日クラスタ bootstrap 95%。★ = 確認期の残差が BH-FDR q<0.10 で有意、発見期と同符号、|残差| ≥ 2pt (確認期) かつ ≥ 1pt (発見期)。", "",
          "| 期 | 頭数 | R | 単勝 均等 | 複勝 均等 |", "|---|---|---|---|---|"]
    for nm, b in base.items():
        md.append(f"| {nm} | {b['n']:,} | {b['races']:,} | {b['win_roi']:.1f}% | {b['plc_roi']:.1f}% |")
    stars = [x for x in rows if x.get("win_flag") or x.get("plc_flag")]
    md += ["", f"## ★ 両期で再現した癖 ({len(stars)} 件)", ""]
    if stars:
        md += ["| 次元 | セグメント | n(確認) | 単勝 ROI 確認 [CI] | 単勝 残差 発見/確認 | 複勝 ROI 確認 [CI] | 複勝 残差 発見/確認 | 有意 |", "|---|---|---|---|---|---|---|---|"]
        for x in sorted(stars, key=lambda x: -max(abs(x.get("win_res_c", 0)) * x.get("win_flag", 0), abs(x.get("plc_res_c", 0)) * x.get("plc_flag", 0))):
            md.append(f"| {x['dim']} | {x['cat']} | {x['n_c']:,} | {x['win_roi_c']:.1f} [{x['win_roi_lo_c']:.1f},{x['win_roi_hi_c']:.1f}] | "
                      f"{x['win_res_d']:+.1f} / {x['win_res_c']:+.1f} | {x['plc_roi_c']:.1f} [{x['plc_roi_lo_c']:.1f},{x['plc_roi_hi_c']:.1f}] | "
                      f"{x['plc_res_d']:+.1f} / {x['plc_res_c']:+.1f} | {'単' if x.get('win_flag') else ''}{'複' if x.get('plc_flag') else ''} |")
    else:
        md.append("（該当なし）")
    best = sorted(rows, key=lambda x: -x.get("win_roi_c", 0))[:5]
    md += ["", "## 確認期で単勝回収率が最も高かった 5 セグメント (参考、100% 超えの有無)", "",
           "| 次元 | セグメント | n | 単勝 ROI [CI] | 複勝 ROI [CI] |", "|---|---|---|---|---|"]
    for x in best:
        md.append(f"| {x['dim']} | {x['cat']} | {x['n_c']:,} | {x['win_roi_c']:.1f} [{x['win_roi_lo_c']:.1f},{x['win_roi_hi_c']:.1f}] | {x['plc_roi_c']:.1f} [{x['plc_roi_lo_c']:.1f},{x['plc_roi_hi_c']:.1f}] |")
    md += ["", "## 全セグメント", ""]
    cur = None
    for x in rows:
        if x["dim"] != cur:
            cur = x["dim"]
            md += ["", f"### {cur}", "", "| セグメント | n 発見/確認 | 単勝 ROI 発見 → 確認 [CI] | 単勝 残差 発見 / 確認 | 複勝 ROI 発見 → 確認 [CI] | 複勝 残差 発見 / 確認 | q<.1 |", "|---|---|---|---|---|---|---|"]
        md.append(f"| {x['cat']} | {x['n_d']:,} / {x['n_c']:,} | {x['win_roi_d']:.1f} → {x['win_roi_c']:.1f} [{x['win_roi_lo_c']:.1f},{x['win_roi_hi_c']:.1f}] | "
                  f"{x['win_res_d']:+.1f} / {x['win_res_c']:+.1f} | {x['plc_roi_d']:.1f} → {x['plc_roi_c']:.1f} [{x['plc_roi_lo_c']:.1f},{x['plc_roi_hi_c']:.1f}] | "
                  f"{x['plc_res_d']:+.1f} / {x['plc_res_c']:+.1f} | {'★単' if x.get('win_flag') else ''}{'★複' if x.get('plc_flag') else ''} |")
    OUT_MD.write_bytes("\n".join(md).encode("utf-8"))
    log(f"→ {OUT_MD} ({len(rows)} セグメント, ★ {len(stars)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
