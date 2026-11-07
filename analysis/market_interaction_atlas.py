# -*- coding: utf-8 -*-
"""
market_interaction_atlas.py — 市場の癖 × 条件の 2 次元地図と、AI の得意・不得意の条件地図 (2026-11-07)
================================================================================
ユーザー「12R × 本命 のような交互作用を全探索の中から見つけてほしい」。

Stage 1 (市場だけ、2013-2025):  オッズ帯 (6) × 属性 (25 次元) の 2 次元セルで 100 円均等買いの回収率。
  残差 = セル回収率 − その期のオッズ帯基準。発見期 2013-20 / 確認期 2021-25 で同符号に再現するものを、
  確認期の開催日 bootstrap + BH-FDR で選ぶ。例: 「12R × 〜1.9 倍」は +10/+15pt。
Stage 2 (AI、2023-2026): 条件ごとに「AI 1 位 (生スコア最大、同率なし) の 単勝/複勝 均等買い」と
  「同じレースの市場 1 番人気」を比べる。差 = AI − 市場 (pt)。発見期 2023-24 (offline, 本番 v6 の生スコア
  data/_ev_grid_scores.parquet)、確認期 2025 (offline) + 2026 (実配信 bundle)。再現 + bootstrap + BH。
  AI の弱点条件 (差が大きく負) と、AI が市場に勝つ条件 (正) を列挙する。先読み漏れ: 生スコアは連続値で同率なし、
  2026 は本番と同じ bundle 順 (馬番) のタイブレーク。
出力: reports/market_interaction_atlas.md / .json
実行: python -m analysis.market_interaction_atlas [--boot 800]
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from analysis.market_bias_atlas import load_ext, load_place_pay, load_master, norm_class, bh  # noqa: E402

OUT_MD = BASE / "reports" / "market_interaction_atlas.md"
OUT_JSON = BASE / "reports" / "market_interaction_atlas.json"
SCORES = BASE / "data" / "_ev_grid_scores.parquet"
BANDS = [0, 1.9, 4.9, 9.9, 19.9, 49.9, 1e9]
BAND_LABELS = ["〜1.9", "2-4.9", "5-9.9", "10-19", "20-49", "50+"]
MIN_N1 = 300      # Stage 1 セル最小頭数 (期ごと)
MIN_R2 = 60       # Stage 2 セル最小レース数 (期ごと)


def log(m):
    print(m, flush=True)


def seg_frame(df: pd.DataFrame, top_j: list[str], top_t: list[str]) -> dict[str, pd.Series]:
    """market_bias_atlas.build_segments 相当 (attrs 依存を外した版)。"""
    s: dict[str, pd.Series] = {}
    s["頭数"] = pd.cut(df.field, [0, 8, 12, 15, 18], labels=["≤8", "9-12", "13-15", "16-18"]).astype(str)
    s["クラス"] = df.klass.map(norm_class)
    dband = pd.cut(df.dist, [0, 1400, 1800, 2200, 4000], labels=["短", "マ", "中", "長"]).astype(str)
    s["芝ダ×距離"] = df.surface.astype(str) + dband
    s["競馬場"] = df.place.astype(str).str.strip()
    s["月"] = df.date.str[4:6]
    s["レース番号"] = pd.cut(df["Ｒ"], [0, 4, 8, 10, 11, 12], labels=["1-4R", "5-8R", "9-10R", "11R", "12R"]).astype(str)
    s["G1開催日"] = np.where(df.g1day, "G1の日", "通常日")
    s["性別"] = df.sex.astype(str).str.strip()
    s["牝馬限定"] = np.where(df["性別限定"].astype(str).str.contains("牝"), "牝限定", "混合")
    s["年齢"] = pd.cut(df.age, [0, 2, 3, 4, 5, 20], labels=["2歳", "3歳", "4歳", "5歳", "6歳+"]).astype(str)
    s["斤量"] = pd.cut(df.kin, [0, 53, 55, 57, 70], labels=["≤53", "54-55", "56-57", "58+"]).astype(str)
    s["騎手(上位15)"] = np.where(df.jockey.isin(top_j), df.jockey, "その他")
    s["調教師(上位15)"] = np.where(df.trainer.isin(top_t), df.trainer, "その他")
    s["間隔"] = pd.cut(df["間隔"], [0, 2, 4, 9, 19, 999], labels=["1-2週", "3-4週", "5-9週", "10-19週", "20週+"]).astype(str)
    s["休み明け～戦目"] = pd.cut(df["休み明け～戦目"], [0, 1, 2, 3, 99], labels=["1(休み明け)", "2", "3", "4+"]).astype(str)
    s["前走着順"] = pd.cut(df["前走確定着順"], [0, 1, 3, 5, 9, 99], labels=["1着", "2-3着", "4-5着", "6-9着", "10着+"]).astype(str)
    dd = df.dist - df["前距離"]
    s["距離変更"] = np.where(dd <= -200, "短縮", np.where(dd >= 200, "延長", "同距離"))
    s["芝ダ替わり"] = np.where(df["前芝・ダ"].notna() & (df["前芝・ダ"] != df.surface), "替わり", "同じ")
    s["枠"] = pd.cut(df.waku, [0, 2, 4, 6, 8], labels=["1-2枠", "3-4枠", "5-6枠", "7-8枠"]).astype(str)
    s["馬場"] = df["馬場状態"].astype(str).str.strip()
    s["天気"] = df["天気"].astype(str).str.replace(" ", "")
    for k in s:
        s[k] = pd.Series(np.asarray(s[k]).astype(str), index=df.index).replace({"nan": "不明", "None": "不明"})
    return s


def boot_diff(g: pd.DataFrame, n_boot: int, rng, a: str, b: str, stake: str = "n") -> tuple[float, float, float]:
    """開催日集計 g (列 n, a, b) から (a−b)/stake ×100 の bootstrap CI と両側 p。"""
    if len(g) < 2:
        return np.nan, np.nan, np.nan
    idx = rng.integers(0, len(g), size=(n_boot, len(g)))
    st = g[stake].values[idx].sum(1).astype(float)
    d = (g[a].values[idx].sum(1) - g[b].values[idx].sum(1)) / st      # 払戻は 100 円あたりの円 = pt なのでそのまま
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float(min(1, p))


# ------------------------------------------------------------------ Stage 1
def stage1(df: pd.DataFrame, segs: dict, n_boot: int, rng) -> list[dict]:
    disc = (df.year <= 2020); conf = (df.year >= 2021)
    df = df.assign(band=pd.cut(df.odds, BANDS, labels=BAND_LABELS).astype(str))
    rows = []
    for pname, mask in (("d", disc), ("c", conf)):
        sub = df[mask]
        base_w = sub.groupby("band").win_ret.mean(); base_p = sub.groupby("band").plc_ret.mean()
        df.loc[mask, "exp_w"] = sub.band.map(base_w).values; df.loc[mask, "exp_p"] = sub.band.map(base_p).values
    for dim, lab in segs.items():
        for cat in sorted(lab.unique()):
            if cat == "不明":
                continue
            for band in BAND_LABELS:
                r = {"dim": dim, "cat": cat, "band": band}
                ok = True
                for pname, mask in (("d", disc), ("c", conf)):
                    sel = mask & (lab == cat) & (df.band == band)
                    sub = df[sel]; n = len(sub); r[f"n_{pname}"] = int(n)
                    if n < MIN_N1:
                        ok = False; continue
                    for bt, rc, ec in (("win", "win_ret", "exp_w"), ("plc", "plc_ret", "exp_p")):
                        r[f"{bt}_roi_{pname}"] = float(sub[rc].mean())
                        r[f"{bt}_res_{pname}"] = float((sub[rc] - sub[ec]).mean())
                if ok:
                    rows.append(r)
    # pre-screen → bootstrap on confirm
    for r in rows:
        for bt in ("win", "plc"):
            same = np.sign(r[f"{bt}_res_d"]) == np.sign(r[f"{bt}_res_c"])
            if same and abs(r[f"{bt}_res_c"]) >= 3 and abs(r[f"{bt}_res_d"]) >= 1.5:
                sel = conf & (segs[r["dim"]] == r["cat"]) & (df.band == r["band"])
                rc, ec = ("win_ret", "exp_w") if bt == "win" else ("plc_ret", "exp_p")
                g = df[sel].groupby("date").agg(n=(rc, "size"), a=(rc, "sum"), b=(ec, "sum"))
                lo, hi, p = boot_diff(g, n_boot, rng, "a", "b")
                r[f"{bt}_lo"], r[f"{bt}_hi"], r[f"{bt}_p"] = lo, hi, p
            else:
                r[f"{bt}_p"] = 1.0
    for bt in ("win", "plc"):
        flags = bh([r[f"{bt}_p"] for r in rows], 0.10)
        for r, f in zip(rows, flags):
            r[f"{bt}_flag"] = bool(f and r[f"{bt}_p"] < 1.0)
    return rows


# ------------------------------------------------------------------ Stage 2
def load_served_2026() -> pd.DataFrame:
    rows = []
    for bp in sorted(glob.glob(str(BASE / "reports" / "cowork_input" / "2026????_bundle.json"))):
        d = Path(bp).name[:8]
        kp = BASE / "data" / "kekka" / f"{d}.csv"
        if not kp.exists():
            continue
        k = pd.read_csv(kp, encoding="cp932", low_memory=False)
        k["rid"] = k["レースID(新)"].astype(str).str[:16]; k["ban"] = pd.to_numeric(k["馬番"], errors="coerce")
        k["fin"] = pd.to_numeric(k["確定着順"], errors="coerce"); k["fpay"] = pd.to_numeric(k["複勝配当"], errors="coerce")
        t = k["単勝配当"].astype(str)
        k["odds"] = np.where(t.str.match(r"^\d+$"), pd.to_numeric(t, errors="coerce") / 100, pd.to_numeric(t.str.strip("()"), errors="coerce"))
        kd = {(x.rid, int(x.ban)): (x.fin, x.fpay if np.isfinite(x.fpay) else 0.0, x.odds) for x in k.itertuples() if pd.notna(x.ban)}
        b = json.loads(Path(bp).read_text(encoding="utf-8"))
        rs = b["races"] if isinstance(b["races"], list) else list(b["races"].values())
        for r in rs:
            rid = re.sub(r"\D", "", str(r["race_id"]))[:16]
            hs = [h for h in r["horses"] if h.get("p_win") is not None]
            if len(hs) < 5:
                continue
            meta = r.get("race_meta") or {}
            for h in hs:
                res = kd.get((rid, int(h["umaban"])))
                if not res or not np.isfinite(res[0]) or res[0] < 1:
                    continue
                runs = (h.get("history") or {}).get("runs") or []
                rows.append(dict(rid16=rid, date=d, year=2026, ban=int(h["umaban"]), score=float(h["p_win"]), fin=res[0], fpay=res[1], odds=res[2],
                                 field=len(hs), klass=str(meta.get("class", "")), place=str(meta.get("place", "")), course=str(meta.get("course", "")),
                                 R=int(rid[14:16]), sex=str(h.get("sex", "")), age=h.get("age"), prev_pos=(runs[0].get("pos") if runs else None)))
    df = pd.DataFrame(rows)
    df["surface"] = df.course.str[:1]
    df["dist"] = pd.to_numeric(df.course.str.extract(r"(\d{3,4})")[0], errors="coerce")
    df["age"] = pd.to_numeric(df.age, errors="coerce")
    return df


def race_level(df: pd.DataFrame) -> pd.DataFrame:
    """馬行 → レース行: AI1位 (score 最大、同率は馬番昇順) と 市場1番人気 (odds 最小、同率は馬番昇順) の成績。"""
    df = df.sort_values(["rid16", "ban"])
    df["win_ret"] = np.where(df.fin == 1, 100 * df.odds, 0.0); df["plc_ret"] = np.where(df.fin <= 3, df.fpay.fillna(0), 0.0)
    ai = df.loc[df.groupby("rid16").score.idxmax()]
    fav = df.loc[df.groupby("rid16").odds.idxmin()]
    out = ai[["rid16", "date", "year", "field", "klass", "place", "surface", "dist", "R", "sex", "age", "prev_pos", "odds", "win_ret", "plc_ret", "fin"]].copy()
    out = out.rename(columns={"odds": "ai_odds", "win_ret": "ai_w", "plc_ret": "ai_p", "fin": "ai_fin"})
    f = fav.set_index("rid16")
    out["fav_w"] = out.rid16.map(f.win_ret); out["fav_p"] = out.rid16.map(f.plc_ret); out["fav_fin"] = out.rid16.map(f.fin)
    out["agree"] = out.rid16.map(f.ban).values == ai.ban.values
    for c in ("間隔", "休み明け～戦目"):
        if c in ai.columns:
            out[c] = ai[c].values
    return out


def stage2(off: pd.DataFrame, srv: pd.DataFrame, n_boot: int, rng) -> list[dict]:
    rl = pd.concat([race_level(off), race_level(srv)], ignore_index=True)
    rl["period"] = np.where(rl.year <= 2024, "d", "c")
    s: dict[str, pd.Series] = {}
    s["レース番号"] = pd.cut(rl.R, [0, 4, 8, 10, 11, 12], labels=["1-4R", "5-8R", "9-10R", "11R", "12R"]).astype(str)
    s["頭数"] = pd.cut(rl.field, [0, 8, 12, 15, 18], labels=["≤8", "9-12", "13-15", "16-18"]).astype(str)
    s["クラス"] = rl.klass.map(norm_class)
    s["芝ダ×距離"] = rl.surface.astype(str) + pd.cut(rl.dist, [0, 1400, 1800, 2200, 4000], labels=["短", "マ", "中", "長"]).astype(str)
    s["競馬場"] = rl.place.astype(str).str.strip()
    s["月"] = rl.date.str[4:6]
    s["AI1位のオッズ帯"] = pd.cut(rl.ai_odds, BANDS, labels=BAND_LABELS).astype(str)
    s["AI1位=1番人気"] = np.where(rl.agree, "一致", "不一致")
    s["AI1位の前走着順"] = pd.cut(pd.to_numeric(rl.prev_pos, errors="coerce"), [0, 1, 3, 5, 9, 99], labels=["1着", "2-3着", "4-5着", "6-9着", "10着+"]).astype(str)
    s["AI1位の性別"] = rl.sex.astype(str).str.strip()
    s["AI1位の年齢"] = pd.cut(rl.age, [0, 2, 3, 4, 5, 20], labels=["2歳", "3歳", "4歳", "5歳", "6歳+"]).astype(str)
    if "間隔" in rl:
        s["AI1位の間隔"] = pd.cut(pd.to_numeric(rl["間隔"], errors="coerce"), [0, 2, 4, 9, 19, 999], labels=["1-2週", "3-4週", "5-9週", "10-19週", "20週+"]).astype(str)
    # 2 次元: レース番号 × AI1位オッズ帯 / 頭数 × 一致 / クラス × 一致
    s["レース番号×AI1位オッズ帯"] = s["レース番号"] + "×" + s["AI1位のオッズ帯"]
    s["頭数×一致"] = s["頭数"] + "×" + s["AI1位=1番人気"]
    s["クラス×一致"] = s["クラス"] + "×" + s["AI1位=1番人気"]
    s["レース番号×一致"] = s["レース番号"] + "×" + s["AI1位=1番人気"]
    rows = []
    for dim, lab in s.items():
        lab = pd.Series(np.asarray(lab).astype(str), index=rl.index).replace({"nan": "不明"})
        for cat in sorted(lab.unique()):
            if "不明" in cat:
                continue
            r = {"dim": dim, "cat": cat}; ok = True
            for pname in ("d", "c"):
                sub = rl[(lab == cat) & (rl.period == pname)]
                r[f"n_{pname}"] = int(len(sub))
                if len(sub) < MIN_R2:
                    ok = False; continue
                r[f"ai_win_{pname}"] = float(100 * (sub.ai_fin == 1).mean()); r[f"fav_win_{pname}"] = float(100 * (sub.fav_fin == 1).mean())
                r[f"ai_top3_{pname}"] = float(100 * (sub.ai_fin <= 3).mean()); r[f"fav_top3_{pname}"] = float(100 * (sub.fav_fin <= 3).mean())
                r[f"ai_tan_{pname}"] = float(sub.ai_w.mean()); r[f"fav_tan_{pname}"] = float(sub.fav_w.mean())
                r[f"ai_fuku_{pname}"] = float(sub.ai_p.mean()); r[f"fav_fuku_{pname}"] = float(sub.fav_p.mean())
                r[f"dtan_{pname}"] = r[f"ai_tan_{pname}"] - r[f"fav_tan_{pname}"]; r[f"dfuku_{pname}"] = r[f"ai_fuku_{pname}"] - r[f"fav_fuku_{pname}"]
            if ok:
                rows.append(r)
    for r in rows:
        for bt, a, b in (("tan", "ai_w", "fav_w"), ("fuku", "ai_p", "fav_p")):
            same = np.sign(r[f"d{bt}_d"]) == np.sign(r[f"d{bt}_c"])
            if same and abs(r[f"d{bt}_c"]) >= 4 and abs(r[f"d{bt}_d"]) >= 2:
                lab = s[r["dim"]]
                sel = (pd.Series(np.asarray(lab).astype(str), index=rl.index) == r["cat"]) & (rl.period == "c")
                g = rl[sel].groupby("date").agg(n=(a, "size"), a=(a, "sum"), b=(b, "sum"))
                lo, hi, p = boot_diff(g, n_boot, rng, "a", "b")
                r[f"{bt}_lo"], r[f"{bt}_hi"], r[f"{bt}_p"] = lo, hi, p
            else:
                r[f"{bt}_p"] = 1.0
    for bt in ("tan", "fuku"):
        flags = bh([r[f"{bt}_p"] for r in rows], 0.10)
        for r, f in zip(rows, flags):
            r[f"{bt}_flag"] = bool(f and r[f"{bt}_p"] < 1.0)
    return rows


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--boot", type=int, default=800); a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    rng = np.random.default_rng(0)
    log("[1/4] 市場データ 2013-2025 ...")
    df = load_ext().merge(load_place_pay(), on="rid18", how="left").merge(load_master(), on=["rid16", "ban"], how="inner")
    df = df[~df.klass.astype(str).str.startswith("ＪＧ")].copy()
    df["win_ret"] = np.where(df.fin == 1, 100 * df.odds, 0.0); df["plc_ret"] = np.where(df.fin <= 3, df.fpay.fillna(0), 0.0)
    g1 = set(df.loc[df.klass.map(norm_class) == "G1", "date"]); df["g1day"] = df.date.isin(g1)
    topj = df[df.year <= 2020].jockey.value_counts().head(15).index.tolist(); topt = df[df.year <= 2020].trainer.value_counts().head(15).index.tolist()
    segs = seg_frame(df, topj, topt)
    log(f"   {len(df):,} 頭 / {df.rid16.nunique():,} R")
    log("[2/4] Stage 1: オッズ帯 × 属性 ...")
    s1 = stage1(df, segs, a.boot, rng)
    f1 = [r for r in s1 if r.get("win_flag") or r.get("plc_flag")]
    log(f"   {len(s1)} セル → 再現 {len(f1)}")
    log("[3/4] Stage 2: AI 1 位 vs 市場 1 番人気 (2023-26) ...")
    sc = pd.read_parquet(SCORES)
    sc["rid16"] = sc["レースID(新/馬番無)"].astype(str).str[:16]; sc["ban"] = pd.to_numeric(sc["馬番"], errors="coerce")
    off = df[df.year >= 2023].merge(sc[["rid16", "ban", "_score"]], on=["rid16", "ban"], how="inner").rename(columns={"_score": "score"})
    off["prev_pos"] = off["前走確定着順"]; off["R"] = off["Ｒ"]
    srv = load_served_2026()
    log(f"   offline {off.rid16.nunique():,} R (2023-25) / served {srv.rid16.nunique():,} R (2026)")
    s2 = stage2(off, srv, a.boot, rng)
    f2 = [r for r in s2 if r.get("tan_flag") or r.get("fuku_flag")]
    log(f"   {len(s2)} セル → 再現 {len(f2)}")
    log("[4/4] 出力 ...")
    OUT_JSON.write_bytes(json.dumps({"stage1": s1, "stage2": s2, "boot": a.boot}, ensure_ascii=False, indent=1).encode("utf-8"))
    md = [f"# 市場の癖 × 条件 / AI の得意・不得意 地図 ({pd.Timestamp.now():%Y-%m-%d})", "",
          "## Stage 1: オッズ帯 × 属性 (市場だけ、2013-20 → 2021-25 で同符号に再現、BH q<0.10)", "",
          "残差 = セルの均等買い回収率 − その期のオッズ帯基準 (pt)。正 = 市場が買い足りない、負 = 買いすぎ。", "",
          "| 属性 | セグメント | オッズ帯 | n 発見/確認 | 単勝 残差 発見/確認 [CI] | 複勝 残差 発見/確認 [CI] | 単勝 ROI 確認 | 複勝 ROI 確認 | 有意 |", "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(f1, key=lambda x: -max(abs(x["win_res_c"]) * x.get("win_flag", 0), abs(x["plc_res_c"]) * x.get("plc_flag", 0))):
        wci = f"[{r['win_lo']:.1f},{r['win_hi']:.1f}]" if r.get("win_flag") else ""; pci = f"[{r['plc_lo']:.1f},{r['plc_hi']:.1f}]" if r.get("plc_flag") else ""
        md.append(f"| {r['dim']} | {r['cat']} | {r['band']} | {r['n_d']:,}/{r['n_c']:,} | {r['win_res_d']:+.1f}/{r['win_res_c']:+.1f} {wci} | {r['plc_res_d']:+.1f}/{r['plc_res_c']:+.1f} {pci} | {r['win_roi_c']:.1f} | {r['plc_roi_c']:.1f} | {'単' if r.get('win_flag') else ''}{'複' if r.get('plc_flag') else ''} |")
    md += ["", "## Stage 2: AI 1 位 vs 市場 1 番人気 (発見 2023-24 offline → 確認 2025 offline + 2026 実配信)", "",
           "差 = AI1位の均等買い回収率 − 同じレースの市場1番人気 (pt)。負 = その条件では AI は市場に負ける。", "",
           "| 条件 | セグメント | R 発見/確認 | AI 勝率 確認 (市場) | AI 3着内 確認 (市場) | 単勝 差 発見/確認 [CI] | 複勝 差 発見/確認 [CI] | AI 単勝/複勝 ROI 確認 | 有意 |", "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(f2, key=lambda x: min(x["dtan_c"] if x.get("tan_flag") else 0, x["dfuku_c"] if x.get("fuku_flag") else 0)):
        tci = f"[{r['tan_lo']:.1f},{r['tan_hi']:.1f}]" if r.get("tan_flag") else ""; fci = f"[{r['fuku_lo']:.1f},{r['fuku_hi']:.1f}]" if r.get("fuku_flag") else ""
        md.append(f"| {r['dim']} | {r['cat']} | {r['n_d']:,}/{r['n_c']:,} | {r['ai_win_c']:.1f} ({r['fav_win_c']:.1f}) | {r['ai_top3_c']:.1f} ({r['fav_top3_c']:.1f}) | {r['dtan_d']:+.1f}/{r['dtan_c']:+.1f} {tci} | {r['dfuku_d']:+.1f}/{r['dfuku_c']:+.1f} {fci} | {r['ai_tan_c']:.1f} / {r['ai_fuku_c']:.1f} | {'単' if r.get('tan_flag') else ''}{'複' if r.get('fuku_flag') else ''} |")
    md += ["", "## Stage 2 全セル (参考)", "", "| 条件 | セグメント | R 発見/確認 | AI 勝率 発見→確認 | 市場 勝率 発見→確認 | 単勝 差 発見/確認 | 複勝 差 発見/確認 |", "|---|---|---|---|---|---|---|"]
    for r in s2:
        md.append(f"| {r['dim']} | {r['cat']} | {r['n_d']:,}/{r['n_c']:,} | {r['ai_win_d']:.1f}→{r['ai_win_c']:.1f} | {r['fav_win_d']:.1f}→{r['fav_win_c']:.1f} | {r['dtan_d']:+.1f}/{r['dtan_c']:+.1f} | {r['dfuku_d']:+.1f}/{r['dfuku_c']:+.1f} |")
    OUT_MD.write_bytes("\n".join(md).encode("utf-8"))
    log(f"→ {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
