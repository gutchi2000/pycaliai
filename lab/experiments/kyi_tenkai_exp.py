# -*- coding: utf-8 -*-
"""
kyi_tenkai_exp.py — JRDB の発走前の展開予想 (KYI) は新しい情報か (2026-11-15)
==========================================================================
背景
----
`pace_granularity_ceiling.py` (同日) の結果: 展開で効くのは「今日その馬が前半どこにいたか」だけで、
完全に分かれば確定市場+v7 に M −0.045 nats/R。しかし自前の発走前情報での再現度は R² 0.30 で、予測値に置き換えると M 0。
再開条件は「発走前に前半の位置を決める新しい情報」。JRDB の前日データ KYI には展開予想
(道中・後3F・ゴールの順位/差/内外、展開記号、テン/ペース/上がり/位置指数、ペース予想、脚質) がある。
JRDB が過去走から機械的に作った予想なら v7 と市場の焼き直し、取材などが入っていれば新情報。それを測る。

データ
------
KYI (data/jrdb/raw/Kyi/*.zip、2013〜) のうち次の列だけ (1 始まりのバイト位置、cp932):
  racekey 1-8, 馬番 9-10, 脚質 90, テン指数 359(5), ペース指数 364(5), 上がり指数 369(5), 位置指数 374(5),
  ペース予想 379(1 H/M/S), 道中順位 380(2), 道中差 382(2), 道中内外 384(1), 後3F順位 385(2), 後3F差 387(2),
  後3F内外 389(1), ゴール順位 390(2), ゴール差 392(2), ゴール内外 394(1), 展開記号 395(1)
SED と racekey+馬番で結合。KYI は前日の配布物 (アーカイブ。後から直されていないかは未確認 = CW-12 で前向きに確認中)。

事前に決めたこと
----------------
A. 再現度: 実際の最初のコーナー相対位置 pf を LightGBM 回帰で予測 (2013-20 学習、2021-22 早期終了)、test 2024-25 の R²。
   A0 = 自前の発走前情報 (pace_granularity_ceiling の PREF)、A1 = A0 + KYI 展開、A2 = KYI 展開だけ (+距離/芝ダ/場/頭数)。
B. ゲート (gate_common、valid 2023 で重み、test 2024-25 を 1 回):
   z_pre  = 1 着 LightGBM、PRE だけ                         (11/15 の値の再現)
   z_kyi  = PRE + KYI 展開 (位置予想 + 指数 + 記号)
   z_kpos = PRE + KYI の位置予想だけ (道中/後3F/ゴールの順位・差・内外、展開記号。指数を除く)
   z_phat = P オラクルモデル (pace_granularity_ceiling の P 集合) に A1 の予測値を入れたもの
   偽薬: z_kyi_shuf = KYI 列をレース内で馬に振り直してから同じモデルで予測
判定: M ゲート (確定市場+v7 → +z) が PASS、かつ 2024 年と 2025 年の ΔNLL が両方負、かつ偽薬が PASS しない
      → 「KYI 展開予想は市場の外の情報」。P だけ PASS は「v7 の上乗せだが市場は既に持つ」= 馬券には効かない。
      z は 4 本 (偽薬除く) なので、1 本だけ PASS で年別を通らなければ偶然の範囲と書く。

出力: reports/kyi_tenkai_exp.json (集計値のみ。JRDB の馬別値は出さない)
      E:/PyCaLiAI/data/jrdb/kyi_tenkai.parquet (作業用、data/jrdb は git 管理外)
再実行: worktree root から E:/PyCaLiAI/venv311/Scripts/python.exe lab/experiments/kyi_tenkai_exp.py
"""
from __future__ import annotations

import io
import json
import re
import sys
import time
import zipfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate_common as gc  # noqa: E402
import pace_forward_problem_exp as pfx  # noqa: E402
import pace_granularity_ceiling as pgc  # noqa: E402

T0 = time.time()
RAW = gc.ROOT / "data" / "jrdb" / "raw" / "Kyi"
KYI_PQ = gc.ROOT / "data" / "jrdb" / "kyi_tenkai.parquet"
OUT = gc.REP / "kyi_tenkai_exp.json"
FIELDS = [("racekey", 1, 8, "s"), ("ban", 9, 2, "n"), ("k_style", 90, 1, "n"), ("k_ten", 359, 5, "n"),
          ("k_pidx", 364, 5, "n"), ("k_agari", 369, 5, "n"), ("k_posidx", 374, 5, "n"), ("k_pace", 379, 1, "s"),
          ("k_d_jun", 380, 2, "n"), ("k_d_sa", 382, 2, "n"), ("k_d_io", 384, 1, "n"), ("k_a_jun", 385, 2, "n"),
          ("k_a_sa", 387, 2, "n"), ("k_a_io", 389, 1, "n"), ("k_g_jun", 390, 2, "n"), ("k_g_sa", 392, 2, "n"),
          ("k_g_io", 394, 1, "n"), ("k_tenkai", 395, 1, "n")]


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def parse_kyi() -> pd.DataFrame:
    if KYI_PQ.exists():
        return pd.read_parquet(KYI_PQ)
    rows, short = [], 0
    files = sorted(RAW.glob("*.zip"))
    for zp in files:
        with zipfile.ZipFile(zp) as z:
            for nm in z.namelist():
                if not re.search(r"KYI\d{6}", nm, re.I):
                    continue
                for line in z.read(nm).split(b"\r\n"):
                    if len(line) < 395:
                        short += len(line) > 0
                        continue
                    rows.append(tuple(line[s - 1:s - 1 + n] for _, s, n, _ in FIELDS))
    log(f"KYI files {len(files)} rows {len(rows):,} short {short}")
    k = pd.DataFrame(rows, columns=[c for c, *_ in FIELDS])
    for c, _, _, t in FIELDS:
        v = k[c].str.decode("cp932", errors="replace").str.strip()
        k[c] = pd.to_numeric(v, errors="coerce") if t == "n" else v
    k = k.drop_duplicates(["racekey", "ban"], keep="last")
    k["k_pace_q"] = k.k_pace.map(pfx.QMAP)
    k = k.drop(columns=["k_pace"])
    k.to_parquet(KYI_PQ, index=False)
    return k


KPOS = ["k_d_jun_rel", "k_d_sa", "k_d_io", "k_a_jun_rel", "k_a_sa", "k_a_io", "k_g_jun_rel", "k_g_sa", "k_g_io", "k_tenkai"]
KIDX = ["k_ten", "k_pidx", "k_agari", "k_posidx", "k_pace_q", "k_style"]
KALL = KPOS + KIDX


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    k = parse_kyi()
    d = pfx.build_runs()
    d = pgc.add_fine(d)
    d = d[d.year >= 2013].reset_index(drop=True)
    d = d.merge(k, on=["racekey", "ban"], how="left")
    for p in ("d", "a", "g"):
        d[f"k_{p}_jun_rel"] = (d[f"k_{p}_jun"] - 1) / (d.n_start - 1).clip(lower=1)
    cov = d.groupby("year")["k_d_jun"].apply(lambda s: float(s.notna().mean()))
    log("KYI coverage by year " + json.dumps({int(y): round(v, 3) for y, v in cov.items()}))
    res = {"experiment": "kyi_tenkai", "date": "2026-11-15", "kyi_coverage": {int(y): v for y, v in cov.items()}}

    # 相手関係 (発走前に分かる)
    d["ten3_rank"] = d.groupby("rid_s")["ten3"].rank(ascending=False, pct=True)
    d["pf5_rank"] = d.groupby("rid_s")["pf5"].rank(pct=True)
    d["n_lead_others"] = (d.lead5 >= 0.4).astype(float).groupby(d.rid_s).transform("sum") - (d.lead5 >= 0.4).astype(float)
    d["ten3_top_others"] = d.groupby("rid_s")["ten3"].transform("max")
    PREF = pgc.PRE + ["ten3_rank", "pf5_rank", "n_lead_others", "ten3_top_others"]
    COND = ["dist", "td", "venue", "n_start", "ban_rel"]

    # A. 前半の位置の再現度
    reg = dict(pgc.PARAMS, objective="regression")
    dd = d[d.pf.notna()]
    res["A_r2_pf"] = {}
    preds = {}
    for tag, feats in (("A0_own", PREF), ("A1_own+kyi", PREF + KALL), ("A2_kyi_only", COND + KALL)):
        m, pred, tte = pgc.fit_predict(dd, feats, target="pf", params=reg)
        t = (tte.year >= 2024).values
        y = tte.pf.values[t]
        res["A_r2_pf"][tag] = float(1 - np.mean((y - pred[t]) ** 2) / np.var(y))
        preds[tag] = dict(zip(zip(tte.rid_s, tte.ban), pred))
        log(f"[A {tag}] R2(pf) test2024-25 {res['A_r2_pf'][tag]:.3f}")
    te = d[d.year >= 2023].reset_index(drop=True)
    sp = te.dropna(subset=["k_d_jun", "pf"])
    sp = sp[sp.year >= 2024]
    res["A_spearman_kyi_dochu_vs_pf"] = float(sp.groupby("rid_s")[["k_d_jun", "pf"]].corr(method="spearman")
                                              .xs("k_d_jun", level=1)["pf"].mean())
    log(f"[A] レース内 Spearman(KYI 道中順位, 実際の 1 角位置) 平均 {res['A_spearman_kyi_dochu_vs_pf']:.3f}")

    # B. ゲート
    z = te[["rid_s", "ban"]].copy()
    rng = np.random.default_rng(1)
    for tag, feats in (("pre", pgc.PRE), ("kyi", pgc.PRE + KALL), ("kpos", pgc.PRE + KPOS)):
        m, zz, tte = pgc.fit_predict(d, feats)
        z[f"z_{tag}"] = zz
        if tag == "kyi":
            sh = tte.copy()
            perm = sh.groupby("rid_s").cumcount().values  # placeholder for ordering
            idx = np.arange(len(sh))
            # レース内で KYI 列の行を入れ替える
            g = sh.groupby("rid_s").indices
            new = idx.copy()
            for _, ii in g.items():
                new[ii] = rng.permutation(ii)
            sh[KALL] = sh[KALL].values[new]
            z["z_kyi_shuf"] = m.predict(sh[feats], raw_score=True)
            del perm
    # P オラクルモデルに A1 の位置予測を入れる
    mP, _, _ = pgc.fit_predict(d, pgc.PRE + pgc.ORA["P"])
    rep = te.copy()
    rep["pf"] = [preds["A1_own+kyi"].get(kk, np.nan) for kk in zip(rep.rid_s, rep.ban)]
    rep["pf_dev"] = rep.pf - rep.pf5
    for c, src in (("lead", "lead"), ("deokure", "deokure")):
        mm, pr, tt = pgc.fit_predict(d[d[src].notna()], PREF + KALL, target=src, params=reg)
        mp = dict(zip(zip(tt.rid_s, tt.ban), pr))
        rep[c] = [mp.get(kk, np.nan) for kk in zip(rep.rid_s, rep.ban)]
    z["z_phat"] = mP.predict(rep[pgc.PRE + pgc.ORA["P"]], raw_score=True)

    res["gates"] = {}
    for c in ("pre", "kyi", "kpos", "phat", "kyi_shuf"):
        res["gates"][c] = pfx.short(gc.gate(z, [f"z_{c}"], name=c))
        res["gates"][c]["by_year"] = pfx.gate_years(z, [f"z_{c}"])
    log("[by_year] " + json.dumps({c: res["gates"][c]["by_year"] for c in res["gates"]}, ensure_ascii=False)[:2000])
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"wrote {OUT}")


if __name__ == "__main__":
    main()
