# -*- coding: utf-8 -*-
"""
build_substrate_v2.py — 馬券判断 AI v2 の学習用データ（substrate v2）
=====================================================================
仕様: docs/research/BET_DECISION_AI_V2_SPEC.md v0.1 §2・§3・§4
棚卸し: docs/research/BET_DECISION_AI_DATA_INVENTORY_20261022.md
v1: analysis/bet_decision_ai/build_substrate.py（PL 式・クラス正規化を import で流用。v1 のコードは変更しない）

行 = レース × 判断時点（O1 S1 断面、発走約 11 分前）で単勝オッズが付いている全出走馬 × {単勝, 複勝}。
期間 2021-01-05 〜 2026-08-22（O1 原本の範囲）。平地のみ。

**このスクリプトは方策の成績（回収率・的中率・ROI・損益の集計）を一切計算しない。** 検査は label-free のみ。
res_* 列（決済・結果）は作るが集計しない（状態の件数と、確定オッズとの一致率だけを記録）。

入力（すべて読み取りのみ）
  E:/競馬過去走データ/TAN_PUKU_WAKU_20210105-20260922.txt    JV O1 原本（判断時点 S1・確定 S3）
  data/master_v2_20130105-20251228.csv                        2021-2025 のレース情報・障害判定（トラックコード 51-59）
  data/weekly/{date}.csv                                      2026 のレース情報（出走表のレース見出し行）
  data/_research/mcond/oof_scores_c1master.parquet            2021-2023 C1 OOF 生スコア
  data/_policy/v6_scores.parquet                              2024-2025 C1 本番生スコア
  data/_research/bet_decision_ai/v2/c1_2026_scores.parquet    2026 C1 作り直しスコア（rescore_c1_2026.py）
  data/_research/mcond/base.parquet                           温度 τ の値だけ（OOF / 本番）
  data/harville_lambda.json                                   λ 補正 PL（v1 と同じ）
  data/kekka_20160105_20251228_v2.csv                         2021-2025 公式払戻（全頭版）
  data/kekka_20130105-20251228.csv                            同（1-3 着版、照合用）
  data/kekka/{date}.csv                                       2026 公式払戻（全頭）
出力
  data/_research/bet_decision_ai/v2/substrate_v2.parquet        候補行
  data/_research/bet_decision_ai/v2/substrate_v2_races.parquet  レース単位（除外理由・時刻・照合列）
  analysis/bet_decision_ai/v2/out/substrate_v2_checks.json      検査の数値（正本）
  analysis/bet_decision_ai/v2/SUBSTRATE_V2_REPORT.md            検査レポート（自動生成）
実行: python -m analysis.bet_decision_ai.v2.build_substrate_v2
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[2]
sys.path.insert(0, str(BASE))

from analysis.bet_decision_ai.build_substrate import (  # noqa: E402  v1 を流用（変更しない）
    CLASS_MAP, derived, load_race_meta, pl_tensors, sha256)
from analysis.bet_decision_ai.v2.o1_parser import classify_token, frame_for_horse, iter_o1  # noqa: E402

OUT_DIR = BASE / "data" / "_research" / "bet_decision_ai" / "v2"
OUT = OUT_DIR / "substrate_v2.parquet"
RACES_OUT = OUT_DIR / "substrate_v2_races.parquet"
CHECKS_OUT = HERE / "out" / "substrate_v2_checks.json"
REPORT = HERE / "SUBSTRATE_V2_REPORT.md"

O1_RAW = Path(r"E:\競馬過去走データ\TAN_PUKU_WAKU_20210105-20260922.txt")
O1_SHA256 = "54d8d7551a5c4e8911efd7ac83a2426a3c2bef7fd8dbd077aff14972a15c2b8b"   # EXP22 Stage 0 / v1 と同じ原本
WEEKLY = BASE / "data" / "weekly"
OOF = BASE / "data" / "_research" / "mcond" / "oof_scores_c1master.parquet"
V6S = BASE / "data" / "_policy" / "v6_scores.parquet"
C1_26 = OUT_DIR / "c1_2026_scores.parquet"
C1_26_MAN = OUT_DIR / "c1_2026_manifest.json"
BASE_PQ = BASE / "data" / "_research" / "mcond" / "base.parquet"
LAMBDA = BASE / "data" / "harville_lambda.json"
KEKKA_V2 = BASE / "data" / "kekka_20160105_20251228_v2.csv"
KEKKA_TOP3 = BASE / "data" / "kekka_20130105-20251228.csv"
KEKKA26 = BASE / "data" / "kekka"
JUMP_MANIFEST = BASE / "data" / "history_only" / "jump" / "manifest.json"
FORBIDDEN_INPUTS = ["oof_scores_v6params.parquet", "deepvalue_races_oof.pkl"]   # P0-5 リーク入り（使わない）

DATE_MIN, DATE_MAX = "20210105", "20260822"
JUMP_MIN, JUMP_MAX = 51, 59
S1_MIN_OK, S1_MAX_OK = 5.0, 20.0          # 判断時点として妥当とみなす「発走何分前」の範囲（検査用）
PERIOD_ROLE = {2021: "train", 2022: "train", 2023: "select", 2024: "confirm", 2025: "confirm", 2026: "holdout_2026"}
ZEN = str.maketrans("０１２３４５６７８９", "0123456789")


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


# =============================================================== loaders
def load_o1() -> tuple[dict, dict]:
    g = defaultdict(list)
    for s in iter_o1(O1_RAW):
        d = s.race_id[:8]
        if DATE_MIN <= d <= DATE_MAX:
            g[s.race_id].append(s)
    for v in g.values():
        v.sort(key=lambda x: x.announced_at)
    return dict(g), {"n_races": len(g), "n_records": int(sum(len(v) for v in g.values()))}


def load_weekly_meta(dates: list[str]) -> tuple[dict, list[str]]:
    """2026: 出走表 CSV のレース見出し行（列: レースID, 日付S, 曜日, 場所, 開催, R, レース名, クラス名, 芝・ダート,
    距離, コース区分, コーナー回数, 馬場状態(暫定), 天候(暫定), フルゲート頭数, 発走時刻, ...）。"""
    out, missing = {}, []
    for d in dates:
        p = WEEKLY / f"{d}.csv"
        if not p.exists():
            missing.append(d)
            continue
        with open(p, encoding="cp932", newline="") as f:
            for row in csv.reader(f):
                if row and re.fullmatch(r"\d{16}", row[0].strip()) and len(row) >= 16:
                    out[row[0].strip()] = {"venue": row[3].strip(), "class_raw": row[7].strip(),
                                           "surface_raw": row[8].strip(), "distance": row[9].strip(),
                                           "post_time": row[15].strip()}
    return out, missing


def _num(s) -> float:
    try:
        return float(str(s).strip())
    except (TypeError, ValueError):
        return np.nan


def load_kekka(rids: set) -> tuple[dict, dict]:
    """rid16 -> {ban: (fin, tan_pay, tan_paren_odds, fuku_pay)}。fin 0 = 完走していない（取消・除外・中止など）。"""
    cols = ["レースID(新)", "馬番", "確定着順", "単勝配当", "複勝配当"]
    parts = [pd.read_csv(KEKKA_V2, encoding="cp932", dtype=str, usecols=cols)]
    for d in sorted({r[:8] for r in rids if r[:4] == "2026"}):
        p = KEKKA26 / f"{d}.csv"
        if p.exists():
            parts.append(pd.read_csv(p, encoding="cp932", dtype=str, usecols=cols))
    k = pd.concat(parts, ignore_index=True)
    k["rid16"] = k["レースID(新)"].astype(str).str.strip().str[:16]
    k = k[k["rid16"].isin(rids)].copy()
    k["ban"] = pd.to_numeric(k["馬番"], errors="coerce")
    k["fin"] = pd.to_numeric(k["確定着順"].astype(str).str.strip().str.translate(ZEN), errors="coerce")
    ts = k["単勝配当"].astype(str).str.strip()
    k["tan_pay"] = pd.to_numeric(ts, errors="coerce")
    k["tan_paren"] = pd.to_numeric(ts.str.extract(r"^\(([\d.]+)\)$")[0], errors="coerce")
    k["fuku_pay"] = pd.to_numeric(k["複勝配当"].astype(str).str.strip(), errors="coerce")
    stats = {"rows": int(len(k)), "races": int(k["rid16"].nunique()),
             "dup_rows": int(k.duplicated(["rid16", "ban"]).sum()), "fin_nan": int(k["fin"].isna().sum())}
    k = k.dropna(subset=["ban"]).drop_duplicates(["rid16", "ban"], keep="first")
    out: dict = {}
    for r in k[["rid16", "ban", "fin", "tan_pay", "tan_paren", "fuku_pay"]].itertuples(index=False):
        out.setdefault(r.rid16, {})[int(r.ban)] = (r.fin, r.tan_pay, r.tan_paren, r.fuku_pay)
    return out, stats


def load_kekka_top3(rids: set) -> dict:
    k = pd.read_csv(KEKKA_TOP3, encoding="cp932", dtype=str)
    k.columns = ["rid", "ban", "hid", "fin", "tan", "fuku", "waku", "umaren", "umatan", "sanpuku", "sanrentan"]
    k["rid16"] = k["rid"].astype(str).str[:16]
    k = k[k["rid16"].isin(rids)]
    out = {}
    for r in k.itertuples(index=False):
        out[(r.rid16, int(r.ban))] = (_num(r.tan), _num(r.fuku))
    return out


def load_scores() -> tuple[pd.DataFrame, dict, dict]:
    for f in FORBIDDEN_INPUTS:
        assert f not in str(OOF) and f not in str(V6S)
    b = pd.read_parquet(BASE_PQ, columns=["v6_src", "tau"]).drop_duplicates()
    tau = {r.v6_src: float(r.tau) for r in b.itertuples(index=False)}
    assert set(tau) == {"oof", "prod"}, tau
    o = pd.read_parquet(OOF).rename(columns={"rid": "rid16"})
    o["rid16"] = o["rid16"].astype(str)
    o = o[o["year"].between(2021, 2023)][["rid16", "ban", "score"]].assign(score_src="oof_c1master", tau=tau["oof"])
    v = pd.read_parquet(V6S).rename(columns={"rid": "rid16"})
    v["rid16"] = v["rid16"].astype(str)
    v = v[v["rid16"].str[:4].astype(int).between(2024, 2025)][["rid16", "ban", "score"]]
    v = v.assign(score_src="v6_scores_c1", tau=tau["prod"])
    parts = [o, v]
    man: dict = {}
    if C1_26.exists():
        c = pd.read_parquet(C1_26)
        # 2026-03-07〜04-12 は data/hosei の H_*.csv 結合（predict_weekly._load_hosei が全ファイルを結合し、
        # 同じ馬に別の補正値を持つ行が残る）で 1 頭が 2 行に増え、スコアが 2 つ出る。結果に依存しない規則として
        # 平均を取り、重複数と幅を列に残す。
        c = (c.groupby(["rid16", "ban"], as_index=False)
             .agg(score=("score", "mean"), score_dup_n=("score", "size"),
                  score_dup_range=("score", lambda s: float(s.max() - s.min()))))
        c = c.assign(score_src="c1_rescore_2026", tau=tau["prod"])
        parts.append(c)
        if C1_26_MAN.exists():
            man = json.loads(C1_26_MAN.read_text(encoding="utf-8"))
    s = pd.concat(parts, ignore_index=True)
    s["score"] = s["score"].astype(float)
    if "score_dup_n" not in s:
        s["score_dup_n"] = 1
    s["score_dup_n"] = s["score_dup_n"].fillna(1).astype(int)
    s["score_dup_range"] = s.get("score_dup_range", pd.Series(0.0, index=s.index)).fillna(0.0)
    dup = int(s.duplicated(["rid16", "ban"]).sum())
    assert dup == 0, dup
    c26 = s[s["score_src"] == "c1_rescore_2026"]
    d26 = c26[c26["score_dup_n"] > 1]
    return s, {"tau": tau, "dup_rows_after_merge": dup, "rows_by_src": s["score_src"].value_counts().to_dict(),
               "c1_2026_dup_horses": int(len(d26)), "c1_2026_dup_dates": sorted(d26["rid16"].str[:8].unique().tolist()),
               "c1_2026_dup_range": _q(d26["score_dup_range"], (0.5, 0.9, 1.0)) if len(d26) else {}}, man


# =============================================================== helpers
def snap_pick(rows: list, post_dt):
    """S1（判断時点）を選ぶ。4 録（S0/S1/S2/S3）以上なら発表時刻順の 2 番目（EXP22 の S1 定義）。
    2 録以上で 2 番目が区分 1 でなければ選ばない。1 録だけのレースは、その録が区分 1 で発走 5〜20 分前なら S1 とする。"""
    n = len(rows)
    if n >= 2:
        s1 = rows[1] if rows[1].division == 1 else None
        rule = "second_of_n" if s1 is not None else "second_not_div1"
    else:
        s1, rule = None, "single_record_out_of_window"
        r0 = rows[0]
        if r0.division == 1 and post_dt is not None:
            m = (post_dt - r0.announced_at).total_seconds() / 60.0
            if S1_MIN_OK <= m <= S1_MAX_OK:
                s1, rule = r0, "single_record_in_window"
    s0 = rows[0] if n >= 2 else None
    s3 = rows[-1] if (n >= 3 and rows[-1].division in (3, 4, 5)) else None
    return s0, s1, s3, rule


def norm_entropy(q: np.ndarray) -> float:
    q = q[q > 1e-15]
    if len(q) < 2:
        return np.nan
    return float(-(q * np.log(q)).sum() / np.log(len(q)))


def parse_post(year: int, rid: str, s: str):
    m = re.match(r"(\d{1,2}):(\d{2})", str(s or ""))
    if not m:
        return None
    return datetime(year, int(rid[4:6]), int(rid[6:8]), int(m.group(1)), int(m.group(2)))


# =============================================================== build
def build() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    t0 = time.time()
    lam = json.loads(LAMBDA.read_text(encoding="utf-8"))
    l1, l2 = float(lam["lambda1"]), float(lam["lambda2"])

    log("O1 raw")
    o1_sha = sha256(O1_RAW)
    assert o1_sha == O1_SHA256, f"O1 原本が変わった: {o1_sha}"
    o1, o1_stats = load_o1()
    rids = sorted(o1)
    log(f"  O1 races={len(rids):,} records={o1_stats['n_records']:,}")

    log("race meta: master_v2 (2021-2025)")
    meta, _waku = load_race_meta()
    dates26 = sorted({r[:8] for r in rids if r[:4] == "2026"})
    log("race meta: weekly (2026)")
    wmeta, weekly_missing = load_weekly_meta(dates26)
    jm = json.loads(JUMP_MANIFEST.read_text(encoding="utf-8")) if JUMP_MANIFEST.exists() else {}

    log("scores")
    scores, score_info, man26 = load_scores()
    smap = {rid: dict(zip(g["ban"].astype(int), zip(g["score"], g["score_src"], g["tau"], g["score_dup_n"])))
            for rid, g in scores.groupby("rid16", sort=False)}
    day26 = man26.get("days", {})

    log("kekka")
    kek, kek_stats = load_kekka(set(rids))
    k3 = load_kekka_top3({r for r in rids if r[:4] <= "2025"})

    rows = defaultdict(list)
    rrec = []
    log(f"race loop ({len(rids):,})")
    for ri, rid in enumerate(rids):
        if ri and ri % 4000 == 0:
            log(f"  {ri:,}")
        year = int(rid[:4])
        snaps = o1[rid]
        rr = {"rid16": rid, "year": year, "date": rid[:8], "venue_code": rid[8:10],
              "n_o1_records": len(snaps), "o1_divisions": "".join(str(s.division) for s in snaps),
              "n_registered": snaps[0].registered_headcount}
        # ---------- レース情報・障害除外
        if year <= 2025:
            if rid not in meta.index:
                rr["excluded"] = "no_master_meta"
                rrec.append(rr)
                continue
            m = meta.loc[rid]
            if bool(m["is_jump"]):
                rr["excluded"] = "jump_track_code"
                rrec.append(rr)
                continue
            rr.update(venue=m["venue"], surface=m["surface"], distance=int(m["distance"]) if pd.notna(m["distance"]) else np.nan,
                      class_raw=m["class_raw"], track_code=int(m["track_code"]) if pd.notna(m["track_code"]) else np.nan,
                      post_time=m["post_time"], meta_src="master_v2")
        else:
            w = wmeta.get(rid)
            if w is None:
                # data/weekly は 2026-03-07 以降 障害レースを含まない（data/history_only/jump/manifest.json の
                # known_unrecoverable_gap）。出走表に無い O1 レースは障害とみなして除外し、件数を記録する
                rr["excluded"] = "not_in_weekly_assumed_jump"
                rrec.append(rr)
                continue
            if w["surface_raw"].startswith("障害"):
                rr["excluded"] = "jump_weekly_surface"
                rrec.append(rr)
                continue
            rr.update(venue=w["venue"], surface={"ダート": "ダ"}.get(w["surface_raw"], w["surface_raw"]),
                      distance=int(w["distance"]) if w["distance"].isdigit() else np.nan,
                      class_raw=w["class_raw"], track_code=np.nan, post_time=w["post_time"], meta_src="weekly_csv")
        rr["class_norm"] = CLASS_MAP.get(rr["class_raw"], rr["class_raw"])
        post_dt = parse_post(year, rid, rr["post_time"])

        def mbp(s):
            return (post_dt - s.announced_at).total_seconds() / 60.0 if (s is not None and post_dt is not None) else np.nan

        s0, s1, s3, rule = snap_pick(snaps, post_dt)
        rr["s1_rule"] = rule
        rr["s0_min_before_post"] = mbp(s0)
        rr["s1_min_before_post"] = mbp(s1)
        rr["s3_min_after_post"] = -mbp(s3) if s3 is not None else np.nan
        rr["s3_division"] = s3.division if s3 is not None else np.nan
        if s1 is None:
            rr["excluded"] = "no_s1_snapshot"
            rrec.append(rr)
            continue
        rr["s1_announced"] = f"{rid[:4]}{s1.announced_mmddhhmm}"
        rr["s1_to_s3_min"] = ((s3.announced_at - s1.announced_at).total_seconds() / 60.0) if s3 is not None else np.nan
        rr["s1_timing_ok"] = bool(S1_MIN_OK <= rr["s1_min_before_post"] <= S1_MAX_OK) if np.isfinite(rr["s1_min_before_post"]) else False
        rr["s1_division"] = s1.division
        cands = sorted(s1.tansho)
        n = len(cands)
        rr["n_field_s1"] = n
        rr["n_starters_hdr_s1"] = s1.starter_count
        rr["n_scratch_before_s1"] = s1.registered_headcount - n
        rr["n_tan_asterisk_s1"] = sum(1 for t in s1.tansho_tokens.values() if classify_token(t) == "asterisk")
        if n < 2:
            rr["excluded"] = "lt2_priced_at_s1"
            rrec.append(rr)
            continue
        fuku_places = int(s1.place_pay_key) if s1.place_pay_key.isdigit() else np.nan
        fuku_offered = s1.sale_flags[1:2] == "7"
        rr.update(fuku_places=fuku_places, fuku_offered=fuku_offered, sale_flags_s1=s1.sale_flags,
                  votes_tan_s1=s1.total_tansho_votes, votes_fuku_s1=s1.total_fukusho_votes,
                  fuku_places_s3=int(s3.place_pay_key) if (s3 is not None and s3.place_pay_key.isdigit()) else np.nan)

        # ---------- 判断時点の価格（S1）
        tan = np.array([s1.tansho[b] for b in cands])
        inv = 1.0 / tan
        q_tan = inv / inv.sum()
        tan_rank = pd.Series(tan).rank(method="min").to_numpy()
        lo = np.array([s1.fukusho[b][0] if b in s1.fukusho else np.nan for b in cands])
        hi = np.array([s1.fukusho[b][1] if b in s1.fukusho else np.nan for b in cands])
        mid = (lo + hi) / 2.0
        okf = np.isfinite(mid)
        q_fuku = np.full(n, np.nan)
        if okf.any() and np.isfinite(fuku_places):
            im = 1.0 / mid[okf]
            q_fuku[okf] = im / im.sum() * fuku_places
            rr["overround_fuku_mid"] = float(im.sum() / fuku_places)
            rr["overround_fuku_lo"] = float((1.0 / lo[okf]).sum() / fuku_places)
            rr["overround_fuku_hi"] = float((1.0 / hi[okf]).sum() / fuku_places)
        rr["n_fuku_priced_s1"] = int(okf.sum())
        qs = np.sort(q_tan)[::-1]
        rr.update(overround_tan=float(inv.sum()), mkt_entropy_tan=norm_entropy(q_tan),
                  mkt_q_top1=float(qs[0]), mkt_q_gap12=float(qs[0] - qs[1]))

        # ---------- モデル（生スコア → レース内 softmax(score/τ) → λ 補正 PL）
        sc = smap.get(rid, {})
        has = np.array([b in sc for b in cands])
        score = np.array([sc[b][0] if b in sc else np.nan for b in cands])
        dupn = np.array([sc[b][3] if b in sc else np.nan for b in cands], dtype=float)
        src = next((sc[b][1] for b in cands if b in sc), None)
        tau = next((sc[b][2] for b in cands if b in sc), np.nan)
        rr["n_scored"] = int(has.sum())
        rr["n_scores_not_priced_s1"] = len(set(sc) - set(cands))
        rr["score_src"] = src
        p_win = np.full(n, np.nan)
        p_top2 = np.full(n, np.nan)
        p_top3 = np.full(n, np.nan)
        mrank = np.full(n, np.nan)
        if has.sum() >= 2:
            z = score[has] / tau
            wv = np.exp(z - z.max())
            pw = wv / wv.sum()
            p_win[has] = pw
            idx = np.flatnonzero(has)
            order = sorted(range(len(idx)), key=lambda i: (-score[idx[i]], cands[idx[i]]))
            for r_, i in enumerate(order, 1):
                mrank[idx[i]] = r_
            if has.sum() >= 3:
                A, P3 = pl_tensors(pw, l1, l2)
                D = derived(A, P3)
                p_top2[has] = D["top2"]
                p_top3[has] = D["top3"]
                rr["psum_top3"] = float(D["top3"].sum())
                rr["psum_top2"] = float(D["top2"].sum())
            else:
                p_top2[has] = 1.0
            ps = np.sort(pw)[::-1]
            rr.update(model_entropy=norm_entropy(pw), model_p_top1=float(ps[0]), model_p_gap12=float(ps[0] - ps[1]),
                      psum_win=float(pw.sum()))
            if has.sum() >= 3:
                rr["spearman_score_vs_mkt"] = float(pd.Series(score[has]).corr(pd.Series(-np.log(tan[has])),
                                                                                 method="spearman"))
        p_place = p_top3 if fuku_places == 3 else p_top2
        if year == 2026:
            di = day26.get(rid[:8], {})
            rr["score_day_gate_fail"] = di.get("gate_fail")
            rr["score_day_dead_gain_pct"] = di.get("dead_gain_pct")
            rr["score_day_kako5_missing"] = di.get("kako5_missing")

        # ---------- 決済（結果列。学習入力に使わない）
        kr = kek.get(rid)
        s3_started = set(s3.tansho) if s3 is not None else None
        fins = {}
        if kr:
            for b_, (fin, tp, tpar, fp) in kr.items():
                fins[b_] = fin
        winners = [b_ for b_, v in (kr or {}).items() if v[0] == 1]
        placed_n = sum(1 for v in (kr or {}).values() if np.isfinite(v[3]) and v[3] > 0)
        rr["deadheat_win"] = len(winners) >= 2
        rr["n_fuku_paid"] = placed_n
        if np.isfinite(fuku_places):
            rr["deadheat_place"] = sum(1 for v in (kr or {}).values() if np.isfinite(v[0]) and 1 <= v[0] <= fuku_places) > fuku_places
        rr["kekka_found"] = kr is not None
        rr["tan_paid_in_race"] = any(np.isfinite(v[1]) for v in (kr or {}).values())
        n_inc = n_scr = n_dnf = n_krow_missing = 0
        st_rows = []
        for i, b in enumerate(cands):
            kv = kr.get(b) if kr else None
            if kv is not None:
                fin, tp, tpar, fp = kv
                k_started = (np.isfinite(fin) and fin > 0) or np.isfinite(tp) or np.isfinite(tpar)
            else:
                fin = tp = tpar = fp = np.nan
                k_started = None
            started = (b in s3_started) if s3_started is not None else k_started
            if s3_started is not None and k_started is not None and (b in s3_started) != k_started:
                n_inc += 1
            if kv is None:
                n_krow_missing += 1
            st_rows.append((fin, tp, fp, started, kv is not None))
            if started is False:
                n_scr += 1
            elif started and np.isfinite(fin) and fin == 0:
                n_dnf += 1
        rr.update(n_scratch_after_s1=n_scr, n_dnf=n_dnf, n_kekka_o1_inconsistent=n_inc, n_kekka_row_missing=n_krow_missing)

        def settle(bt, fin, tp, fp, started, krow):
            if not rr["kekka_found"]:
                return np.nan, "no_kekka_race", "unsettleable"
            if started is False:
                return 100.0, "refund_scratch", "refund"
            if started is None or not krow:
                return np.nan, "kekka_row_missing", "unsettleable"
            if np.isfinite(fin) and fin == 0:
                return 0.0, "dnf", "dnf"
            if bt == "tan":
                if not rr["tan_paid_in_race"]:
                    return np.nan, "no_tan_payout_in_race", "unsettleable"
                if np.isfinite(tp) and tp > 0:
                    return float(tp), "kekka", "win"
                return 0.0, "lose", "lose"
            if placed_n == 0:
                return np.nan, "no_fuku_payout_in_race", "unsettleable"
            if np.isfinite(fp) and fp > 0:
                return float(fp), "kekka", "win"
            return 0.0, "lose", "lose"

        # ---------- 確定オッズ（S3）と公式払戻の照合（label-free: 一致率だけ）
        if s3 is not None and kr:
            if len(winners) == 1:
                w1 = winners[0]
                tp = kr[w1][1]
                if w1 in s3.tansho and np.isfinite(tp):
                    rr["chk_tan_eq"] = bool(round(s3.tansho[w1] * 100) == round(tp))
                    rr["chk_tan_ratio"] = float(tp / (s3.tansho[w1] * 100))
            elif len(winners) >= 2:
                rat = [kr[w][1] / (s3.tansho[w] * 100) for w in winners if w in s3.tansho and np.isfinite(kr[w][1])]
                rr["chk_tan_deadheat_ratio_mean"] = float(np.mean(rat)) if rat else np.nan
            ins = []
            for b_, v in kr.items():
                if rr.get("deadheat_place"):
                    break                      # 複勝圏の同着は払戻頭数が増え確定オッズ範囲より低くなる（別記）
                if np.isfinite(v[3]) and v[3] > 0 and b_ in s3.fukusho:
                    flo, fhi = s3.fukusho[b_]
                    ins.append(bool(flo * 100 - 0.5 <= v[3] <= fhi * 100 + 0.5))
            if ins:
                rr["chk_fuku_in_range"] = float(np.mean(ins))
                rr["chk_fuku_n"] = len(ins)
        if year <= 2025 and kr:
            agree = []
            for b_, v in kr.items():
                t3 = k3.get((rid, b_))
                if t3 is not None:
                    a = (np.isnan(v[1]) and np.isnan(t3[0])) or (np.isfinite(v[1]) and np.isfinite(t3[0]) and v[1] == t3[0])
                    f = (np.isnan(v[3]) and np.isnan(t3[1])) or (np.isfinite(v[3]) and np.isfinite(t3[1]) and v[3] == t3[1])
                    agree.append(a and f)
            rr["chk_kekka_v2_eq_top3file"] = float(np.mean(agree)) if agree else np.nan

        # ---------- 行の出力
        for i, b in enumerate(cands):
            fin, tp, fp, started, krow = st_rows[i]
            for bt in ("tan", "fuku"):
                if bt == "fuku" and not fuku_offered:
                    continue
                pay, psrc, st = settle(bt, fin, tp, fp, started, krow)
                R = rows
                R["rid16"].append(rid); R["umaban"].append(b); R["bet_type"].append(bt)
                R["waku"].append(frame_for_horse(b, s1.registered_headcount))
                R["odds_tan"].append(tan[i]); R["odds_fuku_lo"].append(lo[i]); R["odds_fuku_hi"].append(hi[i])
                R["odds_fuku_mid"].append(mid[i])
                R["q_tan_mkt"].append(q_tan[i]); R["q_fuku_mkt"].append(q_fuku[i]); R["tan_odds_rank"].append(tan_rank[i])
                R["score"].append(score[i]); R["score_missing"].append(not has[i]); R["score_dup_n"].append(dupn[i])
                R["p_win_sm"].append(p_win[i]); R["p_top2_pl"].append(p_top2[i]); R["p_top3_pl"].append(p_top3[i])
                R["p_place_model"].append(p_place[i]); R["model_rank"].append(mrank[i])
                pb = p_win[i] if bt == "tan" else p_place[i]
                qb = q_tan[i] if bt == "tan" else q_fuku[i]
                R["p_bet"].append(pb); R["q_bet"].append(qb)
                R["log_p_over_q"].append(np.log(pb / qb) if (np.isfinite(pb) and np.isfinite(qb) and pb > 0 and qb > 0) else np.nan)
                R["res_status"].append(st); R["res_payout_100"].append(pay); R["res_payout_src"].append(psrc)
                R["res_fin"].append(fin)
                R["res_scratched_after_s1"].append(started is False)
                R["res_dnf"].append(bool(started and np.isfinite(fin) and fin == 0))
                R["res_close_tan"].append(s3.tansho.get(b, np.nan) if s3 is not None else np.nan)
                cl = s3.fukusho.get(b) if s3 is not None else None
                R["res_close_fuku_lo"].append(cl[0] if cl else np.nan)
                R["res_close_fuku_hi"].append(cl[1] if cl else np.nan)
        rrec.append(rr)

    log("assemble")
    races = pd.DataFrame(rrec)
    sub = pd.DataFrame(rows)
    race_cols = ["year", "date", "venue_code", "venue", "surface", "distance", "class_raw", "class_norm", "track_code",
                 "post_time", "meta_src", "n_registered", "n_field_s1", "n_starters_hdr_s1", "fuku_places", "fuku_offered",
                 "s1_announced", "s1_min_before_post", "s1_timing_ok", "s1_to_s3_min", "s1_rule", "n_o1_records", "o1_divisions",
                 "overround_tan", "overround_fuku_mid", "overround_fuku_lo", "overround_fuku_hi",
                 "votes_tan_s1", "votes_fuku_s1", "mkt_entropy_tan", "mkt_q_top1", "mkt_q_gap12",
                 "score_src", "n_scored", "model_entropy", "model_p_top1", "model_p_gap12",
                 "score_day_gate_fail", "score_day_dead_gain_pct", "score_day_kako5_missing",
                 "deadheat_win", "deadheat_place", "n_scratch_after_s1", "n_dnf", "s3_min_after_post"]
    for c in race_cols:
        if c not in races.columns:
            races[c] = np.nan
    races["period_role"] = races["year"].map(PERIOD_ROLE)
    keep = races[races["excluded"].isna()] if "excluded" in races else races
    sub = sub.merge(keep[["rid16", "period_role"] + race_cols], on="rid16", how="left")
    # 結果に依存するレース列は res_ を付けて区別する
    sub = sub.rename(columns={"deadheat_win": "res_race_deadheat_win", "deadheat_place": "res_race_deadheat_place",
                              "n_scratch_after_s1": "res_race_n_scratch_after_s1", "n_dnf": "res_race_n_dnf",
                              "s3_min_after_post": "res_s3_min_after_post", "s1_to_s3_min": "res_s1_to_s3_min"})
    for c in ["umaban", "waku", "n_registered", "n_field_s1", "year"]:
        sub[c] = sub[c].astype(np.int16)
    for c in ["bet_type", "res_status", "res_payout_src", "score_src", "s1_rule", "period_role", "venue", "surface",
              "class_raw", "class_norm", "meta_src", "venue_code"]:
        sub[c] = sub[c].astype("category")
    order = (["rid16", "year", "date", "period_role", "venue_code", "venue", "surface", "distance", "class_raw",
              "class_norm", "track_code", "post_time", "meta_src", "n_registered", "n_field_s1", "n_starters_hdr_s1",
              "umaban", "waku", "bet_type",
              "s1_announced", "s1_min_before_post", "s1_timing_ok", "s1_rule", "n_o1_records", "o1_divisions",
              "odds_tan", "odds_fuku_lo", "odds_fuku_hi", "odds_fuku_mid", "fuku_places", "fuku_offered",
              "overround_tan", "overround_fuku_mid", "overround_fuku_lo", "overround_fuku_hi", "votes_tan_s1", "votes_fuku_s1",
              "q_tan_mkt", "q_fuku_mkt", "tan_odds_rank", "mkt_entropy_tan", "mkt_q_top1", "mkt_q_gap12",
              "score", "score_missing", "score_src", "score_dup_n", "n_scored", "p_win_sm", "p_top2_pl", "p_top3_pl", "p_place_model",
              "model_rank", "model_entropy", "model_p_top1", "model_p_gap12", "p_bet", "q_bet", "log_p_over_q",
              "score_day_gate_fail", "score_day_dead_gain_pct", "score_day_kako5_missing",
              "res_status", "res_payout_100", "res_payout_src", "res_fin", "res_scratched_after_s1", "res_dnf",
              "res_race_deadheat_win", "res_race_deadheat_place", "res_race_n_scratch_after_s1", "res_race_n_dnf",
              "res_close_tan", "res_close_fuku_lo", "res_close_fuku_hi", "res_s3_min_after_post", "res_s1_to_s3_min"])
    sub = sub[order]
    info = {"elapsed_sec": round(time.time() - t0, 1), "lambda": [l1, l2], "o1_sha256": o1_sha, "o1_stats": o1_stats,
            "score_info": score_info, "kekka_stats": kek_stats, "weekly_missing_dates": weekly_missing,
            "jump_manifest_gap": jm.get("known_unrecoverable_gap"), "c1_2026_manifest": {k: v for k, v in man26.items() if k != "days"},
            "c1_2026_days": day26}
    return sub, races, info


# =============================================================== label-free checks
def _q(s: pd.Series, qs=(0.0, 0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99, 1.0)) -> dict:
    s = pd.Series(s).dropna().astype(float)
    if len(s) == 0:
        return {"n": 0}
    return {f"p{int(round(q * 100)):02d}": round(float(s.quantile(q)), 4) for q in qs} | {"n": int(len(s))}


def _sd(s: pd.Series) -> dict:
    """(Multi)Index の Series → JSON に出せる {str: int}。"""
    return {str(k): int(v) for k, v in s.items()}


OPTIONAL_RACE_COLS = ["chk_tan_eq", "chk_tan_ratio", "chk_tan_deadheat_ratio_mean", "chk_fuku_in_range", "chk_fuku_n",
                      "chk_kekka_v2_eq_top3file", "psum_top3", "psum_top2", "psum_win", "spearman_score_vs_mkt",
                      "overround_fuku_mid", "overround_fuku_lo", "overround_fuku_hi", "deadheat_place", "excluded"]


def checks(sub: pd.DataFrame, races: pd.DataFrame, info: dict) -> dict:
    """方策の成績（回収率・的中・ROI・損益の集計）は計算しない。"""
    c: dict = {"build_info": info}
    for col in OPTIONAL_RACE_COLS:
        if col not in races.columns:
            races[col] = np.nan
    kept = races[races["excluded"].isna()].copy()
    kept["deadheat_win"] = kept["deadheat_win"].eq(True)
    kept["deadheat_place"] = kept["deadheat_place"].eq(True)
    c["n_rows"] = int(len(sub))
    c["n_races"] = int(sub["rid16"].nunique())
    c["columns"] = list(sub.columns)
    c["races_o1_by_year"] = races.groupby("year").size().to_dict()
    c["excluded_by_year_reason"] = (races[races["excluded"].notna()].groupby(["year", "excluded"]).size()
                                    .unstack(fill_value=0).to_dict(orient="index"))
    c["races_kept_by_year"] = kept.groupby("year").size().to_dict()
    c["rows_by_year_type"] = sub.groupby(["year", "bet_type"], observed=True).size().unstack(fill_value=0).to_dict(orient="index")
    c["cands_per_race_by_year"] = {int(y): _q(g["n_field_s1"], (0.0, 0.5, 1.0)) for y, g in kept.groupby("year")}
    c["dates_by_year"] = kept.groupby("year")["date"].nunique().to_dict()
    # 時点
    c["s1_rule_counts"] = kept["s1_rule"].value_counts().to_dict()
    c["s1_rule_all_races"] = races["s1_rule"].value_counts(dropna=False).astype(int).rename(index=str).to_dict()
    c["s1_min_before_post_by_year"] = {int(y): _q(g["s1_min_before_post"]) for y, g in kept.groupby("year")}
    c["s0_min_before_post_by_year"] = {int(y): _q(g["s0_min_before_post"], (0.01, 0.5, 0.99)) for y, g in kept.groupby("year")}
    c["s3_min_after_post_by_year"] = {int(y): _q(g["s3_min_after_post"], (0.01, 0.5, 0.99)) for y, g in kept.groupby("year")}
    m = kept["s1_min_before_post"]
    c["s1_flags"] = {"post_time_missing": int(m.isna().sum()), "at_or_after_post": int((m <= 0).sum()),
                     "lt_5min": int(((m > 0) & (m < S1_MIN_OK)).sum()), "gt_20min": int((m > S1_MAX_OK).sum()),
                     "s1_division_not1": int((kept["s1_division"] != 1).sum()),
                     "races_lt_5_or_gt_20": kept.loc[(m < S1_MIN_OK) | (m > S1_MAX_OK),
                                                     ["rid16", "s1_min_before_post", "o1_divisions"]].head(30).to_dict(orient="records")}
    c["s1_to_s3_min_by_year"] = {int(y): _q(g["s1_to_s3_min"], (0.0, 0.01, 0.5, 0.99, 1.0)) for y, g in kept.groupby("year")}
    bad_t = kept[~kept["s1_timing_ok"].astype(bool)]
    c["s1_timing_not_ok"] = {"n": int(len(bad_t)), "by_date": _sd(bad_t.groupby("date").size()),
                             "s1_to_s3_min": _q(bad_t["s1_to_s3_min"], (0.0, 0.5, 1.0)),
                             "s3_min_after_sched_post": _q(bad_t["s3_min_after_post"], (0.0, 0.5, 1.0))}
    c["s1_min_before_post_value_counts"] = m.round(0).value_counts().sort_index().astype(int).rename(index=lambda x: str(int(x))).to_dict()
    # 価格
    c["overround_tan_by_year"] = {int(y): _q(g["overround_tan"], (0.01, 0.05, 0.5, 0.95, 0.99)) for y, g in kept.groupby("year")}
    c["overround_fuku_mid_by_year"] = {int(y): _q(g["overround_fuku_mid"], (0.01, 0.05, 0.5, 0.95, 0.99)) for y, g in kept.groupby("year")}
    c["overround_fuku_lo_by_year"] = {int(y): _q(g["overround_fuku_lo"], (0.05, 0.5, 0.95)) for y, g in kept.groupby("year")}
    c["overround_fuku_hi_by_year"] = {int(y): _q(g["overround_fuku_hi"], (0.05, 0.5, 0.95)) for y, g in kept.groupby("year")}
    c["overround_tan_lt_1_23_rate_by_year"] = kept.assign(f=kept["overround_tan"] < 1.23).groupby("year")["f"].mean().round(4).to_dict()
    tanr = sub[sub["bet_type"] == "tan"]
    qs = tanr.groupby("rid16")["q_tan_mkt"].sum()
    c["q_tan_sum"] = _q(qs, (0.0, 0.5, 1.0))
    fr = sub[sub["bet_type"] == "fuku"]
    qf = fr.groupby("rid16").agg(s=("q_fuku_mkt", "sum"), k=("fuku_places", "first"))
    c["q_fuku_sum_minus_places"] = _q(qf["s"] - qf["k"], (0.0, 0.5, 1.0))
    c["fuku_places_s1"] = kept["fuku_places"].value_counts().to_dict()
    c["fuku_places_s1_vs_n_field"] = _sd(kept.groupby([kept["n_field_s1"] >= 8, "fuku_places"]).size())
    c["fuku_places_s1_ne_s3"] = int(((kept["fuku_places"] != kept["fuku_places_s3"]) & kept["fuku_places_s3"].notna()).sum())
    c["fuku_offered_false_races"] = int((~kept["fuku_offered"].astype(bool)).sum())
    # モデル確率
    c["score_src_rows_by_year"] = sub.groupby(["year", "score_src"], observed=True).size().unstack(fill_value=0).to_dict(orient="index")
    c["score_missing_rate_by_year"] = tanr.groupby("year")["score_missing"].mean().round(5).to_dict()
    c["races_all_scores_missing_by_year"] = kept.assign(f=kept["n_scored"] == 0).groupby("year")["f"].sum().astype(int).to_dict()
    sm = tanr[tanr["score_missing"]]
    cause = np.where(sm["res_scratched_after_s1"], "scratched_after_s1",
                     np.where(sm["res_dnf"], "dnf", "other"))
    c["score_missing_cause_by_year"] = (pd.DataFrame({"year": sm["year"], "cause": cause}).groupby(["year", "cause"])
                                        .size().unstack(fill_value=0).to_dict(orient="index"))
    c["psum_win"] = _q(kept["psum_win"], (0.0, 0.5, 1.0))
    c["psum_top3_minus3"] = _q(kept["psum_top3"] - 3, (0.0, 0.5, 1.0))
    c["psum_top2_minus2"] = _q(kept["psum_top2"] - 2, (0.0, 0.5, 1.0))
    c["score_dist_by_year"] = {int(y): _q(g["score"], (0.01, 0.5, 0.99)) | {"mean": round(float(g["score"].mean()), 4),
                                                                           "std": round(float(g["score"].std()), 4)}
                               for y, g in tanr.groupby("year")}
    c["model_entropy_by_year"] = {int(y): _q(g["model_entropy"], (0.05, 0.5, 0.95)) for y, g in kept.groupby("year")}
    c["model_p_top1_by_year"] = {int(y): _q(g["model_p_top1"], (0.05, 0.5, 0.95)) for y, g in kept.groupby("year")}
    c["spearman_score_vs_mkt_by_year"] = {int(y): _q(g["spearman_score_vs_mkt"], (0.25, 0.5, 0.75)) for y, g in kept.groupby("year")}
    k26 = kept[kept["year"] == 2026]
    if len(k26):
        c["spearman_2026_by_gate"] = {str(k): _q(g["spearman_score_vs_mkt"], (0.25, 0.5, 0.75))
                                      for k, g in k26.groupby(k26["score_day_gate_fail"].astype(str))}
        c["races_2026_by_gate"] = k26["score_day_gate_fail"].astype(str).value_counts().to_dict()
        c["spearman_2026_by_kako5"] = {str(k): _q(g["spearman_score_vs_mkt"], (0.25, 0.5, 0.75))
                                       for k, g in k26.groupby(k26["score_day_kako5_missing"].astype(str))}
    c["missing_rate_rows"] = {col: round(float(sub[col].isna().mean()), 5) for col in
                              ["odds_tan", "odds_fuku_lo", "q_fuku_mkt", "score", "p_win_sm", "p_top3_pl", "p_place_model",
                               "venue", "surface", "distance", "class_norm", "s1_min_before_post", "res_payout_100",
                               "res_close_tan"]}
    c["missing_rate_by_type"] = {bt: {col: round(float(g[col].isna().mean()), 5) for col in
                                      ["odds_tan", "odds_fuku_mid", "p_bet", "q_bet", "res_payout_100"]}
                                 for bt, g in sub.groupby("bet_type", observed=True)}
    # 取消・中止
    c["scratch_dnf_by_year"] = kept.groupby("year").agg(
        races=("rid16", "size"), scratch_before_s1=("n_scratch_before_s1", "sum"),
        scratch_after_s1=("n_scratch_after_s1", "sum"), races_scratch_after_s1=("n_scratch_after_s1", lambda s: int((s > 0).sum())),
        dnf=("n_dnf", "sum"), races_dnf=("n_dnf", lambda s: int((s > 0).sum())),
        kekka_o1_inconsistent=("n_kekka_o1_inconsistent", "sum"), kekka_row_missing=("n_kekka_row_missing", "sum"),
        kekka_race_missing=("kekka_found", lambda s: int((~s.astype(bool)).sum()))).astype(int).to_dict(orient="index")
    c["res_status_by_type"] = {bt: g["res_status"].value_counts().loc[lambda s: s > 0].astype(int).to_dict()
                               for bt, g in sub.groupby("bet_type", observed=True)}
    c["res_status_by_year_type"] = _sd(sub.groupby(["year", "bet_type", "res_status"], observed=True).size())
    c["res_payout_src_by_type"] = {bt: g["res_payout_src"].value_counts().loc[lambda s: s > 0].astype(int).to_dict()
                                   for bt, g in sub.groupby("bet_type", observed=True)}
    # 払戻と確定オッズ（S3）
    nd = kept[~kept["deadheat_win"]]
    c["chk_tan_eq_by_year"], c["chk_fuku_in_range_by_year"] = {}, {}
    for y, g in nd.groupby("year"):
        s = g["chk_tan_eq"].dropna().astype(float)
        c["chk_tan_eq_by_year"][int(y)] = [round(float(s.mean()), 5) if len(s) else None, int(len(s))]
    for y, g in kept.groupby("year"):
        nn = g["chk_fuku_n"].fillna(0)
        c["chk_fuku_in_range_by_year"][int(y)] = ([round(float((g["chk_fuku_in_range"].fillna(0) * nn).sum() / nn.sum()), 5),
                                                  int(nn.sum())] if nn.sum() > 0 else None)
    ne = nd[nd["chk_tan_eq"].astype(float) == 0.0]
    c["chk_tan_mismatch_dates"] = ne.groupby("date").agg(n=("rid16", "size"),
                                                        ratio_med=("chk_tan_ratio", "median")).reset_index().to_dict(orient="records")
    fne = kept[kept["chk_fuku_in_range"] < 1]
    c["chk_fuku_out_dates"] = fne.groupby("date").size().astype(int).to_dict()
    c["deadheat_win_races_by_year"] = kept.groupby("year")["deadheat_win"].sum().astype(int).to_dict()
    c["deadheat_place_races_by_year"] = kept.groupby("year")["deadheat_place"].sum().astype(int).to_dict()
    dp = kept[kept["deadheat_place"]]
    c["deadheat_place_fuku_paid"] = _sd(dp.groupby(["fuku_places", "n_fuku_paid"]).size())
    c["deadheat_tan_ratio"] = _q(kept["chk_tan_deadheat_ratio_mean"], (0.0, 0.5, 1.0))
    c["chk_kekka_v2_eq_top3file_by_year"] = kept.groupby("year")["chk_kekka_v2_eq_top3file"].mean().round(5).dropna().to_dict()
    c["n_fuku_paid_vs_places"] = _sd(kept.groupby(["fuku_places", "n_fuku_paid"]).size())
    # 文脈
    c["class_norm_counts"] = kept["class_norm"].value_counts().to_dict()
    c["surface_counts"] = kept["surface"].value_counts().to_dict()
    c["context_missing_rate"] = {col: round(float(kept[col].isna().mean()), 5) for col in
                                 ["venue", "surface", "distance", "class_norm", "post_time"]}
    return c


# =============================================================== report
def _tbl(d: dict, cols: list, idx_name="年") -> str:
    lines = [f"| {idx_name} | " + " | ".join(str(x) for x in cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for k, v in d.items():
        lines.append(f"| {k} | " + " | ".join(str(v.get(x, "")) if isinstance(v, dict) else "" for x in cols) + " |")
    return "\n".join(lines)


def _qrow(d: dict, keys) -> str:
    return " / ".join(str(d.get(k, "")) for k in keys)


def write_report(c: dict, out_sha: str) -> None:
    bi = c["build_info"]
    L = []
    L.append("# 馬券判断 AI v2 substrate 検査レポート（label-free）\n")
    L.append(f"> 自動生成: `python -m analysis.bet_decision_ai.v2.build_substrate_v2`（{datetime.now():%Y-%m-%d %H:%M}）。"
             "仕様 `docs/research/BET_DECISION_AI_V2_SPEC.md` v0.1 §2-§4。\n"
             "> **方策の成績（回収率・的中率・ROI・損益の集計）は一切計算していない。** 決済列 res_* は作ったが集計していない"
             "（状態の件数と、確定オッズとの一致率だけを記録）。数値の正本は `out/substrate_v2_checks.json`。\n")
    L.append("## 0. 出力\n")
    L.append(f"- `data/_research/bet_decision_ai/v2/substrate_v2.parquet`: **{c['n_rows']:,} 行 / {c['n_races']:,} レース**"
             f"（sha256 `{out_sha[:16]}…`、data/_research は .gitignore 済み）")
    L.append("- `data/_research/bet_decision_ai/v2/substrate_v2_races.parquet`: レース単位（除外理由・時刻・照合列）")
    L.append("- `data/_research/bet_decision_ai/v2/c1_2026_scores.parquet` / `c1_2026_manifest.json` / `c1_2026/`: 2026 C1 作り直し"
             "（`python -m analysis.bet_decision_ai.v2.rescore_c1_2026`）")
    L.append(f"- 構築時間 {bi['elapsed_sec']} 秒。O1 原本 sha256 `{bi['o1_sha256'][:16]}…`（EXP22・v1 と同一）\n")

    L.append("## 1. 行・列\n")
    L.append("行 = レース × S1 断面で単勝オッズ（数値）がある馬 × {tan, fuku}。複勝は S1 で複勝が発売されているレースだけ"
             "（全レース発売。§3）。\n")
    L.append("| 列 | 意味 |\n|---|---|")
    coldoc = [
        ("rid16, year, date, period_role", "レースキー・年・日付。period_role は仕様 §6（train 2021-22 / select 2023 / confirm 2024-25 / holdout_2026）"),
        ("venue_code, venue, surface, distance, class_raw, class_norm, track_code, post_time, meta_src",
         "レース文脈。2021-25 は master_v2、2026 は出走表 CSV の見出し行（track_code は 2026 で欠損）。class_norm は v1 と同じ正規化"),
        ("n_registered, n_field_s1, n_starters_hdr_s1", "O1 の登録頭数 / **判断時点の頭数**（S1 で単勝が数値の馬の数）/ S1 ヘッダの出走頭数"),
        ("umaban, waku, bet_type", "馬番・枠番（登録頭数から決まる。判断時点で既知）・券種 tan/fuku"),
        ("s1_announced, s1_min_before_post, s1_rule, n_o1_records, o1_divisions",
         "S1 の発表時刻（YYYYMMDDHHMM）、発走何分前か、S1 の選び方、O1 録数、録の区分列"),
        ("odds_tan, odds_fuku_lo/hi/mid, fuku_places, fuku_offered", "S1 の単勝・複勝 Lo/Hi・中点、複勝払戻着数（O1 複勝着払キー）、複勝発売フラグ"),
        ("overround_tan, overround_fuku_mid/lo/hi", "S1 の Σ1/odds（複勝は Σ1/中点・Lo・Hi を払戻着数で割る）"),
        ("votes_tan_s1, votes_fuku_s1", "S1 時点の総票数（O1）"),
        ("q_tan_mkt, q_fuku_mkt, tan_odds_rank", "市場確率（単勝 1/odds を和 1 に、複勝 1/中点を和=払戻着数に正規化）、単勝オッズ順位（1=1 番人気、同値は min）"),
        ("mkt_entropy_tan, mkt_q_top1, mkt_q_gap12", "市場確率の正規化エントロピー、1 位の確率、1 位 − 2 位"),
        ("score, score_missing, score_src, score_dup_n", "C1 生スコアと欠損フラグ。出所 oof_c1master（2021-23）/ v6_scores_c1（2024-25）/ c1_rescore_2026。"
         "score_dup_n = 2026 の作り直しで同じ馬に出たスコアの数（2 なら平均、§11）"),
        ("n_scored, p_win_sm, p_top2_pl, p_top3_pl, p_place_model, model_rank",
         "スコアのある S1 候補だけで softmax(score/τ)（τ: OOF 0.845・本番 0.865、v1 と同じ）→ λ 補正 PL（λ1=0.8405, λ2=0.7542、v1 と同じ式）の 2 着内・3 着内。"
         "p_place_model は払戻着数 3 なら 3 着内・2 なら 2 着内。model_rank はスコア順位（1=最上位）"),
        ("model_entropy, model_p_top1, model_p_gap12", "p_win_sm の正規化エントロピー・1 位・1 位 − 2 位"),
        ("p_bet, q_bet, log_p_over_q", "券種ごとの確率（tan=p_win_sm, fuku=p_place_model）と市場確率、その対数比（**較正前**）"),
        ("score_day_gate_fail, score_day_dead_gain_pct, score_day_kako5_missing", "2026 のみ: 作り直し時の本番品質ゲート不合格、無言死特徴の gain 割合、過去 5 走 CSV の有無（日単位）"),
        ("res_status, res_payout_100, res_payout_src", "**結果列**。win / lose / refund（取消・除外 100）/ dnf（競走中止など 0）/ unsettleable（NaN）。払戻は 100 円あたりの公式払戻"),
        ("res_fin, res_scratched_after_s1, res_dnf", "**結果列**。確定着順（0=完走せず）、S1 後に取消・除外、出走したが完走せず"),
        ("res_race_deadheat_win, res_race_deadheat_place, res_race_n_scratch_after_s1, res_race_n_dnf", "**結果列**（レース単位）"),
        ("res_close_tan, res_close_fuku_lo/hi, res_s3_min_after_post", "**結果列**。O1 S3（確定オッズ、発走後）。入力禁止"),
    ]
    for a, b in coldoc:
        L.append(f"| {a} | {b} |")
    L.append("\n**学習の入力にしてはいけない列**: `res_*` すべて（払戻・状態・着順・取消/中止・確定オッズ）。"
             "加えて `score_missing` は 2021-25 では結果依存（§5）。\n")

    L.append("## 2. 年別の被覆\n")
    L.append("| 年 | O1 レース | 平地で採用 | 開催日 | 単勝行 | 複勝行 | 頭数 p00/p50/p100 | 除外内訳 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for y in sorted(c["races_o1_by_year"]):
        ex = c["excluded_by_year_reason"].get(y, {})
        exs = ", ".join(f"{k}={v}" for k, v in ex.items() if v)
        rt = c["rows_by_year_type"].get(y, {})
        cp = c["cands_per_race_by_year"].get(y, {})
        L.append(f"| {y} | {c['races_o1_by_year'][y]} | {c['races_kept_by_year'].get(y, 0)} | {c['dates_by_year'].get(y, 0)} | "
                 f"{rt.get('tan', 0):,} | {rt.get('fuku', 0):,} | {cp.get('p00')}/{cp.get('p50')}/{cp.get('p100')} | {exs} |")
    L.append("")
    L.append(f"- 2026 の出走表 CSV が無い O1 開催日: {bi['weekly_missing_dates'] or 'なし'}")
    L.append(f"- 障害の既知欠落（manifest）: {json.dumps(bi.get('jump_manifest_gap') or {}, ensure_ascii=False)[:300]}\n")

    L.append("## 3. 判断時点（S1）の時刻\n")
    L.append(f"- S1 の選び方（採用レース）: {c['s1_rule_counts']}。全 O1 レース: {c['s1_rule_all_races']}")
    L.append(f"- 異常の件数: {json.dumps({k: v for k, v in c['s1_flags'].items() if k != 'races_lt_5_or_gt_20'}, ensure_ascii=False)}")
    if c["s1_flags"]["races_lt_5_or_gt_20"]:
        L.append(f"- 5 分未満・20 分超の S1: {c['s1_flags']['races_lt_5_or_gt_20']}")
    L.append("\n発走から見た S1 の時刻（分前、年別 p00 / p01 / p05 / p10 / p50 / p90 / p95 / p99 / p100）:\n")
    L.append("| 年 | 分前 | n |\n|---|---|---|")
    ks = ["p00", "p01", "p05", "p10", "p50", "p90", "p95", "p99", "p100"]
    for y, d in c["s1_min_before_post_by_year"].items():
        L.append(f"| {y} | {_qrow(d, ks)} | {d.get('n')} |")
    L.append(f"\nS1 分前の度数: {c['s1_min_before_post_value_counts']}\n")
    L.append("S1 → S3（確定）の間隔（分、p00/p01/p50/p99/p100）: " + "; ".join(
        f"{y}: {_qrow(d, ['p00', 'p01', 'p50', 'p99', 'p100'])}" for y, d in c["s1_to_s3_min_by_year"].items()))
    L.append(f"\n予定発走時刻から見て S1 が 5〜20 分前に入らないレース（s1_timing_ok=False）: {c['s1_timing_not_ok']}\n")
    L.append("参考: S0（p01/p50/p99 分前）" + "; ".join(f"{y}: {_qrow(d, ['p01', 'p50', 'p99'])}" for y, d in c["s0_min_before_post_by_year"].items()))
    L.append("\n参考: S3＝確定（発走後 p01/p50/p99 分）" + "; ".join(f"{y}: {_qrow(d, ['p01', 'p50', 'p99'])}" for y, d in c["s3_min_after_post_by_year"].items()) + "\n")

    L.append("## 4. 価格の分布（S1）\n")
    L.append("| 年 | 単勝 Σ1/odds p01/p05/p50/p95/p99 | 複勝 Σ1/中点÷着数 p01/p05/p50/p95/p99 | 複勝 Lo 版 p05/p50/p95 | 複勝 Hi 版 p05/p50/p95 | 単勝 Σ<1.23 の率 |")
    L.append("|---|---|---|---|---|---|")
    for y in c["overround_tan_by_year"]:
        L.append(f"| {y} | {_qrow(c['overround_tan_by_year'][y], ['p01', 'p05', 'p50', 'p95', 'p99'])} | "
                 f"{_qrow(c['overround_fuku_mid_by_year'].get(y, {}), ['p01', 'p05', 'p50', 'p95', 'p99'])} | "
                 f"{_qrow(c['overround_fuku_lo_by_year'].get(y, {}), ['p05', 'p50', 'p95'])} | "
                 f"{_qrow(c['overround_fuku_hi_by_year'].get(y, {}), ['p05', 'p50', 'p95'])} | "
                 f"{c['overround_tan_lt_1_23_rate_by_year'].get(y)} |")
    L.append(f"\n- q_tan_mkt のレース内和: {c['q_tan_sum']}。q_fuku_mkt の和 − 払戻着数: {c['q_fuku_sum_minus_places']}")
    L.append(f"- 複勝払戻着数（S1 の着払キー）: {c['fuku_places_s1']}。（頭数≥8, 着数）別: {c['fuku_places_s1_vs_n_field']}。"
             f"S1 と S3 で着数が違うレース: {c['fuku_places_s1_ne_s3']}。複勝非発売レース: {c['fuku_offered_false_races']}")
    L.append(f"- 実際に複勝が払い戻された頭数（着数, 払戻頭数）: {c['n_fuku_paid_vs_places']}\n")

    L.append("## 5. モデル確率\n")
    L.append(f"- 温度 τ: {bi['score_info']['tau']}。スコア出所の行数: {bi['score_info']['rows_by_src']}。2026 の重複採点: {bi['score_info']['c1_2026_dup_horses']} 頭（日: {bi['score_info']['c1_2026_dup_dates']}、2 つのスコアの差 {bi['score_info']['c1_2026_dup_range']}）→ 平均を採用、score_dup_n に記録")
    L.append("\n| 年 | 出所（単勝+複勝 行） | スコア欠損率（単勝行） | 全馬欠損レース | 欠損の原因（単勝行） | score 平均/標準偏差 | p_win エントロピー p05/p50/p95 | ◎ p_win p05/p50/p95 | レース内順位相関（スコア vs 市場）p25/p50/p75 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for y in sorted(c["score_missing_rate_by_year"]):
        sd = c["score_dist_by_year"].get(y, {})
        L.append(f"| {y} | {c['score_src_rows_by_year'].get(y, {})} | {c['score_missing_rate_by_year'][y]} | "
                 f"{c['races_all_scores_missing_by_year'].get(y, 0)} | {c['score_missing_cause_by_year'].get(y, {})} | "
                 f"{sd.get('mean')}/{sd.get('std')} | {_qrow(c['model_entropy_by_year'].get(y, {}), ['p05', 'p50', 'p95'])} | "
                 f"{_qrow(c['model_p_top1_by_year'].get(y, {}), ['p05', 'p50', 'p95'])} | "
                 f"{_qrow(c['spearman_score_vs_mkt_by_year'].get(y, {}), ['p25', 'p50', 'p75'])} |")
    L.append(f"\n- 確率の和（スコアのある候補内）: p_win {c['psum_win']}、3 着内 − 3: {c['psum_top3_minus3']}、2 着内 − 2: {c['psum_top2_minus2']}")
    if "spearman_2026_by_gate" in c:
        L.append(f"- 2026 の日単位品質ゲート別レース数: {c['races_2026_by_gate']}。順位相関: {c['spearman_2026_by_gate']}。"
                 f"過去 5 走 CSV 欠落日別: {c['spearman_2026_by_kako5']}")
    L.append("")

    L.append("## 6. 取消・除外・中止\n")
    L.append("| 年 | 採用R | S1 前の取消（頭） | **S1 後の取消・除外（頭 / R）** | **競走中止など（頭 / R）** | kekka と O1 S3 の食い違い | kekka 行欠落 | kekka レース欠落 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for y, d in c["scratch_dnf_by_year"].items():
        L.append(f"| {y} | {d['races']} | {d['scratch_before_s1']} | {d['scratch_after_s1']} / {d['races_scratch_after_s1']} | "
                 f"{d['dnf']} / {d['races_dnf']} | {d['kekka_o1_inconsistent']} | {d['kekka_row_missing']} | {d['kekka_race_missing']} |")
    L.append("\n決済状態（行数。回収率は計算しない）:\n")
    L.append("| 券種 | res_status | res_payout_src |\n|---|---|---|")
    for bt in c["res_status_by_type"]:
        L.append(f"| {bt} | {c['res_status_by_type'][bt]} | {c['res_payout_src_by_type'][bt]} |")
    L.append("")

    L.append("## 7. 公式払戻と確定オッズ（O1 S3）の一致\n")
    L.append("単勝: 1 着同着でないレースの勝ち馬で round(S3 単勝×100) == 公式払戻。複勝: 公式払戻 ∈ [S3 Lo, Hi]×100（±0.5）。\n")
    L.append("| 年 | 単勝一致率 (n) | 複勝 範囲内率 (n) | 1 着同着R | 複勝圏同着R | kekka 全頭版 = 1-3 着版 |")
    L.append("|---|---|---|---|---|---|")
    for y in sorted(c["chk_tan_eq_by_year"]):
        L.append(f"| {y} | {c['chk_tan_eq_by_year'][y]} | {c['chk_fuku_in_range_by_year'].get(y)} | "
                 f"{c['deadheat_win_races_by_year'].get(y)} | {c['deadheat_place_races_by_year'].get(y)} | "
                 f"{c['chk_kekka_v2_eq_top3file_by_year'].get(y, '—')} |")
    L.append(f"\n- 単勝が一致しない日（同着除く、件数と 払戻÷確定オッズ の中央値）: {c['chk_tan_mismatch_dates']}")
    L.append(f"- 複勝が範囲外の日（複勝圏同着を除いたレース数）: {c['chk_fuku_out_dates'] or 'なし'}")
    L.append(f"- 複勝圏同着レースの（着数, 払戻頭数）: {c['deadheat_place_fuku_paid']}（3 着同着で 4 頭に払戻。払戻は確定オッズ範囲より"
             "低くなるので照合から外した。決済は公式払戻）")
    L.append(f"- 1 着同着の単勝: 払戻 ÷ (S3 オッズ×100) のレース平均 {c['deadheat_tan_ratio']}（同着は払戻が分かれるので確定オッズと一致しない。決済は公式払戻）\n")

    L.append("## 8. 欠損率（行単位）\n")
    L.append(f"{c['missing_rate_rows']}\n\n券種別: {c['missing_rate_by_type']}\n")
    L.append(f"- 文脈列の欠損率（採用レース）: {c['context_missing_rate']}")
    L.append(f"- クラス: {c['class_norm_counts']}\n- 芝ダ: {c['surface_counts']}\n")

    L.append("## 9. 2026 の確率の作り直し\n")
    man = bi.get("c1_2026_manifest", {})
    days = bi.get("c1_2026_days", {})
    L.append(f"- モデル sha256 `{str(man.get('model_sha256'))[:16]}…`（{man.get('pins_generation')}）、"
             f"設定 {man.get('env')}、馬履歴を開催日前に切り落とし={man.get('horse_history_truncated_lt_date')}")
    L.append(f"- 対象 {man.get('n_days_o1')} 日 / 作り直したスコア {man.get('n_rows')} 行 / {man.get('n_races')} レース")
    if days:
        L.append("\n| 日 | rc | 品質ゲート不合格 | 無言死 gain% | 過去5走CSV欠落 | 採点R | 履歴解決 |\n|---|---|---|---|---|---|---|")
        for d, v in sorted(days.items()):
            L.append(f"| {d} | {v.get('rc')} | {v.get('gate_fail')} | {v.get('dead_gain_pct')} | {v.get('kako5_missing')} | "
                     f"{v.get('n_race_json')} | {v.get('history_resolved')} |")
    L.append("")
    L.extend(DECISIONS_MD)
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")


DECISIONS_MD = """## 10. 仕様の曖昧点で判断したこと

1. **S1 の定義**: O1 の録を発表時刻順に並べた 2 番目（EXP22 の S0/S1/S2/S3 と同じ。4 録のレースは全て 2 番目が区分 1）。
   2 録以上で 2 番目が区分 1 でなければ S1 無し。1 録だけのレース（2026 の不完全レース）は、その録が区分 1 で予定発走の
   5〜20 分前のときだけ S1 とする（該当 0、5R は S1 無しで除外）。予定発走時刻から見て 5〜20 分前に入らない S1 は
   `s1_timing_ok=False` で残した（§3。発走時刻が変更された日と推定。S1→S3 の間隔は他のレースと同じ）。
2. **候補**: S1 で単勝が数値の馬（S1 前の取消は候補外）。複勝は S1 で複勝発売フラグが立つレース（全レース）の同じ馬。
   複勝の Lo/Hi が数値でない馬も行は残し odds_fuku_* = NaN。
3. **頭数**: `n_field_s1` = S1 で単勝が数値の馬の数（判断時点の頭数）。複勝の払戻着数は O1 の複勝着払キー（S1）を使った
   （S1 と S3 で違うレース 0）。
4. **障害除外**: 2021-25 は master_v2 のトラックコード 51-59（v1 と同じ）。2026 は出走表 CSV の芝・ダートが「障害…」の
   レースと、**出走表 CSV に無い O1 レース**（data/weekly は 2026-03-07 以降 障害を含まない。
   `data/history_only/jump/manifest.json` の known_unrecoverable_gap）を障害とみなして除外した。後者は推定で、件数を §2 に記録。
5. **モデル確率**: 生スコアを保存し、確率は「S1 候補のうちスコアのある馬」だけで softmax(score/τ)（τ は base.parquet の値:
   OOF 0.845・本番 0.865。2026 の作り直しは本番モデルなので 0.865）。2 着内・3 着内は v1 と同じ λ 補正 PL
   （`build_substrate.pl_tensors` を import）。**年をまたぐ尺度合わせ（過去だけで較正）はしていない**（仕様どおり学習段で行う）。
   2023 は OOF（本番 v6 は 2023 を early stop と較正に使っているため。base.parquet と同じ）。
6. **決済**: 取消・除外（S1 で単勝が数値、S3＝確定で数値でない馬）= 返還 100。出走したが着順 0（競走中止・失格など）= 0。
   的中は公式払戻（2021-25 `kekka_20160105_20251228_v2.csv` 全頭版、2026 `data/kekka/{date}.csv`）。同着は公式払戻をそのまま
   （単勝 1 着同着は両馬とも分割後の払戻、複勝 3 着同着は 4 頭とも払戻）。上乗せの日は無かった（単勝は全レース一致）。
   S3 が無いレース（2026 の不完全 3R）は kekka の着順と単勝欄（数値・括弧オッズ＝出走、空欄＝取消）で出走を判定した。
7. **2026 の重複採点**: 2026-03-07〜04-12 の作り直しでは、補正タイム表（data/hosei の H_*.csv を全部結合する
   predict_weekly._load_hosei）に同じ馬で値の違う行が残り、1 頭が 2 行になって 2 つのスコアが出た（本番の当時の配信でも同じ結合）。
   どちらが正しいかは決められないので平均を採り、`score_dup_n` に残した。
8. **2026 の確率**: §9 と §11。作り直したが、過去 5 走 CSV の無い日は本番の品質ゲート不合格（重要特徴が欠損）で、
   日単位のフラグ列を付けて残した（行は落としていない）。
9. **period_role** は仕様 §6 の予定（2021-22 学習 / 2023 選択 / 2024-25 確認 / 2026 追試）をそのまま列にした。

## 11. 異常・注意（学習段で扱うこと）

1. **2021-25 のスコア欠損は結果で決まっている（先読み）**: master_v2 に完走しなかった馬の行が無いので、OOF・v6_scores にも
   スコアが無い。2021-25 で score_missing の馬は §5 のとおりほぼ全てが「S1 後に取消・除外」か「競走中止」。
   欠損そのもの（NaN の有無、p_win_sm の正規化母集団の大きさ）が結果を示すので、**score_missing を入力にしない・
   NaN のまま学習器に渡さない**こと（結果に依存しない規則で埋めるか、PyCaLiAI 版と確率なし版の両方から該当行を外した
   感度分析を併記する）。p_win_sm もこれらの馬を除いて正規化しているので、該当レースの他馬の確率がわずかに高い。
   2026 の作り直しは出走表の全馬を採点するので、この先読みは無い（欠損の意味が年で違う）。
2. **2026 のスコアは配信経路（出走表 CSV → export_weekly_marks の特徴生成）、2021-25 はオフライン経路（master_v2）**。
   同じ C1 モデルでも特徴の欠け方が違う。過去 5 走 CSV（data/kako5）が無い日は kako5_* 特徴 16 個が全欠損（本番ゲート不合格）。
   スコアと市場の順位相関（§5）で、2026 は 2021-25 より低い。2026 を追試に使うなら日単位のフラグで層別すること。
3. 補助ファイル（騎手・調教師コード表、level_norms、補正タイム推定器など）は現在版（replay_serve_2026 と同じ限界）。
   補正タイム表も data/hosei の全ファイル（9 月に書き出した H_20260104-20260913.csv を含む）を結合するので、
   前走の補正値は開催日の朝より後に書き出された版を含む（その日のレース自身の結果ではない）。
   馬履歴は開催日前に切り落として as-of にした。
4. **単勝 Σ1/odds < 1.23** のレースが 2021-23 で年 16%、2024-25 で 1-2%（v1 の T−28 と同じ傾向。新馬・未勝利に偏る。
   原因は未確定）。q_*_mkt はプール内正規化なので和は保たれる。
5. 複勝の市場確率は 1/中点の正規化で粗い。odds_fuku_lo / hi を併用すること。
6. 2026 は O1 原本に 53 開催日しか無い（出走表は同期間 69 日）。O1 に無い日はそもそも行が無い。
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    sub, races, info = build()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sub.to_parquet(OUT, index=False)
    races.to_parquet(RACES_OUT, index=False)
    out_sha = sha256(OUT)
    log(f"saved {len(sub):,} rows -> {OUT}")
    c = checks(sub, races, info)
    c["output_sha256"] = out_sha
    CHECKS_OUT.parent.mkdir(parents=True, exist_ok=True)
    CHECKS_OUT.write_text(json.dumps(c, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
                          encoding="utf-8")
    write_report(c, out_sha)
    log(f"checks -> {CHECKS_OUT}\nreport -> {REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
