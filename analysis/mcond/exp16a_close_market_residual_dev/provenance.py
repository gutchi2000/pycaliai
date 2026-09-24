# -*- coding: utf-8 -*-
"""
provenance.py — EXP16A Stage 0: 単勝市場データの素性実測 + dry run
==================================================================
学習しない。2024/2025 は読み込み時に捨てる。ROI 評価も候補生成もしない。
測るもの:
  - historical_pre_snapshot / final(確定, 区分4) の取得時刻と **実測 post との差**
  - 欠損・取消(出走表に無い馬番)・返還相当の扱い
  - race 内 de-vig (比例) と overround
  - ファイル sha256
出力: out/market_provenance.json (STAGE0_DRY_RUN.json は race_population.py が作る)
実行: python -m analysis.mcond.exp16a_close_market_residual_dev.provenance
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[2]
OUT = HERE / "out"
MASTER = BASE / "data" / "master_v2_20130105-20251228.csv"
TANPUK_DIR = BASE / "data" / "Time _series_odds"
D_MIN, D_MAX = 20160101, 20231231
PERIODS = {"train": (20160101, 20211231), "selection": (20220101, 20221231),
           "development": (20230101, 20231231)}
MIN_GAP_PRE = 15  # analysis/mcond/market.py と同じ契約 (確定の 15 分以上前)
# JRA-VAN トラックコード 51..59 = 障害。production の P0 hard gate (race_eligibility.py:48-49) と同一定義
JUMP_MIN, JUMP_MAX = 51, 59


def sha256(p: Path, limit_mb: int = 400) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_master():
    cols = ["日付", "レースID(新/馬番無)", "馬番", "着順", "発走時刻", "場所", "芝・ダ",
            "出走頭数", "クラス名", "トラックコード(JV)"]
    parts = []
    for ch in pd.read_csv(MASTER, encoding="utf-8-sig", dtype=str, usecols=cols, chunksize=200_000):
        d = pd.to_numeric(ch["日付"], errors="coerce")
        parts.append(ch[(d >= D_MIN) & (d <= D_MAX)])
    df = pd.concat(parts, ignore_index=True)
    df["date"] = pd.to_numeric(df["日付"], errors="coerce").astype(int)
    assert df["date"].max() <= D_MAX and df["date"].min() >= D_MIN
    df["rid16"] = df["レースID(新/馬番無)"].astype(str).str.replace(r"\.0$", "", regex=True).str[:16]
    df["ban"] = pd.to_numeric(df["馬番"], errors="coerce").astype(int)
    df["jyun"] = pd.to_numeric(df["着順"], errors="coerce")
    df = df.dropna(subset=["jyun"]).copy()
    df["win"] = (df["jyun"] == 1).astype(int)
    hm = df["発走時刻"].astype(str).str.extract(r"(\d{1,2}):(\d{2})")
    df["post_min"] = pd.to_numeric(hm[0], errors="coerce") * 60 + pd.to_numeric(hm[1], errors="coerce")
    df["year"] = df["date"] // 10000
    tc = pd.to_numeric(df["トラックコード(JV)"], errors="coerce")
    df["track_code"] = tc
    df["is_jump"] = ((tc >= JUMP_MIN) & (tc <= JUMP_MAX)).fillna(False)
    df["period"] = None
    for k, (a, b) in PERIODS.items():
        df.loc[(df["date"] >= a) & (df["date"] <= b), "period"] = k
    return df


def load_tanpuk(rids: set[str]):
    """2016-2023 の単勝オッズ時系列。rid16 が master に無い行は捨てる。"""
    rows = []
    for p in sorted(TANPUK_DIR.glob("TANPUK_*.csv")):
        for ch in pd.read_csv(p, encoding="cp932", dtype=str, chunksize=200_000, low_memory=False):
            ch["rid16"] = ch["レースID"].astype(str).str[:16]
            ch = ch[ch["rid16"].isin(rids)]
            if len(ch):
                rows.append(ch)
    t = pd.concat(rows, ignore_index=True)
    t["kubun"] = pd.to_numeric(t["区分"], errors="coerce")
    md = t["月日時分"].astype(str).str.zfill(8)
    t["mmdd"] = md.str[:4]
    t["snap_min"] = pd.to_numeric(md.str[4:6], errors="coerce") * 60 + pd.to_numeric(md.str[6:8], errors="coerce")
    t["votes_tan"] = pd.to_numeric(t["単勝票数"], errors="coerce")
    return t


def odds_matrix(row: pd.Series) -> dict[int, float]:
    out = {}
    for n in range(1, 19):
        v = pd.to_numeric(row.get(f"{n}単"), errors="coerce")
        if pd.notna(v) and v > 1.0:   # market.py と同じ: <=1.0 は欠損扱い
            out[n] = float(v)
    return out


def devig(od: dict[int, float], starters: list[int]) -> tuple[dict[int, float], float, int]:
    """race 内比例 de-vig。出走表にある馬だけで正規化。戻り: (pi, overround, 欠損馬数)"""
    inv = {b: 1.0 / od[b] for b in starters if b in od}
    miss = len(starters) - len(inv)
    s = sum(inv.values())
    if s <= 0:
        return {}, float("nan"), miss
    return {b: v / s for b, v in inv.items()}, s, miss


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m = load_master()
    rids = set(m["rid16"])
    t = load_tanpuk(rids)
    print(f"master rows {len(m):,} races {len(rids):,} / tanpuk rows {len(t):,}", flush=True)

    # レースごとの starters・post・メタ
    meta = (m.groupby("rid16")
              .agg(date=("date", "first"), year=("year", "first"), period=("period", "first"),
                   post_min=("post_min", "first"), venue=("場所", "first"), surface=("芝・ダ", "first"),
                   cls=("クラス名", "first"), n_start=("ban", "size")))
    starters = m.groupby("rid16")["ban"].apply(list)
    winner = m[m["win"] == 1].groupby("rid16")["ban"].apply(list)

    recs = []
    for rid, g in t.groupby("rid16", sort=False):
        if rid not in meta.index:
            continue
        mrow = meta.loc[rid]
        st = starters[rid]
        mmdd = f"{(mrow.date % 10000):04d}"
        fin = g[g["kubun"] == 4]
        pre_c = g[(g["kubun"] == 1) & (g["mmdd"] == mmdd)]
        rec = {"rid16": rid, "date": int(mrow.date), "year": int(mrow.year), "period": mrow.period,
               "venue": mrow.venue, "surface": mrow.surface, "cls": mrow.cls,
               "n_start": int(mrow.n_start), "n_snaps_kubun1_sameday": int(len(pre_c)),
               "has_final": bool(len(fin))}
        if len(fin):
            f0 = fin.iloc[-1]
            rec["final_snap_min"] = float(f0.snap_min)
            rec["final_minus_post_min"] = float(f0.snap_min - mrow.post_min)
            od = odds_matrix(f0)
            pi, ov, miss = devig(od, st)
            rec.update({"final_overround": ov, "final_missing_horses": miss,
                        "final_entries_not_in_master": len([b for b in od if b not in st])})
            rec["final_pi"] = pi
        if len(pre_c) and len(fin):
            gap = fin.iloc[-1].snap_min - pre_c["snap_min"]
            ok = pre_c[gap >= MIN_GAP_PRE]
            if len(ok):
                p0 = ok.iloc[-1]
                rec["pre_snap_min"] = float(p0.snap_min)
                rec["pre_minus_post_min"] = float(p0.snap_min - mrow.post_min)
                rec["pre_gap_to_final_min"] = float(fin.iloc[-1].snap_min - p0.snap_min)
                rec["pre_votes_tan"] = float(p0.votes_tan) if pd.notna(p0.votes_tan) else None
                od = odds_matrix(p0)
                pi, ov, miss = devig(od, st)
                rec.update({"pre_overround": ov, "pre_missing_horses": miss,
                            "pre_entries_not_in_master": len([b for b in od if b not in st])})
                rec["pre_pi"] = pi
        recs.append(rec)
    R = pd.DataFrame(recs)
    R["winner"] = R["rid16"].map(lambda r: winner.get(r, [None])[0] if r in winner.index else None)
    R["dead_heat"] = R["rid16"].map(lambda r: len(winner.get(r, [])) > 1 if r in winner.index else False)

    # ---- 年別 provenance
    def q(s, qs=(0.01, 0.5, 0.99)):
        s = pd.to_numeric(s, errors="coerce").dropna()
        return {f"p{int(x*100)}": float(s.quantile(x)) for x in qs} | {"mean": float(s.mean())} if len(s) else None

    per_year = {}
    for y, g in R.groupby("year"):
        per_year[int(y)] = {
            "races_in_master": int((meta["year"] == y).sum()),
            "races_with_tanpuk": int(len(g)),
            "races_with_final": int(g["has_final"].sum()),
            "races_with_pre": int(g["pre_snap_min"].notna().sum()) if "pre_snap_min" in g else 0,
            "pre_minutes_before_post": q(-g.get("pre_minus_post_min", pd.Series(dtype=float))),
            "final_minutes_after_post": q(g.get("final_minus_post_min", pd.Series(dtype=float))),
            "pre_gap_to_final_min": q(g.get("pre_gap_to_final_min", pd.Series(dtype=float))),
            "pre_overround": q(g.get("pre_overround", pd.Series(dtype=float))),
            "final_overround": q(g.get("final_overround", pd.Series(dtype=float))),
            "races_with_missing_horse_in_pre": int((g.get("pre_missing_horses", pd.Series(0)) > 0).sum()),
            "races_with_missing_horse_in_final": int((g.get("final_missing_horses", pd.Series(0)) > 0).sum()),
            "entries_not_in_master_pre_total": int(pd.to_numeric(g.get("pre_entries_not_in_master", pd.Series(0))).sum()),
            "dead_heat_races": int(g["dead_heat"].sum()),
        }

    prov = {
        "scope": "2016-01-01..2023-12-31 のみ (2024/2025 は読み込み時に破棄)",
        "files": {p.name: {"sha256": sha256(p), "size_bytes": p.stat().st_size, "encoding": "cp932"}
                  for p in sorted(TANPUK_DIR.glob("TANPUK_*.csv"))},
        "master_sha256_note": "master_v2 は 519MB のため hash は取らない (行 hash は EXP15 out/feature_contract.json)",
        "definitions": {
            "historical_pre_snapshot": "区分1 (途中) のうち、レース当日かつ 区分4 の 15 分以上前の最後の1本 "
                                        "(analysis/mcond/market.py と同契約)。T-10 とは呼ばない",
            "final_confirmed": "区分4 (確定)。記録時刻は post の後",
            "close_odds": "JRA は pari-mutuel で post 時に締切。手元データには『締切時点の板』そのものの断面は無く、"
                          "最も近いのは 区分4 (確定) である。close と final を別物として区別できるデータは無い",
            "result_file_odds": "kekka 系ファイル内のオッズ列 (単勝配当の括弧など)。**時刻が無く素性不明**のため "
                                "Q0/Q3 のどちらにも使わない。照合用のみ",
            "devig": "race 内比例正規化 pi = (1/o) / Σ(1/o)。出走表 (master, 着順が数値の馬) に居る馬だけで正規化",
            "missing_rule": "オッズ <= 1.0 は欠損扱い (market.py と同じ)",
            "scratch_rule": "オッズはあるが master に居ない馬番 = 取消/除外。de-vig の分母から外す",
        },
        "per_year": per_year,
    }
    (OUT / "market_provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1,
                                                           default=float), encoding="utf-8")

    # STAGE0_DRY_RUN.json は race_population.py が正式 race set 上で作る (ここでは書かない)
    print(json.dumps({k: v for k, v in per_year.items() if k in (2022, 2023)}, ensure_ascii=False)[:400])
    print("[saved] out/market_provenance.json")


if __name__ == "__main__":
    main()
