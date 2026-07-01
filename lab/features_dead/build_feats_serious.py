# -*- coding: utf-8 -*-
"""
build_feats_serious.py — 情報源が構造的に違う「本気の」新規特徴ファミリ (全て as-of/leak-safe)
==============================================================================================
前回の feats_1000 は kako5/hosei 背骨の焼き直し(積・比・順位化)で新情報ゼロ→死亡。
今回は「背骨に無い情報源」を狙って新ファミリを設計する:

  S1 スピード指数      : prev_hosei(補正タイム)の馬キャリア集計 + 距離/馬場バケット相対
  S2 ペース/脚質       : 前走コーナー通過から脚質を推定→レース内ペース投影→pace-fit
  S3 種牡馬/母父 as-of適性: 産駒の過去成績(芝ダ×距離帯)を厳密as-of(当日除外)で集計
  S4 騎手/厩舎コンテキスト率: 騎手×場所×芝ダ・厩舎・騎手厩舎コンボの過去率(当日除外)
  S5 レイオフ/間隔角度  : 馬の休み明け実績・間隔×キャリアの交互作用
  S6 フィールド競争構造 : レース内の戦力分散・トップとの差・エントロピー(背骨でなく集団構造)
  S7 クラス力学        : 生涯最高クラス・現級経験・初挑戦フラグ

★as-of 厳密性: 集計率(S3/S4)は「当日(レース日付)を丸ごと除外」して過去のみ。
  → [[project_v8_affinity_insample_leak]] の自レース込みleakを構造的に排除。
  course_affinity_*.parquet(全期間集計)は使わない。
  馬キャリア/コーナーは前走列 or 厳密過去のみ。

出力: data/feats_serious.parquet (rid16, ban + 特徴) / data/_feats_serious_manifest.json
実行: PYTHONUTF8=1 ./venv311/Scripts/python.exe build_feats_serious.py
"""
from __future__ import annotations
import json, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
BASE = Path(__file__).parent
MASTER_CSV = BASE / "data/master_v2_20130105-20251228.csv"
OUT_PARQUET = BASE / "data/feats_serious.parquet"
OUT_MANIFEST = BASE / "data/_feats_serious_manifest.json"

COL_RID = "レースID(新/馬番無)"; COL_BAN = "馬番"; COL_PED = "血統登録番号"
COL_DATE = "日付"; COL_JYUN = "着順"

GOING_MAP = {"良": 0, "稍": 1, "稍重": 1, "重": 2, "不": 3, "不良": 3}


def log(m): print(m, flush=True)
def _num(s): return pd.to_numeric(s, errors="coerce")
def _rid16(s): return s.astype(str).str.replace(r"\D", "", regex=True).str[:16]


def safe_div(a, b):
    b = b.replace(0, np.nan)
    return (a / b).replace([np.inf, -np.inf], np.nan)


def asof_group_rate(work, gcols, valcol):
    """グループ(gcols)ごとに valcol の率を「当日(_dd)を丸ごと除外」で as-of 集計。
    自レース・同日他レース込みの leak を構造的に排除。戻り: (rate, prior_count) を work 行順で。"""
    d = (work.groupby(gcols + ["_dd"], sort=False)[valcol]
              .agg(s="sum", c="count").reset_index())
    d = d.sort_values(gcols + ["_dd"], kind="mergesort")
    gg = d.groupby(gcols, sort=False)
    d["cs"] = gg["s"].cumsum() - d["s"]      # 当日より前の合計のみ
    d["cc"] = gg["c"].cumsum() - d["c"]
    d["rate"] = d["cs"] / d["cc"].replace(0, np.nan)
    m = work[gcols + ["_dd", "_rowid"]].merge(
        d[gcols + ["_dd", "rate", "cc"]], on=gcols + ["_dd"], how="left")
    m = m.sort_values("_rowid")
    return m["rate"].to_numpy(np.float32), m["cc"].to_numpy(np.float32)


def build():
    import os
    t0 = time.time()
    nrows = os.environ.get("SERIOUS_NROWS")
    nrows = int(nrows) if nrows else None
    log(f"[load] {MASTER_CSV.name}  nrows={nrows or 'ALL'}")
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False, nrows=nrows)
    log(f"  rows={len(df):,}  ({time.time()-t0:.0f}s)")

    out = pd.DataFrame(index=df.index)
    out["rid16"] = _rid16(df[COL_RID])
    out["ban"] = _num(df[COL_BAN]).fillna(0).astype(int)

    rid = df[COL_RID].astype(str)
    grp = df.groupby(rid, sort=False)
    dd = _num(df[COL_DATE])
    ped = df[COL_PED].astype(str)
    jyun = _num(df[COL_JYUN])
    top3 = (jyun <= 3).astype("float32")
    win = (jyun == 1).astype("float32")
    fc = {}  # family counts

    def addcol(fam, name, series):
        out[name] = pd.to_numeric(series, errors="coerce").astype("float32")
        fc[fam] = fc.get(fam, 0) + 1

    # 馬ごとソート（キャリア expanding 用）
    order = pd.DataFrame({"_ped": ped, "_dd": dd, "_rid": rid}).sort_values(
        ["_ped", "_dd", "_rid"], kind="mergesort").index

    def horse_expanding(values, op):
        """馬ごと(日付順) expanding。values: Series(df.index)。op: 'cummin'/'cummax'/'mean'/'std'/'last3'."""
        s = pd.Series(values, index=df.index).reindex(order)
        g = s.groupby(ped.reindex(order), sort=False)
        if op == "cummin":  r = g.cummin()
        elif op == "cummax": r = g.cummax()
        elif op == "mean":
            r = g.cumsum() / (g.cumcount() + 1)
        elif op == "last3":
            r = g.rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
        elif op == "shift_mean":   # 当該を除いた過去平均
            cs = g.cumsum() - s
            cc = g.cumcount()
            r = cs / cc.replace(0, np.nan)
        else: raise ValueError(op)
        return r.reindex(df.index)

    # =====================================================================
    # S1 スピード指数 (prev_hosei / prev_hosei9 のキャリア集計 + バケット相対)
    # =====================================================================
    log("[S1] スピード指数 ...")
    for base in ["prev_hosei", "prev_hosei9"]:
        if base not in df.columns: continue
        v = _num(df[base])
        addcol("S1", f"s1_{base}_best", horse_expanding(v, "cummin"))    # 最速(=最小)
        addcol("S1", f"s1_{base}_worst", horse_expanding(v, "cummax"))
        addcol("S1", f"s1_{base}_mean", horse_expanding(v, "mean"))
        addcol("S1", f"s1_{base}_last3", horse_expanding(v, "last3"))
        pm = horse_expanding(v, "shift_mean")
        addcol("S1", f"s1_{base}_vs_career", v - pm)                     # 今回(前走)図 - 過去平均
    # 距離×芝ダ バケット相対 (train期間mapで固定→leak-safe)
    dist = _num(df.get("距離")); sd = df["芝・ダ"].astype(str).str.strip()
    distbin = (dist // 200).clip(4, 14).fillna(-1).astype(int)
    bucket = sd + "_" + distbin.astype(str)
    if "prev_hosei" in df.columns:
        tr_mask = (df["split"] == "train") if "split" in df.columns else slice(None)
        med = _num(df.loc[tr_mask, "prev_hosei"]).groupby(bucket[tr_mask]).median()
        addcol("S1", "s1_hosei_vs_bucket", _num(df["prev_hosei"]) - bucket.map(med))
    log(f"  S1={fc.get('S1',0)}")

    # =====================================================================
    # S2 ペース/脚質 (前走コーナー → 脚質 → レース内ペース投影 → pace-fit)
    # =====================================================================
    log("[S2] ペース/脚質 ...")
    # 前1角は38%しか無い→前1→2→3角でコアレス(89%)、さらに馬キャリアffill(as-of)で補完。
    c1 = _num(df.get("前1角")); c2 = _num(df.get("前2角"))
    c3 = _num(df.get("前3角")); c4 = _num(df.get("前4角"))
    pf = _num(df.get("前走出走頭数")).clip(lower=2)
    early_raw = (c1.fillna(c2).fillna(c3) / pf).clip(0, 1)      # 早め位置 frac (0=先頭,1=最後方)
    late_raw = (c4.fillna(c3) / pf).clip(0, 1)

    def horse_ffill(vals):
        s = pd.Series(np.asarray(vals, float), index=df.index).reindex(order)
        s = s.groupby(ped.reindex(order), sort=False).ffill()   # 過去の直近既知値を前進補完(as-of)
        return s.reindex(df.index)

    early = early_raw.fillna(pd.Series(horse_ffill(early_raw.values), index=df.index))
    late = late_raw.fillna(pd.Series(horse_ffill(late_raw.values), index=df.index))
    addcol("S2", "s2_style_early", early)
    addcol("S2", "s2_style_late", late)
    addcol("S2", "s2_style_move", late - early)                 # 道中の押し上げ/下げ
    is_front = (early <= 0.25).astype("float32")                # 逃げ/先行
    is_closer = (early >= 0.6).astype("float32")                # 差し/追込
    addcol("S2", "s2_is_front", is_front)
    addcol("S2", "s2_is_closer", is_closer)
    # レース内ペース投影: 先行馬の数 → ペース圧
    tmp = df.assign(_isf=is_front.values, _early=early.values)
    n_front = tmp.groupby(rid, sort=False)["_isf"].transform("sum")
    field = _num(df.get("出走頭数")).clip(lower=2)
    pace_press = (n_front / field).astype("float32")
    addcol("S2", "s2_race_n_front", n_front)
    addcol("S2", "s2_race_pace_press", pace_press)
    # pace-fit: 先行馬は高ペースで不利, 差し馬は高ペースで有利
    addcol("S2", "s2_fit_front", -(is_front * pace_press))
    addcol("S2", "s2_fit_closer", (is_closer * pace_press))
    # 脚質のレース内希少性 (同型が少ないほど展開利)
    style_rank = tmp.groupby(rid, sort=False)["_early"].rank(pct=True)
    addcol("S2", "s2_style_rank_in_race", style_rank)
    log(f"  S2={fc.get('S2',0)}")

    # =====================================================================
    # S3 種牡馬/母父 as-of 適性 (当日除外で産駒過去成績を集計)
    # =====================================================================
    log("[S3] 種牡馬/母父 as-of適性 ...")
    work = pd.DataFrame({"_dd": dd, "_rowid": np.arange(len(df)),
                         "_sire": df.get("種牡馬").astype(str),
                         "_msire": df.get("母父馬").astype(str),
                         "_sd": sd, "_db": distbin.astype(str),
                         "_top3": top3.values, "_win": win.values})
    for key, kc in [("sire", ["_sire"]), ("msire", ["_msire"])]:
        r, c = asof_group_rate(work, kc, "_top3")
        addcol("S3", f"s3_{key}_top3_asof", r)
        addcol("S3", f"s3_{key}_n_asof", np.log1p(c))
        rw, _ = asof_group_rate(work, kc, "_win")
        addcol("S3", f"s3_{key}_win_asof", rw)
        # 芝ダ別
        r2, c2 = asof_group_rate(work, kc + ["_sd"], "_top3")
        addcol("S3", f"s3_{key}_surf_top3_asof", r2)
        # 芝ダ×距離帯別
        r3, c3 = asof_group_rate(work, kc + ["_sd", "_db"], "_top3")
        addcol("S3", f"s3_{key}_surfdist_top3_asof", r3)
        addcol("S3", f"s3_{key}_surfdist_n_asof", np.log1p(c3))
    log(f"  S3={fc.get('S3',0)}")

    # =====================================================================
    # S4 騎手/厩舎コンテキスト率 (当日除外 as-of)
    # =====================================================================
    log("[S4] 騎手/厩舎コンテキスト率 ...")
    work4 = pd.DataFrame({"_dd": dd, "_rowid": np.arange(len(df)),
                          "_jky": df.get("騎手コード").astype(str),
                          "_trn": df.get("調教師コード").astype(str),
                          "_pl": df.get("場所").astype(str), "_sd": sd,
                          "_top3": top3.values, "_win": win.values})
    work4["_combo"] = work4["_jky"] + "|" + work4["_trn"]
    for key, kc in [("jky_course", ["_jky", "_pl", "_sd"]),
                    ("jky_overall", ["_jky"]),
                    ("trn_course", ["_trn", "_pl", "_sd"]),
                    ("trn_overall", ["_trn"]),
                    ("combo", ["_combo"])]:
        r, c = asof_group_rate(work4, kc, "_top3")
        addcol("S4", f"s4_{key}_top3_asof", r)
        addcol("S4", f"s4_{key}_n_asof", np.log1p(c))
    rw, _ = asof_group_rate(work4, ["_jky"], "_win")
    addcol("S4", "s4_jky_win_asof", rw)
    log(f"  S4={fc.get('S4',0)}")

    # =====================================================================
    # S5 レイオフ/間隔角度
    # =====================================================================
    log("[S5] レイオフ/間隔角度 ...")
    interval = _num(df.get("間隔"))
    rest_n = _num(df.get("休み明け～戦目"))
    addcol("S5", "s5_interval", interval)
    addcol("S5", "s5_is_fresh", (interval >= 12).astype("float32"))     # 約3ヶ月+
    addcol("S5", "s5_is_quick", (interval <= 2).astype("float32"))      # 連闘/中1
    # 当該含めない過去走数(キャリア)
    s_one = pd.Series(1.0, index=df.index).reindex(order)
    car = s_one.groupby(ped.reindex(order), sort=False).cumcount().reindex(df.index)
    addcol("S5", "s5_career_runs", car)
    addcol("S5", "s5_fresh_x_runs", (interval >= 12).astype("float32") * np.log1p(car))
    addcol("S5", "s5_interval_x_age", interval * _num(df.get("年齢")))
    log(f"  S5={fc.get('S5',0)}")

    # =====================================================================
    # S6 フィールド競争構造 (レース内の戦力分散・トップ差・エントロピー)
    # =====================================================================
    log("[S6] フィールド競争構造 ...")
    for col, tag in [("kako5_avg_pos", "k5pos"), ("prev_hosei", "hosei"),
                     ("horse_fuku30", "hf30")]:
        if col not in df.columns: continue
        v = _num(df[col])
        tmp2 = df.assign(_v=v.values)
        g2 = tmp2.groupby(rid, sort=False)["_v"]
        vmean = g2.transform("mean"); vstd = g2.transform("std")
        vmin = g2.transform("min"); vmax = g2.transform("max")
        addcol("S6", f"s6_{tag}_field_std", vstd)               # 戦力分散(混戦度)
        addcol("S6", f"s6_{tag}_field_range", vmax - vmin)
        # 自分とトップ(最良)の差。pos系は小さいほど良→min が良
        addcol("S6", f"s6_{tag}_gap_to_best", v - vmin)
        addcol("S6", f"s6_{tag}_z_in_field", safe_div(v - vmean, vstd))
    # 戦力エントロピー (fuku30 を確率化したレース内エントロピー)
    if "horse_fuku30" in df.columns:
        p = _num(df["horse_fuku30"]).clip(lower=1e-4)
        tmp3 = df.assign(_p=p.values)
        psum = tmp3.groupby(rid, sort=False)["_p"].transform("sum")
        pr = (p / psum).clip(1e-6, 1)
        ent_term = -(pr * np.log(pr))
        race_ent = tmp3.assign(_e=ent_term.values).groupby(rid, sort=False)["_e"].transform("sum")
        addcol("S6", "s6_field_entropy", race_ent)              # 高=混戦, 低=1強
        addcol("S6", "s6_my_prob_share", pr)
    log(f"  S6={fc.get('S6',0)}")

    # =====================================================================
    # S7 クラス力学 (生涯最高クラス・現級経験・初挑戦)
    # =====================================================================
    log("[S7] クラス力学 ...")
    CLASS_MAP = {"新馬":0,"未勝利":0,"500万":1,"1勝":1,"1000万":2,"2勝":2,"1600万":3,
                 "3勝":3,"ｵｰﾌﾟﾝ":4,"オープン":4,"OP(L)":4,"OP":4,"重賞":5,"Ｇ３":5,"Ｇ２":6,"Ｇ１":7}
    cls = df["クラス名"].astype(str).str.strip().map(CLASS_MAP)
    addcol("S7", "s7_class_ord", cls)
    best_cls = horse_expanding(cls, "cummax")        # 生涯最高(当該含む≈過去最高)
    addcol("S7", "s7_best_class_ever", best_cls)
    addcol("S7", "s7_class_vs_best", cls - best_cls) # <0=格落ち
    # 現級での過去経験回数(当日除外)
    work7 = pd.DataFrame({"_dd": dd, "_rowid": np.arange(len(df)),
                          "_ped": ped, "_cls": cls.astype("Int64").astype(str),
                          "_one": 1.0})
    r, c = asof_group_rate(work7, ["_ped", "_cls"], "_one")
    addcol("S7", "s7_at_class_count", np.log1p(c))
    addcol("S7", "s7_first_at_class", (np.nan_to_num(c) == 0).astype("float32"))
    log(f"  S7={fc.get('S7',0)}")

    # =====================================================================
    # 保存
    # =====================================================================
    feat_cols = [c for c in out.columns if c not in ("rid16", "ban")]
    dup = int(out.duplicated(subset=["rid16", "ban"]).sum())
    log(f"\n[total] serious特徴 {len(feat_cols)} 列  key_dup={dup}")
    OUT_PARQUET.parent.mkdir(exist_ok=True)
    out.to_parquet(OUT_PARQUET, index=False)
    log(f"[saved] {OUT_PARQUET}  ({OUT_PARQUET.stat().st_size/1e6:.0f} MB)")

    manifest = {"n_features": len(feat_cols), "family_counts": fc,
                "key_dup_rows": dup, "elapsed_sec": round(time.time()-t0, 1),
                "asof_policy": "S3/S4/S7率は当日(_dd)を丸ごと除外して過去のみ。自レース込みleak構造排除。"}
    OUT_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    build()
