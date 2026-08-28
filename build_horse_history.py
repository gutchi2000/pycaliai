# -*- coding: utf-8 -*-
"""
build_horse_history.py
======================
serve skew 残差回収 (hist_same_* / course_* / jockey_* / 騎手・調教師コード) 用の
馬ごと全キャリア履歴 parquet を構築する。

  入力:
    data/master_v2_20130105-20251228.csv   ... 2013〜2025 の全走 (血統登録番号あり)
    data/kekka/2026*.csv                   ... 2026 週次結果 (全出走馬 + 確定着順)
    data/weekly/2026*.csv                  ... 2026 週次出走表 (騎手/調教師/父/芝ダ/距離)
    E:\競馬過去走データ\kisyu_code.csv      ... 騎手名(ターゲット表記) → 騎手コード
    E:\競馬過去走データ\2026POG\２歳戦_調教師.csv ... 調教師名 → 調教師コード (+騎手も補完)

  出力:
    data/_horse_history.parquet   ... 1行=1走 (ped_id, name, sire, birth_year,
                                      date, place, surface, dist, pos, jockey_code, trainer_code)
    data/serve_code_maps.json     ... {"jockey": {名前: コード}, "trainer": {...}}

serve 側 (serve_history_feats.py) はこの parquet を馬名+父名/生年で引き、
レース日より厳密に前の走のみで学習と同一定義の特徴を再計算する。

週次運用: Phase C (weekly_post.ps1) で kekka 配置後に再実行して 2026 分を追随させる。
実行: python build_horse_history.py
"""
from __future__ import annotations

import json
import sys
import zlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = Path(__file__).parent
MASTER_CSV = BASE / "data" / "master_v2_20130105-20251228.csv"
KEKKA_DIR  = BASE / "data" / "kekka"
WEEKLY_DIR = BASE / "data" / "weekly"
OUT_PARQUET = BASE / "data" / "_horse_history.parquet"
OUT_MAPS    = BASE / "data" / "serve_code_maps.json"

EXT_DIR     = Path(r"E:\競馬過去走データ")
KISYU_CODE  = EXT_DIR / "kisyu_code.csv"
TRAINER_SRC = EXT_DIR / "2026POG" / "２歳戦_調教師.csv"

# 週次CSV 行フォーマット (predict_weekly.py と同期)
from predict_weekly import (RACE_COLS, HORSE_COLS_33, HORSE_COLS_46,
                            HORSE_COLS_48, HORSE_COLS_49, HORSE_COLS_99)

_HORSE_COLS = {33: HORSE_COLS_33, 46: HORSE_COLS_46, 48: HORSE_COLS_48,
               49: HORSE_COLS_49, 99: HORSE_COLS_99}

# 芝・ダ 正規化 (master は 芝/ダ、週次は 芝/ダート)
_SURF = {"芝": "芝", "ダ": "ダ", "ダート": "ダ"}


def log(msg: str):
    print(msg, flush=True)


def _clean_name(s) -> str:
    return str(s).replace("\u3000", "").strip()


def synthetic_id(name: str, sire: str) -> int:
    """master に居ない馬 (2026 デビュー等) 用の安定な負 ID。"""
    key = f"{name}|{sire}".encode("utf-8")
    return -int(zlib.crc32(key))


# ------------------------------------------------------------
# コードマップ (騎手名 → 騎手コード / 調教師名 → 調教師コード)
# ------------------------------------------------------------
def build_code_maps() -> dict:
    jockey: dict[str, int] = {}
    trainer: dict[str, int] = {}

    # 2歳戦CSV (歴史分, 騎手コード+調教師コードの実対応)。日付昇順に上書き
    # して最新対応を残す (同名は稀だが後勝ち)。
    if TRAINER_SRC.exists():
        t = pd.read_csv(TRAINER_SRC, encoding="cp932", dtype=str,
                        usecols=["年", "月", "日", "騎手名", "騎手コード",
                                 "調教師", "調教師コード"])
        for c in ("年", "月", "日"):
            t[c] = pd.to_numeric(t[c], errors="coerce")
        t = t.sort_values(["年", "月", "日"])
        for _, r in t.iterrows():
            jn, jc = _clean_name(r.get("騎手名")), r.get("騎手コード")
            tn, tc = _clean_name(r.get("調教師")), r.get("調教師コード")
            try:
                if jn and jc == jc and str(jc).strip():
                    jockey[jn] = int(str(jc))
            except (ValueError, TypeError):
                pass
            try:
                if tn and tc == tc and str(tc).strip():
                    trainer[tn] = int(str(tc))
            except (ValueError, TypeError):
                pass
        log(f"  2歳戦CSV: 騎手 {len(jockey)} / 調教師 {len(trainer)} 名")

    # kisyu_code.csv (現役中心の正マスタ) を優先で上書き
    if KISYU_CODE.exists():
        k = pd.read_csv(KISYU_CODE, encoding="cp932", dtype=str)
        n_add = 0
        for _, r in k.iterrows():
            name = _clean_name(r.get("名前(ターゲット内表記)"))
            code = r.get("コード")
            try:
                if name and code == code and str(code).strip():
                    jockey[name] = int(str(code))
                    n_add += 1
            except (ValueError, TypeError):
                pass
        log(f"  kisyu_code.csv: {n_add} 名で上書き → 騎手計 {len(jockey)} 名")
    else:
        log(f"  ⚠ kisyu_code.csv 未配置: {KISYU_CODE}")

    return {"jockey": jockey, "trainer": trainer}


# ------------------------------------------------------------
# master → 履歴行
# ------------------------------------------------------------
def load_master_history() -> pd.DataFrame:
    log(f"master 読み込み: {MASTER_CSV.name}")
    cols = ["血統登録番号", "馬名", "日付", "場所", "芝・ダ", "距離",
            "着順", "騎手コード", "調教師コード", "種牡馬", "年齢"]
    m = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", usecols=cols,
                    low_memory=False)
    log(f"  {len(m):,} 行")

    out = pd.DataFrame({
        "ped_id": pd.to_numeric(m["血統登録番号"], errors="coerce"),
        "name":   m["馬名"].map(_clean_name),
        "sire":   m["種牡馬"].map(_clean_name),
        "date":   pd.to_numeric(m["日付"], errors="coerce"),
        "place":  m["場所"].astype(str).str.strip(),
        "surface": m["芝・ダ"].astype(str).str.strip().map(_SURF).fillna(""),
        "dist":   pd.to_numeric(m["距離"], errors="coerce"),
        "pos":    pd.to_numeric(m["着順"], errors="coerce"),
        "jockey_code": pd.to_numeric(m["騎手コード"], errors="coerce"),
        "trainer_code": pd.to_numeric(m["調教師コード"], errors="coerce"),
        "age":    pd.to_numeric(m["年齢"], errors="coerce"),
    })
    out = out.dropna(subset=["ped_id", "date"]).copy()
    out["ped_id"] = out["ped_id"].astype(np.int64)
    out["date"] = out["date"].astype(np.int32)
    out["birth_year"] = (out["date"] // 10000 - out["age"]).astype("Int32")
    out["pos"] = out["pos"].where(out["pos"] > 0)  # 0/負 = 中止扱い → NaN
    out = out.drop(columns=["age"])
    out["src"] = "master"
    return out


# ------------------------------------------------------------
# 2026 補完: kekka (全馬着順) × weekly (騎手/父/芝ダ/距離)
# ------------------------------------------------------------
def parse_weekly_light(path: Path) -> pd.DataFrame:
    """parse_csv の軽量版: 調教/補正 JOIN なしで馬行だけ取り出す。"""
    for enc in ("cp932", "shift_jis", "utf-8"):
        try:
            text = path.read_bytes().decode(enc)
            break
        except Exception:
            continue
    else:
        return pd.DataFrame()

    rows: list[dict] = []
    current_race: dict | None = None
    for line in text.splitlines():
        cols = line.split(",")
        if cols[0] in ("レースID(新)", "枠番", "番", ""):
            continue
        if len(cols) == 19:
            current_race = dict(zip(RACE_COLS, cols))
        elif len(cols) in _HORSE_COLS and current_race:
            h = dict(zip(_HORSE_COLS[len(cols)], cols))
            rows.append({
                "rid16": str(current_race["レースID(新)"]).strip()[:16],
                "馬番": h.get("馬番"),
                "name": _clean_name(h.get("馬名S")),
                "sire": _clean_name(h.get("父")),
                "age": h.get("年齢"),
                "jockey_name": _clean_name(h.get("騎手")),
                "trainer_name": _clean_name(h.get("調教師")),
                "surface": _SURF.get(str(current_race.get("芝・ダート", "")).strip(), ""),
                "dist": current_race.get("距離"),
                "place": str(current_race.get("場所", "")).strip(),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for c in ("馬番", "age", "dist"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_2026_history(master_max_date: int, jockey_map: dict,
                      trainer_map: dict) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    kekka_files = sorted(KEKKA_DIR.glob("*.csv"))
    for kp in kekka_files:
        stem = kp.stem
        if not (stem.isdigit() and len(stem) == 8):
            continue
        d = int(stem)
        if d <= master_max_date:
            continue
        wp = WEEKLY_DIR / f"{stem}.csv"
        if not wp.exists():
            log(f"  ⚠ {stem}: weekly CSV なし → skip (kekka のみでは芝ダ/距離不明)")
            continue
        try:
            k = pd.read_csv(kp, encoding="cp932", dtype=str)
        except Exception as e:
            log(f"  ⚠ {stem}: kekka 読込失敗 {e} → skip")
            continue
        if "レースID(新)" not in k.columns or "確定着順" not in k.columns:
            log(f"  ⚠ {stem}: kekka 列不足 → skip")
            continue
        k = k.assign(
            rid16=k["レースID(新)"].astype(str).str.strip().str[:16],
            馬番=pd.to_numeric(k["馬番"], errors="coerce"),
            pos=pd.to_numeric(k["確定着順"], errors="coerce"),
        )
        k["pos"] = k["pos"].where(k["pos"] > 0)
        w = parse_weekly_light(wp)
        if w.empty:
            log(f"  ⚠ {stem}: weekly パース 0 行 → skip")
            continue
        merged = k[["rid16", "馬番", "馬名", "pos"]].merge(
            w, on=["rid16", "馬番"], how="inner")
        # 馬名照合 (結合キー化けの保険)
        merged = merged[merged["馬名"].map(_clean_name) == merged["name"]]
        if merged.empty:
            log(f"  ⚠ {stem}: kekka×weekly 結合 0 行")
            continue
        merged["date"] = np.int32(d)
        merged["birth_year"] = (d // 10000 - merged["age"]).astype("Int32")
        merged["jockey_code"] = merged["jockey_name"].map(jockey_map).astype("float64")
        merged["trainer_code"] = merged["trainer_name"].map(trainer_map).astype("float64")
        rows.append(merged[["name", "sire", "birth_year", "date", "place",
                            "surface", "dist", "pos", "jockey_code", "trainer_code"]])
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["src"] = "kekka2026"
    return out


def resolve_2026_ped_ids(sup: pd.DataFrame, master_h: pd.DataFrame) -> pd.DataFrame:
    """2026 行を master の血統登録番号へ紐付け。曖昧/新馬は合成 ID。"""
    ent = (master_h.groupby("ped_id")
           .agg(name=("name", "last"), sire=("sire", "last"),
                birth_year=("birth_year", "last")).reset_index())
    by_name: dict[str, list] = {}
    for r in ent.itertuples(index=False):
        by_name.setdefault(r.name, []).append(r)

    ped_ids = np.empty(len(sup), dtype=np.int64)
    n_master, n_synth, n_ambig = 0, 0, 0
    for i, r in enumerate(sup.itertuples(index=False)):
        cands = by_name.get(r.name, [])
        hit = [c for c in cands
               if (r.sire and c.sire and c.sire == r.sire)
               or (pd.notna(c.birth_year) and pd.notna(r.birth_year)
                   and abs(int(c.birth_year) - int(r.birth_year)) <= 1)]
        if len(hit) == 1:
            ped_ids[i] = int(hit[0].ped_id)
            n_master += 1
        elif len(hit) > 1:
            # 父+生年の両方一致で絞る
            hit2 = [c for c in hit
                    if r.sire and c.sire == r.sire
                    and pd.notna(c.birth_year) and pd.notna(r.birth_year)
                    and abs(int(c.birth_year) - int(r.birth_year)) <= 1]
            if len(hit2) == 1:
                ped_ids[i] = int(hit2[0].ped_id)
                n_master += 1
            else:
                ped_ids[i] = synthetic_id(r.name, r.sire)
                n_ambig += 1
        else:
            ped_ids[i] = synthetic_id(r.name, r.sire)
            n_synth += 1
    sup = sup.copy()
    sup["ped_id"] = ped_ids
    log(f"  2026 紐付け: master {n_master:,} / 新馬(合成ID) {n_synth:,} / 曖昧→合成 {n_ambig:,}")
    return sup


def main() -> int:
    log("=" * 64)
    log("build_horse_history.py")
    log("=" * 64)

    log("\n[1/4] コードマップ構築")
    maps = build_code_maps()

    log("\n[2/4] master 履歴")
    mh = load_master_history()
    master_max = int(mh["date"].max())
    log(f"  master 最終日: {master_max}")

    log("\n[3/4] 2026 補完 (kekka × weekly)")
    sup = load_2026_history(master_max, maps["jockey"], maps["trainer"])
    if len(sup):
        jc_cov = sup["jockey_code"].notna().mean() * 100
        tc_cov = sup["trainer_code"].notna().mean() * 100
        log(
            f"  2026 補完: {len(sup):,} 行 "
            f"({int(sup['date'].min())}〜{int(sup['date'].max())}, "
            f"騎手コード被覆 {jc_cov:.1f}% / 調教師コード被覆 {tc_cov:.1f}%)")
        sup = resolve_2026_ped_ids(sup, mh)
        hist = pd.concat([mh, sup], ignore_index=True)
    else:
        log("  2026 補完なし")
        hist = mh

    log("\n[4/4] 保存")
    hist = hist.sort_values(["ped_id", "date"]).reset_index(drop=True)
    hist["dist"] = pd.to_numeric(hist["dist"], errors="coerce")
    for c, t in [("date", np.int32), ("ped_id", np.int64)]:
        hist[c] = hist[c].astype(t)
    hist.to_parquet(OUT_PARQUET, index=False)
    log(f"  {OUT_PARQUET.name}: {len(hist):,} 行 / {hist['ped_id'].nunique():,} 頭 "
        f"({hist['date'].min()}〜{hist['date'].max()})")

    with open(OUT_MAPS, "w", encoding="utf-8") as f:
        json.dump({"generated_at": datetime.now().isoformat(timespec="seconds"),
                   "jockey": maps["jockey"], "trainer": maps["trainer"]},
                  f, ensure_ascii=False)
    log(f"  {OUT_MAPS.name}: 騎手 {len(maps['jockey'])} / 調教師 {len(maps['trainer'])} 名")
    return 0


if __name__ == "__main__":
    sys.exit(main())
