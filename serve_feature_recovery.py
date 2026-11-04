# -*- coding: utf-8 -*-
"""
serve_feature_recovery.py — 週次CSV (parse_csv) の DataFrame に対し、学習時 (master_v2) と同じ定義で
serve 側で作れるのに「欠損」または「別の意味の値」になっていた特徴を復元する (2026-11-04)。

背景 (analysis/serve_coverage_gap_table.py / data/serve_feature_baseline.json 2026-11-04 再測定):
  * 間隔          学習 = round((日付 - 前走日付) / 7) [週]。serve は predict_weekly の列対応
                  「前走間隔 → 間隔」で TARGET 1走前ブロックの「前走間隔」= **1走前の、そのさらに前の
                  レースとの間隔** を流していた (kako5 の 1走前ブロックと 100% 一致、真の現走間隔とは
                  4〜7% しか一致しない)。しかも前走が新馬だった馬は '初' → NaN で 15% 欠損。
                  → kako5 CSV の先頭「間隔」列 (現走の間隔) を正とし、無い馬は 前走月/日 から再計算。
  * 前走日付      学習 = YYMMDD の数値 (例 251012)。serve は列が無く常に死亡 (gain 0.46%)。
                  → 前走月/日 + 年推定 (間隔と整合する年) で復元。
  * 休み明け～戦目 学習 = 間隔>=10週 を「休み明け=1」とし、以後 間隔<10 のレースごとに +1
                  (master 上の馬ごとの連鎖で再現率 96%、新馬戦は NaN で次走は 2)。serve は常に死亡 (gain 0.45%)。
                  → 現走間隔 + kako5 各ブロックの「前走間隔」の連鎖で復元。5走以内に起点が無ければ NaN。
  * 父タイプ名 / 母父タイプ名  TARGET の血統タイプ DB が新しい種牡馬を持たず、週次CSV では 7% / 86% しか
                  埋まらない。master 由来の 種牡馬→父タイプ名 / 母父馬→母父タイプ名 表
                  (data/pedigree_type_map.json、曖昧 0) で埋まる分だけ埋める。
  * 斤量体重比    学習 = 斤量 / 当日馬体重 ×100。当日体重は発走前に無いので serve は死亡 (gain 1.07%)。
                  CB_SERVE_RECOVER=2 のときだけ 斤量 / 前走馬体重 ×100 を代理値として入れる
                  (master 上で真値との相関 0.97)。代理値は意味が少し変わるため既定では入れない。

有効化: 環境変数 CB_SERVE_RECOVER = "1" (決定論的な復元のみ) / "2" (+ 斤量体重比 代理)。
        未設定・"0" は何もしない (従来挙動)。export_weekly_marks.py と analysis/measure_serve_coverage.py
        (baseline 再測定) の両方から同じ関数を呼ぶので、baseline と本番の定義は常に一致する。
検証: analysis/replay_serve_2026.py で 2026 全開催を CB_SERVE_RECOVER=0/1 で再生して ◎top3 を比較。
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parent
PED_MAP_PATH = BASE / "data" / "pedigree_type_map.json"
LAYOFF_WEEKS = 10          # TARGET「休み明け」の閾値 (master で 休み明け==1 の 間隔 最小値 = 10)
MAX_LOOKBACK_YEARS = 3     # 前走日付の年推定で遡る範囲

RID_CANDIDATES = ("レースID(新/馬番無)", "レースID(新)")
BAN_CANDIDATES = ("馬番", "番")


def enabled() -> int:
    v = os.environ.get("CB_SERVE_RECOVER", "0").strip()
    return int(v) if v in ("1", "2") else 0


# ---------------------------------------------------------------------------
# kako5 CSV: 現走の間隔 と 各過去走ブロックの「前走間隔」を (race_id16, 馬番) で引く
# ---------------------------------------------------------------------------
def _to_int(s: str):
    s = (s or "").strip()
    if s == "" or s == "初":
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def read_kako5_intervals(path: Path) -> dict[tuple[str, int], dict]:
    """戻り値: {(rid16, umaban): {"cur": int|None, "debut": bool, "gaps": [str,...], "month": int|None, "day": int|None}}
    cur   = 現走の間隔 (週)。debut=True なら新馬 (TARGET '初')。
    gaps  = 1走前, 2走前, ... 各ブロックの「前走間隔」生文字列 ('初' / '' / 数字)。
    month/day = 1走前の月日 (前走日付の復元用)。"""
    out: dict[tuple[str, int], dict] = {}
    if not Path(path).exists():
        return out
    text = Path(path).read_text(encoding="cp932", errors="replace")
    rid = None
    idx = None
    for row in csv.reader(text.splitlines()):
        if not row:
            continue
        c0 = row[0].strip()
        if re.fullmatch(r"\d{16}", c0):
            rid = c0
            continue
        if c0 == "枠番":
            # ヘッダ: 間隔 (現走) / 番 / 月,日 (1走前) / 前走間隔 ×ブロック
            try:
                i_kan = row.index("間隔")
                i_ban = row.index("番")
            except ValueError:
                idx = None
                continue
            months = [i for i, c in enumerate(row) if c == "月"]
            gaps = [i for i, c in enumerate(row) if c == "前走間隔"]
            idx = dict(kan=i_kan, ban=i_ban, m1=months[0] if months else None,
                       gaps=gaps)
            continue
        if rid is None or idx is None or not c0.isdigit():
            continue
        try:
            ban = int(row[idx["ban"]])
        except (ValueError, IndexError):
            continue
        kan_raw = row[idx["kan"]].strip() if idx["kan"] < len(row) else ""
        rec = dict(cur=_to_int(kan_raw), debut=(kan_raw == "初"),
                   gaps=[(row[i].strip() if i < len(row) else "") for i in idx["gaps"]],
                   month=None, day=None)
        if idx["m1"] is not None and idx["m1"] + 1 < len(row):
            rec["month"] = _to_int(row[idx["m1"]])
            rec["day"] = _to_int(row[idx["m1"] + 1])
        out[(rid, ban)] = rec
    return out


# ---------------------------------------------------------------------------
# 個別の復元ロジック (純関数)
# ---------------------------------------------------------------------------
def prev_date_from_month_day(race_day: dt.date, month, day, interval_weeks=None):
    """前走の年月日を推定する。年は「間隔 (週) と最も整合する年」、間隔が無ければ直近の過去年。"""
    if month is None or day is None:
        return None
    cands = []
    for back in range(0, MAX_LOOKBACK_YEARS + 1):
        try:
            d = dt.date(race_day.year - back, int(month), int(day))
        except ValueError:
            continue
        if d < race_day:
            cands.append(d)
    if not cands:
        return None
    if interval_weeks is None or not np.isfinite(interval_weeks):
        return cands[0]
    return min(cands, key=lambda d: abs(round((race_day - d).days / 7) - interval_weeks))


def interval_weeks(race_day: dt.date, prev_day: dt.date) -> int:
    return int(round((race_day - prev_day).days / 7))


def yasumi_ake_count(cur_interval, gaps: list[str]):
    """休み明け～戦目。cur_interval=現走の間隔 (週, None=新馬)。gaps=1走前,2走前,... の「前走間隔」。"""
    if cur_interval is None or not np.isfinite(cur_interval):
        return np.nan
    if cur_interval >= LAYOFF_WEEKS:
        return 1.0
    count = 2.0                      # 前走が起点 (休み明け or 新馬) なら 2
    for g in gaps:
        g = (g or "").strip()
        if g == "初":                # 前走系列がデビュー戦で始まる → ここで確定
            return count
        if g == "":                  # ブロック空 = 過去走が尽きた (本来 '初' が立つはずだが保険)
            return np.nan
        gi = _to_int(g)
        if gi is None:
            return np.nan
        if gi >= LAYOFF_WEEKS:
            return count
        count += 1.0
    return np.nan                    # 5走以内に起点が無い → TARGET と同様に不明


# ---------------------------------------------------------------------------
# メイン: DataFrame をその場で更新し、ログ用 stats を返す
# ---------------------------------------------------------------------------
def _col(df: pd.DataFrame, cands) -> str | None:
    for c in cands:
        if c in df.columns:
            return c
    return None


def recover_serve_features(df: pd.DataFrame, date_str: str, kako5_path: Path | None = None,
                           level: int | None = None) -> dict:
    """df を in-place 更新。戻り値 stats = {"level": int, "coverage": {feat: float}, "changed": {feat: int}, ...}。
    level None のときは環境変数 CB_SERVE_RECOVER を見る。0 なら何もせず {} を返す。"""
    lv = enabled() if level is None else int(level)
    if lv <= 0 or df is None or len(df) == 0:
        return {}
    if not (str(date_str).isdigit() and len(str(date_str)) == 8):
        logger.warning(f"[serve recover] date_str が YYYYMMDD でない ({date_str}) → スキップ")
        return {}
    race_day = dt.date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))
    stats: dict = {"level": lv, "coverage": {}, "changed": {}, "notes": []}

    rid_col = _col(df, RID_CANDIDATES)
    ban_col = _col(df, BAN_CANDIDATES)
    k5: dict = {}
    if kako5_path is None:
        kako5_path = BASE / "data" / "kako5" / f"{date_str}.csv"
    if rid_col and ban_col:
        try:
            k5 = read_kako5_intervals(Path(kako5_path))
        except Exception as e:      # noqa: BLE001
            stats["notes"].append(f"kako5 読込失敗: {e}")
    if not k5:
        stats["notes"].append("kako5 無し → 間隔は 前走月/日 からの再計算のみ、休み明け～戦目は復元不可")

    rids = df[rid_col].astype(str).str.replace(r"\D", "", regex=True).str[:16] if rid_col else None
    bans = pd.to_numeric(df[ban_col], errors="coerce") if ban_col else None
    pm = pd.to_numeric(df["前走月"], errors="coerce") if "前走月" in df.columns else pd.Series(np.nan, index=df.index)
    pdd = pd.to_numeric(df["前走日"], errors="coerce") if "前走日" in df.columns else pd.Series(np.nan, index=df.index)
    has_prev = pd.to_numeric(df["前走確定着順"], errors="coerce").notna() if "前走確定着順" in df.columns \
        else pm.notna()

    new_kan = np.full(len(df), np.nan)
    new_pdate = np.full(len(df), np.nan)
    new_yasumi = np.full(len(df), np.nan)
    n_k5 = 0
    for i, (ix, hp) in enumerate(zip(df.index, has_prev.values)):
        rec = None
        if k5 and rids is not None:
            try:
                rec = k5.get((rids.iat[i], int(bans.iat[i])))
            except (ValueError, TypeError):
                rec = None
        if rec is not None:
            n_k5 += 1
            if rec["debut"]:
                continue                     # 新馬: 学習時も 間隔/前走日付/休み明け は NaN
            cur = rec["cur"]
            m, d = rec["month"], rec["day"]
            if m is None:
                m = pm.iat[i] if np.isfinite(pm.iat[i]) else None
                d = pdd.iat[i] if np.isfinite(pdd.iat[i]) else None
            pdate = prev_date_from_month_day(race_day, m, d, cur)
            if cur is None and pdate is not None:
                cur = interval_weeks(race_day, pdate)
            if cur is not None:
                new_kan[i] = cur
                new_yasumi[i] = yasumi_ake_count(cur, rec["gaps"])
            if pdate is not None:
                new_pdate[i] = int(pdate.strftime("%y%m%d"))
        else:
            if not hp:
                continue
            m = pm.iat[i] if np.isfinite(pm.iat[i]) else None
            d = pdd.iat[i] if np.isfinite(pdd.iat[i]) else None
            pdate = prev_date_from_month_day(race_day, m, d, None)
            if pdate is not None:
                new_kan[i] = interval_weeks(race_day, pdate)
                new_pdate[i] = int(pdate.strftime("%y%m%d"))

    def _put(col: str, vals: np.ndarray):
        # 週次CSV 由来の旧列は「別の意味の値」(間隔) か TARGET のプレースホルダ定数 (休み明け～戦目,
        # 斤量体重比=11.8 固定) なので残さず全置換する。復元できない馬は学習時と同じく NaN。
        old = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)
        v = pd.Series(vals, index=df.index, dtype=float)
        changed = int(((old.fillna(-1e9) != v.fillna(-1e9))).sum())
        df[col] = v
        stats["changed"][col] = changed
        stats["coverage"][col] = float(v.notna().mean())

    _put("間隔", new_kan)                     # 誤った列対応を上書き (kako5 が正)
    _put("前走日付", new_pdate)
    _put("休み明け～戦目", new_yasumi)
    stats["kako5_hit"] = n_k5

    # --- 血統タイプ (map で埋まる分だけ) ---
    if PED_MAP_PATH.exists():
        try:
            mp = json.loads(PED_MAP_PATH.read_text(encoding="utf-8"))
            for src, dst, key in (("種牡馬", "父タイプ名", "sire"), ("母父馬", "母父タイプ名", "bms")):
                if src in df.columns and dst in df.columns:
                    cur = df[dst].astype(str).str.strip()
                    blank = cur.isin(["", "nan", "None", "__NaN__"])
                    filled = df[src].astype(str).str.strip().map(mp.get(key, {}))
                    use = blank & filled.notna()
                    df.loc[use, dst] = filled[use]
                    stats["changed"][dst] = int(use.sum())
                    stats["coverage"][dst] = float((~df[dst].astype(str).str.strip().isin(["", "nan", "None", "__NaN__"])).mean())
        except Exception as e:  # noqa: BLE001
            stats["notes"].append(f"pedigree map 失敗: {e}")

    # --- 斤量体重比 代理 (level 2 のみ) ---
    if lv >= 2 and "斤量" in df.columns and "前走馬体重" in df.columns:
        kin = pd.to_numeric(df["斤量"], errors="coerce")
        pw = pd.to_numeric(df["前走馬体重"], errors="coerce")
        proxy = (kin / pw * 100.0).where((pw > 0) & kin.notna() & has_prev)   # 新馬の前走馬体重は無意味
        _put("斤量体重比", proxy.values)
        stats["notes"].append("斤量体重比 = 斤量/前走馬体重×100 (代理値)")
    return stats


def format_stats(stats: dict) -> str:
    if not stats:
        return "[serve recover] 無効 (CB_SERVE_RECOVER 未設定)"
    cov = ", ".join(f"{k}={v*100:.0f}%" for k, v in stats.get("coverage", {}).items())
    ch = ", ".join(f"{k}:{v}" for k, v in stats.get("changed", {}).items())
    s = f"[serve recover L{stats.get('level')}] kako5 {stats.get('kako5_hit', 0)} 頭 | cov {cov} | 更新 {ch}"
    if stats.get("notes"):
        s += " | " + "; ".join(stats["notes"])
    return s


if __name__ == "__main__":   # 手動検査: python serve_feature_recovery.py 20261101
    import sys
    import logging as _lg
    _lg.disable(_lg.CRITICAL)
    from predict_weekly import parse_csv
    d8 = sys.argv[1]
    _df = parse_csv(Path(f"data/weekly/{d8}.csv"))
    _lg.disable(_lg.NOTSET)
    before = _df["間隔"].copy() if "間隔" in _df.columns else None
    st = recover_serve_features(_df, d8, level=int(sys.argv[2]) if len(sys.argv) > 2 else 2)
    print(format_stats(st))
    for c in ("間隔", "前走日付", "休み明け～戦目", "斤量体重比"):
        if c in _df.columns:
            v = pd.to_numeric(_df[c], errors="coerce")
            print(f"  {c:<10} cov={v.notna().mean():.3f} mean={v.mean():.2f} min={v.min()} max={v.max()}")
    if before is not None:
        print("  間隔 旧→新 一致率:", float((pd.to_numeric(before, errors='coerce') == _df['間隔']).mean()))
    print(_df[["馬名", "前走月", "前走日", "間隔", "前走日付", "休み明け～戦目"]].head(12).to_string())
