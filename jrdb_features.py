# -*- coding: utf-8 -*-
"""
jrdb_features.py — JRDB 成績データ (SED) から「過去走の中身」の特徴を作る (v7 候補、2026-11-13)
=================================================================================
背景: AI の最大の見落としは前走のスピード指数と、位置取り・ペース・不利の補正だった
      (docs/research/FEATURE_GAP_SCAN_20261113.md、EXCUSABLE_FLOP_20261113.md)。
      JRDB の過去 1-3 走の数値を足す A/B は ◎3着内 +1.10pt (lab/experiments/jrdb_feats_exp.py、J1 アーム)。

作るもの (59 列、接頭辞 j1_):
  j1_{項目}_{k}   k=1,2,3 走前。項目 = idm, soten, furi, mae_furi, naka_furi, ato_furi, deokure, ichidori, course_dori,
                  yon_course, joushou, ten_idx, agari_idx, pace_idx, racep_idx, batai, kehai, uma_pace_n, race_pace_n
  j1_idm_mean5 / j1_idm_max5   近 5 走の IDM 平均・最大

先読み対策: SED はそのレースの週の木曜 16 時に公開される。対象レースの 5 日以上前の走だけを使う (MIN_GAP_DAYS)。
学習 (master_v2 の各行) と配信 (週次出走表の各馬) で **同じ関数** を通す = 作り方の食い違いが起きない。
キー: 血統登録番号の下 8 桁 + レース日 (YYYYMMDD)。

規約: JRDB のデータ・指数は加工しても第三者が見られる場所に出せない。この特徴の値・寄与を bundle やサイトに書き出さないこと
      (JRDB_FEATURE_PREFIX で始まる列は公開物から除外する)。data/jrdb/ は git 管理外。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
SED = BASE / "data" / "jrdb" / "sed.parquet"
JRDB_FEATURE_PREFIX = "j1_"
MODE = "j1_v1"
MIN_GAP_DAYS = 5
NUM_COLS = ["idm", "soten", "furi", "mae_furi", "naka_furi", "ato_furi", "deokure", "ichidori", "course_dori", "yon_course", "joushou",
            "ten_idx", "agari_idx", "pace_idx", "racep_idx", "batai", "kehai", "uma_pace_n", "race_pace_n"]
FEATURES = [f"j1_{c}_{k}" for k in (1, 2, 3) for c in NUM_COLS] + ["j1_idm_mean5", "j1_idm_max5"]
_PACE = {"H": 1, "M": 2, "S": 3}
_cache: dict = {}


def _norm_name(x) -> str:
    """馬名の正規化: 半角・全角の空白を除く。"""
    return str(x).replace(chr(0x3000), "").replace(" ", "").strip()


def ped8(s: pd.Series) -> pd.Series:
    """血統登録番号 → 下 8 桁の文字列 (数値で読まれていても、10 桁でも 8 桁でも同じ結果)。"""
    t = s.astype(str).str.replace(r"\.0$", "", regex=True).str.replace(r"[^0-9]", "", regex=True)
    return t.str[-8:].str.zfill(8)


def load_history(path: Path | None = None) -> dict:
    """SED を (馬, 日付) 順に並べ、検索用の整数キーと数値行列を返す。プロセス内でキャッシュ。"""
    p = Path(path) if path else SED
    key = (str(p), p.stat().st_mtime_ns)
    if _cache.get("key") == key:
        return _cache["h"]
    h = pd.read_parquet(p, columns=["ped", "ymd", "name", "idm", "soten", "furi", "mae_furi", "naka_furi", "ato_furi", "deokure", "ichidori",
                                    "course_dori", "yon_course", "joushou", "ten_idx", "agari_idx", "pace_idx", "racep_idx",
                                    "batai_cd", "kehai_cd", "uma_pace", "race_pace"])
    h["ped"] = ped8(h["ped"])
    h["dt"] = pd.to_datetime(h["ymd"].astype("int64").astype(str), format="%Y%m%d", errors="coerce")
    h = h.dropna(subset=["dt"])
    h["batai"] = pd.to_numeric(h["batai_cd"], errors="coerce")
    h["kehai"] = pd.to_numeric(h["kehai_cd"], errors="coerce")
    h["uma_pace_n"] = h["uma_pace"].map(_PACE)
    h["race_pace_n"] = h["race_pace"].map(_PACE)
    h = h.sort_values(["ped", "dt"], kind="mergesort").reset_index(drop=True)
    peds, code = np.unique(h["ped"].values, return_inverse=True)
    days = (h["dt"].values.astype("datetime64[D]") - np.datetime64("1970-01-01")).astype(np.int64)
    # 馬名 → [(初出走日, 血統登録番号 下 8 桁), ...]。2026 年に初めて走った馬は配信側の履歴に登録番号が無い
    # (名前由来の仮番号) ので、名前で引き直すための表。同名の別馬は「対象レースより前に初出走した中で最も新しい馬」を採る。
    first = h.drop_duplicates("ped", keep="first")[["name", "ped", "ymd"]]
    name_map: dict = {}
    for n_, pd_, y_ in zip(first["name"].astype(str).map(_norm_name).values, first["ped"].values, first["ymd"].values):
        name_map.setdefault(n_, []).append((int(y_), pd_))
    out = dict(peds=peds, code=code.astype(np.int64), days=days, hkey=code.astype(np.int64) * 100000 + days,
               num=h[NUM_COLS].astype(float).values, max_ymd=int(h["ymd"].max()), n=len(h), name_map=name_map)
    _cache["key"], _cache["h"] = key, out
    return out


NAME_MAX_AGE_DAYS = 1100   # 名前で引き直すのは、その名前の馬が約 3 年以内に走っている場合だけ (名前の再利用を避ける)


def past_run_features(ped: pd.Series, ymd: pd.Series, history: dict | None = None, name: pd.Series | None = None) -> pd.DataFrame:
    """各 (馬, レース日) について、5 日以上前の走から 1-3 走前の数値と近 5 走 IDM を返す (index は ped と同じ)。

    name を渡すと、登録番号が SED に無い行 (配信側の仮番号) を馬名で引き直す。学習 (master) では渡さない。
    """
    h = history or load_history()
    p8 = ped8(ped).values.astype(object)
    if name is not None:
        known0 = np.isin(p8, h["peds"])
        nm = name.map(lambda x: _norm_name(x) if isinstance(x, str) else "").values
        yv = pd.to_numeric(ymd, errors="coerce").values
        for i in np.where(~known0)[0]:
            if not nm[i] or not np.isfinite(yv[i]):
                continue
            cands = [c for c in h["name_map"].get(nm[i], ()) if c[0] < int(yv[i])]
            if not cands:
                continue
            ped_c = max(cands)[1]
            # その馬が対象レースの NAME_MAX_AGE_DAYS 日以内に走っているか (古い同名馬を拾わない)
            code_c = int(np.searchsorted(h["peds"], ped_c))
            tday = int((np.datetime64(pd.Timestamp(str(int(yv[i]))), "D") - np.datetime64("1970-01-01")).astype(np.int64))
            j = int(np.searchsorted(h["hkey"], code_c * 100000 + tday - 1, side="right")) - 1
            if j >= 0 and h["code"][j] == code_c and tday - int(h["days"][j]) <= NAME_MAX_AGE_DAYS:
                p8[i] = ped_c
    p8 = p8.astype(str)
    dt = pd.to_datetime(pd.to_numeric(ymd, errors="coerce").astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    tdays = (dt.values.astype("datetime64[D]") - np.datetime64("1970-01-01")).astype("float64")
    pos = np.searchsorted(h["peds"], p8)
    pos_c = np.clip(pos, 0, len(h["peds"]) - 1)
    known = (h["peds"][pos_c] == p8) & np.isfinite(tdays)
    code = np.where(known, pos_c, -1).astype(np.int64)
    limit = np.where(known, tdays - MIN_GAP_DAYS, 0).astype(np.int64)
    # その馬の「limit 日以前」の最後の走の位置 = searchsorted(right) − 1
    last = np.searchsorted(h["hkey"], code * 100000 + limit, side="right") - 1
    n = len(p8)
    out = {}
    lag_idm = []
    for k in (1, 2, 3, 4, 5):
        src = last - (k - 1)
        ok = known & (src >= 0)
        src_c = np.clip(src, 0, h["n"] - 1)
        ok &= h["code"][src_c] == code
        if k <= 3:
            blk = np.where(ok[:, None], h["num"][src_c], np.nan)
            for j, c in enumerate(NUM_COLS):
                out[f"j1_{c}_{k}"] = blk[:, j]
        lag_idm.append(np.where(ok, h["num"][src_c, 0], np.nan))
    stack = np.stack(lag_idm, axis=1)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            out["j1_idm_mean5"] = np.nanmean(stack, axis=1)
            out["j1_idm_max5"] = np.nanmax(stack, axis=1)
    assert n == len(out["j1_idm_mean5"])
    return pd.DataFrame(out, index=ped.index)[FEATURES]


def add_jrdb_feats(df: pd.DataFrame, mode: str = MODE, ped_col: str = "血統登録番号", date_col: str = "日付",
                   date_value: str | int | None = None, history: dict | None = None) -> pd.DataFrame:
    """df に j1_* 列を足して返す。date_value を渡すと全行そのレース日として扱う (週次出走表用)。"""
    if mode != MODE:
        raise ValueError(f"未知の jrdb_mode: {mode}")
    if ped_col not in df.columns:
        raise KeyError(f"{ped_col} 列が無い (JRDB 特徴は血統登録番号で引く)")
    ymd = pd.Series(int(date_value), index=df.index) if date_value is not None else df[date_col]
    f = past_run_features(df[ped_col], ymd, history)
    df = df.drop(columns=[c for c in FEATURES if c in df.columns])
    return pd.concat([df, f], axis=1)


def staleness_days(race_ymd: str | int, history: dict | None = None) -> int:
    """SED の最終日からレース日までの日数。9 日を超えていたら先週ぶんが未取込 (特徴の走順がずれる)。"""
    h = history or load_history()
    a = pd.Timestamp(str(int(race_ymd))); b = pd.Timestamp(str(h["max_ymd"]))
    return int((a - b).days)


def coverage(df: pd.DataFrame) -> float:
    """j1_idm_1 (前走 IDM) が埋まっている割合。"""
    return float(df["j1_idm_1"].notna().mean()) if "j1_idm_1" in df.columns and len(df) else 0.0
