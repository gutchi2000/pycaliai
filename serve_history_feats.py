# -*- coding: utf-8 -*-
"""
serve_history_feats.py
======================
serve skew 残差回収: 週次出走表 (馬名のみ・血統登録番号なし) に対し、
data/_horse_history.parquet (build_horse_history.py 生成) を馬名で JOIN し、
レース日より厳密に前の走のみ (leak-safe as-of) で学習と同一定義の
以下 10 特徴 + 騎手/調教師コードを再計算して埋める。

学習側の定義 (同一実装であること):
  hist_same_cond_*        parse_kako5.build_from_master: 同TD(芝/ダ) かつ
                          距離±200m の全キャリア着順から best/top3率/回数。
                          過去走ゼロ or 着順全NaN → NaN
  hist_same_place_best_pos 同場所の全キャリア最高着順。なければ NaN
  course_*                build_master_v2.compute_history_features:
                          course_key = 場所|芝ダ|距離帯(短≤1400/マ≤1700/中≤2200/長)。
                          n_prev = 過去同 key 走数 (着順NaN含む, 初出走=0)、
                          win/top3 rate = n_prev>0 のときのみ (else NaN)
  jockey_*                馬×騎手コード ペアの累積 (騎手単独ではない)。
                          n_prev = 過去同騎手騎乗数 (初コンビ=0)、rate 同上
  騎手コード/調教師コード   カテゴリ特徴。serve_code_maps.json の名前→コードで復元

同名馬の曖昧性解消: 父名 (種牡馬) 一致 or 生年 (レース年-年齢) ±1 一致。
解消不能 (複数候補が残る) → 従来通り NaN (安全側)。
未知の馬名 (新馬等) → 学習と同じく n_prev=0 / rate,hist は NaN。

馬同定は開催日 day-start 時点の情報に限る (2026-10-04 再現性修正 D2):
候補は「その馬名で date < レース日 の行を 1 行以上持つ個体」、照合する父名・生年は
「date < レース日 の行のうち最新の行」の値。結果は、履歴を date < レース日 で切り落として
索引を作り直した場合と完全一致する (tests/test_serve_repro_invariance.py の T3)。
旧実装は全期間の最終行 (後日の出走表の登録情報) で照合していた。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BASE = Path(__file__).parent
HIST_PARQUET = BASE / "data" / "_horse_history.parquet"
CODE_MAPS = BASE / "data" / "serve_code_maps.json"

# 予測経路の版 (serve_path.py が bundle へ刻む)。この module の挙動を変えたら必ず上げる。
# 仕様: docs/research/SERVE_REPRODUCIBILITY_FIX_SPEC_20261004.md §7
#   sp-20261004-0: 旧経路 (日付だけの不安定ソート、全期間の最終行で同定)
#   sp-20261004-1: 再現性修正 (順序キーの全順序 → 同定の day-start 限定 → 引用符の正規化)
SERVE_PATH_VERSION = "sp-20261004-1"
ROLLING_ORDER_KEY = "date,post_time,race_id,umaban"   # 騎手・調教師の直近集計の並び (全順序)
IDENTITY_MODE = "asof-day-start"                # 馬同定: date < レース日 の行だけで候補・照合属性
NAME_NORMALIZATION = "fullwidth-space+strip+outer-quotes"   # 馬名・父名の正規化 (_clean_name)

# 直近集計の順序キー K。学習側 build_dataset.add_rolling_stats の並び
# (日付, 発走時刻, レースID(新/馬番無), 馬番) と同じ。1 走を一意に指すので同順位は無い。
ORDER_COLUMNS = ("date", "post_time", "race_id", "umaban")


class HistorySchemaError(RuntimeError):
    """履歴 parquet が順序キーの契約を満たさない (列が無い・欠損がある・重複がある)。

    黙って旧挙動 (日付だけの並び) に落とさず、ここで止める (spec §2.2、9/11 の
    「race_id 列なしで無言 fallback」の再発防止)。発走時刻の欠損は 0 件しか許さない。
    """


def validate_history(hist: pd.DataFrame) -> dict:
    """順序キー K の列・欠損・重複・発走時刻の形を検査する。違反は HistorySchemaError。"""
    missing = [c for c in ORDER_COLUMNS if c not in hist.columns]
    if missing:
        raise HistorySchemaError(
            f"履歴 parquet に順序キーの列が無い: {missing} "
            "(python build_horse_history.py で作り直す)")
    pt = pd.to_numeric(hist["post_time"], errors="coerce")
    ub = pd.to_numeric(hist["umaban"], errors="coerce")
    n_pt_na, n_ub_na = int(pt.isna().sum()), int(ub.isna().sum())
    if n_pt_na or n_ub_na:
        raise HistorySchemaError(
            f"順序キーに欠損: post_time {n_pt_na} 行 / umaban {n_ub_na} 行 (許容 0 件)")
    bad_pt = ~((pt // 100).between(0, 23) & (pt % 100).between(0, 59))
    if bad_pt.any():
        raise HistorySchemaError(
            f"post_time が HHMM でない: {int(bad_pt.sum())} 行 (例 {pt[bad_pt].head(3).tolist()})")
    rid = hist["race_id"].astype(str)
    n_dup = int(pd.DataFrame({"date": hist["date"], "post_time": pt, "race_id": rid, "umaban": ub})
                .duplicated().sum())
    if n_dup:
        raise HistorySchemaError(f"順序キー K の重複 {n_dup} 行 (1 走を一意に指していない)")
    return {"rows": int(len(hist)), "post_time_missing": 0, "umaban_missing": 0, "order_key_duplicates": 0}

# 埋める特徴 (v6 feature_cols 名)
NUM_FEATS = [
    "hist_same_cond_best_pos", "hist_same_cond_top3_rate",
    "hist_same_cond_count", "hist_same_place_best_pos",
    "course_n_prev", "course_win_rate", "course_top3_rate",
    "jockey_n_prev", "jockey_win_rate", "jockey_top3_rate",
    "horse_fuku10", "horse_fuku30",
    "jockey_fuku30", "jockey_fuku90",
    "trainer_fuku30", "trainer_fuku90",
]
CAT_FEATS = ["騎手コード", "調教師コード"]
PED_ID_COL = "_serve_ped_id"                      # 同定できた馬 (hit) の血統登録番号。JRDB 過去走特徴 (jrdb_features.py) が引くキー
IDENTITY_STATUS_COL = "_serve_identity_status"   # fill_history_features が馬ごとの同定状態を書く列

_SURF = {"芝": "芝", "ダ": "ダ", "ダート": "ダ"}


def _dist_band(d) -> str:
    # build_master_v2.compute_history_features と同一
    if pd.isna(d):
        return "?"
    d = int(d)
    if d <= 1400:
        return "短"
    if d <= 1700:
        return "マ"
    if d <= 2200:
        return "中"
    return "長"


def _clean_name(s) -> str:
    """馬名・父名・騎手名・調教師名の正規化 (履歴生成時と配信時で同じものを通す)。

    全角空白の除去・前後空白の除去に加え、外側の引用符 1 組を外す (2026-10-04 再現性修正 §3.2)。
    出走表 CSV は空白を含む外国産の父名を引用符つきで出す ("American Pharoah")。master 由来の
    履歴は引用符なしなので、外さないと同じ父名が不一致になる。実データで引用符が付くのは
    父名だけ (2026 年の出走表 887 行・履歴 843 行。馬名・騎手名・調教師名・コード表は 0 件)。
    """
    s = str(s).replace("　", "").strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1].strip()
    return s


class _HistoryIndex:
    """馬名 → [entity] の索引。entity = 1頭分の全走 (numpy 配列)。

    馬名・父名・生年は行ごとに持ち、同定では date < レース日 の最新行の値を使う (as-of)。
    by_name には、その個体がどこかの行で名乗った名前すべてから引けるよう登録し、
    resolve() がレース日時点の名前で絞る。
    """

    def __init__(self, hist: pd.DataFrame):
        self.by_name: dict[str, list[dict]] = {}
        hist = hist.sort_values(["ped_id", "date"])
        for ped_id, g in hist.groupby("ped_id", sort=False):
            # 履歴生成時 (build_horse_history) と同じ正規化を索引側でも通す (旧 parquet 対策)
            names = [_clean_name(v) for v in g["name"]]
            sires = [_clean_name(v) if pd.notna(v) else "" for v in g["sire"]]
            birth_years = [int(v) if pd.notna(v) else None for v in g["birth_year"]]
            ent = {
                "ped_id": int(ped_id),
                "names": names,
                "sires": sires,
                "birth_years": birth_years,
                # 全期間の最終行の値 (race_date を渡さない旧来の呼び出し元・監査スクリプト用)
                "sire": sires[-1],
                "birth_year": birth_years[-1],
                "date": g["date"].to_numpy(np.int32),
                "place": g["place"].astype(str).to_numpy(),
                "surface": g["surface"].astype(str).to_numpy(),
                "dist": pd.to_numeric(g["dist"], errors="coerce").to_numpy(np.float64),
                "pos": pd.to_numeric(g["pos"], errors="coerce").to_numpy(np.float64),
                "jockey": pd.to_numeric(g["jockey_code"], errors="coerce").to_numpy(np.float64),
            }
            for name in dict.fromkeys(names):
                self.by_name.setdefault(name, []).append(ent)
        self.by_jockey = self._build_code_index(hist, "jockey_code")
        self.by_trainer = self._build_code_index(hist, "trainer_code")

    @staticmethod
    def _build_code_index(hist: pd.DataFrame, code_col: str) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """主体コード別の (日付, 複勝フラグ) 列。日付未満だけを rolling 集計する（C1）。

        P0-5整合(2026-09-07策定、2026-09-11 C1へ訂正): 学習側
        (build_dataset.add_rolling_stats) は過去30/90「馬走」という window の
        意味(行レベル counting)を維持したままリークだけを除去するC1を採用した。
        serve側もこれに合わせて行(馬走)単位のまま扱う。

        訂正の経緯: 当初はレース単位に集約(C2)する実装だったが、これは誤り
        だった。レース単位集約が必要だったのは「学習対象レース自身」の同一
        レース内リークを消すためであり、この`_HistoryIndex`が扱うのは
        `rolling_rate()`呼び出し時に`date < race_date`で厳密に絞り込まれる
        **既に確定済みの過去レースのみ**（今日これから予測するレース自身の
        結果を含むことは構造的にあり得ない、P1-6の日付締切で担保済み）。
        過去レースで同一主体の複数頭が異なる結果(的中/不的中等)だった場合、
        それは実際に起きた2つの別事実でありリークではない。レース単位集約
        (旧C2実装)はこれを1つの値に握りつぶしてしまっていた。

        並び (2026-10-04 再現性修正 D1): 主体ごとの行を順序キー K = (date, post_time,
        race_id, umaban) の全順序で並べる (学習側と同じ並び)。旧実装は日付だけの不安定
        ソートで、窓の端が同じ日付の行の途中に落ちると、その日のどの行を含めるかが
        入力配列 (履歴の追加・行順) で変わった。K は 1 走を一意に指すので結果は入力順に
        依らないが、取り違え防止のため安定ソートを明示する。
        """
        missing = [c for c in (code_col, "pos", *ORDER_COLUMNS) if c not in hist.columns]
        if missing:
            raise HistorySchemaError(f"直近集計に必要な列が無い: {missing}")
        work = hist[["pos", code_col, *ORDER_COLUMNS]].copy()
        work["code"] = pd.to_numeric(work[code_col], errors="coerce")
        work = work.dropna(subset=["date", "code"])
        work["flag"] = (pd.to_numeric(work["pos"], errors="coerce") <= 3).astype(float)
        work["race_id"] = work["race_id"].astype(str)
        work = work.sort_values(["code", *ORDER_COLUMNS], kind="mergesort")
        out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for code, g in work.groupby("code", sort=False):
            out[int(code)] = (
                g["date"].to_numpy(np.int32),
                g["flag"].to_numpy(np.float64),
            )
        return out

    def rolling_rate(self, kind: str, code, race_date: int, window: int,
                     min_periods: int = 5) -> float:
        """レース日より前の直近 N 走(馬走単位)複勝率。同日を除き未来参照を防ぐ。"""
        if pd.isna(code):
            return np.nan
        table = self.by_jockey if kind == "jockey" else self.by_trainer
        values = table.get(int(code))
        if values is None:
            return np.nan
        dates, flags = values
        end = int(np.searchsorted(dates, race_date, side="left"))
        start = max(0, end - window)
        flags = flags[start:end]
        if len(flags) < min_periods:
            return np.nan
        return float(np.mean(flags))


    def candidates(self, name: str, race_date: int | None) -> list[dict]:
        """候補: date < race_date の行を持ち、その最新行の馬名が name の個体。

        戻り値の要素は {"ent", "sire", "birth_year"} (照合属性は date < race_date の最新行の値)。
        race_date=None は旧来の「全期間の最終行」(配信経路では使わない。EXP05-F の
        live_history など race_date を渡さない呼び出し元の挙動をそのまま保つため)。
        """
        out = []
        for ent in self.by_name.get(name, ()):
            k = (len(ent["date"]) if race_date is None
                 else int(np.searchsorted(ent["date"], int(race_date), side="left")))
            if k == 0 or ent["names"][k - 1] != name:
                continue
            out.append({"ent": ent, "sire": ent["sires"][k - 1],
                        "birth_year": ent["birth_years"][k - 1]})
        return out

    def resolve(self, name: str, sire: str, birth_year,
                race_date: int | None = None) -> tuple[dict | None, str]:
        """→ (entity or None, status)  status: hit/new/ambiguous

        race_date (YYYYMMDD) を渡すと、その日より前の行だけで候補と照合属性を作る
        (day-start 限定)。配信経路 (fill_history_features) は必ず渡す。
        """
        cands = self.candidates(name, race_date)
        if not cands:
            return None, "new"
        if len(cands) == 1:
            c = cands[0]
            # 同名の別馬 (名前再利用) ガード: 父・生年が両方分かって両方不一致なら別馬
            sire_known = bool(sire) and bool(c["sire"])
            by_known = birth_year is not None and c["birth_year"] is not None
            sire_ok = (not sire_known) or (c["sire"] == sire)
            by_ok = (not by_known) or abs(c["birth_year"] - int(birth_year)) <= 1
            if (sire_known and not sire_ok) and (by_known and not by_ok):
                return None, "new"
            if (sire_known and not sire_ok) or (by_known and not by_ok):
                # 片方だけ不一致 → 判定不能。安全側で曖昧扱い
                return None, "ambiguous"
            return c["ent"], "hit"
        hit = [c for c in cands
               if (sire and c["sire"] and c["sire"] == sire)
               or (birth_year is not None and c["birth_year"] is not None
                   and abs(c["birth_year"] - int(birth_year)) <= 1)]
        if len(hit) > 1:
            hit = [c for c in hit
                   if sire and c["sire"] == sire
                   and birth_year is not None and c["birth_year"] is not None
                   and abs(c["birth_year"] - int(birth_year)) <= 1]
        if len(hit) == 1:
            return hit[0]["ent"], "hit"
        return None, "ambiguous"


_CACHE: dict = {}


def _load(base: Path):
    key = str(base)
    if key in _CACHE:
        return _CACHE[key]
    hist_path = base / "data" / "_horse_history.parquet"
    maps_path = base / "data" / "serve_code_maps.json"
    if not hist_path.exists():
        raise FileNotFoundError(
            f"{hist_path} 未生成 (python build_horse_history.py を先に実行)")
    hist = pd.read_parquet(hist_path)
    validate_history(hist)          # 列不足・欠損・重複は HistorySchemaError (無言 fallback しない)
    idx = _HistoryIndex(hist)
    maps = {"jockey": {}, "trainer": {}}
    if maps_path.exists():
        with open(maps_path, encoding="utf-8") as f:
            raw = json.load(f)
        maps["jockey"] = raw.get("jockey", {})
        maps["trainer"] = raw.get("trainer", {})
    meta = {"n_rows": len(hist), "max_date": int(hist["date"].max()),
            "columns": sorted(str(c) for c in hist.columns),
            "path": str(hist_path), "maps_path": str(maps_path)}
    _CACHE[key] = (idx, maps, meta)
    return _CACHE[key]


def compute_row_feats(ent: dict, race_date: int, place: str, surface: str,
                      dist, jockey_code) -> dict:
    """1頭分: entity の as-of (date < race_date) 走から 10 特徴を計算。"""
    out: dict[str, float] = {}
    if ent is None:
        past = None
    else:
        m = ent["date"] < race_date  # 厳密に前 (同日は除く = leak-safe)
        past = {k: ent[k][m] for k in ("date", "place", "surface", "dist", "pos", "jockey")}

    surface = _SURF.get(str(surface).strip(), "")
    dist_v = float(dist) if pd.notna(dist) else np.nan
    # --- horse_fuku10/30 (build_dataset.add_horse_rolling_stats と同じ窓/最小数) ---
    for window, min_periods in ((10, 3), (30, 5)):
        key = f"horse_fuku{window}"
        if past is None or len(past["pos"]) < min_periods:
            out[key] = np.nan
        else:
            pos = past["pos"][-window:]
            out[key] = float(np.mean(pos <= 3))

    # --- hist_same_cond_* (parse_kako5: 同TD + 距離±200, 着順ありのみ) ---
    if past is not None and len(past["date"]) and surface and not np.isnan(dist_v):
        sel = ((past["surface"] == surface)
               & ~np.isnan(past["dist"]) & (np.abs(past["dist"] - dist_v) <= 200)
               & ~np.isnan(past["pos"]))
        pos = past["pos"][sel]
        if len(pos):
            out["hist_same_cond_best_pos"] = float(pos.min())
            out["hist_same_cond_top3_rate"] = float((pos <= 3).mean())
            out["hist_same_cond_count"] = float(len(pos))
        else:
            out["hist_same_cond_best_pos"] = np.nan
            out["hist_same_cond_top3_rate"] = np.nan
            out["hist_same_cond_count"] = np.nan
    else:
        out["hist_same_cond_best_pos"] = np.nan
        out["hist_same_cond_top3_rate"] = np.nan
        out["hist_same_cond_count"] = np.nan

    # --- hist_same_place_best_pos (同場所, 着順ありのみ) ---
    if past is not None and len(past["date"]) and place:
        sel = (past["place"] == place) & ~np.isnan(past["pos"])
        pos = past["pos"][sel]
        out["hist_same_place_best_pos"] = float(pos.min()) if len(pos) else np.nan
    else:
        out["hist_same_place_best_pos"] = np.nan

    # --- course_* (build_master_v2: 場所|芝ダ|距離帯, NaN着順も走数に含む) ---
    band = _dist_band(dist_v)
    if past is not None:
        if len(past["date"]):
            bands = np.array([_dist_band(d) for d in past["dist"]])
            sel = (past["place"] == place) & (past["surface"] == surface) & (bands == band)
        else:
            sel = np.zeros(0, dtype=bool)
        n = int(sel.sum())
        out["course_n_prev"] = float(n)
        if n > 0:
            pos = past["pos"][sel]
            wins = float(np.nansum(pos == 1))
            top3 = float(np.nansum(pos <= 3))
            out["course_win_rate"] = wins / n
            out["course_top3_rate"] = top3 / n
        else:
            out["course_win_rate"] = np.nan
            out["course_top3_rate"] = np.nan
    else:
        # 未知馬 (新馬): 学習では cumcount=0
        out["course_n_prev"] = 0.0
        out["course_win_rate"] = np.nan
        out["course_top3_rate"] = np.nan

    # --- jockey_* (馬×騎手コード ペア累積) ---
    jc = float(jockey_code) if pd.notna(jockey_code) else np.nan
    if past is not None:
        if not np.isnan(jc) and len(past["date"]):
            sel = past["jockey"] == jc
            n = int(sel.sum())
            out["jockey_n_prev"] = float(n)
            if n > 0:
                pos = past["pos"][sel]
                out["jockey_win_rate"] = float(np.nansum(pos == 1)) / n
                out["jockey_top3_rate"] = float(np.nansum(pos <= 3)) / n
            else:
                out["jockey_win_rate"] = np.nan
                out["jockey_top3_rate"] = np.nan
        elif not np.isnan(jc):
            out["jockey_n_prev"] = 0.0
            out["jockey_win_rate"] = np.nan
            out["jockey_top3_rate"] = np.nan
        else:
            # 騎手コード不明 → ペア集計不能 (従来通り NaN)
            out["jockey_n_prev"] = np.nan
            out["jockey_win_rate"] = np.nan
            out["jockey_top3_rate"] = np.nan
    else:
        jc_ok = not np.isnan(jc)
        out["jockey_n_prev"] = 0.0 if jc_ok else np.nan
        out["jockey_win_rate"] = np.nan
        out["jockey_top3_rate"] = np.nan

    return out


def fill_history_features(df: pd.DataFrame, base: Path = BASE,
                          jockey_code_override: pd.Series | None = None,
                          trainer_code_override: pd.Series | None = None) -> dict:
    """df (parse_csv 済み weekly frame) に NUM_FEATS + CAT_FEATS を in-place で埋める。

    必要列: 馬名, 日付(YYYYMMDD str), 場所, 芝・ダ, 距離。
    任意列: 種牡馬, 年齢, 騎手, 調教師 (あるほど解決精度が上がる)。
    jockey_code_override: 検証用 — 騎手コードを名前マップでなく直接与える。
    戻り値: 統計 dict (coverage 等)。
    """
    idx, maps, meta = _load(base)
    jmap, tmap = maps["jockey"], maps["trainer"]

    n = len(df)
    num_out = {c: np.full(n, np.nan) for c in NUM_FEATS}
    jockey_code_col = np.array(["__NaN__"] * n, dtype=object)
    trainer_code_col = np.array(["__NaN__"] * n, dtype=object)
    identity_col = np.array(["bad_date"] * n, dtype=object)
    ped_col = np.full(n, np.nan)
    stats = {"hit": 0, "new": 0, "ambiguous": 0, "bad_date": 0,
             "jockey_code": 0, "trainer_code": 0, "hist_max_date": meta["max_date"],
             "hist_rows": meta["n_rows"], "hist_columns": meta["columns"],
             "hist_path": meta["path"], "maps_path": meta["maps_path"]}

    names = df["馬名"].map(_clean_name) if "馬名" in df.columns else pd.Series([""] * n)
    sires = (df["種牡馬"].map(_clean_name) if "種牡馬" in df.columns
             else pd.Series([""] * n, index=df.index))
    ages = (pd.to_numeric(df["年齢"], errors="coerce") if "年齢" in df.columns
            else pd.Series([np.nan] * n, index=df.index))
    dates = pd.to_numeric(df.get("日付"), errors="coerce")
    places = (df["場所"].astype(str).str.strip() if "場所" in df.columns
              else pd.Series([""] * n, index=df.index))
    surfs = df.get("芝・ダ", pd.Series([""] * n, index=df.index))
    dists = pd.to_numeric(df.get("距離"), errors="coerce")
    jnames = (df["騎手"].map(_clean_name) if "騎手" in df.columns
              else pd.Series([""] * n, index=df.index))
    tnames = (df["調教師"].map(_clean_name) if "調教師" in df.columns
              else pd.Series([""] * n, index=df.index))

    for i, (ix, name) in enumerate(names.items()):
        d = dates.loc[ix] if ix in dates.index else np.nan
        if pd.isna(d):
            stats["bad_date"] += 1
            continue
        race_date = int(d)

        # 騎手・調教師コード
        if jockey_code_override is not None:
            jc = pd.to_numeric(pd.Series([jockey_code_override.loc[ix]]),
                               errors="coerce").iloc[0]
        else:
            jc = jmap.get(jnames.loc[ix], np.nan)
        if trainer_code_override is not None:
            tc = pd.to_numeric(pd.Series([trainer_code_override.loc[ix]]),
                               errors="coerce").iloc[0]
        else:
            tc = tmap.get(tnames.loc[ix], np.nan)
        if pd.notna(jc):
            jockey_code_col[i] = str(int(jc))
            stats["jockey_code"] += 1
        if pd.notna(tc):
            trainer_code_col[i] = str(int(tc))
            stats["trainer_code"] += 1
        for kind, prefix, code in (
                ("jockey", "jockey", jc), ("trainer", "trainer", tc)):
            for window in (30, 90):
                num_out[f"{prefix}_fuku{window}"][i] = idx.rolling_rate(
                    kind, code, race_date, window, min_periods=5)

        age = ages.loc[ix]
        birth_year = int(race_date // 10000 - age) if pd.notna(age) else None
        ent, status = idx.resolve(name, sires.loc[ix], birth_year, race_date=race_date)
        stats[status] += 1
        identity_col[i] = status
        if status == "ambiguous":
            continue  # 従来通り NaN (安全側)
        if status == "hit" and ent is not None and ent.get("ped_id") is not None:
            ped_col[i] = float(ent["ped_id"])

        feats = compute_row_feats(ent, race_date, places.loc[ix],
                                  surfs.loc[ix], dists.loc[ix], jc)
        for c, value in feats.items():
            num_out[c][i] = value

    for c in NUM_FEATS:
        df[c] = num_out[c]
    df["騎手コード"] = jockey_code_col
    df["調教師コード"] = trainer_code_col
    # 馬ごとの同定状態 (hit / new / ambiguous / bad_date)。bundle と品質ゲートが使う。
    df[IDENTITY_STATUS_COL] = identity_col
    df[PED_ID_COL] = ped_col

    stats["coverage"] = {c: round(float(pd.Series(num_out[c]).notna().mean()), 3)
                         for c in NUM_FEATS}
    return stats
