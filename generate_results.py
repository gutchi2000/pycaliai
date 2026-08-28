"""
generate_results.py
===================
reports/pred_*.csv × data/kekka/*.csv を突合して
data/results.json (HAHO / HALO / LALO / CQC 4プラン形式) を再構築する。

使い方:
    python generate_results.py
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Iterator  # noqa: F401  (used by aggregate_cowork_bets type hints)

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE_DIR  = Path(__file__).parent
PRED_DIR  = BASE_DIR / "reports"
KEKKA_DIR = BASE_DIR / "data" / "kekka"
OUT_PATH  = BASE_DIR / "data" / "results.json"
COWORK_BETS_DIR   = BASE_DIR / "reports" / "cowork_bets"     # 個別レース形式: {date}/{race_id}.json
COWORK_OUTPUT_DIR = BASE_DIR / "reports" / "cowork_output"   # bundle 形式: {date}_bets.json
COWORK_OUT_PATH = BASE_DIR / "data" / "cowork_results.json"
WIDE_PARQUET    = BASE_DIR / "data" / "wide_payouts_2016-2025.parquet"  # 2016-2025
WIDE_WEEKLY_CSV = BASE_DIR / "data" / "kekka" / "wide_kekka.csv"        # 2026〜 (週次配置)

PLACE_CODE_TO_NAME = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
    "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉",
}


# =========================================================
# ユーティリティ
# =========================================================
def parse_date_to_key(date_str: str) -> str:
    """'2026.2.22' → '20260222'"""
    parts = str(date_str).replace("-", ".").split(".")
    if len(parts) >= 3:
        return f"{parts[0]}{parts[1].zfill(2)}{parts[2].zfill(2)}"
    return str(date_str).replace(".", "")


def parse_haitou(v) -> float:
    """払戻額（括弧付き・nan・空文字除外）を float で返す。取得不可は 0.0。"""
    s = str(v).strip()
    if s.startswith("(") or s in ("nan", "", "None"):
        return 0.0
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return 0.0


def _safe_num(v, default: float = 0.0) -> float:
    """NaN / None / 変換失敗を default に変換した float を返す。

    `float(x or 0)` は x=NaN のとき NaN を返してしまう (NaN は truthy)。
    pred CSV に NaN が混じると round() が ValueError で死ぬので、
    すべての金額系フィールドはこの関数で読む。
    """
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return default if pd.isna(f) else f


def _safe_round(v, default: int = 0) -> int:
    """NaN セーフな round + int キャスト。"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if pd.isna(f):
        return default
    return int(round(f))


def split_combos(bet_str: str) -> list[frozenset]:
    """'1-2 / 3-4' → [{1,2}, {3,4}]"""
    result = []
    if pd.isna(bet_str) or str(bet_str).strip() in ("", "nan"):
        return result
    for part in str(bet_str).split("/"):
        nums = [n for n in part.strip().split("-") if n.strip().isdigit()]
        if nums:
            result.append(frozenset(int(n) for n in nums))
    return result


def to_int(v) -> int | None:
    try:
        return int(float(str(v).strip()))
    except (ValueError, TypeError):
        return None


# =========================================================
# kekka 読み込み
# =========================================================
def load_kekka_all() -> dict[str, pd.DataFrame]:
    """date_key → kekka DataFrame のキャッシュを返す。

    対応フォーマット:
      1. 個別ファイル形式: data/kekka/{YYYYMMDD}.csv (週次)
      2. 統合ファイル形式: data/kekka/kekka_{YYYYMMDD}-{YYYYMMDD}.csv
         (日付列が yymmdd 形式で複数日付を 1 ファイルに保持)

    個別ファイルが優先 (週次運用で新しい方を反映するため)。
    """
    cache: dict[str, pd.DataFrame] = {}

    # 1. 個別 8 桁ファイル
    for f in sorted(KEKKA_DIR.glob("????????.csv")):
        try:
            df = pd.read_csv(f, encoding="cp932")
            cache[f.stem] = df
        except Exception as e:
            log.warning(f"kekka 読み込みスキップ {f.name}: {e}")
    n_indiv = len(cache)

    # 2. 統合ファイル kekka_*.csv
    n_aggr = 0
    for f in sorted(KEKKA_DIR.glob("kekka_*.csv")):
        try:
            df = pd.read_csv(f, encoding="cp932")
            if df.shape[1] < 8:
                continue
            date_col = df.columns[0]
            # 日付列を yymmdd → YYYYMMDD に正規化
            for raw_dk, sub in df.groupby(date_col):
                raw = str(raw_dk).strip()
                if not raw.isdigit():
                    continue
                if len(raw) == 6:
                    date_key = f"20{raw}"
                elif len(raw) == 8:
                    date_key = raw
                else:
                    continue
                if date_key in cache:
                    continue  # 個別ファイル優先
                cache[date_key] = sub.reset_index(drop=True)
                n_aggr += 1
        except Exception as e:
            log.warning(f"kekka 統合ファイル読み込みスキップ {f.name}: {e}")

    log.info(f"kekka 読み込み完了 (個別 {n_indiv}, 統合 {n_aggr}, 計 {len(cache)})")
    return cache


def get_race_kk(kekka_cache: dict, date_key: str, place: str, r_num: str) -> pd.DataFrame:
    kk = kekka_cache.get(date_key)
    if kk is None:
        return pd.DataFrame()
    return kk[
        (kk.iloc[:, 1].astype(str).str.strip() == place) &
        (kk.iloc[:, 2].astype(str) == r_num)
    ]


def get_top3(race_kk: pd.DataFrame) -> list[int]:
    """確定着順 1-3 の馬番リスト（馬番昇順）。三連複判定用。"""
    rows = race_kk[race_kk.iloc[:, 6].astype(str).isin(["1", "2", "3"])]
    return sorted(to_int(x) for x in rows.iloc[:, 4].tolist() if to_int(x) is not None)


def get_top2(race_kk: pd.DataFrame) -> frozenset:
    """1・2着馬番セット（馬連判定用）。着順でフィルタするため馬番の並びに依存しない。"""
    rows = race_kk[race_kk.iloc[:, 6].astype(str).isin(["1", "2"])]
    nums = [to_int(x) for x in rows.iloc[:, 4].tolist() if to_int(x) is not None]
    return frozenset(nums) if len(nums) == 2 else frozenset()


def get_winner(race_kk: pd.DataFrame) -> int | None:
    rows = race_kk[race_kk.iloc[:, 6].astype(str) == "1"].iloc[:, 4]
    return to_int(rows.iloc[0]) if len(rows) > 0 else None


def _bet_cis(settled: pd.DataFrame, n_boot: int = 2000, seed: int = 42) -> dict:
    """券種別の信頼区間。hit_ci95=Wilson (的中率)、roi_ci95=レース単位
    cluster bootstrap (実効投資加重 ROI)。race_id が無い旧入力だけ bet 単位へフォールバック。

    roi_verdict: 95%CI と控除率 80% の位置関係。
      above_takeout  = CI 下限 > 80  (真に控除率超の証拠)
      below_takeout  = CI 上限 < 80  (真に控除率未満の証拠)
      inconclusive   = CI が 80 を跨ぐ (点推定での増減判断は禁止)
    """
    import numpy as np
    n = len(settled)
    hits = int(settled["的中"].sum())
    cost = (settled["購入額"] - settled["返還"]).to_numpy(dtype=float)
    ret = settled["払戻"].to_numpy(dtype=float)

    # Wilson score interval (的中率)
    z = 1.96
    p = hits / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    hit_ci = [round(max(0.0, center - half) * 100, 1),
              round(min(1.0, center + half) * 100, 1)]

    # 同一レース内の馬券は結果が相関するため、race_id があればレースを再抽出する。
    # bet 単位 bootstrap は分散を過小評価するので、旧形式の入力だけに限定する。
    unit_cost, unit_ret = cost, ret
    if "race_id" in settled.columns:
        work = pd.DataFrame({"cost": cost, "ret": ret}, index=settled.index)
        keys = settled["race_id"].astype("string")
        missing = keys.isna() | (keys.str.strip() == "")
        if missing.any():
            keys = keys.astype(object)
            keys.loc[missing] = [f"__ticket_{i}" for i in work.index[missing]]
        work["race_id"] = keys.to_numpy()
        clusters = work.groupby("race_id", sort=False, dropna=False)[["cost", "ret"]].sum()
        unit_cost = clusters["cost"].to_numpy(dtype=float)
        unit_ret = clusters["ret"].to_numpy(dtype=float)

    rng = np.random.default_rng(seed)
    n_units = len(unit_cost)
    idx = rng.integers(0, n_units, size=(n_boot, n_units))
    boot_cost = unit_cost[idx].sum(axis=1)
    boot_ret = unit_ret[idx].sum(axis=1)
    valid = boot_cost > 0
    rois = np.full(n_boot, np.nan)
    rois[valid] = boot_ret[valid] / boot_cost[valid] * 100
    lo, hi = (float(np.nanpercentile(rois, 2.5)),
              float(np.nanpercentile(rois, 97.5)))
    verdict = ("above_takeout" if lo > 80.0
               else "below_takeout" if hi < 80.0
               else "inconclusive")
    return {
        "hit_ci95": hit_ci,
        "roi_ci95": [round(lo, 1), round(hi, 1)],
        "roi_verdict": verdict,
    }


def get_cancelled(race_kk: pd.DataFrame) -> set[int]:
    """出走取消・競走除外・競走中止の馬番セット。

    TARGET の kekka CSV は取消・除外・中止馬を着順 "0" で出力する
    (2026 年全 kekka で 138 行確認、文字列マーカーは存在しない)。
    現実の JRA では取消馬絡みの馬券は返還されるため、これを全額損失と
    計上すると券種別 ROI が系統的に過小になる (audit 2026-06-11)。
    """
    rows = race_kk[race_kk.iloc[:, 6].astype(str).str.strip() == "0"]
    return {to_int(x) for x in rows.iloc[:, 4].tolist() if to_int(x) is not None}


def get_payout_rengo(race_kk: pd.DataFrame) -> float:
    """馬連配当（per 100円）。"""
    col = "馬連"
    if col not in race_kk.columns:
        return 0.0
    vals = race_kk[col].dropna()
    vals = vals[~vals.astype(str).str.startswith("(")]
    if len(vals) == 0:
        return 0.0
    return parse_haitou(vals.iloc[0])


def get_payout_sanrenpuku(race_kk: pd.DataFrame) -> float:
    """三連複配当（per 100円）。"""
    col = "３連複"
    if col not in race_kk.columns:
        return 0.0
    vals = race_kk[col].dropna()
    vals = vals[~vals.astype(str).str.startswith("(")]
    if len(vals) == 0:
        return 0.0
    return parse_haitou(vals.iloc[0])


def get_payout_tansho(race_kk: pd.DataFrame) -> float:
    """単勝配当（per 100円）。"""
    col = "単勝配当"
    if col not in race_kk.columns:
        return 0.0
    vals = race_kk[col].dropna()
    vals = vals[~vals.astype(str).str.startswith("(")]
    if len(vals) == 0:
        return 0.0
    return parse_haitou(vals.iloc[0])


def get_payout_fukusho(race_kk: pd.DataFrame, horse_num: int) -> float:
    """指定馬の複勝配当（per 100円）。"""
    col = "複勝配当"
    if col not in race_kk.columns:
        return 0.0
    row = race_kk[race_kk.iloc[:, 4].astype(str).str.strip() == str(horse_num)]
    if len(row) == 0:
        return 0.0
    v = str(row[col].iloc[0]).strip()
    if v.startswith("("):
        v = v[1:-1]
    return parse_haitou(v)


# === Cowork 集計用: 馬単 / 三連単 / ワイド の追加ヘルパー ===
def get_top3_ordered(race_kk: pd.DataFrame) -> list[int]:
    """確定着順 1-3 の馬番を着順 (1着, 2着, 3着) で返す。"""
    rows = race_kk[race_kk.iloc[:, 6].astype(str).isin(["1", "2", "3"])].copy()
    if len(rows) < 3:
        return []
    rows["_pos"] = rows.iloc[:, 6].astype(int)
    rows = rows.sort_values("_pos")
    return [to_int(x) for x in rows.iloc[:, 4].tolist() if to_int(x) is not None]


def get_payout_umatan(race_kk: pd.DataFrame) -> float:
    """馬単配当（per 100円）。"""
    col = "馬単"
    if col not in race_kk.columns:
        return 0.0
    vals = race_kk[col].dropna()
    vals = vals[~vals.astype(str).str.startswith("(")]
    return parse_haitou(vals.iloc[0]) if len(vals) > 0 else 0.0


def get_payout_sanrentan(race_kk: pd.DataFrame) -> float:
    """三連単配当（per 100円）。"""
    col = "３連単"
    if col not in race_kk.columns:
        return 0.0
    vals = race_kk[col].dropna()
    vals = vals[~vals.astype(str).str.startswith("(")]
    return parse_haitou(vals.iloc[0]) if len(vals) > 0 else 0.0


# =========================================================
# ワイド配当キャッシュ
# =========================================================
#   key   : (date_key:str8, place_name:str, race_no:str)
#   value : dict[frozenset({i,j}) -> payout_per_100:int]
# =========================================================
_WIDE_CACHE: dict[tuple[str, str, str], dict] | None = None


def _parse_wide_combos(combo_str: str) -> dict:
    """'11-14 \\220 (1)/ 04-11 \\870 (12)/ 04-14 \\400 (4)' を
    {frozenset({11,14}): 220, ...} に変換。"""
    out: dict = {}
    if pd.isna(combo_str):
        return out
    for chunk in str(combo_str).split("/"):
        s = chunk.strip()
        if not s:
            continue
        # フォーマット: "{i}-{j} \\{payout} ({pop})" or "{i}-{j} \\{payout}"
        # 円記号は \\ or ¥ や全角の可能性あり
        s2 = s.replace("\\", " ").replace("¥", " ").replace("￥", " ")
        # 括弧 () 内の人気を除去
        if "(" in s2:
            s2 = s2[: s2.index("(")]
        parts = s2.split()
        if len(parts) < 2:
            continue
        pair_str = parts[0]
        try:
            i_s, j_s = pair_str.split("-")
            i, j = int(i_s), int(j_s)
        except ValueError:
            continue
        try:
            payout = int(parts[1].replace(",", ""))
        except ValueError:
            continue
        out[frozenset((i, j))] = payout
    return out


def load_wide_payouts() -> dict[tuple[str, str, str], dict]:
    """全期間のワイド配当を統合キャッシュとして返す。

    ソース:
      - data/wide_payouts_2016-2025.parquet  (race_id ベース、2016-2025)
      - data/kekka/wide_kekka.csv            (年月日 + 場所 + R ベース、2026〜週次)
    """
    cache: dict[tuple[str, str, str], dict] = {}

    # 1. parquet (2016-2025)
    if WIDE_PARQUET.exists():
        try:
            wp = pd.read_parquet(WIDE_PARQUET)
            for _, row in wp.iterrows():
                date_key = str(row["date"]).zfill(8)
                place    = str(row["place"]).strip()
                race_no  = str(int(row["race_no"]))
                d: dict = {}
                for i_col, j_col, p_col in [
                    ("w1_i", "w1_j", "w1_pay"),
                    ("w2_i", "w2_j", "w2_pay"),
                    ("w3_i", "w3_j", "w3_pay"),
                ]:
                    try:
                        i, j, p = int(row[i_col]), int(row[j_col]), int(row[p_col])
                    except (ValueError, TypeError):
                        continue
                    d[frozenset((i, j))] = p
                if d:
                    cache[(date_key, place, race_no)] = d
            log.info(f"wide_payouts.parquet: {sum(1 for k in cache if k[0].startswith('20'))} races loaded")
        except Exception as e:
            log.warning(f"wide_payouts.parquet 読み込み失敗: {e}")

    # 2. wide_kekka.csv (2026〜週次)
    if WIDE_WEEKLY_CSV.exists():
        try:
            wk = pd.read_csv(WIDE_WEEKLY_CSV, encoding="cp932", header=None)
            # 列: 0=year 1=month 2=day 3=place 4=race_no 5=class 6=芝ダ 7=距離 8=頭数 9=combo
            n_added = 0
            for _, row in wk.iterrows():
                try:
                    y = int(row[0]); m = int(row[1]); d = int(row[2])
                except (ValueError, TypeError):
                    continue
                date_key = f"{y:04d}{m:02d}{d:02d}"
                place    = str(row[3]).strip()
                race_no  = str(int(row[4]))
                combos   = _parse_wide_combos(row[9])
                if combos:
                    cache[(date_key, place, race_no)] = combos
                    n_added += 1
            log.info(f"wide_kekka.csv (weekly): {n_added} races loaded")
        except Exception as e:
            log.warning(f"wide_kekka.csv 読み込み失敗: {e}")

    return cache


def get_wide_cache() -> dict[tuple[str, str, str], dict]:
    """グローバルキャッシュをレイジロード。"""
    global _WIDE_CACHE
    if _WIDE_CACHE is None:
        _WIDE_CACHE = load_wide_payouts()
    return _WIDE_CACHE


def get_payout_wide(date_key: str, place: str, race_no: str,
                    pair: frozenset) -> float:
    """ワイド (i, j) の配当 (per 100 円) を返す。無ければ 0。"""
    cache = get_wide_cache()
    d = cache.get((date_key, place, race_no))
    if not d:
        return 0.0
    return float(d.get(pair, 0))


# === Cowork bet 照合 ===
def match_cowork_bet(bet: dict, race_kk: pd.DataFrame,
                     date_key: str = "", place: str = "",
                     race_no: str = "") -> tuple[bool, float, float]:
    """Cowork bet (1 件) を kekka と照合して (hit, payout_per_100, refund_ratio) を返す。

    payout_per_100 は「100円ベース配当」÷「点数」(複数点なら平均化)。
    ワイドは kekka に配当列が無いため、的中時も payout=0 を返す
    (ROI 計算で正の貢献は無いが、的中 1 はカウントされる)。

    refund_ratio: 取消・除外馬 (着順=0) を含む組の点数比率 (0.0〜1.0)。
    現実の JRA では該当組は返還されるので、集計側で
    実効投資 = 購入額 × (1 - refund_ratio) として扱う。
    """
    btype = bet.get("馬券種")
    sel = str(bet.get("買い目", "")).strip()
    # 馬単/三連単の買い目は "4→8" / "1→2→3" のように矢印区切り。
    # ペア/トリオ判定は "-" 区切り前提なので正規化する（カンマ区切りは温存）。
    sel = sel.replace("→", "-").replace("＞", "-").replace(">", "-").replace("－", "-")
    if not sel or not btype:
        return False, 0.0, 0.0

    top1 = get_winner(race_kk)
    top2 = get_top2(race_kk)
    top3 = get_top3(race_kk)
    top3_ordered = get_top3_ordered(race_kk)
    cancelled = get_cancelled(race_kk)

    def _split_pairs(unordered: bool):
        out = []
        for c in sel.split(","):
            parts = c.strip().split("-")
            if len(parts) == 2:
                try:
                    a, b = int(parts[0]), int(parts[1])
                    out.append(frozenset((a, b)) if unordered else (a, b))
                except ValueError:
                    continue
        return out

    def _split_trios(unordered: bool):
        out = []
        for c in sel.split(","):
            parts = c.strip().split("-")
            if len(parts) == 3:
                try:
                    nums = [int(p) for p in parts]
                    out.append(frozenset(nums) if unordered else tuple(nums))
                except ValueError:
                    continue
        return out

    def _refund_ratio(combos) -> float:
        """組リストのうち取消馬を含む組の比率 (= 返還される投資比率)。"""
        if not combos or not cancelled:
            return 0.0
        n_ref = sum(1 for c in combos if any(x in cancelled for x in c))
        return n_ref / len(combos)

    if btype == "単勝":
        try:
            n = int(sel)
        except ValueError:
            return False, 0.0, 0.0
        if n in cancelled:
            return False, 0.0, 1.0   # 全額返還
        hit = (n == top1) and (top1 is not None)
        return hit, (get_payout_tansho(race_kk) if hit else 0.0), 0.0

    if btype == "複勝":
        try:
            n = int(sel)
        except ValueError:
            return False, 0.0, 0.0
        if n in cancelled:
            return False, 0.0, 1.0   # 全額返還
        hit = (n in top3) and (len(top3) >= 1)
        return hit, (get_payout_fukusho(race_kk, n) if hit else 0.0), 0.0

    if btype == "馬連":
        pairs = _split_pairs(unordered=True)
        if not pairs or not top2:
            return False, 0.0, 0.0
        rr = _refund_ratio(pairs)
        hits = sum(1 for p in pairs if p == top2)
        if hits:
            return True, get_payout_rengo(race_kk) / len(pairs), rr
        return False, 0.0, rr

    if btype == "馬単":
        pairs = _split_pairs(unordered=False)
        if not pairs or len(top3_ordered) < 2:
            return False, 0.0, 0.0
        rr = _refund_ratio(pairs)
        target = (top3_ordered[0], top3_ordered[1])
        hits = sum(1 for p in pairs if p == target)
        if hits:
            return True, get_payout_umatan(race_kk) / len(pairs), rr
        return False, 0.0, rr

    if btype == "ワイド":
        pairs = _split_pairs(unordered=True)
        if not pairs or len(top3) < 3:
            return False, 0.0, 0.0
        rr = _refund_ratio(pairs)
        top3_set = set(top3)
        hit_pairs = [p for p in pairs if all(x in top3_set for x in p)]
        if not hit_pairs:
            return False, 0.0, rr
        # ワイド配当を wide_payouts キャッシュから取得
        # (2016-2025: parquet, 2026〜: wide_kekka.csv)
        total_payout = sum(get_payout_wide(date_key, place, race_no, p) for p in hit_pairs)
        # 点数で割って per-100 円配当として返す (match_cowork_bet の他種と同じ流儀)
        return True, total_payout / len(pairs), rr

    if btype == "三連複":
        trios = _split_trios(unordered=True)
        if not trios or len(top3) < 3:
            return False, 0.0, 0.0
        rr = _refund_ratio(trios)
        target_fs = frozenset(top3)
        hits = sum(1 for t in trios if t == target_fs)
        if hits:
            return True, get_payout_sanrenpuku(race_kk) / len(trios), rr
        return False, 0.0, rr

    if btype == "三連単":
        trios = _split_trios(unordered=False)
        if not trios or len(top3_ordered) < 3:
            return False, 0.0, 0.0
        rr = _refund_ratio(trios)
        target = tuple(top3_ordered)
        hits = sum(1 for t in trios if t == target)
        if hits:
            return True, get_payout_sanrentan(race_kk) / len(trios), rr
        return False, 0.0, rr

    return False, 0.0, 0.0


# =========================================================
# pred CSV 読み込み
# =========================================================
def load_pred_all() -> pd.DataFrame:
    """reports/pred_*.csv を全結合して返す。"""
    files = sorted(PRED_DIR.glob("pred_????????.csv"))
    if not files:
        raise FileNotFoundError(f"pred CSVが見つかりません: {PRED_DIR}/pred_*.csv")
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f, encoding="utf-8-sig"))
        except Exception:
            try:
                dfs.append(pd.read_csv(f, encoding="utf-8"))
            except Exception as e:
                log.warning(f"pred CSV スキップ {f.name}: {e}")
    pred = pd.concat(dfs, ignore_index=True)
    log.info(f"pred {len(files)} ファイル / {len(pred)} 行 読み込み完了")
    return pred


# =========================================================
# プラン別レース集計
# =========================================================
def calc_plan_races(pred: pd.DataFrame, kekka_cache: dict,
                    plan: str) -> list[dict]:
    """plan = 'HAHO' | 'HALO' | 'LALO' | 'CQC'"""
    target_col = f"{plan}_戦略対象"
    if target_col not in pred.columns:
        log.warning(f"{target_col} 列が pred CSV に存在しません。")
        return []

    target_pred = pred[pred[target_col].astype(str).str.strip() == "✅"].copy()
    if target_pred.empty:
        return []

    records: list[dict] = []

    for (date_raw, place, r_num), grp in target_pred.groupby(["日付", "場所", "R"]):
        place   = str(place).strip()
        r_num   = str(r_num)
        date_key = parse_date_to_key(date_raw)

        race_kk = get_race_kk(kekka_cache, date_key, place, r_num)
        if race_kk.empty:
            continue  # kekka 未登録（未来日程 or データなし）

        hon_rows = grp[grp["印"].astype(str) == "◎"]
        if hon_rows.empty:
            continue
        hon = hon_rows.iloc[0]
        h1  = to_int(hon.get("馬番"))
        if h1 is None:
            continue

        top3   = get_top3(race_kk)
        winner = get_winner(race_kk)

        rec: dict = {
            "日付":   date_raw,
            "場所":   place,
            "R":      int(r_num),
            "クラス": str(hon.get("クラス", "")),
        }

        if plan == "HAHO":
            # 馬連
            ren_inv  = _safe_num(hon.get("HAHO_馬連_購入額"))
            ren_bets = split_combos(str(hon.get("HAHO_馬連_買い目", "")))
            top2     = get_top2(race_kk)
            ren_hit  = any(c == top2 for c in ren_bets) if top2 and ren_bets else False
            ren_ret  = 0.0
            if ren_hit:
                odds = get_payout_rengo(race_kk)
                per  = ren_inv / len(ren_bets) if ren_bets else 0
                ren_ret = per * odds / 100

            # 三連複
            san_inv  = _safe_num(hon.get("HAHO_三連複_購入額"))
            san_bets = split_combos(str(hon.get("HAHO_三連複_買い目", "")))
            top3_fs  = frozenset(top3) if len(top3) == 3 else frozenset()
            san_hit  = any(c == top3_fs for c in san_bets) if top3_fs and san_bets else False
            san_ret  = 0.0
            if san_hit:
                odds    = get_payout_sanrenpuku(race_kk)
                per     = san_inv / len(san_bets) if san_bets else 0
                san_ret = per * odds / 100

            rec.update({
                "馬連_投資": ren_inv, "馬連_払戻": _safe_round(ren_ret), "馬連_的中": int(ren_hit),
                "三連複_投資": san_inv, "三連複_払戻": _safe_round(san_ret), "三連複_的中": int(san_hit),
                "総投資": ren_inv + san_inv,
                "総払戻": _safe_round(ren_ret + san_ret),
                "収支":   _safe_round(ren_ret + san_ret - ren_inv - san_inv),
            })

        elif plan == "HALO":
            san_inv  = _safe_num(hon.get("HALO_三連複_購入額"))
            san_bets = split_combos(str(hon.get("HALO_三連複_買い目", "")))
            top3_fs  = frozenset(top3) if len(top3) == 3 else frozenset()
            san_hit  = any(c == top3_fs for c in san_bets) if top3_fs and san_bets else False
            san_ret  = 0.0
            if san_hit:
                odds    = get_payout_sanrenpuku(race_kk)
                per     = san_inv / len(san_bets) if san_bets else 0
                san_ret = per * odds / 100

            rec.update({
                "三連複_投資": san_inv, "三連複_払戻": _safe_round(san_ret), "三連複_的中": int(san_hit),
                "総投資": san_inv,
                "総払戻": _safe_round(san_ret),
                "収支":   _safe_round(san_ret - san_inv),
            })

        elif plan == "LALO":
            fuku_inv = _safe_num(hon.get("LALO_複勝_購入額"))
            bet_horse = to_int(hon.get("LALO_複勝_買い目"))
            fuku_hit  = (bet_horse in top3) if bet_horse and top3 else False
            fuku_ret  = 0.0
            if fuku_hit and bet_horse:
                odds     = get_payout_fukusho(race_kk, bet_horse)
                fuku_ret = fuku_inv * odds / 100

            rec.update({
                "複勝_投資": fuku_inv, "複勝_払戻": _safe_round(fuku_ret), "複勝_的中": int(fuku_hit),
                "総投資": fuku_inv,
                "総払戻": _safe_round(fuku_ret),
                "収支":   _safe_round(fuku_ret - fuku_inv),
            })

        elif plan == "CQC":
            tan_inv   = _safe_num(hon.get("CQC_単勝_購入額"))
            bet_horse = to_int(hon.get("CQC_単勝_買い目"))
            tan_hit   = (bet_horse == winner) if bet_horse and winner else False
            tan_ret   = 0.0
            if tan_hit:
                odds    = get_payout_tansho(race_kk)
                tan_ret = tan_inv * odds / 100

            rec.update({
                "単勝_投資": tan_inv, "単勝_払戻": _safe_round(tan_ret), "単勝_的中": int(tan_hit),
                "総投資": tan_inv,
                "総払戻": _safe_round(tan_ret),
                "収支":   _safe_round(tan_ret - tan_inv),
            })

        elif plan == "TRIPLE":
            # 三連複◎◯▲
            san_inv  = _safe_num(hon.get("TRIPLE_三連複_購入額"))
            san_bets = split_combos(str(hon.get("TRIPLE_三連複_買い目", "")))
            san_hit  = (any(c <= set(top3) for c in san_bets)) if san_bets and top3 else False
            san_ret  = 0.0
            if san_hit:
                odds    = get_payout_sanrenpuku(race_kk)
                san_ret = san_inv * odds / 100
            # 複勝◎
            fuku_inv  = _safe_num(hon.get("TRIPLE_複勝_購入額"))
            bet_horse = to_int(hon.get("TRIPLE_複勝_買い目"))
            fuku_hit  = (bet_horse in top3) if bet_horse and top3 else False
            fuku_ret  = 0.0
            if fuku_hit and bet_horse:
                odds     = get_payout_fukusho(race_kk, bet_horse)
                fuku_ret = fuku_inv * odds / 100

            rec.update({
                "三連複_投資": san_inv,  "三連複_払戻": _safe_round(san_ret),  "三連複_的中": int(san_hit),
                "複勝_投資":   fuku_inv, "複勝_払戻":   _safe_round(fuku_ret), "複勝_的中":   int(fuku_hit),
                "総投資": san_inv + fuku_inv,
                "総払戻": _safe_round(san_ret + fuku_ret),
                "収支":   _safe_round(san_ret + fuku_ret - san_inv - fuku_inv),
            })

        records.append(rec)

    log.info(f"{plan}: {len(records)} レース集計完了")
    return records


# =========================================================
# サマリー生成
# =========================================================
def build_summary(plan: str, records: list[dict]) -> dict:
    if not records:
        return {
            "total": {"races": 0, "bet": 0, "ret": 0, "pnl": 0, "roi": 0},
            "by_type": {}, "by_place": [], "weekly": [], "races": [],
        }

    df = pd.DataFrame(records)
    df["日付_dt"] = pd.to_datetime(df["日付"].apply(parse_date_to_key), format="%Y%m%d", errors="coerce")
    df["週"]      = df["日付_dt"].dt.to_period("W").apply(lambda p: str(p.end_time.date()))

    total_bet = float(df["総投資"].sum())
    total_ret = float(df["総払戻"].sum())

    # 馬券種別
    if plan == "HAHO":
        type_keys = {"馬連": ("馬連_投資", "馬連_払戻", "馬連_的中"),
                     "三連複": ("三連複_投資", "三連複_払戻", "三連複_的中")}
    elif plan in ("HALO",):
        type_keys = {"三連複": ("三連複_投資", "三連複_払戻", "三連複_的中")}
    elif plan == "LALO":
        type_keys = {"複勝": ("複勝_投資", "複勝_払戻", "複勝_的中")}
    elif plan == "TRIPLE":
        type_keys = {"三連複": ("三連複_投資", "三連複_払戻", "三連複_的中"),
                     "複勝":   ("複勝_投資",   "複勝_払戻",   "複勝_的中")}
    else:  # CQC
        type_keys = {"単勝": ("単勝_投資", "単勝_払戻", "単勝_的中")}

    by_type: dict = {}
    for tk, (inv_c, ret_c, hit_c) in type_keys.items():
        bet = float(df[inv_c].sum()) if inv_c in df.columns else 0
        ret = float(df[ret_c].sum()) if ret_c in df.columns else 0
        hit = int(df[hit_c].sum())   if hit_c in df.columns else 0
        n   = int((df[inv_c] > 0).sum()) if inv_c in df.columns else 0
        by_type[tk] = {
            "bet": int(bet), "ret": int(ret),
            "roi": round(ret / bet * 100, 1) if bet > 0 else 0,
            "hit": hit, "races": n,
            "hit_rate": round(hit / n * 100, 1) if n > 0 else 0,
        }

    # 週次
    wdf = df.groupby("週", sort=True).agg(
        レース数=("総投資", "count"),
        総投資=("総投資", "sum"),
        総払戻=("総払戻", "sum"),
    ).reset_index()
    wdf["ROI"] = (wdf["総払戻"] / wdf["総投資"] * 100).round(1)
    weekly = wdf.rename(columns={"週": "週", "レース数": "レース数", "総投資": "総投資",
                                  "総払戻": "総払戻", "ROI": "ROI"}).to_dict("records")

    # 会場別
    bpdf = df.groupby("場所", sort=True).agg(
        レース数=("総投資", "count"),
        総投資=("総投資", "sum"),
        総払戻=("総払戻", "sum"),
    ).reset_index()
    bpdf["ROI"] = (bpdf["総払戻"] / bpdf["総投資"] * 100).round(1)
    bpdf["収支"] = bpdf["総払戻"] - bpdf["総投資"]
    by_place = bpdf.to_dict("records")

    # 個別レース（日付降順）
    sort_cols = ["日付_dt", "R"]
    races_out = df.sort_values(sort_cols, ascending=[False, True]).drop(
        columns=["日付_dt", "週"], errors="ignore"
    ).to_dict("records")

    return {
        "total": {
            "races": len(records),
            "bet":   int(total_bet),
            "ret":   int(total_ret),
            "pnl":   int(total_ret - total_bet),
            "roi":   round(total_ret / total_bet * 100, 1) if total_bet > 0 else 0,
        },
        "by_type":  by_type,
        "by_place": by_place,
        "weekly":   weekly,
        "races":    races_out,
    }


# =========================================================
# Cowork 集計
# =========================================================
def parse_race_id_16(rid: str) -> dict | None:
    """16桁 race_id を分解。例: '2026042606010109' →
    {date_key, place_code, place_name, kai, nichi, race_no}"""
    rid = str(rid).strip()
    if len(rid) < 16 or not rid[:16].isdigit():
        return None
    return {
        "date_key":   rid[:8],
        "place_code": rid[8:10],
        "place_name": PLACE_CODE_TO_NAME.get(rid[8:10], ""),
        "kai":        rid[10:12],
        "nichi":      rid[12:14],
        "race_no":    str(int(rid[14:16])),  # "01" → "1" に正規化
    }


def _iter_cowork_race_dicts() -> "Iterator[tuple[str, dict, str]]":
    """全 Cowork 入力ソースから (date_key, race_dict, source_label) を順に yield する。

    対応フォーマット:
      1. 個別レース形式: reports/cowork_bets/{date}/{race_id}.json  (旧)
      2. bundle 形式:    reports/cowork_output/{date}_bets.json    (Cowork Anthropic Desktop 現行)

    同じ (date, race_id) が両ソースに存在する場合は bundle を優先。
    """
    seen: set[tuple[str, str]] = set()

    # --- bundle 形式 (現行) を先に処理して優先 ---
    if COWORK_OUTPUT_DIR.exists():
        for jf in sorted(COWORK_OUTPUT_DIR.glob("*_bets.json")):
            stem = jf.stem
            if not stem.endswith("_bets"):
                continue
            date_key = stem[:-len("_bets")]
            if not (date_key.isdigit() and len(date_key) == 8):
                continue
            try:
                with open(jf, encoding="utf-8") as f:
                    bundle = json.load(f)
            except Exception as e:
                log.warning(f"cowork_output 読み込み失敗 {jf.name}: {e}")
                continue
            # wrapper 形式 ({"bets": [...], "grade_scope": [...]}) と
            # レガシー (top-level array) の両対応
            if isinstance(bundle, dict) and "bets" in bundle:
                bundle = bundle["bets"]
            if not isinstance(bundle, list):
                log.warning(f"cowork_output 形式不正 (list/wrapper 期待) {jf.name}")
                continue
            for race in bundle:
                if not isinstance(race, dict):
                    continue
                rid = str(race.get("race_id", "")).strip()
                if not rid:
                    continue
                key = (date_key, rid)
                if key in seen:
                    continue
                seen.add(key)
                yield date_key, race, f"cowork_output/{jf.name}"

    # --- 個別レース形式 (旧) を後追い ---
    if COWORK_BETS_DIR.exists():
        for date_dir in sorted(COWORK_BETS_DIR.iterdir()):
            if not date_dir.is_dir() or not date_dir.name.isdigit():
                continue
            date_key = date_dir.name
            for jf in sorted(date_dir.glob("*.json")):
                try:
                    with open(jf, encoding="utf-8") as f:
                        cd = json.load(f)
                except Exception as e:
                    log.warning(f"cowork_bets 読み込み失敗 {jf.name}: {e}")
                    continue
                rid = str(cd.get("race_id", jf.stem)).strip()
                key = (date_key, rid)
                if key in seen:
                    continue
                seen.add(key)
                yield date_key, cd, f"cowork_bets/{date_key}/{jf.name}"


def aggregate_cowork_bets(kekka_cache: dict) -> dict:
    """Cowork 由来の買い目 (per-race ファイル + bundle ファイル) を全部読んで
    kekka と突合して集計サマリを返す。

    入力ソース:
      - reports/cowork_bets/{YYYYMMDD}/{race_id}.json (旧形式)
      - reports/cowork_output/{YYYYMMDD}_bets.json    (Cowork Anthropic Desktop bundle)

    Returns:
        {
          "generated_at": "...",
          "total": {"races": int, "bet_count": int, "bet": int, "ret": int,
                    "pnl": int, "roi": float, "hit_count": int, "hit_rate": float},
          "by_type": {"単勝": {...}, "複勝": {...}, ...},
          "by_place": [...], "weekly": [...], "races": [...],
        }
    """
    if not COWORK_BETS_DIR.exists() and not COWORK_OUTPUT_DIR.exists():
        log.info("cowork_bets/ / cowork_output/ どちらも無し、スキップ")
        return {"generated_at": pd.Timestamp.now().isoformat(),
                "total": {"races": 0, "bet_count": 0, "bet": 0, "ret": 0,
                          "pnl": 0, "roi": 0, "hit_count": 0, "hit_rate": 0},
                "by_type": {}, "by_place": [], "weekly": [], "races": []}

    races_out: list[dict] = []
    bet_records: list[dict] = []   # 馬券単位 (集計用)

    for date_key, cd, _source in _iter_cowork_race_dicts():
        rid = cd.get("race_id", "")
        parsed = parse_race_id_16(rid)
        if not parsed:
            log.warning(f"race_id 解析失敗: {rid}")
            continue

        place_name = parsed["place_name"]
        r_num = parsed["race_no"]
        race_kk = get_race_kk(kekka_cache, date_key, place_name, r_num)
        if race_kk.empty:
            log.info(f"kekka 未到達 (未開催 or 取得待ち): {date_key} {place_name} {r_num}R")
            race_kk = pd.DataFrame()

        bets = cd.get("bets", []) or []
        race_bet = 0.0
        race_ret = 0.0
        race_hits = 0
        for b in bets:
            amount = _safe_num(b.get("購入額"))
            btype = b.get("馬券種", "?")
            sel = str(b.get("買い目", ""))
            hit, payout_per_100, refund_ratio = (False, 0.0, 0.0)
            settlement = "未開催" if race_kk.empty else "確定"
            if not race_kk.empty:
                try:
                    hit, payout_per_100, refund_ratio = match_cowork_bet(
                        b, race_kk,
                        date_key=date_key, place=place_name, race_no=str(r_num),
                    )
                except Exception as e:
                    log.warning(f"bet 照合失敗 {rid} {btype} {sel}: {e}")
                    settlement = "照合失敗"

            # 取消・除外馬を含む組は返還 (実効投資から除く)。全額損失計上だと
            # 券種別 ROI が系統的に過小になる (audit 2026-06-11)
            # 未開催・照合例外は払戻0の損失にせず、決済対象から完全に除外する。
            refund = ret = 0.0
            if settlement == "確定":
                refund = amount * refund_ratio
                eff_amount = amount - refund
                ret = (amount * payout_per_100 / 100.0) if hit else 0.0
                race_bet += eff_amount
                race_ret += ret
                if hit:
                    race_hits += 1

            bet_records.append({
                "date":     date_key,
                "race_id":  rid,
                "場所":     place_name,
                "R":        r_num,
                "馬券種":   btype,
                "買い目":   sel,
                "購入額":   _safe_round(amount),
                "返還":     _safe_round(refund),
                "払戻":     _safe_round(ret),
                "的中":     int(hit),
                "決着":     settlement,
            })

        races_out.append({
            "date":        date_key,
            "race_id":     rid,
            "場所":        place_name,
            "R":           r_num,
            "race_label":  cd.get("race_label", ""),
            "race_nature": cd.get("race_nature", ""),
            "race_reason": cd.get("race_reason", ""),
            "総投資":      _safe_round(race_bet),
            "総払戻":      _safe_round(race_ret),
            "収支":        _safe_round(race_ret - race_bet),
            "点数":        len(bets),
            "的中点数":    race_hits,
            "決着":        "未開催" if race_kk.empty else "確定",
        })

    # ── サマリ集計 ──
    total_bet  = sum(r["総投資"] for r in races_out)
    total_ret  = sum(r["総払戻"] for r in races_out)
    n_races    = len(races_out)
    n_pass     = sum(1 for r in races_out if r["点数"] == 0)
    n_settled  = sum(1 for r in races_out if r["決着"] == "確定")
    settled_records = [b for b in bet_records if b["決着"] == "確定"]
    bet_count  = len(settled_records)
    hit_count  = sum(b["的中"] for b in settled_records)

    # 馬券種別
    by_type: dict = {}
    if bet_records:
        bdf = pd.DataFrame(bet_records)
        for btype, grp in bdf.groupby("馬券種"):
            settled = grp[grp["決着"] == "確定"]
            # 実効投資 = 購入額 − 返還 (取消馬絡みの組は投資から除外)
            bet = int(settled["購入額"].sum() - settled["返還"].sum())
            ret = int(settled["払戻"].sum())
            hits = int(settled["的中"].sum())
            n = len(settled)
            by_type[btype] = {
                "bet": bet, "ret": ret,
                "pnl": ret - bet,
                "roi": round(ret / bet * 100, 1) if bet > 0 else 0,
                "hit": hits, "races": n,
                "hit_rate": round(hits / n * 100, 1) if n > 0 else 0,
            }
            # --- 信頼区間 (audit 2026-06-11: 小標本での券種ポリシーフリップ防止) ---
            # 単勝 30bets ROI120.8% → 445bets で 96.3% に回帰した事故の構造対策。
            # 規律: boost/全廃などのポリシー変更は「roi_ci95 が控除率 (≈80%) を
            # 片側に外れたときだけ」行う。点推定での判断は禁止。
            # CI の標本単位はレース。10枚ではなく10レースを最低条件にする。
            n_ci_units = (settled["race_id"].nunique()
                          if "race_id" in settled.columns else n)
            if n_ci_units >= 10:
                by_type[btype].update(_bet_cis(settled))

    # 会場別 / 週次 (確定レースのみ集計)
    by_place: list = []
    weekly: list = []
    if races_out:
        rdf = pd.DataFrame(races_out)
        rdf = rdf[rdf["決着"] == "確定"].copy()
        rdf["日付_dt"] = pd.to_datetime(rdf["date"], format="%Y%m%d", errors="coerce")
        # 会場別
        bpdf = rdf.groupby("場所", sort=True).agg(
            レース数=("race_id", "count"),
            総投資=("総投資", "sum"),
            総払戻=("総払戻", "sum"),
        ).reset_index()
        bpdf["ROI"] = (bpdf["総払戻"] / bpdf["総投資"] * 100).round(1)
        bpdf["収支"] = bpdf["総払戻"] - bpdf["総投資"]
        by_place = bpdf.to_dict("records")
        # 週次
        rdf["週"] = rdf["日付_dt"].dt.to_period("W").apply(
            lambda p: str(p.end_time.date()) if pd.notna(p) else ""
        )
        wdf = rdf.groupby("週", sort=True).agg(
            レース数=("race_id", "count"),
            総投資=("総投資", "sum"),
            総払戻=("総払戻", "sum"),
        ).reset_index()
        wdf["ROI"] = (wdf["総払戻"] / wdf["総投資"] * 100).round(1)
        weekly = wdf.to_dict("records")

    return {
        "generated_at": pd.Timestamp.now().isoformat(),
        "total": {
            "races":     n_races,
            "settled":   n_settled,
            "見送り":    n_pass,
            "bet_count": bet_count,
            "bet":       int(total_bet),
            "ret":       int(total_ret),
            "pnl":       int(total_ret - total_bet),
            "roi":       round(total_ret / total_bet * 100, 1) if total_bet > 0 else 0,
            "hit_count": hit_count,
            "hit_rate":  round(hit_count / bet_count * 100, 1) if bet_count > 0 else 0,
        },
        "by_type":  by_type,
        "by_place": by_place,
        "weekly":   weekly,
        "races":    sorted(races_out, key=lambda r: (r["date"], r["場所"], int(r["R"] or 0)),
                           reverse=True),
        "bets":     bet_records,
    }


# =========================================================
# メイン
# =========================================================
def main() -> None:
    pred = load_pred_all()
    kekka_cache = load_kekka_all()

    result = {"generated_at": pd.Timestamp.now().isoformat()}
    for plan in ["HAHO", "HALO", "LALO", "CQC", "TRIPLE"]:
        records = calc_plan_races(pred, kekka_cache, plan)
        result[plan] = build_summary(plan, records)

    # アトミック書込 (tmp→replace)。クラッシュ時に破損 JSON が残って
    # そのまま commit/HF 同期される事故を防ぐ (audit 2026-06-11)
    _tmp = OUT_PATH.with_suffix(".tmp")
    with open(_tmp, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    _tmp.replace(OUT_PATH)
    log.info(f"results.json 保存完了: {OUT_PATH}")

    # PyCaLi 履歴DBも更新
    try:
        import subprocess, sys
        subprocess.run([sys.executable, "build_pycali_history.py"], check=False)
    except Exception as e:
        log.warning(f"build_pycali_history 実行失敗: {e}")

    # ── Cowork 集計 ──
    try:
        cowork_result = aggregate_cowork_bets(kekka_cache)
        _tmp2 = COWORK_OUT_PATH.with_suffix(".tmp")
        with open(_tmp2, "w", encoding="utf-8") as f:
            json.dump(cowork_result, f, ensure_ascii=False, indent=2, default=str)
        _tmp2.replace(COWORK_OUT_PATH)
        log.info(f"cowork_results.json 保存完了: {COWORK_OUT_PATH}")
    except Exception as e:
        import traceback
        # 黙殺すると cowork_results.json が古いまま凍結し、後段の git add も
        # 「変更なし」で素通りする (2026-05-26 凍結事故)。error + traceback で必ず可視化。
        log.error(f"Cowork 集計失敗 (cowork_results.json は更新されません): {e}")
        log.error(traceback.format_exc())
        cowork_result = None

    print("\n=== 集計結果 ===")
    for plan in ["HAHO", "HALO", "LALO", "CQC", "TRIPLE"]:
        t = result[plan]["total"]
        print(f"{plan}: {t['races']}R  bet={t['bet']:,}  ret={t['ret']:,}  "
              f"pnl={t['pnl']:+,}  ROI={t['roi']}%")
    if cowork_result and cowork_result.get("total", {}).get("races", 0) > 0:
        t = cowork_result["total"]
        print(f"Cowork: {t['races']}R ({t.get('settled', 0)}確定/{t.get('見送り', 0)}見送り)  "
              f"bet={t['bet']:,}  ret={t['ret']:,}  pnl={t['pnl']:+,}  "
              f"ROI={t['roi']}%  hit={t['hit_count']}/{t['bet_count']} ({t['hit_rate']}%)")


if __name__ == "__main__":
    main()
