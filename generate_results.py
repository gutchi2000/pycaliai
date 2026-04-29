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

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE_DIR  = Path(__file__).parent
PRED_DIR  = BASE_DIR / "reports"
KEKKA_DIR = BASE_DIR / "data" / "kekka"
OUT_PATH  = BASE_DIR / "data" / "results.json"


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
    """date_key → kekka DataFrame のキャッシュを返す。"""
    cache: dict[str, pd.DataFrame] = {}
    for f in sorted(KEKKA_DIR.glob("????????.csv")):
        try:
            df = pd.read_csv(f, encoding="cp932")
            cache[f.stem] = df
        except Exception as e:
            log.warning(f"kekka 読み込みスキップ {f.name}: {e}")
    log.info(f"kekka {len(cache)} ファイル読み込み完了")
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
# メイン
# =========================================================
def main() -> None:
    pred = load_pred_all()
    kekka_cache = load_kekka_all()

    result = {"generated_at": pd.Timestamp.now().isoformat()}
    for plan in ["HAHO", "HALO", "LALO", "CQC", "TRIPLE"]:
        records = calc_plan_races(pred, kekka_cache, plan)
        result[plan] = build_summary(plan, records)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    log.info(f"results.json 保存完了: {OUT_PATH}")

    # PyCaLi 履歴DBも更新
    try:
        import subprocess, sys
        subprocess.run([sys.executable, "build_pycali_history.py"], check=False)
    except Exception as e:
        log.warning(f"build_pycali_history 実行失敗: {e}")

    print("\n=== 集計結果 ===")
    for plan in ["HAHO", "HALO", "LALO", "CQC", "TRIPLE"]:
        t = result[plan]["total"]
        print(f"{plan}: {t['races']}R  bet={t['bet']:,}  ret={t['ret']:,}  "
              f"pnl={t['pnl']:+,}  ROI={t['roi']}%")


if __name__ == "__main__":
    main()
