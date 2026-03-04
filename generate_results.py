"""
generate_results.py
===================
data/kekka/*.csv  × data/weekly/*.csv (pred) を突合して
data/results.json を生成するスクリプト。

使い方:
    python generate_results.py
    python generate_results.py --kekka_dir data/kekka --pred_dir data/weekly --out data/results.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# =========================================================
# 定数
# =========================================================
BASE_DIR  = Path(__file__).parent
KEKKA_DIR = BASE_DIR / "data" / "kekka"
PRED_DIR  = BASE_DIR / "data" / "weekly"
OUT_PATH  = BASE_DIR / "data" / "results.json"


# =========================================================
# ユーティリティ
# =========================================================
def parse_haitou(x) -> float | None:
    """括弧付きオッズ・nan を除外して配当を数値化。"""
    s = str(x).strip()
    if s.startswith("(") or s in ("nan", ""):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_combos(s: str) -> list[frozenset]:
    """'4-11-14 / 4-14-16' → [{4,11,14}, {4,14,16}]"""
    if pd.isna(s):
        return []
    combos: list[frozenset] = []
    for part in str(s).split("/"):
        nums = part.strip().split("-")
        try:
            combos.append(frozenset(map(int, nums)))
        except ValueError:
            pass
    return combos


# =========================================================
# データ読み込み
# =========================================================
def load_kekka(kekka_dir: Path) -> pd.DataFrame:
    """kekka/*.csv を全読みして結合。"""
    files = sorted(kekka_dir.glob("????????.csv"))
    if not files:
        raise FileNotFoundError(f"kekkaCSVが見つかりません: {kekka_dir}")
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f, encoding="cp932"))
            log.info(f"読み込み: {f.name}")
        except Exception as e:
            log.warning(f"スキップ {f.name}: {e}")
    kekka = pd.concat(dfs, ignore_index=True)
    kekka["ID_str"]     = kekka["レースID(新)"].astype(str).str.zfill(18)
    kekka["馬番_k"]     = kekka["ID_str"].str[-2:].astype(int)
    kekka["レースキー"] = kekka["ID_str"].str[:16]
    kekka["複勝配当_n"] = pd.to_numeric(kekka["複勝配当"], errors="coerce")
    kekka["馬連_n"]     = pd.to_numeric(kekka["馬連"],     errors="coerce")
    kekka["三連複_n"]   = pd.to_numeric(kekka["３連複"],   errors="coerce")
    log.info(f"kekka合計: {len(kekka)}行 / {kekka['レースキー'].nunique()}レース")
    return kekka


def load_pred(pred_dir: Path) -> pd.DataFrame:
    """weekly/*.csv (pred) を全読みして結合。"""
    files = sorted(pred_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"predCSVが見つかりません: {pred_dir}")
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f, encoding="utf-8"))
        except Exception as e:
            log.warning(f"スキップ {f.name}: {e}")
    pred = pd.concat(dfs, ignore_index=True)
    pred["レースキー"] = pred["レースID"].astype(str).str.zfill(16)
    pred["馬番_k"]     = pred["馬番"].astype(int)
    pred["日付_dt"]    = pd.to_datetime(pred["日付"].str.replace(".", "/"), errors="coerce")
    for col in ["複勝_購入額", "馬連_購入額", "三連複_購入額"]:
        pred[col] = pd.to_numeric(pred[col], errors="coerce").fillna(0)
    log.info(f"pred合計: {len(pred)}行 / {pred['レースキー'].nunique()}レース")
    return pred


# =========================================================
# レース単位の配当 dict 生成
# =========================================================
def build_race_haitou(kekka: pd.DataFrame) -> dict:
    """レースキー → 配当情報 の dict を返す。"""
    race_haitou: dict = {}
    for _, row in kekka.iterrows():
        rk  = row["レースキー"]
        ban = row["馬番_k"]
        jun = row["確定着順"]
        if rk not in race_haitou:
            race_haitou[rk] = {"複勝": {}, "馬連": None, "三連複": None, "top3": set()}
        if jun == 1:
            race_haitou[rk]["馬連"]   = row["馬連_n"]
            race_haitou[rk]["三連複"] = row["三連複_n"]
        if jun in [1, 2, 3]:
            race_haitou[rk]["複勝"][ban] = row["複勝配当_n"]
            race_haitou[rk]["top3"].add(ban)
    return race_haitou


# =========================================================
# レース単位集計
# =========================================================
def calc_records(pred: pd.DataFrame, race_haitou: dict) -> list[dict]:
    records = []
    for rk, rdf in pred.groupby("レースキー"):
        h = race_haitou.get(rk)
        if not h:
            continue
        hon_rows = rdf[rdf["印"] == "◎"]
        if hon_rows.empty:
            continue
        h1 = int(hon_rows.iloc[0]["馬番_k"])

        # 複勝◎
        fuku_amt = float(hon_rows.iloc[0]["複勝_購入額"])
        fuku_ret = 0.0
        fuku_hit = False
        if h1 in h["複勝"] and h["複勝"][h1]:
            fuku_ret = fuku_amt * h["複勝"][h1] / 100
            fuku_hit = True

        # 馬連
        rengo_amt = rengo_ret = 0.0
        rengo_hit = False
        top2 = frozenset(list(h["top3"])[:2]) if len(h["top3"]) >= 2 else frozenset()
        for _, brow in rdf[rdf["馬連_買い目"].notna() & (rdf["馬連_購入額"] > 0)].iterrows():
            combos = parse_combos(brow["馬連_買い目"])
            per = brow["馬連_購入額"] / max(len(combos), 1)
            rengo_amt += brow["馬連_購入額"]
            if h["馬連"]:
                for c in combos:
                    if c == top2:
                        rengo_ret += per * h["馬連"] / 100
                        rengo_hit = True

        # 三連複
        sanfuku_amt = sanfuku_ret = 0.0
        sanfuku_hit = False
        if len(h["top3"]) == 3:
            top3_fs = frozenset(h["top3"])
            for _, brow in rdf[rdf["三連複_買い目"].notna() & (rdf["三連複_購入額"] > 0)].iterrows():
                combos = parse_combos(brow["三連複_買い目"])
                per = brow["三連複_購入額"] / max(len(combos), 1)
                sanfuku_amt += brow["三連複_購入額"]
                if h["三連複"]:
                    for c in combos:
                        if c == top3_fs:
                            sanfuku_ret += per * h["三連複"] / 100
                            sanfuku_hit = True

        records.append({
            "レースキー":   rk,
            "日付":         rdf.iloc[0]["日付"],
            "日付_dt":      rdf.iloc[0]["日付_dt"],
            "場所":         rdf.iloc[0]["場所"],
            "R":            int(rdf.iloc[0]["R"]),
            "クラス":       rdf.iloc[0]["クラス"],
            "複勝_投資":    fuku_amt,    "複勝_払戻":    fuku_ret,    "複勝_的中":    int(fuku_hit),
            "馬連_投資":    rengo_amt,   "馬連_払戻":    rengo_ret,   "馬連_的中":    int(rengo_hit),
            "三連複_投資":  sanfuku_amt, "三連複_払戻":  sanfuku_ret, "三連複_的中":  int(sanfuku_hit),
            "総投資":       fuku_amt + rengo_amt + sanfuku_amt,
            "総払戻":       fuku_ret + rengo_ret + sanfuku_ret,
        })
    return records


# =========================================================
# 集計
# =========================================================
def summarize(records: list[dict]) -> dict:
    df = pd.DataFrame(records)
    df["収支"]    = df["総払戻"] - df["総投資"]
    df["日付_dt"] = pd.to_datetime(df["日付_dt"])
    df["週"]      = df["日付_dt"].dt.to_period("W").apply(lambda x: str(x.end_time.date()))

    total_bet = df["総投資"].sum()
    total_ret = df["総払戻"].sum()

    # 馬券種別
    by_type: dict = {}
    for k in ["複勝", "馬連", "三連複"]:
        bet = df[f"{k}_投資"].sum()
        ret = df[f"{k}_払戻"].sum()
        hit = df[f"{k}_的中"].sum()
        n   = (df[f"{k}_投資"] > 0).sum()
        by_type[k] = {
            "bet": int(bet), "ret": int(ret),
            "roi": round(ret / bet * 100, 1) if bet > 0 else 0,
            "hit": int(hit), "races": int(n),
            "hit_rate": round(hit / n * 100, 1) if n > 0 else 0,
        }

    # 週次
    weekly = df.groupby("週").agg(
        レース数=("総投資", "count"),
        総投資=("総投資", "sum"),
        総払戻=("総払戻", "sum"),
        収支=("収支", "sum"),
    ).reset_index()
    weekly["ROI"] = (weekly["総払戻"] / weekly["総投資"] * 100).round(1)

    # 会場別
    by_place = df.groupby("場所").agg(
        レース数=("総投資", "count"),
        総投資=("総投資", "sum"),
        総払戻=("総払戻", "sum"),
    ).reset_index()
    by_place["ROI"]  = (by_place["総払戻"] / by_place["総投資"] * 100).round(1)
    by_place["収支"] = by_place["総払戻"] - by_place["総投資"]

    # 個別レース（日付降順）
    races_out = df.sort_values("日付_dt", ascending=False)[[
        "日付", "場所", "R", "クラス",
        "複勝_投資", "複勝_払戻", "複勝_的中",
        "馬連_投資", "馬連_払戻", "馬連_的中",
        "三連複_投資", "三連複_払戻", "三連複_的中",
        "総投資", "総払戻", "収支",
    ]].to_dict("records")

    return {
        "generated_at": pd.Timestamp.now().isoformat(),
        "total": {
            "races": len(df),
            "bet":   int(total_bet),
            "ret":   int(total_ret),
            "pnl":   int(total_ret - total_bet),
            "roi":   round(total_ret / total_bet * 100, 1) if total_bet > 0 else 0,
        },
        "by_type":  by_type,
        "weekly":   weekly.to_dict("records"),
        "by_place": by_place.sort_values("ROI", ascending=False).to_dict("records"),
        "races":    races_out,
    }


# =========================================================
# メイン
# =========================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="kekka × pred 突合 → results.json 生成")
    parser.add_argument("--kekka_dir", default=str(KEKKA_DIR))
    parser.add_argument("--pred_dir",  default=str(PRED_DIR))
    parser.add_argument("--out",       default=str(OUT_PATH))
    args = parser.parse_args()

    kekka_dir = Path(args.kekka_dir)
    pred_dir  = Path(args.pred_dir)
    out_path  = Path(args.out)

    kekka       = load_kekka(kekka_dir)
    pred        = load_pred(pred_dir)
    race_haitou = build_race_haitou(kekka)
    records     = calc_records(pred, race_haitou)
    summary     = summarize(records)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    log.info(f"生成完了: {out_path}")
    log.info(f"集計レース: {summary['total']['races']}R  ROI: {summary['total']['roi']}%")


if __name__ == "__main__":
    main()
