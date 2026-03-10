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
BASE_DIR      = Path(__file__).parent
KEKKA_DIR     = BASE_DIR / "data" / "kekka"
PRED_DIR      = BASE_DIR / "reports"
OUT_PATH      = BASE_DIR / "data" / "results.json"
STRATEGY_JSON = BASE_DIR / "data" / "strategy_weights.json"

EXCLUDE_PLACES  = {"東京", "小倉"}
EXCLUDE_CLASSES = {"新馬", "障害"}
BUDGET          = 10_000   # 1レースあたりの投資予算（円）

CLASS_NORMALIZE = {
    "新馬":"新馬","未勝利":"未勝利","1勝":"1勝","500万":"1勝",
    "2勝":"2勝","1000万":"2勝","3勝":"3勝","1600万":"3勝",
    "OP(L)":"OP(L)","Ｇ１":"Ｇ１","Ｇ２":"Ｇ２","Ｇ３":"Ｇ３",
}


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
    files = sorted(pred_dir.glob("pred_*.csv"))
    if not files:
        raise FileNotFoundError(f"predCSVが見つかりません: {pred_dir}")
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f, encoding="utf-8-sig"))
        except UnicodeDecodeError:
            try:
                dfs.append(pd.read_csv(f, encoding="cp932"))
            except Exception as e:
                log.warning(f"スキップ {f.name}: {e}")
        except Exception as e:
            log.warning(f"スキップ {f.name}: {e}")
    pred = pd.concat(dfs, ignore_index=True)
    pred["レースキー"] = pred["レースID"].astype(str).str.zfill(16)
    pred["馬番_k"]     = pred["馬番"].astype(int)
    pred["日付_dt"]    = pd.to_datetime(pred["日付"].str.replace(".", "/"), errors="coerce")

    # ──────────────────────────────────────────────────────────
    # 後方互換: 旧フォーマットCSV（HAHO/HALO列なし）を自動マッピング
    #   旧列: 馬連_買い目, 馬連_購入額, 三連複_買い目, 三連複_購入額
    #   → HAHO: 馬連 + 三連複（按分額そのまま）
    #   → HALO: 旧全額合計（馬連+三連複+複勝+三連単）を三連複へ充当
    #            ※旧システムは全額を按分していたので、合計≒1万円 = HALO想定
    # ──────────────────────────────────────────────────────────
    if "HAHO_馬連_購入額" not in pred.columns and "馬連_購入額" in pred.columns:
        log.info("旧フォーマットCSV検出: 馬連/三連複 → HAHO/HALO/LALO にマッピング")
        pred["HAHO_馬連_購入額"]   = pd.to_numeric(pred["馬連_購入額"],   errors="coerce").fillna(0)
        pred["HAHO_三連複_購入額"] = pd.to_numeric(pred["三連複_購入額"], errors="coerce").fillna(0)
        pred["HAHO_馬連_買い目"]   = pred["馬連_買い目"].fillna("")
        pred["HAHO_三連複_買い目"] = pred["三連複_買い目"].fillna("")
        # HALO = 全予算をまるごと三連複に：三連複対象レースは一律 BUDGET 円
        sf_amt = pd.to_numeric(pred["三連複_購入額"], errors="coerce").fillna(0)
        pred["HALO_三連複_購入額"] = sf_amt.apply(lambda x: BUDGET if x > 0 else 0)
        pred["HALO_三連複_買い目"] = pred["三連複_買い目"].fillna("")
        # LALO = 複勝◎1点：HAHO対象レース（馬連 or 三連複に1円以上）は全て対象
        haho_amt = pred["HAHO_馬連_購入額"] + pred["HAHO_三連複_購入額"]
        pred["LALO_複勝_購入額"] = haho_amt.apply(lambda x: BUDGET if x > 0 else 0)
        pred["LALO_複勝_買い目"] = ""   # ◎馬番は calc_records で馬番_k から取得

    # HAHO/HALO/LALO 購入額（新フォーマット or マッピング後の数値化）
    for col in ["HAHO_馬連_購入額", "HAHO_三連複_購入額", "HALO_三連複_購入額", "LALO_複勝_購入額"]:
        if col in pred.columns:
            pred[col] = pd.to_numeric(pred[col], errors="coerce").fillna(0)
        else:
            pred[col] = 0.0
    # HAHO/HALO/LALO 買い目（新フォーマット or マッピング後の文字列保証）
    for col in ["HAHO_馬連_買い目", "HAHO_三連複_買い目", "HALO_三連複_買い目", "LALO_複勝_買い目"]:
        if col not in pred.columns:
            pred[col] = ""
    # 印列（念のため保証）
    if "印" not in pred.columns:
        pred["印"] = ""
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
            race_haitou[rk] = {"複勝": {}, "馬連": None, "三連複": None, "top3": set(), "1着": None, "2着": None}
        if jun == 1:
            race_haitou[rk]["馬連"]   = row["馬連_n"]
            race_haitou[rk]["三連複"] = row["三連複_n"]
            race_haitou[rk]["1着"]    = ban
        if jun == 2:
            race_haitou[rk]["2着"]    = ban
        if jun in [1, 2, 3]:
            race_haitou[rk]["複勝"][ban] = row["複勝配当_n"]
            race_haitou[rk]["top3"].add(ban)
    return race_haitou


# =========================================================
# レース単位集計
# =========================================================
def calc_records(pred: pd.DataFrame, race_haitou: dict, strategy: dict) -> list[dict]:
    records = []
    for rk, rdf in pred.groupby("レースキー"):
        h = race_haitou.get(rk)
        if not h:
            continue
        place   = str(rdf.iloc[0].get("場所", ""))
        cls_raw = str(rdf.iloc[0].get("クラス", ""))
        if place in EXCLUDE_PLACES or cls_raw in EXCLUDE_CLASSES:
            continue
        # 戦略フィルタ
        if strategy:
            cls = CLASS_NORMALIZE.get(cls_raw, cls_raw)
            bet_info = strategy.get(place, {}).get(cls) or strategy.get(place, {}).get(cls_raw, {})
            if not bet_info:
                continue
        hon_rows = rdf[rdf["印"] == "◎"]
        if hon_rows.empty:
            continue
        hon = hon_rows.iloc[0]

        top2    = frozenset([h["1着"], h["2着"]]) if h["1着"] and h["2着"] else frozenset()
        top3_fs = frozenset(h["top3"]) if len(h["top3"]) == 3 else None

        # ── HAHO ─────────────────────────────────────────────────────────
        haho_rengo_amt = float(hon.get("HAHO_馬連_購入額", 0))
        haho_sf_amt    = float(hon.get("HAHO_三連複_購入額", 0))
        haho_target    = (haho_rengo_amt + haho_sf_amt) > 0
        haho_rengo_ret = 0.0; haho_rengo_hit = False
        haho_sf_ret    = 0.0; haho_sf_hit    = False

        if haho_rengo_amt > 0 and h["馬連"] and top2:
            mask = rdf["HAHO_馬連_買い目"].notna() & (pd.to_numeric(rdf["HAHO_馬連_購入額"], errors="coerce").fillna(0) > 0)
            for _, brow in rdf[mask].iterrows():
                combos = parse_combos(brow["HAHO_馬連_買い目"])
                per = float(brow["HAHO_馬連_購入額"]) / max(len(combos), 1)
                for c in combos:
                    if c == top2:
                        haho_rengo_ret += per * h["馬連"] / 100
                        haho_rengo_hit = True

        if haho_sf_amt > 0 and h["三連複"] and top3_fs:
            mask = rdf["HAHO_三連複_買い目"].notna() & (pd.to_numeric(rdf["HAHO_三連複_購入額"], errors="coerce").fillna(0) > 0)
            for _, brow in rdf[mask].iterrows():
                combos = parse_combos(brow["HAHO_三連複_買い目"])
                per = float(brow["HAHO_三連複_購入額"]) / max(len(combos), 1)
                for c in combos:
                    if c == top3_fs:
                        haho_sf_ret += per * h["三連複"] / 100
                        haho_sf_hit = True

        # ── HALO ─────────────────────────────────────────────────────────
        halo_sf_amt = float(hon.get("HALO_三連複_購入額", 0))
        halo_target = halo_sf_amt > 0
        halo_sf_ret = 0.0; halo_sf_hit = False

        if halo_sf_amt > 0 and h["三連複"] and top3_fs:
            mask = rdf["HALO_三連複_買い目"].notna() & (pd.to_numeric(rdf["HALO_三連複_購入額"], errors="coerce").fillna(0) > 0)
            for _, brow in rdf[mask].iterrows():
                combos = parse_combos(brow["HALO_三連複_買い目"])
                per = float(brow["HALO_三連複_購入額"]) / max(len(combos), 1)
                for c in combos:
                    if c == top3_fs:
                        halo_sf_ret += per * h["三連複"] / 100
                        halo_sf_hit = True

        # ── LALO: 複勝◎1点 ───────────────────────────────────────────────
        lalo_fuku_amt = float(hon.get("LALO_複勝_購入額", 0))
        lalo_target   = lalo_fuku_amt > 0
        lalo_fuku_ret = 0.0; lalo_fuku_hit = False

        if lalo_fuku_amt > 0:
            hon_ban      = int(hon["馬番_k"])
            fuku_payout  = h["複勝"].get(hon_ban, 0) or 0
            if fuku_payout > 0:
                lalo_fuku_ret = lalo_fuku_amt * fuku_payout / 100
                lalo_fuku_hit = True

        records.append({
            "レースキー":        rk,
            "日付":              rdf.iloc[0]["日付"],
            "日付_dt":           rdf.iloc[0]["日付_dt"],
            "場所":              rdf.iloc[0]["場所"],
            "R":                 int(rdf.iloc[0]["R"]),
            "クラス":            rdf.iloc[0]["クラス"],
            # HAHO
            "HAHO_対象":         int(haho_target),
            "HAHO_馬連_投資":    haho_rengo_amt, "HAHO_馬連_払戻":    haho_rengo_ret, "HAHO_馬連_的中":    int(haho_rengo_hit),
            "HAHO_三連複_投資":  haho_sf_amt,    "HAHO_三連複_払戻":  haho_sf_ret,    "HAHO_三連複_的中":  int(haho_sf_hit),
            "HAHO_総投資":       haho_rengo_amt + haho_sf_amt,
            "HAHO_総払戻":       haho_rengo_ret + haho_sf_ret,
            # HALO
            "HALO_対象":         int(halo_target),
            "HALO_三連複_投資":  halo_sf_amt,    "HALO_三連複_払戻":  halo_sf_ret,    "HALO_三連複_的中":  int(halo_sf_hit),
            "HALO_総投資":       halo_sf_amt,
            "HALO_総払戻":       halo_sf_ret,
            # LALO
            "LALO_対象":         int(lalo_target),
            "LALO_複勝_投資":    lalo_fuku_amt,  "LALO_複勝_払戻":    lalo_fuku_ret,  "LALO_複勝_的中":    int(lalo_fuku_hit),
            "LALO_総投資":       lalo_fuku_amt,
            "LALO_総払戻":       lalo_fuku_ret,
        })
    return records


# =========================================================
# 集計
# =========================================================
def _plan_summary(df: pd.DataFrame, prefix: str) -> dict:
    """HAHO / HALO のサマリーを計算。"""
    tgt = df[df[f"{prefix}_対象"] == 1].copy()
    if tgt.empty:
        return {"total": {"races": 0, "bet": 0, "ret": 0, "pnl": 0, "roi": 0},
                "by_type": {}, "weekly": [], "by_place": [], "races": []}

    total_bet = tgt[f"{prefix}_総投資"].sum()
    total_ret = tgt[f"{prefix}_総払戻"].sum()

    # 馬券種別
    by_type: dict = {}
    if prefix == "HAHO":
        type_keys = ["馬連", "三連複"]
    elif prefix == "LALO":
        type_keys = ["複勝"]
    else:
        type_keys = ["三連複"]
    for k in type_keys:
        col_i = f"{prefix}_{k}_投資"
        col_r = f"{prefix}_{k}_払戻"
        col_h = f"{prefix}_{k}_的中"
        if col_i not in tgt.columns:
            continue
        bet = tgt[col_i].sum()
        ret = tgt[col_r].sum()
        hit = tgt[col_h].sum()
        n   = (tgt[col_i] > 0).sum()
        by_type[k] = {
            "bet": int(bet), "ret": int(ret),
            "roi": round(ret / bet * 100, 1) if bet > 0 else 0,
            "hit": int(hit), "races": int(n),
            "hit_rate": round(hit / n * 100, 1) if n > 0 else 0,
        }

    # 週次
    weekly = tgt.groupby("週").agg(
        レース数=(f"{prefix}_総投資", "count"),
        総投資=(f"{prefix}_総投資", "sum"),
        総払戻=(f"{prefix}_総払戻", "sum"),
    ).reset_index()
    weekly["収支"] = weekly["総払戻"] - weekly["総投資"]
    weekly["ROI"]  = (weekly["総払戻"] / weekly["総投資"] * 100).round(1)
    weekly.loc[weekly["総投資"] == 0, "ROI"] = 0

    # 会場別
    by_place = tgt.groupby("場所").agg(
        レース数=(f"{prefix}_総投資", "count"),
        総投資=(f"{prefix}_総投資", "sum"),
        総払戻=(f"{prefix}_総払戻", "sum"),
    ).reset_index()
    by_place["ROI"]  = (by_place["総払戻"] / by_place["総投資"] * 100).round(1)
    by_place.loc[by_place["総投資"] == 0, "ROI"] = 0
    by_place["収支"] = by_place["総払戻"] - by_place["総投資"]

    # 個別レース（日付降順）
    rename_map = {
        f"{prefix}_馬連_投資":   "馬連_投資",   f"{prefix}_馬連_払戻":   "馬連_払戻",   f"{prefix}_馬連_的中":   "馬連_的中",
        f"{prefix}_三連複_投資": "三連複_投資", f"{prefix}_三連複_払戻": "三連複_払戻", f"{prefix}_三連複_的中": "三連複_的中",
        f"{prefix}_複勝_投資":   "複勝_投資",   f"{prefix}_複勝_払戻":   "複勝_払戻",   f"{prefix}_複勝_的中":   "複勝_的中",
        f"{prefix}_総投資":      "総投資",      f"{prefix}_総払戻":      "総払戻",
    }
    base_cols = ["日付", "場所", "R", "クラス"]
    plan_cols = [c for c in rename_map if c in tgt.columns]
    races_df  = tgt.sort_values("日付_dt", ascending=False)[base_cols + plan_cols].rename(columns=rename_map)
    races_df["収支"] = races_df["総払戻"] - races_df["総投資"]

    return {
        "total": {
            "races": len(tgt),
            "bet":   int(total_bet),
            "ret":   int(total_ret),
            "pnl":   int(total_ret - total_bet),
            "roi":   round(total_ret / total_bet * 100, 1) if total_bet > 0 else 0,
        },
        "by_type":  by_type,
        "weekly":   weekly.to_dict("records"),
        "by_place": by_place.sort_values("ROI", ascending=False).to_dict("records"),
        "races":    races_df.to_dict("records"),
    }


def summarize(records: list[dict]) -> dict:
    df = pd.DataFrame(records)
    df["日付_dt"] = pd.to_datetime(df["日付_dt"])
    df["週"]      = df["日付_dt"].dt.to_period("W").apply(lambda x: str(x.end_time.date()))

    return {
        "generated_at": pd.Timestamp.now().isoformat(),
        "HAHO": _plan_summary(df, "HAHO"),
        "HALO": _plan_summary(df, "HALO"),
        "LALO": _plan_summary(df, "LALO"),
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

    strategy: dict = {}
    if STRATEGY_JSON.exists():
        with open(STRATEGY_JSON, encoding="utf-8") as f:
            strategy = json.load(f)
        log.info(f"戦略ファイル読み込み: {STRATEGY_JSON.name}")
    else:
        log.warning("strategy_weights.json が見つかりません。全レースを集計します。")

    kekka       = load_kekka(kekka_dir)
    pred        = load_pred(pred_dir)
    race_haitou = build_race_haitou(kekka)
    records     = calc_records(pred, race_haitou, strategy)
    summary     = summarize(records)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    log.info(f"生成完了: {out_path}")
    haho = summary["HAHO"]["total"]
    halo = summary["HALO"]["total"]
    lalo = summary["LALO"]["total"]
    log.info(f"HAHO: {haho['races']}R  ROI: {haho['roi']}%  収支: {haho['pnl']:+,}円")
    log.info(f"HALO: {halo['races']}R  ROI: {halo['roi']}%  収支: {halo['pnl']:+,}円")
    log.info(f"LALO: {lalo['races']}R  ROI: {lalo['roi']}%  収支: {lalo['pnl']:+,}円")


if __name__ == "__main__":
    main()
