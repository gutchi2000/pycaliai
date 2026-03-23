"""
check_past_ev.py
週次CSV の EV推奨馬を kekka CSV で答え合わせ

Usage:
    python check_past_ev.py --dates 20260308 20260315
    python check_past_ev.py --all          # data/kekka/ にある全日付
    python check_past_ev.py --ev 1.5       # EV閾値を変更（デフォルト1.5）
"""
from __future__ import annotations
import argparse, logging, sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR  = Path(__file__).parent
WEEKLY    = BASE_DIR / "data" / "weekly"
KEKKA_DIR = BASE_DIR / "data" / "kekka"
REPORT    = BASE_DIR / "reports"

# predict_weekly.py と同じ定数
EXCLUDE_PLACES  = {"東京", "小倉"}
EXCLUDE_CLASSES = {"新馬", "障害"}
RETURN_RATE     = 0.80

# ---------------------------------------------------------------
# predict_weekly.py の ensemble_predict を再利用
# ---------------------------------------------------------------
sys.path.insert(0, str(BASE_DIR))
try:
    from predict_weekly import (
        ensemble_predict, _get_cached,
        LGBM_PATH, CAT_PATH, STRATEGY_JSON,
    )
    import joblib as _jl
    lgbm_obj = _jl.load(LGBM_PATH) if LGBM_PATH.exists() else None
    cat_obj  = _jl.load(CAT_PATH)  if CAT_PATH.exists()  else None
except Exception as e:
    print(f"[ERROR] モデルロード失敗: {e}")
    sys.exit(1)

# ---------------------------------------------------------------
# 戦略 JSON ロード
# ---------------------------------------------------------------
import json
strategy: dict = {}
if STRATEGY_JSON.exists():
    with open(STRATEGY_JSON, encoding="utf-8") as f:
        strategy = json.load(f)

CLASS_NORMALIZE = {
    "新馬":"新馬","未勝利":"未勝利","1勝":"1勝","500万":"1勝",
    "2勝":"2勝","1000万":"2勝","3勝":"3勝","1600万":"3勝",
    "OP(L)":"OP(L)","Ｇ１":"Ｇ１","Ｇ２":"Ｇ２","Ｇ３":"Ｇ３",
}

# ---------------------------------------------------------------
# weekly CSV 予測
# ---------------------------------------------------------------
def predict_weekly(csv_path: Path) -> pd.DataFrame:
    from predict_weekly import parse_csv, assign_marks
    rows = []
    try:
        df = parse_csv(csv_path)
    except Exception as e:
        logger.warning(f"CSV parse 失敗: {e}")
        return pd.DataFrame()

    race_id_col = "レースID(新/馬番無)"
    for race_id, race_df in df.groupby(race_id_col):
        race_df = race_df.copy().reset_index(drop=True)
        meta    = race_df.iloc[0]
        place   = str(meta.get("場所", ""))
        cls_raw = str(meta.get("クラス名", ""))
        if place in EXCLUDE_PLACES or cls_raw in EXCLUDE_CLASSES:
            continue
        cls = CLASS_NORMALIZE.get(cls_raw, cls_raw)
        bet_info = strategy.get(place, {}).get(cls) or strategy.get(place, {}).get(cls_raw, {})
        if not bet_info or "単勝" not in bet_info:
            continue  # CQC (単勝) 戦略のみ対象

        try:
            race_df["prob"] = ensemble_predict(race_df, lgbm_obj, cat_obj)
            race_df = assign_marks(race_df)
            tansho = pd.to_numeric(race_df.get("単勝", pd.Series(dtype=float)), errors="coerce")
            race_df["ev_score"] = (race_df["prob"] * tansho / RETURN_RATE).round(3)
        except Exception as e:
            logger.warning(f"予測失敗 {race_id}: {e}")
            continue

        hon = race_df[race_df["mark"] == "◎"]
        if hon.empty:
            continue
        hon_row = hon.iloc[0]
        rows.append({
            "日付S":        str(meta.get("日付S", "")),
            "場所":         place,
            "R":            str(meta.get("R", "")),
            "クラス":       cls_raw,
            "距離":         str(meta.get("距離", "")),
            "レースID":     str(race_id),
            "馬番":         int(hon_row["馬番"]) if pd.notna(hon_row.get("馬番")) else 0,
            "馬名":         str(hon_row.get("馬名S", hon_row.get("馬名", ""))),
            "単勝オッズ":   float(hon_row.get("単勝", 0)) if pd.notna(hon_row.get("単勝")) else 0.0,
            "EV":           float(hon_row.get("ev_score", 0)),
            "モデルスコア": float(hon_row.get("prob", 0)),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------
# kekka CSV から着順・払戻を取得
# ---------------------------------------------------------------
def load_kekka(date_str: str) -> pd.DataFrame:
    p = KEKKA_DIR / f"{date_str}.csv"
    if not p.exists():
        return pd.DataFrame()
    for enc in ("cp932", "utf-8-sig", "utf-8"):
        try:
            df = pd.read_csv(p, encoding=enc, dtype=str)
            # エンコード成功を確認：最初の列に日本語文字が含まれるか
            if any("\u3000" <= c <= "\u9fff" for c in "".join(str(x) for x in df.columns[:3])):
                df.columns = [c.strip() for c in df.columns]
                return df
        except (UnicodeDecodeError, Exception):
            continue
    # 最後のフォールバック
    try:
        df = pd.read_csv(p, encoding="cp932", dtype=str)
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame()


def get_result(kekka_df: pd.DataFrame, race_id: str, uma: int):
    """(着順, 単勝払戻) を返す。未発見は (None, None)。"""
    if kekka_df.empty:
        return None, None
    # レースIDの照合: kekka の 'レースID(新)' は 18桁
    id_col = next((c for c in kekka_df.columns if "レースID" in c), None)
    uma_col = next((c for c in kekka_df.columns if "馬番" == c), None)
    rank_col = next((c for c in kekka_df.columns if "着順" in c or "確定着順" in c), None)
    pay_col = next((c for c in kekka_df.columns
                    if any(kw in c for kw in ("単勝払戻", "単勝配当", "単勝払", "単勝_払"))), None)

    if not all([id_col, uma_col, rank_col]):
        return None, None

    # race_id 照合: 18桁 or 16桁どちらでも
    mask = kekka_df[id_col].str[:16] == str(race_id)[:16]
    race_kekka = kekka_df[mask]
    if race_kekka.empty:
        return None, None

    horse = race_kekka[race_kekka[uma_col].astype(str).str.strip() == str(uma)]
    if horse.empty:
        return None, None

    rank = horse.iloc[0][rank_col]
    pay  = horse.iloc[0][pay_col] if pay_col else None
    return rank, pay


# ---------------------------------------------------------------
# メイン
# ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dates", nargs="+", help="日付 (例: 20260308 20260315)")
    parser.add_argument("--all",   action="store_true", help="kekkaが存在する全日付")
    parser.add_argument("--ev",    type=float, default=1.5, help="EV閾値 (デフォルト: 1.5)")
    args = parser.parse_args()

    if args.all:
        dates = sorted(p.stem for p in KEKKA_DIR.glob("*.csv"))
    elif args.dates:
        dates = args.dates
    else:
        parser.print_help()
        return

    all_rows = []
    for date_str in dates:
        weekly_csv = WEEKLY / f"{date_str}.csv"
        if not weekly_csv.exists():
            print(f"[SKIP] weekly CSV なし: {date_str}")
            continue
        print(f"予測中: {date_str} ...", end=" ", flush=True)
        preds = predict_weekly(weekly_csv)
        if preds.empty:
            print("CQC対象なし")
            continue

        kekka_df = load_kekka(date_str)
        for _, row in preds.iterrows():
            rank, pay = get_result(kekka_df, row["レースID"], int(row["馬番"]))
            hit = (str(rank).strip() == "1")
            all_rows.append({
                "日付":    row["日付S"],
                "場所":    row["場所"],
                "R":       row["R"],
                "クラス":  row["クラス"],
                "馬名":    row["馬名"],
                "単勝":    row["単勝オッズ"],
                "EV":      row["EV"],
                "着順":    rank,
                "的中":    "◎" if hit else "×",
                "払戻":    pay if hit else "-",
            })
        print(f"{len(preds)}レース処理")

    if not all_rows:
        print("結果なし")
        return

    df = pd.DataFrame(all_rows)
    df["EV"] = pd.to_numeric(df["EV"], errors="coerce").fillna(0)

    # 全体
    print(f"\n{'='*65}")
    print(f"答え合わせ結果  EV閾値: {args.ev}  対象日: {', '.join(dates)}")
    print(f"{'='*65}")
    print(f"\n【CQC 単勝◎ 全候補】 {len(df)}点")
    print(df[["日付","場所","R","クラス","馬名","単勝","EV","着順","的中","払戻"]].to_string(index=False))

    # EV フィルタ後
    ev_df = df[df["EV"] >= args.ev].copy()
    if ev_df.empty:
        print(f"\nEV >= {args.ev} の候補なし")
        return

    hits   = (ev_df["的中"] == "◎").sum()
    total  = len(ev_df)
    hit_r  = hits / total * 100
    invest = total * 1000

    # 払戻合計（払戻列: 単勝払戻額 per 100円 × 10 = per 1000円）
    def parse_pay(x):
        try:
            s = str(x).replace("(","").replace(")","").strip()
            return float(s) * 10  # per 1000円換算
        except:
            return 0.0
    ev_df["払戻額"] = ev_df["払戻"].apply(parse_pay)
    total_pay = ev_df["払戻額"].sum()
    roi = total_pay / invest * 100 if invest > 0 else 0

    print(f"\n【★EV >= {args.ev} 候補】 {total}点 / 的中: {hits}点 ({hit_r:.0f}%)")
    print(ev_df[["日付","場所","R","クラス","馬名","単勝","EV","着順","的中","払戻"]].sort_values("EV", ascending=False).to_string(index=False))
    print(f"\n  投資額: {invest:,}円  払戻: {total_pay:,.0f}円  ROI: {roi:.1f}%")

    # CSV 保存
    out = REPORT / f"past_ev_check_{'_'.join(dates)}.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n  保存: {out}")


if __name__ == "__main__":
    main()
