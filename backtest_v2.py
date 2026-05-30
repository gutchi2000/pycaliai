"""
backtest_v2.py
Stage 1 検証用バックテスト。app.py の get_bets を直接呼ぶ版。

旧 backtest.py との違い:
  - app.py の get_bets を流用 → EV ゲート STANDARD / trifecta_model_v1 HALO / HAHO / TRIPLE の
    実運用と同じ買い目を生成
  - plan (HAHO/HALO/STANDARD/TRIPLE) ごとに ROI を集計
  - 馬券種別の内訳も出力

Usage:
    python backtest_v2.py
    python backtest_v2.py --n_races 500
    python backtest_v2.py --period 2025   # 2025 年のみ
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

# =========================================================
# streamlit スタブ: app.py の @st.cache_data / st.* 呼び出しを無害化
# =========================================================
class _StStub:
    """Streamlit のデコレータ・UI 呼び出しを no-op 化する軽量スタブ。"""
    def __init__(self):
        # キャッシュデコレータ群
        self.cache_data     = self._passthrough_decorator
        self.cache_resource = self._passthrough_decorator
        self.cache          = self._passthrough_decorator
        self.session_state  = {}

    def _passthrough_decorator(self, *d_args, **d_kwargs):
        # st.cache_data(ttl=60) と @st.cache_data の両方に対応
        if len(d_args) == 1 and callable(d_args[0]) and not d_kwargs:
            return d_args[0]
        def wrapper(fn): return fn
        return wrapper

    def __getattr__(self, name):
        # st.markdown / st.info / st.error / st.sidebar.xxx 等を no-op に
        mock = MagicMock(name=f"st.{name}")
        setattr(self, name, mock)
        return mock

sys.modules["streamlit"] = _StStub()

# japanize_matplotlib / shap を軽量化（app.py の import 時のコスト削減）
try:
    import shap  # noqa: F401
except Exception:
    sys.modules["shap"] = MagicMock()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# =========================================================
# app.py / backtest.py を流用
# =========================================================
BASE_DIR  = Path(r"E:\PyCaLiAI")
DATA_DIR  = BASE_DIR / "data"
MASTER_CSV = DATA_DIR / "master_kako5.csv"  # 108列 full master (kako5/hist 含む, 2026-04-20 監査で切替)
KEKKA_CSV  = DATA_DIR / "kekka_20130105-20251228.csv"
STRATEGY_WEIGHTS = BASE_DIR / "data" / "strategy_weights.json"
COL_RACE_ID = "レースID(新/馬番無)"
BUDGET = 10_000

logger.info("app.py を import 中（streamlit スタブ経由）...")
sys.path.insert(0, str(BASE_DIR))
import app  # noqa: E402
import backtest as bt_old  # noqa: E402  # 予測関数と kekka ローダを流用

# =========================================================
# 的中判定 → 払戻
# =========================================================
def _bet_to_combo_ordered(bet_type: str, kaime: str) -> tuple[list[int], bool]:
    """買い目文字列を combo, ordered に変換。"""
    if bet_type in ("単勝", "複勝"):
        return [int(kaime)], False
    sep = "→" if "→" in kaime else "-"
    parts = [int(x) for x in kaime.split(sep) if x.strip()]
    if bet_type in ("馬単", "三連単"):
        return parts, True
    return parts, False


def get_actual_payout(bet_type: str, kaime: str, kekka_entry: dict) -> int:
    """100円あたりの払戻。的中しなければ 0。kekka_entry は backtest.load_kekka() の出力。"""
    combo, ordered = _bet_to_combo_ordered(bet_type, kaime)
    if bet_type == "単勝":
        # kekka には単勝が無い場合あり → 複勝で代替判定せずそのまま 0
        return int(kekka_entry.get("単勝", {}).get(combo[0], 0)) if "単勝" in kekka_entry else 0
    if bet_type == "複勝":
        return int(kekka_entry["複勝"].get(combo[0], 0))
    if bet_type == "馬連":
        key = "-".join(map(str, sorted(combo)))
        return int(kekka_entry["馬連"].get(key, 0))
    if bet_type == "馬単":
        key = "-".join(map(str, combo))
        return int(kekka_entry["馬単"].get(key, 0))
    if bet_type == "三連複":
        key = "-".join(map(str, sorted(combo)))
        return int(kekka_entry["三連複"].get(key, 0))
    if bet_type == "三連単":
        key = "-".join(map(str, combo))
        return int(kekka_entry["三連単"].get(key, 0))
    return 0


# =========================================================
# メイン
# =========================================================
def run(args):
    # strategy_weights
    with open(STRATEGY_WEIGHTS, encoding="utf-8") as f:
        strategy = json.load(f)

    # master CSV
    logger.info(f"master CSV 読み込み: {MASTER_CSV}")
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    df["日付"] = pd.to_numeric(df["日付"], errors="coerce")

    if args.period == "2025":
        test_df = df[(df["日付"] >= 20250101) & (df["日付"] < 20260101)].copy()
        period_label = "2025 (OOS)"
    elif args.period == "2024":
        test_df = df[(df["日付"] >= 20240101) & (df["日付"] < 20250101)].copy()
        period_label = "2024 (Val)"
    else:
        test_df = df[df["split"] == "test"].copy()
        period_label = "split==test"
    test_df = test_df.reset_index(drop=True)
    logger.info(f"期間={period_label} 行数={len(test_df):,}")

    # kekka
    kekka_dict = bt_old.load_kekka(KEKKA_CSV)
    # backtest.load_kekka は単勝を入れないので補填
    logger.info("単勝配当補填...")
    k_raw = pd.read_csv(KEKKA_CSV, encoding="cp932", low_memory=False)
    k_raw["race_id"] = k_raw["レースID(新)"].astype(str).str[:16]
    k_raw["確定着順"] = pd.to_numeric(k_raw["確定着順"], errors="coerce")
    k_raw["馬番"]     = pd.to_numeric(k_raw["馬番"], errors="coerce")
    win_rows = k_raw[k_raw["確定着順"] == 1]
    for _, row in win_rows.iterrows():
        rid = str(row["race_id"])
        ban = row["馬番"]; pay_raw = row.get("単勝配当", 0)
        pay = pd.to_numeric(pay_raw, errors="coerce")
        if pd.notna(ban) and pd.notna(pay) and rid in kekka_dict:
            kekka_dict[rid].setdefault("単勝", {})[int(ban)] = int(pay)

    # アンサンブル予測 + 較正 + 印付与
    logger.info("アンサンブル予測...")
    raw_prob = bt_old.ensemble_predict_batch(test_df)
    # app.py と同じ較正パイプラインを適用（raw prob は真の勝率より過大なので必須）
    # 2026-04-20: v4(fit_split=test_2024) と v3(cal_split=train) はリーク のため除外
    # v1/v2 は valid=2023 fit のクリーン版
    cal_path_candidates = [
        BASE_DIR / "models" / "ensemble_calibrator_v1.pkl",
        BASE_DIR / "models" / "ensemble_calibrator_v2.pkl",
    ]
    import joblib as _jb
    cal = None
    for cp in cal_path_candidates:
        if cp.exists():
            obj = _jb.load(cp)
            cal = obj["calibrator"] if isinstance(obj, dict) else obj
            logger.info(f"calibrator ロード: {cp.name}")
            break
    if cal is not None:
        cal_prob = cal.transform(raw_prob)
    else:
        logger.warning("calibrator 見つからず raw prob のまま")
        cal_prob = raw_prob
    test_df["prob"] = cal_prob
    logger.info("印付与...")
    test_df = bt_old.assign_marks_df(test_df)

    race_ids = test_df[COL_RACE_ID].unique()
    if args.n_races:
        race_ids = race_ids[:args.n_races]
    logger.info(f"バックテスト: {len(race_ids):,} レース")

    rows = []
    plan_no_bet = {"HAHO":0, "HALO":0, "STANDARD":0, "TRIPLE":0}
    for rid in tqdm(race_ids, desc="backtest"):
        race_df = test_df[test_df[COL_RACE_ID] == rid].copy()
        if len(race_df) < 3:
            continue
        r0 = race_df.iloc[0]
        place = str(r0.get("場所", ""))
        cls_raw = str(r0.get("クラス名", ""))
        date_s = str(int(r0.get("日付", 0)))

        try:
            bets_all = app.get_bets(race_df, place, cls_raw, strategy, BUDGET)
        except Exception as e:
            logger.warning(f"race {rid} get_bets 失敗: {e}")
            continue
        if not bets_all:
            continue

        kekka_entry = kekka_dict.get(str(rid), {"複勝":{}, "馬連":{}, "馬単":{}, "三連複":{}, "三連単":{}, "単勝":{}})

        # HALO source 診断（trifecta_model_v1 を通っているか）
        halo_info = bets_all.get("_HALO_INFO", {})
        halo_source = halo_info.get("source", "none") if halo_info else "none"
        halo_pattern = halo_info.get("pattern", "") if halo_info else ""

        for plan, bets in bets_all.items():
            if plan.startswith("_"):
                continue
            if not bets:
                plan_no_bet[plan] = plan_no_bet.get(plan, 0) + 1
                continue
            for b in bets:
                bt_type = b["馬券種"]
                kaime   = b["買い目"]
                amt     = int(b.get("購入額", 0))
                if amt <= 0:
                    continue
                pay_per100 = get_actual_payout(bt_type, kaime, kekka_entry)
                hit        = 1 if pay_per100 > 0 else 0
                actual_pay = int(amt * pay_per100 / 100) if hit else 0
                rows.append({
                    "race_id": rid, "日付": date_s, "場所": place, "クラス": cls_raw,
                    "plan": plan, "馬券種": bt_type, "買い目": kaime,
                    "購入額": amt, "的中": hit, "払戻": actual_pay,
                    "収支": actual_pay - amt,
                    "EV": b.get("EV", np.nan),  # STANDARD は格納済
                    "halo_source": halo_source if plan == "HALO" else "",
                    "halo_pattern": halo_pattern if plan == "HALO" else "",
                })

    if not rows:
        logger.error("結果 0 件")
        return

    res = pd.DataFrame(rows)
    res.to_csv(BASE_DIR / "reports" / "backtest_v2_results.csv", index=False, encoding="utf-8-sig")

    # ===== 集計 =====
    def _fmt(label, sub: pd.DataFrame):
        n_bets = len(sub); bet = sub["購入額"].sum(); pay = sub["払戻"].sum()
        roi = (pay / bet * 100) if bet > 0 else 0
        hit = sub["的中"].sum(); hr = (hit / n_bets * 100) if n_bets else 0
        print(f"  {label:<22s} n={n_bets:>6,}  bet=¥{bet:>12,}  pay=¥{pay:>12,}  ROI={roi:>6.2f}%  hit={hr:>5.2f}%  収支={pay-bet:>+12,}")

    n_races = res["race_id"].nunique()
    print(f"\n{'='*95}\nbacktest_v2 サマリ  period={period_label}  races={n_races:,}\n{'='*95}")
    _fmt("[全体]", res)

    print("\n— plan 別 —")
    for plan in ["HAHO", "HALO", "STANDARD", "TRIPLE"]:
        sub = res[res["plan"] == plan]
        if len(sub):
            _fmt(plan, sub)
        else:
            print(f"  {plan:<22s} (no bets)")

    print("\n— 馬券種別 —")
    for bt_type in ["単勝", "複勝", "馬連", "馬単", "三連複", "三連単"]:
        sub = res[res["馬券種"] == bt_type]
        if len(sub):
            _fmt(bt_type, sub)

    print("\n— plan × 馬券種 —")
    for plan in ["HAHO", "HALO", "STANDARD", "TRIPLE"]:
        for bt_type in ["単勝", "複勝", "馬連", "三連複", "三連単"]:
            sub = res[(res["plan"] == plan) & (res["馬券種"] == bt_type)]
            if len(sub):
                _fmt(f"{plan} × {bt_type}", sub)

    # 投資頻度 (週当たり)
    res["日付_dt"] = pd.to_datetime(res["日付"], format="%Y%m%d", errors="coerce")
    res["週"] = res["日付_dt"].dt.strftime("%Y-W%V")
    weekly = res.groupby("週").agg(races=("race_id","nunique"), bet=("購入額","sum"), pay=("払戻","sum"))
    weekly["roi"] = weekly["pay"] / weekly["bet"] * 100
    print(f"\n— 週次統計 (投資 ≥ ¥1,000 の週のみ n={len(weekly[weekly['bet']>=1000])}) —")
    active = weekly[weekly["bet"] >= 1000]
    if len(active):
        print(f"  週当たり投資 中央値=¥{active['bet'].median():,.0f}  平均=¥{active['bet'].mean():,.0f}")
        print(f"  週当たり R 数 中央値={active['races'].median():.0f}  平均={active['races'].mean():.1f}")
        print(f"  週 ¥100,000 以上投資した週: {(active['bet'] >= 100000).sum()} / {len(active)}")

    # ===== 診断 =====
    print("\n— 診断: HALO source 内訳 —")
    halo_rows = res[res["plan"] == "HALO"]
    if len(halo_rows):
        # race 単位で集計 (1R に複数点あるので race_id で ensure unique)
        halo_uniq = halo_rows.drop_duplicates(subset=["race_id"])
        src_cnt = halo_uniq.groupby(["halo_source", "halo_pattern"]).size().sort_values(ascending=False)
        for (src, pat), n in src_cnt.items():
            # ROI for that source
            src_bets = halo_rows[halo_rows["halo_source"] == src]
            bet_sum = src_bets["購入額"].sum(); pay_sum = src_bets["払戻"].sum()
            roi = (pay_sum / bet_sum * 100) if bet_sum else 0
            print(f"  source={src:<30s} pattern={pat:<30s} R={n:>5d}  ROI={roi:>6.2f}%")

    print("\n— 診断: STANDARD EV 分布 —")
    std_rows = res[res["plan"] == "STANDARD"].copy()
    std_rows["EV_num"] = pd.to_numeric(std_rows["EV"], errors="coerce")
    for bt_type in ["単勝", "複勝", "馬連"]:
        sub = std_rows[std_rows["馬券種"] == bt_type]
        if len(sub) == 0: continue
        evs = sub["EV_num"].dropna()
        if len(evs) == 0:
            print(f"  {bt_type:<6s} EV データなし"); continue
        # EV ビン別 ROI
        print(f"  {bt_type:<6s} n={len(sub):>5d}  EV 中央値={evs.median():.3f}  平均={evs.mean():.3f}")
        for lo, hi in [(1.00,1.10), (1.10,1.30), (1.30,1.60), (1.60,2.00), (2.00,99)]:
            b = sub[(sub["EV_num"] >= lo) & (sub["EV_num"] < hi)]
            if len(b) == 0: continue
            bet_s = b["購入額"].sum(); pay_s = b["払戻"].sum()
            roi = (pay_s / bet_s * 100) if bet_s else 0
            hit = b["的中"].mean() * 100
            print(f"    EV [{lo:.2f},{hi:.2f}) n={len(b):>5d}  ROI={roi:>6.2f}%  hit={hit:>5.2f}%")

    print(f"\n結果 CSV: {BASE_DIR / 'reports' / 'backtest_v2_results.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_races", type=int, default=None)
    ap.add_argument("--period", type=str, default="test", choices=["test", "2024", "2025"])
    args = ap.parse_args()
    run(args)
