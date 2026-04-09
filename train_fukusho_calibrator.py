"""
train_fukusho_calibrator.py

既存 backtest_results_*.csv (raw ensemble prob + 的中 flag) を使って
複勝専用 Isotonic キャリブレータを学習する。

- Train: reports/backtest_results_train.csv (2013-2022)
- Test:  reports/backtest_results.csv (2024)
- 対象券種: 複勝のみ
- 特徴: 推定的中確率 (raw ensemble prob for fukusho_flag)
- ターゲット: 的中 (1/0)
"""
from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).parent
TRAIN_CSV = Path(r"E:\PyCaLiAI\reports\backtest_results_train.csv")
TEST_CSV  = BASE / "reports" / "backtest_results.csv"
OUT_PKL   = BASE / "models" / "fukusho_calibrator_v1.pkl"
OUT_PNG   = BASE / "reports" / "fukusho_calibration_curve.png"
OUT_MD    = BASE / "reports" / "fukusho_calibrator_results.md"

# cp932 列名 → 正式
COL_MAP = {
    "race_id": "race_id",
    "\u0093\u00fa\u0095t": "日付",
    "\u008f\u0087\u008f\u008a": "場所",
    "\u008b\u009f\u009b": "距離",
    "\u008e\u00c5\u0083_": "芝ダ",
    "\u008aN\u0083\u0089\u0083X": "クラス",
    "\u008an\u0097Y\u008eX": "馬券種",
    "\u008ac\u0082\u00a2\u0082\u00da": "買い目",
    "\u0090\u0084\u0092\u008a\u0093I\u0092\u0087\u008a\u006d\u008a\u00a6": "推定的中確率",
    "\u0090\u0084\u0092\u008a\u0093I\u0083b\u0083Y": "推定オッズ",
    "\u0090\u0084\u0092\u008a\u008a\u00fa\u0091\u00d2\u0092l": "推定期待値",
    "\u008dw\u0093\u00fc\u008az": "購入額",
    "\u008eY\u0093\u0096(100\u0089~)": "配当100",
    "\u008e\u00c0\u0083b\u0083Y": "実オッズ",
    "\u0093I\u0092\u0087": "的中",
    "\u008e\u00c0\u0089\u00f1\u008e\u00fb\u008az": "実回収",
    "\u0097\u0098\u0076": "利益",
}


def load_fukusho(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    # 14列目が馬券種(cp932 mojibake化した列名)。位置で取得
    # 正規の位置: columns[5]=クラス, columns[6]=馬券種
    bet_col = df.columns[6]
    prob_col = df.columns[8]
    hit_col = df.columns[14]
    df = df.rename(columns={bet_col: "bet_type", prob_col: "prob", hit_col: "hit"})
    sub = df[df["bet_type"] == "複勝"].copy()
    sub["prob"] = pd.to_numeric(sub["prob"], errors="coerce")
    sub["hit"] = pd.to_numeric(sub["hit"], errors="coerce")
    sub = sub.dropna(subset=["prob", "hit"])
    return sub[["prob", "hit"]].reset_index(drop=True)


def calibration_bins(p: np.ndarray, y: np.ndarray, n_bins: int = 10):
    edges = np.linspace(0, 1, n_bins + 1)
    mids, emps = [], []
    for i in range(n_bins):
        mask = (p >= edges[i]) & (p < edges[i + 1] if i < n_bins - 1 else p <= edges[i + 1])
        if mask.sum() > 0:
            mids.append(p[mask].mean())
            emps.append(y[mask].mean())
    return np.array(mids), np.array(emps)


def main():
    print("Loading train/test...")
    tr = load_fukusho(TRAIN_CSV)
    te = load_fukusho(TEST_CSV)
    print(f"  train n={len(tr):,} hit_rate={tr['hit'].mean():.3f}")
    print(f"  test  n={len(te):,} hit_rate={te['hit'].mean():.3f}")

    # Fit Isotonic
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(tr["prob"].values, tr["hit"].values)
    print("Isotonic fitted.")

    # Test metrics
    raw = te["prob"].values.astype(float)
    y = te["hit"].values.astype(int)
    cal = iso.predict(raw)

    brier_raw = brier_score_loss(y, raw)
    brier_cal = brier_score_loss(y, cal)
    ll_raw = log_loss(y, np.clip(raw, 1e-9, 1 - 1e-9))
    ll_cal = log_loss(y, np.clip(cal, 1e-9, 1 - 1e-9))

    print(f"\n=== Test (2024) ===")
    print(f"  Brier: raw={brier_raw:.5f} → cal={brier_cal:.5f} ({brier_cal-brier_raw:+.5f})")
    print(f"  LogLoss: raw={ll_raw:.5f} → cal={ll_cal:.5f} ({ll_cal-ll_raw:+.5f})")

    # Calibration curves
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfect")
    mx, my = calibration_bins(raw, y)
    ax.plot(mx, my, "o-", label=f"raw (Brier {brier_raw:.4f})")
    cx, cy = calibration_bins(cal, y)
    ax.plot(cx, cy, "s-", label=f"cal (Brier {brier_cal:.4f})")
    ax.set_xlabel("predicted prob")
    ax.set_ylabel("empirical hit rate")
    ax.set_title("Fukusho Calibration Curve (2024 Test)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=120)
    print(f"Saved: {OUT_PNG}")

    OUT_PKL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(iso, OUT_PKL)
    print(f"Saved: {OUT_PKL}")

    # === Filter-gate backtest ===
    # 複勝◎ にフィルタ cal_prob > threshold を掛けた場合の ROI
    # te: prob, hit. 100円賭けなら bet=100 per row.
    # オッズは raw なら推定オッズ。payout = hit_actual * 実オッズ * 100.
    # backtest_results.csv に実回収列があるのでそれを使う
    te_full = pd.read_csv(TEST_CSV, encoding="utf-8-sig")
    bet_col = te_full.columns[6]
    prob_col = te_full.columns[8]
    hit_col = te_full.columns[14]
    buy_col = te_full.columns[11]  # 購入額
    ret_col = te_full.columns[15]  # 実回収
    fuku = te_full[te_full[bet_col] == "複勝"].copy()
    fuku["prob"] = pd.to_numeric(fuku[prob_col], errors="coerce")
    fuku["buy"] = pd.to_numeric(fuku[buy_col], errors="coerce")
    fuku["ret"] = pd.to_numeric(fuku[ret_col], errors="coerce")
    fuku = fuku.dropna(subset=["prob", "buy", "ret"])
    fuku["cal_prob"] = iso.predict(fuku["prob"].values)

    bt_rows = []
    # baseline (no filter)
    base_bet = fuku["buy"].sum()
    base_ret = fuku["ret"].sum()
    base_roi = base_ret / base_bet * 100 if base_bet > 0 else 0
    bt_rows.append({
        "filter": "baseline",
        "threshold": 0.0,
        "n_bets": len(fuku),
        "bet": int(base_bet),
        "ret": int(base_ret),
        "ROI%": round(base_roi, 1),
        "hit%": round(pd.to_numeric(fuku[hit_col], errors="coerce").mean() * 100, 1),
    })

    for th in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
        sub = fuku[fuku["cal_prob"] >= th]
        if sub.empty:
            continue
        b = sub["buy"].sum()
        r = sub["ret"].sum()
        roi = r / b * 100 if b > 0 else 0
        hit = pd.to_numeric(sub[hit_col], errors="coerce").mean() * 100
        bt_rows.append({
            "filter": f"cal>={th:.2f}",
            "threshold": th,
            "n_bets": len(sub),
            "bet": int(b),
            "ret": int(r),
            "ROI%": round(roi, 1),
            "hit%": round(hit, 1),
        })

    bt_df = pd.DataFrame(bt_rows)
    print("\n=== Filter-Gate Backtest (Fukusho, 2024) ===")
    print(bt_df.to_string(index=False))
    bt_df.to_csv(BASE / "reports" / "fukusho_calibrator_summary.csv",
                 index=False, encoding="utf-8-sig")

    # Markdown
    lines = ["# 複勝専用 Isotonic キャリブレータ 検証結果", ""]
    lines.append(f"- train: {TRAIN_CSV.name} ({len(tr):,} 複勝bet)")
    lines.append(f"- test: {TEST_CSV.name} ({len(te):,} 複勝bet)")
    lines.append("")
    lines.append("## キャリブレーション精度 (2024)")
    lines.append(f"- Brier: raw={brier_raw:.5f} → cal={brier_cal:.5f} ({brier_cal-brier_raw:+.5f})")
    lines.append(f"- LogLoss: raw={ll_raw:.5f} → cal={ll_cal:.5f} ({ll_cal-ll_raw:+.5f})")
    lines.append("")
    lines.append("## フィルタ閾値別 ROI (2024, 複勝全印)")
    lines.append("")
    lines.append("| filter | n_bets | bet | ret | ROI% | hit% |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for _, r in bt_df.iterrows():
        lines.append(f"| {r['filter']} | {r['n_bets']} | {r['bet']:,} | {r['ret']:,} | {r['ROI%']} | {r['hit%']} |")
    lines.append("")
    # best
    best = bt_df.sort_values("ROI%", ascending=False).iloc[0]
    lines.append(f"## 推奨閾値: **{best['filter']}**")
    lines.append(f"- ROI {best['ROI%']}% (baseline {base_roi:.1f}% から {best['ROI%']-base_roi:+.1f}pt)")
    lines.append(f"- 賭け数 {best['n_bets']} (baseline {len(fuku)})")
    lines.append(f"- 的中率 {best['hit%']}%")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved: {OUT_MD}")


if __name__ == "__main__":
    main()
