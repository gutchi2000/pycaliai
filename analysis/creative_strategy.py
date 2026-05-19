"""
creative_strategy.py
====================
ユーザー指示「あなたが好きなように馬券組んだらプラスになるかね？」への応答。

v4 モデルをベースに、オッズフィルタ付きの複数戦略を試し、
test 2024-2025 の 3 年 OOS (valid 年含む) で + ROI を目指す。

戦略候補:
  A: 複勝◎ (top1 単勝オッズ ≤ 閾値)
  B: 複勝◎+複勝〇 (top1 単勝オッズ ≤ 閾値)
  C: 馬連◎-〇 (top1 単勝オッズ ≤ 閾値)
  D: ワイド◎-〇 (top1 単勝オッズ ≤ 閾値)
  E: 複勝◎ (top1 と top2 のオッズ差が閾値以内) ← confidence
  F: 馬連◎〇流し (7点) + 単勝◎ 併用 (top1 オッズ ≤ 閾値)
  G: 複勝◎ + 馬連◎-〇 + ワイド◎-〇 合成 (top1 オッズ ≤ 閾値)

各戦略で閾値を振り、ROI と race 数を記録。
条件: races ≥ 300 かつ 3年 ROI > 100% を満たす設定を出力。
"""
from __future__ import annotations
import io
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
warnings.filterwarnings("ignore")

from backtest_pl_ev import (
    load_payouts, score_test, COL_RID, COL_BAN,
)
import backtest_pl_ev as be

BASE = Path(__file__).parent
UNIT = 100

# ---------------------------------------------------------------
# オッズ読み込み
# ---------------------------------------------------------------
print("[odds] loading ...")
odds_files = [
    BASE / "data/odds/odds_20230105-20231228.csv",
    BASE / "data/odds/odds_20240106-20241228.csv",
    BASE / "data/odds/odds_20250105-20261228.csv",
]
odds_df = []
for f in odds_files:
    if not f.exists():
        continue
    d = pd.read_csv(f, encoding="cp932")
    odds_df.append(d)
odds = pd.concat(odds_df, ignore_index=True)
odds = odds.rename(columns={
    "レースID(新)": "rid",
    "馬番": "ban",
    "単勝オッズ": "tansho_odds",
    "複勝オッズ下限": "fuku_low",
    "複勝オッズ上限": "fuku_high",
})
odds["rid_s"] = odds["rid"].astype(str).str[:16]  # race_id (馬番除く) 16桁
odds["ban"] = pd.to_numeric(odds["ban"], errors="coerce").astype("Int64")
odds["tansho_odds"] = pd.to_numeric(odds["tansho_odds"], errors="coerce")
print(f"  odds rows: {len(odds):,}  races: {odds['rid_s'].nunique():,}")

# ---------------------------------------------------------------
# モデル設定 + 予測
# ---------------------------------------------------------------
tag = "v4"
be.MODEL_PKL = BASE / f"models/unified_rank_{tag}.pkl"
be.CAL_PKL   = BASE / f"models/pl_calibrators_{tag}.pkl"
be.CURVE_PKL = BASE / f"data/pl_payout_curve_{tag}.pkl"
print(f"[model] {be.MODEL_PKL}")

payouts = load_payouts()
print(f"[payouts] races={len(payouts):,}")

# ワイド配当
wide_pq = BASE / "data/wide_payouts_2016-2025.parquet"
wide_df = pd.read_parquet(wide_pq)
wide_df["rid_s"] = wide_df["race_id"].astype(str).str[:16]
# wide_map[rid_s] = list of (ban_i, ban_j, pay)  (3 combinations per race)
wide_map: dict[str, list] = {}
for _, r in wide_df.iterrows():
    rid_s = r["rid_s"]
    lst = []
    for col_i, col_j, col_p in [("w1_i","w1_j","w1_pay"), ("w2_i","w2_j","w2_pay"), ("w3_i","w3_j","w3_pay")]:
        try:
            i = int(r[col_i]); j = int(r[col_j])
            p = float(r[col_p])
            if p > 0:
                lst.append((min(i,j), max(i,j), p))
        except Exception:
            pass
    wide_map[rid_s] = lst
print(f"[wide] races={len(wide_map):,}")

te = score_test(include_valid=True)
print(f"[test+valid] rows={len(te):,}  races={te[COL_RID].nunique():,}")

# ---------------------------------------------------------------
# race 単位で top-5 と ◎〇▲△△ を抽出 + オッズ結合
# ---------------------------------------------------------------
print("[build] race-level table ...")
rows = []
skipped = 0

# oddsをrid_s+banでインデックス化
odds_idx = odds.set_index(["rid_s", "ban"])["tansho_odds"].to_dict()

for rid, g in te.groupby(COL_RID, sort=False):
    if len(g) < 5:
        skipped += 1; continue
    rid_s = str(int(rid)) if isinstance(rid, (int, np.integer)) else str(rid)
    if rid_s not in payouts:
        skipped += 1; continue
    po = payouts[rid_s]

    g = g.sort_values(COL_BAN).reset_index(drop=True)
    ban = g[COL_BAN].astype(int).values
    b2i = {int(b): i for i, b in enumerate(ban)}

    wi = b2i.get(po["win"]); pi_ = b2i.get(po["plc"])
    if wi is None or pi_ is None:
        skipped += 1; continue

    order = np.argsort(-g["_score"].values)  # index
    hon_idx = int(order[0]); tai_idx = int(order[1])
    hon_ban = int(ban[hon_idx]); tai_ban = int(ban[tai_idx])
    top5_idx = [int(x) for x in order[:5]]
    top5_set = set(top5_idx)
    top2_set = {hon_idx, tai_idx}

    # オッズ
    hon_odds = odds_idx.get((rid_s, hon_ban), np.nan)
    tai_odds = odds_idx.get((rid_s, tai_idx), np.nan)
    if pd.isna(hon_odds):
        skipped += 1; continue

    # 配当 (pay は per-100)
    tansho_pay = float(po.get("tansho") or 0.0)
    fuku_pay_win = float(po.get("fuku_win") or 0.0)
    fuku_pay_plc = float(po.get("fuku_plc") or 0.0)
    fuku_pay_sho = float(po.get("fuku_sho") or 0.0)
    umaren_pay = float(po.get("umaren") or 0.0)

    # ◎ が 1 着?
    hit_tansho_hon = (hon_idx == wi)
    # ◎ が 1-3 着? (複勝)
    sho_idx = b2i.get(po.get("sho")) if po.get("sho") is not None else None
    hit_fuku_hon = hon_idx in {wi, pi_, sho_idx}
    # 〇 が 1-3 着?
    hit_fuku_tai = tai_idx in {wi, pi_, sho_idx}
    # 馬連 ◎-〇 (順不同)
    hit_umaren_hono = {hon_idx, tai_idx} == {wi, pi_}
    # 馬連 7点流し (◎〇軸 × 5頭相手) = {1着,2着}⊂top5 AND {1着,2着}∩top2≠∅
    hit_umaren_nagashi = ({wi, pi_}.issubset(top5_set)) and bool({wi, pi_} & top2_set)

    # 複勝◎配当
    if hit_fuku_hon:
        if hon_idx == wi: fuku_ret_hon = fuku_pay_win
        elif hon_idx == pi_: fuku_ret_hon = fuku_pay_plc
        elif sho_idx is not None and hon_idx == sho_idx: fuku_ret_hon = fuku_pay_sho
        else: fuku_ret_hon = 0.0
    else:
        fuku_ret_hon = 0.0
    # 複勝〇配当
    if hit_fuku_tai:
        if tai_idx == wi: fuku_ret_tai = fuku_pay_win
        elif tai_idx == pi_: fuku_ret_tai = fuku_pay_plc
        elif sho_idx is not None and tai_idx == sho_idx: fuku_ret_tai = fuku_pay_sho
        else: fuku_ret_tai = 0.0
    else:
        fuku_ret_tai = 0.0

    # ワイド 配当取得
    wide_hono_pay = 0.0
    hit_wide_hono = False
    if rid_s in wide_map:
        for (bi, bj, p) in wide_map[rid_s]:
            if {bi, bj} == {hon_ban, tai_ban}:
                wide_hono_pay = p
                hit_wide_hono = True
                break

    # 7点流しの actual ret
    if hit_umaren_nagashi:
        umaren_nagashi_ret = umaren_pay
    else:
        umaren_nagashi_ret = 0.0

    year = int(g["year"].iloc[0])

    rows.append({
        "rid_s": rid_s,
        "year": year,
        "hon_odds": hon_odds,
        "tai_odds": tai_odds,
        # 単勝◎
        "hit_tansho_hon": int(hit_tansho_hon),
        "ret_tansho_hon": tansho_pay if hit_tansho_hon else 0.0,
        # 複勝◎
        "hit_fuku_hon": int(hit_fuku_hon),
        "ret_fuku_hon": fuku_ret_hon,
        # 複勝〇
        "hit_fuku_tai": int(hit_fuku_tai),
        "ret_fuku_tai": fuku_ret_tai,
        # 馬連 ◎-〇 (1点)
        "hit_umaren_hono": int(hit_umaren_hono),
        "ret_umaren_hono": umaren_pay if hit_umaren_hono else 0.0,
        # 馬連 7点流し
        "hit_umaren_nagashi": int(hit_umaren_nagashi),
        "ret_umaren_nagashi": umaren_nagashi_ret,
        # ワイド◎-〇 (1点)
        "hit_wide_hono": int(hit_wide_hono),
        "ret_wide_hono": wide_hono_pay,
    })

df = pd.DataFrame(rows)
print(f"[build] races with odds: {len(df):,}  skipped: {skipped:,}")
df.to_csv(BASE / "reports/creative_race_table.csv", index=False, encoding="utf-8-sig")
print(f"  saved: reports/creative_race_table.csv")

# ---------------------------------------------------------------
# Strategy evaluators
# ---------------------------------------------------------------
def eval_strategy(sub, stake_per_race, ret_col, name):
    R = len(sub)
    if R == 0:
        return None
    stake = R * stake_per_race
    ret = sub[ret_col].sum()
    roi = ret / stake
    hits = (sub[ret_col] > 0).sum()
    hr = hits / R
    return {
        "name": name,
        "races": R,
        "stake": stake,
        "ret": int(ret),
        "roi_pct": roi * 100,
        "hit_pct": hr * 100,
        "pl": int(ret - stake),
    }


def print_table(res_list, header):
    print(f"\n=== {header} ===")
    print(f"  {'strategy':40s} {'R':>5s} {'hit%':>6s} {'ROI%':>7s} {'P&L':>10s}")
    for r in res_list:
        if r is None: continue
        flag = "✅" if r["roi_pct"] >= 100 and r["races"] >= 300 else ("🔶" if r["roi_pct"] >= 95 and r["races"] >= 300 else "  ")
        print(f"  {flag} {r['name']:38s} {r['races']:>5d} {r['hit_pct']:>5.1f}% {r['roi_pct']:>6.2f}% {r['pl']:>+10,d}")


# ---------------------------------------------------------------
# 実行: 3年 (2023-2025) + 真OOS (2024-2025) で評価
# ---------------------------------------------------------------
for label, year_set in [("3年 (2023-2025)", {2023, 2024, 2025}), ("真OOS (2024-2025)", {2024, 2025})]:
    base = df[df["year"].isin(year_set)].copy()
    print(f"\n{'='*70}")
    print(f"【{label}】base races = {len(base):,}")
    print(f"{'='*70}")

    results = []

    # --- Strategy A: 複勝◎ only (no filter) ---
    results.append(eval_strategy(base, UNIT, "ret_fuku_hon", "A0 複勝◎ ALL"))

    # --- Strategy A: 複勝◎ with odds filter ---
    for thr in [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        sub = base[base["hon_odds"] <= thr]
        results.append(eval_strategy(sub, UNIT, "ret_fuku_hon", f"A 複勝◎ (odds≤{thr})"))

    print_table(results, f"{label} - A: 複勝◎")
    results = []

    # --- Strategy B: 複勝◎+〇 合計 (200/race) ---
    for thr in [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 999]:
        sub = base[base["hon_odds"] <= thr].copy()
        sub["ret_B"] = sub["ret_fuku_hon"] + sub["ret_fuku_tai"]
        results.append(eval_strategy(sub, UNIT*2, "ret_B", f"B 複勝◎+〇 (odds≤{thr})"))

    print_table(results, f"{label} - B: 複勝◎+〇")
    results = []

    # --- Strategy C: 馬連◎-〇 1 点 ---
    for thr in [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 999]:
        sub = base[base["hon_odds"] <= thr]
        results.append(eval_strategy(sub, UNIT, "ret_umaren_hono", f"C 馬連◎-〇 (odds≤{thr})"))

    print_table(results, f"{label} - C: 馬連◎-〇 1点")
    results = []

    # --- Strategy D: ワイド◎-〇 1 点 ---
    for thr in [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 999]:
        sub = base[base["hon_odds"] <= thr]
        results.append(eval_strategy(sub, UNIT, "ret_wide_hono", f"D ワイド◎-〇 (odds≤{thr})"))

    print_table(results, f"{label} - D: ワイド◎-〇 1点")
    results = []

    # --- Strategy E: 複勝◎ (オッズ差フィルタ: ◎-〇 のオッズ差 → ◎ 独走) ---
    for thr in [1.0, 1.5, 2.0, 3.0, 5.0, 999]:
        sub = base[(base["tai_odds"] - base["hon_odds"]) >= thr]
        results.append(eval_strategy(sub, UNIT, "ret_fuku_hon", f"E 複勝◎ (tai-hon≥{thr})"))

    print_table(results, f"{label} - E: 複勝◎ (独走度フィルタ)")
    results = []

    # --- Strategy F: 馬連 7点流し + odds フィルタ (低オッズ = 固い展開) ---
    for thr in [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 999]:
        sub = base[base["hon_odds"] <= thr]
        results.append(eval_strategy(sub, UNIT*7, "ret_umaren_nagashi", f"F 馬連7点流し (odds≤{thr})"))

    print_table(results, f"{label} - F: 馬連 7点流し")
    results = []

    # --- Strategy G: 複勝◎ + ワイド◎-〇 (2点買い) ---
    for thr in [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 999]:
        sub = base[base["hon_odds"] <= thr].copy()
        sub["ret_G"] = sub["ret_fuku_hon"] + sub["ret_wide_hono"]
        results.append(eval_strategy(sub, UNIT*2, "ret_G", f"G 複勝◎+ワイド◎-〇 (odds≤{thr})"))

    print_table(results, f"{label} - G: 複勝◎+ワイド◎-〇")
    results = []

    # --- Strategy H: 高オッズ穴狙い: ◎ のオッズが高い場合のみ馬連 7点流し ---
    for lo, hi in [(3.0, 999), (5.0, 999), (8.0, 999), (3.0, 8.0), (5.0, 15.0)]:
        sub = base[(base["hon_odds"] >= lo) & (base["hon_odds"] < hi)]
        results.append(eval_strategy(sub, UNIT*7, "ret_umaren_nagashi", f"H 馬連7点流し ({lo}≤odds<{hi})"))

    print_table(results, f"{label} - H: 高オッズ帯 馬連7点流し")

print("\n" + "="*70)
print("[done]")
