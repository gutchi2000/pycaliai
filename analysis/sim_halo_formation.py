"""
sim_halo_formation.py
HALO旧ロジック（◎◯2頭軸マルチ24点×¥400）と
新ロジック（AI/スコアベース フォーメーション）のROI比較。

データ源:
  - reports/backtest_marks_2024.csv   (印: hon/taikou/sabo/delta/batsu)
  - reports/backtest_results_2024.csv (race_id→race16 紐付け用)
  - data/kekka_*.csv                  (三連単払戻 + 着順)
  - data/strategy_weights.json        (HALO対象 場所×クラス)

使い方:
    python sim_halo_formation.py
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import pandas as pd

BASE       = Path(__file__).parent
BT_CSV     = BASE / "reports/backtest_results_2024.csv"
MARKS_CSV  = BASE / "reports/backtest_marks_2024.csv"
KEKKA_CSV  = BASE / "data/kekka_20160105_20251228_v2.csv"
STRAT_JSON = BASE / "data/strategy_weights.json"

BUDGET_OLD = 9600       # 旧: 24点 × ¥400
BUDGET_NEW = 9600       # 新: point数×per_bet ≤ 9600

CLASS_NORMALIZE = {
    "新馬":"新馬","未勝利":"未勝利","1勝":"1勝","500万":"1勝",
    "2勝":"2勝","1000万":"2勝","3勝":"3勝","1600万":"3勝",
    "OP(L)":"OP(L)","Ｇ１":"Ｇ１","Ｇ２":"Ｇ２","Ｇ３":"Ｇ３",
}


# ── 1. HALO対象条件読み込み ────────────────────────────────────
with open(STRAT_JSON, encoding="utf-8") as f:
    sw = json.load(f)
halo_keys = set()
for place, classes in sw.items():
    for cls, bets in classes.items():
        if any("三連単" in b for b in bets.keys()):
            halo_keys.add((place, cls))
print(f"HALO対象条件: {len(halo_keys)} 組")


# ── 2. kekka 読み込み ─────────────────────────────────────────
print("kekka 読み込み中...")
kk = pd.read_csv(KEKKA_CSV, encoding="cp932", dtype=str)
kc = kk.columns.tolist()
kk["race16"] = kk[kc[7]].astype(str).str[:16]
kk["ban"]    = pd.to_numeric(kk[kc[4]], errors="coerce")
kk["jyun"]   = pd.to_numeric(kk[kc[6]], errors="coerce")
kk[kc[14]]   = pd.to_numeric(kk[kc[14]], errors="coerce")   # 三連単払戻
kk_dict = {r16: grp for r16, grp in kk.groupby("race16")}
print(f"  kekka レース数: {len(kk_dict):,}")


def top_ban(race16: str, jyun: int):
    sub = kk_dict.get(race16)
    if sub is None:
        return None
    r = sub[sub["jyun"] == jyun]["ban"].dropna()
    return int(r.iloc[0]) if not r.empty else None


def santan_payout(race16: str):
    sub = kk_dict.get(race16)
    if sub is None:
        return None, None, None, 0.0
    w1 = top_ban(race16, 1)
    w2 = top_ban(race16, 2)
    w3 = top_ban(race16, 3)
    r = sub[sub["jyun"] == 1]
    pay = 0.0
    if not r.empty:
        v = r.iloc[0][kc[14]]
        pay = float(v) if pd.notna(v) and v > 0 else 0.0
    return w1, w2, w3, pay


# ── 3. BT読み込み → HALO対象race_id抽出 ──────────────────────
print("BT読み込み...")
bt = pd.read_csv(BT_CSV, encoding="utf-8-sig")
c  = bt.columns.tolist()
race_col, place_col, cls_col = c[0], c[2], c[5]
bt["race16"]   = bt[race_col].astype(str).str[:16]
bt["cls_norm"] = bt[cls_col].map(CLASS_NORMALIZE).fillna(bt[cls_col])
halo_race_ids = set()
for _, row in bt.drop_duplicates(race_col).iterrows():
    if (row[place_col], row["cls_norm"]) in halo_keys:
        halo_race_ids.add(row[race_col])
race_to_r16 = bt.drop_duplicates(race_col).set_index(race_col)["race16"].to_dict()
print(f"HALO対象レース: {len(halo_race_ids):,}R")


# ── 4. 印 + スコア読み込み ─────────────────────────────────
marks_df = pd.read_csv(MARKS_CSV, encoding="utf-8-sig")
marks: dict[str, dict] = {}
for _, row in marks_df.iterrows():
    rid = row["race_id"]
    if rid not in halo_race_ids:
        continue
    m: dict = {}
    for key in ["hon", "taikou", "sabo", "delta", "batsu"]:
        v = row.get(key)
        if pd.notna(v):
            m[key] = int(v)
    # スコアがあれば拾う（列名は柔軟に）
    for k_src, k_dst in [
        ("hon_score", "s_hon"), ("taikou_score", "s_tai"),
        ("sabo_score", "s_sab"), ("delta_score", "s_del"),
    ]:
        v = row.get(k_src)
        if pd.notna(v):
            m[k_dst] = float(v)
    marks[rid] = m
print(f"◎◯揃い: {sum(1 for m in marks.values() if 'hon' in m and 'taikou' in m):,} / {len(marks):,} R")


# ── 5. ロジック定義 ────────────────────────────────────────
def logic_old(m: dict) -> list[tuple[int, int, int]]:
    """◎◯2頭軸マルチ: 1-2着 {◎◯} × 3着 {他4頭} = 24点"""
    if "hon" not in m or "taikou" not in m:
        return []
    h1, h2 = m["hon"], m["taikou"]
    others = [m[k] for k in ["sabo", "delta", "batsu"] if k in m]
    if len(others) < 1:
        return []
    combos = []
    for a, b in [(h1, h2), (h2, h1)]:
        for o in others:
            if o in (a, b):
                continue
            # 2着にも相手を追加 (1着固定, 2着{◎◯+他}, 3着{他})
            pass
    # 旧仕様: 2頭軸マルチ = {◎or◯}→{◎or◯}→{他} かつ {◎◯}→{他}→{◎or◯}
    combos = set()
    axis = [h1, h2]
    for f in axis:
        for s in axis:
            if f == s: continue
            for t in others:
                if t in (f, s): continue
                combos.add((f, s, t))
        for s in others:
            if s in axis: continue
            for t in axis:
                if t in (f, s): continue
                combos.add((f, s, t))
    return list(combos)


def logic_new(m: dict) -> list[tuple[int, int, int]]:
    """スコアベース フォーメーション（着順モデル無しフォールバック版）"""
    if "hon" not in m or "taikou" not in m:
        return []
    h1 = m["hon"]; h2 = m["taikou"]
    h3 = m.get("sabo"); h4 = m.get("delta"); h5 = m.get("batsu")

    s_hon = m.get("s_hon", 0.0)
    s_tai = m.get("s_tai", 0.0)
    s_del = m.get("s_del", 0.0)
    s_sab = m.get("s_sab", 0.0)
    gap_12   = s_hon - s_tai
    gap_top4 = (s_hon - s_del) if s_del > 0 else (s_hon - s_sab)

    if gap_12 >= 10:
        first  = [h1]
        second = [x for x in [h2, h3] if x is not None]
        third  = [x for x in [h2, h3, h4, h5] if x is not None]
    elif gap_12 <= 5 and gap_top4 <= 15:
        first  = [h1, h2]
        second = [x for x in [h1, h2, h3] if x is not None]
        third  = [x for x in [h1, h2, h3, h4] if x is not None]
    else:
        first  = [h1, h2]
        second = [x for x in [h1, h2, h3, h4] if x is not None]
        third  = [x for x in [h1, h2, h3, h4, h5] if x is not None]

    combos = set()
    for f in first:
        for s in second:
            for t in third:
                if len({f, s, t}) == 3:
                    combos.add((f, s, t))
    return list(combos)


# ── 6. シミュレーション ───────────────────────────────────────
def simulate(name: str, logic_fn) -> dict:
    tb = tr = hits = races = 0
    n_combos_total = 0
    per_bets: list[int] = []
    for rid, m in marks.items():
        combos = logic_fn(m)
        if not combos:
            continue
        r16 = race_to_r16.get(rid)
        if not r16:
            continue
        w1, w2, w3, pay = santan_payout(r16)
        n = len(combos)
        per = max(100, (BUDGET_NEW // n // 100) * 100)
        per_bets.append(per)
        n_combos_total += n
        races += 1
        race_hit = False
        for f, s, t in combos:
            tb += per
            if (w1, w2, w3) == (f, s, t):
                tr += per * pay / 100.0
                race_hit = True
        if race_hit:
            hits += 1
    roi = tr / tb * 100 if tb else 0.0
    hrate = hits / races * 100 if races else 0.0
    avg_pts = n_combos_total / races if races else 0.0
    return {
        "name": name,
        "races": races,
        "hits": hits,
        "hrate": hrate,
        "avg_points": avg_pts,
        "total_bet": tb,
        "total_ret": tr,
        "roi": roi,
    }


print("\nシミュレーション中...")
res_old = simulate("HALO旧 (◎◯マルチ24点)", logic_old)
res_new = simulate("HALO新 (フォーメーション)", logic_new)


# ── 7. 結果出力 ───────────────────────────────────────────────
def _fmt(r: dict) -> str:
    return (
        f"  対象R:      {r['races']:>5,}\n"
        f"  的中R:      {r['hits']:>5,}  ({r['hrate']:.2f}%)\n"
        f"  平均点数:   {r['avg_points']:>5.1f}\n"
        f"  投資総額:   {r['total_bet']:>10,.0f}円\n"
        f"  払戻総額:   {r['total_ret']:>10,.0f}円\n"
        f"  ROI:        {r['roi']:>5.2f}%\n"
    )


print("\n==========================================================")
print(f"【{res_old['name']}】")
print(_fmt(res_old))
print(f"【{res_new['name']}】")
print(_fmt(res_new))
print("==========================================================")
roi_diff = res_new["roi"] - res_old["roi"]
hr_diff  = res_new["hrate"] - res_old["hrate"]
print(f"新-旧 ROI差:      {roi_diff:+.2f}pt")
print(f"新-旧 的中率差:   {hr_diff:+.2f}pt")
