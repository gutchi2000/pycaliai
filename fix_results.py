"""results.json の的中フラグをpred CSV × kekkaで全件検証して修正する。
0→1（新的中）も kekka から払戻額を自動補完する。
"""
import json, pandas as pd, copy
from pathlib import Path

with open("data/results.json", encoding="utf-8") as f:
    rj = json.load(f)

kekka_cache, pred_cache = {}, {}

def load_kekka(ds):
    if ds in kekka_cache: return kekka_cache[ds]
    p = Path(f"data/kekka/{ds}.csv")
    if not p.exists(): return None
    df = pd.read_csv(p, encoding="cp932"); kekka_cache[ds] = df; return df

def load_pred(ds):
    if ds in pred_cache: return pred_cache[ds]
    p = Path(f"reports/pred_{ds}.csv")
    if not p.exists(): return None
    df = pd.read_csv(p, encoding="utf-8"); pred_cache[ds] = df; return df

def parse_date(raw):
    parts = raw.replace("-", ".").split(".")
    return f"{parts[0]}{parts[1].zfill(2)}{parts[2].zfill(2)}" if len(parts) >= 3 else raw.replace(".", "")

def to_int(v):
    try: return int(float(str(v).strip()))
    except: return None

def get_top(race_kk, n):
    rows = race_kk[race_kk.iloc[:, 6].astype(str).isin([str(i) for i in range(1, n+1)])]
    return sorted([int(x) for x in rows.iloc[:, 4].tolist()])

def get_bets(race_pred, col):
    if col not in race_pred.columns: return []
    return [str(b).strip() for b in race_pred[col].dropna().unique() if str(b).strip() not in ["", "nan"]]

def split_combos(bets):
    """'1-2 / 2-3' 形式を ['1-2', '2-3'] に展開する。"""
    combos = []
    for b in bets:
        for part in b.split("/"):
            part = part.strip()
            if part: combos.append(part)
    return combos

def check_rentan(bets, top2):
    for b in split_combos(bets):
        horses = {to_int(x) for x in b.split("-") if to_int(x)}
        if horses == set(top2): return True
    return False

def check_sanrenpuku(bets, top3):
    for b in split_combos(bets):
        horses = {to_int(x) for x in b.split("-") if to_int(x)}
        if horses == set(top3): return True
    return False

def check_fukusho(bets, top3):
    return any(to_int(b) in top3 for b in bets if to_int(b))

def check_tansho(bets, winner):
    return any(to_int(b) == winner for b in bets if to_int(b))

def get_payout(race_kk, bet_type, inv, bet_horses=None):
    """kekka から実際の払戻額（円）を取得する。inv=投資額(円)"""
    col_map = {"馬連": "馬連", "三連複": "３連複", "単勝": "単勝配当"}
    if bet_type in col_map:
        col = col_map[bet_type]
        if col not in race_kk.columns: return 0
        # 1着行から取得（または非空の最初の行）
        vals = race_kk[col].dropna()
        vals = vals[~vals.astype(str).str.startswith("(")]
        if len(vals) == 0: return 0
        try:
            odds_100 = float(str(vals.iloc[0]).replace(",", ""))
            return int(odds_100 * inv / 100)
        except: return 0
    elif bet_type == "複勝":
        # 対象馬の複勝配当を取得
        if not bet_horses: return 0
        col = "複勝配当"
        if col not in race_kk.columns: return 0
        horse = to_int(bet_horses[0])
        row = race_kk[race_kk.iloc[:, 4].astype(str).str.strip() == str(horse)]
        if len(row) == 0: return 0
        v = str(row[col].iloc[0]).strip()
        if v.startswith("("): v = v[1:-1]
        try: return int(float(v) * inv / 100)
        except: return 0
    return 0

fixes = []
rj_new = copy.deepcopy(rj)

for pk in ["HAHO", "HALO", "LALO", "CQC"]:
    races = rj_new[pk]["races"]
    for r in races:
        date_raw = r["日付"]; venue = r["場所"].strip(); race_no = str(r["R"])
        ds = parse_date(date_raw)
        kk = load_kekka(ds); pred = load_pred(ds)
        if kk is None or pred is None: continue

        race_kk = kk[(kk.iloc[:, 1].astype(str).str.strip() == venue) & (kk.iloc[:, 2].astype(str) == race_no)]
        if len(race_kk) == 0: continue

        top3 = get_top(race_kk, 3)
        top2 = get_top(race_kk, 2)
        winner_rows = race_kk[race_kk.iloc[:, 6].astype(str) == "1"].iloc[:, 4]
        winner = to_int(winner_rows.iloc[0]) if len(winner_rows) > 0 else None
        race_pred = pred[(pred["場所"].astype(str).str.strip() == venue) & (pred["R"].astype(str) == race_no)]

        changed = False

        def apply_fix(r, key_ret, key_hit, actual_hit, inv, bet_type, bets, race_kk):
            recorded = r.get(key_hit, 0)
            if inv <= 0 or recorded == int(actual_hit): return False
            direction = "0→1" if actual_hit else "1→0"
            if not actual_hit:
                r[key_ret] = 0.0; r[key_hit] = 0
            else:
                pay = get_payout(race_kk, bet_type, inv, bets)
                r[key_ret] = float(pay); r[key_hit] = 1
            return True, direction

        if pk == "HAHO":
            b_ren = get_bets(race_pred, "HAHO_馬連_買い目")
            actual_ren = check_rentan(b_ren, top2) if b_ren else False
            inv_ren = r.get("馬連_投資", 0) or 0
            res = apply_fix(r, "馬連_払戻", "馬連_的中", actual_ren, inv_ren, "馬連", b_ren, race_kk)
            if res:
                fixes.append(f"FIX {pk} {date_raw} {venue} {race_no}R 馬連: 的中{res[1]} 買:{b_ren} 実:{top2} 払:{r['馬連_払戻']:.0f}")
                changed = True

            b_san = get_bets(race_pred, "HAHO_三連複_買い目")
            actual_san = check_sanrenpuku(b_san, top3) if b_san else False
            inv_san = r.get("三連複_投資", 0) or 0
            res = apply_fix(r, "三連複_払戻", "三連複_的中", actual_san, inv_san, "三連複", b_san, race_kk)
            if res:
                fixes.append(f"FIX {pk} {date_raw} {venue} {race_no}R 三連複: 的中{res[1]} 買:{b_san} 実:{top3} 払:{r['三連複_払戻']:.0f}")
                changed = True

        elif pk == "HALO":
            b = get_bets(race_pred, "HALO_三連複_買い目")
            actual_hit = check_sanrenpuku(b, top3) if b else False
            inv = r.get("三連複_投資", 0) or 0
            res = apply_fix(r, "三連複_払戻", "三連複_的中", actual_hit, inv, "三連複", b, race_kk)
            if res:
                fixes.append(f"FIX {pk} {date_raw} {venue} {race_no}R 三連複: 的中{res[1]} 買:{b} 実:{top3} 払:{r['三連複_払戻']:.0f}")
                changed = True

        elif pk == "LALO":
            b = get_bets(race_pred, "LALO_複勝_買い目")
            actual_hit = check_fukusho(b, top3) if b else False
            inv = r.get("複勝_投資", 0) or 0
            res = apply_fix(r, "複勝_払戻", "複勝_的中", actual_hit, inv, "複勝", b, race_kk)
            if res:
                fixes.append(f"FIX {pk} {date_raw} {venue} {race_no}R 複勝: 的中{res[1]} 買:{b} 実:{top3} 払:{r['複勝_払戻']:.0f}")
                changed = True

        elif pk == "CQC":
            b = get_bets(race_pred, "CQC_単勝_買い目")
            actual_hit = check_tansho(b, winner) if b else False
            inv = r.get("単勝_投資", 0) or 0
            res = apply_fix(r, "単勝_払戻", "単勝_的中", actual_hit, inv, "単勝", b, race_kk)
            if res:
                fixes.append(f"FIX {pk} {date_raw} {venue} {race_no}R 単勝: 的中{res[1]} 買:{b} 実1着:{winner} 払:{r['単勝_払戻']:.0f}")
                changed = True

        if changed:
            pay_keys = [k for k in r if "払戻" in k and "総" not in k]
            r["総払戻"] = sum(r.get(k, 0) or 0 for k in pay_keys)
            r["収支"] = r["総払戻"] - r["総投資"]

    # totals 再計算
    t = rj_new[pk]["total"]
    t["bet"] = sum(r.get("総投資", 0) or 0 for r in races)
    t["ret"] = sum(r.get("総払戻", 0) or 0 for r in races)
    t["pnl"] = t["ret"] - t["bet"]
    t["roi"] = round(t["ret"] / t["bet"] * 100, 1) if t["bet"] > 0 else 0

print("=== 修正内容 ===")
for f in fixes:
    print(f)
print(f"\n修正件数: {len(fixes)}")

with open("data/results.json", "w", encoding="utf-8") as f:
    json.dump(rj_new, f, ensure_ascii=False, indent=2)
print("results.json 保存完了\n")

print("=== 修正後 totals ===")
for pk in ["HAHO", "HALO", "LALO", "CQC"]:
    t = rj_new[pk]["total"]
    print(f"{pk}: ROI {t['roi']}%  bet={t['bet']:,}  ret={t['ret']:,}  pnl={t['pnl']:+,}")
