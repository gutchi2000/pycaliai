"""
sim_fukusho_wide_combo.py
複勝+ワイド セット購入 バックテスト

odds_test.csv (ワイド払戻) + kekka_v2.csv (複勝払戻/馬番別) +
backtest_marks_ev_test.csv (◎◯▲) + top1_features.csv (レース情報)

13パターン × 2024(IS)/2025(OOS) × 全会場/4会場 で ROI/的中率/DD/シャープを比較。
"""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(r"E:\PyCaLiAI")
ODDS_CSV = Path(r"E:\競馬過去走データ\odds_test.csv")
KEKKA_CSV = BASE / "data" / "kekka_20160105_20251228_v2.csv"
MARKS_CSV = BASE / "reports" / "backtest_marks_ev_test.csv"
FEAT_CSV = BASE / "reports" / "top1_features.csv"
OUT_MD = BASE / "reports" / "fukusho_wide_combo_results.md"
OUT_CSV = BASE / "reports" / "fukusho_wide_combo_summary.csv"

UNIT = 100  # 1点100円
FOUR_PLACES = {"中山", "中京", "新潟", "福島"}


def parse_wide(s: str) -> dict[frozenset, float]:
    res = {}
    if not isinstance(s, str):
        return res
    for part in s.split("/"):
        m = re.match(r"\s*(\d+)-(\d+)\s*\\(\d+)", part)
        if m:
            res[frozenset({int(m.group(1)), int(m.group(2))})] = float(m.group(3))
    return res


def load_odds() -> dict:
    print("Loading odds_test.csv (wide)...")
    df = pd.read_csv(ODDS_CSV, encoding="cp932", header=None)
    df.columns = ["年","月","日","場所","R","レース名","芝ダ","距離","天候","馬場",
                  "頭数","単勝","複勝","枠連","馬連","ワイド","馬単","三連複","三連単"]
    odds = {}
    for _, r in df.iterrows():
        try:
            key = (int(r["年"]), int(r["月"]), int(r["日"]),
                   str(r["場所"]).strip(), int(r["R"]))
            w = parse_wide(str(r["ワイド"]))
            if w:
                odds[key] = w
        except Exception:
            continue
    print(f"  wide races: {len(odds):,}")
    return odds


def load_fukusho() -> dict:
    """kekka v2 から (年,月,日,場所,R,馬番) -> 複勝払戻 を構築"""
    print("Loading kekka v2 (fukusho)...")
    df = pd.read_csv(KEKKA_CSV, encoding="cp932", low_memory=False)
    df.columns = ["日付","場所","R","枠番","馬番","馬名","着順","race_id",
                  "単勝","複勝","枠連","馬連","馬単","三連複","三連単"]
    df["複勝"] = pd.to_numeric(df["複勝"], errors="coerce").fillna(0.0)
    df["日付"] = df["日付"].astype(str)
    # 日付は "YYMMDD" の 6桁 (例: 251228)
    df["年"] = 2000 + df["日付"].str[:2].astype(int)
    df["月"] = df["日付"].str[2:4].astype(int)
    df["日"] = df["日付"].str[4:6].astype(int)
    df["R"] = pd.to_numeric(df["R"], errors="coerce").fillna(0).astype(int)
    df["馬番"] = pd.to_numeric(df["馬番"], errors="coerce").fillna(0).astype(int)
    fuku = {}
    for _, r in df.iterrows():
        key = (int(r["年"]), int(r["月"]), int(r["日"]),
               str(r["場所"]).strip(), int(r["R"]), int(r["馬番"]))
        fuku[key] = float(r["複勝"])
    print(f"  fukusho rows: {len(fuku):,}")
    return fuku


def load_marks() -> pd.DataFrame:
    print("Loading marks + features...")
    m = pd.read_csv(MARKS_CSV, encoding="utf-8-sig")
    m["rid"] = m["race_id"].apply(lambda x: str(int(x)).zfill(16))
    f = pd.read_csv(FEAT_CSV, encoding="utf-8")
    f["rid"] = f["race_id"].apply(lambda x: str(int(float(x))).zfill(16) if pd.notna(x) else None)
    info = f[["rid","日付","場所","Ｒ","クラス名"]].drop_duplicates("rid").dropna(subset=["rid"])
    info = info.rename(columns={"Ｒ":"R"})
    merged = m.merge(info, on="rid", how="inner")
    merged["年"] = merged["日付"].astype(str).str[:4].astype(int)
    merged["月"] = merged["日付"].astype(str).str[4:6].astype(int)
    merged["日"] = merged["日付"].astype(str).str[6:8].astype(int)
    merged["R"] = pd.to_numeric(merged["R"], errors="coerce").fillna(0).astype(int)
    merged["場所"] = merged["場所"].astype(str).str.strip()
    print(f"  merged races: {len(merged):,}")
    return merged


def simulate(merged: pd.DataFrame, odds: dict, fuku: dict) -> pd.DataFrame:
    """各レースについて13パターンの bet/return を計算"""
    patterns = ["A1","A2","A3","A4","B1","B2","B3","B4","C1","C2","C3","C4","C5"]
    rows = []
    skip = 0
    for _, r in merged.iterrows():
        key = (int(r["年"]), int(r["月"]), int(r["日"]),
               r["場所"], int(r["R"]))
        wmap = odds.get(key)
        if not wmap:
            skip += 1; continue
        try:
            h = int(r["hon"]); o = int(r["taikou"]); s = int(r["sabo"])
        except Exception:
            continue
        fh = fuku.get(key + (h,), 0.0)
        fo = fuku.get(key + (o,), 0.0)
        fs = fuku.get(key + (s,), 0.0)
        w_ho = wmap.get(frozenset({h,o}), 0.0)
        w_ha = wmap.get(frozenset({h,s}), 0.0)
        w_oa = wmap.get(frozenset({o,s}), 0.0)

        # 各パターン: (投資円, 回収円)
        def pat(bets: list[float]) -> tuple[float, float]:
            # bets = list of payouts (0 if miss). 投資 = len*UNIT
            return (len(bets) * UNIT, sum(bets))

        p = {}
        p["A1"] = pat([fh])
        p["A2"] = pat([fo])
        p["A3"] = pat([fs])
        p["A4"] = pat([fh, fo, fs])
        p["B1"] = pat([w_ho])
        p["B2"] = pat([w_ha])
        p["B3"] = pat([w_oa])
        p["B4"] = pat([w_ho, w_ha, w_oa])
        p["C1"] = pat([fh, w_ho])
        p["C2"] = pat([fh, w_ho, w_ha, w_oa])
        p["C3"] = pat([fh, fo, w_ho])
        p["C4"] = pat([fh, fo, fs, w_ho, w_ha, w_oa])
        p["C5"] = pat([fh, w_ho, w_ha])

        base = {"年": int(r["年"]), "月": int(r["月"]), "日": int(r["日"]),
                "場所": r["場所"], "R": int(r["R"]),
                "fh": fh, "fo": fo, "fs": fs,
                "w_ho": w_ho, "w_ha": w_ha, "w_oa": w_oa}
        for k, (bet, ret) in p.items():
            base[f"{k}_bet"] = bet
            base[f"{k}_ret"] = ret
            base[f"{k}_hit"] = 1 if ret > 0 else 0
        rows.append(base)
    print(f"  simulated: {len(rows):,} / skipped(no wide odds): {skip:,}")
    return pd.DataFrame(rows)


def metrics(df: pd.DataFrame, patterns: list[str]) -> pd.DataFrame:
    out = []
    for p in patterns:
        bet = df[f"{p}_bet"].sum()
        ret = df[f"{p}_ret"].sum()
        roi = (ret / bet * 100) if bet > 0 else 0.0
        hit = df[f"{p}_hit"].mean() * 100
        # 日次収支シリーズで DD / std / sharpe
        daily = df.groupby(["年","月","日"]).apply(
            lambda g: g[f"{p}_ret"].sum() - g[f"{p}_bet"].sum()
        )
        cum = daily.cumsum()
        peak = cum.cummax()
        dd = (peak - cum).max() if len(cum) else 0.0
        std = daily.std() if len(daily) > 1 else 0.0
        mean = daily.mean() if len(daily) else 0.0
        sharpe = (mean / std * np.sqrt(250)) if std > 0 else 0.0
        out.append({
            "pattern": p,
            "n_races": len(df),
            "bet": int(bet),
            "ret": int(ret),
            "ROI%": round(roi, 1),
            "hit%": round(hit, 1),
            "maxDD": int(dd),
            "dailyStd": round(std, 1),
            "sharpe": round(sharpe, 2),
        })
    return pd.DataFrame(out)


def main():
    odds = load_odds()
    fuku = load_fukusho()
    merged = load_marks()
    sim = simulate(merged, odds, fuku)
    if sim.empty:
        print("ERROR: no simulated rows")
        return

    patterns = ["A1","A2","A3","A4","B1","B2","B3","B4","C1","C2","C3","C4","C5"]

    splits = {
        "ALL_全会場": sim,
        "ALL_4会場": sim[sim["場所"].isin(FOUR_PLACES)],
        "2024_全会場": sim[sim["年"] == 2024],
        "2024_4会場": sim[(sim["年"] == 2024) & (sim["場所"].isin(FOUR_PLACES))],
        "2025_全会場": sim[sim["年"] == 2025],
        "2025_4会場": sim[(sim["年"] == 2025) & (sim["場所"].isin(FOUR_PLACES))],
    }

    all_rows = []
    for name, d in splits.items():
        if d.empty:
            continue
        mt = metrics(d, patterns)
        mt["split"] = name
        all_rows.append(mt)
    summary = pd.concat(all_rows, ignore_index=True)
    summary = summary[["split","pattern","n_races","bet","ret","ROI%","hit%","maxDD","dailyStd","sharpe"]]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {OUT_CSV}")

    # 相関: A1 hit vs B1 hit
    c_a1_b1 = sim["A1_hit"].corr(sim["B1_hit"])
    c_a1_b4 = sim["A1_hit"].corr(sim["B4_hit"])

    # Markdown
    lines = ["# 複勝+ワイド セット購入 検証結果", ""]
    lines.append(f"- 対象: {len(sim):,} レース (odds_test×marks 結合)")
    lines.append(f"- 単位: 1点 {UNIT}円")
    lines.append(f"- 相関 A1(複◎)×B1(ワ◎-◯): {c_a1_b1:.3f}")
    lines.append(f"- 相関 A1(複◎)×B4(ワBOX3): {c_a1_b4:.3f}")
    lines.append("")
    lines.append("## パターン凡例")
    lines.append("- A1 複◎ / A2 複◯ / A3 複▲ / A4 複◎◯▲(3点)")
    lines.append("- B1 ワ◎-◯ / B2 ワ◎-▲ / B3 ワ◯-▲ / B4 ワBOX3(3点)")
    lines.append("- C1 複◎+ワ◎-◯ / C2 複◎+ワBOX3 / C3 複◎◯+ワ◎-◯ / C4 複◎◯▲+ワBOX3 / C5 複◎+ワ◎-◯+ワ◎-▲")
    lines.append("")
    for name, d in splits.items():
        if d.empty:
            continue
        lines.append(f"## {name} (n={len(d):,})")
        lines.append("")
        lines.append("| pattern | ROI% | hit% | maxDD | std | sharpe |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        mt = metrics(d, patterns)
        for _, r in mt.iterrows():
            lines.append(f"| {r['pattern']} | {r['ROI%']} | {r['hit%']} | {r['maxDD']} | {r['dailyStd']} | {r['sharpe']} |")
        lines.append("")

    # 推奨判定 (2025_4会場 ベース)
    key = "2025_4会場"
    if key in splits and not splits[key].empty:
        mt = metrics(splits[key], patterns).set_index("pattern")
        a1_roi = mt.loc["A1","ROI%"]
        a1_dd = mt.loc["A1","maxDD"]
        a1_sh = mt.loc["A1","sharpe"]
        cand = mt[(mt["ROI%"] >= a1_roi) & (mt["maxDD"] <= a1_dd)]
        lines.append("## 推奨判定 (基準: 2025_4会場)")
        lines.append("")
        lines.append(f"- A1 (現行主軸): ROI {a1_roi}% / DD {a1_dd} / sharpe {a1_sh}")
        if not cand.empty:
            best = cand.sort_values(["sharpe","ROI%"], ascending=False).iloc[0]
            lines.append(f"- **推奨**: {best.name} — ROI {best['ROI%']}% / DD {int(best['maxDD'])} / sharpe {best['sharpe']}")
            lines.append(f"- A1 に対する優位性: ROI +{best['ROI%']-a1_roi:.1f}pt / DD {int(best['maxDD']-a1_dd):+d} / sharpe {best['sharpe']-a1_sh:+.2f}")
        else:
            lines.append("- **採否: 不採用** — A1 を上回る(ROI≥ かつ DD≤)パターンなし")
        lines.append("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {OUT_MD}")

    # コンソール要約
    print("\n=== 2025_4会場 (OOS主判定) ===")
    if "2025_4会場" in splits and not splits["2025_4会場"].empty:
        print(metrics(splits["2025_4会場"], patterns).to_string(index=False))


if __name__ == "__main__":
    main()
