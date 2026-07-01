# -*- coding: utf-8 -*-
"""
participation_analyzer.py — 参加レース選別(見送り)の OOS 実証
================================================================================
目的(③ 自分用ツール=負け最小化): 「どのレースを避ければ負けが減るか」を valid(2023)で
閾値決定→test(2024-25)で検証。儲け(>100%)は期待しない。最も負けない参加点を出す。

レース信頼度(v6 校正勝率から):
  dom1 = p1 - p2          ◎独走度(大=堅い)
  conc = p1 + p2          上位2頭集中度
  entropy = -Σp log p / log n   混戦度(大=荒れ)
  agree = corr(ai_rank, odds_rank)  市場一致度(要オッズ)

代表戦略(flat¥100): ◎複勝 / ◎単勝。各信頼度の四分位ごとに ROI/的中/レース数を出し、
valid で決めた「上位四分位だけ参加」が test でも負けを減らすかを bootstrap CI 付きで検証。
出力: reports/participation_result.json / participation_summary.txt
実行: PYTHONUTF8=1 ./venv311/Scripts/python.exe participation_analyzer.py
"""
from __future__ import annotations
import io, json, sys, warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import backtest_pl_ev as BT   # guarded stdout wrap (安全)
import pl_probs as PL

BASE = Path(__file__).parent
SCORE_CACHE = BASE / "data/_ev_grid_scores.parquet"
V6_CAL = BASE / "models/pl_calibrators_v6.pkl"
OUT_JSON = BASE / "reports/participation_result.json"
OUT_TXT = BASE / "reports/participation_summary.txt"
COL_RID = "レースID(新/馬番無)"; COL_BAN = "馬番"; COL_JYUN = "着順"
BET = 100; NBOOT = 2000


def log(m): print(m, flush=True)
def _f(x):
    try:
        v = float(x); return v if v == v else 0.0
    except Exception:
        return 0.0


def boot_ci(pairs, seed=42):
    if not pairs: return (0., 0., 0.)
    st = np.array([p[0] for p in pairs], float); rt = np.array([p[1] for p in pairs], float)
    roi = rt.sum()/st.sum() if st.sum() else 0.
    rng = np.random.default_rng(seed); m=len(pairs); o=np.empty(NBOOT)
    for b in range(NBOOT):
        i=rng.integers(0,m,m); s=st[i].sum(); o[b]=rt[i].sum()/s if s else 0.
    return (float(roi), float(np.percentile(o,2.5)), float(np.percentile(o,97.5)))


def per_race(sc, payouts, cal):
    """各レース: 信頼度メトリクス + ◎複勝/◎単勝の(stake,ret)。"""
    rows = []
    for rid, g in sc.groupby(COL_RID, sort=False):
        if len(g) < 5: continue
        rid_s = str(rid)
        if rid_s not in payouts: continue
        po = payouts[rid_s]
        g = g.sort_values(COL_BAN).reset_index(drop=True)
        ban = g[COL_BAN].astype(int).values
        b2i = {int(b): i for i, b in enumerate(ban)}
        wi = b2i.get(po["win"]); pi_ = b2i.get(po["plc"]); si = b2i.get(po["sho"])
        if wi is None or pi_ is None or si is None: continue
        w = PL.pl_weights(g["_score"].values); n = len(w)
        p = cal["tansho"].predict(w / w.sum())
        ps = cal["fukusho"].predict(BT.all_fukusho_vec_fast(w))
        sp = np.sort(p)[::-1]
        dom1 = float(sp[0] - sp[1]); conc = float(sp[0] + sp[1])
        pp = np.clip(p, 1e-9, 1); ent = float(-(pp*np.log(pp)).sum()/np.log(n))
        t1 = int(np.argmax(p))                 # ◎ = p_win 最上位
        s1 = int(np.argmax(ps))
        tan_ret = _f(po["tansho"]) if t1 == wi else 0.0
        fuku_ret = _f(po["fuku_win"]) if s1 == wi else (_f(po["fuku_plc"]) if s1 == pi_ else (_f(po["fuku_sho"]) if s1 == si else 0.0))
        rows.append({"dom1": dom1, "conc": conc, "ent": ent,
                     "tan": tan_ret, "fuku": fuku_ret, "year": int(str(rid)[:4])})
    return pd.DataFrame(rows)


def roi_of(df, col):
    st = len(df) * BET
    return (df[col].sum() / st) if st else 0.0


def main():
    OUT_JSON.parent.mkdir(exist_ok=True)
    payouts = BT.load_payouts(); log(f"[payouts] {len(payouts):,}")
    cal = joblib.load(V6_CAL)["calibrators"]
    sc = pd.read_parquet(SCORE_CACHE); sc[COL_RID] = sc[COL_RID].astype(str)
    sc["year"] = sc[COL_RID].str[:4].astype(int)
    va = per_race(sc[sc["split"] == "valid"], payouts, cal)
    te = per_race(sc[sc["year"].isin([2024, 2025])], payouts, cal)
    log(f"[races] valid={len(va):,} test={len(te):,}")

    METRICS = [("dom1", "◎独走度", True), ("conc", "上位2集中", True), ("ent", "混戦度(低い=堅い)", False)]
    BETS = [("fuku", "◎複勝"), ("tan", "◎単勝")]
    result = {"protocol": "v6 / valid2023で閾値→test2024-25検証 / flat¥100 / 信頼度上位で参加",
              "baseline": {}, "gated": {}, "n_valid": len(va), "n_test": len(te)}

    for bcol, bname in BETS:
        result["baseline"][bname] = {"valid_roi": round(roi_of(va, bcol), 4),
                                     "test_roi": round(roi_of(te, bcol), 4), "test_n": len(te)}

    lines = []
    def P(s): lines.append(s); log(s)
    P("="*86)
    P("参加レース選別(見送り)の OOS 実証  [v6 / valid=2023→test=2024-25 / flat¥100]")
    P("="*86)
    P("\n■ 全レース参加(baseline)")
    for bcol, bname in BETS:
        b = result["baseline"][bname]
        P(f"  {bname}: valid {b['valid_roi']*100:.1f}% / test {b['test_roi']*100:.1f}% (n={b['test_n']:,})")

    P("\n■ 信頼度メトリクス四分位ごとの test ROI（堅い側で負けが減るか）")
    for mcol, mname, high_good in METRICS:
        qs = va[mcol].quantile([.25, .5, .75]).values
        P(f"\n  [{mname}] valid四分位境界={[round(x,3) for x in qs]}")
        for bcol, bname in BETS:
            def q_roi(df):
                edges = [-1e9, qs[0], qs[1], qs[2], 1e9]
                out = []
                for k in range(4):
                    sub = df[(df[mcol] > edges[k]) & (df[mcol] <= edges[k+1])]
                    out.append(roi_of(sub, bcol) if len(sub) else 0.0)
                return out
            tq = q_roi(te)
            P(f"    {bname} test四分位ROI(低→高): " + " ".join(f"{x*100:5.1f}%" for x in tq))

    # 参加ゲート: valid で各メトリクスの「最良四分位」を選び、test で検証(bootstrap)
    P("\n■ valid で選んだ『参加ゲート』を test 検証(bootstrap CI)")
    best = None
    for mcol, mname, high_good in METRICS:
        thr = va[mcol].quantile(0.75 if high_good else 0.25)
        for bcol, bname in BETS:
            if high_good:
                va_sub = va[va[mcol] >= thr]; te_sub = te[te[mcol] >= thr]
                gate_desc = f"{mname}≥{thr:.3f}(上位25%)"
            else:
                va_sub = va[va[mcol] <= thr]; te_sub = te[te[mcol] <= thr]
                gate_desc = f"{mname}≤{thr:.3f}(堅い25%)"
            v_roi = roi_of(va_sub, bcol); t_pairs = [(BET, r) for r in te_sub[bcol].values]
            roi, lo, hi = boot_ci(t_pairs)
            base_t = result["baseline"][bname]["test_roi"]
            rec = {"gate": gate_desc, "bet": bname, "valid_roi": round(v_roi, 4),
                   "test_roi": round(roi, 4), "test_ci95": [round(lo, 4), round(hi, 4)],
                   "test_n": len(te_sub), "base_test_roi": base_t,
                   "improve_pt": round((roi - base_t) * 100, 2)}
            result["gated"][f"{mname}|{bname}"] = rec
            P(f"  {gate_desc:24s} {bname}: test {roi*100:5.1f}% [{lo*100:.1f},{hi*100:.1f}] "
              f"n={len(te_sub):,}  vs全{base_t*100:.1f}% ({rec['improve_pt']:+.1f}pt)")
            if best is None or roi > best["test_roi"]:
                best = rec
    result["best_gate"] = best
    P("\n" + "="*86)
    P(f"最良ゲート: {best['gate']} × {best['bet']} → test {best['test_roi']*100:.1f}% "
      f"(全{best['base_test_roi']*100:.1f}%比 {best['improve_pt']:+.1f}pt, CI下限{best['test_ci95'][0]*100:.1f}%)")
    P("※プラス転換(>100%)でなくても、負けを減らす/分散を抑える運用点として使う。CI下限>100%なら本物。")
    P("="*86)
    json.dump(result, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
    log(f"[saved] {OUT_JSON}\n[saved] {OUT_TXT}")


if __name__ == "__main__":
    main()
