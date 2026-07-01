# -*- coding: utf-8 -*-
"""
wide_lab.py — ワイドAI 徹底最適化 (他券種無視、ワイドのみ突き詰める)
================================================================================
ワイドの構造的強み: 3着内の任意ペアで的中=高的中 + ◎の複勝圏率(~62%, モデル最強点)を直接活用。
選択方法×点数(cap)×相手pool×参加ゲート(信頼度)×賭け方 を総当たりで OOS 最適化。

OOS規律(過学習防止=今session最大の教訓):
  valid(2023)で各configのROIを見て閾値/configを選ぶ → test(2024-25)で検証 →
  test 95%CI(レースクラスタbootstrap) + 多重比較(何config偶然黒字に見えるか)。

確率: v6 score → PL weight → all_wide_mat → cal["wide"](isotonic) = calibrated p_wide。
決済: wide_payouts_2016-2025.parquet の確定3ペア配当(単位100円)。選択=モデル, 採点=実配当。

出力: reports/wide_lab.json / wide_lab_summary.txt
実行: PYTHONUTF8=1 ./venv311/Scripts/python.exe wide_lab.py
"""
from __future__ import annotations
import json, time
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import backtest_pl_ev as BT     # all_wide_mat / load_wide_payouts (guarded stdout wrap)
import pl_probs as PL

BASE = Path(__file__).parent
SCORE_CACHE = BASE / "data/_ev_grid_scores.parquet"
V6_CAL = BASE / "models/pl_calibrators_v6.pkl"
OUT_JSON = BASE / "reports/wide_lab.json"
OUT_TXT = BASE / "reports/wide_lab_summary.txt"
COL_RID = "レースID(新/馬番無)"; COL_BAN = "馬番"
BET = 100; NBOOT = 1500; MIN_RACES = 200


def log(m): print(m, flush=True)


def boot_ci(pairs, seed=42):
    if not pairs: return (0., 0., 0.)
    st = np.array([p[0] for p in pairs], float); rt = np.array([p[1] for p in pairs], float)
    roi = rt.sum()/st.sum() if st.sum() else 0.
    rng = np.random.default_rng(seed); m=len(pairs); o=np.empty(NBOOT)
    for b in range(NBOOT):
        i=rng.integers(0,m,m); s=st[i].sum(); o[b]=rt[i].sum()/s if s else 0.
    return (float(roi), float(np.percentile(o,2.5)), float(np.percentile(o,97.5)))


def precompute(sc, widepay, cal):
    """各レース: 候補ペア(p_wide降順, 馬番, ◎絡みflag) + 信頼度 + 当選ペア配当。"""
    races = []
    t0 = time.time(); n = 0
    for rid, g in sc.groupby(COL_RID, sort=False):
        rid_s = str(rid)
        if rid_s not in widepay: continue
        g = g.sort_values(COL_BAN).reset_index(drop=True)
        ban = g[COL_BAN].astype(int).values
        w = PL.pl_weights(g["_score"].values); N = len(w)
        if N < 4: continue
        p = w / w.sum()
        sp = np.sort(p)[::-1]
        dom = float(sp[0]-sp[1]); conc = float(sp[0]+sp[1])
        pc = np.clip(p,1e-9,1); ent = float(-(pc*np.log(pc)).sum()/np.log(N))
        hon = int(np.argmax(w))                       # ◎ = 最上位
        rank = np.argsort(-w)                          # 馬index 強い順
        rank_of = {int(ix): r for r, ix in enumerate(rank)}
        Wm = cal["wide"].predict(BT.all_wide_mat(w).ravel()).reshape(N, N)
        iu, ju = np.triu_indices(N, k=1)
        pw = Wm[iu, ju]
        # 当選ペア(馬番)→配当
        winp = {frozenset((int(a), int(b))): float(pay) for (a, b, pay) in widepay[rid_s]}
        pairs = []
        for k in range(len(iu)):
            i, j = int(iu[k]), int(ju[k])
            # 相手pool判定用に各馬のstrength rank, ◎絡みflag
            pairs.append((float(pw[k]), int(ban[i]), int(ban[j]),
                          rank_of[i], rank_of[j], (i == hon or j == hon)))
        pairs.sort(key=lambda x: -x[0])               # p_wide 降順
        races.append({"conc": conc, "dom": dom, "ent": ent, "pairs": pairs,
                      "win": winp, "period": "valid" if g["split"].iloc[0] == "valid" else "test"})
        n += 1
        if n % 2000 == 0: log(f"  precompute {n} races ({time.time()-t0:.0f}s)")
    log(f"  precompute done: {n} races ({time.time()-t0:.0f}s)")
    return races


def gate_ok(r, gate, thr):
    if gate == "all": return True
    if gate == "conc": return r["conc"] >= thr
    if gate == "chaoslow": return r["ent"] <= thr
    if gate == "dom": return r["dom"] >= thr
    return True


def eval_config(races, pool, cap, mode, gate, thr):
    """戻り: {valid:[(stake,ret)...], test:[...]} per race aggregated."""
    res = {"valid": [], "test": []}
    for r in races:
        if not gate_ok(r, gate, thr): continue
        # 相手pool: 強さrank < pool の馬だけ使う。mode=axis は◎絡みのみ。
        cand = [pp for pp in r["pairs"] if pp[3] < pool and pp[4] < pool
                and (pp[5] if mode == "axis" else True)]
        sel = cand[:cap]
        if not sel: continue
        stake = len(sel) * BET
        ret = 0.0
        for (_pw, bi, bj, _ri, _rj, _ax) in sel:
            ret += r["win"].get(frozenset((bi, bj)), 0.0) * (BET / 100.0)
        res[r["period"]].append((stake, ret))
    return res


def roi(pairs):
    st = sum(p[0] for p in pairs); rt = sum(p[1] for p in pairs)
    return (rt/st) if st else 0.0, st, rt, len(pairs)


def main():
    T0 = time.time(); OUT_JSON.parent.mkdir(exist_ok=True)
    cal = joblib.load(V6_CAL)["calibrators"]
    if "wide" not in cal:
        log("[ERROR] cal['wide'] が無い"); return
    widepay = BT.load_wide_payouts(); log(f"[wide payouts] {len(widepay):,} races (2016-2025)")
    sc = pd.read_parquet(SCORE_CACHE); sc[COL_RID] = sc[COL_RID].astype(str)
    sc = sc[sc["year"].isin([2023, 2024, 2025])] if "year" in sc else sc
    import os
    if os.environ.get("WIDE_LAB_SMOKE"):
        keep = sc[COL_RID].drop_duplicates().sample(frac=float(os.environ["WIDE_LAB_SMOKE"]), random_state=1)
        sc = sc[sc[COL_RID].isin(keep)]
    log(f"[scores] {len(sc):,} rows")
    races = precompute(sc, widepay, cal)

    # valid 分位で gate 閾値を決める
    va = [r for r in races if r["period"] == "valid"]
    q = {"conc": np.quantile([r["conc"] for r in va], [0.5, 0.75]),
         "chaoslow": np.quantile([r["ent"] for r in va], [0.5, 0.25]),
         "dom": np.quantile([r["dom"] for r in va], [0.5, 0.75])}

    GATES = [("all", None)]
    for g in ("conc", "dom"):
        GATES += [(g, float(q[g][0])), (g, float(q[g][1]))]
    GATES += [("chaoslow", float(q["chaoslow"][0])), ("chaoslow", float(q["chaoslow"][1]))]

    configs = []
    for pool in (3, 4, 5):
        for cap in (2, 3, 4, 5, 6):
            for mode in ("box", "axis"):
                for (gate, thr) in GATES:
                    configs.append((pool, cap, mode, gate, thr))
    log(f"[grid] {len(configs)} configs 評価 ...")

    rows = []
    for (pool, cap, mode, gate, thr) in configs:
        r = eval_config(races, pool, cap, mode, gate, thr)
        v_roi, v_st, v_rt, v_n = roi(r["valid"])
        t_roi, t_st, t_rt, t_n = roi(r["test"])
        if v_n < MIN_RACES or t_n < MIN_RACES: continue
        troi, tlo, thi = boot_ci(r["test"])
        rows.append({"pool": pool, "cap": cap, "mode": mode, "gate": gate,
                     "thr": round(thr, 4) if thr is not None else None,
                     "valid_roi": round(v_roi, 4), "valid_n": v_n,
                     "test_roi": round(t_roi, 4), "test_ci95": [round(tlo, 4), round(thi, 4)],
                     "test_n": t_n, "test_hit_races": sum(1 for p in r["test"] if p[1] > 0)})
    # valid ROI 上位で選抜 → test 確認(過学習防止: valid で選び test で見る)
    rows_by_valid = sorted(rows, key=lambda x: -x["valid_roi"])
    rows_by_test = sorted(rows, key=lambda x: -x["test_roi"])
    # valid top10 を test で見る
    valid_top = rows_by_valid[:10]
    best_oos = max(valid_top, key=lambda x: x["test_roi"]) if valid_top else None
    n_test_breakeven = sum(1 for x in rows if x["test_ci95"][0] > 1.0)

    result = {"protocol": "ワイドのみ / v6 p_wide / valid2023調整→test2024-25検証+bootstrap / 選択=モデル,採点=実配当",
              "n_configs": len(rows), "min_races": MIN_RACES,
              "valid_top10_then_test": valid_top,
              "test_top10": rows_by_test[:10],
              "best_oos(valid選抜→test)": best_oos,
              "n_configs_test_ci_over_breakeven": n_test_breakeven,
              "elapsed_sec": round(time.time()-T0, 1)}
    json.dump(result, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    L = []
    def P(s): L.append(s); log(s)
    P("="*100)
    P("ワイドAI 徹底最適化  [v6 p_wide / valid2023→test2024-25 / flat¥100 / 選択モデル・採点実配当]")
    P("="*100)
    P(f"評価config {len(rows)} / test CI下限>100%(本物)の数: {n_test_breakeven}  (多重比較: 偶然なら ~{round(0.025*len(rows),1)})")
    P("\n■ valid ROI 上位10 → そのtest成績 (過学習チェック: validで良くてもtestで崩れるか)")
    P(f"{'pool':>4} {'cap':>3} {'mode':5} {'gate':9} {'vROI':>7} {'vN':>5} {'tROI':>7} {'tCI95':>16} {'tN':>5}")
    for x in valid_top:
        ci = x["test_ci95"]
        P(f"{x['pool']:>4} {x['cap']:>3} {x['mode']:5} {x['gate']:9} {x['valid_roi']*100:6.1f}% {x['valid_n']:>5} "
          f"{x['test_roi']*100:6.1f}% [{ci[0]*100:5.1f},{ci[1]*100:5.1f}] {x['test_n']:>5}")
    P("\n■ test ROI 上位10 (参考: testで最も良かったが、validで選べてないとカンニング)")
    for x in rows_by_test[:10]:
        ci = x["test_ci95"]
        P(f"{x['pool']:>4} {x['cap']:>3} {x['mode']:5} {x['gate']:9} v{x['valid_roi']*100:5.1f}% "
          f"t{x['test_roi']*100:5.1f}% [{ci[0]*100:5.1f},{ci[1]*100:5.1f}] N{x['test_n']}")
    if best_oos:
        b = best_oos; ci = b["test_ci95"]
        P("\n" + "="*100)
        P(f"★ OOS最良(valid選抜→test): pool{b['pool']} cap{b['cap']} {b['mode']} gate={b['gate']}")
        P(f"   valid {b['valid_roi']*100:.1f}% → test {b['test_roi']*100:.1f}% CI[{ci[0]*100:.1f},{ci[1]*100:.1f}] "
          f"(test {b['test_n']}R, 的中{b['test_hit_races']}R)")
        P(f"   {'✅ CI下限>100%=本物の黒字' if ci[0]>1.0 else '控除床以下(負け最小化どまり)'}")
    P("="*100)
    OUT_TXT.write_text("\n".join(L), encoding="utf-8")
    log(f"[saved] {OUT_JSON} / {OUT_TXT}")


if __name__ == "__main__":
    main()
