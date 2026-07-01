# -*- coding: utf-8 -*-
"""
backtest_frontier.py — 当てに行く: モデル確信 × 市場の早い金(9時前オッズ短縮) × 妙味
================================================================================
仮説: モデルが好む馬を、さらに「早い金も同方向に動いた(9時前オッズが短縮=買われた)」で
      二重に絞れば、モデル単独より的中/ROIが跳ねるかもしれない。odds_ou は予測ゲートを
      唯一通った信号(=市場マイクロストラクチャ)。これを SELECTION で ROI 検証する(初)。

信号: traj_logchg = log(9時直前オッズ) - log(売出オッズ)。負=短縮=賢い金が入った。
      (data/odds_traj_feats.parquet, 9時前のみ leak-safe, test カバレッジ46%)

戦略(flat ¥100, 単勝/複勝/ワイド/馬連):
  *_top1        : モデル最上位を素で買う(baseline)
  *_short       : 上位 かつ 短縮(traj_logchg<thr)
  *_long        : 上位 かつ 伸び(対照, 効くなら短縮の逆が出るはず)
  *_overlay     : 上位 かつ 妙味(p×実オッズ≥1)  ※単複のみ(全頭オッズ有)
  *_ov_short    : 上位 かつ 妙味 かつ 短縮   ← フロンティア本命

評価: valid(2023) と test(2024-25) を別集計、両方 ROI>100% かつ test 95%CI下限>100% を「本物」。
出力: reports/frontier_result.json / frontier_summary.txt
実行: PYTHONUTF8=1 ./venv311/Scripts/python.exe backtest_frontier.py
"""
from __future__ import annotations
import io, json, sys, warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import glob
import backtest_pl_ev as BT
import pl_probs as PL


def load_odds():
    """全頭確定オッズ {rid16:{ban:(tan,flo,fhi)}}. (backtest_ev_grid と同一; import衝突回避でインライン)"""
    out = {}
    for f in sorted(glob.glob(str(BASE / "data/odds/odds_*.csv"))):
        df = pd.read_csv(f, encoding="cp932", low_memory=False)
        rid = df["レースID(新)"].astype(str).str[:16]
        ban = pd.to_numeric(df["馬番"], errors="coerce")
        tan = pd.to_numeric(df["単勝オッズ"], errors="coerce")
        flo = pd.to_numeric(df["複勝オッズ下限"], errors="coerce")
        fhi = pd.to_numeric(df["複勝オッズ上限"], errors="coerce")
        for r, b, t, lo, hi in zip(rid.values, ban.values, tan.values, flo.values, fhi.values):
            if b != b:
                continue
            out.setdefault(r, {})[int(b)] = (float(t) if t == t else np.nan,
                                             float(lo) if lo == lo else np.nan,
                                             float(hi) if hi == hi else np.nan)
    return out

BASE = Path(__file__).parent
SCORE_CACHE = BASE / "data/_ev_grid_scores.parquet"
V6_CAL = BASE / "models/pl_calibrators_v6.pkl"
TRAJ_PQ = BASE / "data/odds_traj_feats.parquet"
OU_PQ = BASE / "data/odds_ou_feats.parquet"
OUT_JSON = BASE / "reports/frontier_result.json"
OUT_TXT = BASE / "reports/frontier_summary.txt"

COL_RID = "レースID(新/馬番無)"; COL_BAN = "馬番"; COL_JYUN = "着順"
BET = 100
SHORT = -0.05        # 短縮閾値(log)。9時前オッズが約5%以上絞られた
STRONG = -0.20
NBOOT = 2000
MIN_RACES = 50


def log(m): print(m, flush=True)
def _f(x):
    try:
        v = float(x); return v if v == v else 0.0
    except Exception:
        return 0.0


def boot_ci(pairs, seed=42):
    if not pairs: return (0.0, 0.0, 0.0, 0)
    st = np.array([p[0] for p in pairs], float); rt = np.array([p[1] for p in pairs], float)
    roi = rt.sum() / st.sum() if st.sum() > 0 else 0.0
    rng = np.random.default_rng(seed); m = len(pairs); out = np.empty(NBOOT)
    for b in range(NBOOT):
        idx = rng.integers(0, m, m); s = st[idx].sum()
        out[b] = rt[idx].sum() / s if s > 0 else 0.0
    return (float(roi), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), m)


def load_scores():
    sc = pd.read_parquet(SCORE_CACHE)
    sc["rid16"] = sc[COL_RID].astype(str).str.replace(r"\D", "", regex=True).str[:16]
    sc["ban"] = pd.to_numeric(sc[COL_BAN], errors="coerce").fillna(0).astype(int)
    tr = pd.read_parquet(TRAJ_PQ); tr["rid16"] = tr["rid16"].astype(str)
    tr["ban"] = pd.to_numeric(tr["ban"], errors="coerce").fillna(0).astype(int)
    sc = sc.merge(tr[["rid16", "ban", "traj_has", "traj_logchg"]], on=["rid16", "ban"], how="left")
    return sc


def run_period(sc, payouts, widepays, oddsmap, cal):
    """1期間。戻り: {strat: [(stake,ret), ...] per race}."""
    rec = {}
    def add(strat, stake, ret):
        if stake <= 0: return
        rec.setdefault(strat, []).append((stake, ret))

    n_race = 0
    for rid, g in sc.groupby(COL_RID, sort=False):
        if len(g) < 5: continue
        rid_s = str(int(rid)) if isinstance(rid, (int, np.integer)) else str(rid)
        if rid_s not in payouts: continue
        po = payouts[rid_s]
        g = g.sort_values(COL_BAN).reset_index(drop=True)
        ban = g[COL_BAN].astype(int).values
        b2i = {int(b): i for i, b in enumerate(ban)}
        wi = b2i.get(po["win"]); pi_ = b2i.get(po["plc"]); si = b2i.get(po["sho"])
        if wi is None or pi_ is None or si is None: continue
        n = len(g); n_race += 1
        w = PL.pl_weights(g["_score"].values)
        p_win = cal["tansho"].predict(w / w.sum())
        p_sho = cal["fukusho"].predict(BT.all_fukusho_vec_fast(w))
        mv = pd.to_numeric(g["traj_logchg"], errors="coerce").values   # 負=短縮
        has = np.isfinite(mv)
        od = oddsmap.get(rid_s, {})
        tan = np.array([od.get(int(b), (np.nan, np.nan, np.nan))[0] for b in ban])
        flo = np.array([od.get(int(b), (np.nan, np.nan, np.nan))[1] for b in ban])
        fhi = np.array([od.get(int(b), (np.nan, np.nan, np.nan))[2] for b in ban])
        midf = (flo + fhi) / 2.0

        # ---------- 単勝 ----------
        t1 = int(np.argmax(p_win))
        def win_ret(i): return _f(po["tansho"]) if i == wi else 0.0
        add("win_top1", BET, win_ret(t1))
        if has[t1] and mv[t1] < SHORT: add("win_top1_short", BET, win_ret(t1))
        if has[t1] and mv[t1] > -SHORT: add("win_top1_long", BET, win_ret(t1))
        if has[t1] and mv[t1] < STRONG: add("win_top1_strong", BET, win_ret(t1))
        ov1 = np.isfinite(tan[t1]) and (p_win[t1] * tan[t1] >= 1.0)
        if ov1: add("win_top1_overlay", BET, win_ret(t1))
        if ov1 and has[t1] and mv[t1] < SHORT: add("win_top1_ov_short", BET, win_ret(t1))
        # value全頭: 妙味かつ短縮を全部
        ev = p_win * np.where(np.isfinite(tan), tan, 0.0)
        sel = np.where((ev >= 1.0) & has & (mv < SHORT))[0]
        if len(sel):
            stake = len(sel) * BET; ret = sum(win_ret(i) for i in sel)
            add("win_value_short", stake, ret)
        sel2 = np.where(ev >= 1.0)[0]
        if len(sel2):
            add("win_value_all", len(sel2) * BET, sum(win_ret(i) for i in sel2))

        # ---------- 複勝 ----------
        s1 = int(np.argmax(p_sho))
        def sho_ret(i):
            if i == wi: return _f(po["fuku_win"])
            if i == pi_: return _f(po["fuku_plc"])
            if i == si: return _f(po["fuku_sho"])
            return 0.0
        add("plc_top1", BET, sho_ret(s1))
        if has[s1] and mv[s1] < SHORT: add("plc_top1_short", BET, sho_ret(s1))
        ovp = np.isfinite(midf[s1]) and (p_sho[s1] * midf[s1] >= 1.0)
        if ovp: add("plc_top1_overlay", BET, sho_ret(s1))
        if ovp and has[s1] and mv[s1] < SHORT: add("plc_top1_ov_short", BET, sho_ret(s1))

        # ---------- ワイド(モデル最良ペア) ----------
        if "wide" in cal:
            Wm = BT.all_wide_mat(w)
            iu, ju = np.triu_indices(n, k=1)
            pw = cal["wide"].predict(Wm[iu, ju])
            kbest = int(np.argmax(pw)); a, b = int(iu[kbest]), int(ju[kbest])
            wpairs = widepays.get(rid_s, []); paymap = {}
            for (bi, bj, pv) in wpairs:
                x, y = b2i.get(bi), b2i.get(bj)
                if x is not None and y is not None: paymap[frozenset((x, y))] = pv
            wret = paymap.get(frozenset((a, b)), 0.0)
            add("wide_best", BET, wret)
            if has[a] and has[b] and mv[a] < SHORT and mv[b] < SHORT:
                add("wide_best_short", BET, wret)

        # ---------- 馬連(モデル最良ペア) ----------
        Um = BT.all_umaren_mat(w)
        iu, ju = np.triu_indices(n, k=1)
        pu = cal["umaren"].predict(Um[iu, ju])
        kbest = int(np.argmax(pu)); a, b = int(iu[kbest]), int(ju[kbest])
        uret = _f(po["umaren"]) if {a, b} == {wi, pi_} else 0.0
        add("umaren_best", BET, uret)
        if has[a] and has[b] and mv[a] < SHORT and mv[b] < SHORT:
            add("umaren_best_short", BET, uret)

    log(f"  races={n_race}")
    return rec


def main():
    OUT_JSON.parent.mkdir(exist_ok=True)
    payouts = BT.load_payouts(); log(f"[payouts] {len(payouts):,}")
    try: widepays = BT.load_wide_payouts()
    except Exception: widepays = {}
    oddsmap = load_odds(); log(f"[odds] {len(oddsmap):,}")
    cal = joblib.load(V6_CAL)["calibrators"]
    sc = load_scores(); log(f"[scores] {len(sc):,} rows")

    periods = {}
    for nm, mask in [("valid", sc["split"] == "valid"), ("test", sc["year"].isin([2024, 2025]))]:
        log(f"=== {nm} ===")
        periods[nm] = run_period(sc[mask], payouts, widepays, oddsmap, cal)

    strats = sorted(set(periods["valid"]) | set(periods["test"]))
    rows = []
    for st in strats:
        vroi, vlo, vhi, vn = boot_ci(periods["valid"].get(st, []))
        troi, tlo, thi, tn = boot_ci(periods["test"].get(st, []))
        real = bool(vroi >= 1.0 and troi >= 1.0 and tlo > 1.0 and vn >= MIN_RACES and tn >= MIN_RACES)
        rows.append({"strat": st, "valid_roi": round(vroi, 4), "valid_n": vn,
                     "test_roi": round(troi, 4), "test_ci95": [round(tlo, 4), round(thi, 4)],
                     "test_n": tn, "real": real})
    rows.sort(key=lambda r: r["test_roi"], reverse=True)
    result = {"protocol": "v6 / valid2023・test2024-25 / flat¥100 / market-confirmation=9時前オッズ短縮(traj_logchg<-0.05)",
              "short_threshold": SHORT, "rows": rows,
              "n_real": sum(r["real"] for r in rows)}
    json.dump(result, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    lines = []
    def P(s): lines.append(s); log(s)
    P("=" * 92)
    P("フロンティア: モデル確信 × 早い金(9時前オッズ短縮) × 妙味  [flat¥100, valid=2023/test=2024-25]")
    P("=" * 92)
    P(f"{'戦略':22s} {'vROI':>8s} {'vN':>6s} | {'tROI':>8s} {'tCI95':>18s} {'tN':>6s}  本物")
    for r in rows:
        ci = r["test_ci95"]
        P(f"{r['strat']:22s} {r['valid_roi']*100:7.1f}% {r['valid_n']:>6d} | "
          f"{r['test_roi']*100:7.1f}% [{ci[0]*100:6.1f}%,{ci[1]*100:6.1f}%] {r['test_n']:>6d}  {'★YES' if r['real'] else ''}")
    P("=" * 92)
    P(f"valid∩test 黒字 かつ test 95%CI下限>100% の『本物』戦略: {result['n_real']}")
    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
    log(f"[saved] {OUT_JSON}\n[saved] {OUT_TXT}")


if __name__ == "__main__":
    main()
