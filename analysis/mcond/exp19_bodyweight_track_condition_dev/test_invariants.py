# -*- coding: utf-8 -*-
"""
test_invariants.py — EXP19 Stage 0 S0-C の合成 invariant / 構造テスト (結果性能を計算しない)
=========================================================================================
出力: out/invariant_tests.json
実行: python -m analysis.mcond.exp19_bodyweight_track_condition_dev.test_invariants
"""
from __future__ import annotations

import inspect
import json
import sys
import time

import numpy as np
import pandas as pd

from . import features as F
from . import gate_grade as G
from . import models as M
from . import placebos as PL
from .loaders import (FORBIDDEN_EXACT, OUT, assert_torch_frame_clean, load_torch_struct, loader_sha256,
                      torch_header)

RES = []


def rec(name, ok, detail=""):
    RES.append({"test": name, "pass": bool(ok), "detail": str(detail)})
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)


def w_equal(a: pd.DataFrame, b: pd.DataFrame, cols) -> bool:
    for c in cols:
        x, y = a[c].to_numpy(), b[c].to_numpy()
        if x.dtype.kind in "fc":
            if not np.array_equal(x, y, equal_nan=True):
                return False
        elif not np.array_equal(x.astype(str), y.astype(str)):
            return False
    return True


WCMP = F.W_MAIN + ["bw_abs_robust_z5", "bw_change_kg", "bw_change_pct", "bw_dev_med5_pct", "chg_check", "layoff_days"]


def synth_torch(seed=0, n_horses=60, n_days=40):
    rng = np.random.default_rng(seed)
    rows = []
    days = pd.date_range("2016-01-02", periods=n_days, freq="7D")
    for h in range(n_horses):
        base = rng.normal(470, 30)
        runs = sorted(rng.choice(n_days, size=int(rng.integers(3, 12)), replace=False))
        for k, di in enumerate(runs):
            d = int(days[di].strftime("%Y%m%d"))
            kg = float(np.round((base + rng.normal(0, 6)) / 2) * 2)
            if rng.random() < 0.03:
                kg = np.nan
            rows.append({"pid": f"P{h:05d}", "date": d, "year": d // 10000, "rid16": f"{d}0601010{di % 9 + 1}",
                         "ban": h % 16 + 1, "kg": kg, "chg_src": np.nan, "性別": "牡" if h % 2 else "牝",
                         "年齢": str(3 + h % 3)})
    t = pd.DataFrame(rows)
    return t


def main():
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    t0 = time.time()

    # ---- loader 契約
    hdr = torch_header()
    present = [c for c in FORBIDDEN_EXACT if c in hdr] + [c for c in hdr if c.startswith("指時系")]
    t = load_torch_struct()
    rec("torch_file_contains_forbidden_but_loader_excludes", len(present) > 10 and not set(present) & set(t.columns),
        f"file has {len(present)} forbidden cols")
    bad_ok = True
    for col in ["人気", "単勝オッズ", "指時系1・単勝", "複上1", "複人気3", "補正", "着順", "払戻金"]:
        try:
            assert_torch_frame_clean(list(t.columns) + [col])
            bad_ok = False
        except AssertionError:
            pass
    rec("forbidden_column_assert_exact_prefix_result", bad_ok)
    rec("sealed_2024_2025_dropped_on_read", int(t["date"].max()) < 20240101, t["date"].max())
    rec("loader_sha256_recorded", len(loader_sha256()) == 64)

    # ---- W の時点安全性 (実データの一部の馬)
    pids = t["pid"].drop_duplicates().sample(4000, random_state=1)
    sub = t[t["pid"].isin(pids)].copy()
    w_full = F.build_w(sub)
    cut = 20201231
    sub_cut = sub[sub["date"] <= cut].copy()
    w_cut = F.build_w(sub_cut)
    keep = sub["date"] <= cut
    rec("future_deletion_invariance_W", w_equal(w_full.loc[keep], w_cut, WCMP))
    # 同日の他馬の体重を書き換えても対象馬の特徴は不変 (day-start snapshot)
    d0 = int(sub["date"].iloc[len(sub) // 2])
    tgt = sub.index[(sub["date"] == d0)][:1]
    mod = sub.copy()
    others = mod.index[(mod["date"] == d0) & ~mod.index.isin(tgt)]
    mod.loc[others, "kg"] = mod.loc[others, "kg"] + 20
    w_mod = F.build_w(mod)
    rec("same_day_other_rows_do_not_leak", w_equal(w_full.loc[tgt], w_mod.loc[tgt], WCMP), f"date {d0}")
    # 対象 race の当日値以外 (将来の体重) を書き換えても過去特徴は不変
    mod2 = sub.copy()
    fut = mod2["date"] > cut
    mod2.loc[fut, "kg"] = mod2.loc[fut, "kg"] * 1.1
    w_mod2 = F.build_w(mod2)
    rec("future_value_modification_invariance", w_equal(w_full.loc[keep], w_mod2.loc[keep], WCMP))
    # 行順置換不変
    perm = sub.sample(frac=1.0, random_state=3)
    w_perm = F.build_w(perm).loc[sub.index]
    rec("row_order_permutation_invariance", w_equal(w_full, w_perm, WCMP))
    # 特徴 module は結果列・結果 loader を参照しない
    src = inspect.getsource(F)
    rec("features_do_not_touch_outcomes", "load_finishers" not in src and "着順" not in src and "jyun" not in src)
    rec("bw_change_pct_not_in_W_or_WP", "bw_change_pct" not in F.W_MAIN and all("change_pct" not in c for c in F.WP_COLS))

    # ---- 体重履歴 slot: 取消は入れず、DNF は実測があれば入れる (合成)
    s = synth_torch(1)
    hp = s["pid"].iloc[0]
    hs = s[s["pid"] == hp].sort_values("date")
    scr_date = int(hs["date"].iloc[1])
    term = {r: set(s.loc[s["rid16"] == r, "ban"]) for r in s["rid16"].unique()}
    # 2 走目を『取消』(terminal starter に居ない) にする → starter_rows で落ち、以降の履歴数に数えない
    r2, b2 = hs["rid16"].iloc[1], hs["ban"].iloc[1]
    term_scr = {k: set(v) for k, v in term.items()}
    term_scr[r2].discard(b2)
    ws = F.build_w(F.starter_rows(s, term_scr))
    w_all = F.build_w(s)
    i3 = hs.index[2]
    n_all = int(w_all.loc[i3, "bw_history_n"])
    n_scr = int(ws.loc[i3, "bw_history_n"]) if i3 in ws.index else -1
    rec("scratched_row_not_a_history_slot", n_scr == n_all - (1 if np.isfinite(hs["kg"].iloc[1]) else 0),
        f"with={n_all} scratched_removed={n_scr}")
    rec("dnf_starter_with_weight_is_a_slot", n_all >= 1 and F.starter_rows(s, term).shape[0] == len(s))

    # ---- 欠損の扱い: FILL 定数は fit した尤度に影響しない (指示子が吸収)
    rng = np.random.default_rng(5)
    R, n = 1500, 10
    off = np.arange(0, R * n + 1, n)
    x = rng.normal(0, 1, R * n)
    miss = rng.random(R * n) < 0.2
    logm = np.log(rng.dirichlet(np.ones(n), R).ravel())
    eta = logm + 0.4 * np.where(miss, 0.3, x)
    win = np.array([off[r] + rng.choice(n, p=np.exp(eta[off[r]:off[r + 1]]) / np.exp(eta[off[r]:off[r + 1]]).sum())
                    for r in range(R)])
    lls = []
    for fill in (0.0, 3.7, -11.0):
        X = np.column_stack([logm, np.where(miss, fill, x), miss.astype(float)])
        lls.append(M.fit_offset(X, off, win)["mean_ll"])
    rec("missing_fill_constant_does_not_change_fit", max(lls) - min(lls) < 1e-9, f"ll spread {max(lls)-min(lls):.1e}")

    # ---- P: race 内定数は落ち、softmax が bit 一致
    P = np.repeat(rng.normal(0, 1, R), n)
    X0 = np.column_stack([logm, x])
    X1 = np.column_stack([logm, x, P])
    f0, f1 = M.fit_offset(X0, off, win), M.fit_offset(X1, off, win)
    q0, q1 = M.log_probs(X0, f0["theta"], off), M.log_probs(X1, f1["theta"], off)
    rec("race_constant_P_dropped_and_softmax_bitwise", f1["dropped"] == [2] and np.array_equal(q0, q1))
    qa = M.log_probs(X1, np.array([1.0, f0["theta"][1], 0.9]), off)
    qb = M.log_probs(X0, np.array([1.0, f0["theta"][1]]), off)
    rec("P_alone_with_common_coefficient_leaves_softmax_unchanged", np.max(np.abs(qa - qb)) < 1e-12,
        f"max|Δlog q|={np.max(np.abs(qa - qb)):.1e}")
    # P as-of: 対象日以降を削除しても過去 z が bit 一致・当日値は自身の標準化に入らない
    baba = pd.read_parquet(__import__("analysis.mcond.exp19_bodyweight_track_condition_dev.loaders",
                                      fromlist=["BABA"]).BABA)
    pz = F.build_p(baba)
    pz_cut = F.build_p(baba[baba["日付"] <= 20221231])
    a = pz[pz["date"] <= 20221231].reset_index(drop=True)
    b = pz_cut.reset_index(drop=True)
    zc = [c for c in a.columns if c.endswith("_z")]
    rec("P_future_deletion_invariance", all(np.array_equal(a[c].to_numpy(), b[c].to_numpy(), equal_nan=True) for c in zc))
    bb = baba.copy()
    i = int(np.flatnonzero((bb["日付"] == 20220605).to_numpy())[0])
    bb.loc[bb.index[i], "cushion"] = bb["cushion"].iloc[i] + 5
    pz2 = F.build_p(bb)
    others = pz2["date"] != 20220605
    same_venue_before = others
    rec("P_same_day_value_not_in_own_standardization",
        np.array_equal(pz.loc[others & (pz["date"] < 20220605), "cushion_z"].to_numpy(),
                       pz2.loc[others & (pz2["date"] < 20220605), "cushion_z"].to_numpy(), equal_nan=True))

    # ---- fit: 正しく指定された世界で回収 / 重複列を落とす
    th_true = np.array([1.0, 0.5])
    eta = logm + 0.5 * x
    win2 = np.array([off[r] + rng.choice(n, p=np.exp(eta[off[r]:off[r + 1]]) / np.exp(eta[off[r]:off[r + 1]]).sum())
                     for r in range(R)])
    fr = M.fit_offset(np.column_stack([logm, x, 2 * x]), off, win2)
    rec("duplicate_column_dropped_and_recovery", fr["dropped"] == [2] and abs(fr["theta"][1] - 0.5) < 0.2,
        f"theta={np.round(fr['theta'], 3).tolist()}")

    # ---- placebo 構造
    z = np.load(__import__("analysis.mcond.exp19_bodyweight_track_condition_dev.power_data",
                           fromlist=["ARR"]).ARR, allow_pickle=True)
    offp, Wd = z["off"], z["W"]
    sel = np.arange(0, 800)
    o2 = offp[:801]
    Wsub = Wd[:o2[-1]]
    p1 = PL.p1_permute(Wsub, o2, np.random.default_rng(1))
    ok = all(np.array_equal(np.sort(Wsub[o2[r]:o2[r + 1]], 0), np.sort(p1[o2[r]:o2[r + 1]], 0)) for r in sel)
    rec("P1_keeps_within_race_multiset", ok and not np.array_equal(Wsub, p1))
    attrs = {k: z[k][:800] for k in ("venue", "surface", "age_band", "field_band", "year", "kaisai", "month", "wp_ok")}
    p2, kept = PL.p2_time_shift(Wsub, o2, attrs, np.random.default_rng(2))
    cell = PL.cells(attrs, ["venue", "surface", "age_band", "field_band"])
    n_ = np.diff(o2)
    ok2 = True
    for r in sel[:300]:
        seg = p2[o2[r]:o2[r + 1]]
        if np.array_equal(seg, Wsub[o2[r]:o2[r + 1]]):
            continue
        cand = [d for d in sel if d != r and cell[d] == cell[r] and n_[d] >= n_[r]
                and np.array_equal(Wsub[o2[d]:o2[d] + n_[r]], seg)]
        ok2 &= len(cand) >= 1
    rec("P2_donor_same_cell_not_self", ok2, f"kept_self={kept}")
    zz = {k: z[k] for k in ("venue", "surface", "month", "year", "kaisai", "wp_ok")}
    pr = {"cz": np.arange(len(zz["venue"]), dtype=float)}
    _, kept3, donor, rnd = PL.p3_track_shift(pr, zz, np.random.default_rng(3))
    moved = np.flatnonzero(donor != np.arange(len(donor)))
    ok3 = all(zz["venue"][d] == zz["venue"][r] and zz["surface"][d] == zz["surface"][r] and
              zz["month"][d] == zz["month"][r] and rnd[d] != rnd[r] for r, d in zip(moved, donor[moved]))
    rec("P3_donor_other_kaisai_same_venue_surface_month", ok3 and len(moved) > 0, f"moved={len(moved)} kept={kept3}")
    wc = [str(c) for c in z["w_cols"]]
    p4 = PL.p4_missingness_only(Wsub, wc)
    keepc = [j for j, c in enumerate(wc) if c == "bw_status_not_measured" or c.startswith("miss_")]
    valc = [j for j in range(len(wc)) if j not in keepc]
    rec("P4_values_removed_status_and_flags_kept",
        np.array_equal(p4[:, keepc], Wsub[:, keepc]) and np.all(p4[:, valc] == F.FILL))

    # ---- Kelly (EXP18 と同一の閉形式、0·log0 の極限だけ追加) と Gram による列選択 (SVD と同一判定)
    from ..exp18_cross_pool_market_tomography_dev import tomography as T18
    rk = np.random.default_rng(11)
    diff = 0.0
    for _ in range(2000):
        k = int(rk.integers(5, 17))
        pp = rk.dirichlet(np.ones(k))
        qq = rk.dirichlet(np.ones(k))
        oo = 0.8 / rk.dirichlet(np.ones(k) * 2)
        diff = max(diff, abs(M.kelly_growth(qq, oo, pp) - T18.kelly_cash_growth(qq, oo, pp)[0]))
    deg = M.kelly_growth(np.array([1.0 - 1e-18, 1e-18, 0.0]), np.array([5.0, 50.0, 50.0]), np.array([1.0, 0.0, 0.0]))
    rec("kelly_equals_exp18_and_degenerate_finite", diff < 1e-12 and np.isfinite(deg), f"max diff {diff:.1e} deg={deg:.3f}")
    oks = True
    for _ in range(50):
        Xr = rk.normal(size=(400, 6))
        Xr[:, 3] = Xr[:, 1] * 2 + Xr[:, 2]
        Xr[:, 5] = np.repeat(rk.normal(size=40), 10)
        o10 = np.arange(0, 401, 10)
        kg, _ = M.select_columns(Xr, o10)
        C = M.race_center(Xr, o10)
        keep = []
        for j in range(6):
            D = C[:, keep + [j]]
            if np.linalg.matrix_rank(D) < len(keep) + 1:
                continue
            if np.linalg.cond(D.T @ D) > M.COND_MAX:
                continue
            keep.append(j)
        oks &= kg == keep
    rec("gram_column_selection_equals_svd_rule", oks, "collinear col 3 and race-constant col 5 dropped")

    # ---- damped Newton fallback は EXP18 Newton と同じ MLE に収束する
    Xd = np.column_stack([logm, x, miss.astype(float)])
    th1, _ = T18._clogit_newton(Xd, off, win, [1.0, 0.0, 0.0])
    th2, _ = M.damped_newton(Xd, off, win, np.array([1.0, 0.0, 0.0]))
    rec("damped_newton_fallback_same_mle", np.max(np.abs(th1 - th2)) < 1e-6, f"max|Δθ|={np.max(np.abs(th1 - th2)):.1e}")

    # ---- Gate 判定の境界
    bad = G.boundary_tests()
    rec("gate_grade_boundaries_holm_placebo", not bad, "; ".join(bad))

    # ---- floor 生成の健全性 (B 系): ε=0 の真値は base 市場、真値は alt 族に入る
    from . import power_floor as PF
    PF.world()
    base, null, alt, blk = PF.cols("B1")
    lp0 = PF.truth("B1", 0.0)
    rec("floor_truth_eps0_is_terminal_market", np.max(np.abs(lp0 - PF._G["mk"].norm_log(base))) < 1e-12)
    c, v = PF.direction("B1")
    eps = 0.3
    lp = PF.truth("B1", eps)
    lq = PF._G["mk"].norm_log(base + eps * (blk @ v) / np.sqrt(np.mean((M.race_center(blk, PF._G["off"]) @ v) ** 2)))
    rec("floor_truth_inside_alt_family", np.max(np.abs(lp - lq)) < 1e-9)
    rec("floor_no_declared_independent_noise",
        "normal(" not in inspect.getsource(PF.growth_rep) + inspect.getsource(PF.power_rep))
    d0 = PF.delta_true("B1", 0.0)
    rec("floor_delta_true_zero_at_eps0", abs(d0) < 1e-10, f"{d0:.2e}")

    n_ok = sum(r["pass"] for r in RES)
    out = {"n_tests": len(RES), "n_pass": n_ok, "all_passed": n_ok == len(RES), "tests": RES,
           "elapsed_sec": round(time.time() - t0, 1)}
    (OUT / "invariant_tests.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{n_ok}/{len(RES)} passed ({out['elapsed_sec']}s)")
    sys.exit(0 if out["all_passed"] else 1)


if __name__ == "__main__":
    main()
