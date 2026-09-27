# -*- coding: utf-8 -*-
"""
test_invariants.py — EXP21 Stage 0 の合成 invariant / 構造テスト (結果 ROI は計算しない)
出力: out/invariant_tests.json
"""
from __future__ import annotations

import inspect
import json
import sys
import time
from itertools import combinations

import numpy as np
import pandas as pd

from . import bands as B
from . import g0_audit as G
from . import loaders as L
from . import power_audit as P

RES = []


def rec(name, ok, detail=""):
    RES.append({"test": name, "pass": bool(ok), "detail": str(detail)})
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)


def synth_race(n, rng, seed_frame=True):
    """整合した合成 TARGET 1 race (91 + trio 列)。戻り: DataFrame 行 (ban index), 真の winner/payout"""
    pi = rng.dirichlet(np.ones(n) * 2)
    tan = np.round(0.8 / pi, 1)
    waku = np.array([min(8, 1 + (b - 1) * 8 // n) for b in range(1, n + 1)])
    um = np.zeros((19, 19))
    ut = np.zeros((19, 19))
    wl = np.zeros((19, 19))
    wh = np.zeros((19, 19))
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if i != j:
                ut[i, j] = np.round(0.75 / (pi[i - 1] * pi[j - 1] / (1 - pi[i - 1])), 1)
            if i < j:
                v = np.round(0.775 / (pi[i - 1] * pi[j - 1] * 2), 1)
                um[i, j] = um[j, i] = v
                lo = np.round(v / 4, 1) + 1.0
                wl[i, j] = wl[j, i] = lo
                wh[i, j] = wh[j, i] = np.round(lo * 1.2, 1)
    wk = np.zeros((9, 9))
    for fi in range(1, 9):
        for fj in range(fi, 9):
            m = [um[i, j] for i in range(1, n + 1) for j in range(1, n + 1) if i < j and
                 {waku[i - 1], waku[j - 1]} == {fi, fj}]
            if m:
                wk[fi, fj] = wk[fj, fi] = np.round(min(m) * 0.9, 1)
    trio = {}
    for c in combinations(range(1, n + 1), 3):
        trio[c] = np.round(0.75 / np.prod(pi[np.array(c) - 1]) / 50, 1)
    rows = []
    for b in range(1, n + 1):
        v = [tan[b - 1], np.round(tan[b - 1] / 3, 1) + 1.0, np.round(tan[b - 1] / 2, 1) + 1.1]
        v += [um[b, j] for j in range(1, 19)] + [wk[waku[b - 1], f] for f in range(1, 9)]
        for j in range(1, 19):
            v += [wl[b, j], wh[b, j]]
        v += [ut[b, j] for j in range(1, 19)]
        v += [trio.get(tuple(sorted((b, j, k))), 0.0) for j, k in L.TRIO_PAIRS[b]]
        rows.append({"ban": b, "waku": int(waku[b - 1]), "n": n, "vals": np.array(v, float)})
    g = pd.DataFrame(rows).set_index("ban")
    a, b2, c = [int(x) for x in rng.choice(np.arange(1, n + 1), 3, replace=False)]
    win = {"first": [a], "second": [b2], "third": [c], "dead_heat": False, "tan": round(tan[a - 1] * 100),
           "fuku": {x: round(np.sqrt(g.loc[x, "vals"][1] * g.loc[x, "vals"][2]) * 100) for x in (a, b2, c)},
           "waku": round(wk[waku[a - 1], waku[b2 - 1]] * 100), "umaren": round(um[a, b2] * 100),
           "umatan": round(ut[a, b2] * 100), "trio": round(trio[tuple(sorted((a, b2, c)))] * 100), "trifecta": np.nan}
    wide = [(min(x, y), max(x, y), round(np.sqrt(wl[x, y] * wh[x, y]) * 100)) for x, y in ((a, b2), (a, c), (b2, c))]
    return g, win, wide


def main():
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    t0 = time.time()
    rng = np.random.default_rng(7)

    # ---- parser / 配置
    rec("parse_key_hex_nichi", L.parse_key("0623591201") == ("06", "23", 5, 9, 12, 1)
        and L.parse_key("0526AC0101")[3] == 12)
    v = np.arange(219, dtype=float)
    s = L.split_vals(v, True)
    rec("split_vals_block_sizes", len(s["umaren"]) == 18 and len(s["wakuren"]) == 8 and len(s["wide_lo"]) == 18
        and len(s["umatan"]) == 18 and len(s["trio"]) == 136 and s["wide_lo"][0] == 29 and s["wide_hi"][0] == 30)
    cover = {}
    for i in range(1, 19):
        for j, k in L.TRIO_PAIRS[i]:
            cover[tuple(sorted((i, j, k)))] = cover.get(tuple(sorted((i, j, k))), 0) + 1
    rec("trio_anchor_layout_136_and_each_triple_three_times", all(len(L.TRIO_PAIRS[i]) == 136 for i in L.TRIO_PAIRS)
        and len(cover) == 816 and set(cover.values()) == {3})

    # ---- 合成 race で配置逆引きと key 検査
    rows, wins, wides = {}, {}, {}
    for r in range(60):
        g, w, wd = synth_race(int(rng.integers(9, 17)), rng)
        rid = f"2023010106010{r:03d}"[:16]
        rows[rid], wins[rid], wides[rid] = g, w, wd
    lay = G.verify_layout(rows, wins, wides, with_trio=True)
    rec("layout_verifier_passes_consistent_synthetic", all(lay[t]["layout_pass"] for t in
                                                             ("tansho", "fukusho", "wakuren", "umaren", "wide", "umatan", "sanrenpuku")),
        {t: lay[t]["hyp"]["rate"] for t in ("wakuren", "umatan", "sanrenpuku")})
    bad = {rid: g.copy() for rid, g in rows.items()}
    for rid, g in bad.items():
        g["vals"] = [np.concatenate([x[:65], x[65:83][::-1], x[83:]]) for x in g["vals"]]
    lay_bad = G.verify_layout(bad, wins, wides, with_trio=True)
    rec("layout_verifier_rejects_scrambled_umatan", not lay_bad["umatan"]["layout_pass"])
    kc = G.key_checks(rows, with_trio=True)
    rec("key_checks_pass_on_consistent_synthetic", kc["pass"], kc)
    asym = {rid: g.copy() for rid, g in list(rows.items())[:3]}
    for rid, g in asym.items():
        x = g["vals"].iloc[0].copy()
        x[3 + 1] += 1.0
        g.at[g.index[0], "vals"] = x
    rec("key_checks_detect_asymmetry", G.key_checks(asym, with_trio=True)["umaren_asym"] > 0)

    # ---- 帯 (label-free)
    key = np.exp(rng.normal(3, 1.2, 50000))
    q = rng.random(50000)
    q = q / q.sum()
    race = rng.integers(0, 3000, 50000)
    day = race // 12
    mb = B.mass_bands(key, q, 10)
    masses = np.bincount(mb, weights=q, minlength=10)
    rec("mass_bands_equal_mass", np.max(np.abs(masses - 0.1)) <= q.max() + 1e-12, np.round(masses, 4).tolist())
    order_ok = all(key[mb == b].max() <= key[mb == b + 1].min() for b in range(9))
    rec("mass_bands_monotone_in_odds", order_ok)
    cb = B.count_bands(key, 10)
    rec("count_bands_equal_count", set(np.bincount(cb)) <= {5000})
    fb = B.fixed_bands(np.array([1.0, 2.0, 2.0001, 3.0, 5.1, 15.0, 15.1, 2500.0, 20000.0]))
    rec("fixed_bands_right_closed", fb.tolist() == [0, 0, 1, 1, 3, 4, 5, 11, 14], fb.tolist())
    rec("bands_take_no_outcome_argument", all("pay" not in inspect.signature(f).parameters
                                              for f in (B.mass_bands, B.count_bands, B.fixed_bands)))
    src = inspect.getsource(B)
    rec("bands_module_reads_no_payouts", "load_kekka" not in src and "load_wide" not in src and "確定着順" not in src)
    summ = B.summarize(key, q * 5000, race, day, mb, 10, key)
    rec("insufficient_rule_applies", all(("insufficient" in r) for r in summ if r["tickets"]))

    # ---- G1 判定と power の較正 null
    base = 0.8
    rd = np.array([0.85, 0.84, 0.83, 0.82, 0.81, 0.79, 0.78, 0.77, 0.76, 0.75])
    rec("g1_pass_on_replicated_shape", P.g1_pass(rd, rd - 0.001, base))
    re = rd.copy()
    re[[0, 1, 2]] = [0.79, 0.78, 0.795]
    rec("g1_fail_when_3_signs_flip", not P.g1_pass(rd, re, base))
    d = {"key": np.array([2.0, 4.0, 8.0, 1.5, 3.0, 12.0] * 200), "race": np.repeat(np.arange(400), 3)}
    qq = np.tile(np.array([0.5, 0.3, 0.2]), 400) * 1.0
    d["q"] = qq * 3
    d["pay"] = d["key"]
    m, sd = P.band_moments(d, "fukusho", 0.0)
    rec("fukusho_null_roi_equals_baseline", np.allclose(m[np.isfinite(m)], 0.8, atol=1e-9), np.round(m, 4).tolist())
    rec("stage1_bootstrap_frozen", P.STAGE1_BOOT_B == 10000 and P.SEED == 20260927 and P.STAGE1_BOOT_SEED == 20260928)

    # ---- 封印
    k = L.load_kekka_master()
    rec("kekka_master_2024_2025_dropped", int(k["date"].max()) < 20240101)
    w = L.load_wide_payouts()
    rec("wide_payouts_2024_2025_dropped", int(w["date"].max()) < 20240101)

    n_ok = sum(r["pass"] for r in RES)
    out = {"n_tests": len(RES), "n_pass": n_ok, "all_passed": n_ok == len(RES), "tests": RES,
           "elapsed_sec": round(time.time() - t0, 1)}
    (L.OUT / "invariant_tests.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{n_ok}/{len(RES)} passed ({out['elapsed_sec']}s)")
    sys.exit(0 if out["all_passed"] else 1)


if __name__ == "__main__":
    main()
