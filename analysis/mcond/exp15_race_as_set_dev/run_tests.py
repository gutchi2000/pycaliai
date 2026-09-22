# -*- coding: utf-8 -*-
"""
run_tests.py — 必須テスト T1-T16 (spec.required_tests)
======================================================
学習前 (合成データ + データ契約) の G0。出力: out/tests_pre.json
学習済みモデルでの T1 (2023 実レース)・T6・T15 は train_nn.py が out/tests_post.json に記録する。
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd
import torch

from . import common as C
from . import nnlib as N

RES = {}


def rec(name, ok, detail=""):
    RES[name] = {"pass": bool(ok), "detail": detail}
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def synth(n_races=4, sizes=(5, 9, 14, 18), nc=3, nn_=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    B = len(sizes)
    cat = torch.randint(0, 7, (B, 18, nc), generator=g)
    num = torch.randn(B, 18, nn_, generator=g)
    mask = torch.zeros(B, 18, dtype=torch.bool)
    for b, n in enumerate(sizes):
        mask[b, :n] = True
    cat[~mask] = 0
    num[~mask] = 0
    return cat.to(N.DEV), num.to(N.DEV), mask.to(N.DEV)


def probs(s, mask):
    z = s.masked_fill(~mask, -torch.inf)
    return torch.softmax(z, dim=1)


def model_pair(nc=3, nn_=6):
    torch.manual_seed(1)
    cards = [7] * nc
    return (N.R1(cards, nn_, 64, 0.1).to(N.DEV).eval(),
            N.R1NoCtx(cards, nn_, 64, 0.1, N.noctx_width(cards, nn_, 64, 0.1)).to(N.DEV).eval())


@torch.no_grad()
def t1_perm(model, cat, num, mask, tol=1e-5, reps=5):
    base = model(cat, num, mask)
    worst = 0.0
    g = torch.Generator().manual_seed(7)
    for _ in range(reps):
        cat2, num2, s_perm = cat.clone(), num.clone(), []
        perms = []
        for b in range(cat.shape[0]):
            n = int(mask[b].sum())
            pi = torch.randperm(n, generator=g).to(N.DEV)
            perms.append(pi)
            cat2[b, :n] = cat[b, pi]
            num2[b, :n] = num[b, pi]
        out = model(cat2, num2, mask)
        for b, pi in enumerate(perms):
            n = len(pi)
            worst = max(worst, float((out[b, :n] - base[b, pi]).abs().max()))
            pa = probs(out[b:b + 1], mask[b:b + 1])[0, :n]
            pb = probs(base[b:b + 1], mask[b:b + 1])[0, pi]
            worst = max(worst, float((pa - pb).abs().max()))
    return worst <= tol, worst


def pre():
    torch.use_deterministic_algorithms(True)
    cat, num, mask = synth()
    r1, r0 = model_pair()
    # T1 合成
    for nm, m in (("R1", r1), ("R1-noctx", r0)):
        ok, w = t1_perm(m, cat, num, mask)
        rec(f"T1_synthetic_{nm}", ok, f"max|diff|={w:.2e}")
    # T2 race 間混入 + 静的検査
    with torch.no_grad():
        a = r1(cat, num, mask)
        num_b = num.clone()
        num_b[1:] = torch.randn_like(num_b[1:]) * mask[1:].unsqueeze(-1)
        b = r1(cat, num_b, mask)
    d = float((a[0, :5] - b[0, :5]).abs().max())
    bad = [type(mm).__name__ for mm in list(r1.modules()) + list(r0.modules())
           if "BatchNorm" in type(mm).__name__ or "InstanceNorm" in type(mm).__name__
           or "GroupNorm" in type(mm).__name__]
    rec("T2_no_cross_race", d == 0.0 and not bad, f"diff={d:.2e} batchstat_layers={bad}")
    # T3 padding
    with torch.no_grad():
        cat_g, num_g = cat.clone(), num.clone()
        num_g[~mask] = torch.randn_like(num_g[~mask]) * 100
        cat_g[~mask] = 5
        out = r1(cat_g, num_g, mask)
        base = r1(cat, num, mask)
        pz = probs(out, mask)
    d = float(((out - base).abs() * mask).max())
    rec("T3_padding", d <= 1e-6 and float(pz[~mask].abs().max()) == 0.0, f"diff={d:.2e}")
    # T4 取消: 馬 h を除いて集合を作り直した出力と一致、Σp=1
    with torch.no_grad():
        b_, h = 2, 3
        n = int(mask[b_].sum())
        keep = [j for j in range(n) if j != h]
        cat_s = torch.zeros_like(cat[b_:b_ + 1]); num_s = torch.zeros_like(num[b_:b_ + 1])
        m_s = torch.zeros_like(mask[b_:b_ + 1])
        cat_s[0, :n - 1] = cat[b_, keep]; num_s[0, :n - 1] = num[b_, keep]; m_s[0, :n - 1] = True
        rebuilt = r1(cat_s, num_s, m_s)[0, :n - 1]
        # マスクで h を外しただけ (文脈から h が消える) の出力
        m_h = mask[b_:b_ + 1].clone(); m_h[0, h] = False
        masked = r1(cat[b_:b_ + 1], num[b_:b_ + 1], m_h)[0, keep]
        full = r1(cat[b_:b_ + 1], num[b_:b_ + 1], mask[b_:b_ + 1])[0, keep]
        ps = float(probs(rebuilt.unsqueeze(0), m_s[:, :n - 1]).sum())
    d = float((rebuilt - masked).abs().max())
    changed = float((full - masked).abs().max())
    rec("T4_scratch_recompute", d <= 1e-5 and abs(ps - 1) < 1e-6 and changed > 0,
        f"rebuilt_vs_masked={d:.2e} sum_p={ps:.6f} context_changed_by={changed:.2e}")
    # T5 n=5..18, n>18 assert
    ok5 = True
    for n in range(5, 19):
        c2, n2, m2 = synth(sizes=(n,), seed=n)
        with torch.no_grad():
            p = probs(r1(c2, n2, m2), m2)
        ok5 &= abs(float(p.sum()) - 1) < 1e-5
    try:
        import pandas as _pd
        fake = _pd.DataFrame({"rid16": ["X"] * 19, "period": ["train"] * 19, "ban": range(19),
                              "rel": 0, "win": 0})
        class E:  # 最小 enc
            cat_codes = np.zeros((19, 1), dtype=np.int64); num_nn = np.zeros((19, 1), np.float32)
        N.RaceTensors(E, fake, "train")
        raised = False
    except AssertionError:
        raised = True
    rec("T5_field_5_18", ok5 and raised, f"sum_ok={ok5} n19_assert={raised}")
    # T16 noctx は他馬を見ない
    with torch.no_grad():
        a = r0(cat, num, mask)
        num_o = num.clone(); num_o[0, 1:] = torch.randn_like(num_o[0, 1:])
        b = r0(cat, num_o, mask)
    rec("T16_noctx_blind", float((a[0, 0] - b[0, 0]).abs()) == 0.0, "")

    # ---- データ契約系
    con = C.load_contract()
    df = C.load_rows()
    rec("T12_no_2024plus", int(df["date"].max()) <= 20231231 and int(df["date"].min()) >= 20160101,
        f"range=[{int(df['date'].min())},{int(df['date'].max())}] hash={C.row_hash(df)[:16]}")
    clean = con["features"]["clean"]
    ids = ["馬主(最新/仮想)", "前走レースID(新)", "前走レースID(新/馬番無)", "血統登録番号", "馬名"]
    over = con["unique_id_guard"]["over_limit"]
    rec("T9_no_ids", not (set(ids) & set(clean)) and set(over) <= set(con["unique_id_guard"]["allowlist"]),
        f"over_limit={list(over)}")
    rid_like = ["レースID(新)", "レースID(新/馬番無)", "rid16", "meeting_day", "rid"]
    rec("T10_no_race_id", not (set(rid_like) & set(clean)), "Ｒ・前走日付は許可列")
    forbid = ["単勝オッズ", "単勝", "人気", "馬体重", "馬体重増減", "着順", "走破タイム", "通過順",
              "上り3F", "fukusho_flag", "roi_target", "確定着順", "着差タイム"]
    rec("T11_no_current_outcome", not (set(forbid) & set(clean)), "")
    enc = C.Encoded(df, con, "clean")
    tr = (df["period"] == "train").to_numpy()
    ok8 = enc.fit_stats["fit_rows"] == int(tr.sum())
    for c in enc.cats:
        ok8 &= set(enc.vocab[c]) <= set(df.loc[tr, c].astype(str)) | {"__NaN__"}
    num_tr = enc.num_raw[tr]
    ok8 &= np.allclose(enc.fit_stats["median"], np.nanmedian(num_tr, axis=0), equal_nan=True)
    rec("T8_fit_on_base_train_only", ok8, f"fit_rows={enc.fit_stats['fit_rows']}")
    # T7 未来削除: (i) T=20211231 で全再構築しても T 以前の入力が完全一致
    cut = df[df["date"] <= 20211231].reset_index(drop=True)
    enc2 = C.Encoded(cut, con, "clean")
    m1 = df["date"].to_numpy() <= 20211231
    same_r0 = np.array_equal(enc.X_r0[m1], enc2.X_r0)
    same_nn = (np.array_equal(enc.num_nn[m1], enc2.num_nn)
               and np.array_equal(enc.cat_codes[m1], enc2.cat_codes))
    # (ii) T=20190630: 生入力と race 集合 (文脈の構成員) が一致
    cut2 = df[df["date"] <= 20190630]
    m2 = df["date"].to_numpy() <= 20190630
    raw_same = all(df.loc[m2, c].fillna("<NA>").equals(cut2[c].fillna("<NA>")) for c in clean)
    ra = C.race_index(df, m2)
    rb = C.race_index(cut2.reset_index(drop=True), np.ones(len(cut2), bool))
    sets_same = len(ra) == len(rb) and all(len(x) == len(y) for x, y in zip(ra, rb))
    rec("T7_future_deletion", same_r0 and same_nn and raw_same and sets_same,
        f"T20211231 R0={same_r0} NN={same_nn}; T20190630 raw={raw_same} sets={sets_same}")
    # T13 同一集合規則
    ok13 = True
    for per in ("train", "sel", "dev"):
        rr = C.race_index(df, (df["period"] == per).to_numpy())
        sizes = df[df["period"] == per].groupby("rid16").size()
        ok13 &= sorted(len(i) for i in rr) == sorted(sizes.tolist())
        ok13 &= all(df["rid16"].to_numpy()[i].tolist().count(df["rid16"].to_numpy()[i][0]) == len(i)
                    for i in rr[:500])
    rec("T13_same_set_rule", ok13, "rid16 同一行=集合 (train/sel/dev)")
    # T14 カテゴリ表記
    sys.path.insert(0, str(C.BASE))
    import category_normalize as CN
    t = pd.DataFrame({"芝・ダ": ["ダート", "芝"]})
    t2 = CN.normalize_categorical(t)
    ok14 = set(t2["芝・ダ"]) <= set(enc.vocab["芝・ダ"])
    rec("T14_category_norm", ok14, f"{t['芝・ダ'].tolist()}->{t2['芝・ダ'].tolist()} vocab={list(enc.vocab['芝・ダ'])}")
    C.dump(RES, "tests_pre.json")
    return all(v["pass"] for v in RES.values())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.parse_args()
    ok = pre()
    print("ALL PASS" if ok else "SOME FAIL")
    sys.exit(0 if ok else 1)
