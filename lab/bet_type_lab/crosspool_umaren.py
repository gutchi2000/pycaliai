# -*- coding: utf-8 -*-
"""
crosspool_umaren.py — クロスプール整合性裁定（単勝プール → 馬連プール）
======================================================================
仮説: シャープな単勝プールの含意確率 q_win から Harville で組んだ馬連同時分布
  f(q_win) と、ソフトな馬連プールの実価格 q_umaren が「同時刻で」不整合なら、
  予測の優位ゼロでも +EV が取れる（市場 vs 市場）。カジュアル資金が偏在する
  構造セル（多頭数・重賞・人気集中）で不整合が最大化するか。

設計（Stage1 = 存在テスト, oracle上限版）:
  - 両プールとも 区分4=確定 オッズを使用（同時刻クロスセクショナル比較。
    タイミング裁定ではなく「単勝と馬連が互いに整合しているか」を測る）。
  - q_win = de-vig(確定単勝)。f = all_umaren_mat(q_win) [Harville]。
  - q_um  = de-vig(確定馬連ペアodds)。o_um = 確定馬連odds（決済倍率）。
  - EV_ij = f_ij * o_um_ij。realized hit = 着1-2 のペアか。
  - bet rule: EV>τ のペアを買い、ROI=mean(o_um*hit)。控除≈0.775 を超えるか。
  - 分割: fit/threshold選定=≤2023、評価=2024-25 OOS。
  - 構造層別: 頭数 / 重賞 / 人気集中HHI / odds帯。

★これは「自モデル vs 市場」ではなく「単勝市場 vs 馬連市場」。Harville が真の
  演算子と仮定する Stage1。Stage2 で残差学習(構造条件付き)と前売り(区分1)運用版へ。

実行: PYTHONUTF8=1 python crosspool_umaren.py
出力: reports/crosspool_umaren.json
"""
from __future__ import annotations
import glob, io, json, re, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

BASE = Path(__file__).parent
ODIR = BASE / "data/Time _series_odds"
MASTER = BASE / "data/master_v2_20130105-20251228.csv"
OUT = BASE / "reports/crosspool_umaren.json"

from backtest_pl_ev import all_umaren_mat   # Harville: feed q_win (sums to 1)

COL_RID = "レースID(新/馬番無)"


def _rid16(x):
    return re.sub(r"\D", "", str(x))[:16]


def _final_snaps(pattern, col_filter):
    """区分4(確定) を rid 毎に最終スナップ1行へ畳み、対象列を numeric 化して返す。
    col_filter(col_name)->key で {key: colname} を作る。"""
    frames = []
    keymap = None
    for f in sorted(glob.glob(str(ODIR / pattern))):
        d = pd.read_csv(f, encoding="cp932", low_memory=False)
        cols = list(d.columns); RID, KB, TM = cols[0], cols[1], cols[2]
        if keymap is None:
            keymap = {}
            for c in cols:
                k = col_filter(str(c))
                if k is not None:
                    keymap[k] = c
        d[KB] = pd.to_numeric(d[KB], errors="coerce")
        d = d[d[KB] == 4].copy()                       # ★確定のみ
        d["rid"] = d[RID].map(_rid16)
        d[TM] = pd.to_numeric(d[TM], errors="coerce")
        d = d.sort_values(TM).drop_duplicates("rid", keep="last")
        frames.append(d[["rid"] + list(keymap.values())])
    df = pd.concat(frames, ignore_index=True).drop_duplicates("rid", keep="last")
    arr = df[list(keymap.values())].apply(pd.to_numeric, errors="coerce").to_numpy()
    return df["rid"].to_numpy(), list(keymap.keys()), arr


def load_final_odds():
    """区分4(確定) の 単勝(per-horse) と 馬連(per-pair) odds を rid 毎に返す（ベクトル化）。"""
    tan = {}; ume = {}
    # --- TANPUK 単勝 ---
    tan_filter = lambda c: (int(re.match(r"^\s*(\d+)\s*単\s*$", c).group(1))
                            if re.match(r"^\s*(\d+)\s*単\s*$", c) else None)
    rids, keys, arr = _final_snaps("TANPUK_*.csv", tan_filter)
    keys = np.array(keys)
    for r in range(len(rids)):
        row = arr[r]
        m = np.isfinite(row) & (row > 1.0)
        if m.sum() >= 4:
            tan[rids[r]] = {int(b): float(v) for b, v in zip(keys[m], row[m])}
    # --- UMAREN 馬連 ---
    pair_re = re.compile(r"馬(\d{2})-(\d{2})")
    def pair_filter(c):
        mm = pair_re.search(c)
        return (int(mm.group(1)), int(mm.group(2))) if mm else None
    rids, keys, arr = _final_snaps("UMAREN_*.csv", pair_filter)
    for r in range(len(rids)):
        row = arr[r]
        m = np.isfinite(row) & (row > 1.0)
        if m.sum() >= 3:
            ume[rids[r]] = {keys[k]: float(row[k]) for k in np.nonzero(m)[0]}
    print(f"[odds] 確定 単勝 races={len(tan):,}  馬連 races={len(ume):,}")
    return tan, ume


def load_results():
    """rid -> (着1馬番, 着2馬番, 頭数, 重賞flag, クラス名)。master の 着順 から。"""
    df = pd.read_csv(MASTER, encoding="utf-8-sig",
                     usecols=lambda c: c in [COL_RID, "馬番", "着順", "出走頭数", "クラス名", "split"],
                     low_memory=False)
    df["rid"] = df[COL_RID].map(_rid16)
    df["着順"] = pd.to_numeric(df["着順"], errors="coerce")
    df["馬番"] = pd.to_numeric(df["馬番"], errors="coerce")
    res = {}
    grade_re = re.compile(r"(G1|G2|G3|GI|GII|GIII|重賞|J\.G|ジャンプ|オープン|OP)")
    for rid, g in df.groupby("rid", sort=False):
        g1 = g[g["着順"] == 1]; g2 = g[g["着順"] == 2]
        if len(g1) == 0 or len(g2) == 0:
            continue
        b1 = int(g1["馬番"].iloc[0]); b2 = int(g2["馬番"].iloc[0])
        cls = str(g["クラス名"].iloc[0])
        ntou = int(pd.to_numeric(g["出走頭数"].iloc[0], errors="coerce") or len(g))
        sp = str(g["split"].iloc[0])
        res[rid] = (b1, b2, ntou, 1 if grade_re.search(cls) else 0, cls, sp)
    print(f"[results] races with 1-2 finishers={len(res):,}")
    return res


def build_rows(tan, ume, res):
    recs = []
    for rid, od_um in ume.items():
        if rid not in tan or rid not in res:
            continue
        od_win = tan[rid]
        b1, b2, ntou, grade, cls, sp = res[rid]
        bans = sorted(od_win.keys())
        if len(bans) < 4:
            continue
        # q_win de-vig
        inv = np.array([1.0 / od_win[b] for b in bans])
        qw = inv / inv.sum()
        idx = {b: k for k, b in enumerate(bans)}
        f_mat = all_umaren_mat(qw)            # Harville (N,N), sums over i<j ~ 1
        hhi = float((qw ** 2).sum())          # 人気集中度
        # 馬連 de-vig（同レース内の全ペア）
        inv_um = {p: 1.0 / o for p, o in od_um.items()}
        s_um = sum(inv_um.values())
        win_pair = tuple(sorted((b1, b2)))
        for (i, j), o in od_um.items():
            if i not in idx or j not in idx:
                continue
            f_ij = float(f_mat[idx[i], idx[j]])
            q_um = inv_um[(i, j)] / s_um
            ev = f_ij * o
            hit = 1 if tuple(sorted((i, j))) == win_pair else 0
            recs.append((rid, sp, i, j, ntou, grade, hhi, o, f_ij, q_um, ev, hit))
    cols = ["rid", "split", "i", "j", "ntou", "grade", "hhi", "o_um", "f_win", "q_um", "ev", "hit"]
    df = pd.DataFrame.from_records(recs, columns=cols)
    print(f"[rows] pair rows={len(df):,}  races={df['rid'].nunique():,}")
    return df


def roi_by_threshold(df, label):
    out = {}
    for tau in [1.0, 1.1, 1.2, 1.3, 1.5, 2.0]:
        sub = df[df["ev"] >= tau]
        if len(sub) < 50:
            out[tau] = dict(n=int(len(sub)), roi=None, hit=None)
            continue
        roi = float((sub["o_um"] * sub["hit"]).mean())
        out[tau] = dict(n=int(len(sub)), roi=round(roi, 4), hit=round(float(sub["hit"].mean()), 4))
    print(f"  [{label}] ROI by EV-threshold:")
    for tau, v in out.items():
        if v["roi"] is not None:
            print(f"     EV>={tau}: n={v['n']:>6}  ROI={v['roi']:.4f}  hit={v['hit']:.4f}")
    return out


def main():
    tan, ume = load_final_odds()
    res = load_results()
    df = build_rows(tan, ume, res)

    # 整合性の素描: f_win vs q_um のズレが realized hit を予測するか
    df["mis"] = np.log(df["f_win"].clip(1e-9) / df["q_um"].clip(1e-9))  # >0: 単勝含意>馬連価格=馬連が過小
    tr = df[df["split"] == "train"]; te = df[df["split"].isin(["test"])]
    va = df[df["split"] == "valid"]
    te_all = df[df["split"].isin(["valid", "test"])]   # 2023+ を OOS 扱いで広めに見る
    print(f"\n[split] train rows={len(tr):,}  valid={len(va):,}  test={len(te):,}")

    print("\n=== 全体 ROI（控除越え=ROI>≈0.775 / 利益=ROI>1.0）===")
    res_json = {"all_train": roi_by_threshold(tr, "TRAIN<=2022"),
                "all_test": roi_by_threshold(te, "TEST 2024-25")}

    # 構造層別（OOS test）。カジュアル資金プロキシ別に EV>=1.2 の ROI
    print("\n=== 構造層別 ROI（test, EV>=1.2）===")
    strat = {}
    sub = te[te["ev"] >= 1.2]
    def cell(mask, name):
        s = sub[mask]
        if len(s) < 30:
            return
        roi = float((s["o_um"] * s["hit"]).mean())
        strat[name] = dict(n=int(len(s)), roi=round(roi, 4))
        print(f"   {name:22s} n={len(s):>6}  ROI={roi:.4f}")
    cell(sub["grade"] == 1, "重賞(casual多)")
    cell(sub["grade"] == 0, "平場")
    cell(sub["ntou"] >= 14, "多頭数>=14")
    cell(sub["ntou"] <= 10, "少頭数<=10")
    cell(sub["hhi"] >= sub["hhi"].median(), "人気集中hi")
    cell(sub["hhi"] < sub["hhi"].median(), "人気集中lo")
    cell((sub["o_um"] >= 10) & (sub["o_um"] < 50), "中穴帯 馬連10-50")
    cell(sub["o_um"] >= 50, "穴帯 馬連50+")
    cell(sub["o_um"] < 10, "本命帯 馬連<10")
    res_json["strat_test_ev1.2"] = strat

    # しきい値を train で選び test 評価（過学習チェック）
    print("\n=== train最良EVしきい値を test で評価 ===")
    best_tau, best = None, -1
    for tau in [1.0, 1.1, 1.2, 1.3, 1.5, 2.0]:
        s = tr[tr["ev"] >= tau]
        if len(s) < 500:
            continue
        roi = float((s["o_um"] * s["hit"]).mean())
        if roi > best:
            best, best_tau = roi, tau
    s = te[te["ev"] >= best_tau]
    te_roi = float((s["o_um"] * s["hit"]).mean()) if len(s) else None
    print(f"   train最良 EV>={best_tau} (train ROI={best:.4f}) → test ROI={te_roi:.4f} n={len(s)}")
    res_json["selected"] = dict(tau=best_tau, train_roi=round(best, 4),
                                test_roi=round(te_roi, 4) if te_roi else None, test_n=int(len(s)))

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fp:
        json.dump(res_json, fp, indent=2, ensure_ascii=False)
    print(f"\n[saved] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
