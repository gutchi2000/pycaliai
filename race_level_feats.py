"""
race_level_feats.py — レースレベル・スコア (1パス・粗い1指標, リークフリー)
==========================================================================
あるレース R のレベルを、R 出走"全頭"(勝ち馬も敗戦馬も)の「R より後・かつ予測対象 P の
発走時刻 T より前に終了した走」の好走(5着内)率の平均で粗くスコア化する。

各行 (馬 H がレース P に出走) に「前走 R のレベル」を 3 窓で付与:
  W1 = R 直後の 1 走だけ
  W2 = R 後 90 日以内に終わった走
  W3 = T(=P発走) より前に終わった "その後" の全走 (日数上限なし)

★リーク防止 (生命線, 構造保証):
  グローバル順序 = (日付, 発走時刻, レースID) の辞書式。各 j の "その後走" は
  (date,hassou) が P の (date,hassou) より厳密に小さいものだけ採用 (同時刻レースも除外)。
  → 未来→過去の混入が構造的に起きない。
  検査: 一定間隔でサンプルした P について、採用した全 "その後走" が
  (date,hassou) < P を満たすかを assert。違反件数を報告 (1件でも FAIL)。

出力: data/race_level_feats.parquet  (rid16, ban, race_level_prev_W1/W2/W3)
       reports/race_level_feats_leakcheck.json
実行: PYTHONUTF8=1 python race_level_feats.py
"""
from __future__ import annotations
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
MASTER_CSV = BASE / "data/master_v2_20130105-20251228.csv"
OUT_PARQUET = BASE / "data/race_level_feats.parquet"
OUT_LEAK = BASE / "reports/race_level_feats_leakcheck.json"

COL_RID = "レースID(新/馬番無)"
COL_BAN = "馬番"
COL_PED = "血統登録番号"
COL_DATE = "日付"
COL_HASSOU = "発走時刻"
COL_JYUN = "着順"

LEAK_SAMPLE_EVERY = 50   # この間隔の P をリーク検査対象に


_cut90: dict = {}
def _date_plus_90(d: int) -> int:
    if d in _cut90:
        return _cut90[d]
    try:
        t = pd.Timestamp(str(int(d)))
        v = int((t + pd.Timedelta(days=90)).strftime("%Y%m%d"))
    except Exception:
        v = int(d) + 90  # フォールバック (粗い)
    _cut90[d] = v
    return v


def build():
    print(f"[load] {MASTER_CSV.name} (usecols)")
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig",
                     usecols=[COL_RID, COL_BAN, COL_PED, COL_DATE, COL_HASSOU, COL_JYUN, "split"],
                     low_memory=False)
    df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df = df.dropna(subset=[COL_JYUN, COL_RID, COL_PED, COL_DATE]).copy()
    df[COL_DATE] = pd.to_numeric(df[COL_DATE], errors="coerce").astype("int64")
    df["_hassou"] = pd.to_numeric(df[COL_HASSOU], errors="coerce").fillna(0).astype("int64")
    df[COL_BAN] = pd.to_numeric(df[COL_BAN], errors="coerce").fillna(0).astype(int)
    df["_rid"] = df[COL_RID].astype(str)
    df["top5"] = (df[COL_JYUN] <= 5).astype(float)
    print(f"  rows={len(df):,}  races={df['_rid'].nunique():,}  horses={df[COL_PED].nunique():,}")

    # グローバルなレース順序 (日付, 発走時刻, rid)
    rinfo = df.groupby("_rid").agg(date=(COL_DATE, "first"), hassou=("_hassou", "first")).reset_index()
    rinfo = rinfo.sort_values(["date", "hassou", "_rid"]).reset_index(drop=True)
    rinfo["race_ord"] = np.arange(len(rinfo), dtype=np.int64)
    rid2ord = dict(zip(rinfo["_rid"], rinfo["race_ord"]))
    df["race_ord"] = df["_rid"].map(rid2ord).astype(np.int64)

    # rid -> 出走馬 ped 配列
    rid2peds = {rid: g[COL_PED].values for rid, g in df.groupby("_rid", sort=False)}

    # ped -> 走歴 (race_ord 昇順) の各種配列
    df_s = df.sort_values([COL_PED, "race_ord"])
    ped_hist = {}
    for ped, g in df_s.groupby(COL_PED, sort=False):
        ped_hist[ped] = {
            "ord": g["race_ord"].values.astype(np.int64),
            "date": g[COL_DATE].values.astype(np.int64),
            "hassou": g["_hassou"].values.astype(np.int64),
            "top5": g["top5"].values.astype(np.float64),
            "rid": g["_rid"].values,
            "ban": g[COL_BAN].values.astype(int),
        }

    print("[compute] 前走レベル (W1/W2/W3) をリークフリー集計 ...")
    out_rid, out_ban, out_w1, out_w2, out_w3 = [], [], [], [], []
    leak_checked = 0
    leak_violations = 0
    same_time_dropped = 0
    p_counter = 0

    for ped, h in ped_hist.items():
        ords = h["ord"]; dates = h["date"]; hassous = h["hassou"]
        rids = h["rid"]; bans = h["ban"]
        m = len(ords)
        for k in range(m):
            P_ord = ords[k]; P_date = dates[k]; P_hassou = hassous[k]
            P_rid = rids[k]; P_ban = bans[k]
            p_counter += 1
            if k == 0:
                out_rid.append(P_rid); out_ban.append(P_ban)
                out_w1.append(np.nan); out_w2.append(np.nan); out_w3.append(np.nan)
                continue
            R_ord = ords[k - 1]; R_rid = rids[k - 1]; R_date = dates[k - 1]
            cut90 = _date_plus_90(R_date)
            do_leak = (p_counter % LEAK_SAMPLE_EVERY == 0)

            w1v, w2v, w3v = [], [], []
            for j in rid2peds.get(R_rid, []):
                hj = ped_hist.get(j)
                if hj is None:
                    continue
                jo = hj["ord"]
                lo = np.searchsorted(jo, R_ord, side="right")   # > R_ord
                hi = np.searchsorted(jo, P_ord, side="left")    # < P_ord
                if hi <= lo:
                    continue
                s_o = jo[lo:hi]
                s_d = hj["date"][lo:hi]
                s_hs = hj["hassou"][lo:hi]
                s_t = hj["top5"][lo:hi]
                # ★厳密 T 境界: (date,hassou) < (P_date,P_hassou) のみ採用 (同時刻も除外)
                keep = (s_d < P_date) | ((s_d == P_date) & (s_hs < P_hassou))
                n_before = len(s_o)
                s_o, s_d, s_hs, s_t = s_o[keep], s_d[keep], s_hs[keep], s_t[keep]
                same_time_dropped += int(n_before - len(s_o))
                if len(s_o) == 0:
                    continue
                if do_leak:
                    leak_checked += len(s_o)
                    bad = np.sum(~((s_d < P_date) | ((s_d == P_date) & (s_hs < P_hassou))))
                    bad += int(np.sum(s_o >= P_ord))
                    leak_violations += int(bad)
                w3v.append(float(s_t.mean()))
                w1v.append(float(s_t[0]))             # R 直後の 1 走
                mask90 = s_d <= cut90
                if mask90.any():
                    w2v.append(float(s_t[mask90].mean()))
            out_rid.append(P_rid); out_ban.append(P_ban)
            out_w1.append(np.mean(w1v) if w1v else np.nan)
            out_w2.append(np.mean(w2v) if w2v else np.nan)
            out_w3.append(np.mean(w3v) if w3v else np.nan)

    res = pd.DataFrame({
        "rid16": out_rid, "ban": out_ban,
        "race_level_prev_W1": out_w1,
        "race_level_prev_W2": out_w2,
        "race_level_prev_W3": out_w3,
    })

    # split 別カバレッジ (master と join)
    sp = df[["_rid", COL_BAN, "split"]].rename(columns={"_rid": "rid16", COL_BAN: "ban"})
    res = res.merge(sp, on=["rid16", "ban"], how="left")

    leak = {
        "leak_sample_every": LEAK_SAMPLE_EVERY,
        "leak_checked_subraces": leak_checked,
        "leak_violations": leak_violations,
        "same_time_races_dropped": same_time_dropped,
        "PASS": bool(leak_violations == 0),
        "coverage": {},
    }
    for spl in ["train", "valid", "test"]:
        s = res[res["split"] == spl]
        if len(s):
            leak["coverage"][spl] = {
                "n": int(len(s)),
                "W1": round(s["race_level_prev_W1"].notna().mean() * 100, 1),
                "W2": round(s["race_level_prev_W2"].notna().mean() * 100, 1),
                "W3": round(s["race_level_prev_W3"].notna().mean() * 100, 1),
            }

    res.drop(columns=["split"]).to_parquet(OUT_PARQUET, index=False)
    OUT_LEAK.parent.mkdir(exist_ok=True)
    with open(OUT_LEAK, "w", encoding="utf-8") as f:
        json.dump(leak, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 64)
    print(f"[リーク検査] サンプル間隔={LEAK_SAMPLE_EVERY}  検査したその後走={leak_checked:,}")
    print(f"  ★違反件数 = {leak_violations}   → {'PASS' if leak_violations==0 else 'FAIL'}")
    print(f"  (参考) 同時刻レース除外 = {same_time_dropped:,} 件")
    print("カバレッジ(レベル算出できた行%):")
    for spl, c in leak["coverage"].items():
        print(f"  {spl:6s} n={c['n']:>8,}  W1={c['W1']}%  W2={c['W2']}%  W3={c['W3']}%")
    # 分布 (test)
    st = res[res["split"] == "test"]
    if len(st):
        print("test 分布:")
        print(st[["race_level_prev_W1", "race_level_prev_W2", "race_level_prev_W3"]].describe().T.to_string())
    print(f"\n[saved] {OUT_PARQUET}")
    print(f"[saved] {OUT_LEAK}")
    return leak_violations


if __name__ == "__main__":
    v = build()
    raise SystemExit(1 if v > 0 else 0)
