"""
race_level_feats_v2.py — レースレベル(集約=「好走の強さ」重み付け版, リークフリー)
=================================================================================
第二実験。前回(v1)は集約=「その後5着内(0/1)率の平均」(粗い二値)。
今回は集約=「その後どれだけ"強く"好走したか」の連続スコアの平均に精緻化。
変えるのは集約定義のみ。撃ち方(1パス・反復なし)とリーク構造は前回完全踏襲。

その後走 X の "強さ" strength_X (高いほど強い好走) を 3 成分の加重平均で:
  place_s  = clip((6 - 着順)/5, 0, 1)         重み0.3   (1着=1.0, 5着=0.2, 6着以下≈0)
  margin_s = clip((3 - 着差秒)/3, 0, 1)       重み0.3   (僅差/勝ち=1, 3秒差以上=0)  ※着順と+0.64相関
  hosei_s  = clip((補正タイム-60)/45, 0, 1)   重み0.4   (大きいほど速い/強い)        ※着順と-0.74相関(最強)
  欠損成分は除外して利用可能成分で重み再正規化 (着順は常に有効)。
  ※走破タイム絶対値(master欠損)・賞金(master列なし)は使用不可のため不採用 (報告明記)。

着差・補正タイムは master_v2 に「今走」値が無いため chain reconstruction:
  走 X の (着差, 補正) = 同馬の次走行の (前走着差タイム, prev_hosei)。最新走は欠損→着順成分のみ。

レースRのレベル = R出走"全頭"(敗戦馬含む)の strength を、Rより後・T未満に終了した走で集約(平均)。
窓 W2(R後90日以内) / W3(T未満全走)。W1は前回最弱のため省略。

★リーク防止: 前回 race_level_feats.py と同一構造。(date,hassou,rid)辞書式順序で
  (date,hassou) が P より厳密に小さい走のみ採用 (同時刻除外)。検査で違反0をassert。

出力: data/race_level_feats_v2.parquet (rid16, ban, race_level_str_W2, race_level_str_W3)
       reports/race_level_feats_v2_leakcheck.json
実行: PYTHONUTF8=1 python race_level_feats_v2.py
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
OUT_PARQUET = BASE / "data/race_level_feats_v2.parquet"
OUT_LEAK = BASE / "reports/race_level_feats_v2_leakcheck.json"

COL_RID = "レースID(新/馬番無)"
COL_BAN = "馬番"
COL_PED = "血統登録番号"
COL_DATE = "日付"
COL_HASSOU = "発走時刻"
COL_JYUN = "着順"
COL_MARGIN = "前走着差タイム"   # chain: 走X+1 のこの値 = 走X の着差
COL_HOSEI = "prev_hosei"        # chain: 走X+1 のこの値 = 走X の補正タイム

LEAK_SAMPLE_EVERY = 50
W_PLACE, W_MARGIN, W_HOSEI = 0.3, 0.3, 0.4


_cut90: dict = {}
def _date_plus_90(d: int) -> int:
    if d in _cut90:
        return _cut90[d]
    try:
        t = pd.Timestamp(str(int(d)))
        v = int((t + pd.Timedelta(days=90)).strftime("%Y%m%d"))
    except Exception:
        v = int(d) + 90
    _cut90[d] = v
    return v


def _strength(pos, margin, hosei):
    """3成分(着順/着差/補正)の加重平均。欠損成分は除外し重み再正規化。ベクトル化。"""
    pos = np.asarray(pos, float); margin = np.asarray(margin, float); hosei = np.asarray(hosei, float)
    place_s = np.clip((6.0 - pos) / 5.0, 0, 1)              # 着順は常に有効前提
    margin_s = np.clip((3.0 - margin) / 3.0, 0, 1)
    hosei_s = np.clip((hosei - 60.0) / 45.0, 0, 1)
    comps = np.stack([place_s, margin_s, hosei_s], axis=1)  # (n,3)
    W = np.array([W_PLACE, W_MARGIN, W_HOSEI])
    mask = ~np.isnan(comps)
    num = np.nansum(comps * W, axis=1)
    den = (mask * W).sum(axis=1)
    return np.where(den > 0, num / den, np.nan)


def build():
    print(f"[load] {MASTER_CSV.name} (usecols)")
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig",
                     usecols=[COL_RID, COL_BAN, COL_PED, COL_DATE, COL_HASSOU, COL_JYUN,
                              COL_MARGIN, COL_HOSEI, "split"], low_memory=False)
    df[COL_JYUN] = pd.to_numeric(df[COL_JYUN], errors="coerce")
    df = df.dropna(subset=[COL_JYUN, COL_RID, COL_PED, COL_DATE]).copy()
    df[COL_DATE] = pd.to_numeric(df[COL_DATE], errors="coerce").astype("int64")
    df["_hassou"] = pd.to_numeric(df[COL_HASSOU], errors="coerce").fillna(0).astype("int64")
    df[COL_BAN] = pd.to_numeric(df[COL_BAN], errors="coerce").fillna(0).astype(int)
    df[COL_MARGIN] = pd.to_numeric(df[COL_MARGIN], errors="coerce")
    df[COL_HOSEI] = pd.to_numeric(df[COL_HOSEI], errors="coerce")
    df["_rid"] = df[COL_RID].astype(str)
    print(f"  rows={len(df):,}  races={df['_rid'].nunique():,}  horses={df[COL_PED].nunique():,}")

    rinfo = df.groupby("_rid").agg(date=(COL_DATE, "first"), hassou=("_hassou", "first")).reset_index()
    rinfo = rinfo.sort_values(["date", "hassou", "_rid"]).reset_index(drop=True)
    rinfo["race_ord"] = np.arange(len(rinfo), dtype=np.int64)
    rid2ord = dict(zip(rinfo["_rid"], rinfo["race_ord"]))
    df["race_ord"] = df["_rid"].map(rid2ord).astype(np.int64)

    rid2peds = {rid: g[COL_PED].values for rid, g in df.groupby("_rid", sort=False)}

    # ped 走歴 (race_ord 昇順)。chain: 各走の着差/補正 = 次走行の前走フィールド (group内 shift(-1))
    df_s = df.sort_values([COL_PED, "race_ord"])
    ped_hist = {}
    for ped, g in df_s.groupby(COL_PED, sort=False):
        pos = g[COL_JYUN].values.astype(float)
        margin_chain = g[COL_MARGIN].shift(-1).values   # 走X の着差 = 次走の前走着差
        hosei_chain = g[COL_HOSEI].shift(-1).values
        strg = _strength(pos, margin_chain, hosei_chain)
        ped_hist[ped] = {
            "ord": g["race_ord"].values.astype(np.int64),
            "date": g[COL_DATE].values.astype(np.int64),
            "hassou": g["_hassou"].values.astype(np.int64),
            "strg": strg.astype(np.float64),
            "rid": g["_rid"].values,
            "ban": g[COL_BAN].values.astype(int),
        }

    print("[compute] 前走レベル(強さ重み付け W2/W3) をリークフリー集計 ...")
    out_rid, out_ban, out_w2, out_w3 = [], [], [], []
    leak_checked = leak_violations = same_time_dropped = p_counter = 0

    for ped, h in ped_hist.items():
        ords = h["ord"]; dates = h["date"]; hassous = h["hassou"]
        rids = h["rid"]; bans = h["ban"]
        m = len(ords)
        for k in range(m):
            P_ord = ords[k]; P_date = dates[k]; P_hassou = hassous[k]
            P_rid = rids[k]; P_ban = bans[k]
            p_counter += 1
            if k == 0:
                out_rid.append(P_rid); out_ban.append(P_ban); out_w2.append(np.nan); out_w3.append(np.nan)
                continue
            R_ord = ords[k - 1]; R_rid = rids[k - 1]; R_date = dates[k - 1]
            cut90 = _date_plus_90(R_date)
            do_leak = (p_counter % LEAK_SAMPLE_EVERY == 0)
            w2v, w3v = [], []
            for j in rid2peds.get(R_rid, []):
                hj = ped_hist.get(j)
                if hj is None:
                    continue
                jo = hj["ord"]
                lo = np.searchsorted(jo, R_ord, side="right")
                hi = np.searchsorted(jo, P_ord, side="left")
                if hi <= lo:
                    continue
                s_o = jo[lo:hi]; s_d = hj["date"][lo:hi]; s_hs = hj["hassou"][lo:hi]; s_s = hj["strg"][lo:hi]
                keep = (s_d < P_date) | ((s_d == P_date) & (s_hs < P_hassou))
                nb = len(s_o)
                s_o, s_d, s_s = s_o[keep], s_d[keep], s_s[keep]
                same_time_dropped += int(nb - len(s_o))
                if len(s_o) == 0:
                    continue
                if do_leak:
                    leak_checked += len(s_o)
                    leak_violations += int(np.sum(s_o >= P_ord))
                # strength の NaN (両補助成分欠損かつ着順も無い等) は除外
                valid = ~np.isnan(s_s)
                if valid.any():
                    w3v.append(float(np.nanmean(s_s)))
                    mask90 = (s_d <= cut90) & valid
                    if mask90.any():
                        w2v.append(float(np.nanmean(s_s[(s_d <= cut90)])))
            out_rid.append(P_rid); out_ban.append(P_ban)
            out_w2.append(np.mean(w2v) if w2v else np.nan)
            out_w3.append(np.mean(w3v) if w3v else np.nan)

    res = pd.DataFrame({"rid16": out_rid, "ban": out_ban,
                        "race_level_str_W2": out_w2, "race_level_str_W3": out_w3})
    sp = df[["_rid", COL_BAN, "split"]].rename(columns={"_rid": "rid16", COL_BAN: "ban"})
    res = res.merge(sp, on=["rid16", "ban"], how="left")

    leak = {"leak_sample_every": LEAK_SAMPLE_EVERY, "leak_checked_subraces": leak_checked,
            "leak_violations": leak_violations, "same_time_races_dropped": same_time_dropped,
            "PASS": bool(leak_violations == 0),
            "strength_weights": {"place": W_PLACE, "margin": W_MARGIN, "hosei": W_HOSEI},
            "unused_signals": ["走破タイム絶対値(master欠損)", "賞金(master列なし)"], "coverage": {}}
    for spl in ["train", "valid", "test"]:
        s = res[res["split"] == spl]
        if len(s):
            leak["coverage"][spl] = {"n": int(len(s)),
                                     "W2": round(s["race_level_str_W2"].notna().mean() * 100, 1),
                                     "W3": round(s["race_level_str_W3"].notna().mean() * 100, 1)}

    res.drop(columns=["split"]).to_parquet(OUT_PARQUET, index=False)
    OUT_LEAK.parent.mkdir(exist_ok=True)
    with open(OUT_LEAK, "w", encoding="utf-8") as f:
        json.dump(leak, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 64)
    print(f"[リーク検査] サンプル間隔={LEAK_SAMPLE_EVERY}  検査その後走={leak_checked:,}")
    print(f"  ★違反件数 = {leak_violations}   → {'PASS' if leak_violations==0 else 'FAIL'}")
    print(f"  (参考) 同時刻レース除外 = {same_time_dropped:,}")
    print(f"  strength 重み: place={W_PLACE} margin={W_MARGIN} hosei={W_HOSEI}")
    print("カバレッジ:")
    for spl, c in leak["coverage"].items():
        print(f"  {spl:6s} n={c['n']:>8,}  W2={c['W2']}%  W3={c['W3']}%")
    st = res[res["split"] == "test"]
    if len(st):
        print("test 分布:")
        print(st[["race_level_str_W2", "race_level_str_W3"]].describe().T.to_string())
    print(f"\n[saved] {OUT_PARQUET}")
    return leak_violations


if __name__ == "__main__":
    v = build()
    raise SystemExit(1 if v > 0 else 0)
