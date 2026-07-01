# -*- coding: utf-8 -*-
"""
daybias_within_card_test.py — 当日 as-of within-card トラックバイアスにエッジがあるか検定
=========================================================================================
gutchi が手でやる「その日の前レースが前残り→後のレースで逃げ先行を厚く」を特徴量化し検定。

★leak修正(v2): 外部ファイルの`脚質`は「そのレースで実際にどう走ったか」=事後ラベル。
  ターゲットレースの馬を realized 脚質で選ぶのは leak。
  → 各馬の **過去走の脚質から期待前型(past_front_rate=過去で逃げ/先行だった率)** を serve-safe に作り、
    ターゲット前馬の選択はこれで行う(build_site.classify_style と同発想)。
  as-of bias 自体は「完走済みの前レース」由来なので realized 脚質でOK(未来leak無し)。

as-of front-bias index (race r): 同日同場同surfaceで r より前に発走したレースのみ、
  各先行レースの (前が3着内割合 − 前が出走割合) を平均。正=その日その馬場は前残り。

関門: ①本物か(index高帯ほど期待前馬の複勝率↑/単調か) ②馬場越えか(going固定で残るか)
       ③市場越えか(前残り日の期待前馬の複勝ROIが低帯/break-evenを上回るか=既に売られてないか)
"""
import sys, io
from pathlib import Path
import numpy as np, pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
BASE = Path(r"E:\PyCaLiAI"); sys.path.insert(0, str(BASE))
from backtest_pl_ev import load_payouts

MASTER = BASE / "data/master_v2_20130105-20251228.csv"
EXT = "E:/競馬過去走データ/raw_data/kekka_2010_2025_fix_raceid_v2__keyed.csv"
FRONT = {"逃げ", "先行"}
MIN_PRIOR_RACES = 2          # 当日その surface での発走済み最低本数
MIN_HORSE_HIST = 2           # 期待前型を作る最低過去走数

print("[load] master ...")
m = pd.read_csv(MASTER, encoding="utf-8-sig", low_memory=False, usecols=[
    "レースID(新/馬番無)", "血統登録番号", "場所", "芝・ダ", "発走時刻", "着順",
    "馬番", "馬場状態", "split", "fukusho_flag"])
m["rid"] = pd.to_numeric(m["レースID(新/馬番無)"], errors="coerce").astype("Int64").astype(str)
m["date"] = m["rid"].str[:8]
m["ban"] = pd.to_numeric(m["馬番"], errors="coerce")
m["jyun"] = pd.to_numeric(m["着順"], errors="coerce")
m["t"] = pd.to_numeric(m["発走時刻"].astype(str).str.replace(":", "", regex=False), errors="coerce")
m = m.dropna(subset=["ban", "jyun", "t", "血統登録番号"])

print("[load] 脚質 (外部) ...")
e = pd.read_csv(EXT, encoding="utf-8-sig", low_memory=False, usecols=["race_id", "馬番", "脚質"])
e["rid"] = pd.to_numeric(e["race_id"], errors="coerce").astype("Int64").astype(str)
e["ban"] = pd.to_numeric(e["馬番"], errors="coerce")
m = m.merge(e[["rid", "ban", "脚質"]], on=["rid", "ban"], how="inner").dropna(subset=["脚質"])
m["is_front_real"] = m["脚質"].isin(FRONT).astype(int)   # ★realized(このレース)= 事後。bias集計のみ可
m["top3"] = (m["jyun"] <= 3).astype(int)

# ---- serve-safe 期待前型: 各馬の過去走で逃げ/先行だった率 (current除外) ----
m = m.sort_values(["血統登録番号", "date", "rid"]).reset_index(drop=True)
gh = m.groupby("血統登録番号")
m["_cum"] = gh["is_front_real"].cumsum()
m["_cnt"] = gh.cumcount()
m["_sum_prev"] = m["_cum"] - m["is_front_real"]
m["past_front_rate"] = np.where(m["_cnt"] >= 1, m["_sum_prev"] / m["_cnt"].replace(0, np.nan), np.nan)
m["exp_front"] = np.where(m["_cnt"] >= MIN_HORSE_HIST, (m["past_front_rate"] >= 0.5), np.nan)
ov = m.dropna(subset=["exp_front"])
print(f"  期待前型 定義馬={len(ov):,}  realized一致率(exp vs real)="
      f"{(ov['exp_front'].astype(bool)==ov['is_front_real'].astype(bool)).mean():.3f} "
      f"(=1なら完全leak、0.6-0.8なら事前推定として妥当)")

# ---- 各レース(完走済み)の前残り度: realized脚質でOK ----
g = m.groupby("rid").agg(
    date=("date", "first"), place=("場所", "first"), surf=("芝・ダ", "first"),
    t=("t", "first"), going=("馬場状態", "first"), split=("split", "first"),
    n=("ban", "size"), front_n=("is_front_real", "sum")).reset_index()
top3f = m[m["top3"] == 1].groupby("rid")["is_front_real"].sum().rename("front_in_top3")
g = g.merge(top3f, on="rid", how="left").fillna({"front_in_top3": 0})
g["front_overperf"] = g["front_in_top3"] / 3.0 - g["front_n"] / g["n"]

# ---- as-of within-card index (同日同場同surface・時刻が前のみ) ----
g = g.sort_values(["date", "place", "surf", "t"]).reset_index(drop=True)
idx = np.full(len(g), np.nan)
for _, sub in g.groupby(["date", "place", "surf"], sort=False):
    op = sub["front_overperf"].values; ii = sub.index.values
    cs = np.cumsum(op)
    for k in range(len(op)):
        if k >= MIN_PRIOR_RACES:
            idx[ii[k]] = cs[k - 1] / k
g["asof_bias"] = idx
g = g.dropna(subset=["asof_bias"])

# ---- 馬単位に index を戻し、期待前馬だけ抽出 ----
mm = m.merge(g[["rid", "asof_bias"]], on="rid", how="inner")
front = mm[mm["exp_front"] == True].copy()

print("[load] payouts ...")
pays = load_payouts()
def fuku_payout(rid, ban):
    po = pays.get(str(rid)[:16])
    if not po: return None
    for bn, key in [(po["win"], "fuku_win"), (po["plc"], "fuku_plc"), (po["sho"], "fuku_sho")]:
        if int(bn) == int(ban):
            v = po.get(key); return float(v) if v == v and v else 0.0
    return 0.0
front["fpay"] = [fuku_payout(r, b) for r, b in zip(front["rid"], front["ban"])]

def tercile(s):
    q1, q2 = s["asof_bias"].quantile([1/3, 2/3])
    return np.where(s["asof_bias"] <= q1, "低(後残り寄)", np.where(s["asof_bias"] <= q2, "中", "高(前残り)"))

def report(df, label):
    print(f"\n{'='*78}\n{label}  (期待前馬 n={len(df):,})\n{'='*78}")
    df = df.copy(); df["band"] = tercile(df)
    base = df["top3"].mean() * 100
    print(f"期待前型馬 全体の複勝率(top3)={base:.1f}%  realized前率={df['is_front_real'].mean():.2f}")
    print("--- ①本物か: as-of前残り帯 × 期待前馬 複勝率/複勝ROI ---")
    print(f"{'帯':<12}{'前馬n':>7}{'複勝率%':>9}{'vs全体pt':>9}{'複勝ROI%':>10}")
    for b in ["低(後残り寄)", "中", "高(前残り)"]:
        s = df[df["band"] == b]; sp = s[s["fpay"].notna()]
        roi = sp["fpay"].sum() / (len(sp) * 100) * 100 if len(sp) else float("nan")
        print(f"{b:<12}{len(s):>7}{s['top3'].mean()*100:>9.1f}{s['top3'].mean()*100-base:>+9.1f}{roi:>10.1f}")
    print("--- ②馬場越えか: going固定で 高帯−低帯 複勝率差 ---")
    print(f"{'馬場':<8}{'低帯%':>8}{'高帯%':>8}{'高-低pt':>9}{'n低/n高':>13}")
    for go, s in df.groupby("馬場状態"):
        lo = s[s["band"] == "低(後残り寄)"]; hi = s[s["band"] == "高(前残り)"]
        if len(lo) < 100 or len(hi) < 100: continue
        print(f"{str(go):<8}{lo['top3'].mean()*100:>8.1f}{hi['top3'].mean()*100:>8.1f}"
              f"{hi['top3'].mean()*100-lo['top3'].mean()*100:>+9.1f}{f'{len(lo)}/{len(hi)}':>13}")

for sp in ["valid", "test"]:
    ff = front[front["split"] == sp]
    if len(ff) > 50: report(ff, f"split={sp}")
print("\n[読み] leak修正後も ①高帯ほど複勝率↑かつ複勝ROIが低帯/100%超を保つか。"
      "②馬場固定で高-低差が残るか。valid/test一致なら本物。realized一致率が0.6-0.8なら事前推定として妥当。")
