# -*- coding: utf-8 -*-
"""
crossday_bias_test.py — クロスデー(土→日)トラックバイアスにエッジがあるか検定
==============================================================================
仮説: 土曜の「完走済み・全レース確定」のトラックバイアスを観測し、
      翌日(日)の同開催・同コースのレースを買う。
within-card 同日版(daybias_within_card_test.py で死亡, 78.4%)とは別物:
  前日カードは完全に終わっているので翌日賭けでは leak-safe。

leak規律は daybias_within_card_test を継承:
  - 外部脚質はrealized(事後)。bias集計(完走済みの前日)には使ってOK。
  - ターゲット(日曜)の馬選択は serve-safe な exp_front(各馬の過去走脚質, current除外)で。
  - MIN_HORSE_HIST>=2。

検定:
  1) PERSISTENCE  土front-bias vs 日front-bias の Spearman/Pearson。+ 枠バイアス持続。
  2) INVERSION    符号反転率。乾燥(cushion↑/含水率↓)で反転率が上がるか。
  3) BETABILITY   土front-biasが上位tercileの日曜=同place+surfで、日曜の期待前馬の複勝ROI。
                  control=土バイアス無関係の同馬の複勝ROI。valid/test方向一致+bootstrap CI。
"""
import sys, io
from pathlib import Path
import numpy as np, pandas as pd
from datetime import datetime, timedelta
from scipy.stats import spearmanr, pearsonr

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
BASE = Path(r"E:\PyCaLiAI"); sys.path.insert(0, str(BASE))
from backtest_pl_ev import load_payouts

MASTER = BASE / "data/master_v2_20130105-20251228.csv"
EXT = "E:/競馬過去走データ/raw_data/kekka_2010_2025_fix_raceid_v2__keyed.csv"
BABA = BASE / "data/baba_feats.parquet"
FRONT = {"逃げ", "先行"}
MIN_HORSE_HIST = 2
MIN_RACES_PER_DAY = 3   # 日ごとのbias安定化に最低本数
MIN_CELL = 100          # 帯ごと最低ベット数ガード
RNG = np.random.default_rng(42)

print("[load] master ...")
m = pd.read_csv(MASTER, encoding="utf-8-sig", low_memory=False, usecols=[
    "レースID(新/馬番無)", "血統登録番号", "場所", "芝・ダ", "発走時刻", "着順",
    "馬番", "馬場状態", "split", "fukusho_flag"])
m["rid"] = pd.to_numeric(m["レースID(新/馬番無)"], errors="coerce").astype("Int64").astype(str)
m["date"] = m["rid"].str[:8]
m["ban"] = pd.to_numeric(m["馬番"], errors="coerce")
m["jyun"] = pd.to_numeric(m["着順"], errors="coerce")
m = m.dropna(subset=["ban", "jyun", "血統登録番号"])
m = m[m["date"].str.len() == 8]

print("[load] 脚質 (外部) ...")
e = pd.read_csv(EXT, encoding="utf-8-sig", low_memory=False, usecols=["race_id", "馬番", "脚質"])
e["rid"] = pd.to_numeric(e["race_id"], errors="coerce").astype("Int64").astype(str)
e["ban"] = pd.to_numeric(e["馬番"], errors="coerce")
m = m.merge(e[["rid", "ban", "脚質"]], on=["rid", "ban"], how="inner").dropna(subset=["脚質"])
m["is_front_real"] = m["脚質"].isin(FRONT).astype(int)
m["top3"] = (m["jyun"] <= 3).astype(int)

# ---- serve-safe 期待前型 (current除外) ----
m = m.sort_values(["血統登録番号", "date", "rid"]).reset_index(drop=True)
gh = m.groupby("血統登録番号")
m["_cum"] = gh["is_front_real"].cumsum()
m["_cnt"] = gh.cumcount()
m["_sum_prev"] = m["_cum"] - m["is_front_real"]
m["past_front_rate"] = np.where(m["_cnt"] >= 1, m["_sum_prev"] / m["_cnt"].replace(0, np.nan), np.nan)
m["exp_front"] = np.where(m["_cnt"] >= MIN_HORSE_HIST, (m["past_front_rate"] >= 0.5), np.nan)
ov = m.dropna(subset=["exp_front"])
print(f"  期待前型 定義馬={len(ov):,}  realized一致率="
      f"{(ov['exp_front'].astype(bool)==ov['is_front_real'].astype(bool)).mean():.3f}")

# ---- 各レース(完走済み)の前残り度 ----
g = m.groupby("rid").agg(
    date=("date", "first"), place=("場所", "first"), surf=("芝・ダ", "first"),
    going=("馬場状態", "first"), split=("split", "first"),
    n=("ban", "size"), front_n=("is_front_real", "sum")).reset_index()
top3f = m[m["top3"] == 1].groupby("rid")["is_front_real"].sum().rename("front_in_top3")
g = g.merge(top3f, on="rid", how="left").fillna({"front_in_top3": 0})
g["front_overperf"] = g["front_in_top3"] / 3.0 - g["front_n"] / g["n"]
# winner normalized draw for draw-bias
win = m[m["jyun"] == 1].groupby("rid").agg(win_ban=("ban", "first")).reset_index()
g = g.merge(win, on="rid", how="left")
g["win_ndraw"] = (g["win_ban"] - 1) / (g["n"] - 1).replace(0, np.nan)

# ---- (place, surf, date) 日次 front-bias ----
day = g.groupby(["place", "surf", "date"]).agg(
    front_bias=("front_overperf", "mean"),
    win_ndraw=("win_ndraw", "mean"),
    nrace=("rid", "size"),
    split=("split", lambda s: s.mode().iat[0] if len(s) else None)).reset_index()
day = day[day["nrace"] >= MIN_RACES_PER_DAY].copy()
print(f"  日次セル (place,surf,date) n={len(day):,}  (>= {MIN_RACES_PER_DAY}本/日)")

# ---- consecutive-day same-meeting pairing (date_next = date_prev + 1) ----
def to_dt(s): return datetime.strptime(s, "%Y%m%d")
day["dt"] = day["date"].map(to_dt)
day_idx = {(r.place, r.surf, r.date): r for r in day.itertuples()}
pairs = []
for r in day.itertuples():
    nxt = (r.dt + timedelta(days=1)).strftime("%Y%m%d")
    key = (r.place, r.surf, nxt)
    if key in day_idx:
        nx = day_idx[key]
        pairs.append(dict(
            place=r.place, surf=r.surf, date_prev=r.date, date_next=nxt,
            sat_bias=r.front_bias, sun_bias=nx.front_bias,
            sat_ndraw=r.win_ndraw, sun_ndraw=nx.win_ndraw,
            sun_split=nx.split))
P = pd.DataFrame(pairs)
print(f"  consecutive-day pairs n={len(P):,}")

# =====================================================================
# 1) PERSISTENCE
# =====================================================================
print("\n" + "=" * 78 + "\n1) PERSISTENCE  土→日 front-bias 持続\n" + "=" * 78)
pp = P.dropna(subset=["sat_bias", "sun_bias"])
sp_r, sp_p = spearmanr(pp["sat_bias"], pp["sun_bias"])
pe_r, pe_p = pearsonr(pp["sat_bias"], pp["sun_bias"])
print(f"n_pairs={len(pp):,}")
print(f"Spearman(front Sat,Sun) = {sp_r:+.4f}  (p={sp_p:.3g})")
print(f"Pearson (front Sat,Sun) = {pe_r:+.4f}  (p={pe_p:.3g})")
dd = P.dropna(subset=["sat_ndraw", "sun_ndraw"])
dr_r, dr_p = pearsonr(dd["sat_ndraw"], dd["sun_ndraw"])
ds_r, _ = spearmanr(dd["sat_ndraw"], dd["sun_ndraw"])
print(f"winner-draw persistence: Pearson={dr_r:+.4f} (p={dr_p:.3g}) Spearman={ds_r:+.4f} n={len(dd):,}")

# =====================================================================
# 2) INVERSION + moisture-drying caveat
# =====================================================================
print("\n" + "=" * 78 + "\n2) INVERSION  符号反転率 と 乾燥caveat\n" + "=" * 78)
pp = pp.copy()
pp["flip"] = (np.sign(pp["sat_bias"]) != np.sign(pp["sun_bias"])).astype(int)
flip_rate = pp["flip"].mean()
print(f"sign-flip rate (all pairs) = {flip_rate:.3f}  n={len(pp):,}")

# merge moisture: cushion + surface-specific 含水率 (gp=ゴール前)
b = pd.read_parquet(BABA)
b.columns = ["place", "date", "cushion", "shiba_gp", "shiba_4c", "dirt_gp", "dirt_4c"]
b["date"] = b["date"].astype(str)
# moisture metric per (place,date,surf): cushion for 芝(higher=drier/firmer), 含水率 gp
def moist_merge(P, suffix, datecol):
    mm = P.merge(b, left_on=["place", datecol], right_on=["place", "date"], how="left",
                 suffixes=("", "_b")).drop(columns=["date"], errors="ignore")
    return mm
PP = pp.copy()
PP = PP.merge(b.add_suffix("_sat"), left_on=["place", "date_prev"],
              right_on=["place_sat", "date_sat"], how="left")
PP = PP.merge(b.add_suffix("_sun"), left_on=["place", "date_next"],
              right_on=["place_sun", "date_sun"], how="left")
# surface-specific 含水率: 芝 uses shiba_gp, ダ uses dirt_gp. cushion only meaningful for 芝.
PP["moist_sat"] = np.where(PP["surf"] == "芝", PP["shiba_gp_sat"], PP["dirt_gp_sat"])
PP["moist_sun"] = np.where(PP["surf"] == "芝", PP["shiba_gp_sun"], PP["dirt_gp_sun"])
PP["d_moist"] = PP["moist_sun"] - PP["moist_sat"]      # 含水率: 負=乾く
PP["d_cushion"] = PP["cushion_sun"] - PP["cushion_sat"]  # cushion: 正=乾く/締まる
mq = PP.dropna(subset=["d_moist"])
if len(mq) >= 40:
    dried = mq[mq["d_moist"] < 0]      # 含水率下降=乾燥
    wetter = mq[mq["d_moist"] >= 0]    # 維持/湿る
    print(f"含水率(gp) 取得 pairs n={len(mq):,}")
    print(f"  乾く(d含水率<0) : flip={dried['flip'].mean():.3f}  n={len(dried):,}")
    print(f"  維持/湿(>=0)    : flip={wetter['flip'].mean():.3f}  n={len(wetter):,}")
    print(f"  Δflip(乾-湿)    = {dried['flip'].mean()-wetter['flip'].mean():+.3f}")
    if len(mq) > 30:
        cr, cp = spearmanr(mq["d_moist"], mq["flip"])
        print(f"  Spearman(Δ含水率, flip)={cr:+.4f} (p={cp:.3g})  [負なら乾くほどflip増]")
else:
    print(f"含水率 取得 pairs 不足 n={len(mq)}")
cq = PP.dropna(subset=["d_cushion"])
if len(cq) >= 40:
    dryc = cq[cq["d_cushion"] > 0]    # cushion上昇=乾く/締まる
    wetc = cq[cq["d_cushion"] <= 0]
    moist_summary = (f"cushion: dried(Δcushion>0) flip={dryc['flip'].mean():.3f} n={len(dryc)} "
                     f"vs stay/wet flip={wetc['flip'].mean():.3f} n={len(wetc)} "
                     f"Δ={dryc['flip'].mean()-wetc['flip'].mean():+.3f}; "
                     f"含水率: dried flip={dried['flip'].mean():.3f} vs wet {wetter['flip'].mean():.3f} "
                     f"Δ={dried['flip'].mean()-wetter['flip'].mean():+.3f} (n_moist={len(mq)})")
    print(f"cushion 取得 pairs n={len(cq):,}")
    print(f"  乾く(Δcushion>0): flip={dryc['flip'].mean():.3f}  n={len(dryc):,}")
    print(f"  維持/湿(<=0)    : flip={wetc['flip'].mean():.3f}  n={len(wetc):,}")
    print(f"  Δflip(乾-湿)    = {dryc['flip'].mean()-wetc['flip'].mean():+.3f}")
else:
    moist_summary = (f"含水率: dried flip={dried['flip'].mean():.3f} vs wet {wetter['flip'].mean():.3f} "
                     f"Δ={dried['flip'].mean()-wetter['flip'].mean():+.3f} (n_moist={len(mq)}); cushion不足") if len(mq)>=40 else "moisture取得不足"
    print("cushion 取得 pairs 不足")

# =====================================================================
# 3) BETABILITY
# =====================================================================
print("\n" + "=" * 78 + "\n3) BETABILITY  土front-bias上位tercileの日曜=期待前馬の複勝ROI\n" + "=" * 78)
# Saturday front-bias tercile cut (leak-safe: from Saturday only, by surf, ALL pairs)
# tercile thresholds computed across pair set's Saturday biases
pp2 = P.dropna(subset=["sat_bias"]).copy()
# per-surface tercile to avoid surf confound; top tercile = "土前残り"
def front_flag(df):
    out = np.zeros(len(df), dtype=bool)
    for sf, sub in df.groupby("surf"):
        q23 = sub["sat_bias"].quantile(2/3)
        out[df.index.isin(sub.index)] = df.loc[df.index.isin(sub.index), "sat_bias"] >= q23
    return out
pp2["sat_front_top"] = front_flag(pp2)
# map each Sunday (place,surf,date_next) -> was Saturday front-biased top tercile?
sun_front = {(r.place, r.surf, r.date_next): bool(r.sat_front_top) for r in pp2.itertuples()}

print("[load] payouts ...")
pays = load_payouts()
def fuku_payout(rid, ban):
    po = pays.get(str(rid)[:16])
    if not po: return None
    for bn, key in [(po["win"], "fuku_win"), (po["plc"], "fuku_plc"), (po["sho"], "fuku_sho")]:
        if int(bn) == int(ban):
            v = po.get(key); return float(v) if v == v and v else 0.0
    return 0.0

# Sunday expected-front horses
mm = m.merge(g[["rid", "date", "place", "surf"]].rename(columns={"date":"rdate"}), on="rid", how="left")
front = m[m["exp_front"] == True].copy()
front = front.merge(g[["rid", "place", "surf", "date"]].rename(columns={"date":"rdate"}),
                    on="rid", how="left")
front["sun_key"] = list(zip(front["place"], front["surf"], front["rdate"]))
front["sat_was_front"] = front["sun_key"].map(sun_front)   # NaN if this date not a Sunday-of-pair
front = front.dropna(subset=["sat_was_front"])
front["sat_was_front"] = front["sat_was_front"].astype(bool)
front["fpay"] = [fuku_payout(r, b) for r, b in zip(front["rid"], front["ban"])]
front = front.dropna(subset=["fpay"])
print(f"  日曜(pair翌日) 期待前馬 n={len(front):,}  "
      f"うち土前残り={int(front['sat_was_front'].sum()):,}")

def roi_of(df):
    if len(df) == 0: return float("nan")
    return df["fpay"].sum() / (len(df) * 100) * 100

def boot_ci(df, B=2000):
    # resample over races (rid) to respect intra-race correlation
    rids = df["rid"].unique()
    grp = {r: sub["fpay"].values for r, sub in df.groupby("rid")}
    rois = []
    for _ in range(B):
        samp = RNG.choice(rids, size=len(rids), replace=True)
        tot = 0.0; cnt = 0
        for r in samp:
            v = grp[r]; tot += v.sum(); cnt += len(v)
        rois.append(tot / (cnt * 100) * 100 if cnt else np.nan)
    rois = np.array(rois)
    return np.nanpercentile(rois, 2.5), np.nanpercentile(rois, 97.5)

results = {}
print(f"\n{'split':<8}{'cond':<18}{'n_bet':>8}{'n_race':>8}{'複勝率%':>9}{'ROI%':>9}")
for sp in ["valid", "test"]:
    sub = front[front["split"] == sp]
    test_grp = sub[sub["sat_was_front"]]
    base_grp = sub
    for label, dd2 in [("土前残りのみ(test)", test_grp), ("baseline(全期待前)", base_grp)]:
        roi = roi_of(dd2); hit = dd2["top3"].mean()*100 if len(dd2) else float("nan")
        nr = dd2["rid"].nunique()
        print(f"{sp:<8}{label:<18}{len(dd2):>8}{nr:>8}{hit:>9.1f}{roi:>9.1f}")
    results[sp] = dict(test=test_grp, base=base_grp)

valid_roi = roi_of(results["valid"]["test"])
test_roi = roi_of(results["test"]["test"])
base_test_roi = roi_of(results["test"]["base"])
base_valid_roi = roi_of(results["valid"]["base"])

# guard tiny cells
def guarded(roi, df): return roi if len(df) >= MIN_CELL else float("nan")
valid_roi_g = guarded(valid_roi, results["valid"]["test"])
test_roi_g = guarded(test_roi, results["test"]["test"])

ci_lo, ci_hi = (float("nan"), float("nan"))
if len(results["test"]["test"]) >= MIN_CELL:
    ci_lo, ci_hi = boot_ci(results["test"]["test"])

# direction consistency: both above baseline (lift same sign)
v_lift = valid_roi - base_valid_roi
t_lift = test_roi - base_test_roi
consistent = bool(np.sign(v_lift) == np.sign(t_lift) and not np.isnan(v_lift) and not np.isnan(t_lift))

print("\n--- SUMMARY ---")
print(f"valid_roi(土前残り日曜)={valid_roi:.1f}%  baseline_valid={base_valid_roi:.1f}%  lift={v_lift:+.1f}pt")
print(f"test_roi (土前残り日曜)={test_roi:.1f}%  baseline_test ={base_test_roi:.1f}%  lift={t_lift:+.1f}pt")
print(f"test CI95=[{ci_lo:.1f}, {ci_hi:.1f}]  n_bets_test={len(results['test']['test'])}")
print(f"valid/test direction consistent (both vs baseline)={consistent}")
print(f"bars: beat baseline? test {test_roi:.1f} vs {base_test_roi:.1f} | "
      f"beat within-card-dead 78.4? {test_roi>78.4} | beat ceiling ~88? {test_roi>88}")

# verdict
def verdict():
    if np.isnan(test_roi) or len(results['test']['test']) < MIN_CELL:
        return "INCONCLUSIVE: test cell < MIN_CELL guard"
    beats_base = test_roi > base_test_roi + 0.5
    beats_dead = test_roi > 78.4
    beats_ceil = test_roi > 88.0
    if beats_base and beats_dead and beats_ceil and consistent and ci_lo > 100:
        return "LIVE: clears all bars with CI>100"
    if beats_base and beats_dead and beats_ceil and consistent:
        return f"PROMISING but CI includes <=100 ({ci_lo:.0f}-{ci_hi:.0f}): not profitable, priced"
    if beats_base and consistent:
        return "WEAK/PRICED: small lift, below ceiling or CI weak — priced/variance lever"
    return "DEAD: no robust lift over baseline / inconsistent / below dead-result"
VERD = verdict()
print(f"VERDICT: {VERD}")

# emit machine-readable
import json
out = dict(
    n_pairs=int(len(pp)),
    spearman_front=float(sp_r), pearson_front=float(pe_r),
    draw_pearson=float(dr_r), draw_spearman=float(ds_r),
    flip_rate=float(flip_rate),
    moist_summary=moist_summary,
    valid_roi=float(valid_roi) if not np.isnan(valid_roi) else None,
    test_roi=float(test_roi) if not np.isnan(test_roi) else None,
    base_test_roi=float(base_test_roi) if not np.isnan(base_test_roi) else None,
    base_valid_roi=float(base_valid_roi) if not np.isnan(base_valid_roi) else None,
    ci_lo=float(ci_lo) if not np.isnan(ci_lo) else None,
    ci_hi=float(ci_hi) if not np.isnan(ci_hi) else None,
    n_bets_test=int(len(results['test']['test'])),
    consistent=consistent, verdict=VERD)
print("JSON " + json.dumps(out, ensure_ascii=False))
