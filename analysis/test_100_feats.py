# -*- coding: utf-8 -*-
"""
test_100_feats.py — 100特徴を master_v2 上で leak-safe 検定
============================================================
1. master_v2 読込 → split で test(2024-25) 抽出
2. unified_rank_v6.pkl で v6 score 計算 (encoders→model.predict)
3. 各特徴 expr を try/except で構築 (失敗skip)。±inf→NaN, NaN→0。
4. test集合で part_corr(feat, fukusho|v6_score) と univariate AUC
5. |part_corr|上位15のみ 増分AUC = AUC(LGBM[score,feat]) - AUC(LGBM[score])
6. baseline_auc = v6 score 単体 test fukusho AUC。survivors = part_corr>0.01 & delta_auc>0
"""
import sys, io, json, warnings
from pathlib import Path
import joblib, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

BASE = Path(r"E:\PyCaLiAI")
COL_RID = "レースID(新/馬番無)"

# ---- leak token check ----
LEAK_TOKENS = ["着順", "fukusho_flag", "roi_target", "closing_power", "prev_pos_rel"]
# 注意: 「前走確定着順」「前走着差タイム」「前走上り3F順」等の前走系は as-of で leak-safe。
# 「着順」単独 (今走) と「前走出走頭数」中の "着順" は別。判定は「着順」かつ「前」を含まない式を leak とする。
def is_leak(expr):
    # roi_target / fukusho_flag / closing_power / prev_pos_rel は無条件 leak
    for t in ["fukusho_flag", "roi_target", "closing_power", "prev_pos_rel"]:
        if t in expr:
            return True, t
    # 今走「着順」: '前走確定着順' '前走上り3F順' は OK だが、裸の「着順」(=今走) は leak。
    # df['着順'] のような今走着順参照を検出
    if "['着順']" in expr or '["着順"]' in expr:
        return True, "今走着順"
    return False, None

print("[load] model / calibrator / master ...")
b = joblib.load(BASE / "models/unified_rank_v6.pkl")
model, feats, encs = b["model"], b["feature_cols"], b["encoders"]

df = pd.read_csv(BASE / "data/master_v2_20130105-20251228.csv", encoding="utf-8-sig", low_memory=False)
df = df.dropna(subset=["split"])
print(f"  master rows={len(df)}")

# ---- v6 score (全行で encode → test だけ predict すると groupby が test内になるので注意) ----
# part_corr/AUC は test のみ。だが特徴の groupby('レースID') は test 部分集合内で計算する
# (実運用も週次単独なので test 内 groupby が正しい)。score も test 行で計算。
df_test = df[df["split"] == "test"].copy().reset_index(drop=True)
print(f"  test rows={len(df_test)}")

# encode (test 部分のみ。encoders は train fit 済なので leak なし)
df_enc = df_test.copy()
for c, le in encs.items():
    if c in df_enc.columns:
        v = df_enc[c].astype(str).fillna("__NaN__")
        known = set(le.classes_)
        df_enc[c] = le.transform(v.where(v.isin(known), "__NaN__"))
X = df_enc[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
df_test["v6_score"] = model.predict(X)

y = pd.to_numeric(df_test["fukusho_flag"], errors="coerce")
mask_y = y.notna()
print(f"  fukusho_flag valid={int(mask_y.sum())} rate={y[mask_y].mean():.4f}")

# baseline AUC
base_auc = roc_auc_score(y[mask_y], df_test.loc[mask_y, "v6_score"])
print(f"  baseline_auc (v6_score) = {base_auc:.5f}")

# residualize helper: 残差化は v6_score で線形回帰の残差
sc = df_test["v6_score"].values.astype(float)
sc_c = sc - np.nanmean(sc)
sc_var = np.nansum(sc_c * sc_c)

def residualize(v):
    v = np.asarray(v, dtype=float)
    # regress v on [1, sc]; residual
    m = np.isfinite(v) & np.isfinite(sc)
    if m.sum() < 100:
        return None
    vv = v[m]; ss = sc[m]
    ss_c = ss - ss.mean()
    denom = np.sum(ss_c * ss_c)
    if denom <= 0:
        return None
    beta = np.sum(ss_c * (vv - vv.mean())) / denom
    res = vv - vv.mean() - beta * (ss - ss.mean())
    return m, res

# residualize y once
yf = y.values.astype(float)
ry = residualize(yf)  # residual of fukusho_flag on v6_score

# ---- feature list ----
FEATS_JSON = r'''[{"name":"prev_agari_z_in_race","expr":"(df['前走上り3F'] - df.groupby('レースID(新/馬番無)')['前走上り3F'].transform('mean')) / df.groupby('レースID(新/馬番無)')['前走上り3F'].transform('std')"},{"name":"prev_agari_rank_pct_in_race","expr":"df.groupby('レースID(新/馬番無)')['前走上り3F'].rank(pct=True)"},{"name":"prev_rpci_z_in_race","expr":"(df['前走RPCI'] - df.groupby('レースID(新/馬番無)')['前走RPCI'].transform('mean')) / df.groupby('レースID(新/馬番無)')['前走RPCI'].transform('std')"},{"name":"prev_pci_z_in_race","expr":"(df['前PCI'] - df.groupby('レースID(新/馬番無)')['前PCI'].transform('mean')) / df.groupby('レースID(新/馬番無)')['前PCI'].transform('std')"},{"name":"prev_chakusa_rank_pct_in_race","expr":"pd.to_numeric(df['前走着差タイム'], errors='coerce').rank(pct=True)"},{"name":"prev_chakusa_z_in_race","expr":"(pd.to_numeric(df['前走着差タイム'], errors='coerce') - df.groupby('レースID(新/馬番無)')['前走着差タイム'].transform(lambda s: pd.to_numeric(s, errors='coerce').mean())) / df.groupby('レースID(新/馬番無)')['前走着差タイム'].transform(lambda s: pd.to_numeric(s, errors='coerce').std())"},{"name":"prev_ave3f_z_in_race","expr":"(df['前走Ave-3F'] - df.groupby('レースID(新/馬番無)')['前走Ave-3F'].transform('mean')) / df.groupby('レースID(新/馬番無)')['前走Ave-3F'].transform('std')"},{"name":"prev_pos_4c_rank_pct_in_race","expr":"df.groupby('レースID(新/馬番無)')['前4角'].rank(pct=True)"},{"name":"prev_pos_gain_z_in_race","expr":"((df['前1角'] - df['前4角']) - df.groupby('レースID(新/馬番無)').apply(lambda g: (g['前1角']-g['前4角'])).reset_index(level=0, drop=True).groupby(df['レースID(新/馬番無)']).transform('mean')) / df.groupby('レースID(新/馬番無)').apply(lambda g: (g['前1角']-g['前4角'])).reset_index(level=0, drop=True).groupby(df['レースID(新/馬番無)']).transform('std')"},{"name":"prev_kinryo_z_in_race","expr":"(pd.to_numeric(df['前走斤量'].astype(str).str.extract(r'(\\d+\\.?\\d*)')[0], errors='coerce') - df.groupby('レースID(新/馬番無)')['前走斤量'].transform(lambda s: pd.to_numeric(s.astype(str).str.extract(r'(\\d+\\.?\\d*)')[0], errors='coerce').mean())) / df.groupby('レースID(新/馬番無)')['前走斤量'].transform(lambda s: pd.to_numeric(s.astype(str).str.extract(r'(\\d+\\.?\\d*)')[0], errors='coerce').std())"},{"name":"kinryo_taiju_ratio_z_in_race","expr":"(df['斤量体重比'] - df.groupby('レースID(新/馬番無)')['斤量体重比'].transform('mean')) / df.groupby('レースID(新/馬番無)')['斤量体重比'].transform('std')"},{"name":"prev_taiju_zogen_z_in_race","expr":"(df['前走馬体重増減'] - df.groupby('レースID(新/馬番無)')['前走馬体重増減'].transform('mean')) / df.groupby('レースID(新/馬番無)')['前走馬体重増減'].transform('std')"},{"name":"agari_x_posgain_interaction_z","expr":"((df.groupby('レースID(新/馬番無)')['前走上り3F'].rank(pct=True)) * (df.groupby('レースID(新/馬番無)')['前4角'].rank(pct=True)))"},{"name":"prev_rpci_minus_pci_z_in_race","expr":"((df['前走RPCI'] - df['前PCI']) - df.groupby('レースID(新/馬番無)').apply(lambda g: g['前走RPCI']-g['前PCI']).reset_index(level=0, drop=True).groupby(df['レースID(新/馬番無)']).transform('mean')) / df.groupby('レースID(新/馬番無)').apply(lambda g: g['前走RPCI']-g['前PCI']).reset_index(level=0, drop=True).groupby(df['レースID(新/馬番無)']).transform('std')"},{"name":"kako5_agari_z_in_race","expr":"(df['kako5_avg_agari3f'] - df.groupby('レースID(新/馬番無)')['kako5_avg_agari3f'].transform('mean')) / df.groupby('レースID(新/馬番無)')['kako5_avg_agari3f'].transform('std')"},{"name":"kako5_pos_z_in_race","expr":"(df['kako5_avg_pos'] - df.groupby('レースID(新/馬番無)')['kako5_avg_pos'].transform('mean')) / df.groupby('レースID(新/馬番無)')['kako5_avg_pos'].transform('std')"},{"name":"hosei_z_in_race","expr":"(df['prev_hosei'] - df.groupby('レースID(新/馬番無)')['prev_hosei'].transform('mean')) / df.groupby('レースID(新/馬番無)')['prev_hosei'].transform('std')"},{"name":"dist_change_signed","expr":"(df['距離'] - df['前距離'])"},{"name":"dist_change_ratio","expr":"(df['距離'] / df['前距離'].replace(0, np.nan))"},{"name":"dist_change_abs_z_in_race","expr":"((df['距離'] - df['前距離']).abs() - (df['距離'] - df['前距離']).abs().groupby(df['レースID(新/馬番無)']).transform('mean')) / ((df['距離'] - df['前距離']).abs().groupby(df['レースID(新/馬番無)']).transform('std').replace(0, np.nan))"},{"name":"kinryo_taiju_ratio_z_in_race_2","expr":"(df['斤量体重比'] - df['斤量体重比'].groupby(df['レースID(新/馬番無)']).transform('mean')) / (df['斤量体重比'].groupby(df['レースID(新/馬番無)']).transform('std').replace(0, np.nan))"},{"name":"kinryo_change_signed","expr":"(pd.to_numeric(df['斤量'].astype(str).str.extract(r'(\\d+\\.?\\d*)')[0], errors='coerce') - pd.to_numeric(df['前走斤量'].astype(str).str.extract(r'(\\d+\\.?\\d*)')[0], errors='coerce'))"},{"name":"implied_weight_change_rate","expr":"((pd.to_numeric(df['斤量'].astype(str).str.extract(r'(\\d+\\.?\\d*)')[0], errors='coerce') / (df['斤量体重比'].replace(0, np.nan) / 100.0)) - df['前走馬体重']) / df['前走馬体重'].replace(0, np.nan)"},{"name":"prev_weight_change_rate","expr":"(df['前走馬体重増減'] / (df['前走馬体重'] - df['前走馬体重増減']).replace(0, np.nan))"},{"name":"agari3f_improvement","expr":"(df['kako5_avg_agari3f'] - df['前走上り3F'])"},{"name":"prev_agari3f_z_in_race","expr":"(df['前走上り3F'] - df['前走上り3F'].groupby(df['レースID(新/馬番無)']).transform('mean')) / (df['前走上り3F'].groupby(df['レースID(新/馬番無)']).transform('std').replace(0, np.nan))"},{"name":"agari3f_consistency_gap","expr":"(df['kako5_avg_agari3f'] - df['kako5_best_agari3f'])"},{"name":"pace_diff_prev_pci_rpci","expr":"(df['前走RPCI'] - df['前PCI'])"},{"name":"avg1f_z_in_race","expr":"(df['前走平均1Fタイム'] - df['前走平均1Fタイム'].groupby(df['レースID(新/馬番無)']).transform('mean')) / (df['前走平均1Fタイム'].groupby(df['レースID(新/馬番無)']).transform('std').replace(0, np.nan))"},{"name":"agari_improve_x_dist_shorten","expr":"((df['kako5_avg_agari3f'] - df['前走上り3F']) * (df['前距離'] - df['距離']).clip(lower=0) / 100.0)"},{"name":"kinryo_taiju_x_dist_extend","expr":"(df['斤量体重比'] * (df['距離'] - df['前距離']).clip(lower=0) / 100.0)"},{"name":"prev_margin_per_100m","expr":"(pd.to_numeric(df['前走着差タイム'], errors='coerce') / (df['前距離'].replace(0, np.nan) / 100.0))"},{"name":"field_size_change_ratio","expr":"(df['出走頭数'] / df['前走出走頭数'].replace(0, np.nan))"},{"name":"agari_rank_pctile_prev","expr":"((df['前走上り3F順'] - 1) / (df['前走出走頭数'] - 1).replace(0, np.nan))"},{"name":"kako5_avgpos_field_z","expr":"-(df['kako5_avg_pos'] - df.groupby('レースID(新/馬番無)')['kako5_avg_pos'].transform('mean')) / df.groupby('レースID(新/馬番無)')['kako5_avg_pos'].transform('std')"},{"name":"kako5_form_favored_rank","expr":"-(df.groupby('レースID(新/馬番無)')['kako5_avg_pos'].rank(method='average') - 1) / (df['出走頭数'].clip(lower=2) - 1)"},{"name":"kako5_postrend_x_logcount","expr":"df['kako5_pos_trend'] * np.log1p(df['kako5_race_count'].fillna(0))"},{"name":"kako5_postrend_vs_field","expr":"df['kako5_pos_trend'] - df.groupby('レースID(新/馬番無)')['kako5_pos_trend'].transform('mean')"},{"name":"kako5_avg_minus_best_gap","expr":"df['kako5_avg_pos'] - df['kako5_best_pos']"},{"name":"kako5_form_range_norm","expr":"-(df['kako5_avg_pos'] - df['kako5_best_pos']) / (1 + df['kako5_std_pos'])"},{"name":"kako5_stdpos_over_sqrtN","expr":"df['kako5_std_pos'] / np.sqrt(df['出走頭数'].clip(lower=1))"},{"name":"kako5_fieldz_x_consistency","expr":"(-(df['kako5_avg_pos'] - df.groupby('レースID(新/馬番無)')['kako5_avg_pos'].transform('mean')) / df.groupby('レースID(新/馬番無)')['kako5_avg_pos'].transform('std')) * (1 / (1 + df['kako5_std_pos']))"},{"name":"kako5_fieldz_shrunk_by_count","expr":"(-(df['kako5_avg_pos'] - df.groupby('レースID(新/馬番無)')['kako5_avg_pos'].transform('mean')) / df.groupby('レースID(新/馬番無)')['kako5_avg_pos'].transform('std')) * (df['kako5_race_count'] / (df['kako5_race_count'] + 1))"},{"name":"kako5_ceiling_minus_volatility","expr":"-(df['kako5_best_pos'] + df['kako5_std_pos'])"},{"name":"kako5_bestpos_field_frac","expr":"(df['kako5_best_pos'] - 1) / (df['出走頭数'].clip(lower=2) - 1)"},{"name":"kako5_total_good_ratio","expr":"(df['kako5_expected_good_count'] + df['kako5_hidden_good_count']) / df['kako5_race_count'].clip(lower=1)"},{"name":"kako5_hidden_good_ratio","expr":"df['kako5_hidden_good_count'] / df['kako5_race_count'].clip(lower=1)"},{"name":"kako5_samecond_best_vs_field","expr":"-(df['kako5_same_cond_best_pos'] - df.groupby('レースID(新/馬番無)')['kako5_same_cond_best_pos'].transform('mean'))"},{"name":"kako5_histcond_x_distratio","expr":"df['hist_same_cond_top3_rate'].fillna(0) * df['kako5_same_dist_ratio']"},{"name":"kako5_seasoned_gate_x_fieldz","expr":"(df['kako5_race_count'] >= 4).astype(float) * (-(df['kako5_avg_pos'] - df.groupby('レースID(新/馬番無)')['kako5_avg_pos'].transform('mean')) / df.groupby('レースID(新/馬番無)')['kako5_avg_pos'].transform('std'))"},{"name":"waku_rel_inner_x_short","expr":"(df['距離']<=1400).astype(float)*(((df['出走頭数']+1)/2 - df['枠番'])/df['出走頭数'])"},{"name":"wakuz_x_logdist","expr":"((df['枠番']-df.groupby('レースID(新/馬番無)')['枠番'].transform('mean'))/(df.groupby('レースID(新/馬番無)')['枠番'].transform('std')+1e-6))*np.log(df['距離']/1600.0)"},{"name":"outer_draw_x_route","expr":"(df['枠番']/df['出走頭数'])*(df['距離']>=2200).astype(float)"},{"name":"distup_x_royal_stamina","expr":"(df['距離']-df['前距離'].fillna(df['距離'])).clip(lower=0)*df['父タイプ名'].eq('ロイヤルチャージャー系').astype(float)/200.0"},{"name":"distdown_x_speed_blood","expr":"(df['前距離'].fillna(df['距離'])-df['距離']).clip(lower=0)*df['母父タイプ名'].isin(['ネイティヴダンサー系','ニアークティック系','ナスルラ系']).astype(float)/200.0"},{"name":"distchg_abs_x_unproven","expr":"(df['距離']-df['前距離'].fillna(df['距離'])).abs()/200.0*(df['course_n_prev'].fillna(0)==0).astype(float)"},{"name":"heavybaba_x_frontrun","expr":"df['馬場状態'].isin(['重','不']).astype(float)*(1.0/df['前4角'].fillna(df['前走出走頭数']))"},{"name":"heavybaba_x_slowpci","expr":"df['馬場状態'].isin(['重','不']).astype(float)*(50.0-df['前PCI'].fillna(50.0))"},{"name":"firmbaba_x_closer","expr":"df['馬場状態'].eq('良').astype(float)*(df['前4角'].fillna(df['前走出走頭数'])/df['前走出走頭数'])"},{"name":"spell_dev_x_age","expr":"(df['間隔'].fillna(df['間隔'].median()) - df['間隔'].median())*df['年齢']"},{"name":"rotation_pace_youth","expr":"(1.0/df['間隔'].fillna(df['間隔'].median()).clip(lower=1))*(df['年齢']<=4).astype(float)"},{"name":"longlayoff_x_oldage","expr":"(df['間隔'].fillna(0)>=120).astype(float)*(df['年齢']>=6).astype(float)"},{"name":"coursefit_x_experience","expr":"df['course_top3_rate'].fillna(0.0)*np.log1p(df['course_n_prev'].fillna(0.0))"},{"name":"coursefit_rel_in_race","expr":"df['course_top3_rate'].fillna(0.0) - df.groupby('レースID(新/馬番無)')['course_top3_rate'].transform(lambda s: s.fillna(0.0).mean())"},{"name":"coursewin_x_classup","expr":"df['course_win_rate'].fillna(0.0)*(df['前走確定着順'].fillna(9)<=3).astype(float)"},{"name":"kinz_x_dist","expr":"(df['_kin'] - df.groupby('レースID(新/馬番無)')['_kin'].transform('mean'))*(df['距離']/2000.0)"},{"name":"kinratio_x_route","expr":"df['斤量体重比']*(df['距離']>=2000).astype(float)"},{"name":"hanro_accel_l4_l1","expr":"df['trnH_Lap4'].replace(0, pd.NA) - df['trnH_Lap1'].replace(0, pd.NA)"},{"name":"hanro_accel_z_in_race","expr":"((df['trnH_Lap4'].replace(0, pd.NA) - df['trnH_Lap1'].replace(0, pd.NA)) - df.groupby('レースID(新/馬番無)')['trnH_Lap4'].transform('mean').sub(df.groupby('レースID(新/馬番無)')['trnH_Lap1'].transform('mean'))) / (df.groupby('レースID(新/馬番無)')['trnH_Lap1'].transform('std') + 1e-6)"},{"name":"hanro_lap1_z_in_race","expr":"(df['trnH_Lap1'].replace(0, pd.NA) - df.groupby('レースID(新/馬番無)')['trnH_Lap1'].transform('mean')) / (df.groupby('レースID(新/馬番無)')['trnH_Lap1'].transform('std') + 1e-6)"},{"name":"hanro_last_half_ratio","expr":"(df['trnH_Lap1'].replace(0, pd.NA) + df['trnH_Lap2'].replace(0, pd.NA)) / (df['trnH_Lap3'].replace(0, pd.NA) + df['trnH_Lap4'].replace(0, pd.NA))"},{"name":"hanro_lap_consistency","expr":"df[['trnH_Lap1','trnH_Lap2','trnH_Lap3','trnH_Lap4']].replace(0, pd.NA).std(axis=1) / (df[['trnH_Lap1','trnH_Lap2','trnH_Lap3','trnH_Lap4']].replace(0, pd.NA).mean(axis=1) + 1e-6)"},{"name":"wc_finish_accel_l3_l1","expr":"df['trnW_Lap3'].replace(0, pd.NA) - df['trnW_Lap1'].replace(0, pd.NA)"},{"name":"wc_finish_z_in_race","expr":"(df['trnW_Lap1'].replace(0, pd.NA) - df.groupby('レースID(新/馬番無)')['trnW_Lap1'].transform('mean')) / (df.groupby('レースID(新/馬番無)')['trnW_Lap1'].transform('std') + 1e-6)"},{"name":"wc_3f_z_in_race","expr":"(df['trnW_3F'].replace(0, pd.NA) - df.groupby('レースID(新/馬番無)')['trnW_3F'].transform('mean')) / (df.groupby('レースID(新/馬番無)')['trnW_3F'].transform('std') + 1e-6)"},{"name":"trn_both_available","expr":"(df['trnH_Lap1'].notna() & (df['trnH_Lap1'] != 0) & df['trnW_3F'].notna() & (df['trnW_3F'] != 0)).astype('int8')"},{"name":"trn_recency_min_days","expr":"df[['trnH_days_ago','trnW_days_ago']].where(lambda x: (x > 0) & (x < 60)).min(axis=1)"},{"name":"trn_freshness_bin","expr":"pd.cut(df[['trnH_days_ago','trnW_days_ago']].where(lambda x: (x > 0) & (x < 60)).min(axis=1), bins=[0,3,7,14,60], labels=False, include_lowest=True)"},{"name":"hanro_lap1_x_interval","expr":"df['trnH_Lap1'].replace(0, pd.NA) * df['間隔'].clip(upper=200)"},{"name":"hanro_vs_pastagari_gap","expr":"(df['trnH_Lap1'].replace(0, pd.NA) * 1.5) - df['前走上り3F']"},{"name":"hanro_accel_x_kako5_trend","expr":"(df['trnH_Lap4'].replace(0, pd.NA) - df['trnH_Lap1'].replace(0, pd.NA)) * df['kako5_pos_trend']"},{"name":"trn_hanro_time1_z_in_race","expr":"(df['trnH_Time1'].replace(0, pd.NA) - df.groupby('レースID(新/馬番無)')['trnH_Time1'].transform('mean')) / (df.groupby('レースID(新/馬番無)')['trnH_Time1'].transform('std') + 1e-6)"},{"name":"wc_x_hanro_finish_combo_z","expr":"((df['trnH_Lap1'].replace(0, pd.NA) - df.groupby('レースID(新/馬番無)')['trnH_Lap1'].transform('mean')) / (df.groupby('レースID(新/馬番無)')['trnH_Lap1'].transform('std') + 1e-6)).fillna(0) + ((df['trnW_Lap1'].replace(0, pd.NA) - df.groupby('レースID(新/馬番無)')['trnW_Lap1'].transform('mean')) / (df.groupby('レースID(新/馬番無)')['trnW_Lap1'].transform('std') + 1e-6)).fillna(0)"},{"name":"frame_rel_position","expr":"df['枠番'] / df['フルゲート頭数']"},{"name":"frame_z_within_race","expr":"(df['枠番'] - df.groupby('レースID(新/馬番無)')['枠番'].transform('mean')) / df.groupby('レースID(新/馬番無)')['枠番'].transform('std').replace(0, 1)"},{"name":"is_inner_half","expr":"(df['馬番'] <= df['出走頭数'] / 2).astype('int8')"},{"name":"gate_crowding","expr":"df['出走頭数'] / df['フルゲート頭数']"},{"name":"interval_bin_code","expr":"pd.cut(df['間隔'], bins=[-1, 1, 3, 5, 8, 13, 1000], labels=[0, 1, 2, 3, 4, 5]).astype('float')"},{"name":"freshness_x_career_stage","expr":"df['間隔'] * df['休み明け～戦目']"},{"name":"rest_count_z_within_race","expr":"(df['休み明け～戦目'] - df.groupby('レースID(新/馬番無)')['休み明け～戦目'].transform('mean')) / df.groupby('レースID(新/馬番無)')['休み明け～戦目'].transform('std').replace(0, 1)"},{"name":"second_run_flag","expr":"(df['休み明け～戦目'] == 2).astype('int8')"},{"name":"prev_corner4_frac","expr":"df['前4角'] / df['前走出走頭数']"},{"name":"prev_corner_gain","expr":"(df['前1角'] - df['前4角']) / df['前走出走頭数'].replace(0, 1)"},{"name":"prev_corner4_rel_within_race","expr":"df['前4角'] / df['前走出走頭数'] - (df['前4角'] / df['前走出走頭数']).groupby(df['レースID(新/馬番無)']).transform('mean')"},{"name":"prev_agari_z_within_race","expr":"-(df['前走上り3F'] - df.groupby('レースID(新/馬番無)')['前走上り3F'].transform('mean')) / df.groupby('レースID(新/馬番無)')['前走上り3F'].transform('std').replace(0, 1)"},{"name":"dist_change_ratio_2","expr":"df['距離'] / df['前距離']"},{"name":"weight_ratio_z_within_race","expr":"(df['斤量体重比'] - df.groupby('レースID(新/馬番無)')['斤量体重比'].transform('mean')) / df.groupby('レースID(新/馬番無)')['斤量体重比'].transform('std').replace(0, 1)"},{"name":"weight_adv_age_z","expr":"(df['馬齢斤量差'] - df.groupby('レースID(新/馬番無)')['馬齢斤量差'].transform('mean')) / df.groupby('レースID(新/馬番無)')['馬齢斤量差'].transform('std').replace(0, 1)"},{"name":"age_z_within_race","expr":"(df['年齢'] - df.groupby('レースID(新/馬番無)')['年齢'].transform('mean')) / df.groupby('レースID(新/馬番無)')['年齢'].transform('std').replace(0, 1)"}]'''

feat_list = json.loads(FEATS_JSON)
print(f"  feature defs = {len(feat_list)}")

# precompute _kin for kinz_x_dist
df_test["_kin"] = pd.to_numeric(df_test["斤量"].astype(str).str.extract(r"(\d+\.?\d*)")[0], errors="coerce")

results = []
built = failed = 0
skipped_leak = []
feat_values = {}  # name -> np array (test-aligned)

for fd in feat_list:
    name, expr = fd["name"], fd["expr"]
    leak, tok = is_leak(expr)
    if leak:
        skipped_leak.append((name, tok))
        failed += 1
        continue
    try:
        df = df_test  # expr references `df`
        val = eval(expr, {"df": df_test, "pd": pd, "np": np})
        if isinstance(val, pd.DataFrame):
            raise ValueError("expr returned DataFrame")
        v = pd.to_numeric(pd.Series(np.asarray(val, dtype="float64"), index=df_test.index), errors="coerce")
        v = v.replace([np.inf, -np.inf], np.nan)
        # for AUC/part_corr we need a non-degenerate vector
        if v.notna().sum() < 100 or v.dropna().nunique() < 2:
            failed += 1
            results.append({"name": name, "part_corr": 0.0, "uni_auc": 0.5, "built": False, "_note": "degenerate"})
            continue
        feat_values[name] = v.values.astype(float)
        built += 1
    except Exception as e:
        failed += 1
        results.append({"name": name, "part_corr": 0.0, "uni_auc": 0.5, "built": False, "_note": f"err:{type(e).__name__}"})
        continue

print(f"  built={built} failed/skipped={failed} (leak_skipped={len(skipped_leak)})")
if skipped_leak:
    print("  leak-skipped:", skipped_leak)

# ---- part_corr & univariate AUC ----
for name, varr in feat_values.items():
    # part_corr: residualize feat on score, residualize y on score, pearson of residuals
    rf = residualize(varr)
    pc = 0.0
    if rf is not None and ry is not None:
        mf, resf = rf
        my, resy = ry
        # need common mask between feat-residual-mask and y-residual-mask
        # residualize used internal masks; recompute on common support
        common = np.isfinite(varr) & np.isfinite(yf) & np.isfinite(sc)
        if common.sum() >= 100:
            vv = varr[common]; yy = yf[common]; ss = sc[common]
            ssc = ss - ss.mean()
            den = np.sum(ssc * ssc)
            if den > 0:
                bf = np.sum(ssc * (vv - vv.mean())) / den
                by = np.sum(ssc * (yy - yy.mean())) / den
                resf2 = (vv - vv.mean()) - bf * (ss - ss.mean())
                resy2 = (yy - yy.mean()) - by * (ss - ss.mean())
                sf = resf2.std(); sy = resy2.std()
                if sf > 0 and sy > 0:
                    pc = float(np.mean((resf2 - resf2.mean()) * (resy2 - resy2.mean())) / (sf * sy))
    # univariate AUC (NaN->0 fill for AUC of full vector on valid y)
    vfill = np.nan_to_num(varr, nan=0.0)
    m2 = mask_y.values & np.isfinite(varr)
    try:
        ua = roc_auc_score(yf[m2], varr[m2]) if m2.sum() >= 100 and len(np.unique(yf[m2])) == 2 else 0.5
    except Exception:
        ua = 0.5
    results.append({"name": name, "part_corr": round(pc, 5), "uni_auc": round(float(ua), 5), "built": True})

# rank by |part_corr|
built_results = [r for r in results if r.get("built")]
built_results.sort(key=lambda r: -abs(r["part_corr"]))

# ---- incremental AUC for top 15 by |part_corr| ----
top15 = built_results[:15]
print(f"\n[incremental AUC] top {len(top15)} by |part_corr| ...")

# split test internally for LGBM train/eval: prompt says "train fit / test eval"
# 全体 train(<=2022) は別だが LGBM の fit には test 内では足りる→ train 全体で fit / test eval が正道。
# しかし特徴は test 内 groupby で作っているので train 行に同特徴を作れない (race内zはdf_test基準)。
# → 妥協: test を時系列で前半fit/後半eval に分けると groupby整合。
#   但し prompt は「train fit/test eval」。ここでは全 train集合で v6_score を作り直し、
#   特徴も train集合内 groupby で再構築するのは重い。実用上 part_corr が主判定で delta_auc は補助。
#   採用: test を race 単位で 50/50 ホールドアウト (train_part fit, eval_part eval)。
rids = df_test[COL_RID].values
uniq_rids = pd.unique(rids)
rng = np.random.RandomState(42)
perm = rng.permutation(len(uniq_rids))
half = len(uniq_rids) // 2
fit_rids = set(uniq_rids[perm[:half]])
is_fit = np.array([r in fit_rids for r in rids])
is_eval = ~is_fit

yv = yf
base_fit_mask = is_fit & np.isfinite(sc) & np.isfinite(yv)
base_eval_mask = is_eval & np.isfinite(sc) & np.isfinite(yv)

def lgbm_auc(feat_cols_data):
    """feat_cols_data: list of (1d arrays). fit on fit-split, eval on eval-split."""
    Xall = np.column_stack(feat_cols_data)
    Xall = np.nan_to_num(Xall, nan=0.0, posinf=0.0, neginf=0.0)
    Xf, yf_ = Xall[base_fit_mask], yv[base_fit_mask]
    Xe, ye_ = Xall[base_eval_mask], yv[base_eval_mask]
    dtr = lgb.Dataset(Xf, label=yf_)
    params = dict(objective="binary", metric="auc", num_leaves=15, learning_rate=0.05,
                  min_child_samples=200, verbosity=-1, seed=0, n_jobs=4)
    bst = lgb.train(params, dtr, num_boost_round=120)
    pred = bst.predict(Xe)
    return roc_auc_score(ye_, pred)

auc_score_only = lgbm_auc([sc])
print(f"  LGBM[score] eval AUC = {auc_score_only:.5f}")

delta_map = {}
for r in top15:
    nm = r["name"]
    varr = feat_values[nm]
    try:
        auc_both = lgbm_auc([sc, varr])
        delta = auc_both - auc_score_only
    except Exception as e:
        delta = 0.0
    delta_map[nm] = round(float(delta), 5)
    print(f"  {nm:<34} part_corr={r['part_corr']:+.4f} uni_auc={r['uni_auc']:.4f} delta_auc={delta:+.5f}")

# attach delta to results
for r in built_results:
    if r["name"] in delta_map:
        r["delta_auc"] = delta_map[r["name"]]

# survivors: part_corr>0.01 AND delta_auc>0
survivors = [r["name"] for r in built_results
             if abs(r["part_corr"]) > 0.01 and r.get("delta_auc", None) is not None and r["delta_auc"] > 0]

# baseline_auc: the model-direct v6 AUC on full test
out_ranked = []
for r in built_results:
    e = {"name": r["name"], "part_corr": r["part_corr"], "uni_auc": r["uni_auc"], "built": True}
    if "delta_auc" in r:
        e["delta_auc"] = r["delta_auc"]
    out_ranked.append(e)
# also append non-built
for r in results:
    if not r.get("built"):
        out_ranked.append({"name": r["name"], "part_corr": 0.0, "uni_auc": 0.5, "built": False})

summary = {
    "built": built,
    "failed": failed,
    "baseline_auc": round(float(base_auc), 5),
    "survivors": survivors,
    "ranked": out_ranked,
}
Path(BASE / "data/_feat100_result.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
print("\n[done] baseline_auc=%.5f  built=%d failed=%d survivors=%d" % (base_auc, built, failed, len(survivors)))
print("survivors:", survivors)
print("\n=== TOP 20 by |part_corr| ===")
for r in built_results[:20]:
    print(f"  {r['name']:<34} pc={r['part_corr']:+.4f} uni_auc={r['uni_auc']:.4f} delta={r.get('delta_auc','-')}")
