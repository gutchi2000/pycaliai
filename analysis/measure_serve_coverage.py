# -*- coding: utf-8 -*-
"""
measure_serve_coverage.py
=========================
serve 経路(parse_csv + _SERVE_RENAME)が、v6の各モデル特徴を実際どれだけ埋められるかを
直近の週次CSVで実測。「生きてる/死んでる」をファミリー別に出し、カナリア用の
ベースライン(data/serve_feature_baseline.json)を生成する。
"""
import sys, io, json, os
from pathlib import Path
import numpy as np, pandas as pd, joblib

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
BASE = Path(r"E:\PyCaLiAI"); sys.path.insert(0, str(BASE))
from weekly_parse import parse_csv

# export_weekly_marks の _SERVE_RENAME と同一
_SERVE_RENAME = {
    "R": "Ｒ",
    "前走補正": "prev_hosei", "前走補9": "prev_hosei9",
    "trn_hanro_4f": "trnH_Time1", "trn_hanro_3f": "trnH_Time2",
    "trn_hanro_2f": "trnH_Time3", "trn_hanro_1f": "trnH_Time4",
    "trn_hanro_lap1": "trnH_Lap1", "trn_hanro_lap2": "trnH_Lap2",
    "trn_hanro_lap3": "trnH_Lap3", "trn_hanro_lap4": "trnH_Lap4",
    "trn_hanro_days": "trnH_days_ago",
    "trn_wc_5f": "trnW_5F", "trn_wc_4f": "trnW_4F", "trn_wc_3f": "trnW_3F",
    "trn_wc_lap1": "trnW_Lap1", "trn_wc_lap2": "trnW_Lap2",
    "trn_wc_lap3": "trnW_Lap3", "trn_wc_days": "trnW_days_ago",
}

def fam(c):
    if c.startswith(("trnH_","trnW_")): return "調教"
    if c.startswith("prev_hosei"): return "補正"
    if c.startswith("hist_same_"): return "hist_same"
    if c.startswith("course_"): return "course成績"
    if c.startswith("jockey_n")or c.startswith("jockey_win")or c.startswith("jockey_top"): return "jockey成績(馬×騎手)"
    if c.startswith("kako5"): return "kako5"
    if c in ("騎手コード","調教師コード"): return "コード(cat)"
    if c.startswith(("jockey_fuku","trainer_fuku")): return "騎手厩舎30/90"
    return "その他(当日条件/血統等)"

model = joblib.load(BASE/"models/unified_rank_v6.pkl")
feats = model["feature_cols"]
gain = np.asarray(model["model"].feature_importance(importance_type="gain"), dtype=float)
gain_total = float(gain.sum())
gain_pct = {f: (float(v) / gain_total * 100.0 if gain_total else 0.0)
            for f, v in zip(feats, gain)}
# 週の既定は「bundle を実配信した直近 4 開催日」。固定リストだと古い週で baseline を
# 上書きしてしまう (2026-09-27 に 7 月週で再生成され canary 基準が巻き戻った事故)。
import argparse
_ap = argparse.ArgumentParser(description="serve 特徴カバレッジ baseline の再生成")
_ap.add_argument("--weeks", nargs="*", default=None,
                 help="YYYYMMDD... 省略時は reports/cowork_input に bundle がある直近 4 開催日")
_ap.add_argument("--out", default=None,
                 help="出力先 (省略時は本番 data/serve_feature_baseline.json)。診断は別パスへ")
_args = _ap.parse_args()
if _args.weeks:
    WEEKS = list(_args.weeks)
else:
    _b = sorted(p.name[:8] for p in (BASE/"reports"/"cowork_input").glob("2026*_bundle.json")
                if (BASE/"data"/"weekly"/f"{p.name[:8]}.csv").exists())
    WEEKS = _b[-4:][::-1]
OUT_PATH = Path(_args.out) if _args.out else (BASE/"data"/"serve_feature_baseline.json")
print(f"[weeks] {WEEKS}  → {OUT_PATH}")

baseline = {}   # feat -> median coverage across weeks
percol = {f: [] for f in feats}
for wk in WEEKS:
    p = BASE/"data"/"weekly"/f"{wk}.csv"
    if not p.exists(): continue
    df = parse_csv(p)
    df = df.rename(columns={k:v for k,v in _SERVE_RENAME.items() if k in df.columns and v in feats and v not in df.columns})
    # serve 履歴特徴の回収 (export_weekly_marks と同じ経路) を再現してから測る
    try:
        from serve_history_feats import fill_history_features
        from export_weekly_marks import ensure_date_column, apply_hosei_proxy
        df = ensure_date_column(df)
        fill_history_features(df)
        # 本番 (weekly_nicegui.ps1 は CB_HOSEI_PROXY=1) と同じ経路で prev_hosei を埋める。
        # 環境変数が無いと本番より低い充足率を baseline に焼き込むので明示的に出す。
        if os.environ.get("CB_HOSEI_PROXY", "0") == "1":
            filled = apply_hosei_proxy(df, model["encoders"])
            print(f"  [hosei proxy] 適用: " + ", ".join(f"{k}={v*100:.0f}%" for k, v in filled.items()))
        else:
            print("  [hosei proxy] CB_HOSEI_PROXY 未設定 → prev_hosei は本番より低く測られる "
                  "(本番は weekly_nicegui.ps1 が =1 で起動)")
        # serve 特徴復元 (export_weekly_marks と同じ関数・同じ環境変数 CB_SERVE_RECOVER)
        from serve_feature_recovery import recover_serve_features, format_stats
        _rec = recover_serve_features(df, p.stem)
        print("  " + format_stats(_rec))
    except Exception as e:
        print(f"  [warn] serve history fill 失敗 (履歴特徴は0%扱い): {e}")
    # canary と同一定義の実効カバレッジ ("__NaN__"/空文字=欠損, 定数列=0.0) で測る。
    # 定義が食い違うと baseline 比較が無意味になるので必ず export 側の関数を使う。
    from export_weekly_marks import feature_coverage, CONST_OK_COLS
    for f in feats:
        cov = (feature_coverage(df[f], allow_constant=(f in CONST_OK_COLS))
               if f in df.columns else 0.0)
        percol[f].append(cov)
    print(f"[{wk}] parsed {len(df):,}頭 / {df['レースID(新/馬番無)'].nunique()}R")

for f in feats:
    baseline[f] = round(float(np.median(percol[f])) if percol[f] else 0.0, 3)

# ファミリー別サマリ
print(f"\n{'='*70}\nserve カバレッジ (直近{len([w for w in WEEKS if (BASE/'data/weekly'/(w+'.csv')).exists()])}週 中央値)\n{'='*70}")
fdf = pd.DataFrame({"feat": feats, "coverage": [baseline[f] for f in feats],
                    "gain_pct": [gain_pct[f] for f in feats], "fam":[fam(f) for f in feats]})
summ = fdf.groupby("fam").agg(n=("feat","size"), avg=("coverage","mean"),
                              alive=("coverage", lambda s:(s>=0.4).sum()), dead=("coverage", lambda s:(s<0.05).sum()))
summ["avg"]=(summ["avg"]*100).round(0)
print(summ.sort_values("avg").to_string())

dead = fdf[fdf["coverage"]<0.05].sort_values(["fam","feat"])
deadnames = dead.feat.tolist()
regression = fdf[(fdf["coverage"] < 0.40) & (fdf["gain_pct"] > 0)].copy()
regression = regression.sort_values("gain_pct", ascending=False)
dead_gain_pct = float(regression["gain_pct"].sum())

# master_v2 照合: serve死だが master では生 = serve-skew / master でも死 = 常時死
mcov = {}
if deadnames:
    m = pd.read_csv(BASE/"data/master_v2_20130105-20251228.csv", encoding="utf-8-sig",
                    usecols=[c for c in deadnames], low_memory=False)
    for c in deadnames:
        s = m[c].replace("", np.nan) if m[c].dtype == object else m[c]
        mcov[c] = float(s.notna().mean())

print(f"\n--- serve死亡(<5%) {len(deadnames)}特徴: master での生死 ---")
serveskew, alwaysdead = [], []
for _,r in dead.iterrows():
    mc = mcov[r['feat']]
    if mc >= 0.4:
        tag = "★serve-skew(master生→直せる)"; serveskew.append(r['feat'])
    elif mc < 0.05:
        tag = "常時死(masterも死=無害)"; alwaysdead.append(r['feat'])
    else:
        tag = f"部分({mc*100:.0f}%)"
    print(f"   serve{r['coverage']*100:4.0f}% / master{mc*100:4.0f}%  [{r['fam']}] {r['feat']}  {tag}")

print(f"\n[要約] serve死={len(deadnames)}: ★serve-skew(回収可)={len(serveskew)} / 常時死(無害)={len(alwaysdead)}")
print(f"[絶対水準] gain>0 かつ serve coverage<40%: "
      f"{len(regression)}特徴 / gain {dead_gain_pct:.2f}%")
for _, r in regression.head(15).iterrows():
    print(f"   gain={r['gain_pct']:5.2f}% / cov={r['coverage']*100:4.0f}%  {r['feat']}")
regression_rows = [{"feature": r["feat"], "coverage": round(float(r["coverage"]), 3),
                    "gain_pct": round(float(r["gain_pct"]), 4)} for _, r in regression.iterrows()]
out = {"model":"v6","weeks":WEEKS,"baseline_cov":baseline,
       "serve_dead":deadnames,"serve_skew_recoverable":sorted(serveskew),
       "always_dead_harmless":sorted(alwaysdead),
       "regression_candidates": regression_rows,
       "dead_gain_pct": round(dead_gain_pct, 4),
       "dead_gain_threshold_pct": 35.0}
# 学習時充足率 (absolute canary の基準, analysis/add_train_cov_baseline.py が書く) は
# serve 週の再測定で消してはいけない。既存ファイルにあれば引き継ぐ。
_prod = BASE/"data"/"serve_feature_baseline.json"
if _prod.exists():
    try:
        _old = json.loads(_prod.read_text(encoding="utf-8"))
        for k in ("train_cov", "gain_pct", "train_cov_note"):
            if k in _old and k not in out:
                out[k] = _old[k]
        if "train_cov" in out:
            print(f"[keep] train_cov {len(out['train_cov'])} 特徴を既存 baseline から引き継ぎ")
        else:
            print("[warn] 既存 baseline に train_cov が無い → python -m analysis.add_train_cov_baseline を実行すること")
    except Exception as e:
        print(f"[warn] 既存 baseline 読込失敗 ({e})")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUT_PATH.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
print(f"[saved] {OUT_PATH}")
