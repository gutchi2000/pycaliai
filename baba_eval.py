# -*- coding: utf-8 -*-
"""
baba_eval.py — 新観測量(クッション値/含水率)の予測上乗せを ablation で検定
==============================================================================
クッション値はレース内で定数(場所×日)→生値は順位に効かない(softmaxで打ち消える)。
効くのは「各馬の馬場適性 × 今日の硬度」の相互作用(レース内で馬ごとに変わる)。

特徴 (per race×horse, as-of, leak-safe):
  baba_cushion       今日のクッション値(芝, 場所×日)         within-race定数
  baba_hum           今日の含水率(芝/ダ surface一致, ゴール前)  within-race定数
  baba_firmness_today 今日の馬場状態 firmness(0-1)            within-race定数
  baba_cushion_z     今日のクッションの全体z                 within-race定数
  baba_soft_aff      馬の軟馬場適性 = top3率(軟)-top3率(硬) (馬場状態の長履歴, as-of, 収縮)
  baba_aff_x_soft    soft_aff × (1-firmness_today)           ★相互作用(軟適性馬が軟い日)
  baba_aff_x_cushion soft_aff × (-cushion_z)                 ★相互作用(軟適性馬が低クッション日)
  baba_cush_pref_gap 今日cushion - 馬の過去平均cushion        馬にとって異常な硬さか

物理: クッション値=反発係数。馬場状態(良/稍/重/不)を超える連続解像度を今日の硬度が与え、
      各馬の好みとのマッチが着順に効く、という仮説。

検定: v6 BASE_FEATS(physics_eval流用) vs +baba, fukusho_flag test2024-25 AUC/logloss + part_corr。
事前登録: ΔAUC>+0.002 かつ best|part_corr|>0.01。
実行: python baba_eval.py
"""
from __future__ import annotations
import io, json, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
except Exception:
    pass

from physics_eval import BASE_FEATS, fit_eval, _rid16
from corr_features import COL_RID, add_style

BASE = Path(__file__).parent
MASTER = BASE / "data/master_v2_20130105-20251228.csv"
BABA = BASE / "data/baba_feats.parquet"
OUT = BASE / "reports/baba_feats_eval.json"
COL_BAN = "馬番"; COL_DATE = "日付"; COL_PED = "血統登録番号"; COL_PLACE = "場所"
BABA_FEATS = ["baba_cushion_zt", "baba_hum_zt", "baba_cushion_hard", "baba_cushion_soft",
              "baba_firmness_today", "baba_soft_aff", "baba_aff_x_cushion", "baba_aff_x_hum",
              "baba_dirt_wet_x_style", "baba_wsbias", "baba_style_match"]
SHRINK_K = 6.0


def firmness(s):
    s = str(s)
    if "不" in s: return 0.0      # 不良
    if "稍" in s: return 0.66     # 稍重
    if "重" in s: return 0.33
    if "良" in s: return 1.0
    return np.nan


def build_baba(df):
    """df(master) に baba 特徴を付与。as-of leak-safe。
    ★公式準拠(JRA表): クッション値=高い→硬め/乾燥, 低い→軟らかめ/湿潤。
      12+硬め / 8-10標準 / 7-軟らかめ。競馬場ごとに基準が違う→場所別標準化必須。"""
    b = pd.read_parquet(BABA)
    df = df.merge(b.rename(columns={"日付": "_bd"}), left_on=[COL_PLACE, COL_DATE],
                  right_on=[COL_PLACE, "_bd"], how="left")
    surf = df["芝・ダ"].astype(str)
    is_dirt = surf.str.contains("ダ")
    cu = pd.to_numeric(df["cushion"], errors="coerce")       # 芝のみ; 高い=硬/乾, 低い=軟/湿
    hum = pd.to_numeric(pd.Series(np.where(is_dirt, df["dirt_gp"], df["shiba_gp"]),
                                  index=df.index), errors="coerce")
    df["baba_firmness_today"] = df["馬場状態"].map(firmness)
    tr = df["split"].astype(str) == "train"

    # --- 競馬場ごと標準化 (札幌/函館は系統的に低い等, 生値は場所間で比較不能) ---
    def ztrack(s, key):
        s = pd.to_numeric(s, errors="coerce")
        m = s[tr].groupby(key[tr]).mean()
        sd = s[tr].groupby(key[tr]).std()
        return (s - key.map(m)) / key.map(sd).replace(0, np.nan)
    df["baba_cushion_zt"] = ztrack(cu, df[COL_PLACE])
    surf_key = pd.Series(df[COL_PLACE].astype(str) + np.where(is_dirt, "|D", "|T"), index=df.index)
    df["baba_hum_zt"] = ztrack(hum, surf_key)

    # --- 絶対閾値レジーム (JRA表: >10 硬い/乾燥側, <8 軟らかめ/湿潤側) ---
    df["baba_cushion_hard"] = (cu - 10.0).clip(lower=0)   # 高い=硬め/乾燥の深さ
    df["baba_cushion_soft"] = (8.0 - cu).clip(lower=0)    # 低い=軟らかめ/湿潤の深さ

    # --- 馬の軟馬場適性 (馬場状態の長履歴で as-of, current除外。馬場状態mappingは正) ---
    df = df.sort_values([COL_PED, COL_DATE], kind="stable").reset_index(drop=True)
    fz = df["baba_firmness_today"]
    t3 = pd.to_numeric(df["fukusho_flag"], errors="coerce").fillna(0)
    g = df.groupby(COL_PED, sort=False)
    df["_s"] = (fz < 0.7).astype(float); df["_f"] = (fz >= 0.7).astype(float)
    df["_st3"] = df["_s"] * t3; df["_ft3"] = df["_f"] * t3
    for c in ["_s", "_f", "_st3", "_ft3"]:
        df[c] = g[c].cumsum() - df[c]
    base = float(t3[tr].mean())
    soft_rate = (df["_st3"] + SHRINK_K * base) / (df["_s"] + SHRINK_K)
    firm_rate = (df["_ft3"] + SHRINK_K * base) / (df["_f"] + SHRINK_K)
    df["baba_soft_aff"] = soft_rate - firm_rate   # 正=軟馬場巧者

    # --- 相互作用 (符号修正済: 高cushion=軟 → soft_aff × +cushion_zt) ---
    czt = df["baba_cushion_zt"].fillna(0.0); hzt = df["baba_hum_zt"].fillna(0.0)
    df["baba_aff_x_cushion"] = df["baba_soft_aff"] * (-czt)     # 軟適性×軟馬場(低cushion=軟)
    df["baba_aff_x_hum"] = df["baba_soft_aff"] * hzt            # 軟適性×高含水(軟)
    # --- ダート: 湿る=速い→先行有利。dirt湿り × 前付け度(0.5-style; 正=前) ---
    style = pd.to_numeric(df.get("style_rank"), errors="coerce").fillna(0.5)
    df["baba_dirt_wet_x_style"] = np.where(is_dirt, hzt * (0.5 - style), 0.0)

    # === ★ユーザー指摘: 会場×芝ダで「求められるもの」が違う ===
    # (会場×surface) ごとに「濡れると勝ち脚質がどう動くか」D を過去から as-of 推定し、
    # 今日の条件で favored な脚質 × その馬の脚質 のマッチを作る。ダートは符号が逆でも
    # データから推定するので自動。leak-safe(過去レースのみ)・レース内可変。
    df["_style"] = pd.to_numeric(df.get("style_rank"), errors="coerce")
    df["_vs"] = df[COL_PLACE].astype(str) + np.where(is_dirt, "D", "T")
    df["_wet"] = hzt                                  # per (会場×surface) 正規化済 wetness
    df["_a"] = df["_style"] * t3                      # top3馬の脚質寄与
    df["_b"] = t3 * df["_style"].notna().astype(float)
    gr = df.groupby(COL_RID, sort=False)
    race = pd.DataFrame({"_vs": gr["_vs"].first(), "_date": gr[COL_DATE].first(),
                         "_wet": gr["_wet"].first(), "_sa": gr["_a"].sum(), "_sb": gr["_b"].sum()})
    race["t3s"] = race["_sa"] / race["_sb"].replace(0, np.nan)   # そのレース top3 平均脚質
    race = race.sort_values(["_vs", "_date"], kind="stable")
    race["_isw"] = (race["_wet"] > 0).astype(float)

    def asof_mean(val, mask, by):                    # by内 date順 expanding平均(当該除外)
        v = (val * mask).fillna(0.0); m = mask * val.notna()
        cs = v.groupby(by).cumsum() - v
        cn = m.groupby(by).cumsum() - m
        return cs / cn.replace(0, np.nan)
    race["wet_s"] = asof_mean(race["t3s"], race["_isw"], race["_vs"])
    race["dry_s"] = asof_mean(race["t3s"], 1 - race["_isw"], race["_vs"])
    race["D"] = race["wet_s"] - race["dry_s"]        # >0=この会場×surfaceで濡れ→差し勝ち
    df = df.merge(race[["D"]], left_on=COL_RID, right_index=True, how="left")
    df["baba_wsbias"] = df["D"].fillna(0.0) * df["_wet"].fillna(0.0)   # 今日のfavored脚質シフト
    df["baba_style_match"] = (df["_style"].fillna(0.5) - 0.5) * df["baba_wsbias"]  # 馬の脚質が合致

    return df.drop(columns=["_s", "_f", "_st3", "_ft3", "_bd", "_a", "_b", "_vs", "_wet", "D"],
                   errors="ignore")


def main():
    need = list(dict.fromkeys(
        [COL_RID, COL_BAN, COL_DATE, COL_PED, COL_PLACE, "芝・ダ", "馬場状態",
         "split", "fukusho_flag", "前走出走頭数", "前1角", "前2角", "前3角", "前4角"] + BASE_FEATS))
    need = [c for c in need if c not in ("style_rank", "pace_pressure")]  # add_styleが生成
    df = pd.read_csv(MASTER, encoding="utf-8-sig",
                     usecols=lambda c: c in need, low_memory=False)
    df[COL_DATE] = pd.to_numeric(df[COL_DATE], errors="coerce")
    df["fukusho_flag"] = pd.to_numeric(df["fukusho_flag"], errors="coerce")
    df = add_style(df)
    df = build_baba(df)
    df = df[df["fukusho_flag"].isin([0, 1])].copy()
    df["rid16"] = df[COL_RID].map(_rid16)
    for c in BASE_FEATS + BABA_FEATS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    n = {s: int((df["split"] == s).sum()) for s in ["train", "valid", "test"]}
    cov = df.loc[df["split"] == "test", "baba_cushion_zt"].notna().mean() * 100
    print(f"[rows] train={n['train']:,} valid={n['valid']:,} test={n['test']:,}  "
          f"test cushion cov={cov:.1f}%")

    print("\n=== H1: v6 baseline vs +baba (test=2024+, fukusho) ===")
    res = {}
    res["base"], p_base, m_base, te = fit_eval(df, BASE_FEATS, "baseline")
    res["+baba"], p_b, m_b, _ = fit_eval(df, BASE_FEATS + BABA_FEATS, "base+baba")
    d_auc = res["+baba"]["auc"] - res["base"]["auc"]
    d_ll = res["+baba"]["logloss"] - res["base"]["logloss"]
    print(f"\n  ΔAUC +baba = {d_auc:+.5f}   Δlogloss = {d_ll:+.6f} (負=改善)")

    print("\n=== 直交性 part_corr(feature, y - p_base) on test ===")
    resid = te["fukusho_flag"].values - p_base
    ortho = {}
    for f in BABA_FEATS:
        v = te[f].astype(float); mask = v.notna().values
        if mask.sum() < 1000:
            print(f"  {f:18s} (n<1000 skip)"); continue
        pc = float(np.corrcoef(v.values[mask], resid[mask])[0, 1])
        ortho[f] = pc
        print(f"  {f:18s} part_corr={pc:+.4f}  cov={mask.mean()*100:.0f}%")

    imp = dict(zip(m_b.feature_name(), m_b.feature_importance(importance_type="gain")))
    tot = sum(imp.values()) or 1
    share = sum(imp.get(f, 0) for f in BABA_FEATS) / tot * 100
    print(f"\n=== baba gain share = {share:.2f}% ===")
    for f in sorted(BABA_FEATS, key=lambda x: -imp.get(x, 0)):
        print(f"  {f:18s} gain={imp.get(f,0):,.0f} ({imp.get(f,0)/tot*100:.2f}%)")

    # === race-level: cushion が「本命崩壊」を頭数/エントロピー統制下で予測するか ===
    # (cushionはレース内定数→ランキング不可。正しい問いはレースレベルの荒れ予測)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score as _auc

    def race_frame(split):
        d = df[df["split"] == split]
        X = d[BASE_FEATS].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
        d = d.assign(_p=m_base.predict(X))
        rows = []
        for _, g in d.groupby(COL_RID, sort=False):
            if len(g) < 4 or pd.isna(g["baba_cushion_zt"].iloc[0]):
                continue
            pp = g["_p"].values
            q = np.exp(pp - pp.max()); q = q / q.sum()
            fav = int(np.argmax(pp))
            rows.append(dict(
                H=float(-np.sum(q * np.log(q + 1e-12))), n=float(len(g)),
                c_zt=float(g["baba_cushion_zt"].iloc[0]),
                c_hard=float(g["baba_cushion_hard"].iloc[0]),
                c_soft=float(g["baba_cushion_soft"].iloc[0]),
                collapse=int(pd.to_numeric(g["fukusho_flag"].values[fav], errors="coerce") == 0)))
        return pd.DataFrame(rows)

    rtr, rte = race_frame("train"), race_frame("test")
    print(f"\n=== race-level 本命崩壊予測 (cushion有 turf, train={len(rtr):,}/test={len(rte):,}, "
          f"崩壊率test={rte['collapse'].mean():.3f}) ===")
    rl_auc = {}
    for name, cols in [("base[H,n]", ["H", "n"]), ("+cushion", ["H", "n", "c_zt", "c_hard", "c_soft"])]:
        sc = StandardScaler().fit(rtr[cols])
        clf = LogisticRegression(max_iter=1000).fit(sc.transform(rtr[cols]), rtr["collapse"])
        a = float(_auc(rte["collapse"], clf.predict_proba(sc.transform(rte[cols]))[:, 1]))
        rl_auc[name] = a
        print(f"  {name:14s} AUC={a:.5f}")
    d_rl = rl_auc["+cushion"] - rl_auc["base[H,n]"]
    print(f"  ΔAUC(+cushion - base) = {d_rl:+.5f}")

    best_pc = max((abs(v) for v in ortho.values()), default=0)
    win = d_auc > 0.002 and best_pc > 0.01
    print(f"\n=== 判定 ===")
    print(f"  H1 ΔAUC>0.002: {'○' if d_auc>0.002 else '×'} ({d_auc:+.5f})")
    print(f"  best|part_corr|>0.01: {'○' if best_pc>0.01 else '×'} ({best_pc:.4f})")
    print(f"  → {'KEEP (新観測量が効く)' if win else 'marginal/KILL (馬場状態に吸収)'}")

    out = dict(rows=n, test_cushion_cov=float(cov), models=res,
               delta_auc=float(d_auc), delta_logloss=float(d_ll),
               orthogonality=ortho, gain_share_pct=float(share),
               verdict="KEEP" if win else "marginal")
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[saved] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
