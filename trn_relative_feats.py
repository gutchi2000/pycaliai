"""
trn_relative_feats.py
=====================
坂路 (H) 生データから「相対化・推移・同日正規化」特徴を as-of (leak-free) で導出する。

設計思想:
  生 H の各 workout 行に対し「その workout 時点での as-of 派生量」を事前計算しておき、
  master へは merge_asof(backward, tol=14d, allow_exact_matches=False) で
  「レース日の直前(14日以内)の追い切り」を 1 件引く。これにより
    - 派生量は workout 時点までの同馬履歴のみ使用 (未来リークなし)
    - master へは race_date より厳密前の workout のみ結合 (未来リークなし)

導出する新特徴 (既存 trnH_* 16列は置換せず追加):
  (b) 自己過去比  : 直前追い切り vs その馬の過去追い切り (ベスト/平均)。Time1=4F合計, Time4=最後1F。
        trnH_t1_vs_best, trnH_t1_vs_mean, trnH_t4_vs_best, trnH_t4_vs_mean, trnH_self_n
  (d) 推移        : 1本前→直前 の変化 (負=タイム短縮=上向き)
        trnH_t1_delta, trnH_t4_delta
  (c) 同日水準正規化: 同 (train_date, 場所) の全馬調教タイム分布で z化 / robust MAD差
        trnH_t1_zday, trnH_t1_madday, trnH_t4_zday, trnH_t4_madday
  カバレッジ flag : trnH_rel_has (直前追い切りが取れたか), trnH_self_has_hist (過去本があるか)

公開関数:
  add_trn_relative_feats(df, missing="sentinel"|"median", train_mask=None)
    df は 馬名 と _rdate(レース日 datetime) を持つこと。
    missing="median" のとき train_mask (bool Series) で train 中央値を計算し補完 + *_isna flag を追加。

単体実行: python trn_relative_feats.py  → カバレッジ・分布・リーク検査を表示
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE      = Path(__file__).parent
TRAINING_H = BASE / "data/training/H-20150401-20260313.csv"

# 直前追い切りとして認める最大日数 (train_lgbm_trn_diag と整合)
ASOF_TOL_DAYS = 14

# value 特徴 (欠損補完の対象)
VALUE_FEATS = [
    "trnH_t1_vs_best", "trnH_t1_vs_mean", "trnH_t4_vs_best", "trnH_t4_vs_mean",
    "trnH_self_n", "trnH_t1_delta", "trnH_t4_delta",
    "trnH_t1_zday", "trnH_t1_madday", "trnH_t4_zday", "trnH_t4_madday",
]
FLAG_FEATS = ["trnH_rel_has", "trnH_self_has_hist"]
ALL_NEW_FEATS = VALUE_FEATS + FLAG_FEATS


def _parse_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s.astype(str).str.replace(r"[/\-]", "", regex=True),
                          format="%Y%m%d", errors="coerce")


_WORKOUT_CACHE: pd.DataFrame | None = None


def _build_workout_table() -> pd.DataFrame:
    """生 H を読み、各 workout 行に as-of 派生量を付与したテーブルを返す (プロセス内キャッシュ)。"""
    global _WORKOUT_CACHE
    if _WORKOUT_CACHE is not None:
        return _WORKOUT_CACHE
    h = pd.read_csv(TRAINING_H, encoding="cp932", low_memory=False)
    # 列名: 場所, 年月日, 馬名, Time1..Time4, Lap4..Lap1
    h = h.rename(columns={"年月日": "wd", "馬名": "name",
                          "Time1": "t1", "Time4": "t4", "場所": "place"})
    h["date_dt"] = _parse_date(h["wd"])
    h["name"] = h["name"].astype(str).str.strip()
    for c in ["t1", "t4"]:
        h[c] = pd.to_numeric(h[c], errors="coerce")
    # 異常値除外 (parse_training と同じ規約: 4F合計 40〜100秒)
    h = h[(h["t1"] > 40) & (h["t1"] < 100)].dropna(subset=["date_dt", "name", "t1", "t4"])
    h = h.sort_values(["name", "date_dt"]).reset_index(drop=True)

    g = h.groupby("name", sort=False)
    cc = g.cumcount()              # 同馬で現在より前の本数 (0始まり)
    h["_cc"] = cc

    # --- (b) 自己過去 (現在より前) ---
    # 過去ベスト = cummin を 1 つ shift (現在を含めない)。group 境界は _cc==0 を NaN。
    t1_cummin = g["t1"].cummin()
    t4_cummin = g["t4"].cummin()
    h["_t1_pastbest"] = t1_cummin.shift(1).where(cc > 0)
    h["_t4_pastbest"] = t4_cummin.shift(1).where(cc > 0)
    # 過去平均 = (累積和 - 現在) / 過去本数
    t1_cumsum = g["t1"].cumsum()
    t4_cumsum = g["t4"].cumsum()
    h["_t1_pastmean"] = ((t1_cumsum - h["t1"]) / cc.where(cc > 0))
    h["_t4_pastmean"] = ((t4_cumsum - h["t4"]) / cc.where(cc > 0))

    # --- (d) 推移 (1本前との差) ---
    h["_t1_prev"] = g["t1"].shift(1)   # groupby.shift は group 単位 (境界安全)
    h["_t4_prev"] = g["t4"].shift(1)

    # --- (c) 同日 (train_date, place) 正規化 ---
    gd = h.groupby(["date_dt", "place"], sort=False)
    h["_t1_dmean"] = gd["t1"].transform("mean")
    h["_t1_dstd"]  = gd["t1"].transform("std")
    h["_t1_dmed"]  = gd["t1"].transform("median")
    h["_t4_dmean"] = gd["t4"].transform("mean")
    h["_t4_dstd"]  = gd["t4"].transform("std")
    h["_t4_dmed"]  = gd["t4"].transform("median")
    h["_t1_absdev"] = (h["t1"] - h["_t1_dmed"]).abs()
    h["_t4_absdev"] = (h["t4"] - h["_t4_dmed"]).abs()
    h["_t1_dmad"] = gd["_t1_absdev"].transform("median")
    h["_t4_dmad"] = gd["_t4_absdev"].transform("median")

    # --- 派生特徴 (タイム短縮=値が小→ delta 負=上向き / vs_best 負=自己ベスト更新) ---
    h["trnH_t1_vs_best"] = h["t1"] - h["_t1_pastbest"]
    h["trnH_t1_vs_mean"] = h["t1"] - h["_t1_pastmean"]
    h["trnH_t4_vs_best"] = h["t4"] - h["_t4_pastbest"]
    h["trnH_t4_vs_mean"] = h["t4"] - h["_t4_pastmean"]
    h["trnH_self_n"]     = cc.astype(float)
    h["trnH_t1_delta"]   = h["t1"] - h["_t1_prev"]
    h["trnH_t4_delta"]   = h["t4"] - h["_t4_prev"]
    with np.errstate(invalid="ignore", divide="ignore"):
        h["trnH_t1_zday"]   = (h["t1"] - h["_t1_dmean"]) / h["_t1_dstd"].replace(0, np.nan)
        h["trnH_t4_zday"]   = (h["t4"] - h["_t4_dmean"]) / h["_t4_dstd"].replace(0, np.nan)
        h["trnH_t1_madday"] = (h["t1"] - h["_t1_dmed"]) / (1.4826 * h["_t1_dmad"]).replace(0, np.nan)
        h["trnH_t4_madday"] = (h["t4"] - h["_t4_dmed"]) / (1.4826 * h["_t4_dmad"]).replace(0, np.nan)
    h["trnH_self_has_hist"] = (cc > 0).astype(float)

    keep = ["name", "date_dt"] + VALUE_FEATS + ["trnH_self_has_hist"]
    _WORKOUT_CACHE = h[keep].copy()
    return _WORKOUT_CACHE


def add_trn_relative_feats(df: pd.DataFrame, missing: str = "sentinel",
                           train_mask: pd.Series | None = None,
                           verbose: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """df (馬名, _rdate を持つ) に坂路相対特徴を merge_asof 付与。

    Returns: (df_with_feats, new_feature_columns)
    missing:
      "sentinel" → 欠損はそのまま (呼び出し側 fillna(-9999) を想定)。flag 列 trnH_rel_has 追加。
      "median"   → value 列を train 中央値で補完 + 各 value に *_isna flag を追加。
    """
    assert "_rdate" in df.columns, "df は _rdate (レース日 datetime) が必要"
    name_col = "馬名" if "馬名" in df.columns else "name"
    R = _build_workout_table()

    L = df.copy()
    L["_name_key"] = L[name_col].astype(str).str.strip()
    # master の既存 date_dt 列と R(workout)の date_dt が merge_asof で衝突するため左側を落とす
    L = L.drop(columns=[c for c in ["date_dt"] if c in L.columns])
    R = R.rename(columns={"name": "_name_key"})
    L = L.sort_values("_rdate").reset_index(drop=False)  # index 保持
    R = R.sort_values("date_dt").reset_index(drop=True)

    merged = pd.merge_asof(
        L, R, left_on="_rdate", right_on="date_dt", by="_name_key",
        direction="backward", tolerance=pd.Timedelta(days=ASOF_TOL_DAYS),
        allow_exact_matches=False,
    )
    # 直前追い切り(14d以内)が結合できたか = 右側 date_dt が入ったか
    merged["trnH_rel_has"] = merged["date_dt"].notna().astype(float)
    # 元の行順へ戻す
    merged = merged.sort_values("index").set_index("index")
    merged.index.name = None

    out = df.copy()
    for c in VALUE_FEATS + ["trnH_self_has_hist"]:
        out[c] = merged[c].values
    out["trnH_rel_has"] = merged["trnH_rel_has"].values

    new_feats = list(ALL_NEW_FEATS)

    if missing == "median":
        if train_mask is None:
            raise ValueError("missing='median' は train_mask が必要")
        added = []
        for c in VALUE_FEATS:
            isna = out[c].isna()
            out[c + "_isna"] = isna.astype(float)
            med = pd.to_numeric(out.loc[train_mask.values, c], errors="coerce").median()
            out[c] = out[c].fillna(med)
            added.append(c + "_isna")
        new_feats = VALUE_FEATS + added + ["trnH_self_has_hist", "trnH_rel_has"]

    if verbose:
        cov = out["trnH_rel_has"].mean() * 100
        cov_hist = out["trnH_self_has_hist"].fillna(0).astype(bool).mean() * 100
        print(f"  [trn_relative] 直前追い切り(14d) カバレッジ={cov:.1f}%  過去本あり={cov_hist:.1f}%  新特徴={len(new_feats)}")
    return out, new_feats


# ============================================================
# 単体実行: カバレッジ・分布・リーク検査
# ============================================================
def _selftest():
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    except Exception:
        pass
    print("=" * 70)
    print("trn_relative_feats.py 単体検査 (生Hの派生量 + master 一部結合)")
    print("=" * 70)

    R = _build_workout_table()
    print(f"\nworkout table: {len(R):,} 行")
    print("派生量 describe:")
    print(R[VALUE_FEATS].describe().T.to_string())

    # master の一部で結合カバレッジ
    MASTER = BASE / "data/master_v2_20130105-20251228.csv"
    usecols = ["馬名", "日付", "着順", "レースID(新/馬番無)", "split", "trnH_Time1", "trnH_days_ago"]
    df = pd.read_csv(MASTER, encoding="utf-8-sig", usecols=usecols, low_memory=False)
    df["_rdate"] = _parse_date(df["日付"])
    df = df.dropna(subset=["_rdate"])
    print(f"\nmaster rows: {len(df):,}")

    out, feats = add_trn_relative_feats(df, missing="sentinel")
    # split 別カバレッジ
    for sp in ["train", "valid", "test"]:
        s = out[out["split"] == sp]
        if len(s):
            print(f"  split={sp:6s} n={len(s):>8,}  rel_has={s['trnH_rel_has'].mean()*100:5.1f}%  "
                  f"hist={s['trnH_self_has_hist'].fillna(0).astype(bool).mean()*100:5.1f}%")

    # リーク検査: 既存 trnH_days_ago と新結合の整合 (14日以内に絞ったので days_ago<=14 の行で rel_has が高いはず)
    m = out["trnH_days_ago"].notna()
    le14 = out.loc[m, "trnH_days_ago"] <= 14
    print(f"\n[リーク/整合検査] 既存 trnH_days_ago<=14 の行で rel_has 率: "
          f"{out.loc[m & (out['trnH_days_ago']<=14), 'trnH_rel_has'].mean()*100:.1f}%  "
          f"(>14 の行: {out.loc[m & (out['trnH_days_ago']>14), 'trnH_rel_has'].mean()*100:.1f}%)")
    # 相関 (情報の重複の目安): 新 delta/vs_best と既存 trnH_Time1
    sub = out.dropna(subset=["trnH_t1_vs_best", "trnH_Time1"])
    if len(sub) > 1000:
        print(f"  corr(trnH_t1_vs_best, trnH_Time1) = {sub['trnH_t1_vs_best'].corr(sub['trnH_Time1']):.3f}")
        print(f"  corr(trnH_t1_zday,   trnH_Time1) = {sub['trnH_t1_zday'].corr(sub['trnH_Time1']):.3f}")
    print("\n[OK] 単体検査完了")


if __name__ == "__main__":
    _selftest()
