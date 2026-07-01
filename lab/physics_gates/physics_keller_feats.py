# -*- coding: utf-8 -*-
"""
physics_keller_feats.py — Keller 競走力学モデルに基づく「場のペース×エネルギー」特徴
====================================================================================
J. B. Keller (1973) "A theory of competitive running" の発想を競馬に移植する。

骨子:
  ランナーは有限の無酸素エネルギー貯蔵 E0 を持ち、推進力 f と抵抗 v/τ の力学
      dv/dt = f - v/τ,    dE/dt = σ - f·v  (E>=0)
  に従う。短〜中距離では「前で脚を使う＝早期にE枯渇＝直線で失速」、
  「後方で温存＝直線で残存Eを末脚に変換」という非線形が支配する。

  各馬の f,τ,E0 は直接観測できないが、以下で代理する（すべて前走/場構成のみ＝leak-free）:
    - late-speed capacity (末脚＝残存Eの放出能力): 前走上り3F / closing_power / 前PCI 等の race内 z
    - 早期消耗 burn: 自分の脚質(style_rank) × 場のペース硬度(pace_pressure) × 距離係数
    - drafting (空力スリップストリーム): 隊列内位置に応じた抗力低減（先頭は0、好位で最大）

  ★物理の付加価値は「この場の」ペース戦から来る相互作用項であり、
    単体集計統計(勝率/平均着順)が取りこぼす成分を狙う。

入力 : data/master_v2_20130105-20251228.csv  (leak-free 列のみ)
出力 : data/keller_pace_feats.parquet  (key = rid16 + 馬番)
実行 : PYTHONUTF8=1 python physics_keller_feats.py
"""
from __future__ import annotations
import io, re, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

BASE = Path(__file__).parent
MASTER = BASE / "data/master_v2_20130105-20251228.csv"
OUT = BASE / "data/keller_pace_feats.parquet"

from corr_features import add_style, COL_RID  # style_rank / pace_pressure / n_front / field_size_eff

COL_BAN = "馬番"
COL_DIST = "距離"
COL_AGARI = "前走上り3F"
COL_PCI = "前PCI"
COL_RPCI = "前走RPCI"
COL_CLOSE = "closing_power"
COL_K5AGARI = "kako5_avg_agari3f"


def _to_num(s):
    return pd.to_numeric(s, errors="coerce")


def _rid16(x):
    return re.sub(r"\D", "", str(x))[:16]


def _race_z(val: pd.Series, key: pd.Series) -> pd.Series:
    """レース内 z-score（場内相対化）。std=0 や n<3 は 0。"""
    g = val.groupby(key)
    mu = g.transform("mean")
    sd = g.transform("std")
    z = (val - mu) / sd.replace(0, np.nan)
    return z.fillna(0.0).clip(-4, 4)


def build() -> pd.DataFrame:
    usecols = [COL_RID, COL_BAN, "枠番", COL_DIST, "出走頭数",
               "前走出走頭数", "前1角", "前2角", "前3角", "前4角",
               COL_AGARI, COL_PCI, COL_RPCI, COL_CLOSE, COL_K5AGARI,
               "split", "fukusho_flag"]
    df = pd.read_csv(MASTER, encoding="utf-8-sig",
                     usecols=lambda c: c in usecols, low_memory=False)
    print(f"[load] master rows={len(df):,}")

    # --- 脚質・場ペース（既存 corr_features を再利用; leak-free 前走由来）---
    df = add_style(df)   # style_rank, pace_pressure, n_front, field_size_eff
    sr = df["style_rank"].fillna(0.5)          # NaN は中立
    pp = df["pace_pressure"].clip(0, 1)
    fs = df["field_size_eff"].clip(lower=1)
    nf = df["n_front"]
    dist = _to_num(df[COL_DIST])

    # ---------- 1. 場のペース硬度（距離で重み付け）----------
    # スプリントほど早期ペースの消耗影響が大きい（dist_w: 1200m=1.0 付近、長距離で減衰）
    dist_w = np.clip((2200.0 - dist) / 1000.0, 0.2, 1.3).fillna(0.8)
    pace_hardness = (pp * dist_w).clip(0, 1.5)
    contested = (nf > 1).astype(float)          # 先行争いが起きるか
    solo_front = ((sr < 0.15) & (nf <= 1)).astype(float)  # 単騎楽逃げ

    # ---------- 2. Keller 早期消耗 burn と残存エネルギー reserve ----------
    # 前方(style_rank小)ほど、かつ場が硬い(pace_hardness大)ほど早期にEを消費
    fwd = np.clip((0.30 - sr) / 0.30, 0, 1)     # 0..1 (逃げ先行で1)
    burn = fwd * pace_hardness * contested      # 単騎なら contested=0 で消耗小
    reserve = (1.0 - burn).clip(0, 1) + 0.15 * solo_front  # ゴール時残存E

    # ---------- 3. drafting（スリップストリーム）----------
    # 好位(style_rank≈0.30)で抗力低減が最大。先頭(<0.05)は前に馬無し→0。
    draft_kernel = np.exp(-((sr - 0.30) / 0.20) ** 2)
    draft_kernel = np.where(sr < 0.05, 0.0, draft_kernel)
    density = (fs / 18.0).clip(0, 1)            # 多頭数ほど隊列が密＝draft機会増
    draft = draft_kernel * density * np.clip(dist_w, 0, 1)

    # ---------- 4. 末脚放出能力 late-speed capacity ----------
    agari_z = -_race_z(_to_num(df[COL_AGARI]), df[COL_RID])   # 速い(小)ほど +
    close_z = _race_z(_to_num(df[COL_CLOSE]), df[COL_RID])
    pci_z = _race_z(_to_num(df[COL_PCI]), df[COL_RID])        # 後半型ほど +
    k5_z = -_race_z(_to_num(df[COL_K5AGARI]), df[COL_RID])
    ls_cap = (agari_z + close_z + 0.5 * pci_z + k5_z) / 3.5

    # ---------- 5. 合成: pace_fit / surge ----------
    # pace_fit: 硬いペースは差し有利・緩いペースは先行有利（符号一致で +）
    pace_fit = (pp - 0.33) * (sr - 0.5) * 2.0
    # surge: 残存E × 末脚能力 + draft 補正（直線で伸びる物理ポテンシャル）
    surge = reserve * ls_cap + 0.30 * draft

    out = pd.DataFrame({
        "rid16": df[COL_RID].map(_rid16),
        "ban": _to_num(df[COL_BAN]).astype("Int64"),
        "kl_reserve": reserve.astype(float),
        "kl_burn": burn.astype(float),
        "kl_draft": draft.astype(float),
        "kl_lscap": ls_cap.astype(float),
        "kl_pace_fit": pace_fit.astype(float),
        "kl_surge": surge.astype(float),
        "kl_solo_front": solo_front.astype(float),
        "kl_pace_hardness": pace_hardness.astype(float),
        "split": df["split"].values,
    })
    out = out[out["ban"].notna()].copy()
    return out


def main():
    out = build()
    out.to_parquet(OUT, index=False)
    feats = [c for c in out.columns if c.startswith("kl_")]
    print(f"[saved] {OUT}  rows={len(out):,}")
    print("\n=== feature describe (全期間) ===")
    print(out[feats].describe().T[["mean", "std", "min", "max"]].to_string())
    # 簡易: test 期間で style_rank/pace と物理特徴の素の相関（過度な重複が無いか確認用）
    print("\n=== サンプル head ===")
    print(out[["rid16", "ban"] + feats].head(6).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
