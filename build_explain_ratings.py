# -*- coding: utf-8 -*-
"""
build_explain_ratings.py — 馬名→最新as-ofレーティング辞書を生成(分析カード復活用)
================================================================================
elo/glicko/feats_serious parquet は master(〜2025-12-28)被覆で、2026週次は被覆外。
だが「馬ごとの直近as-ofレーティング(その馬の最終出走時の値=leak-safe)」は引ける。
これを 馬名 キーで書き出し、explain_card.py が 2026 カードで RTG/vs/Glicko/位置 を
再構成する(vs と race_level は 2026 出走馬の elo から field 平均で算出)。

  ./venv311/Scripts/python.exe build_explain_ratings.py            # 鮮度OKなら自動skip
  ./venv311/Scripts/python.exe build_explain_ratings.py --force    # 強制再生成
  → data/explain_horse_ratings.json  {馬名: {elo, mu, rd, se, sl}}

被覆: 直近2025までに出走歴のある馬のみ(2026初出走/新馬は None=graceful)。
再生成: 自動化済。ensure() が「元parquet/master が json より新しい時のみ」再生成する
  (build_site.py が毎回 ensure() を呼ぶ=parquet更新の四半期に自動追従、平時は mtime 比較だけで即skip)。
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent
MASTER = BASE / "data/master_v2_20130105-20251228.csv"
OUT = BASE / "data/explain_horse_ratings.json"
_SRC = [BASE / "data/elo_feats.parquet", BASE / "data/glicko_feats.parquet",
        BASE / "data/feats_serious.parquet", MASTER]


def _is_stale():
    """元データ(parquet/master)のどれかが出力より新しい / 出力欠落なら True。"""
    if not OUT.exists():
        return True
    om = OUT.stat().st_mtime
    return any(p.exists() and p.stat().st_mtime > om for p in _SRC)


def ensure(force=False, quiet=False):
    """鮮度を見て必要時のみ再生成。build_site から毎回呼ばれる(平時は mtime 比較で即return)。"""
    if not force and not _is_stale():
        if not quiet:
            print(f"[explain_ratings] fresh, skip ({OUT.name})")
        return None
    return build()


def _norm(df):
    df["rid16"] = df["rid16"].astype(str)
    df["ban"] = pd.to_numeric(df["ban"], errors="coerce").fillna(0).astype(int)
    return df


def build():
    elo = _norm(pd.read_parquet(BASE / "data/elo_feats.parquet")[["rid16", "ban", "elo_T2M1_horse"]])
    gl = _norm(pd.read_parquet(BASE / "data/glicko_feats.parquet")[["rid16", "ban", "g2_mu", "g2_rd"]])
    fs = _norm(pd.read_parquet(BASE / "data/feats_serious.parquet")[["rid16", "ban", "s2_style_early", "s2_style_late"]])

    m = pd.read_csv(MASTER, encoding="utf-8-sig", low_memory=False,
                    usecols=["レースID(新/馬番無)", "馬番", "馬名", "日付"])
    m["rid16"] = m["レースID(新/馬番無)"].astype(str).str.replace(r"\D", "", regex=True).str[:16]
    m["ban"] = pd.to_numeric(m["馬番"], errors="coerce").fillna(0).astype(int)
    m["d"] = m["日付"].astype(str).str.replace(r"\D", "", regex=True).str[:8]

    df = (m.merge(elo, on=["rid16", "ban"], how="left")
            .merge(gl, on=["rid16", "ban"], how="left")
            .merge(fs, on=["rid16", "ban"], how="left"))
    last = df.sort_values("d").groupby("馬名").tail(1)

    def rnd(v, nd):
        v = pd.to_numeric(v, errors="coerce")
        return None if v != v else round(float(v), nd)

    lut = {}
    for _, r in last.iterrows():
        nm = r["馬名"]
        if not isinstance(nm, str) or not nm:
            continue
        elo_v, mu_v = rnd(r["elo_T2M1_horse"], 1), rnd(r["g2_mu"], 0)
        rd_v = rnd(r["g2_rd"], 0)
        se_v, sl_v = rnd(r["s2_style_early"], 4), rnd(r["s2_style_late"], 4)
        if all(x is None for x in (elo_v, mu_v, se_v)):
            continue
        lut[nm] = {"elo": elo_v, "mu": mu_v, "rd": rd_v, "se": se_v, "sl": sl_v}

    OUT.write_text(json.dumps(lut, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    n_elo = sum(1 for v in lut.values() if v["elo"] is not None)
    print(f"[explain_ratings saved] {OUT}  horses={len(lut)}  with_elo={n_elo}")
    return len(lut)


if __name__ == "__main__":
    ensure(force="--force" in sys.argv)
