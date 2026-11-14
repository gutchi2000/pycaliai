# -*- coding: utf-8 -*-
"""
market_residual_verify.py — M ゲートを通った 3 本 (馬の市場サプライズ・種牡馬×芝ダ・ムラ馬) の事後検証 (2026-11-14)
土台 [確定市場 + v7 (+ 対照)] に足して勝ち馬 NLL が下がるか。対照 = 前走着順ダミー (1着 / 2-3着 / 6-9着 / 10着+ / 前走なし) + キャリアの深さ。
偽薬 = 種牡馬の値をレース内で入れ替え。valid=2023 で fit、test=2024-25、開催日 bootstrap、年別符号。
入力: reports/mls_states.parquet (market_learning_speed_exp.py)、reports/vol_features.parquet (volatility_option_exp.py)
実行: E:/PyCaLiAI/venv311/Scripts/python.exe lab/experiments/market_residual_verify.py  → reports/market_residual_verify.json
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate_common as gc
from volatility_option_verify import run

Z = ["horse_top3_h3", "sire_surface_top3_h200", "logsig"]
D = ["d1", "d23", "d69", "d10", "dnone", "ncar"]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    b = gc.load_base()
    m = pd.read_parquet(gc.REP / "mls_states.parquet")[["rid_s", "ban", "horse_top3_h3", "sire_surface_top3_h200"]]
    v = pd.read_parquet(gc.REP / "vol_features.parquet")[["rid_s", "ban", "logsig"]]
    b = b.merge(m, on=["rid_s", "ban"], how="left").merge(v, on=["rid_s", "ban"], how="left")
    s = gc.load_sed(["racekey", "ymd", "ban", "ped", "td", "jyun", "tosu", "ijou"])
    s = s[s.td.isin([1, 2])].sort_values(["ped", "date"]); g = s.groupby("ped"); pj = g.jyun.shift(1)
    s["d1"] = (pj == 1).astype(float); s["d23"] = pj.between(2, 3).astype(float); s["d69"] = pj.between(6, 9).astype(float)
    s["d10"] = (pj >= 10).astype(float); s["dnone"] = pj.isna().astype(float); s["ncar"] = np.log1p(g.cumcount())
    b = b.merge(s[["rid_s", "ban"] + D].drop_duplicates(["rid_s", "ban"]), on=["rid_s", "ban"], how="left")
    r0 = np.random.default_rng(5)
    b["plc"] = b.groupby("rid_s")["sire_surface_top3_h200"].transform(lambda x: r0.permutation(x.values))
    for c in Z + D + ["plc"]:
        b[c] = pd.to_numeric(b[c], errors="coerce"); b[c] = b[c].fillna(b.groupby("rid_s")[c].transform("mean")).fillna(0.0)
        b[c] = b[c] - b.groupby("rid_s")[c].transform("mean"); sd = b.loc[b.split == "valid", c].std(); b[c] = b[c] / (sd if sd > 0 else 1)
    rng = np.random.default_rng(0); M = ["lk", "lm7"]; res = {}
    res["horse"] = run(b, M, [], [Z[0]], "M: 馬の市場サプライズ", rng)
    res["horse_ctrl"] = run(b, M, D, [Z[0]], "M: 馬サプライズ | 対照", rng)
    res["sire"] = run(b, M, [], [Z[1]], "M: 種牡馬×芝ダ", rng)
    res["sire_ctrl"] = run(b, M, D, [Z[1]], "M: 種牡馬×芝ダ | 対照", rng)
    res["sire_placebo"] = run(b, M, [], ["plc"], "M: 種牡馬 偽薬", rng)
    res["dummies"] = run(b, M, [], D, "M: 対照だけ", rng)
    res["three"] = run(b, M, [], Z, "M: 3 本", rng)
    res["three_ctrl"] = run(b, M, D, Z, "M: 3 本 | 対照", rng)
    res["three_M0"] = run(b, ["lk"], [], Z, "M0: 3 本", rng)
    res["three_A10"] = run(b, ["lk10", "lm7"], [], Z, "A10: 3 本", rng)
    res["three_P"] = run(b, ["lm7"], [], Z, "P: 3 本", rng)
    (gc.REP / "market_residual_verify.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
