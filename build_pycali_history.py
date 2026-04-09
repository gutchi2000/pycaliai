"""
build_pycali_history.py
reports/pred_*.csv を全走査し、(馬名, 日付, スコア) を集約して
data/pycali_history.parquet に保存する。
app.py の全頭分析タブでスパークライン描画に使用する。
"""
from __future__ import annotations
from pathlib import Path
import glob
import pandas as pd

BASE = Path(__file__).parent
OUT = BASE / "data" / "pycali_history.parquet"


def main() -> None:
    files = sorted(glob.glob(str(BASE / "reports" / "pred_*.csv")))
    print(f"pred files: {len(files)}")
    rows = []
    for f in files:
        try:
            df = pd.read_csv(f, encoding="utf-8-sig", usecols=["日付", "馬名", "スコア", "レースID"])
        except Exception as e:
            print(f"skip {Path(f).name}: {e}")
            continue
        df["スコア"] = pd.to_numeric(df["スコア"], errors="coerce")
        df = df.dropna(subset=["馬名", "スコア", "日付"])
        df["日付"] = df["日付"].astype(str).str[:8]
        rows.append(df[["日付", "馬名", "スコア", "レースID"]])
    if not rows:
        print("no data")
        return
    hist = pd.concat(rows, ignore_index=True)
    hist = hist.drop_duplicates(subset=["日付", "馬名", "レースID"])
    hist = hist.sort_values(["馬名", "日付"]).reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    try:
        hist.to_parquet(OUT, index=False)
    except Exception:
        OUT_CSV = OUT.with_suffix(".csv")
        hist.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
        print(f"parquet失敗 → csv保存: {OUT_CSV}")
    print(f"saved: {OUT}  rows={len(hist):,}  horses={hist['馬名'].nunique():,}")


if __name__ == "__main__":
    main()
