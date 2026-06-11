"""
build_chaos_quantiles.py
========================
混戦度(field_chaos_score=正規化エントロピー)が 0.80-1.0 に圧縮され判別力が無い問題の対策。
過去 bundle 全レースの chaos 生値から 101 分位テーブルを作り、data/chaos_quantiles.json に保存。
nicegui_app.py がこれで生値→パーセンタイル(0-1)変換し、相対的な堅さ/荒れを均等表示する。

top1_dominance / top2_concentration も同様に圧縮しているので分位テーブルを作る。
"""
import argparse
import json, glob
from pathlib import Path
import numpy as np

BASE = Path(__file__).parent
OUT = BASE / "data/chaos_quantiles.json"


def walk(o, key, out):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == key and isinstance(v, (int, float)): out.append(float(v))
            else: walk(v, key, out)
    elif isinstance(o, list):
        for x in o: walk(x, key, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-dir", default=None,
                    help="bundle 走査ディレクトリ (default: reports/cowork_input)。"
                         "補正復活後の分布で再生成する場合は rebuild_quantile_bundles.py "
                         "の出力 reports/_requant_bundles を指定")
    args = ap.parse_args()
    bundle_dir = Path(args.bundle_dir) if args.bundle_dir else (BASE / "reports/cowork_input")

    keys = ["field_chaos_score", "top1_dominance", "top2_concentration"]
    coll = {k: [] for k in keys}
    files = glob.glob(str(bundle_dir / "*_bundle.json"))
    print(f"[scan] {bundle_dir} -> {len(files)} bundles")
    for f in files:
        try: d = json.load(open(f, encoding="utf-8"))
        except Exception: continue
        for k in keys: walk(d, k, coll[k])

    tables = {}
    for k, vals in coll.items():
        a = np.array(vals, dtype=float)
        if len(a) < 20:
            print(f"  {k}: サンプル不足 ({len(a)}) スキップ"); continue
        tables[k] = [float(np.percentile(a, p)) for p in range(101)]
        print(f"  {k}: n={len(a)}  min={a.min():.3f} 中央={np.median(a):.3f} max={a.max():.3f}")

    OUT.write_text(json.dumps({"n_bundles": len(files), "quantiles": tables},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
