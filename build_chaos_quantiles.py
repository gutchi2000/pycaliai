"""
build_chaos_quantiles.py
========================
混戦度(field_chaos_score=正規化エントロピー)が 0.80-1.0 に圧縮され判別力が無い問題の対策。
過去 bundle 全レースの chaos 生値から 101 分位テーブルを作り、data/chaos_quantiles.json に保存。
nicegui_app.py がこれで生値→パーセンタイル(0-1)変換し、相対的な堅さ/荒れを均等表示する。

top1_dominance / top2_concentration も同様に圧縮しているので分位テーブルを作る。
"""
import argparse
import json
from datetime import datetime
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
    ap.add_argument("--out", default=str(OUT),
                    help="出力先。canary生成時は本番dataを上書きしないパスを指定")
    ap.add_argument("--reference-id", default=None,
                    help="凍結参照分布ID。本番policy用では必須")
    args = ap.parse_args()
    bundle_dir = Path(args.bundle_dir) if args.bundle_dir else (BASE / "reports/cowork_input")
    out_path = Path(args.out)

    keys = ["field_chaos_score", "top1_dominance", "top2_concentration"]
    coll = {k: [] for k in keys}
    files = sorted(bundle_dir.glob("*_bundle.json"))
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

    if not tables:
        raise SystemExit("有効な分位表を生成できない")
    reference_id = args.reference_id or (
        f"adhoc-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    payload = {
        "schema_version": 2,
        "reference_id": reference_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_bundle_dir": str(bundle_dir.resolve()),
        "n_bundles": len(files),
        "n_values": {k: len(v) for k, v in coll.items()},
        "quantiles": tables,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(out_path)
    print(f"[saved] {out_path} reference_id={reference_id}")


if __name__ == "__main__":
    main()
