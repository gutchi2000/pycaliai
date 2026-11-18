"""目的関数の検証用: 凍結 bundle の各開催日を現在の配信経路 (v6) で作り直し、
モデルに入る特徴行列 X (エンコード済み・-9999 補完済み) とスコアを保存する (分析専用)。

- export_weekly_marks.py をそのまま子プロセスで実行し、export_marks_json.export_race を包んで
  X を横取りする。出力 (bundle) は作業ディレクトリにだけ書く。本番の reports/cowork_input には触れない。
- 対象レースは凍結 bundle (reports/cowork_input_frozen) に載っていたレースに固定 (障害判定は当時に従う)。
- 履歴 parquet は当日より前の行だけに切る (analysis.replay_serve_2026 の --truncate と同じ)。
- 環境は現在の本番と同じ: CB_HOSEI_PROXY=1、CB_SERVE_RECOVER=1。品質ゲート不合格の日も書き出す。

使い方: python -m analysis.objective_check_replay --out <dir> [--since YYYYMMDD --until YYYYMMDD]
出力: <dir>/feats/{date}.parquet (rid, umaban, score, f0..fN), <dir>/feats/_feature_cols.json
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

BASE = Path("E:/PyCaLiAI")
FROZEN = BASE / "reports" / "cowork_input_frozen"

_CHILD = r"""
import sys, runpy, atexit, json, re
sys.path.insert(0, r'{base}')
import numpy as np, pandas as pd
import eligibility_coverage_gate as g
class _Pass:
    ok = True; expected = 0; flat_rids = []; jump_rids = []
    def as_dict(self): return {{"replay": True}}
g.check_coverage = lambda *a, **k: _Pass()
g.AtomicPublish.abort = g.AtomicPublish.commit
import race_eligibility as re_
_b = json.load(open(r'{served}', encoding='utf-8'))
_served = {{re.sub(r'\D', '', str(r.get('race_id', '')))[:16] for r in _b['races']}}
_orig_ev = re_.evaluate_race
def _ev(rid, *a, **k):
    el = dict(_orig_ev(rid, *a, **k))
    el['prediction_eligible'] = re.sub(r'\D', '', str(rid))[:16] in _served
    return el
re_.evaluate_race = _ev
_rp = pd.read_parquet
def _read(path, *a, **k):
    df = _rp(path, *a, **k)
    if str(path).endswith('_horse_history.parquet'):
        df = df[df['date'].astype(int) < {date}].copy()
    return df
pd.read_parquet = _read
import export_marks_json as emj
from backtest_pl_ev import COL_BAN
_orig_er = emj.export_race
_rows, _cols = [], {{}}
def _er(rid, g_orig, model, feats, encs, *a, **k):
    out = _orig_er(rid, g_orig, model, feats, encs, *a, **k)
    gg = g_orig.copy().sort_values(COL_BAN).reset_index(drop=True)
    for c, le in encs.items():
        if c not in gg.columns: continue
        v = gg[c].astype(str).fillna("__NaN__")
        v = v.where(v.isin(set(le.classes_)), "__NaN__")
        gg[c] = le.transform(v)
    X = gg[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
    sc = model.predict(X)
    _cols['feats'] = list(feats)
    for i in range(len(gg)):
        _rows.append([str(out['race_id']), int(gg[COL_BAN].iloc[i]), float(sc[i])] + [float(x) for x in X[i]])
    return out
emj.export_race = _er
def _dump():
    if not _rows: return
    n = len(_cols['feats'])
    df = pd.DataFrame(_rows, columns=['rid', 'umaban', 'score'] + [f'f{{i}}' for i in range(n)])
    df.to_parquet(r'{feat_out}', index=False)
    json.dump(_cols['feats'], open(r'{cols_out}', 'w', encoding='utf-8'), ensure_ascii=False)
atexit.register(_dump)
sys.argv = ['export_weekly_marks.py', '--csv', r'{csv}', '--shap-topk', '0', '--out-dir', r'{out}', '--model', 'v6']
runpy.run_path(r'{base}\export_weekly_marks.py', run_name='__main__')
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--since", default="20260418")
    ap.add_argument("--until", default="20261115")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    (out / "feats").mkdir(parents=True, exist_ok=True)
    (out / "bundles").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    dates = sorted(p.name[:8] for p in FROZEN.glob("2026????_bundle.json") if a.since <= p.name[:8] <= a.until)
    for d in dates:
        fo = out / "feats" / f"{d}.parquet"
        if fo.exists():
            continue
        csv = BASE / "data" / "weekly" / f"{d}.csv"
        if not csv.exists():
            print(d, "no weekly csv", flush=True)
            continue
        code = _CHILD.format(base=str(BASE), served=str(FROZEN / f"{d}_bundle.json"), date=int(d),
                             feat_out=str(fo), cols_out=str(out / "feats" / "_feature_cols.json"),
                             csv=str(csv), out=str(out / "bundles" / d))
        env = dict(os.environ, CB_HOSEI_PROXY="1", CB_SERVE_RECOVER="1", CB_ZURE_CONTRIB="0",
                   PYTHONIOENCODING="utf-8")
        with open(out / "logs" / f"{d}.log", "w", encoding="utf-8") as lf:
            r = subprocess.run([sys.executable, "-c", code], cwd=str(BASE), env=env, stdout=lf, stderr=subprocess.STDOUT)
        print(d, "rc", r.returncode, "ok" if fo.exists() else "NO FEATS", flush=True)


if __name__ == "__main__":
    main()
