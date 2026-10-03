# -*- coding: utf-8 -*-
"""
replay_serve_2026.py — 2026 年の過去開催を「現在の配信経路」で作り直し、当時実際に配信した
bundle と同じ物差しで比べる (2026 年の成績は実力か配信故障かの切り分け)。

- 現在の export_weekly_marks.py をそのまま呼ぶ。出力は --out で指定した作業ディレクトリのみ
  (reports/cowork_input は読み取りだけ)。
- eligibility coverage gate (障害判定の証拠ファイル、2026-09 導入) は過去日に証拠が無いため
  この再生に限り素通しにし、対象は「当時の bundle に載っていたレース」に固定する
  (障害判定は当時の配信に従う)。
- 履歴特徴は serve_history_feats の as-of (date < レース日) なので未来の成績は混ざらない。
  ただし騎手/調教師コード表・level_norms 等の補助ファイルは現在版 (記述上の限界)。

実行: python -m analysis.replay_serve_2026 run --out <dir> [--since 20260101] [--until 20260919]
      python -m analysis.replay_serve_2026 report --out <dir>
"""
from __future__ import annotations
import argparse, json, os, re, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
SERVED = BASE / "reports" / "cowork_input"

_CHILD = r"""
import sys, runpy
sys.path.insert(0, r'{base}')
import eligibility_coverage_gate as g
class _Pass:
    ok = True; expected = 0; flat_rids = []; jump_rids = []
    def as_dict(self): return {{"replay": True}}
g.check_coverage = lambda *a, **k: _Pass()
import json, re, race_eligibility as re_
_b = json.load(open(r'{served}', encoding='utf-8'))
_rs = _b['races'] if isinstance(_b['races'], list) else list(_b['races'].values())
_served = {{re.sub(r'\D', '', str(r.get('race_id', '')))[:16] for r in _rs}}
_orig = re_.evaluate_race
def _ev(rid, *a, **k):
    el = dict(_orig(rid, *a, **k))
    el['prediction_eligible'] = re.sub(r'\D', '', str(rid))[:16] in _served   # 当時配信したレースだけ再生
    return el
re_.evaluate_race = _ev
if {nogate}:
    g.AtomicPublish.abort = g.AtomicPublish.commit   # 品質ゲート不合格でも作業ディレクトリへは書き出す (入力欠落日の記述用)
if {trunc}:
    import pandas as _pd
    _rp = _pd.read_parquet
    def _read(path, *a, **k):
        df = _rp(path, *a, **k)
        if str(path).endswith('_horse_history.parquet'):
            df = df[df['date'].astype(int) < {date}].copy()      # 当日朝より後の行を物理的に落とす
        return df
    _pd.read_parquet = _read
sys.argv = ['export_weekly_marks.py', '--csv', r'{csv}', '--shap-topk', '0', '--out-dir', r'{out}']
runpy.run_path(r'{base}\export_weekly_marks.py', run_name='__main__')
"""


def dates(since, until):
    out = []
    for p in sorted(SERVED.glob("2026????_bundle.json")):
        d = p.name[:8]
        if since <= d <= until and (BASE / "data" / "weekly" / f"{d}.csv").exists() \
                and (BASE / "data" / "kekka" / f"{d}.csv").exists():
            out.append(d)
    return out


def cmd_run(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for d in dates(a.since, a.until):
        if (out / f"{d}_bundle.json").exists():
            continue
        code = _CHILD.format(base=str(BASE), csv=str(BASE / "data" / "weekly" / f"{d}.csv"), out=str(out / d),
                             served=str(SERVED / f"{d}_bundle.json"), trunc=bool(a.truncate), date=int(d),
                             nogate=bool(a.ignore_gate))
        env = dict(os.environ, CB_HOSEI_PROXY="1" if a.proxy else "0", PYTHONIOENCODING="utf-8")
        r = subprocess.run([sys.executable, "-c", code], cwd=str(BASE), capture_output=True, env=env,
                           text=True, encoding="utf-8", errors="replace")
        (out / f"{d}.log").write_text((r.stdout or "") + (r.stderr or ""), encoding="utf-8")
        print(f"{d} rc={r.returncode} bundle={'yes' if (out / f'{d}_bundle.json').exists() else 'NO'}", flush=True)
    return 0


def served_record(d):
    """当時配信した bundle。開催日以前にコミットされた最新版を git から取る。
    戻り値 (bundle dict | None, 来歴ラベル)。開催日以前の版が無ければ None。"""
    rel = f"reports/cowork_input/{d}_bundle.json"
    log = subprocess.run(["git", "log", "--format=%H %cs", "--", rel], cwd=str(BASE),
                         capture_output=True, text=True).stdout.split("\n")
    log = [x.split() for x in log if x.strip()]
    pre = [h for h, cs in log if cs.replace("-", "") <= d]
    if not pre:
        return None, "開催日以前の記録なし"
    raw = subprocess.run(["git", "show", f"{pre[0]}:{rel}"], cwd=str(BASE), capture_output=True).stdout
    label = "当時版" if pre[0] == log[0][0] else "当時版(後日再生成あり→git の開催日以前版を使用)"
    return json.loads(raw.decode("utf-8")), label


def _races(path):
    b = path if isinstance(path, dict) else json.loads(Path(path).read_text(encoding="utf-8"))
    rs = b["races"] if isinstance(b["races"], list) else list(b["races"].values())
    out = {}
    for r in rs:
        rid = re.sub(r"\D", "", str(r.get("race_id", "")))[:16]
        hs = [h for h in r.get("horses", []) if h.get("p_win") is not None and h.get("umaban") is not None]
        if len(hs) >= 5:
            out[rid] = hs
    return out


def _odds(h):
    for k in ("tansho_odds", "odds", "tan_odds"):
        v = h.get(k)
        try:
            if v not in (None, "") and float(v) > 0:
                return float(v)
        except (TypeError, ValueError):
            pass
    return None


def _kekka(d):
    k = pd.read_csv(BASE / "data" / "kekka" / f"{d}.csv", encoding="cp932", low_memory=False)
    k["rid16"] = k["レースID(新)"].astype(str).str[:16]
    k["ban"] = pd.to_numeric(k["馬番"], errors="coerce")
    k["fin"] = pd.to_numeric(k["確定着順"], errors="coerce")
    k = k[k.fin.between(1, 3) & k.ban.notna()]
    return {rid: (set(g.ban.astype(int)), set(g[g.fin == 1].ban.astype(int))) for rid, g in k.groupby("rid16")}


PERIODS = [("v5配信期 0418-0516", "20260418", "20260516"),
           ("v6故障期 0517-0830", "20260517", "20260830"),
           ("移行期   0905-0919", "20260905", "20260919"),
           ("修正後   0920-0927", "20260920", "20260927"),
           (" うち直近 0926-0927", "20260926", "20260927")]


def cmd_report(a):
    out = Path(a.out); rows = []; miss = []
    for d in dates(a.since, a.until):
        rp = out / f"{d}_bundle.json"
        sb, label = served_record(d)
        if sb is None:
            miss.append((d, label, len(_races(SERVED / f"{d}_bundle.json")))); continue
        served = _races(sb)
        if not rp.exists():
            miss.append((d, "再生不能 (現行の品質ゲートで不合格)", len(served))); continue
        lg = out / f"{d}.log"
        gate_ng = lg.exists() and "品質ゲート不合格" in lg.read_text(encoding="utf-8", errors="replace")
        if gate_ng:
            label += " / 再生は入力欠落あり(現行ゲート不合格)"
        replay, kek = _races(rp), _kekka(d)
        model = sb.get("model")
        n_norep = n_nokek = n_noodds = 0
        for rid, hs in served.items():
            if rid not in replay: n_norep += 1; continue
            if rid not in kek: n_nokek += 1; continue
            od = [(h, _odds(h)) for h in hs]; od = [(h, o) for h, o in od if o]
            if len(od) < 5: n_noodds += 1; continue
            top3, win = kek[rid]
            fav = int(min(od, key=lambda t: t[1])[0]["umaban"])
            s1 = int(max(hs, key=lambda h: float(h["p_win"]))["umaban"])
            r1 = int(max(replay[rid], key=lambda h: float(h["p_win"]))["umaban"])
            rows.append(dict(date=d, model=model, gate_ng=gate_ng, same=int(s1 == r1),
                             s_agree=int(s1 == fav), r_agree=int(r1 == fav),
                             s_win=int(s1 in win), r_win=int(r1 in win), f_win=int(fav in win),
                             s_top3=int(s1 in top3), r_top3=int(r1 in top3), f_top3=int(fav in top3)))
        if n_norep or n_nokek or n_noodds or label != "当時版":
            miss.append((d, f"{label} / 再生に無い {n_norep}R・結果なし {n_nokek}R・オッズ無し {n_noodds}R", 0))
    df = pd.DataFrame(rows)
    if df.empty:
        print("比較できるレースなし"); return 1
    rng = np.random.default_rng(0)

    def ci(g, a_, b_):
        per = g.groupby("date").agg(n=("same", "size"), x=(a_, "sum"), y=(b_, "sum"))
        if len(per) < 2: return float("nan"), float("nan")
        idx = rng.integers(0, len(per), size=(4000, len(per)))
        diff = 100 * (per.x.values[idx].sum(1) - per.y.values[idx].sum(1)) / per.n.values[idx].sum(1)
        return np.percentile(diff, 2.5), np.percentile(diff, 97.5)

    def line(name, g):
        if g.empty: print(f"  {name:<22} n=0"); return
        lo, hi = ci(g, "r_top3", "s_top3"); lo2, hi2 = ci(g, "r_top3", "f_top3")
        print(f"  {name:<22} n={len(g):>4} 日={g.date.nunique():>2} | 1位同一 {100*g.same.mean():5.1f}% | 一致率 当時 {100*g.s_agree.mean():5.1f} 再生 {100*g.r_agree.mean():5.1f} "
              f"| 1着 当時 {100*g.s_win.mean():5.1f} 再生 {100*g.r_win.mean():5.1f} 市場 {100*g.f_win.mean():5.1f} "
              f"| top3 当時 {100*g.s_top3.mean():5.1f} 再生 {100*g.r_top3.mean():5.1f} 市場 {100*g.f_top3.mean():5.1f} "
              f"| 再生−当時 {100*(g.r_top3.mean()-g.s_top3.mean()):+5.1f} [{lo:+.1f},{hi:+.1f}] 再生−市場 {100*(g.r_top3.mean()-g.f_top3.mean()):+5.1f} [{lo2:+.1f},{hi2:+.1f}]")
    print("当時配信した印 vs 現在の配信経路で作り直した印 (同一レース集合・当時 bundle のオッズ・CI は開催日 bootstrap)")
    ok = df[~df.gate_ng]
    print("[A] 再生が現行ゲート合格の開催日のみ")
    for name, lo, hi in PERIODS:
        line(name, ok[(ok.date >= lo) & (ok.date <= hi)])
    line("全体", ok)
    if df.gate_ng.any():
        print("[B] 入力欠落日 (現行ゲート不合格のまま再生) — 参考")
        line("入力欠落日", df[df.gate_ng])
    print(f"\n当時記録が欠ける / 一部欠ける開催日 ({len(miss)}):")
    for d, why, n in miss:
        print(f"  {d} {why}" + (f" ({n}R 集計外)" if n else ""))
    return 0


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "report"):
        s = sp.add_parser(name)
        s.add_argument("--out", required=True)
        s.add_argument("--since", default="20260101"); s.add_argument("--until", default="20261231")
        s.add_argument("--proxy", action="store_true",
                       help="補正タイム推定を有効化 (CB_HOSEI_PROXY=1)。本番の週次フローはこの設定")
        s.add_argument("--ignore-gate", action="store_true",
                       help="現行の品質ゲートで不合格の日も作業ディレクトリへ書き出す (報告では別枠)")
        s.add_argument("--truncate", action="store_true",
                       help="馬履歴を開催日より前の行だけに物理的に切ってから再生 (as-of の実測検査用)")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    return cmd_run(a) if a.cmd == "run" else cmd_report(a)


if __name__ == "__main__":
    sys.exit(main())
