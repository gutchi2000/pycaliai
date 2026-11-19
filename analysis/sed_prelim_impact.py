# -*- coding: utf-8 -*-
"""
sed_prelim_impact.py — JRDB 成績 (SED) の速報版／確定版のズレが配信の予測にどれだけ効くかを測る (2026-11-19、分析のみ)
=====================================================================================================
背景: JRDB は同じ日付の SED を開催当日の夕方に速報版 (着順・タイム・オッズはあるが JRDB の指数・補正値はすべて空)、
      翌週木曜に確定版で同名差し替えする。scripts/jrdb_fetch.py は既にあるファイルを取り直さないので、手元
      (= 配信の data/jrdb/sed.parquet) は速報版のまま残る (11/07・11/08・11/14)。

作る成績の表 (どれも data/jrdb/analysis/sed_prelim_impact/ = git 管理外。本番の sed.parquet には触れない):
  A  … 2026-11-14 07:45 の控え (snapshots/sed_20261114.parquet) = 11/14・11/15 の配信が実際に読んだ表 (11/07・11/08 は速報版)
  B  … A の 11/07・11/08 を確定版 (snapshots/Sed/files の差し替え後の版) に置き換えた表
  C  … B の 2026 年の土曜開催をすべて速報版の形 (行はあるが指数・補正値が空) にした仮想の表
        = 直さずに 1 年回したときの姿。土曜の成績は土曜 19:40 のタスクが速報版で取り、以後取り直されない
          (日曜の分は金曜 19:40 に確定版で取る) ので、放置すると土曜開催がこの形になる。
予測: 現在の export_weekly_marks.py を --out-dir 付きで子プロセス実行 (本番の出力先・凍結版・serve_path の控えには書かない)。
      成績の表は jrdb_features.SED を差し替えるだけ。他の入力 (出走表・履歴・KYI の発走前保存) は全アームで同じ。
      対象は当時配信したレース。履歴 parquet はレース日より前の行だけ (replay_serve_2026 と同じ)。
比べるもの: v8 (11/21〜の本番) で A/B/C、v7 (11/14〜15 の本番) で A/B。A は同じ条件で 2 回走らせて再現性を確認。

実行 (root から):
  python -m analysis.sed_prelim_impact build
  python -m analysis.sed_prelim_impact run
  python -m analysis.sed_prelim_impact report     # → docs/research/SED_PRELIM_IMPACT_20261119.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("PYCALIAI_DATA_ROOT", "E:/PyCaLiAI"))
sys.path.insert(0, str(BASE / "scripts"))
import jrdb_parse as jp  # noqa: E402

SNAP = ROOT / "data/jrdb/snapshots"
WORK = ROOT / "data/jrdb/analysis/sed_prelim_impact"
SERVED = ROOT / "reports/cowork_input"
DATES = ["20261114", "20261115"]
PRELIM_DAYS = ["20261107", "20261108"]
# 速報版で空になっている項目 (11/07・11/08 の速報版と確定版の比較で、速報版の空欄率 100%)
PRELIM_BLANK = ["idm", "soten", "babasa", "pace", "deokure", "ichidori", "furi", "mae_furi", "naka_furi", "ato_furi", "race_adj",
                "course_dori", "joushou", "class_cd", "batai_cd", "kehai_cd", "race_pace", "uma_pace", "ten_idx", "agari_idx",
                "pace_idx", "racep_idx", "yon_course"]
OUT_JSON = BASE / "docs/research/SED_PRELIM_IMPACT_20261119.json"
ARMS = [("v8", "A"), ("v8", "B"), ("v8", "C"), ("v8", "A2"), ("v7", "A"), ("v7", "B"), ("v8", "C4"), ("v8", "C8")]
# C4 / C8 = 直近 4 週 / 8 週の土曜だけを速報版の形にした表 (放置して 4 週後・8 週後に相当)。C = 2026 年の土曜すべて
C_FROM = {"C4": 20261017, "C8": 20260919, "C": 20260101}
PQ = {"A": SNAP / "sed_20261114.parquet", "A2": SNAP / "sed_20261114.parquet",
      "B": WORK / "sed_B_final.parquet", "C": WORK / "sed_C_saturday_prelim.parquet",
      "C4": WORK / "sed_C4_saturday_prelim.parquet", "C8": WORK / "sed_C8_saturday_prelim.parquet"}


def parse_zip(path: Path) -> pd.DataFrame:
    """jrdb_parse.parse_type と同じ読み方で 1 本の zip を表にする。"""
    reclen, spec = jp.SPEC["Sed"]
    rows = []
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if not name.lower().endswith(".txt"):
                continue
            for line in z.read(name).split(b"\n"):
                line = line.rstrip(b"\r")
                if line.strip() and len(line) == reclen - 2:
                    rows.append(tuple(line[s - 1:s - 1 + n] for _, s, n, _ in spec) + (name,))
    df = pd.DataFrame(rows, columns=[c for c, *_ in spec] + ["src"])
    for c, _, _, t in spec:
        s = df[c].map(lambda b: b.decode("cp932", errors="replace").strip())
        df[c] = pd.to_numeric(s, errors="coerce") if t == "n" else (s.where(s != "", None) if t == "c" else s)
    df["src"] = df["src"].astype(str)
    return df.drop_duplicates(["racekey", "ban"], keep="last").reset_index(drop=True)


def versions(name):
    ms = [json.loads(x) for x in (SNAP / "Sed/manifest.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    return [m for m in ms if m.get("kind") == "file" and m["name"] == name]


def cmd_build(a):
    WORK.mkdir(parents=True, exist_ok=True)
    A = pd.read_parquet(PQ["A"])
    info = {"A_rows": len(A), "A_max_ymd": int(A.ymd.max())}
    parts = [A[~A.ymd.astype(int).astype(str).isin(PRELIM_DAYS)]]
    for d in PRELIM_DAYS:
        vs = versions(f"SED{d[2:]}.zip")
        first, last = parse_zip(SNAP / "Sed" / vs[0]["file"]), parse_zip(SNAP / "Sed" / vs[-1]["file"])
        # 控えの最初の版 (手元の速報版) を同じ読み方で読むと A の行と一致すること = 読み方の確認
        a_d = A[A.ymd.astype(int).astype(str) == d].sort_values(["racekey", "ban"]).reset_index(drop=True)
        f_d = first.sort_values(["racekey", "ban"]).reset_index(drop=True)
        cols = [c for c in a_d.columns if c != "src"]
        same = len(a_d) == len(f_d) and all(
            ((a_d[c].astype(str) == f_d[c].astype(a_d[c].dtype).astype(str)) | (a_d[c].isna() & f_d[c].isna())).all() for c in cols)
        info[d] = {"prelim_rows": len(first), "final_rows": len(last), "prelim_equals_A": bool(same),
                   "prelim_idm_na": float(first.idm.isna().mean()), "final_idm_na": float(last.idm.isna().mean()),
                   "final_last_modified": vs[-1].get("last_modified")}
        assert same, f"{d}: 控えの速報版と A が一致しない"
        parts.append(last[A.columns].astype(A.dtypes.to_dict()))
    B = pd.concat(parts, ignore_index=True)
    B.to_parquet(PQ["B"], index=False)
    info["B_rows"] = len(B)
    dt = pd.to_datetime(B.ymd.astype(int).astype(str), format="%Y%m%d")
    for arm, frm in C_FROM.items():
        C = B.copy()
        sat = (B.ymd.astype(int) >= frm) & (dt.dt.dayofweek == 5)
        for c in PRELIM_BLANK:
            C.loc[sat, c] = np.nan if pd.api.types.is_numeric_dtype(C[c]) else None
        C.to_parquet(PQ[arm], index=False)
        info[arm] = {"from": frm, "saturday_rows_blanked": int(sat.sum()), "saturday_days": int(dt[sat].dt.date.nunique())}
    (WORK / "build.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=1))
    return 0


_CHILD = r"""
import sys, runpy
sys.path.insert(0, r'{base}')
import jrdb_features as _jf
from pathlib import Path as _P
_jf.SED = _P(r'{sed}')
import bundle_provenance as _bp
_bp.snapshot_serve_inputs = lambda *a, **k: {{"skipped": "sed_prelim_impact"}}
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
    el['prediction_eligible'] = re.sub(r'\D', '', str(rid))[:16] in _served
    return el
re_.evaluate_race = _ev
g.AtomicPublish.abort = g.AtomicPublish.commit
import pandas as _pd
_rp = _pd.read_parquet
def _read(path, *a, **k):
    df = _rp(path, *a, **k)
    if str(path).endswith('_horse_history.parquet'):
        df = df[df['date'].astype(int) < {date}].copy()
    return df
_pd.read_parquet = _read
sys.argv = ['export_weekly_marks.py', '--csv', r'{csv}', '--shap-topk', '0', '--out-dir', r'{out}', '--model', '{model}']
runpy.run_path(r'{base}\export_weekly_marks.py', run_name='__main__')
"""


def run_dir(model, arm):
    return WORK / "run" / f"{model}_{arm}"


def cmd_run(a):
    for model, arm in ARMS:
        for d in DATES:
            od = run_dir(model, arm)
            od.mkdir(parents=True, exist_ok=True)
            if (od / f"{d}_bundle.json").exists():
                continue
            code = _CHILD.format(base=str(ROOT), sed=str(PQ[arm]), served=str(SERVED / f"{d}_bundle.json"), date=int(d),
                                 csv=str(ROOT / "data/weekly" / f"{d}.csv"), out=str(od / d), model=model)
            env = dict(os.environ, CB_HOSEI_PROXY="1", CB_SERVE_RECOVER="1", CB_ZURE_CONTRIB="0", PYTHONIOENCODING="utf-8")
            r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), capture_output=True, env=env,
                               text=True, encoding="utf-8", errors="replace")
            (od / f"{d}.log").write_text((r.stdout or "") + (r.stderr or ""), encoding="utf-8")
            jl = [l for l in (r.stdout + r.stderr).splitlines() if "[jrdb]" in l]
            print(f"{model} {arm} {d} rc={r.returncode} bundle={'yes' if (od / f'{d}_bundle.json').exists() else 'NO'} "
                  f"{jl[-1].split('[jrdb]')[-1].strip() if jl else ''}", flush=True)
    return 0


MARKS = ["◎", "〇", "▲", "△"]


def races(path):
    b = json.loads(Path(path).read_text(encoding="utf-8"))
    rs = b["races"] if isinstance(b["races"], list) else list(b["races"].values())
    out = {}
    for r in rs:
        rid = re.sub(r"\D", "", str(r.get("race_id", "")))[:16]
        hs = {int(h["umaban"]): h for h in r.get("horses", []) if h.get("umaban") is not None and h.get("p_win") is not None}
        out[rid] = (hs, r)
    return out


def kekka(d):
    k = pd.read_csv(ROOT / "data/kekka" / f"{d}.csv", encoding="cp932", low_memory=False)
    k["rid16"] = k["レースID(新)"].astype(str).str[:16]
    k["ban"] = pd.to_numeric(k["馬番"], errors="coerce")
    k["fin"] = pd.to_numeric(k["確定着順"], errors="coerce")
    k = k[k.fin.between(1, 3) & k.ban.notna()]
    return {rid: (set(g.ban.astype(int)), set(g[g.fin == 1].ban.astype(int))) for rid, g in k.groupby("rid16")}


def mark_of(h):
    m = h.get("mark") or ""
    return m if m in MARKS else ""


def compare(model, base_arm, alt_arm):
    """base (確定版の側) と alt を同じレース・同じ馬で比べる。差は alt − base。"""
    dp, rows = [], []
    per_date = {}
    for d in DATES:
        pb, pa = run_dir(model, base_arm) / f"{d}_bundle.json", run_dir(model, alt_arm) / f"{d}_bundle.json"
        if not (pb.exists() and pa.exists()):
            continue
        rb, ra, kk = races(pb), races(pa), kekka(d)
        n_r = ch_top = ch_hon = ch_marks = ch_markset = 0
        hit = {"base": [0, 0], "alt": [0, 0]}
        for rid in sorted(set(rb) & set(ra)):
            hb, _ = rb[rid]
            ha, _ = ra[rid]
            common = sorted(set(hb) & set(ha))
            if len(common) < 5:
                continue
            n_r += 1
            for u in common:
                d_ = (float(ha[u]["p_win"]) - float(hb[u]["p_win"])) * 100
                dp.append(d_)
                rows.append({"date": d, "rid": rid, "u": u, "dp": d_, "rank_b": hb[u].get("ai_rank"), "rank_a": ha[u].get("ai_rank"),
                             "mark_b": mark_of(hb[u]), "mark_a": mark_of(ha[u])})
            top = lambda hs: max(common, key=lambda u: float(hs[u]["p_win"]))  # noqa: E731
            hon = lambda hs: next((u for u in common if mark_of(hs[u]) == "◎"), None)  # noqa: E731
            ch_top += top(hb) != top(ha)
            ch_hon += hon(hb) != hon(ha)
            mb = {u: mark_of(hb[u]) for u in common if mark_of(hb[u])}
            ma = {u: mark_of(ha[u]) for u in common if mark_of(ha[u])}
            ch_marks += mb != ma
            ch_markset += set(mb) != set(ma)
            if rid in kk:
                t3, w = kk[rid]
                for key, hs in (("base", hb), ("alt", ha)):
                    h_ = hon(hs)
                    if h_ is not None:
                        hit[key][0] += h_ in w
                        hit[key][1] += h_ in t3
        per_date[d] = {"races": n_r, "ai1_changed": ch_top, "hon_changed": ch_hon, "any_mark_changed": ch_marks,
                       "marked_set_changed": ch_markset, "hon_win_top3_base": hit["base"], "hon_win_top3_alt": hit["alt"]}
    df = pd.DataFrame(rows)
    if df.empty:
        return {"error": "bundle が足りない"}
    a = df.dp.abs()
    rk = df.dropna(subset=["rank_b", "rank_a"])
    tot = {k: int(sum(v[k] for v in per_date.values())) for k in ("races", "ai1_changed", "hon_changed", "any_mark_changed", "marked_set_changed")}
    return {"base": base_arm, "alt": alt_arm, "model": model, "horses": int(len(df)),
            "share_changed": float((a > 1e-9).mean()),
            "share_abs_ge": {t: float((a >= t).mean()) for t in (0.1, 0.5, 1.0, 2.0)},
            "dp_quantiles_pt": {q: float(np.quantile(df.dp, q)) for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)},
            "abs_dp_mean_pt": float(a.mean()), "abs_dp_max_pt": float(a.max()),
            "abs_dp_mean_changed_pt": float(a[a > 1e-9].mean()) if (a > 1e-9).any() else 0.0,
            "rank_changed_share": float((rk.rank_b != rk.rank_a).mean()), "per_date": per_date, "totals": tot}


def side_checks():
    """学習側が確定版か (年別・日別の空欄率、速報版の日、その走を 1〜3 走前に持つ出走の割合) と、放置したときの露出の見積もり。"""
    d = pd.read_parquet(ROOT / "data/jrdb/sed.parquet", columns=["ped", "ymd", "idm", "class_cd", "src"])
    d["yr"] = d.ymd // 10000
    by_year = {int(y): {"rows": int(len(g)), "idm_na": round(float(g.idm.isna().mean()), 4), "class_na": round(float(g.class_cd.isna().mean()), 4)}
               for y, g in d.groupby("yr")}
    day = d.groupby("ymd").agg(rows=("idm", "size"), idm_na=("idm", lambda s: s.isna().mean()), src=("src", "first"))
    pre_days = day[day.idm_na > 0.2]
    pre = set(pre_days.index.astype(int))
    s = d.sort_values(["ped", "ymd"]).reset_index(drop=True)
    s["pre"] = s.ymd.isin(pre)
    g = s.groupby("ped")["pre"]
    sh = [g.shift(k).astype("boolean").fillna(False).astype(bool).values for k in (1, 2, 3)]
    s["any13"], s["prev1"] = sh[0] | sh[1] | sh[2], sh[0]
    s["period"] = pd.cut(s.ymd, [0, 20151231, 20221231, 20231231, 20251231, 20991231],
                         labels=["~2015", "train_2016_22", "valid_2023", "test_2024_25", "2026"])
    expo = {str(k): {"runs": int(len(v)), "any13_pct": round(100 * float(v.any13.mean()), 3), "prev1_pct": round(100 * float(v.prev1.mean()), 3)}
            for k, v in s.groupby("period", observed=True)}
    # 放置したときの見積もり: 2025-09〜12 の各出走で、直近 w 週の土曜の走を速報版とみなしたときの割合
    q = d[(d.ymd >= 20230101) & (d.ymd <= 20251231)][["ped", "ymd"]].sort_values(["ped", "ymd"]).reset_index(drop=True)
    q["dt"] = pd.to_datetime(q.ymd.astype(str), format="%Y%m%d")
    gg = q.groupby("ped")["dt"]
    for k in (1, 2, 3):
        q[f"p{k}"] = gg.shift(k)
    tg = q[q.ymd >= 20250901]
    proj = {}
    for w in (1, 2, 3, 4, 6, 8, 13, 26, 52):
        cut = tg.dt - pd.Timedelta(days=7 * w)
        f = [tg[f"p{k}"].notna() & (tg[f"p{k}"].dt.dayofweek == 5) & (tg[f"p{k}"] >= cut) for k in (1, 2, 3)]
        proj[w] = {"prev1_prelim_pct": round(100 * float(f[0].mean()), 1), "any13_prelim_pct": round(100 * float((f[0] | f[1] | f[2]).mean()), 1)}
    return {"by_year": by_year, "prelim_days": {int(k): {"rows": int(v.rows), "src": v.src} for k, v in pre_days.iterrows()},
            "exposure_by_period": expo, "projection_weeks": proj, "has_prev_run_pct": round(100 * float(tg.p1.notna().mean()), 1)}


def cmd_report(a):
    info = json.loads((WORK / "build.json").read_text(encoding="utf-8"))
    res = {"build": info, "comparisons": {}}
    for model, base, alt in [("v8", "A", "A2"), ("v8", "B", "A"), ("v7", "B", "A"), ("v8", "B", "C4"), ("v8", "B", "C8"), ("v8", "B", "C")]:
        res["comparisons"][f"{model}:{alt}-{base}"] = compare(model, base, alt)
    # 配信が読んだ表 A での「速報版の走を 1〜3 走前に持つ馬」の割合 (特徴の側の露出)
    res["coverage_logs"] = {}
    for model, arm in ARMS:
        for d in DATES:
            lp = run_dir(model, arm) / f"{d}.log"
            if lp.exists():
                jl = [l.split("[jrdb]")[-1].strip() for l in lp.read_text(encoding="utf-8").splitlines() if "[jrdb]" in l]
                res["coverage_logs"][f"{model}_{arm}_{d}"] = jl[-1] if jl else None
    res["side"] = side_checks()
    OUT_JSON.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, v in res["comparisons"].items():
        if "error" in v:
            print(k, v["error"])
            continue
        print(f"{k}: 馬 {v['horses']}  変わる馬 {100 * v['share_changed']:.1f}%  |Δ|≥0.5pt {100 * v['share_abs_ge'][0.5]:.1f}%  "
              f"|Δ|≥1pt {100 * v['share_abs_ge'][1.0]:.1f}%  平均|Δ| {v['abs_dp_mean_pt']:.3f}pt  最大 {v['abs_dp_max_pt']:.2f}pt  "
              f"順位変化 {100 * v['rank_changed_share']:.1f}%  {v['totals']}")
    return 0


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build").set_defaults(fn=cmd_build)
    sub.add_parser("run").set_defaults(fn=cmd_run)
    sub.add_parser("report").set_defaults(fn=cmd_report)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
