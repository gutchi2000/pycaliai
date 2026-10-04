# -*- coding: utf-8 -*-
"""
serve_repro_invariance.py — 配信再現性の実データ検査 (T1〜T4) と段階比較 (§6)
==============================================================================
仕様: docs/research/SERVE_REPRODUCIBILITY_FIX_SPEC_20261004.md §5 / §6
合成データの同じ検査は tests/test_serve_repro_invariance.py。

段階 (stage) ごとの serve_history_feats.py を --module-dir に置き、その stage の履歴 parquet と
コード表を --module-dir/data/ に置く (本番の data/ は読むだけで、書かない)。

  features  : 開催日ごとに、出走表 (data/weekly) から作った frame に履歴特徴を埋め、
              全履歴 / 開催日より前に切り落とした履歴 (T1・T3) / 行を並べ替えた履歴と出走表 (T2)
              で一致を数える。全履歴版の特徴と同定状態を後の段階比較用に保存する。
  bundles   : export_weekly_marks を子プロセスで動かし bundle を作る (全履歴 / 切り落とし /
              並べ替え)。本番の reports/cowork_input は読むだけ。
  t4        : bundles の条件間で p_win と印が完全一致するかを数える (T4)。
  stagediff : 2 つの段階の features / bundles を比べ、§6 の数値を出す。
  hybrid    : 2024〜25 年 (出走表 CSV が無い年) は master の学習時特徴のうち履歴特徴だけを
              配信経路の値に差し替えて採点し、段階間で比べる (出走表由来の特徴は再現できない)。

実行は本番 root から: python -m analysis.serve_repro_invariance <cmd> ...
"""
from __future__ import annotations

import argparse
import ctypes
import gzip
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# データと未変更 module の置き場所 (本番 root)。worktree から動かすときは --root で本番を指す。
BASE = Path(os.environ.get("SERVE_REPRO_ROOT") or Path(__file__).resolve().parents[1])
SERVED = BASE / "reports" / "cowork_input"
ROLL4 = ["jockey_fuku30", "jockey_fuku90", "trainer_fuku30", "trainer_fuku90"]


def set_root(root: str | None) -> None:
    global BASE, SERVED, MASTER
    if root:
        BASE = Path(root).resolve()
        SERVED = BASE / "reports" / "cowork_input"
        MASTER = BASE / "data" / "master_v2_20130105-20251228.csv"
    if str(BASE) not in sys.path:
        sys.path.insert(0, str(BASE))


def lower_priority() -> None:
    """開催中の本番ジョブ (T-10 など) より後回しにする (Windows BELOW_NORMAL)。"""
    try:
        k32 = ctypes.windll.kernel32
        k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00004000)
    except Exception:
        pass


# ------------------------------------------------------------------ 共通
def load_stage_module(module_dir: Path):
    path = Path(module_dir) / "serve_history_feats.py"
    tag = hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:10]
    spec = importlib.util.spec_from_file_location(f"shf_stage_{tag}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ensure_date(df: pd.DataFrame) -> pd.DataFrame:
    if "日付" in df.columns and df["日付"].astype(str).str.len().min() == 8:
        return df
    df = df.copy()
    df["日付"] = df["日付S"].map(lambda s: "{:04d}{:02d}{:02d}".format(*[int(x) for x in str(s).split(".")]))
    return df


def weekly_frame(d: str, cache: Path) -> pd.DataFrame:
    """export_weekly_marks と同じく parse_csv → 日付列。開催日ごとに pickle で再利用。"""
    p = cache / f"weekly_{d}.pkl"
    if p.exists():
        return pd.read_pickle(p)
    from predict_weekly import parse_csv
    df = _ensure_date(parse_csv(BASE / "data" / "weekly" / f"{d}.csv"))
    cache.mkdir(parents=True, exist_ok=True)
    df.to_pickle(p)
    return df


def row_keys(df: pd.DataFrame) -> pd.MultiIndex:
    from backtest_pl_ev import COL_BAN, COL_RID
    rid = df[COL_RID].astype(str).str.replace(r"\D", "", regex=True).str[:16]
    ban = pd.to_numeric(df[COL_BAN], errors="coerce").fillna(-1).astype(int)
    return pd.MultiIndex.from_arrays([rid, ban], names=["rid", "umaban"])


def make_base(work: Path, tag: str, hist: pd.DataFrame, maps: Path) -> Path:
    base = work / "bases" / tag
    (base / "data").mkdir(parents=True, exist_ok=True)
    hist.to_parquet(base / "data" / "_horse_history.parquet", index=False)
    (base / "data" / "serve_code_maps.json").write_bytes(Path(maps).read_bytes())
    return base


def identify(mod, idx, name: str, sire: str, birth_year, race_date: int):
    import inspect
    if "race_date" in inspect.signature(idx.resolve).parameters:
        ent, status = idx.resolve(name, sire, birth_year, race_date=race_date)
    else:
        ent, status = idx.resolve(name, sire, birth_year)
    return status, (None if ent is None else int(ent["ped_id"]))


def compute(mod, base: Path, df: pd.DataFrame, d: int) -> pd.DataFrame:
    """履歴特徴 (NUM_FEATS + CAT_FEATS) と同定 (状態・個体 ID) を行キーで返す。"""
    work = df.copy()
    mod.fill_history_features(work, base=base)
    out = work[mod.NUM_FEATS + mod.CAT_FEATS].copy()
    idx = mod._CACHE[str(base)][0]
    names = work["馬名"].map(mod._clean_name)
    sires = (work["種牡馬"].map(mod._clean_name) if "種牡馬" in work.columns
             else pd.Series([""] * len(work), index=work.index))
    ages = pd.to_numeric(work.get("年齢"), errors="coerce")
    st, pid = [], []
    for ix in work.index:
        by = int(d // 10000 - ages.loc[ix]) if pd.notna(ages.loc[ix]) else None
        s, p = identify(mod, idx, names.loc[ix], sires.loc[ix], by, d)
        st.append(s)
        pid.append(p)
    out["identity_status"] = st
    out["identity_ped_id"] = pd.array(pid, dtype="Int64")
    out["horse_name"] = work["馬名"].astype(str).to_numpy()
    out.index = row_keys(work)
    return out.sort_index()


def diff_rows(a: pd.DataFrame, b: pd.DataFrame, cols: list[str]) -> pd.Series:
    """行ごとに cols のどれかが違うか (欠損同士は一致)。"""
    a, b = a.loc[b.index.intersection(a.index)], b.loc[a.index.intersection(b.index)]
    a, b = a.sort_index(), b.sort_index()
    bad = pd.Series(False, index=a.index)
    for c in cols:
        x, y = a[c], b[c]
        if x.dtype == object or y.dtype == object or str(x.dtype) in ("string", "str"):
            bad |= x.astype(str).to_numpy() != y.astype(str).to_numpy()
        else:
            xv, yv = x.to_numpy(dtype=float), y.to_numpy(dtype=float)
            bad |= ~((xv == yv) | (np.isnan(xv) & np.isnan(yv)))
    return bad


def max_abs(a: pd.DataFrame, b: pd.DataFrame, cols: list[str]) -> float:
    a, b = a.sort_index(), b.loc[a.index].sort_index()
    m = 0.0
    for c in cols:
        if a[c].dtype == object:
            continue
        xv, yv = a[c].to_numpy(dtype=float), b[c].to_numpy(dtype=float)
        both = ~np.isnan(xv) & ~np.isnan(yv)
        if both.any():
            m = max(m, float(np.max(np.abs(xv[both] - yv[both]))))
    return m


def compare(mod, ref: pd.DataFrame, got: pd.DataFrame) -> dict:
    other = [c for c in mod.NUM_FEATS if c not in ROLL4]
    common = ref.index.intersection(got.index)
    ref, got = ref.loc[common], got.loc[common]
    roll = diff_rows(ref, got, ROLL4)
    horse = diff_rows(ref, got, other)
    codes = diff_rows(ref, got, mod.CAT_FEATS)
    ident = diff_rows(ref, got, ["identity_status", "identity_ped_id"])
    anyd = roll | horse | codes | ident
    ex = ref.loc[ident[ident].index, ["horse_name", "identity_status", "identity_ped_id"]].copy()
    ex["other_status"] = got.loc[ex.index, "identity_status"].to_numpy()
    ex["other_ped_id"] = got.loc[ex.index, "identity_ped_id"].astype(str).to_numpy()
    return {"n": int(len(common)), "any": int(anyd.sum()), "rolling4": int(roll.sum()),
            "horse_feats": int(horse.sum()), "codes": int(codes.sum()), "identity": int(ident.sum()),
            "rolling4_max_abs": round(max_abs(ref, got, ROLL4), 6),
            "horse_feats_max_abs": round(max_abs(ref, got, other), 6),
            "identity_examples": [
                {"rid": str(i[0]), "umaban": int(i[1]), "name": r.horse_name,
                 "ref": [r.identity_status, None if pd.isna(r.identity_ped_id) else int(r.identity_ped_id)],
                 "other": [r.other_status, r.other_ped_id]}
                for i, r in ex.head(20).iterrows()]}


def history_checks(mod, hist: pd.DataFrame, work: Path, maps: Path, df: pd.DataFrame) -> dict:
    """T5 (順序キーの重複) と T6 (列不足の履歴でエラー停止するか)。"""
    out = {}
    k = ["date", "post_time", "race_id", "umaban"]
    if all(c in hist.columns for c in k):
        out["T5_order_key_duplicates"] = int(hist.duplicated(k).sum())
        out["T5_missing_post_time"] = int(hist["post_time"].isna().sum())
    else:
        out["T5_order_key_duplicates"] = None
        out["T5_columns_present"] = sorted(c for c in k if c in hist.columns)
    for drop in ("post_time", "umaban"):
        if drop not in hist.columns:
            out[f"T6_without_{drop}"] = "column absent in this history"
            continue
        base = make_base(work, f"t6_no_{drop}", hist.drop(columns=[drop]).head(5000), maps)
        mod._CACHE.pop(str(base), None)
        try:
            mod.fill_history_features(df.head(20).copy(), base=base)
            out[f"T6_without_{drop}"] = "no error (silent fallback)"
        except Exception as exc:          # noqa: BLE001 — 例外の型と文面をそのまま記録する
            out[f"T6_without_{drop}"] = f"error: {type(exc).__name__}: {str(exc)[:160]}"
        mod._CACHE.pop(str(base), None)
    return out


# ------------------------------------------------------------------ features (T1・T2・T3)
def cmd_features(a) -> int:
    lower_priority()
    mod = load_stage_module(Path(a.module_dir))
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    work = Path(a.work)
    hist_path = Path(a.module_dir) / "data" / "_horse_history.parquet"
    maps = Path(a.module_dir) / "data" / "serve_code_maps.json"
    hist = pd.read_parquet(hist_path)
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()] if a.seeds else []
    summary = {"module": str(Path(a.module_dir) / "serve_history_feats.py"),
               "serve_path_version": getattr(mod, "SERVE_PATH_VERSION", "sp-20261004-0"),
               "history_sha256": hashlib.sha256(hist_path.read_bytes()).hexdigest(),
               "history_rows": int(len(hist)), "history_max_date": int(hist["date"].max()),
               "dates": {}}
    dates = [x for x in a.dates.split(",") if x]
    frames = {d: weekly_frame(d, work / "weekly_cache") for d in dates}
    # 索引は 1 つ 20 秒・数 GB なので、同時に持つのは全履歴版ともう 1 つだけにする。
    full_base = make_base(work, "full", hist, maps)
    fulls = {}
    for d in dates:
        fulls[d] = compute(mod, full_base, frames[d], int(d))
        fulls[d].reset_index().to_parquet(out_dir / f"feats_{d}.parquet", index=False)
        summary["dates"][d] = {"n_horses": int(len(fulls[d])),
                               "identity_full": fulls[d]["identity_status"].value_counts().to_dict()}
    if not a.no_trunc:
        for d in dates:
            di = int(d)
            tb = make_base(work, f"trunc_{d}", hist[hist["date"].astype(int) < di].reset_index(drop=True), maps)
            trunc = compute(mod, tb, frames[d], di)
            mod._CACHE.pop(str(tb), None)
            rec = summary["dates"][d]
            rec["T1_T3_truncated_vs_full"] = compare(mod, fulls[d], trunc)
            rec["identity_truncated"] = trunc["identity_status"].value_counts().to_dict()
            t1 = rec["T1_T3_truncated_vs_full"]
            print(f"{d} n={rec['n_horses']} T1 any={t1['any']} roll={t1['rolling4']} "
                  f"horse={t1['horse_feats']} codes={t1['codes']} T3 ident={t1['identity']}", flush=True)
    for s in seeds:
        rng = np.random.default_rng(s)
        sb = make_base(work, f"shuf{s}", hist.iloc[rng.permutation(len(hist))].reset_index(drop=True), maps)
        for d in dates:
            rng2 = np.random.default_rng(1000 + s)
            dfs = frames[d].iloc[rng2.permutation(len(frames[d]))].reset_index(drop=True)
            summary["dates"][d][f"T2_shuffle_seed{s}"] = compare(mod, fulls[d], compute(mod, sb, dfs, int(d)))
        mod._CACHE.pop(str(sb), None)
        print(f"T2 seed{s}: any=" + str([summary["dates"][d][f"T2_shuffle_seed{s}"]["any"] for d in dates]),
              flush=True)
    first_df = frames[dates[0]] if dates else None
    if first_df is not None:
        summary["history_checks"] = history_checks(mod, hist, work, maps, first_df)
    tot = {"T1_any": 0, "T1_rolling4": 0, "T1_horse_feats": 0, "T3_identity": 0, "T2_any": 0, "n": 0}
    for rec in summary["dates"].values():
        tot["n"] += rec["n_horses"]
        t1 = rec.get("T1_T3_truncated_vs_full")
        if t1:
            tot["T1_any"] += t1["any"]; tot["T1_rolling4"] += t1["rolling4"]
            tot["T1_horse_feats"] += t1["horse_feats"]; tot["T3_identity"] += t1["identity"]
        tot["T2_any"] += sum(rec[k]["any"] for k in rec if k.startswith("T2_"))
    summary["totals"] = tot
    summary["verdict"] = {"T1": "PASS" if tot["T1_any"] == 0 else "FAIL",
                          "T2": "PASS" if tot["T2_any"] == 0 else "FAIL",
                          "T3": "PASS" if tot["T3_identity"] == 0 else "FAIL"}
    (out_dir / "features_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"totals": tot, "verdict": summary["verdict"],
                      "history_checks": summary.get("history_checks")}, ensure_ascii=False, default=str))
    return 0


# ------------------------------------------------------------------ bundles (T4 / §6)
_CHILD = r'''
import sys, json, re
import numpy as np
sys.path.insert(0, r"{prod}")
sys.path.insert(0, r"{moddir}")
import eligibility_coverage_gate as g
class _Pass:
    ok = True; expected = 0; flat_rids = []; jump_rids = []
    def as_dict(self): return {{"replay": True}}
g.check_coverage = lambda *a, **k: _Pass()
import race_eligibility as re_
_served = set(json.loads(r"""{served}"""))
_orig = re_.evaluate_race
def _ev(rid, *a, **k):
    el = dict(_orig(rid, *a, **k))
    el["prediction_eligible"] = re.sub(r"\D", "", str(rid))[:16] in _served
    return el
re_.evaluate_race = _ev
g.AtomicPublish.abort = g.AtomicPublish.commit
import pandas as _pd
_rp = _pd.read_parquet
def _read(path, *a, **k):
    df = _rp(path, *a, **k)
    if str(path).endswith("_horse_history.parquet"):
        if {trunc}:
            df = df[df["date"].astype(int) < {date}].copy()
        if {seed} >= 0:
            df = df.iloc[np.random.default_rng({seed}).permutation(len(df))].reset_index(drop=True)
    return df
_pd.read_parquet = _read
import predict_weekly as _pw
_pc = _pw.parse_csv
def _parse(*a, **k):
    df = _pc(*a, **k)
    if {seed} >= 0:
        df = df.iloc[np.random.default_rng(1000 + {seed}).permutation(len(df))].reset_index(drop=True)
    return df
_pw.parse_csv = _parse
sys.argv = ["export_weekly_marks.py", "--csv", r"{csv}", "--shap-topk", "0", "--out-dir", r"{out}"]
_src = open(r"{export_src}", encoding="utf-8").read()
exec(compile(_src, r"{export_src}", "exec"),
     {{"__name__": "__main__", "__file__": r"{prod}\export_weekly_marks.py"}})
'''


def served_rids(d: str) -> list[str]:
    b = json.loads((SERVED / f"{d}_bundle.json").read_text(encoding="utf-8"))
    rs = b["races"] if isinstance(b["races"], list) else list(b["races"].values())
    return sorted({re.sub(r"\D", "", str(r.get("race_id", "")))[:16] for r in rs})


def cmd_bundles(a) -> int:
    out = Path(a.out)
    export_src = Path(a.export_src) if a.export_src else BASE / "export_weekly_marks.py"
    conds = [c for c in a.conditions.split(",") if c]
    for d in [x for x in a.dates.split(",") if x]:
        rids = served_rids(d)
        for cond in conds:
            dest = out / cond / f"{d}_bundle.json"
            if dest.exists():
                continue
            trunc = cond == "trunc"
            seed = int(cond[4:]) if cond.startswith("shuf") else -1
            code = _CHILD.format(prod=str(BASE), moddir=str(Path(a.module_dir).resolve()),
                                 served=json.dumps(rids), trunc=trunc, date=int(d), seed=seed,
                                 csv=str(BASE / "data" / "weekly" / f"{d}.csv"),
                                 out=str(out / cond / d), export_src=str(export_src))
            env = dict(os.environ, CB_HOSEI_PROXY="1" if a.proxy else "0", PYTHONIOENCODING="utf-8")
            r = subprocess.run([sys.executable, "-c",
                                "import ctypes, sys\n"
                                "try:\n k=ctypes.windll.kernel32; k.SetPriorityClass(k.GetCurrentProcess(), 0x4000)\n"
                                "except Exception:\n pass\n"
                                "exec(open(sys.argv[1], encoding='utf-8').read())",
                                _write_child(out, code)],
                               cwd=str(BASE), capture_output=True, env=env, text=True,
                               encoding="utf-8", errors="replace")
            (out / cond).mkdir(parents=True, exist_ok=True)
            (out / cond / f"{d}.log").write_text((r.stdout or "") + (r.stderr or ""), encoding="utf-8")
            print(f"{d} {cond} rc={r.returncode} bundle={'yes' if dest.exists() else 'NO'}", flush=True)
    return 0


def _write_child(out: Path, code: str) -> str:
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"_child_{hashlib.sha1(code.encode()).hexdigest()[:12]}.py"
    p.write_text(code, encoding="utf-8")
    return str(p)


def bundle_horses(path: Path) -> dict:
    b = json.loads(Path(path).read_text(encoding="utf-8"))
    rs = b["races"] if isinstance(b["races"], list) else list(b["races"].values())
    out = {}
    for r in rs:
        rid = re.sub(r"\D", "", str(r.get("race_id", "")))[:16]
        for h in r.get("horses", []):
            if h.get("umaban") is None:
                continue
            out[(rid, int(h["umaban"]))] = (h.get("p_win"), h.get("mark") or "", h.get("tansho_odds"),
                                            h.get("horse_name"))
    return out


def cmd_t4(a) -> int:
    out = Path(a.out)
    conds = [c for c in a.conditions.split(",") if c and c != "full"]
    res = {"dates": {}, "totals": {}}
    tot = {c: {"horses": 0, "p_win_diff": 0, "mark_diff": 0, "missing": 0} for c in conds}
    for d in [x for x in a.dates.split(",") if x]:
        ref = bundle_horses(out / "full" / f"{d}_bundle.json")
        rec = {}
        for c in conds:
            p = out / c / f"{d}_bundle.json"
            if not p.exists():
                rec[c] = "bundle なし"
                continue
            got = bundle_horses(p)
            keys = set(ref) | set(got)
            miss = sum(1 for k in keys if k not in ref or k not in got)
            pdiff = [k for k in keys if k in ref and k in got and ref[k][0] != got[k][0]]
            mdiff = [k for k in keys if k in ref and k in got and ref[k][1] != got[k][1]]
            rec[c] = {"horses": len(keys), "missing": miss, "p_win_diff": len(pdiff), "mark_diff": len(mdiff),
                      "p_win_max_abs": max([abs(float(ref[k][0]) - float(got[k][0])) for k in pdiff] or [0.0])}
            for f in ("horses", "missing", "p_win_diff", "mark_diff"):
                tot[c][f] += rec[c][f]
        res["dates"][d] = rec
        print(d, json.dumps(rec, ensure_ascii=False), flush=True)
    res["totals"] = tot
    res["verdict"] = "PASS" if all(v["p_win_diff"] == 0 and v["mark_diff"] == 0 and v["missing"] == 0
                                   for v in tot.values()) else "FAIL"
    Path(a.report).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"totals": tot, "verdict": res["verdict"]}, ensure_ascii=False))
    return 0


# ------------------------------------------------------------------ §6 段階比較
def _kekka(d: str) -> dict:
    p = BASE / "data" / "kekka" / f"{d}.csv"
    if not p.exists():
        return {}
    k = pd.read_csv(p, encoding="cp932", low_memory=False)
    k["rid16"] = k["レースID(新)"].astype(str).str[:16]
    k["ban"] = pd.to_numeric(k["馬番"], errors="coerce")
    k["fin"] = pd.to_numeric(k["確定着順"], errors="coerce")
    k = k[k.fin.between(1, 3) & k.ban.notna()]
    return {rid: set(g.ban.astype(int)) for rid, g in k.groupby("rid16")}


def _t10_tansho(d: str) -> dict:
    """forward_prices の t10 録 (本番 T-10 の単勝)。{rid16: {umaban: odds}}。"""
    out = {}
    for p in sorted((BASE / "data" / "forward_prices" / d).glob("*_t10_*.json.gz")):
        try:
            rec = json.loads(gzip.open(p).read())
        except Exception:
            continue
        m = rec.get("market") or {}
        if not m.get("ok"):
            continue
        out[str(m.get("race_id") or p.name[:16])[:16]] = {int(k): float(v) for k, v in (m.get("tansho") or {}).items()}
    return out


def _a95(horses: dict, rid: str, odds: dict | None):
    """a95_engine.build を同じ T-10 単勝で当てる。戻り値 (帯, [(券種, 買い目)]) か None。"""
    import a95_engine
    hs = []
    for (r, ban), v in horses.items():
        if r != rid or v[0] is None:
            continue
        o = (odds or {}).get(ban) if odds is not None else v[2]
        hs.append({"umaban": ban, "p_win": float(v[0]), "tansho_odds": o})
    try:
        plan = a95_engine.build(hs)
    except a95_engine.A95Error:
        return None
    return plan["band"], tuple((t["kind"], t["selection"]) for t in plan["tickets"])


def _boot_diff(per_day: pd.DataFrame, a: str, b: str, reps: int = 4000, seed: int = 0):
    if len(per_day) < 2:
        return (float("nan"), float("nan"))
    idx = np.random.default_rng(seed).integers(0, len(per_day), size=(reps, len(per_day)))
    n = per_day["n"].to_numpy()[idx].sum(1)
    d = (per_day[a].to_numpy()[idx].sum(1) - per_day[b].to_numpy()[idx].sum(1)) / n * 100
    return (round(float(np.percentile(d, 2.5)), 2), round(float(np.percentile(d, 97.5)), 2))


def cmd_stagediff(a) -> int:
    fa, fb = Path(a.feats_a), Path(a.feats_b)
    ba, bb = Path(a.bundles_a), Path(a.bundles_b)
    rows, days, ident_trans, ident_examples = [], [], {}, []
    roll_changed = horses_total = 0
    a95_cmp = {"races_t10": 0, "changed_t10": 0, "races_bundle_odds": 0, "changed_bundle_odds": 0}
    pdiffs = []
    mark_changed = 0
    for d in [x for x in a.dates.split(",") if x]:
        if (fa / f"feats_{d}.parquet").exists() and (fb / f"feats_{d}.parquet").exists():
            xa = pd.read_parquet(fa / f"feats_{d}.parquet").set_index(["rid", "umaban"])
            xb = pd.read_parquet(fb / f"feats_{d}.parquet").set_index(["rid", "umaban"])
            com = xa.index.intersection(xb.index)
            xa, xb = xa.loc[com].sort_index(), xb.loc[com].sort_index()
            horses_total += len(com)
            roll_changed += int(diff_rows(xa, xb, ROLL4).sum())
            for k, sa, sb, nm in zip(com, xa["identity_status"], xb["identity_status"], xa["horse_name"]):
                if sa != sb:
                    ident_trans[f"{sa}->{sb}"] = ident_trans.get(f"{sa}->{sb}", 0) + 1
                    if len(ident_examples) < 40:
                        ident_examples.append({"date": d, "rid": k[0], "umaban": int(k[1]), "name": nm,
                                               "from": sa, "to": sb})
        pa, pb = ba / f"{d}_bundle.json", bb / f"{d}_bundle.json"
        if not (pa.exists() and pb.exists()):
            continue
        ha, hb = bundle_horses(pa), bundle_horses(pb)
        kek, t10 = _kekka(d), _t10_tansho(d)
        rids = sorted({k[0] for k in ha} & {k[0] for k in hb})
        day = {"date": d, "n": 0, "top1_changed": 0, "a_top3": 0, "b_top3": 0, "a_fav": 0, "b_fav": 0,
               "n_fav": 0}
        for rid in rids:
            ka = {k: v for k, v in ha.items() if k[0] == rid and v[0] is not None}
            kb = {k: v for k, v in hb.items() if k[0] == rid and v[0] is not None}
            com = set(ka) & set(kb)
            if len(com) < 5:
                continue
            for k in com:
                pdiffs.append(abs(float(ka[k][0]) - float(kb[k][0])))
                mark_changed += int(ka[k][1] != kb[k][1])
            hon_a = next((k[1] for k in com if ka[k][1] == "◎"), None)
            hon_b = next((k[1] for k in com if kb[k][1] == "◎"), None)
            day["top1_changed"] += int(hon_a != hon_b)
            if rid in kek:
                day["n"] += 1
                day["a_top3"] += int(hon_a in kek[rid])
                day["b_top3"] += int(hon_b in kek[rid])
                od = [(k[1], ka[k][2]) for k in com if ka[k][2]]
                if len(od) >= 5:
                    fav = min(od, key=lambda t: float(t[1]))[0]
                    day["n_fav"] += 1
                    day["a_fav"] += int(hon_a == fav)
                    day["b_fav"] += int(hon_b == fav)
            for src, odds in (("t10", t10.get(rid)), ("bundle_odds", None)):
                if src == "t10" and not odds:
                    continue
                pa95, pb95 = _a95(ka, rid, odds), _a95(kb, rid, odds)
                a95_cmp[f"races_{src}"] += 1
                a95_cmp[f"changed_{src}"] += int(pa95 != pb95)
            day.setdefault("races", 0)
            day["races"] += 1
        days.append(day)
    pdf = pd.DataFrame(days)
    pd_arr = np.array(pdiffs) if pdiffs else np.zeros(1)
    res = {"a": str(fa), "b": str(fb), "dates": len(days),
           "horses_feature_level": horses_total, "rolling4_changed_horses": roll_changed,
           "identity_transitions": ident_trans, "identity_examples": ident_examples,
           "races": int(pdf["races"].sum()) if len(pdf) else 0,
           "top1_changed_races": int(pdf["top1_changed"].sum()) if len(pdf) else 0,
           "p_win_abs_diff": {"median": float(np.median(pd_arr)), "p95": float(np.percentile(pd_arr, 95)),
                              "max": float(pd_arr.max()), "n_horses": len(pdiffs)},
           "mark_changed_horses": mark_changed, "a95": a95_cmp}
    if len(pdf) and pdf["n"].sum():
        res["hon_top3"] = {"n_races": int(pdf["n"].sum()),
                           "a_pct": round(100 * pdf["a_top3"].sum() / pdf["n"].sum(), 2),
                           "b_pct": round(100 * pdf["b_top3"].sum() / pdf["n"].sum(), 2),
                           "b_minus_a_ci95": _boot_diff(pdf[pdf.n > 0], "b_top3", "a_top3")}
        pf = pdf[pdf.n_fav > 0].rename(columns={"n_fav": "n_"}).assign(n=lambda x: x["n_"])
        if len(pf):
            res["agree_market_fav"] = {"n_races": int(pf["n"].sum()),
                                       "a_pct": round(100 * pf["a_fav"].sum() / pf["n"].sum(), 2),
                                       "b_pct": round(100 * pf["b_fav"].sum() / pf["n"].sum(), 2),
                                       "b_minus_a_ci95": _boot_diff(pf, "b_fav", "a_fav")}
    Path(a.report).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "identity_examples"}, ensure_ascii=False))
    return 0


# ------------------------------------------------------------------ hybrid (2024〜25)
MASTER = BASE / "data" / "master_v2_20130105-20251228.csv"


def master_days(year_from: int, year_to: int, n: int, seed: int) -> list[str]:
    """n<=0 なら期間の全開催日、それ以外は種 seed で無作為に n 日。"""
    d = pd.read_csv(MASTER, encoding="utf-8-sig", usecols=["日付"], low_memory=False)
    ds = sorted({str(int(x)) for x in pd.to_numeric(d["日付"], errors="coerce").dropna()
                 if year_from <= int(x) // 10000 <= year_to})
    if n <= 0:
        return ds
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(ds, size=min(n, len(ds)), replace=False).tolist())


def read_master_days(days: list[str], cols: set[str]) -> pd.DataFrame:
    """master から指定日の行だけを読む (列を絞り、塊ごとに読んでメモリを抑える)。"""
    head = pd.read_csv(MASTER, encoding="utf-8-sig", nrows=0).columns
    use = [c for c in head if c in cols]
    want = set(days)
    parts = []
    for ch in pd.read_csv(MASTER, encoding="utf-8-sig", usecols=use, low_memory=False, chunksize=100_000):
        ch = ch[ch["日付"].astype("Int64").astype(str).isin(want)]
        if len(ch):
            parts.append(ch)
    return pd.concat(parts, ignore_index=True)


def cmd_hybrid(a) -> int:
    """master の学習時特徴のうち履歴特徴だけを、段階ごとの配信経路の値に差し替えて採点する。"""
    lower_priority()
    import joblib
    import pl_probs as PL
    from backtest_pl_ev import COL_BAN, COL_RID
    days = master_days(2024, 2025, a.n_days, a.seed) if not a.dates else a.dates.split(",")
    mb = joblib.load(BASE / "models" / "unified_rank_v6.pkl")
    model, feats, encs = mb["model"], mb["feature_cols"], mb["encoders"]
    cal = joblib.load(BASE / "models" / "pl_calibrators_v6_serve.pkl")
    cal = cal.get("calibrators", cal)
    need = set(feats) | {COL_RID, COL_BAN, "日付", "馬名", "種牡馬", "年齢", "場所", "芝・ダ", "距離",
                         "騎手コード", "調教師コード", "着順"}
    m = read_master_days(days, need)
    out = {"days": days, "stages": {}}
    stages = [s for s in a.stages.split(",") if s]
    for st in stages:
        name, mdir = st.split("=", 1)
        df = m.copy()
        df["日付"] = df["日付"].astype(int).astype(str)
        if mdir != "-":     # "-" = 学習時の値のまま (差し替えない)
            mod = load_stage_module(Path(mdir))
            base = Path(mdir)
            mod._CACHE.pop(str(base), None)
            mod.fill_history_features(df, base=base, jockey_code_override=m["騎手コード"],
                                      trainer_code_override=m["調教師コード"])
            mod._CACHE.pop(str(base), None)
        recs = []
        for rid, g in df.groupby(COL_RID, sort=False):
            g = g.sort_values(COL_BAN)
            enc = g.copy()
            for c, le in encs.items():
                if c not in enc.columns:
                    continue
                v = enc[c].astype(str).fillna("__NaN__")
                v = v.where(v.isin(set(le.classes_)), "__NaN__")
                enc[c] = le.transform(v)
            X = enc[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
            s = model.predict(X)
            w = PL.pl_weights(s)
            pw = PL.all_tansho(w)
            if "tansho" in cal:
                pw = cal["tansho"].predict(pw)
            for ban, p, sc, fin in zip(g[COL_BAN].astype(int), pw, s, pd.to_numeric(g["着順"], errors="coerce")):
                recs.append((str(rid)[:16], int(ban), float(p), float(sc), fin))
        r = pd.DataFrame(recs, columns=["rid", "umaban", "p_win", "score", "fin"])
        r.to_parquet(Path(a.out) / f"hybrid_{name}.parquet", index=False)
        out["stages"][name] = {"module": mdir, "horses": int(len(r)), "races": int(r["rid"].nunique())}
        print(name, out["stages"][name], flush=True)
    Path(a.report).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


def cmd_hybridcmp(a) -> int:
    """hybrid の段階どうしを比べる: 1 位が変わるレース、p_win 差、◎top3 率 (開催日 bootstrap)。"""
    out = Path(a.out)
    res = {}
    for pair in [p for p in a.pairs.split(",") if p]:
        x, y = pair.split(":")
        ra = pd.read_parquet(out / f"hybrid_{x}.parquet")
        rb = pd.read_parquet(out / f"hybrid_{y}.parquet")
        m = ra.merge(rb, on=["rid", "umaban"], suffixes=("_a", "_b"))
        m["date"] = m["rid"].str[:8]
        top_a = m.loc[m.groupby("rid")["score_a"].idxmax(), ["rid", "date", "umaban", "fin_a"]]
        top_b = m.loc[m.groupby("rid")["score_b"].idxmax(), ["rid", "umaban", "fin_b"]]
        t = top_a.merge(top_b, on="rid", suffixes=("_a", "_b"))
        t["a_top3"] = (pd.to_numeric(t["fin_a"], errors="coerce") <= 3).astype(int)
        t["b_top3"] = (pd.to_numeric(t["fin_b"], errors="coerce") <= 3).astype(int)
        per = t.groupby("date").agg(n=("rid", "size"), a_top3=("a_top3", "sum"), b_top3=("b_top3", "sum")).reset_index()
        d = (m["p_win_a"] - m["p_win_b"]).abs()
        res[pair] = {"races": int(len(t)), "horses": int(len(m)), "days": int(per.shape[0]),
                     "top1_changed_races": int((t["umaban_a"] != t["umaban_b"]).sum()),
                     "p_win_abs_diff": {"median": float(d.median()), "p95": float(d.quantile(0.95)),
                                        "max": float(d.max()), "n_changed": int((d > 0).sum())},
                     "hon_top3": {"a_pct": round(100 * t["a_top3"].mean(), 2), "b_pct": round(100 * t["b_top3"].mean(), 2),
                                  "b_minus_a_ci95": _boot_diff(per, "b_top3", "a_top3")}}
        print(pair, json.dumps(res[pair], ensure_ascii=False), flush=True)
    Path(a.report).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


def cmd_parity(a) -> int:
    """§4-A: 2024〜25 年の全開催日で、配信経路の履歴特徴と master に格納された学習時の値を馬単位で比べる。

    騎手・調教師の 4 特徴は定義差 (学習側は同日の先行レースを窓に含める) と P0-5 修正前の
    同一レース内混入 (調教師側、ブロックの 2 頭目以降) を別枠にし、どちらにも当たらない
    部分集合で完全一致を要求する。その前に履歴 parquet の master 由来行と master の行集合が
    一致することを確かめる。騎手・調教師コードは master のコードを与える (名前→コード表の解決は
    master に名前列が無いので検査できない)。
    """
    lower_priority()
    import joblib
    from backtest_pl_ev import COL_BAN, COL_RID
    mod = load_stage_module(Path(a.module_dir))
    base = Path(a.module_dir)
    res: dict = {"module": str(base / "serve_history_feats.py"),
                 "serve_path_version": getattr(mod, "SERVE_PATH_VERSION", None)}
    # 1) 履歴行集合 (master 由来行) と master の行集合
    keys = ["血統登録番号", "日付", "レースID(新/馬番無)"]
    mk = pd.read_csv(MASTER, encoding="utf-8-sig", usecols=keys, low_memory=False).dropna()
    mset = set(zip(mk["血統登録番号"].astype(np.int64), mk["日付"].astype(np.int64),
                   mk["レースID(新/馬番無)"].astype(str)))
    hist = pd.read_parquet(base / "data" / "_horse_history.parquet")
    hm = hist[hist["src"] == "master"]
    hset = set(zip(hm["ped_id"].astype(np.int64), hm["date"].astype(np.int64), hm["race_id"].astype(str)))
    res["history_row_set"] = {"master_rows": len(mset), "history_master_rows": len(hset),
                              "only_in_master": len(mset - hset), "only_in_history": len(hset - mset),
                              "identical": mset == hset}
    print("row set:", res["history_row_set"], flush=True)
    del mk, mset, hset, hm
    # 2) 2024〜25 年の行で配信経路の特徴を作る
    days = master_days(2024, 2025, 0, 0)
    cols = {COL_RID, COL_BAN, "日付", "発走時刻", "馬名", "種牡馬", "年齢", "場所", "芝・ダ", "距離",
            "騎手コード", "調教師コード", "血統登録番号", *mod.NUM_FEATS}
    m = read_master_days(days, cols)
    df = m.copy()
    df["日付"] = df["日付"].astype(int).astype(str)
    mod._CACHE.pop(str(base), None)
    mod.fill_history_features(df, base=base, jockey_code_override=m["騎手コード"],
                              trainer_code_override=m["調教師コード"])
    idx = mod._CACHE[str(base)][0]
    names, sires = m["馬名"].map(mod._clean_name), m["種牡馬"].map(mod._clean_name)
    ages = pd.to_numeric(m["年齢"], errors="coerce")
    resolved = []
    for i in m.index:
        d = int(m.at[i, "日付"])
        by = int(d // 10000 - ages.at[i]) if pd.notna(ages.at[i]) else None
        resolved.append(identify(mod, idx, names.at[i], sires.at[i], by, d)[1])
    m["resolved_ped"] = pd.array(resolved, dtype="Int64")
    m["identity_ok"] = m["resolved_ped"].astype("float") == pd.to_numeric(m["血統登録番号"], errors="coerce")
    st = df.get(getattr(mod, "IDENTITY_STATUS_COL", "_none"))
    res["identity"] = {"rows": int(len(m)), "resolved_to_true_ped": int(m["identity_ok"].sum()),
                       "status": (st.value_counts().to_dict() if st is not None else None)}
    print("identity:", res["identity"], flush=True)
    # 3) 同日の先行レース・同一レース内ブロックの印
    m["_pt"] = m["発走時刻"].astype(str)
    m["_rid"] = m[COL_RID].astype(str)
    m = m.sort_values(["日付", "_pt", "_rid", COL_BAN], kind="mergesort")
    m["jockey_sameday_prior"] = m.groupby(["騎手コード", "日付"]).cumcount() > 0
    first_race = m.groupby(["調教師コード", "日付"])["_rid"].transform("first")
    m["trainer_sameday_prior"] = m["_rid"] != first_race
    m["trainer_block_nonfirst"] = m.groupby(["調教師コード", "_rid"]).cumcount() > 0
    m = m.sort_index()
    gain = {}
    try:
        mb = joblib.load(BASE / "models" / "unified_rank_v6.pkl")
        g = mb["model"].feature_importance(importance_type="gain")
        tot = float(np.sum(g))
        gain = {f: round(100 * float(v) / tot, 3) for f, v in zip(mb["feature_cols"], g)}
    except Exception:
        pass

    def _cmp(sel: pd.Series, c: str) -> dict:
        x = pd.to_numeric(m.loc[sel, c], errors="coerce").to_numpy(float)
        y = pd.to_numeric(df.loc[sel[sel].index, c], errors="coerce").to_numpy(float)
        both = ~np.isnan(x) & ~np.isnan(y)
        eq = (x == y) | (np.isnan(x) & np.isnan(y))
        dd = np.abs(x[both] - y[both])
        return {"n": int(len(x)), "exact": int(eq.sum()), "exact_pct": round(100 * float(eq.mean()), 3) if len(x) else None,
                "both_missing": int((np.isnan(x) & np.isnan(y)).sum()),
                "missing_mismatch": int((np.isnan(x) ^ np.isnan(y)).sum()),
                "diff_median": float(np.median(dd)) if len(dd) else 0.0,
                "diff_p95": float(np.percentile(dd, 95)) if len(dd) else 0.0}

    allrows = pd.Series(True, index=m.index)
    ok_id = m["identity_ok"].fillna(False)
    feats = {}
    for c in mod.NUM_FEATS:
        rec = {"gain_pct": gain.get(c), "all_rows": _cmp(allrows, c)}
        if c.startswith("jockey_fuku"):
            sub = ~m["jockey_sameday_prior"]
            rec["no_sameday_prior (完全一致を要求)"] = _cmp(sub, c)
            rec["sameday_prior (定義差)"] = _cmp(~sub, c)
        elif c.startswith("trainer_fuku"):
            sub = ~m["trainer_sameday_prior"] & ~m["trainer_block_nonfirst"]
            rec["no_sameday_prior_block_first (完全一致を要求)"] = _cmp(sub, c)
            rec["sameday_prior (定義差)"] = _cmp(m["trainer_sameday_prior"] & ~m["trainer_block_nonfirst"], c)
            rec["block_nonfirst (P0-5 修正前の混入)"] = _cmp(m["trainer_block_nonfirst"], c)
        else:
            rec["identity_resolved_to_true_ped (完全一致を要求)"] = _cmp(ok_id, c)
            rec["identity_not_resolved"] = _cmp(~ok_id, c)
        feats[c] = rec
        print(c, json.dumps({k: (v if not isinstance(v, dict) else v.get("exact_pct")) for k, v in rec.items()},
                            ensure_ascii=False), flush=True)
    res["features"] = dict(sorted(feats.items(), key=lambda kv: -(kv[1]["gain_pct"] or 0)))
    res["days"] = len(days)
    Path(a.report).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    f = sp.add_parser("features")
    f.add_argument("--module-dir", required=True)
    f.add_argument("--dates", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--work", required=True)
    f.add_argument("--seeds", default="0,1,2,3,4")
    f.add_argument("--no-trunc", action="store_true")
    b = sp.add_parser("bundles")
    b.add_argument("--module-dir", required=True)
    b.add_argument("--export-src", default=None)
    b.add_argument("--dates", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--conditions", default="full")
    b.add_argument("--proxy", action="store_true")
    t = sp.add_parser("t4")
    t.add_argument("--out", required=True)
    t.add_argument("--dates", required=True)
    t.add_argument("--conditions", required=True)
    t.add_argument("--report", required=True)
    s = sp.add_parser("stagediff")
    for k in ("feats_a", "feats_b", "bundles_a", "bundles_b", "dates", "report"):
        s.add_argument("--" + k.replace("_", "-"), dest=k, required=True)
    h = sp.add_parser("hybrid")
    h.add_argument("--stages", required=True, help="name=module_dir,...")
    h.add_argument("--out", required=True)
    h.add_argument("--report", required=True)
    h.add_argument("--n-days", type=int, default=20)
    h.add_argument("--seed", type=int, default=20261004)
    h.add_argument("--dates", default="")
    hc = sp.add_parser("hybridcmp")
    hc.add_argument("--out", required=True)
    hc.add_argument("--pairs", required=True, help="a:b,c:d (hybrid の stage 名)")
    hc.add_argument("--report", required=True)
    pa = sp.add_parser("parity")
    pa.add_argument("--module-dir", required=True)
    pa.add_argument("--report", required=True)
    ap.add_argument("--root", default=None, help="本番 root (データと未変更 module の置き場所)")
    a = ap.parse_args()
    set_root(a.root)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    return {"features": cmd_features, "bundles": cmd_bundles, "t4": cmd_t4,
            "stagediff": cmd_stagediff, "hybrid": cmd_hybrid, "hybridcmp": cmd_hybridcmp,
            "parity": cmd_parity}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
