"""ズレ警告の事前登録検証（2026-11-18 仕様: docs/research/ZURE_VALIDATION_SPEC_20261118.md）。

「モデル順位は低いのに他の指標が突出している馬（ズレ馬）」が実際に走るかを、公開済みの日別 JSON
(site/data/{date}.json) と確定払戻 (data/kekka/{date}.csv) で集計する。

  第 1 段階: python -m analysis.zure_validation               → 発見期間だけ
  第 2 段階: python -m analysis.zure_validation --phase validation --criteria-fixed
             （仕様書に採用候補と目安を追記して固定した後でだけ回す）

条件（事前登録、ここ以外は試さない）:
  D1 モデル 6 位以下 かつ 馬レベル順位がレース内 3 位以内
  D2 モデル 6 位以下 かつ 馬レベル >= 90
  D3 モデル 6 位以下 かつ PW 順位 1 位
  D4 モデル 6 位以下 かつ スピード指数順位 1 位
  D5 モデル 6 位以下 かつ 重賞掲示板確率 (出世予測 p_graded12) >= 50%（重賞のみ）
  D6 モデル 6 位以下 かつ D1〜D5 のうち 2 つ以上
対照群: 同じモデル順位帯で D1〜D5 のどれにも当てはまらない馬。
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from build_site import parse_kekka  # noqa: E402

SHUSSE_PRED = Path(r"E:\PyCaLiAI\analysis\shusse\out_v2\live\predictions_latest.parquet")
GRADED = {"G1", "G2", "G3"}
BANDS = [("6-8位", 6, 8), ("9-12位", 9, 12), ("13位以下", 13, 99)]
CONDS = ["D1", "D2", "D3", "D4", "D5", "D6"]
EXTRA = ["D3x"]   # 探索扱いの参考行
COND_LABEL = {
    "D1": "馬レベル順位 3 位以内", "D2": "馬レベル >= 90", "D3": "PW 順位 1 位",
    "D4": "スピード指数順位 1 位", "D5": "重賞掲示板確率 >= 50%（重賞のみ）",
    "D6": "D1〜D5 のうち 2 つ以上",
    "D3x": "PW 順位 1 位（同率 1 位が 2 頭以上のレースを除く、探索扱い）",
}


def day_files() -> list[Path]:
    return sorted(Path(f) for f in glob.glob(str(ROOT / "site" / "data" / "2026*.json"))
                  if re.fullmatch(r"2026\d{4}\.json", os.path.basename(f)))


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (None, None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, c - h, c + h)


def rank_desc(vals: dict[int, float]) -> dict[int, int]:
    """馬番 → 順位（大きい方が 1 位、同値は同順位 = min 方式）。値の無い馬は入らない。"""
    out = {}
    xs = sorted(vals.values(), reverse=True)
    for u, v in vals.items():
        out[u] = 1 + sum(1 for x in xs if x > v)
    return out


def load_shusse():
    p = pd.read_parquet(SHUSSE_PRED, columns=["馬名", "父", "k", "base_date", "p_graded12"])
    p["base_date"] = pd.to_datetime(p["base_date"])
    by_ns = {k: g.sort_values("base_date") for k, g in p.groupby(["馬名", "父"])}
    name_n = p.groupby("馬名")["父"].nunique()
    by_n = {k: g.sort_values("base_date") for k, g in p.groupby("馬名") if name_n.get(k, 0) == 1}
    return by_ns, by_n


def shusse_pg12(by_ns, by_n, name: str, sire: str | None, race_date: pd.Timestamp):
    """レース日より前の基準点のうち、予測窓（基準日 + 365 日）がまだ開いている最新のもの。
    サイトの出世株バッジ（判定待ちの馬のレース前の評価）と同じ考え方。無ければ None。"""
    g = by_ns.get((name, sire)) if sire else None
    if g is None:
        g = by_n.get(name)
    if g is None:
        return None
    g = g[(g["base_date"] < race_date) & (g["base_date"] + pd.Timedelta(days=365) > race_date)]
    if g.empty:
        return None
    return float(g.iloc[-1]["p_graded12"])


def build_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    by_ns, by_n = load_shusse()
    rows, races = [], []
    for f in day_files():
        d = json.loads(f.read_text(encoding="utf-8"))
        date = d["date"]
        dt = pd.Timestamp(date)
        prov = d.get("provenance") or {}
        kek = parse_kekka(date, {})
        for r in d["races"]:
            rid = str(r["race_id"])[:16]
            klass = r.get("klass") or ""
            course = r.get("course") or ""
            res = r.get("result") or {}
            hs = r.get("horses") or []
            reason = None
            if course.startswith("障") or klass.startswith("J"):
                reason = "障害"
            elif not any(h.get("mark") == "◎" for h in hs):
                reason = "◎なし"
            elif not res.get("top3") or not res.get("order"):
                reason = "結果なし"
            elif any(h.get("mark") == "◎" and int(h["umaban"]) in set(res.get("scratched") or []) for h in hs):
                reason = "◎発売前取消"   # 通信簿 (HONEST RECORD) と同じ除外
            races.append({"date": date, "race_id": rid, "klass": klass, "excluded": reason,
                          "era": prov.get("model_era") or "unknown",
                          "verified": bool(prov.get("verified_pre_race")),
                          "has_level": any(h.get("level") for h in hs)})
            if reason:
                continue
            scratched = set(res.get("scratched") or [])
            no_finish = set(res.get("no_finish") or [])
            kr = kek.get(rid) or {}
            pays = kr.get("pays") or {}
            tan_by = pays.get("tan_by") or {}
            fuku = pays.get("fuku") or {}
            runners = [h for h in hs if int(h["umaban"]) not in scratched]
            lv = {int(h["umaban"]): float(h["level"]["score"]) for h in runners
                  if isinstance(h.get("level"), dict) and h["level"].get("score") is not None}
            pw = {int(h["umaban"]): float(h["pw"]) for h in runners if h.get("pw") is not None}
            sp = {int(h["umaban"]): float(h["speed"]["v"]) for h in runners
                  if isinstance(h.get("speed"), dict) and h["speed"].get("v") is not None}
            lv_rk, pw_rk, sp_rk = rank_desc(lv), rank_desc(pw), rank_desc(sp)
            graded = klass in GRADED
            for h in runners:
                u = int(h["umaban"])
                pos = (res.get("order") or {}).get(str(u)) or 0
                pg = shusse_pg12(by_ns, by_n, h.get("name"), (h.get("pedigree") or {}).get("sire"), dt)
                rows.append({
                    "date": date, "race_id": rid, "klass": klass, "graded": graded,
                    "era": prov.get("model_era") or "unknown", "verified": bool(prov.get("verified_pre_race")),
                    "umaban": u, "ai_rank": h.get("ai_rank"),
                    "level": lv.get(u), "lv_rank": lv_rk.get(u), "pw_rank": pw_rk.get(u), "sp_rank": sp_rk.get(u),
                    "pg12": pg, "pos": pos, "win": pos == 1, "place": 0 < pos <= 3,
                    "settleable": u not in no_finish,
                    "tan": (tan_by.get(str(u)) or 0) if pos == 1 else 0,
                    "fuku": fuku.get(str(u)) or 0,
                    "has_pays": bool(kr),
                })
    return pd.DataFrame(rows), pd.DataFrame(races)


def flag(df: pd.DataFrame) -> pd.DataFrame:
    low = df["ai_rank"] >= 6
    df["D1"] = low & (df["lv_rank"] <= 3)
    df["D2"] = low & (df["level"] >= 90)
    df["D3"] = low & (df["pw_rank"] == 1)
    df["D4"] = low & (df["sp_rank"] == 1)
    # D3' (探索扱い・事後に見つけた定義の穴): 新馬戦などで全頭の PW が同じ初期値のとき全員が「1 位」になる。
    # 同率 1 位が 2 頭以上いるレースを除いた版。対照群の定義 (D1〜D5 のどれにも当てはまらない) は変えない。
    n_top = df[df["pw_rank"] == 1].groupby("race_id").size()
    df["D3x"] = df["D3"] & df["race_id"].map(n_top).eq(1)
    df["D5"] = low & df["graded"] & (df["pg12"] >= 0.5)
    df["D5_all"] = low & (df["pg12"] >= 0.5)   # 層別（重賞 / 非重賞）表示用
    df["n_hit"] = df[["D1", "D2", "D3", "D4", "D5"]].sum(axis=1)
    df["D6"] = low & (df["n_hit"] >= 2)
    df["control"] = low & (df["n_hit"] == 0)
    df["band"] = None
    for name, lo, hi in BANDS:
        df.loc[(df["ai_rank"] >= lo) & (df["ai_rank"] <= hi), "band"] = name
    return df


def roi_ci(g: pd.DataFrame, col: str, B: int = 2000, seed: int = 0):
    s = g[g["settleable"]]
    if s.empty:
        return (None, None, None)
    by = s.groupby("race_id").agg(stake=(col, "size"), ret=(col, "sum"))
    stake, ret = by["stake"].to_numpy() * 100.0, by["ret"].to_numpy().astype(float)
    point = ret.sum() / stake.sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(by), size=(B, len(by)))
    bs = ret[idx].sum(1) / stake[idx].sum(1)
    return (point, float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975)))


def summarize(g: pd.DataFrame, ctrl: pd.DataFrame) -> dict:
    """ctrl の複勝率は順位帯ごとに求め、該当馬の順位帯の構成で重み付けする（帯が 1 つなら普通の平均と同じ）。
    6位以下計の行で、該当馬が 6-8 位に偏るぶん対照が低く出て lift が膨らむのを防ぐ。"""
    n = len(g)
    w = wilson(int(g["win"].sum()), n)
    p = wilson(int(g["place"].sum()), n)
    cp = None
    if len(ctrl) and n:
        rate = ctrl.groupby("band")["place"].mean()
        mix = g["band"].value_counts(normalize=True)
        if set(mix.index) <= set(rate.index):
            cp = float((mix * rate.reindex(mix.index)).sum())
    elif len(ctrl):
        cp = float(ctrl["place"].mean())
    wr, pr = roi_ci(g, "tan"), roi_ci(g, "fuku")
    return {
        "n_horses": n, "n_races": g["race_id"].nunique(),
        "win_rate": w, "place_rate": p, "win_roi": wr, "place_roi": pr,
        "n_unsettled": int((~g["settleable"]).sum()),
        "control_n": len(ctrl), "control_place_rate": cp,
        "lift_place": (p[0] / cp) if (n and cp) else None,
        "wilson_lo_gt_control": (p[1] > cp) if (n and cp is not None) else None,
    }


def table(df: pd.DataFrame) -> list[dict]:
    out = []
    for c in CONDS + EXTRA:
        for bname, lo, hi in BANDS + [("6位以下計", 6, 99)]:
            b = df[(df["ai_rank"] >= lo) & (df["ai_rank"] <= hi)]
            ctrl = b[b["control"]]
            if c == "D5":
                ctrl = ctrl[ctrl["graded"]]   # D5 は重賞だけが対象なので対照も重賞内
            out.append({"cond": "D3'(探索)" if c == "D3x" else c, "band": bname, **summarize(b[b[c]], ctrl)})
    # D5 のクラス層別（重賞 / 非重賞）。条件は同じ「重賞掲示板確率 >= 50%」
    for gname, gmask in (("重賞", True), ("非重賞", False)):
        for bname, lo, hi in BANDS + [("6位以下計", 6, 99)]:
            b = df[(df["ai_rank"] >= lo) & (df["ai_rank"] <= hi) & (df["graded"] == gmask)]
            out.append({"cond": f"D5[{gname}]", "band": bname,
                        **summarize(b[b["D5_all"]], b[b["control"]])})
    return out


def fmt_pct(x, d=1):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:.{d}f}"


def fmt_ci(t):
    if t[0] is None:
        return "—"
    return f"{t[0] * 100:.1f} [{max(0.0, t[1]) * 100:.1f}, {t[2] * 100:.1f}]"


def to_md(rows: list[dict]) -> str:
    head = ("| 条件 | 順位帯 | n_horses | n_races | win_rate % [95%CI] | place_rate % [95%CI] | "
            "win_roi % [95%CI] | place_roi % [95%CI] | lift_place | control_place_rate % (n) | Wilson下限>対照 |\n"
            "|---|---|---:|---:|---|---|---|---|---:|---|:-:|\n")
    lines = []
    for r in rows:
        lift = "—" if r["lift_place"] is None else f"{r['lift_place']:.2f}"
        cpr = "—" if r["control_place_rate"] is None else f"{fmt_pct(r['control_place_rate'])} ({r['control_n']})"
        gt = "—" if r["wilson_lo_gt_control"] is None else ("○" if r["wilson_lo_gt_control"] else "×")
        lines.append(f"| {r['cond']} | {r['band']} | {r['n_horses']} | {r['n_races']} | {fmt_ci(r['win_rate'])} | "
                     f"{fmt_ci(r['place_rate'])} | {fmt_ci(r['win_roi'])} | {fmt_ci(r['place_roi'])} | {lift} | {cpr} | {gt} |")
    return head + "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["discovery", "validation"], default="discovery")
    ap.add_argument("--criteria-fixed", action="store_true",
                    help="検証期間を出すのは、仕様書に採用候補と目安を追記して固定した後だけ")
    ap.add_argument("--out", default=str(ROOT / "reports" / "zure_validation"))
    a = ap.parse_args()
    if a.phase == "validation" and not a.criteria_fixed:
        sys.exit("検証期間は採用候補と目安を仕様書に固定してから --criteria-fixed を付けて回す")

    rows, races = build_rows()
    elig = races[races["excluded"].isna()].copy()
    # 分割日: 対象期間（2026-04-18〜）の対象レースを日付順に並べ、累計が半分に達した開催日までを発見期間
    per_day = elig.groupby("date").size().sort_index()
    cum = per_day.cumsum()
    split_date = cum[cum >= len(elig) / 2].index[0]
    disc_dates = set(per_day.index[per_day.index <= split_date])

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "all_races": len(races), "excluded": dict(Counter(races["excluded"].dropna())),
        "eligible_races": len(elig), "split_date_inclusive": split_date,
        "first_date": per_day.index[0], "last_date": per_day.index[-1],
    }
    phase_dates = disc_dates if a.phase == "discovery" else set(per_day.index) - disc_dates
    pe = elig[elig["date"].isin(phase_dates)]
    meta["phase"] = a.phase
    meta["phase_races"] = len(pe)
    meta["phase_days"] = len(phase_dates)
    meta["phase_era"] = {f"{k[0]}|{'確認済み' if k[1] else '未検証'}": int(v)
                         for k, v in pe.groupby(["era", "verified"]).size().items()}
    meta["phase_races_without_level"] = int((~pe["has_level"]).sum())

    df = rows[rows["date"].isin(phase_dates)].copy()
    df = flag(df)
    # 馬レベルが JSON に無い日 (v5 期) は D1/D2 が判定できず、対照群「どの条件にも当てはまらない」も
    # 決まらないので、全条件とも馬レベルのある日だけで集計する。
    lvl_days = set(pe.loc[pe["has_level"], "date"])
    meta["analysis_days"] = len(lvl_days)
    meta["analysis_races"] = int(pe["has_level"].sum())
    meta["analysis_era"] = {f"{k[0]}|{'確認済み' if k[1] else '未検証'}": int(v)
                            for k, v in pe[pe["has_level"]].groupby(["era", "verified"]).size().items()}
    df = df[df["date"].isin(lvl_days)]
    meta["horses_rank6plus"] = int((df["ai_rank"] >= 6).sum())
    meta["missing_pays_races"] = int(df.loc[~df["has_pays"], "race_id"].nunique())
    meta["coverage_rank6plus"] = {
        "level": float(df.loc[df["ai_rank"] >= 6, "level"].notna().mean()),
        "pw": float(df.loc[df["ai_rank"] >= 6, "pw_rank"].notna().mean()),
        "speed": float(df.loc[df["ai_rank"] >= 6, "sp_rank"].notna().mean()),
        "pg12_graded": float(df.loc[(df["ai_rank"] >= 6) & df["graded"], "pg12"].notna().mean())
        if ((df["ai_rank"] >= 6) & df["graded"]).any() else None,
    }
    meta["graded_races"] = int(df.loc[df["graded"], "race_id"].nunique())
    tab = table(df)
    (out / f"{a.phase}.json").write_text(json.dumps({"meta": meta, "rows": tab}, ensure_ascii=False,
                                                     indent=1, default=str), encoding="utf-8")
    md = to_md(tab)
    (out / f"{a.phase}.md").write_text(md, encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=1, default=str))
    print(md)


if __name__ == "__main__":
    main()
