"""EV 閾値 単勝/複勝 ルールのバックテスト (分析専用、本番経路には不干渉)。

ルール: 各レースで EV = 勝率 × 単勝オッズ が閾値以上の馬の単勝を 100 円ずつ買う。
閾値は事前に 1.0 / 1.1 / 1.15 / 1.2 の 4 通りを固定 (事後選択しない)。
  - tan_all   : 単勝 EV(p_win × 最終単勝オッズ) ≥ 閾値の全馬
  - tan_top1  : 同じ EV の最大馬 1 頭だけ (EV ≥ 閾値のとき)
  - fuku_all  : 複勝 EV(p_sho × 複勝オッズ下限) ≥ 閾値の全馬
  - fuku_top1 : 同じ複勝 EV の最大馬 1 頭だけ
感度分析: tan_* を bundle の export 時点の単勝オッズで選んだ版 (tan_exp_*)。

入力:
  reports/cowork_input_frozen/{date}_bundle.json  (凍結版 = 発走前の公開時点の確率)
  data/kekka/{date}.csv                           (確定着順・払戻・非勝ち馬の最終単勝オッズ)
決済規則は build_site.parse_kekka と同じ (着順 0 = 取消/除外/中止は top3 外、単勝欄が空なら返還、
最終オッズのある着順 0 の馬は中止/発売後除外の区別不能 → その馬の券は集計外)。

使い方:
  python -m analysis.ev_threshold_backtest --data-root E:/PyCaLiAI --out-json <path>
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

import build_site

THRESHOLDS = [1.0, 1.1, 1.15, 1.2]
START, END = "20260418", "20261115"
HIGH_CHAOS_PRIMARY = 0.667     # 本番の参戦ガードと同じ境界 (事前に固定)
HIGH_CHAOS_ALT = 0.8           # 感度用
N_BOOT = 4000
RNG = np.random.default_rng(20261118)


def final_win_odds(date: str, kekka_dir: Path) -> dict[str, dict[int, float]]:
    """rid16 → 馬番 → 最終単勝オッズ。勝ち馬は単勝配当/100、他は "(x.x)"。"""
    import csv
    import io
    p = kekka_dir / f"{date}.csv"
    out: dict[str, dict[int, float]] = {}
    rd = csv.reader(io.StringIO(build_site._decode(p.read_bytes())))
    next(rd, None)
    for row in rd:
        if len(row) < 9:
            continue
        rid16 = str(row[7]).strip()[:16]
        uma = build_site._int(row[4])
        raw = str(row[8]).strip()
        if uma is None or not raw:
            continue
        m = re.fullmatch(r"\(([\d.]+)\)", raw)
        if m:
            o = float(m.group(1))
        else:
            v = build_site._int(raw)
            if v is None:
                continue
            o = v / 100.0
        out.setdefault(rid16, {})[uma] = o
    return out


def load_rows(data_root: Path) -> tuple[list[dict], dict]:
    frozen = data_root / "reports" / "cowork_input_frozen"
    kekka_dir = data_root / "data" / "kekka"
    build_site.KEKKA_DIR = kekka_dir
    rows: list[dict] = []
    info = {"dates_used": [], "dates_no_frozen": [], "dates_no_kekka": [],
            "races_no_result": 0, "races": 0, "horses_no_final_odds": 0,
            "bets_excluded_no_finish": 0}
    kekka_dates = sorted(p.stem for p in kekka_dir.glob("2026????.csv") if START <= p.stem <= END)
    bundle_dates = sorted(p.name[:8] for p in frozen.glob("2026????_bundle.json") if START <= p.name[:8] <= END)
    info["dates_no_frozen"] = [d for d in kekka_dates if d not in bundle_dates]
    for d in bundle_dates:
        prov_p = frozen / f"{d}_bundle.provenance.json"
        prov = json.loads(prov_p.read_text(encoding="utf-8")) if prov_p.exists() else {}
        if not prov.get("verified_pre_race"):
            info["dates_no_frozen"].append(d)
            continue
        if not (kekka_dir / f"{d}.csv").exists():
            info["dates_no_kekka"].append(d)
            continue
        b = json.loads((frozen / f"{d}_bundle.json").read_text(encoding="utf-8"))
        era = build_site_model_era(d, b)
        res_all = build_site.parse_kekka(d, {})
        fin_all = final_win_odds(d, kekka_dir)
        info["dates_used"].append(d)
        for race in b.get("races", []):
            rid16 = str(race.get("race_id", ""))[:16]
            res = res_all.get(rid16)
            if not res or not res.get("top3"):
                info["races_no_result"] += 1
                continue
            info["races"] += 1
            fin = fin_all.get(rid16, {})
            bj = race.get("buy_judgment") or {}
            chaos = bj.get("chaos_pct")
            tan_by = (res.get("pays") or {}).get("tan_by") or {}
            fuku = (res.get("pays") or {}).get("fuku") or {}
            scratched = set(res.get("scratched") or [])
            no_finish = set(res.get("no_finish") or [])
            for h in race.get("horses", []):
                u = int(h.get("umaban", 0))
                if u in scratched:
                    continue
                pw, ps = h.get("p_win"), h.get("p_sho")
                fo = fin.get(u)
                if fo is None:
                    info["horses_no_final_odds"] += 1
                rows.append({
                    "date": d, "rid": rid16, "era": era, "model": str(b.get("model") or ""),
                    "chaos": chaos, "uma": u, "p_win": pw, "p_sho": ps,
                    "o_fin": fo, "o_exp": h.get("tansho_odds"), "fo_low": h.get("fuku_odds_low"),
                    "no_finish": u in no_finish,
                    "tan_ret": (tan_by.get(str(u)) or 0) * 1.0,   # 100 円あたり払戻
                    "fuku_ret": (fuku.get(str(u)) or 0) * 1.0,
                })
    return rows, info


def build_site_model_era(d: str, b: dict) -> str:
    from bundle_provenance import model_era
    return model_era(d, b)


def select(rows: list[dict], kind: str, thr: float) -> list[dict]:
    """kind: tan_all / tan_top1 / fuku_all / fuku_top1 / tan_exp_all / tan_exp_top1"""
    by_race: dict[str, list[dict]] = {}
    for r in rows:
        by_race.setdefault(r["rid"], []).append(r)
    out = []
    for rid, hs in by_race.items():
        cand = []
        for r in hs:
            if kind.startswith("tan_exp"):
                p, o, ret = r["p_win"], r["o_exp"], r["tan_ret"]
            elif kind.startswith("tan"):
                p, o, ret = r["p_win"], r["o_fin"], r["tan_ret"]
            else:
                p, o, ret = r["p_sho"], r["fo_low"], r["fuku_ret"]
            if p is None or o is None or o <= 0:
                continue
            ev = float(p) * float(o)
            if ev >= thr - 1e-12:
                cand.append((ev, r, ret))
        if not cand:
            continue
        if kind.endswith("top1"):
            cand = [max(cand, key=lambda t: (t[0], -t[1]["uma"]))]
        for ev, r, ret in cand:
            out.append({**r, "ev": ev, "ret": ret})
    return out


def summarize(bets: list[dict], weight=None) -> dict:
    """weight: bet → 賭け金倍率 (縮小ルール用)。no_finish の馬券は集計外。"""
    bets = [b for b in bets if not b["no_finish"]]
    if not bets:
        return {"n_bets": 0, "n_races": 0}
    w = np.array([weight(b) if weight else 1.0 for b in bets])
    ret = np.array([b["ret"] for b in bets]) / 100.0
    hit = np.array([b["ret"] > 0 for b in bets])
    rid = np.array([b["rid"] for b in bets])
    stake = w.sum()
    roi = float((ret * w).sum() / stake)
    # レース単位 bootstrap (同じレース内の買い目はまとめて引く)
    uniq, inv = np.unique(rid, return_inverse=True)
    s_r = np.bincount(inv, weights=w)
    r_r = np.bincount(inv, weights=ret * w)
    idx = RNG.integers(0, len(uniq), size=(N_BOOT, len(uniq)))
    boots = r_r[idx].sum(1) / s_r[idx].sum(1)
    # 上位 10 払戻を除いた回収率 (高配当依存の確認)
    order = np.argsort(-(ret * w))
    keep = order[10:]
    roi_ex10 = float((ret[keep] * w[keep]).sum() / w[keep].sum()) if len(keep) else None
    return {
        "n_bets": int(len(bets)), "n_races": int(len(uniq)), "n_hits": int(hit.sum()),
        "hit_rate": float(hit.mean()), "stake_units": float(stake), "roi": roi,
        "roi_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "roi_ex_top10": roi_ex10, "mean_odds_hit": float(ret[hit].mean()) if hit.any() else None,
    }


def paired_diff(bets: list[dict], wfun) -> dict:
    """縮小ルール (wfun) − 均等 の回収率差をレース単位で対応付けて bootstrap。"""
    bets = [b for b in bets if not b["no_finish"]]
    rid = np.array([b["rid"] for b in bets])
    ret = np.array([b["ret"] for b in bets]) / 100.0
    w = np.array([wfun(b) for b in bets])
    uniq, inv = np.unique(rid, return_inverse=True)
    s1, r1 = np.bincount(inv), np.bincount(inv, weights=ret)
    s2, r2 = np.bincount(inv, weights=w), np.bincount(inv, weights=ret * w)
    base = r1.sum() / s1.sum()
    shr = r2.sum() / s2.sum()
    idx = RNG.integers(0, len(uniq), size=(N_BOOT, len(uniq)))
    d = r2[idx].sum(1) / s2[idx].sum(1) - r1[idx].sum(1) / s1[idx].sum(1)
    return {"roi_flat": float(base), "roi_shrink": float(shr), "diff": float(shr - base),
            "diff_ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_improve": float((d > 0).mean()),
            "stake_flat": float(s1.sum()), "stake_shrink": float(s2.sum()),
            "profit_flat": float(r1.sum() - s1.sum()), "profit_shrink": float(r2.sum() - s2.sum())}


CHAOS_BANDS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0001)]


def band_of(c) -> str:
    if c is None:
        return "なし(〜05-31)"
    for lo, hi in CHAOS_BANDS:
        if lo <= c < hi:
            return f"{lo:.1f}-{min(hi, 1.0):.1f}"
    return "?"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="E:/PyCaLiAI")
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    rows, info = load_rows(Path(a.data_root))
    kinds = ["tan_all", "tan_top1", "fuku_all", "fuku_top1", "tan_exp_all", "tan_exp_top1"]
    out = {"info": {**info, "n_dates": len(info["dates_used"]), "n_horse_rows": len(rows)},
           "main": {}, "by_era": {}, "by_chaos": {}, "shrink": {}, "baseline": {}}
    eras = sorted({r["era"] for r in rows})
    out["info"]["eras"] = {e: {"dates": len({r["date"] for r in rows if r["era"] == e}),
                               "races": len({r["rid"] for r in rows if r["era"] == e})} for e in eras}
    out["info"]["races_with_chaos"] = len({r["rid"] for r in rows if r["chaos"] is not None})
    # 基準線: 全馬の単勝/複勝を均等に買ったときの回収率 (控除の目安)
    allt = [{**r, "ret": r["tan_ret"]} for r in rows if r["o_fin"] is not None]
    allf = [{**r, "ret": r["fuku_ret"]} for r in rows if r["fo_low"] is not None]
    out["baseline"] = {"tan_all_horses": summarize(allt), "fuku_all_horses": summarize(allf)}
    for k in kinds:
        for t in THRESHOLDS:
            bets = select(rows, k, t)
            key = f"{k}@{t}"
            out["main"][key] = summarize(bets)
            out["by_era"][key] = {e: summarize([b for b in bets if b["era"] == e]) for e in eras}
            out["by_chaos"][key] = {bd: summarize([b for b in bets if band_of(b["chaos"]) == bd])
                                    for bd in [band_of(None)] + [band_of(lo) for lo, _ in CHAOS_BANDS]}
            sh = {}
            for name, cut in (("p667", HIGH_CHAOS_PRIMARY), ("p80", HIGH_CHAOS_ALT)):
                wf = (lambda b, c=cut: 0.5 if (b["chaos"] is not None and b["chaos"] >= c) else 1.0)
                with_chaos = [b for b in bets if b["chaos"] is not None]
                sh[name] = {"all_races": paired_diff(bets, wf) if bets else None,
                            "chaos_races_only": paired_diff(with_chaos, wf) if with_chaos else None}
            out["shrink"][key] = sh
    Path(a.out_json).write_text(json.dumps(out, ensure_ascii=False, indent=None), encoding="utf-8")
    m = out["main"]
    for k in kinds:
        for t in THRESHOLDS:
            s = m[f"{k}@{t}"]
            if s.get("n_bets"):
                print(f"{k:13s} {t:<5} bets={s['n_bets']:5d} races={s['n_races']:5d} hit={s['hit_rate']*100:5.1f}% "
                      f"roi={s['roi']*100:6.1f}% CI[{s['roi_ci95'][0]*100:.1f},{s['roi_ci95'][1]*100:.1f}] "
                      f"ex10={s['roi_ex_top10']*100:.1f}%")
    print(json.dumps(out["info"], ensure_ascii=False)[:1500])


if __name__ == "__main__":
    main()
