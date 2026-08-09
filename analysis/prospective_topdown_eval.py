# -*- coding: utf-8 -*-
"""topdown エンジン前向き検証 (Priority 1, 憲章 PYcALiAI_RESEARCH.md §21).

Experiment ID: P1-TOPDOWN-PROSPECTIVE-2026
Hypothesis   : リプレイ改善 (topdown 82.8% vs shape 74.1%, in-sample) が
               未来データ (2026-08-15 以降の実運用) でも再現する。
Baseline     : CB_ENGINE=shape (旧経路 + 構築層4修正) のシャドー買い目
               (reports/engine_shadow/{date}_shadow.json — 同一レース・同一 T-10 オッズ)。
Treatment    : CB_ENGINE=topdown の実買い目 (reports/cowork_output/{date}_bets.json)。
Data         : 確定払戻 data/kekka/{date}.csv (generate_results の照合ロジックを再利用)。
判定 (事前登録, 2026-08-10):
  n_bets(topdown) >= 300 到達後に paired bootstrap (レース単位リサンプル 10,000回) で
  ΔROI = ROI_topdown - ROI_shape の CI95 を算出し、
    PASS         : Δ > 0 かつ CI95下限 > -2pt
    FAIL         : Δ < 0 かつ CI95上限 < +2pt
    INCONCLUSIVE : それ以外 (必要追加サンプルを報告して継続)
  未来の結果を見てエンジンを切り替えない (判定日まで topdown 固定)。
Leakage check: shadow は本番と同一 T-10 スナップショットで同時生成 (未来オッズ不使用)。
               決済は確定 kekka のみ。START_DATE=20260815 (topdown 既定化 08-09 20:46 より後の
               最初の開催週) 以前は前向きに含めない。

実行: python -m analysis.prospective_topdown_eval  [--start 20260815] [--n-boot 10000]
出力: reports/prospective_topdown_eval.json + stdout サマリ
"""
from __future__ import annotations
import argparse, json, sys
from datetime import datetime
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from generate_results import (  # noqa: E402
    load_kekka_all, get_race_kk, match_cowork_bet, parse_race_id_16, _safe_num,
)

START_DATE = 20260815   # 前向き開始日 (これ未満は in-sample/旧エンジン期間につき除外)
SEED = 42


def settle_races(races: list[dict], date_key: str, kekka_cache: dict) -> list[dict]:
    """レース entry 群を決済して [{race_id, stake, ret, hits, n_bets, settled}] を返す。
    実効投資 = 購入額 - 返還 (generate_results と同一流儀)。"""
    out = []
    for cd in races:
        rid = str(cd.get("race_id", ""))
        parsed = parse_race_id_16(rid)
        if not parsed:
            continue
        race_kk = get_race_kk(kekka_cache, date_key, parsed["place_name"], parsed["race_no"])
        stake = ret = 0.0
        hits = 0
        for b in (cd.get("bets") or []):
            amount = _safe_num(b.get("購入額"))
            hit, pay100, refund_ratio = (False, 0.0, 0.0)
            if not race_kk.empty:
                try:
                    hit, pay100, refund_ratio = match_cowork_bet(
                        b, race_kk, date_key=date_key,
                        place=parsed["place_name"], race_no=str(parsed["race_no"]))
                except Exception as e:
                    print(f"[warn] 照合失敗 {rid} {b.get('馬券種')} {b.get('買い目')}: {e}")
            stake += amount * (1 - refund_ratio)
            ret += (amount * pay100 / 100.0) if hit else 0.0
            hits += int(hit)
        out.append({"race_id": rid, "date": date_key, "stake": stake, "ret": ret,
                    "hits": hits, "n_bets": len(cd.get("bets") or []),
                    "nature": cd.get("race_nature", ""),
                    "settled": not race_kk.empty})
    return out


def arm_summary(rs: list[dict]) -> dict:
    st = sum(r["stake"] for r in rs)
    rt = sum(r["ret"] for r in rs)
    nb = sum(r["n_bets"] for r in rs)
    hi = sum(r["hits"] for r in rs)
    pnl = np.cumsum([r["ret"] - r["stake"] for r in rs]) if rs else np.array([0.0])
    return {"races": len(rs), "races_bet": sum(1 for r in rs if r["n_bets"] > 0),
            "bets": nb, "stake": round(st), "ret": round(rt), "pnl": round(rt - st),
            "roi": round(rt / st * 100, 2) if st > 0 else None,
            "hit_count": hi, "hit_rate": round(hi / nb * 100, 1) if nb else None,
            "max_drawdown": round(float(pnl.min())) if len(pnl) else 0}


def paired_bootstrap(td: list[dict], sh: list[dict], n_boot: int) -> dict:
    """レース単位ペアリサンプルで ΔROI (pt) の分布を得る。両腕とも stake>0 の試行のみ有効。"""
    assert [r["race_id"] for r in td] == [r["race_id"] for r in sh], "pairing broken"
    n = len(td)
    rng = np.random.default_rng(SEED)
    ts, tr = np.array([r["stake"] for r in td]), np.array([r["ret"] for r in td])
    ss, sr = np.array([r["stake"] for r in sh]), np.array([r["ret"] for r in sh])
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        a, b = ts[idx].sum(), ss[idx].sum()
        if a > 0 and b > 0:
            deltas.append(tr[idx].sum() / a * 100 - sr[idx].sum() / b * 100)
    d = np.array(deltas)
    if not len(d):
        return {"n_valid": 0}
    return {"n_valid": int(len(d)), "delta_mean": round(float(d.mean()), 2),
            "ci95": [round(float(np.percentile(d, 2.5)), 2),
                     round(float(np.percentile(d, 97.5)), 2)],
            "p_improve": round(float((d > 0).mean()), 3)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=START_DATE)
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()

    kekka_cache = load_kekka_all()
    sh_dir = BASE / "reports" / "engine_shadow"
    co_dir = BASE / "reports" / "cowork_output"
    td_races, sh_races = [], []
    for f in sorted(sh_dir.glob("????????_shadow.json")):
        date_key = f.stem[:8]
        if int(date_key) < args.start:
            continue
        shadow = json.load(open(f, encoding="utf-8"))
        bets_f = co_dir / f"{date_key}_bets.json"
        if not bets_f.exists():
            print(f"[warn] {date_key}: shadow はあるが bets.json 不在 — skip")
            continue
        prim = json.load(open(bets_f, encoding="utf-8"))
        prim_races = prim.get("bets", prim if isinstance(prim, list) else [])
        sh_by_rid = {str(e.get("race_id")): e for e in shadow.get("races", [])}
        # ペア成立レースのみ (primary が topdown で shadow が存在する race)
        for cd in prim_races:
            rid = str(cd.get("race_id"))
            if rid in sh_by_rid and cd.get("race_nature") in ("topdown", "見送り"):
                td_races.append((date_key, cd))
                sh_races.append((date_key, sh_by_rid[rid]))

    if not td_races:
        print("[INFO] 前向きペアデータ 0R (開始日以降の shadow ログ未蓄積)。"
              "初回開催週 (2026-08-15/16) の T-10 実行後に再実行のこと。")
        return 0

    td_s, sh_s = [], []
    for (dk, cd), (_, sd) in zip(td_races, sh_races):
        td_s += settle_races([cd], dk, kekka_cache)
        sh_s += settle_races([sd], dk, kekka_cache)
    # 未決着 (kekka 未到達) は判定から除外
    keep = [i for i, r in enumerate(td_s) if r["settled"]]
    td_s = [td_s[i] for i in keep]; sh_s = [sh_s[i] for i in keep]

    res = {"experiment_id": "P1-TOPDOWN-PROSPECTIVE-2026",
           "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           "start_date": args.start, "paired_races_settled": len(td_s),
           "topdown": arm_summary(td_s), "shape_shadow": arm_summary(sh_s),
           "paired_bootstrap": paired_bootstrap(td_s, sh_s, args.n_boot)}
    nb = res["topdown"]["bets"]
    res["judgment_ready"] = nb >= 300
    if res["judgment_ready"]:
        pb = res["paired_bootstrap"]
        if pb.get("n_valid"):
            d, lo, hi = pb["delta_mean"], pb["ci95"][0], pb["ci95"][1]
            res["verdict"] = ("PASS" if d > 0 and lo > -2.0 else
                              "FAIL" if d < 0 and hi < 2.0 else "INCONCLUSIVE")
    else:
        res["verdict"] = f"PENDING (bets {nb}/300)"

    out = BASE / "reports" / "prospective_topdown_eval.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"\n[saved] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
