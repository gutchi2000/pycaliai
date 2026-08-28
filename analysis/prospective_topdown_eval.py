# -*- coding: utf-8 -*-
"""topdown エンジン前向き検証 (Priority 1, 憲章 PYcALiAI_RESEARCH.md §21).

Experiment ID: data/production_policy.json の prospective.experiment_id
Hypothesis   : リプレイ改善 (topdown 82.8% vs shape 74.1%, in-sample) が
               policy effective_from 以降の未来データ でも再現する。
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
               決済は確定 kekka のみ。START_DATE と policy_id は data/production_policy.json で凍結し、異なる来歴を混ぜない。

実行: python -m analysis.prospective_topdown_eval  [--start YYYYMMDD] [--policy-id ID] [--n-boot 10000]
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
from production_policy import load_policy  # noqa: E402

_POLICY = load_policy()
START_DATE = int(_POLICY["prospective"]["start_date"])
EXPECTED_POLICY_ID = str(_POLICY["policy_id"])
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
    cum = np.cumsum([r["ret"] - r["stake"] for r in rs]) if rs else np.array([0.0])
    peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]
    dd = float((cum - peak).min()) if len(cum) else 0.0   # ピークからの最大下落 (<=0)
    return {"races": len(rs), "races_bet": sum(1 for r in rs if r["n_bets"] > 0),
            "bets": nb, "stake": round(st), "ret": round(rt), "pnl": round(rt - st),
            "roi": round(rt / st * 100, 2) if st > 0 else None,
            "hit_count": hi, "hit_rate": round(hi / nb * 100, 1) if nb else None,
            "max_drawdown": round(dd)}


def by_nature(rs: list[dict]) -> dict:
    """race_nature 別の集計 (shape 腕は 本命勝負/◎軸/広め流し/... に分解される)。"""
    grp: dict[str, list[dict]] = {}
    for r in rs:
        grp.setdefault(r["nature"] or "?", []).append(r)
    return {k: arm_summary(v) for k, v in sorted(grp.items())}


def paired_bootstrap(td: list[dict], sh: list[dict], n_boot: int) -> dict:
    """レース単位ペアリサンプルで ΔROI (pt) / Δprofit (円) の分布を得る。"""
    assert [r["race_id"] for r in td] == [r["race_id"] for r in sh], "pairing broken"
    n = len(td)
    if n == 0:
        return {"n_valid": 0}
    rng = np.random.default_rng(SEED)
    ts, tr = np.array([r["stake"] for r in td]), np.array([r["ret"] for r in td])
    ss, sr = np.array([r["stake"] for r in sh]), np.array([r["ret"] for r in sh])
    d_roi, d_pnl = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        a, b = ts[idx].sum(), ss[idx].sum()
        d_pnl.append(float((tr[idx] - ts[idx]).sum() - (sr[idx] - ss[idx]).sum()))
        if a > 0 and b > 0:
            d_roi.append(tr[idx].sum() / a * 100 - sr[idx].sum() / b * 100)
    d = np.array(d_roi); dp = np.array(d_pnl)
    if not len(d):
        return {"n_valid": 0}
    return {"n_valid": int(len(d)),
            "delta_roi_mean": round(float(d.mean()), 2),
            "delta_roi_ci95": [round(float(np.percentile(d, 2.5)), 2),
                               round(float(np.percentile(d, 97.5)), 2)],
            "delta_pnl_mean": round(float(dp.mean())),
            "delta_pnl_ci95": [round(float(np.percentile(dp, 2.5))),
                               round(float(np.percentile(dp, 97.5)))],
            "p_improve": round(float((d > 0).mean()), 3)}


def outlier_dependence(td: list[dict], sh: list[dict], n_boot: int) -> dict:
    """ROI 一撃依存の点検: 各腕の最高払戻レースの寄与率 + そのレースを除いた
    leave-one-out ΔROI 再計算 (topdown 側の最高払戻レースを両腕から外す)。"""
    def top_share(rs):
        tot = sum(r["ret"] for r in rs)
        if tot <= 0 or not rs:
            return None
        best = max(rs, key=lambda r: r["ret"])
        return {"race_id": best["race_id"], "date": best["date"], "ret": round(best["ret"]),
                "share_of_total_return": round(best["ret"] / tot, 3)}
    res = {"topdown_top_race": top_share(td), "shape_top_race": top_share(sh)}
    if len(td) > 1:
        drop = max(range(len(td)), key=lambda i: td[i]["ret"])
        td2 = [r for i, r in enumerate(td) if i != drop]
        sh2 = [r for i, r in enumerate(sh) if i != drop]
        res["loo_drop_topdown_best"] = {
            "topdown_roi": arm_summary(td2)["roi"], "shape_roi": arm_summary(sh2)["roi"],
            "paired_bootstrap": paired_bootstrap(td2, sh2, n_boot)}
    return res


def coverage_check(dates: dict[str, dict]) -> list[dict]:
    """週次監視 (規律7): 日付ごとの実績/shadow ペア成立状況。異常があれば warn を付ける。"""
    rows = []
    for dk, c in sorted(dates.items()):
        warn = []
        if c["shadow"] == 0:
            warn.append("shadow 0R (t10 が --apply で走ったか確認)")
        if c["paired"] < c["primary_topdown"]:
            warn.append(f"未ペア {c['primary_topdown'] - c['paired']}R (shadow 欠落)")
        rows.append({"date": dk, **c, **({"warn": warn} if warn else {})})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=START_DATE)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--policy-id", default=EXPECTED_POLICY_ID,
                    help="このpolicy_idだけを集計する。不一致や欠損はエラー終了")
    args = ap.parse_args()
    policy = load_policy()
    expected_policy_id = str(args.policy_id)
    minimum_bets = int(policy["prospective"]["minimum_bets"])

    kekka_cache = load_kekka_all()
    sh_dir = BASE / "reports" / "engine_shadow"
    co_dir = BASE / "reports" / "cowork_output"
    td_races, sh_races = [], []
    cov: dict[str, dict] = {}
    lineage_errors: list[str] = []
    for f in sorted(sh_dir.glob("????????_shadow.json")):
        date_key = f.stem[:8]
        if int(date_key) < args.start:
            continue
        shadow = json.load(open(f, encoding="utf-8"))
        root_policy_id = (shadow.get("policy") or {}).get("policy_id")
        if shadow.get("races") and root_policy_id != expected_policy_id:
            lineage_errors.append(
                f"{date_key} shadow root policy={root_policy_id!r} expected={expected_policy_id!r}")
        bets_f = co_dir / f"{date_key}_bets.json"
        if not bets_f.exists():
            print(f"[warn] {date_key}: shadow はあるが bets.json 不在 — skip")
            cov[date_key] = {"primary_topdown": 0, "shadow": len(shadow.get("races", [])),
                             "paired": 0}
            continue
        prim = json.load(open(bets_f, encoding="utf-8"))
        prim_races = prim.get("bets", prim if isinstance(prim, list) else [])
        sh_by_rid = {str(e.get("race_id")): e for e in shadow.get("races", [])}
        # ペア成立レースのみ (primary が topdown/見送り で shadow が存在する race)
        n_td = n_pair = 0
        for cd in prim_races:
            rid = str(cd.get("race_id"))
            if cd.get("race_nature") in ("topdown", "見送り"):
                n_td += 1
                primary_pid = (cd.get("stamp") or {}).get("policy_id")
                shadow_entry = sh_by_rid.get(rid)
                shadow_pid = ((shadow_entry or {}).get("stamp") or {}).get("policy_id")
                if primary_pid != expected_policy_id:
                    lineage_errors.append(
                        f"{date_key}/{rid} primary policy={primary_pid!r} expected={expected_policy_id!r}")
                    continue
                if shadow_entry is None:
                    continue
                if shadow_pid != expected_policy_id:
                    lineage_errors.append(
                        f"{date_key}/{rid} shadow policy={shadow_pid!r} expected={expected_policy_id!r}")
                    continue
                n_pair += 1
                td_races.append((date_key, cd))
                sh_races.append((date_key, shadow_entry))
        cov[date_key] = {"primary_topdown": n_td, "shadow": len(sh_by_rid), "paired": n_pair}

    if lineage_errors:
        print("[ERROR] 前向きcohortのpolicy混在/欠損を検出。集計を中止します。")
        for msg in lineage_errors[:50]:
            print(f"  - {msg}")
        if len(lineage_errors) > 50:
            print(f"  ...ほか {len(lineage_errors) - 50}件")
        return 2

    if not td_races:
        print(f"[INFO] 前向きペアデータ 0R (policy={expected_policy_id}, "
              f"start={args.start})。開始日以降のT-10実行後に再実行のこと。")
        return 0

    td_s, sh_s = [], []
    for (dk, cd), (_, sd) in zip(td_races, sh_races):
        td_s += settle_races([cd], dk, kekka_cache)
        sh_s += settle_races([sd], dk, kekka_cache)
    # 未決着 (kekka 未到達) は判定から除外
    keep = [i for i, r in enumerate(td_s) if r["settled"]]
    td_s = [td_s[i] for i in keep]; sh_s = [sh_s[i] for i in keep]

    res = {"experiment_id": policy["prospective"]["experiment_id"],
           "policy_id": expected_policy_id,
           "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           "start_date": args.start, "paired_races_settled": len(td_s),
           "coverage": coverage_check(cov),
           "topdown": arm_summary(td_s), "shape_shadow": arm_summary(sh_s),
           "topdown_by_nature": by_nature(td_s), "shape_by_nature": by_nature(sh_s),
           "paired_bootstrap": paired_bootstrap(td_s, sh_s, args.n_boot),
           "outlier_dependence": outlier_dependence(td_s, sh_s, args.n_boot)}
    nb = res["topdown"]["bets"]
    res["minimum_bets"] = minimum_bets
    res["judgment_ready"] = nb >= minimum_bets
    if res["judgment_ready"]:
        pb = res["paired_bootstrap"]
        if pb.get("n_valid"):
            d = pb["delta_roi_mean"]; lo, hi = pb["delta_roi_ci95"]
            res["verdict"] = ("PASS" if d > 0 and lo > -2.0 else
                              "FAIL" if d < 0 and hi < 2.0 else "INCONCLUSIVE")
            # ROI 一撃依存の注記 (判定基準は変えない。PASS が最高払戻1Rの除去で
            # 消えるなら outlier_sensitive=true を付けて報告する)
            loo = res["outlier_dependence"].get("loo_drop_topdown_best", {})
            lpb = loo.get("paired_bootstrap", {})
            if res["verdict"] == "PASS" and lpb.get("n_valid"):
                ld, llo = lpb["delta_roi_mean"], lpb["delta_roi_ci95"][0]
                res["outlier_sensitive"] = not (ld > 0 and llo > -2.0)
    else:
        res["verdict"] = f"PENDING (bets {nb}/{minimum_bets})"

    out = BASE / "reports" / "prospective_topdown_eval.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"\n[saved] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
