# -*- coding: utf-8 -*-
"""
shadow_a95.py — 「◎オッズ帯 × 上位1点券種 ROI>=95 セル、1R 1万円 均等配分」の前向き shadow 台帳
==================================================================================================
**実弾ゼロ・本番の買い目/サイト/Discord には一切出さない。** レース前に買い目を凍結記録し、
結果が出た後に決済するだけの紙上ライン。

  decide  : bundle から買い目を作り decisions/{date}.json に書く (上書きしない)。
            発走前 (当日 09:00 JST より前) に書けた場合だけ preregistered=true。
  settle  : decisions と kekka / wide_kekka から決済し settlements/{date}.json に書く。
            bundle は読まない (レース後に bundle が再生成されても買い目は変わらない)。
  report  : 累積の集計。前向き (preregistered かつ start_date 以降) と 記述用 (late) を分けて出す。

方策の正本: data/shadow_policies/a95_equal_v1.json (sha256 自己検査つき。変更は新しい policy id で)
出力      : data/shadow_ledger/a95_equal_v1/

使い方:
  python shadow_a95.py decide --date 20261003
  python shadow_a95.py settle --date 20261003
  python shadow_a95.py report
"""
from __future__ import annotations
import argparse, csv, hashlib, io, json, re, sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
POLICY_PATH = BASE / "data" / "shadow_policies" / "a95_equal_v1.json"
LEDGER = BASE / "data" / "shadow_ledger" / "a95_equal_v1"
KEKKA_DIR = BASE / "data" / "kekka"
BUNDLE_DIR = BASE / "reports" / "cowork_input"
JST = timezone(timedelta(hours=9))
KINDS = ["tan", "fuku", "umaren", "umatan", "wide", "wakuren", "sanpuku", "sanrentan"]
JP = dict(tan="単勝", fuku="複勝", umaren="馬連", umatan="馬単", wide="ワイド", wakuren="枠連",
          sanpuku="三連複", sanrentan="三連単")


# ------------------------------------------------------------------ policy
def _canon(obj) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def load_policy(path: Path = POLICY_PATH) -> dict:
    pol = json.loads(path.read_text(encoding="utf-8"))
    body = {k: v for k, v in pol.items() if k != "policy_sha256"}
    sha = hashlib.sha256(_canon(body)).hexdigest()
    if pol.get("policy_sha256") != sha:
        raise SystemExit(f"policy sha256 不一致: file={pol.get('policy_sha256')} calc={sha} "
                         f"(方策を変えるなら新しい policy id を作ること)")
    return pol


def band_of(odds: float, pol: dict) -> str:
    for lo, hi, name in pol["bands"]:
        if lo <= odds < (hi if hi is not None else float("inf")):
            return name
    return pol["bands"][-1][2]


def equal_stakes(kinds: list[str], cap: int, unit: int) -> dict[str, int]:
    """cap を kinds に均等割り (unit 格子)。端数は先頭から unit ずつ。"""
    if not kinds:
        return {}
    base = (cap // len(kinds)) // unit * unit
    st = {k: base for k in kinds}
    rem = cap - base * len(kinds)
    i = 0
    while rem >= unit:
        st[kinds[i % len(kinds)]] += unit
        rem -= unit
        i += 1
    return st


def waku_of(umaban: int, n: int) -> int:
    """JRA の枠番割当 (出馬表確定時の頭数 n)。"""
    if n <= 8:
        return umaban
    if n <= 16:
        singles = 16 - n
        return umaban if umaban <= singles else singles + (umaban - singles + 1) // 2
    if n == 17:
        return (umaban + 1) // 2 if umaban <= 14 else 8
    if umaban <= 12:
        return (umaban + 1) // 2
    return 7 if umaban <= 15 else 8


# ------------------------------------------------------------------ decide
def _rid16(x) -> str:
    return re.sub(r"\D", "", str(x or ""))[:16]


def build_decisions(bundle: dict, pol: dict) -> list[dict]:
    races = bundle["races"] if isinstance(bundle["races"], list) else list(bundle["races"].values())
    out = []
    for r in races:
        rid = _rid16(r.get("race_id"))
        hs = [h for h in r.get("horses", []) if h.get("p_win") is not None and h.get("umaban") is not None]
        rec = dict(rid=rid, n=len(hs))
        if len(hs) < 5:
            rec.update(valid=False, reason="few_horses"); out.append(rec); continue
        order = sorted(hs, key=lambda h: -float(h["p_win"]))
        top = order[:3]

        def _o(h):
            try:
                v = float(h.get("tansho_odds"))
                return v if v > 0 and np.isfinite(v) else None
            except Exception:
                return None
        a = [int(h["umaban"]) for h in top]
        o = [_o(h) for h in top]
        rec.update(a1=a[0], a2=a[1], a3=a[2], o1=o[0], o2=o[1], o3=o[2],
                   p1=float(top[0]["p_win"]), p2=float(top[1]["p_win"]), p3=float(top[2]["p_win"]),
                   waku1=waku_of(a[0], len(hs)), waku2=waku_of(a[1], len(hs)))
        if o[0] is None:
            rec.update(valid=False, reason="no_odds"); out.append(rec); continue
        band = band_of(o[0], pol)
        kinds = pol["plan"][band]
        rec.update(valid=True, band=band,
                   band_o2=(band_of(o[1], pol) if o[1] else None),   # 記述列 (設計には使わない)
                   band_o3=(band_of(o[2], pol) if o[2] else None),
                   tickets=equal_stakes(kinds, pol["cap_per_race_yen"], pol["unit_yen"]))
        out.append(rec)
    return out


def cmd_decide(date: str, pol: dict) -> int:
    dpath = LEDGER / "decisions" / f"{date}.json"
    if dpath.exists():
        print(f"[a95] decisions 既存 → 上書きしない: {dpath.name}")
        return 0
    bp = BUNDLE_DIR / f"{date}_bundle.json"
    if not bp.exists():
        print(f"[a95] bundle なし: {bp}"); return 2
    raw = bp.read_bytes()
    now = datetime.now(JST)
    deadline = datetime.strptime(date, "%Y%m%d").replace(tzinfo=JST, hour=9)
    prereg = now < deadline
    decs = build_decisions(json.loads(raw.decode("utf-8")), pol)
    rec = dict(policy_id=pol["policy_id"], policy_sha256=pol["policy_sha256"], date=date,
               decided_at=now.isoformat(timespec="seconds"), preregistered=bool(prereg),
               late_reason=(None if prereg else "decided_after_0900_JST_of_race_day"),
               bundle_sha256=hashlib.sha256(raw).hexdigest(), races=decs)
    dpath.parent.mkdir(parents=True, exist_ok=True)
    tmp = dpath.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(dpath)
    nv = sum(1 for d in decs if d.get("valid"))
    print(f"[a95] decide {date}: {nv}/{len(decs)}R 有効 preregistered={prereg} → decisions/{dpath.name}")
    return 0


# ------------------------------------------------------------------ settle
def _decode(b: bytes) -> str:
    for enc in ("cp932", "utf-8-sig", "utf-8"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            continue
    return b.decode("cp932", errors="replace")


def _num(x) -> float:
    s = str(x).strip()
    if not s or s.startswith("(") or s.lower() == "nan":
        return float("nan")
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return float("nan")


def load_wide(path: Path | None = None) -> dict:
    p = path or (KEKKA_DIR / "wide_kekka.csv")
    if not p.exists():
        return {}
    pat = re.compile(r"(\d+)\s*[-―]\s*(\d+)\s*[\\¥￥]\s*(\d+)")
    out = {}
    for row in csv.reader(io.StringIO(_decode(p.read_bytes()))):
        if len(row) < 10:
            continue
        try:
            key = (int(row[0]), int(row[1]), int(row[2]), str(row[3]).strip(), int(row[4]))
        except ValueError:
            continue
        pairs = {}
        for chunk in (row[9] or "").split("/"):
            m = pat.search(chunk)
            if m:
                a, b, pay = int(m.group(1)), int(m.group(2)), int(m.group(3))
                pairs[f"{min(a, b)}-{max(a, b)}"] = pay
        if pairs:
            out[key] = pairs
    return out


def load_kekka(date: str, wide: dict, kekka_dir: Path | None = None) -> dict:
    p = (kekka_dir or KEKKA_DIR) / f"{date}.csv"
    if not p.exists():
        return {}
    rows = list(csv.DictReader(io.StringIO(_decode(p.read_bytes()))))
    by = {}
    for r in rows:
        by.setdefault(_rid16(r.get("レースID(新)")), []).append(r)
    res = {}
    for rid, g in by.items():
        fin = {}
        for r in g:
            f = _num(r.get("確定着順"))
            if np.isfinite(f):
                fin.setdefault(int(f), []).append(r)
        if any(len(fin.get(k, [])) != 1 for k in (1, 2, 3)):
            res[rid] = dict(valid=False, reason="dead_heat_or_missing_top3")
            continue
        r1, r2, r3 = fin[1][0], fin[2][0], fin[3][0]
        d = str(r1["日付"]).strip()
        wkey = (2000 + int(d[:2]), int(d[2:4]), int(d[4:6]), str(r1["場所"]).strip(), int(_num(r1["Ｒ"])))
        res[rid] = dict(
            valid=True, first=int(_num(r1["馬番"])), second=int(_num(r2["馬番"])), third=int(_num(r3["馬番"])),
            w1=int(_num(r1["枠番"])), w2=int(_num(r2["枠番"])),
            tan=_num(r1.get("単勝配当")),
            fuku={int(_num(r["馬番"])): _num(r.get("複勝配当")) for r in (r1, r2, r3)},
            wakuren=_num(r1.get("枠連")), umaren=_num(r1.get("馬連")), umatan=_num(r1.get("馬単")),
            sanpuku=_num(r1.get("３連複")), sanrentan=_num(r1.get("３連単")),
            wide=wide.get(wkey),
        )
    return res


def payout_per_100(kind: str, d: dict, k: dict) -> float:
    """買い目 kind の 100 円あたり払戻。外れは 0、払戻情報が無い場合は nan。"""
    a1, a2, a3 = d["a1"], d["a2"], d["a3"]
    f, s, t = k["first"], k["second"], k["third"]

    def ok(v):  # 的中時の払戻が欠損なら nan を返して呼び元で無効化
        return v if np.isfinite(v) else float("nan")
    if kind == "tan":
        return ok(k["tan"]) if a1 == f else 0.0
    if kind == "fuku":
        if a1 not in (f, s, t):
            return 0.0
        v = k["fuku"].get(a1, float("nan"))
        return v if np.isfinite(v) else 0.0   # 7 頭以下の 3 着は複勝対象外
    if kind == "umaren":
        return ok(k["umaren"]) if {a1, a2} == {f, s} else 0.0
    if kind == "umatan":
        return ok(k["umatan"]) if (a1, a2) == (f, s) else 0.0
    if kind == "wakuren":
        return ok(k["wakuren"]) if {d["waku1"], d["waku2"]} == {k["w1"], k["w2"]} else 0.0
    if kind == "sanpuku":
        return ok(k["sanpuku"]) if {a1, a2, a3} == {f, s, t} else 0.0
    if kind == "sanrentan":
        return ok(k["sanrentan"]) if (a1, a2, a3) == (f, s, t) else 0.0
    if kind == "wide":
        if k["wide"] is None:
            return float("nan")
        return float(k["wide"].get(f"{min(a1, a2)}-{max(a1, a2)}", 0.0))
    raise ValueError(kind)


def settle_race(d: dict, k: dict, pol: dict) -> dict:
    if not d.get("valid"):
        return dict(rid=d["rid"], valid=False, reason=d.get("reason"))
    if k is None:
        return dict(rid=d["rid"], valid=False, reason="no_kekka")
    if not k.get("valid"):
        return dict(rid=d["rid"], valid=False, reason=k.get("reason"))
    per100 = {kind: payout_per_100(kind, d, k) for kind in KINDS}
    arms = {}
    plans = {"a95_equal": d["tickets"]}
    for name, kinds in pol["reference_arms"].items():
        plans[name] = {kk: pol["unit_yen"] for kk in kinds}
    for name, tk in plans.items():
        if any(not np.isfinite(per100[kk]) for kk in tk):
            arms[name] = dict(valid=False, reason="missing_payout")
            continue
        stake = int(sum(tk.values()))
        ret = float(sum(per100[kk] / 100.0 * s for kk, s in tk.items()))
        arms[name] = dict(valid=True, stake=stake, ret=round(ret, 1),
                          hit=bool(any(per100[kk] > 0 for kk in tk)),
                          by_kind={kk: dict(stake=int(s), ret=round(per100[kk] / 100.0 * s, 1)) for kk, s in tk.items()})
    return dict(rid=d["rid"], valid=True, band=d["band"], band_o2=d.get("band_o2"), band_o3=d.get("band_o3"),
                o1=d["o1"], a=[d["a1"], d["a2"], d["a3"]], result=[k["first"], k["second"], k["third"]],
                arms=arms)


def cmd_settle(date: str, pol: dict) -> int:
    dpath = LEDGER / "decisions" / f"{date}.json"
    if not dpath.exists():
        print(f"[a95] decisions なし ({date})。先に decide を実行すること"); return 2
    dec = json.loads(dpath.read_text(encoding="utf-8"))
    if dec["policy_sha256"] != pol["policy_sha256"]:
        print("[a95] decisions の policy sha が現行と違う → 決済しない"); return 3
    kek = load_kekka(date, load_wide())
    if not kek:
        print(f"[a95] kekka 未配置 ({date}) → 未決済のまま"); return 0
    races = [settle_race(d, kek.get(d["rid"]), pol) for d in dec["races"]]
    rec = dict(policy_id=pol["policy_id"], policy_sha256=pol["policy_sha256"], date=date,
               preregistered=dec["preregistered"], decided_at=dec["decided_at"],
               settled_at=datetime.now(JST).isoformat(timespec="seconds"), races=races)
    spath = LEDGER / "settlements" / f"{date}.json"
    spath.parent.mkdir(parents=True, exist_ok=True)
    tmp = spath.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(spath)
    v = [r for r in races if r.get("valid") and r["arms"]["a95_equal"]["valid"]]
    st = sum(r["arms"]["a95_equal"]["stake"] for r in v); rt = sum(r["arms"]["a95_equal"]["ret"] for r in v)
    print(f"[a95] settle {date}: 有効 {len(v)}/{len(races)}R  a95_equal 投資 ¥{st:,} 払戻 ¥{rt:,.0f} "
          f"({(100*rt/st if st else 0):.1f}%)  preregistered={dec['preregistered']}")
    return 0


# ------------------------------------------------------------------ report
def _collect(pol: dict):
    rows = []
    for sp in sorted((LEDGER / "settlements").glob("*.json")):
        s = json.loads(sp.read_text(encoding="utf-8"))
        if s["policy_sha256"] != pol["policy_sha256"]:
            continue
        fwd = bool(s["preregistered"]) and s["date"] >= pol["registration"]["start_date"]
        for r in s["races"]:
            if not r.get("valid"):
                continue
            for arm, a in r["arms"].items():
                if a.get("valid"):
                    rows.append(dict(date=s["date"], fwd=fwd, arm=arm, band=r["band"], stake=a["stake"],
                                     ret=a["ret"], hit=a["hit"], by_kind=a["by_kind"]))
    return rows


def _boot(rows, reps=3000, seed=0):
    days = sorted({r["date"] for r in rows})
    if len(days) < 2:
        return (float("nan"), float("nan"))
    by = {d: [0.0, 0.0] for d in days}
    for r in rows:
        by[r["date"]][0] += r["stake"]; by[r["date"]][1] += r["ret"]
    c = np.array([by[d][0] for d in days]); v = np.array([by[d][1] for d in days])
    idx = np.random.default_rng(seed).integers(0, len(days), size=(reps, len(days)))
    roi = 100 * v[idx].sum(1) / c[idx].sum(1)
    return (float(np.percentile(roi, 2.5)), float(np.percentile(roi, 97.5)))


def _summ(rows, title, pol):
    print(f"\n=== {title} ===")
    if not rows:
        print("  (記録なし)"); return
    reg = pol["registration"]
    for arm in ["a95_equal"] + list(pol["reference_arms"]):
        rs = [r for r in rows if r["arm"] == arm]
        if not rs:
            continue
        n = len(rs); st = sum(r["stake"] for r in rs); rt = sum(r["ret"] for r in rs)
        hit = np.mean([r["hit"] for r in rs]); plus = np.mean([r["ret"] >= r["stake"] for r in rs])
        trig = np.mean([r["hit"] and r["ret"] < r["stake"] for r in rs])
        lo, hi = _boot(rs)
        print(f"  {arm:<11} R={n:>5} 日={len({r['date'] for r in rs}):>3} | 投資 ¥{st:>11,} 収支 ¥{rt-st:>+12,.0f} "
              f"ROI {100*rt/st:6.1f}% [{lo:5.1f},{hi:5.1f}] (記述) | R的中 {100*hit:4.1f}% 収支+R {100*plus:4.1f}% トリガミ {100*trig:4.1f}%")
        if arm == "a95_equal":
            se = np.sqrt(max(hit * (1 - hit), 1e-9) / n); ub = hit + 1.645 * se
            need = reg["min_valid_races"]
            verdict = ("未到達" if n < need else ("棄却 (R的中率の上限が床未満)" if ub < reg["reject_if_hit_upper_below"] else "棄却されず"))
            print(f"    登録判定: R的中率 {100*hit:.1f}% 片側95%上限 {100*ub:.1f}% / 床 {100*reg['reject_if_hit_upper_below']:.1f}% / "
                  f"有効R {n}/{need} → {verdict}")
            for b in [x[2] for x in pol["bands"]]:
                bs = [r for r in rs if r["band"] == b]
                if bs:
                    s2 = sum(r["stake"] for r in bs); r2 = sum(r["ret"] for r in bs)
                    print(f"      {b:<6} R={len(bs):>4} ROI {100*r2/s2:6.1f}% R的中 {100*np.mean([r['hit'] for r in bs]):4.1f}%")
            kk = {}
            for r in rs:
                for k, v in r["by_kind"].items():
                    a = kk.setdefault(k, [0, 0.0, 0, 0]); a[0] += v["stake"]; a[1] += v["ret"]; a[2] += 1; a[3] += int(v["ret"] > 0)
            print("      券種別: " + " | ".join(f"{JP[k]} {100*a[1]/a[0]:.0f}% ({a[3]}/{a[2]})" for k, a in kk.items()))


def cmd_report(pol: dict) -> int:
    rows = _collect(pol)
    print(f"policy {pol['policy_id']} sha256 {pol['policy_sha256'][:16]}…  前向き起算日 {pol['registration']['start_date']}")
    print("※ 実弾ゼロの shadow。ROI は記述のみ (500R の ROI 標準誤差は約 17pt)。登録判定は R的中率。")
    _summ([r for r in rows if r["fwd"]], "前向き (レース前に凍結した買い目のみ)", pol)
    _summ([r for r in rows if not r["fwd"]], "記述用 (事後に作った買い目・起算日前。判定に使わない)", pol)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("decide", "settle"):
        p = sub.add_parser(c); p.add_argument("--date", required=True)
    sub.add_parser("report")
    a = ap.parse_args()
    pol = load_policy()
    if a.cmd == "decide":
        return cmd_decide(a.date, pol)
    if a.cmd == "settle":
        return cmd_settle(a.date, pol)
    return cmd_report(pol)


if __name__ == "__main__":
    sys.exit(main())
