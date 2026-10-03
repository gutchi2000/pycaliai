# -*- coding: utf-8 -*-
"""
serve_health.py — 本番配信の印が壊れていないかを毎週測る監視。

物差しは「AI 1位 (生 p_win 最大) と市場 1番人気 (bundle の単勝オッズ最小) の一致率」。
offline 2023-25 では 53.9%、serve が壊れていた 2026-04〜08 は 30〜35% だった。
着順を使わないので、土曜朝 (bundle 生成直後・レース前) にも判定できる。

窓: 直近 4 開催日。100R に満たなければ 100R を超えるまで過去へ延ばす
    (健全時の誤報 約4%、故障時 34% の検出 約99%)。
判定: 一致率 < 45% で WARN。kekka がある日は AI/市場の top3 率も併記する (記述のみ)。

実行: python serve_health.py [--notify] [--until 20261004]
  → data/serve_health.json を更新。WARN のとき exit 2 (--notify で Discord にも送る)。
本番の買い目・印には一切干渉しない。
"""
from __future__ import annotations
import argparse, datetime, json, re, sys
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent
BUNDLE_DIR = BASE / "reports" / "cowork_input"
KEKKA_DIR = BASE / "data" / "kekka"
OUT = BASE / "data" / "serve_health.json"

WARN_BELOW = 0.45      # 一致率の警告線
MIN_DAYS = 4
MIN_RACES = 100
OFFLINE_REF = 0.539    # 2023-25 OOS


def _odds(h) -> float | None:
    for k in ("tansho_odds", "odds", "tan_odds"):
        v = h.get(k)
        try:
            if v not in (None, "") and float(v) > 0:
                return float(v)
        except (TypeError, ValueError):
            pass
    return None


def bundle_rows(path: Path) -> list[dict]:
    b = json.loads(path.read_text(encoding="utf-8"))
    races = b["races"] if isinstance(b["races"], list) else list(b["races"].values())
    out = []
    for r in races:
        rid = re.sub(r"\D", "", str(r.get("race_id", "")))[:16]
        hs = [h for h in r.get("horses", []) if h.get("p_win") is not None and h.get("umaban") is not None]
        od = [(h, _odds(h)) for h in hs]
        od = [(h, o) for h, o in od if o is not None]
        if len(hs) < 5 or len(od) < 5:
            continue
        ai1 = int(max(hs, key=lambda h: float(h["p_win"]))["umaban"])
        fav = int(min(od, key=lambda t: t[1])[0]["umaban"])
        out.append(dict(rid=rid, ai1=ai1, fav=fav, agree=int(ai1 == fav)))
    return out


def top3_map(date: str) -> dict[str, set[int]]:
    p = KEKKA_DIR / f"{date}.csv"
    if not p.exists():
        return {}
    k = pd.read_csv(p, encoding="cp932", low_memory=False)
    k["rid16"] = k["レースID(新)"].astype(str).str[:16]
    k["ban"] = pd.to_numeric(k["馬番"], errors="coerce")
    k["fin"] = pd.to_numeric(k["確定着順"], errors="coerce")
    k = k[k.fin.between(1, 3) & k.ban.notna()]
    return {rid: {int(x) for x in g.ban} for rid, g in k.groupby("rid16")}


def collect(until: str | None) -> list[dict]:
    days = []
    for p in sorted(BUNDLE_DIR.glob("????????_bundle.json"), reverse=True):
        date = p.name[:8]
        if until and date > until:
            continue
        try:
            rows = bundle_rows(p)
        except (OSError, ValueError, KeyError) as e:
            print(f"  [skip] {p.name}: {e}")
            continue
        if not rows:
            continue
        t3 = top3_map(date)
        settled = [r for r in rows if r["rid"] in t3]
        days.append(dict(
            date=date, n=len(rows), agree=sum(r["agree"] for r in rows),
            n_settled=len(settled),
            ai_top3=sum(r["ai1"] in t3[r["rid"]] for r in settled),
            fav_top3=sum(r["fav"] in t3[r["rid"]] for r in settled)))
        if len(days) >= MIN_DAYS and sum(d["n"] for d in days) >= MIN_RACES:
            break
    return days


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--until", default=None, help="この日付(YYYYMMDD)までの bundle で判定")
    ap.add_argument("--notify", action="store_true", help="WARN を Discord にも送る")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    days = collect(a.until)
    n = sum(d["n"] for d in days)
    if n < MIN_RACES:
        print(f"serve_health: 判定保留 (オッズ付き {n}R < {MIN_RACES}R)")
        return 0
    rate = sum(d["agree"] for d in days) / n
    ns = sum(d["n_settled"] for d in days)
    status = "WARN" if rate < WARN_BELOW else "OK"
    rec = dict(
        generated_at=datetime.datetime.now().isoformat(timespec="seconds"),
        status=status, agree_rate=round(rate, 4), n_races=n, n_days=len(days),
        window=[days[-1]["date"], days[0]["date"]], warn_below=WARN_BELOW, offline_ref=OFFLINE_REF,
        ai_top3=(round(sum(d["ai_top3"] for d in days) / ns, 4) if ns else None),
        fav_top3=(round(sum(d["fav_top3"] for d in days) / ns, 4) if ns else None),
        n_settled=ns, days=days)
    OUT.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"serve_health [{status}] AI1位×市場1番人気 一致率 {100*rate:.1f}% "
          f"({n}R / {len(days)}日 {rec['window'][0]}〜{rec['window'][1]}) "
          f"警告線 {100*WARN_BELOW:.0f}% / offline {100*OFFLINE_REF:.1f}%")
    if ns:
        print(f"  top3 率 (決済済み {ns}R・記述): AI {100*rec['ai_top3']:.1f}% / 市場 {100*rec['fav_top3']:.1f}%")
    for d in reversed(days):
        print(f"  {d['date']} n={d['n']:>3} 一致 {100*d['agree']/d['n']:5.1f}%")
    if status == "WARN":
        msg = (f"⚠️ **serve_health WARN** 印の一致率 {100*rate:.1f}% < {100*WARN_BELOW:.0f}% "
               f"({n}R, {rec['window'][0]}〜{rec['window'][1]})。配信側の特徴欠損を疑うこと "
               f"(feature coverage / serve_code_maps / hosei・kako5 の配置)。")
        print(msg)
        if a.notify:
            from t10_runner import notify
            notify(msg)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
