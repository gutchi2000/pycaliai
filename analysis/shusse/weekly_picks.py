# -*- coding: utf-8 -*-
"""
analysis/shusse/weekly_picks.py — 今週の注目若駒（出走表 × 出世予測 の突合）
==========================================================================
site/data/{date}.json（build_site.py が bundle から作る出走表）に載っている馬のうち、
出世予測の判定待ちセット（site/data/shusse/{debut_year}.json）に居る馬を名寄せし、
**そのレースの前の時点の予測**（hist のうち基準日 < レース日 の最新）で

  - top    : P(重賞掲示板 12m) 上位 TOP_N
  - rising : 新馬・未勝利・1 勝クラス出走で、レース前の P(重賞) が同デビュー年の上位 5% 以内

を選び site/data/shusse/weekly.json に書く。レース日がデータ末日以前（走り終えて予測に
取り込み済み）なら、着順と「走った後の評価」も付ける＝予想 → 結果 → 評価更新 の答え合わせ。
土曜朝に新しい出走表が来れば自動で「これから走る馬」のプレビューに切り替わる。

デビュー戦（レース前の予測が存在しない）の馬は対象外。
名寄せは (馬名, 父) の NFKC 一致 + 年齢とデビュー年の整合。市場オッズ・人気は出さない。
build_site.py から毎ビルド呼ばれる。

単体実行（確認用）:
    ./venv311/Scripts/python.exe -m analysis.shusse.weekly_picks
"""
from __future__ import annotations

import bisect
import json
import sys
import unicodedata
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SITE_DATA = ROOT / "site" / "data"
SHUSSE_DIR = SITE_DATA / "shusse"
OUT = SHUSSE_DIR / "weekly.json"

TOP_N = 12                # P(重賞) 上位
LOW_CLASS = {"新馬", "未勝利", "1勝"}
LOW_PCT = 95.0            # 下級条件からの浮上: 同デビュー年の上位 5% 以内（レース前の予測で）
LOW_N = 8
WEEKEND_SPAN_DAYS = 2     # 最新開催日から 2 日以内を「同じ週末」とみなす


def _nf(x) -> str:
    return unicodedata.normalize("NFKC", str(x or "")).strip()


def _load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def _weekend_dates(manifest: dict) -> list[str]:
    dates = sorted((e["date"] for e in manifest.get("dates", [])), reverse=True)
    if not dates:
        return []
    d0 = datetime.strptime(dates[0], "%Y%m%d")
    return sorted(d for d in dates if 0 <= (d0 - datetime.strptime(d, "%Y%m%d")).days <= WEEKEND_SPAN_DAYS)


def _live_index(shusse_dir: Path) -> tuple[dict[tuple[str, str], dict], dict[int, list[float]]]:
    """(馬名, 父) → 予測行、デビュー年 → 現在の pg12 の昇順リスト（百分位用）。"""
    idx: dict[tuple[str, str], dict | None] = {}
    dist: dict[int, list[float]] = {}
    for fp in sorted(shusse_dir.glob("*.json")):
        if not fp.stem.isdigit():
            continue
        for h in _load(fp).get("horses", []):
            key = (_nf(h.get("name")), _nf(h.get("sire")))
            idx[key] = None if key in idx else h     # 同名同父が 2 頭いたら曖昧なので落とす
            dist.setdefault(int(h["dy"]), []).append(float(h["pg12"]))
    for v in dist.values():
        v.sort()
    return {k: v for k, v in idx.items() if v is not None}, dist


def _pct_in(dist: list[float], x: float) -> float:
    return 100.0 * bisect.bisect_left(dist, x) / len(dist) if dist else 0.0


def _age_ok(age, debut_year: int, race_year: int) -> bool:
    """age 歳の馬は 2 歳以降にデビューしているので debut_year ∈ [race_year − age + 2, race_year]。"""
    try:
        a = int(age)
    except (TypeError, ValueError):
        return True
    return race_year - a + 2 <= debut_year <= race_year


def _pre_race(h: dict, race_date: str) -> dict | None:
    """レース日より前の基準点の最新予測。hist = [k, base, pg12, pd, lvl]。"""
    rd = race_date[:4] + "-" + race_date[4:6] + "-" + race_date[6:8]
    prev = [r for r in (h.get("hist") or []) if r[1] < rd]
    if not prev:
        return None
    k, base, pg12, pd, lvl = prev[-1]
    return {"k": int(k), "base": base, "pg12": float(pg12), "pd": float(pd), "lvl": int(lvl)}


def build(site_data: Path = SITE_DATA, shusse_dir: Path = SHUSSE_DIR, out: Path | None = OUT) -> dict:
    manifest = _load(site_data / "manifest.json")
    dates = _weekend_dates(manifest)
    live, dist = _live_index(shusse_dir)
    index_meta = _load(shusse_dir / "index.json") if (shusse_dir / "index.json").exists() else {}
    data_end = (index_meta.get("data_end") or index_meta.get("asof") or "").replace("-", "")
    cands: list[dict] = []
    n_entered = n_matched = n_pre = 0
    for d in dates:
        day = _load(site_data / f"{d}.json")
        ry = int(d[:4])
        ran = bool(data_end) and d <= data_end          # 予測データにこのレースが取り込み済み
        for r in day.get("races", []):
            res = (r.get("result") or {}).get("order") or {}
            for h in r.get("horses", []):
                n_entered += 1
                key = (_nf(h.get("name")), _nf((h.get("pedigree") or {}).get("sire")))
                lv = live.get(key)
                if not lv or not _age_ok(h.get("age"), int(lv["dy"]), ry):
                    continue
                n_matched += 1
                pre = _pre_race(lv, d)
                if pre is None:                            # デビュー戦: レース前の予測が無い
                    continue
                n_pre += 1
                pre["pct"] = round(_pct_in(dist.get(int(lv["dy"]), []), pre["pg12"]), 1)
                pos = res.get(str(h.get("umaban")))
                now = {"k": lv["k"], "pg12": lv["pg12"], "pd": lv["pd"], "lvl": lv["lvl"], "pct": lv["pct"]} if (ran and lv["base"].replace("-", "") >= d) else None
                cands.append({
                    "id": lv["id"], "name": lv["name"], "dy": lv["dy"], "sex": lv.get("sex"), "sire": lv.get("sire"),
                    "trainer": lv.get("trainer"), "summary": lv.get("summary"), "why": lv.get("why", []),
                    "pre": pre, "now": now,
                    "race": {"date": d, "place": r.get("place"), "rno": r.get("rno"), "klass": r.get("klass"),
                             "name": r.get("race_name") or "", "course": r.get("course"), "start": r.get("start_time"),
                             "field": r.get("field_size"), "umaban": h.get("umaban"), "mark": h.get("mark"),
                             "pos": (int(pos) if pos is not None else None)},
                })
    cands.sort(key=lambda c: (-c["pre"]["pg12"], c["pre"]["pd"]))
    top = cands[:TOP_N]
    chosen = {c["id"] for c in top}
    for c in top:
        c["tag"] = "top"
    low = [c for c in cands if c["id"] not in chosen and c["race"]["klass"] in LOW_CLASS and c["pre"]["pct"] >= LOW_PCT][:LOW_N]
    for c in low:
        c["tag"] = "rising"
    picks = top + low
    obj = {
        "dates": dates, "built_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "asof": index_meta.get("asof"), "data_end": index_meta.get("data_end"),
        "mode": ("review" if (dates and data_end and dates[-1] <= data_end) else "preview"),
        "n_entered": n_entered, "n_matched": n_matched, "n_with_pre": n_pre,
        "n_with_result": sum(1 for c in cands if c["race"]["pos"] is not None),
        "note": ("出走表の馬を (馬名, 父) で出世予測の判定待ちセットと突合し、レース前の基準点の予測 (pre) で選ぶ。"
                 "top = P(重賞掲示板 12m) 上位、rising = 新馬/未勝利/1勝クラス出走で同デビュー年の上位 5% 以内。"
                 "now = そのレースを走った後の評価（取り込み済みのときだけ）。市場情報は含まない。"),
        "picks": picks,
    }
    if out is not None:
        out.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return obj


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    o = build()
    print(f"mode {o['mode']} dates {o['dates']} data_end {o['data_end']} entered {o['n_entered']} matched {o['n_matched']} "
          f"with_pre {o['n_with_pre']} with_result {o['n_with_result']} picks {len(o['picks'])}")
    for c in o["picks"]:
        r, p, n = c["race"], c["pre"], c["now"]
        nowstr = "-" if not n else f"k{n['k']} {n['pg12']:.3f}"
        print(f"[{c['tag']:6s}] {c['name']:12s} dy{c['dy']} pre k{p['k']} {p['pg12']:.3f} pct {p['pct']:5.1f} lvl{p['lvl']}"
              f" | {r['date']} {r['place']}{r['rno']}R {r['klass']} {r['name']} #{r['umaban']} {r['mark'] or ''} pos={r['pos']}"
              f" | now {nowstr} | {c['summary']}")
