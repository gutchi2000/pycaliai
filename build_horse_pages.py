# -*- coding: utf-8 -*-
"""
build_horse_pages.py — 馬ページ用データ（馬ごとの AI 評価履歴）
================================================================
公開済み site/data/{date}.json（2026 年の全開催日）を馬名で束ね、
  site/data/h/{id[:2]}.json  256 シャード {id: 馬} （2026 年の出走と AI の印・勝率・結果、近走、出世予測）
  site/data/h/_index.json    検索用 [[馬名, 出走数, 最終出走日], ...]（id は馬名から計算）
を書く。馬ごとに 1 ファイルだと 9,000+ ファイルになり Workers assets の上限 (20,000) と
週次の差分が重いので、id 先頭 2 桁で 256 ファイルに束ねる（1 ファイル 30〜60KB）。表示は site/horse.html + site/js/horse.js（horse.html?name=馬名）。

id は馬名の FNV-1a 32bit ×2（seed 違い）の 16 桁 hex。JS 側 (horse.js) と同じ定義で、
直リンクは index を読まずに id を計算できる。数字は公開 JSON の値だけ（オッズ・EV なし）。
build_site.py の末尾から呼ばれる。単体: python build_horse_pages.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "site" / "data"
OUT = DATA / "h"
MARKS = ("◎", "〇", "○", "▲", "△")


def fnv1a32(s: str, seed: int) -> int:
    h = seed
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def horse_id(name: str) -> str:
    return f"{fnv1a32(name, 0x811C9DC5):08x}{fnv1a32(name, 0x01000193):08x}"


def _f(v, nd=4):
    try:
        return round(float(v), nd)
    except (TypeError, ValueError):
        return None


def load_shusse() -> dict[str, dict]:
    """馬名 → 最新年の出世予測（shusse.html#{dy}:{id} へのリンク用）。"""
    out: dict[str, dict] = {}
    for f in sorted((DATA / "shusse").glob("20??.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        for h in d.get("horses") or []:
            nm = h.get("name")
            if not nm:
                continue
            out[nm] = {"id": h.get("id"), "dy": h.get("dy"), "pct": h.get("pct"), "lvl": h.get("lvl"),
                       "rank": h.get("rank"), "pg12": h.get("pg12"), "summary": h.get("summary"),
                       "n": h.get("n"), "base": h.get("base")}
    return out


def build() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    shusse = load_shusse()
    horses: dict[str, dict] = {}
    days = sorted(DATA.glob("2026????.json"))
    for f in days:
        date = f.name[:8]
        day = json.loads(f.read_text(encoding="utf-8"))
        for r in day.get("races") or []:
            order = (r.get("result") or {}).get("order") or {}
            j = r.get("judgment") or {}
            for h in r.get("horses") or []:
                nm = h.get("name")
                if not nm:
                    continue
                pos = order.get(str(h.get("umaban")))
                e = horses.setdefault(nm, {"name": nm, "id": horse_id(nm), "runs": []})
                e["runs"].append({
                    "date": date, "rid": r["race_id"], "place": r.get("place"), "rno": r.get("rno"),
                    "course": r.get("course"), "klass": r.get("klass"), "race_name": r.get("race_name") or "",
                    "mark": h.get("mark") or "", "ai_rank": h.get("ai_rank"),
                    "p_win": _f(h.get("p_win")), "p_sho": _f(h.get("p_sho")),
                    "ninki": h.get("ninki"), "jockey": h.get("jockey"), "kinryo": h.get("kinryo"),
                    "umaban": h.get("umaban"), "field_size": r.get("field_size"),
                    "pos": int(pos) if pos else None, "has_result": bool(order),
                    "umami": (h.get("umami") or {}).get("grade"), "chaos_pct": j.get("chaos_pct"),
                    "scratched": h.get("scratched") or None,
                })
                # 属性は最新出走のものを採用（age は出走時点）
                e["sex"] = h.get("sex"); e["age"] = h.get("age")
                e["trainer"] = h.get("trainer"); e["shozoku"] = h.get("shozoku")
                e["pedigree"] = h.get("pedigree"); e["level"] = h.get("level"); e["style"] = h.get("style")
                e["past"] = (h.get("history") or {}).get("runs") or []
                e["updated"] = date
    index = []
    n_written = 0
    shards: dict[str, dict] = {}
    for nm, e in horses.items():
        e["runs"].sort(key=lambda x: (x["date"], x["rid"]), reverse=True)
        runs = [x for x in e["runs"] if x["has_result"] and x["pos"]]
        marked = [x for x in runs if x["mark"] in MARKS]
        hon = [x for x in runs if x["mark"] == "◎"]
        e["stats"] = {
            "n": len(e["runs"]), "n_settled": len(runs),
            "wins": sum(1 for x in runs if x["pos"] == 1),
            "top3": sum(1 for x in runs if x["pos"] <= 3),
            "marked_n": len(marked), "marked_top3": sum(1 for x in marked if x["pos"] <= 3),
            "hon_n": len(hon), "hon_win": sum(1 for x in hon if x["pos"] == 1),
            "hon_top3": sum(1 for x in hon if x["pos"] <= 3),
            "avg_pwin": _f(sum(x["p_win"] or 0 for x in runs) / len(runs), 4) if runs else None,
            "best_ai_rank": min((x["ai_rank"] for x in e["runs"] if x["ai_rank"]), default=None),
        }
        e["shusse"] = shusse.get(nm)
        shards.setdefault(e["id"][:2], {})[e["id"]] = e
        index.append([nm, len(e["runs"]), e["updated"]])
    for key, sh in shards.items():
        blob = json.dumps(dict(sorted(sh.items())), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        p = OUT / f"{key}.json"
        if not p.exists() or p.read_bytes() != blob:
            p.write_bytes(blob)
            n_written += 1
    index.sort(key=lambda x: (-int(x[2]), x[0]))
    blob = json.dumps(index, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    pi = OUT / "_index.json"
    if not pi.exists() or pi.read_bytes() != blob:
        pi.write_bytes(blob)
    # 消えた馬（データ修正でレース自体が消えた場合）のファイルを掃除
    keep = {f"{k}.json" for k in shards} | {"_index.json"}
    n_removed = 0
    for f in OUT.glob("*.json"):
        if f.name not in keep:
            f.unlink(); n_removed += 1
    return {"horses": len(horses), "written": n_written, "removed": n_removed, "days": len(days),
            "shards": len(shards)}


def main() -> int:
    st = build()
    print(f"horse pages: {st['horses']} 頭 ({st['days']} 日) / {st['shards']} シャード, 更新 {st['written']}, 削除 {st['removed']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
