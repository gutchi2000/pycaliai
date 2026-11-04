# -*- coding: utf-8 -*-
"""
build_jt_pages.py — 騎手ページ / 調教師ページ用データ（人ごとの AI 評価 × 結果の履歴）
=====================================================================================
公開済み site/data/{date}.json（2026 年の全開催日）を騎手名・調教師名で束ね、
  site/data/jt/jockeys.json   {騎手名: {name, kind, stats, months, runs}}
  site/data/jt/trainers.json  {調教師名: 同上}
  site/data/jt/_index.json    検索用 [[名前, 出走数, 最終出走日, kind("j"|"t")], ...]
を書く。表示は site/jt.html + site/js/jt.js（jt.html?j=騎手名 / jt.html?t=調教師名）。

数字は公開 JSON の値だけ（印・AI 順位・勝率・人気・着順・件数・率）。オッズ・EV・払戻は読まない。
24,000 行超の出走行を 1 ファイルに収めるため、run は配列で持ち、クラスは辞書化する:
  run = [rid16, name_idx, 印, AI順, p_win, 人気, 着順, klass_idx(, 取消等)]
    - 馬名は names 表（ファイル内）の index、クラスは klass 表の index
    - 日付は rid[:8]、R 番号は rid[14:16]、競馬場コードは rid[8:10]（places 表で名前へ）
    - 着順: 1 以上 = 確定着順 / 0 = 結果はあるが着順なし（除外・中止等） / null = 結果未着
    - 末尾の 9 要素目は取消・除外の文字列（あるときだけ）
build_site.py の末尾から呼ばれる想定。単体: python build_jt_pages.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "site" / "data"
OUT = DATA / "jt"
MARKS = ("◎", "〇", "○", "▲", "△")
KINDS = (("j", "jockey"), ("t", "trainer"))


def _f(v, nd=3):
    try:
        return round(float(v), nd)
    except (TypeError, ValueError):
        return None


def _i(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _write(path: Path, obj) -> bool:
    blob = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if path.exists() and path.read_bytes() == blob:
        return False
    path.write_bytes(blob)  # LF/UTF-8 固定（write_text は Windows で CRLF になる）
    return True


def _stats(runs: list[list], klass_names: list[str]) -> dict:
    """runs は新しい順の配列行。結果の確定した行だけで率を出す。"""
    settled = [r for r in runs if r[6]]  # 着順 1 以上
    marked = [r for r in settled if r[2] in MARKS]
    hon = [r for r in settled if r[2] == "◎"]
    ai1 = [r for r in settled if r[3] == 1]
    ranks = [r[3] for r in runs if r[3]]
    pwins = [r[4] for r in runs if r[4] is not None]
    ninkis = [r[5] for r in runs if r[5]]
    wins = sum(1 for r in settled if r[6] == 1)
    top3 = sum(1 for r in settled if r[6] <= 3)
    n_set = len(settled)
    months: dict[str, list[int]] = {}
    for r in runs:
        m = r[0][:6]
        a = months.setdefault(m, [0, 0, 0, 0])  # 出走, 結果あり, 勝ち, 3着内
        a[0] += 1
        if r[6]:
            a[1] += 1
            a[2] += int(r[6] == 1)
            a[3] += int(r[6] <= 3)
    by_klass: dict[str, list[int]] = {}
    for r in settled:
        a = by_klass.setdefault(klass_names[r[7]] if r[7] is not None else "", [0, 0, 0])
        a[0] += 1; a[1] += int(r[6] == 1); a[2] += int(r[6] <= 3)
    return {
        "n": len(runs), "n_settled": n_set, "wins": wins, "top3": top3,
        "win_rate": round(wins / n_set, 4) if n_set else None,
        "top3_rate": round(top3 / n_set, 4) if n_set else None,
        "marked_n": len(marked), "marked_win": sum(1 for r in marked if r[6] == 1),
        "marked_top3": sum(1 for r in marked if r[6] <= 3),
        "hon_n": len(hon), "hon_win": sum(1 for r in hon if r[6] == 1),
        "hon_top3": sum(1 for r in hon if r[6] <= 3),
        "ai1_n": len(ai1), "ai1_win": sum(1 for r in ai1 if r[6] == 1),
        "ai1_top3": sum(1 for r in ai1 if r[6] <= 3),
        "mean_ai_rank": round(sum(ranks) / len(ranks), 2) if ranks else None,
        "mean_pwin": round(sum(pwins) / len(pwins), 4) if pwins else None,
        "mean_ninki": round(sum(ninkis) / len(ninkis), 2) if ninkis else None,
        # AI が 1 位評価した出走が全体のどれだけか（"AI の期待度"）
        "ai1_share": round(len(ai1) / n_set, 4) if n_set else None,
        "months": dict(sorted(months.items())),
        "by_klass": dict(sorted(by_klass.items(), key=lambda kv: -kv[1][0])),
    }


def build() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    people: dict[str, dict[str, dict]] = {"j": {}, "t": {}}
    klass_idx: dict[str, int] = {}
    klass_names: list[str] = []
    name_idx: dict[str, int] = {}
    names: list[str] = []
    places: dict[str, str] = {}
    days = sorted(DATA.glob("2026????.json"))
    for f in days:
        try:
            day = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        for r in day.get("races") or []:
            rid = str(r.get("race_id") or "")
            if len(rid) != 16:
                continue
            place = r.get("place") or ""
            if place:
                places.setdefault(rid[8:10], place)
            kl = r.get("klass") or ""
            if kl not in klass_idx:
                klass_idx[kl] = len(klass_names); klass_names.append(kl)
            order = (r.get("result") or {}).get("order") or {}
            has_res = bool(order)
            for h in r.get("horses") or []:
                nm = h.get("name")
                if not nm:
                    continue
                pos = _i(order.get(str(h.get("umaban"))))
                pos_enc = pos if pos else (0 if has_res else None)
                if nm not in name_idx:
                    name_idx[nm] = len(names); names.append(nm)
                run = [rid, name_idx[nm], h.get("mark") or "", _i(h.get("ai_rank")), _f(h.get("p_win")),
                       _i(h.get("ninki")), pos_enc, klass_idx[kl]]
                scr = h.get("scratched")
                if scr:
                    run.append(str(scr))
                for kind, field in KINDS:
                    who = h.get(field)
                    if not who:
                        continue
                    e = people[kind].setdefault(who, {"name": who, "kind": kind, "runs": []})
                    e["runs"].append(run)
    index: list[list] = []
    n_written = 0
    sizes: dict[str, int] = {}
    for kind, fname in (("j", "jockeys.json"), ("t", "trainers.json")):
        book = people[kind]
        for nm, e in book.items():
            e["runs"].sort(key=lambda x: x[0], reverse=True)
            e["stats"] = _stats(e["runs"], klass_names)
            e["updated"] = e["runs"][0][0][:8]
            index.append([nm, len(e["runs"]), e["updated"], kind])
        obj = {"kind": kind, "klass": klass_names, "places": dict(sorted(places.items())), "names": names,
               "people": dict(sorted(book.items()))}
        p = OUT / fname
        n_written += int(_write(p, obj))
        sizes[fname] = p.stat().st_size
    index.sort(key=lambda x: (-x[1], x[0]))
    pi = OUT / "_index.json"
    n_written += int(_write(pi, index))
    sizes["_index.json"] = pi.stat().st_size
    return {"jockeys": len(people["j"]), "trainers": len(people["t"]), "runs": sum(len(e["runs"]) for e in people["j"].values()),
            "days": len(days), "written": n_written, "sizes": sizes}


def main() -> int:
    st = build()
    sz = ", ".join(f"{k} {v / 1024:.0f}KB" for k, v in st["sizes"].items())
    print(f"jt pages: 騎手 {st['jockeys']} 人 / 調教師 {st['trainers']} 人 ({st['days']} 日, {st['runs']} 出走行), 更新 {st['written']} ファイル [{sz}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
