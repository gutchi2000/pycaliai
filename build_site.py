# -*- coding: utf-8 -*-
"""
build_site.py — 静的サイト (site/) 用の表示データを生成する。

入力:
  reports/cowork_input/{date}_bundle.json   export_weekly_marks.py の出力
  reports/cowork_output/*.json|txt|md       Cowork の買い目/論評 (任意)

出力:
  site/data/{date}.json     日別 view-model (枠番・単EV・Cowork マージ済み)
  site/data/manifest.json   日付インデックス (新しい順)

使い方:
  python build_site.py             # 全 bundle を変換
  python build_site.py 20260613    # 指定日だけ再変換 (manifest は全日分で更新)
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BUNDLE_DIR = ROOT / "reports" / "cowork_input"
COWORK_OUT_DIR = ROOT / "reports" / "cowork_output"
SITE_DATA_DIR = ROOT / "site" / "data"

PLACE_ORDER = ["札幌", "函館", "福島", "新潟", "東京", "中山",
               "中京", "京都", "阪神", "小倉"]


# ---------------------------------------------------------------- 枠番
def waku_of(umaban: int, field_size: int) -> int:
    """JRA の枠順割当 (多頭数は後ろの枠から 2〜3 頭詰め) で馬番→枠番を求める。"""
    n = field_size
    if n <= 8:
        return umaban
    if n <= 16:
        counts = [2 if b >= 17 - n else 1 for b in range(1, 9)]
    else:  # 17, 18 頭
        counts = [2] * 8
        for b in range(8, 8 - (n - 16), -1):
            counts[b - 1] = 3
    cum = 0
    for b, k in enumerate(counts, start=1):
        cum += k
        if umaban <= cum:
            return b
    return 8


# ---------------------------------------------------------------- Cowork 出力
def _parse_one_cowork_file(path: Path) -> dict[str, dict]:
    """1 ファイル → race_id 別 dict。nicegui_app._parse_one_cowork_file と同等の寛容パース。"""
    try:
        raw = path.read_bytes()
        text = ""
        for enc in ("utf-8-sig", "utf-8", "cp932", "shift_jis"):
            try:
                text = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            text = raw.decode("utf-8", errors="replace")
    except Exception:
        return {}

    m = re.search(r"```(?:json|JSON)?\s*\n([\s\S]+?)\n\s*```", text)
    raw_json = m.group(1) if m else text.strip()
    try:
        data = json.loads(raw_json)
    except Exception:
        return {}

    if isinstance(data, dict):
        data = data.get("bets") or data.get("races", [data])
    if not isinstance(data, list):
        return {}

    out: dict[str, dict] = {}
    for entry in data:
        if not isinstance(entry, dict):
            continue
        rid_raw = entry.get("race_id") or entry.get("レースID") or entry.get("rid")
        if not rid_raw:
            continue
        rid = str(rid_raw)[:16]
        bets = []
        for b in entry.get("bets", entry.get("買い目", [])) or []:
            if not isinstance(b, dict):
                continue
            bets.append({
                "type": b.get("馬券種") or b.get("type", ""),
                "selection": b.get("買い目") or b.get("selection", ""),
                "amount": b.get("購入額") or b.get("amount", 0),
                "reason": b.get("理由") or b.get("reason", ""),
            })
        advisor = []
        for a in entry.get("advisor", []) or []:
            if not isinstance(a, dict):
                continue
            advisor.append({
                "umaban": a.get("umaban") or a.get("馬番"),
                "horse_name": str(a.get("horse_name") or a.get("馬名", "")),
                "grade": str(a.get("grade") or ""),
                "tag": a.get("tag") or None,
                "comment": str(a.get("comment") or a.get("コメント", "")),
            })
        out[rid] = {
            "race_label": str(entry.get("race_label", "")),
            "race_nature": str(entry.get("race_nature", "")),
            "race_reason": str(entry.get("race_reason", "")),
            "bets": bets,
            "advisor": advisor,
            "source": path.name,
        }
    return out


def load_all_cowork() -> dict[str, dict]:
    """cowork_output/ 直下の全ファイルを mtime 昇順で読み、race_id 別にマージ (後勝ち)。"""
    if not COWORK_OUT_DIR.exists():
        return {}
    files = sorted(
        (p for p in COWORK_OUT_DIR.iterdir()
         if p.is_file() and p.suffix.lower() in (".json", ".txt", ".md")),
        key=lambda p: p.stat().st_mtime,
    )
    out: dict[str, dict] = {}
    for p in files:
        out.update(_parse_one_cowork_file(p))
    return out


# ---------------------------------------------------------------- bundle 変換
def transform_bundle(path: Path, cowork: dict[str, dict]) -> dict:
    with open(path, encoding="utf-8") as f:
        bundle = json.load(f)

    date_str = path.name[:8]
    races_out = []
    for race in bundle.get("races", []):
        meta = race.get("race_meta", {})
        rid = str(race.get("race_id", ""))
        field_size = meta.get("field_size") or len(race.get("horses", []))

        horses = []
        for h in race.get("horses", []):
            umaban = h.get("umaban")
            p_win = h.get("p_win")
            odds = h.get("tansho_odds")
            ev_tan = round(p_win * odds, 2) if (p_win and odds) else None
            horses.append({
                "umaban": umaban,
                "waku": waku_of(umaban, field_size) if umaban else None,
                "name": h.get("horse_name", ""),
                "mark": h.get("mark") or "",
                "ai_rank": h.get("ai_rank"),
                "ai_score": h.get("ai_score"),
                "p_win": p_win,
                "p_plc": h.get("p_plc"),
                "p_sho": h.get("p_sho"),
                "odds": odds,
                "fuku_low": h.get("fuku_odds_low"),
                "fuku_high": h.get("fuku_odds_high"),
                "vs_market": h.get("ai_vs_market"),
                "ev_tan": ev_tan,
                "sex": h.get("sex", ""),
                "age": h.get("age"),
                "why": h.get("why", []),
                "history": h.get("history"),
                "pedigree": h.get("pedigree"),
            })

        races_out.append({
            "race_id": rid,
            "rno": int(rid[-2:]) if rid[-2:].isdigit() else None,
            "place": meta.get("place", ""),
            "course": meta.get("course", ""),
            "klass": meta.get("class", ""),
            "race_name": meta.get("race_name", ""),
            "field_size": field_size,
            "class_prior": meta.get("class_prior"),
            "confidence": race.get("race_confidence", {}),
            "judgment": race.get("buy_judgment", {}),
            "horses": horses,
            "cowork": cowork.get(rid),
        })

    places_seen = {r["place"] for r in races_out}
    places = [p for p in PLACE_ORDER if p in places_seen]
    places += sorted(places_seen - set(places))  # 想定外の場名も末尾に

    return {
        "date": date_str,
        "places": places,
        "races": races_out,
    }


# ---------------------------------------------------------------- main
def main() -> None:
    only_date = sys.argv[1] if len(sys.argv) > 1 else None
    SITE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    bundles = sorted(BUNDLE_DIR.glob("*_bundle.json"))
    if not bundles:
        print(f"bundle が見つかりません: {BUNDLE_DIR}")
        sys.exit(1)

    cowork = load_all_cowork()
    print(f"cowork_output: {len(cowork)} races 読込")

    manifest_entries = []
    for path in bundles:
        date_str = path.name[:8]
        out_path = SITE_DATA_DIR / f"{date_str}.json"
        if only_date is None or date_str == only_date:
            day = transform_bundle(path, cowork)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(day, f, ensure_ascii=False, separators=(",", ":"))
            n_cowork = sum(1 for r in day["races"] if r["cowork"])
            print(f"  {date_str}: {len(day['races'])} races "
                  f"(cowork {n_cowork}) -> {out_path.relative_to(ROOT)}")
            places = day["places"]
            n_races = len(day["races"])
        elif out_path.exists():
            with open(out_path, encoding="utf-8") as f:
                prev = json.load(f)
            places = prev.get("places", [])
            n_races = len(prev.get("races", []))
        else:
            continue
        manifest_entries.append(
            {"date": date_str, "places": places, "n_races": n_races})

    manifest_entries.sort(key=lambda e: e["date"], reverse=True)
    manifest = {
        "built_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "model": "v6",
        "dates": manifest_entries,
    }
    with open(SITE_DATA_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"manifest: {len(manifest_entries)} dates")


if __name__ == "__main__":
    main()
