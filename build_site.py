# -*- coding: utf-8 -*-
"""
build_site.py — 静的サイト (site/) 用の表示データを生成する。

入力:
  reports/cowork_input/{date}_bundle.json   export_weekly_marks.py の出力
  reports/cowork_output/*.json|txt|md       Cowork の買い目/論評 (任意)
  data/weekly/{date}.csv                    TARGET 出走表 (騎手/斤量/人気/ZI/発走時刻)
  data/kekka/{date}.csv                     TARGET 結果 (着順・払戻、終了日のみ)
  data/kekka/wide_kekka.csv                 ワイド払戻 (任意)

出力:
  site/data/{date}.json     日別 view-model
  site/data/manifest.json   日付インデックス (新しい順)

使い方:
  python build_site.py             # 全 bundle を変換
  python build_site.py 20260613    # 指定日だけ再変換 (manifest は全日分で更新)
"""
from __future__ import annotations

import csv
import io
import json
import re
import sys
import unicodedata
from datetime import datetime
from pathlib import Path

try:
    from umami import umami_total, explain as umami_explain
except Exception:  # umami / audit json 不在でもサイト生成は継続
    umami_total = None
    umami_explain = None

ROOT = Path(__file__).resolve().parent
BUNDLE_DIR = ROOT / "reports" / "cowork_input"
COWORK_OUT_DIR = ROOT / "reports" / "cowork_output"
WEEKLY_DIR = ROOT / "data" / "weekly"
KEKKA_DIR = ROOT / "data" / "kekka"
TRAINING_DIR = ROOT / "data" / "training"
COURSE_STATS_PATH = ROOT / "data" / "course_stats.json"
PEDIGREE_STATS_PATH = ROOT / "data" / "pedigree_stats.json"
SITE_DATA_DIR = ROOT / "site" / "data"

PLACE_ORDER = ["札幌", "函館", "福島", "新潟", "東京", "中山",
               "中京", "京都", "阪神", "小倉"]

# ---- TARGET weekly CSV 列定義 (nicegui_app.py と同一) ----
RACE_COLS = [
    "レースID(新)", "日付S", "曜日", "場所", "開催", "R", "レース名", "クラス名",
    "芝・ダート", "距離", "コース区分", "コーナー回数", "馬場状態(暫定)", "天候(暫定)",
    "フルゲート頭数", "発走時刻", "性別限定", "重量種別", "年齢限定",
]
# 46/48 列形式: 先頭 17 列のインデックス
IDX_46 = {"枠番": 0, "B": 1, "馬番": 2, "人気": 6, "単勝": 7, "ZI": 9, "ZI順": 10,
          "斤量": 11, "替": 13, "騎手": 14, "所属": 15, "調教師": 16}
# 49/99 列形式 (馬体重 3 列が挿入される)
IDX_49 = {"枠番": 0, "B": 1, "馬番": 2, "馬体重": 6, "増減": 8, "人気": 9, "単勝": 10,
          "ZI": 12, "ZI順": 13, "斤量": 14, "替": 16, "騎手": 17, "所属": 18, "調教師": 19}

STYLE_MAP = {"逃げ": "逃げ", "先行": "先行", "中団": "差し", "差し": "差し",
             "ﾏｸﾘ": "差し", "マクリ": "差し", "後方": "追込", "追込": "追込"}
STYLE_ORDER = ["逃げ", "先行", "差し", "追込"]


def _int(s):
    try:
        return int(str(s).strip())
    except (ValueError, TypeError):
        return None


def _float(s):
    try:
        return float(str(s).strip())
    except (ValueError, TypeError):
        return None


def _decode(raw: bytes) -> str:
    for enc in ("cp932", "utf-8-sig", "utf-8"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def deep_zen(o):
    """半角カナ→全角 / 全角英数→半角 (NFKC)。TARGET 出力の ｵｰﾌﾟﾝ 等の潰れ対策。"""
    if isinstance(o, str):
        return unicodedata.normalize("NFKC", o)
    if isinstance(o, list):
        return [deep_zen(x) for x in o]
    if isinstance(o, dict):
        return {k: deep_zen(v) for k, v in o.items()}
    return o


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


# ---------------------------------------------------------------- weekly CSV
def parse_weekly(date_str: str) -> tuple[dict, dict]:
    """data/weekly/{date}.csv → (race_extra, horse_extra)。

    race_extra:  rid16 → {race_name, start_time, baba, weather, fullgate}
    horse_extra: (rid16, umaban) → {waku, jockey, kinryo, ninki, odds_pre,
                                    zi, zi_rank, kawari, blinker, taiju, taiju_diff,
                                    trainer, shozoku}
    """
    p = WEEKLY_DIR / f"{date_str}.csv"
    if not p.exists():
        p = WEEKLY_DIR / f"{date_str}.CSV"
    if not p.exists():
        return {}, {}

    race_extra: dict[str, dict] = {}
    horse_extra: dict[tuple, dict] = {}
    rid = None
    for line in _decode(p.read_bytes()).splitlines():
        cols = line.split(",")
        if not cols or cols[0] in ("レースID(新)", "枠番", "番", ""):
            continue
        if len(cols) == 19:  # race 行
            rid = str(cols[0]).strip()[:16]
            race_extra[rid] = {
                "race_name": cols[6].strip(),
                "start_time": cols[15].strip(),
                "baba": cols[12].strip(),
                "weather": cols[13].strip(),
                "fullgate": _int(cols[14]),
            }
        elif len(cols) in (46, 48, 49, 99) and rid:
            idx = IDX_46 if len(cols) in (46, 48) else IDX_49
            uma = _int(cols[idx["馬番"]])
            if uma is None:
                continue
            g = lambda k: cols[idx[k]].strip() if k in idx else ""
            horse_extra[(rid, uma)] = {
                "waku": _int(g("枠番")),
                "jockey": g("騎手"),
                "kinryo": _float(g("斤量")),
                "ninki": _int(g("人気")),
                "odds_pre": _float(g("単勝")),
                "zi": _float(g("ZI")),
                "zi_rank": _int(g("ZI順")),
                "kawari": g("替"),
                "blinker": g("B"),
                "taiju": _int(g("馬体重")),
                "taiju_diff": g("増減"),
                "trainer": g("調教師"),
                "shozoku": g("所属"),
            }
    return race_extra, horse_extra


# ---------------------------------------------------------------- 脚質
def classify_style(history: dict | None) -> str | None:
    """近 5 走の脚質ラベル (直近ほど重み大) から 逃げ/先行/差し/追込 を推定。"""
    runs = (history or {}).get("runs") or []
    score: dict[str, float] = {}
    for u in runs:
        st = STYLE_MAP.get(str(u.get("weight_change", "")).strip())
        if not st:
            continue
        w = max(1, 6 - (u.get("n_ago") or 5))
        score[st] = score.get(st, 0) + w
    if not score:
        return None
    best = max(score.values())
    for st in STYLE_ORDER:
        if score.get(st) == best:
            return st
    return None


# ---------------------------------------------------------------- ペア (馬連/ワイド)
def pairs_top(race: dict, top_n: int = 8) -> list[dict]:
    pp = race.get("pair_probs") or {}
    matrix = race.get("umaren_matrix") or {}
    out = []
    for key, v in pp.items():
        try:
            a, b = (int(x) for x in key.split("-"))
        except ValueError:
            continue
        out.append({
            "a": a, "b": b,
            "p_umaren": (v or {}).get("umaren"),
            "p_wide": (v or {}).get("wide"),
            # umaren_matrix = data/odds/OD*.CSV 由来の「実際の馬連オッズ」(市場配当)。
            # 旧名 "fair" は誤り (適正値ではなく実オッズ) だったので umaren_odds に改名。
            "umaren_odds": matrix.get(key),
        })
    out.sort(key=lambda x: -(x["p_umaren"] or 0))
    return out[:top_n]


# ---------------------------------------------------------------- 結果 (kekka)
def parse_wide_kekka() -> dict[tuple, dict]:
    p = KEKKA_DIR / "wide_kekka.csv"
    if not p.exists():
        return {}
    pair_pat = re.compile(r"(\d+)\s*[-―]\s*(\d+)\s*[\\¥￥]\s*(\d+)")
    out: dict = {}
    for row in csv.reader(io.StringIO(_decode(p.read_bytes()))):
        if len(row) < 10:
            continue
        y, m, d, rn = _int(row[0]), _int(row[1]), _int(row[2]), _int(row[4])
        if None in (y, m, d, rn):
            continue
        pairs = {}
        for chunk in (row[9] or "").split("/"):
            mt = pair_pat.search(chunk)
            if mt:
                a, b, pay = int(mt.group(1)), int(mt.group(2)), int(mt.group(3))
                pairs[f"{min(a, b)}-{max(a, b)}"] = pay
        if pairs:
            out[(y, m, d, str(row[3]).strip(), rn)] = pairs
    return out


def parse_kekka(date_str: str, wide_data: dict) -> dict[str, dict]:
    """data/kekka/{date}.csv → rid16 → {order, top3, pays}"""
    p = KEKKA_DIR / f"{date_str}.csv"
    if not p.exists():
        return {}
    races: dict[str, dict] = {}
    reader = csv.reader(io.StringIO(_decode(p.read_bytes())))
    next(reader, None)  # header
    for row in reader:
        if len(row) < 15:
            continue
        rid16 = str(row[7]).strip()[:16]
        if len(rid16) < 16:
            continue
        uma, pos = _int(row[4]), _int(row[6])
        if uma is None or pos is None:
            continue
        r = races.setdefault(rid16, {
            "place": str(row[1]).strip(), "rno": _int(row[2]),
            "order": {}, "fuku": {},
            "pays": {"wakuren": None, "umaren": None, "umatan": None,
                     "sanrenpuku": None, "sanrentan": None},
        })
        r["order"][str(uma)] = pos
        tan_raw = str(row[8]).strip()
        if pos == 1 and tan_raw and not tan_raw.startswith("("):
            r["pays"]["tan"] = _int(tan_raw)
        if pos <= 3 and _int(row[9]) is not None:
            r["fuku"][str(uma)] = _int(row[9])
        for key, i in [("wakuren", 10), ("umaren", 11), ("umatan", 12),
                       ("sanrenpuku", 13), ("sanrentan", 14)]:
            v = _int(row[i]) if i < len(row) else None
            if v is not None and r["pays"].get(key) is None:
                r["pays"][key] = v

    for rid16, r in races.items():
        top = sorted(((int(u), p) for u, p in r["order"].items()),
                     key=lambda t: t[1])
        r["top3"] = [u for u, p in top if p <= 3]
        wkey = (int(rid16[:4]), int(rid16[4:6]), int(rid16[6:8]),
                r["place"], r["rno"] or 0)
        r["pays"]["wide"] = wide_data.get(wkey) or {}
        r["pays"]["fuku"] = r.pop("fuku")
        del r["place"], r["rno"]
    return races


# ---------------------------------------------------------------- Cowork 出力
def _parse_one_cowork_file(path: Path) -> dict[str, dict]:
    """1 ファイル → race_id 別 dict。nicegui_app._parse_one_cowork_file と同等の寛容パース。"""
    try:
        text = _decode(path.read_bytes())
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


def _parse_one_grade_scope(path: Path) -> dict[str, dict]:
    """1 ファイルの top-level 'grade_scope' (重賞 LLM 詳細見解) を race_id 別に。"""
    try:
        text = _decode(path.read_bytes())
    except Exception:
        return {}
    m = re.search(r"```(?:json|JSON)?\s*\n([\s\S]+?)\n\s*```", text)
    raw = m.group(1) if m else text.strip()
    try:
        data = json.loads(raw)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, dict] = {}
    for g in data.get("grade_scope") or []:
        if not isinstance(g, dict):
            continue
        rid = str(g.get("race_id") or "")[:16]
        if not rid or not g.get("markdown"):
            continue
        out[rid] = {
            "klass": str(g.get("class", "")),
            "race_label": str(g.get("race_label", "")),
            "markdown": str(g.get("markdown", "")),
            "source": path.name,
        }
    return out


def load_all_grade_scope() -> dict[str, dict]:
    if not COWORK_OUT_DIR.exists():
        return {}
    files = sorted(
        (p for p in COWORK_OUT_DIR.iterdir()
         if p.is_file() and p.suffix.lower() in (".json", ".txt", ".md")),
        key=lambda p: p.stat().st_mtime,
    )
    out: dict[str, dict] = {}
    for p in files:
        out.update(_parse_one_grade_scope(p))
    return out


def _nfkc(s) -> str:
    return unicodedata.normalize("NFKC", str(s or "")).strip()


# ---------------------------------------------------------------- コース成績
def load_course_stats() -> dict:
    if not COURSE_STATS_PATH.exists():
        return {}
    with open(COURSE_STATS_PATH, encoding="utf-8") as f:
        return json.load(f)


def _slim_course(entry: dict) -> dict:
    """course_stats の 1 コース分を表示に必要な列だけに圧縮。"""
    out = {k: entry.get(k) for k in ("place", "surface", "distance",
                                     "n_races", "n_starts")}
    for grp in ("waku", "uma", "kyaku", "age", "sex"):
        out[grp] = [{
            "label": r.get("label"),
            "win": r.get("win_rate"),
            "ren": r.get("rentai_rate"),
            "fuku": r.get("fuku_rate"),
            "n": r.get("n_total"),
        } for r in (entry.get(grp) or [])]
    return out


# ---------------------------------------------------------------- 血統
TD_RE = re.compile(r"([芝ダ])\D*(\d+)")


def parse_course_str(course: str):
    """'芝1200' -> ('芝', 1200) / 'ダ1800' -> ('ダ', 1800)。"""
    m = TD_RE.search(course or "")
    return (m.group(1), int(m.group(2))) if m else (None, None)


def load_pedigree_index():
    """(place, td) -> [course entry,...] の索引。"""
    if not PEDIGREE_STATS_PATH.exists():
        return None
    with open(PEDIGREE_STATS_PATH, encoding="utf-8") as f:
        ps = json.load(f)
    idx: dict = {}
    for e in (ps.get("courses") or {}).values():
        idx.setdefault((e.get("place"), e.get("td")), []).append(e)
    return idx


def ped_course_entry(ped_index, place: str, course: str):
    """レースの place / 距離に最も近い血統コース統計を返す。"""
    if not ped_index:
        return None
    td, dist = parse_course_str(course)
    if td is None:
        return None
    best, best_d = None, 1e9
    for e in ped_index.get((place, td)) or []:
        band = e.get("dist_band") or [0, 0]
        if band[0] <= dist <= band[1]:
            d = abs((e.get("dist_bucket") or dist) - dist)
            if d < best_d:
                best, best_d = e, d
    return best


# ---------------------------------------------------------------- 調教
TRN_RE = re.compile(r"[HW]-(\d{8})-(\d{8})\.csv$", re.I)


def pick_training_file(prefix: str, date_str: str):
    """race date 以前で最も新しい週次調教ファイル (H=坂路 / W=WC) を選ぶ。"""
    if not TRAINING_DIR.exists():
        return None
    best, best_end = None, ""
    for p in TRAINING_DIR.glob(f"{prefix}-*.csv"):
        m = TRN_RE.search(p.name)
        if not m:
            continue
        start, end = m.group(1), m.group(2)
        if start <= date_str and end > best_end:
            best, best_end = p, end
    return best


def parse_training_file(path, kind: str) -> dict:
    """name(NFKC) -> 最新追い切り row。kind: 'hanro'(坂路) / 'wc'(ウッド)。"""
    if not path or not path.exists():
        return {}
    rows: dict = {}
    reader = csv.reader(io.StringIO(_decode(path.read_bytes())))
    next(reader, None)  # header
    for row in reader:
        if len(row) < 4:
            continue
        date = row[0].strip()
        name = _nfkc(row[1])
        if not name:
            continue
        if kind == "hanro":
            # 年月日,馬名,Time1(4F),Time2(3F),Time3(2F),Time4(1F),Lap4,Lap3,Lap2,Lap1
            laps = [_float(row[i]) for i in (6, 7, 8, 9) if i < len(row)]
            rec = {"date": date, "t4f": _float(row[2]) if len(row) > 2 else None,
                   "t1f": _float(row[5]) if len(row) > 5 else None,
                   "laps": laps, "lap1": laps[-1] if laps else None}
        else:
            # 年月日,馬名,5F,4F,3F,Lap3,Lap2,Lap1
            laps = [_float(row[i]) for i in (5, 6, 7) if i < len(row)]
            rec = {"date": date, "f5": _float(row[2]) if len(row) > 2 else None,
                   "f4": _float(row[3]) if len(row) > 3 else None,
                   "f3": _float(row[4]) if len(row) > 4 else None,
                   "laps": laps, "lap1": laps[-1] if laps else None}
        prev = rows.get(name)
        if prev is None or date >= prev["date"]:
            rows[name] = rec
    return rows


# ---------------------------------------------------------------- bundle 変換
def transform_bundle(path: Path, cowork: dict, wide_data: dict,
                     course_stats: dict, ped_index, grade_map: dict) -> dict:
    with open(path, encoding="utf-8") as f:
        bundle = json.load(f)

    date_str = path.name[:8]
    race_extra, horse_extra = parse_weekly(date_str)
    results = parse_kekka(date_str, wide_data)
    hanro_map = parse_training_file(pick_training_file("H", date_str), "hanro")
    wc_map = parse_training_file(pick_training_file("W", date_str), "wc")

    races_out = []
    for race in bundle.get("races", []):
        meta = race.get("race_meta", {})
        rid = str(race.get("race_id", ""))
        field_size = meta.get("field_size") or len(race.get("horses", []))
        rext = race_extra.get(rid, {})

        # このレースのコース血統統計 (種牡馬/母父 → fuku/rank)
        pced = ped_course_entry(ped_index, meta.get("place", ""),
                                meta.get("course", ""))
        sire_map = {_nfkc(s.get("name")): s for s in (pced.get("sire") or [])} \
            if pced else {}
        bms_map = {_nfkc(s.get("name")): s for s in (pced.get("broodmare_sire") or [])} \
            if pced else {}
        base_fuku = pced.get("baseline_fuku_rate") if pced else None

        horses = []
        for h in race.get("horses", []):
            umaban = h.get("umaban")
            hext = horse_extra.get((rid, umaban), {})
            p_win = h.get("p_win")
            odds = h.get("tansho_odds")
            ev_tan = round(p_win * odds, 2) if (p_win and odds) else None

            # 調教 (坂路 / WC) を馬名で結合
            nm = _nfkc(h.get("horse_name"))
            hanro, wc = hanro_map.get(nm), wc_map.get(nm)
            training = {"hanro": hanro, "wc": wc} if (hanro or wc) else None

            # UMAMI (実測補正後の期待回収率 xROI) を全頭ぶん算出
            umami_obj = None
            if umami_total:
                ut = umami_total(h)
                umj = ut["um"]
                umami_obj = {
                    "xroi": ut["xroi"], "side": ut["side"], "grade": ut["grade"],
                    "ev_tan": umj["tan"]["ev"], "ev_fuku": umj["fuku"]["ev"],
                    "tan_xroi": umj["tan"]["xroi"], "fuku_xroi": umj["fuku"]["xroi"],
                    "reason": umami_explain(h, ut) if umami_explain else "",
                }

            # 血統スタッツ (このコースでの父/母父 fuku/rank)
            ped = h.get("pedigree") or {}
            srow = sire_map.get(_nfkc(ped.get("sire")))
            brow = bms_map.get(_nfkc(ped.get("broodmare_sire")))
            ped_stats = None
            if srow or brow:
                ped_stats = {
                    "baseline": base_fuku,
                    "sire": {"rank": srow.get("rank"), "fuku": srow.get("fuku_rate"),
                             "n": srow.get("n_runs")} if srow else None,
                    "bms": {"rank": brow.get("rank"), "fuku": brow.get("fuku_rate"),
                            "n": brow.get("n_runs")} if brow else None,
                }

            horses.append({
                "umaban": umaban,
                "waku": hext.get("waku")
                        or (waku_of(umaban, field_size) if umaban else None),
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
                "style": classify_style(h.get("history")),
                "jockey": hext.get("jockey", ""),
                "kinryo": hext.get("kinryo"),
                "ninki": hext.get("ninki"),
                "zi": hext.get("zi"),
                "zi_rank": hext.get("zi_rank"),
                "kawari": hext.get("kawari", ""),
                "blinker": hext.get("blinker", ""),
                "taiju": hext.get("taiju"),
                "taiju_diff": hext.get("taiju_diff"),
                "trainer": hext.get("trainer", ""),
                "shozoku": hext.get("shozoku", ""),
                "why": h.get("why", []),
                "history": h.get("history"),
                "pedigree": h.get("pedigree"),
                "training": training,
                "ped_stats": ped_stats,
                "umami": umami_obj,
            })

        # 人気が weekly に無い場合はオッズ昇順から導出
        if horses and all(h["ninki"] is None for h in horses):
            ranked = sorted((h for h in horses if h["odds"]),
                            key=lambda x: x["odds"])
            for i, h in enumerate(ranked, start=1):
                h["ninki"] = i

        races_out.append({
            "race_id": rid,
            "rno": int(rid[-2:]) if rid[-2:].isdigit() else None,
            "place": meta.get("place", ""),
            "course": meta.get("course", ""),
            "klass": meta.get("class", ""),
            "race_name": meta.get("race_name", "") or rext.get("race_name", ""),
            "start_time": rext.get("start_time", ""),
            "baba": rext.get("baba", ""),
            "weather": rext.get("weather", ""),
            "field_size": field_size,
            "class_prior": meta.get("class_prior"),
            "confidence": race.get("race_confidence", {}),
            "judgment": race.get("buy_judgment", {}),
            "pairs": pairs_top(race),
            "horses": horses,
            "cowork": cowork.get(rid),
            "grade_scope": grade_map.get(rid),
            "result": results.get(rid),
        })

    places_seen = {r["place"] for r in races_out}
    places = [p for p in PLACE_ORDER if p in places_seen]
    places += sorted(places_seen - set(places))

    # コース成績 (place|course でユニーク化、日次で共有)
    courses: dict = {}
    for r in races_out:
        key = f"{r['place']}|{r['course']}"
        if key not in courses and course_stats.get(key):
            courses[key] = _slim_course(course_stats[key])

    # 好調教 Best5 (坂路 終い 200m が速い順、当日の出走馬に限る)
    name_loc = {}
    for r in races_out:
        for h in r["horses"]:
            name_loc.setdefault(_nfkc(h["name"]),
                                (r["place"], r["rno"], h["umaban"], h["name"]))
    top5 = []
    for nm, rec in hanro_map.items():
        loc = name_loc.get(nm)
        if loc and rec.get("lap1"):
            pl, rno, uma, disp = loc
            top5.append({"name": disp, "place": pl, "rno": rno, "umaban": uma,
                         "lap1": rec["lap1"], "t4f": rec.get("t4f")})
    top5.sort(key=lambda x: (x["lap1"], x.get("t4f") or 99))

    return deep_zen({
        "date": date_str,
        "places": places,
        "races": races_out,
        "courses": courses,
        "training_top5": top5[:5],
    })


# ---------------------------------------------------------------- main
def main() -> None:
    only_date = sys.argv[1] if len(sys.argv) > 1 else None
    SITE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    bundles = sorted(BUNDLE_DIR.glob("*_bundle.json"))
    if not bundles:
        print(f"bundle が見つかりません: {BUNDLE_DIR}")
        sys.exit(1)

    cowork = load_all_cowork()
    grade_map = load_all_grade_scope()
    wide_data = parse_wide_kekka()
    course_stats = load_course_stats()
    ped_index = load_pedigree_index()
    print(f"cowork_output: {len(cowork)} races / grade_scope: {len(grade_map)} races / "
          f"wide_kekka: {len(wide_data)} races / course_stats: {len(course_stats)} courses / "
          f"pedigree: {'OK' if ped_index else '無し'}")

    manifest_entries = []
    for path in bundles:
        date_str = path.name[:8]
        out_path = SITE_DATA_DIR / f"{date_str}.json"
        if only_date is None or date_str == only_date:
            day = transform_bundle(path, cowork, wide_data, course_stats,
                                   ped_index, grade_map)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(day, f, ensure_ascii=False, separators=(",", ":"))
            n_cw = sum(1 for r in day["races"] if r["cowork"])
            n_res = sum(1 for r in day["races"] if r["result"])
            print(f"  {date_str}: {len(day['races'])} races "
                  f"(cowork {n_cw} / result {n_res}) -> {out_path.relative_to(ROOT)}")
            places, n_races = day["places"], len(day["races"])
            has_results = n_res > 0
        elif out_path.exists():
            with open(out_path, encoding="utf-8") as f:
                prev = json.load(f)
            places = prev.get("places", [])
            n_races = len(prev.get("races", []))
            has_results = any(r.get("result") for r in prev.get("races", []))
        else:
            continue
        manifest_entries.append({"date": date_str, "places": places,
                                 "n_races": n_races, "has_results": has_results})

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
