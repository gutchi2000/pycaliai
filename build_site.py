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
LEVEL_NORMS_PATH = ROOT / "data" / "level_norms.json"

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
                                    kawari, blinker, taiju, taiju_diff,
                                    trainer, shozoku}
    ※ ZI(TARGET独自指数)は Web 再掲不可のため取り込まない。
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
        # 新schema: style=決手 / 旧schema(〜2026-06): weight_change に決手が入っていた
        st = (STYLE_MAP.get(str(u.get("style", "")).strip())
              or STYLE_MAP.get(str(u.get("weight_change", "")).strip()))
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


# ---------------------------------------------------------------- 馬券決済
def _combos(selection: str, n: int, ordered: bool) -> list[tuple]:
    """'4-7,4-9' を combo タプル列に。n=2(馬連系)/3(三連系)。"""
    out = []
    for c in str(selection).split(","):
        parts = c.strip().split("-")
        if len(parts) != n:
            continue
        try:
            nums = [int(x) for x in parts]
        except (ValueError, TypeError):
            continue
        out.append(tuple(nums) if ordered else tuple(sorted(nums)))
    return out


# ---------------------------------------------------------------- TACT (topdown エンジン)
# 理由文のオッズ値はガイドライン対応で落とす (旧brain「（複X.X）」/ topdown「（券種 X.X倍）」両対応)
_TACT_ODDS_RE = re.compile(r"[（(](?:複[\d.]+|[^（）()]*?[\d.]+倍)[)）]")


def build_tact(race: dict) -> dict | None:
    """bundle の 1 レースに TACT (topdown エンジン) の推奨買い目を付ける。
    2026-08-09: gutchi_brain 決定木 (同一526RリプレイROI 72.4%) を本番と同一の
    compute_bets topdown (82.8%) に置換 — note の買い目が実際の馬券ラインと一致する。
    bets=[] は見送り (§0 hard / クリーン帯外)。engine 不在・例外時は None。"""
    try:
        from compute_bets import compute_race_bets
        tickets = compute_race_bets(race, budget=10000, force_floor=True).get("bets") or []
    except Exception as e:
        print(f"[tact skip] {race.get('race_id')}: {e}")
        return None
    return {
        "version": "1.0td",
        # 金額は出さない (買う人が決める)。買い目+定性理由のみ公開
        "bets": [{"type": t["馬券種"], "selection": t["買い目"],
                  "reason": _TACT_ODDS_RE.sub("", t["理由"]).strip()}
                 for t in tickets],
    }


def settle_bet(btype: str, selection: str, cost: float, res: dict) -> dict:
    """1 bet を決済。返り値 {is_win, payout(配当/100円), received, profit, settled}。
    settled=False は決済不能 (ワイド払戻未取込等) で集計外。compute_bet_pl と同ロジック。"""
    btype = (btype or "").strip()
    selection = str(selection or "").strip()
    pays = res.get("pays") or {}
    top3 = res.get("top3") or []
    miss = {"is_win": False, "payout": 0, "received": 0.0, "profit": -cost, "settled": True}
    if cost <= 0:
        return {"is_win": False, "payout": 0, "received": 0.0, "profit": 0.0, "settled": True}

    def win(pay, unit):
        recv = unit * (pay or 0) / 100.0
        return {"is_win": True, "payout": pay or 0, "received": recv,
                "profit": recv - cost, "settled": True}

    if btype == "単勝":
        try:
            u = int(selection)
        except ValueError:
            return miss
        return win(pays.get("tan"), cost) if (top3 and u == top3[0]) else miss
    if btype == "複勝":
        try:
            u = int(selection)
        except ValueError:
            return miss
        if u in top3:
            return win((pays.get("fuku") or {}).get(str(u)), cost)
        return miss
    if btype in ("馬連", "馬単"):
        ordered = btype == "馬単"
        combos = _combos(selection, 2, ordered)
        if not combos or len(top3) < 2:
            return miss
        w = (top3[0], top3[1]) if ordered else tuple(sorted([top3[0], top3[1]]))
        if w in combos:
            return win(pays.get("umatan" if ordered else "umaren"), cost / len(combos))
        return miss
    if btype == "ワイド":
        combos = _combos(selection, 2, False)
        if not combos:
            return miss
        wide = pays.get("wide") or {}
        if not wide:  # 払戻未取込 → 決済不能 (集計外)
            return {"is_win": False, "payout": 0, "received": 0.0, "profit": None, "settled": False}
        unit = cost / len(combos)
        recv, best, nhit = 0.0, 0, 0
        for a, b in combos:
            p = wide.get(f"{a}-{b}")
            if p:
                recv += unit * p / 100.0
                best = max(best, p)
                nhit += 1
        if nhit:
            return {"is_win": True, "payout": best, "received": recv,
                    "profit": recv - cost, "settled": True}
        return miss
    if btype in ("三連複", "三連単"):
        ordered = btype == "三連単"
        combos = _combos(selection, 3, ordered)
        if not combos or len(top3) < 3:
            return miss
        w = tuple(top3[:3]) if ordered else tuple(sorted(top3[:3]))
        if w in combos:
            return win(pays.get("sanrentan" if ordered else "sanrenpuku"), cost / len(combos))
        return miss
    return miss


# ---------------------------------------------------------------- 馬/メンバーレベル
# ELO/Glicko(蓄積)にも ZI/補正タイム(TARGET外部指数=Web再掲不可)にも依存せず、
# history.runs の公開事実(着順/人気)だけで「馬レベル」を作る (level_metric.py)。
# data/level_norms.json (build_level_norms.py 生成) で 0-100 正規化 + クラス別基準。
import level_metric as _LM

_LEVEL_NORMS_CACHE = None
# メンバーレベル: 上位3頭平均 level を ref のどの分位以上かで判定 (境界キー, tier, ラベル)
_LEVEL_TIERS = [("p80", "S", "ハイレベル"), ("p60", "A", "やや強め"),
                ("p40", "B", "標準"), ("p20", "C", "やや軽い"), (None, "D", "低調")]


def _level_norms() -> dict:
    global _LEVEL_NORMS_CACHE
    if _LEVEL_NORMS_CACHE is None:
        try:
            _LEVEL_NORMS_CACHE = json.loads(LEVEL_NORMS_PATH.read_text(encoding="utf-8"))
        except Exception:
            _LEVEL_NORMS_CACHE = {}
    return _LEVEL_NORMS_CACHE


def _raw_to_100(raw, anchors):
    n = len(anchors)
    step = 100 / (n - 1)
    # 線形補間 (numpy 非依存)
    for i in range(n - 1):
        if raw <= anchors[i + 1]:
            lo, hi = anchors[i], anchors[i + 1]
            frac = 0 if hi == lo else (raw - lo) / (hi - lo)
            return max(0.0, min(100.0, (i + frac) * step))
    return 100.0


def _level_tier(score100):
    return ("S" if score100 >= 80 else "A" if score100 >= 60
            else "B" if score100 >= 40 else "C" if score100 >= 20 else "D")


def horse_level(history) -> dict | None:
    """各馬の近走成績レベル {score(0-100), tier}。出走歴なしは None。"""
    norms = _level_norms()
    raw = _LM.horse_level_raw(history)
    if raw is None or not norms.get("raw_anchors"):
        return None
    s = round(_raw_to_100(raw, norms["raw_anchors"]))
    return {"score": s, "tier": _level_tier(s)}


def _pct_in_class(value, ref):
    """value(top3/平均レベル) を クラス分位 p20/40/60/80 で 0-100 パーセンタイルに。"""
    if value is None or ref.get("p20") is None:
        return None
    anchors = [(0.0, ref["p20"] - 8), (0.2, ref["p20"]), (0.4, ref["p40"]),
               (0.6, ref["p60"]), (0.8, ref["p80"]), (1.0, ref["p80"] + 8)]
    for (q0, v0), (q1, v1) in zip(anchors, anchors[1:]):
        if value <= v1:
            p = (q0 + (q1 - q0) * ((value - v0) / (v1 - v0 or 1))) * 100
            return max(0, min(100, round(p)))
    return 100


def compute_member_level(klass: str, horse_levels: list) -> dict | None:
    """出走馬の level(score) 群 → メンバーレベル(上位3頭平均を同クラス分布で位置づけ)。"""
    norms = _level_norms()
    if not norms:
        return None
    scores = sorted((hl["score"] for hl in horse_levels if hl), reverse=True)
    if len(scores) < 3:
        return {"tier": None, "label": "実績データ不足", "top_level": None,
                "field_level": round(sum(scores) / len(scores)) if scores else None,
                "class_key": None, "class_avg": None, "n_class": 0,
                "pct": None, "avg_pct": None}
    top3 = round(sum(scores[:3]) / 3)
    field_level = round(sum(scores) / len(scores))
    key = _LM.class_group(klass)
    ref = (norms.get("classes", {}) or {}).get(key) or norms.get("global") or {}
    tier, label = "D", "低調"
    for pk, t, lb in _LEVEL_TIERS:
        if pk is None or top3 >= ref.get(pk, 1e9):
            tier, label = t, lb
            break
    return {"tier": tier, "label": label, "top_level": top3, "field_level": field_level,
            "class_key": key, "class_avg": ref.get("mean"), "n_class": ref.get("n"),
            "pct": _pct_in_class(top3, ref), "avg_pct": _pct_in_class(ref.get("mean"), ref)}


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
                "level": horse_level(h.get("history")),
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
            "member_level": compute_member_level(
                meta.get("class", ""), [h["level"] for h in horses]),
            "confidence": race.get("race_confidence", {}),
            "judgment": race.get("buy_judgment", {}),
            "pairs": pairs_top(race),
            "horses": horses,
            "cowork": cowork.get(rid),
            "tact": build_tact(race),
            "grade_scope": grade_map.get(rid),
            "result": results.get(rid),
        })

        # Cowork / TACT 買い目を結果で決済 (的中/配当/収支)
        cw = cowork.get(rid)
        res = results.get(rid)
        if cw and cw.get("bets") and res:
            races_out[-1]["bets_settled"] = [
                settle_bet(b.get("type"), b.get("selection"),
                           float(b.get("amount") or 0), res)
                for b in cw["bets"]
            ]
        tact = races_out[-1]["tact"]
        if tact and tact.get("bets") and res:
            # 金額非公開のため名目 ¥100 で決済 (的中判定バッジ用。収支は出さない)
            races_out[-1]["tact_settled"] = [
                settle_bet(b["type"], b["selection"], 100.0, res)
                for b in tact["bets"]
            ]

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

    # オッズ取得時点 = TARGET 出走表 CSV のエクスポート時刻 (ファイル mtime)。
    # weekly CSV が無ければ bundle 生成時刻で代用。
    odds_src = WEEKLY_DIR / f"{date_str}.csv"
    odds_asof = datetime.fromtimestamp(
        (odds_src if odds_src.exists() else path).stat().st_mtime
    ).strftime("%m/%d %H:%M")

    return deep_zen({
        "date": date_str,
        "odds_asof": odds_asof,
        "places": places,
        "races": races_out,
        "courses": courses,
        "training_top5": top5[:5],
    })


# ---------------------------------------------------------------- 成績集計
def build_results_json() -> dict:
    """site/data/*.json を全走査し、Cowork 的中一覧 + 累計集計を results.json に。"""
    # 撤廃券種 (2026-06-18): 馬単は構造的回収不能(2/122)で本番から撤廃済 → 累計/ROI/by_type/
    # カードから除外する。生データ(site/data)は保持し、agg.excluded に件数を残して透明化。
    DISCONTINUED = {"馬単"}
    hits = []
    by_type: dict = {}
    by_date: dict = {}
    tot_cost = tot_profit = 0.0
    n_bets = n_wins = n_unset = 0
    unset_cost = 0.0
    ex_n = 0; ex_cost = ex_profit = 0.0

    for p in sorted(SITE_DATA_DIR.glob("[0-9]" * 8 + ".json")):
        with open(p, encoding="utf-8") as f:
            day = json.load(f)
        date = day.get("date", p.stem)
        for r in day.get("races", []):
            bs = r.get("bets_settled")
            cw = r.get("cowork")
            if not bs or not cw:
                continue
            bets = cw.get("bets") or []
            for bet, st in zip(bets, bs):
                cost = float(bet.get("amount") or 0)
                t = bet.get("type") or "?"
                if not st.get("settled", True):
                    n_unset += 1
                    unset_cost += cost
                    continue
                profit = st.get("profit") or 0.0
                if t in DISCONTINUED:        # 撤廃券種は累計/by_type から除外(別枠で記録)
                    ex_n += 1; ex_cost += cost; ex_profit += profit
                    continue
                n_bets += 1
                tot_cost += cost
                tot_profit += profit
                bt = by_type.setdefault(t, {"n": 0, "wins": 0, "cost": 0.0, "profit": 0.0})
                bt["n"] += 1
                bt["cost"] += cost
                bt["profit"] += profit
                d = by_date.setdefault(date, {"cost": 0.0, "profit": 0.0, "n": 0, "wins": 0})
                d["cost"] += cost
                d["profit"] += profit
                d["n"] += 1
                if st.get("is_win"):
                    n_wins += 1
                    bt["wins"] += 1
                    d["wins"] += 1
                    hits.append({
                        "date": date, "race_id": r.get("race_id"),
                        "place": r.get("place"), "rno": r.get("rno"),
                        "name": r.get("race_name") or r.get("klass") or "",
                        "btype": t, "selection": bet.get("selection"),
                        "payout": st.get("payout"), "stake": cost,
                        "profit": round(profit),
                    })

    hits.sort(key=lambda h: (h["date"], h.get("payout") or 0), reverse=True)
    for t in by_type.values():
        t["cost"] = round(t["cost"]); t["profit"] = round(t["profit"])
        t["roi"] = round((t["cost"] + t["profit"]) / t["cost"] * 100, 1) if t["cost"] else 0.0
    cum = 0.0
    date_series = []
    for date in sorted(by_date):
        d = by_date[date]
        cum += d["profit"]
        date_series.append({"date": date, "cost": round(d["cost"]),
                            "profit": round(d["profit"]), "cum": round(cum),
                            "n": d["n"], "wins": d["wins"]})

    return {
        "agg": {
            "total_cost": round(tot_cost), "total_profit": round(tot_profit),
            "roi": round((tot_cost + tot_profit) / tot_cost * 100, 1) if tot_cost else 0.0,
            "n_bets": n_bets, "n_wins": n_wins,
            "hit_rate": round(n_wins / n_bets * 100, 1) if n_bets else 0.0,
            "n_unsettled": n_unset, "unsettled_cost": round(unset_cost),
            "excluded_btypes": sorted(DISCONTINUED), "n_excluded": ex_n,
            "excluded_cost": round(ex_cost), "excluded_profit": round(ex_profit),
            "by_type": by_type, "by_date": date_series,
        },
        "hits": hits,
    }


# ---------------------------------------------------------------- 公開スクラブ
# SHAP 根拠 (why[].value) に出してはいけない特徴。
#   TARGET 独自指数 (補正タイム) / 調教タイム / 上り3F・着差などの計時データ。
#   ※ 順位・回数・率は公知事実の加工なので残す (前走上り3F順 など)。
_WHY_BLOCK_FEATS = frozenset({
    "prev_hosei", "prev_hosei9",                       # TARGET 補正タイム (独自指数)
    "trnH_Time1", "trnH_Time4", "trnH_Lap4",           # 坂路 調教タイム
    "trnW_3F", "trnW_Lap1", "trnW_Lap3",               # WC 調教タイム
    "kako5_avg_agari3f", "前走上り3F", "前走Ave-3F",     # 上り3F タイム
    "前3F", "前PCI", "前走着差タイム",                    # ペース・着差の計時データ
})

# Cowork の地の文に混じるオッズ生値。
#   例: "おいしい(単勝 5.9倍)" / "[T-10オッズ 単複14頭/ワイド91組/馬単182組]"
_ODDS_PAREN_RE = re.compile(
    r"[（(]\s*(?:単勝|複勝|ワイド|馬連|馬単|三連複|三連単|オッズ)?\s*[0-9]+(?:\.[0-9]+)?\s*倍[^）)]*[）)]"
)
_ODDS_BRACKET_RE = re.compile(r"[\[［][^\]］]*オッズ[^\]］]*[\]］]")
# 数値+倍。直後の助詞まで一緒に食う ("1.8倍の断然人気"→"断然人気", "38倍と人気薄"→"人気薄")
_ODDS_BARE_RE = re.compile(r"[0-9]+(?:\.[0-9]+)?\s*倍(?:台)?(?:の|と|で|から|まで)?")
# 生値を抜いた跡に残る句読点の連なりを畳む
_ARTIFACT_SUBS = (
    (re.compile(r"[、,]\s*[。.]"), "。"),
    (re.compile(r"[。.]\s*[。.]+"), "。"),
    (re.compile(r"[（(]\s*[）)]"), ""),
    (re.compile(r"[、,]\s*[、,]+"), "、"),
    (re.compile(r"\s{2,}"), " "),
)


def _scrub_text(s):
    """Cowork 生成文からオッズ生値を落とす (文意は残す)。

    advisor のように list[dict] で来る欄があるので再帰で潜る。
    """
    if isinstance(s, list):
        return [_scrub_text(x) for x in s]
    if isinstance(s, dict):
        return {k: _scrub_text(v) for k, v in s.items()}
    if not isinstance(s, str):
        return s
    s = _ODDS_BRACKET_RE.sub("", s)
    s = _ODDS_PAREN_RE.sub("", s)
    s = _ODDS_BARE_RE.sub("", s)
    for pat, rep in _ARTIFACT_SUBS:
        s = pat.sub(rep, s)
    return s.strip(" 、,").strip()


def scrub_public(day: dict) -> dict:
    """JRA-VAN 投稿ガイドライン対応 (2026-07-31): 公開 JSON から生データを落とす。

    根拠 (jra-van.jp/info/post_guide.html 原文確認済み):
      - 「投稿できないコンテンツ: 調教タイム」 → training_top5 / horses[].training
      - 「JV-Linkから取得したデータは投稿できません」 → ライブ馬体重 (taiju)
      - 「有料会員限定情報の過度な転載・公開」の禁止 → オッズ生値・複勝レンジ・
        馬連ペアオッズ・払戻金全券種の網羅掲載、EV等オッズが逆算可能な数値
    自作の予想・確率・印・集計 (表やグラフ) は OK 側 (出典表記はサイト footer)。

    2026-08-06 追加 (note 販売記事の生成時に発見した本番混入 2 経路):
      - SHAP 根拠 why[].value に補正タイム/調教タイム/上り3F が生値で出ていた
        (app.js の根拠バーが label + value を描画。8/2 単日で 305 件)
      - Cowork の race_reason / bets[].reason にオッズ生値が地の文で入っていた
        ("おいしい(単勝 5.9倍)" 等。8/2 単日で 136 件)
    """
    day.pop("training_top5", None)
    day.pop("odds_asof", None)
    for r in day.get("races", []):
        res = r.get("result")
        if res:
            res.pop("pays", None)
        cw = r.get("cowork")
        if isinstance(cw, dict):
            for k in ("race_reason", "race_label", "race_nature", "advisor"):
                if k in cw:
                    cw[k] = _scrub_text(cw[k])
            for b in cw.get("bets") or []:
                if isinstance(b, dict) and "reason" in b:
                    b["reason"] = _scrub_text(b["reason"])

        # ---- 買い目・枠格付け・見送り理由は note 有料記事の専売 (2026-08-06)
        # 結果が出るまでは伏せ、確定後は「実績の正直開示」として出す。
        # 予想そのもの (印・確率・レベル・妙味グレード・根拠) は無料のまま。
        if not r.get("result"):
            if isinstance(cw, dict):
                for k in ("bets", "race_nature", "race_reason"):
                    cw.pop(k, None)
            tact = r.get("tact")
            if isinstance(tact, dict):
                tact.pop("bets", None)
            jd = r.get("judgment")
            if isinstance(jd, dict):
                # value_horses (妙味馬) は予想側なので残す
                for k in ("category", "headline", "detail",
                          "kenshu_hint", "waku_tag"):
                    jd.pop(k, None)
        for p in r.get("pairs", []) or []:
            p.pop("umaren_odds", None)
        vhs = (r.get("judgment") or {}).get("value_horses") or []
        for v in vhs:
            for k in ("ev_tan", "ev_fuku", "umami_tan", "umami_fuku"):
                v.pop(k, None)
        for h in r.get("horses", []) or []:
            for k in ("odds", "fuku_low", "fuku_high", "ev_tan",
                      "training", "taiju", "taiju_diff"):
                h.pop(k, None)
            # 根拠バーは「どの特徴がどれだけ押した/引いた」までは残し、
            # 禁止カテゴリの特徴そのものを行ごと落とす (value だけ消すと
            # app.js の value!=null フィルタで空行になり寄与も失われるため)
            why = h.get("why")
            if isinstance(why, list):
                h["why"] = [w for w in why
                            if not (isinstance(w, dict)
                                    and w.get("feat") in _WHY_BLOCK_FEATS)]
            um = h.get("umami")
            if isinstance(um, dict):
                h["umami"] = {k: um[k] for k in ("grade", "side") if k in um}
            # 過去走の上り3Fタイムは計時データの網羅転載色が強いので落とす
            # (着順/人気/コース等の公知事実は残す)
            for run in ((h.get("history") or {}).get("runs") or []):
                run.pop("agari3f", None)
    return day


# ---------------------------------------------------------------- main
def main() -> None:
    only_date = sys.argv[1] if len(sys.argv) > 1 else None
    SITE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 分析カードの as-of レーティング辞書を鮮度チェック(元parquet更新時のみ再生成、平時は即skip)
    try:
        import build_explain_ratings as _ber
        _ber.ensure()
    except Exception as e:
        print(f"[explain_ratings skip] {e}")

    # 馬レベル norm を鮮度チェック(バンドル更新時のみ再生成、平時は即skip)
    try:
        import build_level_norms as _bln
        _bln.ensure()
        globals()["_LEVEL_NORMS_CACHE"] = None   # 再生成後に読み直す
    except Exception as e:
        print(f"[level_norms skip] {e}")

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
            scrub_public(day)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(day, f, ensure_ascii=False, separators=(",", ":"))
            # 分析カード(Explainability) を同時生成 (LLMは重いので既定OFF; EXPLAIN_LLM=1で有効)
            try:
                import os as _os
                from explain_card import write_day_explain
                n_card = write_day_explain(day, SITE_DATA_DIR / "explain",
                                           with_llm=bool(_os.environ.get("EXPLAIN_LLM")))
                print(f"    explain cards: {n_card}")
            except Exception as e:
                print(f"    [explain skip] {e}")
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

    # 各馬の指数推移 (全キャリア) — 最新2開催日ぶんだけ生成 (drawer のチャート用)
    try:
        from build_horse_career import build_careers
        latest = sorted((e["date"] for e in manifest_entries), reverse=True)[:2]
        build_careers(latest)
    except Exception as e:
        print(f"[career skip] {e}")

    manifest_entries.sort(key=lambda e: e["date"], reverse=True)
    manifest = {
        "built_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "model": "v6",
        "dates": manifest_entries,
    }
    with open(SITE_DATA_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"manifest: {len(manifest_entries)} dates")

    # 分析カードの manifest (explain.html 用)
    try:
        from explain_card import write_explain_manifest
        write_explain_manifest(SITE_DATA_DIR / "explain")
    except Exception as e:
        print(f"[explain manifest skip] {e}")

    # 成績 (Cowork 的中一覧 + 累計収支)
    results_payload = build_results_json()
    results_payload["built_at"] = manifest["built_at"]
    with open(SITE_DATA_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(results_payload, f, ensure_ascii=False, separators=(",", ":"))
    a = results_payload["agg"]
    print(f"results: hits {len(results_payload['hits'])} / "
          f"{a['n_bets']} bets ROI {a['roi']}% hit {a['hit_rate']}% "
          f"profit {a['total_profit']:+,} (unsettled {a['n_unsettled']})")

    # 今日の馬場バイアス (fetch_baba_today.py 出力をサイトに同梱)
    baba_src = ROOT / "data" / "baba_today.json"
    if baba_src.exists():
        (SITE_DATA_DIR / "baba_today.json").write_text(
            baba_src.read_text(encoding="utf-8"), encoding="utf-8")
        print("baba_today: copied to site/data")

    # 実現トラックバイアス (build_realized_bias.py 出力。出走表タブの妙味隣カード)
    rb_src = ROOT / "data" / "realized_bias.json"
    if rb_src.exists():
        (SITE_DATA_DIR / "realized_bias.json").write_text(
            rb_src.read_text(encoding="utf-8"), encoding="utf-8")
        print("realized_bias: copied to site/data")


if __name__ == "__main__":
    main()
