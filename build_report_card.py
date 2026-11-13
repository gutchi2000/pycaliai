# -*- coding: utf-8 -*-
"""
build_report_card.py — AI の通信簿（2026-11-04）
=================================================
公開済みの site/data/{YYYYMMDD}.json（scrub_public 済み）だけから、条件別に
「◎の勝率・複勝圏率（Wilson 95% CI）」と「サイト表示買い目の的中率・回収率」を集計し、
site/data/report_card.json に書く。LLM は使わない。build_site.py の最後から呼ばれる（単体実行も可）。

次元: クラス / 芝ダ×距離帯 / 混戦度帯 / ◎の朝人気 / 頭数帯 / 場所 / ◎の AI 勝率帯 / 月
回収率はサイトに表示した買い目（cowork.bets の amount と bets_settled.received）。表示専用で実弾ではない。
払戻額そのもの（1 点ごとの payout）は出さず、合計の回収率だけを出す。

追加ブロック（2026-11-04）:
  weekly  : 開催週（土日を 1 組、キーは土曜の日付）ごとの ◎勝率・複勝圏率（Wilson 95% CI）と表示買い目の回収率、
            および累積（運用開始からの通算 ◎勝率・複勝圏率・回収率）の時系列。グラフ「AI の調子（週ごと）」用。
  miokuri : 参戦ガードで見送ったレースを「買っていたら」の成績（report_card_miokuri.py）。参戦 vs 見送りの比較。

実行:
    venv311\\Scripts\\python.exe build_report_card.py
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import defaultdict
from datetime import date as _date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SITE_DATA = ROOT / "site" / "data"
OUT = SITE_DATA / "report_card.json"

KLASS_ORDER = ["新馬", "未勝利", "1勝", "2勝", "3勝", "OP", "重賞"]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def klass_group(k: str) -> str:
    k = k or ""
    if re.search(r"G[123]|JG", k):
        return "重賞"
    if k.startswith("OP") or "オープン" in k:
        return "OP"
    for g in ("新馬", "未勝利", "1勝", "2勝", "3勝"):
        if k.startswith(g):
            return g
    return "その他"


def surface_dist(course: str) -> str | None:
    m = re.match(r"(芝|ダート|ダ|障害)[^\d]*(\d+)", course or "")
    if not m:
        return None
    s = {"芝": "芝", "ダート": "ダ", "ダ": "ダ", "障害": "障害"}[m.group(1)]
    if s == "障害":
        return None          # 障害は印も買い目も対象外
    dist = int(m.group(2))
    band = "〜1400" if dist <= 1400 else "1500〜1800" if dist <= 1800 else "1900〜2200" if dist <= 2200 else "2300〜"
    return f"{s} {band}"


def chaos_band(pct: float | None) -> str:
    """混戦度の百分位（judgment.chaos_pct、0=最も堅い〜1=最も混戦）で 4 分位に分ける。
    生の field_chaos_score は較正器差し替えで分布が動くので使わない（[[project_calibrator_shifts_chaos_gate]]）。"""
    if pct is None:
        return "不明"
    return "低（下位25%）" if pct < 0.25 else "やや低" if pct < 0.5 else "やや高" if pct < 0.75 else "高（上位25%）"


def ninki_band(n) -> str:
    if n is None:
        return "不明"
    return "1番人気" if n == 1 else "2〜3番人気" if n <= 3 else "4〜6番人気" if n <= 6 else "7番人気以下"


def field_band(n) -> str:
    if n is None:
        return "不明"
    return "〜10頭" if n <= 10 else "11〜14頭" if n <= 14 else "15頭〜"


def pwin_band(p) -> str:
    if p is None:
        return "不明"
    p *= 100
    return "〜15%" if p < 15 else "15〜25%" if p < 25 else "25〜35%" if p < 35 else "35%〜"


class Acc:
    __slots__ = ("n", "win", "top3", "fav", "bets_n", "bets_hit", "stake", "ret")

    def __init__(self):
        self.n = self.win = self.top3 = self.fav = self.bets_n = self.bets_hit = 0
        self.stake = self.ret = 0.0

    def add(self, win: bool, top3: bool, fav: bool, bets: list[tuple[float, float, bool]]):
        self.n += 1
        self.win += win
        self.top3 += top3
        self.fav += fav
        for amount, received, hit in bets:
            self.bets_n += 1
            self.bets_hit += hit
            self.stake += amount
            self.ret += received

    def row(self, key: str) -> dict:
        wlo, whi = wilson(self.win, self.n)
        tlo, thi = wilson(self.top3, self.n)
        return {
            "key": key, "n": self.n, "win": self.win, "top3": self.top3,
            "win_rate": round(self.win / self.n * 100, 1) if self.n else None, "win_lo": round(wlo * 100, 1), "win_hi": round(whi * 100, 1),
            "top3_rate": round(self.top3 / self.n * 100, 1) if self.n else None, "top3_lo": round(tlo * 100, 1), "top3_hi": round(thi * 100, 1),
            "fav_rate": round(self.fav / self.n * 100, 1) if self.n else None,
            "bets_n": self.bets_n, "bets_hit": self.bets_hit,
            "bets_hit_rate": round(self.bets_hit / self.bets_n * 100, 1) if self.bets_n else None,
            "stake": int(self.stake), "roi": round(self.ret / self.stake * 100, 1) if self.stake else None,
        }


ERA_LABEL = {"v5": "v5（〜5/16）", "v6-C0": "v6 旧版 C0", "v6-C1": "v6 C1", "v7": "v7（JRDB 特徴つき）", "unknown": "不明"}

DIMS = [
    ("era", "予測モデルの版"), ("verified", "予測の来歴"),
    ("klass", "クラス"), ("surface_dist", "芝ダ × 距離"), ("chaos", "混戦度"), ("honmei_ninki", "◎の朝の人気"),
    ("honmei_pwin", "◎の AI 勝率"), ("field", "頭数"), ("place", "競馬場"), ("month", "月"),
]


def week_key(d8: str) -> str:
    """開催週のキー = その週の土曜日 (YYYYMMDD)。土→当日、日→前日、月(祝日開催)→2 日前。
    金曜以前の変則開催は「次の土曜」ではなく直前の土曜に寄せる（土曜起点で 1 週を切る）。"""
    d = _date(int(d8[:4]), int(d8[4:6]), int(d8[6:8]))
    back = (d.weekday() - 5) % 7          # Sat=5 → 0, Sun=6 → 1, Mon=0 → 2, …
    return (d - timedelta(days=back)).strftime("%Y%m%d")


def weekly_series(weeks: dict[str, "Acc"]) -> dict:
    """週ごとの行 + 累積の行。Acc は既存の条件別集計と同じ定義（◎の勝率/複勝圏率/表示買い目の回収率）。"""
    rows, cum = [], []
    cn = cw = ct = cb = cbh = 0
    cs = cr = 0.0
    for wk in sorted(weeks):
        a = weeks[wk]
        r = a.row(wk)
        rows.append({
            "week": wk, "label": f"{int(wk[4:6])}/{int(wk[6:8])}", "n": a.n,
            "win_rate": r["win_rate"], "win_lo": r["win_lo"], "win_hi": r["win_hi"],
            "top3_rate": r["top3_rate"], "top3_lo": r["top3_lo"], "top3_hi": r["top3_hi"],
            "bets_n": a.bets_n, "bets_hit_rate": r["bets_hit_rate"], "stake": r["stake"], "roi": r["roi"],
        })
        cn += a.n; cw += a.win; ct += a.top3; cb += a.bets_n; cbh += a.bets_hit; cs += a.stake; cr += a.ret
        wlo, whi = wilson(cw, cn)
        tlo, thi = wilson(ct, cn)
        cum.append({
            "week": wk, "label": rows[-1]["label"], "n": cn,
            "win_rate": round(cw / cn * 100, 1) if cn else None, "win_lo": round(wlo * 100, 1), "win_hi": round(whi * 100, 1),
            "top3_rate": round(ct / cn * 100, 1) if cn else None, "top3_lo": round(tlo * 100, 1), "top3_hi": round(thi * 100, 1),
            "bets_n": cb, "bets_hit_rate": round(cbh / cb * 100, 1) if cb else None,
            "stake": int(cs), "roi": round(cr / cs * 100, 1) if cs else None,
        })
    return {"key": "saturday", "rows": rows, "cumulative": cum}


def build(write: bool = True) -> dict:
    files = sorted(SITE_DATA.glob("2026????.json"))
    overall = Acc()
    # 見出し用: 発走前の版と確認できた日を、予測モデルの世代ごとに数える。
    # 見出しは本番モデル (production_policy.json) の世代。その世代の開催日がまだ無いあいだ
    # (切替直後) は、確認済みの日がある最後の世代を出し、どの版かを明示する。
    by_era: dict[str, Acc] = defaultdict(Acc)
    era_dates: dict[str, list[str]] = defaultdict(list)
    try:
        from bundle_provenance import CURRENT_ERA, model_era
    except Exception:   # noqa: BLE001
        CURRENT_ERA, model_era = "unknown", (lambda d, b: "unknown")
    acc: dict[str, dict[str, Acc]] = {k: defaultdict(Acc) for k, _ in DIMS}
    weeks: dict[str, Acc] = defaultdict(Acc)
    n_days = 0
    dates = []
    for fp in files:
        day = json.loads(fp.read_text(encoding="utf-8"))
        used = False
        prov = day.get("provenance") or {}
        era = prov.get("model_era") or model_era(day["date"], {})
        verified = bool(prov.get("verified_pre_race"))
        for r in day.get("races", []):
            res = r.get("result") or {}
            if not res.get("top3") or not res.get("order"):
                continue
            sd = surface_dist(r.get("course", ""))
            if sd is None:
                continue
            hon = next((h for h in r.get("horses", []) if h.get("mark") == "◎"), None)
            if not hon:
                continue
            order = res["order"]
            pos = order.get(str(hon["umaban"]))
            # 着順 0 = 取消/除外/中止 (同着は同じ着順で入る。0 ではない)。
            # 発売前の取消 (scratched) は出走していないので分母から外す。
            # 最終オッズのある着順 0 (発売後の除外 or 競走中止) は出走扱いで「圏外」。
            # 以前は top3 の先頭に入った 0 着馬を 1 着として数えていた (2026-11-13 監査)。
            if hon["umaban"] in (res.get("scratched") or []):
                continue
            win, top3 = pos == 1, bool(pos) and pos <= 3
            fav = hon.get("ninki") == 1
            bets = []
            for b, s in zip((r.get("cowork") or {}).get("bets") or [], r.get("bets_settled") or []):
                if not isinstance(b, dict) or not isinstance(s, dict) or not s.get("settled"):
                    continue
                bets.append((float(b.get("amount") or 0), float(s.get("received") or 0), bool(s.get("is_win"))))
            keys = {
                "era": ERA_LABEL.get(era, era),
                "verified": "発走前の版と確認済み" if verified else "未検証（発走前の版が残っていない・作り直し）",
                "klass": klass_group(r.get("klass")), "surface_dist": sd,
                "chaos": chaos_band((r.get("judgment") or {}).get("chaos_pct")),
                "honmei_ninki": ninki_band(hon.get("ninki")), "honmei_pwin": pwin_band(hon.get("p_win")),
                "field": field_band(r.get("field_size")), "place": r.get("place") or "不明",
                "month": f"{day['date'][:4]}-{day['date'][4:6]}",
            }
            overall.add(win, top3, fav, bets)
            if verified and era != "unknown":
                by_era[era].add(win, top3, fav, bets)
                if not era_dates[era] or era_dates[era][-1] != day["date"]:
                    era_dates[era].append(day["date"])
            for k, v in keys.items():
                acc[k][v].add(win, top3, fav, bets)
            weeks[week_key(day["date"])].add(win, top3, fav, bets)
            used = True
        if used:
            n_days += 1
            dates.append(day["date"])

    def order_rows(k: str, rows: dict[str, Acc]) -> list[dict]:
        if k == "klass":
            keys = sorted(rows, key=lambda x: KLASS_ORDER.index(x) if x in KLASS_ORDER else 99)
        elif k == "chaos":
            order = ["低（下位25%）", "やや低", "やや高", "高（上位25%）", "不明"]
            keys = sorted(rows, key=lambda x: order.index(x) if x in order else 99)
        elif k == "honmei_pwin":
            order = ["〜15%", "15〜25%", "25〜35%", "35%〜", "不明"]
            keys = sorted(rows, key=lambda x: order.index(x) if x in order else 99)
        elif k == "surface_dist":
            bands = ["〜1400", "1500〜1800", "1900〜2200", "2300〜"]
            keys = sorted(rows, key=lambda x: (x.split(" ")[0], bands.index(x.split(" ")[1]) if x.split(" ")[1] in bands else 9))
        elif k in ("honmei_ninki", "field", "month"):
            keys = sorted(rows)
        else:
            keys = sorted(rows, key=lambda x: -rows[x].n)
        return [rows[x].row(x) for x in keys]

    head_era = CURRENT_ERA if by_era[CURRENT_ERA].n else max(
        (e for e in era_dates if by_era[e].n), key=lambda e: era_dates[e][-1], default=CURRENT_ERA)
    hd = era_dates.get(head_era) or []
    head = {**by_era[head_era].row("発走前確認済み"), "era": head_era, "era_label": ERA_LABEL.get(head_era, head_era),
            "is_production": head_era == CURRENT_ERA,
            "production_era_label": ERA_LABEL.get(CURRENT_ERA, CURRENT_ERA),
            "from": hd[0] if hd else None, "to": hd[-1] if hd else None, "n_days": len(hd)}
    out = {
        "built_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "period": {"from": dates[0] if dates else None, "to": dates[-1] if dates else None, "n_days": n_days},
        "n_races": overall.n,
        "overall": overall.row("全体"),
        # 見出し用: いまの本番モデル (v6-C1) で、発走前の版と確認できた日だけ。
        # 全体 (overall) は v5 期・旧版 C0 期・作り直した日を含む (2026-11-13 監査)。
        "current": head,
        "dims": [{"key": k, "label": lbl, "rows": order_rows(k, acc[k])} for k, lbl in DIMS],
        "weekly": weekly_series(weeks),
        "note": ("◎の勝率・複勝圏率は AI 本命（◎）の結果。CI は Wilson 95%。回収率はサイトに表示した買い目（表示専用、実弾ではない）の"
                 "合計払戻 ÷ 合計購入額。障害レースと◎無しのレースは除外。同着は着順どおりに扱う。"
                 "◎が発売前に取消になったレースは除外し、競走中止・発売後の除外は圏外として数える。"
                 "取消馬を含む買い目は返還、中止か除外か分からない馬を含む買い目は集計から外す。"
                 "見出しは本番の予測モデルで発走前の版と確認できた日だけ（切替直後でその版の開催日がまだ無いときは直前の版）。"
                 "「予測モデルの版」「予測の来歴」の表で内訳を示す。"),
    }
    # 見送り判断の成績（参戦 vs 買っていたら）。入力（cowork_output の a95_all / kekka）が無くても本体は壊さない。
    try:
        import report_card_miokuri as _rcm
        out["miokuri"] = _rcm.build_miokuri()
    except Exception as e:   # noqa: BLE001
        print(f"[report_card] miokuri skip: {e}")
        out["miokuri"] = None
    if write:
        OUT.write_bytes(json.dumps(out, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        mk = out.get("miokuri") or {}
        print(f"report_card: {overall.n} races / {n_days} days / {len(out['weekly']['rows'])} weeks -> site/data/report_card.json "
              f"(◎勝率 {out['overall']['win_rate']}% 複勝圏 {out['overall']['top3_rate']}% 回収率 {out['overall']['roi']}%"
              + (f" | 見送り比較: 参戦 {mk['sanse']['n_races']}R ROI {mk['sanse']['roi']}% / 見送り {mk['miokuri']['n_races']}R "
                 f"ROI {mk['miokuri']['roi']}%" if mk else "") + ")")
    return out


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    build()
