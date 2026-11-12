# -*- coding: utf-8 -*-
"""
power_rating.py — 全馬共通ものさしの「強さスコア」（非公開・ローカル専用）
================================================================================
「この馬とこの馬、どっちが強い?」に 1 本の数字で答えるための、レース横断レーティング。
全レースの着順だけから多頭数 Elo を逐次更新し、現役最上位馬 = 100 の線形スケールに直す。

  * 入力は着順（公開事実）だけ。オッズ・人気・外部指数は使わない。
  * 逐次更新（そのレースの発走前の値しか予測に使わない）なので未来情報は混ざらない。
  * 予測モデル v6 とは独立。買い目・サイト・bundle には一切つながない（出力は
    reports/power_rating/ = .gitignore 済み）。馬券用の指標ではない。

データ源（存在するものだけ読む）:
  1. data/master_v2_20130105-20251228.csv   2013〜2025 の全着順（ローカルのみ）
  2. data/kekka/{YYYYMMDD}.csv              2026〜 の全着順（TARGET 結果）
     + data/weekly/{YYYYMMDD}.csv           そのレースのクラス名・芝ダ（新規馬の初期値用）

使い方:
  python power_rating.py build                       # レーティング生成 + 前向き検証
  python power_rating.py compare クロワデュノール ホウオウビスケッツ
  python power_rating.py top --n 30                  # 現役上位
  python power_rating.py card 20261108               # その日の出走表を強さ順に
  python power_rating.py tune                        # K の格子探索（master がある環境で）
  python power_rating.py selftest                    # 合成データで復元性を検算

スコアの読み方:
  score = 100 × (R − R_lo) / (R_hi − R_lo)   R_hi = 現役最上位, R_lo = 現役の下位 1% 点
  線形なので「点差」に意味がある。点差 d のとき、同じレースで先着する確率は
      P = 1 / (1 + 10^(−d × elo_per_point / 400))
  （elo_per_point は build 時に meta.json へ保存。compare が自動で換算する）
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import sys
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Iterator

BASE = Path(__file__).resolve().parent
MASTER = BASE / "data" / "master_v2_20130105-20251228.csv"
KEKKA_DIR = BASE / "data" / "kekka"
WEEKLY_DIR = BASE / "data" / "weekly"
OUT_DIR = BASE / "reports" / "power_rating"

log = logging.getLogger("power_rating")

# ---------------------------------------------------------------- 定数
ELO_SCALE = 400.0
# 新規馬の初期値（そのレースに実績馬が 3 頭未満のときだけ使う。クラス名は NFKC 後）
CLASS_INIT: dict[str, float] = {
    "新馬": 1400.0, "未勝利": 1400.0,
    "500万": 1500.0, "1勝": 1500.0,
    "1000万": 1580.0, "2勝": 1580.0,
    "1600万": 1650.0, "3勝": 1650.0,
    "オープン": 1720.0, "OP(L)": 1720.0,
    "重賞": 1760.0, "G3": 1760.0, "G2": 1800.0, "G1": 1850.0,
}
DEFAULT_INIT = 1450.0
ESTABLISHED_N = 3          # これ以上走っていれば「実績馬」
NAME_REUSE_GAP_DAYS = 1800  # 同名でもこれ以上あいていれば別馬（馬名再使用）とみなす
ACTIVE_DAYS = 365           # 現役 = 基準日からこの日数以内に出走
MIN_STARTS_FOR_SCALE = 3    # スケールのアンカーに使う最低出走数


@dataclass
class Params:
    """更新パラメータ。K は出走数とともに k0 → k0×floor_ratio へ減衰する。"""

    k0: float = 96.0
    floor_ratio: float = 0.25
    tau: float = 6.0

    def k(self, n_starts: int) -> float:
        floor = self.k0 * self.floor_ratio
        return floor + (self.k0 - floor) * math.exp(-n_starts / self.tau)


@dataclass
class Race:
    """1 レース分の着順。runners = [(馬名, 着順>=1)]。"""

    day: date
    race_id: str
    klass: str | None
    surface: str | None
    runners: list[tuple[str, int]]


@dataclass
class Horse:
    rating: float
    n: int = 0
    last_day: date | None = None
    last_class: str | None = None
    peak: float = 0.0
    n_turf: int = 0
    n_dirt: int = 0


@dataclass
class EvalAcc:
    """前向き検証の集計器（更新前のレーティングで予測 → 実着順と照合）。"""

    races: int = 0
    top_win: int = 0
    top_in3: int = 0
    rand_win: float = 0.0
    rand_in3: float = 0.0
    pairs: int = 0
    concordant: float = 0.0
    logloss: float = 0.0
    bins: dict[int, list[float]] = field(default_factory=dict)

    def add_race(self, pre: list[float], pos: list[int]) -> None:
        n = len(pre)
        self.races += 1
        top = max(range(n), key=lambda i: pre[i])
        self.top_win += int(pos[top] == 1)
        self.top_in3 += int(pos[top] <= 3)
        self.rand_win += 1.0 / n
        self.rand_in3 += min(3, n) / n
        for i in range(n):
            for j in range(i + 1, n):
                if pos[i] == pos[j]:
                    continue
                p = expected(pre[i], pre[j])
                y = 1.0 if pos[i] < pos[j] else 0.0
                self.pairs += 1
                self.concordant += 1.0 if (p > 0.5) == (y == 1.0) else (0.5 if p == 0.5 else 0.0)
                pc = min(max(p, 1e-6), 1 - 1e-6)
                self.logloss -= y * math.log(pc) + (1 - y) * math.log(1 - pc)
                # 較正: 強い側から見た確率で 5pt 刻み
                ps, ys = (p, y) if p >= 0.5 else (1 - p, 1 - y)
                b = self.bins.setdefault(min(int(ps * 20), 19), [0.0, 0.0, 0.0])
                b[0] += 1
                b[1] += ps
                b[2] += ys

    def summary(self) -> dict:
        if not self.races:
            return {"races": 0}

        def wilson(k: float, n: int) -> list[float]:
            if n == 0:
                return [0.0, 0.0]
            z, ph = 1.96, k / n
            den = 1 + z * z / n
            c = (ph + z * z / (2 * n)) / den
            h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
            return [round(c - h, 4), round(c + h, 4)]

        return {
            "races": self.races,
            "top_win_rate": round(self.top_win / self.races, 4),
            "top_win_ci95": wilson(self.top_win, self.races),
            "top_in3_rate": round(self.top_in3 / self.races, 4),
            "top_in3_ci95": wilson(self.top_in3, self.races),
            "random_win_rate": round(self.rand_win / self.races, 4),
            "random_in3_rate": round(self.rand_in3 / self.races, 4),
            "pairs": self.pairs,
            "pair_concordance": round(self.concordant / max(self.pairs, 1), 4),
            "pair_logloss": round(self.logloss / max(self.pairs, 1), 5),
            "calibration": [
                {"pred": round(v[1] / v[0], 3), "actual": round(v[2] / v[0], 3), "n": int(v[0])}
                for _, v in sorted(self.bins.items())
            ],
        }


# ---------------------------------------------------------------- 基本関数
def nfkc(s: object) -> str:
    return unicodedata.normalize("NFKC", str(s or "")).strip()


def expected(ra: float, rb: float) -> float:
    """A が B に先着する確率（Elo ロジスティック）。"""
    return 1.0 / (1.0 + 10.0 ** (-(ra - rb) / ELO_SCALE))


def is_jump(klass: str | None, surface: str | None) -> bool:
    k, s = nfkc(klass), nfkc(surface)
    return k.startswith("J") or "障" in k or "障" in s


def norm_surface(s: object) -> str | None:
    t = nfkc(s)
    if t.startswith("芝") or t == "T":
        return "芝"
    if t.startswith("ダ") or t == "D":
        return "ダ"
    if "障" in t:
        return "障"
    return None


# ---------------------------------------------------------------- 読み込み
def load_master(path: Path = MASTER) -> list[Race]:
    """master_v2 から平地の全着順を読む。無ければ空リスト。"""
    if not path.exists():
        log.info("master なし（%s）→ 2026 データのみで計算します", path.name)
        return []
    import pandas as pd  # 遅延 import（compare/top は pandas 不要）

    cols = ["日付", "レースID(新/馬番無)", "馬名", "クラス名", "トラックコード(JV)", "着順", "芝・ダ"]
    head = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns
    use = [c for c in cols if c in head]
    missing = sorted(set(cols[:3] + ["着順"]) - set(use))
    if missing:
        raise KeyError(f"master に必須列がありません: {missing}")
    df = pd.read_csv(path, usecols=use, encoding="utf-8-sig", low_memory=False)
    df["pos"] = pd.to_numeric(df["着順"], errors="coerce").fillna(0).astype(int)
    df = df[df["pos"] >= 1]
    if "トラックコード(JV)" in df:
        tc = pd.to_numeric(df["トラックコード(JV)"], errors="coerce")
        df = df[~tc.between(51, 59)]
    df["day"] = pd.to_datetime(df["日付"].astype(str), format="%Y%m%d").dt.date
    races: list[Race] = []
    for rid, g in df.groupby("レースID(新/馬番無)", sort=False):
        klass = nfkc(g["クラス名"].iloc[0]) if "クラス名" in g else None
        surf = norm_surface(g["芝・ダ"].iloc[0]) if "芝・ダ" in g else None
        if is_jump(klass, surf):
            continue
        races.append(Race(g["day"].iloc[0], str(rid), klass, surf,
                          list(zip(g["馬名"].map(nfkc), g["pos"]))))
    log.info("master: %d レース (%s 〜 %s)", len(races),
             min(r.day for r in races), max(r.day for r in races))
    return races


def _decode(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("cp932", "utf-8-sig", "utf-8"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("cp932", errors="replace")


def read_weekly(path: Path) -> list[dict]:
    """週次出走表 CSV → [{race_id, klass, surface, place, rno, name, horses:[(馬番, 馬名)]}]。"""
    out: list[dict] = []
    cur: dict | None = None
    hcols: list[str] = []
    for line in _decode(path).splitlines():
        c = line.split(",")
        if c[0] == "レースID(新)":
            continue
        if c[0] == "枠番":
            hcols = c
            continue
        if len(c) == 19 and len(c[0]) == 16 and c[0].isdigit():
            cur = {"race_id": c[0], "place": c[3], "rno": c[5], "name": c[6],
                   "klass": nfkc(c[7]), "surface": norm_surface(c[8]), "dist": c[9], "horses": []}
            out.append(cur)
        elif cur is not None and hcols and len(c) == len(hcols):
            try:
                i_name = hcols.index("馬名S")
                i_ban = hcols.index("番") if "番" in hcols else hcols.index("馬番")
                cur["horses"].append((int(c[i_ban]), nfkc(c[i_name])))
            except (ValueError, IndexError):
                continue
    return out


def load_2026(after: date | None) -> list[Race]:
    """data/kekka/{date}.csv（全着順）+ data/weekly/{date}.csv（クラス）。after 以前は捨てる。"""
    races: list[Race] = []
    n_noclass = 0
    for f in sorted(KEKKA_DIR.glob("[0-9]" * 8 + ".csv")):
        day = datetime.strptime(f.stem, "%Y%m%d").date()
        if after and day <= after:
            continue
        meta: dict[str, dict] = {}
        wk = WEEKLY_DIR / f.name
        if wk.exists():
            meta = {r["race_id"]: r for r in read_weekly(wk)}
        by_race: dict[str, list[tuple[str, int]]] = {}
        rows = csv.DictReader(_decode(f).splitlines())
        for row in rows:
            rid18 = (row.get("レースID(新)") or "").strip()
            try:
                pos = int(float(row.get("確定着順") or 0))
            except ValueError:
                pos = 0
            if len(rid18) < 16 or pos < 1:
                continue
            by_race.setdefault(rid18[:16], []).append((nfkc(row.get("馬名")), pos))
        for rid, runners in by_race.items():
            m = meta.get(rid)
            klass, surf = (m["klass"], m["surface"]) if m else (None, None)
            if m is None:
                n_noclass += 1
            if is_jump(klass, surf):
                continue
            races.append(Race(day, rid, klass, surf, runners))
    if races:
        log.info("kekka: %d レース (%s 〜 %s)、クラス不明 %d", len(races),
                 min(r.day for r in races), max(r.day for r in races), n_noclass)
    return races


def load_all() -> list[Race]:
    master = load_master()
    last = max((r.day for r in master), default=None)
    races = master + load_2026(last)
    races.sort(key=lambda r: (r.day, r.race_id))
    if not races:
        raise FileNotFoundError("着順データが 1 件もありません（data/kekka か master を確認）")
    return races


# ---------------------------------------------------------------- レーティング本体
def run_ratings(
    races: Iterable[Race],
    params: Params,
    eval_from: date | None = None,
    history: list[tuple] | None = None,
) -> tuple[dict[str, Horse], EvalAcc]:
    """着順を時系列に 1 パスで流し、多頭数 Elo を更新する。

    1 レースの更新（N 頭、着順 pos）:
        S_i = 先着した頭数 / (N−1)                     （同着は 0.5）
        E_i = Σ_j 1/(1+10^(−(R_i−R_j)/400)) / (N−1)
        R_i ← R_i + K(n_i) × sqrt(N−1) × (S_i − E_i)
    予測・検証には必ず更新前の R を使う。
    """
    horses: dict[str, Horse] = {}
    acc = EvalAcc()
    for race in races:
        rn = [(nm, p) for nm, p in race.runners if nm]
        n = len(rn)
        if n < 2:
            continue
        # 馬名再使用の判定 → 別馬として作り直す
        for nm, _ in rn:
            h = horses.get(nm)
            if h and h.last_day and (race.day - h.last_day).days > NAME_REUSE_GAP_DAYS:
                del horses[nm]
        est = [horses[nm].rating for nm, _ in rn
               if nm in horses and horses[nm].n >= ESTABLISHED_N]
        if len(est) >= 3:
            init = sum(est) / len(est)
        else:
            init = CLASS_INIT.get(race.klass or "", DEFAULT_INIT)
        known = [nm in horses and horses[nm].n >= 1 for nm, _ in rn]
        for nm, _ in rn:
            if nm not in horses:
                horses[nm] = Horse(rating=init, peak=init)
        pre = [horses[nm].rating for nm, _ in rn]
        pos = [p for _, p in rn]
        if eval_from and race.day >= eval_from and all(known):
            acc.add_race(pre, pos)
        scale = math.sqrt(n - 1)
        deltas: list[float] = []
        for i in range(n):
            s = e = 0.0
            for j in range(n):
                if i == j:
                    continue
                s += 1.0 if pos[i] < pos[j] else (0.5 if pos[i] == pos[j] else 0.0)
                e += expected(pre[i], pre[j])
            deltas.append(params.k(horses[rn[i][0]].n) * scale * (s - e) / (n - 1))
        for (nm, p), d, r0 in zip(rn, deltas, pre):
            h = horses[nm]
            h.rating = r0 + d
            h.n += 1
            h.last_day = race.day
            h.last_class = race.klass or h.last_class
            h.peak = max(h.peak, h.rating)
            if race.surface == "芝":
                h.n_turf += 1
            elif race.surface == "ダ":
                h.n_dirt += 1
            if history is not None:
                history.append((race.day.isoformat(), race.race_id, nm, race.klass or "",
                                race.surface or "", p, n, round(r0, 1), round(h.rating, 1)))
    return horses, acc


def scale_anchors(horses: dict[str, Horse], asof: date) -> tuple[float, float]:
    """(R_lo, R_hi)。R_hi = 現役最上位、R_lo = 現役の下位 1% 点。"""
    cut = asof - timedelta(days=ACTIVE_DAYS)
    act = sorted(h.rating for h in horses.values()
                 if h.last_day and h.last_day >= cut and h.n >= MIN_STARTS_FOR_SCALE)
    if len(act) < 20:
        act = sorted(h.rating for h in horses.values())
    lo = act[int(0.01 * (len(act) - 1))]
    hi = act[-1]
    if hi - lo < 1e-9:
        hi = lo + 1.0
    return lo, hi


def to_score(rating: float, lo: float, hi: float) -> float:
    return 100.0 * (rating - lo) / (hi - lo)


# ---------------------------------------------------------------- 出力の読み書き
def write_outputs(horses: dict[str, Horse], acc: EvalAcc, params: Params,
                  asof: date, eval_from: date | None, n_races: int,
                  history: list[tuple]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lo, hi = scale_anchors(horses, asof)
    cut = asof - timedelta(days=ACTIVE_DAYS)
    rows = sorted(horses.items(), key=lambda kv: -kv[1].rating)
    with (OUT_DIR / "ratings.csv").open("w", encoding="utf-8-sig", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["馬名", "score", "rating", "peak_score", "出走数", "芝", "ダ",
                    "最終出走", "最終クラス", "現役"])
        for nm, h in rows:
            w.writerow([nm, round(to_score(h.rating, lo, hi), 1), round(h.rating, 1),
                        round(to_score(h.peak, lo, hi), 1), h.n, h.n_turf, h.n_dirt,
                        h.last_day.isoformat() if h.last_day else "", h.last_class or "",
                        int(bool(h.last_day and h.last_day >= cut))])
    with (OUT_DIR / "history.csv").open("w", encoding="utf-8-sig", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["日付", "race_id", "馬名", "クラス", "芝ダ", "着順", "頭数", "R前", "R後"])
        w.writerows(history)
    meta = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "asof": asof.isoformat(),
        "n_races": n_races,
        "n_horses": len(horses),
        "params": params.__dict__,
        "anchors": {"rating_lo": round(lo, 2), "rating_hi": round(hi, 2)},
        "elo_per_point": round((hi - lo) / 100.0, 4),
        "eval_from": eval_from.isoformat() if eval_from else None,
        "eval": acc.summary(),
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    log.info("出力: %s (ratings.csv / history.csv / meta.json)", OUT_DIR.relative_to(BASE))


def read_outputs() -> tuple[dict[str, dict], dict]:
    p = OUT_DIR / "ratings.csv"
    if not p.exists():
        raise FileNotFoundError("先に `python power_rating.py build` を実行してください")
    with p.open(encoding="utf-8-sig", newline="") as fp:
        table = {r["馬名"]: r for r in csv.DictReader(fp)}
    meta = json.loads((OUT_DIR / "meta.json").read_text(encoding="utf-8"))
    return table, meta


def win_prob(score_a: float, score_b: float, elo_per_point: float) -> float:
    return expected(score_a * elo_per_point, score_b * elo_per_point)


def _w(s: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in "FWA" else 1 for ch in s)


def pad(s: object, width: int) -> str:
    t = str(s)
    return t + " " * max(0, width - _w(t))


def confidence(n: int) -> str:
    return "高" if n >= 8 else "中" if n >= 4 else "低"


# ---------------------------------------------------------------- コマンド
def default_eval_from(races: list[Race]) -> date:
    """master があれば test 期間（2024-01-01〜）、無ければ開催日の後ろ 30%。"""
    days = sorted({r.day for r in races})
    if days[0] <= date(2022, 12, 31):
        return date(2024, 1, 1)
    return days[int(len(days) * 0.7)]


def cmd_build(args: argparse.Namespace) -> int:
    races = load_all()
    params = Params(k0=args.k0, floor_ratio=args.floor_ratio, tau=args.tau)
    eval_from = (datetime.strptime(args.eval_from, "%Y-%m-%d").date()
                 if args.eval_from else default_eval_from(races))
    history: list[tuple] = []
    horses, acc = run_ratings(races, params, eval_from, history)
    asof = max(r.day for r in races)
    write_outputs(horses, acc, params, asof, eval_from, len(races), history)
    s = acc.summary()
    print(f"\n基準日 {asof} / {len(races):,} レース / {len(horses):,} 頭")
    print(f"前向き検証（{eval_from} 以降、全馬に出走歴のあるレースのみ・更新前の値で予測）")
    if s["races"]:
        print(f"  対象 {s['races']:,} レース")
        print(f"  スコア1位の勝率     {s['top_win_rate']:.1%}  CI95 {s['top_win_ci95']}"
              f"  (無作為 {s['random_win_rate']:.1%})")
        print(f"  スコア1位の3着内率  {s['top_in3_rate']:.1%}  CI95 {s['top_in3_ci95']}"
              f"  (無作為 {s['random_in3_rate']:.1%})")
        print(f"  2頭の先着を当てた率 {s['pair_concordance']:.1%}  logloss {s['pair_logloss']}"
              f"  ({s['pairs']:,} ペア)")
        print("  較正（強い側の予測先着率 → 実際）:")
        for b in s["calibration"]:
            print(f"    {b['pred']:.2f} → {b['actual']:.2f}  n={b['n']:,}")
    else:
        print("  検証対象レースなし（データ期間が短い）")
    return 0


def cmd_tune(args: argparse.Namespace) -> int:
    races = load_all()
    eval_from = default_eval_from(races)
    # master があるときは valid(2023) で選び、test(2024〜) は build の検証に残す
    upto = None
    if min(r.day for r in races) <= date(2022, 12, 31):
        eval_from, upto = date(2023, 1, 1), date(2023, 12, 31)
        races = [r for r in races if r.day <= upto]
    print(f"探索期間: {eval_from} 〜 {upto or max(r.day for r in races)}（ペア logloss 最小を選ぶ）")
    best: tuple[float, Params] | None = None
    for k0 in (12.0, 16.0, 24.0, 32.0, 48.0, 64.0, 96.0, 128.0):
        for fr in (0.25, 0.4, 0.6, 1.0):
            p = Params(k0=k0, floor_ratio=fr, tau=args.tau)
            _, acc = run_ratings(races, p, eval_from)
            s = acc.summary()
            if not s.get("pairs"):
                continue
            print(f"  k0={k0:>4} floor={fr:<4} logloss={s['pair_logloss']:.5f} "
                  f"concord={s['pair_concordance']:.4f} top3={s['top_in3_rate']:.4f}")
            if best is None or s["pair_logloss"] < best[0]:
                best = (s["pair_logloss"], p)
    if best:
        print(f"\n最良: --k0 {best[1].k0} --floor-ratio {best[1].floor_ratio}"
              f"  (logloss {best[0]:.5f})")
    return 0


def _resolve(table: dict[str, dict], name: str) -> str | None:
    q = nfkc(name)
    if q in table:
        return q
    cand = [k for k in table if q in k]
    if len(cand) == 1:
        return cand[0]
    if cand:
        print(f"  「{name}」は複数候補: {', '.join(cand[:8])}")
    return None


def cmd_compare(args: argparse.Namespace) -> int:
    table, meta = read_outputs()
    epp = float(meta["elo_per_point"])
    found: list[dict] = []
    for nm in args.names:
        key = _resolve(table, nm)
        if key is None:
            print(f"  「{nm}」は未収録（出走歴なし / 馬名違い）")
            continue
        found.append(table[key])
    if not found:
        return 1
    found.sort(key=lambda r: -float(r["score"]))
    print(f"\n強さスコア（基準日 {meta['asof']}、現役最上位=100）")
    print(f"{pad('馬名', 20)}{'score':>6} {'最高':>6} {'走':>3} 信頼 {pad('最終出走', 11)} クラス")
    for r in found:
        print(f"{pad(r['馬名'], 20)}{float(r['score']):>6.1f} {float(r['peak_score']):>6.1f} "
              f"{int(r['出走数']):>3}  {confidence(int(r['出走数']))}  "
              f"{pad(r['最終出走'], 11)} {r['最終クラス']}"
              f"{'' if r['現役'] == '1' else '  (1年以上出走なし)'}")
    if len(found) >= 2:
        print("\n同じレースで先着する確率（条件・斤量は平均的と仮定）")
        for i in range(len(found)):
            for j in range(i + 1, len(found)):
                a, b = found[i], found[j]
                p = win_prob(float(a["score"]), float(b["score"]), epp)
                print(f"  {a['馬名']} > {b['馬名']} : {p:.0%}"
                      f"  (差 {float(a['score']) - float(b['score']):+.1f} 点)")
    return 0


def cmd_top(args: argparse.Namespace) -> int:
    table, meta = read_outputs()
    rows = [r for r in table.values()
            if (args.all or r["現役"] == "1") and int(r["出走数"]) >= args.min_starts]
    if args.surface:
        col = "芝" if args.surface == "芝" else "ダ"
        rows = [r for r in rows if int(r[col]) * 2 > int(r["出走数"])]
    rows.sort(key=lambda r: -float(r["score"]))
    print(f"\n強さスコア上位（基準日 {meta['asof']}）")
    for i, r in enumerate(rows[: args.n], 1):
        print(f"{i:>3} {pad(r['馬名'], 20)}{float(r['score']):>6.1f} "
              f"{int(r['出走数']):>3}走 {pad(r['最終出走'], 11)} {r['最終クラス']}")
    return 0


def cmd_card(args: argparse.Namespace) -> int:
    table, meta = read_outputs()
    wk = WEEKLY_DIR / f"{args.date}.csv"
    if not wk.exists():
        print(f"出走表がありません: {wk}")
        return 1
    if meta["asof"] >= f"{args.date[:4]}-{args.date[4:6]}-{args.date[6:]}":
        print("※ このレーティングは当日以降の結果を含みます（回顧用。事前評価には使えません）")
    for race in read_weekly(wk):
        if is_jump(race["klass"], race["surface"]):
            continue
        if args.place and race["place"] != args.place:
            continue
        if args.rno and str(race["rno"]) != str(args.rno):
            continue
        print(f"\n{race['place']}{race['rno']}R {race['klass']} "
              f"{race['surface'] or ''}{race['dist']} {race['name']}")
        rows = []
        for ban, nm in race["horses"]:
            r = table.get(nm)
            rows.append((float(r["score"]) if r else None, ban, nm,
                         int(r["出走数"]) if r else 0))
        rows.sort(key=lambda x: (x[0] is None, -(x[0] or 0)))
        for sc, ban, nm, n in rows:
            s = f"{sc:>6.1f}" if sc is not None else "    --"
            print(f"  {ban:>2} {pad(nm, 20)}{s}  {n:>2}走 信頼{confidence(n) if n else '-'}")
    return 0


# ---------------------------------------------------------------- 検算
def make_mock_races(n_horses: int = 400, n_races: int = 3000, seed: int = 42
                    ) -> tuple[list[Race], dict[str, float]]:
    """真の強さが既知の合成データ。近い強さ同士が同じレースに出て、レースには
    その帯に応じたクラス名が付く（実データのクラス分けの模擬）。"""
    classes = ["未勝利", "1勝", "2勝", "3勝", "オープン"]
    rng = random.Random(seed)
    truth = {f"H{i:04d}": rng.gauss(0.0, 1.0) for i in range(n_horses)}
    ranked = sorted(truth, key=truth.get)
    races: list[Race] = []
    d0 = date(2020, 1, 4)
    for k in range(n_races):
        c = rng.randrange(n_horses)
        pool = ranked[max(0, c - 60): c + 60]
        field_ = rng.sample(pool, min(len(pool), rng.randint(8, 16)))
        perf = sorted(field_, key=lambda h: -(truth[h] + rng.gauss(0.0, 1.0)))
        klass = classes[min(c * len(classes) // n_horses, len(classes) - 1)]
        races.append(Race(d0 + timedelta(days=k // 24), f"M{k:06d}", klass, "芝",
                          [(h, i + 1) for i, h in enumerate(perf)]))
    return races, truth


def spearman(a: list[float], b: list[float]) -> float:
    def rank(v: list[float]) -> list[float]:
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for k, i in enumerate(o):
            r[i] = float(k)
        return r

    ra, rb = rank(a), rank(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = math.sqrt(sum((x - ma) ** 2 for x in ra))
    vb = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return cov / (va * vb)


def cmd_selftest(_: argparse.Namespace) -> int:
    races, truth = make_mock_races()
    horses, acc = run_ratings(races, Params(), races[len(races) * 7 // 10].day)
    names = [n for n in truth if n in horses and horses[n].n >= 5]
    rho = spearman([truth[n] for n in names], [horses[n].rating for n in names])
    s = acc.summary()
    # 真の強さを知っている場合の上限（同じ検証レースで真値の大小から先着を当てる）
    hit = tot = 0
    for r in races[len(races) * 7 // 10:]:
        for i, (a, pa) in enumerate(r.runners):
            for b, pb in r.runners[i + 1:]:
                tot += 1
                hit += int((truth[a] > truth[b]) == (pa < pb))
    oracle = hit / tot
    print(f"合成データ: {len(races)} レース / {len(names)} 頭")
    print(f"  真の強さとの順位相関 (Spearman) = {rho:.3f}   期待 > 0.85")
    print(f"  先着の的中率 = {s['pair_concordance']:.3f}   真値を知る上限 {oracle:.3f}、"
          f"期待 > 上限 − 0.03")
    ok = rho > 0.85 and s["pair_concordance"] > oracle - 0.03
    print("OK" if ok else "NG")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="強さスコア（レース横断レーティング、ローカル専用）")
    ap.add_argument("-v", "--verbose", action="store_true", help="DEBUG ログ")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("build", "tune"):
        b = sub.add_parser(name)
        b.add_argument("--k0", type=float, default=Params.k0)
        b.add_argument("--floor-ratio", type=float, default=Params.floor_ratio)
        b.add_argument("--tau", type=float, default=Params.tau)
        b.add_argument("--eval-from", default=None, help="検証開始日 YYYY-MM-DD")
    c = sub.add_parser("compare")
    c.add_argument("names", nargs="+")
    t = sub.add_parser("top")
    t.add_argument("--n", type=int, default=30)
    t.add_argument("--min-starts", type=int, default=3)
    t.add_argument("--surface", choices=["芝", "ダ"], default=None)
    t.add_argument("--all", action="store_true", help="1年以上出走のない馬も含める")
    k = sub.add_parser("card")
    k.add_argument("date", help="YYYYMMDD")
    k.add_argument("--place", default=None)
    k.add_argument("--rno", default=None)
    sub.add_parser("selftest")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # Windows コンソールの cp932 対策
        except (AttributeError, ValueError):
            pass
    cmd = {"build": cmd_build, "tune": cmd_tune, "compare": cmd_compare,
           "top": cmd_top, "card": cmd_card, "selftest": cmd_selftest}[args.cmd]
    try:
        return cmd(args)
    except (FileNotFoundError, KeyError) as e:
        log.error("%s", e)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
