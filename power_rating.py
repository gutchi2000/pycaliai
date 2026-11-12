# -*- coding: utf-8 -*-
"""
power_rating.py — 全馬共通ものさしの「強さスコア」（非公開・ローカル専用）
================================================================================
「この馬とこの馬、どっちが強い?」に 1 本の数字で答えるための、レース横断レーティング。
全レースの着順だけから多頭数 Elo を逐次更新し、現役最上位馬 = 100 の線形スケールに直す。
**芝とダートは別々のレーティング**（芝の現役最上位 = 100、ダートの現役最上位 = 100）。芝馬と
ダート馬は直接対戦がほぼ無く、1 本のものさしに載せると上下が決まらないため分けている。
芝スコアとダートスコアの数字同士は比べられない。

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
  python power_rating.py card 20261108               # その日の出走表を強さ順に + 好走気配/凡走注意
  python power_rating.py card 20261108 --out reports/power_rating/card_20261108.txt
  python power_rating.py chart クロワデュノール ダノンデサイル   # スコア推移の PNG（要 matplotlib）
  python power_rating.py site                        # ローカル閲覧用ページ（公開サイトには載らない）
  python power_rating.py tune                        # K の格子探索（master がある環境で）
  python power_rating.py ablate                      # 好走/凡走モデルの条件ごとの効き方を比較
  python power_rating.py selftest                    # 合成データで復元性を検算

スコアの読み方:
  score = 100 × (R − R_lo) / (R_hi − R_lo)   R_hi = 現役最上位, R_lo = 現役の下位 1% 点
  （R_hi・R_lo は芝・ダートそれぞれで決める）
  線形なので「点差」に意味がある。点差 d のとき、同じレースで先着する確率は
      P = 1 / (1 + 10^(−d × elo_per_point / 400))
  （elo_per_point は build 時に meta.json へ保存。compare が自動で換算する）

好走気配 / 凡走注意（card に表示）:
  各走の「上振れ」= 実績 S − 期待 E（スコアから見込まれる着順より上か下か）を、発走前に
  分かる条件（間隔・昇級/降級の段数・芝ダ替わり・距離変更・前走と近 3 走の上振れ・キャリア・
  乗り替わり・斤量増減・その馬自身の芝ダ/距離帯/競馬場での過去の上振れ）の区分ごとの平均で
  説明する ridge 回帰。予測上振れが学習期間の上位 15% なら「好走気配」、
  下位 15% なら「凡走注意」。3着内率はスコア（+ 上振れ補正）から Harville 式で計算する。
  build のたびに「検証期間で本当に上振れ/下振れしたか」を出すので、効いていなければ分かる。
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
# 登録抹消（引退）馬の一覧。power_rating_retired.py が JV-Link から作る（無ければ使わない）
RETIRED_CSV = OUT_DIR / "retired.csv"

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
SURFACES = ("芝", "ダ")     # レーティングを別々に持つ区分
CLASS_LEVEL: dict[str, int] = {
    "新馬": 0, "未勝利": 1, "500万": 2, "1勝": 2, "1000万": 3, "2勝": 3, "1600万": 4, "3勝": 4,
    "オープン": 5, "OP(L)": 5, "重賞": 6, "G3": 6, "G2": 7, "G1": 8,
}
FORM_RIDGE = 50.0           # 好走/凡走モデルの ridge 係数
FORM_Q = 0.15               # 上位/下位この割合にラベルを付ける
# 好走/凡走モデルが使う条件。既定は全条件（2026-11-12、master 込みで 2023 年を検証にして選定:
# 予測上振れと実績の相関 基本 0.162 → +hs 0.188 → 全条件 0.197。2024 年以降でも 0.163 → 0.198）。
# 効きの大半は hs（その芝ダでの実績）。jk は単独では効かない（±0）が害も無いので残す。
# `ablate --eval-from 2023-01-01 --until 20240101` で選び直せる
FORM_BASE = ("lay", "cls", "surf", "dist", "r1", "r3", "n")
FORM_EXTRA = ("jk", "kin", "hs", "hd", "hp")
FORM_DEFAULT = FORM_BASE + FORM_EXTRA


@dataclass
class Params:
    """更新パラメータ。K は出走数とともに k0 → k0×floor_ratio へ減衰する。

    既定値は master 込みで 2023 年のペア logloss 最小（2026-11-12 選定）。最適域は平坦で、
    k0=96〜160・tau=3〜6 の差は logloss 0.002 以内。
    """

    k0: float = 128.0
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
    dist: int | None = None
    place: str | None = None
    # 馬名 → (騎手キー, 斤量)。騎手キーは "n:名前"（出走表）か "c:コード"（master）
    extra: dict[str, tuple[str | None, float | None]] = field(default_factory=dict)


@dataclass
class Horse:
    rating: float
    n: int = 0
    last_day: date | None = None
    last_class: str | None = None
    peak: float = 0.0
    n_turf: int = 0
    n_dirt: int = 0
    last_surface: str | None = None
    last_dist: int | None = None
    resids: list[float] = field(default_factory=list)  # 直近 3 走の上振れ (S−E)、新しい順
    last_jockey: str | None = None
    last_kinryo: float | None = None
    # 条件別の上振れ累計: "s:芝" / "d:短" / "p:東京" → [合計, 走数]
    apt: dict[str, list[float]] = field(default_factory=dict)
    # 芝ダ別のレーティング: "芝" / "ダ" → [rating, 走数, 最高 rating, 最終出走日]
    # rating / peak フィールドは「最後に走った芝ダ」の値の写し
    sr: dict[str, list] = field(default_factory=dict)

    def main_surface(self) -> str | None:
        """走数の多いほう（同数なら最後に走ったほう）。"""
        if not self.sr:
            return None
        return max(self.sr, key=lambda k: (self.sr[k][1], k == self.last_surface))


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


def to_int(v: object) -> int | None:
    try:
        return int(float(str(v).strip()))
    except (TypeError, ValueError):
        return None


def to_float(v: object) -> float | None:
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return None


def dist_band(dist: int | None) -> str | None:
    if dist is None:
        return None
    return "短" if dist <= 1400 else "マ" if dist <= 1800 else "中" if dist <= 2200 else "長"


def apt_keys(surface: str | None, dist: int | None, place: str | None) -> list[str]:
    keys = []
    if surface in ("芝", "ダ"):
        keys.append(f"s:{surface}")
    if dist_band(dist):
        keys.append(f"d:{dist_band(dist)}")
    if place:
        keys.append(f"p:{place}")
    return keys


def _apt_bin(h: "Horse", key: str | None) -> str:
    if key is None:
        return "unk"
    v = h.apt.get(key)
    if not v or v[1] < 1:
        return "none"
    m = v[0] / v[1]
    return "good" if m > 0.1 else "bad" if m < -0.1 else "mid"


def pre_features(h: Horse, day: date, klass: str | None, surface: str | None,
                 dist: int | None, place: str | None = None, jockey: str | None = None,
                 kinryo: float | None = None) -> tuple[str, ...] | None:
    """発走前に分かる条件だけで作る区分特徴。出走歴の無い馬は None（判定しない）。"""
    if h.n < 1 or h.last_day is None or not h.resids:
        return None
    lay = (day - h.last_day).days
    lay_b = ("<=14" if lay <= 14 else "15-28" if lay <= 28 else "29-63" if lay <= 63
             else "64-119" if lay <= 119 else "120-239" if lay <= 239 else "240+")
    a, b = CLASS_LEVEL.get(h.last_class or ""), CLASS_LEVEL.get(klass or "")
    if a is None or b is None:
        cls_b = "unk"
    else:
        d = b - a
        cls_b = ("up2" if d >= 2 else "up1" if d == 1 else "same" if d == 0
                 else "down1" if d == -1 else "down2")
    if not surface or not h.last_surface:
        surf_b = "unk"
    elif surface == h.last_surface:
        surf_b = "same"
    else:
        surf_b = f"{h.last_surface}>{surface}"
    if dist is None or h.last_dist is None:
        dist_b = "unk"
    else:
        dd = dist - h.last_dist
        dist_b = "short" if dd <= -200 else "long" if dd >= 200 else "same"

    def rb(x: float) -> str:
        return "bad" if x < -0.15 else "good" if x > 0.15 else "mid"

    n_b = "1" if h.n == 1 else "2" if h.n == 2 else "3-5" if h.n <= 5 else "6-10" if h.n <= 10 else "11+"
    # 乗り替わり: 同じ種類のキー（名前同士 / コード同士）のときだけ比較する
    if jockey and h.last_jockey and jockey[:2] == h.last_jockey[:2]:
        jk_b = "same" if jockey == h.last_jockey else "change"
    else:
        jk_b = "unk"
    if kinryo is None or h.last_kinryo is None:
        kin_b = "unk"
    else:
        dk = kinryo - h.last_kinryo
        kin_b = "up" if dk >= 1.0 else "down" if dk <= -1.0 else "same"
    db = dist_band(dist)
    return (f"lay:{lay_b}", f"cls:{cls_b}", f"surf:{surf_b}", f"dist:{dist_b}",
            f"r1:{rb(h.resids[0])}", f"r3:{rb(sum(h.resids) / len(h.resids))}", f"n:{n_b}",
            f"jk:{jk_b}", f"kin:{kin_b}",
            f"hs:{_apt_bin(h, f's:{surface}' if surface in ('芝', 'ダ') else None)}",
            f"hd:{_apt_bin(h, f'd:{db}' if db else None)}",
            f"hp:{_apt_bin(h, f'p:{place}' if place else None)}")


# ---------------------------------------------------------------- 読み込み
def load_master(path: Path = MASTER) -> list[Race]:
    """master_v2 から平地の全着順を読む。無ければ空リスト。"""
    if not path.exists():
        log.info("master なし（%s）→ 2026 データのみで計算します", path.name)
        return []
    import pandas as pd  # 遅延 import（compare/top は pandas 不要）

    cols = ["日付", "レースID(新/馬番無)", "馬名", "クラス名", "トラックコード(JV)", "着順", "芝・ダ",
            "距離", "場所", "騎手", "騎手コード", "斤量"]
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
        names = list(g["馬名"].map(nfkc))
        if "騎手" in g:
            jks = [f"n:{nfkc(v)}" if nfkc(v) else None for v in g["騎手"]]
        elif "騎手コード" in g:
            jks = [f"c:{v}" if str(v) not in ("", "nan") else None for v in g["騎手コード"]]
        else:
            jks = [None] * len(names)
        kins = [to_float(v) for v in g["斤量"]] if "斤量" in g else [None] * len(names)
        races.append(Race(g["day"].iloc[0], str(rid), klass, surf,
                          list(zip(names, g["pos"])),
                          to_int(g["距離"].iloc[0]) if "距離" in g else None,
                          nfkc(g["場所"].iloc[0]) if "場所" in g else None,
                          {nm: (jk, kn) for nm, jk, kn in zip(names, jks, kins)}))
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
    """週次出走表 CSV → [{race_id, klass, surface, place, rno, name,
    horses:[(馬番, 馬名, 騎手キー, 斤量)]}]。同名列（前走の騎手・斤量）は先頭＝今走を使う。"""
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
                jk = nfkc(c[hcols.index("騎手")]) if "騎手" in hcols else ""
                kn = to_float(c[hcols.index("斤量")]) if "斤量" in hcols else None
                cur["horses"].append((int(c[i_ban]), nfkc(c[i_name]),
                                      f"n:{jk}" if jk else None, kn))
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
        place_of: dict[str, str] = {}
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
            place_of[rid18[:16]] = nfkc(row.get("場所"))
        for rid, runners in by_race.items():
            m = meta.get(rid)
            klass, surf = (m["klass"], m["surface"]) if m else (None, None)
            dist = to_int(m["dist"]) if m else None
            if m is None:
                n_noclass += 1
            if is_jump(klass, surf):
                continue
            extra = {nm: (jk, kn) for _, nm, jk, kn in m["horses"]} if m else {}
            races.append(Race(day, rid, klass, surf, runners, dist,
                              place_of.get(rid) or None, extra))
    if races:
        log.info("kekka: %d レース (%s 〜 %s)、クラス不明 %d", len(races),
                 min(r.day for r in races), max(r.day for r in races), n_noclass)
    return races


def load_retired(path: Path | None = None) -> dict[str, date | None]:
    """登録抹消馬 {馬名: 抹消日（不明なら None）}。ファイルが無ければ空。"""
    path = path or RETIRED_CSV
    if not path.exists():
        log.info("抹消馬リストなし（%s）→ 引退馬も 1 年以内に走っていれば現役扱いになります。"
                 "py -3.12-32 power_rating_retired.py で作れます", path.name)
        return {}
    out: dict[str, date | None] = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            try:
                d = datetime.strptime(r.get("抹消日") or "", "%Y%m%d").date()
            except ValueError:
                d = None
            out[nfkc(r.get("馬名"))] = d
    log.info("抹消馬リスト: %d 頭", len(out))
    return out


def retired_on(retired: dict[str, date | None], name: str, h: "Horse", asof: date) -> date | None:
    """その馬が基準日時点で抹消済みなら抹消日（日付不明は基準日）を返す。現役なら None。

    抹消日がその馬の最終出走より前なら同名の別馬（馬名の再使用）なので現役扱い。
    抹消日が基準日より後（--until で過去の状態を作るとき）も現役扱い。
    """
    if name not in retired:
        return None
    d = retired[name]
    if d is None:
        return asof
    if d > asof or (h.last_day and d < h.last_day):
        return None
    return d


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
    """着順を時系列に 1 パスで流し、多頭数 Elo を更新する。芝とダートは別のレーティング。

    1 レースの更新（N 頭、着順 pos。R はそのレースの芝ダのレーティング）:
        S_i = 先着した頭数 / (N−1)                     （同着は 0.5）
        E_i = Σ_j 1/(1+10^(−(R_i−R_j)/400)) / (N−1)
        R_i ← R_i + K(n_i) × sqrt(N−1) × (S_i − E_i)   n_i はその芝ダでの走数
    予測・検証には必ず更新前の R を使う。芝ダが不明のレースは使わない。
    間隔・前走クラスなどの条件特徴は芝ダをまたいだ通算の状態から作る。
    """
    horses: dict[str, Horse] = {}
    acc = EvalAcc()
    n_skip = 0
    for race in races:
        rn = [(nm, p) for nm, p in race.runners if nm]
        n = len(rn)
        if n < 2:
            continue
        sf = race.surface
        if sf not in SURFACES:
            n_skip += 1
            continue
        # 馬名再使用の判定 → 別馬として作り直す
        for nm, _ in rn:
            h = horses.get(nm)
            if h and h.last_day and (race.day - h.last_day).days > NAME_REUSE_GAP_DAYS:
                del horses[nm]
        est = [horses[nm].sr[sf][0] for nm, _ in rn
               if nm in horses and sf in horses[nm].sr and horses[nm].sr[sf][1] >= ESTABLISHED_N]
        if len(est) >= 3:
            init = sum(est) / len(est)
        else:
            init = CLASS_INIT.get(race.klass or "", DEFAULT_INIT)
        known = [nm in horses and sf in horses[nm].sr and horses[nm].sr[sf][1] >= 1
                 for nm, _ in rn]
        for nm, _ in rn:
            if nm not in horses:
                horses[nm] = Horse(rating=init, peak=init)
            if sf not in horses[nm].sr:
                horses[nm].sr[sf] = [init, 0, init, None]
        pre = [horses[nm].sr[sf][0] for nm, _ in rn]
        pos = [p for _, p in rn]
        feats = [pre_features(horses[nm], race.day, race.klass, race.surface, race.dist,
                              race.place, *race.extra.get(nm, (None, None)))
                 for nm, _ in rn]
        akeys = apt_keys(race.surface, race.dist, race.place)
        if eval_from and race.day >= eval_from and all(known):
            acc.add_race(pre, pos)
        scale = math.sqrt(n - 1)
        deltas: list[float] = []
        resids: list[float] = []
        for i in range(n):
            s = e = 0.0
            for j in range(n):
                if i == j:
                    continue
                s += 1.0 if pos[i] < pos[j] else (0.5 if pos[i] == pos[j] else 0.0)
                e += expected(pre[i], pre[j])
            resids.append((s - e) / (n - 1))
            deltas.append(params.k(horses[rn[i][0]].sr[sf][1]) * scale * resids[-1])
        for (nm, p), d, r0, rs, ft in zip(rn, deltas, pre, resids, feats):
            h = horses[nm]
            st = h.sr[sf]
            st[0] = r0 + d
            st[1] += 1
            st[2] = max(st[2], st[0])
            st[3] = race.day
            h.rating, h.peak = st[0], st[2]
            h.n += 1
            h.last_day = race.day
            h.last_class = race.klass or h.last_class
            if sf == "芝":
                h.n_turf += 1
            else:
                h.n_dirt += 1
            h.last_surface = sf
            h.last_dist = race.dist or h.last_dist
            h.resids = [rs] + h.resids[:2]
            jk, kn = race.extra.get(nm, (None, None))
            h.last_jockey = jk or h.last_jockey
            h.last_kinryo = kn if kn is not None else h.last_kinryo
            for ak in akeys:
                v = h.apt.setdefault(ak, [0.0, 0])
                v[0] += rs
                v[1] += 1
            if history is not None:
                history.append((race.day.isoformat(), race.race_id, nm, race.klass or "",
                                sf, p, n, round(r0, 1), round(st[0], 1),
                                round(rs, 4), "|".join(ft) if ft else ""))
    if n_skip:
        log.info("芝ダ不明のため使わなかったレース: %d", n_skip)
    return horses, acc


def scale_anchors(horses: dict[str, Horse], asof: date, surface: str,
                  retired: dict[str, date | None] | None = None) -> tuple[float, float]:
    """その芝ダの (R_lo, R_hi)。R_hi = 現役最上位、R_lo = 現役の下位 1% 点。

    現役 = その芝ダで基準日から 1 年以内に出走し、その芝ダで 3 走以上、登録抹消されていない。
    """
    cut = asof - timedelta(days=ACTIVE_DAYS)
    states = [h.sr[surface] for nm, h in horses.items() if surface in h.sr
              and not (retired and retired_on(retired, nm, h, asof))]
    act = sorted(st[0] for st in states
                 if st[3] and st[3] >= cut and st[1] >= MIN_STARTS_FOR_SCALE)
    if len(act) < 20:
        act = sorted(st[0] for st in states)
    if not act:
        return DEFAULT_INIT - 300.0, DEFAULT_INIT + 300.0
    lo = act[int(0.01 * (len(act) - 1))]
    hi = act[-1]
    if hi - lo < 1e-9:
        hi = lo + 1.0
    return lo, hi


def to_score(rating: float, lo: float, hi: float) -> float:
    return 100.0 * (rating - lo) / (hi - lo)


# ---------------------------------------------------------------- 好走気配 / 凡走注意
def fit_form(rows: list[tuple[tuple[str, ...], float]],
             prefixes: tuple[str, ...] | None = None) -> dict:
    """区分特徴 → 上振れ (S−E) の ridge 回帰。rows = [(特徴, 上振れ)]。

    prefixes を渡すと、その条件（"lay" "cls" …）だけを使う。予測側は係数に無いキーを
    0 として扱うので、ここで絞れば学習・予測の両方が絞られる。
    """
    import numpy as np

    if prefixes is not None:
        rows = [(tuple(k for k in ft if k.split(":", 1)[0] in prefixes), y) for ft, y in rows]
    keys = sorted({k for ft, _ in rows for k in ft})
    idx = {k: i + 1 for i, k in enumerate(keys)}
    x = np.zeros((len(rows), len(keys) + 1))
    x[:, 0] = 1.0
    for r, (ft, _) in enumerate(rows):
        for k in ft:
            x[r, idx[k]] = 1.0
    y = np.array([v for _, v in rows])
    reg = np.eye(x.shape[1]) * FORM_RIDGE
    reg[0, 0] = 0.0
    w = np.linalg.solve(x.T @ x + reg, x.T @ y)
    pred = x @ w
    # 理由表示用: 条件ごとに「学習データの平均的な馬」からの差に直した係数。生の係数は区分の
    # 基準の取り方で全体が上下にずれる（例: hs:none が大きく負 → 他の hs 区分が全部正に見える）
    freq = x.mean(axis=0)
    base: dict[str, float] = {}
    for k, i in idx.items():
        p = k.split(":", 1)[0]
        base[p] = base.get(p, 0.0) + float(w[i]) * float(freq[i])
    return {
        "intercept": float(w[0]),
        "coef": {k: round(float(w[i]), 5) for k, i in idx.items()},
        "coef_c": {k: round(float(w[i]) - base[k.split(":", 1)[0]], 5) for k, i in idx.items()},
        "q_lo": float(np.quantile(pred, FORM_Q)),
        "q_hi": float(np.quantile(pred, 1 - FORM_Q)),
        "n_train": len(rows),
        "features": list(prefixes) if prefixes is not None else None,
    }


def form_pred(model: dict, ft: tuple[str, ...] | None) -> float | None:
    if ft is None:
        return None
    return model["intercept"] + sum(model["coef"].get(k, 0.0) for k in ft)


def form_label(model: dict, pred: float | None) -> str:
    if pred is None:
        return ""
    return "好走気配" if pred >= model["q_hi"] else "凡走注意" if pred <= model["q_lo"] else ""


_FACTOR_JP = {
    "lay:<=14": "連闘〜中1週", "lay:15-28": "中2〜3週", "lay:29-63": "中4〜8週",
    "lay:64-119": "2〜4か月ぶり", "lay:120-239": "4〜8か月ぶり", "lay:240+": "8か月以上ぶり",
    "cls:up1": "昇級", "cls:up2": "2段以上の昇級", "cls:down1": "降級",
    "cls:down2": "2段以上の降級", "cls:same": "同クラス",
    "jk:same": "継続騎乗", "jk:change": "乗り替わり",
    "kin:up": "斤量増", "kin:down": "斤量減", "kin:same": "斤量据え置き",
    # hs:bad は係数が正（下振れ続きでスコアが下がり過ぎ → 戻す方向）なので、その向きで書く
    "hs:good": "この芝ダで上振れ実績", "hs:bad": "この芝ダで下振れ続きの反動",
    "hs:mid": "この芝ダで実績相応", "hs:none": "この芝ダは初",
    "hd:good": "この距離帯で上振れ実績", "hd:bad": "この距離帯で下振れ実績",
    "hd:mid": "この距離帯で実績相応", "hd:none": "この距離帯は初",
    "hp:good": "この競馬場で上振れ実績", "hp:bad": "この競馬場で下振れ実績",
    "hp:mid": "この競馬場で実績相応", "hp:none": "この競馬場は初",
    "surf:same": "同じ芝ダ", "surf:芝>ダ": "芝→ダ", "surf:ダ>芝": "ダ→芝",
    "dist:short": "距離短縮", "dist:long": "距離延長", "dist:same": "同距離帯",
    "r1:good": "前走上振れ", "r1:bad": "前走下振れ", "r1:mid": "前走相応",
    "r3:good": "近3走上振れ", "r3:bad": "近3走下振れ", "r3:mid": "近3走相応",
    "n:1": "キャリア1戦", "n:2": "キャリア2戦", "n:3-5": "キャリア3〜5戦",
    "n:6-10": "キャリア6〜10戦", "n:11+": "キャリア11戦以上",
}


def form_reason(model: dict, ft: tuple[str, ...] | None, label: str) -> str:
    """ラベルの向きに効いている条件を係数の大きい順に 2 つまで。"""
    if not ft or not label:
        return ""
    sign = 1.0 if label == "好走気配" else -1.0
    coef = model.get("coef_c") or model["coef"]
    c = sorted(((sign * coef.get(k, 0.0), k) for k in ft), reverse=True)
    why = [_FACTOR_JP.get(k, k) for v, k in c[:2] if v >= 0.008]
    return f"（{'・'.join(why)}）" if why else ""


def top3_probs(ratings: list[float], temp: float = 1.0) -> list[float]:
    """Harville 式の 3 着内率（O(N²)）。強さ = 10^(R/(400×temp))。

    P(i が 3 着内) = p_i + Σ_j p_j·p_i/(1−p_j) + Σ_j Σ_k p_j·p_k/(1−p_j)·p_i/(1−p_j−p_k)
    """
    n = len(ratings)
    if n <= 3:
        return [1.0] * n
    m = max(ratings)
    st = [10.0 ** ((r - m) / (ELO_SCALE * temp)) for r in ratings]
    tot = sum(st)
    p = [v / tot for v in st]
    eps = 1e-12
    # t[j] = Σ_{k≠j} p_k / (1 − p_j − p_k)
    t = [sum(p[k] / max(1.0 - p[j] - p[k], eps) for k in range(n) if k != j) for j in range(n)]
    out = []
    for i in range(n):
        p2 = p3 = 0.0
        for j in range(n):
            if j == i:
                continue
            a_j = p[j] / max(1.0 - p[j], eps)
            p2 += a_j * p[i]
            p3 += a_j * p[i] * (t[j] - p[i] / max(1.0 - p[j] - p[i], eps))
        out.append(min(1.0, p[i] + p2 + p3))
    return out


def _hist_races(history: list[tuple]) -> Iterator[list[tuple]]:
    cur: list[tuple] = []
    for row in history:
        if cur and row[1] != cur[0][1]:
            yield cur
            cur = []
        cur.append(row)
    if cur:
        yield cur


def _race_metrics(races: list[list[tuple]], model: dict | None, gamma: float, temp: float,
                  want: str = "both") -> dict:
    """ペア logloss と 3 着内 logloss（補正 gamma・温度 temp を掛けた場合）。"""
    pl = pn = tl = tn = 0.0
    for race in races:
        adj = []
        for row in race:
            ft = tuple(row[10].split("|")) if row[10] else None
            fp = form_pred(model, ft) if model else None
            adj.append(row[7] + gamma * (fp or 0.0))
        pos = [row[5] for row in race]
        if want in ("both", "pair"):
            for i in range(len(race)):
                for j in range(i + 1, len(race)):
                    if pos[i] == pos[j]:
                        continue
                    q = min(max(expected(adj[i], adj[j]), 1e-6), 1 - 1e-6)
                    pl -= math.log(q if pos[i] < pos[j] else 1 - q)
                    pn += 1
        if want in ("both", "top3"):
            for q, ps in zip(top3_probs(adj, temp), pos):
                q = min(max(q, 1e-6), 1 - 1e-6)
                tl -= math.log(q if ps <= 3 else 1 - q)
                tn += 1
    return {"pair_logloss": pl / max(pn, 1), "top3_logloss": tl / max(tn, 1)}


def _cluster_ci(per_race: list[tuple[float, int]], reps: int = 300, seed: int = 42
                ) -> list[float]:
    """レース単位の再標本化による平均の 95% CI。per_race = [(合計, 頭数)]。"""
    rng = random.Random(seed)
    m = len(per_race)
    if m < 2:
        return [float("nan"), float("nan")]
    means = []
    for _ in range(reps):
        sm = cn = 0.0
        for _ in range(m):
            a, b = per_race[rng.randrange(m)]
            sm += a
            cn += b
        means.append(sm / max(cn, 1))
    means.sort()
    return [round(means[int(0.025 * (reps - 1))], 4), round(means[int(0.975 * (reps - 1))], 4)]


def build_form(history: list[tuple], eval_from: date,
               prefixes: tuple[str, ...] = FORM_DEFAULT) -> tuple[dict, dict]:
    """学習期間（eval_from より前）で fit → 検証期間で前向きに確認 → 全期間で fit し直して返す。

    学習・ラベル検証は「条件特徴のある馬（=出走歴のある馬）」すべて。logloss は全レースで、
    特徴の無い馬は補正 0 として扱う。

    Returns:
        (card が使うモデル, 検証結果)
    """
    cut = eval_from.isoformat()
    races = [r for r in _hist_races(history) if len(r) >= 2]
    tr = [r for r in races if r[0][0] < cut]
    te = [r for r in races if r[0][0] >= cut]

    def flat(rs: list[list[tuple]]) -> list[tuple[tuple[str, ...], float]]:
        return [(tuple(row[10].split("|")), row[9]) for r in rs for row in r if row[10]]

    if len(flat(tr)) < 500 or len(flat(te)) < 300:
        log.warning("好走/凡走モデル: データ不足（学習 %d / 検証 %d 頭）",
                    len(flat(tr)), len(flat(te)))
        return {}, {"note": "insufficient data"}
    m = fit_form(flat(tr), prefixes)
    # 3 着内率の温度 temp と補正の強さ gamma は学習期間だけで選ぶ（gamma=0 は「補正しない」）。
    # card の 3 着内率に使うので、選ぶ基準は 3 着内 logloss
    temp = min((0.4, 0.5, 0.65, 0.8, 1.0, 1.25, 1.5, 2.0),
               key=lambda t: _race_metrics(tr, None, 0.0, t, "top3")["top3_logloss"])
    gamma = min((0.0, 200.0, 400.0, 800.0, 1200.0, 1600.0),
                key=lambda g: _race_metrics(tr, m, g, temp, "top3")["top3_logloss"])
    base = _race_metrics(te, None, 0.0, temp)
    adj = _race_metrics(te, m, gamma, temp)
    # ラベル別に「実際に上振れしたか」と 3 着内率（スコアだけの見込み vs 実際）
    labs = ("好走気配", "", "凡走注意")
    tot = {k: [0, 0.0, 0.0, 0] for k in labs}            # 頭数, 上振れ, 見込み3着内, 実際3着内
    per_race: dict[str, list[tuple[float, int]]] = {k: [] for k in labs}
    for race in te:
        exp3 = top3_probs([row[7] for row in race], temp)
        acc_r = {k: [0.0, 0] for k in labs}
        for row, e3 in zip(race, exp3):
            if not row[10]:
                continue
            lab = form_label(m, form_pred(m, tuple(row[10].split("|"))))
            t = tot[lab]
            t[0] += 1
            t[1] += row[9]
            t[2] += e3
            t[3] += int(row[5] <= 3)
            acc_r[lab][0] += row[9]
            acc_r[lab][1] += 1
        for k in labs:
            if acc_r[k][1]:
                per_race[k].append((acc_r[k][0], acc_r[k][1]))
    ev: dict = {
        "features": list(prefixes),
        "train_races": len(tr), "test_races": len(te),
        "train_rows": len(flat(tr)), "test_rows": len(flat(te)),
        "gamma": gamma, "temp": temp,
        "test_pair_logloss_base": round(base["pair_logloss"], 5),
        "test_pair_logloss_adj": round(adj["pair_logloss"], 5),
        "test_top3_logloss_base": round(base["top3_logloss"], 5),
        "test_top3_logloss_adj": round(adj["top3_logloss"], 5),
        "labels": {},
    }
    for lab in labs:
        n = tot[lab][0]
        if not n:
            continue
        ev["labels"][lab or "無印"] = {
            "n": n,
            "mean_resid": round(tot[lab][1] / n, 4),
            "resid_ci95": _cluster_ci(per_race[lab]),
            "top3_expected": round(tot[lab][2] / n, 4),
            "top3_actual": round(tot[lab][3] / n, 4),
        }
    final = fit_form(flat(races), prefixes)
    final.update({"gamma": gamma, "temp": temp})
    return final, ev


# ---------------------------------------------------------------- 出力の読み書き
def write_outputs(horses: dict[str, Horse], acc: EvalAcc, params: Params,
                  asof: date, eval_from: date | None, n_races: int,
                  history: list[tuple], form: dict | None = None) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    retired = load_retired()
    anc = {sf: scale_anchors(horses, asof, sf, retired) for sf in SURFACES}
    cut = asof - timedelta(days=ACTIVE_DAYS)

    def main_score(h: Horse) -> float:
        m = h.main_surface()
        return to_score(h.sr[m][0], *anc[m]) if m else -1e9

    rows = sorted(horses.items(), key=lambda kv: -main_score(kv[1]))
    with (OUT_DIR / "ratings.csv").open("w", encoding="utf-8-sig", newline="") as fp:
        w = csv.writer(fp)
        head = ["馬名", "主戦"]
        for sf in SURFACES:
            head += [f"score_{sf}", f"rating_{sf}", f"peak_{sf}", f"走_{sf}", f"最終_{sf}",
                     f"現役_{sf}"]
        w.writerow(head + ["出走数", "最終出走", "最終クラス", "最終芝ダ", "最終距離",
                           "近走上振れ", "最終騎手", "最終斤量", "適性", "抹消日"])
        n_ret = 0
        for nm, h in rows:
            ret = retired_on(retired, nm, h, asof)
            n_ret += int(bool(ret and h.last_day and h.last_day >= cut))
            row: list = [nm, h.main_surface() or ""]
            for sf in SURFACES:
                st = h.sr.get(sf)
                if st and st[1]:
                    row += [round(to_score(st[0], *anc[sf]), 1), round(st[0], 1),
                            round(to_score(st[2], *anc[sf]), 1), st[1],
                            st[3].isoformat() if st[3] else "",
                            int(bool(st[3] and st[3] >= cut and not ret))]
                else:
                    row += ["", "", "", 0, "", 0]
            w.writerow(row + [
                h.n, h.last_day.isoformat() if h.last_day else "", h.last_class or "",
                h.last_surface or "", h.last_dist or "",
                "/".join(f"{x:.4f}" for x in h.resids),
                h.last_jockey or "", h.last_kinryo if h.last_kinryo is not None else "",
                json.dumps({k: [round(v[0], 4), v[1]] for k, v in h.apt.items()},
                           ensure_ascii=False, separators=(",", ":")),
                ret.isoformat() if ret else ""])
    if retired:
        log.info("1 年以内に出走があるが登録抹消済みのため現役から外した馬: %d 頭", n_ret)
    with (OUT_DIR / "history.csv").open("w", encoding="utf-8-sig", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["日付", "race_id", "馬名", "クラス", "芝ダ", "着順", "頭数", "R前", "R後",
                    "上振れ", "条件"])
        w.writerows(history)
    meta = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "asof": asof.isoformat(),
        "n_races": n_races,
        "n_horses": len(horses),
        "params": params.__dict__,
        "anchors": {sf: {"rating_lo": round(lo, 2), "rating_hi": round(hi, 2),
                         "elo_per_point": round((hi - lo) / 100.0, 4)}
                    for sf, (lo, hi) in anc.items()},
        "eval_from": eval_from.isoformat() if eval_from else None,
        "eval": acc.summary(),
        "form": form,
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
    if table and "主戦" not in next(iter(table.values())):
        raise KeyError("ratings.csv が旧形式（芝ダ共通）です。`python power_rating.py build` で作り直してください")
    return table, meta


def surf_row(r: dict, surface: str) -> dict | None:
    """ratings.csv の 1 行から、その芝ダの値を取り出す。走っていなければ None。"""
    n = int(r.get(f"走_{surface}") or 0)
    if n < 1:
        return None
    return {"score": float(r[f"score_{surface}"]), "rating": float(r[f"rating_{surface}"]),
            "peak": float(r[f"peak_{surface}"]), "n": n, "last": r[f"最終_{surface}"],
            "active": r[f"現役_{surface}"] == "1", "retired": r.get("抹消日") or ""}


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
    if args.until:
        until = datetime.strptime(args.until, "%Y%m%d").date()
        races = [r for r in races if r.day < until]
        log.info("--until %s: %d レースまでで打ち切り", args.until, len(races))
    params = Params(k0=args.k0, floor_ratio=args.floor_ratio, tau=args.tau)
    eval_from = (datetime.strptime(args.eval_from, "%Y-%m-%d").date()
                 if args.eval_from else default_eval_from(races))
    history: list[tuple] = []
    horses, acc = run_ratings(races, params, eval_from, history)
    asof = max(r.day for r in races)
    prefixes = tuple(x for x in args.form_features.split(",") if x)
    bad = sorted(set(prefixes) - set(FORM_BASE + FORM_EXTRA))
    if bad:
        raise KeyError(f"--form-features に未知の条件: {bad}（使えるのは {FORM_BASE + FORM_EXTRA}）")
    form_model, form_ev = build_form(history, eval_from, prefixes)
    write_outputs(horses, acc, params, asof, eval_from, len(races), history, form_ev)
    (OUT_DIR / "form_model.json").write_text(
        json.dumps(form_model, ensure_ascii=False, indent=2), encoding="utf-8")
    s = acc.summary()
    print(f"\n基準日 {asof} / {len(races):,} レース / {len(horses):,} 頭（芝・ダートは別レーティング）")
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
    if form_ev.get("labels"):
        print(f"\n好走気配/凡走注意の前向き検証（学習 {form_ev['train_rows']:,} 頭 / "
              f"検証 {form_ev['test_rows']:,} 頭・{form_ev['test_races']:,} レース）")
        print("  ラベル      頭数   上振れ平均 [レース単位 CI95]   3着内: 見込み → 実際")
        for lab in ("好走気配", "無印", "凡走注意"):
            v = form_ev["labels"].get(lab)
            if v:
                print(f"  {pad(lab, 10)}{v['n']:>6,}  {v['mean_resid']:+.4f} {v['resid_ci95']}"
                      f"   {v['top3_expected']:.1%} → {v['top3_actual']:.1%}")
        print(f"  ペア logloss  補正なし {form_ev['test_pair_logloss_base']} → "
              f"補正あり {form_ev['test_pair_logloss_adj']}  (gamma={form_ev['gamma']})")
        print(f"  3着内 logloss 補正なし {form_ev['test_top3_logloss_base']} → "
              f"補正あり {form_ev['test_top3_logloss_adj']}  (temp={form_ev['temp']})")
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
    for k0 in (32.0, 48.0, 64.0, 96.0, 128.0, 160.0, 200.0, 256.0):
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


def cmd_ablate(args: argparse.Namespace) -> int:
    """好走/凡走モデルの条件を 1 つずつ足して、検証期間での効き方を比べる。"""
    races = load_all()
    if args.until:   # 条件選び用: --eval-from 2023-01-01 --until 20240101 で 2023 年だけを検証に使う
        until = datetime.strptime(args.until, "%Y%m%d").date()
        races = [r for r in races if r.day < until]
    eval_from = (datetime.strptime(args.eval_from, "%Y-%m-%d").date()
                 if args.eval_from else default_eval_from(races))
    history: list[tuple] = []
    run_ratings(races, Params(k0=args.k0, floor_ratio=args.floor_ratio, tau=args.tau),
                None, history)
    cut = eval_from.isoformat()
    rows = [(tuple(r[10].split("|")), r[9], r[0]) for r in history if r[10]]
    tr = [(a, b) for a, b, d in rows if d < cut]
    te = [(a, b) for a, b, d in rows if d >= cut]
    print(f"学習 {len(tr):,} 頭 / 検証 {len(te):,} 頭（{eval_from} 以降）")
    print(f"{pad('条件セット', 26)}  相関    好走気配の上振れ   凡走注意の上振れ")
    sets = [("基本7条件", FORM_BASE)]
    sets += [(f"基本+{x}", FORM_BASE + (x,)) for x in FORM_EXTRA]
    sets += [("既定", FORM_DEFAULT), ("全条件", FORM_BASE + FORM_EXTRA)]
    for name, pf in sets:
        m = fit_form(tr, pf)
        ps = [form_pred(m, a) or 0.0 for a, _ in te]
        ys = [b for _, b in te]
        mp, my = sum(ps) / len(ps), sum(ys) / len(ys)
        cov = sum((x - mp) * (y - my) for x, y in zip(ps, ys))
        den = math.sqrt(sum((x - mp) ** 2 for x in ps) * sum((y - my) ** 2 for y in ys))
        hi = [y for x, y in zip(ps, ys) if x >= m["q_hi"]]
        lo = [y for x, y in zip(ps, ys) if x <= m["q_lo"]]
        print(f"{pad(name, 26)}  {cov / den if den else 0.0:.4f}  "
              f"{sum(hi) / max(len(hi), 1):+.4f} (n={len(hi):,})  "
              f"{sum(lo) / max(len(lo), 1):+.4f} (n={len(lo):,})")
    return 0


def cmd_chart(args: argparse.Namespace) -> int:
    """history.csv からスコア推移の折れ線を PNG に保存する。"""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib がありません: pip install matplotlib")
        return 2
    table, meta = read_outputs()
    names = [k for k in (_resolve(table, nm) for nm in args.names) if k]
    if not names:
        print("該当する馬がいません")
        return 1
    surface = args.surface or table[names[0]]["主戦"]
    lo = float(meta["anchors"][surface]["rating_lo"])
    hi = float(meta["anchors"][surface]["rating_hi"])
    series: dict[str, list[tuple[date, float]]] = {n: [] for n in names}
    with (OUT_DIR / "history.csv").open(encoding="utf-8-sig", newline="") as fp:
        for r in csv.DictReader(fp):
            if r["馬名"] in series and r["芝ダ"] == surface:
                series[r["馬名"]].append((datetime.strptime(r["日付"], "%Y-%m-%d").date(),
                                          to_score(float(r["R後"]), lo, hi)))
    if not any(series.values()):
        print(f"{surface}での出走がありません（--surface で切り替え）")
        return 1
    # 日本語フォント: Windows は Yu Gothic / Meiryo、Linux は IPAGothic / Noto を順に試す
    plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "IPAGothic",
                                   "Noto Sans CJK JP", "sans-serif"]
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    fig, ax = plt.subplots(figsize=(9, 5))
    for nm in names:
        if series[nm]:
            xs, ys = zip(*series[nm])
            ax.plot(xs, ys, marker="o", label=nm)
    sname = "芝" if surface == "芝" else "ダート"
    ax.set_ylabel(f"{sname}の強さスコア（現在のものさし）")
    ax.set_title(f"{sname}の強さスコアの推移（基準日 {meta['asof']}）")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    out = Path(args.out) if args.out else OUT_DIR / f"chart_{surface}_{'_'.join(names)[:60]}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"保存: {out}")
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
    found: list[dict] = []
    for nm in args.names:
        key = _resolve(table, nm)
        if key is None:
            print(f"  「{nm}」は未収録（出走歴なし / 馬名違い）")
            continue
        found.append(table[key])
    if not found:
        return 1
    print(f"\n強さスコア（基準日 {meta['asof']}。芝・ダートは別のものさしで、数字同士は比べられません）")
    for sf in SURFACES:
        rows = [(r, surf_row(r, sf)) for r in found]
        rows = sorted([x for x in rows if x[1]], key=lambda x: -x[1]["score"])
        if not rows:
            continue
        epp = float(meta["anchors"][sf]["elo_per_point"])
        sname = "芝" if sf == "芝" else "ダート"
        print(f"\n【{sname}】{sname}の現役最上位 = 100")
        print(f"{pad('馬名', 20)}{'score':>6} {'最高':>6} {'走':>3} 信頼 {pad('最終出走', 11)}")
        for r, v in rows:
            print(f"{pad(r['馬名'], 20)}{v['score']:>6.1f} {v['peak']:>6.1f} {v['n']:>3}  "
                  f"{confidence(v['n'])}  {pad(v['last'], 11)}"
                  + (f" (引退・{v['retired']} 抹消)" if v["retired"] else
                     "" if v["active"] else f" ({sname}は1年以上出走なし)"))
        if len(rows) >= 2:
            print("  同じレースで先着する確率（条件・斤量は平均的と仮定）")
            for i in range(len(rows)):
                for j in range(i + 1, len(rows)):
                    (a, va), (b, vb) = rows[i], rows[j]
                    p = win_prob(va["score"], vb["score"], epp)
                    print(f"    {a['馬名']} > {b['馬名']} : {p:.0%}"
                          f"  (差 {va['score'] - vb['score']:+.1f} 点)")
    print("\n条件別の上振れ（スコアの見込みに対する着順の上下。+ が得意、カッコ内は走数）")
    for r in found:
        apt = json.loads(r.get("適性") or "{}")
        parts = []
        for key, lab in (("s:芝", "芝"), ("s:ダ", "ダ"), ("d:短", "〜1400"), ("d:マ", "〜1800"),
                         ("d:中", "〜2200"), ("d:長", "2300〜")):
            v = apt.get(key)
            if v and v[1]:
                parts.append(f"{lab} {v[0] / v[1]:+.2f}({int(v[1])})")
        print(f"  {pad(r['馬名'], 20)}{'  '.join(parts) if parts else '記録なし'}")
    return 0


def top_rows(table: dict[str, dict], surface: str, min_starts: int = 3,
             include_inactive: bool = False) -> list[tuple[dict, dict]]:
    """その芝ダのスコア順 [(行, その芝ダの値)]。走数はその芝ダでの走数で数える。"""
    rows = [(r, surf_row(r, surface)) for r in table.values()]
    rows = [x for x in rows if x[1] and x[1]["n"] >= min_starts
            and (include_inactive or x[1]["active"])]
    rows.sort(key=lambda x: -x[1]["score"])
    return rows


def cmd_top(args: argparse.Namespace) -> int:
    table, meta = read_outputs()
    for sf in ([args.surface] if args.surface else SURFACES):
        sname = "芝" if sf == "芝" else "ダート"
        print(f"\n【{sname}】強さスコア上位（基準日 {meta['asof']}、{sname}で {args.min_starts} 走以上）")
        for i, (r, v) in enumerate(top_rows(table, sf, args.min_starts, args.all)[: args.n], 1):
            print(f"{i:>3} {pad(r['馬名'], 20)}{v['score']:>6.1f} {v['n']:>3}走 "
                  f"{pad(v['last'], 11)} {r['最終クラス']}")
    return 0


def cmd_card(args: argparse.Namespace) -> int:
    if not args.out:
        return _card(args)
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = _card(args)
    text = buf.getvalue()
    print(text, end="")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    log.info("保存: %s", out)
    return rc


def _card(args: argparse.Namespace) -> int:
    table, meta = read_outputs()
    wk = WEEKLY_DIR / f"{args.date}.csv"
    if not wk.exists():
        print(f"出走表がありません: {wk}")
        return 1
    if meta["asof"] >= f"{args.date[:4]}-{args.date[4:6]}-{args.date[6:]}":
        print("※ このレーティングは当日以降の結果を含みます（回顧用。事前評価には使えません）。"
              f"事前の状態で見るには build --until {args.date} で作り直してください")
    fm = load_form_model()
    if not fm:
        print("※ 好走/凡走モデルなし（データ不足）。スコアだけ表示します")
    for race in card_races(args.date, table, fm, args.place, args.rno):
        print(f"\n{race['place']}{race['rno']}R {race['klass']} "
              f"{race['surface'] or ''}{race['dist']} {race['name']}"
              f"  ［{'芝' if race['surface'] == '芝' else 'ダート'}スコア］")
        for x in race["rows"]:
            s = f"{x['score']:>6.1f}" if x["score"] is not None else "    --"
            p3 = (f"{x['p3']:>4.0%}{'?' if x['score'] is None else ' '}"
                  if x["p3"] is not None else "     ")
            other = (f"（別の芝ダで {x['n_all']} 走）"
                     if x["score"] is None and x["n_all"] else "")
            print(f"  {x['ban']:>2} {pad(x['name'], 20)}{s}  3着内{p3} "
                  f"{x['n']:>2}走 信頼{confidence(x['n']) if x['n'] else '-'}  "
                  f"{x['label']}{x['reason']}{other}")
    return 0


def card_races(date_str: str, table: dict[str, dict], fm: dict,
               place: str | None = None, rno: str | None = None) -> list[dict]:
    """その日の出走表を、強さスコア順の行つきで返す（card 表示とローカルページの共通部）。

    Returns:
        [{race_id, place, rno, klass, surface, dist, name,
          rows: [{ban, name, score, n, n_all, p3, label, reason}]}]
        score・n はそのレースの芝ダでの値（その芝ダが初めての馬は score=None, n=0）。
        n_all は芝ダ通算の走数。rows は score 降順、score の無い馬は末尾。
    """
    day = datetime.strptime(date_str, "%Y%m%d").date()
    out: list[dict] = []
    for race in read_weekly(WEEKLY_DIR / f"{date_str}.csv"):
        if is_jump(race["klass"], race["surface"]):
            continue
        if place and race["place"] != place:
            continue
        if rno and str(race["rno"]) != str(rno):
            continue
        rows = []
        for ban, nm, jk, kn in race["horses"]:
            r = table.get(nm)
            fp = ft = None
            if r and fm:
                h = Horse(rating=0.0, n=int(r["出走数"]),
                          last_day=(datetime.strptime(r["最終出走"], "%Y-%m-%d").date()
                                    if r["最終出走"] else None),
                          last_class=r["最終クラス"] or None,
                          last_surface=r.get("最終芝ダ") or None,
                          last_dist=to_int(r.get("最終距離")),
                          resids=[float(x) for x in (r.get("近走上振れ") or "").split("/") if x],
                          last_jockey=r.get("最終騎手") or None,
                          last_kinryo=to_float(r.get("最終斤量")),
                          apt=json.loads(r.get("適性") or "{}"))
                ft = pre_features(h, day, race["klass"], race["surface"], to_int(race["dist"]),
                                  race["place"], jk, kn)
                fp = form_pred(fm, ft)
            lab = form_label(fm, fp) if fm else ""
            sv = surf_row(r, race["surface"]) if r and race["surface"] in SURFACES else None
            rows.append({"score": sv["score"] if sv else None, "ban": ban, "name": nm,
                         "n": sv["n"] if sv else 0,
                         "n_all": int(r["出走数"]) if r else 0,
                         "rating": sv["rating"] if sv else None, "fp": fp, "p3": None,
                         "label": lab, "reason": form_reason(fm, ft, lab) if fm else ""})
        # 3 着内率: 未収録馬はそのレースの収録馬の平均で仮置き（表示側で "?" を付ける）
        known = [x["rating"] for x in rows if x["rating"] is not None]
        if known:
            fill = sum(known) / len(known)
            adj = [(x["rating"] if x["rating"] is not None else fill)
                   + (fm.get("gamma", 0.0) * x["fp"] if fm and x["fp"] is not None else 0.0)
                   for x in rows]
            for x, p3 in zip(rows, top3_probs(adj, fm.get("temp", 1.0) if fm else 1.0)):
                x["p3"] = p3
        rows.sort(key=lambda x: (x["score"] is None, -(x["score"] or 0)))
        out.append({**{k: race[k] for k in ("race_id", "place", "rno", "klass", "surface",
                                             "dist", "name")}, "rows": rows})
    return out


def load_form_model() -> dict:
    fm_path = OUT_DIR / "form_model.json"
    return json.loads(fm_path.read_text(encoding="utf-8")) if fm_path.exists() else {}


def cmd_site(args: argparse.Namespace) -> int:
    import power_rating_site  # ページ生成は別ファイル（HTML テンプレートが長いため）

    return power_rating_site.build(args.date, Path(args.out) if args.out else None)


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
    for name in ("build", "tune", "ablate"):
        b = sub.add_parser(name)
        b.add_argument("--k0", type=float, default=Params.k0)
        b.add_argument("--floor-ratio", type=float, default=Params.floor_ratio)
        b.add_argument("--tau", type=float, default=Params.tau)
        b.add_argument("--eval-from", default=None, help="検証開始日 YYYY-MM-DD")
        b.add_argument("--until", default=None,
                       help="YYYYMMDD より前のレースだけで作る（その日の事前評価用）")
        b.add_argument("--form-features", default=",".join(FORM_DEFAULT),
                       help="好走/凡走モデルに使う条件（カンマ区切り）")
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
    k.add_argument("--out", default=None, help="表示内容をこのテキストファイルにも保存")
    ch = sub.add_parser("chart")
    ch.add_argument("names", nargs="+")
    ch.add_argument("--out", default=None, help="PNG の保存先")
    ch.add_argument("--surface", choices=list(SURFACES), default=None,
                    help="芝 / ダ（省略時は 1 頭目の主戦）")
    sub.add_parser("selftest")
    st = sub.add_parser("site")
    st.add_argument("--date", action="append", default=None,
                    help="出走表を載せる日 YYYYMMDD（複数可。省略時は直近 2 開催日）")
    st.add_argument("--out", default=None, help="出力先フォルダ（既定 reports/power_rating/site）")
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
    cmd = {"build": cmd_build, "tune": cmd_tune, "compare": cmd_compare, "top": cmd_top,
           "card": cmd_card, "selftest": cmd_selftest, "ablate": cmd_ablate,
           "chart": cmd_chart, "site": cmd_site}[args.cmd]
    try:
        return cmd(args)
    except (FileNotFoundError, KeyError) as e:
        log.error("%s", e)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
