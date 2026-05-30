"""20260531 競馬予想スクリプト（オッズ＋近走の簡易スコア方式）。

bundle JSON を入力に、各馬の勝率/連対率/複勝率を推定し、
印（◎◯▲△×）と予算制約下の推奨買い目を生成して bets.json を出力する。

外部データ・ネットワークは一切使わず、bundle 内の情報のみで完結する（再現性重視）。

依存:
    Python 3.10+ / 標準ライブラリのみ（pandas/numpy 不要）。

設計メモ:
    - 市場オッズは「群衆の知恵」として最も強い情報源。単勝オッズから控除率を
      補正した暗黙勝率 p_odds を求める。
    - recent_form（近4走の着順、例 "1-1-2-1"）を着順スコア化し、近走の勢いを
      加点要素 s_form として使う。
    - 最終スコア = w_odds * p_odds_norm + w_form * s_form_norm をレース内で
      softmax 的に正規化して勝率 p_win を得る（合計=1.0）。
    - 連対率・複勝率は勝率からの順序統計近似（Harville 法）で推定。
    - リーク防止: 当日結果・着順確定など未来情報は bundle に含まれず、使用しない。
"""
from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("predict_bets")

# --- ハイパーパラメータ（再現性のため固定値で明示） -----------------------
TAKEOUT = 0.20            # 単勝の概算控除率（日本の単勝は約20%）
W_ODDS = 0.75             # 最終スコアにおけるオッズ項の重み
W_FORM = 0.25             # 最終スコアにおける近走項の重み
SOFTMAX_TEMP = 1.0        # スコア→勝率変換の温度（1.0=標準）
BUDGET_PER_RACE = 1000    # 1レースあたり予算（JPY）
BET_UNIT = 100            # 馬券最小単位（JPY）

# 印の閾値（推定勝率 p_win ベース、レース内ランキングと併用）
# ◎=本命, ◯=対抗, ▲=単穴, △=連下, ×=押さえ
MARK_RANK = ["◎", "◯", "▲", "△", "×"]


@dataclass
class Horse:
    """1頭分の特徴量と推定結果を保持する。"""

    num: int
    draw: int
    name: str
    sex_age: str
    weight_carried: float
    jockey: str
    odds: float
    popularity: int
    recent_form: str
    comment: str
    p_odds: float = 0.0       # オッズ由来の暗黙勝率（控除率補正後・未正規化）
    s_form: float = 0.0       # 近走スコア（0-1）
    score: float = 0.0        # 合成スコア
    p_win: float = 0.0        # 推定勝率
    p_place2: float = 0.0     # 推定連対率（2着以内）
    p_show: float = 0.0       # 推定複勝率（3着以内）
    mark: str = ""            # 印


def parse_recent_form(form: str) -> float:
    """近走文字列(例 "1-1-2-1")を 0-1 の近走スコアに変換する。

    着順を指数減衰で得点化（1着=満点、着順が悪いほど低下）し、
    直近走ほど重く重み付けする。出走取消等の非数値は中立(0.4)扱い。

    Args:
        form: "着-着-着-着" 形式の近走着順文字列。

    Returns:
        0.0-1.0 のスコア。情報がない場合は 0.4 を返す。
    """
    if not form:
        return 0.4
    tokens = re.split(r"[-/,\s]+", form.strip())
    vals: list[float] = []
    weights: list[float] = []
    # 直近ほど重く（左側を直近とみなす一般的な並び）
    for i, tok in enumerate(tokens):
        w = 0.6 ** i  # 直近=1.0, 1走前=0.6, ...
        m = re.match(r"(\d+)", tok)
        if m:
            rank = int(m.group(1))
            # 着順スコア: 1着=1.0, 2着≈0.78, 3着≈0.61, ... 指数減衰
            s = 0.78 ** (rank - 1)
        else:
            s = 0.4  # 非数値（中止・除外等）は中立
        vals.append(s * w)
        weights.append(w)
    if not weights:
        return 0.4
    return sum(vals) / sum(weights)


def implied_prob_from_odds(odds: float) -> float:
    """単勝オッズから控除率補正前の素の暗黙確率(1/odds)を返す。

    Args:
        odds: 単勝オッズ（例 3.2）。0以下は微小確率にフォールバック。

    Returns:
        1/odds（後段でレース内正規化する）。
    """
    if odds is None or odds <= 0:
        return 0.001
    return 1.0 / odds


def softmax(xs: list[float], temp: float = 1.0) -> list[float]:
    """数値安定な softmax。合計1.0のリストを返す。"""
    if not xs:
        return []
    m = max(xs)
    exps = [math.exp((x - m) / temp) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]


def harville_place_probs(win_probs: list[float]) -> tuple[list[float], list[float]]:
    """Harville モデルで勝率から連対率(2着内)・複勝率(3着内)を近似する。

    P(iが2着内) = p_i + Σ_{j≠i} p_j * p_i/(1-p_j)
    P(iが3着内) も同様に展開（計算量 O(n^2)〜O(n^3) だが頭数少なく問題なし）。

    Args:
        win_probs: レース内各馬の勝率（合計≈1.0）。

    Returns:
        (連対率リスト, 複勝率リスト)。各々 0-1 にクリップ。
    """
    n = len(win_probs)
    p = win_probs
    place2: list[float] = []
    show: list[float] = []
    for i in range(n):
        pi = p[i]
        # 2着以内 = 1着 + (誰かが1着で自分が2着)
        p2 = pi
        for j in range(n):
            if j == i:
                continue
            denom = 1.0 - p[j]
            if denom > 1e-9:
                p2 += p[j] * (pi / denom)
        place2.append(min(p2, 0.999))

        # 3着以内 = 2着以内 + (jが1着,kが2着で自分が3着) を加算
        p3 = pi
        for j in range(n):
            if j == i:
                continue
            dj = 1.0 - p[j]
            if dj <= 1e-9:
                continue
            p3 += p[j] * (pi / dj)  # 2着パート
            for k in range(n):
                if k == i or k == j:
                    continue
                djk = 1.0 - p[j] - p[k]
                if djk > 1e-9:
                    p3 += p[j] * (p[k] / dj) * (pi / djk)
        show.append(min(p3, 0.999))
    return place2, show


def assign_marks(horses: list[Horse]) -> None:
    """勝率降順で上位5頭に ◎◯▲△× を付与する（破壊的）。"""
    ranked = sorted(horses, key=lambda h: h.p_win, reverse=True)
    for idx, h in enumerate(ranked):
        h.mark = MARK_RANK[idx] if idx < len(MARK_RANK) else ""


@dataclass
class BetTicket:
    """1点の買い目。"""

    bet_type: str          # "単勝" / "複勝" / "馬連"
    selection: list[int]   # 馬番（馬連は2頭）
    stake: int             # 賭金(JPY)
    est_prob: float        # 的中推定確率
    note: str = ""


def build_bets(horses: list[Horse], budget: int) -> list[BetTicket]:
    """予算制約下で点数最小・期待回収重視の買い目を生成する。

    方針（的中率と回収率のトレードオフ）:
        - 軸（◎）の複勝で取りこぼしを抑える（的中率寄り）。
        - ◎◯の馬連1点で回収を狙う（回収率寄り）。
        - 予算1000円なら 複勝400 + 馬連600 を基本配分。
        - 残予算が単位に満たない場合は切り捨て。

    Args:
        horses: 印・確率が計算済みの馬リスト。
        budget: レース予算(JPY)。

    Returns:
        買い目チケットのリスト。
    """
    by_mark = {h.mark: h for h in horses if h.mark}
    honmei = by_mark.get("◎")
    taikou = by_mark.get("◯")
    tickets: list[BetTicket] = []
    remaining = budget

    if honmei is None:
        return tickets

    # 1) 軸の複勝（的中率の土台）: 予算の約40%
    show_stake = (int(budget * 0.4) // BET_UNIT) * BET_UNIT
    if show_stake >= BET_UNIT and remaining >= show_stake:
        tickets.append(
            BetTicket(
                bet_type="複勝",
                selection=[honmei.num],
                stake=show_stake,
                est_prob=round(honmei.p_show, 4),
                note="軸の複勝で取りこぼし防止",
            )
        )
        remaining -= show_stake

    # 2) ◎◯ 馬連（回収狙い）: 残り全額
    if taikou is not None:
        ren_stake = (remaining // BET_UNIT) * BET_UNIT
        if ren_stake >= BET_UNIT:
            # 馬連的中の概算: 両馬が共に2着以内に入る確率の近似
            est = honmei.p_place2 * taikou.p_place2 * 0.85
            tickets.append(
                BetTicket(
                    bet_type="馬連",
                    selection=sorted([honmei.num, taikou.num]),
                    stake=ren_stake,
                    est_prob=round(est, 4),
                    note="◎◯の組み合わせで回収を狙う",
                )
            )
            remaining -= ren_stake
    else:
        # 対抗不在時は軸の単勝に回す
        win_stake = (remaining // BET_UNIT) * BET_UNIT
        if win_stake >= BET_UNIT:
            tickets.append(
                BetTicket(
                    bet_type="単勝",
                    selection=[honmei.num],
                    stake=win_stake,
                    est_prob=round(honmei.p_win, 4),
                    note="対抗不在のため軸の単勝",
                )
            )
            remaining -= win_stake

    return tickets


def process_race(race: dict[str, Any]) -> dict[str, Any]:
    """1レースを処理して出力用 dict を返す。"""
    horses = [
        Horse(
            num=h["num"],
            draw=h["draw"],
            name=h["horse"],
            sex_age=h["sex_age"],
            weight_carried=float(h["weight_carried"]),
            jockey=h["jockey"],
            odds=float(h["odds"]),
            popularity=int(h["popularity"]),
            recent_form=h.get("recent_form", ""),
            comment=h.get("comment", ""),
        )
        for h in race["horses"]
    ]

    # 特徴量計算
    raw_odds = [implied_prob_from_odds(h.odds) for h in horses]
    s_odds_sum = sum(raw_odds)
    forms = [parse_recent_form(h.recent_form) for h in horses]
    form_max = max(forms) if forms else 1.0
    form_min = min(forms) if forms else 0.0
    form_range = (form_max - form_min) or 1.0

    for h, ro, fm in zip(horses, raw_odds, forms):
        # オッズ暗黙確率（控除率補正してレース内で正規化）
        h.p_odds = (ro / s_odds_sum) * (1.0 - TAKEOUT) if s_odds_sum else 0.0
        # 近走スコアを 0-1 に min-max 正規化
        h.s_form = (fm - form_min) / form_range

    # 合成スコア → softmax で勝率化
    scores = [W_ODDS * h.p_odds + W_FORM * h.s_form for h in horses]
    p_wins = softmax([s / max(scores) for s in scores] if max(scores) else scores,
                     temp=SOFTMAX_TEMP)
    for h, sc, pw in zip(horses, scores, p_wins):
        h.score = sc
        h.p_win = pw

    # 連対率・複勝率
    place2, show = harville_place_probs([h.p_win for h in horses])
    for h, p2, p3 in zip(horses, place2, show):
        h.p_place2 = p2
        h.p_show = p3

    # 印付与
    assign_marks(horses)

    # 買い目
    tickets = build_bets(horses, BUDGET_PER_RACE)
    total_stake = sum(t.stake for t in tickets)

    return {
        "race_id": race["race_id"],
        "race_meta": race["race_meta"],
        "num_horses": len(horses),
        "predictions": [
            {
                "num": h.num,
                "draw": h.draw,
                "horse": h.name,
                "jockey": h.jockey,
                "odds": h.odds,
                "popularity": h.popularity,
                "mark": h.mark,
                "win_prob": round(h.p_win, 4),
                "place2_prob": round(h.p_place2, 4),
                "show_prob": round(h.p_show, 4),
                "score": round(h.score, 4),
            }
            for h in sorted(horses, key=lambda x: x.p_win, reverse=True)
        ],
        "bets": [
            {
                "bet_type": t.bet_type,
                "selection": t.selection,
                "stake": t.stake,
                "est_hit_prob": t.est_prob,
                "note": t.note,
            }
            for t in tickets
        ],
        "total_stake": total_stake,
        "budget": BUDGET_PER_RACE,
    }


def main(in_path: str, out_path: str) -> None:
    """bundle を読み込み bets.json を出力する。"""
    logger.info("read bundle: %s", in_path)
    bundle = json.loads(Path(in_path).read_text(encoding="utf-8"))
    races = bundle.get("races", [])
    logger.info("races: %d", len(races))

    out_races = []
    for r in races:
        res = process_race(r)
        logger.debug("processed %s stake=%d", res["race_id"], res["total_stake"])
        out_races.append(res)

    out = {
        "meta": {
            "date": bundle.get("meta", {}).get("date"),
            "generated_by": "predict_bets.py",
            "method": "odds(0.75)+recent_form(0.25) softmax / Harville place / budget1000",
            "takeout": TAKEOUT,
            "budget_per_race": BUDGET_PER_RACE,
            "marks": "◎◯▲△× (win_prob 降順 上位5頭)",
            "leak_guard": "当日結果・未来情報は不使用",
        },
        "races": out_races,
    }
    Path(out_path).write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("wrote: %s", out_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="競馬予想 bets.json 生成")
    parser.add_argument(
        "--in", dest="in_path", default="20260531_bundle.json", help="入力bundle"
    )
    parser.add_argument(
        "--out", dest="out_path", default="20260531_bets.json", help="出力bets"
    )
    args = parser.parse_args()
    main(args.in_path, args.out_path)
