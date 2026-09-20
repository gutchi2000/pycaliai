# -*- coding: utf-8 -*-
"""
evaluate.py — 決済ドライバ(仕様書§9/追加指示5)。実着順・払戻は評価時だけ使用する。

設計方針: 同着・複勝レンジ等の複雑なJRA払戻規則を自前で再計算しない。常に
`data/kekka_*.csv`等の**JRA公式確定払戻テーブル**から券種・選択(ticket_type, selection)
に対応する「100円あたり払戻額」を直接引き当てて使う。これにより同着・複勝オッズレンジ
といった規則依存の複雑さを設計から排除し、fail-closed(欠損・不正入力は例外)にできる。

`docs/plans/exhaustive_betting_policy_verdict_20260825.md`が明記する通り「既存settlerには
順位0、返還、1〜3着同着、複数払戻を誤決済する経路があった」という過去の実害領域のため、
このモジュールは新規に慎重設計する(過去のsettle実装を継承しない)。

実行: このファイルは単体実行を想定しない。test_evaluate.py等から使う。
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field

_RACE_ID_RE = re.compile(r"^\d{16}$")


class SettlementError(Exception):
    """決済不能(欠損結果・不正race_id等)。fail-closedのため例外にする。"""


@dataclass(frozen=True)
class RaceOutcome:
    """1レース分の確定結果。payout_per_100_yenは(ticket_type, selection)→100円あたり
    払戻額(円、整数)の辞書。同着・複勝レンジ等は呼び出し側がJRA公式確定払戻から
    そのまま渡す(このモジュールでは再計算しない)。"""

    race_id: str
    race_void: bool = False  # 全額返還(競走除外・中止等でレース自体が不成立)
    scratched_horses: frozenset[int] = field(default_factory=frozenset)  # 取消(出走取消)馬番
    payout_per_100_yen: dict[tuple[str, tuple[int, ...]], int] = field(default_factory=dict)
    is_missing: bool = False  # 結果未着(まだ確定していない)

    def __post_init__(self) -> None:
        if not _RACE_ID_RE.match(self.race_id):
            raise SettlementError(f"invalid race_id: {self.race_id!r} (16桁数字である必要がある)")


def _normalize_selection(ticket_type: str, selection: tuple[int, ...]) -> tuple[int, ...]:
    """順不同券種(複勝以外の組系)は昇順に正規化し、payout_per_100_yen辞書のキーと
    一致させる。単勝・複勝・馬単は順序/単一馬のまま。"""
    if ticket_type in ("umaren", "wide", "sanrenpuku"):
        return tuple(sorted(selection))
    return tuple(selection)


def settle_ticket(ticket_type: str, selection: tuple[int, ...], stake_yen: int, outcome: RaceOutcome) -> dict:
    """1枚の馬券を決済する。戻り値: {status, payout_yen, profit_yen}。

    status:
      "missing"   結果未着(決済不能、profit集計には含めない、呼び出し側が扱いを決める)
      "void"      レース自体が全額返還
      "refunded"  選択馬に取消があり当該券が返還
      "hit"       的中(公式払戻テーブルに基づく)
      "miss"      不的中(stake全損)
    """
    if not isinstance(stake_yen, int) or stake_yen <= 0 or stake_yen % 100 != 0:
        raise SettlementError(f"stake_yen must be a positive multiple of 100, got {stake_yen}")
    if not _RACE_ID_RE.match(outcome.race_id):
        raise SettlementError(f"invalid race_id: {outcome.race_id!r}")

    if outcome.is_missing:
        return {"status": "missing", "payout_yen": None, "profit_yen": None}

    if outcome.race_void:
        return {"status": "void", "payout_yen": stake_yen, "profit_yen": 0}

    if any(h in outcome.scratched_horses for h in selection):
        return {"status": "refunded", "payout_yen": stake_yen, "profit_yen": 0}

    key = (ticket_type, _normalize_selection(ticket_type, selection))
    payout_per_100 = outcome.payout_per_100_yen.get(key)

    if payout_per_100 is None or payout_per_100 == 0:
        # 公式テーブルに載っていない、または明示的に0 = 不的中として扱う(的中していれば
        # 必ず払戻テーブルに正の値で載るはずというJRA決済の前提に基づく。0を明示的に
        # missへ倒すのは防御的措置で、財務的にはどちらの経路でも損益は同じ-stake_yen)。
        # ただし呼び出し側がテーブルを不完全にしか埋めていない場合との区別がつかないため、
        # fail-closed運用ではStage 2Aのevaluate呼び出し側がテーブル完全性を別途検証すること
        return {"status": "miss", "payout_yen": 0, "profit_yen": -stake_yen}

    payout_yen = (stake_yen // 100) * payout_per_100
    return {"status": "hit", "payout_yen": payout_yen, "profit_yen": payout_yen - stake_yen}


def settle_portfolio(stakes: dict[tuple[str, tuple[int, ...]], int], outcome: RaceOutcome) -> dict:
    """複数券をまとめて決済する。ワイド等の複数的中を各券独立に評価する
    (的中判定はbuild_scenarios.ticket_wins_in_stateと整合、複数のワイドが同時に
    的中することを妨げない設計)。"""
    results = {}
    total_stake = 0
    total_payout = 0
    any_missing = False
    for (ticket_type, selection), stake_yen in stakes.items():
        r = settle_ticket(ticket_type, selection, stake_yen, outcome)
        results[(ticket_type, selection)] = r
        if r["status"] == "missing":
            any_missing = True
        else:
            total_stake += stake_yen
            total_payout += r["payout_yen"]
    return {
        "per_ticket": results,
        "total_stake_yen": total_stake,
        "total_payout_yen": total_payout,
        "total_profit_yen": total_payout - total_stake,
        "any_missing": any_missing,
    }
