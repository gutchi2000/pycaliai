"""公開サイトに何を出すかのスイッチ（build_site.py / build_report_card.py が読む）。

BETS_HIDDEN_FROM: この開催日以降の買い目はサイトに出さない（YYYYMMDD、None なら全部出す）。
  2026-11-16 ユーザー指示「複勝のやつはもう表では出さずに裏で集計する」「すでに公開済みのやつは残して」。
  11/14 起算の本線（a95-sniper-v2: ◎の T-10 単勝 1.9 倍以下だけ参戦、◎複勝 1 点）から非公開。
  それより前に公開した買い目・的中一覧・回収率・見送りの成績はそのまま残す（サイトの集計はこの日の前日まで）。
  非公開の分も計算・記録（compute_bets / T-10 / reports/cowork_output / data/cowork_results.json）と
  私用 Discord への通知は続く。公開 JSON から外すだけなので、None に戻せば元どおり表示される。
"""
BETS_HIDDEN_FROM = "20261114"


def bets_hidden(date: str) -> bool:
    return BETS_HIDDEN_FROM is not None and str(date) >= BETS_HIDDEN_FROM
