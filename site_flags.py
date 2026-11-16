"""公開サイトに何を出すかのスイッチ（build_site.py / build_report_card.py が読む）。

SHOW_BETS: 買い目（T-10 確定・T-20 速報の券種と金額、的中一覧、回収率、見送り比較）を公開するか。
  2026-11-16 ユーザー指示「買い目の複勝のやつは裏でやって」で False。
  計算・記録（compute_bets / T-10 / reports/cowork_output / data/cowork_results.json）と
  私用 Discord（notify_config.json の discord_webhook）への通知はそのまま続く。公開 JSON から外すだけ。
  True に戻せば過去分も含めて元どおり表示される（データは消していない）。
"""
SHOW_BETS = False
