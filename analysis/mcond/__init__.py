# -*- coding: utf-8 -*-
"""市場条件付き追加情報検定の共通基盤 (docs/research/NEXT_GEN_RESEARCH_PLAN_20260918.md)。

構成:
  market.py    購入可能時点の単勝オッズ → 市場確率 (取得時刻・overround・異常フラグ付き)
  v6base.py    v6 の時点安全なスコア (2015-23 OOF / 2024-25 本番) → PL 勝率・3着内率
  evaluate.py  M0..Mk 比較 (ロジスティック, 年別, レース日ブロック bootstrap, 部分集合)
実験は analysis/mcond/expNN_*/ に置く。本番経路 (compute_bets / export_weekly_marks) には接続しない。
"""
