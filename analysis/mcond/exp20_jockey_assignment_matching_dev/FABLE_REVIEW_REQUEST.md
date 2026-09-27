# Fable宛て — EXP20 v0.1仕様レビュー依頼

EXP20「騎手配置の制約付きマッチング残差」の仕様レビューをお願いします。実装・学習・結果評価はまだ行っていません。

読むファイル:

1. `analysis/mcond/exp20_jockey_assignment_matching_dev/SPEC.md`
2. `analysis/mcond/exp20_jockey_assignment_matching_dev/spec.json`
3. `analysis/mcond/exp20_jockey_assignment_matching_dev/README.md`
4. 境界確認用: `analysis/mcond/exp01_choice_dev/REPORT.md`
5. 境界確認用: `docs/research/RESEARCH_STOPLINE_20260921.md`

特に次を厳しく確認してください。

1. 観測済みsame-race jockey roster内のswap集合という限定で、観測不能な依頼・辞退を捏造せず識別可能な問いになっているか。
2. EXP01のraw choice / team deviation、EXP11の疎な騎手×調教師pair探索と実質同値でないか。
3. C0（365日最低斤量）、C1（730日）、C2（斤量制約なし）の候補集合と安定性Gateで、恣意的な候補集合依存を十分に検出できるか。
4. A0独立edge nullとA1 Sinkhorn matchingの差が、本当に一対一競合による増分を分離しているか。
5. `assign_logp`等のresidual blockが「人気騎手」「継続騎乗」「厩舎との頻度」を別表現しただけで通る経路が残っていないか。
6. assignment likelihoodを結果とは別の観測対象として2019〜2023で評価してよいか。正則化を≤2018で固定する規則は十分か。
7. day-start snapshot、同日カード利用、過去騎手成績、最低騎乗斤量にtime leakageが残らないか。
8. Stage 1の最初の性能Gateをterminal close市場、主比較をM3−M2に置いた設計が十分か。
9. placebo P1〜P4がproperで、特にP1が騎手の当日騎乗本数を保持できる定義になっているか。
10. practical floor、meeting-day bootstrap、LOO、5年方向条件が既存EXP16A/18と整合するか。

レビュー結果は「凍結可」「修正後に凍結可」「Stage 0前に停止」のいずれかで、必須修正と推奨修正を分けて返してください。

レビュー段階では実装・学習・2019〜2023の着順評価・2024/2025開封・ROI・候補生成・production変更を行わないでください。

