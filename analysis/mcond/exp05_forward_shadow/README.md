# EXP05-F 市場条件付き第2段モデルの前向き検証基盤

共通基盤は `analysis/mcond/README.md`。本番の印・買い目・資金配分には一切接続しない。
データ在庫監査は `DATA_SOURCE_AUDIT.md`、特徴パリティ監査は `FEATURE_PARITY.csv`、
カテゴリ正規化の全列監査は `CATEGORY_PARITY.md`/`CATEGORY_PARITY.csv`、
予測・成績指標への影響は `PREDICTION_IMPACT.md`、動的能力の識別解決監査は
`dyn_skill_resolution_audit.py`の出力(`out/dyn_skill_resolution.json`)、
凍結モデルの詳細は `MODEL_FREEZE.md`、定義同値性チェックは `SERVE_PARITY.md`、
必要サンプル数は `POWER_ANALYSIS.md`、実データ試験は `LIVE_PREFLIGHT.md`、
スケジューラ設計は `SCHEDULER_PLAN.md`、混在コミットの記録は `MIXED_COMMIT_NOTE.md`、
事前登録の仕様は `spec.json`。

## 証拠水準 (必ず守ること)
2023-2025年はEXP04で仮説発見・EXP05で再評価済みであり「未使用期間」ではない。EXP05の結果は
「頑健な開発結果」。**唯一の最終確認は、これから前向きに保存する未来データ。**
2026年の過去開催分は31-38分前の市場スナップショットが1件も存在しないため(`DATA_SOURCE_AUDIT.md`)、
「準確認的な時系列OOS」にすらできない。EXP05-Fは前向き観測ゼロから開始する。

## 登録前Gate (spec「EXP05-F 最終確認」§10、状況)
- [x] 重要4カテゴリ(芝・ダ/芝(内・外)/馬場状態/天気)の正規化後unknown_rate — 0.0%〜0.9%
  (障害レース起因の残差のみ、`CATEGORY_PARITY.md`)
- [x] 全カテゴリ監査完了 (28列、`CATEGORY_PARITY.csv`。種牡馬/母父馬/生産者は既知の
  高基数・データ欠損として区別)
- [x] 恒久canary実装 (`export_weekly_marks.py`、unknown_rate>5%でgate_errors、fail-closed)
- [x] 単体テストPASS (`tests/test_category_normalize.py` 13件、v6 encoder語彙との
  直接一致検証込み)
- [x] controlled replay結果記録 (`PREDICTION_IMPACT.md`。2026年12週375レースのcontrolled
  replay、◎変更5.9%・成績指標は統計的に中立。spec採用条件=学習時語彙との意味的一致は
  満たしている)
- [x] 動的能力の履歴保有馬解決率 — **100%** (`_HistoryIndex`種牡馬+生年解決、旧v2は65.0%、
  `out/dyn_skill_resolution.json`)
- [x] 全テストPASS (`pytest tests/ analysis/mcond/exp05_forward_shadow/test_forward_shadow.py
  analysis/mcond/exp05_market_residual_dev/test_time_safety.py -q` → 193件)
- [x] freeze metadata更新 (`MODEL_FREEZE.md` v3節、`out/freeze_manifest_v2.json`。
  モデルartifact自体は不変、入力生成コードのhashのみ更新)
- [x] 実開催T-35確認PASS (`LIVE_PREFLIGHT.md`、2026-09-19実施)
- [x] 本番非干渉確認PASS (`reports/live_odds`・`reports/site_odds`とも無汚染)

## 独立に見つけて修正した問題 (EXP05-Fのスコープ外、ユーザー承認の上で本番修正済み)
週次CSVのカテゴリ表記(`芝・ダ`="ダート"等)が学習時の表記(`master_v2.csv`="ダ"等)と
食い違っており、本番`export_weekly_marks.py`の`apply_encoders()`が正規化しないまま
`models/unified_rank_v6.pkl`のLabelEncoderへ渡していた。**2026-09-19、ユーザー承認の上で
`export_weekly_marks.py`を修正済み**(`category_normalize.py`が正本、
`CATEGORY_PARITY.md`/`PREDICTION_IMPACT.md`参照)。

## 実行順
```
# 一度だけ (凍結、既に実行済み)
python -m analysis.mcond.exp05_forward_shadow.freeze_model        # M1/M3/M4凍結

# 監査 (再実行可能、都度出力を上書き)
python -m analysis.mcond.exp05_forward_shadow.feature_parity_audit
python -m analysis.mcond.exp05_forward_shadow.category_parity_audit
python -m analysis.mcond.exp05_forward_shadow.serve_parity_check
python -m analysis.mcond.exp05_forward_shadow.dyn_skill_resolution_audit --date YYYYMMDD
python -m pytest tests/test_category_normalize.py analysis/mcond/exp05_forward_shadow/test_forward_shadow.py -q

# 週次 (土曜朝、bundle生成後)
python -m analysis.mcond.exp05_forward_shadow.feature_snapshot --date YYYYMMDD

# レース毎 (発走31-38分前を狙う、通常はタスクスケジューラ経由)
python -m analysis.mcond.exp05_forward_shadow.market_snapshot --once <rid16> --date YYYYMMDD

# 結果確定後 (日曜夜)
python -m analysis.mcond.exp05_forward_shadow.join_results --date YYYYMMDD
```

## タスクスケジューラへの登録 (2026-09-19登録、同日中に毎日トリガーへ修正)
`t35_shadow.ps1` は `t20_site.ps1` と全く同じ「レース毎タスク」方式で実装済み・実データで
動作確認済み。当初は`PyCaLiAI_T20_Site`と同じ土日9:00起動で登録したが、2026-09-21(月・祝)
開催を取りこぼすとの指摘を受け**毎日9:00起動**へ修正(モデル・特徴・購入ルールは無変更)。
`data/weekly/{date}.csv`が無い日(開催なし)は即座に正常終了、レース毎タスクは冪等に登録
(重複登録せず、発走枠を過ぎた陳腐化タスクだけ削除)。詳細・解除コマンドは
`SCHEDULER_PLAN.md`「2026-09-19 毎日トリガー化」節参照。

## 非干渉の設計
- 生成物は全て専用ディレクトリ: `reports/exp05fs_odds/`, `data/_research/mcond/exp05fs_*/`
- `jvlink_odds.py`と`forward_prices.py`への変更は「新しいstage文字列を1つ追加」のみ
  (既存t10/t20/vote/close/manualの動作は不変、`tests/test_forward_prices.py`で確認済み。
  このコミットに他セッションの未コミット変更が混入した経緯は`MIXED_COMMIT_NOTE.md`参照)
- `weekly_nicegui.ps1` / `compute_bets.py` / `t10_runner.py` / `t20_site_bets.py` は未変更
- 例外は`logs/exp05fs_errors.log`に記録し、他の週次タスクをブロックしない

## 必要サンプル数
**6,600レース ≈ 約99開催週 ≈ 約1.9年**(`POWER_ANALYSIS.md`参照、race-level 1標本z検定、
両側α=1%・検出力80%)。途中経過でのROI判断は禁止。

## 次の一手 (未着手)
- 前向き観測の蓄積を待つ (6,600レース到達まで主評価Gate1を実行しない)
- 2026年デビュー馬の同姓同名衝突リスク低減 (`_HistoryIndex`が種牡馬・生年とも不明/曖昧な
  ケースのみ残存、現状0件)
