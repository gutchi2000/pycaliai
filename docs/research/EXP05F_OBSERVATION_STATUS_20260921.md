# EXP05-F 前向き観測状況 (2026-09-21時点、収集完全性のみ)

**読み取り専用の確認**。`observation_report.py`（既存の読み取り専用ツール、
t35_shadow.ps1等の収集ロジックには一切干渉しない）の出力をそのまま記録する。
**性能・ROI・的中率は見ていない**。目標6,600件（valid_primary_observations、
`POWER_ANALYSIS.md`参照）に対する収集の進み具合と、収集基盤が正しく機能
しているかのみを確認する。

## 累計 (2026-09-21 14:20時点)

| 指標 | 値 |
|---|---|
| cumulative_market_observations | 43 |
| cumulative_complete_prediction_observations | 41 |
| cumulative_valid_primary_observations | **39** |
| target_valid_primary_observations | 6,600 |
| 進捗率 | 0.6% |

## 日付別

| 日付 | calendar取得 | 登録対象レース数 | 発火タスク数 | market_observations | complete_prediction | valid_primary | market-only | missed | T-35窓外等 | errors |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-09-19 | 未生成(JSON書込拒否) | — | 11 | 10 | 9 | 8 | 0 | 0 | 1件(オッズ取得失敗) | 0 |
| 2026-09-20 | success | 24 | 24 | 24 | 24 | 23 | 0 | 0 | 1件(jvlink_odds exit≠0) | 1件(RACE_DAY_INPUT_PENDING、後にt35_shadow.ps1手動実行で回収) |
| 2026-09-21 | success | 24 | 19 | 9 | 8 | 8 | 0 | 0 | 0 | **10件、収集中に進行中の障害あり(下記)** |

## ⚠️ 2026-09-21、現在進行中の収集障害を発見

`logs/exp05fs_errors.log`に、本日09:10〜14:15の間、約30分間隔で以下の
エラーが**繰り返し発生し続けている**（このレポート生成時点でも継続中の
可能性が高い）:

```
2026-09-21T09:10:02 2026092106040701 store_prediction失敗(未分類): bundle に 2026092106040701 が無い (20260921_bundle.json)
2026-09-21T09:45:03 2026092106040702 store_prediction失敗(未分類): bundle に 2026092106040702 が無い
2026-09-21T10:15:09 2026092106040703 store_prediction失敗(未分類): bundle に 2026092106040703 が無い
2026-09-21T10:45:03 2026092106040704 store_prediction失敗(未分類): bundle に 2026092106040704 が無い
2026-09-21T11:35:02 2026092106040705 store_prediction失敗(未分類): bundle に 2026092106040705 が無い
2026-09-21T12:05:02 2026092106040706 store_prediction失敗(未分類): bundle に 2026092106040706 が無い
2026-09-21T12:35:07 2026092106040707 store_prediction失敗(未分類): bundle に 2026092106040707 が無い
2026-09-21T13:05:03 2026092106040708 store_prediction失敗(未分類): bundle に 2026092106040708 が無い
2026-09-21T13:35:03 2026092106040709 store_prediction失敗(未分類): bundle に 2026092106040710 が無い
2026-09-21T14:15:04 2026092106040710 store_prediction失敗(未分類): bundle に 2026092106040710 が無い
```

会場コード「06」(中山)の第1〜10レースについて、`20260921_bundle.json`に
当該レースIDが存在しないため`store_prediction`が失敗し続けている。市場
snapshot自体(T-35オッズ取得)は取得できている可能性があるが
(`market_observations`が該当レースで記録されているか要確認)、完全な
予測保存(complete_prediction_observations)には到達していない。

**原因の推定（未確認、要調査）**: `20260921_bundle.json`（`export_weekly_marks.py`
が生成する印バンドル）に会場06のレースが含まれていない、または生成が
遅延している可能性が高い。これは`observation_report.py`の集計ロジックの
バグではなく、収集パイプライン側(bundle生成)の問題と見られる。

**この障害は「収集障害の修正」として対応可能な範囲だが、本レポートは
読み取り専用の確認に限定したため修正は行っていない**。別途対応が必要。

## 収集基盤の健全性についての所見

- calendarタスクは2026-09-19のみ失敗（JSON書込拒否、サニティチェックに
  よるfail-closed動作。危険な誤データを書き込まない設計が機能している）。
- 2026-09-20は`RACE_DAY_INPUT_PENDING`（TARGET出走表エクスポート遅延）が
  一度発生したが、ログ上は後続タスクで回収されている（`missed_count=0`）。
- 2026-09-21は「既知開催オーバーライド」（日曜開催として認識）が機能し
  T-35タスクは正常に24件登録されたが、上記のbundle不整合で会場06が
  完全予測保存まで到達していない。
- 全期間を通じ`missed_count=0`（入力遅延で一度もタスクを作れなかった
  レースはゼロ）。

## 結論

収集基盤は概ね機能しているが、**2026-09-21に進行中の収集障害**（会場06
のbundle不一致）があり、対応が必要。目標6,600件に対し現在39件(0.6%)、
収集開始から3日目。性能・ROI・的中率の評価はまだ実施していない（目標
件数到達まで実施しない、`POWER_ANALYSIS.md`/spec §16の規律通り）。

関連: [[project_mcond_exp05f_forward_shadow]] [[docs/research/RESEARCH_STOPLINE_20260921]]
