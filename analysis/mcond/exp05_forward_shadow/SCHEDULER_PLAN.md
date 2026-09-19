# EXP05-F タスクスケジューラ設計 (spec §10)

**2026-09-19登録、同日中にマスタートリガーを土日限定→毎日9:00へ修正**(2026-09-21月曜祝日
開催の取りこぼしをユーザーが指摘、末尾「2026-09-19 毎日トリガー化」節参照)。

`t20_site.ps1`/`t10.ps1`と全く同じ「レース毎タスク」方式を採用する
(`t35_shadow.ps1`実装済み)。理由: 既に本番で実績のある設計をそのまま踏襲することで
新規に検証すべき箇所を最小化する。モデル・特徴・購入ルールは一切変更していない。

## 起動方式 (2026-09-19修正後)

- **毎日**9:00に常設タスク("PyCaLiAI_EXP05FS_T35")が`.\t35_shadow.ps1 -Schedule`を起動
  (曜日を一切見ない、`New-ScheduledTaskTrigger -Daily`)
- `-Schedule`はまず`data\weekly\{date}.csv`の存在だけを確認する。**無ければ即座に
  正常終了(exit 0)** — 開催が無い日は`bundle`が来るのを待たない(旧実装は15:00まで
  空回りしてから`exit 1`していたが、毎日トリガーではこれが平日毎に発生し無駄だった)
- 存在すれば`reports\cowork_input\{date}_bundle.json`の生成を15:00まで待ってから
  `feature_snapshot.py --date`を1回実行し、その日の全レースについて
  「発走-35分」にWindowsタスクを1つずつ登録する(`New-ScheduledTaskTrigger -Once`,
  `-WakeToRun`, 2時間の実行有効期限)
- 各レースタスクは発走時刻を過ぎていたら何もせず終了 (t20_site_bets.pyと同じガード)

## 冪等なレース毎タスク登録 (2026-09-19修正)

旧実装は毎回`PyCaLiAI_EXP05FS_T35R_*`を全削除してから作り直していた。修正後は
既存タスク一覧を1回取得し:
- 対象レースと同名タスクが既に**未来時刻**で存在 → 触らない(重複登録しない)
- 対象レースと同名タスクが**過去時刻**のまま残存 → 削除するだけ(陳腐化除去、
  発走枠を過ぎているので再登録しない=誤って再実行させない)
- 今回の対象に含まれない(=別日・処理済みの)タスクで、トリガー時刻が過去のもの → 削除
  (将来時刻のものは他日の正当な予約の可能性があるため触らない)

動作確認 (2026-09-19実施):
```
.\t35_shadow.ps1 -Schedule -Date 20260916   # weekly CSV無し → [no-race] ... exit 0
.\t35_shadow.ps1 -Schedule -Date 20260919   # 2回目実行 → 新規登録0/既存スキップ8/陳腐化削除1
```

## 祝日開催・月曜開催・代替開催への対応

`load_post_times`は`data/weekly/{date}.csv`から発走時刻を読むだけで曜日を仮定しない
(t10_runner.pyの既存実装をそのまま利用)。毎日トリガー化により**祝日(月曜等)開催でも
自動的に拾われる**(その日の朝9:00に`data/weekly/{date}.csv`が存在してさえいれば)。

**残る制約(EXP05-F固有ではなく、TARGET出走表エクスポートのタイミングに起因する制約)**:
`data/weekly/{date}.csv`はユーザーがTARGET側でその開催の出走表をエクスポートした時点で
初めて現れる。祝日月曜開催のエクスポートが朝9:00より後になった場合、その日のマスター
トリガーは「開催なし」と判定して何も登録しない(誤検知ではなく、時点でのデータ不在に
忠実な判定)。この場合は**エクスポート後に手動で`.\t35_shadow.ps1 -Schedule`を再実行**
すれば良い(既存t10.ps1が「祝日開催のみ手動」としているのと同じ運用パターン、
本番T-10ラインも同じ制約を持つ)。

## 2026-09-21(月・祝、中山・阪神)についての確認結果

2026-09-19時点で`data\weekly\20260921.csv`はまだ存在しない(TARGET未エクスポート、
2026-09-20分も同様に未エクスポート)。よって**現時点で9/21の個別タスクを実際に
作成することはできない**(存在しないデータから発走時刻を読むことはできないため)。
これは正直に報告する — 実装の不備ではなくデータの前提条件。

毎日トリガー化により、9/21朝9:00の時点で`data\weekly\20260921.csv`が存在していれば
自動的にその日のレース毎タスクが登録される(上記「動作確認」の`-Date 20260919`実行と
全く同じコードパスで、既に正しく動作することを確認済み)。存在していない場合は
エクスポート後に手動`-Schedule`で拾う。

## 多重起動・スリープ

- `-MultipleInstances IgnoreNew`: 同一レースタスクの多重起動を防止 (t20_site.ps1と同じ設定)
- `-WakeToRun` + `-StartWhenAvailable`: PCスリープ時も起床して実行 (同上)
- タスクは実行後6時間で自動削除 (`-DeleteExpiredTaskAfter`)

## 実行環境の前提 (要文書化、spec §10)

| 項目 | 値 |
|---|---|
| 実行ユーザー | タスク登録時のユーザー(通常ログインユーザー)。t10/t20と同じ前提 |
| ログオン状態 | ログオフ中でも`-WakeToRun`で起動可能 (要: タスクのRun設定で「ログオンしていなくても実行する」) |
| 作業ディレクトリ | `E:\PyCaLiAI` (`t35_shadow.ps1`冒頭で`Set-Location`) |
| PowerShell実行ポリシー | `-ExecutionPolicy Bypass`を都度指定 (t20_site.ps1と同じ、システム全体のポリシーは変更しない) |
| 開始時刻 | 発走-35分 (許容ウィンドウ31-38分に収まることを狙う、実際の起床遅延は§8で検証) |
| 最大実行時間 | 30分 (`-ExecutionTimeLimit`) |
| 多重起動時の動作 | 無視 (`IgnoreNew`) |
| 失敗時の再試行 | 無し (次のレースタスクは独立、当該レースはそのまま`ok=false`で記録) |
| PCスリープ時の扱い | `-WakeToRun`で起床 |
| 標準出力・エラーログ | `logs\t35_shadow_{date}.log` (Start-Transcript)、EXP05-F固有の失敗は
  `logs\exp05fs_errors.log`にも追記 (`market_snapshot.py`) |
| タスク削除方法 | `Get-ScheduledTask -TaskName 'PyCaLiAI_EXP05FS_T35R_*' | Unregister-ScheduledTask` |

## 登録コマンド・解除コマンド (登録前Gate通過後にユーザー提示、今回は未実行)

登録:
```powershell
.\t35_shadow.ps1 -Schedule
```

解除:
```powershell
Get-ScheduledTask -TaskName 'PyCaLiAI_EXP05FS_T35R_*' | Unregister-ScheduledTask -Confirm:$false
Get-ScheduledTask -TaskName 'PyCaLiAI_EXP05FS_T35' | Unregister-ScheduledTask -Confirm:$false
```

## 登録結果 (2026-09-19実施、当初版)

`PyCaLiAI_T20_Site`と同じトリガー形式で常設マスタータスクを登録した (`New-ScheduledTaskTrigger
-Weekly -DaysOfWeek Saturday,Sunday -At 9am`)。加えて`.\t35_shadow.ps1 -Schedule`を1回実行し、
本日(2026-09-19)残りの9レース分のレース毎タスクも登録済み。

## 2026-09-19 毎日トリガー化 (同日中の修正)

ユーザーが2026-09-21(月・祝、中山・阪神開催)を土日限定トリガーでは取りこぼすと指摘。
モデル・特徴・購入ルールは変更せず、**タスクスケジュールのみ**修正した。

| 項目 | 修正前 | 修正後 |
|---|---|---|
| マスタートリガー | `Weekly -DaysOfWeek Saturday,Sunday` | **`Daily`** (`DaysInterval=1`) |
| no-race日の判定 | `bundle`生成を15:00まで待ってから`exit 1` | `data\weekly\{date}.csv`の有無を即確認、
  無ければ**即`exit 0`**(正常終了) |
| レース毎タスク登録 | 毎回`T35R_*`を全削除→作り直し | 既存タスク名を確認し**冪等**に登録
  (未来時刻の既存は温存、過去時刻の陳腐化タスクだけ削除) |

| 項目 | 値 (2026-09-19修正後) |
|---|---|
| マスタータスク名 | `PyCaLiAI_EXP05FS_T35` |
| トリガー種別 | `MSFT_TaskDailyTrigger`, `DaysInterval=1` (毎日) |
| 次回実行日時 | 2026-09-20 (日) 9:00:00 |
| 実行ユーザー | gutch (Interactive) |
| 作業ディレクトリ | `E:\PyCaLiAI` |
| 多重起動防止 | `MultipleInstances=IgnoreNew` (マスター・レース毎タスクとも確認済み) |
| WakeToRun/StartWhenAvailable | 両方True |
| 実行時間上限 | マスター2時間 / レース毎30分 |
| no-race日の終了コード | **0** (`[no-race] data\weekly\{date}.csv 無し → 本日は開催なしと判断し正常終了`、
  `-Date 20260916`で実測確認済み) |
| ログ出力先 | `logs\t35_shadow_{date}.log` (Start-Transcript) + `logs\exp05fs_errors.log` |
| 冪等性の動作確認 | `-Date 20260919`を2回実行 → 2回目は「新規登録0 / 既存スキップ8 /
  陳腐化削除1」(発走枠を過ぎた1レースだけ削除、残り8件は無変更) |
| 解除コマンド | 上記 (変更なし) |

**2026-09-21(月・祝)個別タスクの確認結果**: 2026-09-19時点で`data\weekly\20260920.csv`
(日曜分)・`data\weekly\20260921.csv`(月曜分)とも未エクスポートのため**現時点では作成できない**
(実データが無く発走時刻を読めないため、実装の不備ではない)。毎日トリガー化により、
該当日朝9:00時点でTARGET出走表エクスポートが完了していれば自動的に登録される
(上記「冪等性の動作確認」と全く同じコードパスで既に動作確認済み)。9:00までに
エクスポートが間に合わなかった場合は、エクスポート後に手動`.\t35_shadow.ps1 -Schedule`
を実行する(t10.ps1の「祝日開催のみ手動」と同じ運用、本番T-10ラインも同じ制約を持つ
EXP05-F固有ではない制約)。

## 本番への影響範囲

`t10_runner.py`/`t20_site_bets.py`/`masters_vote.py`/`compute_bets.py`/
`weekly_nicegui.ps1`のいずれも未変更。共有変更は`jvlink_odds.py`/`forward_prices.py`への
stage追加のみ(`MIXED_COMMIT_NOTE.md`参照)。EXP05-Fのタスクが失敗・多重起動・スリープ未起床
しても、本番T-10/T-20/大会投票ラインのタスクはそれぞれ独立したタスク名・独立した
プロセスなので影響しない。
