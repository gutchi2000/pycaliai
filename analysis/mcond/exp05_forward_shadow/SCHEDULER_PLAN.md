# EXP05-F タスクスケジューラ設計 (spec §10)

**2026-09-19登録、同日中に2段階で修正**:
1. マスタートリガーを土日限定→毎日9:00へ (「2026-09-19 毎日トリガー化」節)
2. 8:30への前倒し + 開催日判定を「weekly CSV有無だけ」から3値判定(NO_RACES_TODAY /
   通常フロー / RACE_DAY_INPUT_PENDING)へ強化 (「2026-09-19 開催日判定の独立化」節、
   2026-09-21月曜祝日開催が「非開催日」と「開催日だがCSV未生成」を区別できないと
   ユーザーが指摘したことへの対応)

`t20_site.ps1`/`t10.ps1`と全く同じ「レース毎タスク」方式を採用する
(`t35_shadow.ps1`実装済み)。理由: 既に本番で実績のある設計をそのまま踏襲することで
新規に検証すべき箇所を最小化する。モデル・特徴・購入ルールは一切変更していない。

## 起動方式 (2026-09-19 二次修正後)

- **毎日**8:30に常設タスク("PyCaLiAI_EXP05FS_T35")が`.\t35_shadow.ps1 -Schedule`を起動
  (曜日を一切見ない、`New-ScheduledTaskTrigger -Daily`)。8:30である理由は「起動時刻」節参照。
- `-Schedule`は`data\weekly\{date}.csv`の存在をまず確認する:
  - **存在する** → 従来どおり`bundle`生成を待ってから`feature_snapshot.py --date`を実行し、
    その日の全レースについて「発走-35分」にWindowsタスクを1つずつ登録する
    (`New-ScheduledTaskTrigger -Once`, `-WakeToRun`, 2時間の実行有効期限)
  - **存在しない** → 「開催日判定の独立化」節の3値判定へ (無条件にexit 0にしない)
- 各レースタスクは発走時刻を過ぎていたら何もせず終了 (t20_site_bets.pyと同じガード)

## 開催日判定の独立化 (2026-09-19、3値判定)

weekly CSVが無い時点で即座に「開催なし」と判定していた旧実装を、以下の3値判定に置き換えた
(モデル・特徴・購入ルールは無変更、判定ロジックのみ)。

**JV-Linkによる将来日程の独立照会は実機検証の結果、不可能と判明した**
(`jvlink_race_day_probe.py`参照): `jvlink_trio_odds.py`が実績を持つ
`JVOpen("RACE", from_ts, 1)`蓄積系dataspecで試したが、当日(確定済みレースがある日)は
検出できるものの、翌日以降の未確定/未来日は`JVOpen`自体が`rc=-1`で失敗する
(option=1/2とも同じ)。つまりこのdataspecは「確定済み過去〜当日」専用で、
「まだ開催されていない将来日の開催予定」を検出する用途には使えない。

代わりに次の判定順を採用した (どちらも新規のJV-Link通信を必要としない、低リスクな方法):

1. `data\jra_known_race_days_override.json`に当日が列挙されているか
   (祝日・振替開催をユーザーから聞いたら追記する台帳。2026-09-21を登録済み)
2. 土曜日または日曜日か (JRAは通常この2日に開催するため)

上記のどちらにも該当しない平日でweekly CSVが無い場合だけ**NO_RACES_TODAY**とし、
個別タスクを作らずexit 0で正常終了する。該当する場合は**RACE_DAY_INPUT_PENDING**とし、
2分間隔で8:55まで再試行してからweekly CSVを再確認する。8:55までに生成されなければ、
これを黙って「開催なし」にせず**明示的にFAIL**する: `logs\exp05fs_errors.log`へ記録し、
`exit 3`(=手動対応が必要というシグナル)を返す。市場スナップショット自体は
この判定より後のレース毎タスクの話なので、この段階ではまだ何も失っていない。

## 冪等なレース毎タスク登録 (2026-09-19修正)

旧実装は毎回`PyCaLiAI_EXP05FS_T35R_*`を全削除してから作り直していた。修正後は
既存タスク一覧を1回取得し:
- 対象レースと同名タスクが既に**未来時刻**で存在 → 触らない(重複登録しない)
- 対象レースと同名タスクが**過去時刻**のまま残存 → 削除するだけ(陳腐化除去、
  発走枠を過ぎているので再登録しない=誤って再実行させない)
- 今回の対象に含まれない(=別日・処理済みの)タスクで、トリガー時刻が過去のもの → 削除
  (将来時刻のものは他日の正当な予約の可能性があるため触らない)

## 起動時刻 (8:30、実測に基づく決定)

直近15開催週の`data/weekly/*.csv`から最速発走時刻を集計した結果、最速は09:40〜09:50
(標本最小値09:40)。T-35ウィンドウ[31,38]分前の**開始**は最速で 09:40-38分=**09:02**。
マスターを8:30起動にすれば、weekly CSVが即座に存在する通常ケースなら
(`feature_snapshot.py`の所要時間 約20-30秒を見ても) 09:02までに十分間に合う。
`RACE_DAY_INPUT_PENDING`の再試行締切を8:55に設定しているのも、09:02より前に
登録処理を終える余地を残すため。9:00起動では猶予が5-15分しかなく不足していたため
8:30へ前倒しした。

## T-35市場取得の優先 (spec: 市場snapshotは捨てない)

`feature_snapshot.py`が生成する特徴量(週次バッチ)が無い・失敗した場合でも、
`market_snapshot.py`が取得できたT-35市場データ自体は失わない。
`predict_and_store.store_prediction()`が`FileNotFoundError`(特徴量snapshot未生成)を
送出した場合、`predict_and_store.store_market_only()`が同じ`exp05fs_predictions/{date}/`
配下へ`{rid}_marketonly_rev{n}.json`として市場snapshotを保存する
(`market_snapshot_saved=true`, `prediction_saved=false`, `invalid_for_primary=true`,
`reason=weekly_input_unavailable`)。`tests/`ではなく
`analysis/mcond/exp05_forward_shadow/test_forward_shadow.py`で動作確認済み。

なお、JV-Link側から当日のレースID・発走時刻を直接取得しT-35タスク登録をweekly CSVの
有無から完全に分離する「理想形」(spec提案の順序)は、上記「開催日判定の独立化」節の
とおりJV-Link将来日程照会が実機で機能しないと判明したため今回は実装していない
(既知の限界として記録)。

## 動作確認 (2026-09-19実施)

| シナリオ | コマンド | 結果 |
|---|---|---|
| 1. 非開催日・CSVなし | `-Schedule -Date 20260916` (水) | `[NO_RACES_TODAY]` exit 0 |
| 2. 開催日・CSVあり | `-Schedule -Date 20260919` (本日) | 個別タスク登録 (冪等) |
| 3. 開催日(既知開催オーバーライド)・CSVなし | `-Schedule -Date 20260921` (月祝) | `[RACE_DAY_INPUT_PENDING]`→再試行→`[FAIL]` exit 3
  (実行時刻が既に8:55を過ぎていたため即FAIL、`logs\exp05fs_errors.log`記録を確認) |
| 3'. 開催日(週末)・CSVなし | `-Schedule -Date 20260920` (日) | 同上パターンで`[RACE_DAY_INPUT_PENDING]`→`[FAIL]` exit 3 |
| 6. 同日2回実行(冪等性) | `-Schedule -Date 20260919`を2回 | 2回目: 新規登録0/既存スキップ8/陳腐化削除1、重複なし |
| 7. 市場取得成功・予測入力失敗 | `pytest test_forward_shadow.py` | `store_market_only`が市場snapshotを保持、
  prediction_saved=falseで区別して保存されることを確認 |

シナリオ4(起動後にCSVが生成される→再試行で検出)は`while (-not (Test-Path ...)) { Sleep }`
という、既存`t20_site.ps1`のbundle待機ループと全く同じ実装パターンを使っている
(2026-06以降本番で実績あり)。実際に2分おきにポーリングしてCSV出現を検出する挙動は
これと同一コードパスなので、既存実装の実績をもって動作を確認済みとする。

## 2026-09-20(日)・2026-09-21(月・祝、中山・阪神)についての確認結果

2026-09-19時点で`data\weekly\20260920.csv`・`data\weekly\20260921.csv`とも未存在
(TARGET未エクスポート)。よって**現時点でこの2日分の個別タスクを実際に作成することは
できない**(存在しないデータから発走時刻を読むことはできないため、実装の不備ではない)。

上記「動作確認」表のとおり、この2日は**もはや「開催なし」に誤判定されない**
(9/20=週末、9/21=既知開催オーバーライド登録済みのため、どちらも`RACE_DAY_INPUT_PENDING`
経路に入り、CSVが無ければ明示的にFAILしてログに残る)。毎日8:30起動のマスターが
自動的にこの判定を行い、CSVが用意されていれば個別タスクを登録し、用意されていなければ
`exp05fs_errors.log`に記録した上でexit 3を返す。後者の場合はTARGET出走表エクスポート後に
手動`-Schedule`を実行する必要がある(既存t10.ps1の「祝日開催のみ手動」と同じ運用、
本番T-10ラインも同じ制約を持つ)。

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
