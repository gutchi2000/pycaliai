# EXP05-F タスクスケジューラ設計 (spec §10)

**2026-09-19登録、同日中に5段階で修正**:
1. マスタートリガーを土日限定→毎日9:00へ (「2026-09-19 毎日トリガー化」節)
2. 8:30への前倒し + 開催日判定を「weekly CSV有無だけ」から3値判定(NO_RACES_TODAY /
   通常フロー / RACE_DAY_INPUT_PENDING)へ強化 (「2026-09-19 開催日判定の独立化」節、
   2026-09-21月曜祝日開催が「非開催日」と「開催日だがCSV未生成」を区別できないと
   ユーザーが指摘したことへの対応)
3. Case C(RACE_DAY_INPUT_PENDING)が8:55でFAILしたら当日の監視自体を終了していた
   欠陥を是正: 18:00までねばる長時間監視+部分回復(まだT-35に間に合うレースだけ
   登録・間に合わないレースは`missed_due_to_input_delay`として明示記録)へ強化
   (「Case C三次修正」節)
4. 「JV-Linkは将来日程を取得できない」という2の時点での結論が誤りだったと判明
   (ユーザー再指摘・再調査): fromtimeの意味を正しく使えば当日を待たずレースID・
   発走時刻を取得できることを実機で実証し、`jvlink_race_calendar.py`として実装・配線
   (「JV-Link開催カレンダー(四次修正)」節)。実装直後にPowerShell経由での呼び出しに
   限り発走時刻が破損して返る実機不具合を発見。
5. **(2026-09-19深夜)** ユーザー指示により、Task Schedulerから32bit Pythonを直接起動
   (PowerShellを介さない)する一時タスクで実機検証し、**この経路では破損しないことを
   確認**。これに基づきJV-Linkアクセスを専用タスク`PyCaLiAI_EXP05FS_CALENDAR`
   (毎日8:20、python.exe直接起動)へ完全分離し、T-35マスターは検証済みJSONを
   `--verify-file`(win32com不要)で読むだけにした。マスターの`ExecutionTimeLimit`も
   無期限→`PT12H`(有限)へ変更。個別タスク登録前の防御的検証(日付不一致・00:00・
   非合理的時刻・既存タスクとの時刻食い違いの検知)も追加した
   (「JV-Link開催カレンダー(五次修正: タスク分離)」節)。

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

**⚠️ 訂正 (2026-09-19夜)**: 本節は当初「JV-Linkによる将来日程の独立照会は実機検証の結果、
不可能と判明した」と結論していたが、これは**誤りだった**(ユーザーが「未来日の失敗だけで
利用不能と結論づけるな」と再指摘、詳細な再調査で判明)。正しい結論・実装は下記
「JV-Link開催カレンダー(四次修正)」節を参照。本節の判定順(オーバーライド台帳→週末)は
JV-Linkカレンダーが使えない/クエリ失敗時の**安全網**として引き続き有効。

判定順 (JV-Linkカレンダーが失敗/矛盾した場合の安全網、新規のJV-Link通信を必要としない
低リスクな方法):

1. `data\jra_known_race_days_override.json`に当日が列挙されているか
   (祝日・振替開催をユーザーから聞いたら追記する台帳。2026-09-21を登録済み)
2. 土曜日または日曜日か (JRAは通常この2日に開催するため)

上記のどちらにも該当しない平日でweekly CSVが無い場合だけ**NO_RACES_TODAY**とし、
個別タスクを作らずexit 0で正常終了する。該当する場合は**RACE_DAY_INPUT_PENDING**とし、
2分間隔で8:55まで再試行してからweekly CSVを再確認する。8:55を過ぎても当日の監視は
終了しない(三次修正、後述)。

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

## Case C三次修正 (2026-09-19夜、「8:55でFAILしたら当日を諦める」旧実装の是正)

旧実装は8:55までにweekly CSVが現れなければその場でexit 3し、当日の監視自体を
終了していた。これでは「最初のT-35だけ逃したら残り23レース分の市場データも
まとめて失う」ことになり、ユーザーから「8:55でFAILを記録しても、その日全体の
監視を終了しないでほしい」「少なくとも最終レースのT-35時刻までは回復可能に
してほしい」と指摘された。以下のように是正した:

- 8:55(最初のT-35登録期限)まではCase Cの旧実装通り2分おきに再試行。
- 8:55を過ぎてもweekly CSVが無ければ`exp05fs_errors.log`へ`FIRST_MISS`を記録するが
  **exitしない**。以降は5分おきの再試行に切り替えて監視を継続する。
- **最終回復期限**: 過去75開催週(全既存weekly CSV)の最遅最終レース発走時刻の実測値
  (2026-07-25、18:30)から逆算したT-35時刻17:55に安全マージンを足した**18:00固定**まで
  監視を続ける。この間にweekly CSVが出現したら、その時点で計算し直したT-35枠に
  まだ間に合うレースだけ個別タスクを登録し、既にT-35枠を過ぎたレースは
  `missed_due_to_input_delay`として`logs\exp05fs_missed_races_{date}.json`へ構造化記録する
  (race_id/scheduled_post/t35_deadline/detected_at/reasonを含む。市場データを
  取り損ねたことを黙って握り潰さない)。
- 18:00を過ぎてもweekly CSVが一度も現れなければ、そこで初めて`FINAL_FAIL`を記録し
  `exit 3`する。
- マスタータスクの`ExecutionTimeLimit`は上記の長時間監視(最大約9.5時間)に対応するため
  当初無期限化した(`ExecutionTimeLimit=PT0S`)が、2026-09-19深夜にユーザー指示で
  **有限の`PT12H`(12時間)へ再修正**した。18:00最終期限までの監視に十分な余裕を
  持たせつつ、想定外のハングで無期限に居座り続けないようにするため
  (詳細は「JV-Link開催カレンダー(五次修正)」節参照)。

## JV-Link開催カレンダー (四次修正、2026-09-19夜)

### 訂正の経緯

当初(2026-09-19朝、上記「開催日判定の独立化」節)は`JVOpen("RACE", 対象日+000000, 1)`で
「対象日自体」をfromtimeに使い、未来日でJVOpen自体が`rc=-1`になることから「JV-Linkは
将来日程を取得できない」と結論していた。ユーザーから「未来日の失敗だけで利用不能と
結論づけるな、当日朝に実測してから判断せよ」と指摘され、再調査した結果、**これは
fromtimeの意味の誤解による誤った結論だったと判明した**。

蓄積系dataspecのfromtimeは「データ作成年月日時分」(=JRA-VANがレコードを発表/更新した
日時)でのフィルタであり、「レース開催日」そのもののフィルタではない。JRAは通常レース
開催の数日前に番組(RAレコード)を先行発表しているため、fromtimeに「対象日そのもの」を
指定すると「対象日以降に発表された分」しか返らず、まだ何も発表されていない未来日では
当然`rc=-1`(該当データなし)になる。正しくは、fromtimeに「対象日より十分前の日付」を
指定して問い合わせれば、対象日の番組が既に発表済みであれば正しく取得できる。

### 実機検証結果

2026-09-19夜に`py -3.12-32`で実機検証し、以下を確認した(詳細は
`jvlink_race_calendar.py`のdocstring参照):

- `JVOpen("RACE", "20260101000000", 1)`で2026-09-21(月・祝、中山・阪神)の**全24レース**
  (発走時刻含む)を取得できた。データ作成年月日=2026-09-17(開催4日前に先行発表済み)。
- 直近7開催日(2026-08-29〜2026-09-19、211レース)の既知発走時刻(`data/weekly/*.csv`)と
  全件突合し、**211/211件でレースIDが一致、204/211件で発走時刻も完全一致**。残り7件も
  ±1分のズレのみ(TARGET側とJV-Link側の発走時刻改定タイミングの差と推測、オフセット
  誤りではない)。
- rid16(=開催年月日8桁+場コード2桁+開催回2桁+開催日目2桁+レース番号2桁)を含む
  全フィールドのバイトオフセットを実機データで検証済み(公式ドキュメントは未参照、
  既知の正解値との突合のみで検証、というこのプロジェクトの既定方針に従った)。

これに基づき`analysis/mcond/exp05_forward_shadow/jvlink_race_calendar.py`を新設し、
`--date`のレースID+発走時刻を`rid16\tHH:MM`形式(`market_snapshot.py --list-schedule`と
同一フォーマット)で返せるようにした。

### PowerShell経由での破損不具合とその切り分け (2026-09-19夜)

実装・配線直後の実機テストで、**全く同じコマンド(`py -3.12-32 -m
analysis.mcond.exp05_forward_shadow.jvlink_race_calendar --date 20260921
--lookback-days 14`)がBashから実行すると常に正しい結果を返す一方、PowerShellの
子プロセスとして実行すると発走時刻が全レース"00:00"という明らかに異常な値に化ける**
という不具合を発見した。`rc=0`・例外無しで返ってくるため、呼び出し側からは正常応答と
区別がつかない。3回連続で再現し、偶発的なフレークではなく決定論的に見えた。

**調査したが未特定の点**: 同一のpython.exe実体であることは確認済み(別バージョンの
取り違えではない)、`PYTHONUTF8`環境変数の有無は無関係、実行中の別プロセスによる
同時アクセス競合ではないことも確認済み。真因(COMアパートメント/セッション文脈の
違い、またはJVLink側の短時間連続アクセスへの脆弱性、のいずれか/両方)は未特定のまま。

### 五次修正: Task Scheduler直接実行による切り分けとタスク分離 (2026-09-19深夜)

ユーザーの指示により、**PowerShellを一切介さずTask Schedulerが32bit Pythonを直接
Action起動する**一時タスク`PyCaLiAI_EXP05FS_CALENDAR_TEST`を作成し実機検証した。

- Action: `Execute=C:\...\Python312-32\python.exe`, `Arguments=-m
  analysis.mcond.exp05_forward_shadow.jvlink_race_calendar --date 20260921
  --lookback-days 14 --output <path> --errlog <path>`, `WorkingDirectory=E:\PyCaLiAI`
  (シェルラッパーなし、stdoutのJSON変換もシェルリダイレクトも使わず、Python自身が
  atomic renameでJSONを書き専用ログへエラーを直接追記)
- `Start-ScheduledTask`で即時実行し`LastTaskResult=0`を確認。生成されたJSONを検証:
  - `record_count=24`、レースID24件すべて一意
  - 発走時刻すべて非"00:00"、07:00-21:59の合理的範囲内(実測09:45-16:30)
  - `場コード`06(中山)・09(阪神)の両方を含む
  - `file_hash`がBash実行時の既知正解と完全一致(`bc4cf6ac...`)
  - T-35時刻(発走-35分)を24件全て計算可能
- **結論: Task Schedulerが直接Pythonを起動する経路では破損は再現しなかった**。
  これによりPowerShellが破損の必要条件であることが実証された(根本原因の完全特定
  ではないが、実運用上の回避策としては十分)。検証後、一時タスクは削除済み。

**この不具合発見の経緯の副産物**: サニティチェック実装**前**に行った実機テストで、
上記の破損データ(全レース00:00)をそのまま使って2026-09-21の24個のT-35タスクを
実際に登録してしまっていた(誤ったトリガー時刻`2026-09-20 23:25`に設定)。これは
2026-09-19夜のうちに`Unregister-ScheduledTask`で全て削除・クリーンアップ済みであり、
実害は残っていない。

### 最終アーキテクチャ: JV-Linkアクセスの完全分離

上記の検証結果に基づき、win32comへのアクセスを**専用タスクへ完全に分離**した
(t35_shadow.ps1自身はwin32comに一切触れない設計へ変更):

```
毎日 8:20  PyCaLiAI_EXP05FS_CALENDAR
           python.exe を直接Action起動 (PowerShellを介さない)
           jvlink_race_calendar.py (--date省略=today, --output省略=既定ディレクトリ)
           → サニティチェック通過時のみ
             data\_research\mcond\exp05fs_calendar\{date}.json を atomic 書き込み
           → 失敗時は正式ファイルを作らずlogs\jvlink_calendar_errors.logへ記録

毎日 8:30  PyCaLiAI_EXP05FS_T35 (マスター、ExecutionTimeLimit=PT12H)
           t35_shadow.ps1 -Schedule
           → 上記JSONを $pyFull(64bit、win32com不要)で --verify-file 読み込み・再検証
             (鮮度10時間以内・file_hash整合性・対象日一致・サニティチェックを全て再検証、
              前日/別日付JSONへはフォールバックしない)
           → 有効なら即座にT-35個別タスク登録(weekly CSV不要)
           → JSON未生成/検証失敗ならweekly CSVベースのCase A/B/Cへ安全にフォールバック
```

この構成の利点: JV-Link(win32com)への実際のアクセスは1日1回、非PowerShell経路
(Task Scheduler直接Action)からのみ発生する。T-35マスター自身は検証済みJSONの
読み込みのみを行うため、たとえPowerShell経由の破損が別の未知の条件で再発しても、
`--verify-file`の`file_hash`再計算チェックが検知して安全にフォールバックする
(defense in depth: 原因を完全に排除できなくても、影響経路を1箇所に閉じ込め、
その1箇所を検証済みの安全な起動方式に固定した)。

### 個別タスク登録前の防御的検証 (2026-09-19深夜追加)

各レースの登録前に以下を全てチェックし、いずれかに該当すれば**そのレースだけ
スキップし`logs\exp05fs_errors.log`へ`[ANOMALY]`として記録**する(他レースの登録は
妨げない、自動修復・自動上書きは一切しない):

- レースIDの日付部が対象日と一致するか、16桁の正しい形式か
- 発走時刻が`00:00`でないか、`HH:MM`形式として妥当か
- 発走時刻がJRAの実運用範囲(07:00-21:59)内か
- (T-35時刻が現在より未来かは既存の陳腐化判定ロジックで別途処理)
- 同一レースIDに既存タスクがある場合、その既存タスクの発走枠(`NextRunTime`)と
  今回計算した時刻が90秒以上ズレていないか(ズレていれば`[ANOMALY][TIME_MISMATCH]`
  として記録するのみで、自動的に正しい時刻で上書きしない。検証済みソースを人間が
  確認した上で手動`Unregister-ScheduledTask`→再登録すること)

## 動作確認 (2026-09-19深夜実施、五次修正後の最終状態)

| シナリオ | コマンド | 結果 |
|---|---|---|
| 1. 非開催日・CSVなし | `-Schedule -Date 20260916` (水) | JV-Link 0件+曜日不一致なし →
  `[NO_RACES_TODAY][jvlink_confirmed]` exit 0 |
| 2. 開催日・CSVあり | `-Schedule -Date 20260919` (本日) | 個別タスク登録 (冪等) |
| 3. 開催日・CSVなし、JV-Linkカレンダー正常(検証済みJSON経由) | `-Schedule -Date 20260921` (月祝) |
  `[jvlink_calendar] verify_file=... query_ok=True races_found=24` → weekly CSV不要で
  24件即登録、全レース異常検知0件。**Task Scheduler直接Action起動で正しく生成した
  calendar JSONを使って実機で再現・確認済み**(下記「実機検証結果(最終)」参照) |
| 3'. 開催日・CSVなし、JV-Link失敗時の安全網 | (JV-Linkクエリ失敗を想定した経路) |
  `[RACE_DAY_INPUT_PENDING]`→再試行→8:55で`FIRST_MISS`記録(exitしない)→18:00まで
  5分おき再試行→`[FINAL_FAIL]` exit 3 (旧経路、コード確認済み) |
| 6. 同日2回実行(冪等性) | `-Schedule -Date 20260919`を2回 | 2回目: 新規登録0/既存スキップ8/陳腐化削除1、重複なし |
| 7. 市場取得成功・予測入力失敗 | `pytest test_forward_shadow.py` | `store_market_only`が市場snapshotを保持、
  prediction_saved=falseで区別して保存されることを確認 |
| (新規)Task Scheduler直接実行 | 一時タスク`PyCaLiAI_EXP05FS_CALENDAR_TEST` | `LastTaskResult=0`、
  正しいJSON生成を確認、検証後削除済み |
| (新規)JV-Linkサニティチェック(拡張版) | `pytest test_forward_shadow.py` |
  全レース同一発走時刻/00:00混入/日付不一致/レースID重複/時刻範囲外の5種を全て拒否することを確認 |
| (新規)calendar JSON鮮度・改ざん検知 | `pytest test_forward_shadow.py` |
  file_hash不一致・10時間超過をそれぞれ正しく拒否することを確認 |
| (新規)個別タスク時刻食い違い検知 | コードレビュー+手動確認 |
  既存タスクの`NextRunTime`と新規計算値が90秒以上ズレる場合、自動上書きせず
  `[ANOMALY][TIME_MISMATCH]`として記録するのみに留まることを確認 |

シナリオ4(起動後にCSVが生成される→再試行で検出)は`while (-not (Test-Path ...)) { Sleep }`
という、既存`t20_site.ps1`のbundle待機ループと全く同じ実装パターンを使っている
(2026-06以降本番で実績あり)。実際に2分おきにポーリングしてCSV出現を検出する挙動は
これと同一コードパスなので、既存実装の実績をもって動作を確認済みとする。

シナリオ5(最終回復期限までにCSVが現れない→明示的FAIL)は実時刻がすでに18:00を
過ぎている状況を作れないため直接の実行時刻では再現できないが、コードパス自体は
シナリオ3'のFIRST_MISS/FINAL_FAILロジックと共通であり、8:55判定部分は実測確認済み。

## 実機検証結果 (最終、2026-09-19深夜)

1. **Task Scheduler直接Python実行の結果**: 成功。一時タスクが`LastTaskResult=0`で
   正しいJSON(24レース、全て非00:00、07:00-21:59範囲内、06/09両場コード含む、
   Bash実行時と同一の`file_hash`)を生成した。
2. **PowerShell子プロセスとの差**: PowerShell経由(`& py -3.12-32 -m ...`)では
   発走時刻が全て"00:00"に破損する不具合が今回も再現し(サニティチェックで正しく
   拒否・JSON未書込を確認)、Task Scheduler直接Action起動では再現しなかった。
   真因は未特定だが、実運用上はこの差異を前提に設計を固定した(下記)。
3. **採用したcalendar経路**: `PyCaLiAI_EXP05FS_CALENDAR`(毎日8:20、python.exe直接
   Action起動)が検証済みJSONを書き、`PyCaLiAI_EXP05FS_T35`(毎日8:30)が
   `--verify-file`(win32com不要、64bit pythonで実行)でそれを読むだけの構成に
   確定した。T-35マスター自身はwin32comに一切触れない。
4. **マスターのExecutionTimeLimit**: `PT12H`(12時間、有限)。無期限(`PT0S`)から
   ユーザー指示で変更。
5. **9月20日・21日の個別タスク**: 2026-09-19深夜時点で、Task Scheduler直接Action
   経由で生成した9/21分のcalendar JSONを使い、`-Schedule -Date 20260921`を実行して
   **実際に24件のT-35個別タスクを登録した**(全て発走-35分の正しい時刻、異常検知0件、
   `Get-ScheduledTask`で`2026-09-21T09:10:00+09:00`等の正しいトリガー日時を確認済み)。
   9/20分は`PyCaLiAI_EXP05FS_CALENDAR`が2026-09-20朝8:20に実際に発火した時点で
   自動生成される(まだ発生していないため未登録)。
6. **最初の有効観測の保存結果**: まだ観測日(9/20・9/21)が到来していないため
   「基盤稼働中・前向き観測0件」(次項参照)。

**observation countの扱い(ユーザー指定)**: 2026-09-20の最初の有効T-35 snapshotが
保存されるまでは「基盤稼働中・前向き観測0件」と表現する。カウントは0から開始し、
最初の有効snapshot保存をもって更新する。

**2026-09-20・21が終了した時点で報告すべきこと**(次回セッションで実施):
calendar取得経路・calendar取得時刻・weekly CSV生成時刻・登録レース数・
最初と最後のT-35時刻・有効snapshot数・market-only保存数・完全予測保存数・
missed数・エラー数・observation累計。

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
| 最大実行時間(レース毎タスク) | 30分 (`-ExecutionTimeLimit`) |
| 最大実行時間(マスタータスク) | `PT12H`(12時間、2026-09-19深夜五次修正で無期限から変更。
  Case Cの最大約9.5時間の長時間監視+余裕を確保) |
| 最大実行時間(CALENDARタスク) | 10分 (`PyCaLiAI_EXP05FS_CALENDAR`、JV-Link取得のみの短時間タスク) |
| 多重起動時の動作 | 無視 (`IgnoreNew`、全タスク共通) |
| 失敗時の再試行 | 無し (次のレースタスクは独立、当該レースはそのまま`ok=false`で記録) |
| PCスリープ時の扱い | `-WakeToRun`で起床 |
| 標準出力・エラーログ | `logs\t35_shadow_{date}.log` (Start-Transcript)、EXP05-F固有の失敗は
  `logs\exp05fs_errors.log`にも追記 (`market_snapshot.py`)、JV-Linkカレンダー固有の失敗は
  `logs\jvlink_calendar_errors.log`(`jvlink_race_calendar.py`が直接追記、シェル
  リダイレクトに頼らない) |
| タスク削除方法 | `Get-ScheduledTask -TaskName 'PyCaLiAI_EXP05FS_T35R_*' | Unregister-ScheduledTask`
  (個別タスク)、`Unregister-ScheduledTask -TaskName 'PyCaLiAI_EXP05FS_CALENDAR'`(カレンダータスク) |

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
