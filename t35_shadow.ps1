##############################################################
# t35_shadow.ps1 --- EXP05-F 前向き検証用 T-35 市場snapshot起動ラッパー
#
# t10.ps1 / t20_site.ps1 と同じ「レース毎タスク」方式。発走 T-35 (31-38分前ウィンドウを
# 狙う) に各レース 1 個ずつ Windows タスクを登録 (WakeToRun)。実行のたびに:
#   1) analysis.mcond.exp05_forward_shadow.market_snapshot --once <rid> でオッズ取得+検証
#   2) predict_and_store.store_prediction() で凍結モデル(M1/M3/M4)の予測をappend-only保存
#      (bundle.json/特徴量snapshotが無ければ store_market_only() で市場だけ保存、後述)
#
# 本番 T-10 ライン (t10.ps1) / サイト T-20 (t20_site.ps1) / 大会実投票 (T-4) には
# 一切干渉しない。買い目・印・資金配分への接続もしない (研究専用)。
#
# 運用 (2026-09-19登録、同日中に5段階で修正):
#   マスタータスク "PyCaLiAI_EXP05FS_T35" が毎日8:30に -Schedule を起動する
#   (実測: 直近15週の最速発走09:40 → T-35窓の開始は最速09:02。8:30起動なら
#   再試行の余裕を見ても09:02より十分前に登録処理を終えられる)。
#   ExecutionTimeLimitは**PT12H(12時間、有限)**(2026-09-19深夜五次修正で無期限から変更、
#   後述「五次修正」参照)。
#
#   当日の状態を3つに分けて扱う (data/weekly/{date}.csv の有無だけに頼らない):
#     A. NO_RACES_TODAY   : 開催なしと判断 → 個別タスクを作らず exit 0
#     B. (通常フロー)      : 開催ありと確認 → 個別T-35タスクを冪等に登録
#     C. RACE_DAY_INPUT_PENDING : 開催があるはずなのにweekly CSVがまだ無い
#        (JV-Link開催カレンダーでも判定できなかった場合の安全網) → 一日を通して
#        回復可能にする(下記「三次修正」参照)。
#
#   四次修正(2026-09-19夜、jvlink_race_calendar.py新設): 当初(2026-09-19朝)は
#   「JVOpen("RACE", 対象日+000000, 1)で未来日はrc=-1になる」ことから
#   「JV-Linkは将来日程を取得できない」と結論していたが、これは fromtime の意味の
#   誤解による誤った結論だった(ユーザー指摘により再調査)。蓄積系dataspecのfromtimeは
#   「データ作成年月日時分」でのフィルタであり「レース開催日」そのもののフィルタでは
#   ない。JRAは通常レース開催の数日前に番組を先行発表しているため、fromtimeに
#   「対象日より十分前の日付」を指定すればまだ来ていない開催日の情報も取得できる。
#   実機で2026-09-21(月・祝、中山・阪神)の全24レースを2026-09-19時点で取得できる
#   ことを実証し(データ作成日2026-09-17=開催4日前に先行発表済み)、さらに直近7開催日
#   211レースの既知発走時刻と突合して211/211件のレースID一致・204/211件で発走時刻も
#   完全一致(残り7件も±1分差のみ)を確認した。詳細・フィールドオフセットは
#   jvlink_race_calendar.py参照。
#
#   五次修正(2026-09-19深夜、PowerShell経由破損の実機切り分けとタスク分離):
#   四次修正の実装直後、PowerShellの子プロセスとしてJV-Linkカレンダーを呼ぶと
#   発走時刻が全レース"00:00"に破損する不具合を発見(rc=0・例外無し)。一時タスク
#   `PyCaLiAI_EXP05FS_CALENDAR_TEST`で「Task Schedulerが32bit Pythonを直接Action起動
#   (PowerShellを介さない)すれば正常に取得できるか」を実機検証し、**成功を確認した**
#   (24レース全て正しい時刻、確認後に一時タスクは削除済み)。これに基づき、
#   win32comへのアクセスを専用タスク`PyCaLiAI_EXP05FS_CALENDAR`(毎日8:20、
#   python.exeを直接起動、PowerShellを一切介さない)へ完全に分離した。この
#   T-35マスタースクリプト自身はwin32comに一切触れず、CALENDARタスクが書き出した
#   検証済みJSON(`data\_research\mcond\exp05fs_calendar\{date}.json`)を
#   `--verify-file`モード(win32com不要、64bit $pyFullで実行可能)で読むだけになった。
#   `--verify-file`は鮮度(10時間以内)・file_hash整合性・対象日一致・サニティチェックを
#   全て再検証するため、前日/別日付のJSONへはフォールバックしない。
#
#   これにより「開催があるはず」の判定は次の優先順で行う:
#     1. `PyCaLiAI_EXP05FS_CALENDAR`タスクが当日8:20に書き出した検証済みJSONを
#        `--verify-file`で再検証して読む(weekly CSVに依存しない独立シグナル)。
#        1件以上あれば即座にBのT-35タスク登録へ(weekly CSVを待たない、spec提案の
#        理想順序「JV-Link検出→T-35収集タスク登録→weekly CSVを待つ→特徴・予測を
#        保存」を実現)。JSON自体が「0件で確定」を記録していた場合は次の
#        ヒューリスティクスと照合し、一致すればA、不一致(食い違い)なら安全側として
#        weekly CSVベースのCase A/B/C判定へフォールバックする。
#     2. JSON未生成/検証失敗(鮮度切れ・ハッシュ不一致・対象日不一致等)の場合のみ、
#        従来の安全網: data/jra_known_race_days_override.json に当日が列挙されている
#        (祝日・振替開催をユーザーから聞いたら追記する) か、土曜/日曜であるかを見る。
#   上記どちらでも開催が見込めない平日でweekly CSVも無い場合だけ A (開催なし) とみなす。
#
#   Cの三次修正(2026-09-19、「8:55でFAILしたら当日を諦める」旧実装の是正。
#   四次修正でJV-Link開催カレンダーが使えるようになったため実際に発動する場面は
#   減った想定だが、JV-Linkクエリ失敗時の安全網として引き続き有効):
#     - 8:55(最初のT-35登録期限)までは2分おきに再試行 (旧実装と同じ)。
#     - 8:55を過ぎてweekly CSVが無くても exit しない。exp05fs_errors.logへ
#       FIRST_MISS を記録した上で、5分おきの再試行へ切り替えて監視を継続する。
#     - 最終回復期限(過去75開催週の最遅最終レース発走18:30の実測に基づき、
#       そのT-35時刻17:55に安全マージンを足した18:00固定)まで監視を続ける。
#     - この間にweekly CSVが出現したら、その時点でまだT-35枠に間に合うレースだけ
#       個別タスクを登録し、既にT-35枠を過ぎたレースは`missed_due_to_input_delay`として
#       `logs\exp05fs_missed_races_{date}.json`へ記録する(市場データを取り損ねたことを
#       黙って握り潰さない)。
#     - 最終回復期限を過ぎてもweekly CSVが無ければ、そこで初めて明示的にFAIL (exit 3)。
#     - bundle.json (reports\cowork_input\{date}_bundle.json) 待ちループは廃止した:
#       predict_and_store.store_prediction()はbundle.json・特徴量snapshotのどちらが
#       無くてもFileNotFoundErrorを送出し、market_snapshot.py側のstore_market_only()が
#       市場snapshotだけを保存する(既に検証済み・非干渉)。よってT-35タスクの「登録」自体を
#       bundle生成完了まで待たせる技術的必要はなく、weekly CSVさえあれば即登録してよい
#       (T-35収集をweekly特徴生成より優先するという方針に合わせた変更)。
#     - ただし store_market_only() が働くのは「個別タスクが実際に作成され、そのタスクが
#       発火してmarket_snapshot.pyが実行された後」に限る。入力遅延でタスク自体を
#       一度も作れなかったレース(=missed_due_to_input_delay)については、
#       市場データも一切取得されない(取りに行くコード自体が動かないため)。
#
#   五次修正の続き(2026-09-19深夜、ExecutionTimeLimitと個別タスク検証の強化):
#     - マスタータスクのExecutionTimeLimitを**無期限からPT12H(12時間)へ変更**した。
#       8:30起動+18:00最終回復期限=最大約9.5時間の監視に12時間なら十分な余裕があり、
#       かつ想定外のハング(例: JVLink呼び出しが応答不能になった等)が起きても
#       無期限に居座り続けず、翌日8:30の次回起動(IgnoreNewで多重起動は防止済み)を
#       妨げない設計にした。
#     - 個別タスク登録前の検証を強化(下記「冪等なレース毎タスク登録」節参照):
#       発走時刻が空/00:00/非合理的範囲でないか、T-35時刻が現在より未来か、
#       既存タスクがあれば時刻が一致するか(不一致は異常として扱い自動上書きしない)
#       を全てチェックしてから登録する。
#
# 手動:
#   .\t35_shadow.ps1 -Schedule           # 今すぐ判定・登録を実行
#   .\t35_shadow.ps1 -Once 2026...11     # 1レースだけ即処理 (テスト)
#   .\t35_shadow.ps1 -Once 2026...11 -Dry
##############################################################
param(
    [string]$Date = "",
    [string]$Once = "",
    [double]$LeadMin = 35,
    [switch]$Dry,
    [switch]$Schedule
)
Set-Location 'E:\PyCaLiAI'
$env:PYTHONUTF8 = '1'

$py = 'venv311\Scripts\python.exe'
if (-not (Test-Path $py)) { $py = 'python' }
$pyFull = (Resolve-Path $py).Path
$mod = 'analysis.mcond.exp05_forward_shadow.market_snapshot'

if ($Once -ne "") {
    if ($Date -eq "") { $Date = Get-Date -Format 'yyyyMMdd' }
    New-Item -ItemType Directory -Force logs | Out-Null
    try { Start-Transcript -Path ("logs\t35_shadow_{0}.log" -f $Date) -Append | Out-Null } catch {}
    $argv = @('-m', $mod, '--date', $Date, '--once', $Once, '--lead-min', $LeadMin)
    if ($Dry) { $argv += '--dry' }
    & $pyFull @argv
    $code = $LASTEXITCODE
    try { Stop-Transcript | Out-Null } catch {}
    exit $code
}

if ($Schedule) {
    if ($Date -eq "") { $Date = Get-Date -Format 'yyyyMMdd' }
    New-Item -ItemType Directory -Force logs | Out-Null
    try { Start-Transcript -Path ("logs\t35_shadow_{0}.log" -f $Date) -Append | Out-Null } catch {}

    # ---- 開催日判定 (data/weekly/{date}.csvの有無だけに頼らない) ----
    $weeklyCsv = "data\weekly\${Date}.csv"

    function Test-KnownRaceDayOverride([string]$d) {
        $p = "data\jra_known_race_days_override.json"
        if (-not (Test-Path $p)) { return $false }
        try {
            $ov = Get-Content $p -Raw -Encoding UTF8 | ConvertFrom-Json
            return ($ov.known_race_days.PSObject.Properties.Name -contains $d)
        } catch { return $false }
    }
    function Get-DowJa([string]$d) {
        try { return ([datetime]::ParseExact($d, 'yyyyMMdd', $null)).DayOfWeek } catch { return $null }
    }
    function Add-MissedRace([string]$d, [string]$rid, [string]$post, [datetime]$runAt, [string]$reason) {
        # weekly CSV到着遅延のため一度もT-35タスクを作れなかったレースを記録する
        # (市場データも一切取得できていない=完全な欠損。黙って握り潰さない)。
        $p = "logs\exp05fs_missed_races_${d}.json"
        $arr = @()
        if (Test-Path $p) {
            try { $arr = @(Get-Content $p -Raw -Encoding UTF8 | ConvertFrom-Json) } catch { $arr = @() }
        }
        $arr = @($arr) + [PSCustomObject]@{
            race_id = $rid; date = $d; scheduled_post = $post
            t35_deadline = $runAt.ToString('s'); detected_at = (Get-Date -Format 's')
            reason = $reason
        }
        New-Item -ItemType Directory -Force logs | Out-Null
        ($arr | ConvertTo-Json -Depth 5) | Set-Content -Path $p -Encoding UTF8
    }

    $wentThroughInputDelay = $false
    $lines = $null

    # ---- JV-Link開催カレンダー (weekly CSVに依存しない独立シグナル、2026-09-19夜追加) ----
    # 対象日の番組が既に発表済みなら、weekly CSVの有無に関わらずレースID+発走時刻を
    # 事前に取得できる(実機で211/211件のrid一致・204/211件で完全時刻一致まで検証済み)。
    #
    # 2026-09-19深夜、重要な構成変更: 当初はこのPowerShellスクリプト自身が
    # `py -3.12-32 -m jvlink_race_calendar --date ...`でJV-Link(win32com)を直接呼んでいたが、
    # これがまさにPowerShell経由でのみ発走時刻が"00:00"に破損する不具合の発生経路そのもの
    # だった。実機検証(一時タスクPyCaLiAI_EXP05FS_CALENDAR_TESTで確認)により、Task
    # Schedulerが32bit Pythonを直接Action起動する経路(PowerShellを一切介さない)では
    # 問題なく正しい値が返ることを確認したため、win32comへのアクセス自体を専用タスク
    # `PyCaLiAI_EXP05FS_CALENDAR`(毎日8:20、python.exeを直接起動)へ完全に分離した。
    # このT-35マスタースクリプトは、CALENDARタスクが既に書き出した検証済みJSON
    # (`data\_research\mcond\exp05fs_calendar\{date}.json`)を`--verify-file`モードで
    # 読むだけであり、win32comには一切触れない(64bitの$pyFullで実行可能・
    # PowerShellの子プロセスからwin32comを呼ぶ経路がそもそも存在しなくなった)。
    # `--verify-file`は鮮度(既定10時間以内)・file_hash整合性・対象日一致・
    # サニティチェックを全て再検証してから返すため、前日/別日付のJSONへは
    # フォールバックしない。JSON未生成/検証失敗時は既存のweekly CSVベースの
    # Case A/B/C判定へ安全にフォールバックする(このステップ自体が主フローを止めることはない)。
    $jvRaces = @()
    $jvQueryOk = $false
    $calendarJson = "data\_research\mcond\exp05fs_calendar\${Date}.json"
    try {
        $jvOut = & $pyFull -m 'analysis.mcond.exp05_forward_shadow.jvlink_race_calendar' `
            --verify-file $calendarJson --date $Date 2>$null
        $jvExit = $LASTEXITCODE
        $jvQueryOk = ($jvExit -eq 0)
        if ($jvQueryOk) {
            foreach ($l in $jvOut) {
                $parts = $l -split "`t"
                if ($parts.Count -ge 2) { $jvRaces += [PSCustomObject]@{ rid = $parts[0]; post = $parts[1] } }
            }
        }
    } catch {
        $jvQueryOk = $false
    }
    Write-Host ("[jvlink_calendar] verify_file={0} query_ok={1} races_found={2}" -f $calendarJson, $jvQueryOk, $jvRaces.Count)

    $useJvSchedule = $false
    if ($jvQueryOk -and $jvRaces.Count -gt 0) {
        Write-Host "[jvlink_calendar] $($jvRaces.Count)件のレースを検出 → weekly CSVの有無に関わらずT-35タスクを直ちに登録する(理想順序)"
        $useJvSchedule = $true
    } elseif ($jvQueryOk -and $jvRaces.Count -eq 0) {
        # JV-Linkが「0件」と明確に回答した。既存の曜日/オーバーライドヒューリスティクスと
        # 一致すれば独立シグナル2つの合意としてNO_RACES_TODAYを確信を持って判定できる。
        # 不一致なら食い違いなので安全側(weekly CSVベースのCase A/B/C)へフォールバックする。
        $dow0 = Get-DowJa $Date
        $expectRace0 = (Test-KnownRaceDayOverride $Date) -or
                       ($dow0 -eq [System.DayOfWeek]::Saturday) -or ($dow0 -eq [System.DayOfWeek]::Sunday)
        if (-not $expectRace0) {
            Write-Host "[NO_RACES_TODAY][jvlink_confirmed] JV-Link開催カレンダーも0件、曜日=$dow0、オーバーライドにも無し → 開催なしと確信して正常終了"
            try { Stop-Transcript | Out-Null } catch {}
            exit 0
        }
        Write-Host "[warn][jvlink_disagreement] JV-Linkは0件と回答したが曜日=$dow0/オーバーライドにより開催ありと推定 → 食い違いのためweekly CSVベース判定へフォールバック"
    } else {
        Write-Host "[jvlink_calendar] クエリ失敗 → weekly CSVベースの判定へフォールバック"
    }

    if ($useJvSchedule) {
        $lines = $jvRaces | ForEach-Object { "$($_.rid)`t$($_.post)" }
    } else {
        # ---- weekly CSVベースのCase A/B/C判定 (JV-Linkで確定できなかった場合の安全網) ----
        if (-not (Test-Path $weeklyCsv)) {
            $dow = Get-DowJa $Date
            $isOverride = Test-KnownRaceDayOverride $Date
            $isWeekend = ($dow -eq [System.DayOfWeek]::Saturday) -or ($dow -eq [System.DayOfWeek]::Sunday)
            $expectRace = $isOverride -or $isWeekend

            if (-not $expectRace) {
                Write-Host "[NO_RACES_TODAY] $weeklyCsv 無し、曜日=$dow、既知開催オーバーライドにも無し → 開催なしと判断し正常終了"
                try { Stop-Transcript | Out-Null } catch {}
                exit 0
            }

            # ---- Case C: 開催があるはずなのに weekly CSV がまだ無い ----
            # 8:55(最初のT-35登録期限)まではFAIL可能性を見て2分おきに再試行。それを過ぎても
            # exitせず、18:00(過去75開催週の最遅最終レース発走18:30のT-35時刻17:55+安全マージン)
            # まで5分おきの再試行を続ける。1日を通してweekly CSVが一度も現れなければそこで
            # 初めて明示的にFAILする(=手動対応が必要というシグナル)。
            $wentThroughInputDelay = $true
            $why = if ($isOverride) { "既知開催オーバーライド" } else { "曜日=$dow(週末)" }
            Write-Host "[RACE_DAY_INPUT_PENDING] $weeklyCsv 無し、しかし $why により開催ありと推定 → 再試行する"
            $firstDeadline = (Get-Date -Hour 8 -Minute 55 -Second 0)
            $finalDeadline = (Get-Date -Hour 18 -Minute 0 -Second 0)
            $firstMissLogged = $false
            while (-not (Test-Path $weeklyCsv)) {
                if ((Get-Date) -ge $firstDeadline -and -not $firstMissLogged) {
                    $msg = "[RACE_DAY_INPUT_PENDING][FIRST_MISS] ${Date}: $why により開催ありと推定されるが " +
                           "$firstDeadline (最初のT-35登録期限)までに $weeklyCsv が生成されなかった。" +
                           "後続レースの回収は $finalDeadline まで諦めずに継続する。"
                    Write-Host $msg
                    Add-Content -Path "logs\exp05fs_errors.log" -Value ("{0} {1}" -f (Get-Date -Format 's'), $msg)
                    $firstMissLogged = $true
                }
                if ((Get-Date) -ge $finalDeadline) {
                    $msg = "[RACE_DAY_INPUT_PENDING][FINAL_FAIL] ${Date}: $why により開催ありと推定されるが " +
                           "本日の最終回復期限 $finalDeadline までに $weeklyCsv が生成されなかった。" +
                           "TARGET出走表エクスポートを確認し、エクスポート後に手動で " +
                           ".\t35_shadow.ps1 -Schedule -Date $Date を実行すること " +
                           "(本日分は個別タスク0件のまま終了、市場データも一切取得できていない)。"
                    Write-Host $msg
                    Add-Content -Path "logs\exp05fs_errors.log" -Value ("{0} {1}" -f (Get-Date -Format 's'), $msg)
                    try { Stop-Transcript | Out-Null } catch {}
                    exit 3
                }
                $sleepSec = if ($firstMissLogged) { 300 } else { 120 }
                Write-Host "[wait] $weeklyCsv 未生成 → ${sleepSec}秒後に再確認 (最終回復期限 $finalDeadline)"
                Start-Sleep -Seconds $sleepSec
            }
            $csvMtime = (Get-Item $weeklyCsv).LastWriteTime
            Write-Host "[RACE_DAY_INPUT_PENDING][RESOLVED] $weeklyCsv 生成時刻(mtime)=$csvMtime 検出時刻=$(Get-Date) → 通常フローへ"
        }
        $lines = & $pyFull -m $mod --date $Date --list-schedule --lead-min $LeadMin
    }

    # bundle.json (Phase A の印付け結果) は待たない: predict_and_store.store_prediction()は
    # bundle.json・特徴量snapshotのどちらが未生成でもFileNotFoundErrorを送出し、
    # market_snapshot.py側のstore_market_only()が市場snapshotだけを保存する経路が
    # 既にある(非干渉・検証済み)。T-35収集をweekly特徴生成より優先するため、
    # ここでbundle生成を待つ技術的必要はない(2026-09-19三次修正で待機ループを廃止)。
    $bundle = "reports\cowork_input\${Date}_bundle.json"
    if (-not (Test-Path $bundle)) {
        Write-Host "[info] $bundle 未生成(市場のみ保存になる可能性あり、store_market_only()で捕捉される想定)"
    }
    # 特徴量snapshotも事前に作っておく (weekly CSVが既にあれば実行、市場とは無関係・
    # 時点安全。失敗しても後続の個別タスク登録自体は妨げない=市場収集を優先する)
    if (Test-Path $weeklyCsv) {
        & $pyFull -m 'analysis.mcond.exp05_forward_shadow.feature_snapshot' --date $Date
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[warn] feature_snapshot.py 失敗 (exit $LASTEXITCODE) → 各レースはstore_market_only()にフォールバックする見込み"
        }
    } else {
        Write-Host "[info] $weeklyCsv 未生成のためfeature_snapshotは未実行 (store_market_only()にフォールバックする見込み)"
    }

    # ---- 冪等なレース毎タスク登録 ----
    # 既存タスク一覧を1回だけ取得し、(a) 今回計算した対象と同名で未来時刻のものは
    # 触らない (重複登録防止)、(b) 過去時刻のまま残っている陳腐化タスクだけ個別に
    # 削除する (誤って再実行されないよう、かつ他日・他レースのタスクは残す)。
    $existing = @{}
    foreach ($t in (Get-ScheduledTask -TaskName 'PyCaLiAI_EXP05FS_T35R_*' -ErrorAction SilentlyContinue)) {
        $info = Get-ScheduledTaskInfo -TaskName $t.TaskName -ErrorAction SilentlyContinue
        $existing[$t.TaskName] = $info
    }

    # $Date の暦日を基準に使う (Get-Date -Hour -Minute だけでは「今日」の日付が
    # 暗黙に残ってしまい、$Date が実際の今日と異なる場合に誤った日で計算してしまう
    # バグがあった。JV-Link開催カレンダー経由で「今日ではない対象日」の登録が
    # 実際に到達可能になったため2026-09-19夜に修正: $Date を明示的な基準日とする)
    $dateBase = [datetime]::ParseExact($Date, 'yyyyMMdd', $null)

    $n_new = 0; $n_skipped = 0; $n_stale_removed = 0; $n_missed = 0; $n_anomaly = 0
    $targetNames = @{}
    foreach ($l in $lines) {
        $parts = $l -split "`t"
        if ($parts.Count -lt 2) { continue }
        $rid = $parts[0]; $post = $parts[1]

        # ---- 登録前の防御的検証 (2026-09-19深夜追加、ユーザー指摘) ----
        # 誤ったタスク(日付不一致・00:00・非合理的時刻)を絶対に作らない。異常を検出
        # したら記録してこのレースだけスキップする(他レースの登録は妨げない)。
        if ($rid.Length -ne 16 -or $rid.Substring(0, 8) -ne $Date) {
            $msg = "[ANOMALY][RID_DATE_MISMATCH] '$rid' の日付部が対象日 $Date と不一致、または16桁でない → このレースはスキップ"
            Write-Host $msg
            Add-Content -Path "logs\exp05fs_errors.log" -Value ("{0} {1}" -f (Get-Date -Format 's'), $msg)
            $n_anomaly++
            continue
        }
        if ($post -notmatch '^\d{1,2}:\d{2}$' -or $post -eq '00:00') {
            $msg = "[ANOMALY][BAD_POST_TIME] $rid の発走時刻 '$post' が不正(00:00または形式不正) → このレースはスキップ"
            Write-Host $msg
            Add-Content -Path "logs\exp05fs_errors.log" -Value ("{0} {1}" -f (Get-Date -Format 's'), $msg)
            $n_anomaly++
            continue
        }
        $ph, $pm = $post -split ':'
        $phI = [int]$ph
        if ($phI -lt 7 -or $phI -gt 21) {
            $msg = "[ANOMALY][POST_TIME_OUT_OF_RANGE] $rid の発走時刻 '$post' がJRA実運用範囲外(07:00-21:59) → このレースはスキップ"
            Write-Host $msg
            Add-Content -Path "logs\exp05fs_errors.log" -Value ("{0} {1}" -f (Get-Date -Format 's'), $msg)
            $n_anomaly++
            continue
        }

        $runAt = $dateBase.AddHours($phI).AddMinutes([int]$pm).AddMinutes(-$LeadMin)
        $taskName = "PyCaLiAI_EXP05FS_T35R_$rid"
        $targetNames[$taskName] = $true

        if ($runAt -lt (Get-Date)) {
            if ($existing.ContainsKey($taskName)) {
                # 既にタスクが動いた(または動く機会があった)後の通常の陳腐化削除
                Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
                $n_stale_removed++
                Write-Host ("  削除(陳腐化) {0}  (発走枠 {1:HH:mm} は既に経過)" -f $taskName, $runAt)
            } elseif ($wentThroughInputDelay) {
                # weekly CSV到着遅延のため一度もタスクを作れないままT-35枠を過ぎた
                # → 市場データも一切取得できていない完全な欠損。記録して次のレースへ
                Add-MissedRace $Date $rid $post $runAt "weekly_csv_delayed"
                $n_missed++
                Write-Host ("  [MISSED] {0}  T-35枠 {1:HH:mm} は入力遅延のため回収不能 (発走 {2})" -f $rid, $runAt, $post)
            }
            continue
        }
        if ($existing.ContainsKey($taskName)) {
            # 既に未来時刻で登録済み(想定) → 重複登録しない。ただし「同一race IDに
            # 既存タスクがあるが記録されている発走枠が今回の計算と食い違う」場合は
            # 誤登録の疑いがあるため、無条件にスキップせず異常として扱う
            # (2026-09-19深夜追加、ユーザー指摘: race IDが同じで時刻が違う場合は
            # 自動的に正しい時刻で上書きせず、異常ログを残し検証済みソースがある
            # 場合だけ手動で削除・再登録すること)。
            $existingNext = $existing[$taskName].NextRunTime
            $diffSec = $null
            if ($existingNext) { $diffSec = [math]::Abs((New-TimeSpan -Start $existingNext -End $runAt).TotalSeconds) }
            if ($null -ne $diffSec -and $diffSec -gt 90) {
                $msg = "[ANOMALY][TIME_MISMATCH] ${taskName}: 既存タスクの発走枠 $existingNext と " +
                       "今回算出した $runAt が${diffSec}秒ズレている(発走 $post)。誤登録の疑いのため" +
                       "自動上書きしない。検証済みソース(JV-LinkカレンダーJSON等)を確認の上、" +
                       "手動で Unregister-ScheduledTask 後に再登録すること。"
                Write-Host $msg
                Add-Content -Path "logs\exp05fs_errors.log" -Value ("{0} {1}" -f (Get-Date -Format 's'), $msg)
                $n_anomaly++
                continue
            }
            $n_skipped++
            Write-Host ("  既存(スキップ) {0}  {1:HH:mm} 処理 → {2} 発走" -f $taskName, $runAt, $post)
            continue
        }
        $act = New-ScheduledTaskAction -Execute 'powershell.exe' `
            -Argument ("-NoProfile -ExecutionPolicy Bypass -File E:\PyCaLiAI\t35_shadow.ps1 " +
                       "-Once $rid -Date $Date -LeadMin $LeadMin") `
            -WorkingDirectory 'E:\PyCaLiAI'
        $trg = New-ScheduledTaskTrigger -Once -At $runAt
        $trg.EndBoundary = $runAt.AddHours(2).ToString("yyyy-MM-ddTHH:mm:ss")
        $set = New-ScheduledTaskSettingsSet -StartWhenAvailable -WakeToRun `
            -ExecutionTimeLimit (New-TimeSpan -Minutes 30) `
            -MultipleInstances IgnoreNew `
            -DeleteExpiredTaskAfter (New-TimeSpan -Hours 6)
        Register-ScheduledTask -TaskName $taskName -Action $act -Trigger $trg `
            -Settings $set -Description "EXP05-F T-35 市場snapshot $rid ($post 発走)" -Force | Out-Null
        $n_new++
        Write-Host ("  登録 {0}  {1:HH:mm} 処理 → {2} 発走  ({3})" -f $taskName, $runAt, $post, $rid)
    }

    # 今回の対象に無い「別日の取り残し」等、$targetNames に無い陳腐化タスクも掃除する
    # (発走枠が過去のものだけ、将来枠は誤って触らない)
    foreach ($name in $existing.Keys) {
        if ($targetNames.ContainsKey($name)) { continue }
        $t = Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
        if ($null -eq $t) { continue }
        $trigStart = $null
        try { $trigStart = [datetime]$t.Triggers[0].StartBoundary } catch {}
        if ($null -ne $trigStart -and $trigStart -lt (Get-Date)) {
            Unregister-ScheduledTask -TaskName $name -Confirm:$false -ErrorAction SilentlyContinue
            $n_stale_removed++
            Write-Host ("  削除(陳腐化・対象外日) {0}" -f $name)
        }
    }

    Write-Host "[schedule] 新規登録 $n_new / 既存スキップ $n_skipped / 陳腐化削除 $n_stale_removed / 入力遅延で取りこぼし $n_missed / 異常検知 $n_anomaly"
    try { Stop-Transcript | Out-Null } catch {}
    exit 0
}

Write-Host "使い方: .\t35_shadow.ps1 -Schedule  または  .\t35_shadow.ps1 -Once <rid16>"
exit 1
