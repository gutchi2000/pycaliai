##############################################################
# t35_shadow.ps1 --- EXP05-F 前向き検証用 T-35 市場snapshot起動ラッパー
#
# t10.ps1 / t20_site.ps1 と同じ「レース毎タスク」方式。発走 T-35 (31-38分前ウィンドウを
# 狙う) に各レース 1 個ずつ Windows タスクを登録 (WakeToRun)。実行のたびに:
#   1) analysis.mcond.exp05_forward_shadow.market_snapshot --once <rid> でオッズ取得+検証
#   2) predict_and_store.store_prediction() で凍結モデル(M1/M3/M4)の予測をappend-only保存
#
# 本番 T-10 ライン (t10.ps1) / サイト T-20 (t20_site.ps1) / 大会実投票 (T-4) には
# 一切干渉しない。買い目・印・資金配分への接続もしない (研究専用)。
#
# 運用 (2026-09-19登録、同日中に毎日8:30トリガー+開催日判定の強化へ修正):
#   マスタータスク "PyCaLiAI_EXP05FS_T35" が毎日8:30に -Schedule を起動する
#   (実測: 直近15週の最速発走09:40 → T-35窓の開始は最速09:02。8:30起動なら
#   再試行の余裕を見ても09:02より十分前に登録処理を終えられる)。
#
#   当日の状態を3つに分けて扱う (data/weekly/{date}.csv の有無だけに頼らない):
#     A. NO_RACES_TODAY   : 開催なしと判断 → 個別タスクを作らず exit 0
#     B. (通常フロー)      : weekly CSVあり → 個別T-35タスクを冪等に登録
#     C. RACE_DAY_INPUT_PENDING : 開催があるはず(週末 or 既知開催オーバーライド)なのに
#        weekly CSVがまだ無い → 再試行してから、それでも無ければ明示的にFAIL (exit 3)
#        し exp05fs_errors.log に記録する。これをAと混同して黙って exit 0 にはしない。
#
#   「開催があるはず」の判定は次の順で行う (JV-Linkの将来日程照会は2026-09-19実機検証で
#   不可能と判明、jvlink_race_day_probe.py参照):
#     1. data/jra_known_race_days_override.json に当日が列挙されている
#        (祝日・振替開催をユーザーから聞いたら追記する)
#     2. 土曜/日曜 (JRAは通常この2日に開催する)
#   上記どちらにも当てはまらない平日でweekly CSVが無い場合だけ A (開催なし) とみなす。
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

        # ---- Case C: 開催があるはずなのに weekly CSV がまだ無い → 再試行してから明示的にFAIL ----
        $why = if ($isOverride) { "既知開催オーバーライド" } else { "曜日=$dow(週末)" }
        Write-Host "[RACE_DAY_INPUT_PENDING] $weeklyCsv 無し、しかし $why により開催ありと推定 → 再試行する"
        $retryDeadline = (Get-Date -Hour 8 -Minute 55 -Second 0)
        while (-not (Test-Path $weeklyCsv)) {
            if ((Get-Date) -ge $retryDeadline) {
                $msg = "[RACE_DAY_INPUT_PENDING][FAIL] ${Date}: $why により開催ありと推定されるが " +
                       "$retryDeadline までに $weeklyCsv が生成されなかった。TARGET出走表エクスポートを確認し、" +
                       "エクスポート後に手動で .\t35_shadow.ps1 -Schedule -Date $Date を実行すること。"
                Write-Host $msg
                Add-Content -Path "logs\exp05fs_errors.log" -Value ("{0} {1}" -f (Get-Date -Format 's'), $msg)
                try { Stop-Transcript | Out-Null } catch {}
                exit 3
            }
            Write-Host "[wait] $weeklyCsv 未生成 → 2分後に再確認 (締切 $retryDeadline)"
            Start-Sleep -Seconds 120
        }
        Write-Host "[RACE_DAY_INPUT_PENDING][RESOLVED] $weeklyCsv が生成された → 通常フローへ"
    }

    $bundle = "reports\cowork_input\${Date}_bundle.json"
    $deadline = (Get-Date -Hour 15 -Minute 0 -Second 0)
    while (-not (Test-Path $bundle)) {
        if ((Get-Date) -ge $deadline) {
            Write-Host "[ERROR] 開催表はあるのに15:00までにbundle未生成 → 異常 (Phase A失敗の疑い)"
            try { Stop-Transcript | Out-Null } catch {}
            exit 1
        }
        Write-Host "[wait] $bundle 未生成 → 2 分後に再確認"
        Start-Sleep -Seconds 120
    }
    # 特徴量snapshotも事前に作っておく (市場と無関係、時点安全)
    & $pyFull -m 'analysis.mcond.exp05_forward_shadow.feature_snapshot' --date $Date

    # ---- 冪等なレース毎タスク登録 ----
    # 既存タスク一覧を1回だけ取得し、(a) 今回計算した対象と同名で未来時刻のものは
    # 触らない (重複登録防止)、(b) 過去時刻のまま残っている陳腐化タスクだけ個別に
    # 削除する (誤って再実行されないよう、かつ他日・他レースのタスクは残す)。
    $existing = @{}
    foreach ($t in (Get-ScheduledTask -TaskName 'PyCaLiAI_EXP05FS_T35R_*' -ErrorAction SilentlyContinue)) {
        $info = Get-ScheduledTaskInfo -TaskName $t.TaskName -ErrorAction SilentlyContinue
        $existing[$t.TaskName] = $info
    }

    $lines = & $pyFull -m $mod --date $Date --list-schedule --lead-min $LeadMin
    $n_new = 0; $n_skipped = 0; $n_stale_removed = 0
    $targetNames = @{}
    foreach ($l in $lines) {
        $parts = $l -split "`t"
        if ($parts.Count -lt 2) { continue }
        $rid = $parts[0]; $post = $parts[1]
        $ph, $pm = $post -split ':'
        $runAt = (Get-Date -Hour ([int]$ph) -Minute ([int]$pm) -Second 0).AddMinutes(-$LeadMin)
        $taskName = "PyCaLiAI_EXP05FS_T35R_$rid"
        $targetNames[$taskName] = $true

        if ($runAt -lt (Get-Date)) {
            # 発走枠を過ぎたレース: 既存タスクが残っていれば陳腐化なので削除するだけ (再登録しない)
            if ($existing.ContainsKey($taskName)) {
                Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
                $n_stale_removed++
                Write-Host ("  削除(陳腐化) {0}  (発走枠 {1:HH:mm} は既に経過)" -f $taskName, $runAt)
            }
            continue
        }
        if ($existing.ContainsKey($taskName)) {
            # 既に未来時刻で登録済み → 重複登録しない
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

    Write-Host "[schedule] 新規登録 $n_new / 既存スキップ $n_skipped / 陳腐化削除 $n_stale_removed"
    try { Stop-Transcript | Out-Null } catch {}
    exit 0
}

Write-Host "使い方: .\t35_shadow.ps1 -Schedule  または  .\t35_shadow.ps1 -Once <rid16>"
exit 1
