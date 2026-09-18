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
# 運用: タスクスケジューラへの自動起動登録は未実施 (2026-09-19)。ユーザー確認の上で
#   t20_site.ps1 と同様に "PyCaLiAI_EXP05FS_T35" を土日9:00等に登録することを推奨。
#
# 手動:
#   .\t35_shadow.ps1 -Schedule           # 今すぐレース毎タスクを登録 (bundle必須)
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

    $bundle = "reports\cowork_input\${Date}_bundle.json"
    $deadline = (Get-Date -Hour 15 -Minute 0 -Second 0)
    while (-not (Test-Path $bundle)) {
        if ((Get-Date) -ge $deadline) {
            Write-Host "[ERROR] 15:00 までに bundle 未生成 → 諦め (開催日でない/Phase A 未実行)"
            try { Stop-Transcript | Out-Null } catch {}
            exit 1
        }
        Write-Host "[wait] $bundle 未生成 → 2 分後に再確認"
        Start-Sleep -Seconds 120
    }
    # 特徴量snapshotも事前に作っておく (市場と無関係、時点安全)
    & $pyFull -m 'analysis.mcond.exp05_forward_shadow.feature_snapshot' --date $Date

    Get-ScheduledTask -TaskName 'PyCaLiAI_EXP05FS_T35R_*' -ErrorAction SilentlyContinue |
        Unregister-ScheduledTask -Confirm:$false

    $lines = & $pyFull -m $mod --date $Date --list-schedule --lead-min $LeadMin
    $n = 0
    foreach ($l in $lines) {
        $parts = $l -split "`t"
        if ($parts.Count -lt 2) { continue }
        $rid = $parts[0]; $post = $parts[1]
        $ph, $pm = $post -split ':'
        $runAt = (Get-Date -Hour ([int]$ph) -Minute ([int]$pm) -Second 0).AddMinutes(-$LeadMin)
        if ($runAt -lt (Get-Date)) { continue }
        $taskName = "PyCaLiAI_EXP05FS_T35R_$rid"
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
        $n++
        Write-Host ("  登録 {0}  {1:HH:mm} 処理 → {2} 発走  ({3})" -f $taskName, $runAt, $post, $rid)
    }
    Write-Host "[schedule] $n レースのタスクを登録 (WakeToRun)"
    try { Stop-Transcript | Out-Null } catch {}
    exit 0
}

Write-Host "使い方: .\t35_shadow.ps1 -Schedule  または  .\t35_shadow.ps1 -Once <rid16>"
exit 1
