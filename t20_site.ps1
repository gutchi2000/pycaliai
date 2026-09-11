##############################################################
# t20_site.ps1 --- サイト公開用 T-20 買い目プレビュー起動ラッパー
#
# t10.ps1 と同じ「レース毎タスク」方式。発走 T-20 に各レース 1 個ずつ
# Windows タスクを登録 (WakeToRun)。実行のたびに:
#   1) t20_site_bets.py --once <rid> でオッズ取得 + 大会仕様(aite_switch)判定
#   2) reports/masters_vote_site/{date}.json へ書込み
#   3) build_site.py -> sync-hf-umami.ps1 で自動的にサイトへ反映 (確認なし)
#
# 本番 T-10 ライン (t10.ps1) / 大会実投票 (masters_vote.py, T-4) には一切干渉しない。
# Discord 通知はしない。
#
# 運用: タスクスケジューラ "PyCaLiAI_T20_Site" が土日 9:00 に -Schedule を起動。
#
# 手動:
#   .\t20_site.ps1 -Schedule           # 今すぐレース毎タスクを登録 (bundle 必須)
#   .\t20_site.ps1 -Once 2026...11     # 1 レースだけ即処理 (テスト)
#   .\t20_site.ps1 -Once 2026...11 -Dry
##############################################################
param(
    [string]$Date = "",
    [string]$Once = "",
    [double]$LeadMin = 20,
    [switch]$Dry,
    [switch]$Schedule
)
Set-Location 'E:\PyCaLiAI'
$env:PYTHONUTF8 = '1'

$py = 'venv311\Scripts\python.exe'
if (-not (Test-Path $py)) { $py = 'python' }
$pyFull = (Resolve-Path $py).Path

# ---------------------------------------------------------------
# -Once: 1 レース処理 (レース毎タスクの実体 + 手動テスト)
# ---------------------------------------------------------------
if ($Once -ne "") {
    if ($Date -eq "") { $Date = Get-Date -Format 'yyyyMMdd' }
    New-Item -ItemType Directory -Force logs | Out-Null
    try { Start-Transcript -Path ("logs\t20_site_{0}.log" -f $Date) -Append | Out-Null } catch {}
    $argv = @('t20_site_bets.py', $Date, '--once', $Once)
    if ($Dry) { $argv += '--dry' }
    & $pyFull @argv
    $code = $LASTEXITCODE
    try { Stop-Transcript | Out-Null } catch {}
    exit $code
}

# ---------------------------------------------------------------
# -Schedule: レース毎タスクを登録 (bundle 待機つき)
# ---------------------------------------------------------------
if ($Schedule) {
    if ($Date -eq "") { $Date = Get-Date -Format 'yyyyMMdd' }
    New-Item -ItemType Directory -Force logs | Out-Null
    try { Start-Transcript -Path ("logs\t20_site_{0}.log" -f $Date) -Append | Out-Null } catch {}

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

    Get-ScheduledTask -TaskName 'PyCaLiAI_T20R_*' -ErrorAction SilentlyContinue |
        Unregister-ScheduledTask -Confirm:$false

    $lines = & $pyFull 't20_site_bets.py' $Date '--list-schedule' --lead-min $LeadMin
    $n = 0
    foreach ($l in $lines) {
        $parts = $l -split "`t"
        if ($parts.Count -lt 2) { continue }
        $rid = $parts[0]; $post = $parts[1]
        $ph, $pm = $post -split ':'
        $runAt = (Get-Date -Hour ([int]$ph) -Minute ([int]$pm) -Second 0).AddMinutes(-$LeadMin)
        if ($runAt -lt (Get-Date)) { continue }
        $taskName = "PyCaLiAI_T20R_$rid"
        $act = New-ScheduledTaskAction -Execute 'powershell.exe' `
            -Argument ("-NoProfile -ExecutionPolicy Bypass -File E:\PyCaLiAI\t20_site.ps1 " +
                       "-Once $rid -Date $Date -LeadMin $LeadMin") `
            -WorkingDirectory 'E:\PyCaLiAI'
        $trg = New-ScheduledTaskTrigger -Once -At $runAt
        $trg.EndBoundary = $runAt.AddHours(2).ToString("yyyy-MM-ddTHH:mm:ss")
        $set = New-ScheduledTaskSettingsSet -StartWhenAvailable -WakeToRun `
            -ExecutionTimeLimit (New-TimeSpan -Minutes 30) `
            -MultipleInstances IgnoreNew `
            -DeleteExpiredTaskAfter (New-TimeSpan -Hours 6)
        Register-ScheduledTask -TaskName $taskName -Action $act -Trigger $trg `
            -Settings $set -Description "PyCaLiAI T-20 サイト買い目 $rid ($post 発走)" -Force | Out-Null
        $n++
        Write-Host ("  登録 {0}  {1:HH:mm} 処理 → {2} 発走  ({3})" -f $taskName, $runAt, $post, $rid)
    }
    Write-Host "[schedule] $n レースのタスクを登録 (WakeToRun)"
    try { Stop-Transcript | Out-Null } catch {}
    exit 0
}

Write-Host "使い方: .\t20_site.ps1 -Schedule  または  .\t20_site.ps1 -Once <rid16>"
exit 1
