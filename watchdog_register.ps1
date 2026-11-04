##############################################################
# watchdog_register.ps1 --- タスク "PyCaLiAI_Watchdog" を登録する (2026-11-04)
#
# 土日 09:10 に scripts\task_watchdog.py を起こし、PyCaLiAI_T10 / PyCaLiAI_T20_Site が
# 9:00 に起動していない・失敗した・レース毎タスクが 0 件・bundle が無い ときだけ Discord へ通知する。
# 正常時は無言。開催日でない (data\weekly\{today}.csv が無い) 日は何もしない。
#
#   .\watchdog_register.ps1          # 登録 (既存なら置換)
#   .\watchdog_register.ps1 -Remove  # 削除
#   python scripts\task_watchdog.py --date 20261101 --dry   # 手動テスト
##############################################################
param([switch]$Remove)
Set-Location 'E:\PyCaLiAI'
$name = 'PyCaLiAI_Watchdog'
if ($Remove) {
    Unregister-ScheduledTask -TaskName $name -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "removed $name"
    exit 0
}
$py = (Resolve-Path 'venv311\Scripts\python.exe').Path
$act = New-ScheduledTaskAction -Execute $py -Argument 'scripts\task_watchdog.py' -WorkingDirectory 'E:\PyCaLiAI'
$trg = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Saturday,Sunday -At 09:10
$set = New-ScheduledTaskSettingsSet -WakeToRun -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 5)
Register-ScheduledTask -TaskName $name -Action $act -Trigger $trg -Settings $set `
    -Description 'PyCaLiAI watchdog: 土日09:10 に T-10/T-20 タスクの起動を確認し失敗時だけ Discord' -Force | Out-Null
Get-ScheduledTask -TaskName $name | Select-Object TaskName, State | Format-Table -AutoSize
