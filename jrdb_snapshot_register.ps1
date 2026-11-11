##############################################################
# jrdb_snapshot_register.ps1 --- タスク "PyCaLiAI_JRDB_Snapshot" を登録する (2026-11-12)
#
# JRDB の前日ファイル (KYI) を発走前に保存する (収集待ち台帳 CW-12)。
#   金・土 19:40  翌日ぶんが 19:00 に出るので、その直後に取る
#   土・日 07:30  取りこぼしの保険。開催日なのに保存が無ければ Discord へ通知
# 祝日開催は前日 19:40 のトリガーが無いので、当日朝に手動で: python scripts\jrdb_snapshot.py capture
#
#   .\jrdb_snapshot_register.ps1          # 登録 (既存なら置換)
#   .\jrdb_snapshot_register.ps1 -Remove  # 削除
#   python scripts\jrdb_snapshot.py status
##############################################################
param([switch]$Remove)
Set-Location 'E:\PyCaLiAI'
$name = 'PyCaLiAI_JRDB_Snapshot'
if ($Remove) {
    Unregister-ScheduledTask -TaskName $name -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "removed $name"
    exit 0
}
$py = (Resolve-Path 'venv311\Scripts\python.exe').Path
$act = New-ScheduledTaskAction -Execute $py -Argument 'scripts\jrdb_snapshot.py capture --notify' -WorkingDirectory 'E:\PyCaLiAI'
$trg = @(
    (New-ScheduledTaskTrigger -Weekly -DaysOfWeek Friday,Saturday -At 19:40),
    (New-ScheduledTaskTrigger -Weekly -DaysOfWeek Saturday,Sunday -At 07:30)
)
$set = New-ScheduledTaskSettingsSet -WakeToRun -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 10)
Register-ScheduledTask -TaskName $name -Action $act -Trigger $trg -Settings $set `
    -Description 'PyCaLiAI: JRDB 前日ファイル KYI を発走前に保存 (CW-12)' -Force | Out-Null
Get-ScheduledTask -TaskName $name | Select-Object TaskName, State | Format-Table -AutoSize
