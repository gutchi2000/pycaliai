##############################################################
# am9_register.ps1 --- タスク "PyCaLiAI_AM9" を登録する (2026-11-13, CW-14)
#
# 土日 09:00 に scripts\am9_snapshot.py を起こし、その日の全レースの単勝・複勝オッズを 1 回だけ保存する。
# 「朝 9 時に売れていて発走前に売れなくなった馬」の前向き確認用 (docs\research\POSTCOVID_SEARCH_20261113.md)。
# 読むだけの収集で、本番の買い目・投票・reports\live_odds には書かない。所要 1〜2 分。
# 開催日でない (data\weekly\{today}.csv が無い) 日は何もしない。祝日開催は手動で 9:00 に実行する。
#
#   .\am9_register.ps1          # 登録 (既存なら置換)
#   .\am9_register.ps1 -Remove  # 削除
#   python scripts\am9_snapshot.py --dry                       # レース一覧の確認だけ
#   python scripts\am9_snapshot.py --date 20261114 --limit 1   # 1 レースだけ試す
##############################################################
param([switch]$Remove)
Set-Location 'E:\PyCaLiAI'
$name = 'PyCaLiAI_AM9'
if ($Remove) {
    Unregister-ScheduledTask -TaskName $name -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "removed $name"
    exit 0
}
$py = (Resolve-Path 'venv311\Scripts\python.exe').Path
$act = New-ScheduledTaskAction -Execute $py -Argument 'scripts\am9_snapshot.py' -WorkingDirectory 'E:\PyCaLiAI'
$trg = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Saturday,Sunday -At 09:00
$set = New-ScheduledTaskSettingsSet -WakeToRun -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 15)
Register-ScheduledTask -TaskName $name -Action $act -Trigger $trg -Settings $set `
    -Description 'PyCaLiAI am9: 土日09:00 に全レースの単複オッズを 1 回保存 (CW-14 前向き確認、読取のみ)' -Force | Out-Null
Get-ScheduledTask -TaskName $name | Select-Object TaskName, State | Format-Table -AutoSize
