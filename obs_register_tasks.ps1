##############################################################
# obs_register_tasks.ps1 --- register / remove the weekly trigger for obs_schedule.ps1
#
# Persistent Windows Task Scheduler change: run only with the user's explicit approval.
# Without -Apply or -Unregister it only prints what it would do.
#
#   .\obs_register_tasks.ps1 -Dry -Apply     first race day: Dry mode (plan v2.1 5.1 start condition)
#   .\obs_register_tasks.ps1 -Apply          after the Dry day passed analysis/obs_dry_check.py
#   .\obs_register_tasks.ps1 -Unregister     remove the weekly trigger and pending OBS one-shots
#
# Weekly trigger: Sat/Sun 09:05 (5 minutes after PyCaLiAI_T10 so the production scheduler starts first).
# Holiday (Monday) race days: run  .\obs_schedule.ps1 -Schedule [-Dry]  by hand, like t10.ps1.
##############################################################
param([switch]$Apply, [switch]$Unregister, [switch]$Dry)
$root = $PSScriptRoot
$name = 'PyCaLiAI_OBS_Schedule'
$args2 = "-NoProfile -ExecutionPolicy Bypass -File `"$(Join-Path $root 'obs_schedule.ps1')`" -Schedule"
if ($Dry) { $args2 += " -Dry" }

if ($Unregister) {
    foreach ($p in @($name, 'PyCaLiAI_OBS_T2R_*', 'PyCaLiAI_OBS_TRIO_*', 'PyCaLiAI_OBS_FINAL_*', 'PyCaLiAI_OBS_REFETCH_*')) {
        Get-ScheduledTask -TaskName $p -ErrorAction SilentlyContinue |
            Unregister-ScheduledTask -Confirm:$false
    }
    Write-Host "[obs] unregistered $name and pending OBS one-shot tasks"
    exit 0
}

Write-Host "[obs] task   : $name"
Write-Host "[obs] trigger: weekly Saturday/Sunday 09:05, WakeToRun, StartWhenAvailable"
Write-Host "[obs] action : powershell.exe $args2"
Write-Host "[obs] workdir: $root"
if (-not $Apply) {
    Write-Host "[obs] dry listing only (no change). Re-run with -Apply after approval."
    exit 0
}
$act = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $args2 -WorkingDirectory $root
$trg = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Saturday, Sunday -At '09:05'
$set = New-ScheduledTaskSettingsSet -StartWhenAvailable -WakeToRun `
    -ExecutionTimeLimit (New-TimeSpan -Hours 7) -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName $name -Action $act -Trigger $trg -Settings $set `
    -Description "Observation plan v2.1 Phase 2 shadow collectors (read-only JV-Link)" -Force | Out-Null
Write-Host "[obs] registered $name"
exit 0
