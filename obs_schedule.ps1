##############################################################
# obs_schedule.ps1 --- Observation plan v2.1 Phase 2 shadow collectors
#
# SHADOW ONLY. Reads JV-Link and appends to data/forward_prices (schema v2) and
# data/jvlink_fetch_journal. Never touches bets, votes, masters_vote, live_odds,
# Discord, the site, git or HF.
#
#   -Schedule        wait for the day's bundle, then register one-shot WakeToRun tasks:
#                      PyCaLiAI_OBS_T2R_<rid>    at post - T2LeadMin   (-T2 <rid>)
#                      PyCaLiAI_OBS_TRIO_<rid>   at post - TrioLeadMin (-Trio <rid>)
#                      PyCaLiAI_OBS_FINAL_<date>   today 18:30   (-Final)
#                      PyCaLiAI_OBS_REFETCH_<date> next day 20:00 (-Refetch)
#   -T2 <rid>        jvlink_obs.py --race <rid> --stage t2_candidate (0B31..0B35, one process)
#   -Trio <rid>      jvlink_trio_odds.py --race <rid>  (0B35 + 0B31, T-10)
#   -Final           jvlink_obs.py --final-rt <date>, then --final-stock <date>
#   -Refetch         jvlink_obs.py --final-stock <date>  (next-day immutability check)
#   -Dry             everything goes to *_dry stores (not counted for Stage 0 / 500R)
#
# The daily 9:00 trigger for -Schedule is registered by obs_register_tasks.ps1 (needs approval).
##############################################################
param(
    [string]$Date = "",
    [string]$T2 = "",
    [string]$Trio = "",
    [string]$Post = "",
    [switch]$Schedule,
    [switch]$Final,
    [switch]$Refetch,
    [switch]$Dry,
    [double]$T2LeadMin = 2,
    [double]$TrioLeadMin = 10
)
$root = $PSScriptRoot
Set-Location $root
$env:PYTHONUTF8 = '1'
if ($Date -eq "") { $Date = Get-Date -Format 'yyyyMMdd' }
$dryArg = @()
if ($Dry) { $dryArg = @('--dry') }
$dryPs = ""
if ($Dry) { $dryPs = " -Dry" }
New-Item -ItemType Directory -Force (Join-Path $root 'logs') | Out-Null
$log = Join-Path $root ("logs\obs_{0}{1}.log" -f $Date, $(if ($Dry) { '_dry' } else { '' }))
try { Start-Transcript -Path $log -Append | Out-Null } catch {}

function Invoke-Py32([string[]]$argv) {
    & py -3.12-32 @argv
    return $LASTEXITCODE
}

if ($T2 -ne "") {
    $argv = @('jvlink_obs.py', '--race', $T2, '--stage', 't2_candidate', '--date', $Date) + $dryArg
    if ($Post -ne "") { $argv += @('--scheduled-post', $Post) }
    $code = Invoke-Py32 $argv
    try { Stop-Transcript | Out-Null } catch {}
    exit $code
}

if ($Trio -ne "") {
    $code = Invoke-Py32 (@('jvlink_trio_odds.py', '--race', $Trio, '--date', $Date) + $dryArg)
    try { Stop-Transcript | Out-Null } catch {}
    exit $code
}

if ($Final) {
    $c1 = Invoke-Py32 (@('jvlink_obs.py', '--final-rt', $Date) + $dryArg)
    $c2 = Invoke-Py32 (@('jvlink_obs.py', '--final-stock', $Date) + $dryArg)
    try { Stop-Transcript | Out-Null } catch {}
    if ($c1 -ne 0) { exit $c1 }
    exit $c2
}

if ($Refetch) {
    $code = Invoke-Py32 (@('jvlink_obs.py', '--final-stock', $Date) + $dryArg)
    try { Stop-Transcript | Out-Null } catch {}
    exit $code
}

if ($Schedule) {
    $bundle = Join-Path $root "reports\cowork_input\${Date}_bundle.json"
    $deadline = (Get-Date -Hour 15 -Minute 0 -Second 0)
    while (-not (Test-Path $bundle)) {
        if ((Get-Date) -ge $deadline) {
            Write-Host "[obs] no bundle by 15:00 -> not a race day or Phase A not run. exit"
            try { Stop-Transcript | Out-Null } catch {}
            exit 1
        }
        Write-Host "[obs] waiting for $bundle"
        Start-Sleep -Seconds 120
    }
    Get-ScheduledTask -TaskName 'PyCaLiAI_OBS_T2R_*' -ErrorAction SilentlyContinue |
        Unregister-ScheduledTask -Confirm:$false
    Get-ScheduledTask -TaskName 'PyCaLiAI_OBS_TRIO_*' -ErrorAction SilentlyContinue |
        Unregister-ScheduledTask -Confirm:$false

    $self = Join-Path $root 'obs_schedule.ps1'
    function Register-Once([string]$name, [datetime]$at, [string]$psArgs, [int]$limitMin, [string]$desc) {
        $act = New-ScheduledTaskAction -Execute 'powershell.exe' `
            -Argument ("-NoProfile -ExecutionPolicy Bypass -File `"$self`" " + $psArgs) `
            -WorkingDirectory $root
        $trg = New-ScheduledTaskTrigger -Once -At $at
        $trg.EndBoundary = $at.AddHours(2).ToString("yyyy-MM-ddTHH:mm:ss")
        $set = New-ScheduledTaskSettingsSet -StartWhenAvailable -WakeToRun `
            -ExecutionTimeLimit (New-TimeSpan -Minutes $limitMin) `
            -MultipleInstances IgnoreNew -DeleteExpiredTaskAfter (New-TimeSpan -Hours 6)
        Register-ScheduledTask -TaskName $name -Action $act -Trigger $trg -Settings $set `
            -Description $desc -Force | Out-Null
    }

    $lines = & py -3.12-32 'jvlink_obs.py' '--list-schedule' $Date
    $n = 0
    foreach ($l in $lines) {
        $parts = $l -split "`t"
        if ($parts.Count -lt 2) { continue }
        $rid = $parts[0]; $post = $parts[1]
        $ph, $pm = $post -split ':'
        if ([int]$ph -eq 0 -and [int]$pm -eq 0) {
            Write-Host "[obs] post time 00:00 for $rid -> suspected corruption, not scheduled"
            continue
        }
        $postAt = Get-Date -Hour ([int]$ph) -Minute ([int]$pm) -Second 0
        $t2At = $postAt.AddMinutes(-$T2LeadMin)
        $trioAt = $postAt.AddMinutes(-$TrioLeadMin)
        if ($t2At -ge (Get-Date)) {
            Register-Once "PyCaLiAI_OBS_T2R_$rid" $t2At "-T2 $rid -Date $Date -Post $post$dryPs" 5 `
                "OBS v2.1 t2_candidate $rid ($post post)"
            $n++
        }
        if ($trioAt -ge (Get-Date)) {
            Register-Once "PyCaLiAI_OBS_TRIO_$rid" $trioAt "-Trio $rid -Date $Date$dryPs" 5 `
                "OBS v2.1 trio 0B35 T-10 $rid ($post post)"
        }
        Write-Host ("  {0}  post {1}  T2 {2:HH:mm}  TRIO {3:HH:mm}" -f $rid, $post, $t2At, $trioAt)
    }
    $finalAt = Get-Date -Hour 18 -Minute 30 -Second 0
    if ($finalAt -ge (Get-Date)) {
        Register-Once "PyCaLiAI_OBS_FINAL_$Date" $finalAt "-Final -Date $Date$dryPs" 60 `
            "OBS v2.1 final_rt/final_stock candidates $Date"
    }
    $refetchAt = (Get-Date -Hour 20 -Minute 0 -Second 0).AddDays(1)
    Register-Once "PyCaLiAI_OBS_REFETCH_$Date" $refetchAt "-Refetch -Date $Date$dryPs" 45 `
        "OBS v2.1 final_stock re-fetch (next day) $Date"
    Write-Host "[obs] registered $n races (T2 + TRIO), FINAL 18:30, REFETCH next day 20:00"
    try { Stop-Transcript | Out-Null } catch {}
    exit 0
}

Write-Host "usage: obs_schedule.ps1 -Schedule | -T2 <rid> | -Trio <rid> | -Final | -Refetch  [-Date yyyyMMdd] [-Dry]"
try { Stop-Transcript | Out-Null } catch {}
exit 2
