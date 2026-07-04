# =============================================================================
# baba_daily.ps1 - Sat/Sun morning: fetch JRA track condition -> today's bias -> site
# -----------------------------------------------------------------------------
#   1) fetch_baba_today.py  : JRA _data_cushion/_data_moist -> data/baba_today.json
#   2) if race day AND content changed -> sync-hf-umami.ps1 (build_site + HF push)
#   On non-race-day / fetch failure: publish nothing (fail-safe).
#
#   JRA posts race-day cushion ~08:20, sometimes later. A single 08:30 shot can
#   miss it and grab the prior day's read. So the scheduled task REPEATS every
#   30 min from 08:30 for 3 hours; each run re-fetches and only pushes when the
#   deployed content actually changed (skips redundant pushes). Once JRA posts
#   today's measurement, the next repetition catches it and publishes.
#
#   .\baba_daily.ps1            # normal (fetch -> HF publish if changed)
#   .\baba_daily.ps1 -DryRun    # fetch + build_site only (no HF push)
#   .\baba_daily.ps1 -Register  # (re)register Sat/Sun 08:30 task w/ 30min repeat
# (ASCII-only by design: avoids PS 5.1 .ps1 encoding pitfalls.)
# =============================================================================
param(
    [switch]$DryRun,
    [switch]$Register
)
$ErrorActionPreference = "Stop"
$ROOT = "E:\PyCaLiAI"
$PY = Join-Path $ROOT "venv311\Scripts\python.exe"

# Content signature (ignores fetched_at): measured_label + per-venue cushions.
function Get-BabaSig($path) {
    if (-not (Test-Path $path)) { return "" }
    try {
        $j = Get-Content $path -Raw -Encoding UTF8 | ConvertFrom-Json
        if (-not $j.venues -or @($j.venues).Count -eq 0) { return "empty" }
        $v = ($j.venues | ForEach-Object { "$($_.venue):$($_.cushion):$($_.shiba_gp):$($_.dirt_gp)" }) -join "|"
        return "$($j.measured_label)#$($j.is_measured_today)#$v"
    }
    catch { return "" }
}

if ($Register) {
    $act = New-ScheduledTaskAction -Execute "powershell.exe" `
        -Argument "-NoProfile -ExecutionPolicy Bypass -File $ROOT\baba_daily.ps1" `
        -WorkingDirectory $ROOT
    $trg = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Saturday, Sunday -At 8:30AM
    # Attach a repetition (every 30 min for 3h) so late-posted race-day data is caught.
    $rep = (New-ScheduledTaskTrigger -Once -At 8:30AM `
            -RepetitionInterval (New-TimeSpan -Minutes 30) `
            -RepetitionDuration (New-TimeSpan -Hours 3)).Repetition
    $trg.Repetition = $rep
    $set = New-ScheduledTaskSettingsSet -StartWhenAvailable -WakeToRun `
        -ExecutionTimeLimit (New-TimeSpan -Minutes 15) -MultipleInstances IgnoreNew
    Register-ScheduledTask -TaskName "PyCaLiAI_Baba" -Action $act -Trigger $trg `
        -Settings $set -Description "JRA baba -> today bias -> site (Sat/Sun 8:30, 30min repeat 3h)" -Force | Out-Null
    Write-Host "[registered] PyCaLiAI_Baba  Sat/Sun 08:30 +30min repeat x3h (WakeToRun)"
    exit 0
}

Set-Location $ROOT
$bt = Join-Path $ROOT "data\baba_today.json"
$deployed = Join-Path $ROOT "site\data\baba_today.json"
$sigBefore = Get-BabaSig $deployed

Write-Host "[1/2] fetch_baba_today.py"
& $PY (Join-Path $ROOT "fetch_baba_today.py")

# Did we get active venues (venues > 0)?
$hasV = $false
$isToday = $false
if (Test-Path $bt) {
    try {
        $j = Get-Content $bt -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($j.venues -and @($j.venues).Count -gt 0) { $hasV = $true }
        if ($j.is_measured_today -eq $true) { $isToday = $true }
    }
    catch { $hasV = $false }
}
if (-not $hasV) { Write-Host "[skip] no race today / fetch failed -> publish nothing"; exit 0 }

$sigAfter = Get-BabaSig $bt
if ($sigAfter -eq $sigBefore) {
    Write-Host "[skip] baba content unchanged (measured_today=$isToday) -> no push"
    exit 0
}

if ($DryRun) {
    Write-Host "[2/2] DryRun: build_site.py only (no HF push)  measured_today=$isToday"
    & $PY (Join-Path $ROOT "build_site.py")
    exit $LASTEXITCODE
}

Write-Host "[2/2] content changed (measured_today=$isToday) -> sync-hf-umami.ps1"
& (Join-Path $ROOT "sync-hf-umami.ps1")
exit $LASTEXITCODE
