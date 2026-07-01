# =============================================================================
# baba_daily.ps1 - Sat/Sun 8:30: fetch JRA track condition -> today's bias -> site
# -----------------------------------------------------------------------------
#   1) fetch_baba_today.py  : JRA _data_cushion/_data_moist -> data/baba_today.json
#   2) if race day -> sync-hf-umami.ps1 (build_site.py embeds into site/data + HF push)
#   On non-race-day / fetch failure: publish nothing (fail-safe).
#
#   .\baba_daily.ps1            # normal (fetch -> HF publish)
#   .\baba_daily.ps1 -DryRun    # fetch + build_site only (no HF push)
#   .\baba_daily.ps1 -Register  # register Sat/Sun 08:30 scheduled task (once)
# (ASCII-only by design: avoids PS 5.1 .ps1 encoding pitfalls.)
# =============================================================================
param(
    [switch]$DryRun,
    [switch]$Register
)
$ErrorActionPreference = "Stop"
$ROOT = "E:\PyCaLiAI"
$PY = Join-Path $ROOT "venv311\Scripts\python.exe"

if ($Register) {
    $act = New-ScheduledTaskAction -Execute "powershell.exe" `
        -Argument "-NoProfile -ExecutionPolicy Bypass -File $ROOT\baba_daily.ps1" `
        -WorkingDirectory $ROOT
    $trg = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Saturday, Sunday -At 8:30AM
    $set = New-ScheduledTaskSettingsSet -StartWhenAvailable -WakeToRun `
        -ExecutionTimeLimit (New-TimeSpan -Minutes 20) -MultipleInstances IgnoreNew
    Register-ScheduledTask -TaskName "PyCaLiAI_Baba" -Action $act -Trigger $trg `
        -Settings $set -Description "JRA baba -> today bias -> site (Sat/Sun 8:30)" -Force | Out-Null
    Write-Host "[registered] PyCaLiAI_Baba  Sat/Sun 08:30 (WakeToRun)"
    exit 0
}

Set-Location $ROOT
Write-Host "[1/2] fetch_baba_today.py"
& $PY (Join-Path $ROOT "fetch_baba_today.py")

# Did we get active venues (venues > 0)?
$bt = Join-Path $ROOT "data\baba_today.json"
$hasV = $false
if (Test-Path $bt) {
    try {
        $j = Get-Content $bt -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($j.venues -and @($j.venues).Count -gt 0) { $hasV = $true }
    }
    catch { $hasV = $false }
}
if (-not $hasV) { Write-Host "[skip] no race today / fetch failed -> publish nothing"; exit 0 }

if ($DryRun) {
    Write-Host "[2/2] DryRun: build_site.py only (no HF push)"
    & $PY (Join-Path $ROOT "build_site.py")
    exit $LASTEXITCODE
}

Write-Host "[2/2] sync-hf-umami.ps1 (build_site + HF push)"
& (Join-Path $ROOT "sync-hf-umami.ps1")
exit $LASTEXITCODE
