##############################################################
# weekly_post.ps1  --- post-race workflow
#
# Usage:
#   .\weekly_post.ps1              # auto-detect latest uncommitted kekka CSV
#   .\weekly_post.ps1 20260322     # specify date
##############################################################
param([string]$Date = "")

Set-Location 'E:\PyCaLiAI'

# -- Determine date --
if ($Date -eq "") {
    $untracked = git status --short data/kekka/ 2>$null |
                 Where-Object { $_ -match '^\?\?' -or $_ -match '^A' } |
                 ForEach-Object { ($_ -replace '^\s*\S+\s+', '').Trim() } |
                 Where-Object { $_ -match 'data.kekka.\d{8}\.csv$' }

    if (-not $untracked) {
        $latest = Get-ChildItem 'data\kekka' -Filter '????????.csv' |
                  Sort-Object Name -Descending |
                  Select-Object -First 1
        if ($latest) {
            $Date = $latest.BaseName
            Write-Host "Auto-detect (latest): $Date" -ForegroundColor Yellow
        } else {
            Write-Error "No CSV found in data\kekka\"
            exit 1
        }
    } else {
        $Date = [System.IO.Path]::GetFileNameWithoutExtension(
            ($untracked | Select-Object -Last 1))
        Write-Host "Auto-detect (uncommitted): $Date" -ForegroundColor Cyan
    }
}

$kekkaPath = "data\kekka\$Date.csv"
if (-not (Test-Path $kekkaPath)) {
    Write-Error "$kekkaPath not found. Place it in data\kekka\ first."
    exit 1
}

Write-Host ""
Write-Host "=== weekly_post: $Date ===" -ForegroundColor Green
Write-Host ""

# -- Step 1: rebuild results.json --
Write-Host "[1/4] Running generate_results.py ..." -ForegroundColor Cyan
python generate_results.py
if ($LASTEXITCODE -ne 0) {
    Write-Error "generate_results.py failed."
    exit 1
}
Write-Host "      data\results.json updated." -ForegroundColor Green

# -- Step 2: live_results_2026.csv に実績を照合 --
Write-Host "[2/4] Updating live_results_2026.csv ..." -ForegroundColor Cyan
python update_live_results.py --date $Date
if ($LASTEXITCODE -ne 0) {
    Write-Warning "update_live_results.py failed (non-fatal, continuing...)"
}
Write-Host "      live_results_2026.csv updated." -ForegroundColor Green

# -- Step 3: git add --
Write-Host "[3/4] git add ..." -ForegroundColor Cyan
git add $kekkaPath data/results.json data/live_results_2026.csv

# Cowork bets (if exists for this date) - 的中判定対象
$coworkBetsDir = "reports\cowork_bets\$Date"
if (Test-Path $coworkBetsDir) {
    git add ("{0}/*" -f $coworkBetsDir.Replace('\','/'))
    Write-Host "      staged: $kekkaPath + results.json + live_results_2026.csv + cowork_bets/$Date/" -ForegroundColor Green
} else {
    Write-Host "      staged: $kekkaPath + results.json + live_results_2026.csv (no cowork_bets/$Date/)" -ForegroundColor Green
}

# -- Step 4: git commit & push --
Write-Host "[4/4] git commit & push ..." -ForegroundColor Cyan
$y = $Date.Substring(0,4)
$m = $Date.Substring(4,2)
$d = $Date.Substring(6,2)
git commit -m "add kekka $y-$m-$d / update results"
git pull --rebase --autostash origin master
git push origin HEAD:master
if ($LASTEXITCODE -ne 0) {
    Write-Error "git push failed."
    exit 1
}

Write-Host ""
Write-Host "=== Done ===" -ForegroundColor Green
Write-Host "Streamlit Cloud results page will update in a few seconds." -ForegroundColor Gray

# -- Value Model月次再学習（月初の日曜のみ）--
$dow = (Get-Date).DayOfWeek   # Sunday=0
$dayOfMonth = (Get-Date).Day

if ($dow -eq 0 -and $dayOfMonth -le 7) {
    Write-Host ""
    Write-Host "=== Value Model月次再学習 ===" -ForegroundColor Yellow
    $endDate = (Get-Date -Format "yyyyMMdd")
    python retrain_value_model.py --end-date $endDate
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "retrain_value_model.py failed (non-fatal, continuing...)"
    }
}

# -- 週次監査（毎週日曜に自動実行）--
if ($dow -eq 0) {
    Write-Host ""
    Write-Host "=== 週次監査を実行します ===" -ForegroundColor Magenta
    & "$PSScriptRoot\run_audit.ps1"
}
