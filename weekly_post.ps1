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
Write-Host "[1/3] Running generate_results.py ..." -ForegroundColor Cyan
python generate_results.py
if ($LASTEXITCODE -ne 0) {
    Write-Error "generate_results.py failed."
    exit 1
}
Write-Host "      data\results.json updated." -ForegroundColor Green

# -- Step 2: git add --
Write-Host "[2/3] git add ..." -ForegroundColor Cyan
git add $kekkaPath data/results.json
Write-Host "      staged: $kekkaPath + results.json" -ForegroundColor Green

# -- Step 3: git commit & push --
Write-Host "[3/3] git commit & push ..." -ForegroundColor Cyan
$y = $Date.Substring(0,4)
$m = $Date.Substring(4,2)
$d = $Date.Substring(6,2)
git commit -m "add kekka $y-$m-$d / update results"
git push origin master
if ($LASTEXITCODE -ne 0) {
    Write-Error "git push failed."
    exit 1
}

Write-Host ""
Write-Host "=== Done ===" -ForegroundColor Green
Write-Host "Streamlit Cloud results page will update in a few seconds." -ForegroundColor Gray

# -- 月次監査（毎月第4週日曜に自動実行）--
$dow     = (Get-Date).DayOfWeek   # Sunday=0
$dom     = (Get-Date).Day
$isLastSundayRange = ($dow -eq 0) -and ($dom -ge 22)

if ($isLastSundayRange) {
    Write-Host ""
    Write-Host "=== 月次監査を実行します ===" -ForegroundColor Magenta
    & "$PSScriptRoot\run_audit.ps1"
} else {
    Write-Host ""
    Write-Host "(月次監査: 毎月第4週日曜に自動実行)" -ForegroundColor DarkGray
}
