##############################################################
# weekly_pre.ps1  --- pre-race workflow
#
# Usage:
#   .\weekly_pre.ps1              # auto-detect latest uncommitted weekly CSV
#   .\weekly_pre.ps1 20260322     # specify date
##############################################################
param([string]$Date = "")

Set-Location 'E:\PyCaLiAI'

# -- Determine date --
if ($Date -eq "") {
    $untracked = git status --short data/weekly/ 2>$null |
                 Where-Object { $_ -match '^\?\?' -or $_ -match '^ M' -or $_ -match '^A' } |
                 ForEach-Object { ($_ -replace '^\s*\S+\s+', '').Trim() } |
                 Where-Object { $_ -match 'data.weekly.\d{8}\.csv$' }

    if (-not $untracked) {
        $latest = Get-ChildItem 'data\weekly' -Filter '????????.csv' |
                  Sort-Object Name -Descending |
                  Select-Object -First 1
        if ($latest) {
            $Date = $latest.BaseName
            Write-Host "Auto-detect (latest): $Date" -ForegroundColor Yellow
        } else {
            Write-Error "No CSV found in data\weekly\"
            exit 1
        }
    } else {
        $Date = [System.IO.Path]::GetFileNameWithoutExtension(
            ($untracked | Select-Object -Last 1))
        Write-Host "Auto-detect (uncommitted): $Date" -ForegroundColor Cyan
    }
}

$csvPath = "data\weekly\$Date.csv"
if (-not (Test-Path $csvPath)) {
    Write-Error "$csvPath not found. Place it in data\weekly\ first."
    exit 1
}

Write-Host ""
Write-Host "=== weekly_pre: $Date ===" -ForegroundColor Green
Write-Host ""

# -- Step 1: generate pred CSV --
Write-Host "[1/3] Running predict_weekly.py ..." -ForegroundColor Cyan
python predict_weekly.py --csv $csvPath
if ($LASTEXITCODE -ne 0) {
    Write-Error "predict_weekly.py failed."
    exit 1
}
Write-Host "      reports\pred_$Date.csv created." -ForegroundColor Green

# -- Step 2: git add --
Write-Host "[2/3] git add ..." -ForegroundColor Cyan
git add $csvPath
Write-Host "      staged: $csvPath" -ForegroundColor Green

# -- Step 3: git commit & push --
Write-Host "[3/3] git commit & push ..." -ForegroundColor Cyan
$y = $Date.Substring(0,4)
$m = $Date.Substring(4,2)
$d = $Date.Substring(6,2)
git commit -m "add weekly csv $y-$m-$d"
git push origin master
if ($LASTEXITCODE -ne 0) {
    Write-Error "git push failed."
    exit 1
}

Write-Host ""
Write-Host "=== Done ===" -ForegroundColor Green
Write-Host "Streamlit Cloud will update in a few seconds." -ForegroundColor Gray
Write-Host "Open the URL to check buy orders. pred CSV will be auto-saved." -ForegroundColor Gray
