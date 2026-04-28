##############################################################
# weekly_cowork.ps1  --- Cowork (Anthropic Desktop App) bridge
#
# Purpose:
#   Generate v5 marks JSON from data/weekly/YYYYMMDD.csv
#   so it can be fed to Cowork (Anthropic Desktop App)
#   to receive proposed bets.
#
# Output:
#   reports/cowork_input/{YYYYMMDD}/{race_id}.json   (1 race / 1 file)
#   reports/cowork_input/{YYYYMMDD}_bundle.json       (aggregated)
#
# Manual follow-up:
#   1. Drop reports/cowork_input/{YYYYMMDD}_bundle.json into Cowork
#   2. Read the bets returned by Cowork
#   3. streamlit run app.py -> "Cowork (Anthropic Desktop)" tab
#      -> input the bets -> save
#
# Usage:
#   .\weekly_cowork.ps1              # auto-detect latest weekly CSV
#   .\weekly_cowork.ps1 20260426     # specify date
#   .\weekly_cowork.ps1 20260426 v5  # specify model tag (default v5)
##############################################################
param(
    [string]$Date  = "",
    [string]$Model = "v5"
)

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

# -- v5 model presence --
$modelPath = "models\unified_rank_$Model.pkl"
if (-not (Test-Path $modelPath)) {
    Write-Error "$modelPath not found. v5 trained model is required."
    exit 1
}

Write-Host ""
Write-Host "=== weekly_cowork: $Date  (model=$Model) ===" -ForegroundColor Green
Write-Host ""

# -- Step 1: optional hosei generation --
Write-Host "[1/3] make_weekly_hosei.py (optional, skip if not present) ..." -ForegroundColor Cyan
python make_weekly_hosei.py --csv $csvPath
if ($LASTEXITCODE -ne 0) {
    Write-Host "      hosei skipped" -ForegroundColor Yellow
}

# -- Step 2: build Cowork-input JSON --
Write-Host ""
Write-Host "[2/3] export_weekly_marks.py ($Model) ..." -ForegroundColor Cyan
python export_weekly_marks.py --csv $csvPath --model $Model
if ($LASTEXITCODE -ne 0) {
    Write-Error "export_weekly_marks.py failed."
    exit 1
}

# -- Step 3: summary --
Write-Host ""
Write-Host "[3/3] summary" -ForegroundColor Cyan
$bundlePath = "reports\cowork_input\${Date}_bundle.json"
$indivDir   = "reports\cowork_input\$Date"
if (Test-Path $bundlePath) {
    $size = [math]::Round((Get-Item $bundlePath).Length / 1024, 1)
    Write-Host ("      bundle : {0} ({1} KB)" -f $bundlePath, $size) -ForegroundColor Green
}
if (Test-Path $indivDir) {
    $n = (Get-ChildItem $indivDir -Filter '*.json' | Measure-Object).Count
    Write-Host ("      per-race: {0}\  ({1} races)" -f $indivDir, $n) -ForegroundColor Green
}

Write-Host ""
Write-Host "=== git add / commit / push ===" -ForegroundColor Cyan

# -- Step 4: git add cowork_input artifacts --
git add data/weekly/$Date.csv 2>$null
if (Test-Path $bundlePath) { git add $bundlePath }
if (Test-Path $indivDir)   { git add ("{0}/*" -f $indivDir.Replace('\','/')) }

$y = $Date.Substring(0,4)
$m = $Date.Substring(4,2)
$d = $Date.Substring(6,2)

# -- Step 5: commit (skip if no diff) --
$staged = git diff --cached --name-only
if (-not $staged) {
    Write-Host "      no changes to commit" -ForegroundColor Yellow
} else {
    git commit -m ("add cowork_input {0}-{1}-{2} (model={3})" -f $y, $m, $d, $Model)
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "git commit failed (continuing)"
    } else {
        # -- Step 6: pull --rebase + push --
        git pull --rebase --autostash origin master
        git push origin HEAD:master
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "git push failed. Re-run manually: git push origin HEAD:master"
        } else {
            Write-Host "      pushed to origin/master" -ForegroundColor Green
        }
    }
}

Write-Host ""
Write-Host "=== Done ===" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host ("  1. Send {0} to Cowork (Anthropic Desktop App)" -f $bundlePath)
Write-Host "  2. Read the bets returned by Cowork"
Write-Host "  3. streamlit run app.py  ->  'Cowork (Anthropic Desktop)' tab"
Write-Host "     -> input the bets in the form  ->  press the save button"
Write-Host "     (the save button will auto git push the bets to Streamlit Cloud)"
