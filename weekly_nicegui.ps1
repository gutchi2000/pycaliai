##############################################################
# weekly_nicegui.ps1  --- pre-race workflow for NiceGUI on HF Spaces
#
# Purpose:
#   Generate hosei + bundle.json (and optionally pred CSV), commit
#   to GitHub master, and push hf-spaces orphan to HuggingFace Spaces.
#   This is the one-shot weekly script for NiceGUI deployment.
#
# Usage:
#   .\weekly_nicegui.ps1                        # auto-detect latest CSV
#   .\weekly_nicegui.ps1 20260502               # specify date
#   .\weekly_nicegui.ps1 20260502 -SkipHF       # generate only, no HF push
#   .\weekly_nicegui.ps1 20260502 -SkipPredict  # skip predict_weekly.py
#                                                  (NiceGUI doesn't need it)
#   .\weekly_nicegui.ps1 20260502 -SkipGit      # local generation only
#
# Steps:
#   1. make_weekly_hosei.py     -> data/hosei/H_{date}.csv
#   2. predict_weekly.py        -> reports/pred_{date}.csv  (Streamlit)
#   3. export_weekly_marks.py   -> reports/cowork_input/{date}_bundle.json
#   4. git add + commit + push origin master  (Streamlit Cloud)
#   5. sync-hf.ps1                            (HuggingFace Spaces)
##############################################################
param(
    [string]$Date = "",
    [string]$Model = "v5",
    [switch]$SkipHF,
    [switch]$SkipPredict,
    [switch]$SkipGit
)

$ErrorActionPreference = "Continue"
Set-Location 'E:\PyCaLiAI'

function Step($msg) {
    Write-Host ""
    Write-Host "==> $msg" -ForegroundColor Cyan
}
function OK($msg) {
    Write-Host "    OK: $msg" -ForegroundColor Green
}
function Warn($msg) {
    Write-Host "    WARN: $msg" -ForegroundColor Yellow
}
function Fail($msg) {
    Write-Host "    FAIL: $msg" -ForegroundColor Red
    exit 1
}

# -- Determine date --
if ($Date -eq "") {
    $untracked = git status --short data/weekly/ 2>$null |
                 Where-Object { $_ -match '^\?\?' -or $_ -match '^ M' -or $_ -match '^A' } |
                 ForEach-Object { ($_ -replace '^\s*\S+\s+', '').Trim() } |
                 Where-Object { $_ -match 'data.weekly.\d{8}\.csv$' }

    if ($untracked) {
        $Date = [System.IO.Path]::GetFileNameWithoutExtension(
            ($untracked | Select-Object -Last 1))
        Write-Host "Auto-detect (uncommitted): $Date" -ForegroundColor Cyan
    } else {
        $latest = Get-ChildItem 'data\weekly' -Filter '????????.csv' |
                  Sort-Object Name -Descending |
                  Select-Object -First 1
        if ($latest) {
            $Date = $latest.BaseName
            Write-Host "Auto-detect (latest): $Date" -ForegroundColor Yellow
        } else {
            Fail "No CSV found in data\weekly\"
        }
    }
}

$csvPath = "data\weekly\$Date.csv"
if (-not (Test-Path $csvPath)) {
    Fail "$csvPath not found. Place TARGET CSV in data\weekly\ first."
}

# -- Verify model --
$modelPath = "models\unified_rank_$Model.pkl"
if (-not (Test-Path $modelPath)) {
    Fail "$modelPath not found. v5 trained model is required."
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  weekly_nicegui : $Date  (model=$Model)" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green

# -- Step 1: hosei --
Step "[1/5] make_weekly_hosei.py"
python make_weekly_hosei.py --csv $csvPath
if ($LASTEXITCODE -ne 0) {
    Warn "hosei generation failed (continuing without)"
} else {
    OK "data\hosei\H_$Date.csv"
}

# -- Step 2: predict (Streamlit only, optional) --
if ($SkipPredict) {
    Step "[2/5] predict_weekly.py SKIPPED (-SkipPredict)"
} else {
    Step "[2/5] predict_weekly.py (for Streamlit)"
    python predict_weekly.py --csv $csvPath
    if ($LASTEXITCODE -ne 0) {
        Warn "predict_weekly failed (NiceGUI doesn't need this, continuing)"
    } else {
        OK "reports\pred_$Date.csv"
    }
}

# -- Step 3: bundle.json (NiceGUI required) --
Step "[3/5] export_weekly_marks.py (model=$Model) -> bundle.json"
python export_weekly_marks.py --csv $csvPath --model $Model
if ($LASTEXITCODE -ne 0) {
    Fail "export_weekly_marks.py failed. NiceGUI needs the bundle."
}
$bundlePath = "reports\cowork_input\${Date}_bundle.json"
if (Test-Path $bundlePath) {
    $size = [math]::Round((Get-Item $bundlePath).Length / 1024, 1)
    OK ("{0} ({1} KB)" -f $bundlePath, $size)
} else {
    Fail "bundle.json not created at $bundlePath"
}

# -- Step 4: git add + commit + push origin --
if ($SkipGit) {
    Step "[4/5] git push SKIPPED (-SkipGit)"
} else {
    Step "[4/5] git add / commit / push origin master"

    git add $csvPath 2>$null

    $hoseiPath = "data\hosei\H_$Date.csv"
    if (Test-Path $hoseiPath) { git add $hoseiPath 2>$null }

    if (Test-Path $bundlePath) { git add $bundlePath 2>$null }

    $predPath = "reports\pred_$Date.csv"
    if (Test-Path $predPath) { git add -f $predPath 2>$null }

    $kako5Path = "data\kako5\$Date.csv"
    if (Test-Path $kako5Path) { git add $kako5Path 2>$null }

    # 直近の training CSV があればそれも (週次更新)
    Get-ChildItem 'data\training' -Filter "[HW]-*$Date*.csv" -ErrorAction SilentlyContinue |
        ForEach-Object { git add $_.FullName 2>$null }

    $staged = git diff --cached --name-only 2>$null
    if (-not $staged) {
        Write-Host "    no changes to commit" -ForegroundColor Yellow
    } else {
        $y = $Date.Substring(0,4); $m = $Date.Substring(4,2); $d = $Date.Substring(6,2)
        git commit -m "weekly $y-$m-$d (NiceGUI bundle, model=$Model)"
        if ($LASTEXITCODE -ne 0) {
            Warn "git commit failed"
        } else {
            git pull --rebase --autostash origin master
            git push origin HEAD:master
            if ($LASTEXITCODE -ne 0) {
                Warn "git push failed; retry: git push origin HEAD:master"
            } else {
                OK "pushed to origin/master"
            }
        }
    }
}

# -- Step 5: sync to HuggingFace Spaces --
if ($SkipHF) {
    Step "[5/5] sync-hf.ps1 SKIPPED (-SkipHF)"
} else {
    Step "[5/5] sync-hf.ps1 (master -> hf-spaces orphan -> HF push)"
    if (-not (Test-Path 'sync-hf.ps1')) {
        Warn "sync-hf.ps1 not found; skipping HF push"
    } else {
        & .\sync-hf.ps1
        if ($LASTEXITCODE -ne 0) {
            Warn "sync-hf.ps1 returned non-zero"
        }
    }
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  Done." -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next:" -ForegroundColor Yellow
Write-Host "  - HF Spaces : https://gutchi15300-pycaliai.hf.space" -ForegroundColor Gray
Write-Host "  - Streamlit : streamlit run app.py" -ForegroundColor Gray
Write-Host "  - NiceGUI local : python nicegui_app.py" -ForegroundColor Gray
Write-Host ""
Write-Host "After race day, run weekly_post.ps1 for results aggregation." -ForegroundColor Gray
