# =====================================================================
# sync-hf.ps1 - sync master to hf-spaces orphan branch and push to HF
# =====================================================================
# Usage:
#   .\sync-hf.ps1            # copy HF files from master to hf-spaces,
#                            # commit, push to hf/main
#   .\sync-hf.ps1 -DryRun    # show diff only, no commit/push
#
# Prerequisites:
#   - hf-spaces orphan branch exists locally
#   - hf remote is configured (https://huggingface.co/spaces/USER/SPACE)
#
# Why this script:
#   hf-spaces is an orphan branch (independent history from master),
#   so git merge cannot be used. Instead, we checkout the HF-relevant
#   files from master onto hf-spaces and re-commit.
# =====================================================================

[CmdletBinding()]
param(
    [switch]$DryRun
)

# Note: $ErrorActionPreference="Stop" is intentionally NOT set here.
# Windows PowerShell 5.1 wraps every native-stderr line as ErrorRecord
# (e.g. `git checkout` writes "Switched to branch 'X'" via a path that
# triggers NativeCommandError). With Stop that would abort the script
# mid-flight even on successful operations. We let each step decide
# success via $LASTEXITCODE.
$ErrorActionPreference = "Continue"

# Files that actually need to be deployed to HF Spaces.
# (large files in data/, reports/, models/ are excluded via .dockerignore at
# build time; here we list the small data subsets that NiceGUI actually reads)
$SyncFiles = @(
    "Dockerfile",
    "README.md",
    ".dockerignore",
    "requirements-nicegui.txt",
    "nicegui_app.py",
    "data/course_stats.json"
)

# Regex patterns matched against `git ls-tree -r --name-only master`.
# git ls-tree's pathspec does NOT expand globs, so we list ALL master files
# once and filter in PowerShell with -match.
# Large multi-year files (H_2013-2025, H-2015*-...) are NOT in the regex
# to keep the hf-spaces repo under 1 GB.
$SyncDataPatterns = @(
    # 8-digit YYYYMMDD basename only (rejects test.csv etc)
    '^data/weekly/[0-9]{8}\.csv$',
    '^data/hosei/H_2026[0-9]+(-[0-9]+)?\.csv$',
    '^data/training/H-2026[0-9]+(-[0-9]+)?\.csv$',
    '^data/training/W-2026[0-9]+(-[0-9]+)?\.csv$',
    '^data/kako5/[0-9]{8}\.csv$',
    '^data/kekka/[0-9]{8}\.csv$',
    # cowork_input: top-level YYYYMMDD_bundle.json + per-race subfolder
    '^reports/cowork_input/[0-9]{8}_bundle\.json$',
    '^reports/cowork_input/[0-9]{8}/[^/]+\.json$',
    # cowork_output: any file with date in name
    '^reports/cowork_output/[0-9]{8}.*$'
)

function Write-Step($msg) {
    Write-Host "==> $msg" -ForegroundColor Cyan
}

function Fail($msg) {
    Write-Host "ERROR: $msg" -ForegroundColor Red
    exit 1
}

# 1. record current branch
$origBranch = (git rev-parse --abbrev-ref HEAD).Trim()
Write-Step "Current branch: $origBranch"

if ($origBranch -ne "master") {
    Fail "Run this from master branch (current: $origBranch)"
}

# 2. record master HEAD
$masterSha = (git rev-parse HEAD).Trim()
Write-Step "master HEAD: $masterSha"

# 3. verify sync targets exist
foreach ($f in $SyncFiles) {
    if (-not (Test-Path $f)) {
        Fail "Sync target missing: $f"
    }
}
Write-Step "Verified $($SyncFiles.Count) sync target(s)"

# 4. switch to hf-spaces (--force discards stale LFS pointer junk that
#    sometimes shows up as 'modified' on master but is not actually edited)
Write-Step "Switching to hf-spaces"
git checkout --force hf-spaces
if ($LASTEXITCODE -ne 0) { Fail "checkout hf-spaces failed" }

# 5. checkout files from master
Write-Step "Checking out files from master"
foreach ($f in $SyncFiles) {
    git checkout master -- $f
    if ($LASTEXITCODE -ne 0) {
        git checkout $origBranch
        Fail "checkout failed for $f"
    }
}

# 5b. checkout data files matching regex patterns. We pull the ENTIRE
#     master tree's filename list once, then filter via -match.
Write-Step "Checking out data files (weekly / hosei / training / kako5 / cowork)"
$allMasterFiles = git ls-tree -r --name-only master
foreach ($pat in $SyncDataPatterns) {
    $files = $allMasterFiles | Where-Object { $_ -match $pat }
    if (-not $files) {
        Write-Host "    $pat -> 0 files" -ForegroundColor DarkGray
        continue
    }
    foreach ($f in $files) {
        git checkout master -- $f
    }
    $count = ($files | Measure-Object).Count
    Write-Host "    $pat -> $count files" -ForegroundColor DarkGray
}

# 6. show status
Write-Step "Status on hf-spaces:"
git status --short

$hasDiff = (git status --porcelain).Length -gt 0
if (-not $hasDiff) {
    Write-Step "No diff. Switching back to original branch."
    git checkout $origBranch
    exit 0
}

if ($DryRun) {
    Write-Step "DryRun: stopping here. Changes are staged."
    Write-Host "  commit  -> git commit -m 'sync from master'"
    Write-Host "  push    -> git push hf hf-spaces:main"
    Write-Host "  rollback-> git checkout . ; git checkout $origBranch"
    exit 0
}

# 7. commit. We rely on the staging done by `git checkout master -- <file>`
#    above (it auto-stages new files). For modified files we re-add them by
#    walking `git status --porcelain` and matching the regex patterns.
Write-Step "Committing"
git add $SyncFiles
$workingChanges = git status --porcelain | ForEach-Object { ($_ -replace '^...', '').Trim() }
foreach ($f in $workingChanges) {
    $pathOk = $false
    foreach ($pat in $SyncDataPatterns) {
        if ($f -match $pat) { $pathOk = $true; break }
    }
    if ($pathOk) { git add $f 2>$null }
}
$msg = "sync: master $($masterSha.Substring(0,7))"
git commit -m $msg
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: commit failed (likely no changes)." -ForegroundColor Red
    git checkout $origBranch
    exit 1
}

# 8. push to HF (HF default branch is main)
Write-Step "Pushing to HuggingFace Spaces (hf/main)"
git push hf hf-spaces:main
if ($LASTEXITCODE -ne 0) {
    Write-Host "push failed; retry manually: git push hf hf-spaces:main" -ForegroundColor Yellow
}

# 9. switch back
Write-Step "Switching back to $origBranch"
git checkout $origBranch

Write-Step "Done. Check build at https://huggingface.co/spaces/gutchi15300/pycaliAI"
