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

$ErrorActionPreference = "Stop"

# Files that actually need to be deployed to HF Spaces.
# (large files in data/, reports/, models/ are excluded via .dockerignore at
# build time; here we list the small data subsets that NiceGUI actually reads)
$SyncFiles = @(
    "Dockerfile",
    "README.md",
    ".dockerignore",
    "requirements-nicegui.txt",
    "nicegui_app.py"
)

# Data file glob patterns to sync (relative to repo root). Each entry is
# matched against `git ls-files master -- <pattern>` and all results checked
# out. Large multi-year files (H_2013-2025, H-2015*-2026*) are intentionally
# omitted to keep the hf-spaces repo under 1 GB.
$SyncDataPatterns = @(
    "data/weekly/*.csv",
    "data/hosei/H_2026*.csv",
    "data/training/H-2026*.csv",
    "data/training/W-2026*.csv",
    "data/kako5/*.csv",
    "reports/cowork_input/*.json",
    "reports/cowork_output/*"
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

# 5b. checkout data files matching glob patterns
Write-Step "Checking out data files (weekly / hosei / training 2026 / kako5)"
foreach ($pat in $SyncDataPatterns) {
    $files = git ls-tree -r --name-only master -- $pat 2>$null
    if (-not $files) { continue }
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

# 7. commit
Write-Step "Committing"
git add $SyncFiles
foreach ($pat in $SyncDataPatterns) {
    git add $pat 2>$null
}
$msg = "sync: master $($masterSha.Substring(0,7))"
git commit -m $msg
if ($LASTEXITCODE -ne 0) { Fail "commit failed" }

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
