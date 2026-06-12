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
#
# Bug-fix history:
#   2026-05-17: $realDiff = git status --porcelain --untracked-files=no が
#               PS 5.1 で空配列扱いされる事例 → `git diff --cached/diff` に変更
#   2026-05-23: 大量 git checkout 後の diff 検出が依然不安定 →
#               diff 判定 自体を放棄、commit を素直に試みる方式に変更。
#               また 246 件以上の per-file checkout が原因らしいので、
#               全 file path を 1 回の `git checkout master -- $batch` で渡す。
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
$SyncFiles = @(
    "Dockerfile",
    "README.md",
    ".dockerignore",
    "requirements-nicegui.txt",
    "nicegui_app.py",
    "betting_judgment.py",
    "umami.py",
    "reports/audit_ev_bin_roi.json",
    "data/course_stats.json",
    "data/chaos_quantiles.json",
    "data/kekka/wide_kekka.csv",
    "data/pedigree_stats.json"
)

# Regex patterns matched against `git ls-tree -r --name-only master`.
$SyncDataPatterns = @(
    '^data/weekly/[0-9]{8}\.[cC][sS][vV]$',   # 大小文字非依存 (TARGET が .CSV 出力する事あり)
    '^data/hosei/H_20[0-9]{2}[0-9]+(-[0-9]+)?\.csv$',    # 年非依存 (2027 以降も同期継続)
    '^data/training/H-20[0-9]{2}[0-9]+(-[0-9]+)?\.csv$',
    '^data/training/W-20[0-9]{2}[0-9]+(-[0-9]+)?\.csv$',
    '^data/kako5/[0-9]{8}\.csv$',
    '^data/kekka/[0-9]{8}\.csv$',
    '^reports/cowork_input/[0-9]{8}_bundle\.json$',
    '^reports/cowork_input/[0-9]{8}/[^/]+\.json$',
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

# 2b. stash uncommitted tracked changes before the `git checkout --force` in
#     step 4. Without this, --force silently destroys any unstaged edit to a
#     tracked file (e.g. work-in-progress docs/*.md or *.py) every time this
#     script runs — and it runs from every weekly_nicegui phase. We restore
#     the stash after switching back to $origBranch.
$autoStashed = $false
$dirty = git status --porcelain --untracked-files=no
if ($dirty) {
    Write-Step "Uncommitted tracked changes detected -> stashing before checkout --force"
    git stash push --quiet -m "sync-hf auto-stash $masterSha"
    if ($LASTEXITCODE -ne 0) { Fail "git stash failed; commit or stash manually before running sync-hf" }
    $autoStashed = $true
}

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

# 5. checkout SyncFiles from master (small list, 1 file = 1 call OK)
Write-Step "Checking out fixed sync files from master"
foreach ($f in $SyncFiles) {
    git checkout master -- $f
    if ($LASTEXITCODE -ne 0) {
        git checkout $origBranch
        Fail "checkout failed for $f"
    }
}

# 5b. checkout data files matching regex patterns.
# Bug fix (2026-05-23): per-file checkout を 246+ 回繰り返すと、後段の
# git diff/status コマンドが空配列を返すバグが PS 5.1 で発生。
# → 全 file path を集めて 1 回の `git checkout master -- $batch` で一括 checkout。
Write-Step "Checking out data files (weekly / hosei / training / kako5 / cowork)"
$allMasterFiles = git ls-tree -r --name-only master
$batchFiles = @()
foreach ($pat in $SyncDataPatterns) {
    $files = @($allMasterFiles | Where-Object { $_ -match $pat })
    if ($files.Count -eq 0) {
        Write-Host "    $pat -> 0 files" -ForegroundColor DarkGray
        continue
    }
    $batchFiles += $files
    Write-Host "    $pat -> $($files.Count) files" -ForegroundColor DarkGray
}
if ($batchFiles.Count -gt 0) {
    git checkout master -- $batchFiles
    Write-Host "    total batched: $($batchFiles.Count) files" -ForegroundColor DarkGray
}

# 6. show status (display only, never used for diff judgment after 5/23)
Write-Step "Status on hf-spaces:"
git status --short

if ($DryRun) {
    Write-Step "DryRun: status shown above. Reverting working tree and switching back to $origBranch (no commit/push)."
    git checkout --force $origBranch
    if ($autoStashed) {
        Write-Step "Restoring stashed changes"
        git stash pop
    }
    exit 0
}

# 7. commit. Bug fix (2026-05-23): diff 検出に依存せず、ガンガン add して
#    commit を試みる。"nothing to commit" でも graceful に push 段階に進む。
Write-Step "Committing"
git add $SyncFiles
if ($batchFiles.Count -gt 0) {
    git add $batchFiles
}

$msg = "sync: master $($masterSha.Substring(0,7))"
git commit -m $msg
if ($LASTEXITCODE -ne 0) {
    Write-Host "    nothing to commit (working tree clean vs hf-spaces HEAD)" -ForegroundColor Yellow
}

# 8. push to HF (HF default branch is main)
# commit が成功 or 失敗どちらでも、local hf-spaces が remote hf/main より進んでる
# 場合はそれを push する (前回の push 失敗を必ずリカバリ)。
git fetch hf main 2>$null
$localSha  = (git rev-parse hf-spaces).Trim()
$remoteSha = (git rev-parse hf/main 2>$null).Trim()
if ($localSha -ne $remoteSha) {
    Write-Step "Pushing to HuggingFace Spaces (hf/main)"
    Write-Host "    local hf-spaces=$($localSha.Substring(0,7)) vs hf/main=$($remoteSha.Substring(0,7))" -ForegroundColor DarkGray
    git push hf hf-spaces:main
    if ($LASTEXITCODE -ne 0) {
        Write-Host "push failed; retry manually: git push hf hf-spaces:main" -ForegroundColor Yellow
    }
} else {
    Write-Step "Already in sync with hf/main (no push needed)."
}

# 9. switch back
Write-Step "Switching back to $origBranch"
git checkout $origBranch
if ($autoStashed) {
    Write-Step "Restoring stashed changes"
    git stash pop
    if ($LASTEXITCODE -ne 0) {
        Write-Host "    stash pop conflicted; resolve manually (your changes are in 'git stash list')" -ForegroundColor Yellow
    }
}

Write-Step "Done. Check build at https://huggingface.co/spaces/gutchi15300/pycaliAI"
