# =====================================================================
# sync-hf.ps1 - sync master to hf-spaces branch (via worktree) and push to HF
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
#   2026-07-29: (1) ブランチ往復を全廃し .worktrees/hf-spaces の常設 worktree
#               方式へ。旧方式の `git checkout --force hf-spaces` →
#               `git checkout master` 往復は「master で未追跡だが hf-spaces
#               では追跡」のファイルを旧版で上書き→物理削除していた
#               (2026-07-29 に data/kekka/20260418〜20260510.csv 7件消失)。
#               worktree 方式は master 作業ツリーに一切触れないので
#               この事故クラスごと消滅、auto-stash も不要になった。
#               (2) `git checkout/add master -- $batchFiles` の一括引数渡しが
#               1030 件で Windows のコマンドライン長制限
#               ("The filename or extension is too long") で死んでいたのを
#               --pathspec-from-file (一時ファイル渡し) に置換。
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

$ROOT = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
$WT = Join-Path $ROOT ".worktrees\hf-spaces"

# 1. sanity: run from master (sync source is the master *commit*, and the
#    weekly flow always runs here; also guards against hf-spaces being
#    checked out in the main tree, which would block worktree add)
$origBranch = (git rev-parse --abbrev-ref HEAD).Trim()
Write-Step "Current branch: $origBranch"
if ($origBranch -ne "master") {
    Fail "Run this from master branch (current: $origBranch)"
}

# 2. record master HEAD
$masterSha = (git rev-parse HEAD).Trim()
Write-Step "master HEAD: $masterSha"

# 3. collect sync paths from the master COMMIT (not the working tree).
#    ls-tree membership also verifies the fixed SyncFiles are actually
#    committed to master (Test-Path alone let uncommitted files pass and
#    then silently deploy a stale committed version).
$allMasterFiles = git ls-tree -r --name-only master
$masterSet = @{}
foreach ($p in $allMasterFiles) { $masterSet[$p] = $true }

foreach ($f in $SyncFiles) {
    if (-not $masterSet.ContainsKey($f)) {
        Fail "Sync target not tracked in master: $f (commit it first)"
    }
}
Write-Step "Verified $($SyncFiles.Count) fixed sync target(s) tracked in master"

Write-Step "Collecting data files (weekly / hosei / training / kako5 / cowork)"
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
$allSyncPaths = @($SyncFiles) + $batchFiles
Write-Host "    total sync paths: $($allSyncPaths.Count)" -ForegroundColor DarkGray

# Write the pathspec to a temp file (UTF-8 no BOM; a BOM would corrupt the
# first pathspec) to dodge the Windows command-line length limit.
$specFile = Join-Path $env:TEMP "synchf_pathspec.txt"
[System.IO.File]::WriteAllLines($specFile, [string[]]$allSyncPaths)

# 4. ensure the persistent hf-spaces worktree exists and is healthy
git worktree prune 2>$null
$wtPorcelain = git worktree list --porcelain
$wtRegistered = ($wtPorcelain | Where-Object { $_ -match '^worktree ' } |
                 ForEach-Object { $_.Substring(9) }) -contains ($WT -replace '\\', '/')
if (-not $wtRegistered) {
    # also match native backslash form just in case
    $wtRegistered = ($wtPorcelain | Where-Object { $_ -match '^worktree ' } |
                     ForEach-Object { $_.Substring(9).Replace('/', '\') }) -contains $WT
}
if (-not $wtRegistered) {
    if (Test-Path $WT) {
        Fail "worktree dir exists but is not registered: $WT  (remove it, then re-run; git worktree add will recreate it)"
    }
    Write-Step "Creating persistent worktree: $WT (branch hf-spaces)"
    git worktree add $WT hf-spaces
    if ($LASTEXITCODE -ne 0) { Fail "git worktree add failed" }
} else {
    Write-Step "Using existing worktree: $WT"
}

# E: drive FS が ownership を記録しないため、worktree の gitdir が
# "dubious ownership" で拒否される (2026-07-29 DryRun 検証で実測)。
# safe.directory 例外を冪等に登録して自己修復する。
$wtSlash = $WT -replace '\\', '/'
$safeDirs = @(git config --global --get-all safe.directory 2>$null)
if ($safeDirs -notcontains $wtSlash) {
    Write-Step "Registering safe.directory exception: $wtSlash"
    git config --global --add safe.directory $wtSlash
}

$wtBranch = (git -C $WT rev-parse --abbrev-ref HEAD).Trim()
if ($wtBranch -ne "hf-spaces") {
    Fail "worktree $WT is on '$wtBranch', expected hf-spaces. Fix: git -C `"$WT`" checkout hf-spaces"
}

# 5. reset the worktree to a clean hf-spaces HEAD (it is a machine-managed
#    mirror; nothing hand-edited lives there, so hard reset + clean is safe)
git -C $WT reset --hard -q
git -C $WT clean -fdq

# 6. checkout sync paths from master into the worktree
Write-Step "Checking out $($allSyncPaths.Count) path(s) from master into worktree"
git -C $WT checkout master --pathspec-from-file=$specFile
if ($LASTEXITCODE -ne 0) { Fail "checkout from master into worktree failed" }

# 7. show status (display only)
Write-Step "Status on hf-spaces worktree:"
git -C $WT status --short

if ($DryRun) {
    Write-Step "DryRun: status shown above. Reverting worktree (no commit/push). master tree untouched."
    git -C $WT reset --hard -q
    git -C $WT clean -fdq
    exit 0
}

# 8. stage + commit. Verify the stage actually took (see weekly_post.ps1
#    2026-07-29 silent-add incident) instead of blindly trusting git add.
Write-Step "Committing"
git -C $WT add --pathspec-from-file=$specFile
if ($LASTEXITCODE -ne 0) { Fail "git add in worktree failed (exit $LASTEXITCODE)" }

$staged = @(git -C $WT diff --cached --name-only)
if ($staged.Count -gt 0) {
    $msg = "sync: master $($masterSha.Substring(0,7))"
    git -C $WT commit -m $msg
    if ($LASTEXITCODE -ne 0) { Fail "git commit in worktree failed" }
    Write-Host "    committed $($staged.Count) file(s)" -ForegroundColor Green
} else {
    Write-Host "    nothing to commit (worktree clean vs hf-spaces HEAD)" -ForegroundColor Yellow
}

# 9. push to HF (HF default branch is main)
# commit の有無に関わらず、local hf-spaces が remote hf/main より進んでる
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

Write-Step "Done (master tree never left $origBranch). Check build at https://huggingface.co/spaces/gutchi15300/pycaliAI"
