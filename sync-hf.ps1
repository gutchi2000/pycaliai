# =====================================================================
# sync-hf.ps1 - master → hf-spaces 同期 + HuggingFace Spaces デプロイ
# =====================================================================
# 使い方:
#   .\sync-hf.ps1            # master の HF 関連ファイルを hf-spaces に
#                            # コピーして commit + hf/main へ push
#   .\sync-hf.ps1 -DryRun    # commit/push せず差分のみ表示
#
# 前提:
#   - hf-spaces ブランチが orphan で既に存在 (作成済み)
#   - hf remote が https://huggingface.co/spaces/USERNAME/pycaliAI 設定済み
#
# 仕組み:
#   hf-spaces は orphan branch (master と履歴が独立)。
#   そのため git merge では同期不可。代わりに「HF に必要なファイルだけ」
#   master からチェックアウトして hf-spaces 上で再コミットする。
# =====================================================================

[CmdletBinding()]
param(
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

# HF Spaces にデプロイするファイル一覧
# (data/, reports/ は中身が大きいため Dockerfile の COPY で事前 .dockerignore
#  済み。ここでは小さいファイルのみリポジトリに含める)
$SyncFiles = @(
    "Dockerfile",
    "README.md",
    ".dockerignore",
    "requirements-nicegui.txt",
    "nicegui_app.py"
)

function Write-Step($msg) {
    Write-Host "==> $msg" -ForegroundColor Cyan
}

function Fail($msg) {
    Write-Host "ERROR: $msg" -ForegroundColor Red
    exit 1
}

# 1. 現在のブランチを記録
$origBranch = (git rev-parse --abbrev-ref HEAD).Trim()
Write-Step "現在のブランチ: $origBranch"

if ($origBranch -ne "master") {
    Fail "master ブランチで実行してください (現在: $origBranch)"
}

# 2. master の最新 commit を取得
$masterSha = (git rev-parse HEAD).Trim()
Write-Step "master HEAD: $masterSha"

# 3. 同期対象ファイルが master に存在するか検証
foreach ($f in $SyncFiles) {
    if (-not (Test-Path $f)) {
        Fail "同期対象ファイルが存在しません: $f"
    }
}
Write-Step "同期対象 $($SyncFiles.Count) ファイル確認 OK"

# 4. hf-spaces ブランチに切替
Write-Step "hf-spaces ブランチに切替"
git checkout hf-spaces
if ($LASTEXITCODE -ne 0) { Fail "hf-spaces 切替失敗" }

# 5. master から HF 用ファイルをチェックアウト
Write-Step "master からファイル取得"
foreach ($f in $SyncFiles) {
    git checkout master -- $f
    if ($LASTEXITCODE -ne 0) {
        git checkout $origBranch
        Fail "$f のチェックアウトに失敗"
    }
}

# 6. 差分確認
Write-Step "hf-spaces 上の差分:"
git status --short

$hasDiff = (git status --porcelain).Length -gt 0
if (-not $hasDiff) {
    Write-Step "差分なし。同期はスキップして元のブランチに戻ります"
    git checkout $origBranch
    exit 0
}

if ($DryRun) {
    Write-Step "DryRun: ここで終了。変更は staged のまま。手動で commit/reset してください"
    Write-Host "  commit  → git commit -m 'sync from master'"
    Write-Host "  push    → git push hf hf-spaces:main"
    Write-Host "  rollback→ git checkout . ; git checkout $origBranch"
    exit 0
}

# 7. commit
Write-Step "commit"
git add $SyncFiles
$msg = "sync: master $($masterSha.Substring(0,7)) からの更新を反映"
git commit -m $msg
if ($LASTEXITCODE -ne 0) { Fail "commit 失敗" }

# 8. push to HF (HF のデフォルトブランチは main)
Write-Step "HuggingFace Spaces (hf/main) へ push"
git push hf hf-spaces:main
if ($LASTEXITCODE -ne 0) {
    Write-Host "push 失敗。手動で再試行: git push hf hf-spaces:main" -ForegroundColor Yellow
}

# 9. 元ブランチに戻る
Write-Step "$origBranch に戻る"
git checkout $origBranch

Write-Step "完了。https://huggingface.co/spaces/gutchi15300/pycaliAI でビルド状況を確認"
