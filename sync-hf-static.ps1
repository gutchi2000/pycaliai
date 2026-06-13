# =====================================================================
# sync-hf-static.ps1 — 静的サイト (site/) を HF static Space に公開
# =====================================================================
# 既存の sync-hf.ps1 (NiceGUI / Docker Space) とは別物。
# 新規に作った static Space (例: gutchi15300/pycaliai-web) に site/ の中身を
# ルート配置で push する。
#
# 仕組み:
#   master を一切触らず、$STAGE (TEMP 下のクローン) に site/ を流し込んで
#   commit & push する。orphan ブランチ操作も master の checkout 切替も無し
#   → master の作業ツリーは安全。
#
# 使い方:
#   .\sync-hf-static.ps1 -SpaceUrl https://huggingface.co/spaces/<user>/<space>
#   .\sync-hf-static.ps1 -SpaceUrl <url> -DryRun   # 組立だけ (push しない)
#
# 初回は Space を HF 側で作成しておくこと (sdk: static)。
# =====================================================================
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$SpaceUrl,
    [switch]$DryRun
)
$ErrorActionPreference = "Continue"
$ROOT = $PSScriptRoot
$SITE = Join-Path $ROOT "site"
$STAGE = Join-Path $env:TEMP "pycaliai_web_deploy"

function Step($m) { Write-Host "==> $m" -ForegroundColor Cyan }
function Fail($m) { Write-Host "ERROR: $m" -ForegroundColor Red; exit 1 }

# 1. データ生成
Step "build_site.py でデータ再生成"
& (Join-Path $ROOT "venv311\Scripts\python.exe") (Join-Path $ROOT "build_site.py")
if ($LASTEXITCODE -ne 0) { Fail "build_site.py に失敗" }

foreach ($f in @("index.html", "css\style.css", "js\app.js", "data\manifest.json", "README_hf.md")) {
    if (-not (Test-Path (Join-Path $SITE $f))) { Fail "site\$f が見つからない" }
}

# 2. staging (Space のローカルクローン) を用意
if (-not (Test-Path (Join-Path $STAGE ".git"))) {
    Step "Space を clone: $SpaceUrl -> $STAGE"
    if (Test-Path $STAGE) { Remove-Item $STAGE -Recurse -Force }
    git clone $SpaceUrl $STAGE
    if ($LASTEXITCODE -ne 0) { Fail "clone 失敗。Space を HF 側で作成済みか / URL を確認" }
} else {
    Step "staging 更新 (git pull)"
    git -C $STAGE pull --rebase 2>$null
}

# 3. staging の中身を入れ替え (.git は残す)
Step "staging を site/ の内容で更新"
Get-ChildItem $STAGE -Force | Where-Object { $_.Name -ne ".git" } |
    Remove-Item -Recurse -Force
Copy-Item (Join-Path $SITE "index.html") $STAGE
Copy-Item (Join-Path $SITE "css") $STAGE -Recurse
Copy-Item (Join-Path $SITE "js") $STAGE -Recurse
Copy-Item (Join-Path $SITE "data") $STAGE -Recurse
Copy-Item (Join-Path $SITE "README_hf.md") (Join-Path $STAGE "README.md")
# Git LFS 誤適用を避けるため .gitattributes は置かない (json/js/css/html は通常 blob)

$nFiles = (Get-ChildItem (Join-Path $STAGE "data") -File).Count
Step "data/*.json: $nFiles 件配置"

# 4. commit
$sha = (git -C $ROOT rev-parse --short HEAD).Trim()
git -C $STAGE add -A
$pending = git -C $STAGE status --porcelain
if (-not $pending) {
    Step "変更なし (push 不要)"
    exit 0
}

if ($DryRun) {
    Step "DryRun: staging 組立完了。差分:"
    git -C $STAGE status --short
    Write-Host "  (push は行いませんでした)" -ForegroundColor Yellow
    exit 0
}

git -C $STAGE commit -m "deploy: site $sha"
if ($LASTEXITCODE -ne 0) { Fail "commit 失敗" }

# 5. push
Step "HF static Space へ push"
git -C $STAGE push origin HEAD:main
if ($LASTEXITCODE -ne 0) {
    git -C $STAGE push origin HEAD:master
    if ($LASTEXITCODE -ne 0) { Fail "push 失敗。手動: git -C `"$STAGE`" push origin HEAD:main" }
}

Step "完了。数十秒で反映: $SpaceUrl"
