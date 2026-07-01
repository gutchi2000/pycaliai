# =====================================================================
# sync-hf-static.ps1 - publish the static site (site/) to an HF static Space
# =====================================================================
# Separate from sync-hf.ps1 (NiceGUI / Docker Space). Pushes the contents of
# site/ (index.html, css/, js/, data/, README) to the ROOT of a NEW static
# Space (e.g. gutchi15300/pycaliai-web), running in parallel with the NiceGUI
# Space.
#
# It never touches the master working tree: it builds data, then mirrors site/
# into a persistent clone under TEMP and commits/pushes from there.
#
# Usage:
#   .\sync-hf-static.ps1 -SpaceUrl https://huggingface.co/spaces/<user>/<space>
#   .\sync-hf-static.ps1 -SpaceUrl <url> -DryRun   # assemble only, no push
#
# NOTE: ASCII-only on purpose (Windows PowerShell 5.1 misreads BOM-less UTF-8).
# =====================================================================
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$SpaceUrl,
    [switch]$DryRun
)
$ErrorActionPreference = "Continue"
# $PSScriptRoot is empty when the body is run inline (not as a .ps1 file);
# fall back to the current directory so Join-Path never gets an empty Path.
$ROOT = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
$SITE = Join-Path $ROOT "site"
$STAGE = Join-Path $env:TEMP "pycaliai_web_deploy"

function Step($m) { Write-Host "==> $m" -ForegroundColor Cyan }
function Fail($m) { Write-Host "ERROR: $m" -ForegroundColor Red; exit 1 }

# 1. regenerate data
Step "build_site.py (regenerate data)"
& (Join-Path $ROOT "venv311\Scripts\python.exe") (Join-Path $ROOT "build_site.py")
if ($LASTEXITCODE -ne 0) { Fail "build_site.py failed" }

foreach ($f in @("index.html", "css\style.css", "js\app.js", "data\manifest.json", "README_hf.md")) {
    if (-not (Test-Path (Join-Path $SITE $f))) { Fail "missing site\$f" }
}

# 2. ensure a local clone of the Space exists
if (-not (Test-Path (Join-Path $STAGE ".git"))) {
    Step "clone Space: $SpaceUrl"
    if (Test-Path $STAGE) { Remove-Item $STAGE -Recurse -Force }
    git clone $SpaceUrl $STAGE
    if ($LASTEXITCODE -ne 0) { Fail "clone failed (Space exists? URL correct?)" }
} else {
    Step "update existing clone (git pull)"
    git -C $STAGE pull --rebase 2>$null
}

# 3. replace staging contents with site/ (keep .git)
Step "mirror site/ into staging root"
Get-ChildItem $STAGE -Force | Where-Object { $_.Name -ne ".git" } |
    Remove-Item -Recurse -Force
Copy-Item (Join-Path $SITE "index.html") $STAGE
Copy-Item (Join-Path $SITE "css") $STAGE -Recurse
Copy-Item (Join-Path $SITE "js") $STAGE -Recurse
Copy-Item (Join-Path $SITE "data") $STAGE -Recurse
Copy-Item (Join-Path $SITE "README_hf.md") (Join-Path $STAGE "README.md")

$nFiles = (Get-ChildItem (Join-Path $STAGE "data") -File).Count
Step "data/*.json placed: $nFiles files"

# 4. commit
$sha = (git -C $ROOT rev-parse --short HEAD).Trim()
git -C $STAGE add -A
$pending = git -C $STAGE status --porcelain
if (-not $pending) {
    Step "no changes (nothing to push)"
    exit 0
}

if ($DryRun) {
    Step "DryRun: staging assembled. diff:"
    git -C $STAGE status --short
    Write-Host "  (no push performed)" -ForegroundColor Yellow
    exit 0
}

git -C $STAGE commit -m "deploy: site $sha"
if ($LASTEXITCODE -ne 0) { Fail "commit failed" }

# 5. push
Step "push to HF static Space"
git -C $STAGE push origin HEAD:main
if ($LASTEXITCODE -ne 0) {
    git -C $STAGE push origin HEAD:master
    if ($LASTEXITCODE -ne 0) { Fail "push failed; retry manually: git -C `"$STAGE`" push origin HEAD:main" }
}

Step "done. live in ~30s: $SpaceUrl"
