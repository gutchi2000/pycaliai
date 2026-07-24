# =====================================================================
# sync-hf-umami.ps1 - publish the static site (site/) to a DOCKER HF Space
# =====================================================================
# Target Space : gutchi15300/pycaliai-umami (sdk: docker)
# Live URL     : https://gutchi15300-pycaliai-umami.hf.space  (no .static)
#
# Fully independent from:
#   - sync-hf.ps1         (NiceGUI Docker Space gutchi15300/pycaliai)
#   - sync-hf-static.ps1  (static-SDK Space gutchi15300/pycaliai-web)
# It never touches the master working tree or those Spaces. It builds data,
# then mirrors site/ + Dockerfile + README into a TEMP clone and pushes.
#
# Prereq: create the Space once on HF (New Space -> SDK: Docker -> Blank).
#
# Usage:
#   .\sync-hf-umami.ps1
#   .\sync-hf-umami.ps1 -DryRun        # assemble only, no push
#   .\sync-hf-umami.ps1 -SpaceUrl https://huggingface.co/spaces/<user>/<space>
#
# NOTE: ASCII-only on purpose (Windows PowerShell 5.1 misreads BOM-less UTF-8).
# =====================================================================
[CmdletBinding()]
param(
    [string]$SpaceUrl = "https://huggingface.co/spaces/gutchi15300/pycaliai-umami",
    [switch]$DryRun
)
$ErrorActionPreference = "Continue"
# $PSScriptRoot is empty when run inline; fall back to cwd so Join-Path is safe.
$ROOT = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
$SITE = Join-Path $ROOT "site"
$DOCKER = Join-Path $ROOT "deploy\pycaliai-umami"
$STAGE = Join-Path $env:TEMP "pycaliai_umami_deploy"

function Step($m) { Write-Host "==> $m" -ForegroundColor Cyan }
function Fail($m) { Write-Host "ERROR: $m" -ForegroundColor Red; exit 1 }

# 0. best-effort: refresh today's track bias so a deploy NEVER ships last week's
#    baba (baba.js hides any baba whose date != viewer's today, so a stale file
#    = blank panel). JRA posts race-day cushion ~08:20; a pre-08:20 run yields
#    yesterday's read (shown as "prev-day"), which the 08:30 PyCaLiAI_Baba repeat
#    later upgrades to today's measurement. A JRA hiccup must never block publish:
#    fetch_baba_today.py raises on failure and leaves the existing file intact,
#    and we swallow any nonzero exit and deploy whatever baba_today.json exists.
Step "refresh baba_today.json (best-effort)"
try {
    & (Join-Path $ROOT "venv311\Scripts\python.exe") (Join-Path $ROOT "fetch_baba_today.py")
    if ($LASTEXITCODE -ne 0) { Write-Host "  baba fetch nonzero exit ($LASTEXITCODE); deploying existing baba_today.json" -ForegroundColor Yellow }
} catch {
    Write-Host "  baba fetch threw ($($_.Exception.Message)); deploying existing baba_today.json" -ForegroundColor Yellow
}

# 1. regenerate data
Step "build_site.py (regenerate data)"
& (Join-Path $ROOT "venv311\Scripts\python.exe") (Join-Path $ROOT "build_site.py")
if ($LASTEXITCODE -ne 0) { Fail "build_site.py failed" }

foreach ($f in @("index.html", "explain.html", "css\style.css", "js\app.js", "data\manifest.json",
                 "manifest.webmanifest", "sw.js", "apple-touch-icon.png", "favicon.ico",
                 "icons\icon-192.png", "icons\icon-512.png")) {
    if (-not (Test-Path (Join-Path $SITE $f))) { Fail "missing site\$f" }
}
foreach ($f in @("Dockerfile", "README.md", "server.py")) {
    if (-not (Test-Path (Join-Path $DOCKER $f))) { Fail "missing deploy\pycaliai-umami\$f" }
}

# 2. ensure a local clone of the Space exists
if (-not (Test-Path (Join-Path $STAGE ".git"))) {
    Step "clone Space: $SpaceUrl"
    if (Test-Path $STAGE) { Remove-Item $STAGE -Recurse -Force }
    git clone $SpaceUrl $STAGE
    if ($LASTEXITCODE -ne 0) { Fail "clone failed. Create the Space first on HF (New Space -> Docker)." }
} else {
    Step "update existing clone (git pull)"
    git -C $STAGE pull --rebase
}

# 3. replace staging contents (keep .git): Dockerfile + README + site/
Step "mirror Dockerfile + README + site/ into staging root"
Get-ChildItem $STAGE -Force | Where-Object { $_.Name -ne ".git" } |
    Remove-Item -Recurse -Force
Copy-Item (Join-Path $DOCKER "Dockerfile") $STAGE
Copy-Item (Join-Path $DOCKER "README.md") $STAGE
Copy-Item (Join-Path $DOCKER "server.py") $STAGE
Copy-Item (Join-Path $SITE "*.html") $STAGE
Copy-Item (Join-Path $SITE "css") $STAGE -Recurse
Copy-Item (Join-Path $SITE "js") $STAGE -Recurse
Copy-Item (Join-Path $SITE "data") $STAGE -Recurse
Copy-Item (Join-Path $SITE "icons") $STAGE -Recurse
Copy-Item (Join-Path $SITE "manifest.webmanifest") $STAGE
Copy-Item (Join-Path $SITE "sw.js") $STAGE
Copy-Item (Join-Path $SITE "apple-touch-icon.png") $STAGE
Copy-Item (Join-Path $SITE "favicon.ico") $STAGE

$nFiles = (Get-ChildItem (Join-Path $STAGE "data") -File).Count
Step "data/*.json placed: $nFiles files"

$sha = (git -C $ROOT rev-parse --short HEAD).Trim()

# 3.5 auto cache-bust: rewrite asset ?v= tokens in staged HTML to the repo sha.
#     Staging-only on purpose (this script never edits the master working tree).
#     Prevents the 2026-07-24 incident: JS/CSS deployed but browsers kept the old
#     cached file because the manual ?v= in index.html was not bumped.
Step "cache-bust staged HTML: ?v=$sha"
foreach ($html in Get-ChildItem $STAGE -Filter "*.html" -File) {
    $txt = [System.IO.File]::ReadAllText($html.FullName)
    $new = $txt -replace '(\.(css|js)\?v=)[A-Za-z0-9_.-]+', ('${1}' + $sha)
    if ($new -ne $txt) {
        [System.IO.File]::WriteAllText($html.FullName, $new, (New-Object System.Text.UTF8Encoding $false))
        Write-Host "  $($html.Name): bumped" -ForegroundColor DarkGray
    }
}

# 4. commit
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
Step "push to HF Docker Space"
git -C $STAGE push origin HEAD:main
if ($LASTEXITCODE -ne 0) {
    git -C $STAGE push origin HEAD:master
    if ($LASTEXITCODE -ne 0) { Fail "push failed; retry manually: git -C `"$STAGE`" push origin HEAD:main" }
}

Step "done. Docker build takes ~1-2 min, then live: $SpaceUrl"
