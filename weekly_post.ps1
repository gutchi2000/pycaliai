##############################################################
# weekly_post.ps1  ─── レース後ワークフロー
#
# 使い方:
#   .\weekly_post.ps1              # 最新の未コミット kekka CSV を自動検出
#   .\weekly_post.ps1 20260322     # 日付を直接指定
##############################################################
param([string]$Date = "")

Set-Location 'E:\PyCaLiAI'

# ── 日付を決定 ──────────────────────────────────────────
if ($Date -eq "") {
    $untracked = git status --short data/kekka/ 2>$null |
                 Where-Object { $_ -match '^\?\?' -or $_ -match '^A' } |
                 ForEach-Object { ($_ -replace '^\s*\S+\s+', '').Trim() } |
                 Where-Object { $_ -match 'data.kekka.\d{8}\.csv$' }

    if (-not $untracked) {
        $latest = Get-ChildItem 'data\kekka' -Filter '????????.csv' |
                  Sort-Object Name -Descending |
                  Select-Object -First 1
        if ($latest) {
            $Date = $latest.BaseName
            Write-Host "自動検出（最新）: $Date" -ForegroundColor Yellow
        } else {
            Write-Error "data\kekka\ に CSV が見つかりません。"
            exit 1
        }
    } else {
        $Date = [System.IO.Path]::GetFileNameWithoutExtension($untracked | Select-Object -Last 1)
        Write-Host "自動検出（未コミット）: $Date" -ForegroundColor Cyan
    }
}

$kekkaPath = "data\kekka\$Date.csv"
if (-not (Test-Path $kekkaPath)) {
    Write-Error "$kekkaPath が見つかりません。data\kekka\ に配置してください。"
    exit 1
}

Write-Host ""
Write-Host "=== 週次後半ワークフロー: $Date ===" -ForegroundColor Green
Write-Host ""

# ── Step 1: generate_results.py で results.json 再構築 ──
Write-Host "[1/3] generate_results.py 実行中..." -ForegroundColor Cyan
python generate_results.py
if ($LASTEXITCODE -ne 0) {
    Write-Error "generate_results.py が失敗しました。"
    exit 1
}
Write-Host "      → data\results.json 更新完了" -ForegroundColor Green

# ── Step 2: git add ──────────────────────────────────────
Write-Host "[2/3] git add..." -ForegroundColor Cyan
git add $kekkaPath data/results.json
Write-Host "      → $kekkaPath + results.json をステージング" -ForegroundColor Green

# ── Step 3: git commit & push ────────────────────────────
Write-Host "[3/3] git commit & push..." -ForegroundColor Cyan
$y = $Date.Substring(0,4)
$m = $Date.Substring(4,2)
$d = $Date.Substring(6,2)
git commit -m "add kekka $y-$m-$d / update results"
git push origin master
if ($LASTEXITCODE -ne 0) {
    Write-Error "git push が失敗しました。"
    exit 1
}

Write-Host ""
Write-Host "=== 完了 ===" -ForegroundColor Green
Write-Host "Streamlit Cloud の的中実績ページに反映されます（数十秒後）。" -ForegroundColor Gray
