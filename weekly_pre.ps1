##############################################################
# weekly_pre.ps1  ─── レース前ワークフロー
#
# 使い方:
#   .\weekly_pre.ps1              # 最新の未コミット weekly CSV を自動検出
#   .\weekly_pre.ps1 20260322     # 日付を直接指定
##############################################################
param([string]$Date = "")

Set-Location 'E:\PyCaLiAI'

# ── 日付を決定 ──────────────────────────────────────────
if ($Date -eq "") {
    # git status で未コミットの data/weekly/*.csv を検出
    $untracked = git status --short data/weekly/ 2>$null |
                 Where-Object { $_ -match '^\?\?' -or $_ -match '^A' -or $_ -match '^ M' } |
                 ForEach-Object { ($_ -replace '^\s*\S+\s+', '').Trim() } |
                 Where-Object { $_ -match 'data.weekly.\d{8}\.csv$' }

    if (-not $untracked) {
        # 未コミットがなければ最新ファイルを使用
        $latest = Get-ChildItem 'data\weekly' -Filter '????????.csv' |
                  Sort-Object Name -Descending |
                  Select-Object -First 1
        if ($latest) {
            $Date = $latest.BaseName
            Write-Host "自動検出（最新）: $Date" -ForegroundColor Yellow
        } else {
            Write-Error "data\weekly\ に CSV が見つかりません。"
            exit 1
        }
    } else {
        $Date = [System.IO.Path]::GetFileNameWithoutExtension($untracked | Select-Object -Last 1)
        Write-Host "自動検出（未コミット）: $Date" -ForegroundColor Cyan
    }
}

$csvPath = "data\weekly\$Date.csv"
if (-not (Test-Path $csvPath)) {
    Write-Error "$csvPath が見つかりません。data\weekly\ に配置してください。"
    exit 1
}

Write-Host ""
Write-Host "=== 週次前半ワークフロー: $Date ===" -ForegroundColor Green
Write-Host ""

# ── Step 1: predict_weekly.py で pred CSV 生成 ──────────
Write-Host "[1/3] predict_weekly.py 実行中..." -ForegroundColor Cyan
python predict_weekly.py --csv $csvPath
if ($LASTEXITCODE -ne 0) {
    Write-Error "predict_weekly.py が失敗しました。"
    exit 1
}
Write-Host "      → reports\pred_$Date.csv 生成完了" -ForegroundColor Green

# ── Step 2: git add ──────────────────────────────────────
Write-Host "[2/3] git add..." -ForegroundColor Cyan
git add $csvPath
Write-Host "      → $csvPath をステージング" -ForegroundColor Green

# ── Step 3: git commit & push ────────────────────────────
Write-Host "[3/3] git commit & push..." -ForegroundColor Cyan
$y = $Date.Substring(0,4)
$m = $Date.Substring(4,2)
$d = $Date.Substring(6,2)
git commit -m "add weekly csv $y-$m-$d"
git push origin master
if ($LASTEXITCODE -ne 0) {
    Write-Error "git push が失敗しました。"
    exit 1
}

Write-Host ""
Write-Host "=== 完了 ===" -ForegroundColor Green
Write-Host "Streamlit Cloud が更新されます（数十秒後）。" -ForegroundColor Gray
Write-Host "URLを開いて買い目を確認してください → pred CSV が自動保存されます。" -ForegroundColor Gray
