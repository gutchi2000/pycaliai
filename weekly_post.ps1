##############################################################
# weekly_post.ps1  --- post-race workflow
#
# Usage:
#   .\weekly_post.ps1              # auto-detect latest uncommitted kekka CSV
#   .\weekly_post.ps1 20260322     # specify date
##############################################################
param([string]$Date = "")

Set-Location 'E:\PyCaLiAI'

# -- Determine date --
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
            Write-Host "Auto-detect (latest): $Date" -ForegroundColor Yellow
        } else {
            Write-Error "No CSV found in data\kekka\"
            exit 1
        }
    } else {
        $Date = [System.IO.Path]::GetFileNameWithoutExtension(
            ($untracked | Select-Object -Last 1))
        Write-Host "Auto-detect (uncommitted): $Date" -ForegroundColor Cyan
    }
}

$kekkaPath = "data\kekka\$Date.csv"
if (-not (Test-Path $kekkaPath)) {
    Write-Error "$kekkaPath not found. Place it in data\kekka\ first."
    exit 1
}

Write-Host ""
Write-Host "=== weekly_post: $Date ===" -ForegroundColor Green
Write-Host ""

# -- Step 1: rebuild results.json --
Write-Host "[1/4] Running generate_results.py ..." -ForegroundColor Cyan
python generate_results.py
if ($LASTEXITCODE -ne 0) {
    Write-Error "generate_results.py failed."
    exit 1
}
Write-Host "      data\results.json updated." -ForegroundColor Green

# -- Step 2: live_results_2026.csv に実績を照合 --
Write-Host "[2/4] Updating live_results_2026.csv ..." -ForegroundColor Cyan
python update_live_results.py --date $Date
if ($LASTEXITCODE -ne 0) {
    Write-Warning "update_live_results.py failed (non-fatal, continuing...)"
}
Write-Host "      live_results_2026.csv updated." -ForegroundColor Green

# -- Step 2.5: 馬ごと履歴 parquet 更新 (serve 履歴特徴の 2026 追随) --
# export_weekly_marks の hist_same_*/course_*/jockey_* 再計算が読む
# data/_horse_history.parquet に今週の kekka を取り込む。失敗しても
# 翌週 Phase A で「parquet が古い」warning が出るだけなので non-fatal。
Write-Host "[2.5/4] Updating _horse_history.parquet ..." -ForegroundColor Cyan
python build_horse_history.py
if ($LASTEXITCODE -ne 0) {
    Write-Warning "build_horse_history.py failed (non-fatal, continuing...)"
} else {
    Write-Host "      data\_horse_history.parquet updated." -ForegroundColor Green
}

# -- Step 2.7: 決済ドリフト係数の鮮度管理 (SettleAI Phase 0, 2026-07-31) --
# reports/settle_drift.json が 28 日より古ければ live_odds 全量で再実測、
# 毎週は --check で compute_bets の SETTLE_DRIFT_* 配線値との乖離を警告 (non-fatal)。
Write-Host "[2.7/4] Settle drift freshness ..." -ForegroundColor Cyan
$driftJson = "reports\settle_drift.json"
$needRefit = (-not (Test-Path $driftJson)) -or
             ((Get-Date) - (Get-Item $driftJson).LastWriteTime).Days -ge 28
if ($needRefit) {
    Write-Host "      settle_drift.json stale (>28d) - re-measuring ..." -ForegroundColor Yellow
    python -m analysis.measure_settle_drift --notify
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "measure_settle_drift failed (non-fatal, continuing...)"
    }
}
python -m analysis.measure_settle_drift --check --notify
if ($LASTEXITCODE -ne 0) {
    Write-Warning "settle drift check failed (non-fatal, continuing...)"
}

# -- Step 3: git add (verified) --
Write-Host "[3/4] git add ..." -ForegroundColor Cyan
# data/cowork_results.json は generate_results.py が更新する Cowork 集計。
# これを add し忘れると HF/master に反映されず、ダッシュボードの累計 P/L が凍結する
# (2026-05-26 凍結事故の原因)。必ず含める。
#
# 2026-07-29 ガード追加: この git add が「staged:」表示にもかかわらず実際には
# 何もステージせず、直後の commit が "no changes added to commit" で素通りする
# 事故が慢性化していた (7/12 以降 3 週分の kekka/results/cowork_results が
# 未コミットのまま集計凍結ガードをすり抜けた)。同じ add を手動で打つと成功する
# ためタイミング/状態依存で単独原因は未確定。以後は add の結果を
# `git diff --cached` で実測検証し、0 件なら 1 回リトライ、それでも 0 件かつ
# 変更が残っているなら fail-hard (exit 1) で停止する。exit 1 は
# weekly_nicegui.ps1 の fail-hard に乗り、HF 同期ごと中止される。
$addTargets = @(
    $kekkaPath.Replace('\','/'),
    'data/results.json',
    'data/live_results_2026.csv',
    'data/cowork_results.json'
)
$coworkBetsDir = "reports\cowork_bets\$Date"
if (Test-Path $coworkBetsDir) {
    $addTargets += "reports/cowork_bets/$Date"
}

function Invoke-GitAddVerified([string[]]$targets) {
    # Out-Host: git add の stdout が関数戻り値 ($staged) に混入するのを防ぐ
    git add -- $targets | Out-Host
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "git add exited with code $LASTEXITCODE"
    }
    return ,@(git diff --cached --name-only)
}

$staged  = Invoke-GitAddVerified $addTargets
$pending = @(git status --porcelain -- $addTargets)
if ($staged.Count -eq 0 -and $pending.Count -gt 0) {
    Write-Warning "git add staged 0 files but $($pending.Count) change(s) are pending:"
    $pending | ForEach-Object { Write-Host "      pending: $_" -ForegroundColor Yellow }
    Write-Host "      retrying git add in 3s ..." -ForegroundColor Yellow
    Start-Sleep -Seconds 3
    $staged = Invoke-GitAddVerified $addTargets
    if ($staged.Count -eq 0) {
        Write-Error ("git add staged nothing twice while changes are pending. " +
            "Aborting before commit/push (fail-hard). Run manually and inspect: " +
            "git add -- $($addTargets -join ' ')")
        exit 1
    }
}

if ($staged.Count -gt 0) {
    Write-Host "      staged $($staged.Count) file(s): $($staged -join ', ')" -ForegroundColor Green
} else {
    Write-Host "      nothing to stage (targets already committed) - commit will be skipped" -ForegroundColor Yellow
}

# -- Step 4: git commit & push --
Write-Host "[4/4] git commit & push ..." -ForegroundColor Cyan
$y = $Date.Substring(0,4)
$m = $Date.Substring(4,2)
$d = $Date.Substring(6,2)
if ($staged.Count -gt 0) {
    git commit -m "add kekka $y-$m-$d / update results"
    if ($LASTEXITCODE -ne 0) {
        Write-Error "git commit failed (staged $($staged.Count) file(s) but commit returned $LASTEXITCODE)."
        exit 1
    }
}
git pull --rebase --autostash origin master
git push origin HEAD:master
if ($LASTEXITCODE -ne 0) {
    Write-Error "git push failed."
    exit 1
}

Write-Host ""
Write-Host "=== Done ===" -ForegroundColor Green
Write-Host "Streamlit Cloud results page will update in a few seconds." -ForegroundColor Gray

# -- Value Model月次再学習（月初の日曜のみ）--
$dow = (Get-Date).DayOfWeek   # Sunday=0
$dayOfMonth = (Get-Date).Day

if ($dow -eq 0 -and $dayOfMonth -le 7) {
    Write-Host ""
    Write-Host "=== Value Model月次再学習 ===" -ForegroundColor Yellow
    $endDate = (Get-Date -Format "yyyyMMdd")
    python retrain_value_model.py --end-date $endDate
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "retrain_value_model.py failed (non-fatal, continuing...)"
    }
}

# -- 週次監査（毎週日曜に自動実行）--
if ($dow -eq 0) {
    Write-Host ""
    Write-Host "=== 週次監査を実行します ===" -ForegroundColor Magenta
    & "$PSScriptRoot\run_audit.ps1"
}
