##############################################################
# jump_history_collect.ps1 --- 障害レース 履歴専用 collector 起動ラッパー
#
# ★Phase C 準備段階: **まだタスクスケジューラへ登録していない**。
#   本番接続もしていない。登録はユーザー承認後。
#
# 何をするか:
#   1) data/bunseki/{date}.csv から障害レースだけを raw_card layer へ append
#   2) data/kekka/{date}.csv があれば settled layer へ append (無ければ skip)
#   3) 成否をログへ記録。失敗時は非 0 で終了
#
# 何をしないか (hard invariant):
#   - 予測・印・買い目・bundle へ一切書かない
#   - T-35/T-20/T-10/Vote のタスクを作らない
#   - 原本 (data/bunseki, data/kekka) を変更しない
#   - HF 同期・git push をしない
#
# 出力先 (production が読まない専用 namespace):
#   data/history_only/jump/raw_card/{date}.jsonl
#   data/history_only/jump/settled/{date}.jsonl
#   data/history_only/jump/manifest.json
#   logs/jump_history_{date}.log
#
# 手動:
#   .\analysis\jump_history_only\jump_history_collect.ps1              # 当日
#   .\analysis\jump_history_only\jump_history_collect.ps1 -Date 20260905
#   .\analysis\jump_history_only\jump_history_collect.ps1 -All -Dry
##############################################################
param(
    [string]$Date = "",
    [switch]$All,
    [switch]$Dry
)
Set-Location 'E:\PyCaLiAI'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

$py = 'venv311\Scripts\python.exe'
if (-not (Test-Path $py)) { Write-Error "python が見つからない: $py"; exit 2 }

if (-not $Date -and -not $All) { $Date = (Get-Date).ToString('yyyyMMdd') }
$logDate = if ($All) { (Get-Date).ToString('yyyyMMdd') } else { $Date }

$logDir = 'logs'
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory $logDir | Out-Null }
$log = Join-Path $logDir "jump_history_$logDate.log"

function Write-Log($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg
    Write-Output $line
    Add-Content -Path $log -Value $line -Encoding utf8
}

Write-Log "=== jump history collector start (Date=$Date All=$All Dry=$Dry) ==="

# 収集対象日に bunseki が無ければ「何もしない正常終了」
if (-not $All) {
    $src = "data\bunseki\$Date.csv"
    if (-not (Test-Path $src)) {
        Write-Log "bunseki なし ($src) -> skip (正常終了)"
        exit 0
    }
}

$argsList = @('-m', 'analysis.jump_history_only.jump_history_collector')
if ($All) { $argsList += '--all' } else { $argsList += @('--date', $Date) }
if ($Dry) { $argsList += '--dry' }

Write-Log "run: $py $($argsList -join ' ')"
$out = & $py @argsList 2>&1
$code = $LASTEXITCODE
$out | ForEach-Object { Write-Log "  $_" }

if ($code -ne 0) {
    Write-Log "!! collector 失敗 (exit=$code)。append-only のため部分書込は発生しない。"
    Write-Log "!! CollisionError の場合は同一キーで内容が変化している。"
    Write-Log "!! 原本 bunseki/kekka を確認し、意図的な差し替えなら手動で判断すること。"
    exit $code
}

# 収集後に hard invariant テストを必ず回す (FAIL なら異常終了)
Write-Log "hard invariant test 実行"
$t = & $py -m pytest 'analysis/jump_history_only/test_jump_history_invariants.py' -q 2>&1
$tcode = $LASTEXITCODE
$t | Select-Object -Last 5 | ForEach-Object { Write-Log "  $_" }
if ($tcode -ne 0) {
    Write-Log "!! hard invariant FAIL -> 収集物を信用しない。特徴接続は絶対にしない。"
    exit $tcode
}

Write-Log "=== done (ok) ==="
exit 0
