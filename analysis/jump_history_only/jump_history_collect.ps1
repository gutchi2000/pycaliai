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

# 開催日判定と bunseki 有無の扱いは collector 側 (Python) が行う。
#   非開催日 & bunseki なし          -> exit 0
#   開催日   & bunseki なし          -> exit 3 (MISSING_BUNSEKI_EXPORT)
#   障害があるはずなのに raw card 0  -> exit 4 (JUMP_RACE_MISSING)
#   venue coverage 異常              -> exit 5 (COVERAGE_ANOMALY)
#   キー衝突                         -> exit 6 (COLLISION)
# ここで「bunseki が無ければ常に exit 0」としてはいけない (欠落を見逃すため)。

$argsList = @('-m', 'analysis.jump_history_only.jump_history_collector')
if ($All) { $argsList += '--all' } else { $argsList += @('--date', $Date) }
if ($Dry) { $argsList += '--dry' }

Write-Log "run: $py $($argsList -join ' ')"
$out = & $py @argsList 2>&1
$code = $LASTEXITCODE
$out | ForEach-Object { Write-Log "  $_" }

if ($code -ne 0) {
    $meaning = switch ($code) {
        3 { "MISSING_BUNSEKI_EXPORT — 開催日なのに bunseki が無い。TARGET『出走馬分析』を export し data\_inbox へ置いて place_weekly.py を走らせること" }
        4 { "JUMP_RACE_MISSING — 他ソースが障害レースを示すのに raw card が 0 件" }
        5 { "COVERAGE_ANOMALY — venue ごとの race/horse 数が異常" }
        6 { "COLLISION — 同一キーで内容が変化。append-only のため全体停止 (部分書込なし)" }
        default { "collector 失敗" }
    }
    Write-Log "!! exit=$code : $meaning"
    Add-Content -Path (Join-Path $logDir "jump_history_error.log") -Encoding utf8 `
        -Value ("[{0}] date={1} exit={2} {3}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $logDate, $code, $meaning)
    exit $code
}

# 収集後に hard invariant テストを必ず回す (FAIL なら異常終了)
Write-Log "hard invariant test 実行"
$t = & $py -m pytest 'analysis/jump_history_only/test_jump_history_invariants.py' `
         'tests/test_jump_race_p0_gate.py' -q 2>&1
$tcode = $LASTEXITCODE
$t | Select-Object -Last 5 | ForEach-Object { Write-Log "  $_" }
if ($tcode -ne 0) {
    Write-Log "!! hard invariant / P0 gate FAIL -> 収集物を信用しない。特徴接続は絶対にしない。"
    Add-Content -Path (Join-Path $logDir "jump_history_error.log") -Encoding utf8 `
        -Value ("[{0}] date={1} TEST_FAIL exit={2}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $logDate, $tcode)
    exit $tcode
}

Write-Log "=== done (ok) ==="
exit 0
