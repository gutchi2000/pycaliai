##############################################################
# t10.ps1 --- 当日 T-10 自動馬券ライン起動ラッパー
#
# 手動起動:
#   .\t10.ps1                  # 最新 bundle の日付で起動
#   .\t10.ps1 20260614         # 日付指定
#   .\t10.ps1 20260614 -Dry    # 計算のみ (bets.json へ書き込まない)
#   .\t10.ps1 -LeadMin 12      # T-12 で処理
#
# ルーチン起動 (タスクスケジューラ "PyCaLiAI_T10" が土日 9:00 に呼ぶ):
#   .\t10.ps1 -Routine
#   → Date=今日に固定 / bundle 未生成なら Phase A 完了を 15:00 まで待機 /
#     logs\t10_{date}.log に全出力を記録 / 多重起動は .t10_lock でガード
#   ※ 祝日(月曜)開催はトリガー外なので手動で .\t10.ps1 を起動すること
#
# 中身: t10_runner.py が各レース発走 T-10 に
#   jvlink_odds.py (32-bit) → compute_bets.py --apply → validate_cowork_bets.py
# を自動実行し、買い目をコンソール表示する。投票は人間が IPAT で。
# HF への push はしない (ローカル NiceGUI は ui.timer で自動反映)。
##############################################################
param(
    [string]$Date = "",
    [double]$LeadMin = 10,
    [double]$MaxAgeMin = 20,
    [switch]$Dry,
    [switch]$Routine
)
Set-Location 'E:\PyCaLiAI'
$env:PYTHONUTF8 = '1'

$py = 'venv311\Scripts\python.exe'
if (-not (Test-Path $py)) { $py = 'python' }

if ($Routine) {
    if ($Date -eq "") { $Date = Get-Date -Format 'yyyyMMdd' }
    New-Item -ItemType Directory -Force logs | Out-Null
    try { Start-Transcript -Path ("logs\t10_{0}.log" -f $Date) -Append | Out-Null } catch {}
}

$argv = @('t10_runner.py')
if ($Date -ne "") { $argv += $Date }
$argv += @('--lead-min', $LeadMin, '--max-age-min', $MaxAgeMin)
if ($Dry) { $argv += '--dry' }
if ($Routine) { $argv += '--wait-bundle' }

& $py @argv
$code = $LASTEXITCODE

if ($Routine) { try { Stop-Transcript | Out-Null } catch {} }
exit $code
