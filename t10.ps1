##############################################################
# t10.ps1 --- 当日 T-10 自動馬券ライン起動ラッパー
#
# 土日の朝 (Phase A 完了後)、レース開始前に 1 回起動しておく:
#   .\t10.ps1                  # 最新 bundle の日付で起動
#   .\t10.ps1 20260614         # 日付指定
#   .\t10.ps1 20260614 -Dry    # 計算のみ (bets.json へ書き込まない)
#   .\t10.ps1 -LeadMin 12      # T-12 で処理
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
    [switch]$Dry
)
Set-Location 'E:\PyCaLiAI'
$env:PYTHONUTF8 = '1'

$py = 'venv311\Scripts\python.exe'
if (-not (Test-Path $py)) { $py = 'python' }

$argv = @('t10_runner.py')
if ($Date -ne "") { $argv += $Date }
$argv += @('--lead-min', $LeadMin, '--max-age-min', $MaxAgeMin)
if ($Dry) { $argv += '--dry' }

& $py @argv
exit $LASTEXITCODE
