##############################################################
# changes.ps1 --- 当日変更情報の取得→サイト反映 (取消/騎手変更/時刻/馬体重)
#
#   jvlink_changes.py (32-bit) → site/data/changes_{date}.json
#   → 差分があれば commit + push (Cloudflare Workers が push で自動デプロイ
#     = pycaliai.com に数分で反映。HF Space は週次 sync のままで触らない)
#
# 起動: t10.ps1 の -Schedule (朝一) と -Once (各レース T-10) から自動で呼ばれる。
#       手動: .\changes.ps1 [-Date 20260801] [-NoPush] [-DumpRaw]
##############################################################
param(
    [string]$Date = "",
    [switch]$NoPush,
    [switch]$DumpRaw
)
Set-Location 'E:\PyCaLiAI'
$env:PYTHONUTF8 = '1'
if ($Date -eq "") { $Date = Get-Date -Format 'yyyyMMdd' }

$argv = @('jvlink_changes.py', $Date)
if ($DumpRaw) { $argv += '--dump-raw' }
& py -3.12-32 @argv
if ($LASTEXITCODE -ne 0) {
    Write-Host "[changes] jvlink_changes 失敗 (exit $LASTEXITCODE) → サイト反映スキップ"
    exit 0   # T-10 本流 (オッズ/買い目) を巻き添えにしない
}

$f = "site/data/changes_$Date.json"
if (-not (Test-Path $f)) { exit 0 }

# 内容が変わった時だけ commit+push
git add -- $f 2>$null
$st = git status --porcelain -- $f
if ($st) {
    git commit -m "site: 当日変更情報 $Date (取消/騎手変更/時刻/馬体重)" -- $f | Out-Null
    Write-Host "[changes] commit $f"
    if (-not $NoPush) {
        git push origin master 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) { Write-Host "[changes] push → pycaliai.com 自動デプロイ" }
        else { Write-Host "[changes] push 失敗 (次回レースで再試行)" }
    }
} else {
    Write-Host "[changes] 変更なし"
}
