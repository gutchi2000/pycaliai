##############################################################
# t15.ps1 --- T-15 直前補正印のサイト反映 (レース毎タスクの実体)
#
#   発走17分前に起動 (push→Cloudflare デプロイ 1-3分を見込み、表示は T-15 前後):
#   1. jvlink_odds.py (32-bit) でライブ単勝オッズ取得
#   2. publish_hosei.py で blend 補正印 → site/data/changes_{date}.json に追記
#   3. 差分 commit + push (pycaliai.com 自動デプロイ)
#
#   登録は t10.ps1 -Schedule が行う (PyCaLiAI_T15R_*)。
#   手動テスト: .\t15.ps1 -Once 2026080101010101 -Date 20260801 [-NoPush]
##############################################################
param(
    [string]$Once = "",
    [string]$Date = "",
    [switch]$NoPush
)
Set-Location 'E:\PyCaLiAI'
$env:PYTHONUTF8 = '1'
if ($Date -eq "") { $Date = Get-Date -Format 'yyyyMMdd' }
if ($Once -eq "") { Write-Host "[t15] -Once rid16 が必要"; exit 1 }

$py = 'venv311\Scripts\python.exe'
if (-not (Test-Path $py)) { $py = 'python' }
try { Start-Transcript -Path ("logs\t15_{0}.log" -f $Date) -Append | Out-Null } catch {}

& py -3.12-32 jvlink_odds.py --race $Once
& $py publish_hosei.py --date $Date --race $Once
if ($LASTEXITCODE -ne 0) {
    Write-Host "[t15] publish_hosei 失敗 → skip"
    try { Stop-Transcript | Out-Null } catch {}
    exit 0
}

$f = "site/data/changes_$Date.json"
if (Test-Path $f) {
    git add -- $f 2>$null
    $st = git status --porcelain -- $f
    if ($st) {
        git commit -m "site: T-15 補正印 $Date $Once" -- $f | Out-Null
        if (-not $NoPush) {
            git push origin master 2>&1 | Out-Null
            if ($LASTEXITCODE -eq 0) { Write-Host "[t15] push → pycaliai.com" }
            else { Write-Host "[t15] push 失敗" }
        }
    }
}
try { Stop-Transcript | Out-Null } catch {}
exit 0
