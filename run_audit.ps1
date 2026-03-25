##############################################################
# run_audit.ps1  --- keiba-auditor を起動する
#
# Usage:
#   .\run_audit.ps1          # 手動実行
#   weekly_post.ps1 から月次自動呼び出し
##############################################################

Set-Location 'E:\PyCaLiAI'

$today    = Get-Date -Format 'yyyyMMdd'
$logFile  = "reports\audit_$today.md"

Write-Host ""
Write-Host "=== keiba-auditor 起動: $today ===" -ForegroundColor Magenta
Write-Host "  レポート出力先: $logFile" -ForegroundColor Gray
Write-Host ""

# Claude CLI でauditorを実行
# --print: 非インタラクティブモードで出力を標準出力へ
claude --print "keiba-auditorを実行してください。
今日の日付は $today です。
レポートは reports\audit_$today.md に保存してください。
完了したらサマリーだけ表示してください。"

Write-Host ""
Write-Host "=== 監査完了 ===" -ForegroundColor Magenta
if (Test-Path $logFile) {
    Write-Host "  詳細: $logFile" -ForegroundColor Gray
}
