# t10_trio_shadow.ps1 — 三連複 shadow collector (BET-TRIO-T10-CONSERVATIVE-PORTFOLIO-2026-v2)
#
# 設計書 docs/plans/plan_trio_t10_conservative_portfolio_engine_v2.md §5.1/§8.1-1。
# 本番 t10.ps1 とは **別プロセス・別ロック・別出力**。本番の買い目・サイト・Discord・
# IPAT には一切干渉しない。出力は reports/trio_portfolio_shadow_v2/ のみ。
#
# 使い方（開催日の朝、本番 t10 と並走させて置いておく）:
#   .\t10_trio_shadow.ps1              … 最新 bundle の日付で起動
#   .\t10_trio_shadow.ps1 20260822     … 日付指定
param([string]$Date = "")

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not $Date) {
  $b = Get-ChildItem "reports\cowork_input\*_bundle.json" | Sort-Object Name -Descending | Select-Object -First 1
  if (-not $b) { Write-Error "bundle が無い。先に weekly_nicegui.ps1 (Phase A) を回す"; exit 1 }
  $Date = $b.Name.Substring(0, 8)
}
Write-Host "[trio-shadow] date=$Date  → reports/trio_portfolio_shadow_v2/" -ForegroundColor Cyan
& py -3.12-32 jvlink_trio_odds.py --watch $Date --lead-min 10
