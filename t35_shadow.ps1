##############################################################
# t35_shadow.ps1 --- EXP05-F 前向き検証用 T-35 市場snapshot起動ラッパー
#
# t10.ps1 / t20_site.ps1 と同じ「レース毎タスク」方式。発走 T-35 (31-38分前ウィンドウを
# 狙う) に各レース 1 個ずつ Windows タスクを登録 (WakeToRun)。実行のたびに:
#   1) analysis.mcond.exp05_forward_shadow.market_snapshot --once <rid> でオッズ取得+検証
#   2) predict_and_store.store_prediction() で凍結モデル(M1/M3/M4)の予測をappend-only保存
#
# 本番 T-10 ライン (t10.ps1) / サイト T-20 (t20_site.ps1) / 大会実投票 (T-4) には
# 一切干渉しない。買い目・印・資金配分への接続もしない (研究専用)。
#
# 運用 (2026-09-19登録、2026-09-19毎日トリガーへ修正): マスタータスク "PyCaLiAI_EXP05FS_T35"
#   が毎日9:00に -Schedule を起動する。曜日を一切見ないため祝日開催・月曜開催・代替開催も
#   自動的に拾う (data/weekly/{date}.csv の有無だけで判定、t10_runner.load_post_times と
#   同じロジック)。当日 data/weekly/{date}.csv が無ければ (=開催なし) 即座に正常終了 (exit 0)。
#   レース毎タスクの登録はタスク名の存在チェックによる冪等処理: 既存かつ未来時刻のタスクは
#   触らない、過去時刻のまま残っている陳腐化タスクだけ削除してから (発走時刻を過ぎていれば)
#   再登録しない。
#
# 手動:
#   .\t35_shadow.ps1 -Schedule           # 今すぐレース毎タスクを登録 (当日開催が無ければ即終了)
#   .\t35_shadow.ps1 -Once 2026...11     # 1レースだけ即処理 (テスト)
#   .\t35_shadow.ps1 -Once 2026...11 -Dry
##############################################################
param(
    [string]$Date = "",
    [string]$Once = "",
    [double]$LeadMin = 35,
    [switch]$Dry,
    [switch]$Schedule
)
Set-Location 'E:\PyCaLiAI'
$env:PYTHONUTF8 = '1'

$py = 'venv311\Scripts\python.exe'
if (-not (Test-Path $py)) { $py = 'python' }
$pyFull = (Resolve-Path $py).Path
$mod = 'analysis.mcond.exp05_forward_shadow.market_snapshot'

if ($Once -ne "") {
    if ($Date -eq "") { $Date = Get-Date -Format 'yyyyMMdd' }
    New-Item -ItemType Directory -Force logs | Out-Null
    try { Start-Transcript -Path ("logs\t35_shadow_{0}.log" -f $Date) -Append | Out-Null } catch {}
    $argv = @('-m', $mod, '--date', $Date, '--once', $Once, '--lead-min', $LeadMin)
    if ($Dry) { $argv += '--dry' }
    & $pyFull @argv
    $code = $LASTEXITCODE
    try { Stop-Transcript | Out-Null } catch {}
    exit $code
}

if ($Schedule) {
    if ($Date -eq "") { $Date = Get-Date -Format 'yyyyMMdd' }
    New-Item -ItemType Directory -Force logs | Out-Null
    try { Start-Transcript -Path ("logs\t35_shadow_{0}.log" -f $Date) -Append | Out-Null } catch {}

    # ---- no-race日の早期正常終了 ----
    # data/weekly/{date}.csv は TARGET の出走表エクスポートがあった日にしか存在しない
    # (t10_runner.load_post_times と同じ判定源)。これが無い時点で「今日は開催が無い」と
    # 確定できるので、bundle 生成を待つ意味が無い (毎日トリガーだと待っても15:00まで
    # 空回りするだけになる)。曜日は一切見ないので祝日開催・月曜開催・代替開催も
    # 自動的に拾う。
    $weeklyCsv = "data\weekly\${Date}.csv"
    if (-not (Test-Path $weeklyCsv)) {
        Write-Host "[no-race] $weeklyCsv 無し → 本日は開催なしと判断し正常終了"
        try { Stop-Transcript | Out-Null } catch {}
        exit 0
    }

    $bundle = "reports\cowork_input\${Date}_bundle.json"
    $deadline = (Get-Date -Hour 15 -Minute 0 -Second 0)
    while (-not (Test-Path $bundle)) {
        if ((Get-Date) -ge $deadline) {
            Write-Host "[ERROR] 開催表はあるのに15:00までにbundle未生成 → 異常 (Phase A失敗の疑い)"
            try { Stop-Transcript | Out-Null } catch {}
            exit 1
        }
        Write-Host "[wait] $bundle 未生成 → 2 分後に再確認"
        Start-Sleep -Seconds 120
    }
    # 特徴量snapshotも事前に作っておく (市場と無関係、時点安全)
    & $pyFull -m 'analysis.mcond.exp05_forward_shadow.feature_snapshot' --date $Date

    # ---- 冪等なレース毎タスク登録 ----
    # 既存タスク一覧を1回だけ取得し、(a) 今回計算した対象と同名で未来時刻のものは
    # 触らない (重複登録防止)、(b) 過去時刻のまま残っている陳腐化タスクだけ個別に
    # 削除する (誤って再実行されないよう、かつ他日・他レースのタスクは残す)。
    $existing = @{}
    foreach ($t in (Get-ScheduledTask -TaskName 'PyCaLiAI_EXP05FS_T35R_*' -ErrorAction SilentlyContinue)) {
        $info = Get-ScheduledTaskInfo -TaskName $t.TaskName -ErrorAction SilentlyContinue
        $existing[$t.TaskName] = $info
    }

    $lines = & $pyFull -m $mod --date $Date --list-schedule --lead-min $LeadMin
    $n_new = 0; $n_skipped = 0; $n_stale_removed = 0
    $targetNames = @{}
    foreach ($l in $lines) {
        $parts = $l -split "`t"
        if ($parts.Count -lt 2) { continue }
        $rid = $parts[0]; $post = $parts[1]
        $ph, $pm = $post -split ':'
        $runAt = (Get-Date -Hour ([int]$ph) -Minute ([int]$pm) -Second 0).AddMinutes(-$LeadMin)
        $taskName = "PyCaLiAI_EXP05FS_T35R_$rid"
        $targetNames[$taskName] = $true

        if ($runAt -lt (Get-Date)) {
            # 発走枠を過ぎたレース: 既存タスクが残っていれば陳腐化なので削除するだけ (再登録しない)
            if ($existing.ContainsKey($taskName)) {
                Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
                $n_stale_removed++
                Write-Host ("  削除(陳腐化) {0}  (発走枠 {1:HH:mm} は既に経過)" -f $taskName, $runAt)
            }
            continue
        }
        if ($existing.ContainsKey($taskName)) {
            # 既に未来時刻で登録済み → 重複登録しない
            $n_skipped++
            Write-Host ("  既存(スキップ) {0}  {1:HH:mm} 処理 → {2} 発走" -f $taskName, $runAt, $post)
            continue
        }
        $act = New-ScheduledTaskAction -Execute 'powershell.exe' `
            -Argument ("-NoProfile -ExecutionPolicy Bypass -File E:\PyCaLiAI\t35_shadow.ps1 " +
                       "-Once $rid -Date $Date -LeadMin $LeadMin") `
            -WorkingDirectory 'E:\PyCaLiAI'
        $trg = New-ScheduledTaskTrigger -Once -At $runAt
        $trg.EndBoundary = $runAt.AddHours(2).ToString("yyyy-MM-ddTHH:mm:ss")
        $set = New-ScheduledTaskSettingsSet -StartWhenAvailable -WakeToRun `
            -ExecutionTimeLimit (New-TimeSpan -Minutes 30) `
            -MultipleInstances IgnoreNew `
            -DeleteExpiredTaskAfter (New-TimeSpan -Hours 6)
        Register-ScheduledTask -TaskName $taskName -Action $act -Trigger $trg `
            -Settings $set -Description "EXP05-F T-35 市場snapshot $rid ($post 発走)" -Force | Out-Null
        $n_new++
        Write-Host ("  登録 {0}  {1:HH:mm} 処理 → {2} 発走  ({3})" -f $taskName, $runAt, $post, $rid)
    }

    # 今回の対象に無い「別日の取り残し」等、$targetNames に無い陳腐化タスクも掃除する
    # (発走枠が過去のものだけ、将来枠は誤って触らない)
    foreach ($name in $existing.Keys) {
        if ($targetNames.ContainsKey($name)) { continue }
        $t = Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
        if ($null -eq $t) { continue }
        $trigStart = $null
        try { $trigStart = [datetime]$t.Triggers[0].StartBoundary } catch {}
        if ($null -ne $trigStart -and $trigStart -lt (Get-Date)) {
            Unregister-ScheduledTask -TaskName $name -Confirm:$false -ErrorAction SilentlyContinue
            $n_stale_removed++
            Write-Host ("  削除(陳腐化・対象外日) {0}" -f $name)
        }
    }

    Write-Host "[schedule] 新規登録 $n_new / 既存スキップ $n_skipped / 陳腐化削除 $n_stale_removed"
    try { Stop-Transcript | Out-Null } catch {}
    exit 0
}

Write-Host "使い方: .\t35_shadow.ps1 -Schedule  または  .\t35_shadow.ps1 -Once <rid16>"
exit 1
