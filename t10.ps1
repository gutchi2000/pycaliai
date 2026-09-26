##############################################################
# t10.ps1 --- 当日 T-10 自動馬券ライン起動ラッパー
#
# 運用 (推奨・スリープ耐性あり): レース毎タスク方式
#   タスク "PyCaLiAI_T10" が土日 9:00 に -Schedule を起動 →
#   bundle 完成を待って **各レースの発走 T-10 に 1 個ずつ Windows タスクを登録**
#   (それぞれ WakeToRun)。以降は PC がスリープしても各レースで自動起床し、
#   そのレースだけ処理 (オッズ取得→compute_bets→validate→Discord 通知) して
#   また眠る。取りこぼしが構造的に起きない。
#
# 手動起動:
#   .\t10.ps1 -Schedule          # 今すぐレース毎タスクを登録 (bundle 必須)
#   .\t10.ps1 -Once 2026...11     # 1 レースだけ即処理 (テスト)
#   .\t10.ps1 20260614 -Loop      # 旧方式: 1 本ループで一日中回す (PC 起動必須)
#   .\t10.ps1 20260614 -Loop -Dry # 計算のみ
#
# 投票は人間 (IPAT)。HF への push はしない (ローカル NiceGUI が ui.timer 反映)。
##############################################################
param(
    [string]$Date = "",
    [string]$Once = "",        # rid16: 1 レースだけ処理
    [string]$Vote = "",        # rid16: 学生大会 API へ 1 レース投票 (締切=発走3分前)
    [double]$LeadMin = 10,
    [double]$VoteLeadMin = 4.5, # 大会投票: 発走4分30秒前に起動→3分前の締切までに送信
                               # (大会サーバが JRA-VAN からオッズを取るのも T-4:30)
    [double]$MaxAgeMin = 20,
    [switch]$Dry,
    [switch]$Schedule,         # レース毎タスクを登録
    [switch]$WithVote,         # 学生大会 投票タスク (PyCaLiAI_VOTE_*) も登録する
                               # 2026-09-26 大会終了につき既定 OFF (Discord 投票通知停止)
    [switch]$Routine,          # = -Schedule (9:00 タスクが渡す旧名・後方互換)
    [switch]$Loop              # 旧方式: 1 本ループで一日中回す
)
Set-Location 'E:\PyCaLiAI'
$env:PYTHONUTF8 = '1'

$py = 'venv311\Scripts\python.exe'
if (-not (Test-Path $py)) { $py = 'python' }
$pyFull = (Resolve-Path $py).Path

# -Routine は -Schedule の別名 (登録済み 9:00 タスクとの後方互換)
if ($Routine) { $Schedule = $true }

# ---------------------------------------------------------------
# -Vote: 学生大会 API へ 1 レース投票 (レース毎タスクの実体 + 手動テスト)
#   公式ルール: 投票は枠番発表後〜**発走 3 分前**まで。再投票は上書き。
#   T-4 に JV-Link 価格を取り、3 分前までに POST → 1 分後に check → logout。
# ---------------------------------------------------------------
if ($Vote -ne "") {
    # 大会終了 (2026-09-26)。残存タスクが起動しても投票・Discord 通知はしない。
    if (-not $WithVote) {
        Write-Host "[vote] 大会終了につき投票停止 (再開は -WithVote): $Vote"
        exit 0
    }
    if ($Date -eq "") { $Date = Get-Date -Format 'yyyyMMdd' }
    New-Item -ItemType Directory -Force logs | Out-Null
    try { Start-Transcript -Path ("logs\masters_vote_{0}.log" -f $Date) -Append | Out-Null } catch {}
    $argv = @('masters_vote.py', '--race', $Vote, '--date', $Date)
    if ($Dry) { $argv += '--dry' }
    $line = (& $pyFull 't10_runner.py' $Date '--list-schedule' |
             Where-Object { $_ -like "$Vote*" } | Select-Object -First 1)
    if ($line) {
        $post = ($line -split "`t")[1]
        if ($post) { $argv += @('--post', $post) }
    }
    & $pyFull @argv
    $code = $LASTEXITCODE
    # 大会側の最終判断 (投票 or 見送り/失敗) が台帳に書かれた直後にサイトを更新する。
    # T-20 速報 (t20_site_bets.py) はこのレースの最終状態を build_site.py 側で見て
    # 「もう推奨として出さない」に切り替わるが、次にどこかの publish が走るまで
    # 反映されない (このレースが当日最後の処理なら、それが来ないことがある) ので
    # ここで明示的に反映する。voted=False (見送り) でも exit code に関わらず実行
    # する (台帳自体は masters_vote.py が正常時に必ず書くため)。-Dry はサイトを
    # 変えないのでスキップ。
    if (-not $Dry) {
        & powershell -NoProfile -ExecutionPolicy Bypass -File .\sync-hf-umami.ps1 -Date $Date
    }
    try { Stop-Transcript | Out-Null } catch {}
    exit $code
}

# ---------------------------------------------------------------
# -Once: 1 レース処理 (レース毎タスクの実体 + 手動テスト)
# ---------------------------------------------------------------
if ($Once -ne "") {
    if ($Date -eq "") { $Date = Get-Date -Format 'yyyyMMdd' }
    New-Item -ItemType Directory -Force logs | Out-Null
    try { Start-Transcript -Path ("logs\t10_{0}.log" -f $Date) -Append | Out-Null } catch {}
    $argv = @('t10_runner.py', $Date, '--once', $Once,
              '--max-age-min', $MaxAgeMin)
    if ($Dry) { $argv += '--dry' }
    # 発走時刻まで Discord 予算返信を受け付ける (--poll-until)。発走時刻は schedule から。
    $line = (& $pyFull 't10_runner.py' $Date '--list-schedule' --lead-min $LeadMin |
             Where-Object { $_ -like "$Once*" } | Select-Object -First 1)
    if ($line) {
        $post = ($line -split "`t")[1]
        if ($post) { $argv += @('--poll-until', $post) }
    }
    & $pyFull @argv
    $code = $LASTEXITCODE
    # 当日変更情報 (取消/騎手変更/時刻/馬体重) をサイトへ反映 (レース毎 ≈30分間隔で更新)
    try {
        if ($Dry) { & powershell -NoProfile -File .\changes.ps1 -Date $Date -NoPush }
        else      { & powershell -NoProfile -File .\changes.ps1 -Date $Date }
    } catch { Write-Host "[changes] 例外: $_" }
    try { Stop-Transcript | Out-Null } catch {}
    exit $code
}

# ---------------------------------------------------------------
# -Schedule: レース毎タスクを登録 (bundle 待機つき)
# ---------------------------------------------------------------
if ($Schedule) {
    if ($Date -eq "") { $Date = Get-Date -Format 'yyyyMMdd' }
    New-Item -ItemType Directory -Force logs | Out-Null
    try { Start-Transcript -Path ("logs\t10_{0}.log" -f $Date) -Append | Out-Null } catch {}

    # bundle 完成待ち (Phase A)。15:00 まで 2 分間隔。
    $bundle = "reports\cowork_input\${Date}_bundle.json"
    $deadline = (Get-Date -Hour 15 -Minute 0 -Second 0)
    while (-not (Test-Path $bundle)) {
        if ((Get-Date) -ge $deadline) {
            Write-Host "[ERROR] 15:00 までに bundle 未生成 → 諦め (開催日でない/Phase A 未実行)"
            & $pyFull 't10_runner.py' '--test-notify' 2>$null | Out-Null
            try { Stop-Transcript | Out-Null } catch {}
            exit 1
        }
        Write-Host "[wait] $bundle 未生成 → 2 分後に再確認"
        Start-Sleep -Seconds 120
    }

    # 古いレース毎タスクを掃除してから今日のぶんを登録
    Get-ScheduledTask -TaskName 'PyCaLiAI_T10R_*' -ErrorAction SilentlyContinue |
        Unregister-ScheduledTask -Confirm:$false
    Get-ScheduledTask -TaskName 'PyCaLiAI_T15R_*' -ErrorAction SilentlyContinue |
        Unregister-ScheduledTask -Confirm:$false
    Get-ScheduledTask -TaskName 'PyCaLiAI_VOTE_*' -ErrorAction SilentlyContinue |
        Unregister-ScheduledTask -Confirm:$false

    $lines = & $pyFull 't10_runner.py' $Date '--list-schedule' --lead-min $LeadMin
    $n = 0
    foreach ($l in $lines) {
        $parts = $l -split "`t"
        if ($parts.Count -lt 2) { continue }
        $rid = $parts[0]; $post = $parts[1]
        $ph, $pm = $post -split ':'
        # 処理時刻 = 発走 - LeadMin
        $runAt = (Get-Date -Hour ([int]$ph) -Minute ([int]$pm) -Second 0).AddMinutes(-$LeadMin)
        if ($runAt -lt (Get-Date)) { continue }   # 既に過ぎたレースは登録しない
        $taskName = "PyCaLiAI_T10R_$rid"
        $act = New-ScheduledTaskAction -Execute 'powershell.exe' `
            -Argument ("-NoProfile -ExecutionPolicy Bypass -File E:\PyCaLiAI\t10.ps1 " +
                       "-Once $rid -Date $Date -LeadMin $LeadMin -MaxAgeMin $MaxAgeMin") `
            -WorkingDirectory 'E:\PyCaLiAI'
        $trg = New-ScheduledTaskTrigger -Once -At $runAt
        # 一度きり (発走直前のみ)。実行後 6h で自動削除。WakeToRun で PC を起こす。
        $trg.EndBoundary = $runAt.AddHours(2).ToString("yyyy-MM-ddTHH:mm:ss")
        $set = New-ScheduledTaskSettingsSet -StartWhenAvailable -WakeToRun `
            -ExecutionTimeLimit (New-TimeSpan -Minutes 30) `
            -MultipleInstances IgnoreNew `
            -DeleteExpiredTaskAfter (New-TimeSpan -Hours 6)
        Register-ScheduledTask -TaskName $taskName -Action $act -Trigger $trg `
            -Settings $set -Description "PyCaLiAI T-10 レース $rid ($post 発走)" -Force | Out-Null
        $n++
        Write-Host ("  登録 {0}  {1:HH:mm} 処理 → {2} 発走  ({3})" -f $taskName, $runAt, $post, $rid)

        # --- 学生大会 自動投票 (発走 VoteLeadMin 分前。締切=発走3分前) ---
        $voteAt = (Get-Date -Hour ([int]$ph) -Minute ([int]$pm) -Second 0).AddMinutes(-$VoteLeadMin)
        if ($WithVote -and $voteAt -ge (Get-Date)) {
            $voteName = "PyCaLiAI_VOTE_$rid"
            $vAct = New-ScheduledTaskAction -Execute 'powershell.exe' `
                -Argument ("-NoProfile -ExecutionPolicy Bypass -File E:\PyCaLiAI\t10.ps1 " +
                           "-Vote $rid -Date $Date -VoteLeadMin $VoteLeadMin -WithVote") `
                -WorkingDirectory 'E:\PyCaLiAI'
            $vTrg = New-ScheduledTaskTrigger -Once -At $voteAt
            $vTrg.EndBoundary = $voteAt.AddHours(1).ToString("yyyy-MM-ddTHH:mm:ss")
            $vSet = New-ScheduledTaskSettingsSet -StartWhenAvailable -WakeToRun `
                -ExecutionTimeLimit (New-TimeSpan -Minutes 10) `
                -MultipleInstances IgnoreNew `
                -DeleteExpiredTaskAfter (New-TimeSpan -Hours 6)
            Register-ScheduledTask -TaskName $voteName -Action $vAct -Trigger $vTrg `
                -Settings $vSet -Description "学生大会 投票 $rid ($post 発走)" -Force | Out-Null
            Write-Host ("  登録 {0}  {1:HH:mm:ss} 投票 → {2} 発走" -f $voteName, $voteAt, $post)
        }

        # T-15 補正印タスクは登録停止 (2026-07-31): 投稿ガイドライン「JV-Linkから取得した
        # データは投稿できません」対応。posting-support 照会で許可が出たら t15.ps1 登録を復活。
    }
    Write-Host "[schedule] $n レースのタスクを登録 (WakeToRun)"
    # 朝一の変更情報チェック (前日発表の取消等をサイトへ即反映 + 生録 dump でパーサ検証材料を残す)
    try { & powershell -NoProfile -File .\changes.ps1 -Date $Date -DumpRaw } catch { Write-Host "[changes] 例外: $_" }
    & $pyFull -c "import t10_runner as t; t.notify('PyCaLiAI T-10 ${Date}: $n レースを各 T-$LeadMin で自動起動登録しました。スリープしても各レースで起床します。')" 2>$null
    try { Stop-Transcript | Out-Null } catch {}
    exit 0
}

# ---------------------------------------------------------------
# -Loop: 旧方式 1 本ループ (PC 起動必須・budget返信を常時受付)
# ---------------------------------------------------------------
if ($Date -eq "") { $Date = Get-Date -Format 'yyyyMMdd' }
$argv = @('t10_runner.py', $Date, '--lead-min', $LeadMin, '--max-age-min', $MaxAgeMin)
if ($Dry) { $argv += '--dry' }
& $pyFull @argv
exit $LASTEXITCODE
