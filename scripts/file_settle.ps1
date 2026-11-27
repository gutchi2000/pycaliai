# file_settle.ps1 --- TARGET のエクスポートが書き終わるまで待つ (weekly_post.ps1 が dot-source)
#
# 2026-11-27: TARGET が data\kekka\wide_kekka.csv を書いている最中に Phase C が走り、
#   書きかけ (3,066 行 -> 295 行) のファイルが commit され、直後の pull の reset --hard が
#   開かれたままのファイルを置き換えられずに「リベース中」で止まった (11/21〜23 の 3 回)。
#   Excel ではなく TARGET が 1 分ほどかけて少しずつ書いていた。
#
# Wait-FileSettled: 書き込み用に排他で開けて、間隔をおいて 2 回見てもサイズと更新時刻が
#   変わらず、更新から MinAgeSec 秒以上経っていれば「書き終わった」とみなす。
#   TimeoutSec 待っても落ち着かなければ $false。ファイルが無ければ $true (有無の判断は呼び出し側)。
# Test-NotShrunk: git の HEAD の版より行数が減っていれば $false (書きかけ・別期間での上書き)。

function Test-FileOpenExclusive([string]$Path) {
    try {
        $fs = [System.IO.File]::Open($Path, 'Open', 'ReadWrite', 'None')
        $fs.Close()
        return $true
    } catch {
        return $false
    }
}

function Wait-FileSettled {
    param(
        [Parameter(Mandatory)][string]$Path,
        [int]$TimeoutSec = 600,
        [int]$IntervalSec = 5,
        [int]$MinAgeSec = 10
    )
    if (-not (Test-Path -LiteralPath $Path)) { return $true }
    $full = (Resolve-Path -LiteralPath $Path).Path
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    $prev = $null
    $announced = $false
    while ($true) {
        $item = Get-Item -LiteralPath $full
        $sig = "$($item.Length)|$($item.LastWriteTimeUtc.Ticks)"
        $age = ((Get-Date).ToUniversalTime() - $item.LastWriteTimeUtc).TotalSeconds
        $open = Test-FileOpenExclusive $full
        # 1 分以上触られていなければ待たない (普段はここで即 OK)
        if ($open -and ($age -ge [Math]::Max(60, $MinAgeSec) -or ($age -ge $MinAgeSec -and $sig -eq $prev))) {
            if ($announced) {
                Write-Host "      $Path : 書き込みが終わった ($($item.Length) bytes)" -ForegroundColor Green
            }
            return $true
        }
        if ((Get-Date) -ge $deadline) { return $false }
        if (-not $announced -and -not ($open -and $age -ge $MinAgeSec)) {
            $why = if (-not $open) { '別のプロセスが開いている' } else { "更新 $([int]$age) 秒前" }
            Write-Host "      $Path : 書き込み中の様子 ($why)。TARGET のエクスポートが終わるのを待ちます (最大 $TimeoutSec 秒) ..." -ForegroundColor Yellow
            $announced = $true
        }
        $prev = $sig
        Start-Sleep -Seconds $IntervalSec
    }
}

function Test-NotShrunk {
    param([Parameter(Mandatory)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return $true }
    $gitPath = $Path.Replace('\', '/')
    git cat-file -e "HEAD:$gitPath" 2>$null
    if ($LASTEXITCODE -ne 0) { return $true }   # まだ commit されていないファイル
    $headLines = @(git show "HEAD:$gitPath").Count
    $nowLines = @(Get-Content -LiteralPath $Path).Count
    if ($nowLines -lt $headLines) {
        Write-Host "      $Path : 行数が HEAD の $headLines 行から $nowLines 行に減っている" -ForegroundColor Red
        return $false
    }
    return $true
}

# kekka / wide_kekka をまとめて確認。問題があれば $false (呼び出し側で exit 1)。
function Assert-KekkaInputsSettled {
    param([string[]]$Paths, [string]$Stage)
    $timeout = 600
    if ($env:WEEKLY_POST_SETTLE_TIMEOUT) { $timeout = [int]$env:WEEKLY_POST_SETTLE_TIMEOUT }
    foreach ($p in $Paths) {
        if (-not (Wait-FileSettled -Path $p -TimeoutSec $timeout)) {
            Write-Error ("[$Stage] $p がまだ書き込み中 (または他のプロセスが開いたまま) で $timeout 秒待っても落ち着かない。" +
                " TARGET のエクスポートが終わってから再実行すること。commit/push 前に止めた。")
            return $false
        }
    }
    $wide = $Paths | Where-Object { $_ -match 'wide_kekka\.csv$' }
    foreach ($p in $wide) {
        if (-not (Test-NotShrunk -Path $p)) {
            Write-Error ("[$Stage] $p が HEAD の版より短い。書きかけか、短い期間で書き出したファイルで上書きした可能性。" +
                " TARGET から全期間で書き出し直してから再実行すること (意図した置き換えなら先に手で commit する)。commit/push 前に止めた。")
            return $false
        }
    }
    return $true
}
