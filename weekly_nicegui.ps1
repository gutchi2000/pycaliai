##############################################################
# weekly_nicegui.ps1  --- weekly workflow for NiceGUI on HF Spaces
#
# Purpose:
#   One script that handles three phases:
#     [A] PRE-RACE  (default)  : hosei + bundle.json + push
#                                 + sync-hf.ps1 (NiceGUI) + sync-hf-umami.ps1 (static 本番)
#     [B] BETS-ONLY (-BetsOnly): push reports/cowork_output/ only
#                                 (after Cowork returns the bets JSON)
#                                 + sync-hf.ps1 + sync-hf-umami.ps1
#     [C] POST-RACE (-Post)    : run weekly_post.ps1 + sync-hf.ps1 + sync-hf-umami.ps1
#
# Typical week:
#   TARGET から出したら data\_inbox\ に全部放り込む (出走表=S / 過去5走=K を先頭に)
#     -> Phase A/C 冒頭の Step 0 で place_weekly.py が自動で正しい場所へ振り分ける。
#        手動で place_weekly.py を先に走らせる必要はもう無い。 (-SkipIntake で無効化)
#   Sat morning  (after TARGET export):
#     .\weekly_nicegui.ps1                    # phase A: intake -> bundle to HF
#   Sat noon  (after Cowork returns bets):
#     # save Cowork's JSON to reports\cowork_output\{date}_bets.json first
#     .\weekly_nicegui.ps1 -BetsOnly          # phase B: bets to HF
#   Sun night (after TARGET kekka export):
#     .\weekly_nicegui.ps1 -Post              # phase C: intake -> results + HF
#
# Other usage:
#   .\weekly_nicegui.ps1 20260502             # specify date
#   .\weekly_nicegui.ps1 20260502 -SkipHF     # local + GitHub only
#   .\weekly_nicegui.ps1 20260502 -SkipPredict # skip predict_weekly.py
#   .\weekly_nicegui.ps1 20260502 -SkipGit    # local generation only
#   .\weekly_nicegui.ps1 -SkipIntake          # data\_inbox\ 自動振り分けをしない
#
# Phase A steps (default):
#   0. place_weekly.py          -> data\_inbox\ の CSV を weekly/kako5/kekka/training/bias へ振り分け
#   1. make_weekly_hosei.py     -> data/hosei/H_{date}.csv
#   2. predict_weekly.py        -> reports/pred_{date}.csv  (Streamlit)
#   3. export_weekly_marks.py   -> reports/cowork_input/{date}_bundle.json
#   4. git add + commit + push origin master  (Streamlit Cloud)
#   5. sync-hf.ps1                            (HuggingFace Spaces)
##############################################################
param(
    [string]$Date = "",
    [string]$Model = "v6",
    [switch]$SkipHF,
    [switch]$SkipPredict,   # 旧互換 (現在はデフォルトで skip)
    [switch]$WithPredict,   # 旧8モデルアンサンブル (Streamlit 用) を回したい週だけ opt-in
    [switch]$SkipGit,
    [switch]$BetsOnly,
    [switch]$Post,
    [switch]$SkipIntake,  # data\_inbox\ の自動振り分け (place_weekly.py) をスキップ
    [switch]$Force   # 見送りガードが実行できなくても続行する明示バイパス (fail-closed の解除)
)

$ErrorActionPreference = "Continue"
Set-Location 'E:\PyCaLiAI'

function Step($msg) {
    Write-Host ""
    Write-Host "==> $msg" -ForegroundColor Cyan
}
function OK($msg) {
    Write-Host "    OK: $msg" -ForegroundColor Green
}
function Warn($msg) {
    Write-Host "    WARN: $msg" -ForegroundColor Yellow
}
function Fail($msg) {
    Write-Host "    FAIL: $msg" -ForegroundColor Red
    exit 1
}

# =================================================================
# Step 0: intake auto-sort  (data\_inbox\ -> weekly / kako5 / kekka / training / bias)
#   place_weekly.py を週次フローに前置。data\_inbox\ に放り込んだ TARGET エクスポートを
#   ファイル名(S/K/H-/W-/OD) と中身(15列=結果 / 174列=払戻→実現バイアス自動生成) で
#   自動振り分けする。ここで data\weekly\{date}.csv 等が置かれるので、この後の
#   「日付自動検出」が効く。BetsOnly(Phase B) は _inbox を使わないため対象外。
#   -SkipIntake で明示スキップ。CSV が無ければ何もしない (no-op)。
# =================================================================
if ((-not $BetsOnly) -and (-not $SkipIntake)) {
    $inboxCsv = @()
    if (Test-Path 'data\_inbox') {
        $inboxCsv = @(Get-ChildItem 'data\_inbox' -Filter '*.csv' -ErrorAction SilentlyContinue)
    }
    if ($inboxCsv.Count -gt 0) {
        Step "[0] place_weekly.py (data\_inbox\ の $($inboxCsv.Count) 件を自動振り分け)"
        $prevUtf8 = $env:PYTHONUTF8
        $env:PYTHONUTF8 = '1'
        python place_weekly.py
        if ($LASTEXITCODE -ne 0) {
            Warn "place_weekly.py が非ゼロ終了 (未振り分けは data\_inbox\ に残置)。続行します。"
        } else {
            OK "intake 完了 (振り分け先は上記ログ参照)"
        }
        if ($null -eq $prevUtf8) { Remove-Item Env:\PYTHONUTF8 -ErrorAction SilentlyContinue }
        else { $env:PYTHONUTF8 = $prevUtf8 }
    } else {
        Write-Host "==> [0] intake: data\_inbox\ に CSV なし -> 振り分けスキップ" -ForegroundColor DarkGray
    }
}

# -- Determine date (mode-aware) --
# BetsOnly mode -> auto-detect from reports/cowork_output/YYYYMMDD_bets.json
# Other modes   -> auto-detect from data/weekly/YYYYMMDD.csv
# Both insist on exact 8-digit basename so test.csv etc are rejected.
$re8Digit = '^[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]$'
$re8DigitBets = '^[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_bets$'

if (($Date -eq "") -and $BetsOnly) {
    $allBets = Get-ChildItem 'reports\cowork_output' -Filter '*_bets.json' -ErrorAction SilentlyContinue
    $betsFile = $allBets | Where-Object { $_.BaseName -match $re8DigitBets } | Sort-Object Name -Descending | Select-Object -First 1
    if ($betsFile) {
        $Date = $betsFile.BaseName -replace '_bets$',''
        Write-Host "Auto-detect from cowork_output: $Date" -ForegroundColor Yellow
    }
}

if (($Date -eq "") -and (-not $BetsOnly)) {
    $allWeekly = Get-ChildItem 'data\weekly' -Filter '*.csv' -ErrorAction SilentlyContinue
    $latestCsv = $allWeekly | Where-Object { $_.BaseName -match $re8Digit } | Sort-Object Name -Descending | Select-Object -First 1
    if ($latestCsv) {
        $Date = $latestCsv.BaseName
        Write-Host "Auto-detect (latest weekly): $Date" -ForegroundColor Yellow
    }
}

if ($Date -eq "") {
    if ($BetsOnly) {
        Fail "No date given and no YYYYMMDD_bets.json in reports/cowork_output/. Save Cowork response or pass date."
    } else {
        Fail "No date given and no 8-digit YYYYMMDD.csv in data/weekly/."
    }
}

$csvPath = "data\weekly\$Date.csv"

# =================================================================
# Phase B: -BetsOnly  --- push reports/cowork_output/ only
# =================================================================
if ($BetsOnly) {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Magenta
    Write-Host "  weekly_nicegui [BetsOnly] : $Date" -ForegroundColor Magenta
    Write-Host "============================================================" -ForegroundColor Magenta

    # Verify cowork_output has the bets JSON for this date
    $betsPaths = @(
        "reports\cowork_output\${Date}_bets.json",
        "reports\cowork_output\${Date}.json"
    )
    $betsFound = $betsPaths | Where-Object { Test-Path $_ }
    if (-not $betsFound) {
        # also accept any file in cowork_output that contains the date
        $betsFound = Get-ChildItem 'reports\cowork_output' -ErrorAction SilentlyContinue |
                     Where-Object { $_.Name -match $Date }
    }
    if (-not $betsFound) {
        Fail ("No bets JSON found in reports\cowork_output\ for $Date. " +
              "Save Cowork's response as reports\cowork_output\${Date}_bets.json first.")
    }
    Write-Host ""
    Write-Host "Bets file(s):" -ForegroundColor Cyan
    foreach ($p in $betsFound) { Write-Host "    $p" -ForegroundColor Gray }

    # -- Guard: 見送り条件 (絶対禁則 2 の 4 条件) をコードで強制 --
    #    Cowork が見送りすべき race に買い目を付けても、ここで bets:[] に矯正する。
    #    (2026-05-30: Cowork が全 23 R に買い目を付けた事故への恒久対策)
    Step "[guard] validate_cowork_bets.py --apply (見送り条件チェック)"
    python validate_cowork_bets.py --date $Date --apply
    # exit 1 = ガードが実行できなかった (bundle/bets 不在・JSON 破損・例外)。
    # 無検証の bets を HF に push しないため fail-closed で停止する (2026-05-30 全23R 事故の再発防止)。
    if ($LASTEXITCODE -eq 1) {
        if ($Force) {
            Warn "見送りガードを実行できませんでした (bundle/bets 不在等) が、-Force 指定のため続行します。未検証の bets が HF に出ます。"
        } else {
            Fail "見送りガードを実行できませんでした (bundle/bets 不在・破損)。未検証の bets を push しないため停止します。原因確認後、どうしても push するなら -Force を付けて再実行してください。"
        }
    } else {
        OK "見送りガード通過 (exit $LASTEXITCODE)"
    }

    if (-not $SkipGit) {
        Step "[1/2] git add / commit / push origin master"
        git add reports/cowork_output 2>$null
        if (Test-Path "reports\cowork_bets\$Date") {
            git add ("reports/cowork_bets/$Date/*") 2>$null
        }
        $staged = git diff --cached --name-only 2>$null
        if (-not $staged) {
            Write-Host "    no changes to commit" -ForegroundColor Yellow
        } else {
            $y = $Date.Substring(0,4); $m = $Date.Substring(4,2); $d = $Date.Substring(6,2)
            git commit -m "cowork bets $y-$m-$d"
            git pull --rebase --autostash origin master
            git push origin HEAD:master
            if ($LASTEXITCODE -ne 0) { Warn "push failed; retry manually" } else { OK "pushed" }
        }
    }
    if (-not $SkipHF) {
        Step "[2/3] sync-hf.ps1 (NiceGUI Space)"
        & .\sync-hf.ps1
        Step "[3/3] sync-hf-umami.ps1 (static 本番サイト pycaliai-umami)"
        if (Test-Path 'sync-hf-umami.ps1') {
            & .\sync-hf-umami.ps1
            if ($LASTEXITCODE -ne 0) { Warn "sync-hf-umami.ps1 returned non-zero" }
        } else {
            Warn "sync-hf-umami.ps1 not found; skipping umami push"
        }
    }
    Write-Host ""
    Write-Host "Done. NiceGUI 'Cowork buy' tab will show the bets after rebuild." -ForegroundColor Green
    exit 0
}

# =================================================================
# Phase C: -Post  --- run weekly_post.ps1 + sync-hf.ps1
# =================================================================
if ($Post) {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Magenta
    Write-Host "  weekly_nicegui [Post] : $Date" -ForegroundColor Magenta
    Write-Host "============================================================" -ForegroundColor Magenta

    if (-not (Test-Path 'weekly_post.ps1')) {
        Fail "weekly_post.ps1 not found"
    }
    Step "[1/2] weekly_post.ps1 $Date"
    & .\weekly_post.ps1 $Date
    # weekly_post の失敗 (generate_results / git push 等) を握りつぶして HF 同期に
    # 進むと、結果更新漏れのまま緑の Done が出て翌週まで気付けない。失敗は伝搬させる。
    if ($LASTEXITCODE -ne 0) { Fail "weekly_post.ps1 failed (exit $LASTEXITCODE)。HF 同期を中止します。" }

    # generated_at が当日に更新されたか確認 (cowork_results 凍結事故 5/26 の再発検知)
    try {
        $cw = Get-Content data\cowork_results.json -Encoding UTF8 -TotalCount 3 | Out-String
        if ($cw -match '"generated_at":\s*"(\d{4}-\d{2}-\d{2})') {
            $genDate = $Matches[1]
            $today = Get-Date -Format "yyyy-MM-dd"
            if ($genDate -eq $today) { OK "cowork_results.json generated_at=$genDate (当日)" }
            else { Warn "cowork_results.json generated_at=$genDate が当日でない! 集計凍結の可能性。generate_results.py のログを確認" }
        }
    } catch { Warn "cowork_results.json の generated_at 確認に失敗 (非致命)" }

    if (-not $SkipHF) {
        Step "[2/3] sync-hf.ps1 (NiceGUI Space)"
        & .\sync-hf.ps1
        Step "[3/3] sync-hf-umami.ps1 (static 本番サイト pycaliai-umami)"
        if (Test-Path 'sync-hf-umami.ps1') {
            & .\sync-hf-umami.ps1
            if ($LASTEXITCODE -ne 0) { Warn "sync-hf-umami.ps1 returned non-zero" }
        } else {
            Warn "sync-hf-umami.ps1 not found; skipping umami push"
        }
    }
    Write-Host ""
    Write-Host "Done." -ForegroundColor Green
    exit 0
}

# =================================================================
# Phase A: PRE-RACE (default)
# =================================================================
if (-not (Test-Path $csvPath)) {
    Fail "$csvPath not found. Place TARGET CSV in data\weekly\ first."
}

# -- Verify model --
$modelPath = "models\unified_rank_$Model.pkl"
if (-not (Test-Path $modelPath)) {
    Fail "$modelPath not found. v5 trained model is required."
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  weekly_nicegui [Pre] : $Date  (model=$Model)" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green

# -- Step 1: hosei --
Step "[1/5] make_weekly_hosei.py"
python make_weekly_hosei.py --csv $csvPath
if ($LASTEXITCODE -ne 0) {
    Warn "hosei generation failed (continuing without)"
} else {
    OK "data\hosei\H_$Date.csv"
}

# -- Step 2: predict (Streamlit only, optional) --
# 2026-06-11: デフォルトを SKIP に反転 (audit P1-1)。旧8モデルアンサンブルは重く
# NiceGUI/Cowork ラインには不要。Streamlit 用に欲しい週だけ -WithPredict を付ける。
if ($WithPredict -and -not $SkipPredict) {
    Step "[2/5] predict_weekly.py (for Streamlit, -WithPredict)"
    python predict_weekly.py --csv $csvPath
    if ($LASTEXITCODE -ne 0) {
        Warn "predict_weekly failed (NiceGUI doesn't need this, continuing)"
    } else {
        OK "reports\pred_$Date.csv"
    }
} else {
    Step "[2/5] predict_weekly.py SKIPPED (default; opt-in は -WithPredict)"
}

# -- Step 3: bundle.json (NiceGUI required) --
Step "[3/5] export_weekly_marks.py (model=$Model) -> bundle.json"
python export_weekly_marks.py --csv $csvPath --model $Model
if ($LASTEXITCODE -ne 0) {
    Fail "export_weekly_marks.py failed. NiceGUI needs the bundle."
}
$bundlePath = "reports\cowork_input\${Date}_bundle.json"
if (Test-Path $bundlePath) {
    $size = [math]::Round((Get-Item $bundlePath).Length / 1024, 1)
    OK ("{0} ({1} KB)" -f $bundlePath, $size)
} else {
    Fail "bundle.json not created at $bundlePath"
}

# -- Step 3b: course_stats.json (NiceGUI コース分析タブ用、master_v2 から
#             集計、HF にも同期される ~600KB の事前計算ファイル) --
$masterCsv = Get-ChildItem 'data' -Filter 'master_v2_*.csv' -ErrorAction SilentlyContinue |
             Sort-Object Name -Descending | Select-Object -First 1
if ($masterCsv) {
    Step "[3b/5] build_course_stats.py -> data/course_stats.json"
    python build_course_stats.py --master $masterCsv.FullName
    if ($LASTEXITCODE -ne 0) {
        Warn "build_course_stats failed (continuing; existing JSON is used)"
    } else {
        $statsSize = [math]::Round((Get-Item 'data\course_stats.json').Length / 1024, 1)
        OK ("data\course_stats.json ({0} KB)" -f $statsSize)
    }
} else {
    Write-Host "    master_v2_*.csv not found -> skipping course_stats rebuild" -ForegroundColor Yellow
}

# -- Step 4: git add + commit + push origin --
if ($SkipGit) {
    Step "[4/5] git push SKIPPED (-SkipGit)"
} else {
    Step "[4/5] git add / commit / push origin master"

    git add $csvPath 2>$null

    $hoseiPath = "data\hosei\H_$Date.csv"
    if (Test-Path $hoseiPath) { git add $hoseiPath 2>$null }

    if (Test-Path $bundlePath) { git add $bundlePath 2>$null }

    if (Test-Path 'data\course_stats.json') {
        git add 'data\course_stats.json' 2>$null
    }

    $predPath = "reports\pred_$Date.csv"
    if (Test-Path $predPath) { git add -f $predPath 2>$null }

    $kako5Path = "data\kako5\$Date.csv"
    if (Test-Path $kako5Path) { git add $kako5Path 2>$null }

    # 直近の training CSV があればそれも (週次更新)
    # 注: -Filter は [HW] のような文字クラスを解釈しない (FS ワイルドカードのみ)。
    #     *$Date*.csv で拾い、H-/W- 前置を -match で絞る。
    Get-ChildItem 'data\training' -Filter "*$Date*.csv" -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^[HW]-' } |
        ForEach-Object { git add $_.FullName 2>$null }

    $staged = git diff --cached --name-only 2>$null
    if (-not $staged) {
        Write-Host "    no changes to commit" -ForegroundColor Yellow
    } else {
        $y = $Date.Substring(0,4); $m = $Date.Substring(4,2); $d = $Date.Substring(6,2)
        git commit -m "weekly $y-$m-$d (NiceGUI bundle, model=$Model)"
        if ($LASTEXITCODE -ne 0) {
            Warn "git commit failed"
        } else {
            git pull --rebase --autostash origin master
            git push origin HEAD:master
            if ($LASTEXITCODE -ne 0) {
                Warn "git push failed; retry: git push origin HEAD:master"
            } else {
                OK "pushed to origin/master"
            }
        }
    }
}

# -- Step 5: sync to HuggingFace Spaces --
if ($SkipHF) {
    Step "[5/5] sync-hf.ps1 / sync-hf-umami.ps1 SKIPPED (-SkipHF)"
} else {
    Step "[5/5] sync-hf.ps1 (master -> hf-spaces orphan -> NiceGUI Space)"
    if (-not (Test-Path 'sync-hf.ps1')) {
        Warn "sync-hf.ps1 not found; skipping HF push"
    } else {
        & .\sync-hf.ps1
        if ($LASTEXITCODE -ne 0) {
            Warn "sync-hf.ps1 returned non-zero"
        }
    }

    Step "[5b/5] sync-hf-umami.ps1 (build_site.py -> static 本番サイト pycaliai-umami)"
    if (-not (Test-Path 'sync-hf-umami.ps1')) {
        Warn "sync-hf-umami.ps1 not found; skipping umami push"
    } else {
        & .\sync-hf-umami.ps1
        if ($LASTEXITCODE -ne 0) {
            Warn "sync-hf-umami.ps1 returned non-zero"
        }
    }
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  Phase A (pre-race) Done." -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Confirm:" -ForegroundColor Yellow
Write-Host "  - HF Spaces     : https://gutchi15300-pycaliai.hf.space" -ForegroundColor Gray
Write-Host "  - Streamlit     : streamlit run app.py" -ForegroundColor Gray
Write-Host "  - NiceGUI local : python nicegui_app.py" -ForegroundColor Gray
Write-Host ""
Write-Host "Next phases:" -ForegroundColor Yellow
Write-Host "  Phase B (after Cowork returns bets):" -ForegroundColor Gray
Write-Host "    1. Save Cowork JSON as reports\cowork_output\${Date}_bets.json" -ForegroundColor Gray
Write-Host "    2. .\weekly_nicegui.ps1 -BetsOnly" -ForegroundColor Gray
Write-Host "  Phase C (after race results):" -ForegroundColor Gray
Write-Host "    1. Place TARGET kekka CSV in data\kekka\${Date}.csv" -ForegroundColor Gray
Write-Host "    2. .\weekly_nicegui.ps1 -Post" -ForegroundColor Gray
