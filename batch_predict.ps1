Set-Location 'E:\PyCaLiAI'
$files = Get-ChildItem 'data\weekly' -Filter '2026*.csv' | Sort-Object Name
$total = $files.Count
$i = 0
foreach ($f in $files) {
    $i++
    Write-Host "[$i/$total] $($f.Name)..." -NoNewline
    python predict_weekly.py --csv "data\weekly\$($f.Name)" 2>&1 | Out-Null
    Write-Host " done"
}
Write-Host "Finished $total files"
