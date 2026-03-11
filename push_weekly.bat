@echo off
setlocal
:: =====================================================
:: 前半（レース前）: 週次CSV + 着度数CSV を push
:: 使い方: push_weekly.bat 20260308
:: =====================================================
set DATE=%1
if "%DATE%"=="" (
    echo 使い方: push_weekly.bat YYYYMMDD
    exit /b 1
)

cd /d E:\PyCaLiAI

echo === 週次 push 開始: %DATE% ===

:: --- weekly CSV チェック ---
if not exist "data\weekly\%DATE%.csv" (
    echo [ERROR] data\weekly\%DATE%.csv が見つかりません
    exit /b 1
)
echo [OK] data\weekly\%DATE%.csv

:: --- git add ---
git add "data\weekly\%DATE%.csv"

if exist "data\tyaku\%DATE%.csv" (
    git add "data\tyaku\%DATE%.csv"
    echo [OK] data\tyaku\%DATE%.csv
    set MSG=add weekly + tyaku csv %DATE%
) else (
    echo [SKIP] data\tyaku\%DATE%.csv なし
    set MSG=add weekly csv %DATE%
)

:: --- commit & push ---
git commit -m "%MSG%"
if errorlevel 1 (
    echo [INFO] コミットするものがありません
    exit /b 0
)

git push origin master
echo.
echo 完了^^! Streamlit Cloud に反映中...
