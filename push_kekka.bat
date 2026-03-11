@echo off
setlocal
:: =====================================================
:: 後半（レース後）: kekka CSV → results.json 更新 → push
:: 使い方: push_kekka.bat 20260308
:: =====================================================
set DATE=%1
if "%DATE%"=="" (
    echo 使い方: push_kekka.bat YYYYMMDD
    exit /b 1
)

cd /d E:\PyCaLiAI

echo === kekka push 開始: %DATE% ===

:: --- kekka CSV チェック ---
if not exist "data\kekka\%DATE%.csv" (
    echo [ERROR] data\kekka\%DATE%.csv が見つかりません
    exit /b 1
)
echo [OK] data\kekka\%DATE%.csv

:: --- generate_results.py 実行 ---
echo [RUN] generate_results.py ...
venv311\Scripts\python.exe generate_results.py
if errorlevel 1 (
    echo [ERROR] generate_results.py が失敗しました
    exit /b 1
)

:: --- git add ---
git add "data\kekka\%DATE%.csv" data\results.json

:: --- commit & push ---
git commit -m "add kekka %DATE%"
if errorlevel 1 (
    echo [INFO] コミットするものがありません
    exit /b 0
)

git push origin master
echo.
echo 完了^^! 的中実績ページに反映中...
