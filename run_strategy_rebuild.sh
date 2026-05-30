#!/bin/bash
cd /e/PyCaLiAI

# Wait for both backtests
wait 980 983 2>/dev/null

echo "[check] backtest outputs..."
if [ ! -f reports/backtest_results_valid.csv ] || [ ! -f reports/backtest_results_2024.csv ]; then
    echo "ERROR: backtest CSV not found"
    exit 1
fi

echo "[3] build_strategy_stable.py..."
python build_strategy_stable.py >> logs/strategy_rebuild.log 2>&1
echo "strategy done. exit=$?"

echo "[4] generate_results.py..."
python generate_results.py >> logs/strategy_rebuild.log 2>&1
echo "generate_results done. exit=$?"

echo "=== ALL DONE ==="
