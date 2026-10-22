#!/usr/bin/env bash
# 本番モデルを C1 に戻し、ハッシュ照合ガードを入れる（2026-10-22）
# 経緯: docs/research/PROD_MODEL_GENERATION_REGRESSION_20261021.md
set -euo pipefail
cd /e/PyCaLiAI

SRC=candidates/c1_migration_v2_final/models
want_model=db2ffe65ee27fb9014dcd78724851e77d292fcaf5d77aa33e40b83f3b186d0d5
want_cal=a08da05124c489c0d72c416bf17bf70c1c3829ffda2a381cc62e66007cad9221

check() { [ "$(sha256sum "$1" | cut -d' ' -f1)" = "$2" ] || { echo "NG: $1 のハッシュが想定と違う"; exit 1; }; }

if [ -d .git/rebase-merge ] || [ -d .git/rebase-apply ]; then echo "NG: rebase 途中"; exit 1; fi
if [ "$(git branch --show-current)" != "master" ]; then echo "NG: master にいない"; exit 1; fi

echo "[1/4] 置き直し元を確認"
check "$SRC/unified_rank_v6.pkl" "$want_model"
check "$SRC/pl_calibrators_v6.pkl" "$want_cal"

echo "[2/4] C1 を配置"
cp "$SRC/unified_rank_v6.pkl" models/unified_rank_v6.pkl
cp "$SRC/pl_calibrators_v6.pkl" models/pl_calibrators_v6.pkl
check models/unified_rank_v6.pkl "$want_model"
check models/pl_calibrators_v6.pkl "$want_cal"

echo "[3/4] 記録（commit）とガードの取り込み"
git add models/unified_rank_v6.pkl models/pl_calibrators_v6.pkl
git commit -q -m "prod: C1 モデル/基本較正器を再配置（9/20 autostash で C0 に退行していた）"
git merge -q --no-edit prod/artifact-pin-guard

echo "[4/4] テスト"
./venv311/Scripts/python.exe -m pytest -q -p no:cacheprovider tests/test_artifact_pins.py tests/test_production_policy.py

echo "完了: $(git log --oneline -1)"
