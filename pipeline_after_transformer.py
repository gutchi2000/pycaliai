"""
pipeline_after_transformer.py
transformer_pl_v2.pkl が生成/更新されるまで待機し、
完了後に stacking.py → calibrate.py を順番に実行する。
"""
import os
import sys
import time
import subprocess
from pathlib import Path

BASE_DIR = Path(r"E:\PyCaLiAI")
MODEL_PATH = BASE_DIR / "models" / "transformer_pl_v2.pkl"
LOG_PATH = BASE_DIR / "reports" / "pipeline_after_transformer.log"

os.chdir(BASE_DIR)


def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def wait_for_transformer(poll_sec: int = 60, timeout_sec: int = 36000):
    """transformer_pl_v2.pkl が生成されるまで待つ。"""
    log(f"Transformer完了待機中... ({MODEL_PATH})")
    start_mtime = MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else None
    deadline = time.time() + timeout_sec

    while time.time() < deadline:
        if MODEL_PATH.exists():
            cur_mtime = MODEL_PATH.stat().st_mtime
            if start_mtime is None or cur_mtime != start_mtime:
                log(f"transformer_pl_v2.pkl 更新検出（mtime={cur_mtime}）")
                return True
        time.sleep(poll_sec)
        log(f"  ... 待機中 (経過: {(time.time() - (deadline - timeout_sec)) / 60:.0f}分)")

    log("タイムアウト: transformer が完了しませんでした")
    return False


def run_script(script: str) -> bool:
    log(f"\n{'='*60}")
    log(f"実行: {script}")
    log(f"{'='*60}")
    result = subprocess.run(
        [sys.executable, script],
        cwd=str(BASE_DIR),
        capture_output=False,
    )
    ok = result.returncode == 0
    log(f"{script} 終了 (returncode={result.returncode}, {'OK' if ok else 'WARNING'})")
    return ok


if __name__ == "__main__":
    log("=== pipeline_after_transformer.py 開始 ===")

    if wait_for_transformer():
        log("Transformer完了。スタッキング開始...")
        run_script("stacking.py")
        log("キャリブレーション開始...")
        run_script("calibrate.py")
        log("=== パイプライン完了 ===")
    else:
        log("=== Transformer未完了のためパイプライン中断 ===")
        sys.exit(1)
