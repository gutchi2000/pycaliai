# ==========================================================
# PyCaLiAI - NiceGUI 版 (HuggingFace Spaces / Docker 共通)
# ==========================================================
# HF Spaces で使う場合:
#   - sdk: docker (README.md の YAML で指定)
#   - HF が自動で PORT=7860 を環境変数に注入
#   - app.py が PORT 環境変数を読んで bind
#
# ローカルで使う場合:
#   docker build -t pycaliai .
#   docker run -p 8080:8080 -e PORT=8080 pycaliai
# ==========================================================

FROM python:3.11-slim

# git は LFS データ取得 / 依存ビルドで必要なケースあり
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 依存をまず install (キャッシュ効率)
COPY requirements-nicegui.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-nicegui.txt

# アプリ本体 + 表示用データ
COPY nicegui_app.py .
COPY data data
COPY reports reports

# HF Spaces のデフォルトポート
EXPOSE 7860

# Docker 環境フラグ (host=0.0.0.0 にするため)
ENV DOCKER_CONTAINER=1

# HF Spaces は app_port=7860 を期待するため、明示的に PORT=7860 を設定
# (HF が自動注入しないケースに備える)
ENV PORT=7860

CMD ["python", "nicegui_app.py"]
