---
title: PyCaLiAI UMAMI
emoji: 🏇
colorFrom: indigo
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
---

# PyCaLiAI UMAMI

JRA 中央競馬の AI 予想 — 静的サイト版。`site/`（index.html / css / js / data）を
極小の Docker (python http.server) で配信する。Docker SDK のため URL は
`https://gutchi15300-pycaliai-umami.hf.space`（`.static` が付かないクリーン形）。

デプロイは `sync-hf-umami.ps1`（リポジトリ本体側）。NiceGUI Space / 週次パイプラインとは独立。
