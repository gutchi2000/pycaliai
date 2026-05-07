---
title: PyCaLiAI
emoji: 🏇
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: AI 競馬予想 (NiceGUI 版)
---

# 🏇 PyCaLiAI

LightGBM v5 LambdaRank モデルで JRA 中央競馬の印付け (◎〇▲△△) と
PL 確率を出し、Anthropic Cowork で馬券構築する個人運用システム。
NiceGUI 版は **表示専用** (推論済データを読んで可視化)。

## NiceGUI 版でできること

- 📋 出走表 (印 / 勝率 / 単勝EV / vs市場)
- 🔍 全頭分析 (散布図 + レーダーチャート)
- 🎫 Cowork 買い目 (race ごとに自動表示、予算 + 理由)
- 📊 過去同コース成績 + 枠バイアス (master_v2 がある場合のみ)

## ローカルで動かす

```powershell
cd E:\PyCaLiAI
.\venv311\Scripts\Activate.ps1
pip install -r requirements-nicegui.txt
python nicegui_app.py
```

ブラウザで `http://localhost:8080` を開く。

## HuggingFace Spaces デプロイ

このリポジトリは HF Spaces (Docker SDK) でそのままデプロイ可能:

1. https://huggingface.co/new-space で新規 Space 作成
2. SDK: **Docker**, リポジトリ: GitHub の URL を指定
3. ビルド完了後 `https://huggingface.co/spaces/USERNAME/pycaliai` でアクセス

または既存の HF Space に直接 push:

```bash
git remote add hf https://huggingface.co/spaces/USERNAME/pycaliai
git push hf master
```

## データソース

NiceGUI 版が読むファイル:

- `data/weekly/{YYYYMMDD}.csv` — 週次出走表
- `reports/cowork_input/{YYYYMMDD}_bundle.json` — 印 + 確率 (weekly_cowork.ps1 で生成)
- `reports/cowork_output/{YYYYMMDD}_bets.json` — Cowork からの買い目返答
- `reports/cowork_bets/{YYYYMMDD}/{race_id}.json` — Streamlit で保存された個別買い目
- `data/master_v2_*.csv` — 過去成績用 (約 390MB、Cloud には未配置、ローカルのみ)

## 既存 Streamlit 版との関係

このリポジトリには Streamlit 版 `app.py` も含まれる (Streamlit Cloud 用)。
NiceGUI 版 `nicegui_app.py` は Streamlit と並行運用、お好みで選択。

詳細は `docs/operation_roadmap.md` 参照。
