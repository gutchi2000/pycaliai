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

LightGBM **v6** LambdaRank モデル (`unified_rank_v6.pkl`) で JRA 中央競馬の
印付け (◎〇▲△△) と Plackett-Luce 確率を算出し、買い目は完全トップダウン
エンジン (`compute_bets.py`, `CB_ENGINE=topdown`) が生成、Cowork (Claude) は
narrative 論評を担当する個人運用システム。

## 表示レイヤー（3系統）

| 系統 | 状態 | URL / 起動 |
|---|---|---|
| **静的サイト** (HF Docker Space `pycaliai-umami`) | ✅ **本番** | https://pycaliai.com / https://gutchi15300-pycaliai-umami.hf.space |
| NiceGUI (この Space / `nicegui_app.py`) | 🗄️ 旧本番（併行更新中） | https://gutchi15300-pycaliai.hf.space |
| Streamlit (`app.py`) | 副系統 | Streamlit Cloud |

いずれも **表示専用**（推論済みデータを読んで可視化するだけ）。
本番サイトのソースは `site/`、データ生成は `build_site.py`、
デプロイは `sync-hf-umami.ps1`。

## 週次運用（`weekly_nicegui.ps1` 1本で完結）

```powershell
.\weekly_nicegui.ps1            # Phase A: 土曜朝 (出走表 → 印/bundle → HF 同期)
.\weekly_nicegui.ps1 -BetsOnly  # Phase B: Cowork 返答 (narrative) 反映後
.\weekly_nicegui.ps1 -Post      # Phase C: 日曜夜 (結果集計 → HF 同期)
```

- TARGET 出力 CSV は `data/_inbox/` に放り込むだけ（`place_weekly.py` が自動振り分け。詳細は `data/_inbox/README.txt`）
- 当日は T-10 自動ライン（タスクスケジューラ `PyCaLiAI_T10` → `t10.ps1`）が
  発走10分前に JV-Link オッズ取得 → `compute_bets.py` → 検証 → 買い目表示。投票は人間が IPAT で行う

## パイプライン概要

```
出走表 CSV
  → export_weekly_marks.py --model v6   (印 + PL 確率 + calibration)
  → reports/cowork_input/{date}_bundle.json
  → compute_bets.py (topdown: 全馬 p_win → λ補正 PL → 確率順候補 → 適応トリガミ床)
  → validate_cowork_bets.py (見送りガード強制, fail-closed)
  → build_site.py → site/data/*.json → sync-hf-umami.ps1 (本番反映)
```

## ローカルで動かす（NiceGUI 版）

```powershell
cd E:\PyCaLiAI
.\venv311\Scripts\Activate.ps1
pip install -r requirements-nicegui.txt
python nicegui_app.py
```

ブラウザで `http://localhost:8080` を開く。

## 主要データソース

- `data/weekly/{YYYYMMDD}.csv` — 週次出走表 (TARGET)
- `reports/cowork_input/{YYYYMMDD}_bundle.json` — 印 + 確率 (`export_weekly_marks.py` 生成)
- `reports/cowork_output/{YYYYMMDD}_bets.json` — 買い目 + narrative
- `data/cowork_results.json` / `data/live_results_2026.csv` — 実績集計
- `data/master_v2_*.csv` — 学習用マスター (約 515MB、ローカルのみ・HF 未配置)

## ドキュメント

- 全体像・引き継ぎ: `CLAUDE.md`（最重要）
- 週次フロー詳細: `WORKFLOW.md`
- 買い目エンジン仕様: `docs/compute_bets_spec.md`
- bundle スキーマ: `docs/marks_schema.md`
- 実験スクリプト群: `lab/README.md`（root から `python -m lab.<theme>.<name>` で実行）
