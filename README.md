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

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-pytest-0A9EDC.svg)](tests)

PyCaLiAI は、JRA中央競馬を題材にした**時系列リーク耐性のあるランキング・確率校正・
意思決定パイプライン**です。LightGBM LambdaRank、Plackett–Luce確率、時系列分割、
forward shadow、training/serving parity監査、fail-closed検証を、継続運用できる一つの
システムとして公開しています。

単なる予想デモではなく、「予測時点で本当に利用可能だった情報だけで評価する」こと、
漏洩した実験を不採用として記録すること、オフライン指標から本番判断までの境界を
明示することを重視しています。

> [!IMPORTANT]
> MITライセンスの対象は、PyCaLiAI独自のソースコードとドキュメントです。
> JRA-VAN、TARGET frontier JV、JRA、netkeiba等に由来するデータの権利は含みません。
> 詳細は [NOTICE.md](NOTICE.md) と
> [オープンソース範囲](docs/OPEN_SOURCE_SCOPE.md) を参照してください。

## 主な特徴

- **Leak-aware evaluation** — as-of / OOFを前提とし、目的変数リークや時系列混入を監査
- **Calibrated ranking** — LambdaRankの順位スコアをPlackett–Luce確率と校正器へ接続
- **Forward-first validation** — retrospective、shadow、live-forwardの証拠を区別
- **Fail-closed operations** — ガード検証不能時は未検証の出力を公開しない
- **Research ledger** — 不採用モデルや反証結果も台帳へ残し、再採用事故を防止
- **Public presentation layer** — 生成済みの安全な出力を静的サイトで可視化

## 現在の状態

- 主開発ブランチ: `master`
- Python: 3.11
- 本番モデル系統: unified rank / calibrated Plackett–Luce stack
- 保守形態: primary maintainerによる継続運用
- 公開サイト: https://pycaliai.com

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

### 週次フロー外の手動コマンド

```powershell
# note 有料記事ドラフト生成 (会場パック / 重賞単体、JRA-VAN 準拠スクラブ付き)
python scripts/build_note_article.py 20260802
# → reports/note/{date}/{会場}.md + _compliance_report.txt

# 祝日(月)開催の T-10 手動起動 / テスト
.\t10.ps1
.\t10.ps1 20260614 -Dry
```

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

コアのテストは次で実行します。

```powershell
python -m pip install -r requirements.txt
python -m pytest
```

フルパイプラインには別途ライセンスされたデータやWindows固有の連携が必要です。
公開データを含む完全再現パッケージではありません。再現性の範囲と今後の分離方針は
[docs/OPEN_SOURCE_SCOPE.md](docs/OPEN_SOURCE_SCOPE.md) に記載しています。

## 主要データソース

- `data/weekly/{YYYYMMDD}.csv` — 週次出走表 (TARGET)
- `reports/cowork_input/{YYYYMMDD}_bundle.json` — 印 + 確率 (`export_weekly_marks.py` 生成)
- `reports/cowork_output/{YYYYMMDD}_bets.json` — 買い目 + narrative
- `data/cowork_results.json` / `data/live_results_2026.csv` — 実績集計
- `data/master_v2_*.csv` — 学習用マスター (約 515MB、ローカルのみ・HF 未配置)

## ドキュメント

- OSSの範囲・再現性: `docs/OPEN_SOURCE_SCOPE.md`
- コントリビューション: `CONTRIBUTING.md`
- セキュリティ報告: `SECURITY.md`
- 全体像・引き継ぎ: `AGENTS.md`
- 週次フロー詳細: `WORKFLOW.md`
- 買い目エンジン仕様: `docs/compute_bets_spec.md`
- bundle スキーマ: `docs/marks_schema.md`
- 実験スクリプト群: `lab/README.md`（root から `python -m lab.<theme>.<name>` で実行）

## Contributing

バグ報告、ドキュメント改善、合成fixture、リーク検査、校正・評価手法の改善を歓迎します。
第三者の生データや資格情報はissue・PRへ添付しないでください。詳しくは
[CONTRIBUTING.md](CONTRIBUTING.md) を参照してください。

## License

独自のソースコードとドキュメントは [MIT License](LICENSE) で公開します。
第三者データ・商標・サービスに関する除外事項は [NOTICE.md](NOTICE.md) に記載しています。
