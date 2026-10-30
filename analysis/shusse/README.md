# 出世予測 AI（analysis/shusse）

馬券と無関係の読み物用 AI。デビュー k 走（k = 1 / 2 / 3 / 5）を終えた時点で、
「1 年以内に走り続けるか（離脱）」「走った場合の 1 年後の最高到達クラス（未勝利圏〜重賞で掲示板）」
「2 年以内に重賞で掲示板に載るか」を予測する。**市場オッズ・人気は一切使わない。馬券へは転用しない。**

設計と結果: `docs/research/SHUSSE_AI_DESIGN_20261030.md`（v1）→ `docs/research/SHUSSE_AI_V2_RESULTS_20261031.md`（v2、現行）。

## 再実行（root から、所要 約 6 分）
```powershell
venv311\Scripts\python.exe -m analysis.shusse.build_career_v2   # 外部 kekka の走破タイム等を結合 → out_v2/starts_v2, horses_v2 (19s)
venv311\Scripts\python.exe -m analysis.shusse.train_v2          # k=1/2/3/5 × ヘッド A/B/D 学習・評価・アブレーション (4min)
venv311\Scripts\python.exe -m analysis.shusse.extend_2026       # data/kekka + data/weekly の 2026 分を連結 (5s)
venv311\Scripts\python.exe -m analysis.shusse.predict_live      # 2023〜26 デビュー現役馬へ適用 + 前向き検証 (45s)。最後に site_export も走る
venv311\Scripts\python.exe -m analysis.shusse.site_export       # （単体で回すとき）公開サイト用 JSON out_v2/live/site/{index,YYYY}.json (1s)
venv311\Scripts\python.exe -m analysis.shusse.narrate --demo    # test 2022 の文章カード 45 枚
venv311\Scripts\python.exe -m analysis.shusse.report_v2         # 自己完結 HTML レポート out_v2/shusse_report.html
venv311\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_shusse_asof.py tests/test_shusse_narrate.py tests/test_shusse_live.py tests/test_shusse_site.py
```
週次更新は `extend_2026` → `predict_live`（内部で `site_export`）→ `report_v2` の 3 本だけでよい（学習は年 1 回）。
**週次の自動化（2026-10-31）**: `weekly_nicegui.ps1 -Post`（Phase C）の `[1e/3]` が `extend_2026` → `predict_live` を回し、
続く `sync-hf-umami.ps1` 内の `build_site.py` が `out_v2/live/site/*.json` を `site/data/shusse/` にコピーして公開する。
失敗しても Warn のみ（印・買い目・成績には影響せず、サイトは前回の出世予測のまま）。

## 公開サイト（2026-10-31 配線）
- ページ: `site/shusse.html` + `site/js/shusse.js`（分析カードと同じ独立ページ。トップのナビ「出世予測」と表紙の 05 節からリンク）
- データ: `site/data/shusse/index.json`（メタ・件数・検証数値）+ `site/data/shusse/{debut_year}.json`（その年デビューの未確定馬の予測行 + 上位 30 頭の文章カード）。
  年ごとに分けるのは 1 ファイル 4MB 超を避けるため（タブ切替で遅延読込、最大 1.9MB）。
- 表示は「未確定（基準日 + 365 日 > asof）かつ各馬の最新基準点」の馬だけ。確定した馬は前向き検証（`live_validation.json`）にだけ使う。
- 根拠は SHAP 上位 3 を `narrate.direction_phrase` で日本語句にして持たせる（ラベル＋数値は載せない）。市場オッズ・人気はどこにも出ない。
- `sync-hf-umami.ps1` の allowlist に `^site/data/shusse/[^/]+\.json$` を追加済み（生成データとして自動 commit 対象）。

## ファイル
| ファイル | 役割 |
|---|---|
| `build_career_v2.py` | master_v2 + `E:\競馬過去走データ\raw_data\kekka_2010_2025_fix_raceid_v2.csv`（読み取りのみ）を (日付,場所,R,血統登録番号) で結合。着差・上り3F・馬体重・生年月日・騎手 as-of 率 |
| `features_v2.py` | 目的変数（離脱 / 到達クラス / 24m 掲示板）、as-of ターゲットエンコーディング、`build_features(starts, horses, k, ...)`（学習と推論で共用） |
| `train_v2.py` | LightGBM 3 ヘッド × 4 基準点、valid で選んだパラメータを固定して test 2022 を評価。v1 定義比較・群別アブレーション |
| `extend_2026.py` | 2026 の週次 CSV と結果 CSV から starts_v2 と同列の行を作る（血統登録番号は `data/_horse_history.parquet` で解決） |
| `predict_live.py` | 現役馬の最新基準点で予測。確定した馬で前向き検証（`live/live_validation.json`） |
| `narrate.py` | テンプレートで日本語プロフィール文（LLM 不使用）。調教師名は `data/serve_code_maps.json` から |
| `report_v2.py` | 単一 HTML レポート（CSS インライン、外部参照なし） |
| `site_export.py` | 公開サイト用の軽量 JSON（`out_v2/live/site/`）。`predict_live` の最後に自動実行、`build_site.py` が `site/data/shusse/` へ |
| `build_career.py` / `train_baseline.py` | v1（記録用、`out/`） |

出力は `out_v2/`（大きい parquet・モデル・`live/site/` は .gitignore、再生成可能）。`out_v2/live/shusse_live.json` は `report_v2` 用の完全版、サイトは `live/site/` を読む。

## 落とし穴
- 外部 kekka の `race_id` は 4.4% 壊れている。結合は (日付, 場所, R, 血統登録番号)。
- 父タイプ名は 2023 以降ほぼ欠損。欠損のまま流す（学習時も欠損あり、寄与は小）。
- 各馬の最新走の着差・上り3F は次走の週次が出るまで NaN。
- 離脱のうち好走馬の故障は予測できない（外形情報に無い）。
- as-of 表は同 entity 同日の複数頭を 1 行に潰す（潰さないと途中の累積行を拾う）。
