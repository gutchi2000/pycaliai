# 旧系統退役 — 依存図と切り出し案（レビュー用・未実装）

作成 2026-10-04。本書は案であり、コードは一切変更していない。実装はレビュー後。

## 1. 結論

- 旧 8 モデルアンサンブルの本体 `predict_weekly.py`（2,094 行）は、本番が**出走表の読み込み部分だけ**を借りている。予測・買い目部分（736 行目以降）を本番は一度も呼ばない。
- 読み込み部分（122〜735 行）は予測・買い目部分に依存していないので、**中身を 1 行も書き換えずに別ファイルへ移すだけ**で切り離せる。出力を変えない抽出は可能と判断する。
- ただし読み込み部分は調教データの結合を `optuna_lgbm.py`（旧モデルの学習スクリプト）から借りており、ここも同じ方法で切り出す必要がある。切り出し対象は 2 か所。
- 退役そのもの（ファイルの移動・削除、Streamlit 停止、月次の自動再学習の停止）は切り出しの後の別段階とし、それぞれ承認を取る。

## 2. 依存図

```
本番（v6 / a95）                              旧系統
────────────────────────────                 ─────────────────────────────────
export_weekly_marks.py ──parse_csv──────┐
build_bet_plan.py ───────parse_csv──────┤    predict_weekly.py
  ↑ compute_bets.py / t10_runner.py     ├──▶  [A] 122-735  読み込み部   ← 本番が使うのはここだけ
build_horse_history.py ──RACE_COLS,─────┤         ├─ parse_kako5.py（共用・root 維持）
                         HORSE_COLS_*   │         ├─ parse_bunseki.py（共用・root 維持）
make_weekly_hosei.py（列定義を複製保持）  │         └─ optuna_lgbm.load_chukyo / merge_chukyo  ──▶ [C]
tests/test_serve_parse.py ──_load_tyaku─┘     [B] 39-121, 736-2094 予測・買い目部
                                                  ├─ ev_filter.py
analysis/ 13 本・logs/ 4 本（研究・診断）──parse_csv      ├─ 旧モデル pkl 10 種 / ensemble_weights.json
                                                  ├─ value_model_v2 / order_model_v1
                                                  └─ data/strategy_weights.json
                                              optuna_lgbm.py
                                                [C] 148-262 調教結合（load_chukyo, merge_chukyo）
                                                [D] それ以外 = 旧 LGBM の Optuna 学習（import 時に optuna / lightgbm を読む）

旧系統だけが使うもの
  app.py（Streamlit）──▶ predict_weekly [B], optuna_lgbm, strategy_weights.json
  precompute_buylist.py ──▶ predict_weekly [B]（呼び出し元なし）
  weekly_pre.ps1 ──▶ predict_weekly.py（weekly_nicegui.ps1 からは -WithPredict 指定時のみ）
  weekly_post.ps1:182 ──▶ retrain_value_model.py（月初の日曜に自動実行）──▶ train_value_model.py
  build_strategy_walkforward.py / build_strategy_stable.py / simulate_patterns.py ──▶ strategy_weights.json
  optuna_catboost.py / train_lgbm.py ──▶ optuna_lgbm
  nicegui_app.py + sync-hf.ps1（旧 NiceGUI Space。週次 3 フェーズすべてで今も同期中）
```

### 本番が predict_weekly から取っているシンボル（全数）

| 取り込み側 | シンボル | 区分 |
|---|---|---|
| `export_weekly_marks.py:58` | `parse_csv` | 本番（bundle 生成） |
| `build_bet_plan.py:65` | `parse_csv`（関数内 import） | 本番（T-10 経路から到達） |
| `build_horse_history.py:54` | `RACE_COLS`, `HORSE_COLS_33/46/48/49/99` | 本番（馬履歴の再生成） |
| `tests/test_serve_parse.py` | `_load_tyaku`, `TYAKU_HORSE_COLS*`, `TYAKU_DIR`（monkeypatch） | テスト |
| `analysis/` 13 本, `logs/` 4 本 | `parse_csv`（logs の 3 本は `predict_lgbm` 等も） | 研究・診断 |

`parse_csv` が内部で参照するモジュール変数は `BASE_DIR`, `COLUMN_MAP`, `RACE_COLS`, `HORSE_COLS_33/44/46/48/49/99`, `TYAKU_DIR`, `TYAKU_HORSE_SCHEMAS`, `HOSSEI_DIR`, `logger` のみ。[B] の関数・定数は参照していない（122〜735 行を機械走査して確認）。

### 現状の副作用

`import predict_weekly` は冒頭で `ev_filter` を読み、`parse_csv` 実行時に `optuna_lgbm` を読む。後者は import 時に `optuna`・`lightgbm`・`sklearn` を読み込む。つまり本番の bundle 生成は、使わない旧学習スクリプトの import に成功することを前提にしている。これらが壊れると bundle 生成が止まる（調教結合だけは try/except で握りつぶされ、**調教特徴が無言で全欠損になる**）。切り出しの実益はここにある。

## 3. 切り出し案

### 段階 1: 移すだけ（出力不変・本番の挙動を変えない）

1. 新設 `weekly_parse.py` — `predict_weekly.py` の [A]（列定義・`COLUMN_MAP`・`_load_tyaku`・`_load_kako5_warnings`・`_load_hosei`・`parse_csv`）を**そのまま移す**。編集は import 文の調整だけ。
2. 新設 `training_join.py` — `optuna_lgbm.py` の [C]（`load_chukyo`・`merge_chukyo` と、それが使うパス定数）を**そのまま移す**。
3. 元ファイルは移した名前を再 export する（`from weekly_parse import *` 相当を明示列挙）。旧系統・研究スクリプト 17 本は無修正で動き続ける。
4. 本番 3 本（`export_weekly_marks.py`, `build_bet_plan.py`, `build_horse_history.py`）と `tests/test_serve_parse.py` の import 先を新ファイルへ向ける。
5. `make_weekly_hosei.py` が複製している列定義は段階 1 では触らない（統合は挙動差の検証が別途要るため）。

`CLASS_NORMALIZE` と `EXCLUDE_*` は [B] 側に残す（`parse_csv` は使っていない）。

### 段階 2: 旧系統を週次フローから外す（要承認）

- `weekly_post.ps1:182` の月次 `retrain_value_model.py` 自動実行を止める（value_model は旧系統専用）。
- `weekly_nicegui.ps1` の `-WithPredict` 分岐と `weekly_pre.ps1` を外す。
- `sync-hf.ps1`（旧 NiceGUI Space）を週次から外すかは別判断。外すと旧 Space の更新が止まる。

### 段階 3: 退役（要承認・不可逆部分あり）

- `predict_weekly.py` の [B]、`app.py`、`precompute_buylist.py`、`ev_filter.py`、`betting.py`/`kelly.py`、strategy 系 3 本、旧学習スクリプトを `legacy/` へ移す。
- 旧モデル pkl 10 種と `strategy_weights.json` は `models/archive/` へ移すだけにして削除はしない。
- Streamlit Cloud の停止はユーザー操作。

## 4. 出力不変を確かめる手順（段階 1 の受け入れ条件）

実装前に基準値を採り、実装後に同じ手順で照合する。1 つでも不一致なら差し戻す。

| # | 確認 | 方法 | 合格条件 |
|---|---|---|---|
| 1 | 読み込み結果が同一 | `data/weekly/*.csv` の 2026 年全 82 本で `parse_csv` の DataFrame を切り出し前後で比較（列順・dtype・値。NaN 同士は一致扱い） | 82 本すべて完全一致 |
| 2 | bundle が同一 | `analysis/replay_serve_2026.py run` を 6 開催日（列数の異なる出走表形式 33/46/48/49/99 を網羅する日を選ぶ）で前後実行し、`p_win`・印・特徴カバレッジを比較 | 全馬の `p_win` が完全一致 |
| 3 | 馬履歴が同一 | `build_horse_history.py` を作業ディレクトリへ出力させ、行数と内容ハッシュを比較 | 一致 |
| 4 | T-10 経路が同一 | 9/27 の保存済み T-10 オッズで `compute_bets.py` を空実行し、買い目 JSON を比較 | 一致 |
| 5 | 調教結合が生きている | 切り出し後の bundle 生成ログで `trnH_*` のカバレッジが基準値と同じ | 一致（0% なら結合が無言で落ちている） |
| 6 | 既存テスト | `pytest`（現在 427 本） | 全通過 |
| 7 | 旧系統が壊れていない | `python -c "import predict_weekly, app_imports"` 相当の import 確認と `predict_weekly.parse_csv is weekly_parse.parse_csv` | 真 |
| 8 | 依存が切れた | `weekly_parse` を import した直後の `sys.modules` に `optuna`, `ev_filter`, `catboost`, `torch` が無い | 無い |

### 実装前に確認が要る点

1. **monkeypatch の向き先**: `tests/test_serve_parse.py` は `predict_weekly.TYAKU_DIR` を差し替えている。関数を移すと、差し替えは移動先モジュールに対して行わないと効かない。同じ手法で `BASE_DIR`・`HOSSEI_DIR` を差し替えている研究スクリプトが無いことは機械検索で確認済み（該当はこのテスト 2 か所のみ）だが、`candidates/` と `analysis/p0_5_verification/` はレビュー時に再確認する。
2. **`from x import *` ではなく明示列挙にする**: 先頭が `_` の名前（`_load_tyaku` など）は `*` では再 export されない。
3. **`optuna_lgbm.py` の [C] が使うパス定数**（`CHUKYO_DIR` と `E:\競馬過去走データ\` のマスター参照）を移動先へ正しく持っていくこと。参照先は読み取りのみで、不可侵ディレクトリへの書き込みは発生しない。
4. **進行中の他セッションの作業**と衝突しないこと（`predict_weekly.py`・`optuna_lgbm.py` に未コミット変更が無いことを着手時に確認）。
5. **a95 への影響**: なし。a95 は bundle の `p_win` と T-10 オッズだけを読む。確認 #2・#4 が一致すれば a95 の買い目も一致する。凍結方策・10/3 起算は変更しない。

## 5. やらないこと

- 段階 1 で `parse_csv` の中身を整理・改善しない（移動と改善を混ぜると出力不変の検証が意味を失う）。
- モデル・較正器・特徴定義には触れない。
- ファイル削除はしない（段階 3 でも移動のみ）。
