# 週次運用チートシート

最終更新: 2026-03-22

---

## 前半：レース前（金〜土朝）

### やること
週次CSV を `data/weekly/` に置いて PS1 を叩くだけ

```powershell
# YYYYMMDD = 開催日（例: 20260322）
.\weekly_pre.ps1 20260322
```

**PS1 が自動でやること（4ステップ）:**
1. `make_weekly_hosei.py` — 補正タイム当週分を生成（カバレッジ約65%）
2. `predict_weekly.py` — 予測実行 → `reports/pred_YYYYMMDD.csv` 生成
3. `git add` — weekly CSV をステージ
4. `git commit & push` — Streamlit Cloud に反映

> **確認**: Streamlit の「買い目」ページを開き、当該日付のレースが表示されれば OK

---

## 後半：レース後（日夜〜月）

### やること
払戻 CSV を `data/kekka/` に置いて PS1 を叩くだけ

```powershell
# YYYYMMDD = 開催日（例: 20260322）
.\weekly_post.ps1 20260322
```

**PS1 が自動でやること（3ステップ）:**
1. `generate_results.py` — 的中実績を再集計 → `data/results.json` 更新
2. `git add` — kekka CSV + results.json をステージ
3. `git commit & push` — 的中実績ページに反映

> **確認**: Streamlit の「的中実績」ページで HAHO/HALO のレース数が増えれば OK

---

## 戦略再構築（新モデル訓練後・四半期ごと）

モデル再訓練したら必ず戦略も再構築すること（旧モデル向け戦略は使えない）

```powershell
# バックテスト再実行（2本同時でも可）
python backtest.py --no_strategy --period valid --output_suffix _valid
python backtest.py --no_strategy --period test  --output_suffix _2024

# 戦略再生成
python build_strategy_stable.py   # → data/strategy_weights.json 更新

# 全pred CSV 再生成
.\batch_predict.ps1

# 的中実績更新
python generate_results.py
```

**現在の閾値設定（build_strategy_stable.py）:**
- MIN_ROI_PCT = 80（valid・test 両期間で80%以上）
- MIN_RACES = 30（各期間30R以上）
- 採用条件数: 30条件・8会場

---

## ファイル名規則

| フォルダ | ファイル名 | 中身 |
|---|---|---|
| `data/weekly/` | `YYYYMMDD.csv` | 出走表（JRA出力CSV） |
| `data/kekka/` | `YYYYMMDD.csv` | 払戻結果（JRA出力CSV） |
| `data/hosei/` | `H_YYYYMMDD.csv` | 補正タイム当週分（make_weekly_hosei.py 自動生成） |
| `data/training/` | `H-YYYYMMDD-YYYYMMDD.csv` | 坂路調教週次 |
| `data/training/` | `W-YYYYMMDD-YYYYMMDD.csv` | WC調教週次 |
| `reports/pred_*.csv` | `pred_YYYYMMDD.csv` | 予測結果（自動生成） |

---

## トラブルシューティング

| 症状 | 原因 | 対処 |
|---|---|---|
| 買い目ページにレースが出ない | weekly CSV がない or push 忘れ | `git push` 済みか確認 |
| 的中実績が増えない | results.json が古い | `python generate_results.py` 再実行 → push |
| スコアが全馬 0% | モデルファイル不整合 | `models/` 以下の pkl が揃っているか確認 |
| hosei カバレッジ 0% | kekka CSVがない or 18桁IDの型ズレ | `data/kekka/` に該当週の前走日付分があるか確認 |
| スタッキングWARNING | キャリブレーター出力が想定外 | 無害・エンサンブルにフォールバックして動作中 |
| weekly_pre でgit commit スキップ | 週次CSVが既にコミット済み | 正常（再実行時はスキップされる） |
| 対象レース0R | strategy_weights.json が現開催場所未対応 | 戦略再構築が必要（build_strategy_stable.py） |
