# Phase 5 実行TODO

**目的:** PyCaLiAIを「不必要に負けない構造」に修正し、「勝てる構造」に改善する

**ユーザー条件:**
- 1Rあたり上限10,000円
- 週10R以上は確保（16頭全面除外はしない）
- オッズはモデル入力にしない（EV判定のみ）
- アンサンブルで既存モデルを弱くしない
- 過去同条件好走パターンを3分類して検出する

---

## Step 1: バグ修正・設定修正 -- 完了 (2026-04-04)

### 1a. 京都を全面除外に変更 -- 完了
- `ev_filter.py`: `SKIP_VENUES = {"阪神", "京都"}`
- 根拠: 京都全体ROI 12.9%、年間-348k

### 1b. strategy_weights.jsonから阪神エントリ削除 -- 完了
- 阪神は全面除外なのにJSON内にエントリが残っている矛盾を解消

### 1c. BetFilterでev_calを使用 -- 完了
- `predict_weekly.py`: hon_ev_calを抽出しBetFilterに渡す
- `ev_filter.py`: EV_CAL_LOWER_LIMIT=0.78追加、ev_cal優先ルール

### 1d. 全pred CSVを現フィルタ付きコードで再生成 -- 保留
- モデル全更新後にまとめて実施予定

---

## Step 2: 確信度ベース賭け金 -- 完了 (2026-04-04)

- `predict_weekly.py` get_triple_bets() 改修済み
- ◎と4番手の確率ギャップで賭け金調整（0.5x〜1.5x）
- 16頭以上 × 低確信度 → 0.5倍（レース除外ではなく金額調整）

---

## Step 3: 好走3分類 + 隠れた適性特徴量 -- 完了 (2026-04-04)

### 3a. 好走の3分類を特徴量化 -- 完了
- `parse_kako5.py` _compute_features() に追加:
  - `kako5_expected_good_count`: 人気<=3 AND 着順<=3 (89.2%カバー, mean 1.048)
  - `kako5_upset_good_count`: 人気>=8 AND 着順<=5 (master CSVでは人気なしのため0%、週次予測時は有効)
  - `kako5_hidden_good_count`: 着順>5 AND 上り3F<34.5 (89.2%カバー, mean 0.250)
  - `kako5_same_cond_best_pos`: 過去5走の同TD+同距離帯ベスト (74.2%カバー, mean 4.063)

### 3b. 全キャリア同条件ベスト着順 -- 完了
- `parse_kako5.py` build_from_master() に追加:
  - `hist_same_cond_best_pos`: 全キャリア同条件ベスト (75.9%カバー, mean 3.333)
  - `hist_same_cond_top3_rate`: 全キャリア同条件複勝率 (75.9%, mean 0.334)
  - `hist_same_cond_count`: 全キャリア同条件出走回数 (75.9%, mean 6.467)
  - `hist_same_place_best_pos`: 全キャリア同場所ベスト (59.4%, mean 4.517)

### 3c-3d. NUM_FEATURES追加 + master_kako5.csv再生成 -- 完了
- `train_lgbm.py`, `train_catboost.py` に8特徴量追加
- master_kako5.csv: 626,774行 × 108列

### 3e. モデル再トレーニング -- 完了

| モデル | 旧Valid AUC | 新Valid AUC | 新Test AUC | V-T gap |
|--------|------------|------------|-----------|---------|
| LightGBM  | 0.7752 | 0.7723 | 0.7758 | 0.0035 |
| CatBoost  | 0.7691 | 0.7653 | 0.7698 | 0.0045 |

- AUC微減だが過学習なし。NaN率100%のkako5_upset_good_countの影響あり。
- lgbm_fukusho_v1.pkl → lgbm_optuna_v1.pkl にコピー済み（同上catboost）
- 旧optunaモデルは models/archive/ にバックアップ済み

---

## Step 4: ランキング学習 + 重み最適化 -- 完了 (2026-04-05)

### 4a. LightGBM LambdaRank -- 完了
- **新規ファイル:** `train_lgbm_rank.py`
- **保存先:** `models/lgbm_rank_v1.pkl`
- 着順をrelevanceスコアに変換（1着=4, 2着=3, 3着=2, 4-5着=1, 6着以降=0）
- グループ: `レースID(新/馬番無)` (レースレベルID)

| 指標 | Valid | Test | V-T gap |
|------|-------|------|---------|
| NDCG@3 | 0.5513 | 0.5645 | 0.013 |
| NDCG@5 | 0.6063 | 0.6183 | 0.012 |
| Top3Hit | 0.5891 | 0.5940 | 0.005 |
| BinaryAUC | 0.7487 | 0.7533 | 0.005 |

### 4b. 着順回帰モデル -- 完了
- **新規ファイル:** `train_regression.py`
- **保存先:** `models/lgbm_regression_v1.pkl`
- Huber損失（delta=3.0）でロバスト回帰

| 指標 | Valid | Test | V-T gap |
|------|-------|------|---------|
| MAE | 2.891 | 2.897 | 0.006 |
| RMSE | 3.620 | 3.625 | 0.005 |
| Top3Hit | 0.5828 | 0.5904 | 0.008 |
| BinaryAUC | 0.7652 | 0.7692 | 0.004 |

### 4c. アンサンブル重み最適化 -- 完了
- **新規ファイル:** `optimize_weights.py`
- **保存先:** `models/ensemble_weights.json`
- Nelder-Mead法、softmax制約、20回リスタート
- AUC目的とAUC+PR-AUC複合の2パターン探索
- Valid-Test gap < 0.01 のみ採用（超過時は均等重みフォールバック）

**結果:**

| 手法 | Valid AUC | Test AUC | V-T gap |
|------|-----------|----------|---------|
| 均等重み | 0.7753 | 0.7798 | — |
| **Nelder-Mead AUC最適** | **0.7767** | **0.7809** | **0.004** |

**最適化重み（データ駆動）:**
```
fuku_lgbm   : 57.4%  -- 主力（Phase5再トレーニング済みモデル）
regression  : 16.1%  -- 新規モデル、最大貢献
rank_cat    : 11.2%  -- YetiRank
catboost    :  6.6%
rank_lgbm   :  3.3%  -- LambdaRank
lgbm        :  2.8%
fuku_cat    :  2.7%
lgbm_win    :  0.1%  -- ほぼ不要
```

### 4d. ensemble_predict() 改修 -- 完了
- `predict_weekly.py` に以下を追加:
  - `predict_lgbm_rank()`: LambdaRankモデル予測 + レース内min-max正規化
  - `predict_lgbm_regression()`: 回帰モデル予測 + 着順反転 + レース内正規化
  - `_load_ensemble_weights()`: ensemble_weights.json自動ロード
  - `_ensemble_with_optimized_weights()`: 最適化重みベースアンサンブル
  - `_ensemble_fallback()`: 従来ロジック（重みファイル未存在時）
- モデルパス追加: `RANK_LGBM_PATH`, `REGRESS_PATH`, `ENS_WEIGHTS_PATH`

---

## Step 5: Mixture of Experts -- 未着手（次回以降）

- **ファイル:** `train_expert.py`（新規）
- **分割:**
  - Expert 1: 芝短距離（〜1400m）
  - Expert 2: 芝中距離（1600-2200m）
  - Expert 3: 芝長距離（2400m〜）
  - Expert 4: ダート全距離
- **判定:** レース条件からExpertを自動選択（if文）
- **フォールバック:** サンプル < 50,000 のExpertは使わず全体モデルを使用

---

## 変更ファイル一覧

### 修正ファイル
| ファイル | 変更内容 |
|---------|---------|
| `ev_filter.py` | 京都全面除外、ev_cal対応、EV_CAL_LOWER_LIMIT追加 |
| `predict_weekly.py` | ev_cal渡し、確信度賭け金、新モデル予測関数、最適化重みアンサンブル |
| `parse_kako5.py` | 好走3分類 + 全キャリア同条件ベスト特徴量 |
| `train_lgbm.py` | NUM_FEATURES +8特徴量 |
| `train_catboost.py` | NUM_FEATURES +8特徴量 |
| `data/strategy_weights.json` | 阪神エントリ削除 |

### 新規ファイル
| ファイル | 内容 |
|---------|------|
| `train_lgbm_rank.py` | LambdaRankモデル学習 |
| `train_regression.py` | 着順回帰モデル学習 |
| `optimize_weights.py` | Nelder-Meadアンサンブル重み最適化 |
| `TODO_phase5.md` | この実行計画 |

### 新規モデル
| モデル | サイズ | 用途 |
|-------|-------|------|
| `models/lgbm_rank_v1.pkl` | LambdaRank | ランキング予測 |
| `models/lgbm_regression_v1.pkl` | 着順回帰 | 着順直接予測 |
| `models/ensemble_weights.json` | 最適化重み | アンサンブル自動設定 |

---

## 次回TODO
1. Step 5 (MoE) 実装
2. pred CSV全再生成 + results.json再構築
3. calibrate.py再実行（新アンサンブル構成でキャリブレーション更新）
4. backtestで実戦ROI検証
