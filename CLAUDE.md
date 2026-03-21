# PyCaLiAI — Claude Code 引き継ぎメモ

## Claude への作業指示

### 自律性（重要）
ファイル編集・読み込み・モデル学習・バックテスト・スクリプト実行・コミットなどの通常作業では、いちいち許可を求めない。情報をもらったら自分で判断して自律的に進める。

確認が必要なのは以下のみ：
- git push（リモートへの反映）
- データ削除（不可逆な操作）
- 明らかにスコープ外の大きな変更

「やりますか？」「進めてもいいですか？」は言わない。やる。

---

## プロジェクト概要
競馬予測 AI。JRA の出走表 CSV を入力として複勝/馬連/三連複の買い目と購入額を自動生成し、
Streamlit UI で週末レースの推奨を提示する。

- **ターゲット変数**: `fukusho_flag`（3着以内 = 1、正例率 ≈ 21.9%）
- **予算**: 1R = 1万円、土曜・日曜それぞれ 10R 以上を目標
- **除外レース**: 東京・小倉・新馬・障害（`EXCLUDE_PLACES` / `EXCLUDE_CLASSES`）
- **三連単**: 廃止済み（predict_weekly.py でスキップ）

---

## 環境
- Python 3.11（`venv311\`）
- Windows 11 / `E:\PyCaLiAI`
- UI 起動: `streamlit run app.py`
- 仮想環境アクティベート: `venv311\Scripts\activate`

### GPU / torch
`venv311` に torch・CUDA 12.8 インストール済み。4モデル全て動作中。
スタッキングキャリブレーターは WARNING が出るが無害（エンサンブルにフォールバック）。

---

## 時系列分割
| セット | 期間 |
|---|---|
| Train | 〜 2022-12-31 |
| Valid | 2023-01-01 〜 2023-12-31 |
| Test  | 2024-01-01 〜 |

---

## 現役モデルファイル（models/）
| ファイル | 状態 | 備考 |
|---|---|---|
| `lgbm_rank_v1.pkl` | ✅ 現役 | LGBM（調教特徴量拡張版）2026-03-20再訓練 |
| `lgbm_fukusho_v1.pkl` | ✅ 現役 | LGBM 複勝特化版 |
| `catboost_fukusho_v1.pkl` | ✅ 現役 | CatBoost YetiRank 2026-03-20再訓練 |
| `transformer_optuna_v1.pkl` | ✅ 現役 | Transformer PL、CUDA動作 2026-03-20再訓練 |
| `ensemble_calibrator_v1.pkl` | ✅ 現役 | 4モデル加重平均用 Isotonic |
| `stacking_calibrator_v1.pkl` | ⚠️ WARNING | 出力異常でエンサンブルにフォールバック中（無害） |

---

## 週次運用フロー（通常操作）
```bash
# 1. 週末出走表を data/weekly/ に配置（YYYYMMDD.csv）

# レース前（週次CSV配置後）
.\weekly_pre.ps1 YYYYMMDD
# → make_weekly_hosei.py → predict_weekly.py → git push まで自動

# レース後（kekka CSV配置後）
.\weekly_post.ps1 YYYYMMDD
# → generate_results.py → git push まで自動
```

---

## 戦略構築フロー（月次〜四半期）
```bash
# 1. バックテスト（Valid セット）
python backtest.py --output_suffix _valid

# 2. 条件確認（参考）
python check_all_conditions.py      # valid データ使用（デフォルト）
python check_all_conditions.py --train  # train データで確認したい場合

# 3. 戦略構築（どちらか選ぶ）
python build_strategy_walkforward.py  # 推奨: walk-forward × OOS 二段階フィルタ → strategy_weights.json
python build_strategy_stable.py       # strict: valid+test 両方黒字のみ（条件数が少ない）
```

**現在の strategy_weights.json**: build_strategy_stable 版（30条件・8会場、MIN_ROI_PCT=80）
**2026年実績（新モデル+新戦略）**: HAHO 73.5% / HALO **102.7%（黒字）** / LALO 81.6% / CQC 77.6%

---

## 学習パイプライン（再学習時のみ）
```bash
python build_dataset.py          # 1. master CSV 生成
python optuna_lgbm.py            # 2. LightGBM チューニング
python optuna_catboost.py        # 3. CatBoost チューニング
# python optuna_transformer.py  # torch 必要
python stacking.py               # 4. スタッキング（torch 必要）
python calibrate.py              # 5. キャリブレーター生成（ensemble + stacking）
python backtest.py --output_suffix _valid   # 6. バックテスト
python build_strategy_walkforward.py        # 7. 戦略構築
```

---

## アーキテクチャ（予測フロー）

```
出走表 CSV
    ↓ parse_csv()
predict_stacking()  ←── lgbm_optuna_v1.pkl
                    ←── catboost_optuna_v1.pkl
                    ←── transformer_optuna_v1.pkl  ← torch 未インストール時スキップ
                    ←── stacking_meta_v1.pkl
                    ↓ (None の場合フォールバック)
ensemble_predict()  ←── lgbm 0.5 + catboost 0.5
                    ↓
calibrator.transform()  ←── ensemble_calibrator_v1.pkl or stacking_calibrator_v1.pkl
                    ↓
スコア / 印 / 買い目 / Kelly 金額
```

---

## 主要定数（全ファイル共通にすべき値）

| 定数 | 値 | 定義場所 |
|---|---|---|
| `EXCLUDE_PLACES` | `{"東京", "小倉"}` | predict_weekly.py, backtest.py, app.py |
| `EXCLUDE_CLASSES` | `{"新馬", "障害"}` | 同上 |
| `MIN_UNIT` | `100`（円） | 各ファイルに個別定義（要統一） |
| `BUDGET` | `10_000`（円/R） | 各ファイルに個別定義（要統一） |
| `CLASS_NORMALIZE` | 旧クラス名 → 新クラス名 | predict_weekly.py, app.py, calibrate.py |

⚠️ `config.py` に上記が定義されているが **誰も import していない**（完全デッドコード）

---

## ファイル役割マップ

### 本番（週次で触る）
| ファイル | 役割 |
|---|---|
| `app.py` | Streamlit UI |
| `predict_weekly.py` | 週次CSV → 買い目CSV |
| `utils.py` | 共通関数: `add_meta()`, `parse_time_str()`, `backup_model()` |

### 戦略構築（月次〜四半期）
| ファイル | 役割 |
|---|---|
| `backtest.py` | 実オッズ バックテスト → `reports/backtest_results_*.csv` |
| `build_strategy_walkforward.py` | ✅ 現役: walk-forward → `strategy_weights.json` |
| `build_strategy_stable.py` | ✅ 現役: valid+test strict → `strategy_weights.json` |
| `calibrate.py` | キャリブレーター生成 → `ensemble_calibrator_v1.pkl` / `stacking_calibrator_v1.pkl` |
| `check_all_conditions.py` | 条件別 ROI 確認（分析用参考スクリプト） |

### 学習（再学習時のみ）
| ファイル | 役割 |
|---|---|
| `build_dataset.py` | CSV 結合 → master CSV |
| `train_lgbm.py` / `train_catboost.py` / `train_transformer.py` | baseline 学習 |
| `optuna_lgbm.py` / `optuna_catboost.py` / `optuna_transformer.py` | Optuna チューニング |
| `stacking.py` | Level-2 meta model（torch 必要） |
| `torch_csv_builder.py` | Transformer 用 CSV 生成 |
| `generate_course_trend.py` | `course_trend.json` 生成 |
| `generate_results.py` | `results.json` 生成 |

### 分析・確認（アドホック）
| ファイル | 役割 |
|---|---|
| `simulation.py` | 複数戦略 ROI 比較 |
| `validation.py` | 時系列/ドローダウン/モンテカルロ |
| `check_*.py` (8本) | 使い捨て分析スクリプト群（削除候補） |

### ❌ デッドコード / 旧版（削除候補）
| ファイル | 理由 |
|---|---|
| `config.py` | 誰も import していない（128行） |
| `build_strategy.py` | walkforward/stable に置き換え済みの旧34行スクリプト |
| `train_lgbm_rank.py` | 本番未使用の可能性大 |
| `train_transformer_pl.py` | PyTorch Lightning 版、旧実装 |
| `ensemble.py` | 役割不明（stacking.py と重複の可能性） |
| `betting.py` | 旧版買い目生成、`predict_weekly.py` に統合済み |
| `report.py` | 役割不明 |
| `kelly.py` | 役割不明（`predict_weekly.py` の Kelly と重複？） |
| `filter_analysis.py` | アドホック分析スクリプト |
| `inspect_csv.py` | アドホック分析スクリプト |

---

## データファイル
```
data/
  master_20130105-20251228.csv     訓練マスター（約62万行、split列あり）
  kekka_20130105-20251228.csv      払戻マスター
  strategy_weights.json            ★戦略設定（stable版、30条件8会場、MIN_ROI=80%）
  course_trend.json                コース傾向
  results.json                     直近成績
  weekly/YYYYMMDD.csv              週次入力
  kekka/YYYYMMDD.csv               週次払戻
  hossei/H_20130105-20251228.csv   補正タイム（cp932、JOINキー: レースID(新)+馬番）

E:\競馬過去走データ\              ← TARGETフロンティア出力（プロジェクト外）
  H-20150401-20260313.csv         坂路調教マスター（520万行、cp932）
  W-20150401-20260313.csv         WC調教マスター（70万行、cp932、2021/7〜）
```

### 調教データ仕様
- **坂路（H）列**: 場所, 年月日, 馬名, Time1(4F合計), Lap4, Lap3, Lap2, Lap1(最終200m)
- **WC（W）列**: 場所, コース, 回り, 年月日, 馬名, 5F, 4F, 3F, Lap3, Lap2, Lap1
- **JOIN方法**: 馬名 + 年月日（レース14日前以内の最終追い切り）で merge_chukyo()
- **特徴量**: trn_hanro_4f, trn_hanro_lap1, trn_hanro_days, trn_wc_3f, trn_wc_lap1, trn_wc_days
- **カバレッジ**: 坂路80%（2015〜）、WC25%（2021〜、2022以降は67%）
- **週次更新**: ユーザーがTARGETフロンティアから日別エクスポート → 月次or週次でマスター更新

---

## ⚠️ 残課題（優先順）

### 🔴 高優先
1. **スタッキングキャリブレーター修正**: WARNING が出てエンサンブルフォールバック中。`calibrate.py` 再実行で解消できる可能性あり
2. **`config.py` の整理**: 削除 or 各スクリプトから実際に import する

### 🟡 中優先
3. **`backtest.py` の三連単**: `BUDGET_RATIO` から三連単を外す（predict_weekly はスキップ済みなのに backtest は 25%割当中 → 回収率計算がズレる）
4. **`models/archive/`** に旧モデルを退避（`lgbm_fukusho_v1.pkl` 等 8本）
5. **ルートの整理**: `check_*.py` (8本) / `ensemble.py` / `betting.py` 等を `analysis/` や `scripts/` に移動

### 🟢 低優先
6. **`catboost_info/`** を `.gitignore` に追加
7. **`MIN_UNIT` / `BUDGET`** を `config.py` (or 新設定ファイル) に集約
8. `predict_weekly.py` の出力列から `三連単_買い目`, `三連単_購入額` を削除

---

## 馬券種と印の対応（確定仕様）
| 馬券種 | 対象 | 買い目 |
|---|---|---|
| 複勝 | ◎◯▲ 各1頭 | 計3点 |
| 馬連 | ◎◯▲ | ◎-◯, ◎-▲, ◯-▲（3点流し） |
| 三連複 | ◎◯軸 × ▲△ | 最大2点 |
| 三連単 | **廃止** | — |

---

## キャリブレーション結果（2023 Valid）
- Brier スコア: 0.1990 → **0.1462**（+26.5%改善）
- 予測確率 mean: 0.438 → **0.219**（実際の正例率と一致）
- 図: `reports/calibration_ensemble.png`

---

## 注意事項
- 当日情報（馬体重・オッズ等）は学習特徴量から除外済み（`FORBIDDEN_COLS` は `config.py` 参照）
- `COL_RACE_ID` のカラム名が `"レースID(新)"` と `"レースID(新/馬番無)"` で揺れている
  - master CSV では `"レースID(新/馬番無)"` を使用
  - `backtest.py` は `COL_RACE_ID = "レースID(新/馬番無)"` で統一済み
