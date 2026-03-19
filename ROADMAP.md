# PyCaLiAI 強化ロードマップ

最終更新: 2026-03-19（Transformer PL訓練完了・4モデルアンサンブル統合）

---

## Phase 1: データ強化 ✅ 完了

| タスク | 状態 | 結果 |
|---|---|---|
| 調教データ（坂路・WC）JOIN | ✅ | カバレッジ: 坂路65.8% / WC21.2% |
| 前走単勝オッズ JOIN | ✅ | kekka CSVから（今走はリークのため除外） |
| LGBM 再訓練 | ✅ | Valid 0.7412→**0.7551** / Test 0.7474→**0.7594** |
| CatBoost 再訓練 | ✅ | Valid 0.7433→**0.7656** / Test 0.7431→**0.7706** |
| Ensemble Calibrator 更新 | ✅ | Brier score +29.6%改善 |
| 全pred CSV 再生成（24週分） | ✅ | |
| git commit | ✅ | `6ddb114` |

---

## Phase 2: ランキング学習 ✅ 完了

**目標**: 二値分類（勝つ/負け）→ レース内順位最適化に変更

CatBoostの **YetiRank**（GPU対応、精度が高い）から実装する。

| タスク | 状態 | 概要 |
|---|---|---|
| optuna_catboost_rank.py 作成 | ✅ | YetiRank目的関数、groupパラメータでレース単位グループ化 |
| 評価指標の設定 | ✅ | NDCG（CatBoost組み込み）、AUCで分類器と比較 |
| YetiRankモデル訓練 | ✅ | Valid AUC=0.7482 / Test AUC=0.7546（分類器: 0.7656/0.7706） |
| predict_weekly.pyへの統合 | ✅ | LGBM 0.4 + CatBoost 0.4 + Rank 0.2 の加重平均 |
| 二値分類モデルとROI比較 | ✅ | Test ROI: 2モデル61.9% → 3モデル69.9%（+8.0pt） |

**なぜ有効か**: 今は「馬が複勝に絡むか」を単体で予測しているが、ランキング学習は「このレースの中でどの馬が上位か」を直接最適化する。馬券は相対評価なのでこちらが本質的に正しい。

---

## Phase 3: Transformer × Plackett-Luce ✅ 完了

**注**: 訓練データは2015年〜分が揃っている。実装コストが高いため後回しにしているだけ。

| タスク | 状態 | 概要 |
|---|---|---|
| Transformer実装 | ✅ | レース内全馬を一括入力（注意機構で馬間関係を学習） |
| Plackett-Luce損失 | ✅ | 全順列の確率を最大化する損失関数 |
| Optuna 30試行 | ✅ | Best Valid AUC=0.7255（d_model=64, n_heads=2, n_layers=4） |
| 最終訓練 | ✅ | Valid AUC=0.7231 / Test AUC=0.7304（Early stopping Epoch 16） |
| 4モデルアンサンブル統合 | ✅ | LGBM×0.30 + CatBoost×0.30 + YetiRank×0.20 + TransPL×0.20 |
| Ensemble Calibrator 更新 | ✅ | Brier score +32.8%改善（前回3モデル比+3.2pt） |
| 全pred CSV 再生成（24週分） | ✅ | |

**なぜ有効か**: 競馬は「このレースの出走馬全員の相互関係」が重要。Transformerはそれを自然に扱える。Plackett-Luceは順位全体の確率を一度に最適化する理論的に最も正しい損失関数。

---

## 週次運用（常時）

| タスク | 頻度 | 状態 |
|---|---|---|
| 週次CSVをdata/weekly/に配置 | 毎週 | ✅ 運用中 |
| predict_weekly.py 実行 | 毎週 | ✅ 運用中 |
| master CSVにデータ追加（年次） | 年1回 | ⬜ 次回2026年末 |

---

## 現在のモデル性能

| モデル | Valid AUC | Test AUC |
|---|---|---|
| LGBM（65特徴量） | 0.7551 | 0.7594 |
| CatBoost（65特徴量） | 0.7656 | 0.7706 |
| CatBoost YetiRank（ランキング） | 0.7482 | 0.7546 |
| Transformer Plackett-Luce | 0.7231 | 0.7304 |
| 4モデルアンサンブル（+TransPL） | - | ROI: LALO **76.8%** / CQC **75.3%** |

---

## メモ: データリーク注意事項

- `単勝オッズ`（今走）= kekka CSVの`単勝配当` → 勝ち馬のみ値あり = **完全リーク** → 除外済み
- `前走単勝オッズ` = 前走の配当（前走の勝敗情報として有効）→ 使用中
- train/valid/test分割: train=〜2022年 / valid=2023年 / test=2024年〜
