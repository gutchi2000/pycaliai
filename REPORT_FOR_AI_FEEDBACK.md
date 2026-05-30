# PyCaLiAI 印信頼度向上 — 他 AI への相談書

> **作成日**: 2026-05-25
> **目的**: 印精度 (特に ◎top3, ◎top1) の向上のための新しいアプローチを募集
> **読者**: ChatGPT, Gemini, Grok などの他 AI

---

## 0. TL;DR

JRA 中央競馬の AI 予想 (印付け ◎〇▲△△) を作っている。 LightGBM LambdaRank をベースとした本番モデル `v6` は test 期間で **◎top3 = 62.03%** を達成しているが、 そこから先の改善が頭打ち。 過去 1 ヶ月で 8 つのアプローチを試したが、 **2 つしか採用候補にならなかった**。 印信頼度をさらに上げるための **構造的に新しいアプローチを募集**。

---

## 1. プロジェクト概要

### 1.1 タスク

JRA 中央競馬の出走馬 (1 レース 8-18 頭) について、 上位 5 頭に印 ◎〇▲△△ を付ける順位学習タスク。

**主要な評価指標**:
- `◎top3` (= hon_top3_rate): ◎ が実際に 1-3 着以内に入った率
- `◎top1`: ◎ が実際に 1 着だった率
- `◎top2`: ◎ が実際に 1-2 着だった率
- `top2_eq_top2`: ◎〇 のセット = 実 1-2 着セット率 (馬連的中)
- `top3_eq_top3`: ◎〇▲ のセット = 実 1-3 着セット率 (三連複的中)

### 1.2 役割分担 (重要)

```
PyCaLiAI (このプロジェクト)        → 印付け + Plackett-Luce 確率算出 (bundle.json 出力)
Cowork (Claude Desktop App)       → bundle.json を読んで買い目・点数・予算配分を決定
NiceGUI/Streamlit                 → 表示専用
```

つまり、**印精度を上げると Cowork の判断材料が良くなる** = 馬券 ROI に直結する。

### 1.3 データ

- **マスター**: `data/master_v2_20130105-20251228.csv` (515MB, 約 62 万行)
  - 個馬×レースの行データ
  - 120 列 (個馬特徴 + レース context + 過去 5 走集約)
- **払戻**: `data/kekka_20130105-20251228.csv` (cp932 エンコード)
  - 各レースの 1-3 着馬番 + 各馬券種払戻
- **時系列分割** (重要、 リーク防止):
  - Train: 〜 2022-12-31 (約 48.5 万行)
  - Valid: 2023-01-01 〜 2023-12-31 (約 4.7 万行、 3,456 races)
  - Test: 2024-01-01 〜 (約 9.4 万行、 6,909 races)

### 1.4 ターゲット変数 (LightGBM の label)

```python
df["label"] = np.clip(6 - df["着順"].astype(int), 0, 5).astype(int)
```
→ 1着=5, 2着=4, 3着=3, 4着=2, 5着=1, 6着以下=0

### 1.5 除外データ

- 障害レース、新馬戦中心の除外（旧 EXCLUDE_PLACES は v5 で廃止）
- 三連単馬券は使用しない (廃止)

---

## 2. 本番モデル `unified_rank_v6.pkl` の詳細

### 2.1 アーキテクチャ

- **ライブラリ**: LightGBM 4.x
- **objective**: `lambdarank`
- **lambdarank_truncation_level**: **5** (上位 5 位までで損失打ち切り)
- **metric**: NDCG@5
- **group**: レース ID 単位

### 2.2 Optuna 探索した best hyperparams

```json
{
  "alpha": 0.0308,        // sample_weight 用係数 (winner_tansho 比例)
  "lr": 0.0508,           // learning_rate
  "num_leaves": 59,
  "max_depth": 12,
  "min_data_in_leaf": 197,
  "ff": 0.876,            // feature_fraction
  "bf": 0.703,            // bagging_fraction
  "l1": 0.0011,           // lambda_l1
  "l2": 7.538,            // lambda_l2
  "seed": 42,
  "deterministic": true,
  "best_iter": 515        // 学習後の num_trees
}
```

### 2.3 sample_weight 設計

```python
w = 1.0 + alpha * np.log1p(winner_tansho / 100.0)
```
- `winner_tansho` = そのレースの 1 着馬の単勝払戻 (円、 100 円ベース)
- 穴勝ちレースに少し重み (alpha=0.0308 なので影響は微小)
- 上位馬個別ではなく **レース全体**にかかる weight

### 2.4 Optuna 目的関数 (composite metric)

5-fold CV で valid (2023) を評価:

```python
composite = (
    0.30 * NDCG@5
  + 0.25 * ◎top3 (hon_top3_rate)
  + 0.20 * top3_subset_top5  # 実1-3着が予測top5に全部含まれる率
  + 0.15 * winner_in_top5    # 1着馬が予測top5に含まれる率
  + 0.10 * ◎top2 (hon_top2_rate)
) - 0.5 * ECE_high_p
```

`ECE_high_p` = 予測勝率 (PL prob) が高い領域 (≥0.10) で |予測平均 - 実勢平均|。 calibration awareness のため減点。

### 2.5 特徴量 (120 列)

**個馬特徴 (LightGBM への入力)**:
- 騎手統計 (`jockey_fuku30`, `jockey_fuku90`, `jockey_top3_rate`)
- 厩舎統計 (`trainer_fuku30`, `trainer_fuku90`)
- 馬統計 (`horse_fuku10`, `horse_fuku30`)
- 過去 5 走集約 (`kako5_avg_pos`, `kako5_best_pos`, `kako5_avg_ninki`, `kako5_pos_vs_ninki`, `kako5_avg_agari3f`, `kako5_best_agari3f`, etc.)
- コース統計 (`course_n_prev`, `course_win_rate`, `course_top3_rate`)
- 同条件履歴 (`hist_same_cond_top3_rate`, `hist_same_cond_count`, etc.)
- 調教データ (`trn_hanro_4f`, `trn_hanro_lap1`, `trn_wc_3f`, etc.) — 坂路 80%, WC 25% カバレッジ
- カテゴリ: 場所、 芝・ダ、 距離、 クラス、 馬場、 天気、 種牡馬、 母父、 騎手コード、 調教師コード、 etc.

**禁則 (LEAK_COLS)**:
- 着順、 fukusho_flag、 roi_target
- レースID、 馬名、 レース名、 発走時刻、 日付
- 当日オッズ・馬体重 (リークリスク)

### 2.6 本番モデルの性能 (test 2024-25, 6,908 races)

```
◎top1 (◎=1着)         : 30.24%
◎top2 (◎=1-2着)       : 49.46%
◎top3 (◎=1-3着)       : 62.03%
top2_eq_top2 (◎〇=12)  : 14.11%
top3_eq_top3 (◎〇▲=123): 7.22%
winner_in_top5         : 78.20%
```

**集中モード ROI (固定買い目)**:
- 馬連 ◎-〇▲ 各 2,000 円
- ワイド ◎-〇▲△△ 各 1,000 円
- ワイド ◎-〇▲ 追加各 1,000 円
- 計 10,000 円/R
- **ROI = 82.12%, P&L = -12.3M / 6,908R**

---

## 3. 試行履歴 (失敗 ❌)

### 3.1 ❌ v7: race-relative 特徴量 + Wide ROI 目的関数

**変更点**:
- 22 列の race-relative 特徴量を追加 (z-score, rank, peer_max/min など)
- Optuna 目的関数に直接 `wide_roi - ECE_wide` を加算

**結果**: valid 過学習。 test で全指標悪化:
- ◎top3: 62.03% → 59.90% (**-2.13 ppt**)
- ECE_high_p: 0.0112 → 0.0162 (悪化)

**教訓**: valid metric を直接 objective に入れると過学習する。

### 3.2 ❌ v8: + course_affinity 特徴量

**変更点** (v7 ベース):
- 4 つの集計テーブル追加 (sire/msire/jockey/horse-dist × コース)
- 12 列追加 (合計 +34 列 = 154 列)

**結果**:
- ◎top3: 62.03% → **52.92%** (-9.11 ppt) 大幅悪化
- NaN 65% の horse_dist 列がノイズ

**教訓**: 高 NaN 列は学習を傷つける、 集計テーブル粒度を細かくすると test で破綻。

### 3.3 ❌ v9: LambdaRank truncation=3 + 上位 3 位 composite

**変更点** (v6 ベース):
- `lambdarank_truncation_level`: 5 → 3
- composite: `0.40 × NDCG@3 + 0.30 × ◎top3 + 0.20 × ◎top2 + 0.10 × top3⊂top3`

**結果**: 全項目悪化
- ◎top3: 62.03% → 61.12% (-0.91 ppt)
- ◎top1: 30.24% → 29.05% (-1.19 ppt)
- 集中モード ROI: 82.12% → 81.83% (-0.28 ppt)

**教訓**: truncation=3 にすると LambdaRank が下位馬の序列を捨ててしまい、 結果として上位 3 位の選別も荒れる。 truncation=5 が局所最適。

### 3.4 ❌ 多 seed アンサンブル (v6 × seeds=123/456/789/1234)

**手法**: v6 best params を流用、 seed だけ変えて 4 model 追加学習 → 5 model の score 平均化

**結果**:
- ◎top3: 62.03% → 62.16% (+0.13 ppt) 微改善
- 集中モード ROI: 82.12% → **80.27%** (**-1.85 ppt**) 悪化

**spearman(seed=42, seed=123) = 0.9856** (多様性ゼロ)

**教訓**: 同じ hyperparams で seed だけ変えても LightGBM は決定的に類似モデルを作る。 アンサンブル効果なし。

### 3.5 ❌ コース別 calibrator (芝/ダ × 距離区分 = 8 isotonic)

**手法**: PL 確率の isotonic calibrator を 8 コース区分別に fit

**結果** (test 2024-25 で OOS 評価):
- 単勝 ECE: global 0.00305 vs course 0.00877 (**-187%** 悪化)
- 馬連 ECE: global 0.00579 vs course 0.01121 (-94% 悪化)
- ワイド ECE: global 0.00450 vs course 0.00857 (-91% 悪化)

**教訓**: valid (2023) のサンプル数が小区分で不足 (ダ_long 54R, ダ_middle 105R)、 in-sample 過学習が OOS で破綻。

### 3.6 ⚪ Phase 2: transformer × v6 真面目スタック

**手法**: 既存の transformer_pl_v2 と v6 を `α * v6 + (1-α) * trans` で blend、 α grid search

**結果**: 全指標で v6 single を微妙に超える
- ◎top3 (α=0.85): 62.03% → 62.17% (+0.14 ppt)
- ◎top1 (α=0.95): 30.24% → 30.34% (+0.10 ppt)
- top3_eq_top3 (α=0.80): 7.22% → 7.59% (+0.36 ppt)
- 集中モード ROI (α=0.95): 82.12% → 82.31% (+0.19 ppt)

**判定**: 改善幅が小さく、 transformer の本番組込み (CUDA + torch) コストに見合わない → 不採用

---

## 4. 試行履歴 (採用候補 ✅)

### 4.1 ✅ 馬連 pair 二値分類 + v6 stack

**手法**: 馬連の全ペア (i, j) で二値分類 (`is_top2`) を学習し、 v6 PL prob と blend
- LightGBM binary, AUC=0.8152, 130 万ペア学習

**結果** (馬連 EV top-3, test 2024-25):
- v6 単独: ROI 75.37%
- **v6 80% + pair 20% (α=0.8)**: ROI **81.89%** (**+6.52 ppt** 改善)

注: ◎top3 などの印精度は変わらず、 馬連選択精度のみ改善。

### 4.2 ✅ 複勝特化 二値分類 + v6 stack

**手法**: 個馬の `is_top3` binary を v6 score 含む特徴量で学習
- LightGBM binary, AUC=0.7717

**結果** (複勝 top-3 per race, test 2024-25):
- v6 PL 単独: ROI 82.73%
- **binary 単独 (α=1.0)**: ROI **83.71%** (+0.98 ppt)

注: hit 率は微減 (49.53% → 49.12%) だが avg_pay 微増 (167 → 170)。 印精度ではなく EV ベースの選択改善。

### 4.3 ✅ 波乱予想モデル (Phase 3) ⭐ **強力**

**手法**: レース単位 binary でレース前情報から「1着オッズ ≥ 5 倍になる確率」を予測
- LightGBM binary, AUC=0.6273 (弱いが)

**特徴量** (レース単位):
- 出走頭数, クラス, 場所, 距離, 芝・ダ, 馬場状態, 天気
- 出走馬の v6 score 分布 (max, min, std, range, top3_mean, top5_mean)
- 1番人気の v6 score、 PL prob 上位 3 sum
- レース全体の Plackett-Luce entropy

**結果** (test 2024-25、 6,908 races):

| カテゴリ | n | 実 havoc | ◎top1 | ◎top3 | 集中 ROI |
|---|---|---|---|---|---|
| 予想固い (p<0.3) | 706 | **23%** | **48.87%** | **85.69%** | **84.30%** |
| 中庸 (0.3-0.5) | 3,849 | 42% | 32.32% | 65.55% | 83.06% |
| 予想波乱 (0.5-0.7) | 2,186 | **56%** | 21.55% | 49.86% | 79.77% |
| 予想大波乱 (p≥0.7) | 167 | 69% | 17.37% | 40.12% | 81.84% |
| 全レース | 6,908 | 45% | 30.24% | 62.03% | 82.12% |

**示唆**:
- AUC 0.63 と弱い分類器でも **予想固いレース** (706R) で実 havoc rate を 23% に削減
- 予想固いで **◎top3 = 85.69%** (全体 62% から **+23.7 ppt** 改善)
- 「予想波乱以上スキップ」で ROI +1.13 ppt + 損失 40% 削減

**注**: 印そのものは変えていない。「**v6 が強いレースを事前に選別**」している。

---

## 5. 試したが当てはまらなかった / 検討中

- LightGBM の `dart` mode (drop-out trees)
- Optuna `multi_objective` (NDCG + ECE 二目的)
- HALO formation (旧式の Kelly 系)
- 三連単 (廃止)

---

## 6. 質問 — 印信頼度を上げる方法は？

現状 ◎top3 = 62.03% (test 2024-25)、 集中モード ROI = 82.12%。

**目標**:
- ◎top3 を **65%+** に押し上げる (現状 +3 ppt)
- ◎top1 を **33%+** に押し上げる (現状 +3 ppt)
- 集中モード ROI を **85%+** に押し上げる (控除率 80% を明確に超える)

### 6.1 我々が**既に検討中**で軽い順:

**B. Class weighting (top3 馬に高 weight)** (期待: ◎top3 +0.5-1 ppt)
- 現状: `weight = 1 + alpha * log1p(winner_tansho / 100)` (レース全体)
- 追加案: 個別馬の `weight = (3 if 着順≤3 else 1) × 上記`
- リスク: alpha と相互作用、 オーバーフィット

**C. Smooth label (上位差強調)** (期待: ◎top3 +0.5-1 ppt)
- 現状 label: `[5, 4, 3, 2, 1, 0, 0, ...]`
- 案: `[5.5, 4.7, 4.0, 2.0, 1.0, 0, 0, ...]` (上位 3 位の差を明示)
- LambdaRank が上位 3 位の相対 ranking に集中

**A. CatBoost LambdaRank ensemble** (期待: ◎top3 +0.5-1.5 ppt)
- v6 LGBM + CatBoost LambdaRank の真の混合 (depth-wise vs leaf-wise で多様性確実)
- Optuna 3-5h コスト

### 6.2 我々が**まだ検討していない**方向 (聞きたい):

1. **Multi-task learning** (LGBM では難しいが PyTorch なら可)
   - 同時に学習: `clip(6-着順, 0, 5)` ranking + `is_top1` binary + `is_top3` binary
   - 共有 encoder + 複数 head

2. **Custom objective**: composite metric を differentiable loss として LightGBM custom obj に書く
   - 直接 composite を最適化 (今は Optuna で間接最適化)
   - 実装複雑、 不安定

3. **Bagging by races**: train データの race 単位 bootstrap で 10 model 学習
   - 同 hyperparams でもデータ多様性で ensemble 効果
   - 案 2 (seed 多様性) が失敗した代替

4. **Distillation / Self-training**:
   - v6 score 高 confidence の test を疑似ラベル化して train に追加
   - リーク疑惑あり、 careful 必要

5. **Stage 1: 印精度信頼スコア → Stage 2: 印補正**
   - Stage 1: 「この馬が v6 で正しく top3 と評価されているか」を予測する binary
   - Stage 2: 低信頼の馬を ◎ から外す、 高信頼を ▲ → 〇 に格上げ etc.
   - 一種の self-supervised calibration

6. **時系列重み付け (recent emphasis)**:
   - 直近 2 年に exp decay weight
   - 競馬市場の時間ドリフトに追従

7. **Pseudo-labeling with rejection**:
   - 信頼度高い test data を train に組込
   - 低信頼は reject

8. **モデル**: Transformer / Tabular Net / NeuralNDCG / LightGCN (graph-based) など完全別系統

9. **特徴量**: 当日オッズ snap (例 9:00 オッズ) — リークなし、 強力なシグナル。 ただし取得運用コスト発生

10. **ラベル smoothing 系の数学的工夫**:
    - 着順を Gaussian smoothing で連続化
    - 着順差を log scale で表現

### 6.3 求める回答

以下の **どれか or 複数** について具体的な技術的アドバイス：

1. **未試の有望なアプローチ** (上記 6.2 か独自案)
2. **試した失敗から学べる教訓** (v7/v8/v9/多 seed/コース別 cal の失敗パターンに共通点はあるか)
3. **構造的に何を変えるべき** (LightGBM LambdaRank では本質的に到達できない天井があるなら指摘)
4. **競馬予測のドメイン知識** (我々が見落としているシグナル、 特徴量、 ラベル設計)

---

## 7. 環境

- Python 3.11
- LightGBM 4.x
- Windows 11 / RTX 3070 Ti (CUDA 12.8 動作中)
- ハードウェア: 一般的なゲーミング PC (GBM は CPU、 PyTorch は GPU)
- 学習時間: v6 1 回学習 = 約 30-60 分

---

## 8. 補足: 重要な制約

- **三連単は廃止** (Cowork で扱わない)
- **当日情報 (馬体重、 リアルタイムオッズ) は学習特徴量に入れない** (リーク防止)
- **`単勝オッズ` (今走) は完全リーク**、 EV 補正でのみ使用
- **`前走単勝オッズ` は OK**
- 競馬の控除率は **20-25%** (期待値 -20% スタート)

---

## 9. 参考: 競合手法

- netkeiba.com の予想 AI: 公開情報なし
- 各種 SaaS 予想サイト (Speed 指数等): hand-crafted feature engineering 中心
- 学術論文では LSTM/Transformer 系が散見、 ROI 報告は希少

---

**ご教示お願いします。 構造的な打破口や、 我々が見落としている観点を歓迎します。**
