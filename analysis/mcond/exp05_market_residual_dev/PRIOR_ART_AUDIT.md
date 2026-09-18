# EXP05 先行研究監査

実装前にコードを grep・直接精査して確認した。EXP05 と実質的に同じ実験があれば実装せず報告する規則
(spec §2) に従う。結論: **完全に同一のものは無い**が、`wide_residual_shadow.py`(本番稼働中の前向き
shadow)が「市場残差」という語彙・前向き検証インフラの両面で最も近い。この資産は EXP05 の将来の
前向きシャドー (spec §19) の実装基盤として再利用すべきであり、車輪の再発明をしない。

## 比較表

| 項目 | Benter (`stage1_benter_blend.py`) | value_model_v2 (`train_value_model.py`) | DeepValue T-10 (`exp_deepvalue_t10.py`) | EXP04 M2 (`exp04_invariant_info_dev/run.py`) | 本番EV (`compute_bets.py`) | SettleAI (決済層) | wide_residual_shadow (本番shadow) | 旧stacking (`stacking.py`) | PL/isotonic較正器 (`build_pl_calibrators.py`) | **EXP05** |
|---|---|---|---|---|---|---|---|---|---|---|
| 市場確率を入力 | ✅ (π9/πc) | ✅ (オッズ列) | ✅ (o9/oc) | ✅ (f_mkt) | ✅ | — (決済のみ) | ✅ (T-10 wide中点) | — | — | ✅ (pre, 31-40分前) |
| v6生スコアを入力 | △ (旧v5較正確率) | △ (旧8モデルensemble base_prob) | △ (v6較正PL確率) | ✅ (f_v6, offset) | ✅ | — | △ (topdown λ補正PL) | △ (旧4モデル) | ✅ (PLスコア) | ✅ (raw score + rank) |
| v6較正確率を入力 | ✅ | — | ✅ | ✅ (offset) | ✅ | — | ✅ (λ補正PL経由) | — | (これ自体が較正器) | ✅ (train-refit isotonic) |
| 元の表特徴(120特徴由来)を入力 | ✗ | ✗ (オッズ由来特徴のみ) | ✗ | ✅ (145候補) | ✗ | ✗ | ✗ | ✗ (旧モデルのメタのみ) | ✗ | ✅ (F-full/F-serve) |
| 払戻を目的変数 | ✗ (的中確率、ROIは事後測定) | ✅ (pred_roi) | ✅ (直接効用=期待利益) | ✗ | ✗ (EV計算のみ、学習なし) | ✗ (係数は実測ドリフト比) | ✗ (的中確率→フラット100円で事後測定) | ✗ | ✗ | ✗ (的中確率が目的変数、経済評価は事後・固定ルール) |
| 的中確率を目的変数 | ✅ (win MLE) | △ (calibratorのみ) | ✗ | ✅ (top3) | — | — | ✅ (pair top3相当) | ✅ (fukusho系) | ✅ | ✅ (top3主, win副) |
| OOF stacking (train行にin-sample予測を使わない) | ✅ (v6較正値は既存OOF資産流用) | ✗ (Layer1 ensemble予測をそのまま特徴に使用、in-sample/out-sample切り分け未検証) | ✅ (`deepvalue_races_oof.pkl` expanding-window) | ✅ (v6base OOF) | — | — | — (本番配線のλ補正PLをそのまま使用) | △ (Level0はtrain、Level1はvalid=2023でfit、OOFではなく単純split) | — | ✅ (v6base既存OOFをそのまま利用、独自にはtrain-only isotonic再fit) |
| 購入領域の条件付き較正 | ✗ (五分位EVのみ) | ✗ | ✗ | ✗ (Gate2Dで人気帯breakdownのみ) | ✗ | ✗ | ✗ (残差帯で固定閾値0.05のみ) | ✗ | ✗ (全体ECEのみ) | ✅ (spec §13, 8領域+edge十分位単調性) |
| 完全未使用期間で評価 | △ (test=2024-25、既に多数の実験で開封済み) | △ (test2024) | △ (test2024-25、開封規律あり) | △ (2023-25、EXP01-03と同じ再利用OOS) | — | ✅ (前向きT-10ログ) | ✅ (2026-08-29以降、前向きshadow、real_money=false) | △ (test2024〜) | △ | △ (2023-25は再利用済み探索的期間と明記。2026は本監査でロック不可と判定、前向きshadowは今回実装しない) |

## 個別の違い

- **Benter (`stage1_benter_blend.py`)**: レース単位の softmax 合成 `p=softmax(α·log f+β·log π)` を単勝の
  MLE で fit。表特徴は一切使わない。EXP05 の M6 として対照モデルにそのまま採用 (α,β を EXP05 の
  train=2016-2021で独自に再fit。結果: α=0.186, β=0.951 — 元のBenter実験のα=1.008,β=0.32とは値が
  大きく異なるが、対象モデル(v5較正確率 vs v6生PL確率)・市場スナップショット・期間が違うため直接
  比較不可。あくまでEXP05内の対照として独立に fit した)。
- **value_model_v2 (`train_value_model.py`)**: 目的変数が的中確率ではなく ROI (回収率) そのもの。
  Layer1 の旧8モデルアンサンブル base_prob を特徴として使うが、Layer1自身のtrain行に対してOOF化
  されているかは train_value_model.py 内で検証されていない (2026-05当時の資産、現在は本番未使用)。
  v6/市場とは無関係のオッズ列中心。EXP05とは目的変数・入力とも別物。
- **DeepValue T-10 (`exp_deepvalue_t10.py`)**: 意思決定ネット(PyTorch)で「賭ける/見送る」を直接
  効用最大化で学習。的中確率の推定が目的ではなく、値がRSE(ネット確信度)とROIが単調逆相関という
  結論で dead 判定済み ([[project_deepvalue_t10_net_dead]])。入力は o9/oc(オッズ)のみで表特徴を
  持たない。EXP05はまず正しい確率を推定してから固定ルールで経済価値を測る設計であり、目的関数の
  取り方自体が異なる。
- **EXP04 M2**: 「offset(v6+市場, 係数固定1) + 全145候補プール」の残差回帰。EXP05 の仮説の直接の
  出発点(このモデルが2023-2025で一貫してM1を上回った)。EXP04では offset の係数を1に固定していたが、
  EXP05のM1-M3は係数を自由にfitする(情報圧縮診断のため)。EXP05のM5は「offset(M3)+F-full」であり
  EXP04のM2とはoffsetの構成が異なる(EXP04offset=v6+市場固定、EXP05offset=M3の自由fit結果)。
  Gate1で「EXP04と同じ方向の結果が出るか」を再現性チェックとして実施する。
- **本番EV (`compute_bets.py`)**: 学習を一切行わない。v6較正確率とオッズから決定論的にEVを計算する
  ルールベース層。EV順選抜は既に有害と実証済み ([[project_ev_selection_harmful_probfirst]])。
  EXP05はEV選抜を行わず prob-first (R1: edge比>=1.15の均等額、R2: レース最大確率1点)を流用する。
- **SettleAI/決済層**: T-10表示オッズが確定までに縮む(steam)ことによるEV過大計上を補正する層。
  学習モデルではなく実測ドリフト比のテーブル。市場残差ではなく「オッズ表示 vs 確定払戻」のギャップ
  を扱う。ドリフト方向での銘柄選別は両方向エッジゼロと実証済み封印済み ([[project_settlement_layer]])。
  EXP05はこの層と無関係(決済は確定複勝配当のみ使用、ドリフト補正なし)。
- **wide_residual_shadow (本番稼働中)**: 2026-08-29から稼働する前向きshadow。ワイド馬券のペア確率
  (topdown λ補正PL)と市場中点オッズの残差が小さい(<0.05, モデルと市場が一致する)馬を選ぶ設計で、
  「モデルと市場の乖離が大きい方を買う」のではなく「一致する方を買う」という
  [[project_benter_dead_crosspool_oracle]] の教訓(反市場ピックは逆符号)を踏まえた設計になっている。
  表特徴は使わずtopdownのPL確率のみ。**EXP05とは検証対象(表特徴の追加情報 vs ワイド馬券選別)が違う
  別実験だが、前向き検証のインフラ(`forward_price_integration.py`, `data/forward_prices/`,
  `data/shadow_policies/*.json`, formal_look/futility_checkpointのプロトコル)はEXP05が将来
  前向きシャドーに進む場合そのまま再利用すべき**。今回のEXP05では前向きシャドーは実装しない
  (LOCKED_PERIOD_AUDIT.md参照、2026の必要データが揃っていないため)。
- **旧stacking (`stacking.py`)**: Level0(旧4モデル)→Level1(Meta LGBM)。Level1はvalid=2023の1回
  fitであり、EXP05が要求するfold毎独立re-fitのOOF stackingではない。v6/市場は使わず、対象も旧モデル
  体系。現在は本番未使用({{git status上modifiedなし}}、事実上凍結資産)。
- **PL/isotonic較正器 (`build_pl_calibrators.py`)**: `models/pl_calibrators_v6.pkl` は PL勝率→実勝率の
  isotonic回帰そのもの。これ自体がEXP05のB1/M1で使う「calibrated_v6_probability」の生成源だが、
  valid=2023でfitされているため2023年の評価に使うとリークする。EXP05はDATA_AUDIT.mdに記載の通り
  train(2016-2021)のみで自前isotonicを再fitして置き換えた。

## 判定

EXP05はEXP01-04・Benter・value_model_v2・DeepValue・EXP04 M2・本番EV・SettleAI・
旧stackingのいずれとも実質的に同一ではない。表特徴の追加情報を「情報圧縮の分離」→
「offset残差の第2段モデル」→「購入領域の条件付き較正」→「固定ルールでの経済評価」という順で
検証する構成は本コードベースに前例がない。**実装を進める。**
