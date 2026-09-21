# EXP14 — レジーム分離・Mixture of Experts

## 状態: Stage 0（監査）完了、実装前にユーザー報告・承認待ち

2026-09-22、EXP13のGate0終了後、ユーザーの明示指示によりStage 0監査を
実施した。**本Stage0では実装・学習・バックテスト・2024/2025年性能開封・
ROI評価を一切行っていない**（既存の過去実験結果の引用のみ、新規の
2024/2025年開封は行っていない）。

## 目的・主仮説

> 単一pooled modelでは平均化される異質な予測関係があり、事前情報だけで
> 選択した専門家モデルが、同じ特徴・同程度の容量を持つpooled modelより
> 未使用期間で良い。

## ★Stage0の中心的発見: 手動条件分割は既に3件で否定済み

| 実験 | regime軸 | 結果 |
|---|---|---|
| `train_expert.py`（旧8アンサンブル） | 距離帯（芝短/中/長・ダート） | 4分の3失敗（1件僅差採用のみ） |
| `analysis/exp_surface_split.py`（v6世代） | 芝・ダート | プールが両方で僅差優位（有意差なし） |
| [[project_summer_specific_model_dead]]（v6世代、最も厳密） | 季節×地域 | 完全否定（複数の緩和策も無効、「data量>regime純度」原則を確立） |

**この3件は既存の成果物であり、本セッションで新規実行はしていない**
（`exp_surface_split.json`・`expert_metrics.json`・summer-specificの
検証結果を読んだのみ）。詳細は`PRIOR_ART_AUDIT.md`参照。

## v6の条件特徴interaction capacity（新規実測）

v6（515本・深さ12のLightGBM）の条件特徴（場所・芝ダ・距離・クラス・年齢
等）は、個々のgainは低いが**split回数は192〜261回と多い**——tree
ensembleが既に条件×他特徴の交互作用を内部で大量に捉えている可能性が高い。
これが手動分割が軒並み失敗する一因かもしれない（仮説）。詳細は
`DATA_AUDIT.md`§4参照。

## EXP14が新規性を持ちうる範囲

- **未踏査のregime軸**: venue単独・クラス単独・年齢単独・新馬/未勝利
  専用・重馬場専用（`PRIOR_ART_AUDIT.md`§2-3参照）
- **未踏査のgating手法**: 教師なしクラスタリング・decision tree gate・
  soft gating network・mixture of logistic/LightGBM experts
  （手動条件分割以外、`MINIMAL_FALSIFICATION_PLAN.md`参照）
- **ただし芝・ダート軸は`exp_surface_split.py`が既に2024-2025年を消費
  済み**——この軸を再度使う場合は新規OOS検証と呼べない（`DATA_AUDIT.md`
  §10）

## 標本数（2023年development、実測）

単軸regime（芝ダ・venue・距離帯・クラス・年齢・季節）は概ね数千行規模で
確保できる（EXP13のDNF陽性145件のような希少性問題ではない）。ただし
上位クラス（G1等、数百行）・一部距離帯（2200-2400m、948行）・組合せ
regimeは薄い。詳細は`DATA_AUDIT.md`§8参照。

## gating構築の時点安全性

regime定義自体（芝ダ・距離・場所・クラス・年齢・季節）は全て発走前確定
情報で構築可能、結果・確定オッズ不要。ただし比較baselineに「市場確率込み
pooled model」を含める場合、**EXP13 Gate0Aで確定した制約（2023年に
historical_pre_snapshotが存在しない）をEXP14も継承する**。詳細は
`DATA_AUDIT.md`§9参照。

## 比較baseline（Stage1着手時に必須、5種）

pooled v6＋市場／条件特徴を追加した単一モデル／regime別モデル／同程度の
総パラメータ数を持つpooled model／単純な手動分割。詳細・理由は
`MINIMAL_FALSIFICATION_PLAN.md`参照。

## Stage 0 成果物

| ファイル | 内容 |
|---|---|
| `PRIOR_ART_AUDIT.md` | 既存研究監査（mixture of experts/regime/cluster等）、3件の収束的な手動分割失敗、未踏査領域の特定 |
| `DATA_AUDIT.md` | v6の条件interaction実測、年度regime drift、比較基盤、標本数、gating時点安全性、2023固定/2024-25評価の実現可能性と既消費部分 |
| `MINIMAL_FALSIFICATION_PLAN.md` | 7方式の区別・5種比較baseline・Gate構造・中止条件（設計のみ、未実装） |

## 絶対条件

- EXP01-13・v6・compute_bets.pyは変更しない。新規作業は
  `analysis/mcond/exp14_regime_moe_dev/`へ限定する。
- 2024・2025年は本Stage0を含め新規に開封しない（既存成果物の引用のみ許可）。
- 手動条件分割の追試を「新規性」として主張しない（既に3件で否定済み）。
- 5種の比較baselineを全て含めない部分的な比較は行わない。

## 次の一手

**実装はまだ一切行っていない**。本Stage0の報告後、ユーザー承認を経てから
Stage1（未踏査regime軸または非手動gating手法の選定・実装）へ着手する。

関連: [[project_summer_specific_model_dead]] [[project_exp13_nonfinish_risk]]
[[project_research_stopline_20260921]] [[project_v6_pastform_dominance]]
