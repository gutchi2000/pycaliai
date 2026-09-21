# EXP14 Stage 0 — 最小反証実験の設計案（未実装）

**作成日**: 2026-09-22　**状態**: 設計・実装可能性確認のみ。**本Stage0では
実行しない**（実装・学習・バックテスト・2024/2025年性能開封・ROI評価は
一切行っていない）。

---

## 主仮説

> 単一pooled modelでは平均化される異質な予測関係があり、事前情報だけで
> 選択した専門家モデルが、同じ特徴・同程度の容量を持つpooled modelより
> 未使用期間で良い。

**`PRIOR_ART_AUDIT.md`の通り、この仮説の一部（手動条件分割による専門家
モデル）は既に3件の独立実験で否定されている**。EXP14がStage1へ進む場合、
以下のいずれかで新規性を確保する必要がある:
(a) 未踏査のregime軸（venue単独・クラス・年齢・組合せ）で再検証する、
(b) 手動分割以外のgating機構（既存3実験が試していない設計）を用いる。

## 手法の区別（ユーザー指定7方式、実装可能性のみ）

| 方式 | 概要 | 既存実装 | 実装可能性 |
|---|---|---|---|
| 手動条件分割 | 条件でハードフィルタし別々に学習 | `train_expert.py`・`exp_surface_split.py`・summer-specific（3件とも既存、失敗済み） | 実装済み・追加検証は新規regime軸限定で可能 |
| 教師なしクラスタリング | 特徴空間（過去成績統計等）をKMeans等でクラスタ化しregime定義 | なし | 実装可能（`sklearn.cluster`等既存依存関係で対応可、要新規実装） |
| decision tree gate | 浅い決定木でregimeを予測しルーティング | なし | 実装可能（LightGBM/sklearn既存で対応可） |
| soft gating network | 各regimeモデルの出力を学習可能な重みで混合 | なし | 実装可能だが最も実装コストが高い（別途小規模NN or ロジスティック回帰の学習が必要） |
| mixture of logistic experts | 各expertがロジスティック回帰、gateも学習 | なし | 実装可能（既存の`statmech`系実装等を参考にできる） |
| mixture of LightGBM experts | 各expertがLightGBM、hard/soft gate併用 | `train_expert.py`が手動版のプロトタイプ（ただしgateは学習しない固定ルール） | 部分的に実装済み資産あり（regime定義・学習ループの再利用可） |
| regime-switching time-series model | 時系列上のregime遷移を確率モデル化（HMM等） | なし | **実現可能性に強い留保**——本データは「レース単位の断面」であり連続時系列ではない（同一馬の出走間隔は不規則、レース間の時間的連続性が弱い）。時系列regime-switchingの前提（連続する観測系列上のregime遷移）が成立するか自体が疑わしい |

## 比較baseline（ユーザー指定5種、必須）

Stage1へ進む場合、以下5種類を**全て**含めること（一部のみの比較は禁止）:

1. **pooled v6＋市場**: 本番相当のpooled model（`unified_rank_v6.pkl`）に
   市場確率を加えたもの。**ただし`DATA_AUDIT.md`§9の通り、2023年development
   では市場確率が構築不能（EXP13 Gate0Aの制約を継承）**。この比較を含める
   場合、市場確率抜きの変種を主とし、市場確率込みは別途注記する必要がある。
2. **条件特徴を追加した単一モデル**: v6の120特徴に、現状使われていない
   条件×条件の明示的交互作用項（例: 芝ダ×距離のカテゴリ結合、クラス×年齢
   等）を追加した単一pooled model。「木が既に交互作用を捉えている」という
   `DATA_AUDIT.md`§4の仮説を、明示的な交互作用特徴の追加で検証できる。
3. **regime別モデル**: 上記7方式のいずれか（Stage1で選定）。
4. **同程度の総パラメータ数を持つpooled model**: [[project_summer_specific_
   model_dead]]の`M_allsub`（regimeと同データ量のプールサブセット）と同型の
   設計。LightGBMの場合「パラメータ数」は木の本数×葉数で近似する。
5. **単純な手動分割**: `PRIOR_ART_AUDIT.md`で既に実施済みの`exp_surface_
   split.py`型設計（新規regime軸で再現）。

**5種全てを含める理由**: 2のみでregime分離の効果を「明示的特徴」に帰属
できるか、3のみでgating機構の効果を検証できるか、4のみでデータ量統制の
効果を検証できるか、をそれぞれ独立に切り分けるため。1つでも欠けると、
「本当にregime分離が効いたのか、単に特徴が増えた／データが減った／
モデル容量が変わっただけなのか」を判別できない。

---

## Gate構造（EXP06-13から継承）

- **Gate0**: 本Stage0の内容（データ・時点・母集団の健全性）
- **Gate1**: ラベル・regime定義の構造的妥当性（regime定義が年度間で安定、
  人為的な事後調整でない）
- **Gate2**: 単純方式（baseline 1・2）との比較
- **Gate3（最重要）**: baseline 4（同程度パラメータ数pooled）との比較——
  **これが主仮説の直接検証**。ここを通らなければregime分離に意味がない
  ことを意味する（[[project_summer_specific_model_dead]]のM_allsubパターン）
- **Gate4**: 年度・regime間での安定性
- **Gate5（経済評価）**: Gate1-4 PASS後のみ

**統計単位**: meeting-day（EXP06-13から継承、同一レース内の馬を独立標本
としてbootstrapしない）。

**development/OOS分離**: 2023年development固定→2024/2025年は本Stage0
含め未開封。**ただし`DATA_AUDIT.md`§10の通り、芝・ダート軸は既に
`exp_surface_split.py`により2024-2025年が消費済み**。Stage1で芝・ダート軸を
使う場合、これは「新規OOS検証」ではなく「既知の結果の追試」として扱う
（新規性の主張はしない）。

## 中止条件

- Gate3（同程度パラメータ数pooledとの比較）でFAILした場合、統制を緩めた
  再検定は行わない。
- regimeの標本数が不足する場合（`DATA_AUDIT.md`§8、上位クラス・一部
  距離帯・組合せregime）、対象regimeを主解析から除外し、それを理由に
  regime定義を事後的に広げない。
- 手動分割以外の6手法いずれも、`PRIOR_ART_AUDIT.md`の3件の収束的な失敗
  パターンと同型の結果（プールに対し有意差なし）になった場合、それ以上の
  手法バリエーション探索（例: ハイパラ再チューニングの繰り返し）は行わない
  （[[project_summer_specific_model_dead]]の「24grid再チューニングでも
  並ぶだけ」という前例を踏まえる）。

関連: [[project_summer_specific_model_dead]] [[project_exp10_downside_risk]]
[[project_research_stopline_20260921]]
