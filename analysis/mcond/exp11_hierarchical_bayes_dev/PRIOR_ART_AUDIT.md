# EXP11 Stage 0 — 先行研究監査（階層ベイズによる疎データ・未知条件への外挿）

**作成日**: 2026-09-21　**状態**: Stage 0 監査のみ。実装・学習・バックテストなし。
2024・2025年の性能・ROIは未開封。

## 対象チェックリスト項目

- 項目1: hierarchical Bayes / partial pooling / mixed effects / random effects /
  empirical Bayes / shrinkageに相当する既存研究
- 項目7: EXP01の調教師選択、EXP02の出走回数・休養日数との重複
- 項目8（一部）: LightGBMのカテゴリ処理と階層縮約が数理的に何を追加するか

---

## 1. 「経験ベイズ収縮」という数学的技法自体は、既に複数箇所で実装・検定・死亡している

`grep`で`階層ベイズ|hierarchical|partial pooling|mixed effect|random effect|
empirical bayes|shrinkage|収縮`をリポジトリ全体（*.py）に対して実行した結果、
以下3件の**本物の経験ベイズ収縮実装**が見つかった（コード確認済み）:

### 1.1 `lab/features_dead/build_evt_feats.py`（個体分散のEVT特徴、死亡確定）

`prev_hosei`（補正タイム指数）の馬ごとas-of分散`σ_i`を、古典的な経験ベイズ式
`σ_s = w·σ_h + (1-w)·prior, w = n/(n+k)`で全体priorへ収縮する実装
（`build_evt_feats.py:148`、`--shrink-k`で収縮強度kを可変）。

**検定結果（[[project_evt_extreme_value_tested]]、3つの独立手法で収束的に反証済み）**:
GBM ablationで複勝ΔAUC+0.0017・単勝+0.0029（ノイズ帯）、直交part_corr<0.01で
全死亡帯、理論と逆符号のH2反証。**厳密N体不等分散Gumbel argmax注入
（evt_phase2.py）ではσ注入がγ↑で単調悪化**——等分散softmax（現行PL）が
厳密に最良という強い反証。`lab/features_dead/`（死亡機能ディレクトリ）に
格納済み。

### 1.2 `baba_eval.py`（馬場適性の経験ベイズ収縮、死亡確定）

`soft_rate = (_st3 + SHRINK_K*base) / (_s + SHRINK_K)`（`baba_eval.py:100-101`、
`SHRINK_K=6.0`固定）——古典的なBeta-Binomial事後平均式による、馬ごとの
軟馬場複勝率の全体base rateへの収縮。

**検定結果（[[project_baba_cushion_tested]]、KILL確定）**: ランキングΔAUC
-0.00025、頭数・エントロピー統制後-0.00195（交絡）。「馬場状態は
course_top3_rate・pace_pressure・各馬履歴に内包済み」と結論。

### 1.3 `analysis/mcond/exp01_choice_dev/build_features.py`（陣営選択の逸脱、Gate2 FAIL）

`K_TRAINER=30.0`, `K_JOCKEY=50.0`, `K_HORSE=3.0`という**固定強度**の経験ベイズ
収縮定数（`build_features.py:45-47`）。`jq`（騎手quality）は
`jq = (prior_top3 + K_JOCKEY*global_prior_rate)/(prior_one + K_JOCKEY)`
（`build_features.py:129-136`）で全体平均へ収縮。`shrunk_mean(col)`は
**調教師**の行動分布（間隔・距離変更・競馬場変更等）を全調教師平均へ同型の式で
収縮（`build_features.py:170-186`）。馬レベルも`K_HORSE=3.0`で同様
（`build_features.py:188-196`）。

**検定結果**: 「逸脱固有の価値」Gate2 FAIL（M4 vs M2のCIがゼロを跨ぎ、
上位20調教師除外で符号反転、`analysis/mcond/exp01_choice_dev/REPORT.md:58-61`）。

### 1.4 EXP02のWeng-Lin PL（馬レベルのみ、条件間の部分プーリング、死亡）

`analysis/mcond/exp02_dynamic_skill_dev/dyn_skill.py`は真にベイズ的な逐次
フィルタ（Weng-Lin PL、OpenSkill型）だが、**馬レベルのみ**（`State`クラスは
`h`=血統登録番号のみ追跡、`dyn_skill.py:101-113`、騎手・調教師の辞書は一切
存在しない）。T2機構（`dyn_skill.py:121-131,206-220`）は同一馬内の
**条件（芝ダ×距離帯）間の部分プーリング**（Kalman型分散加重更新）であり、
騎手・調教師という別カテゴリをまたぐプーリングではない。

**検定結果**: T2固有の追加価値なし（M5 vs M3 FAIL、`REPORT.md:80,85`）。
動的能力の効果の3/4は出走回数・休養日数という生の事実由来
（[[project_mcond_exp02_dynamic_skill]]）。

### 1.5 重要な区別: 「収縮という技法」の死亡と「EXP11の実際の仮説」は別問題

上記4件はいずれも**経験ベイズ収縮を"特徴量"としてv6へ追加する**話であり、
「v6全体の予測精度を上げられるか」という土俵で検定され、いずれも死亡した。

これに対しEXP11の仮説は:
> 「**全体精度では差がなくても**、疎データ領域に限れば、階層縮約が現在の
> unknown処理・頻度統計・LightGBMより**安定した確率**を出せる」

という、(a) 全体平均ではなく**疎データ領域限定**、(b) 精度ではなく**確率の
安定性/較正**、という2点で明確に異なる主張である。上記4件のいずれも
「疎データ領域限定でのGate」を実施していない（EXP01のGate2は全調教師の
逸脱を対象、EXP02のT2は全馬全条件を対象、EVT/babaのablationも全体population
対象）。**「疎データ領域限定・確率の安定性」という土俵は真に未検証**と
判定する。

ただし、上記4件全てで「固定強度kの経験ベイズ収縮」という**同一の数学的
機構**が繰り返しテストされ死亡していることは、EXP11のM2（empirical Bayes
収縮）が新規に実装する際の技術的土台としては既に確立している一方、
「同じ機構をまた追加しても全体では効かない」という強い事前情報として
警戒すべきである（中止規律の設計に反映する）。

---

## 2. LightGBMのカテゴリ処理と階層縮約が数理的に何を追加するか（項目8前半）

コードベース精査の結果（`train_unified_rank.py`・`optuna_v6_marks.py`の
両方を確認、後者が実際のunified_rank_v6.pklの生成元と確認済み）:

**現状は「LightGBMのネイティブカテゴリ分割」すら使っていない**。
`categorical_feature`パラメータは`lgb.Dataset`/`lgb.train`のどちらにも
一切渡されておらず（grep確認: `train_unified_rank.py`・`optuna_v6_marks.py`・
`export_marks_json.py`・`predict_weekly.py`のいずれにも
`categorical_feature`・`astype("category")`・`pd.Categorical`が0件）、
種牡馬・母父馬・馬主・生産者・騎手コード・調教師コードは全て**単純な
LabelEncoder（順序符号、target encodingでもfrequency encodingでもない）**で
整数化された後、**通常の数値特徴としてLightGBMへ渡される**
（`train_unified_rank.py:127`・`optuna_v6_marks.py:150`、
`df[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values`）。

これは実務上よく見られるが理論的には劣った手法（任意の順序を持つ整数への
閾値分割）であり、LightGBM自身が持つFisher最適分割（カテゴリのサブセットを
target勾配でソートして二分探索する、より情報を活かす分割法）すら
活用していない。疎カテゴリに対する正則化は**汎用ハイパーパラメータ
`min_data_in_leaf`のみ**（unified_rank_v6は`min_data_in_leaf=197`で確定、
Optunaで`[20,200]`から選択）——カテゴリ固有の仕組みは一切ない。

**数理的に階層ベイズが追加しうるもの**:
1. **LightGBMネイティブカテゴリ分割との比較でも追加余地がある**（現状は
   それすら使っていない、単純な順序符号への閾値分割のみ）。
2. **不確実性の定量化**: 経験ベイズ/階層ベイズは「このカテゴリの自己統計を
   どれだけ信頼するか」を分散比（群内分散 vs 群間分散）から明示的に導出する
   （w=n/(n+k)や完全事後分散）。LightGBMの`min_data_in_leaf`は汎用的な
   葉サイズ制約に過ぎず、カテゴリごとの「どれだけ疎なら信頼度を下げるべきか」
   という情報を明示的には持たない。
3. **階層構造の明示的活用**: 「この騎手は不明でもこの調教師の平均へ、この
   調教師も不明なら地区平均へ、それも不明なら全体平均へ」という多段の
   部分プーリングは、単純な`"__NaN__"`一括バケット（現状、項目3参照）や
   単一レベルの経験ベイズ収縮（EXP01で既に死亡確認済み）では表現できない。

ただし1.5節の警告の通り、「理論的に追加できる」ことと「実際に予測精度/
確率の安定性を改善する」ことは別問題であり、EXP11自身がこれをfull-control
比較で検証する必要がある。

---

関連: [[project_evt_extreme_value_tested]] [[project_baba_cushion_tested]]
[[project_mcond_exp01_choice_dev]] [[project_mcond_exp02_dynamic_skill]]
[[project_exp10_downside_risk]]
