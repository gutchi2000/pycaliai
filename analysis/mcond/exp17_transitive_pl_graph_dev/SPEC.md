# EXP17 仕様書 — 共通対戦馬を介した動的Hodge–Plackett–Luce

**版:** 0.1-draft
**作成日:** 2026-09-24
**状態:** レビュー前・未凍結
**本番影響:** なし

## 1. 目的

次の仮説を、既に終了したElo・Glicko・EXP02・EXP12の言い換えにせず検証する。

> 直接対戦したことがない2頭でも、双方が過去に対戦した共通対戦馬を介せば、その2頭に固有の相対能力情報を時点安全に推定できる。この2-hop証拠を現在の出走馬全組合せについて保持すれば、馬ごとの単一能力値、R0-clean、terminal close単勝市場を超える情報が得られる。

問いを二つに分離する。

1. **機構の問い:** 共通対戦馬の証拠は、未対戦2頭のどちらが先着するかを予測するか。
2. **実務情報の問い:** そのペア情報から作るレース確率は、既存動的能力・R0-cleanを統制した後にもterminal close単勝市場を0.005 nats/race以上改善するか。

機構が通っても市場価値を意味しない。terminal close市場を通らなければ、候補馬券や配分へ進まない。

## 2. 先行研究との境界

### 2.1 実施済みで再実験しないもの

- 静的・逐次Elo（多頭数同時更新を含む）
- Glicko-2のペア分解と不確実性
- EXP02の動的ベイズPlackett–Luce能力と条件別部分プーリング
- EXP12の1-hop対戦相手identity集約（degree・年齢・経験量placeboでFAIL）
- field-strength平均、レースレベル等のスカラー集約

### 2.2 等価性の罠

各馬に単一の`theta_h(t)`だけを推定し、

`P(i > j) = exp(theta_i) / (exp(theta_i) + exp(theta_j))`

とするなら、共通対戦相手による推移比較はEXP02の能力値に既に内包される。これをグラフと呼び直して再実装することを禁止する。

### 2.3 EXP17でのみ検証する部分

対象ペア`(i,j)`ごとに、共通対戦馬のidentityと履歴から`d_ij(t)`を作る。馬ごとの平均特徴へ先に潰さず、現在レースのペア行列として保持した後、Hodge射影で整合する馬スコアへ変換する。主実験は2-hopまでであり、PageRank・embedding・GNNではない。

### 2.4 FAILの範囲

FAIL時の正式文は次に限定する。

> 事前固定した時点安全な2-hop共通対戦馬表現は、対象母集団・期間・統制の下で必要なペア固有情報を追加しなかった。

PageRank、3-hop以上、embedding、GNN、地方・海外履歴、新規データ、他券種、すべての関係モデルが失敗したとは書かない。

## 3. 母集団とID

### 3.1 馬ID

- 主キーは`血統登録番号`を文字列正規化して使用する。
- 馬名joinは禁止する。
- 欠損・不正IDは件数を記録して除外し、馬名から補完しない。

### 3.2 正式race set

- JRA平地のみ。
- `トラックコード(JV)` 51〜59を除外。
- 公式勝馬が一意。
- 有効starterが5頭以上。
- 取消・除外は除く。
- full-starter scoring契約が無いため、正式race-level GateではDNFを含むレースを除く。
- 同着した2頭間のpair labelは作らない。他順位馬との比較は公式順位が一意なら利用可能。

DNF除外は結果条件付きの暫定処理である。障害、取消・除外、DNF、同着、ID欠損、最終eligible数を年別に報告する。

### 3.3 履歴グラフ

- 対象日の開始時点より前のレースだけを使用する。
- 同日の全レースは同じday-start snapshotを使う。
- 当日結果は最終レース終了後に一括反映する。
- 同日先行レースの結果も使わない保守的設計とする。
- 主モデルの履歴はJRA平地に限定する。
- edge方向は公式最終着順だけで作り、オッズ・払戻・ROIを使わない。

## 4. 数理定義

### 4.1 過去の2頭間証拠

対象日`t`より前に馬`h`と共通対戦候補`c`が走った履歴から、時間減衰付き有効回数を作る。

- `W_hc(t)`: `h`が`c`より先着した有効回数
- `L_hc(t)`: `c`が`h`より先着した有効回数
- 同着は勝ちにも負けにも加えない

対称Beta-Binomial縮約を一次仕様とする。

`p_hc = (W_hc + alpha) / (W_hc + L_hc + 2 alpha)`

`ell_hc = logit(clip(p_hc, eps, 1-eps))`

`alpha`、`eps`、lookback、半減期、条件一致重み、weight capは、2019〜2023の結果を見る前にStage 0で凍結する。探索可能な格子は`spec.json`に限定する。

### 4.2 共通対戦馬を介した間接比較

現在レースの未対戦ペア`(i,j)`と共通対戦馬`c`について、

`d_ij^(c)(t) = ell_ic(t) - ell_jc(t)`

とする。複数の`c`がある場合は、信頼度・新しさ・条件一致度だけから事前固定した`omega_ijc`で加重平均する。

`d_ij(t) = sum_c omega_ijc d_ij^(c) / sum_c omega_ijc`

必須条件:

- `d_ji = -d_ij`
- 共通対戦証拠が無いpairはmissingのままにし、観測値0として捏造しない
- 主機構検定は`i`と`j`の直接対戦歴が無いpairだけ
- `(i,c)`と`(j,c)`の両履歴が対象日より前に存在すること
- 直接対戦ありpairは別表で報告し、主機構検定に混ぜない

### 4.3 現在レース内のHodge射影

現在の出走馬について、観測されたpairだけを使い、

`min_s sum_(i<j) w_ij (s_i - s_j - d_ij)^2 + lambda sum_i s_i^2`

を解く。各連結成分内で`sum_i s_i = 0`とする。

- `lambda`は結果開封前に凍結。
- 非連結成分は独立してzero-centerする。
- 証拠が無い馬はscore=0とし、`graph_uncovered=true`を必ず付ける。
- component数・被覆率を結果の代理として黙って利用しない。

確率は、

`p_graph(i) = exp(s_i/tau) / sum_j exp(s_j/tau)`

とする。`tau`はrolling評価の各年について前年以前だけでfitする。pair行列は監査用に保存し、確率armには射影後scoreだけを入れる。

### 4.4 Plackett–Luceの位置づけ

過去レースは多頭数の順序結果を与え、最終確率は整合するscoreのPL softmaxで作る。新規性はPL自体ではなく、単一能力値が捨てる可能性のあるpair固有の共通対戦経路にある。

## 5. Stage 0 — 必須監査

Stage 0成果物をcommitするまで、2019〜2023の成績指標を開封しない。

### 5.1 数理・実測等価性

`PRIOR_ART_EQUIVALENCE_AUDIT.md`で、Elo多頭数版、Glicko、EXP02 T1/T2、EXP12 O3、既存scratchコードと式単位で比較する。

次のいずれかならE0 FAILで終了する。

- 一つのスカラー能力値へ代数的に還元される。
- labelを使わない2022 snapshotで、新scoreがEXP02能力のaffine変換になる。
- Hodge射影前後に対象pair固有の状態が残らない。

### 5.2 時点安全性の必須テスト

1. `t`より後の全レースを削除しても`t`以前の出力がbit一致。
2. 対象レースの結果を書き換えてもpre-race特徴が不変。
3. 同日全レースが同じday-start snapshotを使う。
4. 共通対戦馬の対象日後のレースが`d_ij(t)`へ入らない。
5. ID衝突と馬名補完はfail-closed。
6. antisymmetry誤差が`1e-12`以下。
7. 行順・馬順を入れ替えてもHodge解が不変。
8. `sum(p_graph)=1`の誤差が`1e-12`以下。
9. 非連結成分とuncovered馬が凍結fallbackに従う。
10. 主機構sampleで直接対戦証拠と間接証拠が混ざらない。

### 5.3 被覆率監査

年・頭数帯・芝ダート・年齢・出走数帯別に、全unordered pair、直接対戦歴ありpair、未対戦pair、未対戦かつ共通対戦馬ありpair、有効共通対戦馬数、graph-covered馬・レース、component数、最大component比率、edge age、条件一致度、新馬・低履歴馬の被覆を測る。

レビュー前の暫定続行床:

- 未対戦pairの50%以上に共通対戦証拠がある。
- 80%以上のレースで全pairの50%以上がcovered。
- 新馬以外のstarterの90%以上が履歴graphに属する。
- 主要な芝ダート・頭数帯の被覆が全体の半分未満へ落ちない。

被覆不足なら、成績を見た後に床を下げない。停止または新しいデータ源を別版で設計する。

### 5.4 検出力監査

- 推論単位はmeeting-day。
- 同一レース内pairを独立標本として扱わない。
- labelを使わないrace構造へのeffect injectionで検出力を測る。
- 機構効果はscalar dynamic PLに対するpairwise logloss差。
- race-level実務床はEXP16Aから0.005 nats/raceを継承。
- 機構Gateの科学的floorはStage 0で固定し、2019〜2023結果開封後に変更しない。
- simulation回数、seed、成功回数、power、Wilson CIを保存する。

凍結効果に対するpowerが80%以上で、Wilson下限が80%を大きく否定しない場合のみ進む。

## 6. Rolling評価

### 6.1 期間

- 利用可能な最古年からgraphをwarm-up。
- retrospective rolling-crossfit developmentは2019〜2023。
- 2024/2025は封印。

評価年`Y`について、graphは対象日より前だけを使う。ハイパーパラメータ、較正、`tau`、subset境界、結合係数はすべて`Y`より前だけで決める。2022で決めた値を2019〜2021へ遡及適用しない。

pooled CIはyear-stratified meeting-day bootstrap。各年、pooled、leave-one-year-out、seed別を報告する。この期間はretrospective crossfitであり、完全な未使用holdoutではない。通過しても2024/2025の自動開封を許可しない。

### 6.2 seed

- 学習を含む確率要素は5 seed。
- 主報告はmedian seed。
- 5 seed中4以上で方向一致を要求。
- graph構築とHodge solveは決定論的でseed間完全一致を要求。

## 7. 比較arm

全armでrace setと出走馬集合を一致させる。

| arm | 定義 | 目的 |
|---|---|---|
| `M0_CLOSE` | de-vig terminal close単勝市場 | 締切市場基準 |
| `M1_DYNPL` | EXP02相当のscalar dynamic PL | 既存推移比較基準 |
| `M2_R0` | rolling OOF R0-clean 111列 | clean表モデル基準 |
| `M3_BASE` | close offset + M1 + M2 + graph構造・交絡control | 新path値を含まない完全対照 |
| `M4_PATH` | close offset + common-opponent射影score | path単独診断 |
| `M5_FULL` | M3 + path score + 事前固定したpath不確実性 | race-level主arm |
| `P_PLACEBO` | degree・年齢・経験量を保ってidentityをrewireしたM5 | graph identity placebo |
| `PRE_FULL` | historical_pre_snapshot offset + M5の非市場項 | 情報時点診断 |

`M3_BASE`には最低限、出走数、休養日数、horse degree、pair/common-opponent被覆、component size、missing flag、EXP02能力平均・不確実性、R0-clean確率、市場entropy、頭数を入れる。

主race比較は`M5_FULL - M3_BASE`。

主機構比較は、直接未対戦かつ共通対戦馬ありpairについて、path pair確率と`M1_DYNPL` pair確率を比較する。terminal close pair確率を統制した比較も併記する。

## 8. Placeboと負の対照

### 8.1 degree-preserving rewire

年齢帯、出走数帯、芝ダート、時期、node degree、edge数・年齢分布、頭数分布を保って、対戦相手identityをセル内でrewireする。97.5 percentileを安定して測れるよう200 draw以上とする。

### 8.2 direction-null

edge endpointと被覆を保ったまま、許可されたpairの方向をランダム反転する。

### 8.3 identity-free control

path値を使わずdegree、経験量、新しさ、component sizeだけで構成する。

real改善は、事前登録した全placeboの97.5 percentileを超えなければならない。超えなければ共通対戦馬identityではなくgraph構造・経験量で説明可能と判定する。

## 9. 指標とGate

### 9.1 G1 — 機構Gate

対象は、現在同走しているが過去に直接対戦せず、共通対戦馬を持つpair。

主指標はbinary pairwise logloss。同一race内で平均してからmeeting-day単位で推論する。

全条件:

- `PATH - DYNPL`のCI95上限<0。
- 5年中4年以上で改善。
- 5 seed中4以上で改善。
- leave-one-year-out全てでCI95上限<0。
- real効果が全placebo 97.5 percentileを超える。
- Stage 0で凍結した機構floor以上。

FAIL時は終了し、PageRank・長いpath・embedding・GNNを救済実装しない。

### 9.2 G2 — terminal close残差Gate

G1通過時のみ実行する。主指標はrace-level categorical winner loglossの`M5_FULL - M3_BASE`。

EXP16Aと同じ等級:

- `PASS-PRACTICAL`: CI95上限<−0.005かつ全安定性・placebo条件を満たす。
- `PASS-SIGNAL`: CI95上限<0だが−0.005未満ではなく、全安定性・placebo条件を満たす。
- `FAIL`: CIが0を含む、またはいずれかの安定性・placebo条件が不成立。

追加条件は5年中4年、5 seed中4、全leave-one-year-outでCI上限<0、単一日・単一年依存なし。

`PASS-PRACTICAL`だけが、別途レビュー・凍結する実務Stageを提案できる。`PASS-SIGNAL`は記録して終了し、候補生成へ進まない。

### 9.3 G3 — 情報時点診断

`PRE_FULL`は凍結したrun条件を満たす場合だけ評価する。G2を救済できない。pre市場には効くがterminal closeに効かない場合、後の市場反映と整合する時間情報として扱い、取引可能な価値とは呼ばない。

### 9.4 副指標

- multiclass Brier
- calibration intercept/slope、adaptive-bin ECE
- top1、top3、NDCG@3
- pairwise AUC・accuracy（記述のみ）
- 頭数、芝ダート、年齢、出走数、人気帯別の被覆と誤差
- 直接対戦ありpairと間接のみpairの分離

副指標で主Gateを救済しない。

## 10. 停止規律

次のいずれかで停止する。

1. EXP02 scalar abilityと代数的・実測上等価。
2. 被覆が凍結床未満。
3. 未来edgeまたは削除不変性違反。
4. 機構floorへのpowerが80%未満。
5. G1 FAIL。
6. G2がFAILまたはPASS-SIGNAL。
7. placeboがreal改善を説明。
8. degree、出走数、休養、missingnessで改善が消える。
9. 一年・小標本区分だけで結果が成立。

停止後にlookback、条件セル、母集団を結果に合わせて変えたり、PageRank、長いpath、embedding、GNNを救済実装したりしない。必要なら新しい実験番号と未使用評価計画を作る。

## 11. Stage別の許可範囲

### 凍結前に許可

- read-only先行研究監査
- labelを見ない被覆・連結性監査
- 合成テスト
- power simulation
- 計算量・容量dry-run

### 凍結前に禁止

- 2019〜2023の成績評価
- 2024/2025成績へのアクセス
- ROI・払戻分析
- production変更
- 候補馬券生成

G2がPASS-PRACTICALでも、独立レビュー済みの別仕様を凍結するまで経済評価は禁止する。

## 12. 必須成果物

Stage 0:

- `PRIOR_ART_EQUIVALENCE_AUDIT.md`
- `GRAPH_DATA_AUDIT.md`
- `GRAPH_COVERAGE.json`
- `POWER_AUDIT.md`と機械可読結果
- `COMPUTE_DRY_RUN.json`
- 合成invariant tests
- 最終版`spec.json`

Stage 1以降:

- source hash付きgraph snapshot manifest
- 年・seed別model/prediction hash
- deletion-invariance report
- pair matrix監査sample
- 単一のgrading関数が出すGate JSON
- 範囲限定した最終REPORT

## 13. 凍結規則

レビュー後に版番号とcommit hashを記録する。以後の変更は、結果開封前に新版番号、変更理由、影響範囲をcommitする。結果開封後はGate、閾値、母集団、arm、placebo、停止規律を変更しない。
