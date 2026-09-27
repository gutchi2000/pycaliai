# EXP20 仕様案 — 騎手配置の制約付きマッチング残差

**版**: v0.1-review-draft（2026-09-27）
**状態**: 未凍結。Fableレビュー前
**禁止**: レビュー・Stage 0完了前の結果評価、2024/2025開封、ROI、候補生成、賭金配分、production変更

## 0. 目的と結論範囲

同一レースに実際に出走する馬集合と騎乗する騎手集合の一対一対応を、制約付き二部マッチングとして扱う。馬・騎手の既知能力、騎手継続、調教師と騎手の過去関係、斤量実現可能性で説明した後にも、実際に成立した組合せ固有の情報が残るかを検定する。

本実験が観測できるのは、最終的に成立した騎乗配置だけである。依頼されたが断った馬、契約、営業、エージェント交渉、騎手の主観は観測できない。したがって、次の表現を禁止する。

- 「騎手がこの馬を選んだ」
- 「陣営の勝負気配を直接観測した」
- 「依頼候補全体からの選択を復元した」

許される結論は次に限る。

> 観測済みの同一レース騎手 roster を条件としたとき、実際の馬－騎手配置に、事前固定した既知要因では説明できずterminal close市場にも残る予測情報が検出された／検出されなかった。

## 1. 既存研究との境界

### 1.1 EXP01との非同値性

EXP01は馬を独立行として、間隔、距離・場所・芝ダ・クラス変更、騎手継続、騎手格変化、および陣営通常行動からの逸脱を検証した。生の選択には小さい情報があったが、逸脱モデル固有の価値はなかった。

EXP20は次の点で異なる。

- 解析単位は馬単独ではなく、1レースの馬集合×騎手集合の完全割当。
- ある馬への騎手配置を、同一レースの他馬への配置と同時に評価する。
- 出力は騎手能力や乗り替わりそのものではなく、制約付き割当nullに対するedge residual。
- EXP01のraw choice blockを主対照として必ず入れ、これを超えない場合は「EXP01の再表現」として停止する。

### 1.2 他実験との境界

- EXP11の低頻度騎手×調教師ペア探索を再実施しない。疎なpair IDの暗記は禁止する。
- EXP14のhard split / regime specialistを再実施しない。
- EXP15のrace-as-set、EXP17の対戦graphとは異なり、対象はレース前に確定した人馬配置である。
- EXP16A/EXP18の教訓に従い、最初の性能Gateをterminal close市場に対する残差に置く。pre市場の改善だけでは進まない。

## 2. データ契約

### 2.1 主キーと使用列

歴史基盤は`master_v2`およびその生成前raw assetとする。候補列は次に限定し、Stage 0で実列名・型・被覆をmanifest化する。

- race: 日付、発走時刻、16桁race ID、場所、R、距離、芝ダ、クラス、トラックコード(JV)
- horse: 血統登録番号、馬番、性別、年齢、斤量
- people: 騎手コード、調教師コード
- as-of history: 過去の人馬騎乗、調教師－騎手騎乗、騎手の過去騎乗斤量、前走騎手
- market: historical_pre_snapshot、terminal_close_market
- outcome: Stage 1の結果loaderだけでwinnerを読む。Stage 0の構造loaderには着順・払戻・タイムを入れない。

馬名・騎手名によるjoinは禁止する。ID欠損、不正、衝突はfail-closedで別集計する。`馬主(最新/仮想)`と`前走レースID`生値は使わない。

### 2.2 時点

対象日Dの割当特徴はDのday-startより前の履歴だけで構築する。同日後続レース、対象レース結果、対象日以後の騎乗成績を使わない。同日カード自体はレース前に公表済みなので、対象venue-dayの観測済み騎手予定本数・レース間隔は利用できるが、結果により更新しない。

### 2.3 市場

- `historical_pre_snapshot`: 発走約26〜30分前。副診断に使用。
- `terminal_close_market`: 締切時点の公開情報集合。主性能Gateのnull。
- `result_file_odds`: 時刻不明のため不使用。

## 3. 条件付き候補集合

### 3.1 主候補集合 C0

各race rについて、最終starterに実際に騎乗した騎手の集合J_rと、最終starterの馬集合H_rを作る。候補edgeはH_r×J_rのうち、次を満たすものとする。

1. 騎手ID・馬IDが有効。
2. 騎手jの対象日前365日に観測された最低騎乗斤量`min_weight_365_asof(j)`に対し、馬hの負担斤量がそれ以上。履歴不足時はedgeを「不明」とし、可能扱いへ0埋めしない。
3. 同一騎手は同一raceで1頭だけ、各馬も騎手1人だけという一対一制約。

この候補集合は「実際に依頼可能だった集合」ではない。**実際にraceへ集まった騎手roster内のswap可能集合**である。

### 3.2 感度候補集合

結果を見ず、次の3定義をStage 0で固定して比較する。

- C0: 主定義。365日最低斤量制約。
- C1: 730日最低斤量制約。
- C2: 斤量制約を外したroster完全二部グラフ。

C0/C1/C2で主residualのrace内順位Spearman中央値が0.80未満、または符号一致が80%未満なら候補集合依存が強すぎるため終了する。新しい候補集合を結果後に追加しない。

### 3.3 除外

- 障害。`トラックコード(JV) 51..59`。
- 新馬。人馬履歴が構造的に乏しく別仮説になるため主解析から除外し、件数だけ報告。
- starter 5頭未満。
- 騎手ID重複、騎手不明、完全matchingが存在しないrace。
- DNFを含むraceは既存OOFとの整合のため主性能母集団から除外し、結果条件付き除外であることを明記する。Stage 0の割当被覆監査には含め、件数を分離する。

Stage 1の正式race setはEXP16A/17の平地・DNF除外基盤とrace ID差分を出す。2024/2025は封印する。

## 4. 割当nullと残差

### 4.1 辺特徴 X(h,j,r)

対象日前だけから次を作る。

1. `horse_jockey_same_prev`: 前走と同騎手。
2. `horse_jockey_n_365`: 当該人馬の過去365日騎乗数。
3. `horse_jockey_days_since`: 最終騎乗からの日数。
4. `trainer_jockey_n_365`: 調教師－騎手の過去365日騎乗数。
5. `trainer_jockey_share_365`: 当該調教師の過去365日騎乗に占める騎手比率。
6. `jockey_quality_365`: 対象日前365日の騎手成績。着順由来値は過去分だけで、Laplace縮約を固定する。
7. `jockey_weight_margin`: 当該馬斤量－騎手のas-of最低斤量。
8. `jockey_card_load`: 当日同venueで公表済みの騎乗予定本数。
9. `adjacent_race_load`: 前後raceの騎乗予定有無。結果は使わない。
10. horse sideの既知能力、jockey sideの既知能力、trainer marginal。pair IDそのものは入力しない。

連続値の標準化・カテゴリ語彙・縮約事前分布はY−1以前だけでfitする。

### 4.2 A0 — 独立edge null

各候補edgeの効用を`u0(h,j)=theta^T X_raw(h,j)`とする。`X_raw`は騎手能力、継続騎乗、調教師－騎手頻度、斤量margin、card loadを含むが、pair identity embeddingや未来成績を含まない。

### 4.3 A1 — 制約付きmatching model

同一raceの全edge効用行列Uに対し、一対一制約を持つentropy-regularized matchingを使う。主実装はSinkhornとし、行・列marginalが1になること、馬順・騎手順置換不変、同一入力でbit再現することを必須とする。

正則化強度は2018年以前のassignment likelihoodだけで固定する。2019〜2023の着順は使わない。巨大なpermutation全列挙やrace結果による選択は行わない。

### 4.4 主residual block R

実際に成立したedge(h,j*)について次を出す。

- `assign_logp`: A1が実edgeへ与えた条件付きlog probability。
- `assign_surprise`: `-log p_A1(j*|h,r)`。
- `assign_gain_vs_A0`: `log p_A1(j*|h,r) - log p_A0(j*|h,r)`。
- `assigned_jockey_rank`: 馬hの候補騎手中のA1順位を[0,1]へ正規化。
- `scarcity`: その騎手を高く評価する他馬数を反映した競合度。
- `dual_horse` / `dual_jockey`: matchingの双対価格。race内中心化して使用。

race全体で一定のmatching scoreは馬順位を変えないので主modelへ入れず、診断値だけにする。Rは上記6列を1 blockとして検定し、結果後に列を選ばない。

## 5. Stage 0 — 結果性能を見ない監査

### S0-A provenance・被覆

2013〜2023について年別に次を報告する。

- race/horse行数、騎手・調教師・馬ID被覆。
- 1 race内の騎手ID重複、不明、完全matching不存在。
- C0/C1/C2のedge密度、馬あたり候補騎手数。
- observed edgeが斤量制約で不可能になる件数。
- 新馬、障害、DNF含有race、starter<5の件数。
- 人馬pair、調教師－騎手pairの過去観測回数分布。1回だけのedge比率。
- 新人・長期休養騎手、短期免許騎手、乗り替わり、競馬場、芝ダ、頭数帯の被覆。

主進行床:

- 2019〜2023の各年で正式raceの95%以上に有効matching。
- observed edgeの99.9%以上が主候補集合C0でfeasible。
- 非新馬starterの90%以上で候補騎手5人以上。
- 主要層の有効race率が全体の0.8倍以上。

未達をmissing=0で補わない。

### S0-B assignment predictionと等価性

着順を使わず、assignment自体だけを目的変数としてrolling評価する。各年YはY−1以前でfitしYの実配置を評価する。

- B0: 一様なfeasible matching。
- B1: A0独立edge null。
- B2: A1制約付きmatching。

指標は実edge negative log likelihood、top-k assignment recall、calibration、完全matching likelihoodの近似値。B2がB1を改善すること自体は性能仮説のPASSではなく、残差定義が成立する最低条件である。

等価性対照E1を置く。RをEXP01 raw choice、騎手能力、調教師－騎手頻度、R0-cleanの既存列だけでrolling予測し、次を満たさなければ終了する。

- `assign_gain_vs_A0`のcross-fit R² < 0.80。
- residual 6列からEXP01 raw choice blockを予測するだけの逆向き対照でも、単一列置換になっていない。
- B2−B1のassignment NLL改善がmeeting-day bootstrapでCI95上限<0。
- C0/C1/C2でresidual順位のSpearman中央値>=0.80かつ符号一致>=80%。

ここを通らなければ「EXP01または既知marginalの再表現」としてStage 0で終了する。

### S0-C time-safety・invariants

最低限、次を合成データと実データで検査する。

1. 対象日以後の履歴を削除しても特徴がbit一致。
2. 対象race・同日raceの着順、タイム、払戻を書き換えても割当特徴がbit一致。
3. 同日全raceは同一day-start履歴snapshotを使う。
4. 馬順・騎手順・入力行順の置換不変。
5. Sinkhornの行列和が許容誤差1e-10以内で1。
6. infeasible edgeの確率が1e-12以下。
7. observed matchingが候補集合に含まれない場合はfail-closed。
8. pair IDを乱数または連番へ置換しても、ID自体を使わないarmはbit一致。
9. race内一定値を全馬に足してもsoftmax確率がbit一致。
10. 同一seed・同一入力でbit再現。
11. DNF・取消・新馬の扱いがspecと一致。
12. outcome loaderをStage 0構造処理から呼べないこと。

### S0-D 検出力

Stage 1で実際に使うterminal-close offset conditional logitへ、race内中心化したR block方向を注入する。方向はassignmentデータだけで凍結し、着順で選ばない。

- 評価期間: 2019〜2023。
- 推論単位: 暦日meeting-day。
- bootstrap: B=10,000。
- 合成outcomeのfit件数は各年のrolling実件数と一致。
- SIGNAL: ΔloglossのCI95上限<0。
- practical floor: terminalオッズ、現金込みKelly、控除20%、期待log成長1e-4/raceを超える最小効果。EXP18 v0.5方式で有限標本fit誤差を含める。
- floor、MDE、seed、注入方向、検出力を2019〜2023の実着順開封前にspecへcommitする。
- floorでの`PASS-PRACTICAL ∪ PASS-SIGNAL`検出力80%未満なら終了する。

## 6. Stage 1 — terminal close残差Gate

Stage 0の全Gate、Fable再レビュー、数値floor commit後だけ実施する。

### 6.1 arm

- `M0_CLOSE`: rolling temperature-calibrated terminal close単勝市場。
- `M1_CLEAN`: M0 + EXP16A R0-clean rolling OOF score。
- `M2_RAW_ASSIGN`: M1 + EXP01 raw jockey/choice対照 + A0 edge特徴。
- `M3_MATCH_RESIDUAL`: M2 + R block。

結合は`log q = a*log(m_close) + b*s_clean + theta_raw^T X_raw + phi^T R - log Z`。全係数はY−1以前だけでfitする。主比較はM3−M2であり、M3−M1だけを根拠にしない。

### 6.2 Gate M

2019〜2023 pooled、暦日bootstrapで次を全て要求する。

1. ΔloglossのCI95上限<0。
2. 5年中4年以上で改善方向。
3. leave-one-year-out 5通り全てでCI95上限<0。
4. 実ΔがP1/P2/P3の2.5%分位より小さい。
5. 5 deterministic refit中4以上で改善方向。同一モデルが決定論的なら再実行bit一致を代わりに報告し、空のseed条件を作らない。

等級:

- `PASS-PRACTICAL`: 上記を満たしCI95上限<`-practical_floor_nats`。
- `PASS-SIGNAL`: 上記を満たすが実務床未達。
- `FAIL`: それ以外。

PASS-SIGNALは科学的記録で終了し、ROIへ進まない。PASS-PRACTICALでも本番号ではROI・候補生成・賭金配分・production変更を行わない。pre市場は時点吸収の記述診断に限り、close FAILを救済しない。

## 7. placebo

- `P1_JOCKEY_ID_REWIRE`: 同一venue-day、同race頭数帯、騎手の当日騎乗本数を保つdegree-preserving rewire。200 draw。
- `P2_EDGE_DIRECTION_NULL`: R blockをrace内で符号反転または馬間置換し、multisetを保持。200 draw。
- `P3_ROSTER_SWAP`: 同venue×月×頭数帯の別raceから騎手rosterを借り、候補集合とmatchingを再構築。自身除外、singletonはfail-closed。200 draw。
- `P4_IDENTITY_FREE`: 騎手・調教師identity履歴を落とし、斤量・card load・marginal qualityだけを残す。診断対照。

判定は`real_delta < quantile(placebo_delta, 0.025)`。placeboはrolling fitを含めて再実行し、結果モデルの係数だけを固定して特徴を乱す近道を使わない。

## 8. 停止規律

次のいずれかで本番号を終了する。

1. 候補集合C0の被覆またはobserved-edge feasibility未達。
2. C0/C1/C2でresidualが不安定。
3. B2がB1のassignment NLLを改善しない。
4. RがEXP01/raw marginalからcross-fit R²>=0.80で再現される。
5. 時点安全性・置換不変性・完全matching invariantのいずれかFAIL。
6. Stage 1検出力80%未満。
7. Gate M FAILまたはPASS-SIGNAL。
8. placebo未超過。

停止後に候補集合、残差列、正則化、floor、期間を変更して救済しない。再開は別実験番号・新版事前登録・未使用期間を要求する。

## 9. 成果物

Stage 0:

- `PRIOR_ART_AND_EQUIVALENCE_AUDIT.md`
- `CANDIDATE_SET_AND_COVERAGE_AUDIT.md`
- `ASSIGNMENT_DATA_MANIFEST.json`
- `TIME_SAFETY_TESTS.json`
- `EQUIVALENCE_AUDIT.json`
- `POWER_AUDIT.md` / `out/power_audit.json`
- `COMPUTE_DRY_RUN.json`
- 凍結後`spec.json`

Stage 1へ進んだ場合:

- `REPORT.md`
- `out/gate_m.json`
- 年別・競馬場別・新人/短期免許・乗り替わり別の診断表

## 10. 本番号で主張しないこと

- 騎手や陣営の意思・自信を直接測った。
- 観測されない依頼候補を復元した。
- 騎手配置から長期利益が得られる。
- JRA全市場、他券種、地方競馬、2024/2025へ一般化できる。
- PASSした特徴を直ちにv6へ入れられる。

