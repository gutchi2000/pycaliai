# EXP09 — 最小反証実験・停止条件(確定版、spec.json凍結・実装直前)

**2026-09-21改訂**: ユーザー指摘により全面改訂。旧版は「coverage」という語を
参加率(participation rate)の意味で使っており、conformal predictionの用語
(予測集合が真の結果を含む割合)と混同していた。本改訂で用語を分離し、
予測対象・nonconformity score・単位・APS数式・tie-break・例外処理・raw PL
provenanceを全て確定した(`spec.json`へ凍結済み)。以後2024-2025結果を
見た後に変更しない。

## 0. 用語の分離(最重要、以後全文書・列名・出力で厳守)

| 用語 | 定義 |
|---|---|
| `nominal_conformal_coverage` | conformal手続きが理論上主張する被覆率(例: 1-α=0.90) |
| `empirical_conformal_coverage` | 評価期間で実測した、予測集合が真の結果(勝ち馬)を含んだ割合 |
| `participation_rate` | 参加/見送りゲートが実際に参加を選んだレースの割合(旧「coverage」) |
| `abstention_rate` | 1 - participation_rate |

**「participation_rate 75%でcoverage達成」のような混同表現は本ドキュメント・
実装コード・出力のいずれにも一切使わない。**

## 1. Conformalの予測対象・単位(Stage1で実データ・既存モデルから確定、後述§Aで暫定結論)

以下はStage1のデータ監査で実データ・既存モデル構造から確定する(推測しない)。
本ドキュメント末尾§Aに暫定結論を記載、DATA_AUDIT.md完成後に正式確定する。

- 予測対象(1着/3着内/レース内順位)
- nonconformity scoreの定義
- 予測集合の単位(馬単位でなく**レース単位**であることは構造上自明だが、
  Stage1で明文化する)
- race-level abstention scoreへの変換方法
- 同一レース内の馬の依存関係の扱い(**同一レース内の馬を独立標本として
  coverageを計算しない**、周辺(marginal)coverageのみを主張し、条件付き
  (conditional)coverageや個別レース保証は主張しない)
- calibration sampleの独立単位(レース or 開催日)

## 2. 時系列分割(固定)

- base prediction model: **2022年末までで固定**(v6 train<=2022と同一)
- conformal calibration: **2023年のみ**
- historical OOS evaluation: **2024年・2025年**(閾値・nonconformity score・
  Mondrian区分を再fitしない)
- **2026年は「新しいforward confirmation」として使わない**。既に結果を確認済みの
  期間であるため、prior-art evidenceとしてのみ記録する
  (`PRIOR_ART_AUDIT.md`§2のparticipation gate符号反転の参照情報として)。
- **本番採用判断には、EXP09を固定した後に新しく到来する未開封期間が必要**と
  明記する(2026年データはこの目的に使えない、既に開封済みのため)。

## 3. モデル版の固定(hash付き)

以下をhash付きで記録し、spec.jsonへ凍結する(Stage1のデータ監査で実際の
ファイルパス・hashを確定):

- prediction model(`models/unified_rank_v6.pkl`)
- calibrator(`models/pl_calibrators_v6.pkl`または`_serve`版、要確認)
- feature schema(`data/serve_feature_baseline.json`等)
- serve normalization(該当コードの版)
- nonconformity definition(EXP09独自実装、コード自体のhash)
- calibration dataset(2023年データの範囲・hash)
- abstention rule(EXP09独自実装のhash)

**これらのいずれかが変わった場合、以前のcoverage保証は失効し再較正が必要**
と明記する([[project_calibrator_shifts_chaos_gate]]の実測例(calibrator
差し替えでchaos分布が丸ごと平行移動)がこの規律の根拠)。

## 3.5 APSの数式・有限標本quantile・例外処理・coverage範囲(2026-09-21追加確定)

詳細は`DATA_AUDIT.md`§1.2・§2・§7・§8・§9、`spec.json`に完全な定義を凍結。
要点:

- **nonconformity score**: `S_i = Σ_{j=1}^{r_i} p_(j)`(確率降順、真の勝ち馬の
  順位r_iまでの累積確率質量)。
- **有限標本quantile**: `k = ceil((n+1)*(1-alpha))`、`q_hat` = 2023 calibration
  scoreのk番目に小さい値(一般的なpercentile関数は使わない)。k>nはfail-closed
  (`q_hat=1.0`)。
- **non-randomized APS固定**(再現性優先)。
- **coverage範囲**: nominal 90%保証は**全eligibleレース上のmarginal coverage
  だけ**。参加/見送り部分集合のcoverageは診断値(`empirical_coverage_
  participating_races`/`empirical_coverage_abstained_races`)として別出力、
  Gate1の保証判定には`empirical_coverage_all_eligible_races`のみ使う。
- **例外処理(主解析eligible race集合から除外)**: 同着(2023=3/2024=6/2025=6件、
  構造監査済み)・取消除外馬(既にmaster_v2で除外済み、欠損率0%)・NaN確率馬・
  確率和許容誤差1e-6超過・n_field<3(実測0件)・race_id重複(実測0件)・結果欠損
  (実測0件)。全7方式で同一のeligible race集合を使う。
- **raw PL OOS provenance確認済み(Gate0 PASS)**: `unified_rank_v6.pkl`は
  train<=2022のみ(manifest確認)、Git LFS artifact hash([[project_calibrator_shifts_chaos_gate]]
  型の検証)で2026-04-30コミット以降無変更を確認済み。2023年のraw PLは
  calibration期間に対し真にOOS。

## 4. 比較する7方式(全て同一participation_rateで完全に同じレース数を選ぶ)

1. **Conformal由来スコア**(新規、本実験の対象)
2. maximum probability(◎のp_win、較正済み)
3. entropy(全馬確率分布のnormalized entropy)
4. OOD/support(`ood_support.py`と同型設計をEXP09で独自実装、2023-only fit・
   PCA・k-NN距離、EXP06のコード自体は再利用しない)
5. feature missingness(特徴欠損率)
6. **2023年のみでfitしたLR_CONTROL**(`stage_b_gate_eval.py::_fit_lr_control()`と
   同型設計を独自実装)
7. 現行chaos/participation gate(`production_policy.py`のchaos percentile、
   読み取り専用の参照、コード変更なし)

**各participation_rate(90/75/50/25%)で全7方式が完全に同じレース数を選ぶ**よう
実装する(スコアで昇順/降順ソートし上位N件を選ぶ、Nは全方式共通)。
**Conformal由来のスコアはprediction_set_sizeを3段階tie-breakで連続順位化した
`APS-derived abstention score`**(§3.5参照、DATA_AUDIT.md §2に数式)であり、
「set sizeだけ」ではない。**Conformalだけ参加率の未達・超過を許さない**
(Conformalの予測集合サイズが自然に生む参加率ではなく、比較のためにスコア
ランキング化して同数を選ぶ)。

## 5. 主判定点と多重比較(結果を見て最良点を選ばない)

- **主判定はparticipation_rate=75%に固定**。90/50/25%は感度分析。
- **75%でFAILなら、他の参加率で良好でも主仮説はFAIL**とする(多重比較による
  有利な参加率の事後選択を禁止)。

## 6. Gate構造(0→1→2→3→4→5、順序厳守)

### Gate0: データ・時点・モデルhash
§1-3・§3.5の確定結果、判断時点での入力再現性(市場系はhistorical_pre_snapshot
規約)、モデル版hashの記録、**raw PL OOS provenance確認(PASS済み)**、
eligible race集合の生成(例外処理カテゴリ適用後の実件数)。

### Gate1: Conformal coverageの実測妥当性
2024年・2025年それぞれで**`empirical_coverage_all_eligible_races`**を測定し、
`nominal_conformal_coverage`(0.90)を**下回らない**ことを確認する(周辺coverage
の定義上の妥当性検証、経験的検証)。**`empirical_coverage_participating_races`/
`empirical_coverage_abstained_races`は診断値として別途記録するが、Gate1の
判定には使わない**(§3.5参照、参加部分集合にnominal coverageを主張しない)。
2024/2025で乖離が大きい場合は「exchangeability前提がこの期間で崩れている」と
解釈し、以降のGateで保証の主張を縮小する。

### Gate2: 同一participation_rateでの情報指標比較(主判定=75%)
2023developmentでfitした7方式を2024年・2025年へ固定適用(再fitなし)。
participation_rate=75%で、Conformal由来スコアの平均logloss/Brier
(レース単位)が、他6方式**全て**を下回ることを要求する。要件:
- 2024・2025両方で同方向
- 開催日単位paired bootstrap CI(EXP07/08と同じ規約)
- Conformalがmax-probability・entropy・LR_CONTROLの**すべて**を上回る
- 特定年度・競馬場・人気帯だけへの集中でない(Gate4で正式検証するが
  Gate2でも明らかな偏りがあれば記録)
90/50/25%は感度分析として同じ手順で記録するが、主判定には使わない。

### Gate3: full-control後の固有上積み(★最重要、EXP06 Jevが落ちた関門)
Conformal由来スコアを、モデル自身の確信度(m1-m4相当のtop prob・entropy・
市場確率・市場entropy・AI市場乖離・頭数・人気帯・競馬場・芝ダ・距離帯・
クラス帯・年度)**全てを統制した回帰**に追加項として入れ、係数の95%CIが
ゼロを跨がないことを要求する。**単純相関(Gate2)でPASSしても、この完全統制で
FAILすれば終了**(EXP06 Jevと同じ判定基準)。

### Gate4: 年度・競馬場・人気帯・芝ダートでの安定性
Gate2(participation_rate=75%)の結果を、年度(2024/2025個別)・10競馬場・
人気帯・芝ダートで層別確認する。特定区分だけへの依存が無いことを確認する。

### Gate5(Gate1-4全通過後のみ): 経済評価
Gate1〜4のいずれかがFAILならROIは実行しない。情報指標(logloss/Brier)を
主としてGate1-4を判定し、**ROIを使って方式・閾値・participation_rateを
選択しない**。通過した場合のみ、固定した同一馬券ルール・同一1レース投資額
(EXP07/08と同じflat配分の思想)で評価する。**Conformal方式だけ賭け金・
候補馬券・券種を変更しない**(比較対象と完全に同一条件)。**+5.31ptの教訓により
PASSしても「2026 as-served相当の独立期間で再確認するまでは本番配線しない」**
と明記する。

## 中止規律(結果を見た後に変更しない)

- participation_rate点(90/75/50/25%、主判定=75%)、7方式の定義、Gate0-4の
  閾値・統制変数リスト、LR_CONTROLの目的変数定義、OOD方式のk・PCA次元・
  距離関数、nonconformity scoreの定義、tie-break規則、のいずれも2024-2025
  結果を見た後に変更しない。
- Gate3(完全統制)でFAILしたら、統制変数を減らして再検定しない。
- Gate2で一部の方式にしか勝てない場合、比較対象から都合の悪い方式を外して
  再判定しない。
- Gate1でempirical coverageがnominal coverageを下回った場合、nonconformity
  scoreや閾値を事後調整して通過させない。
- 75%でFAILした場合、90/50/25%が良好でも主仮説はFAILのまま(§5)。
- どのGateで落ちても、それ以降のGateには進まない。

## 想定される終了条件(事前登録)

- Gate1 FAIL: empirical conformal coverageがnominal coverageを継続的に
  下回る → 「この期間ではexchangeabilityが成立せず、conformal保証を主張
  できない」と結論し終了(または保証の射程を大幅縮小した上でGate2以降は
  探索的位置づけとする)。
- Gate2 FAIL(participation_rate=75%): 単純方式(max-prob/entropy/LR_CONTROL)
  のいずれかに負ける → 終了(「Conformal機構固有の価値なし」)。
- Gate3 FAIL: モデル自身の確信度を統制すると価値が消える(EXP06 Jevと同型)
  → 終了。
- Gate4 FAIL: 特定区分への依存が強い → 終了、または射程を当該区分に限定して
  記録するのみ(本番化は見送り)。

## Stage 2実装順(ユーザー指定、厳守)

1. eligible race集合の生成
2. 2023 calibration score生成
3. q_hat固定
4. 2023を再利用せず2024・2025 prediction set生成
5. 全対象上のempirical coverage確認(Gate1、all_eligible_racesのみ)
6. APS-derived abstention score生成
7. 同一participation_rate比較(7方式)
8. Gate1〜4
9. 通過した場合だけGate5 ROI

**2024・2025年の結果を読んでからAPS形式・tie-break・例外処理を変更しない**。

## §A. Stage1で正式確定した論点(`DATA_AUDIT.md`参照、以後変更しない)

- **予測対象: 単勝(1着、レース内のどの馬が勝つか)**。理由: (a) 標準的な
  多クラスconformal分類(APS)がそのまま適用できる明確な単一正解ラベル問題、
  (b) 3着内(top3)はマルチラベル問題で標準的な多クラスconformalの枠組みに
  直接載らない、(c) 本番の参加ゲート(chaos=p_winのエントロピー、◎選択=
  argmax p_win、p_win<0.05 hard skip)は既に単勝確率ベースで設計されている。
- **nonconformity score: APS(Adaptive Prediction Sets、Romano et al. 2020)型、
  non-randomized固定**。`S_i=Σ_{j=1}^{r_i} p_(j)`(確率降順、真の勝ち馬の順位
  r_iまでの累積確率質量)。quantileは`k=ceil((n+1)*(1-alpha))`、q_hat=2023
  calibration scoreのk番目に小さい値(有限標本補正、percentile関数不使用)。
  alpha=0.10。k>nはfail-closed(q_hat=1.0)。
- **確率の入力は生PL確率(`pl_probs.all_tansho(w)`)、Isotonic較正前**。
  既存calibrator(`pl_calibrators_v6.pkl`)がvalid=2023でfitされておりin-sample
  問題を起こすため(DATA_AUDIT.md §1.4)、v6生スコア由来の確率をそのまま使う
  (v6自体はtrain<=2022でfit、2023を汚染しない)。比較対象のmax-probability・
  entropyも同じ生確率を使う。**「現行chaos/participation gate」だけは例外**で
  本番実運用の較正済みpipeline(`pl_calibrators_v6_serve.pkl`)を読み取り専用
  参照として使う。
- **予測集合の単位: レース単位**(1レースにつき1つの予測集合、要素はそのレースの
  出走馬)。
- **同一レース内の依存: 周辺(marginal)coverageのみ主張**。レース=1つの独立観測、
  レース内の複数馬を複数の独立標本として扱わない。conditional coverageや
  個別レース保証は主張しない。
- **calibration sampleの単位**: 一次的にはレース(APS標準単位)。統計的検定
  (Gate1のempirical coverage・Gate2のbootstrap)はEXP07/08と同じ開催日単位。
- **race-level abstention score = APS-derived abstention score**(prediction_set_size
  を主要因とし、3段階tie-break(§3.5)で連続順位化したもの、小さいほど参加寄り)。
- **coverage範囲は全eligibleレースのmarginal coverageのみ**、参加部分集合には
  保証を主張しない(§3.5)。
- **nominal_conformal_coverage = 0.90**(標準的な慣行値、2023データを見る前に
  固定。感度分析0.80/0.95も記録するが主判定は0.90)。
- **例外処理カテゴリを固定**(同着・取消除外馬・NaN確率・確率和異常・極小頭数・
  race_id重複・結果欠損、§3.5参照)。
- **raw PL OOS provenance確認済み(Gate0 PASS)**(§3.5参照)。
- モデル版hash・時系列分割境界(train<=2022年末/calibration=2023/OOS=2024-2025)
  はDATA_AUDIT.md §3-4・§9に記録済み。
