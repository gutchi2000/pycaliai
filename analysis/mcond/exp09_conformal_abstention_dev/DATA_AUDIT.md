# EXP09 Stage 1 — データ・定義監査

**作成日**: 2026-09-21　**方法**: 実データ・既存モデル構造の直接確認。
**2024・2025年の性能・ROIは一切開封していない**(確認したのはファイルhash・
calibrator来歴・split境界(日付範囲のみ)・関数シグネチャ等の構造情報のみ)。

## 1. 予測対象・nonconformity score・単位(正式確定)

### 1.1 予測対象: 単勝(1着、レース内のどの馬が勝つか)

**確定理由**(実データ・既存コードから、推測ではなく):
- `pl_probs.py::all_tansho(w)`が、レースの全馬に対する「勝つ確率」の
  カテゴリカル分布(合計1)をそのまま返す関数として既に存在する
  (`w = pl_weights(scores)`はv6生スコアから直接計算、他の追加変換不要)。
  これは標準的な多クラス分類の予測確率分布そのものであり、APS
  (Adaptive Prediction Sets)等の確立されたconformal分類手法をそのまま
  適用できる。
- 本番の参加ゲート(`production_policy.py::hard_skip_reasons`)は既に
  **単勝確率(p_win)ベース**で設計されている: `field_chaos_score`=較正後p_win分布の
  正規化エントロピー、◎選択=`argmax p_win`、hard skip条件=`◎ p_win<0.05`。
  同じ量にconformalを適用するのが本番の意思決定構造と最も直接対応する。
- 3着内(top3/複勝)は「n頭中3頭が同時に正解」というマルチラベル問題であり、
  標準的な(single-label)多クラスconformal分類の枠組みに直接載らない
  (マルチラベルconformalへの拡張は追加の設計・検証が必要になり、本実験の
  スコープを超える)。単勝(single-label)を最小反証実験の対象とし、複勝は
  Gate5(経済評価)で単勝ゲートの副次効果として観察するに留める。

### 1.2 nonconformity score: APS(Adaptive Prediction Sets, Romano et al. 2020)型【2026-09-21数式確定】

既存の確立された定義を採用する(独自指標を発明しない)。レースiの全馬を
生PL確率降順に並べ p_(1)>=p_(2)>=...>=p_(n)、真の勝ち馬の順位をr_iとして:

```
S_i = Σ_{j=1}^{r_i} p_(j)
```

をnonconformity scoreとする(真の勝ち馬までの累積確率質量)。

**有限標本補正付きquantile(一般的なpercentile関数は使わない)**:
2023 calibration件数をn、alpha=0.10として、

```
k = ceil((n + 1) * (1 - alpha))
q_hat = 2023 calibration score集合のk番目に小さい値(k-th smallest order statistic)
```

**fail-closed処理**: kがnを超える場合(nが極端に小さいときにのみ起こりうる)、
`q_hat = 1.0`(予測集合が常に全馬を含む、最も保守的な状態)とする。この場合
Gate0で警告を記録し、eligible race集合の実件数(通常は数千件を想定)を報告する。

**randomized vs non-randomized APS**: **non-randomized APSを主解析に固定**する
(再現性優先)。randomized APS(境界馬の包含確率をランダム化してempirical
coverageをちょうどnominal値に近づける手法)は使わない。non-randomizedは
一般にconservative(empirical coverageがnominalをやや上回る傾向)になりうる
ことを明記する。

(RAPS(Regularized APS)的な正則化項は使わない、上記の素のAPS定義に固定する。)

### 1.3 予測集合の単位: レース単位

1レースにつき1つの予測集合(要素=そのレースの出走馬の部分集合)。馬単位の
独立予測集合ではない。予測集合サイズ(要素数)は、nominal coverageを満たす
ために必要な「上位から累積確率質量がnominal coverageに達するまでの馬数」で
決まる(APS標準手順)。

### 1.4 較正済み確率 vs 生のPL確率 — 較正器の2023 in-sample問題への対処

`models/pl_calibrators_v6.pkl`(および本番実運用の`pl_calibrators_v6_serve.pkl`)
はいずれも**fit_split="valid=2023"**(実測確認済み、n_races=3456)。EXP09の
conformal calibrationもまた2023年で行う計画のため、**この既存calibratorを
そのまま使うと、calibratorのfit元データとconformal calibrationデータが
同一になりin-sample**になる(EXP07 Gate J1で発見・修正したのと同型の問題)。

**対処(確定)**: nonconformity score計算には、Isotonic較正**前**の生PL確率
(`pl_probs.all_tansho(w)`、v6の生スコアのみに由来、v6自体はtrain<=2022で
fit)を使う。v6は2023データを一切見ていないため、2023全体を真にconformal
calibration専用として使える(較正器の二重使用問題が構造的に発生しない)。
これは同時に「conformal法は下流モデルが較正済みである必要がない」という
conformal predictionの一般的な利点とも整合する。

**比較対象への適用**: 「maximum probability」「entropy」の2方式もConformal
候補と同じ生PL確率(`all_tansho`)を入力とする(同一土俵での比較)。
**「現行chaos/participation gate」だけは例外**とし、本番が実際に使っている
較正済みpipeline(`pl_calibrators_v6_serve.pkl`)をそのまま読み取り専用参照
として使う(「本番が実際に何をしているか」を代表する比較対象のため、
生確率に置き換えると本番との対応が崩れる)。

### 1.5 同一レース内の依存関係の扱い

**周辺(marginal)coverageのみを主張する**。1レース=1つの独立観測
(「そのレースの真の勝ち馬」という1つのラベル)とみなし、レース内の
出走馬同士を複数の独立標本として扱わない。条件付き(conditional)coverage
(例:「特定の頭数帯・特定の競馬場では必ずnominal coverageを満たす」)や
個別レース単位の保証は主張しない(APS等の標準conformal分類が理論的に
保証するのは周辺coverageのみであり、それ以上は主張しない)。

### 1.6 calibration sampleの独立単位

**一次的にはレース**(APS型conformal分類の標準単位、1レース=1観測)。
ただし統計的検定(Gate1のempirical coverage検証・Gate2のbootstrap CI)は
EXP07/08と同じ**開催日単位**(meeting-day、rid16[0:10])で行い、レース間の
日内相関(同日同競馬場のレースが互いに独立でない可能性、EXP08で扱った
当日状態のような現象)を考慮する。conformal calibrationの母集団としては
レース単位を採用しつつ、有意性検定の頑健性は開催日単位で確保する
二段構えとする。

## 2. race-level abstention scoreへの変換(APS-derived abstention score)【2026-09-21tie-break確定】

各レースについて、APSでnominal coverageを満たすために必要な予測集合サイズ
(要素数、`prediction_set_size`)を計算する。これが小さいほど「モデルが自信を
持って少数の馬に絞り込めている」ことを意味する。**ただし予測集合サイズは
離散値のため、そのままではparticipation_rate(90/75/50/25%)を全方式で完全に
揃えられない**。結果ラベルを一切使わない、以下の3段階tie-break規則を
事前固定する:

```
1. prediction_set_sizeが小さいレースを優先(参加寄り)
2. 同じsizeなら、APS境界までのfractional effective set sizeで順位付け
   (境界(r_i番目)の馬の「必要な追加分」の割合、連続値。小さいほど参加寄り)
3. それも同値なら、race_id(rid16)と固定seed(20260921)から作った
   決定論的hash(sha256(f"{seed}:{rid16}")の先頭バイトを整数化)で順位付け
```

**この3段階を経た参加スコアは「prediction-set sizeだけ」ではなく、正確には
`APS-derived abstention score`と表記する**(以後この用語で統一)。

**各participation_rate点(90/75/50/25%)で、全7方式が完全に同じレース数を
選ぶ**よう実装する(スコアで昇順ソートし上位N件、Nは全方式共通)。
Conformalだけ参加率の未達・超過を許さない。

## 3. 時系列分割・モデル版の確認

| 区分 | 期間 | 実測(構造確認のみ) |
|---|---|---|
| base model(v6) train | 2013-01-05〜2022-12-28 | n=485,252行(master_v2実測) |
| conformal calibration | 2023-01-05〜2023-12-28 | n=47,273行(master_v2実測、valid split一致) |
| historical OOS evaluation | 2024-01-06〜2025-12-28 | n=94,249行(master_v2実測、test split一致) |

2026年は使わない(既に開封済み、prior-art evidence専用、`PRIOR_ART_AUDIT.md`
§2参照)。**本番採用判断にはEXP09固定後に新しく到来する未開封期間が必要**。

## 4. モデル版hash(固定、2026-09-21時点)

| 成果物 | パス | SHA256 |
|---|---|---|
| prediction model | `models/unified_rank_v6.pkl` | `0a040c4ec74fd7df00c05ef1440048def21ba940dc1ae67df00487b932701e8f` |
| calibrator(較正版参照、未較正で使用) | `models/pl_calibrators_v6.pkl` | `c3defc2457f35dcfb8edead68bbc14c23aeb4b117be3ec23cd3ee80dbb2d4a8c` |
| calibrator(本番実運用、chaos/participation gate比較専用) | `models/pl_calibrators_v6_serve.pkl` | `90318635646a277aeda48bc1a37b48d31e43ae7d13f6a6f9ce1aa25650eaf50b` |
| feature schema | `data/serve_feature_baseline.json` | `7e3bbc3224a75bd59cb9de919e2b03c9d7c84ef620e6bf8a2a606cd3ad8f7e0e` |
| nonconformity definition | EXP09独自実装(未実装、実装後にhash記録) | — |
| abstention rule | EXP09独自実装(未実装、実装後にhash記録) | — |

**いずれかが変わった場合、以前のcoverage保証は失効し再較正が必要**
([[project_calibrator_shifts_chaos_gate]]の実測例が根拠)。

## 5. nominal coverageの固定

**nominal_conformal_coverage = 0.90**(標準的なconformal predictionの慣行値、
2023データを見る前に固定)。感度分析として0.80/0.95も記録するが、主判定は
0.90で固定する(§Aのparticipation_rate=75%主判定とは別の軸、両方とも
2024-2025結果を見て変更しない)。

## 6. 交換可能性が破れた場合の扱い

Gate1(empirical conformal coverage検証)で2024年と2025年のいずれかが
nominal_conformal_coverageを継続的に下回った場合:
1. 保証の主張を「このモデル・calibrator版が変わらない期間内でのみ」に
   明示的に限定する。
2. それでも下回り方が大きい場合(事前指定許容誤差超過)、Gate1をFAILとし、
   Gate2以降は探索的位置づけに格下げする(正式なconformal保証としては
   主張しない)。
3. 許容誤差は実装前にspec.jsonへ数値で固定する(結果を見てから決めない)。

## 7. coverage保証の範囲(3種類の実測値を分離)【2026-09-21確定】

APSのnominal 90% coverage保証は、**交換可能性を仮定した全対象(all eligible)
レース上のmarginal coverageだけに限定する**。APS-derived abstention scoreで
選択した参加レース部分集合について「90% coverage」と表現してはならない
(参加選別は予測集合サイズに基づく選択であり、選択後の部分集合にAPSの
理論保証はそのまま及ばない)。以下を分けて出力する:

- `empirical_coverage_all_eligible_races`(**Gate1の保証判定に使うのはこれだけ**)
- `empirical_coverage_participating_races`(診断値、保証の対象外)
- `empirical_coverage_abstained_races`(診断値、保証の対象外)

## 8. 例外処理カテゴリ(実装前に固定、2023-2025構造監査で実測済み)

以下いずれも2024-2025**性能・ROI**を見ずに、構造(着順・馬番・頭数等の
存在/形式)のみを監査して確定した(下表の件数は着順の重複パターン等の
データ品質チェックであり、勝敗の予測性能評価ではない)。

| カテゴリ | ルール | 2023-2025実測(構造監査のみ) |
|---|---|---|
| 同着(1着馬複数) | 単一ラベルAPSの前提を満たさないため**主解析のeligible race集合から除外**。全7方式で同じレース集合を使う。 | 2023=3件、2024=6件、2025=6件(合計15件、`着順==1`が同一rid16で複数行出現するレース数) |
| 取消・除外馬 | master_v2は着順非数値/欠損の行を既に除外済みの構造 | 着順欠損率0%(3年とも) |
| probabilityがNaNの馬 | 1頭でもNaNならそのレース全体をeligibleから除外(部分馬だけ除外すると確率質量が未定義になるため) | Stage2実装時にv6スコア計算で実測・報告 |
| probability合計が1から外れる | 許容誤差`\|sum(p)-1.0\|<=1e-6`は丸め誤差として再正規化可能、超過分は黙って正規化せずeligibleから除外・異常件数記録 | 許容誤差は2024-2025結果を見る前に固定(1e-6) |
| 極端な少頭数(n_field<3) | eligibleから除外 | 2023-2025実測でn_field最小値=5、該当0件見込み(構造監査で確認済み) |
| race_id重複 | eligibleから除外・件数記録 | (rid16,馬番)重複行=0件(構造監査で確認済み) |
| 結果欠損 | eligibleから除外 | 着順欠損率0%(3年とも、上記と同じ確認) |
| 判断時点後の出走取消 | historical batch評価(本Stage)では発生しない設計(master_v2は確定済み出走馬のみ収録)。前向き評価を将来行う場合は別途定義が必要 | 該当なし(設計上) |

## 9. raw PL OOS provenance(Gate0ブロッキング要件、確認完了)

**確認方法**: manifest(`train_unified_rank.py`)・pickle内メタデータ・
git LFS artifact hashの3経路。

- **manifest**: `train_unified_rank.py:6` "Split: train ≤ 2022, valid = 2023,
  test = 2024-2025"、`:86` `df[df["split"]=="train"]`でtrain行のみ学習に使用、
  `:249` split定義に`"train_max":2022`を明記。
- **artifact hash provenance**: `models/unified_rank_v6.pkl`はGit LFS管理。
  commit `5d7cd9d0`(2026-04-30 "v6: train + audit complete")が唯一この経路を
  変更したコミット。LFSポインタのoid
  (`sha256:0a040c4ec74fd7df00c05ef1440048def21ba940dc1ae67df00487b932701e8f`)が
  現在の作業ツリーファイルのsha256と**完全一致**することを確認済み
  (`git show 5d7cd9d0:models/unified_rank_v6.pkl`でLFSポインタの中身を直接
  検証)。2026-04-30以降、内容の変更が一切ないことをcryptographicに確認した。
  ファイルのmtime(2026-09-20)はLFSチェックアウト等による見かけ上の更新であり
  内容変更の証拠ではない(git status/git diffともにclean)。
- **結論**: **Gate0 PASS**。2023年のraw PLはcalibration期間に対して真にOOS。

## 次の一手

`MINIMAL_FALSIFICATION_PLAN.md`§Aを本監査の結果で正式確定へ更新し、
spec.jsonへ凍結済み(コミット前提)。その後実装(Stage2)へ。
