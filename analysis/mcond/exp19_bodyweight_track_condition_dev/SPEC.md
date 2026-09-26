# EXP19 仕様案 — 当日馬体重状態 × JRA公式馬場物理値

**版**: v0.1-draft  
**状態**: Fable仕様レビュー前。Stage 0未着手  
**対象**: JRA平地、馬体重発表後のlate-decision予測  
**禁止**: 承認前の学習・結果評価、2024/2025開封、ROI、候補生成、賭金配分、production変更

## 0. 目的と結論範囲

今走馬体重をその馬自身の平常値からの偏差として表現し、その意味がJRA公式のクッション値・含水率によって変わるかを検定する。

主問いは三段階である。

1. 馬体重発表後の判断時点市場に対し、当日馬体重状態は追加情報を持つか。
2. その情報はterminal close市場にも残るか。
3. 馬体重状態×公式馬場物理値は、馬体重単独を超えるか。

本実験は「重い馬が強い」「増減±Xkgが良い」といった全体則を探索しない。馬体重は成長、休養、個体差で意味が異なるため、as-ofの馬内基準からの偏差を使う。

## 1. 既存研究との境界

- `baba_eval.py`: 公式クッション/含水率と既存の馬場適性・脚質相互作用は実施済みでOOS悪化。再実施しない。
- EXP08: 同日先行レース結果から推定するオンライン馬場状態は終了。再実施しない。
- EXP15/EXP16A: clean tabular baselineと市場残差の検定基盤を再利用する。
- EXP13/P0 DNF監査: `master_v2`のfinishers-only母集団を使わない。DNFはstarterの損失として分母に残す。

新規性は「今走馬体重のas-of状態」と「その状態×公式物理馬場」に限定する。

## 2. データ契約

### 2.1 当日馬体重

歴史候補は `data/torch_20130105-20251228.csv`、forward候補は `data/forward_bodyweight/` のWH snapshotとする。結合キーは正規化16桁race ID + 馬番。血統登録番号は馬内履歴のkeyに使い、馬名joinは禁止する。

値の状態を区別する。

- `measured`: 200〜800kgの実測値。
- `unmeasurable`: JV `999`。
- `scratched_or_not_started`: JV `000`または確定した除外。
- `not_yet_published`: snapshot時点でWH record無し。
- `missing_source`: 原票欠損。

これらを0kg、増減0kgへ変換しない。増減は、前走のas-of実測体重がある場合に `current - previous` で検算し、原票値と不一致ならfail-closedで記録する。

### 2.2 公式馬場物理値

歴史は `data/baba_feats.parquet`、当日は `data/baba_today.json`。使用可能列は次だけ。

- turf cushion value
- turf moisture GP / 4C
- dirt moisture GP / 4C
- 各測定時刻
- 公式の粗い馬場状態と天気（baseline側）

`baba_today.json` 内の経験的 `shiba_bias` / `dirt_bias` は結果履歴から作った派生物なのでEXP19入力には使わない。生の物理値だけを使う。

### 2.3 市場

- `historical_pre_snapshot`: 発走約26〜30分前。判断時点基準。
- `terminal_close_market`: 締切プール。決済価格の情報集合であり結果情報ではない。
- `result_file_odds`: 時刻不明なので不使用。

## 3. 正式race set

- JRA平地。`トラックコード(JV) 51..59`の障害を除外。
- starter 5頭以上。
- starterは時点安全な出走情報と単勝オッズ>1.0を用いて確定し、取消・除外を除く。
- DNFはstarterとして残し、勝者でなければloss。DNFを理由にraceを除外しない。
- winnerが一意に定まらないrace、race ID/馬番衝突、必要な市場格子不備はfail-closedで別集計。
- 全armは同じrace setとstarter setを使う。
- 開発期間は2019〜2023。クッションinteractionの主期間は通年被覆の2021〜2023。
- 2024/2025はEXP19仮説について未使用holdoutとして封印する。

## 4. 特徴契約

全てrace day-startまでの過去値と、当日発表済みの馬体重・公式馬場値だけで作る。

### 4.1 W — 当日馬体重状態（主block）

1. `bw_log_kg`: log(current kg)。
2. `bw_sex_age_z`: 性別×年齢×時期の基準を過去年だけで作ったz。
3. `bw_change_kg`: current−previous measured kg。
4. `bw_change_pct`: `(current−previous)/previous`。
5. `bw_dev_med5_pct`: currentと直近最大5走のas-of中央値との差率。
6. `bw_robust_z5`: 直近最大5走のmedian/MADによるz。履歴2走未満はmissing。
7. `bw_abs_robust_z5`: 上記絶対値。
8. `bw_change_x_layoff`: change_pct×log1p(休養日数)。
9. `bw_history_n`: 使用できた過去体重数。
10. status/missing flags。欠損そのものと値を分ける。

体重閾値は結果を見て選ばない。winsorize幅、MAD floor、履歴必要数はStage 0で分布だけを見て固定する。

### 4.2 P — 公式物理馬場

1. `cushion_z_place_asof`: 競馬場別、過去年だけで標準化。
2. `moist_gp_z_place_surface_asof`。
3. `moist_4c_z_place_surface_asof`。
4. `moist_gradient_z`: GP−4C。
5. `measurement_age_minutes`: 発走時刻−測定時刻。
6. physical missing flags。

Pはrace内定数であり、P単独がsoftmax順位を変えないことを構成assertする。

### 4.3 WP — 事前固定interaction（主仮説）

探索を避け、次の6本だけに固定する。

1. `bw_robust_z5 × cushion_z`（芝、2021〜2023）。
2. `bw_abs_robust_z5 × abs(cushion_z)`（芝、2021〜2023）。
3. `bw_robust_z5 × moisture_gp_z`（surface別、2019〜2023）。
4. `bw_abs_robust_z5 × abs(moisture_gp_z)`（surface別、2019〜2023）。
5. `bw_change_pct × moisture_gradient_z`。
6. `bw_change_x_layoff × track_extreme_z`。`track_extreme_z`は芝ではcushion/moisture、ダートではmoistureからStage 0で結果を使わず一意に定義する。

方向は固定しない。interactionの符号は学習するが、結果を見て項を追加・削除しない。

## 5. armと比較

勝者に対するrace-level categorical loglossを主損失とする。各年Yの係数・較正・標準化はY−1以前だけでfitする。推論単位はmeeting-day。

- `N0_PRE_MARKET`: rolling temperature-calibrated historical_pre_snapshot単勝市場。
- `N1_CLEAN`: N0 + EXP15/16AのR0-clean OOF score。現在の表特徴を統制する主null。
- `C_TRACK_ONLY`: N1 + 既存 `baba_eval` 相当block。負対照でGate対象外。
- `W_BODY`: N1 + W。
- `WP_BODY_TRACK`: W_BODY + WP。
- `T0_CLOSE_MARKET`: rolling temperature-calibrated terminal close市場。
- `TW_CLOSE_BODY`: T0 + N1のclean score + W。
- `TWP_CLOSE_INTERACTION`: TW + WP。

offset conditional-logitで市場の温度を自由にし、追加blockの係数を過去年だけでfitする。marketを固定offsetにして温度ずれを新情報と誤認しない。

同容量対照とするため、W/WPと同数の履歴安全な無関係候補を足すplaceboではなく、下記identity-preserving placeboを使う。

## 6. Stage 0 — 結果性能を見ない監査

### S0-A provenance・parity

- 歴史torchとforward WHの同一race ID+馬番で、kgと増減の一致率を測る。
- 2開催日以上、200 horse rows以上、一致率99.5%以上、race coverage99%以上を暫定floorとする。
- 不一致を丸め・未計測・取消・更新snapshotに分解する。
- earliest complete WH snapshotのpost時刻差をrace別に測る。主判断時点T−28までに全starterが揃うraceが95%未満なら、historical_preをactionable基準にする設計を停止または判断時点を後ろへ変更し、新版specを要求する。
- `baba_today` と同日の公式保存値について、値・venue・測定日・測定時刻のparityを確認する。

### S0-B population・coverage

- 構造loaderは歴史torchを明示的なusecols whitelistで読み、`人気`、全オッズ、指数、補正、結果、払戻を保持しない。市場loader・結果loaderと別artifactにし、禁止列不存在をassertする。
- 2019〜2023の年別race/horse行数、体重status、過去履歴数、馬場値被覆を結果列なしで集計。
- 既知の構造値として、平地231,068行/16,645R、current kg有効率99.807%、血統登録番号欠損0を再現する。
- WP主評価で、Wと必要なPが同時に利用可能なraceが各年90%以上。未達の場合はmissingを0にせず、そのinteractionを未検証として停止する。
- 競馬場、芝/ダート、年齢、休養期間、人気集中帯で全体被覆の0.8倍未満になる層を報告する。

### S0-C time-safety・invariants

- 対象日以後を削除しても過去年特徴がbit一致。
- 対象raceの結果、着順、タイムを書き換えてもfit前特徴がbit一致。
- 同日全raceは同一day-start履歴snapshotを使う。
- 行順・馬番順置換不変。
- 体重履歴へ取消をslotとして入れず、DNFは実測体重があればslotとして残す。
- P単独を全馬へ同じ係数で足したsoftmaxがbit一致。
- raw input、feature manifest、race setをsha256で固定。

### S0-D power・floor

2019〜2023の構造、meeting-day cluster、実際のmissing patternだけを使うlabel-free注入で、Gate A/B/CそれぞれのMDEを算出する。practical floorはMDEを確認後、結果開封前に数値として `spec.json`へcommitする。floorを結果後に変更しない。

進行条件は、事前floorの効果に対する `PASS ∪ PASS-SUBFLOOR` 検出力80%以上。検出力不足ならStage 1を開かず終了する。

### S0-E compute

1年分の特徴構築、fit、10,000 bootstrap、placebo 200 drawの時間と最大RSSを測る。

## 7. Stage 1 Gate

Stage 0の仕様レビューとfloor commit後だけ実施する。

### Gate A — 判断時点の馬体重情報

主比較は `W_BODY − N1_CLEAN`。

解釈のため、`bw_log_kg`と`bw_sex_age_z`だけの安定した体格blockを`W_SIZE`として副分解し、`W_BODY − W_SIZE`（当日の状態情報）も必須報告する。Gateは事前固定した主比較だけで判定し、副分解を理由に救済しない。

- pooled meeting-day bootstrap CI95上限<0。
- 4/5年で改善方向。
- leave-one-year-out全てでCI上限<0。
- P1/P2 placeboの改善側97.5 percentileを超える。

floor以上なら `PASS-PRACTICAL`、有意だがfloor未満なら `PASS-SUBFLOOR`、それ以外はFAIL。FAILならGate B/Cを行わず終了。

### Gate B — terminal close残差

Gate A通過時だけ、`TW_CLOSE_BODY − (T0_CLOSE_MARKET + N1 clean score)`を同じ規則で検定する。

- FAIL: 馬体重は判断時点から締切までの価格変動を先読みしたが、決済価格には残らない。市場吸収研究として記録し、ROIへ進まない。
- PASS-SUBFLOOR: 科学的残差のみ。ROIへ進まない。
- PASS-PRACTICAL: Gate Cを許可。

### Gate C — 馬体重×物理馬場

Gate B PASS-PRACTICAL時だけ `TWP_CLOSE_INTERACTION − TW_CLOSE_BODY` を検定する。6 interactionを1 blockとして主検定し、個別係数は説明用でGateに使わない。芝cushion subsetと全surface moisture subsetの2本はHolm補正する。

Gate C PASS-PRACTICALでも本番号ではproduction化・ROI評価をしない。同一ハイパーパラメータのclean vNext retrainを別番号で事前登録できるだけである。

## 8. placebo

- `P1_IDENTITY_WITHIN_RACE`: 同一race内でW blockを馬identity間で置換。体重値multiset、市場、馬場、頭数を保持。200 draw。
- `P2_TIME_SHIFT`: 同競馬場×surface×年齢構成帯×頭数帯で、W blockを別raceへ置換。自身除外。200 draw。
- `P3_TRACK_SHIFT`: WPだけについて、同競馬場×surface×月でPを別開催日へ置換。W、市場、結果を保持。200 draw。
- `P4_MISSINGNESS_ONLY`: 値を消しstatus/missing flagsだけ残す。値の情報と取得成否を分ける。

real improvementは各placebo分布の改善側97.5 percentileを超えること。符号は実装テストで固定する。

## 9. 停止規律

次のいずれかで終了する。

1. 歴史/forward馬体重parity未達。
2. T−28までのforward complete coverage未達。
3. 正式母集団またはWP被覆未達。
4. 検出力未達。
5. Gate A FAIL。
6. Gate B FAILまたはSUBFLOOR。
7. placebo未超過。
8. track-only負対照だけが改善し、W/WPが改善しない。
9. 同値の先行実験が見つかる。

停止文は対象期間・母集団・特徴定義・市場時点に限定し、「馬体重は無価値」「JRA市場は完全効率的」と一般化しない。

## 10. productionと外部公開

- 現行v6、朝のbundle、印、買い目、予算を変更しない。
- `data/forward_bodyweight/` と馬体重生値をsite/HFへ公開しない。
- TM/DM等の外部指数を入力しない。
- raw WH/baba、clean特徴、model artifactを混ぜず、版とhashをmanifest化する。
- rollbackは研究artifactの不使用で完了する。append-only rawは削除しない。

## 11. 凍結と再開

Fable承認後にv0.2-frozenを作り、Stage 0を実装する。凍結後の変更は版番号、理由、結果開封前commitを必須とする。Fableへのレビュー依頼では、特に次を確認する。

1. T−28市場とterminal closeの二段Gateが長期収益の問いに十分か。
2. W/WP特徴が多すぎず、既存baba検定の救済になっていないか。
3. historical torchとforward WH parity floorが十分か。
4. DNFをstarterに残す母集団が市場分母と一致するか。
5. Gate CをGate B PASS-PRACTICAL後に限定する順序が妥当か。

