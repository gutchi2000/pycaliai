# EXP19 仕様案 — 当日馬体重状態 × JRA公式馬場物理値

**版**: v0.2-frozen（Fable最終承認。2026-09-26凍結）
**状態**: Stage 0実装可。結果未開封
**対象**: JRA平地、馬体重発表後のlate-decision予測
**禁止**: 承認前の学習・結果評価、2024/2025開封、ROI、候補生成、賭金配分、production変更

## 0. 目的と結論範囲

今走馬体重をその馬自身の平常値からの偏差として表現し、その意味がJRA公式のクッション値・含水率によって変わるかを検定する。

主問いは三段階である。

1. 馬体重発表後の判断時点市場に対し、当日馬体重状態は追加情報を持つか（A1）。
2. 同じ判断時点で、馬体重状態×公式馬場物理値は馬体重単独を超えるか（A2）。A1の成否に依存させない。
3. A1/A2で検出した情報はterminal close市場にも残るか（B1/B2）。

本実験は「重い馬が強い」「増減±Xkgが良い」といった全体則を探索しない。馬体重は成長、休養、個体差で意味が異なるため、as-ofの馬内基準からの偏差を使う。

## 1. 既存研究との境界

- `baba_eval.py`: 公式クッション/含水率と既存の馬場適性・脚質相互作用は実施済みでOOS悪化。再実施しない。
- EXP08: 同日先行レース結果から推定するオンライン馬場状態は終了。再実施しない。
- EXP15/EXP16A: 市場残差の検定基盤は再利用する。ただしR0-cleanの`斤量体重比`が今走馬体重由来と判明したため、EXP19では同列を除く`R0-clean-nobw`（110列）のrolling OOFを新規構築する。既存OOFは再利用しない。
- EXP13/P0 DNF監査: 市場分母ではDNFをstarterに残すのが正しい。ただし主解析はfinisher-only N1 OOFとの整合のためDNF含有raceを除外し、full-starter版を感度分析に分ける。

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
- 主母集団はEXP16A/17/18と同じく、DNFを含むraceを除外する。`starter − finisher`または既存proxyで判定し、既存15,951R基盤とのrace ID差分を報告する。
- この除外は結果条件付きであり、N1 OOFのfinisher-only被覆と既存実験との比較可能性を優先する暫定措置と明記する。
- DNFをstarterとして残す版は感度分析とし、N1欠損を0埋めしない。market-onlyおよびW-onlyで実行可能な範囲だけ報告する。
- winnerが一意に定まらないrace、race ID/馬番衝突、必要な市場格子不備はfail-closedで別集計。
- 全armは同じrace setとstarter setを使う。
- A1/B1の開発期間は2019〜2023。A2/B2のWP主blockはクッション通年被覆に合わせ2021〜2023。
- 2024/2025はEXP19仮説について未使用holdoutとして封印する。

## 4. 特徴契約

全てrace day-startまでの過去値と、当日発表済みの馬体重・公式馬場値だけで作る。

### 4.1 W — 当日馬体重状態（主block）

共線を避け、主blockは次の6種＋履歴数に縮約する。

1. `bw_log_kg`: log(current kg)。
2. `bw_sex_age_z`: 性別×年齢×時期の基準を過去年だけで作ったz。
3. `bw_robust_z5`: 直近最大5走のmedian/MADによるz。履歴2走未満はmissing。
4. `bw_abs_robust_z5`: 上記絶対値。
5. `bw_change_x_layoff`: `(current−previous)/previous × log1p(休養日数)`。
6. `bw_status`: measured/unmeasurable/missing-sourceを明示したone-hot。取消は母集団外。歴史fitで出現しない`not_yet_published`は設計行列から落とし、forward parityで出現率だけを監査する。
7. `bw_history_n`: 使用できた過去体重数。

`bw_change_kg`、`bw_change_pct`、`bw_dev_med5_pct`は説明用に計算できるが主modelへ投入しない。WPにも使わない。体重閾値は結果を見て選ばない。winsorize幅、MAD floor、履歴必要数はStage 0で分布だけを見て固定する。ridge等の追加正則化を結果後に導入しない。

`斤量体重比`は今走馬体重由来なので`R0-clean-nobw`から除く。`前走馬体重`、`前走馬体重増減`、`斤量`、`馬齢斤量差`は前日までに確定するため残す。

### 4.2 P — 公式物理馬場

1. `cushion_z_place_asof`: 競馬場別、対象日の前日までのexpanding windowで標準化。
2. `moist_gp_z_place_surface_asof`: 競馬場×surface別、対象日の前日までのexpanding windowで標準化。
3. `moist_4c_z_place_surface_asof`: 同上。
4. `moist_gradient_z`: GP−4C。
5. `measurement_age_minutes`: 発走時刻−測定時刻。
6. physical missing flags。

`measurement_age_minutes`はrace内定数の単独項にせず、WPの信頼度/重み側にのみ使う。Pはrace内定数であり、P単独がsoftmax順位を変えないことを構成assertする。

### 4.3 WP — 事前固定interaction（主仮説）

探索を避け、次の6本だけに固定する。

1. `bw_robust_z5 × cushion_z`（芝、2021〜2023）。
2. `bw_abs_robust_z5 × abs(cushion_z)`（芝、2021〜2023）。
3. `bw_robust_z5 × moisture_gp_z`（surface別、2019〜2023）。
4. `bw_abs_robust_z5 × abs(moisture_gp_z)`（surface別、2019〜2023）。
5. `bw_robust_z5 × moisture_gradient_z`。
6. `bw_change_x_layoff × track_extreme_z`。`track_extreme_z`は芝ではcushion/moisture、ダートではmoistureからStage 0で結果を使わず一意に定義する。

方向は固定しない。interactionの符号は学習するが、結果を見て項を追加・削除しない。WPは6本を1 blockとして2021〜2023で検定する。2019〜2020へcushion欠損を0投入しない。moisture-onlyの2019〜2023結果はsecondaryでGateに使わない。事前期待はA2で小さい正、B2でゼロ寄りと記録し、結果後に期待を変更しない。

## 5. armと比較

勝者に対するrace-level categorical loglossを主損失とする。各年Yの係数・較正・標準化はY−1以前だけでfitする。推論単位は暦日（YYYYMMDD）のmeeting-day。

- `N0_PRE_MARKET`: rolling temperature-calibrated historical_pre_snapshot単勝市場。
- `N1_CLEAN_NOBW`: N0 + `R0-clean-nobw`（110列）の新規rolling OOF score。`斤量体重比`を除外し、manifestにfeature list/model/input/loader sha256を保存する主null。
- `C_TRACK_ONLY`: N1 + 既存`baba_eval` block。負対照でGate対象外。
- `W_BODY`: N1 + W。
- `WP_BODY_TRACK`: W_BODY + WP。
- `T0_CLOSE_MARKET`: rolling temperature-calibrated terminal close市場。
- `TW_CLOSE_BODY`: T0 + clean-nobw score + W。
- `TWP_CLOSE_INTERACTION`: TW + WP。

主比較はA1=`W_BODY−N1_CLEAN_NOBW`、A2=`WP_BODY_TRACK−W_BODY`。両方をpre市場で独立に実行しHolm補正する。副報告では`W_BODY + 既存baba block`をnested controlとして構築し、`WP_BODY_TRACK − (W_BODY + 既存baba block)`を出す。片側だけWを持つ非nested比較は使わない。

B1/B2は、対応するA1/A2が`PASS-SUBFLOOR`以上の場合だけterminal closeで実行する。offset conditional-logitでは市場温度を自由にし、追加blockの係数を過去年だけでfitする。marketを固定offsetにして温度ずれを新情報と誤認しない。

結合式を固定する。`log q_N1 = a*log(m_pre) + b*s_clean - log Z`、`log q_W = a*log(m_pre) + b*s_clean + theta^T*W - log Z`。A2はさらに`phi^T*WP`を加える。全係数はY−1以前だけでfitする。EXP18の温度項と追加項を同時fitする`fit_cross`系実装を再利用し、独自の別fit経路を作らない。

## 6. Stage 0 — 結果性能を見ない監査

### S0-A provenance・parity

- 歴史torchとforward WHの同一race ID+馬番で、kgと増減の一致率を測る。
- 4開催日以上、400 horse rows以上、kg/増減の一致率99.5%以上、race coverage99%以上をfloorとする。`000`/`999`/not-yet-publishedのstatus一致率も99.5%以上を要求する。
- 不一致を丸め・未計測・取消・更新snapshotに分解する。
- earliest complete WH snapshotのpost時刻差をrace別・開催日別に分布として保存する。主判断時点T−28までに全starterが揃うraceが95%未満なら、historical_preをactionable基準にする設計を停止または判断時点を後ろへ変更し、新版specを要求する。
- historical_pre_snapshotは通常馬体重公表後なので、Gate Aは「市場の初期反応の過不足」を測ると明記する。
- `baba_today` と同日の公式保存値について、値・venue・測定日・測定時刻のparityを確認する。

### S0-B population・coverage

- 構造loaderは歴史torchを明示的なusecols whitelistで読む。読み込み後に`人気`、`単勝オッズ`、`複勝オッズ下限`、`複勝オッズ上限`、`複勝シェア`、`補正`、`指時系1〜4・単勝/人気/複下/複上/複人気`、結果、払戻が存在しないことを具体名とprefixでassertする。`指時系*`はprovenance不明のため市場源にも使わない。市場loader・結果loaderと別artifactにし、loaderコードsha256をmanifestへ保存する。
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

EXP18 v0.5と同じ方式を結果開封前に固定する。

- 生成: `log p = a*log(m_pre) + epsilon*c - log Z`。`c`はrace内中心化したWまたはWP blockを、構造データだけで凍結したfit方向。
- 推定器: 実際に使うoffset conditional-logitを、年Yの`n_fit(Y)`と同数の合成outcomeでfitする。独立な宣言ノイズは足さない。
- 決済: 実terminal単勝オッズ上の現金込みKelly、控除20%。
- 成長閾値: `1e-4/race`。年間約3,300Rならlog成長0.33、複利約+39%相当の厳しい基準である。
- bootstrap: 暦日cluster、B=10,000。seedはStage 0実装前に固定してmanifestへ保存する。
- SIGNAL: CI95上限<0。
- PASS-PRACTICAL: CI95上限<`-practical_floor_nats`。

A1/A2/B1/B2それぞれのMDEとfloorを算出し、数値を結果開封前に`spec.json`へcommitする。進行条件は宣言効果に対する`PASS-PRACTICAL ∪ PASS-SUBFLOOR`検出力80%以上。数値または方式を結果後に変更しない。

### S0-E compute

1年分の特徴構築、fit、10,000 bootstrap、placebo 200 drawの時間と最大RSSを測る。

## 7. Stage 1 Gate

Stage 0の再レビュー承認、`R0-clean-nobw` OOF manifest、数値floor commit後だけ実施する。判定関数以外から等級を付けない。

### Gate A1 — 判断時点の馬体重情報

`W_BODY − N1_CLEAN_NOBW`を2019〜2023で検定する。

- pooled暦日bootstrapのCI95上限<0。
- 4/5年で改善方向。
- leave-one-year-out 5通り全てでCI95上限<0。
- 実ΔがP1/P2分布の2.5%分位より小さい（改善側97.5 percentileを超える）。

### Gate A2 — 判断時点の馬体重×馬場interaction

A1の結果に依存せず、`WP_BODY_TRACK − W_BODY`を2021〜2023で検定する。

- pooled暦日bootstrapのCI95上限<0。
- 3/3年で改善方向。
- leave-one-year-out 3通り全てでCI95上限<0。
- 実ΔがP1/P2/P3分布の2.5%分位より小さい。

A1/A2のp値はHolm補正する。各GateはCI上限<`-floor`なら`PASS-PRACTICAL`、CI上限<0だがfloor未達なら`PASS-SUBFLOOR`、その他はFAIL。副分解やtrack-only結果を理由に救済しない。

### Gate B1/B2 — terminal close残差

- A1が`PASS-SUBFLOOR`以上ならB1=`TW_CLOSE_BODY − terminal clean-nobw null`を2019〜2023で検定する。
- A2が`PASS-SUBFLOOR`以上ならB2=`TWP_CLOSE_INTERACTION − TW_CLOSE_BODY`を2021〜2023で検定する。
- B1は4/5年・LOO 5通り、B2は3/3年・LOO 3通りを要求する。対応するplaceboとfloor規則はAと同じ。

B FAILは「pre時点で見えた情報が締切までに市場へ吸収された」と整合するが、唯一原因とは断定しない。B PASS-SUBFLOORは科学的残差のみでROIへ進まない。B PASS-PRACTICALでも本番号ではproduction化・ROI評価をしない。同一ハイパーパラメータのclean vNext比較を別番号で事前登録できるだけである。

### 記述診断（Gate外）

pre→terminal単勝オッズ変化と`bw_robust_z5`の相関を年別に報告する。選択・救済・Gate判定には使わない。

## 8. placebo

- `P1_IDENTITY_WITHIN_RACE`: 同一race内でW blockを馬identity間で置換。体重値multiset、市場、馬場、頭数を保持。200 draw。
- `P2_TIME_SHIFT`: 同競馬場×surface×年齢構成帯×頭数帯で、W blockを別raceへ置換。自身除外。200 draw。
- `P3_TRACK_SHIFT`: WPだけについて、同競馬場×surface×月の**別開催回**からPを置換する。同一開催回の日は借用しない。W、市場、結果を保持。200 draw。
- `P4_MISSINGNESS_ONLY`: 値を消しstatus/missing flagsだけ残す。値の情報と取得成否を分ける。

判定式は`real_delta < quantile(placebo_delta, 0.025)`。実Δがplacebo分布の2.5%分位より小さいとき、改善側97.5 percentileを超えたとする。境界を合成テストで固定する。

## 9. 停止規律

次のいずれかで該当経路を終了する。

1. 歴史/forward馬体重parity未達。
2. T−28までのforward complete coverage未達。
3. 正式母集団またはW/WP被覆未達。
4. 検出力未達。
5. A1 FAILならB1を行わない。A2 FAILならB2を行わない。A1とA2は互いを停止させない。
6. B1/B2がFAILまたはSUBFLOORなら、その経路はROI・vNextへ進まない。
7. 対応placebo未超過。
8. track-only負対照だけが改善し、W/WPが改善しない。
9. 同値の先行実験が見つかる。

全A経路がFAIL、または全terminal経路がFAIL/SUBFLOORならEXP19を終了する。停止文は対象期間・母集団・特徴定義・市場時点に限定し、「馬体重は無価値」「JRA市場は完全効率的」と一般化しない。

## 10. productionと外部公開

- 現行v6、朝のbundle、印、買い目、予算を変更しない。
- `data/forward_bodyweight/` と馬体重生値をsite/HFへ公開しない。
- TM/DM等の外部指数を入力しない。
- raw WH/baba、clean特徴、model artifactを混ぜず、版とhashをmanifest化する。
- rollbackは研究artifactの不使用で完了する。append-only rawは削除しない。

## 11. 凍結と再開

本v0.2-frozenをStage 0契約とする。凍結後の変更は版番号、理由、結果開封前commitを必須とする。Fableレビューは完了した。Stage 0実装では次を不変条件として確認する。

1. T−28市場とterminal closeの二段Gateが長期収益の問いに十分か。
2. W/WP特徴が多すぎず、既存baba検定の救済になっていないか。
3. historical torchとforward WHの4日/400行、値・status一致99.5%、race coverage99%、T−28 complete 95%が十分か。
4. 主解析でDNF含有raceを除外し、full-starterを感度分析に分ける扱いがN1被覆と市場分母の双方に対して妥当か。
5. A1/A2を独立にpreで検定し、対応するSUBFLOOR以上の経路だけB1/B2へ進める順序が妥当か。

