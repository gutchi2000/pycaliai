# EXP18 仕様案 — 複数プール市場トモグラフィー

**版**: v0.3-draft
**状態**: Fable再レビュー反映済み・差分再確認前・未凍結
**主対象**: 馬連（順不同の1着・2着組）  
**禁止**: Stage 0承認前の学習・成績評価、T2/offsetでの2024/2025開封、ROI、候補生成、資金配分、production変更

## 0. 研究目的

単勝市場が効率的でも、単勝・複勝・馬連など別々の投票プールが、同じ潜在着順分布と整合するとは限らない。本研究は、複数プールを一つのレース結果に対する異なる周辺観測として扱い、対象プールを入力から完全に外した状態で、その対象プールより結果分布を良く説明できるかを検証する。

最初の対象は馬連とする。単勝・複勝等から潜在着順分布を復元して得た馬連確率が、較正済みterminal UMAREN marketに対して残差情報を追加するかを、offset conditional-logitと実際の1着・2着順不同組に対するrace-level categorical loglossで比較する。tomography確率が馬連市場を単独で置き換えられるかという頭対頭比較は副指標とする。

これは次の既存研究とは異なる。

- EXP07: 同じ候補馬券に対する配分方法の比較。
- 既存`plan_exotics_real_ev.md`: 予測確率×実オッズによるEV選別・ROI比較。
- `build_joint_substrate.py`: 単勝marginalと馬連価格の基盤生成。
- EXP16A: 表特徴がterminal単勝市場へ残差情報を足すかの検定。

EXP18の主仮説は「馬の表特徴が市場に勝つ」ではなく、「他プールから導いた共同分布が、対象プール固有の価格歪みを検出する」である。

## 1. 仮説

### H1: cross-pool coherence

単勝・複勝等の非対象プールから時点安全に復元した馬連確率 `q_LOPO(i,j)` は、対象プール自身のde-vig確率 `m_UMAREN(i,j)` と系統的に異なる。

### H2: terminal residual

対象UMARENを入力に使わない `q_LOPO` が、較正済みterminal UMAREN marketの温度補正だけでは説明できない追加情報を与える。主比較は市場温度だけのnested nullに対するcross-pool係数βの上乗せである。

### H3: actionability

H2が実務床以上で通った場合に限り、historical_pre_snapshot時点で観測できるプールから、terminal UMAREN marketに対して残る歪みを予測できる。

H1だけでは勝てる根拠にならない。H2は情報の存在、H3は判断時点での利用可能性を問う。ROIはこの番号では扱わない。

## 2. データと名称

### 2.1 既知の履歴資産

- `data/Time _series_odds/TANPUK_*.csv`: 単勝・複勝Lo/Hi、区分、記録時刻、単勝票数、複勝票数。
- `data/Time _series_odds/UMAREN_*.csv`: 全馬連組合せオッズ、区分、記録時刻、馬連票数。
- 無接頭辞の大容量CSV、2023年91列ファイル、2026年227列ODファイル、ワイド払戻資産はStage 0で列・券種・時点・被覆を再監査するまで使用可否を決めない。
- 新たに試みた全体票数出力は整数オーバーフローで取得不能。追加取得をStage 0の要件にしない。

既存CSVの`単勝票数`・`複勝票数`・`馬連票数`は券種全体の票数であり、組合せ別票数ではない。流動性層別の補助変数としてのみ使用し、組合せ別実票数と呼ばない。オッズから組合せ票数を逆算する処理は、JRA払戻式・控除・丸め・返還を含む往復再現テストが通るまで禁止する。

### 2.2 時点名称

- `historical_pre_snapshot`: 発走約26〜30分前、中央値約28分。T-10と呼ばない。
- `terminal_close_market`: 区分4の確定プール。記録は発走後だが情報時点は締切。oracle、final result odds、実配信再現と呼ばない。
- `result_file_odds`: 時刻不明。主入力・主基準に使わない。

### 2.3 正式race set

EXP16Aと同じ母集団を初期候補とし、Stage 0で馬連固有の決済条件を追加して固定する。

- JRA平地。`トラックコード(JV) 51..59`の障害を除外。
- `race_id`は正規化済み16桁、馬番は整数。馬名joinは禁止。
- starter 5頭以上。
- TANPUK側で全starterに単勝・複勝オッズがある。UMAREN格子欠損は母集団定義に使わず、後段で「対象市場で決済不能」として年別件数を報告して除外する。
- DNFを含むレースは主評価から除外し、結果条件付き除外であることを明記する。
- 1着・2着の同着など複数の馬連的中組が生じるレースは主評価から除外し、settlement感度分析へ分離する。
- 取消・除外・返還・不成立を券種ごとにfail-closedで処理し、件数を年別に報告する。
- 全armは完全に同じrace setとticket setを使用する。

## 3. 確率空間と評価単位

馬連の標本空間は、race rのstarter集合 `S_r` に対する全順不同組 `C(S_r,2)` とする。各armは全組に正の確率を与え、race内合計を1にする。missing組を0として捏造しない。

主損失は、実際の1着・2着順不同組 `y_r` に対する

`LL_r(q) = -log q_r(y_r)`

である。ticket行を独立標本として検定しない。差分はrace内で一つに集約し、推論単位はmeeting-dayとする。

対象UMAREN市場の主基準は、年YについてY-1以前だけでfitしたpower-law較正de-vig `m_cal(i,j) ∝ (1/o_ij)^γ` とする。`γ=1` の比例正規化はsecondaryとする。

温度nullは `q_temp(i,j) ∝ m_cal(i,j)^a0`、cross-pool代替は `q_cross(i,j) ∝ m_cal(i,j)^a * q_LOPO(i,j)^β` とし、`a0`・`a`・`β`はY-1以前だけでfitする。主差分は `Δ_r=LL_r(q_cross)-LL_r(q_temp)`。これにより市場の年内温度ずれとcross-pool情報を分離する。`q_LOPO=m_cal`で説明変数が完全共線になる場合は重複列を落として`q_cross=q_temp`をbit一致で返し、係数値を解釈しない。比例市場だけに勝ちrolling較正済み温度nullに勝てない場合はFAILとする。

## 4. tomographyモデル

### 4.1 leave-one-pool-out原則

UMARENを評価対象とするとき、`q_LOPO`のfit、特徴、制約、較正、ハイパーパラメータ選択に、同じ時点のUMARENオッズ・票数・人気順位・払戻を一切使わない。UMARENは比較基準と結果決済にだけ使う。

### 4.2 Stage 0で監査する候補族

1. **T0_HARVILLE**: terminal TANSHOからPL/Harvilleで順不同top-two確率へ写像する既存再現アンカー。
2. **T1_STERN**: TANSHOだけを入力とするrolling依存補正。頭対頭は既実施で、主用途はoffset残差score。
3. **T2_SOFT_FUKUSHO_MAXENT**: MaxEnt事前と複勝由来soft制約。単勝marginalは等式制約、複勝は `w_fuku*Σ_i(P_q(i∈top-k)-τ_i)^2` の単一ペナルティ形、`k=places(n)`とする。`w_fuku∈[0,1]`はY-1以前だけでfitする。観測だけで真の共同分布が識別されるとは呼ばず、w=0でT1とbit一致させる。他の関数形へ結果後に変更しない。
4. **T3_NULL_CALIBRATED**: rolling power-law較正済みterminal UMAREN市場。
5. **T3_NULL_PROPORTIONAL**: 比例de-vig感度。これだけに勝ってもPASSしない。

全順列の列挙を必須にしない。n≤8の合成問題では全列挙oracleと一致させ、実レースでは動的計画、近似推論または周辺制約最適化を使える。ただし近似誤差をStage 0で上限化し、arm間差より十分小さいことを示す。

### 4.3 禁止する情報

- 対象UMARENの同時点価格をtomography入力へ混ぜる。
- 2024/2025の結果、払戻、成績指標。
- result-file oddsを判断時点価格として使う。
- outcomeを見てde-vig方式、依存補正、券種、人気帯を選ぶ。
- R0-cleanや馬固有特徴をprimary market-only armへ混ぜる。馬モデル追加はmarket-only機構が通った後の別仮説。

### 4.4 既知の2024/2025開封範囲

`crux_joint.py`は9時snapshotのT0/T1頭対頭を2024/2025で既に評価している。したがって未開封はT0/T1頭対頭には適用しない。EXP18でpristineとして封印するのはT2 soft-fukushoとoffset残差検定である。

## 5. Stage 0 — 実装前監査

Stage 0は2019〜2023の結果性能を開封しない。γ・λ・w・anchorのdry-run fitは2018年以前だけで許可し、全Stage 0 loaderで`max(year)<=2018`をassertする。以下を完了し、Fable再レビューを受ける。

### S0-A 先行研究・等価性

- `build_joint_substrate.py`、`plan_exotics_real_ev.md`、deep bet search、Harville/Stern/Dr.Z、EXP07、EXP16Aを全件監査。
- `crux_joint.py`の既知結果（9時、λ fit≤2023、2024/25評価、market LL 3.343 < Harville 3.380）をT0/T1頭対頭の既実施結果として扱う。Stage 0 anchorは2018年以前だけを使い、9時snapshotで市場優位の方向一致かつ`|Δ|`が0.037の0.5〜2倍であることを要求する。2019〜2023 anchorはStage 1でGateと同時にのみ算出する。
- T0/T1頭対頭を新規研究として再実施しない。新規性はT2 soft制約と較正済みUMAREN市場へのoffset残差に限定する。
- offset残差またはT2まで同値に実施済みなら終了。単なるEV選別、配分、PL jointの再実行でも終了する。

### S0-B provenance・schema

各ファイルについて期間、sha256、encoding、列、区分、記録時刻、情報時点、券種、組合せ完全性、票数列の意味をmanifestへ記録する。EXP18専用loaderは2024/2025行を読み込み直後に破棄してassertする。T0/T1頭対頭は既開封と記録し、T2・offsetの性能と払戻は開封しない。

### S0-C exactness・決済

- raceごとの理論組数 `n(n-1)/2` と実列の一致。
- オッズ<=1、欠損、取消、返還、同着、丸め、払戻の処理。
- 比例de-vig後の確率和1、馬番順序不変性、対称性。
- n≤8の全順列oracleとtomography周辺確率の一致。
- synthetic coherent marketではTANSHO由来とUMAREN市場が一致する。
- 意図的に一つのプールだけ歪めた合成市場で、そのプールを正しく検出する。
- `U_SELF`: UMAREN自身を検算入力にした経路が同じde-vig市場確率をbit一致で再現する。
- 複勝Lo/Hi逆算を使う場合、使用するJRA払戻式（純プール、的中票控除、places(n)による分配、10円breakage）を実行前に監査文書へ固定する。許容は馬行ごとに `|再生成−表示|<=max(0.05,0.02*表示)`、年別99%以上。未達ならT2未実装で停止する。place確率のrace内和`places(n)±1e-6`は構成assertとして記録する。
- `w_fuku=0`でT2がT1とbit一致する。

### S0-D coverage

2019〜2023について、結果ラベルを使わず次を年別・競馬場別・頭数帯別・人気集中帯別に測る。人気集中帯はTANPUK単勝entropyだけで定義し、UMARENを使わない。

- TANSHO terminalとUMAREN terminalのrace coverage。
- 全組合せcoverage。
- historical_pre_snapshotの同時被覆。
- pool-total vote-count coverage。ただしGate要件にはしない。
- formal race setの予定件数と除外理由。

母集団は先にTANPUKとレース情報だけで定義する。その後、UMAREN格子欠損は「対象市場で決済不能」として除外件数を報告する。UMAREN格子完全性で母集団自体を定義しない。terminalの0.0セルは頭数外と取消・返還可能性をTANPUK starter集合で分離する。

暫定floorは、terminalの正式race set coverage 95%以上、全組合せ完全race 99%以上、preとterminalの同時coverage 90%以上とする。数値は性能開封前のStage 0レビューで確定する。

### S0-E power

暦日単位のmeeting-day clusterで効果注入とMDEを測る。SIGNALは`CI95上限(Δ)<0`で数値floorを置かない。PRACTICAL floorは、生成構造`q_est=m_cal*exp(ε*s_true+noise)`、`SD(noise)=sqrt(2Δ)`、控除22.5%、参加条件`max(q_est/m_cal)>1/0.775`、成長閾値`1e-4/race`、seedを結果開封前に固定して算出する。算出値を`spec.json.practical_floor_nats`へ数値でcommitし、以後の変更は新版番号を要求する。EXP16Aの0.005を流用しない。`-log(0.775)=0.255 nats`は全額比例投入の参考条件でhard gateにしない。宣言効果で検出力80%以上を進行条件とする。

### S0-F compute dry run

1年分の市場構築、全組確率、bootstrap、placeboの時間・最大RSS・推定全期間時間を測る。近似法を使う場合はoracle誤差も記録する。

## 6. Stage 1 — rolling retrospective evaluation

Stage 0承認時だけ実施する。developmentは2019〜2023。2024/2025はT2とoffset残差について封印する。T0/T1頭対頭は既存`crux_joint.py`で開封済みのため再利用しない。

- 年Yのfit、依存補正、較正、de-vig感度パラメータはY-1以前だけを使用。
- 5 seedが必要な学習要素は独立fitし、主判定はmedian seed、4/5 seed方向一致。
- 主推論は年層化meeting-day bootstrap。4/5年方向一致とleave-one-year-outを要求する。

### Gate M1: terminal market residual

主armは`UB2_OFFSET_U2`。`UB3_OFFSET_U3`は複勝往復Gate通過時だけ追加し、`UB1_OFFSET_U1`はanchor専用でGateに使わない。Gate対象armは1本または2本で、その数に応じBonferroni補正する。各offset armは温度nullとの差で判定し、`UB2−UB1`と`UB3−UB2`を必須decompositionとして報告する。頭対頭`T1−T0`と`T2−T1`は副指標とする。

- **PASS-PRACTICAL**: CI95上限<0、pooled点推定がStage 0実務床以上、4/5年、4/5 seed、全LOOでCI上限<0、P1/P2/P5超過。
- **PASS-SIGNAL**: 同じ統計条件を満たすが実務床未満。
- **FAIL**: CI、方向、LOO、placeboのいずれかを満たさない。

PASS-SIGNALは科学的結果であり候補生成へ進めない。PASS-PRACTICALだけがGate M2を許可する。

### Gate M2: historical_pre_snapshot actionability

M1 PASS-PRACTICAL時だけ実施する。pre時点の非対象プール確率と、Y-1以前だけでfitした馬連pre→terminalドリフト表から予測したterminal UMAREN価格を用い、terminal市場に対して残る情報を検定する。terminal価格を直接入力したarmは診断専用で、actionableとは呼ばない。

M2の具体的なforecast target、fit窓、損失、floorはM1結果を見る前にStage 1仕様として凍結する。M2 PASS-PRACTICALまではROI、候補券、賭金を扱わない。

## 7. arm

- `U0_TERMINAL_UMAREN_CAL`: rolling power-law較正済みterminal UMAREN市場。
- `U0P_TERMINAL_UMAREN_PROP`: 比例de-vig感度。
- `U1_HARVILLE_TAN`: terminal TANSHO由来の再現アンカー。
- `U2_STERN_TAN`: 過去年fit依存補正アンカー。
- `U3_SOFT_FUKUSHO_MAXENT`: MaxEnt事前とsoft複勝制約。
- `UB1/2/3_OFFSET`: `q_temp∝U0^a0`に対する`q_cross∝U0^a*Uk^β`。UB1はanchor、UB2が主arm、UB3はT2 Gate通過時のみ。
- `U_SELF`: 対象市場を検算入力にしたbit一致テスト。評価armではない。
- `U_ANCHOR`: 2018年以前だけでcrux_joint 9時版の方向・同桁を再現。
- `U4_PRE_UMAREN`: historical_pre_snapshot UMAREN。M2基準候補。
- `P1_IDENTITY_PERMUTE`: 同一race内でsource-poolの馬identityを置換し、オッズmultiset・頭数・target marketを保持。
- `P2_TIME_SHIFT`: 同競馬場×頭数帯×人気集中帯でsource snapshotを別raceへ置換。
- `P3_SYNTHETIC_COHERENT`: 同一潜在PLから全プールを生成した負対照。
- `P4_TARGET_NULL_RESAMPLE`: pipeline sanity専用。γ・a・βはrealのY-1以前fit値に固定し、Yのoutcomeだけをm_calから200回再標本化する。Gate証拠には使わない。
- `P5_UNIFORM_SCORE`: q_LOPOをrace内一様分布へ置換し、市場温度補正だけで生じる改善を測る。real ΔがP5改善側97.5 percentileを超えることをM1必須条件にする。

最良armを結果で選ばない。Stage 0で固定した階層順に検定し、複数比較補正を行う。U3が識別不能ならU1/U2だけで判定する。

## 8. placeboと漏洩防止

- P1/P2/P5は最低200 draw、real improvementが改善側97.5 percentileを超えること。P2 matchingの人気集中帯はTANPUK entropyで定義する。P4はsanity controlでGate証拠に数えない。
- source poolとtarget poolの同一時点行をrace_idで厳密に結合し、日付ずれ・馬番ずれ・取消集合差をfail-closed。
- target UMAREN列をdropした入力からモデル行列を構築し、列名・hashをmanifest化。
- target outcomeを変更しても、fit前の市場確率・race set・tomography出力がbit一致すること。
- 未来年削除で過去年出力がbit一致すること。
- 行順・馬番順置換不変性。
- 同じレースをticket行に展開しても、有効標本数はrace数またはmeeting-day数とする。

## 9. 経済的解釈

市場loglossをわずかに上回ることと、控除後に利益が出ることは別である。控除22.5%に対し、全額比例投入には`-log(1-0.225)=0.255 nats`を超える情報差が必要になる。選択的参加では必要量が変わるため、Stage 0で推定ノイズを含む成長シミュレーションからPRACTICAL floorを固定する。

Gate M1/M2を通過しても、ROIや利益を主張しない。M2 PASS-PRACTICAL後にだけ、別番号・別spec・未使用期間で候補生成と決済評価を提案できる。

## 10. 停止規律

次のいずれかで終了する。

1. 既存研究と同値。
2. 対象プールが入力へ混入する。
3. provenance、時点、組合せ列割当を確定できない。
4. coverage floor未達。
5. power未達。
6. Gate M1 FAILまたはPASS-SIGNAL。
7. Gate M2 FAILまたはPASS-SIGNAL。
8. 結論がde-vig方式、少数年、特定競馬場、少数レースに依存。

FAIL後に券種、人気帯、モデル、floor、期間を選び直して救済しない。ワイド・馬単・三連系への変更は新番号とする。

## 11. 結論の射程

FAILしても「JRA全市場は効率的」「共同分布は無意味」「馬券で勝てない」と一般化しない。閉じるのは、監査を通過した入力プール、対象券種、期間、de-vig、tomography族に限る。

PASSしても「利益が出る」とは言わない。terminal市場に対するproper-score残差と、判断時点でのactionabilityを分けて報告する。

## 12. Stage 0成果物

- `PRIOR_ART_EQUIVALENCE_AUDIT.md`
- `MARKET_POOL_PROVENANCE.md`
- `POOL_SCHEMA_MANIFEST.json`
- `RACE_AND_TICKET_COVERAGE.json`
- `TOMOGRAPHY_IDENTIFIABILITY_AUDIT.md`
- `POWER_AUDIT.md` / `out/power_audit.json`
- `COMPUTE_DRY_RUN.json`
- synthetic oracle/invariant tests
- 改訂済み`SPEC.md`、`spec.json`、`README.md`

Stage 0成果物をコミット後に停止し、Fableの凍結判断を待つ。
