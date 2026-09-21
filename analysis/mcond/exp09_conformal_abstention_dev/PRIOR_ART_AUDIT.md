# EXP09 Stage 0 — 先行研究・既存実装監査

**作成日**: 2026-09-21　**方法**: リポジトリ全体をキーワード検索
(conformal/selective prediction/risk-coverage/abstention/prediction set/OOD/
coverage gate/chaos gate/confidence gate/LR_CONTROL/Q5等)し、ヒットしたファイルを
実際に読んで検証(メモリ要約だけに頼らない)。**コード変更・学習・バックテスト・
2024/2025結果開封は一切行っていない。**

## 総合判定

**「厳密なConformal」は未着手。しかし極めて近い要素技術(2023-only fit・OOD距離
指標・same-coverage比較・LR_CONTROL)が既に3系統独立に実装・検証されており、
うち1系統(participation gate)は最重要な数値[+5.31pt]が2026実績で符号反転して
撤回済み、もう1系統(EXP06 OOD/Jev)は「素朴なOOD/不確実性シグナルは、モデル自身の
確信度を統制すると価値が消える」という直接の反証を既に持つ。EXP09を実施する
価値は残るが、この2つの反証パターンを踏まえた設計が必須。**

## 1. Conformal/selective prediction/risk-coverage/abstention/prediction set/OOD/coverage gateの直接該当

`conformal`という語のヒットはゼロ(未着手を確認)。近縁概念は以下の3系統に分散して存在:

### 1a. 本番の"hard gate"(`production_policy.py`)— 固定閾値の見送り、実施済み
- `hard_skip_reasons()`: (1) `field_chaos_score`の**パーセンタイル**(凍結分位表
  `data/chaos_quantiles.json`に対する線形補間)が`skip_percentile`(現在0.667)以上、
  (2) `field_size<=7`、(3) ◎の`tansho_odds`欠損、(4) ◎の`p_win<0.05`、の4条件で
  レース単位skip。`validate_cowork_bets.py`もこの関数へ委譲(単一ソース)。
- 分位表(`build_chaos_quantiles.py`)は**2026年の実serve bundle**(`reports/cowork_input/*_bundle.json`)
  から構築(`reference_id="serve34-forward-policy-20260825"`)。**2023年developmentのみで
  fitしたものではない**。
- **分類: 固定閾値の見送りとして実施済み。厳密なConformalではない**(coverage保証・
  exchangeability前提・キャリブレーション誤差の定量化がいずれもない、単なる
  経験分布のパーセンタイルカット)。

### 1b. 参加ゲート実証ツール(`participation_analyzer.py`)— 2023-only fit・同一coverage類似の実施済み
- v6較正確率から`dom1`(◎独走度)・`conc`(上位2集中)・`entropy`(混戦度)を計算、
  **valid(2023)で閾値(75/25パーセンタイル)を決定→test(2024-2025)で固定評価**
  (再fitなし)。◎複勝で最良ゲート(`conc>=0.564`)がtest ROI 90.9% [88.4,93.5]
  (全レース参加85.6%比+5.31pt)。
- **分類: 固定閾値の見送りとして実施済み、かつ2023-only fit→2024-25固定評価の
  基盤として直接再利用可能**(item4の答え=YESの実例)。ROIのみ見た実験でもある
  (的中率やlogloss等の情報量指標は見ていない)。

### 1c. EXP06のOOD距離指標(`ood_support.py`)— 最も厳密、2023-only fit・凍結パラメータ
- `in_distribution_support`(2023年参照集合への20-NN距離のECDF逆変換)・
  `similar_past_case_count`(半径内近傍数)を、**2023年のみでPCA(累積寄与率95%,
  最大20次元)・ロバスト標準化・one-hotをfitし、2024-2025は変換のみ**(再fitしない)
  で計算。距離関数・k・半径は結果を見て変更しない設計。
- **分類: 「厳密なConformal」ではないが、conformal的なexchangeability前提付き
  距離ベースOOD指標として実施済み**。ただし目的はJev(LLM判断層)への入力特徴量
  であり、OOD指標**単体**の予測誤差抑制力は間接的にしか検証されていない
  (下記2参照)。

## 2. participation gateの「+5.31pt」— 算出方法・対象期間・リーク安全性・**再現性(重要)**

- 算出: `participation_analyzer.py`。v6較正確率(`models/pl_calibrators_v6.pkl`)から
  信頼度指標を計算、valid(2023)で四分位/閾値決定、test(2024-2025)で
  bootstrap CI付き検証。データソースは`data/_ev_grid_scores.parquet`(事前計算済み
  score cache)。
- **リーク安全性の細部**: 閾値選択に使う信頼度指標(dom1/conc/entropy)は
  `pl_calibrators_v6.pkl`(**2023年でfit**)による較正確率から計算される。この
  calibratorをvalid(2023)自身の閾値決定に使うのは形式的にはin-sample的だが、
  Isotonic回帰は単調変換のため**パーセンタイル(順位)ベースの閾値選択には実質
  影響しない**(順位はraw scoreでも較正後確率でも同じ)。実害は小さいと判断する。
- **★2026-07-23、+5.31ptは2026 as-served実データで符号反転して撤回済み**
  (`analysis/reverify_clean_band_2026.py`、686R・21開催日・4/26-7/19)。
  クリーン帯(pct≤0.33)◎複勝ROI=77.6%[65.0,89.9] **<** 帯外=87.9%[77.5,99.0]、
  Δ=**−10.31pt**(2024-25実測+5.31ptと逆方向)。的中率の優位(clean54.6%>chaotic46.6%)
  は残るが、ROI面の優位は消滅・逆転。本番の二段化配線(`compute_bets.py`の
  `force_floor`/`demote_budget`機構)は**この結果を受けて配線中止のまま**
  (コード自体はコメントアウトでなく残置、`main()`から`demote_budget`を渡さない
  形で無効化)。
- **今回の研究への意味**: +5.31ptという点推定は**単一期間・単一手法の産物であり
  regime-stableではない**ことが実測で確定している。EXP09で同種の「信頼度で
  見送る」設計をする場合、(a) 2024-2025だけでなく可能なら2026 as-served相当の
  第三の独立期間でも方向確認する、(b) 点推定のROIだけで判定せず的中率/logloss等の
  情報量指標も併記する、(c) 単一の点推定を過信せず前掲の[[project_unwired_roi_audit]]
  の教訓「配線より最新期間でのCI付き再検証」を踏襲する、ことが必須。

## 3. chaos gate・confidence gate・欠損率gate・人気帯gate・LR_CONTROL・Jev Q5との重複

| 既存要素 | 内容 | EXP09との関係 |
|---|---|---|
| chaos gate(§1a) | 混戦度percentile skip | 単純max-confidence的gateと概念重複。EXP09の比較対象「現行participation gate」そのもの |
| confidence gate(§1b: dom1/conc) | ◎独走度・上位2集中 | EXP09の「単純な最大予測確率gate」と概念的に同型(較正確率の順位に基づく) |
| 欠損率gate | `feature_missing_rate`はEXP06 `ood_support.py`のCONTINUOUS_COLSに含まれる(distance入力の一部)が、**独立のgateとして単体評価されたことはない** | EXP09の「feature-missing gate」は**未実施**、新規性あり |
| 人気帯gate | `popularity_band`はEXP06のCATEGORICAL_COLSに含まれる(distance入力の一部)。独立gateとしての単体評価なし。過去の別研究`race_selection_confidence_place.md`は「自信度で絞るほどROI悪化」で**2026 serveで符号反転・失効**タグ済み | 同じくexchangeability崩壊の実例、item6に直結 |
| **LR_CONTROL**(EXP06 `stage_b_gate_eval.py`) | **2023年developmentのみでfit**した単純ロジスティック回帰(目的変数=2023年内Brier中央値超の2値)。Jevの実coverageに合わせた同一coverage比較で**Jevに完勝**(Jev選択698件Brier0.219 vs LR_CONTROL選択698件Brier0.107、年度/競馬場/人気帯/芝ダート/距離帯/support quintileの全区分で一貫) | **EXP09が要求する「2023年だけで学習したLR gate」は、EXP06にほぼそのまま使える実装が既に存在**(`_fit_lr_control()`)。ただし目的変数は「2023年内Brier中央値超」で、EXP09が実際に測りたい量(top3的中率やROI)とは異なる可能性がある点に注意 |
| **Jev Q5**(`risk_prob`=P(PASS_UNCERTAIN)+P(PASS_OOD)) | LLM判断層の5択質問からの不確実性シグナル | Gate1(単純相関)ではPASSだったが、**Gate2(m4_top_prob等の予測確率そのものを含む全変数統制)ではFAIL**(係数95%CIがゼロを跨ぐ、年度間で符号も不一致)。「素朴な不確実性シグナルは、モデル自身の確信度(m4_top_prob等)を統制すると消える」という**直接の反証パターン**。EXP09のGate2A/2B的な設計(単純比較→完全統制比較)を最初から要求する根拠 |

## 4. 2023年だけで較正・2024/2025固定評価できる基盤

**存在する**。3系統全てが既にこの設計:
- `participation_analyzer.py`: valid(2023)で閾値→test(2024-25)固定。
- `ood_support.py`: `SupportModel.fit(df_2023)`→`.score(df_query)`(再fitなし)。
- `stage_b_gate_eval.py::_fit_lr_control()`: 2023developmentのみでfit。

EXP09はこれらのコード**自体**を改変・再利用してはならない(絶対条件:
EXP01-08・v6・compute_bets.py・EXP06は無変更)が、**設計パターン**(2023 fit →
2024-25 frozen apply)は既に3例で実証済みであり、EXP09独自実装でもこの型を
踏襲すればよい。

## 5. レース単位・馬単位・馬券単位のどこで棄権するのが数学的に妥当か

- 本番のhard gate・participation gate・EXP06のLR_CONTROLは**すべてレース単位**
  (◎馬1頭を代表として評価し、レース全体を参加/見送りに二分する)。
- 馬単位の棄権(特定の馬だけを予測対象から除外)は、CLAUDE.mdのPlackett-Luce
  全馬確率同時推定という設計と相性が悪い(1頭除外しても他馬の相対確率は再正規化
  が必要、かつ馬券(馬連・ワイド等)は複数馬の組合せのため「1頭だけ棄権」は
  馬券単位の意味を失わせやすい)。
- 馬券単位の棄権(EXP07のno-bet/exposure調整に近い)は、EXP07で「Stage 2Bは
  no-bet比較を含む」と設計されていたが実施は保留のまま終了。
- **数学的に最も自然なのはレース単位**(既存3系統と整合、Plackett-Luce同時推定の
  単位と一致、馬券構築の前段階として位置づけやすい)。EXP09もレース単位棄権を
  既定とし、item2のROI再現性問題を踏まえ**的中率/logloss等の情報量指標を主指標、
  ROIは副次**とする(EXP08のGate2A/2B設計を踏襲)。

## 6. 交換可能性(exchangeability)が年度・競馬場・芝ダートで崩れていないか

**崩れている実例が複数確認された**。Conformal predictionの妥当性はcalibration
setとtest setの交換可能性に依存するが、以下はいずれもこの前提を崩す既知の
regime shift:

1. **calibrator差し替えでchaos分布が丸ごと平行移動**([[project_calibrator_shifts_chaos_gate]])。
   2026-08-24のserve calibrator差し替え実測: chaos平均0.8909→0.8201、
   chaos>=0.92のレース233R→58R、見送り34.5%→10.3%。**固定閾値は
   calibratorが変わるたびに意味を失う**(新規に買われた170Rの◎複勝的中48.8%
   <全体53.4%=閾値の意味が崩れた状態で参戦が増えた)。
2. **participation gateの符号反転**(§2)。2024-25 backtestと2026 as-servedで
   方向が逆転=時間軸でのexchangeability崩壊の直接証拠。
3. **v6のカテゴリエンコーディング不一致**([[project_v6_category_encoding_mismatch]])。
   本番servingで芝・ダ特徴が学習時と異なる表記になり毎週`__NaN__`化していた
   (2026-09-19修正)。**train分布とserve分布がバグにより一時的に別物になっていた**
   実例。
4. **race_selection_confidence_placeの符号反転**([[project_race_selection_confidence_place]])。
   「自信度で絞るほどROI悪化」という2026serveでの符号反転が別途タグ済み。

**結論**: この本番システムでは、モデル・calibrator・特徴量エンコーディングが
非定期的に変更されており、**2023/2024/2025という年度区分をまたぐ交換可能性は
無条件には成立しない**。EXP09で正式なconformal coverage保証を主張する場合、
(a) 固定モデル・固定calibrator版で完結する期間内(例: v6は2013-2022 train、
2023 valid、2024-2025 test で一貫)に限定して主張する、(b) 保証の射程を
「このモデル・calibratorの版が変わらない限り」と明記する、(c) 版が変わった
場合は無条件に再calibration必須と設計する、のいずれかが要る。

## 7. 既存の確率較正器を再利用するとin-sampleになる箇所

- `pl_calibrators_v6.pkl`は**valid=2023でfit**(EXP07で確認済み)。EXP09がこの
  calibratorの出力(確率)を使ってconformal/coverage閾値を**2023年で**決める場合、
  較正器のfit元データと閾値決定データが同一になり**in-sample**。EXP07 Gate J1と
  同じ問題。
- 対処法もEXP07で確立済み: 2023を前半(H1)/後半(H2)に時系列blocked splitし、
  H1でIsotonic較正器を新規fit→H2で評価、または(較正器自体は2023全体でfitした
  ものを使うにしても)conformal閾値の決定は較正器のfit元と重ならない別のholdout
  (例えばH2のみ)で行う必要がある。
- EXP08の`expected_time_model.py`と同型の「expanding-window/leave-year-out」の
  考え方もここに適用可能(較正器バージョンごとにfit期間を明確に分離する)。

## 8. 本番の判断時点で利用可能な入力だけで再現できるか

- `ood_support.py`のdistance入力(m1/m3/m4確率・entropy・市場確率・市場entropy・
  AI市場乖離・頭数・場所・芝ダ・距離帯・クラス帯・人気帯・特徴欠損率)は、
  EXP06で「結果・着順・払戻を一切使わない」設計が明記されており、**市場確率
  部分を除けば本番のT-10/判断時点で概ね再現可能**(市場確率はEXP07で確立した
  historical_pre_snapshot方式を踏襲すれば時点安全)。
- `participation_analyzer.py`のdom1/conc/entropyはv6較正確率のみから計算され、
  **判断時点で完全に再現可能**(市場情報すら不要)。
- chaos gate(§1a)は本番稼働中そのものなので当然再現可能。
- **結論**: 主要な入力候補はいずれも判断時点で再現可能。市場系の入力を使う場合は
  EXP07/EXP08で確立したhistorical_pre_snapshot(発走26-30分前、中央値28分前)の
  規約を踏襲すること。

## Stage 0結論(ユーザー指定の分類)

| 分類 | 該当 |
|---|---|
| 厳密なConformalとして未着手 | ✅ 該当(exchangeability前提の明示的検証・coverage保証の数式的主張はゼロ) |
| 固定閾値の見送りとして実施済み | ✅ 該当(chaos gate、participation gate) |
| 同一coverage比較まで実施済み | ✅ 該当(EXP06 Gate3、Jev vs LR_CONTROL vs OOD-only) |
| ROIだけを見た過去実験 | ✅ 該当(participation_analyzer.pyはROIのみ、情報量指標なし) |
| +5.31ptの再現性 | ❌ **2026 as-servedで符号反転、撤回済み。今回の研究へ「参照値」としては使えるが「実証済みの成功例」としては使えない** |

固有仮説(EXP09が新規に問うべきこと)は残る: (a) LR_CONTROLはEXP06で
「Jevより優れている」ことは示したが「単純な最大確率gate/entropy gateより
優れている」かは未検証、(b) 厳密なconformal coverage保証(exchangeability
前提の明示・有限標本妥当性)は誰も試みていない、(c) feature-missing gateは
単体評価されたことがない、(d) 情報量指標(的中率/logloss)を主指標とした
同一coverage比較(ROIでなく)は未実施。

## 次の一手(Stage 0完了、実装前)

最小反証実験と停止条件を提示する(別途)。
