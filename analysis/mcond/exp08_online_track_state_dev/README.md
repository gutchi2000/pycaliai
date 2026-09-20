# EXP08 — 当日馬場状態オンライン推定AI

## 絶対条件

EXP01〜EXP07・v6・EXP05・EXP05-F・compute_bets.pyは変更しない。本番の印・買い目へ
接続しない。実結果を見て状態変数・半減期・Gateを変更しない。対象レースより後の
結果を使わない。2024〜2025年を完全未使用期間と呼ばない。ROIを主目的にしない。
他セッションの未コミット変更(`analysis/day_state_counting*.py`,
`analysis/day_waku_z_forensics.py`等)を巻き込まない(読み取り専用で参照)。

新規作業は`analysis/mcond/exp08_online_track_state_dev/`へ限定する。

## スコープ(2026-09-20夜、ユーザー承認済み・確定)

`PRIOR_ART_AUDIT.md`§6の推奨をユーザーが承認。以下へ確定し、以降は変更しない:

- **主仮説**: 時計(走破タイム残差)・上がり性能・ペースの当日オンライン状態のみ
- **内外(枠)・脚質(前残り/差し)**: 採用候補から除外、negative control専用
  (再救済しない。時計・上がり・ペースいずれにも追加情報がなければEXP08を終了する)
- permutation placebo testをGateに事前登録(`spec.json` `permutation_placebo_gate`)
- 2023年developmentのみで状態定義・パラメータ・閾値を固定
- RAW/EWMA比較必須

詳細な数理定義・凍結値はすべて`spec.json`に記録する(2026-09-20夜、Stage3着手前に
初回コミット、以後2024-2025結果を見て変更しない)。

## 現在の進捗(2026-09-20夜時点)

| 段階 | 状態 | 成果物 |
|---|---|---|
| Stage 0(先行研究・既存実装監査) | **完了** | `PRIOR_ART_AUDIT.md` — 内外(枠)は`day_state_counting_stage2.py`(別セッション、未コミット、参照のみ)でpermutation placebo失格により却下済み、脚質も却下済み。時計・上がり・ペースのみ先行研究なし |
| Stage 1データ可用性監査(§5) | **完了(訂正済み)** | `DATA_AUDIT.md` — 走破タイム/上がり3F/PCI・RPCI等は外部`kekka_2010_2025_fix_raceid_v2__keyed.csv`で全年安定して取得可能。**結果利用可能時刻の根拠を全面訂正**(下記参照): TANPUK確定オッズは結果レコードの取得時刻の代理にならないと判明、JV-Link等にも信頼できる時刻が存在しないことを確認、保守的な固定遅延(+20分主解析/+30分感度分析)へ変更 |
| Gate 0 | **PASS(9/9条件)** | `DATA_AUDIT.md`末尾+`spec.json` `gates.gate0_data_health`。9条件目(利用可能時刻の行単位不変条件)はイベント駆動アルゴリズムの設計上数学的に保証され、`test_online_state.py`で直接検証済み |
| Stage 2(観測信号構築、期待値モデル) | **完了(expanding-window化・venue追加・agari次元追加済み)** | `expected_time_model.py`(距離50m×**競馬場**×芝ダ×コース区分×公表馬場状態×クラス名の中央値lookup、**年単位expanding-window/leave-year-out**、2023年以降は2022年末までの単一モデルに固定)、`build_observations.py`(レース単位**speed_signal・agari_signal・pace_signal**の3独立信号 + negative control用inside_signal/front_signal、利用可能時刻列付与)、`test_build_observations.py`(9テスト)、`test_time_safety.py`(7テスト、削除不変性・利用可能時刻チェッカー含む) |
| Stage 3(Kalman状態空間モデル・M0-M3比較・placebo) | **コード完成・テスト完了・コミット直後** | `online_state.py`(Kalman本体+RAW/EWMA対照、13テスト)、`build_features.py`(M0-M3特徴量構築)、`evaluate.py`(offset付きロジスティック回帰・meeting-day bootstrap・permutation shuffle、11テスト)、`stage3_run.py`(Gate評価ドライバ)。2024-2025実評価は本コミット後に実行する |
| spec.json | **凍結・コミット(Stage3分も追記)** | 観測信号定義・利用可能時刻ルール・期待値モデル設計・permutation placebo定義・状態空間モデル定義(Q/R選択値・EWMA半減期・interaction定義・M0-M3定義)・Gate定義・中止規律を全て記録 |

## 【2026-09-20夜、ユーザー指摘による訂正】結果利用可能時刻の根拠

Stage1完了時点の初版は、TANPUK区分4(確定オッズ)タイムスタンプを「競走結果(着順・
走破タイム・上がり・通過順)が利用可能になった時刻」の実測値として扱い、これを
根拠に固定遅延15分を凍結していた。**これは誤り**: TANPUK確定はオッズ・払戻の
確定時刻であり、結果レコード自体の取得・公開時刻ではない。

再監査の結果、**JV-Link・TARGET・保存済みデータのいずれにも、結果レコード自体の
取得/公開/更新時刻を示す信頼できるタイムスタンプは存在しない**ことを確認した
(JV-Link RACE(RA)レコードの「データ作成年月日」はレース前の番組表発表日で別物、
`jvlink_results.py`のfetchedはローカル実行時刻でJRA-VAN発行のものではない)。

したがって「実測」ではなく保守的な仮定へ変更: **主解析=発走時刻+20分、感度分析=
発走時刻+30分**(両者で効果方向が不一致ならFAIL)。行単位の不変条件
`prior_result_available_timestamp <= target_decision_timestamp`をStage3で全ペア
検査し、1件でも違反があればGate 0 FAILとする。詳細は`spec.json`
`result_availability_timing`、`DATA_AUDIT.md`§5.2参照。

## Stage 2実装中に発見・修正した実害バグ(5件)

外部結果ファイル(`kekka_2010_2025_fix_raceid_v2__keyed.csv`)を直接使うのは
本セッションで初めてで、以下の5件が数値を大きく歪める実害バグとして発覚・修正済み:

1. **コース区分がダートで構造的にNaN**(A/B/C/D等の回り設定は芝のみの概念)なのに
   dropnaで**全ダートレースが消えていた**。"D_NA"カテゴリで埋めて修正。
2. **障害(ジャンプ)レースがクラス名を平地と共有**("未勝利"等)しており、
   距離bucketの中央値が障害の遅いタイムで歪んでいた。レース名の"障害"部分文字列で除外。
3. **着順が1-9着=全角数字/10着以降=半角数字という不均一エンコード**で、
   `pd.to_numeric`へ直接通すと**勝ち馬を含む上位9頭が解析から消えていた**
   (value_countsで10-18のみ検出、1-9が0件という形で発覚)。全角→半角正規化して修正。
4. **枠番(1-8固定)を内外percentileの分母に**使っており、大頭数レースで
   `is_outer`の閾値が非現実的になりinside_signalが全レースNaNになっていた。
   馬番(1〜頭数、個別)へ変更して修正。
5. **外部kekkaの"race_id16"列がmaster_v2と別スキーマ**(uu/pp/yy/k/n/rr起源の内部
   再採番)で、結合が全件失敗していた。無印の"race_id"列がmaster_v2の
   「レースID(新/馬番無)」と一致することを実データで確認し、以後"rid16"へ
   リネームして統一。

いずれも実データで実際に検証(value_counts/describe/sample行確認/結合成功率)して
発見。`test_build_observations.py`に5件とも回帰テストとして固定化。

## Stage 3実装の要点

- **状態単位**: 開催日×競馬場×芝ダート。時計・上がり・ペースを独立なKalman状態
  として持つ(内外・脚質はnegative control専用、状態化しない)。
- **イベント駆動アルゴリズム**: 観測利用可能イベントと判断(スナップショット)
  イベントを時刻順にマージして処理。「利用可能時刻より後の観測が対象レースの
  スナップショットに混入しない」ことがアルゴリズム設計上数学的に保証される
  (`test_online_state.py::test_no_future_leakage_*`で直接検証)。
- **Q/R/prior_var**: 2023developmentのone-step-ahead尤度で次元ごとに選択。
  primary(+20分)とsensitivity(+30分)は数学的に同一の選択結果になる(固定遅延の
  一様シフトは観測間の相対時間間隔を変えないため、実測でも完全一致を確認)。
- **EWMA半減期**: 2時間で事前固定(ラウンドナンバー、2023結果を見て決めていない)。
- **interaction**: horse_aptitude(closing_power、既存serve-safe特徴量)×Kalman
  状態平均。欠損馬は0へ縮約(行を除外しない)。
- **M0-M3/RAW/EWMA/ZERO**: offset付きロジスティック回帰(baseline_logit_top3=
  v6+市場を固定offset、追加特徴量だけL2正則化でfit)。2023development でfit、
  2024・2025はgenuinely OOS評価(再fitしない)。
- **permutation placebo**: 競馬場×日×芝ダート内で観測値の対応関係だけをシャッフル
  (タイムスタンプ・欠損位置・レース数・baseline予測は不変)。2023でfitした凍結
  betaをシャッフル後データへ適用し、実信号のΔloglossがplacebo分布の2.5パーセン
  タイル(=改善方向で97.5%点)を下回るか検定。>=1000回。

在庫バグ発見: build_feature_table構築中、race-levelの状態特徴量(RAW/Kalman平均)は
定義上レース内で全馬同一値となるため、素の加法項では「レース内の相対順位」への
寄与が原理的に小さい(v6+市場の合計確率が既にレースあたり≈3.0付近へ強く較正されて
いるため尚更)。この構造上の制約はM3のinteraction項(馬ごとに値が変わる)で部分的に
緩和される設計だが、2023年in-sampleサニティチェックでも相関はごく小さいことを確認
済み(バグではなく、モデル仕様として想定内の挙動)。詳細はspec.json
`state_space_model_stage3.model_definitions`参照。

## 次の一手

commit後、`stage3_run.py`を実行して2024・2025年でgenuinely OOS評価する
(fit済みbetaは再fitしない)。Gate2A(M1 vs M0)・Gate2B(M3 vs M1)・permutation
placebo・meeting-day bootstrap 97.5%CIの結果に基づき、spec.json `gates`の基準で
判定する。
