# Stage B データ可用性監査 (2026-09-20)

実行順の1番目 (ユーザー指定)。競走結果・払戻を見て評価設計を変えていないことの記録。

## 1. 市場データ (T-35相当) は2023-2025に実在する

`LOCKED_PERIOD_AUDIT.md` (EXP05既存監査) の通り、TANPUKアーカイブ
(`data/Time _series_odds/TANPUK_20210105-20251228.csv`) が「発走31-40分前(pre)の
overround除去済み市場確率」を2021-01-05〜2025-12-28まで提供している。2026年のみ
このアーカイブが途絶している(EXP05-Fが新規収集を必要とした理由そのもの)。
よってStage Bの対象期間(2023development/2024-2025主評価)には**実データの
T-35相当市場確率が存在する**(プロキシではない)。

## 2. 既存の時点安全設計行列をそのまま使う

`data/_research/mcond/exp05_design.parquet`(477,638行、2016-2025、EXP05既存生成物)に
以下が既に含まれている:
- `train`/`sel`列: train=2016-2021、sel=2022の真偽マスク(EXP05のtrain/sel split)
- `mkt_p3_pre`/`mkt_pi_pre`: TANPUK由来のpre市場top3/win確率(overround除去済み)
- `lp_cal_top3`/`f_mkt`: M0-M3が使うlogit変換済み特徴
- `top3`/`win`/`fin`/`fpay`: 結果ラベル・払戻(support構築には使わない、prediction error計算にのみ使う)
- `c1__場所__*`/`c1__芝・ダ__芝`/`c1__距離`/`c1__クラス名_ord`/`c1__出走頭数`: レースメタデータ
- `rank_mkt_pre`: 人気順(市場オッズ順位)

## 3. 【重要な決定】AI予測にはEXP05-F凍結モデルではなくEXP05自前のtrain/sel-fitモデルを使う

EXP05-Fの`frozen_model.joblib`(`freeze_manifest.json`: `n_rows_final_fit: 477638`)は
`exp05_design.parquet`と同じ行数であり、**2016-2025全期間を含む最終fit**である
(本番運用向けの設計として正しい)。これをそのまま2024-2025の「主評価」に使うと、
そのモデル自身が2024-2025を学習に使っているため真のOOSにならない
(Stage Bが検定したい「Jevの確率が予測誤差を説明するか」という問いにとって
決定的な瑕疵になる)。

代わりに、`analysis/mcond/exp05_market_residual_dev/models.py`が既に実装している
train(2016-2021)のみでfit・sel(2022)でハイパラ選択するM0-M5モデル
(`model_compare.csv`で`confirm_2023`/`oos_2024`/`oos_2025`として既に評価されている、
このプロジェクト既存の検証済みOOS設計)を再利用する。対応関係:
- EXP05のM1(`lp_cal_top3`+`f_mkt`) ≈ Jev state の m1_top_prob 相当
- EXP05のM3(M1+v6_score+rank+gap×2+percentile+dispersion) ≈ m3_top_prob 相当
- EXP05のM4(offset(M3)+F_serve残差回帰) ≈ m4_top_prob 相当 (EXP05-FのM4と同一アーキテクチャ、
  fit方法だけがtrain/sel限定で真にOOS)
これは新しいモデルを作るのではなく、既存の検証済みコードをtarget_col="top3"で
呼び出すだけ(`M.fit_m0_m3`, `M.fit_offset_residual`)。

## 4. レース単位の集約定義

Jevのstateはレース単位(馬単位ではない)。各レースについて「M4確率が最大の馬」を
そのレースの代表候補(candidate)としてEXP05-F既存の`predict_and_store.py`の
`top_i = argmax(p_m4)`と同じ選び方を踏襲する:
- `m1/m3/m4_top_prob` = candidateのM1/M3/M4確率
- 各モデルのentropy = そのレース全頭にわたる各モデルの確率分布のシャノンentropy
- `model_rank_disagreement` = M1/M3/M4のもとでのcandidateの順位の分散(0なら全モデル一致)
- `model_prob_variance` = [m1_top_prob, m3_top_prob, m4_top_prob]の分散
- `market_prob` = candidateの`mkt_p3_pre`
- 市場entropy = そのレース全頭の`mkt_p3_pre`分布のシャノンentropy
- `ai_market_divergence` = log(candidateのm4_top_prob / market_prob)
- `popularity_band` = candidateの`rank_mkt_pre`を3値帯(1-3/4-6/7+)に区分
- `feature_missing_rate`/`unknown_category_rate` = candidateについてF_serve列の
  NaN率、および各カテゴリ変数グループ(場所/芝ダ等)で全one-hot列が0の割合

## 5. 対象レース数 (2023/2024/2025、時点安全条件を満たす全適格レース)

`n_field>=5 & mkt_p3_pre notna & v6_p3 notna & mkt_pi_pre notna`
(EXP04/EXP05と同一フィルタ)適用後、レース単位(rid16ユニーク)実測: **2023年=3,456件、
2024年=3,453件、2025年=3,455件**(いずれも`data/_research/mcond/exp05_design.parquet`から
直接カウント、2026-09-20実施)。詳細な費用試算は`stage_b_dry_run.py`の出力を参照。

## 6. api_conformance_amendmentとの関係

この監査自体は競走結果・払戻・ROIを一切見ていない(結果ラベルの列名の存在確認のみ、
値は読んでいない)。support指標の構築(次節)も2023年の結果ラベルを一切使わない。

## 7. race-state vector構築の実データ検証 (2026-09-20)

`historical_state.py`で2016-2025全期間(34,545レース)のrace-state vectorを実際に構築し、
以下を確認した:
- M1/M3/M4相当確率・entropy・市場確率・divergence等、全ての連続値フィールドで欠損0件、
  分布は妥当な範囲(m1/m3/m4_top_prob平均0.67-0.68、ai_market_divergence平均-0.026等)。
- 実装当初、one-hotグループの検出ロジックにバグがあり、スカラー特徴65列
  (`c1__年齢`等、"__"を1回だけ含む列)を誤って単一の偽グループ"c1"に束ねていた
  (`unknown_category_rate`が全レースで一律0.333333になっていたことで発覚)。
  "__"が2回以上出現する列だけを真のone-hotグループとして扱うよう修正した。
- 修正後さらに検証したところ、「場所」one-hotグループ(9列)を根拠にした
  `unknown_category_rate`計算は依然として誤りだった: 実データで平均10.8%が
  「全列0=不明」と判定されたが、これは9列に含まれない**中京競馬場**
  (JRA中央10場のうち`c1__場所__*`に含まれない1場)のn-1ダミー参照カテゴリであり、
  真の未知カテゴリではなかった(10.8%というシェアはJRA10場のうち1場が担う典型的な
  開催割合と整合)。他のone-hotグループ(天気・馬場状態等)も同様にn-1ダミー方式
  (最頻値を参照カテゴリとして削除)の疑いが強く、F_serveの保存済み one-hot 列だけからは
  「真に未知」と「単に参照カテゴリ」を区別する情報が失われている。
  **結論**: `unknown_category_rate`は歴史データセットからは根拠を持って再構成できないため、
  歴史評価では一律0(算出不可)として扱う(spec.jsonの`support_metric_definition`に記録)。
  この指標が意味のある変動をするのはEXP05-Fの前向きシステム(frozen_encode.pyが
  エンコード時点で真の未知カテゴリを検出・記録する)においてのみである。
- `in_distribution_support`/`similar_past_case_count`をood_support.SupportModelで
  2023年fit→2024/2025年score した結果、`in_distribution_support`の分布が概ね一様
  (2024: mean=0.503, std=0.291; 2025: mean=0.496, std=0.287)であることを確認した。
  これは2024/2025が2023と同じ母集団から生成されていれば理論的に期待される挙動であり、
  パイプラインが正しく機能していることの傍証とする。
