# EXP13 Gate 0B / Gate 0D 再開評価報告（2026-09-22）

**状態**: モデル学習・2024/2025年開封・ROI評価は一切行っていない。本報告は
ユーザー指示「まず次だけを報告してください: Gate 0B parity / 全starter確率
再計算 / market coverage差 / Gate 0D power / S0〜S5へ進めるか」への回答。

## 0. 再開の経緯

2026-09-22、市場データprovenance横断監査により、EXP13 Gate 0Aの「2023年
historical_pre_snapshot coverage 0%」は誤りと判明した
（[[project_market_provenance_audit_20260922]]）。`data/Time _series_odds/
TANPUK_*.csv`（2011-2025年）由来のhistorical_pre_snapshot（発走26-30分前、
中央値28分）が実在し、2023年flatで正常完走馬91.0%・止（DNF）87.6%の
coverageがある。ユーザー承認により、**EXP13はGate 0Aを訂正PASSとし、同一
実験番号のままGate 0Bから再開した**（新しい実験番号は使わない）。

---

## 1. Gate 0B: full-starter numeric parity

### 1.1 全120特徴の分類（詳細は`GATE0B_FEATURE_AUDIT.md`）

| verdict | 件数 | 内容 |
|---|---:|---|
| structurally_immune（raw passthrough / asof key join / 既検証rolling） | 92 | TARGET輸出そのまま68 + jockey/trainer_fuku(既検証)4 + prev_pos_rel/closing_power(決定論的)2 + prev_hosei/prev_hosei9(asof)2 + trnH_*/trnW_*(asof)16 |
| likely_immune（同型パターン、数値未再検証） | 2 | horse_fuku10/30 |
| **confirmed_immune（実測済み）** | **4** | hist_same_cond_best_pos/top3_rate/count, hist_same_place_best_pos — 626,774行中diff=0 |
| **confirmed_divergent（実測済み）** | **19** | course_n_prev/win_rate/top3_rate, jockey_n_prev/win_rate/top3_rate（6）+ kako5_avg_pos/std_pos/best_pos/avg_agari3f/best_agari3f/same_td_ratio/same_dist_ratio/same_place_ratio/pos_trend/race_count/expected_good_count/hidden_good_count/same_cond_best_pos（13） |
| N/A（学習データに「人気」列が存在せず常時NaN） | 3 | kako5_avg_ninki, kako5_pos_vs_ninki, kako5_upset_good_count |
| **合計** | **120** | |

**根本原因（実測で特定済み、jockey/trainer_fukuの過去バグ修正と対称）**:
`parse_kako5.py`は`data/master_20130105-20251228.csv`（dropna**後**、
626,774行）を入力に馬ごとの直近5走ウィンドウを構築し、`build_master_v2.py
compute_history_features()`は同じdropna後の母集団でcourse/jockeyの
無制限累積カウントを行う。DNF走はこの入力に一度も現れないため、
①kako5系は「直近5走」の走数ウィンドウがDNF走の分だけ古い方向にずれる
（jockey/trainer_fukuの"window slot consumption"と全く同型、ただし
**dropna後**に作用する点が逆）、②course/jockey_n_prev系は無制限
expanding cumulativeのため、DNF走以降**恒久的に**経験数を過小カウントする。
hist_same_cond/place系だけが無罪だった理由は、`if isnan(着順): continue`で
着順**値**ベースにフィルタしており、DNF行（着順=NaN）は元々このフィルタで
除外される設計のため、母集団にDNF行があってもなくても結果が変わらないため
（window位置ベースの消費とは異なる機構）。

### 1.2 実測結果①: 数値特徴・raw score parity

`gate0b_course_jockey_history_parity.py`（course/jockey 6特徴）:

| 特徴 | 比較行数 | 不一致数 | NaN不一致 | 最大絶対差 |
|---|---:|---:|---:|---:|
| course_n_prev | 626,774 | 3,401 | 0 | 2.0 |
| course_win_rate | 276,771 | 1,051 | 781 | 0.667 |
| course_top3_rate | 276,771 | 1,802 | 781 | 0.667 |
| jockey_n_prev | 626,774 | 3,854 | 0 | 4.0 |
| jockey_win_rate | 298,839 | 1,626 | 500 | 0.667 |
| jockey_top3_rate | 298,839 | 2,505 | 500 | 0.667 |

`gate0b_kako5_history_parity.py`（kako5系13特徴、代表列）:

| 特徴 | 比較行数 | 不一致数 | 最大絶対差 |
|---|---:|---:|---:|
| kako5_avg_pos | 558,801 | 8,401 | 4.53 |
| kako5_best_pos | 558,801 | 1,513 | 13.0 |
| kako5_avg_agari3f | 500,299 | 10,024 | 16.4 |
| kako5_race_count | 626,774 | 3,843 | 2.0 |
| kako5_same_cond_best_pos | 465,082 | 1,014 | 16.0 |

（他8列は`out/gate0b_kako5_history_parity.json`参照、全て不一致数千件・実質差あり）

**raw v6 score parity**（`gate0b_raw_score_parity.py`、course/jockey 6特徴のみ
true-universe値へ差し替え、他114列は現行値のまま固定して既存学習済み
`unified_rank_v6.pkl`でスコアリング）:

- 2013-2025年、course/jockeyいずれかが不一致な行 = **6,453 / 626,774行（1.03%）**
- raw score差（affected行のみ）: 平均絶対差 **0.0220**、中央値 0.0128、
  p95 0.0790、最大 **0.326**
- 許容誤差1e-6を超える行: 5,168/6,453（80.1%）— 浮動小数点誤差ではなく実質差
- kako5系13特徴のraw score影響は計算コストの都合で個別定量化していないが、
  不一致行数が上記より多く（最大1.8%）、個々の特徴のモデル内寄与度も
  一般に条件特徴（course/jockey等）より過去成績集約特徴の方が大きい傾向が
  あるため、**course/jockey群で実測した影響（1.03%行・平均差0.022）を
  下回ることはないと想定するのが妥当**（過小評価しない方向の仮定）。

**判定基準への照合**: ユーザー事前固定の基準「raw score：許容誤差以内」
「不一致は列別・原因別に全件記録」に対し、**19/120特徴（15.8%）で実測
不一致を確認し、原因は完全に特定済み**（dropna後母集団での位置/カウント
ベース集計）。原因は説明できているが、数値は一致していない。

### 1.3 実測結果②: 全starterを含めた確率再計算 / ③ 既存馬への影響

DNF馬（2023年flat 145頭）について、raw passthrough 68列（cat/torch/add生CSV
から直接）+ jockey/trainer/horse_fuku・pace特徴（build_dataset.py実関数を
pre-dropna全体631,965行に再実行）+ prev_hosei/trnH/trnW（build_master_v2.py
実関数のasof結合）+ kako5/hist_same/course/jockey系（同ロジックを個別計算）
の**全120特徴を新規構築**し、既存の学習済みモデルでスコアリングした
（モデルは一切変更していない）。

- 145頭中 **133頭（91.7%）** で全120特徴の構築に成功（残り12頭は
  lgbm/cat/torch/add生CSVの結合キーが揃わずJOIN不能、数値を捏造せず欠落
  のまま記録）
- DNFが実在する**129レース**で、既存の完走馬のみのPL正規化確率と、
  DNF馬を分母に加えた全starter正規化確率を比較:
  - 完走馬の確率変化: レース平均の絶対差（レースごとの平均を更に平均）
    **0.68ポイント**、最大（1レース内の1頭の最大変化）**16.4ポイント**
    （15頭立てのレースでDNF馬のraw score(1.44)が中位以上だったケース）
  - 全starter確率の合計は全レースで1.0（正規化は正しく機能）
  - DNF馬自身のraw scoreは-4.21〜+1.86（平均-0.95、完走馬平均より
    低い側に偏る傾向はあるが個体差が大きい）

**結論**: 非完走馬をPL分母に正しく含めると、完走馬の確率は平均で無視できない
規模（0.68pt、最大16.4pt）だけ変化する。現行の本番運用（完走馬のみで
正規化）は、DNF馬がいたレースで他馬の勝率をわずかに過大評価している
可能性が構造的に存在する。

---

## 2. market coverage差（Gate 0Bの母集団定義・欠損取り扱い）

`gate0b_market_coverage_audit.py`（2023年flat started母集団、止含む・外消
除外、race_id結合）:

- 正常完走馬 coverage: 41,978/46,107 = **91.04%**
- DNF coverage: 127/145 = **87.59%**
- **coverage差 = 3.46pt、95%CI [-1.91, +8.83] — ゼロを跨ぐ（統計的に有意差なし）**
- 除外されるDNF陽性数（complete-case時）= 18、除外される完走馬数 = 4,129
- complete-case陽性率 0.3016% vs all-starter陽性率 0.3135%（差0.012pt、
  ほぼ同一 — complete-case制限自体が陽性率を大きく歪めるわけではない）
- missingness予測可能性: is_dnf単独では欠損を予測できない（chi2 p=0.190、
  非有意）。venue+人気帯+is_dnfのロジスティック回帰AUC=0.622 —
  欠損はDNF状態そのものよりvenue・時期（2023年5月に顕著な欠損スパイク、
  finisher 69.4%/DNF 66.7%——DNFと無関係な月次データ収集ギャップ）に
  構造化されている

**母集団定義（事前固定、結果を見て変更しない）**:
starter母集団はkekka着順コード基準（数値+止+丸数字、外・消は除外）とし、
historical_pre_snapshotの有無では母集団を変更しない（snapshot欠損馬は
「市場統制ありの変種」でのみcomplete-case除外、v6単体のraw score/全starter
確率再計算には無関係）。final oddsによる穴埋めは行っていない。

---

## 3. Gate 0D再評価（full-starter population + market snapshot complete-case）

`gate0d_full_starter_power.py`（S4想定特徴数=13、事前固定）:

| 期間 | all-starter陽性数 | all-starter EPV | complete-case陽性数 | complete-case EPV | 欠損による陽性減少 |
|---|---:|---:|---:|---:|---:|
| train 2013-2021 | 1,456 | 112.0 | 1,202 | 92.5 | 254 |
| selection 2022 | 145 | 11.15 | 116 | **8.92** | 29 |
| development 2023 | 145 | 11.15 | 127 | **9.77** | 18 |

**train期間は市場snapshot制約後も十分（EPV92.5）。selection(2022)・
development(2023)はcomplete-case化でEPVがそれぞれ8.92・9.77となり、
目安下限10を下回る**（all-starterのEPV11.15は既に境界線上だったが、
市場snapshotを必須特徴に含めるとさらに悪化する）。

検出力について: S4対S2のlogloss/Brier差の効果量に関する事前情報
（パイロットデータ）が存在しないため、厳密な検出力計算はできない
（`GATE0_REPORT.md`の既存結論のまま）。恣意的な仮効果量を用いた見かけの
精度は作らない。

**Gate 0D判定**: 市場snapshotを含む変種（S1以降）のEPVは目安下限を下回り、
市場snapshotを含まない変種（S0/S2の一部）はEPV92.5〜11.15の範囲で相対的に
安定。**市場統制ありの変種は2022年selectionの時点で既にEPV不足**。

---

## 4. EXP10・EXP12への影響注記

両実験とも`単勝オッズ`（外部kekkaファイルの結果内オッズ、タイムスタンプ
なし、確定オッズに近い）を市場特徴に使用しており、historical_pre_snapshot
ではない。「市場統制には時刻不明で確定オッズに近い結果ファイル由来単勝
オッズを使用したため、historical_pre_snapshot時点の実運用評価とは呼べない。
結果は、より後時点の市場情報を統制した保守的分析として解釈する」という
注記を両実験の`spec.json`（`market_data_provenance_note_20260922`）と
memoryへ追記済み。**両実験とも再実行していない**。EXP12は決定的placebo
FAILのため終了判断を維持。EXP10も事前停止規律により再実行せず、結論範囲を
訂正する変更は行っていない（元の結論はDNF母集団を最初からスコープ外と
明記していたため、訂正の必要自体がなかった）。

---

## 5. S0〜S5へ進めるかの判定材料（決定はユーザーに委ねる）

ユーザー事前基準: 「Gate 0Bと0Dを通過した場合だけS0〜S5へ進んでください」

- **Gate 0B**: 119/120特徴ではなく**19/120特徴（15.8%）で実測不一致を確認**。
  原因は完全に特定済み（dropna後母集団での位置/カウントベース集計、
  kako5系は5走ウィンドウのズレ、course/jockey系は恒久的な経験数過小
  カウント）。raw v6 scoreへの実測影響は該当行の1.03%（course/jockey群
  実測分のみ）で平均0.022・最大0.326、kako5群を含めるとさらに広い範囲へ
  及ぶと推定される。**「raw score：許容誤差以内」という事前基準には
  達していない**。
- **Gate 0D**: all-starterのEPV(11.15)は既に境界線上だったが、市場snapshot
  を必須とするcomplete-case制限で2022年selection EPV=8.92・2023年
  development EPV=9.77となり、**目安下限10を下回る**。

**両ゲートとも、ユーザーが事前に定めた基準に照らして「通過」とは言えない
状態にある**。ただし本報告はGate 0B/0Dの実測結果を提示するものであり、
S0〜S5へ進むか・course/jockey/kako5のDNF-population issueを先に是正して
から再測定するか・EPV不足を理由に終了するかは、この報告を踏まえてユーザー
が判断する。モデル学習・2024/2025年開封・ROI評価はいずれの選択が取られる
までは行っていない。

## 6. 副次的発見（EXP13の範囲外、production影響の可能性）

course_n_prev/course_win_rate/course_top3_rate・jockey_n_prev/
jockey_win_rate/jockey_top3_rate・kako5_*（13特徴）は、**EXP13固有の問題
ではなく、DNF走が過去に1回でもある馬について、現行の本番`master_v2`（v6の
学習データそのもの）の値が実測で不正確であることを意味する**
（2013-2025年で1.03%以上の行に影響、raw score差最大0.326）。これは
2026-09-11に発見・修正されたjockey/trainer_fukuの"window slot consumption"
バグ（P0-5、`analysis/p0_5_verification/`）と構造的に対称だが**未修正の
まま残っている**。EXP13のスコープでは訂正しない（本番pipeline修正は
別途のスコープ外大規模変更であり、CLAUDE.mdの確認対象）。ユーザーへ
別途報告する。

---

## 7. 成果物一覧

- `GATE0B_FEATURE_AUDIT.md`: 120特徴の全数分類+実測結果反映済み
- `gate0b_market_coverage_audit.py` / `out/gate0b_coverage_breakdown.json`
- `gate0d_full_starter_power.py` / `out/gate0d_full_starter_power.json`
- `gate0b_course_jockey_history_parity.py` / `out/gate0b_course_jockey_history_parity.json`
- `gate0b_kako5_history_parity.py` / `out/gate0b_kako5_history_parity.json`
- `gate0b_raw_score_parity.py` / `out/gate0b_raw_score_parity.json`
- `gate0b_dnf_full_starter_score.py` / `out/gate0b_dnf_full_starter_score.json`

いずれも読み取り専用・再現可能スクリプト。本番ファイル（`data/master_v2_*.csv`・
`models/unified_rank_v6.pkl`・`build_dataset.py`・`build_master_v2.py`・
`parse_kako5.py`）は一切変更していない。

関連: [[project_exp13_nonfinish_risk]] [[project_market_provenance_audit_20260922]]
[[project_kekka_ext_data_quirks]] [[feedback_asof_population_definition]]
