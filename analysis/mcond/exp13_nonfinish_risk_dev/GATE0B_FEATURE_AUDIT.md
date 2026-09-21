# Gate 0B — 120特徴 全数DNF-population監査

## ★2026-09-22 数値実測による最終決着(追記)

本ファイル執筆後、`gate0b_course_jockey_history_parity.py`・`gate0b_kako5_history_parity.py`・
`gate0b_raw_score_parity.py`で26特徴すべてを実測した。結論:

| 特徴群 | 件数 | 最終verdict | 実測根拠 |
|---|---|---|---|
| course_n_prev/win_rate/top3_rate, jockey_n_prev/win_rate/top3_rate | 6 | **CONFIRMED_DIVERGENT** | 3,401〜3,854/626,774行(0.5-0.6%)で不一致。affected 6,453行(1.03%)のraw v6 score差: mean=0.022, median=0.013, p95=0.079, max=0.326(`gate0b_raw_score_parity.json`) |
| kako5_avg_pos/std_pos/best_pos/avg_agari3f/best_agari3f/same_td_ratio/same_dist_ratio/same_place_ratio/pos_trend/race_count/expected_good_count/hidden_good_count/same_cond_best_pos | 13 | **CONFIRMED_DIVERGENT** | 最大10,024/558,801行(1.8%)で不一致(kako5_avg_agari3f)。max_abs_diffはkako5_avg_pos=4.53・kako5_best_pos=13.0・kako5_avg_agari3f=16.4など浮動小数点誤差ではない実質差(`gate0b_kako5_history_parity.json`) |
| kako5_avg_ninki, kako5_pos_vs_ninki, kako5_upset_good_count | 3 | **N/A(常にNaN)** | 学習時masterに「人気」列が存在しないためcompared=0、DNF問題とは無関係に常時NaN |
| hist_same_cond_best_pos/top3_rate/count, hist_same_place_best_pos | 4 | **CONFIRMED_IMMUNE** | 626,774行中diff=0(浮動小数点誤差1.1e-16のみ)。理由: 全キャリアスキャンで`if isnan(着順): continue`により着順値ベースでDNF行を自然除外するため、kako5(固定5走ウィンドウの行数ベース消費)とは異なりwindow slot consumption機構が働かない |

**Gate 0B結論(数値照合完了版)**: 120特徴中92がstructurally_immune(実測不要、raw passthrough/asof join)、
2がlikely_immune(horse_fuku、パターン一致・未再検証のまま)、4がconfirmed_immune(hist_same系、実測済み)、
**19がconfirmed_divergent**(course/jockey_n_prev系6 + kako5系13)、3がN/A。
Gate 0Bは「アーキテクチャ上実現可能」ではなく**「19/120特徴(15.8%)で実測非一致、原因は特定済み
(dropna後の母集団でexpanding/windowed位置ベース集計を行っているため)」が最終結論**。
詳細は`GATE0B_REPORT.md`参照。

---


対象: `models/unified_rank_v6.pkl` の `feature_cols`（120特徴、学習データ `data/master_v2_20130105-20251228.csv`）。
目的: 「DNF(除外・中止・失格)行の除去タイミングが、位置/カウントベースの特徴量計算に紛れ込んでいないか」を
全120特徴について機械的に分類する。READ-ONLY、コード/データ変更なし。

## 0. 確定した事実（コード読解 + 実測で検証済み）

- `build_dataset.py`: 生4CSV(`lgbm/cat/torch_transformer/add`, 631,965行, TARGET輸出そのまま) を
  `レースID(新)+馬番` でJOIN → `add_rolling_stats()`(jockey/trainer_fuku, **検証済み・確定無罪**) →
  `add_horse_rolling_stats()`(horse_fuku10/30) → `add_pace_features()`(prev_pos_rel/closing_power) →
  **この後に** `dropna(subset=["着順"])` (631,965→626,774行) → `master_20130105-20251228.csv` 保存。
  つまりこの4つの関数はすべて **DNF行を含むフル母集団の上で** 計算される。
- `parse_kako5.py --mode master`: 入力は `data/master_20130105-20251228.csv`
  (**dropna後、626,774行** — build_dataset.pyの最終保存物。実測で行数確認: 626,774行、
  master_kako5.csv も同じ626,774行で一致)。つまり kako5_* / hist_same_cond_* / hist_same_place_*
  は **DNF行が既に消えた母集団の上で** 馬ごとの自己結合（位置ベース）を行っている。
- `build_master_v2.py`: 入力は `data/master_kako5.csv`（同じく626,774行、DNF除去後）。
  Stage 1-5 `compute_history_features()` の course_*/jockey_n_prev 系も **DNF除去後の母集団**上で
  `groupby(...).cumcount()/cumsum()` を実行している。
- `build_master_v2.py` Stage 1-3 (`merge_hosei`, prev_hosei/prev_hosei9) と Stage 1-4
  (`merge_training`, trnH_*/trnW_*) はどちらも **キー結合**（前者は `レースID(新)+馬番` の単純left
  join、後者は `merge_asof` による馬名+日付のas-of結合、相手はmaster外部の坂路/WCファイル）であり、
  結合先の行がDNFかどうかに関わらず、対象行が存在しさえすれば結果は変わらない
  （母集団の行数・順序に依存しない）。
- `course_affinity_feats.py` / `trn_relative_feats.py` は **本番学習パイプライン未使用**
  （`trn_relative_feats.py` の唯一の呼び出し元は `lab/experiments/exp_trn_relative.py`。
  `course_affinity_feats.py` も同様にgrepで本番チェーンから未参照を確認）。
  `serve_history_feats.py` は course_win_rate 等と同名の列を生成するが、これは
  **週次サーブ時(predict_weekly.py/export_weekly_marks.py)専用の別実装**であり、
  master_v2 に格納された学習時の値には無関係（Gate 0Bはmaster_v2の値のみが対象）。

**この結果、jockey/trainer_fuku (既検証) と全く対称の理由で新たに2つのグループが判明した:**
1. **build_dataset.py内で計算される4系統(jockey/trainer_fuku, horse_fuku, prev_pos_rel/closing_power)
   = dropna前 = 免疫**（既知＋パターン一致）
2. **parse_kako5.py / build_master_v2.py Stage1-5で計算される3系統(kako5_*, hist_same_cond/place_*,
   course_*/jockey_n_prev系) = dropna後の母集団の上で位置/カウントベース計算 = 要検証（構造的リスクを実測で確認）**

すなわち、既検証のjockey/trainer_fukuの「無罪」は **同じ理由で全特徴に一般化できない**。
計算がdropnaの前か後かで真逆の結論になる、という当初の仮説どおりの分岐が実際に存在した。

---

## 1. 全120特徴 分類表

凡例（mechanism）: raw_passthrough / asof_key_join / positional_rolling_pre_dropna /
positional_rolling_post_dropna_or_unclear / other
凡例（verdict）: structurally_immune / likely_immune_same_pattern_as_verified / needs_verification /
unknown_need_more_reading

### 1-A. 生CSVそのまま（raw passthrough, 68特徴）— 開催情報・馬基本情報・前走系

`build_dataset.py` の `df_lgbm/df_cat/df_torch/df_add` に **列名そのまま存在**することを
`pd.read_csv(..., nrows=0).columns` で実測確認済み（TARGET輸出時点で確定している値であり、
build_dataset.py／build_master_v2.py 側では一切の再計算をしていない）。

| feature | source script/function | mechanism | verdict | note |
|---|---|---|---|---|
| 開催 | build_dataset.py (lgbm/cat/torch生CSV) | raw_passthrough | structurally_immune | TARGET輸出そのまま |
| 場所 | 同上 | raw_passthrough | structurally_immune | |
| Ｒ | 同上(lgbm/torch) | raw_passthrough | structurally_immune | |
| 枠番 | 同上 | raw_passthrough | structurally_immune | |
| 馬番 | 同上（JOINキー） | raw_passthrough | structurally_immune | |
| 芝・ダ | 同上 | raw_passthrough | structurally_immune | |
| 距離 | 同上 | raw_passthrough | structurally_immune | |
| コース区分 | 同上 | raw_passthrough | structurally_immune | |
| 芝(内・外) | 同上 | raw_passthrough | structurally_immune | |
| 馬場状態 | 同上 | raw_passthrough | structurally_immune | |
| 天気 | 同上 | raw_passthrough | structurally_immune | |
| クラス名 | 同上 | raw_passthrough | structurally_immune | |
| トラックコード(JV) | 同上 | raw_passthrough | structurally_immune | |
| 前走走破タイム | lgbm/torch生CSV | raw_passthrough | structurally_immune | TARGET側が自馬の直前走を指す前走ポインタ列。build側は無加工 |
| 前走着差タイム | 同上 | raw_passthrough | structurally_immune | |
| 前1角 | 同上 | raw_passthrough | structurally_immune | |
| 前2角 | 同上 | raw_passthrough | structurally_immune | |
| 前3角 | 同上 | raw_passthrough | structurally_immune | |
| 前4角 | 同上 | raw_passthrough | structurally_immune | |
| 前走上り3F | 同上 | raw_passthrough | structurally_immune | |
| 前走上り3F順 | 同上 | raw_passthrough | structurally_immune | |
| 前走確定着順 | 同上 | raw_passthrough | structurally_immune | |
| 前走日付 | 同上 | raw_passthrough | structurally_immune | |
| 前走場所 | 同上 | raw_passthrough | structurally_immune | |
| 前芝・ダ | 同上 | raw_passthrough | structurally_immune | |
| 前距離 | 同上 | raw_passthrough | structurally_immune | |
| 前走馬場状態 | 同上 | raw_passthrough | structurally_immune | |
| 前走出走頭数 | 同上 | raw_passthrough | structurally_immune | prev_pos_rel/closing_power の入力にもなる |
| 前走競走種別 | 同上 | raw_passthrough | structurally_immune | |
| 前走トラックコード(JV) | 同上 | raw_passthrough | structurally_immune | |
| 前走斤量 | 同上 | raw_passthrough | structurally_immune | |
| 前走馬体重 | 同上 | raw_passthrough | structurally_immune | |
| 前走馬体重増減 | 同上 | raw_passthrough | structurally_immune | |
| 前走Ave-3F | 同上 | raw_passthrough | structurally_immune | |
| 前PCI | 同上 | raw_passthrough | structurally_immune | |
| 前好走 | 同上 | raw_passthrough | structurally_immune | |
| 前走PCI3 | 同上 | raw_passthrough | structurally_immune | |
| 前走RPCI | 同上 | raw_passthrough | structurally_immune | |
| 前走平均1Fタイム | 同上 | raw_passthrough | structurally_immune | |
| 前走レースID(新) | 同上 | raw_passthrough | structurally_immune | |
| 前走レースID(新/馬番無) | 同上 | raw_passthrough | structurally_immune | |
| 種牡馬 | cat/torch生CSV | raw_passthrough | structurally_immune | |
| 父タイプ名 | 同上 | raw_passthrough | structurally_immune | |
| 母馬 | 同上(catのみ) | raw_passthrough | structurally_immune | |
| 母父馬 | 同上 | raw_passthrough | structurally_immune | |
| 母父タイプ名 | 同上 | raw_passthrough | structurally_immune | |
| 毛色 | 同上 | raw_passthrough | structurally_immune | |
| 騎手コード | 同上 | raw_passthrough | structurally_immune | jockey_fuku等の集計キーでもある |
| 調教師コード | 同上 | raw_passthrough | structurally_immune | |
| 馬主(最新/仮想) | 同上 | raw_passthrough | structurally_immune | |
| 生産者 | 同上 | raw_passthrough | structurally_immune | |
| 年齢限定 | 同上 | raw_passthrough | structurally_immune | |
| 限定 | 同上 | raw_passthrough | structurally_immune | |
| 性別限定 | 同上 | raw_passthrough | structurally_immune | |
| 指定条件 | 同上 | raw_passthrough | structurally_immune | |
| 重量種別 | 同上 | raw_passthrough | structurally_immune | |
| 性別 | torch生CSV | raw_passthrough | structurally_immune | |
| 年齢 | 同上 | raw_passthrough | structurally_immune | |
| 斤量 | 同上 | raw_passthrough | structurally_immune | |
| 馬齢斤量差 | 同上 | raw_passthrough | structurally_immune | TARGET側で計算済み、build側は無加工 |
| 斤量体重比 | 同上 | raw_passthrough | structurally_immune | 同上 |
| ブリンカー | 同上 | raw_passthrough | structurally_immune | |
| 間隔 | 同上 | raw_passthrough | structurally_immune | |
| 休み明け～戦目 | 同上 | raw_passthrough | structurally_immune | |
| 出走頭数 | add生CSV | raw_passthrough | structurally_immune | |
| フルゲート頭数 | add生CSV | raw_passthrough | structurally_immune | build_dataset.py L250-252で欠損12件を中央値補完。補完自体もdropna**前**のadd全体で計算、位置非依存の定数補完なので免疫 |
| 騎手年齢 | add生CSV | raw_passthrough | structurally_immune | |
| 調教師年齢 | add生CSV | raw_passthrough | structurally_immune | |

### 1-B. build_dataset.py内で計算（dropna前, 4特徴・既検証）

| feature | source | mechanism | verdict | note |
|---|---|---|---|---|
| jockey_fuku30 | build_dataset.py `add_rolling_stats()` | positional_rolling_pre_dropna | structurally_immune | **既検証（diff=0, 624,760行照合済み）**。2026-09-11 C1修正後の実装でも、dropna(L321)より前のL289で計算されるためDNF全母集団上で計算されることに変わりなし |
| jockey_fuku90 | 同上 | positional_rolling_pre_dropna | structurally_immune | 同上 |
| trainer_fuku30 | 同上 | positional_rolling_pre_dropna | structurally_immune | 同上 |
| trainer_fuku90 | 同上 | positional_rolling_pre_dropna | structurally_immune | 同上 |

### 1-C. build_dataset.py内で計算（dropna前, パターン一致・未再検証, 2特徴）

| feature | source | mechanism | verdict | note |
|---|---|---|---|---|
| horse_fuku10 | build_dataset.py `add_horse_rolling_stats()`（L292、dropnaのL321より前） | positional_rolling_pre_dropna | likely_immune_same_pattern_as_verified | `groupby("血統登録番号").shift(1).rolling(10,min_periods=3).mean()`。jockey/trainer_fukuと全く同型の「shift+rolling on pre-dropna population」。数値再検証はしていない |
| horse_fuku30 | 同上（window=30, min_periods=5） | positional_rolling_pre_dropna | likely_immune_same_pattern_as_verified | 同上 |

### 1-D. build_dataset.py内で計算（行内完結の決定論的な式, 2特徴）

| feature | source | mechanism | verdict | note |
|---|---|---|---|---|
| prev_pos_rel | build_dataset.py `add_pace_features()`（L172-178） | other（行単位の四則演算、cross-row依存なし） | structurally_immune | `(前1角-1)/(前走出走頭数-1)` — 全て**自分の行のraw passthrough列のみ**から計算。groupby/shift/rolling/cumcountを一切使わない。DNF行の有無・並び順に一切依存しない。likely_immune以上に強い"厳密免疫" |
| closing_power | 同上 | other | structurally_immune | `(前1角-前4角)/(前走出走頭数-1)`。同上の理由で厳密免疫 |

### 1-E. parse_kako5.py `build_from_master()`（**dropna後の母集団**で位置ベース自己結合, 16特徴）

`master_path = data/master_20130105-20251228.csv`（dropna後、626,774行 — 実測で行数一致確認済み）
を読み込み、馬ごとに `idxs[max(0, seq_i-5):seq_i]`（**直近5"行"**、行=DNF除去済みレコード）で
「過去5走」を構築する。DNF行はこの入力に一切存在しないため、真の過去5走（DNF含む）とは
異なる馬・異なる期間で「5走ウィンドウ」がズレる可能性がある — jockey/trainer_fukuの
"window slot consumption" と全く同型の機構だが、**dropna後**に作用する点が逆。

| feature | source | mechanism | verdict | note |
|---|---|---|---|---|
| kako5_avg_pos | parse_kako5.py `build_from_master()`→`_compute_features()` | positional_rolling_post_dropna_or_unclear | needs_verification | 直近5"行"(post-dropna)の平均着順 |
| kako5_std_pos | 同上 | 同上 | needs_verification | |
| kako5_best_pos | 同上 | 同上 | needs_verification | |
| kako5_avg_ninki | 同上 | 同上 | needs_verification | |
| kako5_pos_vs_ninki | 同上 | 同上 | needs_verification | |
| kako5_avg_agari3f | 同上 | 同上 | needs_verification | |
| kako5_best_agari3f | 同上 | 同上 | needs_verification | |
| kako5_same_td_ratio | 同上 | 同上 | needs_verification | 分母nが直近5"行"のカウント自体 |
| kako5_same_dist_ratio | 同上 | 同上 | needs_verification | |
| kako5_same_place_ratio | 同上 | 同上 | needs_verification | |
| kako5_pos_trend | 同上 | 同上 | needs_verification | 5点の線形回帰傾き。ウィンドウの中身がズレれば傾きも変わる |
| kako5_race_count | 同上 | 同上 | needs_verification | n=len(past_races)。DNFがあれば本来より過大（過去に遡って5走に届くまで探す）にはならないが、「5走に含まれる期間」が延伸する |
| kako5_expected_good_count | 同上 | 同上 | needs_verification | |
| kako5_upset_good_count | 同上 | 同上 | needs_verification | |
| kako5_hidden_good_count | 同上 | 同上 | needs_verification | |
| kako5_same_cond_best_pos | 同上 | 同上 | needs_verification | |

### 1-F. parse_kako5.py `build_from_master()`（全キャリア版、**dropna後**, 4特徴）

ウィンドウ固定ではなく `range(seq_i)`（その馬の**それまでの全行**、dropna後）を舐める点は
kako5_*と異なるが、母集団がdropna後である点は同じ。DNF走はここでも一切カウントされない
（「同条件出走回数」「同条件複勝率」からDNF走が構造的に除外される）。

| feature | source | mechanism | verdict | note |
|---|---|---|---|---|
| hist_same_cond_best_pos | parse_kako5.py `build_from_master()` L279-302 | positional_rolling_post_dropna_or_unclear（非ウィンドウ・全キャリア累積） | needs_verification | 全キャリアで同TD+同距離帯(±200m)の最高着順。DNF走は集計母数に入らない |
| hist_same_cond_top3_rate | 同上 | 同上 | needs_verification | |
| hist_same_cond_count | 同上 | 同上 | needs_verification | |
| hist_same_place_best_pos | 同上 L304-314 | 同上 | needs_verification | 全キャリアで同場所の最高着順 |

### 1-G. build_master_v2.py Stage 1-5 `compute_history_features()`（**dropna後**の母集団で累積カウント, 6特徴）

入力 `data/master_kako5.csv` は1-E/1-Fの出力そのもの（626,774行 = dropna後、実測確認済み）。
`groupby([血統登録番号, コースキー]).cumcount()/cumsum()`（course系）、
`groupby([血統登録番号, 騎手コード]).cumcount()/cumsum()`（jockey系）と、**ウィンドウ固定ではなく
無制限expanding cumulative**である点が kako5_* / hist_same_* とも異なる特徴。DNF走が
その馬の「同コース」「同騎手」経験からまるごと消えるため、n_prev（経験数）がDNF発生以降
**永続的に**過小カウントされる（ウィンドウのように後で"追いつく"ことがない）。

| feature | source | mechanism | verdict | note |
|---|---|---|---|---|
| course_n_prev | build_master_v2.py `compute_history_features()` L201-209 | positional_rolling_post_dropna_or_unclear（expanding cumcount） | needs_verification | 同(場所×芝ダ×距離帯)の過去出走"行"数。DNF走は生涯カウントから恒久的に欠落 |
| course_win_rate | 同上 | 同上 | needs_verification | 分母=course_n_prev |
| course_top3_rate | 同上 | 同上 | needs_verification | 同上 |
| jockey_n_prev | 同上 L212-219 | 同上 | needs_verification | 同(馬×騎手)ペアの過去出走"行"数。**jockey_fuku30/90とは別系統・別の集計軸**（fuku系=騎手コード単位でDNF行込みdropna前、こちらは馬×騎手ペア単位でdropna後）。名前が紛らわしいが無関係な計算 |
| jockey_win_rate | 同上 | 同上 | needs_verification | |
| jockey_top3_rate | 同上 | 同上 | needs_verification | |

### 1-H. build_master_v2.py Stage 1-3（キー結合, 2特徴）

| feature | source | mechanism | verdict | note |
|---|---|---|---|---|
| prev_hosei | build_master_v2.py `merge_hosei()` L66-84 | asof_key_join | structurally_immune | `レースID(新)+馬番` の単純left join。結合先`data/hosei/H_*.csv`側の値がどう作られたかはこの監査の範囲外。off-by-one問題は別途2026-09で修正済み（memory `project_serve_prev_hosei_offbyone`）。join自体はmasterの行数・順序に非依存 |
| prev_hosei9 | 同上 | asof_key_join | structurally_immune | 同上 |

### 1-I. build_master_v2.py Stage 1-4（`merge_asof`, 外部ファイル結合, 9+7特徴）

`merge_training()` は `data/training/H-*.csv` / `W-*.csv`（プロジェクト外部の坂路/WC調教マスター、
masterのレース母集団と無関係に存在するファイル）を **馬名+日付のmerge_asof(direction="backward")**
で結合する。「その馬の直近の調教」を探す処理自体はmaster側の行の存在・順序に一切依存しない
（`race_df[["_name_key","日付"]].drop_duplicates()` をキーにasofするだけ）。

| feature | source | mechanism | verdict | note |
|---|---|---|---|---|
| trnH_Time1 | build_master_v2.py `merge_training()`→`_latest_before()` | asof_key_join | structurally_immune | 馬名+日付でH-20150401-20260313.csvへmerge_asof |
| trnH_Time2 | 同上 | asof_key_join | structurally_immune | |
| trnH_Time3 | 同上 | asof_key_join | structurally_immune | |
| trnH_Time4 | 同上 | asof_key_join | structurally_immune | |
| trnH_Lap1 | 同上 | asof_key_join | structurally_immune | |
| trnH_Lap2 | 同上 | asof_key_join | structurally_immune | |
| trnH_Lap3 | 同上 | asof_key_join | structurally_immune | |
| trnH_Lap4 | 同上 | asof_key_join | structurally_immune | |
| trnH_days_ago | 同上 | asof_key_join | structurally_immune | asof結果の日付差、決定論的 |
| trnW_5F | 同上(W版) | asof_key_join | structurally_immune | 馬名+日付でW-20150401-20260313.csvへmerge_asof |
| trnW_4F | 同上 | asof_key_join | structurally_immune | |
| trnW_3F | 同上 | asof_key_join | structurally_immune | |
| trnW_Lap1 | 同上 | asof_key_join | structurally_immune | |
| trnW_Lap2 | 同上 | asof_key_join | structurally_immune | |
| trnW_Lap3 | 同上 | asof_key_join | structurally_immune | |
| trnW_days_ago | 同上 | asof_key_join | structurally_immune | |

---

## 2. 集計（120特徴の内訳）

| verdict bucket | 件数 | 内訳 |
|---|---:|---|
| structurally_immune | **92** | raw_passthrough 68 + build_dataset.py既検証4系統(jockey/trainer_fuku) 4 + prev_pos_rel/closing_power 2 + prev_hosei/prev_hosei9(asof) 2 + trnH_* 9 + trnW_* 7 |
| likely_immune_same_pattern_as_verified | **2** | horse_fuku10, horse_fuku30 |
| needs_verification | **26** | kako5_* 16 + hist_same_cond/place_* 4 + course_*/jockey_n_prev系 6 |
| unknown_need_more_reading | **0** | — |
| **合計** | **120** | |

**結論の要旨**: 120特徴のうち92特徴(76.7%)は生CSV直輸入かasof系キー結合であり母集団の行数・順序に
一切依存しない。既検証4特徴＋パターン一致2特徴(horse_fuku)の計6特徴はdropna前の位置ベース計算だが
全母集団(DNF込み)の上で行われるため無罪。**残る26特徴(21.7%)は逆にdropna後(DNF除去済み)の
母集団の上で位置/カウントベースの自己結合・累積集計を行っており、構造的にDNF-population riskに
さらされていることをコード読解＋実測(行数一致)で確認した。** これはjockey/trainer_fukuのケースの
「無罪」とは対称的な、真逆の結果になる可能性のあるグループである。

---

## 3. needs_verification 26特徴の数値再検証手順（実装者向け仕様）

`analysis/p0_5_verification/reconstruct_true_pipeline_universe.py` と同じ方法論
（①真のフル母集団を再構築 → ②同じロジックで再計算 → ③既に照合可能な行に絞る → ④master_v2の
格納値とdiffを取る）を、3つのサブグループに分けて実装する。

### 3-1. kako5_* (16特徴) + hist_same_cond/place_* (4特徴) の再検証

**目的**: parse_kako5.py の `build_from_master()` を、dropna**前**の母集団(631,965行、DNF込み)
に対して実行した場合の値と、現在の格納値(dropna後の母集団で計算)を比較する。

**手順**:
1. `data/master_20130105-20251228.csv` の**保存前**の中間状態を再現する必要がある。
   これは `build_dataset.py` の `build_master()` 内、L321 `dropna(subset=["着順"])` の**直前**の
   DataFrame（631,965行）に相当する。build_dataset.py は中間生成物を保存していないため、
   `build_dataset.py` の `build_master()` を改造版（dropna行をコメントアウト、または
   `dropna_flag` 引数を追加）で再実行し、631,965行版の「フル `master_20130105-20251228.csv`」を
   別名で保存する必要がある（例: `master_full_prednonfinish.csv`。**書き込み先はワーキングディレクトリ
   外のスクラッチ領域にすること、data/ 直下の既存ファイルは上書きしない**）。
2. `parse_kako5.py` の `build_from_master()` をこの631,965行版に対して実行し、
   `kako5_*` / `hist_same_cond_*` / `hist_same_place_*` を再計算する（"true_universe" 版と呼ぶ）。
3. 一方、現行の格納値は `data/master_v2_20130105-20251228.csv` から読める（"stored" 版、626,774行）。
4. 2つを `レースID(新)+馬番`（またはユニーク行キー）で突合し、
   "stored" 側に存在する626,774行についてのみ diff を取る（"true_universe" 版はDNF行も含むため
   626,774行に絞り込むフィルタが必要 = `dropna(subset=["着順"])` を再計算後のDataFrameに適用）。
5. `kako5_avg_pos` 等の連続値は `abs(diff) > 1e-6` のカウント、`kako5_race_count` 等の整数値は
   厳密不一致カウントを取り、**「馬の過去走にDNF走が実在する行」に絞って不一致率を見る**
   （DNF走が過去5走以内に一切存在しない馬にとっては両者は理論上一致するはずなので、
   全行平均だと真の不一致が希釈されて見えなくなる。分母を「対象馬の直近5走以内に最低1回
   DNF走がある行」に絞ることが重要）。
6. **リスク方向の確認ポイント**: dropna前提の"true_universe"版のほうが、DNF走を跨いで
   より古い日付の走まで遡って5走を構成する可能性がある（=現行の"stored"版はDNF走がない
   ふりをして直近5走を数えるため、実際には「6走前・7走前」まで含んだ走を「5走前まで」と
   誤認する形になりうる）。

### 3-2. course_n_prev/course_win_rate/course_top3_rate/jockey_n_prev/jockey_win_rate/jockey_top3_rate (6特徴) の再検証

**目的**: build_master_v2.py `compute_history_features()` を、DNF行込みの母集団に対して
実行した場合との差分を見る。

**手順**:
1. 3-1と同じ「631,965行版フルmaster」（dropna前）に対して、`build_master_v2.py` の
   `compute_history_features()` と同一ロジック（`groupby([血統登録番号, コースキー]).cumcount()`等）
   を再実行する。**ただし** `_is_win`/`_is_top3` は `着順` がNaNの行（DNF行）では両方0または
   NaN扱いになる点に注意（現行コードは `(df[COL_JYUN]==1).astype("Int8")` なのでNaN比較は
   Falseになり0扱い＝勝ち/複勝としてはカウントされないが、**cumcount自体は"行の存在"にのみ
   依存するのでDNF行も1件として数えられる**、というのが「あるべき」計算）。
2. 現行の"stored"値（DNF行がそもそも存在しないので、その分n_prevが恒久的に少ない）と比較。
3. 期待される不一致パターン: ある馬が過去にDNF走をn回経験している場合、それ以降の全レースで
   `course_n_prev`/`jockey_n_prev` が「真の値」より**最大n少なく**格納されているはず
   （expanding cumulative のため、window的な"時間とともに解消"がなく、恒久的にズレ続ける）。
   これはkako5系(固定5走ウィンドウ)よりも影響が大きい可能性がある。
4. 分母が0の行（course_n_prev==0でcourse_win_rate=NaNになるケース）については、
   「本来はDNF走が1件あるだけでn_prev>=1になっていた」行が誤ってNaN(初出走扱い)になっている
   ケースを個別に数える（これは連続値のdiffでは見えない、カテゴリ的な誤判定なので別集計が必要）。

### 3-3. 優先度・見積り

- 3-2 (course/jockey, 6特徴) を先に実施することを推奨: **expanding cumulative** のため
  理論上もっとも累積的に不一致が拡大しやすく、かつ計算コストも軽い(単純cumcount/cumsum)。
- 3-1 (kako5/hist_same, 20特徴) は計算コストが重い（Python forループでの馬ごと逐次処理、
  `parse_kako5.py` 自体 626,774行に対して数分〜十数分オーダー。631,965行フル版の再構築+
  同ロジック再実行はその2倍弱の手間感）。
- どちらも「631,965行版フルmasterの再構築」という共通の前提作業が必要なため、
  1回作れば3-1/3-2両方で使い回せる。

---

## 4. 対象外・補足

- `course_affinity_feats.py`, `trn_relative_feats.py` は本番学習チェーン（build_dataset.py →
  parse_kako5.py → build_master_v2.py）から未参照（grep確認済み）。前者は今回のリストの
  course_n_prev等の実装先ではなく、後者は `lab/experiments/exp_trn_relative.py` 専用。
- `serve_history_feats.py` はcourse_win_rate等と同名列を生成するが、これは週次サーブ時
  (`predict_weekly.py`/`export_weekly_marks.py`)専用の別実装であり、**master_v2の学習時格納値には
  無関係**。Gate 0Bはmaster_v2の値のみを対象とするため、この監査では対象外とした
  （ただしserve側とtrain側で同名特徴の計算ロジックが分岐している事実自体は、別途
  「train/serve parity」観点での懸念として記録しておく価値がある）。
- prev_hosei/prev_hosei9の結合先 `data/hosei/H_*.csv` 自体がどう計算されているか（母集団の
  dropna状態を含む）はこの監査の範囲外。既知の off-by-one 修正（memory
  `project_serve_prev_hosei_offbyone`）に照らして別途扱う。
