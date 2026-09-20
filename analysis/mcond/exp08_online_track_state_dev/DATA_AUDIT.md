# EXP08 Stage 1 — データ可用性監査

**前提**: `PRIOR_ART_AUDIT.md`の推奨をユーザーが承認し、スコープを以下へ確定した
(2026-09-20夜、ユーザー指示):
- **主仮説**: 時計(走破タイム残差)・上がり性能・ペースの当日オンライン状態のみ
- **内外(枠)・脚質(前残り/差し)**: 採用候補から除外、negative control専用
- 予測対象レースより前に結果が利用可能だったレースのみ使用
- 2023年development単独で状態定義・パラメータ・閾値を固定、2024/2025で再調整しない
- RAW/EWMA比較必須(状態モデル固有の上積みをGateで判定)
- permutation placebo testをGateに事前登録
- 既存`day_state_counting.py`系は変更しない(参照のみ)

## §5.1 必須項目の可用性

### 先行レース(観測源)

| 項目 | ソース | 確認結果 |
|---|---|---|
| race_id/競馬場/開催日/芝ダ/距離 | `data/master_v2_20130105-20251228.csv` | ✅ rid16、`場所`、`芝・ダ`、`距離`列で完備 |
| 発走予定/実発走時刻 | `master_v2`の`発走時刻`(実測値、TARGET出走表由来) | ✅ ただし「予定」と「実」の区別列は無い(下記§5.2で保守的に対応) |
| **結果確定/取得可能時刻** | 【2026-09-20夜訂正、下記§5.2参照】信頼できる時刻は存在しないと判明 | ⚠️ 実測不能と確認。保守的な固定遅延(+20分/+30分)を仮定として採用(「実測」ではない) |
| 枠番・馬番・着順 | `master_v2`、外部`kekka_2010_2025_fix_raceid_v2__keyed.csv` | ✅ 両方に存在 |
| **走破タイム** | 外部kekka(`走破タイム`列、M+SS+d の4桁エンコード、例1336=1:33.6) | ✅ 全年missing率0%、距離別平均値の妥当性を実測確認済み(1200m≈71.8s/1600m≈96.6s/2000m≈122.7s) |
| **上がり3F/上がり3F順** | 外部kekka(`上り3F`,`上り3F順`) | ✅ missing率0.8-0.9%(全年安定) |
| **通過順位(コーナー)** | 外部kekka(`1角`〜`4角`) | ✅ 3角missing率1.3-1.5%(短距離で3角自体が存在しないレースは構造的欠損、EXP08§7.4の「存在しない値を0で補わない」規約通り除外扱いとする) |
| **ペース関連値** | 外部kekka(`PCI`,`RPCI`,`Ave-3F`,`平均1Fタイム`,`平均速度`,`-3F平均速度`,`上り3F平均速度`) | ✅ RPCI missing率2.9-3.2%(全年安定) |
| 頭数 | `master_v2`(`出走頭数`) | ✅ |
| 公表馬場状態/天候 | `master_v2`(`馬場状態`,`天気`)、外部kekkaにも同名列あり(整合性は要突合、Stage2実装時に検証) | ✅ 存在 |
| 市場確率 | `data/Time _series_odds/TANPUK_*.csv`(区分1=前売り、区分4=確定) | ✅ EXP07で確立済みhistorical_pre_snapshot方式を再利用可能 |
| v6/OOS予測確率 | `reports/marks_v5/*.json`(day_state_counting.pyが使用、2024-2025 OOS)または`data/_research/mcond/exp05_design.parquet`(v6_score、EXP07で使用) | ✅ 2系統とも存在、後者は2024-2025を含む(EXP07で相関0.9915を確認済み) |

### 対象馬(予測対象)

| 項目 | ソース | 確認結果 |
|---|---|---|
| 過去走由来の事前脚質 | `master_v2`の`prev_pos_rel`,`closing_power`(v6特徴量として既存、serve-safe) | ✅ ただしnegative control専用(脚質次元は採用候補外) |
| 枠位置百分位 | `master_v2`の`枠番`/`出走頭数`から計算可能 | ✅ ただしnegative control専用 |
| v6/EXP05確率 | 上記と同じ | ✅ |
| 市場確率 | TANPUK historical_pre_snapshot | ✅ |
| OOD/support | 未確認(EXP04/EXP05系の該当機能を要調査) | ⚠️ Stage2実装時に確認、無ければ「OOD情報なし」として進める(必須項目ではない) |
| 判断時点 | TANPUK区分1最終スナップショット(historical_pre_snapshot、EXP07で確立済み、中央値28分前) | ✅ |

## §5.2 利用可能時刻ルール【2026-09-20夜、全面訂正】

### 訂正の経緯

初版(Stage1完了時点)は、TANPUKの`区分4`(確定オッズ)タイムスタンプと`発走時刻`の
差を「結果(着順・走破タイム・上がり・通過順)が利用可能になった時刻」の**実測**と
称し、これを根拠に固定遅延15分を凍結していた。**これは誤り**: TANPUK確定は
オッズ・払戻が確定した時刻であり、着順等の結果レコード自体がいつ取得・公開された
かを直接示すものではない(オッズ確定には結果確定が前提だが、両者の時刻が一致する
保証はない)。ユーザー指摘により全面的に訂正する。

### 再監査(2026-09-20夜)

1. **JV-Link/保存データ内に結果レコードの取得・公開・更新時刻が存在するか調査**:
   コードベース全体を調査した結果、**存在しないことを確認した**。
   - JV-LinkのRACE(RA)レコードにある「データ作成年月日」(`jvlink_race_calendar.py`)は
     レース**前**の番組表発表日であり、結果とは無関係。日付のみで時刻情報もない。
   - `jvlink_results.py`の`fetched`フィールドはローカルのスクリプト実行時刻であり、
     JRA-VAN発行のタイムスタンプではない。同ファイルのパーサー自体も
     `PARSER_VALIDATED = False`で未検証。
   - 保存済みCSV(`data/kekka/*.csv`等)には`取得日時`/`更新日時`に相当する列がない。
2. **信頼できる時刻が存在しないため、「実測」ではなく保守的な仮定と明記する**。
3. **固定ルール(事前固定、実結果を見て変更しない)**:
   ```
   prior_result_available_timestamp = actual_post_time + 20分  (主解析)
   prior_result_available_timestamp = actual_post_time + 30分  (時点安全の感度分析)
   ```
   +20分と+30分で効果の方向が一致しない場合はFAILとする(spec.json
   `result_availability_timing`参照)。
4. **行単位の不変条件**: `prior_result_available_timestamp <= target_decision_timestamp`
   を行単位で保存し、違反が1件でもあればGate 0 FAILとする。この検査ユーティリティは
   `test_time_safety.py::assert_no_availability_violations`として先行実装済み
   (online_state.py実装後、全先行レース×対象レースペアに対して実際に検査する)。

対象レース自身の判断時点`decision_timestamp`は、EXP07で確立した
`historical_pre_snapshot`(TANPUK区分1最終、発走26-30分前・中央値28分前)を再利用する
(こちらは市場オッズの取得時点として実測済みであり、今回の訂正の対象ではない)。

実装: `build_observations.py`の`_attach_availability_timestamps()`が
`actual_post_datetime`(master_v2発走時刻)・`prior_result_available_ts_primary`
(+20分)・`prior_result_available_ts_sensitivity`(+30分)の3列を
`observations.parquet`へ付与する。

## §6 事前期待値モデル【Stage2完了、2026-09-20夜expanding-window化】

EXP08§6/§7.3は「同条件の期待走破タイムに対する残差」を要求し、「期待タイムモデルが
時点安全に構築できない場合、この状態次元は除外する」と明記している。

**実装済み**(`expected_time_model.py`): 距離50m単位×**競馬場**×芝ダ×コース区分×
公表馬場状態×クラス名の中央値ルックアップ(段階的フォールバック: full→no_going→
no_class→coarse→global)。**expanding-window(年単位leave-year-out)**で構築し、
年Y(<2023)の期待値はYより前の年だけでfitしたモデルから作る。2023年以降は
2022年末までの単一モデルに固定し、2023・2024・2025のどの年の結果を見ても再fitしない。
削除不変性(未来年を削除しても過去年の期待値が変わらないこと)は
`test_time_safety.py`で直接検証済み。

RPCI(ペース指数)も同じexpanding-window機構を流用して期待値化(RPCIはレース単位で
既に1値のため、中央値集約ではなく残差化のみ)。

競馬場を条件付けキーに含めるのは、競馬場固有の恒常的な時計の速さ/遅さを
「日次で変動する状態」と混同しないため(spec.json `expected_value_model`参照)。

## Gate 0 判定(§5末尾のチェックリスト)

| 条件 | 判定 | 根拠 |
|---|---|---|
| 競馬場と開催日を完全識別可能 | ✅ | rid16[0:8]=日付, rid16[8:10]=場コード(EXP07で確立済み) |
| 10競馬場を区別可能 | ✅ | 同上、EXP07 REPORT.md§9aで10場すべて確認済み |
| 芝とダートを区別可能 | ✅ | `芝・ダ`列 |
| 発走時刻が取得可能 | ✅ | `master_v2`の`発走時刻` |
| 先行レースの利用可能性を保守的に判定可能 | ✅ | §5.2の固定+20分(主)/+30分(感度)ルール(実測不能と確認済み、保守的仮定と明記) |
| 着順と枠情報が利用可能 | ✅ | `master_v2`/外部kekka |
| 事前脚質を対象レース結果なしで作成可能 | ✅(ただしnegative control専用) | `prev_pos_rel`/`closing_power`は既にserve-safe(v6特徴量として運用中) |
| 未来レースを除外できる | ✅ | rid16の日付+発走時刻での時系列ソートで対応可能 |

**Gate 0: PASS(8条件)。9条件目(行単位の利用可能時刻不変条件)はonline_state.py
実装後に実際のペアで検査する(spec.json `gates.gate0_data_health`参照)。**

## 次の一手

Stage 3: `online_state.py`(Kalman状態空間モデル本体)。spec.json凍結・
build_observations.py訂正・時点安全テスト追加が完了した本コミット後に着手する。
