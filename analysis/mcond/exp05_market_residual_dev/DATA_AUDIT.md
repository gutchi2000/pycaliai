# EXP05 データ監査

対象データ: `data/_research/mcond/base.parquet` (v6base, 時点安全), `data/_research/mcond/exp04_candidates.parquet`
(EXP04の145候補, 時点安全性はEXP04で検証済み再利用), `data/serve_feature_baseline.json` (2026-07-18/19/25/26
の4週次実行を対象にしたserve可用性監査), `models/pl_calibrators_v6.pkl`。

## 1. 候補データセットの規模

`build_features.py` 実行結果 (`out/feature_lists.json`):

| 項目 | 値 |
|---|---|
| 行数 (馬単位) | 477,638 |
| レース数 | 34,545 |
| 対象年 | 2016-2025 |
| train期間 | 2016-2021 |
| sel年 (ハイパラ選択) | 2022 |
| F-full候補数 | 145 (=EXP04のC1 133 + C2 8 + C3 4、そのまま流用) |
| F-serve候補数 | 117 |
| serve不可で除外 | 28 |

フィルタ条件はEXP04と同一 (`n_field>=5`、市場pre確率・v6の3着内確率とも非欠損)。

## 2. F-serve (本番で継続取得可能な特徴) の決め方

`data/serve_feature_baseline.json` (2026-07-18〜26の週次実行4回の監査、CLAUDE.mdが言及する
「v6は学習gainの28.15%をserveで失っていた」の元データ) の `serve_dead` (被覆率ほぼ0%、本番serveで
実質死んでいる列) 30列を、EXP04のC1候補(v6の120特徴のうち機械監査で残った105→133列展開後) の
由来列名と突き合わせ、28列を除外した。

除外された28列 (由来元):
`トラックコード(JV)` `前PCI` `前走PCI3` `前走RPCI` `前走トラックコード(JV)` `前走出走頭数`
`前走平均1Fタイム` `前走日付` `前走競走種別` `前走馬体重` `前走馬体重増減` `斤量体重比` `調教師年齢`
`馬齢斤量差` `騎手年齢` (数値、各1列) + `性別限定` `毛色` `限定` のダミー変数群 (計約13列)。

除外しなかった serve_dead 由来の列: `母馬` `生産者` `馬主(最新/仮想)` `前走場所` `前走走破タイム` は
EXP04の時点でR5(高基数)/R6(数値変換不能)/R4(馬主上書き)によりC1候補に**そもそも含まれていない**
ため、この時点で二重に除外する必要はない。`前走レースID(新)` 等の識別子列もEXP04時点で候補に無い。

**注意**: `serve_feature_baseline.json` は2026-07時点のスナップショット。CLAUDE.mdの2026-09-10更新
によれば、そこに含まれる一部の欠損原因(P0-1騎手/調教師stats、P0-2着度数CSV列数、P0-4Ｒ全角)は
その後の実データ再検証で解消済みと記録されている。つまり**現在のserve可用性はこのファイルが示す
より良い可能性がある**。EXP05のF-serveはこの古い監査に基づく保守的な(厳しめの)部分集合であり、
「F-serveで成立しなかった特徴の一部は、実は現在は取得できる」という含みがある。この点は
LOCKED_PERIOD_AUDIT.mdの結論(2026年再監査が必要)と合わせて次の一手として記録する。

C2 (EXP01由来の陣営選択特徴8列) と C3 (EXP02/03由来の能力・経験特徴4列) はいずれも v6の生120特徴
ではなく別実験で作られた派生特徴であり、`serve_feature_baseline.json`の対象外。供給元 (騎手コード・
場所・芝ダ・距離・クラス・日付・前走情報の有無) はいずれもserve_deadに無いため、F-serveにそのまま
保持した。

## 3. calibrated_v6_probability の自前再fit (リーク回避)

本番 `models/pl_calibrators_v6.pkl` はCLAUDE.md記載の通り valid=2023でfitされている。2023年の
評価にこの成果物をそのまま使うと較正器自身が2023年の分布を見ていることになり、spec §8が禁止する
「test期間でfitした較正器」に該当する。

そのため `build_features.py` は **train (2016-2021) のみ**でIsotonicRegression (`out_of_bounds="clip"`,
`y_min=0, y_max=1`、本番`build_pl_calibrators.py`と同じ設定) を2本 (win用・top3用) 独立に再fitし、
`calibrated_v6_probability` / `calibrated_v6_probability_win` として2016-2025の全行に適用した。
本番較正器そのものとの整合性は §5(以下)で本番較正器がtrueにOOSな2024-2025のみを使って別途確認した。

## 4. p_win同値問題 (spec §20) 実測

`tie_diagnosis.py` (train-refit isotonicに対して測定):

| 対象 | レース内最大タイ率 | タイ時の平均タイ頭数 |
|---|---|---|
| calibrated_v6_probability_win (自前train-refit) | 5.00% | 2.09頭 |
| calibrated_v6_probability (top3, 自前train-refit) | 3.85% | 2.05頭 |
| v6_score (生スコア) | 0.00% | — |
| v6_pwin (生PL勝率) | 0.00% | — |
| **本番pl_calibrators_v6.pkl (2024-2025、較正器にとって真にOOS)** | **8.08%** | 2.12頭 |

spec本文が言う「約18%」よりも実測値は低い(本番較正器実測でも約8%)。この差の原因は不明
(spec作成側が別の定義・別の年・別の丸め精度で測定した可能性が高い)。ただし**方向は確認できた**:
較正確率は生スコアに比べ明確にタイを生む(0%→4-8%)。タイの発生自体は稀だが構造として実在する。

タイのあるレースは無いレースよりv6◎(最大確率馬)の3着内率が低い(51.4% vs 60.5%)。Brier差は僅か
(0.1398 vs 0.1376)。よって「タイ問題が改善全体の主因」とまでは言えないが、v6較正確率(B1/M1の
入力)には情報損失があり得るという仮説自体は裏付けられる。B2(=M2, raw_v6_score+rank追加)/
B3(=M3, レース内相対特徴追加)でどこまで説明できるかはGate1/model_compareの結果を見て判定する
(REPORT.md参照)。

## 5. 表特徴の時点安全性

EXP04で既に検証済み (`exp04_invariant_info_dev/test_time_safety.py`, `test_environment_split.py`)。
EXP05はEXP04の候補パーケットをそのまま再利用しており、追加の特徴エンジニアリングは行っていない
(レース内相対特徴score_gap_*とcalibrated_v6_probabilityのみ新規で、いずれもv6_score/v6_pwinという
既に時点安全なスコアから導出される)。
