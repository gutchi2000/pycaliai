# DNF特徴意味定義 spec（修正前に固定、結果を見て変更しない）

**作成日**: 2026-09-22　**状態**: 意味定義の固定のみ。まだ本番コード・
`data/master_v2_*.csv`・`models/unified_rank_v6.pkl`は一切変更していない。
本specの固定後に実装差分・影響監査（`DNF_HISTORY_FEATURE_PARITY_AUDIT_20260922.md`）
を行う。

このspecはEXP13 Gate 0B/0Dで発見された19特徴（course_n_prev系6、kako5系13）
のDNF-population依存性issueに対する、P0監査（`analysis/mcond/
p0_dnf_history_parity_audit/`、EXP13・EXP14いずれの継続でもない独立監査）が
準拠する唯一の正解定義である。

## 対象母集団の三分類（着順コード基準、[[project_kekka_ext_data_quirks]]・
`LABEL_CODEBOOK.md`と同一の分類を流用）

| 分類 | 着順コード | 「出走経験」に数えるか | 備考 |
|---|---|---|---|
| 正常完走 | 数値着順 | 数える（win/top3判定も通常通り） | |
| 降着（丸数字） | ①②③... | 数える（走破タイム記録あり、正常完走扱い） | EXP13 LABEL_CODEBOOK.md§2で確定済み |
| **止（DNF）** | 止 | **数える**（出走経験1回、win/top3は不成立） | 本specの主対象 |
| 外（発走前除外） | 外 | **数えない** | 発走前確定、そもそも走っていない |
| 消（発走前取消） | 消 | **数えない** | 同上 |

## 1. course_n_prev / course_win_rate / course_top3_rate（同一場所×芝ダ×距離帯）

- **denominator（course_n_prev）**: 止を含む「出走経験」でcumcountする
  （現在のバグ: dropna後の母集団で計算しているため止がまるごと欠落し、
  発生後永続的に過小カウントされる）。外・消は含めない。
- **numerator（course_wins_prev / course_top3_prev）**: 止は勝利・複勝の
  いずれにも該当しない（着順が存在しないため）。既存コードの
  `_is_win = (着順==1)`, `_is_top3 = (着順<=3)`は着順NaNに対し自動的に
  False/0を返すため、**numerator側のロジックは変更不要**（母集団に
  止を含めてcumsumしても、止の寄与は常に0のまま）。
- **win_rate/top3_rate**: `numerator / denominator`のまま（分母だけが
  止を含む正しい値になる）。

## 2. jockey_n_prev / jockey_win_rate / jockey_top3_rate（同一馬×騎手ペア）

- course系と全く同型。denominatorに止を含める、numeratorのロジックは
  変更不要、外・消は含めない。

## 3. kako5_*（直近5走ウィンドウ）+ hist_same_cond/place_*（全キャリア）

### 3.1 window slot（「直近5走」を構成する走の集合）

- **止は「直近1走」としてwindow slotへ含める**（現在のバグ: dropna後の
  母集団を入力にしているため止が見えず、実際には6走前・7走前だった走を
  「直近5走以内」と誤認する——本来より古い走が繰り上がってウィンドウへ
  混入する）。
- **外・消は1走として数えない**（そもそも走っていないため、window slot
  を消費させてはならない）。
- 丸数字（降着）は正常完走と同じ扱いで既にwindow slotへ含まれている
  （変更なし）。

### 3.2 止の走における各フィールドの値

止の走をwindow slotへ含める際、`_compute_features()`が受け取る
`past_races`の1要素として、以下の値を設定する:

| フィールド | 値 | 理由 |
|---|---|---|
| 着順 | **None（欠損）** | 結果が存在しないため。数値を代入しない（0や最下位順位で埋めない） |
| 人気 | None（既存master入力では常にNone、変更なし） | |
| 上り3F | **None（欠損）** | 走破しなかったため上がりタイムが存在しない |
| TD（芝・ダ） | **実際の値**（その日出走したコース種別） | 出走自体はした事実であり、条件は既知 |
| 距離 | **実際の値** | 同上 |
| 場所 | **実際の値** | 同上 |

### 3.3 各派生特徴への影響（既存`_compute_features()`ロジックは変更しない）

上記フィールド設計により、`_compute_features()`自体のロジックを一切
変更せずに正しい挙動が得られる（既存コードは全てNone値を
`if r["着順"] is not None`等でフィルタしているため）:

- `kako5_avg_pos`/`kako5_std_pos`/`kako5_best_pos`/`kako5_pos_trend`:
  止の走は着順Noneのため`positions`リストから自動除外される
  （＝止自身の着順は集計に混ざらない）。
- `kako5_avg_agari3f`/`kako5_best_agari3f`: 同様に上り3F Noneで自動除外。
- `kako5_same_td_ratio`/`kako5_same_dist_ratio`/`kako5_same_place_ratio`:
  分母nは`len(past_races)`のため**止も1走として数える**。TD/距離/場所は
  実値を持つため、条件が一致すれば分子にも正しく寄与する
  （止の走が「同条件での出走」だった事実は活かす）。
- `kako5_race_count`: `n = len(past_races)`のため**止を含む**（「直近
  5走以内に何走したか」という定義になる。「有効な結果データが何件か」
  ではない、という点を明記する——この意味変更を許容する）。
- `kako5_expected_good_count`/`kako5_upset_good_count`/
  `kako5_hidden_good_count`: 着順・上り3F Noneのため止は「好走」として
  絶対にカウントされない（自動的に0寄与）。
- `kako5_same_cond_best_pos`: 着順Noneのため止は同条件ベストの候補から
  自動除外（正しい——完走していない走にベスト着順は存在しない）。
- `hist_same_cond_best_pos`/`hist_same_cond_top3_rate`/
  `hist_same_cond_count`/`hist_same_place_best_pos`（全キャリア版）:
  既存ロジックは着順NaNの走を`if np.isnan(pos): continue`で除外する
  設計であり、これは**window位置ではなく値ベースのフィルタ**のため、
  母集団に止を含めても含めなくても結果が変わらない（EXP13 Gate 0Bで
  実測済み: diff=0）。**この4特徴は本specの対象外**（既に無罪）。

## 4. 変更しないもの

- モデルのハイパーパラメータ（`unified_rank_v6.pkl`の`optuna_best_params`
  等）は一切変更しない。
- 92のstructurally_immune特徴・2のlikely_immune特徴（horse_fuku10/30）・
  4のconfirmed_immune特徴（hist_same系）は本specの対象外、変更しない。
- ラベル（`fukusho_flag`等）・split（train/valid/test）の定義は変更しない。
- 丸数字（降着）・外・消の扱いは`LABEL_CODEBOOK.md`の既存定義のまま
  変更しない。

## 5. 検証すべき不変条件（shadow artifact構築時に必須）

- 行数・レース数が元のmaster_v2と完全一致すること（母集団自体は増減しない、
  19特徴の値だけが変わる）。
- 結果ラベル（`fukusho_flag`, `roi_target`）が完全一致すること。
- 変更対象19特徴**以外**の101特徴が完全一致すること。
- 未来削除不変性: ある年のデータを削除しても、それより前の年の値が
  変わらないこと（[[feedback_asof_population_definition]]の削除不変性
  テストに準拠）。
- 同日情報を使用しないこと（当日馬体重等は元々使用していない、変更なし）。
- raw source（`lgbm/cat/torch_transformer/add_20130105-20251228.csv`、
  `data/Time _series_odds/`は無関係）から再生成可能であること。

関連: [[project_exp13_nonfinish_risk]] [[project_gate0b_feature_dnf_audit_20260922]]
[[project_kekka_ext_data_quirks]] [[feedback_asof_population_definition]]
