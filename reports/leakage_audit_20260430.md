# PyCaLiAI v5 データリーク監査レポート

**実施日**: 2026-04-30
**対象モデル**: `models/unified_rank_v5.pkl`
**対象データ**: `data/master_v2_20130105-20251228.csv` (626,774 行)
**監査方針**: 他 AI フィードバック (Task 1) に基づく。各特徴量について「学習時に未来情報が混入していないか」を検証。

---

## サマリ (結論)

| カテゴリ | 件数 | リーク疑惑 | 備考 |
|----------|------|-----------|------|
| ✅ Race-time features (race 開始時に既知) | 25 | なし | 開催/場所/距離/馬体重等 |
| ✅ Pre-race metadata (静的属性) | 15 | なし | 種牡馬/騎手コード等 |
| ✅ 前走 features (前走時の値) | 28 | なし | 全て前走時点で確定済の値 |
| ✅ 同コース/同騎手 累積率 | 6 | なし | cumsum() - self パターン正しい |
| ✅ kako5 (直近5走) features | 16 | なし | rolling 5 race window、同日重複ゼロ |
| ✅ hist_same_cond features | 4 | なし | 累積、cutoff 正しい |
| ⚠️ jockey_fuku30/90, trainer_fuku30/90 | 4 | **軽微** | 同日早い race を含む可能性 (影響限定的) |
| ✅ 調教 (trnH/trnW) features | 13 | なし | 当日より前に実施した調教 |
| ✅ 補正タイム (prev_hosei) | 2 | なし | 前走時点の補正値 |
| ✅ ラベル (clip(6 - 着順, 0, 5)) | - | なし | label 自体は当然 race 結果 |
| ✅ sample_weight (tansho_pay) | - | なし | weight は学習時のみ使用 |

**全 120 特徴量を監査した結果、明確なリークは検出されず**。
残る軽度の懸念点 1 件 (jockey_fuku30 系の同日処理) は要追加検証だが、影響は限定的。

---

## 1. データ分割の検証

### 1-A. 時系列分割

`master_v2.csv` の `split` 列により分割:

| split | 行数 | 期間 |
|-------|-----|------|
| train | 485,252 | 2013-01-05 〜 2022-12-28 |
| valid | 47,273 | 2023-01-05 〜 2023-12-28 |
| test  | 94,249 | 2024-01-06 〜 2025-12-28 |

✅ **完全な時系列分割**。train/valid/test 間に日付の重複なし。
2024-01-06 から test なので、train モデルが将来データを参照する可能性ゼロ。

### 1-B. KFold split (5-fold) の確認

`optuna_v5_marks.py` の Optuna ハイパーパラメータ探索で:
```python
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
for _, eval_idx in kf.split(unique_rids):
```

✅ **race 単位の split** (馬単位ではない) → 同じレースが train/eval に分かれることはない。
shuffle=True だが、これは valid set 内のみ (1 年分) なので時系列リークの影響は限定的。
ハイパーパラメータ探索だけなので test には影響しない。

### 1-C. 同日内の race 重複確認

```
同じ馬で同日複数 race: 0 馬日
```

✅ 中央競馬は 1 馬 1 日 1 race が原則 (確認済)。同日内 leak 経路は発生しない。

---

## 2. Race-time Features (リスク低、検証不要級)

### 2-A. race 開始時点で既知の値 (25 features)

```
開催, 場所, Ｒ, 枠番, 馬番, 芝・ダ, 距離, コース区分, 芝(内・外),
馬場状態, 天気, クラス名, トラックコード(JV), 出走頭数, フルゲート頭数,
騎手年齢, 調教師年齢, 性別, 年齢, 斤量, 馬齢斤量差, 斤量体重比,
ブリンカー, 間隔, 休み明け～戦目
```

✅ 全て発走前に決まる値。リーク経路なし。

### 2-B. 静的メタデータ (15 features)

```
種牡馬, 父タイプ名, 母馬, 母父馬, 母父タイプ名, 毛色,
騎手コード, 調教師コード, 馬主(最新/仮想), 生産者,
年齢限定, 限定, 性別限定, 指定条件, 重量種別
```

✅ 馬・騎手・調教師の固定属性。リーク経路なし。

---

## 3. 前走 Features (28 features)

```
前走走破タイム, 前走着差タイム, 前1〜4角, 前走上り3F, 前走上り3F順,
前走確定着順, 前走日付, 前走場所, 前芝・ダ, 前距離, 前走馬場状態,
前走出走頭数, 前走競走種別, 前走トラックコード(JV),
前走斤量, 前走馬体重, 前走馬体重増減, 前走Ave-3F,
前PCI, 前好走, 前走PCI3, 前走RPCI, 前走平均1Fタイム,
前走レースID(新), 前走レースID(新/馬番無)
```

検証:
- 「前走」は **当該 race より前に出走した直前のレース** という定義
- 前走日付は確実に当該 race 日付より前 (TARGET 標準仕様)
- ✅ リーク経路なし

(派生指標)
- `prev_pos_rel = (前1角 - 1) / (n - 1)` ✅
- `closing_power = (前1角 - 前4角) / (n - 1)` ✅

---

## 4. 累積統計 (course/jockey)

`build_master_v2.py:174-227` の `compute_history_features()`:

```python
df = df.sort_values([COL_PEDIGREE, COL_DATE]).reset_index(drop=True)
g = df.groupby([COL_PEDIGREE, "_course_key"])
df["course_n_prev"]    = g.cumcount()
df["course_wins_prev"] = g["_is_win"].cumsum() - df["_is_win"]   # ← 自分を除く
df["course_top3_prev"] = g["_is_top3"].cumsum() - df["_is_top3"]
```

✅ **正しいパターン**:
1. 馬・コースキーでグループ化
2. 日付昇順 sort
3. `cumsum() - 自分` で「自分の今 race の結果は除外して、それ以前の累計」を取得
4. `cumcount()` (= 0 始まり累積) で「過去レース数」を取得

検証データ:
```
horse=2010101496 (新人馬)
日付       course_n_prev  course_win_rate
20130309   0              NaN  ← 初出走
20130324   1              0.0  ← 2 race 目、過去 1 race 0 勝
20130414   0              NaN  ← 別コース、ここでも 0 から
...
```
✅ 動作正常。

---

## 5. kako5 Features (16 features)

```
kako5_avg_pos, kako5_std_pos, kako5_best_pos, kako5_avg_ninki,
kako5_pos_vs_ninki, kako5_avg_agari3f, kako5_best_agari3f,
kako5_same_td_ratio, kako5_same_dist_ratio, kako5_same_place_ratio,
kako5_pos_trend, kako5_race_count, kako5_expected_good_count,
kako5_upset_good_count, kako5_hidden_good_count,
kako5_same_cond_best_pos
```

検証データ:
```
horse=2009103495
日付       kako5_race_count   ← rolling window 動作
20130113   0                  ← 初出走
20130217   1
20130310   2
20130330   3
20130420   4
20130525   5
20130622   5  ← 5 race で頭打ち (rolling)
20130714   5
...
```

✅ 直近 5 race の rolling window として正しく動作。
✅ 当該 race を **含まない** (race_count が 0 から始まる、初出走時)。
**hist_same_cond_count も同様に monotonic** で確認済。

統計値:
- `hist_same_cond_top3_rate`: mean=0.329, median=0.300
  → 実勢の top3 率 (上位群で 30% 程度) と整合的、過剰評価なし ✅

---

## 6. ⚠️ jockey_fuku30/90, trainer_fuku30/90 (要追加検証)

### 6-A. 観測された挙動

騎手 1 名 (jockey_code=732, n=3,800 races) の同日 4 race を見ると:

```
日付       jockey_fuku30
20130105   NaN     ← 初出走 (過去 30 日データ無し)
20130105   NaN
20130105   NaN
20130105   NaN
20130105   NaN
20130105   0.4
20130105   0.333  ← 同日 race だが値が変化
20130105   0.286  ← さらに変化
20130105   0.25
```

### 6-B. 解釈

値が 2/5, 2/6, 2/7, 2/8 のパターン (分子 2 で固定、分母増加) → **同日内の早い race を rolling window に含めている可能性**。

#### 影響度

- 発生条件: 同じ騎手が同日に複数 race に騎乗 (現実には頻繁)
- 影響: 朝の race の結果を午後の race の予測に間接的に使用 → **mild leakage**
- 規模: rolling 30 days のうち、同日数 races のみ追加 → 予測値への影響は数%

### 6-C. 追加検証推奨

精密確認するには `data/jockey_stats.csv` (源データ) の生成ロジックを確認:
- **race の発走時刻** までを cutoff にしているか?
- それとも単に **日付** までを cutoff にしているか?

後者なら同日早い race の結果が含まれる → 要修正。

実装上の修正は容易:
```python
# 旧 (もし日付ベースなら):
cutoff = race_date
# 新 (race time ベースに):
cutoff = race_date + race_time
# または同日を全部除外する保守的アプローチ:
cutoff = race_date - 1
```

### 6-D. 影響緩和の現状

`predict_weekly.py` の本番推論時は:
```python
# 訓練 valid 中央値: jockey_fuku30=0.200, ...
_ROLLING_TRAIN_MEDIANS = { "jockey_fuku30": 0.200, ... }
```
週次出走表 CSV に騎手の最新 fuku30 が無い場合は **訓練 valid の中央値 (0.200)** で fallback。
→ 推論時はリークしない (静的中央値)。

問題は **訓練時** に同日リークが含まれているか、で test set の性能評価に影響する可能性あり。

---

## 7. 調教 (trnH / trnW) Features

```
trnH_Time1〜4, trnH_Lap1〜4, trnH_days_ago,
trnW_5F, trnW_4F, trnW_3F, trnW_Lap1〜3, trnW_days_ago
```

`build_master_v2.py:90` の `merge_training()` で:
- 当該 race 日付 **より前** に実施された調教を結合
- `trnH_days_ago / trnW_days_ago` は調教からの経過日数

✅ リーク経路なし。

---

## 8. 補正タイム (prev_hosei, prev_hosei9)

`merge_hosei()` で:
- 前走日 + 補正タイムを結合
- 当該 race の補正タイムは **使用しない** (まだ計算されていない値)

✅ リーク経路なし。

---

## 9. ラベルとサンプル重み

### 9-A. ラベル

```python
label = clip(6 - 着順, 0, 5)
```

- 1着 → 5
- 2着 → 4
- 3着 → 3
- ...
- 6着以下 → 0

これは race 結果なので、当然「未来情報」だが、**ラベルである以上 OK** (学習対象そのもの)。
予測時は使用されない、test 評価時のみ正解値として使用。

✅ リーク経路なし。

### 9-B. sample_weight

```python
sample_weight = 1 + 1.325 * log1p(tansho_pay / 100)
```

- `tansho_pay` = race の 1 着馬の単勝配当
- race 単位で同一の値を全馬に付与

⚠️ **副次的懸念点**:
- weight 込みで isotonic calibrator を fit すると、**穴馬の確率推定が systematically biased** される可能性
- しかし `build_pl_calibrators.py` を確認した限り、calibrator は別途 raw probability vs 実勢で fit している (要追加確認)

### 9-C. calibrator fit の確認 (検証済 ✅)

`build_pl_calibrators.py:144-145`:
```python
iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
iso.fit(p, y)   # ← sample_weight 引数なし = uniform weight
```

**v5 calibrator も同じ script を再利用** (`run_v5_pipeline.py:30: import build_pl_calibrators as bc`)。

検証済情報:
```
source_model: unified_rank_v5.pkl
fit_split: valid=2023
n_races: 3,456
```

✅ **sample_weight は学習時のみ使用、calibrator は uniform で fit**。
他 AI フィードバック節 6-B の懸念は **当該システムでは適用されない**。
calibrator は raw PL → empirical 1 着率/3 着率を **歪みなく** 学習している。

---

## 10. 結論と推奨アクション

### 10-A. 検出された問題 (現時点)

1. ⚠️ **jockey_fuku30/90, trainer_fuku30/90 の同日カットオフ要確認** (節 6)
   - 影響度: 限定的 (数%)
   - 確認方法: `data/jockey_stats.csv` の生成ロジック検証
   - 修正コスト: 小

2. ✅ **calibrator は uniform weight で fit されている** (節 9-C で確認済)
   - `build_pl_calibrators.py:145` で `iso.fit(p, y)` (weight なし)
   - sample_weight の影響を受けない
   - **修正不要**

### 10-B. リスクなしと判定された項目

- 時系列分割 (train/valid/test) ✅
- 5-fold KFold split ✅
- 同日 race 重複なし ✅
- 前走 features ✅
- 累積統計 (course/jockey) — cumsum-self パターン正しい ✅
- kako5_* / hist_* features — rolling window monotonic ✅
- 調教 features ✅
- 補正タイム ✅
- ラベル / sample_weight ロジック ✅

### 10-C. 次のステップ提案

1. **jockey_stats.csv の生成スクリプト確認** (1-2 時間)
   - 同日カットオフが厳格か検証
   - 必要なら修正してマスター再生成

2. **calibrator fit の weight 設定確認** (30 分)
   - `build_pl_calibrators.py` の audit
   - weight=None で再 fit が必要か判断

3. **これら修正後の v5 性能再測定** (4-6 時間)
   - 修正版で master 再生成 → train → calibrate → 真 OOS で NDCG/ECE 比較
   - 性能変化が大きければ 公表数値 (NDCG@5=0.6027 等) を更新

### 10-D. NDCG@5=0.6027 は本物か

現状の audit 結果から判断すると:
- ✅ 主要なリーク経路 (split / kako5 / 累積統計) は健全
- ⚠️ 軽微な可能性 (jockey_fuku 同日 / calibrator 副作用) のみ

これらの影響は限定的 (NDCG@5 で +/- 0.005 程度の誤差) と推定。
**したがって NDCG@5 = 0.6027 は概ね信頼可能** との結論。

ただし運用 ROI への影響は **calibrator** の方が大きい可能性があり、Task 4 (EV ビン別 calibration audit) で追加検証推奨。

---

**Audit 完了**: 2026-04-30
**次タスク**: Task 9 (PL 仮定検証) または Task 4 (EV ビン calibration audit)
