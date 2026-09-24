# EXP16A — rolling OOF と `retrospective_rolling_crossfit_development`（Stage 1 で実行、Stage 0 では未実行）

**作成日**: 2026-09-24　**改訂**: 2026-09-24（改訂2: 2019〜2023 の rolling crossfit を採用、artifact 契約を追加）
**目的**: Q2/Q3 の係数学習に in-sample な Q1 予測を使わないこと。および、年ごとに
**その年より前のデータだけ**で全要素を決めること。

---

## 1. なぜ必要か

Q2a/Q2b/Q3a/Q3b は `offset(market) + log(Q1)` の条件付きロジットである。
ここで使う Q1 の確率が **その年を学習または early stopping に使ったモデル**から出ていると、
係数が in-sample の当たりを拾い、Gate の判定が楽観側に歪む。

EXP15 の R0-clean は **2022 を early stopping に使っている**ので、
**その 2022 スコアを OOS として再利用しない**。

## 2. 開発設計の正式名称と位置づけ

**`retrospective_rolling_crossfit_development`**（2019〜2023）。

- これは**独立した未使用 holdout ではない**。2019〜2023 は過去研究で接触済みの期間を含む
  （v6 の valid=2023、較正器 fit、各種 mcond 実験など）。したがって位置づけは
  **仮説探索・内部再現性の評価**であり、「未使用データでの確認」ではない。
- **2024/2025 は封印を維持する**。開封条件はこの crossfit Gate 通過後に**別途レビュー**とし、
  spec に自動開封を書かない。

## 3. 評価年 Y の全要素（すべて Y−1 以前から作る）

| 要素 | 規則 |
|---|---|
| Q1 train | 年 ≤ Y−2 |
| Q1 early stopping | Y−1 |
| Q1 prediction | Y |
| Q2/Q3 係数 fit | **Y−1 以前に得た rolling OOF のみ** |
| 確率較正 | Y−1 以前のみ（`models/pl_calibrators_v6*.pkl` は valid=2023 fit なので**使用禁止**） |
| 温度 τ | Y−1 以前のみ |
| learned subset 境界（市場 entropy・favorite odds など） | Y−1 以前のみ |
| encoder / 欠損処理 / 標準化 | Y−1 以前のみ |
| placebo セル境界 | Y−1 以前のみ |
| 変種選択（残差特徴セット・shrinkage 量） | Y−1 以前のみ |

- **2022 で固定した値を 2019〜2021 へ遡及適用しない**（未来情報になる）。
- 固定値を使えるのは、結果を見ずに決めた**純粋なドメイン区分**だけ:
  芝／ダート、競馬場、事前固定した頭数帯 [5-8] [9-12] [13-15] [16-18]、事前固定したクラス区分。
- 学習で決まる境界の年別実測値は `out/race_population.json` の
  `learned_subset_boundaries_by_eval_year`（RACE_POPULATION_AUDIT.md §8）に記録済み。

## 4. 作成する年

| Q1 OOF 年 Y | train | ES / τ | 用途 |
|---|---|---|---|
| 2016 | ≤2014 | 2015 | 係数学習用 OOF |
| 2017 | ≤2015 | 2016 | 同上 |
| 2018 | ≤2016 | 2017 | 同上 |
| 2019 | ≤2017 | 2018 | **crossfit 評価年** ＋ 以降の係数学習用 OOF |
| 2020 | ≤2018 | 2019 | 同上 |
| 2021 | ≤2019 | 2020 | 同上 |
| 2022 | ≤2020 | 2021 | 同上 |
| 2023 | ≤2021 | 2022 | **crossfit 評価年**（最後に開封する） |

- 2016 の train は 2013-2014 の 2 年しかない。行数が不足する年は `oof_manifest.json` に
  train 行数・レース数を記録し、必要なら「係数学習に使うのは 2018 年以降の OOF のみ」と
  Stage 1 冒頭で**事前固定してから**実行する（結果を見てからの変更は禁止）。
- **2024/2025 は一切使わない**。2023 の OOF を作る時点でも ES は 2022 までで閉じる。
- 正式 race set（障害除外・DNF なし・平地）の各年レース数は
  STAGE0_DRY_RUN.json の `retrospective_rolling_crossfit_development` を参照。

## 5. 5 年判定（2019〜2023）

**主判定（すべて満たすこと）**

1. 2019〜2023 pooled の Δ（= LL(arm) − LL(baseline)）の median seed 推定が **≤ −0.005 nats/race**
2. **年層化 meeting-day bootstrap** の CI95 上限 < 0
3. 5 年中 **4 年以上**で改善方向
4. 5 seed 中 **4 以上**で改善方向
5. **最大効果年を除いた leave-one-year-out** でも 1. と 2. を満たす
   （単一年の大勝ちで pooled PASS にならないことを保証する hard 条件）

**必ず報告するもの**

- 各年の効果、各年の race 数・meeting-day 数
- pooled 効果と CI
- 年別異質性（年効果の分散・最大年と最小年の差）
- 最大年を除いた leave-one-year-out
- 2019〜2022 だけ / 2023 だけ
- seed ごとの値・median・方向一致数

## 6. seed 規約

- Q1 は **5 seed すべてについて OOF 予測**を作る（seed = 20260923〜20260927）。
- Q2/Q3 も **seed ごとに独立に fit**（seed 平均確率を先に作って 1 本にまとめない）。
- **主判定は median seed**、bootstrap CI も median seed の per-race 値に対して計算する。
- 方向一致は 5 seed 中 4 以上を要求する。

## 7. artifact 契約（`out/oof_manifest.json` に必須記録）

| 項目 | 内容 | Stage 0 時点の実測 |
|---|---|---|
| master_v2 absolute path | `E:\PyCaLiAI\data\master_v2_20130105-20251228.csv` | 記録済み |
| size | バイト数 | 518,835,750 |
| mtime | ローカル時刻 | 2026-09-11 22:32:42 |
| sha256 | ファイル全体 | `b8032d1b…9e780a76` |
| row hash | 2016-2023 の `rid16_ban` 昇順連結の sha256 | STAGE0_DRY_RUN.json に記録（EXP15 記録値との一致も判定） |
| feature schema hash | R0-clean 111 列を昇順連結した sha256 | 記録済み。現 master 実ヘッダに 111 列すべて在ることを再検証 |
| encoder hash | category encoder の内容 hash | **Stage 1**（未作成） |
| race population hash | `out/race_population.json` の sha256 ＋ official rid 一覧の sha256 | 記録済み |
| 各年の train/ES/predict 期間 | 実際に読んだ日付範囲と行数 | Stage 1 |
| 各 seed の model hash | LightGBM model と score npz の sha256、best_iter、τ | Stage 1 |
| P0-5 関連修正後 artifact か | 同一 (調教師, レース) ブロック内で `trainer_fuku30` が一定かで**実測判定** | 記録済み（判定結果は STAGE0_DRY_RUN.json） |
| `jockey_fuku90` provenance | train: `build_dataset.add_rolling_stats`（as-of, C1 2026-09-11）／serve: `data/jockey_stats.csv` スナップショット | 記録済み |
| `prev_hosei` provenance | train: `build_master_v2.py:70` で `前走補正` を rename（TARGET 由来）／serve: `make_weekly_hosei.py` の proxy（旧実装は前々走の off-by-one） | 記録済み |

- **EXP15 の hash を流用する場合も、今回読み込んだ実ファイルから再計算して一致を検証する**。
  不一致なら Stage 1 を開始せず、原因（master 再生成・列追加）を先に特定する。
- 予測ファイルは `data/_research/mcond/exp16a/scores/Q1_Y{year}_s{seed}.npz`（gitignore 対象）に置き、
  hash だけを manifest に残す。
- **EXP15 の既存 scores は再利用しない**（train/ES 期間が異なるため）。

## 8. 検出力の事前確認（Stage 1 の最初の門）

rolling OOF を作った**直後・結果を開封する前**に、`power_audit.py` を
`q1_log_ratio`（= log(Q1/π)）方向で再実行し、pass 確率を測る。
Stage 0 では Q1 が無いため事前固定した残差特徴の線形予測子と乱数方向でのみ測っている
（POWER_AUDIT.md §3）。

## 9. 計算量の見積り

LightGBM の 1 fit は EXP15 実測で 12〜23 秒（約 29 万行 × 111 列）。
8 年 × 5 seed = 40 fit ≒ 10〜15 分（train が短い年はさらに速い）。GPU 不要。

## 10. Stage 0 の時点で未実行であること

本計画の実行（＝学習）は Stage 1 に属する。Stage 0 では
`STAGE0_DRY_RUN.json` の `reference_values_2022_official_set.R0_clean_table_only` を空欄のままにし、
Stage 1 で rolling OOF を作ってから同一 race set 上で埋める。
