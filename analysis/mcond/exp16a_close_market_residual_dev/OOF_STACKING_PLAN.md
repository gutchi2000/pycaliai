# EXP16A — Q1 rolling OOF の作成計画（Stage 1 で実行、Stage 0 では未実行）

**作成日**: 2026-09-24　**目的**: Q2/Q3 の係数学習に R0-clean の in-sample 予測を使わないこと。

---

## 1. なぜ必要か

Q2a/Q2b/Q3a/Q3b は `offset(market) + log(Q1)` の条件付きロジットである。
ここで使う Q1 の確率が **その年を学習または early stopping に使ったモデル**から出ていると、
係数が in-sample の当たりを拾い、Gate の判定が楽観側に歪む。

EXP15 の R0-clean は **2022 を early stopping に使っている**ので、
**その 2022 スコアを OOS として再利用しない**。

## 2. 各評価年 Y の固定手順

| 項目 | 規則 |
|---|---|
| train | 年 ≤ Y−2 |
| early stopping / selection | Y−1 |
| prediction | Y |
| features | R0-clean 111列（`exp15_race_as_set_dev/out/feature_contract.json` の `features.clean`） |
| hyperparameter family | EXP15 で 2022 選択済みの `v6_lr_half`（lambdarank / trunc 5 / uniform weight / ES 100 / max 3000） |
| seeds | EXP15 と同じ 5 seed（20260923〜20260927） |
| 温度 τ | 各 (Y, seed) について **Y−1 のみ**で条件付きロジット最尤 |

**作成する年**: 2016〜2022（rolling OOF）＋ 2023（完全 OOS）。

| 評価年 Y | train | ES/τ | 用途 |
|---|---|---|---|
| 2016 | ≤2014 | 2015 | 係数学習用 OOF |
| 2017 | ≤2015 | 2016 | 同上 |
| 2018 | ≤2016 | 2017 | 同上 |
| 2019 | ≤2017 | 2018 | 同上 |
| 2020 | ≤2018 | 2019 | 同上 |
| 2021 | ≤2019 | 2020 | 同上 |
| **2022** | **≤2020** | **2021** | **subset・Gate 閾値・較正・係数選択に使う 2022 OOF** |
| **2023** | **≤2021** | **2022** | **development の完全 OOS 予測** |

- 2016 の train は 2013-2014 の 2 年しかない。行数が不足する年は `manifest.json` に
  train 行数・レース数を記録し、必要なら「2018 年以降のみを係数学習に使う」と Stage 1 冒頭で
  事前固定してから実行する（結果を見てからの変更は禁止）。
- **2024/2025 は一切使わない**。2023 の OOF を作る時点でも ES は 2022 までで閉じる。

## 3. 係数学習・較正の担当年

| 用途 | 使う年 | 根拠 |
|---|---|---|
| Q2/Q3 の係数 fit | 2016-2021 の rolling OOF | 各年が自分の学習に使われていない |
| 変種選択（Q2a/Q2b の残差特徴セット、shrinkage 量） | **2022 OOF のみ** | selection 年 |
| 確率較正（必要な場合） | 2016-2021 または 2022。`pl_calibrators_v6*.pkl` は **使用禁止**（valid=2023 fit） | 較正器リーク回避 |
| Gate 判定 | 2023 の完全 OOS 予測 | development |

## 4. manifest（必須）

`out/oof_manifest.json` に、年ごとに次を保存する。

```
{
  "2022": {
    "train_period": ["2013-01-05", "2020-12-28"],
    "es_period":    ["2021-01-05", "2021-12-28"],
    "predict_period": ["2022-01-05", "2022-12-28"],
    "n_train_rows": ..., "n_train_races": ...,
    "feature_contract_sha256": "...",
    "hyperparams": {...},
    "seeds": [20260923, ...],
    "artifacts": {"20260923": {"score_npz_sha256": "...", "best_iter": ..., "tau": ...}, ...},
    "code_sha256": {"train_r0.py": "...", "common.py": "..."}
  }, ...
}
```

- `feature_contract_sha256` は `exp15_race_as_set_dev/out/feature_contract.json` の hash。
- 予測ファイルは `data/_research/mcond/exp16a/scores/Q1_Y{year}_s{seed}.npz`（gitignore 対象）に置き、
  hash だけを manifest に残す。
- **EXP15 の既存 scores は再利用しない**（train/ES 期間が異なるため）。

## 5. seed 規約

- Q1 は **5 seed すべてについて OOF 予測**を作る。
- Q2/Q3 も **seed ごとに独立に fit** する（seed 平均確率を先に作って 1 本にまとめない）。
- 報告: seed ごとの全指標・median・5 seed 中の方向一致数。
- **主判定は median**、かつ **4/5 以上の方向一致**を要求する。

## 6. 計算量の見積り

LightGBM の 1 fit は EXP15 実測で 12〜23 秒（約 29 万行 × 111 列）。
8 年 × 5 seed = 40 fit ≒ 10〜15 分（train が短い年はさらに速い）。
GPU 不要。Stage 1 の最初の工程として実行する。

## 7. Stage 0 の時点で未実行であること

本計画の実行（＝学習）は Stage 1 に属する。Stage 0 では
`STAGE0_DRY_RUN.json` の `reference_values_2022_official_set.R0_clean_table_only` を空欄のままにし、
Stage 1 で rolling OOF を作ってから同一 race set 上で埋める。
