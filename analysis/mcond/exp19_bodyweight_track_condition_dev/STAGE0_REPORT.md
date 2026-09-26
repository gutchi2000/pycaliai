# EXP19 Stage 0 報告 — 当日馬体重状態 × JRA 公式馬場物理値

**仕様**: v0.2-frozen（commit `4a4c4903`）。本報告は Stage 0 の成果物だけを記録する。**Stage 1 は開始していない**。
2019〜2023 の実着順を使った A1/A2/B1/B2 の性能、2024/2025、ROI、候補生成、賭金、production 変更は行っていない。

---

## 0. 結論（Gate ごとの進行可否）

| 項目 | 判定 | 要点 |
|---|---|---|
| S0-A forward parity・T−28 | **未達（収集中）** | forward 収集は 2026-09-26 の 1 開催日だけ（完全 snapshot 7R / 99 馬行）。歴史 TARGET 値との対は 0 件（2026 年分の torch 形式 export が無い）。floor 4 日 / 400 行に対し残り 4 日 / 400 行 |
| S0-B 母集団・被覆 | **PASS** | 既知構造値を完全再現。主母集団 15,951R（EXP16A 基盤との差 0）。WP 同時利用可能率 2021 93.3% / 2022 99.6% / 2023 100%（各年 ≥ 90%） |
| S0-C time-safety・invariants | **PASS** | 合成 invariant / 構造テスト 30/30 |
| N1 `R0-clean-nobw` OOF | **PASS** | 110 列 × 40 fit、再現性 bit 一致、未来行違反 0、manifest 完備。結果評価はしていない |
| S0-D floor・検出力 A1 | **floor 定義不能（仕様の欠陥）** | 凍結式（真値 = pre 市場、決済 = terminal オッズ）では Δ=0 でも期待成長 0.0135/race。参考 MDE: SIGNAL 検出力 0.45（Δ=0.002）/ 0.95（Δ=0.004） |
| S0-D floor・検出力 A2 | **floor 定義不能 + 検出力不足** | Δ=0 成長 0.0144/race。参考 SIGNAL 検出力 Δ=0.008 でも 0.37 |
| S0-D floor・検出力 B1 | **PASS** | floor **0.006474** nats/race、床での SIGNAL 検出力 **0.925**（Wilson [0.895, 0.947]） |
| S0-D floor・検出力 B2 | **FAIL（検出力未達）** | floor **0.005154**、床での SIGNAL 検出力 **0.165**（要求 0.80） |
| S0-E 計算量 | 測定済み | W 全期間 43 秒、rolling fit 5 年 12.7 秒、B=10,000 bootstrap 0.06 秒、placebo 200 draw ≈ 27 分 / 種、最大 RSS 884 MB |

**Stage 1 へは進めない**。未達・停止理由:

1. S0-A: forward parity と T−28 complete coverage の floor に届いていない（収集中。値を推測しない・floor を緩めない）。
2. S0-D: A1/A2 の実務 floor が凍結仕様のままでは定義できない（§5）。版番号を上げた仕様（v0.3）が必要。
3. S0-D: B2 は検出力未達（停止規律 4）。A2 も参考検出力が低く、WP 経路は現行の 8 列設計では検出できる見込みが小さい。

---

## 1. loader（S0-B の loader 契約）

- `loaders.load_torch_struct()` は歴史 torch を **15 列の明示 whitelist** だけで読む。読み込み後に禁止列の具体名・prefix・結果/払戻パターンの不存在を assert する。
- 実ファイルには禁止列が 26 列あるが（`人気`、`単勝オッズ`、`複勝オッズ下限/上限`、`複勝シェア`、`補正`、`指時系1〜4・*`、`複上1〜4`、`複人気1〜4`）、whitelist の外なので一切読まない。
  - `複上N` / `複人気N` は `指時系` の prefix を持たないが、SPEC 本文の「指時系1〜4・単勝/人気/複下/複上/複人気」に該当するので具体名として禁止に加えた（spec.json の禁止リストには無い。§6 の v0.3 提案 4）。
- 2024/2025 行は読み込み直後に破棄して assert。市場 loader（TANPUK、EXP16A/18 と同契約）と結果 loader（finisher 判定専用）は別関数。
- loader sha256 `9c25ef45…`（`out/stage0_manifest.json` に全桁）。

## 2. N1 `R0-clean-nobw` rolling OOF（性能は未評価）

- EXP15 R0-clean 111 列から `斤量体重比` だけを除いた 110 列。`前走馬体重`・`前走馬体重増減`・`斤量`・`馬齢斤量差` は残した（assert 済み）。
- レシピは EXP16A と同一（関数を import して再利用）: train ≤ Y−2 / ES Y−1 / predict Y、Y = 2016〜2023、5 seed = **40 fit**、595 秒。
- 再現性: Y2019・seed 20260923 を再学習し、予測もモデル文字列も **bit 一致**。未来行検査は全年 OK。
- manifest（`out/oof_nobw_manifest.json`）: feature list sha256、入力 sha256（master_v2・行 cache・EXP15 契約・正式 set）、loader sha256、40 モデルの sha256、予測行 hash。
- `s_clean` は 5 seed の OOF score の平均とする（Stage 0 で固定）。

## 3. 母集団・被覆（S0-B）

- **既知構造値の再現**: 2019〜2023 平地 **231,068 行 / 16,645R**、current kg 有効率 **99.807%**、血統登録番号欠損 **0** — 完全一致。
- starter = terminal 単勝 > 1.0。torch の非 starter 行 1,863 行（取消・除外）は体重履歴の slot に入れない。DNF 馬は starter なので実測体重が slot に残る。
- **主母集団** = EXP16A 正式 set（DNF 含有 race 除外）で、全 starter が torch 行・pre/terminal オッズを持つ race。**15,951R、基盤との race ID 差分 0**（年別 3,185 / 3,169 / 3,200 / 3,206 / 3,191）。主母集団の全 starter が measured。
- **full-starter 感度母集団**（DNF 含有 race も残す）: 年 3,318〜3,326R、うち DNF 含有 119〜149R、N1 を持たない DNF 馬 122〜166 頭/年。N1 は 0 埋めしない。感度分析は market-only と W-only の比較だけ実行可能。
- **年別被覆（主母集団）**: `bw_robust_z5` 利用可能 79.7〜80.4%、`bw_change_x_layoff` 89.6〜90.0%、`bw_sex_age_z` 99.9%。芝の cushion 利用可能 race: 2019 0% / 2020 14.2% / 2021 89.8% / 2022・2023 100%。
- **WP 同時利用可能 race**（P 利用可能 かつ 全 starter measured）: 2019 47.0% / 2020 56.9%（moisture-only の副年）、**2021 93.3% / 2022 99.6% / 2023 100% → floor 90% を各年で満たす**。
- **層別被覆（全体の 0.8 倍未満）**: `bw_robust_z5` は 2 歳（32.8%）と初出走（0%）で低い（履歴 2 走未満は定義上欠損）。measured と WP 同時利用可能は該当層なし。
- 原票の `馬体重増減` と as-of 前走実測からの再計算: 一致 約 87%、as-of 前走なし 約 10%、不一致 約 2%、原票欠損 0.2〜0.3%。不一致は記録だけで値を書き換えない。

## 4. 特徴と Stage 0 で固定した値（分布だけで固定、結果不使用）

| 項目 | 固定値 | 根拠 |
|---|---|---|
| MAD floor | **2.0 kg** | max(記録刻み 2 kg, ≤2018 の生 MAD の p10 = 0)。MAD = 0 が 16.7% |
| winsorize | **±7.5** | ≤2018 の \|z\| の p99.5 = 7.42 を 0.5 刻みで切り上げ（切られる行 0.39%） |
| 履歴必要数 / 最大 | 2 / 5 | SPEC の既定 |
| 性別×年齢×時期の基準 | 四半期、対象年より前の年だけ、セル最小 30 件 | — |
| 馬場の as-of 標準化 | 前日までの venue-day（shift 1 の expanding）、最小 10 venue-day | — |
| `track_extreme_z` | 芝 = max(\|cushion_z\|, \|moist_gp_z\|)、ダ = \|moist_gp_z\| | 結果を使わず一意に定義 |
| WP の列 | 6 本を 8 列（#3・#4 は芝/ダ別の列、#1・#2 は芝のみ） | SPEC の「surface 別」を係数分離で実装 |
| 欠損の扱い | 欠損指示子 + 定数 FILL（指示子が吸収するので FILL の値は尤度に影響しない。テストで確認） | 0 kg・増減 0 とはみなさない |
| 歴史の `bw_status` | measured / not_measured の 2 値 | 歴史 torch は 999 と原票欠損を区別できない |

- `bw_change_kg`・`bw_change_pct`・`bw_dev_med5_pct` は説明用だけで、W にも WP にも入れていない。

## 5. 実務 floor と検出力（S0-D、EXP18 v0.5 方式、結果ラベル不使用）

**方式**: 生成 `log p = log m_base + ε·c − log Z`。c は block（A1/B1 = W 設計 10 列、A2/B2 = WP 8 列）の race 内中心化・標準化後の第 1 主成分方向（構造データだけで凍結）。真の Δ は、期待 logloss 最良の null arm に対する KL の評価 race 平均。推定器は実際の offset conditional-logit を、各年 Y の実 fit 窓 race 数と同数の合成 outcome で fit する（宣言ノイズなし）。決済は実 terminal 単勝オッズ上の現金込み Kelly。成長閾値 1e-4/race（年約 3,300R で log 成長 0.33、約 +39%/年に相当）。seed 20260926、成長 24 rep × 15 格子（0〜0.1）、検出力は床 400 rep・0.5 倍 / 2 倍 / Δ=0 各 200 rep、bootstrap B=10,000。placebo 条件は通過と仮定（宣言）。A は Holm の保守側 α/2 で近似。

**Δ=0 健全性（真値 = base 市場のときの oracle 期待成長）**: A1 **0.01355** / A2 **0.01443** / B1 0 / B2 0。

- **A1/A2 は凍結式では floor を定義できない**。真値を pre 市場に置き、決済を terminal オッズで行うと、pre → terminal のオッズ変化だけで正の期待成長が出る（効果ゼロで閾値の 135 倍）。「成長 ≥ 1e-4 となる最小の Δ」が 0 に潰れる。値は作らず `null` とした。

| Gate | fit 窓 n_fit（Y 別） | 評価 race | floor（nats/race） | 床での SIGNAL 検出力 | PASS_PRACTICAL 検出力 | 0.5 倍 / 2 倍 / Δ=0 の SIGNAL |
|---|---|---|---|---|---|---|
| B1 | 9,550〜22,310 | 15,951 | **0.006474** | **0.925** [0.895, 0.947] | 0.0025 | 0.88 / 0.955 / 0.00 |
| B2 | 3,537〜9,714 | 9,368 | **0.005154** | **0.165** [0.132, 0.205] | 0.000 | 0.005 / 0.635 / 0.00 |

- **B1**: 成長曲線は Δ=0.004 で 1.09e-5、0.0075 で 1.47e-4（MC SE ≤ 5e-6）。有限標本 fit は oracle の 78〜87%。**進行条件（≥ 0.80）PASS**。
- **B2**: fit 窓が小さい（2021 評価は 3,537R）うえに WP 8 列 + W 10 列を推定するので、効果ゼロでも推定 Δ の中央値が +0.003（過学習のコスト）。Δ=0.005 では推定器の期待成長が負（−0.000014、oracle は +0.0028）、Δ=0.0075 でも oracle の 38%。floor での検出力は 0.165。**停止規律 4（検出力未達）に該当**。
- **A1 の参考 MDE**（決済に依存しない SIGNAL 検出力、100 rep）: Δ = 0.001 → 0.04、0.002 → 0.45、0.004 → 0.95、0.008 → 0.93。MDE ≈ 0.003 nats/race。
- **A2 の参考 MDE**: Δ = 0.004 → 0.06、0.008 → 0.37。0.008 でも 80% に届かない。
- damped Newton への fallback: B1・A1 は 0 回、A2/B2 は数回（完全分離に近い合成 draw）。

**実装上の注記**（結果を見る前・commit 前に直した）:
- 初回実行で Δ=0.3/0.5 の合成真値が完全分離に近づき、Newton が特異行列で停止した。格子の上限を 0.1 に下げた（単勝市場で真の Δ=0.1 nats は巨大な効果量）。
- Kelly に `0·log 0 = 0` の極限処理を追加した（EXP18 の関数と通常ケースで差 0 をテストで確認）。
- 列選択は Gram 行列で SVD と同一判定にした（テストで確認）。
- EXP18 の Newton が特異 Hessian で止まったときだけ、同じ目的関数の damped Newton を使う（同一 MLE をテストで確認）。

## 6. v0.3 として提案する変更（本 Stage 0 では適用していない）

1. **A1/A2 の floor 生成式**（必須）。凍結式は Δ=0 で正の成長を生むので floor が定まらない。候補:
   - (a) 真値を terminal 市場に置き、A の推定器（pre 市場 arm）で q を作り、terminal オッズで決済する。pre の情報不足はそのまま損失として計上される保守的な定義で、Δ=0 の成長は ≤ 0 になる見込み。
   - (b) A の floor を、同じ block の B の floor と同値と定める（経済価値は terminal 決済でしか実現しないため）。
   - (c) 真値を pre に置き、pre オッズで仮想決済する（内部整合的だが実際の決済ではない）。
   私の推奨は (a)。ただし採否と方式の固定は Fable の判断とする。
2. **WP 経路の検出力**（必須の判断）。凍結の 8 列設計では B2 の検出力が 0.165、A2 は Δ=0.008 でも 0.37。
   - (i) WP 経路をこのまま停止規律 4 で閉じる。
   - (ii) 結果開封前に WP を少数の事前固定項（例: #1 と #3 のみ）へ縮約し、検出力を再計算する。
   (ii) は探索にならないよう、結果を見る前に 1 回だけ固定する必要がある。
3. **`measurement_age_minutes`**: 歴史 `baba_feats.parquet` に測定時刻の列が無く、歴史 fit で信頼度の重みに使えない。監査専用（当日 JSON）に格下げするか、歴史の測定時刻を別途取得する。
4. **禁止列リスト**: spec.json の `forbidden_exact` に `複上1〜4`・`複人気1〜4` を追加する（実装済み。SPEC 本文と一致させるだけ）。
5. **C_TRACK_ONLY（負対照）と nested control**（`W_BODY + 既存 baba block`）: `baba_eval.py` の block を移植していないため未実装。Stage 1 前に移植するか、負対照を省く判断が要る。
6. §4 で Stage 0 に固定した実装細目の確認（四半期、最小件数、WP の 8 列化、欠損指示子方式、`s_clean` の 5 seed 平均、A の Holm 近似）。

## 7. forward parity（S0-A）の現状と必要なもの

| 項目 | 実測 | floor |
|---|---|---|
| 収集開催日 | 1（2026-09-26。11:41 開始のため当日 24R 中 10R だけ試行） | 4 日 |
| 完全 snapshot | 7R / 99 馬行（weight 全馬 normal。うち 29 行は増減欄が未記載の早い snapshot） | 400 行 |
| 最初の完全 snapshot の発走差 | −28.2 〜 −50.0 分（中央値 −45 分） | — |
| T−28 までに完全 | 観測可能 7R 中 7R | 95% |
| 歴史 TARGET 値との対 | **0 件**（torch export は 2025-12-28 まで） | 一致 99.5%、race coverage 99% |
| baba_today と保存値 | 照合不能（baba_feats は 2025-12-28 まで） | — |

**必要なもの**: forward を集めた開催日について、TARGET から torch と同じ形式（`馬体重`・`馬体重増減` を含む）で export したファイル。同じスクリプト（`forward_audit.py`）で照合する。4 開催日 / 400 行に届くまで Stage 1 には進めない。

## 8. 成果物

`loaders.py` `features.py` `fix_w_params.py` `build_oof_nobw.py` `population.py` `power_data.py` `models.py` `gate_grade.py` `placebos.py` `power_floor.py` `forward_audit.py` `compute_dry_run.py` `test_invariants.py` `stage0_checks.py`、
`out/`（`oof_nobw_manifest.json` `oof_nobw_checks.json` `w_param_fixing.json` `population_coverage.json` `stage0_manifest.json` `power_data_meta.json` `power_floor.json` `forward_parity.json` `invariant_tests.json` `compute_dry_run.json`）。
大きな中間生成物（OOF score、特徴 parquet、power 配列）は `data/_research/mcond/exp19/`（git 管理外）。
