# EXP15 — 最小反証実験計画 v2（Race-as-a-Set / race context 固有効果の識別）

**改訂**: v2（2026-09-23、ユーザーレビュー11項目を反映）。v1 は commit `11845b5b`。
**状態**: `spec.json` を凍結し、その後 Stage 1 を実行する。機械が読む正本は `spec.json`、
本書はその説明である。**両者が食い違う場合は `spec.json` を優先する**。

## 0. 位置づけ

本研究は**「完全新規アーキテクチャ」ではない**。旧 `RaceTransformer`（2026-03、同一レース
全頭の self-attention）が既に存在し、単独成績は低かった（監査 §3.A）。ただしその比較は
特徴・master・選択指標が交絡していた。本実験は、次の条件をそろえて
**race context 固有の効果を識別する再設計実験**である。

- 同一入力
- 同一期間
- 同容量の no-context 対照

### v1 からの主な変更

| # | 変更 | 概要 |
|---|---|---|
| 1 | Gate を二段に分離 | Context Gate / Replacement Gate |
| 2 | R0 を二種類に分離 | R0-clean / R0-production-reference |
| 3 | 時間分割 | 2023 年で学習しない fixed-model 評価へ変更（v1 の 2023 forward-chaining は廃止） |
| 4 | seed | 5 seed を対にし、安定性を報告する |
| 5 | placebo | 仮説別（C1/C2）に再設計。P1 は不変性テストへ移す |
| 6 | raw ID 列 | 除外を正式化 |
| 7 | 容量対照 | 報告項目を追加 |
| 8 | MDE | 学習前に算出する |
| 9 | Gate 指標 | 再定義 |
| 10 | 既存文書 | ERRATUM を反映（本改訂と同一 commit） |
| 11 | Stage 1 | 範囲を限定 |

---

## 1. 仮説

| 記号 | 内容 | 検定する Gate |
|---|---|---|
| H-ctx | 入力列・horse encoder・容量・optimizer・loss・batch・seed を全てそろえたとき、race context の経路を持つモデル（R1）は、持たない双子（R1-noctx）よりも 2023 年の race-level win logloss と Brier が小さい | Context Gate |
| H-C1 | その改善は、条件（出走頭数・場・芝ダ・クラス）が一致する別レースの文脈では再現できない。つまり**そのレース固有の出走馬構成**に情報がある | C1 Gate |
| H-C2 | その改善は、構成統計を近似的に保った別の組み合わせでも再現できない。つまり**特定の馬同士の関係**に情報がある | C2 Gate |
| H-rep | R1（のちに R2）は R0-clean を置き換える価値がある | Replacement Gate |

---

## 2. 時間分割（fixed-model development）

| 用途 | 期間 |
|---|---|
| base training | 2016-01-01 〜 2021-12-31 |
| model selection / early stopping / 温度 τ / ハイパラ凍結 | 2022-01-01 〜 2022-12-31 |
| development evaluation | 2023-01-01 〜 2023-12-31 |

- **全モデルを 2022 年末で凍結**し、2023 年全体を純粋な fixed-model development として評価する。
  2023 年の行は学習・early stopping・較正・選択のいずれにも使わない。online refit も行わない。
- **2024-01-01 以降の行は読み込み時に chunk 単位で捨てる**（保持しない）。
  ローダーで assert し、読み込んだ行の hash をログに残す。
- **探索範囲は `spec.json` で事前に固定**し、選択は 2022 年だけで行う。
  2023 年の結果を見てからアーキテクチャやハイパラを選び直すことはしない。
  選び直した場合、その 2023 年の値は最終検定値として扱わない。
- bootstrap の単位は meeting_day（`rid16[:10]` = 日付＋場所）。

---

## 3. 入力契約（clean feature contract）

- **出発点**: v6 の `feature_cols`（120列）と v6 の変換
  - カテゴリ28列は LabelEncoder
  - それ以外は `pd.to_numeric(errors="coerce")`
- **正式入力から除外する列**
  - `馬主(最新/仮想)`（最新値で上書き＝as-of 違反）
  - `前走レースID(新)`・`前走レースID(新/馬番無)`（raw ID）
  - race_id・血統登録番号・馬名（元々 v6 の対象外）
  - その他の一意識別子（カーディナリティが base_train 行数の 1% を超える列は、許可リストに無ければ停止）
- **v6 で定数になっていて情報ゼロの列**: v6 の変換後、base_train での被覆率が 0% の列は、
  v6 が −9999 定数として受け取っていた列である。**両モデルから除外**し、情報集合を v6 と
  同一に保つ（候補: `開催`・`前走走破タイム`・`前走着差タイム`・`母馬`・`前走斤量`・`斤量`。
  実測は契約スクリプトで行う）。これらを正しく読み直すのは別の ablation とし、今回は行わない。
- **R0 と NN の表現の違い**: R0-clean と NN は**同一の列・同一の値**を受け取る。
  - カテゴリ: 同じ base_train fit の語彙を、R0 は整数として、NN は埋め込みとして使う。
  - 数値: 同じ coerce 後の値。R0 は欠損を −9999、NN は中央値補完＋欠損フラグ＋robust scale。
- **派生情報は今回の基本モデルに入れない**: 過去同一レース経験数・過去対戦回数・as-of 集約値
  などは許可された派生情報だが、別の ablation とする。
- **DNF の扱い**: master_v2 は dropna 後の母集団なので DNF 馬の行が無い。R0 と条件を
  そろえるため、主解析はこのまま使う（DNF を含む文脈は Stage 1 の範囲外）。

---

## 4. モデル

| 記号 | 用途 | 構成 |
|---|---|---|
| **R0-clean** | 正式な科学比較の基準 | LightGBM lambdarank（trunc 5、ラベル clip(6−着順,0,5)、重みは一様）。§3 の clean 入力。2022 年の NDCG@5 で early stopping。ハイパラは v6 の値を中心にした4点格子から 2022 年で選択 |
| R0-prodref | 現行 v6 との差を説明する参考値（**Gate に使わない**） | R0-clean と同じレシピだが、v6 の120列そのまま（疑義列を含む）と v6 の race 重み α=0.0308。本番モデルには触れない |
| **R1-noctx** | Context Gate の対照 | 馬ごとの NN。φ(x_i) → ρ_noctx(h_i)。ρ の幅を調整して**パラメータ数を R1 の ±5% に合わせる** |
| **R1** | 検定対象 | DeepSets。h_i = φ(x_i)。自分を除いた平均 m_{−i} と自分を除いた最大 M_{−i}、log n から文脈 c_i を作り、s_i = ρ([h_i, c_i, h_i − m_{−i}]) |
| R1-noctx-2x | 容量の診断 | パラメータ数を R1 の約2倍にした R1-noctx |

**R1 と R1-noctx で共通にするもの（違いは文脈経路の有無だけ）**:
- φ の構造と幅
- 損失: rel に対する pointwise MSE（レース内で結合しない）
- optimizer: AdamW
- batch: 256 レース
- seed の組
- 入力列
- early stopping の規則: 2022 年の τ 補正済み win logloss、patience 4、最大 30 epoch
- ハイパラ（8点格子から、R1 と R1-noctx の 2022 年 win logloss の平均が最小の1点を選び、両者に適用）

BatchNorm は使わない。

**R2 以降**: Stage 1 では作らない（Context Gate の判定前は禁止）。

---

## 5. seed

- **5 seed**: 20260923〜20260927。R1・R1-noctx・R1-noctx-2x・R0 は**同じ seed を対にする**。
  ハイパラ選択には別の seed 7 を使う。
- **seed ごとに固定するもの**: データ順（epoch ごとの shuffle 用 `torch.Generator`）、初期化、
  dropout、early stopping の規則、最大 epoch。`torch.use_deterministic_algorithms(True)` を使う。
- **報告**: seed ごとの全指標、median、min/max、seed 間 SD、方向が一致した seed の数、
  meeting-day の paired bootstrap。
- **主推定**: レースごとの差 Δ_r = mean_s[L_r(A_s) − L_r(B_s)]（seed で対にした差の平均）。
  ensemble 予測は使わない（ensemble による容量の上乗せを避けるため）。
  **最良 seed だけを採用することはしない**。

---

## 6. 指標

| 区分 | 指標 | 定義 |
|---|---|---|
| **主** | race-level win logloss | −(1/R) Σ_r log p_win(勝ち馬_r) |
| **主** | race-level win Brier | (1/R) Σ_r Σ_i (p_ri − y_ri)² |
| 副 | NDCG@3・NDCG@5 | rel = clip(6−着順,0,5)、gain 2^rel−1 |
| 副 | ◎top3 | スコア最大の馬が3着以内に入った割合 |
| 副 | ECE | 馬ごとの p_win、等幅10 bin |
| 副 | 穴帯の較正 | p_win 帯 [0,0.02) / [0.02,0.05) / [0.05,0.10) の A/E。**帯は R0-clean の p で固定** |

- 勝率は p_win = softmax(s/τ)。τ はモデル・seed ごとに 2022 年で最尤推定する。
- 評価母集団は 2023 年の出走頭数5以上のレース。win 系の指標では1着同着のレースを除き、件数を報告する。
- 層別: 出走頭数 5-8 / 9-12 / 13-15 / 16-18、芝ダ、競馬場。

---

## 7. MDE（学習前に算出してコミットする）

- **入力**
  - 2023 年の予定レース数と meeting_day 数
  - 既存の R0 代理
    - 主: `oof_scores_c1master` の 2023 年分。train ≤2021 / early stopping 2022 の v6 レシピ OOF で、本実験とほぼ同じ時間構造
    - 対になる代理: 旧 OOF `oof_scores_v6params`
- **方法**: 2つの代理モデルの差について、レースごとの差の meeting-day クラスタ SE を求める。
  MDE = (z_0.975 + z_0.8) × SE。
  感度分析として、モデル間の相関 ρ ∈ {0.90, 0.95, 0.98} の格子版も出す。
- **対象指標**: logloss、Brier、NDCG@3、◎top3。
- **実務閾値（事前固定）**

  | 指標 | Context Gate | Replacement Gate |
  |---|---|---|
  | logloss・Brier | R0 代理の 2023 年水準に対して相対 0.2% | 相対 0.5%（複雑さに見合う幅） |
  | NDCG@3 | 0.002 | — |
  | ◎top3 | 0.5pt（PRED-03A の前例） | — |

- **FAIL の区別**
  - 統計的に非有意: CI95 が 0 を含む
  - 実務的に小さい: |Δ| < 実務閾値
  - 検出力不足: MDE > 実務閾値
- **FAIL 時の正式文**: 「現在の特徴、事前固定したモデル容量、2023 development、および本実験の
  検出力では、事前基準を超えるrace-context増分を検出できなかった」。

---

## 8. Gate

### 8.1 Context Gate（R1 対 R1-noctx）— PASS 条件は全て満たすこと

1. logloss と Brier が両方改善する（Δ < 0）
2. meeting-day bootstrap（3,000回）の CI95 上限 < 0（両方）
3. 5 seed のうち4以上で、logloss と Brier が両方同じ方向
4. |Δ| ≥ min(MDE, 実務閾値)（logloss・Brier それぞれ）
5. 出走頭数の4帯のどこにも致命的な悪化がない（ΔLL の CI95 下限 > 0 の帯が無い）
6. permutation invariance（T1）が学習済みモデルでも PASS

### 8.2 C1 Gate（構成）と C2 Gate（個体間関係）

Context Gate が PASS した場合だけ実行する。placebo は §9 のとおり。

- **C1**: 実際の文脈の効果 G_real（5 seed の per-seed 効果の median）が、P3 の効果分布の
  97.5 パーセンタイルを超える。
- **C2**: G_real が P2 と P4 の両方で、効果分布の 97.5 パーセンタイルを超える。
- **分類**

  | C1 | C2 | 分類 |
  |---|---|---|
  | PASS | PASS | 「構成＋個体間関係」 |
  | PASS | FAIL | 「構成効果のみ／個体間関係なし」（モデル全体の FAIL にはしない） |
  | FAIL | — | 「レース固有の構成効果は未検出（条件一致の別レース文脈で再現可能）」と記録し、R2 へ進むかをユーザーに諮る |

### 8.3 Replacement Gate（R0-clean を置き換える価値）

- Stage 1 では **R1 対 R0-clean を参考判定**する。本判定は Stage 2 で R2 対 R0-clean として行う。
- PASS 条件
  - logloss と Brier の両方が改善し、CI95 上限 < 0
  - 5 seed のうち4以上で方向一致
  - NDCG@3 と ◎top3 の少なくとも一方が悪化しない（悪化方向に有意でない）
  - ECE が悪化しない（悪化方向に有意でない）
  - |Δ| ≥ max(MDE, 相対 0.5%)

### 8.4 進行規則

| 状況 | 次の手 |
|---|---|
| Context Gate FAIL | R2 を作らずに終了 |
| Context Gate PASS、R1 が R0-clean に FAIL | R2 へ進んでよい。**R1 が R0 に負けたことを理由に Context Gate を無効にしない** |
| R2 も R0-clean に FAIL | 終了 |
| R2 が R0-clean に PASS | 2024/2025 OOS の候補 |

Stage 2（R2）への着手は、Stage 1 の報告後にユーザーの承認を得てから行う。

---

## 9. 不変性テストと placebo

### 9.1 P1（placebo ではなく必須の不変性テスト）

同一レース内で馬の行順を並べ替えても、出力は同じ順列で動くだけで、レースの確率と
loss は変わらない。合成データと学習済みモデル（2023 年の実レース）の両方で検査し、
破れたら FAIL。

### 9.2 置き換えるのは文脈経路だけ

自馬の経路は常に本物の x_i を使い、文脈経路の相手集合だけを置き換える。

- **相手馬ベクトルは一まとまりのまま使う**。列ごとの独立 shuffle で合成馬を作ることはしない。
- **レース条件列の上書き**: レース内で一定な列（距離・場所など、base_train で実測して決める）は、
  相手ベクトル側も自レースの値に上書きする。条件の食い違いによる分布外入力で、
  placebo が不当に不利になるのを防ぐため。
- **区分の分離**: 相手は同じ区分（base_train / 2022 / 2023）の中から取る。
  学習用も評価用も同じ規則で置き換える（全体を再学習する方式）。

### 9.3 placebo の種類

| 記号 | 仮説 | 作り方 |
|---|---|---|
| **P3** race-context exchange | C1 | 出走頭数・場・芝ダ・クラス群が一致する別レース r′ の出走馬集合を文脈にする。r′ から1頭をランダムに除いて n−1 頭の相手にする。一致する相手がいなければ、場を外す → 頭数 ±1 → 頭数 ±2 の順に緩める |
| **P2** composition-preserving relational | C2 | P3 の一致条件に加え、能力分布（レース平均 `kako5_avg_pos` の三分位）と逃げ・先行馬の頭数（`prev_pos_rel` ≤ 0.25 の頭数の区分）も一致する別レースの集合を使う |
| **P4** horse-feature coherence break | C2 | 相手を1頭ずつ、同じ区分・同じ芝ダ×クラス群の層で、同じ（先行フラグ × 能力十分位）セルに属する**他レースの実在馬ベクトル**に置き換える。構成は保たれ、特定の相手との関係だけが壊れる |

### 9.4 実行と判定

- 各 placebo で B = 39 回、全体を再学習する（各回1 seed。5 seed を順に回す）。
- placebo の効果 G_b = LL(R1-noctx_s) − LL(R1^{placebo}_b)。
- 判定: G_real（5 seed の median）> q97.5(G_b)。Brier も併せて報告する。

---

## 10. 容量の対照

報告する項目（R1・R1-noctx・R1-noctx-2x・R0-clean）:
- パラメータ数（R0 は木の本数と葉の数）
- 1レースあたりの概算 FLOPs
- 1 epoch の時間
- peak GPU memory
- early stopping で止まった epoch

R1-noctx は同程度の MLP 容量を持つので、R1 との差は文脈経路の構造によるものとみなせる。
R1-noctx-2x で、容量を増やすだけでは同じ改善が出ないことを確認する（診断用）。

---

## 11. 必須テスト（G0、学習の前に全て PASS させる）

| # | テスト |
|---|---|
| T1 | 並べ替えに対する equivariance（=P1） |
| T2 | レース間の情報混入がない（BatchNorm 禁止の静的検査を含む） |
| T3 | padding が結果に影響しない |
| T4 | 取消後に文脈を計算し直して再正規化する |
| T5 | 出走頭数 5〜18 で動き、n > 18 は assert で落とす |
| T6 | 同一レース内の Σp_win = 1 |
| T7 | 未来削除に対する不変性 |
| T8 | 前処理は base_train だけで fit する |
| T9 | 馬 ID・一意識別子が入力に無い |
| T10 | race ID が入力に無い |
| T11 | 当日・結果由来の列が入力に無い |
| T12 | 2024 年以降の行が存在しない |
| T13 | 学習時と評価時で同じ集合規則を使う |
| T14 | カテゴリ表記が正規化後に一致する |
| T15 | 同じ seed で再学習すると同じ結果になる（再現性） |
| T16 | R1-noctx の文脈経路が遮断されている |

具体的な判定条件は `spec.json` の `required_tests`。

---

## 12. Stage 1 の範囲

- **許可**
  - `spec.json` の凍結
  - clean feature contract
  - R0-clean（＋参考値の R0-prodref）
  - R1-noctx（＋容量診断の 2x）
  - R1
  - 合成データによる不変性テスト
  - 2023 年の fixed-model development 評価
  - MDE
  - 5 seed の実行
  - Context Gate PASS の場合に限り、C1/C2 の placebo
- **禁止**
  - R2 以降
  - 2024/2025 性能の開封
  - ROI
  - 馬券の生成
  - 本番の変更
  - 現行 v6 の置換
  - EXP14 との混合
- **中間データ**: `data/_research/mcond/exp15/`（gitignore 対象）
- **小さな成果物**: `analysis/mcond/exp15_race_as_set_dev/out/`（コミット対象）

関連: `RACE_AS_SET_PRIOR_ART_AUDIT.md`、`spec.json`
