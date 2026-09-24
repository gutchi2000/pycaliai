# EXP17 Stage 0 — 検出力監査・機構 floor の固定・placebo 設計

**作成日**: 2026-09-25　**コード**: `power_audit.py`　**結果**: `out/power_audit.json`（機械可読）
**規律**: 2019〜2023 の着順は使わない。実 race 構造（未対戦かつ共通対戦馬あり pair の配置、開催日、EXP02 as-of 能力）に合成結果を注入する。

---

## 1. 推論単位

- **開催日（暦日）** を推論単位とする。2019〜2023 で 540 日（年 106〜110 日）。同一レース内の pair、同一日の複数レース・複数場を独立標本として数えない。
- EXP16A は `rid16[:10]`（場×日、年 288）を cluster にした。EXP17 は同日の複数場を 1 cluster にまとめる**より保守的**な定義を採る（天候・馬場の同日相関を吸収）。spec v0.2 に明記。

## 2. 生成モデル（label-free）

- `perf_h = b·μ_h + η_h + Gumbel`、η_h ~ N(0, σ_η²)、σ_η = 0.5 × レース内 SD(b·μ) = 0.222。b = 0.1033（等価性監査で 2018 年以前の対戦から fit）。
- pair label `y_ij = 1[perf_i > perf_j]`（1 つの完走順から全 pair を作るので、pair 間の従属は実データと同じ構造）。
- DYNPL arm: `logit = a·Δμ`（scalar rating が知り得る全て。a は較正 rep で fit・凍結）。
- PATH arm: `logit = a'·Δμ + β·ẑ`、`ẑ = Δη + ε`、ε ~ N(0, s²)。s を二分探索して E[pairwise logloss 改善] を目標 Δ に合わせる。
- 注意: これは「scalar rating が持たない race-day 情報を pair 表現が部分観測する」**最良ケースの代理**であり、真の非推移 pair 効果ではない。検出力の上界側の見積り。

母集団: 14,391R（正式 15,951R のうち主 sample pair を 1 つ以上持つレース）、697,784 pair、平均 48.5 pair/R。

## 3. 判定規則（G1 の統計条件）と結果

detect = CI95 上限 < 0（開催日 cluster、正規近似）∧ 5 年中 4 年で改善 ∧ leave-one-year-out 5 通り全てで CI95 上限 < 0。
placebo 条件は合成できないため「満たされたもの」として扱う（実 power はこれより低い）。400 rep、seed 20260925。

| 真の効果（nats/pair） | 較正ノイズ s | detect power | Wilson95 | CI のみ | detect ∧ 点推定 ≤ −floor | 実現効果 平均 / rep 間 SD |
|---|---|---|---|---|---|---|
| 0.0005 | 1.371 | **1.000** | [0.990, 1.000] | 1.000 | 0.000 | 0.00053 / 0.00007 |
| **0.001（floor）** | 0.951 | **1.000** | [0.990, 1.000] | 1.000 | **0.638** | 0.00104 / 0.00010 |
| 0.002 | 0.640 | 1.000 | [0.990, 1.000] | 1.000 | 1.000 | 0.00203 / 0.00013 |
| 0.005 | 0.329 | 1.000 | [0.990, 1.000] | 1.000 | 1.000 | 0.00507 / 0.00021 |
| 0.010 | 0.078 | 1.000 | [0.990, 1.000] | 1.000 | 1.000 | 0.01002 / 0.00036 |
| 0.020 | — | 生成モデルの上限（完全な η 知識でも 0.0106）を超えるため未測定 | | | | |

- **MDE（80% detect）は 0.0005 未満**（最小の試験水準で既に 100%）。開催日平均の SE ≈ 1.0e-4 nats/pair。
- cluster 正規近似の検証（1 rep、floor 水準）: 正規 SE 1.00e-4 vs bootstrap SE 1.01e-4、CI95 上限 −0.000708 vs −0.000708。一致。
- **結論: G1 の統計条件は floor 0.001 で power 1.0。検出力不足は停止理由にならない。** EXP17 が Stage 0 で止まる理由は検出力ではなく E0（等価性）と被覆床である。

## 4. 機構 floor（Stage 0 で固定、結果開封後に変更しない）

| 項目 | 値 |
|---|---|
| **G1 機構 floor** | **0.001 nats/pair**（PATH − DYNPL の pairwise logloss、race 内平均 → 開催日平均） |
| 根拠 | (a) MDE < 0.0005 で検出可能。(b) EXP02 が ELO+Glicko に対して示した固有上積み（馬行 top3 二値 logloss −0.0006）と同桁で、それを下回る効果は既存 rating 系の再探索と区別できない。(c) 生成モデルの上限 0.0106 の約 1/10 |
| 等級 | EXP16A と同型に分離: **検出**（CI95 上限 < 0 ∧ 年 ∧ LOO ∧ placebo）と **floor 到達**（CI95 上限 < −0.001）。true = floor で点推定 ≤ −floor は 64% にしかならないので、点推定は等級条件に使わない |
| race-level 実務床（G2） | **0.005 nats/race**（EXP16A から継承、変更なし） |

## 5. Placebo 設計（Stage 0 で固定、結果開封前）

| 名称 | 操作 | 保存する構造量 | 壊すもの | 置換セル | draw / seed / 判定 |
|---|---|---|---|---|---|
| **P1 degree-preserving identity rewire**（主） | 対象馬 h の履歴辺 (h,c,date,won,surf,dband) について、c の identity だけをセル内の相手プールから degree(h) 個復元抽出で置換。辺属性は保持 | 各馬の degree、辺の年齢分布、W/L 回数、条件一致率、レース頭数、年齢帯・出走数帯の構成 | 相手の identity → 共通対戦馬の集合（n_common は変わる。M3_BASE に n_common・被覆を control として入れ、draw ごとに n_common 分布を記録） | 年 × 芝ダ × 頭数帯 × 年齢帯 × 出走数帯（対象馬側） | 200 draw、seed 20260925+draw、real 改善 > placebo 分布の 97.5 percentile |
| **P2 direction null** | 各 (h,c) 辺の won を確率 0.5 で反転。endpoint・n_hc・被覆は不変 | 全構造量（identity 含む） | 方向情報のみ | セル不要 | 200 draw、同 seed 規約、97.5 percentile |
| **P3 identity-free structural control** | d_ij を (degree 差, 出走数差, 休養日差, 直近性, 成分サイズ) の Y−1 以前 fit 関数で置換 | — | identity と方向の両方（構造だけの arm） | — | 単一 arm（draw なし）。real は P3 を上回ること |

判定: real 改善は **P1・P2 の 97.5 percentile を共に超え、かつ P3 を超える**こと。超えなければ「共通対戦馬 identity ではなく graph 構造・経験量・方向ノイズで説明可能」と判定する。

**EXP12 placebo との比較**

| 観点 | EXP12（O3） | EXP17 P1 |
|---|---|---|
| 置換対象 | 馬行に集約した相手能力 3 特徴の値 | 辺リスト上の相手 identity そのもの |
| 近似 | CLT 近似（層内プールの平均・SD からサンプリング平均を合成）、mean 系 3 特徴のみ | 近似なし（真の identity 再抽選、全 pair 表現を再計算） |
| 保持 | degree、層（年齢帯 × 出走数帯）、欠損率 | 上記 + 辺年齢・W/L・条件一致・頭数 |
| 壊す | 相手 identity | 相手 identity（→ 共通対戦馬構造） |
| 保持しなかったもの | 順序統計（max/top3/dispersion） | 共通対戦馬数（意図的に壊れる; control で吸収） |
| draw | 1,000 | 200（1 draw ≈ 6.8 分、16 並列で約 1.5 時間） |
| 結果 | real +0.00037 < 97.5%ile +0.00057 → FAIL | 未実施（E0 FAIL のため Stage 1 に進まない） |

## 6. 位置づけ

検出力監査は「進めた場合に検出できたか」を記録するために実施した。**E0 FAIL（`PRIOR_ART_EQUIVALENCE_AUDIT.md`）と被覆床 FAIL（`GRAPH_DATA_AUDIT.md` §5）により、G1/G2 の実評価には進まない。** 本監査の floor・placebo 設計は、将来「pair 固有状態を持つ新しいデータ源」で別番号の実験を設計する際の再利用資産である。
