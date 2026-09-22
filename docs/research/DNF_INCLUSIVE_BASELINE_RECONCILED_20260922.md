# DNF-inclusive Baseline — 算術整合版（2026-09-22）

**状態**: READ-ONLY再監査。前回報告値の算術矛盾をユーザーが指摘し、年別完全
分離+加重平均恒等式のhard gate検証により再構築した。**production master/
model変更・Optuna・EXP再実行・EXP14 Stage1・ROI評価は一切行っていない。**

再現スクリプト: `analysis/mcond/p0_dnf_history_parity_audit/reconciled_dnf_baseline.py`
テスト: `analysis/mcond/p0_dnf_history_parity_audit/test_dnf_baseline_identity.py`
生データ: `out/dnf_baseline_reconciled.json`

---

## 0. 前回値の無効化

前回報告した以下の値は算術的に矛盾しており`provisional_invalid_aggregation`
として無効化した:

- 全体差 約0.2pt（2023/2024/2025年別の値）
- DNFありレースの3.66pt差（75.01%→71.35%）
- DNFなしレース74.31%
- これらに基づく解釈

**原因**: 年別overall（2023/2024/2025個別集計）とDNFあり/なしレース別
（2013-2025**全期間ブレンド**集計）を同一文脈で比較し、異なる母集団（年範囲）
の値を無断で並べていた。全期間ブレンドにはtrain期間（2013-2022、モデルが
学習時に直接見た行）が混入しており、その分◎top3率が高く出る
（train期間平均76-81% vs OOS期間60-63%）。回帰テスト
`test_original_bug_reproduction_blended_vs_year_specific`で、この不整合量
（0.1364、許容誤差1e-12を大幅に超過）を再現・固定化した。

---

## 1. Hard Gate（加重誤差1e-12）

`overall = dnf_share×dnf_race_val + (1-dnf_share)×non_dnf_race_val`
を年別・Effect A/B/C別に全て検証し、**全て許容誤差1e-12以内でPASS**
（実測誤差は0または浮動小数点誤差1.11e-16のみ）。違反時は
`AssertionError`でレポート生成自体を停止するhard gateとして実装済み
（`verify_identity_hard_gate()`）。

---

## 2. Score Provenance

| 項目 | 値 |
|---|---|
| model artifact (unified_rank_v6.pkl) sha256 | `0a040c4ec74fd7df00c05ef1440048def21ba940dc1ae67df00487b932701e8f` |
| shadow retrained model sha256 | `a1870380a3d410b512f1a70bacd6d14bae9ddd5569fb13a7fee5084a90c9ed01` |
| training master (master_v2) sha256 | `b8032d1bd78176a939b615921fa4fd25faadbf493f685e44a06e7f2e9e780a76` |
| feature artifact | 本監査で`model.predict()`により毎回再計算（master_v2保存済み値ではない、現行モデルは無変更） |
| ◎選択列・tie-break | race内`_score`最大(argmax)、tie-breakなし(本データで完全数値タイは未観測) |
| DNF馬スコアの特徴契約 | Effect A=training契約(post-dropna母集団)、Effect B/C=DNF_SEMANTIC_SPEC.md準拠corrected契約——finisherとDNFで異なる契約を混在させていない(各Effect内で統一) |
| score不能馬の扱い | raw CSV(lgbm/cat/torch/add)のJOINキーに実在しないDNF行のみを「score不能」と定義し対象レースから除外(今回の全期間再構築では0/2946件、EXP13 2023限定の旧集計で見られた12/145はJOIN範囲の違いによるものであり今回は無関係)。個々の特徴のNaNは本番同様fillna(-9999)で扱い、行を落とす理由にしていない |

**年区分（leak-safety）**:

| 年 | split | 意味 |
|---|---|---|
| 2013-2022 | train | **in-sample。モデルが学習時に直接見た行。遡及適用は診断値であり、OOS性能とは呼ばない** |
| 2023 | valid | selection/development |
| 2024-2025 | test | OOS(正式な性能指標として扱える唯一の期間、2023と合わせて) |

---

## 3. 三効果の分離（Effect A / B / C）

- **Effect A（denominator-only）**: finisherは現行の既存スコア(`_score_old`、
  変更なし)のまま、DNF馬だけを**現行training契約(post-dropna母集団、
  DNF_SEMANTIC_SPEC.mdの新定義ではない)**で構築した特徴でスコアしsoftmax
  分母へ追加。
- **Effect B（feature-semantic correction）**: finisher・DNFとも
  DNF_SEMANTIC_SPEC.md準拠のcorrected特徴で、**現行v6モデル(無変更)**により
  再スコア。
- **Effect C（retrained model）**: finisher・DNFともcorrected特徴で、
  **同一ハイパーパラメータでshadow再学習したモデル**により再スコア。

---

## 4. Leak-safe期間（2023 valid・2024-2025 test）の年別完全分離表

| 年(split) | 全レース数 | DNFありレース数 | DNFなしレース数 | DNF構成比 | Effect | old◎top3 | new◎top3 | 加重平均再構成 | 直接集計との差 |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 2023(valid) | 3,456 | 191 | 3,265 | 5.53% | A | 60.71% | 60.62% | 60.62% | 0.00e+00 |
| 2023(valid) | 3,456 | 191 | 3,265 | 5.53% | B | 60.62% | 60.45% | 60.45% | 0.00e+00 |
| 2023(valid) | 3,456 | 191 | 3,265 | 5.53% | C | 59.26% | 59.11% | 59.11% | 0.00e+00 |
| 2024(test) | 3,454 | 221 | 3,233 | 6.40% | A | 62.80% | 62.57% | 62.57% | 0.00e+00 |
| 2024(test) | 3,454 | 221 | 3,233 | 6.40% | B | 62.74% | 62.51% | 62.51% | 0.00e+00 |
| 2024(test) | 3,454 | 221 | 3,233 | 6.40% | C | 62.57% | 62.25% | 62.25% | 0.00e+00 |
| 2025(test) | 3,455 | 195 | 3,260 | 5.64% | A | 61.33% | 61.13% | 61.13% | 1.11e-16 |
| 2025(test) | 3,455 | 195 | 3,260 | 5.64% | B | 61.33% | 61.13% | 61.13% | 1.11e-16 |
| 2025(test) | 3,455 | 195 | 3,260 | 5.64% | C | 60.41% | 60.12% | 60.12% | 1.11e-16 |

（train年2013-2022は診断値、`out/dnf_baseline_reconciled.json`の`by_year`
参照。dnf_share=5.0-6.5%・old_overall=76.5-81.5%と、OOS期間より高い
in-sample性能を示す——これはモデルが訓練時にこれらの行を直接見ているためで
あり、OOS性能とは解釈しない。）

### 効果の分解（2023valid・2024test・2025testの平均的傾向）

- **denominator-onlyだけで ◎top3は約0.09-0.23pt低下**（Effect Aのold→new）。
  現行モデル・現行スコアは変更せず、DNF馬をsoftmax分母へ正しく含めるだけで
  この規模の低下が生じる——これは「モデルが間違っている」のではなく
  「評価方法が甘かった」ことを意味する。
- **feature-semantic correctionを追加すると、さらに約0.06-0.17pt低下**
  （Effect A_new → Effect B_old、finisher側の6,152行=0.98%の特徴修正効果）。
- **retrained model(同一HP)にすると、さらに約1.1-1.5pt低下**（Effect
  B_new → Effect C_old相当、これは前回報告済みのshadow retrain結果と整合）。

**「◎top3約62%」は、denominator-only補正だけを見れば、2024年62.80%→62.57%
（-0.23pt）・2025年61.33%→61.13%（-0.20pt）とごくわずかな低下に留まる。
feature補正を含めても2024年62.51%・2025年61.13%と、依然として「約62%」の
範囲内にある。retrained modelまで適用すると2024年62.25%・2025年60.12%と、
より明確な低下が見える(ただしこれは同一HPでの比較であり、Optuna再探索を
経ていない暫定値)。**

---

## 6. Category B の実serve影響（結果ラベルなし、item8）

`analysis/mcond/p0_dnf_history_parity_audit/category_b_real_production_impact.py`
で、実際の2026年serve入力（`data/kako5/2026*.csv`、48ファイル、sha256を
`out/category_b_real_production_impact.json`に記録）を対象に、
`parse_kako5.build_from_kako5()`が実際に計算した値(current serve値)と、
同じ実データにtraining契約(post-dropna、DNFをwindow slotとして数える)を
適用した場合の値を比較した。**結果・ROIは一切見ていない**。

| 対象 | 値 |
|---|---|
| スキャン対象ファイル数 | 48 |
| 総kako5行数 | 20,833 |
| 直近5走windowに止/外/消を含む実データ行 | **424（2.0%）** |

| 特徴 | 比較数 | 不一致数 | 不一致率 | 平均絶対差 | 最大絶対差 |
|---|---:|---:|---:|---:|---:|
| kako5_race_count | 424 | 424 | **100%** | 1.01 | 2.0 |
| kako5_same_td_ratio | 414 | 121 | 29.2% | 0.041 | 0.667 |
| kako5_same_dist_ratio | 414 | 172 | 41.6% | 0.053 | 0.5 |
| kako5_same_place_ratio | 414 | 225 | 54.4% | 0.068 | 0.5 |

**Category Bは実データで確定的に確認された**: 止/外/消が直近5走windowに
含まれる実際の2026年出走馬(全体の2.0%)について、`kako5_race_count`は
**例外なく100%不一致**（現行serveは常にtraining契約より少なく数える）、
同条件比率3種も29-54%の不一致率で実測された。raw score/p_win/◎変更件数の
定量化は、実際の週次serving pipeline（`predict_weekly.py`/
`export_weekly_marks.py`）をこれら48日程分再実行する必要があり
（2026年のraw学習用CSVは存在しないため、本監査の既存の特徴再構築基盤
では代替できない）、本ラウンドでは実施していない——必要であれば別途
明確なスコープとして切り出す。

## 7. Legacy-compatible patch の要否（item9）

Category Bが実データで確認されたため、shadowでの修正候補を試作した
(`analysis/mcond/p0_dnf_history_parity_audit/shadow_kako5_from_horse_history.py`、
**本番へは一切適用していない**)。

**アーキテクチャ上の発見**: 現行`build_from_kako5()`はTARGET週次kako5 CSV
(固定5列横持ち)を読むため、直近5枠にDNF/取消/除外が挟まっていても、
それより前の実走で埋め直す手段がコードレベルで存在しない
(データソース自体が5列を超える履歴を保持しない)。したがって
**真の意味でのtraining契約parityは、`build_from_kako5()`自体へのコード
修正だけでは達成できない**——course/jockey系が既に使っている
`data/_horse_history.parquet`(全キャリア深度、659,037行、70,724頭)へ
データソースを切り替えることで初めて達成可能である。

プロトタイプ`build_kako5_from_horse_history_training_contract()`を
2026年のDNF歴を持つ馬224頭中、後続レースがある104頭の実例に適用し、
before/afterバンドルを`out/shadow_kako5_legacy_patch_prototype.json`に
保存した(DNFの新しい意味定義=DNF_SEMANTIC_SPEC.mdは混入させていない、
あくまで現行training契約への整合のみ)。

**推奨**: legacy-compatible修正は「`build_from_kako5()`のコードだけを
直す」のではなく、「kako5計算のデータソースを`data/_horse_history.parquet`
へ切り替える」アーキテクチャ変更として計画すべきである。本監査は
プロトタイプの実現可能性を確認したのみで、本番適用の判断・実装は
行っていない。

## 8. corrected-vNext の要否

`MIGRATION_PATHS_AND_VNEXT_DECISION.md`の5条件のうち、本ラウンドで新たに
充足したのは「DNF-inclusive baselineの完成」（本ファイル§1-4）のみ。
「実際のtrain/serve skewの定量化」はCategory Bについてのみ完了（§6）、
course/jockey(A)・kako5残り9特徴(E)は未定量化のまま。「予測悪化が一時的
問題か」の検証も未実施。**5条件は依然として揃っておらず、corrected-vNextへ
今すぐ移行する合理性は引き続き不十分**。

## 9. 結論

- 加重平均恒等式は全年・全Effectでhard gate(1e-12)をPASSした。前回の
  「-0.2pt」「3.66pt」等の値は異なる母集団（年範囲）を混同した誤りであり、
  本ファイルの年別完全分離表に置き換える。
- denominator-onlyの真の効果は前回主張した「全体では希釈されて小さい」より
  もやや明確（valid/test各年で-0.09〜-0.23pt）だが、依然として小さい。
  「◎top3約62%」という既存の理解を覆すほどの規模ではない。
- feature-semantic correction・retrained modelの追加効果も含め、3つの効果を
  混同せず提示した。

関連: `docs/research/DNF_HISTORY_FEATURE_PARITY_AUDIT_20260922.md`,
`analysis/mcond/p0_dnf_history_parity_audit/reconciled_dnf_baseline.py`,
`analysis/mcond/p0_dnf_history_parity_audit/test_dnf_baseline_identity.py`,
`analysis/mcond/p0_dnf_history_parity_audit/category_b_real_production_impact.py`,
`analysis/mcond/p0_dnf_history_parity_audit/shadow_kako5_from_horse_history.py`
