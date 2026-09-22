# kako5 Legacy-v6 Contract Shadow Replay — 結果（2026-09-22）

**状態**: READ-ONLY。production master/model・Optuna・EXP再実行・EXP14
Stage1・ROI評価は一切行っていない。**Gate 9のFAIL条件が実際に発動し、
production適用前の最終レビュー用diffは作成していない。**

再現: `analysis/mcond/p0_dnf_history_parity_audit/kako5_contract_shadow_replay.py`
（+ 確証用の追加調査、下記§3）
生データ: `out/kako5_contract_shadow_replay.json`

---

## 0. 用語訂正

前回「legacy patch warranted」と表現したが、**「legacy-compatible shadow
replay warranted」に訂正する**。理由: 特徴不一致は確認済みだったが、
raw score・p_win・順位・◎への実影響、およびデータソース交換がCategory Eへ
波及する可能性が未測定だったため。本ラウンドでその両方を測定した結果、
**Category Eへの波及の主因が、当初想定した「DNFで正しく遡れるように
なったこと」ではなく、候補実装が依拠する`data/_horse_history.parquet`
自体の2026年カバレッジ欠損であることが判明した**（詳細§3）。

---

## 1. 契約の二分（item1）

| | Legacy-v6 contract | Corrected-vNext contract |
|---|---|---|
| 対象 | 直近5件の**完走**レースのみ | 直近5件の**実出走**（完走+DNF） |
| DNF(止) | **除外**（window slotとして数えない） | **1スロットとして数える** |
| 取消・除外 | 除外 | 除外 |
| current v6への入力 | **可能（これのみ）** | **絶対に入力しない** |
| 本ラウンドでの扱い | 候補実装を検証(§2-3) | 未実装・未評価 |

---

## 2. `_horse_history.parquet`候補実装のLegacy-v6契約再現性検証（item2）

実装(`build_legacy_v6_kako5()`):
- 完走条件filter(`pos.notna()`)してから直近5件を取る（単に直近5行を取らない）
- `date < race_date`の厳密未満で同日内リークを禁止
- horse ID(`ped_id`)を主キーとし、`_HistoryIndex.resolve()`（
  serve_history_feats.pyの既存・検証済みロジックをそのまま再利用、
  馬名+種牡馬+生年での曖昧性解消込み）で解決済みのIDのみを使う。
  自前の馬名joinは行っていない。
- deletion invariance: 対象日以降の履歴行を削除しても対象日以前の特徴が
  不変であることをサンプル500頭で確認（`PASS`、実装は未来日を最初から
  参照しない設計のため当然の結果ではあるが、実測でも確認した）。

**この検証だけでは「Legacy-v6契約を正確に再現している」と結論できない
ことが、§3の実データ比較で判明した。**

---

## 3. 48日分の実データshadow replay（item3-5）

48日分の保存済みweekly CSV + kako5 CSVを対象に、`predict_weekly.parse_csv()`
→ `serve_history_feats.fill_history_features()`という**実際のserve pipeline
そのもの**でdf(120特徴)を構築し、current(`build_from_kako5()`由来のkako5)と
legacy_shadow(上記候補実装)を比較した。結果・払戻・ROI列は一切読んでいない。

### Hard Gate結果（item7）

| Gate | 結果 |
|---|---|
| 120特徴の列名・順序・dtype不変 | PASS |
| **kako5以外の107特徴が完全一致** | **PASS**（48日全てで完全一致、差分ゼロ） |
| race/horse行数不変 | PASS |
| モデルhash不変 | PASS（`0a040c4e...`） |
| encoder hash不変 | PASS（`06b20309...`） |
| raceごとのp_win合計≈1 | PASS |
| 未来レース削除で過去特徴不変 | PASS（サンプル500頭） |
| current production artifactを上書きしない | PASS（読み取り専用） |

### 13特徴の比較（item4）— 1,520レース、48日、比較対象行数は特徴により19,543-21,214

| 特徴 | 分類(旧) | 比較数 | 不一致数 | 不一致率 | 差分mean | 差分max |
|---|---|---:|---:|---:|---:|---:|
| kako5_race_count | B | 21,214 | 878 | 4.14% | 1.36 | 5.0 |
| kako5_same_td_ratio | B | 19,543 | 748 | 3.83% | 0.30 | 1.0 |
| kako5_same_dist_ratio | B | 19,543 | 950 | 4.86% | 0.25 | 1.0 |
| kako5_same_place_ratio | B | 19,543 | 770 | 3.94% | 0.19 | 0.8 |
| kako5_avg_pos | E | 19,543 | 1,886 | 9.65% | 1.73 | 14.0 |
| kako5_std_pos | E | 19,543 | 1,884 | 9.64% | 1.00 | 8.0 |
| kako5_best_pos | E | 19,543 | 1,109 | 5.67% | 3.79 | 16.0 |
| kako5_pos_trend | E | 19,543 | 1,888 | 9.66% | 1.40 | 16.0 |
| kako5_expected_good_count | E | 19,543 | 8,229 | **42.11%** | 1.31 | 5.0 |
| kako5_hidden_good_count | E | 19,543 | 4,036 | 20.65% | 1.46 | 5.0 |
| kako5_same_cond_best_pos | E | 16,578 | 591 | 3.56% | 4.07 | 14.0 |
| kako5_avg_ninki/pos_vs_ninki/avg_agari3f/best_agari3f/upset_good_count | E/NA | 0 | — | — | 実データでは両側とも常時NaN(上り3F/人気列が母集団上ほぼ未使用) | — |

**Category Eの不一致率は当初の懸念通り無視できない規模だった
（kako5_expected_good_countは42.11%）。ただし、この不一致の"原因"を
確定するまでは「DNF architectureの正しい波及効果」と決めつけない
（次節参照）。**

### モデル出力への影響（item5）

| 指標 | 値 |
|---|---|
| raw score絶対差: mean/p95/p99/max | 0.107 / 0.319 / 0.597 / 1.461 |
| p_win絶対差: mean/p95/p99/max | 0.0057 / 0.0219 / 0.0404 / 0.1399 |
| 馬順位変更レース数 | 1,264 / 1,520 (83.2%) |
| top3順位集合変更レース数 | 332 / 1,520 (21.8%) |
| ◎変更レース数 | 169 / 1,520 (11.1%) |
| ◎〇▲(mark set)変更レース数 | 372 / 1,520 (24.5%) |
| 最大変化race | 2026091306040412 (20260913)、score差1.461 |

**これらの数値は下記§3.1の理由により、そのまま「Category Bの真の実影響」
として採用できない。**

---

## 3.1 決定的な確証: `_horse_history.parquet`の2026年カバレッジ欠損を発見

`diff_reason_counts`: `{"other": 577, "dnf_shortens_history_window": 293,
"scratch_or_dnf_mixed": 8}`。kako5_race_count不一致878件のうち**577件
(65.7%)が、直近5走windowにDNF/取消/除外を一切含まない「other」理由**
だった——これはDNFアーキテクチャの問題では説明できない。

個別に5件を実データで確認し、追加で15日分をサンプル抽出して方向性を
確認した結果:

> **`kako5_race_count`が不一致な「other」理由579件中、サンプル179件全て
> (100%)で current(TARGET由来kako5 CSV) > legacy_shadow
> (`_horse_history.parquet`由来)——legacy_shadowが常に過小カウントしていた。**

具体例（サーガスターレ、ped_id=-3832211738）: `_horse_history.parquet`には
2026-08-02の1走しか記録されていないが、TARGETの週次kako5 CSVは同馬について
直近2走を認識している。`data/kekka/20260802.csv`・`data/weekly/20260802.csv`
はいずれも実在するにもかかわらず、`_horse_history.parquet`（2026-09-11
ビルド）には反映されていない。

**結論**: `build_horse_history.py`の2026年取り込みパイプライン
(`load_2026_history()`/`resolve_2026_ped_ids()`)に、TARGETの週次kako5 CSVが
持つ完全性に対して**系統的な取りこぼしが存在する**（根本原因はkekka×weekly
のINNER JOIN条件・馬名解決ロジックのいずれかに起因すると推定されるが、
本ラウンドでは完全な特定に至っていない）。

**これにより、§2で「実装可能」と確認したLegacy-v6契約の候補実装
(`_horse_history.parquet`ベース)は、実データでは正確な再現に失敗している
と判定する。** Category Eの不一致(9.65-42.11%)は、DNFアーキテクチャの
差ではなく、この未解決のデータ欠損が主因である可能性が高い
(577/878=65.7%が「other」理由、DNF/scratch起因は301/878=34.3%のみ)。

---

## 4. Category B/E分離の結論（item6）

**当初の想定通り「4特徴だけのlegacy patch」とは呼ばない。** ただし、
本ラウンドで判明したのは「13特徴serve-contract migrationとして扱うべき」
という当初の想定を**さらに超える**事実——候補実装自体がLegacy-v6契約を
正確に再現できておらず、Category Eへの波及の大部分がDNFとは無関係な
データ欠損に起因する。したがって:

- Category B（4特徴）の実在性は、`_horse_history.parquet`を経由しない
  独立した測定（前ラウンドの`category_b_real_production_impact.py`、
  TARGET kako5 CSV自身の5スロット内での比較）により**確定的に支持される
  ままである**（424/20,833行、kako5_race_count 100%不一致）。この結論は
  §3.1の欠損発見によって覆らない。
- Category E（9特徴）への波及は、**本ラウンドの実装では説明できない**
  （65.7%がデータ欠損起因、34.3%のみDNF/scratch起因と推定されるが正確な
  内訳は未確定）。

---

## 5. 判定（item9）

- **Legacy-v6 contractの候補実装(`_horse_history.parquet`ベース)は、
  実データ検証でFAILと判定する。** データ欠損が未解決のまま同ソースへの
  移行を進めることはできない。
- **Category Eへの波及は説明できない → FAILの2つ目の条件にも該当する。**
  (65.7%がデータ欠損起因と判明したが、残り34.3%のDNF/scratch起因分だけを
  Category Eの「真の」DNFアーキテクチャ影響として分離measurementする追加
  検証は本ラウンドでは実施していない)
- **107特徴の完全一致Gateは唯一PASSした**（kako5以外は無傷、この部分の
  設計は健全）。
- **したがって、production適用前の最終レビュー用diffは作成しない。**
  §3で測定したmodel output impact(raw score/p_win/順位/◎変更)の数値は、
  データ欠損混入により「Category Bの真の影響」として採用しない
  （参考値として記録するに留める）。

### 次のステップ（本ラウンドでは未着手、production変更ではない）

1. `build_horse_history.py`の2026取り込みパイプラインのデータ欠損を
   別途調査・修正する（`load_2026_history()`のINNER JOIN条件、
   `resolve_2026_ped_ids()`の馬名解決ロジックを疑う）。
2. 欠損修正後、本ラウンドと同じ48日分replayを再実行し、
   Category Bの真の実影響とCategory Eの真の波及(あるとすれば)を
   分離して再測定する。
3. それでもLegacy-v6契約が正確に再現できる場合のみ、legacy-compatible
   shadow patchの実装検討へ進む。

いずれもユーザーの明示指示なしには着手しない。

---

## 6. 停止事項

本ラウンドでも: production master/model変更・Optuna・EXP再実行・
EXP14 Stage1・ROI評価・corrected-vNext実装は一切行っていない。

関連: `docs/research/DNF_INCLUSIVE_BASELINE_RECONCILED_20260922.md`,
`analysis/mcond/p0_dnf_history_parity_audit/SEMANTIC_VS_PARITY_CLASSIFICATION.md`,
`analysis/mcond/p0_dnf_history_parity_audit/category_b_real_production_impact.py`,
`analysis/mcond/p0_dnf_history_parity_audit/kako5_contract_shadow_replay.py`
