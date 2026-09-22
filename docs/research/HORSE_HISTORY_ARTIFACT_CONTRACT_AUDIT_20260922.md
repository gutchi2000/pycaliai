# `data/_horse_history.parquet` Artifact Contract Audit（2026-09-22）

**状態**: READ-ONLY監査。結果・ROIは一切使用していない。production master/
model変更・scheduler変更・Optuna・EXP再実行・EXP14 Stage1・ROI評価は
一切行っていない。

再現: `analysis/mcond/p0_dnf_history_parity_audit/horse_history_contract_audit.py`
生データ: `out/horse_history_contract_audit.json`

---

> ## ⚠ ERRATUM（2026-09-22、`ALT_SOURCE_RECOVERY_AUDIT_20260922.md` により更新）
>
> 本書 §5 は欠落原因を「上流 `data/weekly/{date}.csv` のレース単位欠落」と
> 記述したが、その**正体を特定できていなかった**。後続監査で確定した:
>
> - 欠落 50 レースのうちクラス照合できた**丸ごと欠落 16 件は 16/16 が障害競走**
>   （距離 2750〜3390m）。`data/weekly` は障害競走を含まない。
> - これは CLAUDE.md「除外レース: 障害・新馬中心」と整合する
>   **意図的な scope filter** であり、偶発的な収集失敗ではない。
>   本書 §5 の「上流の週次データ収集・保存プロセスの欠落」という表現は、
>   意図性を取り違えている点で不正確だった。
> - ただし学習側 `master_v2` は障害競走を**含む**（1,538 レース / 3.42%）ため、
>   **train/serve 間の母集団非対称は実在する**。よって本書の固定結論
>   （`data/weekly` は 2026 履歴の authoritative source として不完全 /
>   `_horse_history.parquet` は全出走履歴を必要とする consumer には不完全）は
>   **そのまま維持される**。
> - 本書 §5 が「不明」とした馬名一致 filter での 1 行消失は、
>   `data/kekka` の**馬名 9 文字切り詰め**が原因と特定済み。
>
> 最新の根拠は `docs/research/ALT_SOURCE_RECOVERY_AUDIT_20260922.md` を参照。

---

## 1. Artifact Contract（`build_horse_history.py`のdocstring+コード読解で確定）

| 項目 | 内容 |
|---|---|
| 生成元スクリプト | `build_horse_history.py` |
| 呼び出し元 | `serve_history_feats.py`（`fill_history_features()`/`_HistoryIndex`/`rolling_rate()`）。実際にこれを呼ぶのは`export_weekly_marks.py`（v6本番経路）のみ。`predict_weekly.py`（旧アンサンブル経路）は使用していない（コード確認済み） |
| 更新方法 | **全再構築（incrementalではない）**。`main()`は既存parquetを読まず、毎回`master_v2`（固定、2013-2025）+`data/kekka/2026*.csv`全件+`data/weekly/2026*.csv`全件から最初から作り直す |
| 更新トリガー | docstring記載: 週次運用でPhase C（`weekly_post.ps1`）にてkekka配置後に再実行、2026分を追随させる想定 |
| 想定開始日・終了日 | 2013-01-05（master_v2開始）〜実行時点で`data/kekka/`に存在する最新2026日付まで。終了日の上限はない |
| 1頭あたり保持件数 | **全キャリア**（docstring6行目: 「馬ごと全キャリア履歴parquetを構築する」）。コード中に`.tail(N)`等の切り詰め処理は存在しない（`load_master_history`/`load_2026_history`/`resolve_2026_ped_ids`を全読し確認済み） |
| 完走/DNF/取消/除外の包含規則（2013-2025） | 止/外/消は`master_v2`自体に存在しない（`build_dataset.py`のdropnaで構造的に不在）ため、この期間はpos=NaN行が原理的に発生しない |
| 完走/DNF/取消/除外の包含規則（2026） | kekka『確定着順』の0/空値をpos=NaNとして残す。**止・外・消を区別せず一律NaN**（データソース自体がこの3種を区別しない、`LIVE_SERVE_DNF_TRACE.md`で既確認） |
| horse ID解決規則（2013-2025） | `master_v2`の血統登録番号を直接使用（権威あるID） |
| horse ID解決規則（2026） | `resolve_2026_ped_ids()`: 馬名完全一致でmaster由来のエンティティ表と照合、（種牡馬一致 or 生年±1一致）で1件に絞れればそのped_idを採用。0件/曖昧なら`synthetic_id(name, sire)`（決定論的hash）を新規発行。**2026どうしの相互照合はしない**（比較対象はmasterのみ） |
| production実使用箇所 | `course_n_prev`/`course_win_rate`/`course_top3_rate`/`jockey_n_prev`/`jockey_win_rate`/`jockey_top3_rate`/`hist_same_cond_best_pos`/`top3_rate`/`count`/`hist_same_place_best_pos`/`horse_fuku10`/`30`/`jockey_fuku30`/`90`/`trainer_fuku30`/`90`（`serve_history_feats.NUM_FEATS`、計16特徴）。**kako5_\*13特徴はこのparquetを使わない**（`parse_kako5.build_from_kako5()`が別のTARGET kako5 CSVを直接読む） |
| 「全履歴を保持する」明示契約の有無 | **存在する**。docstring 6行目が明示的契約 |

---

## 2. bug / 用途外使用 / 契約未定義 の判定

**「artifact completeness bug」と判定する**（明示契約=全キャリアより実際の
保持が少ないため）。ただし**`build_horse_history.py`自体のコードバグではない**
——後述の通り、同スクリプトのロジックは入力データに対して100%忠実に動作
していることを実測で確認した。根本原因は入力データ（`data/weekly/{date}.csv`）
自体の欠落であり、コード修正では解決しない構造的な問題である。

---

## 3. Authoritative Universe（race_id+馬番+date、馬名joinなし）

`data/kekka/2026*.csv`（全出走馬+確定着順）の行存在を一次証拠として構築。
2026年分、`data/weekly/`も存在する72日を対象（`build_horse_history.py`自身の
対象範囲と同一）。

- 対象日数: 72
- kekka合計行数: 32,861
- 非完走(pos<=0/空値、DNF・取消・除外区別不能): 別集計済み（`out/`内`n_nonfinish_kekka`列）

---

## 4. `build_horse_history.py`の3段階funnelの実データ再現 + parquetとの完全性比較

`build_horse_history.py`自身の実関数（`parse_weekly_light`等）をそのまま
importし、複製ではなく実コードで3段階を追跡した:

| 段階 | 行数 |
|---|---:|
| stage0: kekka読込(全出走馬) | 32,861 |
| stage1: weekly inner join後 | 32,264（**-597**） |
| stage2: 馬名完全一致filter後 | 32,263（**-1**） |
| `_horse_history.parquet`実物(2026分) | **32,263** |

**funnel再現とparquet実物が72日全てで完全一致（diff=0/72）。**

**結論: `build_horse_history.py`のコード自体は、与えられた入力データに対して
100%正確・再現可能に動作している。**「全再構築のたびに壊れる」「incremental
更新の取りこぼし」といった実装バグは存在しない。

---

## 5. 欠落原因の分類

| 原因 | 該当 | 詳細 |
|---|---|---|
| weekly側のinner join消失 | **確認済み（主因、597/598=99.8%）** | 72日中46日で発生、1日あたり典型的に7〜27行（多くは約10〜14行=典型的な1レースの出走頭数に一致）。個別に検証した具体例（2026-08-02、race_id=2026080207020409）ではkekka側10頭に対しweekly側0頭——**レース全体が週次出走表エクスポートに欠落していた**（`data/kekka/20260802.csv`・結果は存在するが`data/weekly/20260802.csv`にこのレースの出走表が含まれていなかった）。個々の馬番ミスマッチではなく、レース単位の欠落が支配的パターン |
| 馬名完全一致filterでの消失 | 確認済み（従因、1/598=0.2%） | 無視できる規模 |
| incremental更新境界 | 該当なし | 全再構築方式のため無関係 |
| horse ID解決失敗 | 該当なし | `synthetic_id`への確実なfallbackがあり行自体を落とすことはない |
| race ID不一致 | 該当なし | rid16の16桁切り出しは安定して機能 |
| groupby/重複除去 | 該当なし | 重複除去ロジックは存在しない、重複行があれば残る設計（後述の追加確認要） |
| 最新N件への切詰め | 該当なし | 契約通り無制限 |
| DNF/dropna | 該当なし | 2026分はpos=NaNのまま行として残る（止/外/消は区別なし） |
| scratch filter | 該当なし | 明示的なscratch除外ロジックはload_2026_history内に存在しない |
| **その他/不明** | **上記2つで全消失を説明済み** | 個々の馬名不一致の具体的文字差分（1件のみ）までは特定していない |

**根本原因（確定）**: `data/weekly/{date}.csv`（TARGET週次出走表エクスポート）
が、特定のレース（1日あたり0〜2レース程度）について、対応する
`data/kekka/{date}.csv`（結果）が存在するにもかかわらず欠落している。
これは`build_horse_history.py`のコードの問題ではなく、**上流の週次データ
保存・収集プロセスの欠落**であり、同スクリプトを何度再実行しても
（入力データが変わらない限り）同じ欠落が再現される。

**馬単位の欠落位置分析**（2026年に複数走を持つ馬300頭サンプル、
`_HistoryIndex.resolve()`による名前ベース解決を使用——この分析ステップのみ）:

| 位置 | 件数 |
|---|---:|
| 経歴の最古側にギャップ | 166 |
| 経歴の途中にギャップ | 38 |
| 経歴の最新側にギャップ | 73 |
| ギャップなし | 23 |

（注: この分析は「2026年に2走以上visibleな馬」のみが対象。前ラウンドで
発見した「サーガスターレ」のような「本来2走あるはずが1走しかvisibleでない」
馬は、この特定の分析手法では捕捉されない——過小報告のリスクがある点に留意）

---

## 6. Production影響監査（結果・ROIは使用していない）

| consumer | 列読み取り | 全履歴必要か | 欠落で変わる特徴 | 現行serve使用中 | fallback | 影響状態 |
|---|---|---|---|---|---|---|
| `serve_history_feats.py` (`fill_history_features`/`_HistoryIndex`/`rolling_rate`) | ped_id/name/sire/birth_year/date/place/surface/dist/pos/jockey_code/trainer_code | Yes | course_n_prev系6・hist_same系4・horse_fuku2・jockey_fuku2・trainer_fuku2(計16) | **Yes(v6本番)** | fail-open(失敗時NaNのまま、行単位の欠落自体にfallbackはなくエラーにもならず黙って過小カウント) | **CONFIRMED**(2026年分のみ、2013-2025分は対象外) |
| `predict_weekly.py`(旧アンサンブル) | — | No | — | No | 別ファイル(jockey_stats.csv等)使用 | NO_IMPACT |
| `parse_kako5.build_from_kako5()`(kako5_\*13特徴) | — | No | — | Yes | 別データソース(TARGET kako5 CSV直読み) | NO_IMPACT(本parquet不使用、Category B/Eとして別途評価済み) |
| `analysis/validate_serve_history_feats.py`・`analysis/measure_serve_coverage.py`(既存検証ツール) | 全16列 | Yes | 同上16特徴 | No(オフライン検証) | 該当なし | POSSIBLE——**検証範囲はtest=2024-2025のmaster_v2由来分のみで、2026補完分のカバレッジは検証対象外だった可能性が高い**（本監査で発見した欠損は事前に検知されていなかった） |

---

## 7. Shadow Corrected Artifact — **実施しない（実施不可能と判定）**

item2でartifact completeness bugと判定したが、§4-5で確認した通り
**根本原因は`build_horse_history.py`のコードではなく上流入力データ
（`data/weekly/{date}.csv`）自体の欠落**である。したがって:

- 同スクリプトを「完全再構築」しても、入力データが同じである限り
  **全く同一の欠落が再現される**（funnel再現の完全一致がこれを実証済み）。
- 欠落レースを補うには、weeklyに相当する情報（騎手・調教師・父・芝ダ・距離）
  を別の情報源から取得する必要があるが、`data/kekka/{date}.csv`自体には
  これらの列が存在しない（確認済み: 15列のみ、芝ダ/距離/騎手/調教師/父は
  含まれない）。
- **本読み取り専用監査の範囲では、欠落を補う代替情報源の探索・実装は
  行っていない**。したがって「別パスへの完全再構築」は現時点で実施しても
  原本と同じ欠損を持つ複製を作るだけであり、意味のある改善にならない。

---

## 8. Legacy-v6候補の再開条件（5条件）

| 条件 | 状態 |
|---|---|
| authoritative universeとの完全性が説明済み | ✅ 説明済み(本監査) |
| Legacy-v6 contract（直近5完走）が再現可能 | ❌ **未達**——データソース自体が約1.8%の2026年出走行を欠くため、これに依拠する候補実装は本質的に不正確 |
| Category Eの差分原因が説明可能 | △ 部分的(65.7%はこのデータ欠損で説明可能と判明、残り34.3%のDNF/scratch起因分は未分離のまま) |
| kako5以外107特徴が完全一致 | ✅ 前ラウンドで確認済み(PASS) |
| corrected artifactが原本と隔離されている | N/A(§7の通りcorrected artifactを作成していない) |

**5条件中2つが未達/N/A。ユーザー指示「次の全条件を満たすまで再開しない」に
より、Legacy-v6候補実装の作業は再開しない。**

---

## 9. 結論

`data/_horse_history.parquet`は**「production bug」ではなく「（データ欠損を
抱えた）artifact completeness issue」**である——`build_horse_history.py`
自体は完全に契約通りに（入力データに対して忠実に）動作しているが、上流の
`data/weekly/{date}.csv`収集プロセスに約1.8%（2026年、72日サンプル中46日で
検出）のレース単位の欠落がある。この欠損は現行v6本番のcourse/jockey/
hist_same/fuku系16特徴（2026年分のみ）に影響する。kako5_\*13特徴は本parquet
を使わないため無関係（Category B/Eとして別途評価済み）。

前ラウンドのshadow replayで測定したraw score/p_win/順位/◎変更数は、
Category Bの真の影響としては引き続き使用しない（invalid runとして
`docs/research/KAKO5_CONTRACT_SHADOW_REPLAY_20260922.md`に記録済み）。
**Legacy-v6候補実装の作業はここでは再開しない**——欠損データソースの補完
（代替情報源の探索）が先決事項であり、本監査の範囲外として別タスクへ
切り出し済み（`task_0d3b5349`、build_horse_history.py側の詳細root cause
特定を依頼、本監査の§4-5の発見を踏まえて更新可能）。

course_n_prev/jockey_n_prev等16特徴への実際の2026年運用影響（v6の現行
serveでの過小カウント）は、Category Aの一部として既に前ラウンドで
確認済みの事実の延長线上にあり、新たな懸念ではない——本監査は「なぜ
2026年分でも一部欠損するのか」を特定したに留まる。

---

## 10. 停止事項

本監査でも: production master/model・scheduler変更・Optuna・EXP再実行・
EXP14 Stage1・ROI評価・corrected-vNext実装は一切行っていない。

関連: `docs/research/KAKO5_CONTRACT_SHADOW_REPLAY_20260922.md`,
`analysis/mcond/p0_dnf_history_parity_audit/horse_history_contract_audit.py`,
`analysis/mcond/p0_dnf_history_parity_audit/SEMANTIC_VS_PARITY_CLASSIFICATION.md`
