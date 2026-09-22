# P1 forward-only collector — Phase A/B 完了・Phase C 準備（2026-09-22）

**状態**: collector は完成し全 hard test PASS。**現行 v6 の特徴計算・予測・印・
買い目には一切接続していない。** scheduler 登録もしていない。
モデル変更・再学習・Optuna・EXP14 Stage 1・ROI 評価・
`_horse_history.parquet` の置換・weekly bundle への障害追加もしていない。

成果物:
- `analysis/jump_history_only/phase_a_bunseki_audit.py`
- `analysis/jump_history_only/jump_history_collector.py`
- `analysis/jump_history_only/test_jump_history_invariants.py`（24 テスト）
- `analysis/jump_history_only/jump_history_collect.ps1`（**未登録**）
- ストア: `data/history_only/jump/`（production は読まない）

---

## ⚠ 0. 最重要の新発見 — `data/weekly` の障害除外は**一貫していない**

前回まで「`data/weekly` は障害競走を含まない、意図的な scope filter」と
結論していたが、**これは誤りだった**。bunseki の 8 日を weekly と突き合わせた
実測:

| 日付 | 障害 race_id | weekly 内の行数 |
|---|---|---:|
| 20260905 | 2026090506040101 | **0** |
| 20260906 | 2026090606040201 | **0** |
| 20260912 | 2026091209040301 | **0** |
| **20260913** | 2026091306040401 | **10** |
| **20260919** | 2026091909040504 | **12** |
| **20260920** | 2026092009040601 | **12** |
| 20260921 | 2026092106040701 | **0** |
| **20260922** | 2026092206040701 | **14** |

**8 日中 4 日で障害レースが weekly に入っていた。**

### 0.1 結果: 障害レースが本番 bundle に入り、v6 に採点されていた

`reports/cowork_input/*_bundle.json` を検査したところ、
**3 レースが実際に bundle へ混入していた**:

- `2026091306040401`（20260913 中山 R01 障害）
- `2026091909040504`（20260919 阪神 R04 障害）
- `2026092009040601`（20260920 中山 R01 障害）

これらは v6 に採点され、印が付き、Cowork へ提示されていた。

### 0.2 実害: 買い目は発注されていない（ただし偶然）

3 レースとも `reports/cowork_output/*_bets.json` で **`bets: []`**
（見送り）だった。**金銭的な実害は発生していない。**

しかし**これは Cowork の判断結果であって、コードレベルのゲートではない**。
`compute_bets.py` / `validate_cowork_bets.py` のいずれにも
「障害レースを除外する」判定は存在しない。weekly に障害が入った日は、
**買い目が付く可能性が構造的に開いている**。

### 0.3 前回結論の訂正

1. 「`data/weekly` は障害を含まない意図的な scope filter」→
   **誤り。日によって入ったり入らなかったりする不安定な挙動**。
2. 「欠落は 2026-09-06 を最後に停止している（0912 以降ゼロ）」→
   **誤り**。0912 以降は `data/kekka` が未配置のため
   欠落解析の対象外だっただけで、「欠落が止まった」根拠は無かった。
3. §0.1 の 3 件は本 collector が作ったものではなく、
   **collector 以前から存在する本番側の状態**である。
   hard test `test_t2_no_new_jump_race_leaks_into_bundle` で
   この 3 件を baseline として固定し、**新規混入を検出**する。

> これは「対象外レースを誤って購入する危険」が理論上のリスクではなく
> **既に起きかけていた**ことを意味する。P1 の hard gate の必要性は
> 設計上の予防ではなく、実測に基づく要件である。

---

## 1. Phase A — bunseki 生成経路の監査

| 項目 | 結果 |
|---|---|
| 生成元 | **TARGET GUI の「出走馬分析」エクスポート（手動）** |
| intake | `data/_inbox/` へ投入 → `place_weekly.py` が `data/bunseki/{date}.csv` へ振り分け |
| 判定方法 | ヘッダ行あり（先頭列 `No.`）かつ `馬齢斤量差`/`前場所` 列の有無（`place_weekly.py:80-96, 237-239`） |
| 自動か手動か | **手動**（TARGET からの出力はユーザー操作） |
| scheduler 登録 | **無し**。`Get-ScheduledTask` に該当なし（`PyCaLiAI_Baba` / `T10` / `T20_Site` / `EXP05FS_*` のみ） |
| 呼び出し | `weekly_nicegui.ps1` Phase A/C の Step 0 で `place_weekly.py` が走る（`-SkipIntake` で無効化可） |
| 生成時刻 | TARGET からエクスポートした時刻。ファイル mtime に記録（監査 JSON に保存済み） |
| 全レースを含むか | **含む**。20260905/0906 で race 回収率 **100%**、horse-row 回収率 **100%** |
| 障害を含む保証 | **保証されていない**。TARGET 側の項目選択・出力条件に依存する。ただし**実測では 8 日すべてに障害 1 レースが含まれていた**（weekly と違い bunseki は安定して含む） |
| 上書きか日付別か | 日付別（`{date}.csv`）。ただし同一日付を再投入すると `place_weekly.py` は**上書き**する |
| 失敗ログ | **無し**。`place_weekly.py` は標準出力へ出すだけで専用ログ・通知は無い |

---

## 2. Phase A — 保存済み 8 日の完全性

| date | bunseki 行 | race | 障害R | race 回収 | row 回収 | ped_id | jockey | trainer | dup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20260905 | 455 | 36 | 1 | **1.0000** | **1.0000** | 1.000 | 1.000 | 1.000 | 0 |
| 20260906 | 491 | 36 | 1 | **1.0000** | **1.0000** | 1.000 | 1.000 | 1.000 | 0 |
| 20260912 | 316 | 24 | 1 | —* | —* | 1.000 | 1.000 | 1.000 | 0 |
| 20260913 | 314 | 24 | 1 | —* | —* | 1.000 | 1.000 | 1.000 | 0 |
| 20260919 | 287 | 24 | 1 | —* | —* | 1.000 | 1.000 | 1.000 | 0 |
| 20260920 | 334 | 24 | 1 | —* | —* | 1.000 | 1.000 | 1.000 | 0 |
| 20260921 | 320 | 24 | 1 | —* | —* | 1.000 | 1.000 | 1.000 | 0 |
| 20260922 | 161 | 12 | 1 | —* | —* | 1.000 | 1.000 | 1.000 | 0 |

\* `data/kekka` が 20260906 までしか無いため、0912 以降は結果側との突合ができない
（= settled layer へ入らない。これは仕様どおりの挙動）。

- **血統登録番号・騎手コード・調教師コードは全 8 日で coverage 100%**
- **duplicate 0**（`race_id+馬番`・`race_id+血統登録番号` とも）
- **kekka 結合率 100%**（20260905/0906、`race_id+馬番`、馬名不使用）
- **全 8 日にちょうど 1 つの障害レースが存在**

### 2.1 finish code の判別可能性

| 状態 | 判別 |
|---|---|
| completed（完走） | ⭕ `確定着順 > 0` |
| non-completed | ⭕ `確定着順 = 0/空` で検出できる |
| **DNF(止) と 除外 の分離** | ❌ **不可**。kekka の `確定着順` が両者を同じ値に潰す（既確認） |
| scratched（取消）**候補** | ⭕ card にあり結果に無い行として検出（`started=false`） |
| 当日変更情報ルート | ❌ `site/data/changes_{date}.json` は 8 日すべて `races={}` で空。取消の独立ソースにならない |

→ settled layer は `dnf_jogai_separable: false` を明示的に持ち、
**分離できないことをデータ自身に記録**する。

---

## 3. Phase A — 必須 fixture（20260905 / 20260906）

両日とも **PASS**:

| チェック | 0905 | 0906 |
|---|---|---|
| 障害レースが存在 | ✅ | ✅ |
| bunseki に全出走馬が存在（row 回収 100%） | ✅ | ✅ |
| kekka と `race_id + 馬番` で結合可能（100%） | ✅ | ✅ |
| 馬名を使わない | ✅ | ✅ |
| 血統登録番号 coverage 100% | ✅ | ✅ |
| 騎手・調教師コード coverage 100% | ✅ | ✅ |
| duplicate 0 | ✅ | ✅ |
| 完走 / 取消候補 を区別できる | ✅ | ✅ |
| DNF と 除外 を分離できる | ❌（データソース制約、明示記録） | ❌ |

---

## 4. Phase B — append-only 二層 artifact

### 4.1 レイアウト

```
data/history_only/jump/
  raw_card/{date}.jsonl     … 判断前情報（bunseki 由来）
  settled/{date}.jsonl      … 結果確定後（kekka 由来）
  manifest.json             … coverage_start / 復元不能 gap / 収集済み日付
```

**raw card を結果で上書きしない**。別ファイル・別 schema で append する。

### 4.2 収集実績

| layer | 日数 | 行数 |
|---|---:|---:|
| raw_card | 8 | 8+7+8+10+12+12+14+14 = **85** |
| settled | 2（kekka がある日のみ） | **15** |

### 4.3 レコード例（抜粋）

raw card:
```json
{"schema_version":"jump-history-only/raw_card/1",
 "history_only":true,"prediction_eligible":false,
 "bet_eligible":false,"task_registration_eligible":false,
 "race_id":"2026090506040101","ped_id":"2022104929","umaban":1,
 "race_date":20260905,"scheduled_post":"10:05","venue":"中山",
 "race_number":1,"track_code":52,"jump_flag":true,"distance":3200,
 "jockey_code":"01196","trainer_code":"01160",
 "pedigree":{"種牡馬":"アドマイヤマーズ","母名":"トレジャリング", ...},
 "captured_at":"2026-09-22T22:21:58+09:00",
 "source_file":"data/bunseki/20260905.csv","source_sha256":"604975c..."}
```

settled:
```json
{"schema_version":"jump-history-only/settled/1",
 "history_only":true,"prediction_eligible":false,
 "bet_eligible":false,"task_registration_eligible":false,
 "race_id":"2026090506040101","ped_id":"2022104929","umaban":1,
 "race_date":20260905,"finish_code_raw":"3",
 "started":true,"completed":true,"dnf":false,"scratched":false,
 "dnf_jogai_separable":false,
 "result_available_at":"2026-09-07T21:37:32+09:00",
 "settled_at":"2026-09-22T22:21:58+09:00",
 "result_source_file":"data/kekka/20260905.csv",
 "result_source_sha256":"fe27b39..."}
```

### 4.4 hard test（24 件、**全 PASS**）

| ID | 内容 | 結果 |
|---|---|---|
| T1 | 全レコードが 4 つの eligibility フラグを持つ | PASS |
| T2 | **新規**の bundle 混入が無い（既知 3 件を baseline 固定） | PASS |
| T2b | 既知混入 3 件に買い目が付いていない | PASS |
| T2c | collector 出力が予測/購入/タスク対象にならない | PASS |
| T3 | 障害 race_id の scheduler タスクが存在しない | PASS |
| T4 | `race_id+ped_id` / `race_id+馬番` が一意・決定的 | PASS |
| T4b | collector がコード上で馬名を参照しない | PASS |
| T5 | 結果が無い日は settled へ入らない | PASS |
| T5b | kekka 未配置の日に settled ファイルが存在しない | PASS |
| T6 | 未来日が settled に入らない | PASS |
| T7 | raw card が結果項目を持たない | PASS |
| T7b | 二層が別ファイル | PASS |
| T8 | 同一 source の再処理が idempotent（added=0） | PASS |
| T9 | 内容変化で `CollisionError`・**1 行も書かれない** | PASS |
| T9b | `captured_at` だけの変化は衝突扱いしない | PASS |
| T10 | 収集対象が障害（track_code 51-59）のみ | PASS |
| T11 | schema / provenance が揃っている | PASS |
| T11b | `source_sha256` が実ファイルと一致 | PASS |
| T12 | collector が原本ディレクトリへ書かない | PASS |
| T12b | ストアが production 入力ディレクトリの外 | PASS |
| T12c | **production スクリプトがストアを読んでいない** | PASS |
| T13 | manifest に coverage_start と復元不能 gap がある | PASS |
| T14 | fixture 2 日が完全 | PASS |

その他の invariant:
- **atomic write**: 同一ディレクトリへ一時ファイル → `os.fsync` → `os.replace`
- **原本不変**: `git status data/bunseki data/kekka` に collector 由来の変更なし

### 4.5 欠落期間の扱い（manifest に固定）

```json
"jump_history_coverage_start": "20260905",
"known_unrecoverable_gap": {
  "period": "2026-03-07 〜 収集開始日の前日",
  "jump_races": 47, "rows": 582,
  "reason": "共有 JV-Link ストアでの過去 backfill は隔離不可のため永久中止（判断分岐 B）",
  "policy": "推測補完しない / 馬名 join しない / 0 埋めしない"
}
```

**「履歴なし」と「履歴 coverage 不足」を区別する**ため、
将来 rolling 特徴へ接続する際は entity ごとに `history_complete` を持たせる。
判定規則（**未実装、設計のみ**）:

| window | `history_complete=true` の条件 |
|---|---|
| `horse_fuku10` / `horse_fuku30` | その馬の直近 10 / 30 走がすべて `coverage_start` 以降 |
| `jockey_fuku30` / `jockey_fuku90` | その騎手の直近 30 / 90 **騎乗**がすべて `coverage_start` 以降 |
| `trainer_fuku30` / `trainer_fuku90` | その調教師の直近 30 / 90 **出走**がすべて `coverage_start` 以降 |
| `jockey_n_prev` 系（窓なし累積） | 当該 馬×騎手 ペアの**全キャリア**が `coverage_start` 以降（＝実質デビュー馬のみ true） |

それまでは `history_complete=false` とし、**coverage 不足を明示したまま**扱う。

---

## 5. Phase C 準備（**scheduler 未登録・本番未接続**）

### 5.1 具体的な Action

| 項目 | 内容 |
|---|---|
| **Action** | `powershell -NoProfile -ExecutionPolicy Bypass -File E:\PyCaLiAI\analysis\jump_history_only\jump_history_collect.ps1` |
| **作業ディレクトリ** | `E:\PyCaLiAI` |
| **提案タスク名** | `PyCaLiAI_JumpHistory`（既存の `PyCaLiAI_*` 命名に合わせる） |
| **実行時刻（案）** | **開催日 22:30**（土・日・祝開催日）。理由: ① bunseki は当日中にユーザーが TARGET からエクスポートし `place_weekly.py` が配置する ② kekka は Phase C（日曜夜）で入る ③ T-10/T-20/Vote（〜16:30 頃）と時間帯が重ならない |
| **代替案** | 毎日 22:30 実行にして、bunseki が無い日は即 `exit 0`（実装済みの skip 経路）。開催日判定を持たなくて済むぶん堅い |
| **出力先** | `data/history_only/jump/{raw_card,settled}/{date}.jsonl` + `manifest.json` |
| **ログ** | `logs/jump_history_{date}.log`（タイムスタンプ付き追記） |
| **多重起動** | 同一日を再実行しても idempotent（added=0）。ファイルロックは未実装（単一タスクのため） |

### 5.2 失敗時動作

| 事象 | 動作 |
|---|---|
| bunseki が無い | ログに記録して **`exit 0`**（正常終了。開催日でない等） |
| collector が例外 | ログに記録して **非 0 終了**。append-only のため**部分書込は発生しない** |
| `CollisionError`（同一キーで内容変化） | **即停止・非 0 終了**。silent overwrite しない。ログに「原本を確認し、意図的な差し替えなら手動判断」と出す |
| hard test FAIL | **非 0 終了**し、ログに「収集物を信用しない。特徴接続は絶対にしない」と記録 |
| 通知 | Discord 等の外部通知は**付けない**（既存 T-20 Site と同方針）。異常はタスクスケジューラの「前回の実行結果」と `logs/` で確認 |

収集のたびに **hard test 24 件を自動実行**し、FAIL なら非 0 で終わる設計にしてある。

### 5.3 rollback 手順

collector は**追加しかしない**ため rollback は単純:

1. タスクを無効化: `Disable-ScheduledTask -TaskName PyCaLiAI_JumpHistory`
2. ストアごと削除: `Remove-Item -Recurse data\history_only`
3. それだけで**元の状態に完全復帰する**
   - production の入力（`data/weekly` / `data/bunseki` / `data/kekka` /
     `_horse_history.parquet`）を一切変更していない
   - production コードからストアへの参照が無いことは T12c が保証
   - モデル・calibrator・bundle・buy-list に一切触れていない

### 5.4 実行前レビュー用 diff

新規追加のみ（既存ファイルの変更は 0 件）:

```
A  analysis/jump_history_only/phase_a_bunseki_audit.py
A  analysis/jump_history_only/jump_history_collector.py
A  analysis/jump_history_only/test_jump_history_invariants.py
A  analysis/jump_history_only/jump_history_collect.ps1      ← 未登録
A  analysis/jump_history_only/out/phase_a_bunseki_audit.json
A  data/history_only/jump/raw_card/*.jsonl   (8 files, 85 行)
A  data/history_only/jump/settled/*.jsonl    (2 files, 15 行)
A  data/history_only/jump/manifest.json
```

**production ファイルの変更は 1 件も無い。**

---

## 6. 観測 Gate（自動収集開始後、2 開催日で確認する項目）

| 指標 | 目標 | 現在（保存済み 8 日） |
|---|---|---|
| bunseki 全 race coverage | 100% | **100%**（0905/0906 で実測） |
| 障害 race coverage | 100% | **100%**（8/8 日で 1 レースずつ検出） |
| horse-row coverage | 100% | **100%**（0905/0906） |
| 必須 ID coverage | 100% | **100%**（ped/jockey/trainer、8 日すべて） |
| kekka settlement coverage | 100% または説明可能な取消等のみ | **100%**（0905/0906）。0912 以降は kekka 未配置で未評価 |
| duplicate / collision | 0 | **0** |
| prediction/betting task への混入 | 0 | collector 由来 **0**。**ただし collector 以前から本番 bundle に 3 件の混入がある（§0）** |
| append-only 違反 | 0 | **0** |

---

## 7. まだ行っていないこと（禁止事項の確認）

- `_horse_history.parquet` の置換 — **していない**
- current-v6 入力への追加 — **していない**
- horse/jockey/trainer rolling の変更 — **していない**
- weekly bundle への障害追加 — **していない**
- モデル変更・再学習・Optuna — **していない**
- EXP14 Stage 1 — **していない**
- ROI 評価 — **していない**
- scheduler 登録 — **していない**

特徴 shadow は、2 開催日の collection Gate 通過後に、
coverage 不足を明示したまま作る。

---

## 8. ユーザー判断が要る点

1. **§0 の扱い** — 障害レースが本番 bundle に入る日がある。
   買い目は付いていないが、コードレベルのゲートは無い。
   `compute_bets.py` / `validate_cowork_bets.py` に
   「`track_code` 51-59 を除外」する hard gate を入れるかどうか
   （**これは production 変更なので、指示があるまで着手しない**）。
2. **`PyCaLiAI_JumpHistory` タスクの登録可否**と実行時刻
   （案: 毎日 22:30、bunseki が無い日は即終了）。
3. bunseki のエクスポート運用 — 現在は手動。
   forward collection を安定させるには毎開催日の出力が要る。
